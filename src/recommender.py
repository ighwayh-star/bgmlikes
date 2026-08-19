"""推荐引擎深模块：小接口 recommend()。

内部（对调用方隐藏）：
- User-CF（阶段 1.5 验证过的算法）从 SQLite 语料构建
- 流行度对数回退（冷启动 / 混合），系数 blend_lambda 可调
- 过滤：候选集排除已收藏条目
- 排序取 top-K，附加条目名

领域术语见 docs/CONTEXT.md。
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.bangumi_api import CollectionEntry


@dataclass(frozen=True)
class Recommendation:
    subject_id: int
    name: str
    score: float
    popularity_rank: int = 0  # 全局热度排名（1 = 最热门）


@dataclass(frozen=True)
class SimilarItem:
    """浮窗"相似动画"条目（卡片详情用，算法见 Recommender.similar_items）。"""

    subject_id: int
    name: str
    rating: float = 0.0  # BGM 平均分
    popularity_rank: int = 0
    score: float = 0.0  # 混合相似度（标签余弦 × 共同观看余弦）


@dataclass(frozen=True)
class RecommendResult:
    """单区推荐结果（2026-08-11 删冷门发现后）。

    normal: 推荐列表——rank ≤ rank_cap 的日本动画里，渗透归一化 CF 高分 top-k。
    冷门发现区（rank > 阈值 + 当季新番剔除）已随功能一并删除。
    """
    normal: list[Recommendation]


def score_items(
    Bn: np.ndarray,
    R: np.ndarray,
    log_pop: np.ndarray,
    q: np.ndarray,
    blend_lambda: float,
    min_profile: int,
    *,
    knn: int | None = None,
) -> np.ndarray:
    """评分核心：User-CF + 流行度对数混合，冷启动回退流行度。

    从 Recommender 抽出，评估脚本复用同一实现，防漂移。
    min_profile 按 profile 中实际作品数（非零项）判定。
    knn: 只保留最相似的 top-K 用户参与打分（候选池收敛，阶段 3-3 生效）。
    """
    if (q != 0).sum() >= min_profile:
        qn = q / np.sqrt((q * q).sum())
        sim = np.asarray(Bn @ qn).ravel()  # 稀疏 Bn 行式 matvec（等价 qn@Bn.T，省去每次全矩阵转置）
        if knn is not None:
            idx = np.argpartition(-sim, knn)[:knn]
            sim = sim[idx]
            R = R[idx]  # 稀疏行切片同样成立
        base = np.asarray(sim @ R).ravel()  # 兼容稀疏 R
        return base + blend_lambda * log_pop
    return log_pop.copy()


def quota_rank(
    scores: np.ndarray,
    pop_rank: np.ndarray,
    cold_quota: int,
    cold_rank_threshold: int,
    cand: list[int],
    k: int,
) -> list[int]:
    """冷门配额：保证返回的 k 条里至少 cold_quota 条来自冷门池（热度排名 > 阈值）。

    背景（2026-08-06 定稿）：用户要求"推荐别全是热门"。原型验证过两条路：
    - 排名融合（把冷门度掺进评分）会打乱相关度排序，Recall@10 直接归零；
    - 配额不碰排序——冷门池里取 CF 分最高的 N 条 + 非冷门池 CF 分最高的补足，
      再按 CF 分合并。冷门名额里填的是"最相关的那批冷门"，Recall 零损失
      （20k 复测：20中8 冷门 Recall@10 0.0254 = 纯 CF，列表 热51%/中9%/冷40%）。

    返回按 CF 分降序的物品索引列表，长度 ≤ k。
    """
    if cold_quota <= 0 or cold_rank_threshold <= 0:
        return sorted(cand, key=lambda i: -scores[i])[:k]
    cold_pool = [i for i in cand if pop_rank[i] > cold_rank_threshold]
    if not cold_pool:  # 候选里没有冷门（如全站小众场景），退化为纯 CF
        return sorted(cand, key=lambda i: -scores[i])[:k]
    n_cold = min(cold_quota, k, len(cold_pool))
    cold_top = sorted(cold_pool, key=lambda i: -scores[i])[:n_cold]
    # 剩余名额从全部候选取（排除已选冷门）——配额是"至少"而非"恰好"：
    # 冷门偏好用户会自然获得 >cold_quota 的冷门，主流用户行为不变（非冷门评分占优）。
    cold_set = set(cold_top)
    rest = sorted(
        (i for i in cand if i not in cold_set),
        key=lambda i: -scores[i],
    )[:k - n_cold]
    return sorted(cold_top + rest, key=lambda i: -scores[i])[:k]


def build_encoding(
    n_users: int,
    n_items: int,
    ui: np.ndarray,
    ii: np.ndarray,
    values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """编码交互为 IDF 加权矩阵（阶段 3：反频权重生效；评分中心化经 values 由调用方传入）。

    ui, ii: 训练交互的 (用户索引, 物品索引) 数组/序列；调用方负责过滤（rate>0、train-only 防泄漏）。
    values: 每个 (u,i) 对的数值乘子（如 1-10 评分）；None = 1（纯 IDF 二值）。
        值 A[u, i] = idf[i] * values[pair]（默认 idf[i]）。
    返回 (A, Bn, idf)；Bn 为行 L2 归一化，是余弦相似度的来源。
    20k 语料：稀疏 csr 存储（~4.6M 非零 ≈ 75MB），替代稠密 U×I（~5GB×2）会爆内存。
    """
    ui = np.asarray(ui, dtype=np.int64).ravel()
    ii = np.asarray(ii, dtype=np.int64).ravel()
    if len(ui) == 0:
        empty = sp.csr_matrix((n_users, n_items))
        idf = np.log((n_users + 1) / (np.ones(n_items) + 1)) + 1.0
        return empty, empty, idf
    df = np.bincount(ii, minlength=n_items).astype(np.float64)
    idf = np.log((n_users + 1) / (df + 1)) + 1.0
    if values is None:
        val = idf[ii]
    else:
        val = idf[ii] * np.asarray(values, dtype=np.float64).ravel()
    A = sp.csr_matrix((val, (ui, ii)), shape=(n_users, n_items))
    row_norm = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    inv = np.zeros(n_users)
    nz = row_norm > 0
    inv[nz] = 1.0 / row_norm[nz]
    Bn = sp.diags(inv) @ A  # 行 L2 归一化（余弦相似度的来源）
    return A, Bn, idf


def encode_profile(
    items: list[int],
    iidx: dict[int, int],
    idf: np.ndarray,
    weights: list[float] | None = None,
) -> np.ndarray:
    """目标用户口味 → 与矩阵同构的 IDF 加权向量 q（余弦要同尺度对比）。

    weights: 每个 item 的额外乘子（如 1-10 评分）；None = 1。
    """
    q = np.zeros(len(iidx))
    if weights is None:
        for sid in items:
            i = iidx.get(sid)
            if i is not None:
                q[i] = idf[i]
    else:
        for sid, w in zip(items, weights):
            i = iidx.get(sid)
            if i is not None:
                q[i] = idf[i] * w
    return q


def build_franchise(
    item_ids,
    conn,
) -> dict[int, int]:
    """从 subject_relations 表构建 franchise 分组（同一部动画的不同季/续作/剧场版/番外）。

    union-find 连通分量，根 = 分量内最小 subject_id（稳定可复现）。孤立条目不返回，
    调用方用 sid 自身兜底（去重时互不合并）。
    relation_type 限定 {2 续作, 3 前传, 6 系列}——20k 语料验证 type 4（不同世界观/跨界）
    与 99（其他）含跨界边，会错误合并 鲁邦三世×柯南 这类 crossover（见 docs/adr/0003）。
    从 Recommender._load 与 evaluate 共用，防漂移。
    """
    item_set = set(item_ids)
    parent: dict[int, int] = {}

    def _find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    try:
        rel_rows = conn.execute(
            "SELECT subject_id, related_subject_id FROM subject_relations"
            " WHERE relation_type IN (2,3,6)"
        )
    except sqlite3.OperationalError:
        # subject_relations 未导入（未跑 scripts/import_relations）：无 franchise 信息，
        # 全部条目视作孤立（不分组、不去重）。服务器仍可启动，只是丢失该特性。
        return {}
    for a, b in rel_rows:
        if a in item_set and b in item_set:
            _union(a, b)

    comp: dict[int, list[int]] = {}
    for sid in item_set:
        comp.setdefault(_find(sid), []).append(sid)
    root_map: dict[int, int] = {}
    for members in comp.values():
        root = min(members)
        for m in members:
            root_map[m] = root
    return root_map


# ---- 老动画标签 boost（2026-08-18 路由 A 打分层）----
TAG_DF_MIN = 15      # 标签词表下界：滤掉超稀有/噪音标签
TAG_DF_MAX = 5000    # 标签词表上界：滤掉超普遍标签（TV/日本/剧场版…）
# 停用词：地区/格式/来源/分级 类标签（非题材，对相似度是噪音）
STOP_TAGS = {
    "中国", "欧美", "美国", "法国", "韩国", "英国", "捷克", "苏联", "台湾",
    "香港", "俄罗斯", "WEB", "OVA", "短片", "MV", "PV", "CM", "动态漫画",
    "漫画改", "原创", "小说改", "游戏改", "影视改", "R18",
}


def build_tags(
    items: list[int],
    subject_meta: dict[int, dict],
    old_year: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """老番标签 boost 的基建：P(item×tag, idf 加权)、Pn(行归一化)、old_mask(date<old_year)。

    目标用户口味投影到题材空间 q_tag = P.T·q，Pn[i]·q_tag 即 tag_cos——晚入坑用户
    对老年代没评分信号，但靠题材亲和仍能浮现老动画（非保底硬塞，见 docs/adr/0008）。
    β 通过 Recommender(old_tag_beta=) 注入，.env OLD_TAG_BETA 可调，0=关。
    """
    tag_doc: Counter[str] = Counter()
    item_tags: dict[int, set[str]] = {}
    for i, sid in enumerate(items):
        tags = set(subject_meta[sid].get("meta_tags") or [])
        item_tags[i] = tags
        for t in tags:
            tag_doc[t] += 1
    vocab = [t for t in tag_doc if TAG_DF_MIN <= tag_doc[t] <= TAG_DF_MAX and t not in STOP_TAGS]
    vocab.sort()
    tidx = {t: j for j, t in enumerate(vocab)}
    n_items, n_tags = len(items), len(vocab)
    tag_idf = {t: math.log((n_items + 1) / (tag_doc[t] + 1)) + 1 for t in vocab}

    rows, cols, vals = [], [], []
    for i, tags in item_tags.items():
        for t in tags:
            j = tidx.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(tag_idf[t])
    P = sp.csr_matrix((vals, (rows, cols)), shape=(n_items, n_tags))

    # Pn：item 标签向量行归一化（与 qn_tag 点积即 tag_cos）
    rn = np.asarray(P.multiply(P).sum(axis=1)).ravel()
    inv = np.zeros(n_items)
    nz = rn > 0
    inv[nz] = 1.0 / np.sqrt(rn[nz])
    Pn = (sp.diags(inv) @ P).tocsr()

    old_mask = np.zeros(n_items, dtype=bool)
    for i, sid in enumerate(items):
        d = subject_meta[sid].get("date") or ""
        if d[:4].isdigit() and int(d[:4]) < old_year:
            old_mask[i] = True
    return P, Pn, old_mask


class Recommender:
    """从 SQLite 语料加载进内存；构造注入，可测试。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        blend_lambda: float = 0.0,  # 流行度对数混合系数（0 = 纯 CF，不掺流行度）
        min_profile: int = 5,
        knn: int = 200,
        rank_cap: int = 3000,       # 推荐池热度上限：只推 rank ≤ 该值的日本动画（2026-08-11 删冷门后）
        matrix_min_rate: int = 0,   # 训练矩阵只收 rate>该值的对（0 = 全部 rate>0，ADR 0005）
        taste_min_rate: int = 0,    # 2026-08-12 实验：取消低分过滤（rate>0 全部算口味信号）；rank_cap 低热度过滤保留
        adaptive_gamma: bool = False,  # 2026-08-12 实验：关闭自适应，全部用户固定 γ
        gamma: float = 1.0,            # 固定去热强度（自适应关闭时对所有用户生效，λ=1 的流行度加成不再被对冲）
        df_min_rated: int = 0,         # 2026-08-12 去热分母只计评分条数≥该值的重度用户（剔除轻度用户回暖热门）；0=全语料
        rate_center: float = 5.0,      # 2026-08-12 相似度中心化（仅相似度）：Bn 用 idf*(rate-rate_center)，打分 A 保持 idf*rate。5 分中界——1-4 负偏好、5 中性、6-10 正偏好；0=全原始分
        idf_in_score: bool = True,     # 2026-08-13 打分矩阵 A 是否带 idf 乘子（分子 idf 实验）：False=去掉（A 保持原分），相似度 Bn/q 的 idf 保留。实测 noA 比全去更好
        tag_beta_all: float = 0.0,     # 2026-08-18 全池标签 boost（题材浮现，不限年代）：>0 时所有池内候选加 β×scale×tag_cos（用户真实需求——题材贴但邻居没覆盖的作品浮现）；老候选再叠加 old_tag_beta
        old_tag_beta: float = 0.0,     # 2026-08-18 老动画标签 boost 额外倍率（在 tag_beta_all 之上）：0=关；>0 时老候选(date<old_tag_year)再加 β×scale×tag_cos（深盲区老题材救援）。仅开它 = 原仅老版本，向后兼容
        old_tag_year: int = 2010,      # "老动画"年份上界：date < 该值 视为老（.env OLD_TAG_YEAR 可调）
        era_gap_beta: float = 0.0,     # 2026-08-18 年份差 boost（正确算法版，替代年份门控）：>0 时对池内候选加 λ×scale×cos_i×f(|year_i − anchor|)，anchor = 相似用户（sim 加权）的平均观看年份，逐用户自适应。无全局年份常量、天然对称：新口味邻居锚点新→救老题材；老口味邻居锚点老→救新题材（老口味用户新番概率不再被全局门压低）。tag_cos 做相关度守门，非硬塞
        era_gap_year_span: float = 50.0,  # 年份差权重 f 的饱和跨度：Δ=year_span 时 f=1（.env ERA_GAP_YEAR_SPAN 可调）
        era_gap_shape: str = "log",    # 年份差权重形状：'log'=log1p(Δ)/log1p(span) 对数饱和（推荐，对称性好）；'lin'=Δ/span 线性 clip
        similar_alpha: float = 0.5,    # 相似动画混合系数：α×标签余弦 + (1−α)×共同观看余弦（浮窗"相似"用，.env SIMILAR_ALPHA 可调）
        site_blocked: set[int] | frozenset[int] = frozenset(),  # 站点级永久屏蔽：推荐/相似候选一律排除（2026-08-19）
        hot_rank_threshold: int = 500,  # "高热度"＝全局热度排名 < 该值
        hot_share_target: float = 0.5,  # 推荐池前 min(k,40) 里高热度占比目标（自适应 γ 校准，仅 adaptive 时用）
        gamma_max: float = 0.8,         # 自适应 γ 上限
    ):
        self._blend_lambda = blend_lambda
        self._min_profile = min_profile
        self._knn = knn
        self._rank_cap = rank_cap
        self._matrix_min_rate = matrix_min_rate
        self._taste_min_rate = taste_min_rate
        self._adaptive_gamma = adaptive_gamma
        self._gamma = gamma
        self._df_min_rated = df_min_rated
        self._rate_center = rate_center
        self._idf_in_score = idf_in_score
        self._tag_beta_all = tag_beta_all
        self._old_tag_beta = old_tag_beta
        self._old_tag_year = old_tag_year
        self._era_gap_beta = era_gap_beta
        self._era_gap_year_span = era_gap_year_span
        self._era_gap_shape = era_gap_shape
        self._similar_alpha = similar_alpha
        self._site_blocked = set(site_blocked)
        self._hot_rank_threshold = hot_rank_threshold
        self._hot_share_target = hot_share_target
        self._gamma_max = gamma_max
        self._last_gamma: float = 0.0  # 最近一次 recommend 用的 γ（观测/调参用）
        self._load(db_path)

    # ---- 加载 ---------------------------------------------------

    def _load(self, db_path: str | Path) -> None:
        conn = sqlite3.connect(str(db_path))

        # 条目元数据（名称 + 公共标签 + 流行度 + nsfw + 首播日 + 平台）
        self.subject_meta: dict[int, dict] = {}
        for sid, name, name_cn, meta_tags, fav_done, nsfw, date, platform, score in conn.execute(
            "SELECT id, name, name_cn, meta_tags, fav_done, nsfw, date, platform, score FROM subjects"
        ):
            self.subject_meta[sid] = {
                "name": name_cn or name,
                "name_cn": name_cn,
                "name_ja": name,
                "meta_tags": json.loads(meta_tags) if meta_tags else [],
                "fav_done": int(fav_done or 0),
                "nsfw": bool(nsfw),
                "date": date,
                "platform": int(platform or 0),
                "score": float(score or 0),  # BGM 平均分（前端卡片展示用）
            }

        # 语料交互：rate>0 收藏（阶段 1.5 验证的强信号），1-10 评分加权（2026-08-08 ADR 0005）。
        # 数据源 collections_rated_rate（迁移自 collections WHERE rate>0，含 rate，覆盖索引
        # (user_hash, subject_id, updated_at)）。matrix_min_rate 可再过滤低分对。
        # 20k 语料流式化：不 fetchall 全量、不建 by_user 字典、无临时排序，峰值内存 <400MB。
        # 第一遍：物品全集（限有元数据的，用于候选集与 iidx）
        item_set = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT subject_id FROM collections_rated_rate"
            )
        }
        item_set = {sid for sid in item_set if sid in self.subject_meta}
        self._items = sorted(item_set)
        self._iidx = {sid: i for i, sid in enumerate(self._items)}
        n_items = len(self._items)

        # 站点屏蔽：语料内命中者建索引，推荐/相似候选过滤用（语料外条目本就不在候选）
        self._blocked_idx: set[int] = {
            self._iidx[sid] for sid in self._site_blocked if sid in self._iidx
        }

        # 每部动画的首播年份（era gap boost 用）：date 前 4 位有效才计入，缺失=0
        self._item_year = np.zeros(n_items, dtype=np.float64)
        for i, sid in enumerate(self._items):
            d = self.subject_meta[sid].get("date") or ""
            if d[:4].isdigit():
                self._item_year[i] = int(d[:4])

        # 第二遍：按用户流式收集训练对 + 评分（用户索引按首次出现递增；索引序即 user_hash 序）
        ui_list: list[int] = []
        ii_list: list[int] = []
        rate_list: list[float] = []
        uid = 0
        last_user: str | None = None
        # df_min_rated>0：去热分母只计重度用户（评分条数≥阈值）。轻度用户只看热门番，
        # 剔除后超热番分母缩得最多、冷番基本不动，热门番出现概率温和回升（2026-08-12 解耦版）。
        user_cnt: dict[str, int] | None = None
        user_heavy: list[bool] = []
        if self._df_min_rated > 0:
            user_cnt = dict(conn.execute(
                "SELECT user_hash, COUNT(*) FROM collections_rated_rate GROUP BY user_hash"
            ))
        for uh, sid, rate in conn.execute(
            "SELECT user_hash, subject_id, rate FROM collections_rated_rate"
            " WHERE rate > ? ORDER BY user_hash",
            (self._matrix_min_rate,),
        ):
            if uh != last_user:
                last_user = uh
                uid += 1
                if user_cnt is not None:
                    user_heavy.append(user_cnt.get(uh, 0) >= self._df_min_rated)
            ii = self._iidx.get(sid)
            if ii is not None:
                ui_list.append(uid - 1)
                ii_list.append(ii)
                rate_list.append(rate)

        # 每用户"已看动画年份"聚合（era gap 的锚点数据，2026-08-18）：
        # user_year_sum[u] = Σ 用户 u 看过条目的年份（只计有有效日期的），user_year_cnt[u] = 有效条数。
        # 相似用户平均观看年份 = Σ(sim·year_sum) / Σ(sim·cnt)——按 sim 加权、且条数多的用户置信度更高。
        _ii_arr = np.asarray(ii_list, dtype=np.int64)
        _ui_arr = np.asarray(ui_list, dtype=np.int64)
        _y_pair = self._item_year[_ii_arr]
        self._user_year_sum = np.bincount(_ui_arr, weights=_y_pair, minlength=uid).astype(np.float64)
        self._user_year_cnt = np.bincount(_ui_arr, weights=(_y_pair > 0), minlength=uid).astype(np.float64)

        # franchise：同一部动画的不同季/续作/剧场版/番外（见 build_franchise 与 docs/adr/0003）
        self._franchise_root = build_franchise(self._items, conn)
        # franchise 内"第一部"：组内最早首播日(date) 的条目；全部缺日期则最小 subject_id 兜底。
        # 用于推荐时把续作/剧场版替换为系列第一部（2026-08-11 产品要求）。
        first_by_root: dict[int, int] = {}
        members_by_root: dict[int, list[int]] = {}
        for sid, root in self._franchise_root.items():
            members_by_root.setdefault(root, []).append(sid)
        for root, members in members_by_root.items():
            dated = [(self.subject_meta[s]["date"], s) for s in members
                     if self.subject_meta[s]["date"]]
            first_by_root[root] = min(dated)[1] if dated else min(members)

        # 只保留日本动画（2026-08-11 产品要求）：meta_tags 含任一非日产地标签 → 排除
        # （国漫/欧美/韩等）。日本+非日双标签（如一人之下，实为国漫）一并排除。
        # platform 2006 = 动态漫画（非真动画），一并排除。
        _non_jp_kw = ("中国", "国漫", "国产", "大陆", "欧美", "美国", "法国", "英国",
                      "德国", "韩国", "台湾", "香港", "俄罗斯", "加拿大", "意大利",
                      "西班牙", "泰国", "印度")
        self._is_jp = np.array([
            self.subject_meta[sid]["platform"] != 2006
            and not any(any(k in t for k in _non_jp_kw)
                        for t in self.subject_meta[sid]["meta_tags"])
            for sid in self._items
        ], dtype=bool)

        # 番外/剧场版排除（2026-08-12 产品要求升级）：任何有 type-11 入边的条目
        # （"XX 的番外/剧场版"），不论 TV/OVA/WEB/剧场版 一律排除——番外篇不进推荐列表。
        # 原（2026-08-11）只排除剧场版（platform==3）番外；用户要求扩展至全部平台。
        # type 11 是"XX 的番外/剧场版"关系（间谍过家家→剧场版、素晴→OAD、
        # 命运石之门→WEB 短篇都是 11）；千与千寻/你的名字等独立剧场版无 type-11 入边，保留。
        _side_targets = {
            tgt for src, tgt in conn.execute(
                "SELECT subject_id, related_subject_id FROM subject_relations"
                " WHERE relation_type = 11"
            )
        }
        self._is_side_content = np.array(
            [sid in _side_targets for sid in self._items], dtype=bool
        )
        conn.close()

        # 交互矩阵 + 行归一化。2026-08-12 相似度中心化（用户要求：同一部番打低分与打高分的
        # 用户应降低相似度）。打分矩阵 A 保持 idf[i]*rate（原始分，ADR 0005），相似度来源 Bn
        # 用中心化 idf[i]*(rate-rate_center)：1-4 负权重、5 中性、6-10 正权重，低分/高分用户
        # 余弦贡献相反符号 → 相似度降低。rate_center=0 时 Bn 与 A 同源，回到原全正相似度。
        raw = np.asarray(rate_list, dtype=np.float64)
        self._A, _, self._idf = build_encoding(
            uid, n_items, ui_list, ii_list, values=raw,
        )
        if not self._idf_in_score:
            # 分子 idf 实验（2026-08-13）：打分矩阵 A 去掉 idf 乘子（A=idf×rate→rate），
            # 相似度 Bn/q 的 idf 保留（它做邻居选择有用，实测 noA > all）。结果更热但 NDCG/Recall 大涨。
            self._A = self._A.multiply(1.0 / self._idf).tocsr()
        _, self._Bn, _ = build_encoding(
            uid, n_items, ui_list, ii_list, values=raw - self._rate_center,
        )
        # 转置缓存（浮窗相似动画用）：CSR.T → CSC 共享 data、近零拷贝。_A 构建后不可变，
        # 供 co-watch 余弦一次 matvec（_A_T @ col），避免每次重算转置。
        self._A_T = self._A.T

        # 每部动画的语料收藏人数（渗透率归一化分母，自适应 γ 用）：按训练对 bincount 计看过人数
        # （不看评分值——中心化后 rate=5 的条目权重为 0，若按 A 非零行计会漏掉实际看过的人）。
        # df_min_rated>0 时只计重度用户（分子/相似度仍用全语料，不动 idf——全量重建会被 idf 重标抵消，实测反而更糟）。
        if self._df_min_rated > 0:
            heavy_pairs = np.asarray(user_heavy)[np.asarray(ui_list)]
            self._df = np.bincount(
                np.asarray(ii_list, dtype=np.int64)[heavy_pairs],
                minlength=n_items,
            ).astype(np.float64)
        else:
            self._df = np.bincount(
                np.asarray(ii_list, dtype=np.int64), minlength=n_items,
            ).astype(np.float64)

        self._pop = np.array([self.subject_meta[sid]["fav_done"] for sid in self._items], dtype=float)
        self._log_pop = np.log1p(self._pop)

        # 全局流行度排名（1 = 最热门）：推荐池上限判定（rank ≤ rank_cap）与自适应 γ 的热度判定
        order = np.argsort(-self._pop)
        rank = np.empty(n_items, dtype=np.int64)
        rank[order] = np.arange(1, n_items + 1)
        self._pop_rank = rank

        # 与物品索引对齐的 franchise 根（孤立条目以自身为根，保证去重时互不干扰）
        self._fr = np.array(
            [self._franchise_root.get(sid, sid) for sid in self._items],
            dtype=np.int64,
        )
        # 与物品索引对齐的"系列第一部"索引（孤立条目即自身）——续作替换用
        self._first_idx = np.array(
            [
                self._iidx[first_by_root.get(self._franchise_root.get(sid, sid), sid)]
                for sid in self._items
            ],
            dtype=np.int64,
        )
        # 与物品索引对齐的 nsfw 标记（无 nsfw 口味时不推黄片，见 recommend）
        self._nsfw = np.array(
            [self.subject_meta[sid]["nsfw"] for sid in self._items],
            dtype=bool,
        )

        # 老动画标签 boost 基建（2026-08-18 路由 A 打分层，见 build_tags/ADR 0008）：
        # old_tag_beta>0 时 recommend() 触发；=0 时纯标签矩阵不影响任何输出（无分支进入）。
        self._tag_P, self._tag_Pn, self._tag_old_mask = build_tags(
            self._items, self.subject_meta, self._old_tag_year
        )
        self._tag_vocab = self._tag_P.shape[1]
        self._tag_old_count = int(self._tag_old_mask.sum())

    # ---- 对外接口 ------------------------------------------------

    def _calibrate_gamma(
        self,
        scores: np.ndarray,
        pool: list[int],
        k: int,
    ) -> float:
        """自适应去热强度：找 γ ∈ [0, gamma_max] 使推荐池前 pool_size 条的
        高热度占比（rank < hot_rank_threshold）收敛到 hot_share_target。

        渗透率归一化 score = base / (df+1)^γ（df = 语料收藏人数）：γ 越大热门压得
        越狠。厚画像口味宽、中坚动画自身分就高，小 γ 即可达标；主流向薄画像要更大 γ。
        hot_share(γ) 随 γ 单调下降（同一条的分母单调变大，热门受压制最重），二分搜索。

        pool_size 取 min(k, 40)：前端活跃池就是前 40 条非隐藏；列表后段按同一归一化
        分排序，热门/中坚按渗透率交错排列，换一批不会重新塌回全热门。
        """
        if not pool:
            return 0.0
        pool_size = min(k, 40)
        df = self._df
        rank = self._pop_rank

        def hot_share(gamma: float) -> float:
            sc = scores / np.power(df + 1, gamma)
            order = sorted(pool, key=lambda i: -sc[i])[:pool_size]
            hot = sum(1 for i in order if rank[i] < self._hot_rank_threshold)
            return hot / pool_size

        if hot_share(0.0) <= self._hot_share_target:
            return 0.0  # 不压已达标（厚画像/小众口味）
        if hot_share(self._gamma_max) >= self._hot_share_target:
            return self._gamma_max  # 压到底仍超标，取上限（薄画像/主流口味）
        lo, hi = 0.0, self._gamma_max
        for _ in range(12):
            mid = (lo + hi) / 2
            if hot_share(mid) > self._hot_share_target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _neighbor_avg_year(self, q: np.ndarray) -> float:
        """相似用户的平均观看年份（era gap 锚点）。

        与 score_items 同源：sim = Bn @ qn 取 top-knn，锚点 = Σ(sim·year_sum)/Σ(sim·cnt)
        ——sim 加权均值，且已看条数多的用户置信度更高。返回 0 表示无可靠年代锚点
        （冷启动 / 邻居都没有有效日期），调用方应跳过 era boost。
        """
        if (q != 0).sum() < self._min_profile:
            return 0.0
        qn = q / np.sqrt((q * q).sum())
        sim = np.asarray(self._Bn @ qn).ravel()
        idx = np.argpartition(-sim, self._knn)[:self._knn]
        num = float(sim[idx] @ self._user_year_sum[idx])
        den = float(sim[idx] @ self._user_year_cnt[idx])
        if den < 1e-9:
            return 0.0
        return num / den

    def similar_items(
        self,
        sid: int,
        k: int = 10,
        alpha: float | None = None,
    ) -> list[SimilarItem]:
        """条目详情浮窗的"相似动画"：混合 α×标签余弦 + (1−α)×共同观看余弦。

        两个信号都是一次 matvec，不物化 n_items² 全矩阵：
        - tag_cos[i] = Pn[i]·Pn[target]（Pn 行已 L2 归一化 → 余弦），题材亲和。
        - co_watch_cos[i] = dot(col_i, col_j) / (‖col_i‖·‖col_j‖)，列范数归一（idf 在
          余弦里抵消，与 A 是否带 idf 无关）；A 的列=条目被哪些用户看过/打分。
        过滤：自身 / 同 franchise / 番外·剧场版(type-11) / 非日本动画（与推荐池同口径）。
        不在语料（_iidx 未命中）→ 返回空列表（调用方显示"暂无相似"）。
        """
        if alpha is None:
            alpha = self._similar_alpha
        i = self._iidx.get(sid)
        if i is None:
            return []
        n_items = len(self._items)
        k = max(1, min(k, 30))

        # ① 标签余弦：Pn 行归一化，行点积即 tag_cos。
        #    注意：稀疏×稀疏→稀疏，np.asarray(稀疏) 是 0 维 object 数组；目标行转稠密后
        #    csr.dot(稠密)→稠密 ndarray（recommend() 里 _tag_Pn[pidx].dot(qn_tag) 同款）。
        tag_row = np.asarray(self._tag_Pn[i].toarray()).ravel()  # (n_tags,) 稠密
        tag_cos = np.asarray(self._tag_Pn.dot(tag_row)).ravel()  # (n_items,) 稠密

        # ② 共同观看余弦：A 的列 j = 看过 j 的用户向量；A_T @ col_i = 全部 j 与 col_i 的内积
        col = np.asarray(self._A.getcol(i).todense()).ravel()  # (n_users,) 稠密，避免 CSR 列切
        raw = np.asarray(self._A_T @ col).ravel()  # (n_items,)
        # 列范数（按条目）：axis=0 对 items 维求和；axis=1 是每用户——会错。
        col_norm = np.sqrt(np.asarray(self._A.multiply(self._A).sum(axis=0)).ravel())
        co = np.zeros(n_items)
        if col_norm[i] > 1e-9:
            nz = col_norm > 1e-9
            co[nz] = raw[nz] / (col_norm[i] * col_norm[nz])

        mixed = alpha * tag_cos + (1.0 - alpha) * co

        mask = np.ones(n_items, dtype=bool)
        mask[i] = False  # 自身
        mask[self._fr == self._fr[i]] = False  # 同 franchise（续作/剧场版/系列）
        mask[self._is_side_content] = False  # 番外/剧场版（type-11）
        mask[~self._is_jp] = False  # 非日本动画
        for bi in self._blocked_idx:
            mask[bi] = False  # 站点屏蔽：相似列表也不出现（2026-08-19）
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            return []
        top = cand[np.argpartition(-mixed[cand], min(k, cand.size))[:k]]
        top = top[np.argsort(-mixed[top])]
        return [
            SimilarItem(
                subject_id=self._items[j],
                name=self.subject_meta[self._items[j]]["name"],
                rating=self.subject_meta[self._items[j]].get("score", 0.0),
                popularity_rank=int(self._pop_rank[j]),
                score=float(mixed[j]),
            )
            for j in top
        ]

    def recommend(
        self,
        profile: list[CollectionEntry],
        already_collected: set[int],
        k: int = 20,
    ) -> RecommendResult:
        """给目标用户返回推荐 top-K（2026-08-11 删冷门发现后为单区）。

        profile: 目标用户"看过且评分"的收藏（口味信号）
        already_collected: 目标用户全部收藏的 subject_id（过滤候选）

        过滤：排除已看 / franchise 只保留第一部（续作/剧场版/番外一律排除）/ nsfw /
        非日本动画 / 番外篇（type-11 不限平台）；池子限 rank ≤ rank_cap；γ 渗透归一化后取 CF 高分 top-k。
        """
        # 口味信号：rate ≤ taste_min_rate 不算正信号（2026-08-12 实验：取消低分过滤，taste_min_rate=0；
        # 原 2026-08-08 定稿为 4，"四分以下不算"）。相似度中心化后作权重（q[i]=idf[i]*(rate-rate_center)），
        # 目标用户自己打低分的番负权重——与打高分的语料用户相似度下降，一致于 Bn（相似度）的语义。
        # franchise 排除与 nsfw 判断仍用全部 profile。
        taste = [e for e in profile if e.rate > self._taste_min_rate]
        q = encode_profile(
            [e.subject_id for e in taste],
            self._iidx,
            self._idf,
            weights=[float(e.rate) - self._rate_center for e in taste],
        )

        scores = score_items(
            self._Bn, self._A, self._log_pop, q,
            blend_lambda=self._blend_lambda, min_profile=self._min_profile,
            knn=self._knn,
        )

        # 排除用户收藏过（任意状态：看过/在看/想看）的 franchise：该系列的续作/剧场版
        # 都不进候选。2026-08-07 要求"已看续作别推"；2026-08-11 升级为"收藏过任一部即
        # 排除整系列"，覆盖看过未打分、在看、想看等状态（无职转生 Part 2 一类漏网不再出现）。
        collected_roots = {
            self._franchise_root.get(sid, sid) for sid in already_collected
        }
        # nsfw 保守默认：profile 无 nsfw 口味时，候选过滤黄片（纯黄片用户不受影响）
        profile_nsfw = any(
            self.subject_meta.get(e.subject_id, {}).get("nsfw") for e in profile
        )
        cand = [
            i for i in range(len(self._items))
            if self._items[i] not in already_collected
            and int(self._fr[i]) not in collected_roots
            and (profile_nsfw or not self._nsfw[i])
            and self._is_jp[i]               # 只保留日本动画（2026-08-11）
            and not self._is_side_content[i]  # 番外/剧场版一律不进列表（2026-08-12 升级：不限平台）
            and i not in self._blocked_idx    # 站点屏蔽：永久排除（2026-08-19）
        ]
        # franchise 去重 + 续作排除（2026-08-12 产品要求升级）：每系列只保留"第一部"
        # （组内最早首播日），续作/剧场版/番外一律丢弃，不回落。
        # 孤立条目以自身为根，天然互不合并；第一部若因候选约束（收藏/黄片/非日/番外）
        # 被过滤掉，则整个系列都不出现。原 2026-08-11 是"续作换成第一部"，改为直接排除。
        cand = [i for i in cand if int(self._first_idx[i]) == i]

        # 推荐池切分（2026-08-11 删冷门发现后）：所有合法候选里 rank ≤ rank_cap 的
        # 日本动画。冷门区（rank > 阈值）与当季新番剔除逻辑已随冷门发现功能删除。
        pool = [i for i in cand if self._pop_rank[i] <= self._rank_cap]
        # 去热（2026-08-12 实验改版）：渗透率归一化 score = base/(df+1)^γ。
        # adaptive_gamma=True 时 γ 逐用户校准到热门占比≈target；False 时全部用户固定 self._gamma。
        # 冷启动（口味信号不足走 log_pop 兜底）不归一化，保持流行度基线。
        if (q != 0).sum() >= self._min_profile:
            if self._adaptive_gamma:
                self._last_gamma = self._calibrate_gamma(scores, pool, k)
            else:
                self._last_gamma = self._gamma
            scores = scores / np.power(self._df + 1, self._last_gamma)
        else:
            self._last_gamma = self._gamma if not self._adaptive_gamma else 0.0

        # 标签 boost（2026-08-18 全池混合档 A_hyb + 年份差 era gap）：除热后给池内候选加
        # max(scale,1e-9) × boost，scale = 该用户池内除热后分数的 90 分位，boost 由两部分组成：
        #   ① tag_beta_all × cos_i                  —— 全池题材浮现（用户真实需求：题材贴但邻居没覆盖的作品）
        #      + old_tag_beta × cos_i × [date<old_tag_year]  —— 老候选额外救援（深盲区老题材，向后兼容原仅老版）
        #   ② era_gap_beta × f(|year_i − anchor|) × cos_i —— 年份差 boost（正确算法版，替代全局 2010 门）：
        #      anchor = 相似用户平均观看年份（_neighbor_avg_year，sim 加权、逐用户自适应）；
        #      f = clip(Δ / era_gap_year_span, 0, 1)。无年份常量、天然对称：新口味邻居锚点新→救老题材，
        #      老口味邻居锚点老→救新题材（老口味用户新番概率不再被全局门压低）。tag_cos 做相关度守门，非硬塞。
        # ③ 旧参数 tag_beta_all / old_tag_beta（年份门控版）保留用于回滚与对比。
        if (self._tag_beta_all > 0 or self._old_tag_beta > 0 or self._era_gap_beta > 0) and pool:
            q_tag = self._tag_P.T.dot(q)
            qn_tag = q_tag / (np.sqrt((q_tag * q_tag).sum()) + 1e-9)
            pidx = np.asarray(pool, dtype=np.int64)
            cos_pool = np.asarray(self._tag_Pn[pidx].dot(qn_tag)).ravel()
            sc_pool = scores[pidx]
            scale = float(np.percentile(sc_pool, 90)) if len(sc_pool) else 0.0
            boost = np.zeros(len(pidx))
            if self._tag_beta_all > 0:
                boost += self._tag_beta_all * cos_pool
            if self._old_tag_beta > 0:
                oldp = self._tag_old_mask[pidx]
                boost[oldp] += self._old_tag_beta * cos_pool[oldp]
            if self._era_gap_beta > 0:
                anchor = self._neighbor_avg_year(q)
                if anchor > 0:
                    dy = np.abs(self._item_year[pidx] - anchor)
                    if self._era_gap_shape == "log":
                        w = np.clip(np.log1p(dy) / np.log1p(max(self._era_gap_year_span, 1e-9)), 0.0, 1.0)
                    else:
                        w = np.clip(dy / max(self._era_gap_year_span, 1e-9), 0.0, 1.0)
                    boost += self._era_gap_beta * w * cos_pool
            scores[pidx] += max(scale, 1e-9) * boost
        top = sorted(pool, key=lambda i: -scores[i])[:k]

        def _rec(i: int) -> Recommendation:
            return Recommendation(
                subject_id=self._items[i],
                name=self.subject_meta[self._items[i]]["name"],
                score=float(scores[i]),
                popularity_rank=int(self._pop_rank[i]),
            )

        return RecommendResult(normal=[_rec(i) for i in top])

    def stats(self) -> dict:
        return {
            "users": int(self._A.shape[0]),
            "items": len(self._items),
            "rank_cap": self._rank_cap,
            "adaptive_gamma": self._adaptive_gamma,
            "hot_rank_threshold": self._hot_rank_threshold,
            "hot_share_target": self._hot_share_target,
            "gamma_max": self._gamma_max,
            "gamma": self._gamma,
            "df_min_rated": self._df_min_rated,
            "rate_center": self._rate_center,
            "idf_in_score": self._idf_in_score,
            "old_tag_beta": self._old_tag_beta,
            "tag_beta_all": self._tag_beta_all,
            "old_tag_year": self._old_tag_year,
            "era_gap_beta": self._era_gap_beta,
            "era_gap_year_span": self._era_gap_year_span,
            "era_gap_shape": self._era_gap_shape,
            "similar_alpha": self._similar_alpha,
            "tag_vocab": self._tag_vocab,
            "tag_old_count": self._tag_old_count,
            "blend_lambda": self._blend_lambda,
            "last_gamma": self._last_gamma,
            "franchise_groups": len({int(r) for r in self._fr}),
            "franchise_multi": int((self._fr != np.array(self._items)).sum()),
        }
