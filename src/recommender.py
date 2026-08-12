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
import sqlite3
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
    """编码交互为 IDF 加权矩阵（阶段 3：反频权重生效，评分中心化被数据否决）。

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
        taste_min_rate: int = 4,    # 口味信号排除 rate≤该值（"四分以下不算"）
        adaptive_gamma: bool = False,  # 2026-08-12 实验：关闭自适应，全部用户固定 γ
        gamma: float = 1.0,            # 固定去热强度（自适应关闭时对所有用户生效，λ=1 的流行度加成不再被对冲）
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

        # 第二遍：按用户流式收集训练对 + 评分（用户索引按首次出现递增；索引序即 user_hash 序）
        ui_list: list[int] = []
        ii_list: list[int] = []
        rate_list: list[float] = []
        uid = 0
        last_user: str | None = None
        for uh, sid, rate in conn.execute(
            "SELECT user_hash, subject_id, rate FROM collections_rated_rate"
            " WHERE rate > ? ORDER BY user_hash",
            (self._matrix_min_rate,),
        ):
            if uh != last_user:
                last_user = uh
                uid += 1
            ii = self._iidx.get(sid)
            if ii is not None:
                ui_list.append(uid - 1)
                ii_list.append(ii)
                rate_list.append(rate)

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

        # 剧场版（platform 3）若是某部非剧场版动画的番外/剧场版（存在 type-11 入边，
        # 源非剧场版）→ 排除。type 11 是"XX 的番外/剧场版"关系（间谍过家家→剧场版
        # 间谍过家家、素晴→OAD、命运石之门→WEB 短篇都是 11）；千与千寻/你的名字等
        # 独立剧场版无 type-11 入边，保留。2026-08-11 产品要求。
        _movie_series = {
            tgt for src, tgt in conn.execute(
                "SELECT r.subject_id, r.related_subject_id FROM subject_relations r"
                " JOIN subjects s ON s.id = r.subject_id"
                " WHERE r.relation_type = 11 AND s.platform != 3"
            )
        }
        self._is_movie_series = np.array(
            [sid in _movie_series for sid in self._items], dtype=bool
        )
        conn.close()

        # 评分加权 IDF 交互矩阵 + 行归一化（A[u,i]=idf[i]*rate；rate 为 1-10 分值）。
        # 与二值相比稀疏结构不变、零额外成本（ADR 0005 全量实测 Recall +6.3% / NDCG +7.2%）。
        self._A, self._Bn, self._idf = build_encoding(
            uid, n_items, ui_list, ii_list,
            values=np.asarray(rate_list, dtype=np.float64),
        )

        # 每部动画的语料收藏人数（渗透率归一化分母，自适应 γ 用）：A 列非零行数
        self._df = np.asarray((self._A != 0).sum(axis=0)).ravel()

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

    def recommend(
        self,
        profile: list[CollectionEntry],
        already_collected: set[int],
        k: int = 20,
    ) -> RecommendResult:
        """给目标用户返回推荐 top-K（2026-08-11 删冷门发现后为单区）。

        profile: 目标用户"看过且评分"的收藏（口味信号）
        already_collected: 目标用户全部收藏的 subject_id（过滤候选）

        过滤：排除已看 / franchise 排除+去重+续作替换 / nsfw / 非日本动画 /
        剧场版番外；池子限 rank ≤ rank_cap；自适应 γ 渗透归一化后取 CF 高分 top-k。
        """
        # 口味信号：排除四分以下（rate ≤ taste_min_rate 不算正信号，2026-08-08），
        # 评分作为权重（q[i]=idf[i]*rate）。franchise 排除与 nsfw 判断仍用全部 profile。
        taste = [e for e in profile if e.rate > self._taste_min_rate]
        q = encode_profile(
            [e.subject_id for e in taste],
            self._iidx,
            self._idf,
            weights=[float(e.rate) for e in taste],
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
            and self._is_jp[i]              # 只保留日本动画（2026-08-11）
            and not self._is_movie_series[i]  # 去掉 TV 的剧场版续作/番外（2026-08-11）
        ]
        # franchise 去重：每个 franchise 只保留 CF 分最高的一条，续作/剧场版不重复占位。
        # 孤立条目以自身为根，天然互不合并。
        best: dict[int, tuple[float, int]] = {}
        for i in cand:
            r = int(self._fr[i])
            if r not in best or scores[i] > best[r][0]:
                best[r] = (scores[i], i)
        # 续作替换：组内 CF 分最高的那条若是续作/剧场版（非第一部），换成该系列第一部。
        # 仅当第一部仍是合法候选（未被收藏、非黄片）时替换，否则保留原条目。
        # 2026-08-11 产品要求：列表里出现的续作都换成第一部。
        cand = []
        for r, (_, i) in best.items():
            fi = int(self._first_idx[i])
            if (fi != i
                    and self._items[fi] not in already_collected
                    and (profile_nsfw or not self._nsfw[fi])
                    and self._is_jp[fi]
                    and not self._is_movie_series[fi]):
                cand.append(fi)
            else:
                cand.append(i)

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
            "blend_lambda": self._blend_lambda,
            "last_gamma": self._last_gamma,
            "franchise_groups": len({int(r) for r in self._fr}),
            "franchise_multi": int((self._fr != np.array(self._items)).sum()),
        }
