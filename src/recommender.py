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
from datetime import date
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.bangumi_api import CollectionEntry


@dataclass(frozen=True)
class Recommendation:
    subject_id: int
    name: str
    score: float
    cold: bool = False  # 该条目是否来自冷门池（热度排名 > cold_rank_threshold）
    popularity_rank: int = 0  # 全局热度排名（1 = 最热门）


@dataclass(frozen=True)
class RecommendResult:
    """两区推荐结果（2026-08-08 产品改版，不再混合配额）。

    normal: 动画推荐区——非冷门池（rank ≤ cold_rank_threshold）的 CF 高分 top-k；
    cold:   冷门发现区——冷门池（rank > cold_rank_threshold）的 CF 高分 top-k。
    两区各自独立取 top-k，互不混合。
    """
    normal: list[Recommendation]
    cold: list[Recommendation]


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
        blend_lambda: float = 0.0,
        min_profile: int = 5,
        knn: int = 200,
        cold_quota: int = 8,
        cold_rank_threshold: int = 3000,
        matrix_min_rate: int = 0,   # 训练矩阵只收 rate>该值的对（0 = 全部 rate>0，ADR 0005）
        taste_min_rate: int = 4,    # 口味信号排除 rate≤该值（"四分以下不算"）
    ):
        self._blend_lambda = blend_lambda
        self._min_profile = min_profile
        self._knn = knn
        self._cold_quota = cold_quota
        self._cold_rank_threshold = cold_rank_threshold
        self._matrix_min_rate = matrix_min_rate
        self._taste_min_rate = taste_min_rate
        self._load(db_path)

    # ---- 加载 ---------------------------------------------------

    def _load(self, db_path: str | Path) -> None:
        conn = sqlite3.connect(str(db_path))

        # 条目元数据（名称 + 公共标签 + 流行度 + nsfw + 首播日）
        self.subject_meta: dict[int, dict] = {}
        for sid, name, name_cn, meta_tags, fav_done, nsfw, date in conn.execute(
            "SELECT id, name, name_cn, meta_tags, fav_done, nsfw, date FROM subjects"
        ):
            self.subject_meta[sid] = {
                "name": name_cn or name,
                "name_cn": name_cn,
                "name_ja": name,
                "meta_tags": json.loads(meta_tags) if meta_tags else [],
                "fav_done": int(fav_done or 0),
                "nsfw": bool(nsfw),
                "date": date,
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
        conn.close()

        # 评分加权 IDF 交互矩阵 + 行归一化（A[u,i]=idf[i]*rate；rate 为 1-10 分值）。
        # 与二值相比稀疏结构不变、零额外成本（ADR 0005 全量实测 Recall +6.3% / NDCG +7.2%）。
        self._A, self._Bn, self._idf = build_encoding(
            uid, n_items, ui_list, ii_list,
            values=np.asarray(rate_list, dtype=np.float64),
        )

        self._pop = np.array([self.subject_meta[sid]["fav_done"] for sid in self._items], dtype=float)
        self._log_pop = np.log1p(self._pop)

        # 全局流行度排名（1 = 最热门）：冷门池判定（rank > cold_rank_threshold）
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

    @staticmethod
    def _current_season_window() -> tuple[str, str]:
        """当前新番季 [起, 止) 的 ISO 日期（新番季对齐 1/4/7/10 月）。

        用于从冷门池剔除当季新番：新番刚开播收藏少、热度排名天然靠后（rank 高），
        会淹没冷门发现区（2026-08-11）。开区间：止 = 下一季起点。
        """
        today = date.today()
        q = ((today.month - 1) // 3) * 3 + 1  # 本季度首月：1/4/7/10
        start = date(today.year, q, 1)
        if q + 3 > 12:
            end = date(today.year + 1, 1, 1)
        else:
            end = date(today.year, q + 3, 1)
        return start.isoformat(), end.isoformat()

    def recommend(
        self,
        profile: list[CollectionEntry],
        already_collected: set[int],
        k: int = 20,
    ) -> RecommendResult:
        """给目标用户返回两区推荐（各 top-K，见 RecommendResult）。

        profile: 目标用户"看过且评分"的收藏（口味信号）
        already_collected: 目标用户全部收藏的 subject_id（过滤候选）

        两区共用同一份 CF 分与过滤（排除已看 / franchise 排除+去重 / nsfw），
        仅按热度池子切分：normal=非冷门池、cold=冷门池，各自取 CF 分最高的 k 条。
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
                    and (profile_nsfw or not self._nsfw[fi])):
                cand.append(fi)
            else:
                cand.append(i)

        # 两区切分（2026-08-08 产品改版，替代 quota_rank 混合）：
        #   normal = 非冷门池（rank ≤ 阈值）CF 高分 top-k
        #   cold   = 冷门池（rank > 阈值）CF 高分 top-k
        normal_pool = [i for i in cand if self._pop_rank[i] <= self._cold_rank_threshold]
        # 当季新番剔除冷门区：首播日在当前新番季（1/4/7/10 月对齐）内的不进冷门池——
        # 新番刚开播收藏少、热度排名天然靠后（rank 高），会淹没真正的冷门宝藏。
        # 无日期信息的条目无法判定，留在冷门池（2026-08-11）。
        s_start, s_end = self._current_season_window()
        cold_pool = [
            i for i in cand
            if self._pop_rank[i] > self._cold_rank_threshold
            and not (s_start <= (self.subject_meta[self._items[i]]["date"] or "0000-00-00") < s_end)
        ]
        top_normal = sorted(normal_pool, key=lambda i: -scores[i])[:k]
        top_cold = sorted(cold_pool, key=lambda i: -scores[i])[:k]

        def _rec(i: int, cold: bool) -> Recommendation:
            return Recommendation(
                subject_id=self._items[i],
                name=self.subject_meta[self._items[i]]["name"],
                score=float(scores[i]),
                cold=cold,
                popularity_rank=int(self._pop_rank[i]),
            )

        return RecommendResult(
            normal=[_rec(i, False) for i in top_normal],
            cold=[_rec(i, True) for i in top_cold],
        )

    def stats(self) -> dict:
        return {
            "users": int(self._A.shape[0]),
            "items": len(self._items),
            "cold_quota": self._cold_quota,
            "cold_rank_threshold": self._cold_rank_threshold,
            "franchise_groups": len({int(r) for r in self._fr}),
            "franchise_multi": int((self._fr != np.array(self._items)).sum()),
        }
