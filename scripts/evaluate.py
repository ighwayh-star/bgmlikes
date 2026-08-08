"""双门槛评估·门槛 1：离线指标（固定评估子集，时间留出）。

复用 recommender.score_items（与产品算法同源，防漂移）。
- 重度用户（评分≥30）按时间留出 20% 作测试集
- 对比：流行度基线 / 纯 CF / 产品默认混合(λ=0.0)
- 矩阵编码：IDF 加权 + KNN 候选池收敛（与产品 recommender 同源，防漂移）
- 指标：NDCG@10、Recall@10，报告相对流行度的提升%

20k 语料内存适配：SQL 按用户流式读取，逐用户切分并直接发射训练对，
避免 4.6M 条 Python 元组/字典驻留内存（峰值 <300MB；稠密矩阵版需 ~5GB）。

用法（项目根目录）：
    python -m scripts.evaluate
"""
from __future__ import annotations

import math
import sqlite3
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.recommender import build_encoding, encode_profile, score_items

DB = "data/collections.db"
MIN_COLLECT = 30
MIN_TRAIN = 10
BLEND_LAMBDA = 0.0  # 产品默认混合系数（阶段3：IDF 后 λ 归零）
KNN_POOL = 200  # 产品默认候选池收敛（阶段3-3 定稿）


def load():
    """流式加载并时间切分。返回 (fav, iidx, item_list, ui, ii, rate, eval_users, n_users)。

    ui/ii/rate: 全体用户 train 交互的 (用户索引, 物品索引, 评分) 数组，供 build_encoding
    （评分加权，ADR 0005）。
    eval_users: [{ui, train(列表), train_rates, test(set)}]，仅重度用户，用于打分评估。
    """
    conn = sqlite3.connect(DB)
    fav = dict(conn.execute("SELECT id, fav_done FROM subjects").fetchall())

    # 第一遍：物品全集（含 test 部分，保证 iidx 覆盖候选池）。
    # collections_rated_rate 为物化子表（rate>0 + rate，user_hash 索引序），无过滤无排序，扫描快。
    item_set = {
        r[0] for r in conn.execute("SELECT DISTINCT subject_id FROM collections_rated_rate")
    }
    item_list = sorted(item_set)
    iidx = {sid: i for i, sid in enumerate(item_list)}
    n_items = len(item_list)

    ui_list: list[int] = []
    ii_list: list[int] = []
    rate_list: list[float] = []
    eval_users: list[dict] = []
    uid = 0
    last_user: str | None = None
    buf: list[tuple[int, int, str]] = []  # (sid, rate, updated_at)，按时间排序后切分

    def flush() -> None:
        nonlocal buf
        buf.sort(key=lambda x: x[2])  # 每用户内按收藏时间排序，时间切分
        n = len(buf)
        split = int(n * 0.8) if n >= MIN_COLLECT else n
        train = buf[:split]
        test = buf[split:]
        for sid, rate, _ in train:
            ii_list.append(iidx[sid])
            ui_list.append(uid - 1)
            rate_list.append(rate)
        if n >= MIN_COLLECT and test and len(train) >= MIN_TRAIN:
            eval_users.append({
                "ui": uid - 1,
                "train": [s for s, _, _ in train],
                "train_rates": [r for _, r, _ in train],
                "test": set(s for s, _, _ in test),
            })
        buf = []

    for uh, sid, rate, ts in conn.execute(
        "SELECT user_hash, subject_id, rate, updated_at"
        " FROM collections_rated_rate ORDER BY user_hash"
    ):
        if uh != last_user:
            flush()
            last_user = uh
            uid += 1
        buf.append((sid, rate, ts))
    flush()
    conn.close()

    return (
        fav,
        iidx,
        item_list,
        np.array(ui_list, dtype=np.int64),
        np.array(ii_list, dtype=np.int64),
        np.array(rate_list, dtype=np.float64),
        eval_users,
        uid,
    )


def ndcg_recall(ranked, test, k=10):
    if not test:
        return 0.0, 0.0
    hits = dcg = 0
    for rank, sid in enumerate(ranked[:k]):
        if sid in test:
            hits += 1
            dcg += 1.0 / math.log2(rank + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(test), k)))
    return (dcg / idcg if idcg else 0.0), hits / len(test)


def main():
    t0 = time.time()
    fav, iidx, item_list, ui, ii, rate, eval_users, n_users = load()
    n_items = len(item_list)
    print(
        f"[load] {n_users} 用户 / {n_items} 物品 / 评估 {len(eval_users)} 重度用户"
        f"（{time.time()-t0:.0f}s）",
        flush=True,
    )

    # 训练矩阵（仅 train，防泄漏）——评分加权 IDF 稀疏编码，与产品 recommender 同源（ADR 0005）
    t1 = time.time()
    A, Bn, idf = build_encoding(n_users, n_items, ui, ii, values=rate)
    log_pop = np.log1p(np.array([fav.get(sid, 0) for sid in item_list], dtype=float))
    print(f"[build] 稀疏矩阵构建完成（{time.time()-t1:.0f}s）", flush=True)

    methods = {"popularity": [0.0, 0.0], "cf_pure": [0.0, 0.0], "cf_hybrid": [0.0, 0.0]}
    cnt = 0
    t2 = time.time()
    for eu in eval_users:
        ui_, train, train_rates, test = eu["ui"], eu["train"], eu["train_rates"], eu["test"]
        train_idx = {iidx[sid] for sid in train}
        cand = [i for i in range(n_items) if i not in train_idx]

        # 口味信号：排除四分以下（rate>4），评分加权（与产品 recommend() 同源）
        taste = [(s, r) for s, r in zip(train, train_rates) if r > 4]
        q = encode_profile(
            [s for s, _ in taste], iidx, idf, weights=[float(r) for _, r in taste])

        s_pop = log_pop
        # BLEND_LAMBDA=0.0 时 cf_pure 与 cf_hybrid 完全相同，只算一次（省一半计算）
        s_cf = score_items(Bn, A, log_pop, q, blend_lambda=0.0, min_profile=MIN_TRAIN, knn=KNN_POOL)
        s_hyb = s_cf if BLEND_LAMBDA == 0.0 else score_items(
            Bn, A, log_pop, q, blend_lambda=BLEND_LAMBDA, min_profile=MIN_TRAIN, knn=KNN_POOL)

        for name, sc in [("popularity", s_pop), ("cf_pure", s_cf), ("cf_hybrid", s_hyb)]:
            order = sorted(cand, key=lambda i: -sc[i])
            ndcg, rec = ndcg_recall([item_list[i] for i in order], test)
            methods[name][0] += ndcg
            methods[name][1] += rec
        cnt += 1
        if cnt % 1000 == 0:
            print(f"  ...评估 {cnt}/{len(eval_users)}（{time.time()-t2:.0f}s）", flush=True)

    print(f"\n[结果] 评估 {cnt} 用户，用时 {time.time()-t2:.0f}s", flush=True)
    print(f"{'方法':<12}{'NDCG@10':>10}{'Recall@10':>12}{'NDCG提升':>10}{'Recall提升':>10}")
    base = methods["popularity"]
    for name in ["popularity", "cf_pure", "cf_hybrid"]:
        v = methods[name]
        n = cnt or 1
        d_ndcg = (v[0] / n - base[0] / n) / (base[0] / n + 1e-9) * 100
        d_rec = (v[1] / n - base[1] / n) / (base[1] / n + 1e-9) * 100
        print(f"{name:<12}{v[0]/n:>10.4f}{v[1]/n:>12.4f}{d_ndcg:>+9.1f}%{d_rec:>+10.1f}%")

    print("\n门槛1判据：NDCG@10 与 Recall@10 相对流行度基线提升（产品默认 cf_hybrid）")


if __name__ == "__main__":
    main()
