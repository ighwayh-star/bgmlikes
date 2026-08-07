"""产品路径离线评估：测量与线上完全一致的推荐配置。

线上产品 = CF（idf_binary + KNN=200, λ=0）+ franchise 排除/去重 + 冷门配额(20中8)。
此脚本把这一整条路径在固定评估集（时间留出）上跑，报告：
- Recall@10 / NDCG@10：纯 CF 对比产品路径（诚实代价：续作被赶出后 NDCG 会掉、Recall 应保持）
- 产品路径的组成：冷门占比（配额应恰好 ~40%）、同 franchise 重复数（应恒为 0）

复用 recommender 的共享函数（build_encoding/encode_profile/score_items/quota_rank/
build_franchise + Recommender 本体），与产品实现同源防漂移。

用法（项目根目录）：
    python -m scripts.evaluate_product --users 500
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.recommender import (
    Recommender,
    encode_profile,
    quota_rank,
    score_items,
)
from scripts.evaluate import load, MIN_TRAIN, ndcg_recall

DB = "data/collections.db"
K = 20  # 产品返回 20 条；指标 @10 取其中前 10
COLD_QUOTA = 8
COLD_THRESHOLD = 3000
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=500, help="采样评估用户数")
    args = parser.parse_args()

    t0 = time.time()
    _, _, _, _, _, eval_users, _ = load()
    r = Recommender(DB)
    iidx = r._iidx
    n = len(r._items)
    print(f"[load] 评估池 {len(eval_users)} 用户，模型 {r.stats()}（{time.time()-t0:.0f}s）", flush=True)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(eval_users), size=min(args.users, len(eval_users)), replace=False)
    sample = [eval_users[int(i)] for i in pick]

    acc = {"cf": [0.0, 0.0], "fr": [0.0, 0.0], "product": [0.0, 0.0]}  # [recall, ndcg]
    n_ok = 0
    cold_total = 0
    dup_total = 0
    sequel_in_test = 0  # test 集中属于已看 franchise 的条目数（= franchise 排除的命中损失来源）
    for eu in sample:
        train = [s for s in eu["train"] if s in iidx]
        test = {s for s in eu["test"] if s in iidx}
        if not train or not test:
            continue
        train_idx = {iidx[s] for s in train}
        cand0 = [i for i in range(n) if i not in train_idx]
        q = encode_profile(train, iidx, r._idf)
        scores = score_items(r._Bn, r._A, r._log_pop, q, 0.0, MIN_TRAIN, knn=r._knn)

        # 已看 franchise 集合 + test 中被排除的续作/同季数
        watched = {r._franchise_root.get(s, s) for s in train}
        sequel_in_test += sum(
            1 for s in test if r._franchise_root.get(s, s) in watched
        )

        # 纯 CF（top-K by score）
        top_cf = sorted(cand0, key=lambda i: -scores[i])[:K]
        # franchise 排除 + 去重（不加配额）
        cand = [i for i in cand0 if int(r._fr[i]) not in watched]
        best: dict[int, tuple[float, int]] = {}
        for i in cand:
            rr = int(r._fr[i])
            if rr not in best or scores[i] > best[rr][0]:
                best[rr] = (scores[i], i)
        cand = [v[1] for v in best.values()]
        top_fr = sorted(cand, key=lambda i: -scores[i])[:K]
        # 产品路径：franchise + 配额
        top_prod = quota_rank(scores, r._pop_rank, COLD_QUOTA, COLD_THRESHOLD, cand, K)

        for name, top in [("cf", top_cf), ("fr", top_fr), ("product", top_prod)]:
            d, rec = ndcg_recall([r._items[i] for i in top], test)
            acc[name][0] += rec
            acc[name][1] += d

        # 产品组成统计（@K=20 全列表）
        rks = r._pop_rank[top_prod]
        cold_total += int((rks > COLD_THRESHOLD).sum())
        roots = [int(r._fr[i]) for i in top_prod]
        dup_total += int(len(roots) != len(set(roots)))
        n_ok += 1

    n = n_ok
    print(f"\n[结果] {n} 用户，k={K}（指标@10），配额 {COLD_QUOTA}/{K} 冷门（热度>{COLD_THRESHOLD}）", flush=True)
    print(f"{'方法':<10}{'Recall@10':>12}{'NDCG@10':>12}")
    base = acc["cf"]
    for name in ["cf", "fr", "product"]:
        v = acc[name]
        print(f"{name:<10}{v[0]/n:>12.4f}{v[1]/n:>12.4f}", flush=True)
    d_rec = (acc["product"][0] - base[0]) / base[0] * 100
    d_ndcg = (acc["product"][1] - base[1]) / base[1] * 100
    d_rec_fr = (acc["fr"][0] - base[0]) / base[0] * 100
    print(f"\n产品 vs 纯CF：Recall {d_rec:+.1f}% / NDCG {d_ndcg:+.1f}%")
    print(f"  其中 franchise 排除+去重（不加配额）：Recall {d_rec_fr:+.1f}%")
    print(f"  test 中被 franchise 排除的续作/同季命中数：{sequel_in_test} 条 / {n} 用户"
          f"（平均 {sequel_in_test/n:.1f}，这是 Recall 损失的来源，也是用户要求清除的对象）")
    print(f"产品列表组成：冷门占比 {cold_total/(n*K)*100:.1f}%（配额目标 40%）；同 franchise 重复用户数 {dup_total}/{n}")


if __name__ == "__main__":
    main()
