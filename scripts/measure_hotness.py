"""测量"推荐列表变冷了吗"：纯 CF vs franchise vs 产品路径 的热度分布对比。

用户反馈"热门推荐没有以前热门"。此脚本在同一批评估用户上比较三条路径
top-20 的热度特征（popularity_rank / fav_done），把"变冷"拆解到 franchise 与配额。

用法（项目根目录）：python -m scripts.measure_hotness --users 500
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

from src.recommender import Recommender, encode_profile, quota_rank, score_items
from scripts.evaluate import load, MIN_TRAIN

DB = "data/collections.db"
K = 20
COLD_QUOTA = 8
COLD_THRESHOLD = 3000
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=500)
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

    # 每条路径收集：所有 top-20 条目的 popularity_rank 与 fav_done
    ranks = {"cf": [], "fr": [], "product": []}
    favs = {"cf": [], "fr": [], "product": []}
    n_ok = 0
    for eu in sample:
        train = [s for s in eu["train"] if s in iidx]
        if not train:
            continue
        train_idx = {iidx[s] for s in train}
        cand0 = [i for i in range(n) if i not in train_idx]
        q = encode_profile(train, iidx, r._idf)
        scores = score_items(r._Bn, r._A, r._log_pop, q, 0.0, MIN_TRAIN, knn=r._knn)

        watched = {r._franchise_root.get(s, s) for s in train}
        top_cf = sorted(cand0, key=lambda i: -scores[i])[:K]
        cand = [i for i in cand0 if int(r._fr[i]) not in watched]
        best: dict[int, tuple[float, int]] = {}
        for i in cand:
            rr = int(r._fr[i])
            if rr not in best or scores[i] > best[rr][0]:
                best[rr] = (scores[i], i)
        cand = [v[1] for v in best.values()]
        top_fr = sorted(cand, key=lambda i: -scores[i])[:K]
        top_prod = quota_rank(scores, r._pop_rank, COLD_QUOTA, COLD_THRESHOLD, cand, K)

        for name, top in [("cf", top_cf), ("fr", top_fr), ("product", top_prod)]:
            ranks[name].extend(r._pop_rank[top].tolist())
            favs[name].extend(r._pop[top].tolist())
        n_ok += 1

    print(f"\n[结果] {n_ok} 用户，k={K}，配额 {COLD_QUOTA}/{K} 冷门（rank>{COLD_THRESHOLD}）", flush=True)
    print(f"{'路径':<10}{'中位rank':>10}{'平均fav':>12}{'rank≤1000':>12}{'rank≤3000':>12}{'冷门>3000':>12}")
    for name in ["cf", "fr", "product"]:
        rk = np.array(ranks[name])
        fv = np.array(favs[name])
        pct = lambda cond: f"{cond.mean()*100:>12.1f}%"  # noqa: E731
        print(
            f"{name:<10}{np.median(rk):>10.0f}{fv.mean():>12.0f}"
            f"{pct(rk <= 1000)}{pct(rk <= COLD_THRESHOLD)}{pct(rk > COLD_THRESHOLD)}",
            flush=True,
        )
    print("\n说明：rank 越小越热门（1=全站最热）。产品 = franchise + 配额；fr = 仅 franchise（不加配额）；")
    print("cf = 纯 CF（对应你记忆里的'以前'）。产品 vs cf 的差异 = 总变冷；fr vs cf = franchise 份额；")
    print("product vs fr = 配额份额。")


if __name__ == "__main__":
    main()
