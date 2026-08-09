"""测量"推荐变冷了吗"：纯 CF vs franchise vs 产品两区 的热度分布对比。

用户反馈"推荐别全是热门"。产品 2026-08-08 改版为两区（动画推荐/冷门发现）。
此脚本在同一批评估用户上比较 top-20 的热度特征（popularity_rank / fav_done）：
- cf = 纯 CF（对应"没做冷门处理"的记忆）
- fr = 仅 franchise 排除+去重（不加池子切分）
- normal / cold = 产品两区（动画区=非冷门池、冷门区=冷门池）

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

from src.recommender import Recommender, encode_profile, score_items
from scripts.evaluate import load, MIN_TRAIN

DB = "data/collections.db"
K = 20
COLD_THRESHOLD = 3000
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=500)
    args = parser.parse_args()

    t0 = time.time()
    _, _, _, _, _, _, eval_users, _ = load()
    r = Recommender(DB)
    iidx = r._iidx
    n = len(r._items)
    print(f"[load] 评估池 {len(eval_users)} 用户，模型 {r.stats()}（{time.time()-t0:.0f}s）", flush=True)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(eval_users), size=min(args.users, len(eval_users)), replace=False)
    sample = [eval_users[int(i)] for i in pick]

    # 每条路径收集：所有 top-20 条目的 popularity_rank 与 fav_done
    ranks = {"cf": [], "fr": [], "normal": [], "cold": []}
    favs = {"cf": [], "fr": [], "normal": [], "cold": []}
    n_ok = 0
    for eu in sample:
        train = [s for s in eu["train"] if s in iidx]
        if not train:
            continue
        train_idx = {iidx[s] for s in train}
        cand0 = [i for i in range(n) if i not in train_idx]
        # 口味信号：排除四分以下（rate>4），评分加权（与产品 recommend() 同源）
        taste = [(s, r_) for s, r_ in zip(eu["train"], eu["train_rates"])
                 if s in iidx and r_ > 4]
        q = encode_profile([s for s, _ in taste], iidx, r._idf,
                           weights=[float(r_) for _, r_ in taste])
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
        # 产品两区（与 recommend() 同源）：normal=非冷门池、cold=冷门池
        top_normal = sorted((i for i in cand if r._pop_rank[i] <= COLD_THRESHOLD),
                            key=lambda i: -scores[i])[:K]
        top_cold = sorted((i for i in cand if r._pop_rank[i] > COLD_THRESHOLD),
                          key=lambda i: -scores[i])[:K]

        for name, top in [("cf", top_cf), ("fr", top_fr), ("normal", top_normal), ("cold", top_cold)]:
            ranks[name].extend(r._pop_rank[top].tolist())
            favs[name].extend(r._pop[top].tolist())
        n_ok += 1

    print(f"\n[结果] {n_ok} 用户，k={K}，冷门 = 热度排名 > {COLD_THRESHOLD}", flush=True)
    print(f"{'路径':<10}{'中位rank':>10}{'平均fav':>12}{'rank≤1000':>12}{'rank≤3000':>12}{'冷门>3000':>12}")
    for name in ["cf", "fr", "normal", "cold"]:
        rk = np.array(ranks[name])
        fv = np.array(favs[name])
        pct = lambda cond: f"{cond.mean()*100:>12.1f}%"  # noqa: E731
        print(
            f"{name:<10}{np.median(rk):>10.0f}{fv.mean():>12.0f}"
            f"{pct(rk <= 1000)}{pct(rk <= COLD_THRESHOLD)}{pct(rk > COLD_THRESHOLD)}",
            flush=True,
        )
    print("\n说明：rank 越小越热门（1=全站最热）。fr = 仅 franchise 排除+去重；normal/cold = 产品两区；")
    print("cf = 纯 CF（对应'没做冷门处理'的记忆）。normal 应几乎全部 rank≤3000，cold 应几乎全部 >3000，")
    print("二者之和即'变冷'的两个去向；fr vs cf 展示 franchise 的贡献。")


if __name__ == "__main__":
    main()
