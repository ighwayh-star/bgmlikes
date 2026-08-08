"""实验：1-10 正向评分加权 vs 现二值（矩阵编码对比）。

动机：用户问"能否按 1-10 分值加权计算"。阶段 3-2 在旧判据（NDCG）下测过
多种评分形态劣于"只看过"二值（ADR 0001）；现产品转"发现"方向，按新判据
（Recall + 冷门占比）在采样子集上重测 1-10 线性加权是否有意外收益。

数据路径（本机 3.6GB 库全表扫描慢：ORDER BY 走索引+rowid 随机读 ~2M行/73s）：
- collections_rated 覆盖索引 (user_hash, subject_id, updated_at) 快速拿重度用户
- 逐用户走 collections 唯一索引 (user_hash, subject_id) 取 (subject_id, rate, updated_at)
- 时间切分 80/20，矩阵只含 train（防泄漏）

编码对比（同一 (user,item) 对集 = rate>0，仅数值不同）：
- binary: A[u,i] = idf[i]                （现产品）
- rated:  A[u,i] = idf[i] * rate         （1-10 线性加权；query 同乘 rate）

指标：产品路径（CF + franchise 排除/去重 + 冷门配额 20中8）
Recall@10 / NDCG@10 / 冷门占比（对比在同一采样子集上进行，绝对数与全量产品评估不同）。

用法（项目根目录）：
    python -m scripts.experiment_rating_weight --matrix-users 2000 --users 400
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.recommender import build_encoding, build_franchise, encode_profile, quota_rank, score_items
from scripts.evaluate import MIN_TRAIN, ndcg_recall

DB = "data/collections.db"
K = 20
COLD_QUOTA = 8
COLD_THRESHOLD = 3000
KNN_POOL = 200
SEED = 42


def load_sampled(matrix_users: int, eval_users: int, min_collect: int):
    """采样重度用户，逐用户取 rate>0 收藏，时间切分，返回矩阵与评估集。"""
    conn = sqlite3.connect(DB)
    t0 = time.time()
    counts = dict(conn.execute(
        "SELECT user_hash, COUNT(*) FROM collections_rated GROUP BY user_hash"
    ))
    heavy = sorted(u for u, n in counts.items() if n >= min_collect)
    print(f"[load] 重度用户(≥{min_collect}) {len(heavy)} 个（{time.time()-t0:.0f}s）", flush=True)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(heavy), size=min(matrix_users, len(heavy)), replace=False)
    matrix_users_list = [heavy[int(i)] for i in pick]
    epick = rng.choice(len(matrix_users_list), size=min(eval_users, len(matrix_users_list)), replace=False)
    eval_set = {matrix_users_list[int(i)] for i in epick}

    rows_by_user: dict[str, list] = {}
    for uh in matrix_users_list:
        rows_by_user[uh] = conn.execute(
            "SELECT subject_id, rate, updated_at FROM collections WHERE user_hash=? AND rate>0",
            (uh,),
        ).fetchall()

    item_set: set[int] = set()
    for rows in rows_by_user.values():
        item_set.update(sid for sid, _, _ in rows)
    item_list = sorted(item_set)
    iidx = {sid: i for i, sid in enumerate(item_list)}
    n_items = len(item_list)
    print(f"[load] 采样 {len(matrix_users_list)} 用户 / {n_items} 物品（{time.time()-t0:.0f}s）", flush=True)

    # 元数据：fav_done → 流行度排名；franchise 根
    fav = dict(conn.execute("SELECT id, fav_done FROM subjects").fetchall())
    fr_map = build_franchise(item_list, conn)
    conn.close()
    pop = np.array([fav.get(sid, 0) for sid in item_list], dtype=float)
    log_pop = np.log1p(pop)
    order = np.argsort(-pop)
    pop_rank = np.empty(n_items, dtype=np.int64)
    pop_rank[order] = np.arange(1, n_items + 1)
    fr = np.array([fr_map.get(sid, sid) for sid in item_list], dtype=np.int64)

    # train 对（全部矩阵用户）+ 评估集
    ui_list: list[int] = []
    ii_list: list[int] = []
    rate_list: list[float] = []
    eval_list: list[dict] = []
    uid = 0
    for uh in matrix_users_list:
        rows = rows_by_user[uh]
        rows.sort(key=lambda r: r[2])  # updated_at 时间切分
        n = len(rows)
        split = int(n * 0.8)
        train = rows[:split]
        test = rows[split:]
        for sid, rate, _ in train:
            ii_list.append(iidx[sid])
            ui_list.append(uid)
            rate_list.append(rate)
        if uh in eval_set and test and len(train) >= MIN_TRAIN:
            eval_list.append({
                "ui": uid,
                "train": [(s, r) for s, r, _ in train],
                "test": set(s for s, _, _ in test),
            })
        uid += 1

    return (
        item_list,
        iidx,
        np.array(ui_list, dtype=np.int64),
        np.array(ii_list, dtype=np.int64),
        np.array(rate_list, dtype=np.float64),
        eval_list,
        uid,
        log_pop,
        pop_rank,
        fr,
    )


def eval_model(A, Bn, idf, log_pop, iidx, item_list, pop_rank, fr, eval_list,
               min_rate_query: int = 0, weight_mode: str = "none"):
    """产品路径评估（CF + franchise + 配额）。

    min_rate_query: 口味信号只取 rate>该阈值的条目（"排除四分以下"作用于查询层）。
    weight_mode: "none"=纯二值 idf；"rate"=idf*rate。
    候选排除与 franchise 排除始终基于全部已看（rate>0），不受阈值影响。
    返回 (acc, cold_total, dup_total, n_ok)。
    """
    n = len(item_list)
    acc = {"cf": [0.0, 0.0], "fr": [0.0, 0.0], "product": [0.0, 0.0]}
    cold_total = dup_total = n_ok = 0
    for eu in eval_list:
        all_train = [s for s, _ in eu["train"]]  # 全部已看（含低分）：候选排除 + franchise 排除
        pairs = [(s, r) for s, r in eu["train"] if r > min_rate_query]  # 口味信号子集
        train = [s for s, _ in pairs]
        rates = [r for _, r in pairs]
        test = eu["test"]
        train_idx = {iidx[s] for s in all_train}
        cand0 = [i for i in range(n) if i not in train_idx]
        q = encode_profile(train, iidx, idf) if weight_mode == "none" else encode_profile(
            train, iidx, idf, weights=rates
        )
        scores = score_items(Bn, A, log_pop, q, 0.0, MIN_TRAIN, knn=KNN_POOL)

        watched = {int(fr[iidx[s]]) for s in all_train}
        top_cf = sorted(cand0, key=lambda i: -scores[i])[:K]
        cand = [i for i in cand0 if int(fr[i]) not in watched]
        best: dict[int, tuple[float, int]] = {}
        for i in cand:
            rr = int(fr[i])
            if rr not in best or scores[i] > best[rr][0]:
                best[rr] = (scores[i], i)
        cand = [v[1] for v in best.values()]
        top_fr = sorted(cand, key=lambda i: -scores[i])[:K]
        top_prod = quota_rank(scores, pop_rank, COLD_QUOTA, COLD_THRESHOLD, cand, K)

        for name, top in [("cf", top_cf), ("fr", top_fr), ("product", top_prod)]:
            d, rec = ndcg_recall([item_list[i] for i in top], test)
            acc[name][0] += rec
            acc[name][1] += d

        rks = pop_rank[top_prod]
        cold_total += int((rks > COLD_THRESHOLD).sum())
        roots = [int(fr[i]) for i in top_prod]
        dup_total += int(len(roots) != len(set(roots)))
        n_ok += 1
    return acc, cold_total, dup_total, n_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-users", type=int, default=2000)
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--min-collect", type=int, default=30)
    args = parser.parse_args()

    item_list, iidx, ui, ii, rate, eval_list, n_users, log_pop, pop_rank, fr = load_sampled(
        args.matrix_users, args.users, args.min_collect
    )
    n_items = len(item_list)

    CONFIGS = [
        ("binary",    "none", 0, 0),  # 现产品：二值 IDF（基线）
        ("rated",     "rate", 0, 0),  # 1-10 加权，全部条目
        ("excl_q",    "rate", 0, 4),  # 加权 + 查询层排除 ≤4（候选产品形态）
        ("excl_both", "rate", 4, 4),  # 加权 + 矩阵和查询都排除 ≤4
    ]
    results: dict = {}
    for label, wmode, mmin, qmin in CONFIGS:
        t = time.time()
        mask = rate > mmin
        if wmode == "none":
            A, Bn, idf = build_encoding(n_users, n_items, ui[mask], ii[mask])
        else:
            A, Bn, idf = build_encoding(n_users, n_items, ui[mask], ii[mask], values=rate[mask])
        acc, cold, dup, n_ok = eval_model(
            A, Bn, idf, log_pop, iidx, item_list, pop_rank, fr, eval_list, qmin, wmode)
        results[label] = {"acc": acc, "cold": cold, "dup": dup, "n_ok": n_ok}
        print(f"[{label}] 矩阵 mmin={mmin} 权重={wmode}，评估 {n_ok} 用户（{time.time()-t:.0f}s）", flush=True)

    n = min(r["n_ok"] for r in results.values())
    print(f"\n[结果] 评估 {n} 用户，矩阵 {n_users} 用户/{n_items} 物品，k={K}（指标@10），"
          f"配额 {COLD_QUOTA}/{K} 冷门（热度>{COLD_THRESHOLD}）", flush=True)
    print(f"{'编码':<11}{'路径':<8}{'Recall@10':>12}{'NDCG@10':>12}")
    for label, r in results.items():
        for name in ["cf", "fr", "product"]:
            v = r["acc"][name]
            print(f"{label:<11}{name:<8}{v[0]/n:>12.4f}{v[1]/n:>12.4f}", flush=True)
        print(f"{label:<11}{'冷门占比':<8}{r['cold']/(n*K)*100:>12.1f}%"
              f"{'同fr重复':>14}{r['dup']}/{n}", flush=True)

    base = results["binary"]["acc"]["product"]
    print("\n对比（产品路径 vs binary）：")
    for label, r in results.items():
        if label == "binary":
            continue
        v = r["acc"]["product"]
        print(f"{label:<11}Recall {((v[0]-base[0])/base[0]*100):+.2f}%"
              f" / NDCG {((v[1]-base[1])/base[1]*100):+.2f}%", flush=True)

    os.makedirs("experiments", exist_ok=True)
    out = {
        "experiment": "rating_weight_configs",
        "date": "2026-08-08",
        "matrix_users": args.matrix_users,
        "eval_users": n,
        "n_items": n_items,
        "configs": {label: {k: [v[0] / n, v[1] / n] for k, v in r["acc"].items()}
                    for label, r in results.items()},
    }
    with open(f"experiments/rating_weight_configs_{args.matrix_users}u.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] experiments/rating_weight_configs_{args.matrix_users}u.json", flush=True)


if __name__ == "__main__":
    main()
