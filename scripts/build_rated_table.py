"""一次性迁移：物化 collections_rated_rate 子表（rate>0 收藏 + rate），供评分加权矩阵加载。

背景（2026-08-08）：产品改用 1-10 评分加权（A[u,i]=idf[i]*rate），但现有 collections_rated
子表（仅 rate>0 的 (user_hash, subject_id, updated_at)）不含 rate。collections 全表
ORDER BY user_hash 在本机是慢路径（索引扫描+rowid 随机读 ~2M行/73s），故一次性物化带 rate
的快速读取子表（覆盖索引 (user_hash, subject_id, updated_at) 保持用户索引序）。

用法（项目根目录）：python -m scripts.build_rated_table
"""
from __future__ import annotations

import sqlite3
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

DB = "data/collections.db"


def main() -> None:
    conn = sqlite3.connect(DB)
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS collections_rated_rate")
    conn.execute("""
        CREATE TABLE collections_rated_rate AS
        SELECT user_hash, subject_id, rate, updated_at
        FROM collections WHERE rate > 0
    """)
    conn.commit()
    print(f"[build] collections_rated_rate 物化完成（{time.time()-t0:.0f}s）", flush=True)
    t1 = time.time()
    # 覆盖索引必须含 rate（否则读 rate 触发每行 rowid 随机查，本机慢路径 ~90s）
    conn.execute(
        "CREATE INDEX idx_crr_cover ON collections_rated_rate"
        "(user_hash, subject_id, rate, updated_at)"
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM collections_rated_rate").fetchone()[0]
    print(f"[index] 索引完成（{time.time()-t1:.0f}s），行数 {n}", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
