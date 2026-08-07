"""导入 Archive 的 subject-relations 到 SQLite（阶段 3：franchise 去重所需）。

只保留两端都是动画（subjects 表中存在）的关系边，全部 relation_type 都存
（recommender 加载时按 type∈{2,3,6} 建 franchise；其余类型留作他用）。
幂等：先清空 subject_relations 再写入，可重复运行。

用法（项目根目录）：
    python -m scripts.import_relations
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

DUMP = Path("data/archive/dump.zip")
DB = Path("data/collections.db")
MEMBER = "subject-relations.jsonlines"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_relations (
    subject_id          INTEGER NOT NULL,
    related_subject_id  INTEGER NOT NULL,
    relation_type       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sr_subject ON subject_relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_sr_related ON subject_relations(related_subject_id);
"""


def main() -> None:
    conn = sqlite3.connect(DB, timeout=30)
    conn.executescript(_SCHEMA)
    conn.execute("DELETE FROM subject_relations")
    conn.commit()

    anime = set(r[0] for r in conn.execute("SELECT id FROM subjects"))
    t0 = time.time()
    batch: list[tuple[int, int, int]] = []
    n = 0
    with zipfile.ZipFile(DUMP) as z, z.open(MEMBER) as f:
        for line in f:
            o = json.loads(line)
            a, b, t = o["subject_id"], o["related_subject_id"], o["relation_type"]
            if a not in anime or b not in anime:
                continue
            batch.append((a, b, t))
            n += 1
            if len(batch) >= 5000:
                conn.executemany("INSERT INTO subject_relations VALUES (?,?,?)", batch)
                conn.commit()
                batch.clear()
        if batch:
            conn.executemany("INSERT INTO subject_relations VALUES (?,?,?)", batch)
            conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM subject_relations").fetchone()[0]
    print(f"完成：写入 {n} 条动画-动画关系（表共 {total} 行），用时 {time.time()-t0:.0f}s")
    conn.close()


if __name__ == "__main__":
    main()
