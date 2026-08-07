"""导入 Archive 条目录入 SQLite subjects 表（阶段 1）。

从 dump.zip 里的 subject.jsonlines 流式读取，只保留动画（type=2）。
- 不落盘整文件：直接从 zip 流式解压处理
- 幂等：INSERT OR REPLACE，可重复运行
- 含聚合数据（tags / score / rank / favorite 各状态人数）——流行度基线所需

用法（项目根目录）：
    python -m scripts.import_archive
输出：data/collections.db 的 subjects 表
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DUMP = Path("data/archive/dump.zip")
DB = Path("data/collections.db")
MEMBER = "subject.jsonlines"
BATCH = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL DEFAULT '',
    name_cn    TEXT NOT NULL DEFAULT '',
    summary    TEXT NOT NULL DEFAULT '',
    platform   INTEGER NOT NULL DEFAULT 0,
    date       TEXT NOT NULL DEFAULT '',
    nsfw       INTEGER NOT NULL DEFAULT 0,
    tags       TEXT NOT NULL DEFAULT '[]',     -- JSON [{name, count}]
    meta_tags  TEXT NOT NULL DEFAULT '[]',     -- JSON [str]
    score      REAL NOT NULL DEFAULT 0,
    score_total INTEGER NOT NULL DEFAULT 0,
    rank       INTEGER NOT NULL DEFAULT 0,
    fav_done   INTEGER NOT NULL DEFAULT 0,     -- 看过人数（流行度基线）
    fav_total  INTEGER NOT NULL DEFAULT 0,
    series     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_subjects_fav ON subjects(fav_done DESC);
"""


def extract_subject(o: dict) -> tuple:
    fav = o.get("favorite") or {}
    sd = o.get("score_details") or {}
    score_total = sum(int(v) for v in sd.values())
    return (
        int(o["id"]),
        o.get("name") or "",
        o.get("name_cn") or "",
        o.get("summary") or "",
        int(o.get("platform") or 0),
        o.get("date") or "",
        1 if o.get("nsfw") else 0,
        json.dumps(o.get("tags") or [], ensure_ascii=False),
        json.dumps(o.get("meta_tags") or [], ensure_ascii=False),
        float(o.get("score") or 0),
        score_total,
        int(o.get("rank") or 0),
        int(fav.get("done") or 0),
        int(fav.get("wish") or 0) + int(fav.get("done") or 0)
        + int(fav.get("doing") or 0) + int(fav.get("on_hold") or 0)
        + int(fav.get("dropped") or 0),
        1 if o.get("series") else 0,
    )


def main() -> None:
    conn = sqlite3.connect(DB, timeout=30)
    conn.executescript(_SCHEMA)  # 多语句需 executescript
    conn.commit()

    t0 = time.time()
    inserted = 0
    batch: list[tuple] = []
    with zipfile.ZipFile(DUMP) as z, z.open(MEMBER) as f:
        for line in f:
            o = json.loads(line)
            if o.get("type") != 2:
                continue
            batch.append(extract_subject(o))
            inserted += 1
            if len(batch) >= BATCH:
                conn.executemany(
                    "INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    batch,
                )
                conn.commit()
                batch.clear()
                if inserted % 20000 == 0:
                    print(f"  {inserted} 条动画（{time.time()-t0:.0f}s）")
        if batch:
            conn.executemany(
                "INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
    print(f"完成：导入 {inserted} 条动画，subjects 表共 {total} 行，用时 {time.time()-t0:.0f}s")
    conn.close()


if __name__ == "__main__":
    main()
