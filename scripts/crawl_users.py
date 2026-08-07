"""爬虫：爬取种子用户"看过"动画收藏，存入 SQLite（阶段 1）。

- 断点续爬：data/crawl_progress.json 记录已爬用户名，重启跳过
- 规模参数化：--limit 控制本次最多爬多少用户（原型 1000，可改 20000）
- 隐私：DB 只存 username 的 sha256 哈希；明文仅存在于 data/seed_users.txt（已 gitignore）
- 节流：BangumiAPI 内部已按 MIN_INTERVAL 限流，保持礼貌

用法（项目根目录）：
    python -m scripts.crawl_users --limit 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.bangumi_api import BangumiAPI
from src.config import load_token

DB = Path("data/collections.db")
SEED = Path("data/seed_users.txt")
PROGRESS = Path("data/crawl_progress.json")


def hash_user(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS collections (
            user_hash  TEXT NOT NULL,
            subject_id INTEGER NOT NULL,
            state      TEXT NOT NULL,
            rate       INTEGER NOT NULL,
            tags       TEXT NOT NULL DEFAULT '[]',
            comment    TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (user_hash, subject_id)
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_collections_subject ON collections(subject_id)")
    conn.commit()


def load_progress() -> set[str]:
    if PROGRESS.exists():
        return set(json.loads(PROGRESS.read_text(encoding="utf-8")))
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="本次最多爬多少用户")
    args = parser.parse_args()

    seed_users = [line.strip() for line in SEED.read_text(encoding="utf-8").splitlines() if line.strip()]
    done = load_progress()
    todo = [u for u in seed_users if u not in done]
    print(f"种子用户 {len(seed_users)}，已爬 {len(done)}，待爬 {len(todo)}（本次 limit={args.limit}）")

    if not todo:
        print("无需爬取。")
        return

    api = BangumiAPI(load_token())
    conn = sqlite3.connect(DB)
    init_db(conn)

    crawled = 0
    t0 = time.time()
    for username in todo:
        if crawled >= args.limit:
            break
        try:
            entries = api.fetch_collections(username, state="看过")
            uh = hash_user(username)
            conn.executemany(
                "INSERT OR REPLACE INTO collections"
                " (user_hash, subject_id, state, rate, tags, comment, updated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                [
                    (uh, e.subject_id, e.state, e.rate,
                     json.dumps(e.tags, ensure_ascii=False), e.comment, e.updated_at)
                    for e in entries
                ],
            )
            conn.commit()
            crawled += 1
            done.add(username)
            if crawled % 10 == 0:
                save_progress(done)
                elapsed = time.time() - t0
                rate = crawled / elapsed * 60 if elapsed > 0 else 0
                print(f"  已爬 {crawled}/{args.limit}（{rate:.1f} 用户/分），本用户 {len(entries)} 条")
        except Exception as e:  # noqa: BLE001 单个用户失败不影响整体
            print(f"  !! {username} 失败: {type(e).__name__} {str(e)[:100]}（不记进度，下次重试）")

    save_progress(done)
    total_rows = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
    users_done = conn.execute("SELECT COUNT(DISTINCT user_hash) FROM collections").fetchone()[0]
    print(f"完成：collections 表共 {total_rows} 行，覆盖 {users_done} 个用户")
    conn.close()


if __name__ == "__main__":
    main()
