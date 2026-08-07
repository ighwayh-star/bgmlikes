"""种子用户发现管道（阶段 1，扩量版）。

从动画条目的公开收藏页收集用户名，去重落盘，供爬虫使用。
v0 API 没有"条目→收藏者"方向接口，只能爬公开 HTML：
    https://bgm.tv/subject/{id}/collections?page=N

扩量改造（20k 目标）：
- 增量续爬：已有 seed_users.txt 作为起点，跳过已爬过的条目（data/discover_progress.json）
- 大条目池：/v0/subjects?type=2 用 offset 分页取 top-N 条目
- 每条目落盘：进度与种子每扫完一个条目即写（崩溃即恢复，进度可见）
- 抗代理挂死：连接异常自动重建客户端（同 bangumi_api._reset_client）；单条目超时跳过

用法（项目根目录）：
    python -m scripts.discover_seed_users --target 20000 --subjects 2000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.bangumi_api import BangumiAPI
from src.config import load_token

BASE = "https://bgm.tv/subject"
OUT = Path("data/seed_users.txt")
PROGRESS = Path("data/discover_progress.json")
PAGE_DELAY = 1.0  # 对网站保持礼貌：页间隔 1 秒
MAX_PAGES_PER_SUBJECT = 20  # 每条目最多翻 20 页（约 400 个用户名）
SUBJECT_DEADLINE = 180.0  # 单条目最大耗时（秒），防代理挂死拖死进程
USER_RE = re.compile(r'href="/user/([^"/]+)"')


def make_client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"},
    )


def load_seeds() -> set[str]:
    if OUT.exists():
        return {line.strip() for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()


def load_progress() -> set[int]:
    if PROGRESS.exists():
        return set(json.loads(PROGRESS.read_text(encoding="utf-8")))
    return set()


def save_progress(scraped: set[int]) -> None:
    PROGRESS.write_text(json.dumps(sorted(scraped)), encoding="utf-8")


def write_seeds(found: set[str]) -> None:
    OUT.write_text("\n".join(sorted(found)), encoding="utf-8")


def subject_pool(n: int) -> list[int]:
    api = BangumiAPI(load_token())
    ids: list[int] = []
    offset = 0
    while len(ids) < n:
        resp = api._get("/v0/subjects", params={"type": 2, "limit": 100, "offset": offset})
        batch = [int(s["id"]) for s in resp.get("data", [])]
        if not batch:
            break
        ids.extend(batch)
        offset += len(batch)
        if len(batch) < 100:  # 到底了
            break
    return ids[:n]


def collect_subject(subject_id: int, client: httpx.Client, found: set[str]) -> None:
    """翻条目收藏页收集用户名；超时自动放弃（不抛异常拖死主循环）。"""
    t0 = time.time()
    page = 1
    while page <= MAX_PAGES_PER_SUBJECT:
        if time.time() - t0 > SUBJECT_DEADLINE:
            print(f"  subject {subject_id} 超时跳过（{int(time.time()-t0)}s）", flush=True)
            return
        html = client.get(f"{BASE}/{subject_id}/collections", params={"page": page}).text
        users = set(USER_RE.findall(html))
        found.update(users)
        if not users:
            break  # 到底了
        page += 1
        time.sleep(PAGE_DELAY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=20000)
    parser.add_argument("--subjects", type=int, default=2000, help="最多扫多少个条目")
    args = parser.parse_args()

    found = load_seeds()
    scraped = load_progress()
    print(f"起点：已有种子 {len(found)}，已扫条目 {len(scraped)}，目标 {args.target}", flush=True)
    if len(found) >= args.target:
        # 目标已达成：直接收尾，不再拉条目池（避免重启时被代理拖慢）
        print("已达到目标，无需再发现。", flush=True)
        save_progress(scraped)
        write_seeds(found)
        return

    client = make_client()
    n_fetched = 0
    t0 = time.time()
    try:
        pool = subject_pool(args.subjects)
        print(f"条目池 {len(pool)} 个（前 10：{pool[:10]}）", flush=True)
        for sid in pool:
            if len(found) >= args.target:
                print("已达到目标，停止。", flush=True)
                break
            if sid in scraped:
                continue
            before = len(found)
            try:
                collect_subject(sid, client, found)
                scraped.add(sid)
                n_fetched += 1
                # 每扫完一个条目即落盘：崩溃可立即恢复，进度可见
                save_progress(scraped)
                write_seeds(found)
                rate = n_fetched / (time.time() - t0) * 60
                print(
                    f"  已扫 {n_fetched} 条目，累计 {len(found)} 用户名"
                    f"（{rate:.1f} 条目/分），本条目 +{len(found)-before}",
                    flush=True,
                )
            except httpx.TransportError:
                # 代理抖动：重建客户端，跳过本条目，下个继续
                print(f"  !! subject {sid} 连接异常，重建客户端后继续", flush=True)
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
                client = make_client()
            except Exception as e:  # noqa: BLE001 单个条目失败不影响整体
                print(f"  !! subject {sid} 失败: {type(e).__name__} {str(e)[:80]}（下次重试）", flush=True)
    finally:
        client.close()

    save_progress(scraped)
    write_seeds(found)
    print(f"完成：{len(found)} 个唯一用户名（目标 {args.target}）-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
