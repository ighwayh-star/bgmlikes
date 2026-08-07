"""20k 扩量流水线驱动：发现种子 → 爬取用户（串行，两步均可断点续爬）。

爬取步骤带**挂死看门狗**：本机代理抖动曾导致爬虫停在 CLOSE_WAIT 整小时无产出
（2026-08-06 实测）。若 crawl_progress.json 计数连续 STALL_MINUTES 分钟不涨，
判定挂死 → 强杀子进程重启续爬。断点续爬保证最多丢 10 个用户，无数据损失。

用法（项目根目录）：
    python scripts/run_20k.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EXE = sys.executable
PROGRESS = Path("data/crawl_progress.json")
TARGET = 20000
STALL_MINUTES = 10  # 连续多久无进展视为挂死（重度用户单次可拖数分钟，留足余量）


def crawl_progress() -> int:
    if not PROGRESS.exists():
        return 0
    try:
        return len(json.loads(PROGRESS.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 文件写入中读到半截 JSON 时当 0 处理，下轮重试
        return 0


def run_crawl_with_watchdog() -> None:
    """循环拉起爬虫；无进展超时就杀重启，直到达到 TARGET 或子进程正常收尾。"""
    while True:
        done = crawl_progress()
        if done >= TARGET:
            print(f"[watchdog] 已达目标 {done}/{TARGET}，爬取完成", flush=True)
            return

        print(f"[watchdog] 启动爬虫（当前 {done}/{TARGET}）", flush=True)
        child = subprocess.Popen(
            [EXE, "-m", "scripts.crawl_users", "--limit", str(TARGET)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        last_count, last_t = crawl_progress(), time.time()
        while True:
            time.sleep(60)  # 每分钟看一眼
            if child.poll() is not None:
                print(f"[watchdog] 爬虫进程退出 rc={child.returncode}（正常收尾）", flush=True)
                break  # 回到外层 while：若未达目标则重启续爬
            now = crawl_progress()
            if now > last_count:
                last_count, last_t = now, time.time()
            elif time.time() - last_t > STALL_MINUTES * 60:
                print(
                    f"[watchdog] 挂死 {STALL_MINUTES} 分钟无进展（停在 {last_count}），强杀重启",
                    flush=True,
                )
                child.kill()
                child.wait()
                break


def main() -> None:
    # 步骤 1：发现种子（已完成时瞬间返回 rc=0）
    print("[driver] 启动: 发现种子", flush=True)
    subprocess.call(
        [EXE, "-m", "scripts.discover_seed_users", "--target", "20000", "--subjects", "2000"]
    )
    # 步骤 2：爬取用户（带挂死看门狗）
    run_crawl_with_watchdog()
    print("[driver] 20k 流水线结束", flush=True)


if __name__ == "__main__":
    main()
