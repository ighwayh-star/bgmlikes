#!/usr/bin/env python3
"""百度主动推送：把站内核心 URL 推送给百度搜索。

用法：
    python -m scripts.baidu_push --token <你的token>

token 在 百度搜索资源平台 → 链接提交 → 主动推送 页面生成。
注意：主动推送 / 快速收录通常要求国内备案；未备案时请用站长平台的 sitemap / 手动提交
（见 docs/PROMOTE-BAIDU.md）。
"""
from __future__ import annotations

import argparse
import urllib.request

URLS = [
    "https://bgmhiway.asia/",
    "https://bgmhiway.asia/likes",
    "https://bgmhiway.asia/daily",
    "https://bgmhiway.asia/rate",
]

API = "https://data.zz.baidu.com/urls"


def main() -> None:
    ap = argparse.ArgumentParser(description="百度主动推送")
    ap.add_argument("--token", required=True, help="百度站长平台主动推送 token")
    ap.add_argument("--site", default="https://bgmhiway.asia", help="站点地址")
    args = ap.parse_args()

    body = "\n".join(URLS)
    req = urllib.request.Request(
        f"{API}?site={args.site}&token={args.token}",
        data=body.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"HTTP {resp.status}: {resp.read().decode('utf-8')}")


if __name__ == "__main__":
    main()
