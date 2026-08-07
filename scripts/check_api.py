"""阶段 0 门禁：验证 token + fetch_collections 闭环。

用法（在项目根目录）：
    python -m scripts.check_api

流程：/v0/me 拿当前用户名 -> 拉取该用户"看过"动画 -> 打印统计与样例。
"""
from __future__ import annotations

import sys

# Windows 中文控制台默认 GBK，无法编码 emoji 等字符：强制 UTF-8 输出。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.bangumi_api import BangumiAPI
from src.config import load_token


def main() -> None:
    api = BangumiAPI(load_token())

    me = api.fetch_me()
    username = me.get("username")
    print(f"[1/2] [OK] token 有效，当前用户：{username}")
    if not username:
        raise SystemExit("未取到用户名，检查 token 权限")

    entries = api.fetch_collections(username, state="看过")
    rated = [e for e in entries if e.rate > 0]
    print(f"[2/2] [OK] 拉取《{username}》看过动画 {len(entries)} 条，其中带评分 {len(rated)} 条")

    if entries:
        print("      样例：")
        for e in entries[:5]:
            name = e.subject.get("name_cn") or e.subject.get("name") or e.subject_id
            print(f"        - id={e.subject_id:>6} rate={e.rate:<2} tags={e.tags[:4]} {name}")

    print("\n门禁结论：", "通过" if entries else "待定（该用户无看过动画）")


if __name__ == "__main__":
    main()
