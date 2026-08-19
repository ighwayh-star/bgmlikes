"""配置加载：从项目根目录 .env 读取 BGM_TOKEN。

刻意不引入 python-dotenv——几行代码即可，少一个依赖。
"""
from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # 不覆盖已存在的环境变量（如 CI 里注入的）
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_token() -> str:
    """返回 BGM_TOKEN；未设置时给出明确指引。"""
    _load_env_file()
    token = os.environ.get("BGM_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "BGM_TOKEN 未设置：请把 .env.example 复制为项目根目录 .env，"
            "并写入 BGM_TOKEN=<你的令牌>。令牌在 https://next.bgm.tv/demo/access-token 生成。"
        )
    return token


def load_optional(key: str, default: str = "") -> str:
    """读取可选配置项（OAuth client/secret/session 等）；未设置返回 default，不报错。

    这样 OAuth 未配置时服务能正常启动，仅登录功能显示"未启用"。
    """
    _load_env_file()
    return os.environ.get(key, default).strip()


def oauth_configured() -> bool:
    """OAuth 是否已配置（client_id/secret/redirect 三件齐才算可用）。"""
    return bool(
        load_optional("OAUTH_CLIENT_ID")
        and load_optional("OAUTH_CLIENT_SECRET")
        and load_optional("OAUTH_REDIRECT_URI")
    )


# 站点级永久屏蔽（subject_id 集合）：任何发现入口（推荐/每日放送/热门/搜索/相似动画/详情）
# 一律不再出现。《我的英雄学院》全系列 24 条（TV 1-7 季 + 最终季 + Memories + Vigilante 外传
# + 剧场版/OVA/特别篇/其他平台），2026-08-19 站点政策决定。
# .env SITE_BLOCKLIST 可追加更多（逗号分隔，自动并入）。
DEFAULT_BLOCKED: frozenset[int] = frozenset({
    # TV 季（platform 1）
    150955, 185761, 226677, 262162, 303399, 236657, 425587,  # 第一~七季
    518413,                                        # FINAL SEASON（2025-10）
    488960,                                        # Memories
    529995, 567417,                                # 正义使者(非法英雄) 外传 S1/S2
    644516,                                        # No.170＋1『More』
    # OAD / OVA / 剧场版总集篇特别篇（platform 2）
    190704, 339266, 299532,
    # 剧场版（platform 3）
    231647, 278429, 321117, 449154, 467909, 536626,
    # 其他平台（5/0）
    381212, 386475, 646464,
})


def load_blocklist() -> set[int]:
    """返回站点屏蔽 subject_id 集合（默认全系列 + .env SITE_BLOCKLIST 追加）。

    推荐/每日放送/热门/搜索/相似/详情共用一份；为空=不屏蔽。
    """
    out = set(DEFAULT_BLOCKED)
    raw = load_optional("SITE_BLOCKLIST", "")
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.add(int(part))
    return out
