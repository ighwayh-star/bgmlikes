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
