"""UserInteractions seam + 两个真实 adapter。

seam 为什么值得存在：线上 API 与本地 SQLite 缓存是两个真实会切换的实现
（2 个 adapter = 真 seam）。推荐流程只依赖此接口。

MVP：查询路径用 APISource（实时、全状态，正确性优先）；
CacheSource 已实现，作为降级与后续 TTL 缓存优化的 adapter。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Protocol

from src.bangumi_api import BangumiAPI, CollectionEntry


def hash_user(username: str) -> str:
    """存储侧用户名哈希（隐私最小化，见 docs/PLAN.md 风险节）。"""
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


class UserInteractions(Protocol):
    """数据源 seam：给定用户名，返回其全部状态的 CollectionEntry 列表。"""

    def collections_for(self, username: str) -> list[CollectionEntry]: ...


class APISource:
    """adapter A：直接走线上 API（每次即时拉取全部状态）。"""

    def __init__(self, api: BangumiAPI):
        self._api = api

    def collections_for(self, username: str, *, max_seconds: float | None = None) -> list[CollectionEntry]:
        return self._api.fetch_collections(username, state=None, max_seconds=max_seconds)


class CacheSource:
    """adapter B：SQLite 本地缓存（按 username 哈希查找，未命中返回空）。

    当前仅含种子用户"看过"的收藏（爬虫只存了该状态）。
    TTL / 每周增量刷新留待阶段 3。
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)

    def collections_for(self, username: str) -> list[CollectionEntry]:
        uh = hash_user(username)
        conn = sqlite3.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT subject_id, state, rate, tags, updated_at, comment"
                " FROM collections WHERE user_hash = ?",
                (uh,),
            ).fetchall()
        finally:
            conn.close()
        return [
            CollectionEntry(
                subject_id=sid,
                state=state,
                rate=rate,
                tags=json.loads(tags) if tags else [],
                comment=comment,
                updated_at=updated_at,
                subject={},
            )
            for sid, state, rate, tags, updated_at, comment in rows
        ]
