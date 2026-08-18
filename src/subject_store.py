"""条目简介存储（深模块，小接口）：给卡片浮窗返回简介文本，隐藏数据来源链。

背景（2026-08-18 卡片浮窗功能）：推荐卡片点击弹出浮窗，展示动画简介。简介来源三级链：
    ① subjects 表（本地语料已爬取的简介，~74% 条目非空）
    ② subject_summary_cache 表（本服务此前经 Bangumi API 实时拉取并落库的）
    ③ 实时 GET /v0/subjects/{id}（含 summary 字段），成功后落库 + 内存，后续秒回

只在浮窗冷路径（低频）用——subjects 表 PK 查询 <1ms，实时拉取有 deadline 兜底
（Bangumi 宕机/超时快速返回空串，不阻塞整页）。不为 UI 功能给 Recommender._load
塞 30–50MB 中文简介（30,556 条里 ~22,656 非空）。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.bangumi_api import BangumiAPI


class SubjectStore:
    """条目简介缓存 + 拉取兜底。构造注入 api（BangumiAPI），可测试。"""

    def __init__(
        self,
        db_path: str | Path,
        api: BangumiAPI | None = None,
    ):
        self._db_path = str(db_path)
        self._api = api
        self._mem: dict[int, str] = {}  # 本进程会话内缓存（同一条目重复打开秒回）
        self._ensure_table()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA busy_timeout=5000")  # 并发写锁等待（不阻塞读）
        return conn

    def _ensure_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS subject_summary_cache ("
                " subject_id INTEGER PRIMARY KEY,"
                " summary TEXT NOT NULL DEFAULT '',"
                " updated_at TEXT NOT NULL DEFAULT '')"
            )

    def summary(self, subject_id: int, *, deadline: float | None = None) -> str:
        """返回简介文本（空串 = 无可展示）。404 抛 httpx.HTTPStatusError（调用方转 404），
        网络失败/超时抛 RuntimeError（调用方转 502）。"""
        if subject_id in self._mem:
            return self._mem[subject_id]

        # ① 本地 subjects 表（主数据源，~74% 非空）
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT summary FROM subjects WHERE id=?", (subject_id,)
            ).fetchone()
        finally:
            conn.close()
        if row and row[0]:
            self._mem[subject_id] = row[0]
            return row[0]

        # ② 独立缓存表（此前经 API 拉取的）
        hit = self._cache_get(subject_id)
        if hit is not None:
            self._mem[subject_id] = hit
            return hit

        # ③ 实时拉取 → 落库 + 内存
        if self._api is None:
            return ""
        data = self._api.fetch_subject(subject_id, deadline=deadline)
        s = data.get("summary") or ""
        self._cache_set(subject_id, s)
        self._mem[subject_id] = s
        return s

    def _cache_get(self, subject_id: int) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT summary FROM subject_summary_cache WHERE subject_id=?",
                (subject_id,),
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def _cache_set(self, subject_id: int, summary: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO subject_summary_cache"
                    " (subject_id, summary, updated_at) VALUES (?,?,?)",
                    (subject_id, summary, time.strftime("%Y-%m-%d %H:%M:%S")),
                )
        except sqlite3.Error:  # 落库失败不影响本次响应
            pass
