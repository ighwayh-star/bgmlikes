"""Bangumi OAuth 登录 + 用户偏好存储 深模块。

小接口，多实现：
- 授权跳转 / code 换 token / /v0/me 取用户名 / refresh 续期
- 登录会话（httponly cookie 签名，存 auth.db sessions）
- 用户"不感兴趣"偏好（隐藏列表）服务端存储，绑定登录账号

数据存独立 data/auth.db（WAL），不污染推荐语料库 collections.db。
OAuth 未配置（client_id 缺）时，各方法由调用方判断 `configured` 决定是否可用。
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import load_optional

AUTH_HOST = "https://bgm.tv"
API_HOST = "https://api.bgm.tv"
USER_AGENT = "bgmlikes/0.1 (anime recommendation)"
SESSION_COOKIE = "bgmlikes_session"
SESSION_TTL = 30 * 24 * 3600        # 30 天
REFRESH_TTL = 30 * 24 * 3600        # refresh 也按 30 天


@dataclass
class OAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: float   # epoch 秒
    user_id: int


class OAuthError(Exception):
    pass


def _client_id() -> str:
    return load_optional("OAUTH_CLIENT_ID")


def configured() -> bool:
    return bool(_client_id() and load_optional("OAUTH_CLIENT_SECRET"))


def authorize_url(state: str) -> str:
    """构造 Bangumi 授权页 URL（用户点登录 → 跳这里）。"""
    from urllib.parse import urlencode
    redirect = load_optional("OAUTH_REDIRECT_URI")
    return (
        f"{AUTH_HOST}/oauth/authorize?"
        + urlencode({
            "client_id": _client_id(),
            "response_type": "code",
            "redirect_uri": redirect,
            "state": state,
        })
    )


def exchange_code(code: str) -> OAuthTokens:
    """用授权码换 access_token/refresh_token，并返回 user_id。"""
    if not configured():
        raise OAuthError("OAuth 未配置：缺少 OAUTH_CLIENT_ID/SECRET")
    data = {
        "grant_type": "authorization_code",
        "client_id": _client_id(),
        "client_secret": load_optional("OAUTH_CLIENT_SECRET"),
        "code": code,
        "redirect_uri": load_optional("OAUTH_REDIRECT_URI"),
    }
    with httpx.Client(timeout=15) as c:
        r = c.post(f"{AUTH_HOST}/oauth/access_token", data=data,
                   headers={"User-Agent": USER_AGENT})
    if r.status_code != 200:
        raise OAuthError(f"token 交换失败：HTTP {r.status_code} {r.text[:200]}")
    j = r.json()
    # refresh 可能与 access 相同或为空，取到即用
    return OAuthTokens(
        access_token=j.get("access_token", ""),
        refresh_token=j.get("refresh_token", j.get("access_token", "")),
        expires_at=time.time() + int(j.get("expires_in", 604800)),
        user_id=int(j.get("user_id", 0)),
    )


def fetch_me(access_token: str) -> str:
    """用 OAuth access_token 调 /v0/me，返回用户名（用于后续查询）。"""
    with httpx.Client(timeout=15) as c:
        r = c.get(
            f"{API_HOST}/v0/me",
            headers={"Authorization": f"Bearer {access_token}",
                     "User-Agent": USER_AGENT},
        )
    if r.status_code != 200:
        raise OAuthError(f"/v0/me 失败：HTTP {r.status_code}")
    return str(r.json().get("username", ""))


def _sign(value: str, secret: str) -> str:
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()


@dataclass
class Session:
    token: str
    user_id: int
    username: str


class AuthStore:
    """auth.db：users / sessions / preferences。WAL，线程安全。"""

    def __init__(self, db_path: str | Path):
        self._db = str(db_path)
        conn = sqlite3.connect(self._db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                access_token TEXT,
                refresh_token TEXT,
                token_expires_at REAL,
                updated_at REAL
            );
            CREATE TABLE IF NOT EXISTS sessions(
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at REAL,
                expires_at REAL
            );
            CREATE TABLE IF NOT EXISTS preferences(
                user_id INTEGER NOT NULL,
                subject_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                updated_at REAL,
                PRIMARY KEY(user_id, subject_id, action)
            );
            """
        )
        conn.commit()
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, timeout=10)

    # ---- users ----
    def upsert_user(self, user_id: int, username: str, tok: OAuthTokens | None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO users(user_id, username, access_token, refresh_token,"
                " token_expires_at, updated_at) VALUES(?,?,?,?,?,?)"
                " ON CONFLICT(user_id) DO UPDATE SET username=excluded.username,"
                " access_token=excluded.access_token,"
                " refresh_token=excluded.refresh_token,"
                " token_expires_at=excluded.token_expires_at,"
                " updated_at=excluded.updated_at",
                (user_id, username,
                 tok.access_token if tok else None,
                 tok.refresh_token if tok else None,
                 tok.expires_at if tok else None,
                 time.time()),
            )

    def get_user(self, user_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, username, access_token, refresh_token, token_expires_at"
                " FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        return ({"user_id": row[0], "username": row[1], "access_token": row[2],
                 "refresh_token": row[3], "token_expires_at": row[4]}
                if row else None)

    # ---- sessions ----
    def create_session(self, user_id: int) -> Session:
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions(token, user_id, created_at, expires_at)"
                " VALUES(?,?,?,?)",
                (token, user_id, now, now + SESSION_TTL),
            )
        return Session(token=token, user_id=user_id, username="")

    def user_for_session(self, token: str) -> Session | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT s.token, s.user_id, u.username FROM sessions s"
                " JOIN users u ON u.user_id=s.user_id"
                " WHERE s.token=? AND s.expires_at>?",
                (token, time.time()),
            ).fetchone()
        return Session(token=row[0], user_id=row[1], username=row[2]) if row else None

    def delete_session(self, token: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))

    # ---- preferences ----
    def set_hidden(self, user_id: int, subject_id: int, hidden: bool) -> None:
        with self._conn() as conn:
            if hidden:
                conn.execute(
                    "INSERT INTO preferences(user_id, subject_id, action, updated_at)"
                    " VALUES(?,?,?,?) ON CONFLICT DO NOTHING",
                    (user_id, subject_id, "hidden", time.time()),
                )
            else:
                conn.execute(
                    "DELETE FROM preferences WHERE user_id=? AND subject_id=?"
                    " AND action='hidden'", (user_id, subject_id),
                )

    def get_hidden(self, user_id: int) -> set[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject_id FROM preferences WHERE user_id=? AND action='hidden'",
                (user_id,),
            ).fetchall()
        return {r[0] for r in rows}

    def clear_hidden(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM preferences WHERE user_id=? AND action='hidden'",
                (user_id,),
            )

    # ---- 每日放送隐藏（同 preferences 表，action='daily_hidden'）----
    def set_daily_hidden(self, user_id: int, subject_id: int, hidden: bool) -> None:
        with self._conn() as conn:
            if hidden:
                conn.execute(
                    "INSERT INTO preferences(user_id, subject_id, action, updated_at)"
                    " VALUES(?,?,?,?) ON CONFLICT DO NOTHING",
                    (user_id, subject_id, "daily_hidden", time.time()),
                )
            else:
                conn.execute(
                    "DELETE FROM preferences WHERE user_id=? AND subject_id=?"
                    " AND action='daily_hidden'", (user_id, subject_id),
                )

    def get_daily_hidden(self, user_id: int) -> set[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT subject_id FROM preferences WHERE user_id=? AND action='daily_hidden'",
                (user_id,),
            ).fetchall()
        return {r[0] for r in rows}