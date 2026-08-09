"""FastAPI 薄层 adapter：只做转发与编排，不写业务逻辑。

删除测试：删掉本模块，复杂度应全部消失（数据层/算法层仍在）。
接口：POST /v1/recommend {username, k} -> {normal, cold}
OAuth 登录：/auth/login · /auth/callback · /auth/me · /auth/logout
偏好（登录后有"不感兴趣"隐藏）：/preferences/hidden
"""
from __future__ import annotations

import logging
import secrets
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel

from src.auth import (
    AuthStore,
    OAuthError,
    SESSION_COOKIE,
    Session,
    authorize_url,
    configured as oauth_configured,
    exchange_code,
    fetch_me,
)
from src.bangumi_api import BangumiAPI
from src.config import load_token
from src.dataset import APISource, CacheSource
from src.images import CoverCache
from src.recommender import Recommendation, Recommender

logger = logging.getLogger("bgmlikes")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "collections.db"
AUTH_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "auth.db"
WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"
COVER_DIR = Path(__file__).resolve().parent.parent / "data" / "covers"


def _cookie_max_age() -> int:
    return 30 * 24 * 3600


def _set_session_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        SESSION_COOKIE, token, max_age=_cookie_max_age(),
        httponly=True, samesite="lax",
        # secure 由请求协议决定：https 请求置 secure，http（本机/反代被 Caddy 转 https）
        # 不置——反代到 Caddy 后始终 https。本机 http 调试也不受影响。
        secure=secure,
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE)


def _session_from_request(request: Request, auth: AuthStore) -> Session | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return auth.user_for_session(token)


class RecommendRequest(BaseModel):
    username: str
    k: int = 20


class ItemOut(BaseModel):
    subject_id: int
    name: str
    score: float
    cold: bool = False
    popularity_rank: int = 0


class RecommendResponse(BaseModel):
    username: str
    normal: list[ItemOut] = []  # 动画推荐区（非冷门池 top-k）
    cold: list[ItemOut] = []    # 冷门发现区（冷门池 top-k）
    source: str = "api"  # "api" 实时拉取 / "cache" 本地语料降级（Bangumi 宕机时）


def create_app() -> FastAPI:
    api = BangumiAPI(load_token())
    source = APISource(api)
    cache_source = CacheSource(DB_PATH)  # 降级链：实时 API 失败时用本地语料兜底
    recommender = Recommender(DB_PATH)
    cover_cache = CoverCache(COVER_DIR, api)  # 封面图中转（大陆直连 lain.bgm.tv 超时）
    auth = AuthStore(AUTH_DB_PATH)  # 用户登录会话 + "不感兴趣"偏好（独立 auth.db）
    logger.info("recommender loaded: %s", recommender.stats())
    logger.info("oauth logged: %s", oauth_configured())

    app = FastAPI(title="bgmlikes", version="0.1")

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", **recommender.stats()}

    # ---- OAuth 登录 ----
    @app.get("/auth/login")
    def auth_login(request: Request):
        """开始登录：生成 state，302 跳 Bangumi 授权页。未配置则 503。"""
        if not oauth_configured():
            raise HTTPException(status_code=503,
                                detail="登录功能未启用（缺少 OAUTH_* 配置，见 docs/DEPLOY-OAUTH.md）")
        state = secrets.token_urlsafe(16)
        resp = RedirectResponse(authorize_url(state), status_code=302)
        # state 存短期 cookie（供 callback 校验 CSRF）
        resp.set_cookie("oauth_state", state, max_age=600, httponly=True, samesite="lax")
        return resp

    @app.get("/auth/callback")
    def auth_callback(request: Request, code: str = "", state: str = ""):
        """OAuth 回调：校验 state → 换 token → 取用户名 → 建登录会话。"""
        if not oauth_configured():
            raise HTTPException(status_code=503,
                                detail="登录功能未启用（缺少 OAUTH_* 配置）")
        oauth_state = request.cookies.get("oauth_state", "")
        if not state or not oauth_state or state != oauth_state:
            raise HTTPException(status_code=400, detail="state 校验失败（请重试登录）")
        try:
            tok = exchange_code(code)
            username = fetch_me(tok.access_token)
        except OAuthError as e:
            logger.warning("OAuth callback 失败：%s", e)
            raise HTTPException(status_code=502, detail=f"登录失败：{e}")
        if not username:
            raise HTTPException(status_code=502, detail="未能获取 Bangumi 用户名")
        auth.upsert_user(tok.user_id, username, tok)
        sess = auth.create_session(tok.user_id)
        resp = RedirectResponse("/", status_code=302)
        _set_session_cookie(resp, sess.token,
                            secure=request.url.scheme == "https")
        resp.delete_cookie("oauth_state")
        return resp

    @app.get("/auth/me")
    def auth_me(request: Request) -> dict:
        """登录态查询：{logged_in, username, user_id}。"""
        sess = _session_from_request(request, auth)
        if sess is None:
            return {"logged_in": False, "username": None, "user_id": None}
        return {"logged_in": True, "username": sess.username, "user_id": sess.user_id}

    @app.post("/auth/logout")
    def auth_logout(request: Request) -> dict:
        sess = _session_from_request(request, auth)
        if sess is not None:
            auth.delete_session(sess.token)
        resp = Response('{"ok":true}', media_type="application/json")
        _clear_session_cookie(resp)
        return resp

    # ---- 偏好（"不感兴趣"隐藏，需登录）----
    @app.get("/preferences/hidden")
    def get_hidden(request: Request) -> dict:
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        return {"hidden": sorted(auth.get_hidden(sess.user_id))}

    class HiddenIn(BaseModel):
        hidden: bool

    @app.post("/preferences/hidden/{subject_id}")
    def set_hidden(request: Request, subject_id: int, body: HiddenIn) -> dict:
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        auth.set_hidden(sess.user_id, subject_id, body.hidden)
        return {"ok": True, "hidden": body.hidden, "subject_id": subject_id}

    @app.delete("/preferences/hidden")
    def clear_hidden(request: Request) -> dict:
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        auth.clear_hidden(sess.user_id)
        return {"ok": True}

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        username = req.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="username 不能为空")
        if not (1 <= req.k <= 100):
            raise HTTPException(status_code=400, detail="k 须在 1~100 之间")

        data_source = "api"
        try:
            entries = source.collections_for(username, max_seconds=15)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"用户不存在：{username}")
            # 非 404 的 API 故障：降级到本地语料（已爬取用户仍可服务）
            entries = cache_source.collections_for(username)
            if not entries:
                raise HTTPException(status_code=502, detail=f"Bangumi API 错误：{e.response.status_code}")
            data_source = "cache"
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("拉取收藏失败（尝试本地降级）：%s", username)
            entries = cache_source.collections_for(username)
            if not entries:
                raise HTTPException(status_code=502, detail="Bangumi API 暂时不可用，请稍后重试")
            data_source = "cache"

        already = {e.subject_id for e in entries}
        profile = [e for e in entries if e.state == "看过" and e.rate > 0]

        recs = recommender.recommend(profile, already, k=req.k)

        def _out(r: Recommendation) -> ItemOut:
            return ItemOut(
                subject_id=r.subject_id,
                name=r.name,
                score=r.score,
                cold=r.cold,
                popularity_rank=r.popularity_rank,
            )

        return RecommendResponse(
            username=username,
            normal=[_out(r) for r in recs.normal],
            cold=[_out(r) for r in recs.cold],
            source=data_source,
        )

    @app.get("/img/{subject_id}")
    def subject_image(subject_id: int) -> Response:
        """封面图中转：客户端经本服务加载 lain.bgm.tv 图片（大陆直连超时）。

        首次请求拉取并落盘，之后走磁盘缓存（data/covers/）。拉取失败返回 404，
        前端 onerror 兜底（去掉图，不影响列表）。
        """
        result = cover_cache.get(subject_id)
        if result is None:
            raise HTTPException(status_code=404, detail="封面图暂不可用")
        data, ctype = result
        return Response(
            content=data,
            media_type=ctype,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/")
    def index() -> FileResponse:
        # no-cache：页面每次重新校验，避免浏览器缓存旧 HTML（前端字段会随 API 演进，
        # 缓存过期页引用已删字段会崩出 "reading 'replace'" 类错误，2026-08-07 实测）。
        return FileResponse(WEB_INDEX, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
