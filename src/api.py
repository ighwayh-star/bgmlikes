"""FastAPI 薄层 adapter：只做转发与编排，不写业务逻辑。

删除测试：删掉本模块，复杂度应全部消失（数据层/算法层仍在）。
接口：POST /v1/recommend {username, k} -> {normal}
OAuth 登录：/auth/login · /auth/callback · /auth/me · /auth/logout
偏好（登录后有"不感兴趣"隐藏）：/preferences/hidden
"""
from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
from src.config import load_optional, load_token
from src.dataset import APISource, CacheSource
from src.images import CoverCache
from src.recommender import Recommendation, Recommender

logger = logging.getLogger("bgmlikes")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "collections.db"
AUTH_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "auth.db"
WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"
WEB_HOME = Path(__file__).resolve().parent.parent / "web" / "home.html"
WEB_DAILY = Path(__file__).resolve().parent.parent / "web" / "daily.html"
COVER_DIR = Path(__file__).resolve().parent.parent / "data" / "covers"
PICS_DIR = Path(__file__).resolve().parent.parent / "pics"  # 发帖用截图直链（本地不上 git）

CALENDAR_URL = "https://api.bgm.tv/calendar"
# 每日放送「只看我的」覆盖的收藏状态（"抛弃"不显示）
COLLECTED_STATES = {"在看", "看过", "想看", "搁置"}
# /daily/collected 内存缓存（重用户收藏数百条，拉取数秒；5 分钟 TTL 避免每次页面加载都打 Bangumi）
_COLLECTED_CACHE: dict[int, tuple[float, list[dict]]] = {}
_COLLECTED_TTL = 300


def _cookie_max_age() -> int:
    return 30 * 24 * 3600


def _set_session_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        SESSION_COOKIE, token, max_age=_cookie_max_age(),
        httponly=True, samesite="none",
        # secure 由请求协议决定：https 请求置 secure，http（本机/反代被 Caddy 转 https）
        # 不置——反代到 Caddy 后始终 https。本机 http 调试也不受影响。
        # samesite="none"：让 dailyanimation 扩展/widget（cross-site 源）能携带会话 cookie，
        # 复用站点的 Bangumi 登录态同步隐藏列表。必须与 Secure 同用（https 下满足）。
        secure=secure,
    )


def _clear_session_cookie(response: Response) -> None:
    # samesite 必须与登录时一致（SameSite=None），否则部分浏览器不按此路径清除
    response.delete_cookie(SESSION_COOKIE, samesite="none")


def _session_from_request(request: Request, auth: AuthStore) -> Session | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    return auth.user_for_session(token)


class RecommendRequest(BaseModel):
    username: str
    k: int = 20


class HiddenIn(BaseModel):
    hidden: bool


class ItemOut(BaseModel):
    subject_id: int
    name: str
    score: float
    rating: float = 0        # BGM 平均分（卡片展示）
    popularity_rank: int = 0


class RecommendResponse(BaseModel):
    username: str
    normal: list[ItemOut] = []  # 推荐列表（rank ≤ 阈值 + 自适应 γ 去热的 top-k）
    source: str = "api"  # "api" 实时拉取 / "cache" 本地语料降级（Bangumi 宕机时）


def create_app() -> FastAPI:
    api = BangumiAPI(load_token())
    source = APISource(api)
    cache_source = CacheSource(DB_PATH)  # 降级链：实时 API 失败时用本地语料兜底
    # df_min_rated：去热分母只计重度用户（剔除轻度用户回暖热门），.env 可调，0=全语料
    # rate_center：相似度中心化（rate-5，5 分中界，1-4 负偏好），打分保持原始分，.env 可调，0=全原始分
    # idf_in_score：打分矩阵 A 是否带 idf 乘子（0=去掉，相似度 Bn/q 的 idf 保留），.env 可调
    recommender = Recommender(
        DB_PATH, df_min_rated=int(load_optional("DF_MIN_RATED", "300")),
        rate_center=float(load_optional("RATE_CENTER", "5")),
        idf_in_score=load_optional("IDF_IN_SCORE", "1") == "1",
    )
    cover_cache = CoverCache(COVER_DIR, api)  # 封面图中转（大陆直连 lain.bgm.tv 超时）
    auth = AuthStore(AUTH_DB_PATH)  # 用户登录会话 + "不感兴趣"偏好（独立 auth.db）
    logger.info("recommender loaded: %s", recommender.stats())
    logger.info("oauth logged: %s", oauth_configured())

    app = FastAPI(title="bgmlikes", version="0.1")

    # CORS：只对 dailyanimation widget 的运行源 https://bgm.tv 放行（跨站复用站点登录同步
    # 隐藏列表）。allow_credentials=True 不能配通配 origin；MV3 扩展有 host_permissions 绕过 CORS，
    # 故这里只服务 widget 的 fetch。同源站点 /likes /daily 不受影响。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://bgm.tv"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", **recommender.stats()}

    # ---- 每日放送：Bangumi 放送表代理（服务端取数，避免浏览器 CORS / 大陆直连超时）----
    @app.get("/api/calendar")
    def calendar_proxy() -> Response:
        try:
            resp = httpx.get(
                CALENDAR_URL,
                headers={"User-Agent": "bgmlikes/1.0", "Accept": "application/json"},
                timeout=15,
            )
        except httpx.HTTPError as e:
            logger.exception("拉取 Bangumi 放送表失败：%s", e)
            raise HTTPException(status_code=502, detail=f"获取放送表失败：{e}")
        if resp.status_code != 200:
            raise HTTPException(status_code=502,
                                detail=f"Bangumi 放送表返回 {resp.status_code}")
        return Response(content=resp.content, media_type="application/json",
                        headers={"Cache-Control": "public, max-age=1800"})

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
        resp = RedirectResponse("/likes", status_code=302)
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

    # ---- 每日放送隐藏（需登录；未登录 /daily 页隐藏功能禁用）----
    @app.get("/daily/hidden")
    def get_daily_hidden(request: Request) -> dict:
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        return {"hidden": sorted(auth.get_daily_hidden(sess.user_id))}

    @app.post("/daily/hidden/{subject_id}")
    def set_daily_hidden(request: Request, subject_id: int, body: HiddenIn) -> dict:
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        auth.set_daily_hidden(sess.user_id, subject_id, body.hidden)
        return {"ok": True, "hidden": body.hidden, "subject_id": subject_id}

    @app.delete("/daily/hidden")
    def clear_daily_hidden(request: Request) -> dict:
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        auth.clear_daily_hidden(sess.user_id)
        return {"ok": True}

    # ---- 用户收藏（每日放送「只看我的」筛选的数据源）----
    @app.get("/daily/collected")
    def get_daily_collected(request: Request) -> dict:
        """登录用户的收藏状态（在看/看过/想看/搁置，不含"抛弃"）：以服务器 token 拉 /v0 公开收藏。

        每日放送页拿到 [{id, state}] 后，与本季放送表（calendar）取交集即"我收藏的本季番"，
        卡片上可展示各自的状态标签。
        """
        sess = _session_from_request(request, auth)
        if sess is None:
            raise HTTPException(status_code=401, detail="请先登录")
        cached = _COLLECTED_CACHE.get(sess.user_id)
        if cached and time.time() - cached[0] < _COLLECTED_TTL:
            return {"items": cached[1]}
        try:
            entries = api.fetch_collections(sess.username, subject_type=2, max_seconds=15)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception("拉取收藏列表失败：%s", e)
            raise HTTPException(status_code=502, detail=f"获取收藏列表失败：{e}")
        items = [
            {"id": e.subject_id, "state": e.state}
            for e in entries if e.state in COLLECTED_STATES
        ]
        _COLLECTED_CACHE[sess.user_id] = (time.time(), items)
        return {"items": items}

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest) -> RecommendResponse:
        username = req.username.strip()
        if not username:
            raise HTTPException(status_code=400, detail="username 不能为空")
        if not (1 <= req.k <= 300):
            raise HTTPException(status_code=400, detail="k 须在 1~300 之间")

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
                rating=recommender.subject_meta.get(r.subject_id, {}).get("score", 0.0),
                popularity_rank=r.popularity_rank,
            )

        return RecommendResponse(
            username=username,
            normal=[_out(r) for r in recs.normal],
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
    def home() -> FileResponse:
        # 主页：导航到 推荐系统(/likes) / 每日放送(/daily)
        return FileResponse(WEB_HOME, headers={"Cache-Control": "no-cache"})

    @app.get("/likes")
    def likes() -> FileResponse:
        # 推荐系统页（原 index.html，改 /likes 前缀）
        return FileResponse(WEB_INDEX, headers={"Cache-Control": "no-cache"})

    @app.get("/daily")
    def daily() -> FileResponse:
        # 每日放送页
        return FileResponse(WEB_DAILY, headers={"Cache-Control": "no-cache"})

    @app.get("/pics/{filename}")
    def pics(filename: str) -> FileResponse:
        """发帖用截图直链：/pics/<filename>（只服务 pics/ 目录内文件，防路径穿越）。"""
        target = (PICS_DIR / filename).resolve()
        if PICS_DIR.resolve() not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(target, headers={"Cache-Control": "public, max-age=86400"})

    return app


app = create_app()
