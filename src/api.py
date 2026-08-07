"""FastAPI 薄层 adapter：只做转发与编排，不写业务逻辑。

删除测试：删掉本模块，复杂度应全部消失（数据层/算法层仍在）。
接口：POST /v1/recommend {username, k} -> {recommendations}
"""
from __future__ import annotations

import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from src.bangumi_api import BangumiAPI
from src.config import load_token
from src.dataset import APISource, CacheSource
from src.images import CoverCache
from src.recommender import Recommendation, Recommender

logger = logging.getLogger("bgmlikes")
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "collections.db"
WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"
COVER_DIR = Path(__file__).resolve().parent.parent / "data" / "covers"


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
    count: int
    recommendations: list[ItemOut]
    source: str = "api"  # "api" 实时拉取 / "cache" 本地语料降级（Bangumi 宕机时）


def create_app() -> FastAPI:
    api = BangumiAPI(load_token())
    source = APISource(api)
    cache_source = CacheSource(DB_PATH)  # 降级链：实时 API 失败时用本地语料兜底
    recommender = Recommender(DB_PATH)
    cover_cache = CoverCache(COVER_DIR, api)  # 封面图中转（大陆直连 lain.bgm.tv 超时）
    logger.info("recommender loaded: %s", recommender.stats())

    app = FastAPI(title="bgmlikes", version="0.1")

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", **recommender.stats()}

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
        out = [
            ItemOut(
                subject_id=r.subject_id,
                name=r.name,
                score=r.score,
                cold=r.cold,
                popularity_rank=r.popularity_rank,
            )
            for r in recs
        ]
        return RecommendResponse(
            username=username, count=len(out), recommendations=out, source=data_source
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
