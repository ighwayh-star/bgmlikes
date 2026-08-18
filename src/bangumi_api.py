"""深模块：Bangumi HTTP 客户端。

对外只暴露两个小接口（fetch_me / fetch_collections），把以下复杂度全部
藏在实现里：
- Bearer token 认证
- 请求节流（对 Bangumi API 保持礼貌）
- 429 / 5xx 退避重试
- offset+limit 分页
- 已知 bug 兜底：`limit > 用户总收藏数` 会返回 400（见 bangumi/server#126）
  策略：先 limit=1 取 total，再按 min(50, total) 分页；分页时若仍遇 400 则降为 30 重试。

领域术语见 docs/CONTEXT.md；state/rate 的语义对齐 CollectionEntry。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# 收藏状态 -> API 的 type 参数（1=想看 2=看过 3=在看 4=搁置 5=抛弃）
_STATE_TO_API_TYPE: dict[str, int] = {
    "想看": 1,
    "看过": 2,
    "在看": 3,
    "搁置": 4,
    "抛弃": 5,
}
_API_TYPE_TO_STATE: dict[int, str] = {v: k for k, v in _STATE_TO_API_TYPE.items()}

# 单页上限。默认 limit=30、上限 100，但存在 limit>total 报 400 的已知问题，
# 我们用 min(limit, total) 规避（先探 total 再分页，limit 永不超过 total），出错再降级到 30。
# 2026-08-05 提速实验：50 → 100（分页请求减半）。
_PAGE_LIMIT = 100
_FALLBACK_PAGE_LIMIT = 30


@dataclass(frozen=True)
class CollectionEntry:
    """用户对某 Subject 的一条收藏记录（领域原子数据单元）。"""

    subject_id: int
    state: str
    rate: int  # 0~10；0 表示未打分，与打了 1 分是不同信号
    tags: list[str]
    comment: str
    updated_at: str
    subject: dict[str, Any] = field(default_factory=dict)


class BangumiAPI:
    BASE_URL = "https://api.bgm.tv"
    # 相邻请求最小间隔（秒）。保持礼貌；若 429 频发可调大。
    # 2026-08-05 提速实验：1.5 → 0.5（配合 _PAGE_LIMIT 100，含 429 自动退避兜底）
    MIN_INTERVAL = 0.5

    def __init__(self, token: str):
        self._token = token
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "bgmlikes/0.1 (anime recommendation prototype)",
            },
            timeout=30,
        )
        self._last_request_at = 0.0

    # ---- 对外小接口 -------------------------------------------------

    def fetch_me(self) -> dict[str, Any]:
        """验证 token 并取回当前用户信息（含 username）。"""
        return self._get("/v0/me")

    def fetch_subject_image(
        self,
        subject_id: int,
        *,
        size: str = "medium",
    ) -> tuple[bytes, str]:
        """取条目封面图（跟随 302 到 lain.bgm.tv），返回 (字节, content-type)。

        size: {small|grid|large|medium|common}。无图条目 API 返回默认占位图
        （no_icon_subject.png），本方法不区分、一律返回字节。
        服务器侧调用：大陆直连 lain.bgm.tv 超时（Cloudflare 海外 CDN），
        封面图经 /img/{id} 由本服务中转给前端（见 src/images.py）。
        """
        self._throttle()
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = self._client.get(
                    f"/v0/subjects/{subject_id}/image",
                    params={"type": size},
                    follow_redirects=True,
                    timeout=30,
                )
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "image/jpeg").split(";")[0]
                return resp.content, ctype
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise  # 条目不存在：调用方兜底
                last_exc = e
            except httpx.TransportError as e:
                last_exc = e
            time.sleep(2**min(attempt, 3))
        raise RuntimeError(f"拉取封面图 {subject_id} 失败") from last_exc

    def fetch_subject(
        self,
        subject_id: int,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        """取条目完整信息（含简介 summary 字段）。

        复用 _get 的节流 / 429 重试 / 退避。404 抛 httpx.HTTPStatusError（调用方转 404）；
        网络失败或超时抛 RuntimeError。浮窗端点必须传 deadline（_get 默认重试 5 次，
        纯 5xx 最坏退避可达数十秒——deadline 约束单次请求与总时长，超时快速转 502）。
        """
        return self._get(f"/v0/subjects/{subject_id}", deadline=deadline)

    def fetch_collections(
        self,
        username: str,
        *,
        subject_type: int = 2,  # 2 = 动画
        state: str | None = None,  # None = 全部状态
        max_seconds: float | None = None,  # 整体超时上限；实时路径用（API 宕机时快速降级到缓存）
    ) -> list[CollectionEntry]:
        """取回某用户的收藏（指定状态或全部），处理分页与 400 兜底。

        max_seconds: 总耗时上限。爬虫批量场景不设（大数据量需要时间）；线上实时路径传小值，
        超时抛 RuntimeError 由 api.py 降级到本地语料缓存，避免宕机时请求卡死重试。
        """
        deadline = time.monotonic() + max_seconds if max_seconds else None
        # 第 1 步：limit=1 探 total，规避 limit>total 的 400 已知问题
        first_params = {"subject_type": subject_type, "limit": 1, "offset": 0}
        if state is not None:
            first_params["type"] = _STATE_TO_API_TYPE[state]
        first = self._get(f"/v0/users/{username}/collections", params=first_params, deadline=deadline)
        total = int(first["total"])
        if total == 0:
            return []

        entries: list[CollectionEntry] = []
        limit = min(_PAGE_LIMIT, total)
        offset = 0
        while offset < total:
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError("拉取收藏超时")
            page_params = {"subject_type": subject_type, "limit": limit, "offset": offset}
            if state is not None:
                page_params["type"] = _STATE_TO_API_TYPE[state]
            page = self._get(f"/v0/users/{username}/collections", params=page_params, deadline=deadline)
            data = page.get("data") or []
            for raw in data:
                entries.append(self._to_entry(raw, state))
            offset += len(data)

        return entries

    # ---- 内部实现 ---------------------------------------------------

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        retries: int = 5,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        self._throttle()
        transport_failures = 0
        for attempt in range(retries + 1):
            if deadline is not None and time.monotonic() > deadline:
                raise RuntimeError(f"GET {path} 超时")
            try:
                req_timeout: float | None = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 1.0:
                        raise RuntimeError(f"GET {path} 超时")
                    req_timeout = remaining  # 单次请求也受 deadline 约束，避免挂满默认 30s
                resp = self._client.get(path, params=params, timeout=req_timeout)
            except httpx.TransportError as e:
                # 网络级失败：设了 deadline（实时路径）时，ConnectError/ConnectTimeout 说明
                # 链路不通（如系统代理宕机），重试无益——立即失败由 api.py 降级到缓存
                if deadline is not None and isinstance(
                    e, (httpx.ConnectError, httpx.ConnectTimeout)
                ):
                    raise RuntimeError(f"GET {path} 网络不可达") from e
                # 代理抖动/读超时：退避重试；连续失败则重置连接池
                transport_failures += 1
                if transport_failures >= 2:
                    self._reset_client()
                time.sleep(2**min(attempt, 4))
                continue
            if resp.status_code == 429:
                retry_after = _safe_float(resp.headers.get("Retry-After"), 10.0)
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if resp.status_code == 400:
                # 已知问题：limit 超过用户总收藏数时返回 400。降级到更小页再试一次。
                if params and int(params.get("limit", 0)) > _FALLBACK_PAGE_LIMIT:
                    params = {**params, "limit": _FALLBACK_PAGE_LIMIT}
                    time.sleep(0.5)
                    continue
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"GET {path} 重试 {retries} 次仍失败")

    def _reset_client(self) -> None:
        """连接池疑似中毒（连续连接失败）时重建客户端。"""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "bgmlikes/0.1 (anime recommendation prototype)",
            },
            timeout=30,
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _to_entry(raw: dict[str, Any], state: str | None = None) -> CollectionEntry:
        api_type = int(raw.get("type") or 0)
        state_name = state or _API_TYPE_TO_STATE.get(api_type, f"unknown_{api_type}")
        return CollectionEntry(
            subject_id=int(raw["subject_id"]),
            state=state_name,
            rate=int(raw.get("rate") or 0),
            tags=list(raw.get("tags") or []),
            comment=raw.get("comment") or "",
            updated_at=raw.get("updated_at") or "",
            subject=raw.get("subject") or {},
        )


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value else default
    except (TypeError, ValueError):
        return default
