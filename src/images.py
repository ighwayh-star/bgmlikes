"""封面图中转（深模块，小接口）：给客户端返回封面图字节，隐藏 lain.bgm.tv。

背景（2026-08-07 实测 + 文档）：lain.bgm.tv 是 Cloudflare 托管的海外 CDN，
大陆网络直连超时（bgm.tv 全站同源）。前端若直接 <img src="https://lain.bgm.tv/...">
大陆用户会看到裂图。所以封面图统一经本服务 /img/{id} 中转：

    磁盘缓存命中 → 直接返回
    未命中      → 经 GET /v0/subjects/{id}/image（302 到 lain.bgm.tv）拉取字节后落盘

无图条目 API 返回默认占位图（no_icon_subject.png），无需特殊处理。
缓存目录 data/covers/，随 data/ 一起被 gitignore，不上库。

缓存策略：只缓存"被看过"的图，新站自暖；冷图首次请求带 0.5s 礼貌节流。
"""
from __future__ import annotations

from pathlib import Path

from src.bangumi_api import BangumiAPI

_EXT_BY_TYPE = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}
_TYPE_BY_EXT = {v: k for k, v in _EXT_BY_TYPE.items()}


class CoverCache:
    """封面图磁盘缓存 + 拉取兜底。构造注入 api（BangumiAPI），可测试。"""

    def __init__(
        self,
        cache_dir: str | Path,
        api: BangumiAPI | None = None,
        *,
        size: str = "medium",
    ):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._api = api
        self._size = size

    def get(self, subject_id: int) -> tuple[bytes, str] | None:
        """返回 (图片字节, content-type)；未缓存且拉取失败返回 None（前端 onerror 兜底）。"""
        hit = self._read(subject_id)
        if hit is not None:
            return hit
        if self._api is None:
            return None
        try:
            data, ctype = self._api.fetch_subject_image(subject_id, size=self._size)
        except Exception:  # noqa: BLE001 —— 封面图失败不应拖垮推荐主流程
            return None
        self._write(subject_id, data, ctype)
        return data, ctype

    def _read(self, subject_id: int) -> tuple[bytes, str] | None:
        for p in self._dir.glob(f"{subject_id}.*"):
            if p.is_file():
                return p.read_bytes(), _TYPE_BY_EXT.get(p.suffix.lstrip("."), "image/jpeg")
        return None

    def _write(self, subject_id: int, data: bytes, ctype: str) -> None:
        ext = _EXT_BY_TYPE.get(ctype, "jpg")
        try:
            (self._dir / f"{subject_id}.{ext}").write_bytes(data)
        except OSError:  # 磁盘满等：缓存失败不影响本次响应
            pass
