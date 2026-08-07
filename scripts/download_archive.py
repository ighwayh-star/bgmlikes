"""下载 Bangumi Archive dump（多线程分片 + sha256 校验）。

用法（项目根目录）：
    python -m scripts.download_archive
输出：data/archive/dump.zip

策略：每个线程把它的字节区间写入独立 part 文件（避免共享文件指针的竞态），
全部完成后拼接，再用 latest.json 里的 sha256 校验，不符则报错。
单连接被限速时多线程并发可显著提速。
"""
from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "https://github.com/bangumi/Archive/releases/download/archive/dump-2026-07-28.210449Z.zip"
EXPECTED_SHA256 = "d979b9874517895d6353c031ef2970da0f2adc57b764889c20798aafa5a767a3"
OUT = Path("data/archive/dump.zip")
NUM_THREADS = 8
USER_AGENT = "bgmlikes/0.1 (archive downloader)"
RETRIES = 8


def get_total_and_final_url(client: httpx.Client) -> tuple[str, int]:
    """用 Range 请求读回 Content-Range，拿到最终 URL 的真实总字节数。"""
    r = client.get(URL, headers={"Range": "bytes=0-0"}, follow_redirects=True)
    r.raise_for_status()
    cr = r.headers.get("Content-Range") or ""
    total = int(cr.split("/")[-1])
    return str(r.url), total


def download_range(client: httpx.Client, final_url: str, start: int, end: int,
                   part: Path, errors: list[tuple[int, str]]) -> None:
    for attempt in range(RETRIES + 1):
        try:
            with client.stream("GET", final_url, headers={"Range": f"bytes={start}-{end}"}, timeout=120) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for data in r.iter_bytes(chunk_size=1 << 16):
                        f.write(data)
            return
        except Exception as e:  # noqa: BLE001
            errors.append((start, f"attempt{attempt}: {type(e).__name__}: {e}"))
            if attempt < RETRIES:
                time.sleep(2 ** min(attempt, 4))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120) as client:
        final_url, total = get_total_and_final_url(client)
        print(f"总大小 {total/1e6:.0f}MB，{NUM_THREADS} 线程分片下载...")

        chunk = (total + NUM_THREADS - 1) // NUM_THREADS
        parts: list[Path] = []
        errors: list[tuple[int, str]] = []
        threads: list[threading.Thread] = []
        done = [0] * NUM_THREADS

        def worker(idx: int, start: int, end: int) -> None:
            part = OUT.parent / f"dump.part{idx}"
            parts.append(part)
            download_range(client, final_url, start, end, part, errors)
            done[idx] = 1

        t0 = time.time()
        for i in range(NUM_THREADS):
            start = i * chunk
            end = min(start + chunk - 1, total - 1)
            t = threading.Thread(target=worker, args=(i, start, end), daemon=True)
            threads.append(t)
            t.start()

        while any(t.is_alive() for t in threads):
            try:
                n = sum(done)
                downloaded = sum(p.stat().st_size for p in parts if p.exists())
                speed = downloaded / (time.time() - t0) / 1e6
                print(f"\r{n}/{NUM_THREADS} 片完成  {downloaded/1e6:.0f}MB  {speed:.1f}MB/s", end="", flush=True)
            except Exception:  # noqa: BLE001 监控异常绝不拖垮下载
                pass
            time.sleep(2)
        for t in threads:
            t.join()

    if errors:
        print(f"\n失败 {len(errors)} 片：{errors[:3]}")
        for p in parts:
            p.unlink(missing_ok=True)
        raise SystemExit(1)

    # 合并
    with open(OUT, "wb") as out:
        for p in sorted(parts, key=lambda x: int(x.suffix[5:])):
            with open(p, "rb") as f:
                for data in iter(lambda: f.read(1 << 20), b""):
                    out.write(data)
            p.unlink()

    # sha256 校验
    h = hashlib.sha256()
    with open(OUT, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    got = h.hexdigest()
    if got != EXPECTED_SHA256:
        print(f"sha256 不符！\n  got      {got}\n  expected {EXPECTED_SHA256}")
        OUT.unlink(missing_ok=True)
        raise SystemExit(1)

    print(f"\n完成：{OUT} ({OUT.stat().st_size/1e6:.0f}MB) sha256 校验通过")


if __name__ == "__main__":
    main()
