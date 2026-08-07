"""看门狗：轮询 API 健康，进程死了自动重启，常驻运行。

背景（2026-08-07）：本机内存被常驻进程（SD-webui 等）挤压，API 服务器已多次被
压死且无 traceback（日志戛然而止）。每小时 cron 依赖会话，会话断就没人盯了。
此脚本独立 DETACHED 运行，不依赖会话，每 INTERVAL 秒健康检查，挂了用同命令重启。

注意：2026-08-07 上午把原 07:30 自我退出移除——那是夜间巡检的临时窗口限制；
正常运营期服务器整天都要用，看门狗应持续保活。

用法：
    python -m scripts.watch_server        # 前台（调试）
    # 独立后台部署见 README 或直接 DETACHED 启动
"""
from __future__ import annotations

import datetime
import subprocess
import sys
import time
import urllib.request

HEALTH = "http://127.0.0.1:8000/v1/health"
CWD = r"D:\PROJECTS\bgmlikes"
LOG = r"D:\PROJECTS\bgmlikes\data\api_server.log"
INTERVAL = 90  # 秒


def alive() -> bool:
    # 绕过系统代理（本机代理对 127.0.0.1 返回 502，见 docs/adr 与 memory）
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(HEALTH, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def is_listening() -> bool:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    return any("127.0.0.1:8000" in line and "LISTENING" in line for line in out.splitlines())


def start() -> None:
    logf = open(LOG, "ab", buffering=0)
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=CWD,
        stdout=logf, stderr=subprocess.STDOUT,
        creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    print(f"[watch] {datetime.datetime.now().strftime('%H:%M:%S')} 启动 server pid {p.pid}", flush=True)


def main() -> None:
    print(f"[watch] 看门狗启动，间隔 {INTERVAL}s，常驻保活", flush=True)
    while True:
        now = datetime.datetime.now().time()
        if not alive():
            if not is_listening():
                print(f"[watch] {now.strftime('%H:%M:%S')} 服务器无监听 → 重启", flush=True)
                start()
            else:
                print(f"[watch] {now.strftime('%H:%M:%S')} 端口在听但健康检查失败（可能加载中，跳过）", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
