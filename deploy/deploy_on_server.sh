#!/bin/bash
# bgmlikes 服务器部署（在 /opt/bgmlikes 下运行）
# 用法：bash deploy_on_server.sh
# 前提：/opt/bgmlikes 里已有 代码（src/ web/ scripts/ requirements.txt）+ deploy/，且 data/collections.db 已就位
set -euo pipefail

cd /opt/bgmlikes
echo "==> 当前目录: $(pwd)"

# ---- 1. 建 Python 虚拟环境 + 装依赖 ----
if [ ! -d venv ]; then
  echo "==> 创建 venv"
  python3 -m venv venv
fi
echo "==> 安装依赖"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# ---- 2. 检查 .env 是否已填（至少要有 BGM_TOKEN）----
if [ ! -f .env ]; then
  echo "!! .env 不存在。请先创建，参考 .env.example 填入："
  echo "   BGM_TOKEN / OAUTH_CLIENT_ID / OAUTH_CLIENT_SECRET / OAUTH_REDIRECT_URI / SESSION_SECRET"
  echo "   创建后再运行本脚本。"
  exit 1
fi
grep -q '^BGM_TOKEN=.\+' .env || { echo "!! .env 里 BGM_TOKEN 为空"; exit 1; }

# ---- 3. 静态编译检查 ----
echo "==> 编译检查"
./venv/bin/python -m compileall -q src || true

echo "==> 部署前置完成。下一步用 make_services.sh 建服务+反代"
echo "   请先确认 data/collections.db 已存在:"
ls -lh data/collections.db 2>&1 || echo "  !! 缺少 collections.db"