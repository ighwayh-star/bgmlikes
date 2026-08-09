#!/bin/bash
# bgmlikes 服务器一键部署脚本（Ubuntu/Debian）
# 在服务器上以 root 运行：bash deploy/setup_server.sh
set -euo pipefail

DOMAIN="${1:-bgmhiway.asia}"      # 你的域名，可传参覆盖
APP_DIR="/opt/bgmlikes"
PY="python3"

echo "==> 安装系统依赖（Python / git / Caddy）"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends python3 python3-venv python3-pip git curl

# 安装 Caddy（官方脚本，自动 HTTPS）
if ! command -v caddy >/dev/null 2>&1; then
  curl -fsSL https://getcaddy.com | bash -s personal
fi

echo "==> 创建目录 $APP_DIR"
mkdir -p "$APP_DIR/data" "$APP_DIR/web" "$APP_DIR/src"

echo "==> 创建 Python 虚拟环境"
cd "$APP_DIR"
$PY -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r "$APP_DIR/requirements.txt"

echo "==> 下一步（人工）"
cat <<'EOF'

────────────────────────────────────────────────
部署剩余步骤（需要你动手，因为涉及密钥/数据）：
  1) 代码：git clone 到 $APP_DIR
  2) .env：把 BGM_TOKEN / OAUTH_CLIENT_ID / OAUTH_CLIENT_SECRET
     / OAUTH_REDIRECT_URI / SESSION_SECRET 填进 $APP_DIR/.env
     （.env 已 gitignore，别提交）
  3) 数据：把本地 3.8GB collections.db 迁移到 $APP_DIR/data/
     （scp 或重新跑导入脚本）
  4) 建 systemd 服务：把 deploy/bgmlikes.service 启用并启动
  5) Caddy：把域名解析到本机 IP（腾讯云 DNS），装好 Caddyfile，
     systemctl 启动，自动签发 HTTPS

你也可以直接跑下方的 make_services.sh 来装 systemd + Caddy。
────────────────────────────────────────────────
EOF