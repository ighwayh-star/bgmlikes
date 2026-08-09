#!/bin/bash
# 启用 systemd 服务 + Caddy 反代（在 setup_server.sh 之后、已 clone 代码 + 填好 .env 后运行）
set -euo pipefail

DOMAIN="${1:-bgmhiway.asia}"
APP_DIR="/opt/bgmlikes"

echo "==> 安装 systemd 服务 bgmlikes.service"
install -o www-data -g www-data -d "$APP_DIR/data"
install -m 644 "$APP_DIR/deploy/bgmlikes.service" /etc/systemd/system/bgmlikes.service
# data/ 里的 sqlite 要让 www-data 可写
chown -R www-data:www-data "$APP_DIR/data"
systemctl daemon-reload
systemctl enable --now bgmlikes
systemctl status bgmlikes --no-pager || true

echo "==> 配置 Caddyfile（$DOMAIN → 127.0.0.1:8000，自动 HTTPS）"
sed "s/bgmhiway\.asia/$DOMAIN/g" "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile
systemctl enable caddy
systemctl restart caddy
systemctl status caddy --no-pager || true

cat <<EOF

────────────────────────────────────────────
完成！ 服务已启动。
  健康检查： curl http://127.0.0.1:8000/v1/health
  公网访问： https://$DOMAIN
  （需 DNS 已把 $DOMAIN 的 A 记录指向服务器公网 IP，并已开 80/443 端口）
  OAuth 回调： https://$DOMAIN/auth/callback
────────────────────────────────────────────
EOF