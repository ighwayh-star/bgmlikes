# 部署到公网服务器

一键部署 bgmlikes 到东京 Ubuntu/Debian，启用 HTTPS + Bangumi OAuth 登录。

## 前提

- 服务器已开通，能 SSH 登录（root 或有 sudo）
- 域名 `bgmhiway.asia` 的 **A 记录已指向服务器公网 IP**，且已开 80/443 端口
- 本地已有项目代码（含 data/ 语料）

## 文件

| 文件 | 用途 |
|---|---|
| `setup_server.sh` | 装 Python/Caddy/venv/依赖（服务器端，root 跑） |
| `bgmlikes.service` | systemd 守护：uvicorn 绑 127.0.0.1:8000，开机自启、崩溃重启 |
| `Caddyfile` | 反向代理 + 自动 HTTPS（域名 → 127.0.0.1:8000） |
| `make_services.sh` | 装 systemd 服务 + 配 Caddy 并启动（代码 clone 好、.env 填好后跑） |
| `.env.server.example` | 服务器 `.env` 模板（含 OAuth 凭证位） |

## 步骤

1. **代码上服务器**：`git clone <你的仓库>` 到 `/opt/bgmlikes`（含 `deploy/`、`src/`、`web/`、`requirements.txt`）。
2. **填 `.env`**：`cp deploy/.env.server.example /opt/bgmlikes/.env`，填入：
   - `BGM_TOKEN`（主账号 PAT）
   - `OAUTH_CLIENT_ID` / `OAUTH_CLIENT_SECRET`（刚注册的应用）
   - `OAUTH_REDIRECT_URI=https://bgmhiway.asia/auth/callback`（与注册一致）
   - `SESSION_SECRET`（`openssl rand -base64 48`）
3. **数据**：把本地 `data/collections.db` 迁到服务器 `data/`（scp，3.8GB 需些时间）。
4. **装环境**：`bash deploy/setup_server.sh`（root）。
5. **起服务**：`bash deploy/make_services.sh`。

> 也可在服务器上重新爬/导入数据，省去 3.8GB 传输——见 scripts/。

## 验证

- `curl http://127.0.0.1:8000/v1/health`
- 浏览器开 `https://bgmhiway.asia` → 登录 → 隐藏 → 刷新 → 恢复
- OAuth 回调 `https://bgmhiway.asia/auth/callback`（注册时的回调地址）

## 安全

- `.env` 含密钥，已在 `.gitignore`，**别提交/外传**。
- `data/`（auth.db 含登录会话、collections.db）也在 gitignore。
- 部署完成后**务必轮换**曾在聊天里出现过的服务器密码/密钥。