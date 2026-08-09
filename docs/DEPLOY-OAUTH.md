# 部署到云服务器 + Bangumi OAuth 登录注册指引

本篇是「把 bgmlikes 部署到公网服务器、启用 Bangumi 账号登录」的**人类操作手册**。
Claude Code 负责写代码/部署命令，但以下步骤需要你动手：买服务器、买域名、注册 Bangumi 应用、填 `.env`。
边做边把进度告诉我，我接手装服务和配置。

---

## 〇、总览（先读完再动手）

```
买服务器（东京，Linux,2G+）  ──►  拿到公网 IP + root/密钥
   │
       买海外域名            ──►  A 记录解析 ──► 指到服务器 IP
   │
       安装运行环境          ──►  Python + bgmlikes + Caddy 反代 + 自动 HTTPS
   │
       注册 Bangumi 应用     ──►  client_id / client_secret
   │
       填 .env              ──►  OAUTH_CLIENT_ID / SECRET / REDIRECT_URI / SESSION_SECRET
   │
       部署后真机验证        ──►  登录 → 隐藏 → 刷新 → 恢复
```

四个"我必须动手"点：**买服务器 / 买域名 / 注册应用 / 填 .env**。
其余（装环境、配 Caddy、跑服务、OAuth 代码）Claude 来。

---

## 一、购买云服务器

### 选购要点（与项目匹配）

| 项 | 建议 | 原因 |
|---|---|---|
| 来源/镜像 | **基于操作系统镜像**（纯净 Linux） | 项目自己装 Python，不要应用模板/面板 |
| 操作系统 | **Ubuntu 22.04 LTS** 或 **Debian 12** | 文档多、包新、Stable |
| 地域 | **海外**（东京让拉 Bangumi 顺畅） | 大陆直连 api.bgm.tv / 封面 CDN 超时 |
| 内存 | **2GB 起，推荐 4GB** | 模型加载 0.2-0.4G，留余量 |
| CPU | 2 核 | 够 |
| 系统盘 | 40GB | 富余 |
| 带宽 | ≥3Mbps | 中转封面图给用户 |
| 登录 | **密钥对 SSH**（别用弱密码） | 对外服务，安全底子 |

> **别选**：Windows 系统、WordPress/Typecho/Halo/宝塔-应用模板。面板（宝塔/1Panel）非必需，纯净系统 + Claude 部署最干净。

### 购买页几个问题的回答
- **域名解析输入框**：**留空**（你现在还没域名，之后在 DNS 平台加 A 记录即可，随时能做）。
- **登录方式**：**密钥对**。私钥 `.pem` 保存好，别提交、别外传。

### 拿到手的东西
- 公网 IP
- root 登录信息（密钥 `.pem` 或密码）
- （可选）腾讯云控制台能看 SSH 命令样例

---

## 二、购买域名（海外，不用备案）

你的服务器在东京，**海外服务器不受国内 ICP 备案约束**，用海外域名即可。

| 注册商 | 说明 | 推荐 |
|---|---|---|
| **Cloudflare Registrar** | 成本价、自带 DNS 托管、和 Caddy/HTTPS 契合 | ⭐ 首选 |
| **Namecheap** | 大厂便宜 | ⭐ |
| Dynadot / Porkbun | 价格好 | 可 |

- 域名后缀：`.com` / `.dev` / `.io` 等皆可。
- **别买国内备案域名**配海外服务器——不对口、易出问题。

买完 **DNS 解析**：在注册商/Cloudflare 的 DNS 面板，加一条 **A 记录**：`你的域名` → **服务器公网 IP**。
（Cloudflare 若开 "代理/橙色云朵" 会额外走 CDN，Caddy 需要看真实源站 IP；建议先用"仅 DNS（灰色云朵）"或确认是源站直连，避免 HTTPS 证书回源问题。=== 部署时 Claude 会帮你核对。）

---

## 三、安装运行环境 + 部署 bgmlikes（Claude 接手）

服务器可 SSH 登录后，告诉我公网 IP + 登录方式，我会在你的服务器上执行（大致流程）：

1. 安装 Python 3.11+、git、Caddy（自动 HTTPS）
2. 克隆 `ighwayh-star/bgmlikes` 到服务器
3. `pip install -r requirements.txt`
4. 导入/迁移数据（Archive → subjects、import_relations、build_rated_table）
5. 配置 Caddyfile：`你的域名` 反代到 `127.0.0.1:8000`，自动签发 HTTPS
6. 用 systemd 或后台方式跑 uvicorn，配看门狗/健康检查

> 数据 `collections.db`（3.8GB）迁移到服务器需要时间/带宽，到时确认来源：可重新爬，或你提供 dump。

---

## 四、注册 Bangumi 第三方应用（拿 OAuth 凭证）

> ⚠️ Bangumi 开发者平台入口可能随官方改版变动；如链接失效，从 bgm.tv 首页找"开发者/API"入口。以下流程基于官方 OAuth 2.0 授权码模式。

1. 打开 Bangumi 开发者/应用管理（参考 https://bgm.tv/dev 附近入口）。
2. **新建第三方应用（Web 应用 / OAuth 应用）**。
3. 填应用的：
   - **名称**：如 `bgmlikes`
   - **回调地址（redirect_uri / 回调 URL）**：填你服务器的 HTTPS 回调，例如
     `https://你的域名/auth/callback`
     （**必须**和 `.env` 里的 `OAUTH_REDIRECT_URI` 完全一致）
4. 创建后拿到两个值：
   - **client_id**（App ID）
   - **client_secret**（App 密钥）
5. 记下这两个值，下一步填进 `.env`。

> 授权流程（供理解，不用你做）：用户点登录 → 跳 bgm.tv 授权页 → 同意 → 重定向回调带 `code` → 服务器用 code 换 access_token，取到该用户的 Bangumi 用户名 → 建立本站登录会话。

---

## 五、填 `.env`（服务端配置）

在**服务器项目根目录**的 `.env` 里新增以下几行（`BGM_TOKEN` 保留不变）：

```env
# 现有主账号 token（保留）
BGM_TOKEN=<你的主账号 PAT>

# —— Bangumi OAuth 登录（新增，来自第四节注册的应用）——
OAUTH_CLIENT_ID=<client_id>
OAUTH_CLIENT_SECRET=<client_secret>
OAUTH_REDIRECT_URI=https://你的域名/auth/callback
# 会话签名密钥：随便长随机串，务必保密、别提交
SESSION_SECRET=<随机长串，可用 openssl rand -base64 48 生成>
```

**安全**：`.env` 含密钥，已被 `.gitignore` 排除，**不要提交/外传**。

> 部署前这些字段留空也能启动（登录功能显示"未启用"），填上才启用真实登录。

---

## 六、部署后真机验证清单

浏览器打开 `https://你的域名`：

- [ ] 未登录：能看任意用户推荐；点某条 `✕` 提示"请先登录"，不修改任何内容
- [ ] 点「登录」→ 跳转到 bangumi 授权页 → 登录并同意 → 跳回
- [ ] 已登录：顶部显示用户名 + 「退出」
- [ ] 已登录：点某条 `✕` → 该条隐藏、下一位补位、meta 显示「已隐藏 N 条」
- [ ] 刷新页面 → 隐藏仍在（已跨浏览器/设备生效，存服务端）
- [ ] 展开「已隐藏」→ 单项恢复 / 清空全部 正常
- [ ] 点「退出」→ 回到未登录，但之前隐藏的列表已不显示（偏好绑定登录账号）
- [ ] 导出 JPG 正常（跟随当前可见列表）

---

## 七、常见问题

**Q：登录后想看的不是自己？**
本产品设计：登录后可查看任意用户名，但「不感兴趣」隐藏始终绑定**你的登录账号**。

**Q：token / 会话多久过期？**
OAuth access_token ~7 天，refresh_token 续期；本站登录会话由 `SESSION_SECRET` 签名、有过期时间。过期需重新登录。

**Q：服务器内存只 2GB 会卡吗？**
模型加载约 0.2-0.4G，够用；别在服务器上跑 SD-webui 等重进程即可。

**Q：一定要海外服务器才能访问 Bangumi 吗？**
对本服务是：服务器要在海外（大陆直连 api.bgm.tv/封面 CDN 超时）。你的东京节点满足。

---

## 关联

- 代码实现：`src/auth.py`、`src/api.py`（`/auth/*`、`/preferences/*`）、`src/config.py`、`web/index.html`
- 决策背景：`docs/adr/0007-bangumi-oauth-login.md`
- 术语/架构：`docs/CONTEXT.md`、`docs/PLAN.md`