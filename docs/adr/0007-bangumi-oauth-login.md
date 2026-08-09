# ADR 0007：Bangumi OAuth 登录 + 服务端"不感兴趣"偏好

日期：2026-08-09

## 背景

用户要求："只有登录了自己的账号后才能修改自己的喜好信息"。现状：

- "不感兴趣"隐藏列表存浏览器 localStorage，无需登录即可改，且不跨设备。
- 推荐查询是"输入任意用户名 → 看推荐"，无任何登录/session 体系。

目标：登录后可查看任意用户名，但**修改（隐藏）偏好必须绑定登录账号、存服务端、跨设备生效**。

## 决策

1. **登录方式**：Bangumi 官方 OAuth 2.0 授权码流程。
   - `GET /auth/login` 生成 state（存 cookie）→ 302 跳 `bgm.tv/oauth/authorize`。
   - `GET /auth/callback?code&state` 校验 state → POST `bgm.tv/oauth/access_token` 换 token → `GET api.bgm.tv/v0/me` 取用户名 → 建本站会话。
   - access_token ~7 天，refresh 续期（存 users 表）。

2. **隐藏偏好存服务端**：不再用 localStorage。未登录点 `✕` 提示"请先登录"、不修改；登录后隐藏存服务端、绑定该登录账号、作用于任何查看的推荐。满足"只有登录后才能改喜好"。

3. **独立 `data/auth.db`**（WAL）：users / sessions / preferences 三表。不污染 3.8GB 推荐语料库 `collections.db`。`data/` 已 gitignore，token 不泄库。

4. **零第三方依赖**：手写 `.env` 解析（`load_optional`）+ 手写 httponly cookie 会话（sessions 表）。OAuth 未配置（无 client_id）时服务照常启动，仅登录功能显示"未启用"（503 带指引）。

5. **会话 cookie** `bgmlikes_session`：httponly、samesite=lax、30 天；secure 标注随请求协议（Caddy 反代后恒为 https）。

## 后果

- 隐藏偏好跨浏览器/设备同步（存服务端），但绑定登录账号——切换账号隐藏列表随之切换。
- 需用户注册 Bangumi 第三方应用拿 client_id/client_secret，填入 `.env`（`OAUTH_*`）才启用真实登录。
- "不感兴趣"数据从 localStorage 迁出，旧 localStorage 缓存不再读取（不迁移，量小）。
- OAuth 回调需公网 HTTPS 域名；未部署前无法端到端验证，本地仅能做未配置/未登录的降级冒烟。