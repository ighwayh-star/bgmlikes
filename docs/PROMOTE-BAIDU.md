# 百度站长平台接入

目标：让 `bgmhiway.asia` 被百度收录。本仓库已自动化的部分 + 需要你在百度账号里点几下的人工步骤如下。

## 已完成（部署后即生效）

| 资源 | 说明 |
|---|---|
| `https://bgmhiway.asia/sitemap.xml` | 站点地图（`/`、`/likes`、`/daily`、`/rate`），百度/必应都可提交 |
| `https://bgmhiway.asia/robots.txt` | 放行内容页、屏蔽 `/auth/`、指向 sitemap |
| 四页 `og:*` 标签 | 首页 / 推荐 / 每日放送 / 打分，各带 description + 分享卡图 |

## 人工步骤（需要你的百度账号，一次性）

1. **注册/登录** [百度搜索资源平台](https://ziyuan.baidu.com)（原"站长平台"，用百度账号扫码/登录）。
2. **添加站点**：用户中心 → 站点管理 → 添加网站 → 填 `https://bgmhiway.asia` → 选协议 `https`。
3. **验证所有权**（三选一，推荐前两种，不用动 DNS）：
   - **HTML 标签验证**：平台给你一段 `<meta name="baidu-site-verification" content="...">`，把 content 发我，我加到 `web/home.html` 并重新部署；**或**
   - **文件验证**：平台给你一个 `xxx.html` 文件内容，把**文件名 + 内容**发我，我加个路由部署；**或**
   - **CNAME 验证**：在域名解析处加一条 `CNAME` 记录（你在域名商控制台操作，我不碰 DNS）。
4. **提交收录**（站点状态变为"验证通过"后）：
   - 左侧 **普通收录 → sitemap** → 提交 `https://bgmhiway.asia/sitemap.xml`；
   - 同一页面 **手动提交**，把下面四个 URL 逐条贴进去：
     ```
     https://bgmhiway.asia/
     https://bgmhiway.asia/likes
     https://bgmhiway.asia/daily
     https://bgmhiway.asia/rate
     ```
5. （可选）若日后给域名**办了 ICP 备案**，可以用 **主动推送（快速收录）** 接口，token 在「链接提交 → 主动推送」页面生成，然后跑本仓库的推送脚本：
   ```bash
   python -m scripts.baidu_push --token <你的token>
   ```
   > 注意：主动推送/快速收录通常要求国内备案；服务器在海外未备案时，走第 4 步的 sitemap + 手动提交即可收录。

## 附：必应（Bing）顺手提交

Bing 支持 sitemap 直接抓取，且默认从 Google 导入。在 [Bing Webmaster](https://www.bing.com/webmasters) 添加站点后提交同一 sitemap 即可，不用验证文件。
