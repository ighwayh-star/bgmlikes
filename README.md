# 🌸 bgmLikes · Bangumi 动画推荐系统

> 🖥️ **在线体验**：[https://bgmhiway.asia](https://bgmhiway.asia)　·　💻 **GitHub**：[ighwayh-star/bgmlikes](https://github.com/ighwayh-star/bgmlikes)

**输入 Bangumi 用户名，获得个性化动画推荐。** 它读你的收藏、算你的口味，从几十万部动画里替你挑出「下一部该看什么」。

不用注册，打开就能用；登录后打分会直接写回你自己的 Bangumi 账号。

![推荐页](https://bgmhiway.asia/pics/article_likes.jpg)

## ✨ 功能一览

### 🎯 个性化推荐

- 输入任意 Bangumi 用户名 → 抓取公开收藏 → **协同过滤 + 标签偏好 + 观看年份拟合 + 热度抑制**，一次产出 300 部候选
- 每次渲染 15 张，**滚动渐进加载**；🔄 **刷新**按钮随机重挑一组
- 登录后卡片右上角 ✕ 把不感兴趣的动画移出推荐池（恢复区可找回）；⬇ **导出图片**把当前推荐列表画成 JPG 发帖分享
- 点击卡片弹出**简介浮窗**：评分 / 日期 / 标签 / 剧情简介 / 相似动画，右上角图标一键打开 Bangumi 条目或跳 B 站搜索

![简介浮窗](https://bgmhiway.asia/pics/article_modal.jpg)

### ⭐ 10 星快速打分

- 浮窗底部内置 10 星打分器，点击即「标记看过 + 评分」，**直接写回你的真实 Bangumi 账号**，清除评分一键撤销
- 独立打分页 `/rate`：中 / 日 / 英文名实时搜索 + 热门列表，逐卡快打

![浮窗打分](https://bgmhiway.asia/pics/article_modal_rated.jpg)

### 📺 每日放送

- 今日 / 昨日分组追踪番剧更新，**本季全览**整季新番一目了然；不追的 ✕ 隐藏（灰化可恢复），列表越用越干净

![每日放送](https://bgmhiway.asia/pics/article_daily.jpg)

## 🚀 快速开始（本地开发）

```bash
pip install -r requirements.txt          # httpx numpy scipy fastapi uvicorn
# 复制 .env.example 为 .env，填入 BGM_TOKEN（https://next.bgm.tv/demo/access-token）

# 1. 数据准备（一次性）
python -m scripts.import_archive         # Archive dump → subjects 表
python -m scripts.import_relations       # 条目关系 → subject_relations 表
python -m scripts.run_20k                # 20k 用户爬取（后台，1-3 天墙钟，看门狗自动重启）

# 2. 启动 API 服务（后台）
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

# 3. 打开 http://127.0.0.1:8000 输入用户名自测
```

## 🧠 推荐算法（线上配置摘要）

```
User-CF：评分加权（打分矩阵 A 的 idf 乘子经 idf_in_score 可关）+ 相似度中心化（rate-5）+ KNN=200, λ=0
→ 候选过滤：排除已看 / nsfw（profile 无黄片口味时）/ 非日本动画 / 番外篇
→ franchise 排除：收藏过任一部 → 整系列不进候选；每系列只留第一部，续作/剧场版/番外一律排除
→ 推荐池：热度排名 rank ≤ 3000 的日本动画
→ 去热：渗透归一化 score = base/(df+1)^γ，γ 固定 1.0，分母只计重度用户（评分 ≥300 条）
→ 按 CF 分降序取 top-k
```

- 冷启动（口味 < 5 条）：回退流行度排序兜底。
- 宕机降级：Bangumi 实时拉取失败自动回退本地 2 万用户语料缓存，仍可出推荐（`source="cache"`）。
- 封面图中转：lain.bgm.tv 大陆直连超时，封面经本服务 `/img/{id}` 中转 + 磁盘缓存。

## 🛠 技术栈

Python · FastAPI · SQLite · numpy / scipy（自研 User-CF）· 静态单页前端 · 数据源 Bangumi API + Archive dump

## 📁 关键文件

| 文件 | 职责 |
|---|---|
| `src/recommender.py` | 推荐引擎深模块（评分/过滤/去重/去热）+ 共享函数 |
| `src/api.py` | FastAPI 薄层（`/v1/recommend`、`/v1/health`、`/auth/*`、`/preferences/*`、`/daily/*`、`/img/*`、`/pics/*`、`/sitemap.xml`） |
| `src/auth.py` | Bangumi OAuth 登录 + 会话 + 偏好（独立 data/auth.db） |
| `src/bangumi_api.py` | Bangumi API 深模块（限流/重试/分页/400 兜底） |
| `src/dataset.py` | 数据源 seam（线上 API / SQLite 缓存双 adapter，宕机降级链） |
| `src/images.py` | 封面图中转（磁盘缓存 data/covers/） |
| `scripts/evaluate_product.py` | 产品路径评估（与线上同源，防漂移） |
| `scripts/crawl_users.py` / `run_20k.py` | 用户爬取 + 看门狗 |

## 📜 决策记录 & 运维

- 关键取舍见 `docs/adr/`（0001 IDF 加权、0002 冷门配额、0003 franchise 去重、0005 评分加权、0006 两区改版、0007 Bangumi OAuth 登录、0008 老动画可见性研究、0009 老动画标签打分层）。
- **登录**：需在 `.env` 配 `OAUTH_CLIENT_ID/SECRET/REDIRECT_URI/SESSION_SECRET`（见 `docs/DEPLOY-OAUTH.md`），OAuth 回调需公网 HTTPS 域名；未配时服务照常启动、仅登录显示"未启用"。
- **本机代理坑**：检查 API 用 `curl`，或 `httpx` 加 `trust_env=False`——本机系统代理会对 127.0.0.1 返回 502。
- **看门狗**：`scripts/watch_server.py` 每 90s 健康检查、进程死了自动重启。
- **评分加权数据源**：需先跑一次 `python -m scripts.build_rated_table` 物化 `collections_rated_rate`；增量刷新后需重跑。
