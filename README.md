# bgmLikes · Bangumi 动画推荐系统

> 🖥️ **在线体验**：[https://bgmhiway.asia](https://bgmhiway.asia)　·　💻 **源码**：[github.com/ighwayh-star/bgmlikes](https://github.com/ighwayh-star/bgmlikes)

输入 Bangumi 用户名，获得个性化动画推荐。产品定位是**发现**——用自研 User-CF 从 2 万用户语料挖出高分佳作，并刻意过滤你已看过的内容：不推已看动画的续作，每系列只推荐一部。

## 页面与功能

| 页面 | 说明 |
|---|---|
| **推荐系统** `/likes` | 输入用户名 → 一次拉取 top-300 → 池子里（前 50 条按分排序）加权随机展示 10 条。🎲 **换一批**重新抽取（上次展示过的权重 ×0.1）；✕ **不感兴趣**隐藏并自动补位；⬇ **导出图片**把当前列表画成 JPG 发帖用 |
| **每日放送** `/daily` | 服务端代理 Bangumi 放送表，**每日** / **本季全览**两个视图，可隐藏不喜欢的新番 |
| **主页** `/` | 导航入口 |

- **Bangumi OAuth 登录**：隐藏列表绑定账号、存服务端、跨设备生效；未登录查看功能照常（隐藏按钮提示登录）。
- **宕机降级**：Bangumi 实时拉取失败自动回退本地 2 万用户语料缓存，仍可出推荐（`source="cache"`）。
- **封面图中转**：lain.bgm.tv 大陆直连超时，封面经本服务 `/img/{id}` 中转 + 磁盘缓存。

技术栈：Python · SQLite · numpy/scipy（自研 User-CF）· FastAPI · 静态单页。数据源：Bangumi API + Archive dump。

## 快速开始

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

## 算法（当前线上配置）

```
User-CF：评分加权 idf×rate + 相似度中心化（rate-5）+ KNN=200, λ=0
→ 候选过滤：排除已看 / nsfw（profile 无黄片口味时）/ 非日本动画 / 番外篇
→ franchise 排除：收藏过任一部 → 整系列不进候选；每系列只留第一部，续作/剧场版/番外一律排除
→ 推荐池：热度排名 rank ≤ 3000 的日本动画
→ 去热：渗透归一化 score = base/(df+1)^γ，γ 固定 1.0，分母只计重度用户（评分 ≥300 条）
→ 按 CF 分降序取 top-k
```

- **冷启动**（口味 < 5 条）：回退流行度排序兜底。

## 配置旋钮（Recommender 构造参数，.env 可调）

| 参数 | 默认 | 含义 |
|---|---|---|
| `knn` | 200 | 相似用户候选池 |
| `rank_cap` | 3000 | 推荐池热度上限：只推 rank ≤ 该值的日本动画 |
| `gamma` | 1.0 | 去热强度（渗透归一化指数，线上固定） |
| `df_min_rated` | 300 | 去热分母只统计评分条数 ≥ 该值的重度用户（剔除轻度用户回暖热门） |
| `rate_center` | 5.0 | 相似度中心化：1-4 分负偏好、5 中性、6-10 正偏好（打分保持原始分） |
| `taste_min_rate` | 0 | 口味信号收 rate>该值（0 = 全部 rate>0 都算口味信号） |
| `matrix_min_rate` | 0 | 训练矩阵只收 rate>该值的对 |
| `min_profile` | 5 | 口味信号条数少于该值 → 冷启动流行度兜底 |

## 验证

```bash
curl http://127.0.0.1:8000/v1/health                       # 模型统计（rank_cap/γ/df_min_rated/rate_center）
python -m scripts.evaluate                                 # 纯 CF vs 流行度基线（Recall/NDCG）
python -m scripts.evaluate_product --users 500             # 产品路径离线评估（与线上同源）
python -m scripts.measure_hotness --users 500              # 热度分布对比（验证"变冷"来源）
```

## 关键文件

| 文件 | 职责 |
|---|---|
| `src/recommender.py` | 推荐引擎深模块（评分/过滤/去重/去热）+ 共享函数 |
| `src/api.py` | FastAPI 薄层（`/v1/recommend`、`/v1/health`、`/auth/*`、`/preferences/*`、`/daily/*`、`/img/*`） |
| `src/auth.py` | Bangumi OAuth 登录 + 会话 + 偏好（独立 data/auth.db） |
| `src/bangumi_api.py` | Bangumi API 深模块（限流/重试/分页/400 兜底） |
| `src/dataset.py` | 数据源 seam（线上 API / SQLite 缓存双 adapter，宕机降级链） |
| `src/images.py` | 封面图中转（磁盘缓存 data/covers/） |
| `scripts/evaluate_product.py` | 产品路径评估（与线上同源，防漂移） |
| `scripts/crawl_users.py` / `run_20k.py` | 用户爬取 + 看门狗 |

## 决策记录

关键取舍见 `docs/adr/`（0001 IDF 加权、0002 冷门配额、0003 franchise 去重、0005 评分加权、0006 两区改版、0007 Bangumi OAuth 登录）。两区推荐已于 2026-08-11 下线，`docs/PLAN.md` 含演进过程。

## 运维注意

- **登录**：需在 `.env` 配 `OAUTH_CLIENT_ID/SECRET/REDIRECT_URI/SESSION_SECRET`（见 `docs/DEPLOY-OAUTH.md`），OAuth 回调需公网 HTTPS 域名；未配时服务照常启动、仅登录显示"未启用"。
- **本机代理坑**：检查 API 用 `curl`，或 `httpx` 加 `trust_env=False`——本机系统代理会对 127.0.0.1 返回 502。
- **看门狗**：`scripts/watch_server.py` 每 90s 健康检查、进程死了自动重启。服务器进程被系统内存压力压死时（502 且 RSS 掉到 ~2MB），按上文命令重启。
- **评分加权数据源**：需先跑一次 `python -m scripts.build_rated_table` 物化 `collections_rated_rate`；增量刷新后需重跑。
