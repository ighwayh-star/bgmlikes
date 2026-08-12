# bgmLikes · Bangumi 动画推荐系统

> 🖥️ **在线体验**：[https://bgmhiway.asia](https://bgmhiway.asia)　·　💻 **源码**：[github.com/ighwayh-star/bgmlikes](https://github.com/ighwayh-star/bgmlikes)

输入 Bangumi 用户名，即可获得个性化动画推荐。产品定位是**发现**：不推已看动画的续作，结果分成「动画推荐」与「冷门发现」两个独立区域，帮你挖出高分佳作和热门之外的好番。

## 功能

- **两区推荐**：动画推荐区（热度前 3000 的常规高分）+ 冷门发现区（热度 3000 之外的作品），顶部 tab 切换，各一批 10 部
- **✕ 不感兴趣**：隐藏不想要的推荐，空位自动由推荐池替补（池子 50 部）
- **换一批**：从推荐池按排名加权重新抽取 10 部替换当前列表
- **Bangumi OAuth 登录**：偏好（隐藏列表）绑定账号、存服务端、跨设备生效；未登录查看照常
- **宕机降级**：实时拉取失败自动回退 2 万用户本地语料缓存，仍可出推荐

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

## 算法概览

```
User-CF（评分加权 idf×rate + KNN=200, λ=0）
→ franchise 排除：已看系列的其余成员不进候选（续作/剧场版不占位）
→ franchise 去重：每系列至多 1 条（保留 CF 分最高）
→ 两区输出：动画推荐区（热度 ≤3000）与 冷门发现区（热度 >3000），各取 CF 高分 top-10
→ nsfw 保守过滤：profile 无 nsfw 口味时，候选排除黄片
```

- **冷启动**（画像不足）：回退流行度排序，两区仍各取池子 top-N。

## 配置旋钮（Recommender 构造参数）

| 参数 | 默认 | 含义 |
|---|---|---|
| `knn` | 200 | 相似用户候选池（评估定稿） |
| `cold_rank_threshold` | 3000 | 两区切分阈值：冷门池 = 热度排名超过该值 |
| `matrix_min_rate` | 0 | 训练矩阵只收 rate>该值的对（0 = 全部 rate>0） |
| `taste_min_rate` | 4 | 口味信号排除 rate≤该值（四分以下不算正信号） |
| `rate_center` | 5.0 | 相似度中心化：低分（<5）用户与高分用户相似度减半 |
| `df_min_rated` | 300（生产） | 去热分母只统计评分条数 ≥ 该值的重度用户 |

## 验证

```bash
curl http://127.0.0.1:8000/v1/health                       # 模型统计
python -m scripts.evaluate                                 # 纯 CF vs 流行度基线（Recall/NDCG）
python -m scripts.evaluate_product --users 500             # 产品路径离线评估（两区 + 组成统计）
python -m scripts.measure_hotness --users 500              # 热度分布对比（验证"变冷"来源）
```

## 关键文件

| 文件 | 职责 |
|---|---|
| `src/recommender.py` | 推荐引擎深模块（评分/过滤/去重/两区切分） |
| `src/api.py` | FastAPI 薄层（`/v1/recommend`、`/v1/health`、`/auth/*`、`/preferences/*`） |
| `src/auth.py` | Bangumi OAuth 登录 + 会话 + 服务端"不感兴趣"偏好 |
| `src/bangumi_api.py` | Bangumi API 深模块（限流/重试/分页/400 兜底） |
| `src/dataset.py` | 数据源 seam（线上 API / SQLite 缓存双 adapter） |
| `src/images.py` | 封面图中转（隐藏 lain.bgm.tv，磁盘缓存 data/covers/） |
| `scripts/evaluate_product.py` | 产品路径评估（franchise + 两区，与线上同源） |
| `scripts/crawl_users.py` / `run_20k.py` | 用户爬取 + 看门狗 |

## 决策记录

关键取舍见 `docs/adr/`（0001 IDF 加权、0002 冷门配额、0003 franchise 去重、0005 评分加权、0006 两区改版、0007 Bangumi OAuth 登录）与 `docs/PLAN.md`、`docs/CONTEXT.md`。

## 运维注意

- **登录**：需在 `.env` 配 `OAUTH_CLIENT_ID/SECRET/REDIRECT_URI/SESSION_SECRET`（见 `docs/DEPLOY-OAUTH.md`），且 OAuth 回调需公网 HTTPS 域名；未配时服务照常启动、仅登录显示"未启用"。
- **本机代理坑**：检查 API 用 `curl`，或 `httpx` 加 `trust_env=False`——本机系统代理会对 127.0.0.1 返回 502。
- **宕机降级链**：`/v1/recommend` 实时拉取失败自动降级本地缓存（`source="cache"`），前端提示"本地缓存"；未知用户 + 宕机 = 502。
- **看门狗**：`scripts/watch_server.py` 每 90s 健康检查、进程死了自动重启。服务器进程被系统内存压力压死时（502 且 RSS 掉到 ~2MB），按上文命令重启。
- **封面图**：lain.bgm.tv 大陆直连超时，封面经 `/img/{id}` 本服务中转，首次拉取落盘 data/covers/。
- **评分加权数据源**：需先跑一次 `python -m scripts.build_rated_table` 物化 `collections_rated_rate`；增量刷新后需重跑。
