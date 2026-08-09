# bgmlikes · Bangumi 动画推荐系统

输入 Bangumi 用户名 → 返回个性化动画推荐。产品定位是**发现**：不推已看动画的续作，并分成两个独立区域——**动画推荐**（常规高分）与**冷门发现**（热度排名前 12% 之外的作品），前端顶部 tab 切换、各 20 条、不再混合。每条推荐可点 **✕ 不感兴趣** 隐藏，空位由推荐池内下一位替补（每区推荐池 40 部）。**修改偏好需登录 Bangumi 账号（OAuth）**：登录后可查看任意用户推荐，但隐藏列表绑定你的登录账号、存服务端、跨浏览器/设备生效；未登录查看功能正常、隐藏按钮提示"请先登录"。

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

## 算法流水线（产品当前配置）

```
User-CF（评分加权 idf×rate + KNN=200, λ=0）  评分核心（ADR 0005，替代 0001 的二值）
   口味信号排除四分以下（rate ≤ 4 不算正信号）：用户打低分的动画不再"帮推"同类
→ franchise 排除：已看系列的其余成员不进候选（续作/剧场版不占位）  （ADR 0003）
→ franchise 去重：每区内每个系列至多 1 条（保留 CF 分最高）
→ 两区输出（2026-08-08 改版，替代混合配额，见 ADR 0006）          （ADR 0002）
   - 动画推荐区 normal：非冷门池（热度排名 ≤ 3000）的 CF 高分 top-20
   - 冷门发现区 cold：冷门池（热度排名 > 3000）的 CF 高分 top-20
   前端顶部 tab 切换两个区域，各 20 条，不再混合
→ nsfw 保守过滤：profile 无 nsfw 口味时，候选排除黄片（纯黄片用户不受影响）
→ 每区各自按 CF 分降序取 top-20
```

- **冷启动**（画像不足）：回退流行度排序，两区仍各自取池子 top-N（冷门区给冷门池最热几条）。

## 旋钮（Recommender 构造参数）

| 参数 | 默认 | 含义 |
|---|---|---|
| `knn` | 200 | 相似用户候选池（评估定稿） |
| `blend_lambda` | 0.0 | 流行度混合系数（IDF 后归零） |
| `cold_quota` | 8 | 旧配额混合参数（ADR 0002）；2026-08-08 产品改两区后不再被 recommend() 使用，仅历史评估脚本/quota_rank 用 |
| `cold_rank_threshold` | 3000 | 两区切分阈值：冷门池 = 热度排名超过该值（全站前 12% 之外） |
| `matrix_min_rate` | 0 | 训练矩阵只收 rate>该值的对（0 = 全部 rate>0，ADR 0005） |
| `taste_min_rate` | 4 | 口味信号排除 rate≤该值（四分以下不算正信号） |

## 验证

```bash
curl http://127.0.0.1:8000/v1/health      # 模型统计（含 cold_quota / franchise_groups）
python -m scripts.evaluate                # 门槛1：纯 CF vs 流行度基线（Recall/NDCG）
python -m scripts.evaluate_product --users 500   # 产品路径离线评估（两区，含组成统计）
python -m scripts.measure_hotness --users 500    # 热度分布对比（纯CF/franchise/两区，验证"变冷"来源）
```

`evaluate_product.py` 测量与线上**完全一致**的路径（CF + franchise 排除/去重 + 两区切分），报告 Recall/NDCG 对比 + 两区冷门占比（动画区应≈0%、冷门区应=100%）+ 同系列重复数。注意：franchise 排除会系统性降低 Recall@10（因为"续作命中"被刻意赶出），这是产品要求，不是回归。

## 关键文件

| 文件 | 职责 |
|---|---|
| `src/recommender.py` | 推荐引擎深模块（评分/过滤/去重/两区切分）+ 共享函数 |
| `src/api.py` | FastAPI 薄层（`/v1/recommend`、`/v1/health`、`/auth/*`、`/preferences/*`） |
| `src/auth.py` | Bangumi OAuth 登录 + 会话 + 服务端"不感兴趣"偏好（独立 data/auth.db） |
| `src/bangumi_api.py` | Bangumi API 深模块（限流/重试/分页/400 兜底） |
| `src/dataset.py` | 数据源 seam（线上 API / SQLite 缓存双 adapter） |
| `src/images.py` | 封面图中转（隐藏 lain.bgm.tv，磁盘缓存 data/covers/） |
| `scripts/evaluate.py` | 门槛1 评估（纯 CF 基线） |
| `scripts/evaluate_product.py` | 产品路径评估（franchise + 配额） |
| `scripts/import_archive.py` | Archive dump → subjects |
| `scripts/import_relations.py` | 条目关系 → subject_relations（franchise 数据源） |
| `scripts/crawl_users.py` / `run_20k.py` | 用户爬取 + 看门狗 |
| `scripts/build_rated_table.py` | 一次性迁移：物化 collections_rated_rate（rate>0+rate，评分加权数据源） |
| `scripts/experiment_rating_weight.py` | 评分加权对比实验（矩阵/查询双层低分阈值） |

## 决策记录

关键取舍见 `docs/adr/`（0001 IDF 加权、0002 冷门配额、0003 franchise 去重、0005 评分加权、0006 两区改版、0007 Bangumi OAuth 登录）与 `docs/PLAN.md`。术语定义见 `docs/CONTEXT.md`。

## 已知注意事项

- **登录**：点顶部「登录 (Bangumi)」走官方 OAuth。需先在 `.env` 配 `OAUTH_CLIENT_ID/SECRET/REDIRECT_URI/SESSION_SECRET`（见 `docs/DEPLOY-OAUTH.md`，逆向指引注册 Bangumi 应用）；未配时服务照常启动、仅登录显示"未启用"。隐藏列表存服务端 `data/auth.db`，绑定登录账号、跨设备生效；未登录查看正常、隐藏按钮提示"请先登录"。OAuth 回调需公网 HTTPS 域名，本地调试登录不可端到端。
- 本机内存吃紧（常驻 SD-webui / 其他进程时仅剩 ~2GB）：大型评估请用小采样（`--users 500`）；服务器加载只需 ~0.2GB，但系统级内存压力可能压死进程，若服务 502 且进程 RSS 掉到 ~2MB，说明进程已死，按上文命令重启。
- 检查 API 时用 `curl`，或 `httpx` 加 `trust_env=False`——本机系统代理会对 127.0.0.1 返回 502。
- **Bangumi 可达性：轻量实时拉取走直连即可**（2026-08-07 实测：服务器进程无代理环境变量、纯直连，`/v1/recommend` 返回 `source="api"`）。大规模爬取（run_20k）期间直连会被限流，那时需要走系统代理换出口 IP——"必须走代理"是爬取期的观察，不代表当前常态。代理状态与推荐服务解耦：qyt 崩/TUN 关都不影响实时推荐。
- **宕机降级链**（2026-08-07）：`/v1/recommend` 实时拉取失败时自动降级到本地语料缓存（已爬取的 2 万用户仍可出推荐），响应 `source="cache"` 标注、前端提示"本地缓存"。实时拉取有 15s 整体超时；网络不通（ConnectError）时 ~1s 快速失败。未知用户 + 宕机 = 502。
- **看门狗**（2026-08-07）：本机内存压力可能压死服务器（夜里已发生两次）。`scripts/watch_server.py` 每 90s 健康检查、进程死了自动重启（独立 DETACHED 运行，不依赖会话）。手动部署：`python -m scripts.watch_server`（前台）或 DETACHED 后台跑。
- **封面图**（2026-08-07）：lain.bgm.tv（bgm.tv 封面 CDN）是 Cloudflare 海外节点，**大陆直连超时**。所以封面图经 `/img/{id}` 由本服务中转：首次拉取落盘（data/covers/，随 data/ 被 gitignore），之后秒开。测试：`curl http://127.0.0.1:8000/img/1`。
- **评分加权数据源**（2026-08-08）：评分加权（ADR 0005）需要 rate 走快速加载路径，先跑一次 `python -m scripts.build_rated_table` 物化 `collections_rated_rate`（含 rate 的覆盖索引）。未跑则 Recommender 加载报错（表不存在）。增量刷新后需重跑该迁移。
- **评分加权实测**（2026-08-08）：`scripts/experiment_rating_weight.py` 全量（12000 重度用户/700 评估）实测：1-10 加权 + 查询排除≤4，产品路径 Recall +6.7% / NDCG +5.8%（相对原二值）；矩阵是否剔除低分对结果无差别。成本零（稀疏结构不变）。
