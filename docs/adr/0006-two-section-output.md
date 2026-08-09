# ADR 0006：两区输出（动画推荐 / 冷门发现），替代混合配额

日期：2026-08-08

## 背景

ADR 0002 的冷门配额（20 中 8，`quota_rank`）把热门与冷门**混合**进一个列表，前端用冷/热角标区分。用户要求"不再将热门和冷门混合在一起"，改为：

- 前端分成两个区域（动画推荐 / 冷门发现），顶部按钮切换；
- 各返回 20 条（总量从 20 增到 40，但分 tab 呈现不显多）；
- 导出 JPG 跟随当前 tab，分开导出。

## 决策

- **引擎** `Recommender.recommend()` 返回 `RecommendResult(normal, cold)`：
  - `normal` = 非冷门池（热度排名 ≤ cold_rank_threshold）的 CF 高分 top-k；
  - `cold` = 冷门池（排名 > 阈值）的 CF 高分 top-k。
  - 两区**共用同一份 CF 分与过滤**（排除已看 / franchise 排除+去重 / nsfw），仅按热度池子切分，各自取 top-k。
- **API** `/v1/recommend` 响应改为 `{username, normal: [], cold: [], source}`（破坏性变更：原 `count`/`recommendations` 删除）。
- **前端** 顶部两个 tab 按钮切换渲染；分区域后冷/热角标冗余，去掉。
- **JPG 导出** 导出当前 tab 所在区域，标题 `{用户名} 的动画推荐 / 冷门发现`，文件名带 `-cold` 后缀区分。
- **保留** `quota_rank` 函数与 `cold_quota` 旋钮（历史评估脚本 `experiment_rating_weight.py`、`proto_20k_quota.py` 仍用），产品路径不再调用。
- 评估脚本 `evaluate_product.py` / `measure_hotness.py` 同步为两区测量（动画区冷门占比应≈0%、冷门区应=100%）。

## 后果

- "至少 8 条冷门"的兜底语义变为**显式独立区域**：冷门口味用户不再被"至少"上限约束在混合列表里，动画区与冷门区各取所好。
- 两区同源同过滤，Recall/NDCG 应保持（动画区 = 纯 CF 的非冷门 top，与纯 CF 接近）。
- API 结构变更要求前端同步；旧客户端（如有）需适配。
