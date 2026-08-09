# 周期性增量重爬 + 新番刷新（更新最新动画）——设计方案

> 状态：**设计稿，暂未实施**（2026-08-09 归档）
> 目标读者：后续实施时的自己 / 协作者

## Context

bgmlikes 已上线（东京服务器 43.167.208.246，systemd 服务 `bgmlikes` + Caddy）。语料库是 2026-08 的快照：20,007 用户、7.67M 收藏、30,556 subjects。**数据是静态的**，不含之后的新动画和新评分。

用户目标：**每隔一段时间重爬用户数据，更新最新动画**。

已确认决策：
- **每周**增量重爬（全量重爬 20k 用户 ≈ 10+ 小时，无法每周）
- **顺带爬「在看」状态**（type=3）——新番先出现在用户「在看」，只爬「看过」会滞后一季

## 现状约束（探索确认）

| 项 | 现状 | 影响 |
|---|---|---|
| `crawl_progress.json` | 无时间戳的数组，20k 全 done | 重跑是 no-op，需升级为 `{username: {last_crawled, seen_max}}` |
| `subjects` 表 | 只由 Archive dump 填充，URL 硬编码 `dump-2026-07-28`（最新已是 08-04） | 新番本体需刷新 |
| 推荐器加载 | 启动时全量加载 `collections_rated_rate` | 重爬后必须重建评分表 + 重启服务 |
| `collections.updated_at` | 存每条收藏更新时间 | 是「用户数据是否变化」的现成信号 |
| `build_rated_table.py` | `WHERE rate>0` 无状态过滤 | 在看带评分自动进矩阵，无需改 |
| `meta_tags` | 模型只存不用（recommender.py:250-257） | 新番缺 meta_tags 不影响 |
| 服务器 | cron + systemd timer 可用，GitHub + bgm.tv API 直连通 | 调度无障碍 |

## 方案总览

两条刷新路径，合一脚本编排，systemd timer 每周跑：

1. **subjects 刷新**（新番本体）：GitHub API 自动发现最新 Archive dump → 有新版才下载 → `import_archive` + `import_relations`（幂等）
2. **收藏增量重爬**（新评分 + 新番被看）：只爬「数据有变化的活跃用户」，看过 + 在看都爬
3. **重建评分表 + 重启服务**：`build_rated_table` → `systemctl restart bgmlikes`（只有真的爬到了东西才做）

## 实现

### 1. 爬虫改造 `scripts/crawl_users.py`
- **进度格式升级**：`crawl_progress.json` 数组 → 字典 `{username: {"last_crawled": ISO, "seen_max": ISO}}`；`load_progress`/`save_progress` 兼容旧数组（旧条目视为已爬、seen_max 空）。
- **新增 `select_users(conn, active_days=30)`**：选出本次要爬的用户，条件（任一）：
  - 不在进度字典（未爬过）
  - `MAX(collections.updated_at) > seen_max`（数据自上次爬取后有变化）
  - 兜底：`MAX(updated_at) >= now-30d` 且 `last_crawled` 早于 30 天前（首次运行/长间隔）
  - SQL 用 `GROUP BY user_hash HAVING MAX(updated_at)`；`updated_at` 是 ISO 字符串可直接比较
- **多状态**：`--states` 参数（默认 `看过,在看`），对每个状态分别 `fetch_collections(username, state=...)`，`INSERT OR REPLACE` 幂等（同一 (user_hash, subject_id) 后状态覆盖先状态）。返回后更新该用户 `seen_max = max(所有条目 updated_at)`、`last_crawled = now`。
- 保留 `--limit`（控制单次规模，便于冒烟）。
- 异常处理沿用现状：单用户失败记日志不记 seen_max，下次重试。

### 2. 下载脚本 `scripts/download_archive.py`
- **自动发现最新 URL**：GET `https://api.github.com/repos/bangumi/Archive/releases/latest` → 资产名按字典序取最大 `.zip`（dump-YYYY-MM-DD 前缀天然可排序）。替换硬编码 `URL`/`EXPECTED_SHA256`。
- **checksum 改为可选**：自动化运行时未知 sha，跳过校验（GitHub 资产走 HTTPS，可接受）；仍保留下载后记录 sha 到 `data/archive/latest.txt` 便于比对「是否新版本」。
- 现有 8 线程分片下载逻辑不变。

### 3. 编排脚本 `scripts/refresh_data.py`（新）
```
1. subjects 刷新：discover 最新 dump URL；若 != data/archive/latest.txt 记录值 → 下载 + import_archive + import_relations，更新 latest.txt，删 dump.zip 省磁盘
2. 增量爬取：select_users() → 逐用户爬 看过+在看（复用 crawl_users 逻辑）
3. 若 1 或 2 有实际变更 → 调 build_rated_table 重建 + （--restart 时）重启服务
   无变更 → 跳过 rebuild/restart，避免无谓停机
```
参数：`--limit`、`--skip-subjects`、`--no-restart`。

### 4. 服务器部署（新）
- `deploy/refresh.service`（oneshot，root）：
  - `runuser -u www-data -- /opt/bgmlikes/venv/bin/python -m scripts.refresh_data --restart`（以 www-data 写 data/ 的 DB；root 最后重启服务）
  - 例：`ExecStart=/bin/bash -c 'runuser -u www-data -- .../refresh_data --restart && systemctl restart bgmlikes'`
- `deploy/refresh.timer`：`OnCalendar=Sun 03:00`（CST），`Persistent=true`（错过补跑）
- 部署：scp 到 `/opt/bgmlikes/deploy/`，`systemctl daemon-reload && enable --now refresh.timer`
- 首次上线跑一次「在看回填」：`runuser -u www-data -- python -m scripts.refresh_data --backfill-watching`（给全部种子用户补爬一次在看，建立 seen_max；一次性 ~8h 后台慢跑）

### 5. 服务保持在线
- 爬取期只写 `collections` 表，服务内存模型不受影响 → 服务全程在线
- 只有 rebuild + restart 窗口短暂停机（`collections_rated_rate` 4.6M 行重建 ~3-5min + 模型加载 ~30-60s）

## 成本估算
- 首次/长间隔：近 30 天活跃 8,537 用户 × 每用户 ~7 请求（看过 probe+页 + 在看 probe+页）≈ 60k 请求 ≈ **6-9h**
- 稳态每周：只爬「数据有变化」的用户（通常 1-3k）≈ **1.5-3h**
- 在看一次性回填：20k 用户 × ~3 请求 ≈ **~8h**（仅首次）

## 数据语义（回答问题：「看过但未评分」怎么处理）
- **排除已看**（api.py:211）：所有状态都计入，已看过的即使未评分也不会被推荐
- **口味信号**（api.py:212）：只看「看过 且 评分 >0」，未评分不参与相似度
- **评分矩阵**（`build_rated_table` `WHERE rate>0`）：未评分不进矩阵
- 爬「在看」后：邻居的在看评分（>0）进矩阵 → 新番有候选分；但用户自己的「在看」仍不进自己的口味信号（profile 只看「看过」），且已被 `already` 排除，逻辑自洽

## 不动的部分
- `src/recommender.py` / `src/api.py` 推荐引擎：零改动
- 前端、OAuth 登录、封面图、隐藏偏好：不变
- `crawl_users.py` 的初始全量爬语义保留（`--states` 默认看过,在看 但 --incremental 不默认开）

## 验证
1. **本地**：`python -m scripts.crawl_users --states 看过 --limit 50 --incremental` 冒烟——确认 select 逻辑、进度文件迁移、只爬有变化的用户
2. **本地**：改 `--states 看过,在看`，确认在看条目进 `collections`，`build_rated_table` 后 `collections_rated_rate` 含在看行
3. **服务器**：首跑 `runuser -u www-data -- python -m scripts.refresh_data --limit 200` → health 200 → `collections` 行数增长 → rebuild → restart 后新数据生效
4. **新番本体**：确认 subjects 表出现 dump 中新增的最近动画 subject_id，且 `_items` 含它（一旦有用户看过）
5. **定时器**：`systemctl list-timers` 见 `bgmlikes-refresh.timer`；下个周期 `journalctl -u bgmlikes-refresh` 看日志
6. **端到端**：推荐某用户名，确认最新一季动画出现在候选
