# KVP40 保守候选正式生成

以 H2 Categorical QVAE 的 5 份 checkpoint 在 KVP40 WT 的 90%–100% 全局编辑相似性邻域生成完整序列。只做替换，不生成 InDel；不训练模型；不使用测试集。用户授权范围与预算见 `plan.json`。

## 采样

- 30 个掩码组：掩码预算 `[4,12,20,28,34]` × 每预算 6 组；位点从 336 个非保护位点无放回均匀抽取，跨 5 个 checkpoint 共享，预算间不嵌套。
- 每组 16 次先验采样，`(32,8)` one-hot z 在每条轨迹内固定；z 按 checkpoint 独立，抽样流与掩码跨 checkpoint 共享。每轨迹 5 个 stage：全掩码后交替两半更新。
- 掩码位置只表示允许更新，可从 20 种氨基酸中抽回 WT，也可在后续 stage 恢复 WT；不强制任何位置改变。
- 保护位点 `[1,14,17,135,138]` 不进入掩码；Nudix 关键残基仍可编辑。

## 判定与分母

- 每条尝试保留完整序列、实际编辑位置、掩码保留数、恢复位点、每 stage 变化、prior seed、z 状态与哈希、显式 `KEEP/SUB` 脚本。
- `PASS`：长度 341、脚本精确复原、实际位点属于掩码、保护位点未变、Levenshtein identity ≥ 0.9。
- 零编辑记为 `WT_REFERENCE`，不算新候选；重复序列保留并记录首次出现；全部尝试进入分母。
- `mask_budget` 是允许更新的位点数，不是实际编辑配额；实际编辑距离单独统计。

## 输出

`raw_candidates.csv`、`coverage/`、逐组轨迹与编辑脚本、`run_identity.json`、`generation_receipt.json`。HMM、家族与结构评价属于下一环节，不在本运行中完成。
