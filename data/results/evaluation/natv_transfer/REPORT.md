# 完整 NatV 两臂迁移研究报告

## 问题与设计

比较 NMNAT 辅助域预训练是否改善完整 341 aa KVP40 NatV 的条件 QVAE。两臂：

- `random`：随机初始化，1,644 条 NatV 全长训练。
- `auxinit`：从辅助域 `QVAE_seed8601_best.pt` 完整初始化，再跑同一全长训练。

两臂共享冻结 ESM-2、模型容量、27 种全长 mask、优化器、2,704 steps、主任务 seeds 8301/8302/8303（共 6 fits）。辅助预训练只有一个 seed 8601，额外成本单列。test 231 未读取。

## 训练

选择规则：128 个 component 均衡背景 × NMNAT_{8,16,32}_count 三个 panel（384 queries），K=16，每 5 epoch 及 initial/last；三任务与各 component 等权，按 per-target-residue `prior_joint NLL` 选 best，改善阈值 1e-4。

| run | init | initial NLL | best epoch | best NLL | last NLL | seconds |
|---|---|---:|---:|---:|---:|---:|
| random_8301 | random | 3.1718 | 20 | 1.8358 | 1.9677 | 1334.5 |
| random_8302 | random | 3.1507 | 15 | 1.8418 | 2.0127 | 1289.1 |
| random_8303 | random | 3.2292 | 20 | 1.8432 | 1.9748 | 1264.3 |
| auxinit_8301 | aux-QVAE | 2.0750 | 20 | 1.7981 | 1.9940 | 1261.0 |
| auxinit_8302 | aux-QVAE | 2.0748 | 15 | 1.7963 | 2.0062 | 1249.3 |
| auxinit_8303 | aux-QVAE | 2.0752 | 15 | 1.7832 | 2.0257 | 1234.2 |

六 fits 合计 7,632 s（2.12 h）。辅助预训练本身 1,137 s（18.95 min），另有首次失败 21 步/约 9 s、无 checkpoint。训练曲线见 `training_curves.png` 与 `training_curves.csv`。

## 全开发验证评价

对 227 条 validation 的 27 个混合 mask 做 4 次重复评价；按 component 与 seed 配对，bootstrap 2,000 次（seed 8802）。最终配对使用 44 个共同 component。

| run | global prior NLL | global posterior NLL | ESS |
|---|---:|---:|---:|
| random_8301 | 1.759936 | 1.759956 | 60.16 |
| random_8302 | 1.762161 | 1.762178 | 59.92 |
| random_8303 | 1.774799 | 1.774826 | 59.10 |
| auxinit_8301 | 1.740761 | 1.740756 | 60.45 |
| auxinit_8302 | 1.752424 | 1.752425 | 59.95 |
| auxinit_8303 | 1.763566 | 1.763582 | 60.00 |

配对差值定义为 `auxinit - random`，负值表示辅助初始化更好：

| region | mean Δ prior NLL | 95% CI | 判断 |
|---|---:|---:|---|
| global | -0.0170 | [-0.0268, -0.0070] | 改善 |
| NMNAT | -0.0490 | [-0.0603, -0.0376] | 改善，跨 3 seeds 一致 |
| Nudix | +0.0067 | [-0.0063, 0.0198] | 区间含 0，无显著退化 |
| linker | +0.0136 | [-0.0123, 0.0388] | 区间含 0 |
| random masks | -0.0192 | [-0.0279, -0.0110] | 改善 |

按 seed 的 NMNAT 差值：8301 -0.0412、8302 -0.0408、8303 -0.0650。posterior 结果与 prior 基本相同（NMNAT -0.0489，CI [-0.0602, -0.0375]），说明这是条件分布层面的迁移，而不是某个采样估计的偶然波动。

## 候选生成

真实 KVP40 全长上下文，NMNAT mask 8/16/32 各 16 个位置集合 ×64 draws ×3 seeds；每臂 9,216 条，共 18,432 条。两臂共享 group、mask、draw 和 uniform seed，逐行核对一致；全部 Nudix 不变、全长 341 aa。

- 总 unique 全长序列 18,413（19 次重复 attempts 保留在分母）；所有候选只改 NMNAT 区域。
- 原始统计：auxinit 的 Hamming 略低（13.03/13.26/13.35 vs random 13.24/13.61/13.37）；口袋位点行比例略低（0.673/0.686/0.689 vs 0.682/0.713/0.696）；锚点行在 seed 8301/8302 更低。
- Hamming 相等配对分别为 1,404/1,239/976 对；匹配后口袋行比例几乎相同（如 seed 8301 0.601 vs 0.600），锚点行 auxinit 仍略低。
- 结论：辅助初始化改善了条件概率，但没有证据表明它增加了候选数量、唯一性或口袋/锚点定向变化；在本固定预算下，auxinit 候选更偏保守，而不是更多或更“催化环境密集”。

代表候选按 3 对 Hamming 匹配（7/11/15 编辑）选出，见 `representative_candidates.fasta/csv`；优先匹配编辑强度与研究差异，而不是单看距离。

## 结构

结构仅使用既有 BioEmu2 structure 模式，8 帧/序列，512 行固定 MSA；无配体、无 pLDDT/PAE、无新折叠或对接。所有目标 8/8 帧通过 physicality 检查。

辅助域阶段（`data/results/evaluation/kvp40_auxiliary_structures/`）WT + 9 条唯一候选共 10 个目标，764.7 s。候选到 WT 的 median full RMSD 为 0.56–1.10 Å，NMNAT RMSD 0.294–0.914 Å；I145 邻域在 QVAE_R03/R04 和 deterministic_R04 达到 2.1–3.1 Å，其余约 0.28–1.17 Å。这里只有模板/预测邻近性，不是活性或稳定性证据。

迁移阶段 3 对 Hamming 匹配代表 + WT（`data/results/evaluation/natv_transfer_structures/`），709.3 s：

| 目标 | Hamming | full RMSD (Å) | NMNAT RMSD (Å) | I145 邻域 (Å) | anchor135 CA (Å) |
|---|---:|---:|---:|---:|---:|
| random_H07 | 7 | 0.581 | 0.309 | 0.301 | 0.243 |
| auxinit_H07 | 7 | 0.539 | 0.326 | 0.278 | 0.196 |
| random_H11 | 11 | 0.556 | 0.284 | 0.333 | 0.165 |
| auxinit_H11 | 11 | 0.555 | 0.296 | 0.252 | 0.145 |
| random_H15 | 15 | 0.619 | 0.455 | 0.798 | 0.262 |
| auxinit_H15 | 15 | 0.605 | 0.445 | 0.824 | 0.344 |

配对间的结构差异很小且没有一致方向；H15 的 I145 邻域和锚点位移较大，与更多编辑一致，但不能解释为功能改善。2QJO 仅用于 NMNAT 口袋/锚点映射（14/17/135/138 → H18/H21/T140/R143，33 个 ≤4.5 Å 口袋位点）；2QJT/2R5W 不使用于 NMNAT 锚点。所有结构分数和模板邻近都不等于酶活、稳定性或 NAD⁺ 净积累。

## 解释与边界

1. **迁移收益存在**：辅助初始化在完整 NatV 的 NMNAT 条件 NLL 上稳定改善约 0.049 nats/残基，global 约 0.017；Nudix 没有显著退化。这支持“更广的 NMNAT 演化信息可以改善完整 NatV 的条件序列建模”。
2. **候选供给没有同步改善**：固定 18,432 次尝试下，auxinit 没有更多 unique、更多口袋/锚点变化；Hamming 匹配后差异也很小。因此不能仅凭 NLL 宣称“产生了更多更好的设计候选”。
3. **潜变量不是主要机制**：prior 与 posterior 的差异很小、ESS 约 60/64；本结果更适合解释为辅助初始化对共享上下文/解码器的表征迁移，而不是共享 z 的联合建模优势。
4. **单一辅助 seed**：auxinit 的三次主训练共享同一个辅助预训练 seed 8601；主 seed 变化已报告，但辅助 seed 的不确定性未覆盖。辅助预训练 18.95 min 单列，不能与 random 臂的总成本视为相等。
5. **数据与结构限制**：30–35% identity 处于 twilight zone；70% 序列簇不是独立家族；2QJO 是 NAD⁺ 产物参照，没有 ATP/NMN 三元结构；结构邻近、模板位置和模型分数不等于活性、稳定性或 kcat/Km。
6. 所有候选的酶学活性、底物特异性、Nudix 水解与 NAD⁺ 净积累仍为 UNKNOWN；test 未参与。

## 复现路径

- 训练：`data/results/training/natv_transfer/`，权重 `data/weights/qvae/{random,auxinit}_seed83xx/{initial,best,last}.pt`。
- 生成：`data/results/generation/natv_transfer/`。
- 评价：`data/results/evaluation/natv_transfer/{run}/evaluation.json` 与 `per_query.jsonl`；配对汇总 `paired_summary.json`。
- 训练曲线：`data/results/evaluation/natv_transfer/training_curves.png`。
- 生成统计：`data/results/evaluation/natv_transfer/generation_summary.json`。
- 结构：`data/results/evaluation/kvp40_auxiliary_structures/`；迁移代表结构 `data/results/evaluation/natv_transfer_structures/`。
