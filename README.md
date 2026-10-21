# NatV｜KVP40 双结构域蛋白的计算生成与保守工程化设计

本项目研究噬菌体 KVP40 的 NatV 蛋白（341 aa，NMNAT–Nudix 双结构域），探索利用家族演化信息、蛋白质生成模型和结构证据设计保守变体，为改善 NAD⁺ 净积累提供可检验的候选序列。

## 科学问题

NatV 的 NMNAT 结构域参与 NAD⁺ 合成，Nudix 结构域具有相关水解功能。工程目标是在保留具有应用价值的 NMNAT 合成能力的前提下，提高纯化酶体系中的 NAD⁺ 净积累。

当前主要研究 KVP40 WT 邻域内的序列生成，允许 90%–100% 全局编辑相似性和有依据的少量短 InDel。

## 研究路线

研究对象与功能目标 → 高质量数据 → 训练与模型 → 生成候选 → 初步计算评价 → 遴选工程化 NatV 候选 → 优选候选序列库 → 未来酶学验证与反馈。

当前已建立经过核对的 NatV 家族数据、正式训练与验证划分、同源结构对应关系，并完成多轮 QVAE 相关生成研究。首轮 H2 条件采样已完成，但其候选不作为下一阶段的主要输入；原始实验结论及限制见[退役生成实验证据](discovery/natv_generation_evidence_decision.md)。

**正在执行条件 Categorical QVAE 研究。** 完整序列后验、学习型条件先验与共享离散潜变量用于检验多位点联合建模及候选序列供给，范围与预算见[本轮方案](experiments/natv_conditional_qvae/PLAN.md)。[此前重训](experiments/kvp40_target_posterior_training/REPORT.md)和[历史生成](experiments/kvp40_conditional_generation/REPORT.md)保留为研究证据。

## 项目结构

| 目录 | 内容 |
|---|---|
| `data/manifest.json` | 外部来源定位与 SHA256 |
| `data/raw/source/` | 指向只读来源的本地符号链接 |
| `data/raw/structures/` | 2QJO/2QJT/2R5W 实验结构 |
| `data/raw/metadata/` | 对应 RCSB 模板元数据 |
| `data/processed/` | 正式训练/验证数据、群体划分、序列—结构映射与核验记录 |
| `data/interim/`、`data/models/`、`data/results/` | 中间产物、权重与运行结果，按需创建 |
| `src/natvqvae/` | 现役条件模型、序列、区域、来源校验与 CLI |
| `scripts/` | 数据准备、训练/生成与审计入口 |
| `experiments/` | 研究方案、报告与摘要 |
| `discovery/` | 数据发现、模型继承及历史裁决 |
| `contracts/` | 科学目标和序列约定 |
| `docs/` | 研究计划、数据来源和结构对应说明 |

数据群体及来源见 [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md)，实验同源结构与残基编号见 [NATV_PAIRING_AUDIT.md](docs/NATV_PAIRING_AUDIT.md)。

## 环境与数据检查

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"

pytest -q
python -m natvqvae doctor --root /path/to/nad_qvae
```
