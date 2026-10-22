# NatV｜KVP40 双结构域蛋白的计算生成与保守工程化设计

本项目研究噬菌体 KVP40 的 NatV 蛋白（341 aa，NMNAT–Nudix 双结构域），探索利用家族演化信息、蛋白质生成模型和结构证据设计保守变体，为改善 NAD⁺ 净积累提供可检验的候选序列。

## 科学问题

NatV 的 NMNAT 结构域参与 NAD⁺ 合成，Nudix 结构域具有相关水解功能。工程目标是在保留具有应用价值的 NMNAT 合成能力的前提下，提高纯化酶体系中的 NAD⁺ 净积累。

当前主要研究 KVP40 WT 邻域内的序列生成，允许 90%–100% 全局编辑相似性和有依据的少量短 InDel。

## 研究路线

研究对象与功能目标 → 高质量数据 → 训练与模型 → 生成候选 → 初步计算评价 → 遴选工程化 NatV 候选 → 优选候选序列库 → 未来酶学验证与反馈。

当前已建立经过核对的 NatV 家族数据、正式训练与验证划分、同源结构对应关系，并完成多轮 QVAE 相关生成研究。首轮 H2 条件采样已完成，但其候选不作为下一阶段的主要输入；原始实验结论及限制见[退役生成实验证据](discovery/natv_generation_evidence_decision.md)。

**已完成条件 Categorical QVAE 的训练、联合评价与候选生成。** 38次拟合和两模型各61,440次单轮采样提供48条代表序列。QVAE开发集联合NLL小幅改善，但共变统计与候选供给优势尚无充分证据；[报告](experiments/natv_conditional_qvae/REPORT.md)、[训练曲线](experiments/natv_conditional_qvae/training_curves.png)和[候选清单](experiments/natv_conditional_qvae/CANDIDATES.md)保留完整判断。此前重训与生成证据由[研究计划](docs/RESEARCH_PLAN.md)追溯。

**已完成 NMNAT 演化设计证据 Phase A/B1。** 从审计后的有界来源得到 7 条待检验设计假设；B1 只重排现有候选池，配对 F13Y+A134G 缺失，未判 Gate B1 通过，B2 未启动。[实验报告](experiments/natv_evolution_informed_nmnat/REPORT.md)保留方法、负结果与限制；这些来源不是训练人口。

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

条件 QVAE 的 GPU 环境版本见[运行环境](experiments/natv_conditional_qvae/environment.json)，方案及输入边界见本轮 `PLAN.md`。完整入口与已完成训练后的评价入口分别为：

```bash
python scripts/run_conditional_study.py --run experiments/natv_conditional_qvae
python scripts/run_conditional_study.py --run experiments/natv_conditional_qvae --finish-only
```
