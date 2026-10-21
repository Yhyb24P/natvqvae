# NatV-QVAE｜KVP40 保守序列生成研究

以噬菌体 KVP40 NatV（341 aa，NMNAT–Nudix 双结构域）为研究对象，通过计算生成可检验的保守变体，面向纯化酶体系中的 NAD⁺ 净积累改善；最终需要保留有应用价值的 NMNAT 合成能力。

## 研究路线

**研究对象与功能目标 → 高质量数据 → 训练与模型 → 生成候选 → 初步计算评价 → 遴选工程化 NatV 候选 → 优选候选序列库 → 未来酶学验证与反馈。**

当前重点是**④生成候选**：基于现有 Categorical QVAE，在 KVP40 WT 的 **90%–100% 全局编辑相似性**范围内充分生成完整序列。让模型在可变位置上选择残基，包括保留 WT；不预先要求每条提案符合 NMNAT/Nudix 功能假设。冻结 ESM-2 仅作为必要参照。⑤评价序列、结构和催化环境，⑥根据证据遴选候选，必要时返回⑤做针对性分析。研究判断见[研究计划](docs/RESEARCH_PLAN.md)。

已完成一轮 2,400 次不强制突变的正式生成，实际覆盖 91.79%–100% identity，31–34 编辑层尚无样本；见[保守候选生成报告](experiments/natv_conservative_generation/REPORT.md)。已有[WT 对齐列先验生成](experiments/natv_wt_columns/REPORT.md)和[保守条件编辑](experiments/natv_conservative_editing/REPORT.md)的真实运行记录；它们不等于已完成候选遴选。

## 资料与目录

| 位置 | 用途 |
|---|---|
| [研究计划](docs/RESEARCH_PLAN.md) | 八环节科学问题、当前生成任务与评价标准、文献 |
| [数据与模型来源](docs/DATA_PROVENANCE.md) | 序列角色、划分、权重继承、数据边界 |
| [序列与结构对应](docs/NATV_PAIRING_AUDIT.md) | WT、同源构建体、残基坐标及配体状态 |
| `contracts/` | 科学目标和历史设计约定 |
| `inputs/manifest.json` | 只读来源路径与校验值 |
| `discovery/` | 数据与模型来源核对、研究决定 |
| `experiments/natv_candidate_sequences/` | 历史局部编辑 |
| `experiments/natv_wt_columns/` | WT 对齐列先验生成比较 |
| `experiments/natv_conservative_editing/` | 2,976 次保守编辑生成及真实轨迹 |
| `experiments/natv_conservative_generation/` | 2,400 次 QVAE 保守候选生成、覆盖统计及审计 |
| `src/natvqvae/` | 现役序列工具、条件生成与评价接口 |
| `tests/` | 代码和数据约定的核对 |

## 安装与检查

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m natvqvae doctor --root /path/to/nad_qvae
```

`doctor` 核对 `inputs/manifest.json` 指向的本地来源；大型数据与权重保存在研究主机。各次实验以自身 `REPORT.md` 为复现入口，贡献与执行边界见 [AGENTS.md](AGENTS.md)。
