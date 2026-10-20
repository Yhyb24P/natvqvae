# NatV-QVAE｜KVP40 保守蛋白质生成

以 KVP40 NatV（341 aa，NMNAT–Nudix 双结构域）为参考，研究 WT 邻域的条件序列生成，最终为提高纯化酶体系中的 NAD⁺ 净积累提出可检验变体。Nudix 水解可减弱，NMNAT 合成需保留实际应用价值。

## 科学主线

高质量 NatV 数据 → 生成模型研究 → 工程化序列生成 → 候选自身结构与催化环境评价 → 候选库 → 未来酶学验证。

当前优先 **90%–100% WT 全局相似性**的保守设计：多位点协同替换为主体，允许有定位依据的少量短 InDel。用冻结 ESM-2/SE 条件掩码生成与合理简单编辑作为基线，检验 QVAE 潜变量能否提供实际生成收益；不预设 QVAE 胜出。相似性不等于活性，计算分数不等于 NAD⁺ 净积累。

此前完成 WT 对齐列先验生成比较，但其不受控 gap 和大幅序列偏离不能代表现役保守编辑任务。结果及判断见[原实验报告](experiments/natv_wt_columns/REPORT.md)；本轮[保守条件编辑结果](experiments/natv_conservative_editing/REPORT.md)已完成；任务和比较标准见[研究计划](docs/RESEARCH_PLAN.md)。

## 资料与目录

| 位置 | 用途 |
|---|---|
| [研究计划](docs/RESEARCH_PLAN.md) | 研究问题、生成任务、比较与证据标准、文献 |
| [数据来源](docs/DATA_PROVENANCE.md) | 家族角色、训练划分、旧模型与衍生输入来源 |
| [结构配对](docs/NATV_PAIRING_AUDIT.md) | 序列—构建体—坐标—同源关系 |
| `contracts/` | 科学目标及已有实验的位点与评价约定 |
| `inputs/manifest.json` | 旧项目只读输入与 SHA256 |
| `discovery/` | 数据发现与继承裁决的实际记录 |
| `experiments/natv_candidate_sequences/` | 历史局部编辑结果 |
| `experiments/natv_wt_columns/` | 全列先验生成训练、序列与评价 |
| `experiments/natv_conservative_editing/` | 保守条件编辑比较、序列与完整轨迹 |
| `src/natvqvae/` | 当前研究中实际使用的模型接口与序列工具 |
| `tests/` | 序列身份、输入和代码行为检查 |

## 安装与核对

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m natvqvae doctor --root /path/to/nad_qvae
```

源数据与大型权重保存在只读来源及当地研究主机；`doctor` 检查 `inputs/manifest.json` 所列本地文件。已有运行的来源与复现入口以各自报告为准。贡献与执行边界见 [AGENTS.md](AGENTS.md)。
