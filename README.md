# NatV-QVAE｜KVP40 保守蛋白质生成

以 KVP40 NatV（341 aa，NMNAT–Nudix 双结构域）为参考，研究 WT 邻域的条件序列生成，最终为提高纯化酶体系中的 NAD⁺ 净积累提出可检验变体。Nudix 水解可减弱，NMNAT 合成需保留实际应用价值。

## 科学主线

高质量 NatV 数据 → 生成模型研究 → 工程化序列生成 → 候选自身结构与催化环境评价 → 候选库 → 未来酶学验证。

当前处于 **②生成模型研究向③工程化候选生成的过渡**。首轮以 **90%–100% WT 全局相似性**为范围，从家族演化和已核对的 NMNAT–Nudix 实验同源结构提出编辑位置，重点探索 Nudix 扰动与 NMNAT 合成环境保留。**Categorical QVAE 负责条件生成，冻结 ESM-2 提供必要参照**；短 InDel 仅在具备相应结构理由时纳入。研究定义见[研究计划](docs/RESEARCH_PLAN.md)。

已有[WT 对齐列先验生成](experiments/natv_wt_columns/REPORT.md)及[保守条件编辑](experiments/natv_conservative_editing/REPORT.md)两轮运行。首轮有目的候选生成的依据、范围与评价见研究计划；历史实测运行结果以各自报告为准。

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
