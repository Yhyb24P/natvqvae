# NatV-QVAE｜KVP40 保守序列生成研究

以噬菌体 KVP40 NatV（341 aa，NMNAT–Nudix 双结构域）为研究对象，通过计算生成可检验的保守变体，面向纯化酶体系中的 NAD⁺ 净积累改善；最终需要保留有应用价值的 NMNAT 合成能力。

## 研究路线

**研究对象与功能目标 → 高质量数据 → 训练与模型 → 生成候选 → 初步计算评价 → 遴选工程化 NatV 候选 → 优选候选序列库 → 未来酶学验证与反馈。**

当前阶段与首轮 H2 采样的研究判断见[研究计划](docs/RESEARCH_PLAN.md)；该轮证据与限制见[退役生成实验证据](discovery/natv_generation_evidence_decision.md)。

## 资料与目录

| 位置 | 用途 |
|---|---|
| [研究计划](docs/RESEARCH_PLAN.md) | 八环节科学问题、当前生成任务与评价标准、文献 |
| [数据与模型来源](docs/DATA_PROVENANCE.md) | 序列角色、划分、权重继承、数据边界 |
| [序列与结构对应](docs/NATV_PAIRING_AUDIT.md) | WT、同源构建体、残基坐标及配体状态 |
| `contracts/` | 科学目标和历史设计约定 |
| `inputs/manifest.json` | 只读来源路径与校验值 |
| `discovery/` | 来源核对、数据裁决与退役运行证据 |
| `src/natvqvae/` | 现役序列、区域、来源校验与 CLI 接口 |
| `tests/` | 代码和数据约定的核对 |

## 安装与检查

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m natvqvae doctor --root /path/to/nad_qvae
```

`doctor` 核对 `inputs/manifest.json` 指向的本地来源；大型数据与权重保存在研究主机。现役只保留契约、数据身份与检查接口；贡献与执行边界见 [AGENTS.md](AGENTS.md)。
