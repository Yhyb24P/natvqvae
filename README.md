# NatV-QVAE | NAD⁺ 净生成酶设计

研究 KVP40 NatV 衍生的 NMNAT–Nudix 双功能酶，目标是在纯化酶体系中提高
NMN + ATP 生成 NAD⁺ 的净积累。允许 Nudix 水解下降或丢失，前提是 NMNAT
合成保留到具有实际应用意义的程度。目前没有变体经实验证明优于野生型。

## 科学目标

NMNAT 催化 `NMN + ATP ⇌ NAD⁺ + PPi`；Nudix 催化 `NAD⁺ + H₂O → NMN + AMP`。
提高合成和降低水解是两条不同的因果路径。判据是受控条件下实测的 NAD⁺
净积累，NMNAT 动力学与 Nudix 水解分别测量。模型分数只说明相容性，
不是活性标签。

## 研究路线

1. **数据。** 使用来源可追溯、结构域对应明确、训练与验证边界清楚的
   NatV 家族和同源实验结构。
2. **生成。** QVAE 与无潜在变量 SE 基线在同一任务上公平比较。野生型
   相对局部编辑与家族全长生成是不同任务；H2 权重只覆盖前者。
3. **评价与选择。** 序列相容性、候选自身全长结构、NMNAT 催化环境、
   Nudix 扰动；选出机制可描述、假设多样的候选面板。
4. **酶学验证。** 分别测量合成、水解与净积累。

A/B/C 是设计假设：A 改 NMNAT，B 改 Nudix，C 两者都改。

## 仓库内容

| 位置 | 用途 |
|---|---|
| `docs/RESEARCH_PLAN.md` | 研究问题、路线与判断标准 |
| `docs/NATV_PAIRING_AUDIT.md` | 序列、构建体、坐标与同源关系核对 |
| `docs/DATA_PROVENANCE.md` | 数据、模型与证据来源 |
| `contracts/` | 目标、位点规则与评价定义 |
| `inputs/manifest.json` | 固定 SHA256 的源文件清单 |
| `experiments/natv_candidate_sequences/` | 一次探索性局部编辑运行及其报告 |
| `src/natvqvae/` | 当前研究路线用到的可复用代码 |
| `tests/` | 对实际能力和科学约定的测试 |

本仓库没有训练模型；一次限定范围的候选运行使用旧 H2 权重做局部编辑与
注释，结果与限制见
[`REPORT.md`](experiments/natv_candidate_sequences/REPORT.md)。

## 安装与检查

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m natvqvae doctor --root /path/to/nad_qvae
```

`doctor` 用 `inputs/manifest.json` 核对本地源文件，缺失或校验不一致即
失败；`inputs/link_inputs.py --check-only --root /path/to/nad_qvae` 只检查
不建链。源数据不在本仓库中。

贡献边界见 `AGENTS.md`。
