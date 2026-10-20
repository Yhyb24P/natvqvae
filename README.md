# NatV-QVAE | NAD⁺ 净生成酶设计

研究 KVP40 NatV 衍生的 NMNAT–Nudix 双功能酶，目标是在纯化酶体系中提高
NMN + ATP 生成 NAD⁺ 的净积累。允许 Nudix 水解下降或丢失，前提是 NMNAT
合成保留到具有实际应用意义的程度。目前没有变体经实验证明优于野生型。

## 研究路线

1. **数据。** 使用来源可追溯、结构域对应明确、训练与验证边界清楚的
   NatV 家族和同源实验结构。
2. **生成。** 优先复用已有 QVAE 实现，与同任务无潜变量基线公平比较，以结果确定后续生成方式。
3. **评价与选择。** 序列相容性、候选自身全长结构、NMNAT 催化环境、
   Nudix 扰动；选出机制可描述、假设多样的候选面板。
4. **酶学验证。** 分别测量合成、水解与净积累。

研究终点、证据层级、设计假设与比较标准见
[研究计划](docs/RESEARCH_PLAN.md)。

## 仓库内容

| 位置 | 用途 |
|---|---|
| `docs/RESEARCH_PLAN.md` | 研究问题、路线与判断标准 |
| `docs/NATV_PAIRING_AUDIT.md` | 序列、构建体、坐标与同源关系核对 |
| `docs/DATA_PROVENANCE.md` | 数据群体、划分与模型来源 |
| `contracts/` | 目标、位点规则与评价定义 |
| `inputs/manifest.json` | 固定 SHA256 的源文件清单 |
| `experiments/natv_candidate_sequences/` | 探索性局部编辑运行 |
| `experiments/natv_wt_columns/` | WT 对齐列先验生成比较、产物与报告 |
| `src/natvqvae/` | 当前研究路线用到的可复用代码 |
| `tests/` | 对实际能力和科学约定的测试 |

已有运行及其限制见[局部编辑报告](experiments/natv_candidate_sequences/REPORT.md)和
[WT 对齐列生成比较](experiments/natv_wt_columns/REPORT.md)。

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
