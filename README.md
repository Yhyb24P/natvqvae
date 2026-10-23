# NatV QVAE

研究噬菌体 KVP40 的 341 aa NMNAT–Nudix 双域 NatV。利用天然序列和演化信息训练条件 Categorical QVAE，从条件先验生成完整变体，再以序列、结构和化学环境证据辅助选择。计算指标不能代替实测酶活性或 NAD⁺ 净积累。

## 目录

```text
config/                 研究目标、数据来源及运行配置
data/raw/               不可变的序列、实验结构与参考文件
data/weights/           ESM-2 与 NatV 权重（本地，不提交）
data/results/
  preprocessing/        数据清洗、正式划分及必要特征
  training/             训练记录及必要报告
  generation/           候选序列和生成记录
  evaluation/           模型、结构与候选评价
src/natvqvae/           正式实现及 CLI
tests/                  现役功能测试
docs/                   研究计划与有效结果
AGENTS.md               项目贡献、文件组织和写作规范
```

结果按具体数据集或运行分开存储。权重和大批计算产物默认不提交 Git；重要报告、图表和小型正式结果可选择性提交。

## 数据与权重

`data/raw/sequences/natv_population.fasta` 是上游筛选后的 1,871 条固定输入，不是全部 2,430 条发现序列。正式划分在 `data/results/preprocessing/natv_family/`：训练 1,644 条、验证 227 条；封存测试仅保留 231 个 ID。输入与划分的校验值见 `config/inputs.sha256`。

`config/sources.yaml` 统一引用 `data/weights/esm2/` 及 `data/weights/qvae/`；现役 QVAE 的五组 `seed*/best.pt` 权重由本地放置，不进入 Git。迁移既有权重仅改变路径，内容依 `config/checkpoints.sha256` 核对。

## 使用

```bash
python -m pip install -e ".[dev]"
natvqvae doctor
pytest -q

natvqvae generate \
  --checkpoint data/weights/qvae/seed8301/best.pt \
  --group NMNAT_m08_r00 --draws 16 \
  --output data/results/generation/kvp40_nmnat_example

natvqvae preprocess --config config/pf01467.yaml

natvqvae nmnat-map \
  --config config/nmnat_mapping.yaml \
  --output data/results/evaluation/natv_nmnat_state_mapping

natvqvae nmnat-audit \
  --config config/nmnat_mapping.yaml \
  --output data/results/evaluation/natv_nmnat_atp_audit

natvqvae epcr-nmnat \
  --config config/epcr_nmnat_environment.yaml \
  --output data/results/evaluation/kvp40_epcr_nmnat_environment
```

`train`、`generate`、`evaluate` 的 `--output` 分别位于 `data/results/training/`、`generation/`、`evaluation/` 下的独立运行目录；新训练的权重另存 `data/weights/qvae/<run>/checkpoint.pt`。`preprocess` 将旧 PF01467-only 8,849 条恢复到 `data/results/preprocessing/pf01467_8849/`。更多参数见 `natvqvae <command> --help`。

NMNAT 辅助域任务使用 `config/aux_qvae.yaml`，入口为 `aux-train`、`aux-generate`、`aux-evaluate`；`aux-summary` 汇总现有两臂结果并选择代表序列。

结构工作使用独立配置与结果：`design-build/design-boltz/design-summarize`（`config/design.yaml`）、`microenv-build/microenv-boltz/microenv-summarize`（`config/microenvironment.yaml`）和 `epcr-build/epcr-boltz/epcr-summarize`（`config/epcr_full_length.yaml`）。易错 PCR 比较从 Sanger 峰图与 `NatV.dna` 重建 WT/NatV10/NatV89，使用同一 512 行 MSA 与 Boltz-2 设置，结果在 `data/results/evaluation/kvp40_epcr_full_length_structures/`。

`nmnat-map`（`config/nmnat_mapping.yaml`）把 1F9A ATP/Mg、1NUS NMN/APC 和 2QJO NAD/POP 的实验接触刚性映射到与 A1–A3 同一 Boltz 运行的 WT 全长模型；只做接触、变换和空间冲突检查，不预测复合物。结果在 `data/results/evaluation/natv_nmnat_state_mapping/`。`nmnat-audit` 对 1F9A ATP/Mg 做原子/基团级化学分类和冲突分列，结果在 `data/results/evaluation/natv_nmnat_atp_audit/`。`epcr-nmnat`（`config/epcr_nmnat_environment.yaml`）复用既有 WT/NatV10/NatV89 9 个全长模型，在共同 WT0 的 NMNAT 框架中比较三种实验配体姿态下的突变侧链环境和接触状态；不重跑预测，不做 MD/对接。结果在 `data/results/evaluation/kvp40_epcr_nmnat_environment/`。输出目录必须为新目录或空目录；已有结果不会被覆盖，重跑请换新目录。

现役科学目标与方法见 [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)，已完成结果与重要负结果见 [RESULTS.md](docs/RESULTS.md)。
