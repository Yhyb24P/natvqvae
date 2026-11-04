# NatV QVAE

研究噬菌体 KVP40 的 341 aa NMNAT–Nudix 双域 NatV。利用天然序列和演化信息训练条件 Categorical QVAE，从条件先验生成完整变体，再以序列、结构和化学环境证据辅助选择。计算指标不能代替实测酶活性或 NAD⁺ 净积累。14 条蛋白的序列、来源和结构文件见[资料说明](docs/DELIVERY.md)。

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

`data/raw/sequences/natv_population.fasta` 是上游筛选后的 1,871 条训练、验证输入。正式划分在 `data/results/preprocessing/natv_family/`：训练 1,644 条、验证 227 条、测试 231 条。原测试清单只有 ID 和分组；完整测试序列及 CDD 注释已从原始数据恢复到 `data/results/preprocessing/natv_test_recovered/`，来源见其中的 `manifest.json`。原训练输入与划分的校验值见 `config/inputs.sha256`。

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

natvqvae epcr-demo \
  --config config/epcr_clone_demo.yaml \
  --output data/results/evaluation/kvp40_epcr_clone_demo

natvqvae delivery-build \
  --config config/natv_candidate_delivery.yaml \
  --output data/results/evaluation/natv_candidate_delivery

natvqvae nmnat-binding-build --config config/nmnat_binding.yaml \
  --output data/results/evaluation/kvp40_nmnat_substrate_conformations
natvqvae nmnat-binding-run --config config/nmnat_binding.yaml \
  --output data/results/evaluation/kvp40_nmnat_substrate_conformations
natvqvae nmnat-binding-summarize --config config/nmnat_binding.yaml \
  --output data/results/evaluation/kvp40_nmnat_substrate_conformations

natvqvae nmnat-nmn-sources --config config/nmnat_nmn_sources.yaml \
  --output data/results/evaluation/kvp40_nmnat_p2_sources

natvqvae nmnat-binding-build --config config/nmnat_joint.yaml \
  --output data/results/evaluation/kvp40_nmnat_joint
natvqvae nmnat-binding-run --config config/nmnat_joint.yaml \
  --output data/results/evaluation/kvp40_nmnat_joint
natvqvae nmnat-joint-summarize --config config/nmnat_joint.yaml \
  --output data/results/evaluation/kvp40_nmnat_joint
natvqvae nmnat-joint-poses --config config/nmnat_joint.yaml \
  --output data/results/evaluation/kvp40_nmnat_joint
natvqvae nmnat-joint-reference --config config/nmnat_joint.yaml \
  --output data/results/evaluation/kvp40_nmnat_joint
```

`train`、`generate`、`evaluate` 的 `--output` 分别位于 `data/results/training/`、`generation/`、`evaluation/` 下的独立运行目录；新训练的权重另存 `data/weights/qvae/<run>/checkpoint.pt`。`preprocess` 将旧 PF01467-only 8,849 条恢复到 `data/results/preprocessing/pf01467_8849/`。更多参数见 `natvqvae <command> --help`。

NMNAT 辅助域任务使用 `config/aux_qvae.yaml`，入口为 `aux-train`、`aux-generate`、`aux-evaluate`；`aux-summary` 汇总现有两臂结果并选择代表序列。

投稿准备使用 `config/fragment_transfer_comparison.yaml` 记录比较条件。`fragment-overlap --config config/pf01467_fragment_relations.yaml` 检查辅助片段关系；`fragment-pretraining --config config/fragment_pretraining.yaml` 准备外部与自身片段的配对输入；`aux-train` 可用 `config/fragment_aux_external.yaml` 或 `config/fragment_aux_self.yaml`，并通过 `--seed` 指定重复。既有运行目录拒绝覆盖。

`fragment-plot --directory data/results/preprocessing/natv_matched_fragment_pretraining` 从配对清单绘制片段长度与共同映射位置分布。九份模型评价完成后，可用 `fragment-compare --evaluation-dir <同一划分的九份评价目录> --esm-dir <ESM评价目录> --output <比较目录>` 汇总三条件结果；该命令只读取已有分数。

`esm-evaluate --config config/transfer.yaml --output <评价目录>` 评价冻结 ESM-2 预测头；`esm-compare --esm-dir <ESM评价目录> --output <比较目录>` 与原六个迁移模型比较。`evaluate` 和 `esm-evaluate` 默认仅评价验证集；测试评价须明确提供 `--split test --test-data data/results/preprocessing/natv_test_recovered`。恢复原始 Parquet 测试数据另用 `test-recover --config config/natv_test_recovery.yaml`，需要 `python -m pip install -e ".[data]"`，该命令不做预测评分。

`test-history --recovered-dir data/results/preprocessing/natv_test_recovered` 核对旧评价的序列成员和用途，生成 `historical_use.json`。这批序列在旧项目中已有评价记录；测试比较必须同时读取这份历史说明。

结构工作使用独立配置与结果：`design-build/design-boltz/design-summarize`（`config/design.yaml`）、`microenv-build/microenv-boltz/microenv-summarize`（`config/microenvironment.yaml`）和 `epcr-build/epcr-boltz/epcr-summarize`（`config/epcr_full_length.yaml`）。易错 PCR 比较从 Sanger 峰图与 `NatV.dna` 重建 WT/NatV10/NatV89，使用同一 512 行 MSA 与 Boltz-2 设置，结果在 `data/results/evaluation/kvp40_epcr_full_length_structures/`。

`nmnat-binding-build/run/summarize`（`config/nmnat_binding.yaml`）用完整 341 aa WT 做 ATP+1 Mg 与 NMN 共折叠，输出在 `data/results/evaluation/kvp40_nmnat_substrate_conformations/`；A1/A2/A3 的 ATP-only 比较使用同一组入口与 `config/nmnat_atp_variants.yaml`，输出在 `data/results/evaluation/kvp40_nmnat_atp_variants/`。`run` 需要 config 中的 checkpoint；慢速 WSL 存储可用 `--checkpoint-stage-dir` 暂存已校验权重，`--wait-for-pid` 可接续已有任务。`nmnat-nmn-sources`（`config/nmnat_nmn_sources.yaml`）把 1NUS/1NUP/1GZU/1HYB 的 NMN 接触按化学组与来源家族映射，结果在 `data/results/evaluation/kvp40_nmnat_p2_sources/`；它不定义唯一 NMN 姿态。方法与数值见各结果目录的 `REPORT.md`。

`nmnat_joint.yaml` 的联合条件把 ATP 与 NMN 放在同一 WT 输入中（可另加 1 Mg），`nmnat-joint-reference` 只做既有单底物姿态的逐对注册参照，`nmnat-joint-summarize` 输出共同占位、三候选反应几何与金属近邻，`nmnat-joint-poses` 用 NMNAT 4–181 CA 叠合比较联合与同 seed 单底物姿态（全原子与非磷酸 RMSD，附单底物自身采样背景）；联合模型不是固定单底物姿态的简单拼接，这些 RMSD 也不用于活性评价。

`config/nmnat_md.yaml` 把联合模型转成显式水体系并做初始响应 MD：

```
natvqvae nmnat-md-build --config config/nmnat_md.yaml \
  --output data/results/evaluation/kvp40_nmnat_md
natvqvae nmnat-md-em --config config/nmnat_md.yaml \
  --output data/results/evaluation/kvp40_nmnat_md
natvqvae nmnat-md-run --config config/nmnat_md.yaml \
  --output data/results/evaluation/kvp40_nmnat_md --wait-gpu
natvqvae nmnat-md-map --config config/nmnat_md.yaml \
  --output data/results/evaluation/kvp40_nmnat_md
natvqvae nmnat-md-analyze --config config/nmnat_md.yaml \
  --output data/results/evaluation/kvp40_nmnat_md
```

`build` 用 Amber14SB、GAFF2/AM1-BCC、TIP3P、Li/Merz 12-6 Mg²⁺ 和 Na⁺ 中和，为 18 个联合模型各建一个体系；`md_initial_states.csv` 固定 9 个初态（noMg、Mg_NMNAT、Mg_Nudix 各 3 seed，取最低 model index），不按结果换组。`em` 以数值 Fmax 判定收敛。`run` 对每个初态用 2 个速度种子跑 100 ps NVT + 100 ps NPT + 1 ns production（合计 21.6 ns）；NVT/NPT 仅约束蛋白重原子，production 完全释放，ATP/NMN/Mg 全程无位置约束。`map` 写 `atom_mapping.json`（拓扑 atom index ↔ ACPYPE atom name ↔ CCD name，附 SMILES、CCD、参数和 CIF 哈希），后续几何不得按 GROMACS atom name 猜原子。`analyze` 只汇总已完成轨迹，按 PA 中心最小镜像、NMNAT 4–181 CA 配准、水/配体/蛋白氧分类统计，并分列跨配体全原子最小距离与两个磷酸端 O–O 最小距离。1 ns 只描述局部初始响应，不解释为平衡、Mg 交换、占有率或活性。

`config/nmnat_af3.yaml` 用本地 AF3 3.0.4 对完整 WT 做 ATP+NMN 与 ATP+NMN+Mg 的无约束预测，并用 1F9A 单链做技术校准：

```
natvqvae nmnat-af3-build --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3
natvqvae nmnat-af3-run --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3 --wait-gpu
natvqvae nmnat-af3-analyze --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3
```

`build` 使用既有 512 行 WT MSA、`run_data_pipeline=false`、`templates=[]` 和 3 seeds × 3 samples；`analyze` 保留三个 NMN 候选氧、ATP 链内连接、跨配体/蛋白接触、金属近邻，以及 AF3 与既有 Boltz 坐标的蛋白配准姿态比较（C/N 骨架与全重原子对称 RMSD）。`nmnat-af3-apo`（同一配置）另行把既有 BioEmu2 `predict-structure`/`sample-equilibrium` 帧按模式分层比较 102–105 环、锚点残基与 Nudix 域主链；它需要 bioemu2 环境的 mdtraj，结果在 `data/results/evaluation/natv_nmnat_af3/`。结果目录归档 27 个逐样本 CIF.gz（不含重复 best）、两套 apo 的未压缩 `topology.cif`/`samples.xtc` 小输入及 `build.json`/`identity.json`；因此只凭归档即可运行 `nmnat-af3-analyze`/`nmnat-af3-apo` 复算几何与 apo 比较，但归档不含逐样本 confidence，复算不包含 pTM/ipTM（缺失留空，不按 0 处理）。

`nmnat-map`（`config/nmnat_mapping.yaml`）把 1F9A ATP/Mg、1NUS NMN/APC 和 2QJO NAD/POP 的实验接触刚性映射到与 A1–A3 同一 Boltz 运行的 WT 全长模型；只做接触、变换和空间冲突检查，不预测复合物。结果在 `data/results/evaluation/natv_nmnat_state_mapping/`。`nmnat-audit` 对 1F9A ATP/Mg 做原子/基团级化学分类和冲突分列，结果在 `data/results/evaluation/natv_nmnat_atp_audit/`。`epcr-nmnat`（`config/epcr_nmnat_environment.yaml`）复用既有 WT/NatV10/NatV89 9 个全长模型，在共同 WT0 的 NMNAT 框架中比较三种实验配体姿态下的突变侧链环境和接触状态；不重跑预测，不做 MD/对接。结果在 `data/results/evaluation/kvp40_epcr_nmnat_environment/`。输出目录必须为新目录或空目录；已有结果不会被覆盖，重跑请换新目录。

`epcr-demo`（`config/epcr_clone_demo.yaml`）从上述已按 NMNAT 4–181 Cα 对齐的 WT/NatV10/NatV89 model_0 PDB 生成两张独立静图：三 panel 全长总览（域色、各克隆编辑与真实 CA 投影标记）和全长叠合。它不重算指标、不做动画；结果在 `data/results/evaluation/kvp40_epcr_clone_demo/`，同样要求新目录或空目录。

`delivery-build`（`config/natv_candidate_delivery.yaml`）把 12 条设计候选与 2 条 ePCR 克隆合并成 14 条唯一蛋白的交接包，包含统一 CSV/FASTA、来源/身份清单、42 个候选 PDB、3 个 ePCR 批次 WT 参照和必要结构附件；不重算结构或指标。结果在 `data/results/evaluation/natv_candidate_delivery/`，要求新目录或空目录。

现役科学目标与方法见 [RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)，已完成结果与重要负结果见 [RESULTS.md](docs/RESULTS.md)。
