# NatV 蛋白序列与结构资料

本资料包包括野生型（WT）、11 条设计变体和 NatV10、NatV89 两条易错 PCR 克隆，共 14 条不同的蛋白序列，每条均为 341 个氨基酸。WT 是两组研究的共同序列对照。

## 蛋白来源与氨基酸替换

编号从 WT 的第 1 个氨基酸开始。例如，L18I 表示第 18 位亮氨酸被异亮氨酸替换。

| 名称 | 来源 | 完整氨基酸替换 | 比较对象 |
|---|---|---|---|
| WT | 野生型 | 无 | — |
| A1 | QVAE 生成序列 | L18I;D57S;R94N;M127A | WT |
| A2 | QVAE 生成序列 | T22Q;G60L;L111K | WT |
| A3 | QVAE 生成序列 | V61H;Q86I;A134G | WT |
| B1 | 依据同源酶机制设计 | E248Q | WT |
| B2 | 依据同源酶机制设计 | D308N | WT |
| A1B1 | A 与 B 的组合 | L18I;D57S;R94N;M127A;E248Q | WT、A1、B1 |
| A1B2 | A 与 B 的组合 | L18I;D57S;R94N;M127A;D308N | WT、A1、B2 |
| A2B1 | A 与 B 的组合 | T22Q;G60L;L111K;E248Q | WT、A2、B1 |
| A2B2 | A 与 B 的组合 | T22Q;G60L;L111K;D308N | WT、A2、B2 |
| A3B1 | A 与 B 的组合 | V61H;Q86I;A134G;E248Q | WT、A3、B1 |
| A3B2 | A 与 B 的组合 | V61H;Q86I;A134G;D308N | WT、A3、B2 |
| NatV10 | 易错 PCR 克隆 | D83V;M162V;L317P;M323K | WT |
| NatV89 | 易错 PCR 克隆 | I21N;F137Y;I156T;P301S;V320E | WT |

A1–A3 的生成记录、B1/B2 的设计依据以及两条克隆的 Sanger 测序核查记录保存在 `evidence/`。A+B 是相应 A、B 变体的组合，并非另一次 QVAE 生成的结果。NatV89 另有 DNA 同义变化 561c>t（对应 I187I），不改变蛋白序列，单独记录在 CSV 的 DNA 变化栏。

## 文件怎么看

| 文件或文件夹 | 内容和用途 |
|---|---|
| `candidates.csv` | 14 条完整序列、所有氨基酸替换、来源、亲本和序列校验值（SHA-256） |
| `candidates.fasta` | 同一组 14 条序列，可导入序列分析软件 |
| `coordinates/` | 每条蛋白 3 个预测结构，共 42 个 PDB 文件 |
| `references/` | 易错 PCR 研究中另外预测的 3 个 WT 结构，供复算该批比较结果 |
| `evidence/` | 原始序列表、生成和设计记录、Sanger 测序核查记录 |
| `attachments/` | 全长结构图、突变位点图和 RMSD 结果 |
| `manifest.json` | 文件来源、模型编号和校验值，用于确认文件与原资料一致 |

PDB 文件可用 PyMOL 或 ChimeraX 打开。文件名中的 `model_0`、`model_1`、`model_2` 是同一条件下的 3 个预测样本，不是实验重复。所有 PDB 均保留原预测坐标，没有重新拟合或松弛。

42 个候选结构包括设计集合的 36 个结构和 NatV10/NatV89 的 6 个结构。`references/` 中的 3 个 WT 来自易错 PCR 研究的另一批预测；它们与设计集合的 WT 序列相同，但坐标不同。复算时应使用原比较所对应的 WT。

RMSD 表示相应区域的 Cα 原子经刚体叠合后的位置差异，单位为 Å。跨序列比较使用 3×3＝9 个模型对；这些模型对不是 9 次独立实验。设计集合与易错 PCR 克隆分别按各自预测批次比较，具体数值见附件。

## 研究这些变体要回答什么

- A1–A3：NMNAT 域的氨基酸替换是否改变 NAD⁺ 合成能力？
- B1/B2：Nudix 域的 E248Q、D308N 是否减少 NAD⁺ 水解？
- A+B：两类替换同时存在时，是否保留合成能力，并增加指定反应条件下的 NAD⁺ 积累？

**目前没有任何经验证的活性排序。**结构接近不能证明活性保持或增强。完整 NatV 同时合成和水解 NAD⁺，因此 NAD⁺ 净积累不能直接当作 NMNAT 的独立合成速率。现有 A570 终点吸光记录也不足以判断这些蛋白的活性高低。

## 图和测序证据

| 文件 | 内容 |
|---|---|
| [`design_wt_overview.png`](attachments/design_wt_overview.png) | 12 条设计候选在 WT 上的位点总览 |
| [`design_a_class_panels.png`](attachments/design_a_class_panels.png) | A1/A2/A3 同框架比较 |
| [`design_nudix_site.png`](attachments/design_nudix_site.png) | Nudix 位点局部图 |
| [`design_nmnat_inset.png`](attachments/design_nmnat_inset.png) | NMNAT ATP/NMN 联合模型局部图 |
| [`design_parent_comparison.png`](attachments/design_parent_comparison.png) | A+B 对 parent A 的 RMSD 描述图 |
| [`clone_overview.png`](attachments/clone_overview.png) | WT/NatV10/NatV89 三 panel 总览 |
| [`clone_overlay.png`](attachments/clone_overlay.png) | WT/NatV10/NatV89 全长叠合（NMNAT 4-181 配准） |
| [`design_mutation_map.csv`](attachments/design_mutation_map.csv) | 设计位点映射 |
| [`design_candidate_structure_summary.csv`](attachments/design_candidate_structure_summary.csv) | 设计候选结构汇总 |
| [`design_comparison_summary.csv`](attachments/design_comparison_summary.csv) | 设计候选比较表 |
| [`clone_mutation_sites.csv`](attachments/clone_mutation_sites.csv) | 克隆编辑位点和 aligned CA 坐标 |
| [`clone_rmsd_summary.png`](attachments/clone_rmsd_summary.png) | ePCR 克隆全长/域内局部位移汇总 |
| [`clone_nmnat_environment.png`](attachments/clone_nmnat_environment.png) | ePCR 克隆 NMNAT 环境比较图 |

结构图和 PDB 均来自预测。NMNAT 配体局部图另取自 WT 的 ATP/NMN 联合预测，不属于上述 42 个无配体单体结构。

`evidence/clone_sanger_verification.json` 保存测序位点与读段质量的核查结果；`clone_variants.csv` 保存克隆序列和替换，`clone_build.json` 保存序列重建与计算记录。原始 `.ab1` 测序峰图未放入本包，仍保存在原始实验材料中。其余文件记录 A 类序列的生成来源及 B 类设计依据；这些设计依据仍是假说，尚未证明能提高活性。

## 需要详细方法时查阅

- [设计与结构比较](https://github.com/Yhyb24P/natvqvae/blob/a9afa30c0f9d87421b24745e51982c6f1aaa5782/data/results/evaluation/natv_dual_domain_design/REPORT.md)
- [NatV10/NatV89 全长结构比较](https://github.com/Yhyb24P/natvqvae/blob/a9afa30c0f9d87421b24745e51982c6f1aaa5782/data/results/evaluation/kvp40_epcr_full_length_structures/REPORT.md)
- [NatV10/NatV89 的 NMNAT 底物环境](https://github.com/Yhyb24P/natvqvae/blob/a9afa30c0f9d87421b24745e51982c6f1aaa5782/data/results/evaluation/kvp40_epcr_nmnat_environment/REPORT.md)
- [功能比较的说明](https://github.com/Yhyb24P/natvqvae/blob/main/data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md)
