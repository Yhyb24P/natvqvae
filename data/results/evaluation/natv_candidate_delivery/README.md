# NatV 候选交接包（14 条唯一蛋白）

本包把计算设计候选和湿实验 ePCR 克隆合并为一份可独立交接的快照：共同 WT 1 条、计算设计 11 条、ePCR 克隆 2 条，合计 14 条不同的 341 aa 蛋白序列。完整序列、编辑、亲本、来源和 SHA 见 `candidates.csv` / `candidates.fasta`；本页只列身份和编辑，不复制长序列。

## 14 条对象

| ID | 来源类别 | 完整蛋白编辑 | 比较亲本 | 预测批次 |
|---|---|---|---|---|
| WT | shared_reference | none | 两组共同参照 | design_boltz2_seed20261022 |
| A1 | computational_design | L18I;D57S;R94N;M127A | WT | design_boltz2_seed20261022 |
| A2 | computational_design | T22Q;G60L;L111K | WT | design_boltz2_seed20261022 |
| A3 | computational_design | V61H;Q86I;A134G | WT | design_boltz2_seed20261022 |
| B1 | computational_design | E248Q | WT | design_boltz2_seed20261022 |
| B2 | computational_design | D308N | WT | design_boltz2_seed20261022 |
| A1B1 | computational_design | L18I;D57S;R94N;M127A;E248Q | A1 + B1 | design_boltz2_seed20261022 |
| A1B2 | computational_design | L18I;D57S;R94N;M127A;D308N | A1 + B2 | design_boltz2_seed20261022 |
| A2B1 | computational_design | T22Q;G60L;L111K;E248Q | A2 + B1 | design_boltz2_seed20261022 |
| A2B2 | computational_design | T22Q;G60L;L111K;D308N | A2 + B2 | design_boltz2_seed20261022 |
| A3B1 | computational_design | V61H;Q86I;A134G;E248Q | A3 + B1 | design_boltz2_seed20261022 |
| A3B2 | computational_design | V61H;Q86I;A134G;D308N | A3 + B2 | design_boltz2_seed20261022 |
| NatV10 | epcr_clone | D83V;M162V;L317P;M323K | 独立克隆；WT 仅作参照 | epcr_boltz2_seed20261025 |
| NatV89 | epcr_clone | I21N;F137Y;I156T;P301S;V320E | 独立克隆；WT 仅作参照 | epcr_boltz2_seed20261025 |

WT 只保留一行，作为设计和 ePCR 两组比较的共同参照。A1–A3 来自 QVAE 生成，B1/B2 为机制假说设计（不由 QVAE 生成，`source_run` 为空），A+B 为组合；A+B 的 `source_run`/`generation_candidate_id` 继承对应 A，不代表 A+B 自身由 QVAE 直接生成。NatV10/NatV89 是独立 ePCR 克隆，不是 B1/B2。

## 可查看的预测结构（coordinates/ 与 references/）

每条候选 3 个固定 diffusion samples 的全长 apo 预测结构，`coordinates/<ID>_model_{0,1,2}.pdb`，共 14 × 3 = 42 个 PDB，由已跟踪的 `.cif.gz` 归档转换而来，未重新预测。`references/epcr_WT_model_{0,1,2}.pdb` 另存 ePCR 批次的 WT 3 个样本：序列与统一 WT 相同，但属于不同预测批次，仅作额外比较参照，不计入 42 个候选坐标，也不替代设计批次的 WT。全包坐标合计 45 个。

- 12 条设计（含 WT）来自设计批次；NatV10/NatV89 来自 ePCR 批次；两批的 WT 预测坐标不同，不要互相替代，也不要拼接成同一构象。
- 这些是**代表性预测单体**：无配体、不是实验结构、不是候选平均构象，也不能当作活性证据。
- 3 个 diffusion samples 不是独立种子或独立重复；包内图片固定展示 model_0，但 3 个样本都可在 `coordinates/` 中查看，并支持按批次复算原有 RMSD。
- 查看示例：`pymol coordinates/*.pdb`；叠合比较可用 viewer 的 CA 叠合命令，本包不为此重跑渲染或生成新模型。

## 先看什么

1. `candidates.csv` / `candidates.fasta`：14 条序列、编辑、亲本关系和 SHA。
2. `coordinates/`：42 个可打开的全长预测 PDB（14 候选 × 3 samples），另 `references/` 有 3 个 ePCR 批次 WT 参照。
3. `attachments/`：必要的结构图、位点表和 RMSD 表。
4. `evidence/`：原始设计表、Sanger 核查记录和生成来源快照。
5. `manifest.json`：逐文件 SHA-256、来源路径和派生关系。

## 三个问题

- A 类组合是否改变合成相关表现？不能把完整 NatV 的净 NAD⁺ 增加直接称为独立 vNMNAT 活性。
- B1/B2 是否改变 NAD⁺ 水解？结构距离或模型置信度不能替代水解释放读数。
- A 与 B 是否兼容并改善目标表现？应在选定条件下比较 WT/A/B/A+B 的净积累，并结合可获得的合成与水解信息。

**目前没有任何经验证的活性、稳定性或 NAD⁺ 净积累排序**；本包只提供序列身份、亲本关系和预测结构证据。具体方法、对照和待湿实验确认事项见仓库中的 [candidate_handoff.md](https://github.com/Yhyb24P/natvqvae/blob/29907d042c4b567b1aff62ccf5695e917429b2a9/data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md)。

## 附件

| 文件 | 用途 |
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

说明：12 条设计与 NatV10/NatV89 的模型批次、WT 参照和分母不同，不能把两组 RMSD 拼成同一条件排名。

## 证据来源与边界

- `evidence/design_candidates_original.csv`：统一表合并前的原始 12 设计表；`evidence/clone_variants.csv`、`evidence/clone_sanger_verification.json`、`evidence/clone_build.json`：ePCR 克隆的序列、Sanger 核查、序列重建与运行记录。原始 `.ab1` 峰图仍在原始湿实验材料中，不包含在本包内。
- `evidence/design.json`、`evidence/design.yaml`、`evidence/random_seed8301_generation.json`、`evidence/random_seed8303_generation.json`：生成/设计来源快照，用于追溯候选 ID；其中机制措辞是历史设计记录/假说，不表示已验证增强。
- 结构图与坐标来自 Boltz-2 预测，不是实验结构；pLDDT/pTM、Cα RMSD 和局部差异不能证明活性、稳定性或表达。
- 完整 NatV 的 NAD⁺ 净积累不是独立 vNMNAT 活性；A570 的历史终点吸光信号不能用于排序或机制归因。
- 本包不含 DNA/构建、表达纯化、检测 SOP、对照/重复/判定标准或功能结果；这些需湿实验团队确认。

## 详细报告（仓库追溯）

- [设计报告](https://github.com/Yhyb24P/natvqvae/blob/29907d042c4b567b1aff62ccf5695e917429b2a9/data/results/evaluation/natv_dual_domain_design/REPORT.md)
- [ePCR 全长结构比较](https://github.com/Yhyb24P/natvqvae/blob/29907d042c4b567b1aff62ccf5695e917429b2a9/data/results/evaluation/kvp40_epcr_full_length_structures/REPORT.md)
- [ePCR NMNAT 环境比较](https://github.com/Yhyb24P/natvqvae/blob/29907d042c4b567b1aff62ccf5695e917429b2a9/data/results/evaluation/kvp40_epcr_nmnat_environment/REPORT.md)
- [RESEARCH_PLAN.md](https://github.com/Yhyb24P/natvqvae/blob/29907d042c4b567b1aff62ccf5695e917429b2a9/docs/RESEARCH_PLAN.md)

交付时记录 Git commit；序列、身份文件和报告以该 commit 为准。
