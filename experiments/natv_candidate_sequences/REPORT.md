# NatV 候选序列生成与结构注释

本次运行沿用 nad_qvae 中已经训练好的两组 H2 模型参数，生成 32 条全长 KVP40 NatV 候选序列，并把每处替换标注到结构域和此前核对过的 12 个关键残基上。模型做的是以野生型为起点、固定改动 4 或 8 个位点的局部编辑，不是从头生成全长序列。

## 运行范围

- H2_no_latent：SHA256 `6bdecac9f55604a66a84fac0090626bff55b4bf0455e363d201de3cd5c784e06`，结构：no_latent
- H2_continuous_mmd：SHA256 `ce12ef55b45dc38696a34776e35553044d19940db01a9cc1670af9422e8d2b6c`，结构：continuous_mmd

- 两组模型共同使用 H2 上游设置、ESM2 词表、1,644 条训练序列和原有划分。
- 编辑方案种子 `20261201`，每种编辑数 8 组方案，编辑数 [4, 8]，每组 4 条序列，共 7 步掩码编辑，温度 1.0，禁止 T135A。
- 使用连续潜在变量的组在每个四序列组开始时抽取一次先验，七步共用；不使用潜在变量的组没有这一分支。
- 共 32 条候选。

## 输入与来源

- H2 上游设置文件 SHA256：`01d9b47efa22a1c89b704effd5885c797287e23c432c2be93cc95c3b95e49a76`。
- 结构注释依据：`discovery/natv_pairing_audit/structural_interpretation_basis.json`，SHA256 `8cc0577feb96bebeb44bf0eb19432f81485ce00320c3835f7f973f6a0902f7d7`。它只用于注释，没有进入训练、同源检索或候选挑选。
- 本次读取的文件和校验值见 `input_files.json`、`input_checks.json`、`sampling_plan_checksums.json`。
- 旧 WP5 复核：24 条候选；ESM2 交叉熵最大绝对差 `4.44089e-16`；CDD 结果行完全一致：`True`。

## 生成结果是否有效

- 32/32 条候选为 341 aa，恰好 4 或 8 处替换。
- 最终替换与生成方案中的位点一致；T135A 违规 0。
- 不同序列数：27/32。两组模型在编辑数为 4 的方案中有 4 对序列相同，编辑数为 8 的方案中有 1 对相同；这些重复保留在所有按候选计数的分母中。
- 与 2,430 条家族角色表逐字符比较，32/32 为新序列；与 1,644 条训练序列比较，32/32 为新序列。
- CDD 同时命中 NMNAT 和 NUDIX 两个结构域：32/32。
- WT 分 8 段遮蔽的 ESM2 交叉熵：1.865023817（参考 1.865023817；差 0）。

## 各组结果

| source_set | model_group | plan_type | edit_count | n_candidates | n_unique_sequences | n_new_family_2430 | n_CDD_double_domain | n_ESM2_below_WT | median_ESM2_CE | n_no_direct_key_residue_change | n_NMNAT_key_residue_changed | n_NUDIX_key_residue_changed | n_T135_changed | n_H236_changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| new_candidates | H2_continuous_mmd | 固定编辑位点 | 4 | 8 | 8 | 8 | 8 | 2 | 1.88299 | 5 | 0 | 3 | 0 | 0 |
| new_candidates | H2_continuous_mmd | 固定编辑位点 | 8 | 8 | 8 | 8 | 8 | 1 | 1.89536 | 3 | 0 | 5 | 0 | 0 |
| new_candidates | H2_no_latent | 固定编辑位点 | 4 | 8 | 8 | 8 | 8 | 1 | 1.88055 | 5 | 0 | 3 | 0 | 0 |
| new_candidates | H2_no_latent | 固定编辑位点 | 8 | 8 | 8 | 8 | 8 | 1 | 1.90027 | 3 | 0 | 5 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 固定编辑位点 | 4 | 2 | 2 | 2 | 2 | 1 | 1.87142 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 固定编辑位点 | 8 | 2 | 2 | 2 | 2 | 0 | 1.91415 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 自适应位点 | 4 | 2 | 2 | 2 | 2 | 1 | 1.85692 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 自适应位点 | 8 | 2 | 2 | 2 | 2 | 1 | 1.89174 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_S | 固定编辑位点 | 4 | 2 | 2 | 2 | 2 | 0 | 1.87865 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_S | 固定编辑位点 | 8 | 2 | 2 | 2 | 2 | 1 | 1.89016 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_S | 自适应位点 | 4 | 2 | 2 | 2 | 2 | 1 | 1.87363 | 1 | 1 | 0 | 1 | 0 |
| previous_WP5 | previous_S | 自适应位点 | 8 | 2 | 2 | 2 | 2 | 0 | 1.88263 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_SE | 固定编辑位点 | 4 | 2 | 2 | 2 | 2 | 1 | 1.8742 | 1 | 0 | 1 | 0 | 0 |
| previous_WP5 | previous_SE | 固定编辑位点 | 8 | 2 | 2 | 2 | 2 | 0 | 1.90658 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_SE | 自适应位点 | 4 | 2 | 2 | 2 | 2 | 0 | 1.88503 | 2 | 0 | 0 | 0 | 0 |
| previous_WP5 | previous_SE | 自适应位点 | 8 | 2 | 2 | 2 | 2 | 1 | 1.87328 | 2 | 0 | 0 | 0 | 0 |

表中的 `previous_WP5` 为旧候选复核结果。

## 结构域上的替换

| source_set | model_group | edit_count | domain | n_candidates | n_candidates_with_change | n_changes | n_direct_key_residue_changes | n_changes_within_4_positions |
|---|---|---|---|---|---|---|---|---|
| new_candidates | H2_continuous_mmd | 4 | LINKER | 8 | 4 | 4 | 0 | 0 |
| new_candidates | H2_continuous_mmd | 4 | NMNAT | 8 | 8 | 13 | 0 | 3 |
| new_candidates | H2_continuous_mmd | 4 | NUDIX | 8 | 8 | 15 | 3 | 3 |
| new_candidates | H2_continuous_mmd | 8 | LINKER | 8 | 6 | 6 | 0 | 0 |
| new_candidates | H2_continuous_mmd | 8 | NMNAT | 8 | 8 | 28 | 0 | 4 |
| new_candidates | H2_continuous_mmd | 8 | NUDIX | 8 | 8 | 30 | 5 | 5 |
| new_candidates | H2_no_latent | 4 | LINKER | 8 | 4 | 4 | 0 | 0 |
| new_candidates | H2_no_latent | 4 | NMNAT | 8 | 8 | 13 | 0 | 3 |
| new_candidates | H2_no_latent | 4 | NUDIX | 8 | 8 | 15 | 3 | 3 |
| new_candidates | H2_no_latent | 8 | LINKER | 8 | 6 | 6 | 0 | 0 |
| new_candidates | H2_no_latent | 8 | NMNAT | 8 | 8 | 28 | 0 | 4 |
| new_candidates | H2_no_latent | 8 | NUDIX | 8 | 8 | 30 | 5 | 5 |
| previous_WP5 | previous_ESM2 | 4 | NMNAT | 2 | 2 | 4 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 4 | NUDIX | 2 | 2 | 4 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 8 | NMNAT | 2 | 2 | 11 | 0 | 1 |
| previous_WP5 | previous_ESM2 | 8 | NUDIX | 2 | 1 | 5 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 4 | LINKER | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 4 | NMNAT | 2 | 2 | 4 | 0 | 1 |
| previous_WP5 | previous_ESM2 | 4 | NUDIX | 2 | 2 | 3 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 8 | CTERM | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 8 | NMNAT | 2 | 2 | 9 | 0 | 0 |
| previous_WP5 | previous_ESM2 | 8 | NUDIX | 2 | 2 | 6 | 0 | 0 |
| previous_WP5 | previous_S | 4 | NMNAT | 2 | 2 | 4 | 0 | 0 |
| previous_WP5 | previous_S | 4 | NUDIX | 2 | 2 | 4 | 0 | 1 |
| previous_WP5 | previous_S | 8 | NMNAT | 2 | 2 | 8 | 0 | 0 |
| previous_WP5 | previous_S | 8 | NUDIX | 2 | 2 | 8 | 0 | 2 |
| previous_WP5 | previous_S | 4 | CTERM | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_S | 4 | NMNAT | 2 | 2 | 5 | 1 | 1 |
| previous_WP5 | previous_S | 4 | NUDIX | 2 | 2 | 2 | 0 | 1 |
| previous_WP5 | previous_S | 8 | LINKER | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_S | 8 | NMNAT | 2 | 2 | 10 | 0 | 1 |
| previous_WP5 | previous_S | 8 | NUDIX | 2 | 2 | 5 | 0 | 2 |
| previous_WP5 | previous_SE | 4 | LINKER | 2 | 2 | 2 | 0 | 0 |
| previous_WP5 | previous_SE | 4 | NMNAT | 2 | 1 | 2 | 0 | 2 |
| previous_WP5 | previous_SE | 4 | NTERM | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_SE | 4 | NUDIX | 2 | 2 | 3 | 1 | 1 |
| previous_WP5 | previous_SE | 8 | LINKER | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_SE | 8 | NMNAT | 2 | 2 | 8 | 0 | 0 |
| previous_WP5 | previous_SE | 8 | NUDIX | 2 | 2 | 7 | 0 | 3 |
| previous_WP5 | previous_SE | 4 | NMNAT | 2 | 2 | 6 | 0 | 2 |
| previous_WP5 | previous_SE | 4 | NUDIX | 2 | 2 | 2 | 0 | 0 |
| previous_WP5 | previous_SE | 8 | CTERM | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_SE | 8 | LINKER | 2 | 1 | 2 | 0 | 0 |
| previous_WP5 | previous_SE | 8 | NMNAT | 2 | 2 | 7 | 0 | 1 |
| previous_WP5 | previous_SE | 8 | NTERM | 2 | 1 | 1 | 0 | 0 |
| previous_WP5 | previous_SE | 8 | NUDIX | 2 | 2 | 5 | 0 | 0 |

## 已核对的 12 个关键残基

| domain | wt_position | wt_residue | identity_conserved_across_templates | variable_key_residue | new_candidates_changed | new_substitutions | previous_candidates_changed | previous_substitutions |
|---|---|---|---|---|---|---|---|---|
| NMNAT | 14 | H | 是 | 否 | 0 |  | 0 |  |
| NMNAT | 17 | H | 是 | 否 | 0 |  | 0 |  |
| NMNAT | 135 | T | 是 | 否 | 0 |  | 1 | T135S |
| NMNAT | 138 | R | 是 | 否 | 0 |  | 0 |  |
| NUDIX | 232 | G | 是 | 否 | 6 | G232A;G232S | 0 |  |
| NUDIX | 236 | H | 否 | 是 | 0 |  | 0 |  |
| NUDIX | 247 | R | 是 | 否 | 0 |  | 0 |  |
| NUDIX | 248 | E | 是 | 否 | 4 | E248G;E248S | 0 |  |
| NUDIX | 251 | E | 是 | 否 | 0 |  | 0 |  |
| NUDIX | 252 | E | 是 | 否 | 2 | E252K | 0 |  |
| NUDIX | 307 | D | 是 | 否 | 0 |  | 1 | D307S |
| NUDIX | 308 | D | 是 | 否 | 4 | D308E;D308N | 0 |  |

直接改变关键残基的候选共 18 条（本次 16 条 + 旧 WP5 2 条），但只有 15 条不同序列，因为有些候选在两组模型之间完全相同。完整列表见 `structure_analysis/interpretation.json` 的 `direct_key_residue_candidates`。

| candidate_id | source_set | model_group | edit_count | direct_key_residue_changes | n_NMNAT_key_residue_changes | n_NUDIX_key_residue_changes | H236_key_residue_changed | T135_changed | ESM2_blocked_CE | CDD_double_domain |
|---|---|---|---|---|---|---|---|---|---|---|
| H2_no_latent_b4_s03 | new_candidates | H2_no_latent | 4 | G232S | 0 | 1 | 否 | 否 | 1.88767 | 是 |
| H2_no_latent_b4_s05 | new_candidates | H2_no_latent | 4 | E248S | 0 | 1 | 否 | 否 | 1.88631 | 是 |
| H2_no_latent_b4_s06 | new_candidates | H2_no_latent | 4 | D308N | 0 | 1 | 否 | 否 | 1.89396 | 是 |
| H2_no_latent_b8_s02 | new_candidates | H2_no_latent | 8 | G232A | 0 | 1 | 否 | 否 | 1.88591 | 是 |
| H2_no_latent_b8_s03 | new_candidates | H2_no_latent | 8 | G232A | 0 | 1 | 否 | 否 | 1.89617 | 是 |
| H2_no_latent_b8_s04 | new_candidates | H2_no_latent | 8 | E252K | 0 | 1 | 否 | 否 | 1.93871 | 是 |
| H2_no_latent_b8_s05 | new_candidates | H2_no_latent | 8 | E248G | 0 | 1 | 否 | 否 | 1.87208 | 是 |
| H2_no_latent_b8_s06 | new_candidates | H2_no_latent | 8 | D308E | 0 | 1 | 否 | 否 | 1.91084 | 是 |
| H2_continuous_mmd_b4_s03 | new_candidates | H2_continuous_mmd | 4 | G232S | 0 | 1 | 否 | 否 | 1.8863 | 是 |
| H2_continuous_mmd_b4_s05 | new_candidates | H2_continuous_mmd | 4 | E248S | 0 | 1 | 否 | 否 | 1.88631 | 是 |
| H2_continuous_mmd_b4_s06 | new_candidates | H2_continuous_mmd | 4 | D308N | 0 | 1 | 否 | 否 | 1.89396 | 是 |
| H2_continuous_mmd_b8_s02 | new_candidates | H2_continuous_mmd | 8 | G232A | 0 | 1 | 否 | 否 | 1.87876 | 是 |
| H2_continuous_mmd_b8_s03 | new_candidates | H2_continuous_mmd | 8 | G232A | 0 | 1 | 否 | 否 | 1.90264 | 是 |
| H2_continuous_mmd_b8_s04 | new_candidates | H2_continuous_mmd | 8 | E252K | 0 | 1 | 否 | 否 | 1.94077 | 是 |
| H2_continuous_mmd_b8_s05 | new_candidates | H2_continuous_mmd | 8 | E248G | 0 | 1 | 否 | 否 | 1.87208 | 是 |
| H2_continuous_mmd_b8_s06 | new_candidates | H2_continuous_mmd | 8 | D308E | 0 | 1 | 否 | 否 | 1.90776 | 是 |
| previous:20261084_S_adaptive_4_010 | previous_WP5 | previous_S | 4 | T135S | 1 | 0 | 否 | 是 | 1.88789 | 是 |
| previous:20261085_SE_matched_4_022 | previous_WP5 | previous_SE | 4 | D307S | 0 | 1 | 否 | 否 | 1.89738 | 是 |

### NMNAT 催化环境

H14、H17、T135、R138 的直接替换落在 2QJO（NAD⁺ 产物结合状态）已核对的 ATP 结合基序或催化邻近区域；其他 NMNAT 结构域内的替换只按结构域归类。本次直接改变 NMNAT 关键残基的候选：0/32 条（0 条不同序列）；旧 WP5：1/24 条（1 条不同序列）。T135 改变：本次 0/32（无）；旧 WP5 1/24（T135S）。T135A 违规 0。

### Nudix 水解

G232、H236、R247、E248、E251、E252、D307、D308 的直接替换落在 2QJT（AMP + Mn²⁺ 状态）已核对的 Nudix 基序或金属-水网络附近。H236 的模板差异见 `docs/NATV_PAIRING_AUDIT.md`。本次直接改变 NUDIX 关键残基的候选：16/32 条（13 条不同序列）；旧 WP5：1/24 条（1 条不同序列）。H236 改变：本次 0/32（无）；旧 WP5 0/24（无）。

## 解释限制

两组模型共用训练设置，但结构和参数量不同；单一种子、有限条数和单一亲本下的描述性计数不能分离潜在变量效应，也不足以判断方法优劣。旧 WP5 的 24 条序列是此前筛选后的样本，不能作为独立新样本比较。

结构注释来自既有序列—模板对应；候选自身的全长结构、催化方向和净 NAD⁺ 生成均为 UNKNOWN。“相距不超过 4 个序列位点”是序列邻近，不是三维距离。ESM2 交叉熵与其预训练分布相关，CDD 表示家族注释相容性；证据层级见 `docs/RESEARCH_PLAN.md`。

## 如何重跑

执行范围与预算要求见 `AGENTS.md`。

```bash
export PATH=/home/yhshy/miniconda3/envs/nad_data/bin:$PATH
export NATV_SOURCE_ROOT=/home/yhshy/git_files/nad_qvae
python experiments/natv_candidate_sequences/run_candidates.py --stage preflight
python experiments/natv_candidate_sequences/run_candidates.py --stage sample
python experiments/natv_candidate_sequences/run_candidates.py --stage evaluate
```

## 文件说明

- `sampling_plan.json`、`sampling_plan_checksums.json`、`input_checks.json`、`input_files.json`、`run_status.json`
- `candidate_sequences.csv`、`sampling_steps.jsonl.gz`、`homolog_search_records.json.gz`、`homolog_search_summary.csv`
- `evaluation/ESM2_scores.csv`、`evaluation/CDD_domain_hits.csv`、`evaluation/CDD_repeat_check.csv`、`evaluation/results_summary.json`
- `structure_analysis/interpretation.json`
- 原始 MMseqs 输出在 `homolog_search/` 下，仅本地保留，不提交。
