# NatV 双域 A/B/A+B 设计：结构兼容性评价

## 设计

| 组 | 父本 | 来源 run / candidate | 完整编辑 | Hamming | 理由 |
|---|---|---|---|---|---|
| A1 | — | random_seed8301 / QVAE_s8301_NMNAT_m08_r11_d14 | L18I;D57S;R94N;M127A | 4 | L18I motif 邻近 |
| A2 | — | random_seed8301 / QVAE_s8301_NMNAT_m08_r02_d22 | T22Q;G60L;L111K | 3 | T22Q N 端 motif 邻近 |
| A3 | — | random_seed8303 / QVAE_s8303_NMNAT_m08_r12_d30 | V61H;Q86I;A134G | 3 | A134G SXTXXR motif |
| B1 | — | WT | E248Q | 1 | 同源 E250 接触 AMP/ADPR/金属；去电荷假说 |
| B2 | — | WT | D308N | 1 | 同源 D308 金属配位；去电荷假说 |

A1/A2/A3 均无 Nudix 变化、无 NMNAT 锚点改变；它们是完整多突变组合，不能单归因 L18I/T22Q/A134G。F13Y 在池中仅 1 条 26 编辑且含 T135I 锚点改变，列为风险备选，不进入 3A；F13Y+A134G 为 0。A+B 只增加对应 B 单点。

## Boltz-2 与序列核对

- Boltz 2.2.1，torch 2.7.1+cu128；checkpoint `/mnt/e/models/boltz2/boltz2_conf.ckpt`，SHA256 `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1`。
- 12 条 ×3 diffusion samples = 36 个单体全长预测；固定 seed 20261022、sampling_steps 200、recycling_steps 3、mmcif；512 行冻结 MSA；无 MSA server、配体、多聚体、对接/MD。
- 12/12 成功，0 失败，总 543.2 s（9.1 min）；WT 是正式 36 个之一。
- 36 个模型 CIF 均核对 chain A 序列与设计序列一致，逐文件 SHA 与期望序列 SHA 记录在 `structure_summary.json` 的 `structure_sequence_checks`；`tool_identity` 保存版本/checkpoint/SHA。

## 结构结果

- 置信度：候选 pLDDT 中位 0.914–0.925，pTM 0.902–0.913；WT 0.922/0.913。
- 同一候选 3 个样本自身全长 RMSD 中位：WT 1.04 Å；候选 0.72–1.48 Å。修正 MSA 后样本离散明显低于错误批次，但 3 个样本仍只作描述，不能称为收敛。
- A+B 对同一 A/B/WT 的配对中位 RMSD：全长 0.73–0.99 Å；NMNAT 0.78–1.14 Å；Nudix 0.22–0.35 Å。A+B 的结构变化很小，未观察到一致的“新增风险”或结构优势；3 个样本仍只作描述。

按 NMNAT CA 对齐后的锚点局部指标（3×3 模型配对为描述）：

| 比较 | H14/H17/T135/R138 中位位移 (Å) | 10–24 邻域 RMSD | 130–141 邻域 RMSD |
|---|---:|---:|---:|
| B1 vs WT | 0.13/0.14/0.27/0.16 | 0.17 | 0.21 |
| B2 vs WT | 0.13/0.16/0.23/0.20 | 0.20 | 0.25 |
| A1B1 vs A1 | 0.19/0.15/0.37/0.23 | 0.18 | 0.30 |
| A1B2 vs A1 | 0.17/0.16/0.37/0.30 | 0.16 | 0.33 |
| A2B1 vs A2 | 0.16/0.12/0.45/0.31 | 0.15 | 0.40 |
| A2B2 vs A2 | 0.13/0.09/0.36/0.25 | 0.14 | 0.32 |
| A3B1 vs A3 | 0.18/0.18/0.19/0.15 | 0.25 | 0.31 |
| A3B2 vs A3 | 0.20/0.20/0.15/0.14 | 0.24 | 0.33 |
| WT 自身样本 | 0.23/0.18/0.31/0.23 | 0.24 | 0.25 |

B 与 A+B 的锚点/邻域差异落在父本自身样本离散附近，不能据此判断活性、稳定性或催化几何改善。

模板叠合（按 Nudix CA 拟合同源参考链）fit RMSD：NMNAT 2.9–4.3 Å；Nudix 1.2–1.6 Å；全长 2.6–3.6 Å。该叠合可用于有边界的模板对应，但不是 KVP40 实验结构或复合物预测。

## 源实验位点与 assembly

同源结构不是 KVP40 NatV。源原子记录及 assembly 解析如下：

| 源 PDB | 生物学组装 | 源 chain (entity / label / auth) | 源残基 (label_seq / auth_seq) | KVP40 WT | B 对应 |
|---|---|---|---|---|---|
| 2QJO | hexamer, count 6, oper 1,2 | A (1 / A / A) | GLU 250/248 | E248 | ft E250 |
| 2QJO | hexamer | A | GLU 254/252 | E252 | ft E254 |
| 2QJO | hexamer | A | D308 未映射 | — | — |
| 2QJT | dimer, count 2 | B (1 / A / B) | GLU 255/250 | E248 | B1 E248Q ↔ ft E250Q |
| 2QJT | dimer | B | GLU 259/254 | E252 | ft E254 |
| 2QJT | dimer | B | ASP 313/308 | D308 | B2 D308N ↔ ft D308N |
| 2R5W | dimer, count 2 | B (1 / A / B) | GLU 255/250；GLU 259/254；ASP 313/308 | E248/E252/D308 | 同上 |

因此用户提到的 ft **E250/E254/D308** 对应 WT **E248/E252/D308**；不能把同源 auth 编号直接当 KVP40 编号。

距离为源结构重原子（不含 H）最小距离；配体阈值 4.5 Å。2QJT 的 E250/E254/D308 距 MN 2.22/2.17、2.01/2.40、2.48/2.21 Å，距 AMP 3.42/3.05/2.94 Å；2R5W 对应位点距 MG 2.1–3.6 Å；2QJO E248/E252 距 APR 2.91/2.66 Å。按实际 assembly 变换，2QJO hexamer 的 B 位点到其他 copy 的最小距离约 11.1–13.8 Å；2QJT/2R5W dimer 到另一链约 10.5–13.4 Å。这些位点本身不在同源组装界面；但 Nudix 底物口袋的 adenine 夹在 Phe234 与另一单体 Tyr197_B 之间，Nudix 功能环境整体涉及另一亚基，KVP40 是否二聚未知。

候选叠合到 2QJT 的 B 位点中位距离：E248 到另一链约 10.1–10.5 Å、到配体约 2.5–2.8 Å；D308 到另一链约 10.5–13.4 Å、到配体约 3.2–4.1 Å。A+B 与 A/B/WT 的差异很小；这只说明模板邻近性，不是 KVP40 实际几何、活性或稳定性。

## 结论与边界

1. 36 个正式单体结构均与设计序列一致；A+B 的域骨架与 NMNAT 锚点邻域未显示超出父本样本离散的明显变化。单链预测和同源模板叠合不足以排除实际组装风险。
2. B1/B2 是 E→Q/D→N 去电荷假说；结构只显示它们在模板中邻近 AMP/金属，不能证明水解下降。A 的多编辑组合不能单归因于 L18I/T22Q/A134G。
3. 同源 Nudix 的 B 位点不在实验二聚体界面上，但 Nudix 口袋整体涉及另一亚基；KVP40 组装未知，不能用单体预测或模板距离替代实验。
4. 3 个 diffusion 样本只作描述；无配体结合预测、MD、对接或亲和力排名。所有活性、稳定性、kcat/Km、NAD⁺ 净积累仍 UNKNOWN。
5. 下一步是纯化 WT/A/B/A+B，在相同缓冲和底物条件下测 vNMNAT、vNudix、vnet 时间曲线及目标时间 NAD⁺ 产量；不能用各自最适条件速率相减。

## 复现

- 输入：`data/results/generation/natv_dual_domain_design/` 的 `candidates.fasta/csv`、`design.json` 与唯一 `source_msa_WT_512.a3m` 副本；12 份 A3M/YAML 可由 `design-build` 从这些输入重建，不提交重复副本。
- 结构：已提交 36 个 `.cif.gz`，命名 `<name>_<原model文件名>.cif.gz`；解压到 `<name>/boltz_results_<name>/predictions/<name>/<原model文件名>` 即可由 `design-summarize` 复算。36 份 `confidence_*.json` 已按原预测目录提交，不需要 GPU。汇总为 `structure_summary.json`、`structure_pairs.csv`、`boltz_run_summary.json`、`boltz_runs.jsonl`。
- 重建 12 个输入时，用一份 config 副本把 `msa_source` 指向 `source_msa_WT_512.a3m` 后运行 `design-build`；不修改历史 `design.json`/config SHA。
- 统一入口：`natvqvae design-build`、`design-boltz`、`design-summarize`；配置 `config/design.yaml`。技术失败记录：`TECHNICAL_FAILURES.json`。

## 技术失败记录

第一次 A3M 读取将小写插入大写化，造成 36 个预测作废，耗时约 575 s；另一次输入构建在首行校验时中止。错误批次已清除，只保留 `TECHNICAL_FAILURES.json` 中的原因、范围、参数、校验值与成本。上述结果不用于研究判断。

参考文献：
- Lee, Li & Miller 2017, J Bacteriol 199:e00855-16: https://pmc.ncbi.nlm.nih.gov/articles/PMC5388814/ ; DOI 10.1128/JB.00855-16
- Huang et al. 2008, Structure 16:196-209: https://pmc.ncbi.nlm.nih.gov/articles/PMC2258087/ ; DOI 10.1016/j.str.2007.11.017
- RCSB 2QJO: https://www.rcsb.org/structure/2QJO ; 2QJT: https://www.rcsb.org/structure/2QJT ; 2R5W: https://www.rcsb.org/structure/2R5W
- Mildvan et al. 2005: https://doi.org/10.1016/j.abb.2004.08.017
