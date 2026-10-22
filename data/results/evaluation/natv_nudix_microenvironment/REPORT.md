# NatV Nudix 金属催化微环境：D2/D4 条件性结构初探

日期：2026-10-23。范围：冻结方案（`docs/RESEARCH_PLAN.md`）问题 1–2。只做条件性结构证据，不预测活性、稳定性或 NAD⁺ 净产量；未进入 MD。

## 输入与方法

- 序列：WT、B1（E248Q）、B2（D308N），来自 `data/results/generation/natv_dual_domain_design/candidates.fasta`；MSA 为 512 行 WT 源 MSA，查询行替换为对应序列。
- D2 的化学组成：AMP + 3 个 Mn；催化簇数量与位点映射参考 2QJT B 链 B505/B506/B507（AMP 参照 B602；B509 与 A 链额外 Mn 不计入），但 `microenv-build` 只写 CCD 组成与数量，不迁移 2QJT 原子坐标，坐标由 Boltz 预测。D4：NAD⁺ + 3 个 Mg，金属数量为假说。两者均以 pocket 约束将配体锚定到 Nudix 盒（KVP40 232/233/248/252/307/308）。
- 无约束对照：同一输入去掉 pocket 约束，检查配体是否自主落在 Nudix 位点。
- Boltz-2 2.2.1，`/mnt/e/models/boltz2/boltz2_conf.ckpt`（SHA256 前缀 090e82ac8c92f5e9），RTX 5090 D；seeds 20261023/20261024，每个 seed 3 个 diffusion samples；sampling 200、recycling 3。共 54 个模型：D2/D4 各 3 序列 × 2 seeds × 3 样本（约束态）+ 3 序列 × 1 seed × 3 样本（无约束对照）。
- 指标：各残基 O/N 原子到最近金属的距离；L9（KVP40 302–314）CA 在 Nudix 对齐后的 RMSD；NAD⁺ 姿态在 Nudix 对齐后的 RMSD；磷酸 O–金属距离。逐模型数据见 `metrics.csv`，运行身份见 `summary.json`。

## 位点验证

- D2 约束态：AMP 质心到 Nudix 域内 CA 的最近距离 5.7–6.3 Å、到 NMNAT 域内 CA 14.0–15.2 Å；WT 的最近侧链 O–金属距离为 E248 2.05–2.19 Å、D308 2.11–2.93 Å（`metrics.csv` 的 `E248_O_min_A`/`D308_O_min_A`），数值上与 2QJT 的约 2.2 Å 量级接近，但最近距离不是完整金属网络的复现。D307 在模型中有直接配位而晶体证据为水介导，且模型无显式水。该距离只是配体质心到域内 CA 的最近值，不能单独确定口袋姿态；位点由 pocket 约束定义。
- D4 约束态：NAD⁺ 质心到 Nudix 域内 CA 4.1–5.2 Å、到 NMNAT 13.3–15.9 Å；以 WT_D4_model_0 为参考、按 Nudix 域 CA 对齐后，NAD⁺ 全部共同重原子的 RMSD 为 WT 0.0–1.57 Å（中位 1.02，含 seed2）、B1 0.85–5.24 Å（中位 3.15）、B2 0.65–3.37 Å（中位 1.83）；约束态磷酸氧到最近金属 1.61–1.99 Å（O2A/O2N 等，见 `metrics.csv`）。该 RMSD 只表示自预测样本间的一致，不是实验正确性；无约束时 NAD⁺ 落 NMNAT 位点，RMSD 约 29 Å，不用于比较 Nudix 姿态。
- 无约束对照：AMP 在 9/9 模型中落在 NMNAT 位点（质心到 NMNAT 域内 CA 5.4–5.5 Å），NAD⁺ 在 9/9 模型中落在 NMNAT 位点（3.8–3.9 Å）。因此 D2/D4 的配体姿态是**约束条件态**，不是模型自主预测的位点；无约束对照中金属位点出现相似距离模式，但这不能证明结果独立于约束、真实 NAD⁺ 环境或金属簇维持。

## 突变效应（按最近金属的描述性距离，不建立原子一一对应）

约束态，每序列 6 个模型（2 seeds × 3 样本）。WT Asp/Glu 的羧酸 O 取 OD1/OD2 或 OE1/OE2 到最近金属的最小值；B2 Asn 取酰胺羰基 O（OD1），B1 Gln 取酰胺羰基 O（OE1）。酰胺 N（ND2/NE2）单列但不作为等价供体。数值为范围（中位数）：

| 状态 | WT 羧酸 O 最近距离 | B1 Q248 酰胺羰基 O（OE1） | B2 N308 酰胺羰基 O（OD1） |
|---|---|---|---|
| D2 E248/Q248 | 2.05–2.19 (2.15) | 2.56–2.86 (2.75) | — |
| D4 E248/Q248 | 1.90–2.53 (2.31) | 2.08–3.23 (3.07) | — |
| D2 D308/N308 | 2.11–2.93 (2.63) | — | 5.41–5.97 (5.65) |
| D4 D308/N308 | 2.31–2.85 (2.73) | — | 2.92–6.42 (5.42) |

- 最近金属的链标签（Boltz 离子链 C/D/E）可互换，不能把标签变化解释为物理位点变化。上表只读作“到最近簇金属的距离”，不能读作“对某个特定金属位点的配位变化”；三金属簇内的具体位点归属尚未建立。
- B2 的 OD1 是酰胺羰基 O，不是羧酸 O；ND2 是酰胺 N（D2 3.21–3.75 Å；D4 2.97–4.82 Å），不能当作等价供体。B2 的 OD1 距离在多数模型中增加，但三金属簇内位点归属未建立，D4 还有 2.92 Å 的接近样本。
- B1 的 OE1 是酰胺羰基 O；NE2 是酰胺 N（D2 2.51–3.49 Å；D4 2.44–3.31 Å），不能当作等价供体。
- 由于三金属簇内位点归属未建立、无显式水且 D307 直接配位与晶体不一致，**不把这些距离差概括为定量的配位效应**；只保留“在约束模型中，突变体的酰胺羰基 O 到最近簇金属的距离通常高于 WT 羧酸 O 的最近距离”这一描述。
- 其他配位：E252 OE 1.68–2.19 Å、D307 OD2 1.90–3.60 Å、G232 主链 O 1.87–2.93 Å、核苷酸磷酸 O 约 1.6–2.2 Å。D307 在模型中有直接配位（OD2），而 Huang 2008 的晶体证据是水介导、D307 仅“may”广义碱；模型可能高估 D307 的直接配位，且模型无显式水，金属网络化学尚不可靠。
- 静态样本中 L9 CA RMSD（按 Nudix 域 CA 对齐后）0.71–0.85 Å；这只是静态预测样本的域内测量，不能写“全程无重排”，也不能单独排除域间重排。
- 置信度：D2 全部 27 个模型/状态（18 约束 + 9 无约束）pTM 0.941–0.955、ligand ipTM 0.912–0.953、complex pLDDT 0.922–0.946；D4 全部 27 个模型/状态 pTM 0.927–0.950、ligand ipTM 0.883–0.967、pLDDT 0.914–0.951。置信度不等于机制正确性。

## 对冻结方案四行问题的回答

1. **E248Q/D308N 是否扰动 Nudix 金属催化环境？** 现有模型不足以回答。约束模型中突变体的酰胺羰基 O 到最近簇金属的距离通常高于 WT 羧酸 O 的最近距离（B2 D2 5.41–5.97 Å vs WT 2.11–2.93 Å；D4 2.92–6.42 Å vs 2.31–2.85 Å；B1 D2 2.56–2.86 Å vs 2.05–2.19 Å；D4 2.08–3.23 Å vs 1.90–2.53 Å），但三金属簇内位点归属未建立（C/D/E 标签可互换），且无显式水、D307 直接配位与晶体不一致，不能把距离差概括为配位扰动或功能效应。
2. **是否与 NAD⁺ 水解环境有关？** D4 是约束条件态，没有实验 NAD⁺ 结合构象；无约束时 NAD⁺ 落在 NMNAT 位点。只能在“模型给出的 Nudix–NAD⁺ 假说姿态”下描述距离变化，不能当作真实 NAD⁺ 水解环境。
3. **是否间接影响 NMNAT？** 本轮没有建立可判定的 NMNAT 状态配对，也未评估域间组织；不能把结构保持解释为合成活性保持。
4. **是否提高 NAD⁺ 净生成？** 本轮不能回答，仍需同条件酶学。

## 实现与可复现性

- 规范化入口：`src/natvqvae/microenvironment.py`，CLI 为 `microenv-build`、`microenv-boltz`、`microenv-summarize`；配置为 `config/microenvironment.yaml`。`microenv-build` 从 candidates、源 MSA、2QJT 与状态定义重建全部输入；`microenv-boltz` 按记录的 seeds/settings 运行；`microenv-summarize` 从现有模型重算 `metrics.csv` 与 `summary.json`。
- 54 个模型为 D2/D4 各 18 个约束态 + 各 9 个无约束对照。现有模型在 CLI 规范化之前产生，但输入、命令、seed、checkpoint 与汇总口径已由上述入口固定；本次只重跑 `microenv-summarize`，未重跑预测。
- 正式汇总命令：`PYTHONPATH=src python -m natvqvae.cli microenv-summarize --config config/microenvironment.yaml --output data/results/evaluation/natv_nudix_microenvironment`。输出 `metrics.csv` 54 行；每行含 `model_path` 与 `model_sha256`，字段区分 E248/D308 的侧链 O 与酰胺 N，并记录最近金属标签；汇总同时校验每 run 3 个样本、模型序列、配体身份与离子种类/数量，`summary.json` 给出各状态/序列的数值范围。
- 运行命令模板：`BOLTZ_CACHE=/mnt/e/models/boltz2 /home/yhshy/miniconda3/envs/boltz2/bin/boltz predict <yaml> --out_dir <dir> --accelerator gpu --devices 1 --diffusion_samples 3 --sampling_steps 200 --recycling_steps 3 --seed <seed> --output_format mmcif --num_workers 2`。

## 边界与停止决定

- D307 在 Boltz 模型中经 OD2 直接配位金属（1.9–3.6 Å），而 Huang 2008 的晶体证据是水介导、D307 仅“may”广义碱；模型可能高估 D307 的直接配位。模型没有显式水，无法评估金属–水网络，因此“金属簇维持”不能作为结论。
- 金属与核苷酸参数未做力场验证；D2/D4 的配体姿态是约束条件态；结果属条件性结构证据，不能解释为活性、稳定性或净产量。
- 按冻结方案，**未进入 MD**：GROMACS 可用不等于 12-6 金属参数对蛋白羧酸/多金属中心可信；短轨迹的配位寿命更可能反映参数与初态，而不是突变效应，不能构成新增判别证据。若后续进入 MD，前置条件是已验证的金属/核苷酸参数、显式水与质子化，以及预先定义的稳定性判据。
- AF3 数据库尚未完成（`_DOWNLOAD_COMPLETE` 不存在，下载脚本仍在运行），本报告未使用 AF3；按“无新增判别问题即停”，不默认追加 Protenix/AF3 或 MD。

## 文件

- `inputs/`：12 份 MSA 与 Boltz YAML（D2/D4、约束/无约束）。
- `build.json`：配置、candidates、源 MSA、2QJT 与 12 份输入的 SHA256。
- 各运行目录下的 mmCIF 与 confidence JSON（不作为小型结果提交）。
- `metrics.csv`：54 个模型的原子级指标；`summary.json`：工具、权重、seed、运行范围与指标列。
- 正式入口：`src/natvqvae/microenvironment.py`；配置：`config/microenvironment.yaml`。
