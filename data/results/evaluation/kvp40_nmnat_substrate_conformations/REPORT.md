# WT ATP/Mg 与 NMN 共折叠：27 模型分析

日期：2026-10-23。对象：完整 341 aa KVP40 WT；ATP+1 Mg 与 NMN 两组共折叠；同一 512 行 WT MSA、Boltz-2 2.2.1、3 seeds × 3 diffusion samples。本报告只分析实际坐标，不改变预测条件、权重或输入，不做 MD、对接或活性推断。

## 结果摘要

- **ATP 位于 NMNAT 域**：18 个 ATP 模型中，ATP 配体域最近蛋白域全部为 NMNAT（4–181）；ATP–蛋白重原子接触 108–127 个，全部在 NMNAT，未出现 linker/Nudix 接触。仅 1 个模型有 1 个 <2 Å 接触（free seed28 model1，T135 OG1–O3G 1.98 Å）。
- **锚点–磷酸邻近存在**：18 个模型中 H14 2.44–3.51 Å、H17 2.69–3.01 Å、T135 1.98–2.89 Å、R138 3.00–3.56 Å 到 ATP 磷酸氧。约束/无约束范围相近；这些是距离，不是氢键证明（无氢、角度、质子化）。
- **Mg 多数在 Nudix、未与 ATP 共定位**：16/18 模型的 Mg 在 3.2 Å 内没有 ATP 原子，最近蛋白位点在 Nudix 的 E252/E248/G232 附近（4 个蛋白原子近邻）。只有 2/18 模型 Mg 距 ATP 氧 ≤3.2 Å（free seed28 model0 1.93 Å、5 个 ATP 原子近邻；anchor seed27 model2 1.90 Å、4 个 ATP 原子近邻；其中包含 P 等非 O 原子，不能当作配位氧数），且靠近 SER107。不能把 18 个模型统称为 ATP/Mg 结合态；Mg 落在 Nudix 是当前共折叠的放置结果，不是 KVP40 的真实金属偏好或生理位点。
- **ATP 内部键长没有明显异常**：33 个 CCD 键，1.32–1.61 Å，无 <0.9 或 >2.0 Å 的粗筛异常；未做键角、价态、手性或立体化学完整检查。
- **Y102–D105 环**：ATP 模型中环–配体最近 2.03–2.60 Å；环 CA 相对 WT0 apo 为 2.95–3.51 Å。该值只是相对 WT0 的坐标差，不证明真实重排或动力学柔性。
- **NMN**：9 个模型均在 NMNAT 域，NMN–蛋白接触 78–91 个，无 <2 Å 接触；不含 Mg。环 CA 相对 WT0 为 3.08–6.10 Å。NMN 是独立状态，不能与 ATP/Mg 结果合并成三元结论。
- **约束依赖**：anchor 与 unconstrained 的 ATP 域归属、接触数和锚点距离范围相近；当前 pocket 特征（force false）没有产生明显不同的 ATP 组织，不能把约束一致当作自然结合证据。
- **ATP 姿态本身较集中**：以首个 ATP 模型为参考、先叠合 NMNAT 4–181 再比较 31 个 ATP 重原子，18 个模型的 ATP pose RMSD 为 0.27–1.23 Å（排除参考自身；17 个模型；NMN 22 原子为 0.32–1.08 Å，8 个模型；只比较同名重原子，未做对称等价匹配）。这说明 ATP 的蛋白侧姿态在样本间较一致；它不涉及 Mg，也不是实验精度。

## 输入、身份与方法

- 27 模型：ATP_MG 18（unconstrained/anchor × 3 seeds × 3 samples）；NMN 9（unconstrained × 3 seeds × 3 samples）。
- 预测身份：`run_identity.json` 记录实际 checkpoint `/dev/shm/natvqvae_boltz2/boltz2_conf.ckpt`、SHA256 `090e82ac…`、Boltz 2.2.1/torch 2.7.1+cu128、config/build hash；`analysis_identity.json` 记录本次分析源码 hash、参考结构 hash、27 个模型坐标 hash。
- 配体采用本地 Boltz CCD pickle 表示：ATP 31 个重原子、formal charge 0；NMN 22 个重原子、+1（吡啶鎓）；MG +2。形式电荷与质子化是 CCD 预测表示，不是目标溶液离子态或质子化态；三个 pickle 的实际 SHA256 见 `analysis_identity.json` 的 `ccd_pickle_sha256`。
- 域：N_terminal 1–3、NMNAT 4–181、linker 182–203、Nudix 204–338、C_terminal 339–341。配体域取配体原子中心最近蛋白原子所属域。
- 锚点：H14 ND1/NE2、H17 ND1/NE2、T135 OG1、R138 NE/NH1/NH2；磷酸基团按 CCD 邻接：PA O1A/O2A/O3A/O5′，PB O1B/O2B/O3B/O3A，PG O1G/O2G/O3G/O3B（桥氧可属两组）。
- 所有重原子对 ≤4.0 Å 写入 `heavy_atom_contacts.csv`；不是氢键或完整接触网络。Mg 3.2 Å 内全部蛋白/核苷酸原子写入 `metal_neighbors.csv`；无显式水，不能称完整配位层。
- ATP 键检查使用本地 CCD ATP 键图；只检查键长粗异常，未做角度/价态/手性。

## 逐项结果

### 1. ATP 的蛋白侧

- ATP 最近蛋白域：18/18 NMNAT。
- ATP–蛋白接触数：unconstrained 110–127；anchor 108–126。
- 接触域计数：NMNAT 108–127；linker/Nudix/terminal 0。
- 唯一 severe 接触：free seed20261028 model1，T135 OG1–O3G 1.98 Å；不是所有模型共有的冲突。
- ATP 的 CCD 键长：33 个键，1.32–1.61 Å，无粗筛 outlier。

### 2. H14/H17/T135/R138 与磷酸氧

| 锚点 | unconstrained 最小距离范围 (Å) | anchor 最小距离范围 (Å) |
|---|---:|---:|
| H14（ND1/NE2） | 2.44–2.93 | 2.52–3.51 |
| H17（ND1/NE2） | 2.69–2.88 | 2.80–3.01 |
| T135（OG1） | 1.98–2.82 | 2.27–2.89 |
| R138（NE/NH1/NH2） | 3.04–3.56 | 3.00–3.36 |

- 这些是锚点原子到磷酸 O 的最近距离；多数在常见氢键距离范围，但没有氢坐标、角度、质子化/互变异构分析，不能称为氢键或磷酸识别。
- 约束与无约束的范围相近；没有看到约束把某一锚点显著拉近或推远的系统效应。

### 3. Mg 的定位

- 16/18 模型：Mg 的 3.2 Å 内**没有核苷酸原子**；只有 4 个蛋白原子近邻，集中在 Nudix 的 E252（32 个记录）、E248（16）、G232（16）。
- 2/18 模型：Mg 距 ATP O ≤3.2 Å：
  - free seed20261028 model0：Mg–ATP O 1.93 Å，5 个 ATP 原子近邻（含 P 等非 O 原子），最近蛋白 SER107 OG 3.07 Å；
  - anchor seed20261027 model2：Mg–ATP O 1.90 Å，4 个 ATP 原子近邻（含 P 等非 O 原子），最近蛋白 SER107 OG 2.65 Å。
- 其余 16 个模型的 Mg–ATP O 约 26.9–28.4 Å；它们不是 ATP/Mg 复合物姿态。Mg 的蛋白近邻有 1.57–1.90 Å 的偏短记录（E252 等），本身也需要谨慎解释，不能因 ATP–蛋白无 severe 接触就说整个复合物化学合理。
- 因此：ATP 的蛋白侧可以单独讨论；ATP/Mg 的金属配位在当前 18 个模型中未建立。两个 Mg 接近 ATP 的模型是少数、seed 依赖的候选，不是可重复的配位证据。

### 4. Y102–D105 环

- ATP 模型环–配体最近 2.03–2.60 Å；环接触原子对 14–16（≤4 Å）。
- 环 CA 相对 WT0 apo：2.95–3.51 Å。该差异说明共折叠模型与 apo 参考的局部位置不同，不能说明真实重排必须发生或柔性大小。
- NMN 模型环–配体最近 2.72–6.91 Å；CA 相对 WT0 3.08–6.10 Å，作为独立状态记录。

### 5. NMN

- 9/9 模型 NMN 最近蛋白域 NMNAT；接触 78–91，全部 NMNAT，无 severe；不含 Mg。
- 没有 H14/H17/T135/R138 与磷酸氧的对应（NMN 只有磷酸但本报告不把这些 ATP 锚点规则套到 NMN）；NMN 的局部化学需单独报告。
- 当前 NMN 自由预测可作为独立构象假说，不与 ATP/Mg 合并成三元复合物。

## 能说明与不能说明

能说明：

- 在这些 co-folded 模型中，ATP 可以落入 NMNAT 域，并与 H14/H17/T135/R138 的侧链保持 2–3.6 Å 的磷酸氧邻近；大多数模型没有 ATP–蛋白 <2 Å 冲突。
- Mg 在 16/18 模型被放到 Nudix 的 E252/E248/G232 附近，而不是 ATP；ATP 与 Mg 的共定位不是当前预测的稳定结果。该分布只说明本次共折叠的放置行为，不证明 KVP40 金属位点在 Nudix，也不证明其偏好 Nudix。
- 解释条件：输入是完整双域 WT + ATP + 单个 Mg，未指定 Mg 归属；1F9A 的一枚 Mg 只提供 NMNAT 局部参照，不证明完整 NatV 的总金属占位。16/18 的频率同时受单 Mg 组成、两个潜在金属相关口袋和模型先验影响，不是金属占有率，也不能判为模型定位失败。
- 两例 Mg–ATP 近邻模型提供了可检查的候选，但数量少、seed 依赖，不能作为验证。

不能说明：

- 不能把这 18 个模型称为完整 ATP/Mg²⁺ 催化复合物；没有水、角度、质子化、金属配位数或实验验证。
- 不能把锚点距离称为氢键/磷酸识别，也不能用 ligand_iptm/pTM 覆盖金属未共定位。
- 不能把 loop 相对 WT0 的 3.3 Å 当作真实重排或动力学柔性。
- 不能把 NMN 结果与 ATP/Mg 结果合并为三元生产性构象；不能推断活性、亲和力或 NAD⁺ 产量。

## 对下一步的含义

- P1 的可用产物是“ATP 蛋白侧的姿态假说 + Mg 未与 ATP 共定位的负结果”，不是一个完整的 ATP/Mg 结合态。
- 如果继续 ATP 侧比较，应明确使用 **ATP-only**（或把 Mg 作为单独变量）而不能把当前 18 个模型当作 ATP/Mg 复合物；A1/A2/A3 的配对比较应在可用共同状态确定后再做。
- 不建议用更强 Mg 约束直接把 Mg 拉回 ATP 来“修正”结果；这会把待检验的配位位置变成输入假设。是否用 Protenix/独立方法探索 Mg 位点是下一步的方法选择，不是本报告自动追加的计算。

## 文件

- `detailed_metrics.csv`：27 模型逐模型域计数、锚点磷酸距离、Mg 近邻、loop、键长粗检、置信度。
- `heavy_atom_contacts.csv`：全部 ≤4 Å 重原子接触，含域和 severe 标记。
- `anchor_phosphate_distances.csv`：H14/H17/T135/R138 的每个侧链原子–磷酸 O 距离。
- `metal_neighbors.csv`：每个 Mg 的 3.2 Å 内蛋白/核苷酸近邻。
- `ligand_bond_check.csv`：ATP/NMN CCD 键长粗检。
- `ligand_pose_consistency.csv`：以首个模型为参考、叠合 NMNAT 4–181 后的配体重原子 pose RMSD。
- `binding_analysis.png`：Mg 近邻、Mg–ATP、锚点距离、loop 位移。
- `analysis_identity.json`：本次分析源码/参考/27 坐标 hash；预测身份在 `run_identity.json`。
- `coordinates/*.cif.gz` 与 `coordinates_manifest.csv`：提交版保留 27 个确定性 gzip 坐标，记录原始/压缩 SHA256、字节数与运行内路径；`runs/` 的未压缩副本只留在本地。
- 另有单样本 ATP/Mg 技术验证（`/tmp/nmnat_binding_smoke`，1 model）和 3 个 WT ATP-only 探索模型（`data/results/evaluation/kvp40_nmnat_atp_only_trial/`）；均不计入本报告 27 模型，也不作为独立验证。路径、配置/输入/结果 hash 与实际 sample 数见 `analysis_identity.json` 的 `preliminary_runs`。
