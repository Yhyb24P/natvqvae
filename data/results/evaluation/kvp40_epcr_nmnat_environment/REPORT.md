# ePCR WT/NatV10/NatV89 的 NMNAT 底物环境比较

日期：2026-10-23。对象：KVP40 NatV WT、易错 PCR 克隆 NatV10/NatV89 的既有 9 个全长 apo 预测模型（每条序列 3 个 Boltz-2 样本），以及 1F9A、1NUS、2QJO 三个实验来源的配体姿态。目的：在不重跑预测、不做 MD/对接的前提下，检查两克隆的 NMNAT 突变是否在共同参考系中改变底物/产物附近的侧链环境。本报告只作结构描述，不推断活性、稳定性、底物亲和力或 NAD⁺ 产量。

## 结论摘要

- 9 个模型、三个来源状态中，没有看到 NatV10/NatV89 的 NMNAT 底物口袋发生一致、可分离的重组。五个 NMNAT 突变的新增或丢失侧链原子，在共同 WT0 框架下都没有直接接触所迁移的配体。
- NatV10 的接触状态与 WT 最接近：1F9A ATP 的 `supported_consistent` 为 WT 16、NatV10 16；2QJO NAD 为 WT 30、NatV10 31。D83V 失去羧基、M162V 失去硫醚是序列事实，但在这些姿态中侧链最近配体距离仍为 7–21 Å，不能据此说它们直接改变底物结合。
- NatV89 的 1F9A ATP 和 2QJO POP 接触更多落在 clash 或 with-clash：ATP `supported_consistent` 仅 2（WT 16），POP 为 0（WT 1、NatV10 5），POP 有 20/24 个接触至少一个模型 clash。该差异并非三个模型一致（1F9A 主要由 model 0 驱动，model 2 恢复较多接触），也不发生在突变侧链本身；应作为“F137Y 与 R138/T135 环区局部耦合”的可检验假说，不能当作已证实的功能破坏。
- 1NUS 的全局对齐最弱（TM-score 0.58–0.68，TMalign RMSD 3.25 Å），NMN 接触大多数为 `role_mismatch`；1NUS 的 NMN/APC 结果不足以判断 NMN 口袋或 K140′ 界面。
- 锚点 H14、H17、T135、R138 的 CA 变化均小于 0.5 Å，侧链–配体距离仍在 WT 自身附近；没有证据显示锚点重排。所有“距离近”都不等于氢键、配位或催化几何。

## 输入与身份

### 预测模型

- 9 个模型来自既有 ePCR 全长结构运行：Boltz-2 2.2.1，同一 512 行 WT 源 MSA，seed 20261025，每条序列 3 个 diffusion samples，200 sampling steps，3 recycling steps，全长 341 aa、无配体单体。
- WT：`WT_WT_model_{0,1,2}.cif.gz`；NatV10：`NatV10_NatV10_model_{0,1,2}.cif.gz`；NatV89：`NatV89_NatV89_model_{0,1,2}.cif.gz`。
- 以 WT model 0 为公共参考系。每个模型的 NMNAT 4–181 CA 经 Kabsch 叠合到 WT0；叠合 RMSD：WT 0、1.177、1.116 Å；NatV10 1.108、1.152、0.996 Å；NatV89 1.294、1.189、1.085 Å。该 RMSD 是模型间骨架相似度，不是实验精度。
- 突变身份：NatV10 为 D83V、M162V、L317P、M323K；NatV89 为 I21N、F137Y、I156T、P301S、V320E（I187I 为同义，不计）。序列 SHA256 见 `provenance.json`。

### 实验来源与配体

| 状态 | 来源 | 配体 | 结构条件 | 对齐到 WT0 |
|---|---|---|---|---|
| 1F9A | *Methanococcus jannaschii* NMNAT | ATP（Mg²⁺ 未进入接触表） | 2.0 Å，hexamer；链 A 1–164 | TM 0.840/0.781；TMalign RMSD 2.11 Å；拟合 1.96 Å，156 对 |
| 1NUS | 人胞质 NMN/NaMN 腺苷酰转移酶 | NMN + APC（ATP 类似物） | 2.2 Å，tetramer；链 A 3–234；无沉积 Mg | TM 0.584/0.678；TMalign RMSD 3.25 Å；拟合 2.31 Å，135 对 |
| 2QJO | *Synechocystis* 双功能 NadM 的 N 端 NMNAT 域 | NAD + POP | 2.6 Å，hexamer；链 A 1–181 | TM 0.844/0.858；TMalign RMSD 1.94 Å；拟合 1.53 Å，162 对 |

- 2QJT（AMP/Mn 位于 Nudix 域）为负对照，不进入本分析。
- 每个来源只做一次 source→WT0 的 TMalign/Kabsch；配体原子随该变换迁移一次，之后在 9 个模型中保持同一姿态。这样比较的是“同一实验姿态下的蛋白环境”，不是每个模型各自重新对接。
- 1NUS 的 NMN 与 APC 是同一实验结构中的两个配体；1F9A 的 ATP/Mg、2QJO 的 NAD/POP 同理。三个状态在图中分成三个独立面板，不表示同时结合，也不构成一个三元复合物。

## 判定口径

- 接触：配体原子–蛋白重原子距离 ≤4.0 Å；金属用 ≤3.2 Å。1F9A 的 Mg 在 3.2 Å 内没有蛋白接触，故接触表只有 ATP。
- 每个模型–接触行先判定：`unmapped`；`uncertain_mapping`（映射几何置信度低）；`not_recovered`（目标原子距离 > 阈值）；`clash`（局部或全局 <2.0 Å）；`role_mismatch`（没有可支持的对应原子/基团）；`supported_candidate`（角色匹配且 ≤4.0 Å）；`geometric_only`（几何邻近但角色不匹配）。
- 跨三个模型汇总：`supported_consistent` = 3/3 支持且无 clash；`supported_majority` = 2/3 支持且无 clash；`model_dependent_supported` = 1/3 支持且无 clash；`supported_majority_with_model_clash` / `model_dependent_supported_with_clash` = 有支持但至少一个模型 clash；`clash` = 无支持且至少一个 clash；`role_mismatch`、`not_recovered`、`unmapped`、`uncertain_mapping` 按名称。
- `supported_consistent` 不是“3 个独立实验验证”，也不是完整相互作用。它包含主链–主链接触；行数不是独立接触数。近邻与 clash 可以同时为真，只有严重冲突才被计为 clash。
- `ca_median_A` 是相对固定 WT0 的 CA 距离；WT 自己包含恒为 0 的参考模型，因此 `wt_ca_min/max` 只是三个 WT 模型相对 WT0 的描述范围。`models_outside_wt_ca_range` 只表示超出该描述范围，不是同 clone 自基线，也不是突变效应。克隆自身样本基线引用既有 `data/results/evaluation/kvp40_epcr_full_length_structures/mutation_local.csv` 与 `mutation_local_baseline.csv`（domain-fit，9 个跨序列模型对）。
- 配体距离是固定迁移姿态下的最近侧链重原子距离，不是对接分数、结合自由能或亲和力。3 个 diffusion 样本的 min–max 是描述性范围，不是置信区间。

## 接触结果

每个状态的源接触数：1F9A 61、1NUS 77（其中 NMN 31、APC 46）、2QJO 101（NAD 77、POP 24）；三个克隆使用同一组接触定义。

| 状态 | 克隆 | 3/3 无 clash | 2/3 无 clash | 1/3 无 clash | 有支持但含 clash | 全 clash | role mismatch | not recovered | uncertain/unmapped |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1F9A | WT | 16 | 0 | 1 | 17 | 1 | 14 | 12 | 0 |
| 1F9A | NatV10 | 16 | 0 | 3 | 15 | 4 | 14 | 9 | 0 |
| 1F9A | NatV89 | 2 | 0 | 0 | 32 | 2 | 14 | 11 | 0 |
| 1NUS | WT | 7 | 1 | 1 | 12 | 2 | 42 | 8 | 4 |
| 1NUS | NatV10 | 9 | 0 | 3 | 9 | 3 | 42 | 7 | 4 |
| 1NUS | NatV89 | 4 | 0 | 0 | 14 | 3 | 42 | 10 | 4 |
| 2QJO | WT | 30 | 4 | 7 | 19 | 11 | 12 | 18 | 0 |
| 2QJO | NatV10 | 36 | 4 | 10 | 15 | 7 | 12 | 17 | 0 |
| 2QJO | NatV89 | 30 | 0 | 3 | 16 | 24 | 12 | 16 | 0 |

### 1F9A ATP/Mg

- WT 与 NatV10 都保持 16 个 `supported_consistent`，接触集合几乎相同；NatV10 的 3/3 支持没有增加，也没有出现新的稳定接触。
- NatV89 只剩 2 个 `supported_consistent`，32 个接触含 clash（11 个 2/3 支持但第三模型 clash，21 个 1/3 支持且其他模型 clash）。额外冲突集中在 ATP 的磷酸/核糖接触：KVP40 R9、A134、T135、R138、H17 附近（来源 1F9A R8、G126、T127、R130、H16）。这些位置不在 I21N/F137Y/I156T 的侧链上；F137Y 与 R138 相邻，且 NatV89 的 T135/R138 CA 相对 WT0 为 0.29/0.26 Å，因此“F137Y 局部 packing 影响 R138/T135 的 ATP 磷酸接触”是一个可检验假说，不是已证明的机制。
- 该差异并非三模型一致：NatV89 model 0 在 1F9A 仅有 6 个支持接触，model 1 为 25，model 2 为 18；model 2 在多数上述接触上恢复为 `supported_candidate`。因此不能把 NatV89 的 ATP 接触计数当作稳定的克隆性质。

### 1NUS NMN/APC

- 1NUS 是三个来源中全局对齐最弱者。每个克隆都有 42 个 `role_mismatch`；NMN 的 31 个接触中 24 个为 role mismatch、3 个 unmapped、2 个 not recovered，三个克隆几乎没有稳定支持。NMN 结果不能解释为“NMN 口袋未保留”，只能说明当前刚性映射/原子角色规则无法判别。
- APC 的 46 个接触中，WT/NatV10 各有 7/8 个 `supported_consistent`，NatV89 为 3；NatV89 的 with-clash 为 14（WT 12、NatV10 9）。差异同样不大且模型依赖。
- 本状态不评估 K140′ 界面替代，也不把 APC 的近距离接触当作 ATP 结合姿态。1NUS 的配体姿态继承 3.25 Å 的 TMalign 误差和四聚体界面条件，不能支持更细的化学结论。

### 2QJO NAD/POP

- NAD 接触在三个克隆中保持得最好：`supported_consistent` WT 29、NatV10 31、NatV89 30；说明当前产物态 NAD 部分没有显示克隆特异的口袋重组。
- POP 是差异最大的子状态：`supported_consistent` WT 1、NatV10 5、NatV89 0；NatV89 的 24 个 POP 接触中 14 个全 clash、6 个 with-clash，只有 1 个 model-dependent 支持、3 个 role mismatch。WT 自身也有 7 个全 clash，因此这不是“WT 完美、NatV89 破坏”的干净对照；但 NatV89 的三个模型都表现为低支持（支持数 0/2/5），比单模型离群更值得追踪。
- 2QJO 的 POP 来自细菌双功能 NadM 的产物态，不是 KVP40 的实验结构；POP 的 20/24 冲突不能直接解释为 KVP40 的焦磷酸结合缺陷。它提出的是一个有界问题：NatV89 的局部骨架/侧链变化是否独立地排斥 POP 姿态，还是刚性迁移的产物。

## 突变位点与侧链

下表同时列出本分析的 WT0 相对 CA 值，以及既有 domain-fit 分析中的同位置局部值和同模型自基线。两套数值的叠合方式不同，不能逐位相加或互相替代；自基线列才是判断“是否超出同 clone 预测离散”的依据。

| 位点 | 克隆 | WT0 相对 CA 中位（范围） | 既有 domain-fit 局部中位（范围） | 同位置自基线 | 侧链新/丢失原子到配体的最近距离（1F9A/1NUS/2QJO） | 判定 |
|---|---|---|---|---|---|---|
| D83V | NatV10 | 0.51 (0.34–0.74) | 0.64 (0.30–1.43) | 0.78 | 羧基 OD1/OD2 丢失；最近侧链 21.30/7.37/8.53 Å | 在自基线内；无直接接触 |
| M162V | NatV10 | 0.14 (0.09–0.16) | 0.15 (0.09–0.30) | 0.15 | 硫醚 SD 丢失；最近侧链 19.08/19.55/19.93 Å | 与自基线相当；无直接接触 |
| I21N | NatV89 | 0.27 (0.14–0.28) | 0.23 (0.04–0.31) | 0.24 | ND2/OD1 为 7.47/7.57/7.81 Å | 在自基线内；无直接接触 |
| F137Y | NatV89 | 0.31 (0.25–0.47) | 0.32 (0.25–0.65) | 0.29 | 新 OH 为 7.47/8.67/8.62 Å；环最近原子约 5.9/7.3/6.5 Å | 接近/略高于自基线；OH 不接触配体，不能称新氢键 |
| I156T | NatV89 | 0.65 (0.57–0.66) | 0.54 (0.44–0.66) | 0.18 | 新 OG1 为 10.14/11.39/11.36 Å | 高于自基线；只有局部骨架差异，无直接接触 |

- NatV10 的 D83V/M162V 都不在迁移配体的直接接触范围。D83 在 1F9A 中离 ATP 约 21 Å，在 1NUS/2QJO 中约 7–8 Å；失去羧基没有使侧链更靠近配体，当前数据不能支持“D83 直接配位/识别底物”的说法。
- NatV89 的 I21N、F137Y、I156T 在三个状态中都没有侧链原子进入配体接触距离。F137Y 的新 OH 与配体最近 7.47 Å（1F9A）到 8.67 Å（1NUS）；没有氢坐标、角度或受体类别分析，不能称其为新氢键。
- I156T 的 CA 位移高于同位置自基线（既有 domain-fit：0.54 vs 0.18 Å），说明它是当前模型中较明显的局部骨架差异；但 OG1 离配体 10–11 Å，不能把该骨架差异解释为底物环境改变。

### Nudix 突变（仅作分开记录）

以下位置不属于本报告的 NMNAT 配体状态；本分析的 NMNAT-only 叠合会让 Nudix 的整体域位移混入 CA 数值，不能用于突变效应判断。按既有 domain-fit `mutation_local.csv`/`mutation_local_baseline.csv`：

- L317P：局部 1.17 Å，自基线 0.22 Å；M323K：1.13 Å，自基线 0.21 Å；V320E：1.50 Å，自基线 0.42 Å。三者高于同位置自基线，属于预测模型中的局部骨架差异。
- P301S：0.24 Å，自基线 0.19 Å，未显示超出基线。
- 这些 Nudix 数值与 NMNAT 底物环境比较分开；没有 NAD⁺ 产量、活性或域间耦合证据。

## 锚点与局部环境

| 锚点 | WT CA 中位（范围） | NatV10 CA 中位（范围） | NatV89 CA 中位（范围） | 侧链–配体距离范围（全部状态/克隆） |
|---|---|---|---:|---:|
| H14 | 0.12 (0.00–0.16) | 0.14 (0.13–0.18) | 0.21 (0.17–0.31) | 2.60–3.99 Å |
| H17 | 0.10 (0.00–0.15) | 0.18 (0.18–0.28) | 0.18 (0.06–0.23) | 2.78–3.56 Å |
| T135 | 0.14 (0.00–0.33) | 0.28 (0.18–0.42) | 0.29 (0.18–0.47) | 1.85–2.43 Å |
| R138 | 0.13 (0.00–0.30) | 0.22 (0.12–0.38) | 0.26 (0.17–0.45) | 2.77–3.31 Å |

锚点 CA 变化均小于 0.5 Å；部分模型超出 WT 三模型范围，但幅度小，不能据此称锚点重排。侧链距离在 WT 与克隆之间保持同一量级；T135 的 1.85 Å 属于 distance flag，不是有利氢键或催化几何。当前没有氢、键角、水、金属或质子化状态，不能评价 anchor 的化学角色是否保持。

## 能说明与不能说明

能说明：

- 在这 9 个 apo 模型和三个固定迁移姿态中，五个 NMNAT 突变的新增/丢失侧链原子均未直接接触配体；NatV10 的 ATP/NAD 接触状态与 WT 接近；NatV89 在 ATP 和 POP 上出现更多模型依赖的 clash。
- 锚点 H14/H17/T135/R138 的骨架位置在三个克隆中保持接近；没有看到大幅度的底物口袋重组。
- 1NUS 的 NMN 部分在当前映射下不可判别；2QJO 的 NAD 部分比 POP 部分稳定；POP 的 NatV89 差异值得作为假说追踪。

不能说明：

- 不能说明突变提高或降低活性、稳定性、底物亲和力、催化效率或 NAD⁺ 产量；不能作 NatV10/NatV89 的排名。
- 不能说明真实结合姿态、结合顺序、限速步、诱导契合或金属配位；没有 MD、对接、实验结构或酶学测定。
- 不能把 `supported_consistent` 当作 3 个独立相互作用；它可以是主链接触，也不含氢键/角度/能量。
- 不能把“距离近”当作氢键或催化几何；不能把“clash”直接当作配体不能结合，它只说明刚性迁移到该 apo 模型时当前侧链/环区需要重排。
- 不能把 1NUS 的 NMN/APC 结果外推到 KVP40 的 NMN 口袋或 K140′ 界面；也不能把 2QJO 的 POP 结果直接外推到 KVP40 的 Nudix 催化。
- 3 个 diffusion 样本不是独立实验重复，min–max 不是置信区间；模型间差异不能自动解释为突变效应。

## 可检验问题

1. **NatV89 的 ATP/POP 冲突是否可重复？** 在不改变模型的前提下，独立配体放置或短时 MD 是否能重现 R9/A134/T135/R138 与 ATP 磷酸、以及 POP 的冲突？若不能，按迁移伪影处理；若能，再检验 F137Y 与 R138/T135 环区的局部耦合。当前没有授权或执行这类计算。
2. **D83V 的羧基丢失是否有边缘壳层效应？** 当前三个姿态中 D83 侧链离配体 7–21 Å，不支持直接接触；只有在独立姿态或实验结构中看到 D83 进入配体附近，才值得进一步研究。
3. **1NUS 的映射是否需要更严格的局部对齐？** 当前 TM 0.58–0.68 和大量 role mismatch 使 NMN 结论不可用；若要回答 NMN 口袋问题，需要改进来源选择/局部映射，而不是增加对当前表的解释。

## 文件与复算

运行命令：

```bash
/home/yhshy/miniconda3/envs/nad_data/bin/natvqvae epcr-nmnat \
  --config config/epcr_nmnat_environment.yaml \
  --output data/results/evaluation/kvp40_epcr_nmnat_environment
```

测试：`python -m pytest -q`，16 passed。

- `contact_rows.csv`：2151 行（239 个源接触 × 9 个模型），逐模型接触状态。
- `contact_summary.csv`：717 行（239 × 3 克隆），跨模型汇总、WT 范围、距离差。
- `position_environment.csv`：279 行，突变/锚点的逐模型环境。
- `position_environment_summary.csv`：93 行，按位点/克隆/状态汇总。
- `provenance.json`：输入 SHA256、模型身份、TMalign/Kabsch 指标、源码哈希、命令与图路径。
- `coordinates/`：`reference_WT0.pdb`、`ligands_{1F9A,1NUS,2QJO}.pdb`、9 个共同框架模型 PDB。
- `figures/epcr_nmnat_structure.png`：三个状态分面板；橙色球 = NatV10 D83/M162，品红球 = NatV89 I21/F137/I156，黄/绿/蓝配体 = 1F9A/1NUS/2QJO；每面板仅一个来源状态，不是同时结合。
- `figures/epcr_nmnat_sidechains.png`：D83V、M162V、I21N、F137Y、I156T 五个位点的 WT0 与实际克隆侧链对照（灰=WT0，橙/品红=3 个克隆模型）。
- `figures/epcr_nmnat_differences.png`/`.svg`：WT0 相对 CA 位移，以及三个状态各自独立的最近侧链–配体距离；没有跨状态距离中位。
- `figures/render_structure.pml`、`render_sidechains.pml`：可复算的 PyMOL 渲染脚本。

既有自基线来源：`data/results/evaluation/kvp40_epcr_full_length_structures/mutation_local.csv`、`mutation_local_baseline.csv`（其输入哈希与运行身份见该目录 `build.json`）。本报告没有改写这些文件。
