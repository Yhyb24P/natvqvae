# ePCR WT/NatV10/NatV89 的 NMNAT 底物环境比较

日期：2026-10-23。对象：KVP40 NatV WT、易错 PCR 克隆 NatV10/NatV89 的既有 9 个全长 apo 预测模型（每条序列 3 个 Boltz-2 样本），以及 1F9A、1NUS、2QJO 三个实验来源的配体姿态。目的：在不重跑预测、不做 MD/对接的前提下，检查两克隆的 NMNAT 突变是否在共同参考系中改变底物/产物附近的侧链环境。本报告只作结构描述，不推断活性、稳定性、底物亲和力或 NAD⁺ 产量。

## 结论摘要

- 9 个模型、三个来源状态中，没有看到 NatV10/NatV89 的 NMNAT 底物口袋发生一致、可分离的重组。五个 NMNAT 突变的新增或丢失侧链原子，在共同 WT0 框架下都没有直接接触所迁移的配体。
- NatV10 的 ATP/NAD 接触状态与 WT 接近：1F9A ATP 的 `supported_consistent` 为 WT 16、NatV10 16、NatV89 2；2QJO NAD 为 WT 29、NatV10 31、NatV89 30；2QJO POP 为 1/5/0。
- NatV89 在 1F9A ATP 和 2QJO POP 上的冲突记录更多，但实际最近原子不是源接触所映射的 R9/H17/R138 等残基。1F9A ATP 的 55 个 clash 行去重为 29 个 `(model, ligand atom, protein atom)` 对，主要是 TYR102/ASP103/LYS104/ASP105，另有 ALA134 主链 N 一个；2QJO POP 的 44 行去重为 20 对，含 SER133/THR135、LYS104、ASP105。据此撤回“F137Y 与 R138/T135 耦合”的优先解释；它没有直接结构证据，本轮也不另造。
- 1NUS 的全局叠合最弱（TM-score 0.58–0.68，TMalign RMSD 3.25 Å），NMN 接触大多数为 `role_mismatch`；1NUS 的 NMN/APC 结果不足以判断 NMN 口袋或 K140′ 界面。
- 锚点 H14、H17、T135、R138 的 CA 变化均小于 0.5 Å；这只说明未见大幅主链位移，不排除侧链旋转或微环境重排。所有“距离近”都不等于氢键、配位或催化几何。

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
- TMalign/Kabsch 是全局骨架叠合；表中的 RMSD 不是局部配体姿态误差，也不能给单个接触距离赋实验误差。
- 1NUS 的 NMN 与 APC 是同一实验结构中的两个配体；1F9A 的 ATP/Mg、2QJO 的 NAD/POP 同理。三个状态在图中分成三个独立面板，不表示同时结合，也不构成一个三元复合物。

## 判定口径

- 接触：配体原子–蛋白重原子距离 ≤4.0 Å；金属用 ≤3.2 Å。1F9A 的 Mg 在 3.2 Å 内没有蛋白接触，故接触表只有 ATP。
- 每个模型–接触行先判定：`unmapped`；`uncertain_mapping`（映射几何置信度低）；`not_recovered`（目标原子距离 > 阈值）；`clash`（局部或全局 <2.0 Å）；`role_mismatch`（没有可支持的对应原子/基团）；`supported_candidate`（角色匹配且 ≤4.0 Å）；`geometric_only`（几何邻近但角色不匹配）。
- 跨三个模型汇总：`supported_consistent` = 3/3 支持且无 clash；`supported_majority` = 2/3 支持且无 clash；`model_dependent_supported` = 1/3 支持且无 clash；`supported_majority_with_model_clash` / `model_dependent_supported_with_clash` = 有支持但至少一个模型 clash；`clash` = 无支持且至少一个模型 clash；`role_mismatch`、`not_recovered`、`unmapped`、`uncertain_mapping` 按名称。
- `supported_consistent` 不是“3 个独立实验验证”，也不是完整相互作用。它包含主链–主链接触；行数不是独立接触数。近邻与 clash 可以同时为真，只有严重冲突才被计为 clash。
- `global_nearest` 是某个迁移配体原子到任意 WT 蛋白重原子的最近原子，不等同于该源接触映射到的 KVP40 残基。同一配体原子可被多个源接触记录重复引用；`global_nearest_pairs.csv` 和 `global_nearest_summary.csv` 按 state/clone/model/ligand/atom 去重，用于实际冲突定位。
- `ca_median_A` 是相对固定 WT0 的 CA 距离；WT 自己包含恒为 0 的参考模型，因此 `wt_ca_min/max` 只是三个 WT 模型相对 WT0 的描述范围。`models_outside_wt_ca_range` 只表示超出该描述范围，不是同 clone 自基线，也不是突变效应。克隆自身样本基线引用既有 `data/results/evaluation/kvp40_epcr_full_length_structures/mutation_local.csv` 与 `mutation_local_baseline.csv`（domain-fit，9 个跨序列模型对）。
- 配体距离是固定迁移姿态下的最近侧链重原子距离，不是对接分数、结合自由能或亲和力。3 个 diffusion 样本的 min–max 是描述性范围，不是置信区间。

## 接触结果

每个状态的源接触数：1F9A 61、1NUS 77（其中 NMN 31、APC 46）、2QJO 101（NAD 77、POP 24）；三个克隆使用同一组接触定义。

| 状态 | 克隆 | 3/3 无 clash | 2/3 无 clash | 1/3 无 clash | 有支持但含 clash | 无支持且≥1 clash | role mismatch | not recovered | uncertain/unmapped |
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

- 2QJO 行是 NAD + POP 合计，因此 `supported_consistent` 合计为 30/36/30；其 NAD 子状态为 WT 29、NatV10 31、NatV89 30，POP 为 1/5/0。
- “无支持且≥1 clash”不是 3/3 全冲突：例如 1F9A NatV10 contact 6 为 `n_clash=1`、`n_not_recovered=2`。逐模型 `n_clash` 分布见 `contact_summary.csv`；不能把该列当作三个模型都冲突。

### 实际最近原子对（去重）

`contact_status` 的 clash 可以由局部或全局距离触发，而 `global_nearest` 指向任意 WT 蛋白原子；因此不能用源接触对应的残基名（如 R9/H17/R138）来定位实际冲突。下表按“每个模型、每个配体原子、每个最近蛋白原子”去重。

| 状态/配体 | 克隆 | clash 行 | 去重最近原子对 | 主要实际最近残基（按对数） |
|---|---|---:|---:|---|
| 1F9A ATP | WT | 33 | 18 | TYR102 12、ASP105 4、ASP103 1、LYS104 1 |
| 1F9A ATP | NatV10 | 26 | 13 | TYR102 8、ASP105 4、ASP103 1 |
| 1F9A ATP | NatV89 | 55 | 29 | TYR102 11、LYS104 8、ASP105 5、ASP103 4、ALA134 1 |
| 1NUS APC | NatV89 | 30 | 25 | ASP103 11、TYR102 6、ASP105 4、LYS104 4 |
| 2QJO POP | WT | 26 | 13 | SER133 6、LYS104 5、THR135 2 |
| 2QJO POP | NatV10 | 15 | 9 | SER133 4、THR135 3、ASP105 1、LYS104 1 |
| 2QJO POP | NatV89 | 44 | 20 | LYS104 7、SER133 6、ASP105 5、THR135 2 |

- 1F9A ATP 的实际最近原子几乎都在 TYR102–ASP105 环区；NatV89 只有 ALA134 主链 N 一个例外。R9、H17、R138 等源接触对应残基不是实际最近原子，不能据此提出 F137Y→R138/T135 耦合；该假说撤回。
- 2QJO POP 的实际最近原子包含 SER133、THR135、LYS104、ASP105，仍不是单一位点或单一网络；WT 自己也有 26 行→13 对。NatV89 的 20 对更多，但不能把行数或对数当作严重程度或独立相互作用。
- 1NUS NMN 的最近原子主要为 TRP85；NMN 本身大多 `role_mismatch`，不用于判断 NMN 口袋。NAD 也有 TYR102/ASP103/VAL131 等实际近距离，但其 `supported_consistent` 仍为 29–30，不能仅凭 clash 对数否定 NAD 环境。

### 1F9A ATP/Mg

- WT 与 NatV10 都保持 16 个 `supported_consistent`；NatV89 只剩 2 个。NatV89 三个模型的支持数为 6/25/18（model 1 最高、model 0 最低），差异跨模型不均，不能归因于单一样本。
- NatV89 的 55 个 clash 行去重后是 29 个最近原子对，主要是 TYR102/ASP103/LYS104/ASP105，另有 ALA134 N；不是源接触对应的 R9/H17/R138。没有直接结构证据支持 F137Y 与 R138/T135 的耦合解释，本报告不把它列为优先假说。
- WT 自身也有 33 行→18 对，说明该环区在当前刚性迁移姿态下本来就有冲突；NatV89 的 29 对更多，但不能据此判定配体不能结合或克隆功能受损。

### 1NUS NMN/APC

- 1NUS 是三个来源中全局叠合最弱者。每个克隆都有 42 个 `role_mismatch`；NMN 的 31 个接触中 24 个为 role mismatch、3 个 unmapped、2 个 not recovered，三个克隆几乎没有稳定支持。NMN 结果不能解释为“NMN 口袋未保留”，只能说明当前刚性映射/原子角色规则无法判别。
- APC 的 46 个接触中，WT/NatV10 各有 7/8 个 `supported_consistent`，NatV89 为 3；NatV89 的 with-clash 为 14（WT 12、NatV10 9）。实际最近原子主要在 TYR102–ASP105 环区，不指向 K140′ 界面。
- 1NUS 的全局 TMalign RMSD 3.25 Å 不是局部配体姿态的实验误差；局部姿态还受构象差异和四聚体界面条件影响。本状态不评估 K140′ 替代，也不把 APC 近距离接触当作 ATP 结合姿态。

### 2QJO NAD/POP

- NAD 接触在三个克隆中保持得最好：`supported_consistent` WT 29、NatV10 31、NatV89 30；当前产物态 NAD 部分没有显示克隆特异的口袋重组。
- POP 的 `supported_consistent` WT 1、NatV10 5、NatV89 0。NatV89 的 44 个 clash 行去重为 20 个最近原子对，实际最近原子为 LYS104 7、SER133 6、ASP105 5、THR135 2；WT 为 26 行→13 对（SER133 6、LYS104 5、THR135 2），NatV10 为 15 行→9 对。NatV89 的对数更多，但并非单一位点破坏，也不能以行数判严重程度。
- 2QJO 的 POP 来自细菌双功能 NadM 的产物态，不是 KVP40 的实验结构；POP 的冲突不能直接解释为 KVP40 的焦磷酸结合缺陷。它提出的问题是：NatV89 的局部构象或姿态选择是否使 POP 更容易落入该环区，还是刚性迁移的产物。

## 突变位点与侧链

下表同时列出本分析的 WT0 相对 CA 值，以及既有 domain-fit 分析中的同位置局部值和同模型自基线。两套数值的叠合方式不同，不能逐位相加或互相替代；自基线列才是判断“是否超出同 clone 预测离散”的依据。

| 位点 | 克隆 | WT0 相对 CA 中位（范围） | 既有 domain-fit 局部中位（范围） | 同位置自基线 | WT 对应基团/变体侧链到配体距离（1F9A/1NUS/2QJO） | 判定 |
|---|---|---|---|---|---|---|
| D83V | NatV10 | 0.51 (0.34–0.74) | 0.64 (0.30–1.43) | 0.78 | WT OD1/OD2：21.08/7.59/7.64 Å；变体最近侧链：21.30/7.37/8.53 Å | 在自基线内；无直接接触 |
| M162V | NatV10 | 0.14 (0.09–0.16) | 0.15 (0.09–0.30) | 0.15 | WT SD：21.29/21.74/22.05 Å；变体最近侧链：19.08/19.55/19.93 Å | 与自基线相当；无直接接触 |
| I21N | NatV89 | 0.27 (0.14–0.28) | 0.23 (0.04–0.31) | 0.24 | ND2/OD1：7.47/7.57/7.81 Å | 在自基线内；无直接接触 |
| F137Y | NatV89 | 0.31 (0.25–0.47) | 0.32 (0.25–0.65) | 0.29 | 新 OH：7.47/8.67/8.62 Å；环最近原子约 5.9/7.3/6.5 Å | 接近/略高于自基线；OH 不接触配体，不能称新氢键 |
| I156T | NatV89 | 0.65 (0.57–0.66) | 0.54 (0.44–0.66) | 0.18 | 新 OG1：10.14/11.39/11.36 Å | 高于自基线；只有局部骨架差异，无直接接触 |

- D83V 失去羧基是序列事实；WT 的 OD1/OD2 本身在 1F9A 中已约 21 Å、在 1NUS/2QJO 中约 7–8 Å。变体整体侧链的最近距离为 21.30/7.37/8.53 Å，不能把它当作“丢失羧基的距离”，也不能据此说 D83 直接识别底物。
- M162V 的 WT SD 在三个姿态中为 21.29/21.74/22.05 Å；变体整体侧链为 19.08/19.55/19.93 Å。两者都远离配体，当前数据不支持直接接触。
- F137Y 的新 OH 与配体最近 7.47 Å（1F9A）到 8.67 Å（1NUS）；没有氢坐标、角度或受体类别分析，不能称其为新氢键。环最近原子约 5.9/7.3/6.5 Å，但这是整个侧链的最近原子，不是 OH 的接触。
- I156T 的 CA 位移高于同位置自基线（既有 domain-fit：0.54 vs 0.18 Å），说明它是当前模型中较明显的局部骨架差异；OG1 离配体 10–11 Å，不能把该骨架差异解释为底物环境改变。
- NatV10/NatV89 都是多突变克隆；不能把任何克隆级差异归因于单个位点。

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

锚点 CA 变化均小于 0.5 Å；这只限制大幅主链位移，不排除侧链旋转、质子化或微环境重排。侧链距离与 WT 的 2.6–4.0 Å 相近；T135 的 1.85 Å 属于 distance flag，不是有利氢键或催化几何。当前没有氢、键角、水、金属或质子化状态，不能评价 anchor 的化学角色是否保持。

## 边界与可检验问题

能支持：

- 在这 9 个 apo 模型和三个固定迁移姿态中，五个 NMNAT 突变的新增/丢失侧链原子均未直接接触配体；NatV10 的 ATP/NAD 接触状态与 WT 接近；NatV89 在 ATP 和 POP 上有更多实际环区近距离，但跨模型不均且不是源接触对应的残基。
- 1NUS 的 NMN 部分在当前映射下不可判别；2QJO 的 NAD 部分比 POP 部分稳定；POP 的 NatV89 差异值得作为有界问题追踪。

不能支持：

- 不能说明突变提高或降低活性、稳定性、底物亲和力、催化效率或 NAD⁺ 产量；不能作 NatV10/NatV89 的排名。
- 不能说明真实结合姿态、结合顺序、限速步、诱导契合或金属配位；没有 MD、对接、实验结构或酶学测定。
- 不能把 `supported_consistent` 当作独立相互作用，把 clash 行数当严重程度，或把“距离近”当作氢键/催化几何；不能把“clash”直接当作配体不能结合。
- 不能把 1NUS 的 NMN/APC 结果外推到 KVP40 的 NMN 口袋或 K140′ 界面；也不能把 2QJO 的 POP 结果直接外推到 KVP40 的 Nudix 催化。
- 全局 TMalign/Kabsch RMSD 不是局部接触误差；3 个 diffusion 样本不是独立实验重复，min–max 不是置信区间。

可检验问题：

1. **NatV89 的 ATP/POP 近距离是否稳定？** 当前实际最近原子集中在 TYR102–ASP105 环区（ATP 另有 ALA134 N；POP 另有 SER133/THR135），不是源接触对应的 R9/H17/R138。独立配体放置或短时 MD 可以帮助判断这些近距离是姿态依赖还是稳定接触，但不能单独证明 F137Y 耦合；未重现不自动证伪，重现也可能继承相同的初态和参数。本轮不启动额外 MD/预测。
2. **D83V/M162V 的远端或 second-shell 效应？** 当前姿态没有直接接触，但羧基/硫醚丢失仍可能通过局部静电、溶剂化或远端网络影响；不以未来直接靠近配体为必要条件，现有数据也不能排除这类效应。
3. **1NUS 是否需要更严格的局部对齐？** 当前 TM 0.58–0.68 和大量 role mismatch 使 NMN 结论不可用；若要回答 NMN 口袋问题，应先改进来源选择/局部映射，而不是增加对当前 `role_mismatch` 表的解释。

## 文件与复算

运行命令：

```bash
/home/yhshy/miniconda3/envs/nad_data/bin/natvqvae epcr-nmnat \
  --config config/epcr_nmnat_environment.yaml \
  --output data/results/evaluation/kvp40_epcr_nmnat_environment
```

测试：`python -m pytest -q`，16 passed。

- `contact_rows.csv`：2151 行（239 个源接触 × 9 个模型），逐模型接触状态。
- `contact_summary.csv`：717 行（239 × 3 克隆），跨模型汇总、WT 范围、距离差；`n_clash` 为每个接触实际 clash 模型数。
- `global_nearest_pairs.csv`：649 行，逐 state/clone/model/ligand atom/global nearest 去重的最近蛋白原子对；仅列 global close/severe，包含被合并的源 contact 数。
- `global_nearest_summary.csv`：45 行，按 state/clone/model/ligand 去重统计 clash 行数、最近原子对数、按残基的对数分布。
- `position_environment.csv`：279 行，突变/锚点的逐模型环境。
- `position_environment_summary.csv`：93 行，按位点/克隆/状态汇总。
- `provenance.json`：输入 SHA256、模型身份、TMalign/Kabsch 指标、源码哈希、命令与图路径。
- `coordinates/`：`reference_WT0.pdb`、`ligands_{1F9A,1NUS,2QJO}.pdb`、9 个共同框架模型 PDB。
- `figures/epcr_nmnat_structure.png`：三个状态分面板；橙色球 = NatV10 D83/M162，品红球 = NatV89 I21/F137/I156，黄/绿/蓝配体 = 1F9A/1NUS/2QJO；每面板仅一个来源状态，不是同时结合。
- `figures/epcr_nmnat_sidechains.png`：D83V、M162V、I21N、F137Y、I156T 五个位点的 WT0 与实际克隆侧链对照（灰=WT0，橙/品红=3 个克隆模型）。
- `figures/epcr_nmnat_differences.png`/`.svg`：WT0 相对 CA 位移，以及三个状态各自独立的最近侧链–配体距离；没有跨状态距离中位。
- `figures/render_structure.pml`、`render_sidechains.pml`：可复算的 PyMOL 渲染脚本。

既有自基线来源：`data/results/evaluation/kvp40_epcr_full_length_structures/mutation_local.csv`、`mutation_local_baseline.csv`（其输入哈希与运行身份见该目录 `build.json`）。本报告没有改写这些文件。
