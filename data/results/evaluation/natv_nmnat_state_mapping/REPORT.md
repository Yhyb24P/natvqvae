# KVP40 NatV NMNAT 状态分层接触映射

日期：2026-10-23。对象：KVP40 NatV WT 的 NMNAT 域（4–181）。目的：把 1F9A、1NUS、2QJO 的实验接触迁移到匹配的全长 WT 背景，区分实验观察、KVP40 映射和建模推断；不构建真实 Michaelis 复合物，不推断活性。

## 输入与身份

- 全长 WT：`data/results/evaluation/natv_dual_domain_design/coordinates/WT_WT_model_{0,1,2}.cif.gz`。Boltz 2.2.1，seed 20261022，3 diffusion samples，200 sampling steps，3 recycling steps，512 行冻结 MSA，全长 341 aa apo 单体；WT pLDDT 0.922、pTM 0.913。与 A1/A2/A3 同一批运行。
- 实验结构（`data/raw/structures/nmnat/`，SHA256 见 `provenance.json`）：
  - 1F9A：2.0 Å，hexamer；链 A ATP+Mg²⁺。
  - 1NUS：2.2 Å，tetramer；链 A NMN+APC，无沉积 Mg；链 B K140′ 为条件性跨链接触。
  - 2QJO：2.6 Å，hexamer；链 A NAD+POP（PPi）；APR 属 Nudix 域，未混入 NMNAT。
  - 2QJT：dimer；AMP/Mn 在 Nudix 域，仅作负对照，不进入 NMNAT 映射。

## 方法

1. 各源结构取选定的蛋白链/残基范围，TMalign 对齐到每个 WT 模型的 NMNAT 4–181；记录 TM-score、aligned length、fit RMSD。
2. 以对齐的 CA 对做 Kabsch 变换，把源配体原子（以及 1NUS 链 B K140 NZ）刚性迁移到 WT 框架。
3. 实验层记录配体–蛋白原子接触（≤4.0 Å；MG/MN 用 ≤3.2 Å）及 occupancy/altloc/B-factor。
4. 映射层记录 KVP40 残基、匹配符号、变换后 CA 距离、映射置信度、映射残基距离、最近 WT 原子和冲突状态。
5. 在完整 341 aa 背景检查配体–蛋白、配体–配体（金属–核苷酸除外）、金属环境和 1NUS 界面探针。

## 结果

### 状态汇总（3 个 WT 模型的范围）

| 状态 | 源接触 | 已映射 | 高置信 | 恢复接触 | 配体–蛋白最小距离 (Å) | 严重冲突模型 |
|---|---:|---:|---:|---:|---:|---:|
| 1F9A ATP/Mg | 61 | 61/61 | 50 | 35–36 | 0.29–2.25 | 2/3 |
| 1NUS NMN/APC | 77 | 74 | 35–38 | 35–42 | 0.20–1.22 | 3/3 |
| 2QJO NAD/POP | 101 | 101 | 79–86 | 66–77 | 0.24–1.36 | 3/3 |

- TMalign 对齐 fit RMSD：1F9A 1.88–1.95 Å；1NUS 2.25–2.31 Å；2QJO 1.36–1.37 Å。迁移后的配体位置继承这一不确定性，单个 <2 Å 的冲突不能当作精确定量。
- 1F9A：Mg–ATP 氧最小距离保持 2.2 Å（与源结构一致）；但 ATP/Mg 与 WT 蛋白的最近距离为 0.29–2.25 Å，2/3 模型严重重叠。ATP/Mg 局部接触网络可部分映射，但 apo 模型不能直接容纳该姿态。
- 1NUS：NMN–APC 配体间最小距离 4.13 Å，没有配体–配体冲突；但配体–蛋白在 3/3 模型严重重叠。联合占位不被两个配体本身阻断，但当前 apo 口袋需要重排。
- 2QJO：NAD–POP 最小距离 2.66 Å；配体–蛋白在 3/3 模型严重重叠。产物态接触映射最好（高置信 79–86/101），但仍不是可直接接受的姿态。
- 1NUS K140′ 界面：源结构 K140 NZ–APC O1G = 2.73 Å。迁移到 WT 后：模型 0 最近碱性残基 K104 NZ 4.95 Å（`weak_candidate`，蛋白最近 2.00 Å）；模型 1 K104 NZ 4.08 Å（`overlap`，蛋白最近 1.89 Å）；模型 2 最近碱性 R138 NH2 7.73 Å（`no_candidate`）。没有一致的界面替代证据；K104 不能被称为 K140′ 的替代者。

### 变体位置选择（仅基于 WT 接触图）

- **直接候选**（变体侧链在至少一个状态/模型距离迁移配体 ≤4.5 Å）：
  - A3 A134G：三种状态、三个模型均为 direct（2.25–3.10 Å）。
  - A3 Q86I：仅在 1NUS 为 direct（NMN 3.37–3.82 Å）；在 1F9A/2QJO 未入选。
- **基序候选**（位于 N 端 8–22 或 C 端 130–138，但没有直接配体邻近）：
  - A1 L18I、A2 T22Q；这是“可提出基序/环区假说”，不是接触证据。
- **未入选**：A1 D57S/R94N/M127A；A2 G60L/L111K；A3 V61H。未入选表示当前没有明确可判别的接触假说，不代表低活性或淘汰。
- A1 M127A 所在位置在本次 TMalign 映射中对应 1F9A R121，但 KVP40 WT 本身是 M127；不能用 1F9A 的 Arg 接触解释 A1 的该位点。
- 本表只做 WT 位置选择，没有比较 A1/A2/A3 模型，也没有活性排名。

## 能说明与不能说明

能说明：

- N 端 G8–R18 与 C 端 S133–R138 的局部接触可以映射到 KVP40；1F9A 的 ATP/Mg 接触、1NUS 的 C 端接触、2QJO 的产物态接触均有高置信映射。
- 在当前 apo WT 模型上，三种状态的刚性配体迁移都出现模型依赖的蛋白–配体冲突；1NUS 的联合占位不被配体间冲突阻断，但界面替代证据不一致。
- A3 A134G 是唯一在三种状态都直接贴近迁移配体的变体位点；A3 Q86I 仅在 1NUS 直接；A1 L18I、A2 T22Q 只有基序级候选。

不能说明：

- 不能说明 KVP40 的真实底物结合姿态、结合顺序、限速步或活性；没有诱导契合、MD、对接或实验验证。
- `mapping_confidence` 是相对于 TMalign/Kabsch 全局对齐的局部 CA 距离，不是独立实验验证；高置信不等于已证实的功能接触。
- 不能把“严重重叠”解释为配体不能结合；它只说明刚性迁移到 apo 预测模型时，当前侧链/环区需要重排。
- 不能把 K104 称为 K140′ 的替代者；1NUS 界面仍是条件性、未解决。
- 不能把 A3 A134G 的近距离接触解释为活性改善；它只是当前最具体的结构假说。
- 预测模型不是实验结构；3 个 diffusion 样本是描述性范围，不是置信区间。

## 文件

- `experimental_contacts.csv`：实验层原子接触。
- `mapping.csv`：每个源接触在每个 WT 模型上的映射、置信度、恢复和冲突。
- `state_summary.csv` / `state_summary.json`：状态/模型汇总。
- `interface.csv`：1NUS K140′ 界面探针。
- `variant_selection.csv`：A1–A3 位置选择。
- `transforms.json`：TMalign 对齐与 Kabsch 变换。
- `provenance.json`：输入哈希、模型身份、软件版本和参数。
