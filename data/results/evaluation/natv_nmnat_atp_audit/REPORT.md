# 1F9A ATP/Mg 接触的原子/基团级审计（重做版）

日期：2026-10-23。对象：KVP40 NatV WT 的 NMNAT 域（4–181）；源状态仅 1F9A 链 A 的 ATP/Mg²⁺（61 个源原子接触记录）。本报告替代 `e1db337` 中的审计结论；只做几何迁移、原子角色匹配和冲突标记，不构建复合物、不做预测或 MD。

## 输入与身份

- WT：与 A1–A3 同一 Boltz-2 运行的三个全长 apo 模型，seed 20261022，3 samples；身份与哈希见 `provenance.json`。
- 1F9A：2.0 Å，hexamer；链 A ATP+Mg²⁺；输入 CIF 为 `data/raw/structures/nmnat/1F9A.cif`。
- 对齐：TMalign 对齐 1F9A 链 A（1–164）到每个 WT 的 NMNAT 4–181；刚性 Kabsch 迁移配体原子。

## 判定方法（与 e1db337 的关键差别）

- **不再取目标残基的最近任意重原子。** 先根据源原子角色确定目标原子/基团：
  - 主链：目标同名的 N/CA/C/O/OXT（Pro 与非 Pro 的 N 不互换）；
  - 同残基侧链：同名原子，或明确列出的对称等价原子（Asp/Glu 羧基 O、Arg 胍基 NH1/NH2、Phe/Tyr 环对称对等）；
  - 跨残基：仅羧基 O（D↔E）、羟基 O（S/T/Y）、酰胺 O/N（N↔Q）、阳离子 N（K↔R）作为 role-equivalent；芳香/疏水侧链不做跨残基原子等价；
  - His NE2 不会用 CD2 代替；Thr OG1 不会用主链 N 代替；Pro 主链 N 单独标记。
- `_sidechain_role` 是审计用简化角色表，不是完备化学分类器；芳香残基的所有侧链 C 都会被标为 `aromatic_C`（包括 PHE CB 等），当前 CE1 等是否真正环原子仅按实际结构证据解释。
- `geometry_confidence` 只表示 TMalign/Kabsch 后的 CA 几何；`target_atom_match` 表示原子对应关系（same_name、symmetric_equivalent、backbone_same_name、role_equivalent、role_mismatch、missing）。
- `local_flag`/`global_flag` 是距离标记（severe <2.0 Å、close <2.5 Å）；不是普适化学冲突裁决。
- 汇总状态：`supported_consistent` = 3/3 支持且无 clash；`supported_majority` = 2/3 支持且无 clash；`model_dependent_supported` = 1/3 且无 clash；`supported_majority_with_model_clash` = 2/3 支持且第三模型 clash；`model_dependent_supported_with_clash` = 1/3 支持且其他模型 clash；`role_mismatch` = 当前映射/规则下没有可支持的对应原子/基团。

## 结果

### 逐模型状态（61 个源接触 × 3 个 WT 模型）

| WT 模型 | supported | role_mismatch | not_recovered | clash |
|---|---:|---:|---:|---:|
| model 0 | 24 | 14 | 12 | 11 |
| model 1 | 28 | 14 | 12 | 7 |
| model 2 | 34 | 14 | 13 | 0 |

### 跨模型汇总（61 个源接触）

- `supported_consistent`（3/3，无 clash）：21
- `supported_majority_with_model_clash`（2/3 支持，第三模型 clash）：8
- `model_dependent_supported`（1/3，无 clash）：2
- `model_dependent_supported_with_clash`（1/3，其他模型 clash）：5
- `role_mismatch`：14
- `not_recovered`：11

`supported_consistent` 是特定原子/基团接触候选，不是完整复合物或催化相互作用的验证。21 个中 16 个是主链–主链接触，5 个是侧链特异接触；不能把 21 当独立相互作用数。WT model0 对应的这 21 行中有 17 行 local/global close（2–2.5 Å；16 行 global close、1 行 both close、4 行 none）；角色构成为 backbone N 6、backbone C 5、backbone O 5、hydroxyl O 1、cationic N 1、imidazole N 3。该计数只是同角色+距离阈值邻近；5 个主链碳不是氢键证据，17 个 close 不能被“无 clash”掩盖；不等于 21 个合理物理相互作用或完整 ATP 口袋。

### 侧链特异的支持

- **H13/H16（对应 KVP40 H14/H17）NE2** → ATP O1B、O2A：3/3 supported_consistent；O3A、O5′、PA：2/3 支持但另一模型 clash。His 质子化/互变异构未定，不能称为已证实氢键。
- **T127 OG1（对应 KVP40 T135）** → ATP O1G：3/3 supported_consistent；PG 仅 1/3 支持且另两模型 clash。**T127 主链 N → O3G 是独立的主链接触，不能当作 T135 羟基作用保留。**
- **R130 NH2（对应 KVP40 R138）** → ATP O3G：3/3 supported_consistent；目标原子为 NH1，属对称等价胍基 N。
- 上述接触附近仍常有 `close_distance_flag`（2.0–2.5 Å），且无显式氢/键角；这些是原子/基团候选，不是催化几何证明。

### Role mismatch 与映射不确定性

- 在当前映射与原子角色规则下，14 个源接触未找到可支持的对应原子/基团，主要是 1F9A R121 侧链（对应 KVP40 M127，与 A1 的 M127A 同区）、MET118 SD、VAL19 CB、ILE5 CD1 等；这些不能作为支持。该结果不能证明真实口袋不可迁移。
- 21 个低几何置信行全部同时为 role_mismatch；`atp_contacts.csv` 保留 `geometry_confidence=low`，但汇总状态优先显示 `role_mismatch`。不能把 role_mismatch 当高置信排除，也不能把低置信当支持。

### Mg 的边界

- 1F9A 的 Mg 在 3.2 Å 内没有蛋白原子接触；审计表没有 MG–蛋白记录。审计未显式分析水，因此不能确定完整水配位层或 KVP40 金属位点。
- 刚性迁移后 Mg–ATP 内部距离是变换不变量，不代表 KVP40 金属配位被恢复；迁移 Mg 与 WT 蛋白的近距离是姿态冲突，不是蛋白金属位点。

## 与 e1db337 的差异

- 撤回：旧“17 个 supported_consistent”及其中基于最近任意重原子的 T135/H17 侧链支持；旧 R9 chi/RMSD/移植距离和 WT 96–108 loop 数字；旧“KVP 特异 loop 导致全部冲突”的确定性解释。
- 保留：75e2e6c 的几何映射结果；1F9A 的 ATP/Mg 接触框架；Mg 在 3.2 Å 内无蛋白接触这一事实。
- 新增：原子角色优先的目标原子选择、role_mismatch、3/3–2/3–1/3 汇总、真实的 per-model clash 计数。

## 能说明与不能说明

能说明：

- 在原子/基团层面，H14/H17 的 NE2 到 ATP O1B/O2A、T135 的 OG1 到 O1G、R138 的 NH2 到 O3G，在当前几何/角色规则下是 3/3 支持候选；
- 这些支持的多数是主链接触；侧链特异支持只有少数原子对；
- 在当前规则下，14 个源接触未找到可支持的对应原子/基团；这是条件性结果，不能证明真实口袋不可迁移。

不能说明：

- 不能说明完整 ATP/Mg 复合物可接受；其他位置仍有 clash；
- 不能在没有氢坐标/键角时宣称氢键；2.0/2.5 Å 只是距离标记；
- 不能说明 R9 差异或任何 A 变体的效应；A3、1NUS、2QJO 不在此审计；
- 没有 loop/R9 的正式实现，相关临时数字已从正式报告撤回；没有 MD、预测或实验验证。

## 文件

- `atp_contacts.csv`：61×3 条原子/基团级记录（含 target_atom_match、chemistry_class、local/global flags、uncertainty）。
- `atp_contact_summary.csv`：按源接触跨模型汇总（3/3、2/3、1/3、role_mismatch）。
- `provenance.json`：输入哈希、模型身份、对齐、阈值、状态定义。
