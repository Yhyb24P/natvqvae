# P2 NMN 来源结构对应（正式版）

日期：2026-10-23。目的：在自由 WT+NMN 共折叠之外，检查 1NUS、1NUP、1GZU、1HYB 的 NMN 接触能否提供可迁移的 KVP40 子位点线索。只做来源结构对应，不合成唯一 NMN 姿态，不把同源结构的一致性当作实验验证，也不把来源计数当作独立证据。

## 输入与来源家族

| 来源 | 体系 | 分辨率/组装 | 配体/限制 | 家族 |
|---|---|---|---|---|
| 1NUS | 人胞质 NMN/NaMN 腺苷酰转移酶 | 2.2 Å，tetramer | NMN + APC | human_NMNAT3 |
| 1NUP | 人胞质 NMN/NaMN 腺苷酰转移酶（WT） | 1.9 Å，tetramer | NMN + SO4 | human_NMNAT3 |
| 1GZU | 人 NMNAT1（WT） | 2.9 Å，hexamer | 仅 NMN；MSE/缺失 | human_NMNAT1 |
| 1HYB | *Methanothermobacter thermautotrophicus* NadM | 2.0 Å，hexamer | NMN + SO4；**H19A 突变** | archaeal_NadM |

- 1NUS/1NUP 是同一人源酶的两种 NMN 结合状态（分别为 NMN+APC 与 NMN+SO4），不是两个独立家族或独立验证。
- 1HYB 是古菌 NadM 的 H19A 活性位点突变体，不能当作 WT 标准。
- 1GZU 低分辨、含 MSE 修饰和缺失区；映射时把 MSE 按化学 parent MET 保留 CA/残基编号。只有 Se 标为不直接等价；MSE 的其他 C/主链按 MET parent 处理。没有 NMN 接触落在 MSE 位点。
- 新 CIF 来自官方 `https://files.rcsb.org/download/<ID>.cif`，存于 `data/raw/structures/nmnat_p2/`；SHA256 见 `provenance.json`。

## 方法

1. 取各来源链 A 的指定蛋白范围；1GZU 的 MSE 在临时映射结构中转为 MET，保留原残基编号；无 CA 的 Q271 不参与 CA 拟合，但保留在来源接触表中。
2. TMalign 将来源蛋白对齐到 WT0 NMNAT 4–181；记录 TM-score、TMalign RMSD、拟合 RMSD 和 CA 距离。映射置信度只是 CA 几何（高 ≤2 Å、中 ≤5 Å、低其余），不是功能可信度。
3. 列出 NMN 重原子 ≤4.0 Å 的全部蛋白原子–配体原子对；配体原子分为 phosphate（P/O1P/O2P/O3P）、phosphate_bridge（O5R）、ribose（C5R/C4R/O4R/C3R/O3R/C2R/O2R/C1R）、nicotinamide（N1/C2/C3/C7/O7/N7/C4/C5/C6）。
4. 记录来源原子/组、目标残基/原子、配体原子/组、距离、来源家族、映射 CA 距离和可对应性；`source_function`/`target_functions` 保留 Tyr OH、Trp NE1、His 咪唑等官能团。`contact_rows` 是原子对记录数，不是独立相互作用或网络边数。
5. 目标汇总按 `(target residue, ligand group)` 分组，保留来源家族；不同配体基团不合并成同一“共识”。

## 结果

完整逐接触表：`source_contacts.csv`（330 行）；按目标/配体组汇总：`target_group_summary.csv`（32 行）；来源身份：`provenance.json`。

### 同一目标位置不等于同一接触

| 目标 | 涉及家族 | 实际记录 |
|---|---|---|
| I7 | human_NMNAT3、archaeal_NadM（2） | 1NUP/1NUS 的 CYS C/CB 与 1HYB 的 VAL CG1→ILE；骨架/脂肪族记录，不是同一原子角色 |
| G8 | human_NMNAT3、human_NMNAT1、archaeal_NadM（3） | 全部是 backbone；1NUP/1NUS 接触 ribose，1GZU/1HYB 接触 phosphate 或 phosphate_bridge |
| G38 | 三个家族 | 1NUP/1NUS/1GZU 的 V49/V51 CG2 侧链接触映射到 G38，但目标 Gly 无侧链，`target_lacks_sidechain`；只有 1HYB G38 CA/C/O 主链可对应 |
| Y79 | human_NMNAT3、archaeal_NadM（2） | human_NMNAT3 的 W90 芳香环接触；1HYB I81 主链接触；不能合并为同一芳香作用 |
| W85 | 三个家族 | 1NUP/1NUS/1GZU 的主链/脂肪族接触；1HYB W87 的芳香环接触；官能团不同 |
| Y110 | 三个家族 | human_NMNAT3 W148、1GZU W169、1HYB V108 等不同来源原子；有 aromatic/backbone 多种标签 |

- G8 的 `phosphate`、`phosphate_bridge`、`ribose` 在汇总中分开；G38 的侧链–Gly 记录保留但不构成共识。不同目标位置不能相加成“共同口袋”。
- 1GZU 的 W92、E94、W169 中，E94 是 Glu（主链/CB/CG），不是芳香残基；不能把它写成 W94。
- 1NUP/1NUS 的 NMN 接触高度一致，但这是同一家族内的状态一致性，不是两个独立来源的相互验证。

### 来源层面的稳定线索

- 1NUS/1NUP 支持 human_NMNAT3 在 NMN+APC 与 NMN+SO4 状态下的局部 NMN 组织；对 KVP40 的全局 TM 仅 0.58–0.68，许多目标映射为低/中 CA 置信。
- 1HYB 的拟合最好（TM ≈ 0.845），NMN 接触主要落在 KVP40 I7/G8/R9/G38/S39/D78/D82/W85/E106/Y109/Y110/L113 附近；但它是 H19A 突变体。
- 1GZU 的 NMN 接触包含 G15/S16/V51/Y55/K57/W92/E94/T95/L96/L159/L168/W169/D173 等；只作低分辨/缺失条件下的定性参照。

### 可迁移的子位点信息

- 反复出现的目标位置包括 I7、G8、G38、Y79、W85、Y110，但每个位置的来源家族数、配体基团、原子角色和可对应性不同，见上表；不能压缩成一条“三家族共识”。
- 自由 WT+NMN 共折叠仍是独立的主结果；这些对应尚不足以定义可靠约束集，本轮不把 NMN 引导残基作为必需条件。

## 边界

- 不能从这些对应确定 KVP40 的真实 NMN 姿态、亲和力、催化构象或活性；高 CA 置信度、同一目标残基、来源数量都不是化学可信度。
- 不能把 1HYB 当 WT，不能把 1GZU 的 MSE/低分辨结构用于精确原子对应，也不能把 1NUS/1NUP 当作两个独立实验验证。
- 不能把 Trp/Tyr 环接触按 exact atom 等同；环角色、官能团角色和骨架接触必须分开解释。

## 文件与复算

```bash
/home/yhshy/miniconda3/envs/nad_data/bin/natvqvae nmnat-nmn-sources \
  --config config/nmnat_nmn_sources.yaml \
  --output data/results/evaluation/kvp40_nmnat_p2_sources
```

- `source_contacts.csv`：330 条来源–配体原子对，含配体基团、蛋白组/官能团、目标原子/官能团、映射和可对应性。
- `target_group_summary.csv`：32 个目标残基/配体基团组，含 `n_families`、来源、原子对记录数、最小距离、来源/目标原子组和可对应性；不是独立证据计数。
- `provenance.json`：配置、参考结构、四个来源的 SHA256、TM/拟合指标、源码哈希、TMalign 和选链/范围。
- 本分析不修改既有 `natv_nmnat_state_mapping/`、ePCR 或 binding 结果。
