# WT ATP/NMN 联合占位的 AF3 交叉评价与 BioEmu apo 构象参照

日期：2026-10-24

## 问题与范围

本轮只回答一个问题：KVP40 WT NMNAT 的双底物联合占位，以及 Mg²⁺ 同时靠近两种底物的组织方式，是否具有超出 Boltz-2 结构先验和既有 MD 初始条件的证据支持。

本轮以 27 个新运行的 AF3 模型作为独立结构预测：WT ATP+NMN 与 WT ATP+NMN+Mg 各 9 个，1F9A ATP+Mg 参照 9 个；同时复用既有 Boltz-2 联合模型和既有 BioEmu2 apo 帧做跨模型与构象比较。AF3 的输入、运行命令、逐样本坐标哈希和构建/预测/分析身份分别见 `build.json`、`runs.csv`、`identity.json`。没有延长或新增 MD，也没有运行 A1/A2/A3、B1/B2、NatV10/89、Protenix、OpenFold3 或 BioEmu2 配体模式。

## 输入与身份

AF3：

- 本地 AF3 3.0.4，权重 `/mnt/e/models/alphafold3/af3.bin`，SHA-256 `df8bbf2621f17dd3ee21c2a921e84a50bc2b80cdc0c7971cb915c2826fee1f9b`；运行脚本 `/home/yhshy/models/alphafold3/src/run_alphafold.py`，SHA-256 `a1dc5033fdd5473de06a1dfa02bdf3f53120ea83294f880198bcb26e64c2d075`。
- 完整 WT 341 aa，使用既有 512 行 A3M（`data/results/evaluation/kvp40_nmnat_binding/inputs/WT.a3m`，SHA-256 `360babfaa9b5a1a8ac0fa36a630b171c14b328f0ffeb408b0f22347963508cc6`），`run_data_pipeline=false`，`templates=[]`，没有口袋、距离或接触约束。
- 两个状态：ATP+NMN 与 ATP+NMN+Mg。每个状态 3 个 seed × 3 个 diffusion sample，共 9 个模型；`buckets=512`，`num_recycles=10`。
- AF3 另在任务目录顶层写出一个跨 sample 的最佳模型；该文件不是额外独立样本，分析未将其计入。正式模型数为 18 个 WT 坐标。
- 1F9A 参照：单链 A 的观测残基 1–164（canonical 168，未解析末端 QTDK），ATP+Mg，query-only MSA，3×3 共 9 个模型。单链、无同源多聚体、无显式水；只作为输入/解析/局部几何的技术校准，不是训练外验证。

BioEmu2：

- 本地 `bioemu2` 0.1.0rc0，checkpoint `/mnt/e/bioemu2/checkpoints/bioemu2-preview/checkpoint.ckpt`，SHA-256 `3b4ba9236b7b458f47b40b694847b6bb4110c744becd6d2c53de7b1d04c07a81`。
- `struct8`：`predict-structure`，8 帧，seed 20261022；`eq8`：`sample-equilibrium`，8 帧，同 seed。两者是不同模式，不能合并成 16 个同分布样本，也不是物理轨迹或平衡分布。
- MSA 由 BioEmu 内部 `get_msa` 从 ColabFold 获得，使用 `use_env=true`。现存缓存 `/mnt/e/bioemu2/msa/7e90e212b21b14a8a647ae3767cb0b13d3caef551fff889af616530d61ff9310.json` 的完整 query 序列与当前 WT 341 aa 一致，共 8170 行，A3M 文本 SHA-256 `cdf8f65708c57ff4b39852acbcaed62e8566367c38dec25f038dd2fa3d14fda7`。该文件是“按完整 WT 序列匹配到的现存缓存”，两次运行实际使用的 MSA 版本未单独记录；它不是 AF3/Boltz 使用的 512 行 MSA，原始检索服务器数据库版本和日期也未记录，因此不能声称三者同源。
- apo 分析只描述蛋白主链、102–105 环、H14/H17/S133/T135/R138 邻域和 NMNAT–Nudix 相对取向，不对配体或金属作任何推断。

## AF3 结果

9 个模型不是 9 次独立实验重复。以下范围按 seed 分层保留，不把极值当作概率。

| 指标 | ATP+NMN | ATP+NMN+Mg |
| --- | --- | --- |
| ATP 与 NMN 同时位于 NMNAT | 9/9 | 9/9 |
| 跨配体最近重原子 | 2.207–3.620 Å，中位 2.633 | 2.352–3.452 Å，中位 2.949 |
| 磷酸 O–O 最近距离 | 同上 | 同上 |
| ATP PA–O3A–PB 连接完整 | 9/9 | 9/9 |
| 蛋白 severe（<2.0 Å） | 0 | 0 |
| pTM / ipTM 中位 | 0.94 / 0.96 | 0.94 / 0.96 |

PA–NMN 候选亲核氧（三个候选全部保留）：

| 状态 | 候选 | PA–Onuc 中位（范围） | O3A–PA–Onuc 中位（范围） |
| --- | --- | --- | --- |
| ATP+NMN | O1P | 6.070（3.840–6.938） | 84.0°（78.4–124.2） |
| ATP+NMN | O2P | 4.068（3.759–5.649） | 92.4°（83.3–133.3） |
| ATP+NMN | O3P | 4.835（3.454–6.885） | 88.6°（81.9–125.7） |
| ATP+NMN+Mg | O1P | 6.206（3.813–6.780） | 130.7°（127.9–139.7） |
| ATP+NMN+Mg | O2P | 4.152（3.865–5.382） | 140.2°（132.8–145.1） |
| ATP+NMN+Mg | O3P | 4.780（3.580–6.665） | 132.8°（124.6–139.9） |

按 seed 的最短候选：无 Mg 组中位 4.311 / 3.997 / 3.840 Å，对应角 84.9° / 95.5° / 97.4°；Mg 组 4.452 / 4.152 / 3.964 Å，对应角 131.8° / 140.2° / 139.9°。Mg 组角度更钝，但没有更短的 PA–Onuc。三个候选的原子标签是 CCD 标号，不表示化学等价氧的固定角色；没有跨模型挑选“最佳”候选。

“最近候选”的原子身份随 seed 改变：无 Mg 组 seed1 的三个模型均选 O3P，seed2 三个均选 O2P，seed3 为 O2P/O3P/O1P；Mg 组 seed1 三个均选 O3P，seed2 三个均选 O2P，seed3 为 O2P/O3P/O1P。O1P/O2P/O3P 的等价置换会改变“哪一个氧最近”的标签，但不会改变该磷酸基团附近的局部几何。因此，3.926 Å/92.4° 与 3.995 Å/137.9° 是逐模型最低候选的中位，不是固定同一原子标签的配对。可以描述为“输入含 Mg 时 AF3 的预测反应几何发生条件性变化”，不能描述为某个特定氧被金属定向，也不能据此判断催化。

Mg 位置与近邻：

| 指标 | 结果 |
| --- | --- |
| Mg 位于 NMNAT | 9/9 |
| Mg–ATP 任意 O 最近 | 1.946–2.147 Å，中位 2.089 |
| Mg–NMN 任意 O 最近 | 2.379–3.831 Å，中位 3.360 |
| 同时 ≤2.8 Å 接触 ATP 与 NMN 的模型 | 2/9 |
| Mg 2.8 Å 内蛋白 O | 9/9 为 0 |
| Mg 2.8 Å 内蛋白 N | 0–1 个，中位 0 |

按 seed：seed1 Mg–NMN 中位 3.360 Å、0/3 同时近邻；seed2 3.665 Å、0/3；seed3 2.620 Å、2/3。也就是说，AF3 把 Mg 主要放在 ATP 磷酸一侧，并没有像 Boltz-2 Mg_NMNAT 子集或既有 MD 初态那样稳定地同时贴近两种底物。2.8 Å 只表示近邻计数，不是已验证的配位壳或金属桥接；AF3 CIF 不含水，不能评价第一水合壳。

### 跨配体短接触与输入化学状态

| 条件 | seed | <2.8 Å | <2.5 Å | 最小距离 / 原子对 |
| --- | --- | --- | --- | --- |
| 无 Mg | 1 | 1/3 | 1/3 | 2.207 Å，ATP O1A–NMN O3P |
| 无 Mg | 2 | 2/3 | 1/3 | 2.364 Å，ATP O2A–NMN O2P |
| 无 Mg | 3 | 3/3 | 2/3 | 2.399 Å，ATP O1A–NMN O2P |
| Mg | 1 | 1/3 | 1/3 | 2.352 Å，ATP O1A–NMN O3P |
| Mg | 2 | 0/3 | 0/3 | 2.949 Å，ATP O1A–NMN O2P |
| Mg | 3 | 1/3 | 1/3 | 2.470 Å，ATP O1A–NMN O1P |

AF3 的最近对（包括 ≥2.8 Å 的模型）均为 ATP O1A/O2A 与 NMN O1P/O2P/O3P；没有最近对涉及核糖、碱基或蛋白原子。这个原子类型结论只适用于 AF3 的最近对/短接触子集，不能外推到 Boltz 的全部最近对：Boltz 全体最近对还包括 noMg 的 ATP O1A–NMN C5R 3.912 Å 与 Mg 的 ATP O1G–NMN O3P 2.921 Å，两者都不在 <2.8 Å 短接触子集内。2.8 Å 只是筛查值，不是通用 O–O 范德华或碰撞硬阈值；2.207–2.470 Å 的磷酸氧间距不能因为 pTM/ipTM 高、且没有 <2 Å severe 就被视为物理合理。生理条件下磷酸端带负电，但本轮 AF3 使用的 CCD 表示不是 ATP−4/NMN−1。直接读取 AF3 自带 `ccd.pickle`（`/home/yhshy/venvs/natv_af3_v3_0_4_20261008/lib/python3.12/site-packages/alphafold3/constants/converters/ccd.pickle`，SHA-256 `7d73aa94a0b7351e206a3133c7f112e51380f8adb2420fb38d46c2fd4d3666e9`）确认：ATP 47 原子/16 H，总形式电荷 0，PG O1G、PB O1B、PA O1A 为 P=O，O2G/O3G/O2B/O2A 为 P–O 单键并带 H（HOG2/HOG3/HOB2/HOA2）；NMN 38 原子/16 H，总形式电荷 +1（烟酰胺 N1 为 +1），P=O3P 双键，O1P/O2P 单键并带 H（H1PO/H2PO）；MG 为 +2。即 ATP 的磷酸氧部分质子化、NMN 的磷酸部分质子化而烟酰胺带正电。该表示与 Boltz CCD（ATP 0、NMN +1）一致，但都不是 MD 的生理离子态。AF3 输入只给 CCD 代码，CIF 输出也不含水，因此 2.2–2.5 Å 的 O–O 不能直接当成已去质子化阴离子间的排斥来定性“不可能”；反过来也不能因为高 pTM 或 CCD 中性就认为物理合理。需要显式氢键/质子化/金属/溶剂检验才能判断；距离本身不能替代化学检验。

报告中的末端 O 置换对称 RMSD 只是重原子几何比较约定，按去质子化/共振等价末端 O 归一化标签；对字面 CCD 的 P=O/P–OH 并非严格分子图自同构。它不证明 AF3/Boltz 输入的电荷/质子化相同，也不证明化学结合态相同；数值不需重算。

同一模型比较：AF3 独立支持“ATP 与 NMN 都可放在 NMNAT 内”，但没有独立支持“Mg 持续同时桥接两种底物”。它也不支持把更钝的 O3A–PA–Onuc 角直接解释为更接近生产性进攻。

## AF3 与 Boltz-2 的跨模型姿态

用 NMNAT 4–181 的 CA 叠合后比较；AF3 与 Boltz 的 seed/sample 不配对，以下只作集合层面的描述。每对模型都计入一次；81 对不是 81 个独立样本。

下表“C/N 骨架”只包含配体的 C、N 重原子，排除了核糖氧、磷酸氧和 NMN 酰胺氧，不能称为“全部非磷酸重原子”。

| 比较 | 蛋白 CA RMSD 中位 | ATP C/N 骨架 | NMN C/N 骨架 |
| --- | --- | --- | --- |
| AF3 noMg 组内（36 对） | 0.528 | 0.168 | 0.206 |
| AF3 Mg 组内（36 对） | 0.484 | 0.276 | 0.202 |
| Boltz ATP_NMN 组内（36 对） | 0.253 | 0.247 | 0.330 |
| Boltz ATP_NMN_MG 组内（36 对） | 0.287 | 0.289 | 0.368 |
| AF3 noMg × Boltz ATP_NMN（81 对） | 0.644 | 0.382 | 0.660 |
| AF3 Mg × Boltz ATP_NMN_MG（81 对） | 0.782 | 0.431 | 0.754 |

全重原子对称性 RMSD：先按蛋白 CA 配准，不额外做配体自配准；ATP 仅允许同一磷酸的 O1A/O2A、O1B/O2B、O1G/O2G/O3G 各组内部置换，O3A/O3B 桥氧、糖和碱基按原子名固定；NMN 仅允许 O1P/O2P/O3P 置换。它比 C/N 骨架更能判断整体姿态是否相同，也避免等价磷酸氧标号造成的假大差异。

| 比较 | ATP 全重原子对称 RMSD | NMN 全重原子对称 RMSD |
| --- | --- | --- |
| AF3 noMg 组内 | 0.291 | 0.513 |
| AF3 Mg 组内 | 0.389 | 0.492 |
| Boltz ATP_NMN 组内 | 0.500 | 0.774 |
| Boltz ATP_NMN_MG 组内 | 0.508 | 0.636 |
| AF3 noMg × Boltz ATP_NMN | 0.545 | 0.940 |
| AF3 Mg × Boltz ATP_NMN_MG | 0.563 | 0.989 |

对称性感知的磷酸基团 RMSD（仅保留局部定位信息，不代替全配体姿态）：

| 比较 | ATP PA 端基 | ATP PG 端基 | NMN 磷酸 |
| --- | --- | --- | --- |
| AF3 noMg 组内 | 0.272 | 0.231 | 1.050 |
| AF3 Mg 组内 | 0.162 | 0.195 | 0.941 |
| Boltz ATP_NMN 组内 | 0.522 | 0.350 | 1.410 |
| Boltz ATP_NMN_MG 组内 | 0.457 | 0.398 | 1.023 |
| AF3 noMg × Boltz ATP_NMN | 0.436 | 0.659 | 1.486 |
| AF3 Mg × Boltz ATP_NMN_MG | 0.437 | 0.642 | 1.549 |

跨模型中位数普遍高于各自组内中位，但必须看分布重叠。以下为 Q1 / 中位 / Q3（Å）；81 个跨模型对共享同一批模型、只有 3 个 seed，不作独立样本或显著性推断。

| 指标 | 条件 | AF3 组内 | Boltz 组内 | 跨模型 |
| --- | --- | --- | --- | --- |
| NMN 磷酸端 | 无 Mg | 0.779 / 1.050 / 1.305 | 0.983 / 1.410 / 1.996 | 1.214 / 1.486 / 1.917 |
| NMN 磷酸端 | Mg | 0.716 / 0.941 / 1.324 | 0.571 / 1.023 / 1.256 | 1.394 / 1.549 / 1.846 |
| NMN 全重原子对称 | 无 Mg | 0.395 / 0.513 / 0.649 | 0.503 / 0.774 / 1.328 | 0.780 / 0.940 / 1.123 |
| NMN 全重原子对称 | Mg | 0.355 / 0.492 / 0.690 | 0.530 / 0.636 / 0.709 | 0.926 / 0.989 / 1.136 |
| ATP 全重原子对称 | 无 Mg | 0.226 / 0.291 / 0.433 | 0.323 / 0.500 / 0.632 | 0.430 / 0.545 / 0.672 |
| ATP 全重原子对称 | Mg | 0.191 / 0.389 / 0.460 | 0.365 / 0.508 / 0.605 | 0.470 / 0.563 / 0.695 |

- 无 Mg 的 NMN 磷酸：跨模型 IQR 1.214–1.917 Å 几乎落在 Boltz 组内 IQR 0.983–1.996 Å 内，中位 1.486 vs 1.410 Å；NMN 全重原子跨模型 IQR 0.780–1.123 Å 也与 Boltz 组内 0.503–1.328 Å 广泛重叠。不能写“普遍明显超出背景”。
- Mg 的 NMN 磷酸：跨模型 IQR 1.394–1.846 Å 与 Boltz 组内 0.571–1.256 Å、AF3 组内 0.716–1.324 Å 更分离，但仍有边界重叠。这是条件相关的模型间差异，不是独立样本推断，也不是多种天然结合态。
- ATP 跨模型 IQR 与组内 IQR 均明显重叠，中位只高约 0.05–0.25 Å；相对一致性是描述性的。

因此，可说的是“NMN 的精确朝向和局部磷酸组织尚未达到模型间一致”；不能说“跨模型差异显著超出组内背景”，更不能说“发现了两种明确的生理结合模式”。按 seed 的 AF3×Boltz NMN 磷酸中位为 1.31–1.71 Å（无 Mg）和 1.32–2.21 Å（Mg），seed 之间差异明显，也说明当前样本不足以稳定估计模型间差异。

## 1F9A 技术校准

晶体 chain A 的 Mg–O 距离：O2B 2.202、O3A 2.480、O2G 2.604、O3B 2.812 Å；晶体 3 Å 内没有水氧。

| 模型 | 蛋白 CA RMSD | ATP 全重原子 RMSD（原子名） | ATP 全重原子对称 RMSD | ATP C/N 骨架 | ATP PA 端基（对称） | ATP PG 端基（对称） | Mg 位移 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed1_sample0 | 1.296 | 1.441 | 0.885 | 0.974 | 0.724 | 0.590 | 0.979 |
| seed1_sample1 | 1.213 | 1.715 | 1.548 | 0.892 | 0.711 | 3.140 | 1.979 |
| seed1_sample2 | 1.400 | 1.253 | 0.931 | 1.054 | 0.799 | 0.353 | 0.797 |
| seed2_sample0 | 1.063 | 1.304 | 0.783 | 0.842 | 0.837 | 0.548 | 0.818 |
| seed2_sample1 | 0.967 | 1.363 | 0.771 | 0.828 | 0.723 | 0.311 | 0.830 |
| seed2_sample2 | 1.050 | 1.072 | 0.869 | 0.790 | 0.735 | 0.399 | 0.812 |
| seed3_sample0 | 2.184 | 8.472 | 8.435 | 11.405 | 3.958 | 3.714 | 3.635 |
| seed3_sample1 | 2.711 | 3.376 | 3.334 | 4.254 | 1.720 | 2.225 | 3.080 |
| seed3_sample2 | 2.877 | 3.064 | 2.938 | 3.705 | 1.511 | 2.051 | 3.352 |

前 6 个模型把 ATP 放在晶体口袋附近（ATP C/N 骨架 0.79–1.05 Å，全重原子对称 RMSD 0.771–1.548 Å，Mg 位移 0.80–1.98 Å），但没有复现晶体中固定的 O3A 桥氧近邻：预测的 O3A 多在 3.4–3.8 Å，O3B 也未稳定保持晶体的 2.812 Å。端基氧 O2B/O2G 的同名距离（约 2.2 Å 或约 4.5 Å）受等价氧标号置换影响，只作为原始原子名附表保留，不能据此判断整个磷酸端基是否离开。seed3 的 3 个模型整体失败（ATP 全重原子对称 RMSD 2.938–8.435 Å，Mg 位移 3.08–3.64 Å）。

这里的 6/9 不是六次独立成功：seed1、seed2 各 3 个样本表现较好，seed3 的 3 个样本整体偏离；seed 才是重复单位，3 个 seed 也不能保证覆盖更多真实构象态。参考体系只使用 1F9A 单链 164 aa 观测序列和 query-only MSA，与 WT 的 512 行深 MSA、全长 341 aa 和不同组装背景不匹配；晶体也没有解析完整水合壳。因此该 6/9 不能作为严格通过率、AF3 一般金属能力测试或“AF3 能/不能预测金属配位”的总体判断；它只说明在本输入下，AF3 的口袋/配体重建部分可用，但不足以仲裁 Boltz/MD 的金属组织。原子名 RMSD 作为原始附加项保留；判断整体姿态应看对称性列。

## BioEmu apo 构象参照

以 NMNAT 4–181 叠合后，比较 102–105 环 + H14/H17/S133/T135/R138 的 CA、Nudix 204–338 的 CA，以及叠合误差。每个 holo 模型对 8 个 apo 帧，共 72 对/来源/模式；这些是描述性配对，不是概率或诱导契合证据。

| apo 模式 | 来源 | 局部 CA RMSD 中位 | Nudix CA RMSD 中位 | 叠合 CA RMSD 中位 |
| --- | --- | --- | --- | --- |
| struct8 | AF3 noMg | 1.478（0.922–3.018） | 2.471（1.112–3.698） | 0.605 |
| struct8 | AF3 Mg | 1.421（0.694–3.348） | 2.205（0.967–3.308） | 0.609 |
| struct8 | Boltz ATP_NMN | 1.388（0.833–1.839） | 1.046（0.491–1.956） | 0.563 |
| struct8 | Boltz ATP_NMN_MG | 1.348（0.797–2.001） | 1.285（0.650–2.203） | 0.606 |
| eq8 | AF3 noMg | 3.563（2.892–5.175） | 3.170（1.784–10.239） | 1.812 |
| eq8 | AF3 Mg | 4.147（2.846–5.452） | 3.358（1.592–10.508） | 1.879 |
| eq8 | Boltz ATP_NMN | 2.886（2.446–3.420） | 4.139（1.248–11.534） | 1.699 |
| eq8 | Boltz ATP_NMN_MG | 2.938（2.432–4.025） | 4.341（1.634–12.195） | 1.671 |

两种模式的 apo 帧都不是同一分布：holo 模型与 `predict-structure` 帧的局部主链更接近（约 1.3–1.5 Å），与 `sample-equilibrium` 帧差异更大（约 2.9–4.1 Å）。这只说明现有 apo 样本中存在与 holo 主链大致相容的构象，不能证明结合不需要诱导契合，也不能给出构象概率；8 帧样本极小。

## 可以说什么，不能说什么

可以支持：

- AF3 在无约束、只用序列和 CCD 配体的条件下，独立地把 ATP 和 NMN 都放进 NMNAT，18/18 WT 模型如此；ATP 链内几何完整，无 severe 冲突，跨配体磷酸端保持 2.2–3.6 Å 的近距离。
- AF3 与 Boltz-2 的 ATP 放置和非磷酸骨架差异较小；NMN 的精确朝向和局部磷酸组织尚未达到模型间一致。无 Mg 的 NMN 磷酸跨模型 IQR 几乎落在 Boltz 组内 IQR 内（1.214–1.917 vs 0.983–1.996 Å），Mg 条件更分离但仍有重叠；不能写成普遍显著超出组内背景，也不能解释为多种天然结合态。
- AF3 的 Mg 组并没有稳定形成同时贴近 ATP 与 NMN 的金属组织（仅 2/9 同时 ≤2.8 Å），其金属主要贴 ATP 磷酸；这不支持“金属桥接是跨模型稳健特征”。
- AF3 的 1F9A 校准只部分成功：seed1、seed2 的 6 个样本保留口袋与 ATP 位置，seed3 的 3 个样本严重偏离；这是两个 seed 好、一个 seed 坏，不是 6/9 次独立成功。即使保留口袋，Mg–O 配位也未复现。因此 AF3 不能作为金属配位的仲裁者。
- AF3 的局部短接触显示：无 Mg 6/9、Mg 2/9 模型有 <2.8 Å 的跨配体磷酸 O–O 接触，最低 2.207/2.352 Å。高 pTM/ipTM 和无 <2 Å severe 不能证明这些接触物理合理；输入只给 CCD 代码，未显式指定质子化/电荷，CIF 无水。这是局部化学一致性不足的筛查信号，不是结合姿态的判据。
- 现有 apo 帧中，`predict-structure` 模式与 holo 主链的局部差异约 1.4 Å，`sample-equilibrium` 更大；这是相容性描述，不是构象分布或诱导契合证明。

不能支持：

- 不能把 AF3 的 9 个 sample 当作 9 次独立实验重复，也不能把 18/18、6/9 或 2/9 当作概率、占有率或自由能结论。
- 不能把 AF3 的 Mg–ATP 2.1 Å 或更钝的 O3A–PA–Onuc 角称为已验证的金属配位、桥接或生产性进攻几何；也不能把 2.2–2.5 Å 的磷酸 O–O 短接触当作物理合理，或把 <2.8 Å 当作通用碰撞硬阈值。
- 不能把跨模型 NMN 差异称为“显著超出组内背景”或“两种明确的生理结合模式”；当前只有“原子级姿态尚未达到模型间一致”。
- 不能把 AF3 与 Boltz 的一致称为独立实验验证；两者可能共享训练先验、同源序列和建模偏差。
- 不能从 apo 主链相容推断结合机制、催化活性、kcat/Km、NAD⁺ 产量，或对 A1/A2/A3 排序。
- 不能把 1F9A 单链/query-only 校准当作训练外验证或 AF3 一般金属能力测试，也不能把其晶体配位当作当前力场或模型的验证标准。

## 对研究判断的影响

本轮把“联合占位”和“金属桥接”进一步分开：

- 联合占位：Boltz-2 与 AF3 都支持 ATP 和 NMN 可同时落在 NMNAT；这比单一模型的先验更稳，但仍只是结构预测层面的相容性。
- 金属桥接：AF3 在多数模型中不把 Mg 同时贴近两种底物，且 AF3 对 1F9A 的 Mg 配位重建不可靠；现有 MD 的持续桥接也不能独立证明自由能稳定性。金属组织仍然是模型和初始条件依赖最强的部分。
- 反应几何：AF3 的 PA–Onuc 距离和角度没有给出可判定的生产性几何；本轮没有预定义生产性阈值，也没有与同输入的单底物姿态作配对比较，因此不能据此否定真实体系催化。

本轮作为 WT NMNAT 多模型结构研究的审阅节点收口：暂停继续追加同类共折叠/金属预测，不启动 Protenix、OpenFold3、BioEmu 配体采样或更长 MD。`6629b2f` 继续作为 WT 显式溶剂初始响应结果保留。AF3 结果支持“双底物联合占位值得继续作为结构假说”，但没有确认一个稳定的金属态，也不支持把 Mg 桥接提升为稳健机制结论，或据此启动 A1/A2/A3 MD、变体活性排序。下一步回到 NatV 序列设计：复核 A1/A2/A3 的完整组合、可追溯的接触/基序依据及可被功能实验区分的假说（见[设计报告新增表](../natv_dual_domain_design/REPORT.md#a-类设计依据事实假说与可区分结果)），不以当前结构预测排序。若未来出现能约束 KVP40 NMN 识别或 Mg–ATP/NMN 配位方式的独立结构/生化/物理化学证据，可再考虑同一 WT–ATP–NMN 起点下的受控金属比较并评估 Mg²⁺–水–磷酸参数；这是可选方法，不是预设的下一步。

## 文件与复现

- 输入：`inputs/`（两个 WT JSON、1F9A JSON、manifest、build.json）
- 身份：`build.json`（构建时输入与代码）、`identity.json`（构建/实际预测/最终分析分列）、`runs.csv`
- 运行：`runs.csv`；`logs/`、`runs/<condition>/` 是本地原始运行目录，不在公开归档中
- 逐模型：`cifs/`（27 个逐样本 CIF.gz，不含重复 best，也不含 confidence JSON）、`cifs/manifest.csv`、`models.csv`、`reaction_candidates.csv`
- 姿态：`pose_pairs.csv`、`pose_summary.csv`、`cross_model_pose_pairs.csv`、`cross_model_pose_summary.csv`
- 参照：`reference_reconstruction.csv`
- apo：`apo_compatibility.csv`、`apo_compatibility_summary.csv`、`apo_identity.json`、`apo_inputs/`（两套未压缩 `topology.cif`/`samples.xtc`、`system.yaml`、`command.txt`；配置的 `apo.sets` 指向这里）
- 汇总：`summary.json`（`confidence` 字段记录可用 summary 数）

复现命令：

```bash
# 在已归档结果目录上复算几何、姿态、参考重建与 apo 比较；不重跑预测
PYTHONPATH=src python -m natvqvae.cli nmnat-af3-analyze --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3
PYTHONPATH=src /home/yhshy/miniconda3/envs/bioemu2/bin/python -m natvqvae.cli nmnat-af3-apo \
  --config config/nmnat_af3.yaml --output data/results/evaluation/natv_nmnat_af3

# 从零重跑预测时，build 必须指向新的空目录；不要把新结果写进已归档目录
PYTHONPATH=src python -m natvqvae.cli nmnat-af3-build --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3_reproduction
PYTHONPATH=src python -m natvqvae.cli nmnat-af3-run --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3_reproduction --wait-gpu
```

公开归档不含 `runs/` 和原 E 盘路径；`nmnat-af3-analyze` 会从 `cifs/` 读取坐标，`nmnat-af3-apo` 会从 `apo_inputs/` 读取小输入，因此上述两条复算命令可直接执行。该复算不包含 pTM/ipTM：这些值只存在于未归档的逐样本 `*_summary_confidences.json`，缺失时留空且不汇总，不按 0 处理。最后一条 apo 命令需要 bioemu2 环境中的 mdtraj；普通几何分析不需要 mdtraj。
