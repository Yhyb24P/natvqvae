# 已完成结果与证据边界

## 条件 QVAE

- 1,644 条 training、227 条 validation、5 个配对正式 seed。
- 完整目标验证 NLL：QVAE 后验重要性采样 1.737290，条件先验 1.737287，自身边缘 1.737355，确定性对照 1.738152（nats/目标残基）。
- QVAE 相对确定性有小幅改善，但自身边缘独立模型同样改善；共享 z 相对自身边缘仅约 6–7e-5 nats/残基。成对/三位点共变统计区间多包含零，候选供给与确定性基本相当。
- 结论：没有足够证据支持有意义的共享潜变量联合建模或候选供给优势。该结论来自 227 条开发验证序列，test 未参与。

## PF01467 语料恢复与迁移资格

- 旧 Stage1–3 的 13,972 条中，双域部分 2,391 条有序记录全部已在当前 NatV roles；不能作为新增双域训练数据。
- 8,849 条 PF01467-only 原始全长序列及 26,695 个来源标识/header 对应已恢复。NatV 主任务新增 0；486 条来源记录提供较广 WT 对应片段、5,991 条提供局部核心，二者均不自动证明完整生物学域。旧固定搜索的 63 条/19 簇与本次不同对齐、覆盖定义所得的 486 条不能直接比较。
- 辅助输入去重为 4,164 条序列：默认训练 2,395 条/152 组、验证 1,684 条/38 组，另有 85 条风险序列。实际辅助域与现役域共同聚类，按 50% identity、80% 双向覆盖分组；30% 层仍有连接现役背景的远缘同源，不能称完全独立。
- 默认训练中仅 40 条/9 组带单独 NMNAT EC 注释；NadR、双 EC、文本和未知层分别保留，广义辅助池不能整体充当功能正例。训练输入覆盖相对现役训练未见的 1,311 种位点—残基事件，属于可用序列信息，不是已证实的迁移收益。逐层非冗余统计、实际输入与复现方法见[数据报告](../data/results/preprocessing/pf01467_8849/REPORT.md)。
- 从有界来源得到 7 条序列/注释支持的 NMNAT 设计假设：L18I、F13Y、T22Q、A134G、K130R、L140M、F13Y+A134G；这些不是功能或活性结论。

## 候选池诊断

- 现有 15,360 条/模型的 NMNAT 候选含多数单点假设，但不含 F13Y+A134G 配对。
- 现役 QVAE 历史代表 0/3 命中；QVAE 与确定性证据重排均为 3/3、6 个支持事件；没有 held-out 证据，不能作为独立验证或模型优势证据。
- 配对缺失是供给限制，不补采、不据此推断模型能力。

## NMNAT 辅助域 QVAE

- 单 seed 的 QVAE 与同信息确定性模型各训练 3,000 步，均采用第 10 轮最佳权重。2,395 条域训练、1,684 条开发验证；固定 128 条背景选权重后，两臂均出现验证退化。
- 全开发验证 NLL 为 QVAE 后验 IS 1.8743113、先验 MC 1.8741606、确定性精确值 1.8744126；38 个域组的配对比较区间包含零，共享潜变量相对自身边缘的差值区间也包含零，没有明确优势证据。
- 完成 18,432 次两臂生成；其中每臂 3,072 次是 KVP40 WT 域条件生成，得到 10 条代表记录、9 条不同的 341 aa 序列。两臂各生成一条含 F13Y+A134G 的多替换候选，不构成该组合的独立功能验证，也不是 QVAE 独有供给。
- 本次验证的是辅助域模型与候选供给，没有检验辅助预训练对完整 NatV 主模型的迁移收益。结果、训练曲线、代表序列与复现依据见[研究报告](../data/results/evaluation/aux_qvae_seed8601/REPORT.md)。

## 完整 NatV 辅助迁移

- 随机初始化与辅助域 QVAE 初始化两臂，各 3 个主训练 seed、2,704 steps、同一 27-mask 训练与 384-query NMNAT 选择规则。六次训练合计 2.12 h；辅助预训练 18.95 min 单列，只有一个辅助 seed。
- 全 227×27 开发验证、按 44 个 component 与 seed 配对：辅助初始化使 NMNAT 条件先验 NLL 改善 0.0490 nats/残基，95% CI [-0.0603, -0.0376]；global 区域掩码改善 0.0170，CI [-0.0268, -0.0070]；全部掩码加权总体平均差约 0.0134；Nudix 差值 +0.0067，CI [-0.0063, 0.0198]，未检出明确差异，但不能排除约 +0.02 退化。三个主 seed 的 NMNAT 差值分别为 -0.0412、-0.0408、-0.0650。overall CI 是固定 3 个训练 seed 后对 component 做 bootstrap；四次重复的 NMNAT MC SE 为 2.1e-5–7.0e-5。辅助预训练额外步骤未与 random 臂匹配，收益属于辅助初始化整体方案效应。
- 18,432 次共同 mask/uniform 生成（每臂 9,216）中，两臂 unique、口袋/锚点编辑没有显示辅助初始化优势；原始 Hamming 和锚点变化在 auxinit 略低，Hamming 匹配后口袋行比例接近。说明条件概率迁移没有同步转化为更多或更口袋定向的候选供给。
- 结构：辅助域 9 条唯一候选 + WT、迁移 3 对 Hamming 匹配代表 + WT 均完成 8 帧 BioEmu2 prediction，全部 8/8 physicality 通过。迁移配对 median full RMSD 0.54–0.62 Å、NMNAT 0.28–0.46 Å；H15 的 I145 邻域较大，配对间无一致方向。结构仅作条件性证据；2QJO 是 NAD⁺ 产物参照，2QJT/2R5W 不作为 NMNAT 锚点模板；无配体、无 pLDDT/PAE、无活性结论。完整报告与复现路径见[迁移报告](../data/results/evaluation/natv_transfer/REPORT.md)。

## NatV 双域 A/B 设计

- 已完成 WT + 3A + 2B + 6A+B 的 12 条候选设计及 36 个 Boltz-2 单体结构。联合候选相对各自亲本未显示明显的骨架和 NMNAT 锚点邻域破坏；这不等于 NMNAT 活性保持或 NAD⁺ 净产量提高。序列、结构、位点对应和复现方法见[设计报告](../data/results/evaluation/natv_dual_domain_design/REPORT.md)。
- A2 的 NMNAT 域自身预测样本 RMSD 中位数约 1.78 Å，WT 约 1.26 Å；T135、R138 与 130–141 邻域的自身样本离散也较大，两个 A2 联合候选延续该现象。它表示局部预测不确定性，不能解释为真实柔性增加或作为淘汰依据。
- 所有候选共用含 WT 相同序列及近缘同源的 MSA。该条件下的 WT-like 预测不能区分先验约束与实际骨架保持，现阶段不据此重算。B1 E248Q/B2 D308N 仍为基于同源金属催化环境的待测假说；同源寡聚体只提供组装参照，KVP40 实际寡聚状态尚未确定。

## Nudix 金属催化微环境计算（2026-10-23）

- D2（AMP+3 Mn）与 D4（NAD⁺+3 Mg）的 WT/B1/B2 共 54 个 Boltz-2 模型先前产生；现由正式 CLI 校验并汇总：D2/D4 各 18 个约束态（2 seeds × 3 samples）+ 9 个无约束对照；每 run 的样本数、序列、配体与离子组成均有校验，指标可由 `microenv-summarize` 重算。
- 约束模型中，突变体的酰胺羰基 O 到最近簇金属的距离通常高于 WT 羧酸 O 的最近距离：B2 D2 5.41–5.97 Å（WT 2.11–2.93）、D4 2.92–6.42 Å（WT 2.31–2.85）；B1 D2 2.56–2.86 Å（WT 2.05–2.19）、D4 2.08–3.23 Å（WT 1.90–2.53）。三金属簇内位点归属未建立（C/D/E 标签可互换），模型无显式水，D307 直接配位与晶体不一致，因此不能判配位丢失、结合强弱或功能后果。
- 无约束时 AMP 与 NAD⁺ 均落在 NMNAT 位点，D2/D4 的配体姿态是约束条件态；L9 在 Nudix 域对齐后的静态 RMSD 为 0.71–0.85 Å，不能排除域间重排。未进入 MD；NAD⁺ 净生成与 B1/B2 机制仍未定。详见[微环境报告](../data/results/evaluation/natv_nudix_microenvironment/REPORT.md)，后续计算的选择依据见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。
- 54 个模型的原始 CIF 已按确定性 gzip 选择性归档到 `coordinates/`，`coordinates_manifest.csv` 记录原始/压缩 SHA256 与路径；远端可复算原子几何，未提交 confidence/缓存/重复副本。

## Nudix 实验结构锚点核查（2026-10-23）

- 核查 5IW4（WT E. coli NudC–NAD）、8ZB4（WT M. abscessus NudC–NAD）、5ISY（E178Q）和 6O3P（E219A/E220A/E221A，AMP+3 Mg）。5ISY 功能对照显示 E178Q 使 NAD⁺ 水解几乎不可检出但仍结合 NAD⁺；6O3P 与 5ISY 均为突变背景，8ZB4 实际沉积金属为 Ca²⁺。
- 5IW4/8ZB4 的腺嘌呤/核糖由另一亚基 Tyr124/Phe156 接触，5IW4 原文明确催化口袋由两个单体共同组成；现有单链模型未支持完整口袋迁移。6O3P 不能作 WT 催化簇参照；8ZB4 NAD 与 6O3P Mg 刚性组合有 1.13–1.30 Å 位阻。
- 结论：不构造 D4_exp；D4 仍为条件性假说。原 54 模型 D4 距离差来自组成/约束共折叠，不是模板转移结果；B2 效应既不能证明也不能否定。详见微环境报告。

## 易错 PCR 全长克隆结构比较（2026-10-23）

- Sanger 确认 NatV10 为 D83V/M162V/L317P/M323K，NatV89 为 I21N/F137Y/I156T/P301S/V320E（另有 I187I 同义）；WT/10/89 使用同一 512 行 MSA、Boltz-2 2.2.1、seed 20261025、3 samples，共 9 个无配体单体。
- 对 WT 的 Cα RMSD 中位：NatV10 full 0.87、NMNAT 1.00、Nudix 0.48 Å；NatV89 full 0.84、NMNAT 0.74、Nudix 0.42 Å；整体在自身样本离散附近，无整体折叠或域级分离证据。
- 局部 Cα 位移高于同位置自基线的是 L317P、M323K、I156T、V320E；D83V、M162V、I21N、F137Y、P301S 与自基线相当。域质心距离 31.3–31.7 Å，仅距离、不测方向。仅预测结构，不推断活性/稳定性；NatV10/89 是多突变克隆，不是 B1/B2。详见[结构比较报告](../data/results/evaluation/kvp40_epcr_full_length_structures/REPORT.md)。

## NMNAT 状态分层接触映射（2026-10-23）

- 使用与 A1–A3 同一 Boltz-2 运行的 WT 全长三模型（seed 20261022、3 samples、apo、pLDDT 0.922、pTM 0.913），把 1F9A ATP/Mg²⁺、1NUS NMN/APC 和 2QJO NAD/POP 的实验接触经 TMalign 对齐后刚性迁移；不做诱导契合、MD、对接或新预测。结果在 `data/results/evaluation/natv_nmnat_state_mapping/`。
- 1F9A：61 个源接触记录全部有映射，50 个高几何匹配；35–36 个邻近记录；Mg–ATP 氧 2.2 Å 是刚性变换不变量，不代表 KVP40 配位恢复；配体–蛋白最近 0.29–2.25 Å，2/3 模型严重重叠。
- 1NUS：77 个源接触记录中 74 个映射，35–38 个高几何匹配；35–42 个邻近记录；NMN–APC 最近 4.13 Å，配体间无严重互相穿插；但配体–蛋白 0.20–1.22 Å，3/3 模型严重重叠。K140′–APC O1G 源距离 2.73 Å；迁移后最近碱性残基为 K104（4.08–4.95 Å，弱候选/重叠）或 R138（7.73 Å，无候选），**没有一致的界面替代证据**。
- 2QJO：101 个源接触记录全部映射，79–86 个高几何匹配；66–77 个邻近记录；NAD–POP 最近 2.66 Å；配体–蛋白 0.24–1.36 Å，3/3 模型严重重叠。
- 邻近记录与 close/severe 冲突不互斥：模型 0 的 1F9A 36 个邻近记录中 29 个同时 close/severe，1NUS 42/42，2QJO 75/49；不能把行数当独立恢复相互作用数。
- 变体位置选择：A3 A134G 在三种状态均有 direct 几何邻近（2.25–3.10 Å）；A3 Q86I 仅 1NUS direct；A1 L18I、A2 T22Q 为 motif 候选；其余未入选。未入选不代表低活性；三状态不是独立验证，没有性能优选。
- 1F9A 原子/基团审计（`natv_nmnat_atp_audit/`，重做版）：61×3 记录中，逐模型 supported 24/28/34、role_mismatch 14/14/14、not_recovered 12/12/13、clash 11/7/0；跨模型 21 个 3/3 且无 clash 的 supported_consistent（16 主链–主链、5 侧链特异），8 个 2/3 支持但第三模型 clash，2 个单模型支持，5 个单模型支持且带 clash，14 个 role_mismatch，11 个 not_recovered。21 个支持只是角色匹配+距离阈值邻近；WT model0 中 17 个为 close、含 5 个主链 C，不能当合理物理相互作用或完整 ATP 口袋。
- 侧链特异支持：H13/H16（KVP40 H14/H17）NE2→ATP O1B/O2A 为 3/3；O3A/O5′/PA 为 2/3 且第三模型 clash；T127（T135）OG1→O1G 为 3/3，但 T127 主链 N→O3G 是独立主链接触，不能当羟基作用保留；R130（R138）NH2→O3G 为 3/3（目标 NH1 为对称等价胍基 N）。His 质子化/互变异构未定，不能称已证实氢键。
- 在当前映射/原子角色规则下，14 个源接触为 role_mismatch（如 1F9A R121 侧链对应 KVP40 M127、MET118 SD、VAL19 CB、ILE5 CD1），不能作为支持；这是条件性结果，不证明真实口袋不可迁移。21 个低几何置信行全部同时为 role_mismatch，`atp_contacts.csv` 保留 low 与距离/冲突字段。1F9A 的 Mg 在 3.2 Å 内无蛋白接触；审计未显式分析水，不能确定完整水配位层或 KVP40 金属位点。
- 旧 `e1db337` 中基于“最近任意重原子”的 T135/H17 侧链支持、旧 17 个计数、R9 chi/移植和 96–108 loop 临时数字均已撤回；loop/R9 没有正式实现，未进入本结论，也不启动 repacking/MD。
- 边界：预测模型为 apo，刚性迁移的蛋白–配体冲突说明需要侧链/环区重排，不说明配体不能结合；fit RMSD 1.36–2.31 Å，单个近距离不能精确解释；没有实验验证。详见[映射报告](../data/results/evaluation/natv_nmnat_state_mapping/REPORT.md)和[ATP/Mg 审计](../data/results/evaluation/natv_nmnat_atp_audit/REPORT.md)。

## ePCR 克隆的 NMNAT 底物环境（2026-10-23）

- 使用既有 9 个 ePCR 全长模型（WT/NatV10/NatV89 各 3，seed 20261025），在共同 WT0 的 NMNAT 4–181 框架中，把 1F9A ATP/Mg、1NUS NMN/APC、2QJO NAD/POP 的实验姿态各迁移一次；不重跑预测，不做 MD/对接。结果在 `data/results/evaluation/kvp40_epcr_nmnat_environment/`。
- 五个 NMNAT 突变的新增/丢失侧链原子均未直接接触迁移配体：D83V 羧基丢失但侧链距配体 7–21 Å；M162V 硫醚 19–20 Å；I21N ND2/OD1 7.5–7.8 Å；F137Y OH 7.5–8.7 Å；I156T OG1 10–11 Å。没有证据支持这些侧链直接改变底物环境。
- 接触汇总（3/3 无 clash 的 `supported_consistent`）：1F9A ATP 为 WT 16、NatV10 16、NatV89 2；2QJO NAD 为 30/31/30，POP 为 1/5/0；1NUS APC 为 7/8/3。NatV89 在 ATP 和 POP 上出现更多模型依赖 clash（1F9A 32 个 with-clash；POP 20/24 至少一个模型 clash），但 1F9A 主要由 model 0 驱动、model 2 恢复较多，且不在突变侧链上。只能作为 F137Y 与 R138/T135 局部耦合的可检验假说，不是功能结论。
- 1NUS 的全局对齐最弱（TM 0.58–0.68，TMalign RMSD 3.25 Å）；NMN 每个克隆有 24/31 个 role_mismatch、3 个 unmapped，不能判断 NMN 口袋。锚点 H14/H17/T135/R138 的 CA 变化均 <0.5 Å，没有重排证据；距离近不代表氢键或催化几何。
- 边界：固定迁移姿态、3 个 diffusion 样本；WT 自身迁移姿态也有 clash；不能作活性/产量排名，不能说明真实结合姿态、顺序、限速步或金属配位。详见[ePCR NMNAT 环境报告](../data/results/evaluation/kvp40_epcr_nmnat_environment/REPORT.md)。

## 证据边界

- 30–35% identity 处于 twilight zone；局部比对、域边界和锚点对应仍可能不确定。
- 2QJO 是 NAD⁺ 产物参照，没有 ATP/NMN 三元结构；锚点距离不是催化几何。
- 70% 序列簇不是独立家族；reviewed annotation 不是实验测定。
- 所有候选的活性、稳定性、底物特异性、kcat/Km 与 NAD⁺ 净积累仍为 UNKNOWN。
