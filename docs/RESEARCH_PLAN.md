# NatV 研究计划

## 研究对象与目标

KVP40 NatV 为 341 aa 的 NMNAT–Nudix 双结构域酶。工程目标是在匹配条件的纯化酶体系中，使 NMN + ATP 生成 NAD⁺ 的**净积累**超过 WT，同时保留具有应用价值的 NMNAT 催化能力；Nudix 的 NAD⁺ 水解可以下降或丢失，不能因此默认其结构域可以删除。

NMNAT：`NMN + ATP ⇌ NAD⁺ + PPi`；Nudix 的已报道 NAD⁺ 水解：`NAD⁺ + H₂O → NMN + AMP`。净积累、NMNAT 初速度和 Nudix 水解是不同实验终点。需记录酶量、NMN/ATP、Mg²⁺、pH、温度、PPi 处理和测定时段；NMNAT 活性的应用门槛由未来 WT 动力学数据确定。

**主问题：** 在保持 KVP40 骨架的有限变异范围内，怎样产生多样、合理且能用于检验 NMNAT 保留和 Nudix 水解扰动的蛋白序列？计算结果只提出机制假设；实测净积累才决定工程目标是否达成。

## 生成任务：保守 WT 条件编辑

候选以公开 KVP40 全长 WT 为条件，优先探索 **90%–100% WT 全局序列相似性**的变体。相似性按完整序列全局编辑距离定义：

```
I_WT = 1 − Levenshtein_distance(candidate, WT) / max(len(candidate), 341)
```

这个定义按单残基替换、插入、删除计费；对应的全局对齐和 `KEEP/SUB/INSERT/DELETE` 编辑脚本仍须保存，以免高相似局部片段、截短或插入抵消删除造成误判。90% 对 341 aa 大致相当于最多 34 个残基编辑；这是设计空间边界，不是功能安全阈值。WT 零编辑仅作参照，不计新候选。训练数据不受 90% WT 相似性限制。

核心任务是 **固定 WT 骨架的多位点替换和条件重采样**：可更新位置与编辑数量在采样协议中明确，不再固定 4/8 位点；其余 WT 残基及已修改位置成为生成上下文，输出必须是完整氨基酸序列。连续迭代时重算当前序列条件，且保留编辑次数、位置和各步变化记录。90%–100% 邻域按实际编辑距离分层比较，不能用事后筛选代替模型的生成能力。

允许 **有限短 InDel**，以一次 1–3 aa 局部插入或删除作为首个可检验的操作范围；这只是待研究的起始范围，不意味着已发现普适安全区。先使用明确的局部事件和 WT 坐标编辑，后以完整变体为条件尝试邻域补偿性替换；不将独立采样的对齐列 gap 直接当成具有生物学依据的缺失。InDel 的位置应结合训练群体的可靠事件、保守锚点及结构背景，并与替换生成分别统计。G1-b 的 8 个事件定义是可供审查的演化候选，不是已验证安全编辑区域。

A、B、C 是共用生成器的设计假设：**A** 探索 NMNAT 环境的保守改善；**B** 探索 Nudix 水解相关扰动并保留 NMNAT；**C** 联合探索。它们不代表已知酶学效应，也不要求分别训练模型。NMNAT 关键位点优先保持并检查其邻域；Nudix 关键位点可为有目的的设计对象，不能将天然保守性当作永久不可编辑禁令。历史 `contracts/regions.json` 的替换规则适用于旧 341 aa 试验，不是本任务的 InDel 法则。

## 生成模型研究与实际设计

六环节主线保持：① 高质量数据 → ② 生成模型研究 → ③ 工程化序列生成 → ④ 计算评价 → ⑤ 优选候选库 → ⑥ 未来酶学验证。当前要把已有家族数据和生成模型用于有明确生物学问题的 **KVP40 保守序列设计**，而不是继续扩大方法排名。

**核心假设：** 已训练的 Categorical QVAE 潜变量可能编码了 NatV 家族的某些联合序列规律，在 WT 条件下可用于探索不同的多位点组合。这是有依据的研究假设，尚无证据表明其潜变量已经编码 NMNAT 合成、Nudix 水解或其他特定功能。潜变量是否逐维可解释并非使用前提；改变 z 导致 logits 改变，仅证明模型响应 z，不证明生成有生物学收益。

**主要生成方法**保留旧 H2 Categorical QVAE 权重及其当前序列条件掩码恢复能力；训练目标、模型使用的输入信息及实际 prior 采样应与生成任务对应。以冻结 ESM-2 条件掩码编辑作为必要参照，并保留 WT 和简单匹配编辑用于解释候选。SE、列频、Continuous MMD-VAE 及其他旧模型的运行结果由原报告保留，不再默认重新训练或进行四臂排名。出现明确的新科学问题、能够区分的评价证据时，再决定是否比较或调整架构，而不是为了获得正结果反复调参。

**从给定位置恢复走向实际设计。** 旧研究在共同随机编辑位置上对比各模型的氨基酸采样；这并不等于模型会选择适合 NatV 工程改造的位置。后续依据 NMNAT 保留、Nudix 扰动及联合改造的假设，结合真实家族变异与结构对应，确定可研究的编辑区域，再由 QVAE 对当前完整序列进行条件生成。编辑位置可由生物学假设确定或由模型提出，但须说明来源；不把频率高、模型概率高当成潜在酶活提升。维持 WT 邻域约束、真实多位点组合、当前上下文更新及真实采样来源。短 InDel 是可研究的特殊编辑，不另立必须胜出的变长模型。

**评价回答生成和设计问题，而非构造排行榜。** 对实际提案保留编辑脚本、先验 z、完整序列、相似性、重复及失效记录；检查不同 z 或条件是否能提供不同的多位点组合，但不将 z 响应当作功能性上位效应。HMM 双域命中、序列新颖率和固定共同位点内的唯一组合比例在近 WT 条件下可能饱和，不能作为主要优劣证据。对有明确且不同设计假设的代表性候选，进一步用候选自身结构及催化环境分析其合理性与不确定性。候选选择不依赖 QVAE 是否优于 ESM-2；新模型比较只有在可能改变设计决策时才开展。

正式生成采用现役训练及检索群体；保留验证与测试边界，评价者与选位/采样分数共享先验时如实说明。候选是否真实提高 NAD⁺ 净积累，仍需未来酶学测定。

## 计算评价和候选库

正式生成结束后，以该次生成的真实候选为对象，按以下证据层次分析：**序列与家族比对 → 候选自身全长预测结构 → NMNAT/Nudix 催化相关几何和已存放实验结构参照 → 代表性候选库**。实验结构和残基对应以 [NATV_PAIRING_AUDIT.md](NATV_PAIRING_AUDIT.md) 为入口。计算阶段不能把模型分数、pLDDT、对接或几何距离写成预测 `kcat` 或净产量。

结构评价要区分已观测结构、预测结构、家族比对及相同模板来源引起的相关证据。特别检查 NMNAT 催化相关位点、Nudix 可能的水解扰动，以及不同长度变体的局部几何与双域相对组织；未测量记 UNKNOWN，不把非保守 Nudix 替换自动认定有害或有益。

候选库应包含 WT、匹配预算的简单编辑对照，以及覆盖 A/B/C 不同机制假设的代表性序列。模型选择和工程候选选择是两个不同问题；生成方法没有优于 SE 的证据，仍可产生值得研究的候选。

## 已有结果及其适用范围

[保守条件编辑运行](../experiments/natv_conservative_editing/REPORT.md)生成 2,976 条符合编辑约束的提案。四种方法的双域 HMM 命中接近，高编辑预算下共同位点内不同组合比例达到饱和；现有评价不能有效判定 QVAE 与 SE 的生物学价值高低。随机指定编辑位置后生成氨基酸，并非已完成有生物学依据的自主选位。已有 z 和上下文响应仅证明参与计算，不是 NatV 功能机制证据。

[WT 对齐列先验生成](../experiments/natv_wt_columns/REPORT.md)的全列独立采样产生大幅 WT 偏离及 gap，不适合作为当前保守设计的现成方案。[历史局部编辑](../experiments/natv_candidate_sequences/REPORT.md)只覆盖较少位点的条件提案。方法身份、划分差异与复用界限见 [模型继承记录](../discovery/natv_model_inheritance/model_reuse_decision.md)；历史负结果仅限各自运行任务，不构成对 QVAE 在其他生成任务中无效的证明。

## 未来酶学验证

在固定纯化体系测定 WT 与候选的 NMNAT 初始合成速度、Nudix 对 NAD⁺ 的水解，以及 NAD⁺ 净积累时间曲线；条件允许时分析 `kcat` 与表观 `Km`。WT 是必要参照。具结构完整性的 Nudix 失活变体及 NMNAT 失活变体属于待验证机制对照。只有实测净积累提高且保留可用 NMNAT 合成，才能成立工程改进结论。

## 文献依据

Huang, N., et al. (2008). Bifunctional NMN adenylyltransferase/ADP-ribose pyrophosphatase: Structure and function in bacterial NAD metabolism. *Structure*, 16, 196–209. https://doi.org/10.1016/j.str.2007.11.017

Lee, J. Y., Li, Z., & Miller, E. S. (2017). Vibrio phage KVP40 encodes a functional NAD+ salvage pathway. *Journal of Bacteriology*, 199, e00855-16. https://doi.org/10.1128/JB.00855-16

Hayes, T., et al. (2025). Simulating 500 million years of evolution with a language model. *Science*. https://doi.org/10.1126/science.ads0018

Wang, X., et al. (2026). Towards A Generative Protein Evolution Machine with DPLM-Evo. *Proceedings of Machine Learning Research*, 306. https://proceedings.mlr.press/v306/wang26by.html

Deutschmann, N., et al. (2026). EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering. https://arxiv.org/abs/2603.11703
