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

## 生成模型研究与公平比较

研究主线保持：① 高质量数据 → ② 训练与比较生成模型 → ③ 工程化生成 → ④ 计算评价 → ⑤ 候选库 → ⑥ 未来酶学验证。当前重点是将 **② 和 ③ 连成一项真实生成研究**，而不是继续扩充序列数据库或围绕历史候选增加评分。

优先继承旧仓库 `design/g0_benchmark.py`、`formal_generation.py` 中的冻结 ESM-2/SE 迭代掩码更新和 train-only 演化检索；复用 `vnext_edits.py` 的显式编辑脚本。生成器需实际消费当前变体的残基上下文，而非对 WT 对齐列独立取样后删除 gap。G1-a 选位结论仅支持策略差异，不提供哪种变体更可能提高酶活的排名。

**比较对象**为同一保守编辑任务上的 ESM-2 或 SE 条件生成、简单匹配预算编辑，以及能够明确使用先验潜变量的 conditional QVAE。先核实旧 H2 权重能否支持新的位置和编辑预算；若训练目标不匹配，再针对任务训练。旧 `SequenceGenerator` 的全列先验解码不适合作为现役保守任务的现成模型，但其参数与训练经验可用于方法研究。Continuous MMD-VAE 仅在需要区分连续与离散潜变量的解释时参与，不默认再次训练所有架构。

应保持同一训练与检索群体、同一 WT、编辑距离分层、合法编辑区域、采样尝试与评价信息预算；分别记录模型容量、实际调用和耗时。潜变量干预可以检验输出是否依赖 z，但不能替代对独立拟合基线的比较。对于长度变化，替换臂与 InDel 臂先分别比较任务完成能力，再判断联合编辑价值，不能把自由变长带来的额外搜索空间冒充模型增益。

**生成能力的判断**主要依据：给定相同编辑预算，能产生多少不同的有效组合；在保留完整 WT 骨架时能否利用已有编辑的上下文；与简单编辑及 SE 比较能否扩展合理的联合序列分布；不同设计假设是否有机制上可描述的提案。报告全部尝试、重复、无效和拒绝原因。家族 HMM 有序双域、ESM-2 交叉熵和天然列频是**相容性证据**，若在近 WT 范围饱和，不能成为模型优胜指标；采用相同来源的选位与评分时，也不能将其称为独立判据。

训练角色及相应的同源隔离、来源身份以 [DATA_PROVENANCE.md](DATA_PROVENANCE.md) 为准。任何新训练必须确认演化检索仅用训练成员，验证集不反向影响选位规则；历史已暴露验证数据的推断只适用于开发比较。固定的训练与生成预算、真实先验使用方式、停训规则及模型选择指标应在观察结果前确定。

## 计算评价和候选库

正式生成结束后，以该次生成的真实候选为对象，按以下证据层次分析：**序列与家族比对 → 候选自身全长预测结构 → NMNAT/Nudix 催化相关几何和已存放实验结构参照 → 代表性候选库**。实验结构和残基对应以 [NATV_PAIRING_AUDIT.md](NATV_PAIRING_AUDIT.md) 为入口。计算阶段不能把模型分数、pLDDT、对接或几何距离写成预测 `kcat` 或净产量。

结构评价要区分已观测结构、预测结构、家族比对及相同模板来源引起的相关证据。特别检查 NMNAT 催化相关位点、Nudix 可能的水解扰动，以及不同长度变体的局部几何与双域相对组织；未测量记 UNKNOWN，不把非保守 Nudix 替换自动认定有害或有益。

候选库应包含 WT、匹配预算的简单编辑对照，以及覆盖 A/B/C 不同机制假设的代表性序列。模型选择和工程候选选择是两个不同问题；生成方法没有优于 SE 的证据，仍可产生值得研究的候选。

## 已有结果及其适用范围

[WT 对齐列先验生成比较](../experiments/natv_wt_columns/REPORT.md) 对正式训练/验证数据完成 6 次拟合、6,144 次先验采样。QVAE 相对等参数的确定性对照未显示清晰优势；每种方法约 39% 的 HMM 有序双域命中、较短的去 gap 序列以及大幅 WT 偏离，证明该实验的输出不能直接当作保守工程候选。保留其完整运行产物，不调整旧评价口径。

[历史局部编辑](../experiments/natv_candidate_sequences/REPORT.md) 已产生 32 条候选（27 条不同序列），只覆盖旧 4/8 位点探索；[H2 历史结果](../discovery/natv_model_inheritance/model_reuse_decision.md) 的负结果同样限于相应掩码编辑。旧 G0/SE 的生成收益未获得决定性支持；G1-a 显示选位评分容易与评价指标共享信息。G1-b 的连接区双峰由注释偏移导致，InDel 事件仍只提供候选假设。历史编号用于证据追溯，不代表必须逐阶段重做。

## 未来酶学验证

在固定纯化体系测定 WT 与候选的 NMNAT 初始合成速度、Nudix 对 NAD⁺ 的水解，以及 NAD⁺ 净积累时间曲线；条件允许时分析 `kcat` 与表观 `Km`。WT 是必要参照。具结构完整性的 Nudix 失活变体及 NMNAT 失活变体属于待验证机制对照。只有实测净积累提高且保留可用 NMNAT 合成，才能成立工程改进结论。

## 文献依据

Huang, N., et al. (2008). Bifunctional NMN adenylyltransferase/ADP-ribose pyrophosphatase: Structure and function in bacterial NAD metabolism. *Structure*, 16, 196–209. https://doi.org/10.1016/j.str.2007.11.017

Lee, J. Y., Li, Z., & Miller, E. S. (2017). Vibrio phage KVP40 encodes a functional NAD+ salvage pathway. *Journal of Bacteriology*, 199, e00855-16. https://doi.org/10.1128/JB.00855-16

Hayes, T., et al. (2025). Simulating 500 million years of evolution with a language model. *Science*. https://doi.org/10.1126/science.ads0018

Wang, X., et al. (2026). Towards A Generative Protein Evolution Machine with DPLM-Evo. *Proceedings of Machine Learning Research*, 306. https://proceedings.mlr.press/v306/wang26by.html

Deutschmann, N., et al. (2026). EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering. https://arxiv.org/abs/2603.11703
