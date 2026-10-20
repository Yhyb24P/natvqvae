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

A、B、C 是同一生成器下的设计假设：**A** 保持 NMNAT 催化相关组织并探索邻域调整；**B** 在保留 NMNAT 与双域组织的前提下探索 Nudix 水解相关扰动；**C** 联合组合二者。首轮以 **B 为重点**，A、C 用于提供不同的结构及机制问题，不要求等量采样。关键位点的直接改变与邻域改变分别记录；天然保守性不是普适禁令。`contracts/regions.json` 的固定长度掩码仅约束其历史试验，不是当前变长设计规则。

## 当前研究：演化与结构指导的保守 QVAE 生成

六环节为①数据、②生成模型研究、③工程化序列生成、④计算评价、⑤候选库、⑥未来酶学验证。①已有正式群体与来源校验；②的全列生成及保守编辑比较已足够支持方法选择，未显示 QVAE 潜变量优越性，也没有充分区分模型生物学价值。当前正式进入③的第一轮有目的生成。③选位可以使用已有实验结构与家族分析；④仍负责生成**候选自身**的全长结构与催化环境评价，两者不是严格串行的证据收集过程。

**科学假设。** Categorical QVAE 的离散潜变量可能包含 NatV 家族的联合序列规律，可在 KVP40 WT 条件下支持不同的保守多位点组合。这是待检验的生成假设，不等于已经学习到合成、水解或净积累的功能方向。选择 QVAE 不要求先证明它优于 SE；研究重点是能否实际产生有来源、可解释、值得进一步检验的变体，而非继续进行四种生成器排名。

**先提出位置假设。** 使用正式训练群体的家族变异、同源组分和稳定比对，结合 [NATV_PAIRING_AUDIT.md](NATV_PAIRING_AUDIT.md) 中的 WT—实验构建体映射确定候选区域。以 2QJO 的 NAD⁺ 产物结合结构作为 NMNAT 参考，以 2QJT 的 AMP/Mn 状态及 2R5W 的 Mg/Cl 状态研究 Nudix 邻域；两种 Nudix 状态来自同一同源序列，不能算两份独立支持。优先研究明确映射的关键位点邻域、可能影响金属/底物环境的周边位置及具有天然变异支持的区域；先区分直接催化位点、结构支持位点和域间组织位点，再判断是否适合编辑。H236 在所比较同源蛋白中存在残基差异；配体邻近也只提供结构假设，不能充当水解速率标签。位点选择记录 WT 坐标、结构对应、家族支持与所要检验的机制；没有可信对应时不补写确定的几何结论。

**再进行条件序列生成。** 使用现有 H2 Categorical QVAE 及已核对 checkpoint，以 WT 或已编辑完整序列为条件，在选定区域内迭代掩码生成，记录先验采样的 z、随机种子、上下文更新、实际替换和完整编辑脚本。区分“允许被编辑的位置”和“实际发生替换的位置”，不因位置入选就强制全部突变；替换数量由科学假设与最终 WT 相似性共同约束，不沿用 2/12/24/34 的横向排名设置。保留不同 z 产生的实质性序列组合，但不把 logits 响应视作功能协同。旧对齐列的 gap 不能作为 InDel；仅在具备可信天然事件和局部结构问题时提出 1–3 aa 短 InDel，单独描述事件与邻域补偿，不强制纳入首轮样本。

**合理参照与评价。** 以 WT、冻结 ESM-2 在相同编辑问题下的提案以及必要的简单匹配替换作为解释参照；不重复训练 SE、列频、MMD-VAE 或预设三模型排名。生成过程保留全部尝试及失败，按真实编辑距离、区域、模型条件和组合差异描述提案。结构域 HMM、新颖率、ESM-2 或家族分数仅提供相容性信息，在近 WT 任务中可能饱和且与选位信息相关。重要的是候选能否对应明确的 A/B/C 机制问题，是否存在不同的联合编辑方案，以及后续结构分析是否揭示值得研究的差异。

**首轮研究成果。** 应形成有明确位点依据和变异机制假设的 QVAE 完整序列及其来源记录，再选择代表不同生物学问题的候选进入④；包括 WT、合适的参照与保留的不确定性。候选的序列差异和结构假设构成计算研究结果，不能直接判定 NMNAT 活性保留或 Nudix 水解下降。新模型或更多样本比较仅在可能改变实际设计判断时开展。


## 计算评价和候选库

针对③实际产生并选择的完整候选进行**候选自身的全长结构**分析，比较 WT 与不同设计假设下的 NMNAT 环境、Nudix 配体/金属邻域、域间组织及可能的 InDel 局部影响。保留实验结构、家族比对、预测结构和模型分数的不同证据层级；预测的侧链接触变化或折叠置信度不能直接换算成催化效率。无法获得可靠结构或原子对应时记 UNKNOWN，单个预测指标通常只能调整候选的研究优先级。

④首先使用能在现有资源上可信运行的全长结构预测与实验模板对应。若出现静态结构确实无法回答的局部柔性或域间构象问题，再考虑 BioEmu2 的候选—WT 配对构象分析，并确认实际显存可行性；其未校准的构象采样频率不等于热力学自由能或酶活。RFdiffusion3 的原子级条件设计可提供构建功能环境的思路，但不作为本轮 WT 保守序列生成的替代方法，FreeBindCraft 的结合物设计也不对应当前任务。

⑤从结构与序列证据中选择机制互补、疑问明确的代表性候选，保留 WT 和匹配参照。优选是为了形成后续实验可检验的假设，不根据单一模型分数声称工程改进。


## 已有结果及其适用范围

[保守条件编辑运行](../experiments/natv_conservative_editing/REPORT.md)生成 2,976 条符合编辑约束的提案。四种方法的双域 HMM 命中接近，高预算下共同位点内不同组合比例饱和；这些比较不能证明 QVAE 的生物学优劣。位点由随机规则预设，尚非有设计目的的工程候选生成；z 与上下文响应只证明参与计算。

[WT 对齐列先验生成](../experiments/natv_wt_columns/REPORT.md)的全列独立采样产生大幅 WT 偏离及 gap，不适合作为当前保守设计的现成方案。[历史局部编辑](../experiments/natv_candidate_sequences/REPORT.md)只覆盖较少位点的条件提案。方法身份、划分差异与复用界限见 [模型继承记录](../discovery/natv_model_inheritance/model_reuse_decision.md)；历史负结果仅限各自运行任务，不构成对 QVAE 在其他生成任务中无效的证明。

## 未来酶学验证

在固定纯化体系测定 WT 与候选的 NMNAT 初始合成速度、Nudix 对 NAD⁺ 的水解，以及 NAD⁺ 净积累时间曲线；条件允许时分析 `kcat` 与表观 `Km`。WT 是必要参照。具结构完整性的 Nudix 失活变体及 NMNAT 失活变体属于待验证机制对照。只有实测净积累提高且保留可用 NMNAT 合成，才能成立工程改进结论。

## 文献依据

Huang, N., et al. (2008). Bifunctional NMN adenylyltransferase/ADP-ribose pyrophosphatase: Structure and function in bacterial NAD metabolism. *Structure*, 16, 196–209. https://doi.org/10.1016/j.str.2007.11.017

Lee, J. Y., Li, Z., & Miller, E. S. (2017). Vibrio phage KVP40 encodes a functional NAD+ salvage pathway. *Journal of Bacteriology*, 199, e00855-16. https://doi.org/10.1128/JB.00855-16

Hayes, T., et al. (2025). Simulating 500 million years of evolution with a language model. *Science*. https://doi.org/10.1126/science.ads0018

Wang, X., et al. (2026). Towards A Generative Protein Evolution Machine with DPLM-Evo. *Proceedings of Machine Learning Research*, 306. https://proceedings.mlr.press/v306/wang26by.html

Deutschmann, N., et al. (2026). EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering. https://arxiv.org/abs/2603.11703

Butcher, J., et al. (2026). Atomic functional site scaffolding in biopolymers with RFdiffusion3. *Science*. https://doi.org/10.1126/science.aeh2982

Microsoft Research (2026). BioEmu2 model card. https://github.com/microsoft/bioemu2/blob/main/MODEL_CARD.md
