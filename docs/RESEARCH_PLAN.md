# NatV 研究计划

## 研究对象与功能目标

研究对象为噬菌体 KVP40 的 341 aa NMNAT–Nudix 双域 NatV。目标是在相同条件的纯化酶体系中提高 NAD⁺ 净积累，同时保留有应用价值的 NMNAT 合成。NMNAT 催化 `NMN + ATP ⇌ NAD⁺ + PPi`；Nudix 水解相关功能可能抵消净合成，但减少水解并不自动保证净积累提高，也不能假定 Nudix 结构域可安全删除。合成初速、水解与净积累是不同功能终点，只有未来实验能确认改善。

当前科学问题是：**已训练的 Categorical QVAE 能否在 KVP40 WT 邻域生成充分、多样、可追溯的完整 NatV 变体；其中哪些经序列、结构与化学环境评价后值得进入工程候选库？** QVAE 潜变量可能包含 NatV 家族的联合序列规律，改变 z 影响输出已被观察到，但其功能含义及优于其他方法的效果均未证实。不因模型名称或潜变量黑盒性预设优势，也不要求先完成更多模型排名才能生成候选。

## 研究路线与当前进度

| 环节 | 科学职责 | 当前认识 |
|---|---|---|
| ① 研究对象与功能目标 | 定义 KVP40 NatV、工程性质及实测终点 | 已确定研究范围 |
| ② 构建高质量数据 | 建立序列、家族、结构与角色可靠的数据来源 | 正式数据与来源核查基本就绪 |
| ③ 训练与模型 | 学习家族序列规律，形成适用于任务的生成器 | QVAE 可用于条件编辑；既有比较已足够选择当前方法 |
| ④ 生成候选 | 充分探索 WT 保守邻域，保留所有真实生成记录 | **当前重点**；尚缺不强制突变的正式生成 |
| ⑤ 初步计算评价 | 对候选建立序列、结构及催化环境证据 | 已有参考信息，候选自身系统评价待开展 |
| ⑥ 遴选工程化 NatV 候选 | 综合证据与不确定性判断设计价值，可返回⑤ | 待开展 |
| ⑦ 优选候选序列库 | 形成有代表性、互补、可追溯的候选集合 | 待形成 |
| ⑧ 未来酶学验证与反馈 | 测定净积累与组分活性，反馈至数据和生成方法 | 后续研究 |

环节说明研究内容而非逐项放行条件：②的已有结构资源可用于⑤；⑥发现重要疑问时返回⑤，生成能力存在问题时返回③；⑧的结果可修正②—⑦的认识。**③模型研究和④生成候选是不同职责，不以无止境模型比较取代实际序列生成。**

## ④ KVP40 保守候选生成

以 KVP40 WT 完整序列为参考，主要探索 **90%–100% 全局编辑相似性**的序列：

```
I_WT = 1 − Levenshtein_distance(candidate, WT) / max(len(candidate), 341)
```

距离按残基替换、插入及删除计数；约 34 次单残基编辑即接近 90% 下界。该值是明确的生成空间，不是蛋白活性概率或标准全局比对 identity 的替代定义。WT 本身作为零编辑参照，不能算新增候选。正式天然家族训练群体不受此相似性范围过滤。

**生成方式。** 以现有 H2 Categorical QVAE 和可核查的 checkpoint 为主要生成器，结合完整 WT 或当前已编辑序列、变化的掩码位置和先验 z 做多轮条件重采样。给模型足够广泛的位置与编辑程度探索空间，**掩码位置只是允许被更新的位置，不等于必须改变的位置**；允许在抽样时保留或恢复 WT 残基，根据最终序列统计实际编辑位置、替换数量和组合。选掩码本身仍是外部采样规则，不能将结果表述为 QVAE 自主学习了位置决策。现役 `conservative_editing.py` 中强制选中位置全部为非 WT 的旧采样行为，需要在本任务实施时调整；此前的 2/12/24/34 预算为历史比较设置，不是此次生成配额。

采用覆盖适当序列空间的掩码采样、先验采样和有限计算预算，记录**总尝试数**以及不同编辑距离与区域的实际覆盖情况；不通过反复重采把低产或重复掩盖掉，也不要求 90%–100% 每段相似性有相同样本数。候选输出为完整天然氨基酸序列，并保存对应的 z、种子、逐步上下文、WT 锚点、`KEEP/SUB/INSERT/DELETE` 编辑脚本、重复与不满足约束的原因。生成充分性依据研究空间的覆盖与新增组合是否仍有信息价值判断，不靠无限增加数量。

短 InDel 可在明确的来源事件支持下探索单个 1–3 aa 插入或删除，单独统计；旧 G1-b 事件是演化线索而非功能耐受性证据。现有实现是在选定来源事件后进行条件替换，不是已训练的自由 InDel 生成器。InDel 不是首轮必须满足的配额，不将对齐列 gap 误当作经验证的插入/删除生成。

**④不预设 A/B/C 功能归属，也不强制依据结构预选全部编辑位点。** A（NMNAT 环境）、B（Nudix 扰动）、C（联合改变）是⑤—⑥解释和遴选变体时可研究的机制假设，而不是三个必需的生成臂。允许模型探索人工未指定但仍满足序列约束的组合。生成与评价的区别是：模型产生候选；实验结构、家族变异及化学环境证据判断候选是否值得继续研究。

以冻结 ESM-2 条件编辑作为必要的简单参照，并保留 WT；仅在能够回答具体生成问题时做匹配比较。旧 SE、列频、Continuous MMD-VAE 的结果留在原始运行报告，不继续展开四臂排名。已有权重的训练任务与输入资格以 [DATA_PROVENANCE.md](DATA_PROVENANCE.md) 为准。

## ⑤ 初步计算评价

先对实际生成序列核对 WT 编辑、长度、重复、家族相容性与有序双结构域注释，再对有代表性或疑问明确的候选开展自身全长结构预测，与 WT 和实验同源结构比较。计算强度随科学问题增加，避免对所有序列等量开展昂贵折叠、对接或构象采样，也不以一项模型分数直接淘汰所有不一致候选。

已存放实验结构 2QJO、2QJT、2R5W 用于 NMNAT、Nudix 及其配体/金属邻域的同源参照；它们的实体、标签、编号、缺失残基及对应限制见 [NATV_PAIRING_AUDIT.md](NATV_PAIRING_AUDIT.md)。其中 2QJT/2R5W 是同一同源序列的不同状态，并非两个独立家族样本。候选自身的催化环境推断应建立在正确映射及具体残基几何上，不能把不同同源结构的配体状态拼作已经观察到的 KVP40 反应中间态。

证据按 **实测测定 / 已存放实验结构 / 预测结构 / 家族比对 / 模型分数** 分别呈现；不将 pLDDT、HMM、对接或潜变量分数解释为 `kcat`、水解速率或 NAD⁺ 净积累。已知同源结构与生成器共享的演化先验也不能成为独立的模型优越性证明。无法测量或无法可靠对应的属性记 **UNKNOWN**，不填写 PASS。

BioEmu2 可在个别候选存在真实构象不确定性时研究 WT 与变体的构象集合，须先确认单张 32 GB GPU 的实际可行性；采样频率不是已校准的自由能。RFdiffusion3 的原子级条件设计可供方法参考，不是本轮 WT 保守生成的替代方案；FreeBindCraft 面向结合物设计，不属于此研究的必要工具。

## ⑥ 遴选工程化 NatV 候选与⑦优选序列库

综合⑤的结构对应、家族证据、真实编辑与不确定性，优先选择**具有不同可检验问题、序列与机制互补**的候选，包括 NMNAT 相关变化、Nudix 相关变化或联合改变；这些是评价时的解释，不能追认其已经提高酶活。遇到影响选择的关键分歧时，⑥将相应候选送回⑤做针对性结构或化学环境分析；只有确实需要探索新序列时才返回④。

⑦从已评价候选中形成多样且可追溯的序列库，保留 WT 和适当匹配参照、各候选完整序列、来源、编辑、所支持的机制假设及 UNKNOWN 项。不能把未经酶学验证的计算候选写成已成功提高净积累的工程酶。

## ⑧ 未来酶学验证与反馈

在固定的纯化酶体系中，以 WT 为对照分别测定 NMNAT 合成初速、Nudix 对 NAD⁺ 的水解及 NAD⁺ 净积累时间曲线；记录酶量、底物、Mg²⁺、pH、温度、PPi 条件和观察时段。若条件允许，辨识 `kcat` 与表观 `Km`。实验数据将用于检验候选假设、识别计算评价的误判，并决定是否需要修正生成目标和模型训练。

## 既有研究证据

[保守条件编辑比较](../experiments/natv_conservative_editing/REPORT.md)完成 2,976 次给定位点下的条件生成，约束有效性得到核对；高预算时唯一组合比例饱和，不足以区分模型生物学价值。它采用强制非 WT 替换，不代表④所规划的不强制突变生成。[WT 对齐列实验](../experiments/natv_wt_columns/REPORT.md)的 gap/去 gap 全列生成不能直接用于保守 WT 工程候选。原始输入、负结果、实现身份及限制由报告与[模型继承记录](../discovery/natv_model_inheritance/model_reuse_decision.md)提供。

## 文献依据

Huang, N., et al. (2008). Bifunctional NMN Adenylyltransferase/ADP-Ribose Pyrophosphatase: Structure and Function in Bacterial NAD Metabolism. *Structure*, 16, 196–209. https://doi.org/10.1016/j.str.2007.11.017

Lee, J. Y., Li, Z., & Miller, E. S. (2017). Vibrio Phage KVP40 Encodes a Functional NAD⁺ Salvage Pathway. *Journal of Bacteriology*, 199, e00855-16. https://doi.org/10.1128/JB.00855-16

Hayes, T., et al. (2025). Simulating 500 million years of evolution with a language model. *Science*. https://doi.org/10.1126/science.ads0018

Wang, X., et al. (2026). Towards A Generative Protein Evolution Machine with DPLM-Evo. *Proceedings of Machine Learning Research*, 306. https://proceedings.mlr.press/v306/wang26by.html

Deutschmann, N., et al. (2026). EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering. https://arxiv.org/abs/2603.11703

Butcher, J., et al. (2026). Atomic functional site scaffolding in biopolymers with RFdiffusion3. *Science*. https://doi.org/10.1126/science.aeh2982

Microsoft Research (2026). BioEmu2 model card. https://github.com/microsoft/bioemu2/blob/main/MODEL_CARD.md
