# NatV 研究计划

## 研究对象与功能目标

研究对象为噬菌体 KVP40 的 341 aa NMNAT–Nudix 双域 NatV。目标是在相同条件的纯化酶体系中提高 NAD⁺ 净积累，同时保留有应用价值的 NMNAT 合成。NMNAT 催化 `NMN + ATP ⇌ NAD⁺ + PPi`；Nudix 水解相关功能可能抵消净合成，但减少水解并不自动保证净积累提高，也不能假定 Nudix 结构域可安全删除。合成初速、水解与净积累是不同功能终点，只有未来实验能确认改善。

当前科学问题是：**下一轮生成模型能否在 KVP40 WT 邻域生成充分、多样、可追溯的完整 NatV 变体；其中哪些经序列、结构与化学环境评价后值得进入工程候选库？** 潜变量是否编码家族联合规律仍是待验证假设。

## 研究路线与当前进度

| 环节 | 科学职责 | 当前认识 |
|---|---|---|
| ① 研究对象与功能目标 | 定义 KVP40 NatV、工程性质及实测终点 | 已确定研究范围 |
| ② 构建高质量数据 | 建立序列、家族、结构与角色可靠的数据来源 | 正式数据与来源核查基本就绪 |
| ③ 训练与模型 | 学习家族序列规律，形成适用于任务的生成器 | 已完成条件 Categorical QVAE；联合建模及候选供给优势证据不足 |
| ④ 生成候选 | 充分探索 WT 保守邻域，保留所有真实生成记录 | 代表序列与覆盖摘要见本轮报告 |
| ⑤ 初步计算评价 | 对候选建立序列、结构及催化环境证据 | 已提出具有具体结构问题的代表序列 |
| ⑥ 遴选工程化 NatV 候选 | 综合证据与不确定性判断设计价值，可返回⑤ | 待开展 |
| ⑦ 优选候选序列库 | 形成有代表性、互补、可追溯的候选集合 | 待形成 |
| ⑧ 未来酶学验证与反馈 | 测定净积累与组分活性，反馈至数据和生成方法 | 后续研究 |

环节说明研究内容而非逐项放行条件：②的已有结构资源可用于⑤；⑥发现重要疑问时返回⑤，生成能力存在问题时返回③；⑧的结果可修正②—⑦的认识。**③模型研究和④生成候选是不同职责，不以无止境模型比较取代实际序列生成。**

## ④ KVP40 保守候选生成

以 KVP40 WT 完整序列为参考，主要探索90%–100%相似性的保守邻域。当前定长替换研究以 **Hamming 距离/真实替换数**定义编辑强度：`I_WT = 1 − Hamming(candidate, WT)/341`；Levenshtein只作历史兼容统计。未来若研究InDel，应另行定义长度变化与距离口径。这些距离是设计空间约定，不是活性概率或标准比对identity。WT是零编辑参照，不能算新增候选；天然家族训练群体不受WT相似性范围过滤。

**生成方式。** 当前采用学习型条件先验的一次潜变量采样、多位点单轮联合生成。条件先验与解码器只读取可见上下文及掩码，完整天然目标仅供训练后验推断。共享z在给定条件下耦合多个目标位置；不能仅凭这一结构认定模型学到了正确的天然依赖。正式采样与训练掩码条件应有覆盖关系，具体规模和资源范围见本轮方案。

掩码定义允许更新的位置，位置不由模型学习；允许抽到WT残基，真实替换数由最终序列决定。使用共同条件与匹配采样预算，保留失败、重复和WT返回，不通过补采隐藏低产，不要求各距离区间等量。输出保留完整序列、条件与掩码、z、随机种子、权重身份和真实编辑；覆盖与新增组合的信息价值决定供给是否充分。

短InDel需另行定义生成任务与来源支持；旧G1-b事件只能提供演化线索，不能证明功能耐受性，也不能将对齐gap当作已验证的插入/删除生成机制。

**④不预设 A/B/C 功能归属，也不强制依据结构预选全部编辑位点。** A（NMNAT 环境）、B（Nudix 扰动）、C（联合改变）是⑤—⑥解释和遴选变体时可研究的机制假设，而不是三个必需的生成臂。允许模型探索人工未指定但仍满足序列约束的组合。生成与评价的区别是：模型产生候选；实验结构、家族变异及化学环境证据判断候选是否值得继续研究。

当前研究用相同冻结PLM、训练群体、掩码与采样预算独立训练无潜变量对照，并以QVAE自身边缘独立分布隔离共享z的联合效应。原始供给与匹配编辑强度后的比较并列；不能由匹配分析掩盖生成效率差异。已有权重的训练任务与输入资格以 [DATA_PROVENANCE.md](DATA_PROVENANCE.md) 为准。

模型选择服务于具体生成问题；不同输入信息或任务的方法不能用总分直接比较。评价覆盖多位点组合、序列多样性和设计空间；先验改变输出、恢复 CE 较低或新颖率饱和都不单独构成生成优势。新一轮训练与生成需先绑定任务、评价和预算。

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

上一轮条件生成方法检查提供了覆盖摘要与代表序列，见[历史报告](../experiments/kvp40_conditional_generation/REPORT.md)。目标条件后验重训的六次拟合均在第 22 轮早停并保留第 12 轮权重；改变 z 有非零输出影响，但先验验证 CE 未优于确定性对照，证据见[训练报告](../experiments/kvp40_target_posterior_training/REPORT.md)。条件Categorical QVAE已采用完整序列后验、学习型条件先验及REINFORCE-LOO完成训练、联合评价和候选生成，见[本轮报告](../experiments/natv_conditional_qvae/REPORT.md)。开发验证联合NLL略有改善，但预选成对/三位点共变统计及候选供给优势证据不足；不能将小幅概率改善或生成数量升级为工程收益。48条代表序列已按Nudix探索、匹配参照及家族组合问题遴选，结构与酶学性质仍待后续研究。

四轮已退役生成实验的证据、负结果、限制与提交 SHA 见[退役生成实验证据](../discovery/natv_generation_evidence_decision.md)；模型来源与身份见 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)。这些结果不支持把 H2 首轮候选作为下一阶段必须资产。

2026-10-22 完成 NMNAT 演化设计证据分析：从审计后的 556/44、63/19、318/22 与 reviewed positives 61/7 中得到 7 条待检验设计假设；现有候选池只含多数单点替换、不含 F13Y+A134G 配对，重排结果只作描述性富集，没有 held-out 证据，不能作为独立验证，也未据此启动新的生成或训练。这些集合是设计证据来源，不是训练人口；详细结果与限制见[实验报告](../experiments/natv_evolution_informed_nmnat/REPORT.md)。

## 文献依据

Huang, N., et al. (2008). Bifunctional NMN Adenylyltransferase/ADP-Ribose Pyrophosphatase: Structure and Function in Bacterial NAD Metabolism. *Structure*, 16, 196–209. https://doi.org/10.1016/j.str.2007.11.017

Lee, J. Y., Li, Z., & Miller, E. S. (2017). Vibrio Phage KVP40 Encodes a Functional NAD⁺ Salvage Pathway. *Journal of Bacteriology*, 199, e00855-16. https://doi.org/10.1128/JB.00855-16

Hayes, T., et al. (2025). Simulating 500 million years of evolution with a language model. *Science*. https://doi.org/10.1126/science.ads0018

Wang, X., et al. (2026). Towards A Generative Protein Evolution Machine with DPLM-Evo. *Proceedings of Machine Learning Research*, 306. https://proceedings.mlr.press/v306/wang26by.html

Deutschmann, N., et al. (2026). EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering. https://arxiv.org/abs/2603.11703

Butcher, J., et al. (2026). Atomic functional site scaffolding in biopolymers with RFdiffusion3. *Science*. https://doi.org/10.1126/science.aeh2982

Microsoft Research (2026). BioEmu2 model card. https://github.com/microsoft/bioemu2/blob/main/MODEL_CARD.md
