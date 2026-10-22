# NatV 演化信息利用数据审计

状态：CPU 审计已执行（2026-10-22）；未训练、未下载、未修改旧 `nad_qvae`。执行时曾全量载入 Stage3 并切片，含 test/sealed；范围更正见 [AUDIT.md](AUDIT.md) 第 1 节。test 未进入 MMseqs、训练、模型选择或生成。

## 执行状态与解释修正

- 结果与限制见 [AUDIT.md](AUDIT.md)。本文件保留审计协议；执行后的固定 WT 靶标和全长上下文搜索是探索性补充，不属于原预注册的 30%/qcov0.8 结果。
- 本文原“未读取现役测试序列内容”不再成立；实际接触范围、未使用环节和 test 封存状态见 `scope_correction.json`。
- `profile coverage>=0.8 + 切片长度>=124` 只是 PF01467 profile 核心的探索性筛选；PF01467 HMM LENG=133，NatV WT CDD 域 178 aa，不能把它称为完整 NatV CDD 域或生物学完整域。
- `nearest` 是当前命中中按 identity/query coverage/target coverage 排序的最佳局部命中，不是穷举最近亲缘；无命中只表示本次阈值/搜索设置下无报告。
- 下一步不是启动预训练，而是对 556 条/44 簇 EC 注释候选和全长复核的 63 条/19 簇做来源功能证据与完整域边界审阅。

## 目的

回答三个问题，再决定是否改变训练人口或增加域级预训练：

1. 旧 Stage 1–3 的 13,972 条中，除当前 NatV 2,430 条外，真正非冗余、功能相关、域完整且与 NatV NMNAT 对应可靠的潜在域级语料有多少？
2. 是否存在值得进一步设计训练对照的 NMNAT 域级信息？（“是否改善”必须由未来独立实验回答。）
3. 即使域内建模改善，能否转移到完整 NMNAT–Nudix 双域生成？这需要独立的训练/评价实验，不能由序列数量或同源距离直接推断。

第一问是数据审计；第二、三问需要独立的模型实验，不能由序列数量直接推断。审计的核心成果是识别能改善 NatV NMNAT 域设计的**信息来源**，而不是得到一个更大的 FASTA。

## 已核对的组成与重叠

旧 Stage 3 接受序列 13,972 条（去重人口 13,976，拒绝 4）：

| 架构 | 数量 | 与当前 NatV roles 的关系 |
|---|---:|---|
| PF01467 单独命中记录 | 8,849 | 全部为 stage3-only，非现成双域样本 |
| 两 profile 均未命中 | 2,703 | 全部为 stage3-only；不等于无目标信息 |
| 双 profile 命中 | 2,397 | 2,391 有序；6 条重叠/逆序为 stage3-only |
| PF00293 单独命中记录 | 23 | 21 条 stage3-only；2 条 validation |

注：PF01467 单独命中只表示在两 profile 规则下未命中 PF00293，不表示整条蛋白只有一个结构域；其它结构域未全面检查。两 profile 均未命中也不证明缺少相应功能，只是未命中这两个指定 profile。

精确序列交集：stage3 ∩ 当前 roles(2,430) = 2,393；stage3-only = 11,579；current-only = 37。

2,391 条有序双域全部已在当前 roles：

| 当前角色 | 有序双域 |
|---|---:|
| training | 1,644 |
| validation | 198（另有 27 条 current-only、2 条 Nudix-only，合计 227） |
| test | 225（另有 6 条 current-only，合计 231） |
| annotation_conflict_quarantine | 249 |
| quarantined_old_sealed_component | 71（另有 4 条 current-only，合计 75） |
| excluded_noncanonical_or_domain | 4 |

结论：旧语料的双域部分几乎就是当前 NatV 群体，**不能作为新增双域训练数据**；13,972 与 2,430 不能相加。

旧 strict split 的 `NOT_TRAINING_READY` 仍成立：training 中 ordered dual 为 0，test 无 required-profile 命中，F-train 架构臂为空。它是架构审计，不是可用的 NatV 训练/评价划分。

## 当前 NatV 角色

- training 1,644，47 个同源组分。
- validation 227，44 个组分。
- test 231，44 个组分。
- 三组组分互斥；training/validation/test 的分量不重叠。
- 249 annotation conflict、75 旧封存、4 排除继续隔离，不因增加样本解封。

training 1,644 条中 47 个同源组分；前五个组分合计 1,491/1,644 = 90.7%（`data/processed/samples.csv` 的 training 角色范围）。条数不等于独立演化信息；不引用未注明角色范围和 identity/coverage 口径的内部最近邻统计。这个不均衡比“是否把 227 条验证加入训练”更值得优先关注；但新增数据能否改善共享潜变量的弱利用，仍取决于新增的演化约束，而不取决于训练条数。

## 8,849 条 PF01467 单独命中记录的初步证据

旧 `corpus_boundary_manifest.jsonl` 覆盖全部 8,849 条，`domain_profile_manifest_v4.csv` 提供逐条 profile 分数/覆盖/边界。SHA：

- stage3 FASTA `2277e217904f8ecba748527e673c0f12ce3ff49dfc8ab9528769faac49a46e29`
- domtblout `0ea9e34a0954f13767e30d39259851d885ad689fbbb6c41e079f74b2db116f5a`
- HMM `5b2e42d6884aeb8d35e209e2fe762755452955cc376037e9039f2a06cc466d15`
- corpus manifest `b4c05d8e8e2d77a13a4c70e5e3d66244e7df9666b82d7d3d102425191bbcdc33`
- domain profile manifest `b8961fb95a3c51de887ab433bbc86c66195a73ea3701b50439d6139a98ce4e80`

初步标签（可重叠，不能相加）：

- Stage1 ENRICHED 991、RESCUED 19、UNMAPPED 7,839；UNMAPPED 只是旧映射未成功，不是无价值负例。
- NadR 多功能文本 3,026；partial 228；旧规则 annotation conflict 735；header disagreement 244。
- UniProt EC：missing 8,054；仅 2.7.7.1 302；两 EC 并存 491；仅 2.7.7.18 1；其它 1。
- PF01467 profile coverage 中位数 0.985，但 3,127 条 <0.7；49 条多命中。
- 8,826 条 header 有方括号分类名称；1,023 条同一精确序列对应多个名称，共 5,167 个名称。该层只能作 header 初分，不是权威分类学或独立物种计数。

这些数字说明：**8,849 条 PF01467 单独命中记录不是 8,849 条 NMNAT**，也不是一个干净、完整、已证明可迁移的域语料。PF01467 命中只是广泛 CTP_transferase-like 架构证据；功能、分类学与域完整度仍待分层。

## 功能证据分层

功能类别是独立于“PF01467 是否单独命中”的维度，各类别可重叠，不能相加：

| 功能层 | 可接受证据 | 审计用途 |
|---|---|---|
| 直接实验或可靠人工审校的 NMNAT | 现有来源中的实验/curated 证据、明确 EC 2.7.7.1 | 优先评估迁移价值 |
| NMNAT + NaMNAT 双活性 | 两 EC 或可靠的 dual 注释 | 单独分析底物差异，不因双 EC 当异常 |
| 主要 NaMNAT | EC 2.7.7.18 或对应的可靠注释 | 与 NMNAT 分开，不能混成同一先验 |
| NadR 等多功能蛋白 | 结构域级证据 | 只取 NMNAT 相关域；不整类接纳，也不整类排除 |
| 仅 PF01467/文本/UNMAPPED | 无直接功能证据 | 探索性/UNKNOWN，不作训练资格 |

- 审计只使用现有 manifest、UniProt/EC、gene 和 header 字段；不在本阶段远程检索文献或 taxonomy。没有实验/curated 证据时写 UNKNOWN，不能把文本名升级为实验功能。
- 两 EC 并存不是异常；EC 2.7.7.1 与 2.7.7.18 的反应和底物差异必须分别记录。
- NadR 等多功能蛋白不能因一个总注释被整类接纳或排除；只有可核验的域边界和域级证据才进入 NMNAT 相关域层。
- 旧 `nmnat_specificity_reference_v1` 的分类器未验证，不能作为功能标签。

### 数据资格

只有同时满足以下两项的域才进入域级预训练候选：

1. **域完整度**：HMM profile/query coverage、实际域长度、起止位置、缺失区段和边界对应可核验；多命中记录全部 hit，并说明主域选择规则，不能默认 best local hit 就是完整 NMNAT 域。
2. **功能证据**：属于上表中有可靠证据的 NMNAT/NadR 域层；仅 PF01467 或文本名称不足以进入训练资格。

不满足者保留为 exploratory/UNKNOWN，不能因数量或低 E-value 进入训练。

## MSA 与历史上下文

旧 `natv_q1a_information_decomposition_v1/paired_summary.json` 的 20 个历史 M1 query 显示（单一 homology component，historical_only=true，F_train_compliant=false，inferential_CI_authorized=false）：

- architecture arm（arch-full）：`Neff_full` 中位数 223.8773，`Neff_contract` 中位数 92.1379；
- all-full：`Neff_full` 中位数 228.2801，`Neff_contract` 中位数 92.2618；
- 95.0656 是 all-N matched 的 full/contract 值，不能与 all-full 的 228.2801 拼成一行；
- 这些是单组分、逐 query 的描述性诊断，不是新增 F-train 样本，不能把历史 M1 行并入当前训练库，也不能由这些中位数做推广或新增样本推断；
- 当前 QVAE 依赖冻结 ESM-2 表示，没有直接使用这些 MSA 联合统计。是否建立独立、版本化的 MSA 资源，应在数据审计之后单独决定。

## 审计协议（CPU）

### A. 逐序列/逐域证据表

- 复核 13,972 条身份、长度、架构、stage3 接受状态；单列 6 条 unordered/overlap dual。
- 从 `domain_profile_manifest_v4.csv` 提取 PF01467 best alignment；记录 profile/query coverage、ali_from/to、score，不把 partial 域当完整域。
- `domain_profile_manifest_v4.csv` 不含 E-value；E-value 需从 `domtblout` 按 sequence ID/profile/对应 domain 坐标关联，并记录匹配规则与来源，不在 CSV 中假设该字段存在。
- 解析 `corpus_boundary_manifest.jsonl` 的 accession、gene、EC、stage1 状态、header 名称与 evidence；不做远程检索。
- 当前 2,430 条只用角色 ID/既有元数据计数；域提取限当前可见的 training+validation（1,871）与经 ID 白名单选中的旧独有记录；不给现役 test、旧封存或隔离记录构建新序列输入。
- 输出：`stage3_sequence_audit.csv`、`evidence_summary.json`。

### B. 域完整度与冗余

- PF01467 单独命中的域：按 profile alignment 边界提取；按 coverage ≥0.8、实际域长度、边界合理、缺失区段、非 fragment 分层；coverage 阈值只是操作性完整度分层，不是功能正确判定。
- 多命中：记录全部 profile hits、坐标和覆盖；主域选择必须有明确规则；不得默认 best local hit 代表完整 NMNAT 域。
- 使用本机 `mmseqs 18.8cc5c` 对域序列做同一性/覆盖率聚类：`--cov-mode 0` 为 query+target 双向覆盖，`--cluster-mode 0` 是 greedy set-cover、`1` 是 connected components；两种簇结构都报告，不只报一个。70/50/30% 是三个固定相似性阈值，不是“距离层”；报告簇数及阈值敏感性，每个簇不等于独立演化样本或生物家族。
- 除簇数外，报告每个候选域到当前 NatV NMNAT 域的最近邻、保守催化位置对应、域完整度和功能类型分布。高冗余功能群中的大量序列可能不如少量、功能可靠且催化环境互补的域；与 NatV 相似性较低但同反应的 NMNAT 域也不能仅因不在高 identity 簇中判为不可迁移。
- 比较当前可见合格 1,871 条（training 1,644 + validation 227）与旧独有域记录的新增簇、已有簇扩张、仅同组分重复；当前 2,430 只用于角色/元数据计数；不把 test、旧封存或隔离序列构造成新输入。
- 本审计不读现役 test，因此不宣称未来域级预训练已与 test 完成同源隔离；该隔离须在未来具体训练协议中定义。
- 输出：`pf01467_domain_clusters.csv`、`redundancy_summary.json`。

### C. 功能与分类证据

- 按“功能证据分层”表输出：直接实验/可靠审校 NMNAT、NMNAT+NaMNAT 双活性、主要 NaMNAT、NadR 多功能域、仅 PF01467/文本/UNMAPPED；类别可重叠，不能相加。
- 两 EC 并存不等于错误或负例；PF01467 命中不等于 NMNAT 功能；NadE/NADSYN 或其它反应不得因“NAD 相关”混入；NadR 不整类接纳或排除，按域级证据处理。
- header 方括号名称仅作分类学初分；同一精确序列多个名称要保留冲突，不强行合并成“物种样本”。
- 若需要权威 taxonomy/family 或实验功能证据，必须另立来源与校验，不在本审计中远程下载。

### D. 与 NatV NMNAT 域的对应

- 用当前可见的 training/validation NMNAT 域（CDD 边界）与 PF01467 单独命中的域做序列相似性、覆盖和 HMM 分数比较；报告 70/50/30% 固定相似性阈值下的对应情况。
- 旧 PF01467 alignment 边界与当前 NatV CDD 边界必须通过实际序列比对建立映射；记录 identity、覆盖、缺口、插入/缺失和 NMNAT 锚点残基对应，不能把两套注释坐标直接视为同等输入。
- 同时记录 HMM profile/query coverage、实际域长度、起止位置和缺失区段；域完整性不通过或功能证据不足者标 exploratory/UNKNOWN，不进入域预训练资格。
- 结构模板兼容仅使用现有 2QJO/2QJT/2R5W 与锚点映射；不做折叠、不把结构相似性当功能。
- 这一阶段只回答“是否存在非冗余、对应可靠、功能证据可分的潜在域级语料”；迁移是否有效必须由未来训练对照判断，不能凭同源距离或结构对应裁定。不训练、不拼接双域、不生成候选。

### E. 审计结论

| 审计发现 | 科学判断 | 后续方向 |
|---|---|---|
| 主要是重复序列、片段或功能无关蛋白 | 没有发现有价值的新增域级信息 | 不扩充语料，报告负结果 |
| 存在完整且功能相关的 NMNAT 域，但与 NatV 差异明显 | 具有跨蛋白背景迁移的研究价值 | 比较域级预训练与现有 ESM-2 表示，保留确定性/无潜变量对照 |
| 存在与 NatV 催化区域对应可靠、具有互补变化的 NMNAT 域 | 可能直接改善 NatV 生成所需的演化先验 | 研究域级演化条件如何进入全长 QVAE |

最后一种最值得期待，但不能预设一定出现；第一种同样是有效结果。审计的核心成果是识别可改善 NatV NMNAT 域设计的信息来源，而不是筛出更大的 FASTA。

## 条件性后续（本草案不执行）

### 1. 数据角色实验

1,644 vs 1,871 是与 8,849 条审计独立的数据角色问题；若决定进行，必须：

- 227 并入 1,871 后，不再作为该模型的验证/早停集；
- 开启 231 test 前冻结两 arm 的配置、checkpoint 规则、主要指标、数据曝光/步数与预算；
- 两 arm 用训练内部组分划分或固定步数决定优化，不用 231 调参；
- 若使用 231，则正式开启测试并报告所有预注册 arm；不能按 test 胜负反复修方案；
- 若要保持 test 封存，则 1,871 模型需要另一独立比较群体，不能用其训练数据自证；
- 249/75/4 继续隔离。

### 2. 域级预训练

只有在审计找到非冗余、对应可靠、功能证据可分的潜在域级语料后才设计；迁移有效性仍需未来训练对照：

- 冻结 ESM-2 基线；
- 域级预训练/适配器与无潜变量、确定性对照；
- 域级按家族/组分留出，不能只用精确序列去重；
- 先评估域级泛化，再评估完整 NatV 双域转移；
- 即使域级先验改善残基恢复，也可能进一步削弱共享 z 的作用；未来比较必须同时报告多位点联合生成、共享 z 的增量、以及是否为 KVP40 提供新的合理 NMNAT 改造组合，不能只看 CE 或单残基恢复；
- 与现役 test 的域同源隔离只能在该具体训练协议中定义；本审计未将 test 用于 MMseqs、训练或选择，但执行时曾随 Stage3 全量载入并切片，不能预先宣称已完成隔离；
- 不把不同蛋白的 NMNAT/Nudix 人工拼接，不把单域预训练收益自动外推到双域联合分布。

### 3. 最终部署模型

方法确定后，才讨论用当前合格 2,102 条训练最终生成器。届时不再拥有该模型的独立测试泛化声明；此前预注册比较的证据保留。249 conflict、75 sealed、4 excluded 不自动解封。

## 操作边界

本审计只读旧库与现役角色元数据，不训练、不下载、不折叠；test 未进入 MMseqs、训练或模型选择，但执行时曾随 Stage3 全量载入并切片，实际接触范围见 `scope_correction.json`；不把 13,972 与 2,430 相加，不把 PF01467 命中当 NMNAT 功能，不把旧 strict split 当 NatV 训练/评价划分，不人工拼接双域；249/75/4 继续隔离。未来训练另立协议。

## 交付

- 方案：本文件。
- 报告：[AUDIT.md](AUDIT.md)；范围更正：`scope_correction.json`；产物校验值：`OUTPUTS.sha256`。
- 审计产物：`data/results/natv_evolution_data_audit/raw/`；小型摘要/报告可进入 `experiments/natv_evolution_data_audit/`。
- 旧仓库只读快照与 SHA 在审计清单中记录，不改旧路径或历史身份。
