# NatV 演化信息利用数据审计（草案）

状态：审计协议草案，未训练、未下载、未修改旧 `nad_qvae`，未读取现役测试序列内容。

## 目的

回答三个问题，再决定是否改变训练人口或增加域级预训练：

1. 旧 Stage 1–3 的 13,972 条中，除当前 NatV 2,430 条外，真正非冗余、功能相关、域完整且与 NatV NMNAT 对应可靠的潜在域级语料有多少？
2. 是否存在值得进一步设计训练对照的 NMNAT 域级信息？（“是否改善”必须由未来独立实验回答。）
3. 即使域内建模改善，能否转移到完整 NMNAT–Nudix 双域生成？这需要独立的训练/评价实验，不能由序列数量或同源距离直接推断。

第一问是数据审计；第二、三问需要独立的模型实验，不能由序列数量直接推断。

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

training 1,644 条中 47 个同源组分；前五个组分合计 1,491/1,644 = 90.7%（`data/processed/samples.csv` 的 training 角色范围）。条数不等于独立演化信息；不引用未注明角色范围和 identity/coverage 口径的内部最近邻统计。

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

- PF01467 单独命中的域：按 profile alignment 边界提取；按 coverage ≥0.8、长度/边界合理、非 fragment 分层；coverage 阈值只是操作性完整度分层，不是功能正确判定。
- 使用本机 `mmseqs` 对域序列做同一性/覆盖率聚类：记录版本、identity 定义、cov-mode/cluster-mode 和输入域边界；双向覆盖阈值预定为 ≥0.8；70/50/30% 是三个固定相似性阈值，不是“距离层”。报告簇数及阈值敏感性；每个簇不等于独立演化样本或生物家族。
- 比较当前可见合格 1,871 条（training 1,644 + validation 227）与旧独有域记录的新增簇、已有簇扩张、仅同组分重复；当前 2,430 只用于角色/元数据计数；不把 test、旧封存或隔离序列构造成新输入。
- 本审计不读现役 test，因此不宣称未来域级预训练已与 test 完成同源隔离；该隔离须在未来具体训练协议中定义。
- 输出：`pf01467_domain_clusters.csv`、`redundancy_summary.json`。

### C. 功能与分类证据

- 分层：明确 EC 2.7.7.1、仅 2.7.7.18、两 EC 并存、NadR 多功能、仅 header 名称、UNMAPPED、conflict/partial。
- 两 EC 并存不等于错误或负例；PF01467 命中不等于 NMNAT 功能；NadE/NADSYN 或其它反应不得因“NAD 相关”混入。
- header 方括号名称仅作分类学初分；同一精确序列多个名称要保留冲突，不强行合并成“物种样本”。
- 若需要权威 taxonomy/family，必须另立来源与校验，不在本审计中远程下载。

### D. 与 NatV NMNAT 域的对应

- 用当前可见的 training/validation NMNAT 域（CDD 边界）与 PF01467 单独命中的域做序列相似性、覆盖和 HMM 分数比较；报告 70/50/30% 固定相似性阈值下的对应情况。
- 结构模板兼容仅使用现有 2QJO/2QJT/2R5W 与锚点映射；不做折叠、不把结构相似性当功能。
- 这一阶段只回答“是否存在非冗余、对应可靠、功能证据可分的潜在域级语料”；迁移是否有效必须由未来训练对照判断，不能凭同源距离或结构对应裁定。不训练、不拼接双域、不生成候选。

### E. 审计结论

- 若 8,849 条经聚类后主要是当前家族内重复、低覆盖片段或功能不相关家族：停止扩训练/预训练，报告负结果。
- 若存在非冗余、域完整、功能证据可分层、与 NatV NMNAT 对应可靠的潜在域级语料：再单独设计域级预训练实验；不得凭同源距离或结构对应称迁移有效，也不得直接进入训练。
- 若证据只能支持 header 层分类：只能做探索性数据描述，不能称为家族或物种层独立样本；两 profile 未命中的 2,703 条也不能据此断言无目标信息。

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
- 与现役 test 的域同源隔离只能在该具体训练协议中定义；本审计未读 test，不能预先宣称已完成隔离；
- 不把不同蛋白的 NMNAT/Nudix 人工拼接，不把单域预训练收益自动外推到双域联合分布。

### 3. 最终部署模型

方法确定后，才讨论用当前合格 2,102 条训练最终生成器。届时不再拥有该模型的独立测试泛化声明；此前预注册比较的证据保留。249 conflict、75 sealed、4 excluded 不自动解封。

## 操作边界

本审计只读旧库与现役角色元数据，不训练、不下载、不折叠、不读取现役 test 序列内容；不把 13,972 与 2,430 相加，不把 PF01467 命中当 NMNAT 功能，不把旧 strict split 当 NatV 训练/评价划分，不人工拼接双域；249/75/4 继续隔离。未来训练另立协议。

## 交付

- 方案：本文件。
- 审计产物：`data/results/natv_evolution_data_audit/raw/`；小型摘要/报告可进入 `experiments/natv_evolution_data_audit/`。
- 旧仓库只读快照与 SHA 在审计清单中记录，不改旧路径或历史身份。
