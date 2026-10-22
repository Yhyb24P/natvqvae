# NatV NMNAT 演化信息用于设计：研究计划

状态：Phase A 已执行（Gate A PASS，7 条假设），结果见 [HYPOTHESES.md](HYPOTHESES.md)；Phase B 未启动。

## Context

审计已收口（`096008d`）：8,849 条 PF01467-only 不能整体加入 NatV，也不能按 PF01467 局部切片整体预训练；PF01467 HMM 为 133 aa，而 KVP40 WT CDD NMNAT 域为 178 aa，局部切片与完整域不等价。全长复核后只有 63 条/19 簇覆盖 WT 域 ≥80%，其中 EC 7 条/4 簇、complete+EC 6 条/3 簇，identity 仅 30–33.5%，多数为 text-only 注释。

现役条件 QVAE 的形式比较未取得有意义的共享 z 联合建模或候选供给优势；代表候选的 NMNAT:Nudix = 6:30，主要来自选择阶段只构造 Nudix exploratory 池，而不是生成预算偏置。现有结构资产只有 2QJO/2QJT/2R5W 与已核对的锚点/配体距离表；没有 KVP40 实验结构、ATP/NMN 三元结构、kcat/Km 或底物特异性标签。

审计时曾随 Stage3 全量载入 test/sealed 并切片；这些序列未进入 MMseqs/训练/选择，已由 `scope_correction.json` 记录。本计划不读 test，1,644/227/231 继续作为方法比较划分；最终部署人口另立协议，不在本计划解封 test。

本计划不再扩大审计，也不立即做域级预训练。目标是用已有、边界明确的候选来源，先回答“远缘 NMNAT 在 KVP40 NMNAT 锚点及邻近底物结合区域提供了哪些可核对的天然变化，能否形成此前 NatV 语料没有的合理设计假设”；只有得到可信假设后，才比较 QVAE、证据重排/先验与确定性对照。

## 关键判断

- 远缘不等于无价值，但 30–33% identity 处于 twilight zone；结构邻近不等于功能，EC/文本注释不等于实验。
- 556/44、63/19、318/22 是候选来源集合，不是训练库；簇数不是独立家族数，不能相加成样本量。
- QVAE 仍是主生成器；生物学知识可以先作为设计假设和候选重排信号，不必先转成训练数据或潜变量通道。
- 若证据只能支持当前 NatV 已有的变异，或不能形成不同 70% 序列簇支持，则报告负结果并停止，不降低阈值凑候选。

## Phase A：证据整理与设计假设（CPU，预计 <2 CPU-hours；无下载/折叠）

### A1. 重建来源集合

| 集合 | 来源 | 用途 |
|---|---|---|
| S1 | `data/results/natv_evolution_data_audit/raw/pf01467_only_audit.csv` 中 `complete & (ec_nmnat_only|ec_dual)`：556 条/44 个 70% mode1 簇（238 EC NMNAT-only + 318 EC dual） | 主要 EC 注释候选 |
| S2 | `wt_full_context_hits_t80.csv`：63 条/19 簇（401/71 的 t50 只作敏感性） | 完整上下文序列证据；其中 6 条已在 S1，不能当独立验证 |
| S3 | 318 条 EC-dual/22 簇 | 注释背景层；固定 WT 搜索无命中，不强行对齐 |
| S4（可选） | 旧 `nmnat_specificity_reference_v1/domain_reference_v3/`：61 正域/7 簇（44 REVIEWED_ANNOTATION + 17 PROTEIN_LEVEL_REVIEWED） | 仅交叉核对“该变异是否出现在独立审阅的 NMNAT 注释中”；不训练分类器，不把 1,027 负域当功能阴性 |
| S5 | `data/processed/samples.csv`、`training.fasta`、`validation.fasta`：1,644+227 NMNAT CDD 域 | 家族天然变异与组分基线；不读 test |

`inputs.json` 记录实际输入 SHA256（复用 `src/natvqvae/intake.py:sha256_file`）；断言不包含 test/sealed/conflict/excluded 的 sequence_id。所有序列上下文只取 old-only 的 8,849 条 whitelist 或 S4 正域。

### A2. 代表序列与证据层

- 每个 70% mode1 簇选一个代表：按 tier、target coverage、域长度、S4 evidence tier 排序；未使用 seq_hash 或 C 端缺口 tie-break。记录簇大小与选择规则。
- 证据层：T1 = 全长 ≥80% WT 覆盖 + complete + EC（6/3 簇）；T2 = complete+EC 局部（556/44，注明 C 端窗口）；T3 = 全长 text-only（63/19 减去 T1）；T4 = EC-dual 318/22（无 WT 序列命中，仅注释层）；T5 = reviewed positives 61/7（交叉核对）；S5 = 家族基线。
- 主分析不把 S1/S2/S3/S4 的数量相加；每个结论写清来源层和簇支持数。

### A3. 目标感知的完整域对齐与邻域表

- 以完整蛋白为 query、WT NMNAT 4–181 与 WT 1–341 为 target，使用 BLOSUM62 局部比对（不重跑 MMseqs、不做全局比对）；拒绝仅用 PF01467 局部切片推断锚点。
- 记录：覆盖、identity、gap、锚点列稳定性、H14/H17/T135/R138 的 WT-relative 残基、锚点 ±6 位线性序列邻域。只复用 `data/processed/anchor_mapping_audit.csv` 的最近锚点距离；`domain_pocket_map.json` 未作为候选几何真值，也没有逐残基口袋距离。不新算对接或折叠；2QJO 是 NAD 产物参照，不是 ATP/NMN 中间态。
- 对每个位置统计各证据层的替换、不同 70% 序列簇支持数、S5 中的 NatV 频率；同一序列多名称/多 EC 保留冲突，不强行归类。
- 输出 `source_inventory.csv`、`cluster_representatives.csv`、`target_aware_alignment.csv`、`anchor_neighbor_variation.csv`、`s5_position_frequencies.csv`、`hypothesis_table.csv`、`gate_A.json`。

### A4. 设计假设表

- 目标 6–8 条，硬上限 12 条；仅 NMNAT 4–181、全长 341 aa、单/双替换、无 indel、不碰 Nudix/linker/末端。
- 入选规则：
  - 至少 2 个不同 70% 序列簇在该替换/位置组合上支持；直接改 H14/H17/R138 需要至少 2 个证据层且无结构冲突，T135 的天然 K/S/A 只作候选；
  - 优先锚点邻近/口袋第二壳层、当前 NatV 稀有的替换；与 S5 频率、QVAE 代表候选比较，排除已常见的组合；
  - 记录模板邻近、配体距离、局部空间/电荷风险与 UNKNOWN；不称“提高活性”。
- 每条假设包含：variant_id、位置与替换、支持簇/层、S5 频率、模板依据、与现有 QVAE 候选的差异、匹配 WT/对照、待检验问题。
- **Gate A**：若没有≥2 个不同 70% 序列簇支持、或只有 T4/text-only 支持、或全部是 S5 已常见变异，停止并报告“没有可信的新 NMNAT 替换”；不进入 Phase B，不降阈值。

### A5. Phase A 交付

- `experiments/natv_evolution_informed_nmnat/PLAN.md`、`HYPOTHESES.md`、`inputs.json`、`status.json`。
- `data/results/natv_evolution_informed_nmnat/raw/phase_a/`：来源清单、簇代表、对齐表、锚点/邻域表、假设表、`gate_A.json`、运行日志。
- 脚本：`scripts/build_nmnat_evolution_hypotheses.py`（CPU；只读现有 CSV/FASTA，不重跑 8,849 审计）。

## Phase B：门控比较（B1 CPU；B2 GPU，需单独授权）

只有 Gate A 通过才启动。B1 先回答“证据能否在现有候选池中选出更合理、尚未见过的 NMNAT 变化”，它不改变生成分布；B2 才把证据作为推理时先验，比较真实联合概率。这样避免把证据重排误写成模型分布改善。

### B1. 候选重排诊断（CPU，不重训、不改变分布）

- 输入：现役正式候选池 `data/results/natv_conditional_qvae/raw/candidates.parquet`、`matched_attempts.parquet`、`representative_candidates.csv`；不生成新序列。
- 按 Phase A 的透明证据分重排/筛选 NMNAT 候选；证据分只作用于 NMNAT 4–181（Nudix/linker/末端不变），由不同 70% 序列簇支持、证据层和 S5 频率构成，不训练新分类器。同预算比较：
  - A：现役代表选择（无证据）；
  - B：QVAE 候选按证据分重排；
  - C：deterministic 候选按同一证据分重排；
  - D：QVAE + 簇置换证据分（同分布对照）。
- B1 指标：留出证据簇的替换 precision/recall、与 S5/QVAE 的 novelty、锚点/口袋保持、NMNAT 多样性；选中候选的联合概率只作描述，不称模型联合概率改善。
- **Gate B1**：B 的留出证据 precision 高于 A 且 cluster bootstrap 区间不含 0，多样性不塌缩；若目标替换在现有候选池中缺失，记录“候选供给不足”，不把缺失解释为模型不支持。未通过则停止并报告负结果。

### B2. 推理时证据先验的正式比较（GPU；需新预算授权）

- 复用现役 QVAE/deterministic checkpoint（`experiments/natv_conditional_qvae/checkpoint_manifest.json`、`data/models/natv_conditional_qvae/`），不重训；把 Gate A 的证据支持作为推理时 additive prior/score（只作用于 NMNAT 4–181），改变采样分布。先验权重在运行前冻结，不以 227 或 test 调权重；如需敏感性分析，只限预注册的固定档位。
- 固定 NMNAT 掩码 8/16/32、16 个固定掩码、5 个采样种子、每组合 64 draws，共 15,360 attempts/arm；所有 arm 同种子/掩码/预算，失败与 WT 保留在分母。
- Arms：A 现役 QVAE；B QVAE + 证据先验；C deterministic + 同一证据先验；D QVAE + 簇置换先验（同分布对照）。
- 评价（主要）：NMNAT 位置的真实联合 log 概率（`conditional_evaluation.posterior_joint_probability`、`prior_joint_probability`、确定性精确值），nats/残基与 MC SE；有效氨基酸组合、per-position entropy、unique combinations、ESS、每千次尝试的候选供给。
- 评价（次要）：留出证据簇的替换 precision/recall；锚点/口袋保持、Nudix 不变、与 S5/QVAE 的 novelty 和多样性。
- 负结果判据：NMNAT 联合 NLL 退化超过 `max(1e-4, 2×MC SE)`；有效组合塌缩超过预注册 20%；B 相对 A 的留出证据 precision 区间含 0；C 与 B 相当（说明增益来自证据而非 z）；Gate B1 不通过。出现任一即报告负结果，不调参重试。
- 231 test 继续封存；227 validation 只作 QVAE 评分背景，不用于拟合外部证据；证据簇按 70% cluster 留出。若未来使用 test，必须另立预注册并披露审计时的 test 载入/切片。

### B3（仅 B2 通过后）

- 把证据转成条件/训练信号，需新的 GPU 授权、确定性同信息对照和独立验证；不在本计划默认执行。

## Phase C：可选结构审阅（现有模板）

- 只审阅 4–8 条假设 + WT + 匹配对照；使用 2QJO/2QJT/2R5W 与已有编号/缺口/配体距离表。
- 报告锚点/第二壳层局部几何、模板缺口、配体距离与 UNKNOWN；不把 pLDDT/模板邻近/低 RMSD 当活性。
- 不新折叠、不对接；若必须预测，另立预算与后端授权（BioEmu2 等不默认启用）。

## 协调与分工（与 codex）

- 开工前发送 Phase A 范围、脚本/产物路径、分工；Phase A 完成发 RESULT；Phase B 前再次确认 GPU 预算。
- 我们负责证据表、对齐/邻域提取、Phase B 评价实现；codex 复核证据层定义、锚点/边界解释和比较指标，不并发重跑 8,849 审计。
- 任何一方发现证据不足或口径冲突，先停 Gate，不以更多筛选或调参绕开。

## 验证

- Phase A：运行 `python scripts/build_nmnat_evolution_hypotheses.py --run experiments/natv_evolution_informed_nmnat`；检查 556/44、63/19、318/22、61/7 的来源计数、簇代表唯一性、不包含 protected roles、锚点分母与缺口说明、输入/输出 SHA256。
- 人工核对 7 条假设：每条都有≥2 个不同 70% 序列簇、明确证据层、S5 频率、锚点依据、UNKNOWN 和匹配对照；若没有，`gate_A.json` 必须写明停止原因。
- Phase B：B1 在 CPU 上对既有候选池 dry-run 并核对证据分/簇留出；B2 先 dry-run 校验 arm/掩码/种子/预算/先验权重与 checkpoint SHA，再申请 GPU 授权运行；输出完整失败、WT、重复与 MC 记录。
- 不新增训练/下载/折叠；不修改 1,644/227/231 角色；249/75/4 继续隔离。

## 主要文件

- 复用：`experiments/natv_evolution_data_audit/AUDIT.md`、`data/results/natv_evolution_data_audit/raw/*`、`data/processed/samples.csv`、`anchor_conservation_by_position.csv`、`anchor_mapping_audit.csv`、`structural_interpretation_basis.json`、`experiments/natv_conditional_qvae/checkpoint_manifest.json`、`data/results/natv_conditional_qvae/raw/candidates.parquet`、`matched_attempts.parquet`、`src/natvqvae/conditional_outputs.py`、`conditional_evaluation.py`、`conditional_study.py`、`scripts/audit_wt_full_context.py`。
- 新增：`experiments/natv_evolution_informed_nmnat/`、`scripts/build_nmnat_evolution_hypotheses.py`、Phase B 的 `scripts/run_evolution_informed_qvae.py`（只在 Gate A 通过后创建）。
- 完整候选/轨迹/权重：`data/results/natv_evolution_informed_nmnat/`、`data/models/`；不入 Git。
