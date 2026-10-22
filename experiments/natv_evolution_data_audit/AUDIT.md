# NatV 演化信息利用数据审计报告

状态：CPU 审计已完成；未训练、未下载、未折叠、未启动 GPU。旧 `nad_qvae` 只读。

本报告回答：旧 Stage 1–3 的 13,972 条中，除当前 NatV 群体外，是否存在可整体加入、可域级预训练或值得进一步审阅的 NMNAT 演化信息。结论是：**8,849 条 PF01467-only 不能整体加入 NatV 训练，也不能按本次 PF01467 局部切片整体预训练；现有结果没有验证迁移收益。556 条/44 簇的 EC 注释候选和全长复核后的 63 条/19 簇值得做来源功能证据与完整域边界审阅，但都不是现成训练库。** 这不能被读成“远缘资料没有潜在价值”。

## 1. 范围与必须保留的更正

已执行的主审计脚本 SHA256 为 `2533740d47b2a4981776651be1f4c6d9a429e6b36af448b9ba125334009619f7`，执行源码副本在 `data/results/natv_evolution_data_audit/raw/audit_evolution_data_executed.py`。

该脚本实际把旧 Stage3 FASTA 全部 13,972 条载入内存，并对其全部 profile 行取序列、切 PF01467 域。其中包含现役 test 225 条、旧 sealed 71 条、annotation conflict 249 条、excluded 4 条。因此先前“未读现役测试序列内容”的声明不成立，撤回；原始 `audit_summary.json` 中相同表述也已由 `scope_correction.json` 取代。该文件逐项记录载入角色、未进入的 MMseqs/训练环节、含 test 元数据的输出与保留的 test 封存状态。

实际影响范围必须分开写：

- 被载入并做域切片：13,972 条，包括上述 test/sealed/conflict/excluded；`stage3_audit.csv` 含其角色、域起止和长度。
- 进入两个 MMseqs 输入的只有：8,849 条 stage3-only PF01467 域，以及 1,871 条 training+validation 的 NatV NMNAT 域。test/sealed 没有进入聚类、nearest 目标或任何输出 FASTA。
- 本轮没有训练、模型选择、生成或 test 性能评价；审计的 8,849/聚类/nearest 结论不是由 test 序列计算的。但不能再说 test “未被接触”。未来若使用 test 做正式评价，必须披露这次载入/切片，并继续按预注册边界处理；test 不能删除或解封。
- 后续补充脚本 `audit_wt_nmnat_reference.py` 只读已导出的 8,849 条 old-only 域；`audit_wt_full_context.py` 只用白名单物化这 8,849 条 stage3-only 全长序列，并断言与 protected roles 交集为 0。

## 2. 输入、产物与成本

输入 SHA256：

| 输入 | SHA256 |
|---|---|
| Stage3 accepted FASTA | `2277e217904f8ecba748527e673c0f12ce3ff49dfc8ab9528769faac49a46e29` |
| `domain_profile_manifest_v4.csv` | `b8961fb95a3c51de887ab433bbc86c66195a73ea3701b50439d6139a98ce4e80` |
| `corpus_boundary_manifest.jsonl` | `b4c05d8e8e2d77a13a4c70e5e3d66244e7df9666b82d7d3d102425191bbcdc33` |
| `data/processed/roles.csv` | `2f341929ee945bcb1d0c9665750cb5e1c64236e154f11c67957a63d862f830ac` |
| `data/processed/samples.csv` | `d101f3abbc3c3e8db846110385ee484ca01eb5d2102b97d0ace6547b6c03e4e7` |
| PF01467 HMM | `3055836d6ce3009a9d7f976acddda55cd10f79a29937e8fb442b9a99b1994f73`；LENG=133 |
| WT NatV NMNAT 域 4–181 | 序列 SHA256 `9654b7e3d616ca85d10e52efe4acc60965d10a414d40d2aaff5f75ad39c02eac`；178 aa |

产物在 `data/results/natv_evolution_data_audit/raw/`：`audit_summary.json`、`stage3_audit.csv`、`pf01467_only_audit.csv`、`nearest_natv_nmnat.csv`、`wt_reference_summary.json`、`nearest_wt_nmnat.csv`、`wt_full_context_summary.json`、`wt_full_context_hits_t50.csv`、`wt_full_context_hits_t80.csv`、`anchor_mapping_old_only_reps.csv`、`wt_hit_anchor_mapping.csv`、`audit_run.log`。脚本源码身份另见 `audit_evolution_data_executed.py` 与 `data/interim/natv_evolution_data_audit/` 下的 MMseqs 工作目录。

MMseqs 18.8cc5c，CPU-only；主审计约 1 分钟，两个补充搜索各约 12 秒。未使用 GPU，未训练，未下载。

## 3. 群体与重叠

Stage3 接受 13,972 条：PF01467-only 8,849；双域 2,397（有序 2,391，另 6 条重叠/逆序）；两域均未命中 2,703；Nudix-only 23。

精确交集：stage3 ∩ 当前 roles(2,430) = 2,393；stage3-only = 11,579；current-only = 37。2,391 条有序双域全部已在当前 roles：training 1,644、validation 198、test 225、conflict 249、sealed 71、excluded 4。**旧双域部分几乎就是当前 NatV 群体，不能作为新增双域训练数据；13,972 与 2,430 不能相加。**

## 4. 功能证据：flag 交叉优先于互斥层

`primary_layer` 只是显示优先级：EC dual > EC NMNAT-only > EC NaMNAT-only > NadR > text > unknown，不是互斥生物功能。8,849 条的层计数为 text_only 4,973、NadR 2,990、EC dual 491、EC NMNAT-only 302、unknown 92、EC NaMNAT-only 1；但各 flag 交叉更关键：

| 组合 | 数量 |
|---|---:|
| 任一 EC（2.7.7.1 或 2.7.7.18） | 794 |
| EC 且有 NadR flag | 37 |
| EC 且有 header text NMNAT/NAD flag | 790 |
| EC 且有 annotation conflict | 492 |
| NadR 且有 text flag | 3,020 |
| NadR 但无 text flag | 7 |
| EC dual / EC NMNAT-only / EC NaMNAT-only | 491 / 302 / 1 |

因此不能把“EC 层”当成没有 NadR 背景的纯功能层，也不能把互斥层计数相加成独立类别。`experimental_evidence` 对所有 8,849 条都是 `UNKNOWN`；EC 是 UniProt 注释，不是实验确认。本报告中的 “EC 支持” 只表示注释候选，不表示实验验证、审校可靠或已证明 NMNAT 活性。

318 条 complete EC-dual 记录带旧 `annotation_conflict`，这是旧双 EC 规则造成的，不应机械当作错误剔除；它们是注释提示可能具相关/相同反应的远缘候选层，功能仍需核查，应单独处理。

## 5. 域边界：PF01467 profile 不等于 NatV CDD 域

PF01467 HMM LENG=133；8,849 条 old-only 的 PF01467 切片长度中位 130、profile coverage 中位 0.985。高覆盖记录支持**覆盖该 PF01467 profile 核心**，但不是所有记录都完整，更不是天然完整 NatV CDD 域或生物学完整域。

- 8,849 条中 profile coverage ≥0.8 有 5,537；切片长度 ≥124 有 5,690；两者同时满足 5,436。124 是执行脚本中的探索性长度阈值（约为 HMM LENG 133 的 93%），没有生物学验证。
- 5,436 条中仍有 85 条 `partial=true`、1 条多命中；因此“profile+长度筛选”不能一概称为完整域。
- 556 条 complete+EC（238 EC NMNAT-only + 318 EC dual）没有 partial 和多命中；这只是“profile/长度筛选 + EC 注释候选”，不是可立即预训练的合格功能库。
- 8,849 条 old-only 全长蛋白的 PF01467 域之后仍有中位 57 aa 的 C 端侧翼（6,782 条 ≥43 aa），说明**局部切片不是完整蛋白上下文**。局部切片相对 WT/CDD 的 C 端未覆盖，首先表示“本次截取/对应窗口未包含该区”，不证明源蛋白天然缺少该段，也不能据此判定不可迁移。

## 6. 冗余与聚类语义

聚类命令固定 `--cov-mode 0 -c 0.8`（query 与 target 双向覆盖 0.8）；`cluster-mode 0` 是 greedy set-cover，`cluster-mode 1` 是 connected components。簇数不是独立家族数。

| 阈值/模式 | 总簇 | old-only | both | current-only | old-only 大小中位/最大 |
|---|---:|---:|---:|---:|---:|
| 70% mode1 | 745 | 626 | 0 | 119 | 1 / 1,545 |
| 70% mode0 | 823 | 695 | 0 | 128 | 2 / 1,006 |
| 50% mode1 | 247 | 197 | 0 | 50 | 2 / 1,855 |
| 50% mode0 | 274 | 221 | 1（8 条） | 52 | 2 / 1,158 |
| 30% mode1 | 55 | 52 | 1（2,300 条） | 2 | 2 / 3,181 |
| 30% mode0 | 64 | 60 | 2（中位 1,052.5） | 2 | 4.5 / 2,276 |

70% 下 8,849 条 old 域全部落入 old-only 簇，old-only 中位簇大小 1、最大 1,545；冗余很高。30% 的巨大 both 簇是超家族连通合并，不能当作共同功能家族或独立样本。

## 7. 与 NatV NMNAT 的对应

### 7.1 原始分析：当前可见 NatV 域面板

以当前 training+validation 的 1,871 个 NMNAT CDD 域为目标，`min-seq-id 0.3`、`cov-mode 2 -c 0.8`（query=old PF01467 域），nearest 取当前命中中按 identity/query coverage/target coverage 排序的最佳局部命中，不是穷举天然最近亲缘：

- 3,542 条有命中；≥50% identity 143 条；100% identity 3 条；5,307 条无命中。
- target coverage 中位 0.728、最大 0.828；当前 CDD 域长度在 151–229 aa 间变化，短注释会抬高 target coverage。
- 3 条 100% 命中均为 text-only，域长 138/124/59，target coverage 0.762/0.681/0.309，无 EC/UniProt；它们不是已验证完整 NMNAT。
- “无命中”只表示在本次 min-id 0.3、qcov 0.8 和当前搜索设置下无报告，不表示无同源性或属于其它蛋白家族。

### 7.2 补充分析：固定 WT NMNAT 域

固定 WT 域 4–181（178 aa）为目标，单独运行 `min-seq-id 0.3`、`cov-mode 2 -c 0.8`。输入只有 8,849 条 old-only 域，不读 Stage3/test：

- 496 条有命中；≥50% 仅 1 条；≥70% 0 条；115 个 70% mode1 簇。
- target coverage 按 target 跨度 `(tend-tstart+1)/178` 计，中位 0.702、最大 0.758，**没有一条达到 0.8**。若按 alignment 列数 `alnlen/178` 计，最大 0.775；496 条中有 20 条两种口径不同，但都不改变“无一条达到 0.8”的结论。这是局部切片窗口与 WT CDD 域不等价的表现，不是源蛋白天然缺失。
- 层分布：EC NMNAT-only 51 条/18 簇；NadR 75 条/3 簇（target coverage 中位仅 0.20，多为局部匹配）；text-only 363 条/106 簇；unknown 7 条/5 簇。115 个簇中 18 个含 EC、90 个仅 text、16 个含多个 primary_layer（类别可重叠）。
- PF01467-complete + EC 的 35 条命中落在 8 个簇，全部 EC NMNAT-only；identity 中位 31.3%，target coverage 中位 75.3%，缺 C 端中位 43 aa。35 条中 H14 35/35、H17 35/35、T135 29/32（另 3 条 S）、R138 32/32 与 WT 对应；T135 与 R138 各有 3 条未落在单最优局部比对块内。该锚点结果只是候选对应线索，未处理并列最优/低同源不确定性，不验证反应功能或迁移价值。

20%/qcov0.5 的补充诊断有 3,408 条命中（EC NMNAT-only 289、text 1,903、NadR 1,196、EC dual 0）。EC dual 在固定 WT 域的 30%/0.8 和 20%/0.5 搜索中均为 0 命中；这不能判其功能不相关，只能说明在本次序列搜索下与 WT 域无可报告同源。

### 7.3 补充分析：全长蛋白上下文

局部 PF01467 切片会漏掉 C 端延伸。以 8,849 条 old-only 全长蛋白为 query、WT 178 aa 域为 target，要求覆盖的是 WT 域而不是完整旧蛋白（`cov-mode 1`，`min-seq-id 0.3`）：

| 目标覆盖 | 命中 | 70% 簇 | identity 中位 | target coverage 中位（target span） | 右侧延伸中位 | EC 支持 | complete+EC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ≥50% | 401 | 71 | 31.3% | 0.753 | 0 aa | 49 | 45 / 9 簇 |
| ≥80% | 63 | 19 | 30.8% | 0.904 | 22 aa | 7 | 6 / 3 簇 |

≥80% 的 63 条全部为 30.0–33.5% identity；层分布 text-only 54 条/16 簇、EC NMNAT-only 7 条/4 簇、unknown 2 条/2 簇；19 个簇中 13 个纯 text、2 个 EC+text、2 个纯 EC、1 个 unknown、1 个 unknown+text。target coverage 同样按 target 跨度计；若按 alignment 列数，t50 的 ≥0.8 计数从 63 变为 65（401 条中 21 条两种口径不同），t80 的 63 条不变（2 条两种口径不同但都 ≥0.8）。也就是说，完整上下文能把“局部切片 C 端未覆盖”部分补回来，但得到的主要是远缘、以 text-only 为主的候选；EC 支持的全长命中只有 7 条/4 簇（complete+EC 6 条/3 簇）。

这组全长分析是新增探索性分析，不能替代原始 30%/0.8 的 556/44 注释候选集合，也不能当作独立验证；它只说明局部切片会低估域覆盖，后续应做目标感知的完整域边界重审。

## 8. 结论与下一步

1. **不整体加入。** 8,849 条 PF01467-only 不能整体加入 NatV 训练，也不能按本次 PF01467 局部切片整体做域级预训练；现有结果没有验证迁移收益。13,972 与 2,430 不能相加。
2. **保留候选层。** 556 条/44 簇的 profile+长度+EC 注释候选，以及全长复核后的 63 条/19 簇（EC 7/4，complete+EC 6/3），值得做来源功能证据与完整域边界审阅；它们不是现成训练库。318 条 EC dual（主要 metazoan NMNAT1/2/3）的注释提示可能具相关/相同反应的远缘候选，功能仍需核查；本次固定 WT 序列搜索无命中，不能据此判为功能不相关，也不能与 NatV-like 数据混合。
3. **边界任务独立。** 下一步应先做目标感知的完整域边界重审：以 NatV CDD/WT 域为参照，在完整原蛋白上下文重新定义候选域，而不是继续使用 PF01467 best-alignment 切片；重审前不做阈值扫描或模型诊断。
4. **功能证据独立。** 旧 `nmnat_specificity_reference_v1` 的 61 正域/1,027 负域（7 正簇/10 负簇）历史线索可先核查可靠实验/审校证据；`pretraining_validation_status` 标为 `REFERENCE_DIVERSITY_AUDITED_CLASSIFIER_UNVALIDATED`、`classifier_training_eligible=false`，不能把其输出当真值。不因此下载或新训。
5. **数据角色实验仍单列。** 1,644 vs 1,871 的验证并入实验、最终 2,102 条部署训练、以及任何 test 开启，都需要另立预注册与预算；本审计不授权。249/75/4 继续隔离。

## 9. 复现入口

- `scripts/audit_evolution_data.py`：已执行主审计；其源码身份与 scope 更正见第 1 节。不要把它当作 test-safe 模板重跑。
- `scripts/audit_wt_nmnat_reference.py`：固定 WT 域补充分析，只读 8,849 条 old-only 域。
- `scripts/audit_wt_full_context.py`：白名单全长上下文补充分析，只物化 8,849 条 stage3-only；断言 protected roles 交集为 0。
- 原始摘要：`data/results/natv_evolution_data_audit/raw/audit_summary.json`、`wt_reference_summary.json`、`wt_full_context_summary.json`。
- 范围更正：`scope_correction.json`；保留原执行源码 `audit_evolution_data_executed.py` 及其 SHA256；产物校验值见 `OUTPUTS.sha256`。
- coverage 口径：`target_coverage` 为 target 跨度 `(tend-tstart+1)/178`；`target_coverage_alignment` 为原始 `alnlen/178`，有 gap 时两者不同。该派生列由既有 MMseqs 输出重算，未重跑搜索。
