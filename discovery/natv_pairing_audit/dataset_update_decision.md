# NatV 数据扩充判断

日期：2026-10-20。范围：本地序列—结构核对、限定范围的外部检索，以及 95 条候选的同源增量分析。

## 判断

当前目标下结束这轮序列扩充，正式 NatV 群体及角色划分保持原状。95 条精确序列新颖的 UniProt 记录保留为待研究群体，见 `external/incremental_uniprot_candidates.csv`。它们扩大了序列组分图，但没有增加经验证的功能成员；直接并入还会影响现有验证与测试比较。输入群体含义见 `../../docs/DATA_PROVENANCE.md`，模板和残基对应见 `../../docs/NATV_PAIRING_AUDIT.md`。

## 本地核对

2,430 条记录均为不同序列，序列 ID / hash 不一致及 FASTA / parquet 不一致均为 0。`training_role_v2` 的群体含义与计数见数据来源文档；另有 `split_v2` 训练 / 验证 / 测试 1,879 / 235 / 235，旧封存 / 排除 77 / 4；legacy split 训练 / 验证 / 参考 / 测试 / 排除为 488 / 309 / 71 / 6 / 1,556。不同字段不能互换。

Active-G 有 1,871 条，均在群体中；群体中另有 559 条不在 Active-G。`exclusions.csv` 共 633 条：249 条注释冲突隔离、75 条旧封存、4 条非标准序列或结构域排除，以及 305 个不合格结构实体。

285 个 PDB 条目共 308 个聚合物实体，仅 3 个通过双 Pfam、双 HMM、结构域顺序与长度 ≥300 的筛选；对应两条不同全长序列。三个实体所有链的已解析残基数分别为 995、684、689；配体为 2QJO 的 APR/NAD/POP/SO4、2QJT 的 AMP/MN、2R5W 的 CL/MG。逐链缺口见 `construct_observed_audit.csv`。

12 个关键残基的对应与指定模板配体邻近检查均满足本次几何判定。2QJO NMNAT 到 NAD 的最短重原子距离为 3.638、3.611、7.667、6.775 Å（2 个 ≤4.5 Å，全部 ≤8 Å）；2QJT Nudix 到 AMP/Mn 为 2.142、3.970、4.002、2.167、3.922、2.006、3.445、2.212 Å（全部 ≤4.5 Å）；2R5W 对应距离为 2.201、4.655、4.227、2.083、3.983、2.182、4.158、2.321 Å（E238 为 4.655 Å，其余 ≤4.5 Å）。2QJO Nudix 关键残基距 NMNAT 位点的 NAD 为 23–33 Å，因此 Nudix 指定模板使用 2QJT。

## 外部检索

RCSB 用 KVP40、Q55928、Q5NHR1 三个种子检索（identity ≥30%，E ≤1），并集为 28 个聚合物实体，全部已在本地结构群体中。3 个是已知全长双结构域实体，其余 25 个是单结构域或不合格同源物；例如 3GZ5/3GZ6/3GZ8 是 Shewanella NrtR。没有找到新的合格全长双结构域实验结构。

UniProt 查询 `xref:pfam-PF01467 AND xref:pfam-PF00293` 返回 880 条（1 条 reviewed），378 条通过双结构域 HMM 筛选；长度 280–450、排除已知模板并去重后为 361 条，均无 PDB 交叉引用。266 条与既有群体精确相同，角色为训练 179、注释冲突隔离 44、验证 20、测试 16、旧封存 7。

其余 95 条为精确序列新颖的 unreviewed TrEMBL 记录，没有实验结构或实测活性。其中 8 条关联噬菌体：Pectobacterium phage vB_PcaM_CBB、Serratia phage BF、Sphingobium phage Lacusarx、Burkholderia phage BcepSauron/BcepSaruman、Stenotrophomonas phage Mendera/Moby、Colwellia phage 9A。噬菌体背景不等于 NatV 功能证据。

IPR041750 查询返回 34 条，17 条通过双结构域筛选，26 条已在家族中，2 条带 PDB 引用且均为已知模板。该注释也含单结构域 NMNAT，不能单独定义双结构域成员。

AlphaFold 元数据查询覆盖 10 个 accession，7 个有模型、3 个返回 404。其中属于 95 条新增候选的有 5 条（3 个有模型、2 个无模型），其余 90 条未查询；现有预测模型尚无结构评价。

## 同源增量分析

采用与 `vnext_family_split` 相同的 70/80 判定：局部 identity ≥0.70，双方比对跨度均 ≥序列长度的 0.80。MMseqs2 提供候选命中，影响判断的命中均用 Biopython 精确重比对。

| 项目 | 结果 |
|---|---:|
| 最近家族 identity：中位数、IQR、范围 | 69.4%；52.5–92.9%；21.6–100% |
| identity ≥90% / ≥70% / <70% | 26 / 47 / 48 |
| 候选内部簇 | 45 |
| 并入已有组分 | 12 个簇，47 条候选 |
| 新组分 | 33 个，48 条候选 |
| 组分数变化 | 142 → 175 |
| 桥接 / 已有组分合并 | 0 / 0 |
| 新组分大小 | 两个 6 条、五个 2 条、26 个单条 |

既有群体中 2,359/2,430 条有非自身 ≥70% 命中，最近邻 identity 中位数为 99.45%。新增候选中 6 条在合格局部比对段上达到 100% identity，但完整序列不同；精确新颖不等于家族独立性。

47 条并入已有组分的候选中，34 条进入测试组分、8 条进入训练组分、5 条进入验证组分。8 条噬菌体相关序列中，4 条并入已有组分、4 条形成新组分。若合并用于正式数据，需要重新聚类和划分，不能沿用原有评价基准。

以 WT 和 Q5NHR1 结构域映射关键残基，自检均恢复 12/12。排除已知 H236/E238 差异位点后，51 条保留 11/11，24 条保留 10/11，8 条保留 9/11，12 条少于 9/11；共 26 个关键残基位置未映射。R249、E254、H15、G234 分别在 95、93、92、91 条中恢复；D307 在 66 条中恢复，另有 22 条为 Ser；非保守 E238 位点在 49 条中匹配。

## 解释限制与后续条件

本次配体结构是产物或金属状态参照，不提供候选自身的 NMN+ATP 或 ADPR 底物三元复合物几何。距离和关键残基恢复只支持结构相容性假设，催化方向与净产量仍为 UNKNOWN。证据层级遵循 `../../docs/RESEARCH_PLAN.md`。

外部计数对应 2026-10-20 的数据库快照。RCSB 检索限于 ≥30% 序列 identity；MMseqs 搜索并非穷尽，未做其他阈值聚类。低 identity（约 35%）下的关键残基对应存在较大不确定性，局部序列比对不能替代结构比对。正式群体的分类学分布尚无汇总表。

未来若重新考虑扩充，可依据新的全长双结构域实验结构、实测 NMNAT/Nudix/净 NAD⁺ 数据，或有独立聚类与划分方案的发现群体研究。预测结构可帮助安排优先级，不能替代前两类证据。对现有 3 个 AlphaFold 模型的结构评价、8 条噬菌体相关序列的文献核对、差异更大的同源物结构检索，均需另定任务与预算；这些是可选方向。

## 证据文件

外部原始检索载荷与检索、汇总、增量分析脚本（`query_external.py`、`summarize_external.py`、`analyze_incremental_homology.py`）已退出；本决策保留汇总、评估、组分与残基映射表。`hashes.json` 与 `source_hashes.json` 是 `0ea7264` 前的核对快照，包含已退出载荷的 SHA256；精确重比对（47,275 行，`c6163e79…`）、候选边表（4,069 行，`4d346493…`）与重复 FASTA 可由该提交追溯。重跑外部检索需从该提交取回脚本并重新获取数据库快照，结果可能不同；结论不因载荷退出而改变。

| 路径 | 内容 |
|---|---|
| `audit_summary.json`、`source_hashes.json` | 本地核对、866 个输入文件及语料目录树校验值 |
| `external/external_summary.json`、`external/discovery_overlap.json` | 外部检索结果与群体重叠 |
| `external/query_log.json`、`external/source_hashes.json` | 请求状态与响应、派生文件校验值的快照 |
| `external/incremental_uniprot_candidates.csv`、`external/uniprot_candidates.csv` | 95 条新增候选与 361 条筛后候选 |
| `external/incremental_structures.csv`、`external/alphafold_metadata.csv` | 结构增量与 AlphaFold 元数据 |
| `external/homology/homology_summary.json` | 同源分析汇总及限制 |
| `external/homology/candidate_homology_assessment.csv` | 逐候选 identity、组分、划分邻近与关键残基恢复 |
| `external/homology/component_expansion.csv` | 组分成员扩展与新增/并入结果 |
| `external/homology/anchor_conservation_by_position.csv` | 两个参考的逐位点恢复与替换计数 |
| `external/rcsb_shortlisted_cif/`、`external/rcsb_metadata/` | 2QJO/2QJT/2R5W 实验结构与模板元数据 |

复现方法与保留文件入口见 `README.md`。
