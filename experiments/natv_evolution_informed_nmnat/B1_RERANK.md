# Phase B1：现有候选池的 NMNAT 证据重排诊断

状态：`DESCRIPTIVE_ONLY`；`gate_b1=not_evaluable`；无 held-out 证据；CPU-only 1.3 s。未生成新序列、未训练、未下载、未折叠、未读 test；B2 未启动。总结论见 [REPORT.md](REPORT.md)，运行身份见 [RUN_PROVENANCE.json](RUN_PROVENANCE.json)。

本诊断只回答：Phase A 的 7 条假设能否在**现有** NMNAT 候选池中被选中，以及原始候选池是否包含这些替换。它不检验模型分布、不评价候选优劣、不构成 Gate B1 通过。

## 方法

- 输入：`data/results/natv_conditional_qvae/raw/annotated_candidates.parquet`；每个模型 61,440 条候选，`region=NMNAT` 15,360 条/模型。
- 选择数：3 条/模型，与现役 NMNAT 代表数量一致；同一模型内按 `sequence_sha256` 去重。
- 权重：Phase A 的稳定 old-only 簇数（L18I 6、F13Y 6、T22Q 6、A134G 19、K130R 18、L140M 6、F13Y+A134G 6）；只用于重排，不训练。
- 对照：
  - A：QVAE 历史 NMNAT 代表（3 条；deterministic 历史代表未纳入 summary/selected，选择目标不同）；
  - B：QVAE 按证据分选 3 条；
  - C：deterministic 按同一证据分选 3 条；
  - D：QVAE 对 6 个单点权重做有放回重抽样（seed=20261022），配对权重固定；只作权重重抽样参照，不是无信息对照；
  - A2：QVAE 按 `NMNAT_HMM_score` 选 3 条（描述性 HMM 代理）。
- 原始供给与选择结果分开：供给是全部 15,360 条 NMNAT 候选中精确含该替换的计数；选择结果只描述 top-3。

## 原始供给

| 模型 | NMNAT 候选 | L18I | F13Y | T22Q | A134G | K130R | L140M | F13Y+A134G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QVAE | 15,360 | 198 | 2 | 446 | 95 | 189 | 14 | **0** |
| deterministic | 15,360 | 200 | 2 | 445 | 92 | 187 | 12 | **0** |

单点替换在现有池中可用，但 `F13Y+A134G` 配对在两个模型的 NMNAT 池中都不存在。配对缺失是候选供给限制，不是模型不能在其他 mask 下生成它的证据。

## 选择结果（描述性）

| 选择 | 命中≥1假设 | 支持事件数 | 加权分均值 | Hamming 均值 | NMNAT 编辑均值 | 唯一序列 | HMM 均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A QVAE 历史代表 | 0/3 | 0 | 0 | 15.0 | 15.0 | 3 | 32.92 |
| B QVAE 证据 | 3/3 | 6 | 37.0 | 22.3 | 22.3 | 3 | 32.65 |
| C deterministic 证据 | 3/3 | 6 | 37.0 | 22.3 | 22.3 | 3 | 32.73 |
| D QVAE 权重重抽样 | 3/3 | 3 | 6.0 | 25.0 | 25.0 | 3 | 35.22 |
| A2 QVAE HMM 代理 | 0/3 | 0 | 0 | 22.0 | 22.0 | 3 | 47.93 |

B 和 C 结果几乎相同，说明在当前池中单点命中的差异主要由证据分选择造成，而不是 QVAE 与 deterministic 的模型差异。B 选中的候选平均编辑数高于 A（22.3 vs 15.0），HMM 均值不高；不能据此说候选更合理或更安全。

## 限制与解释

- **无 held-out**：同一 7 条假设同时用于重排和命中统计，因此“命中率”只是描述性证据匹配/富集，不是 held-out precision 或独立有效性证据；`gate_b1=not_evaluable`。
- **A 不是匹配对照**：现役 3/模型代表是为不同目标选择的；不能把 A 的 0/3 当作模型能力差异。
- **D 不是无信息对照**：它对 6 个单点权重有放回重抽样，配对权重固定，仍会命中这些事件；只能说明权重重抽样的作用。
- **配对缺失**：`F13Y+A134G` 在两个 NMNAT 池中均为 0；现有池无法检验该配对，不能补采或据此启动 B2。
- 所有候选的功能、底物特异性、kcat/Km、NAD⁺ 净积累仍 UNKNOWN；证据匹配不是功能验证。

## 结论

现有池包含多数单点假设，但不含 `F13Y+A134G` 配对；证据重排可以在现有池中选出单点命中，且 deterministic 与 QVAE 几乎相同。由于没有 held-out 证据、A 不是匹配对照、D 是权重重抽样而非无信息对照，本轮**不判 Gate B1 通过，不启动 B2**。

若要继续，应先单独设计一个真正留出的证据评估：按 old-only 70% 序列簇划分 fit/hold-out（或预先冻结一组未参与 Phase A 的独立候选来源），再预注册选择数、mask、预算和指标；若还要检验配对，必须在 GPU 授权下用包含目标位置的定向 mask 生成，不能把本次 0 命中解释为模型不支持。当前不执行。

## 文件

- `data/results/natv_evolution_informed_nmnat/raw/phase_b1/supply.csv`
- `data/results/natv_evolution_informed_nmnat/raw/phase_b1/selected_candidates.csv`
- `data/results/natv_evolution_informed_nmnat/raw/phase_b1/rerank_summary.json`
- `experiments/natv_evolution_informed_nmnat/B1_STATUS.json`
- 脚本：`scripts/rerank_nmnat_candidates.py`
