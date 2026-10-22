# NatV NMNAT 演化设计证据：实验报告

状态：Phase A Gate A PASS（7 条待检验设计假设）；Phase B1 为现有候选池的描述性 supply/rank 诊断，`gate_b1=not_evaluable`；B2 未启动。未训练、未下载、未折叠、未读 test。所有活性、底物特异性、kcat/Km 与 NAD⁺ 净积累仍为 UNKNOWN。

本报告是本次研究的归属报告；详细表格见 [HYPOTHESES.md](HYPOTHESES.md)（Phase A）与 [B1_RERANK.md](B1_RERANK.md)（Phase B1），运行身份见 [RUN_PROVENANCE.json](RUN_PROVENANCE.json)，方案见 [PLAN.md](PLAN.md)。

## 方法与结果

Phase A 从审计后的有界来源重建证据：S1 556/44（238 EC NMNAT-only + 318 EC dual）、S2 63/19、S2-only 57/18、S4 reviewed positives 61/7（仅交叉核对）、S5 1,871 training+validation；S1/S2 重叠不重复计数，T5 簇与 old-only 簇分开。以完整蛋白对 WT NMNAT 4–181 做 BLOSUM62 局部比对，在锚点 ±6 线性邻域内按不同 old-only 70% 序列簇、稳定映射、S5 频率和保守替换筛选，得到 7 条假设：

- 单点：L18I、F13Y、T22Q、A134G、K130R、L140M
- 配对：F13Y+A134G

H07 配对在两位点均稳定对应的 1,537 条 NatV training+validation 序列中未见；H14/H17/R138 直接改造未提出，T135S 已常见，Q23W 非保守被排除。A134G/K130R 的 177/295、104/295 支持更像 old-only 分支差异，不能当作独立突变事件。

Phase B1（CPU-only 1.3 s）只使用现有 15,360 条/模型的 NMNAT 候选，不生成新序列：现役 QVAE 历史代表 0/3 命中；QVAE 与 deterministic 证据重排均 3/3 命中、6 个支持事件；权重重抽样参照 3/3、3 个事件；HMM 代理 0/3。配对 F13Y+A134G 在两个池中均为 0。由于同一 7 条假设同时用于重排和命中统计、且没有 held-out 簇，B1 只是描述性富集/供给诊断，不能判 Gate B1 通过，也没有 QVAE 优势证据。

## 限制与负结果

- 30–35% identity 属 twilight zone；局部比对多最优和低同源对应仍不确定；稳定映射排除了 >32 等分最优的记录，但不等于结构或功能正确。
- `nearest_anchor_template_distance_A` 是最近锚点距离，不是候选替换残基的口袋几何；±6 是线性序列邻域，不是空间第二壳层。2QJO 是 NAD⁺ 产物参照，没有 ATP/NMN 三元结构。
- 70% 序列簇不是独立家族或独立进化事件；T5 `ref:` 前缀也不证明与 old 序列独立。
- B1 的 A 是不同目标的历史 QVAE 代表；D 是同 7 事件权重的有放回重抽样，不是无信息对照；B/C 相同说明当前池中差异主要来自证据选择，不是模型差异。
- 配对缺失按供给限制记录，不补采、不启动 B2。任何功能、稳定性或净积累结论仍需未来实验。

## 下一步

若要继续，应先设计真正的 cluster-held-out 证据或经单独 GPU 授权的定向 mask 实验；当前不自动扩展。test 继续封存，249/75/4 继续隔离。
