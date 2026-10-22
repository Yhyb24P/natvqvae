# NatV NMNAT 演化设计证据：报告

从审计后的有界来源得到 7 条待检验的 NMNAT 设计假设；现有候选池诊断只作描述性富集/供给，不作为独立验证。未训练、未下载、未折叠、未读 test；所有活性、底物特异性、kcat/Km 与 NAD⁺ 净积累仍为 UNKNOWN。

详细证据与逐条限制见 [HYPOTHESES.md](HYPOTHESES.md)；候选池诊断见 [B1_RERANK.md](B1_RERANK.md)；方案见 [PLAN.md](PLAN.md)；输入、运行身份与校验见 [inputs.json](inputs.json)、[RUN_PROVENANCE.json](RUN_PROVENANCE.json)、[OUTPUTS.sha256](OUTPUTS.sha256)。

## 结果

来源：S1 556/44、S2 63/19、S2-only 57/18、S4 reviewed positives 61/7、S5 1,871 training+validation；S1/S2 重叠不重复计数，T5 与 old-only 簇分开。完整蛋白对 WT NMNAT 4–181 做 BLOSUM62 局部比对，在锚点 ±6 线性邻域内按不同 old-only 70% 序列簇、稳定映射、S5 频率和保守替换筛选。

7 条假设：

- 单点：L18I、F13Y、T22Q、A134G、K130R、L140M
- 配对：F13Y+A134G（6 个 old-only 簇同时含两替换；S5 配对 0/1,537）

A134G/K130R 的稳定支持分别为 177/295、104/295，更像 old-only 分支与 KVP40 的谱系差异，不能当 19/18 次独立突变。H14/H17/R138 直接改造未提出；T135S 在 S5 中已占 19.15%；Q23W 非保守已排除；Y141I 因丢失芳香环/羟基排除，L140M 因此入选。

现有 NMNAT 候选池（15,360 条/模型）含多数单点替换，但 `F13Y+A134G` 为 0。现役 QVAE 历史代表 0/3 命中；QVAE 与 deterministic 证据重排均 3/3、6 个支持事件；权重重抽样参照 3/3、3 个事件；HMM 代理 0/3。B/C 相同说明当前池中差异主要来自证据选择，不是模型差异。

## 限制

- 30–35% identity 属 twilight zone；局部比对多最优、缺口和低同源对应仍不确定；稳定映射排除 >32 等分最优的记录，不等于结构或功能正确。
- `nearest_anchor_template_distance_A` 是最近锚点距离，不是候选替换残基的口袋几何；±6 是线性序列邻域，不是空间第二壳层。2QJO 是 NAD⁺ 产物参照，没有 ATP/NMN 三元结构。
- 70% 序列簇不是独立家族或独立进化事件；T5 `ref:` 前缀也不证明与 old 序列独立。
- 候选池诊断没有 held-out 证据，且同一 7 条假设用于重排和命中统计，不能称 precision 或独立验证；A 是不同目标的历史选择，D 是权重重抽样而非无信息对照。
- 配对缺失是现有池的供给限制，不补采、不以 0 命中推断模型能力；任何功能、稳定性或净积累结论仍需未来实验。

## 边界

来源人口和角色不变：training 1,644、validation 227、test 231；249/75/4 继续隔离。test 未读。本次没有新的训练、生成、下载或折叠；下一步实际计算应直接服务 `docs/RESEARCH_PLAN.md` 的候选设计与评价问题，而不是继续增加审计或关卡。
