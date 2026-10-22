# NatV NMNAT 演化设计证据

目标：从已有审计来源中识别可与 KVP40 NMNAT 对照的天然变化，形成少量、可追溯的设计假设；不把远缘序列直接扩成训练人口，也不把序列/注释证据当功能。

## 来源

| 来源 | 记录/簇 | 用途 |
|---|---|---|
| S1 complete+EC | 556/44 | 238 EC NMNAT-only + 318 EC dual |
| S2 full-context T80 | 63/19 | 与 S1 有 6 条重叠，不重复计数 |
| S2-only | 57/18 | 全长 text/unknown 证据 |
| S4 reviewed positives | 61/7 | 仅交叉核对；分类器未验证 |
| S5 NatV | 1,871 | 1,644 training + 227 validation；不读 test |
| S3 EC-dual | 318/22 | 注释背景；不强行对齐 |

这些来源不是训练、验证或测试人口，不改变 1,644/227/231；249/75/4 继续隔离。

## 方法

- 每个 70% 序列簇选代表；优先 tier、target coverage、域长度、S4 evidence tier。
- 完整蛋白对 WT NMNAT 4–181 做 BLOSUM62 局部比对；记录锚点 ±6 线性邻域、稳定映射、不同 old-only 序列簇支持、S5 频率和保守替换。
- 等分最优数 >32 的记录按不稳定排除；不抽样前 32 条。
- 候选限于 NMNAT 4–181、341 aa、单/双替换、无 indel；不碰 Nudix/linker/末端。
- 只使用最近锚点距离，不把 ±6 当空间第二壳层，也不声称逐残基口袋几何。

## 现有候选池诊断

- 使用 `data/results/natv_conditional_qvae/raw/annotated_candidates.parquet` 的 `region=NMNAT`，15,360 条/模型。
- 3 条/模型，按 `sequence_sha256` 去重；与现役 QVAE 历史代表、deterministic 证据重排、权重重抽样参照和 HMM 代理比较。
- 同一 7 条假设用于重排和命中统计，且没有提前留出的簇，因此只能报告描述性 evidence match rate，不能称 held-out precision 或独立验证。
- 不生成新序列、不训练、不下载、不折叠、不读 test。

## 产物

- 详细证据与假设：`HYPOTHESES.md`
- 候选池诊断：`B1_RERANK.md`
- 归属报告：`REPORT.md`
- 输入/运行身份：`inputs.json`、`RUN_PROVENANCE.json`、`OUTPUTS.sha256`
- 原始表：`data/results/natv_evolution_informed_nmnat/raw/phase_a/`、`raw/phase_b1/`
