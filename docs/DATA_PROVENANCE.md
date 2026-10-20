# 数据与模型来源

## 序列人口与训练边界

旧项目冻结 NatV 双域人口为 2,430 条。按 `training_role_v2`，正式训练 1,644 条（47 个 70% identity / 80% coverage 组分），验证 227、测试 231、注释冲突隔离 249、旧封存 75、排除 4；组分不等于 47 条独立进化谱系。`inputs/manifest.json` 的发现人口 FASTA 不等于训练资格，`split_v2` 和 `core_training_eligible_v2` 均不能替代 `training_role_v2`。具体文件身份与计数见 `discovery/natv_data_readiness/data_checks.json` 和 `discovery/natv_pairing_audit/audit_summary.json`。

`discovery/natv_data_readiness/` 有正式训练/验证 FASTA、逐条角色与 CDD 位置；检索严格使用训练人口。现役 90%–100% 相似性是**生成候选对 WT 的约束**，并不筛除天然训练家族中的远缘序列。外部新发现序列仍由 `discovery/natv_pairing_audit/dataset_update_decision.md` 裁决，不能直接混入既定训练、验证或测试。

家族 MSA 某次结构对应分析为 1,638 条，较 1,644 的差异来自其额外筛选；不可混称。旧逐查询 ESM-2 和演化特征来自掩码输入规划，不是新任务通用张量。训练来源在参考查询前要核对角色，历史的公开 WT 和同源实验模板是参照，不是可以偷偷补入训练的样本。

## 模型与对齐资料

`nad_qvae` 作为只读研究档案，源码、权重、配置与分割差异入口为 `discovery/natv_model_inheritance/model_reuse_decision.md` 和 `source_checks.json`。旧 `SequenceGenerator` 在 WT ESM-2 条件下直接解码 341 对齐列，输出 20 种氨基酸及 gap；不表示新插入残基。旧角色与现役划分不相容，历史 checkpoint 不可直接当成新划分的独立验证结果。

现役 `experiments/natv_wt_columns/` 重新按正式训练/验证人口训练了 categorical VAE 和确定性对照，其 `data/aligned_samples.jsonl` 保留每条天然序列到 WT 的列映射及插入位置；训练的对齐列丢失了插入残基的生成目标。已保存的模型身份、原始提案和完整负结果归该目录的 `REPORT.md`，不能把本次结果再解释为已训练的保守编辑模型。

旧 `design/g0_benchmark.py`、`formal_generation.py`、`se_full.py` 提供冻结 ESM-2/SE 当前序列掩码更新及 train-only 检索能力；旧 `vnext_edits.py` 支持带 WT 锚点的 `KEEP/SUB/INSERT/DELETE` 编辑脚本、准确复原及来自训练人口的 InDel 分布。对应的 G1-b 结果位于旧档案 `results/natv_g1_event_qualification_v1/event_qualification.json`，属于可疑变长位点筛选而非已训练插入/删除生成器。复用这些算法时须核验其实际源码身份和原数据角色；历史支持数不是 InDel 安全性、发生率或酶活标签。

## 结构和其他来源

KVP40 序列身份、CDD 划分及当前历史替换位置约定分别见 `contracts/regions.json`；实验模板、构建体标签、观测残基和 12 个映射锚点见 [NATV_PAIRING_AUDIT.md](NATV_PAIRING_AUDIT.md)。家族同源模板不等于 KVP40 实验结构。预测结构与实验结构独立标记；结构数据输入先校验文件来源和实际残基对应。

源文件通过 `inputs/manifest.json` 的 SHA256 定位，`NATV_SOURCE_ROOT` 或 `--root` 指向旧仓库。大文件及 checkpoint 位于研究主机，运行报告记录使用的文件及哈希；符号链接不使来源在系统层自动只读。数据准备和运行产生的文件应写入本仓库相应的运行目录，不写回旧档案。
