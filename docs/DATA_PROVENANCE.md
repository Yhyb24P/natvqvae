# 数据与模型来源

## 序列人口与训练边界

旧项目冻结 NatV 双域人口为 2,430 条。按 `training_role_v2`，正式训练 1,644 条（47 个 70% identity / 80% coverage 组分），验证 227、测试 231、注释冲突隔离 249、旧封存 75、排除 4；组分不等于 47 条独立进化谱系。`data/manifest.json` 的发现人口 FASTA 不等于训练资格，`split_v2` 和 `core_training_eligible_v2` 均不能替代 `training_role_v2`。具体文件身份与计数见 `data/processed/data_checks.json` 和 `data/processed/audit_summary.json`。

`data/processed/` 有正式训练/验证 FASTA、逐条角色与 CDD 位置；检索严格使用训练人口。现役 90%–100% 相似性是**生成候选对 WT 的约束**，并不筛除天然训练家族中的远缘序列。外部新发现序列仍由 `discovery/natv_pairing_audit/dataset_update_decision.md` 裁决，不能直接混入既定训练、验证或测试。

家族 MSA 某次结构对应分析为 1,638 条，较 1,644 的差异来自其额外筛选；不可混称。旧逐查询 ESM-2 和演化特征来自掩码输入规划，不是新任务通用张量。训练来源在参考查询前要核对角色，历史的公开 WT 和同源实验模板是参照，不是可以偷偷补入训练的样本。

## 模型与对齐资料

`nad_qvae` 作为只读研究档案，源码、权重、配置与分割差异入口为 `discovery/natv_model_inheritance/model_reuse_decision.md` 和 `discovery/natv_model_inheritance/source_checks.json`。模型来源、权重身份与继承判断保留。

已退出的生成模型与输入：

- WT 对齐列：六份 776,981 参数权重与派生列映射已删除；freeze SHA `5ce6e3b9…`。
- H2 条件编辑与首轮保守生成：H2 freeze `a69e8470…`、旧 `SequenceGenerator` revision `4277b1d…` 与文件 SHA `8c30bba0…` 的身份保留在 `discovery/natv_model_inheritance/source_checks.json`。

运行证据、提交 SHA 与限制见[退役生成实验证据](../discovery/natv_generation_evidence_decision.md)。

旧 `design/g0_benchmark.py`、`formal_generation.py`、`se_full.py` 记录冻结 ESM-2/SE 当前序列掩码更新及 train-only 检索的实现；旧 `vnext_edits.py` 记录带 WT 锚点的 `KEEP/SUB/INSERT/DELETE` 编辑脚本及来自训练人口的 InDel 分布。对应的 G1-b 结果位于旧档案 `results/natv_g1_event_qualification_v1/event_qualification.json`，属于有一定演化证据的短事件筛选而非已训练插入/删除生成器。这些现在是历史实现来源，不再由本仓库直接调用；历史支持数不是 InDel 安全性、发生率或酶活标签。

## 结构和其他来源

KVP40 序列身份、CDD 划分及当前历史替换位置约定分别见 `contracts/regions.json`；实验模板、构建体标签、观测残基和 12 个映射锚点见 [NATV_PAIRING_AUDIT.md](NATV_PAIRING_AUDIT.md)。家族同源模板不等于 KVP40 实验结构。预测结构与实验结构独立标记；结构数据输入先校验文件来源和实际残基对应。

`data/raw/source/` 是指向只读 `nad_qvae` 的本地链接；`data/raw/structures/` 与 `data/raw/metadata/` 保存 2QJO/2QJT/2R5W 实验结构与模板元数据。`data/processed/` 保存正式群体、划分、序列—结构映射和原始核验记录；`data/interim/`、`data/models/`、`data/results/` 存放缓存、权重与运行结果。

外部检索资产的保留、退役与校验快照见 `discovery/natv_pairing_audit/README.md` 与 `discovery/natv_pairing_audit/dataset_update_decision.md`。

源文件通过 `data/manifest.json` 的 SHA256 定位，`NATV_SOURCE_ROOT` 或 `--root` 指向旧仓库。大文件及 checkpoint 位于研究主机；历史运行的文件与哈希由对应 decision 或提交追溯。数据准备和运行产生的文件应写入相应的 `data/` 子目录或 `experiments/`，不写回旧档案。

`experiments/kvp40_conditional_generation/` 从随机 NatV 参数重新训练 QVAE 与等有效参数量的确定性对照，继承现役正式训练/验证角色。两者共用本地冻结 ESM-2 的遮蔽输入表示；不读取历史 NatV checkpoint。具体模型、掩码面板及新权重身份归该运行的方案与来源记录。
