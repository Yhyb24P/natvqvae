# 数据与模型来源

## 序列群体与划分

旧项目的正式 NatV 双结构域训练群体含 1,644 条 core-train 序列，分为 47 个冗余组分（70% identity / 80% coverage）。组分用于控制序列冗余，不代表独立祖先或实测催化表型。部分结构分析使用的家族 MSA 经对应关系和缺口筛选后有 1,638 条，不能把缺少的 6 条计入 MSA。

`inputs/manifest.json` 固定的是旧工作目录中的 2,430 条发现群体 FASTA。`training_role_v2` 将其分为 1,644 条训练、227 条验证、231 条测试、249 条注释冲突隔离、75 条旧封存记录和 4 条排除记录。`split_v2` 与角色表含义不同，不能仅凭 FASTA 或 split 标签确定训练资格；详细计数见 `discovery/natv_pairing_audit/audit_summary.json`。这份 FASTA 不等于正式训练集，也不等于已验证功能的 NatV 群体。

`inputs/manifest.json` 已分别登记 `family_roles`、`core_split`、`allowed_samples`、`training_samples` 与 `train_sequences`。`core_training_eligible_v2` 表示核心群体资格，不能单独作为训练选择条件；正式训练还须满足 `training_role_v2=training`，验证和测试仍按角色隔离。

`discovery/natv_data_readiness/` 提供按既有角色导出的训练与验证 FASTA、含序列和 CDD 坐标的 `samples.csv`，以及不含序列的全群体 `roles.csv`；检索使用同一份训练 FASTA。任务与模型无关的序列输入可直接读取，掩码、张量形状和采样方式仍由具体生成任务决定。本轮 WT 对齐列任务的数据位于 `experiments/natv_wt_columns/data/`：`aligned_samples.jsonl` 保留正式角色、源序列位置对应与插入索引，`training_weights.npy` 按现役训练组分等权，`wt_embeddings.pt` 只含冻结的公开 WT 表示；来源与派生文件校验值见同目录 `input_checks.json`。该次核查结果与复现方法见 `discovery/natv_data_readiness/data_readiness_decision.md`。

其他 NAD 代谢酶（尤其 NadE）不自动属于全长 NatV 同源训练样本。通用蛋白语料可提供固定的预训练权重，或用于另有说明的迁移学习实验；两者均需明确序列选择、架构归属和留出同源物来源。ESM-2 预训练可能包含 NatV；在已核对数据中未发现重叠，不能证明预训练模型未接触该家族。

外部待研究序列及是否扩充数据的判断见 `discovery/natv_pairing_audit/dataset_update_decision.md`。

## 参考序列与结构输入

KVP40 NatV 参考序列及其 SHA256 见 `contracts/regions.json`。CDD 窗口 NMNAT 4–181、连接区 182–203、Nudix 204–338 是注释坐标；其适用范围见 `docs/RESEARCH_PLAN.md`。模板、构建体、天然序列、坐标编号、关键残基与配体状态的对应入口是 `docs/NATV_PAIRING_AUDIT.md`。

旧研究包含 2QJO、2QJT、2R5W 结构及化学口袋资源，部分资产可能仅存于原计算主机。`discovery/natv_pairing_audit/structural_interpretation_basis.json` 汇总已核对的结构对应关系，`formal_input_manifest=false`；实际运行读取时，由该次运行记录路径和校验值。候选运行如何使用它见 `experiments/natv_candidate_sequences/REPORT.md`。

## 目标合同与模型继承

当前合同为 `contracts/objective.yaml`、`contracts/evidence_ladder.yaml` 和 `contracts/regions.json`。旧仓库的 `config/natv/target_contract.yaml` 仅是历史构建体元数据，不是此处固定的输入；其中 `apparent_kcat_over_Km_NMN` 不定义当前的 NAD⁺ 净积累目标。

`nad_qvae` 是外部研究档案。旧 checkpoint 未作为现役模型权重导入；本轮重新训练的权重及其来源见 `experiments/natv_wt_columns/REPORT.md`。继承权重须核对架构源码、训练合同、数据划分、checkpoint SHA256、输出字母表和实际采样方式，不能根据文件名中的 QVAE 判定模型身份。

旧 `design/sequence_generator.py` 与配套 `sequence_corpus.py`、`sequence_experiment.py` 属于 `natv_sequence_design_dev_v1`；其冻结源码、WT 条件与 checkpoint 的本地身份，以及旧角色与现役划分的对应，见 `discovery/natv_model_inheritance/model_reuse_decision.md` 和同目录 `source_checks.json`。实现可继承，数据角色与权重资格须分别判断。

已知历史来源包括 H2 categorical / continuous / no-latent（旧仓库 `experiments/run_reports/natv_h2_architecture_generation_v1_attempt1/PHASE_SUMMARY.md`）、H1/H2 SE / no-latent checkpoint、早期 `qvae_v9_seed2028` 和 H3/G0 chemical adapter。它们的研究判断见 `docs/RESEARCH_PLAN.md`；具体运行使用的权重见各自报告。

`masked_input_plan` 绑定旧任务的查询、掩码、检索 ID 与样本校验值；`masked_contexts` 和 `evolution_features` 分别列出逐查询表示与演化特征的路径和 SHA256。大文件留在只读来源，通过清单定位。它们是历史掩码任务的输入，不是任意全长生成模型的通用特征。

权重的训练任务和使用方式须区分全长先验生成、WT-relative masked editing 与残基恢复验证，不能相互继承能力结论。

## 文件边界

`NATV_SOURCE_ROOT` 或 `--root` 指定旧仓库路径。`inputs/link_inputs.py` 核对校验值并建立本地链接；软链接本身不提供只读权限。贡献与写入边界见 `AGENTS.md`。

源文件在本机是否可用由 README 中的本地检查确定，GitHub CI 不能验证所有者的本地研究数据。大文件和临时输出保存在 Git 外，运行记录须提供校验值和可访问的存储位置。
