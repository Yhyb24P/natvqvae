# KVP40 保守条件编辑比较

本轮完成 2,976 次条件生成，全部满足预定编辑约束，最低 WT 全局相似性为 90.03%。在固定共同位点下，主要不同组合指标大范围饱和，**没有得到 QVAE 优于 SE 的清晰证据**。这是旧 H2 权重在扩大编辑预算及短 InDel 背景下的生成研究，不是新模型训练结果。

## 方法与来源

执行源码提交为 `3e6d22e`；准确文件身份见 `run_identity.json`，本地环境见 `runtime_environment.json`。`plan.json`、`sampling_groups.json` 和 `inputs/source_checks.json` 分别固定方案、位点与抽样流、外部只读来源。此前慢速执行及本次批处理核对见 [execution_check.md](execution_check.md)。本次生成耗时 744.33 秒，其中累计检索 621.10 秒、编码 70.72 秒；逐组耗时包含同批其他组的工作，不能相加当作实际总耗时。

比较 train-only 检索列频、冻结 ESM-2、SE no_latent 和 categorical QVAE。后两者继承三个正式角色划分训练的配对种子；前两者的种子只代表采样重复。SE 为 840,369 参数，QVAE 为 906,289 参数，均额外读取同一冻结 ESM-2。容量仍有差异。检索严格限定 1,644 条正式训练序列；本轮不使用测试序列内容。

替换预算为 2/12/24/34，每预算四组共同位点，每组八次尝试；短 InDel 加替换的总预算为 6/18/30，每预算五个来源事件。每条轨迹五次更新，读取完整当前序列并重新遮蔽与检索；QVAE 从均匀类别先验抽取 z，在整条轨迹内固定。全部方法使用相同位点、抽样流及温度，所选替换位点均要求非 WT。保护位点、邻域选择和事件供体规则见 `plan.json`。

G1-b 的八个定义中，五个长度不超过 3 aa 的事件在现役训练集中重算支持：删除 WT 23–25、55、181，及 WT 30 后插入 1 aa、181 后插入 2 aa。声明比对下分别有 6/4/4/3/4 个训练组分支持，扰动比对下共同支持分别为 2/4/2/3/2。来源和实际载荷见 `inputs/current_event_support.json`。它们是经验事件加条件替换修复，不是已训练的自由变长生成器。

## 全部尝试的结果

| 方法 | 尝试数 | 不同序列数 | HMM 有序双域命中数 |
|---|---:|---:|---:|
| 列频 | 744 | 717 | 686 |
| ESM-2 | 744 | 732 | 698 |
| SE | 744 | 728 | 700 |
| QVAE | 744 | 729 | 697 |

替换共 1,536 次、长度均 341 aa；InDel 加替换共 1,440 次、长度 338–343 aa。全部尝试保留在分母，没有补采。各方法均无与正式训练或验证成员完全相同的输出。不同方法可生成相同序列，表中数字不能相加作为总库大小。

主要量是每组共同位点中满足约束的不同组合数除以八次尝试。替换预算 12/24/34 和 InDel 总预算 18/30，四种方法在所有组均达到 100%，因此无法据此区分模型收益。预算 2 的替换中，QVAE−SE 为 0，条件重采样 95% 区间为 −0.0833 至 0.0521；InDel 预算 6 为 +0.00833，区间 0 至 0.03333。这些区间只描述既有种子及共同位点/事件条件下的差异，不代表独立生物学重复。

按编辑位点数归一化的组内两两差异，在替换中 SE 为 0.7300、QVAE 为 0.7289；InDel 加替换为 0.7524、0.7588。变化不足以支持潜变量的稳定收益。分预算、种子数据见 `evaluation/per_seed_summary.csv`，图见 [figures/conservative_comparison.png](figures/conservative_comparison.png)。

93 次关闭潜变量诊断的最大 logits 差为 0.3506；21 个既定上下文对照中，ESM-2、SE、QVAE 的平均目标分布 L1 响应分别为 0.06460、0.07007、0.06849。模型确实读取了已有编辑的上下文，QVAE 输出也依赖 z；这些是模型内响应诊断，各方法母序列不同，不能据此排序生成质量或宣称功能性协同。

HMM 注释使用独立 E-value ≤ 0.001 及双域顺序，覆盖度单独报告。WT 本身的 NMNAT profile 覆盖度仅 0.3985（参照见 `inputs/WT_HMM_reference.json`），因此未沿用旧 0.5 覆盖度作为有效性门槛。`evaluation/results.json` 中的 `HMM_qualified` 仅指上述命中，不是功能合格。HMM 与训练家族证据相关；催化活性、InDel 功能耐受性和 NAD⁺ 净积累均为 UNKNOWN。

## 结果与复现

完整序列见 `evaluation/candidates.fasta`，逐条数据见 `evaluation/candidate_evaluation.csv`。14,880 步轨迹和 2,976 份显式编辑脚本分别保存为 `trajectory.jsonl.gz`、`edit_scripts.jsonl.gz`；导出校验见 `export_checks.json`。`audit_results.py` 核对每步上下文、遮蔽、保护锚点、实际替换位点、全局距离及逐条精确复原，结果见 `result_checks.json`。本地 MMseqs 中间文件保留于 `generation/`，其来源命令与检索记录随实际运行保存。

在具有已固定源码、权重及环境的主机上，将本目录脚本、`inputs/`、`plan.json` 复制到本仓库 `experiments/` 下的干净同级运行目录，可依次执行 `run.py`、`evaluate.py`、`export_records.py` 和 `audit_results.py`；脚本按自身位置保存结果。旧结果目录不覆盖。执行时使用 `PYTHONDONTWRITEBYTECODE=1`，本次解释器为 `/home/yhshy/miniconda3/envs/nad_data/bin/python`。上下文附加诊断由 `context_diagnostic_protocol.json` 和 `context_diagnostic.py` 定义，需在运行前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

下一步应选择不同设计假设及匹配方法/预算的代表序列，开展候选自身全长结构与催化环境评价；模型研究仍需能区分联合条件分布质量的比较，不能靠继续增加高预算唯一率的采样来证明 QVAE 优势。
