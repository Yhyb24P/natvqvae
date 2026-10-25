# NatV 交付说明：对象、文件用途与阅读顺序

这是交付导航，不替代 [`candidate_handoff.md`](../data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md) 中的科学比较和实验待确认事项。第一次接手时先读本页，再按用途打开对应文件。

## 交付对象

| 组 | 对象 | 正式序列入口 | 结构展示入口 | 状态 |
|---|---|---|---|---|
| 计算设计候选 | 12 条：WT、A1/A2/A3、B1/B2、6 个 A+B；其中 11 条为非 WT 设计 | [`candidates.csv`](../data/results/generation/natv_dual_domain_design/candidates.csv)、[`candidates.fasta`](../data/results/generation/natv_dual_domain_design/candidates.fasta) | [`captions.md`](../data/results/evaluation/natv_dual_domain_design/demo/captions.md)、[`wt_overview.png`](../data/results/evaluation/natv_dual_domain_design/demo/wt_overview.png)、[`demo.mp4`](../data/results/evaluation/natv_dual_domain_design/demo/demo.mp4) | WT 为基准；A1–A3 来自 QVAE 生成，B1/B2 为机制假说设计，A+B 为组合；当前交付没有确认的设计构建或功能数据 |
| 湿实验 ePCR 克隆 | NatV10、NatV89 | [`variants.csv`](../data/results/evaluation/kvp40_epcr_full_length_structures/variants.csv)、[`sanger_verification.json`](../data/results/evaluation/kvp40_epcr_full_length_structures/sanger_verification.json)、[`build.json`](../data/results/evaluation/kvp40_epcr_full_length_structures/build.json) | [`captions.md`](../data/results/evaluation/kvp40_epcr_clone_demo/captions.md)、[`clone_overview.png`](../data/results/evaluation/kvp40_epcr_clone_demo/clone_overview.png)、[`clone_overlay.png`](../data/results/evaluation/kvp40_epcr_clone_demo/clone_overlay.png) | 已 Sanger 验证的全长克隆；历史 A570 汇总不能排序 |

两组共用 KVP40 WT 作为参照：12 条设计集合包含 WT 和 11 条非 WT 设计；NatV10/NatV89 是独立实验克隆，不是 B1/B2，也不能把克隆差异归因于单个设计位点。

## 按用途查找

| 用途 | 文件 | 说明 |
|---|---|---|
| 候选身份与亲本关系 | [`candidates.csv`](../data/results/generation/natv_dual_domain_design/candidates.csv)、[`candidates.fasta`](../data/results/generation/natv_dual_domain_design/candidates.fasta) | 12 条设计候选的正式序列、编辑、来源 ID 和 SHA；不包含 NatV10/NatV89 |
| 实验克隆身份 | [`variants.csv`](../data/results/evaluation/kvp40_epcr_full_length_structures/variants.csv)、[`sanger_verification.json`](../data/results/evaluation/kvp40_epcr_full_length_structures/sanger_verification.json)、[`build.json`](../data/results/evaluation/kvp40_epcr_full_length_structures/build.json) | WT 及 NatV10/NatV89 的序列、克隆的 Sanger 依据和实际编辑；不记录 A/B/A+B 设计变体 |
| 交接问题与边界 | [`candidate_handoff.md`](../data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md) | 两组对象的比较问题、读数含义、对照要求和待湿实验确认事项 |
| 12 条设计结构附件 | [`captions.md`](../data/results/evaluation/natv_dual_domain_design/demo/captions.md) 及其图片/视频、[`candidate_structure_summary.csv`](../data/results/evaluation/natv_dual_domain_design/demo/candidate_structure_summary.csv)、[`comparison_summary.csv`](../data/results/evaluation/natv_dual_domain_design/demo/comparison_summary.csv) | 预测结构展示和比较表；不证明活性、稳定性或候选排序 |
| 2 条克隆结构附件 | [`captions.md`](../data/results/evaluation/kvp40_epcr_clone_demo/captions.md)、[`clone_overview.png`](../data/results/evaluation/kvp40_epcr_clone_demo/clone_overview.png)、[`clone_overlay.png`](../data/results/evaluation/kvp40_epcr_clone_demo/clone_overlay.png)、[`mutation_sites.csv`](../data/results/evaluation/kvp40_epcr_clone_demo/mutation_sites.csv) | WT/NatV10/NatV89 的独立静图；NMNAT 4–181 配准会把 Nudix 相对位移带入 |
| 研究证据与数值 | [`设计报告`](../data/results/evaluation/natv_dual_domain_design/REPORT.md)、[`全长比较`](../data/results/evaluation/kvp40_epcr_full_length_structures/REPORT.md)、[`NMNAT 环境`](../data/results/evaluation/kvp40_epcr_nmnat_environment/REPORT.md)、[`RESULTS.md`](RESULTS.md) | 方法、数值和限制；湿实验团队不必先逐篇读完 |
| 复算资料 | [`README.md`](../README.md)、[`structure demo config`](../config/structure_demo.yaml)、[`clone demo config`](../config/epcr_clone_demo.yaml)、两目录的 `render_input.json`/`render_identity.json` | 代码、配置、渲染输入/输出身份；用于计算侧复核，不是新的序列版本或实验结果 |
| 研究背景 | [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | 科学目标和研究路线，不是交付清单 |

## 核心边界

结构图来自 Boltz-2 预测，不是实验结构；pLDDT/pTM、Cα RMSD 和局部差异都不能证明活性、稳定性或表达。完整 NatV 的 NAD⁺ 净积累不是独立 vNMNAT 活性；A570 的记录范围、构建/检测/对照等具体限制见 [`candidate_handoff.md`](../data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md)。

## 建议阅读顺序

1. 本页：确认对象和文件用途。
2. [`candidate_handoff.md`](../data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md)：交接问题、读数含义和待确认事项。
3. 序列身份：`candidates.csv`/`candidates.fasta` 与 `variants.csv`/Sanger 记录。
4. 结构附件：两个目录的 `captions.md`，按需要查看 PNG/MP4。
5. 研究证据：三份 `REPORT.md` 和 `RESULTS.md`，按需要查阅数值和方法。

交付时记录 Git commit；以该 commit 中的序列文件、身份文件和报告为准。
