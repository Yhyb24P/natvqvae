# NatV 交付入口

统一交接包：[natv_candidate_delivery/README.md](../data/results/evaluation/natv_candidate_delivery/README.md)

包内是 14 条唯一蛋白序列：共同 WT 1 条 + 计算设计 11 条（A1–A3 来自 QVAE、B1/B2 为机制假说设计、6 个 A+B 组合）+ 湿实验 ePCR 克隆 2 条（NatV10、NatV89）。完整序列、编辑、亲本关系和 SHA 在包内 `candidates.csv` / `candidates.fasta`；`coordinates/` 有每条 3 个样本、合计 42 个候选 PDB，`references/` 另存 ePCR 批次的 3 个 WT 参照。来源快照、结构图、位点表和 RMSD 表在 `evidence/` 与 `attachments/`；逐文件 SHA 与派生关系见 `manifest.json`。

两组共用 WT 参照，但不是同一来源：A+B 的 `source_run` 继承对应 A，不代表 A+B 由 QVAE 直接生成；NatV10/NatV89 是独立克隆，不是 B1/B2。12 条设计与 ePCR 克隆的模型批次、WT 参照和分母不同，不能拼成同一条件排名。

科学比较、读数含义、对照与待湿实验确认事项见 [`candidate_handoff.md`](../data/results/evaluation/natv_dual_domain_design/demo/candidate_handoff.md)。结构图是预测，不证明活性、稳定性或表达；NAD⁺ 净积累不是独立 vNMNAT；A570 历史终点吸光信号不能用于排序。本包不含 DNA/构建、表达纯化、检测 SOP、对照/重复/判定标准或功能结果。

交付时记录 Git commit；以该 commit 中的序列、身份文件和报告为准。
