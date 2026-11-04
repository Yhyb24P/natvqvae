# NatV 计算研究论文工作稿

目标期刊为 BMC Bioinformatics。`manuscript.tex` 和 `references.bib` 是英文稿件及文献；编译后的本地 PDF 为 `build/manuscript.pdf`。稿件已写入现有训练、评价与生成结果，也说明辅助片段重叠、开发验证参与模型选择以及 test 序列的历史使用。新的训练对照仍在执行，不能作为可直接投稿的版本。

在此目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex
```

本机使用 TeX Live 2026。`sn-jnl.cls`、`sn-vancouver-num.bst` 来自 Springer Nature 官方 LaTeX 模板 v3.1（2024 年 12 月），保持原样。下载地址：[官方模板 ZIP](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12)，ZIP SHA256 为 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`。稿件采用数字引文、页码和行号，参考[研究论文要求](https://link.springer.com/journal/12859/submission-guidelines/research-article)。

正文数值对应以下资料：

- 原迁移结果：[两臂报告](../../data/results/evaluation/natv_transfer/REPORT.md)、同目录 `paired_summary.json` 与 `generation_summary.json`。
- 实际训练记录：`data/results/training/aux_qvae_seed8601/` 和 `data/results/training/natv_transfer/`。
- 原始数据恢复与片段比较：[序列比较报告](../../data/results/preprocessing/pf01467_natv_overlap/REPORT.md)。
- 冻结 ESM-2 预测头与原模型：[比较结果](../../data/results/evaluation/esm2_natv_validation_comparison/REPORT.md)，图 1 来自同目录 `esm_comparison.pdf`。
- 新比较的输入：`data/results/preprocessing/natv_matched_fragment_pretraining/` 中的配对清单、掩码和序列；图 2 来自 `fragment_matching.pdf`。这是数据准备结果，不是新训练的预测表现。
- test 序列的历史用途：[逐项核查记录](../../data/results/preprocessing/natv_test_recovered/historical_use.json)。记录包含原始逐序列表、模型种子与执行时间；本轮未重新计算或汇总旧模型分数。

后续结果需要回答：在片段级序列关系得到处理后，外部片段预训练相对随机初始化、NatV 自身片段预训练是否仍有改善；附加任务模型相对同一冻结 ESM-2 预测头增加了多少预测能力。比较条件已在本轮评分前确定。231 条 test 序列在旧项目中已有评分，其中 39 条还影响过方法选择，因此本轮不能称为从未接触的独立测试。原有验证组删除诊断也不能代替重新训练。

投稿前还需完整提供权重、逐查询评分和恢复的测试资料，并由作者填写作者名单、单位、贡献、资助及利益冲突。已检查的仓库版本没有项目许可证文件，需由作者确定许可及使用限制；这里不代为授予权利。软件运行条件已依据 `pyproject.toml` 写入数据可用性声明。AI 协助的用途写入方法，具体说明仍需作者确认。没有对外提交稿件。
