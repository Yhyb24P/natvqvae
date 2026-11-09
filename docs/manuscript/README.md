# NatV 计算研究论文工作稿

目标期刊为 BMC Bioinformatics。`manuscript.tex` 和 `references.bib` 是英文稿件及文献；编译后的本地 PDF 为 `build/manuscript.pdf`。2026-11-10 修订稿以六格比较、冻结ESM对照、域相似度分层及预测头互补为主线，原迁移与三条件开发结果移入附录。新增实验复用既有确定性权重，验证集拟合温度与混合参数后冻结，完成一次有历史用途的保留集评价；不据此声称外部预训练、共享潜变量或酶学优势。生成、训练曲线和MC精度结果保留，作者资料与完整公开归档仍未完成，不能作为可直接投稿的版本。

在此目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex
```

本机使用 TeX Live 2026。`sn-jnl.cls`、`sn-vancouver-num.bst` 来自 Springer Nature 官方 LaTeX 模板 v3.1（2024 年 12 月），保持原样。下载地址：[官方模板 ZIP](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12)，ZIP SHA256 为 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`。稿件采用数字引文、页码和行号，参考[研究论文要求](https://link.springer.com/journal/12859/submission-guidelines/research-article)。

正文数值对应以下资料：

- 原迁移结果：[两臂报告](../../data/results/evaluation/natv_transfer/REPORT.md)、同目录 `paired_summary.json` 与 `generation_summary.json`。
- 实际训练记录：`data/results/training/aux_qvae_seed8601/` 和 `data/results/training/natv_transfer/`。
- 原始数据恢复与片段比较：[序列比较报告](../../data/results/preprocessing/pf01467_natv_overlap/REPORT.md)。
- 数据与模型示意图：[SVG 源图](model_design.svg)及 LaTeX 使用的 [PDF](model_design.pdf)，依据 `src/natvqvae/model.py` 的上下文、先验、后验和解码路径；用 fireworks-tech-graph 生成并检查几何，再以 CairoSVG 导出。图中表示数据和两类模型的实际比较设计。
- 冻结 ESM-2 预测头与原模型：[比较结果](../../data/results/evaluation/esm2_natv_validation_comparison/REPORT.md)，相应图来自同目录 `esm_comparison.pdf`。
- 新比较的输入：`data/results/preprocessing/natv_matched_fragment_pretraining/` 中的配对清单、掩码和序列；片段匹配图来自 `fragment_matching.pdf`。这是数据准备结果，不是新训练的预测表现。
- 清洗后三条件 QVAE 的验证分数、实际训练量和分层分数：[分层比较](../../data/results/evaluation/natv_fragment_transfer/validation_stratified_comparison/REPORT.md)；MC 标准误及64/128/256样本敏感性：[分析报告](../../data/results/evaluation/natv_fragment_transfer/validation_latent_sample_sensitivity/REPORT.md)。
- 六格完整预测：[保留集报告](../../data/results/evaluation/natv_six_cell/test/REPORT.md)、[验证报告](../../data/results/evaluation/natv_six_cell/validation/REPORT.md)与[保留集MC分析](../../data/results/evaluation/natv_six_cell/test_latent_precision/REPORT.md)；正文预测图来自 `test/prediction_effects.pdf`，绘图身份另存于 `figure_identity.json`。
- 六格固定背景生成：[生成报告](../../data/results/evaluation/natv_six_cell_generation/REPORT.md)、`component_effects.csv`、`hamming_paired.csv`、`candidate_distances.csv`；正文使用同目录 `generation_effects.pdf`。
- 预测头互补与温度对照：[验证报告](../../data/results/evaluation/natv_head_calibration/validation_protocol_checked/REPORT.md)、[冻结参数保留集报告](../../data/results/evaluation/natv_head_calibration/test/REPORT.md)；对应summary与group_scores记录全部对比、逐seed和逐组结果。初始QVAE/确定性目标概率混合见 `natv_head_mixture/validation_protocol_checked/`，没有追加QVAE温度导出或训练。
- 历史分组重建：[图重建记录](../../data/results/preprocessing/natv_split_relations/historical_components.json)；全长与域最近邻：[搜索汇总](../../data/results/preprocessing/natv_sequence_neighbors/summary.json)。
- test 序列的历史用途：[逐项核查记录](../../data/results/preprocessing/natv_test_recovered/historical_use.json)。记录包含原始逐序列表、模型种子与执行时间；本轮未重新计算或汇总旧模型分数。
- 本轮入口冒烟评分的执行记录：[test_smoke_execution.json](../../data/results/evaluation/natv_fragment_transfer/test_smoke_execution.json)。该次逐任务输出已删除，记录区分可核验输入与执行者报告；其分数不用于研究结论或选模。

六格比较回答：在片段级序列关系得到处理后，外部片段预训练相对随机初始化、NatV自身片段预训练是否仍有改善；附加任务模型相对冻结ESM预测头增加了多少预测能力。预测头混合及温度对照是在看到六格结果后设计的后续实验，其T与α仅用验证集拟合，随后冻结评价。231条test序列在旧项目中已有评分，其中39条影响过方法选择；新实验同样不能称为从未接触的独立测试。原有验证组删除诊断也不能代替重新训练。

投稿前还需完整提供权重、逐查询评分和恢复的测试资料，并由作者填写作者名单、单位、贡献、资助及利益冲突。已检查的仓库版本没有项目许可证文件，需由作者确定许可及使用限制；这里不代为授予权利。软件运行条件已依据 `pyproject.toml` 写入数据可用性声明。AI 协助的用途写入方法，具体说明仍需作者确认。没有对外提交稿件。


## 复算与本地材料

在仓库根目录安装 `pip install -e '.[dev,data]'` 后，使用统一的 `python -m natvqvae.cli`。`config/sources.yaml` 指向冻结 ESM-2 权重；权重文件必须与每次评分的 `esm_files_sha256` 一致。CUDA/bfloat16 是正式评分的实际条件，CPU运行不保证数值逐位一致。

- 两类模型的训练命令及实际运行记录在 `data/results/training/natv_fragment_transfer/` 的 `pretraining_execution.json`、`downstream_execution.json`，以及 `natv_fragment_transfer_deterministic/execution_commands.sh`。每份训练摘要记录选定权重和步数，逐更新 `metrics.jsonl` 支持目标残基暴露量复算。
- 批量评分、六格比较、MC合并与生成使用 `data/results/evaluation/natv_formal_comparison/commands.txt` 索引的正式CLI命令；同目录运行记录保留实际执行及失败后的修正。命令中的本机Python绝对路径在其他环境需替换，原记录不改写成事后执行历史。
- 六格报告由共同逐组分数产生；`paired_group_differences.csv` 保留每项对比，`summary.json` 记录参数和输入身份。`six-cell-plot --comparison <目录>` 仅重新导出图表，不重新评分。
- 原始逐任务评分、全部未筛选生成和训练权重本地保留；Git只选择纳入正式源码、配置、必要输入、报告和适当规模的汇总结果。公开仓库不等于完整权重和数据已经公开存档，投稿前还需完成持久存储与作者声明。

所有比较输出应使用新的空目录，避免覆盖已有正式记录。复算统计无需重新训练；只有需要复现训练随机性或改变方法时才重训。

新增预测头实验的执行入口见 `data/results/evaluation/natv_head_calibration/commands.txt`。本地补充包 `data/results/evaluation/natv_head_calibration/reproduction_data.tar.zst` 包含验证与保留集的完整20AA概率、任务清单、导出身份、冻结校准参数、逐组结果及本轮配置；逐文件清单与包校验值在同目录。它补充而不替代下述六格数据包：共享输入、三个选定确定性权重以及数值一致性检查所需的原目标概率仍从六格包获取。CPU复算可直接使用已存概率，不需要重新执行ESM前向计算；只有重做概率导出时才需要相应GPU环境和冻结ESM权重。输入、模型与实现绑定及有限温度网格的解释限制见报告；补充包仍是本地材料，未公开存档。

校准源码与冻结结果版本为 `d359cc28f6d5ac019ea641a59674ead0cb72bb90`，六格源码为 `a971d6e5646aff6f11fa43ecc1be1e21b75a3777`。补充包23个成员已逐文件校验，大小71,055,029 bytes，SHA256为 `b128d83f7dd28ab760b80cc9806d2542ee1aa743a1f54e4afec44b2d62cc746f`。校准包identity明确区分实际源码提交（source_commit）与执行起点（execution_base_commit）。六格原包及历史identity保持原样；其中base_commit只记录准备时checkout，实际源码应以单独发布的[源码身份修正](../../data/results/evaluation/natv_formal_comparison/reproduction_source_identity.json)为准，该文件逐项列出已与a971d6e核对的源码哈希。修正文件使用独立名称，解压旧包不会覆盖它；校准补充包已包含自身修正后的identity。

本轮本地数据包为 `data/results/evaluation/natv_formal_comparison/reproduction_data.tar.zst`（约1.09 GB，解压约5.29 GB），包含6,510个数据文件：18份选定下游权重、12次辅助预训练的初始/最终权重、训练记录、恢复与配对输入、最近邻原始比对、逐任务预测、全部466,560次生成及比较输出。逐文件路径/大小/SHA256见同目录 `reproduction_manifest.csv`，压缩包校验值见 `.sha256` 文件；代码与配置使用对应Git版本，源码身份也记录在 `reproduction_identity.json`。包未对外存档，不能用本地路径代替持久标识。

在仓库根目录解压 `tar --zstd -xf data/results/evaluation/natv_formal_comparison/reproduction_data.tar.zst --transform='s,^historical_inputs/,data/raw/historical_split_inputs/,'` 可还原相对数据路径。包内 `historical_inputs/` 保存重建历史分组所需的三个原始文件，上述解压命令将其放入 `data/raw/historical_split_inputs/`；复算该检查时将 `config/natv_split_relations.yaml` 的历史绝对路径改为对应归档路径；原路径及哈希在清单中保留。冻结ESM-2不重复打包，须另外提供与 `reproduction_identity.json` 中五项哈希一致的文件。特征缓存和未选定下游权重不在包内；它们不是复算本轮已选模型统计的必要输入。
