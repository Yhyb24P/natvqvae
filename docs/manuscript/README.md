# NatV 计算研究论文工作稿

目标期刊为 BMC Bioinformatics。`manuscript.tex` 和 `references.bib` 是英文稿件及文献；编译后的本地 PDF 为 `build/manuscript.pdf`。2026-11-10 修订稿以六格比较、冻结ESM对照、域相似度分层及预测头互补为主线，原迁移与三条件开发结果移入附录。稿件以原始统一50:50概率混合呈现新阳性结果，温度/面板参数作为补充对照，并补充线性概率池、温度缩放、Tranception及RAG-ESM文献。已有概率的位点分解区分排名变化与top-1不变时的概率改善。新增实验复用既有确定性权重，验证集拟合温度与混合参数后冻结，完成一次有历史用途的保留集评价；不据此声称外部预训练、共享潜变量或酶学优势。生成、训练曲线和MC精度结果保留，两个研究数据包现已提供GitHub Release公开下载；作者资料与声明仍未完成，不能作为可直接投稿的版本。

在此目录执行：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex
```

本机使用 TeX Live 2026。`sn-jnl.cls`、`sn-vancouver-num.bst` 来自 Springer Nature 官方 LaTeX 模板 v3.1（2024 年 12 月），保持原样。下载地址：[官方模板 ZIP](https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12)，ZIP SHA256 为 `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`。稿件采用数字引文、页码和行号，参考[研究论文要求](https://link.springer.com/journal/12859/submission-guidelines/research-article)。

正文数值对应以下资料：

- 原迁移结果：[两臂报告](../../data/results/evaluation/natv_transfer/REPORT.md)、同目录 `paired_summary.json` 与 `generation_summary.json`。
- 实际训练记录：`data/results/training/aux_qvae_seed8601/` 和 `data/results/training/natv_transfer/`。
- 原始数据恢复与片段比较：[序列比较报告](../../data/results/preprocessing/pf01467_natv_overlap/REPORT.md)。
- 现役研究设计与模型路径：[研究设计SVG](../../data/results/evaluation/natv_prediction_figures/study_design.svg)、[QVAE路径SVG](../../data/results/evaluation/natv_prediction_figures/qvae_paths.svg)，对应PDF/PNG由正式绘图CLI以同一语义几何导出。旧 `model_design.svg/pdf` 保留原文件，已退出正文引用。
- 冻结 ESM-2 预测头与原模型：[比较结果](../../data/results/evaluation/esm2_natv_validation_comparison/REPORT.md)，相应图来自同目录 `esm_comparison.pdf`。
- 新比较的输入：`data/results/preprocessing/natv_matched_fragment_pretraining/` 中的配对清单、掩码和序列；片段匹配图来自 `fragment_matching.pdf`。这是数据准备结果，不是新训练的预测表现。
- 清洗后三条件 QVAE 的验证分数、实际训练量和分层分数：[分层比较](../../data/results/evaluation/natv_fragment_transfer/validation_stratified_comparison/REPORT.md)；MC 标准误及64/128/256样本敏感性：[分析报告](../../data/results/evaluation/natv_fragment_transfer/validation_latent_sample_sensitivity/REPORT.md)。
- 六格完整预测：[保留集报告](../../data/results/evaluation/natv_six_cell/test/REPORT.md)、[验证报告](../../data/results/evaluation/natv_six_cell/validation/REPORT.md)与[保留集MC分析](../../data/results/evaluation/natv_six_cell/test_latent_precision/REPORT.md)；完整旧预测对比数据仍保存在该目录；本轮主图由下述正式绘图入口读取冻结结果重组。
- 六格固定背景生成：[生成报告](../../data/results/evaluation/natv_six_cell_generation/REPORT.md)、`component_effects.csv`、`hamming_paired.csv`、`candidate_distances.csv`；完整比较保留，现已改为补充图，并增加每384次尝试的绝对替换数/唯一序列数图。
- 预测头互补与温度对照：[验证报告](../../data/results/evaluation/natv_head_calibration/validation_protocol_checked/REPORT.md)、[冻结参数保留集报告](../../data/results/evaluation/natv_head_calibration/test/REPORT.md)；对应summary与group_scores记录全部对比、逐seed和逐组结果。初始QVAE/确定性目标概率混合见 `natv_head_mixture/validation_protocol_checked/`，没有追加QVAE温度导出或训练。
- 预测头互补的位点分解：[保留集报告](../../data/results/evaluation/natv_head_complementarity/test/REPORT.md)、[验证报告](../../data/results/evaluation/natv_head_complementarity/validation/REPORT.md)，同目录CSV和summary记录可加贡献、Jensen项、top-1/真实残基排序、逐组稳健性及输入/源码哈希；命令见 `data/results/evaluation/natv_head_complementarity/commands.txt`。分解复用现有完整概率，没有新增训练、前向或生成。
- 历史分组重建：[图重建记录](../../data/results/preprocessing/natv_split_relations/historical_components.json)；全长与域最近邻：[搜索汇总](../../data/results/preprocessing/natv_sequence_neighbors/summary.json)。
- test 序列的历史用途：[逐项核查记录](../../data/results/preprocessing/natv_test_recovered/historical_use.json)。记录包含原始逐序列表、模型种子与执行时间；本轮未重新计算或汇总旧模型分数。
- 本轮入口冒烟评分的执行记录：[test_smoke_execution.json](../../data/results/evaluation/natv_fragment_transfer/test_smoke_execution.json)。该次逐任务输出已删除，记录区分可核验输入与执行者报告；其分数不用于研究结论或选模。

六格比较回答：在片段级序列关系得到处理后，外部片段预训练相对随机初始化、NatV自身片段预训练是否仍有改善；附加任务模型相对冻结ESM预测头增加了多少预测能力。预测头混合及温度对照是在看到六格结果后设计的后续实验，其T与α仅用验证集拟合，随后冻结评价。231条test序列在旧项目中已有评分，其中39条影响过方法选择；新实验同样不能称为从未接触的独立测试。原有验证组删除诊断也不能代替重新训练。

权重、逐查询评分和恢复的测试资料已通过下述Release交付；作者名单、单位、贡献、资助及利益冲突仍需由作者填写。已检查的仓库版本没有项目许可证文件，需由作者确定许可及使用限制；这里不代为授予权利。软件运行条件已依据 `pyproject.toml` 写入数据可用性声明。AI 协助的用途写入方法，具体说明仍需作者确认。没有对外提交稿件。


## 公开下载与复算

[公开数据Release](https://github.com/Yhyb24P/natvqvae/releases/tag/natv-prediction-study-2026-11-10) 对应源码tag `natv-prediction-study-2026-11-10`（`8d4eaedb3d70f4560545e1d0caf52c1bc79cf653`），提供以下实际下载。两个包的原始本地basename都为 `reproduction_data.tar.zst`，下载时改用不同名称避免覆盖；包内容及原SHA不变。

| Release文件 | 对应本地材料 | 大小 |
|---|---|---:|
| `natv_six_cell_reproduction.tar.zst` | `natv_formal_comparison/reproduction_data.tar.zst` | 1,094,147,264 bytes |
| `natv_head_calibration_reproduction.tar.zst` | `natv_head_calibration/reproduction_data.tar.zst` | 71,055,029 bytes |
| `six_cell_source_identity.json` | `natv_formal_comparison/reproduction_source_identity.json` | 4,265 bytes |
| `SHA256SUMS`、`REPRODUCE.md` | 下载校验与源版本、解压、CPU复算说明 | 见Release |

源码/结果tag保留数据发布前的文稿；**数据发布时的交付文稿、可用性声明及核验记录**固定于[文档tag `natv-prediction-study-2026-11-10-delivery`](https://github.com/Yhyb24P/natvqvae/tree/natv-prediction-study-2026-11-10-delivery/docs/manuscript)。这两个tag职责不同，原源码tag不移动；复算仍使用原源码tag，阅读数据发布时的交付文稿使用文档tag。Release说明及REPRODUCE附件均给出这一对应关系。

五个附件已匿名下载并逐一核对SHA；独立源码目录用相同字节的数据包还原后，CPU复算的REPORT和两份CSV与正式结果逐字节一致，summary的统计值、输入哈希和一致性检查也一致。执行约12秒，没有链接原工作区数据或加载ESM权重。核验记录见[public_release.json](../../data/results/evaluation/natv_formal_comparison/public_release.json)。这里提供GitHub Release下载，不宣称已获得DOI或长期存档保证，也不替作者决定许可。

下载并切换到该tag后，按附件 `REPRODUCE.md` 从新目录解压及复算；为避免混用basename，附件中的命令直接使用上述下载名称。下文保留原本地路径，以便核对历史执行记录。新的位点分解已进入Git，旧数据包没有重新打包。

在仓库根目录安装 `pip install -e '.[dev,data]'` 后，使用统一的 `python -m natvqvae.cli`。`config/sources.yaml` 指向冻结 ESM-2 权重；权重文件必须与每次评分的 `esm_files_sha256` 一致。CUDA/bfloat16 是正式评分的实际条件，CPU运行不保证数值逐位一致。

- 两类模型的训练命令及实际运行记录在 `data/results/training/natv_fragment_transfer/` 的 `pretraining_execution.json`、`downstream_execution.json`，以及 `natv_fragment_transfer_deterministic/execution_commands.sh`。每份训练摘要记录选定权重和步数，逐更新 `metrics.jsonl` 支持目标残基暴露量复算。
- 批量评分、六格比较、MC合并与生成使用 `data/results/evaluation/natv_formal_comparison/commands.txt` 索引的正式CLI命令；同目录运行记录保留实际执行及失败后的修正。命令中的本机Python绝对路径在其他环境需替换，原记录不改写成事后执行历史。
- 六格报告由共同逐组分数产生；`paired_group_differences.csv` 保留每项对比，`summary.json` 记录参数和输入身份。`six-cell-plot --comparison <目录>` 仅重新导出图表，不重新评分。
- 原始逐任务评分、全部未筛选生成和训练权重本地保留；Git只选择纳入正式源码、配置、必要输入、报告和适当规模的汇总结果。完整已选研究材料另由下述Release附件提供；GitHub下载不等于带DOI的长期归档，作者声明仍待完成。

所有比较输出应使用新的空目录，避免覆盖已有正式记录。复算统计无需重新训练；只有需要复现训练随机性或改变方法时才重训。

新增预测头实验的执行入口见 `data/results/evaluation/natv_head_calibration/commands.txt`。本地补充包 `data/results/evaluation/natv_head_calibration/reproduction_data.tar.zst` 包含验证与保留集的完整20AA概率、任务清单、导出身份、冻结校准参数、逐组结果及本轮配置；逐文件清单与包校验值在同目录。它补充而不替代下述六格数据包：共享输入、三个选定确定性权重以及数值一致性检查所需的原目标概率仍从六格包获取。CPU复算可直接使用已存概率，不需要重新执行ESM前向计算；只有重做概率导出时才需要相应GPU环境和冻结ESM权重。输入、模型与实现绑定及有限温度网格的解释限制见报告；补充包已以 `natv_head_calibration_reproduction.tar.zst` 公开下载，内容与该本地包逐字节相同。

校准源码与冻结结果版本为 `d359cc28f6d5ac019ea641a59674ead0cb72bb90`，六格源码为 `a971d6e5646aff6f11fa43ecc1be1e21b75a3777`。补充包23个成员已逐文件校验，大小71,055,029 bytes，SHA256为 `b128d83f7dd28ab760b80cc9806d2542ee1aa743a1f54e4afec44b2d62cc746f`。校准包identity明确区分实际源码提交（source_commit）与执行起点（execution_base_commit）。六格原包及历史identity保持原样；其中base_commit只记录准备时checkout，实际源码应以单独发布的[源码身份修正](../../data/results/evaluation/natv_formal_comparison/reproduction_source_identity.json)为准，该文件逐项列出已与a971d6e核对的源码哈希。修正文件使用独立名称，解压旧包不会覆盖它；校准补充包已包含自身修正后的identity。

本轮本地数据包为 `data/results/evaluation/natv_formal_comparison/reproduction_data.tar.zst`（约1.09 GB，解压约5.29 GB），包含6,510个清单内数据文件及2个包级元数据成员（总计6,512个归档成员）：18份选定下游权重、12次辅助预训练的初始/最终权重、训练记录、恢复与配对输入、最近邻原始比对、逐任务预测、全部466,560次生成及比较输出。逐文件路径/大小/SHA256见同目录 `reproduction_manifest.csv`，压缩包校验值见 `.sha256` 文件；代码与配置使用对应Git版本；`reproduction_identity.json` 保留历史准备记录和ESM哈希，六格实际源码身份以独立发布的 `reproduction_source_identity.json` 为准。包已以 `natv_six_cell_reproduction.tar.zst` 公开下载，内容与该本地包逐字节相同；公开链接见下表。

在仓库根目录解压 `tar --zstd -xf data/results/evaluation/natv_formal_comparison/reproduction_data.tar.zst --transform='s,^historical_inputs/,data/raw/historical_split_inputs/,'` 可还原相对数据路径。包内 `historical_inputs/` 保存重建历史分组所需的三个原始文件，上述解压命令将其放入 `data/raw/historical_split_inputs/`；复算该检查时将 `config/natv_split_relations.yaml` 的历史绝对路径改为对应归档路径；原路径及哈希在清单中保留。冻结ESM-2不重复打包，须另外提供与 `reproduction_identity.json` 中五项哈希一致的文件。特征缓存和未选定下游权重不在包内；它们不是复算本轮已选模型统计的必要输入。

## 预测头互补的图件修订

本轮四张主图及补充图由 `src/natvqvae/manuscript_figures.py`、`config/manuscript_figures.yaml` 统一生成：

```bash
python -m natvqvae.cli manuscript-figures --config config/manuscript_figures.yaml --output data/results/evaluation/natv_prediction_figures
```

复算使用新的空输出目录。输入是六格汇总、最近训练近邻表、两种划分的冻结预测头逐组评分和贡献分解，以及原六格生成汇总；不需要完整概率导出、训练或ESM前向。`figure_identity.json` 记录所有输入及实现/输出校验值、四象限计数、同源距离分布和本轮总体accuracy条件区间。配置中的2,000次bootstrap以seed平均后的component为单位，概率层accuracy仅作描述，不套用NLL区间。

`study_design`、`pretraining_and_identity`、`head_mixture_test`、`head_decomposition_test` 是正文四图。QVAE路径、完整六格各类对比、原生成差值与绝对量级、验证集两张对应图进入补充材料。图件均提供PDF、SVG和PNG，按170 mm实际宽度绘制，统计图最小字号8.5 pt；架构图和PDF采用同一语义几何，SVG经过fireworks-tech-graph检查，并借鉴diagram-design的减少冗余节点与限制视觉焦点原则。稿件版心为170 mm，保留双倍行距、页码及行号，不改动原模板类文件。旧完整矩阵及数据包保持原记录。

本次图件/稿件快照使用 `natv-prediction-figures-2026-11-10`；数据发布时的原始源码与文稿tag继续保留。后续查看当前论文使用这一图件修订快照，不把原数据发布tag中的旧图件当成这次版本。图件重排没有新增混合头在同源距离上的分层结论，也没有新增混合生成或候选功能证据。
