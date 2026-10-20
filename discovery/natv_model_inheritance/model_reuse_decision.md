# 旧 QVAE 实现继承判断

## 决定

WT 条件下的对齐列先验生成优先复用旧 `design/sequence_generator.py` 中的 `SequenceGenerator`，不另建一套 Categorical QVAE。H2 `H2RecoveryModel` 保留为条件编辑的已训练比较资产；Continuous MMD-VAE 按研究问题决定是否纳入，不默认重训。原始 `core/model.py` 的能量模型、结构与属性模块不随类名整体继承。

源码、报告、配置、权重及其校验值见 `source_checks.json`；角色交叉表见 `split_overlap.csv`。来源为本机只读 `nad_qvae`，以文件 SHA256 和记录的 Git revision 确定身份。所给 GitHub 页面在本次网页读取中返回 404，远端可访问性与本地文件可用性分别判断。

## 实现与已有结果

| 实现 | 已核对的机制 | 适用任务 |
|---|---|---|
| `core/model.py` | 每残基类别潜变量、Gumbel-Softmax、上下文潜变量解码、能量及辅助头；名为 ARDecoder，但不读取移位目标 token | 需逐项核对历史训练与先验，不能只凭 QVAE 类名接入 |
| `design/sequence_generator.py` | 序列级 64×2 类别潜变量；独立均匀 `sample_prior()`；固定 WT ESM-2 条件；`decode(latent)` 不读取目标序列 | 当前优先继承的 WT 对齐列生成实现 |
| `design/h2_model_v1.py` | 继承 `VNextRecoveryModel`，加冻结 masked PLM logits 与可注入的先验潜变量；支持 categorical / continuous_mmd / no_latent | 重新编码当前序列的条件编辑，不能直接视为全序列联合生成 |

`SequenceGenerator` 的解码器在给定 z 与 WT 时逐列输出类别分布；共享 z 的混合可以产生跨位置依赖，但这不是已验证的多位点组合收益。输出为 341 个对齐列，每列 20 种氨基酸加 gap；不能生成新插入片段。旧候选程序先抽取 z，再选择 WT 的 1/2/3 个位点、限制到允许替换并排除 gap；旧候选不是直接从完整解码分布抽取的原始序列。

旧 `natv_sequence_design_dev_v1` 已完成 categorical_vae 与 sequence_denoiser 各三个种子训练。两组 checkpoint 共 6 份，SHA256 与冻结记录一致，当前源码的四项训练代码及配置也与冻结值相同；CPU `load_state_dict(strict=True)` 全部成功，参数均为有限值。WT 条件缓存的校验值亦一致。

该开发运行的去噪基线读取被遮蔽序列的编码表示；VAE 与它的训练目标不同。其测试报告在实际 70/80 组分上复算后，VAE 相对 raw-sequence PLM 的均值差未显示稳定改善。不能将 masked CE 差或潜变量改变 logits 解释成独立生成优势。

H2 的 `h2_model_v1.py` 本机存在、受 Git 跟踪，其 SHA256 与 `formal_source_freeze.json` 一致；依赖的 `vnext_model.py` 也一致。冻结的 15 份生成 checkpoint 全部通过 SHA256、严格加载及参数有限值核查。正式 H2 阶段报告不支持当前固定 4/8 位点任务上的主要生成比例增益；这一负结果保留，不能外推为其他 QVAE 任务无效。

## 数据继承界限

旧 `SequenceGenerator` 训练 / 验证 / 测试为 1,788 / 340 / 36 条，另有参考 9 条、排除 257 条；现役数据身份见 `../../docs/DATA_PROVENANCE.md`。两套群体逐 ID 对应，但角色不同。旧训练集含现役验证 190 条、现役测试 219 条；现役排除的 4 条也曾进入旧训练。因此旧权重不能作为现役划分下正式模型，也不能用现役验证或测试直接确认这些权重的泛化能力。

旧运行的对齐算法可以复用，但角色、组分和权重须取现役清单；不能调用旧 `build_corpus()` 重新分组划分，也不能继承旧来源组别的固定权重。现有掩码表示不等于 `SequenceGenerator` 的输入；它需要源序列到 WT 列的显式对应、gap/unknown 处理，以及可核对的固定 WT 条件。

## 下一步研究定义

先确定是完整 WT 对齐列先验采样，还是限制替换数的 WT 设计；两种输出分别统计。若采用前者，明确 gap 的处理与插入信息的保留，原始采样和后续筛选分开记录尝试数、失败和重复，避免再次只研究稀疏替换。

在同一正式数据与任务上定义 QVAE 与无潜变量基线。历史 sequence_denoiser 可提供实现资产，但其输入信息、目标和容量需要对齐；固定 z、打乱 z 或禁用 z 是机制诊断，不能代替独立训练的无潜变量基线。SE 可作为条件编辑参照，不能在信息不同的任务上直接比较总分。Continuous MMD-VAE 仅在需要区分离散化效应时追加。

评价需覆盖先验多位点组合、序列多样性和设计空间；先验能够影响输出、恢复 CE 较低及新颖率饱和均不足以单独完成该判断。新的训练与生成方案需绑定任务、评价和预算后执行。

## 复现本地核对

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yhshy/miniconda3/envs/nad_data/bin/python discovery/natv_model_inheritance/audit_sources.py
```

在项目根目录执行，依赖现有 pandas、Parquet 引擎、PyTorch、NumPy、PyYAML 与 Biopython。脚本校验历史冻结来源、在 CPU 严格加载权重并交叉统计角色；不调用模型 forward。`--root` 指定只读来源，`--out` 指定核对输出目录。
