# 研究计划

## 目标

研究 KVP40 NatV 的天然演化序列分布，训练条件 Categorical QVAE，生成结构合理、多样、可追溯的完整双域变体；用序列、结构与环境证据辅助选择，未来用真实酶学数据检验并改进生成。研究对象固定为 341 aa KVP40 WT；NMNAT 4–181、linker 182–203、Nudix 204–338。

## 数据与角色

- 当前 raw 序列为 training+validation 的 1,871 条 NatV 双域序列，来自 `data/raw/sequences/natv_population.fasta`。
- 精确 split 保存在 `data/results/preprocessing/natv_family/`：training 1,644/47 组分，validation 227/44 组分，test 231；training/validation 组分互斥。
- test 只保留 231 个 ID，不保存或加载序列；训练、生成和评价默认不读取 test。249 条注释冲突、75 条旧封存、4 条排除不在当前 split 中。
- 输入特征来自冻结 ESM-2；模型不训练 ESM。`config/sources.yaml` 绑定本地 `data/weights/esm2/` 权重路径。

## 模型与训练

- 完整目标序列后验 `q(z|x_full,c,M)`、条件先验 `p(z|c,M)`、共享 16×8 Categorical 潜变量。
- 训练目标为重构 NLL 的 RLOO 估计加 KL；掩码 M 定义可更新位置，解码器与先验只读取可见上下文和 M。
- 训练入口使用 `config/study.yaml` 的模型、优化、掩码和步数配置；不继承历史 NatV 权重。
- 一个显式的确定性同信息对照保留在模型代码中，用于方法比较；当前正式生成只使用 QVAE checkpoint。

## 生成与评价

- 生成从条件先验采样：固定 WT、掩码、潜变量种子与残基均匀种子，输出完整 341 aa 变体、编辑和来源身份。
- 评价只使用 validation：条件先验/后验下的 NLL、目标位点覆盖和候选有效性；当前评价入口不读取 test。
- 结构与环境证据只能作为条件性证据；模板邻近、HMM 命中、pLDDT 或模型分数都不等于活性或净积累。
- 候选选择优先考虑不同可检验问题、序列/机制互补和不确定性，不把计算候选写成已验证工程酶。

## 当前判断

上一轮条件 QVAE 的形式比较没有取得有意义的共享潜变量联合建模优势或候选供给优势；下一步重点应是候选设计与评价，而不是继续增加模型结构或审计流程。具体负结果见 [RESULTS.md](RESULTS.md)。
