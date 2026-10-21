# 目标条件后验训练

在正式训练 1,644 条、验证 227 条上，从头拟合 QVAE 与确定性对照，各使用 20261021、20261022、20261023 三个配对种子。只开展训练与验证。总计单卡 3,600 秒（含输入与缓存核验），每次最多 100 轮；生成接口的组分平衡验证 CE 连续 10 轮不降低即停止该次拟合，保存严格最低值对应权重。每轮覆盖三个验证面板，每个输入使用固定的一次先验采样；这是带抽样误差的 CE，不是边缘似然。

QVAE 后验为 `q(z | x_masked, y_masked)`：遮蔽上下文表示与仅在遮蔽位点启用的目标 embedding 相加，经过一层独立 Transformer，按遮蔽位点池化后产生 32×8 类别 logits。目标只进入后验；解码器读取遮蔽上下文及 z。训练保留温度 1 的软 Gumbel-Softmax、均匀先验和原 KL 系数及预热。先验评价读取 one-hot z，另记录类别概率均值与固定后验样本 CE；软硬接口的差异仍需纳入解释。

共同主体、学习率、正则化、正式数据角色与 8%/15%/30% 轮换掩码保持原设置。配对种子的共同参数初值相同；QVAE 额外包含目标 embedding 和后验 Transformer，总参数不相等。该比较是本次机制修正的方法检查，不能单独分离容量、目标输入和随机潜变量的贡献；没有超参数搜索。训练 CE 按样本数汇总带组分权重的批次均值，KL 按样本与类别因子平均，二者口径分别记录。

先核对旧面板、每份特征的哈希及正式角色，再复用冻结 ESM-2 遮蔽特征。新增目标输入来自缓存中的真实标签，不增加外部数据。旧权重、轨迹及哈希快照保留；本轮权重进入 `data/models/kvp40_target_posterior_training/`，逐输入验证记录与掩码、特征清单进入同名 `data/results/`，实验目录保留方案、曲线和诊断摘要。

保存逐掩码比例验证 CE、各类别因子的后验条件熵、汇总分布熵与 argmax 类别计数。两种熵之差只作该验证人口的描述，不称作固定上下文的条件互信息。在最佳权重下，对按序列 ID 排序的前八条验证序列及各自三个面板（共 24 个输入）比较八个独立先验 z 的预测概率，记录 TV 与 argmax 差异；此诊断不抽取完整序列，也不能单独证明家族联合规律。

启动：`PYTHONDONTWRITEBYTECODE=1 /home/yhshy/miniconda3/envs/nad_data/bin/python scripts/run_conditional_study.py --run experiments/kvp40_target_posterior_training`。逐轮记录与最终权重生成后，使用 `scripts/plot_conditional_training.py --run experiments/kvp40_target_posterior_training` 绘图。数据资格见 `docs/DATA_PROVENANCE.md`，历史方法检查见 `experiments/kvp40_conditional_generation/REPORT.md`。
