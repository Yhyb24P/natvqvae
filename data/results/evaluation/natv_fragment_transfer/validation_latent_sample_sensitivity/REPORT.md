# 潜变量与迁移比较的 Monte Carlo 精度

validation：227 条序列、44 个组，每任务 4 次重复、每次 64 个潜变量样本。单位为 nats/目标残基；负差值表示前者 NLL 较低。

下表为 NMNAT，先在同一重复内平均三个配对训练种子的组级差值，再对组等权平均。MC SE 为重复均值的样本标准差除以重复数平方根。组区间先平均 MC 重复，再重采样组；两者回答不同问题，不把 MC 重复当作独立序列组。

| 比较 | 均值 | MC SE | 95% 组 bootstrap 区间 |
| --- | ---: | ---: | --- |
| random:joint_minus_site_marginal | -0.00007956 | 0.00000103 | [-0.00011403, -0.00004547] |
| self:joint_minus_site_marginal | -0.00005385 | 0.00000184 | [-0.00009056, -0.00002041] |
| external:joint_minus_site_marginal | -0.00005983 | 0.00000194 | [-0.00009049, -0.00002927] |
| external_minus_self:prior_joint_nll_per_residue | -0.00811670 | 0.00004423 | [-0.02208588, +0.00624318] |
| external_minus_self:marginal_nll_per_residue | -0.00811072 | 0.00004255 | [-0.02208886, +0.00625452] |

将连续重复的联合概率及每个位点边缘概率分别用 logmeanexp 合并，得到较大样本量的估计；这不是平均 NLL，也没有生成新样本。不同样本量之间共享原始样本，仅用于检查本次有限样本敏感性。全部重复合并后只剩一次估计，MC SE 记为 NA，不能据此声称已收敛或偏差消失。

| NMNAT 比较 | 每次样本数 | 剩余独立重复数 | 均值 | 相对未合并均值变化 |
| --- | ---: | ---: | ---: | ---: |
| random:joint_minus_site_marginal | 64 | 4 | -0.00007956 | +0.00000000 |
| random:joint_minus_site_marginal | 128 | 2 | -0.00008033 | -0.00000077 |
| random:joint_minus_site_marginal | 256 | 1 | -0.00008054 | -0.00000098 |
| self:joint_minus_site_marginal | 64 | 4 | -0.00005385 | +0.00000000 |
| self:joint_minus_site_marginal | 128 | 2 | -0.00005429 | -0.00000045 |
| self:joint_minus_site_marginal | 256 | 1 | -0.00005455 | -0.00000070 |
| external:joint_minus_site_marginal | 64 | 4 | -0.00005983 | -0.00000000 |
| external:joint_minus_site_marginal | 128 | 2 | -0.00006047 | -0.00000064 |
| external:joint_minus_site_marginal | 256 | 1 | -0.00006104 | -0.00000121 |
| external_minus_self:prior_joint_nll_per_residue | 64 | 4 | -0.00811670 | +0.00000000 |
| external_minus_self:prior_joint_nll_per_residue | 128 | 2 | -0.00811681 | -0.00000011 |
| external_minus_self:prior_joint_nll_per_residue | 256 | 1 | -0.00811708 | -0.00000039 |
| external_minus_self:marginal_nll_per_residue | 64 | 4 | -0.00811072 | +0.00000000 |
| external_minus_self:marginal_nll_per_residue | 128 | 2 | -0.00811064 | +0.00000008 |
| external_minus_self:marginal_nll_per_residue | 256 | 1 | -0.00811059 | +0.00000012 |

原始每任务仅有 4 次重复，SE 本身仍不稳定；它不衡量有限样本 log 概率估计偏差，也不检验 bfloat16/float32 数值误差。区间以当前拟合的三个训练重复为条件，不覆盖新的训练过程。因此，差值超过 MC SE 不能单独证明精确分布的联合建模收益。逐种子、逐区域及逐重复差值保存在 JSON/CSV 中。
