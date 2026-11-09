# ESM 与 NatV 任务头的概率互补诊断

划分：validation。验证阶段仅对混合权重做留一component交叉验证；任务模型的既有checkpoint使用过验证集。
保留集已有评价历史。本实验是在查看六格结果后设计，不能称为独立确认，也不单独证明预测头替换导致信息丢失。
混合逐位点概率后取对数，不混合NLL；QVAE使用自身边缘，未改变共享潜变量。

各MC重复分别混合并计算NLL，再平均NLL；不是先跨重复平均概率。区域内等query平均，再等component平均；all_panels使用panel权重。单一global_mixture权重按all_panels目标拟合，不是五区域等权。旧产物只存目标标签概率，不能据此检查完整20AA归一化。

| 模型 | 对比 | 区域 | NLL差 | 95%组区间 |
|---|---|---|---:|---|
| QVAE | task_marginal_minus_esm | NMNAT | +0.130853 | [+0.096855, +0.165457] |
| QVAE | task_marginal_minus_esm | Nudix | -0.066043 | [-0.138170, +0.009469] |
| QVAE | task_marginal_minus_esm | all_panels | +0.030037 | [-0.014526, +0.079140] |
| QVAE | global_mixture_minus_esm | NMNAT | -0.028051 | [-0.043050, -0.011449] |
| QVAE | global_mixture_minus_esm | Nudix | -0.162242 | [-0.198388, -0.126135] |
| QVAE | global_mixture_minus_esm | all_panels | -0.097441 | [-0.117914, -0.076085] |
| QVAE | global_mixture_minus_task_marginal | NMNAT | -0.158903 | [-0.180364, -0.138940] |
| QVAE | global_mixture_minus_task_marginal | Nudix | -0.096199 | [-0.136837, -0.058112] |
| QVAE | global_mixture_minus_task_marginal | all_panels | -0.127478 | [-0.156206, -0.101432] |
| QVAE | region_mixture_minus_esm | NMNAT | -0.036447 | [-0.046348, -0.025492] |
| QVAE | region_mixture_minus_esm | Nudix | -0.165652 | [-0.207026, -0.124592] |
| QVAE | region_mixture_minus_esm | all_panels | -0.100744 | [-0.121116, -0.078898] |
| QVAE | region_mixture_minus_task_marginal | NMNAT | -0.167300 | [-0.192766, -0.142971] |
| QVAE | region_mixture_minus_task_marginal | Nudix | -0.099610 | [-0.134597, -0.066938] |
| QVAE | region_mixture_minus_task_marginal | all_panels | -0.130781 | [-0.159327, -0.104950] |
| QVAE | region_mixture_minus_global_mixture | NMNAT | -0.008396 | [-0.013946, -0.003165] |
| QVAE | region_mixture_minus_global_mixture | Nudix | -0.003410 | [-0.008915, +0.002381] |
| QVAE | region_mixture_minus_global_mixture | all_panels | -0.003303 | [-0.004806, -0.001744] |
| deterministic | task_marginal_minus_esm | NMNAT | +0.132765 | [+0.098546, +0.167810] |
| deterministic | task_marginal_minus_esm | Nudix | -0.064807 | [-0.136968, +0.011219] |
| deterministic | task_marginal_minus_esm | all_panels | +0.031587 | [-0.013223, +0.080843] |
| deterministic | global_mixture_minus_esm | NMNAT | -0.027710 | [-0.042929, -0.011090] |
| deterministic | global_mixture_minus_esm | Nudix | -0.162440 | [-0.198547, -0.126597] |
| deterministic | global_mixture_minus_esm | all_panels | -0.097366 | [-0.117937, -0.075901] |
| deterministic | global_mixture_minus_task_marginal | NMNAT | -0.160475 | [-0.181884, -0.140313] |
| deterministic | global_mixture_minus_task_marginal | Nudix | -0.097633 | [-0.138630, -0.059351] |
| deterministic | global_mixture_minus_task_marginal | all_panels | -0.128952 | [-0.157854, -0.102648] |
| deterministic | region_mixture_minus_esm | NMNAT | -0.036280 | [-0.046230, -0.025122] |
| deterministic | region_mixture_minus_esm | Nudix | -0.165779 | [-0.207258, -0.124667] |
| deterministic | region_mixture_minus_esm | all_panels | -0.100753 | [-0.121186, -0.078874] |
| deterministic | region_mixture_minus_task_marginal | NMNAT | -0.169045 | [-0.194821, -0.144433] |
| deterministic | region_mixture_minus_task_marginal | Nudix | -0.100972 | [-0.136182, -0.067834] |
| deterministic | region_mixture_minus_task_marginal | all_panels | -0.132340 | [-0.160912, -0.106294] |
| deterministic | region_mixture_minus_global_mixture | NMNAT | -0.008570 | [-0.014137, -0.003314] |
| deterministic | region_mixture_minus_global_mixture | Nudix | -0.003339 | [-0.008859, +0.002447] |
| deterministic | region_mixture_minus_global_mixture | all_panels | -0.003388 | [-0.004873, -0.001849] |
