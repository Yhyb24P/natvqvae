# 预测头互补的位点分解

Mean sites within query, equal queries within component/region, panel-weighted queries for all_panels; then equal components and seeds.

A subset keeps its original denominator; absent sites contribute zero. Partition masses sum to 1 and metric contributions sum to the overall metric.

Post hoc decomposition, overlapping mask contexts, historically evaluated held-out set. Neither independent confirmation nor causal biological information attribution. Panel parameters apply to the entire query, not residue-specific domain routing.

| 方法 | 区域 | 分层 | 类别 | 加权占比 | 对总NLL差的贡献 | 条件组区间 |
|---|---|---|---|---:|---:|---|
| single_mixture | NMNAT | all | all | 1.000000 | -0.027710 | [-0.042929, -0.011090] |
| single_mixture | NMNAT | head_top1 | same | 0.553388 | -0.011381 | [-0.017929, -0.005074] |
| single_mixture | NMNAT | head_top1 | different | 0.446612 | -0.016329 | [-0.028089, -0.003764] |
| single_mixture | NMNAT | esm_true_probability | below_0.01 | 0.029070 | -0.009681 | [-0.013626, -0.006030] |
| single_mixture | NMNAT | esm_true_probability | 0.01_to_0.05 | 0.144092 | -0.018006 | [-0.025605, -0.010353] |
| single_mixture | NMNAT | esm_true_probability | at_least_0.05 | 0.826838 | -0.000024 | [-0.011600, +0.012661] |
| single_mixture | Nudix | all | all | 1.000000 | -0.162440 | [-0.198547, -0.126597] |
| single_mixture | Nudix | head_top1 | same | 0.600007 | -0.047019 | [-0.058634, -0.035272] |
| single_mixture | Nudix | head_top1 | different | 0.399993 | -0.115421 | [-0.142021, -0.089577] |
| single_mixture | Nudix | esm_true_probability | below_0.01 | 0.021558 | -0.009178 | [-0.013538, -0.005195] |
| single_mixture | Nudix | esm_true_probability | 0.01_to_0.05 | 0.137933 | -0.061895 | [-0.078832, -0.044866] |
| single_mixture | Nudix | esm_true_probability | at_least_0.05 | 0.840509 | -0.091367 | [-0.113127, -0.068506] |
| region_calibrated_mixture | NMNAT | all | all | 1.000000 | -0.032038 | [-0.041298, -0.021906] |
| region_calibrated_mixture | NMNAT | head_top1 | same | 0.553388 | -0.010758 | [-0.014992, -0.006597] |
| region_calibrated_mixture | NMNAT | head_top1 | different | 0.446612 | -0.021280 | [-0.028627, -0.013651] |
| region_calibrated_mixture | NMNAT | esm_true_probability | below_0.01 | 0.029070 | -0.015001 | [-0.018773, -0.011431] |
| region_calibrated_mixture | NMNAT | esm_true_probability | 0.01_to_0.05 | 0.144092 | -0.024982 | [-0.029780, -0.020195] |
| region_calibrated_mixture | NMNAT | esm_true_probability | at_least_0.05 | 0.826838 | +0.007946 | [+0.001326, +0.015243] |
| region_calibrated_mixture | Nudix | all | all | 1.000000 | -0.151276 | [-0.188822, -0.113956] |
| region_calibrated_mixture | Nudix | head_top1 | same | 0.600007 | -0.038315 | [-0.050708, -0.026332] |
| region_calibrated_mixture | Nudix | head_top1 | different | 0.399993 | -0.112961 | [-0.140863, -0.086259] |
| region_calibrated_mixture | Nudix | esm_true_probability | below_0.01 | 0.021558 | -0.016805 | [-0.022110, -0.012021] |
| region_calibrated_mixture | Nudix | esm_true_probability | 0.01_to_0.05 | 0.137933 | -0.078954 | [-0.096024, -0.061789] |
| region_calibrated_mixture | Nudix | esm_true_probability | at_least_0.05 | 0.840509 | -0.055517 | [-0.080097, -0.030446] |

## NMNAT整体诊断

以下准确率与真实残基平均排序沿用相同query/component权重；排序1为最高。

| 方法 | 加权单头代价 | Jensen差 | ESM/任务/混合 top1准确率 | 混合减ESM平均排序 | 改善组数 | 去掉最大改善组的均差 | 优于事后逐组最佳单头 |
|---|---:|---:|---|---:|---:|---:|---|
| single_mixture | +0.066382 | 0.094093 | 0.460342/0.442058/0.469130 | -0.064208 | 28/44 | -0.024853 | 28/44, 均差-0.018348 |
| region_calibrated_mixture | +0.028199 | 0.060237 | 0.460342/0.442058/0.471344 | -0.088391 | 35/44 | -0.030024 | 34/44, 均差-0.021243 |
