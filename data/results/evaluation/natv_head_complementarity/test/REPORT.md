# 预测头互补的位点分解

Mean sites within query, equal queries within component/region, panel-weighted queries for all_panels; then equal components and seeds.

A subset keeps its original denominator; absent sites contribute zero. Partition masses sum to 1 and metric contributions sum to the overall metric.

Post hoc decomposition, overlapping mask contexts, historically evaluated held-out set. Neither independent confirmation nor causal biological information attribution. Panel parameters apply to the entire query, not residue-specific domain routing.

| 方法 | 区域 | 分层 | 类别 | 加权占比 | 对总NLL差的贡献 | 条件组区间 |
|---|---|---|---|---:|---:|---|
| single_mixture | NMNAT | all | all | 1.000000 | -0.049884 | [-0.071311, -0.029649] |
| single_mixture | NMNAT | head_top1 | same | 0.573626 | -0.017714 | [-0.024927, -0.011129] |
| single_mixture | NMNAT | head_top1 | different | 0.426374 | -0.032170 | [-0.048386, -0.016631] |
| single_mixture | NMNAT | esm_true_probability | below_0.01 | 0.026594 | -0.010145 | [-0.014961, -0.006205] |
| single_mixture | NMNAT | esm_true_probability | 0.01_to_0.05 | 0.146543 | -0.023818 | [-0.032004, -0.015710] |
| single_mixture | NMNAT | esm_true_probability | at_least_0.05 | 0.826863 | -0.015920 | [-0.029124, -0.002852] |
| single_mixture | Nudix | all | all | 1.000000 | -0.204578 | [-0.239962, -0.170540] |
| single_mixture | Nudix | head_top1 | same | 0.616437 | -0.051035 | [-0.063763, -0.039147] |
| single_mixture | Nudix | head_top1 | different | 0.383563 | -0.153543 | [-0.180476, -0.127789] |
| single_mixture | Nudix | esm_true_probability | below_0.01 | 0.025972 | -0.017680 | [-0.025812, -0.010808] |
| single_mixture | Nudix | esm_true_probability | 0.01_to_0.05 | 0.130438 | -0.072748 | [-0.090914, -0.055871] |
| single_mixture | Nudix | esm_true_probability | at_least_0.05 | 0.843591 | -0.114151 | [-0.135813, -0.092651] |
| region_calibrated_mixture | NMNAT | all | all | 1.000000 | -0.045044 | [-0.058125, -0.032837] |
| region_calibrated_mixture | NMNAT | head_top1 | same | 0.573626 | -0.015203 | [-0.020741, -0.010349] |
| region_calibrated_mixture | NMNAT | head_top1 | different | 0.426374 | -0.029841 | [-0.039259, -0.021019] |
| region_calibrated_mixture | NMNAT | esm_true_probability | below_0.01 | 0.026594 | -0.014899 | [-0.019316, -0.011297] |
| region_calibrated_mixture | NMNAT | esm_true_probability | 0.01_to_0.05 | 0.146543 | -0.029512 | [-0.034595, -0.024491] |
| region_calibrated_mixture | NMNAT | esm_true_probability | at_least_0.05 | 0.826863 | -0.000633 | [-0.008082, +0.006678] |
| region_calibrated_mixture | Nudix | all | all | 1.000000 | -0.201400 | [-0.239747, -0.164632] |
| region_calibrated_mixture | Nudix | head_top1 | same | 0.616437 | -0.043676 | [-0.058328, -0.029351] |
| region_calibrated_mixture | Nudix | head_top1 | different | 0.383563 | -0.157724 | [-0.186614, -0.130830] |
| region_calibrated_mixture | Nudix | esm_true_probability | below_0.01 | 0.025972 | -0.028754 | [-0.037932, -0.020526] |
| region_calibrated_mixture | Nudix | esm_true_probability | 0.01_to_0.05 | 0.130438 | -0.089373 | [-0.108002, -0.071336] |
| region_calibrated_mixture | Nudix | esm_true_probability | at_least_0.05 | 0.843591 | -0.083273 | [-0.106348, -0.058971] |

## NMNAT整体诊断

以下准确率与真实残基平均排序沿用相同query/component权重；排序1为最高。

| 方法 | 加权单头代价 | Jensen差 | ESM/任务/混合 top1准确率 | 混合减ESM平均排序 | 改善组数 | 去掉最大改善组的均差 | 优于事后逐组最佳单头 |
|---|---:|---:|---|---:|---:|---:|---|
| single_mixture | +0.037154 | 0.087038 | 0.463744/0.461433/0.484339 | -0.104764 | 30/44 | -0.046637 | 28/44, 均差-0.018040 |
| region_calibrated_mixture | +0.013168 | 0.058211 | 0.463744/0.461433/0.483459 | -0.114994 | 39/44 | -0.042687 | 31/44, 均差-0.011728 |
