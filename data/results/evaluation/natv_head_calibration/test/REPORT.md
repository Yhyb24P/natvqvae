# 温度校准与预测概率互补

Fit T for ESM and seed-averaged deterministic head, then alpha on the same training components; single all_panels objective or separate five-region objectives. Validation excludes the entire scored component from both fits.

Equal components; equal queries within each region; panel-weighted all_panels. Per-query mean target NLL in nats.

Exploratory after six-cell results. Original checkpoint selection used validation; test has prior use. Conditional component bootstrap does not refit heads or calibration parameters. Temperature grid cannot exclude every possible calibration explanation. Temperatures optimize single heads before alpha, not a joint optimum; mixture distributions are not themselves temperature-calibrated.

| 对比 | 区域 | NLL差 | 95%条件组区间 |
|---|---|---:|---|
| task_minus_esm | NMNAT | +0.074308 | [0.030314595086484437, 0.11708696236391505] |
| task_minus_esm | Nudix | -0.140941 | [-0.21213732146867068, -0.0686696197334185] |
| task_minus_esm | all_panels | -0.060937 | [-0.11035201223995773, -0.008634513313773486] |
| single_calibrated_task_minus_single_calibrated_esm | NMNAT | +0.043892 | [0.006128586311327145, 0.07868723793920032] |
| single_calibrated_task_minus_single_calibrated_esm | Nudix | -0.169804 | [-0.22694166144941585, -0.11197483666636165] |
| single_calibrated_task_minus_single_calibrated_esm | all_panels | -0.086546 | [-0.12654131644477165, -0.04542313828586313] |
| single_calibrated_esm_minus_esm | NMNAT | +0.000000 | [0.0, 0.0] |
| single_calibrated_esm_minus_esm | Nudix | +0.000000 | [0.0, 0.0] |
| single_calibrated_esm_minus_esm | all_panels | +0.000000 | [0.0, 0.0] |
| single_calibrated_task_minus_task | NMNAT | -0.030416 | [-0.04147840456223675, -0.019609778142248502] |
| single_calibrated_task_minus_task | Nudix | -0.028863 | [-0.04580498521907877, -0.0123327734854517] |
| single_calibrated_task_minus_task | all_panels | -0.025610 | [-0.038097478848237366, -0.01323297002462458] |
| single_mixture_minus_esm | NMNAT | -0.049884 | [-0.07131128639463082, -0.029649465729525357] |
| single_mixture_minus_esm | Nudix | -0.204578 | [-0.23996244217141946, -0.17054035060965544] |
| single_mixture_minus_esm | all_panels | -0.138808 | [-0.16271076039146842, -0.11424040115381484] |
| single_mixture_minus_task | NMNAT | -0.124191 | [-0.1470265311154717, -0.10097584567938118] |
| single_mixture_minus_task | Nudix | -0.063637 | [-0.10641323469351548, -0.023957097554135307] |
| single_mixture_minus_task | all_panels | -0.077871 | [-0.10608587337958937, -0.05113792131042725] |
| single_calibrated_mixture_minus_single_calibrated_esm | NMNAT | -0.044730 | [-0.06411012885513026, -0.026520686455806443] |
| single_calibrated_mixture_minus_single_calibrated_esm | Nudix | -0.188243 | [-0.21971373567361724, -0.15774680210847355] |
| single_calibrated_mixture_minus_single_calibrated_esm | all_panels | -0.125328 | [-0.1462251468919877, -0.1042894816867085] |
| single_calibrated_mixture_minus_single_calibrated_task | NMNAT | -0.088622 | [-0.10633933401879049, -0.07079152836421115] |
| single_calibrated_mixture_minus_single_calibrated_task | Nudix | -0.018439 | [-0.04930230281160643, 0.010211189280517107] |
| single_calibrated_mixture_minus_single_calibrated_task | all_panels | -0.038781 | [-0.05878967060193554, -0.018832292119043584] |
| single_calibrated_mixture_minus_single_mixture | NMNAT | +0.005154 | [0.0011214358864232471, 0.008990381264038685] |
| single_calibrated_mixture_minus_single_mixture | Nudix | +0.016334 | [0.010585639018681866, 0.02165671554252772] |
| single_calibrated_mixture_minus_single_mixture | all_panels | +0.013480 | [0.009358255206324677, 0.017287557442140222] |
| region_calibrated_task_minus_region_calibrated_esm | NMNAT | +0.043892 | [0.006128586311327145, 0.07868723793920032] |
| region_calibrated_task_minus_region_calibrated_esm | Nudix | -0.169804 | [-0.22694166144941585, -0.11197483666636165] |
| region_calibrated_task_minus_region_calibrated_esm | all_panels | -0.086546 | [-0.12654131644477165, -0.04542313828586313] |
| region_calibrated_esm_minus_esm | NMNAT | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_esm | Nudix | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_esm | all_panels | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_task | NMNAT | -0.030416 | [-0.04147840456223675, -0.019609778142248502] |
| region_calibrated_task_minus_task | Nudix | -0.028863 | [-0.04580498521907877, -0.0123327734854517] |
| region_calibrated_task_minus_task | all_panels | -0.025610 | [-0.038097478848237366, -0.01323297002462458] |
| region_mixture_minus_esm | NMNAT | -0.050516 | [-0.06483496429001129, -0.036992793800869464] |
| region_mixture_minus_esm | Nudix | -0.213398 | [-0.2534017428715993, -0.17412074503281624] |
| region_mixture_minus_esm | all_panels | -0.141505 | [-0.16516914259804827, -0.11748100689052678] |
| region_mixture_minus_task | NMNAT | -0.124824 | [-0.15436700361775801, -0.0948040535185683] |
| region_mixture_minus_task | Nudix | -0.072457 | [-0.1106571021797729, -0.03734028166554795] |
| region_mixture_minus_task | all_panels | -0.080568 | [-0.10893574376827568, -0.05348156609838785] |
| region_calibrated_mixture_minus_region_calibrated_esm | NMNAT | -0.045044 | [-0.05812485632098079, -0.032836503852725706] |
| region_calibrated_mixture_minus_region_calibrated_esm | Nudix | -0.201400 | [-0.23974715450241096, -0.16463248422643625] |
| region_calibrated_mixture_minus_region_calibrated_esm | all_panels | -0.131368 | [-0.1531291429204699, -0.10907922269778887] |
| region_calibrated_mixture_minus_region_calibrated_task | NMNAT | -0.088936 | [-0.11225950153243194, -0.06453700264761576] |
| region_calibrated_mixture_minus_region_calibrated_task | Nudix | -0.031596 | [-0.05602788569918072, -0.009218092478199795] |
| region_calibrated_mixture_minus_region_calibrated_task | all_panels | -0.044822 | [-0.06381832305025038, -0.025933360904392615] |
| region_calibrated_mixture_minus_region_mixture | NMNAT | +0.005473 | [0.003032433344884935, 0.007956724318945115] |
| region_calibrated_mixture_minus_region_mixture | Nudix | +0.011998 | [0.006503680426329941, 0.016920448804670907] |
| region_calibrated_mixture_minus_region_mixture | all_panels | +0.010137 | [0.006882125245707927, 0.01313134537326431] |
| region_calibrated_esm_minus_single_calibrated_esm | NMNAT | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_single_calibrated_esm | Nudix | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_single_calibrated_esm | all_panels | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_single_calibrated_task | NMNAT | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_single_calibrated_task | Nudix | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_single_calibrated_task | all_panels | +0.000000 | [0.0, 0.0] |
| region_mixture_minus_single_mixture | NMNAT | -0.000632 | [-0.007502180134628228, 0.006445155838001632] |
| region_mixture_minus_single_mixture | Nudix | -0.008821 | [-0.013925317543920427, -0.0035866805753356264] |
| region_mixture_minus_single_mixture | all_panels | -0.002697 | [-0.00441388042926945, -0.0009985354257300515] |
| region_calibrated_mixture_minus_single_calibrated_mixture | NMNAT | -0.000314 | [-0.0062367421370811815, 0.006019177410775583] |
| region_calibrated_mixture_minus_single_calibrated_mixture | Nudix | -0.013156 | [-0.019783380046943626, -0.0064285685265087125] |
| region_calibrated_mixture_minus_single_calibrated_mixture | all_panels | -0.006040 | [-0.008014241458952139, -0.003997054520108975] |
