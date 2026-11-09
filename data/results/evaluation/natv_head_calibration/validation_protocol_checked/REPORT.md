# 温度校准与预测概率互补

Fit T for ESM and seed-averaged deterministic head, then alpha on the same training components; single all_panels objective or separate five-region objectives. Validation excludes the entire scored component from both fits.

Equal components; equal queries within each region; panel-weighted all_panels. Per-query mean target NLL in nats.

Exploratory after six-cell results. Original checkpoint selection used validation; test has prior use. Conditional component bootstrap does not refit heads or calibration parameters. Temperature grid cannot exclude every possible calibration explanation. Temperatures optimize single heads before alpha, not a joint optimum; mixture distributions are not themselves temperature-calibrated.

| 对比 | 区域 | NLL差 | 95%条件组区间 |
|---|---|---:|---|
| task_minus_esm | NMNAT | +0.132765 | [0.09854564183911066, 0.1678102122494664] |
| task_minus_esm | Nudix | -0.064807 | [-0.13696812737827607, 0.011218565355405694] |
| task_minus_esm | all_panels | +0.031587 | [-0.013222828412742037, 0.08084329249870681] |
| single_calibrated_task_minus_single_calibrated_esm | NMNAT | +0.093998 | [0.06550655790612997, 0.12452915537055526] |
| single_calibrated_task_minus_single_calibrated_esm | Nudix | -0.099586 | [-0.1574933040444448, -0.0409225826524937] |
| single_calibrated_task_minus_single_calibrated_esm | all_panels | -0.011260 | [-0.04604751832649973, 0.025428468865531732] |
| single_calibrated_esm_minus_esm | NMNAT | +0.000000 | [0.0, 0.0] |
| single_calibrated_esm_minus_esm | Nudix | +0.000000 | [0.0, 0.0] |
| single_calibrated_esm_minus_esm | all_panels | +0.000000 | [0.0, 0.0] |
| single_calibrated_task_minus_task | NMNAT | -0.038767 | [-0.04928503994881186, -0.028400831086437858] |
| single_calibrated_task_minus_task | Nudix | -0.034779 | [-0.05311112296037639, -0.018061502885276526] |
| single_calibrated_task_minus_task | all_panels | -0.042847 | [-0.05689331026478486, -0.03041091808951886] |
| single_mixture_minus_esm | NMNAT | -0.027710 | [-0.04292928136220057, -0.011089536022726738] |
| single_mixture_minus_esm | Nudix | -0.162440 | [-0.19854686146154504, -0.1265966841361539] |
| single_mixture_minus_esm | all_panels | -0.097366 | [-0.11793732997338395, -0.07590056137602134] |
| single_mixture_minus_task | NMNAT | -0.160475 | [-0.18188360582412388, -0.14031329764487274] |
| single_mixture_minus_task | Nudix | -0.097633 | [-0.1386295077292227, -0.05935058763317581] |
| single_mixture_minus_task | all_panels | -0.128952 | [-0.15785406910049193, -0.10264794185573346] |
| single_calibrated_mixture_minus_single_calibrated_esm | NMNAT | -0.023958 | [-0.03791112717194453, -0.00900981925948947] |
| single_calibrated_mixture_minus_single_calibrated_esm | Nudix | -0.146389 | [-0.17716600718843648, -0.11530305837433082] |
| single_calibrated_mixture_minus_single_calibrated_esm | all_panels | -0.088266 | [-0.10552989107040214, -0.07028883129165488] |
| single_calibrated_mixture_minus_single_calibrated_task | NMNAT | -0.117956 | [-0.13392547391542878, -0.10257195037944405] |
| single_calibrated_mixture_minus_single_calibrated_task | Nudix | -0.046803 | [-0.0758003064184231, -0.019847183725151884] |
| single_calibrated_mixture_minus_single_calibrated_task | all_panels | -0.077005 | [-0.09692961495192541, -0.05858770706939867] |
| single_calibrated_mixture_minus_single_mixture | NMNAT | +0.003752 | [-0.00014270561182172544, 0.007440182177704222] |
| single_calibrated_mixture_minus_single_mixture | Nudix | +0.016051 | [0.00990612245461582, 0.02175767338896584] |
| single_calibrated_mixture_minus_single_mixture | all_panels | +0.009100 | [0.004796846289608338, 0.01323545053641632] |
| region_calibrated_task_minus_region_calibrated_esm | NMNAT | +0.093998 | [0.06550655790612997, 0.12452915537055526] |
| region_calibrated_task_minus_region_calibrated_esm | Nudix | -0.099586 | [-0.1574933040444448, -0.0409225826524937] |
| region_calibrated_task_minus_region_calibrated_esm | all_panels | -0.011260 | [-0.04604751832649973, 0.025428468865531732] |
| region_calibrated_esm_minus_esm | NMNAT | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_esm | Nudix | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_esm | all_panels | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_task | NMNAT | -0.038767 | [-0.04928503994881186, -0.028400831086437858] |
| region_calibrated_task_minus_task | Nudix | -0.034779 | [-0.05311112296037639, -0.018061502885276526] |
| region_calibrated_task_minus_task | all_panels | -0.042847 | [-0.05689331026478486, -0.03041091808951886] |
| region_mixture_minus_esm | NMNAT | -0.036280 | [-0.04623026494293785, -0.02512192462227368] |
| region_mixture_minus_esm | Nudix | -0.165779 | [-0.20725815361186414, -0.1246672917086895] |
| region_mixture_minus_esm | all_panels | -0.100753 | [-0.12118629520430842, -0.07887362101521592] |
| region_mixture_minus_task | NMNAT | -0.169045 | [-0.19482078695685903, -0.1444334731941559] |
| region_mixture_minus_task | Nudix | -0.100972 | [-0.136182030725164, -0.0678340834606842] |
| region_mixture_minus_task | all_panels | -0.132340 | [-0.16091203115697897, -0.10629367506986033] |
| region_calibrated_mixture_minus_region_calibrated_esm | NMNAT | -0.032038 | [-0.0412978896711739, -0.021906013483396162] |
| region_calibrated_mixture_minus_region_calibrated_esm | Nudix | -0.151276 | [-0.1888219625613836, -0.11395604411152024] |
| region_calibrated_mixture_minus_region_calibrated_esm | all_panels | -0.090778 | [-0.10889077014793624, -0.07151432975242177] |
| region_calibrated_mixture_minus_region_calibrated_task | NMNAT | -0.126036 | [-0.1467399858536048, -0.10572881344508793] |
| region_calibrated_mixture_minus_region_calibrated_task | Nudix | -0.051690 | [-0.0739597011523094, -0.0309785630916244] |
| region_calibrated_mixture_minus_region_calibrated_task | all_panels | -0.079518 | [-0.09808465742288335, -0.06222567085494492] |
| region_calibrated_mixture_minus_region_mixture | NMNAT | +0.004242 | [0.001496735217491396, 0.00678011612642303] |
| region_calibrated_mixture_minus_region_mixture | Nudix | +0.014503 | [0.00909734782242514, 0.019863228952748887] |
| region_calibrated_mixture_minus_region_mixture | all_panels | +0.009975 | [0.006349652854169233, 0.013347164378301343] |
| region_calibrated_esm_minus_single_calibrated_esm | NMNAT | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_single_calibrated_esm | Nudix | +0.000000 | [0.0, 0.0] |
| region_calibrated_esm_minus_single_calibrated_esm | all_panels | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_single_calibrated_task | NMNAT | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_single_calibrated_task | Nudix | +0.000000 | [0.0, 0.0] |
| region_calibrated_task_minus_single_calibrated_task | all_panels | +0.000000 | [0.0, 0.0] |
| region_mixture_minus_single_mixture | NMNAT | -0.008570 | [-0.01413714449339292, -0.0033135570206308264] |
| region_mixture_minus_single_mixture | Nudix | -0.003339 | [-0.00885930887840797, 0.0024470349684827222] |
| region_mixture_minus_single_mixture | all_panels | -0.003388 | [-0.004873280232852662, -0.0018485132068328206] |
| region_calibrated_mixture_minus_single_calibrated_mixture | NMNAT | -0.008080 | [-0.013189040170788494, -0.00328885567105654] |
| region_calibrated_mixture_minus_single_calibrated_mixture | Nudix | -0.004887 | [-0.01142185153352802, 0.0020852072632186512] |
| region_calibrated_mixture_minus_single_calibrated_mixture | all_panels | -0.002512 | [-0.004612719833758361, -0.0002851208978585383] |
