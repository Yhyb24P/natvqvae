# WT ATP/NMN 联合占位的 AF3 交叉评价与 BioEmu apo 构象参照

日期：2026-10-24

## 问题与范围

本轮只回答一个问题：KVP40 WT NMNAT 的双底物联合占位，以及 Mg²⁺ 同时靠近两种底物的组织方式，是否具有超出 Boltz-2 结构先验和既有 MD 初始条件的证据支持。

本轮以 27 个新运行的 AF3 模型作为独立结构预测：WT ATP+NMN 与 WT ATP+NMN+Mg 各 9 个，1F9A ATP+Mg 参照 9 个；同时复用既有 Boltz-2 联合模型和既有 BioEmu2 apo 帧做跨模型与构象比较。AF3 的输入、运行命令、逐样本坐标哈希和构建/预测/分析身份分别见 `build.json`、`runs.csv`、`identity.json`。没有延长或新增 MD，也没有运行 A1/A2/A3、B1/B2、NatV10/89、Protenix、OpenFold3 或 BioEmu2 配体模式。

## 输入与身份

AF3：

- 本地 AF3 3.0.4，权重 `/mnt/e/models/alphafold3/af3.bin`，SHA-256 `df8bbf2621f17dd3ee21c2a921e84a50bc2b80cdc0c7971cb915c2826fee1f9b`；运行脚本 `/home/yhshy/models/alphafold3/src/run_alphafold.py`，SHA-256 `a1dc5033fdd5473de06a1dfa02bdf3f53120ea83294f880198bcb26e64c2d075`。
- 完整 WT 341 aa，使用既有 512 行 A3M（`data/results/evaluation/kvp40_nmnat_binding/inputs/WT.a3m`，SHA-256 `360babfaa9b5a1a8ac0fa36a630b171c14b328f0ffeb408b0f22347963508cc6`），`run_data_pipeline=false`，`templates=[]`，没有口袋、距离或接触约束。
- 两个状态：ATP+NMN 与 ATP+NMN+Mg。每个状态 3 个 seed × 3 个 diffusion sample，共 9 个模型；`buckets=512`，`num_recycles=10`。
- AF3 另在任务目录顶层写出一个跨 sample 的最佳模型；该文件不是额外独立样本，分析未将其计入。正式模型数为 18 个 WT 坐标。
- 1F9A 参照：单链 A 的观测残基 1–164（canonical 168，未解析末端 QTDK），ATP+Mg，query-only MSA，3×3 共 9 个模型。单链、无同源多聚体、无显式水；只作为输入/解析/局部几何的技术校准，不是训练外验证。

BioEmu2：

- 本地 `bioemu2` 0.1.0rc0，checkpoint `/mnt/e/bioemu2/checkpoints/bioemu2-preview/checkpoint.ckpt`，SHA-256 `3b4ba9236b7b458f47b40b694847b6bb4110c744becd6d2c53de7b1d04c07a81`。
- `struct8`：`predict-structure`，8 帧，seed 20261022；`eq8`：`sample-equilibrium`，8 帧，同 seed。两者是不同模式，不能合并成 16 个同分布样本，也不是物理轨迹或平衡分布。
- MSA 由 BioEmu 内部 `get_msa` 从 ColabFold 获得，使用 `use_env=true`。现存缓存 `/mnt/e/bioemu2/msa/7e90e212b21b14a8a647ae3767cb0b13d3caef551fff889af616530d61ff9310.json` 的完整 query 序列与当前 WT 341 aa 一致，共 8170 行，A3M 文本 SHA-256 `cdf8f65708c57ff4b39852acbcaed62e8566367c38dec25f038dd2fa3d14fda7`。该文件是“按完整 WT 序列匹配到的现存缓存”，两次运行实际使用的 MSA 版本未单独记录；它不是 AF3/Boltz 使用的 512 行 MSA，原始检索服务器数据库版本和日期也未记录，因此不能声称三者同源。
- apo 分析只描述蛋白主链、102–105 环、H14/H17/S133/T135/R138 邻域和 NMNAT–Nudix 相对取向，不对配体或金属作任何推断。

## AF3 结果

9 个模型不是 9 次独立实验重复。以下范围按 seed 分层保留，不把极值当作概率。

| 指标 | ATP+NMN | ATP+NMN+Mg |
| --- | --- | --- |
| ATP 与 NMN 同时位于 NMNAT | 9/9 | 9/9 |
| 跨配体最近重原子 | 2.207–3.620 Å，中位 2.633 | 2.352–3.452 Å，中位 2.949 |
| 磷酸 O–O 最近距离 | 同上 | 同上 |
| ATP PA–O3A–PB 连接完整 | 9/9 | 9/9 |
| 蛋白 severe（<2.0 Å） | 0 | 0 |
| pTM / ipTM 中位 | 0.94 / 0.96 | 0.94 / 0.96 |

PA–NMN 候选亲核氧（三个候选全部保留）：

| 状态 | 候选 | PA–Onuc 中位（范围） | O3A–PA–Onuc 中位（范围） |
| --- | --- | --- | --- |
| ATP+NMN | O1P | 6.070（3.840–6.938） | 84.0°（78.4–124.2） |
| ATP+NMN | O2P | 4.068（3.759–5.649） | 92.4°（83.3–133.3） |
| ATP+NMN | O3P | 4.835（3.454–6.885） | 88.6°（81.9–125.7） |
| ATP+NMN+Mg | O1P | 6.206（3.813–6.780） | 130.7°（127.9–139.7） |
| ATP+NMN+Mg | O2P | 4.152（3.865–5.382） | 140.2°（132.8–145.1） |
| ATP+NMN+Mg | O3P | 4.780（3.580–6.665） | 132.8°（124.6–139.9） |

按 seed 的最短候选：无 Mg 组中位 4.311 / 3.997 / 3.840 Å，对应角 84.9° / 95.5° / 97.4°；Mg 组 4.452 / 4.152 / 3.964 Å，对应角 131.8° / 140.2° / 139.9°。Mg 组角度更钝，但没有更短的 PA–Onuc。三个候选的原子标签是 CCD 标号，不表示化学等价氧的固定角色；没有跨模型挑选“最佳”候选。

Mg 位置与近邻：

| 指标 | 结果 |
| --- | --- |
| Mg 位于 NMNAT | 9/9 |
| Mg–ATP 任意 O 最近 | 1.946–2.147 Å，中位 2.089 |
| Mg–NMN 任意 O 最近 | 2.379–3.831 Å，中位 3.360 |
| 同时 ≤2.8 Å 接触 ATP 与 NMN 的模型 | 2/9 |
| Mg 2.8 Å 内蛋白 O | 9/9 为 0 |
| Mg 2.8 Å 内蛋白 N | 0–1 个，中位 0 |

按 seed：seed1 Mg–NMN 中位 3.360 Å、0/3 同时近邻；seed2 3.665 Å、0/3；seed3 2.620 Å、2/3。也就是说，AF3 把 Mg 主要放在 ATP 磷酸一侧，并没有像 Boltz-2 Mg_NMNAT 子集或既有 MD 初态那样稳定地同时贴近两种底物。2.8 Å 只表示近邻计数，不是已验证的配位壳或金属桥接；AF3 CIF 不含水，不能评价第一水合壳。

同一模型比较：AF3 独立支持“ATP 与 NMN 都可放在 NMNAT 内”，但没有独立支持“Mg 持续同时桥接两种底物”。它也不支持把更钝的 O3A–PA–Onuc 角直接解释为更接近生产性进攻。

## AF3 与 Boltz-2 的跨模型姿态

用 NMNAT 4–181 的 CA 叠合后比较；AF3 与 Boltz 的 seed/sample 不配对，以下只作集合层面的描述。每对模型都计入一次；81 对不是 81 个独立样本。

下表“C/N 骨架”只包含配体的 C、N 重原子，排除了核糖氧、磷酸氧和 NMN 酰胺氧，不能称为“全部非磷酸重原子”。

| 比较 | 蛋白 CA RMSD 中位 | ATP C/N 骨架 | NMN C/N 骨架 |
| --- | --- | --- | --- |
| AF3 noMg 组内（36 对） | 0.528 | 0.168 | 0.206 |
| AF3 Mg 组内（36 对） | 0.484 | 0.276 | 0.202 |
| Boltz ATP_NMN 组内（36 对） | 0.253 | 0.247 | 0.330 |
| Boltz ATP_NMN_MG 组内（36 对） | 0.287 | 0.289 | 0.368 |
| AF3 noMg × Boltz ATP_NMN（81 对） | 0.644 | 0.382 | 0.660 |
| AF3 Mg × Boltz ATP_NMN_MG（81 对） | 0.782 | 0.431 | 0.754 |

全重原子对称性 RMSD：先按蛋白 CA 配准，不额外做配体自配准；ATP 仅允许同一磷酸的 O1A/O2A、O1B/O2B、O1G/O2G/O3G 各组内部置换，O3A/O3B 桥氧、糖和碱基按原子名固定；NMN 仅允许 O1P/O2P/O3P 置换。它比 C/N 骨架更能判断整体姿态是否相同，也避免等价磷酸氧标号造成的假大差异。

| 比较 | ATP 全重原子对称 RMSD | NMN 全重原子对称 RMSD |
| --- | --- | --- |
| AF3 noMg 组内 | 0.291 | 0.513 |
| AF3 Mg 组内 | 0.389 | 0.492 |
| Boltz ATP_NMN 组内 | 0.500 | 0.774 |
| Boltz ATP_NMN_MG 组内 | 0.508 | 0.636 |
| AF3 noMg × Boltz ATP_NMN | 0.545 | 0.940 |
| AF3 Mg × Boltz ATP_NMN_MG | 0.563 | 0.989 |

对称性感知的磷酸基团 RMSD（仅保留局部定位信息，不代替全配体姿态）：

| 比较 | ATP PA 端基 | ATP PG 端基 | NMN 磷酸 |
| --- | --- | --- | --- |
| AF3 noMg 组内 | 0.272 | 0.231 | 1.050 |
| AF3 Mg 组内 | 0.162 | 0.195 | 0.941 |
| Boltz ATP_NMN 组内 | 0.522 | 0.350 | 1.410 |
| Boltz ATP_NMN_MG 组内 | 0.457 | 0.398 | 1.023 |
| AF3 noMg × Boltz ATP_NMN | 0.436 | 0.659 | 1.486 |
| AF3 Mg × Boltz ATP_NMN_MG | 0.437 | 0.642 | 1.549 |

跨模型差异高于各自组内背景，尤其 NMN 的朝向和磷酸端基；ATP 非磷酸骨架的差异较小。按 seed 的 AF3×Boltz NMN 磷酸中位为 1.31–1.71 Å（无 Mg）和 1.32–2.21 Å（Mg），seed 之间差异明显。这说明 NMN 的具体姿态高度依赖模型与采样，不能把任一模型的 NMN 朝向当成唯一合理解释；ATP 的放置相对更稳定，但仍不是实验测定。

## 1F9A 技术校准

晶体 chain A 的 Mg–O 距离：O2B 2.202、O3A 2.480、O2G 2.604、O3B 2.812 Å；晶体 3 Å 内没有水氧。

| 模型 | 蛋白 CA RMSD | ATP 全重原子 RMSD（原子名） | ATP 全重原子对称 RMSD | ATP C/N 骨架 | ATP PA 端基（对称） | ATP PG 端基（对称） | Mg 位移 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed1_sample0 | 1.296 | 1.441 | 0.885 | 0.974 | 0.724 | 0.590 | 0.979 |
| seed1_sample1 | 1.213 | 1.715 | 1.548 | 0.892 | 0.711 | 3.140 | 1.979 |
| seed1_sample2 | 1.400 | 1.253 | 0.931 | 1.054 | 0.799 | 0.353 | 0.797 |
| seed2_sample0 | 1.063 | 1.304 | 0.783 | 0.842 | 0.837 | 0.548 | 0.818 |
| seed2_sample1 | 0.967 | 1.363 | 0.771 | 0.828 | 0.723 | 0.311 | 0.830 |
| seed2_sample2 | 1.050 | 1.072 | 0.869 | 0.790 | 0.735 | 0.399 | 0.812 |
| seed3_sample0 | 2.184 | 8.472 | 8.435 | 11.405 | 3.958 | 3.714 | 3.635 |
| seed3_sample1 | 2.711 | 3.376 | 3.334 | 4.254 | 1.720 | 2.225 | 3.080 |
| seed3_sample2 | 2.877 | 3.064 | 2.938 | 3.705 | 1.511 | 2.051 | 3.352 |

前 6 个模型把 ATP 放在晶体口袋附近（ATP C/N 骨架 0.79–1.05 Å，全重原子对称 RMSD 0.771–1.548 Å，Mg 位移 0.80–1.98 Å），但没有复现晶体中固定的 O3A 桥氧近邻：预测的 O3A 多在 3.4–3.8 Å，O3B 也未稳定保持晶体的 2.812 Å。端基氧 O2B/O2G 的同名距离（约 2.2 Å 或约 4.5 Å）受等价氧标号置换影响，只作为原始原子名附表保留，不能据此判断整个磷酸端基是否离开。seed3 的 3 个模型整体失败（ATP 全重原子对称 RMSD 2.938–8.435 Å，Mg 位移 3.08–3.64 Å）。因此，AF3 的输入、CCD、解析和整体口袋重建在 6/9 中可用，但不能可靠重建已知金属配位；它不能用来裁定 Mg 的具体位置，也不能用“AF3 有 Mg”来否定或确认 Boltz/MD 的金属组织。原子名 RMSD 作为原始附加项保留；判断整体姿态应看对称性列。

## BioEmu apo 构象参照

以 NMNAT 4–181 叠合后，比较 102–105 环 + H14/H17/S133/T135/R138 的 CA、Nudix 204–338 的 CA，以及叠合误差。每个 holo 模型对 8 个 apo 帧，共 72 对/来源/模式；这些是描述性配对，不是概率或诱导契合证据。

| apo 模式 | 来源 | 局部 CA RMSD 中位 | Nudix CA RMSD 中位 | 叠合 CA RMSD 中位 |
| --- | --- | --- | --- | --- |
| struct8 | AF3 noMg | 1.478（0.922–3.018） | 2.471（1.112–3.698） | 0.605 |
| struct8 | AF3 Mg | 1.421（0.694–3.348） | 2.205（0.967–3.308） | 0.609 |
| struct8 | Boltz ATP_NMN | 1.388（0.833–1.839） | 1.046（0.491–1.956） | 0.563 |
| struct8 | Boltz ATP_NMN_MG | 1.348（0.797–2.001） | 1.285（0.650–2.203） | 0.606 |
| eq8 | AF3 noMg | 3.563（2.892–5.175） | 3.170（1.784–10.239） | 1.812 |
| eq8 | AF3 Mg | 4.147（2.846–5.452） | 3.358（1.592–10.508） | 1.879 |
| eq8 | Boltz ATP_NMN | 2.886（2.446–3.420） | 4.139（1.248–11.534） | 1.699 |
| eq8 | Boltz ATP_NMN_MG | 2.938（2.432–4.025） | 4.341（1.634–12.195） | 1.671 |

两种模式的 apo 帧都不是同一分布：holo 模型与 `predict-structure` 帧的局部主链更接近（约 1.3–1.5 Å），与 `sample-equilibrium` 帧差异更大（约 2.9–4.1 Å）。这只说明现有 apo 样本中存在与 holo 主链大致相容的构象，不能证明结合不需要诱导契合，也不能给出构象概率；8 帧样本极小。

## 可以说什么，不能说什么

可以支持：

- AF3 在无约束、只用序列和 CCD 配体的条件下，独立地把 ATP 和 NMN 都放进 NMNAT，18/18 WT 模型如此；ATP 链内几何完整，无 severe 冲突，跨配体磷酸端保持 2.2–3.6 Å 的近距离。
- AF3 与 Boltz-2 的 ATP 放置和非磷酸骨架差异较小；NMN 的具体朝向和磷酸端基差异明显高于各自组内 seed 背景，说明 NMN 姿态具有模型依赖性，现有 Boltz-2 的 NMN 朝向不是唯一解。
- AF3 的 Mg 组并没有稳定形成同时贴近 ATP 与 NMN 的金属组织（仅 2/9 同时 ≤2.8 Å），其金属主要贴 ATP 磷酸；这不支持“金属桥接是跨模型稳健特征”。
- AF3 的 1F9A 校准只部分成功：6/9 保留口袋与 ATP 位置，但 Mg–O 配位未被复现，3/9 严重偏离。因此 AF3 不能作为金属配位的仲裁者。
- 现有 apo 帧中，`predict-structure` 模式与 holo 主链的局部差异约 1.4 Å，`sample-equilibrium` 更大；这是相容性描述，不是构象分布或诱导契合证明。

不能支持：

- 不能把 AF3 的 9 个 sample 当作 9 次独立实验重复，也不能把 18/18 或 2/9 当作概率、占有率或自由能结论。
- 不能把 AF3 的 Mg–ATP 2.1 Å 或更钝的 O3A–PA–Onuc 角称为已验证的金属配位、桥接或生产性进攻几何。
- 不能把 AF3 与 Boltz 的一致称为独立实验验证；两者可能共享训练先验、同源序列和建模偏差。
- 不能从 apo 主链相容推断结合机制、催化活性、kcat/Km、NAD⁺ 产量，或对 A1/A2/A3 排序。
- 不能把 1F9A 单链校准当作训练外验证，也不能把其晶体配位当作当前力场或模型的验证标准。

## 对研究判断的影响

本轮把“联合占位”和“金属桥接”进一步分开：

- 联合占位：Boltz-2 与 AF3 都支持 ATP 和 NMN 可同时落在 NMNAT；这比单一模型的先验更稳，但仍只是结构预测层面的相容性。
- 金属桥接：AF3 在多数模型中不把 Mg 同时贴近两种底物，且 AF3 对 1F9A 的 Mg 配位重建不可靠；现有 MD 的持续桥接也不能独立证明自由能稳定性。金属组织仍然是模型和初始条件依赖最强的部分。
- 反应几何：AF3 的 PA–Onuc 距离和角度没有给出可判定的生产性几何；本轮没有预定义生产性阈值，也没有与同输入的单底物姿态作配对比较，因此不能据此否定真实体系催化。

因此，`6629b2f` 继续作为 WT 显式溶剂初始响应结果保留。本轮 AF3 结果支持“双底物联合占位值得继续作为结构假说”，但没有确认一个稳定的金属态，也不支持把 Mg 桥接提升为稳健机制结论，或据此启动 A1/A2/A3 MD、变体活性排序。若后续出现能区分金属组织与初始构象/参数效应的具体问题，可再考虑同一 WT–ATP–NMN 起点下的受控金属比较并评估 Mg²⁺–水–磷酸参数；这不是本轮预设的必需下一步。

## 文件与复现

- 输入：`inputs/`（两个 WT JSON、1F9A JSON、manifest、build.json）
- 身份：`build.json`（构建时输入与代码）、`identity.json`（构建/实际预测/最终分析分列）、`runs.csv`
- 运行：`runs.csv`；`logs/`、`runs/<condition>/` 是本地原始运行目录，不在公开归档中
- 逐模型：`cifs/`（27 个逐样本 CIF.gz，不含重复 best，也不含 confidence JSON）、`cifs/manifest.csv`、`models.csv`、`reaction_candidates.csv`
- 姿态：`pose_pairs.csv`、`pose_summary.csv`、`cross_model_pose_pairs.csv`、`cross_model_pose_summary.csv`
- 参照：`reference_reconstruction.csv`
- apo：`apo_compatibility.csv`、`apo_compatibility_summary.csv`、`apo_identity.json`、`apo_inputs/`（两套未压缩 `topology.cif`/`samples.xtc`、`system.yaml`、`command.txt`；配置的 `apo.sets` 指向这里）
- 汇总：`summary.json`（`confidence` 字段记录可用 summary 数）

复现命令：

```bash
# 在已归档结果目录上复算几何、姿态、参考重建与 apo 比较；不重跑预测
PYTHONPATH=src python -m natvqvae.cli nmnat-af3-analyze --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3
PYTHONPATH=src /home/yhshy/miniconda3/envs/bioemu2/bin/python -m natvqvae.cli nmnat-af3-apo \
  --config config/nmnat_af3.yaml --output data/results/evaluation/natv_nmnat_af3

# 从零重跑预测时，build 必须指向新的空目录；不要把新结果写进已归档目录
PYTHONPATH=src python -m natvqvae.cli nmnat-af3-build --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3_new
PYTHONPATH=src python -m natvqvae.cli nmnat-af3-run --config config/nmnat_af3.yaml \
  --output data/results/evaluation/natv_nmnat_af3_new --wait-gpu
```

公开归档不含 `runs/` 和原 E 盘路径；`nmnat-af3-analyze` 会从 `cifs/` 读取坐标，`nmnat-af3-apo` 会从 `apo_inputs/` 读取小输入，因此上述两条复算命令可直接执行。该复算不包含 pTM/ipTM：这些值只存在于未归档的逐样本 `*_summary_confidences.json`，缺失时留空且不汇总，不按 0 处理。最后一条 apo 命令需要 bioemu2 环境中的 mdtraj；普通几何分析不需要 mdtraj。
