# NatV 双域候选结构 demo

本目录把已有的 NatV 双域 Boltz-2 预测整理为可复现的结构展示，不新增预测，也不把结构相似性、模型置信度或配体位置解释为酶学活性、稳定性或结合证据。

查看资料：[WT 总览](wt_overview.png)、[A 类比较](a_class_panels.png)、[亲本比较](parent_comparison.png)、[演示视频](demo.mp4)、[Blender 场景](demo.blend)。场景及其 `demo.blend.MNSession` 会话文件一同保存；原始坐标继续保存在各研究的输入及坐标归档目录，不在本目录重复复制。

## 数据范围与指标

- 12 个候选（WT、A1/A2/A3、B1/B2、A1B1/A1B2/A2B1/A2B2/A3B1/A3B2），统计使用每个候选的 `model_0`、`model_1`、`model_2` 三个预测样本，共 36 个模型；展示固定使用 `model_0`。
- 比较表含 23 组比较 × 3 个区域（full/NMNAT/Nudix）= 69 行；其中 11 组为主比较：A1/A2/A3/B1/B2 对 WT（5 组）和 6 个 A+B 候选对各自 parent A（6 组）。其余为补充比较，见 `comparison_summary.csv` 的 `comparison_scope`。
- `comparison_*` 为 3×3=9 个模型对的 Cα RMSD 中位数/最小值/最大值；`self_*` 为同一候选内 3 个模型对的统计。所有 RMSD 均先做最小二乘刚体叠合。模型对不是独立种子、重复实验或置信区间。
- `plddt_median_0_1`、`ptm_median_0_1` 是 Boltz-2 的 0–1 置信度中位数，不用于候选排序或功能判断。数值定义和逐候选结果见 `candidate_structure_summary.csv`、`summary.json` 与 `comparison_summary.csv`。
- 结构域定义为 NMNAT 4–181、linker 182–203、Nudix 204–338；域外残基显示为灰色。

## 图像

### `wt_overview.png` — WT 设计位点

全长 WT（`model_0`）卡通模型；NMNAT 蓝色、linker 金色、Nudix 绿色。标题注明 `labels: WT -> designed`。12 个设计位点用小球和引线标注：小球位于真实 Cα，标签经引线移到结构前方；A1 的 4 个编辑为洋红，A2 的 3 个为橙色，A3 的 3 个为青色，B1 的 E248Q 为红色，B2 的 D308N 为紫色。颜色只表示所属候选组，不表示效应方向或证据强度。左下角为结构域颜色图例。

### `a_class_panels.png` — A1/A2/A3 同帧比较

三个 A 类候选并排显示，均以 WT 的 NMNAT Cα 4–181 做刚体叠合后放入同一坐标框架；场景中没有 WT 参考对象，避免把参考结构误读为额外候选。每个 panel 标注该候选携带的 A 类编辑及 marker 颜色；该图只显示叠合后的静态构象，不提供动力学或柔性证据。

### `nudix_site.png` — Nudix 位点

Nudix 结构域（apo WT）的局部视图，标出 E248Q（B1，红色）、E252 参考位点（白色；不是设计突变）和 D308N（B2，紫色）。图中没有放置 NAD⁺ 或 Mg 配体；同源位点的金属/水解假设尚未在 KVP40 中得到功能验证。

### `nmnat_inset.png` — 联合模型的 NMNAT 局部

来自 `ATP_NMN_MG_unconstrained_WT_seed20261026_model_0.cif` 的单个 Boltz-2 联合模型；只显示 NMNAT 域（灰色）和 ATP/NMN 的 ball-and-stick，标出两个配体。输入包含 1 个 Mg，但该模型把 Mg 放在 Nudix 一侧，位于本视图之外；图没有移动或重新选择 Mg，也不能据此判断 Mg 的催化位置。该模型不是实验复合物，也不代表 productive catalytic state。

### `parent_comparison.png` — A+B 与 parent A

6 个 A+B 候选对各自 parent A 的 full/NMNAT/Nudix Cα RMSD 中位数（每组 9 个模型对）。图中柱高只作描述；模型对不独立，min–max 也不是置信区间。WT 比较和 B-parent 比较保留在 `comparison_summary.csv`。

### `demo.mp4` — WT 转台

960×540、24 fps、120 帧、5 s 的 H.264 视频。相机固定，WT 结构组（含位点 marker）绕竖直轴匀速旋转一周；标题和图例固定在画面内。为保持动画清晰，`label_*` 与 `leader_*` 只在渲染中隐藏，marker 随结构一起旋转。旋转仅用于展示三维形状，不是构象变化模拟。`render_identity.json` 记录 `motion: static camera; 360 deg turntable`。

## 复现与检查

在项目根目录执行（使用新的空输出目录，避免覆盖已交付的 `demo/`）：

```bash
PYTHONPATH=src /home/yhshy/miniconda3/envs/nad_data/bin/python -m natvqvae.cli structure-demo-summary \
  --config config/structure_demo.yaml \
  --output data/results/evaluation/natv_dual_domain_design/demo_reproduction

PYTHONPATH=src /home/yhshy/miniconda3/envs/nad_data/bin/python -m natvqvae.cli structure-demo-render \
  --config config/structure_demo.yaml \
  --output data/results/evaluation/natv_dual_domain_design/demo_reproduction
```

summary 会拒绝已有文件的输出目录；两条命令必须使用同一个新目录。若只是克隆仓库，设计/joint 的原始 CIF 路径可能不存在：仓库跟踪的是 `coordinates/*.cif.gz`（joint 目录另有 `coordinates_manifest.csv`），需按归档恢复或显式修改配置指向已解压文件，再执行上面的命令。测试使用临时解压的 12 个 model_0 设计归档和 1 个 joint 归档，不依赖本机 `data/results` 中的未跟踪 raw 文件。

`structure-demo-render` 会先清除旧的 `render_identity.json` 和旧渲染输出，再调用 Blender 5.2.3 LTS + Molecular Nodes 520.2.0；只有 7 项输出都存在时才写入身份文件。身份文件不包含自身哈希，记录输入、`blender_demo.py` 与各输出的 SHA-256；读取端会校验哈希、模式（full/stills-only）和输出清单，拒绝陈旧或不完整结果。`--stills-only` 可跳过动画，但本目录交付的是 full 模式。

打开 `demo.blend` 可检查固定相机、`WT_Rotation` 空物体和 1–120 帧范围（关键帧在第 1 与第 121 帧）；文件保存于动画状态，`label_*`/`leader_*` 的 `hide_render` 为真。要查看带标签的静态构图，使用 `wt_overview.png`，或在 Blender 中临时恢复这些对象的渲染可见性。

## 证据边界

- 结构为 Boltz-2 预测；没有实验结构、配体结合测量、酶活测定或独立重复。
- Cα RMSD 只描述本次模型对之间的整体几何差异；不能区分功能、稳定性或表达效应。
- pLDDT/pTM 是模型置信度，不是实验置信度，也不能证明候选更优。
- 同源映射、位点距离和序列观察是假设来源；`mutation_map.csv` 中的 caveat 仍适用。
- 本 demo 不评价 QVAE 生成质量，也不把展示图当作新的科研结论。
