# KVP40 NatV WT/NatV10/NatV89 独立结构展示

本目录展示湿实验来源的 ePCR 全长克隆 NatV10、NatV89 与 WT 的既有 Boltz-2 model_0 预测结构。它不是 12 条计算设计候选 demo 的一部分，也不新增预测、指标或功能结论。

## 数据与配准

- WT/NatV10/NatV89 均为 341 aa。NatV10 的蛋白改变为 D83V、M162V、L317P、M323K；NatV89 为 I21N、F137Y、I156T、P301S、V320E。NatV89 的 DNA `561c>t` 为 I187I 同义变化，不计蛋白替换。
- 三条序列的 Sanger 依据、完整序列和 SHA 见 [`variants.csv`](../kvp40_epcr_full_length_structures/variants.csv)、[`sanger_verification.json`](../kvp40_epcr_full_length_structures/sanger_verification.json)、[`build.json`](../kvp40_epcr_full_length_structures/build.json)。Sanger 记录支持序列身份；菌株或蛋白样品是否仍可用需按实际库存核对。
- 本展示直接使用 [`kvp40_epcr_nmnat_environment/coordinates/`](../kvp40_epcr_nmnat_environment/coordinates/) 中已对齐的 model_0 PDB。它们以 WT model_0 为参考，按 NMNAT 4–181 的 Cα 做刚体叠合；NMNAT 4–181 CA RMSD 为 NatV10 1.108 Å、NatV89 1.294 Å。没有逐域移动坐标。
- 该配准只约束 NMNAT 4–181；Nudix 和域间相对位置会随配准带入位移。叠合图不是各域独立 RMSD，也不是构象变化或功能比较。
- 展示使用每个克隆的 1 个 model_0；完整结构比较报告中的 9 个模型是 WT/NatV10/NatV89 各 3 个 diffusion samples，不是 9 个独立实验重复。

## 图

### `clone_overview.png`

三 panel 同向、同尺度的全长卡通：WT 为参考、不带突变标记；NatV10 标出 4 个编辑位点，NatV89 标出 5 个编辑位点。域色为 NMNAT 蓝、linker 金、Nudix 绿，域外为灰。位点文字位于各 panel 的左右空白区，leader 连到真实 Cα 的二维投影位置；白色描边圆点标出投影位置。二维投影可能显示原本被 ribbon 遮住的真实 Cα，因此只表示序列位置，不表示表面暴露、侧链朝向或相互作用。

### `clone_overlay.png`

WT、NatV10、NatV89 的 model_0 全长叠合，分别用灰、橙、品红区分。三者在共同 WT model_0 的 NMNAT 4–181 框架中显示；Nudix 的位置会随该配准带入相对位移。该图不提供各域独立 RMSD、结合姿态或动力学信息。

### `mutation_sites.csv`

列出每个克隆的编辑、WT 编号、域和 aligned PDB 中的 CA 坐标；用于把图中的投影标签对应回序列位置。

## 能说明与不能说明

- 能说明：三条 341 aa 单体在当前预测和共同配准框架中的整体与局部 Cα 几何关系；局部 Cα 位移高于同位置自基线的位点见[全长结构比较报告](../kvp40_epcr_full_length_structures/REPORT.md)。
- 不能说明：这些差异是否改变催化、稳定性、表达、底物亲和力或 NAD⁺ 净积累；不能据此给 NatV10/NatV89 排序，也不能把多突变克隆的差异归因于单个位点。
- 本展示不使用湿实验 A570 读数。现有材料未提供原始板读、校准、WT/未诱导/无酶对照和足够重复信息；当前 A570 汇总只有历史终点吸光信号，不能确认净 NAD 变化或作功能排序；详见 [`candidate_handoff.md`](../natv_dual_domain_design/demo/candidate_handoff.md)。

## 文件与复算

- `render_input.json`、`render_identity.json`：输入、脚本和输出 SHA-256；身份文件不包含自身。
- `clone_overview.png`、`clone_overlay.png`、`mutation_sites.csv`：本目录的正式产物。
- 结构细节与指标：[全长结构比较](../kvp40_epcr_full_length_structures/REPORT.md)、[NMNAT 环境比较](../kvp40_epcr_nmnat_environment/REPORT.md)、[RMSD 汇总](../kvp40_epcr_full_length_structures/rmsd_summary.png)。

在项目根目录使用新的空输出目录复算：

```bash
PYTHONPATH=src /home/yhshy/miniconda3/envs/nad_data/bin/python -m natvqvae.cli epcr-demo \
  --config config/epcr_clone_demo.yaml \
  --output data/results/evaluation/kvp40_epcr_clone_demo_reproduction
```

`epcr-demo` 需要 PyMOL 和 matplotlib；输出目录非空时拒绝覆盖。身份文件记录输入、脚本和两幅图的 SHA-256；正式目录不包含原始 CIF/大坐标副本。
