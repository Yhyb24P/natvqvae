# NatV 配对核对文件

运行证据、负结果、限制与数据扩充判断见 [dataset_update_decision.md](dataset_update_decision.md)。序列、构建体、坐标及同源关系的对应入口见 [NATV_PAIRING_AUDIT.md](../../docs/NATV_PAIRING_AUDIT.md)。

## 文件与复现

| 脚本 | 输出 |
|---|---|
| `build_local_audit.py` | `audit_summary.json`、序列与结构 registry、残基对应、排除与 Active-G 表 |
| `build_structural_basis.py` | `structural_interpretation_basis.json` |

在本目录执行；执行范围与预算要求见 `../../AGENTS.md`：

```bash
/home/yhshy/miniconda3/envs/nad_data/bin/python build_local_audit.py
/home/yhshy/miniconda3/envs/nad_data/bin/python build_structural_basis.py
```

`build_local_audit.py` 只读本地群体、只读 `nad_qvae` 结构与输入模板，不依赖已退出的外部原始缓存；`build_structural_basis.py` 读取保留下来的 registry、mapping 与结构对应表。依赖 gemmi、pyhmmer、Biopython 与 pandas。

## 当前外部资产

保留 21 项：

- 3 个实验结构 CIF 与 2QJO/2QJT/2R5W 的 6 个模板元数据；
- `external_summary.json`、`discovery_overlap.json`、`query_log.json`、`source_hashes.json`、`incremental_uniprot_candidates.csv`、`uniprot_candidates.csv`、`incremental_structures.csv`、`alphafold_metadata.csv`；
- `homology/` 下的 `homology_summary.json`、`candidate_homology_assessment.csv`、`component_expansion.csv`、`anchor_conservation_by_position.csv`。

退役载荷、退役脚本与校验快照语义见 [dataset_update_decision.md](dataset_update_decision.md)。
