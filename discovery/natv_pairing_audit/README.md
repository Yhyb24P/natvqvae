# NatV 配对核对文件

运行证据、负结果、限制与数据扩充判断见 [dataset_update_decision.md](dataset_update_decision.md)。序列、构建体、坐标及同源关系的对应入口见 [NATV_PAIRING_AUDIT.md](../../docs/NATV_PAIRING_AUDIT.md)。

## 文件与复现

| 脚本 | 输出 |
|---|---|
| `build_local_audit.py` | `audit_summary.json`、序列与结构 registry、残基对应、排除与 Active-G 表 |
| `query_external.py` | `external/rcsb_sequence_search/`、`rcsb_metadata/`、`rcsb_shortlisted_cif/`、`uniprot_raw/`、`alphafold_raw/` 原始缓存与请求记录 |
| `summarize_external.py` | 外部检索汇总、群体重叠、UniProt 候选与 AlphaFold 元数据表 |
| `analyze_incremental_homology.py` | `external/homology/` 同源关系与组分分析 |
| `build_structural_basis.py` | `structural_interpretation_basis.json` |

在本目录执行；执行范围与预算要求见 `../../AGENTS.md`：

```bash
/home/yhshy/miniconda3/envs/nad_data/bin/python build_local_audit.py
NO_PROXY=search.rcsb.org,data.rcsb.org,files.rcsb.org,rest.uniprot.org,alphafold.ebi.ac.uk \
no_proxy=search.rcsb.org,data.rcsb.org,files.rcsb.org,rest.uniprot.org,alphafold.ebi.ac.uk \
/home/yhshy/miniconda3/envs/nad_data/bin/python query_external.py
/home/yhshy/miniconda3/envs/nad_data/bin/python summarize_external.py
/home/yhshy/miniconda3/envs/nad_data/bin/python analyze_incremental_homology.py
/home/yhshy/miniconda3/envs/nad_data/bin/python build_structural_basis.py
```

同源分析依赖 MMseqs2 与 Biopython，历史运行约 95 秒；只读本地既有群体和候选，输出到 `external/homology/`。`NO_PROXY` 是原主机绕过本地代理的设置；重新请求数据库可能改变响应，复核历史结果应使用缓存与记录的校验值。

`source_hashes.json` 记录来源输入；`hashes.json` 记录本目录已提交核对文件的 SHA256。可再生成的 MMseqs 原始输出 `external/homology/raw/*.m8` 及临时文件约 27 MB，仅本地保留，不计入该清单；用于判断的精确重比对已保存在压缩 CSV 中。
