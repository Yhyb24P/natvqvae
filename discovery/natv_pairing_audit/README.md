# NatV 配对核对记录

运行证据与数据扩充裁决见 [dataset_update_decision.md](dataset_update_decision.md)；序列、构建体、坐标与同源对应入口见 [NATV_PAIRING_AUDIT.md](../../docs/NATV_PAIRING_AUDIT.md)。

## 当前数据与脚本

- 数据表：`data/processed/`
- 原始结构：`data/raw/structures/`
- 模板元数据：`data/raw/metadata/`
- 复现脚本：`scripts/build_local_audit.py`（默认输出 `data/interim/pairing_audit`）、`scripts/build_structural_basis.py`（默认输出 `data/interim/structural_basis`）
- 本地来源链接：`python scripts/link_inputs.py`

## 历史快照

`hashes.json`、`source_hashes.json`、`external_source_hashes.json`、`query_log.json` 是 `0ea7264` 前的历史快照。
