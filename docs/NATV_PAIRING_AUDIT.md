# NatV 序列、构建体、坐标与同源关系核对

参考序列、群体和模型输入边界见 [DATA_PROVENANCE.md](DATA_PROVENANCE.md)。核对运行的范围、检索结果、数据扩充判断与限制见 [dataset_update_decision.md](../discovery/natv_pairing_audit/dataset_update_decision.md)。

## 可信的实验模板

| 实体 | UniProt | 构建体/天然序列 | 标签 | 指定用途 |
|---|---|---:|---:|---|
| 2QJO_1 | Q55928 | 341 / 339 | 2 | NMNAT 产物结合状态（NAD⁺）参照 |
| 2QJT_1 | Q5NHR1 | 352 / 347 | 5 | Nudix 产物与金属状态（AMP + Mn）参照 |
| 2R5W_1 | Q5NHR1 | 352 / 347 | 5 | 同一序列的另一状态 |

2QJT 与 2R5W 不是独立的序列多样性。“全长”指构建体序列；坐标中未解析的
缺口记录在 `construct_observed_audit.csv`。

## 编号与 12 个关键残基

`auth_seq_id` 的偏移随模板而变：2QJO 实体 = auth + 2；2QJT 和 2R5W 实体 =
auth + 5。不能把 2QJO 中编号相同的位点直接当作对应位点：12 个关键残基中
有 11 个会落到不同的氨基酸上。这里采用 Q5NHR1→Q55928 全局比对；对 12 个
关键残基，它与 KVP40→Q55928 直接比对的结果一致。

| 结构域 | WT | Q5NHR1/2QJT auth | 2QJO 天然编号 | 2QJO 构建体 | 2QJT 构建体 | Q55928 中是否同残基 |
|---|---|---|---|---|---|---:|
| NMNAT | H14 | H15 | H18 | 20 | 20 | 是 |
| NMNAT | H17 | H18 | H21 | 23 | 23 | 是 |
| NMNAT | T135 | T139 | T140 | 142 | 144 | 是 |
| NMNAT | R138 | R142 | R143 | 145 | 147 | 是 |
| Nudix | G232 | G234 | G232 | 234 | 239 | 是 |
| Nudix | H236 | E238 | K236 | 238 | 243 | **否** |
| Nudix | R247 | R249 | R247 | 249 | 254 | 是 |
| Nudix | E248 | E250 | E248 | 250 | 255 | 是 |
| Nudix | E251 | E253 | E251 | 253 | 258 | 是 |
| Nudix | E252 | E254 | E252 | 254 | 259 | 是 |
| Nudix | D307 | D307 | D304 | 306 | 312 | 是 |
| Nudix | D308 | D308 | D305 | 307 | 313 | 是 |

E238↔K236 是位置对应，残基本身不同。配体距离及几何检查结果见发现目录的决策记录。

## 对应文件

文件均位于 `discovery/natv_pairing_audit/`：

| 文件 | 核对内容 |
|---|---|
| `sequence_registry.csv` | WT、家族、PDB 实体与模板序列的校验值和角色 |
| `structure_registry.csv`、`dual_domain_scan.csv` | 实验结构群体、配体与双结构域筛选 |
| `construct_observed_audit.csv`、`residue_mapping.csv` | 构建体、标签、逐链未解析缺口及坐标对应 |
| `anchor_mapping_audit.csv` | 关键残基对应、残基身份和配体距离 |
| `structural_interpretation_basis.json` | 汇总结构域、模板、构建体偏移、缺口、配体与关键残基对应 |

文件索引与复现方法见 [发现目录 README](../discovery/natv_pairing_audit/README.md)。
