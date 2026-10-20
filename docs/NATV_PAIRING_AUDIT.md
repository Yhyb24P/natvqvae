# NatV pairing audit and structure references

This page records the current, reproducible sequence–construct–coordinate–
homology audit. The detailed artifacts live under
[`discovery/natv_pairing_audit/`](../discovery/natv_pairing_audit/); they are
not moved into `src/`, `contracts/` or other core directories.

## What was audited

- KVP40 WT: 341 aa, SHA256
  `8804bca153eceb5063f1d8a879d4daeff327964d5d1c9cd693d217fb4f86a7e8`.
- Frozen family population: 2,430 records, 2,430 unique sequences; original
  `training_role_v2` and `split_v2` assignments preserved.
- Experimental corpus: 285 PDB entries / 308 polymer entities.
- External discovery: three RCSB sequence searches, a bounded UniProt
  dual-Pfam query, IPR041750 metadata and AlphaFold metadata for 10
  candidates. No bulk download and no model training/generation/folding.

## Trusted experimental templates

| Entity | UniProt | Construct / native | Tag | Designated use |
|---|---|---:|---:|---|
| 2QJO_1 | Q55928 | 341 / 339 | 2 | NMNAT product-bound NAD proxy |
| 2QJT_1 | Q5NHR1 | 352 / 347 | 5 | Nudix AMP + Mn product/metal proxy |
| 2R5W_1 | Q5NHR1 | 352 / 347 | 5 | alternate state of the same sequence |

2QJT and 2R5W are not independent sequence diversity. “Full-length” refers to
the construct sequence; resolved-coordinate gaps are recorded in
`construct_observed_audit.csv`.

## Numbering and the 12 anchors

`auth_seq_id` offsets are template-specific: 2QJO entity = auth + 2; 2QJT and
2R5W entity = auth + 5. The 2QJO column cannot use same-numbered positions:
11 of 12 would land on a different residue. Use the Q5NHR1→Q55928 global
alignment, which agrees with the direct KVP40→Q55928 alignment for all 12
anchors.

| Domain | WT | Q5NHR1/2QJT auth | 2QJO native | 2QJO construct | 2QJT construct | Identity in Q55928 |
|---|---|---|---|---|---|---:|
| NMNAT | H14 | H15 | H18 | 20 | 20 | yes |
| NMNAT | H17 | H18 | H21 | 23 | 23 | yes |
| NMNAT | T135 | T139 | T140 | 142 | 144 | yes |
| NMNAT | R138 | R142 | R143 | 145 | 147 | yes |
| Nudix | G232 | G234 | G232 | 234 | 239 | yes |
| Nudix | H236 | E238 | K236 | 238 | 243 | **no** |
| Nudix | R247 | R249 | R247 | 249 | 254 | yes |
| Nudix | E248 | E250 | E248 | 250 | 255 | yes |
| Nudix | E251 | E253 | E251 | 253 | 258 | yes |
| Nudix | E252 | E254 | E252 | 254 | 259 | yes |
| Nudix | D307 | D307 | D304 | 306 | 312 | yes |
| Nudix | D308 | D308 | D305 | 307 | 313 | yes |

E238↔K236 is positional, not identity-conserved. All 12 designated-template
ligand proximities pass (NMNAT anchors to 2QJO NAD: 3.611–7.667 Å; Nudix
anchors to 2QJT AMP/Mn: 2.006–4.002 Å). These are geometry checks, not
catalytic measurements.

## External discovery result

RCSB found 28 polymer entities, all already in the frozen corpus; none is a
new qualifying full-length dual-domain structure. The UniProt dual-Pfam query
returned 880 records (378 passing the dual-domain screen, 361 deduplicated
candidates); no candidate has a PDB cross-reference. 266 candidates are
already in the frozen population by exact sequence and 95 are exact-sequence
novel, including 8 phage-associated sequences. These 95 are a discovery pool,
not a validated training addition.

The dataset decision is: **keep the formal population, splits and contracts
unchanged** until new experimental structure/activity evidence or a separate,
pre-registered discovery release exists. See
[`dataset_update_decision.md`](../discovery/natv_pairing_audit/dataset_update_decision.md).

## Evidence boundaries

- 2QJO/2QJT are product/metal pocket proxies; no substrate ternary geometry.
- Anchor preservation and ligand proximity do not establish binding, kcat or
  net NAD⁺ yield.
- The RCSB search is bounded at ≥30% identity; divergent structural homologs
  need a separate search.
- AlphaFold models are predictions and were not folded or scored here.

## Reproduction and provenance

```bash
cd discovery/natv_pairing_audit
python build_local_audit.py
python query_external.py
python summarize_external.py
```

- `source_hashes.json`: 866 input files and corpus tree hash
  `ef61977a312de176cce875f6e7c2774a830d3b45a70eb2397db30b3b2eb1b8c6`.
- `external/query_log.json`: every external request, status and response hash.
- `hashes.json`: SHA256 manifest of the 104-file discovery output tree.

No model training, sequence generation, folding or wet-lab work is implied by
this audit.
