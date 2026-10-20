# NatV pairing audit and bounded incremental discovery

Status: completed 2026-10-20 on branch `research/natv-pairing-audit`.
This work performs a local sequence–construct–coordinate–homology audit and a
bounded external metadata search. It does **not** train, generate sequences,
fold structures, run wet-lab work, modify the read-only `nad_qvae` source, or
change the formal NatV population, splits or core contracts.

## Scope and invariants

- Target: KVP40 NatV, 341 aa, SHA256
  `8804bca153eceb5063f1d8a879d4daeff327964d5d1c9cd693d217fb4f86a7e8`.
- Domains (CDD-derived, WT-relative): NMNAT 4–181, linker 182–203,
  Nudix 204–338; N-terminal 1–3 and C-terminal 339–341.
- Read-only sources: `nad_qvae` family release, 285-PDB experimental corpus,
  active-G manifest, pinned inputs under `inputs/`.
- The 2,430-sequence family population and its `training_role_v2` /
  `split_v2` assignments are preserved as-is. Annotation-conflict quarantine
  is **not** an experimental negative label.
- External discovery is metadata-first: three RCSB sequence searches, RCSB
  metadata for their union, coordinates only for dual-domain candidates, two
  bounded UniProt queries, and AlphaFold metadata for at most 10 shortlisted
  sequences. Raw responses are retained.
- All new assets are written under `discovery/natv_pairing_audit/`; existing
  uncommitted repository changes are untouched.

## The three sequence layers

| Layer | Object | Source | Consequence |
|---|---|---|---|
| Native family sequence | 2,430 records | `natv_vnext_family_v2` | population/split identity, not a structure |
| PDB construct/SEQRES | entity sequence | RCSB entity/PDB | may contain expression tag and cloning artifacts |
| Observed coordinates | resolved residues | CIF `label_seq_id` | may miss disordered ranges |
| UniProt reference | Q55928 / Q5NHR1 | SIFTS/entity alignment | numbering differs from PDB auth numbering |
| KVP40 correspondence | cross-species alignment | this audit | not a SIFTS claim |

`auth_seq_id` offsets are template-specific: 2QJO entity = auth + 2; 2QJT and
2R5W entity = auth + 5. Treating same-numbered positions as equivalent across
KVP40, Q5NHR1 and Q55928 would be wrong.

## Local audit results

### 1. Family population and splits

| Quantity | Value |
|---|---:|
| Population records | 2,430 |
| Unique sequences | 2,430 |
| Sequence-ID/hash mismatches | 0 |
| FASTA/parquet sequence mismatches | 0 |
| `training_role_v2`: training | 1,644 |
| `training_role_v2`: annotation-conflict quarantine | 249 |
| `training_role_v2`: validation | 227 |
| `training_role_v2`: test | 231 |
| `training_role_v2`: old sealed component | 75 |
| `training_role_v2`: excluded noncanonical/domain | 4 |
| `split_v2`: training / validation / test | 1,879 / 235 / 235 |
| `split_v2`: old sealed / excluded | 77 / 4 |
| Legacy split: training / validation / reference / test / excluded | 488 / 309 / 71 / 6 / 1,556 |
| Active-G rows | 1,871 |
| Active-G rows missing from population | 0 |
| Population members not in active-G | 559 |

The legacy and v2 counts are different objects and are reported separately.
No role was reassigned by this audit. `exclusions.csv` lists 633 records:
249 annotation-conflict quarantine, 75 old sealed, 4 noncanonical/domain, and
305 non-qualifying structure-corpus entities.

### 2. Experimental structure corpus

285 PDB entries contain 308 polymer entities. Exactly **3** entities pass the
dual-Pfam + dual-HMM + domain-order + length ≥300 screen:

| Entity | UniProt | Construct length | Native length | Tag | Observed residues | Ligands (all chains) |
|---|---|---:|---:|---:|---:|---|
| 2QJO_1 | Q55928 | 341 | 339 | 2 | 995 | APR, NAD, POP, SO4 |
| 2QJT_1 | Q5NHR1 | 352 | 347 | 5 | 684 | AMP, MN |
| 2R5W_1 | Q5NHR1 | 352 | 347 | 5 | 689 | CL, MG |

2QJT and 2R5W are the **same UniProt sequence** in different experimental
states, not independent sequence diversity. The corpus therefore provides
two unique full-length sequences, not three.

Observed-coordinate gaps:

- 2QJO: chain A missing 304–309 and 339–341; chain B missing 1–5;
  chain C missing 1–5, 303–309 and 340–341.
- 2QJT: chain A missing 1–5, 115–118 and 350–352; chain B missing 1–5 and
  350–352.
- 2R5W: chain A missing 1–5 and 350–352; chain B missing 1–5 and 351–352.

“Full-length” here refers to the construct sequence. No single chain resolves
every residue, so residue-level work must use the recorded gaps rather than
assume complete coordinates.

### 3. Anchor mapping audit

Twelve curated anchors were checked in three ways: WT sequence identity,
Q5NHR1 template position, and global KVP40↔Q5NHR1 alignment. The 2QJO column
additionally requires a Q5NHR1→Q55928 global alignment because same-numbered
residues are not homologous across species.

| Domain | WT | Q5NHR1/2QJT auth | 2QJO native (Q55928) | 2QJO construct | 2QJT construct | Identity conserved in Q55928 |
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

Results: 12/12 mapping PASS, 12/12 designated-template ligand proximity PASS.
For 2QJO, 11 of 12 same-number positions would land on a different residue;
the alignment-based column is the valid one and agrees with the direct
KVP40→Q55928 alignment for all 12 anchors. E238↔K236 is the sole
identity-conserved mismatch: the Nudix anchor correspondence is positional,
not sequence-identical, across templates.

Designated-template heavy-atom minimum distances:

- 2QJO NMNAT anchors to NAD: 3.638, 3.611, 7.667 and 6.775 Å
  (two in the ≤4.5 Å first shell, all within ≤8 Å).
- 2QJT Nudix anchors to AMP/Mn: 2.142, 3.970, 4.002, 2.167, 3.922, 2.006,
  3.445 and 2.212 Å (all within ≤4.5 Å).
- 2R5W is an alternate state of the same sequence: 2.201, 4.655, 4.227, 2.083,
  3.983, 2.182, 4.158 and 2.321 Å (seven within ≤4.5 Å; E238 is 4.655 Å).

The 2QJO Nudix anchors are 23–33 Å from the NMNAT-site NAD; the designated
Nudix template is 2QJT. No composite score
is computed, and none of these distances is a catalytic measurement.

### 4. Bounded external incremental discovery

RCSB sequence search used three seeds (KVP40, Q55928, Q5NHR1; ≥30% identity,
E ≤ 1). The union is 28 polymer entities:

- 28/28 are already in the frozen 285-PDB corpus; **0 incremental**.
- 3/28 are the known full-length dual-domain entities (2QJO_1, 2QJT_1, 2R5W_1).
- 0 new qualifying full-length dual-domain experimental structures.

The remaining 25 are single-domain or otherwise non-qualifying homologs
(e.g., 3GZ5/3GZ6/3GZ8 are Shewanella NrtR structures, not NatV).

UniProt dual-Pfam query (`xref:pfam-PF01467 AND xref:pfam-PF00293`):

| Quantity | Value |
|---|---:|
| Records returned | 880 |
| Reviewed (Swiss-Prot) | 1 |
| Pass dual-HMM/full-length screen | 378 |
| Deduplicated candidates (length 280–450; known templates excluded) | 361 |
| Candidates with a PDB cross-reference | 0 |
| Candidates already in family population by exact sequence | 266 |
| Candidates absent by exact sequence | **95** |
| New candidates from phage hosts | 8 |

The 266 existing candidates span the frozen roles: 179 training, 44
annotation-conflict quarantine, 20 validation, 16 test, 7 old sealed. The 95
new sequences have no experimental structure and no measured activity; 8 are
phage-associated (Pectobacterium phage vB_PcaM_CBB, Serratia phage BF,
Sphingobium phage Lacusarx, Burkholderia phages BcepSauron/BcepSaruman,
Stenotrophomonas phages Mendera/Moby, Colwellia phage 9A). They are a
sequence-only discovery pool, not a validated template set.

The InterPro IPR041750 query returned 34 records (17 pass the dual-domain
screen; 26 are already in the family; 2 have PDB cross-references, both known
templates). IPR041750 is not a dual-domain-specific marker and includes
single-domain NMNAT entries, so it is supporting evidence only.

AlphaFold metadata was queried for 10 candidate accessions: 7 have models,
3 returned 404. Among the 95 new candidates, only 5 were queried (3 models,
2 without a model); the other 90 were not queried. Predictions are not
experimental structures, were not folded here, and were not structurally
scored against the templates.

**External result:** no new qualifying full-length experimental NatV
structure was found outside the frozen corpus. This is a valid negative
result, not a claim that no such structure exists anywhere.

## Files

| Path | Content |
|---|---|
| `build_local_audit.py` | local audit implementation / reproduction entry point |
| `query_external.py` | bounded external query implementation |
| `summarize_external.py` | overlap/new-candidate summarization |
| `sequence_registry.csv` | WT, family, PDB-entity and template sequences with hashes/roles |
| `structure_registry.csv` | 285-PDB corpus metadata and ligand inventory |
| `dual_domain_scan.csv` | 308-entity dual-domain screen |
| `residue_mapping.csv` | SIFTS/entity/construct/observed residue mapping |
| `construct_observed_audit.csv` | construct length, tag, missing ranges, observed residues |
| `anchor_mapping_audit.csv` | 12-anchor mapping, identity and ligand-distance audit |
| `exclusions.csv` | preserved exclusion/quarantine records for family and corpus |
| `active_g_registry.csv` | active-G membership and family overlap |
| `audit_summary.json` | machine-readable local audit summary |
| `source_hashes.json` | SHA256 of 866 pinned/local input files and corpus tree hash |
| `external/incremental_structures.csv` | 28 RCSB union entities; all in corpus, none incremental |
| `external/uniprot_dual_pfam_all.csv` | 880 UniProt dual-Pfam records |
| `external/uniprot_candidates.csv` | 361 deduplicated candidates with sequences |
| `external/incremental_uniprot_candidates.csv` | 95 sequences absent from the frozen population |
| `external/uniprot_ipr041750_all.csv` | 34 IPR041750 records |
| `external/alphafold_metadata.csv` | 10 queried accessions; 7 models |
| `external/discovery_overlap.json` | overlap and incremental counts |
| `external/external_summary.json` | external query summary and negative-result flag |
| `external/query_log.json` | all HTTP requests, status, bytes and SHA256 |
| `external/source_hashes.json` | SHA256 of external raw responses and derived tables |
| `hashes.json` | SHA256 manifest of this discovery output tree |

Raw caches: `external/rcsb_sequence_search/`, `external/rcsb_metadata/`,
`external/rcsb_shortlisted_cif/`, `external/uniprot_raw/`,
`external/alphafold_raw/`.

## Reproduction

From `discovery/natv_pairing_audit/`:

```bash
/home/yhshy/miniconda3/envs/nad_data/bin/python build_local_audit.py
NO_PROXY=search.rcsb.org,data.rcsb.org,files.rcsb.org,rest.uniprot.org,alphafold.ebi.ac.uk \
no_proxy=search.rcsb.org,data.rcsb.org,files.rcsb.org,rest.uniprot.org,alphafold.ebi.ac.uk \
/home/yhshy/miniconda3/envs/nad_data/bin/python query_external.py
/home/yhshy/miniconda3/envs/nad_data/bin/python summarize_external.py
```

The `NO_PROXY` setting is environment-specific; it avoids a local proxy that
stalled large RCSB downloads. Cached metadata and CIFs make reruns idempotent
apart from refreshed database responses.

## Evidence boundaries

- The 2QJO and 2QJT ligand states are product/metal pocket proxies. No
  NMN+ATP or ADPR substrate ternary geometry is claimed.
- Ligand proximity and anchor preservation are structural compatibility
  evidence, not binding affinity, catalytic rate or net NAD⁺ production.
- The external RCSB search is limited to ≥30% identity; divergent structural
  homologs would need a separate structural search.
- The 95 new UniProt sequences are exact-sequence novel only. No
  identity/redundancy clustering was performed, so they are not 95
  independent additions.
- The 95 new candidates have no experimental structures or activity labels.
  Their AlphaFold coverage is incomplete by design (bounded query).
- Database results reflect the 2026-10-20 snapshot; counts will drift.
- Structural correspondence must not be reported as catalytic activity.
