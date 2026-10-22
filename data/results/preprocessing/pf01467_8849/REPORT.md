# PF01467-only 8,849 recovery and migration layers

Inputs: old stage3, `domain_profile_manifest_v4.csv`, `corpus_boundary_manifest.jsonl`; current train/validation raw population and exact splits. Only the 8,849 stage3-only PF01467 IDs were extracted; test is IDs only. CPU 8 threads, 39.0 s; no training, generation, download or folding.

## Disposition

| destination | records | meaning |
|---|---:|---|
| main NatV task | 0 | all are PF01467-only; no ordered NMNAT–Nudix dual domain |
| aux full-context | 486 | WT target coverage ≥0.8 by mapped target residues, identity ≥0.3, ≥10 reliable positions |
| aux local core | 5,991 | PF01467 local/core fragment; not a complete NatV domain |
| rejected | 2,372 | no reliable WT mapping under the same criteria |

All 8,849 are in `records.csv` with source header/accessions, overlapping annotation flags, old domain boundaries, WT alignment coverage/ambiguity, destination and reason. Unique full sequences: 8,849; unique PF01467 core domains: 5,970; unique auxiliary domain sequences: 4,164.

## Joint clustering

Old domains/full proteins and current train/validation domains/full proteins were clustered together with MMseqs `cov-mode 0`, coverage 0.8, connected components. Cluster IDs are from the common clustering.

| layer | identity | total | old-only | both | current-only | old classes |
|---|---:|---:|---:|---:|---:|---|
| domain | 70% | 743 | 624 | 0 | 119 | new 624 |
| domain | 50% | 247 | 196 | 1 | 50 | new 196, train-overlap 1 |
| domain | 30% | 56 | 53 | 1 | 2 | new 53, train-overlap 1 |
| full | 70% | 887 | 788 | 0 | 99 | new 788 |
| full | 50% | 297 | 264 | 0 | 33 | new 264 |
| full | 30% | 97 | 94 | 1 | 2 | new 94, mixed-current 1 |

At 70% there is no joint cluster containing both old and current sequences. At 50% one domain cluster overlaps current training (7 old records); at 30% one domain cluster overlaps current training (2,298 old records). Full-length clustering has no overlap at 70/50 and one mixed-current cluster at 30%. These are threshold-dependent: absence at 70% is not evidence of no remote homology.

## Auxiliary split

The auxiliary split uses 70% domain clusters: same cluster stays in one split; clusters with current train/validation members are excluded from aux train; only new clusters are split 80/20 by cluster with seed 20261022.

- Unique auxiliary sequences: 3,816 train / 348 val; record rows: 6,090 train / 387 val; 2,372 excluded.
- Weights are cluster-balanced and normalized to mean 1 within each split; no 70% cluster appears in both splits.
- At 50/30% the old domain records include current-train overlaps (7 and 2,298 records); the full layer has one mixed-current cluster at 30%. The auxiliary split uses 70% clusters, so it is not a fully independent test of all remote homology. Test was not sequence-searched; no claim of complete test independence is made.

## Increment

Reliable WT-mapped substitutions, compared with current 1,644 training domains:

- 3,018 WT-position/residue events; 1,414 new (not observed in current train), 1,208 new supported by ≥2 old clusters; 1,604 support increases; 0 events at current-unmapped positions.
- Full-context double-site pairs: 483,993 observed pairs, 394,587 not observed in current train, 109,110 new pairs supported by ≥2 old clusters. Only the top 50 are written; the large pair counts are combinatorial and low-support, not a transfer effect.
- Event counts are descriptive, not activity or transfer effects.

## Files

- `records.csv`: all 8,849 with source, flags, boundary/alignment, destination and reason.
- `full_length.fasta`, `pf01467_core_domains.fasta`, `aux_domains.fasta`: deduplicated by sequence SHA.
- `aux_masks.json`: reliable WT positions for unique auxiliary sequences.
- `aux_split.csv`: unique auxiliary sequences, split, cluster and weight.
- `joint_domain_clustering.fasta`, `joint_full_clustering.fasta`: clustered inputs.
- `new_single_events.csv`, `new_double_events_top.csv`, `summary.json`: event summaries and run identity.

Protocol limits: PF01467 boundary is not the NatV CDD domain; `local_core` is not a complete domain; 30–35% identity is twilight-zone; `max_optimal=16` treats records with more optimal alignments as having no reliable positions; annotation flags are preserved, not used as hard exclusions.
