# Bounded NatV generation and structure-informed annotation

This run generated 32 full-length KVP40 NatV variants from two frozen H2
checkpoints and annotated them against the previously audited experimental
template anchors. It is a finite conditional local-editing study, not a
de-novo generator and not a catalytic-activity measurement.

## Declared scope

- Checkpoints: `H2_no_latent/20261081` (SHA256 `6bdecac9f55604a66a84fac0090626bff55b4bf0455e363d201de3cd5c784e06`) and
  `H2_continuous_mmd/20261081` (SHA256 `ce12ef55b45dc38696a34776e35553044d19940db01a9cc1670af9422e8d2b6c`).
- Shared H2 training contract, tokenizer, F-train retrieval database and split.
- Matched streams: seed `20261201`, 8 streams per edit budget, budgets
  [4, 8], 4 sequences per batch, 7 masked-edit stages, temperature
  1.0, T135A forbidden.
- 32 proposals total; the `continuous_mmd` arm draws one latent prior per
  four-sequence trajectory and holds it fixed across all seven stages; the
  `no_latent` arm has no latent branch.
- No training, folding, docking, MD, remote download, wet-lab work or test
  sequence payload reads.

## Inputs and provenance

- H2 contract SHA256: `01d9b47efa22a1c89b704effd5885c797287e23c432c2be93cc95c3b95e49a76`.
- Structural basis: `discovery/natv_pairing_audit/structural_interpretation_basis.json`,
  SHA256 `8cc0577feb96bebeb44bf0eb19432f81485ce00320c3835f7f973f6a0902f7d7`. It remains a
  discovery artifact (`formal_input_manifest=false`); it is used only for
  annotation, not training or candidate selection.
- Run-specific input hashes: `input_manifest.json`.
- Pre-sampling freeze: `pre_sampling_freeze.json`.
- Evaluation-stage amendment: `post_sampling_amendment.json` (4 evaluation-script revision(s); sampling artifacts unchanged: True). The evaluation code was corrected after sampling was frozen; no sampling output was regenerated.
- Formal WP5 retrospective reproduction: `24`
  candidates; maximum |CE difference| `4.44089e-16`;
  CDD rows identical: `True`.

## Sampling validity

- 32/32 proposals are canonical 341 aa, with exactly 4 or 8 substitutions.
- Final edits equal the frozen stream positions; T135A violations: 0.
- Unique sequence hashes: 27/32. The two arms
  produced identical sequences for 4 budget-4 stream pairs and
  1 budget-8 stream pair; duplicates remain in all proposal-level
  denominators.
- Exact novelty against the 2,430 family role table:
  32/32; against the 1,644 F-train
  sequences: 32/32.
- Ordered CDD double-domain pair: 32/32.
- WT blocked-8 ESM2 CE: 1.865023817 (reference
  1.865023817; delta 0).

## Per-arm descriptive results

| source_set | arm | stream_type | edit_budget | n_proposals | n_unique_sequences | n_exact_novel_family2430 | n_CDD_pass | n_CE_below_WT | median_PLM_CE | n_anchor_preserving | n_NMNAT_anchor_perturbed | n_NUDIX_anchor_perturbed | n_T135_changed | n_H236_changed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bounded_new | H2_continuous_mmd | matched | 4 | 8 | 8 | 8 | 8 | 2 | 1.88299 | 5 | 0 | 3 | 0 | 0 |
| bounded_new | H2_continuous_mmd | matched | 8 | 8 | 8 | 8 | 8 | 1 | 1.89536 | 3 | 0 | 5 | 0 | 0 |
| bounded_new | H2_no_latent | matched | 4 | 8 | 8 | 8 | 8 | 1 | 1.88055 | 5 | 0 | 3 | 0 | 0 |
| bounded_new | H2_no_latent | matched | 8 | 8 | 8 | 8 | 8 | 1 | 1.90027 | 3 | 0 | 5 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | adaptive | 4 | 2 | 2 | 2 | 2 | 1 | 1.85692 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | adaptive | 8 | 2 | 2 | 2 | 2 | 1 | 1.89174 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | matched | 4 | 2 | 2 | 2 | 2 | 1 | 1.87142 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | matched | 8 | 2 | 2 | 2 | 2 | 0 | 1.91415 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_S | adaptive | 4 | 2 | 2 | 2 | 2 | 1 | 1.87363 | 1 | 1 | 0 | 1 | 0 |
| formal_WP5_frozen | formal_S | adaptive | 8 | 2 | 2 | 2 | 2 | 0 | 1.88263 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_S | matched | 4 | 2 | 2 | 2 | 2 | 0 | 1.87865 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_S | matched | 8 | 2 | 2 | 2 | 2 | 1 | 1.89016 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_SE | adaptive | 4 | 2 | 2 | 2 | 2 | 0 | 1.88503 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_SE | adaptive | 8 | 2 | 2 | 2 | 2 | 1 | 1.87328 | 2 | 0 | 0 | 0 | 0 |
| formal_WP5_frozen | formal_SE | matched | 4 | 2 | 2 | 2 | 2 | 1 | 1.8742 | 1 | 0 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_SE | matched | 8 | 2 | 2 | 2 | 2 | 0 | 1.90658 | 2 | 0 | 0 | 0 | 0 |

The two new H2 arms share the training contract but differ in architecture and
capacity. These are descriptive finite-sample counts, not a method-superiority
test. Internal ESM2 CE shares the ESM2 prior and is not activity evidence.

## Domain substitution summary

| source_set | arm | edit_budget | domain | candidates_total | candidates_with_substitution | substitutions_total | direct_anchor_substitutions | near_window_substitutions |
|---|---|---|---|---|---|---|---|---|
| bounded_new | H2_continuous_mmd | 4 | LINKER | 8 | 4 | 4 | 0 | 0 |
| bounded_new | H2_continuous_mmd | 4 | NMNAT | 8 | 8 | 13 | 0 | 3 |
| bounded_new | H2_continuous_mmd | 4 | NUDIX | 8 | 8 | 15 | 3 | 3 |
| bounded_new | H2_continuous_mmd | 8 | LINKER | 8 | 6 | 6 | 0 | 0 |
| bounded_new | H2_continuous_mmd | 8 | NMNAT | 8 | 8 | 28 | 0 | 4 |
| bounded_new | H2_continuous_mmd | 8 | NUDIX | 8 | 8 | 30 | 5 | 5 |
| bounded_new | H2_no_latent | 4 | LINKER | 8 | 4 | 4 | 0 | 0 |
| bounded_new | H2_no_latent | 4 | NMNAT | 8 | 8 | 13 | 0 | 3 |
| bounded_new | H2_no_latent | 4 | NUDIX | 8 | 8 | 15 | 3 | 3 |
| bounded_new | H2_no_latent | 8 | LINKER | 8 | 6 | 6 | 0 | 0 |
| bounded_new | H2_no_latent | 8 | NMNAT | 8 | 8 | 28 | 0 | 4 |
| bounded_new | H2_no_latent | 8 | NUDIX | 8 | 8 | 30 | 5 | 5 |
| formal_WP5_frozen | formal_ESM2 | 4 | LINKER | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 4 | NMNAT | 2 | 2 | 4 | 0 | 1 |
| formal_WP5_frozen | formal_ESM2 | 4 | NUDIX | 2 | 2 | 3 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 8 | CTERM | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 8 | NMNAT | 2 | 2 | 9 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 8 | NUDIX | 2 | 2 | 6 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 4 | NMNAT | 2 | 2 | 4 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 4 | NUDIX | 2 | 2 | 4 | 0 | 0 |
| formal_WP5_frozen | formal_ESM2 | 8 | NMNAT | 2 | 2 | 11 | 0 | 1 |
| formal_WP5_frozen | formal_ESM2 | 8 | NUDIX | 2 | 1 | 5 | 0 | 0 |
| formal_WP5_frozen | formal_S | 4 | CTERM | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_S | 4 | NMNAT | 2 | 2 | 5 | 1 | 1 |
| formal_WP5_frozen | formal_S | 4 | NUDIX | 2 | 2 | 2 | 0 | 1 |
| formal_WP5_frozen | formal_S | 8 | LINKER | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_S | 8 | NMNAT | 2 | 2 | 10 | 0 | 1 |
| formal_WP5_frozen | formal_S | 8 | NUDIX | 2 | 2 | 5 | 0 | 2 |
| formal_WP5_frozen | formal_S | 4 | NMNAT | 2 | 2 | 4 | 0 | 0 |
| formal_WP5_frozen | formal_S | 4 | NUDIX | 2 | 2 | 4 | 0 | 1 |
| formal_WP5_frozen | formal_S | 8 | NMNAT | 2 | 2 | 8 | 0 | 0 |
| formal_WP5_frozen | formal_S | 8 | NUDIX | 2 | 2 | 8 | 0 | 2 |
| formal_WP5_frozen | formal_SE | 4 | NMNAT | 2 | 2 | 6 | 0 | 2 |
| formal_WP5_frozen | formal_SE | 4 | NUDIX | 2 | 2 | 2 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 8 | CTERM | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 8 | LINKER | 2 | 1 | 2 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 8 | NMNAT | 2 | 2 | 7 | 0 | 1 |
| formal_WP5_frozen | formal_SE | 8 | NTERM | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 8 | NUDIX | 2 | 2 | 5 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 4 | LINKER | 2 | 2 | 2 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 4 | NMNAT | 2 | 1 | 2 | 0 | 2 |
| formal_WP5_frozen | formal_SE | 4 | NTERM | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 4 | NUDIX | 2 | 2 | 3 | 1 | 1 |
| formal_WP5_frozen | formal_SE | 8 | LINKER | 2 | 1 | 1 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 8 | NMNAT | 2 | 2 | 8 | 0 | 0 |
| formal_WP5_frozen | formal_SE | 8 | NUDIX | 2 | 2 | 7 | 0 | 3 |

## Audited anchor mapping

| domain | wt_position | wt_residue | identity_conserved_across_templates | variable_anchor | bounded_new_candidates_changed | bounded_new_mutations | formal_WP5_frozen_candidates_changed | formal_WP5_frozen_mutations |
|---|---|---|---|---|---|---|---|---|
| NMNAT | 14 | H | True | False | 0 |  | 0 |  |
| NMNAT | 17 | H | True | False | 0 |  | 0 |  |
| NMNAT | 135 | T | True | False | 0 |  | 1 | T135S |
| NMNAT | 138 | R | True | False | 0 |  | 0 |  |
| NUDIX | 232 | G | True | False | 6 | G232A;G232S | 0 |  |
| NUDIX | 236 | H | False | True | 0 |  | 0 |  |
| NUDIX | 247 | R | True | False | 0 |  | 0 |  |
| NUDIX | 248 | E | True | False | 4 | E248G;E248S | 0 |  |
| NUDIX | 251 | E | True | False | 0 |  | 0 |  |
| NUDIX | 252 | E | True | False | 2 | E252K | 0 |  |
| NUDIX | 307 | D | True | False | 0 |  | 1 | D307S |
| NUDIX | 308 | D | True | False | 4 | D308E;D308N | 0 |  |

Direct-anchor candidates: 18 proposals
(16 new + 2 retrospective) but only
15 unique sequences, because some cross-arm proposals are
identical. The complete list is in `structural_analysis/direct_anchor_candidates.csv`.

### NMNAT catalytic environment

Direct substitutions at H14, H17, T135 or R138 map to the audited
ATP-motif/catalysis-adjacent proxies from the 2QJO NAD product state. Other
NMNAT-domain substitutions are mapped by domain only. New candidates with
direct NMNAT anchor changes:
0/32 proposals
(0 unique
sequences); retrospective:
1/24
(1 unique).
The retrospective T135 change is T135S, not T135A. T135 changes: new
0/32; retrospective 1/24.
No kcat, binding, flux or net NAD+ claim follows from these mappings.

### NUDIX hydrolysis

Direct substitutions at G232, H236, R247, E248, E251, E252, D307 or D308 map
to the audited Nudix motif/metal-water network proxies from the 2QJT AMP+Mn
state. H236 is a variable, non-identity-conserved positional anchor; changes
there are mapped perturbations, not identity-conserved anchor losses. New
candidates with direct NUDIX anchor changes:
16/32 proposals
(13 unique
sequences); retrospective:
1/24
(1 unique).
H236 changes: new
0/32; retrospective 0/24.
Whether any substitution raises or lowers hydrolysis is UNKNOWN; no risk
direction is assigned.

## Interpretation limits

- The models perform WT-relative masked local editing with fixed 4/8 edits;
  they are not de novo or learned joint full-length generators.
- The structural basis is a discovery-derived mapping, not a new experimental
  structure; `within +/-4 sequence positions` is a sequence-window proxy, not a
  3D distance.
- No folding, docking, MD, wet-lab assay or experimental net NAD+ measurement
  was performed. Model and CDD scores are compatibility/diagnostic evidence.
- The formal 24 candidates are retrospective, selected by the earlier formal
  protocol; they are not independent new samples.
- Proposal-level counts include five cross-arm duplicate pairs; unique-sequence
  counts are reported alongside them.
- Single-seed finite draws do not support population-level or method-superiority
  claims.

## Decision

The structural correspondence is useful for interpreting where generated
substitutions land, but it does not establish a catalytic benefit. The formal
training population and splits remain unchanged. The structural basis is
recorded in the run-specific `input_manifest.json`; it is not promoted to the
global training input manifest because it did not enter training or candidate
selection. Data expansion remains closed for the current objective.

## Reproduction

```bash
export PATH=/home/yhshy/miniconda3/envs/nad_data/bin:$PATH
export NATV_SOURCE_ROOT=/home/yhshy/git_files/nad_qvae
python experiments/natv_generation_structural/run_generation.py --stage preflight
python experiments/natv_generation_structural/run_generation.py --stage sample
python experiments/natv_generation_structural/run_generation.py --stage evaluate
```

## Files

Retained primary evidence:

- Freeze and provenance: `sampling_manifest.json`, `pre_sampling_freeze.json`,
  `preflight_report.json`, `input_manifest.json`, `post_sampling_amendment.json`.
- Sampling: `proposals.csv`, `sampling_trace.jsonl.gz`,
  `retrieval_receipts.json.gz`, `retrieval_hits_summary.csv`.
- Evaluation: `evaluation/CDD_domains.csv`,
  `evaluation/internal_PLM_scores.csv`, `evaluation/CDD_reproduction_check.csv`,
  `evaluation/evaluation_receipt.json`.
- Annotation: `structural_analysis/structural_interpretation.json`.

`retrieval_hits_summary.csv` is retained because
`post_sampling_amendment.json` hashes it as a frozen sampling artifact.

The evaluation stage also generates `job_status.json`,
`evaluation/CDD_queries.fasta`, `evaluation/CDD_pair_summary.csv` and the
per-candidate/per-anchor tables (`structural_analysis/candidate_level.csv`,
`substitution_level.csv`, `anchor_summary.csv`, `domain_summary.csv`,
`arm_summary.csv`, `direct_anchor_candidates.csv`). These are derived,
regenerable tables and were removed from git in the post-run curation commit;
the counts and direct-anchor records remain in
`structural_analysis/structural_interpretation.json` and this report. Raw
MMseqs outputs under `retrieval/` and raw `CDD.asn1`/`CDD.tsv`/`CDD.log` were
local-only and were removed after the run; rerunning `sample` and `evaluate`
regenerates them from the declared inputs.
