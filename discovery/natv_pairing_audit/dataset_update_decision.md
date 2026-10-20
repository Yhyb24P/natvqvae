# Dataset update decision

Date: 2026-10-20. Branch: `research/natv-pairing-audit`.
Scope: local NatV pairing audit plus bounded external discovery.

## Decision

**Keep the formal NatV dataset unchanged.**

- No records are added to or removed from `natv_vnext_family_v2`.
- `training_role_v2`, `split_v2`, the legacy split, the 1,644 core-train
  population and the active-G manifest keep their current assignments.
- No core input contract, WT identity, domain interval or evidence-ladder
  definition is changed.
- `main` is not merged or rewritten. This work is confined to the discovery
  directory on the research branch.

The audit produced **no new experimentally measured activity data and no new
qualifying full-length experimental structure**. The correct update is
therefore a provenance/mapping correction and a discovery pool, not a
training-population change.

## What the audit established

1. The frozen population is internally consistent: 2,430 records, 2,430
   unique sequences, zero sequence-ID/hash mismatches and zero
   FASTA/parquet mismatches. Legacy and v2 split counts are distinct objects
   and were not conflated.
2. Only three experimental polymer entities in the 285-PDB corpus are
   full-length dual-domain NatV constructs: 2QJO_1 (Q55928), 2QJT_1 (Q5NHR1)
   and 2R5W_1 (Q5NHR1). 2QJT and 2R5W are the same sequence in different
   experimental states, so the corpus has two unique full-length sequences.
3. The curated 12-anchor table is numerically consistent with Q5NHR1/2QJT
   auth numbering and with the KVP40 GenBank translation. The 2QJO column
   requires a cross-species Q5NHR1→Q55928 alignment; same-numbered positions
   are invalid. All 12 anchor mappings and all 12 designated-template ligand
   proximities pass under that mapping.
4. E238 in the 2QJT template corresponds to K236 in Q55928, not E. The Nudix
   anchor correspondence is positional, not identity-conserved, at that site.
   This caveat must travel with any use of the anchor table.
5. External RCSB sequence search found 28 polymer entities, all already in
   the frozen corpus. None is a new qualifying full-length structure. This is
   a valid negative result at the ≥30% identity search threshold.
6. The bounded UniProt dual-Pfam query found 880 records, 378 passing the
   dual-domain HMM screen, and 361 deduplicated candidates. None has a PDB
   cross-reference. 266 are already in the frozen population by exact
   sequence; 95 are exact-sequence novel, including 8 phage-associated
   sequences. These are sequence-only candidates without experimental
   structures or activity measurements.
7. The 95 candidates were then analysed under the same 70/80 component
   predicate used to build the frozen split (exact local identity ≥0.70 and
   both alignment spans ≥0.80 of length). They do increase the operational
   sequence graph from 142 to 175 components, but they do not provide a
   validated functional family expansion; details below.

## Incremental homology analysis of the 95 sequences

All 95 are unique, unreviewed TrEMBL entries with no PDB cross-reference and
no measured activity. Exact local re-alignment against the 2,430-sequence
population gives:

| Quantity | Value |
|---|---:|
| Candidates | 95 |
| Nearest family identity: median | 69.4% |
| Nearest family identity: IQR | 52.5–92.9% |
| Nearest family identity: range | 21.6–100% |
| Candidates ≥90% identity to a family sequence | 26 |
| Candidates ≥70% identity | 47 |
| Candidates <70% identity | 48 |
| Candidates with a qualifying 70/80 component | 47 |

For comparison, 2,359 of the 2,430 frozen sequences have a non-self ≥70%
edge, with median nearest-neighbour identity 99.45%. The candidates are
therefore mostly distant or redundant relative to the frozen population.

Under the frozen 70/80 component predicate:

- 45 candidate clusters form.
- 12 clusters (47 candidates) join 12 existing components; 33 new components
  (48 candidates) are added.
- There are no bridges between existing components and no component merges.
- Component count goes from 142 to 175 (+33).
- The 33 new components contain two 6-member clusters, five pairs and 26
  singletons; only 7 contain more than one candidate.

Split integrity is the decisive constraint. Of the 47 candidates that join an
existing component, 34 would join a **test** component, 8 a training
component and 5 a validation component under the frozen assignments. Six
candidates are 100% identical to a family sequence over the qualifying
aligned span while differing elsewhere (usually outside that span), so
exact-sequence novelty is not family-level novelty. Adding the 95 would
require re-clustering and re-splitting the frozen population and would
change test/validation membership, invalidating the current evaluation
baseline.

Anchor conservation was mapped through the WT and Q5NHR1 reference domains.
Self-checks map 12/12 anchors for both references. Excluding the
non-conserved anchor 6, 51 candidates retain 11/11 anchors, 24 retain 10/11,
8 retain 9/11 and 12 retain fewer than 9/11; 26 anchor positions are
unmapped. Per-position matches are high for R249 (95/95), E254 (93/95), H15
(92/95) and G234 (91/95), lower for D307 (66/95, with 22 Ser substitutions)
and E238 (49/95, expected because WT is H236 at this position). Anchor
recovery degrades with sequence distance and is not functional evidence.

**Conclusion for the data question:** the 95 expand the operational
sequence-component graph, but not the *validated* family distribution:
they add no experimental structure, no measured activity, no reviewed
annotation, and no functionally characterised member; most new components
are singletons, and a quarter of the candidates are near-duplicates. There is
no clear gain for the net-NAD⁺ objective. **Data expansion is ended for the
current objective.** The 95 remain a traceable discovery pool only.

## Trusted experimental proxy set

Until new evidence arrives, the only trusted full-length experimental
template set is:

| Template | Use | Restriction |
|---|---|---|
| 2QJO | NMNAT domain, product-bound NAD proxy | Q55928; use cross-species mapping from Q5NHR1; tag length 2 |
| 2QJT | Nudix domain, AMP + Mn product/metal proxy | Q5NHR1; tag length 5; no ADPR substrate geometry |
| 2R5W | alternate state of the same Q5NHR1 sequence | Mg/Cl state, not an independent sequence |

These templates support coordinate hypotheses and residue mapping. They do
not establish substrate binding, catalysis, catalytic efficiency or net NAD⁺
production. The ligand-distance results are geometry checks, not activity
predictions.

## Incremental discovery pool

`external/incremental_uniprot_candidates.csv` contains the 95 sequences absent
from the frozen population by exact SHA256. It is a **discovery pool**, not a
new training set:

- No experimental structure and no measured activity for any of the 95.
- Only 5 of the 95 were queried for AlphaFold metadata (3 models, 2 without a
  model); the other 90 were not queried. AlphaFold models are predictions.
- Exact-sequence novelty is not independence. Under the 70/80 component
  predicate the 95 form 45 clusters, 47 join existing components and 48 fall
  into 33 mostly singleton new components. 26 are ≥90% identical to a
  frozen sequence over the qualifying span and 6 are 100% locally identical.
  They must not be counted as 95 independent family additions.
- The 8 phage-associated sequences are the most relevant to the KVP40 phage
  context, but relevance is not evidence of NatV function or improved NAD⁺
  yield; 4 join existing components and 4 form new components.

Recommended handling: retain the CSV, the homology assessment and their
hashes as a discovery artifact; do not assign roles, do not merge into
training/validation/test, and do not use the sequences as templates without
an experimental or carefully qualified predicted-structure assessment.
Further sequence-only expansion is not planned for the current objective.

## Conditions that would justify a future dataset update

A future update should be a separate, pre-registered change with its own
scope and budget. At minimum it needs one of:

1. A new experimental full-length dual-domain NatV structure outside the
   frozen corpus, with UniProt/construct/observed-coordinate correspondence
   audited as in this work and its ligand state identified.
2. New experimentally measured NMNAT/Nudix/net-NAD⁺ data that defines a
   phenotype or assay endpoint, not a model score.
3. A deliberate discovery-only release that adds sequences with provenance,
   redundancy clustering, split-integrity checks and explicit separation from
   the formal training population.

Predicted structures may support prioritization, but they cannot satisfy
condition 1 or 2 by themselves.

The 95-sequence analysis does not satisfy condition 3 either. Provenance is
present, but the redundancy and split-integrity checks show that a naive
discovery-only merge would change test/validation membership (34 of the 47
component-joining candidates would join test components) and would import 26
near-duplicates. A future discovery release would need its own clustering and
split policy before it could be used, and it would still carry no activity
labels.

## Scientific gain and limits

The gain is a corrected, reproducible correspondence audit, a bounded
negative discovery result and a quantified homology-increment result:

- the 12-anchor table is now tied to the correct Q5NHR1/2QJT auth numbering,
  with an explicit cross-species route for 2QJO and an explicit E238/K236
  caveat;
- the corpus is shown to contain only two unique full-length experimental
  sequences (three entities/states), not three independent sequences;
- external RCSB discovery finds no new qualifying full-length experimental
  structure at the declared search threshold;
- 95 exact-sequence-novel UniProt candidates (8 phage-associated) are retained
  as a traceable discovery pool;
- under the frozen 70/80 predicate the 95 add 33 operational components
  (142→175; 47 candidates join existing components, 48 form new ones), with
  no bridges or merges; the added components are mostly singletons and the
  set includes near-duplicates, so this is sequence-level rather than
  validated functional expansion;
- anchor conservation is quantified against both references, with self-checks
  at 12/12 and a large drop at the non-conserved E238/H236 site, so anchor
  loss can be separated from the known reference difference.

The limits are equally important: no new catalytic measurement, no new
experimental template, no reviewed functional annotation for any of the 95,
no structural assessment of their predicted models, and no taxonomic table
for the frozen population. Anchor mapping at ~35% identity is near the
twilight zone; MMseqs search is not exhaustive. These limits are why the
formal dataset is not updated.

## Recommended next actions (not executed)

1. Sequence-only expansion is closed for the current objective. If a future
   discovery release is considered, first run a bounded structural assessment
   of the 3 available AlphaFold models and optionally fetch models for the 90
   unqueried candidates. Report confidence, domain arrangement and anchor
   geometry as predictions, with a separate clustering/split policy.
2. For the 8 phage-associated new sequences, search experimental structures
   and literature explicitly; a phage context alone is not a functional
   annotation.
3. If the negative RCSB result needs to be strengthened, run a
   structure-based search for divergent homologs (fold/structural alignment),
   not only a ≥30% sequence-identity search.
4. Keep 2QJO/2QJT/2R5W as the experimental template set, with the E238/K236
   and construct-numbering caveats, until new experimental evidence exists.

## Non-actions in this round

- No model training, sequence generation, folding or wet-lab execution.
- No writes to `/home/yhshy/git_files/nad_qvae`; it remains read-only.
- No change to `main`, the formal population, splits, contracts or WT hash.
- No new external query for the homology analysis; it uses the local frozen
  family sequences and the 95 already-fetched candidate sequences.
- The audit and homology outputs are committed only on
  `research/natv-pairing-audit`; nothing is merged to `main`.

## Verification artifacts

- `audit_summary.json` — local audit counts and limitations.
- `source_hashes.json` — 866 pinned input files; corpus tree SHA256
  `ef61977a312de176cce875f6e7c2774a830d3b45a70eb2397db30b3b2eb1b8c6`.
- `external/external_summary.json` — external counts and `negative_result`.
- `external/discovery_overlap.json` — population overlap and new-candidate
  counts.
- `external/homology/homology_summary.json` — nearest-neighbour,
  component-expansion and anchor-conservation results, with limitations.
- `external/homology/candidate_homology_assessment.csv` — per-candidate
  identity, component membership, split proximity and anchor recovery.
- `external/homology/component_expansion.csv` — cluster membership and
  existing-component joins.
- `external/homology/anchor_conservation_by_position.csv` — per-anchor
  substitution counts for both references.
- `external/query_log.json` — HTTP status/size/hash for every external request.
- `hashes.json` — SHA256 manifest of the committed discovery output tree;
  regenerable MMseqs raw output under `external/homology/raw/` is excluded and
  documented in the discovery README.

All structural statements above are conditional coordinate/annotation
evidence, not catalytic measurements.
