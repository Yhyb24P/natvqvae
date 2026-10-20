# Data, model and evidence provenance

This file describes **what existing inputs mean**. It does not declare that the current machine has them, and it does not create a new data pipeline.

## Populations are different objects

The old project's fixed double-domain NatV training cohort contains **1,644 core-train sequences** in **47 redundancy components** (70% identity / 80% coverage clusters). These components mitigate sequence redundancy; they are not 47 independent ancestries or measured catalytic phenotypes. The family MSA used in some structural analyses contains **1,638** sequences due to additional correspondence/gap filters; absent six records must not be silently counted in the MSA.

`inputs/manifest.json` presently pins a **2,430-sequence discovery/population FASTA** from the old worktree. Role assignment across these records yields **1,644 core-train**, **227 validation** and **231 test** sequences; **249** records are quarantined for annotation conflict, **75** retained and **4** excluded. This file is *not* a pre-approved training partition, not automatically a clean NatV-functional cohort, and not interchangeable with the 1,644 train set. A future fit must bind the actual training-role table, validation/test role assignments, family qualification rules, and the hashes of derived examples/retrieval indices. Hashing an upstream FASTA alone is insufficient.

Other NAD-metabolism enzymes (notably NadE) are **not** automatically homologous full-length NatV training examples. Broad protein corpora may supply fixed pretrained weights or a deliberately documented transfer-learning experiment. Sequence selection, architecture membership and held-out homolog provenance must stay explicit.

## Structures and biochemical evidence

KVP40 NatV is the 341-aa reference with canonical sequence SHA256 in `contracts/regions.json`. CDD-derived windows (NMNAT 4–181, linker 182–203, Nudix 204–338) are annotation coordinates, **not measured domain cleavage boundaries or universal immutable mutation limits**. Named functional anchors are mapping/proxy hypotheses; verify chain/numbering with actual homolog structure correspondence before geometry claims.

The old research includes 2QJO/2QJT/2R5W homolog structures and chemically mapped pocket resources. Some assets may exist only on the original compute host. Homologous ligand structures do not supply a proven candidate-specific NMN+ATP reactive complex; ligand-binding proxy differences do not quantify catalysis. KVP40 and homolog enzyme papers support reaction identities but do not provide a high-quality sequence→mutant-activity training dataset.

A research result must identify **whether the source is a measured assay, a deposited experimental structure, a predicted structure, a family alignment or a generator score**. Never rewrite one source tier as another.

## Current contract baseline

`contracts/objective.yaml`, `contracts/evidence_ladder.yaml` and `contracts/regions.json` are the current contracts. The legacy `config/natv/target_contract.yaml` remains in the read-only source repository as historical construct metadata; it is not a pinned input here, and its `apparent_kcat_over_Km_NMN` decision endpoint does not define the current net-NAD-accumulation objective.

## Model inheritance

The older `nad_qvae` repository is an **external research archive**. No checkpoint has been imported here. A legacy checkpoint is not admitted merely because its filename contains QVAE; verify the architecture source, training contract, dataset split, checkpoint SHA, output alphabet and actual sampling mode.

Known source experiments, recorded for identification only: H2 categorical/continuous/no-latent (`experiments/run_reports/natv_h2_architecture_generation_v1_attempt1/PHASE_SUMMARY.md` in the source repository), H1/H2 SE/no-latent checkpoints, earlier `qvae_v9_seed2028` files, and the H3/G0 chemical adapter. Their scientific status is summarized in `docs/RESEARCH_PLAN.md`; none is a current dependency.

Retain a distinction among raw/full-length prior generation, WT-relative masked editing, and residue-restoration validation. Do not convert former claims into later ones. An existing ESM-2 model may contain pretraining overlap with NatV; "not found in audited data" is not proof of pretrained-model family ignorance.

## File/experiment boundaries

`NATV_SOURCE_ROOT` or the `--root` option chooses the read-only legacy source path. `inputs/link_inputs.py` checks hashes and links to local source files; **a symlink is not a filesystem read-only permission mechanism**. Never write through it; generated artifacts belong in this repository's experiment directories.

A run record binds the active Git revision, input and checkpoint hashes, training/retrieval population, generation mode, evaluation definitions, random seeds, attempted proposals and actual costs. Failures stay with the run. Large files and scratch outputs stay outside Git with hashes and an accessible storage location.

GitHub CI cannot authenticate the owner's local research dataset. Local `natvqvae doctor` is the required actual-data check before using linked inputs.
