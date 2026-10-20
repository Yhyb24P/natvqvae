# Data, model and evidence provenance

This file describes **what existing inputs mean**. It does not declare that the current machine has them, and it does not create a new data pipeline.

## Populations are different objects

The old project's frozen double-domain NatV training cohort contains **1,644 core-train sequences** in **47 redundancy components** (70% identity / 80% coverage clusters). These components mitigate sequence redundancy; they are not 47 independent ancestries or measured catalytic phenotypes. The family MSA used in some structural analyses contains **1,638** sequences due to additional correspondence/gap filters; absent six records must not be silently counted in the MSA.

`inputs/manifest.json` presently pins a **2,430-sequence discovery/population FASTA** from the old worktree. This file is *not* a pre-approved training partition, not automatically a clean NatV-functional cohort, and not interchangeable with the 1,644 train set. A future fit must bind the actual training-role table, validation/test role assignments, family qualification rules, and the hashes of derived examples/retrieval indices. Hashing an upstream FASTA alone is insufficient.

Other NAD-metabolism enzymes (notably NadE) are **not** automatically homologous full-length NatV training examples. Broad protein corpora may supply frozen pretrained weights or a deliberately documented transfer-learning experiment. Sequence selection, architecture membership and held-out homolog provenance must stay explicit.

## Structures and biochemical evidence

KVP40 NatV is the 341-aa reference with canonical sequence SHA256 in `contracts/regions_v1.json`. CDD-derived windows (NMNAT 4–181, linker 182–203, Nudix 204–338) are annotation coordinates, **not measured domain cleavage boundaries or universal immutable mutation limits**. Named functional anchors are mapping/proxy hypotheses; verify chain/numbering with actual homolog structure correspondence before geometry claims.

The old research includes 2QJO/2QJT/2R5W homolog structures and chemically mapped pocket resources. Some assets may exist only on the original compute host. Homologous ligand structures do not supply a proven candidate-specific NMN+ATP reactive complex; ligand-binding proxy differences do not quantify catalysis. KVP40 and homolog enzyme papers support reaction identities but do not provide a high-quality sequence→mutant-activity training dataset.

A research result must identify **whether the source is a measured assay, a deposited experimental structure, a predicted structure, a family alignment or a generator score**. Never rewrite one source tier as another.

## Model inheritance

The older `nad_qvae` repository is an **external research archive**. A legacy checkpoint is not admitted merely because its filename contains QVAE. Verify the architecture source, training contract, dataset split, checkpoint SHA, output alphabet and actual sampling mode. H2's 2026-10-10 categorical/continuous/no-latent study is stronger relevant evidence than unsupported early `qvae_v9` checkpoint names, but no old checkpoint has yet been imported here.

Retain a distinction among raw/full-length prior generation, WT-relative masked editing, and residue-restoration validation. Do not convert former claims into later ones. An existing ESM-2 model may contain pretraining overlap with NatV; "not found in audited data" is not proof of pretrained-model family ignorance.

## File/experiment boundaries

`NATV_SOURCE_ROOT` or the `--root` option chooses the read-only legacy source path. `inputs/link_inputs.py` checks hashes and links to local source files; **a symlink is not a filesystem read-only permission mechanism**. Neither user nor agent may write through it. Only this repository's run directory may receive generated artifacts.

A meaningful experiment receipt records: active Git source revision, exact input/checkpoint hashes, training/retrieval population, generation mode, assay or model evaluation definitions, random seeds, attempted and rejected proposals, actual fold/sampling costs, and versioned results. Preserve old run evidence without rewriting it; a corrected statistic is separately identified. Large files and scratch outputs stay outside Git with hashes and accessible storage location, rather than a pointer that silently loses the only copy.

GitHub CI uses self-contained contract tests; it cannot authenticate or verify the owner's local 700GB research dataset. Local `natvqvae doctor` is the required actual-data check before using linked inputs.
