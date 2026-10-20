# Frozen weight provenance

No weights are imported in v0. When a generation arm is chosen, every checkpoint must
be recorded here with:

- absolute source path and SHA256
- architecture class and its source file
- the config/training contract that produced it
- the data split it was trained on
- known limitations

## Admission rule

A checkpoint may be used only if all four identity fields above are present. A
checkpoint without a recoverable training contract is admissible **only** as a
comparator arm, and must be labelled as such in every output.

## Candidates on record (not yet imported)

| arm | source | status | note |
|---|---|---|---|
| SE-no-latent (`PLMPreservingRecoveryModel`) | old repo `experiments/run_reports/natv_formal_training_v1_attempt1/checkpoints` | strongest documentation; H1 confirmation ADOPT for S/E | masked-recovery model, not a free generator |
| QVAE (categorical latent) | old repo `data/foundry_outputs/checkpoints/qvae_v9_seed2028_ep*.pt` | provenance audit previously failed (missing effective config) | comparator only unless provenance is restored |
| SE+3Di / SE+Graph | old repo `experiments/run_reports/natv_multimodal_modality_comparison_v2_attempt2/checkpoints` | modality interaction unresolved | comparator |

Importing a weight does not authorize generation. Each run needs its own contract.
