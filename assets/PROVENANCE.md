# Model weights and provenance

**No checkpoint has been imported into `natvqvae` yet.** Model names below designate potential *source experiments*, not loaded, tested or authorised generative capabilities.

| Evidence source in legacy `nad_qvae` | What is known | Decision |
|---|---|---|
| H2 categorical/continuous/no-latent, `results/natv_h2_architecture_generation_v1_attempt1/PHASE_SUMMARY.md` | Five-seed fixed-4/8-site KVP40 recovery sampling; latent primary yield not superior under that task | Candidate implementations to inspect; no weights imported |
| H1/H2 SE/no-latent trained checkpoints | Reasonable evolutionary + pretrained PLM comparator for masked recovery | Potential baseline; verify exact checkpoint and effective split |
| Earlier `qvae_v9_seed2028` files | Historic naming; effective configuration/provenance may be incomplete | Do not silently adopt as current trained QVAE |
| H3/G0 chemical adapter | No qualified generation benefit established | PARK; no default import |

Before any weight is used, bind its **physical SHA256, architecture and exact source revision, effective training/config source, train/validation role files, tokenizer/alphabet, sampling mode and known limitations**. Do not infer de novo generation from the presence of a latent encoder.

The old repository remains read-only. Candidate generation/folding needs independent owner authorisation; a checkpoint's presence does not imply permission to run.
