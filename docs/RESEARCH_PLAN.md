# Current NatV research plan

**Research object.** KVP40 NatV, an NMNAT–Nudix double-domain enzyme, is the starting scaffold. The application objective is **greater NAD⁺ net accumulation** from NMN and ATP in a specified purified-enzyme system; decreased or absent Nudix NAD⁺ hydrolysis is acceptable. The enzyme must retain NMNAT catalytic competence. This objective supersedes earlier formulations that required preservation of natural dual catalytic function.

**Evidence boundary.** There is no sufficient NatV variant activity dataset to learn or verify `kcat`, `Km` or net production from sequence. Computation generates *rankable hypotheses* for later experiments. Performance labels must never be invented from pLDDT, ESM CE, docking or geometry. Net production reflects formation, hydrolysis, reverse reactions and assay conditions; improvement must be verified biochemically.

## The research, not a sequence of experiment numbers

**Can we learn a useful NatV generative distribution?** Curate target-family sequence identity, ordered domains, homolog redundancy, alignments and held-out populations. Review whether the currently available 1,644 training records/47 redundancy components adequately represent family variation; 2,430 entries in the legacy population file are a discovery population, **not an automatically admissible training set**. External NAD-pathway proteins may help generic pretraining or provide functional comparison, never silently relabel them NatV. Do not treat 47 components as independent evolutionary lineages.

Study QVAE and no-latent SE under matched source data, sampling effort and evaluation conditions. Distinguish **WT-relative masked editing** (what the old H2 run actually evaluated) from **prior-driven NatV family generation** (not yet shown by the repository). Before committing new fit budgets, verify that the proposed architecture, decoder and training objective can support the chosen generation task. If masked-recovery is the only implemented sampler, report its exact initial sequence and masking policy, or implement a genuinely supported family-sampling procedure rather than rename it de novo. No latent-dimension interpretability study is required to claim practical utility; ensure the prior is used and candidate benefits survive a fair non-latent comparison.

**Can we propose biochemically distinct designs?** Generate and retain three *design hypotheses*: A seeks stronger NMNAT synthesis while preserving essential catalytic organization; B seeks less Nudix-mediated NAD⁺ hydrolysis without destroying a productive NMNAT scaffold; C jointly explores both. These are analytic categories that can share one generator. For WT-directed substitutions the current 341-residue partition in `contracts/regions_v1.json` provides an initial controlled comparison; it is **not** an evolutionary fact that every useful design must be substitution-only or avoid linker residues. A candidate editing a mapped catalytic residue is a mechanistic hypothesis, not automatically an effective intervention.

**Which candidates deserve further attention?** Use an interpretable evaluation sequence:

- **Sequence evidence:** canonical residues, construct identity, ancestry/novelty, candidate duplicates, edit burden, double-domain annotations, comparison with natural family variation. Conservative-family scores should not automatically reject purposeful Nudix attenuation.
- **Candidate-specific structures:** local full-length prediction with explicit MSA/template provenance; assess each domain, their relative arrangement and local disruption. WT confidence-based cutoffs cannot be transferred uncritically to altered sequences or across predictors. Record UNKNOWN when not run.
- **Catalytic-site plausibility:** sequence/structural correspondence to experimental NMNAT/Nudix homolog structures; NMN, ATP and Mg²⁺ environments where a chemically consistent pose is available; perturbation of Nudix NAD⁺ hydrolysis environment. Simple docking energies or loss of hydrogen bonds do not establish a change in reaction rate. Functional directions A/B/C remain hypotheses.
- **Panel selection:** balanced representation of A/B/C, structural diversity and plausible biochemical mechanisms, including appropriate WT/natural/simpler-generator comparisons. Preserve failed and unscored proposals in overall accounting. A model-derived total score must not be labelled predicted net synthesis.

Each formal generation/evaluation study freezes its available asset identities, sample counts, information budgets, uncertainty reporting and selection method **before seeing comparative results**. Structural prediction tools reusing the same family MSA or homolog template provide correlated evidence, not independent functional ground truth.

## Prior evidence that constrains decisions

- H1: evolution-informed input improved masked recovery in the exposed development comparison; additional graph structure on top of SE showed no decisive gain. SE is an appropriate current comparator, not a proven optimum for net NAD synthesis.
- H2: five-seed comparisons across no-latent, continuous and categorical architectures generated local KVP40 edits; the reported primary valid/novel/unique yield saturated at 100%. There was no defensible latent-generation advantage **under that task**; a genuinely different generative task remains untested.
- H3/G0: experimental pocket coordinates and local evaluation experience are valuable; the chemical adapter's generation benefit was not identified and the current implementation remains PARK. No inferred chemistry-specific gain or `kcat` labels are imported.
- G1-a: 576 zero-fit, 4/8-edit proposals distinguish site-selection policies only on partially self-referential scores; full-length structural evidence was not collected. It is optional baseline evidence, not the generative objective.
- G1-b: the apparent bimodal linker classes were a domain-annotation artifact; credible transferable short InDels remain unresolved. Do not activate a length-changing generator merely to increase model complexity. Its existing eight flagged definitions are pipeline-dependent screening outputs, not eight proven events.

Old results and code stay in the source repository with their existing frozen identities. No blanket copying, blind refitting or reopening of rejected G0 variants.

## Definition of a meaningful computational completion

A completed study provides: the actual generator and matched baseline; reproducible train/validation population and independent source identities; real candidate sequences with sampling receipts; candidate-specific structural and biochemical-context **computational** evidence; uncertainty and negative results; and a diverse, reproducible candidate panel. A good model score or a large FASTA alone does not meet this bar.

## Future biochemical validation (not currently authorised)

Define assay conditions, enzyme normalisation and time window; use direct NAD⁺ quantification and suitable controls. Separately measure (i) NMNAT formation initial rates and, where supported, two-substrate kinetic parameters; (ii) Nudix NAD⁺ hydrolysis; (iii) net NAD⁺ accumulation and ATP consumption. A structurally intact Nudix-catalytic-dead reference can help isolate hydrolysis contributions once such a reference is experimentally verified; it is **not** an existing positive control. Validate distinct designs and independent biological replicates before declaring a net-production improvement or feeding activities back into model training.
