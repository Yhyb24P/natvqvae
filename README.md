# NatV-QVAE | NAD⁺ net-production enzyme design

Research code for designing **KVP40 NatV-derived enzymes** with greater **net NAD⁺ production** in a specified purified-enzyme NMN + ATP reaction. Loss of Nudix-mediated NAD⁺ hydrolysis is acceptable **if NMNAT synthesis is retained**. This is a computational design project; **no variant has yet been shown experimentally to outperform WT**.

## Scientific objective

NMNAT: `NMN + ATP ⇌ NAD⁺ + PPi`. KVP40 Nudix can hydrolyse NAD⁺ to NMN + AMP. Increasing NMNAT catalysis and reducing NAD⁺ hydrolysis are **distinct causal routes** to greater net accumulation. The primary application endpoint is experimentally measured NAD⁺ accumulation under controlled conditions; NMNAT kinetics and Nudix hydrolysis must be measured separately to explain it. **No current model predicts catalytic efficiency or net yield.**

## Research route

1. **Data and biochemical context.** Reuse the curated NatV NMNAT–Nudix family and experimental homolog structures with traceable source, domain correspondence and train/validation boundaries. Generic NAD-related proteins are *not* automatically NatV family members.
2. **Generate.** Investigate QVAE alongside the existing no-latent SE baseline. Test real prior-driven generation and WT-relative editing as **different tasks**; do not mistake masked residue recovery for unconditional full-length synthesis. Keep training data, comparison budget and candidate provenance explicit.
3. **Assess and select.** Examine sequence novelty/diversity, candidate-specific full-length double-domain structures, catalytic-site plausibility and possible Nudix hydrolysis risk. Model-based readouts are evidence about compatibility, **not kcat labels**. Select a diverse, auditable panel rather than claim an in silico optimum.
4. **Future biochemical validation.** Compare WT and variants for NMNAT initial rates/kinetics, NAD⁺ hydrolysis and net NAD⁺ accumulation. No wet-lab execution is implied or authorised by this repository.

A/B/C describe **hypotheses**, not three required models: **A** modifies NMNAT to seek synthesis improvement; **B** modifies Nudix to seek reduced NAD⁺ hydrolysis; **C** combines both. Preserving fold/domain organisation matters; keeping natural Nudix hydrolytic function is *not* a requirement.

## What is actually available

The repository currently contains the WT identity, **provisional CDD-derived domain ranges**, a WT-relative substitution representation, a read-only source manifest and tests. **No trained model, inference adapter, evaluated candidate or experimentally verified gain has been imported or executed here.** The generator interface currently describes masked recovery only and must not be presented as a family prior sampler.

The older [nad_qvae](https://github.com/Yhyb24P/nad_qvae) remains the **immutable research evidence source**, not an import dependency. Existing research supports an SE comparator; prior H2 QVAE comparisons did *not* establish latent-variable benefit for their fixed-4/8-edit task. G0 chemical conditioning remains parked, G1-a is an optional site-selection comparator, and G1-b variable-length generation is deferred. These are bounded findings, not universal bans on QVAE, chemistry or InDels.

## Repository responsibilities

| Location | Purpose |
|---|---|
| `docs/RESEARCH_PLAN.md` | Current questions, research path and decision criteria |
| `docs/NATV_PAIRING_AUDIT.md` | Sequence–construct–coordinate–homology audit, experimental templates and 12-anchor mapping |
| `docs/DATA_PROVENANCE.md` | Population boundaries, source inheritance, evidence limitations |
| `contracts/` | Objective, **pilot-only** position rules, evaluation semantics |
| `inputs/manifest.json` | SHA-pinned **source files**; does not grant use of all files for training |
| `assets/PROVENANCE.md` | Checkpoint admission and identity; no models imported yet |
| `src/natvqvae/` | Only reusable capabilities exercised by a current research path |
| `tests/` | Tests of actual capabilities and scientific contracts |
| `runs/` | Ignored, per-run outputs with manifest and reproducible source identity |

`AGENTS.md` defines contributor boundaries; it is not another research plan. Results belong to individual run directories, not to a growing collection of "current status" documents. The current pairing audit remains in `discovery/natv_pairing_audit/`; `docs/NATV_PAIRING_AUDIT.md` is its reading entry point.

## Install and inspect

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m natvqvae doctor --root /path/to/nad_qvae
```

The last command checks actual local source files against `inputs/manifest.json` and fails if the source is absent/mismatched. `inputs/link_inputs.py --check-only --root /path/to/nad_qvae` verifies without creating symlinks. The source data are **not** shipped in this repository. No training or generation starts during installation or tests.

**Status:** research and repository foundation only. The next scientific deliverable is a reproducible QVAE-versus-baseline **generation-and-candidate-evaluation** study, not more skeleton code.
