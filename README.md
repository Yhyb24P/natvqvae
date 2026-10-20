# natvqvae

Engineering **KVP40 NatV** for higher **net NAD⁺ synthesis** with computational sequence design.

This repository is intentionally new and self-contained. It references the original
research repository **read-only** (raw sequences, family data, HMMs, experimental
structures) and does not import its package, contracts, or run directories.

## Objective (v1)

Starting from the 341-aa KVP40 NatV (bifunctional NMNAT + Nudix), design sequences
that increase net NAD⁺ accumulation in a defined **in vitro** reaction:

```
J(θ) = ( [NAD⁺]_θ(T) − [NAD⁺](0) ) / E_{θ,0}
```

- NMNAT: `NMN + ATP ⇌ NAD⁺ + PPi` — must remain catalytically competent.
- Nudix: `NAD⁺ + H₂O → NMN + AMP` — may be attenuated, including loss of hydrolysis.
- `J` is an **experimental** quantity. No model in this repository predicts `k_cat`.
- NMNAT `k_cat`, `K_m(NMN)`, `K_m(ATP)`, `k_cat/K_m`, Nudix hydrolysis rate and net
  accumulation are reported as **separate** endpoints. Net accumulation alone does
  not establish improved synthesis.

## Design classes

| Class | Intent | Allowed edits |
|---|---|---|
| A | improve synthesis | NMNAT domain (4–181); Nudix catalytic anchors untouched |
| B | reduce NAD⁺ hydrolysis | Nudix domain (204–338), catalytic center prioritized; NMNAT untouched |
| C | joint | A ∪ B under both regional constraints |

**Hard constraint:** the Nudix domain must remain present and folded. Domain
deletion or truncation designs are forbidden — the Nudix region also contributes to
oligomerization and interdomain organization.

## Non-goals

- Predicting `k_cat`, `k_cat/K_m`, or any catalytic rate.
- Converting docking scores, pLDDT, L_topo, or Vina outputs into catalytic efficiency.
- Modeling cellular metabolism. Scope is the purified-enzyme NMN + ATP system.
- Claiming activity for any computational candidate.

## Inherited rulings (recorded to avoid silent re-opening)

- The G0 chemical-conditioning plan stays **PARK**.
- G1-a is a site-selection baseline only; it does not define the generation mainline.
- G1-b is not re-activated by permission to lose Nudix hydrolysis.
- Historical OD570/λ570 wet-lab data are **not** accepted as functional evidence in
  this project.
- S2 / T135A stay quarantined.
- No run authorization is inherited. Each generation or folding run needs its own
  contract and output directory.

## Scientific honesty note on generation

The frozen NatV sequence models available to this project are **masked-recovery**
models: they score and recover residues under a mask, conditioned on a frozen PLM.
They are not free de novo generators. Any candidate produced here is therefore a
*recovery sample under a specified mask*, and must be described that way.

## Layout

```
contracts/    objective, region map, evidence ladder
inputs/       hash-verified symlinks to read-only source data (manifest.json)
assets/       frozen weights + provenance (empty until a generation arm is chosen)
src/natvqvae/ intake, regions, sequence, CLI, generator protocol
tests/        pytest
runs/         ignored; every run gets its own frozen-config directory
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
python inputs/link_inputs.py --check-only   # verify hashes, create nothing
python inputs/link_inputs.py                # create read-only symlinks
python -m natvqvae doctor
pytest -q
```

## Status

v0 skeleton only: no generators, no checkpoints, no folding, no candidates.
