#!/usr/bin/env python
"""Build the machine-readable structural interpretation basis for NatV.

This consolidates the audited domains, experimental templates, construct
offsets, ligand states and 12-anchor correspondences into one JSON object for
later candidate evaluation. It does not train, generate, fold, or modify the
formal population, splits, contracts or input manifest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
BASIS = OUT / "structural_interpretation_basis.json"

WT_ACCESSION = "AAQ64231.1"
Q5_ACCESSION = "Q5NHR1"
Q559_ACCESSION = "Q55928"

# WT CDD-derived intervals from the local audit; Q5NHR1 specific CDD hits
# cd02168 (4-187) and cd18873 (203-341) from anchors_cdd_domains.parquet.
DOMAINS = {
    "WT": {
        "NTERM": [1, 3],
        "NMNAT": [4, 181],
        "LINKER": [182, 203],
        "NUDIX": [204, 338],
        "CTERM": [339, 341],
    },
    "Q5NHR1": {
        "NMNAT": [4, 187],
        "NUDIX": [203, 341],
    },
}

TEMPLATE_USE = {
    "2QJO": {
        "uniprot": Q559_ACCESSION,
        "designated_use": "NMNAT domain, product-bound NAD proxy",
        "restriction": (
            "Use Q5NHR1-to-Q55928 cross-species alignment; same-numbered "
            "positions are invalid. No NMN+ATP substrate ternary geometry."
        ),
        "independent_sequence": True,
    },
    "2QJT": {
        "uniprot": Q5_ACCESSION,
        "designated_use": "Nudix domain, AMP + Mn product/metal proxy",
        "restriction": (
            "No ADPR substrate geometry; ligand proximity is not a "
            "hydrolysis-rate measurement."
        ),
        "independent_sequence": True,
    },
    "2R5W": {
        "uniprot": Q5_ACCESSION,
        "designated_use": "alternate experimental state of Q5NHR1",
        "restriction": (
            "Mg/Cl state of the same sequence as 2QJT; not an independent "
            "sequence or a second family member."
        ),
        "independent_sequence": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    registry = pd.read_csv(OUT / "sequence_registry.csv", low_memory=False)
    anchors = pd.read_csv(OUT / "anchor_mapping_audit.csv")
    constructs = pd.read_csv(OUT / "construct_observed_audit.csv")
    structures = pd.read_csv(OUT / "structure_registry.csv")

    wt = registry[registry.record_type == "query_native"].iloc[0]
    q5 = registry[
        (registry.record_type == "template_native")
        & (registry.uniprot_ids == Q5_ACCESSION)
    ].iloc[0]
    q559 = registry[
        (registry.record_type == "template_native")
        & (registry.uniprot_ids == Q559_ACCESSION)
    ].iloc[0]
    for reference, row, expected_length in (
        ("WT", wt, 341),
        ("Q5NHR1", q5, 347),
        (Q559_ACCESSION, q559, 339),
    ):
        if len(row.sequence) != expected_length:
            raise SystemExit(
                f"{reference} length {len(row.sequence)} != {expected_length}"
            )

    templates = []
    for pdb_id, use in TEMPLATE_USE.items():
        info = structures[structures.pdb_id == pdb_id].iloc[0]
        chain_rows = constructs[
            (constructs.template_pdb == pdb_id) & (constructs.scope == "chain")
        ]
        template_row = constructs[
            (constructs.template_pdb == pdb_id) & (constructs.scope == "template")
        ].iloc[0]
        tag_length = int(
            template_row.construct_length
            - (len(q559.sequence) if pdb_id == "2QJO" else len(q5.sequence))
        )
        templates.append(
            {
                "entity_id": f"{pdb_id}_1",
                "pdb_id": pdb_id,
                "uniprot": use["uniprot"],
                "method": info.method,
                "resolution_A": float(info.resolution_A),
                "construct_length": int(template_row.construct_length),
                "native_length": (
                    len(q559.sequence) if pdb_id == "2QJO" else len(q5.sequence)
                ),
                "n_terminal_tag_length": tag_length,
                "auth_to_entity_offset": tag_length,
                "observed_residues_total": int(template_row.observed_residues),
                "ligand_inventory": template_row.ligands,
                "chains": [
                    {
                        "chain": row.chain,
                        "observed_residues": int(row.observed_residues),
                        "first_label": int(row.first_label),
                        "last_label": int(row.last_label),
                        "missing_ranges": (
                            None if pd.isna(row.missing_ranges) else row.missing_ranges
                        ),
                        "auth_minus_label": int(row.auth_minus_label),
                        "ligands": row.ligands,
                    }
                    for row in chain_rows.itertuples()
                ],
                "designated_use": use["designated_use"],
                "restriction": use["restriction"],
                "independent_sequence": use["independent_sequence"],
            }
        )

    anchor_rows = []
    for _, row in anchors.iterrows():
        anchor_rows.append(
            {
                "domain": row["domain"],
                "wt_position": int(row["map_wt_position"]),
                "wt_residue": row["map_wt_residue"][0],
                "q5_position": int(row["map_template_position"]),
                "q5_residue": row["q5_uniprot_actual"],
                "q559_position_by_alignment": int(
                    row["wt_to_q559_aligned_pos"]
                ),
                "q559_residue_by_alignment": row["wt_to_q559_aligned_aa"][0],
                "q559_same_position_residue": row[
                    "q559_same_position_actual"
                ],
                "identity_conserved_across_templates": bool(
                    row["identity_conserved_actual"]
                ),
                "construct_positions": {
                    "2QJO": int(row["2QJO_construct_pos"]),
                    "2QJT": int(row["2QJT_construct_pos"]),
                    "2R5W": int(row["2R5W_construct_pos"]),
                },
                "observed": {
                    "2QJO": bool(row["2QJO_observed"]),
                    "2QJT": bool(row["2QJT_observed"]),
                    "2R5W": bool(row["2R5W_observed"]),
                },
                "designated_template": row["designated_template"],
                "min_ligand_distance_A": float(
                    row["designated_min_ligand_distance_A"]
                ),
                "mapping_status": row["mapping_status"],
                "ligand_proximity_status": row["ligand_proximity_status"],
            }
        )

    basis = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "formal_input_manifest": False,
        "intended_use": (
            "Structural interpretation and risk annotation of generated "
            "candidates. Not a training input, activity label or selector."
        ),
        "target": {
            "name": "NatV_KVP40",
            "source": "GenBank AY283928.2 / AAQ64231.1",
            "length": len(wt.sequence),
            "sequence_sha256": wt.sequence_sha256,
            "sequence": wt.sequence,
            "domains": DOMAINS["WT"],
            "domain_source": "CDD-derived WT-relative intervals",
        },
        "experimental_references": [
            {
                "uniprot": Q5_ACCESSION,
                "length": len(q5.sequence),
                "sequence_sha256": q5.sequence_sha256,
                "sequence": q5.sequence,
                "domains": DOMAINS["Q5NHR1"],
                "domain_source": (
                    "anchors_cdd_domains.parquet specific hits "
                    "cd02168 4-187 and cd18873 203-341"
                ),
                "template_entities": ["2QJT_1", "2R5W_1"],
            },
            {
                "uniprot": Q559_ACCESSION,
                "length": len(q559.sequence),
                "sequence_sha256": q559.sequence_sha256,
                "sequence": q559.sequence,
                "domains": None,
                "domain_source": (
                    "No independent domain interval in the pinned release; "
                    "map through Q5NHR1 by global alignment."
                ),
                "template_entities": ["2QJO_1"],
            },
        ],
        "templates": templates,
        "numbering": {
            "wt": "1-based inclusive, GenBank AAQ64231.1",
            "2QJO": (
                "entity/label = auth + 2; auth_minus_label = -2; "
                "construct = Q55928 native + 2"
            ),
            "2QJT": (
                "entity/label = auth + 5; auth_minus_label = -5; "
                "construct = Q5NHR1 native + 5"
            ),
            "2R5W": (
                "entity/label = auth + 5; auth_minus_label = -5; "
                "construct = Q5NHR1 native + 5"
            ),
            "warning": (
                "Do not equate same-numbered positions across KVP40, Q5NHR1 "
                "and Q55928. Use the recorded global alignments."
            ),
        },
        "mapping_routes": {
            "WT_to_Q5NHR1": (
                "Global pairwise alignment; domain-wise anchor mapping. "
                "Self-check 12/12."
            ),
            "WT_to_Q55928": (
                "Direct global pairwise alignment; self-check 12/12. "
                "Preferred route for the 2QJO column."
            ),
            "Q5NHR1_to_Q55928": (
                "Global pairwise alignment. Required for cross-species 2QJO "
                "template correspondence; 11/12 same-number positions would "
                "land on a different residue."
            ),
            "construct_to_native": "construct = UniProt native position + tag length",
            "native_to_observed": (
                "Check chain-specific missing ranges; no chain resolves every "
                "residue."
            ),
        },
        "anchor_table": anchor_rows,
        "anchor_summary": {
            "count": len(anchor_rows),
            "mapping_status": "PASS 12/12",
            "designated_ligand_proximity_status": "PASS 12/12",
            "variable_anchor": {
                "wt": "H236",
                "q5": "E238",
                "q559": "K236",
                "note": (
                    "Positional Nudix correspondence, not identity-conserved. "
                    "Substitutions at this site are not anchor losses."
                ),
            },
            "roles": {
                "NMNAT": (
                    "ATP-motif / catalysis-adjacent proxy from the 2QJO NAD "
                    "product state; not a substrate ternary complex"
                ),
                "NUDIX": (
                    "Nudix motif / metal-water network proxy from 2QJT "
                    "AMP + Mn; not ADPR substrate geometry"
                ),
            },
        },
        "ligand_states": {
            "2QJO": {
                "ligands": "APR, NAD, POP, SO4",
                "interpretation": "NMNAT product-bound NAD proxy",
                "nudix_anchor_distance_to_nmnat_site_A": "23-33",
            },
            "2QJT": {
                "ligands": "AMP, Mn",
                "interpretation": "Nudix product/metal proxy",
            },
            "2R5W": {
                "ligands": "Mg, Cl",
                "interpretation": (
                    "alternate metal state of the same Q5NHR1 sequence; "
                    "E238 is 4.655 A from the designated ligand"
                ),
            },
        },
        "evidence_semantics": {
            "allowed": [
                "Map candidate residues to template positions along the declared routes.",
                "Check whether candidate substitutions hit anchor positions.",
                "Annotate NMNAT catalytic-environment compatibility and Nudix risk as hypotheses.",
                "Report predicted structural confidence separately from experimental templates.",
            ],
            "not_allowed": [
                "Treat anchor preservation or ligand proximity as catalytic activity.",
                "Infer kcat, binding affinity or net NAD+ production from geometry.",
                "Treat 2QJT and 2R5W as independent sequences.",
                "Use a predicted structure as an experimental template.",
            ],
        },
        "limitations": [
            "No chain resolves all residues; use the recorded missing ranges.",
            "Full-length refers to the construct sequence, not complete observed coordinates.",
            "2QJO/2QJT are product/metal states; no NMN+ATP or ADPR ternary geometry.",
            "The 2QJO Nudix anchors are 23-33 A from the NMNAT-site NAD; 2QJT is the Nudix template.",
            "Anchor mapping across species at ~35% identity is near the twilight zone.",
            "The basis contains no catalytic measurement and no candidate sequence.",
        ],
        "sources": {
            "sequence_registry.csv": sha256_file(OUT / "sequence_registry.csv"),
            "anchor_mapping_audit.csv": sha256_file(OUT / "anchor_mapping_audit.csv"),
            "construct_observed_audit.csv": sha256_file(
                OUT / "construct_observed_audit.csv"
            ),
            "structure_registry.csv": sha256_file(OUT / "structure_registry.csv"),
        },
    }
    BASIS.write_text(json.dumps(basis, indent=1) + "\n")
    print(f"wrote {BASIS.name}: {len(anchor_rows)} anchors, {len(templates)} templates")


if __name__ == "__main__":
    main()
