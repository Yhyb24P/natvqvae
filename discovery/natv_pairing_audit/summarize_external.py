#!/usr/bin/env python
"""Quantify external discovery against the frozen NatV family population.

Reads only completed audit/discovery tables. Writes an overlap summary and a
focused list of sequence candidates absent from the frozen population. This
does not modify the formal population, splits or contracts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
EXT = OUT / "external"
FAMILY_ROLE = "family_population_v2"


def main() -> None:
    family = pd.read_csv(OUT / "sequence_registry.csv")
    pop = family[family.record_type == FAMILY_ROLE].copy()
    pop_hashes = set(pop.sequence_sha256)
    pop_unique = pop.drop_duplicates("sequence_sha256")

    rcsb = pd.read_csv(EXT / "incremental_structures.csv")
    dual = pd.read_csv(EXT / "uniprot_dual_pfam_all.csv")
    cand = pd.read_csv(EXT / "uniprot_candidates.csv")
    ipr = pd.read_csv(EXT / "uniprot_ipr041750_all.csv")
    af = pd.read_csv(EXT / "alphafold_metadata.csv")

    cand["in_family_by_sequence"] = cand.sequence_sha256.isin(pop_hashes)
    new = cand[~cand.in_family_by_sequence].copy()
    roles = pop_unique.set_index("sequence_sha256")["training_role_v2"]
    overlap_roles = (
        cand.loc[cand.in_family_by_sequence, "sequence_sha256"]
        .map(roles)
        .value_counts()
        .to_dict()
    )

    af_queried = set(af.accession)
    new = new.merge(
        af.drop_duplicates("accession")[
            ["accession", "af_model_entity_id", "af_latest_version"]
        ],
        on="accession",
        how="left",
    )
    new["af_metadata_queried"] = new.accession.isin(af_queried)
    new["has_af_model"] = new.af_model_entity_id.notna()
    columns = [
        "accession",
        "entry_id",
        "organism",
        "length",
        "hmm_nmnat_cov",
        "hmm_nudix_cov",
        "sequence_sha256",
        "af_metadata_queried",
        "has_af_model",
        "af_model_entity_id",
        "af_latest_version",
    ]
    new[columns].sort_values("accession").to_csv(
        EXT / "incremental_uniprot_candidates.csv", index=False
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "family_population": {
            "records": int(len(pop)),
            "unique_sequences": int(pop.sequence_sha256.nunique()),
        },
        "rcsb_sequence_search": {
            "union_polymer_entities": int(len(rcsb)),
            "in_frozen_corpus": int(rcsb.in_frozen_corpus.sum()),
            "incremental_vs_corpus": int(rcsb.incremental_vs_corpus.sum()),
            "qualifying_full_length_dual": int(
                rcsb.qualifying_full_length_dual.sum()
            ),
            "qualifying_entities": sorted(
                rcsb.loc[rcsb.qualifying_full_length_dual, "entity_id"].tolist()
            ),
            "incremental_qualifying": sorted(
                rcsb.loc[
                    rcsb.qualifying_full_length_dual
                    & rcsb.incremental_vs_corpus,
                    "entity_id",
                ].tolist()
            ),
        },
        "uniprot_dual_pfam": {
            "records": int(len(dual)),
            "reviewed": int(dual.reviewed.sum()),
            "qualifying_full_length_dual": int(
                dual.qualifying_full_length_dual.sum()
            ),
            "in_family_by_sequence": int(dual.sequence_sha256.isin(pop_hashes).sum()),
            "not_in_family_by_sequence": int(
                (~dual.sequence_sha256.isin(pop_hashes)).sum()
            ),
            "with_pdb": int(dual.has_pdb.sum()),
            "with_pdb_not_in_corpus": int(dual.pdb_not_in_corpus.sum()),
        },
        "uniprot_candidates": {
            "records": int(len(cand)),
            "in_family_by_sequence": int(cand.in_family_by_sequence.sum()),
            "new_by_sequence": int(len(new)),
            "new_phage_host": int(
                new.organism.str.contains("phage", case=False, na=False).sum()
            ),
            "new_af_metadata_queried": int(new.af_metadata_queried.sum()),
            "new_with_af_model": int(new.has_af_model.sum()),
            "new_af_no_model_404": int(
                (new.af_metadata_queried & ~new.has_af_model).sum()
            ),
            "new_af_not_queried": int((~new.af_metadata_queried).sum()),
            "overlap_roles": {k: int(v) for k, v in overlap_roles.items()},
        },
        "uniprot_ipr041750": {
            "records": int(len(ipr)),
            "qualifying_full_length_dual": int(
                ipr.qualifying_full_length_dual.sum()
            ),
            "in_family_by_sequence": int(ipr.sequence_sha256.isin(pop_hashes).sum()),
            "with_pdb": int(ipr.has_pdb.sum()),
        },
        "alphafold": {
            "metadata_records": int(len(af)),
            "accessions": int(af.accession.nunique()),
            "with_model": int(af.af_model_entity_id.notna().sum()),
            "without_model_404": int(af.af_model_entity_id.isna().sum()),
        },
    }
    (EXT / "discovery_overlap.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
