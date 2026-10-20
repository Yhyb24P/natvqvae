#!/usr/bin/env python
"""Homology increment analysis for the 95 exact-novel UniProt NatV candidates.

The frozen family population is a 70/80 component graph: an edge requires
nident/alnlen >= 0.70 and both inclusive alignment spans >= 0.80 of their
sequence lengths (see nad_qvae vnext_family_split). This script asks whether
the 95 candidates add components under the same predicate, whether they
bridge existing components, and whether they preserve the curated 12 anchors.

Search is local MMseqs2; every candidate-family hit used for a decision is
re-aligned exactly with Biopython. No population, split, contract or model is
modified. Outputs are derived evidence, not activity measurements.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

OUT = Path(__file__).resolve().parent
EXT = OUT / "external"
HOM = EXT / "homology"
RAW = HOM / "raw"
HOM.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

MMSEQS = Path(sys.executable).with_name("mmseqs")
if not MMSEQS.exists():
    resolved = shutil.which("mmseqs")
    if resolved is None:
        raise SystemExit("mmseqs is required but was not found")
    MMSEQS = Path(resolved)

HIT_COLUMNS = [
    "query",
    "target",
    "pident",
    "alnlen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "qlen",
    "tlen",
    "evalue",
    "bits",
]
FAMILY_ROLE = "family_population_v2"
LOG: list[str] = []


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def run_mmseqs(query: Path, target: Path, output: Path, tag: str) -> None:
    command = [
        str(MMSEQS),
        "easy-search",
        str(query),
        str(target),
        str(output),
        str(RAW / f"tmp_{tag}"),
        "--format-output",
        "query,target,pident,alnlen,qstart,qend,tstart,tend,qlen,tlen,evalue,bits",
        "-s",
        "7.5",
        "--min-seq-id",
        "0.0",
        "-c",
        "0.0",
        "--cov-mode",
        "0",
        "-e",
        "10",
        "--max-seqs",
        "5000",
        "--threads",
        "8",
        "--mask",
        "0",
    ]
    t0 = time.time()
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    LOG.append(
        f"mmseqs {tag}: {time.time() - t0:.1f}s; "
        + " ".join(command[:6])
        + " [options recorded in script]"
    )


def read_m8(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=HIT_COLUMNS)
    hits = pd.read_csv(path, sep="\t", names=HIT_COLUMNS)
    hits["qcov"] = (hits["qend"] - hits["qstart"] + 1) / hits["qlen"]
    hits["tcov"] = (hits["tend"] - hits["tstart"] + 1) / hits["tlen"]
    return hits


def make_local_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.end_gap_score = 0
    return aligner


LOCAL = make_local_aligner()


def exact_local(query: str, target: str) -> dict:
    """Best local alignment with exact identity and inclusive spans."""
    alignment = LOCAL.align(query, target)[0]
    query_aligned, target_aligned = alignment[0], alignment[1]
    matches = sum(
        1
        for q, t in zip(query_aligned, target_aligned)
        if q == t and q != "-"
    )
    alnlen = len(query_aligned)
    qstart = int(alignment.aligned[0][0][0]) + 1
    qend = int(alignment.aligned[0][-1][1])
    tstart = int(alignment.aligned[1][0][0]) + 1
    tend = int(alignment.aligned[1][-1][1])
    return {
        "nident": matches,
        "alnlen": alnlen,
        "exact_pident": 100.0 * matches / alnlen,
        "exact_qstart": qstart,
        "exact_qend": qend,
        "exact_tstart": tstart,
        "exact_tend": tend,
        "exact_qcov": (qend - qstart + 1) / len(query),
        "exact_tcov": (tend - tstart + 1) / len(target),
    }


def map_ref_to_query(reference: str, query: str) -> dict[int, int]:
    """Map 1-based reference positions to 1-based query positions."""
    alignment = LOCAL.align(reference, query)[0]
    mapping: dict[int, int] = {}
    for (r0, r1), (q0, q1) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        for offset in range(int(r1 - r0)):
            mapping[int(r0) + offset + 1] = int(q0) + offset + 1
    return mapping


def union_find_components(
    candidate_ids: list[str],
    edges: list[tuple[str, str]],
    component_targets: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, set[str]]]:
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for accession in candidate_ids:
        find(("cand", accession))
    for accession, components in component_targets.items():
        for component in components:
            union(("cand", accession), ("comp", component))
    for left, right in edges:
        union(("cand", left), ("cand", right))

    roots: dict[tuple[str, str], list[str]] = {}
    for accession in candidate_ids:
        roots.setdefault(find(("cand", accession)), []).append(accession)

    assignments: dict[str, str] = {}
    root_components: dict[str, set[str]] = {}
    ordered = sorted(
        roots.items(),
        key=lambda item: (-len(item[1]), min(item[1])),
    )
    new_index = 0
    existing_index = 0
    for root, members in ordered:
        components = sorted(
            {
                component
                for accession in members
                for component in component_targets.get(accession, set())
            }
        )
        if components:
            existing_index += 1
            label = f"existing_component_cluster_{existing_index:02d}"
        else:
            new_index += 1
            label = f"discovery_component_{new_index:02d}"
        for accession in members:
            assignments[accession] = label
        root_components[label] = set(components)
    return assignments, root_components


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    started = time.time()

    registry = pd.read_csv(OUT / "sequence_registry.csv")
    population = registry[registry.record_type == FAMILY_ROLE].copy()
    candidates_all = pd.read_csv(EXT / "uniprot_candidates.csv")
    candidate_list = pd.read_csv(EXT / "incremental_uniprot_candidates.csv")
    candidates = (
        candidates_all[
            candidates_all["accession"].isin(candidate_list["accession"])
        ]
        .copy()
        .reset_index(drop=True)
    )
    if len(candidates) != 95:
        raise ValueError(f"expected 95 candidates, found {len(candidates)}")
    if candidates["sequence_sha256"].nunique() != len(candidates):
        raise ValueError("candidate sequences are not unique")
    for row in candidates.itertuples():
        if sha256_text(row.sequence) != row.sequence_sha256:
            raise ValueError(f"sequence hash mismatch for {row.accession}")
    if len(population) != 2430 or population["sequence_sha256"].nunique() != len(
        population
    ):
        raise ValueError("unexpected frozen family population shape")

    population_meta = population.set_index("sequence_sha256")
    population_sequence = dict(
        zip(population["sequence_sha256"], population["sequence"])
    )
    candidate_sequence = dict(zip(candidates["accession"], candidates["sequence"]))

    write_fasta(
        HOM / "candidates.fasta",
        [(r.accession, r.sequence) for r in candidates.itertuples()],
    )
    write_fasta(
        HOM / "family.fasta",
        [(r.sequence_sha256, r.sequence) for r in population.itertuples()],
    )

    version = subprocess.check_output(
        [str(MMSEQS), "version"], text=True
    ).strip()
    LOG.append(f"mmseqs version: {version}")
    run_mmseqs(
        HOM / "candidates.fasta",
        HOM / "family.fasta",
        RAW / "candidates_vs_family.m8",
        "candidates_vs_family",
    )
    run_mmseqs(
        HOM / "candidates.fasta",
        HOM / "candidates.fasta",
        RAW / "candidates_vs_candidates.m8",
        "candidates_vs_candidates",
    )

    family_hits = read_m8(RAW / "candidates_vs_family.m8")
    if family_hits.empty:
        raise ValueError("no candidate-family search hits")

    full = family_hits[
        (family_hits["qcov"] >= 0.8) & (family_hits["tcov"] >= 0.8)
    ].copy()
    to_realign = full[full["pident"] >= 45.0].copy()

    # Candidates without a >=45% full-length hit still need an exact nearest
    # neighbour. Re-align their ten best MMseqs hits regardless of coverage.
    covered = set(to_realign["query"])
    for accession in candidates["accession"]:
        if accession in covered:
            continue
        top = (
            family_hits[family_hits["query"] == accession]
            .sort_values("pident", ascending=False)
            .head(10)
        )
        to_realign = pd.concat([to_realign, top], ignore_index=True)
    to_realign = to_realign.drop_duplicates(["query", "target"])

    exact_records = []
    for row in to_realign.itertuples():
        result = exact_local(
            candidate_sequence[row.query], population_sequence[row.target]
        )
        exact_records.append(
            {
                "query": row.query,
                "target": row.target,
                "mmseqs_pident": row.pident,
                **result,
            }
        )
    exact_hits = pd.DataFrame(exact_records)
    exact_hits["qualifies_70_80"] = (
        (exact_hits["exact_pident"] >= 70.0)
        & (exact_hits["exact_qcov"] >= 0.8)
        & (exact_hits["exact_tcov"] >= 0.8)
    )
    exact_hits["target_component"] = exact_hits["target"].map(
        population_meta["component_v2"]
    )
    exact_hits["target_role"] = exact_hits["target"].map(
        population_meta["training_role_v2"]
    )
    exact_hits["target_split"] = exact_hits["target"].map(
        population_meta["split_v2"]
    )
    exact_hits = exact_hits.sort_values(
        ["query", "exact_pident"], ascending=[True, False]
    )
    exact_hits.to_csv(
        HOM / "candidate_family_hits_exact.csv.gz",
        index=False,
        compression="gzip",
    )

    # Per-candidate nearest-neighbour and component evidence.
    nearest_rows = []
    component_targets: dict[str, set[str]] = {}
    for accession in candidates["accession"]:
        hits = exact_hits[exact_hits["query"] == accession].copy()
        if hits.empty:
            raise ValueError(f"no exact hits for {accession}")
        full_hits = hits[(hits["exact_qcov"] >= 0.8) & (hits["exact_tcov"] >= 0.8)]
        best_full = (
            full_hits.sort_values("exact_pident", ascending=False).iloc[0]
            if not full_hits.empty
            else hits.sort_values("exact_pident", ascending=False).iloc[0]
        )
        qualifying = hits[hits["qualifies_70_80"]]
        best_train = hits[hits["target_role"] == "training"]
        best_train = (
            best_train.sort_values("exact_pident", ascending=False).iloc[0]
            if not best_train.empty
            else None
        )
        qualifying_components = {
            component
            for component in qualifying["target_component"]
            if isinstance(component, str) and component
        }
        component_targets[accession] = qualifying_components
        nearest_rows.append(
            {
                "accession": accession,
                "best_family_pident": round(best_full["exact_pident"], 3),
                "best_family_target": best_full["target"],
                "best_family_component": best_full["target_component"],
                "best_family_role": best_full["target_role"],
                "best_family_split": best_full["target_split"],
                "best_family_qcov": round(best_full["exact_qcov"], 4),
                "best_family_tcov": round(best_full["exact_tcov"], 4),
                "best_train_pident": (
                    round(best_train["exact_pident"], 3)
                    if best_train is not None
                    else np.nan
                ),
                "n_exact_hits_realigned": len(hits),
                "n_qualifying_70_80_targets": len(qualifying),
                "n_qualifying_components": len(qualifying_components),
            }
        )
    nearest = pd.DataFrame(nearest_rows)

    # Candidate-candidate 70/80 graph.
    candidate_hits = read_m8(RAW / "candidates_vs_candidates.m8")
    candidate_pairs = []
    for row in candidate_hits.itertuples():
        if row.query >= row.target:
            continue
        result = exact_local(
            candidate_sequence[row.query], candidate_sequence[row.target]
        )
        candidate_pairs.append(
            {
                "left": row.query,
                "right": row.target,
                "mmseqs_pident": row.pident,
                **result,
            }
        )
    candidate_edges = pd.DataFrame(candidate_pairs)
    candidate_edges["qualifies_70_80"] = (
        (candidate_edges["exact_pident"] >= 70.0)
        & (candidate_edges["exact_qcov"] >= 0.8)
        & (candidate_edges["exact_tcov"] >= 0.8)
    )
    candidate_edges.to_csv(HOM / "candidate_candidate_edges.csv", index=False)

    qualifying_edges = list(
        candidate_edges.loc[
            candidate_edges["qualifies_70_80"], ["left", "right"]
        ].itertuples(index=False, name=None)
    )
    assignments, root_components = union_find_components(
        list(candidates["accession"]), qualifying_edges, component_targets
    )

    # Existing component sizes and bridge accounting.
    component_sizes = population["component_v2"].value_counts().to_dict()
    root_rows = []
    for label, components in sorted(root_components.items()):
        members = sorted(
            accession
            for accession, assigned in assignments.items()
            if assigned == label
        )
        root_rows.append(
            {
                "cluster": label,
                "n_candidates": len(members),
                "existing_components": ";".join(sorted(components)),
                "n_existing_components": len(components),
                "existing_component_sizes": ";".join(
                    str(component_sizes.get(component, 0))
                    for component in sorted(components)
                ),
                "candidate_accessions": ";".join(members),
            }
        )
    cluster_table = pd.DataFrame(root_rows)
    cluster_table.to_csv(HOM / "component_expansion.csv", index=False)

    bridge_rows = cluster_table[cluster_table["n_existing_components"] > 1]
    new_clusters = cluster_table[
        cluster_table["n_existing_components"] == 0
    ]
    candidates_in_existing = int(
        cluster_table.loc[
            cluster_table["n_existing_components"] > 0, "n_candidates"
        ].sum()
    )
    candidates_in_new = int(new_clusters["n_candidates"].sum())
    merge_count = int(
        (cluster_table["n_existing_components"] - 1).clip(lower=0).sum()
    )
    components_before = int(population["component_v2"].nunique())
    components_after = components_before - merge_count + len(new_clusters)

    # Anchor conservation. Reference domains: WT CDD intervals from the local
    # audit; Q5NHR1 specific CDD intervals from anchors_cdd_domains.parquet
    # (cd02168 4-187, cd18873 203-341).
    query_native = registry[registry.record_type == "query_native"].iloc[0]
    template_native = registry[
        (registry.record_type == "template_native")
        & (registry["source_id"] == "Q5NHR1")
    ].iloc[0]
    wt_sequence = query_native["sequence"]
    q5_sequence = template_native["sequence"]
    anchor_table = pd.read_csv(OUT / "anchor_mapping_audit.csv")
    reference_domains = {
        "WT": {"NMNAT": (4, 181), "Nudix": (204, 338)},
        "Q5": {"NMNAT": (4, 187), "Nudix": (203, 341)},
    }
    reference_sequences = {"WT": wt_sequence, "Q5": q5_sequence}
    anchor_rows = []
    for row in candidates.itertuples():
        record = {
            "accession": row.accession,
            "length": len(row.sequence),
        }
        for reference_name in ("WT", "Q5"):
            residues = []
            position_labels = []
            for domain, domain_anchors in (
                ("NMNAT", anchor_table[anchor_table.domain == "NMNAT"]),
                ("Nudix", anchor_table[anchor_table.domain == "Nudix"]),
            ):
                start, end = reference_domains[reference_name][domain]
                reference_sequence = reference_sequences[reference_name][
                    start - 1 : end
                ]
                mapping = map_ref_to_query(reference_sequence, row.sequence)
                position_column = (
                    "map_wt_position"
                    if reference_name == "WT"
                    else "map_template_position"
                )
                for anchor in domain_anchors.itertuples():
                    position = int(getattr(anchor, position_column))
                    offset = position - start + 1
                    query_position = mapping.get(offset)
                    residues.append(
                        row.sequence[query_position - 1]
                        if query_position is not None
                        else "-"
                    )
                    position_labels.append(
                        f"{domain}:{position}:"
                        f"{anchor.map_wt_residue if reference_name == 'WT' else anchor.q5_uniprot_actual}"
                    )
            record[f"{reference_name}_anchor_residues"] = "".join(residues)
            record[f"{reference_name}_anchor_mapped"] = sum(
                residue != "-" for residue in residues
            )
        anchor_rows.append(record)
    anchor_df = pd.DataFrame(anchor_rows)

    expected = {
        "WT": [
            anchor.map_wt_residue[0]
            for anchor in anchor_table.itertuples()
        ],
        "Q5": [
            anchor.q5_uniprot_actual
            for anchor in anchor_table.itertuples()
        ],
    }
    for reference_name in ("WT", "Q5"):
        anchor_df[f"{reference_name}_anchor_matches"] = [
            sum(a == b for a, b in zip(residues, expected[reference_name]))
            for residues in anchor_df[f"{reference_name}_anchor_residues"]
        ]
        # Position 6 (WT 236 / Q5 238) is the only anchor that is not
        # identity-conserved between the two experimental references.
        anchor_df[f"{reference_name}_anchor_matches_excl_variable"] = [
            sum(
                a == b
                for index, (a, b) in enumerate(
                    zip(residues, expected[reference_name])
                )
                if index != 5
            )
            for residues in anchor_df[f"{reference_name}_anchor_residues"]
        ]

    # Per-anchor conservation across the 95 candidates.
    anchor_position_rows = []
    for index, anchor in enumerate(anchor_table.itertuples()):
        q5_residues = [
            value[index] for value in anchor_df["Q5_anchor_residues"]
        ]
        wt_residues = [
            value[index] for value in anchor_df["WT_anchor_residues"]
        ]
        q5_counts = Counter(q5_residues)
        wt_counts = Counter(wt_residues)
        anchor_position_rows.append(
            {
                "domain": anchor.domain,
                "wt_position": anchor.map_wt_position,
                "wt_residue": anchor.map_wt_residue[0],
                "q5_position": anchor.map_template_position,
                "q5_residue": anchor.q5_uniprot_actual,
                "q5_match": q5_counts.get(anchor.q5_uniprot_actual, 0),
                "q5_unmapped": q5_counts.get("-", 0),
                "q5_top_substitutions": ";".join(
                    f"{residue}:{count}"
                    for residue, count in q5_counts.most_common(5)
                ),
                "wt_match": wt_counts.get(anchor.map_wt_residue[0], 0),
                "wt_unmapped": wt_counts.get("-", 0),
                "wt_top_substitutions": ";".join(
                    f"{residue}:{count}"
                    for residue, count in wt_counts.most_common(5)
                ),
            }
        )
    anchor_positions = pd.DataFrame(anchor_position_rows)
    anchor_positions.to_csv(HOM / "anchor_conservation_by_position.csv", index=False)

    # Sanity check: each reference maps to itself with 12/12 anchors.
    self_checks = {}
    for reference_name in ("WT", "Q5"):
        residues = []
        for domain, domain_anchors in (
            ("NMNAT", anchor_table[anchor_table.domain == "NMNAT"]),
            ("Nudix", anchor_table[anchor_table.domain == "Nudix"]),
        ):
            start, end = reference_domains[reference_name][domain]
            reference_sequence = reference_sequences[reference_name][start - 1 : end]
            mapping = map_ref_to_query(reference_sequence, reference_sequence)
            position_column = (
                "map_wt_position"
                if reference_name == "WT"
                else "map_template_position"
            )
            for anchor in domain_anchors.itertuples():
                position = int(getattr(anchor, position_column))
                query_position = mapping.get(position - start + 1)
                residues.append(
                    reference_sequence[query_position - 1]
                    if query_position is not None
                    else "-"
                )
        self_checks[reference_name] = {
            "residues": "".join(residues),
            "matches": sum(
                a == b for a, b in zip(residues, expected[reference_name])
            ),
        }

    # Family internal nearest-neighbour baseline from the frozen exact edges.
    edge_path = (
        Path("/home/yhshy/git_files/nad_qvae/data/processed/releases/"
             "natv_vnext_family_v2/homology_edges_family_v2.parquet")
    )
    if edge_path.exists():
        family_edges = pd.read_parquet(edge_path)
        family_edges = family_edges[
            family_edges["query"] != family_edges["target"]
        ].copy()
        family_edges["pident"] = family_edges["nident"] / family_edges["alnlen"]
        family_nn = family_edges.groupby("query")["pident"].max()
        family_nn_stats = {
            "sequences_with_non_self_edge_ge70": int(family_nn.count()),
            "sequences_without_non_self_edge_ge70": int(
                len(population) - family_nn.count()
            ),
            "median": round(float(family_nn.median()), 4),
            "p05": round(float(family_nn.quantile(0.05)), 4),
            "p25": round(float(family_nn.quantile(0.25)), 4),
            "p75": round(float(family_nn.quantile(0.75)), 4),
            "p95": round(float(family_nn.quantile(0.95)), 4),
            "min": round(float(family_nn.min()), 4),
        }
    else:
        family_nn_stats = None

    # Main per-candidate table.
    metadata = candidates[
        [
            "accession",
            "entry_id",
            "organism",
            "length",
            "hmm_nmnat_cov",
            "hmm_nudix_cov",
            "sequence_sha256",
        ]
    ].merge(
        candidate_list[
            ["accession", "af_metadata_queried", "has_af_model"]
        ],
        on="accession",
        how="left",
    )
    metadata["phage_associated"] = metadata["organism"].str.contains(
        "phage", case=False, na=False
    )
    assessment = (
        metadata.merge(nearest, on="accession", how="left")
        .merge(anchor_df.drop(columns=["length"]), on="accession", how="left")
        .merge(
            pd.DataFrame(
                {
                    "accession": list(assignments),
                    "component_cluster": list(assignments.values()),
                }
            ),
            on="accession",
            how="left",
        )
    )
    assessment["component_status"] = np.where(
        assessment["component_cluster"].str.startswith("discovery_component_"),
        "new_component",
        "joins_existing_component",
    )
    assessment = assessment.merge(
        cluster_table[
            [
                "cluster",
                "existing_components",
                "n_existing_components",
                "n_candidates",
            ]
        ],
        left_on="component_cluster",
        right_on="cluster",
        how="left",
    ).drop(columns=["cluster"])
    assessment.to_csv(HOM / "candidate_homology_assessment.csv", index=False)

    # Summaries.
    best = assessment["best_family_pident"].astype(float)
    anchor_q5_excl = assessment["Q5_anchor_matches_excl_variable"].astype(int)
    summary = {
        "generated_at_utc": generated,
        "script_sha256": sha256_text(Path(__file__).read_text()),
        "mmseqs_version": version,
        "inputs": {
            "family_population": len(population),
            "family_unique_sequences": int(population["sequence_sha256"].nunique()),
            "family_components": components_before,
            "candidates": len(candidates),
            "candidate_unique_sequences": int(
                candidates["sequence_sha256"].nunique()
            ),
            "candidates_unreviewed": int(
                (candidates_all.set_index("accession")
                 .loc[candidates["accession"], "reviewed"] == False).sum()
            ),
            "candidates_with_pdb": int(
                candidates_all.set_index("accession")
                .loc[candidates["accession"], "has_pdb"].sum()
            ),
            "phage_associated": int(assessment["phage_associated"].sum()),
        },
        "candidate_nearest_neighbour": {
            "metric": "exact local nident/alnlen over full-length spans >=0.8",
            "median": round(float(best.median()), 3),
            "p05": round(float(best.quantile(0.05)), 3),
            "p25": round(float(best.quantile(0.25)), 3),
            "p75": round(float(best.quantile(0.75)), 3),
            "p95": round(float(best.quantile(0.95)), 3),
            "min": round(float(best.min()), 3),
            "max": round(float(best.max()), 3),
            "ge90": int((best >= 90).sum()),
            "ge70": int((best >= 70).sum()),
            "ge50": int((best >= 50).sum()),
            "ge30": int((best >= 30).sum()),
            "lt70": int((best < 70).sum()),
        },
        "family_internal_nearest_neighbour": family_nn_stats,
        "component_expansion": {
            "predicate": "exact nident/alnlen >=0.70; both spans/length >=0.80",
            "components_before": components_before,
            "candidate_clusters": int(len(cluster_table)),
            "clusters_joining_existing": int(
                (cluster_table["n_existing_components"] > 0).sum()
            ),
            "candidates_joining_existing": candidates_in_existing,
            "new_components_added": int(len(new_clusters)),
            "candidates_in_new_components": candidates_in_new,
            "existing_components_touched": int(
                population[
                    population["component_v2"].isin(
                        set().union(
                            *[
                                set(components)
                                for components in root_components.values()
                                if components
                            ]
                        )
                    )
                ]["component_v2"].nunique()
            ),
            "bridges": [
                {
                    "cluster": row.cluster,
                    "candidates": row.candidate_accessions,
                    "existing_components": row.existing_components,
                    "component_sizes": row.existing_component_sizes,
                }
                for row in bridge_rows.itertuples()
            ],
            "existing_component_merges": merge_count,
            "components_after": components_after,
            "net_component_change": components_after - components_before,
            "new_component_sizes": sorted(
                new_clusters["n_candidates"].astype(int).tolist(), reverse=True
            ),
        },
        "anchor_conservation": {
            "reference": "WT CDD intervals and Q5NHR1 specific CDD intervals",
            "nonvariable_anchor_count": 11,
            "candidates_with_11_of_11_nonvariable_anchors": int(
                (anchor_q5_excl == 11).sum()
            ),
            "candidates_with_10_of_11_nonvariable_anchors": int(
                (anchor_q5_excl == 10).sum()
            ),
            "candidates_with_9_of_11_nonvariable_anchors": int(
                (anchor_q5_excl == 9).sum()
            ),
            "candidates_with_lt9_of_11_nonvariable_anchors": int(
                (anchor_q5_excl <= 8).sum()
            ),
            "q5_anchor_unmapped_total": int(
                (12 - assessment["Q5_anchor_mapped"]).sum()
            ),
            "self_checks": self_checks,
            "position_table": anchor_positions.to_dict(orient="records"),
            "variable_position_note": (
                "Anchor 6 is WT H236 / Q5 E238. It is not identity-conserved "
                "between the experimental references; substitutions at this "
                "site are not treated as anchor losses."
            ),
        },
        "decision": {
            "frozen_dataset_changed": False,
            "population_added": 0,
            "data_expansion_status": "ended_for_current_objective",
            "reason": (
                "The 95 candidates do expand the 70/80 sequence-component "
                "graph, but they have no experimental structure, no measured "
                "activity, and no reviewed functional annotation. A sequence "
                "expansion would require re-clustering and re-splitting the "
                "frozen population, invalidating the current evaluation. "
                "Retain them as a discovery pool; do not add them to training."
            ),
        },
        "limitations": [
            "No experimental structure or measured activity for any of the 95.",
            "UniProt TrEMBL records only; no reviewed entry among the 95.",
            "Frozen population has no per-sequence taxonomy table in the release; taxonomic comparison is not possible from the pinned inputs.",
            "Anchors are alignment-inferred at ~35% global identity, near the twilight zone; domain-wise local mapping reduces but does not eliminate ambiguity.",
            "MMseqs2 search is sensitive but not exhaustive; low-identity hits below the reported nearest neighbour cannot be excluded.",
            "70/80 components are operational sequence clusters, not independently evolved families or functional classes.",
            "No AlphaFold models were fetched or evaluated in this analysis.",
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (HOM / "homology_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    (HOM / "run.log").write_text("\n".join(LOG) + "\n")

    print(json.dumps(summary["component_expansion"], indent=1))
    print(json.dumps(summary["candidate_nearest_neighbour"], indent=1))
    print(json.dumps(
        {
            key: summary["anchor_conservation"][key]
            for key in (
                "candidates_with_11_of_11_nonvariable_anchors",
                "candidates_with_10_of_11_nonvariable_anchors",
                "candidates_with_9_of_11_nonvariable_anchors",
                "candidates_with_lt9_of_11_nonvariable_anchors",
                "q5_anchor_unmapped_total",
                "self_checks",
            )
        },
        indent=1,
    ))


if __name__ == "__main__":
    main()
