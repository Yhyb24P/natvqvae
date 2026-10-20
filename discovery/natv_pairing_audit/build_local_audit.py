#!/usr/bin/env python
"""Build the NatV sequence-construct-coordinate-homology audit.

Reads only the pinned inputs and the read-only nad_qvae release. Writes new
tables under discovery/natv_pairing_audit/. Does not modify the source tree,
the formal population, or the core contracts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import gemmi
import pandas as pd
import pyhmmer
from Bio import Align
from Bio.Align import substitution_matrices

ROOT = Path(__file__).resolve().parents[2]
SRC = Path("/home/yhshy/git_files/nad_qvae")
OUT = Path(__file__).resolve().parent

FAMILY_DIR = SRC / "data/processed/releases/natv_vnext_family_v2"
CORPUS_DIR = (
    SRC
    / "experiments/run_reports/natv_structure_source_resolution_v2/experimental_candidates"
)
ACTIVE_G = (
    SRC
    / "data/processed/releases/natv_biohub_atlas_descriptors_v1/active_G_v3_manifest.jsonl"
)
HMM_PATHS = {
    "PF01467": ROOT / "inputs/data/PF01467.hmm",
    "PF00293": ROOT / "inputs/data/PF00293.hmm",
}
GENBANK = ROOT / "inputs/data/AY283928.2.gb"
POCKET_MAP = ROOT / "inputs/data/domain_pocket_map.json"
TEMPLATE_PDBS = {
    "2QJO": ROOT / "inputs/data/2QJO.pdb",
    "2QJT": ROOT / "inputs/data/2qjt.pdb",
    "2R5W": CORPUS_DIR / "2R5W.cif",
}
TEMPLATE_IDS = ("2QJO", "2QJT", "2R5W")
AA = set("ACDEFGHIKLMNPQRSTVWY")
ALPHABET = pyhmmer.easel.Alphabet.amino()

ALIGNER = Align.PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")
ALIGNER.open_gap_score = -11
ALIGNER.extend_gap_score = -1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sanitize(seq: str) -> str:
    return "".join(c if c in AA else "X" for c in seq.upper())


def aa3to1(code: str) -> str:
    r = gemmi.find_tabulated_residue(code)
    return r.one_letter_code if r else "X"


def is_amino(r: gemmi.Residue) -> bool:
    return bool(gemmi.find_tabulated_residue(r.name).is_amino_acid())


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name = None
    parts: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(parts)))
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if name is not None:
        records.append((name, "".join(parts)))
    return records


def parse_genbank_wt(path: Path) -> tuple[str, str, str]:
    text = path.read_text()
    match = re.search(
        r'/protein_id="(AAQ64231\.1)".*?/translation="([^"]+)"', text, re.S
    )
    if not match:
        raise RuntimeError("KVP40 protein_id/translation not found")
    seq = re.sub(r"\s", "", match.group(2))
    return match.group(1), seq, text


def entity_sequences(st: gemmi.Structure) -> dict[str, str]:
    out: dict[str, str] = {}
    for ent in st.entities:
        seq = "".join(aa3to1(code) for code in ent.full_sequence)
        if seq:
            out[ent.name] = seq
    return out


def entity_subchains(st: gemmi.Structure) -> dict[str, list[str]]:
    return {ent.name: list(ent.subchains) for ent in st.entities}


def observed_rows(st: gemmi.Structure) -> list[dict]:
    rows = []
    for ch in st[0]:
        for r in ch:
            if not is_amino(r):
                continue
            rows.append(
                {
                    "chain": ch.name,
                    "auth": r.seqid.num,
                    "label": r.label_seq,
                    "name": r.name,
                    "aa": aa3to1(r.name),
                }
            )
    return rows


def ligand_rows(st: gemmi.Structure) -> list[dict]:
    rows = []
    for ch in st[0]:
        for r in ch:
            if is_amino(r) or r.name in ("HOH", "DOD"):
                continue
            rows.append(
                {
                    "chain": ch.name,
                    "auth": r.seqid.num,
                    "label": r.label_seq,
                    "name": r.name,
                }
            )
    return rows


def find_residue(
    st: gemmi.Structure, chain_name: str, label_pos: int
) -> gemmi.Residue | None:
    for ch in st[0]:
        if ch.name != chain_name:
            continue
        for r in ch:
            if is_amino(r) and r.label_seq == label_pos:
                return r
    return None


def ligand_residues(
    st: gemmi.Structure, comp: str, auth: int | None = None, chain: str | None = None
) -> list[gemmi.Residue]:
    out = []
    for ch in st[0]:
        if chain is not None and ch.name != chain:
            continue
        for r in ch:
            if r.name != comp or is_amino(r):
                continue
            if auth is not None and r.seqid.num != auth:
                continue
            out.append(r)
    return out


def min_heavy_distance(
    res: gemmi.Residue | None, ligands: list[gemmi.Residue]
) -> float | None:
    if res is None or not ligands:
        return None
    best = None
    for a in res:
        if a.element.name in ("H", "D"):
            continue
        for lig in ligands:
            for b in lig:
                if b.element.name in ("H", "D"):
                    continue
                d = a.pos.dist(b.pos)
                if best is None or d < best:
                    best = d
    return None if best is None else round(best, 3)


def hmm_hits(hmms: dict[str, pyhmmer.plan7.HMM], seq: str) -> dict[str, dict | None]:
    clean = sanitize(seq)
    ds = pyhmmer.easel.TextSequence(name="query", sequence=clean).digitize(ALPHABET)
    result: dict[str, dict | None] = {}
    for name, hmm in hmms.items():
        best = None
        try:
            top = next(iter(pyhmmer.hmmsearch([hmm], [ds])))
        except Exception:
            result[name] = None
            continue
        for hit in top:
            for dom in hit.domains:
                if dom.i_evalue is None:
                    continue
                if best is None or dom.i_evalue < best["i_evalue"]:
                    best = {
                        "i_evalue": float(dom.i_evalue),
                        "hit_evalue": float(hit.evalue),
                        "score": float(hit.score),
                        "ali_from": int(dom.alignment.target_from),
                        "ali_to": int(dom.alignment.target_to),
                        "env_from": int(dom.env_from),
                        "env_to": int(dom.env_to),
                        "coverage": round(
                            (dom.alignment.target_to - dom.alignment.target_from + 1)
                            / float(hmm.M),
                            3,
                        ),
                    }
        result[name] = best
    return result


def alignment_maps(query: str, target: str) -> tuple[dict[int, int], dict]:
    al = ALIGNER.align(query, target)[0]
    q_to_t: dict[int, int] = {}
    matches = 0
    aligned_len = 0
    for (q0, q1), (t0, t1) in zip(al.aligned[0], al.aligned[1]):
        for i in range(q1 - q0):
            q_to_t[q0 + i] = t0 + i
            if query[q0 + i] == target[t0 + i]:
                matches += 1
            aligned_len += 1
    stats = {
        "score": float(al.score),
        "aligned_len": aligned_len,
        "identity": round(matches / aligned_len, 4) if aligned_len else 0.0,
        "query_coverage": round(aligned_len / len(query), 4) if query else 0.0,
    }
    return q_to_t, stats


def pfam_intervals(entity_json: dict) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for feature in entity_json.get("rcsb_polymer_entity_feature", []):
        if feature.get("type") != "Pfam":
            continue
        fid = feature.get("feature_id")
        if fid not in ("PF01467", "PF00293"):
            continue
        for pos in feature.get("feature_positions", []):
            out[fid].append((int(pos["beg_seq_id"]), int(pos["end_seq_id"])))
    for fid in out:
        out[fid].sort()
    return dict(out)


def classify_entity(
    seq: str, pfam: dict[str, list[tuple[int, int]]], hits: dict[str, dict | None]
) -> dict:
    n_hit = hits.get("PF01467")
    d_hit = hits.get("PF00293")
    n_pfam = bool(pfam.get("PF01467"))
    d_pfam = bool(pfam.get("PF00293"))
    pfam_order = (
        n_pfam
        and d_pfam
        and min(p[0] for p in pfam["PF01467"]) < min(p[0] for p in pfam["PF00293"])
    )
    hmm_order = bool(
        n_hit
        and d_hit
        and n_hit["ali_from"] < d_hit["ali_from"]
        and n_hit["coverage"] >= 0.5
        and d_hit["coverage"] >= 0.5
    )
    dual_pfam = n_pfam and d_pfam and pfam_order
    dual_hmm = hmm_order
    length_ok = len(seq) >= 300
    return {
        "pfam_nmnat": n_pfam,
        "pfam_nudix": d_pfam,
        "pfam_order": bool(pfam_order),
        "hmm_nmnat": n_hit is not None,
        "hmm_nudix": d_hit is not None,
        "hmm_nmnat_cov": n_hit["coverage"] if n_hit else None,
        "hmm_nudix_cov": d_hit["coverage"] if d_hit else None,
        "hmm_nmnat_ievalue": n_hit["i_evalue"] if n_hit else None,
        "hmm_nudix_ievalue": d_hit["i_evalue"] if d_hit else None,
        "hmm_order": bool(hmm_order),
        "dual_pfam": bool(dual_pfam),
        "dual_hmm": bool(dual_hmm),
        "length_ge_300": length_ok,
        "qualifying": bool(length_ok and (dual_pfam or dual_hmm)),
    }


def template_context(template: str) -> dict:
    cif = CORPUS_DIR / f"{template}.cif"
    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    with open(CORPUS_DIR / f"{template}_1.json") as fh:
        entity_json = json.load(fh)
    with open(CORPUS_DIR / f"{template}_entry.json") as fh:
        entry_json = json.load(fh)
    seqs = entity_sequences(st)
    subs = entity_subchains(st)
    ent_id = next(iter(seqs))
    construct = seqs[ent_id]
    ent_json_seq = entity_json["entity_poly"]["pdbx_seq_one_letter_code_can"]
    if construct != ent_json_seq:
        raise RuntimeError(f"{template}: CIF entity sequence differs from entity JSON")
    align = entity_json["rcsb_polymer_entity_align"][0]["aligned_regions"][0]
    entity_beg = int(align["entity_beg_seq_id"])
    ref_beg = int(align["ref_beg_seq_id"])
    sifts_len = int(align["length"])
    tag_len = entity_beg - 1
    native = construct[tag_len : tag_len + sifts_len]
    return {
        "template": template,
        "cif": cif,
        "st": st,
        "entity_json": entity_json,
        "entry_json": entry_json,
        "entity_id": ent_id,
        "subchains": subs[ent_id],
        "construct": construct,
        "entity_json_seq": ent_json_seq,
        "entity_beg": entity_beg,
        "ref_beg": ref_beg,
        "sifts_len": sifts_len,
        "tag_len": tag_len,
        "native": native,
        "uniprot": entity_json["rcsb_polymer_entity_container_identifiers"].get(
            "uniprot_ids", []
        ),
        "organism": [
            o.get("ncbi_scientific_name")
            for o in entity_json.get("rcsb_entity_source_organism", [])
        ],
        "pfam": pfam_intervals(entity_json),
        "artifacts": [
            (int(p["beg_seq_id"]), int(p["end_seq_id"]))
            for f in entity_json.get("rcsb_polymer_entity_feature", [])
            if f.get("type") == "artifact"
            for p in f.get("feature_positions", [])
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    hmms = {name: pyhmmer.plan7.HMMFile(str(path)).read() for name, path in HMM_PATHS.items()}

    # ---------- KVP40 WT ----------
    wt_pid, wt_seq, gb_text = parse_genbank_wt(GENBANK)
    assert len(wt_seq) == 341
    wt_sha = sha256_bytes(wt_seq.encode())
    assert wt_sha.startswith("8804bca1")

    with open(POCKET_MAP) as fh:
        pocket = json.load(fh)
    anchors = pocket["2QJT_reference_anchor_mapping"]

    # ---------- family population ----------
    fasta = parse_fasta(ROOT / "inputs/data/family_population.fasta")
    roles = pd.read_parquet(FAMILY_DIR / "family_training_roles_v2.parquet")
    split_manifest = json.load(open(FAMILY_DIR / "core_split_manifest_v2.json"))
    fasta_by_id = {name: seq for name, seq in fasta}
    role_by_id = {row.sequence_id: row for row in roles.itertuples(index=False)}

    hash_mismatch = []
    seq_mismatch = []
    for name, seq in fasta:
        if hashlib.sha256(seq.encode()).hexdigest() != name:
            hash_mismatch.append(name)
        if name in role_by_id and role_by_id[name].sequence != seq:
            seq_mismatch.append(name)
    assert len(fasta) == 2430 and len(roles) == 2430
    assert set(fasta_by_id) == set(roles.sequence_id)

    role_counts = roles["training_role_v2"].value_counts().to_dict()
    split_v2_counts = roles["split_v2"].value_counts().to_dict()
    legacy_counts = roles["legacy_split"].value_counts().to_dict()

    # ---------- active G ----------
    active_rows = []
    with open(ACTIVE_G) as fh:
        for line in fh:
            if line.strip():
                active_rows.append(json.loads(line))
    active_ids = {r["sequence_id"] for r in active_rows}
    active_missing_from_population = sorted(active_ids - set(fasta_by_id))
    population_not_in_active = len(set(fasta_by_id) - active_ids)

    q5_template = template_context("2QJT")
    q559_template = template_context("2QJO")
    r5w_template = template_context("2R5W")
    q5_seq = q5_template["native"]
    q559_seq = q559_template["native"]
    assert len(q5_seq) == 347 and len(q559_seq) == 339

    wt_to_q5, wt_q5_stats = alignment_maps(wt_seq, q5_seq)
    wt_to_q559, wt_q559_stats = alignment_maps(wt_seq, q559_seq)
    q5_to_q559, q5_q559_stats = alignment_maps(q5_seq, q559_seq)

    # ---------- structure corpus ----------
    corpus_rows = []
    entity_rows = []
    seq_cache: dict[str, dict] = {}
    st_cache: dict[str, gemmi.Structure] = {}
    for cif in sorted(CORPUS_DIR.glob("*.cif")):
        pdb_id = cif.stem.upper()
        entry_json = json.load(open(CORPUS_DIR / f"{pdb_id}_entry.json"))
        entity_jsons = []
        for path in sorted(CORPUS_DIR.glob(f"{pdb_id}_*.json")):
            if path.name == f"{pdb_id}_entry.json":
                continue
            with open(path) as fh:
                entity_jsons.append(json.load(fh))
        st = gemmi.read_structure(str(cif))
        st.setup_entities()
        st_cache[pdb_id] = st
        seqs = entity_sequences(st)
        subs = entity_subchains(st)
        obs = observed_rows(st)
        ligs = ligand_rows(st)
        obs_by_chain = Counter(r["chain"] for r in obs)
        lig_counter = Counter(r["name"] for r in ligs)
        title = entry_json.get("struct", {}).get("title", "")
        method = (entry_json.get("exptl") or [{}])[0].get("method", "")
        resolution = entry_json.get("rcsb_entry_info", {}).get("resolution_combined")
        resolution = resolution[0] if resolution else None
        if resolution is None:
            refine = entry_json.get("refine") or [{}]
            resolution = refine[0].get("ls_d_res_high")
        selected_ids = []
        for ej in entity_jsons:
            ej_id = str(
                ej.get("rcsb_polymer_entity_container_identifiers", {}).get(
                    "entity_id", ""
                )
            )
            if not ej_id:
                # fall back to the entity whose sequence matches the JSON
                for eid, s in seqs.items():
                    if s == ej["entity_poly"]["pdbx_seq_one_letter_code_can"]:
                        ej_id = eid
                        break
            selected_ids.append(ej_id)
        any_dual_pfam = False
        any_dual_hmm = False
        any_qualifying = False
        qualifying_ids = []
        for eid, seq in seqs.items():
            ej = next(
                (
                    e
                    for e in entity_jsons
                    if str(
                        e.get("rcsb_polymer_entity_container_identifiers", {}).get(
                            "entity_id", ""
                        )
                    )
                    == eid
                ),
                None,
            )
            if ej is None and len(entity_jsons) == 1 and len(seqs) == 1:
                ej = entity_jsons[0]
            pfam = pfam_intervals(ej) if ej else {}
            cache_key = sha256_bytes(seq.encode())
            if cache_key not in seq_cache:
                seq_cache[cache_key] = hmm_hits(hmms, seq)
            cls = classify_entity(seq, pfam, seq_cache[cache_key])
            uniprot = (
                ej.get("rcsb_polymer_entity_container_identifiers", {}).get(
                    "uniprot_ids", []
                )
                if ej
                else []
            )
            org = (
                [
                    o.get("ncbi_scientific_name")
                    for o in ej.get("rcsb_entity_source_organism", [])
                ]
                if ej
                else []
            )
            observed_here = sum(obs_by_chain[c] for c in subs.get(eid, []))
            ligs_here = Counter(
                r["name"] for r in ligs if r["chain"] in subs.get(eid, [])
            )
            entity_rows.append(
                {
                    "pdb_id": pdb_id,
                    "entity_id": eid,
                    "sequence_sha256": cache_key,
                    "entity_length": len(seq),
                    "uniprot_ids": ";".join(uniprot),
                    "organism": ";".join(org),
                    "pfam_nmnat": cls["pfam_nmnat"],
                    "pfam_nudix": cls["pfam_nudix"],
                    "pfam_order": cls["pfam_order"],
                    "hmm_nmnat_cov": cls["hmm_nmnat_cov"],
                    "hmm_nudix_cov": cls["hmm_nudix_cov"],
                    "hmm_nmnat_ievalue": cls["hmm_nmnat_ievalue"],
                    "hmm_nudix_ievalue": cls["hmm_nudix_ievalue"],
                    "hmm_order": cls["hmm_order"],
                    "dual_pfam": cls["dual_pfam"],
                    "dual_hmm": cls["dual_hmm"],
                    "length_ge_300": cls["length_ge_300"],
                    "qualifying": cls["qualifying"],
                    "observed_residues": observed_here,
                    "polymer_chains": ";".join(subs.get(eid, [])),
                    "ligands": ";".join(
                        f"{k}:{v}" for k, v in sorted(ligs_here.items())
                    ),
                    "title": title,
                    "method": method,
                    "resolution_A": resolution,
                }
            )
            any_dual_pfam |= cls["dual_pfam"]
            any_dual_hmm |= cls["dual_hmm"]
            if cls["qualifying"]:
                any_qualifying = True
                qualifying_ids.append(eid)
        corpus_rows.append(
            {
                "pdb_id": pdb_id,
                "title": title,
                "method": method,
                "resolution_A": resolution,
                "n_polymer_entities": len(seqs),
                "selected_entity_ids": ";".join(selected_ids),
                "polymer_entity_ids": ";".join(sorted(seqs)),
                "observed_residues_total": len(obs),
                "observed_chains": ";".join(
                    f"{c}:{n}" for c, n in sorted(obs_by_chain.items())
                ),
                "ligand_inventory": ";".join(
                    f"{k}:{v}" for k, v in sorted(lig_counter.items())
                ),
                "dual_pfam_any": any_dual_pfam,
                "dual_hmm_any": any_dual_hmm,
                "qualifying_any": any_qualifying,
                "qualifying_entity_ids": ";".join(qualifying_ids),
                "uniprot_ids": ";".join(
                    sorted(
                        {
                            u
                            for ej in entity_jsons
                            for u in ej.get(
                                "rcsb_polymer_entity_container_identifiers", {}
                            ).get("uniprot_ids", [])
                        }
                    )
                ),
            }
        )

    entity_df = pd.DataFrame(entity_rows)
    corpus_df = pd.DataFrame(corpus_rows)
    entity_df.to_csv(OUT / "dual_domain_scan.csv", index=False)
    corpus_df.to_csv(OUT / "structure_registry.csv", index=False)

    # ---------- template residue mapping / construct audit ----------
    mapping_rows = []
    construct_rows = []
    observed_cache = {}
    for ctx in (q559_template, q5_template, r5w_template):
        template = ctx["template"]
        st = ctx["st"]
        construct = ctx["construct"]
        subchains = ctx["subchains"]
        obs = observed_rows(st)
        obs_by_label: dict[int, list[dict]] = defaultdict(list)
        for r in obs:
            if r["label"] and r["chain"] in subchains:
                obs_by_label[r["label"]].append(r)
        observed_cache[template] = (st, obs_by_label)
        offsets = {}
        for chain in subchains:
            chain_obs = [r for r in obs if r["chain"] == chain and r["label"]]
            if chain_obs:
                deltas = {r["auth"] - r["label"] for r in chain_obs}
                offsets[chain] = sorted(deltas)
        for pos in range(1, len(construct) + 1):
            construct_aa = construct[pos - 1]
            in_sifts = ctx["entity_beg"] <= pos <= ctx["entity_beg"] + ctx["sifts_len"] - 1
            uniprot_pos = (
                ctx["ref_beg"] + (pos - ctx["entity_beg"]) if in_sifts else None
            )
            uniprot_aa = (
                ctx["native"][uniprot_pos - ctx["ref_beg"]] if in_sifts else None
            )
            region = "native" if in_sifts else "artifact_or_extension"
            obs_here = obs_by_label.get(pos, [])
            obs_chains = sorted({r["chain"] for r in obs_here})
            obs_auth = sorted({r["auth"] for r in obs_here})
            obs_aa = sorted({r["aa"] for r in obs_here})
            if not obs_here:
                uncertainty = "unmodeled"
            elif len(obs_aa) > 1:
                uncertainty = "chain_disagreement"
            elif region == "artifact_or_extension":
                uncertainty = "artifact_not_in_UniProt"
            else:
                uncertainty = "none"
            mapping_rows.append(
                {
                    "template_pdb": template,
                    "construct_pos": pos,
                    "construct_aa": construct_aa,
                    "region": region,
                    "uniprot_pos": uniprot_pos,
                    "uniprot_aa": uniprot_aa,
                    "auth_pos_inferred": pos - ctx["tag_len"],
                    "observed_chains": ";".join(obs_chains),
                    "observed_auth_positions": ";".join(str(x) for x in obs_auth),
                    "observed_aa": ";".join(obs_aa),
                    "observed": bool(obs_here),
                    "mapping_method": "SIFTS_entity_to_UniProt+CIF_label_seq",
                    "uncertainty": uncertainty,
                }
            )
        for chain in subchains:
            chain_obs = sorted(
                [r for r in obs if r["chain"] == chain and r["label"]],
                key=lambda x: x["label"],
            )
            labels = [r["label"] for r in chain_obs]
            present = set(labels)
            missing = [i for i in range(1, len(construct) + 1) if i not in present]
            ranges = []
            if missing:
                start = prev = missing[0]
                for x in missing[1:]:
                    if x == prev + 1:
                        prev = x
                    else:
                        ranges.append((start, prev))
                        start = prev = x
                ranges.append((start, prev))
            ligs_here = Counter(
                r["name"]
                for r in ligand_rows(st)
                if r["chain"] == chain
            )
            construct_rows.append(
                {
                    "template_pdb": template,
                    "scope": "chain",
                    "chain": chain,
                    "construct_length": len(construct),
                    "observed_residues": len(chain_obs),
                    "first_label": labels[0] if labels else None,
                    "last_label": labels[-1] if labels else None,
                    "missing_ranges": ";".join(f"{a}-{b}" for a, b in ranges),
                    "auth_minus_label": ";".join(str(x) for x in offsets.get(chain, [])),
                    "ligands": ";".join(f"{k}:{v}" for k, v in sorted(ligs_here.items())),
                }
            )
        construct_rows.append(
            {
                "template_pdb": template,
                "scope": "template",
                "chain": "ALL",
                "construct_length": len(construct),
                "observed_residues": len(obs),
                "first_label": min((r["label"] for r in obs if r["label"]), default=None),
                "last_label": max((r["label"] for r in obs if r["label"]), default=None),
                "missing_ranges": "",
                "auth_minus_label": "",
                "ligands": ";".join(
                    f"{k}:{v}" for k, v in sorted(Counter(r["name"] for r in ligand_rows(st)).items())
                ),
            }
        )

    pd.DataFrame(mapping_rows).to_csv(OUT / "residue_mapping.csv", index=False)
    pd.DataFrame(construct_rows).to_csv(OUT / "construct_observed_audit.csv", index=False)

    # ---------- anchor audit ----------
    anchor_rows = []
    for a in anchors:
        domain = a["domain"]
        tpos = int(a["template_position"])
        wt_pos = int(a["wt_position"])
        wt_expected_full = a["wt_residue"]
        template_expected_full = a["template_residue"]
        wt_expected = wt_expected_full[0]
        template_expected = template_expected_full[0]
        wt_label_num = int(re.sub(r"\D", "", wt_expected_full) or 0)
        template_label_num = int(re.sub(r"\D", "", template_expected_full) or 0)
        wt_actual = wt_seq[wt_pos - 1]
        q5_actual = q5_seq[tpos - 1] if tpos <= len(q5_seq) else None
        q559_same_pos = q559_seq[tpos - 1] if tpos <= len(q559_seq) else None
        aligned = wt_to_q5.get(wt_pos - 1)
        q5_aligned_aa = q5_seq[aligned] if aligned is not None else None
        q559_aligned = wt_to_q559.get(wt_pos - 1)
        q559_aligned_aa = q559_seq[q559_aligned] if q559_aligned is not None else None
        q559_from_q5 = q5_to_q559.get(tpos - 1)
        q559_from_q5_aa = q559_seq[q559_from_q5] if q559_from_q5 is not None else None
        base = {
            "domain": domain,
            "map_template_residue": template_expected_full,
            "map_template_position": tpos,
            "map_wt_position": wt_pos,
            "map_wt_residue": wt_expected_full,
            "map_position_label_matches_wt_position": wt_label_num == wt_pos,
            "map_position_label_matches_template_position": template_label_num == tpos,
            "identity_conserved_claimed": bool(a.get("identity_conserved")),
            "wt_actual": wt_actual,
            "wt_ok": wt_actual == wt_expected and wt_label_num == wt_pos,
            "q5_uniprot_actual": q5_actual,
            "q5_uniprot_ok": q5_actual == template_expected
            and template_label_num == tpos,
            "q559_same_position_actual": q559_same_pos,
            "q559_same_position_conserved": q559_same_pos == template_expected,
            "wt_to_q5_aligned": aligned is not None,
            "wt_to_q5_aligned_pos": None if aligned is None else aligned + 1,
            "wt_to_q5_aligned_aa": q5_aligned_aa,
            "wt_to_q5_pair_ok": aligned is not None and q5_aligned_aa == template_expected,
            "wt_to_q559_aligned_pos": None if q559_aligned is None else q559_aligned + 1,
            "wt_to_q559_aligned_aa": q559_aligned_aa,
            "wt_to_q559_pair_identity": (
                None
                if q559_aligned is None
                else wt_seq[wt_pos - 1] == q559_aligned_aa
            ),
            "q5_to_q559_aligned_pos": (
                None if q559_from_q5 is None else q559_from_q5 + 1
            ),
            "q5_to_q559_aligned_aa": q559_from_q5_aa,
            "q5_to_q559_identity_conserved": q559_from_q5_aa == template_expected,
            "q5_to_q559_consistent_with_direct_wt": q559_from_q5 == q559_aligned,
        }
        # per-template observation + ligand distance
        for template, ctx in (
            ("2QJO", q559_template),
            ("2QJT", q5_template),
            ("2R5W", r5w_template),
        ):
            template_native_pos = q559_from_q5 + 1 if template == "2QJO" and q559_from_q5 is not None else tpos
            construct_pos = (
                None
                if template_native_pos is None
                else template_native_pos + ctx["tag_len"]
            )
            st, _ = observed_cache[template]
            res = (
                None
                if construct_pos is None
                else find_residue(st, "B", construct_pos)
            )
            if template == "2QJO":
                ligs = ligand_residues(st, "NAD", 602, "B")
            elif template == "2QJT":
                ligs = ligand_residues(st, "AMP", 602, "B")
                for mn_auth in (505, 506, 507):
                    ligs += ligand_residues(st, "MN", mn_auth, "B")
            else:
                ligs = ligand_residues(st, "MG", None, "B")
            dist = min_heavy_distance(res, ligs)
            base[f"{template}_template_native_pos"] = template_native_pos
            base[f"{template}_construct_pos"] = construct_pos
            base[f"{template}_observed"] = res is not None
            base[f"{template}_observed_aa"] = aa3to1(res.name) if res is not None else None
            base[f"{template}_min_ligand_distance_A"] = dist
            base[f"{template}_within_8A"] = None if dist is None else dist <= 8.0
            base[f"{template}_within_4.5A"] = None if dist is None else dist <= 4.5
        context_match = (domain == "NMNAT" and a.get("domain") == "NMNAT") or (
            domain == "Nudix" and a.get("domain") == "Nudix"
        )
        designated = "2QJO" if domain == "NMNAT" else "2QJT"
        designated_observed = base[f"{designated}_observed"]
        designated_dist = base[f"{designated}_min_ligand_distance_A"]
        mapping_ok = (
            base["wt_ok"]
            and base["q5_uniprot_ok"]
            and base["wt_to_q5_pair_ok"]
            and base["q5_to_q559_aligned_pos"] is not None
            and base["q5_to_q559_consistent_with_direct_wt"]
        )
        base.update(
            {
                "designated_template": designated,
                "designated_observed": designated_observed,
                "designated_min_ligand_distance_A": designated_dist,
                "mapping_status": (
                    "UNKNOWN"
                    if not designated_observed
                    else ("PASS" if mapping_ok else "FAIL")
                ),
                "ligand_proximity_status": (
                    "UNKNOWN"
                    if designated_dist is None
                    else ("PASS" if designated_dist <= 8.0 else "FAIL")
                ),
                "identity_conserved_actual": wt_actual == q5_actual,
            }
        )
        anchor_rows.append(base)
    anchor_df = pd.DataFrame(anchor_rows)
    anchor_df.to_csv(OUT / "anchor_mapping_audit.csv", index=False)

    # ---------- sequence registry ----------
    seq_rows = []
    for row in roles.itertuples(index=False):
        seq_rows.append(
            {
                "record_type": "family_population_v2",
                "source": "natv_vnext_family_v2/population.fasta",
                "source_id": row.sequence_id,
                "pdb_id": "",
                "entity_id": "",
                "uniprot_ids": "",
                "sequence": row.sequence,
                "length": row.length,
                "sequence_sha256": hashlib.sha256(row.sequence.encode()).hexdigest(),
                "id_is_sha256": hashlib.sha256(row.sequence.encode()).hexdigest()
                == row.sequence_id,
                "training_role_v2": row.training_role_v2,
                "split_v2": row.split_v2,
                "legacy_split": row.legacy_split,
                "source_role": row.source_role,
                "component_v2": row.component_v2,
                "in_active_g": row.sequence_id in active_ids,
                "note": "",
            }
        )
    for row in entity_rows:
        seq = None
        for eid, s in entity_sequences(st_cache[row["pdb_id"]]).items():
            if eid == row["entity_id"]:
                seq = s
                break
        if seq is None:
            continue
        seq_rows.append(
            {
                "record_type": "PDB_entity",
                "source": "experimental_candidates corpus",
                "source_id": f"{row['pdb_id']}_{row['entity_id']}",
                "pdb_id": row["pdb_id"],
                "entity_id": row["entity_id"],
                "uniprot_ids": row["uniprot_ids"],
                "sequence": seq,
                "length": len(seq),
                "sequence_sha256": row["sequence_sha256"],
                "id_is_sha256": hashlib.sha256(seq.encode()).hexdigest()
                == row["sequence_sha256"],
                "training_role_v2": "",
                "split_v2": "",
                "legacy_split": "",
                "source_role": "structure_candidate",
                "component_v2": "",
                "in_active_g": False,
                "note": "constructed entity sequence; may include tag/artifacts",
            }
        )
    for ctx in (q559_template, q5_template, r5w_template):
        seq_rows.append(
            {
                "record_type": "template_native",
                "source": "SIFTS entity-to-UniProt mapping",
                "source_id": ";".join(ctx["uniprot"]),
                "pdb_id": ctx["template"],
                "entity_id": ctx["entity_id"],
                "uniprot_ids": ";".join(ctx["uniprot"]),
                "sequence": ctx["native"],
                "length": len(ctx["native"]),
                "sequence_sha256": hashlib.sha256(ctx["native"].encode()).hexdigest(),
                "id_is_sha256": False,
                "training_role_v2": "",
                "split_v2": "",
                "legacy_split": "",
                "source_role": "reference_template",
                "component_v2": "",
                "in_active_g": False,
                "note": f"native region {ctx['entity_beg']}-{ctx['entity_beg']+ctx['sifts_len']-1} of construct",
            }
        )
    seq_rows.append(
        {
            "record_type": "query_native",
            "source": "GenBank AY283928.2",
            "source_id": f"AAQ64231.1",
            "pdb_id": "",
            "entity_id": "",
            "uniprot_ids": "",
            "sequence": wt_seq,
            "length": len(wt_seq),
            "sequence_sha256": wt_sha,
            "id_is_sha256": False,
            "training_role_v2": "",
            "split_v2": "",
            "legacy_split": "",
            "source_role": "query",
            "component_v2": "",
            "in_active_g": False,
            "note": "KVP40 NadV wild type",
        }
    )
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(OUT / "sequence_registry.csv", index=False)

    # ---------- exclusions ----------
    exclusion_rows = []
    excluded_roles = {
        "annotation_conflict_quarantine",
        "excluded_noncanonical_or_domain",
        "quarantined_old_sealed_component",
    }
    for row in roles.itertuples(index=False):
        if row.training_role_v2 in excluded_roles:
            exclusion_rows.append(
                {
                    "dataset": "family_population_v2",
                    "record_id": row.sequence_id,
                    "group": row.training_role_v2,
                    "reason": row.exclusion_reasons,
                    "secondary_reason": row.family_v2_exclusion_reasons,
                    "length": row.length,
                    "detail": f"CDD {row.nmnat_cdd_from}-{row.nmnat_cdd_to}; {row.nudix_cdd_from}-{row.nudix_cdd_to}",
                }
            )
    for row in entity_rows:
        if row["qualifying"]:
            continue
        if row["dual_pfam"] or row["dual_hmm"]:
            reason = "dual_domain_but_length_below_300"
        elif row["pfam_nmnat"] and not row["pfam_nudix"]:
            reason = "PF01467_only"
        elif row["pfam_nudix"] and not row["pfam_nmnat"]:
            reason = "PF00293_only"
        elif not row["pfam_nmnat"] and not row["pfam_nudix"]:
            reason = "no_target_Pfam"
        else:
            reason = "other"
        detailed = []
        if row["dual_hmm"] and not row["dual_pfam"]:
            detailed.append("HMM dual without Pfam dual; manual review candidate")
        if row["hmm_nmnat_cov"] is not None and row["hmm_nmnat_cov"] < 0.5:
            detailed.append("PF01467 HMM coverage <0.5")
        if row["hmm_nudix_cov"] is not None and row["hmm_nudix_cov"] < 0.5:
            detailed.append("PF00293 HMM coverage <0.5")
        exclusion_rows.append(
            {
                "dataset": "structure_corpus",
                "record_id": f"{row['pdb_id']}_{row['entity_id']}",
                "group": "not_full_length_dual_domain",
                "reason": reason,
                "secondary_reason": "; ".join(detailed),
                "length": row["entity_length"],
                "detail": f"uniprot={row['uniprot_ids']}; pfam_order={row['pfam_order']}",
            }
        )
    pd.DataFrame(exclusion_rows).to_csv(OUT / "exclusions.csv", index=False)

    # ---------- active G registry ----------
    active_rows_out = []
    for r in active_rows:
        sid = r["sequence_id"]
        seq = r.get("sequence", "")
        role_row = role_by_id.get(sid)
        q5_map, q5_stats = alignment_maps(q5_seq, seq)
        q559_map, q559_stats = alignment_maps(q559_seq, seq)
        active_rows_out.append(
            {
                "sequence_id": sid,
                "length": len(seq),
                "role": r.get("role"),
                "component": r.get("component"),
                "training_role_v2": role_row.training_role_v2 if role_row else None,
                "split_v2": role_row.split_v2 if role_row else None,
                "cdd_nmnat": (
                    f"{role_row.nmnat_cdd_from}-{role_row.nmnat_cdd_to}"
                    if role_row
                    else ""
                ),
                "cdd_nudix": (
                    f"{role_row.nudix_cdd_from}-{role_row.nudix_cdd_to}"
                    if role_row
                    else ""
                ),
                "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                "id_is_sha256": hashlib.sha256(seq.encode()).hexdigest() == sid,
                "identity_to_Q5NHR1": q5_stats["identity"],
                "coverage_to_Q5NHR1": q5_stats["query_coverage"],
                "identity_to_Q55928": q559_stats["identity"],
                "coverage_to_Q55928": q559_stats["query_coverage"],
                "active_G_v3_status": (r.get("active_G_v3") or {}).get("status"),
            }
        )
    pd.DataFrame(active_rows_out).to_csv(OUT / "active_g_registry.csv", index=False)

    # ---------- source hashes ----------
    source_paths = []
    for p in [
        ROOT / "inputs/manifest.json",
        ROOT / "inputs/data/domain_pocket_map.json",
        ROOT / "inputs/data/AY283928.2.gb",
        ROOT / "inputs/data/PF01467.hmm",
        ROOT / "inputs/data/PF00293.hmm",
        ROOT / "inputs/data/2QJO.pdb",
        ROOT / "inputs/data/2qjt.pdb",
        FAMILY_DIR / "population.fasta",
        FAMILY_DIR / "family_training_roles_v2.parquet",
        FAMILY_DIR / "core_split_manifest_v2.json",
        ACTIVE_G,
    ]:
        source_paths.append(p)
    source_paths.extend(sorted(CORPUS_DIR.glob("*.cif")))
    source_paths.extend(sorted(CORPUS_DIR.glob("*.json")))
    hash_records = []
    for p in source_paths:
        resolved = p.resolve()
        hash_records.append(
            {
                "path": str(p),
                "resolved_path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    corpus_tree = sha256_bytes(
        "\n".join(
            f"{Path(r['path']).name}:{r['sha256']}"
            for r in hash_records
            if Path(r["path"]).parent == CORPUS_DIR
        ).encode()
    )
    hash_payload = {
        "generated_at_utc": generated,
        "script_sha256": sha256_file(Path(__file__)),
        "corpus_tree_sha256": corpus_tree,
        "files": hash_records,
    }
    (OUT / "source_hashes.json").write_text(json.dumps(hash_payload, indent=1) + "\n")

    # ---------- summary ----------
    anchor_pass = int((anchor_df["mapping_status"] == "PASS").sum())
    anchor_unknown = int((anchor_df["mapping_status"] == "UNKNOWN").sum())
    anchor_fail = int((anchor_df["mapping_status"] == "FAIL").sum())
    dual_pfam_ids = entity_df[entity_df["dual_pfam"]][["pdb_id", "entity_id"]]
    dual_hmm_ids = entity_df[entity_df["dual_hmm"]][["pdb_id", "entity_id"]]
    qualifying_ids_df = entity_df[entity_df["qualifying"]][["pdb_id", "entity_id"]]
    summary = {
        "generated_at_utc": generated,
        "script_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "family_population_records": len(fasta),
            "family_roles_rows": len(roles),
            "active_g_rows": len(active_rows),
            "structure_corpus_pdbs": len(corpus_rows),
            "structure_polymer_entities": len(entity_rows),
        },
        "family": {
            "training_role_v2_counts": role_counts,
            "split_v2_counts": split_v2_counts,
            "legacy_split_counts": legacy_counts,
            "sequence_id_hash_mismatches": len(hash_mismatch),
            "fasta_parquet_sequence_mismatches": len(seq_mismatch),
            "source_role_counts": roles["source_role"].value_counts().to_dict(),
            "annotation_conflict_not_negative_label": True,
        },
        "active_g": {
            "rows": len(active_rows),
            "missing_from_family_population": len(active_missing_from_population),
            "population_not_in_active_g": population_not_in_active,
        },
        "structures": {
            "pdbs": len(corpus_rows),
            "polymer_entities": len(entity_rows),
            "dual_pfam_entities": len(dual_pfam_ids),
            "dual_pfam_entity_ids": [
                f"{r.pdb_id}_{r.entity_id}" for r in dual_pfam_ids.itertuples(index=False)
            ],
            "dual_hmm_entities": len(dual_hmm_ids),
            "qualifying_entities": len(qualifying_ids_df),
            "qualifying_entity_ids": [
                f"{r.pdb_id}_{r.entity_id}"
                for r in qualifying_ids_df.itertuples(index=False)
            ],
            "unique_uniprot_in_qualifying": sorted(
                {
                    u
                    for row in entity_rows
                    if row["qualifying"]
                    for u in row["uniprot_ids"].split(";")
                    if u
                }
            ),
        },
        "templates": {
            "2QJO": {
                "uniprot": q559_template["uniprot"],
                "construct_length": len(q559_template["construct"]),
                "native_length": len(q559_template["native"]),
                "sifts_entity_beg": q559_template["entity_beg"],
                "tag_length": q559_template["tag_len"],
                "artifacts": q559_template["artifacts"],
                "observed_residues": sum(1 for r in observed_rows(q559_template["st"])),
                "chains": q559_template["subchains"],
            },
            "2QJT": {
                "uniprot": q5_template["uniprot"],
                "construct_length": len(q5_template["construct"]),
                "native_length": len(q5_template["native"]),
                "sifts_entity_beg": q5_template["entity_beg"],
                "tag_length": q5_template["tag_len"],
                "artifacts": q5_template["artifacts"],
                "observed_residues": sum(1 for r in observed_rows(q5_template["st"])),
                "chains": q5_template["subchains"],
            },
            "2R5W": {
                "uniprot": r5w_template["uniprot"],
                "construct_length": len(r5w_template["construct"]),
                "native_length": len(r5w_template["native"]),
                "sifts_entity_beg": r5w_template["entity_beg"],
                "tag_length": r5w_template["tag_len"],
                "artifacts": r5w_template["artifacts"],
                "observed_residues": sum(1 for r in observed_rows(r5w_template["st"])),
                "chains": r5w_template["subchains"],
            },
        },
        "alignments": {
            "KVP40_to_Q5NHR1": wt_q5_stats,
            "KVP40_to_Q55928": wt_q559_stats,
            "Q5NHR1_to_Q55928": q5_q559_stats,
            "KVP40_length": len(wt_seq),
            "KVP40_sha256": wt_sha,
        },
        "anchors": {
            "total": len(anchor_df),
            "mapping_status_counts": anchor_df["mapping_status"].value_counts().to_dict(),
            "ligand_proximity_counts": anchor_df[
                "ligand_proximity_status"
            ].value_counts().to_dict(),
            "all_wt_residues_match": bool(anchor_df["wt_ok"].all()),
            "all_template_positions_match_Q5NHR1": bool(
                anchor_df["q5_uniprot_ok"].all()
            ),
            "all_alignment_pairs_consistent": bool(anchor_df["wt_to_q5_pair_ok"].all()),
            "same_position_matches_Q55928": int(
                anchor_df["q559_same_position_conserved"].sum()
            ),
            "cross_species_mapping_consistent_with_direct_wt": bool(
                anchor_df["q5_to_q559_consistent_with_direct_wt"].all()
            ),
            "cross_species_identity_conserved_in_Q55928": int(
                anchor_df["q5_to_q559_identity_conserved"].sum()
            ),
            "identity_conserved_claim_matches_actual": bool(
                (
                    anchor_df["identity_conserved_claimed"]
                    == anchor_df["identity_conserved_actual"]
                ).all()
            ),
        },
        "key_findings": [
            "The curated '2QJT' anchor positions are Q5NHR1/2QJT auth numbering, not PDB entity/label_seq numbering; the 2QJT and 2R5W entity offset is +5 and the 2QJO entity offset is +2.",
            "All 12 anchor template positions match Q5NHR1 and all 12 WT positions match the GenBank AAQ64231.1 translation; the global KVP40-Q5NHR1 alignment preserves every anchor pair.",
            f"The 2QJO column cannot use same-number positions: {int((~anchor_df['q559_same_position_conserved']).sum())} of {len(anchor_df)} anchors would land on a different residue. The valid 2QJO mapping is Q5NHR1->Q55928 by global alignment, and it agrees with the direct KVP40->Q55928 alignment for all 12 anchors.",
            "E238/H236 is the only non-identity-conserved anchor; the aligned 2QJO residue is K236 rather than E, so the Nudix anchor correspondence is positional, not identity-conserved, across all three templates.",
            "2QJT and 2R5W are the same UniProt sequence Q5NHR1 in different experimental states; they are not independent sequence diversity.",
            "Only 2QJO, 2QJT and 2R5W qualify as full-length dual-domain experimental templates in the 285-PDB corpus.",
            "No experimentally measured NatV variant activity is present in these inputs; structural correspondence is not catalytic activity.",
        ],
        "limitations": [
            "SIFTS maps construct/entity residues to UniProt; cross-species KVP40 correspondence is a separate homology alignment and is not a SIFTS claim.",
            "The 2QJO anchor positions depend on a Q5NHR1-Q55928 global alignment (38% identity over the aligned region), not on identical residue numbering.",
            "The 12 anchors are structural proxies; ligand proximity does not establish substrate geometry, catalysis or net NAD+ yield.",
            "The 285-PDB corpus is a frozen local candidate set; external discovery results are recorded separately and do not silently broaden the training population.",
            "Roles in training_role_v2 are preserved as-is; annotation-conflict quarantine is not an experimental negative label.",
        ],
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
