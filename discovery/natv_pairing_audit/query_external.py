#!/usr/bin/env python
"""Bounded external NatV discovery: RCSB sequence search, UniProt dual-Pfam,
InterPro IPR041750 and AlphaFold DB metadata.

Metadata is fetched first. Coordinates are downloaded only for entities that
pass the dual-domain screen. Raw responses and hashes are retained.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import gemmi
import pandas as pd
import pyhmmer

import build_local_audit as audit

OUT = Path(__file__).resolve().parent
EXT = OUT / "external"
RCSB_SEQ = EXT / "rcsb_sequence_search"
RCSB_META = EXT / "rcsb_metadata"
RCSB_CIF = EXT / "rcsb_shortlisted_cif"
UNIPROT_RAW = EXT / "uniprot_raw"
ALPHAFOLD_RAW = EXT / "alphafold_raw"
for p in (EXT, RCSB_SEQ, RCSB_META, RCSB_CIF, UNIPROT_RAW, ALPHAFOLD_RAW):
    p.mkdir(parents=True, exist_ok=True)

CORPUS_DIR = audit.CORPUS_DIR
CORPUS_IDS = set()
CORPUS_PDBS = set()
corpus_scan = OUT / "dual_domain_scan.csv"
if corpus_scan.exists():
    for row in pd.read_csv(corpus_scan).itertuples(index=False):
        CORPUS_IDS.add(f"{row.pdb_id}_{row.entity_id}")
        CORPUS_PDBS.add(row.pdb_id)
KNOWN_UNIPROT = {"Q55928", "Q5NHR1"}
LOG: list[dict] = []
UNIPROT_TOTALS: dict[str, int] = {}
HMMS = {
    name: pyhmmer.plan7.HMMFile(str(path)).read()
    for name, path in audit.HMM_PATHS.items()
}


def fetch_with_headers(
    url: str, path: Path, timeout: int = 60, data: bytes | None = None
) -> tuple[bytes, dict]:
    t0 = time.time()
    headers_req = {"User-Agent": "natvqvae-audit/1.0"}
    if data is not None:
        headers_req["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers_req)
    headers: dict[str, str] = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
            headers = {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
        headers = {k.lower(): v for k, v in exc.headers.items()}
    except Exception as exc:  # network failure is a recorded outcome
        raw = str(exc).encode()
        status = -1
    path.write_bytes(raw)
    LOG.append(
        {
            "url": url,
            "path": str(path),
            "status": status,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "elapsed_s": round(time.time() - t0, 3),
        }
    )
    return raw, headers


def fetch(
    url: str, path: Path, timeout: int = 60, data: bytes | None = None
) -> bytes:
    raw, _ = fetch_with_headers(url, path, timeout, data=data)
    return raw


def rcsb_sequence_search(seeds: dict[str, str]) -> dict[str, dict]:
    hits: dict[str, dict] = {}
    for name, seq in seeds.items():
        payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1,
                    "identity_cutoff": 0.3,
                    "sequence_type": "protein",
                    "value": seq,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {"paginate": {"start": 0, "rows": 100}},
        }
        (RCSB_SEQ / f"{name}_query.json").write_text(json.dumps(payload, indent=1))
        raw = fetch(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            RCSB_SEQ / f"{name}_response.json",
            data=json.dumps(payload).encode(),
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for hit in data.get("result_set", []):
            ident = hit["identifier"]
            hits.setdefault(ident, {"scores": {}})["scores"][name] = hit.get("score")
    return hits


def rcsb_metadata(hits: dict[str, dict]) -> list[dict]:
    rows = []
    for ident, hit in sorted(hits.items()):
        pdb_id, entity_id = ident.split("_", 1)
        ent_path = RCSB_META / f"{pdb_id}_{entity_id}.json"
        entry_path = RCSB_META / f"{pdb_id}_entry.json"
        if ent_path.exists():
            raw = ent_path.read_bytes()
        else:
            raw = fetch(
                f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}",
                ent_path,
            )
        try:
            ent = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry_path.exists():
            eraw = entry_path.read_bytes()
        else:
            eraw = fetch(
                f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", entry_path
            )
        try:
            entry = json.loads(eraw)
        except json.JSONDecodeError:
            entry = {}
        seq = ent.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
        if not seq:
            continue
        pfam = audit.pfam_intervals(ent)
        hmm = audit.hmm_hits(HMMS, seq)
        cls = audit.classify_entity(seq, pfam, hmm)
        uniprot = (
            ent.get("rcsb_polymer_entity_container_identifiers", {}).get(
                "uniprot_ids", []
            )
            or []
        )
        organisms = [
            o.get("ncbi_scientific_name")
            for o in ent.get("rcsb_entity_source_organism", [])
        ]
        title = entry.get("struct", {}).get("title", "")
        method = (entry.get("exptl") or [{}])[0].get("method", "")
        resolution = entry.get("rcsb_entry_info", {}).get("resolution_combined")
        resolution = resolution[0] if resolution else None
        if resolution is None:
            resolution = (entry.get("refine") or [{}])[0].get("ls_d_res_high")
        rows.append(
            {
                "entity_id": ident,
                "pdb_id": pdb_id,
                "polymer_entity_id": entity_id,
                "in_frozen_corpus": ident in CORPUS_IDS,
                "incremental_vs_corpus": ident not in CORPUS_IDS,
                "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                "entity_length": len(seq),
                "uniprot_ids": ";".join(uniprot),
                "organism": ";".join(x for x in organisms if x),
                "title": title,
                "method": method,
                "resolution_A": resolution,
                "pfam_nmnat": cls["pfam_nmnat"],
                "pfam_nudix": cls["pfam_nudix"],
                "pfam_order": cls["pfam_order"],
                "hmm_nmnat_cov": cls["hmm_nmnat_cov"],
                "hmm_nudix_cov": cls["hmm_nudix_cov"],
                "hmm_order": cls["hmm_order"],
                "dual_pfam": cls["dual_pfam"],
                "dual_hmm": cls["dual_hmm"],
                "qualifying_full_length_dual": cls["qualifying"],
                "score_KVP40": hit["scores"].get("KVP40"),
                "score_Q55928": hit["scores"].get("Q55928"),
                "score_Q5NHR1": hit["scores"].get("Q5NHR1"),
            }
        )
    return rows


def observed_and_ligands(pdb_id: str) -> dict:
    cif_path = RCSB_CIF / f"{pdb_id}.cif"
    if not cif_path.exists():
        fetch(f"https://files.rcsb.org/download/{pdb_id}.cif", cif_path)
    try:
        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
    except Exception as exc:
        return {"observed_error": str(exc)}
    obs = audit.observed_rows(st)
    ligs = audit.ligand_rows(st)
    from collections import Counter

    return {
        "observed_residues": len(obs),
        "observed_chains": ";".join(
            f"{c}:{n}" for c, n in sorted(Counter(r["chain"] for r in obs).items())
        ),
        "ligand_inventory": ";".join(
            f"{k}:{v}" for k, v in sorted(Counter(r["name"] for r in ligs).items())
        ),
    }


def uniprot_search(query: str, tag: str, max_pages: int = 10) -> list[dict]:
    results: list[dict] = []
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        + urllib.parse.urlencode(
            {
                "query": query,
                "format": "json",
                "size": 500,
                "fields": "accession,id,protein_name,organism_name,length,reviewed,xref_pdb,sequence",
            }
        )
    )
    page = 0
    while url and page < max_pages:
        raw, headers = fetch_with_headers(
            url, UNIPROT_RAW / f"{tag}_page{page}.json"
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            break
        if page == 0:
            UNIPROT_TOTALS[tag] = int(headers.get("x-total-results", 0) or 0)
        results.extend(data.get("results", []))
        page += 1
        if len(data.get("results", [])) < 500:
            break
        link = headers.get("link", "")
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        next_url = match.group(1) if match else None
        if not next_url:
            break
        url = next_url
    return results


def uniprot_rows(results: list[dict]) -> list[dict]:
    rows = []
    for r in results:
        seq = r.get("sequence", {}).get("value", "")
        if not seq:
            continue
        xrefs = r.get("uniProtKBCrossReferences", [])
        pdb_ids = sorted(
            {x["id"] for x in xrefs if x.get("database") == "PDB" and x.get("id")}
        )
        pfam_ids = sorted(
            {x["id"] for x in xrefs if x.get("database") == "Pfam" and x.get("id")}
        )
        interpro_ids = sorted(
            {x["id"] for x in xrefs if x.get("database") == "InterPro" and x.get("id")}
        )
        hmm = audit.hmm_hits(HMMS, seq)
        cls = audit.classify_entity(seq, {}, hmm)
        rows.append(
            {
                "accession": r.get("primaryAccession"),
                "entry_id": r.get("uniProtkbId"),
                "entry_type": r.get("entryType"),
                "reviewed": (r.get("entryType") or "").startswith(
                    "UniProtKB reviewed"
                ),
                "organism": r.get("organism", {}).get("scientificName"),
                "length": r.get("sequence", {}).get("length"),
                "protein_name": (
                    r.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value")
                ),
                "pdb_ids": ";".join(pdb_ids),
                "pfam_xrefs": ";".join(pfam_ids),
                "interpro_xrefs": ";".join(interpro_ids),
                "has_pdb": bool(pdb_ids),
                "pdb_not_in_corpus": any(p not in CORPUS_PDBS for p in pdb_ids)
                if pdb_ids
                else False,
                "hmm_nmnat_cov": cls["hmm_nmnat_cov"],
                "hmm_nudix_cov": cls["hmm_nudix_cov"],
                "hmm_order": cls["hmm_order"],
                "dual_hmm": cls["dual_hmm"],
                "qualifying_full_length_dual": cls["qualifying"],
                "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                "sequence": seq,
            }
        )
    return rows


def main() -> None:
    generated = datetime.now(timezone.utc).isoformat()
    seeds = {
        "KVP40": audit.parse_genbank_wt(audit.GENBANK)[1],
        "Q55928": audit.template_context("2QJO")["native"],
        "Q5NHR1": audit.template_context("2QJT")["native"],
    }
    hits = rcsb_sequence_search(seeds)
    rcsb_rows = rcsb_metadata(hits)
    for row in rcsb_rows:
        if row["qualifying_full_length_dual"] or row["dual_pfam"] or row["dual_hmm"]:
            row.update(observed_and_ligands(row["pdb_id"]))
    rcsb_df = pd.DataFrame(rcsb_rows).sort_values(
        ["qualifying_full_length_dual", "incremental_vs_corpus", "entity_id"],
        ascending=[False, False, True],
    )
    rcsb_df.to_csv(EXT / "incremental_structures.csv", index=False)

    dual_results = uniprot_search(
        "(xref:pfam-PF01467) AND (xref:pfam-PF00293)", "dual_pfam"
    )
    ipr_results = uniprot_search("xref:interpro-IPR041750", "ipr041750")
    dual_rows = uniprot_rows(dual_results)
    ipr_rows = uniprot_rows(ipr_results)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "sequence"} for r in dual_rows]
    ).to_csv(EXT / "uniprot_dual_pfam_all.csv", index=False)
    candidates = [
        r
        for r in dual_rows
        if r["qualifying_full_length_dual"]
        and r["length"]
        and 280 <= int(r["length"]) <= 450
        and r["accession"] not in KNOWN_UNIPROT
    ]
    # Deduplicate exact sequences, preferring reviewed entries.
    by_seq: dict[str, dict] = {}
    for r in sorted(candidates, key=lambda x: (not x["reviewed"], x["accession"])):
        by_seq.setdefault(r["sequence_sha256"], r)
    candidates = list(by_seq.values())
    pd.DataFrame(candidates).to_csv(EXT / "uniprot_candidates.csv", index=False)

    ipr_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "sequence"} for r in ipr_rows]
    )
    ipr_df.to_csv(EXT / "uniprot_ipr041750_all.csv", index=False)

    # AlphaFold metadata only for shortlisted sequences lacking experimental PDB.
    af_rows = []
    for r in sorted(
        [x for x in candidates if not x["has_pdb"]],
        key=lambda x: (not x["reviewed"], -max(x["hmm_nmnat_cov"] or 0, x["hmm_nudix_cov"] or 0)),
    )[:10]:
        acc = r["accession"]
        raw = fetch(
            f"https://alphafold.ebi.ac.uk/api/prediction/{acc}",
            ALPHAFOLD_RAW / f"{acc}.json",
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = []
        if isinstance(data, dict):
            data = [data]
        for entry in data:
            af_rows.append(
                {
                    "accession": acc,
                    "entry_id": r["entry_id"],
                    "reviewed": r["reviewed"],
                    "length": r["length"],
                    "af_model_entity_id": entry.get("modelEntityId"),
                    "af_latest_version": entry.get("latestVersion"),
                    "af_sequence_start": entry.get("sequenceStart"),
                    "af_sequence_end": entry.get("sequenceEnd"),
                    "af_pdb_url": entry.get("pdbUrl"),
                    "af_cif_url": entry.get("cifUrl"),
                    "af_pae_url": entry.get("paeImageUrl"),
                }
            )
    pd.DataFrame(af_rows).to_csv(EXT / "alphafold_metadata.csv", index=False)

    summary = {
        "generated_at_utc": generated,
        "seeds": {
            "KVP40": hashlib.sha256(seeds["KVP40"].encode()).hexdigest(),
            "Q55928": hashlib.sha256(seeds["Q55928"].encode()).hexdigest(),
            "Q5NHR1": hashlib.sha256(seeds["Q5NHR1"].encode()).hexdigest(),
        },
        "rcsb_sequence_search": {
            "union_polymer_entities": len(hits),
            "in_frozen_corpus": int(
                sum(1 for r in rcsb_rows if r["in_frozen_corpus"])
            ),
            "incremental_vs_corpus": int(
                sum(1 for r in rcsb_rows if r["incremental_vs_corpus"])
            ),
            "dual_pfam_entities": int(sum(1 for r in rcsb_rows if r["dual_pfam"])),
            "dual_hmm_entities": int(sum(1 for r in rcsb_rows if r["dual_hmm"])),
            "qualifying_full_length_dual": int(
                sum(1 for r in rcsb_rows if r["qualifying_full_length_dual"])
            ),
            "new_qualifying_not_in_corpus": [
                r["entity_id"]
                for r in rcsb_rows
                if r["qualifying_full_length_dual"] and r["incremental_vs_corpus"]
            ],
        },
        "uniprot_dual_pfam": {
            "fetched_records": len(dual_rows),
            "query_total_expected": UNIPROT_TOTALS.get("dual_pfam", 880),
            "qualifying_full_length_dual": int(
                sum(1 for r in dual_rows if r["qualifying_full_length_dual"])
            ),
            "deduplicated_candidates": len(candidates),
            "candidates_with_pdb": int(sum(1 for r in candidates if r["has_pdb"])),
            "candidates_without_pdb": int(sum(1 for r in candidates if not r["has_pdb"])),
        },
        "uniprot_ipr041750": {
            "fetched_records": len(ipr_rows),
            "query_total_expected": UNIPROT_TOTALS.get("ipr041750", 0),
            "qualifying_full_length_dual": int(
                sum(1 for r in ipr_rows if r["qualifying_full_length_dual"])
            ),
        },
        "alphafold_metadata_records": len(af_rows),
        "negative_result": not any(
            r["qualifying_full_length_dual"] and r["incremental_vs_corpus"]
            for r in rcsb_rows
        ),
    }
    (EXT / "external_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (EXT / "query_log.json").write_text(json.dumps(LOG, indent=1) + "\n")
    hashes = [
        {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(EXT.rglob("*"))
        if p.is_file() and p.name not in ("source_hashes.json",)
    ]
    (EXT / "source_hashes.json").write_text(
        json.dumps(
            {"generated_at_utc": generated, "files": hashes},
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
