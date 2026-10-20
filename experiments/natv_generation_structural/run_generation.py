"""Bounded frozen-H2 generation and structure-informed annotation.

Declared scope: two frozen H2 checkpoints, matched 4/8-edit streams, 8 streams
per budget, 32 proposals, seven masked-edit stages, no training, folding,
docking, MD, remote download, wet-lab work or test-sequence payload reads.

The structural basis is a discovery artifact (formal_input_manifest=false).
It is used only to annotate where substitutions fall relative to the audited
experimental-template anchors. It is not a training or selection input.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

NATV_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("NATV_SOURCE_ROOT", "/home/yhshy/git_files/nad_qvae")).resolve()
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from nad_qvae.design import multimodal_deterministic_runtime_v2 as runtime
from nad_qvae.design.g0_benchmark import constrained_draw, stream_plan, verify_g0_sequence
from nad_qvae.design.h2_execution_v1 import ROOT as H2_ROOT, SEReader
from nad_qvae.design.h2_generation_v1 import H2Retrieval
from nad_qvae.design.h2_model_v1 import H2RecoveryModel
from nad_qvae.design.sequence_corpus import AA
from nad_qvae.design.se_full import masked_sequence
from nad_qvae.design.vnext_evaluation_contracts import FrozenHFMaskedEncoder
from nad_qvae.design.vnext_family_v2 import ordered_pair, parse_domains
from nad_qvae.design.formal_generation import internal_metrics

RUN = NATV_ROOT / "experiments/natv_generation_structural"
H2_CONTRACT = Path("config/natv/h2_architecture_generation_comparison_v1.json")
H2_RUN = Path("experiments/run_reports/natv_h2_architecture_generation_v1_attempt1")
FORMAL_DATA = Path("data/processed/releases/natv_formal_training_data_v1")
FORMAL_PLAN = FORMAL_DATA / "input_plan.json"
DATA_READY = Path("experiments/run_reports/natv_formal_training_data_v1/data_ready.json")
FORMAL_GEN = Path("experiments/run_reports/natv_formal_generation_v1_attempt1")
FORMAL_CANDIDATES = FORMAL_GEN / "frozen_WP5_candidates.csv"
FORMAL_CDD = FORMAL_GEN / "frozen_WP5_CDD_domains.csv"
FORMAL_CE = FORMAL_GEN / "internal_PLM_scores.csv"
WT_SOURCE = Path("config/natv/se_full_g0_v1.json")
CONSTRAINTS = Path("config/natv/g0_target_constraints_v1.json")
RUNTIME_CONTRACT = Path("config/natv/multimodal_deterministic_runtime_v2.json")
TRAIN_CONTRACT = Path("config/natv/formal_training_v1.json")
CORE = Path("data/processed/releases/natv_vnext_family_v2")
STRUCT_BASIS = NATV_ROOT / "discovery/natv_pairing_audit/structural_interpretation_basis.json"
STRUCT_SCRIPT = NATV_ROOT / "discovery/natv_pairing_audit/build_structural_basis.py"

TRAINING_SEED = 20261081
STREAM_SEED = 20261201
PRIOR_SEED = 20261202
EDIT_BUDGETS = [4, 8]
STREAMS_PER_BUDGET = 8
SWEEPS = 3
TEMPERATURE = 1.0
BATCH_SIZE = 4
QUALITY_BLOCKS = 8
T135_INDEX = 134
H236_POSITION = 236
ARMS = [
    {
        "label": "H2_no_latent",
        "architecture": "no_latent",
        "checkpoint": H2_RUN / "fits/no_latent" / str(TRAINING_SEED) / "epoch02.pt",
        "sha256": "6bdecac9f55604a66a84fac0090626bff55b4bf0455e363d201de3cd5c784e06",
        "validation_CE": 1.654838123953725,
    },
    {
        "label": "H2_continuous_mmd",
        "architecture": "continuous_mmd",
        "checkpoint": H2_RUN / "fits/continuous_mmd" / str(TRAINING_SEED) / "epoch02.pt",
        "sha256": "ce12ef55b45dc38696a34776e35553044d19940db01a9cc1670af9422e8d2b6c",
        "validation_CE": 1.6573922181489764,
    },
]
WT_CE_REFERENCE = 1.8650238171350921


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(path):
    path = Path(path).absolute()
    for base, label in ((SOURCE_ROOT, "nad_qvae"), (NATV_ROOT, "natvqvae")):
        try:
            return label, path.relative_to(base.absolute()).as_posix()
        except ValueError:
            continue
    return "absolute", path.as_posix()


def input_entry(path, role, used_by, expected=None, root_hint=None):
    path = Path(path)
    if root_hint == "natvqvae":
        full = NATV_ROOT / path
    else:
        full = SOURCE_ROOT / path if not path.is_absolute() else path
    require(full.exists(), f"Missing input: {full}")
    current = sha(full)
    if expected is not None:
        require(current == expected, f"Input hash changed: {full}")
    root, rel = relative(full)
    return {
        "root": root,
        "path": rel,
        "bytes": full.stat().st_size,
        "sha256": current,
        "role": role,
        "used_by": used_by,
    }


def cdd_assets():
    base = SOURCE_ROOT / "data/reference/ncbi_cdd_3.21"
    require(base.exists(), "CDD reference directory missing")
    db_files = sorted(base.glob("database/Cdd_NCBI*"))
    data_files = sorted((base / "rpsbproc_data").glob("*"))
    binaries = [
        base / "blast/ncbi-blast-2.17.0+/bin/rpsblast",
        base / "rpsbproc/RpsbProc-x64-linux/rpsbproc",
    ]
    require(len(db_files) >= 10 and db_files, "CDD rpsblast database files missing")
    require(len(data_files) >= 5, "rpsbproc data files missing")
    for binary in binaries:
        require(binary.exists() and os.access(binary, os.X_OK), f"CDD executable missing: {binary}")
    return db_files, data_files, binaries


def load_wt():
    source = read_json(SOURCE_ROOT / WT_SOURCE)
    wt = source["WT"]["sequence"]
    digest = source["WT"]["sha256"]
    require(len(wt) == 341 and digest == text_sha(wt), "WT identity changed")
    return wt, digest, source


def prepare_anchors(basis, wt, domains):
    require(basis["formal_input_manifest"] is False, "Structural basis must remain a discovery artifact")
    require(basis["target"]["sequence"] == wt, "Structural-basis target sequence differs from WT")
    require(basis["target"]["sequence_sha256"] == text_sha(wt), "Structural-basis target hash differs")
    require(basis["target"]["length"] == len(wt), "Structural-basis target length differs")
    require(basis["anchor_summary"]["count"] == 12, "Expected 12 audited anchors")
    require(basis["anchor_summary"]["mapping_status"] == "PASS 12/12", "Anchor mapping audit not passed")
    require(basis["anchor_summary"]["designated_ligand_proximity_status"] == "PASS 12/12",
            "Anchor ligand-proximity audit not passed")
    anchors = []
    for item in basis["anchor_table"]:
        position = int(item["wt_position"])
        residue = item["wt_residue"]
        require(1 <= position <= len(wt) and wt[position - 1] == residue,
                f"Anchor {position} does not match WT")
        domain = str(item["domain"]).upper()
        require(domain in domains, f"Unknown anchor domain {domain}")
        require(domain_of(position, domains) == domain, f"Anchor {position} outside declared domain")
        anchors.append({
            "domain": domain,
            "wt_position": position,
            "wt_residue": residue,
            "identity_conserved_across_templates": bool(item["identity_conserved_across_templates"]),
            "designated_template": item["designated_template"],
            "min_ligand_distance_A": float(item["min_ligand_distance_A"]),
            "mapping_status": item["mapping_status"],
            "ligand_proximity_status": item["ligand_proximity_status"],
            "variable_anchor": position == H236_POSITION,
        })
    require(len({a["wt_position"] for a in anchors}) == 12, "Duplicate anchor positions")
    return anchors


def domain_of(position, domains):
    for name, bounds in domains.items():
        if int(bounds[0]) <= int(position) <= int(bounds[1]):
            return name
    raise ValueError(f"Position {position} outside declared domains")


def parse_variants(value):
    result = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        require(len(item) >= 3 and item[1:-1].isdigit(), f"Malformed variant {item}")
        result.append((int(item[1:-1]), item[0], item[-1]))
    return result


def sequence_variants(sequence, wt):
    require(len(sequence) == len(wt), "Candidate length differs from WT")
    return [(index + 1, wt[index], sequence[index])
            for index in range(len(wt)) if sequence[index] != wt[index]]


def variant_text(variants):
    return ";".join(f"{old}{position}{new}" for position, old, new in variants)


def verify_candidate(sequence, wt, editable=None):
    require(len(sequence) == 341, "Candidate is not full-length 341 aa")
    require(all(letter in AA for letter in sequence), "Candidate contains noncanonical residues")
    require(sequence[T135_INDEX] != "A", "T135A quarantine violation")
    variants = sequence_variants(sequence, wt)
    if editable is not None:
        require(len(variants) == len(editable), "Candidate edit count differs from declared budget")
        require({v[0] - 1 for v in variants} == set(editable),
                "Candidate edits differ from the frozen stream positions")
    return variants


def nearest_anchor(position, anchors):
    return min(anchors, key=lambda a: (abs(a["wt_position"] - position), a["wt_position"]))


def substitution_rows(candidate, anchors, domains):
    rows = []
    for position, old, new in parse_variants(candidate["variants"]):
        direct = [a for a in anchors if a["wt_position"] == position]
        closest = direct[0] if direct else nearest_anchor(position, anchors)
        distance = abs(closest["wt_position"] - position)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "source_set": candidate["source_set"],
            "arm": candidate["arm"],
            "stream_type": candidate["stream_type"],
            "edit_budget": candidate["edit_budget"],
            "position_1based": position,
            "wt_aa": old,
            "mutant_aa": new,
            "variant": f"{old}{position}{new}",
            "domain": domain_of(position, domains),
            "direct_anchor": bool(direct),
            "anchor_wt_position": position if direct else "",
            "anchor_wt_residue": old if direct else "",
            "anchor_domain": closest["domain"],
            "anchor_identity_conserved": closest["identity_conserved_across_templates"],
            "anchor_variable": closest["variable_anchor"],
            "anchor_designated_template": closest["designated_template"],
            "anchor_min_ligand_distance_A": closest["min_ligand_distance_A"],
            "nearest_anchor_position": closest["wt_position"],
            "nearest_anchor_distance": distance,
            "within_4_sequence_positions": distance <= 4,
            "within_4_is_sequence_window_not_3d_distance": True,
        })
    return rows


def pair_details(hits):
    pair = ordered_pair(hits)
    nmnat_hits = [h for h in hits if h["accession"] in {"cd02165", "cd02168"}]
    nudix_hits = [h for h in hits if h["accession"] == "cd18873"]
    record = {
        "ordered_CDD_pair": pair is not None,
        "CDD_NMNAT_pssm": "",
        "CDD_NMNAT_accession": "",
        "CDD_NMNAT_range": "",
        "CDD_NMNAT_evalue": np.nan,
        "CDD_NMNAT_bitscore": np.nan,
        "CDD_NUDIX_pssm": "",
        "CDD_NUDIX_accession": "",
        "CDD_NUDIX_range": "",
        "CDD_NUDIX_evalue": np.nan,
        "CDD_NUDIX_bitscore": np.nan,
        "CDD_linker_length": np.nan,
        "n_NMNAT_family_hits": len(nmnat_hits),
        "n_cd18873_hits": len(nudix_hits),
    }
    if pair is not None:
        first, second = pair
        record.update({
            "CDD_NMNAT_pssm": first["pssm_id"],
            "CDD_NMNAT_accession": first["accession"],
            "CDD_NMNAT_range": f"{first['query_from']}-{first['query_to']}",
            "CDD_NMNAT_evalue": float(first["evalue"]),
            "CDD_NMNAT_bitscore": float(first["bitscore"]),
            "CDD_NUDIX_pssm": second["pssm_id"],
            "CDD_NUDIX_accession": second["accession"],
            "CDD_NUDIX_range": f"{second['query_from']}-{second['query_to']}",
            "CDD_NUDIX_evalue": float(second["evalue"]),
            "CDD_NUDIX_bitscore": float(second["bitscore"]),
            "CDD_linker_length": int(second["query_from"]) - int(first["query_to"]) - 1,
        })
    return record


def load_checkpoint(path, architecture, expected_sha, environment, tokens, dimensions):
    path = Path(path)
    require(path.exists(), f"Checkpoint missing: {path}")
    require(sha(path) == expected_sha, f"Checkpoint bytes changed: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(saved["architecture"] == architecture, "Checkpoint architecture differs")
    require(saved["seed"] == TRAINING_SEED, "Checkpoint training seed differs")
    require(saved["technical"] is False, "Technical checkpoint cannot be used for generation")
    require(saved["contract_sha256"] == sha(SOURCE_ROOT / H2_CONTRACT), "Checkpoint contract differs")
    require(saved["runtime"] == environment, "Checkpoint runtime differs")
    require(saved["aa_token_ids"] == tokens, "Checkpoint tokenizer differs")
    model = H2RecoveryModel(aa_token_ids=tokens, architecture=architecture, **dimensions)
    model.load_state_dict(saved["state_dict"], strict=True)
    require(all(torch.isfinite(value).all() for value in saved["state_dict"].values()),
            "Checkpoint contains nonfinite parameters")
    return saved, model


def build_batches(streams):
    batches = []
    for arm in ARMS:
        for budget in EDIT_BUDGETS:
            records = [r for r in streams if r["edit_budget"] == budget]
            require(len(records) == STREAMS_PER_BUDGET, "Stream budget mismatch")
            for start in range(0, len(records), BATCH_SIZE):
                group = records[start:start + BATCH_SIZE]
                require(len(group) == BATCH_SIZE, "Incomplete sampling batch")
                batch_index = len(batches)
                batches.append({
                    "batch_id": f'{arm["label"]}_b{budget}_{start // BATCH_SIZE}',
                    "arm": arm["label"],
                    "architecture": arm["architecture"],
                    "edit_budget": budget,
                    "start_stream": start,
                    "prior_seed": PRIOR_SEED + 1009 * batch_index,
                    "prior_sampled": arm["architecture"] != "no_latent",
                    "streams": group,
                })
    require(len(batches) == 8, "Expected eight four-sequence batches")
    return batches


def proposal_id(batch, stream):
    return f'{batch["arm"]}_b{batch["edit_budget"]}_s{stream:02d}'


def preflight():
    runtime.configure()
    environment = runtime.environment()
    require(torch.cuda.is_available(), "CUDA unavailable; do not change the environment")
    require(SOURCE_ROOT == Path(H2_ROOT).resolve(), "NATV_SOURCE_ROOT differs from the frozen H2 root")
    require(not (RUN / "pre_sampling_freeze.json").exists(), "Preflight already finalized")

    checks = []

    def check(name, detail):
        checks.append({"check": name, "status": "PASS", "detail": detail})

    h2_contract_path = SOURCE_ROOT / H2_CONTRACT
    h2_contract = read_json(h2_contract_path)
    h2_contract_sha = sha(h2_contract_path)
    require(h2_contract["status"] == "FROZEN_AUTHORIZED", "H2 contract not frozen")
    require(h2_contract["fit_authorized"] and h2_contract["generation_authorized"],
            "H2 generation not authorized")
    require(h2_contract["generation"]["total_proposals"] == 1440, "H2 generation contract changed")
    require(h2_contract["population"]["training"] == 1644 and h2_contract["population"]["validation"] == 227,
            "H2 population changed")
    require(h2_contract["population"]["test_quarantine_payload_read"] is False,
            "H2 test payload declaration changed")
    require(h2_contract["dimensions"] == {"hidden_dim": 128, "latent_dim": 32, "latent_states": 8},
            "H2 dimensions changed")
    require(h2_contract["generation"]["edit_budgets"] == EDIT_BUDGETS, "H2 edit budgets changed")
    require(h2_contract["generation"]["batch_size"] == BATCH_SIZE, "H2 batch size changed")
    require(h2_contract["generation"]["sweeps"] == SWEEPS, "H2 sweep count changed")
    require(h2_contract["generation"]["AA_temperature"] == TEMPERATURE, "H2 temperature changed")
    check("H2 frozen generation contract", h2_contract_sha)

    for key, item in h2_contract["inputs"].items():
        current = sha(SOURCE_ROOT / item["path"])
        require(current == item["sha256"], f"H2 input changed: {key}")
    check("H2 input manifest", "input_plan, context_manifest, data_ready")

    formal_plan = read_json(SOURCE_ROOT / FORMAL_PLAN)
    data_ready = read_json(SOURCE_ROOT / DATA_READY)
    require(data_ready["status"] == "FORMAL_TRAINING_DATA_READY", "Formal data not ready")
    require(data_ready["input_plan_sha256"] == sha(SOURCE_ROOT / FORMAL_PLAN), "Formal plan hash differs")
    require(data_ready["test_quarantine_sequence_payload_reads"] == 0, "Test payload was read")
    require(data_ready["optimization_steps"] == 0, "Data stage performed optimization")
    require(data_ready["context_manifest_sha256"] == h2_contract["inputs"]["masked_context_manifest"]["sha256"],
            "Context manifest identity differs")
    require(len(formal_plan["retrieval_database_ids"]) == 1644, "Retrieval database size changed")
    require(formal_plan["test_quarantine_sequence_payload_reads"] == 0, "Test payload declaration changed")
    for name, key in (("training_samples.csv", "training_samples_sha256"),
                      ("train_only.fasta", "train_fasta_sha256"),
                      ("allowed_samples.parquet", "allowed_samples_sha256")):
        require(sha(SOURCE_ROOT / FORMAL_DATA / name) == formal_plan[key], f"Formal data changed: {name}")
    check("Formal F-train inputs", sha(SOURCE_ROOT / FORMAL_PLAN))

    roles_path = SOURCE_ROOT / CORE / "family_training_roles_v2.parquet"
    core_manifest_path = SOURCE_ROOT / CORE / "core_split_manifest_v2.json"
    core_manifest = read_json(core_manifest_path)
    require(sha(roles_path) == core_manifest["core_manifest_sha256"], "Source role manifest changed")
    roles = pd.read_parquet(roles_path)
    require(len(roles) == 2430 and roles.sequence_id.is_unique, "Source population changed")
    train_ids = set(formal_plan["retrieval_database_ids"])
    require(train_ids <= set(roles.loc[roles.core_training_eligible_v2, "sequence_id"]),
            "Retrieval database contains non-core-training records")
    check("Frozen family role metadata", sha(roles_path))

    drift = []
    for name, expected in formal_plan["source_hashes"].items():
        current = sha(SOURCE_ROOT / name)
        match = current == expected
        drift.append({"path": name, "expected": expected, "current": current, "match": match})
        if not match:
            if name == "src/nad_qvae/design/vnext_model.py":
                require(current == h2_contract["historical_basis"][name],
                        "vnext_model.py drift is not the H2 historical basis")
            else:
                raise ValueError(f"Unexplained formal source drift: {name}")
    require(sum(not row["match"] for row in drift) == 1, "Unexpected source drift count")
    check("Formal source drift", "only vnext_model.py; equals H2 historical basis")

    mmseqs = shutil.which("mmseqs")
    require(mmseqs is not None, "MMseqs not on PATH")
    version = subprocess.check_output(["mmseqs", "version"], text=True).strip()
    require(version == formal_plan["mmseqs_version"], "MMseqs version changed")
    check("MMseqs", f"{version} at {mmseqs}")

    encoder_files = data_ready["encoder_files"]
    for name, expected in encoder_files.items():
        current = sha(SOURCE_ROOT / "nad_models/esm2" / name)
        require(current == expected, f"Frozen ESM2 file changed: {name}")
    tokens = list(data_ready["aa_token_ids"])
    require(len(tokens) == 20 and len(set(tokens)) == 20, "Tokenizer table size")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_ROOT / "nad_models/esm2", local_files_only=True)
    require(tokenizer.convert_tokens_to_ids(list(AA)) == tokens, "Tokenizer identity differs")
    check("Frozen ESM2 and tokenizer", encoder_files["model.safetensors"])

    wt, wt_sha, wt_source = load_wt()
    require(wt_sha == h2_contract["generation"]["parent_sha256"], "H2 parent differs from WT")
    constraints = read_json(SOURCE_ROOT / CONSTRAINTS)
    require(constraints["forbidden_substitutions"] == [
        {"position_1based": 135, "WT_AA": "T", "forbidden_AA": "A"}],
        "T135A constraint changed")
    check("WT and T135A constraint", wt_sha)

    checkpoint_records = []
    for arm in ARMS:
        saved, model = load_checkpoint(
            SOURCE_ROOT / arm["checkpoint"], arm["architecture"], arm["sha256"],
            environment, tokens, h2_contract["dimensions"])
        checkpoint_records.append({
            "label": arm["label"],
            "architecture": arm["architecture"],
            "path": arm["checkpoint"].as_posix(),
            "sha256": arm["sha256"],
            "training_seed": TRAINING_SEED,
            "epoch": int(saved["epoch"]),
            "steps": int(saved["steps"]),
            "validation_CE": arm["validation_CE"],
            "parameter_tensors": len(saved["state_dict"]),
            "parameters": int(sum(value.numel() for value in saved["state_dict"].values())),
            "finite_state_dict": True,
        })
        del model, saved
        gc.collect()
    selected = read_json(SOURCE_ROOT / H2_RUN / "selected_checkpoints.json")
    for arm in ARMS:
        record = selected[f'{arm["architecture"]}/{TRAINING_SEED}']
        require(int(record["epoch"]) == 2 and abs(float(record["validation_CE"]) - arm["validation_CE"]) < 1e-12,
                "Selected checkpoint metadata differs")
    check("Frozen H2 checkpoints", [a["sha256"] for a in checkpoint_records])

    basis = read_json(STRUCT_BASIS)
    domains = basis["target"]["domains"]
    anchors = prepare_anchors(basis, wt, domains)
    require(sha(STRUCT_SCRIPT) == basis["script_sha256"], "Structural-basis script identity differs")
    check("Structural interpretation basis", sha(STRUCT_BASIS))

    formal_candidates = pd.read_csv(SOURCE_ROOT / FORMAL_CANDIDATES)
    require(len(formal_candidates) == 25, "Frozen WP5 candidate table size changed")
    control = formal_candidates[formal_candidates.role == "PUBLIC_WT_CONTROL"]
    generated = formal_candidates[formal_candidates.role == "FORMAL_GENERATED_CANDIDATE"]
    require(len(control) == 1 and len(generated) == 24, "Frozen WP5 role counts changed")
    require(control.iloc[0].sequence == wt and control.iloc[0].sequence_hash == wt_sha,
            "Frozen WP5 WT control differs")
    for row in generated.itertuples():
        require(text_sha(row.sequence) == row.sequence_hash, "Frozen WP5 candidate hash differs")
        variants = sequence_variants(row.sequence, wt)
        require(set(variants) == set(parse_variants(row.variants)), "Frozen WP5 variants differ")
        require(len(variants) == int(row.edit_budget), "Frozen WP5 edit budget differs")
        require(row.sequence[T135_INDEX] != "A", "Frozen WP5 contains T135A")
    check("Frozen WP5 retrospective candidates", sha(SOURCE_ROOT / FORMAL_CANDIDATES))

    db_files, data_files, binaries = cdd_assets()
    cdd_digests = {str(path.relative_to(SOURCE_ROOT)): sha(path) for path in db_files + data_files + binaries}
    check("CDD rpsblast/rpsbproc assets", f"{len(cdd_digests)} files")

    streams = stream_plan(341, EDIT_BUDGETS, STREAMS_PER_BUDGET, STREAM_SEED, SWEEPS)
    require(len(streams) == STREAMS_PER_BUDGET * len(EDIT_BUDGETS), "Stream plan size differs")
    for record in streams:
        require(len(record["edit_positions"]) == record["edit_budget"], "Stream budget differs")
        require(len(set(record["edit_positions"])) == record["edit_budget"], "Duplicate edit position")
        require(len(record["update_masks"]) == 1 + 2 * SWEEPS, "Stage count differs")
        require(all(len(mask) == (record["edit_budget"] if i == 0 else record["edit_budget"] // 2)
                    for i, mask in enumerate(record["update_masks"])), "Update-mask size differs")
    batches = build_batches(streams)
    for batch in batches:
        for record in batch["streams"]:
            require(record["edit_budget"] == batch["edit_budget"], "Batch budget mismatch")
    check("Matched stream plan", f"{len(streams)} streams, {len(batches)} batches")

    run_script_sha = sha(Path(__file__))
    sampling_manifest = {
        "schema": "natv-generation-structural-sampling",
        "generated_at_utc": utc_now(),
        "run_script_sha256": run_script_sha,
        "source_root": str(SOURCE_ROOT),
        "target": {"sequence_sha256": wt_sha, "length": len(wt)},
        "checkpoints": checkpoint_records,
        "kernel": {
            "streams_seed": STREAM_SEED,
            "prior_seed_base": PRIOR_SEED,
            "prior_lifecycle": "one prior draw per four-sequence trajectory; fixed across seven stages",
            "edit_budgets": EDIT_BUDGETS,
            "streams_per_budget": STREAMS_PER_BUDGET,
            "sweeps": SWEEPS,
            "temperature": TEMPERATURE,
            "batch_size": BATCH_SIZE,
            "stages": 1 + 2 * SWEEPS,
            "forbidden": "T135A at every update; WT residues at every editable site",
            "retrieval": "fresh masked query and frozen F-train MMseqs search at every stage",
            "plm": "fresh masked frozen ESM2 forward at every stage",
            "stream_type": "matched",
        },
        "streams": streams,
        "batches": batches,
        "proposal_ids": [proposal_id(b, record["stream"]) for b in batches for record in b["streams"]],
        "proposal_count": len(batches) * BATCH_SIZE,
    }
    require(sampling_manifest["proposal_count"] == 32, "Proposal budget differs")
    write_json(RUN / "sampling_manifest.json", sampling_manifest)

    inputs = []
    seen = set()

    def add(path, role, used_by, expected=None, root_hint=None):
        entry = input_entry(path, role, used_by, expected, root_hint)
        key = (entry["root"], entry["path"])
        if key not in seen:
            seen.add(key)
            inputs.append(entry)

    add(H2_CONTRACT, "frozen generation contract", "sampling", h2_contract_sha)
    add(RUNTIME_CONTRACT, "deterministic runtime contract", "sampling/evaluation")
    add(TRAIN_CONTRACT, "training/split contract", "provenance")
    add(FORMAL_PLAN, "formal input plan", "sampling", data_ready["input_plan_sha256"])
    add(DATA_READY, "formal data readiness receipt", "sampling/evaluation")
    add(FORMAL_DATA / "context_manifest.json", "masked context manifest", "provenance",
        h2_contract["inputs"]["masked_context_manifest"]["sha256"])
    add(FORMAL_DATA / "training_samples.csv", "F-train sequences", "sampling", formal_plan["training_samples_sha256"])
    add(FORMAL_DATA / "train_only.fasta", "F-train retrieval database", "sampling", formal_plan["train_fasta_sha256"])
    add(FORMAL_DATA / "allowed_samples.parquet", "allowed sample roles", "sampling", formal_plan["allowed_samples_sha256"])
    add(CORE / "family_training_roles_v2.parquet", "family role metadata; novelty and train scope", "evaluation")
    add(CORE / "core_split_manifest_v2.json", "core split manifest", "provenance")
    add(WT_SOURCE, "public WT sequence and identity", "sampling")
    add(CONSTRAINTS, "T135A target constraint", "sampling")
    add(H2_RUN / "selected_checkpoints.json", "validation-selected H2 checkpoints", "sampling")
    add(H2_RUN / "generation/checkpoint_freeze.json", "H2 generation checkpoint freeze", "sampling")
    for arm in ARMS:
        add(arm["checkpoint"], f'{arm["label"]} frozen checkpoint', "sampling", arm["sha256"])
    for name, digest in encoder_files.items():
        add(Path("nad_models/esm2") / name, "frozen ESM2 encoder", "sampling/evaluation", digest)
    add(STRUCT_BASIS, "structure-informed annotation basis", "evaluation", root_hint="natvqvae")
    add(STRUCT_SCRIPT, "structural-basis construction script", "provenance", root_hint="natvqvae")
    add(FORMAL_CANDIDATES, "retrospective frozen WP5 candidates", "evaluation")
    add(FORMAL_CDD, "frozen WP5 CDD domains", "evaluation/reproduction")
    add(FORMAL_CE, "frozen WP5 internal PLM scores", "evaluation/reproduction")
    for name in (
        "src/nad_qvae/design/h2_model_v1.py",
        "src/nad_qvae/design/h2_execution_v1.py",
        "src/nad_qvae/design/h2_generation_v1.py",
        "src/nad_qvae/design/g0_benchmark.py",
        "src/nad_qvae/design/formal_generation.py",
        "src/nad_qvae/design/formal_training_data.py",
        "src/nad_qvae/design/se_full.py",
        "src/nad_qvae/design/vnext_evaluation_contracts.py",
        "src/nad_qvae/design/vnext_family_v2.py",
        "src/nad_qvae/design/vnext_model.py",
        "src/nad_qvae/design/vnext_residual_model.py",
        "src/nad_qvae/design/vnext_residual_pilot.py",
        "src/nad_qvae/design/vnext_residual_pilot_inputs.py",
        "src/nad_qvae/design/context_restoration.py",
        "src/nad_qvae/design/sequence_corpus.py",
        "src/nad_qvae/design/multimodal_deterministic_runtime_v2.py",
    ):
        add(Path(name), "executed source implementation", "sampling/evaluation")
    for path in db_files + data_files + binaries:
        add(path, "CDD rpsblast/rpsbproc asset", "evaluation")
    inputs.append({
        "root": "environment",
        "path": mmseqs,
        "bytes": Path(mmseqs).stat().st_size,
        "sha256": sha(mmseqs),
        "role": "MMseqs retrieval executable",
        "used_by": "sampling",
        "version": version,
    })
    input_manifest = {
        "schema": "natv-generation-structural-inputs",
        "generated_at_utc": utc_now(),
        "scope": "run-specific evaluation inputs; global training population unchanged",
        "entries": inputs,
    }
    write_json(RUN / "input_manifest.json", input_manifest)

    preflight_report = {
        "schema": "natv-generation-structural-preflight",
        "status": "PREFLIGHT_PASS",
        "generated_at_utc": utc_now(),
        "run_script_sha256": run_script_sha,
        "runtime_environment": environment,
        "device": torch.cuda.get_device_name(0),
        "checks": checks,
        "source_drift": drift,
        "h2_contract_sha256": h2_contract_sha,
        "data_ready_sha256": sha(SOURCE_ROOT / DATA_READY),
        "formal_input_plan_sha256": sha(SOURCE_ROOT / FORMAL_PLAN),
        "family_roles_sha256": sha(roles_path),
        "structural_basis_sha256": sha(STRUCT_BASIS),
        "formal_candidates_sha256": sha(SOURCE_ROOT / FORMAL_CANDIDATES),
        "checkpoints": checkpoint_records,
        "cdd_assets": cdd_digests,
        "population": {
            "training": int(roles.core_training_eligible_v2.sum()),
            "family_rows": len(roles),
            "retrieval_database_ids": len(train_ids),
            "test_payload_reads": 0,
        },
        "declared_scope": {
            "training_runs": 0,
            "folding_runs": 0,
            "docking_runs": 0,
            "md_runs": 0,
            "wet_lab": False,
            "remote_downloads": 0,
            "test_payload_reads": 0,
            "proposal_count": 32,
        },
    }
    write_json(RUN / "preflight_report.json", preflight_report)

    freeze = {
        "schema": "natv-generation-structural-pre-sampling-freeze",
        "status": "PRE_SAMPLING_FREEZE",
        "generated_at_utc": utc_now(),
        "run_script_sha256": run_script_sha,
        "source_root": str(SOURCE_ROOT),
        "sampling_manifest_sha256": sha(RUN / "sampling_manifest.json"),
        "input_manifest_sha256": sha(RUN / "input_manifest.json"),
        "preflight_report_sha256": sha(RUN / "preflight_report.json"),
        "h2_contract_sha256": h2_contract_sha,
        "structural_basis_sha256": sha(STRUCT_BASIS),
        "formal_candidates_sha256": sha(SOURCE_ROOT / FORMAL_CANDIDATES),
        "constraints_sha256": sha(SOURCE_ROOT / CONSTRAINTS),
        "runtime_environment": environment,
        "checkpoints": checkpoint_records,
        "sampling": {
            "streams_seed": STREAM_SEED,
            "prior_seed_base": PRIOR_SEED,
            "edit_budgets": EDIT_BUDGETS,
            "streams_per_budget": STREAMS_PER_BUDGET,
            "sweeps": SWEEPS,
            "batch_size": BATCH_SIZE,
            "proposal_count": 32,
        },
        "state_at_freeze": {
            "proposals_written": 0,
            "evaluation_written": False,
            "training_runs": 0,
            "folding_runs": 0,
            "wet_lab": False,
            "remote_downloads": 0,
            "test_payload_reads": 0,
        },
        "source_drift": drift,
    }
    write_json(RUN / "pre_sampling_freeze.json", freeze)
    print(json.dumps({"status": "PREFLIGHT_PASS", "checkpoints": checkpoint_records,
                      "proposals": sampling_manifest["proposal_count"],
                      "freeze_sha256": sha(RUN / "pre_sampling_freeze.json")}, indent=2))


def ensure_evaluation_amendment():
    freeze = read_json(RUN / "pre_sampling_freeze.json")
    current = sha(Path(__file__))
    if current == freeze["run_script_sha256"]:
        return None
    path = RUN / "post_sampling_amendment.json"
    artifacts = {
        name: sha(RUN / name) for name in (
            "proposals.csv", "sampling_trace.jsonl.gz",
            "retrieval_receipts.json.gz", "retrieval_hits_summary.csv")
    }
    reason = ("Evaluation-stage code changed after sampling was frozen; no "
              "sampling code or output changed. Revisions corrected evaluation-stage "
              "defects and expanded unique-sequence and duplicate reporting.")
    if path.exists():
        amendment = read_json(path)
        if amendment["evaluation_script_sha256"] == current:
            return amendment
        require(not (RUN / "evaluation/evaluation_receipt.json").exists(),
                "Evaluation already finalized; a script change requires a new run")
        for name, digest in amendment["sampling_artifacts"].items():
            require(sha(RUN / name) == digest, f"Sampling output changed: {name}")
        revisions = amendment.get("revisions", [{
            "evaluation_script_sha256": amendment["evaluation_script_sha256"],
            "recorded_at_utc": amendment["generated_at_utc"],
            "reason": amendment["reason"],
        }])
        revisions.append({"evaluation_script_sha256": current,
                          "recorded_at_utc": utc_now(), "reason": reason})
        amendment.update({
            "generated_at_utc": utc_now(),
            "evaluation_script_sha256": current,
            "reason": reason,
            "sampling_artifacts": artifacts,
            "revisions": revisions,
        })
        write_json(path, amendment)
        return amendment
    amendment = {
        "schema": "natv-generation-structural-post-sampling-amendment",
        "status": "POST_SAMPLING_EVALUATION_AMENDMENT",
        "generated_at_utc": utc_now(),
        "sampling_script_sha256": freeze["run_script_sha256"],
        "evaluation_script_sha256": current,
        "changed_stage": "evaluate",
        "reason": reason,
        "sampling_manifest_sha256": freeze["sampling_manifest_sha256"],
        "input_manifest_sha256": freeze["input_manifest_sha256"],
        "preflight_report_sha256": freeze["preflight_report_sha256"],
        "sampling_artifacts": artifacts,
        "sampling_outputs_unchanged": True,
        "revisions": [{"evaluation_script_sha256": current,
                       "recorded_at_utc": utc_now(), "reason": reason}],
    }
    write_json(path, amendment)
    return amendment


def verify_freeze(allow_evaluation_amendment=False):
    freeze = read_json(RUN / "pre_sampling_freeze.json")
    require(freeze["status"] == "PRE_SAMPLING_FREEZE", "Pre-sampling freeze missing")
    if freeze["run_script_sha256"] != sha(Path(__file__)):
        require(allow_evaluation_amendment, "Run script changed after freeze")
        amendment = read_json(RUN / "post_sampling_amendment.json")
        require(amendment["status"] == "POST_SAMPLING_EVALUATION_AMENDMENT",
                "Post-sampling amendment missing")
        require(amendment["sampling_script_sha256"] == freeze["run_script_sha256"],
                "Amendment sampling script differs")
        require(amendment["evaluation_script_sha256"] == sha(Path(__file__)),
                "Amendment evaluation script differs")
        require(amendment["changed_stage"] == "evaluate", "Amendment changed another stage")
        require(amendment["sampling_manifest_sha256"] == freeze["sampling_manifest_sha256"],
                "Amendment sampling manifest differs")
        for name, digest in amendment["sampling_artifacts"].items():
            require(sha(RUN / name) == digest, f"Sampling output changed: {name}")
    for name, key in (("sampling_manifest.json", "sampling_manifest_sha256"),
                      ("input_manifest.json", "input_manifest_sha256"),
                      ("preflight_report.json", "preflight_report_sha256")):
        require(sha(RUN / name) == freeze[key], f"Frozen artefact changed: {name}")
    require(freeze["h2_contract_sha256"] == sha(SOURCE_ROOT / H2_CONTRACT), "H2 contract changed")
    require(freeze["structural_basis_sha256"] == sha(STRUCT_BASIS), "Structural basis changed")
    require(freeze["formal_candidates_sha256"] == sha(SOURCE_ROOT / FORMAL_CANDIDATES),
            "Frozen WP5 table changed")
    require(freeze["constraints_sha256"] == sha(SOURCE_ROOT / CONSTRAINTS), "Constraint changed")
    for arm in ARMS:
        require(sha(SOURCE_ROOT / arm["checkpoint"]) == arm["sha256"], "Checkpoint changed")
    input_manifest = read_json(RUN / "input_manifest.json")
    for item in input_manifest["entries"]:
        if item["root"] == "nad_qvae":
            path = SOURCE_ROOT / item["path"]
        elif item["root"] == "natvqvae":
            path = NATV_ROOT / item["path"]
        elif item["root"] == "environment":
            path = Path(item["path"])
        else:
            path = Path(item["path"])
        require(sha(path) == item["sha256"], f"Input changed after freeze: {item['path']}")
    runtime.check()
    return freeze


def trajectory(model, frozen, retrieval, wt, batch, folder):
    generator = torch.Generator(device="cuda").manual_seed(batch["prior_seed"])
    z = model.sample_prior(BATCH_SIZE, generator=generator, device="cuda")
    if batch["prior_sampled"]:
        require(z is not None and tuple(z.shape) == (BATCH_SIZE, model.latent_dim),
                "Continuous prior shape differs")
        require(bool(torch.isfinite(z).all()), "Continuous prior is nonfinite")
        latent_hashes = [hashlib.sha256(z[i].detach().cpu().numpy().tobytes()).hexdigest()
                         for i in range(BATCH_SIZE)]
    else:
        require(z is None, "no_latent returned a latent")
        latent_hashes = [""] * BATCH_SIZE
    current = [wt] * BATCH_SIZE
    trace = []
    hits_record = []
    model.eval()
    for stage in range(1 + 2 * SWEEPS):
        positions = [record["update_masks"][stage] for record in batch["streams"]]
        queries = [
            {
                "query_id": f's{record["stream"]}_i{index}',
                "current_sequence": current[index],
                "positions": positions[index],
                "masked_sequence": masked_sequence(current[index], positions[index]),
            }
            for index, record in enumerate(batch["streams"])
        ]
        profiles, hits = retrieval.search(queries, folder / f"stage{stage}")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = frozen(current, positions)
        require(encoded["logits"].shape[0] == BATCH_SIZE, "Encoder batch differs")
        evolution = torch.as_tensor(np.stack([profiles[q["query_id"]] for q in queries]), device="cuda")
        kwargs = {"latent_override": z} if batch["prior_sampled"] else {}
        with torch.no_grad():
            result = model(encoded, encoded["target_mask"], evolution=evolution, **kwargs)
        if batch["prior_sampled"]:
            require(result["latent_source"] == "external_override",
                    "Posterior entered prior generation")
        for index, record in enumerate(batch["streams"]):
            position = positions[index]
            before = current[index]
            forbidden = {row: ["A"] for row, value in enumerate(position) if value == T135_INDEX}
            letters, _ = constrained_draw(
                result["logits"][index, position].cpu().numpy(),
                [wt[value] for value in position],
                record["uniform_draws"][stage],
                TEMPERATURE,
                forbidden,
            )
            require(len(letters) == len(position), "Draw length differs")
            updated = list(before)
            for value, letter in zip(position, letters):
                require(wt[value] != letter, "Sampler returned the WT residue")
                updated[value] = letter
            after = "".join(updated)
            require(all(before[i] == after[i] for i in range(len(wt)) if i not in position),
                    "A non-editable position changed")
            require(after[T135_INDEX] != "A", "T135A quarantine violation")
            current[index] = after
            trace.append({
                "batch_id": batch["batch_id"],
                "arm": batch["arm"],
                "edit_budget": batch["edit_budget"],
                "stream": record["stream"],
                "stage": stage,
                "update_positions_0based": position,
                "before": before,
                "after": after,
                "before_sha256": text_sha(before),
                "after_sha256": text_sha(after),
                "masked_query_sha256": text_sha(queries[index]["masked_sequence"]),
                "latent_source": "external_override" if batch["prior_sampled"] else "not_applicable",
                "latent_sha256": latent_hashes[index],
                "hit_count": hits[index]["hit_count"],
                "neff": hits[index]["neff"],
            })
            hits_record.append({
                "batch_id": batch["batch_id"],
                "stage": stage,
                "stream": record["stream"],
                "query_id": queries[index]["query_id"],
                "hit_count": hits[index]["hit_count"],
                "neff": hits[index]["neff"],
                "masked_sequence_sha256": hits[index]["masked_sequence_sha256"],
                "hit_ids_sha256": text_sha(";".join(sorted(hits[index]["hit_ids"]))),
            })
        del encoded, evolution, result
    sequences = []
    for record, sequence in zip(batch["streams"], current):
        verify_g0_sequence(wt, sequence, record["edit_positions"])
        variants = verify_candidate(sequence, wt, record["edit_positions"])
        require(sequence[T135_INDEX] != "A", "T135A quarantine violation")
        sequences.append({
            "proposal_id": proposal_id(batch, record["stream"]),
            "source_set": "bounded_new",
            "arm": batch["arm"],
            "architecture": batch["architecture"],
            "training_seed": TRAINING_SEED,
            "checkpoint_sha256": next(a["sha256"] for a in ARMS if a["label"] == batch["arm"]),
            "batch_id": batch["batch_id"],
            "prior_seed": batch["prior_seed"],
            "prior_sampled": batch["prior_sampled"],
            "stream_type": "matched",
            "edit_budget": batch["edit_budget"],
            "stream": record["stream"],
            "edit_positions_1based": ";".join(str(p + 1) for p in record["edit_positions"]),
            "variants": variant_text(variants),
            "n_edits": len(variants),
            "sequence": sequence,
            "sequence_hash": text_sha(sequence),
            "length": len(sequence),
            "canonical_AA": True,
            "syntax_valid": True,
            "T135A_violation": False,
            "latent_source": "external_override" if batch["prior_sampled"] else "not_applicable",
            "latent_sha256": latent_hashes[record["stream"] - batch["start_stream"]],
        })
    runtime.check()
    return sequences, trace, hits_record


def sample():
    runtime.configure()
    freeze = verify_freeze()
    require(not (RUN / "proposals.csv").exists(), "Sampling already produced proposals")
    require(not (RUN / "sampling_trace.jsonl.gz").exists(), "Sampling trace already exists")
    data = SEReader()
    tokens = list(data.tokens)
    require(tokens == list(read_json(SOURCE_ROOT / DATA_READY)["aa_token_ids"]), "Tokenizer changed")
    manifest = read_json(RUN / "sampling_manifest.json")
    require(sha(RUN / "sampling_manifest.json") == freeze["sampling_manifest_sha256"],
            "Sampling manifest changed")
    retrieval = H2Retrieval(data)
    frozen = FrozenHFMaskedEncoder(SOURCE_ROOT / "nad_models/esm2", device="cuda")
    require(frozen.tokenizer.convert_tokens_to_ids(list(AA)) == tokens, "Tokenizer identity differs")
    wt, wt_sha, _ = load_wt()
    require(wt_sha == manifest["target"]["sequence_sha256"], "WT changed")
    proposals = []
    traces = []
    hit_records = []
    receipts = []
    started = time.monotonic()
    for arm in ARMS:
        saved, model = load_checkpoint(
            SOURCE_ROOT / arm["checkpoint"], arm["architecture"], arm["sha256"],
            runtime.environment(), tokens, read_json(SOURCE_ROOT / H2_CONTRACT)["dimensions"])
        model = model.to("cuda").eval()
        for batch in [b for b in manifest["batches"] if b["arm"] == arm["label"]]:
            folder = RUN / "retrieval" / batch["batch_id"]
            sequences, batch_trace, batch_hits = trajectory(
                model, frozen, retrieval, wt, batch, folder)
            proposals.extend(sequences)
            traces.extend(batch_trace)
            hit_records.extend(batch_hits)
            for stage in range(1 + 2 * SWEEPS):
                receipt_path = folder / f"stage{stage}" / "retrieval_receipt.json"
                require(receipt_path.exists(), "Retrieval receipt missing")
                receipts.append({
                    "batch_id": batch["batch_id"],
                    "stage": stage,
                    "path": receipt_path.relative_to(RUN).as_posix(),
                    "sha256": sha(receipt_path),
                    "receipt": read_json(receipt_path),
                })
            write_json(RUN / "sample_progress.json", {
                "status": "SAMPLING_IN_PROGRESS",
                "proposals": len(proposals),
                "batches_completed": len({p["batch_id"] for p in proposals}),
                "elapsed_seconds": time.monotonic() - started,
            })
            print(json.dumps({"batch": batch["batch_id"], "sequences": len(sequences),
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    frame = pd.DataFrame(proposals)
    require(len(frame) == 32 and frame.proposal_id.is_unique, "Proposal identity/budget mismatch")
    require(set(frame.arm) == {arm["label"] for arm in ARMS}, "Arm coverage differs")
    require(frame.groupby(["arm", "edit_budget"]).size().eq(8).all(), "Cell budget differs")
    require(frame.sequence_hash.map(lambda value: len(value) == 64).all(), "Sequence hash malformed")
    require(frame.apply(lambda row: text_sha(row.sequence) == row.sequence_hash, axis=1).all(),
            "Sequence hash differs")
    frame.to_csv(RUN / "proposals.csv", index=False)
    with gzip.open(RUN / "sampling_trace.jsonl.gz", "wt") as handle:
        for item in traces:
            handle.write(json.dumps(item, allow_nan=False) + "\n")
    with gzip.open(RUN / "retrieval_receipts.json.gz", "wt") as handle:
        json.dump({"schema": "natv-generation-structural-retrieval-receipts",
                   "receipts": receipts}, handle, allow_nan=False)
        handle.write("\n")
    pd.DataFrame(hit_records).to_csv(RUN / "retrieval_hits_summary.csv", index=False)
    status = {
        "status": "SAMPLING_COMPLETE",
        "proposals": len(frame),
        "unique_sequence_hashes": int(frame.sequence_hash.nunique()),
        "batches": len(manifest["batches"]),
        "retrieval_stages": len(receipts),
        "edit_forward_batches": len(manifest["batches"]) * (1 + 2 * SWEEPS),
        "fresh_sequence_encodings": len(manifest["batches"]) * BATCH_SIZE * (1 + 2 * SWEEPS),
        "training_runs": 0,
        "folding_runs": 0,
        "docking_runs": 0,
        "md_runs": 0,
        "wet_lab": False,
        "remote_downloads": 0,
        "test_payload_reads": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(RUN / "job_status.json", status)
    print(json.dumps(status, indent=2))


def formal_reference():
    table = pd.read_csv(SOURCE_ROOT / FORMAL_CANDIDATES)
    control = table[table.role == "PUBLIC_WT_CONTROL"].iloc[0]
    generated = table[table.role == "FORMAL_GENERATED_CANDIDATE"].copy()
    return control, generated


def evaluate():
    runtime.configure()
    amendment = ensure_evaluation_amendment()
    freeze = verify_freeze(allow_evaluation_amendment=True)
    require((RUN / "proposals.csv").exists(), "Proposals missing")
    require(not (RUN / "evaluation").exists(), "Evaluation already exists")
    data = SEReader()
    tokens = list(data.tokens)
    require(tokens == list(read_json(SOURCE_ROOT / DATA_READY)["aa_token_ids"]), "Tokenizer changed")
    wt, wt_sha, _ = load_wt()
    basis = read_json(STRUCT_BASIS)
    domains = basis["target"]["domains"]
    anchors = prepare_anchors(basis, wt, domains)
    proposals = pd.read_csv(RUN / "proposals.csv", keep_default_na=False)
    require(len(proposals) == 32 and proposals.proposal_id.is_unique, "Proposal table changed")
    for row in proposals.itertuples():
        sequence = row.sequence
        require(text_sha(sequence) == row.sequence_hash, "Proposal hash differs")
        editable = [int(value) - 1 for value in row.edit_positions_1based.split(";")]
        variants = verify_candidate(sequence, wt, editable)
        require(variant_text(variants) == row.variants, "Proposal variants differ")
        require(len(variants) == int(row.edit_budget), "Proposal edit budget differs")
    control, generated = formal_reference()
    require(text_sha(control.sequence) == control.sequence_hash == wt_sha, "Formal WT control differs")
    for row in generated.itertuples():
        require(text_sha(row.sequence) == row.sequence_hash, "Formal candidate hash differs")
        variants = verify_candidate(row.sequence, wt)
        require(set(variants) == set(parse_variants(row.variants)), "Formal variants differ")
        require(len(variants) == int(row.edit_budget), "Formal edit budget differs")
    formal_by_hash = {row.sequence_hash: row for row in generated.itertuples()}
    train_ids = set(read_json(SOURCE_ROOT / FORMAL_PLAN)["retrieval_database_ids"])

    frames = [pd.DataFrame([{
        "proposal_id": "PUBLIC_WT_CONTROL",
        "source_set": "control",
        "arm": "WT",
        "stream_type": "control",
        "edit_budget": 0,
        "sequence": wt,
        "sequence_hash": wt_sha,
        "variants": "WT",
        "n_edits": 0,
        "batch_id": "",
        "stream": "",
        "prior_sampled": False,
        "latent_source": "not_applicable",
        "latent_sha256": "",
    }])]
    new_frame = proposals.copy()
    new_frame["source_set"] = "bounded_new"
    frames.append(new_frame)
    formal_frame = pd.DataFrame([{
        "proposal_id": f"formal:{row.proposal_id}",
        "source_set": "formal_WP5_frozen",
        "arm": f"formal_{row.quota_arm}",
        "stream_type": row.quota_stream,
        "edit_budget": int(row.edit_budget),
        "sequence": row.sequence,
        "sequence_hash": row.sequence_hash,
        "variants": row.variants,
        "n_edits": int(row.edit_budget),
        "batch_id": "",
        "stream": "",
        "prior_sampled": False,
        "latent_source": "not_applicable",
        "latent_sha256": "",
        "formal_proposal_id": row.proposal_id,
        "frozen_PLM_CE": float(row.PLM_blocked_CE_diagnostic_only),
    } for row in generated.itertuples()])
    frames.append(formal_frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    require(combined.sequence_hash.notna().sum() == len(combined), "Missing sequence hash")
    evaluation = RUN / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=False)
    frozen = FrozenHFMaskedEncoder(SOURCE_ROOT / "nad_models/esm2", device="cuda")
    require(frozen.tokenizer.convert_tokens_to_ids(list(AA)) == tokens, "Tokenizer identity differs")
    result, wt_quality = internal_metrics(
        SOURCE_ROOT, frozen, tokens, combined, wt,
        {"WT_sha256": wt_sha, "quality_blocks": QUALITY_BLOCKS}, evaluation)
    require(result.sequence_hash.map(lambda value: len(value) == 64).all(), "Metric hash malformed")
    require(abs(wt_quality - WT_CE_REFERENCE) < 0.01, "WT CE diverged beyond tolerance")
    domains_table = pd.read_csv(evaluation / "CDD_domains.csv")
    pairs = {name: pair_details(group.to_dict("records"))
             for name, group in domains_table.groupby("sequence_id")}
    require(pairs.get(wt_sha, {}).get("ordered_CDD_pair") is True, "CDD gate excludes WT")
    formal_frozen_domains = pd.read_csv(SOURCE_ROOT / FORMAL_CDD)
    formal_frozen_ce = pd.read_csv(SOURCE_ROOT / FORMAL_CE).set_index("sequence_hash")
    formal_domain_columns = ["specificity_class", "pssm_id", "query_from", "query_to",
                             "evalue", "bitscore", "accession", "short_name", "incomplete"]
    cdd_reproduction = []
    for sequence_hash in set(formal_frozen_domains.sequence_id):
        frozen_rows = formal_frozen_domains[formal_frozen_domains.sequence_id == sequence_hash]
        current_rows = domains_table[domains_table.sequence_id == sequence_hash]
        sort_keys = ["pssm_id", "query_from", "query_to", "accession"]
        frozen_sorted = frozen_rows[formal_domain_columns].sort_values(sort_keys).reset_index(drop=True)
        current_sorted = current_rows[formal_domain_columns].sort_values(sort_keys).reset_index(drop=True)
        identical = (
            len(frozen_sorted) == len(current_sorted)
            and frozen_sorted[sort_keys].astype(str).equals(current_sorted[sort_keys].astype(str))
            and np.allclose(frozen_sorted.evalue, current_sorted.evalue, rtol=0, atol=1e-12)
            and np.allclose(frozen_sorted.bitscore, current_sorted.bitscore, rtol=0, atol=1e-6)
        )
        cdd_reproduction.append({
            "sequence_hash": sequence_hash,
            "frozen_rows": len(frozen_rows),
            "current_rows": len(current_rows),
            "identical": bool(identical),
            "max_bitscore_delta": float((frozen_sorted.bitscore - current_sorted.bitscore).abs().max())
            if identical else np.nan,
        })
    cdd_reproduction = pd.DataFrame(cdd_reproduction)
    cdd_reproduction.to_csv(evaluation / "CDD_reproduction_check.csv", index=False)

    candidate_rows = []
    substitution_level = []
    for row in result.itertuples():
        sequence = row.sequence
        sequence_hash = row.sequence_hash
        variants = sequence_variants(sequence, wt)
        if row.source_set != "control":
            require(len(variants) == int(row.edit_budget), "Candidate edit count differs")
        pair = pairs.get(sequence_hash, pair_details([]))
        require(bool(pair["ordered_CDD_pair"]) == bool(row.ordered_CDD_pair),
                "CDD pair merge differs")
        candidate = {
            "candidate_id": row.proposal_id,
            "source_set": row.source_set,
            "arm": row.arm,
            "stream_type": row.stream_type,
            "edit_budget": int(row.edit_budget),
            "proposal_id": row.proposal_id,
            "sequence_hash": sequence_hash,
            "sequence": sequence,
            "length": len(sequence),
            "n_edits": len(variants),
            "variants": variant_text(variants),
            "PLM_blocked_CE": float(row.PLM_blocked_CE),
            "delta_CE_vs_WT": float(row.PLM_blocked_CE - wt_quality),
            "PLM_CE_below_WT": bool(float(row.PLM_blocked_CE) <= wt_quality),
            "exact_novel_family2430": bool(row.exact_novel),
            "exact_novel_train1644": sequence_hash not in train_ids,
            "CDD_ordered_pair": bool(pair["ordered_CDD_pair"]),
            "CDD_NMNAT_pssm": pair["CDD_NMNAT_pssm"],
            "CDD_NMNAT_accession": pair["CDD_NMNAT_accession"],
            "CDD_NMNAT_range": pair["CDD_NMNAT_range"],
            "CDD_NMNAT_evalue": pair["CDD_NMNAT_evalue"],
            "CDD_NMNAT_bitscore": pair["CDD_NMNAT_bitscore"],
            "CDD_NUDIX_pssm": pair["CDD_NUDIX_pssm"],
            "CDD_NUDIX_accession": pair["CDD_NUDIX_accession"],
            "CDD_NUDIX_range": pair["CDD_NUDIX_range"],
            "CDD_NUDIX_evalue": pair["CDD_NUDIX_evalue"],
            "CDD_NUDIX_bitscore": pair["CDD_NUDIX_bitscore"],
            "CDD_linker_length": pair["CDD_linker_length"],
            "n_NMNAT_family_hits": pair["n_NMNAT_family_hits"],
            "n_cd18873_hits": pair["n_cd18873_hits"],
            "T135_changed": any(position == 135 for position, _, _ in variants),
            "T135A_violation": any(position == 135 and new == "A" for position, _, new in variants),
            "H236_changed": any(position == H236_POSITION for position, _, _ in variants),
            "frozen_PLM_CE": np.nan,
            "recomputed_minus_frozen_CE": np.nan,
            "frozen_proposal_id": "",
        }
        if row.source_set == "formal_WP5_frozen":
            frozen_row = formal_by_hash[sequence_hash]
            candidate["frozen_PLM_CE"] = float(frozen_row.PLM_blocked_CE_diagnostic_only)
            candidate["recomputed_minus_frozen_CE"] = float(
                row.PLM_blocked_CE - frozen_row.PLM_blocked_CE_diagnostic_only)
            candidate["frozen_proposal_id"] = frozen_row.proposal_id
            require(float(frozen_row.PLM_blocked_CE_diagnostic_only) ==
                    float(formal_frozen_ce.loc[sequence_hash, "PLM_blocked_CE"]),
                    "Frozen CE tables disagree")
        candidate["domain_substitutions_NTERM"] = sum(
            1 for position, _, _ in variants if domain_of(position, domains) == "NTERM")
        candidate["domain_substitutions_NMNAT"] = sum(
            1 for position, _, _ in variants if domain_of(position, domains) == "NMNAT")
        candidate["domain_substitutions_LINKER"] = sum(
            1 for position, _, _ in variants if domain_of(position, domains) == "LINKER")
        candidate["domain_substitutions_NUDIX"] = sum(
            1 for position, _, _ in variants if domain_of(position, domains) == "NUDIX")
        candidate["domain_substitutions_CTERM"] = sum(
            1 for position, _, _ in variants if domain_of(position, domains) == "CTERM")
        if row.source_set != "control":
            rows = substitution_rows(candidate, anchors, domains)
            substitution_level.extend(rows)
            direct = [item for item in rows if item["direct_anchor"]]
            near = [item for item in rows if not item["direct_anchor"]
                    and item["within_4_sequence_positions"]]
            candidate["n_direct_anchor_substitutions"] = len(direct)
            candidate["direct_anchor_mutations"] = ";".join(item["variant"] for item in direct)
            candidate["direct_anchor_domains"] = ";".join(item["domain"] for item in direct)
            candidate["n_near_window_substitutions"] = len(near)
            candidate["near_window_mutations"] = ";".join(item["variant"] for item in near)
            candidate["n_NMNAT_anchor_substitutions"] = sum(item["domain"] == "NMNAT" for item in direct)
            candidate["n_NUDIX_anchor_substitutions"] = sum(item["domain"] == "NUDIX" for item in direct)
            candidate["NMNAT_anchor_perturbed"] = bool(candidate["n_NMNAT_anchor_substitutions"])
            candidate["NUDIX_anchor_perturbed"] = bool(candidate["n_NUDIX_anchor_substitutions"])
            candidate["H236_variable_anchor_changed"] = bool(candidate["H236_changed"])
            candidate["anchor_preserving"] = not direct
            if direct:
                candidate["design_intent_bookkeeping"] = "anchor_perturbed"
            elif candidate["domain_substitutions_NMNAT"] or candidate["domain_substitutions_NUDIX"]:
                candidate["design_intent_bookkeeping"] = "domain_variable_anchor_preserving"
            else:
                candidate["design_intent_bookkeeping"] = "non_domain_only"
        else:
            candidate.update({
                "n_direct_anchor_substitutions": 0,
                "direct_anchor_mutations": "",
                "direct_anchor_domains": "",
                "n_near_window_substitutions": 0,
                "near_window_mutations": "",
                "n_NMNAT_anchor_substitutions": 0,
                "n_NUDIX_anchor_substitutions": 0,
                "NMNAT_anchor_perturbed": False,
                "NUDIX_anchor_perturbed": False,
                "H236_variable_anchor_changed": False,
                "anchor_preserving": True,
                "design_intent_bookkeeping": "control",
            })
        candidate_rows.append(candidate)
    candidates = pd.DataFrame(candidate_rows)
    substitutions = pd.DataFrame(substitution_level)
    (RUN / "structural_analysis").mkdir(parents=True, exist_ok=False)
    candidates.to_csv(RUN / "structural_analysis/candidate_level.csv", index=False)
    substitutions.to_csv(RUN / "structural_analysis/substitution_level.csv", index=False)
    pairs_frame = pd.DataFrame([
        {"sequence_hash": key, **value} for key, value in pairs.items()
    ]).sort_values("sequence_hash")
    pairs_frame.to_csv(evaluation / "CDD_pair_summary.csv", index=False)

    arm_rows = []
    for keys, group in candidates.groupby(
            ["source_set", "arm", "stream_type", "edit_budget"], dropna=False, sort=True):
        source_set, arm, stream_type, edit_budget = keys
        arm_rows.append({
            "source_set": source_set,
            "arm": arm,
            "stream_type": stream_type,
            "edit_budget": int(edit_budget),
            "n_proposals": len(group),
            "n_unique_sequences": int(group.sequence_hash.nunique()),
            "n_exact_novel_family2430": int(group.exact_novel_family2430.sum()),
            "n_exact_novel_train1644": int(group.exact_novel_train1644.sum()),
            "n_CDD_pass": int(group.CDD_ordered_pair.sum()),
            "n_CE_below_WT": int(group.PLM_CE_below_WT.sum()),
            "median_PLM_CE": float(group.PLM_blocked_CE.median()),
            "mean_delta_CE_vs_WT": float(group.delta_CE_vs_WT.mean()),
            "n_anchor_preserving": int(group.anchor_preserving.sum()),
            "n_NMNAT_anchor_perturbed": int(group.NMNAT_anchor_perturbed.sum()),
            "n_NUDIX_anchor_perturbed": int(group.NUDIX_anchor_perturbed.sum()),
            "n_T135_changed": int(group.T135_changed.sum()),
            "n_H236_changed": int(group.H236_changed.sum()),
            "n_substitutions_total": int(group.n_edits.sum()),
            "n_direct_anchor_substitutions": int(group.n_direct_anchor_substitutions.sum()),
            "n_near_window_substitutions": int(group.n_near_window_substitutions.sum()),
            "n_NMNAT_domain_substitutions": int(group.domain_substitutions_NMNAT.sum()),
            "n_NUDIX_domain_substitutions": int(group.domain_substitutions_NUDIX.sum()),
        })
    arm_summary = pd.DataFrame(arm_rows)
    arm_summary.to_csv(RUN / "structural_analysis/arm_summary.csv", index=False)

    domain_rows = []
    candidate_groups = {
        key: value for key, value in candidates.groupby(
            ["source_set", "arm", "stream_type", "edit_budget"], sort=True)
    }
    if len(substitutions):
        for keys, group in substitutions.groupby(
                ["source_set", "arm", "stream_type", "edit_budget", "domain"], sort=True):
            source_set, arm, stream_type, edit_budget, domain = keys
            total = len(candidate_groups[(source_set, arm, stream_type, edit_budget)])
            domain_rows.append({
                "source_set": source_set,
                "arm": arm,
                "stream_type": stream_type,
                "edit_budget": int(edit_budget),
                "domain": domain,
                "candidates_total": total,
                "candidates_with_substitution": int(group.candidate_id.nunique()),
                "substitutions_total": len(group),
                "direct_anchor_substitutions": int(group.direct_anchor.sum()),
                "near_window_substitutions": int(
                    (group.within_4_sequence_positions & ~group.direct_anchor).sum()),
                "unique_positions": int(group.position_1based.nunique()),
            })
    domain_summary = pd.DataFrame(domain_rows)
    domain_summary.to_csv(RUN / "structural_analysis/domain_summary.csv", index=False)

    anchor_rows = []
    for anchor in anchors:
        record = {
            "domain": anchor["domain"],
            "wt_position": anchor["wt_position"],
            "wt_residue": anchor["wt_residue"],
            "identity_conserved_across_templates": anchor["identity_conserved_across_templates"],
            "variable_anchor": anchor["variable_anchor"],
            "designated_template": anchor["designated_template"],
            "min_ligand_distance_A": anchor["min_ligand_distance_A"],
            "mapping_status": anchor["mapping_status"],
            "ligand_proximity_status": anchor["ligand_proximity_status"],
        }
        for prefix, source_set in (("bounded_new", "bounded_new"),
                                   ("formal_WP5_frozen", "formal_WP5_frozen")):
            candidates_set = candidates[candidates.source_set == source_set]
            direct = substitutions[(substitutions.source_set == source_set) &
                                   (substitutions.position_1based == anchor["wt_position"])]
            near = substitutions[(substitutions.source_set == source_set) &
                                 (substitutions.position_1based != anchor["wt_position"]) &
                                 (substitutions.nearest_anchor_position == anchor["wt_position"]) &
                                 substitutions.within_4_sequence_positions]
            record[f"{prefix}_candidates_total"] = len(candidates_set)
            record[f"{prefix}_candidates_changed"] = int(direct.candidate_id.nunique())
            record[f"{prefix}_mutations"] = ";".join(sorted(set(direct.variant)))
            record[f"{prefix}_candidates_near_window"] = int(near.candidate_id.nunique())
            record[f"{prefix}_near_window_mutations"] = ";".join(sorted(set(near.variant)))
        anchor_rows.append(record)
    anchor_summary = pd.DataFrame(anchor_rows)
    anchor_summary.to_csv(RUN / "structural_analysis/anchor_summary.csv", index=False)

    bounded = candidates[candidates.source_set == "bounded_new"]
    formal = candidates[candidates.source_set == "formal_WP5_frozen"]
    direct_candidates = candidates[candidates.n_direct_anchor_substitutions > 0].copy()
    direct_candidates = direct_candidates[[
        "candidate_id", "source_set", "arm", "stream_type", "edit_budget",
        "direct_anchor_mutations", "direct_anchor_domains", "n_direct_anchor_substitutions",
        "n_NMNAT_anchor_substitutions", "n_NUDIX_anchor_substitutions",
        "H236_variable_anchor_changed", "T135_changed", "PLM_blocked_CE",
        "exact_novel_family2430", "CDD_ordered_pair"]]
    direct_candidates.to_csv(RUN / "structural_analysis/direct_anchor_candidates.csv", index=False)

    formal_ce_join = candidates[candidates.source_set == "formal_WP5_frozen"].dropna(
        subset=["frozen_PLM_CE"])
    ce_reproduction = {
        "n": len(formal_ce_join),
        "max_abs_delta": float(formal_ce_join.recomputed_minus_frozen_CE.abs().max()),
        "median_abs_delta": float(formal_ce_join.recomputed_minus_frozen_CE.abs().median()),
        "n_within_1e_4": int((formal_ce_join.recomputed_minus_frozen_CE.abs() <= 1e-4).sum()),
        "n_within_1e_3": int((formal_ce_join.recomputed_minus_frozen_CE.abs() <= 1e-3).sum()),
        "note": "Batch composition differs from the original formal scoring; this is a reproducibility diagnostic.",
    }
    evaluation_receipt = {
        "status": "EVALUATION_COMPLETE",
        "generated_at_utc": utc_now(),
        "unique_sequences": int(candidates.sequence_hash.nunique()),
        "proposals": len(candidates),
        "WT_PLM_blocked_CE": float(wt_quality),
        "WT_PLM_blocked_CE_reference": WT_CE_REFERENCE,
        "WT_PLM_blocked_CE_delta": float(wt_quality - WT_CE_REFERENCE),
        "formal_CE_reproduction": ce_reproduction,
        "formal_CDD_rows_identical": bool(cdd_reproduction.identical.all()),
        "formal_CDD_rows_compared": len(cdd_reproduction),
        "WT_CDD_pair": pairs[wt_sha],
        "post_sampling_amendment": None if amendment is None else {
            "path": "post_sampling_amendment.json",
            "evaluation_script_sha256": amendment["evaluation_script_sha256"],
            "evaluation_revisions": len(amendment["revisions"]),
            "sampling_outputs_unchanged": bool(amendment["sampling_outputs_unchanged"]),
        },
        "test_payload_reads": 0,
        "training_runs": 0,
        "folding_runs": 0,
        "docking_runs": 0,
        "md_runs": 0,
        "wet_lab": False,
    }
    write_json(evaluation / "evaluation_receipt.json", evaluation_receipt)

    def counts(frame):
        direct = frame[frame.n_direct_anchor_substitutions > 0]
        return {
            "n": len(frame),
            "unique": int(frame.sequence_hash.nunique()),
            "duplicates": int(len(frame) - frame.sequence_hash.nunique()),
            "unique_with_direct_anchor": int(direct.sequence_hash.nunique()),
            "unique_anchor_preserving": int(frame[frame.anchor_preserving].sequence_hash.nunique()),
            "unique_NMNAT_anchor_perturbed": int(
                frame[frame.NMNAT_anchor_perturbed].sequence_hash.nunique()),
            "unique_NUDIX_anchor_perturbed": int(
                frame[frame.NUDIX_anchor_perturbed].sequence_hash.nunique()),
            "unique_T135_changed": int(frame[frame.T135_changed].sequence_hash.nunique()),
            "unique_H236_changed": int(frame[frame.H236_changed].sequence_hash.nunique()),
            "exact_novel_family2430": int(frame.exact_novel_family2430.sum()),
            "exact_novel_train1644": int(frame.exact_novel_train1644.sum()),
            "CDD_pass": int(frame.CDD_ordered_pair.sum()),
            "CE_below_WT": int(frame.PLM_CE_below_WT.sum()),
            "anchor_preserving": int(frame.anchor_preserving.sum()),
            "NMNAT_anchor_perturbed": int(frame.NMNAT_anchor_perturbed.sum()),
            "NUDIX_anchor_perturbed": int(frame.NUDIX_anchor_perturbed.sum()),
            "T135_changed": int(frame.T135_changed.sum()),
            "H236_changed": int(frame.H236_changed.sum()),
            "direct_anchor_substitutions": int(frame.n_direct_anchor_substitutions.sum()),
            "near_window_substitutions": int(frame.n_near_window_substitutions.sum()),
            "NMNAT_domain_substitutions": int(frame.domain_substitutions_NMNAT.sum()),
            "NUDIX_domain_substitutions": int(frame.domain_substitutions_NUDIX.sum()),
        }

    by_arm = {
        arm: counts(group) for arm, group in candidates[candidates.source_set == "bounded_new"].groupby("arm")
    }
    formal_by_arm = {
        arm: counts(group) for arm, group in candidates[candidates.source_set == "formal_WP5_frozen"].groupby("arm")
    }
    interpretation = {
        "schema": "natv-generation-structural-interpretation",
        "generated_at_utc": utc_now(),
        "run_script_sha256": sha(Path(__file__)),
        "scope": {
            "bounded_new": "32 actual full-length proposals from two frozen H2 checkpoints, matched 4/8-edit kernel",
            "formal_WP5_frozen": "24 retrospective candidates from the previously frozen formal WP5 set; no new sampling",
            "control": "public KVP40 WT",
        },
        "inputs": {
            "structural_basis": str(STRUCT_BASIS.relative_to(NATV_ROOT)),
            "structural_basis_sha256": sha(STRUCT_BASIS),
            "structural_basis_formal_input_manifest": False,
            "h2_contract_sha256": sha(SOURCE_ROOT / H2_CONTRACT),
            "post_sampling_amendment_sha256": (
                sha(RUN / "post_sampling_amendment.json") if amendment is not None else None),
            "checkpoints": [{"label": a["label"], "sha256": a["sha256"],
                             "architecture": a["architecture"]} for a in ARMS],
        },
        "target": {
            "name": basis["target"]["name"],
            "sequence_sha256": wt_sha,
            "length": len(wt),
            "domains": domains,
            "anchor_count": len(anchors),
            "anchor_mapping_status": basis["anchor_summary"]["mapping_status"],
            "anchor_ligand_proximity_status": basis["anchor_summary"]["designated_ligand_proximity_status"],
            "variable_anchor": basis["anchor_summary"]["variable_anchor"],
        },
        "results": {
            "bounded_new": counts(bounded),
            "bounded_new_by_arm": by_arm,
            "formal_WP5_frozen": counts(formal),
            "formal_WP5_frozen_by_arm": formal_by_arm,
            "control": counts(candidates[candidates.source_set == "control"]),
            "direct_anchor_candidates": direct_candidates.to_dict(orient="records"),
        },
        "interpretation": {
            "NMNAT_environment": (
                "Direct substitutions at H14, H17, T135 or R138 are mapped to the "
                "audited ATP-motif/catalysis-adjacent proxies from the 2QJO NAD "
                "product state. Other NMNAT-domain substitutions are mapped by "
                "domain only. These are structural correspondences; no kcat, "
                "binding, flux or net NAD+ claim follows."
            ),
            "NUDIX_hydrolysis": (
                "Direct substitutions at G232, H236, R247, E248, E251, E252, D307 "
                "or D308 are mapped to the audited Nudix motif/metal-water network "
                "proxies from the 2QJT AMP+Mn state. H236 is a variable, "
                "non-identity-conserved positional anchor. Whether any substitution "
                "raises or lowers hydrolysis is UNKNOWN; no risk direction is assigned."
            ),
            "T135": "Any T135 substitution is an NMNAT anchor perturbation; T135A is forbidden by the frozen constraint.",
            "near_window": "Within +/-4 sequence positions is a sequence-window proxy, not a 3D distance.",
            "A_B_C_note": "The bookkeeping labels are descriptive categories, not a required three-architecture design.",
        },
        "evidence_semantics": basis["evidence_semantics"],
        "limitations": [
            "The H2 models are WT-relative masked local editors with fixed 4/8 edits; they are not de novo or learned joint full-length generators.",
            "The two arms share the training contract but differ in architecture and capacity; arm differences are descriptive and not a causal latent-mechanism test.",
            "Internal blocked-8 ESM2 CE shares the ESM2 family prior and is dependent compatibility evidence, not activity.",
            "CDD ordered double-domain presence is family compatibility, not catalysis.",
            "No folding, docking, MD, wet-lab assay or experimental net NAD+ measurement was performed.",
            "The structural basis is a discovery artifact and is not part of the formal training population or global input manifest.",
            "Finite single-seed draws and one parent cannot support population-level or method-superiority claims.",
            "The formal WP5 set is retrospective and was selected by the earlier formal protocol; it is not an independent new sample.",
        ],
        "test_payload_reads": 0,
        "training_runs": 0,
        "folding_runs": 0,
        "wet_lab": False,
    }
    write_json(RUN / "structural_analysis/structural_interpretation.json", interpretation)
    write_report(candidates, substitutions, arm_summary, domain_summary, anchor_summary,
                 direct_candidates, evaluation_receipt, interpretation, cdd_reproduction)
    status = read_json(RUN / "job_status.json")
    status.update({
        "status": "RUN_COMPLETE",
        "evaluation": evaluation_receipt,
        "structural_interpretation": "structural_analysis/structural_interpretation.json",
        "report": "REPORT.md",
    })
    write_json(RUN / "job_status.json", status)
    print(json.dumps(evaluation_receipt, indent=2))


def md_table(frame, columns):
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                text = "" if not np.isfinite(value) else f"{value:.6g}"
            elif isinstance(value, (bool, np.bool_)):
                text = "True" if value else "False"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(candidates, substitutions, arm_summary, domain_summary, anchor_summary,
                 direct_candidates, receipt, interpretation, cdd_reproduction):
    bounded = candidates[candidates.source_set == "bounded_new"]
    formal = candidates[candidates.source_set == "formal_WP5_frozen"]
    control = candidates[candidates.source_set == "control"].iloc[0]
    bounded_direct = bounded[bounded.n_direct_anchor_substitutions > 0]
    formal_direct = formal[formal.n_direct_anchor_substitutions > 0]
    direct_unique = int(pd.concat([bounded_direct, formal_direct]).sequence_hash.nunique())
    duplicate_b4 = int(len(bounded[bounded.edit_budget == 4]) -
                       bounded[bounded.edit_budget == 4].sequence_hash.nunique())
    duplicate_b8 = int(len(bounded[bounded.edit_budget == 8]) -
                       bounded[bounded.edit_budget == 8].sequence_hash.nunique())
    key_arm_columns = [
        "source_set", "arm", "stream_type", "edit_budget", "n_proposals",
        "n_unique_sequences", "n_exact_novel_family2430", "n_CDD_pass",
        "n_CE_below_WT", "median_PLM_CE", "n_anchor_preserving",
        "n_NMNAT_anchor_perturbed", "n_NUDIX_anchor_perturbed",
        "n_T135_changed", "n_H236_changed",
    ]
    anchor_columns = [
        "domain", "wt_position", "wt_residue", "identity_conserved_across_templates",
        "variable_anchor", "bounded_new_candidates_changed", "bounded_new_mutations",
        "formal_WP5_frozen_candidates_changed", "formal_WP5_frozen_mutations",
    ]
    domain_columns = [
        "source_set", "arm", "edit_budget", "domain", "candidates_total",
        "candidates_with_substitution", "substitutions_total",
        "direct_anchor_substitutions", "near_window_substitutions",
    ]
    direct_columns = [
        "candidate_id", "source_set", "arm", "edit_budget",
        "direct_anchor_mutations", "n_direct_anchor_substitutions",
        "n_NMNAT_anchor_substitutions", "n_NUDIX_anchor_substitutions",
        "H236_variable_anchor_changed", "T135_changed", "CDD_ordered_pair",
    ]
    amendment_info = receipt.get("post_sampling_amendment")
    if amendment_info:
        amendment_note = (
            f"- Evaluation-stage amendment: `post_sampling_amendment.json` "
            f"({amendment_info['evaluation_revisions']} evaluation-script revision(s); "
            f"sampling artifacts unchanged: {amendment_info['sampling_outputs_unchanged']}). "
            f"The evaluation code was corrected after sampling was frozen; no sampling "
            f"output was regenerated.")
    else:
        amendment_note = "- No post-sampling amendment was required."
    report = f"""# Bounded NatV generation and structure-informed annotation

This run generated 32 full-length KVP40 NatV variants from two frozen H2
checkpoints and annotated them against the previously audited experimental
template anchors. It is a finite conditional local-editing study, not a
de-novo generator and not a catalytic-activity measurement.

## Declared scope

- Checkpoints: `H2_no_latent/20261081` (SHA256 `{ARMS[0]['sha256']}`) and
  `H2_continuous_mmd/20261081` (SHA256 `{ARMS[1]['sha256']}`).
- Shared H2 training contract, tokenizer, F-train retrieval database and split.
- Matched streams: seed `{STREAM_SEED}`, 8 streams per edit budget, budgets
  {EDIT_BUDGETS}, 4 sequences per batch, 7 masked-edit stages, temperature
  {TEMPERATURE}, T135A forbidden.
- 32 proposals total; the `continuous_mmd` arm draws one latent prior per
  four-sequence trajectory and holds it fixed across all seven stages; the
  `no_latent` arm has no latent branch.
- No training, folding, docking, MD, remote download, wet-lab work or test
  sequence payload reads.

## Inputs and provenance

- H2 contract SHA256: `{interpretation['inputs']['h2_contract_sha256']}`.
- Structural basis: `{interpretation['inputs']['structural_basis']}`,
  SHA256 `{interpretation['inputs']['structural_basis_sha256']}`. It remains a
  discovery artifact (`formal_input_manifest=false`); it is used only for
  annotation, not training or candidate selection.
- Run-specific input hashes: `input_manifest.json`.
- Pre-sampling freeze: `pre_sampling_freeze.json`.
{amendment_note}
- Formal WP5 retrospective reproduction: `{receipt['formal_CE_reproduction']['n']}`
  candidates; maximum |CE difference| `{receipt['formal_CE_reproduction']['max_abs_delta']:.6g}`;
  CDD rows identical: `{receipt['formal_CDD_rows_identical']}`.

## Sampling validity

- 32/32 proposals are canonical 341 aa, with exactly 4 or 8 substitutions.
- Final edits equal the frozen stream positions; T135A violations: 0.
- Unique sequence hashes: {bounded.sequence_hash.nunique()}/32. The two arms
  produced identical sequences for {duplicate_b4} budget-4 stream pairs and
  {duplicate_b8} budget-8 stream pair; duplicates remain in all proposal-level
  denominators.
- Exact novelty against the 2,430 family role table:
  {int(bounded.exact_novel_family2430.sum())}/32; against the 1,644 F-train
  sequences: {int(bounded.exact_novel_train1644.sum())}/32.
- Ordered CDD double-domain pair: {int(bounded.CDD_ordered_pair.sum())}/32.
- WT blocked-8 ESM2 CE: {control.PLM_blocked_CE:.9f} (reference
  {WT_CE_REFERENCE:.9f}; delta {receipt['WT_PLM_blocked_CE_delta']:.6g}).

## Per-arm descriptive results

{md_table(arm_summary[arm_summary.source_set != 'control'], key_arm_columns)}

The two new H2 arms share the training contract but differ in architecture and
capacity. These are descriptive finite-sample counts, not a method-superiority
test. Internal ESM2 CE shares the ESM2 prior and is not activity evidence.

## Domain substitution summary

{md_table(domain_summary[domain_summary.source_set != 'control'], domain_columns)}

## Audited anchor mapping

{md_table(anchor_summary, anchor_columns)}

Direct-anchor candidates: {len(direct_candidates)} proposals
({len(bounded_direct)} new + {len(formal_direct)} retrospective) but only
{direct_unique} unique sequences, because some cross-arm proposals are
identical. The complete list is in `structural_analysis/direct_anchor_candidates.csv`.

### NMNAT catalytic environment

Direct substitutions at H14, H17, T135 or R138 map to the audited
ATP-motif/catalysis-adjacent proxies from the 2QJO NAD product state. Other
NMNAT-domain substitutions are mapped by domain only. New candidates with
direct NMNAT anchor changes:
{int(bounded.NMNAT_anchor_perturbed.sum())}/32 proposals
({int(bounded[bounded.NMNAT_anchor_perturbed].sequence_hash.nunique())} unique
sequences); retrospective:
{int(formal.NMNAT_anchor_perturbed.sum())}/24
({int(formal[formal.NMNAT_anchor_perturbed].sequence_hash.nunique())} unique).
The retrospective T135 change is T135S, not T135A. T135 changes: new
{int(bounded.T135_changed.sum())}/32; retrospective {int(formal.T135_changed.sum())}/24.
No kcat, binding, flux or net NAD+ claim follows from these mappings.

### NUDIX hydrolysis

Direct substitutions at G232, H236, R247, E248, E251, E252, D307 or D308 map
to the audited Nudix motif/metal-water network proxies from the 2QJT AMP+Mn
state. H236 is a variable, non-identity-conserved positional anchor; changes
there are mapped perturbations, not identity-conserved anchor losses. New
candidates with direct NUDIX anchor changes:
{int(bounded.NUDIX_anchor_perturbed.sum())}/32 proposals
({int(bounded[bounded.NUDIX_anchor_perturbed].sequence_hash.nunique())} unique
sequences); retrospective:
{int(formal.NUDIX_anchor_perturbed.sum())}/24
({int(formal[formal.NUDIX_anchor_perturbed].sequence_hash.nunique())} unique).
H236 changes: new
{int(bounded.H236_changed.sum())}/32; retrospective {int(formal.H236_changed.sum())}/24.
Whether any substitution raises or lowers hydrolysis is UNKNOWN; no risk
direction is assigned.

## Interpretation limits

- The models perform WT-relative masked local editing with fixed 4/8 edits;
  they are not de novo or learned joint full-length generators.
- The structural basis is a discovery-derived mapping, not a new experimental
  structure; `within +/-4 sequence positions` is a sequence-window proxy, not a
  3D distance.
- No folding, docking, MD, wet-lab assay or experimental net NAD+ measurement
  was performed. Model and CDD scores are compatibility/diagnostic evidence.
- The formal 24 candidates are retrospective, selected by the earlier formal
  protocol; they are not independent new samples.
- Proposal-level counts include five cross-arm duplicate pairs; unique-sequence
  counts are reported alongside them.
- Single-seed finite draws do not support population-level or method-superiority
  claims.

## Decision

The structural correspondence is useful for interpreting where generated
substitutions land, but it does not establish a catalytic benefit. The formal
training population and splits remain unchanged. The structural basis is
recorded in the run-specific `input_manifest.json`; it is not promoted to the
global training input manifest because it did not enter training or candidate
selection. Data expansion remains closed for the current objective.

## Reproduction

```bash
export PATH=/home/yhshy/miniconda3/envs/nad_data/bin:$PATH
export NATV_SOURCE_ROOT=/home/yhshy/git_files/nad_qvae
python experiments/natv_generation_structural/run_generation.py --stage preflight
python experiments/natv_generation_structural/run_generation.py --stage sample
python experiments/natv_generation_structural/run_generation.py --stage evaluate
```

## Files

- `sampling_manifest.json`, `pre_sampling_freeze.json`, `preflight_report.json`,
  `input_manifest.json`, `post_sampling_amendment.json`, `job_status.json`
- `proposals.csv`, `sampling_trace.jsonl.gz`, `retrieval_receipts.json.gz`,
  `retrieval_hits_summary.csv`
- `evaluation/CDD_domains.csv`, `evaluation/CDD_pair_summary.csv`,
  `evaluation/internal_PLM_scores.csv`, `evaluation/CDD_reproduction_check.csv`,
  `evaluation/evaluation_receipt.json`
- `structural_analysis/candidate_level.csv`, `substitution_level.csv`,
  `anchor_summary.csv`, `domain_summary.csv`, `arm_summary.csv`,
  `direct_anchor_candidates.csv`, `structural_interpretation.json`
- Raw MMseqs outputs under `retrieval/` are local-only and git-ignored.
"""
    (RUN / "REPORT.md").write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["preflight", "sample", "evaluate"], required=True)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "sample":
        sample()
    else:
        evaluate()


if __name__ == "__main__":
    main()
