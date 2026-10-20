"""生成本次 NatV 候选序列，并标注替换落在哪些结构域和关键残基上。

本次只读取 nad_qvae 中已经训练好的两组 H2 模型参数，不重新训练。每组模型
按提前记录的随机种子，在 4 个或 8 个位点上做局部编辑，每个编辑数 8 组方案，
共 32 条候选序列；每条序列经过 7 步掩码编辑。没有折叠、对接、分子动力学、
远程下载、湿实验，也没有读取测试序列内容。

结构注释依据 discovery/natv_pairing_audit/structural_interpretation_basis.json。
该文件不参与训练、同源检索或候选挑选，只用于说明替换与实验模板中已核对
关键残基的相对位置。
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

RUN = NATV_ROOT / "experiments/natv_candidate_sequences"
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
MODEL_GROUPS = [
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
        "source": root,
        "path": rel,
        "bytes": full.stat().st_size,
        "sha256": current,
        "purpose": role,
        "used_in": used_by,
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
    for position, old, new in parse_variants(candidate["substitutions"]):
        direct = [a for a in anchors if a["wt_position"] == position]
        closest = direct[0] if direct else nearest_anchor(position, anchors)
        distance = abs(closest["wt_position"] - position)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "source_set": candidate["source_set"],
            "model_group": candidate["model_group"],
            "plan_type": candidate["plan_type"],
            "edit_count": candidate["edit_count"],
            "position_1based": position,
            "wt_aa": old,
            "mutant_aa": new,
            "substitution": f"{old}{position}{new}",
            "domain": domain_of(position, domains),
            "direct_key_residue": bool(direct),
            "key_residue_wt_position": position if direct else "",
            "key_residue_wt_residue": old if direct else "",
            "key_residue_domain": closest["domain"],
            "key_residue_identity_conserved": closest["identity_conserved_across_templates"],
            "key_residue_variable": closest["variable_anchor"],
            "key_residue_designated_template": closest["designated_template"],
            "key_residue_min_ligand_distance_A": closest["min_ligand_distance_A"],
            "nearest_key_residue_position": closest["wt_position"],
            "nearest_key_residue_distance": distance,
            "within_4_positions": distance <= 4,
            "within_4_is_sequence_distance_not_3d": True,
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
    for arm in MODEL_GROUPS:
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
    require(not (RUN / "sampling_plan_checksums.json").exists(), "Preflight already finalized")

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
    check("H2 上游设置文件", h2_contract_sha)

    for key, item in h2_contract["inputs"].items():
        current = sha(SOURCE_ROOT / item["path"])
        require(current == item["sha256"], f"H2 input changed: {key}")
    check("H2 输入文件记录", "input_plan, context_manifest, data_ready")

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
    check("正式训练输入", sha(SOURCE_ROOT / FORMAL_PLAN))

    roles_path = SOURCE_ROOT / CORE / "family_training_roles_v2.parquet"
    core_manifest_path = SOURCE_ROOT / CORE / "core_split_manifest_v2.json"
    core_manifest = read_json(core_manifest_path)
    require(sha(roles_path) == core_manifest["core_manifest_sha256"], "Source role manifest changed")
    roles = pd.read_parquet(roles_path)
    require(len(roles) == 2430 and roles.sequence_id.is_unique, "Source population changed")
    train_ids = set(formal_plan["retrieval_database_ids"])
    require(train_ids <= set(roles.loc[roles.core_training_eligible_v2, "sequence_id"]),
            "Retrieval database contains non-core-training records")
    check("家族角色表", sha(roles_path))

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
    check("上游源码差异", "only vnext_model.py; equals H2 historical basis")

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
    check("ESM2 模型与词表", encoder_files["model.safetensors"])

    wt, wt_sha, wt_source = load_wt()
    require(wt_sha == h2_contract["generation"]["parent_sha256"], "H2 parent differs from WT")
    constraints = read_json(SOURCE_ROOT / CONSTRAINTS)
    require(constraints["forbidden_substitutions"] == [
        {"position_1based": 135, "WT_AA": "T", "forbidden_AA": "A"}],
        "T135A constraint changed")
    check("野生型序列与 T135A 限制", wt_sha)

    checkpoint_records = []
    for arm in MODEL_GROUPS:
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
    model_records = [
        {
            "name": record["label"],
            "architecture": record["architecture"],
            "path": record["path"],
            "sha256": record["sha256"],
            "training_seed": record["training_seed"],
            "epoch": record["epoch"],
            "steps": record["steps"],
            "validation_ESM2_CE": record["validation_CE"],
            "parameter_tensors": record["parameter_tensors"],
            "parameters": record["parameters"],
            "finite_parameters": record["finite_state_dict"],
        }
        for record in checkpoint_records
    ]
    selected = read_json(SOURCE_ROOT / H2_RUN / "selected_checkpoints.json")
    for arm in MODEL_GROUPS:
        record = selected[f'{arm["architecture"]}/{TRAINING_SEED}']
        require(int(record["epoch"]) == 2 and abs(float(record["validation_CE"]) - arm["validation_CE"]) < 1e-12,
                "Selected checkpoint metadata differs")
    check("两组已训练模型参数", [a["sha256"] for a in checkpoint_records])

    basis = read_json(STRUCT_BASIS)
    domains = basis["target"]["domains"]
    anchors = prepare_anchors(basis, wt, domains)
    require(sha(STRUCT_SCRIPT) == basis["script_sha256"], "Structural-basis script identity differs")
    check("结构注释依据", sha(STRUCT_BASIS))

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
    check("旧 WP5 候选序列", sha(SOURCE_ROOT / FORMAL_CANDIDATES))

    db_files, data_files, binaries = cdd_assets()
    cdd_digests = {str(path.relative_to(SOURCE_ROOT)): sha(path) for path in db_files + data_files + binaries}
    check("CDD 程序与数据库", f"{len(cdd_digests)} files")

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
    check("编辑方案", f"{len(streams)} streams, {len(batches)} batches")

    script_sha = sha(Path(__file__))
    sampling_plan = {
        "record": "NatV 候选生成方案",
        "written_at_utc": utc_now(),
        "script_sha256": script_sha,
        "source_root": str(SOURCE_ROOT),
        "parent": {"sequence_sha256": wt_sha, "length": len(wt)},
        "model_parameters": model_records,
        "settings": {
            "plan_seed": STREAM_SEED,
            "prior_seed_base": PRIOR_SEED,
            "prior_draw": "每组四条序列在生成开始时抽取一次先验，七步共用",
            "edit_counts": EDIT_BUDGETS,
            "plans_per_edit_count": STREAMS_PER_BUDGET,
            "sweeps": SWEEPS,
            "temperature": TEMPERATURE,
            "group_size": BATCH_SIZE,
            "steps": 1 + 2 * SWEEPS,
            "forbidden_changes": "任何一步都禁止 T135A；可编辑位点禁止改回 WT",
            "homolog_search": "每一步用当前掩码序列，在 1,644 条训练序列中重新检索",
            "ESM2": "每一步用当前掩码序列重新计算",
            "plan_type": "固定编辑位点",
        },
        "edit_plans": [
            {
                "plan": record["stream"],
                "edit_count": record["edit_budget"],
                "edit_positions_0based": record["edit_positions"],
                "update_masks_0based": record["update_masks"],
                "uniform_draws": record["uniform_draws"],
            }
            for record in streams
        ],
        "groups": [
            {
                "group_id": group["batch_id"],
                "model_group": group["arm"],
                "architecture": group["architecture"],
                "edit_count": group["edit_budget"],
                "start_plan": group["start_stream"],
                "prior_seed": group["prior_seed"],
                "latent_from_prior": group["prior_sampled"],
                "plans": [record["stream"] for record in group["streams"]],
            }
            for group in batches
        ],
        "candidate_ids": [proposal_id(b, record["stream"]) for b in batches for record in b["streams"]],
        "candidate_count": len(batches) * BATCH_SIZE,
    }
    require(sampling_plan["candidate_count"] == 32, "Candidate count differs")
    write_json(RUN / "sampling_plan.json", sampling_plan)

    inputs = []
    seen = set()

    def add(path, purpose, used_in, expected=None, root_hint=None):
        entry = input_entry(path, purpose, used_in, expected, root_hint)
        key = (entry["source"], entry["path"])
        if key not in seen:
            seen.add(key)
            inputs.append(entry)

    add(H2_CONTRACT, "H2 上游设置文件", "生成", h2_contract_sha)
    add(RUNTIME_CONTRACT, "确定性运行设置", "生成与评价")
    add(TRAIN_CONTRACT, "训练与划分设置", "来源记录")
    add(FORMAL_PLAN, "正式训练输入方案", "生成", data_ready["input_plan_sha256"])
    add(DATA_READY, "正式训练数据准备记录", "生成与评价")
    add(FORMAL_DATA / "context_manifest.json", "掩码上下文记录", "来源记录",
        h2_contract["inputs"]["masked_context_manifest"]["sha256"])
    add(FORMAL_DATA / "training_samples.csv", "1,644 条训练序列", "生成", formal_plan["training_samples_sha256"])
    add(FORMAL_DATA / "train_only.fasta", "训练序列 FASTA（同源检索库）", "生成", formal_plan["train_fasta_sha256"])
    add(FORMAL_DATA / "allowed_samples.parquet", "允许使用的样本角色", "生成", formal_plan["allowed_samples_sha256"])
    add(CORE / "family_training_roles_v2.parquet", "家族角色表；用于判断序列是否与现有家族重复", "评价")
    add(CORE / "core_split_manifest_v2.json", "核心划分记录", "来源记录")
    add(WT_SOURCE, "野生型序列与身份", "生成")
    add(CONSTRAINTS, "T135A 限制", "生成")
    add(H2_RUN / "selected_checkpoints.json", "验证集选出的 H2 模型参数", "生成")
    add(H2_RUN / "generation/checkpoint_freeze.json", "H2 生成阶段的参数记录", "生成")
    for model_group in MODEL_GROUPS:
        add(model_group["checkpoint"], f'{model_group["label"]} 模型参数', "生成", model_group["sha256"])
    for name, digest in encoder_files.items():
        add(Path("nad_models/esm2") / name, "ESM2 模型文件", "生成与评价", digest)
    add(STRUCT_BASIS, "结构注释依据", "评价", root_hint="natvqvae")
    add(STRUCT_SCRIPT, "结构注释依据的生成脚本", "来源记录", root_hint="natvqvae")
    add(FORMAL_CANDIDATES, "旧 WP5 候选序列", "评价")
    add(FORMAL_CDD, "旧 WP5 的 CDD 结构域结果", "评价与复核")
    add(FORMAL_CE, "旧 WP5 的 ESM2 分数", "评价与复核")
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
        add(Path(name), "实际执行的源码", "生成与评价")
    for path in db_files + data_files + binaries:
        add(path, "CDD 程序与数据库", "评价")
    inputs.append({
        "source": "environment",
        "path": mmseqs,
        "bytes": Path(mmseqs).stat().st_size,
        "sha256": sha(mmseqs),
        "purpose": "MMseqs 程序",
        "used_in": "生成",
        "version": version,
    })
    input_files = {
        "record": "本次运行读取的文件",
        "written_at_utc": utc_now(),
        "scope": "仅记录本次运行实际读取的文件；正式训练序列集没有改变",
        "files": inputs,
    }
    write_json(RUN / "input_files.json", input_files)

    input_checks = {
        "record": "生成前输入检查",
        "status": "INPUT_CHECKS_PASSED",
        "written_at_utc": utc_now(),
        "script_sha256": script_sha,
        "runtime": environment,
        "device": torch.cuda.get_device_name(0),
        "checks": checks,
        "source_changes": drift,
        "h2_settings_sha256": h2_contract_sha,
        "data_ready_sha256": sha(SOURCE_ROOT / DATA_READY),
        "training_plan_sha256": sha(SOURCE_ROOT / FORMAL_PLAN),
        "family_roles_sha256": sha(roles_path),
        "structure_basis_sha256": sha(STRUCT_BASIS),
        "previous_candidates_sha256": sha(SOURCE_ROOT / FORMAL_CANDIDATES),
        "model_parameters": model_records,
        "cdd_files": cdd_digests,
        "sequence_sets": {
            "training": int(roles.core_training_eligible_v2.sum()),
            "family_rows": len(roles),
            "homolog_search_ids": len(train_ids),
            "test_sequence_reads": 0,
        },
        "work_declared": {
            "training_runs": 0,
            "folding_runs": 0,
            "docking_runs": 0,
            "md_runs": 0,
            "wet_lab": False,
            "remote_downloads": 0,
            "test_sequence_reads": 0,
            "candidate_count": 32,
        },
    }
    write_json(RUN / "input_checks.json", input_checks)

    checksums = {
        "record": "NatV 候选生成方案的文件校验记录",
        "status": "SAMPLING_PLAN_RECORDED",
        "written_at_utc": utc_now(),
        "script_sha256": script_sha,
        "source_root": str(SOURCE_ROOT),
        "file_sha256": {
            "sampling_plan.json": sha(RUN / "sampling_plan.json"),
            "input_files.json": sha(RUN / "input_files.json"),
            "input_checks.json": sha(RUN / "input_checks.json"),
        },
        "h2_settings_sha256": h2_contract_sha,
        "structure_basis_sha256": sha(STRUCT_BASIS),
        "previous_candidates_sha256": sha(SOURCE_ROOT / FORMAL_CANDIDATES),
        "constraints_sha256": sha(SOURCE_ROOT / CONSTRAINTS),
        "runtime": environment,
        "model_parameters": model_records,
        "sampling_settings": {
            "plan_seed": STREAM_SEED,
            "prior_seed_base": PRIOR_SEED,
            "edit_counts": EDIT_BUDGETS,
            "plans_per_edit_count": STREAMS_PER_BUDGET,
            "sweeps": SWEEPS,
            "group_size": BATCH_SIZE,
            "candidate_count": 32,
        },
        "before_sampling": {
            "candidates_written": 0,
            "evaluation_written": False,
            "training_runs": 0,
            "folding_runs": 0,
            "wet_lab": False,
            "remote_downloads": 0,
            "test_sequence_reads": 0,
        },
        "source_changes": drift,
    }
    write_json(RUN / "sampling_plan_checksums.json", checksums)
    print(json.dumps({"status": "INPUT_CHECKS_PASSED",
                      "model_parameters": model_records,
                      "candidates": sampling_plan["candidate_count"],
                      "checksums_sha256": sha(RUN / "sampling_plan_checksums.json")}, indent=2))


def verify_plan():
    record = read_json(RUN / "sampling_plan_checksums.json")
    require(record["status"] == "SAMPLING_PLAN_RECORDED", "Sampling plan record missing")
    require(record["script_sha256"] == sha(Path(__file__)),
            "Run script changed after the sampling plan was recorded")
    for name, digest in record["file_sha256"].items():
        require(sha(RUN / name) == digest, f"Recorded file changed: {name}")
    require(record["h2_settings_sha256"] == sha(SOURCE_ROOT / H2_CONTRACT), "H2 settings changed")
    require(record["structure_basis_sha256"] == sha(STRUCT_BASIS), "Structure basis changed")
    require(record["previous_candidates_sha256"] == sha(SOURCE_ROOT / FORMAL_CANDIDATES),
            "Previous WP5 table changed")
    require(record["constraints_sha256"] == sha(SOURCE_ROOT / CONSTRAINTS), "Constraint changed")
    for model_group in MODEL_GROUPS:
        require(sha(SOURCE_ROOT / model_group["checkpoint"]) == model_group["sha256"],
                "Model parameters changed")
    input_files = read_json(RUN / "input_files.json")
    for item in input_files["files"]:
        if item["source"] == "nad_qvae":
            path = SOURCE_ROOT / item["path"]
        elif item["source"] == "natvqvae":
            path = NATV_ROOT / item["path"]
        else:
            path = Path(item["path"])
        require(sha(path) == item["sha256"], f"Input changed after the plan was recorded: {item['path']}")
    runtime.check()
    return record


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
                "latent_source": "从先验分布抽取" if batch["prior_sampled"] else "不适用",
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
            "source_set": "new_candidates",
            "arm": batch["arm"],
            "architecture": batch["architecture"],
            "training_seed": TRAINING_SEED,
            "checkpoint_sha256": next(a["sha256"] for a in MODEL_GROUPS if a["label"] == batch["arm"]),
            "batch_id": batch["batch_id"],
            "prior_seed": batch["prior_seed"],
            "prior_sampled": batch["prior_sampled"],
            "stream_type": "固定编辑位点",
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
            "latent_source": "从先验分布抽取" if batch["prior_sampled"] else "不适用",
            "latent_sha256": latent_hashes[record["stream"] - batch["start_stream"]],
        })
    runtime.check()
    return sequences, trace, hits_record


def sample():
    runtime.configure()
    plan_record = verify_plan()
    require(not (RUN / "candidate_sequences.csv").exists(), "Candidate table already exists")
    require(not (RUN / "sampling_steps.jsonl.gz").exists(), "Sampling steps already exist")
    data = SEReader()
    tokens = list(data.tokens)
    require(tokens == list(read_json(SOURCE_ROOT / DATA_READY)["aa_token_ids"]), "Tokenizer changed")
    plan = read_json(RUN / "sampling_plan.json")
    require(sha(RUN / "sampling_plan.json") == plan_record["file_sha256"]["sampling_plan.json"],
            "Sampling plan changed")
    homolog_search = H2Retrieval(data)
    encoder = FrozenHFMaskedEncoder(SOURCE_ROOT / "nad_models/esm2", device="cuda")
    require(encoder.tokenizer.convert_tokens_to_ids(list(AA)) == tokens, "Tokenizer identity differs")
    wt, wt_sha, _ = load_wt()
    require(wt_sha == plan["parent"]["sequence_sha256"], "WT changed")
    plan_by_id = {
        (item["edit_count"], item["plan"]): {
            "stream": item["plan"],
            "edit_budget": item["edit_count"],
            "edit_positions": item["edit_positions_0based"],
            "update_masks": item["update_masks_0based"],
            "uniform_draws": item["uniform_draws"],
        }
        for item in plan["edit_plans"]
    }
    groups = []
    for group in plan["groups"]:
        groups.append({
            "batch_id": group["group_id"],
            "arm": group["model_group"],
            "architecture": group["architecture"],
            "edit_budget": group["edit_count"],
            "start_stream": group["start_plan"],
            "prior_seed": group["prior_seed"],
            "prior_sampled": group["latent_from_prior"],
            "streams": [plan_by_id[(group["edit_count"], plan_id)]
                        for plan_id in group["plans"]],
        })
    candidates = []
    traces = []
    hit_records = []
    search_records = []
    started = time.monotonic()
    for arm in MODEL_GROUPS:
        saved, model = load_checkpoint(
            SOURCE_ROOT / arm["checkpoint"], arm["architecture"], arm["sha256"],
            runtime.environment(), tokens, read_json(SOURCE_ROOT / H2_CONTRACT)["dimensions"])
        model = model.to("cuda").eval()
        for group in [item for item in groups if item["arm"] == arm["label"]]:
            folder = RUN / "homolog_search" / group["batch_id"]
            sequences, group_trace, group_hits = trajectory(
                model, encoder, homolog_search, wt, group, folder)
            candidates.extend(sequences)
            traces.extend(group_trace)
            hit_records.extend(group_hits)
            for stage in range(1 + 2 * SWEEPS):
                receipt_path = folder / f"stage{stage}" / "retrieval_receipt.json"
                record_path = receipt_path.with_name("search_record.json")
                if receipt_path.exists():
                    receipt_path.rename(record_path)
                require(record_path.exists(), "Homolog search record missing")
                search_records.append({
                    "group_id": group["batch_id"],
                    "step": stage,
                    "path": record_path.relative_to(RUN).as_posix(),
                    "sha256": sha(record_path),
                    "search_record": read_json(record_path),
                })
            write_json(RUN / "sampling_progress.json", {
                "status": "SAMPLING_RUNNING",
                "candidates": len(candidates),
                "groups_completed": len({p["batch_id"] for p in candidates}),
                "elapsed_seconds": time.monotonic() - started,
            })
            print(json.dumps({"group": group["batch_id"], "candidates": len(sequences),
                              "elapsed_seconds": time.monotonic() - started}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    frame = pd.DataFrame(candidates)
    require(len(frame) == 32 and frame.proposal_id.is_unique, "Candidate identity/count mismatch")
    require(set(frame.arm) == {arm["label"] for arm in MODEL_GROUPS}, "Model group coverage differs")
    require(frame.groupby(["arm", "edit_budget"]).size().eq(8).all(), "Edit count differs")
    require(frame.sequence_hash.map(lambda value: len(value) == 64).all(), "Sequence hash malformed")
    require(frame.apply(lambda row: text_sha(row.sequence) == row.sequence_hash, axis=1).all(),
            "Sequence hash differs")
    public_frame = frame.rename(columns={
        "proposal_id": "candidate_id",
        "arm": "model_group",
        "checkpoint_sha256": "model_sha256",
        "batch_id": "group_id",
        "prior_sampled": "latent_from_prior",
        "stream_type": "plan_type",
        "edit_budget": "edit_count",
        "stream": "plan",
        "variants": "substitutions",
        "n_edits": "n_changes",
        "syntax_valid": "sequence_valid",
    })
    public_frame["source_set"] = public_frame["source_set"].replace({"bounded_new": "new_candidates"})
    public_frame["plan_type"] = public_frame["plan_type"].replace({"matched": "固定编辑位点"})
    public_frame.to_csv(RUN / "candidate_sequences.csv", index=False)
    with gzip.open(RUN / "sampling_steps.jsonl.gz", "wt") as handle:
        for item in traces:
            handle.write(json.dumps({
                "group_id": item["batch_id"],
                "model_group": item["arm"],
                "edit_count": item["edit_budget"],
                "plan": item["stream"],
                "step": item["stage"],
                "updated_positions_0based": item["update_positions_0based"],
                "before_sequence": item["before"],
                "after_sequence": item["after"],
                "before_sha256": item["before_sha256"],
                "after_sha256": item["after_sha256"],
                "masked_query_sha256": item["masked_query_sha256"],
                "latent_source": item["latent_source"],
                "latent_sha256": item["latent_sha256"],
                "homolog_hit_count": item["hit_count"],
                "homolog_neff": item["neff"],
            }, allow_nan=False) + "\n")
    with gzip.open(RUN / "homolog_search_records.json.gz", "wt") as handle:
        json.dump({"record": "MMseqs 同源检索记录", "searches": search_records},
                  handle, allow_nan=False)
        handle.write("\n")
    hit_frame = pd.DataFrame(hit_records).rename(columns={
        "batch_id": "group_id",
        "stage": "step",
        "stream": "plan",
        "hit_count": "homolog_hit_count",
        "neff": "homolog_neff",
    })
    hit_frame.to_csv(RUN / "homolog_search_summary.csv", index=False)
    status = {
        "status": "SAMPLING_DONE",
        "candidates": len(public_frame),
        "unique_sequences": int(public_frame.sequence_hash.nunique()),
        "groups": len(plan["groups"]),
        "homolog_search_steps": len(search_records),
        "model_forward_calls": len(plan["groups"]) * (1 + 2 * SWEEPS),
        "sequence_encodings": len(plan["groups"]) * BATCH_SIZE * (1 + 2 * SWEEPS),
        "training_runs": 0,
        "folding_runs": 0,
        "docking_runs": 0,
        "md_runs": 0,
        "wet_lab": False,
        "remote_downloads": 0,
        "test_sequence_reads": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(RUN / "run_status.json", status)
    print(json.dumps(status, indent=2))


def formal_reference():
    table = pd.read_csv(SOURCE_ROOT / FORMAL_CANDIDATES)
    control = table[table.role == "PUBLIC_WT_CONTROL"].iloc[0]
    generated = table[table.role == "FORMAL_GENERATED_CANDIDATE"].copy().rename(columns={
        "proposal_id": "previous_candidate_id",
        "quota_arm": "previous_model_group",
        "quota_stream": "previous_plan_type",
        "edit_budget": "edit_count",
        "variants": "substitutions",
        "PLM_blocked_CE_diagnostic_only": "previous_ESM2_CE",
    })
    return control, generated


def evaluate():
    runtime.configure()
    plan_record = verify_plan()
    require((RUN / "candidate_sequences.csv").exists(), "Candidate table missing")
    require(not (RUN / "evaluation").exists(), "Evaluation already exists")
    data = SEReader()
    tokens = list(data.tokens)
    require(tokens == list(read_json(SOURCE_ROOT / DATA_READY)["aa_token_ids"]), "Tokenizer changed")
    wt, wt_sha, _ = load_wt()
    basis = read_json(STRUCT_BASIS)
    domains = basis["target"]["domains"]
    anchors = prepare_anchors(basis, wt, domains)
    candidates_table = pd.read_csv(RUN / "candidate_sequences.csv", keep_default_na=False)
    require(len(candidates_table) == 32 and candidates_table.candidate_id.is_unique,
            "Candidate table changed")
    for row in candidates_table.itertuples():
        sequence = row.sequence
        require(text_sha(sequence) == row.sequence_hash, "Candidate hash differs")
        editable = [int(value) - 1 for value in row.edit_positions_1based.split(";")]
        variants = verify_candidate(sequence, wt, editable)
        require(variant_text(variants) == row.substitutions, "Candidate substitutions differ")
        require(len(variants) == int(row.edit_count), "Candidate edit count differs")
    control, generated = formal_reference()
    require(text_sha(control.sequence) == control.sequence_hash == wt_sha, "WT control differs")
    for row in generated.itertuples():
        require(text_sha(row.sequence) == row.sequence_hash, "Previous candidate hash differs")
        variants = verify_candidate(row.sequence, wt)
        require(set(variants) == set(parse_variants(row.substitutions)),
                "Previous candidate substitutions differ")
        require(len(variants) == int(row.edit_count), "Previous edit count differs")
    previous_by_hash = {row.sequence_hash: row for row in generated.itertuples()}
    train_ids = set(read_json(SOURCE_ROOT / FORMAL_PLAN)["retrieval_database_ids"])

    control_frame = pd.DataFrame([{
        "candidate_id": "WT_control",
        "source_set": "WT_control",
        "model_group": "WT",
        "plan_type": "无",
        "edit_count": 0,
        "sequence": wt,
        "sequence_hash": wt_sha,
        "substitutions": "WT",
        "n_changes": 0,
        "group_id": "",
        "plan": "",
        "latent_from_prior": False,
        "latent_source": "不适用",
        "latent_sha256": "",
    }])
    new_frame = candidates_table.copy()
    new_frame["source_set"] = "new_candidates"
    previous_frame = pd.DataFrame([{
        "candidate_id": f"previous:{row.previous_candidate_id}",
        "source_set": "previous_WP5",
        "model_group": f"previous_{row.previous_model_group}",
        "plan_type": {"matched": "固定编辑位点", "adaptive": "自适应位点"}.get(
            row.previous_plan_type, row.previous_plan_type),
        "edit_count": int(row.edit_count),
        "sequence": row.sequence,
        "sequence_hash": row.sequence_hash,
        "substitutions": row.substitutions,
        "n_changes": int(row.edit_count),
        "group_id": "",
        "plan": "",
        "latent_from_prior": False,
        "latent_source": "不适用",
        "latent_sha256": "",
        "previous_candidate_id": row.previous_candidate_id,
        "previous_ESM2_CE": float(row.previous_ESM2_CE),
    } for row in generated.itertuples()])
    combined = pd.concat([control_frame, new_frame, previous_frame], ignore_index=True, sort=False)
    require(combined.sequence_hash.notna().sum() == len(combined), "Missing sequence hash")
    evaluation = RUN / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=False)
    encoder = FrozenHFMaskedEncoder(SOURCE_ROOT / "nad_models/esm2", device="cuda")
    require(encoder.tokenizer.convert_tokens_to_ids(list(AA)) == tokens, "Tokenizer identity differs")
    result, wt_quality = internal_metrics(
        SOURCE_ROOT, encoder, tokens, combined, wt,
        {"WT_sha256": wt_sha, "quality_blocks": QUALITY_BLOCKS}, evaluation)
    require(result.sequence_hash.map(lambda value: len(value) == 64).all(), "Metric hash malformed")
    require(abs(wt_quality - WT_CE_REFERENCE) < 0.01, "WT ESM2 CE diverged beyond tolerance")
    for old_name, new_name in (
        ("CDD_domains.csv", "CDD_domain_hits.csv"),
        ("internal_PLM_scores.csv", "ESM2_scores.csv"),
    ):
        old_path = evaluation / old_name
        require(old_path.exists(), f"Missing evaluation output: {old_name}")
        old_path.rename(evaluation / new_name)
    result = result.rename(columns={"PLM_blocked_CE": "ESM2_blocked_CE"})
    domains_table = pd.read_csv(evaluation / "CDD_domain_hits.csv")
    pairs = {name: pair_details(group.to_dict("records"))
             for name, group in domains_table.groupby("sequence_id")}
    require(pairs.get(wt_sha, {}).get("ordered_CDD_pair") is True, "CDD check excludes WT")

    previous_domains = pd.read_csv(SOURCE_ROOT / FORMAL_CDD)
    previous_esm2 = pd.read_csv(SOURCE_ROOT / FORMAL_CE).set_index("sequence_hash").rename(
        columns={"PLM_blocked_CE": "ESM2_blocked_CE"})
    formal_domain_columns = ["specificity_class", "pssm_id", "query_from", "query_to",
                             "evalue", "bitscore", "accession", "short_name", "incomplete"]
    cdd_repeat = []
    for sequence_hash in set(previous_domains.sequence_id):
        previous_rows = previous_domains[previous_domains.sequence_id == sequence_hash]
        current_rows = domains_table[domains_table.sequence_id == sequence_hash]
        sort_keys = ["pssm_id", "query_from", "query_to", "accession"]
        previous_sorted = previous_rows[formal_domain_columns].sort_values(sort_keys).reset_index(drop=True)
        current_sorted = current_rows[formal_domain_columns].sort_values(sort_keys).reset_index(drop=True)
        identical = (
            len(previous_sorted) == len(current_sorted)
            and previous_sorted[sort_keys].astype(str).equals(current_sorted[sort_keys].astype(str))
            and np.allclose(previous_sorted.evalue, current_sorted.evalue, rtol=0, atol=1e-12)
            and np.allclose(previous_sorted.bitscore, current_sorted.bitscore, rtol=0, atol=1e-6)
        )
        cdd_repeat.append({
            "sequence_hash": sequence_hash,
            "previous_rows": len(previous_rows),
            "current_rows": len(current_rows),
            "identical": bool(identical),
            "max_bitscore_delta": float((previous_sorted.bitscore - current_sorted.bitscore).abs().max())
            if identical else np.nan,
        })
    cdd_repeat = pd.DataFrame(cdd_repeat)
    cdd_repeat.to_csv(evaluation / "CDD_repeat_check.csv", index=False)

    candidate_rows = []
    substitution_level = []
    for row in result.itertuples():
        sequence = row.sequence
        sequence_hash = row.sequence_hash
        variants = sequence_variants(sequence, wt)
        if row.source_set != "WT_control":
            require(len(variants) == int(row.edit_count), "Candidate edit count differs")
        pair = pairs.get(sequence_hash, pair_details([]))
        require(bool(pair["ordered_CDD_pair"]) == bool(row.ordered_CDD_pair),
                "CDD pair merge differs")
        candidate = {
            "candidate_id": row.candidate_id,
            "source_set": row.source_set,
            "model_group": row.model_group,
            "plan_type": row.plan_type,
            "edit_count": int(row.edit_count),
            "sequence_hash": sequence_hash,
            "sequence": sequence,
            "length": len(sequence),
            "n_changes": len(variants),
            "substitutions": variant_text(variants),
            "ESM2_blocked_CE": float(row.ESM2_blocked_CE),
            "delta_CE_vs_WT": float(row.ESM2_blocked_CE - wt_quality),
            "ESM2_CE_below_WT": bool(float(row.ESM2_blocked_CE) <= wt_quality),
            "exact_sequence_new_family_2430": bool(row.exact_novel),
            "exact_sequence_new_training_1644": sequence_hash not in train_ids,
            "CDD_double_domain": bool(pair["ordered_CDD_pair"]),
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
            "previous_ESM2_CE": np.nan,
            "recomputed_minus_previous_CE": np.nan,
            "previous_candidate_id": "",
        }
        if row.source_set == "previous_WP5":
            previous_row = previous_by_hash[sequence_hash]
            candidate["previous_ESM2_CE"] = float(previous_row.previous_ESM2_CE)
            candidate["recomputed_minus_previous_CE"] = float(
                row.ESM2_blocked_CE - previous_row.previous_ESM2_CE)
            candidate["previous_candidate_id"] = previous_row.previous_candidate_id
            require(float(previous_row.previous_ESM2_CE) ==
                    float(previous_esm2.loc[sequence_hash, "ESM2_blocked_CE"]),
                    "Previous ESM2 tables disagree")
        for domain_name in ("NTERM", "NMNAT", "LINKER", "NUDIX", "CTERM"):
            candidate[f"domain_changes_{domain_name}"] = sum(
                1 for position, _, _ in variants if domain_of(position, domains) == domain_name)
        if row.source_set != "WT_control":
            rows = substitution_rows(candidate, anchors, domains)
            substitution_level.extend(rows)
            direct = [item for item in rows if item["direct_key_residue"]]
            within4 = [item for item in rows if not item["direct_key_residue"]
                       and item["within_4_positions"]]
            candidate["n_direct_key_residue_changes"] = len(direct)
            candidate["direct_key_residue_changes"] = ";".join(item["substitution"] for item in direct)
            candidate["direct_key_residue_domains"] = ";".join(item["domain"] for item in direct)
            candidate["n_changes_within_4_positions"] = len(within4)
            candidate["changes_within_4_positions"] = ";".join(item["substitution"] for item in within4)
            candidate["n_NMNAT_key_residue_changes"] = sum(item["domain"] == "NMNAT" for item in direct)
            candidate["n_NUDIX_key_residue_changes"] = sum(item["domain"] == "NUDIX" for item in direct)
            candidate["NMNAT_key_residue_changed"] = bool(candidate["n_NMNAT_key_residue_changes"])
            candidate["NUDIX_key_residue_changed"] = bool(candidate["n_NUDIX_key_residue_changes"])
            candidate["H236_key_residue_changed"] = bool(candidate["H236_changed"])
            candidate["no_direct_key_residue_change"] = not direct
            if direct:
                candidate["substitution_category"] = "直接改变关键残基"
            elif candidate["domain_changes_NMNAT"] or candidate["domain_changes_NUDIX"]:
                candidate["substitution_category"] = "只改变结构域内其他位点"
            else:
                candidate["substitution_category"] = "只改变结构域外位点"
        else:
            candidate.update({
                "n_direct_key_residue_changes": 0,
                "direct_key_residue_changes": "",
                "direct_key_residue_domains": "",
                "n_changes_within_4_positions": 0,
                "changes_within_4_positions": "",
                "n_NMNAT_key_residue_changes": 0,
                "n_NUDIX_key_residue_changes": 0,
                "NMNAT_key_residue_changed": False,
                "NUDIX_key_residue_changed": False,
                "H236_key_residue_changed": False,
                "no_direct_key_residue_change": True,
                "substitution_category": "对照",
            })
        candidate_rows.append(candidate)
    candidates = pd.DataFrame(candidate_rows)
    substitutions = pd.DataFrame(substitution_level)
    (RUN / "structure_analysis").mkdir(parents=True, exist_ok=False)

    model_group_rows = []
    for keys, group in candidates.groupby(
            ["source_set", "model_group", "plan_type", "edit_count"], dropna=False, sort=True):
        source_set, model_group, plan_type, edit_count = keys
        model_group_rows.append({
            "source_set": source_set,
            "model_group": model_group,
            "plan_type": plan_type,
            "edit_count": int(edit_count),
            "n_candidates": len(group),
            "n_unique_sequences": int(group.sequence_hash.nunique()),
            "n_new_family_2430": int(group.exact_sequence_new_family_2430.sum()),
            "n_new_training_1644": int(group.exact_sequence_new_training_1644.sum()),
            "n_CDD_double_domain": int(group.CDD_double_domain.sum()),
            "n_ESM2_below_WT": int(group.ESM2_CE_below_WT.sum()),
            "median_ESM2_CE": float(group.ESM2_blocked_CE.median()),
            "mean_delta_CE_vs_WT": float(group.delta_CE_vs_WT.mean()),
            "n_no_direct_key_residue_change": int(group.no_direct_key_residue_change.sum()),
            "n_NMNAT_key_residue_changed": int(group.NMNAT_key_residue_changed.sum()),
            "n_NUDIX_key_residue_changed": int(group.NUDIX_key_residue_changed.sum()),
            "n_T135_changed": int(group.T135_changed.sum()),
            "n_H236_changed": int(group.H236_changed.sum()),
            "n_changes_total": int(group.n_changes.sum()),
            "n_direct_key_residue_changes": int(group.n_direct_key_residue_changes.sum()),
            "n_changes_within_4_positions": int(group.n_changes_within_4_positions.sum()),
            "n_NMNAT_domain_changes": int(group.domain_changes_NMNAT.sum()),
            "n_NUDIX_domain_changes": int(group.domain_changes_NUDIX.sum()),
        })
    model_group_summary = pd.DataFrame(model_group_rows)

    domain_rows = []
    candidate_groups = {
        key: value for key, value in candidates.groupby(
            ["source_set", "model_group", "plan_type", "edit_count"], sort=True)
    }
    if len(substitutions):
        for keys, group in substitutions.groupby(
                ["source_set", "model_group", "plan_type", "edit_count", "domain"], sort=True):
            source_set, model_group, plan_type, edit_count, domain = keys
            total = len(candidate_groups[(source_set, model_group, plan_type, edit_count)])
            domain_rows.append({
                "source_set": source_set,
                "model_group": model_group,
                "plan_type": plan_type,
                "edit_count": int(edit_count),
                "domain": domain,
                "n_candidates": total,
                "n_candidates_with_change": int(group.candidate_id.nunique()),
                "n_changes": len(group),
                "n_direct_key_residue_changes": int(group.direct_key_residue.sum()),
                "n_changes_within_4_positions": int(
                    (group.within_4_positions & ~group.direct_key_residue).sum()),
                "n_unique_positions": int(group.position_1based.nunique()),
            })
    domain_summary = pd.DataFrame(domain_rows)

    key_residue_rows = []
    for anchor in anchors:
        record = {
            "domain": anchor["domain"],
            "wt_position": anchor["wt_position"],
            "wt_residue": anchor["wt_residue"],
            "identity_conserved_across_templates": anchor["identity_conserved_across_templates"],
            "variable_key_residue": anchor["variable_anchor"],
            "designated_template": anchor["designated_template"],
            "min_ligand_distance_A": anchor["min_ligand_distance_A"],
            "mapping_status": anchor["mapping_status"],
            "ligand_proximity_status": anchor["ligand_proximity_status"],
        }
        for prefix, source_set in (("new", "new_candidates"),
                                   ("previous", "previous_WP5")):
            candidates_set = candidates[candidates.source_set == source_set]
            direct = substitutions[(substitutions.source_set == source_set) &
                                   (substitutions.position_1based == anchor["wt_position"])]
            nearby = substitutions[(substitutions.source_set == source_set) &
                                   (substitutions.position_1based != anchor["wt_position"]) &
                                   (substitutions.nearest_key_residue_position == anchor["wt_position"]) &
                                   substitutions.within_4_positions]
            record[f"{prefix}_candidates_total"] = len(candidates_set)
            record[f"{prefix}_candidates_changed"] = int(direct.candidate_id.nunique())
            record[f"{prefix}_substitutions"] = ";".join(sorted(set(direct.substitution)))
            record[f"{prefix}_candidates_within_4_positions"] = int(nearby.candidate_id.nunique())
            record[f"{prefix}_within_4_substitutions"] = ";".join(sorted(set(nearby.substitution)))
        key_residue_rows.append(record)
    key_residue_summary = pd.DataFrame(key_residue_rows)

    new_candidates = candidates[candidates.source_set == "new_candidates"]
    previous = candidates[candidates.source_set == "previous_WP5"]
    direct_candidates = candidates[candidates.n_direct_key_residue_changes > 0].copy()
    direct_candidates = direct_candidates[[
        "candidate_id", "source_set", "model_group", "plan_type", "edit_count",
        "direct_key_residue_changes", "direct_key_residue_domains", "n_direct_key_residue_changes",
        "n_NMNAT_key_residue_changes", "n_NUDIX_key_residue_changes",
        "H236_key_residue_changed", "T135_changed", "ESM2_blocked_CE",
        "exact_sequence_new_family_2430", "CDD_double_domain"]]

    previous_ce_join = candidates[candidates.source_set == "previous_WP5"].dropna(
        subset=["previous_ESM2_CE"])
    ce_reproduction = {
        "n": len(previous_ce_join),
        "max_abs_delta": float(previous_ce_join.recomputed_minus_previous_CE.abs().max()),
        "median_abs_delta": float(previous_ce_join.recomputed_minus_previous_CE.abs().median()),
        "n_within_1e_4": int((previous_ce_join.recomputed_minus_previous_CE.abs() <= 1e-4).sum()),
        "n_within_1e_3": int((previous_ce_join.recomputed_minus_previous_CE.abs() <= 1e-3).sum()),
        "note": "本次评价的批次组成与旧运行不同；这里只检查同一序列的 ESM2 分数能否复现。",
    }
    results_summary = {
        "status": "EVALUATION_DONE",
        "written_at_utc": utc_now(),
        "unique_sequences": int(candidates.sequence_hash.nunique()),
        "candidates": len(candidates),
        "WT_ESM2_blocked_CE": float(wt_quality),
        "WT_ESM2_blocked_CE_reference": WT_CE_REFERENCE,
        "WT_ESM2_blocked_CE_delta": float(wt_quality - WT_CE_REFERENCE),
        "previous_ESM2_reproduction": ce_reproduction,
        "previous_CDD_rows_identical": bool(cdd_repeat.identical.all()),
        "previous_CDD_rows_compared": len(cdd_repeat),
        "WT_CDD_double_domain": {
            "CDD_double_domain": bool(pairs[wt_sha]["ordered_CDD_pair"]),
            "NMNAT_accession": pairs[wt_sha]["CDD_NMNAT_accession"],
            "NMNAT_range": pairs[wt_sha]["CDD_NMNAT_range"],
            "NMNAT_evalue": pairs[wt_sha]["CDD_NMNAT_evalue"],
            "NUDIX_accession": pairs[wt_sha]["CDD_NUDIX_accession"],
            "NUDIX_range": pairs[wt_sha]["CDD_NUDIX_range"],
            "NUDIX_evalue": pairs[wt_sha]["CDD_NUDIX_evalue"],
        },
        "test_sequence_reads": 0,
        "training_runs": 0,
        "folding_runs": 0,
        "docking_runs": 0,
        "md_runs": 0,
        "wet_lab": False,
    }
    write_json(evaluation / "results_summary.json", results_summary)

    def counts(frame):
        direct = frame[frame.n_direct_key_residue_changes > 0]
        return {
            "n": len(frame),
            "unique": int(frame.sequence_hash.nunique()),
            "duplicates": int(len(frame) - frame.sequence_hash.nunique()),
            "unique_with_direct_key_residue_change": int(direct.sequence_hash.nunique()),
            "unique_without_direct_key_residue_change": int(
                frame[frame.no_direct_key_residue_change].sequence_hash.nunique()),
            "unique_NMNAT_key_residue_changed": int(
                frame[frame.NMNAT_key_residue_changed].sequence_hash.nunique()),
            "unique_NUDIX_key_residue_changed": int(
                frame[frame.NUDIX_key_residue_changed].sequence_hash.nunique()),
            "unique_T135_changed": int(frame[frame.T135_changed].sequence_hash.nunique()),
            "unique_H236_changed": int(frame[frame.H236_changed].sequence_hash.nunique()),
            "exact_sequence_new_family_2430": int(frame.exact_sequence_new_family_2430.sum()),
            "exact_sequence_new_training_1644": int(frame.exact_sequence_new_training_1644.sum()),
            "CDD_double_domain": int(frame.CDD_double_domain.sum()),
            "ESM2_below_WT": int(frame.ESM2_CE_below_WT.sum()),
            "no_direct_key_residue_change": int(frame.no_direct_key_residue_change.sum()),
            "NMNAT_key_residue_changed": int(frame.NMNAT_key_residue_changed.sum()),
            "NUDIX_key_residue_changed": int(frame.NUDIX_key_residue_changed.sum()),
            "T135_changed": int(frame.T135_changed.sum()),
            "H236_changed": int(frame.H236_changed.sum()),
            "direct_key_residue_changes": int(frame.n_direct_key_residue_changes.sum()),
            "changes_within_4_positions": int(frame.n_changes_within_4_positions.sum()),
            "NMNAT_domain_changes": int(frame.domain_changes_NMNAT.sum()),
            "NUDIX_domain_changes": int(frame.domain_changes_NUDIX.sum()),
        }

    by_model_group = {
        name: counts(group)
        for name, group in new_candidates.groupby("model_group")
    }
    previous_by_model_group = {
        name: counts(group)
        for name, group in previous.groupby("model_group")
    }
    interpretation = {
        "record": "NatV 候选序列的结构注释",
        "written_at_utc": utc_now(),
        "script_sha256": sha(Path(__file__)),
        "scope": {
            "new_candidates": "本次用两组沿用的 H2 模型参数生成的 32 条全长候选",
            "previous_WP5": "旧 WP5 运行留下的 24 条候选；本次只做复核，没有重新生成",
            "WT_control": "KVP40 野生型",
        },
        "inputs": {
            "structure_basis": str(STRUCT_BASIS.relative_to(NATV_ROOT)),
            "structure_basis_sha256": sha(STRUCT_BASIS),
            "structure_basis_used_in_training": False,
            "h2_settings_sha256": sha(SOURCE_ROOT / H2_CONTRACT),
            "model_parameters": [
                {"name": a["label"], "sha256": a["sha256"], "architecture": a["architecture"]}
                for a in MODEL_GROUPS
            ],
        },
        "parent": {
            "name": basis["target"]["name"],
            "sequence_sha256": wt_sha,
            "length": len(wt),
            "domains": domains,
            "key_residue_count": len(anchors),
            "key_residue_mapping": basis["anchor_summary"]["mapping_status"],
            "key_residue_ligand_distance_check": basis["anchor_summary"]["designated_ligand_proximity_status"],
            "variable_key_residue": basis["anchor_summary"]["variable_anchor"],
        },
        "results": {
            "new_candidates": counts(new_candidates),
            "new_candidates_by_model_group": by_model_group,
            "previous_WP5": counts(previous),
            "previous_WP5_by_model_group": previous_by_model_group,
            "WT_control": counts(candidates[candidates.source_set == "WT_control"]),
            "direct_key_residue_candidates": direct_candidates.to_dict(orient="records"),
        },
        "interpretation": {
            "NMNAT_environment": (
                "H14、H17、T135、R138 的直接替换落在 2QJO（NAD⁺ 产物结合状态）"
                "已核对的 ATP 结合基序或催化邻近区域。其他 NMNAT 结构域内的替换只按"
                "结构域归类。这些是结构对应关系，不能推出 kcat、底物结合、净生成或"
                "任何催化优势。"
            ),
            "NUDIX_hydrolysis": (
                "G232、H236、R247、E248、E251、E252、D307、D308 的直接替换落在"
                "2QJT（AMP + Mn²⁺ 状态）已核对的 Nudix 基序或金属-水网络附近。"
                "H236 在模板间不是序列保守位点，改变它只能说明位置上的扰动，不能"
                "当作保守关键残基丢失。任何替换会提高还是降低水解，目前未知；本次"
                "不给出风险方向。"
            ),
            "T135": "T135 的任何替换都会直接改变一个 NMNAT 关键残基；T135A 被本次限制禁止。",
            "within_4_positions": "“相距不超过 4 个序列位点”只表示序列上的邻近，不是三维距离。",
        },
        "limitations": [
            "两组 H2 模型做的是以 WT 为起点、固定改动 4 或 8 个位点的局部编辑，不是从头生成全长序列。",
            "两组模型使用同一套训练设置，但结构和参数量不同；组间差异只是描述性计数，不能当作潜在变量机制的因果检验。",
            "分 8 段遮蔽计算的 ESM2 交叉熵与 ESM2 自身训练分布相关，只能作为序列相容性的参考，不是活性。",
            "CDD 同时命中 NMNAT 和 NUDIX 两个结构域只说明家族相容性，不是催化活性。",
            "本次没有做折叠、对接、分子动力学、湿实验或净 NAD⁺ 生成测量。",
            "结构注释依据来自此前的序列-结构核对，不是新的实验结构；它没有进入训练、同源检索或候选挑选。",
            "单一随机种子、有限条数和单一亲本序列不足以支持群体水平或方法优劣的结论。",
            "旧 WP5 候选是此前运行挑出的，本次只是复核，不是独立的新样本。",
        ],
        "test_sequence_reads": 0,
        "training_runs": 0,
        "folding_runs": 0,
        "wet_lab": False,
    }
    write_json(RUN / "structure_analysis/interpretation.json", interpretation)
    write_report(candidates, substitutions, model_group_summary, domain_summary,
                 key_residue_summary, direct_candidates, results_summary,
                 interpretation, cdd_repeat)
    status = read_json(RUN / "run_status.json")
    status.update({
        "status": "RUN_DONE",
        "evaluation": results_summary,
        "structure_interpretation": "structure_analysis/interpretation.json",
        "report": "REPORT.md",
    })
    write_json(RUN / "run_status.json", status)
    print(json.dumps(results_summary, indent=2))


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
                text = "是" if value else "否"
            else:
                text = str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(candidates, substitutions, model_group_summary, domain_summary,
                 key_residue_summary, direct_candidates, results, interpretation,
                 cdd_repeat):
    new = candidates[candidates.source_set == "new_candidates"]
    previous = candidates[candidates.source_set == "previous_WP5"]
    control = candidates[candidates.source_set == "WT_control"].iloc[0]
    new_direct = new[new.n_direct_key_residue_changes > 0]
    previous_direct = previous[previous.n_direct_key_residue_changes > 0]
    direct_unique = int(pd.concat([new_direct, previous_direct]).sequence_hash.nunique())
    duplicate_4 = int(len(new[new.edit_count == 4]) - new[new.edit_count == 4].sequence_hash.nunique())
    duplicate_8 = int(len(new[new.edit_count == 8]) - new[new.edit_count == 8].sequence_hash.nunique())
    model_columns = [
        "source_set", "model_group", "plan_type", "edit_count", "n_candidates",
        "n_unique_sequences", "n_new_family_2430", "n_CDD_double_domain",
        "n_ESM2_below_WT", "median_ESM2_CE", "n_no_direct_key_residue_change",
        "n_NMNAT_key_residue_changed", "n_NUDIX_key_residue_changed",
        "n_T135_changed", "n_H236_changed",
    ]
    domain_columns = [
        "source_set", "model_group", "edit_count", "domain", "n_candidates",
        "n_candidates_with_change", "n_changes", "n_direct_key_residue_changes",
        "n_changes_within_4_positions",
    ]
    residue_columns = [
        "domain", "wt_position", "wt_residue", "identity_conserved_across_templates",
        "variable_key_residue", "new_candidates_changed", "new_substitutions",
        "previous_candidates_changed", "previous_substitutions",
    ]
    direct_columns = [
        "candidate_id", "source_set", "model_group", "edit_count",
        "direct_key_residue_changes", "n_NMNAT_key_residue_changes",
        "n_NUDIX_key_residue_changes", "H236_key_residue_changed", "T135_changed",
        "ESM2_blocked_CE", "CDD_double_domain",
    ]
    model_lines = "\n".join(
        f"- {item['name']}：SHA256 `{item['sha256']}`，结构：{item['architecture']}"
        for item in interpretation["inputs"]["model_parameters"]
    )
    def position_changes(source_set, position):
        subset = substitutions[(substitutions.source_set == source_set) &
                               (substitutions.position_1based == position)]
        return ";".join(sorted(set(subset.substitution))) or "无"

    new_t135 = position_changes("new_candidates", 135)
    previous_t135 = position_changes("previous_WP5", 135)
    new_h236 = position_changes("new_candidates", H236_POSITION)
    previous_h236 = position_changes("previous_WP5", H236_POSITION)
    report = f"""# NatV 候选序列生成与结构注释

本次运行沿用 nad_qvae 中已经训练好的两组 H2 模型参数，生成 32 条全长 KVP40 NatV 候选序列，并把每处替换标注到结构域和此前核对过的 12 个关键残基上。模型做的是以野生型为起点、固定改动 4 或 8 个位点的局部编辑，不是从头生成全长序列。本次没有测量任何酶的催化活性。

## 运行范围

{model_lines}

- 两组模型共同使用 H2 上游设置、ESM2 词表、1,644 条训练序列和原有划分。
- 编辑方案种子 `{STREAM_SEED}`，每种编辑数 8 组方案，编辑数 {EDIT_BUDGETS}，每组 4 条序列，共 7 步掩码编辑，温度 {TEMPERATURE}，禁止 T135A。
- 使用连续潜在变量的组在每个四序列组开始时抽取一次先验，七步共用；不使用潜在变量的组没有这一分支。
- 共 32 条候选。没有训练、折叠、对接、分子动力学、远程下载、湿实验，也没有读取测试序列内容。

## 输入与来源

- H2 上游设置文件 SHA256：`{interpretation['inputs']['h2_settings_sha256']}`。
- 结构注释依据：`{interpretation['inputs']['structure_basis']}`，SHA256 `{interpretation['inputs']['structure_basis_sha256']}`。它只用于注释，没有进入训练、同源检索或候选挑选。
- 本次读取的文件和校验值见 `input_files.json`、`input_checks.json`、`sampling_plan_checksums.json`。
- 旧 WP5 复核：{results['previous_ESM2_reproduction']['n']} 条候选；ESM2 交叉熵最大绝对差 `{results['previous_ESM2_reproduction']['max_abs_delta']:.6g}`；CDD 结果行完全一致：`{results['previous_CDD_rows_identical']}`。

## 生成结果是否有效

- 32/32 条候选为 341 aa，恰好 4 或 8 处替换。
- 最终替换与生成方案中的位点一致；T135A 违规 0。
- 不同序列数：{new.sequence_hash.nunique()}/32。两组模型在编辑数为 4 的方案中有 {duplicate_4} 对序列相同，编辑数为 8 的方案中有 {duplicate_8} 对相同；这些重复保留在所有按候选计数的分母中。
- 与 2,430 条家族角色表逐字符比较，{int(new.exact_sequence_new_family_2430.sum())}/32 为新序列；与 1,644 条训练序列比较，{int(new.exact_sequence_new_training_1644.sum())}/32 为新序列。
- CDD 同时命中 NMNAT 和 NUDIX 两个结构域：{int(new.CDD_double_domain.sum())}/32。
- WT 分 8 段遮蔽的 ESM2 交叉熵：{control.ESM2_blocked_CE:.9f}（参考 {WT_CE_REFERENCE:.9f}；差 {results['WT_ESM2_blocked_CE_delta']:.6g}）。

## 各组结果

{md_table(model_group_summary[model_group_summary.source_set != 'WT_control'], model_columns)}

本次两组模型共用训练设置，但结构和参数量不同。这些是有限样本的描述性计数，不是方法优劣检验。ESM2 交叉熵与 ESM2 自身训练分布相关，不是活性证据。表中的 `previous_WP5` 行只用于复核旧候选，不是本次生成。

## 结构域上的替换

{md_table(domain_summary, domain_columns)}

## 已核对的 12 个关键残基

{md_table(key_residue_summary, residue_columns)}

直接改变关键残基的候选共 {len(direct_candidates)} 条（本次 {len(new_direct)} 条 + 旧 WP5 {len(previous_direct)} 条），但只有 {direct_unique} 条不同序列，因为有些候选在两组模型之间完全相同。完整列表见 `structure_analysis/interpretation.json` 的 `direct_key_residue_candidates`。

{md_table(direct_candidates, direct_columns)}

### NMNAT 催化环境

H14、H17、T135、R138 的直接替换落在 2QJO（NAD⁺ 产物结合状态）已核对的 ATP 结合基序或催化邻近区域；其他 NMNAT 结构域内的替换只按结构域归类。本次直接改变 NMNAT 关键残基的候选：{int(new.NMNAT_key_residue_changed.sum())}/32 条（{new[new.NMNAT_key_residue_changed].sequence_hash.nunique()} 条不同序列）；旧 WP5：{int(previous.NMNAT_key_residue_changed.sum())}/24 条（{previous[previous.NMNAT_key_residue_changed].sequence_hash.nunique()} 条不同序列）。T135 改变：本次 {int(new.T135_changed.sum())}/32（{new_t135}）；旧 WP5 {int(previous.T135_changed.sum())}/24（{previous_t135}）。T135A 违规 0。以上对应关系不能推出 kcat、底物结合、净生成或催化优势。

### Nudix 水解

G232、H236、R247、E248、E251、E252、D307、D308 的直接替换落在 2QJT（AMP + Mn²⁺ 状态）已核对的 Nudix 基序或金属-水网络附近。H236 在模板间不是序列保守位点，改变它只能说明位置上的扰动，不能当作保守关键残基丢失。本次直接改变 NUDIX 关键残基的候选：{int(new.NUDIX_key_residue_changed.sum())}/32 条（{new[new.NUDIX_key_residue_changed].sequence_hash.nunique()} 条不同序列）；旧 WP5：{int(previous.NUDIX_key_residue_changed.sum())}/24 条（{previous[previous.NUDIX_key_residue_changed].sequence_hash.nunique()} 条不同序列）。H236 改变：本次 {int(new.H236_changed.sum())}/32（{new_h236}）；旧 WP5 {int(previous.H236_changed.sum())}/24（{previous_h236}）。任何替换会提高还是降低水解，目前未知；本次不给出风险方向。

## 这些结果不能说明什么

- 模型只做固定 4/8 位点的局部编辑，不是从头生成全长序列。
- 结构注释来自此前的序列-结构核对，不是新的实验结构；“相距不超过 4 个序列位点”只表示序列上的邻近，不是三维距离。
- 没有折叠、对接、分子动力学、湿实验或净 NAD⁺ 生成测量；ESM2 和 CDD 只是序列与家族的相容性参考。
- 旧 WP5 的 24 条候选是此前运行挑出的，不是独立的新样本。
- 按候选计数包含两组模型之间的重复序列；不同序列数已在上面单列。
- 单一随机种子、有限条数和单一亲本序列不足以支持群体水平或方法优劣的结论。

## 决定

结构对应关系有助于说明替换落在哪里，但没有建立催化优势。正式训练序列集和原有划分不变。结构注释依据只记录在本次运行的输入文件中，没有进入正式训练输入。就当前目标而言，数据扩充到此为止。

## 如何重跑

```bash
export PATH=/home/yhshy/miniconda3/envs/nad_data/bin:$PATH
export NATV_SOURCE_ROOT=/home/yhshy/git_files/nad_qvae
python experiments/natv_candidate_sequences/run_candidates.py --stage preflight
python experiments/natv_candidate_sequences/run_candidates.py --stage sample
python experiments/natv_candidate_sequences/run_candidates.py --stage evaluate
```

## 文件说明

- `sampling_plan.json`、`sampling_plan_checksums.json`、`input_checks.json`、`input_files.json`、`run_status.json`
- `candidate_sequences.csv`、`sampling_steps.jsonl.gz`、`homolog_search_records.json.gz`、`homolog_search_summary.csv`
- `evaluation/ESM2_scores.csv`、`evaluation/CDD_domain_hits.csv`、`evaluation/CDD_repeat_check.csv`、`evaluation/results_summary.json`
- `structure_analysis/interpretation.json`
- 原始 MMseqs 输出在 `homolog_search/` 下，仅本地保留，不提交。
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
