"""Check actual scientific contract semantics; tests are not empirical validation."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def test_regions_contract_is_a_pilot():
    c = json.loads((CONTRACTS / "regions.json").read_text(encoding="utf-8"))
    assert c["target"]["length"] == 341
    assert c["boundary_conventions"]["cdd_derived"]["status"] == "adopted"
    assert set(c["mask_rules"]) >= {"A", "B", "C"}
    assert c["mask_rules"]["common"]["substitutions_only"] is True
    assert c["scope"] == "WT_relative_substitution_pilot_not_universal_NatV_biology"


def test_objective_distinguishes_synthesis_and_net_accumulation():
    yaml = pytest.importorskip("yaml")
    c = yaml.safe_load((CONTRACTS / "objective.yaml").read_text(encoding="utf-8"))
    assert c["scope"] == "in_vitro_purified_enzyme"
    assert c["application_endpoint"]["experiment_only"] is True
    assert set(c["design_hypotheses"]) >= {"A", "B", "C"}
    assert "kcat_if_identifiable" in c["separate_experimental_endpoints"]["synthesis"]
    assert "NAD_hydrolysis_rate" in c["separate_experimental_endpoints"]["degradation"]
    assert c["current_generation_research"]["task"] == "WT_conditioned_conservative_editing"
    assert c["historical_substitution_pilot"]["substitutions_only"] is True


def test_unmeasured_evidence_does_not_pass():
    yaml = pytest.importorskip("yaml")
    c = yaml.safe_load((CONTRACTS / "evidence_ladder.yaml").read_text(encoding="utf-8"))
    assert "UNKNOWN" in c["uncertainty"]["allowed_statuses"]
    assert "FAIL" in c["uncertainty"]["allowed_statuses"]
    assert "PASS" in c["uncertainty"]["allowed_statuses"]
    assert "G" not in c["evaluation"]  # not an opaque mandatory ladder
    assert "future_wet_lab_controls" in c["comparator_policy"]


def test_discovery_population_is_not_declared_as_training():
    manifest = json.loads((ROOT / "inputs/manifest.json").read_text(encoding="utf-8"))
    population = next(x for x in manifest["entries"] if x["name"] == "family_population")
    assert population["expected_records"] == 2430
    assert population["training_admission"] == "discovery_only_not_core_train"


def test_wt_sequence_matches_regions_contract():
    from natvqvae.sequence import WT_SHA256, wt_sequence

    c = json.loads((CONTRACTS / "regions.json").read_text(encoding="utf-8"))
    assert c["target"]["sequence_sha256"] == WT_SHA256
    assert c["target"]["length"] == len(wt_sequence())
