import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = REPO_ROOT / "contracts"


def test_regions_contract_loads():
    contract = json.loads((CONTRACTS / "regions_v1.json").read_text(encoding="utf-8"))
    assert contract["target"]["length"] == 341
    assert contract["boundary_conventions"]["cdd_derived"]["status"] == "adopted"
    assert contract["boundary_conventions"]["working_hypothesis"]["status"] == "archived_not_used"
    assert set(contract["mask_rules"]) >= {"A", "B", "C"}
    assert contract["mask_rules"]["common"]["substitutions_only"] is True


def test_objective_contract():
    yaml = pytest.importorskip("yaml")
    contract = yaml.safe_load((CONTRACTS / "objective_v1.yaml").read_text(encoding="utf-8"))
    assert contract["scope"] == "in_vitro_purified_enzyme"
    assert set(contract["design_classes"]) == {"A", "B", "C"}
    assert any("no_domain_deletion" in c for c in contract["hard_constraints"])
    assert contract["non_goals"]
    assert contract["future_controls"]["status"] == "not_yet_measured"


def test_evidence_ladder_contract():
    yaml = pytest.importorskip("yaml")
    contract = yaml.safe_load((CONTRACTS / "evidence_ladder_v1.yaml").read_text(encoding="utf-8"))
    assert contract["no_composite_score"] is True
    gate_ids = [g["id"] for g in contract["gates"]]
    assert gate_ids[0] == "G1_sequence"
    assert "G5_descriptive" in gate_ids
    assert contract["thresholds"]["status"] == "to_be_frozen_before_first_screening_run"
    controls = contract["controls_required_before_any_candidate_panel"]["required"]
    assert any("nudix_catalytic_dead" in c for c in controls)
    assert any("deleted" in c for c in controls)
