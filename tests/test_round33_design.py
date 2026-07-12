from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
REGISTRY31 = RESEARCH / "consumed-periods-through-round-031.json"
REGISTRY32 = RESEARCH / "consumed-periods-through-round-032.json"
FAILURE = RESEARCH / "round-032-failure-analysis.json"
DESIGN32 = RESEARCH / "round-032-shared-action-value-viability-design.json"
DESIGN33 = RESEARCH / "round-033-selective-action-design.json"
RISK_SOURCE = RESEARCH / "round-031-frozen-chronological-confirmation-design.json"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _dates(first: str, last: str) -> set[str]:
    start = date.fromisoformat(first)
    end = date.fromisoformat(last)
    assert start <= end
    return {
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    }


def test_round32_registry_and_failure_analysis_are_hash_bound() -> None:
    previous = _read(REGISTRY31)
    registry = _read(REGISTRY32)
    failure = _read(FAILURE)

    assert registry["registry_sha256"] == _canonical_sha256(
        registry, "registry_sha256"
    )
    assert registry["records"][:-1] == previous["records"]
    assert registry["records"][-1] == {
        "round": 32,
        "status": "consumed",
        "outcome": "rejected",
        "design_sha256": (
            "5c8b2ac3618cdeec16dd4a8d21e5792cc4ea4bc26a645bf9ec00bd3997d3ca68"
        ),
        "report_sha256": (
            "e1dee0db73d2729aecfb09ea2f9c8abbd7f41a5985dd6dbd87de4e1575710552"
        ),
        "diagnostic_sha256": (
            "8c2d1cbaf13c2553d24466b27a29b2bb79c1afa2e011e75c75279bd376f36ab0"
        ),
        "windows": [{"start_date": "2023-05-16", "end_date": "2023-07-06"}],
    }
    assert failure["analysis_sha256"] == _canonical_sha256(
        failure, "analysis_sha256"
    )
    observations = failure["observations"]
    assert observations["opportunity_probability_auc"] > 0.65
    assert observations["conditional_direction_auc_on_profitable_opportunities"] < 0.55
    assert observations["selected_top_500_mean_stress_net_bps"] < 0.0
    assert failure["next_hypothesis"]["single_variant_only"] is True
    assert failure["next_hypothesis"]["hyperparameter_search_permitted"] is False


def test_round33_design_is_one_variant_consumed_only_and_safety_equivalent() -> None:
    design = _read(DESIGN33)
    predecessor = _read(DESIGN32)
    registry = _read(REGISTRY32)

    assert design["design_sha256"] == _canonical_sha256(design, "design_sha256")
    assert design["round"] == 33
    assert design["design_revision"] == 1
    governance = design["governance"]
    assert governance["variant_budget"] == 1
    assert governance["hyperparameter_search_permitted"] is False
    assert governance["all_target_dates_already_consumed"] is True
    assert governance["consumed_period_registry_file_sha256"] == hashlib.sha256(
        REGISTRY32.read_bytes()
    ).hexdigest()
    assert governance["consumed_period_registry_canonical_sha256"] == registry[
        "registry_sha256"
    ]
    risk_source = design["risk_profiles_source"]
    assert risk_source["file_sha256"] == hashlib.sha256(
        RISK_SOURCE.read_bytes()
    ).hexdigest()
    assert risk_source["modification_permitted"] is False
    assert design["execution"] == predecessor["execution"]
    assert design["barrier_targets"] == predecessor["barrier_targets"]
    assert design["event_sampler"] == predecessor["event_sampler"]
    assert design["selection"] == predecessor["selection"]
    assert design["runtime_resources"] == predecessor["runtime_resources"]
    assert design["model"]["lightgbm"] == predecessor["model"]["lightgbm"]
    assert tuple(design["model"]["seeds"]) == tuple(predecessor["model"]["seeds"])

    consumed: set[str] = set()
    for record in registry["records"]:
        for window in record["windows"]:
            consumed.update(_dates(window["start_date"], window["end_date"]))
    roles = design["data"]["roles"]
    evaluated = set().union(
        *(
            _dates(roles[name]["start"], roles[name]["end"])
            for name in ("train", "early_stop", "calibration", "policy", "development")
        ),
        _dates(
            roles["distant_confirmation"]["start"],
            roles["distant_confirmation"]["end"],
        ),
    )
    assert evaluated <= consumed
    forbidden = set().union(
        *(
            _dates(window["start"], window["end"])
            for window in design["data"]["forbidden_target_windows"]
        )
    )
    assert not evaluated & forbidden
    assert all(value is False for value in design["claims"].values())
    assert design["acceptance_gates"]["economic"]["leverage_permitted"] is False


def test_round33_factorization_is_symmetric_selective_and_nontrivial() -> None:
    design = _read(DESIGN33)
    model = design["model"]
    factorization = model["probability_factorization"]
    symmetry = model["symmetry"]
    confidence = design["conditional_direction_confidence"]

    assert model["action_sides"] == ["long", "short", "abstain"]
    assert model["labels"]["conditional_direction_population"] == (
        "opportunity_rows_only"
    )
    assert model["labels"]["neither_profitable_action"] == "abstain"
    assert factorization["probabilities_sum_to_one"] is True
    assert symmetry["opportunity_prediction"] == (
        "mean_of_raw_and_mirrored_probability"
    )
    assert symmetry["direction_prediction"] == (
        "antisymmetric_logit_half_difference"
    )
    assert "positive_temperature_only" in symmetry[
        "direction_probability_calibration"
    ]
    assert confidence["conservative"] > confidence["regular"] > confidence["aggressive"]
    architecture = design["acceptance_gates"]["calibration_architecture"]
    failure = _read(FAILURE)["observations"]
    assert architecture["minimum_opportunity_auc"] <= failure[
        "opportunity_probability_auc"
    ]
    assert architecture["minimum_conditional_direction_auc"] > failure[
        "conditional_direction_auc_on_profitable_opportunities"
    ]
    assert architecture["minimum_selected_top_100_mean_stress_net_bps"] == 0.0
    assert architecture["minimum_selected_top_500_mean_stress_net_bps"] == 0.0
