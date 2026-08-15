from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import simple_ai_trading.polymarket_round28_sealed as sealed_module
from simple_ai_trading.polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
)
from simple_ai_trading.polymarket_round28_economics import (
    POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round28_sealed import (
    POLYMARKET_ROUND28_SEALED_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION,
    build_round28_sealed_terminal_result,
    validate_round28_sealed_economic_report,
    validate_round28_sealed_prediction_result,
    validate_round28_sealed_terminal_result,
)
from tools.run_polymarket_round28_sealed import _parser


_HASH = "a" * 64
ROOT = Path(__file__).resolve().parents[1]
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
    "orders_submitted": False,
}


@pytest.fixture(autouse=True)
def _validated_pair_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sealed_module._selection,  # noqa: SLF001
        "_validated_pair_report",
        lambda value: dict(value),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _prediction() -> dict[str, object]:
    paired: dict[str, object] = {
        "model_family": "l2_offset_logistic",
        "matched_ablation": {
            "role": "sealed",
            "condition_count": 90,
            "row_count": 900,
            "base_model": {"model_sha256": "1" * 64},
            "augmented_model": {"model_sha256": "2" * 64},
        },
        "gate_checks": {"strict_probability_uplift": True},
        "probability_gate_passed": True,
    }
    paired["pair_report_sha256"] = _canonical_sha256(paired)
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION,
        "round27_model_contract_sha256": _HASH,
        "round28_preregistration_sha256": "b" * 64,
        "selection_input_manifest_sha256": "c" * 64,
        "selection_claim_sha256": "d" * 64,
        "selection_economic_report_sha256": "e" * 64,
        "source_binding_sha256": "f" * 64,
        "selected_model_family": "l2_offset_logistic",
        "base_model_sha256": "1" * 64,
        "augmented_model_sha256": "2" * 64,
        "condition_count": 90,
        "row_count": 900,
        "condition_population_sha256": "3" * 64,
        "paired_probability_report": paired,
        "prediction_uplift_gate_passed": True,
        "sealed_partition_accessed": True,
        "models_refit": False,
        "hyperparameters_retuned": False,
        "thresholds_changed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return body


def _nested_economics(*, model_sha256: str) -> dict[str, object]:
    scenarios: list[dict[str, object]] = []
    for delay in POLYMARKET_ROUND27_FIXED_DELAYS_MS:
        scenario: dict[str, object] = {
            "delay_ms": delay,
            "gate_checks": {"strict_economic_gate": True},
            "scenario_edge_gate_passed": True,
        }
        scenario["scenario_sha256"] = _canonical_sha256(scenario)
        scenarios.append(scenario)
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "sealed",
        "model_sha256": model_sha256,
        "source_audit_sha256": "f" * 64,
        "resolution_evidence_sha256": "4" * 64,
        "config": {"fixture": True},
        "economic_edge_gate_passed": True,
        "scenarios": scenarios,
        "edge_claim": False,
        "profitability_claim": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def _economics(prediction: dict[str, object]) -> dict[str, object]:
    base = _nested_economics(model_sha256="1" * 64)
    augmented = _nested_economics(model_sha256="2" * 64)
    paired_scenarios: list[dict[str, object]] = []
    for base_scenario, augmented_scenario in zip(
        base["scenarios"],
        augmented["scenarios"],
        strict=True,
    ):
        paired: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
            "delay_ms": base_scenario["delay_ms"],
            "base_scenario_sha256": base_scenario["scenario_sha256"],
            "augmented_scenario_sha256": augmented_scenario["scenario_sha256"],
            "gate_checks": {"strict_after_cost_uplift": True},
            "scenario_uplift_gate_passed": True,
        }
        paired["paired_scenario_sha256"] = _canonical_sha256(paired)
        paired_scenarios.append(paired)
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SEALED_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "sealed",
        "selection_claim_sha256": prediction["selection_claim_sha256"],
        "sealed_prediction_result_sha256": prediction["result_sha256"],
        "source_binding_sha256": prediction["source_binding_sha256"],
        "resolution_evidence_sha256": "4" * 64,
        "selected_model_family": prediction["selected_model_family"],
        "base_model_sha256": prediction["base_model_sha256"],
        "augmented_model_sha256": prediction["augmented_model_sha256"],
        "condition_population_sha256": prediction[
            "condition_population_sha256"
        ],
        "base_economic_report": base,
        "augmented_economic_report": augmented,
        "paired_scenarios": paired_scenarios,
        "economic_uplift_gate_passed": True,
        "sealed_partition_accessed": True,
        "models_refit": False,
        "hyperparameters_retuned": False,
        "thresholds_changed": False,
        "economic_metrics_computed": True,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def test_round28_sealed_operator_has_no_model_or_compute_selection_surface() -> None:
    destinations = {
        action.dest for action in _parser()._actions if action.dest != "help"
    }

    assert {
        "round27_feature_store",
        "round28_overlay_store",
        "round27_target_store",
        "sealed_source_database",
        "selection_input_manifest",
        "selection_claim",
        "selection_economic_report",
        "sealed_prediction_result",
        "sealed_economic_report",
        "terminal_result",
    } <= destinations
    assert "compute_backend" not in destinations
    assert all("model" not in destination for destination in destinations)


def test_round28_sealed_terminal_is_hash_bound_and_never_grants_authority() -> None:
    prediction = validate_round28_sealed_prediction_result(_prediction())
    economics = validate_round28_sealed_economic_report(_economics(prediction))

    terminal = build_round28_sealed_terminal_result(
        sealed_prediction_result=prediction,
        sealed_economic_report=economics,
    )

    assert terminal["observed_after_cost_bbo_uplift_gate_passed"] is True
    assert terminal["models_refit"] is False
    assert terminal["edge_claim"] is False
    assert terminal["profitability_claim"] is False
    assert terminal["orders_submitted"] is False
    assert terminal["trading_authority"] is False
    assert validate_round28_sealed_terminal_result(terminal) == terminal

    tampered = dict(terminal)
    tampered["observed_after_cost_bbo_uplift_gate_passed"] = False
    with pytest.raises(ValueError, match="terminal result differs"):
        validate_round28_sealed_terminal_result(tampered)


def test_round28_sealed_economics_rejects_self_inconsistent_gate() -> None:
    prediction = _prediction()
    economics = _economics(prediction)
    economics["economic_uplift_gate_passed"] = False
    economics["report_sha256"] = _canonical_sha256(
        {key: value for key, value in economics.items() if key != "report_sha256"}
    )

    with pytest.raises(ValueError, match="economic report differs"):
        validate_round28_sealed_economic_report(economics)


def test_round28_sealed_implementation_amendment_is_source_bound() -> None:
    path = (
        ROOT
        / "docs/model-research/polymarket/"
        "round-028-sealed-evaluation-implementation-amendment-v1.json"
    )
    amendment = json.loads(path.read_text(encoding="ascii"))
    claimed = amendment.pop("amendment_sha256")

    assert claimed == _canonical_sha256(amendment)
    assert amendment["knowledge_at_freeze"]["official_outcomes_accessed"] is False
    assert amendment["authority"]["live_trading_authority"] is False
    for relative, expected in amendment["source_text_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
