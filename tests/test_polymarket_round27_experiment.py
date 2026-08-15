from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest
import simple_ai_trading.polymarket_round27_experiment as round27_experiment

from simple_ai_trading.polymarket_round27_experiment import (
    build_round27_selection_economic_claim,
    load_round27_selected_model,
    run_round27_development_selection,
    run_round27_sealed_evaluation,
    validate_round27_sealed_access_artifacts,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round27_model import Round27ModelSample


def _samples() -> tuple[Round27ModelSample, ...]:
    rows: list[Round27ModelSample] = []
    condition_index = 0
    for role_index, role in enumerate(("train", "calibration", "selection", "sealed")):
        for local_index in range(25):
            condition_index += 1
            condition_id = "0x" + f"{condition_index:064x}"
            target = local_index % 2
            direction = 1.0 if target else -1.0
            event_start = (
                1_786_000_000_000 + role_index * 100_000_000 + local_index * 300_000
            )
            for row_index, offset in enumerate((30_000, 120_000, 240_000)):
                values = [0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES)
                values[0] = direction * (1.0 + row_index * 0.1)
                rows.append(
                    Round27ModelSample(
                        slot_id=f"stage1-{chr(ord('a') + role_index)}",
                        role=role,
                        condition_id=condition_id,
                        event_start_ms=event_start,
                        decision_time_ms=event_start + offset,
                        market_prior_probability=0.52 if target else 0.48,
                        values=tuple(values),
                        target_up=target,
                        condition_weight=1.0 / 3.0,
                        feature_row_sha256=f"{condition_index:064x}",
                    ).validated()
                )
    return tuple(rows)


def _contract() -> dict[str, object]:
    return {
        "contract_sha256": "a" * 64,
        "model_implementation_amendment_sha256": "d" * 64,
        "minimum_population": {
            "train_conditions": 20,
            "calibration_conditions": 20,
            "selection_conditions": 20,
            "sealed_conditions": 20,
        },
        "prediction_evaluation": {
            "balanced_accuracy_floor": 0.51,
            "bootstrap_draws": 1_000,
            "calibration_ece_maximum_degradation": 0.01,
        },
    }


def _economic_claim(contract, claim, model, *, passed: bool = True):
    assert model is not None
    report = {
        "schema_version": "polymarket-round27-economic-replay-v4",
        "partition_role": "selection",
        "model_name": model.model_name,
        "model_sha256": model.asdict()["model_sha256"],
        "economic_edge_gate_passed": passed,
        "orders_submitted": False,
        "trading_authority": False,
        "edge_claim": False,
        "profitability_claim": False,
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    claim = build_round27_selection_economic_claim(
        contract=contract,
        selection_claim=claim,
        selected_model=model,
        economic_report=report,
        claim_writer=lambda value: value["claim_sha256"],
    )
    return claim, report


def test_selection_claim_precedes_sealed_evaluation() -> None:
    persisted: dict[str, object] = {}

    def write_claim(value):
        persisted.update(value)
        return value["claim_sha256"]

    claim, model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=write_claim,
        compute_backend="cpu",
    )

    assert model is not None
    assert claim == persisted
    assert claim["sealed_partition_accessed"] is False
    assert claim["economic_metrics_computed"] is False
    assert claim["edge_claim"] is False
    assert claim["model_implementation_amendment_sha256"] == "d" * 64

    restored = load_round27_selected_model(
        selection_claim=persisted,
        contract=_contract(),
    )
    assert restored is not None
    assert restored.asdict() == model.asdict()

    economic_claim, economic_report = _economic_claim(_contract(), claim, restored)
    gated = validate_round27_sealed_access_artifacts(
        contract=_contract(),
        selection_claim=claim,
        selection_economic_claim=economic_claim,
        selection_economic_report=economic_report,
    )
    assert gated is not None
    assert gated.asdict() == restored.asdict()
    sealed = run_round27_sealed_evaluation(
        samples=_samples(),
        contract=_contract(),
        selection_claim=claim,
        selection_economic_claim=economic_claim,
        selection_economic_report=economic_report,
        selected_model=restored,
    )

    assert sealed["prediction_edge_gate_passed"] is True
    assert sealed["economic_edge_gate_evaluated"] is False
    assert sealed["profitability_claim"] is False


def test_non_unit_calibration_scale_survives_selection_claim_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        round27_experiment,
        "select_round27_correction_scale",
        lambda _model, _partition: (
            0.5,
            {"0.0": 1.0, "0.25": 0.8, "0.5": 0.7, "0.75": 0.8, "1.0": 1.0},
        ),
    )

    claim, model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=lambda value: value["claim_sha256"],
        compute_backend="cpu",
    )

    assert model is not None
    assert model.correction_scale == 0.5
    restored = load_round27_selected_model(
        selection_claim=claim,
        contract=_contract(),
    )
    assert restored is not None
    assert restored.asdict() == model.asdict()


def test_sealed_evaluation_rejects_a_tampered_selection_claim() -> None:
    claim, model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=lambda value: value["claim_sha256"],
        compute_backend="cpu",
    )
    tampered = copy.deepcopy(claim)
    tampered["selected_model_name"] = "market_prior"
    economic_claim, economic_report = _economic_claim(_contract(), claim, model)

    with pytest.raises(ValueError, match="selection claim differs"):
        run_round27_sealed_evaluation(
            samples=_samples(),
            contract=_contract(),
            selection_claim=tampered,
            selection_economic_claim=economic_claim,
            selection_economic_report=economic_report,
            selected_model=model,
        )


def test_selected_model_rejects_a_missing_model_amendment_binding() -> None:
    claim, _model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=lambda value: value["claim_sha256"],
        compute_backend="cpu",
    )
    missing = copy.deepcopy(claim)
    missing.pop("model_implementation_amendment_sha256")
    body = dict(missing)
    body.pop("claim_sha256")
    missing["claim_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="selection claim differs"):
        load_round27_selected_model(
            selection_claim=missing,
            contract=_contract(),
        )


def test_sealed_evaluation_rejects_same_name_different_model_artifact() -> None:
    claim, model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=lambda value: value["claim_sha256"],
        compute_backend="cpu",
    )
    assert model is not None
    economic_claim, economic_report = _economic_claim(_contract(), claim, model)
    different_model = replace(model, correction_scale=model.correction_scale + 0.01)

    with pytest.raises(ValueError, match="selection claim differs"):
        run_round27_sealed_evaluation(
            samples=_samples(),
            contract=_contract(),
            selection_claim=claim,
            selection_economic_claim=economic_claim,
            selection_economic_report=economic_report,
            selected_model=different_model,
        )


def test_sealed_evaluation_requires_passing_selection_economics() -> None:
    claim, model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=lambda value: value["claim_sha256"],
        compute_backend="cpu",
    )
    failed_economic_claim, failed_economic_report = _economic_claim(
        _contract(),
        claim,
        model,
        passed=False,
    )

    with pytest.raises(ValueError, match="selection economic claim differs"):
        run_round27_sealed_evaluation(
            samples=_samples(),
            contract=_contract(),
            selection_claim=claim,
            selection_economic_claim=failed_economic_claim,
            selection_economic_report=failed_economic_report,
            selected_model=model,
        )
