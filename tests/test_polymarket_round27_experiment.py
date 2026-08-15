from __future__ import annotations

import copy

import pytest

from simple_ai_trading.polymarket_round27_experiment import (
    run_round27_development_selection,
    run_round27_sealed_evaluation,
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

    sealed = run_round27_sealed_evaluation(
        samples=_samples(),
        contract=_contract(),
        selection_claim=claim,
        selected_model=model,
    )

    assert sealed["prediction_edge_gate_passed"] is True
    assert sealed["economic_edge_gate_evaluated"] is False
    assert sealed["profitability_claim"] is False


def test_sealed_evaluation_rejects_a_tampered_selection_claim() -> None:
    claim, model = run_round27_development_selection(
        samples=_samples(),
        contract=_contract(),
        claim_writer=lambda value: value["claim_sha256"],
        compute_backend="cpu",
    )
    tampered = copy.deepcopy(claim)
    tampered["selected_model_name"] = "market_prior"

    with pytest.raises(ValueError, match="selection claim differs"):
        run_round27_sealed_evaluation(
            samples=_samples(),
            contract=_contract(),
            selection_claim=tampered,
            selected_model=model,
        )
