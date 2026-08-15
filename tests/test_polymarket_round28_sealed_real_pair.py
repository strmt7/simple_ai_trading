from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round28_sealed as sealed_module
from simple_ai_trading.polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    Round27EconomicConfig,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round28_model import (
    Round28ModelSample,
    Round28Partition,
    fit_round28_l2_offset,
)
from simple_ai_trading.polymarket_round28_sealed import (
    POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION,
    evaluate_round28_sealed_economics,
    validate_round28_sealed_economic_report,
    validate_round28_sealed_prediction_result,
)
from simple_ai_trading.polymarket_round28_selection import (
    Round28SelectedPair,
    round28_pair_selection_report,
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


def _partition() -> Round28Partition:
    samples: list[Round28ModelSample] = []
    for index in range(40):
        target = index % 2
        base = (0.0,) * len(POLYMARKET_ROUND27_FEATURE_NAMES)
        overlay = (1.0 if target else -1.0,) + (0.0,) * (
            len(POLYMARKET_ROUND28_BOOK_TICKER_FEATURE_NAMES) - 1
        )
        start = 1_800_000 + index * 1_200_000
        samples.append(
            Round28ModelSample(
                slot_id="stage1-c",
                role="sealed",
                condition_id="0x" + f"{index + 1:064x}",
                event_start_ms=start,
                decision_time_ms=start + 30_000,
                market_prior_probability=0.6,
                base_values=base,
                augmented_values=(*base, *overlay),
                target_up=target,
                condition_weight=1.0,
                feature_row_sha256=hashlib.sha256(
                    f"sealed-{index}".encode("ascii")
                ).hexdigest(),
            ).validated()
        )
    return Round28Partition.from_samples(samples, role="sealed")


def _prediction_fixture() -> tuple[
    Round28Partition,
    object,
    object,
    dict[str, object],
]:
    partition = _partition()
    base = fit_round28_l2_offset(
        partition,
        feature_view="round27_base",
        penalty=1.0,
    )
    augmented = fit_round28_l2_offset(
        partition,
        feature_view="round28_bbo_augmented",
        penalty=0.01,
    )
    pair_report = round28_pair_selection_report(
        partition,
        base_model=base,
        augmented_model=augmented,
        prediction_evaluation={
            "bootstrap_draws": 1_000,
            "balanced_accuracy_floor": 0.51,
            "calibration_ece_maximum_degradation": 0.01,
        },
        training_detail={
            "models_refit": False,
            "hyperparameters_retuned": False,
            "thresholds_changed": False,
        },
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION,
        "round27_model_contract_sha256": "a" * 64,
        "round28_preregistration_sha256": "b" * 64,
        "selection_input_manifest_sha256": "c" * 64,
        "selection_claim_sha256": "d" * 64,
        "selection_economic_report_sha256": "e" * 64,
        "source_binding_sha256": "f" * 64,
        "selected_model_family": "l2_offset_logistic",
        "base_model_sha256": base.model_sha256,
        "augmented_model_sha256": augmented.model_sha256,
        "condition_count": 40,
        "row_count": 40,
        "condition_population_sha256": "1" * 64,
        "paired_probability_report": pair_report,
        "prediction_uplift_gate_passed": True,
        "sealed_partition_accessed": True,
        "models_refit": False,
        "hyperparameters_retuned": False,
        "thresholds_changed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "orders_submitted": False,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return partition, base, augmented, body


def test_round28_sealed_prediction_validator_accepts_real_nested_pair() -> None:
    _partition_value, _base, _augmented, body = _prediction_fixture()

    assert validate_round28_sealed_prediction_result(body) == body


def test_round28_sealed_economics_replays_both_frozen_models_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition, base, augmented, prediction = _prediction_fixture()
    condition_ids = tuple(sample.condition_id for sample in partition.samples)
    calls: list[str] = []

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        model_name = str(kwargs["model_name"])
        calls.append(model_name)
        is_augmented = model_name.endswith("round28_bbo_augmented")
        pnl = Decimal("0.20" if is_augmented else "0.10")
        scenarios: list[dict[str, object]] = []
        for delay in (250, 500, 1_000, 2_000):
            scenario: dict[str, object] = {
                "delay_ms": delay,
                "evaluated_condition_count": len(condition_ids),
                "net_pnl_quote": format(pnl * len(condition_ids), "f"),
                "maximum_drawdown_fraction": (
                    "0.01" if is_augmented else "0.02"
                ),
                "gate_checks": {"fixture_gate": True},
                "scenario_edge_gate_passed": True,
                "trades": [
                    {
                        "condition_id": condition_id,
                        "execution_state": "FILLED",
                        "net_pnl_quote": format(pnl, "f"),
                    }
                    for condition_id in condition_ids
                ],
            }
            scenario["scenario_sha256"] = _canonical_sha256(scenario)
            scenarios.append(scenario)
        body: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
            "partition_role": "sealed",
            "model_name": model_name,
            "model_sha256": kwargs["model_sha256"],
            "source_audit_sha256": kwargs["source_audit_sha256"],
            "resolution_evidence_sha256": kwargs[
                "resolution_evidence_sha256"
            ],
            "config": kwargs["config"].asdict(),
            "scenarios": scenarios,
            "economic_edge_gate_passed": True,
            "edge_claim": False,
            "profitability_claim": False,
            "orders_submitted": False,
            "trading_authority": False,
        }
        body["report_sha256"] = _canonical_sha256(body)
        return body

    monkeypatch.setattr(
        sealed_module,
        "evaluate_round27_economic_scenarios",
        fake_evaluate,
    )
    config = Round27EconomicConfig(
        minimum_executed_trades=20,
        minimum_profitable_conditions=20,
        bootstrap_draws=1_000,
    )
    report = evaluate_round28_sealed_economics(
        samples=partition.samples,
        pair=Round28SelectedPair(
            model_family="l2_offset_logistic",
            base_model=base,
            augmented_model=augmented,
        ),
        selection_claim_sha256="d" * 64,
        sealed_prediction_result=prediction,
        markets=tuple(
            SimpleNamespace(
                condition_id=sample.condition_id,
                event_start_ms=sample.event_start_ms,
            )
            for sample in partition.samples
        ),
        outcomes_up={sample.condition_id: sample.target_up for sample in partition.samples},
        source_binding_sha256="f" * 64,
        resolution_evidence_sha256="4" * 64,
        config=config,
        book_batch_factory=lambda: (),
    )

    assert len(calls) == 2
    assert report["economic_uplift_gate_passed"] is True
    assert report["models_refit"] is False
    assert validate_round28_sealed_economic_report(report) == report
