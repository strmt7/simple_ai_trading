from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from simple_ai_trading import polymarket_mlp as mlp
from simple_ai_trading.polymarket_ridge import (
    PolymarketPolicyEvaluation,
    PolymarketPolicyMetrics,
    PolymarketRidgeDataset,
    PolymarketRidgeReport,
)


def test_mlp_orchestration_progress_callbacks_are_optional() -> None:
    evidence = cast(
        mlp.PolymarketMLPBackendEvidence,
        SimpleNamespace(kind="cpu", device="cpu", fallback_reason=""),
    )
    prepared = cast(
        mlp._PolymarketMLPPreparedData,
        SimpleNamespace(
            training_x=np.empty((1, 39)),
            validation_x=np.empty((1, 39)),
        ),
    )

    mlp._emit_mlp_preflight_progress(None, evidence, prepared)
    mlp._emit_mlp_reproducibility_progress(None, 0.0)
    mlp._emit_mlp_validation_progress(
        None,
        cast(mlp._PolymarketMLPValidationResult, object()),
        cast(mlp.PolymarketRidgeSplit, object()),
    )


def test_mlp_reproducibility_rejects_probability_drift() -> None:
    first = cast(
        mlp.PolymarketMLPMember,
        SimpleNamespace(
            predict_standardized=lambda _features: np.asarray([0.0]),
        ),
    )
    repeated = cast(
        mlp.PolymarketMLPMember,
        SimpleNamespace(
            predict_standardized=lambda _features: np.asarray([1.0]),
        ),
    )

    with pytest.raises(ValueError, match="same-seed probability drift"):
        mlp._reproducibility_drift(
            (first,),
            repeated,
            np.empty((1, 39)),
        )


def test_parent_validation_replay_requires_selected_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = SimpleNamespace(asdict=lambda: {"trial": "actual"})
    evaluation = cast(
        PolymarketPolicyEvaluation,
        SimpleNamespace(metrics=metrics),
    )
    monkeypatch.setattr(
        mlp,
        "evaluate_polymarket_policy",
        lambda *_args, **_kwargs: evaluation,
    )
    parent = cast(
        PolymarketRidgeReport,
        SimpleNamespace(selected_threshold=0.5, validation_trials=()),
    )
    prepared = cast(
        mlp._PolymarketMLPPreparedData,
        SimpleNamespace(validation_indices=np.asarray([0], dtype=np.int64)),
    )

    with pytest.raises(ValueError, match="parent-policy validation simulation"):
        mlp._require_parent_validation_replay(
            cast(PolymarketRidgeDataset, object()),
            parent,
            prepared,
            np.asarray([0.5]),
        )


def test_mlp_admission_reasons_cover_all_validation_failures() -> None:
    evaluation = cast(
        PolymarketPolicyEvaluation,
        SimpleNamespace(metrics=SimpleNamespace(gate_passed=False)),
    )
    uplift = cast(
        mlp.PolymarketMLPBootstrap,
        SimpleNamespace(lower_95=0.0),
    )

    selected = mlp._select_mlp_validation((evaluation,))
    reasons = mlp._mlp_admission_reasons(selected, uplift, None)

    assert selected is None
    assert reasons == (
        "no_validation_threshold_passed",
        "validation_log_loss_uplift_lower_not_positive",
        "validation_stress_utility_not_above_ridge",
    )


def test_mlp_test_reasons_cover_every_failure_family() -> None:
    parent_metrics = SimpleNamespace(
        aggregate_stress_utility_quote=10.0,
        maximum_realized_drawdown_quote=1.0,
        pnl_by_asset={"BTC": 10.0, "ETH": 10.0, "SOL": 10.0},
    )
    parent = cast(
        PolymarketRidgeReport,
        SimpleNamespace(test_log_loss=0.2, test_metrics=parent_metrics),
    )
    test_metrics = cast(
        PolymarketPolicyMetrics,
        SimpleNamespace(
            gate_passed=False,
            gate_reasons=("policy_gate",),
            aggregate_stress_utility_quote=9.0,
            maximum_realized_drawdown_quote=2.0,
            pnl_by_asset={"BTC": 9.0, "ETH": 9.0, "SOL": 9.0},
        ),
    )
    uplift = cast(
        mlp.PolymarketMLPBootstrap,
        SimpleNamespace(lower_95=0.0),
    )

    assert mlp._mlp_test_reasons(parent, 0.3, test_metrics, uplift) == (
        "policy_gate",
        "test_log_loss_not_below_ridge",
        "test_stress_utility_not_above_ridge",
        "test_realized_drawdown_above_ridge",
        "test_asset_utility_below_ridge:BTC",
        "test_asset_utility_below_ridge:ETH",
        "test_asset_utility_below_ridge:SOL",
        "test_utility_bootstrap_lower_not_positive",
    )


def test_mlp_test_progress_supports_optional_and_complete_states() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    metrics = cast(
        PolymarketPolicyMetrics,
        SimpleNamespace(aggregate_stress_utility_quote=4.0),
    )

    mlp._emit_mlp_test_progress(None, status="started")
    mlp._emit_mlp_test_progress(
        lambda phase, payload: events.append((phase, dict(payload))),
        status="complete",
        test_loss=0.2,
        test_metrics=metrics,
        reason_count=3,
    )

    assert events == [
        (
            "polymarket_mlp_test",
            {
                "status": "complete",
                "test_log_loss": 0.2,
                "test_stress_utility_quote": 4.0,
                "gate_reason_count": 3,
            },
        )
    ]


def test_mlp_admission_rejects_missing_selected_validation() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    validation = cast(
        mlp._PolymarketMLPValidationResult,
        SimpleNamespace(admitted=True, selected=None),
    )

    with pytest.raises(RuntimeError, match="admission state is inconsistent"):
        mlp._evaluate_mlp_test_if_admitted(
            cast(PolymarketRidgeDataset, object()),
            cast(PolymarketRidgeReport, object()),
            cast(mlp._PolymarketMLPPreparedData, object()),
            cast(mlp.PolymarketMLPEnsemble, object()),
            validation,
            lambda phase, payload: events.append((phase, dict(payload))),
        )

    assert events == [("polymarket_mlp_test", {"status": "started"})]


def _admitted_test_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ridge_replay_matches: bool,
) -> tuple[
    PolymarketRidgeDataset,
    PolymarketRidgeReport,
    mlp._PolymarketMLPPreparedData,
    mlp.PolymarketMLPEnsemble,
]:
    parent_metrics = SimpleNamespace(
        aggregate_stress_utility_quote=1.0,
        maximum_realized_drawdown_quote=1.0,
        pnl_by_asset={"BTC": 1.0, "ETH": 1.0, "SOL": 1.0},
        asdict=lambda: {"replay": "expected"},
    )
    test_metrics = SimpleNamespace(
        gate_passed=True,
        gate_reasons=(),
        aggregate_stress_utility_quote=2.0,
        maximum_realized_drawdown_quote=0.5,
        pnl_by_asset={"BTC": 2.0, "ETH": 2.0, "SOL": 2.0},
    )
    ridge_metrics = SimpleNamespace(
        asdict=lambda: {"replay": "expected" if ridge_replay_matches else "different"},
    )
    evaluations = iter(
        (
            SimpleNamespace(metrics=test_metrics, selected_indices=()),
            SimpleNamespace(metrics=ridge_metrics, selected_indices=()),
        )
    )
    monkeypatch.setattr(
        mlp, "_validate_partition_label_breadth", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        mlp,
        "_partition_indices",
        lambda *_args: np.asarray([0], dtype=np.int64),
    )
    monkeypatch.setattr(
        mlp,
        "_matrix",
        lambda *_args: (np.zeros((1, 39)), np.asarray([1.0])),
    )
    monkeypatch.setattr(mlp, "_binary_log_loss", lambda *_args: 0.2)
    monkeypatch.setattr(
        mlp,
        "evaluate_polymarket_policy",
        lambda *_args, **_kwargs: next(evaluations),
    )
    monkeypatch.setattr(mlp, "_condition_utility", lambda *_args: {"c": 0.0})
    monkeypatch.setattr(mlp, "_group_utility_uplift", lambda *_args: (1.0,))
    monkeypatch.setattr(
        mlp,
        "_bootstrap",
        lambda *_args, **_kwargs: cast(
            mlp.PolymarketMLPBootstrap,
            SimpleNamespace(lower_95=1.0),
        ),
    )
    dataset = cast(
        PolymarketRidgeDataset,
        SimpleNamespace(dataset_sha256="a" * 64),
    )
    parent = cast(
        PolymarketRidgeReport,
        SimpleNamespace(
            selected_model=SimpleNamespace(
                predict=lambda _values: np.asarray([0.4]),
            ),
            selected_threshold=0.5,
            test_metrics=parent_metrics,
            test_log_loss=0.5,
        ),
    )
    prepared = cast(
        mlp._PolymarketMLPPreparedData,
        SimpleNamespace(split=SimpleNamespace(test_groups=(1,))),
    )
    ensemble = cast(
        mlp.PolymarketMLPEnsemble,
        SimpleNamespace(predict=lambda _values: np.asarray([0.6])),
    )
    return dataset, parent, prepared, ensemble


def test_admitted_mlp_test_evaluates_and_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, parent, prepared, ensemble = _admitted_test_context(
        monkeypatch,
        ridge_replay_matches=True,
    )
    events: list[tuple[str, dict[str, object]]] = []
    validation = cast(
        mlp._PolymarketMLPValidationResult,
        SimpleNamespace(
            admitted=True,
            selected=SimpleNamespace(metrics=SimpleNamespace(threshold=0.5)),
        ),
    )

    result = mlp._evaluate_mlp_test_if_admitted(
        dataset,
        parent,
        prepared,
        ensemble,
        validation,
        lambda phase, payload: events.append((phase, dict(payload))),
    )

    assert result.selected_policy == "causal_mlp"
    assert result.selected_threshold == 0.5
    assert result.test_loss == 0.2
    assert result.test_reasons == ()
    assert events == [
        ("polymarket_mlp_test", {"status": "started"}),
        (
            "polymarket_mlp_test",
            {
                "status": "complete",
                "test_log_loss": 0.2,
                "test_stress_utility_quote": 2.0,
                "gate_reason_count": 0,
            },
        ),
    ]


def test_admitted_mlp_test_rejects_parent_replay_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, parent, prepared, ensemble = _admitted_test_context(
        monkeypatch,
        ridge_replay_matches=False,
    )

    with pytest.raises(ValueError, match="parent test replay differs"):
        mlp._evaluate_admitted_mlp_test(
            dataset,
            parent,
            prepared,
            ensemble,
            0.5,
            None,
        )
