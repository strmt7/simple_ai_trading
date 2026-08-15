from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import simple_ai_trading.polymarket_round23_lead_lag as lead_lag


REPOSITORY = Path(__file__).resolve().parents[1]


def test_round23_lead_lag_spec_is_hash_bound_and_non_authoritative() -> None:
    spec = lead_lag.load_round23_lead_lag_spec(REPOSITORY)

    assert (
        spec["specification_sha256"] == lead_lag.POLYMARKET_ROUND23_LEAD_LAG_SPEC_SHA256
    )
    assert len(spec["candidate_features"]) == 14
    assert set(spec["authority"].values()) == {False}
    assert spec["knowledge_at_freeze"]["condition_level_outcomes_used"] is False
    assert spec["knowledge_at_freeze"]["future_polymarket_targets_constructed"] is False


def test_round23_binance_features_are_causal_returns_flows_and_basis() -> None:
    spot = tuple(
        lead_lag._BinanceBar(  # noqa: SLF001
            close=100.0 + index,
            quote_volume=10.0,
            signed_quote_volume=2.0,
        )
        for index in range(16)
    )
    futures = tuple(
        lead_lag._BinanceBar(  # noqa: SLF001
            close=101.0 + index,
            quote_volume=20.0,
            signed_quote_volume=-5.0,
        )
        for index in range(16)
    )

    values = lead_lag._binance_feature_vector(spot, futures)  # noqa: SLF001

    assert len(values) == 14
    assert values[0] > 0
    assert values[2] > values[1] > values[0]
    assert values[6:9] == (0.2, 0.2, 0.2)
    assert values[9:12] == (-0.25, -0.25, -0.25)
    assert values[12] > 0
    assert values[13] < 0


def test_round23_ridge_comparison_recovers_incremental_candidate_signal() -> None:
    conditions = np.asarray(
        [f"condition-{group}" for group in range(6) for _ in range(20)],
        dtype=np.str_,
    )
    incremental = np.linspace(-2.0, 2.0, conditions.size, dtype=np.float64)
    baseline = np.zeros((conditions.size, 1), dtype=np.float64)
    candidate = np.column_stack((baseline, incremental))
    target = incremental * 0.5
    partition = lead_lag._Partition(  # noqa: SLF001
        role="train",
        baseline=baseline,
        candidate=candidate,
        target=target,
        conditions=conditions,
    )

    baseline_scores = lead_lag._loco_scores(  # noqa: SLF001
        partition,
        baseline,
        (0.1, 1.0, 10.0),
    )
    candidate_scores = lead_lag._loco_scores(  # noqa: SLF001
        partition,
        candidate,
        (0.1, 1.0, 10.0),
    )
    selected = lead_lag._select_penalty(candidate_scores)  # noqa: SLF001
    model = lead_lag._fit_ridge(  # noqa: SLF001
        candidate,
        target,
        conditions,
        penalty=selected,
    )

    assert min(candidate_scores.values()) < min(baseline_scores.values())
    assert np.mean((target - model.predict(candidate)) ** 2) < 0.01


def test_round23_lead_lag_source_has_no_resolution_or_execution_dependency() -> None:
    source = Path(lead_lag.__file__).read_text(encoding="utf-8")

    assert "polymarket_round22_targets" not in source
    assert "FROM target." not in source
    assert "INSERT INTO target." not in source
    assert "polymarket_live" not in source
    assert "create_order" not in source
    assert "place_order" not in source


def test_round23_published_result_and_graph_are_exactly_bound() -> None:
    result_path = (
        REPOSITORY
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-023-lead-lag-results-v1.json"
    )
    graph_path = result_path.with_name("round-023-lead-lag-performance.svg")
    obsolete_graph = result_path.with_name("round-022-diagnostic-performance.svg")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    claimed = result.pop("result_sha256")
    actual = hashlib.sha256(
        json.dumps(
            result,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    assert claimed == actual
    assert result["mechanism_gate_passed"] is True
    assert result["authority"] == {
        "ai_edge_claim": False,
        "economic_backtest": False,
        "live_trading": False,
        "model_promotion": False,
        "paper_trading": False,
        "profitability_claim": False,
    }
    assert f"source-result-sha256:{claimed}" in graph_path.read_text(encoding="utf-8")
    assert not obsolete_graph.exists()
