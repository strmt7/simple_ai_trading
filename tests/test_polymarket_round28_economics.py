from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import simple_ai_trading.polymarket_round28_economics as economics
from simple_ai_trading.polymarket_round27_economics import Round27EconomicConfig
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_FEATURE_NAMES,
)
from simple_ai_trading.polymarket_round28_economics import (
    evaluate_round28_matched_economics,
    paired_round28_economic_scenario,
    project_round28_economic_partition,
)
from simple_ai_trading.polymarket_round28_model import (
    Round28ModelSample,
    Round28Partition,
)


ROOT = Path(__file__).resolve().parents[1]
_START_MS = 1_786_784_400_000
_HASH = "a" * 64


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


def _sample(index: int) -> Round28ModelSample:
    base = (float(index % 3),) + (0.0,) * (len(POLYMARKET_ROUND27_FEATURE_NAMES) - 1)
    augmented = (
        base
        + (float(index % 2),)
        + (0.0,) * (len(POLYMARKET_ROUND28_FEATURE_NAMES) - len(base) - 1)
    )
    return Round28ModelSample(
        slot_id="stage1-b",
        role="selection",
        condition_id="0x" + format(index + 1, "064x"),
        event_start_ms=_START_MS + index * 300_000,
        decision_time_ms=_START_MS + index * 300_000 + 30_000,
        market_prior_probability=0.5,
        base_values=base,
        augmented_values=augmented,
        target_up=index % 2,
        condition_weight=1.0,
        feature_row_sha256=hashlib.sha256(
            f"round28-feature-{index}".encode("ascii")
        ).hexdigest(),
    ).validated()


def _scenario(
    conditions: tuple[str, ...],
    *,
    delay_ms: int,
    pnl: str,
    drawdown: str,
    passed: bool = True,
) -> dict[str, object]:
    trades = [
        {
            "condition_id": condition_id,
            "execution_state": "FILLED",
            "net_pnl_quote": pnl,
        }
        for condition_id in conditions
    ]
    body: dict[str, object] = {
        "delay_ms": delay_ms,
        "evaluated_condition_count": len(conditions),
        "net_pnl_quote": format(Decimal(pnl) * len(conditions), "f"),
        "maximum_drawdown_fraction": drawdown,
        "scenario_edge_gate_passed": passed,
        "trades": trades,
    }
    body["scenario_sha256"] = _canonical_sha256(body)
    return body


def test_round28_projection_preserves_exact_matched_execution_population() -> None:
    selected = Round28Partition.from_samples(
        tuple(_sample(index) for index in range(20)),
        role="selection",
    )

    projected = project_round28_economic_partition(selected)

    assert np.array_equal(projected.features, selected.base_features)
    assert np.array_equal(projected.offsets, selected.offsets)
    assert np.array_equal(projected.targets, selected.targets)
    assert np.array_equal(projected.weights, selected.weights)
    assert np.array_equal(projected.conditions, selected.conditions)
    assert [sample.feature_row_sha256 for sample in projected.samples] == [
        sample.feature_row_sha256 for sample in selected.samples
    ]

    drifted_features = selected.base_features.copy()
    drifted_features[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="projection differs"):
        project_round28_economic_partition(
            replace(selected, base_features=drifted_features)
        )


def test_round28_paired_economics_requires_after_cost_uplift_and_drawdown() -> None:
    conditions = tuple(sample.condition_id for sample in map(_sample, range(20)))
    base = _scenario(conditions, delay_ms=500, pnl="0.10", drawdown="0.02")
    augmented = _scenario(
        conditions,
        delay_ms=500,
        pnl="0.20",
        drawdown="0.01",
    )

    report = paired_round28_economic_scenario(
        base=base,
        augmented=augmented,
        ordered_conditions=conditions,
        bootstrap_draws=1_000,
        bootstrap_seed=28_028,
    )

    assert report["net_pnl_delta_quote"] == "2.00"
    assert report["paired_condition_bootstrap"]["ci95_lower"] > 0
    assert report["scenario_uplift_gate_passed"] is True

    worse_drawdown = _scenario(
        conditions,
        delay_ms=500,
        pnl="0.20",
        drawdown="0.03",
    )
    rejected = paired_round28_economic_scenario(
        base=base,
        augmented=worse_drawdown,
        ordered_conditions=conditions,
        bootstrap_draws=1_000,
        bootstrap_seed=28_028,
    )
    assert rejected["scenario_uplift_gate_passed"] is False
    assert rejected["gate_checks"]["maximum_drawdown_not_worse_than_base"] is False


def test_round28_paired_economics_rejects_tampered_scenario() -> None:
    conditions = tuple(sample.condition_id for sample in map(_sample, range(20)))
    base = _scenario(conditions, delay_ms=500, pnl="0.10", drawdown="0.02")
    augmented = _scenario(
        conditions,
        delay_ms=500,
        pnl="0.20",
        drawdown="0.01",
    )
    augmented["net_pnl_quote"] = "999"

    with pytest.raises(ValueError, match="scenario hash differs"):
        paired_round28_economic_scenario(
            base=base,
            augmented=augmented,
            ordered_conditions=conditions,
            bootstrap_draws=1_000,
            bootstrap_seed=28_028,
        )


class _Model:
    def __init__(self, feature_view: str, probability: float) -> None:
        self.feature_view = feature_view
        self.model_name = "l2_offset_logistic"
        self.model_sha256 = "b" * 64 if feature_view == "round27_base" else "c" * 64
        self._probability = probability

    def predict(self, features: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        assert features.shape[0] == offsets.shape[0]
        return np.full(offsets.shape, self._probability, dtype=np.float64)


def test_round28_coordinator_replays_batch_factory_twice_and_binds_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = tuple(_sample(index) for index in range(20))
    conditions = tuple(sample.condition_id for sample in samples)
    pair = SimpleNamespace(
        model_family="l2_offset_logistic",
        base_model=_Model("round27_base", 0.60),
        augmented_model=_Model("round28_bbo_augmented", 0.80),
    )
    monkeypatch.setattr(
        economics,
        "load_round28_selected_pair",
        lambda *_args, **_kwargs: pair,
    )

    def fake_evaluate(**kwargs: object) -> dict[str, object]:
        augmented = str(kwargs["model_name"]).endswith("round28_bbo_augmented")
        return {
            "economic_edge_gate_passed": True,
            "scenarios": [
                _scenario(
                    conditions,
                    delay_ms=delay,
                    pnl="0.20" if augmented else "0.10",
                    drawdown="0.01" if augmented else "0.02",
                )
                for delay in (250, 500, 1_000, 2_000)
            ],
        }

    monkeypatch.setattr(
        economics,
        "evaluate_round27_economic_scenarios",
        fake_evaluate,
    )
    calls = 0

    def batches() -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        return ()

    preregistration = json.loads(
        (
            ROOT / "docs/model-research/polymarket/"
            "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
        ).read_text(encoding="ascii")
    )
    amendment = json.loads(
        (
            ROOT / "docs/model-research/polymarket/"
            "round-028-selection-implementation-amendment-v1.json"
        ).read_text(encoding="ascii")
    )
    economic_amendment = json.loads(
        (
            ROOT / "docs/model-research/polymarket/"
            "round-028-economic-implementation-amendment-v1.json"
        ).read_text(encoding="ascii")
    )
    report = evaluate_round28_matched_economics(
        samples=samples,
        selection_claim={"claim_sha256": "d" * 64},
        contract={"contract_sha256": "e" * 64},
        preregistration=preregistration,
        implementation_amendment=amendment,
        economic_implementation_amendment=economic_amendment,
        markets=tuple(
            SimpleNamespace(
                condition_id=sample.condition_id,
                event_start_ms=sample.event_start_ms,
            )
            for sample in samples
        ),
        outcomes_up={sample.condition_id: sample.target_up for sample in samples},
        source_audit_sha256="f" * 64,
        resolution_evidence_sha256="1" * 64,
        book_batch_factory=batches,
        config=Round27EconomicConfig(
            minimum_executed_trades=20,
            minimum_profitable_conditions=20,
            bootstrap_draws=1_000,
        ),
    )

    assert calls == 2
    assert report["economic_uplift_gate_passed"] is True
    assert (
        report["round28_selection_implementation_amendment_sha256"]
        == (amendment["amendment_sha256"])
    )
    assert (
        report["round28_economic_implementation_amendment_sha256"]
        == economic_amendment["amendment_sha256"]
    )
    assert report["sealed_partition_accessed"] is False
    assert report["orders_submitted"] is False
