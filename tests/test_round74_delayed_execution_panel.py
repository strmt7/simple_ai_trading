from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simple_ai_trading.impact_absorption import L2BookState
from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    Round74ReplayObservation,
)
from simple_ai_trading.impact_absorption_event_targets import (
    Round74EventTargetOutcome,
)
import simple_ai_trading.round74_delayed_execution_panel as subject


RUN_ID = "1" * 32
PARTITION_SHA256 = "2" * 64
SCALER_SHA256 = "3" * 64
ASSEMBLY_SHA256 = "4" * 64
CAPTURE_SHA256 = "5" * 64
LATENCY_SHA256 = "6" * 64
SPEC_SHA256 = "7" * 64


class _OnePass:
    def __init__(self, values: tuple[Round74ReplayObservation, ...]) -> None:
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("observation stream was scanned more than once")
        yield from self.values


def _state() -> L2BookState:
    return L2BookState(
        symbol="BTCUSDT",
        update_id=1,
        best_bid=99.0,
        best_ask=101.0,
        spread_bps=200.0,
        mid=100.0,
        bid_levels=((99.0, 1.0),),
        ask_levels=((101.0, 1.0),),
        bid_depth_quote_5=99.0,
        ask_depth_quote_5=101.0,
        bid_depth_quote_10=99.0,
        ask_depth_quote_10=101.0,
        bid_depth_quote_20=99.0,
        ask_depth_quote_20=101.0,
        imbalance_5=0.0,
        imbalance_10=0.0,
        imbalance_20=0.0,
    )


def _ineligible_outcome(anchor: object, horizon: int, side: str) -> object:
    result = Round74EventTargetOutcome(
        symbol=str(anchor.symbol),
        anchor_index=int(anchor.anchor_index),
        horizon_seconds=horizon,
        side=side,
        eligible=False,
        ineligible_reason="coverage_end",
        requested_entry_monotonic_ns=int(anchor.decision_monotonic_ns) + 1,
        actual_entry_monotonic_ns=None,
        actual_entry_frame_index=None,
        actual_entry_message_index=None,
        requested_exit_monotonic_ns=None,
        actual_exit_monotonic_ns=None,
        actual_exit_frame_index=None,
        actual_exit_message_index=None,
        base_quantity=None,
        reference_quote_notional=None,
        entry_quote_notional=None,
        entry_average_price=None,
        exit_average_price=None,
        midpoint_payoff_bps=None,
        book_walk_implementation_shortfall_quote=None,
        book_walk_implementation_shortfall_bps=None,
        gross_payoff_bps=None,
        commission_quote=None,
        commission_bps=None,
        additional_slippage_quote=None,
        additional_slippage_bps=None,
        explicit_cost_quote=None,
        explicit_cost_bps=None,
        total_implementation_shortfall_quote=None,
        total_implementation_shortfall_bps=None,
        net_payoff_bps=None,
        capital_scaled_net_payoff_bps=None,
        positive_net_payoff=None,
        maximum_adverse_excursion_bps=None,
        capital_scaled_maximum_adverse_excursion_bps=None,
        maximum_favorable_excursion_bps=None,
        adverse_selection=None,
        regime_unpredictability=None,
        maximum_spread_bps=None,
        minimum_exit_side_capacity_ratio=None,
        entry_update_id=None,
        exit_update_id=None,
        target_spec_sha256=SPEC_SHA256,
        target_context_sha256="8" * 64,
        feature_window_sha256=str(anchor.feature_window_sha256),
    )
    result.validate()
    return result


def test_delayed_execution_replays_three_profiles_in_one_observation_pass(
    monkeypatch,
) -> None:
    rows = 3
    batch = SimpleNamespace(
        role="tuning",
        run_id=(RUN_ID,) * rows,
        symbol=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        anchor_index=np.arange(rows, dtype=np.int64),
        decision_monotonic_ns=np.arange(1, rows + 1, dtype=np.int64) * 1_000,
        decision_wall_ns=np.arange(1, rows + 1, dtype=np.int64) * 2_000,
        endpoint_frame_index=np.arange(rows, dtype=np.int64),
        endpoint_message_index=np.zeros(rows, dtype=np.int64),
        feature_window_sha256=tuple(f"{10 + index:064x}" for index in range(rows)),
        partition_sha256=PARTITION_SHA256,
        scaler_sha256=SCALER_SHA256,
        batch_sha256="9" * 64,
        rows=rows,
        validate=lambda: None,
    )
    context = SimpleNamespace(
        feature_row_sha256=tuple(f"{20 + index:064x}" for index in range(rows))
    )
    monkeypatch.setattr(
        subject,
        "build_round74_action_inference_context",
        lambda selected: context if selected is batch else None,
    )
    assembly = SimpleNamespace(
        spec=SimpleNamespace(spec_sha256=SPEC_SHA256),
        assembly_sha256=ASSEMBLY_SHA256,
        quantity_rules_mapping=lambda: {},
        __post_init__=lambda: None,
    )
    latency = SimpleNamespace(
        scaler_sha256=SCALER_SHA256,
        evidence_sha256=LATENCY_SHA256,
        profiles=tuple(
            SimpleNamespace(
                profile=profile,
                p99_upper_confidence_ns=(index + 1) * 1_000_000,
            )
            for index, profile in enumerate(ROUND74_ACTION_PROFILES)
        ),
        validate=lambda: None,
    )
    engines = []

    class _Engine:
        def __init__(self, *, anchors, execution_overrides, **_kwargs) -> None:
            self.anchors = anchors
            self.execution_overrides = execution_overrides
            self.observations = 0
            engines.append(self)

        def observe_depth(self, **_kwargs) -> None:
            self.observations += 1

        def finish(self):
            return tuple(
                _ineligible_outcome(anchor, horizon, side)
                for anchor in self.anchors
                for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
                for side in ROUND74_EVENT_PAYOFF_SIDES
            )

    monkeypatch.setattr(subject, "Round74EventTargetEngine", _Engine)
    stream = _OnePass(
        (
            Round74ReplayObservation(
                symbol="BTCUSDT",
                event_type="depthUpdate",
                frame_index=1,
                message_index=0,
                received_monotonic_ns=1,
                received_wall_ns=1,
                token=None,
                depth_state=_state(),
                depth_update_is_stale=False,
            ),
        )
    )

    result = subject.replay_round74_delayed_execution_run(
        batch,
        assembly,
        latency,
        stream,
        capture_report_sha256=CAPTURE_SHA256,
    )

    assert stream.iterations == 1
    assert tuple(item.profile for item in result) == ROUND74_ACTION_PROFILES
    assert len(engines) == len(ROUND74_ACTION_PROFILES)
    assert all(engine.observations == 1 for engine in engines)
    assert tuple(item.additional_entry_latency_ns for item in result) == (
        1_000_000,
        2_000_000,
        3_000_000,
    )
    assert all(len(item.rows) == rows for item in result)
    assert all(
        len(engine.execution_overrides) == rows
        for engine in engines
    )
