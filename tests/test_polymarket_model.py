"""Contract tests for leakage-safe Polymarket probability research."""

from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simple_ai_trading.ai_uplift import assess_ai_uplift
from simple_ai_trading.cli import (
    _polymarket_execution_uplift_metrics,
    _polymarket_held_out_prediction_evidence,
    _polymarket_latency_scenarios,
    _polymarket_matched_uplift_periods,
)
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_ai_veto import (
    PolymarketAIVetoConfig,
    benchmark_polymarket_ai_veto,
    build_polymarket_ai_veto_cases,
)
from simple_ai_trading.polymarket_features import (
    POLYMARKET_FEATURE_NAMES,
    PolymarketFeatureConfig,
    PolymarketFeatureDataset,
    PolymarketFeatureRow,
)
from simple_ai_trading.polymarket_model import (
    POLYMARKET_MODEL_FEATURE_NAMES,
    PolymarketModelConfig,
    build_polymarket_model_dataset,
    fit_polymarket_offset_model,
    predict_polymarket_probabilities,
    split_polymarket_model_dataset,
)
from simple_ai_trading.polymarket_model_execution import (
    PolymarketExecutionResearchConfig,
    build_polymarket_policy_selection,
    evaluate_polymarket_execution_policy,
)
from simple_ai_trading.polymarket_publication import (
    _ai_case_rows,
    _validate_ai_evidence,
    publish_polymarket_model_artifact,
    validate_polymarket_model_artifact,
)
from simple_ai_trading.polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketRecordedBook,
    PolymarketReplayDiagnostics,
    PolymarketResolutionEvidence,
)

_ASSETS = ("BTC", "ETH", "SOL")
_SOURCE_SHA256 = "a" * 64


def _market(asset: str, group: int) -> PolymarketFiveMinuteMarket:
    start_ms = 1_800_000_000_000 + group * 300_000
    condition = f"0x{group:062x}{_ASSETS.index(asset):02x}"
    return PolymarketFiveMinuteMarket(
        asset=asset,
        market_id=f"market-{asset.lower()}-{group}",
        condition_id=condition,
        slug=f"{asset.lower()}-updown-5m-{start_ms // 1_000}",
        question=f"Will {asset} be up?",
        event_start_ms=start_ms,
        end_ms=start_ms + 300_000,
        up_token_id=f"1{group:020d}{_ASSETS.index(asset)}",
        down_token_id=f"2{group:020d}{_ASSETS.index(asset)}",
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("5"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("1000000"),
        resolution_source=f"https://data.chain.link/streams/{asset.lower()}-usd",
        gamma_payload_sha256="b" * 64,
        gamma_payload_json="{}",
    )


def _feature_values(
    *,
    remaining_seconds: int,
    official_up: bool,
    predictive: bool,
) -> tuple[float, ...]:
    values = dict.fromkeys(POLYMARKET_FEATURE_NAMES, 0.0)
    values.update(
        {
            "elapsed_fraction": (300 - remaining_seconds) / 300,
            "remaining_seconds": float(remaining_seconds),
            "up_best_bid": 0.48,
            "up_best_ask": 0.50,
            "up_midpoint": 0.49,
            "up_spread": 0.02,
            "up_microprice": 0.492,
            "up_top_imbalance": 0.10 if official_up and predictive else 0.0,
            "up_bid_depth_3": 500.0,
            "up_ask_depth_3": 500.0,
            "down_best_bid": 0.50,
            "down_best_ask": 0.52,
            "down_midpoint": 0.51,
            "down_spread": 0.02,
            "down_microprice": 0.508,
            "down_top_imbalance": -0.10 if official_up and predictive else 0.0,
            "down_bid_depth_3": 500.0,
            "down_ask_depth_3": 500.0,
            "ask_pair_cost": 1.02,
            "bid_pair_value": 0.98,
            "up_book_age_ms": 5.0,
            "down_book_age_ms": 5.0,
            "chainlink_return_from_open_bps": 0.0,
            "binance_distance_from_chainlink_open_bps": (
                (50.0 if official_up else -50.0) if predictive else 0.0
            ),
            "binance_chainlink_basis_bps": (
                (10.0 if official_up else -10.0) if predictive else 0.0
            ),
            "binance_best_bid": 100.0,
            "binance_best_ask": 100.01,
            "binance_spread_bps": 1.0,
            "binance_top_imbalance": (
                (0.5 if official_up else -0.5) if predictive else 0.0
            ),
            "binance_return_100ms_bps": (
                (2.0 if official_up else -2.0) if predictive else 0.0
            ),
            "binance_return_250ms_bps": (
                (5.0 if official_up else -5.0) if predictive else 0.0
            ),
            "binance_return_1000ms_bps": (
                (10.0 if official_up else -10.0) if predictive else 0.0
            ),
            "binance_return_5000ms_bps": (
                (20.0 if official_up else -20.0) if predictive else 0.0
            ),
            "binance_realized_volatility_100ms_bps": 1.0,
            "binance_realized_volatility_1000ms_bps": 2.0,
            "binance_realized_volatility_5000ms_bps": 5.0,
            "binance_trade_imbalance_100ms": (
                (0.30 if official_up else -0.30) if predictive else 0.0
            ),
            "binance_trade_imbalance_250ms": (
                (0.25 if official_up else -0.25) if predictive else 0.0
            ),
            "binance_trade_imbalance_1000ms": (
                (0.20 if official_up else -0.20) if predictive else 0.0
            ),
            "binance_trade_imbalance_5000ms": (
                (0.15 if official_up else -0.15) if predictive else 0.0
            ),
            "log1p_binance_trade_quote_250ms": 5.0,
            "log1p_binance_trade_quote_1000ms": 6.0,
            "log1p_binance_trade_quote_5000ms": 7.0,
            "direct_binance_age_ms": 10.0,
            "chainlink_source_age_ms": 100.0,
            "chainlink_arrival_age_ms": 100.0,
            "chainlink_anchor_gap_ms": 0.0,
            "log1p_market_liquidity_quote": 11.5,
            "log1p_market_volume_quote": 13.8,
        }
    )
    return tuple(float(values[name]) for name in POLYMARKET_FEATURE_NAMES)


def _source_fixture(
    *,
    groups: int = 30,
    predictive: bool = True,
    omit_last_horizon: bool = False,
) -> tuple[PolymarketFeatureDataset, tuple[PolymarketFiveMinuteMarket, ...]]:
    markets: list[PolymarketFiveMinuteMarket] = []
    rows: list[PolymarketFeatureRow] = []
    horizons = (240, 180, 120, 60, 30)
    for group in range(groups):
        official_up = group % 2 == 0
        for asset in _ASSETS:
            market = _market(asset, group)
            markets.append(market)
            selected_horizons = (
                horizons[:-1] if omit_last_horizon and group == 0 else horizons
            )
            for horizon in selected_horizons:
                identity = f"{group}-{asset}-{horizon}"
                wall_ms = market.end_ms - horizon * 1_000
                rows.append(
                    PolymarketFeatureRow(
                        feature_id=(identity.encode().hex() + "0" * 64)[:64],
                        run_id="fixture-run",
                        condition_id=market.condition_id,
                        market_id=market.market_id,
                        asset=asset,
                        decision_event_id=f"event-{identity}",
                        decision_received_wall_ms=wall_ms,
                        decision_received_monotonic_ns=wall_ms * 1_000_000,
                        feature_values=_feature_values(
                            remaining_seconds=horizon,
                            official_up=official_up,
                            predictive=predictive,
                        ),
                        official_up=official_up,
                        resolution_event_id=f"resolution-{group}-{asset}",
                        input_provenance_sha256="c" * 64,
                        row_sha256="d" * 64,
                    )
                )
    source = PolymarketFeatureDataset(
        dataset_id=_SOURCE_SHA256,
        run_id="fixture-run",
        config=PolymarketFeatureConfig(minimum_resolved_markets_per_asset=3),
        rows=tuple(rows),
        candidate_count=len(rows),
        skipped_counts={},
        labeled_market_counts={asset: groups for asset in _ASSETS},
        shadow_errors=(),
        training_errors=(),
        replay_diagnostics={},
        coverage={},
        dataset_sha256=_SOURCE_SHA256,
    )
    return source, tuple(markets)


def _replay_fixture(
    samples: tuple,
    markets: tuple[PolymarketFiveMinuteMarket, ...],
    *,
    displayed_quantity: Decimal = Decimal("100"),
    execution_latency_ms: int = 100,
    execution_ask_offset: Decimal = Decimal("0"),
) -> PolymarketEvidenceReplay:
    selected_conditions = {item.condition_id for item in samples}
    selected_markets = tuple(
        market for market in markets if market.condition_id in selected_conditions
    )
    market_by_condition = {market.condition_id: market for market in selected_markets}
    books: list[PolymarketRecordedBook] = []
    sequence = 0
    for sample in samples:
        market = market_by_condition[sample.condition_id]
        for outcome, token_id, bid, ask in (
            (
                "Up",
                market.up_token_id,
                Decimal(str(sample.up_best_bid)),
                Decimal(str(sample.up_best_ask)),
            ),
            (
                "Down",
                market.down_token_id,
                Decimal(str(sample.down_best_bid)),
                Decimal(str(sample.down_best_ask)),
            ),
        ):
            for phase, latency_ms in (
                ("decision", 0),
                ("execution", execution_latency_ms),
            ):
                sequence += 1
                wall_ms = sample.decision_received_wall_ms + latency_ms
                phase_ask = ask + execution_ask_offset if phase == "execution" else ask
                snapshot = PaperBookSnapshot(
                    venue="polymarket",
                    market_id=market.market_id,
                    asset_id=token_id,
                    bids=(BookLevel(bid, displayed_quantity),),
                    asks=(BookLevel(phase_ask, displayed_quantity),),
                    source_time_ms=wall_ms,
                    received_wall_ms=wall_ms,
                    received_monotonic_ns=(
                        sample.decision_received_monotonic_ns + latency_ms * 1_000_000
                    ),
                    source_payload_sha256=f"{sequence:064x}",
                )
                books.append(
                    PolymarketRecordedBook(
                        run_id="fixture-run",
                        event_id=(
                            f"book-{sample.sample_id[:12]}-{outcome.lower()}-{phase}"
                        ),
                        event_type="book",
                        connection_id=f"connection-{market.condition_id[-8:]}",
                        segment_id=f"segment-{market.condition_id[-8:]}",
                        sequence_number=sequence,
                        sub_index=0,
                        market=market,
                        outcome=outcome,
                        tick_size=market.tick_size,
                        snapshot=snapshot,
                    )
                )
    books.sort(
        key=lambda item: (
            item.received_monotonic_ns,
            item.sequence_number,
            item.event_id,
        )
    )
    resolutions = tuple(
        PolymarketResolutionEvidence(
            run_id="fixture-run",
            event_id=next(
                item.resolution_event_id
                for item in samples
                if item.condition_id == market.condition_id
            ),
            condition_id=market.condition_id,
            winning_asset_id=(
                market.up_token_id
                if next(
                    item.official_up
                    for item in samples
                    if item.condition_id == market.condition_id
                )
                else market.down_token_id
            ),
            winning_outcome=(
                "Up"
                if next(
                    item.official_up
                    for item in samples
                    if item.condition_id == market.condition_id
                )
                else "Down"
            ),
            resolved_at_ms=market.end_ms + 1_000,
            received_wall_ms=market.end_ms + 1_000,
            received_monotonic_ns=(market.end_ms + 1_000) * 1_000_000,
            event_sha256="e" * 64,
            source="official_fixture",
        )
        for market in selected_markets
    )
    return PolymarketEvidenceReplay(
        run_id="fixture-run",
        markets=selected_markets,
        books=tuple(books),
        resolutions=resolutions,
        diagnostics=PolymarketReplayDiagnostics(
            schema_version="polymarket-replay-diagnostics-v2",
            continuity_mode="strict",
            stream_gap_count=0,
            clob_connection_segment_count=len(selected_markets),
            state_reset_count=0,
            discarded_uncorroborated_best_count=0,
            book_sample_interval_ms=0,
            book_state_transition_count=len(books),
            materialized_book_count=len(books),
            suppressed_book_count=0,
            total_event_count=len(books),
            causally_ordered_event_count=len(books),
            late_event_count=0,
            maximum_source_regression_ms=0,
            maximum_late_arrival_delay_ns=0,
            deferred_event_count=0,
            maximum_availability_delay_ns=0,
        ),
    )


def test_market_grouping_equal_weights_and_purged_split() -> None:
    source, markets = _source_fixture()
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)

    assert dataset.training_ready
    assert len(dataset.samples) == 30 * 3 * 5
    assert dataset.market_counts == {"BTC": 30, "ETH": 30, "SOL": 30}
    assert dataset.time_group_count == 30
    assert len(POLYMARKET_MODEL_FEATURE_NAMES) == 27
    up_features = next(item for item in dataset.samples if item.official_up).feature_map()
    down_features = next(
        item for item in dataset.samples if not item.official_up
    ).feature_map()
    for name in (
        "direct_diffusion_market_logit_gap",
        "chainlink_diffusion_market_logit_gap",
    ):
        assert up_features[name] > 0.0
        assert down_features[name] < 0.0
    risk_context = dataset.samples[0].risk_context_map()
    assert risk_context["direct_binance_age_ms"] == 10.0
    assert risk_context["up_ask_depth_3_contracts"] == 500.0
    assert risk_context["log1p_market_liquidity_quote"] == 11.5
    for condition in {item.condition_id for item in dataset.samples}:
        assert sum(
            item.market_weight
            for item in dataset.samples
            if item.condition_id == condition
        ) == pytest.approx(1.0)

    role_conditions = [
        {item.condition_id for item in values}
        for values in (split.train, split.validation, split.test)
    ]
    assert not role_conditions[0] & role_conditions[1]
    assert not role_conditions[0] & role_conditions[2]
    assert not role_conditions[1] & role_conditions[2]
    for group_start in split.train_group_starts_ms:
        assert {
            item.asset for item in split.train if item.event_start_ms == group_start
        } == set(_ASSETS)
    assert set(split.purged_group_starts_ms).isdisjoint(
        set(split.train_group_starts_ms)
        | set(split.validation_group_starts_ms)
        | set(split.test_group_starts_ms)
    )


def test_fit_is_deterministic_bounded_and_has_no_trading_authority() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)

    first_model, first_report = fit_polymarket_offset_model(dataset, split)
    second_model, second_report = fit_polymarket_offset_model(dataset, split)

    assert first_model == second_model
    assert first_report == second_report
    assert first_model.selected_candidate != "market_baseline"
    assert first_model.selected_candidate == first_model.inner_selected_candidate
    assert first_model.inner_fold_count == 3
    assert len(first_model.candidate_inner_log_losses) == 6
    assert [name for name, _loss in first_model.validation_gate_log_losses] == [
        "market_baseline",
        first_model.inner_selected_candidate,
    ]
    for (
        train_start,
        train_end,
        validation_start,
        validation_end,
    ) in first_model.inner_fold_boundaries_ms:
        assert train_start <= train_end < validation_start <= validation_end
        assert validation_end <= max(split.train_group_starts_ms)
        assert validation_end < min(split.validation_group_starts_ms)
    assert first_report.validation_log_loss_delta < 0.0
    assert first_report.test_log_loss_delta < 0.0
    assert not first_model.trading_authority
    assert not first_model.execution_claim
    assert not first_model.profitability_claim
    assert not first_report.trading_authority
    predictions = predict_polymarket_probabilities(first_model, split.test)
    assert predictions.shape == (len(split.test),)
    assert np.all((predictions > 0.0) & (predictions < 1.0))
    baseline_logits = np.log(
        np.asarray([item.baseline_up_probability for item in split.test])
        / (1.0 - np.asarray([item.baseline_up_probability for item in split.test]))
    )
    predicted_logits = np.log(predictions / (1.0 - predictions))
    assert float(np.max(np.abs(predicted_logits - baseline_logits))) <= (
        first_model.config.maximum_absolute_logit_correction + 1e-12
    )
    tampered_context = replace(
        split.test[0],
        risk_context_values=(999.0, *split.test[0].risk_context_values[1:]),
    )
    with pytest.raises(ValueError, match="sample identity is invalid"):
        predict_polymarket_probabilities(first_model, (tampered_context,))


def test_unjustified_correction_falls_back_to_market_probability() -> None:
    source, markets = _source_fixture(predictive=False)
    config = PolymarketModelConfig(
        minimum_validation_log_loss_improvement=0.01,
    )
    dataset = build_polymarket_model_dataset(source, markets, config=config)
    split = split_polymarket_model_dataset(dataset)

    model, report = fit_polymarket_offset_model(dataset, split)

    assert model.selected_candidate == "market_baseline"
    assert model.selected_l2 is None
    assert model.inner_selected_candidate.startswith("offset_l2_")
    assert set(model.coefficients) == {0.0}
    assert report.validation_log_loss_delta == pytest.approx(0.0)
    assert report.test_log_loss_delta == pytest.approx(0.0)


def test_execution_policy_uses_depth_latency_fees_risk_and_official_settlement() -> (
    None
):
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)

    first = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
    )
    second = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
    )

    assert first == second
    assert first.filled_order_count == first.evaluated_market_count
    assert first.winning_order_count == first.filled_order_count
    assert first.losing_order_count == 0
    assert first.net_realized_pnl_quote > 0
    assert first.total_fees_quote > 0
    assert first.maximum_drawdown_quote == 0
    assert len({trade.condition_id for trade in first.trades}) == len(first.trades)
    assert all(trade.official_resolution_event_id for trade in first.trades)
    assert all(trade.execution_book_event_id for trade in first.trades)
    assert not first.trading_authority
    assert not first.execution_claim
    assert not first.profitability_claim


def test_execution_policy_abstains_for_market_baseline_without_edge() -> None:
    source, markets = _source_fixture(predictive=False)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    replay = _replay_fixture(split.test, markets)
    baseline = np.asarray(
        [item.baseline_up_probability for item in split.test],
        dtype=np.float64,
    )

    report = evaluate_polymarket_execution_policy(split.test, baseline, replay)

    assert report.signal_market_count == 0
    assert report.attempted_order_count == 0
    assert report.filled_order_count == 0
    assert report.net_realized_pnl_quote == 0
    assert len(report.equity_curve) == len(split.test_group_starts_ms)
    assert all(point.group_realized_pnl_quote == 0 for point in report.equity_curve)
    assert report.reason_counts == {
        "no_positive_after_cost_edge": report.evaluated_market_count
    }


def test_execution_uses_only_the_latest_book_available_at_arrival() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(
        split.test,
        markets,
        execution_latency_ms=200,
        execution_ask_offset=Decimal("0.45"),
    )

    report = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=PolymarketExecutionResearchConfig(submission_latency_ms=100),
    )

    assert report.filled_order_count == report.evaluated_market_count
    assert all(trade.effective_latency_ms == 100 for trade in report.trades)
    assert all(
        "execution" not in trade.execution_book_event_id for trade in report.trades
    )


def test_execution_charges_ai_decision_delay_before_network_latency() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(
        split.test,
        markets,
        execution_latency_ms=200,
        execution_ask_offset=Decimal("0.45"),
    )
    conditions = {item.condition_id for item in split.test}
    delays = {condition: 100 for condition in conditions}

    delayed = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=PolymarketExecutionResearchConfig(submission_latency_ms=100),
        decision_delay_ms_by_condition=delays,
    )

    assert delayed.signal_market_count == delayed.evaluated_market_count
    assert delayed.filled_order_count == 0
    assert delayed.attempted_order_count == delayed.evaluated_market_count
    assert all(trade.decision_delay_ms == 100 for trade in delayed.trades)
    assert all(trade.submission_latency_ms == 100 for trade in delayed.trades)
    assert all(trade.effective_latency_ms == 200 for trade in delayed.trades)
    assert all("execution" in trade.execution_book_event_id for trade in delayed.trades)
    with pytest.raises(ValueError, match="decision delays must bind every"):
        evaluate_polymarket_execution_policy(
            split.test,
            predictions,
            replay,
            decision_delay_ms_by_condition={},
        )


def test_execution_policy_refuses_unresolved_or_mismatched_labels() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    replay = _replay_fixture(split.test, markets)
    replay.resolutions = replay.resolutions[:-1]
    probabilities = np.asarray([0.9] * len(split.test), dtype=np.float64)

    with pytest.raises(ValueError, match="lacks official replay evidence"):
        evaluate_polymarket_execution_policy(split.test, probabilities, replay)


def test_execution_policy_never_credits_insufficient_displayed_depth() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(
        split.test,
        markets,
        displayed_quantity=Decimal("1"),
    )

    report = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
    )

    assert report.signal_market_count == report.evaluated_market_count
    assert report.filled_order_count == 0
    assert report.net_realized_pnl_quote == 0
    assert report.reason_counts["insufficient_displayed_depth_for_fok"] == (
        report.attempted_order_count
    )


def test_execution_policy_binds_external_fail_closed_permissions() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)
    permissions = {
        condition: False for condition in {item.condition_id for item in split.test}
    }

    report = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        market_permissions=permissions,
    )

    assert report.signal_market_count == 0
    assert report.filled_order_count == 0
    assert report.reason_counts == {
        "external_fail_closed_veto": report.evaluated_market_count
    }
    assert len(report.equity_curve) == len(split.test_group_starts_ms)
    assert len(report.market_permission_sha256) == 64
    with pytest.raises(ValueError, match="permissions must bind every"):
        evaluate_polymarket_execution_policy(
            split.test,
            predictions,
            replay,
            market_permissions={},
        )


def test_ai_veto_cases_are_pre_execution_label_free_and_hash_bound() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, probability_report = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    execution_config = PolymarketExecutionResearchConfig().validated()
    selection = build_polymarket_policy_selection(
        split.test,
        predictions,
        tuple(
            market
            for market in markets
            if market.condition_id in {item.condition_id for item in split.test}
        ),
        config=execution_config,
    )

    first = build_polymarket_ai_veto_cases(
        selection,
        probability_report,
        execution_config,
    )
    second = build_polymarket_ai_veto_cases(
        selection,
        probability_report,
        execution_config,
    )

    assert first == second
    assert len(first) == len(selection.candidates)
    serialized_prompts = json.dumps(
        [case.prompt_payload for case in first],
        sort_keys=True,
    ).lower()
    assert "official_up" not in serialized_prompts
    assert "resolution" not in serialized_prompts
    assert "realized_pnl" not in serialized_prompts
    assert "outcome_net" not in serialized_prompts
    assert "source_freshness_ms" in first[0].prompt_payload
    assert "liquidity_context" in first[0].prompt_payload
    microstructure = first[0].prompt_payload["microstructure"]
    assert isinstance(microstructure, dict)
    assert "direct_return_100ms_bps" in microstructure
    assert "direct_diffusion_market_logit_gap" in microstructure
    assert all(len(case.case_sha256) == 64 for case in first)


def test_ai_veto_permissions_are_fail_closed_and_execution_bound() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, probability_report = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)
    execution_config = PolymarketExecutionResearchConfig().validated()
    selection = build_polymarket_policy_selection(
        split.test,
        predictions,
        replay.markets,
        config=execution_config,
    )
    cases = build_polymarket_ai_veto_cases(
        selection,
        probability_report,
        execution_config,
    )

    def approve(
        url: str,
        payload: dict[str, object],
        _timeout: float,
        _method: str,
    ) -> object:
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b", "digest": "f" * 64}]}
        if url.endswith("/api/show"):
            return {"model": "qwen3.5:9b", "parameters": "9B"}
        assert payload["think"] is False
        return {
            "message": {
                "content": json.dumps(
                    {
                        "action": "approve",
                        "confidence": 0.9,
                        "reason_codes": ["edge_after_fees"],
                        "summary": "Causal evidence and after-fee edge are coherent.",
                    }
                )
            }
        }

    ai_report = benchmark_polymarket_ai_veto(
        cases,
        all_condition_ids=[item.condition_id for item in split.test],
        selection_sha256=selection.selection_sha256,
        risk_benchmark_evidence_sha256="a" * 64,
        config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
        post_json=approve,  # type: ignore[arg-type]
    )
    baseline = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=execution_config,
    )
    approved = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=execution_config,
        market_permissions=ai_report.market_permissions,
    )

    assert ai_report.approval_count == len(cases)
    assert ai_report.provider_failure_count == 0
    assert ai_report.model_parameters_b == 9.0
    assert ai_report.market_permission_sha256 == approved.market_permission_sha256
    assert baseline == approved
    assert not ai_report.trading_authority
    assert not ai_report.profitability_claim


def test_ai_prompt_publication_rejects_rehashed_label_injection() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, probability_report = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)
    execution_config = PolymarketExecutionResearchConfig().validated()
    selection = build_polymarket_policy_selection(
        split.test,
        predictions,
        replay.markets,
        config=execution_config,
    )
    cases = build_polymarket_ai_veto_cases(
        selection,
        probability_report,
        execution_config,
    )

    def approve(
        url: str,
        _payload: dict[str, object],
        _timeout: float,
        _method: str,
    ) -> object:
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b", "digest": "f" * 64}]}
        if url.endswith("/api/show"):
            return {"model": "qwen3.5:9b", "parameters": "9B"}
        return {
            "message": {
                "content": json.dumps(
                    {
                        "action": "approve",
                        "confidence": 0.9,
                        "reason_codes": ["edge_after_fees"],
                        "summary": "The frozen proposal passes the veto-only review.",
                    }
                )
            }
        }

    ai_report = benchmark_polymarket_ai_veto(
        cases,
        all_condition_ids=[item.condition_id for item in split.test],
        selection_sha256=selection.selection_sha256,
        risk_benchmark_evidence_sha256="a" * 64,
        config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
        post_json=approve,  # type: ignore[arg-type]
    )
    model_execution = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=execution_config,
    )
    delays = {condition: 0 for condition in model_execution.market_permissions}
    for result in ai_report.results:
        delays[result.condition_id] = int(
            np.ceil(max(0.0, result.latency_seconds) * 1_000.0)
        )
    ai_execution = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=execution_config,
        market_permissions=ai_report.market_permissions,
        decision_delay_ms_by_condition=delays,
    )
    prediction_rows = _polymarket_held_out_prediction_evidence(
        split.test,
        [item.baseline_up_probability for item in split.test],
        predictions,
    )["rows"]
    assert isinstance(prediction_rows, list)
    uplift = assess_ai_uplift(
        _polymarket_execution_uplift_metrics(
            model_execution,
            dataset_fingerprint=dataset.dataset_sha256,
        ),
        _polymarket_execution_uplift_metrics(
            ai_execution,
            dataset_fingerprint=dataset.dataset_sha256,
        ),
        model_name="qwen3.5:9b",
        model_parameters_b=ai_report.model_parameters_b,
        model_artifact_sha256=ai_report.report_sha256,
        matched_periods=_polymarket_matched_uplift_periods(
            split,
            model_execution,
            ai_execution,
        ),
    )
    ai_evidence: dict[str, object] = {
        "enabled": True,
        "risk_benchmark": {
            "path": "docs/model-research/polymarket/latest/ai-risk-selected.json",
            "sha256": "a" * 64,
            "contract": "finance-risk-review-adversarial-v6",
            "selected_model": "qwen3.5:9b",
            "score": 1.0,
        },
        "policy_selection": selection.asdict(),
        "prompt_cases": [case.asdict() for case in cases],
        "veto_report": ai_report.asdict(),
        "execution": ai_execution.asdict(),
        "uplift": uplift.asdict(),
    }
    assert (
        _validate_ai_evidence(
            ai_evidence,
            predictions=prediction_rows,
            probability=probability_report.asdict(),
            model_execution=model_execution.asdict(),
        )
        == ai_execution.asdict()
    )
    case_rows = _ai_case_rows({"ai": ai_evidence})
    assert len(case_rows) == len(cases)
    assert all("official_up" not in row["prompt_payload_json"] for row in case_rows)

    forged_uplift = json.loads(json.dumps(ai_evidence))
    forged_uplift["uplift"]["accepted"] = not forged_uplift["uplift"]["accepted"]
    with pytest.raises(ValueError, match="AI uplift evidence"):
        _validate_ai_evidence(
            forged_uplift,
            predictions=prediction_rows,
            probability=probability_report.asdict(),
            model_execution=model_execution.asdict(),
        )

    forged_response = json.loads(json.dumps(ai_evidence))
    forged_result = forged_response["veto_report"]["results"][0]
    forged_result["response_payload"]["message"]["content"] = json.dumps(
        {
            "action": "veto",
            "confidence": 0.9,
            "reason_codes": ["insufficient_evidence"],
            "summary": "This response contradicts the recorded parsed decision.",
        }
    )
    forged_result["response_sha256"] = hashlib.sha256(
        json.dumps(
            forged_result["response_payload"],
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    forged_report_identity = dict(forged_response["veto_report"])
    forged_report_identity.pop("report_sha256")
    forged_response["veto_report"]["report_sha256"] = hashlib.sha256(
        json.dumps(
            forged_report_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="AI veto result"):
        _validate_ai_evidence(
            forged_response,
            predictions=prediction_rows,
            probability=probability_report.asdict(),
            model_execution=model_execution.asdict(),
        )

    tampered = json.loads(json.dumps(ai_evidence))
    prompt_case = tampered["prompt_cases"][0]
    prompt_case["prompt_payload"]["official_up"] = True
    prompt_case["case_id"] = hashlib.sha256(
        json.dumps(
            {
                "selection_sha256": tampered["policy_selection"][
                    "selection_sha256"
                ],
                "model_report_sha256": probability_report.report_sha256,
                "sample_id": prompt_case["sample_id"],
                "prompt_payload": prompt_case["prompt_payload"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    prompt_identity = dict(prompt_case)
    prompt_identity.pop("case_sha256")
    prompt_case["case_sha256"] = hashlib.sha256(
        json.dumps(
            prompt_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    veto_report = tampered["veto_report"]
    veto_report["case_set_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "schema_version": "polymarket-ai-veto-case-v2",
                "selection_sha256": tampered["policy_selection"][
                    "selection_sha256"
                ],
                "case_sha256": [
                    item["case_sha256"] for item in tampered["prompt_cases"]
                ],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    veto_report["results"][0]["case_id"] = prompt_case["case_id"]
    report_identity = dict(veto_report)
    report_identity.pop("report_sha256")
    veto_report["report_sha256"] = hashlib.sha256(
        json.dumps(
            report_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()

    with pytest.raises(ValueError, match="AI prompt fields"):
        _validate_ai_evidence(
            tampered,
            predictions=prediction_rows,
            probability=probability_report.asdict(),
            model_execution=model_execution.asdict(),
        )


def test_ai_low_confidence_or_malformed_output_vetoes_without_order_authority() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, probability_report = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)
    execution_config = PolymarketExecutionResearchConfig().validated()
    selection = build_polymarket_policy_selection(
        split.test,
        predictions,
        replay.markets,
        config=execution_config,
    )
    cases = build_polymarket_ai_veto_cases(
        selection,
        probability_report,
        execution_config,
    )
    chat_calls = 0

    def uncertain(
        url: str,
        _payload: dict[str, object],
        _timeout: float,
        _method: str,
    ) -> object:
        nonlocal chat_calls
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b", "digest": "f" * 64}]}
        if url.endswith("/api/show"):
            return {"model": "qwen3.5:9b", "parameters": "9B"}
        chat_calls += 1
        if chat_calls == 1:
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "action": "approve",
                            "confidence": 0.1,
                            "reason_codes": ["edge_after_fees"],
                            "summary": "Uncertain.",
                        }
                    )
                }
            }
        return {"message": {"content": "not-json"}}

    ai_report = benchmark_polymarket_ai_veto(
        cases[:2],
        all_condition_ids=[item.condition_id for item in split.test],
        selection_sha256=selection.selection_sha256,
        risk_benchmark_evidence_sha256="a" * 64,
        config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
        post_json=uncertain,  # type: ignore[arg-type]
    )

    assert ai_report.approval_count == 0
    assert ai_report.provider_failure_count == 2
    assert all(
        not ai_report.market_permissions[case.condition_id] for case in cases[:2]
    )
    assert all(not result.decision.valid for result in ai_report.results)


def test_invalid_execution_risk_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        PolymarketExecutionResearchConfig(
            maximum_loss_fraction_per_market=Decimal("0.02"),
            maximum_loss_fraction_per_time_group=Decimal("0.01"),
        ).validated()


def test_incomplete_market_is_excluded_and_training_fails_closed() -> None:
    source, markets = _source_fixture(groups=10, omit_last_horizon=True)
    config = PolymarketModelConfig(
        minimum_markets_per_asset=10,
        minimum_time_groups=10,
        minimum_train_time_groups=15,
        minimum_validation_time_groups=2,
        minimum_test_time_groups=2,
        minimum_outcome_markets_per_split=1,
    )
    dataset = build_polymarket_model_dataset(source, markets, config=config)

    assert not dataset.training_ready
    assert dataset.skipped_counts == {"incomplete_fixed_horizons": 3}
    assert "insufficient_model_markets:BTC:9/10" in dataset.training_errors
    assert "insufficient_model_markets:ETH:9/10" in dataset.training_errors
    assert "insufficient_model_markets:SOL:9/10" in dataset.training_errors
    with pytest.raises(ValueError, match="not training-ready"):
        split_polymarket_model_dataset(dataset)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision_horizons_seconds": (30, 60)},
        {"l2_candidates": (1.0, 0.1)},
        {"validation_fraction": 0.4},
        {"maximum_absolute_logit_correction": 10.0},
    ],
)
def test_invalid_model_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        PolymarketModelConfig(**kwargs).validated()


def test_latency_scenarios_are_predeclared_bounded_and_include_primary() -> None:
    assert _polymarket_latency_scenarios(
        "1000,50,250,50",
        primary_latency_ms=100,
    ) == (50, 100, 250, 1000)
    with pytest.raises(ValueError, match="must be integers"):
        _polymarket_latency_scenarios("50,fast", primary_latency_ms=100)
    with pytest.raises(ValueError, match="1-12 values"):
        _polymarket_latency_scenarios("0,50", primary_latency_ms=100)


def test_polymarket_publication_is_derived_and_tamper_evident(
    tmp_path: Path,
) -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, probability_report = fit_polymarket_offset_model(dataset, split)
    baseline_probabilities = [item.baseline_up_probability for item in split.test]
    model_probabilities = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)
    config = PolymarketExecutionResearchConfig()
    baseline_execution = evaluate_polymarket_execution_policy(
        split.test,
        baseline_probabilities,
        replay,
        config=config,
    )
    model_execution = evaluate_polymarket_execution_policy(
        split.test,
        model_probabilities,
        replay,
        config=config,
    )
    prediction_evidence = _polymarket_held_out_prediction_evidence(
        split.test,
        baseline_probabilities,
        model_probabilities,
    )
    payload: dict[str, object] = {
        "schema_version": "polymarket-prospective-model-experiment-v1",
        "run_id": source.run_id,
        "feature_dataset": source.summary(),
        "feature_materialization": {
            "dataset_id": source.dataset_id,
            "status": "contract_test_fixture",
            "row_count": len(source.rows),
            "labeled_row_count": len(source.rows),
            "dataset_sha256": source.dataset_sha256,
        },
        "model_dataset": dataset.summary(),
        "split": split.summary(),
        "model": model.asdict(),
        "probability_report": probability_report.asdict(),
        "held_out_prediction_evidence": prediction_evidence,
        "baseline_execution": baseline_execution.asdict(),
        "model_execution": model_execution.asdict(),
        "execution_latency_sensitivity": {
            "schema_version": "polymarket-execution-latency-sensitivity-v1",
            "primary_network_latency_ms": 100,
            "network_latencies_ms": [100],
            "policies": {
                "baseline": {"100": baseline_execution.asdict()},
                "model": {"100": model_execution.asdict()},
            },
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
        "ai": {
            "enabled": False,
            "reason": "contract_test_fixture",
            "trading_authority": False,
            "profitability_claim": False,
        },
        "confirmatory_evidence_contract": {
            "independent_unit": "shared_btc_eth_sol_five_minute_time_group",
            "minimum_untouched_test_time_groups": 30,
            "observed_untouched_test_time_groups": len(split.test_group_starts_ms),
            "minimum_markets_per_asset": 30,
            "confirmatory_ready": len(split.test_group_starts_ms) >= 30,
            "trading_authority": False,
            "profitability_claim": False,
        },
        "evidence_gates": {
            "validation_probability_improved": (
                probability_report.validation_log_loss_delta < 0.0
            ),
            "untouched_test_probability_improved": (
                probability_report.test_log_loss_delta < 0.0
                and probability_report.test_brier_delta < 0.0
            ),
            "after_cost_execution_improved": (
                model_execution.net_realized_pnl_quote
                > baseline_execution.net_realized_pnl_quote
            ),
            "all_positions_officially_settled": True,
            "ai_enabled": False,
            "ai_uplift_accepted": False,
            "live_trading_authority": False,
            "profitability_claim": False,
        },
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    artifact_path = tmp_path / "fixture-artifact.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validated = validate_polymarket_model_artifact(artifact_path)
    assert validated.artifact_sha256 == payload["artifact_sha256"]
    research_root = tmp_path / "publication"
    result = publish_polymarket_model_artifact(
        artifact_path,
        research_root,
        round_number=3,
    )
    assert result.artifact_sha256 == payload["artifact_sha256"]
    repeated = publish_polymarket_model_artifact(
        artifact_path,
        research_root,
        round_number=3,
    )
    assert repeated.manifest_sha256 == result.manifest_sha256
    manifest = json.loads(
        (research_root / "latest" / "publication-integrity.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["claims"]["profitability_claim"] is False
    assert manifest["source_artifact_sha256"] == payload["artifact_sha256"]
    score_summary = json.loads(
        (research_root / "latest" / "held-out-group-score-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert score_summary["scopes"]["ALL"]["time_group_count"] == len(
        split.test_group_starts_ms
    )
    assert score_summary["scopes"]["ALL"]["confirmatory_ready"] is False
    assert score_summary["profitability_claim"] is False
    for entry in manifest["generated_artifacts"]:
        output = research_root / entry["path"]
        assert output.stat().st_size == entry["bytes"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == entry["sha256"]
    for chart in (research_root / "latest" / "charts").glob("*.svg"):
        ET.fromstring(chart.read_text(encoding="utf-8"))
    with (research_root / "latest" / "tables" / "model-selection.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        selection_rows = list(csv.DictReader(handle))
    assert len(selection_rows) == 8
    assert sum(row["stage"] == "inner_selection" for row in selection_rows) == 6
    assert sum(row["selected_for_outer_gate"] == "True" for row in selection_rows) == 2
    selection_chart = ET.fromstring(
        (research_root / "latest" / "charts" / "model-selection.svg").read_text(
            encoding="utf-8"
        )
    )
    selection_text = " ".join(selection_chart.itertext())
    assert "INNER TRAINING FOLDS" in selection_text
    assert "OUTER VALIDATION GATE" in selection_text

    selection_tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered_model = selection_tampered["model"]
    assert isinstance(tampered_model, dict)
    boundaries = tampered_model["inner_fold_boundaries_ms"]
    assert isinstance(boundaries, list) and boundaries
    first_boundary = boundaries[0]
    assert isinstance(first_boundary, list)
    first_boundary[1] = first_boundary[2]
    model_identity = dict(tampered_model)
    model_identity.pop("model_sha256")
    tampered_model["model_sha256"] = hashlib.sha256(
        json.dumps(
            model_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    tampered_probability = selection_tampered["probability_report"]
    assert isinstance(tampered_probability, dict)
    tampered_probability["model_sha256"] = tampered_model["model_sha256"]
    probability_identity = dict(tampered_probability)
    probability_identity.pop("report_sha256")
    tampered_probability["report_sha256"] = hashlib.sha256(
        json.dumps(
            probability_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    selection_identity = dict(selection_tampered)
    selection_identity.pop("artifact_sha256")
    selection_tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            selection_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    selection_tampered_path = tmp_path / "selection-tampered.json"
    selection_tampered_path.write_text(
        json.dumps(selection_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fold boundaries are not chronological"):
        validate_polymarket_model_artifact(selection_tampered_path)

    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["held_out_prediction_evidence"]["rows"][0]["model_up_probability"] = (
        "0.999"
    )
    canonical_tampered = dict(tampered)
    canonical_tampered.pop("artifact_sha256")
    tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            canonical_tampered,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction evidence identity"):
        validate_polymarket_model_artifact(tampered_path)

    metric_tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    metric_evidence = metric_tampered["held_out_prediction_evidence"]
    assert isinstance(metric_evidence, dict)
    metric_rows = metric_evidence["rows"]
    assert isinstance(metric_rows, list) and metric_rows
    metric_row = metric_rows[0]
    assert isinstance(metric_row, dict)
    metric_row["model_up_probability"] = "0.999"
    metric_evidence["rows_sha256"] = hashlib.sha256(
        json.dumps(
            metric_rows,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    metric_identity = dict(metric_tampered)
    metric_identity.pop("artifact_sha256")
    metric_tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            metric_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    metric_tampered_path = tmp_path / "metric-tampered.json"
    metric_tampered_path.write_text(
        json.dumps(metric_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model test metrics .* does not reconcile"):
        validate_polymarket_model_artifact(metric_tampered_path)

    accounting_tampered = json.loads(artifact_path.read_text(encoding="utf-8"))

    def falsify_report(report: dict[str, object]) -> None:
        trades = report["trades"]
        assert isinstance(trades, list) and trades
        trade = trades[0]
        assert isinstance(trade, dict)
        trade["realized_pnl_quote"] = "999"
        trade_identity = dict(trade)
        trade_identity.pop("trade_sha256")
        trade["trade_sha256"] = hashlib.sha256(
            json.dumps(
                trade_identity,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        report_identity = dict(report)
        report_identity.pop("report_sha256")
        report["report_sha256"] = hashlib.sha256(
            json.dumps(
                report_identity,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()

    model_report = accounting_tampered["model_execution"]
    assert isinstance(model_report, dict)
    falsify_report(model_report)
    sensitivity = accounting_tampered["execution_latency_sensitivity"]
    assert isinstance(sensitivity, dict)
    sensitivity_policies = sensitivity["policies"]
    assert isinstance(sensitivity_policies, dict)
    sensitivity_model = sensitivity_policies["model"]
    assert isinstance(sensitivity_model, dict)
    sensitivity_report = sensitivity_model["100"]
    assert isinstance(sensitivity_report, dict)
    falsify_report(sensitivity_report)
    accounting_identity = dict(accounting_tampered)
    accounting_identity.pop("artifact_sha256")
    accounting_tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            accounting_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    accounting_path = tmp_path / "accounting-tampered.json"
    accounting_path.write_text(json.dumps(accounting_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="filled-trade accounting"):
        validate_polymarket_model_artifact(accounting_path)
