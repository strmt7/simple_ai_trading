"""Contract tests for leakage-safe Polymarket probability research."""

from __future__ import annotations

import csv
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from simple_ai_trading import polymarket_ai_veto as ai_veto_module
from simple_ai_trading.ai_model_benchmark import AI_MODEL_BENCHMARK_CONTRACT
from simple_ai_trading.ai_uplift import assess_ai_uplift
from simple_ai_trading.cli import (
    _polymarket_execution_uplift_metrics,
    _polymarket_held_out_prediction_evidence,
    _polymarket_latency_scenarios,
    _polymarket_matched_uplift_periods,
    _polymarket_profile_prediction_evidence,
)
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    POLYMARKET_TAKER_ORDER_DELAY_MS,
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_ai_veto import (
    PolymarketAIVetoConfig,
    benchmark_polymarket_ai_veto,
    build_polymarket_ai_veto_cases,
)
from simple_ai_trading.polymarket_action_value import (
    POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
    build_polymarket_action_feature,
    build_polymarket_action_label,
    build_polymarket_action_value_dataset,
)
from simple_ai_trading.polymarket_features import (
    POLYMARKET_FEATURE_NAMES,
    PolymarketFeatureConfig,
    PolymarketFeatureDataset,
    PolymarketFeatureRow,
    polymarket_feature_row_sha256,
)
from simple_ai_trading.polymarket_model import (
    POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256,
    POLYMARKET_MODEL_FEATURE_NAMES,
    POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION,
    POLYMARKET_PROFILE_CONTRACT_SHA256,
    POLYMARKET_PROFILE_FEATURES,
    POLYMARKET_PROFILE_L2_CANDIDATES,
    PolymarketModelConfig,
    build_polymarket_inference_input,
    build_polymarket_model_dataset,
    fit_polymarket_offset_model,
    fit_polymarket_profile_challenger,
    predict_polymarket_probabilities,
    predict_polymarket_profile_probabilities,
    split_polymarket_model_dataset,
)
from simple_ai_trading.polymarket_model_execution import (
    POLYMARKET_CAUSAL_SETTLEMENT_CONTRACT_SHA256,
    POLYMARKET_RETRY_CHALLENGER_SCHEMA_VERSION,
    POLYMARKET_RETRY_CONTRACT_SHA256,
    PolymarketExecutionResearchConfig,
    build_polymarket_policy_selection,
    evaluate_polymarket_execution_policy,
    evaluate_polymarket_retry_execution_policy,
)
from simple_ai_trading.polymarket_publication import (
    POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION,
    _ai_case_rows,
    _parsed_valid_ai_response,
    _validate_ai_evidence,
    publish_polymarket_model_artifact,
    validate_polymarket_model_artifact,
)
from simple_ai_trading.polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketMarketExecutionEvidence,
    PolymarketRecordedBook,
    PolymarketReplayDiagnostics,
    PolymarketResolutionEvidence,
)
from simple_ai_trading.polymarket_repricing import (
    POLYMARKET_REPRICING_CONTRACT_SHA256,
    PolymarketRepricingConfig,
    PolymarketRepricingExecutionContext,
    evaluate_polymarket_repricing_ceiling,
)
from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_source_verification import (
    POLYMARKET_SOURCE_VERIFICATION_CHECKS,
    POLYMARKET_SOURCE_VERIFICATION_SCHEMA_VERSION,
    PolymarketSourceVerificationReport,
    validate_polymarket_source_verification,
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
    resolution_delay_ms: int = 1_000,
    taker_order_delay_enabled: bool = False,
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
                    market_id=market.condition_id,
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
                            sample.decision_event_id
                            if phase == "decision"
                            else f"book-{sample.sample_id[:12]}-{outcome.lower()}-{phase}"
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
            resolved_at_ms=market.end_ms + resolution_delay_ms,
            received_wall_ms=market.end_ms + resolution_delay_ms,
            received_monotonic_ns=(market.end_ms + resolution_delay_ms) * 1_000_000,
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
        market_execution_evidence=tuple(
            PolymarketMarketExecutionEvidence(
                run_id="fixture-run",
                condition_id=market.condition_id,
                observed_wall_ms=max(0, market.event_start_ms - 1_000),
                observed_monotonic_ns=0,
                maker_base_fee=0,
                taker_base_fee=0,
                taker_order_delay_enabled=taker_order_delay_enabled,
                general_order_delay_seconds=0,
                minimum_order_age_seconds=0,
                clob_info_sha256=hashlib.sha256(
                    f"clob:{market.condition_id}".encode("ascii")
                ).hexdigest(),
                up_fee_rate_sha256=hashlib.sha256(
                    f"up-fee:{market.condition_id}".encode("ascii")
                ).hexdigest(),
                down_fee_rate_sha256=hashlib.sha256(
                    f"down-fee:{market.condition_id}".encode("ascii")
                ).hexdigest(),
                snapshot_sha256=hashlib.sha256(
                    f"snapshot:{market.condition_id}".encode("ascii")
                ).hexdigest(),
            )
            for market in selected_markets
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
    up_features = next(
        item for item in dataset.samples if item.official_up
    ).feature_map()
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


def test_live_inference_is_label_free_hash_bound_and_prediction_equivalent() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    sample = split.test[0]
    row = next(
        item for item in source.rows if item.feature_id == sample.source_feature_id
    )
    market = next(item for item in markets if item.condition_id == sample.condition_id)
    unlabeled = replace(
        row,
        official_up=None,
        resolution_event_id="",
        row_sha256="",
    )
    unlabeled = replace(
        unlabeled,
        row_sha256=polymarket_feature_row_sha256(unlabeled),
    )

    model_input = build_polymarket_inference_input(
        unlabeled,
        market,
        config=dataset.config,
    )
    live_probability = predict_polymarket_probabilities(model, (model_input,))[0]
    historical_probability = predict_polymarket_probabilities(model, (sample,))[0]
    payload = model_input.asdict()

    assert live_probability == pytest.approx(historical_probability, abs=0.0)
    assert model_input.feature_values == sample.feature_values
    assert model_input.risk_context_values == sample.risk_context_values
    assert {
        "official_up",
        "resolution_event_id",
        "realized_pnl_quote",
        "gross_payout_quote",
    }.isdisjoint(payload)
    tampered = replace(
        unlabeled,
        feature_values=(
            unlabeled.feature_values[0] + 1.0,
            *unlabeled.feature_values[1:],
        ),
    )
    with pytest.raises(ValueError, match="feature row identity is invalid"):
        build_polymarket_inference_input(tampered, market, config=dataset.config)

    alternate_config = replace(dataset.config, maximum_horizon_error_ms=500)
    mismatched_input = build_polymarket_inference_input(
        unlabeled,
        market,
        config=alternate_config,
    )
    with pytest.raises(ValueError, match="another model configuration"):
        predict_polymarket_probabilities(model, (mismatched_input,))


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


def test_profile_challenger_is_frozen_deterministic_and_zero_expanded() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    control, _ = fit_polymarket_offset_model(dataset, split)

    first_model, first_report = fit_polymarket_profile_challenger(
        dataset,
        split,
        control,
    )
    second_model, second_report = fit_polymarket_profile_challenger(
        dataset,
        split,
        control,
    )

    assert first_model == second_model
    assert first_report == second_report
    assert first_model.contract_sha256 == POLYMARKET_PROFILE_CONTRACT_SHA256
    assert first_model.control_model_sha256 == control.model_sha256
    assert len(first_model.candidate_inner_log_losses) == 21
    assert first_model.candidate_inner_log_losses[0][0] == "market_baseline"
    assert first_model.inner_selected_candidate.startswith("offset_profile_")
    assert first_model.selected_candidate != "market_baseline"
    assert first_model.selected_profile is not None
    profiles = dict(POLYMARKET_PROFILE_FEATURES)
    assert first_model.selected_feature_names == profiles[first_model.selected_profile]
    active = set(first_model.selected_feature_names)
    for index, name in enumerate(POLYMARKET_MODEL_FEATURE_NAMES):
        if name not in active:
            assert first_model.coefficients[index + 1] == 0.0
    predictions = predict_polymarket_profile_probabilities(
        first_model,
        split.test,
    )
    assert predictions.shape == (len(split.test),)
    assert np.all((predictions > 0.0) & (predictions < 1.0))
    assert first_report.control_model_sha256 == control.model_sha256
    assert first_report.challenger_model_sha256 == first_model.model_sha256
    assert not first_model.trading_authority
    assert not first_model.execution_claim
    assert not first_model.profitability_claim
    assert not first_report.trading_authority
    assert not first_report.execution_claim
    assert not first_report.profitability_claim


def test_split_requires_both_outcome_classes_for_each_asset() -> None:
    source, markets = _source_fixture()
    source = replace(
        source,
        rows=tuple(
            replace(row, official_up=True) if row.asset == "SOL" else row
            for row in source.rows
        ),
    )
    dataset = build_polymarket_model_dataset(source, markets)

    with pytest.raises(ValueError, match="official outcome classes for SOL"):
        split_polymarket_model_dataset(dataset)


def _repricing_replay_fixture(
    *,
    cross_segment_exit: bool = False,
    taker_order_delay_enabled: bool = False,
    entry_tick_drift: bool = False,
    shifted_wall_clock: bool = False,
):
    source, markets = _source_fixture()
    dataset = build_polymarket_model_dataset(source, markets)
    sample = next(item for item in dataset.samples if item.asset == "BTC")
    replay = _replay_fixture(
        (sample,),
        markets,
        execution_latency_ms=500,
        taker_order_delay_enabled=taker_order_delay_enabled,
    )
    base_entry = next(
        item
        for item in replay.books
        if item.outcome == "Up"
        and item.received_wall_ms == sample.decision_received_wall_ms + 500
    )
    books = list(replay.books)
    if entry_tick_drift:
        books = [
            replace(item, tick_size=Decimal("0.001"))
            if item.outcome == "Up"
            and item.received_wall_ms == sample.decision_received_wall_ms
            else item
            for item in books
        ]
    venue_delay_ms = 250 if taker_order_delay_enabled else 0
    entry = base_entry
    sequence = max(item.sequence_number for item in books)
    if venue_delay_ms:
        sequence += 1
        entry_wall_ms = sample.decision_received_wall_ms + 500 + venue_delay_ms
        entry = replace(
            base_entry,
            event_id="repricing-delayed-entry",
            sequence_number=sequence,
            snapshot=replace(
                base_entry.snapshot,
                source_time_ms=entry_wall_ms,
                received_wall_ms=entry_wall_ms,
                received_monotonic_ns=(
                    sample.decision_received_monotonic_ns
                    + (500 + venue_delay_ms) * 1_000_000
                ),
                source_payload_sha256="d" * 64,
            ),
        )
        books.append(entry)
    sequence += 1
    exit_decision_wall_ms = entry.received_wall_ms + 1_000
    exit_decision_monotonic_ns = entry.received_monotonic_ns + 1_000_000_000
    if shifted_wall_clock:
        exit_decision_wall_ms += 1
        exit_decision_monotonic_ns -= 2_000_000
    exit_decision_book = replace(
        entry,
        event_id="repricing-exit-decision",
        sequence_number=sequence,
        snapshot=replace(
            entry.snapshot,
            source_time_ms=exit_decision_wall_ms,
            received_wall_ms=exit_decision_wall_ms,
            received_monotonic_ns=exit_decision_monotonic_ns,
            source_payload_sha256="c" * 64,
        ),
    )
    books.append(exit_decision_book)
    exit_bid = min(
        Decimal("0.95"),
        entry.snapshot.asks[0].price + Decimal("0.20"),
    )
    wall_ms = entry.received_wall_ms + 1_000 + 500 + venue_delay_ms
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id=entry.market.condition_id,
        asset_id=entry.token_id,
        bids=(BookLevel(exit_bid, Decimal("100")),),
        asks=(BookLevel(exit_bid + Decimal("0.01"), Decimal("100")),),
        source_time_ms=wall_ms,
        received_wall_ms=wall_ms,
        received_monotonic_ns=(
            entry.received_monotonic_ns + (1_000 + 500 + venue_delay_ms) * 1_000_000
        ),
        source_payload_sha256="f" * 64,
    )
    exit_book = PolymarketRecordedBook(
        run_id=replay.run_id,
        event_id="repricing-exit",
        event_type="book",
        connection_id=entry.connection_id,
        segment_id=("different-segment" if cross_segment_exit else entry.segment_id),
        sequence_number=sequence + 1,
        sub_index=0,
        market=entry.market,
        outcome="Up",
        tick_size=entry.tick_size,
        snapshot=snapshot,
    )
    books = tuple(
        sorted(
            (*books, exit_book),
            key=lambda item: (
                item.received_monotonic_ns,
                item.sequence_number,
                item.event_id,
            ),
        )
    )
    diagnostics = replace(
        replay.diagnostics,
        book_state_transition_count=len(books),
        materialized_book_count=len(books),
        total_event_count=len(books),
        causally_ordered_event_count=len(books),
    )
    return (
        PolymarketEvidenceReplay(
            run_id=replay.run_id,
            markets=replay.markets,
            books=books,
            resolutions=replay.resolutions,
            diagnostics=diagnostics,
            market_execution_evidence=replay.market_execution_evidence,
        ),
        entry,
        exit_bid,
    )


def _round9_source_row(
    replay: PolymarketEvidenceReplay,
) -> tuple[PolymarketFeatureRow, PolymarketFiveMinuteMarket]:
    source, markets = _source_fixture()
    dataset = build_polymarket_model_dataset(source, markets)
    sample = next(item for item in dataset.samples if item.asset == "BTC")
    row = next(
        item for item in source.rows if item.feature_id == sample.source_feature_id
    )
    provisional = replace(row, row_sha256="")
    row = replace(
        provisional,
        row_sha256=polymarket_feature_row_sha256(provisional),
    ).validated()
    market = next(
        item for item in replay.markets if item.condition_id == row.condition_id
    )
    assert row.decision_event_id == sample.decision_event_id
    return row, market


def _round9_execution(
    replay: PolymarketEvidenceReplay,
    *,
    outcome: str = "Up",
):
    row, market = _round9_source_row(replay)
    context = PolymarketRepricingExecutionContext(replay)
    feature = build_polymarket_action_feature(row, market, outcome)
    decision = context.decision_at(
        market,
        event_id=row.decision_event_id,
        received_wall_ms=row.decision_received_wall_ms,
        received_monotonic_ns=row.decision_received_monotonic_ns,
        outcome=outcome,
        maximum_creation_book_age_ms=500,
    )
    assert decision is not None
    execution = context.execute(
        market,
        decision,
        latency_ms=500,
        holding_period_ms=1_000,
        minimum_remaining_market_time_ms=30_000,
        maximum_order_creation_book_age_ms=500,
        maximum_post_target_execution_observation_delay_ms=500,
    )
    return row, market, feature, execution


def test_profile_challenger_falls_back_without_validation_evidence() -> None:
    source, markets = _source_fixture(predictive=False)
    feature_indexes = {
        name: index for index, name in enumerate(POLYMARKET_FEATURE_NAMES)
    }
    neutral_rows = []
    for row in source.rows:
        values = list(row.feature_values)
        for name in ("up_midpoint", "down_midpoint"):
            values[feature_indexes[name]] = 0.5
        neutral_rows.append(replace(row, feature_values=tuple(values)))
    source = replace(source, rows=tuple(neutral_rows))
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    control, _ = fit_polymarket_offset_model(dataset, split)

    model, report = fit_polymarket_profile_challenger(dataset, split, control)

    assert model.selected_candidate == "market_baseline"
    assert model.selected_profile is None
    assert model.selected_feature_names == ()
    assert model.selected_l2 is None
    assert set(model.coefficients) == {0.0}
    assert report.validation_log_loss_delta_vs_control == pytest.approx(0.0)
    assert report.test_log_loss_delta_vs_control == pytest.approx(0.0)
    assert report.test_brier_delta_vs_control == pytest.approx(0.0)


def test_profile_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-005-profile-challenger-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_PROFILE_CONTRACT_SHA256
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    documented_profiles = contract["candidate_profiles"]
    code_profiles = dict(POLYMARKET_PROFILE_FEATURES)
    assert documented_profiles["full"] == "POLYMARKET_MODEL_FEATURE_NAMES"
    assert tuple(documented_profiles) == tuple(code_profiles)
    for profile, feature_names in code_profiles.items():
        if profile != "full":
            assert tuple(documented_profiles[profile]) == feature_names
    assert POLYMARKET_PROFILE_L2_CANDIDATES == (0.001, 0.01, 0.1, 1.0, 10.0)


def test_causal_settlement_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-006-causal-settlement-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_CAUSAL_SETTLEMENT_CONTRACT_SHA256
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert contract["economic_chronology"]["market_end_is_settlement"] is False
    assert contract["capital_contract"]["profit_reinvestment"] is False
    assert contract["truth_constraints"]["profitability_claim"] is False


def test_live_inference_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-007-label-free-inference-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_LIVE_INFERENCE_CONTRACT_SHA256
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert contract["inference_contract"]["feature_row_must_be_unresolved"] is True
    assert "official_up" in contract["forbidden_live_fields"]
    assert contract["truth_constraints"]["profitability_claim"] is False


def test_repricing_ceiling_walks_both_taker_legs_without_granting_authority() -> None:
    replay, entry, exit_bid = _repricing_replay_fixture()

    report = evaluate_polymarket_repricing_ceiling(replay)
    opportunity = next(
        item
        for item in report.opportunities
        if item.asset == "BTC"
        and item.outcome == "Up"
        and item.per_leg_submission_latency_ms == 500
        and item.holding_period_ms == 1_000
    )
    entry_ask = entry.snapshot.asks[0].price
    gross_without_fees = (exit_bid - entry_ask) * entry.market.minimum_order_size

    assert opportunity.complete_round_trip_count >= 1
    assert opportunity.best_net_quote is not None
    assert Decimal("0") < opportunity.best_net_quote < gross_without_fees
    assert opportunity.best_exit_received_wall_ms == (
        opportunity.best_entry_received_wall_ms + 1_500
    )
    assert opportunity.best_entry_execution_target_wall_ms == (
        opportunity.best_decision_received_wall_ms + 500
    )
    assert opportunity.best_exit_decision_target_wall_ms == (
        opportunity.best_entry_execution_target_wall_ms + 1_000
    )
    assert opportunity.best_exit_execution_target_wall_ms == (
        opportunity.best_exit_decision_target_wall_ms + 500
    )
    assert opportunity.best_entry_venue_taker_delay_ms == 0
    assert opportunity.best_exit_venue_taker_delay_ms == 0
    assert opportunity.terminal_reason_counts["complete_round_trip"] == (
        opportunity.complete_round_trip_count
    )
    assert report.status == "insufficient_market_support"
    assert report.confirmation_eligible
    assert report.noncausal_oracle_upper_bound
    assert report.prepositioned_inventory_assumption
    assert not report.training_authority
    assert not report.trading_authority
    assert not report.profitability_claim
    with pytest.raises(ValueError, match="repricing report is invalid"):
        replace(report, profitability_claim=True).validated()
    with pytest.raises(ValueError, match="repricing report is invalid"):
        replace(report, cells=()).validated()


def test_repricing_ceiling_never_carries_execution_across_segments() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture(cross_segment_exit=True)

    report = evaluate_polymarket_repricing_ceiling(replay)
    opportunity = next(
        item
        for item in report.opportunities
        if item.asset == "BTC"
        and item.outcome == "Up"
        and item.per_leg_submission_latency_ms == 500
        and item.holding_period_ms == 1_000
    )

    assert opportunity.complete_round_trip_count == 0
    assert opportunity.best_net_quote is None
    assert not opportunity.positive


def test_repricing_ceiling_applies_recorded_taker_delay_to_each_leg() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture(
        taker_order_delay_enabled=True
    )

    report = evaluate_polymarket_repricing_ceiling(replay)
    opportunity = next(
        item
        for item in report.opportunities
        if item.asset == "BTC"
        and item.outcome == "Up"
        and item.per_leg_submission_latency_ms == 500
        and item.holding_period_ms == 1_000
    )

    assert opportunity.complete_round_trip_count >= 1
    assert opportunity.best_entry_venue_taker_delay_ms == 250
    assert opportunity.best_exit_venue_taker_delay_ms == 250
    assert opportunity.best_entry_received_wall_ms == (
        opportunity.best_decision_received_wall_ms + 750
    )
    assert opportunity.best_exit_received_wall_ms == (
        opportunity.best_entry_received_wall_ms + 1_750
    )
    assert opportunity.best_entry_execution_target_wall_ms == (
        opportunity.best_decision_received_wall_ms + 750
    )
    assert opportunity.best_exit_execution_target_wall_ms == (
        opportunity.best_exit_decision_target_wall_ms + 750
    )


def test_repricing_ceiling_uses_monotonic_clock_for_causal_elapsed_time() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture(shifted_wall_clock=True)

    report = evaluate_polymarket_repricing_ceiling(replay)
    opportunity = next(
        item
        for item in report.opportunities
        if item.asset == "BTC"
        and item.outcome == "Up"
        and item.per_leg_submission_latency_ms == 500
        and item.holding_period_ms == 1_000
    )

    assert opportunity.complete_round_trip_count >= 1
    assert opportunity.best_exit_decision_received_wall_ms > (
        opportunity.best_exit_decision_target_wall_ms
    )
    assert opportunity.best_exit_decision_received_monotonic_ns < (
        opportunity.best_exit_decision_target_monotonic_ns
    )
    assert opportunity.best_exit_received_monotonic_ns >= (
        opportunity.best_exit_execution_target_monotonic_ns
    )


def test_repricing_ceiling_never_defaults_missing_execution_parameters_to_zero() -> (
    None
):
    replay, _entry, _exit_bid = _repricing_replay_fixture()
    missing = PolymarketEvidenceReplay(
        run_id=replay.run_id,
        markets=replay.markets,
        books=replay.books,
        resolutions=replay.resolutions,
        diagnostics=replay.diagnostics,
    )

    with pytest.raises(ValueError, match="requires recorded execution parameters"):
        evaluate_polymarket_repricing_ceiling(missing)


def test_repricing_ceiling_rejects_limit_invalidated_by_tick_change() -> None:
    baseline, _entry, _exit_bid = _repricing_replay_fixture()
    replay, _entry, _exit_bid = _repricing_replay_fixture(entry_tick_drift=True)

    baseline_report = evaluate_polymarket_repricing_ceiling(baseline)
    report = evaluate_polymarket_repricing_ceiling(replay)
    baseline_opportunity = next(
        item
        for item in baseline_report.opportunities
        if item.asset == "BTC"
        and item.outcome == "Up"
        and item.per_leg_submission_latency_ms == 500
        and item.holding_period_ms == 1_000
    )
    opportunity = next(
        item
        for item in report.opportunities
        if item.asset == "BTC"
        and item.outcome == "Up"
        and item.per_leg_submission_latency_ms == 500
        and item.holding_period_ms == 1_000
    )

    assert opportunity.complete_round_trip_count == (
        baseline_opportunity.complete_round_trip_count - 1
    )
    assert opportunity.terminal_reason_counts["entry_tick_drift"] == 1
    assert sum(opportunity.terminal_reason_counts.values()) == (
        opportunity.decision_count
    )


def test_round9_action_features_are_label_free_and_outcome_oriented() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture()
    row, market = _round9_source_row(replay)

    up = build_polymarket_action_feature(row, market, "Up")
    down = build_polymarket_action_feature(row, market, "Down")
    changed_label = replace(
        row,
        official_up=not bool(row.official_up),
        resolution_event_id="different-resolution",
        row_sha256="",
    )
    changed_label = replace(
        changed_label,
        row_sha256=polymarket_feature_row_sha256(changed_label),
    )
    repeated = build_polymarket_action_feature(changed_label, market, "Up")
    up_values = up.feature_map()
    down_values = down.feature_map()

    assert repeated.action_feature_sha256 == up.action_feature_sha256
    assert repeated.feature_values == up.feature_values
    assert down_values["chosen_return_250ms_bps"] == pytest.approx(
        -up_values["chosen_return_250ms_bps"]
    )
    assert down_values["chosen_microprice_deviation_bps"] == pytest.approx(
        up_values["opposite_microprice_deviation_bps"]
    )
    assert down_values["chosen_book_age_ms"] == pytest.approx(
        up_values["opposite_book_age_ms"]
    )
    assert {
        "official_up",
        "resolution_event_id",
        "row_sha256",
    }.isdisjoint(up.asdict())


def test_round9_complete_action_uses_exact_two_leg_net_value() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture()
    _row, _market, feature, execution = _round9_execution(replay)

    label = build_polymarket_action_label(feature, execution)

    assert execution.terminal_reason == "complete_round_trip"
    assert execution.net_quote is not None and execution.net_quote > 0
    assert label.category == "successful_round_trip"
    assert label.positive_complete
    assert label.stress_utility_quote == execution.net_quote
    assert label.entry_cost_quote == execution.entry_cost_quote
    assert label.exit_proceeds_quote == execution.exit_proceeds_quote
    assert not label.condition_blocked


def test_round9_dataset_builds_both_actions_and_rejects_sampled_execution() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture()
    row, _market = _round9_source_row(replay)
    source, _markets = _source_fixture()
    source = replace(
        source,
        rows=(row,),
        candidate_count=1,
        labeled_market_counts={"BTC": 1, "ETH": 0, "SOL": 0},
    )

    dataset = build_polymarket_action_value_dataset(
        source,
        PolymarketRepricingExecutionContext(replay),
    )

    assert len(dataset.features) == len(dataset.labels) == 2
    assert {item.outcome for item in dataset.features} == {"Up", "Down"}
    assert dataset.dataset_sha256
    assert not dataset.summary()["profitability_claim"]
    sampled = PolymarketEvidenceReplay(
        run_id=replay.run_id,
        markets=replay.markets,
        books=replay.books,
        resolutions=replay.resolutions,
        diagnostics=replace(replay.diagnostics, book_sample_interval_ms=250),
        market_execution_evidence=replay.market_execution_evidence,
    )
    with pytest.raises(ValueError, match="source contract"):
        build_polymarket_action_value_dataset(
            source,
            PolymarketRepricingExecutionContext(sampled),
        )


def test_round9_no_fill_has_zero_utility_and_no_inventory() -> None:
    replay, entry, _exit_bid = _repricing_replay_fixture()
    shallow_entry = replace(
        entry,
        snapshot=replace(
            entry.snapshot,
            asks=(BookLevel(entry.snapshot.asks[0].price, Decimal("1")),),
            source_payload_sha256="9" * 64,
        ),
    )
    books = tuple(
        shallow_entry
        if (item.event_id, item.token_id) == (entry.event_id, entry.token_id)
        else item
        for item in replay.books
    )
    shallow = PolymarketEvidenceReplay(
        run_id=replay.run_id,
        markets=replay.markets,
        books=books,
        resolutions=replay.resolutions,
        diagnostics=replay.diagnostics,
        market_execution_evidence=replay.market_execution_evidence,
    )

    _row, _market, feature, execution = _round9_execution(shallow)
    label = build_polymarket_action_label(feature, execution)

    assert execution.terminal_reason == "entry_not_filled"
    assert not execution.entry_filled
    assert execution.entry_cost_quote is None
    assert label.category == "entry_no_fill"
    assert label.classifier_eligible
    assert label.stress_utility_quote == 0
    assert not label.condition_blocked


def test_round9_rejected_entry_tick_drift_is_classifier_eligible_no_fill() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture(entry_tick_drift=True)

    _row, _market, feature, execution = _round9_execution(replay)
    label = build_polymarket_action_label(feature, execution)

    assert execution.terminal_reason == "entry_tick_drift"
    assert not execution.entry_filled
    assert execution.entry_cost_quote is None
    assert label.category == "entry_no_fill"
    assert label.classifier_eligible
    assert label.stress_utility_quote == 0
    assert not label.condition_blocked


def test_round9_distinguishes_post_submission_close_window_receipt() -> None:
    replay, entry, _exit_bid = _repricing_replay_fixture()
    row, market = _round9_source_row(replay)
    shifted_entry = replace(
        entry,
        snapshot=replace(
            entry.snapshot,
            source_time_ms=entry.received_wall_ms + 1,
            received_wall_ms=entry.received_wall_ms + 1,
            received_monotonic_ns=entry.received_monotonic_ns + 1_000_000,
            source_payload_sha256="8" * 64,
        ),
    )
    books = tuple(
        shifted_entry
        if (item.event_id, item.token_id) == (entry.event_id, entry.token_id)
        else item
        for item in replay.books
    )
    shifted = PolymarketEvidenceReplay(
        run_id=replay.run_id,
        markets=replay.markets,
        books=books,
        resolutions=replay.resolutions,
        diagnostics=replay.diagnostics,
        market_execution_evidence=replay.market_execution_evidence,
    )
    context = PolymarketRepricingExecutionContext(shifted)
    decision = context.decision_at(
        market,
        event_id=row.decision_event_id,
        received_wall_ms=row.decision_received_wall_ms,
        received_monotonic_ns=row.decision_received_monotonic_ns,
        outcome="Up",
        maximum_creation_book_age_ms=500,
    )
    assert decision is not None
    execution = context.execute(
        replace(market, end_ms=row.decision_received_wall_ms + 30_500),
        decision,
        latency_ms=500,
        holding_period_ms=1_000,
        minimum_remaining_market_time_ms=30_000,
        maximum_order_creation_book_age_ms=500,
        maximum_post_target_execution_observation_delay_ms=500,
    )

    assert execution.terminal_reason == (
        "entry_confirmation_enters_excluded_close_window"
    )
    assert not execution.entry_filled


def test_round9_failed_exit_loses_full_entry_cost_and_blocks_condition() -> None:
    replay, _entry, _exit_bid = _repricing_replay_fixture(cross_segment_exit=True)
    _row, _market, feature, execution = _round9_execution(replay)

    label = build_polymarket_action_label(feature, execution)

    assert execution.entry_filled
    assert not execution.exit_filled
    assert execution.entry_cost_quote is not None
    assert execution.terminal_reason == "missing_exit_execution_book"
    assert label.category == "filled_entry_failed_exit"
    assert label.classifier_eligible
    assert label.stress_utility_quote == -execution.entry_cost_quote
    assert label.condition_blocked
    assert not label.positive_complete


def test_round9_action_value_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-009-causal-action-value-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_ACTION_VALUE_CONTRACT_SHA256
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert contract["ai_boundary"]["event_loop_llm_permitted"] is False
    continuity = contract["continuity_contract"]
    assert "zero queue-saturation" in continuity["global_run_gate"]
    assert "entire group" in continuity["synchronized_exclusion"]
    assert "Official outcomes" in continuity["eligibility_freeze"]
    assert "Pre-gap" in continuity["fresh_reconnect_baseline"]
    assert contract["truth_constraints"]["profitability_claim"] is False


def test_round9_primary_source_audit_is_bound_to_action_contract() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-009-primary-source-audit.json"
    )
    audit = json.loads(path.read_text(encoding="utf-8"))

    assert audit["round_9_action_contract_sha256"] == (
        POLYMARKET_ACTION_VALUE_CONTRACT_SHA256
    )
    assert audit["profitability_claim"] is False
    assert audit["trading_authority"] is False


def test_repricing_contract_code_and_document_are_identical() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-008-executable-repricing-ceiling-contract.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")
    canonical = json.dumps(
        contract,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert claimed == POLYMARKET_REPRICING_CONTRACT_SHA256
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert (
        contract["execution_parameter_contract"]["enabled_taker_delay_ms"]
        == POLYMARKET_TAKER_ORDER_DELAY_MS
    )
    assert (
        contract["fixed_grid"]["maximum_order_creation_book_age_ms"]
        == PolymarketRepricingConfig().maximum_order_creation_book_age_ms
    )
    assert (
        contract["fixed_grid"]["maximum_post_target_execution_observation_delay_ms"]
        == PolymarketRepricingConfig().maximum_post_target_execution_observation_delay_ms
    )
    assert contract["economic_contract"]["midpoint_or_last_trade_fill_credit"] is False
    assert contract["truth_constraints"]["profitability_claim"] is False


def test_execution_evidence_rejects_unmodeled_general_delay() -> None:
    evidence = PolymarketMarketExecutionEvidence(
        run_id="run",
        condition_id="0x" + "1" * 64,
        observed_wall_ms=1,
        observed_monotonic_ns=1,
        maker_base_fee=0,
        taker_base_fee=0,
        taker_order_delay_enabled=True,
        general_order_delay_seconds=1,
        minimum_order_age_seconds=0,
        clob_info_sha256="a" * 64,
        up_fee_rate_sha256="b" * 64,
        down_fee_rate_sha256="c" * 64,
        snapshot_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="nonzero CLOB general order delay"):
        evidence.validated()


def test_model_fit_rejects_substituted_split_sample() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    substituted = replace(
        split.train[0],
        feature_values=tuple(value + 1.0 for value in split.train[0].feature_values),
    )
    tampered = replace(split, train=(substituted, *split.train[1:]))

    with pytest.raises(ValueError, match="substituted"):
        fit_polymarket_offset_model(dataset, tampered)


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
    assert (
        first.causal_settlement_contract_sha256
        == POLYMARKET_CAUSAL_SETTLEMENT_CONTRACT_SHA256
    )
    assert len({trade.condition_id for trade in first.trades}) == len(first.trades)
    assert all(trade.official_resolution_event_id for trade in first.trades)
    assert all(trade.execution_book_event_id for trade in first.trades)
    assert not first.trading_authority
    assert not first.execution_claim
    assert not first.profitability_claim


def test_execution_locks_capital_until_official_resolution_is_available() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    resolution_delay_ms = 600_000
    replay = _replay_fixture(
        split.test,
        markets,
        resolution_delay_ms=resolution_delay_ms,
    )

    report = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
    )

    assert report.reason_counts["open_portfolio_risk_budget_exhausted"] > 0
    assert report.filled_order_count < report.signal_market_count
    assert {point.event_start_ms for point in report.equity_curve} == set(
        split.test_group_starts_ms
    )
    assert all(
        point.settled_at_ms == point.event_start_ms + 300_000 + resolution_delay_ms
        for point in report.equity_curve
    )


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


def test_execution_waits_for_the_first_proven_book_after_arrival() -> None:
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

    assert report.filled_order_count == 0
    assert all(trade.effective_latency_ms == 200 for trade in report.trades)
    assert all("execution" in trade.execution_book_event_id for trade in report.trades)


def test_execution_fails_closed_when_confirmation_arrives_after_bound() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(
        split.test,
        markets,
        execution_latency_ms=700,
    )

    report = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
        config=PolymarketExecutionResearchConfig(
            submission_latency_ms=100,
            maximum_execution_observation_delay_ms=500,
        ),
    )

    assert report.attempted_order_count == 1
    assert report.filled_order_count == 0
    assert all(trade.execution_state == "UNKNOWN" for trade in report.trades)
    assert all(not trade.execution_book_event_id for trade in report.trades)
    assert all(trade.effective_latency_ms == 100 for trade in report.trades)
    assert report.reason_counts["portfolio_blocked_by_unknown_state"] == (
        report.signal_market_count - 1
    )


def test_segmented_execution_never_crosses_a_reconnect_boundary() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    model, _ = fit_polymarket_offset_model(dataset, split)
    predictions = predict_polymarket_probabilities(model, split.test)
    replay = _replay_fixture(split.test, markets)
    replay.diagnostics = replace(
        replay.diagnostics,
        continuity_mode="segmented",
        stream_gap_count=1,
    )

    admitted = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
    )
    replay.books = tuple(
        replace(book, segment_id=f"reconnected-{book.segment_id}")
        if "execution" in book.event_id
        else book
        for book in replay.books
    )
    blocked = evaluate_polymarket_execution_policy(
        split.test,
        predictions,
        replay,
    )

    assert admitted.filled_order_count == admitted.evaluated_market_count
    assert blocked.filled_order_count == 0
    assert all(trade.execution_state == "UNKNOWN" for trade in blocked.trades)
    assert blocked.reason_counts["no_gap_free_causal_execution_book_at_latency"] == (
        blocked.attempted_order_count
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


def test_retry_policy_retries_only_after_terminal_zero_fill() -> None:
    source, markets = _source_fixture(predictive=True)
    dataset = build_polymarket_model_dataset(source, markets)
    split = split_polymarket_model_dataset(dataset)
    condition = split.test[0].condition_id
    samples = tuple(
        sorted(
            (item for item in split.test if item.condition_id == condition),
            key=lambda item: item.decision_received_monotonic_ns,
        )[:3]
    )
    replay = _replay_fixture(samples, markets)
    first_execution_ns = samples[0].decision_received_monotonic_ns + 100_000_000
    replay.books = tuple(
        replace(
            book,
            snapshot=replace(
                book.snapshot,
                asks=(BookLevel(book.snapshot.asks[0].price, Decimal("1")),),
            ),
        )
        if book.outcome == "Up" and book.received_monotonic_ns == first_execution_ns
        else book
        for book in replay.books
    )
    probabilities = np.asarray([0.9] * len(samples), dtype=np.float64)

    control = evaluate_polymarket_execution_policy(samples, probabilities, replay)
    retry = evaluate_polymarket_retry_execution_policy(
        samples,
        probabilities,
        replay,
    )

    assert control.attempted_order_count == 1
    assert control.filled_order_count == 0
    assert control.trades[0].execution_state == "CANCELLED"
    assert retry.signal_market_count == 1
    assert retry.attempted_order_count == 2
    assert retry.filled_order_count == 1
    assert [item.execution_state for item in retry.trades] == [
        "CANCELLED",
        "FILLED",
    ]
    assert retry.reason_counts["retry_suppressed_after_fill"] == 1

    delayed_replay = _replay_fixture(samples, markets, execution_latency_ms=700)
    unknown = evaluate_polymarket_retry_execution_policy(
        samples,
        probabilities,
        delayed_replay,
        config=PolymarketExecutionResearchConfig(
            submission_latency_ms=100,
            maximum_execution_observation_delay_ms=500,
        ),
    )
    assert unknown.attempted_order_count == 1
    assert unknown.trades[0].execution_state == "UNKNOWN"
    assert unknown.reason_counts["portfolio_blocked_by_unknown_state"] == 2


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


def test_ai_veto_parser_rejects_type_coercion_and_duplicate_json_keys() -> None:
    malformed = (
        '{"action":"approve","confidence":"0.9","reason_codes":'
        '["edge_after_fees"],"summary":"typed string"}',
        '{"action":"approve","confidence":true,"reason_codes":'
        '["edge_after_fees"],"summary":"typed boolean"}',
        '{"action":"approve","confidence":0.9,"reason_codes":'
        '["edge_after_fees"],"summary":7}',
        '{"action":"veto","action":"approve","confidence":0.9,'
        '"reason_codes":["edge_after_fees"],"summary":"duplicate"}',
    )

    for content in malformed:
        with pytest.raises(ValueError, match="AI response"):
            ai_veto_module._parse_decision({"message": {"content": content}})
        assert _parsed_valid_ai_response({"message": {"content": content}}) is None


def test_ai_veto_supports_preregistered_qwen3_14b_candidate() -> None:
    assert PolymarketAIVetoConfig(model="qwen3:14b").validated().model == "qwen3:14b"


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
    chat_calls = 0

    def approve(
        url: str,
        payload: dict[str, object],
        _timeout: float,
        _method: str,
    ) -> object:
        nonlocal chat_calls
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b", "digest": "f" * 64}]}
        if url.endswith("/api/show"):
            return {"model": "qwen3.5:9b", "parameters": "9B"}
        chat_calls += 1
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
        expected_model_digest="f" * 64,
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
    completed_chat_calls = chat_calls
    with pytest.raises(ValueError, match="differs from benchmark provenance"):
        benchmark_polymarket_ai_veto(
            cases[:1],
            all_condition_ids=[item.condition_id for item in split.test],
            selection_sha256=selection.selection_sha256,
            risk_benchmark_evidence_sha256="a" * 64,
            config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
            post_json=approve,  # type: ignore[arg-type]
            expected_model_digest="e" * 64,
        )
    assert chat_calls == completed_chat_calls

    duplicate_condition = replace(
        cases[1],
        condition_id=cases[0].condition_id,
        case_sha256="",
    )
    duplicate_condition = replace(
        duplicate_condition,
        case_sha256=ai_veto_module._canonical_sha256(
            duplicate_condition.identity_payload()
        ),
    )

    def unexpected_provider(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("duplicate conditions must fail before provider access")

    with pytest.raises(ValueError, match="selection identity is invalid"):
        benchmark_polymarket_ai_veto(
            cases[:1],
            all_condition_ids=[cases[0].condition_id],
            selection_sha256="not-a-digest",
            risk_benchmark_evidence_sha256="a" * 64,
            config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
            post_json=unexpected_provider,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly one case per market condition"):
        benchmark_polymarket_ai_veto(
            (cases[0], duplicate_condition),
            all_condition_ids=[cases[0].condition_id],
            selection_sha256=selection.selection_sha256,
            risk_benchmark_evidence_sha256="a" * 64,
            config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
            post_json=unexpected_provider,  # type: ignore[arg-type]
        )


def test_ai_veto_cache_reuses_only_exact_model_evidence_and_latency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    )[:1]
    assert cases
    chat_calls = 0
    digest = ["f" * 64]

    def approve(
        url: str,
        _payload: dict[str, object],
        _timeout: float,
        _method: str,
    ) -> object:
        nonlocal chat_calls
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3.5:9b", "digest": digest[0]}]}
        if url.endswith("/api/show"):
            return {"model": "qwen3.5:9b", "parameters": "9B"}
        chat_calls += 1
        return {
            "message": {
                "content": json.dumps(
                    {
                        "action": "approve",
                        "confidence": 0.9,
                        "reason_codes": ["edge_after_fees"],
                        "summary": "Causal after-fee evidence passes review.",
                    }
                )
            }
        }

    clock = iter((100.0, 102.25, 200.0, 201.5))
    monkeypatch.setattr(
        "simple_ai_trading.polymarket_ai_veto.time.perf_counter",
        lambda: next(clock),
    )
    progress_rows: list[dict[str, object]] = []

    def run(store: PolymarketEvidenceStore):
        return benchmark_polymarket_ai_veto(
            cases,
            all_condition_ids=[item.condition_id for item in split.test],
            selection_sha256=selection.selection_sha256,
            risk_benchmark_evidence_sha256="a" * 64,
            config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
            post_json=approve,  # type: ignore[arg-type]
            progress=lambda _event, item: progress_rows.append(dict(item)),
            cache_store=store,
        )

    with PolymarketEvidenceStore(tmp_path / "ai-cache.duckdb") as store:
        first = run(store)
        second = run(store)
        digest[0] = "e" * 64
        third = run(store)

    assert first == second
    assert first.results[0].latency_seconds == 2.25
    assert first.market_permissions[cases[0].condition_id]
    assert sum(first.market_permissions.values()) == 1
    assert third.results[0].latency_seconds == 1.5
    assert third.model_digest == "e" * 64
    assert chat_calls == 2
    assert [row["cache_hit"] for row in progress_rows] == [False, True, False]


def test_ai_veto_cache_replays_failures_without_retrying_for_a_better_answer(
    tmp_path: Path,
) -> None:
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
    )[:1]
    chat_calls = 0

    def malformed(
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
        return {"message": {"content": "not-json"}}

    progress_rows: list[dict[str, object]] = []
    with PolymarketEvidenceStore(tmp_path / "invalid-cache.duckdb") as store:
        reports = [
            benchmark_polymarket_ai_veto(
                cases,
                all_condition_ids=[item.condition_id for item in split.test],
                selection_sha256=selection.selection_sha256,
                risk_benchmark_evidence_sha256="a" * 64,
                config=PolymarketAIVetoConfig(model="qwen3.5:9b"),
                post_json=malformed,  # type: ignore[arg-type]
                cache_store=store,
                progress=lambda _event, item: progress_rows.append(dict(item)),
            )
            for _ in range(2)
        ]
        cached_rows = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_ai_veto_cache")
            .fetchone()[0]
        )

    assert reports[0] == reports[1]
    assert reports[0].provider_failure_count == 1
    assert chat_calls == 1
    assert cached_rows == 1
    assert [row["cache_hit"] for row in progress_rows] == [False, True]


def test_ai_uplift_periods_use_initial_capital_returns_and_exact_groups() -> None:
    starts = (1_000, 301_000)

    def report(values: tuple[str, ...], *, capital: str = "100") -> SimpleNamespace:
        return SimpleNamespace(
            initial_capital_quote=Decimal(capital),
            equity_curve=tuple(
                SimpleNamespace(
                    event_start_ms=start_ms,
                    group_realized_pnl_quote=Decimal(value),
                )
                for start_ms, value in zip(starts, values, strict=True)
            ),
        )

    split = SimpleNamespace(test_group_starts_ms=starts)
    periods = _polymarket_matched_uplift_periods(
        split,
        report(("1", "-2")),
        report(("2", "-1")),
    )

    assert [item["baseline_return"] for item in periods] == pytest.approx([0.01, -0.02])
    assert [item["ai_return"] for item in periods] == pytest.approx([0.02, -0.01])
    with pytest.raises(ValueError, match="equity periods differ"):
        _polymarket_matched_uplift_periods(
            split,
            report(("1", "-2")),
            SimpleNamespace(
                initial_capital_quote=Decimal("100"),
                equity_curve=report(("2", "-1")).equity_curve[:1],
            ),
        )
    with pytest.raises(ValueError, match="initial capital differs"):
        _polymarket_matched_uplift_periods(
            split,
            report(("1", "-2")),
            report(("2", "-1"), capital="200"),
        )

    metrics = _polymarket_execution_uplift_metrics(
        SimpleNamespace(
            trades=(),
            maximum_drawdown_fraction=Decimal("0.02"),
            net_realized_pnl_quote=Decimal("10"),
            return_on_initial_capital=Decimal("0.01"),
            report_sha256="f" * 64,
        ),
        dataset_fingerprint="e" * 64,
    )
    assert metrics["downside_return_risk_ratio"] == pytest.approx(0.5)


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
            "path": "docs/ai/risk-review/latest/comparison.json",
            "sha256": "a" * 64,
            "contract": AI_MODEL_BENCHMARK_CONTRACT,
            "selected_model": "qwen3.5:9b",
            "score": 1.0,
            "model_provenance": {
                "path": "docs/ai/risk-review/latest/model-provenance.json",
                "provenance_sha256": "c" * 64,
                "benchmark_sha256": "a" * 64,
                "benchmark_contract": AI_MODEL_BENCHMARK_CONTRACT,
                "model": "qwen3.5:9b",
                "ollama_manifest_digest": "f" * 64,
                "base_blob_sha256": "d" * 64,
                "size_bytes": 6_000_000_000,
            },
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

    altered_source_rows = json.loads(json.dumps(prediction_rows))
    source_row = next(
        row for row in altered_source_rows if row["sample_id"] == cases[0].sample_id
    )
    feature_index = source_row["feature_names"].index("direct_return_100ms_bps")
    source_row["feature_values"][feature_index] = format(
        float(source_row["feature_values"][feature_index]) + 1.0,
        ".17g",
    )
    with pytest.raises(ValueError, match="causal model sample"):
        _validate_ai_evidence(
            ai_evidence,
            predictions=altered_source_rows,
            probability=probability_report.asdict(),
            model_execution=model_execution.asdict(),
        )

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
                "selection_sha256": tampered["policy_selection"]["selection_sha256"],
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
                "selection_sha256": tampered["policy_selection"]["selection_sha256"],
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
    assert not any(ai_report.market_permissions.values())
    assert all(not result.decision.valid for result in ai_report.results)


def test_invalid_execution_risk_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="configuration is invalid"):
        PolymarketExecutionResearchConfig(
            maximum_loss_fraction_per_market=Decimal("0.02"),
            maximum_loss_fraction_per_time_group=Decimal("0.01"),
        ).validated()
    with pytest.raises(ValueError, match="configuration is invalid"):
        PolymarketExecutionResearchConfig(
            maximum_execution_observation_delay_ms=60_001,
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
    profile_model, profile_probability_report = fit_polymarket_profile_challenger(
        dataset, split, model
    )
    baseline_probabilities = [item.baseline_up_probability for item in split.test]
    model_probabilities = predict_polymarket_probabilities(model, split.test)
    profile_probabilities = predict_polymarket_profile_probabilities(
        profile_model,
        split.test,
    )
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
    profile_model_execution = evaluate_polymarket_execution_policy(
        split.test,
        profile_probabilities,
        replay,
        config=config,
    )
    model_retry_execution = evaluate_polymarket_retry_execution_policy(
        split.test,
        model_probabilities,
        replay,
        config=config,
    )
    retry_gates = {
        "probability_model_gates_passed": (
            probability_report.validation_log_loss_delta < 0.0
            and probability_report.test_log_loss_delta < 0.0
            and probability_report.test_brier_delta < 0.0
        ),
        "minimum_confirmatory_test_time_groups_met": (
            len(split.test_group_starts_ms) >= 30
        ),
        "positive_after_cost_at_every_latency": (
            model_retry_execution.net_realized_pnl_quote > 0
        ),
        "improved_after_cost_at_every_latency": (
            model_retry_execution.net_realized_pnl_quote
            > model_execution.net_realized_pnl_quote
        ),
        "return_on_deployed_not_worse_at_every_latency": (
            model_retry_execution.return_on_deployed_capital
            >= model_execution.return_on_deployed_capital
        ),
        "maximum_drawdown_not_worse_at_every_latency": (
            model_retry_execution.maximum_drawdown_fraction
            <= model_execution.maximum_drawdown_fraction
        ),
        "all_order_outcomes_terminal": all(
            trade.execution_state != "UNKNOWN" for trade in model_retry_execution.trades
        ),
    }
    prediction_evidence = _polymarket_held_out_prediction_evidence(
        split.test,
        baseline_probabilities,
        model_probabilities,
    )
    profile_prediction_evidence = _polymarket_profile_prediction_evidence(
        split.test,
        profile_probabilities,
        control_rows_sha256=str(prediction_evidence["rows_sha256"]),
    )
    profile_gates = {
        "validation_log_loss_not_worse_than_control": (
            profile_probability_report.validation_log_loss_delta_vs_control <= 0.0
        ),
        "test_log_loss_strictly_better_than_control": (
            profile_probability_report.test_log_loss_delta_vs_control < 0.0
        ),
        "test_brier_score_strictly_better_than_control": (
            profile_probability_report.test_brier_delta_vs_control < 0.0
        ),
        "minimum_untouched_test_time_groups_met": (
            len(split.test_group_starts_ms) >= 30
        ),
        "positive_after_cost_at_every_latency": (
            profile_model_execution.net_realized_pnl_quote > 0
        ),
        "improved_after_cost_at_every_latency": (
            profile_model_execution.net_realized_pnl_quote
            > model_execution.net_realized_pnl_quote
        ),
        "return_on_deployed_not_worse_at_every_latency": (
            profile_model_execution.return_on_deployed_capital
            >= model_execution.return_on_deployed_capital
        ),
        "maximum_drawdown_not_worse_at_every_latency": (
            profile_model_execution.maximum_drawdown_fraction
            <= model_execution.maximum_drawdown_fraction
        ),
        "all_order_outcomes_terminal": all(
            trade.execution_state != "UNKNOWN"
            for trade in profile_model_execution.trades
        ),
    }
    profile_promotion_gates_passed = all(profile_gates.values())
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION,
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
        "profile_model": profile_model.asdict(),
        "profile_probability_report": profile_probability_report.asdict(),
        "held_out_prediction_evidence": prediction_evidence,
        "profile_held_out_prediction_evidence": profile_prediction_evidence,
        "baseline_execution": baseline_execution.asdict(),
        "model_execution": model_execution.asdict(),
        "profile_model_execution": profile_model_execution.asdict(),
        "model_retry_execution": model_retry_execution.asdict(),
        "retry_challenger": {
            "schema_version": POLYMARKET_RETRY_CHALLENGER_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_RETRY_CONTRACT_SHA256,
            "control_policy": "model",
            "challenger_policy": "model_retry",
            "gates": retry_gates,
            "accepted": all(retry_gates.values()),
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
        "profile_challenger": {
            "schema_version": POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_PROFILE_CONTRACT_SHA256,
            "control_policy": "model",
            "challenger_policy": "profile_model",
            "gates": profile_gates,
            "promotion_gates_passed": profile_promotion_gates_passed,
            "accepted": False,
            "status": (
                "awaiting_later_prospective_confirmation"
                if profile_promotion_gates_passed
                else "exploratory_gates_failed"
            ),
            "requires_later_prospective_confirmation": True,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
        "execution_latency_sensitivity": {
            "schema_version": "polymarket-execution-latency-sensitivity-v1",
            "primary_network_latency_ms": 100,
            "network_latencies_ms": [100],
            "policies": {
                "baseline": {"100": baseline_execution.asdict()},
                "model": {"100": model_execution.asdict()},
                "profile_model": {"100": profile_model_execution.asdict()},
                "model_retry": {"100": model_retry_execution.asdict()},
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
            "minimum_confirmatory_test_time_groups_met": (
                len(split.test_group_starts_ms) >= 30
            ),
            "after_cost_execution_improved": (
                model_execution.net_realized_pnl_quote
                > baseline_execution.net_realized_pnl_quote
            ),
            "after_cost_model_improved_at_every_stress_latency": (
                model_execution.net_realized_pnl_quote
                > baseline_execution.net_realized_pnl_quote
            ),
            "retry_challenger_accepted": all(retry_gates.values()),
            "profile_challenger_promotion_gates_passed": (
                profile_promotion_gates_passed
            ),
            "all_positions_officially_settled": True,
            "all_order_outcomes_terminal": True,
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
    verification_provisional = PolymarketSourceVerificationReport(
        schema_version=POLYMARKET_SOURCE_VERIFICATION_SCHEMA_VERSION,
        status="verified",
        artifact_sha256=str(payload["artifact_sha256"]),
        run_id=source.run_id,
        recorder_report_sha256="1" * 64,
        feature_dataset_sha256=source.dataset_sha256,
        model_dataset_sha256=dataset.dataset_sha256,
        split_sha256=split.split_sha256,
        model_sha256=model.model_sha256,
        probability_report_sha256=probability_report.report_sha256,
        profile_model_sha256=profile_model.model_sha256,
        profile_probability_report_sha256=(profile_probability_report.report_sha256),
        held_out_rows_sha256=str(prediction_evidence["rows_sha256"]),
        profile_held_out_rows_sha256=str(profile_prediction_evidence["rows_sha256"]),
        execution_report_sha256_by_policy_and_latency={
            "baseline": {"100": baseline_execution.report_sha256},
            "model": {"100": model_execution.report_sha256},
            "profile_model": {"100": profile_model_execution.report_sha256},
            "model_retry": {"100": model_retry_execution.report_sha256},
        },
        verified_feature_row_count=len(source.rows),
        verified_model_sample_count=len(dataset.samples),
        verified_held_out_sample_count=len(split.test),
        verified_execution_scenario_count=4,
        verified_execution_trade_count=(
            len(baseline_execution.trades)
            + len(model_execution.trades)
            + len(profile_model_execution.trades)
            + len(model_retry_execution.trades)
        ),
        verified_filled_order_count=(
            baseline_execution.filled_order_count
            + model_execution.filled_order_count
            + profile_model_execution.filled_order_count
            + model_retry_execution.filled_order_count
        ),
        checks={name: True for name in POLYMARKET_SOURCE_VERIFICATION_CHECKS},
        report_sha256="",
    )
    verification_identity = verification_provisional.asdict()
    verification_identity.pop("report_sha256")
    verification = replace(
        verification_provisional,
        report_sha256=hashlib.sha256(
            json.dumps(
                verification_identity,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    )
    validate_polymarket_source_verification(
        verification.asdict(),
        artifact_sha256=str(payload["artifact_sha256"]),
        run_id=source.run_id,
    )
    research_root = tmp_path / "publication"
    result = publish_polymarket_model_artifact(
        artifact_path,
        research_root,
        round_number=3,
        source_verification=verification.asdict(),
    )
    assert result.artifact_sha256 == payload["artifact_sha256"]
    repeated = publish_polymarket_model_artifact(
        artifact_path,
        research_root,
        round_number=3,
        source_verification=verification.asdict(),
    )
    assert repeated.manifest_sha256 == result.manifest_sha256
    incomplete_source_verification = verification.asdict()
    incomplete_source_verification["execution_report_sha256_by_policy_and_latency"].pop(
        "model_retry"
    )
    incomplete_source_verification["verified_execution_scenario_count"] = 3
    incomplete_source_verification["verified_execution_trade_count"] -= len(
        model_retry_execution.trades
    )
    incomplete_source_verification["verified_filled_order_count"] -= (
        model_retry_execution.filled_order_count
    )
    incomplete_source_verification.pop("report_sha256")
    incomplete_source_verification["report_sha256"] = hashlib.sha256(
        json.dumps(
            incomplete_source_verification,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    with pytest.raises(ValueError, match="every artifact execution scenario"):
        publish_polymarket_model_artifact(
            artifact_path,
            research_root,
            round_number=3,
            source_verification=incomplete_source_verification,
        )
    manifest = json.loads(
        (research_root / "latest" / "publication-integrity.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["claims"]["profitability_claim"] is False
    assert manifest["source_artifact_sha256"] == payload["artifact_sha256"]
    assert manifest["source_reconstruction_verified"] is True
    assert manifest["source_verification_report_sha256"] == verification.report_sha256
    assert (
        json.loads(
            (research_root / "latest" / "source-verification.json").read_text(
                encoding="utf-8"
            )
        )
        == verification.asdict()
    )
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
    with (research_root / "latest" / "tables" / "profile-model-selection.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        profile_selection_rows = list(csv.DictReader(handle))
    assert len(profile_selection_rows) == 23
    assert (
        sum(row["stage"] == "inner_selection" for row in profile_selection_rows) == 21
    )
    assert {row["profile"] for row in profile_selection_rows if row["profile"]} == {
        "diffusion_core",
        "fast_cross_venue_flow",
        "full",
        "prediction_book_state",
    }
    profile_selection_chart = ET.fromstring(
        (research_root / "latest" / "charts" / "profile-model-selection.svg").read_text(
            encoding="utf-8"
        )
    )
    profile_selection_text = " ".join(profile_selection_chart.itertext())
    assert "FROZEN PROFILE CHALLENGER GRID" in profile_selection_text.upper()
    assert "OUTER VALIDATION GATE" in profile_selection_text

    gate_tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    gate_tampered["evidence_gates"]["after_cost_execution_improved"] = not bool(
        gate_tampered["evidence_gates"]["after_cost_execution_improved"]
    )
    gate_identity = dict(gate_tampered)
    gate_identity.pop("artifact_sha256")
    gate_tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            gate_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    gate_tampered_path = tmp_path / "gate-tampered.json"
    gate_tampered_path.write_text(
        json.dumps(gate_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evidence gates do not reconstruct"):
        validate_polymarket_model_artifact(gate_tampered_path)

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

    sample_tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    sample_evidence = sample_tampered["held_out_prediction_evidence"]
    sample_rows = sample_evidence["rows"]
    sample_rows[0]["feature_values"][0] = "0.123"
    sample_evidence["rows_sha256"] = hashlib.sha256(
        json.dumps(
            sample_rows,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    sample_identity = dict(sample_tampered)
    sample_identity.pop("artifact_sha256")
    sample_tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            sample_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    sample_tampered_path = tmp_path / "sample-tampered.json"
    sample_tampered_path.write_text(
        json.dumps(sample_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out prediction row is malformed"):
        validate_polymarket_model_artifact(sample_tampered_path)

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

    probability_input_tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    bound_execution = probability_input_tampered["model_execution"]
    bound_execution["probability_input_sha256"] = "f" * 64
    bound_execution_identity = dict(bound_execution)
    bound_execution_identity.pop("report_sha256")
    bound_execution["report_sha256"] = hashlib.sha256(
        json.dumps(
            bound_execution_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    probability_input_identity = dict(probability_input_tampered)
    probability_input_identity.pop("artifact_sha256")
    probability_input_tampered["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            probability_input_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    probability_input_path = tmp_path / "probability-input-tampered.json"
    probability_input_path.write_text(
        json.dumps(probability_input_tampered),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="probability input does not reconstruct"):
        validate_polymarket_model_artifact(probability_input_path)

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
