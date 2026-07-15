"""Contract tests for leakage-safe Polymarket probability research."""

from __future__ import annotations

from decimal import Decimal
import json

import numpy as np
import pytest

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
            "binance_return_250ms_bps": (
                (5.0 if official_up else -5.0) if predictive else 0.0
            ),
            "binance_return_1000ms_bps": (
                (10.0 if official_up else -10.0) if predictive else 0.0
            ),
            "binance_return_5000ms_bps": (
                (20.0 if official_up else -20.0) if predictive else 0.0
            ),
            "binance_realized_volatility_1000ms_bps": 2.0,
            "binance_realized_volatility_5000ms_bps": 5.0,
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
            selected_horizons = horizons[:-1] if omit_last_horizon and group == 0 else horizons
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
) -> PolymarketEvidenceReplay:
    selected_conditions = {item.condition_id for item in samples}
    selected_markets = tuple(
        market for market in markets if market.condition_id in selected_conditions
    )
    market_by_condition = {
        market.condition_id: market for market in selected_markets
    }
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
            for phase, latency_ms in (("decision", 0), ("execution", 200)):
                sequence += 1
                wall_ms = sample.decision_received_wall_ms + latency_ms
                snapshot = PaperBookSnapshot(
                    venue="polymarket",
                    market_id=market.market_id,
                    asset_id=token_id,
                    bids=(BookLevel(bid, displayed_quantity),),
                    asks=(BookLevel(ask, displayed_quantity),),
                    source_time_ms=wall_ms,
                    received_wall_ms=wall_ms,
                    received_monotonic_ns=(
                        sample.decision_received_monotonic_ns
                        + latency_ms * 1_000_000
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
    assert len(POLYMARKET_MODEL_FEATURE_NAMES) == 22
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
    assert set(model.coefficients) == {0.0}
    assert report.validation_log_loss_delta == pytest.approx(0.0)
    assert report.test_log_loss_delta == pytest.approx(0.0)


def test_execution_policy_uses_depth_latency_fees_risk_and_official_settlement() -> None:
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
    assert report.reason_counts == {
        "no_positive_after_cost_edge": report.evaluated_market_count
    }


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
        tuple(market for market in markets if market.condition_id in {
            item.condition_id for item in split.test
        }),
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
    assert all(not ai_report.market_permissions[case.condition_id] for case in cases[:2])
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
        minimum_train_time_groups=5,
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
