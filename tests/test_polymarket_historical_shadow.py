from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.polymarket_historical_dataset import (
    FEATURE_NAMES,
    build_historical_feature_row,
)
from simple_ai_trading.polymarket_historical_shadow import (
    BtcAggregateTradeObservation,
    PolymarketBtcFlowBuffer,
    PolymarketHistoricalShadowScorer,
    PolymarketShadowDataUnavailable,
    load_verified_historical_shadow_predictor,
)


ROOT = Path(__file__).resolve().parents[1]
PRETEST = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-historical-pretest-v1.json"
)
EVALUATION = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-historical-evaluation-v1.json"
)
SUPPORT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-historical-feature-support-v1.json"
)
EVENT_START_MS = 1_782_086_700_000


def _trade(
    market: str,
    *,
    second: int,
    aggregate_id: int,
    price: float,
    buyer_is_maker: bool = False,
) -> BtcAggregateTradeObservation:
    event_time = EVENT_START_MS + second * 1_000 + 100
    return BtcAggregateTradeObservation(
        market=market,
        source=(
            "BINANCE_SPOT"
            if market == "spot"
            else "BINANCE_USD_M_FUTURES"
        ),
        symbol="BTCUSDT",
        event_time_ms=event_time,
        received_at_ms=event_time + 20,
        aggregate_trade_id=aggregate_id,
        first_trade_id=aggregate_id * 2,
        last_trade_id=aggregate_id * 2 + 1,
        price=price,
        quantity=0.01,
        buyer_is_maker=buyer_is_maker,
    )


def _populated_flow() -> PolymarketBtcFlowBuffer:
    flow = PolymarketBtcFlowBuffer()
    for observation in _observations():
        flow.ingest(observation)
    return flow


def _observations() -> tuple[BtcAggregateTradeObservation, ...]:
    observations: list[BtcAggregateTradeObservation] = []
    for index, second in enumerate(range(-62, 31), start=1):
        observations.append(
            _trade(
                "spot",
                second=second,
                aggregate_id=index,
                price=60_000.0 + second,
                buyer_is_maker=index % 2 == 0,
            )
        )
        observations.append(
            _trade(
                "perpetual",
                second=second,
                aggregate_id=10_000 + index,
                price=60_006.0 + second * 1.1,
                buyer_is_maker=index % 3 == 0,
            )
        )
    return tuple(observations)


def _historical_flow() -> dict[str, np.ndarray]:
    day_start_ms = EVENT_START_MS - 300_000
    rows = 1_000
    flow: dict[str, np.ndarray] = {
        "second_ms": day_start_ms + np.arange(rows, dtype=np.int64) * 1_000
    }
    for market in ("spot", "perpetual"):
        flow[f"{market}_close"] = np.full(rows, np.nan, dtype=np.float64)
        for name in (
            "quote_volume",
            "aggressive_buy_quote",
            "aggressive_sell_quote",
            "aggregate_count",
            "constituent_trade_count",
            "maximum_aggregate_quote",
            "squared_aggregate_quote_sum",
            "last_trade_age_seconds",
        ):
            flow[f"{market}_{name}"] = np.zeros(rows, dtype=np.float64)
    for observation in _observations():
        market = observation.market
        index = (observation.event_time_ms - day_start_ms) // 1_000
        quote = observation.quote_notional
        flow[f"{market}_close"][index] = observation.price
        flow[f"{market}_quote_volume"][index] += quote
        side = (
            "aggressive_sell_quote"
            if observation.buyer_is_maker
            else "aggressive_buy_quote"
        )
        flow[f"{market}_{side}"][index] += quote
        flow[f"{market}_aggregate_count"][index] += 1.0
        flow[f"{market}_constituent_trade_count"][
            index
        ] += observation.constituent_trade_count
        flow[f"{market}_maximum_aggregate_quote"][index] = max(
            flow[f"{market}_maximum_aggregate_quote"][index],
            quote,
        )
        flow[f"{market}_squared_aggregate_quote_sum"][index] += quote * quote
    for market in ("spot", "perpetual"):
        close = flow[f"{market}_close"]
        count = flow[f"{market}_aggregate_count"]
        age = flow[f"{market}_last_trade_age_seconds"]
        first_index = int(np.flatnonzero(np.isfinite(close))[0])
        close[:first_index] = close[first_index]
        last_close = np.nan
        last_age = np.iinfo(np.uint32).max
        for index in range(rows):
            if count[index] > 0:
                last_close = close[index]
                last_age = 0
            elif np.isfinite(last_close):
                close[index] = last_close
                last_age += 1
            age[index] = last_age
    return flow


def test_verified_shadow_predictor_scores_without_authority() -> None:
    predictor = load_verified_historical_shadow_predictor(
        pretest_path=PRETEST,
        evaluation_path=EVALUATION,
        support_path=SUPPORT,
    )
    assert predictor.candidate_id == "lgbm-depth2-leaves3"
    assert predictor.trading_authority is False
    probability = predictor.predict_up_probability(
        np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    )
    assert 0.0 < probability < 1.0


def test_shadow_scorer_observes_causal_flow_and_never_grants_authority() -> None:
    predictor = load_verified_historical_shadow_predictor(
        pretest_path=PRETEST,
        evaluation_path=EVALUATION,
        support_path=SUPPORT,
    )
    broad_lower = np.full(len(FEATURE_NAMES), -1e100, dtype=np.float64)
    broad_upper = np.full(len(FEATURE_NAMES), 1e100, dtype=np.float64)
    predictor = replace(
        predictor,
        support_profile=replace(
            predictor.support_profile,
            minimum=broad_lower,
            maximum=broad_upper,
            outer_lower=broad_lower,
            outer_upper=broad_upper,
        ),
    )
    scorer = PolymarketHistoricalShadowScorer(
        predictor=predictor,
        flow=_populated_flow(),
    )
    decision_time = EVENT_START_MS + 30_000
    decision = scorer.evaluate(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=decision_time,
        observed_at_ms=decision_time + 100,
    )
    assert decision.status == "observed"
    assert decision.reason == ""
    assert decision.probability_up is not None
    assert 0.0 < decision.probability_up < 1.0
    assert decision.trading_authority is False
    assert decision.grants_execution_authority is False


def test_real_train_support_abstains_on_synthetic_out_of_distribution_flow() -> None:
    predictor = load_verified_historical_shadow_predictor(
        pretest_path=PRETEST,
        evaluation_path=EVALUATION,
        support_path=SUPPORT,
    )
    decision_time = EVENT_START_MS + 30_000
    decision = PolymarketHistoricalShadowScorer(
        predictor=predictor,
        flow=_populated_flow(),
    ).evaluate(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=decision_time,
        observed_at_ms=decision_time + 100,
    )
    assert decision.status == "abstain"
    assert decision.reason == "feature_support_out_of_distribution"
    assert decision.support_profile_sha256 == (
        predictor.support_profile.artifact_sha256
    )
    assert decision.extreme_outlier_count > 0


def test_live_flow_vector_is_bit_identical_to_frozen_historical_builder() -> None:
    decision_time = EVENT_START_MS + 30_000
    live = _populated_flow().feature_vector(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=decision_time,
        observed_at_ms=decision_time + 100,
    )
    historical = build_historical_feature_row(
        SimpleNamespace(
            event_start_ms=EVENT_START_MS,
            condition_id="0x" + "1" * 64,
            identity_payload_sha256="2" * 64,
            role="test",
        ),
        day_start_ms=EVENT_START_MS - 300_000,
        decision_offset_seconds=30,
        flow=_historical_flow(),
    ).feature_values
    assert live.dtype == np.float32
    assert np.array_equal(live, historical)


def test_flow_is_idempotent_but_latches_conflicting_or_regressed_data() -> None:
    flow = PolymarketBtcFlowBuffer()
    first = _trade("spot", second=-1, aggregate_id=10, price=60_000.0)
    assert flow.ingest(first) is True
    assert flow.ingest(first) is False
    with pytest.raises(
        PolymarketShadowDataUnavailable,
        match="spot_duplicate_identity_mismatch",
    ):
        flow.ingest(replace(first, price=60_001.0))
    assert flow.faults == ("spot_duplicate_identity_mismatch",)

    another = PolymarketBtcFlowBuffer()
    another.ingest(first)
    with pytest.raises(PolymarketShadowDataUnavailable, match="spot_stream_regression"):
        another.ingest(_trade("spot", second=0, aggregate_id=9, price=60_001.0))


def test_missing_or_stale_data_abstains_instead_of_predicting() -> None:
    predictor = load_verified_historical_shadow_predictor(
        pretest_path=PRETEST,
        evaluation_path=EVALUATION,
        support_path=SUPPORT,
    )
    scorer = PolymarketHistoricalShadowScorer(
        predictor=predictor,
        flow=PolymarketBtcFlowBuffer(),
    )
    decision_time = EVENT_START_MS + 30_000
    missing = scorer.evaluate(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=decision_time,
        observed_at_ms=decision_time,
    )
    assert missing.status == "abstain"
    assert missing.reason == "spot_flow_missing"
    assert missing.probability_up is None
    assert missing.grants_execution_authority is False

    late = scorer.evaluate(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=decision_time,
        observed_at_ms=decision_time + 5_001,
    )
    assert late.status == "abstain"
    assert late.reason == "decision_stale"


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(PRETEST.read_text(encoding="utf-8"))
    payload["best_challenger_id"] = "tampered"
    tampered = tmp_path / "pretest.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="historical pretest integrity failed"):
        load_verified_historical_shadow_predictor(
            pretest_path=tampered,
            evaluation_path=EVALUATION,
            support_path=SUPPORT,
        )


def test_shadow_module_has_no_network_account_or_execution_imports() -> None:
    path = (
        ROOT
        / "src"
        / "simple_ai_trading"
        / "polymarket_historical_shadow.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [None])
    }
    assert not {
        "requests",
        "httpx",
        "websockets",
        "py_clob_client",
        "simple_ai_trading.polymarket_live",
        "simple_ai_trading.polymarket_live_v2",
    }.intersection(imports)
