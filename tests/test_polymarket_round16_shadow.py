from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading.polymarket_round16 import (
    load_round16_historical_contract,
)
from simple_ai_trading.polymarket_historical_shadow import (
    BtcAggregateTradeObservation,
    PolymarketBtcFlowBuffer,
    PolymarketShadowDataUnavailable,
)
from simple_ai_trading.polymarket_round16_dataset import (
    ROUND16_FEATURE_NAMES,
    build_round16_feature_row,
)
from simple_ai_trading.polymarket_round16_model import (
    Round16ModelPanel,
    build_round16_pretest_artifact,
    fit_round16_pretest_candidates,
)
from simple_ai_trading.polymarket_round16_shadow import (
    PolymarketRound16LiveFeatureBuilder,
    PolymarketRound16ShadowScorer,
    VerifiedRound16ShadowPredictor,
    load_verified_round16_shadow_predictor,
)


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v1.json"
)
EVENT_START_MS = 1_800_000_000_000
DECISION_TIME_MS = EVENT_START_MS + 60_000


def _observation(
    market: str,
    *,
    second: int,
    aggregate_id: int,
) -> BtcAggregateTradeObservation:
    event_time_ms = EVENT_START_MS + second * 1_000 + 100
    return BtcAggregateTradeObservation(
        market=market,
        source=(
            "BINANCE_SPOT"
            if market == "spot"
            else "BINANCE_USD_M_FUTURES"
        ),
        symbol="BTCUSDT",
        event_time_ms=event_time_ms,
        received_at_ms=event_time_ms + 20,
        aggregate_trade_id=aggregate_id,
        first_trade_id=aggregate_id * 2,
        last_trade_id=aggregate_id * 2 + 1,
        price=(
            60_000.0 + second * 0.5
            if market == "spot"
            else 60_006.0 + second * 0.55
        ),
        quantity=0.01 + aggregate_id % 3 * 0.001,
        buyer_is_maker=aggregate_id % 2 == 0,
    )


def _flow_and_history() -> tuple[
    PolymarketBtcFlowBuffer,
    dict[str, np.ndarray],
]:
    flow = PolymarketBtcFlowBuffer(retention_seconds=300)
    day_start_ms = EVENT_START_MS - 300_000
    row_count = 600
    history: dict[str, np.ndarray] = {
        "second_ms": day_start_ms
        + np.arange(row_count, dtype=np.int64) * 1_000,
    }
    for market in ("spot", "perpetual"):
        history[f"{market}_close"] = np.full(
            row_count,
            np.nan,
            dtype=np.float64,
        )
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
            history[f"{market}_{name}"] = np.zeros(
                row_count,
                dtype=np.float64,
            )
    for index, second in enumerate(range(-151, 61), start=1):
        for market, offset in (("spot", 0), ("perpetual", 10_000)):
            observation = _observation(
                market,
                second=second,
                aggregate_id=offset + index,
            )
            flow.ingest(observation)
            row = (observation.event_time_ms - day_start_ms) // 1_000
            quote = observation.quote_notional
            history[f"{market}_close"][row] = observation.price
            history[f"{market}_quote_volume"][row] += quote
            side = (
                "aggressive_sell_quote"
                if observation.buyer_is_maker
                else "aggressive_buy_quote"
            )
            history[f"{market}_{side}"][row] += quote
            history[f"{market}_aggregate_count"][row] += 1
            history[f"{market}_constituent_trade_count"][
                row
            ] += observation.constituent_trade_count
            history[f"{market}_maximum_aggregate_quote"][row] = quote
            history[f"{market}_squared_aggregate_quote_sum"][row] += (
                quote * quote
            )
    for market in ("spot", "perpetual"):
        close = history[f"{market}_close"]
        count = history[f"{market}_aggregate_count"]
        age = history[f"{market}_last_trade_age_seconds"]
        first_index = int(np.flatnonzero(np.isfinite(close))[0])
        close[:first_index] = close[first_index]
        last_close = np.nan
        last_age = np.iinfo(np.uint32).max
        for index in range(row_count):
            if count[index] > 0:
                last_close = close[index]
                last_age = 0
            elif np.isfinite(last_close):
                close[index] = last_close
                last_age += 1
            age[index] = last_age
    return flow, history


def test_round16_live_vector_is_bit_identical_to_historical_builder() -> None:
    flow, history = _flow_and_history()
    live = PolymarketRound16LiveFeatureBuilder(flow).feature_vector(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
    )
    historical = build_round16_feature_row(
        SimpleNamespace(
            event_start_ms=EVENT_START_MS,
            end_ms=EVENT_START_MS + 900_000,
            condition_id="0x" + "1" * 64,
            identity_payload_sha256="2" * 64,
            role="test",
        ),
        flow_start_ms=EVENT_START_MS - 300_000,
        decision_offset_seconds=60,
        flow=history,
    ).feature_values

    assert live.dtype == np.float32
    assert np.array_equal(live, historical)


def test_round16_live_builder_fails_closed_after_feed_epoch_reset() -> None:
    flow, _ = _flow_and_history()
    flow.reset_market("spot")
    builder = PolymarketRound16LiveFeatureBuilder(flow)

    with pytest.raises(PolymarketShadowDataUnavailable, match="spot_flow_missing"):
        builder.feature_vector(
            event_start_ms=EVENT_START_MS,
            decision_time_ms=DECISION_TIME_MS,
            observed_at_ms=DECISION_TIME_MS + 100,
        )


def test_causal_snapshot_is_a_copy_not_mutable_feed_state() -> None:
    flow, _ = _flow_and_history()
    first = flow.causal_flow_snapshot(
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
        second_count=150,
    )
    first["spot_close"][0] = 0
    second = flow.causal_flow_snapshot(
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
        second_count=150,
    )

    assert second["spot_close"][0] > 0


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _model_panel(
    *,
    role: str,
    condition_count: int,
    seed: int,
) -> Round16ModelPanel:
    generator = np.random.default_rng(seed)
    signal = generator.normal(size=condition_count)
    labels = np.repeat((signal > 0).astype(np.float64), 14)
    condition_ids = np.repeat(
        np.asarray(
            [
                f"0x{seed:08x}{index:056x}"
                for index in range(1, condition_count + 1)
            ],
            dtype=object,
        ),
        14,
    )
    features = generator.normal(
        scale=0.2,
        size=(condition_count * 14, len(ROUND16_FEATURE_NAMES)),
    ).astype(np.float32)
    features[:, 0] += np.repeat(signal, 14).astype(np.float32)
    event_start = np.repeat(
        EVENT_START_MS
        + np.arange(condition_count, dtype=np.int64) * 900_000,
        14,
    )
    offsets = np.tile(np.arange(1, 15, dtype=np.int64), condition_count)
    return Round16ModelPanel(
        condition_ids=condition_ids,
        roles=np.full(len(labels), role, dtype=object),
        event_start_ms=event_start,
        decision_time_ms=event_start + offsets * 60_000,
        features=features,
        labels=labels,
        dataset_sha256="a" * 64,
    )


def _verified_artifacts(tmp_path: Path) -> tuple[Path, Path, str, str]:
    contract = load_round16_historical_contract(CONTRACT_PATH)
    train = _model_panel(role="train", condition_count=40, seed=16015)
    tune = _model_panel(role="tune", condition_count=20, seed=16016)
    candidates = fit_round16_pretest_candidates(
        train,
        tune,
        compute_backend="cpu",
    )
    pretest = build_round16_pretest_artifact(
        train,
        tune,
        candidates,
        contract=contract,
        source_commit="b" * 40,
    )
    pretest_envelope = _canonical_sha256(pretest)
    gates = {
        "minimum_terminal_conditions": True,
        "minimum_outcomes_per_class": True,
        "minimum_decision_rows": True,
        "challenger_log_loss_skill_positive": True,
        "challenger_brier_skill_positive": True,
        "challenger_balanced_accuracy_not_lower": True,
        "paired_log_loss_improvement_lower_positive": True,
        "calibration_slope_in_range": True,
        "expected_calibration_error_at_most_contract_maximum": True,
    }
    evaluation_body = {
        "schema_version": "polymarket-round16-btc-15m-evaluation-v1",
        "contract_sha256": contract.contract_sha256,
        "dataset_sha256": train.dataset_sha256,
        "pretest_artifact_sha256": pretest_envelope,
        "scope": {
            "venue": "polymarket",
            "asset": "BTC",
            "market_variant": "fifteenminute",
            "predictive_screen_only": True,
            "execution_or_profitability_claim": False,
        },
        "best_challenger_id": pretest["selected_best_challenger"],
        "gates": gates,
        "accepted_predictive_edge": True,
        "paper_authority": False,
        "live_authority": False,
        "profitability_claim": False,
    }
    evaluation = {
        **evaluation_body,
        "artifact_sha256": _canonical_sha256(evaluation_body),
    }
    pretest_path = tmp_path / "pretest.json"
    evaluation_path = tmp_path / "evaluation.json"
    pretest_path.write_text(json.dumps(pretest), encoding="utf-8")
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    return (
        pretest_path,
        evaluation_path,
        pretest_envelope,
        _canonical_sha256(evaluation),
    )


def test_round16_verified_predictor_requires_pinned_passing_evidence(
    tmp_path: Path,
) -> None:
    pretest, evaluation, pretest_sha, evaluation_sha = _verified_artifacts(
        tmp_path
    )

    predictor = load_verified_round16_shadow_predictor(
        contract_path=CONTRACT_PATH,
        pretest_path=pretest,
        evaluation_path=evaluation,
        expected_pretest_envelope_sha256=pretest_sha,
        expected_evaluation_envelope_sha256=evaluation_sha,
    )

    assert predictor.trading_authority is False
    assert predictor.pretest_envelope_sha256 == pretest_sha
    assert 0.0 < predictor.predict_up_probability(
        np.zeros(len(ROUND16_FEATURE_NAMES), dtype=np.float32)
    ) < 1.0
    with pytest.raises(ValueError, match="pinned digest differs"):
        load_verified_round16_shadow_predictor(
            contract_path=CONTRACT_PATH,
            pretest_path=pretest,
            evaluation_path=evaluation,
            expected_pretest_envelope_sha256="0" * 64,
            expected_evaluation_envelope_sha256=evaluation_sha,
        )


def test_round16_shadow_scorer_abstains_without_execution_authority() -> None:
    flow, _ = _flow_and_history()
    features = PolymarketRound16LiveFeatureBuilder(flow).feature_vector(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
    )
    width = len(ROUND16_FEATURE_NAMES)
    broad_support = {
        "schema_version": "polymarket-round16-feature-support-v1",
        "partition": "train",
        "labels_used": False,
        "feature_names_sha256": _canonical_sha256(ROUND16_FEATURE_NAMES),
        "training_rows": 14,
        "training_conditions": 1,
        "statistics": {
            "minimum": [format(float(value - 1), ".17g") for value in features],
            "maximum": [format(float(value + 1), ".17g") for value in features],
            "outer_lower": [format(float(value - 2), ".17g") for value in features],
            "outer_upper": [format(float(value + 2), ".17g") for value in features],
        },
        "gate": {
            "maximum_outside_training_range": 4,
            "maximum_extreme_outliers": 0,
            "outer_iqr_multiplier": "5",
            "action": "abstain",
        },
        "test_features_used": False,
        "live_features_used": False,
        "trading_authority": False,
    }
    settlement = {
        "schema_version": "polymarket-round16-settlement-screen-v1",
        "partition": "tune",
        "labels_used": False,
        "quantile": "0.995",
        "quantile_method": "linear",
        "quote_feature": (
            "terminal_spot_log_quote_rate_ratio_30s_to_prior_120s"
        ),
        "quote_upper_threshold": "-1000",
        "disagreement_feature": (
            "terminal_spot_perpetual_signed_aggressive_share_difference_30s"
        ),
        "disagreement_absolute_threshold": "1000",
        "abnormal_action": "abstain",
        "new_exposure_in_final_30_seconds": False,
        "test_features_used": False,
        "live_features_used": False,
        "trading_authority": False,
    }
    candidate = {
        "feature_indices": [],
        "model": {"type": "constant", "probability": 0.6},
        "calibration": {"retained": False},
    }
    predictor = VerifiedRound16ShadowPredictor(
        candidate=candidate,
        candidate_id="fixture",
        pretest_envelope_sha256="1" * 64,
        evaluation_envelope_sha256="2" * 64,
        pretest_file_sha256="4" * 64,
        evaluation_file_sha256="5" * 64,
        dataset_sha256="3" * 64,
        feature_support=broad_support,
        settlement_controls=settlement,
    )
    scorer = PolymarketRound16ShadowScorer(
        predictor=predictor,
        feature_builder=PolymarketRound16LiveFeatureBuilder(flow),
    )

    decision = scorer.evaluate(
        event_start_ms=EVENT_START_MS,
        decision_time_ms=DECISION_TIME_MS,
        observed_at_ms=DECISION_TIME_MS + 100,
    )

    assert decision.status == "abstain"
    assert decision.reason == "settlement_manipulation_anomaly"
    assert len(decision.input_sha256) == 64
    assert decision.trading_authority is False
    assert decision.grants_execution_authority is False
    assert width == 116
