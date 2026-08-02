from __future__ import annotations

from decimal import Decimal
import hashlib

import numpy as np

from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import PolymarketFeeSchedule, PolymarketFiveMinuteMarket
from simple_ai_trading.polymarket_round21_dataset import Round21OfficialOutcome
from simple_ai_trading.polymarket_round21_execution import Round21MarketExecutionEvidence
from simple_ai_trading.polymarket_round21_model import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    Round21DevelopmentPanel,
)
from simple_ai_trading.polymarket_round21_policy import Round21ProbabilityEnvelope
from simple_ai_trading.polymarket_round21_replay import Round21ReplayCondition


START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 120_000
CONDITION_ID = "0x" + "7" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def round21_panel(
    role: str,
    *,
    first_condition: int,
    condition_count: int,
) -> Round21DevelopmentPanel:
    condition_numbers = np.arange(
        first_condition,
        first_condition + condition_count,
        dtype=np.int64,
    )
    labels = (condition_numbers % 2).astype(np.float64)
    event_start = START_MS + condition_numbers * 300_000
    condition_ids = np.asarray(
        ["0x" + format(int(value), "064x") for value in condition_numbers],
        dtype=object,
    )
    signed = labels * 2.0 - 1.0
    core = np.column_stack(
        (
            np.sin(condition_numbers * 0.13),
            np.cos(condition_numbers * 0.07),
            (condition_numbers % 3).astype(np.float64) - 1.0,
        )
    ).astype(np.float32)
    spot_available = condition_numbers % 7 != 0
    usdm_available = spot_available & (condition_numbers % 5 != 0)
    spot = np.column_stack(
        (
            signed * 1.6,
            signed * 0.9 + np.sin(condition_numbers * 0.03),
        )
    ).astype(np.float32)
    usdm = np.column_stack(
        (
            signed * 1.2,
            signed * 0.7 + np.cos(condition_numbers * 0.05),
        )
    ).astype(np.float32)
    spot[~spot_available] = 0.0
    usdm[~usdm_available] = 0.0
    return Round21DevelopmentPanel(
        role=role,
        condition_ids=condition_ids,
        event_start_ms=event_start,
        decision_time_ms=event_start + 150_000,
        labels=labels,
        structural_probability=(
            0.5 + 0.02 * np.sin(condition_numbers * 0.17)
        ).astype(np.float64),
        market_prior_probability=(
            0.5 + 0.015 * np.cos(condition_numbers * 0.11)
        ).astype(np.float64),
        core_features=core,
        spot_features=spot,
        usdm_features=usdm,
        spot_available=spot_available,
        usdm_available=usdm_available,
        core_feature_names_sha256=sha("core-v1"),
        spot_feature_names_sha256=sha("spot-v1"),
        usdm_feature_names_sha256=sha("usdm-v1"),
        dataset_sha256=sha(f"dataset-{role}"),
        target_manifest_sha256=sha(f"targets-{role}"),
        dataset_design_sha256=POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    )


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=START_MS,
        end_ms=START_MS + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.20"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("100000"),
        resolution_source="chainlink",
        gamma_payload_sha256=sha("market"),
        gamma_payload_json="{}",
    )


def _evidence() -> Round21MarketExecutionEvidence:
    return Round21MarketExecutionEvidence.create(
        condition_id=CONDITION_ID,
        observed_wall_ms=DECISION_MS - 1_000,
        observed_monotonic_ns=(DECISION_MS - 1_000) * 1_000_000,
        maker_base_fee=0,
        taker_base_fee=700,
        taker_order_delay_enabled=True,
        general_order_delay_seconds=0,
        minimum_order_age_seconds=0,
        clob_info_sha256=sha("clob"),
        up_fee_rate_sha256=sha("up-fee"),
        down_fee_rate_sha256=sha("down-fee"),
        snapshot_sha256=sha("snapshot"),
    )


def _envelope(*, layer: str) -> Round21ProbabilityEnvelope:
    return Round21ProbabilityEnvelope.create(
        condition_id=CONDITION_ID,
        decision_time_ms=DECISION_MS,
        probability_up=Decimal("0.80"),
        lower_up=Decimal("0.75"),
        upper_up=Decimal("0.85"),
        model_layer=layer,
        source_model_artifact_sha256=sha(f"model-{layer}"),
        source_probability_batch_sha256=sha(f"probability-batch-{layer}"),
        feature_row_sha256=sha("feature-row"),
    )


def _book(outcome: str, received_wall_ms: int) -> PaperBookSnapshot:
    token = UP_TOKEN if outcome == "Up" else DOWN_TOKEN
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token,
        bids=(BookLevel(Decimal("0.49"), Decimal("1000")),),
        asks=(BookLevel(Decimal("0.50"), Decimal("1000")),),
        source_time_ms=received_wall_ms - 10,
        received_wall_ms=received_wall_ms,
        received_monotonic_ns=received_wall_ms * 1_000_000,
        source_payload_sha256=sha(f"{outcome}-{received_wall_ms}"),
        connected=True,
        gap_free=True,
    ).validated()


def round21_replay_condition(*, layer: str = "core") -> Round21ReplayCondition:
    books = [_book("Up", DECISION_MS - 50), _book("Down", DECISION_MS - 50)]
    for latency_offset in (500, 750, 1_250):
        books.extend(
            (
                _book("Up", DECISION_MS + latency_offset),
                _book("Down", DECISION_MS + latency_offset),
            )
        )
    return Round21ReplayCondition.create(
        market=_market(),
        market_evidence=_evidence(),
        envelopes=(_envelope(layer=layer),),
        books=books,
        outcome=Round21OfficialOutcome.create(
            condition_id=CONDITION_ID,
            event_start_ms=START_MS,
            resolved_up=True,
            observed_at_ms=START_MS + 300_100,
            source="official-polymarket-resolution",
            source_payload_sha256=sha("outcome-true"),
        ),
        source_manifest_sha256=sha("source-manifest"),
        reconciliation_sha256=sha("reconciliation"),
    )
