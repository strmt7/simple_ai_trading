from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_replay import PolymarketResolutionEvidence
from simple_ai_trading.polymarket_round17_cohort import (
    POLYMARKET_ROUND17_COHORT_PLAN_SHA256,
    build_round17_cohort_condition,
    build_round17_cohort_condition_label,
    build_round17_cohort_manifest,
    build_round17_condition_label,
    build_round17_development_panel,
    build_round17_development_panels_streaming,
    build_round17_development_target_manifest,
    load_round17_cohort_plan,
    validate_round17_cohort_plan,
)
from simple_ai_trading.polymarket_round17_dataset import (
    PolymarketRound17ConditionDataset,
)
from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
    PolymarketRound17FeatureRow,
)
from simple_ai_trading.polymarket_round17_resolution import (
    POLYMARKET_ROUND17_DEVELOPMENT_RESOLUTION_CONTRACT_SHA256,
    acquire_round17_development_resolutions,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-017-btc-5m-cohort-plan-v1.json"
)
RESOLUTION_CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-017-btc-5m-development-resolution-contract-v1.json"
)
START_MS = 1_785_344_400_000


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _dataset(index: int, *, event_start_ms: int) -> PolymarketRound17ConditionDataset:
    condition_id = "0x" + _sha256(["condition", index])
    admission_sha256 = _sha256(["admission", index])
    segment_sha256 = _sha256(["segment", index])
    values = [0.0 for _ in POLYMARKET_ROUND17_FEATURE_NAMES]
    values[POLYMARKET_ROUND17_FEATURE_NAMES.index("structural_probability_up")] = 0.5
    values[POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up")] = 0.5
    rows = tuple(
        PolymarketRound17FeatureRow(
            condition_id=condition_id,
            decision_time_ms=event_start_ms + offset,
            admission_sha256=admission_sha256,
            causal_segment_sha256=segment_sha256,
            feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
            input_sha256=_sha256(["input", index, offset]),
            values_sha256=_sha256(values),
            values=tuple(values),
        )
        for offset in (60_000, 120_000)
    )
    provisional = PolymarketRound17ConditionDataset(
        run_id=f"run-{index}",
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        event_end_ms=event_start_ms + 300_000,
        admission_sha256=admission_sha256,
        causal_segment_sha256=segment_sha256,
        feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
        base_row_count=len(rows),
        chainlink_event_count=10,
        spot_trade_count=10,
        perpetual_trade_count=10,
        up_book_count=10,
        down_book_count=10,
        binance_layer_eligible=True,
        rows=rows,
        dataset_sha256="0" * 64,
    )
    return replace(
        provisional,
        dataset_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _market(dataset: PolymarketRound17ConditionDataset) -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id=dataset.condition_id,
        condition_id=dataset.condition_id,
        slug=f"btc-updown-5m-{dataset.event_start_ms // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=dataset.event_start_ms,
        end_ms=dataset.event_end_ms,
        up_token_id="1" * 32,
        down_token_id="2" * 32,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.25"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("1000"),
        volume_quote=Decimal("10000"),
        resolution_source="https://example.invalid",
        gamma_payload_sha256="a" * 64,
        gamma_payload_json="{}",
    )


def _label(
    dataset: PolymarketRound17ConditionDataset,
    *,
    outcome: str,
):
    market = _market(dataset)
    return build_round17_condition_label(
        dataset,
        market,
        _resolution(dataset, market=market, outcome=outcome),
    )


def _resolution(
    dataset: PolymarketRound17ConditionDataset,
    *,
    market: PolymarketFiveMinuteMarket,
    outcome: str,
) -> PolymarketResolutionEvidence:
    winning_asset = market.up_token_id if outcome == "Up" else market.down_token_id
    return PolymarketResolutionEvidence(
        run_id=dataset.run_id,
        event_id=f"resolution-{dataset.condition_id}",
        condition_id=dataset.condition_id,
        winning_asset_id=winning_asset,
        winning_outcome=outcome,
        resolved_at_ms=dataset.event_end_ms,
        received_wall_ms=dataset.event_end_ms,
        received_monotonic_ns=dataset.event_end_ms * 1_000_000,
        event_sha256=_sha256(["resolution", dataset.condition_id, outcome]),
        source="clob_gamma_crosscheck",
    )


def _terminal_payloads(
    market: PolymarketFiveMinuteMarket,
    *,
    outcome: str,
    closed: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    up_wins = outcome == "Up"
    return (
        {
            "condition_id": market.condition_id,
            "market_slug": market.slug,
            "tokens": [
                {
                    "token_id": market.up_token_id,
                    "outcome": "Up",
                    "price": "1" if up_wins else "0",
                    "winner": up_wins,
                },
                {
                    "token_id": market.down_token_id,
                    "outcome": "Down",
                    "price": "0" if up_wins else "1",
                    "winner": not up_wins,
                },
            ],
            "closed": closed,
            "accepting_orders": False,
        },
        {
            "id": market.market_id,
            "conditionId": market.condition_id,
            "slug": market.slug,
            "outcomes": json.dumps(["Up", "Down"]),
            "clobTokenIds": json.dumps(list(market.token_ids)),
            "resolutionSource": market.resolution_source,
            "outcomePrices": json.dumps(["1", "0"] if up_wins else ["0", "1"]),
            "closed": closed,
            "acceptingOrders": False,
        },
    )


class _ResolutionClient:
    def __init__(
        self,
        markets: dict[str, PolymarketFiveMinuteMarket],
        *,
        pending_condition_id: str = "",
    ) -> None:
        self.gamma: dict[str, dict[str, object]] = {}
        self.clob: dict[str, dict[str, object]] = {}
        self.gamma_calls: list[tuple[str, ...]] = []
        self.clob_calls: list[str] = []
        for index, market in enumerate(markets.values()):
            clob, gamma = _terminal_payloads(
                market,
                outcome="Up" if index % 2 else "Down",
                closed=market.condition_id != pending_condition_id,
            )
            self.clob[market.condition_id] = clob
            self.gamma[market.market_id] = gamma

    def gamma_markets(
        self,
        market_ids: tuple[str, ...],
    ) -> dict[str, dict[str, object]]:
        self.gamma_calls.append(market_ids)
        return {market_id: self.gamma[market_id] for market_id in market_ids}

    def clob_market(self, condition_id: str) -> dict[str, object]:
        self.clob_calls.append(condition_id)
        return self.clob[condition_id]


def test_round17_cohort_plan_is_self_hashed_and_assigns_fixed_roles() -> None:
    plan = load_round17_cohort_plan(PLAN_PATH)

    assert plan.plan_sha256 == POLYMARKET_ROUND17_COHORT_PLAN_SHA256
    assert (
        plan.role_for_condition(
            event_start_ms=START_MS,
            event_end_ms=START_MS + 300_000,
            source_slot_index=0,
        ).name
        == "train"
    )
    embargo_start = START_MS + 672 * 1_800_000
    assert (
        plan.role_for_condition(
            event_start_ms=embargo_start,
            event_end_ms=embargo_start + 300_000,
            source_slot_index=672,
        ).name
        == "train_tune_embargo"
    )
    test_start = START_MS + 1012 * 1_800_000
    assert (
        plan.role_for_condition(
            event_start_ms=test_start,
            event_end_ms=test_start + 300_000,
            source_slot_index=1012,
        ).name
        == "test"
    )
    with pytest.raises(ValueError, match="source slot differs"):
        plan.role_for_condition(
            event_start_ms=START_MS,
            event_end_ms=START_MS + 300_000,
            source_slot_index=1,
        )
    with pytest.raises(ValueError, match="reserved outside development"):
        build_round17_cohort_condition(
            plan,
            _dataset(9, event_start_ms=test_start),
            source_slot_index=1012,
        )


def test_round17_development_resolution_contract_is_frozen_before_outcomes() -> None:
    payload = json.loads(RESOLUTION_CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed = payload.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND17_DEVELOPMENT_RESOLUTION_CONTRACT_SHA256
    assert claimed == _sha256(payload)
    assert payload["status"] == (
        "preregistered_before_active_campaign_outcome_or_model_score_access"
    )
    assert payload["scope"]["test_conditions_permitted"] is False
    assert payload["scope"]["capture_database_mutation_permitted"] is False


def test_round17_cohort_manifest_and_panel_are_target_separated() -> None:
    plan = load_round17_cohort_plan(PLAN_PATH)
    down = _dataset(0, event_start_ms=START_MS)
    up = _dataset(1, event_start_ms=START_MS + 300_000)
    conditions = (
        build_round17_cohort_condition(plan, down, source_slot_index=0),
        build_round17_cohort_condition(plan, up, source_slot_index=0),
    )
    down_market = _market(down)
    assert build_round17_cohort_condition_label(
        plan,
        conditions[0],
        down_market,
        _resolution(down, market=down_market, outcome="Down"),
    ) == _label(down, outcome="Down")
    manifest = build_round17_cohort_manifest(plan, conditions)

    assert manifest.role_condition_counts["train"] == 2
    assert manifest.role_condition_counts["test"] == 0
    assert manifest.identity_payload()["labels_consulted"] is False
    assert manifest.identity_payload()["test_features_accessed"] is False
    targets = build_round17_development_target_manifest(
        plan,
        manifest,
        (_label(down, outcome="Down"), _label(up, outcome="Up")),
    )
    assert targets.identity_payload()["development_labels_consulted"] is True
    assert targets.identity_payload()["test_targets_accessed"] is False
    panel = build_round17_development_panel(
        plan,
        manifest,
        targets,
        role="train",
        datasets=(down, up),
    )
    assert panel.role == "train"
    assert panel.dataset_sha256 == targets.development_dataset_sha256
    assert panel.target_manifest_sha256 == targets.target_manifest_sha256
    assert panel.features.dtype == np.float32
    assert set(panel.labels) == {0.0, 1.0}
    panels = build_round17_development_panels_streaming(
        plan,
        manifest,
        targets,
        iter((down, up)),
    )
    assert tuple(panels) == ("train",)
    assert panels["train"].features.dtype == np.float32
    np.testing.assert_array_equal(panels["train"].features, panel.features)
    np.testing.assert_array_equal(panels["train"].labels, panel.labels)

    with pytest.raises(ValueError, match="role differs"):
        build_round17_development_panel(
            plan,
            manifest,
            targets,
            role="test",
            datasets=(down, up),
        )
    with pytest.raises(ValueError, match="condition order differs"):
        build_round17_development_panels_streaming(
            plan,
            manifest,
            targets,
            iter((up, down)),
        )
    with pytest.raises(ValueError, match="conditions are incomplete"):
        build_round17_development_panels_streaming(
            plan,
            manifest,
            targets,
            iter((down,)),
        )
    with pytest.raises(ValueError, match="extra conditions"):
        build_round17_development_panels_streaming(
            plan,
            manifest,
            targets,
            iter((down, up, down)),
        )
    with pytest.raises(TypeError, match="dataset type differs"):
        build_round17_development_panels_streaming(
            plan,
            manifest,
            targets,
            iter((object(),)),  # type: ignore[arg-type]
        )


def test_round17_official_resolution_acquisition_is_development_only() -> None:
    plan = load_round17_cohort_plan(PLAN_PATH)
    down = _dataset(0, event_start_ms=START_MS)
    up = _dataset(1, event_start_ms=START_MS + 300_000)
    references = (
        build_round17_cohort_condition(plan, down, source_slot_index=0),
        build_round17_cohort_condition(plan, up, source_slot_index=0),
    )
    cohort = build_round17_cohort_manifest(plan, references)
    markets = {
        down.condition_id: _market(down),
        up.condition_id: _market(up),
    }
    client = _ResolutionClient(markets)

    acquisition = acquire_round17_development_resolutions(
        plan,
        cohort,
        markets,
        client=client,  # type: ignore[arg-type]
        wall_clock_ms=lambda: up.event_end_ms + 1,
        monotonic_clock_ns=lambda: 123,
    )
    labels = acquisition.labels(plan, cohort, markets)

    assert acquisition.complete is True
    assert acquisition.asdict()["test_targets_accessed"] is False
    assert acquisition.gamma_batch_request_count == 1
    assert acquisition.clob_market_request_count == 2
    assert len(client.gamma_calls) == 1
    assert set(client.clob_calls) == set(markets)
    assert {item.winning_outcome for item in labels} == {"Up", "Down"}
    target_manifest = build_round17_development_target_manifest(
        plan,
        cohort,
        labels,
    )
    assert target_manifest.identity_payload()["test_targets_accessed"] is False
    with pytest.raises(ValueError, match="integrity differs"):
        replace(acquisition, acquisition_sha256="f" * 64).validated(
            plan,
            cohort,
            markets,
        )


def test_round17_official_resolution_acquisition_blocks_pending_and_early() -> None:
    plan = load_round17_cohort_plan(PLAN_PATH)
    down = _dataset(0, event_start_ms=START_MS)
    up = _dataset(1, event_start_ms=START_MS + 300_000)
    references = (
        build_round17_cohort_condition(plan, down, source_slot_index=0),
        build_round17_cohort_condition(plan, up, source_slot_index=0),
    )
    cohort = build_round17_cohort_manifest(plan, references)
    markets = {
        down.condition_id: _market(down),
        up.condition_id: _market(up),
    }
    client = _ResolutionClient(
        markets,
        pending_condition_id=up.condition_id,
    )
    acquisition = acquire_round17_development_resolutions(
        plan,
        cohort,
        markets,
        client=client,  # type: ignore[arg-type]
        wall_clock_ms=lambda: up.event_end_ms + 1,
        monotonic_clock_ns=lambda: 123,
    )

    assert acquisition.complete is False
    assert acquisition.pending_condition_ids == (up.condition_id,)
    with pytest.raises(RuntimeError, match="resolutions are incomplete"):
        acquisition.labels(plan, cohort, markets)
    with pytest.raises(RuntimeError, match="have not all ended"):
        acquire_round17_development_resolutions(
            plan,
            cohort,
            markets,
            client=client,  # type: ignore[arg-type]
            wall_clock_ms=lambda: up.event_end_ms - 1,
        )


def test_round17_cohort_rejects_duplicates_and_wrong_winning_token() -> None:
    plan = load_round17_cohort_plan(PLAN_PATH)
    dataset = _dataset(0, event_start_ms=START_MS)
    condition = build_round17_cohort_condition(
        plan,
        dataset,
        source_slot_index=0,
    )
    with pytest.raises(ValueError, match="empty or duplicated"):
        build_round17_cohort_manifest(plan, (condition, condition))

    market = _market(dataset)
    wrong = PolymarketResolutionEvidence(
        run_id=dataset.run_id,
        event_id="resolution-wrong-token",
        condition_id=dataset.condition_id,
        winning_asset_id=market.down_token_id,
        winning_outcome="Up",
        resolved_at_ms=dataset.event_end_ms,
        received_wall_ms=dataset.event_end_ms,
        received_monotonic_ns=dataset.event_end_ms * 1_000_000,
        event_sha256="f" * 64,
        source="clob_gamma_crosscheck",
    )
    with pytest.raises(ValueError, match="identity differs"):
        build_round17_condition_label(dataset, market, wrong)


def test_round17_cohort_plan_rejects_rehashed_role_drift() -> None:
    payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    payload["roles"][0]["last_slot"] = 670
    payload.pop("plan_sha256")
    payload["plan_sha256"] = _sha256(payload)

    with pytest.raises(ValueError, match="integrity differs"):
        validate_round17_cohort_plan(payload)
