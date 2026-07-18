from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import pytest

from simple_ai_trading import (
    polymarket_action_pipeline as action_pipeline_module,
    cli,
    polymarket_continuity as continuity_module,
    polymarket_features as feature_module,
)
from simple_ai_trading.command_contract import command_specs
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_action_value import (
    build_polymarket_action_value_dataset,
    materialize_polymarket_action_value_dataset,
)
from simple_ai_trading.polymarket_action_pipeline import (
    PolymarketActionPipelineConfig,
    materialize_polymarket_action_value_batches,
)
from simple_ai_trading.polymarket_paper import (
    PolymarketPaperBroker,
    PolymarketPaperCoordinator,
)
from simple_ai_trading.polymarket_paper_plan import (
    POLYMARKET_PAPER_PLAN_SCHEMA_VERSION,
    PolymarketPaperPlan,
    run_polymarket_paper_plan,
)
from simple_ai_trading.polymarket_features import (
    POLYMARKET_FEATURE_NAMES,
    PolymarketFeatureConfig,
    build_polymarket_feature_dataset,
    load_polymarket_feature_source_context,
    materialize_polymarket_feature_dataset,
    polymarket_feature_row_sha256,
)
from simple_ai_trading.polymarket_continuity import (
    evaluate_polymarket_continuity_eligibility,
)
from simple_ai_trading.polymarket_model import (
    POLYMARKET_MODEL_FEATURE_NAMES,
    POLYMARKET_MODEL_RISK_CONTEXT_NAMES,
    PolymarketModelSample,
)
from simple_ai_trading.polymarket_model_execution import (
    PolymarketExecutionResearchConfig,
    evaluate_polymarket_execution_policy,
)
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from simple_ai_trading.polymarket_resolution import PolymarketResolutionFinalizer
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay
from simple_ai_trading.polymarket_repricing import (
    PolymarketRepricingExecutionContext,
)
from simple_ai_trading.polymarket_round12_admission import (
    build_round12_action_evidence_datasets,
    load_round12_admission_dataset,
    materialize_round12_admission_dataset,
)
from simple_ai_trading.polymarket_round12_capture import (
    create_round12_capture_manifest,
)
from simple_ai_trading.polymarket_round12_reference import (
    load_round12_confirmation_contract,
    load_round12_reference_from_round11_artifact,
    polymarket_round12_primary_policy,
)
from simple_ai_trading.polymarket_round13 import (
    PolymarketRound13Program,
    load_round13_label_free_dataset,
    polymarket_round13_evaluation_gates,
    polymarket_round13_scenarios,
)


EPOCH = 1_784_058_600
ROUND12_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-012-fixed-calibration-confirmation-contract.json"
)
ROUND11_ARTIFACT = ROUND12_CONTRACT.with_name(
    "round-011-single-leg-directional-value-artifact.json"
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _round13_program(contract_sha256: str = "d" * 64) -> PolymarketRound13Program:
    return PolymarketRound13Program(
        contract={"contract_sha256": contract_sha256},
        contract_sha256=contract_sha256,
        model=load_round12_reference_from_round11_artifact(ROUND11_ARTIFACT),
        policy=polymarket_round12_primary_policy(),
        scenarios=polymarket_round13_scenarios(),
        evaluation_gates=polymarket_round13_evaluation_gates(),
        confirmation_capital_quote=Decimal("1000"),
    ).validated()


def _round12_capture_manifest(
    run_id: str,
    started_at_ms: int,
) -> dict[str, object]:
    contract = load_round12_confirmation_contract(ROUND12_CONTRACT)
    implementation = contract["implementation"]
    model = contract["model_contract"]
    policy = contract["primary_policy"]
    return create_round12_capture_manifest(
        run_id=run_id,
        started_at_ms=started_at_ms,
        repository_commit="b" * 40,
        contract_sha256=str(contract["contract_sha256"]),
        model_sha256=str(model["model_sha256"]),
        policy_sha256=str(policy["policy_sha256"]),
        reference_implementation_sha256=str(
            implementation["reference_implementation_sha256"]
        ),
        action_pipeline_implementation_sha256=str(
            implementation["action_pipeline_implementation_sha256"]
        ),
        required_git_blob_sha256={"tests/fixture.py": "a" * 64},
    )


def _market_payload(asset: str) -> dict[str, object]:
    token_base = {"BTC": "7", "ETH": "8", "SOL": "9"}[asset]
    return {
        "id": f"market-{asset}",
        "question": f"{asset} Up or Down",
        "conditionId": "0x" + token_base * 64,
        "slug": f"{asset.lower()}-updown-5m-{EPOCH}",
        "eventStartTime": "2026-07-14T19:50:00Z",
        "endDate": "2026-07-14T19:55:00Z",
        "active": True,
        "closed": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "clobTokenIds": json.dumps([token_base * 40, token_base * 39 + "1"]),
        "outcomes": '["Up", "Down"]',
        "orderPriceMinTickSize": 0.01,
        "orderMinSize": 5,
        "feesEnabled": True,
        "feeSchedule": {
            "exponent": 1,
            "rate": 0.07,
            "takerOnly": True,
            "rebateRate": 0.2,
        },
        "liquidityNum": 20_000.5,
        "volumeNum": 50_000.25,
        "resolutionSource": f"https://data.chain.link/streams/{asset.lower()}-usd",
    }


def _evidence(asset: str) -> MarketEvidence:
    market = parse_polymarket_five_minute_market(_market_payload(asset))
    clob = _canonical({"c": market.condition_id, "t": list(market.token_ids)})
    fee = _canonical({"base_fee": 1000})
    return MarketEvidence(
        market=market,
        observed_wall_ms=EPOCH * 1_000 + 100,
        observed_monotonic_ns=100_000_000,
        clob_info_json=clob,
        clob_info_sha256=_sha(clob),
        up_fee_rate_json=fee,
        up_fee_rate_sha256=_sha(fee),
        down_fee_rate_json=fee,
        down_fee_rate_sha256=_sha(fee),
        maker_base_fee=1000,
        taker_base_fee=1000,
        taker_order_delay_enabled=False,
        minimum_order_age_seconds=0,
    )


def _official_payloads(asset: str) -> tuple[dict[str, object], dict[str, object]]:
    gamma = deepcopy(_market_payload(asset))
    market = parse_polymarket_five_minute_market(gamma)
    gamma.update(
        {
            "closed": True,
            "active": False,
            "acceptingOrders": False,
            "outcomePrices": '["1", "0"]',
        }
    )
    clob = {
        "condition_id": market.condition_id,
        "market_slug": market.slug,
        "closed": True,
        "active": False,
        "accepting_orders": False,
        "tokens": [
            {
                "token_id": market.up_token_id,
                "outcome": "Up",
                "price": 1,
                "winner": True,
            },
            {
                "token_id": market.down_token_id,
                "outcome": "Down",
                "price": 0,
                "winner": False,
            },
        ],
    }
    return clob, gamma


class _OfficialClient:
    def __init__(self) -> None:
        self.by_condition: dict[str, dict[str, object]] = {}
        self.by_market: dict[str, dict[str, object]] = {}
        for asset in ("BTC", "ETH", "SOL"):
            clob, gamma = _official_payloads(asset)
            self.by_condition[str(clob["condition_id"])] = clob
            self.by_market[str(gamma["id"])] = gamma

    def clob_market(self, condition_id: str) -> dict[str, object]:
        return deepcopy(self.by_condition[condition_id])

    def gamma_market(self, market_id: str) -> dict[str, object]:
        return deepcopy(self.by_market[market_id])


def _message(
    stream: str,
    payload: object,
    *,
    sequence: int,
    wall_offset_ms: int,
    monotonic_ns: int,
) -> RawStreamMessage:
    return RawStreamMessage(
        stream=stream,
        connection_id=f"{stream}-connection",
        sequence_number=sequence,
        received_wall_ms=EPOCH * 1_000 + wall_offset_ms,
        received_monotonic_ns=monotonic_ns,
        raw_text=_canonical(payload),
    )


def _segmented_message(
    payload: object,
    *,
    stream: str = "clob_market",
    connection_id: str,
    sequence: int,
    wall_offset_ms: int,
    monotonic_ns: int,
) -> RawStreamMessage:
    return RawStreamMessage(
        stream=stream,
        connection_id=connection_id,
        sequence_number=sequence,
        received_wall_ms=EPOCH * 1_000 + wall_offset_ms,
        received_monotonic_ns=monotonic_ns,
        raw_text=_canonical(payload),
    )


def _book_payload(
    token: str,
    condition: str,
    source_offset_ms: int,
    *,
    tick_size: str | None = "0.01",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_type": "book",
        "market": condition,
        "asset_id": token,
        "timestamp": str(EPOCH * 1_000 + source_offset_ms),
        "hash": f"book-{token[-4:]}-{source_offset_ms}",
        "bids": [{"price": "0.49", "size": "10"}],
        "asks": [{"price": "0.51", "size": "10"}],
    }
    if tick_size is not None:
        payload["tick_size"] = tick_size
    return payload


def _finish_segmented_store(
    store: PolymarketEvidenceStore,
    run_id: str,
    *,
    gap_stream: str = "clob_market",
    gap_last_sequence: int = 3,
    second_segment_has_baseline: bool = True,
    named_clob_connections: bool = False,
    second_segment_tick_size: str | None = "0.01",
) -> None:
    store.start_run(run_id, EPOCH * 1_000)
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    first = (
        "clob:11111111111111111111111111111111"
        if named_clob_connections
        else "clob-segment-one"
    )
    second = (
        "clob:22222222222222222222222222222222"
        if named_clob_connections
        else "clob-segment-two"
    )
    first_messages = [
        _segmented_message(
            _book_payload(btc.up_token_id, btc.condition_id, 1_000),
            connection_id=first,
            sequence=1,
            wall_offset_ms=1_001,
            monotonic_ns=1_001_000_000,
        ),
        _segmented_message(
            _book_payload(btc.down_token_id, btc.condition_id, 1_000),
            connection_id=first,
            sequence=2,
            wall_offset_ms=1_002,
            monotonic_ns=1_002_000_000,
        ),
        _segmented_message(
            _book_payload(btc.up_token_id, btc.condition_id, 2_000),
            connection_id=first,
            sequence=3,
            wall_offset_ms=2_001,
            monotonic_ns=2_001_000_000,
        ),
    ]
    store.append_messages(run_id, first_messages)
    if gap_stream == "clob_market":
        gap_connection = first
        last_sequence = gap_last_sequence
    else:
        gap_connection = "binance-connection"
        last_sequence = 1
        store.append_messages(
            run_id,
            [
                _segmented_message(
                    {"fixture": True},
                    stream=gap_stream,
                    connection_id=gap_connection,
                    sequence=1,
                    wall_offset_ms=2_100,
                    monotonic_ns=2_100_000_000,
                )
            ],
        )
    store.record_gap(
        run_id,
        StreamGap(
            stream=gap_stream,
            connection_id=gap_connection,
            opened_at_ms=EPOCH * 1_000 + 2_500,
            reason="fixture_disconnect",
            last_sequence_number=last_sequence,
        ),
    )
    if second_segment_has_baseline:
        second_messages = [
            _segmented_message(
                _book_payload(
                    btc.up_token_id,
                    btc.condition_id,
                    3_000,
                    tick_size=second_segment_tick_size,
                ),
                connection_id=second,
                sequence=1,
                wall_offset_ms=3_001,
                monotonic_ns=3_001_000_000,
            ),
            _segmented_message(
                _book_payload(
                    btc.down_token_id,
                    btc.condition_id,
                    3_000,
                    tick_size=second_segment_tick_size,
                ),
                connection_id=second,
                sequence=2,
                wall_offset_ms=3_002,
                monotonic_ns=3_002_000_000,
            ),
        ]
    else:
        second_messages = [
            _segmented_message(
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(EPOCH * 1_000 + 3_000),
                    "price_changes": [
                        {
                            "asset_id": btc.up_token_id,
                            "price": "0.49",
                            "size": "0",
                            "side": "BUY",
                            "hash": "missing-baseline",
                            "best_bid": "0",
                            "best_ask": "0.51",
                        }
                    ],
                },
                connection_id=second,
                sequence=1,
                wall_offset_ms=3_001,
                monotonic_ns=3_001_000_000,
            )
        ]
    store.append_messages(run_id, second_messages)
    existing_streams = {gap_stream} if gap_stream != "clob_market" else set()
    for stream, connection_id in (
        ("polymarket_rtds", "rtds-connection"),
        ("binance_spot", "binance-connection"),
    ):
        if stream in existing_streams:
            continue
        store.append_messages(
            run_id,
            [
                _segmented_message(
                    {"fixture": True},
                    stream=stream,
                    connection_id=connection_id,
                    sequence=1,
                    wall_offset_ms=4_000,
                    monotonic_ns=4_000_000_000,
                )
            ],
        )
    report = store.finish_run(
        run_id,
        started_at_ms=EPOCH * 1_000,
        ended_at_ms=EPOCH * 1_000 + 5_000,
        database=":memory:",
        errors=(),
    )
    assert report.status == "degraded"


def _finish_replay_store(
    store: PolymarketEvidenceStore,
    run_id: str,
    *,
    wrong_best: bool = False,
    trade_resync: bool = False,
    trade_resync_lag_ms: int = 1,
    trade_resync_top_change: bool = False,
    feature_evidence: bool = False,
    synchronized_feature_evidence: bool = False,
    pre_window_trade_quantity: str = "0.1",
    duplicate_tick_transition: bool = False,
    duplicate_tick_source_delta_ms: int = 0,
    duplicate_tick_monotonic_ns: int = 1_013_000_000,
    terminal_clear_burst: bool = False,
    interleaved_best_transitions: bool = False,
    compact_resolution_event: bool = False,
    finalize_official: bool = True,
    run_started_at_ms: int = EPOCH * 1_000,
    run_ended_at_ms: int = EPOCH * 1_000 + 302_000,
    pre_window_binance_gap: bool = False,
    additional_messages: tuple[RawStreamMessage, ...] = (),
    preregistration_manifest: Mapping[str, object] | None = None,
) -> None:
    if synchronized_feature_evidence and not feature_evidence:
        raise ValueError("synchronized feature evidence requires feature evidence")
    store.start_run(
        run_id,
        run_started_at_ms,
        preregistration_manifest=preregistration_manifest,
    )
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
    if pre_window_binance_gap:
        store.record_gap(
            run_id,
            StreamGap(
                "binance_spot",
                "retired-binance-connection",
                EPOCH * 1_000 - 3_000,
                "fixture_disconnect",
                0,
            ),
        )
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    reported_best_ask = "0.49" if wrong_best else "0.50"
    resolution_payload: dict[str, object] = {
        "event_type": "market_resolved",
        "id": btc.market_id,
        "question": btc.question,
        "market": btc.condition_id,
        "slug": btc.slug,
        "assets_ids": list(btc.token_ids),
        "outcomes": ["Up", "Down"],
        "winning_asset_id": btc.up_token_id,
        "winning_outcome": "Up",
        "timestamp": str(btc.end_ms + 1_000),
    }
    if compact_resolution_event:
        for optional_field in ("id", "question", "slug", "outcomes"):
            resolution_payload.pop(optional_field)
    clob_messages = [
        _message(
            "clob_market",
            {
                "event_type": "book",
                "market": btc.condition_id,
                "asset_id": token,
                "timestamp": str(EPOCH * 1_000 + 1_000),
                "tick_size": "0.01",
                "hash": "full-book",
                "bids": [{"price": "0.49", "size": "10"}],
                "asks": [{"price": "0.51", "size": "10"}],
            },
            sequence=1,
            wall_offset_ms=1_001,
            monotonic_ns=1_000_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "best_bid_ask",
                "market": btc.condition_id,
                "asset_id": token,
                "best_bid": "0",
                "best_ask": reported_best_ask,
                "spread": reported_best_ask,
                "timestamp": str(EPOCH * 1_000 + 1_011),
            },
            sequence=2,
            wall_offset_ms=1_010,
            monotonic_ns=1_009_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(EPOCH * 1_000 + 1_010),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.49",
                        "size": "0",
                        "side": "BUY",
                        "hash": "atomic-replacement",
                        "best_bid": "0",
                        "best_ask": reported_best_ask,
                    }
                ],
            },
            sequence=3,
            wall_offset_ms=1_011,
            monotonic_ns=1_010_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(EPOCH * 1_000 + 1_010),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.50",
                        "size": "8",
                        "side": "SELL",
                        "hash": "atomic-replacement",
                        "best_bid": "0",
                        "best_ask": reported_best_ask,
                    }
                ],
            },
            sequence=4,
            wall_offset_ms=1_012,
            monotonic_ns=1_011_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "tick_size_change",
                "market": btc.condition_id,
                "asset_id": token,
                "old_tick_size": "0.01",
                "new_tick_size": "0.001",
                "timestamp": str(EPOCH * 1_000 + 1_012),
            },
            sequence=5,
            wall_offset_ms=1_013,
            monotonic_ns=1_012_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(EPOCH * 1_000 + 1_020),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.499",
                        "size": "5",
                        "side": "BUY",
                        "hash": "new-bid",
                        "best_bid": "0.499",
                        "best_ask": "0.50",
                    }
                ],
            },
            sequence=6,
            wall_offset_ms=1_021,
            monotonic_ns=1_020_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(EPOCH * 1_000 + 1_030),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.499",
                        "size": "0",
                        "side": "BUY",
                        "hash": "remove-new-bid",
                        "best_bid": "0.498",
                        "best_ask": "0.50",
                    },
                    {
                        "asset_id": token,
                        "price": "0.498",
                        "size": "5",
                        "side": "BUY",
                        "hash": "close-bid",
                        "best_bid": "0.498",
                        "best_ask": "0.50",
                    },
                ],
            },
            sequence=7,
            wall_offset_ms=1_031,
            monotonic_ns=1_030_000_000,
        ),
        _message(
            "clob_market",
            resolution_payload,
            sequence=8,
            wall_offset_ms=301_000,
            monotonic_ns=301_000_000_000,
        ),
    ]
    if duplicate_tick_transition:
        clob_messages.insert(
            5,
            _message(
                "clob_market",
                {
                    "event_type": "tick_size_change",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "old_tick_size": "0.01",
                    "new_tick_size": "0.001",
                    "timestamp": str(
                        EPOCH * 1_000 + 1_012 + duplicate_tick_source_delta_ms
                    ),
                },
                sequence=50,
                wall_offset_ms=1_014,
                monotonic_ns=duplicate_tick_monotonic_ns,
            ),
        )
    if terminal_clear_burst:
        terminal_source_ms = EPOCH * 1_000 + 2_000
        clob_messages[-1:-1] = [
            _message(
                "clob_market",
                {
                    "event_type": "best_bid_ask",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "best_bid": "0",
                    "best_ask": "1",
                    "spread": "1",
                    "timestamp": str(terminal_source_ms),
                },
                sequence=60,
                wall_offset_ms=2_001,
                monotonic_ns=2_000_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(terminal_source_ms),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.498",
                            "size": "0",
                            "side": "BUY",
                            "hash": "terminal-up-hash",
                            "best_bid": "0",
                            "best_ask": "1",
                        }
                    ],
                },
                sequence=61,
                wall_offset_ms=2_002,
                monotonic_ns=2_000_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "best_bid_ask",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "best_bid": "0",
                    "best_ask": "1",
                    "spread": "1",
                    "timestamp": str(terminal_source_ms + 1),
                },
                sequence=62,
                wall_offset_ms=2_003,
                monotonic_ns=2_000_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(terminal_source_ms),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.50",
                            "size": "0",
                            "side": "SELL",
                            "hash": "terminal-up-hash",
                            "best_bid": "0",
                            "best_ask": "1",
                        },
                        {
                            "asset_id": token,
                            "price": "0.51",
                            "size": "0",
                            "side": "SELL",
                            "hash": "terminal-up-hash",
                            "best_bid": "0",
                            "best_ask": "1",
                        },
                    ],
                },
                sequence=63,
                wall_offset_ms=2_004,
                monotonic_ns=2_015_000_000,
            ),
        ]
    if interleaved_best_transitions:
        transition_source_ms = EPOCH * 1_000 + 2_000
        clob_messages[-1:-1] = [
            _message(
                "clob_market",
                {
                    "event_type": "best_bid_ask",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "best_bid": "0",
                    "best_ask": "0.51",
                    "spread": "0.51",
                    "timestamp": str(transition_source_ms),
                },
                sequence=70,
                wall_offset_ms=2_001,
                monotonic_ns=2_000_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(transition_source_ms),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.498",
                            "size": "0",
                            "side": "BUY",
                            "hash": "evolving-shared-hash",
                            "best_bid": "0",
                            "best_ask": "0.51",
                        },
                        {
                            "asset_id": token,
                            "price": "0.50",
                            "size": "0",
                            "side": "SELL",
                            "hash": "evolving-shared-hash",
                            "best_bid": "0",
                            "best_ask": "0.51",
                        },
                    ],
                },
                sequence=71,
                wall_offset_ms=2_002,
                monotonic_ns=2_001_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(transition_source_ms),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.20",
                            "size": "3",
                            "side": "BUY",
                            "hash": "evolving-shared-hash",
                            "best_bid": "0.20",
                            "best_ask": "0.51",
                        }
                    ],
                },
                sequence=72,
                wall_offset_ms=2_003,
                monotonic_ns=2_002_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "best_bid_ask",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "best_bid": "0.49",
                    "best_ask": "0.51",
                    "spread": "0.02",
                    "timestamp": str(transition_source_ms + 4),
                },
                sequence=73,
                wall_offset_ms=2_004,
                monotonic_ns=2_003_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(transition_source_ms + 4),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.49",
                            "size": "4",
                            "side": "BUY",
                            "hash": "evolving-shared-hash",
                            "best_bid": "0.49",
                            "best_ask": "0.51",
                        }
                    ],
                },
                sequence=74,
                wall_offset_ms=2_005,
                monotonic_ns=2_004_000_000,
            ),
        ]
    if trade_resync:
        delta_price = "0.49" if trade_resync_top_change else "0.53"
        delta_side = "BUY" if trade_resync_top_change else "SELL"
        delta_best_bid = "0.49" if trade_resync_top_change else "0"
        resync_messages = [
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(EPOCH * 1_000 + 1_011),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": delta_price,
                            "size": "6",
                            "side": delta_side,
                            "hash": "post-trade-delta",
                            "best_bid": delta_best_bid,
                            "best_ask": "0.50",
                        }
                    ],
                },
                sequence=40,
                wall_offset_ms=1_012,
                monotonic_ns=1_011_200_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "book",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "timestamp": str(EPOCH * 1_000 + 1_011 - trade_resync_lag_ms),
                    "tick_size": "0.01",
                    "hash": "atomic-replacement",
                    "bids": [],
                    "asks": [
                        {"price": "0.50", "size": "8"},
                        {"price": "0.52", "size": "7"},
                    ],
                },
                sequence=41,
                wall_offset_ms=1_012,
                monotonic_ns=1_011_500_000,
            ),
        ]
        if trade_resync_top_change:
            resync_messages.append(
                _message(
                    "clob_market",
                    {
                        "event_type": "price_change",
                        "market": btc.condition_id,
                        "timestamp": str(EPOCH * 1_000 + 1_011),
                        "price_changes": [
                            {
                                "asset_id": token,
                                "price": "0.99",
                                "size": "5",
                                "side": "SELL",
                                "hash": "post-resync-followup",
                                "best_bid": "0.49",
                                "best_ask": "0.50",
                            }
                        ],
                    },
                    sequence=42,
                    wall_offset_ms=1_012,
                    monotonic_ns=1_011_700_000,
                )
            )
        clob_messages[4:4] = resync_messages
    if feature_evidence:
        clob_messages.insert(
            1,
            _message(
                "clob_market",
                {
                    "event_type": "book",
                    "market": btc.condition_id,
                    "asset_id": btc.down_token_id,
                    "timestamp": str(EPOCH * 1_000 + 1_000),
                    "tick_size": "0.01",
                    "hash": "down-full-book",
                    "bids": [{"price": "0.49", "size": "12"}],
                    "asks": [{"price": "0.51", "size": "11"}],
                },
                sequence=39,
                wall_offset_ms=1_001,
                monotonic_ns=1_000_100_000,
            ),
        )
        clob_messages.extend(
            [
                _message(
                    "clob_market",
                    {
                        "event_type": "book",
                        "market": btc.condition_id,
                        "asset_id": btc.up_token_id,
                        "timestamp": str(EPOCH * 1_000 + 6_000),
                        "tick_size": "0.001",
                        "hash": "up-feature-book",
                        "bids": [{"price": "0.49", "size": "10"}],
                        "asks": [{"price": "0.51", "size": "10"}],
                    },
                    sequence=42,
                    wall_offset_ms=6_001,
                    monotonic_ns=6_000_000_000,
                ),
                _message(
                    "clob_market",
                    {
                        "event_type": "book",
                        "market": btc.condition_id,
                        "asset_id": btc.down_token_id,
                        "timestamp": str(EPOCH * 1_000 + 6_000),
                        "tick_size": "0.01",
                        "hash": "down-feature-book",
                        "bids": [{"price": "0.49", "size": "12"}],
                        "asks": [{"price": "0.51", "size": "11"}],
                    },
                    sequence=43,
                    wall_offset_ms=6_001,
                    monotonic_ns=6_001_000_000,
                ),
            ]
        )
        if synchronized_feature_evidence:
            for asset_index, asset in enumerate(("ETH", "SOL"), start=1):
                market = parse_polymarket_five_minute_market(_market_payload(asset))
                for outcome_index, token_id in enumerate(market.token_ids):
                    offset = asset_index * 2 + outcome_index
                    clob_messages.append(
                        _message(
                            "clob_market",
                            {
                                "event_type": "book",
                                "market": market.condition_id,
                                "asset_id": token_id,
                                "timestamp": str(EPOCH * 1_000 + 6_000),
                                "tick_size": "0.01",
                                "hash": (
                                    f"{asset.lower()}-"
                                    f"{('up', 'down')[outcome_index]}-"
                                    "feature-book"
                                ),
                                "bids": [{"price": "0.49", "size": "12"}],
                                "asks": [{"price": "0.51", "size": "11"}],
                            },
                            sequence=50 + offset,
                            wall_offset_ms=6_001,
                            monotonic_ns=6_000_000_000 + offset * 1_000_000,
                        )
                    )
    auxiliary = [
        _message(
            "polymarket_rtds",
            {
                "topic": "crypto_prices",
                "type": "update",
                "timestamp": EPOCH * 1_000 + 1_000,
                "payload": {
                    "symbol": "btcusdt",
                    "timestamp": EPOCH * 1_000 + 999,
                    "value": 60_000,
                },
            },
            sequence=1,
            wall_offset_ms=1_001,
            monotonic_ns=1_000_500_000,
        ),
        _message(
            "binance_spot",
            {
                "stream": "btcusdt@trade",
                "data": {
                    "e": "trade",
                    "E": EPOCH * 1_000 + 1_000,
                    "T": EPOCH * 1_000 + 999,
                },
            },
            sequence=1,
            wall_offset_ms=1_001,
            monotonic_ns=1_000_600_000,
        ),
    ]
    if feature_evidence:
        auxiliary = [
            _message(
                "polymarket_rtds",
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "subscribe",
                    "timestamp": EPOCH * 1_000 + 800,
                    "payload": {
                        "symbol": "btc/usd",
                        "data": [
                            {"timestamp": EPOCH * 1_000, "value": 60_000},
                            {
                                "timestamp": EPOCH * 1_000 + 1_000,
                                "value": 60_006,
                            },
                        ],
                    },
                },
                sequence=1,
                wall_offset_ms=800,
                monotonic_ns=800_000_000,
            ),
            _message(
                "polymarket_rtds",
                {
                    "topic": "crypto_prices",
                    "type": "subscribe",
                    "timestamp": EPOCH * 1_000 + 810,
                    "payload": {
                        "symbol": "btcusdt",
                        "data": [
                            {"timestamp": EPOCH * 1_000, "value": 60_001},
                            {
                                "timestamp": EPOCH * 1_000 + 1_000,
                                "value": 60_007,
                            },
                        ],
                    },
                },
                sequence=2,
                wall_offset_ms=810,
                monotonic_ns=810_000_000,
            ),
            _message(
                "binance_spot",
                {
                    "stream": "btcusdt@bookTicker",
                    "data": {
                        "u": 1,
                        "s": "BTCUSDT",
                        "b": "60005",
                        "B": "2",
                        "a": "60007",
                        "A": "3",
                    },
                },
                sequence=1,
                wall_offset_ms=820,
                monotonic_ns=820_000_000,
            ),
            _message(
                "binance_spot",
                {
                    "stream": "btcusdt@trade",
                    "data": {
                        "e": "trade",
                        "E": EPOCH * 1_000 + 830,
                        "T": EPOCH * 1_000 + 829,
                        "s": "BTCUSDT",
                        "p": "60006",
                        "q": pre_window_trade_quantity,
                        "m": False,
                    },
                },
                sequence=2,
                wall_offset_ms=830,
                monotonic_ns=830_000_000,
            ),
            _message(
                "polymarket_rtds",
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "update",
                    "timestamp": EPOCH * 1_000 + 5_100,
                    "payload": {
                        "symbol": "btc/usd",
                        "timestamp": EPOCH * 1_000 + 5_000,
                        "value": 60_006,
                    },
                },
                sequence=3,
                wall_offset_ms=5_100,
                monotonic_ns=5_100_000_000,
            ),
            _message(
                "polymarket_rtds",
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "timestamp": EPOCH * 1_000 + 5_110,
                    "payload": {
                        "symbol": "btcusdt",
                        "timestamp": EPOCH * 1_000 + 5_000,
                        "value": 60_007,
                    },
                },
                sequence=4,
                wall_offset_ms=5_110,
                monotonic_ns=5_110_000_000,
            ),
            _message(
                "binance_spot",
                {
                    "stream": "btcusdt@bookTicker",
                    "data": {
                        "u": 2,
                        "s": "BTCUSDT",
                        "b": "60005",
                        "B": "2.5",
                        "a": "60007",
                        "A": "2.5",
                    },
                },
                sequence=3,
                wall_offset_ms=5_800,
                monotonic_ns=5_800_000_000,
            ),
            _message(
                "binance_spot",
                {
                    "stream": "btcusdt@trade",
                    "data": {
                        "e": "trade",
                        "E": EPOCH * 1_000 + 5_900,
                        "T": EPOCH * 1_000 + 5_899,
                        "s": "BTCUSDT",
                        "p": "60006",
                        "q": "0.2",
                        "m": True,
                    },
                },
                sequence=4,
                wall_offset_ms=5_900,
                monotonic_ns=5_900_000_000,
            ),
        ]
        if synchronized_feature_evidence:
            reference_prices = {"ETH": 3_000, "SOL": 150}
            for asset_index, asset in enumerate(("ETH", "SOL"), start=1):
                symbol = f"{asset.lower()}usdt"
                reference = reference_prices[asset]
                auxiliary.extend(
                    [
                        _message(
                            "polymarket_rtds",
                            {
                                "topic": "crypto_prices_chainlink",
                                "type": "subscribe",
                                "timestamp": EPOCH * 1_000 + 800 + asset_index,
                                "payload": {
                                    "symbol": f"{asset.lower()}/usd",
                                    "data": [
                                        {
                                            "timestamp": EPOCH * 1_000,
                                            "value": reference,
                                        },
                                        {
                                            "timestamp": EPOCH * 1_000 + 1_000,
                                            "value": reference * 1.0001,
                                        },
                                    ],
                                },
                            },
                            sequence=10 + asset_index,
                            wall_offset_ms=800 + asset_index,
                            monotonic_ns=(800 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "polymarket_rtds",
                            {
                                "topic": "crypto_prices",
                                "type": "subscribe",
                                "timestamp": EPOCH * 1_000 + 810 + asset_index,
                                "payload": {
                                    "symbol": symbol,
                                    "data": [
                                        {
                                            "timestamp": EPOCH * 1_000,
                                            "value": reference * 1.00001,
                                        },
                                        {
                                            "timestamp": EPOCH * 1_000 + 1_000,
                                            "value": reference * 1.00011,
                                        },
                                    ],
                                },
                            },
                            sequence=20 + asset_index,
                            wall_offset_ms=810 + asset_index,
                            monotonic_ns=(810 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "binance_spot",
                            {
                                "stream": f"{symbol}@bookTicker",
                                "data": {
                                    "u": 1,
                                    "s": f"{asset}USDT",
                                    "b": str(reference * 1.00008),
                                    "B": "20",
                                    "a": str(reference * 1.00012),
                                    "A": "18",
                                },
                            },
                            sequence=30 + asset_index,
                            wall_offset_ms=820 + asset_index,
                            monotonic_ns=(820 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "binance_spot",
                            {
                                "stream": f"{symbol}@trade",
                                "data": {
                                    "e": "trade",
                                    "E": EPOCH * 1_000 + 830 + asset_index,
                                    "T": EPOCH * 1_000 + 829 + asset_index,
                                    "s": f"{asset}USDT",
                                    "p": str(reference * 1.0001),
                                    "q": "0.2",
                                    "m": False,
                                },
                            },
                            sequence=40 + asset_index,
                            wall_offset_ms=830 + asset_index,
                            monotonic_ns=(830 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "polymarket_rtds",
                            {
                                "topic": "crypto_prices_chainlink",
                                "type": "update",
                                "timestamp": EPOCH * 1_000 + 5_100 + asset_index,
                                "payload": {
                                    "symbol": f"{asset.lower()}/usd",
                                    "timestamp": EPOCH * 1_000 + 5_000,
                                    "value": reference * 1.0001,
                                },
                            },
                            sequence=50 + asset_index,
                            wall_offset_ms=5_100 + asset_index,
                            monotonic_ns=(5_100 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "polymarket_rtds",
                            {
                                "topic": "crypto_prices",
                                "type": "update",
                                "timestamp": EPOCH * 1_000 + 5_110 + asset_index,
                                "payload": {
                                    "symbol": symbol,
                                    "timestamp": EPOCH * 1_000 + 5_000,
                                    "value": reference * 1.00011,
                                },
                            },
                            sequence=60 + asset_index,
                            wall_offset_ms=5_110 + asset_index,
                            monotonic_ns=(5_110 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "binance_spot",
                            {
                                "stream": f"{symbol}@bookTicker",
                                "data": {
                                    "u": 2,
                                    "s": f"{asset}USDT",
                                    "b": str(reference * 1.00008),
                                    "B": "22",
                                    "a": str(reference * 1.00012),
                                    "A": "21",
                                },
                            },
                            sequence=70 + asset_index,
                            wall_offset_ms=5_800 + asset_index,
                            monotonic_ns=(5_800 + asset_index) * 1_000_000,
                        ),
                        _message(
                            "binance_spot",
                            {
                                "stream": f"{symbol}@trade",
                                "data": {
                                    "e": "trade",
                                    "E": EPOCH * 1_000 + 5_900 + asset_index,
                                    "T": EPOCH * 1_000 + 5_899 + asset_index,
                                    "s": f"{asset}USDT",
                                    "p": str(reference * 1.0001),
                                    "q": "0.3",
                                    "m": True,
                                },
                            },
                            sequence=80 + asset_index,
                            wall_offset_ms=5_900 + asset_index,
                            monotonic_ns=(5_900 + asset_index) * 1_000_000,
                        ),
                    ]
                )
    fixture_messages = [*clob_messages, *auxiliary, *additional_messages]
    next_sequence: dict[tuple[str, str], int] = {}
    normalized_messages: list[RawStreamMessage] = []
    for message in fixture_messages:
        lane = (message.stream, message.connection_id)
        sequence = next_sequence.get(lane, 0) + 1
        next_sequence[lane] = sequence
        normalized_messages.append(replace(message, sequence_number=sequence))
    store.append_messages(run_id, normalized_messages)
    report = store.finish_run(
        run_id,
        started_at_ms=run_started_at_ms,
        ended_at_ms=run_ended_at_ms,
        database=str(store.path),
        errors=(),
    )
    assert report.status == ("degraded" if pre_window_binance_gap else "complete")
    if finalize_official:
        finalized = PolymarketResolutionFinalizer(
            store,
            client=_OfficialClient(),  # type: ignore[arg-type]
            wall_clock_ms=lambda: EPOCH * 1_000 + 302_001,
            monotonic_clock_ns=lambda: 302_001_000_000,
        ).finalize(run_id=run_id)
        assert finalized.status == "complete"


def test_replay_reconstructs_depth_tick_resolution_and_post_latency_state(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "replay.duckdb") as store:
        _finish_replay_store(store, "complete-run")
        replay = PolymarketEvidenceReplay.load(store, run_id="complete-run")

    assert len(replay.books) == 4
    full, changed, post_tick, close_book = replay.books
    assert full.snapshot.bids[0].price == Decimal("0.49")
    assert full.snapshot.bids[0].quantity == Decimal("10")
    assert changed.snapshot.bids == ()
    assert changed.snapshot.asks[0].price == Decimal("0.50")
    assert changed.snapshot.source_payload_sha256 != full.snapshot.source_payload_sha256
    assert post_tick.tick_size == Decimal("0.001")
    assert post_tick.snapshot.bids[0].price == Decimal("0.499")
    assert (
        replay.first_book_after_latency(
            full, latency_ms=5, maximum_observation_delay_ms=500
        )
        == changed
    )
    assert (
        replay.first_book_after_latency(
            post_tick, latency_ms=1, maximum_observation_delay_ms=500
        )
        == close_book
    )
    assert (
        replay.first_book_after_latency(
            close_book, latency_ms=1, maximum_observation_delay_ms=500
        )
        is None
    )
    assert replay.book_for_event(changed.event_id, changed.token_id) == changed
    assert replay.resolutions[0].winning_outcome == "Up"
    assert replay.resolutions[0].source == "clob_gamma_crosscheck"


def test_replay_accepts_only_matching_persisted_continuity_proof(
    tmp_path,
    monkeypatch,
) -> None:
    proof_sha256 = "a" * 64
    with PolymarketEvidenceStore(tmp_path / "proof-replay.duckdb") as store:
        _finish_replay_store(store, "proof-replay", feature_evidence=True)
        btc = parse_polymarket_five_minute_market(_market_payload("BTC"))

        class Proof:
            report_sha256 = proof_sha256
            eligible_condition_ids = (btc.condition_id,)

        monkeypatch.setattr(
            continuity_module,
            "evaluate_polymarket_continuity_eligibility",
            lambda *_args, **_kwargs: Proof(),
        )
        monkeypatch.setattr(
            store,
            "raw_message_lane_summaries",
            lambda *_args, **_kwargs: pytest.fail(
                "validated continuity proof must bypass a global lane rescan"
            ),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="proof-replay",
            allow_segmented_gaps=True,
            condition_ids=(btc.condition_id,),
            continuity_report_sha256=proof_sha256,
        )
        with pytest.raises(ValueError, match="continuity proof differs"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="proof-replay",
                allow_segmented_gaps=True,
                condition_ids=(btc.condition_id,),
                continuity_report_sha256="b" * 64,
            )

    assert replay.books


def test_replay_can_reconstruct_only_selected_conditions(tmp_path) -> None:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    with PolymarketEvidenceStore(tmp_path / "selected-replay.duckdb") as store:
        _finish_replay_store(store, "selected-run")
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="selected-run",
            condition_ids=[btc.condition_id, btc.condition_id.upper()],
        )

    assert replay.markets == (btc,)
    assert replay.books
    assert {book.market.condition_id for book in replay.books} == {btc.condition_id}
    assert {item.condition_id for item in replay.resolutions} == {btc.condition_id}


def test_replay_rejects_empty_condition_selection(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "empty-selection.duckdb") as store:
        with pytest.raises(ValueError, match="condition_ids"):
            PolymarketEvidenceReplay.load(store, condition_ids=[])


def _repeated_resolution_message(*, conflicting: bool) -> RawStreamMessage:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    winning_asset_id = btc.up_token_id
    winning_outcome = "Up"
    if conflicting:
        winning_asset_id = btc.down_token_id
        winning_outcome = "Down"
    return _message(
        "clob_market",
        {
            "event_type": "market_resolved",
            "id": btc.market_id,
            "question": btc.question,
            "market": btc.condition_id,
            "slug": btc.slug,
            "assets_ids": list(btc.token_ids),
            "outcomes": ["Up", "Down"],
            "winning_asset_id": winning_asset_id,
            "winning_outcome": winning_outcome,
            "timestamp": str(btc.end_ms + 2_000),
        },
        sequence=90,
        wall_offset_ms=301_100,
        monotonic_ns=301_100_000_000,
    )


def test_replay_treats_consistent_repeated_resolution_as_idempotent(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "repeated-resolution.duckdb") as store:
        _finish_replay_store(
            store,
            "repeated-resolution",
            additional_messages=(_repeated_resolution_message(conflicting=False),),
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="repeated-resolution")

    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    assert len(replay.resolutions) == 3
    resolution = next(
        item for item in replay.resolutions if item.condition_id == btc.condition_id
    )
    assert resolution.winning_outcome == "Up"
    assert resolution.source == "clob_gamma_crosscheck"


def test_replay_rejects_conflicting_repeated_resolution(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "conflicting-resolution.duckdb") as store:
        _finish_replay_store(
            store,
            "conflicting-resolution",
            additional_messages=(_repeated_resolution_message(conflicting=True),),
        )
        with pytest.raises(ValueError, match="conflicting resolution events"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="conflicting-resolution",
            )


def test_websocket_resolution_is_validated_but_cannot_authorize_settlement(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "observed-resolution.duckdb") as store:
        _finish_replay_store(
            store,
            "observed-resolution",
            compact_resolution_event=True,
            finalize_official=False,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="observed-resolution")

    assert replay.resolutions == ()


def test_replay_applies_interleaved_terminal_book_clear_as_one_atomic_batch(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "terminal-clear.duckdb") as store:
        _finish_replay_store(
            store,
            "terminal-clear",
            terminal_clear_burst=True,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="terminal-clear")

    terminal = replay.books[-1]
    assert terminal.snapshot.bids == ()
    assert terminal.snapshot.asks == ()
    assert terminal.snapshot.source_time_ms == EPOCH * 1_000 + 2_000
    assert replay.diagnostics.late_event_count >= 1
    assert replay.diagnostics.maximum_source_regression_ms == 1
    assert replay.diagnostics.deferred_event_count >= 1
    assert replay.diagnostics.maximum_availability_delay_ns >= 2_000_000


def test_replay_matches_interleaved_best_prices_to_ordered_checksum_transitions(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "ordered-hashes.duckdb") as store:
        _finish_replay_store(
            store,
            "ordered-hashes",
            interleaved_best_transitions=True,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="ordered-hashes")

    final = replay.books[-1]
    assert final.snapshot.bids[0].price == Decimal("0.49")
    assert final.snapshot.asks[0].price == Decimal("0.51")
    assert final.snapshot.source_time_ms == EPOCH * 1_000 + 2_004


def _idempotent_checksum_correction_messages(
    *,
    corroborate_stale_top: bool,
    include_corrected_best: bool = True,
    mutation_price: str = "0.499",
    independent_prefix: bool = False,
    independent_prefix_monotonic_ns: int | None = None,
    stale_best_source_delta_ms: int = 0,
    stale_best_monotonic_ns: int = 2_000_000_000,
    pending_monotonic_ns: int = 2_000_000_000,
    replacement_source_delta_ms: int = 0,
    replacement_hash: str = "idempotent-correction",
    replacement_monotonic_ns: int = 2_000_000_000,
) -> tuple[RawStreamMessage, ...]:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    source_time = EPOCH * 1_000 + 2_000
    replacement_source_time = source_time + replacement_source_delta_ms
    common = {
        "market": btc.condition_id,
        "asset_id": token,
        "timestamp": str(source_time),
    }
    messages: list[RawStreamMessage] = []
    prefix_monotonic_ns = (
        pending_monotonic_ns
        if independent_prefix_monotonic_ns is None
        else independent_prefix_monotonic_ns
    )
    if corroborate_stale_top:
        messages.append(
            _message(
                "clob_market",
                {
                    **common,
                    "timestamp": str(source_time + stale_best_source_delta_ms),
                    "event_type": "best_bid_ask",
                    "best_bid": "0.499",
                    "best_ask": "0.50",
                    "spread": "0.001",
                },
                sequence=80,
                wall_offset_ms=2_000,
                monotonic_ns=stale_best_monotonic_ns,
            )
        )
    if independent_prefix:
        messages.append(
            _message(
                "clob_market",
                {
                    "market": btc.condition_id,
                    "timestamp": str(source_time),
                    "event_type": "price_change",
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.40",
                            "size": "4",
                            "side": "BUY",
                            "hash": "independent-prefix",
                            "best_bid": "0.498",
                            "best_ask": "0.50",
                        }
                    ],
                },
                sequence=81,
                wall_offset_ms=2_000,
                monotonic_ns=prefix_monotonic_ns,
            )
        )
    messages.sort(
        key=lambda message: (
            message.received_monotonic_ns,
            message.received_wall_ms,
            message.sequence_number,
        )
    )
    pending_sequence = 82 if independent_prefix else 81
    messages.extend(
        [
            _message(
                "clob_market",
                {
                    "market": btc.condition_id,
                    "timestamp": str(source_time),
                    "event_type": "price_change",
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": mutation_price,
                            "size": "0",
                            "side": "BUY",
                            "hash": "idempotent-correction",
                            "best_bid": "0.499",
                            "best_ask": "0.50",
                        }
                    ],
                },
                sequence=pending_sequence,
                wall_offset_ms=2_000,
                monotonic_ns=pending_monotonic_ns,
            ),
            _message(
                "clob_market",
                {
                    "market": btc.condition_id,
                    "timestamp": str(replacement_source_time),
                    "event_type": "price_change",
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": mutation_price,
                            "size": "0",
                            "side": "BUY",
                            "hash": replacement_hash,
                            "best_bid": "0.498",
                            "best_ask": "0.50",
                        }
                    ],
                },
                sequence=pending_sequence + 2,
                wall_offset_ms=2_001,
                monotonic_ns=replacement_monotonic_ns,
            ),
        ]
    )
    if include_corrected_best:
        messages.insert(
            -1,
            _message(
                "clob_market",
                {
                    **common,
                    "timestamp": str(replacement_source_time),
                    "event_type": "best_bid_ask",
                    "best_bid": "0.498",
                    "best_ask": "0.50",
                    "spread": "0.002",
                },
                sequence=pending_sequence + 1,
                wall_offset_ms=2_001,
                monotonic_ns=replacement_monotonic_ns,
            ),
        )
    return tuple(messages)


def _corrected_best_before_depth_messages(
    *,
    corrected_monotonic_ns: int = 2_000_000_000,
    depth_monotonic_ns: int = 2_000_000_000,
    corrected_source_delta_ms: int = 0,
    depth_source_delta_ms: int = 0,
    corrected_wall_offset_ms: int = 2_000,
    depth_wall_offset_ms: int = 2_000,
) -> tuple[RawStreamMessage, ...]:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    source_time = EPOCH * 1_000 + 2_000
    common = {
        "event_type": "best_bid_ask",
        "market": btc.condition_id,
        "asset_id": token,
        "timestamp": str(source_time),
    }
    return (
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(source_time - 1),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.50",
                        "size": "0",
                        "side": "SELL",
                        "hash": "pre-corrected-best-state",
                        "best_bid": "0.498",
                        "best_ask": "0.51",
                    }
                ],
            },
            sequence=89,
            wall_offset_ms=1_999,
            monotonic_ns=1_999_000_000,
        ),
        _message(
            "clob_market",
            {
                **common,
                "best_bid": "0.498",
                "best_ask": "0.50",
                "spread": "0.002",
            },
            sequence=90,
            wall_offset_ms=2_000,
            monotonic_ns=2_000_000_000,
        ),
        _message(
            "clob_market",
            {
                **common,
                "timestamp": str(source_time + corrected_source_delta_ms),
                "best_bid": "0.498",
                "best_ask": "0.51",
                "spread": "0.012",
            },
            sequence=91,
            wall_offset_ms=corrected_wall_offset_ms,
            monotonic_ns=corrected_monotonic_ns,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(source_time + depth_source_delta_ms),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.50",
                        "size": "0",
                        "side": "SELL",
                        "hash": "corrected-best-before-depth",
                        "best_bid": "0.498",
                        "best_ask": "0.51",
                    }
                ],
            },
            sequence=92,
            wall_offset_ms=depth_wall_offset_ms,
            monotonic_ns=depth_monotonic_ns,
        ),
    )


def _next_group_crossing_messages(
    *,
    full_book_best_ask: str = "0.51",
) -> tuple[RawStreamMessage, ...]:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    source_time = EPOCH * 1_000 + 2_000
    proof_source_time = source_time + 2
    return (
        _message(
            "clob_market",
            {
                "event_type": "best_bid_ask",
                "market": btc.condition_id,
                "asset_id": token,
                "best_bid": "0.50",
                "best_ask": "0.51",
                "spread": "0.01",
                "timestamp": str(source_time + 1),
            },
            sequence=100,
            wall_offset_ms=2_000,
            monotonic_ns=2_000_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(source_time),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.50",
                        "size": "5",
                        "side": "BUY",
                        "hash": "crossing-transition",
                        "best_bid": "0.50",
                        "best_ask": "0.51",
                    }
                ],
            },
            sequence=101,
            wall_offset_ms=2_000,
            monotonic_ns=2_000_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(proof_source_time),
                "price_changes": [
                    {
                        "asset_id": token,
                        "price": "0.50",
                        "size": "6",
                        "side": "BUY",
                        "hash": "next-group-proof",
                        "best_bid": "0.50",
                        "best_ask": "0.51",
                    }
                ],
            },
            sequence=102,
            wall_offset_ms=2_015,
            monotonic_ns=2_015_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "book",
                "market": btc.condition_id,
                "asset_id": token,
                "timestamp": str(proof_source_time),
                "tick_size": "0.001",
                "hash": "next-group-proof",
                "bids": [
                    {"price": "0.50", "size": "6"},
                    {"price": "0.498", "size": "5"},
                ],
                "asks": [{"price": full_book_best_ask, "size": "10"}],
            },
            sequence=103,
            wall_offset_ms=2_015,
            monotonic_ns=2_015_000_000,
        ),
    )


def _next_group_multi_fragment_crossing_messages(
    *,
    include_hash_linked_non_top: bool = False,
    include_unrelated_fragment: bool = False,
) -> tuple[RawStreamMessage, ...]:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    source_time = EPOCH * 1_000 + 2_000
    changes = [
        ("BUY", "0.50", "5"),
        ("SELL", "0.48", "0"),
        ("SELL", "0.49", "0"),
    ]
    if include_unrelated_fragment:
        changes.append(("BUY", "0.40", "3"))
    messages: list[RawStreamMessage] = [
        _message(
            "clob_market",
            {
                "event_type": "best_bid_ask",
                "market": btc.condition_id,
                "asset_id": token,
                "best_bid": "0.50",
                "best_ask": "0.51",
                "spread": "0.01",
                "timestamp": str(source_time + 1),
            },
            sequence=110,
            wall_offset_ms=2_000,
            monotonic_ns=2_000_000_000,
        )
    ]
    for offset, (side, price, size) in enumerate(changes, start=1):
        messages.append(
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(source_time),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": price,
                            "size": size,
                            "side": side,
                            "hash": "multi-fragment-crossing",
                            "best_bid": "0.50",
                            "best_ask": "0.51",
                        }
                    ],
                },
                sequence=110 + offset,
                wall_offset_ms=2_000,
                monotonic_ns=2_000_000_000,
            )
        )
    if include_hash_linked_non_top:
        messages.append(
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(source_time + 7),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.40",
                            "size": "4",
                            "side": "BUY",
                            "hash": "direct-next-group-proof",
                            "best_bid": "0.50",
                            "best_ask": "0.51",
                        }
                    ],
                },
                sequence=119,
                wall_offset_ms=2_015,
                monotonic_ns=2_015_000_000,
            )
        )
    proof_bids = [
        {"price": "0.50", "size": "5"},
        {"price": "0.498", "size": "5"},
    ]
    if include_hash_linked_non_top:
        proof_bids.append({"price": "0.40", "size": "4"})
    messages.append(
        _message(
            "clob_market",
            {
                "event_type": "book",
                "market": btc.condition_id,
                "asset_id": token,
                "timestamp": str(source_time + 7),
                "tick_size": "0.001",
                "hash": "direct-next-group-proof",
                "bids": proof_bids,
                "asks": [{"price": "0.51", "size": "10"}],
            },
            sequence=120,
            wall_offset_ms=2_015,
            monotonic_ns=2_015_000_000,
        )
    )
    return tuple(messages)


def _prefix_stale_full_book_messages(
    *,
    reported_best_ask: str = "0.51",
    full_book_best_ask: str = "0.52",
    full_book_monotonic_ns: int = 2_000_000_000,
) -> tuple[RawStreamMessage, ...]:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    source_time = EPOCH * 1_000 + 2_000
    common_change = {
        "asset_id": token,
        "side": "SELL",
        "hash": "prefix-stale-depth",
        "best_bid": "0.498",
        "best_ask": reported_best_ask,
    }
    messages = [
        _message(
            "clob_market",
            {
                "event_type": "best_bid_ask",
                "market": btc.condition_id,
                "asset_id": token,
                "timestamp": str(source_time),
                "best_bid": "0.498",
                "best_ask": reported_best_ask,
                "spread": "0.012",
            },
            sequence=90,
            wall_offset_ms=2_000,
            monotonic_ns=2_000_000_000,
        )
    ]
    for sequence, price, size in (
        (91, "0.52", "7"),
        (92, "0.50", "0"),
        (93, "0.51", "0"),
    ):
        messages.append(
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(source_time),
                    "price_changes": [{**common_change, "price": price, "size": size}],
                },
                sequence=sequence,
                wall_offset_ms=2_000,
                monotonic_ns=2_000_000_000,
            )
        )
    messages.extend(
        [
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(source_time + 1),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.40",
                            "size": "3",
                            "side": "BUY",
                            "hash": "corroborating-full-book",
                            "best_bid": "0.498",
                            "best_ask": reported_best_ask,
                        }
                    ],
                },
                sequence=94,
                wall_offset_ms=2_001,
                monotonic_ns=2_000_000_000,
            ),
            _message(
                "clob_market",
                {
                    "event_type": "book",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "timestamp": str(source_time + 1),
                    "tick_size": "0.001",
                    "hash": "corroborating-full-book",
                    "bids": [
                        {"price": "0.498", "size": "5"},
                        {"price": "0.40", "size": "3"},
                    ],
                    "asks": [{"price": full_book_best_ask, "size": "7"}],
                },
                sequence=95,
                wall_offset_ms=2_001,
                monotonic_ns=full_book_monotonic_ns,
            ),
        ]
    )
    return tuple(messages)


def _recent_full_book_lagging_bbo_message(
    *,
    source_age_ms: int,
) -> RawStreamMessage:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    full_book_source_time = EPOCH * 1_000 + 2_001
    return _message(
        "clob_market",
        {
            "event_type": "price_change",
            "market": btc.condition_id,
            "timestamp": str(full_book_source_time + source_age_ms),
            "price_changes": [
                {
                    "asset_id": btc.up_token_id,
                    "price": "0.41",
                    "size": "2",
                    "side": "BUY",
                    "hash": "post-book-lagging-bbo",
                    "best_bid": "0.498",
                    "best_ask": "0.519",
                }
            ],
        },
        sequence=96,
        wall_offset_ms=2_001 + source_age_ms,
        monotonic_ns=(2_001 + source_age_ms) * 1_000_000,
    )


def test_replay_accepts_hash_bound_idempotent_checksum_correction(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "corrected-checksum.duckdb") as store:
        _finish_replay_store(
            store,
            "corrected-checksum",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=True
            ),
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="corrected-checksum")

    corrected = replay.books[-1].snapshot
    assert corrected.source_time_ms == EPOCH * 1_000 + 2_000
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.50")


def test_replay_accepts_same_group_ephemeral_top_removal_correction(
    tmp_path,
) -> None:
    messages = _idempotent_checksum_correction_messages(
        corroborate_stale_top=True,
        include_corrected_best=False,
    )
    with PolymarketEvidenceStore(tmp_path / "ephemeral-top-removal.duckdb") as store:
        _finish_replay_store(
            store,
            "ephemeral-top-removal",
            additional_messages=messages,
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="ephemeral-top-removal",
        )

    corrected = replay.books[-1].snapshot
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.50")


def test_replay_rejects_ephemeral_removal_that_does_not_target_stale_top(
    tmp_path,
) -> None:
    messages = _idempotent_checksum_correction_messages(
        corroborate_stale_top=True,
        include_corrected_best=False,
        mutation_price="0.497",
    )
    with PolymarketEvidenceStore(
        tmp_path / "unbound-ephemeral-removal.duckdb"
    ) as store:
        _finish_replay_store(
            store,
            "unbound-ephemeral-removal",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="unbound-ephemeral-removal",
            )


def test_replay_accepts_corrected_best_pair_before_atomic_depth(tmp_path) -> None:
    with PolymarketEvidenceStore(
        tmp_path / "corrected-best-before-depth.duckdb"
    ) as store:
        _finish_replay_store(
            store,
            "corrected-best-before-depth",
            additional_messages=_corrected_best_before_depth_messages(),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="corrected-best-before-depth",
        )

    corrected = replay.books[-1].snapshot
    assert corrected.source_time_ms == EPOCH * 1_000 + 2_000
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.51")


def test_replay_accepts_bounded_prior_group_stale_best_with_same_group_proof(
    tmp_path,
) -> None:
    messages = _corrected_best_before_depth_messages(
        corrected_monotonic_ns=2_050_000_000,
        depth_monotonic_ns=2_050_000_000,
        corrected_source_delta_ms=5,
        depth_source_delta_ms=5,
        corrected_wall_offset_ms=2_050,
        depth_wall_offset_ms=2_050,
    )
    with PolymarketEvidenceStore(tmp_path / "bounded-prior-best.duckdb") as store:
        _finish_replay_store(
            store,
            "bounded-prior-best",
            additional_messages=messages,
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="bounded-prior-best",
        )

    corrected = replay.books[-1].snapshot
    assert corrected.source_time_ms == EPOCH * 1_000 + 2_005
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.51")


def test_replay_rejects_expired_prior_group_stale_best(tmp_path) -> None:
    messages = _corrected_best_before_depth_messages(
        corrected_monotonic_ns=2_251_000_001,
        depth_monotonic_ns=2_251_000_001,
        corrected_source_delta_ms=5,
        depth_source_delta_ms=5,
        corrected_wall_offset_ms=2_251,
        depth_wall_offset_ms=2_251,
    )
    with PolymarketEvidenceStore(tmp_path / "expired-prior-best.duckdb") as store:
        _finish_replay_store(
            store,
            "expired-prior-best",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="best_bid_ask"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="expired-prior-best",
            )


def test_replay_repairs_crossing_only_with_next_group_full_book_proof(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "next-group-crossing.duckdb") as store:
        _finish_replay_store(
            store,
            "next-group-crossing",
            additional_messages=_next_group_crossing_messages(),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="next-group-crossing",
        )

    repaired = next(
        book.snapshot
        for book in replay.books
        if book.snapshot.source_time_ms == EPOCH * 1_000 + 2_000
    )
    assert repaired.bids[0].price == Decimal("0.50")
    assert repaired.asks[0].price == Decimal("0.51")
    assert repaired.received_monotonic_ns >= 2_015_000_000


def test_replay_rejects_crossing_when_next_group_full_book_conflicts(tmp_path) -> None:
    messages = _next_group_crossing_messages(full_book_best_ask="0.52")
    with PolymarketEvidenceStore(tmp_path / "conflicting-next-group.duckdb") as store:
        _finish_replay_store(
            store,
            "conflicting-next-group",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="conflicting-next-group",
            )


def test_replay_repairs_atomic_multi_fragment_crossing_with_direct_full_book(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "multi-crossing.duckdb") as store:
        _finish_replay_store(
            store,
            "multi-crossing",
            additional_messages=_next_group_multi_fragment_crossing_messages(),
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="multi-crossing")

    repaired = next(
        book.snapshot
        for book in replay.books
        if book.snapshot.source_time_ms == EPOCH * 1_000 + 2_000
    )
    assert repaired.bids[0].price == Decimal("0.50")
    assert repaired.asks[0].price == Decimal("0.51")
    assert repaired.received_monotonic_ns >= 2_015_000_000


def test_replay_accepts_hash_linked_non_top_change_before_crossing_proof(
    tmp_path,
) -> None:
    messages = _next_group_multi_fragment_crossing_messages(
        include_hash_linked_non_top=True
    )
    with PolymarketEvidenceStore(tmp_path / "non-top-crossing.duckdb") as store:
        _finish_replay_store(
            store,
            "non-top-crossing",
            additional_messages=messages,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="non-top-crossing")

    assert replay.books[-1].snapshot.bids[0].price == Decimal("0.50")
    assert any(
        level.price == Decimal("0.40") and level.quantity == Decimal("4")
        for level in replay.books[-1].snapshot.bids
    )


def test_replay_rejects_unrelated_fragment_in_crossing_repair(tmp_path) -> None:
    messages = _next_group_multi_fragment_crossing_messages(
        include_unrelated_fragment=True
    )
    with PolymarketEvidenceStore(tmp_path / "unrelated-crossing.duckdb") as store:
        _finish_replay_store(
            store,
            "unrelated-crossing",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="unrelated-crossing",
            )


def test_replay_rejects_corrected_best_pair_from_another_receive_group(
    tmp_path,
) -> None:
    messages = _corrected_best_before_depth_messages(depth_monotonic_ns=2_001_000_000)
    with PolymarketEvidenceStore(
        tmp_path / "cross-group-best-before-depth.duckdb"
    ) as store:
        _finish_replay_store(
            store,
            "cross-group-best-before-depth",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="best_bid_ask"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="cross-group-best-before-depth",
            )


def test_replay_rejects_uncorroborated_idempotent_checksum_correction(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "uncorroborated-checksum.duckdb") as store:
        _finish_replay_store(
            store,
            "uncorroborated-checksum",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=False
            ),
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="uncorroborated-checksum",
            )


def test_replay_accepts_same_timestamp_correction_across_bounded_receive_groups(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "bounded-same-time.duckdb") as store:
        _finish_replay_store(
            store,
            "bounded-same-time",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=True,
                pending_monotonic_ns=2_016_000_000,
                replacement_monotonic_ns=2_016_000_000,
            ),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="bounded-same-time",
        )

    corrected = replay.books[-1].snapshot
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.50")


def test_replay_rejects_same_timestamp_correction_after_receive_group_bound(
    tmp_path,
) -> None:
    messages = _idempotent_checksum_correction_messages(
        corroborate_stale_top=True,
        pending_monotonic_ns=2_251_000_001,
        replacement_monotonic_ns=2_251_000_001,
    )
    with PolymarketEvidenceStore(tmp_path / "expired-same-time.duckdb") as store:
        _finish_replay_store(
            store,
            "expired-same-time",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="expired-same-time",
            )


def test_replay_accepts_receive_group_cross_timestamp_checksum_correction(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "cross-time-correction.duckdb") as store:
        _finish_replay_store(
            store,
            "cross-time-correction",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=True,
                replacement_source_delta_ms=1,
                replacement_hash="corrected-replacement-hash",
            ),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="cross-time-correction",
        )

    corrected = replay.books[-1].snapshot
    assert corrected.source_time_ms == EPOCH * 1_000 + 2_001
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.50")


def test_replay_preserves_independent_prefix_before_cross_timestamp_correction(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "prefixed-cross-time.duckdb") as store:
        _finish_replay_store(
            store,
            "prefixed-cross-time",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=True,
                independent_prefix=True,
                stale_best_source_delta_ms=1,
                replacement_source_delta_ms=1,
                replacement_hash="corrected-replacement-hash",
            ),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="prefixed-cross-time",
        )

    corrected = replay.books[-1].snapshot
    assert corrected.bids[0].price == Decimal("0.498")
    assert any(
        level.price == Decimal("0.40") and level.quantity == Decimal("4")
        for level in corrected.bids
    )


def test_replay_preserves_bounded_earlier_receive_group_prefix_before_correction(
    tmp_path,
) -> None:
    messages = _idempotent_checksum_correction_messages(
        corroborate_stale_top=True,
        independent_prefix=True,
        independent_prefix_monotonic_ns=2_000_000_000,
        stale_best_monotonic_ns=2_016_000_000,
        pending_monotonic_ns=2_016_000_000,
        replacement_source_delta_ms=1,
        replacement_hash="corrected-replacement-hash",
        replacement_monotonic_ns=2_016_000_000,
    )
    with PolymarketEvidenceStore(tmp_path / "earlier-prefix.duckdb") as store:
        _finish_replay_store(
            store,
            "earlier-prefix",
            additional_messages=messages,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="earlier-prefix")

    corrected = replay.books[-1].snapshot
    assert corrected.bids[0].price == Decimal("0.498")
    assert any(
        level.price == Decimal("0.40") and level.quantity == Decimal("4")
        for level in corrected.bids
    )


def test_replay_accepts_bounded_cross_receive_group_checksum_correction(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "cross-group-correction.duckdb") as store:
        _finish_replay_store(
            store,
            "cross-group-correction",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=True,
                replacement_source_delta_ms=1,
                replacement_hash="corrected-replacement-hash",
                replacement_monotonic_ns=2_001_000_000,
            ),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="cross-group-correction",
        )

    assert replay.books[-1].snapshot.bids[0].price == Decimal("0.498")


def test_replay_rejects_cross_receive_group_correction_after_time_bound(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "expired-group-correction.duckdb") as store:
        _finish_replay_store(
            store,
            "expired-group-correction",
            additional_messages=_idempotent_checksum_correction_messages(
                corroborate_stale_top=True,
                replacement_source_delta_ms=1,
                replacement_hash="corrected-replacement-hash",
                replacement_monotonic_ns=2_251_000_001,
            ),
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="expired-group-correction",
            )


def test_replay_accepts_prefix_stale_checksum_proven_by_same_group_full_book(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "prefix-stale-full-book.duckdb") as store:
        _finish_replay_store(
            store,
            "prefix-stale-full-book",
            additional_messages=_prefix_stale_full_book_messages(),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="prefix-stale-full-book",
        )

    corrected = next(
        book.snapshot
        for book in replay.books
        if book.snapshot.source_time_ms == EPOCH * 1_000 + 2_000
    )
    assert corrected.bids[0].price == Decimal("0.498")
    assert corrected.asks[0].price == Decimal("0.52")


def test_replay_rejects_prefix_stale_checksum_without_same_group_full_book(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "prefix-stale-other-group.duckdb") as store:
        _finish_replay_store(
            store,
            "prefix-stale-other-group",
            additional_messages=_prefix_stale_full_book_messages(
                full_book_monotonic_ns=2_001_000_000
            ),
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="prefix-stale-other-group",
            )


def test_replay_rejects_prefix_stale_checksum_when_full_book_depth_conflicts(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "prefix-stale-conflict.duckdb") as store:
        _finish_replay_store(
            store,
            "prefix-stale-conflict",
            additional_messages=_prefix_stale_full_book_messages(
                full_book_best_ask="0.53"
            ),
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="prefix-stale-conflict",
            )


def test_replay_accepts_one_tick_lag_after_recent_authoritative_full_book(
    tmp_path,
) -> None:
    messages = (
        *_prefix_stale_full_book_messages(),
        _recent_full_book_lagging_bbo_message(source_age_ms=16),
    )
    with PolymarketEvidenceStore(tmp_path / "recent-full-book-lag.duckdb") as store:
        _finish_replay_store(
            store,
            "recent-full-book-lag",
            additional_messages=messages,
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="recent-full-book-lag",
        )

    assert replay.books[-1].snapshot.bids[0].price == Decimal("0.498")
    assert replay.books[-1].snapshot.asks[0].price == Decimal("0.52")


def test_replay_rejects_lagging_bbo_after_full_book_age_bound(tmp_path) -> None:
    messages = (
        *_prefix_stale_full_book_messages(),
        _recent_full_book_lagging_bbo_message(source_age_ms=251),
    )
    with PolymarketEvidenceStore(tmp_path / "expired-full-book-lag.duckdb") as store:
        _finish_replay_store(
            store,
            "expired-full-book-lag",
            additional_messages=messages,
        )
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="expired-full-book-lag",
            )


def test_replay_full_book_resynchronizes_trade_depth_absent_from_deltas(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "trade-resync.duckdb") as store:
        _finish_replay_store(store, "trade-resync", trade_resync=True)
        replay = PolymarketEvidenceReplay.load(store, run_id="trade-resync")

    resync_index = next(
        index
        for index, book in enumerate(replay.books)
        if book.event_type == "book"
        and [level.price for level in book.snapshot.asks]
        == [Decimal("0.50"), Decimal("0.52"), Decimal("0.53")]
    )
    resynchronized = replay.books[resync_index].snapshot
    post_resync = replay.books[resync_index + 1].snapshot
    assert resynchronized.bids == ()
    assert resynchronized.asks[0].price == Decimal("0.50")
    assert resynchronized.asks[0].quantity == Decimal("8")
    assert resynchronized.asks[1].price == Decimal("0.52")
    assert resynchronized.asks[1].quantity == Decimal("7")
    assert resynchronized.asks[2].price == Decimal("0.53")
    assert resynchronized.asks[2].quantity == Decimal("6")
    assert post_resync.asks[1].price == Decimal("0.52")
    assert post_resync.bids[0].price == Decimal("0.499")
    assert post_resync.received_monotonic_ns > resynchronized.received_monotonic_ns
    assert replay.diagnostics.late_event_count >= 1
    assert replay.diagnostics.maximum_source_regression_ms == 1
    assert replay.diagnostics.deferred_event_count == 0
    assert replay.diagnostics.maximum_availability_delay_ns == 0


def test_replay_fast_forwards_stale_full_book_with_newer_top_delta(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "stale-top-resync.duckdb") as store:
        _finish_replay_store(
            store,
            "stale-top-resync",
            trade_resync=True,
            trade_resync_top_change=True,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="stale-top-resync")

    rebased = next(
        book
        for book in replay.books
        if book.event_type == "book"
        and book.snapshot.source_time_ms == EPOCH * 1_000 + 1_011
    )
    assert rebased.snapshot.bids[0].price == Decimal("0.49")
    assert [level.price for level in rebased.snapshot.asks] == [
        Decimal("0.50"),
        Decimal("0.52"),
    ]
    followup = next(
        book
        for book in replay.books
        if book.snapshot.received_monotonic_ns > rebased.snapshot.received_monotonic_ns
        and any(level.price == Decimal("0.99") for level in book.snapshot.asks)
    )
    assert followup.snapshot.bids[0].price == Decimal("0.49")
    assert followup.snapshot.asks[0].price == Decimal("0.50")
    assert replay.diagnostics.late_event_count >= 1


def test_replay_binds_exact_duplicate_tick_transition_idempotently(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "duplicate-tick.duckdb") as store:
        _finish_replay_store(
            store,
            "duplicate-tick",
            duplicate_tick_transition=True,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="duplicate-tick")

    assert replay.books[2].tick_size == Decimal("0.001")


def test_replay_accepts_one_ms_duplicate_tick_transition_in_same_receive_group(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "duplicate-tick-one-ms.duckdb") as store:
        _finish_replay_store(
            store,
            "duplicate-tick-one-ms",
            duplicate_tick_transition=True,
            duplicate_tick_source_delta_ms=1,
            duplicate_tick_monotonic_ns=1_012_000_000,
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="duplicate-tick-one-ms",
        )

    assert replay.books[2].tick_size == Decimal("0.001")


def test_replay_rejects_cross_timestamp_tick_duplicate_from_another_group(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(
        tmp_path / "duplicate-tick-other-group.duckdb"
    ) as store:
        _finish_replay_store(
            store,
            "duplicate-tick-other-group",
            duplicate_tick_transition=True,
            duplicate_tick_source_delta_ms=1,
            duplicate_tick_monotonic_ns=1_013_000_000,
        )
        with pytest.raises(ValueError, match="old value disagrees"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="duplicate-tick-other-group",
            )


def test_replay_rejects_events_outside_bounded_causal_reorder_window(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "stale-book.duckdb") as store:
        _finish_replay_store(
            store,
            "stale-book",
            trade_resync=True,
            trade_resync_lag_ms=1_002,
        )
        with pytest.raises(ValueError, match="bounded causal reorder window"):
            PolymarketEvidenceReplay.load(store, run_id="stale-book")


def test_polymarket_feature_dataset_is_causal_hashed_and_officially_labeled(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "features.duckdb") as store:
        _finish_replay_store(store, "features", feature_evidence=True)
        config = PolymarketFeatureConfig(
            cadence_ms=50,
            warmup_ms=0,
            minimum_resolved_markets_per_asset=1,
        )
        first = build_polymarket_feature_dataset(
            store,
            run_id="features",
            config=config,
        )
        source_context = load_polymarket_feature_source_context(
            store,
            run_id="features",
            config=config,
        )
        conditions = tuple(sorted({row.condition_id for row in first.rows}))
        full_replay = PolymarketEvidenceReplay.load(
            store,
            run_id="features",
            book_sample_interval_ms=0,
            condition_ids=conditions,
        )
        sampled_replay = full_replay.with_book_sample_interval(config.cadence_ms)
        direct_sampled_replay = PolymarketEvidenceReplay.load(
            store,
            run_id="features",
            book_sample_interval_ms=config.cadence_ms,
            condition_ids=conditions,
        )
        second = build_polymarket_feature_dataset(
            store,
            run_id="features",
            config=config,
            condition_ids=conditions,
            source_context=source_context,
            preloaded_replay=sampled_replay,
        )
        created = materialize_polymarket_feature_dataset(store, first)
        existing = materialize_polymarket_feature_dataset(store, second)
        store.connect().execute(
            """
            UPDATE polymarket_feature_row SET feature_values_json = '[]'
            WHERE dataset_id = ? AND feature_id = ?
            """,
            [first.dataset_id, first.rows[0].feature_id],
        )
        with pytest.raises(ValueError, match="feature rows are inconsistent"):
            materialize_polymarket_feature_dataset(store, first)

    assert first.dataset_id == second.dataset_id
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.rows == second.rows
    assert sampled_replay.books == direct_sampled_replay.books
    assert sampled_replay.diagnostics == direct_sampled_replay.diagnostics
    assert created.status == "created"
    assert existing.status == "existing"
    assert created.row_count == existing.row_count == len(first.rows)
    assert len(first.rows) >= 1
    row = first.rows[0]
    assert polymarket_feature_row_sha256(row) == (
        "05646644a59b938942785ebc3bdfabe61335edbff1674557ac0a615668cc55bd"
    )
    assert len(row.feature_values) == len(POLYMARKET_FEATURE_NAMES) == 49
    assert row.official_up is True
    assert row.resolution_event_id
    assert row.feature_map()["ask_pair_cost"] == pytest.approx(1.02)
    assert row.feature_map()["chainlink_anchor_gap_ms"] == 0.0
    assert math.isfinite(row.feature_map()["binance_return_100ms_bps"])
    assert math.isfinite(row.feature_map()["binance_realized_volatility_100ms_bps"])
    assert math.isfinite(row.feature_map()["binance_trade_imbalance_100ms"])
    assert first.labeled_market_counts["BTC"] == 1
    assert first.training_ready is False
    assert "insufficient_featured_resolved_markets:ETH:0/1" in first.training_errors


def test_capture_chunk_receipt_index_is_exact_reusable_and_tamper_evident(
    tmp_path,
) -> None:
    database = tmp_path / "receipt-index.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "receipt-index", feature_evidence=True)
        report = store.ensure_capture_chunk_receipt_index("receipt-index")
        all_events = tuple(
            store.iter_public_events(
                "receipt-index",
                streams=("binance_spot", "polymarket_rtds"),
                verified_source=True,
            )
        )
        received = sorted({event.received_wall_ms for event in all_events})
        lower = received[len(received) // 4]
        upper = received[(len(received) * 3) // 4]
        expected = tuple(
            event.event_id
            for event in all_events
            if lower <= event.received_wall_ms <= upper
        )
        bounded = tuple(
            event.event_id
            for event in store.iter_public_events(
                "receipt-index",
                streams=("binance_spot", "polymarket_rtds"),
                verified_source=True,
                minimum_received_wall_ms=lower,
                maximum_received_wall_ms=upper,
            )
        )
        assert bounded == expected
        assert store.ensure_capture_chunk_receipt_index("receipt-index") == report
        assert report["chunk_count"] >= 1
        assert report["message_count"] >= len(all_events)

    with PolymarketEvidenceStore(database, read_only=True) as store:
        assert store.ensure_capture_chunk_receipt_index("receipt-index") == report

    with PolymarketEvidenceStore(database) as store:
        store.connect().execute(
            """
            UPDATE polymarket_chunk_receipt_index
            SET first_received_wall_ms = first_received_wall_ms + 1
            WHERE run_id = ? AND chunk_index = 0
            """,
            ["receipt-index"],
        )

    with PolymarketEvidenceStore(database, read_only=True) as store:
        with pytest.raises(ValueError, match="chunk receipt index row differs"):
            store.ensure_capture_chunk_receipt_index("receipt-index")


def test_feature_source_ignores_irrelevant_and_non_target_rtds(tmp_path) -> None:
    unrelated = _message(
        "polymarket_rtds",
        {"topic": "comments", "type": "update", "payload": None},
        sequence=90,
        wall_offset_ms=300_000,
        monotonic_ns=300_000_000_000,
    )
    non_target_chainlink = _message(
        "polymarket_rtds",
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "payload": {
                "symbol": "doge/usd",
                "timestamp": EPOCH * 1_000 + 300_000,
                "value": "0.25",
            },
        },
        sequence=91,
        wall_offset_ms=300_001,
        monotonic_ns=300_001_000_000,
    )
    with PolymarketEvidenceStore(tmp_path / "rtds-selection.duckdb") as store:
        _finish_replay_store(
            store,
            "rtds-selection",
            feature_evidence=True,
            additional_messages=(unrelated, non_target_chainlink),
        )
        context = load_polymarket_feature_source_context(
            store,
            run_id="rtds-selection",
            config=PolymarketFeatureConfig(
                minimum_resolved_markets_per_asset=1,
            ),
        )

    assert set(context.chainlink) == {"BTC", "ETH", "SOL"}
    assert all(
        point.asset in context.chainlink
        for points in context.chainlink.values()
        for point in points
    )


def test_feature_source_reuses_hash_bound_continuity_coverage(
    tmp_path,
    monkeypatch,
) -> None:
    proof_sha256 = "c" * 64
    with PolymarketEvidenceStore(tmp_path / "source-continuity.duckdb") as store:
        _finish_replay_store(store, "source-continuity", feature_evidence=True)
        btc = parse_polymarket_five_minute_market(_market_payload("BTC"))

        class Proof:
            report_sha256 = proof_sha256
            eligible_condition_ids = (btc.condition_id,)

        monkeypatch.setattr(
            continuity_module,
            "evaluate_polymarket_continuity_eligibility",
            lambda *_args, **_kwargs: Proof(),
        )
        monkeypatch.setattr(
            feature_module,
            "inspect_polymarket_feed_coverage",
            lambda *_args, **_kwargs: pytest.fail(
                "hash-bound source coverage must not rescan every CLOB event"
            ),
        )
        context = load_polymarket_feature_source_context(
            store,
            run_id="source-continuity",
            config=PolymarketFeatureConfig(
                minimum_resolved_markets_per_asset=1,
                allow_segmented_gaps=True,
            ),
            condition_ids=(btc.condition_id,),
            continuity_report_sha256=proof_sha256,
        )

    assert context.coverage.counts["BTC"]["market_snapshots"] == 1
    assert context.coverage.counts["BTC"]["clob_token_baselines"] == 2


def test_feature_source_rejects_malformed_target_chainlink_evidence(tmp_path) -> None:
    malformed_target = _message(
        "polymarket_rtds",
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "payload": {
                "symbol": "btc/usd",
                "timestamp": EPOCH * 1_000 + 300_000,
                "value": "not-a-price",
            },
        },
        sequence=90,
        wall_offset_ms=300_000,
        monotonic_ns=300_000_000_000,
    )
    with PolymarketEvidenceStore(tmp_path / "rtds-target-invalid.duckdb") as store:
        _finish_replay_store(
            store,
            "rtds-target-invalid",
            feature_evidence=True,
            additional_messages=(malformed_target,),
        )
        with pytest.raises(ValueError, match="RTDS source price"):
            load_polymarket_feature_source_context(
                store,
                run_id="rtds-target-invalid",
                config=PolymarketFeatureConfig(
                    minimum_resolved_markets_per_asset=1,
                ),
            )


def test_polymarket_feature_materialization_accepts_a_truthful_empty_dataset(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "empty-features.duckdb") as store:
        _finish_replay_store(store, "empty-features", feature_evidence=True)
        dataset = build_polymarket_feature_dataset(
            store,
            run_id="empty-features",
            config=PolymarketFeatureConfig(
                cadence_ms=50,
                warmup_ms=60_000,
                minimum_resolved_markets_per_asset=1,
            ),
        )
        created = materialize_polymarket_feature_dataset(store, dataset)
        existing = materialize_polymarket_feature_dataset(store, dataset)

    assert dataset.rows == ()
    assert dataset.shadow_ready is False
    assert created.status == "created"
    assert existing.status == "existing"
    assert created.row_count == existing.row_count == 0


def test_polymarket_action_materialization_is_idempotent_and_tamper_evident(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "action-values.duckdb") as store:
        _finish_replay_store(store, "action-values", feature_evidence=True)
        features = build_polymarket_feature_dataset(
            store,
            run_id="action-values",
            config=PolymarketFeatureConfig(
                cadence_ms=250,
                warmup_ms=0,
                minimum_resolved_markets_per_asset=1,
            ),
        )
        materialize_polymarket_feature_dataset(store, features)
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="action-values",
            book_sample_interval_ms=0,
        )
        dataset = build_polymarket_action_value_dataset(
            features,
            PolymarketRepricingExecutionContext(replay),
        )

        created = materialize_polymarket_action_value_dataset(store, dataset)
        existing = materialize_polymarket_action_value_dataset(store, dataset)
        store.connect().execute(
            """
            UPDATE polymarket_action_value_row
            SET stress_utility_quote = '999'
            WHERE dataset_sha256 = ? AND action_index = 0
            """,
            [dataset.dataset_sha256],
        )
        with pytest.raises(ValueError, match="action rows are inconsistent"):
            materialize_polymarket_action_value_dataset(store, dataset)

    assert created.status == "created"
    assert existing.status == "existing"
    assert created.action_count == existing.action_count == len(dataset.features)
    assert dataset.features


def test_round12_admission_materialization_is_compact_idempotent_and_linked(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "round12-admission.duckdb") as store:
        _finish_replay_store(store, "round12-admission", feature_evidence=True)
        features = build_polymarket_feature_dataset(
            store,
            run_id="round12-admission",
            config=PolymarketFeatureConfig(
                cadence_ms=250,
                warmup_ms=0,
                minimum_resolved_markets_per_asset=1,
            ),
        )
        materialize_polymarket_feature_dataset(store, features)
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="round12-admission",
            book_sample_interval_ms=0,
        )
        actions, admissions = build_round12_action_evidence_datasets(
            features,
            PolymarketRepricingExecutionContext(replay),
        )
        materialize_polymarket_action_value_dataset(store, actions)

        created = materialize_round12_admission_dataset(store, admissions)
        existing = materialize_round12_admission_dataset(store, admissions)
        loaded = load_round12_admission_dataset(
            store,
            source_action_dataset_sha256=actions.dataset_sha256,
        )
        columns = {
            str(row[1])
            for row in store.connect()
            .execute("PRAGMA table_info('polymarket_round12_action_local_admission')")
            .fetchall()
        }

    assert created == "created"
    assert existing == "existing"
    assert loaded == admissions
    assert len(admissions.admissions) == len(actions.features)
    assert "entry_book_event_id" in columns
    assert "execution_evidence_json" not in columns


def test_polymarket_condition_cache_preserves_verified_replay_and_empty_markets(
    tmp_path,
) -> None:
    progress_events: list[str] = []
    with PolymarketEvidenceStore(tmp_path / "condition-cache.duckdb") as store:
        _finish_replay_store(store, "condition-cache", feature_evidence=True)
        assert store.integrity_errors("condition-cache") == ()
        btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
        eth = parse_polymarket_five_minute_market(_market_payload("ETH"))
        expected = tuple(
            store.iter_public_events(
                "condition-cache",
                streams=("clob_market", "clob_rest_book"),
                condition_ids=(btc.condition_id,),
                verified_source=True,
            )
        )

        created = store.ensure_condition_message_cache(
            "condition-cache",
            progress=lambda phase, _payload: progress_events.append(phase),
        )
        existing = store.ensure_condition_message_cache("condition-cache")
        empty_manifest = (
            store.connect()
            .execute(
                """
            SELECT frame_count, message_count, first_received_monotonic_ns,
                   last_received_monotonic_ns, last_frame_sha256
            FROM polymarket_condition_message_manifest
            WHERE run_id = ? AND condition_id = ?
            """,
                ["condition-cache", eth.condition_id],
            )
            .fetchone()
        )
        selected = store.ensure_condition_message_cache(
            "condition-cache",
            condition_ids=(btc.condition_id,),
            progress=lambda phase, _payload: progress_events.append(phase),
        )
        actual = tuple(
            store.iter_public_events(
                "condition-cache",
                streams=("clob_market", "clob_rest_book"),
                condition_ids=(btc.condition_id,),
                verified_source=True,
            )
        )
        empty = tuple(
            store.iter_public_events(
                "condition-cache",
                streams=("clob_market", "clob_rest_book"),
                condition_ids=(eth.condition_id,),
                verified_source=True,
            )
        )
        manifest_conditions = tuple(
            row[0]
            for row in store.connect()
            .execute(
                """
                SELECT condition_id FROM polymarket_condition_message_manifest
                WHERE run_id = ? ORDER BY condition_id
                """,
                ["condition-cache"],
            )
            .fetchall()
        )

    assert expected
    assert actual == expected
    assert empty == ()
    assert empty_manifest == (0, 0, 0, 0, "")
    assert existing == created
    assert selected == created
    assert manifest_conditions == tuple(
        sorted(
            (
                btc.condition_id,
                eth.condition_id,
                parse_polymarket_five_minute_market(
                    _market_payload("SOL")
                ).condition_id,
            )
        )
    )
    assert progress_events[0] == "condition-cache"
    assert progress_events[-1] == "condition-cache"


def test_polymarket_condition_cache_payload_tampering_fails_closed(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "condition-cache-tamper.duckdb") as store:
        _finish_replay_store(store, "condition-cache-tamper", feature_evidence=True)
        assert store.integrity_errors("condition-cache-tamper") == ()
        store.ensure_condition_message_cache("condition-cache-tamper")
        btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
        store.connect().execute(
            """
            UPDATE polymarket_condition_message_frame
            SET compressed_payload = ?
            WHERE run_id = ? AND condition_id = ? AND frame_index = 0
            """,
            [b"corrupted", "condition-cache-tamper", btc.condition_id],
        )

        with pytest.raises(ValueError, match="frame identity differs"):
            tuple(
                store.iter_public_events(
                    "condition-cache-tamper",
                    condition_ids=(btc.condition_id,),
                    verified_source=True,
                )
            )


def test_polymarket_action_pipeline_resumes_completed_bounded_batches(
    tmp_path,
    monkeypatch,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "action-pipeline.duckdb") as store:
        _finish_replay_store(store, "action-pipeline", feature_evidence=True)
        assert store._payload_connection().execute(
            "SELECT current_setting('preserve_insertion_order')"
        ).fetchone() == (False,)
        with pytest.raises(ValueError, match="bounded-memory policy"):
            PolymarketActionPipelineConfig(market_groups_per_batch=2).validated()
        config = PolymarketActionPipelineConfig(
            market_groups_per_batch=1,
            feature=PolymarketFeatureConfig(
                cadence_ms=250,
                warmup_ms=0,
                minimum_resolved_markets_per_asset=1,
            ),
        )
        segmented = PolymarketActionPipelineConfig(
            market_groups_per_batch=1,
            feature=replace(config.feature, allow_segmented_gaps=True),
        )
        with pytest.raises(ValueError, match="hash-bound eligible condition IDs"):
            materialize_polymarket_action_value_batches(
                store,
                run_id="action-pipeline",
                config=segmented,
            )

        progress_events: list[str] = []
        first = materialize_polymarket_action_value_batches(
            store,
            run_id="action-pipeline",
            config=config,
            progress=lambda phase, _payload: progress_events.append(phase),
        )
        second = materialize_polymarket_action_value_batches(
            store,
            run_id="action-pipeline",
            config=config,
        )
        monkeypatch.setattr(
            action_pipeline_module,
            "polymarket_action_pipeline_implementation_sha256",
            lambda: "f" * 64,
        )
        changed_implementation = materialize_polymarket_action_value_batches(
            store,
            run_id="action-pipeline",
            config=config,
        )

    assert first.report_sha256 == second.report_sha256
    assert first.action_count == second.action_count
    assert first.batches[0].status == "created"
    assert second.batches[0].status == "existing"
    assert first.batches[0].batch_sha256 == second.batches[0].batch_sha256
    assert len(first.implementation_sha256) == 64
    assert changed_implementation.implementation_sha256 == "f" * 64
    assert changed_implementation.report_sha256 != first.report_sha256
    assert changed_implementation.batches[0].status == "created"
    assert first.excluded_after_event_scope_count >= 0
    assert not first.asdict()["profitability_claim"]
    assert {"integrity-started", "integrity-cache-hit"} & set(progress_events)
    assert "condition-cache" in progress_events
    assert "feature-source-scan" in progress_events
    assert "feature-source-series" in progress_events


def test_round12_action_local_pipeline_uses_full_segmented_scope_and_persists_proof(
    tmp_path,
    capsys,
) -> None:
    contract_sha256 = json.loads(ROUND12_CONTRACT.read_text(encoding="utf-8"))[
        "contract_sha256"
    ]
    database = tmp_path / "round12-pipeline.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(
            store,
            "round12-pipeline",
            feature_evidence=True,
            pre_window_binance_gap=True,
            preregistration_manifest=_round12_capture_manifest(
                "round12-pipeline",
                EPOCH * 1_000,
            ),
        )
        config = replace(
            PolymarketActionPipelineConfig(),
            feature=replace(
                PolymarketActionPipelineConfig().feature,
                allow_segmented_gaps=True,
            ),
        )

        report = materialize_polymarket_action_value_batches(
            store,
            run_id="round12-pipeline",
            config=config,
            eligibility_sha256=contract_sha256,
            continuity_admission_mode="action_local",
        )
        repeated = materialize_polymarket_action_value_batches(
            store,
            run_id="round12-pipeline",
            config=config,
            eligibility_sha256=contract_sha256,
            continuity_admission_mode="action_local",
        )
        admission_counts = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_round12_admission_dataset")
            .fetchone()
        )

    assert report.continuity_admission_mode == "action_local"
    assert report.report_sha256 == repeated.report_sha256
    assert report.action_count > 0
    assert admission_counts == (len(report.batches),)

    status = cli.main(
        [
            "polymarket-action-value",
            "--database",
            str(database),
            "--run-id",
            "round12-pipeline",
            "--round12-contract",
            str(ROUND12_CONTRACT),
            "--memory-limit",
            "512MB",
            "--database-threads",
            "1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["continuity_admission_mode"] == "action_local"
    assert payload["report_sha256"] == report.report_sha256


def test_round13_pipeline_persists_label_free_scenarios_before_outcomes(
    tmp_path,
) -> None:
    contract_sha256 = "d" * 64
    program = _round13_program(contract_sha256)
    database = tmp_path / "round13-pipeline.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(
            store,
            "round13-pipeline",
            feature_evidence=True,
            synchronized_feature_evidence=True,
            pre_window_binance_gap=True,
            finalize_official=False,
        )
        config = replace(
            PolymarketActionPipelineConfig(),
            feature=replace(
                PolymarketActionPipelineConfig().feature,
                allow_segmented_gaps=True,
            ),
        )

        report = materialize_polymarket_action_value_batches(
            store,
            run_id="round13-pipeline",
            config=config,
            eligibility_sha256=contract_sha256,
            continuity_admission_mode="action_local",
            round13_program=program,
        )
        repeated = materialize_polymarket_action_value_batches(
            store,
            run_id="round13-pipeline",
            config=config,
            eligibility_sha256=contract_sha256,
            continuity_admission_mode="action_local",
            round13_program=program,
        )
        scenario = load_round13_label_free_dataset(
            store,
            source_action_dataset_sha256=report.batches[0].action_dataset_sha256,
        )
        resolution_count = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_resolution_evidence")
            .fetchone()[0]
        )
        feature_label_count = (
            store.connect()
            .execute(
                """
            SELECT count(*) FROM polymarket_feature_row
            WHERE dataset_id = ?
              AND (official_up IS NOT NULL OR resolution_event_id != '')
            """,
                [report.batches[0].feature_dataset_sha256],
            )
            .fetchone()[0]
        )

    assert report.report_sha256 == repeated.report_sha256
    assert report.batches[0].round13_scenario_dataset_sha256 == (
        scenario.dataset_sha256
    )
    assert scenario.contract_sha256 == contract_sha256
    assert len(scenario.calibration_snapshots) == 3
    assert resolution_count == 0
    assert feature_label_count == 0


def test_round13_pipeline_rejects_preexisting_official_resolution_evidence(
    tmp_path,
) -> None:
    database = tmp_path / "round13-preexisting-resolution.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(
            store,
            "round13-preexisting-resolution",
            feature_evidence=True,
            pre_window_binance_gap=True,
        )
        config = replace(
            PolymarketActionPipelineConfig(),
            feature=replace(
                PolymarketActionPipelineConfig().feature,
                allow_segmented_gaps=True,
            ),
        )

        with pytest.raises(ValueError, match="requires no official resolution"):
            materialize_polymarket_action_value_batches(
                store,
                run_id="round13-preexisting-resolution",
                config=config,
                eligibility_sha256="d" * 64,
                continuity_admission_mode="action_local",
                round13_program=_round13_program(),
            )


def test_polymarket_continuity_eligibility_is_label_free_and_tamper_evident(
    tmp_path,
    monkeypatch,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "continuity.duckdb") as store:
        _finish_replay_store(store, "continuity", feature_evidence=True)

        first = evaluate_polymarket_continuity_eligibility(
            store,
            run_id="continuity",
        )
        monkeypatch.setattr(
            continuity_module,
            "_continuity_evidence",
            lambda *_args, **_kwargs: pytest.fail(
                "persisted continuity evidence was recomputed"
            ),
        )
        second = evaluate_polymarket_continuity_eligibility(
            store,
            run_id="continuity",
        )
        store.connect().execute(
            """
            UPDATE polymarket_continuity_eligibility_group
            SET reasons_json = '[]'
            WHERE report_sha256 = ?
            """,
            [first.report_sha256],
        )
        with pytest.raises(ValueError, match="continuity report is inconsistent"):
            evaluate_polymarket_continuity_eligibility(
                store,
                run_id="continuity",
            )

    assert first.report_sha256 == second.report_sha256
    assert first.eligible_group_count == 0
    assert not first.confirmation_eligible
    assert "run_started_before_round9_contract_commit" in first.confirmation_reasons
    assert "run_started_after_window_start" in first.groups[0].reasons
    assert "clob_segment_started_after_window:BTC:Up" in first.groups[0].reasons
    assert first.groups[0].evidence["run_bounds"] == {
        "started_at_ms": EPOCH * 1_000,
        "ended_at_ms": EPOCH * 1_000 + 302_000,
    }
    assert first.groups[0].reasons
    assert first.asdict()["outcomes_consulted"] is False
    assert first.asdict()["labels_consulted"] is False


def test_polymarket_continuity_excludes_gap_opened_before_window(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "continuity-gap.duckdb") as store:
        _finish_replay_store(
            store,
            "continuity-gap",
            feature_evidence=True,
            finalize_official=False,
            run_started_at_ms=EPOCH * 1_000 - 10_000,
            pre_window_binance_gap=True,
        )
        report = evaluate_polymarket_continuity_eligibility(
            store,
            run_id="continuity-gap",
        )

    reasons = report.groups[0].reasons
    assert "stream_gap:binance_spot:1" in reasons
    assert "run_started_after_window_start" not in reasons
    assert "binance_segment_started_after_window:BTC" in reasons


def test_polymarket_continuity_cli_and_native_contract_share_controls(
    tmp_path,
    capsys,
) -> None:
    database = tmp_path / "continuity-cli.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "continuity-cli", feature_evidence=True)

    status = cli.main(
        [
            "polymarket-continuity",
            "--database",
            str(database),
            "--run-id",
            "continuity-cli",
            "--memory-limit",
            "512MB",
            "--database-threads",
            "1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    continuity_spec = next(
        spec for spec in command_specs() if spec.name == "polymarket-continuity"
    )
    action_spec = next(
        spec for spec in command_specs() if spec.name == "polymarket-action-value"
    )

    assert status == 2
    assert payload["outcomes_consulted"] is False
    assert payload["confirmation_eligible"] is False
    assert {option.dest for option in continuity_spec.options} >= {
        "database",
        "run_id",
        "memory_limit",
        "database_threads",
        "json",
    }
    assert {"allow_segmented_gaps", "round12_contract"}.issubset(
        {option.dest for option in action_spec.options}
    )


def test_polymarket_feature_provenance_binds_pre_window_causal_events(
    tmp_path,
) -> None:
    config = PolymarketFeatureConfig(
        cadence_ms=50,
        warmup_ms=0,
        minimum_resolved_markets_per_asset=1,
    )
    with PolymarketEvidenceStore(tmp_path / "prefix-a.duckdb") as first_store:
        _finish_replay_store(
            first_store,
            "causal-prefix",
            feature_evidence=True,
            pre_window_trade_quantity="0.1",
        )
        first = build_polymarket_feature_dataset(
            first_store,
            run_id="causal-prefix",
            config=config,
        )
    with PolymarketEvidenceStore(tmp_path / "prefix-b.duckdb") as second_store:
        _finish_replay_store(
            second_store,
            "causal-prefix",
            feature_evidence=True,
            pre_window_trade_quantity="0.2",
        )
        second = build_polymarket_feature_dataset(
            second_store,
            run_id="causal-prefix",
            config=config,
        )

    assert first.rows[0].feature_values == second.rows[0].feature_values
    assert (
        first.rows[0].input_provenance_sha256 != second.rows[0].input_provenance_sha256
    )
    assert first.rows[0].row_sha256 != second.rows[0].row_sha256


def test_polymarket_feature_cli_and_generated_windows_contract_share_options(
    tmp_path,
    capsys,
) -> None:
    database = tmp_path / "feature-cli.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "feature-cli", feature_evidence=True)

    status_code = cli.main(
        [
            "polymarket-features",
            "--database",
            str(database),
            "--run-id",
            "feature-cli",
            "--cadence-ms",
            "50",
            "--warmup-ms",
            "0",
            "--minimum-resolved-markets-per-asset",
            "1",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    spec = next(spec for spec in command_specs() if spec.name == "polymarket-features")

    assert status_code == 2
    assert payload["row_count"] >= 1
    assert payload["labeled_row_count"] >= 1
    assert payload["materialization"]["status"] == "created"
    assert payload["shadow_ready"] is False
    assert payload["training_ready"] is False
    assert {option.dest for option in spec.options} == {
        "database",
        "run_id",
        "cadence_ms",
        "warmup_ms",
        "minimum_resolved_markets_per_asset",
        "allow_segmented_gaps",
        "memory_limit",
        "database_threads",
        "json",
    }


def test_polymarket_model_generated_windows_contract_exposes_typed_controls() -> None:
    spec = next(spec for spec in command_specs() if spec.name == "polymarket-model")

    assert {option.dest for option in spec.options} == {
        "database",
        "run_id",
        "cadence_ms",
        "warmup_ms",
        "minimum_resolved_markets_per_asset",
        "allow_segmented_gaps",
        "latency_ms",
        "latency_stress_ms",
        "max_execution_observation_delay_ms",
        "minimum_edge",
        "initial_capital",
        "maximum_loss_fraction_per_market",
        "maximum_loss_fraction_per_time_group",
        "ai_enabled",
        "ai_model",
        "ai_benchmark",
        "ai_url",
        "ai_timeout",
        "ai_min_confidence",
        "ai_max_latency_seconds",
        "output",
        "memory_limit",
        "database_threads",
        "json",
    }
    assert next(
        option for option in spec.options if option.dest == "latency_ms"
    ).takes_value
    ai_options = [option for option in spec.options if option.dest == "ai_enabled"]
    assert {option.flags for option in ai_options} == {
        ("--enable-ai",),
        ("--disable-ai",),
    }
    assert all(option.default is None for option in ai_options)
    assert (
        next(
            option
            for option in spec.options
            if option.dest == "minimum_resolved_markets_per_asset"
        ).default
        == 30
    )


def test_polymarket_model_ai_mode_uses_runtime_default_and_explicit_overrides(
    monkeypatch,
) -> None:
    parser = cli._build_parser()
    inherited = parser.parse_args(["polymarket-model"])
    enabled = parser.parse_args(["polymarket-model", "--enable-ai"])
    disabled = parser.parse_args(["polymarket-model", "--disable-ai"])
    explicit_model = parser.parse_args(["polymarket-model", "--ai-model", "qwen3:14b"])
    runtime = cli.load_runtime({"ai_enabled": False, "ai_model": "qwen3.5:9b"})
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)

    assert cli._polymarket_ai_enabled(inherited) is False
    assert cli._polymarket_ai_enabled(enabled) is True
    assert cli._polymarket_ai_enabled(disabled) is False
    assert inherited.ai_model is None
    assert cli._polymarket_ai_model(inherited) == "qwen3.5:9b"
    assert cli._polymarket_ai_model(explicit_model) == "qwen3:14b"

    runtime.ai_enabled = True
    assert cli._polymarket_ai_enabled(inherited) is True


def test_polymarket_source_verification_is_in_the_shared_command_contract() -> None:
    verify = next(spec for spec in command_specs() if spec.name == "polymarket-verify")
    publish = next(
        spec for spec in command_specs() if spec.name == "polymarket-publish"
    )

    assert {option.dest for option in verify.options} == {
        "artifact",
        "database",
        "output",
        "memory_limit",
        "database_threads",
        "json",
    }
    assert next(
        option for option in verify.options if option.dest == "artifact"
    ).required
    assert {option.dest for option in publish.options} == {
        "artifact",
        "database",
        "research_root",
        "round",
        "prior_round",
        "memory_limit",
        "database_threads",
        "json",
    }


def test_replay_rejects_semantically_inconsistent_published_best_price(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "bad-best.duckdb") as store:
        _finish_replay_store(store, "bad-best", wrong_best=True)
        with pytest.raises(ValueError, match="checksum disagrees"):
            PolymarketEvidenceReplay.load(store, run_id="bad-best")


def test_replay_refuses_noncomplete_run(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "running.duckdb") as store:
        store.start_run("still-running", EPOCH * 1_000)
        with pytest.raises(ValueError, match="complete gap-free"):
            PolymarketEvidenceReplay.load(store, run_id="still-running")


def test_segmented_replay_resets_books_and_never_executes_across_gap(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "segmented.duckdb") as store:
        _finish_segmented_store(store, "segmented-run")
        with pytest.raises(ValueError, match="complete gap-free"):
            PolymarketEvidenceReplay.load(store, run_id="segmented-run")
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="segmented-run",
            allow_segmented_gaps=True,
        )

    assert replay.diagnostics.continuity_mode == "segmented"
    assert replay.diagnostics.stream_gap_count == 1
    assert replay.diagnostics.clob_connection_segment_count == 2
    assert replay.diagnostics.state_reset_count == 1
    segment_ids = {book.segment_id for book in replay.books}
    assert len(segment_ids) == 2
    first_segment = replay.books[0].segment_id
    old_up = [
        book
        for book in replay.books
        if book.segment_id == first_segment and book.outcome == "Up"
    ][-1]
    assert (
        replay.first_book_after_latency(
            old_up,
            latency_ms=2_000,
            maximum_observation_delay_ms=500,
        )
        is None
    )
    for segment_id in segment_ids:
        assert {
            book.outcome for book in replay.books if book.segment_id == segment_id
        } == {
            "Up",
            "Down",
        }


def test_segmented_replay_uses_reconnect_book_tick_without_metadata_fallback(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "reconnect-tick.duckdb") as store:
        _finish_segmented_store(
            store,
            "reconnect-tick",
            second_segment_tick_size="0.001",
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="reconnect-tick",
            allow_segmented_gaps=True,
        )

    second_segment = replay.books[-1].segment_id
    assert {
        book.tick_size for book in replay.books if book.segment_id == second_segment
    } == {Decimal("0.001")}


def test_segmented_replay_blocks_reconnect_without_same_segment_tick(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "unknown-reconnect-tick.duckdb") as store:
        _finish_segmented_store(
            store,
            "unknown-reconnect-tick",
            second_segment_tick_size=None,
        )
        with pytest.raises(ValueError, match="same-segment active tick"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="unknown-reconnect-tick",
                allow_segmented_gaps=True,
            )


def _scope_boundary_best_messages(
    *, corroborated: bool
) -> tuple[RawStreamMessage, ...]:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    cutoff_offset_ms = btc.end_ms - EPOCH * 1_000 - 29_500
    source_time_ms = EPOCH * 1_000 + cutoff_offset_ms
    return (
        _message(
            "clob_market",
            {
                "event_type": "best_bid_ask",
                "market": btc.condition_id,
                "asset_id": btc.up_token_id,
                "best_bid": "0.499",
                "best_ask": "0.50",
                "spread": "0.001",
                "timestamp": str(source_time_ms),
            },
            sequence=100,
            wall_offset_ms=cutoff_offset_ms - 1,
            monotonic_ns=cutoff_offset_ms * 1_000_000,
        ),
        _message(
            "clob_market",
            {
                "event_type": "price_change",
                "market": btc.condition_id,
                "timestamp": str(source_time_ms),
                "price_changes": [
                    {
                        "asset_id": btc.up_token_id,
                        "price": "0.499" if corroborated else "0.40",
                        "size": "5",
                        "side": "BUY",
                        "hash": "scope-boundary-proof",
                        "best_bid": "0.499",
                        "best_ask": "0.50",
                    }
                ],
            },
            sequence=101,
            wall_offset_ms=cutoff_offset_ms + 1,
            monotonic_ns=cutoff_offset_ms * 1_000_000,
        ),
    )


def test_replay_discards_scope_tail_best_proven_by_excluded_same_group_delta(
    tmp_path,
) -> None:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    cutoff = btc.end_ms - 29_500
    with PolymarketEvidenceStore(tmp_path / "scope-tail-proof.duckdb") as store:
        _finish_replay_store(
            store,
            "scope-tail-proof",
            additional_messages=_scope_boundary_best_messages(corroborated=True),
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="scope-tail-proof",
            condition_ids=(btc.condition_id,),
            maximum_received_wall_ms_by_condition={btc.condition_id: cutoff},
        )

    assert replay.books[-1].snapshot.bids[0].price == Decimal("0.498")
    assert replay.diagnostics.discarded_uncorroborated_best_count == 1
    assert replay.diagnostics.excluded_after_event_scope_count == 1
    assert all(book.received_wall_ms <= cutoff for book in replay.books)


def test_replay_rejects_scope_tail_best_without_exact_excluded_delta_proof(
    tmp_path,
) -> None:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    cutoff = btc.end_ms - 29_500
    with PolymarketEvidenceStore(tmp_path / "scope-tail-mismatch.duckdb") as store:
        _finish_replay_store(
            store,
            "scope-tail-mismatch",
            additional_messages=_scope_boundary_best_messages(corroborated=False),
        )
        with pytest.raises(ValueError, match="not corroborated"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="scope-tail-mismatch",
                condition_ids=(btc.condition_id,),
                maximum_received_wall_ms_by_condition={btc.condition_id: cutoff},
            )


def test_replay_event_scope_excludes_only_post_contract_receipts(tmp_path) -> None:
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    late_book = _segmented_message(
        {
            "event_type": "book",
            "market": btc.condition_id,
            "asset_id": btc.up_token_id,
            "timestamp": str(btc.end_ms + 1_000),
            "tick_size": "0.01",
            "hash": "post-contract-off-tick",
            "bids": [{"price": "0.999", "size": "5"}],
            "asks": [],
        },
        connection_id="late-out-of-scope-connection",
        sequence=1,
        wall_offset_ms=301_100,
        monotonic_ns=301_100_000_000,
    )
    with PolymarketEvidenceStore(tmp_path / "event-scope.duckdb") as store:
        _finish_replay_store(
            store,
            "event-scope",
            additional_messages=(late_book,),
        )
        with pytest.raises(ValueError, match="price off the active tick"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="event-scope",
                condition_ids=(btc.condition_id,),
            )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="event-scope",
            condition_ids=(btc.condition_id,),
            maximum_received_wall_ms_by_condition={
                btc.condition_id: btc.end_ms - 29_500
            },
        )

    assert replay.diagnostics.event_scope_mode == (
        "condition_received_wall_upper_bound"
    )
    assert len(replay.diagnostics.event_scope_sha256) == 64
    assert replay.diagnostics.excluded_after_event_scope_count == 1
    assert all(book.received_wall_ms <= btc.end_ms - 29_500 for book in replay.books)


def test_sampled_replay_hashes_all_transitions_and_keeps_segment_baselines(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "sampled-segments.duckdb") as store:
        _finish_segmented_store(store, "sampled-segments")
        full = PolymarketEvidenceReplay.load(
            store,
            run_id="sampled-segments",
            allow_segmented_gaps=True,
        )
        sampled = PolymarketEvidenceReplay.load(
            store,
            run_id="sampled-segments",
            allow_segmented_gaps=True,
            book_sample_interval_ms=5_000,
        )

    assert sampled.diagnostics.book_state_transition_count == len(full.books)
    assert sampled.diagnostics.materialized_book_count == len(sampled.books)
    assert sampled.diagnostics.suppressed_book_count == len(full.books) - len(
        sampled.books
    )
    assert len(sampled.books) < len(full.books)
    for segment_id in {book.segment_id for book in sampled.books}:
        assert {
            book.outcome for book in sampled.books if book.segment_id == segment_id
        } == {"Up", "Down"}


def test_segmented_replay_requires_fresh_baseline_after_reconnect(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "missing-baseline.duckdb") as store:
        _finish_segmented_store(
            store,
            "missing-baseline",
            second_segment_has_baseline=False,
        )
        with pytest.raises(ValueError, match="without a proven token baseline"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="missing-baseline",
                allow_segmented_gaps=True,
            )


@pytest.mark.parametrize("gap_stream", ["binance_spot", "polymarket_rtds"])
def test_segmented_replay_validates_independent_feed_connection_gaps(
    tmp_path,
    gap_stream: str,
) -> None:
    with PolymarketEvidenceStore(tmp_path / f"{gap_stream}.duckdb") as store:
        _finish_segmented_store(store, "non-clob-gap", gap_stream=gap_stream)
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="non-clob-gap",
            allow_segmented_gaps=True,
        )

    assert replay.diagnostics.continuity_mode == "segmented"
    assert replay.diagnostics.stream_gap_count == 1


def test_segmented_replay_rejects_missing_named_lane_transition_gap(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "missing-transition-gap.duckdb") as store:
        _finish_segmented_store(
            store,
            "missing-transition-gap",
            named_clob_connections=True,
        )
        store.connect().execute(
            "DELETE FROM polymarket_stream_gap WHERE run_id = ?",
            ["missing-transition-gap"],
        )

        with pytest.raises(ValueError, match="transition has no gap evidence"):
            PolymarketEvidenceReplay.validate_stream_gaps(
                store,
                "missing-transition-gap",
                allow_segmented_gaps=True,
            )


def test_segmented_replay_accepts_explicit_zero_message_connection_attempt(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "zero-message-attempt.duckdb") as store:
        store.start_run("zero-message-attempt", EPOCH * 1_000)
        store.record_gap(
            "zero-message-attempt",
            StreamGap(
                stream="polymarket_rtds",
                connection_id=("rtds:chainlink:btc:33333333333333333333333333333333"),
                opened_at_ms=EPOCH * 1_000 + 1_000,
                reason="fixture_connect_failure",
                last_sequence_number=0,
            ),
        )

        assert (
            PolymarketEvidenceReplay.validate_stream_gaps(
                store,
                "zero-message-attempt",
                allow_segmented_gaps=True,
            )
            == 1
        )


def test_feature_feed_windows_never_cross_connection_segments() -> None:
    first = feature_module._BinanceBookPoint(
        asset="BTC",
        connection_id="binance:first",
        received_wall_ms=1_000,
        received_monotonic_ns=1_000_000_000,
        bid=99.0,
        bid_quantity=1.0,
        ask=101.0,
        ask_quantity=1.0,
        event_id="first",
        event_sha256="1" * 64,
    )
    second = replace(
        first,
        connection_id="binance:second",
        received_wall_ms=2_000,
        received_monotonic_ns=2_000_000_000,
        event_id="second",
        event_sha256="2" * 64,
    )
    new_connection_trade = feature_module._BinanceTradePoint(
        asset="BTC",
        connection_id="binance:second",
        received_monotonic_ns=1_500_000_000,
        signed_quote=Decimal("1"),
        gross_quote=Decimal("1"),
        event_id="new-connection-trade",
        event_sha256="5" * 64,
    )
    connection_cursor = feature_module._ConnectionCursor(
        (first,), (new_connection_trade,)
    )
    assert connection_cursor.advance(1_000_000_000) == "binance:first"
    assert connection_cursor.advance(1_500_000_000) == "binance:second"
    tied_trade = replace(
        new_connection_trade,
        received_monotonic_ns=first.received_monotonic_ns,
        event_id=first.event_id,
    )
    tie_cursor = feature_module._ConnectionCursor((first,), (tied_trade,))
    assert tie_cursor.advance(first.received_monotonic_ns) == "binance:second"
    with pytest.raises(ValueError, match="crossed connection segments"):
        feature_module._BookSeries(
            (first, second),
            source_scope_sha256="f" * 64,
        )

    cursor = feature_module._PriceCursor(
        (
            feature_module._PricePoint(
                asset="BTC",
                connection_id="chainlink:first",
                source_time_ms=3_000,
                received_wall_ms=1_001,
                received_monotonic_ns=1_001_000_000,
                price=100.0,
                event_id="anchor-old",
                event_sha256="3" * 64,
            ),
            feature_module._PricePoint(
                asset="BTC",
                connection_id="chainlink:second",
                source_time_ms=2_000,
                received_wall_ms=2_001,
                received_monotonic_ns=2_001_000_000,
                price=102.0,
                event_id="current-new",
                event_sha256="4" * 64,
            ),
        )
    )
    cursor.advance(3_000_000_000)
    assert cursor.latest_at_or_before(3_000).connection_id == "chainlink:first"
    current = cursor.latest_at_or_before(
        3_000,
        connection_id=cursor.active_connection_id(),
    )

    assert current is not None and current.connection_id == "chainlink:second"
    assert (
        cursor.latest_at_or_before(1_000, connection_id=current.connection_id) is None
    )


def test_feature_series_compact_storage_preserves_frozen_prefixes_and_math() -> None:
    first_book = feature_module._BinanceBookPoint(
        asset="BTC",
        connection_id="binance:first",
        received_wall_ms=1_000,
        received_monotonic_ns=1_000_000_000,
        bid=99.0,
        bid_quantity=1.0,
        ask=101.0,
        ask_quantity=1.0,
        event_id="first",
        event_sha256="1" * 64,
    )
    second_book = replace(
        first_book,
        received_wall_ms=2_000,
        received_monotonic_ns=2_000_000_000,
        bid=100.0,
        ask=102.0,
        event_id="second",
        event_sha256="2" * 64,
    )
    first_trade = feature_module._BinanceTradePoint(
        asset="BTC",
        connection_id="binance:first",
        received_monotonic_ns=1_100_000_000,
        signed_quote=Decimal("1.25"),
        gross_quote=Decimal("2.5"),
        event_id="trade-a",
        event_sha256="3" * 64,
    )
    second_trade = replace(
        first_trade,
        received_monotonic_ns=1_900_000_000,
        signed_quote=Decimal("-0.75"),
        gross_quote=Decimal("0.75"),
        event_id="trade-b",
        event_sha256="4" * 64,
    )

    books = feature_module._BookSeries(
        (second_book, first_book),
        source_scope_sha256="f" * 64,
    )
    trades = feature_module._TradeSeries(
        (second_trade, first_trade),
        source_scope_sha256="f" * 64,
    )

    assert isinstance(books.prefix_digests[0], bytes)
    assert isinstance(trades.prefix_digests[0], bytes)
    assert books.causal_prefix(0) == (
        0,
        "a2e6d9ba539aab6aba0654c1edbaa7291c312c58e671b951f6a375a3feaea133",
    )
    assert books.causal_prefix(1_000_000_000) == (
        1,
        "5a1b639566d7a89526793d60e50ba2f4305d9e2082a3bbf2276dd53b5cd11d90",
    )
    assert books.causal_prefix(2_000_000_000) == (
        2,
        "160eab29a9e4599d9e481af98b1907217e1fcd55ddae9e41c46d9d2c1137ca24",
    )
    assert trades.causal_prefix(0) == (
        0,
        "a49f2342cbef728020b248a3d9ca04071a540ac3ebbc861c0a0c02561c4b90d5",
    )
    assert trades.causal_prefix(1_100_000_000) == (
        1,
        "cee3e5f51869a8d560f67d0bf70264c58f82bda5b942d6b3bc04fc6289fd0a5c",
    )
    assert trades.causal_prefix(1_900_000_000) == (
        2,
        "c04dd18d2f1eacaef543b9039efc4d058f9b7ebabe83cb9ea3115e652badc1e7",
    )
    assert books.return_bps(2_000_000_000, 1_000) == 99.50330853168091
    assert books.realized_volatility_bps(2_000_000_000, 1_000) == (99.50330853168091)
    assert trades.stats(2_000_000_000, 1_000) == (
        0.15384615384615385,
        3.25,
    )


def test_compact_binance_books_preserve_point_cursor_and_series_semantics() -> None:
    first = feature_module._BinanceBookPoint(
        asset="BTC",
        connection_id="binance:first",
        received_wall_ms=1_000,
        received_monotonic_ns=1_000_000_000,
        bid=99.0,
        bid_quantity=1.25,
        ask=101.0,
        ask_quantity=1.5,
        event_id="b" * 64,
        event_sha256="1" * 64,
    )
    tied = replace(
        first,
        bid=98.0,
        ask=100.0,
        event_id="a" * 64,
        event_sha256="2" * 64,
    )
    second = replace(
        first,
        received_wall_ms=2_000,
        received_monotonic_ns=2_000_000_000,
        bid=100.0,
        ask=102.0,
        event_id="c" * 64,
        event_sha256="3" * 64,
    )
    compact = feature_module._CompactBinanceBooks("BTC")
    compact.append(first)
    compact.append(tied)
    compact.append(second)
    compact.finish()

    expected = tuple(
        sorted((first, tied, second), key=feature_module._received_order_key)
    )
    assert tuple(compact) == expected
    assert compact.finish() is compact
    assert compact.connection_views()["binance:first"].connection_id == (
        "binance:first"
    )

    compact_series = feature_module._BookSeries(
        compact.connection_views()["binance:first"],
        source_scope_sha256="f" * 64,
    )
    tuple_series = feature_module._BookSeries(
        expected,
        source_scope_sha256="f" * 64,
    )
    for received_ns in (0, 1_000_000_000, 2_000_000_000):
        assert compact_series.causal_prefix(received_ns) == tuple_series.causal_prefix(
            received_ns
        )
    assert compact_series.return_bps(2_000_000_000, 1_000) == (
        tuple_series.return_bps(2_000_000_000, 1_000)
    )
    assert compact_series.realized_volatility_bps(2_000_000_000, 1_000) == (
        tuple_series.realized_volatility_bps(2_000_000_000, 1_000)
    )

    cursor = feature_module._BookCursor(compact)
    assert cursor.advance(999_999_999) is None
    assert cursor.advance(1_000_000_000) == expected[1]
    assert cursor.advance(2_000_000_000) == expected[2]


def test_compact_binance_trades_preserve_point_cursor_and_series_semantics() -> None:
    first = feature_module._BinanceTradePoint(
        asset="BTC",
        connection_id="binance:first",
        received_monotonic_ns=1_000_000_000,
        signed_quote=Decimal("-12.3400"),
        gross_quote=Decimal("12.3400"),
        event_id="b" * 64,
        event_sha256="1" * 64,
    )
    tied = replace(
        first,
        signed_quote=Decimal("2.50"),
        gross_quote=Decimal("2.50"),
        event_id="a" * 64,
        event_sha256="2" * 64,
    )
    second = replace(
        first,
        received_monotonic_ns=2_000_000_000,
        signed_quote=Decimal("5.125"),
        gross_quote=Decimal("5.125"),
        event_id="c" * 64,
        event_sha256="3" * 64,
    )
    compact = feature_module._CompactBinanceTrades("BTC")
    compact.append(first)
    compact.append(tied)
    compact.append(second)
    compact.finish()

    expected = tuple(
        sorted((first, tied, second), key=feature_module._received_order_key)
    )
    assert tuple(compact) == expected
    assert compact.finish() is compact
    view = compact.connection_views()["binance:first"]
    compact_series = feature_module._TradeSeries(
        view,
        source_scope_sha256="f" * 64,
    )
    tuple_series = feature_module._TradeSeries(
        expected,
        source_scope_sha256="f" * 64,
    )
    for received_ns in (0, 1_000_000_000, 2_000_000_000):
        assert compact_series.causal_prefix(received_ns) == tuple_series.causal_prefix(
            received_ns
        )
        assert compact_series.stats(received_ns, 1_000) == tuple_series.stats(
            received_ns,
            1_000,
        )

    books = feature_module._CompactBinanceBooks("BTC")
    books.append(
        feature_module._BinanceBookPoint(
            asset="BTC",
            connection_id="binance:first",
            received_wall_ms=1_000,
            received_monotonic_ns=500_000_000,
            bid=99.0,
            bid_quantity=1.0,
            ask=101.0,
            ask_quantity=1.0,
            event_id="d" * 64,
            event_sha256="4" * 64,
        )
    )
    books.finish()
    cursor = feature_module._ConnectionCursor(books, compact)
    assert cursor.advance(500_000_000) == "binance:first"
    assert cursor.advance(2_000_000_000) == "binance:first"


def test_segmented_replay_requires_gap_to_close_final_sequence(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "bad-gap-sequence.duckdb") as store:
        _finish_segmented_store(
            store,
            "bad-gap-sequence",
            gap_last_sequence=2,
        )
        with pytest.raises(ValueError, match="final sequence"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="bad-gap-sequence",
                allow_segmented_gaps=True,
            )


def test_segmented_polymarket_paper_uses_shared_owned_journal(tmp_path) -> None:
    database = tmp_path / "segmented-paper.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_segmented_store(store, "segmented-paper")

    with pytest.raises(ValueError, match="complete gap-free"):
        PolymarketPaperBroker(database, run_id="segmented-paper")

    with PolymarketPaperBroker(
        database,
        run_id="segmented-paper",
        allow_segmented_gaps=True,
    ) as broker:
        first_segment = broker.replay.books[0].segment_id
        decision = next(
            book
            for book in broker.replay.books
            if book.segment_id == first_segment and book.outcome == "Up"
        )
        position, result = broker.open_position(
            position_id="segmented-owned-position",
            decision=decision,
            outcome="Up",
            quantity="5",
            maximum_price="0.52",
            submission_latency_ms=500,
        )
        reconciliation = broker.reconcile()

    assert position is not None
    assert result.state == "FILLED"
    assert reconciliation.ok is True
    assert reconciliation.journal.inventory[0].venue == "polymarket"


def test_polymarket_broker_opens_and_closes_on_post_latency_depth_with_fees(
    tmp_path,
) -> None:
    database = tmp_path / "broker.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "broker-run")

    with PolymarketPaperBroker(database, run_id="broker-run") as broker:
        full, _changed, post_tick, _close_book = broker.replay.books
        position, opened = broker.open_position(
            position_id="position-1",
            decision=full,
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )

        assert position is not None
        assert opened.state == "FILLED"
        assert position.average_entry_price == Decimal("0.50")
        assert position.remaining_entry_fee_quote == Decimal("0.08750")
        assert broker.reconcile().can_open is True

        closed, close_result = broker.close_position(
            opening_intent_id=position.opening_intent_id,
            decision=post_tick,
            minimum_price="0.490",
            submission_latency_ms=5,
        )

        assert closed is not None
        assert close_result.state == "FILLED"
        assert closed.average_exit_price == Decimal("0.498")
        assert closed.entry_fee_quote == Decimal("0.08750")
        assert closed.exit_fee_quote == Decimal("0.08750")
        assert closed.realized_pnl_quote == Decimal("-0.18500")
        assert broker.positions() == ()
        assert broker.reconcile().can_open is True


def test_polymarket_broker_binds_fok_ai_delay_and_submission_latency(
    tmp_path,
) -> None:
    database = tmp_path / "fok-delay.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "fok-delay-run")

    with PolymarketPaperBroker(database, run_id="fok-delay-run") as broker:
        decision = broker.replay.books[0]
        position, execution = broker.open_position(
            position_id="fok-delay-position",
            decision=decision,
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            decision_delay_ms=5,
            submission_latency_ms=5,
            order_type="FOK",
        )
        assert position is not None
        intent = broker.journal.intent(position.opening_intent_id)
        context = (
            broker.store.connect()
            .execute(
                """
            SELECT requested_latency_ms, effective_latency_ms
            FROM polymarket_paper_order_context WHERE intent_id = ?
            """,
                [position.opening_intent_id],
            )
            .fetchone()
        )

    assert execution.state == "FILLED"
    assert intent.order_type == "FOK"
    assert intent.created_at_ms == decision.received_wall_ms + 5
    assert context is not None
    assert context[0] == 10
    assert context[1] >= context[0]


def test_model_research_and_owned_paper_broker_have_exact_execution_parity(
    tmp_path,
) -> None:
    database = tmp_path / "model-paper-parity.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "model-paper-parity-run")
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id="model-paper-parity-run",
            book_sample_interval_ms=0,
        )
    decision = replay.books[0]
    market = decision.market
    resolution = next(
        item for item in replay.resolutions if item.condition_id == market.condition_id
    )
    sample = PolymarketModelSample(
        sample_id="a" * 64,
        source_run_id=replay.run_id,
        source_feature_id="b" * 64,
        condition_id=market.condition_id,
        market_id=market.market_id,
        asset=market.asset,
        event_start_ms=market.event_start_ms,
        end_ms=market.end_ms,
        decision_received_wall_ms=decision.received_wall_ms,
        decision_received_monotonic_ns=decision.received_monotonic_ns,
        decision_event_id=decision.event_id,
        horizon_seconds=240,
        feature_values=tuple(0.0 for _ in POLYMARKET_MODEL_FEATURE_NAMES),
        risk_context_values=tuple(0.0 for _ in POLYMARKET_MODEL_RISK_CONTEXT_NAMES),
        baseline_up_probability=0.9,
        up_best_bid=0.49,
        up_best_ask=0.51,
        down_best_bid=0.49,
        down_best_ask=0.51,
        official_up=True,
        resolution_event_id=resolution.event_id,
        market_weight=1.0,
        input_provenance_sha256="c" * 64,
        sample_sha256="d" * 64,
    )
    config = PolymarketExecutionResearchConfig(submission_latency_ms=5)
    research = evaluate_polymarket_execution_policy(
        (sample,),
        (0.9,),
        replay,
        config=config,
    )
    assert len(research.trades) == 1
    expected = research.trades[0]

    with PolymarketPaperBroker(database, run_id=replay.run_id) as broker:
        coordinator = PolymarketPaperCoordinator(
            broker,
            control_path=tmp_path / "model-paper-parity.control.json",
        )
        assert coordinator.resume()["state"] == "RUNNING"
        coordinator.require_open_allowed()
        position, execution = broker.open_position(
            position_id=expected.sample_id,
            decision=decision,
            outcome=expected.outcome,
            quantity=expected.quantity,
            maximum_price=expected.limit_price,
            submission_latency_ms=expected.submission_latency_ms,
            decision_delay_ms=expected.decision_delay_ms,
            order_type="FOK",
        )
        assert position is not None
        context = (
            broker.store.connect()
            .execute(
                """
            SELECT decision_event_id, execution_event_id, effective_latency_ms
            FROM polymarket_paper_order_context WHERE intent_id = ?
            """,
                [position.opening_intent_id],
            )
            .fetchone()
        )
        settlement = broker.settle_position(
            opening_intent_id=position.opening_intent_id,
            resolution=resolution,
        )
        assert coordinator.pause()["state"] == "PAUSED"
        assert broker.positions() == ()

    assert execution.state == expected.execution_state == "FILLED"
    assert execution.filled_quantity == expected.filled_quantity
    assert execution.average_fill_price == expected.average_fill_price
    assert execution.fee_quote == expected.fee_quote
    assert execution.source_payload_sha256 == expected.source_payload_sha256
    assert context == (
        expected.decision_book_event_id,
        expected.execution_book_event_id,
        expected.effective_latency_ms,
    )
    assert settlement.gross_payout_quote == expected.gross_payout_quote
    assert settlement.realized_pnl_quote == expected.realized_pnl_quote


def _single_trade_model_plan(
    database,
    *,
    run_id: str,
    maximum_execution_observation_delay_ms: int = 500,
):
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, run_id)
        recorder_report_sha256 = str(
            store.connect()
            .execute(
                "SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
                [run_id],
            )
            .fetchone()[0]
        )
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=run_id,
            book_sample_interval_ms=0,
        )
    decision = replay.books[0]
    market = decision.market
    resolution = next(
        item for item in replay.resolutions if item.condition_id == market.condition_id
    )
    sample = PolymarketModelSample(
        sample_id="e" * 64,
        source_run_id=run_id,
        source_feature_id="f" * 64,
        condition_id=market.condition_id,
        market_id=market.market_id,
        asset=market.asset,
        event_start_ms=market.event_start_ms,
        end_ms=market.end_ms,
        decision_received_wall_ms=decision.received_wall_ms,
        decision_received_monotonic_ns=decision.received_monotonic_ns,
        decision_event_id=decision.event_id,
        horizon_seconds=240,
        feature_values=tuple(0.0 for _ in POLYMARKET_MODEL_FEATURE_NAMES),
        risk_context_values=tuple(0.0 for _ in POLYMARKET_MODEL_RISK_CONTEXT_NAMES),
        baseline_up_probability=0.9,
        up_best_bid=0.49,
        up_best_ask=0.51,
        down_best_bid=0.49,
        down_best_ask=0.51,
        official_up=True,
        resolution_event_id=resolution.event_id,
        market_weight=1.0,
        input_provenance_sha256="1" * 64,
        sample_sha256="2" * 64,
    )
    config = PolymarketExecutionResearchConfig(
        submission_latency_ms=5,
        maximum_execution_observation_delay_ms=(maximum_execution_observation_delay_ms),
    )
    research = evaluate_polymarket_execution_policy(
        (sample,),
        (0.9,),
        replay,
        config=config,
    )
    research_payload = research.asdict()
    blocking_reasons = (
        ("all_order_outcomes_terminal",)
        if any(item.execution_state == "UNKNOWN" for item in research.trades)
        else ()
    )
    provisional = PolymarketPaperPlan(
        schema_version=POLYMARKET_PAPER_PLAN_SCHEMA_VERSION,
        artifact_sha256="3" * 64,
        source_verification_sha256="4" * 64,
        recorder_report_sha256=recorder_report_sha256,
        run_id=run_id,
        allow_segmented_gaps=False,
        policy="model",
        primary_network_latency_ms=5,
        confirmed_for_paper_run=not blocking_reasons,
        research_override=bool(blocking_reasons),
        blocking_reasons=blocking_reasons,
        execution_report_sha256=research.report_sha256,
        execution_config=research_payload["config"],
        trades=tuple(research_payload["trades"]),
        plan_sha256="",
    )
    plan_identity = provisional.asdict()
    plan_identity.pop("plan_sha256")
    plan = replace(
        provisional,
        plan_sha256=_sha(_canonical(plan_identity)),
    )
    return plan, research


def test_verified_model_plan_uses_owned_paper_lifecycle_and_recorder_identity(
    tmp_path,
) -> None:
    database = tmp_path / "verified-model-plan.duckdb"
    run_id = "verified-model-plan-run"
    plan, research = _single_trade_model_plan(database, run_id=run_id)
    wrong_recorder = replace(plan, recorder_report_sha256="5" * 64, plan_sha256="")
    wrong_identity = wrong_recorder.asdict()
    wrong_identity.pop("plan_sha256")
    wrong_recorder = replace(
        wrong_recorder,
        plan_sha256=_sha(_canonical(wrong_identity)),
    )

    with PolymarketPaperBroker(database, run_id=run_id) as broker:
        coordinator = PolymarketPaperCoordinator(
            broker,
            control_path=tmp_path / "wrong-recorder.control.json",
        )
        with pytest.raises(ValueError, match="recorder report identity drifted"):
            run_polymarket_paper_plan(broker, coordinator, wrong_recorder)
        assert (
            broker.store.connect()
            .execute(
                "SELECT count(*) FROM paper_order_intent WHERE venue = 'polymarket'"
            )
            .fetchone()[0]
            == 0
        )

    with PolymarketPaperBroker(database, run_id=run_id) as broker:
        coordinator = PolymarketPaperCoordinator(
            broker,
            control_path=tmp_path / "verified-model.control.json",
        )
        report = run_polymarket_paper_plan(broker, coordinator, plan)
        positions = broker.positions()

    assert report.successful is True
    assert report.status == "COMPLETED"
    assert report.matched_execution_count == report.planned_trade_count == 1
    assert report.filled_order_count == report.settled_position_count == 1
    assert report.realized_pnl_quote == research.net_realized_pnl_quote
    assert report.final_control_state == "PAUSED"
    assert positions == ()


def test_model_plan_unknown_execution_remains_visible_and_stopping(tmp_path) -> None:
    database = tmp_path / "unknown-model-plan.duckdb"
    run_id = "unknown-model-plan-run"
    plan, research = _single_trade_model_plan(
        database,
        run_id=run_id,
        maximum_execution_observation_delay_ms=1,
    )
    assert research.trades[0].execution_state == "UNKNOWN"
    assert plan.research_override is True

    with PolymarketPaperBroker(
        database,
        run_id=run_id,
        maximum_execution_observation_delay_ms=1,
    ) as broker:
        coordinator = PolymarketPaperCoordinator(
            broker,
            control_path=tmp_path / "unknown-model.control.json",
        )
        report = run_polymarket_paper_plan(broker, coordinator, plan)
        reconciliation = broker.reconcile()

    assert report.successful is False
    assert report.status == "STOPPING"
    assert report.matched_execution_count == report.planned_trade_count == 1
    assert report.filled_order_count == report.settled_position_count == 0
    assert report.final_control_state == "STOPPING"
    assert any(error.startswith("blocking_intent:") for error in report.errors)
    assert reconciliation.can_open is False
    assert reconciliation.journal.blocking_intent_ids


def test_polymarket_broker_fails_closed_after_execution_observation_timeout(
    tmp_path,
) -> None:
    database = tmp_path / "observation-timeout.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "observation-timeout-run")

    with PolymarketPaperBroker(
        database,
        run_id="observation-timeout-run",
        maximum_execution_observation_delay_ms=1,
    ) as broker:
        coordinator = PolymarketPaperCoordinator(
            broker,
            control_path=tmp_path / "observation-timeout-control.json",
        )
        coordinator.resume()
        coordinator.require_open_allowed()
        decision = broker.replay.books[0]
        position, execution = broker.open_position(
            position_id="observation-timeout-position",
            decision=decision,
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
            order_type="FOK",
        )
        context = (
            broker.store.connect()
            .execute(
                """
            SELECT requested_latency_ms, effective_latency_ms,
                   maximum_execution_observation_delay_ms, execution_event_id
            FROM polymarket_paper_order_context
            """
            )
            .fetchone()
        )
        reconciliation = broker.reconcile()
        stop_report = coordinator.stop_all_positions(submission_latency_ms=5)
        control = coordinator.status()

    assert position is None
    assert execution.state == "UNKNOWN"
    assert context == (5, 5, 1, "")
    assert reconciliation.can_open is False
    assert stop_report.status == "STOPPING"
    assert control["state"] == "STOPPING"


def test_polymarket_broker_blocks_time_travel_and_context_tampering(tmp_path) -> None:
    database = tmp_path / "chronology.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "chronology-run")

    with PolymarketPaperBroker(database, run_id="chronology-run") as broker:
        full = broker.replay.books[0]
        position, _result = broker.open_position(
            position_id="position-1",
            decision=full,
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        assert position is not None
        with pytest.raises(ValueError, match="previously consumed replay state"):
            broker.close_position(
                opening_intent_id=position.opening_intent_id,
                decision=full,
                minimum_price="0.49",
                submission_latency_ms=5,
            )
        broker.store.connect().execute(
            """
            UPDATE polymarket_paper_order_context SET context_json = '{}'
            WHERE intent_id = ?
            """,
            [position.opening_intent_id],
        )
        report = broker.reconcile()

    assert report.can_open is False
    assert report.can_close is False
    assert any("payload_mismatch" in error for error in report.context_errors)


def test_polymarket_broker_missing_post_latency_state_becomes_restart_blocking_unknown(
    tmp_path,
) -> None:
    database = tmp_path / "unknown.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "unknown-run")

    with PolymarketPaperBroker(database, run_id="unknown-run") as broker:
        final_book = broker.replay.books[-1]
        position, result = broker.open_position(
            position_id="position-unknown",
            decision=final_book,
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        report = broker.reconcile()

    assert position is None
    assert result.state == "UNKNOWN"
    assert report.can_open is False
    assert report.can_close is True
    assert len(report.journal.blocking_intent_ids) == 1

    with PolymarketPaperBroker(database, run_id="unknown-run") as restarted:
        restarted_report = restarted.reconcile()
        stop_report = restarted.stop_all_positions(submission_latency_ms=5)
        assert restarted_report.can_open is False
        assert restarted_report.journal.blocking_intent_ids == (
            report.journal.blocking_intent_ids
        )
        assert stop_report.status == "STOPPING"
        assert stop_report.stopped is False
        assert stop_report.blocking_intent_ids == report.journal.blocking_intent_ids


def test_polymarket_broker_stop_settles_only_bot_owned_official_inventory(
    tmp_path,
) -> None:
    database = tmp_path / "stop.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "stop-run")

    with PolymarketPaperBroker(database, run_id="stop-run") as broker:
        position, opened = broker.open_position(
            position_id="position-stop",
            decision=broker.replay.books[0],
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        assert position is not None
        assert opened.state == "FILLED"

        stop_report = broker.stop_all_positions(submission_latency_ms=5)
        reconciliation = broker.reconcile()

    assert stop_report.status == "STOPPED"
    assert stop_report.stopped is True
    assert stop_report.settlement_count == 1
    assert stop_report.close_fill_count == 0
    assert stop_report.remaining_opening_intent_ids == ()
    assert stop_report.blocking_intent_ids == ()
    assert stop_report.errors == ()
    assert reconciliation.journal.inventory[0].remaining_quantity == 0
    assert reconciliation.can_open is True


def test_polymarket_broker_settles_only_from_exact_official_resolution(
    tmp_path,
) -> None:
    database = tmp_path / "settlement.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "settlement-run")

    with PolymarketPaperBroker(database, run_id="settlement-run") as broker:
        position, _result = broker.open_position(
            position_id="position-settlement",
            decision=broker.replay.books[0],
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        assert position is not None
        resolution = broker.replay.resolutions[0]
        forged = replace(resolution, winning_asset_id="not-a-token")
        with pytest.raises(ValueError, match="not immutable evidence"):
            broker.settle_position(
                opening_intent_id=position.opening_intent_id,
                resolution=forged,
            )

        settlement = broker.settle_position(
            opening_intent_id=position.opening_intent_id,
            resolution=resolution,
        )
        report = broker.reconcile()

    assert settlement.payout_per_unit == 1
    assert settlement.gross_payout_quote == Decimal("5")
    assert settlement.entry_cost_quote == Decimal("2.50")
    assert settlement.entry_fee_quote == Decimal("0.08750")
    assert settlement.realized_pnl_quote == Decimal("2.41250")
    assert report.can_open is True
    assert report.can_close is True
    assert report.context_errors == ()


def test_settled_historical_run_remains_reconcilable_in_later_run(
    tmp_path,
) -> None:
    database = tmp_path / "multiple-runs.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "historical-run")

    with PolymarketPaperBroker(database, run_id="historical-run") as broker:
        position, opened = broker.open_position(
            position_id="historical-position",
            decision=broker.replay.books[0],
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        assert position is not None
        assert opened.state == "FILLED"
        broker.settle_position(
            opening_intent_id=position.opening_intent_id,
            resolution=broker.replay.resolutions[0],
        )

    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "later-run")

    with PolymarketPaperBroker(database, run_id="later-run") as broker:
        reconciliation = broker.reconcile()
        assert reconciliation.ok is True
        assert reconciliation.can_open is True
        assert reconciliation.can_close is True
        assert broker.positions() == ()


def test_active_historical_run_blocks_later_run(tmp_path) -> None:
    database = tmp_path / "active-prior-run.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "active-run")

    with PolymarketPaperBroker(database, run_id="active-run") as broker:
        position, opened = broker.open_position(
            position_id="active-position",
            decision=broker.replay.books[0],
            outcome="Up",
            quantity="5",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        assert position is not None
        assert opened.state == "FILLED"

    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "incompatible-run")

    with PolymarketPaperBroker(database, run_id="incompatible-run") as broker:
        reconciliation = broker.reconcile()
        assert reconciliation.ok is False
        assert reconciliation.can_open is False
        assert reconciliation.can_close is False
        assert any(
            error.startswith("active_paper_context_run_mismatch:")
            for error in reconciliation.context_errors
        )


def test_partial_close_dust_remains_owned_until_official_settlement(tmp_path) -> None:
    database = tmp_path / "partial-settlement.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "partial-settlement-run")

    with PolymarketPaperBroker(
        database,
        run_id="partial-settlement-run",
    ) as broker:
        _full, changed, post_tick, _close_book = broker.replay.books
        position, opened = broker.open_position(
            position_id="position-partial",
            decision=broker.replay.books[0],
            outcome="Up",
            quantity="8",
            maximum_price="0.50",
            submission_latency_ms=5,
        )
        assert position is not None and opened.state == "FILLED"
        assert position.execution_event_id == changed.event_id

        closed, close_result = broker.close_position(
            opening_intent_id=position.opening_intent_id,
            decision=post_tick,
            minimum_price="0.490",
            submission_latency_ms=5,
        )
        assert closed is not None
        assert close_result.state == "CLOSE_PENDING"
        remaining = broker.positions()[0]
        assert remaining.remaining_quantity == Decimal("3")
        assert remaining.remaining_entry_fee_quote == Decimal("0.05250")
        assert broker.reconcile().can_open is False

        settlement = broker.settle_position(
            opening_intent_id=position.opening_intent_id,
            resolution=broker.replay.resolutions[0],
        )
        final = broker.reconcile()

    assert settlement.quantity == 3
    assert settlement.entry_fee_quote == Decimal("0.05250")
    assert settlement.realized_pnl_quote == Decimal("1.44750")
    assert final.journal.inventory[0].remaining_quantity == 0
    assert final.journal.blocking_intent_ids == ()
    assert final.can_open is True


def test_polymarket_paper_cli_and_generated_windows_contract_share_actions(
    tmp_path,
    capsys,
) -> None:
    database = tmp_path / "cli-paper.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, "cli-run")
        replay = PolymarketEvidenceReplay.load(store, run_id="cli-run")
        decision_event_id = replay.books[0].event_id

    status_code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--run-id",
            "cli-run",
            "--json",
        ]
    )
    status_payload = json.loads(capsys.readouterr().out)
    assert status_code == 0
    assert status_payload["reconciliation"]["can_open"] is True
    assert status_payload["control"]["state"] == "STOPPED"
    assert (
        status_payload["replay_diagnostics"]["schema_version"]
        == "polymarket-replay-diagnostics-v3"
    )
    assert status_payload["feed_coverage"]["shadow_ready"] is False
    assert status_payload["feed_coverage"]["training_ready"] is False

    resume_code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--run-id",
            "cli-run",
            "--action",
            "resume",
            "--json",
        ]
    )
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_code == 0
    assert resume_payload["control"]["state"] == "RUNNING"

    pause_code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--run-id",
            "cli-run",
            "--action",
            "pause",
            "--json",
        ]
    )
    pause_payload = json.loads(capsys.readouterr().out)
    assert pause_code == 0
    assert pause_payload["control"]["state"] == "PAUSED"
    assert (
        cli.main(
            [
                "polymarket-paper",
                "--database",
                str(database),
                "--run-id",
                "cli-run",
                "--action",
                "open",
            ]
        )
        == 2
    )
    assert "control blocks open while PAUSED" in capsys.readouterr().err
    assert (
        cli.main(
            [
                "polymarket-paper",
                "--database",
                str(database),
                "--run-id",
                "cli-run",
                "--action",
                "resume",
            ]
        )
        == 0
    )
    capsys.readouterr()

    missing_latency_code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--run-id",
            "cli-run",
            "--action",
            "open",
            "--event-id",
            decision_event_id,
            "--position-id",
            "missing-latency",
            "--outcome",
            "Up",
            "--quantity",
            "5",
            "--limit-price",
            "0.50",
        ]
    )
    assert missing_latency_code == 2
    assert "--latency-ms is required" in capsys.readouterr().err

    open_code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--run-id",
            "cli-run",
            "--action",
            "open",
            "--event-id",
            decision_event_id,
            "--position-id",
            "cli-position",
            "--outcome",
            "Up",
            "--quantity",
            "5",
            "--limit-price",
            "0.50",
            "--latency-ms",
            "5",
            "--json",
        ]
    )
    open_payload = json.loads(capsys.readouterr().out)
    spec = next(spec for spec in command_specs() if spec.name == "polymarket-paper")

    assert open_code == 0
    assert open_payload["operation"]["execution"]["state"] == "FILLED"
    assert len(open_payload["positions"]) == 1
    stop_code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--run-id",
            "cli-run",
            "--action",
            "stop",
            "--latency-ms",
            "5",
            "--json",
        ]
    )
    stop_payload = json.loads(capsys.readouterr().out)
    action_option = next(option for option in spec.options if option.dest == "action")

    assert stop_code == 0
    assert stop_payload["operation"]["stop"]["status"] == "STOPPED"
    assert stop_payload["operation"]["stop"]["settlement_count"] == 1
    assert stop_payload["positions"] == []
    assert stop_payload["control"]["state"] == "STOPPED"
    assert set(action_option.choices) == {
        "status",
        "resume",
        "pause",
        "open",
        "close",
        "settle",
        "stop",
        "run-model",
    }
    assert {option.dest for option in spec.options} == {
        "database",
        "run_id",
        "action",
        "control_path",
        "event_id",
        "position_id",
        "opening_intent_id",
        "outcome",
        "quantity",
        "limit_price",
        "latency_ms",
        "artifact",
        "source_verification",
        "policy",
        "allow_unconfirmed_research",
        "output",
        "max_execution_observation_delay_ms",
        "decision_delay_ms",
        "order_type",
        "allow_segmented_gaps",
        "memory_limit",
        "database_threads",
        "json",
    }


def test_polymarket_model_paper_cli_uses_artifact_configuration_and_atomic_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    database = tmp_path / "model-cli.duckdb"
    run_id = "model-cli-run"
    with PolymarketEvidenceStore(database) as store:
        _finish_replay_store(store, run_id)
        recorder_report_sha256 = str(
            store.connect()
            .execute(
                "SELECT report_sha256 FROM polymarket_recorder_run WHERE run_id = ?",
                [run_id],
            )
            .fetchone()[0]
        )
    provisional = PolymarketPaperPlan(
        schema_version=POLYMARKET_PAPER_PLAN_SCHEMA_VERSION,
        artifact_sha256="6" * 64,
        source_verification_sha256="7" * 64,
        recorder_report_sha256=recorder_report_sha256,
        run_id=run_id,
        allow_segmented_gaps=False,
        policy="model",
        primary_network_latency_ms=250,
        confirmed_for_paper_run=True,
        research_override=False,
        blocking_reasons=(),
        execution_report_sha256="8" * 64,
        execution_config={
            "submission_latency_ms": 250,
            "maximum_execution_observation_delay_ms": 750,
            "maximum_book_age_ms": 1_500,
            "order_ttl_ms": 20_000,
        },
        trades=(),
        plan_sha256="",
    )
    identity = provisional.asdict()
    identity.pop("plan_sha256")
    plan = replace(provisional, plan_sha256=_sha(_canonical(identity)))

    class SuccessfulRun:
        successful = True

        @staticmethod
        def asdict() -> dict[str, object]:
            return {
                "status": "COMPLETED",
                "policy": "model",
                "planned_trade_count": 0,
                "matched_execution_count": 0,
                "realized_pnl_quote": "0",
                "report_sha256": "9" * 64,
            }

    def build_plan(*_args, **kwargs):
        assert kwargs == {
            "policy": "model",
            "allow_unconfirmed_research": False,
        }
        return plan

    def run_plan(broker, coordinator, supplied_plan):
        assert supplied_plan == plan
        assert broker.maximum_execution_observation_delay_ms == 750
        assert broker.maximum_book_age_ms == 1_500
        assert broker.order_ttl_ms == 20_000
        coordinator.resume()
        coordinator.pause()
        return SuccessfulRun()

    monkeypatch.setattr(cli, "build_polymarket_paper_plan", build_plan)
    monkeypatch.setattr(cli, "run_polymarket_paper_plan", run_plan)
    output = tmp_path / "model-paper-report.json"
    code = cli.main(
        [
            "polymarket-paper",
            "--database",
            str(database),
            "--action",
            "run-model",
            "--artifact",
            str(tmp_path / "artifact.json"),
            "--source-verification",
            str(tmp_path / "verification.json"),
            "--policy",
            "model",
            "--control-path",
            str(tmp_path / "model-cli.control.json"),
            "--output",
            str(output),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["operation"]["model_run"]["status"] == "COMPLETED"
    assert payload["control"]["state"] == "PAUSED"
    assert persisted["plan"]["primary_network_latency_ms"] == 250
    assert persisted["model_run"]["report_sha256"] == "9" * 64
