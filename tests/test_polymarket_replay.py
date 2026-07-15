from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json

import pytest

from simple_ai_trading import cli
from simple_ai_trading.command_contract import command_specs
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_paper import PolymarketPaperBroker
from simple_ai_trading.polymarket_features import (
    POLYMARKET_FEATURE_NAMES,
    PolymarketFeatureConfig,
    build_polymarket_feature_dataset,
    materialize_polymarket_feature_dataset,
)
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
    StreamGap,
)
from simple_ai_trading.polymarket_resolution import PolymarketResolutionFinalizer
from simple_ai_trading.polymarket_replay import PolymarketEvidenceReplay


EPOCH = 1_784_058_600


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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


def _book_payload(token: str, condition: str, source_offset_ms: int) -> dict[str, object]:
    return {
        "event_type": "book",
        "market": condition,
        "asset_id": token,
        "timestamp": str(EPOCH * 1_000 + source_offset_ms),
        "hash": f"book-{token[-4:]}-{source_offset_ms}",
        "bids": [{"price": "0.49", "size": "10"}],
        "asks": [{"price": "0.51", "size": "10"}],
    }


def _finish_segmented_store(
    store: PolymarketEvidenceStore,
    run_id: str,
    *,
    gap_stream: str = "clob_market",
    gap_last_sequence: int = 3,
    second_segment_has_baseline: bool = True,
) -> None:
    store.start_run(run_id, EPOCH * 1_000)
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    first = "clob-segment-one"
    second = "clob-segment-two"
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
                _book_payload(btc.up_token_id, btc.condition_id, 3_000),
                connection_id=second,
                sequence=1,
                wall_offset_ms=3_001,
                monotonic_ns=3_001_000_000,
            ),
            _segmented_message(
                _book_payload(btc.down_token_id, btc.condition_id, 3_000),
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
    feature_evidence: bool = False,
    pre_window_trade_quantity: str = "0.1",
    duplicate_tick_transition: bool = False,
    terminal_clear_burst: bool = False,
    interleaved_best_transitions: bool = False,
    compact_resolution_event: bool = False,
    finalize_official: bool = True,
) -> None:
    store.start_run(run_id, EPOCH * 1_000)
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
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
                    "timestamp": str(EPOCH * 1_000 + 1_012),
                },
                sequence=50,
                wall_offset_ms=1_014,
                monotonic_ns=1_013_000_000,
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
        clob_messages.insert(
            4,
            _message(
                "clob_market",
                {
                    "event_type": "book",
                    "market": btc.condition_id,
                    "asset_id": token,
                    "timestamp": str(
                        EPOCH * 1_000 + 1_011 - trade_resync_lag_ms
                    ),
                    "hash": "atomic-replacement",
                    "bids": [],
                    "asks": [
                        {"price": "0.50", "size": "8"},
                        {"price": "0.52", "size": "7"},
                    ],
                },
                sequence=40,
                wall_offset_ms=1_012,
                monotonic_ns=1_011_500_000,
            ),
        )
        clob_messages.insert(
            4,
            _message(
                "clob_market",
                {
                    "event_type": "price_change",
                    "market": btc.condition_id,
                    "timestamp": str(EPOCH * 1_000 + 1_011),
                    "price_changes": [
                        {
                            "asset_id": token,
                            "price": "0.53",
                            "size": "6",
                            "side": "SELL",
                            "hash": "post-trade-delta",
                            "best_bid": "0",
                            "best_ask": "0.50",
                        }
                    ],
                },
                sequence=41,
                wall_offset_ms=1_012,
                monotonic_ns=1_011_200_000,
            ),
        )
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
    store.append_messages(run_id, [*clob_messages, *auxiliary])
    report = store.finish_run(
        run_id,
        started_at_ms=EPOCH * 1_000,
        ended_at_ms=EPOCH * 1_000 + 302_000,
        database=str(store.path),
        errors=(),
    )
    assert report.status == "complete"
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
    assert replay.first_book_after_latency(full, latency_ms=5) == changed
    assert replay.first_book_after_latency(post_tick, latency_ms=1) == close_book
    assert replay.first_book_after_latency(close_book, latency_ms=1) is None
    assert replay.book_for_event(changed.event_id, changed.token_id) == changed
    assert replay.resolutions[0].winning_outcome == "Up"
    assert replay.resolutions[0].source == "clob_gamma_crosscheck"


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
        == [Decimal("0.50"), Decimal("0.52")]
    )
    resynchronized = replay.books[resync_index].snapshot
    post_resync = replay.books[resync_index + 1].snapshot
    assert resynchronized.bids == ()
    assert resynchronized.asks[0].price == Decimal("0.50")
    assert resynchronized.asks[0].quantity == Decimal("8")
    assert resynchronized.asks[1].price == Decimal("0.52")
    assert resynchronized.asks[1].quantity == Decimal("7")
    assert post_resync.asks[1].price == Decimal("0.52")
    assert post_resync.bids[0].price == Decimal("0.499")
    assert post_resync.received_monotonic_ns > resynchronized.received_monotonic_ns
    assert replay.diagnostics.late_event_count >= 1
    assert replay.diagnostics.maximum_source_regression_ms == 1
    assert replay.diagnostics.deferred_event_count == 0
    assert replay.diagnostics.maximum_availability_delay_ns == 0


def test_replay_binds_exact_duplicate_tick_transition_idempotently(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "duplicate-tick.duckdb") as store:
        _finish_replay_store(
            store,
            "duplicate-tick",
            duplicate_tick_transition=True,
        )
        replay = PolymarketEvidenceReplay.load(store, run_id="duplicate-tick")

    assert replay.books[2].tick_size == Decimal("0.001")


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
        second = build_polymarket_feature_dataset(
            store,
            run_id="features",
            config=config,
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
    assert created.status == "created"
    assert existing.status == "existing"
    assert created.row_count == existing.row_count == len(first.rows)
    assert len(first.rows) >= 1
    row = first.rows[0]
    assert len(row.feature_values) == len(POLYMARKET_FEATURE_NAMES) == 46
    assert row.official_up is True
    assert row.resolution_event_id
    assert row.feature_map()["ask_pair_cost"] == pytest.approx(1.02)
    assert row.feature_map()["chainlink_anchor_gap_ms"] == 0.0
    assert first.labeled_market_counts["BTC"] == 1
    assert first.training_ready is False
    assert "insufficient_featured_resolved_markets:ETH:0/1" in first.training_errors


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
        first.rows[0].input_provenance_sha256
        != second.rows[0].input_provenance_sha256
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
    assert replay.first_book_after_latency(old_up, latency_ms=2_000) is None
    for segment_id in segment_ids:
        assert {book.outcome for book in replay.books if book.segment_id == segment_id} == {
            "Up",
            "Down",
        }


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
def test_segmented_replay_never_permits_non_clob_gaps(
    tmp_path,
    gap_stream: str,
) -> None:
    with PolymarketEvidenceStore(tmp_path / f"{gap_stream}.duckdb") as store:
        _finish_segmented_store(store, "non-clob-gap", gap_stream=gap_stream)
        with pytest.raises(ValueError, match="only CLOB market gaps"):
            PolymarketEvidenceReplay.load(
                store,
                run_id="non-clob-gap",
                allow_segmented_gaps=True,
            )


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
        assert restarted_report.can_open is False
        assert restarted_report.journal.blocking_intent_ids == (
            report.journal.blocking_intent_ids
        )


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
    assert (
        status_payload["replay_diagnostics"]["schema_version"]
        == "polymarket-replay-diagnostics-v2"
    )
    assert status_payload["feed_coverage"]["shadow_ready"] is False
    assert status_payload["feed_coverage"]["training_ready"] is False

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
    assert {option.dest for option in spec.options} == {
        "database",
        "run_id",
        "action",
        "event_id",
        "position_id",
        "opening_intent_id",
        "outcome",
        "quantity",
        "limit_price",
        "latency_ms",
        "allow_segmented_gaps",
        "memory_limit",
        "database_threads",
        "json",
    }
