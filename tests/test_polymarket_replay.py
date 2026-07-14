from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json

import pytest

from simple_ai_trading import cli
from simple_ai_trading.command_contract import command_specs
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_paper import PolymarketPaperBroker
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    RawStreamMessage,
)
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


def _finish_replay_store(
    store: PolymarketEvidenceStore,
    run_id: str,
    *,
    wrong_best: bool = False,
) -> None:
    store.start_run(run_id, EPOCH * 1_000)
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
    btc = parse_polymarket_five_minute_market(_market_payload("BTC"))
    token = btc.up_token_id
    reported_best_ask = "0.49" if wrong_best else "0.50"
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
            {
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
            },
            sequence=8,
            wall_offset_ms=301_000,
            monotonic_ns=301_000_000_000,
        ),
    ]
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
    store.append_messages(run_id, [*clob_messages, *auxiliary])
    report = store.finish_run(
        run_id,
        started_at_ms=EPOCH * 1_000,
        ended_at_ms=EPOCH * 1_000 + 302_000,
        database=str(store.path),
        errors=(),
    )
    assert report.status == "complete"


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
        "memory_limit",
        "database_threads",
        "json",
    }
