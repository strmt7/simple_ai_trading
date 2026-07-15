from __future__ import annotations

import asyncio
from decimal import Decimal
import hashlib
import json
import threading

import pytest

from simple_ai_trading import cli
from simple_ai_trading import polymarket_recorder as recorder_module
from simple_ai_trading.command_contract import command_specs
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_coverage import inspect_polymarket_feed_coverage
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
    PolymarketPublicRecorder,
    RawStreamMessage,
    RecorderReport,
    StreamGap,
)


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
        observed_wall_ms=EPOCH * 1_000 + 1_000,
        observed_monotonic_ns=123_000,
        clob_info_json=clob,
        clob_info_sha256=_sha(clob),
        up_fee_rate_json=fee,
        up_fee_rate_sha256=_sha(fee),
        down_fee_rate_json=fee,
        down_fee_rate_sha256=_sha(fee),
        maker_base_fee=1000,
        taker_base_fee=1000,
        taker_order_delay_enabled=True,
        minimum_order_age_seconds=0,
    )


def _message(stream: str, raw: object | str, *, sequence: int = 1) -> RawStreamMessage:
    raw_text = raw if isinstance(raw, str) else _canonical(raw)
    return RawStreamMessage(
        stream=stream,
        connection_id=f"{stream}-connection",
        sequence_number=sequence,
        received_wall_ms=EPOCH * 1_000 + sequence,
        received_monotonic_ns=456_000 + sequence,
        raw_text=raw_text,
    )


def _complete_store(store: PolymarketEvidenceStore, run_id: str) -> None:
    store.start_run(run_id, EPOCH * 1_000)
    for asset in ("BTC", "ETH", "SOL"):
        store.record_market_evidence(run_id, _evidence(asset))
    store.append_messages(
        run_id,
        [
            _message(
                "clob_market",
                [
                    {
                        "event_type": "book",
                        "market": "0x" + "7" * 64,
                        "asset_id": "7" * 40,
                        "timestamp": EPOCH * 1_000,
                    }
                ],
            ),
            _message(
                "polymarket_rtds",
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "timestamp": EPOCH * 1_000,
                    "payload": {
                        "symbol": "btcusdt",
                        "timestamp": EPOCH * 1_000,
                        "value": Decimal("2.5").to_eng_string(),
                    },
                },
            ),
            _message(
                "binance_spot",
                {
                    "stream": "btcusdt@trade",
                    "data": {
                        "e": "trade",
                        "E": EPOCH * 1_000,
                        "T": EPOCH * 1_000 - 1,
                    },
                },
            ),
            _message("polymarket_rtds", "PING", sequence=2),
        ],
    )


def test_evidence_store_round_trip_has_complete_coverage_and_event_indexes(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "paper.duckdb") as store:
        _complete_store(store, "run-complete")
        report = store.finish_run(
            "run-complete",
            started_at_ms=EPOCH * 1_000,
            ended_at_ms=EPOCH * 1_000 + 5_000,
            database=str(tmp_path / "paper.duckdb"),
            errors=(),
        )
        events = (
            store.connect()
            .execute(
                """
            SELECT stream, event_type, symbol FROM polymarket_public_event
            WHERE run_id = ? ORDER BY stream
            """,
                ["run-complete"],
            )
            .fetchall()
        )
        post_finish_integrity = store.integrity_errors("run-complete")

    assert report.status == "complete"
    assert report.assets == ("BTC", "ETH", "SOL")
    assert report.market_snapshot_count == 3
    assert report.raw_message_count == 4
    assert report.normalized_event_count == 3
    assert report.integrity_errors == ()
    assert post_finish_integrity == ()
    assert events == [
        ("binance_spot", "trade", "BTC"),
        ("clob_market", "book", ""),
        ("polymarket_rtds", "crypto_prices:update", "BTC"),
    ]


def test_evidence_store_chunks_large_message_batches(tmp_path) -> None:
    message_count = 2_050
    messages = [
        _message("polymarket_rtds", "PING", sequence=sequence)
        for sequence in range(1, message_count + 1)
    ]
    with PolymarketEvidenceStore(
        tmp_path / "bounded-batch.duckdb",
        memory_limit="512MB",
        threads=1,
    ) as store:
        store.start_run("bounded-batch", EPOCH * 1_000)
        store.append_messages("bounded-batch", messages)
        stats = store.connect().execute(
            """
            SELECT count(*), min(sequence_number), max(sequence_number),
                   count(DISTINCT sequence_number)
            FROM polymarket_raw_message
            """
        ).fetchone()

    assert stats == (message_count, 1, message_count, message_count)


def test_feed_coverage_requires_real_per_asset_sources_and_resolutions(
    tmp_path,
) -> None:
    database = tmp_path / "feed-coverage.duckdb"
    with PolymarketEvidenceStore(database) as store:
        run_id = "feed-coverage"
        store.start_run(run_id, EPOCH * 1_000)
        markets = {}
        for asset in ("BTC", "ETH", "SOL"):
            evidence = _evidence(asset)
            markets[asset] = evidence.market
            store.record_market_evidence(run_id, evidence)

        clob_events = []
        for asset, market in markets.items():
            for token in market.token_ids:
                clob_events.append(
                    {
                        "event_type": "book",
                        "market": market.condition_id,
                        "asset_id": token,
                        "timestamp": str(EPOCH * 1_000 + 1_000),
                        "hash": f"{asset}-{token[-4:]}",
                        "bids": [{"price": "0.49", "size": "10"}],
                        "asks": [{"price": "0.51", "size": "10"}],
                    }
                )
            clob_events.append(
                {
                    "event_type": "market_resolved",
                    "market": market.condition_id,
                    "winning_asset_id": market.up_token_id,
                    "winning_outcome": "Up",
                    "timestamp": str(market.end_ms + 1_000),
                }
            )
        messages = [_message("clob_market", clob_events)]
        for index, asset in enumerate(("BTC", "ETH", "SOL"), start=1):
            lower = asset.lower()
            messages.extend(
                [
                    _message(
                        "polymarket_rtds",
                        {
                            "topic": "crypto_prices",
                            "type": "subscribe",
                            "timestamp": EPOCH * 1_000 + index,
                            "payload": {
                                "symbol": f"{lower}usdt",
                                "data": [
                                    {
                                        "timestamp": EPOCH * 1_000 - 1_000,
                                        "value": "100",
                                    },
                                    {
                                        "timestamp": EPOCH * 1_000,
                                        "value": "101",
                                    },
                                ],
                            },
                        },
                        sequence=index,
                    ),
                    _message(
                        "polymarket_rtds",
                        {
                            "topic": "crypto_prices_chainlink",
                            "type": "update",
                            "timestamp": EPOCH * 1_000 + index,
                            "payload": {
                                "symbol": f"{lower}/usd",
                                "timestamp": EPOCH * 1_000,
                                "value": "100.5",
                            },
                        },
                        sequence=index + 3,
                    ),
                    _message(
                        "binance_spot",
                        {
                            "stream": f"{lower}usdt@bookTicker",
                            "data": {
                                "u": index,
                                "b": "100",
                                "B": "10",
                                "a": "101",
                                "A": "11",
                            },
                        },
                        sequence=index,
                    ),
                    _message(
                        "binance_spot",
                        {
                            "stream": f"{lower}usdt@trade",
                            "data": {
                                "e": "trade",
                                "E": EPOCH * 1_000 + index,
                                "T": EPOCH * 1_000,
                                "p": "100.5",
                                "q": "2",
                                "m": False,
                            },
                        },
                        sequence=index + 3,
                    ),
                ]
            )
        store.append_messages(run_id, messages)
        report = store.finish_run(
            run_id,
            started_at_ms=EPOCH * 1_000,
            ended_at_ms=EPOCH * 1_000 + 5_000,
            database=str(database),
            errors=(),
        )
        coverage = inspect_polymarket_feed_coverage(
            store,
            run_id=run_id,
            minimum_resolved_markets_per_asset=1,
        )

    assert report.status == "complete"
    assert coverage.shadow_ready is True
    assert coverage.training_ready is True
    assert coverage.shadow_errors == ()
    assert coverage.training_errors == ()
    for asset in ("BTC", "ETH", "SOL"):
        assert coverage.counts[asset] == {
            "market_snapshots": 1,
            "clob_token_baselines": 2,
            "direct_binance_book_tickers": 1,
            "direct_binance_trades": 1,
            "rtds_binance_history_samples": 2,
            "rtds_binance_live_updates": 0,
            "rtds_chainlink_history_samples": 0,
            "rtds_chainlink_live_updates": 1,
            "official_resolutions": 1,
        }


def test_rtds_chainlink_bootstrap_topic_is_canonicalized_from_symbol(tmp_path) -> None:
    with PolymarketEvidenceStore(tmp_path / "chainlink-bootstrap.duckdb") as store:
        store.start_run("chainlink-bootstrap", EPOCH * 1_000)
        store.append_messages(
            "chainlink-bootstrap",
            [
                _message(
                    "polymarket_rtds",
                    {
                        "topic": "crypto_prices",
                        "type": "subscribe",
                        "payload": {
                            "symbol": "btc/usd",
                            "data": [
                                {
                                    "timestamp": EPOCH * 1_000,
                                    "value": "60000",
                                }
                            ],
                        },
                    },
                )
            ],
        )
        indexed = store.connect().execute(
            """
            SELECT event_type, symbol FROM polymarket_public_event
            WHERE run_id = ?
            """,
            ["chainlink-bootstrap"],
        ).fetchone()

    assert indexed == ("crypto_prices_chainlink:subscribe", "BTC")


def test_rtds_uses_independent_json_subscriptions_for_each_crypto_feed(
    tmp_path, monkeypatch
) -> None:
    recorder = PolymarketPublicRecorder(tmp_path / "subscriptions.duckdb")
    captured: list[dict[str, object]] = []

    async def _capture_simple_stream(**options: object) -> None:
        captured.append(options)

    monkeypatch.setattr(recorder, "_simple_stream", _capture_simple_stream)
    asyncio.run(recorder._rtds_stream(asyncio.Queue(), asyncio.Event()))

    assert len(captured) == 6
    assert {call["url"] for call in captured} == {
        "wss://ws-live-data.polymarket.com"
    }
    assert {call["stream"] for call in captured} == {"polymarket_rtds"}
    subscription_messages = [call["subscription"] for call in captured]
    assert all(isinstance(message, str) for message in subscription_messages)
    messages = [json.loads(message) for message in subscription_messages]
    assert all(message["action"] == "subscribe" for message in messages)
    assert all(len(message["subscriptions"]) == 1 for message in messages)
    subscriptions = [message["subscriptions"][0] for message in messages]
    binance_filters = {
        json.loads(subscription["filters"])["symbol"]
        for subscription in subscriptions
        if subscription["topic"] == "crypto_prices"
    }
    chainlink_filters = {
        json.loads(subscription["filters"])["symbol"]
        for subscription in subscriptions
        if subscription["topic"] == "crypto_prices_chainlink"
    }
    assert all(subscription["type"] == "update" for subscription in subscriptions)
    assert binance_filters == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert chainlink_filters == {"btc/usd", "eth/usd", "sol/usd"}


def test_periodic_heartbeat_runs_independently_of_busy_message_processing() -> None:
    class _Sender:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send(self, message: str) -> None:
            self.messages.append(message)
            await asyncio.sleep(0)

    async def _exercise() -> list[str]:
        sender = _Sender()
        stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            recorder_module._periodic_text_heartbeat(sender, stop, "PING", 0.01)
        )
        deadline = asyncio.get_running_loop().time() + 0.045
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0)
        stop.set()
        await heartbeat
        return sender.messages

    assert len(asyncio.run(_exercise())) >= 3


def test_writer_persistence_does_not_block_the_network_event_loop(
    tmp_path,
    monkeypatch,
) -> None:
    recorder = PolymarketPublicRecorder(tmp_path / "writer-offload.duckdb")
    main_thread = threading.get_ident()
    writer_threads: list[int] = []
    started = threading.Event()
    release = threading.Event()

    class _SlowStore:
        def __init__(self, path, *, memory_limit: str, threads: int) -> None:
            self.path = path
            self.memory_limit = memory_limit
            self.threads = threads

        def connect(self) -> object:
            writer_threads.append(threading.get_ident())
            return self

        def append_messages(self, run_id: str, messages: object) -> None:
            assert run_id == "writer-offload"
            assert messages
            writer_threads.append(threading.get_ident())
            started.set()
            assert release.wait(timeout=1.0)

        def close(self) -> None:
            writer_threads.append(threading.get_ident())

    source_store = PolymarketEvidenceStore(
        tmp_path / "writer-offload.duckdb",
        memory_limit="1GB",
        threads=1,
    )
    monkeypatch.setattr(recorder_module, "PolymarketEvidenceStore", _SlowStore)

    async def _exercise() -> bool:
        output: asyncio.Queue[
            RawStreamMessage | StreamGap | MarketEvidence | None
        ] = asyncio.Queue()
        await output.put(_message("clob_market", {"event_type": "book"}))
        await output.put(None)
        writer = asyncio.create_task(
            recorder._writer("writer-offload", source_store, output)
        )
        deadline = asyncio.get_running_loop().time() + 0.5
        while not started.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)
        loop_remained_responsive = started.is_set()
        release.set()
        await asyncio.wait_for(writer, timeout=1.0)
        return loop_remained_responsive

    assert asyncio.run(_exercise()) is True
    assert writer_threads and set(writer_threads).isdisjoint({main_thread})
    assert len(set(writer_threads)) == 1


def test_writer_owned_connection_preserves_final_report_cursor(tmp_path) -> None:
    database = tmp_path / "writer-report.duckdb"
    recorder = PolymarketPublicRecorder(database)

    async def _exercise(store: PolymarketEvidenceStore) -> None:
        output: asyncio.Queue[
            RawStreamMessage | StreamGap | MarketEvidence | None
        ] = asyncio.Queue()
        await output.put(_message("polymarket_rtds", "PING"))
        await output.put(None)
        await recorder._writer("writer-report", store, output)

    with PolymarketEvidenceStore(database) as store:
        store.start_run("writer-report", EPOCH * 1_000)
        asyncio.run(_exercise(store))
        report = store.finish_run(
            "writer-report",
            started_at_ms=EPOCH * 1_000,
            ended_at_ms=EPOCH * 1_000 + 1_000,
            database=str(database),
            errors=(),
        )

    assert report.status == "failed"
    assert report.integrity_errors == ()
    assert "missing_streams:binance_spot,clob_market" in report.errors
    assert report.stream_counts == {"polymarket_rtds": 1}
    assert report.raw_message_count == 1


def test_integrity_verifier_detects_raw_event_snapshot_and_gap_tampering(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "tampered.duckdb") as store:
        _complete_store(store, "run-tampered")
        store.record_gap(
            "run-tampered",
            StreamGap("clob_market", "gap-connection", EPOCH * 1_000, "disconnect", 9),
        )
        connection = store.connect()
        connection.execute(
            """
            UPDATE polymarket_raw_message SET raw_text = raw_text || ' '
            WHERE message_id = (SELECT min(message_id) FROM polymarket_raw_message)
            """
        )
        connection.execute(
            """
            UPDATE polymarket_public_event SET event_json = '{}'
            WHERE event_id = (SELECT min(event_id) FROM polymarket_public_event)
            """
        )
        connection.execute(
            """
            UPDATE polymarket_market_snapshot SET snapshot_payload_json = '{}'
            WHERE snapshot_id = (
                SELECT min(snapshot_id) FROM polymarket_market_snapshot
            )
            """
        )
        connection.execute(
            """
            UPDATE polymarket_stream_gap SET reason = 'changed'
            WHERE gap_id = (SELECT min(gap_id) FROM polymarket_stream_gap)
            """
        )
        connection.execute(
            """
            INSERT INTO polymarket_raw_message
            SELECT message_id, run_id, schema_version, stream, connection_id,
                   sequence_number + 10000, received_wall_ms,
                   received_monotonic_ns, raw_payload_sha256, raw_text,
                   parse_status, parse_error
            FROM polymarket_raw_message
            WHERE message_id = (SELECT min(message_id) FROM polymarket_raw_message)
            """
        )
        connection.execute(
            """
            INSERT INTO polymarket_public_event
            SELECT event_id, run_id, message_id, sub_index + 10000, stream,
                   event_type, symbol, condition_id, asset_id, source_time_ms,
                   publisher_time_ms, event_json, event_sha256
            FROM polymarket_public_event
            WHERE event_id = (SELECT min(event_id) FROM polymarket_public_event)
            """
        )
        errors = store.integrity_errors("run-tampered")

    assert any(error.startswith("raw_message_hash_mismatch:") for error in errors)
    assert any(error.startswith("event_hash_mismatch:") for error in errors)
    assert any(error.startswith("snapshot_payload_mismatch:") for error in errors)
    assert any(error.startswith("stream_gap_id_mismatch:") for error in errors)
    assert any(error.startswith("duplicate_raw_message_id:") for error in errors)
    assert any(error.startswith("duplicate_public_event_id:") for error in errors)


def test_finished_report_binds_evidence_counts_against_valid_late_append(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "closed-run.duckdb") as store:
        _complete_store(store, "closed-run")
        report = store.finish_run(
            "closed-run",
            started_at_ms=EPOCH * 1_000,
            ended_at_ms=EPOCH * 1_000 + 5_000,
            database=str(tmp_path / "closed-run.duckdb"),
            errors=(),
        )
        assert report.status == "complete"
        store.append_messages(
            "closed-run",
            [
                _message(
                    "polymarket_rtds",
                    {
                        "topic": "crypto_prices",
                        "type": "update",
                        "timestamp": EPOCH * 1_000 + 3,
                        "payload": {
                            "symbol": "ethusdt",
                            "timestamp": EPOCH * 1_000 + 2,
                            "value": "2500",
                        },
                    },
                    sequence=3,
                )
            ],
        )
        errors = store.integrity_errors("closed-run")

    assert "recorder_report_evidence_mismatch:closed-run:raw_message_count" in errors
    assert (
        "recorder_report_evidence_mismatch:closed-run:normalized_event_count"
        in errors
    )


def test_invalid_stream_payload_fails_run_and_gap_validation_is_fail_closed(
    tmp_path,
) -> None:
    with PolymarketEvidenceStore(tmp_path / "invalid.duckdb") as store:
        _complete_store(store, "run-invalid")
        store.append_messages(
            "run-invalid",
            [_message("clob_market", "{not-json", sequence=2)],
        )
        with pytest.raises(ValueError, match="unsupported public stream"):
            store.record_gap(
                "run-invalid",
                StreamGap("unknown", "connection", EPOCH * 1_000, "disconnect", 0),
            )
        report = store.finish_run(
            "run-invalid",
            started_at_ms=EPOCH * 1_000,
            ended_at_ms=EPOCH * 1_000 + 5_000,
            database=str(tmp_path / "invalid.duckdb"),
            errors=(),
        )

    assert report.status == "failed"
    assert any(
        error.startswith("invalid_stream_message:") for error in report.integrity_errors
    )


def test_evidence_hash_mismatch_is_rejected_before_persistence(tmp_path) -> None:
    evidence = _evidence("BTC")
    corrupted = MarketEvidence(**{**evidence.__dict__, "up_fee_rate_sha256": "0" * 64})
    with PolymarketEvidenceStore(tmp_path / "hash.duckdb") as store:
        store.start_run("run-hash", EPOCH * 1_000)
        with pytest.raises(ValueError, match="Up fee-rate payload hash mismatch"):
            store.record_market_evidence("run-hash", corrupted)
        count = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_market_snapshot")
            .fetchone()[0]
        )

    assert count == 0


def test_finished_report_hash_is_verified_on_reopen(tmp_path) -> None:
    database = tmp_path / "report.duckdb"
    with PolymarketEvidenceStore(database) as store:
        _complete_store(store, "run-report")
        store.finish_run(
            "run-report",
            started_at_ms=EPOCH * 1_000,
            ended_at_ms=EPOCH * 1_000 + 5_000,
            database=str(database),
            errors=(),
        )
    with PolymarketEvidenceStore(database) as store:
        assert store.integrity_errors("run-report") == ()
        store.connect().execute(
            """
            UPDATE polymarket_recorder_run SET report_json = '{}'
            WHERE run_id = 'run-report'
            """
        )
        errors = store.integrity_errors("run-report")

    assert "recorder_report_hash_mismatch:run-report" in errors
    assert "recorder_report_embedded_hash_mismatch:run-report" in errors


def test_polymarket_record_is_generated_from_cli_contract_and_runs(
    monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    class _Recorder:
        def __init__(self, database, **options: object) -> None:
            captured["database"] = database
            captured.update(options)

        async def run(self, *, duration_seconds: int) -> RecorderReport:
            captured["duration_seconds"] = duration_seconds
            return RecorderReport(
                schema_version="polymarket-public-recorder-v1",
                run_id="run-cli",
                status="complete",
                database=str(captured["database"]),
                started_at_ms=1_000,
                ended_at_ms=6_000,
                duration_seconds=5.0,
                market_snapshot_count=3,
                raw_message_count=9,
                normalized_event_count=8,
                stream_gap_count=0,
                stream_counts={
                    "binance_spot": 3,
                    "clob_market": 3,
                    "polymarket_rtds": 3,
                },
                assets=("BTC", "ETH", "SOL"),
                conditions=("condition-1", "condition-2", "condition-3"),
                integrity_errors=(),
                errors=(),
                report_sha256="f" * 64,
            )

    monkeypatch.setattr(cli, "PolymarketPublicRecorder", _Recorder)
    status = cli.main(
        [
            "polymarket-record",
            "--database",
            "paper.duckdb",
            "--duration-seconds",
            "5",
            "--queue-capacity",
            "1000",
            "--memory-limit",
            "512MB",
            "--database-threads",
            "1",
            "--json",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    spec = next(spec for spec in command_specs() if spec.name == "polymarket-record")

    assert status == 0
    assert output["status"] == "complete"
    assert str(captured["database"]) == "paper.duckdb"
    assert {key: value for key, value in captured.items() if key != "database"} == {
        "queue_capacity": 1000,
        "discovery_interval_seconds": 60,
        "memory_limit": "512MB",
        "database_threads": 1,
        "duration_seconds": 5,
    }
    assert {option.dest for option in spec.options} == {
        "database",
        "duration_seconds",
        "discovery_interval_seconds",
        "queue_capacity",
        "memory_limit",
        "database_threads",
        "json",
    }
