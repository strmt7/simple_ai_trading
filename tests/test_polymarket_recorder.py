from __future__ import annotations

from decimal import Decimal
import hashlib
import json

import pytest

from simple_ai_trading import cli
from simple_ai_trading.command_contract import command_specs
from simple_ai_trading.polymarket import parse_polymarket_five_minute_market
from simple_ai_trading.polymarket_recorder import (
    MarketEvidence,
    PolymarketEvidenceStore,
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
        errors = store.integrity_errors("run-tampered")

    assert any(error.startswith("raw_message_hash_mismatch:") for error in errors)
    assert any(error.startswith("event_hash_mismatch:") for error in errors)
    assert any(error.startswith("snapshot_payload_mismatch:") for error in errors)
    assert any(error.startswith("stream_gap_id_mismatch:") for error in errors)


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
