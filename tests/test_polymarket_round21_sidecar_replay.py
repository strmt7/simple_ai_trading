from __future__ import annotations

import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round21_sidecar_replay as replay_module
from simple_ai_trading.polymarket_recorder import RawStreamMessage, StreamGap
from simple_ai_trading.polymarket_round21_sidecar_replay import (
    replay_round21_optional_binance_features,
)


START_MS = 1_800_000_000_000
END_MS = START_MS + 10_000
RUN_ID = "a" * 32


def _message(
    *,
    stream: str,
    connection: str,
    sequence: int,
    wall_ms: int,
    payload: dict[str, object],
) -> RawStreamMessage:
    return RawStreamMessage(
        stream=stream,
        connection_id=connection,
        sequence_number=sequence,
        received_wall_ms=wall_ms,
        received_monotonic_ns=wall_ms * 1_000_000 + sequence,
        raw_text=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def _spot_book(update_id: int, bid: str) -> dict[str, object]:
    return {
        "stream": "btcusdt@bookTicker",
        "data": {
            "u": update_id,
            "s": "BTCUSDT",
            "b": bid,
            "B": "2",
            "a": str(float(bid) + 1.0),
            "A": "3",
        },
    }


def _futures_book(update_id: int, wall_ms: int) -> dict[str, object]:
    return {
        "stream": "btcusdt@bookTicker",
        "data": {
            "e": "bookTicker",
            "E": wall_ms,
            "T": wall_ms - 1,
            "u": update_id,
            "s": "BTCUSDT",
            "b": "60002",
            "B": "4",
            "a": "60003",
            "A": "5",
        },
    }


def _terminal() -> dict[str, object]:
    return {
        "manifest_sha256": "f" * 64,
        "campaign_start_ms": START_MS,
        "campaign_end_ms": END_MS,
        "segments": [
            {
                "status": "degraded",
                "run_id": RUN_ID,
                "started_at_ms": START_MS + 1_000,
                "ended_at_ms": START_MS + 9_000,
                "preregistration_manifest_sha256": "b" * 64,
                "recorder_report_sha256": "c" * 64,
                "raw_message_count": 3,
                "stream_gap_count": 1,
                "stream_counts": {
                    "binance_futures": 1,
                    "binance_spot": 2,
                },
                "eligible_for_optional_feature_replay": True,
            }
        ],
    }


class _Cursor:
    def __init__(self, row: tuple[object, ...] | None) -> None:
        self.row = row

    def fetchone(self) -> tuple[object, ...] | None:
        return self.row


class _Connection:
    def execute(self, sql: str, _parameters: object) -> _Cursor:
        if "polymarket_recorder_run" in sql:
            return _Cursor(
                (
                    "degraded",
                    START_MS + 1_000,
                    START_MS + 9_000,
                    "c" * 64,
                )
            )
        if "polymarket_preregistration_manifest" in sql:
            return _Cursor(("b" * 64,))
        raise AssertionError(sql)


class _Store:
    def __init__(self, _path: Path, **kwargs: object) -> None:
        assert kwargs == {"read_only": True, "memory_limit": "1GB", "threads": 2}

    def __enter__(self) -> _Store:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def connect(self) -> _Connection:
        return _Connection()

    def iter_terminal_stream_gaps(self, run_id: str):
        assert run_id == RUN_ID
        yield StreamGap(
            stream="binance_spot",
            connection_id=f"binance:spot:btc:{'a' * 32}",
            opened_at_ms=START_MS + 2_500,
            reason="transport_closed",
            last_sequence_number=1,
        )

    def iter_terminal_capture_messages(
        self,
        run_id: str,
        *,
        streams: tuple[str, ...],
    ):
        assert run_id == RUN_ID
        assert streams == ("binance_spot", "binance_futures")
        yield _message(
            stream="binance_spot",
            connection=f"binance:spot:btc:{'a' * 32}",
            sequence=1,
            wall_ms=START_MS + 2_000,
            payload=_spot_book(1, "60000"),
        )
        yield _message(
            stream="binance_futures",
            connection=f"binance:futures:btc:{'b' * 32}",
            sequence=1,
            wall_ms=START_MS + 2_001,
            payload=_futures_book(1, START_MS + 2_001),
        )
        yield _message(
            stream="binance_spot",
            connection=f"binance:spot:btc:{'c' * 32}",
            sequence=1,
            wall_ms=START_MS + 2_600,
            payload=_spot_book(2, "60010"),
        )


def test_sidecar_replay_is_causal_and_invalidates_gaps_and_segment_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "sidecar.duckdb"
    database.touch()
    monkeypatch.setattr(
        replay_module,
        "validate_round21_sidecar_terminal_manifest",
        lambda value: value,
    )
    monkeypatch.setattr(replay_module, "PolymarketEvidenceStore", _Store)

    result = replay_round21_optional_binance_features(
        source_database=database,
        terminal_manifest=_terminal(),
        decision_times_ms=(
            START_MS + 1_900,
            START_MS + 2_100,
            START_MS + 2_550,
            START_MS + 2_700,
            START_MS + 9_000,
        ),
    )

    assert result.raw_message_count == 3
    assert result.stream_gap_count == 1
    assert result.stream_counts == {"binance_futures": 1, "binance_spot": 2}
    assert result.features[0].spot_available is False
    assert result.features[1].usdm_available is True
    assert result.features[2].spot_available is False
    assert result.features[2].usdm_available is False
    assert result.features[3].usdm_available is True
    assert result.features[4].spot_available is False
    assert result.live_trading_authority is False


def test_sidecar_replay_rejects_active_or_mismatched_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "sidecar.duckdb"
    database.touch()
    monkeypatch.setattr(
        replay_module,
        "validate_round21_sidecar_terminal_manifest",
        lambda value: value,
    )
    Path(f"{database}.wal").touch()
    with pytest.raises(RuntimeError, match="WAL-free"):
        replay_round21_optional_binance_features(
            source_database=database,
            terminal_manifest=_terminal(),
            decision_times_ms=(START_MS + 2_100,),
        )

    Path(f"{database}.wal").unlink()
    monkeypatch.setattr(replay_module, "PolymarketEvidenceStore", _Store)
    changed = _terminal()
    changed["segments"][0]["recorder_report_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="database identity differs"):
        replay_round21_optional_binance_features(
            source_database=database,
            terminal_manifest=changed,
            decision_times_ms=(START_MS + 2_100,),
        )
