from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import simple_ai_trading.microstructure_data as microstructure


def _stream_line(receive_ns: int, stream: str, data: dict[str, object]) -> str:
    return f"{receive_ns} {json.dumps({'stream': stream, 'data': data}, separators=(',', ':'))}\n"


def test_exchange_filters_require_real_tick_and_lot_sizes() -> None:
    payload = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                ],
            }
        ]
    }

    assert microstructure._exchange_filters(payload, ("BTCUSDT",)) == {"BTCUSDT": (0.1, 0.001)}
    with pytest.raises(ValueError, match="missing exchange filters"):
        microstructure._exchange_filters(payload, ("BTCUSDT", "ETHUSDT"))


def test_initial_snapshot_uses_hftbacktest_depth_snapshot_events() -> None:
    hftbacktest = pytest.importorskip("hftbacktest")
    snapshot = microstructure._initial_snapshot_array(
        {
            "bids": [["100.0", "2.0"], ["99.9", "1.0"]],
            "asks": [["100.1", "3.0"]],
        }
    )

    assert len(snapshot) == 3
    assert int(snapshot[0]["ev"]) == (
        hftbacktest.DEPTH_SNAPSHOT_EVENT
        | hftbacktest.BUY_EVENT
        | hftbacktest.EXCH_EVENT
        | hftbacktest.LOCAL_EVENT
    )
    assert int(snapshot[-1]["ev"]) == (
        hftbacktest.DEPTH_SNAPSHOT_EVENT
        | hftbacktest.SELL_EVENT
        | hftbacktest.EXCH_EVENT
        | hftbacktest.LOCAL_EVENT
    )
    assert snapshot[0]["px"] == pytest.approx(100.0)
    assert snapshot[-1]["qty"] == pytest.approx(3.0)


def test_synchronize_raw_capture_drops_pre_snapshot_events_and_enforces_continuity(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl.gz"
    synchronized_path = tmp_path / "synchronized.jsonl.gz"
    lines = [
        _stream_line(
            1,
            "btcusdt@depth@100ms",
            {"e": "depthUpdate", "s": "BTCUSDT", "U": 95, "u": 100, "pu": 94, "b": [["99", "1"]], "a": []},
        ),
        _stream_line(
            2,
            "btcusdt@trade",
            {"e": "trade", "s": "BTCUSDT", "E": 1, "T": 1, "p": "100", "q": "1", "m": False},
        ),
        _stream_line(
            3,
            "btcusdt@depth@100ms",
            {"e": "depthUpdate", "s": "BTCUSDT", "U": 107, "u": 110, "pu": 100, "b": [["100", "2"]], "a": [["101", "3"]]},
        ),
        _stream_line(
            4,
            "btcusdt@trade",
            {"e": "trade", "s": "BTCUSDT", "E": 2, "T": 2, "p": "0", "q": "0", "X": "NA", "m": True},
        ),
        _stream_line(
            5,
            "btcusdt@bookTicker",
            {"e": "bookTicker", "s": "BTCUSDT", "E": 2, "T": 2, "b": "100", "B": "2", "a": "101", "A": "3"},
        ),
        _stream_line(
            6,
            "btcusdt@depth@100ms",
            {"e": "depthUpdate", "s": "BTCUSDT", "U": 120, "u": 125, "pu": 110, "b": [], "a": [["102", "1"]]},
        ),
    ]
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.writelines(lines)

    messages, estimated_rows = microstructure._synchronize_raw_capture(
        raw_path,
        synchronized_path,
        snapshot_last_update_id=100,
    )

    assert messages == 3
    assert estimated_rows == 5
    with gzip.open(synchronized_path, "rt", encoding="utf-8") as handle:
        synchronized = handle.readlines()
    assert "\"U\":107" in synchronized[0]
    assert "bookTicker" in synchronized[1]
    assert "\"U\":120" in synchronized[2]


def test_synchronize_raw_capture_rejects_post_snapshot_sequence_gap(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl.gz"
    synchronized_path = tmp_path / "synchronized.jsonl.gz"
    lines = [
        _stream_line(
            1,
            "btcusdt@depth@100ms",
            {"e": "depthUpdate", "s": "BTCUSDT", "U": 101, "u": 105, "pu": 100, "b": [], "a": []},
        ),
        _stream_line(
            2,
            "btcusdt@depth@100ms",
            {"e": "depthUpdate", "s": "BTCUSDT", "U": 109, "u": 110, "pu": 108, "b": [], "a": []},
        ),
    ]
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.writelines(lines)

    with pytest.raises(ValueError, match="depth sequence gap"):
        microstructure._synchronize_raw_capture(
            raw_path,
            synchronized_path,
            snapshot_last_update_id=100,
        )
