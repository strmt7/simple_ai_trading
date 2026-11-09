from __future__ import annotations

import gzip
import json
from pathlib import Path
import random

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

    assert microstructure._exchange_filters(payload, ("BTCUSDT",)) == {
        "BTCUSDT": (0.1, 0.001)
    }
    with pytest.raises(ValueError, match="missing exchange filters"):
        microstructure._exchange_filters(payload, ("BTCUSDT", "ETHUSDT"))


def test_stream_urls_separate_public_book_data_from_market_aggregate_trades() -> None:
    public_url, market_url = microstructure._stream_urls(
        ("BTCUSDT", "ETHUSDT"),
        microstructure.BINANCE_FUTURES_PUBLIC_STREAM_URL,
        microstructure.BINANCE_FUTURES_MARKET_STREAM_URL,
    )

    assert "/public/stream?streams=" in public_url
    assert "btcusdt@depth@100ms" in public_url
    assert "ethusdt@bookTicker" in public_url
    assert "aggTrade" not in public_url
    assert "/market/stream?streams=" in market_url
    assert "btcusdt@aggTrade" in market_url
    assert "depth" not in market_url
    assert "bookTicker" not in market_url


def test_capture_segment_limit_leaves_margin_before_binance_24_hour_disconnect(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="82800"):
        microstructure.capture_binance_futures_microstructure(
            ("BTCUSDT",),
            duration_seconds=82_801,
            output_root=tmp_path,
        )


def test_observe_aggregate_trade_counts_underlying_real_fills() -> None:
    stats = microstructure._MutableSymbolStats("BTCUSDT")

    microstructure._observe_event(
        stats,
        {
            "e": "aggTrade",
            "E": 1_700_000_000_010,
            "T": 1_700_000_000_000,
            "s": "BTCUSDT",
            "a": 10,
            "p": "50000",
            "q": "0.06",
            "f": 100,
            "l": 102,
            "m": False,
        },
        receive_time_ns=1_700_000_000_020_000_000,
        clock_offset_ms=0.0,
        rng=random.Random(0),
    )

    assert stats.trade_messages == 1
    assert stats.trade_fill_count == 3
    assert stats.invalid_event_count == 0


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
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "U": 95,
                "u": 100,
                "pu": 94,
                "b": [["99", "1"]],
                "a": [],
            },
        ),
        _stream_line(
            2,
            "btcusdt@trade",
            {
                "e": "trade",
                "s": "BTCUSDT",
                "E": 1,
                "T": 1,
                "p": "100",
                "q": "1",
                "m": False,
            },
        ),
        _stream_line(
            3,
            "btcusdt@depth@100ms",
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "U": 107,
                "u": 110,
                "pu": 100,
                "b": [["100", "2"]],
                "a": [["101", "3"]],
            },
        ),
        _stream_line(
            4,
            "btcusdt@trade",
            {
                "e": "trade",
                "s": "BTCUSDT",
                "E": 2,
                "T": 2,
                "p": "0",
                "q": "0",
                "X": "NA",
                "m": True,
            },
        ),
        _stream_line(
            5,
            "btcusdt@bookTicker",
            {
                "e": "bookTicker",
                "s": "BTCUSDT",
                "E": 2,
                "T": 2,
                "b": "100",
                "B": "2",
                "a": "101",
                "A": "3",
            },
        ),
        _stream_line(
            6,
            "btcusdt@depth@100ms",
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "U": 120,
                "u": 125,
                "pu": 110,
                "b": [],
                "a": [["102", "1"]],
            },
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
    assert '"U":107' in synchronized[0]
    assert "bookTicker" in synchronized[1]
    assert '"U":120' in synchronized[2]


def test_synchronize_raw_capture_rejects_post_snapshot_sequence_gap(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl.gz"
    synchronized_path = tmp_path / "synchronized.jsonl.gz"
    lines = [
        _stream_line(
            1,
            "btcusdt@depth@100ms",
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "U": 101,
                "u": 105,
                "pu": 100,
                "b": [],
                "a": [],
            },
        ),
        _stream_line(
            2,
            "btcusdt@depth@100ms",
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "U": 109,
                "u": 110,
                "pu": 108,
                "b": [],
                "a": [],
            },
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


@pytest.mark.parametrize("field", ["U", "u", "pu"])
@pytest.mark.parametrize("representation", ["fractional", "float", "string", "boolean"])
def test_synchronizer_rejects_lossy_native_update_ids(
    tmp_path: Path, field: str, representation: str
) -> None:
    event = {
        "e": "depthUpdate",
        "s": "BTCUSDT",
        "U": 101,
        "u": 105,
        "pu": 100,
        "b": [],
        "a": [],
    }
    original = event[field]
    event[field] = {
        "fractional": float(original) + 0.5,
        "float": float(original),
        "string": str(original),
        "boolean": True,
    }[representation]
    raw = tmp_path / "raw.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as out:
        out.write(_stream_line(1, "btcusdt@depth", event))
    with pytest.raises(ValueError, match="positive native integer"):
        microstructure._synchronize_raw_capture(
            raw, tmp_path / "out.gz", snapshot_last_update_id=100
        )


@pytest.mark.parametrize("snapshot", [True, False, 0, -1, 100.5, 100.0, "100"])
def test_synchronizer_rejects_invalid_snapshot_identity(
    tmp_path: Path, snapshot: object
) -> None:
    with pytest.raises(ValueError, match="positive native integer"):
        microstructure._synchronize_raw_capture(
            tmp_path / "missing.gz",
            tmp_path / "out.gz",
            snapshot_last_update_id=snapshot,
        )
    assert not (tmp_path / "out.gz").exists()


def test_synchronizer_preserves_native_integer_precision_above_float_range(
    tmp_path: Path,
) -> None:
    snapshot = 2**53 + 1
    event = {
        "e": "depthUpdate",
        "s": "BTCUSDT",
        "U": snapshot + 1,
        "u": snapshot + 5,
        "pu": snapshot,
        "b": [],
        "a": [],
    }
    raw = tmp_path / "raw.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as out:
        out.write(_stream_line(1, "btcusdt@depth", event))
    target = tmp_path / "out.gz"
    assert microstructure._synchronize_raw_capture(
        raw, target, snapshot_last_update_id=snapshot
    ) == (1, 0)
    with gzip.open(target, "rt", encoding="utf-8") as inp:
        assert json.loads(inp.read().split(" ", 1)[1])["data"]["pu"] == snapshot


@pytest.mark.parametrize("event_type", ["trade", "bookTicker"])
@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_observer_rejects_nonfinite_market_values(event_type: str, bad: str) -> None:
    event = {
        "e": event_type,
        "E": 1,
        "T": 1,
        "t": 1,
        "p": bad,
        "q": "1",
        "m": False,
        "b": bad,
        "B": "1",
        "a": "101",
        "A": "1",
    }
    stats = microstructure._MutableSymbolStats("BTCUSDT")
    microstructure._observe_event(
        stats,
        event,
        receive_time_ns=2_000_000,
        clock_offset_ms=0.0,
        rng=random.Random(0),
    )
    assert stats.invalid_event_count == 1
    assert stats.trade_messages == stats.book_ticker_messages == 0
    assert stats.latency_samples_ms == []


@pytest.mark.parametrize("bad_clock", [True, 1.5, "1", "invalid", None])
def test_observer_rejects_malformed_clock_without_crashing_or_advancing_stats(
    bad_clock: object,
) -> None:
    stats = microstructure._MutableSymbolStats("BTCUSDT")
    event = {
        "e": "depthUpdate",
        "E": bad_clock,
        "U": 101,
        "u": 105,
        "pu": 100,
        "b": [],
        "a": [],
    }
    microstructure._observe_event(
        stats,
        event,
        receive_time_ns=2_000_000,
        clock_offset_ms=0.0,
        rng=random.Random(0),
    )
    assert stats.invalid_event_count == 1
    assert stats.previous_depth_update_id is None
    assert stats.depth_messages == 0


def test_observer_invalid_levels_do_not_advance_sequence_or_clock() -> None:
    stats = microstructure._MutableSymbolStats("BTCUSDT")
    event = {
        "e": "depthUpdate",
        "E": 1,
        "U": 101,
        "u": 105,
        "pu": 100,
        "b": [["NaN", "1"]],
        "a": [],
    }
    microstructure._observe_event(
        stats,
        event,
        receive_time_ns=2_000_000,
        clock_offset_ms=0.0,
        rng=random.Random(0),
    )
    assert stats.invalid_event_count == 1
    assert stats.previous_depth_update_id is None
    assert stats.depth_messages == 0
    assert stats.first_exchange_time_ms is None


@pytest.mark.parametrize(
    "snapshot",
    [
        {"lastUpdateId": True, "bids": [["100", "1"]], "asks": [["101", "1"]]},
        {"lastUpdateId": 100.5, "bids": [["100", "1"]], "asks": [["101", "1"]]},
        {"lastUpdateId": "100", "bids": [["100", "1"]], "asks": [["101", "1"]]},
        {"lastUpdateId": 100, "bids": [["NaN", "1"]], "asks": [["101", "1"]]},
        {"lastUpdateId": 100, "bids": [["100", "Infinity"]], "asks": [["101", "1"]]},
    ],
)
def test_snapshot_fetch_rejects_malformed_source_before_normalization(
    snapshot: dict[str, object],
) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return snapshot

    class Session:
        def get(self, *args: object, **kwargs: object) -> Response:
            return Response()

    with pytest.raises(ValueError):
        microstructure._fetch_depth_snapshot(
            Session(), "BTCUSDT", rest_url="https://example.invalid", timeout_seconds=1
        )


def test_prepare_hftbacktest_input_maps_real_aggregate_trade_schema(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synchronized.jsonl.gz"
    output = tmp_path / "hft-input.jsonl.gz"
    line = _stream_line(
        1_700_000_000_020_000_000,
        "btcusdt@aggTrade",
        {
            "e": "aggTrade",
            "E": 1_700_000_000_010,
            "T": 1_700_000_000_000,
            "s": "BTCUSDT",
            "a": 10,
            "p": "50000",
            "q": "0.06",
            "f": 100,
            "l": 102,
            "m": False,
        },
    )
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write(line)

    transformed = microstructure._prepare_hftbacktest_input(source, output)

    assert transformed == 1
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        _received_at, raw_json = handle.readline().split(" ", 1)
    payload = json.loads(raw_json)
    assert payload["stream"] == "btcusdt@trade"
    assert payload["data"]["e"] == "trade"
    assert payload["data"]["t"] == 10
    assert payload["data"]["X"] == "MARKET"
    assert payload["data"]["q"] == "0.06"
