from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pytest

from simple_ai_trading.microstructure_data import _replay_smoke
from simple_ai_trading.microstructure_normalization import (
    normalize_capture,
    prepare_timed_input,
)


BASE_MS = 1_700_000_000_000
BASE_NS = BASE_MS * 1_000_000
SNAPSHOT_ID = (1 << 53) + 37
RECEIVED = BASE_NS + 500_000_000


def source_rows():
    return [
        (
            BASE_NS + 100_000_000,
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "T": BASE_MS + 10,
                "U": SNAPSHOT_ID,
                "u": SNAPSHOT_ID + 1,
                "pu": SNAPSHOT_ID - 1,
                "b": [["100", "1"]],
                "a": [],
            },
        ),
        (
            BASE_NS + 200_000_000,
            {
                "e": "aggTrade",
                "s": "BTCUSDT",
                "T": BASE_MS + 20,
                "a": (1 << 53) + 1,
                "p": "100.5",
                "q": "1",
                "m": False,
            },
        ),
        (
            BASE_NS + 350_000_000,
            {
                "e": "bookTicker",
                "s": "BTCUSDT",
                "T": BASE_MS + 30,
                "b": "100",
                "B": "1",
                "a": "101",
                "A": "2",
            },
        ),
        (
            BASE_NS + 700_000_000,
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "T": BASE_MS + 600,
                "U": SNAPSHOT_ID + 2,
                "u": SNAPSHOT_ID + 2,
                "pu": SNAPSHOT_ID + 1,
                "b": [["100", "3"]],
                "a": [],
            },
        ),
        (
            BASE_NS + 1_000_000_000,
            {
                "e": "depthUpdate",
                "s": "BTCUSDT",
                "T": BASE_MS + 900,
                "U": SNAPSHOT_ID + 3,
                "u": SNAPSHOT_ID + 3,
                "pu": SNAPSHOT_ID + 2,
                "b": [["100", "4"]],
                "a": [],
            },
        ),
    ]


def snapshot():
    return {
        "lastUpdateId": SNAPSHOT_ID,
        "T": BASE_MS,
        "bids": [["100", "2"]],
        "asks": [["101", "2"]],
    }


def write_source(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for received, data in rows:
            stream = "btcusdt@" + data["e"]
            f.write(f"{received} {json.dumps({'stream': stream, 'data': data})}\n")


def test_native_engine_has_no_book_until_receipt_and_then_drains_buffer(tmp_path):
    hft = pytest.importorskip("hftbacktest")
    source, normalized, initial = (
        tmp_path / name for name in ("raw.gz", "data.npz", "initial.npz")
    )
    write_source(source, source_rows())
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    report = normalize_capture(
        source,
        normalized,
        initial,
        snapshot(),
        symbol="BTCUSDT",
        snapshot_received_at_ns=RECEIVED,
        clock_offset_ms=0.0,
    )
    assert report.buffered_messages == 3
    assert report.snapshot_last_update_id == SNAPSHOT_ID
    assert report.input_event_rows == 10
    with np.load(initial, allow_pickle=False) as data:
        assert len(data["data"]) == 0
    asset = (
        hft.BacktestAsset()
        .data([str(normalized)])
        .linear_asset(1.0)
        .constant_order_latency(10_000_000, 10_000_000)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0002, 0.0005)
        .tick_size(1.0)
        .lot_size(1.0)
        .last_trades_capacity(0)
    )
    engine = hft.HashMapMarketDepthBacktest([asset])
    try:
        assert not np.isfinite(engine.depth(0).best_bid)
        assert engine.elapse(0) == 0
        assert engine.current_timestamp == BASE_NS
        assert engine.elapse(499_999_999) == 0
        assert not np.isfinite(engine.depth(0).best_bid)
        assert engine.elapse(1) == 0
        assert engine.current_timestamp == RECEIVED
        assert engine.depth(0).best_bid == 100.0
        assert (
            engine.depth(0).best_bid_qty == 1.0
        )  # Buffered delta, not seed quantity 2.
    finally:
        engine.close()
    assert _replay_smoke(
        normalized, initial, tick_size=1.0, lot_size=1.0, normalization=report
    ) == (True, 100.0, 101.0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_prepared_clock_is_logical_release_not_rewritten_raw_receipt(tmp_path):
    pytest.importorskip("hftbacktest")
    source, output = tmp_path / "raw.gz", tmp_path / "prepared.gz"
    write_source(source, source_rows())
    report = prepare_timed_input(
        source,
        output,
        snapshot(),
        symbol="BTCUSDT",
        snapshot_received_at_ns=RECEIVED,
        clock_offset_ms=0.0000001,
    )
    assert report.clock_offset_ns == 1
    with gzip.open(output, "rt", encoding="utf-8") as f:
        rows = [line.split(" ", 1) for line in f]
    assert [int(r[0]) for r in rows[:4]] == [RECEIVED + 1] * 4
    assert json.loads(rows[2][1])["data"]["t"] == (1 << 53) + 1
    assert json.loads(rows[1][1])["data"]["u"] == SNAPSHOT_ID + 1
    assert int(rows[-1][0]) == BASE_NS + 1_000_000_001


@pytest.mark.parametrize(
    "field,bad",
    [
        ("a", 1.5),
        ("a", True),
        ("a", "9007199254740993"),
        ("T", BASE_MS + 0.5),
        ("m", "false"),
        ("p", True),
        ("q", False),
    ],
)
def test_malformed_native_trade_cannot_be_normalized(tmp_path, field, bad):
    pytest.importorskip("hftbacktest")
    rows = deepcopy(source_rows())
    rows[1][1][field] = bad
    source = tmp_path / "raw.gz"
    write_source(source, rows)
    with pytest.raises(ValueError):
        normalize_capture(
            source,
            tmp_path / "data.npz",
            tmp_path / "initial.npz",
            snapshot(),
            symbol="BTCUSDT",
            snapshot_received_at_ns=RECEIVED,
            clock_offset_ms=0.0,
        )
    assert not (tmp_path / "data.npz").exists()
    assert not (tmp_path / "initial.npz").exists()
    assert not list(tmp_path.glob("hft-timed-*"))


def test_gate_does_not_conceal_a_negative_physical_feed_latency(tmp_path):
    rows = source_rows()
    rows[0][1]["T"] = BASE_MS + 110
    source = tmp_path / "raw.gz"
    write_source(source, rows)
    with pytest.raises(ValueError, match="raw receipt precedes"):
        prepare_timed_input(
            source,
            tmp_path / "prepared.gz",
            snapshot(),
            symbol="BTCUSDT",
            snapshot_received_at_ns=RECEIVED,
            clock_offset_ms=0.0,
        )


def test_sorting_cannot_conceal_a_regressing_physical_receipt(tmp_path):
    rows = source_rows()
    rows[1] = (BASE_NS + 90_000_000, rows[1][1])
    source = tmp_path / "raw.gz"
    write_source(source, rows)
    with pytest.raises(ValueError, match="physical receipt clock regressed"):
        prepare_timed_input(
            source,
            tmp_path / "prepared.gz",
            snapshot(),
            symbol="BTCUSDT",
            snapshot_received_at_ns=RECEIVED,
            clock_offset_ms=0.0,
        )


def test_existing_outputs_are_preserved_before_source_access(tmp_path):
    pytest.importorskip("hftbacktest")
    output = tmp_path / "keep.npz"
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        normalize_capture(
            tmp_path / "missing.gz",
            output,
            tmp_path / "initial.npz",
            snapshot(),
            symbol="BTCUSDT",
            snapshot_received_at_ns=RECEIVED,
            clock_offset_ms=0.0,
        )
    assert output.read_bytes() == b"original"


def test_engine_eof_cannot_pass_smoke_with_a_stale_clock(tmp_path):
    pytest.importorskip("hftbacktest")
    source, normalized, initial = (
        tmp_path / name for name in ("raw.gz", "data.npz", "initial.npz")
    )
    write_source(source, source_rows()[:1])
    report = normalize_capture(
        source,
        normalized,
        initial,
        snapshot(),
        symbol="BTCUSDT",
        snapshot_received_at_ns=RECEIVED,
        clock_offset_ms=0.0,
    )
    assert _replay_smoke(
        normalized, initial, tick_size=1.0, lot_size=1.0, normalization=report
    ) == (False, None, None)


def test_smoke_refuses_static_initial_state(tmp_path):
    pytest.importorskip("hftbacktest")
    from hftbacktest import event_dtype

    source, normalized, initial = (
        tmp_path / name for name in ("raw.gz", "data.npz", "initial.npz")
    )
    write_source(source, source_rows())
    report = normalize_capture(
        source,
        normalized,
        initial,
        snapshot(),
        symbol="BTCUSDT",
        snapshot_received_at_ns=RECEIVED,
        clock_offset_ms=0.0,
    )
    np.savez_compressed(initial, data=np.zeros(1, dtype=event_dtype))
    with pytest.raises(ValueError, match="static initial snapshot"):
        _replay_smoke(
            normalized, initial, tick_size=1.0, lot_size=1.0, normalization=report
        )


def test_gap_is_rejected_before_normalized_artifacts_exist(tmp_path):
    pytest.importorskip("hftbacktest")
    rows = source_rows()
    rows[3][1]["pu"] = SNAPSHOT_ID - 1
    source = tmp_path / "raw.gz"
    write_source(source, rows)
    with pytest.raises(ValueError, match="sequence gap"):
        normalize_capture(
            source,
            tmp_path / "data.npz",
            tmp_path / "initial.npz",
            snapshot(),
            symbol="BTCUSDT",
            snapshot_received_at_ns=RECEIVED,
            clock_offset_ms=0.0,
        )
    assert not (tmp_path / "data.npz").exists()
    assert not (tmp_path / "initial.npz").exists()


@pytest.mark.parametrize("receipt", [None, True, RECEIVED + 0.5, (1 << 63)])
def test_invalid_snapshot_receipt_fails_before_source_access(tmp_path, receipt):
    with pytest.raises(ValueError):
        prepare_timed_input(
            tmp_path / "absent.gz",
            tmp_path / "prepared.gz",
            snapshot(),
            symbol="BTCUSDT",
            snapshot_received_at_ns=receipt,
            clock_offset_ms=0.0,
        )
    assert not (tmp_path / "prepared.gz").exists()
