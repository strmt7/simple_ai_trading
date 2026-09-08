"""Mutation must not turn unreadable retained inventory into an empty ledger."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from simple_ai_trading.positions import ClosedTrade, OpenPosition, PositionsStore


def _position() -> OpenPosition:
    return OpenPosition(
        id="retained",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
    )


def _trade() -> ClosedTrade:
    return ClosedTrade(
        id="retained",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1,
        entry_price=100,
        exit_price=101,
        leverage=1,
        opened_at_ms=1,
        closed_at_ms=2,
        realized_pnl=1,
        realized_pnl_pct=0.01,
    )


def _mutate(store: PositionsStore, operation: str) -> None:
    if operation == "open":
        store.record_open(replace(_position(), id="new"))
    elif operation == "remove":
        store.remove_open("retained")
    elif operation == "close":
        store.record_close(_trade())
    else:
        store.record_close_result(_position(), replace(_trade(), qty=0.5))


@pytest.mark.parametrize("operation", ["open", "remove", "close", "partial_close"])
@pytest.mark.parametrize("body", [b"{broken", b"{}", b"[null]", b"[{}]", b"\xff"])
def test_bad_open_file_is_preserved_before_any_mutation(tmp_path, operation, body):
    store = PositionsStore(tmp_path)
    store.open_path.write_bytes(body)
    store.ledger_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        _mutate(store, operation)
    assert store.open_path.read_bytes() == body
    assert store.ledger_path.read_bytes() == b"[]"


@pytest.mark.parametrize("operation", ["close", "partial_close"])
@pytest.mark.parametrize("body", [b"{broken", b"{}", b"[null]", b"[{}]", b"\xff"])
def test_bad_closed_file_is_preserved_before_any_mutation(tmp_path, operation, body):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    original = store.open_path.read_bytes()
    store.ledger_path.write_bytes(body)
    with pytest.raises(ValueError):
        _mutate(store, operation)
    assert store.open_path.read_bytes() == original
    assert store.ledger_path.read_bytes() == body


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_existing_quantity_cannot_allow_partial_ledger_write(tmp_path, value):
    store = PositionsStore(tmp_path)
    body = json.dumps([asdict(_position())]).replace('"qty": 1', f'"qty": {value}')
    store.open_path.write_text(body, encoding="utf-8")
    original = store.open_path.read_bytes()
    with pytest.raises(ValueError):
        store.record_close(_trade())
    assert store.open_path.read_bytes() == original
    assert not store.ledger_path.exists()


@pytest.mark.parametrize(
    "kind", ["field", "identity", "missing_row", "unknown_field", "invalid_identity"]
)
def test_ambiguous_or_filtered_open_rows_cannot_disappear(tmp_path, kind):
    store = PositionsStore(tmp_path)
    row = asdict(_position())
    if kind == "field":
        body = json.dumps([row]).replace(
            '"id": "retained"', '"id": "foreign", "id": "retained"'
        )
    elif kind == "identity":
        body = json.dumps([row, row])
    elif kind == "missing_row":
        body = json.dumps([row, {"id": "unknown"}])
    elif kind == "unknown_field":
        body = json.dumps([dict(row, unrecognized=True)])
    else:
        body = json.dumps([dict(row, id=[])])
    store.open_path.write_text(body, encoding="utf-8")
    original = store.open_path.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        store.remove_open("retained")
    assert store.open_path.read_bytes() == original


def test_lossless_valid_updates_and_partial_close_keep_existing_rows(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    store.record_open(replace(_position(), id="other"))
    store.record_close_result(_position(), replace(_trade(), qty=0.5))
    assert {p.id: p.qty for p in store.load_open(strict=True)} == {
        "retained": 0.5,
        "other": 1,
    }
    store.record_close(replace(_trade(), qty=0.5))
    assert [p.id for p in store.load_open(strict=True)] == ["other"]
    # Multiple close fills legitimately share a position ID.
    assert [t.qty for t in store.load_ledger(strict=True)] == [0.5, 0.5]
    assert store.remove_open("other")
    assert not store.remove_open("absent")


def test_read_only_legacy_projection_does_not_grant_write_admission(tmp_path):
    store = PositionsStore(tmp_path)
    store.open_path.write_text('[{"id": "incomplete"}]', encoding="utf-8")
    assert store.load_open() == []
    with pytest.raises(ValueError):
        store.record_open(_position())


def test_unreadable_open_file_does_not_create_closed_file(tmp_path):
    store = PositionsStore(tmp_path)
    store.open_path.mkdir()
    with pytest.raises(ValueError):
        store.record_close(_trade())
    assert store.open_path.is_dir()
    assert not store.ledger_path.exists()


def test_legacy_invalid_encoding_is_not_newly_reported_as_empty(tmp_path):
    store = PositionsStore(tmp_path)
    store.open_path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        store.load_open()
