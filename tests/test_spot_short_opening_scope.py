"""Unsupported Spot short openings cannot create uncloseable owned obligations."""

import json
import sqlite3
from dataclasses import replace

import pytest

from simple_ai_trading.autonomous import _submit_durable_open_position
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.positions import PositionsStore
from test_binance_open_intents import _Client, _position, _scope


def test_spot_short_rejects_before_journal_creation_or_submission(tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    with pytest.raises(OpenIntentError):
        journal.prepare(_position(side="SHORT"), scope=_scope())
    assert not journal.path.exists()


def test_durable_spot_short_never_reaches_exchange(tmp_path):
    store = PositionsStore(tmp_path)
    client = _Client(store)
    with pytest.raises(OpenIntentError):
        _submit_durable_open_position(
            client=client, position=_position(side="SHORT"), store=store
        )
    assert client.writes == 0
    assert store.load_open() == []


def test_futures_short_keeps_its_existing_durable_scope(tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    journal.prepare(
        _position(side="SHORT", market_type="futures"), scope=_scope("futures")
    )
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_unsupported_historical_obligation_is_not_deleted_or_rearmed(tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite")
    position = _position()
    journal.prepare(position, scope=_scope())
    with sqlite3.connect(journal.path) as connection:
        original = json.loads(
            connection.execute("SELECT request_json FROM open_intent").fetchone()[0]
        )
        original["side"] = original["position_template"]["side"] = "SHORT"
        historical = json.dumps(original, sort_keys=True, allow_nan=False)
        connection.execute("UPDATE open_intent SET request_json=?", (historical,))
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"
    with pytest.raises(OpenIntentError):
        journal.pending_position(scope=_scope())
    with pytest.raises(OpenIntentError):
        journal.prepare(
            replace(position, id="new", open_client_order_id="sait-o-new"),
            scope=_scope(),
        )
    with sqlite3.connect(journal.path) as connection:
        assert connection.execute(
            "SELECT request_json,state FROM open_intent"
        ).fetchall() == [(historical, "UNKNOWN")]
