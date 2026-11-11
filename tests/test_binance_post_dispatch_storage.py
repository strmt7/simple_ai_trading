"""Accepted synthetic fills must never bootstrap a lost execution namespace."""

from contextlib import closing
import sqlite3

import pytest

from simple_ai_trading import binance_close_intents
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_owned_execution import (
    close_trade_template,
    submit_durable_close,
    submit_durable_open,
)
from simple_ai_trading.positions import PositionsStore
from test_binance_owned_runtime import SCOPE, Venue, position


@pytest.mark.parametrize("phase", ["opening", "closing"])
@pytest.mark.parametrize(
    "point", ["after_dispatch", "inside_accounting", "before_completion"]
)
def test_storage_loss_after_fill_never_creates_a_replacement(
    tmp_path, monkeypatch, enrolled_binance_runtime, phase, point
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE, opening_policy_quote="USDT")
    venue = Venue(store)
    requested = position()
    if phase == "closing":
        requested = submit_durable_open(venue, requested, store).position
    before = store.load_snapshot(strict=True)
    retained = tmp_path / "retained-original.sqlite3"

    def disappear():
        store.opening_intents.path.rename(retained)

    if point == "after_dispatch":
        original = venue.place_order

        def dispatch(*args, **kwargs):
            response = original(*args, **kwargs)
            disappear()
            return response

        monkeypatch.setattr(venue, "place_order", dispatch)
    elif point == "inside_accounting":
        name = "record_open" if phase == "opening" else "record_close_result"
        original = getattr(store, name)

        def account(*args, **kwargs):
            assert kwargs["create"] is False
            disappear()
            return original(*args, **kwargs)

        monkeypatch.setattr(store, name, account)
    elif phase == "opening":
        original = BinanceOpenIntentJournal.record_complete

        def complete(*args, **kwargs):
            disappear()
            return original(*args, **kwargs)

        monkeypatch.setattr(BinanceOpenIntentJournal, "record_complete", complete)
    else:
        original = binance_close_intents.complete_close

        def complete(*args, **kwargs):
            assert kwargs["create"] is False
            disappear()
            return original(*args, **kwargs)

        monkeypatch.setattr(binance_close_intents, "complete_close", complete)

    with pytest.raises((OpenIntentError, ValueError)):
        if phase == "opening":
            submit_durable_open(venue, requested, store)
        else:
            submit_durable_close(
                venue,
                requested,
                close_trade_template(requested, 50000, "loss"),
                store,
                reduce_only=False,
            )
    assert not store.opening_intents.path.exists()
    after = store.load_snapshot(strict=True)
    assert not store.opening_intents.path.exists()
    if point == "before_completion":
        assert len(after[0]) == (1 if phase == "opening" else 0)
        assert len(after[1]) == (0 if phase == "opening" else 1)
    else:
        assert after == before
    table = "open_intent" if phase == "opening" else "close_intent"
    with closing(
        sqlite3.connect(retained.resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM " + table + " WHERE state='UNKNOWN'"
            ).fetchone()[0]
            == 1
        )
    assert venue.writes == (["BUY"] if phase == "opening" else ["BUY", "SELL"])
    authority.control.request_stop(reason="synthetic storage-loss verification")
    assert authority.control.snapshot().state == "stop_requested"
    with authority.control.close_guard(timeout_seconds=1):
        pass


def test_missing_close_journal_is_not_created_before_dispatch(
    tmp_path, enrolled_binance_runtime
):
    store = PositionsStore(tmp_path)
    enrolled_binance_runtime(store, SCOPE, opening_policy_quote="USDT")
    venue = Venue(store)
    opened = submit_durable_open(venue, position(), store).position
    store.opening_intents.path.rename(tmp_path / "retained-original.sqlite3")
    with pytest.raises(ValueError, match="storage"):
        submit_durable_close(
            venue,
            opened,
            close_trade_template(opened, 50000, "loss"),
            store,
            reduce_only=False,
        )
    assert not store.opening_intents.path.exists()
    assert venue.writes == ["BUY"] and store.load_open() == [opened]
