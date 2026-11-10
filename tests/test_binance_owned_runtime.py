"""Real local authority and synthetic native acknowledgments; no network."""

from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_owned_execution import (
    _submit_open_position,
    _submit_close_position,
    close_trade_template,
    submit_durable_open,
    submit_durable_close,
)
from simple_ai_trading.binance_runtime_control import (
    BINANCE_RUNTIME_LEASE_ENV,
    BinanceRuntimeControl,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore

SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-runtime-fence"
)


def position(**changes):
    return replace(
        OpenPosition(
            id="runtime-one",
            symbol="BTCUSDT",
            market_type="spot",
            side="LONG",
            qty=0.001,
            entry_price=50000,
            leverage=1,
            opened_at_ms=1,
            notional=50,
            dry_run=False,
            open_client_order_id="sait-o-runtimeone",
        ),
        **changes,
    )


class Venue:
    def __init__(self, store):
        self.store, self.writes, self.failure = store, [], None

    def execution_scope(self):
        return SCOPE

    def place_order(self, symbol, side, quantity, **options):
        if side == "BUY":
            assert (
                self.store.opening_intents.entry_block_reason()
                == "unresolved_opening_intents=1"
            )
        self.writes.append(side)
        if self.failure:
            raise self.failure
        return {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "status": "FILLED",
            "orderId": len(self.writes),
            "clientOrderId": options["client_order_id"],
            "executedQty": str(quantity),
            "cummulativeQuoteQty": str(quantity * 50000),
            "fills": [
                {
                    "qty": str(quantity),
                    "price": "50000",
                    "commission": "0",
                    "commissionAsset": "BTC" if side == "BUY" else "USDT",
                    "tradeId": len(self.writes),
                }
            ],
        }

    def get_order(self, *args, **options):
        raise BinanceAPIError("synthetic unresolved receipt")


def test_persisted_stop_rejects_before_unknown_even_without_caller_guard(tmp_path):
    store = PositionsStore(tmp_path)
    control = BinanceRuntimeControl(tmp_path / "binance_runtime_control.sqlite3")
    control.provision_storage()
    venue = Venue(store)
    with pytest.raises(OpenIntentError, match="runtime"):
        submit_durable_open(venue, position(), store)
    assert venue.writes == [] and not store.opening_intents.path.exists()
    assert control.snapshot().state == "stop_requested"


@pytest.mark.parametrize(
    "damage",
    [
        "lease",
        "stop",
        "pause",
        "journal",
        "replacement",
        "uuid",
        "marker",
        "controller",
    ],
)
def test_opening_requires_current_lease_and_both_enrollments(
    tmp_path, monkeypatch, enrolled_binance_runtime, damage
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    binding = BinanceExecutionStorage(store.opening_intents, authority.control)
    if damage == "lease":
        monkeypatch.delenv(BINANCE_RUNTIME_LEASE_ENV)
    elif damage == "stop":
        authority.control.request_stop(reason="synthetic stop")
    elif damage == "pause":
        authority.control.set_paused(True, reason="synthetic pause")
    elif damage in {"journal", "replacement"}:
        body = store.opening_intents.path.read_bytes()
        store.opening_intents.path.rename(tmp_path / "retained.sqlite3")
        if damage == "replacement":
            store.opening_intents.path.write_bytes(body)
    elif damage == "uuid":
        with closing(sqlite3.connect(store.opening_intents.path)) as connection:
            connection.execute("DELETE FROM runtime_storage_enrollment")
            connection.commit()
    elif damage == "marker":
        binding.enrollment_path.unlink()
    else:
        authority.control.ledger_path.unlink()
    venue = Venue(store)
    with pytest.raises(OpenIntentError):
        submit_durable_open(venue, position(), store)
    assert venue.writes == [] and store.load_open() == []


def test_default_session_authority_and_explicit_authority_publish_owned_inventory(
    tmp_path, enrolled_binance_runtime
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    venue = Venue(store)
    first = submit_durable_open(venue, position(), store)
    second = submit_durable_open(
        venue,
        position(id="runtime-two", open_client_order_id="sait-o-runtimetwo"),
        store,
        interlock=authority,
    )
    assert store.load_open() == [first.position, second.position]
    assert venue.writes == ["BUY", "BUY"]


@pytest.mark.parametrize("damage", ["stop", "controller", "opening-marker"])
def test_owned_close_survives_opening_authority_loss(
    tmp_path, enrolled_binance_runtime, damage
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    venue = Venue(store)
    opened = submit_durable_open(venue, position(), store).position
    if damage == "stop":
        authority.control.request_stop(reason="close-only")
    elif damage == "controller":
        authority.control.ledger_path.unlink()
    else:
        BinanceExecutionStorage(
            store.opening_intents, authority.control
        ).enrollment_path.unlink()
    closed = submit_durable_close(
        venue,
        opened,
        close_trade_template(opened, 50000, "test"),
        store,
        reduce_only=False,
    )
    assert closed.trade.qty == opened.qty and store.load_open() == []
    assert venue.writes == ["BUY", "SELL"]


def test_lost_closing_lock_cannot_be_recreated_for_close(
    tmp_path, enrolled_binance_runtime
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    venue = Venue(store)
    opened = submit_durable_open(venue, position(), store).position
    authority.control.close_lock_path.unlink()
    with pytest.raises(OpenIntentError):
        submit_durable_close(
            venue,
            opened,
            close_trade_template(opened, 50000, "test"),
            store,
            reduce_only=False,
        )
    assert venue.writes == ["BUY"] and not authority.control.close_lock_path.exists()


@pytest.mark.parametrize("after_prepare", [False, True])
def test_loss_during_admission_never_reinitializes_or_dispatches(
    tmp_path, monkeypatch, enrolled_binance_runtime, after_prepare
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    original = BinanceOpenIntentJournal.prepare

    def interrupted(journal, request, *, scope, create):
        assert create is False
        if after_prepare:
            original(journal, request, scope=scope, create=create)
            BinanceExecutionStorage(journal, authority.control).enrollment_path.unlink()
        else:
            journal.path.rename(tmp_path / "retained.sqlite3")
            original(journal, request, scope=scope, create=create)

    monkeypatch.setattr(BinanceOpenIntentJournal, "prepare", interrupted)
    venue = Venue(store)
    with pytest.raises(OpenIntentError):
        submit_durable_open(venue, position(), store)
    assert venue.writes == []
    if after_prepare:
        assert (
            store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
        )
    else:
        assert not store.opening_intents.path.exists()


def test_cancel_preserves_unknown_and_releases_runtime_gate(
    tmp_path, enrolled_binance_runtime
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    venue = Venue(store)
    venue.failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        submit_durable_open(venue, position(), store)
    assert store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    authority.control.request_stop(reason="after cancellation")
    assert authority.control.snapshot().state == "stop_requested"


def test_legacy_signed_shortcuts_reject_without_any_write(tmp_path):
    venue = Venue(PositionsStore(tmp_path))
    requested = position()
    with pytest.raises(OpenIntentError, match="durable"):
        _submit_open_position(venue, requested)
    with pytest.raises(OpenIntentError, match="durable"):
        _submit_close_position(
            venue,
            requested,
            close_trade_template(requested, 50000, "test"),
            reduce_only=False,
        )
    paper = replace(requested, dry_run=True)
    assert _submit_open_position(venue, paper) == paper
    assert (
        _submit_close_position(
            venue, paper, close_trade_template(paper, 50000, "test"), reduce_only=False
        ).exchange_status
        == "paper"
    )
    assert venue.writes == []


def test_opening_uses_a_private_snapshot_of_the_journaled_request(
    tmp_path, monkeypatch, enrolled_binance_runtime
):
    store = PositionsStore(tmp_path)
    enrolled_binance_runtime(store, SCOPE)
    requested = position()
    original = BinanceOpenIntentJournal.prepare

    def mutate_caller(journal, frozen, *, scope, create):
        original(journal, frozen, scope=scope, create=create)
        requested.qty, requested.symbol = 10, "ETHUSDT"

    monkeypatch.setattr(BinanceOpenIntentJournal, "prepare", mutate_caller)
    result = submit_durable_open(Venue(store), requested, store)
    assert result.position.qty == 0.001 and result.position.symbol == "BTCUSDT"


def test_repeated_owned_cycles_release_locks_connections_and_native_resources(
    tmp_path,
    enrolled_binance_runtime,
    process_resource_sample,
    record_testsuite_property,
):
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(store, SCOPE)
    venue = Venue(store)
    observations = []
    for index in range(24):
        assert authority.control.heartbeat(authority.lease_id)
        requested = position(
            id=f"owned-{index}", open_client_order_id=f"sait-o-owned-{index}"
        )
        opened = submit_durable_open(venue, requested, store).position
        submit_durable_close(
            venue,
            opened,
            close_trade_template(opened, 50000, "cycle"),
            store,
            reduce_only=False,
        )
        if index >= 4:
            observations.append(process_resource_sample())
    for label, column in (("handles", 0), ("mutexes", 1), ("private_bytes", 2)):
        for aggregate, value in (
            ("first", observations[0][column]),
            ("max", max(row[column] for row in observations)),
            ("last", observations[-1][column]),
        ):
            record_testsuite_property(f"owned_lifecycle_{label}_{aggregate}", value)
    assert max(row[0] for row in observations) <= observations[0][0] + 2
    assert max(row[1] for row in observations) == observations[0][1]
    assert max(row[2] for row in observations) <= observations[0][2] + 8 * 1024**2
    assert store.load_open() == [] and len(store.load_ledger()) == 24
