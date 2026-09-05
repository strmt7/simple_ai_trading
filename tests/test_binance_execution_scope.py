from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest

from simple_ai_trading.api import BinanceAPIError, BinanceClient
from simple_ai_trading.autonomous import _submit_durable_open_position
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_DEMO,
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore


def _position(market_type="spot"):
    return OpenPosition(
        id="scope-one",
        symbol="BTCUSDT",
        market_type=market_type,
        side="LONG",
        qty=0.001,
        entry_price=50000.0,
        leverage=1.0,
        opened_at_ms=1,
        notional=50.0,
        dry_run=False,
        open_client_order_id="sait-o-scopeone",
    )


@pytest.fixture
def client():
    instance = BinanceClient(
        "offline-placeholder", "offline-placeholder", max_retries=0
    )
    instance._throttle = lambda: None
    instance.session.request = Mock(side_effect=AssertionError("unexpected transport"))
    yield instance
    instance.session.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "rotated-offline-placeholder"),
        ("base_url", BINANCE_SPOT_DEMO),
        ("market_type", "futures"),
    ],
)
def test_persisted_scope_rejects_configuration_change_before_transmission(
    client, tmp_path, monkeypatch, field, value
):
    store = PositionsStore(tmp_path)
    original_scope = client.execution_scope()
    prepare = BinanceOpenIntentJournal.prepare

    def prepare_then_mutate(journal, position, *, scope):
        prepare(journal, position, scope=scope)
        setattr(client, field, value)

    monkeypatch.setattr(BinanceOpenIntentJournal, "prepare", prepare_then_mutate)
    with pytest.raises(BinanceAPIError, match="scope"):
        _submit_durable_open_position(client, _position(), store)
    client.session.request.assert_not_called()
    assert store.opening_intents.pending_position(scope=original_scope) == _position()
    assert store.opening_intents.entry_block_reason() == "unresolved_opening_intents=1"
    assert store.load_open() == []


def test_scoped_readback_binds_key_without_serializing_it(client, tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    journal.prepare(_position(), scope=scope)
    assert journal.pending_position(scope=scope) == _position()
    assert "offline-placeholder" not in json.dumps(asdict(scope))
    rotated = BinanceExecutionScope.from_api_key(
        BINANCE_SPOT_TESTNET, "spot", "another-offline-placeholder"
    )
    with pytest.raises(OpenIntentError, match="execution scope"):
        journal.pending_position(scope=rotated)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        payload = connection.execute("SELECT request_json FROM open_intent").fetchone()[
            0
        ]
    assert "offline-placeholder" not in payload
    assert json.loads(payload)["execution_scope"] == asdict(scope)


def test_legacy_pending_intent_is_preserved_not_assigned_to_current_key(
    client, tmp_path
):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    journal.prepare(_position(), scope=scope)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute(
            "UPDATE open_intent SET request_json=?", (journal._request(_position()),)
        )
    before = journal.path.read_bytes()
    with pytest.raises(OpenIntentError, match="legacy"):
        journal.pending_position(scope=scope)
    assert journal.path.read_bytes() == before
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"
    client.session.request.assert_not_called()


@pytest.mark.parametrize(
    "payload", ["null", "[]", "{", '{"intent_version":2}', '{"intent_version":3}']
)
def test_unreadable_pending_payload_cannot_authorize_recovery(
    client, tmp_path, payload
):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    journal.prepare(_position(), scope=scope)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("UPDATE open_intent SET request_json=?", (payload,))
    with pytest.raises(OpenIntentError):
        journal.pending_position(scope=scope)
    assert journal.entry_block_reason() == "unresolved_opening_intents=1"


def test_exact_row_identity_and_scope_required_to_release_intent(client, tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    requested = _position()
    recorded = replace(
        requested, exchange_status="FILLED", open_exchange_order_id="123"
    )
    journal.prepare(requested, scope=scope)
    wrong_scope = replace(scope, origin=BINANCE_SPOT_DEMO)
    with pytest.raises(OpenIntentError, match="scope"):
        journal.record_complete(requested, recorded, scope=wrong_scope)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("UPDATE open_intent SET position_id='different'")
    with pytest.raises(OpenIntentError, match="inconsistent"):
        journal.pending_position(scope=scope)
    with pytest.raises(OpenIntentError, match="transition"):
        journal.record_complete(requested, recorded, scope=scope)


def test_missing_or_product_mismatched_scope_cannot_prepare(client, tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    with pytest.raises(OpenIntentError):
        journal.prepare(_position(), scope=None)
    with pytest.raises(OpenIntentError):
        journal.prepare(_position("futures"), scope=client.execution_scope())
    assert not journal.path.exists()


def test_scope_is_checked_for_exact_order_query(client):
    scope = client.execution_scope()
    client.api_key = "rotated-offline-placeholder"
    with pytest.raises(BinanceAPIError, match="scope"):
        client.get_order(
            "BTCUSDT", orig_client_order_id="sait-o-scopeone", expected_scope=scope
        )
    client.session.request.assert_not_called()


def test_scope_cannot_be_used_to_treat_public_data_as_authenticated(client):
    with pytest.raises(BinanceAPIError, match="scope"):
        client._request("GET", "/api/v3/time", expected_scope=client.execution_scope())
    client.session.request.assert_not_called()


def test_unscoped_notional_fallback_still_binds_leverage_request():
    instance = BinanceClient(
        "offline-placeholder", "offline-placeholder", market_type="futures"
    )
    scope = instance.execution_scope()
    instance.api_key = "rotated-offline-placeholder"
    request = Mock(side_effect=AssertionError("unexpected transport"))
    instance.session.request = request
    try:
        with pytest.raises(BinanceAPIError, match="scope"):
            instance.set_leverage("BTCUSDT", 1, expected_scope=scope)
        request.assert_not_called()
    finally:
        instance.session.close()


def test_empty_and_corrupt_journal_recovery_are_distinct(client, tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    assert journal.pending_position(scope=scope) is None
    journal.path.write_bytes(b"not a database")
    with pytest.raises(OpenIntentError, match="unreadable"):
        journal.pending_position(scope=scope)


def test_recorded_history_fences_future_key_changes(client, tmp_path):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    position = _position()
    journal.prepare(position, scope=scope)
    journal.record_complete(
        position,
        replace(position, exchange_status="FILLED", open_exchange_order_id="123"),
        scope=scope,
    )
    next_position = replace(
        position, id="scope-two", open_client_order_id="sait-o-scopetwo"
    )
    client.api_key = "rotated-offline-placeholder"
    with pytest.raises(OpenIntentError, match="scope"):
        journal.prepare(next_position, scope=client.execution_scope())
    journal.prepare(next_position, scope=scope)
    assert journal.pending_position(scope=scope) == next_position


def test_legacy_database_is_not_silently_migrated_even_when_all_rows_recorded(
    client, tmp_path
):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    position = _position()
    journal.prepare(position, scope=scope)
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("UPDATE open_intent SET state='RECORDED'")
        connection.execute("DROP TABLE execution_scope")
        connection.execute("PRAGMA user_version=1")
    before = journal.path.read_bytes()
    with pytest.raises(OpenIntentError, match="migration"):
        journal.prepare(
            replace(position, id="two", open_client_order_id="sait-o-two"), scope=scope
        )
    assert journal.path.read_bytes() == before


def test_lost_scope_and_deleted_journal_are_not_repaired_by_acknowledgement(
    client, tmp_path
):
    journal = BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    scope = client.execution_scope()
    position = _position()
    journal.prepare(position, scope=scope)
    recorded = replace(position, exchange_status="FILLED", open_exchange_order_id="123")
    with closing(sqlite3.connect(journal.path)) as connection, connection:
        connection.execute("DELETE FROM execution_scope")
    with pytest.raises(OpenIntentError, match="scope"):
        journal.record_complete(position, recorded, scope=scope)
    with pytest.raises(OpenIntentError, match="lost"):
        journal.prepare(
            replace(position, id="two", open_client_order_id="sait-o-two"), scope=scope
        )
    journal.path.unlink()
    with pytest.raises(OpenIntentError):
        journal.record_complete(position, recorded, scope=scope)
    assert not journal.path.exists()


@pytest.mark.parametrize("market_type", ["spot", "futures"])
def test_full_signed_operation_preserves_scope_through_all_requests(
    tmp_path, market_type
):
    instance = BinanceClient(
        "offline-placeholder",
        "offline-placeholder",
        market_type=market_type,
        max_retries=0,
    )
    instance._throttle = lambda: None
    store = PositionsStore(tmp_path)
    paths = []

    def request(method, url, **kwargs):
        assert store.opening_intents.pending_position(
            scope=instance.execution_scope()
        ) == _position(market_type)
        assert kwargs["headers"]["X-MBX-APIKEY"] == "offline-placeholder"
        path = url.split("?", 1)[0]
        paths.append(path.rsplit("/", 1)[-1])
        if "leverageBracket" in path:
            payload = [
                {
                    "symbol": "BTCUSDT",
                    "brackets": [
                        {
                            "initialLeverage": 20,
                            "notionalFloor": 0,
                            "notionalCap": 100000,
                        }
                    ],
                }
            ]
        elif path.endswith("/leverage"):
            payload = {"leverage": 1}
        else:
            payload = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "status": "FILLED",
                "executedQty": "0.001",
                "avgPrice": "50000",
                "orderId": "123",
                "clientOrderId": "sait-o-scopeone",
            }
        return Mock(status_code=200, headers={}, json=lambda: payload)

    instance.session.request = request
    try:
        recorded = _submit_durable_open_position(
            instance, _position(market_type), store
        )
        assert store.load_open() == [recorded]
        assert (
            store.opening_intents.pending_position(scope=instance.execution_scope())
            is None
        )
        assert paths == (
            ["leverageBracket", "leverage", "order"]
            if market_type == "futures"
            else ["order"]
        )
    finally:
        instance.session.close()


def test_futures_key_change_after_leverage_never_transmits_order_or_recovery_query(
    tmp_path,
):
    instance = BinanceClient(
        "offline-placeholder",
        "offline-placeholder",
        market_type="futures",
        max_retries=0,
    )
    instance._throttle = lambda: None
    store = PositionsStore(tmp_path)
    scope = instance.execution_scope()
    calls = []

    def request(method, url, **kwargs):
        path = url.split("?", 1)[0].rsplit("/", 1)[-1]
        calls.append(path)
        if path == "leverageBracket":
            payload = [{"symbol": "BTCUSDT", "brackets": [{"initialLeverage": 20}]}]
        else:
            assert path == "leverage"
            instance.api_key = "rotated-offline-placeholder"
            payload = {"leverage": 1}
        return Mock(status_code=200, headers={}, json=lambda: payload)

    instance.session.request = request
    try:
        with pytest.raises(BinanceAPIError, match="scope"):
            _submit_durable_open_position(instance, _position("futures"), store)
        assert calls == ["leverageBracket", "leverage"]
        assert store.opening_intents.pending_position(scope=scope) == _position(
            "futures"
        )
        assert store.load_open() == []
    finally:
        instance.session.close()
