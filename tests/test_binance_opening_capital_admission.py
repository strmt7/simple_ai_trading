"""Real local policy/ledger transactions with a synthetic native transport."""

from dataclasses import replace
import json

import pytest

from simple_ai_trading.autonomous import AutonomousConfig, _loss_budget_guard
from simple_ai_trading.binance_capital_policy import BinanceCapitalPolicyStore
from simple_ai_trading.binance_execution_scope import (
    BINANCE_FUTURES_TESTNET,
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_owned_execution import submit_durable_open
from simple_ai_trading.positions import ClosedTrade, OpenPosition, PositionsStore
from simple_ai_trading.types import StrategyConfig


class NativeClient:
    def __init__(self, market_type="spot"):
        self.market_type = market_type
        self.orders = []

    def execution_scope(self):
        origin = (
            BINANCE_SPOT_TESTNET
            if self.market_type == "spot"
            else BINANCE_FUTURES_TESTNET
        )
        return BinanceExecutionScope.from_api_key(
            origin, self.market_type, "synthetic-policy-client"
        )

    def place_order(self, symbol, side, quantity, **kwargs):
        self.orders.append((symbol, side, quantity))
        return {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "status": "FILLED",
            "orderId": 1,
            "clientOrderId": kwargs["client_order_id"],
            "transactTime": 3000,
            "origQty": str(quantity),
            "executedQty": str(quantity),
            "cummulativeQuoteQty": str(quantity * 100),
            "avgPrice": "100",
            "fills": [
                {
                    "tradeId": 1,
                    "qty": str(quantity),
                    "price": "100",
                    "commission": "0",
                    "commissionAsset": "BTC",
                }
            ],
        }


def policy_document(**changes):
    strategy = StrategyConfig()
    return {
        "quote_asset": "USDC",
        "reference_cash_quote": "1000",
        "maximum_daily_loss_fraction": str(strategy.max_daily_loss_pct),
        "maximum_lifetime_loss_fraction": str(strategy.max_session_loss_pct),
        "maximum_settled_peak_drawdown_fraction": str(strategy.max_drawdown_limit),
        "maximum_consecutive_losses": strategy.max_consecutive_losses,
        **changes,
    }


def requested(market_type="spot"):
    return OpenPosition(
        id="native-open",
        symbol="BTCUSDC",
        market_type=market_type,
        side="LONG",
        qty=1,
        entry_price=100,
        notional=100,
        leverage=1,
        opened_at_ms=3000,
        dry_run=False,
        open_client_order_id="sait-o-native-policy",
    )


def test_missing_policy_blocks_before_unknown_or_transport(
    tmp_path, enrolled_binance_runtime
):
    store, client = PositionsStore(tmp_path), NativeClient()
    authority = enrolled_binance_runtime(store, client.execution_scope())
    with pytest.raises(OpenIntentError, match="activated capital policy"):
        submit_durable_open(client, requested(), store, interlock=authority)
    assert client.orders == []
    assert (
        store.opening_intents.pending_position(scope=client.execution_scope()) is None
    )
    assert not BinanceCapitalPolicyStore(store).enrollment_path.exists()


def test_policy_and_snapshot_are_admitted_inside_unknown_transaction(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, client = PositionsStore(tmp_path), NativeClient()
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    head = BinanceCapitalPolicyStore(store).inspect()["head_sha256"]
    original = requested()

    def forbidden(*args, **kwargs):
        pytest.fail("opening admission reread positions through a nested connection")

    monkeypatch.setattr(PositionsStore, "load_snapshot", forbidden)
    result = submit_durable_open(client, original, store, interlock=authority)
    assert original.capital_policy_sha256 == ""
    assert result.position.capital_policy_sha256 == head
    with store.opening_intents._connect() as connection:
        row = connection.execute(
            "SELECT request_json, state FROM open_intent"
        ).fetchone()
    assert row[1] == "RECORDED"
    assert json.loads(row[0])["position_template"]["capital_policy_sha256"] == head
    assert len(client.orders) == 1


def test_gateway_rejects_loss_added_after_strategy_evaluation(
    tmp_path, enrolled_binance_runtime
):
    store, client = PositionsStore(tmp_path), NativeClient("futures")
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    cfg = AutonomousConfig(dry_run=False, starting_reference_cash=1000)
    assert _loss_budget_guard(
        store,
        100,
        StrategyConfig(),
        cfg,
        now_ms_value=3000,
        mark_symbol="BTCUSDC",
        mark_market_type="futures",
    ).allowed
    store.record_close(
        ClosedTrade(
            id="settled-loss",
            symbol="BTCUSDC",
            market_type="futures",
            side="LONG",
            qty=1,
            entry_price=100,
            exit_price=75,
            leverage=1,
            opened_at_ms=1000,
            closed_at_ms=2000,
            realized_pnl=-25,
            realized_pnl_pct=-0.25,
        )
    )
    with pytest.raises(OpenIntentError, match="session-loss-lockout|drawdown-lockout"):
        submit_durable_open(client, requested("futures"), store, interlock=authority)
    assert client.orders == []
    assert (
        store.opening_intents.pending_position(scope=client.execution_scope()) is None
    )


def test_post_prepare_policy_loss_retains_original_unknown(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, client = PositionsStore(tmp_path), NativeClient()
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    policies = BinanceCapitalPolicyStore(store)
    head = policies.inspect()["head_sha256"]
    original_prepare = BinanceOpenIntentJournal.prepare

    def remove_after_prepare(self, *args, **kwargs):
        original_prepare(self, *args, **kwargs)
        policies.enrollment_path.unlink()

    monkeypatch.setattr(BinanceOpenIntentJournal, "prepare", remove_after_prepare)
    with pytest.raises(OpenIntentError):
        submit_durable_open(client, requested(), store, interlock=authority)
    pending = store.opening_intents.pending_position(scope=client.execution_scope())
    assert pending is not None and pending.capital_policy_sha256 == head
    assert client.orders == []
    assert not policies.enrollment_path.exists()


@pytest.mark.parametrize(
    "change", [{"quote_asset": "USDT"}, {"maximum_daily_loss_fraction": "1e-324"}]
)
def test_unusable_policy_rejects_without_native_attempt(
    tmp_path, enrolled_binance_runtime, change
):
    store, client = PositionsStore(tmp_path), NativeClient()
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document(**change)
    )
    with pytest.raises(OpenIntentError):
        submit_durable_open(client, requested(), store, interlock=authority)
    assert client.orders == []
    assert (
        store.opening_intents.pending_position(scope=client.execution_scope()) is None
    )


def test_new_provenance_preserves_legacy_request_and_recovery_bytes(tmp_path):
    journal, position = (
        BinanceOpenIntentJournal(tmp_path / "legacy.sqlite3"),
        requested(),
    )
    scope = NativeClient().execution_scope()
    payload = json.loads(journal._request(position, scope))
    assert "capital_policy_sha256" not in payload["position_template"]
    journal.prepare(position, scope=scope)
    pending = journal.pending_position(scope=scope)
    assert pending == position and journal._request(pending, scope) == json.dumps(
        payload, sort_keys=True, allow_nan=False
    )
    with pytest.raises(OpenIntentError, match="digest"):
        journal._request(replace(position, capital_policy_sha256="bad"), scope)


@pytest.mark.parametrize("damage", ["delete", "rewrite"])
def test_changed_prepared_identity_cannot_reach_native_transport(
    tmp_path, enrolled_binance_runtime, monkeypatch, damage
):
    store, client = PositionsStore(tmp_path), NativeClient()
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    prepare = BinanceOpenIntentJournal.prepare

    def damage_prepared(self, *args, **kwargs):
        prepare(self, *args, **kwargs)
        with self._connect(write=True) as connection:
            if damage == "delete":
                connection.execute("DELETE FROM open_intent")
            else:
                connection.execute("UPDATE open_intent SET position_id='different'")

    monkeypatch.setattr(BinanceOpenIntentJournal, "prepare", damage_prepared)
    with pytest.raises(OpenIntentError, match="prepared opening intent changed"):
        submit_durable_open(client, requested(), store, interlock=authority)
    assert client.orders == []


def test_cash_loss_after_preparation_blocks_transport_without_erasing_unknown(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, client = PositionsStore(tmp_path), NativeClient("futures")
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    prepare = BinanceOpenIntentJournal.prepare

    def settle_after_prepare(self, *args, **kwargs):
        prepare(self, *args, **kwargs)
        store.record_close(
            ClosedTrade(
                id="new-loss",
                symbol="BTCUSDC",
                market_type="futures",
                side="LONG",
                qty=1,
                entry_price=100,
                exit_price=75,
                leverage=1,
                opened_at_ms=1000,
                closed_at_ms=2000,
                realized_pnl=-25,
                realized_pnl_pct=-0.25,
            )
        )

    monkeypatch.setattr(BinanceOpenIntentJournal, "prepare", settle_after_prepare)
    with pytest.raises(OpenIntentError, match="session-loss-lockout"):
        submit_durable_open(client, requested("futures"), store, interlock=authority)
    assert client.orders == []
    assert (
        store.opening_intents.pending_position(scope=client.execution_scope())
        is not None
    )
    assert len(store.load_ledger(strict=True)) == 1


def test_snapshot_reader_rejects_transaction_for_another_store(tmp_path):
    from simple_ai_trading.position_transactions import position_transaction

    source = PositionsStore(tmp_path / "source")
    foreign = PositionsStore(tmp_path / "foreign")
    with position_transaction(source.opening_intents) as transaction:
        with pytest.raises(ValueError, match="another store"):
            foreign.snapshot_from_transaction(transaction)


def test_policy_evaluation_uses_one_immutable_configuration_snapshot(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    store, client = PositionsStore(tmp_path), NativeClient("futures")
    enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    store.record_close(
        ClosedTrade(
            id="loss",
            symbol="BTCUSDC",
            market_type="futures",
            side="LONG",
            qty=1,
            entry_price=100,
            exit_price=75,
            leverage=1,
            opened_at_ms=1000,
            closed_at_ms=2000,
            realized_pnl=-25,
            realized_pnl_pct=-0.25,
        )
    )
    strategy = StrategyConfig()
    cfg = AutonomousConfig(dry_run=False, starting_reference_cash=1000)
    original = BinanceCapitalPolicyStore.assert_effective

    def change_after_read(self, settings, reference, **kwargs):
        digest = original(self, settings, reference, **kwargs)
        strategy.max_session_loss_pct = 0.5
        cfg.starting_reference_cash = 1_000_000
        return digest

    monkeypatch.setattr(
        BinanceCapitalPolicyStore, "assert_effective", change_after_read
    )
    decision = _loss_budget_guard(
        store,
        100,
        strategy,
        cfg,
        now_ms_value=3000,
        mark_symbol="BTCUSDC",
        mark_market_type="futures",
    )
    assert not decision.allowed and decision.force_close
    assert decision.session_loss == 0.025


@pytest.mark.parametrize(
    "product,cutoff,expected_paths",
    [
        ("spot", 1, []),
        ("futures", 2, ["leverageBracket"]),
        ("futures", 3, ["leverageBracket", "leverage"]),
    ],
)
def test_sdk_rechecks_policy_after_throttle_and_preflight(
    tmp_path, enrolled_binance_runtime, product, cutoff, expected_paths
):
    from types import SimpleNamespace
    from urllib.parse import urlsplit
    from simple_ai_trading.api import BinanceClient

    client = BinanceClient(
        "synthetic-client",
        "synthetic-secret",
        testnet=True,
        market_type=product,
        max_retries=0,
    )
    store = PositionsStore(tmp_path)
    authority = enrolled_binance_runtime(
        store, client.execution_scope(), capital_policy=policy_document()
    )
    policies = BinanceCapitalPolicyStore(store)
    head = policies.inspect()["head_sha256"]
    seen = []
    calls = 0

    def throttle():
        nonlocal calls
        calls += 1
        if calls == cutoff:
            policies.enrollment_path.unlink()

    def request(method, url, **kwargs):
        path = urlsplit(url).path.rsplit("/", 1)[-1]
        seen.append(path)
        assert path != "order", "an unqualified opening reached native transport"
        payload = (
            [
                {
                    "symbol": "BTCUSDC",
                    "brackets": [
                        {"initialLeverage": 20, "notionalCap": 1e9, "notionalFloor": 0}
                    ],
                }
            ]
            if path == "leverageBracket"
            else {"leverage": 1}
        )
        assert set(kwargs) == {"params", "timeout", "headers"}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: payload)

    client._throttle = throttle
    client.session.request = request
    try:
        with pytest.raises(OpenIntentError):
            submit_durable_open(client, requested(product), store, interlock=authority)
        assert seen == expected_paths
        pending = store.opening_intents.pending_position(scope=client.execution_scope())
        assert pending is not None and pending.capital_policy_sha256 == head
    finally:
        client.session.close()


@pytest.mark.parametrize(
    "signed,bound,callback",
    [(False, False, lambda: None), (True, False, lambda: None), (True, True, False)],
)
def test_sdk_admission_requires_a_callable_and_bound_signed_scope(
    signed, bound, callback
):
    from simple_ai_trading.api import BinanceClient, BinanceAPIError

    client = BinanceClient(
        "synthetic-client", "synthetic-secret", testnet=True, max_retries=0
    )
    try:
        with pytest.raises(BinanceAPIError, match="scope"):
            client._request(
                "POST",
                "/api/v3/order",
                signed=signed,
                expected_scope=client.execution_scope() if bound else None,
                before_dispatch=callback,
            )
    finally:
        client.session.close()
