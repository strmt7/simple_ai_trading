"""Command integration with real enrolled storage and an offline venue ledger."""

import argparse
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json

import pytest

from simple_ai_trading import cli
from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from simple_ai_trading.binance_spot_commissions import (
    SpotCommissionComponent,
    SpotCommissionDiscount,
    SpotCommissionSchedule,
)
from simple_ai_trading.binance_spot_roundtrip_cli import (
    LEASE_ENV,
    _sale_amount,
    _quote_amount,
    run_spot_roundtrip,
)
from simple_ai_trading.positions import PositionsStore
from simple_ai_trading.types import RuntimeConfig


SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-command-key"
)
ZERO = SpotCommissionComponent(*(Fraction(0) for _ in range(4)))
SCHEDULE = SpotCommissionSchedule(
    "BTCUSDC",
    SpotCommissionComponent(Fraction(0), Fraction(1, 1000), Fraction(0), Fraction(0)),
    ZERO,
    ZERO,
    SpotCommissionDiscount(False, False, None, None),
)
INFO = {
    "symbol": "BTCUSDC",
    "status": "TRADING",
    "baseAsset": "BTC",
    "quoteAsset": "USDC",
    "orderTypes": ["MARKET"],
    "quoteOrderQtyMarketAllowed": True,
    "quoteAssetPrecision": 8,
    "filters": [
        {
            "filterType": "LOT_SIZE",
            "minQty": "0.001",
            "maxQty": "100",
            "stepSize": "0.001",
        }
    ],
}


class Venue:
    def __init__(self):
        self.base, self.quote = Decimal("10"), Decimal("10000")
        self.orders, self.clients = {}, []
        self.writes, self.reads = [], []
        self.raise_leg, self.failure, self.partial, self.missing_fee = (
            None,
            None,
            False,
            False,
        )
        self.metadata_hook, self.metadata_reads = None, 0
        self.info, self.schedule = deepcopy(INFO), SCHEDULE

    def client(self, runtime):
        venue = self

        class Session:
            closed = False

            def close(self):
                self.closed = True

        class Client:
            session = Session()

            def execution_scope(self):
                return SCOPE

            def ensure_symbol(self, symbol):
                assert symbol == "BTCUSDC"
                venue.metadata_reads += 1
                if venue.metadata_hook:
                    venue.metadata_hook()
                return deepcopy(venue.info)

            def get_account(self, *, expected_scope):
                assert expected_scope == SCOPE
                venue.reads.append("account")
                return {
                    "canTrade": True,
                    "balances": [
                        {"asset": "BTC", "free": str(venue.base), "locked": "0"},
                        {"asset": "USDC", "free": str(venue.quote), "locked": "0"},
                    ],
                }

            def get_spot_commission_schedule(self, symbol, *, expected_scope):
                assert (symbol, expected_scope) == ("BTCUSDC", SCOPE)
                venue.reads.append("commission")
                return venue.schedule

            def place_spot_quote_buy(
                self,
                symbol,
                quote_quantity,
                *,
                dry_run,
                client_order_id,
                expected_scope,
            ):
                assert (symbol, dry_run, expected_scope) == ("BTCUSDC", False, SCOPE)
                return venue.fill("BUY", quote_quantity, client_order_id)

            def place_order(
                self,
                symbol,
                side,
                quantity,
                *,
                dry_run,
                client_order_id,
                expected_scope,
            ):
                assert (symbol, side, dry_run, expected_scope) == (
                    "BTCUSDC",
                    "SELL",
                    False,
                    SCOPE,
                )
                return venue.fill(side, f"{quantity:.8f}", client_order_id)

            def get_order(
                self,
                symbol,
                *,
                orig_client_order_id,
                expected_scope,
                expected_order_binding,
            ):
                assert (symbol, expected_scope) == ("BTCUSDC", SCOPE)
                venue.reads.append("order")
                result = deepcopy(venue.orders[orig_client_order_id])
                expected_order_binding.validate(result)
                result.pop("fills")
                return result

            def get_order_trades(self, symbol, *, order_id, expected_scope):
                assert (symbol, expected_scope) == ("BTCUSDC", SCOPE)
                venue.reads.append("trades")
                order = next(
                    row for row in venue.orders.values() if row["orderId"] == order_id
                )
                return [
                    {
                        **row,
                        "id": row["tradeId"],
                        "symbol": symbol,
                        "orderId": order_id,
                        "isBuyer": order["side"] == "BUY",
                    }
                    for row in order["fills"]
                ]

        client = Client()
        venue.clients.append(client)
        return client

    def fill(self, side, amount, client_id):
        assert client_id not in self.orders, "a native write must never retry its ID"
        with localcontext() as context:
            context.prec = 128
            principal = Decimal(amount)
            quantity = principal / 100 if side == "BUY" else principal
            executed = quantity / 2 if self.partial else quantity
            cash = executed * 100
            fee = executed / 1000 if side == "BUY" else cash / 1000
            if side == "BUY":
                self.quote -= cash
                self.base += executed - fee
            else:
                self.base -= executed
                self.quote += cash - fee
        native_id = len(self.orders) + 1
        fill = {
            "tradeId": native_id,
            "qty": str(executed),
            "price": "100",
            "quoteQty": str(cash),
            "commission": str(fee),
            "commissionAsset": "BTC" if side == "BUY" else "USDC",
        }
        if self.missing_fee:
            fill.pop("commission")
        result = {
            "symbol": "BTCUSDC",
            "side": side,
            "type": "MARKET",
            "status": "PARTIALLY_FILLED" if self.partial else "FILLED",
            "orderId": native_id,
            "clientOrderId": client_id,
            "origQty": str(quantity),
            "executedQty": str(executed),
            "cummulativeQuoteQty": str(cash),
            "fills": [fill],
            **({"origQuoteOrderQty": amount} if side == "BUY" else {}),
        }
        self.orders[client_id] = deepcopy(result)
        self.writes.append((side, amount, client_id))
        if self.raise_leg == len(self.writes):
            raise self.failure or BinanceAPIError("provider-text-must-not-be-printed")
        return result


@pytest.fixture
def context(tmp_path, monkeypatch):
    from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage

    control = BinanceRuntimeControl(tmp_path / "binance_runtime_control.sqlite3")
    control.provision_storage()
    BinanceExecutionStorage(
        PositionsStore(tmp_path).opening_intents, control
    ).provision(scope=SCOPE)
    # The constructed fixture has no prior owned/native obligations.
    assert control.complete_stale_stop(exposure_closed=True)
    lease = control.acquire()
    monkeypatch.setenv(LEASE_ENV, lease)
    runtime = RuntimeConfig(
        market_type="spot",
        testnet=True,
        api_key="offline-command-key",
        api_secret="offline-command-secret",
    )
    venue = Venue()
    return tmp_path, control, lease, runtime, venue


def command(context, action="execute", **options):
    root, _, _, runtime, venue = context
    values = dict(
        action=action,
        data_dir=root,
        yes=True,
        mode="auto",
        quantity=None,
        quote_budget=None,
        operation_id="a" * 29,
    )
    if action == "execute":
        values["quote_budget"] = "100"
    values.update(options)
    return run_spot_roundtrip(
        argparse.Namespace(**values), runtime=runtime, client_factory=venue.client
    )


@pytest.mark.parametrize("mode", ["buy-sell", "sell-buy"])
def test_both_directions_use_exact_principals_preserve_foreign_inventory_and_close_session(
    context, mode
):
    _, control, lease, _, venue = context
    kwargs = {} if mode == "buy-sell" else {"quote_budget": None, "quantity": "1"}
    code, payload = command(context, **kwargs)
    assert code == 0 and payload["phase"] == "COMPLETE"
    assert (
        payload["flat_exposure_qualified"] is False
        and payload["profit_qualified"] is False
    )
    assert payload["mode"] == mode
    assert payload["net_asset_movements"] == (
        {"BTC": "0", "USDC": "-1999/10000"}
        if mode == "buy-sell"
        else {"BTC": "-1999/1000000", "USDC": "0"}
    )
    if mode == "buy-sell":
        assert venue.base == 10 and venue.writes[1][1] == "0.99900000"
    else:
        assert venue.writes[1][1] == "99.9" and venue.quote == 10000
    assert all(client.session.closed for client in venue.clients)
    snapshot = control.snapshot()
    assert snapshot.state == "running" and snapshot.lease_id == lease
    assert lease not in json.dumps(payload)


@pytest.mark.parametrize("leg", [1, 2])
def test_lost_reply_reopens_same_operation_without_duplicate_write(context, leg):
    _, _, _, _, venue = context
    venue.raise_leg = leg
    code, failed = command(context)
    assert code == 2 and failed["phase"] == (
        "FIRST_UNKNOWN" if leg == 1 else "SECOND_UNKNOWN"
    )
    assert (
        failed["net_asset_movements"] is None
        and failed["asset_movements_complete"] is False
    )
    assert "provider-text" not in json.dumps(failed)
    assert all(client.session.closed for client in venue.clients)
    code, blocked = command(context, operation_id="b" * 29)
    assert code == 2 and len(venue.writes) == leg
    venue.raise_leg = None
    code, recovered = command(context, "recover")
    assert code == 0 and recovered["phase"] == (
        "FIRST_TERMINAL" if leg == 1 else "SECOND_TERMINAL"
    )
    assert len(venue.writes) == leg
    code, complete = command(context, "continue")
    assert code == 0 and complete["phase"] == "COMPLETE"
    assert len(venue.writes) == 2
    for action in ("complete", "continue", "recover"):
        assert command(context, action)[0] == 0
    assert len(venue.writes) == 2 and venue.base == 10
    assert all(client.session.closed for client in venue.clients)


def test_keyboard_cancellation_preserves_unknown_and_closes_client(context):
    _, _, _, _, venue = context
    venue.raise_leg, venue.failure = 1, KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        command(context)
    assert venue.clients[-1].session.closed
    assert (
        PositionsStore(context[0]).opening_intents.entry_block_reason()
        == "unresolved_spot_roundtrip=1"
    )
    venue.raise_leg = None
    assert command(context, "recover")[1]["phase"] == "FIRST_TERMINAL"


@pytest.mark.parametrize("mode", ["buy-sell", "sell-buy"])
def test_stop_between_legs_only_allows_owned_sale(context, mode):
    _, control, _, _, venue = context
    venue.metadata_hook = lambda: (
        control.request_stop(reason="constructed-between-leg-stop")
        if venue.metadata_reads == 2
        else None
    )
    kwargs = {} if mode == "buy-sell" else {"quote_budget": None, "quantity": "1"}
    code, payload = command(context, **kwargs)
    assert code == (0 if mode == "buy-sell" else 2)
    assert len(venue.writes) == (2 if mode == "buy-sell" else 1)
    assert control.snapshot().state == "stop_requested"
    assert all(client.session.closed for client in venue.clients)


def test_recovery_and_owned_sale_do_not_require_an_opening_lease(context, monkeypatch):
    _, control, _, _, venue = context
    venue.raise_leg = 1
    assert command(context)[0] == 2
    monkeypatch.delenv(LEASE_ENV)
    control.request_stop(reason="constructed-stop")
    assert command(context, "recover")[0] == 0
    venue.raise_leg = None
    assert command(context, "continue")[1]["phase"] == "COMPLETE"
    assert len(venue.writes) == 2


@pytest.mark.parametrize(
    "change",
    ["none", "both", "float", "zero", "mode", "lot", "quote_precision", "lease"],
)
def test_invalid_command_inputs_send_no_native_write(context, monkeypatch, change):
    _, _, _, _, venue = context
    values = {}
    if change == "none":
        values = {"quote_budget": None}
    elif change == "both":
        values = {"quantity": "1"}
    elif change == "float":
        values = {"quote_budget": 100.0}
    elif change == "zero":
        values = {"quote_budget": "0"}
    elif change == "mode":
        values = {"mode": "sell-buy"}
    elif change == "lot":
        values = {"quote_budget": None, "quantity": "0.0001"}
    elif change == "quote_precision":
        values = {"quote_budget": "100.000000001"}
    else:
        monkeypatch.delenv(LEASE_ENV)
    assert command(context, **values)[0] == 2
    assert venue.writes == [] and venue.reads == []
    assert all(client.session.closed for client in venue.clients)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quoteOrderQtyMarketAllowed", False),
        ("quoteAssetPrecision", True),
        ("baseAsset", "ETH"),
        ("orderTypes", ["LIMIT"]),
        ("filters", []),
    ],
)
def test_instrument_rejection_precedes_signed_account_access(context, field, value):
    venue = context[-1]
    venue.info[field] = value
    assert command(context)[0] == 2
    assert venue.writes == [] and venue.reads == []
    assert venue.clients[-1].session.closed


def test_eligible_external_fee_rejects_before_any_order(context):
    venue = context[-1]
    venue.schedule = replace(
        SCHEDULE, discount=SpotCommissionDiscount(True, True, "BNB", Fraction(1, 4))
    )
    assert command(context)[0] == 2
    assert venue.writes == [] and venue.clients[-1].session.closed


@pytest.mark.parametrize("missing_fee", [False, True])
def test_partial_or_fee_incomplete_first_order_never_sells_existing_base(
    context, missing_fee
):
    venue = context[-1]
    venue.missing_fee, venue.partial = missing_fee, not missing_fee
    code, payload = command(context)
    assert code == 2 and payload["phase"] == "FIRST_UNKNOWN"
    assert len(venue.writes) == 1
    assert venue.base > 10
    assert command(context, "continue")[0] == 2
    assert len(venue.writes) == 1


def test_local_status_creates_no_files_calls_no_client_and_needs_no_credentials(
    tmp_path,
):
    args = argparse.Namespace(action="status", data_dir=tmp_path / "absent")
    code, payload = run_spot_roundtrip(
        args,
        runtime=None,
        client_factory=lambda _: pytest.fail("no client for local status"),
    )
    assert code == 0 and payload["control_available"] is False
    assert (
        payload["execution_store_exists"] is False
        and payload["new_exposure_authorized"] is False
    )
    assert not args.data_dir.exists()


def test_missing_enrollment_never_bootstraps_from_execute(context):
    root, control, _, _, venue = context
    control.enrollment_path.unlink()
    code, payload = command(context)
    assert code == 2 and payload["error_kind"] == "RuntimeControlError"
    assert venue.writes == [] and venue.clients == []
    assert not control.enrollment_path.exists()


def test_missing_operation_is_not_found_proof_or_new_namespace_authority(context):
    root, _, _, _, venue = context
    original = PositionsStore(root).opening_intents.path.read_bytes()
    assert command(context, "recover")[0] == 2
    assert venue.writes == [] and venue.reads == []
    assert PositionsStore(root).opening_intents.path.read_bytes() == original
    assert venue.clients[-1].session.closed


@pytest.mark.parametrize("damage", ["missing", "replacement", "truncated", "marker"])
def test_unknown_journal_loss_cannot_authorize_another_native_buy(context, damage):
    from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage

    root, control, _, _, venue = context
    venue.raise_leg = 1
    assert command(context)[1]["phase"] == "FIRST_UNKNOWN"
    storage = BinanceExecutionStorage(PositionsStore(root).opening_intents, control)
    if damage in {"missing", "replacement"}:
        original = storage.journal.path.read_bytes()
        storage.journal.path.rename(root / "retained.sqlite3")
        if damage == "replacement":
            storage.journal.path.write_bytes(original)
    elif damage == "truncated":
        storage.journal.path.write_bytes(b"")
    else:
        storage.enrollment_path.unlink()
    reads = len(venue.reads)
    metadata = venue.metadata_reads
    venue.raise_leg = None
    code, _ = command(context, operation_id="b" * 29)
    assert code == 2 and len(venue.writes) == 1
    assert len(venue.reads) == reads and venue.metadata_reads == metadata
    assert all(client.session.closed for client in venue.clients)


def test_missing_execution_marker_preserves_recovery_and_receipt_owned_sale(context):
    from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage

    root, control, _, _, venue = context
    venue.raise_leg = 1
    assert command(context)[1]["phase"] == "FIRST_UNKNOWN"
    BinanceExecutionStorage(
        PositionsStore(root).opening_intents, control
    ).enrollment_path.unlink()
    venue.raise_leg = None
    assert command(context, "recover")[1]["phase"] == "FIRST_TERMINAL"
    assert command(context, "continue")[1]["phase"] == "COMPLETE"
    assert [item[0] for item in venue.writes] == ["BUY", "SELL"]


def test_missing_execution_marker_blocks_sell_buy_acquisition(context):
    from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage

    root, control, _, _, venue = context
    venue.raise_leg = 1
    first = command(context, quote_budget=None, quantity="1", mode="sell-buy")
    assert first[1]["phase"] == "FIRST_UNKNOWN"
    venue.raise_leg = None
    assert command(context, "recover")[1]["phase"] == "FIRST_TERMINAL"
    BinanceExecutionStorage(
        PositionsStore(root).opening_intents, control
    ).enrollment_path.unlink()
    reads = len(venue.reads)
    assert command(context, "continue")[0] == 2
    assert [item[0] for item in venue.writes] == ["SELL"]
    assert len(venue.reads) == reads


def test_explicit_storage_actions_are_local_and_leave_stop_latched(context, tmp_path):
    root = tmp_path / "fresh"
    control = BinanceRuntimeControl(root / "binance_runtime_control.sqlite3")
    control.provision_storage()
    fresh = (root, control, None, context[3], context[4])
    before = control.snapshot()
    inspected = command(fresh, "inspect-storage", operation_id=None)
    assert inspected[0] == 0 and inspected[1]["journal_exists"] is False
    assert context[4].clients == []
    result = command(fresh, "provision-storage", operation_id=None)
    assert result[0] == 0 and result[1]["enrollment_state"] == "ready"
    assert control.snapshot() == before
    assert context[4].reads == [] and context[4].writes == []
    assert context[4].metadata_reads == 0
    assert len(context[4].clients) == 1 and context[4].clients[0].session.closed


def test_lot_and_quote_rounding_cannot_enlarge_receipts():
    info = deepcopy(INFO)
    info["filters"] = [
        {"filterType": kind, "minQty": "0", "maxQty": "100", "stepSize": step}
        for kind, step in (("LOT_SIZE", "0.03"), ("MARKET_LOT_SIZE", "0.02"))
    ]
    assert _sale_amount(info, Fraction("0.1"), exact=False) == "0.06"
    assert _quote_amount(INFO, Fraction("1.000000009"), exact=False) == "1"


def test_public_cli_uses_real_command_gateway_and_never_prints_lease(
    context, monkeypatch, capsys
):
    root, _, lease, runtime, venue = context
    monkeypatch.setattr(cli, "load_runtime", lambda: runtime)
    monkeypatch.setattr(cli, "_build_client", venue.client)
    args = cli._parse_args(
        ["spot-roundtrip", "--data-dir", str(root), "--quote-budget", "100", "--yes"]
    )
    assert cli.command_spot_roundtrip(args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["phase"] == "COMPLETE"
    assert not output.err and lease not in output.out
    assert runtime.api_key not in output.out and runtime.api_secret not in output.out


def test_repeated_commands_close_every_owned_session_and_preserve_external_lease(
    context,
):
    _, control, lease, _, venue = context
    for index in range(10):
        assert command(context, operation_id=f"{index + 1:029x}")[0] == 0
    assert len(venue.clients) == 10
    assert all(client.session.closed for client in venue.clients)
    assert control.snapshot().lease_id == lease
    assert len(venue.writes) == 20


def test_complete_command_uses_real_client_methods_with_only_native_http_stubbed(
    context, monkeypatch
):
    from unittest.mock import Mock
    from simple_ai_trading.api import BinanceClient

    root, _, _, runtime, venue = context
    client = BinanceClient(
        runtime.api_key, runtime.api_secret, market_type="spot", testnet=True
    )
    monkeypatch.setattr(
        client.session, "request", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(client.session, "close", Mock(wraps=client.session.close))
    calls = []

    def transport(method, path, params=None, **kwargs):
        calls.append((method, path))
        if kwargs.get("signed"):
            assert kwargs["expected_scope"] == SCOPE
        if path == "/api/v3/exchangeInfo":
            return {"symbols": [deepcopy(INFO)]}
        if path == "/api/v3/account":
            return {
                "canTrade": True,
                "balances": [
                    {"asset": "BTC", "free": str(venue.base), "locked": "0"},
                    {"asset": "USDC", "free": str(venue.quote), "locked": "0"},
                ],
            }
        if path == "/api/v3/account/commission":
            return {
                "symbol": "BTCUSDC",
                **{
                    kind: {
                        "maker": "0",
                        "taker": "0.001" if kind == "standardCommission" else "0",
                        "buyer": "0",
                        "seller": "0",
                    }
                    for kind in (
                        "standardCommission",
                        "taxCommission",
                        "specialCommission",
                    )
                },
                "discount": {
                    "enabledForAccount": False,
                    "enabledForSymbol": False,
                    "discountAsset": "BNB",
                    "discount": "0.75",
                },
            }
        if (method, path) == ("POST", "/api/v3/order"):
            return venue.fill(
                params["side"],
                params.get("quoteOrderQty", params.get("quantity")),
                params["newClientOrderId"],
            )
        pytest.fail("unexpected native transport path")

    monkeypatch.setattr(client, "_request_dict", transport)
    args = argparse.Namespace(
        action="execute",
        data_dir=root,
        yes=True,
        mode="buy-sell",
        quote_budget="100",
        quantity=None,
        operation_id="c" * 29,
    )
    code, payload = run_spot_roundtrip(
        args, runtime=runtime, client_factory=lambda _: client
    )
    assert code == 0 and payload["phase"] == "COMPLETE"
    assert calls.count(("POST", "/api/v3/order")) == 2
    assert venue.base == 10
    client.session.close.assert_called_once()
    client.session.request.assert_not_called()


def test_mutated_client_configuration_returns_no_provider_or_credential_text(
    context, monkeypatch
):
    from unittest.mock import Mock
    from simple_ai_trading import api

    root, _, _, runtime, _ = context
    runtime = replace(runtime)
    # RuntimeConfig normalizes constructor inputs. Exercise a later invalid
    # in-memory mutation explicitly, rather than assuming that normalization failed.
    setattr(runtime, "max_rate_calls_per_minute", "invalid")
    factory = Mock(
        side_effect=AssertionError("no session allocation before scalar validation")
    )
    monkeypatch.setattr(api.requests, "Session", factory)
    args = argparse.Namespace(
        action="execute",
        data_dir=root,
        yes=True,
        quote_budget="100",
        quantity=None,
        mode="auto",
    )
    code, payload = run_spot_roundtrip(
        args, runtime=runtime, client_factory=cli._build_client
    )
    assert code == 2 and payload["error"] == "Binance client configuration is invalid."
    assert runtime.api_key not in json.dumps(
        payload
    ) and runtime.api_secret not in json.dumps(payload)
    factory.assert_not_called()
