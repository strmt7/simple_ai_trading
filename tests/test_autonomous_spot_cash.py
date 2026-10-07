"""Receipt-native cash accounting must not manufacture round-trip profit."""

from dataclasses import asdict, replace
from decimal import localcontext
from fractions import Fraction

import pytest

from simple_ai_trading.autonomous import (
    Decision,
    _apply_close_order,
    _apply_open_order,
    _close_to_trade,
    _submit_durable_close_position,
    _submit_durable_open_position,
    _evaluate_auto_close,
    _entry_gate,
    _loss_budget_guard,
    AutonomousConfig,
)
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.binance_spot_cash import (
    native_spot_cash_mark,
    validate_spot_cash_record,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore, compute_stats
from simple_ai_trading.objective import get_objective
from simple_ai_trading.types import StrategyConfig
from test_autonomous_spot_inventory import Client


def position():
    return OpenPosition(
        id="cash",
        symbol="BTCUSDC",
        market_type="spot",
        side="LONG",
        qty=1,
        entry_price=100,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id="sait-o-cash",
    )


def receipt(side, quantity, *, price="100", fee="0", asset="USDC"):
    return {
        "symbol": "BTCUSDC",
        "side": side,
        "type": "MARKET",
        "status": "FILLED",
        "orderId": "123" if side == "BUY" else "124",
        "clientOrderId": "sait-o-cash" if side == "BUY" else "sait-c-cash",
        "executedQty": quantity,
        "origQty": quantity,
        "fills": [
            {
                "qty": quantity,
                "price": price,
                "commission": fee,
                "commissionAsset": asset,
            }
        ],
    }


def test_flat_price_native_base_fee_is_a_real_cash_loss():
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "receipt-check", clock=lambda: 2),
        receipt("SELL", "0.999"),
        "sait-c-cash",
    )
    assert closed.realized_pnl == pytest.approx(-0.1)


def test_native_entry_cash_loss_triggers_open_position_stop() -> None:
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    should_close, reason = _evaluate_auto_close(
        opened,
        100,
        AutonomousConfig(min_unrealized_close_pct=-0.0005),
        StrategyConfig(),
    )
    assert should_close is True
    assert reason.startswith("auto-stop-loss")


def test_native_entry_cash_loss_reaches_daily_budget(tmp_path) -> None:
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    guard = _loss_budget_guard(
        store,
        100,
        replace(StrategyConfig(), max_daily_loss_pct=0.0005),
        AutonomousConfig(starting_reference_cash=100),
        now_ms_value=2000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )
    assert guard.allowed is False
    assert guard.force_close is True
    assert guard.daily_loss == pytest.approx(0.001)


@pytest.mark.parametrize(
    "asset,fee,expected",
    [
        ("BTC", "0.001", Fraction(-1, 10)),
        ("USDC", "0.4", Fraction(-2, 5)),
        ("BNB", "0", Fraction(0)),
    ],
)
def test_cash_mark_uses_actual_cost_not_modeled_entry_fee(asset, fee, expected) -> None:
    opened = _apply_open_order(
        replace(position(), entry_fees=88), receipt("BUY", "1", fee=fee, asset=asset)
    )
    value = native_spot_cash_mark(opened, 100)
    assert value is not None
    assert value.quote == "USDC"
    assert value.pnl == expected
    assert value.return_fraction == expected / value.entry_cost


@pytest.mark.parametrize(
    "mark",
    [True, 0, -1, float("nan"), float("inf"), "100"],
    ids=["bool", "zero", "negative", "nan", "inf", "string"],
)
def test_native_cash_mark_rejects_invalid_marks(mark) -> None:
    opened = _apply_open_order(position(), receipt("BUY", "1"))
    with pytest.raises(OpenIntentError, match="finite and positive"):
        native_spot_cash_mark(opened, mark)


def test_cash_mark_partial_allocation_conserves_entry_cost(tmp_path) -> None:
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    order = receipt("SELL", "0.1")
    order.update(status="PARTIALLY_FILLED", origQty="0.999")
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "partial"), order, "sait-c-cash"
    )
    store.record_close_result(opened, closed)
    remaining = PositionsStore(tmp_path).load_open(strict=True)[0]
    with localcontext() as context:
        context.prec = 3
        mark = native_spot_cash_mark(remaining, 100)
    assert mark.entry_cost == Fraction(100) * Fraction(899, 999)
    assert float(mark.pnl) + closed.realized_pnl == pytest.approx(-0.1)
    legacy_stats = compute_stats(store, mark_price=100)
    cash_stats = compute_stats(store, mark_price=100, include_native_entry_costs=True)
    assert legacy_stats.unrealized_pnl == 0
    assert cash_stats.unrealized_pnl == float(mark.pnl)


def test_unvalued_fee_blocks_admission_not_reduction_template(tmp_path) -> None:
    opened = _apply_open_order(position(), receipt("BUY", "1", fee="0.1", asset="BNB"))
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    guard = _loss_budget_guard(
        store,
        100,
        StrategyConfig(),
        AutonomousConfig(),
        now_ms_value=2000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )
    assert guard.allowed is False
    assert guard.force_close is False
    assert guard.reason == "cash-valuation-fees-unqualified"
    # An owned reduction does not depend on a guessed third-asset valuation.
    template = _close_to_trade(opened, 100, "operator-stop")
    assert template.qty == opened.qty
    assert template.spot_entry_cash_receipt == opened.spot_entry_cash_receipt


@pytest.mark.parametrize(
    "symbol,market_type,mark",
    [
        ("ETHUSDC", "spot", 100),
        ("BTCUSDC", "futures", 100),
        ("", "", 100),
        ("BTCUSDC", "spot", None),
    ],
)
def test_native_guard_requires_exact_instrument_mark(
    tmp_path, symbol, market_type, mark
) -> None:
    store = PositionsStore(tmp_path)
    store.record_open(_apply_open_order(position(), receipt("BUY", "1")))
    guard = _loss_budget_guard(
        store,
        mark,
        StrategyConfig(),
        AutonomousConfig(),
        now_ms_value=2000,
        mark_symbol=symbol,
        mark_market_type=market_type,
    )
    assert guard.allowed is False
    assert guard.force_close is False
    assert guard.reason == "cash-valuation-instrument-unqualified"


def test_guard_uses_one_coherent_snapshot(tmp_path, monkeypatch) -> None:
    store = PositionsStore(tmp_path)
    store.record_open(_apply_open_order(position(), receipt("BUY", "1")))
    original, calls = store.load_snapshot, []

    def snapshot():
        calls.append(True)
        return original()

    monkeypatch.setattr(store, "load_snapshot", snapshot)
    guard = _loss_budget_guard(
        store,
        100,
        StrategyConfig(),
        AutonomousConfig(),
        now_ms_value=2000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )
    assert guard.allowed is True
    assert calls == [True]


def test_entry_fee_prevents_premature_take_profit() -> None:
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    assert opened.unrealized_pnl_pct(100.1) == pytest.approx(0.001)
    should_close, _ = _evaluate_auto_close(
        opened,
        100.1,
        AutonomousConfig(max_unrealized_close_pct=0.0005),
        StrategyConfig(),
    )
    assert should_close is False


def test_native_guard_rejects_mixed_quote_realized_cash(tmp_path) -> None:
    store = PositionsStore(tmp_path)
    legacy = replace(position(), id="legacy", symbol="ETHUSDT")
    store.record_open(legacy)
    store.record_close(_close_to_trade(legacy, 100, "legacy"))
    store.record_open(_apply_open_order(position(), receipt("BUY", "1")))
    guard = _loss_budget_guard(
        store,
        100,
        StrategyConfig(),
        AutonomousConfig(),
        now_ms_value=2000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )
    assert guard.allowed is False
    assert guard.force_close is False
    assert guard.reason == "cash-valuation-quote-unqualified"


def test_native_guard_never_values_other_instrument_with_current_mark(tmp_path) -> None:
    store = PositionsStore(tmp_path)
    store.record_open(replace(position(), id="legacy", symbol="ETHUSDC"))
    store.record_open(_apply_open_order(position(), receipt("BUY", "1")))
    guard = _loss_budget_guard(
        store,
        100,
        StrategyConfig(),
        AutonomousConfig(),
        now_ms_value=2000,
        mark_symbol="BTCUSDC",
        mark_market_type="spot",
    )
    assert guard.allowed is False
    assert guard.force_close is False
    assert guard.reason == "cash-valuation-instrument-unqualified"


@pytest.mark.parametrize("reference", [100, 0.5])
def test_entry_gate_drawdown_includes_native_entry_cost(tmp_path, reference) -> None:
    store = PositionsStore(tmp_path)
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    store.record_open(opened)
    gate = _entry_gate(
        store,
        Decision(side="LONG", confidence=0.9, mark_price=100),
        StrategyConfig(),
        AutonomousConfig(starting_reference_cash=reference),
        get_objective("default"),
        now_ms_value=2000,
        symbol="BTCUSDC",
        market_type="spot",
    )
    assert gate.allowed is False
    assert gate.drawdown == pytest.approx(0.1 / reference)


def test_opt_in_cash_stats_preserve_legacy_mark_behavior(tmp_path) -> None:
    store = PositionsStore(tmp_path)
    store.record_open(position())
    assert compute_stats(store, mark_price=101, include_native_entry_costs=True) == (
        compute_stats(store, mark_price=101)
    )


@pytest.mark.parametrize(
    "entry_asset,entry_fee,net,expected",
    [
        ("BTC", "0.001", "0.999", -0.3),
        ("USDC", "0.4", "1", -0.6),
        ("BNB", "0", "1", -0.2),
    ],
)
def test_native_cash_overrides_modeled_fees(entry_asset, entry_fee, net, expected):
    opened = _apply_open_order(
        replace(position(), entry_fees=88),
        receipt("BUY", "1", fee=entry_fee, asset=entry_asset),
    )
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "cash", fees=88),
        receipt("SELL", net, fee="0.2"),
        "sait-c-cash",
        exit_taker_fee_bps=999,
    )
    assert closed.realized_pnl == pytest.approx(expected)
    assert closed.fees == pytest.approx(-expected)
    assert closed.realized_pnl_pct == pytest.approx(
        expected / (100 + (0.4 if entry_asset == "USDC" else 0))
    )
    validate_spot_cash_record(closed)


def test_partial_cash_allocations_survive_restart_and_ignore_decimal_context(tmp_path):
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    first_order = receipt("SELL", "0.1", fee="0.01")
    first_order.update(status="PARTIALLY_FILLED", origQty="0.999")
    with localcontext() as context:
        context.prec = 3
        first = _apply_close_order(
            _close_to_trade(opened, 100, "partial"), first_order, "sait-c-cash"
        )
    store.record_close_result(opened, first)
    reopened = PositionsStore(tmp_path)
    remaining = reopened.load_open(strict=True)[0]
    assert remaining.qty == 0.899
    assert remaining.spot_entry_cash_receipt == opened.spot_entry_cash_receipt
    second = _apply_close_order(
        _close_to_trade(remaining, 100, "rest"),
        receipt("SELL", "0.899", fee="0.02"),
        "sait-c-cash",
    )
    reopened.record_close_result(remaining, second)
    ledger = PositionsStore(tmp_path).load_ledger(strict=True)
    assert len(ledger) == 2
    assert sum(t.realized_pnl for t in ledger) == pytest.approx(-0.13)
    assert sum(t.fees for t in ledger) == pytest.approx(0.13)
    assert reopened.load_open(strict=True) == []


@pytest.mark.parametrize(
    "change",
    [
        {"realized_pnl": 42},
        {"fees": 0},
        {"realized_pnl_pct": 42},
        {"spot_close_cash_receipt": ""},
        {"spot_entry_cash_receipt": ""},
        {"qty": 0.998},
        {"entry_price": 101},
        {"exit_price": 101},
        {"open_exchange_order_id": "999"},
        {"close_exchange_order_id": "999"},
        {"market_type": "futures"},
        {"dry_run": True},
    ],
)
def test_retained_cash_and_projection_conflicts_reject_before_write(tmp_path, change):
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "cash"), receipt("SELL", "0.999"), "sait-c-cash"
    )
    tampered = replace(closed, **change)
    assert not PositionsStore._valid_closed_entry(asdict(tampered))
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    before = store.open_path.read_bytes()
    with pytest.raises((OpenIntentError, ValueError)):
        store.record_close_result(opened, tampered)
    assert store.open_path.read_bytes() == before
    assert store.load_ledger() == []


@pytest.mark.parametrize(
    "text",
    ["{}", "[]", "null", "invalid", " " * 1_000_001, 1],
    ids=["object", "array", "null", "invalid", "over-budget", "integer"],
)
def test_cash_encoding_is_bounded_and_fail_closed(text):
    opened = _apply_open_order(position(), receipt("BUY", "1"))
    with pytest.raises(OpenIntentError):
        validate_spot_cash_record(replace(opened, spot_entry_cash_receipt=text))


def test_cash_receipt_drops_unknown_fields_and_rejects_noncanonical_duplicates():
    order = receipt("BUY", "1")
    order["unrelatedResponseField"] = "do-not-retain"
    order["fills"][0]["unrelatedFillField"] = "do-not-retain"
    opened = _apply_open_order(position(), order)
    assert "do-not-retain" not in opened.spot_entry_cash_receipt
    corrupted = opened.spot_entry_cash_receipt.replace("{", '{"symbol":"OTHER",', 1)
    with pytest.raises(OpenIntentError):
        validate_spot_cash_record(replace(opened, spot_entry_cash_receipt=corrupted))


@pytest.mark.parametrize(
    "failure",
    ["missing_fills", "missing_fee", "base_fee", "third_fee", "entry_third_fee"],
)
def test_unqualified_cash_after_sell_preserves_durable_unknown(tmp_path, failure):
    class CashClient(Client):
        def place_order(self, symbol, side, quantity, **kwargs):
            result = super().place_order(symbol, side, quantity, **kwargs)
            if side == "SELL":
                result.update(receipt("SELL", str(quantity)))
                result["clientOrderId"] = kwargs["client_order_id"]
                if failure == "missing_fills":
                    del result["fills"]
                elif failure == "missing_fee":
                    del result["fills"][0]["commission"]
                elif failure in {"base_fee", "third_fee"}:
                    result["fills"][0].update(
                        commission="0.001",
                        commissionAsset="BTC" if failure == "base_fee" else "BNB",
                    )
            return result

    client = CashClient(
        receipt(
            "BUY",
            "1",
            fee="0.001" if failure == "entry_third_fee" else "0",
            asset="BNB",
        )
    )
    store = PositionsStore(tmp_path)
    opened = _submit_durable_open_position(client, position(), store)
    with pytest.raises(OpenIntentError):
        _submit_durable_close_position(
            client,
            opened,
            _close_to_trade(opened, 100, "cash"),
            store,
            reduce_only=False,
        )
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open(strict=True) == [opened]
    assert reopened.load_ledger(strict=True) == []
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    )
    with pytest.raises(OpenIntentError):
        _submit_durable_close_position(
            client,
            opened,
            _close_to_trade(opened, 100, "retry"),
            reopened,
            reduce_only=False,
        )
    assert len(client.orders) == 2  # One BUY and one SELL, never a repeated SELL.


def test_positive_native_profit_uses_actual_proceeds_not_modeled_fee_rate():
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "cash"),
        receipt("SELL", "0.999", price="110", fee="0.2"),
        "sait-c-cash",
        exit_taker_fee_bps=float("nan"),
    )
    assert closed.realized_pnl == 9.69  # 109.89 - 0.2 - 100.
    assert closed.realized_pnl_pct == 0.0969
    assert closed.fees == 0.3


@pytest.mark.parametrize(
    "field,value",
    [
        ("exchange_status", "PARTIALLY_FILLED"),
        ("fees", False),
        ("realized_pnl", False),
        ("realized_pnl_pct", False),
        ("close_client_order_id", "sait-c-foreign"),
    ],
)
def test_native_projection_rejects_status_identity_and_boolean_tampering(field, value):
    opened = _apply_open_order(position(), receipt("BUY", "1"))
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "cash"), receipt("SELL", "1"), "sait-c-cash"
    )
    with pytest.raises(OpenIntentError):
        validate_spot_cash_record(replace(closed, **{field: value}))


@pytest.mark.parametrize(
    "failure",
    [
        "negative_fee",
        "missing_fee",
        "missing_order_id",
        "missing_client_id",
        "wrong_symbol",
        "wrong_side",
        "wrong_type",
        "foreign_id",
    ],
)
def test_entry_cash_never_guesses_missing_or_conflicting_native_evidence(failure):
    order = receipt("BUY", "1")
    if failure == "negative_fee":
        order["fills"][0]["commission"] = "-0.001"
    elif failure == "missing_fee":
        del order["fills"][0]["commission"]
    elif failure == "missing_order_id":
        del order["orderId"]
    elif failure == "missing_client_id":
        del order["clientOrderId"]
    else:
        key, value = {
            "wrong_symbol": ("symbol", "ETHUSDC"),
            "wrong_side": ("side", "SELL"),
            "wrong_type": ("type", "LIMIT"),
            "foreign_id": ("clientOrderId", "sait-o-foreign"),
        }[failure]
        order[key] = value
    with pytest.raises(OpenIntentError):
        _apply_open_order(position(), order)


def test_native_cash_cannot_be_dropped_from_same_lot_before_recording(tmp_path):
    opened = _apply_open_order(position(), receipt("BUY", "1"))
    store = PositionsStore(tmp_path)
    store.record_open(opened)
    closed = _apply_close_order(
        _close_to_trade(opened, 100, "cash"), receipt("SELL", "1"), "sait-c-cash"
    )
    stripped = replace(closed, spot_entry_cash_receipt="", spot_close_cash_receipt="")
    with pytest.raises(ValueError, match="original entry"):
        store.record_close_result(opened, stripped)
    assert store.load_open(strict=True) == [opened]
    assert store.load_ledger(strict=True) == []


def test_legacy_gross_net_lot_is_not_automatically_cash_qualified(tmp_path):
    opened = _apply_open_order(
        position(), receipt("BUY", "1", fee="0.001", asset="BTC")
    )
    legacy = replace(opened, spot_entry_cash_receipt="")
    store = PositionsStore(tmp_path)
    store.record_open(legacy)
    assert PositionsStore(tmp_path).load_open(strict=True) == [legacy]
    closed = _apply_close_order(
        _close_to_trade(legacy, 100, "legacy"), receipt("SELL", "0.999"), "sait-c-cash"
    )
    assert closed.spot_entry_cash_receipt == ""
    assert closed.spot_close_cash_receipt == ""
    assert closed.realized_pnl == 0  # Preserved modeled behavior, not native proof.


def test_interrupted_native_close_keeps_cash_record_and_unknown_after_restart(tmp_path):
    client = Client(receipt("BUY", "1", fee="0.001", asset="BTC"))
    store = PositionsStore(tmp_path)
    opened = _submit_durable_open_position(client, position(), store)
    original_record = store.record_close_result

    def interrupted_record(position, trade):
        original_record(position, trade)
        raise RuntimeError("simulated interruption after paired publication")

    store.record_close_result = interrupted_record
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _submit_durable_close_position(
            client,
            opened,
            _close_to_trade(opened, 100, "cash"),
            store,
            reduce_only=False,
        )
    reopened = PositionsStore(tmp_path)
    assert reopened.load_open(strict=True) == []
    ledger = reopened.load_ledger(strict=True)
    assert len(ledger) == 1
    assert ledger[0].realized_pnl == -0.1
    assert (
        reopened.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    )
    with pytest.raises(OpenIntentError):
        _submit_durable_close_position(
            client,
            opened,
            _close_to_trade(opened, 100, "retry"),
            reopened,
            reduce_only=False,
        )
    assert len(client.orders) == 2
