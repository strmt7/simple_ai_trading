"""Preserve every supplied positive signed residual; no venue or account access."""

from dataclasses import replace
from decimal import Decimal, Subnormal, localcontext

import pytest

from simple_ai_trading.binance_owned_execution import close_trade_template
from simple_ai_trading.positions import OpenPosition, PositionsStore


def stage(root, *, product="futures", side="LONG", quantity=1.0, paper=False):
    store = PositionsStore(root)
    position = OpenPosition(
        id="quantity-owned",
        symbol="BTCUSDT",
        market_type=product,
        side=side,
        qty=quantity,
        entry_price=100.0,
        leverage=1.0,
        opened_at_ms=1,
        notional=quantity * 100.0,
        dry_run=paper,
        open_client_order_id="sait-o-quantityowned",
        open_exchange_order_id="1",
        exchange_status="FILLED",
        entry_fees=1.0,
    )
    store.record_open(position)
    trade = replace(
        close_trade_template(position, 101.0, "quantity test", clock=lambda: 2),
        close_client_order_id="sait-c-quantityowned",
        close_exchange_order_id="2",
        exchange_status="PARTIALLY_FILLED",
    )
    return store, position, trade


@pytest.mark.parametrize(
    "product,side", [("spot", "LONG"), ("futures", "LONG"), ("futures", "SHORT")]
)
@pytest.mark.parametrize(
    "original,executed,residual",
    [(1.0, 0.999999999, "0.000000001"), (1000000.0, 999999.999, "0.001")],
)
def test_signed_residual_is_not_an_epsilon_full_close(
    tmp_path, product, side, original, executed, residual
):
    store, position, trade = stage(
        tmp_path, product=product, side=side, quantity=original
    )
    trade = replace(trade, qty=executed)
    with localcontext() as context:
        context.prec = 2
        context.Emin, context.Emax = -2, 2
        context.traps[Subnormal] = True
        store.record_close_result(position, trade)
    reopened = PositionsStore(tmp_path)
    remaining, ledger = reopened.load_snapshot(strict=True)
    assert len(remaining) == 1
    assert Decimal(str(remaining[0].qty)) == Decimal(residual)
    assert remaining[0].side == side and remaining[0].dry_run is False
    assert remaining[0].entry_fees > 0
    assert ledger == [trade]


@pytest.mark.parametrize(
    "product,side", [("spot", "LONG"), ("futures", "LONG"), ("futures", "SHORT")]
)
def test_exact_full_signed_close_removes_the_lot(tmp_path, product, side):
    store, position, trade = stage(tmp_path, product=product, side=side)
    store.record_close_result(position, replace(trade, exchange_status="FILLED"))
    assert PositionsStore(tmp_path).load_open(strict=True) == []


@pytest.mark.parametrize(
    "quantity", [0, -1, True, float("nan"), float("inf"), "1", None]
)
def test_invalid_signed_close_quantity_cannot_mutate_inventory(tmp_path, quantity):
    store, position, trade = stage(tmp_path)
    before = store.load_snapshot(strict=True)
    with pytest.raises(ValueError, match="finite and positive"):
        store.record_close_result(position, replace(trade, qty=quantity))
    assert store.load_snapshot(strict=True) == before


def test_signed_oversell_rejects_without_tolerance_or_mutation(tmp_path):
    store, position, trade = stage(tmp_path)
    before = store.load_snapshot(strict=True)
    with pytest.raises(ValueError, match="exceeds remaining"):
        store.record_close_result(position, replace(trade, qty=1.0000000001))
    assert store.load_snapshot(strict=True) == before


def test_unrepresentable_signed_remainder_rejects_instead_of_rounding(tmp_path):
    store, position, trade = stage(tmp_path, quantity=0.01)
    before = store.load_snapshot(strict=True)
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        store.record_close_result(position, replace(trade, qty=1e-20))
    assert store.load_snapshot(strict=True) == before


def test_paper_quantity_tolerance_remains_compatible(tmp_path):
    store, position, trade = stage(tmp_path, paper=True)
    store.record_close_result(position, replace(trade, qty=0.999999999))
    assert store.load_open(strict=True) == []
