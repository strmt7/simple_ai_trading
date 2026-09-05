from __future__ import annotations

import pytest

from simple_ai_trading.binance_account_validation import account_quantity_rejection
from simple_ai_trading.execution_lifecycle import build_execution_lifecycle_plan
from simple_ai_trading.positions import OpenPosition, PositionsStore
from simple_ai_trading.reconciliation import (
    exchange_exposures_from_account,
    reconcile_account_positions,
)
from simple_ai_trading.types import RuntimeConfig, StrategyConfig


@pytest.mark.parametrize(
    "market_type,account",
    [
        ("futures", {"positions": [{"symbol": "BTCUSDC", "positionAmt": "NaN"}]}),
        ("spot", {"balances": [{"asset": "BTC", "free": "NaN", "locked": "0"}]}),
        ("spot", {"balances": [{"asset": "BTC", "free": "-1", "locked": "1"}]}),
        ("futures", {"positions": [{"symbol": "BTCUSDC", "positionAmt": "0"}] * 2}),
    ],
)
def test_invalid_rows_cannot_establish_flat_account(tmp_path, market_type, account):
    report = reconcile_account_positions(
        account,
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type=market_type),
        PositionsStore(tmp_path),
    )
    assert report.ok is False
    assert report.invalid_account_payload_count == 1
    assert any(
        row.reason.startswith("account_payload_invalid:") for row in report.mismatches
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        "",
        "not-a-number",
        "NaN",
        "Infinity",
        "-Infinity",
        [],
        {},
        10**400,
    ],
)
@pytest.mark.parametrize("market_type", ["spot", "futures"])
def test_missing_or_invalid_quantity_is_never_zero(value, market_type):
    account = (
        {"positions": [{"symbol": "BTCUSDC", "positionAmt": value}]}
        if market_type == "futures"
        else {"balances": [{"asset": "BTC", "free": "0", "locked": value}]}
    )
    assert account_quantity_rejection(account, market_type) is not None
    with pytest.raises(ValueError):
        exchange_exposures_from_account(
            account, RuntimeConfig(market_type=market_type), []
        )


@pytest.mark.parametrize(
    "rows",
    [
        [None],
        [{}],
        [{"symbol": " BTCUSDC", "positionAmt": "0"}],
        [{"symbol": "BTCUSDC", "positionAmt": "0", "positionSide": None}],
        [{"symbol": "BTCUSDC", "positionAmt": "-1", "positionSide": "LONG"}],
        [{"symbol": "BTCUSDC", "positionAmt": "1", "positionSide": "SHORT"}],
        [
            {"symbol": "BTCUSDC", "positionAmt": "0", "positionSide": "BOTH"},
            {"symbol": "BTCUSDC", "positionAmt": "0", "positionSide": "LONG"},
        ],
    ],
)
def test_ambiguous_futures_identity_or_mode_is_rejected(rows):
    assert account_quantity_rejection({"positions": rows}, "futures") is not None


@pytest.mark.parametrize(
    "balances",
    [
        [{"asset": "BTC", "free": "0", "locked": "0"}] * 2,
        [{"asset": "BTC", "free": "1e308", "locked": "1e308"}],
        [{"asset": "BTC", "free": "0", "locked": "-1"}],
    ],
)
def test_ambiguous_or_overflowing_spot_balances_are_rejected(balances):
    assert account_quantity_rejection({"balances": balances}, "spot") is not None


def _position(side="LONG", quantity=0.2):
    return OpenPosition(
        id=side,
        symbol="BTCUSDC",
        market_type="futures",
        side=side,
        qty=quantity,
        entry_price=50000.0,
        leverage=1.0,
        opened_at_ms=1,
        notional=quantity * 50000,
        dry_run=False,
        open_client_order_id=f"sait-o-{side}",
        open_exchange_order_id="123",
        exchange_status="FILLED",
    )


def test_valid_hedge_sides_remain_separate_and_reconcile(tmp_path):
    store = PositionsStore(tmp_path)
    store.record_open(_position())
    store.record_open(_position("SHORT", 0.3))
    account = {
        "positions": [
            {
                "symbol": "BTCUSDC",
                "positionSide": "LONG",
                "positionAmt": "0.2",
                "entryPrice": "50000",
            },
            {
                "symbol": "BTCUSDC",
                "positionSide": "SHORT",
                "positionAmt": "-0.3",
                "entryPrice": "50000",
            },
        ]
    }
    report = reconcile_account_positions(
        account,
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="futures"),
        store,
    )
    assert report.ok
    assert report.exchange_exposure_count == 2


def test_invalid_account_is_not_evidence_that_verified_local_position_is_stale(
    tmp_path,
):
    store = PositionsStore(tmp_path)
    position = _position()
    store.record_open(position)
    account = {"positions": [{"symbol": "BTCUSDC", "positionAmt": "NaN"}]}
    report = reconcile_account_positions(
        account, RuntimeConfig(symbol="BTCUSDC", market_type="futures"), store
    )
    assert not report.ok
    assert report.invalid_account_payload_count == 1
    assert report.stale_local_position_count == 0
    assert store.load_open() == [position]
    assert [row.reason for row in report.mismatches] == [
        "account_payload_invalid:futures_position_quantity_invalid"
    ]


@pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), -1, True, None])
def test_invalid_tolerance_cannot_bypass_reconciliation(tmp_path, tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        reconcile_account_positions(
            {"positions": []},
            RuntimeConfig(market_type="futures"),
            PositionsStore(tmp_path),
            quantity_tolerance=tolerance,
        )


def test_valid_flat_rows_and_unknown_market_type():
    assert (
        account_quantity_rejection(
            {"positions": [{"symbol": "BTCUSDC", "positionAmt": "0"}]}, "futures"
        )
        is None
    )
    assert (
        account_quantity_rejection(
            {"balances": [{"asset": "BTC", "free": "0", "locked": "0"}]}, "spot"
        )
        is None
    )
    assert account_quantity_rejection({}, "other") == "account_market_type_invalid"


@pytest.mark.parametrize("market_type", ["spot", "futures"])
def test_invalid_account_blocks_lifecycle_that_accepts_valid_flat_account(
    tmp_path, market_type
):
    store = PositionsStore(tmp_path)
    runtime = RuntimeConfig(
        symbol="BTCUSDC",
        symbols=("BTCUSDC", "ETHUSDC", "SOLUSDC"),
        market_type=market_type,
        dry_run=False,
        testnet=True,
        managed_usdc=1000.0,
        api_key="offline-placeholder",
        api_secret="offline-placeholder",
    )
    valid = {"positions": []} if market_type == "futures" else {"balances": []}
    invalid = (
        {"positions": [{"symbol": "BTCUSDC", "positionAmt": "NaN"}]}
        if market_type == "futures"
        else {"balances": [{"asset": "BTC", "free": "NaN", "locked": "0"}]}
    )

    def plan(account):
        return build_execution_lifecycle_plan(
            runtime,
            StrategyConfig(),
            store,
            action="open",
            effective_dry_run=False,
            reconciliation=reconcile_account_positions(account, runtime, store),
            require_api_budget_headroom=False,
        )

    baseline = plan(valid)
    assert baseline.can_open, [
        (check.label, check.detail)
        for check in baseline.checks
        if check.status == "block"
    ]
    blocked = plan(invalid)
    assert not blocked.can_open
    assert [check.label for check in blocked.checks if check.status == "block"] == [
        "reconciliation"
    ]
