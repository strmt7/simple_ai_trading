from __future__ import annotations

from pathlib import Path

from simple_ai_trading.positions import OpenPosition, PositionsStore
from simple_ai_trading.reconciliation import exchange_exposures_from_account, reconcile_account_positions
from simple_ai_trading.types import RuntimeConfig


def _position(
    *,
    symbol: str = "BTCUSDC",
    side: str = "LONG",
    qty: float = 0.25,
    dry_run: bool = False,
    verified: bool = True,
) -> OpenPosition:
    position_id = f"{symbol}-{side}"
    return OpenPosition(
        id=position_id,
        symbol=symbol,
        market_type="futures",
        side=side,
        qty=qty,
        entry_price=100.0,
        leverage=2.0,
        opened_at_ms=1,
        notional=qty * 100.0,
        dry_run=dry_run,
        open_client_order_id=f"sait-o-{position_id}" if verified and not dry_run else "",
        open_exchange_order_id="12345" if verified and not dry_run else "",
        exchange_status="FILLED" if verified and not dry_run else "local",
    )


def test_reconcile_futures_matches_live_local_position(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(qty=0.5))
    account = {
        "positions": [
            {"symbol": "BTCUSDC", "positionAmt": "0.50000000", "entryPrice": "100", "notional": "50"},
            {"symbol": "ETHUSDC", "positionAmt": "0", "entryPrice": "0"},
        ]
    }

    report = reconcile_account_positions(
        account,
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="futures"),
        store,
    )

    assert report.ok is True
    assert report.local_live_open_count == 1
    assert report.exchange_exposure_count == 1
    assert report.mismatches == []


def test_reconcile_detects_exchange_only_futures_exposure(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    account = {"positions": [{"symbol": "BTCUSDC", "positionAmt": "-0.25", "entryPrice": "100", "notional": "-25"}]}

    report = reconcile_account_positions(
        account,
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="futures"),
        store,
    )

    assert report.ok is False
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.symbol == "BTCUSDC"
    assert mismatch.side == "SHORT"
    assert mismatch.reason == "exchange_exposure_without_local_position"
    assert mismatch.exchange_qty == 0.25


def test_reconcile_detects_local_only_live_position(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(qty=0.1))

    report = reconcile_account_positions(
        {"positions": []},
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="futures"),
        store,
    )

    assert report.ok is False
    assert report.mismatches[0].reason == "local_position_without_exchange_exposure"
    assert report.mismatches[0].local_qty == 0.1


def test_reconcile_rejects_matching_live_position_without_bot_ownership(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(qty=0.5, verified=False))
    account = {
        "positions": [
            {"symbol": "BTCUSDC", "positionAmt": "0.50000000", "entryPrice": "100", "notional": "50"},
        ]
    }

    report = reconcile_account_positions(
        account,
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="futures"),
        store,
    )

    reasons = [mismatch.reason for mismatch in report.mismatches]
    assert report.ok is False
    assert report.unverified_local_position_count == 1
    assert report.stale_local_position_count == 1
    assert "exchange_exposure_without_local_position" in reasons
    assert any(reason.startswith("local_position_without_bot_ownership:") for reason in reasons)


def test_reconcile_ignores_paper_positions_but_warns(tmp_path: Path) -> None:
    store = PositionsStore(root=tmp_path)
    store.record_open(_position(qty=0.1, dry_run=True))

    report = reconcile_account_positions(
        {"positions": []},
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="futures"),
        store,
    )

    assert report.ok is True
    assert report.local_paper_open_count == 1
    assert report.warnings == ["paper_positions_ignored=1"]


def test_spot_account_exposure_uses_base_asset_for_runtime_symbols() -> None:
    exposures = exchange_exposures_from_account(
        {
            "balances": [
                {"asset": "BTC", "free": "0.2", "locked": "0.1"},
                {"asset": "USDC", "free": "1000", "locked": "0"},
                {"asset": "ALT", "free": "10", "locked": "0"},
            ]
        },
        RuntimeConfig(symbol="BTCUSDC", symbols=("BTCUSDC",), market_type="spot", quote_asset="USDC"),
        [],
    )

    assert len(exposures) == 1
    assert exposures[0].symbol == "BTCUSDC"
    assert exposures[0].side == "LONG"
    assert abs(exposures[0].qty - 0.3) < 1e-12
