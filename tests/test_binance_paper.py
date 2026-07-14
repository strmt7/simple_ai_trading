from __future__ import annotations

from decimal import Decimal

import pytest

from simple_ai_trading.binance_paper import BinancePaperBroker
from simple_ai_trading.positions import OpenPosition, PositionsStore


class _BookClient:
    def __init__(self, books: list[dict[str, object]]) -> None:
        self.books = list(books)
        self.calls: list[str] = []

    def get_book_ticker(self, symbol: str) -> dict[str, object]:
        self.calls.append(symbol)
        if not self.books:
            raise RuntimeError("no book response configured")
        return self.books.pop(0)


def _book(*, bid: str, bid_qty: str, ask: str, ask_qty: str) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "bidPrice": bid,
        "bidQty": bid_qty,
        "askPrice": ask,
        "askQty": ask_qty,
    }


def _position(*, quantity: float = 1.0) -> OpenPosition:
    return OpenPosition(
        id="position-1",
        symbol="BTCUSDT",
        market_type="futures",
        side="LONG",
        qty=quantity,
        entry_price=100.0,
        leverage=5.0,
        opened_at_ms=1,
        notional=100.0 * quantity,
        dry_run=True,
    )


def test_broker_opens_and_closes_from_real_bbo_with_net_fees(tmp_path) -> None:
    client = _BookClient(
        [
            _book(bid="99", bid_qty="4", ask="100", ask_qty="1"),
            _book(bid="99", bid_qty="1", ask="100", ask_qty="4"),
        ]
    )
    store = PositionsStore(root=tmp_path / "positions")
    with BinancePaperBroker(
        tmp_path / "paper.duckdb",
        client,  # type: ignore[arg-type]
        market_type="futures",
        taker_fee_bps=10,
    ) as broker:
        opened, open_result = broker.open_position(_position())
        assert opened is not None
        store.record_open(opened)
        assert broker.reconcile_positions(store).can_open is True

        trade, close_result = broker.close_position(opened, reason="test-close")
        assert trade is not None
        store.record_close_result(opened, trade)
        reconciled = broker.reconcile_positions(store)

    assert client.calls == ["BTCUSDT", "BTCUSDT"]
    assert open_result.state == "FILLED"
    assert close_result.state == "FILLED"
    assert opened.paper_open_intent_id
    assert opened.entry_price == 100.0
    assert opened.entry_fees == pytest.approx(0.1)
    assert trade.paper_open_intent_id == opened.paper_open_intent_id
    assert trade.paper_close_intent_id
    assert trade.fees == pytest.approx(0.199)
    assert trade.realized_pnl == pytest.approx(-1.199)
    assert reconciled.ok is True
    assert reconciled.journal.inventory[0].remaining_quantity == 0


def test_partial_close_remains_blocking_until_follow_up_close_fills(tmp_path) -> None:
    client = _BookClient(
        [
            _book(bid="99", bid_qty="4", ask="100", ask_qty="2"),
            _book(bid="99", bid_qty="1", ask="100", ask_qty="4"),
            _book(bid="98", bid_qty="1", ask="99", ask_qty="4"),
        ]
    )
    store = PositionsStore(root=tmp_path / "positions")
    with BinancePaperBroker(
        tmp_path / "paper.duckdb",
        client,  # type: ignore[arg-type]
        market_type="futures",
        taker_fee_bps=10,
    ) as broker:
        opened, _ = broker.open_position(_position(quantity=2.0))
        assert opened is not None
        store.record_open(opened)

        first_trade, first_result = broker.close_position(
            opened, reason="operator-stop"
        )
        assert first_trade is not None
        store.record_close_result(opened, first_trade)
        remaining = store.load_open()[0]
        partial_reconciliation = broker.reconcile_positions(store)

        second_trade, second_result = broker.close_position(
            remaining, reason="operator-stop-retry"
        )
        assert second_trade is not None
        store.record_close_result(remaining, second_trade)
        final_reconciliation = broker.reconcile_positions(store)

    assert first_result.state == "CLOSE_PENDING"
    assert partial_reconciliation.can_open is False
    assert partial_reconciliation.can_close is True
    assert partial_reconciliation.journal.inventory[0].remaining_quantity == Decimal(
        "1.0"
    )
    assert remaining.qty == 1.0
    assert remaining.entry_fees == pytest.approx(0.1)
    assert second_result.state == "FILLED"
    assert final_reconciliation.can_open is True
    assert final_reconciliation.journal.inventory[0].remaining_quantity == 0
    assert store.load_open() == []
    assert len(store.load_ledger()) == 2


def test_reconciliation_rejects_legacy_position_without_journal_proof(tmp_path) -> None:
    store = PositionsStore(root=tmp_path / "positions")
    store.record_open(_position())
    with BinancePaperBroker(
        tmp_path / "paper.duckdb",
        _BookClient([]),  # type: ignore[arg-type]
        market_type="futures",
        taker_fee_bps=10,
    ) as broker:
        report = broker.reconcile_positions(store)

    assert report.ok is False
    assert report.can_open is False
    assert report.can_close is False
    assert report.position_errors == ("paper_position_intent_missing:position-1",)


def test_reconciliation_detects_journal_inventory_missing_from_position_ledger(
    tmp_path,
) -> None:
    client = _BookClient([_book(bid="99", bid_qty="4", ask="100", ask_qty="1")])
    store = PositionsStore(root=tmp_path / "positions")
    with BinancePaperBroker(
        tmp_path / "paper.duckdb",
        client,  # type: ignore[arg-type]
        market_type="futures",
        taker_fee_bps=10,
    ) as broker:
        opened, result = broker.open_position(_position())
        assert opened is not None and result.state == "FILLED"
        report = broker.reconcile_positions(store)

    assert report.ok is False
    assert report.can_open is False
    assert report.can_close is False
    assert len(report.position_errors) == 1
    assert report.position_errors[0].startswith("paper_inventory_without_position:")


def test_broker_rejects_position_from_a_different_market_type(tmp_path) -> None:
    client = _BookClient([_book(bid="99", bid_qty="4", ask="100", ask_qty="1")])
    with BinancePaperBroker(
        tmp_path / "paper.duckdb",
        client,  # type: ignore[arg-type]
        market_type="spot",
        taker_fee_bps=10,
    ) as broker:
        with pytest.raises(ValueError, match="market_type"):
            broker.open_position(_position())

    assert client.calls == []


def test_reconciliation_blocks_mixed_live_and_cross_market_position_rows(
    tmp_path,
) -> None:
    store = PositionsStore(root=tmp_path / "positions")
    futures_paper = _position()
    live_spot = OpenPosition(
        id="live-position",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        leverage=1.0,
        opened_at_ms=1,
        notional=100.0,
        dry_run=False,
    )
    store.record_open(futures_paper)
    store.record_open(live_spot)

    with BinancePaperBroker(
        tmp_path / "paper.duckdb",
        _BookClient([]),  # type: ignore[arg-type]
        market_type="spot",
        taker_fee_bps=10,
    ) as broker:
        report = broker.reconcile_positions(store)

    assert report.can_open is False
    assert report.can_close is False
    assert report.position_errors == (
        "paper_mode_live_position_present:live-position",
        "paper_position_market_mismatch:position-1:futures",
    )
