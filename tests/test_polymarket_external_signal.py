from __future__ import annotations

import ast
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_external_signal import (
    BtcPriceDiscoveryTick,
    PolymarketBtcPriceDiscoveryMonitor,
)


NOW_MS = 1_800_000_000_000


def _tick(
    source: str,
    *,
    sequence: int = 1,
    bid: str = "99999",
    ask: str = "100001",
    event_offset_ms: int = -100,
    receive_offset_ms: int = -50,
) -> BtcPriceDiscoveryTick:
    return BtcPriceDiscoveryTick(
        source=source,
        symbol="BTCUSDT",
        event_time_ms=NOW_MS + event_offset_ms,
        received_at_ms=NOW_MS + receive_offset_ms,
        sequence=sequence,
        bid=Decimal(bid),
        ask=Decimal(ask),
    )


def test_reference_features_are_advisory_and_never_increase_exposure() -> None:
    monitor = PolymarketBtcPriceDiscoveryMonitor()

    decision = monitor.evaluate(
        spot=_tick("BINANCE_SPOT"),
        futures=_tick(
            "BINANCE_USD_M_FUTURES",
            bid="100009",
            ask="100011",
        ),
        observed_at_ms=NOW_MS,
    )

    assert decision.action == "preserve"
    assert decision.maximum_size_multiplier == Decimal("1")
    assert decision.grants_execution_authority is False
    assert decision.features is not None
    assert decision.features.futures_basis_bps > 0


def test_stale_or_regressed_reference_data_forces_abstention() -> None:
    monitor = PolymarketBtcPriceDiscoveryMonitor(maximum_staleness_ms=1_000)
    spot = _tick("BINANCE_SPOT")
    futures = _tick("BINANCE_USD_M_FUTURES")
    first = monitor.evaluate(spot=spot, futures=futures, observed_at_ms=NOW_MS)
    assert first.action == "preserve"

    regressed = monitor.evaluate(
        spot=replace(spot, received_at_ms=NOW_MS),
        futures=replace(futures, received_at_ms=NOW_MS),
        observed_at_ms=NOW_MS,
    )
    stale = monitor.evaluate(
        spot=replace(spot, sequence=2),
        futures=replace(futures, sequence=2),
        observed_at_ms=NOW_MS + 2_000,
    )

    assert regressed.action == "abstain"
    assert regressed.maximum_size_multiplier == 0
    assert any("sequence_regression" in reason for reason in regressed.reasons)
    assert stale.action == "abstain"
    assert any(reason.endswith("_stale") for reason in stale.reasons)


def test_wide_reference_spread_can_only_reduce_size() -> None:
    monitor = PolymarketBtcPriceDiscoveryMonitor(
        reduction_spread_bps=Decimal("5")
    )

    decision = monitor.evaluate(
        spot=_tick("BINANCE_SPOT", bid="99900", ask="100100"),
        futures=_tick("BINANCE_USD_M_FUTURES"),
        observed_at_ms=NOW_MS,
    )

    assert decision.action == "reduce"
    assert decision.maximum_size_multiplier == Decimal("0.5")
    assert decision.reasons == ("wide_reference_spread",)


@pytest.mark.parametrize("symbol", ["ETHUSDT", "BTCUSD", ""])
def test_reference_contract_is_btcusdt_only(symbol: str) -> None:
    with pytest.raises(ValueError, match="BTCUSDT-only"):
        replace(_tick("BINANCE_SPOT"), symbol=symbol)


def test_external_signal_module_has_no_execution_or_account_boundary() -> None:
    module_path = (
        Path(__file__).parents[1]
        / "src"
        / "simple_ai_trading"
        / "polymarket_external_signal.py"
    )
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any("binance" in name.lower() for name in imports)
    for forbidden in (
        "private_key",
        "api_secret",
        "place_order",
        "create_order",
        "cancel_order",
        "account_balance",
        "position_risk",
    ):
        assert forbidden not in source.lower()
