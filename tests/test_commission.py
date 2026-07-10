from __future__ import annotations

import pytest

from simple_ai_trading.api import CommissionRates
from simple_ai_trading.commission import (
    apply_offline_commission_floor,
    apply_verified_commission_rate,
    documented_taker_fee_floor_bps,
)
from simple_ai_trading.types import StrategyConfig


def test_documented_offline_fee_floors_are_market_specific() -> None:
    assert documented_taker_fee_floor_bps("futures") == pytest.approx(4.0)
    assert documented_taker_fee_floor_bps("spot") == pytest.approx(10.0)
    with pytest.raises(ValueError, match="market_type"):
        documented_taker_fee_floor_bps("options")


def test_offline_fee_floor_rejects_optimistic_configuration() -> None:
    futures, assumption = apply_offline_commission_floor(
        StrategyConfig(taker_fee_bps=1.0),
        market_type="futures",
        symbol="BTCUSDT",
    )
    assert futures.taker_fee_bps == pytest.approx(4.0)
    assert assumption.modeled_taker_fee_bps == pytest.approx(4.0)
    assert assumption.source == "binance_documented_fee_floor"
    assert assumption.verified is False

    stressed, stressed_assumption = apply_offline_commission_floor(
        StrategyConfig(taker_fee_bps=12.0),
        market_type="spot",
    )
    assert stressed.taker_fee_bps == pytest.approx(12.0)
    assert stressed_assumption.source == "configured_stress_fee"


def test_verified_fee_uses_exchange_rate_or_higher_stress_rate() -> None:
    class Client:
        @staticmethod
        def get_commission_rates(symbol: str) -> CommissionRates:
            return CommissionRates(
                symbol=symbol,
                market_type="futures",
                maker_rate=0.0002,
                taker_rate=0.0004,
                rpi_rate=0.00005,
                source="binance_futures_user_commission_rate",
            )

    verified, assumption, rates = apply_verified_commission_rate(
        StrategyConfig(taker_fee_bps=1.0),
        client=Client(),  # type: ignore[arg-type]
        symbol="BTCUSDT",
    )
    assert verified.taker_fee_bps == pytest.approx(4.0)
    assert rates.taker_bps == pytest.approx(4.0)
    assert assumption.verified is True
    assert assumption.exchange_taker_fee_bps == pytest.approx(4.0)

    stressed, assumption, _rates = apply_verified_commission_rate(
        StrategyConfig(taker_fee_bps=6.0),
        client=Client(),  # type: ignore[arg-type]
        symbol="BTCUSDT",
    )
    assert stressed.taker_fee_bps == pytest.approx(6.0)
    assert assumption.source == "configured_stress_fee_above_exchange_rate"
