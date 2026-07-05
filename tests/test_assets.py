from __future__ import annotations

from simple_ai_trading.assets import (
    DEFAULT_SYMBOLS,
    is_supported_major_symbol,
    major_symbols_for_quote,
    symbol_base_for_supported_quote,
)


def test_major_symbol_helpers_accept_only_btc_eth_sol_usdc_usdt() -> None:
    assert DEFAULT_SYMBOLS == ("BTCUSDC", "ETHUSDC", "SOLUSDC")
    assert major_symbols_for_quote("USDT") == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    assert symbol_base_for_supported_quote("eth/usdt") == "ETH"
    assert is_supported_major_symbol("BTCUSDC")
    assert is_supported_major_symbol("SOL-USDT")
    assert not is_supported_major_symbol("BNBUSDT")
    assert not is_supported_major_symbol("DOGEUSDT")
    assert not is_supported_major_symbol("ETHWUSDT")
    assert not is_supported_major_symbol("BTCBUSD")
    assert not is_supported_major_symbol("1000PEPEUSDT")
