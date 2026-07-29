from __future__ import annotations

import ast
from decimal import Decimal
import inspect
import json

import pytest

import simple_ai_trading.polymarket_binance_signal as signal_module
from simple_ai_trading.polymarket_binance_signal import (
    BinanceBtcPublicSignalProvider,
    parse_binance_btc_public_tick,
)
from simple_ai_trading.polymarket_autonomous import (
    PolymarketAutonomousOpenProposal,
)


NOW_MS = 1_800_000_120_000


def _spot(*, bid: str = "100", ask: str = "100.01") -> str:
    return json.dumps(
        {
            "e": "24hrTicker",
            "E": NOW_MS - 100,
            "s": "BTCUSDT",
            "b": bid,
            "B": "2",
            "a": ask,
            "A": "3",
        }
    )


def _futures(
    *,
    bid: str = "100.01",
    ask: str = "100.02",
    update_id: int = 123,
) -> str:
    return json.dumps(
        {
            "e": "bookTicker",
            "E": NOW_MS - 90,
            "T": NOW_MS - 95,
            "u": update_id,
            "s": "BTCUSDT",
            "b": bid,
            "B": "4",
            "a": ask,
            "A": "5",
        }
    )


def _proposal() -> PolymarketAutonomousOpenProposal:
    return PolymarketAutonomousOpenProposal(
        proposal_id="public-signal-test",
        input_sha256="1" * 64,
        model_artifact_sha256="2" * 64,
        promotion_sha256="3" * 64,
        market_id="0x" + "4" * 64,
        token_id="5" * 40,
        symbol="BTC",
        market_variant="fiveminute",
        outcome="Up",
        selected_outcome_probability=Decimal("0.6"),
        requested_quantity=Decimal("5"),
        event_start_time_ms=1_800_000_000_000,
        event_end_time_ms=1_800_000_300_000,
        decision_time_ms=NOW_MS - 100,
        expires_at_ms=NOW_MS + 1_000,
    )


def test_public_signal_module_imports_no_execution_or_credentials() -> None:
    tree = ast.parse(inspect.getsource(signal_module))
    imports = [
        str(node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert all("execution" not in name for name in imports)
    source = inspect.getsource(signal_module)
    assert "api_key" not in source.lower()
    assert "secret" not in source.lower()
    assert "submit" not in source.lower()


def test_parse_documented_spot_and_futures_payloads() -> None:
    spot = parse_binance_btc_public_tick(
        "BINANCE_SPOT",
        json.loads(_spot()),
        received_at_ms=NOW_MS,
        spot_sequence=1,
    )
    futures = parse_binance_btc_public_tick(
        "BINANCE_USD_M_FUTURES",
        json.loads(_futures()),
        received_at_ms=NOW_MS,
    )

    assert spot.source == "BINANCE_SPOT"
    assert spot.event_time_ms == NOW_MS - 100
    assert spot.sequence == 1
    assert futures.source == "BINANCE_USD_M_FUTURES"
    assert futures.sequence == 123
    assert futures.event_time_ms == NOW_MS - 90


@pytest.mark.parametrize(
    ("source", "payload", "error"),
    [
        ("BINANCE_SPOT", _spot(bid="101"), "crossed"),
        ("BINANCE_SPOT", _spot(), "ticker payload"),
        (
            "BINANCE_USD_M_FUTURES",
            _futures().replace('"bookTicker"', '"trade"'),
            "book ticker",
        ),
        (
            "BINANCE_USD_M_FUTURES",
            _futures().replace(str(NOW_MS - 95), str(NOW_MS)),
            "transaction time",
        ),
    ],
)
def test_parser_rejects_wrong_or_unsafe_payloads(
    source: str,
    payload: str,
    error: str,
) -> None:
    sequence = None if source.endswith("FUTURES") else (
        None if error == "ticker payload" else 1
    )
    with pytest.raises(ValueError, match=error):
        parse_binance_btc_public_tick(
            source,
            json.loads(payload),
            received_at_ms=NOW_MS,
            spot_sequence=sequence,
        )


def test_provider_abstains_until_both_public_feeds_are_fresh() -> None:
    provider = BinanceBtcPublicSignalProvider()

    missing = provider.evaluate(
        proposal=_proposal(),
        observed_at_ms=NOW_MS,
    )

    assert missing.action == "abstain"
    assert missing.reasons == ("binance_spot_unavailable",)
    assert missing.grants_execution_authority is False
    assert provider.snapshot(observed_at_ms=NOW_MS).credentials_used is False
    assert provider.snapshot(observed_at_ms=NOW_MS).execution_authority is False


def test_provider_builds_quality_gated_preserve_and_rejects_regression() -> None:
    provider = BinanceBtcPublicSignalProvider()
    provider._handle_message(
        "BINANCE_SPOT",
        _spot(),
        received_at_ms=NOW_MS,
    )
    provider._handle_message(
        "BINANCE_USD_M_FUTURES",
        _futures(),
        received_at_ms=NOW_MS,
    )
    decision = provider.evaluate(
        proposal=_proposal(),
        observed_at_ms=NOW_MS,
    )

    assert decision.action == "preserve"
    assert decision.maximum_size_multiplier == Decimal("1")
    assert decision.features is not None
    snapshot = provider.snapshot(observed_at_ms=NOW_MS)
    assert snapshot.spot_connected is True
    assert snapshot.futures_connected is True
    with pytest.raises(ValueError, match="regressed"):
        provider._handle_message(
            "BINANCE_USD_M_FUTURES",
            _futures(),
            received_at_ms=NOW_MS + 1,
        )


def test_provider_can_only_reduce_when_public_spread_is_wide() -> None:
    provider = BinanceBtcPublicSignalProvider()
    provider._handle_message(
        "BINANCE_SPOT",
        _spot(bid="99.9", ask="100.1"),
        received_at_ms=NOW_MS,
    )
    provider._handle_message(
        "BINANCE_USD_M_FUTURES",
        _futures(bid="99.9", ask="100.1", update_id=124),
        received_at_ms=NOW_MS,
    )

    decision = provider.evaluate(
        proposal=_proposal(),
        observed_at_ms=NOW_MS,
    )

    assert decision.action == "reduce"
    assert decision.maximum_size_multiplier == Decimal("0.5")
    assert decision.grants_execution_authority is False
