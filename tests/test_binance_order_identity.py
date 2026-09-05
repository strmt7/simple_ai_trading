"""Submission and reconciliation must use exactly the same caller identity."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from simple_ai_trading.api import BinanceAPIError, BinanceClient


@pytest.mark.parametrize("market_type", ["spot", "futures"])
@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize(
    "identity", ["x" * 37, "", " ", " order", "order ", 123, False]
)
def test_invalid_submission_identity_fails_before_any_side_effect(
    monkeypatch, market_type, dry_run, identity
) -> None:
    client = BinanceClient(api_key="", api_secret="", market_type=market_type)
    request = Mock(side_effect=AssertionError("unexpected request"))
    leverage = Mock(side_effect=AssertionError("unexpected leverage mutation"))
    monkeypatch.setattr(client, "_request_dict", request)
    monkeypatch.setattr(client, "set_leverage", leverage)

    with pytest.raises(BinanceAPIError, match="client order ID"):
        client.place_order(
            "BTCUSDT", "BUY", 0.001, dry_run=dry_run, client_order_id=identity
        )

    request.assert_not_called()
    leverage.assert_not_called()


def test_distinct_long_intents_cannot_collapse_to_one_transmitted_id() -> None:
    client = BinanceClient(api_key="", api_secret="")
    for identity in ["x" * 36 + "a", "x" * 36 + "b"]:
        with pytest.raises(BinanceAPIError, match="client order ID"):
            client.place_order(
                "BTCUSDT", "BUY", 0.001, dry_run=True, client_order_id=identity
            )


@pytest.mark.parametrize("identity", ["sait-o-abc123", "x" * 36])
@pytest.mark.parametrize("market_type", ["spot", "futures"])
def test_submission_and_reconciliation_preserve_the_exact_identity(
    monkeypatch, identity, market_type
) -> None:
    client = BinanceClient(api_key="", api_secret="", market_type=market_type)
    request = Mock(return_value={"status": "NEW"})
    monkeypatch.setattr(client, "_request_dict", request)
    monkeypatch.setattr(client, "set_leverage", Mock())

    paper = client.place_order(
        "BTCUSDT", "BUY", 0.001, dry_run=True, client_order_id=identity
    )
    assert paper["clientOrderId"] == identity
    client.place_order("BTCUSDT", "BUY", 0.001, dry_run=False, client_order_id=identity)
    assert request.call_args.args[2]["newClientOrderId"] == identity
    client.get_order("BTCUSDT", orig_client_order_id=identity)
    assert request.call_args.args[2]["origClientOrderId"] == identity


@pytest.mark.parametrize(
    "identity", ["x" * 37, "", " ", " order", "order ", 123, False]
)
def test_invalid_query_identity_is_not_silently_normalized(
    monkeypatch, identity
) -> None:
    client = BinanceClient(api_key="", api_secret="")
    request = Mock(side_effect=AssertionError("unexpected request"))
    monkeypatch.setattr(client, "_request_dict", request)

    with pytest.raises(BinanceAPIError, match="client order ID"):
        client.get_order("BTCUSDT", orig_client_order_id=identity)

    request.assert_not_called()
