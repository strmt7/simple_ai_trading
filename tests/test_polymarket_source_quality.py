from types import SimpleNamespace

from simple_ai_trading.polymarket_source_quality import (
    audit_binance_trade_quality,
)


def _event(stream: str, event_type: str, price: str, quantity: str, index: int):
    return SimpleNamespace(
        stream=stream,
        event_type=event_type,
        symbol="BTC",
        event={"data": {"p": price, "q": quantity}},
        event_sha256=f"{index:064x}",
        message_id=f"{index + 10:064x}",
        received_wall_ms=1_000 + index,
        source_time_ms=900 + index,
    )


def test_source_quality_accepts_documented_positive_trade_types() -> None:
    class Store:
        def iter_public_events(self, run_id, **controls):
            assert run_id == "run"
            assert controls["verified_source"] is False
            yield _event("binance_spot", "trade", "100", "1", 1)
            yield _event("binance_futures", "aggTrade", "101", "2", 2)

    result = audit_binance_trade_quality(Store(), run_id="run")

    assert result["passed"] is True
    assert result["streams"]["binance_futures"]["accepted_trade_count"] == 1


def test_source_quality_rejects_legacy_and_non_positive_futures_frames() -> None:
    class Store:
        def iter_public_events(self, run_id, **controls):
            yield _event("binance_spot", "trade", "100", "1", 1)
            yield _event("binance_futures", "trade", "101", "2", 2)
            yield _event("binance_futures", "trade", "0", "0", 3)

    result = audit_binance_trade_quality(Store(), run_id="run")
    futures = result["streams"]["binance_futures"]

    assert result["passed"] is False
    assert futures["unexpected_trade_type_count"] == 2
    assert futures["non_positive_count"] == 1
    assert len(futures["examples"]) == 2
