from __future__ import annotations

import asyncio
import json

import pytest

from simple_ai_trading.polymarket_recorder import PolymarketPublicRecorder


def _round20_recorder(tmp_path) -> PolymarketPublicRecorder:
    return PolymarketPublicRecorder(
        tmp_path / "round20.duckdb",
        assets=("BTC",),
        include_binance_spot=False,
        include_binance_futures=False,
        include_rtds_binance=False,
        clob_lane_ids=("clob-a", "clob-b"),
    )


def test_round20_recorder_scope_has_no_required_binance_source(tmp_path) -> None:
    recorder = _round20_recorder(tmp_path)

    assert recorder.required_streams == ("clob_market", "polymarket_rtds")
    assert recorder.clob_lane_ids == ("clob-a", "clob-b")
    assert recorder.rtds_topics == ("crypto_prices_chainlink",)
    assert recorder.include_binance_spot is False
    assert recorder.include_binance_futures is False


def test_round20_recorder_starts_two_independent_clob_lanes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _round20_recorder(tmp_path)
    started: list[str] = []

    async def capture_lane(*, lane: str, **_options: object) -> None:
        started.append(lane)

    monkeypatch.setattr(recorder, "_clob_lane", capture_lane)
    asyncio.run(recorder._clob_stream(asyncio.Queue(), asyncio.Event()))

    assert sorted(started) == ["clob-a", "clob-b"]


def test_round20_recorder_subscribes_only_to_chainlink_rtds(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _round20_recorder(tmp_path)
    captured: list[dict[str, object]] = []

    async def capture_simple_stream(**options: object) -> None:
        captured.append(dict(options))

    monkeypatch.setattr(recorder, "_simple_stream", capture_simple_stream)
    asyncio.run(recorder._rtds_stream(asyncio.Queue(), asyncio.Event()))

    assert len(captured) == 1
    assert captured[0]["lane"] == "rtds:chainlink:btc"
    subscription = json.loads(str(captured[0]["subscription"]))
    assert subscription["subscriptions"] == [
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "filters": '{"symbol":"btc/usd"}',
        }
    ]


def test_round20_recorder_requires_exact_lane_and_topic_attestation(
    tmp_path,
) -> None:
    recorder = _round20_recorder(tmp_path)

    def factory(run_id: str, started_at_ms: int) -> dict[str, object]:
        return {
            "run_id": run_id,
            "created_at_ms": started_at_ms,
            "capture_duration_seconds": 5,
            "required_assets": ["BTC"],
            "required_streams": ["clob_market", "polymarket_rtds"],
            "required_clob_lanes": ["clob-a"],
            "required_rtds_topics": ["crypto_prices_chainlink"],
        }

    with pytest.raises(ValueError, match="CLOB lanes differ"):
        asyncio.run(
            recorder.run(
                duration_seconds=5,
                preregistration_manifest_factory=factory,
            )
        )


def test_legacy_recorder_defaults_are_unchanged(tmp_path) -> None:
    recorder = PolymarketPublicRecorder(tmp_path / "legacy.duckdb")

    assert recorder.required_streams == (
        "binance_spot",
        "clob_market",
        "polymarket_rtds",
    )
    assert recorder.clob_lane_ids == ("clob",)
    assert recorder.rtds_topics == (
        "crypto_prices",
        "crypto_prices_chainlink",
    )
