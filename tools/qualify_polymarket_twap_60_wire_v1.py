"""Qualify the current BTC five-minute 60-second TWAP settlement contract."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

from websockets.asyncio.client import connect

from simple_ai_trading.polymarket import (
    POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE,
    POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION,
    PolymarketPublicClient,
    validate_clob_market_info,
)
from simple_ai_trading.polymarket_recorder import (
    POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
    POLYMARKET_RTDS_WEBSOCKET,
    _validate_chainlink_twap_60_frame,
)
from simple_ai_trading.storage import write_bytes_atomic


DEFAULT_OUTPUT = Path(
    "docs/model-research/polymarket/"
    "btc-5m-twap-60-wire-source-qualification-v1-2026-08-14.json"
)
_SYMBOL = "btc/usd"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _subscription() -> str:
    return _canonical_json(
        {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
                    "type": "update",
                    "filters": _canonical_json({"symbol": _SYMBOL}),
                }
            ],
        }
    )


def _qualify_markets(
    client: PolymarketPublicClient,
    *,
    observed_at_ms: int,
) -> dict[str, object]:
    observed = int(observed_at_ms)
    if observed <= 0:
        raise ValueError("source observation time must be positive")
    markets = client.discover_five_minute_markets(
        now_ms=observed,
        include_next=True,
        require_all_assets=True,
        assets=("BTC",),
    )
    expected_epoch_ms = observed // 300_000 * 300_000
    if (
        len(markets) != 2
        or tuple(market.event_start_ms for market in markets)
        != (expected_epoch_ms, expected_epoch_ms + 300_000)
        or any(
            market.resolution_source.rstrip("/")
            != POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
            for market in markets
        )
    ):
        raise ValueError("two consecutive exact BTC TWAP-60 markets are unavailable")
    protocol_version = client.protocol_version()
    if protocol_version != POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION:
        raise ValueError("CLOB protocol version differs")
    evidence: list[dict[str, object]] = []
    for market in markets:
        info = client.clob_market_info(market.condition_id)
        clob = validate_clob_market_info(market, info)
        clob_json = _canonical_json(dict(info))
        evidence.append(
            {
                "slug": market.slug,
                "market_id": market.market_id,
                "condition_id": market.condition_id,
                "event_start_ms": market.event_start_ms,
                "end_ms": market.end_ms,
                "resolution_source": market.resolution_source,
                "gamma_payload_sha256": market.gamma_payload_sha256,
                "gamma_payload_json": market.gamma_payload_json,
                "clob_info_sha256": hashlib.sha256(
                    clob_json.encode("ascii")
                ).hexdigest(),
                "clob_info_json": clob_json,
                "minimum_order_age_seconds": clob["minimum_order_age_seconds"],
                "taker_order_delay_enabled": clob["taker_order_delay_enabled"],
            }
        )
    return {
        "observed_at_ms": observed,
        "clob_protocol_version": protocol_version,
        "market_count": len(evidence),
        "markets": evidence,
    }


async def _probe_twap_updates() -> dict[str, object]:
    observations: list[dict[str, object]] = []
    empty_control_frames = 0
    started_monotonic_ns = time.monotonic_ns()
    async with connect(
        POLYMARKET_RTDS_WEBSOCKET,
        open_timeout=10,
        close_timeout=3,
        ping_interval=None,
        ping_timeout=None,
        max_size=64 * 1024,
        max_queue=32,
        compression=None,
    ) as websocket:
        await websocket.send(_subscription())

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(5)
                await websocket.send("PING")

        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            while len(observations) < 2:
                raw = await asyncio.wait_for(websocket.recv(), timeout=15)
                if not isinstance(raw, str):
                    raise ValueError("TWAP qualification received a binary frame")
                if raw == "":
                    empty_control_frames += 1
                    continue
                if raw == "PING":
                    await websocket.send("PONG")
                    continue
                _validate_chainlink_twap_60_frame(raw, expected_symbol=_SYMBOL)
                event = json.loads(raw)
                if not isinstance(event, Mapping) or not isinstance(
                    event.get("payload"), Mapping
                ):
                    raise ValueError("validated TWAP event structure disappeared")
                payload = event["payload"]
                observations.append(
                    {
                        "received_wall_ms": time.time_ns() // 1_000_000,
                        "received_monotonic_ns": time.monotonic_ns(),
                        "raw_frame_sha256": hashlib.sha256(
                            raw.encode("utf-8")
                        ).hexdigest(),
                        "topic": event["topic"],
                        "type": event["type"],
                        "publisher_timestamp_ms": event["timestamp"],
                        "source_timestamp_ms": payload["timestamp"],
                        "symbol": payload["symbol"],
                        "window_s": payload["window_s"],
                        "full_accuracy_value": payload["full_accuracy_value"],
                    }
                )
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
    source_times = [int(row["source_timestamp_ms"]) for row in observations]
    if len(source_times) != 2 or source_times[0] >= source_times[1]:
        raise ValueError("TWAP qualification updates are not strictly ordered")
    return {
        "endpoint": POLYMARKET_RTDS_WEBSOCKET,
        "subscription_json": _subscription(),
        "subscription_sha256": hashlib.sha256(
            _subscription().encode("ascii")
        ).hexdigest(),
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": time.monotonic_ns(),
        "empty_control_frames": empty_control_frames,
        "observations": observations,
    }


async def _run(output: Path) -> dict[str, object]:
    client = PolymarketPublicClient(
        required_five_minute_resolution_sources={
            "BTC": POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE
        }
    )
    market_qualification = _qualify_markets(
        client,
        observed_at_ms=time.time_ns() // 1_000_000,
    )
    wire_probe = await _probe_twap_updates()
    payload: dict[str, object] = {
        "schema_version": "polymarket-btc-5m-twap-60-wire-source-qualification-v1",
        "status": "passed",
        "observed_at_ms": time.time_ns() // 1_000_000,
        "resolution_source": POLYMARKET_BTC_FIVE_MINUTE_TWAP_60_RESOLUTION_SOURCE,
        "crypto_market_config_id": "btc-5m-twap-60",
        "required_rtds_topic": POLYMARKET_RTDS_CHAINLINK_TWAP_60_TOPIC,
        "twap_window_seconds": 60,
        "exact_e18_value_observed": True,
        "market_qualification": market_qualification,
        "wire_probe": wire_probe,
        "public_sources": {
            "gamma": "https://gamma-api.polymarket.com/markets",
            "clob": "https://clob.polymarket.com",
            "rtds_documentation": (
                "https://docs.polymarket.com/market-data/websocket/rtds"
            ),
        },
        "credentials_used": False,
        "execution_connected": False,
        "orders_submitted": 0,
        "outcomes_accessed": False,
        "model_scores_accessed": False,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["qualification_sha256"] = _canonical_sha256(payload)
    write_bytes_atomic(
        output,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify two current BTC 5m markets and two exact Chainlink TWAP-60 "
            "updates without credentials or execution."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = asyncio.run(_run(arguments.output))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
