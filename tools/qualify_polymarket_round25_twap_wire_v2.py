"""Qualify BTC five-minute market metadata and the exact public 30-second TWAP feed."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

from websockets.asyncio.client import connect

from simple_ai_trading.polymarket import PolymarketPublicClient
from simple_ai_trading.polymarket_recorder import (
    POLYMARKET_RTDS_CHAINLINK_TWAP_30_TOPIC,
    POLYMARKET_RTDS_WEBSOCKET,
    _validate_chainlink_twap_30_frame,
)
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
    qualify_round25_source,
)
from simple_ai_trading.storage import write_bytes_atomic


DEFAULT_OUTPUT = Path(
    "docs/model-research/polymarket/"
    "round-025-twap-wire-source-qualification-v2-2026-08-10.json"
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
                    "topic": POLYMARKET_RTDS_CHAINLINK_TWAP_30_TOPIC,
                    "type": "update",
                    "filters": _canonical_json({"symbol": _SYMBOL}),
                }
            ],
        }
    )


async def _probe_twap_updates(*, update_count: int = 2) -> dict[str, object]:
    if update_count != 2:
        raise ValueError("the qualification requires exactly two consecutive updates")
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
            while len(observations) < update_count:
                raw = await asyncio.wait_for(websocket.recv(), timeout=15)
                if not isinstance(raw, str):
                    raise ValueError("TWAP qualification received a binary frame")
                if raw == "":
                    empty_control_frames += 1
                    continue
                if raw == "PING":
                    await websocket.send("PONG")
                    continue
                _validate_chainlink_twap_30_frame(raw, expected_symbol=_SYMBOL)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qualify two BTC 5m markets and two exact Chainlink TWAP updates."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


async def _run(output: Path) -> dict[str, object]:
    client = PolymarketPublicClient(
        required_five_minute_resolution_sources={
            "BTC": POLYMARKET_ROUND25_RESOLUTION_SOURCE
        }
    )
    market_qualification = qualify_round25_source(
        client,
        observed_at_ms=time.time_ns() // 1_000_000,
    )
    wire_probe = await _probe_twap_updates()
    payload: dict[str, object] = {
        "schema_version": "polymarket-round25-twap-wire-source-qualification-v2",
        "status": "passed",
        "observed_at_ms": time.time_ns() // 1_000_000,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "required_rtds_topic": POLYMARKET_RTDS_CHAINLINK_TWAP_30_TOPIC,
        "twap_window_seconds": 30,
        "exact_e18_value_observed": True,
        "market_qualification": market_qualification,
        "wire_probe": wire_probe,
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


def main() -> int:
    arguments = _parser().parse_args()
    result = asyncio.run(_run(arguments.output))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
