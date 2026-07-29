#!/usr/bin/env python3
"""Run the frozen Polymarket BTC predictor as stdout-only shadow telemetry."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_shadow import (  # noqa: E402
    PolymarketBtcFlowBuffer,
    PolymarketHistoricalShadowScorer,
    load_verified_historical_shadow_predictor,
)
from simple_ai_trading.polymarket_historical_shadow_feed import (  # noqa: E402
    PolymarketHistoricalShadowFeed,
)
from simple_ai_trading.polymarket_historical_shadow_opportunity import (  # noqa: E402
    evaluate_shadow_settlement_opportunity,
    fetch_current_btc_shadow_market_state,
)
from simple_ai_trading.polymarket import PolymarketPublicClient  # noqa: E402


_OFFSETS_MS = frozenset(
    {30_000, 60_000, 90_000, 120_000, 150_000, 180_000, 210_000, 240_000}
)


def _emit(event: str, payload: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "observed_at_ms": time.time_ns() // 1_000_000,
                "payload": payload,
                "trading_authority": False,
                "profitability_claim": False,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hash-verified BTC Polymarket model shadow telemetry."
    )
    parser.add_argument("--pretest", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Stop after this duration; zero runs until interrupted.",
    )
    parser.add_argument(
        "--status-interval-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--no-polymarket-opportunity",
        action="store_false",
        dest="polymarket_opportunity",
        help="Disable public Polymarket after-cost shadow screening.",
    )
    values = parser.parse_args()
    if values.duration_seconds < 0.0:
        parser.error("--duration-seconds cannot be negative")
    if not 1.0 <= values.status_interval_seconds <= 300.0:
        parser.error("--status-interval-seconds must lie in [1, 300]")
    return values


async def _run(args: argparse.Namespace) -> None:
    predictor = load_verified_historical_shadow_predictor(
        pretest_path=args.pretest,
        evaluation_path=args.evaluation,
        support_path=args.support,
    )
    flow = PolymarketBtcFlowBuffer()
    feed = PolymarketHistoricalShadowFeed(flow=flow)
    scorer = PolymarketHistoricalShadowScorer(predictor=predictor, flow=flow)
    public_client = (
        PolymarketPublicClient(timeout_seconds=3.0)
        if args.polymarket_opportunity
        else None
    )
    stop = asyncio.Event()
    feed_task = asyncio.create_task(feed.run(stop), name="polymarket-shadow-feed")
    await asyncio.sleep(0)
    started = time.monotonic()
    next_status = started
    scored: set[tuple[int, int]] = set()
    _emit(
        "started",
        {
            "candidate_id": predictor.candidate_id,
            "dataset_sha256": predictor.dataset_sha256,
            "pretest_artifact_sha256": predictor.pretest_artifact_sha256,
            "evaluation_artifact_sha256": predictor.evaluation_artifact_sha256,
            "support_profile_sha256": (
                predictor.support_profile.artifact_sha256
            ),
        },
    )
    try:
        while not stop.is_set():
            if feed_task.done():
                await feed_task
                raise RuntimeError("Polymarket shadow feed stopped unexpectedly")
            now_ms = time.time_ns() // 1_000_000
            event_start_ms = now_ms // 300_000 * 300_000
            offset_ms = now_ms - event_start_ms
            eligible_offsets = [
                value for value in _OFFSETS_MS if value <= offset_ms
            ]
            if eligible_offsets:
                decision_offset_ms = max(eligible_offsets)
                identity = (event_start_ms, decision_offset_ms)
                if identity not in scored:
                    decision_time_ms = event_start_ms + decision_offset_ms
                    decision = scorer.evaluate(
                        event_start_ms=event_start_ms,
                        decision_time_ms=decision_time_ms,
                        observed_at_ms=now_ms,
                    )
                    _emit("prediction", asdict(decision))
                    if decision.status == "observed" and public_client is not None:
                        try:
                            market_state = await asyncio.wait_for(
                                asyncio.to_thread(
                                    fetch_current_btc_shadow_market_state,
                                    public_client,
                                    now_ms=now_ms,
                                ),
                                timeout=8.0,
                            )
                            opportunity = evaluate_shadow_settlement_opportunity(
                                decision,
                                market_state,
                            )
                            _emit("opportunity", opportunity.asdict())
                        except Exception as exc:
                            _emit(
                                "opportunity_error",
                                {
                                    "error_type": type(exc).__name__,
                                    "message": str(exc)[:500],
                                },
                            )
                    scored.add(identity)
            current = time.monotonic()
            if current >= next_status:
                _emit("health", asdict(feed.health()))
                next_status = current + args.status_interval_seconds
            if (
                args.duration_seconds > 0.0
                and current - started >= args.duration_seconds
            ):
                stop.set()
                break
            await asyncio.sleep(0.05)
    finally:
        stop.set()
        await asyncio.gather(feed_task, return_exceptions=True)
        _emit("stopped", asdict(feed.health()))


def main() -> int:
    args = _arguments()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
