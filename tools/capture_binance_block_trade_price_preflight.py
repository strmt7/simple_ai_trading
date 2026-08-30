from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class LatestTicker:
    payload: dict[str, Any]
    received_at_utc: str
    received_monotonic_ns: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _decimal(payload: dict[str, Any], field: str) -> Decimal:
    try:
        value = Decimal(str(payload[field]))
    except (KeyError, InvalidOperation) as exc:
        raise ValueError(f"invalid decimal field {field}") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError(f"nonpositive decimal field {field}")
    return value


def _integer(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid integer field {field}")
    return value


def evaluate_block_trade(
    block: dict[str, Any],
    ticker: LatestTicker | None,
    *,
    received_at_utc: str,
    received_monotonic_ns: int,
    maximum_ticker_age_ms: int,
    block_fee_bps: Decimal,
) -> dict[str, Any]:
    symbol = str(block.get("s", ""))
    block_id = _integer(block, "t")
    block_event_ms = _integer(block, "E")
    block_trade_ms = _integer(block, "T")
    block_price = _decimal(block, "p")
    block_quantity = _decimal(block, "q")
    buyer_is_maker = block.get("m")
    if (
        block.get("e") != "blockTrade"
        or not symbol
        or not isinstance(buyer_is_maker, bool)
    ):
        raise ValueError("invalid blockTrade identity")

    row: dict[str, Any] = {
        "symbol": symbol,
        "block_trade_id": block_id,
        "block_event_ms": block_event_ms,
        "block_trade_ms": block_trade_ms,
        "block_price": str(block_price),
        "block_quantity": str(block_quantity),
        "buyer_is_maker": buyer_is_maker,
        "received_at_utc": received_at_utc,
        "received_monotonic_ns": received_monotonic_ns,
        "analyzable": False,
        "rejection_reason": None,
    }
    if ticker is None:
        row["rejection_reason"] = "no_prior_ticker"
        return row

    ticker_payload = ticker.payload
    if ticker_payload.get("e") != "24hrTicker" or ticker_payload.get("s") != symbol:
        row["rejection_reason"] = "ticker_identity_mismatch"
        return row
    ticker_event_ms = _integer(ticker_payload, "E")
    ticker_age_ms = block_trade_ms - ticker_event_ms
    row.update(
        {
            "ticker_event_ms": ticker_event_ms,
            "ticker_age_ms": ticker_age_ms,
            "ticker_received_at_utc": ticker.received_at_utc,
            "ticker_received_monotonic_ns": ticker.received_monotonic_ns,
        }
    )
    if ticker.received_monotonic_ns > received_monotonic_ns:
        row["rejection_reason"] = "ticker_received_after_block"
        return row
    if ticker_age_ms < 0:
        row["rejection_reason"] = "ticker_event_after_block_trade"
        return row
    if ticker_age_ms > maximum_ticker_age_ms:
        row["rejection_reason"] = "ticker_too_old"
        return row

    bid = _decimal(ticker_payload, "b")
    bid_quantity = _decimal(ticker_payload, "B")
    ask = _decimal(ticker_payload, "a")
    ask_quantity = _decimal(ticker_payload, "A")
    if bid >= ask:
        row["rejection_reason"] = "crossed_or_locked_ticker"
        return row

    fee_fraction = block_fee_bps / Decimal(10_000)
    buyer_block_cost = block_price * (Decimal(1) + fee_fraction)
    seller_block_proceeds = block_price * (Decimal(1) - fee_fraction)
    buyer_saving_bps = (ask - buyer_block_cost) / ask * Decimal(10_000)
    seller_saving_bps = (seller_block_proceeds - bid) / bid * Decimal(10_000)
    row.update(
        {
            "analyzable": True,
            "rejection_reason": None,
            "best_bid": str(bid),
            "best_bid_quantity": str(bid_quantity),
            "best_ask": str(ask),
            "best_ask_quantity": str(ask_quantity),
            "buyer_zero_public_fee_lower_bound_saving_bps": str(buyer_saving_bps),
            "seller_zero_public_fee_lower_bound_saving_bps": str(seller_saving_bps),
            "buyer_strictly_positive_lower_bound": buyer_saving_bps > 0,
            "seller_strictly_positive_lower_bound": seller_saving_bps > 0,
        }
    )
    return row


def _validate_contract(contract: dict[str, Any], runner_path: Path) -> None:
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise ValueError("frozen_at_utc must be an explicit UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > _utc_now():
        raise ValueError("frozen_at_utc is missing a timezone or is in the future")
    expected_runner_hash = contract.get("runner_sha256")
    actual_runner_hash = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    if expected_runner_hash != actual_runner_hash:
        raise ValueError("runner SHA-256 does not match the frozen contract")
    if contract.get("authority") != "public_unauthenticated_market_data_only":
        raise ValueError("capture authority is not public market-data-only")
    if contract.get("request_body_sha256") != EMPTY_SHA256:
        raise ValueError("unexpected WebSocket request body hash")
    if contract.get("reconnects_allowed") != 0:
        raise ValueError("reconnects must be disabled")
    symbols = contract.get("symbols")
    if symbols != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        raise ValueError("unexpected symbol population")
    expected_streams = [
        f"{symbol.lower()}@{suffix}"
        for symbol in symbols
        for suffix in ("blockTrade", "ticker")
    ]
    if contract.get("streams") != expected_streams:
        raise ValueError("stream population does not match the frozen contract")
    expected_url = "wss://data-stream.binance.vision/stream?streams=" + "/".join(
        expected_streams
    )
    if contract.get("stream_url") != expected_url:
        raise ValueError("stream URL does not match the frozen population")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


async def _capture(
    *, contract: dict[str, Any], raw_path: Path, journal_path: Path
) -> dict[str, Any]:
    duration_seconds = int(contract["duration_seconds"])
    target_block_trades = int(contract["target_block_trades"])
    maximum_ticker_age_ms = int(contract["maximum_ticker_age_ms"])
    block_fee_bps = Decimal(str(contract["block_fee_bps_each_side"]))
    latest: dict[str, LatestTicker] = {}
    ticker_counts = {symbol: 0 for symbol in contract["symbols"]}
    block_counts = {symbol: 0 for symbol in contract["symbols"]}
    rows: list[dict[str, Any]] = []
    started_utc = _utc_text()
    started_mono = time.monotonic()
    _append_jsonl(
        journal_path,
        {
            "phase": "started",
            "started_at_utc": started_utc,
            "stream_url": contract["stream_url"],
        },
    )
    try:
        async with connect(
            contract["stream_url"],
            open_timeout=10,
            close_timeout=5,
            ping_interval=20,
            ping_timeout=60,
            max_size=1_048_576,
        ) as websocket:
            while sum(block_counts.values()) < target_block_trades:
                elapsed = time.monotonic() - started_mono
                remaining = duration_seconds - elapsed
                if remaining <= 0:
                    break
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(), timeout=remaining
                    )
                except TimeoutError:
                    break
                received_utc = _utc_text()
                received_ns = time.monotonic_ns()
                envelope = json.loads(message)
                stream = envelope.get("stream")
                payload = envelope.get("data")
                if not isinstance(stream, str) or not isinstance(payload, dict):
                    raise ValueError("invalid combined-stream envelope")
                if stream not in contract["streams"]:
                    raise ValueError(f"unexpected stream {stream}")
                record = {
                    "received_at_utc": received_utc,
                    "received_monotonic_ns": received_ns,
                    "stream": stream,
                    "data": payload,
                }
                _append_jsonl(raw_path, record)
                symbol = str(payload.get("s", ""))
                event_type = payload.get("e")
                if event_type == "24hrTicker":
                    if stream != f"{symbol.lower()}@ticker":
                        raise ValueError("ticker stream identity mismatch")
                    ticker_counts[symbol] += 1
                    latest[symbol] = LatestTicker(payload, received_utc, received_ns)
                elif event_type == "blockTrade":
                    if stream != f"{symbol.lower()}@blockTrade":
                        raise ValueError("block stream identity mismatch")
                    block_counts[symbol] += 1
                    rows.append(
                        evaluate_block_trade(
                            payload,
                            latest.get(symbol),
                            received_at_utc=received_utc,
                            received_monotonic_ns=received_ns,
                            maximum_ticker_age_ms=maximum_ticker_age_ms,
                            block_fee_bps=block_fee_bps,
                        )
                    )
                else:
                    raise ValueError(f"unexpected event type {event_type}")
    except Exception as exc:
        _append_jsonl(
            journal_path,
            {
                "phase": "failed",
                "failed_at_utc": _utc_text(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "reconnects_attempted": 0,
            },
        )
        raise

    completed_utc = _utc_text()
    elapsed_seconds = time.monotonic() - started_mono
    analyzable = [row for row in rows if row["analyzable"]]
    favorable = {
        "buyer": sum(
            bool(row.get("buyer_strictly_positive_lower_bound")) for row in analyzable
        ),
        "seller": sum(
            bool(row.get("seller_strictly_positive_lower_bound")) for row in analyzable
        ),
    }
    summary = {
        "started_at_utc": started_utc,
        "completed_at_utc": completed_utc,
        "elapsed_seconds": elapsed_seconds,
        "target_block_trades": target_block_trades,
        "ticker_counts": ticker_counts,
        "block_counts": block_counts,
        "block_trade_count": len(rows),
        "analyzable_block_trade_count": len(analyzable),
        "favorable_zero_public_fee_lower_bound_counts": favorable,
        "reconnects_attempted": 0,
        "rows": rows,
    }
    _append_jsonl(
        journal_path,
        {
            "phase": "completed",
            "completed_at_utc": completed_utc,
            "elapsed_seconds": elapsed_seconds,
            "ticker_counts": ticker_counts,
            "block_counts": block_counts,
            "block_trade_count": len(rows),
            "analyzable_block_trade_count": len(analyzable),
            "reconnects_attempted": 0,
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen public Binance block-trade price preflight."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    runner_path = Path(__file__).resolve()
    contract_path = args.contract.resolve()
    raw_path = args.raw_output.resolve()
    journal_path = args.journal.resolve()
    summary_path = args.summary_output.resolve()
    contract = _load_json(contract_path)
    _validate_contract(contract, runner_path)
    if raw_path.exists() or summary_path.exists():
        raise ValueError("one-use raw or summary output already exists")
    if not journal_path.is_file():
        raise ValueError("pre-access journal is missing")
    for output in (raw_path, journal_path, summary_path):
        if not output.parent.is_dir():
            raise ValueError(f"output parent is missing: {output.parent}")
    summary = asyncio.run(
        _capture(contract=contract, raw_path=raw_path, journal_path=journal_path)
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "block_trade_count": summary["block_trade_count"],
                "analyzable_block_trade_count": summary["analyzable_block_trade_count"],
                "favorable_zero_public_fee_lower_bound_counts": summary[
                    "favorable_zero_public_fee_lower_bound_counts"
                ],
                "ticker_counts": summary["ticker_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
