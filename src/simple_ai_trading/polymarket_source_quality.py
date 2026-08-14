"""Target-free source-quality gates for Polymarket predictor feeds."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Mapping

from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_SOURCE_QUALITY_SCHEMA_VERSION = "polymarket-source-quality-v1"
_EXPECTED_TRADE_TYPE = {
    "binance_spot": "trade",
    "binance_futures": "aggTrade",
}
_TRADE_TYPES = frozenset({"aggTrade", "trade"})


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


def audit_binance_trade_quality(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    symbol: str = "BTC",
) -> dict[str, object]:
    """Require documented, finite, positive spot and futures trade records."""

    selected_run = str(run_id or "").strip()
    selected_symbol = str(symbol or "").strip().upper()
    if not selected_run or selected_symbol != "BTC":
        raise ValueError("source-quality audit requires one BTC run")
    counters = {
        stream: {
            "event_types": Counter(),
            "trade_like_count": 0,
            "accepted_trade_count": 0,
            "unexpected_trade_type_count": 0,
            "invalid_numeric_count": 0,
            "non_positive_count": 0,
            "examples": [],
        }
        for stream in _EXPECTED_TRADE_TYPE
    }
    for event in store.iter_public_events(
        selected_run,
        streams=tuple(_EXPECTED_TRADE_TYPE),
        ordered=True,
        verified_source=False,
    ):
        if event.symbol != selected_symbol:
            continue
        stream = event.stream
        state = counters[stream]
        state["event_types"][event.event_type] += 1
        if event.event_type not in _TRADE_TYPES:
            continue
        state["trade_like_count"] += 1
        reason = ""
        if event.event_type != _EXPECTED_TRADE_TYPE[stream]:
            state["unexpected_trade_type_count"] += 1
            reason = "unexpected_trade_type"
        payload = event.event.get("data")
        try:
            price = float(payload.get("p")) if isinstance(payload, Mapping) else math.nan
            quantity = (
                float(payload.get("q")) if isinstance(payload, Mapping) else math.nan
            )
        except (TypeError, ValueError, OverflowError):
            price = math.nan
            quantity = math.nan
        if not math.isfinite(price) or not math.isfinite(quantity):
            state["invalid_numeric_count"] += 1
            reason = reason or "invalid_numeric"
        elif price <= 0 or quantity <= 0:
            state["non_positive_count"] += 1
            reason = reason or "non_positive"
        if not reason:
            state["accepted_trade_count"] += 1
        elif len(state["examples"]) < 3:
            state["examples"].append(
                {
                    "event_sha256": event.event_sha256,
                    "event_type": event.event_type,
                    "message_id": event.message_id,
                    "reason": reason,
                    "received_wall_ms": event.received_wall_ms,
                    "source_time_ms": event.source_time_ms,
                }
            )
    streams: dict[str, object] = {}
    passed = True
    for stream, raw in counters.items():
        item = {
            "expected_trade_type": _EXPECTED_TRADE_TYPE[stream],
            "event_types": dict(sorted(raw["event_types"].items())),
            "trade_like_count": raw["trade_like_count"],
            "accepted_trade_count": raw["accepted_trade_count"],
            "unexpected_trade_type_count": raw["unexpected_trade_type_count"],
            "invalid_numeric_count": raw["invalid_numeric_count"],
            "non_positive_count": raw["non_positive_count"],
            "examples": raw["examples"],
        }
        item["passed"] = (
            item["accepted_trade_count"] > 0
            and item["unexpected_trade_type_count"] == 0
            and item["invalid_numeric_count"] == 0
            and item["non_positive_count"] == 0
        )
        passed = passed and bool(item["passed"])
        streams[stream] = item
    body: dict[str, object] = {
        "schema_version": POLYMARKET_SOURCE_QUALITY_SCHEMA_VERSION,
        "run_id": selected_run,
        "symbol": selected_symbol,
        "target_free": True,
        "passed": passed,
        "streams": streams,
        "model_data_eligible": False,
        "edge_claim": False,
        "profitability_claim": False,
    }
    body["source_quality_sha256"] = _canonical_sha256(body)
    return body


__all__ = [
    "POLYMARKET_SOURCE_QUALITY_SCHEMA_VERSION",
    "audit_binance_trade_quality",
]
