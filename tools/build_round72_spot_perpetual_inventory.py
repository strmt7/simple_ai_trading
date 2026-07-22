"""Freeze a return-independent monthly sample of Binance spot/perpetual flow."""

from __future__ import annotations

import argparse
import calendar
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError

from simple_ai_trading.binance_archive import (
    ArchiveListingItem,
    archive_file_url,
    list_archive_items,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat
from simple_ai_trading.storage import write_json_atomic


ROUND = 72
SCHEMA_VERSION = "round-072-spot-perpetual-inventory-v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MARKET_TYPES = ("spot", "futures")
START_MONTH = "2020-10"
END_MONTH = "2026-06"
SELECTION_SEED = "round72-price-discovery-v1-20260722"
MINIMUM_COMPLETE_MONTHS = 60
_MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
_HEX_PATTERN = re.compile(r"^[0-9a-f]+$")


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


def _parse_month(value: str) -> date:
    text = str(value or "").strip()
    if _MONTH_PATTERN.fullmatch(text) is None:
        raise ValueError("month must be YYYY-MM")
    try:
        return date.fromisoformat(f"{text}-01")
    except ValueError as exc:
        raise ValueError("month must be a valid YYYY-MM value") from exc


def _next_month(value: date) -> date:
    return (value.replace(day=28) + timedelta(days=4)).replace(day=1)


def _month_days(value: date) -> tuple[str, ...]:
    count = calendar.monthrange(value.year, value.month)[1]
    return tuple(
        date(value.year, value.month, day).isoformat()
        for day in range(1, count + 1)
    )


def _normalize_etag(value: object) -> str:
    text = str(value or "").strip().strip('"').lower()
    if (
        not text
        or len(text) > 256
        or any(ord(character) < 0x20 for character in text)
    ):
        raise ValueError("official archive inventory ETag is invalid")
    return text


def _normalize_last_modified(value: object) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("official archive last-modified value is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("official archive last-modified value lacks a timezone")
    return parsed.astimezone(UTC).isoformat()


def _stream_id(market_type: str, symbol: str) -> str:
    return f"{market_type}:{symbol}"


def _normalized_item(
    item: ArchiveListingItem,
    *,
    market_type: str,
    symbol: str,
) -> dict[str, object]:
    period = str(item.period or "")
    try:
        parsed_period = date.fromisoformat(period)
    except ValueError as exc:
        raise ValueError(f"{_stream_id(market_type, symbol)} has an invalid day") from exc
    if parsed_period.isoformat() != period:
        raise ValueError(f"{_stream_id(market_type, symbol)} day is not canonical")
    expected_url = archive_file_url(
        symbol=symbol,
        interval="1s",
        period=period,
        market_type=market_type,
        cadence="daily",
        data_type="aggTrades",
    )
    if str(item.url) != expected_url:
        raise ValueError(f"{_stream_id(market_type, symbol)} URL is not official")
    expected_bytes = int(item.size_bytes)
    checksum_expected_bytes = int(item.checksum_size_bytes)
    if expected_bytes <= 0 or checksum_expected_bytes <= 0:
        raise ValueError(f"{_stream_id(market_type, symbol)} object size is missing")
    return {
        "period": period,
        "url": expected_url,
        "expected_bytes": expected_bytes,
        "last_modified": _normalize_last_modified(item.last_modified),
        "etag": _normalize_etag(item.etag),
        "checksum_expected_bytes": checksum_expected_bytes,
        "checksum_last_modified": _normalize_last_modified(
            item.checksum_last_modified
        ),
        "checksum_etag": _normalize_etag(item.checksum_etag),
    }


def _normalized_stream(
    items: Sequence[ArchiveListingItem],
    *,
    market_type: str,
    symbol: str,
) -> list[dict[str, object]]:
    rows = [
        _normalized_item(item, market_type=market_type, symbol=symbol)
        for item in items
    ]
    rows.sort(key=lambda value: (str(value["period"]), str(value["url"])))
    if not rows:
        raise ValueError(f"{_stream_id(market_type, symbol)} listing is empty")
    periods = [str(value["period"]) for value in rows]
    urls = [str(value["url"]) for value in rows]
    if len(periods) != len(set(periods)) or len(urls) != len(set(urls)):
        raise ValueError(f"{_stream_id(market_type, symbol)} listing has duplicates")
    return rows


def _selection_digest(seed: str, month: str, day: str) -> str:
    return hashlib.sha256(f"{seed}\x00{month}\x00{day}".encode("ascii")).hexdigest()


def build_inventory_artifact(
    listings: Mapping[tuple[str, str], Sequence[ArchiveListingItem]],
    *,
    observed_at_utc: str,
    start_month: str = START_MONTH,
    end_month: str = END_MONTH,
    selection_seed: str = SELECTION_SEED,
    minimum_complete_months: int = MINIMUM_COMPLETE_MONTHS,
) -> dict[str, object]:
    """Build the frozen sample without reading price or return data."""

    required_keys = {
        (market_type, symbol)
        for market_type in MARKET_TYPES
        for symbol in SYMBOLS
    }
    if set(listings) != required_keys:
        raise ValueError("Round 72 requires all six spot/perpetual listings")
    start = _parse_month(start_month)
    end = _parse_month(end_month)
    if start > end:
        raise ValueError("start_month must not follow end_month")
    try:
        observed = datetime.fromisoformat(
            str(observed_at_utc).strip().replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("observed_at_utc is invalid") from exc
    if observed.tzinfo is None:
        raise ValueError("observed_at_utc must include a timezone")
    seed = str(selection_seed or "").strip()
    if not seed or len(seed) > 256 or not 1 <= int(minimum_complete_months) <= 120:
        raise ValueError("Round 72 selection controls are invalid")

    by_day: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    streams: list[dict[str, object]] = []
    range_start = start.isoformat()
    range_end = (_next_month(end) - timedelta(days=1)).isoformat()
    for market_type in MARKET_TYPES:
        for symbol in SYMBOLS:
            key = (market_type, symbol)
            rows = _normalized_stream(
                listings[key],
                market_type=market_type,
                symbol=symbol,
            )
            scoped = [
                row
                for row in rows
                if range_start <= str(row["period"]) <= range_end
            ]
            if not scoped:
                raise ValueError(f"{_stream_id(*key)} has no rows in the frozen range")
            by_day[key] = {str(row["period"]): row for row in scoped}
            streams.append(
                {
                    "stream_id": _stream_id(market_type, symbol),
                    "market_type": market_type,
                    "symbol": symbol,
                    "full_item_count": len(rows),
                    "full_first_period": str(rows[0]["period"]),
                    "full_last_period": str(rows[-1]["period"]),
                    "full_compressed_bytes": sum(
                        int(row["expected_bytes"]) for row in rows
                    ),
                    "full_listing_sha256": _canonical_sha256(rows),
                    "scoped_item_count": len(scoped),
                    "scoped_first_period": str(scoped[0]["period"]),
                    "scoped_last_period": str(scoped[-1]["period"]),
                    "scoped_compressed_bytes": sum(
                        int(row["expected_bytes"]) for row in scoped
                    ),
                    "scoped_listing_sha256": _canonical_sha256(scoped),
                }
            )

    selected_months: list[dict[str, object]] = []
    excluded_months: list[dict[str, object]] = []
    cursor = start
    while cursor <= end:
        month = cursor.strftime("%Y-%m")
        expected_days = _month_days(cursor)
        missing_by_stream: dict[str, list[str]] = {}
        for key in sorted(required_keys):
            missing = [day for day in expected_days if day not in by_day[key]]
            if missing:
                missing_by_stream[_stream_id(*key)] = missing
        if missing_by_stream:
            excluded_months.append(
                {
                    "month": month,
                    "reason": "incomplete_six_stream_calendar_month",
                    "missing_by_stream": missing_by_stream,
                }
            )
            cursor = _next_month(cursor)
            continue
        ranked_days = sorted(
            (_selection_digest(seed, month, day), day) for day in expected_days
        )
        digest, selected_day = ranked_days[0]
        files: list[dict[str, object]] = []
        for market_type in MARKET_TYPES:
            for symbol in SYMBOLS:
                row = by_day[(market_type, symbol)][selected_day]
                files.append(
                    {
                        "market_type": market_type,
                        "symbol": symbol,
                        **row,
                    }
                )
        selected_months.append(
            {
                "month": month,
                "selected_day": selected_day,
                "selection_digest": digest,
                "compressed_bytes": sum(
                    int(value["expected_bytes"]) for value in files
                ),
                "files": files,
            }
        )
        cursor = _next_month(cursor)

    if len(selected_months) < int(minimum_complete_months):
        raise ValueError("Round 72 has too few complete six-stream months")
    selected_files = sum(len(value["files"]) for value in selected_months)
    selected_bytes = sum(int(value["compressed_bytes"]) for value in selected_months)
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "round": ROUND,
        "status": "complete",
        "observed_at_utc": observed.astimezone(UTC).isoformat(),
        "provider": "Binance public data archive",
        "truth_basis": "official_s3_zip_and_checksum_object_listing_metadata",
        "data_type": "aggTrades",
        "archive_cadence": "daily",
        "symbols": list(SYMBOLS),
        "market_types": list(MARKET_TYPES),
        "start_month": start.strftime("%Y-%m"),
        "end_month": end.strftime("%Y-%m"),
        "selection_seed": seed,
        "selection_rule": (
            "for each complete six-stream calendar month choose the UTC day with "
            "minimum sha256(seed NUL month NUL day)"
        ),
        "price_or_return_data_used_for_selection": False,
        "minimum_complete_months": int(minimum_complete_months),
        "complete_months": len(selected_months),
        "excluded_month_count": len(excluded_months),
        "selected_files": selected_files,
        "selected_compressed_bytes": selected_bytes,
        "retained_archive_policy": "discard_after_verified_transactional_ingestion",
        "streams": streams,
        "excluded_months": excluded_months,
        "selected_months": selected_months,
        "profitability_claim": False,
        "trading_authority": False,
    }
    payload["inventory_sha256"] = _canonical_sha256(payload)
    return payload


class ProgressWriter:
    def __init__(self) -> None:
        self.started = time.monotonic()

    def __call__(self, event: str, **details: object) -> None:
        print(
            _canonical_json(
                {
                    "event": event,
                    "elapsed_seconds": round(time.monotonic() - self.started, 3),
                    **details,
                }
            ),
            file=sys.stderr,
            flush=True,
        )


def _fetch_listings(arguments: argparse.Namespace) -> dict[tuple[str, str], list[ArchiveListingItem]]:
    progress = ProgressWriter()
    output: dict[tuple[str, str], list[ArchiveListingItem]] = {}
    for market_type in MARKET_TYPES:
        for symbol in SYMBOLS:
            last_error: BaseException | None = None
            for attempt in range(1, int(arguments.network_retries) + 1):
                try:
                    with progress_heartbeat(
                        progress,
                        phase="round72_official_listing_fetch",
                        interval_seconds=float(arguments.heartbeat_seconds),
                        details={
                            "market_type": market_type,
                            "symbol": symbol,
                            "attempt": attempt,
                        },
                    ):
                        items = list_archive_items(
                            symbol=symbol,
                            interval="1s",
                            market_type=market_type,
                            cadence="daily",
                            data_type="aggTrades",
                            timeout=int(arguments.timeout),
                        )
                    output[(market_type, symbol)] = items
                    progress(
                        "round72_official_listing_complete",
                        market_type=market_type,
                        symbol=symbol,
                        items=len(items),
                    )
                    break
                except (HTTPError, URLError, OSError, TimeoutError) as exc:
                    last_error = exc
                    if attempt >= int(arguments.network_retries):
                        raise
                    delay = min(120.0, float(2 ** min(attempt, 6)))
                    progress(
                        "round72_official_listing_retry",
                        market_type=market_type,
                        symbol=symbol,
                        attempt=attempt,
                        retry_in_seconds=delay,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    time.sleep(delay)
            if (market_type, symbol) not in output:
                raise RuntimeError("official listing fetch ended without a result") from last_error
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/model-research/action-value/round-072-spot-perpetual-inventory.json",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--network-retries", type=int, default=8)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_path = Path(arguments.output).resolve()
    if output_path.exists() and not arguments.force:
        print("Round 72 inventory output already exists; pass --force to replace it", file=sys.stderr)
        return 2
    if (
        not 1 <= int(arguments.timeout) <= 300
        or not 1 <= int(arguments.network_retries) <= 20
        or not 1.0 <= float(arguments.heartbeat_seconds) <= 300.0
    ):
        print("Round 72 inventory runtime controls are invalid", file=sys.stderr)
        return 2
    try:
        listings = _fetch_listings(arguments)
        artifact = build_inventory_artifact(
            listings,
            observed_at_utc=datetime.now(UTC).isoformat(),
        )
        write_json_atomic(output_path, artifact, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"Round 72 inventory failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        "Round 72 inventory: "
        f"months={artifact['complete_months']} files={artifact['selected_files']} "
        f"compressed_bytes={artifact['selected_compressed_bytes']} "
        f"sha256={artifact['inventory_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
