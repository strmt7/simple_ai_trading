"""Checksummed BTC-only Binance flow inventory for Polymarket research."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re

from .binance_archive import ArchiveListingItem, archive_file_url
from .spot_perpetual_corpus import (
    FrozenFlowArchive,
    FrozenFlowContract,
    FrozenFlowDay,
)


POLYMARKET_BTC_HISTORY_SCHEMA_VERSION = (
    "polymarket-btc-flow-history-inventory-v1"
)
POLYMARKET_BTC_HISTORY_RESEARCH_ROUND = 15
POLYMARKET_BTC_HISTORY_SYMBOLS = ("BTCUSDT",)
POLYMARKET_BTC_HISTORY_MARKET_TYPES = ("spot", "futures")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_INVENTORY_BYTES = 16 * 1024 * 1024


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("BTC history inventory contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"BTC history inventory contains {value}")


def _parse_day(value: object, *, name: str) -> date:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid YYYY-MM-DD day") from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{name} must be a canonical YYYY-MM-DD day")
    return parsed


def _days(first: date, last: date) -> tuple[str, ...]:
    if last < first:
        raise ValueError("BTC history end day precedes start day")
    count = (last - first).days + 1
    return tuple((first + timedelta(days=index)).isoformat() for index in range(count))


def _utc_text(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} lacks a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _etag(value: object, *, name: str) -> str:
    text = str(value or "").strip().strip('"').lower()
    if (
        not text
        or len(text) > 256
        or any(ord(character) < 0x20 for character in text)
    ):
        raise ValueError(f"{name} is invalid")
    return text


def _normalized_archive(
    item: ArchiveListingItem,
    *,
    market_type: str,
) -> dict[str, object]:
    period = _parse_day(item.period, name="archive period").isoformat()
    expected_url = archive_file_url(
        symbol="BTCUSDT",
        interval="1s",
        period=period,
        market_type=market_type,
        cadence="daily",
        data_type="aggTrades",
    )
    if item.url != expected_url:
        raise ValueError("BTC history archive URL is not the official object")
    expected_bytes = int(item.size_bytes)
    checksum_bytes = int(item.checksum_size_bytes)
    if expected_bytes <= 0 or checksum_bytes <= 0:
        raise ValueError("BTC history archive size metadata is missing")
    return {
        "market_type": market_type,
        "symbol": "BTCUSDT",
        "period": period,
        "url": expected_url,
        "expected_bytes": expected_bytes,
        "last_modified": _utc_text(
            item.last_modified,
            name="archive last-modified",
        ),
        "etag": _etag(item.etag, name="archive ETag"),
        "checksum_expected_bytes": checksum_bytes,
        "checksum_last_modified": _utc_text(
            item.checksum_last_modified,
            name="checksum last-modified",
        ),
        "checksum_etag": _etag(item.checksum_etag, name="checksum ETag"),
    }


def build_polymarket_btc_history_inventory(
    listings: Mapping[str, Sequence[ArchiveListingItem]],
    *,
    first_day: str,
    last_day: str,
    observed_at_utc: str,
) -> dict[str, object]:
    """Freeze every complete official BTC spot/perpetual day in one interval."""

    if set(listings) != set(POLYMARKET_BTC_HISTORY_MARKET_TYPES):
        raise ValueError("BTC history listings must contain spot and futures")
    first = _parse_day(first_day, name="first_day")
    last = _parse_day(last_day, name="last_day")
    periods = _days(first, last)
    by_market: dict[str, dict[str, dict[str, object]]] = {}
    for market_type in POLYMARKET_BTC_HISTORY_MARKET_TYPES:
        normalized = [
            _normalized_archive(item, market_type=market_type)
            for item in listings[market_type]
        ]
        selected = {
            str(item["period"]): item
            for item in normalized
            if first <= _parse_day(item["period"], name="archive period") <= last
        }
        if len(selected) != len(
            [
                item
                for item in normalized
                if first
                <= _parse_day(item["period"], name="archive period")
                <= last
            ]
        ):
            raise ValueError(f"BTC history {market_type} listing has duplicate days")
        missing = tuple(period for period in periods if period not in selected)
        if missing:
            raise ValueError(
                f"BTC history {market_type} listing misses {len(missing)} days; "
                f"first={missing[0]}"
            )
        by_market[market_type] = selected

    days = []
    for period in periods:
        archives = [
            by_market[market_type][period]
            for market_type in POLYMARKET_BTC_HISTORY_MARKET_TYPES
        ]
        compressed_bytes = sum(int(item["expected_bytes"]) for item in archives)
        days.append(
            {
                "month": period[:7],
                "period": period,
                "selection_digest": _canonical_sha256(
                    {
                        "policy": "all_complete_utc_days",
                        "period": period,
                        "streams": [
                            f"{market_type}:BTCUSDT"
                            for market_type in POLYMARKET_BTC_HISTORY_MARKET_TYPES
                        ],
                    }
                ),
                "compressed_bytes": compressed_bytes,
                "archives": archives,
            }
        )
    artifact: dict[str, object] = {
        "schema_version": POLYMARKET_BTC_HISTORY_SCHEMA_VERSION,
        "created_at_utc": _utc_text(observed_at_utc, name="observed_at_utc"),
        "research_round": POLYMARKET_BTC_HISTORY_RESEARCH_ROUND,
        "venue": "polymarket",
        "asset": "BTC",
        "market_variant": "fiveminute",
        "purpose": "full_available_history_exogenous_flow_corpus",
        "selection_policy": (
            "Every complete UTC day in the frozen interval; no price, return, "
            "outcome, model score, or Polymarket target is consulted."
        ),
        "binance_boundary": (
            "Official public BTCUSDT spot and USD-M aggTrades archives are "
            "read-only exogenous features. No Binance credentials, account, "
            "orders, balances, positions, capital, PnL, risk state, or "
            "execution authority are present."
        ),
        "storage_policy": {
            "raw_archive_retained": False,
            "maximum_concurrent_archive_days": 1,
            "persisted_resolution": "one_second",
            "symbols": list(POLYMARKET_BTC_HISTORY_SYMBOLS),
            "market_types": list(POLYMARKET_BTC_HISTORY_MARKET_TYPES),
        },
        "range": {
            "first_day": periods[0],
            "last_day": periods[-1],
            "day_count": len(periods),
        },
        "source_count": len(periods) * len(POLYMARKET_BTC_HISTORY_MARKET_TYPES),
        "selected_compressed_bytes": sum(
            int(day["compressed_bytes"]) for day in days
        ),
        "days": days,
        "training_authority": False,
        "trading_authority": False,
        "profitability_claim": False,
    }
    return {**artifact, "artifact_sha256": _canonical_sha256(artifact)}


def load_polymarket_btc_history_contract(
    path: str | Path,
) -> FrozenFlowContract:
    inventory_path = Path(path)
    if inventory_path.is_symlink():
        raise ValueError("BTC history inventory cannot be a symlink")
    raw = inventory_path.read_bytes()
    if not raw or len(raw) > _MAX_INVENTORY_BYTES:
        raise ValueError("BTC history inventory size is invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("BTC history inventory is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("BTC history inventory is not an object")
    payload = dict(value)
    claimed = str(payload.pop("artifact_sha256", "")).strip().lower()
    if _SHA256.fullmatch(claimed) is None or _canonical_sha256(payload) != claimed:
        raise ValueError("BTC history inventory integrity failed")
    storage = payload.get("storage_policy")
    interval = payload.get("range")
    raw_days = payload.get("days")
    if (
        payload.get("schema_version") != POLYMARKET_BTC_HISTORY_SCHEMA_VERSION
        or int(payload.get("research_round", 0))
        != POLYMARKET_BTC_HISTORY_RESEARCH_ROUND
        or payload.get("venue") != "polymarket"
        or payload.get("asset") != "BTC"
        or payload.get("market_variant") != "fiveminute"
        or payload.get("training_authority") is not False
        or payload.get("trading_authority") is not False
        or payload.get("profitability_claim") is not False
        or not isinstance(storage, Mapping)
        or storage.get("raw_archive_retained") is not False
        or tuple(storage.get("symbols", ())) != POLYMARKET_BTC_HISTORY_SYMBOLS
        or tuple(storage.get("market_types", ()))
        != POLYMARKET_BTC_HISTORY_MARKET_TYPES
        or not isinstance(interval, Mapping)
        or not isinstance(raw_days, list)
        or not raw_days
    ):
        raise ValueError("BTC history inventory contract differs")
    expected_periods = _days(
        _parse_day(interval.get("first_day"), name="range first day"),
        _parse_day(interval.get("last_day"), name="range last day"),
    )
    if int(interval.get("day_count", 0)) != len(expected_periods):
        raise ValueError("BTC history range day count differs")
    days: list[FrozenFlowDay] = []
    for index, raw_day in enumerate(raw_days):
        if not isinstance(raw_day, Mapping):
            raise ValueError("BTC history day is not an object")
        period = str(raw_day.get("period") or "")
        raw_archives = raw_day.get("archives")
        if (
            index >= len(expected_periods)
            or period != expected_periods[index]
            or raw_day.get("month") != period[:7]
            or not isinstance(raw_archives, list)
        ):
            raise ValueError("BTC history day chronology differs")
        archives = tuple(
            FrozenFlowArchive.from_mapping(archive)
            for archive in raw_archives
            if isinstance(archive, Mapping)
        )
        if len(archives) != len(raw_archives):
            raise ValueError("BTC history archive entry is invalid")
        compressed = sum(archive.expected_bytes for archive in archives)
        if int(raw_day.get("compressed_bytes", 0)) != compressed:
            raise ValueError("BTC history day compressed bytes differ")
        selection_digest = str(raw_day.get("selection_digest") or "").lower()
        if _SHA256.fullmatch(selection_digest) is None:
            raise ValueError("BTC history day selection digest differs")
        days.append(
            FrozenFlowDay(
                month=period[:7],
                period=period,
                selection_digest=selection_digest,
                compressed_bytes=compressed,
                archives=archives,
            )
        )
    if len(days) != len(expected_periods):
        raise ValueError("BTC history day coverage differs")
    selected_bytes = sum(day.compressed_bytes for day in days)
    if (
        int(payload.get("source_count", 0))
        != len(days) * len(POLYMARKET_BTC_HISTORY_MARKET_TYPES)
        or int(payload.get("selected_compressed_bytes", 0)) != selected_bytes
    ):
        raise ValueError("BTC history inventory totals differ")
    return FrozenFlowContract(
        design_sha256=claimed,
        inventory_sha256=claimed,
        inventory_file_sha256=_file_sha256(inventory_path),
        selected_compressed_bytes=selected_bytes,
        days=tuple(days),
        symbols=POLYMARKET_BTC_HISTORY_SYMBOLS,
    )


__all__ = [
    "POLYMARKET_BTC_HISTORY_MARKET_TYPES",
    "POLYMARKET_BTC_HISTORY_RESEARCH_ROUND",
    "POLYMARKET_BTC_HISTORY_SCHEMA_VERSION",
    "POLYMARKET_BTC_HISTORY_SYMBOLS",
    "build_polymarket_btc_history_inventory",
    "load_polymarket_btc_history_contract",
]
