"""Ingest the exact frozen Round 62 Binance book-depth inventory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import requests

from simple_ai_trading.binance_archive import list_archive_items
from simple_ai_trading.depth_stress_evaluation import (
    DEPTH_STRESS_EVALUATION_SYMBOLS,
)
from simple_ai_trading.microstructure_warehouse import (
    MicrostructureWarehouse,
    create_archive_http_session,
)
from simple_ai_trading.progress_heartbeat import progress_heartbeat
from simple_ai_trading.storage import write_json_atomic


DESIGN_DEFAULT = Path(
    "docs/model-research/action-value/round-062-depth-stress-transition-design.json"
)
INVENTORY_DEFAULT = Path(
    "docs/model-research/action-value/round-062-official-archive-inventory.json"
)
_DATA_TYPE = "bookDepth"
_SCHEMA_VERSION = "round-062-frozen-depth-corpus-ingestion-v1"


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


def _read_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _normalize_etag(value: object) -> str:
    text = str(value or "").strip().strip('"').lower()
    if not text or len(text) > 256 or any(ord(character) < 0x20 for character in text):
        raise ValueError("official archive inventory ETag is invalid")
    return text


def _normalized_listing(items: Sequence[object]) -> list[dict[str, object]]:
    normalized = [
        {
            "period": str(getattr(item, "period", "")),
            "url": str(getattr(item, "url", "")),
            "expected_bytes": int(getattr(item, "size_bytes", 0)),
            "last_modified": str(getattr(item, "last_modified", "")),
            "etag": _normalize_etag(getattr(item, "etag", "")),
            "checksum_expected_bytes": int(
                getattr(item, "checksum_size_bytes", 0)
            ),
            "checksum_last_modified": str(
                getattr(item, "checksum_last_modified", "")
            ),
            "checksum_etag": _normalize_etag(
                getattr(item, "checksum_etag", "")
            ),
        }
        for item in items
    ]
    normalized.sort(key=lambda value: (str(value["period"]), str(value["url"])))
    if not normalized:
        raise ValueError("frozen official archive listing is empty")
    periods = [str(item["period"]) for item in normalized]
    urls = [str(item["url"]) for item in normalized]
    if len(periods) != len(set(periods)) or len(urls) != len(set(urls)):
        raise ValueError("frozen official archive listing contains duplicates")
    if any(
        int(item["expected_bytes"]) <= 0
        or int(item["checksum_expected_bytes"]) <= 0
        or not str(item["last_modified"])
        or not str(item["checksum_last_modified"])
        for item in normalized
    ):
        raise ValueError("frozen official archive listing metadata is incomplete")
    return normalized


def _frozen_contract(
    design_path: Path,
    inventory_path: Path,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    design = _read_object(design_path, "Round 62 design")
    canonical_design = dict(design)
    design_sha256 = str(canonical_design.pop("design_sha256", ""))
    source = design.get("source_contract")
    if (
        design.get("round") != 62
        or design.get("schema_version")
        != "round-062-depth-stress-transition-design-v1"
        or design_sha256 != _canonical_sha256(canonical_design)
        or not isinstance(source, Mapping)
        or tuple(source.get("symbols", ())) != DEPTH_STRESS_EVALUATION_SYMBOLS
        or source.get("data_type") != _DATA_TYPE
    ):
        raise ValueError("Round 62 frozen design identity is invalid")
    inventory_file_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    if inventory_file_sha256 != source.get("inventory_file_sha256"):
        raise ValueError("Round 62 frozen inventory file hash is invalid")
    inventory = _read_object(inventory_path, "Round 62 inventory")
    if (
        inventory.get("status") != "ok"
        or inventory.get("inventory_identity_verified") is not True
    ):
        raise ValueError("Round 62 frozen inventory is not verified")
    snapshots = inventory.get("inventory_snapshots")
    if not isinstance(snapshots, list):
        raise ValueError("Round 62 frozen inventory snapshots are missing")
    by_symbol: dict[str, dict[str, object]] = {}
    identity_fields = (
        "snapshot_id",
        "item_count",
        "first_period",
        "last_period",
        "listing_sha256",
    )
    for symbol in DEPTH_STRESS_EVALUATION_SYMBOLS:
        candidates = [
            item
            for item in snapshots
            if isinstance(item, Mapping)
            and item.get("symbol") == symbol
            and item.get("data_type") == _DATA_TYPE
        ]
        identities = {
            tuple(item.get(field) for field in identity_fields) for item in candidates
        }
        if len(identities) != 1:
            raise ValueError(f"{symbol} frozen inventory identity is missing or ambiguous")
        identity = next(iter(identities))
        by_symbol[symbol] = dict(zip(identity_fields, identity, strict=True))
    expected_files = sum(int(value["item_count"]) for value in by_symbol.values())
    if expected_files != int(source.get("book_depth_files", -1)):
        raise ValueError("Round 62 frozen book-depth file count differs")
    contract = {
        "design_sha256": design_sha256,
        "inventory_file_sha256": inventory_file_sha256,
        "first_period": str(source["available_period"][0]),
        "last_period": str(source["available_period"][1]),
        "expected_files": expected_files,
        "expected_compressed_bytes": int(source["book_depth_compressed_bytes"]),
    }
    return contract, by_symbol


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.monotonic()
        self.sequence = 0
        self._lock = threading.Lock()

    def __call__(self, event: str, **details: object) -> None:
        with self._lock:
            self.sequence += 1
            payload = {
                "schema_version": "round-062-depth-ingestion-progress-v1",
                "round": 62,
                "sequence": self.sequence,
                "event": event,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "total_elapsed_seconds": round(time.monotonic() - self.started, 3),
                "details": details,
            }
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            print(_canonical_json(payload), file=sys.stderr, flush=True)


def _required_certificate_bounds(contract: Mapping[str, object]) -> tuple[int, int]:
    first = date.fromisoformat(str(contract["first_period"]))
    last_available = date.fromisoformat(str(contract["last_period"]))
    next_month = (
        last_available.replace(day=28) + timedelta(days=4)
    ).replace(day=1)
    last_complete = next_month - timedelta(days=1)
    if last_available < last_complete:
        last_complete = last_available.replace(day=1) - timedelta(days=1)
    start_ms = int(datetime(first.year, first.month, 1, tzinfo=UTC).timestamp() * 1_000)
    end_ms = int(
        (datetime.combine(last_complete + timedelta(days=1), datetime.min.time(), UTC)).timestamp()
        * 1_000
        - 1
    )
    return start_ms, end_ms


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design_path = Path(arguments.design).resolve()
    inventory_path = Path(arguments.inventory).resolve()
    warehouse_path = Path(arguments.warehouse).resolve()
    output_path = Path(arguments.output).resolve()
    progress_path = Path(arguments.progress).resolve()
    if len(
        {design_path, inventory_path, warehouse_path, output_path, progress_path}
    ) != 5:
        raise ValueError("Round 62 ingestion paths must be distinct")
    if (
        not 1 <= int(arguments.threads) <= 32
        or not 1 <= int(arguments.network_retries) <= 100
        or not 1.0 <= float(arguments.heartbeat_seconds) <= 300.0
        or not 1.0 <= float(arguments.timeout) <= 900.0
    ):
        raise ValueError("Round 62 ingestion runtime limits are invalid")
    contract, expected_by_symbol = _frozen_contract(design_path, inventory_path)
    progress = ProgressWriter(progress_path)
    progress(
        "frozen_inventory_validation_started",
        symbols=list(DEPTH_STRESS_EVALUATION_SYMBOLS),
        expected_files=contract["expected_files"],
        expected_compressed_bytes=contract["expected_compressed_bytes"],
    )

    selected_by_symbol: dict[str, list[object]] = {}
    observed_bytes = 0
    for symbol in DEPTH_STRESS_EVALUATION_SYMBOLS:
        with progress_heartbeat(
            progress,
            phase="official_listing_fetch",
            interval_seconds=float(arguments.heartbeat_seconds),
            details={"symbol": symbol},
        ):
            available = list_archive_items(
                symbol=symbol,
                interval="tick",
                market_type="futures",
                cadence="daily",
                data_type=_DATA_TYPE,
                timeout=min(60, int(float(arguments.timeout))),
            )
        selected = [
            item
            for item in available
            if str(contract["first_period"])
            <= str(getattr(item, "period", ""))
            <= str(contract["last_period"])
        ]
        normalized = _normalized_listing(selected)
        expected = expected_by_symbol[symbol]
        listing_sha256 = _canonical_sha256(normalized)
        if (
            len(normalized) != int(expected["item_count"])
            or str(normalized[0]["period"]) != expected["first_period"]
            or str(normalized[-1]["period"]) != expected["last_period"]
            or listing_sha256 != expected["listing_sha256"]
        ):
            raise ValueError(f"{symbol} official listing drifted from frozen inventory")
        selected_by_symbol[symbol] = selected
        observed_bytes += sum(int(item["expected_bytes"]) for item in normalized)
        progress(
            "frozen_listing_verified",
            symbol=symbol,
            files=len(normalized),
            listing_sha256=listing_sha256,
        )
    if observed_bytes != int(contract["expected_compressed_bytes"]):
        raise ValueError("Round 62 frozen compressed-byte total differs")

    completed_files = 0
    ingested_files = 0
    reused_files = 0
    completed_bytes = 0
    certificate_bounds = _required_certificate_bounds(contract)
    certificates: list[dict[str, object]] = []
    with (
        MicrostructureWarehouse(
            warehouse_path,
            cache_root=str(arguments.cache_root),
            memory_limit=str(arguments.memory_limit),
            threads=int(arguments.threads),
        ) as warehouse,
        create_archive_http_session() as session,
    ):
        reusable: dict[tuple[str, str], object] = {}
        for symbol, items in selected_by_symbol.items():
            snapshot = warehouse.record_official_archive_inventory(
                symbol=symbol,
                data_type=_DATA_TYPE,
                items=items,
                full_history=True,
            )
            if snapshot["snapshot_id"] != expected_by_symbol[symbol]["snapshot_id"]:
                raise ValueError(f"{symbol} persisted frozen inventory identity differs")
            reusable.update(
                {
                    (symbol, period): result
                    for period, result in warehouse.reusable_official_archives(
                        symbol=symbol,
                        data_type=_DATA_TYPE,
                        items=items,
                    ).items()
                }
            )

        total_files = int(contract["expected_files"])
        file_index = 0
        for symbol in DEPTH_STRESS_EVALUATION_SYMBOLS:
            for item in selected_by_symbol[symbol]:
                file_index += 1
                period = str(getattr(item, "period"))
                expected_bytes = int(getattr(item, "size_bytes"))
                existing = reusable.get((symbol, period))
                if existing is not None:
                    result = existing
                    reused_files += 1
                else:
                    result = None
                    for attempt in range(1, int(arguments.network_retries) + 1):
                        try:
                            with progress_heartbeat(
                                progress,
                                phase="archive_ingestion",
                                interval_seconds=float(arguments.heartbeat_seconds),
                                details={
                                    "symbol": symbol,
                                    "period": period,
                                    "file_index": file_index,
                                    "total_files": total_files,
                                    "attempt": attempt,
                                },
                            ):
                                result = warehouse.ingest_public_archive(
                                    symbol=symbol,
                                    data_type=_DATA_TYPE,
                                    period=period,
                                    url=str(getattr(item, "url")),
                                    expected_bytes=expected_bytes,
                                    official_last_modified=str(
                                        getattr(item, "last_modified")
                                    ),
                                    official_etag=str(getattr(item, "etag")),
                                    checksum_object_size_bytes=int(
                                        getattr(item, "checksum_size_bytes")
                                    ),
                                    checksum_last_modified=str(
                                        getattr(item, "checksum_last_modified")
                                    ),
                                    checksum_etag=str(
                                        getattr(item, "checksum_etag")
                                    ),
                                    timeout_seconds=float(arguments.timeout),
                                    max_download_bytes=max(
                                        expected_bytes * 2,
                                        expected_bytes + 8 * 1024**2,
                                    ),
                                    max_uncompressed_bytes=min(
                                        4 * 1024**3,
                                        max(256 * 1024**2, expected_bytes * 100),
                                    ),
                                    retain_archive=bool(arguments.retain_archives),
                                    session=session,
                                )
                            break
                        except requests.RequestException as exc:
                            if attempt >= int(arguments.network_retries):
                                raise
                            delay = min(300.0, float(2 ** min(attempt, 8)))
                            progress(
                                "network_retry",
                                symbol=symbol,
                                period=period,
                                attempt=attempt,
                                retry_in_seconds=delay,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            time.sleep(delay)
                    if result is None:
                        raise RuntimeError("archive ingestion ended without a result")
                    ingested_files += 1
                completed_files += 1
                completed_bytes += expected_bytes
                progress(
                    "archive_completed",
                    symbol=symbol,
                    period=period,
                    file_index=file_index,
                    total_files=total_files,
                    status=str(getattr(result, "status")),
                    rows_read=int(getattr(result, "rows_read")),
                    completed_files=completed_files,
                    reused_files=reused_files,
                    ingested_files=ingested_files,
                    completed_compressed_bytes=completed_bytes,
                    planned_compressed_bytes=contract["expected_compressed_bytes"],
                )

        for symbol in DEPTH_STRESS_EVALUATION_SYMBOLS:
            with progress_heartbeat(
                progress,
                phase="corpus_certification",
                interval_seconds=float(arguments.heartbeat_seconds),
                details={"symbol": symbol},
            ):
                certificates.append(
                    warehouse.require_corpus_certificate(
                        symbol,
                        required_data_types=(_DATA_TYPE,),
                        required_start_ms=certificate_bounds[0],
                        required_end_ms=certificate_bounds[1],
                        require_full_history_inventory=True,
                        allow_official_gap_data_types=(_DATA_TYPE,),
                    )
                )

    report_without_hash: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "round": 62,
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": contract["design_sha256"],
        "inventory_file_sha256": contract["inventory_file_sha256"],
        "warehouse": str(warehouse_path),
        "retained_archives": bool(arguments.retain_archives),
        "planned_files": contract["expected_files"],
        "completed_files": completed_files,
        "ingested_files": ingested_files,
        "reused_files": reused_files,
        "planned_compressed_bytes": contract["expected_compressed_bytes"],
        "completed_compressed_bytes": completed_bytes,
        "database_bytes": warehouse_path.stat().st_size,
        "certificate_required_start_ms": certificate_bounds[0],
        "certificate_required_end_ms": certificate_bounds[1],
        "certificates": certificates,
        "profitability_claim": False,
        "trading_authority": False,
    }
    report = {
        **report_without_hash,
        "report_sha256": _canonical_sha256(report_without_hash),
    }
    write_json_atomic(output_path, report, indent=2, sort_keys=True)
    progress(
        "round62_depth_corpus_complete",
        completed_files=completed_files,
        ingested_files=ingested_files,
        reused_files=reused_files,
        report_sha256=report["report_sha256"],
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default=str(DESIGN_DEFAULT))
    parser.add_argument("--inventory", default=str(INVENTORY_DEFAULT))
    parser.add_argument("--warehouse", default="data/microstructure.duckdb")
    parser.add_argument("--cache-root", default="data/archive-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--network-retries", type=int, default=12)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--retain-archives", action="store_true")
    return parser


def _write_terminal_progress(
    arguments: argparse.Namespace,
    *,
    event: str,
    error: str,
) -> None:
    path = Path(arguments.progress).resolve()
    previous: dict[str, object] = {}
    try:
        if path.is_file():
            previous = _read_object(path, "Round 62 ingestion progress")
    except (OSError, ValueError, json.JSONDecodeError):
        previous = {}
    try:
        sequence = int(previous.get("sequence", 0)) + 1
    except (TypeError, ValueError, OverflowError):
        sequence = 1
    payload = {
        "schema_version": "round-062-depth-ingestion-progress-v1",
        "round": 62,
        "sequence": sequence,
        "event": event,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "total_elapsed_seconds": previous.get("total_elapsed_seconds"),
        "details": {
            "error": error,
            "previous_event": previous.get("event"),
        },
    }
    try:
        write_json_atomic(path, payload, indent=2, sort_keys=True)
    except (OSError, ValueError) as exc:
        print(f"Round 62 terminal progress write failed: {exc}", file=sys.stderr)
        return
    print(_canonical_json(payload), file=sys.stderr, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        report = run(arguments)
    except KeyboardInterrupt:
        _write_terminal_progress(
            arguments,
            event="round62_depth_corpus_interrupted",
            error="KeyboardInterrupt",
        )
        print("Round 62 depth corpus ingestion interrupted; rerun to resume", file=sys.stderr)
        return 130
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_terminal_progress(
            arguments,
            event="round62_depth_corpus_failed",
            error=error,
        )
        print(f"Round 62 depth corpus ingestion failed: {error}", file=sys.stderr)
        return 2
    print(
        "Round 62 depth corpus: "
        f"status={report['status']} files={report['completed_files']} "
        f"ingested={report['ingested_files']} reused={report['reused_files']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
