"""Ingest the exact frozen Round 72 spot/perpetual flow corpus."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import requests

from simple_ai_trading.microstructure_warehouse import create_archive_http_session
from simple_ai_trading.progress_heartbeat import progress_heartbeat
from simple_ai_trading.spot_perpetual_corpus import (
    SpotPerpetualCorpusStore,
    load_frozen_round72_contract,
)
from simple_ai_trading.storage import write_json_atomic


DESIGN_DEFAULT = Path(
    "docs/model-research/action-value/round-072-spot-perpetual-price-discovery-design.json"
)
INVENTORY_DEFAULT = Path(
    "docs/model-research/action-value/round-072-spot-perpetual-inventory.json"
)
WAREHOUSE_DEFAULT = Path("data/microstructure.duckdb")
OUTPUT_DEFAULT = Path("data/round72-spot-perpetual-corpus-ingestion.json")
PROGRESS_DEFAULT = Path("data/round72-spot-perpetual-corpus-ingestion.progress.json")
_REPORT_SCHEMA = "round-072-spot-perpetual-corpus-ingestion-v1"


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
                "schema_version": "round-072-corpus-ingestion-progress-v1",
                "round": 72,
                "sequence": self.sequence,
                "event": event,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "total_elapsed_seconds": round(time.monotonic() - self.started, 3),
                "details": details,
            }
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            print(_canonical_json(payload), file=sys.stderr, flush=True)


class ProgressThrottle:
    def __init__(
        self,
        writer: ProgressWriter,
        *,
        period: str,
        day_index: int,
        total_days: int,
        interval_seconds: float,
    ) -> None:
        self.writer = writer
        self.period = period
        self.day_index = day_index
        self.total_days = total_days
        self.interval_seconds = interval_seconds
        self.last_emit = 0.0

    def __call__(self, phase: str, current: int, total: int | None) -> None:
        now = time.monotonic()
        finished = total is not None and int(current) >= int(total)
        if not finished and now - self.last_emit < self.interval_seconds:
            return
        self.last_emit = now
        self.writer(
            "source_progress",
            period=self.period,
            day_index=self.day_index,
            total_days=self.total_days,
            phase=phase,
            current=int(current),
            total=None if total is None else int(total),
        )


def _terminal_report(
    *,
    status: str,
    event: str,
    error: str,
    started_at_utc: str,
) -> dict[str, object]:
    without_hash: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "round": 72,
        "status": status,
        "event": event,
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "profitability_claim": False,
        "trading_authority": False,
        "error": error,
    }
    return {**without_hash, "report_sha256": _canonical_sha256(without_hash)}


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started_at_utc = datetime.now(UTC).isoformat()
    design_path = Path(arguments.design).resolve()
    inventory_path = Path(arguments.inventory).resolve()
    warehouse_path = Path(arguments.warehouse).resolve()
    cache_root = Path(arguments.cache_root).resolve()
    output_path = Path(arguments.output).resolve()
    progress_path = Path(arguments.progress).resolve()
    if len(
        {
            design_path,
            inventory_path,
            warehouse_path,
            output_path,
            progress_path,
        }
    ) != 5:
        raise ValueError("Round 72 ingestion paths must be distinct")
    if (
        not 1 <= int(arguments.threads) <= 32
        or not 1 <= int(arguments.network_retries) <= 100
        or not 1.0 <= float(arguments.heartbeat_seconds) <= 300.0
        or not 1.0 <= float(arguments.timeout) <= 900.0
        or not 1 <= int(arguments.maximum_uncompressed_gb) <= 16
    ):
        raise ValueError("Round 72 ingestion runtime limits are invalid")
    contract = load_frozen_round72_contract(design_path, inventory_path)
    progress = ProgressWriter(progress_path)
    progress(
        "frozen_contract_verified",
        design_sha256=contract.design_sha256,
        inventory_sha256=contract.inventory_sha256,
        days=len(contract.days),
        files=contract.expected_files,
        compressed_bytes=contract.selected_compressed_bytes,
        expected_flow_rows=contract.expected_rows,
    )

    completed_days = 0
    ingested_days = 0
    reused_days = 0
    completed_files = 0
    completed_bytes = 0
    day_results: list[dict[str, object]] = []
    with (
        SpotPerpetualCorpusStore(
            warehouse_path,
            cache_root=cache_root,
            memory_limit=str(arguments.memory_limit),
            threads=int(arguments.threads),
        ) as store,
        create_archive_http_session() as session,
    ):
        for day_index, day in enumerate(contract.days, start=1):
            throttle = ProgressThrottle(
                progress,
                period=day.period,
                day_index=day_index,
                total_days=len(contract.days),
                interval_seconds=float(arguments.heartbeat_seconds),
            )
            result = None
            for attempt in range(1, int(arguments.network_retries) + 1):
                try:
                    with progress_heartbeat(
                        progress,
                        phase="spot_perpetual_day_ingestion",
                        interval_seconds=float(arguments.heartbeat_seconds),
                        details={
                            "period": day.period,
                            "day_index": day_index,
                            "total_days": len(contract.days),
                            "attempt": attempt,
                        },
                    ):
                        result = store.ingest_day(
                            day,
                            inventory_sha256=contract.inventory_sha256,
                            timeout_seconds=float(arguments.timeout),
                            maximum_uncompressed_bytes=(
                                int(arguments.maximum_uncompressed_gb) * 1024**3
                            ),
                            session=session,
                            progress=throttle,
                        )
                    break
                except requests.RequestException as exc:
                    if attempt >= int(arguments.network_retries):
                        raise
                    delay = min(300.0, float(2 ** min(attempt, 8)))
                    progress(
                        "network_retry",
                        period=day.period,
                        day_index=day_index,
                        total_days=len(contract.days),
                        attempt=attempt,
                        retry_in_seconds=delay,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    time.sleep(delay)
            if result is None:
                raise RuntimeError("Round 72 day ingestion ended without a result")
            completed_days += 1
            completed_files += result.source_count
            completed_bytes += result.compressed_bytes
            if result.status == "skipped":
                reused_days += 1
            else:
                ingested_days += 1
            day_results.append(
                {
                    "day_id": result.day_id,
                    "period": result.period,
                    "status": result.status,
                    "source_count": result.source_count,
                    "flow_rows": result.flow_rows,
                    "compressed_bytes": result.compressed_bytes,
                    "uncompressed_bytes": result.uncompressed_bytes,
                    "combined_flow_sha256": result.combined_flow_sha256,
                }
            )
            progress(
                "day_completed",
                period=day.period,
                day_index=day_index,
                total_days=len(contract.days),
                status=result.status,
                completed_days=completed_days,
                ingested_days=ingested_days,
                reused_days=reused_days,
                completed_files=completed_files,
                completed_compressed_bytes=completed_bytes,
                planned_compressed_bytes=contract.selected_compressed_bytes,
            )

        with progress_heartbeat(
            progress,
            phase="spot_perpetual_corpus_certification",
            interval_seconds=float(arguments.heartbeat_seconds),
            details={
                "days": len(contract.days),
                "expected_flow_rows": contract.expected_rows,
            },
        ):
            certificate = store.certify_corpus(contract)
        if (
            int(certificate["day_count"]) != len(contract.days)
            or int(certificate["source_count"]) != contract.expected_files
            or int(certificate["flow_rows"]) != contract.expected_rows
            or int(certificate["compressed_bytes"])
            != contract.selected_compressed_bytes
        ):
            raise ValueError("Round 72 corpus certificate totals differ")
        store.connect().execute("CHECKPOINT")

    retained_selected_archives = []
    for day in contract.days:
        for archive in day.archives:
            market = "spot" if archive.market_type == "spot" else "usdm"
            candidate = (
                cache_root
                / "binance"
                / market
                / "aggTrades"
                / archive.symbol
                / Path(archive.url).name
            )
            if candidate.exists():
                retained_selected_archives.append(str(candidate))
    if retained_selected_archives:
        raise ValueError("Round 72 selected ZIPs remain after certified ingestion")

    without_hash: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "round": 72,
        "status": "complete",
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "design_path": str(design_path),
        "design_sha256": contract.design_sha256,
        "inventory_path": str(inventory_path),
        "inventory_sha256": contract.inventory_sha256,
        "inventory_file_sha256": contract.inventory_file_sha256,
        "warehouse_path": str(warehouse_path),
        "warehouse_bytes": warehouse_path.stat().st_size,
        "completed_days": completed_days,
        "ingested_days": ingested_days,
        "reused_days": reused_days,
        "completed_files": completed_files,
        "completed_compressed_bytes": completed_bytes,
        "raw_aggregate_trades_retained": False,
        "selected_archives_retained": False,
        "profitability_claim": False,
        "execution_or_fill_claim": False,
        "trading_authority": False,
        "day_results": day_results,
        "corpus_certificate": certificate,
    }
    report = {**without_hash, "report_sha256": _canonical_sha256(without_hash)}
    write_json_atomic(output_path, report, indent=2, sort_keys=True)
    progress(
        "round72_spot_perpetual_corpus_complete",
        report_sha256=report["report_sha256"],
        days=completed_days,
        files=completed_files,
        flow_rows=certificate["flow_rows"],
        warehouse_bytes=without_hash["warehouse_bytes"],
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default=str(DESIGN_DEFAULT))
    parser.add_argument("--inventory", default=str(INVENTORY_DEFAULT))
    parser.add_argument("--warehouse", default=str(WAREHOUSE_DEFAULT))
    parser.add_argument("--cache-root", default="data/archive-cache")
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--progress", default=str(PROGRESS_DEFAULT))
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--network-retries", type=int, default=12)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-uncompressed-gb", type=int, default=8)
    return parser


def _write_terminal_artifacts(
    arguments: argparse.Namespace,
    report: dict[str, object],
) -> None:
    output_path = Path(arguments.output).resolve()
    progress_path = Path(arguments.progress).resolve()
    protected = {
        Path(arguments.design).resolve(),
        Path(arguments.inventory).resolve(),
        Path(arguments.warehouse).resolve(),
    }
    if output_path not in protected and output_path != progress_path:
        write_json_atomic(output_path, report, indent=2, sort_keys=True)
    if progress_path not in protected and progress_path != output_path:
        ProgressWriter(progress_path)(
            str(report["event"]),
            report_sha256=report["report_sha256"],
            error=report["error"],
        )


def main() -> int:
    arguments = build_parser().parse_args()
    started_at_utc = datetime.now(UTC).isoformat()
    try:
        run(arguments)
    except KeyboardInterrupt:
        report = _terminal_report(
            status="interrupted",
            event="round72_spot_perpetual_corpus_interrupted",
            error="KeyboardInterrupt",
            started_at_utc=started_at_utc,
        )
        _write_terminal_artifacts(arguments, report)
        return 130
    except Exception as exc:
        report = _terminal_report(
            status="failed",
            event="round72_spot_perpetual_corpus_failed",
            error=f"{type(exc).__name__}: {exc}",
            started_at_utc=started_at_utc,
        )
        _write_terminal_artifacts(arguments, report)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
