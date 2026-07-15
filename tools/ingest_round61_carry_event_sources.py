"""Ingest and certify Round 61's filtered spot and mark-price event rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.api import Candle  # noqa: E402
from simple_ai_trading.binance_archive import (  # noqa: E402
    _download_to_temp,
    _fetch_archive_checksum,
    _parse_archive_row,
)
from simple_ai_trading.derivatives_archive import (  # noqa: E402
    _canonical_row_digest_update,
    _is_header,
    _validated_csv_member,
)
from simple_ai_trading.market_store import (  # noqa: E402
    FuturesReferenceBar,
    MarketDataStore,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round59_funding_persistence_feasibility import (  # noqa: E402
    SYMBOLS,
    _canonical_sha256,
    _file_sha256,
    _period_bounds_ms,
    _read_object,
)


ROUND = 61
DESIGN_SCHEMA = "round-061-carry-economic-replay-design-v3"
MANIFEST_SCHEMA = "round-061-carry-event-manifest-v2"
CERTIFICATE_SCHEMA = "round-061-carry-event-source-certificate-v1"
DESIGN_SHA256 = "544faedcb31348e08e24212980a6660a79d9e0459b9517dc3f457864ecaf777c"
MANIFEST_SHA256 = "8b5a8037176c5e37af2c261c0ab79dd9f43f6e0d9024e78f6306694293126594"
MANIFEST_FILE_SHA256 = (
    "65a5c20b2ad8a85add95d49f5ea94260d36062c8fed004dfd5ee7e310814700f"
)
SPOT_SOURCE = "binance_public_archive_event_filter_round61"
MARK_SOURCE = "binance_public_archive_markPriceKlines_event_filter_round61"
FUTURES_SOURCE = "binance_public_archive"
LEGACY_CHECKPOINT_DESIGN_SHA256 = (
    "6ee379609c42e480122cff49d9fa3deaa73952857e99ceb8af2175b7c2e4d8f3"
)
LEGACY_CHECKPOINT_IMPLEMENTATION_COMMIT = "80d96fa507b69f38e1869a08a1db6c9f93ca1534"


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _validate_design(path: Path) -> dict[str, object]:
    design = _read_object(path, "Round 61 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    source = design.get("source_contract", {})
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != DESIGN_SHA256
        or _canonical_sha256(canonical) != claimed
        or design.get("event_contract", {}).get("manifest_file_sha256")
        != MANIFEST_FILE_SHA256
        or design.get("event_contract", {}).get("manifest_canonical_sha256")
        != MANIFEST_SHA256
        or source.get("persistent_zip_archive_permitted") is not False
        or source.get("synthetic_interpolation_permitted") is not False
        or source.get("forward_or_backward_fill_permitted") is not False
        or source.get("REST_historical_trade_fallback_permitted") is not False
        or source.get("duplicate_required_row_is_fatal") is not True
        or source.get("missing_required_row_makes_affected_episode_source_ineligible")
        is not True
        or source.get("missing_required_row_is_never_interpolated_or_filled")
        is not True
        or source.get("source_ineligible_episodes_are_not_economically_scored")
        is not True
        or source.get("minimum_source_eligible_fraction_per_symbol") != 0.9
        or source.get("minimum_source_eligible_episodes_per_symbol") != 40
    ):
        raise ValueError("Round 61 source design drifted")
    return design


def _validate_manifest(path: Path) -> dict[str, object]:
    if _file_sha256(path) != MANIFEST_FILE_SHA256:
        raise ValueError("Round 61 event manifest file hash drifted")
    manifest = _read_object(path, "Round 61 event manifest")
    canonical = dict(manifest)
    claimed = str(canonical.pop("manifest_sha256", ""))
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("round") != ROUND
        or claimed != MANIFEST_SHA256
        or _canonical_sha256(canonical) != claimed
        or manifest.get("price_values_read") is not False
        or manifest.get("source_alignment_revision") != 2
        or manifest.get("kline_open_time_mapping")
        != "floor(raw_event_timestamp_ms / 60000) * 60000"
        or tuple(manifest.get("symbols", ())) != SYMBOLS
    ):
        raise ValueError("Round 61 event manifest drifted")
    for item in manifest["symbol_manifests"]:
        if any(
            int(value) % 60_000
            for key in (
                "required_spot_open_times_ms",
                "required_futures_execution_open_times_ms",
                "required_mark_open_times_ms",
            )
            for value in item[key]
        ):
            raise ValueError("Round 61 required kline timestamp is not minute-aligned")
    return manifest


def _candle_values(candle: Candle) -> tuple[object, ...]:
    return (
        candle.open_time,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.close_time,
        candle.quote_volume,
        candle.trade_count,
        candle.taker_buy_base_volume,
        candle.taker_buy_quote_volume,
    )


def _reference_values(candle: Candle) -> tuple[object, ...]:
    return (
        candle.open_time,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.close_time,
    )


def _validate_candle(candle: Candle, *, mark_price: bool) -> None:
    values = (
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.quote_volume,
        candle.taker_buy_base_volume,
        candle.taker_buy_quote_volume,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("archive row contains a non-finite value")
    if min(candle.open, candle.high, candle.low, candle.close) <= 0.0:
        raise ValueError("archive row contains a nonpositive price")
    if candle.high < max(candle.open, candle.low, candle.close) or candle.low > min(
        candle.open, candle.high, candle.close
    ):
        raise ValueError("archive row has invalid OHLC bounds")
    if candle.volume < 0.0 or candle.quote_volume < 0.0 or candle.trade_count < 0:
        raise ValueError("archive row has invalid activity")
    if not mark_price and (
        candle.taker_buy_base_volume < 0.0
        or candle.taker_buy_quote_volume < 0.0
        or candle.taker_buy_base_volume > candle.volume + 1e-9
        or candle.taker_buy_quote_volume > candle.quote_volume + 1e-6
    ):
        raise ValueError("archive row has invalid taker-buy activity")


def _parse_filtered_archive(
    path: Path,
    *,
    period: str,
    required_times: Sequence[int],
    mark_price: bool,
    allow_missing_required: bool = False,
) -> tuple[list[Candle], dict[str, object]]:
    required = set(int(value) for value in required_times)
    period_start, period_end = _period_bounds_ms(period)
    selected: list[Candle] = []
    full_digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    previous_time: int | None = None
    first_time: int | None = None
    rows = 0
    gaps = 0
    missing_minutes = 0
    with zipfile.ZipFile(path) as archive:
        member = _validated_csv_member(archive)
        with archive.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for raw_row in reader:
                candle = _parse_archive_row(raw_row)
                if candle is None:
                    if _is_header(raw_row):
                        continue
                    raise ValueError("archive contains an invalid non-header row")
                _validate_candle(candle, mark_price=mark_price)
                if not period_start <= candle.open_time < period_end:
                    raise ValueError("archive row is outside its declared month")
                if previous_time is not None:
                    if candle.open_time <= previous_time:
                        raise ValueError(
                            "archive timestamps are not strictly increasing"
                        )
                    delta = candle.open_time - previous_time
                    if delta != 60_000:
                        if delta % 60_000:
                            raise ValueError(
                                "archive timestamp is off the one-minute grid"
                            )
                        gaps += 1
                        missing_minutes += delta // 60_000 - 1
                values = (
                    _reference_values(candle) if mark_price else _candle_values(candle)
                )
                _canonical_row_digest_update(full_digest, values)
                if candle.open_time in required:
                    selected.append(candle)
                    _canonical_row_digest_update(selected_digest, values)
                if first_time is None:
                    first_time = candle.open_time
                previous_time = candle.open_time
                rows += 1
    observed = [candle.open_time for candle in selected]
    missing = sorted(required - set(observed))
    if missing and not allow_missing_required:
        raise ValueError(f"archive is missing {len(missing)} required rows")
    evidence = {
        "full_rows": rows,
        "first_open_time_ms": first_time,
        "last_open_time_ms": previous_time,
        "gap_count": gaps,
        "missing_minutes": missing_minutes,
        "full_row_stream_sha256": full_digest.hexdigest(),
        "selected_rows": len(selected),
        "selected_row_stream_sha256": selected_digest.hexdigest(),
        "missing_required_rows": len(missing),
        "missing_required_open_times_ms": missing,
    }
    return selected, evidence


def _stored_selected_digest(
    store: MarketDataStore,
    *,
    symbol: str,
    required_times: Sequence[int],
    mark_price: bool,
) -> tuple[int, str]:
    connection = store.connect()
    digest = hashlib.sha256()
    observed = 0
    for start in range(0, len(required_times), 400):
        times = list(required_times[start : start + 400])
        placeholders = ",".join("?" for _ in times)
        if mark_price:
            rows = connection.execute(
                f"""
                SELECT open_time, open, high, low, close, close_time
                FROM futures_reference_bars
                WHERE symbol=? AND market_type='futures' AND kind='mark_price'
                  AND interval='1m' AND source=?
                  AND open_time IN ({placeholders})
                ORDER BY open_time
                """,
                (symbol, MARK_SOURCE, *times),
            ).fetchall()
            for row in rows:
                _canonical_row_digest_update(
                    digest,
                    (
                        int(row["open_time"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        int(row["close_time"]),
                    ),
                )
        else:
            rows = connection.execute(
                f"""
                SELECT open_time, open, high, low, close, volume, close_time,
                       quote_volume, trade_count, taker_buy_base_volume,
                       taker_buy_quote_volume
                FROM candles
                WHERE symbol=? AND market_type='spot' AND interval='1m'
                  AND source=? AND open_time IN ({placeholders})
                ORDER BY open_time
                """,
                (symbol, SPOT_SOURCE, *times),
            ).fetchall()
            for row in rows:
                _canonical_row_digest_update(
                    digest,
                    (
                        int(row["open_time"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                        int(row["close_time"]),
                        float(row["quote_volume"]),
                        int(row["trade_count"]),
                        float(row["taker_buy_base_volume"]),
                        float(row["taker_buy_quote_volume"]),
                    ),
                )
        observed += len(rows)
    return observed, digest.hexdigest()


def _certificate_payload(
    *,
    implementation_commit: str,
    database_file: str,
    complete: bool,
    archive_evidence: Sequence[Mapping[str, object]],
    futures_evidence: Sequence[Mapping[str, object]],
    series_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "round": ROUND,
        "design_sha256": DESIGN_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "implementation_commit": implementation_commit,
        "database_file": database_file,
        "complete": complete,
        "persistent_zip_archive_created": False,
        "symbols": list(SYMBOLS),
        "filtered_archive_evidence": list(archive_evidence),
        "existing_futures_archive_evidence": list(futures_evidence),
        "series_evidence": list(series_evidence),
        "source_certificate_sha256": "PENDING",
    }
    canonical = dict(payload)
    canonical.pop("source_certificate_sha256")
    payload["source_certificate_sha256"] = _canonical_sha256(canonical)
    return payload


def _load_checkpoint(
    path: Path, *, implementation_commit: str
) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    checkpoint = _read_object(path, "Round 61 source checkpoint")
    canonical = dict(checkpoint)
    claimed = str(canonical.pop("source_certificate_sha256", ""))
    identity = (
        str(checkpoint.get("design_sha256", "")),
        str(checkpoint.get("implementation_commit", "")),
    )
    allowed_identities = {
        (DESIGN_SHA256, implementation_commit),
        (
            LEGACY_CHECKPOINT_DESIGN_SHA256,
            LEGACY_CHECKPOINT_IMPLEMENTATION_COMMIT,
        ),
    }
    if (
        checkpoint.get("schema_version") != CERTIFICATE_SCHEMA
        or checkpoint.get("manifest_sha256") != MANIFEST_SHA256
        or identity not in allowed_identities
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 61 source checkpoint identity drifted")
    evidence = checkpoint.get("filtered_archive_evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("Round 61 source checkpoint archive evidence is invalid")
    normalized: list[dict[str, object]] = []
    for raw_row in evidence:
        if not isinstance(raw_row, dict):
            raise ValueError("Round 61 source checkpoint row is invalid")
        row = dict(raw_row)
        required = int(row.get("required_rows", -1))
        selected = int(row.get("selected_rows", -1))
        missing = required - selected
        if required < 0 or selected < 0 or missing < 0:
            raise ValueError("Round 61 source checkpoint row counts are invalid")
        row.setdefault("missing_required_rows", missing)
        row.setdefault("missing_required_open_times_ms", [])
        if int(row["missing_required_rows"]) != missing:
            raise ValueError("Round 61 source checkpoint missing-row count drifted")
        missing_times = row["missing_required_open_times_ms"]
        if not isinstance(missing_times, list) or len(missing_times) != missing:
            raise ValueError("Round 61 source checkpoint missing timestamps drifted")
        normalized.append(row)
    return normalized


def _write_checkpoint(
    path: Path,
    *,
    implementation_commit: str,
    database_file: str,
    archive_evidence: Sequence[Mapping[str, object]],
) -> None:
    payload = _certificate_payload(
        implementation_commit=implementation_commit,
        database_file=database_file,
        complete=False,
        archive_evidence=archive_evidence,
        futures_evidence=[],
        series_evidence=[],
    )
    write_json_atomic(path, payload, indent=2)


def _ingest_filtered_archives(
    store: MarketDataStore,
    *,
    manifest: Mapping[str, object],
    output: Path,
    implementation_commit: str,
    timeout: int,
) -> list[dict[str, object]]:
    existing = _load_checkpoint(output, implementation_commit=implementation_commit)
    by_key = {(row["kind"], row["symbol"], row["period"]): row for row in existing}
    tasks: list[tuple[str, str, str, str, list[int], bool]] = []
    for item in manifest["symbol_manifests"]:
        symbol = str(item["symbol"])
        spot_by_month: dict[str, list[int]] = {}
        for value in item["required_spot_open_times_ms"]:
            period = time.strftime("%Y-%m", time.gmtime(int(value) / 1000))
            spot_by_month.setdefault(period, []).append(int(value))
        mark_by_month: dict[str, list[int]] = {}
        for value in item["required_mark_open_times_ms"]:
            period = time.strftime("%Y-%m", time.gmtime(int(value) / 1000))
            mark_by_month.setdefault(period, []).append(int(value))
        tasks.extend(
            ("spot", symbol, period, url, sorted(spot_by_month[period]), False)
            for period, url in zip(
                item["spot_archive_months"], item["spot_archive_urls"], strict=True
            )
        )
        tasks.extend(
            ("mark_price", symbol, period, url, sorted(mark_by_month[period]), True)
            for period, url in zip(
                item["mark_archive_months"], item["mark_archive_urls"], strict=True
            )
        )
    archive_evidence = list(existing)
    for index, (kind, symbol, period, url, required, mark_price) in enumerate(tasks, 1):
        key = (kind, symbol, period)
        prior = by_key.get(key)
        if prior is not None:
            observed, digest = _stored_selected_digest(
                store,
                symbol=symbol,
                required_times=required,
                mark_price=mark_price,
            )
            if (
                prior.get("url") == url
                and prior.get("required_rows") == len(required)
                and observed == int(prior.get("selected_rows", -1))
                and digest == prior.get("selected_row_stream_sha256")
            ):
                print(
                    json.dumps(
                        {
                            "completed": index,
                            "total": len(tasks),
                            "kind": kind,
                            "symbol": symbol,
                            "period": period,
                            "status": "checkpoint-verified",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            raise ValueError(f"checkpointed rows drifted: {kind} {symbol} {period}")
        zip_path: Path | None = None
        try:
            zip_path, bytes_downloaded, archive_sha = _download_to_temp(
                url, timeout=timeout
            )
            checksum_sha = (
                _fetch_archive_checksum(url, timeout=max(1, min(timeout, 30))) or ""
            )
            if not checksum_sha or checksum_sha.lower() != archive_sha.lower():
                raise ValueError(f"archive checksum failed: {kind} {symbol} {period}")
            selected, parsed = _parse_filtered_archive(
                zip_path,
                period=period,
                required_times=required,
                mark_price=mark_price,
                allow_missing_required=True,
            )
            ingested_at_ms = int(time.time() * 1000)
            if mark_price:
                inserted = store.upsert_futures_reference_bars(
                    [
                        FuturesReferenceBar(
                            symbol=symbol,
                            market_type="futures",
                            kind="mark_price",
                            interval="1m",
                            open_time=candle.open_time,
                            open=candle.open,
                            high=candle.high,
                            low=candle.low,
                            close=candle.close,
                            close_time=candle.close_time,
                        )
                        for candle in selected
                    ],
                    source=MARK_SOURCE,
                    ingested_at_ms=ingested_at_ms,
                )
            else:
                inserted = store.upsert_candles(
                    symbol,
                    "spot",
                    "1m",
                    selected,
                    source=SPOT_SOURCE,
                    ingested_at_ms=ingested_at_ms,
                )
            observed, selected_digest = _stored_selected_digest(
                store,
                symbol=symbol,
                required_times=required,
                mark_price=mark_price,
            )
            if (
                observed != int(parsed["selected_rows"])
                or selected_digest != parsed["selected_row_stream_sha256"]
            ):
                raise ValueError(f"stored rows failed audit: {kind} {symbol} {period}")
            evidence = {
                "kind": kind,
                "symbol": symbol,
                "period": period,
                "url": url,
                "required_rows": len(required),
                "rows_inserted_or_updated": inserted,
                "bytes_downloaded": bytes_downloaded,
                "archive_sha256": archive_sha,
                "checksum_sha256": checksum_sha,
                "checksum_status": "verified",
                **parsed,
            }
            archive_evidence.append(evidence)
            by_key[key] = evidence
            archive_evidence.sort(
                key=lambda row: (row["kind"], row["symbol"], row["period"])
            )
            _write_checkpoint(
                output,
                implementation_commit=implementation_commit,
                database_file=store.path.name,
                archive_evidence=archive_evidence,
            )
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(tasks),
                        "kind": kind,
                        "symbol": symbol,
                        "period": period,
                        "status": "complete",
                        "archive_rows": parsed["full_rows"],
                        "selected_rows": parsed["selected_rows"],
                        "missing_required_rows": parsed["missing_required_rows"],
                        "bytes_downloaded": bytes_downloaded,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                flush=True,
            )
        finally:
            if zip_path is not None:
                try:
                    zip_path.unlink()
                except OSError:
                    pass
    return sorted(
        archive_evidence, key=lambda row: (row["kind"], row["symbol"], row["period"])
    )


def _audit_existing_futures(
    store: MarketDataStore, manifest: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    connection = store.connect()
    archive_evidence: list[dict[str, object]] = []
    series: list[dict[str, object]] = []
    for item in manifest["symbol_manifests"]:
        symbol = str(item["symbol"])
        urls = list(item["futures_execution_archive_urls"])
        metadata = []
        for url in urls:
            row = connection.execute(
                """
                SELECT url, period, status, sha256, checksum_sha256,
                       checksum_status, rows_inserted, bytes_downloaded
                FROM archive_files WHERE url=?
                """,
                (url,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "complete"
                or row["checksum_status"] != "verified"
                or row["sha256"] != row["checksum_sha256"]
                or len(str(row["sha256"])) != 64
            ):
                raise ValueError(f"existing futures archive drifted: {url}")
            metadata.append(row)
            archive_evidence.append(
                {
                    "kind": "futures_execution",
                    "symbol": symbol,
                    "period": row["period"],
                    "url": row["url"],
                    "rows_inserted": int(row["rows_inserted"]),
                    "bytes_downloaded": int(row["bytes_downloaded"]),
                    "archive_sha256": row["sha256"],
                    "checksum_sha256": row["checksum_sha256"],
                    "checksum_status": row["checksum_status"],
                }
            )
        required = list(item["required_futures_execution_open_times_ms"])
        digest = hashlib.sha256()
        observed = 0
        for start in range(0, len(required), 400):
            times = required[start : start + 400]
            placeholders = ",".join("?" for _ in times)
            rows = connection.execute(
                f"""
                SELECT open_time, open, high, low, close, volume, close_time,
                       quote_volume, trade_count, taker_buy_base_volume,
                       taker_buy_quote_volume
                FROM candles
                WHERE symbol=? AND market_type='futures' AND interval='1m'
                  AND source=? AND open_time IN ({placeholders})
                ORDER BY open_time
                """,
                (symbol, FUTURES_SOURCE, *times),
            ).fetchall()
            for row in rows:
                _canonical_row_digest_update(
                    digest,
                    (
                        int(row["open_time"]),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                        int(row["close_time"]),
                        float(row["quote_volume"]),
                        int(row["trade_count"]),
                        float(row["taker_buy_base_volume"]),
                        float(row["taker_buy_quote_volume"]),
                    ),
                )
            observed += len(rows)
        if observed != len(required):
            raise ValueError(f"{symbol} futures execution rows are incomplete")
        series.append(
            {
                "symbol": symbol,
                "kind": "futures_execution",
                "required_rows": len(required),
                "stored_rows": observed,
                "selected_row_stream_sha256": digest.hexdigest(),
                "archive_count": len(metadata),
            }
        )
    return archive_evidence, series


def _filtered_series_evidence(
    store: MarketDataStore, manifest: Mapping[str, object]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for item in manifest["symbol_manifests"]:
        symbol = str(item["symbol"])
        for kind, key, mark in (
            ("spot_execution", "required_spot_open_times_ms", False),
            ("funding_mark_price", "required_mark_open_times_ms", True),
        ):
            required = list(item[key])
            observed, digest = _stored_selected_digest(
                store,
                symbol=symbol,
                required_times=required,
                mark_price=mark,
            )
            output.append(
                {
                    "symbol": symbol,
                    "kind": kind,
                    "required_rows": len(required),
                    "stored_rows": observed,
                    "missing_required_rows": len(required) - observed,
                    "required_row_availability_fraction": observed / len(required),
                    "selected_row_stream_sha256": digest,
                }
            )
    return output


def run(arguments: argparse.Namespace) -> dict[str, object]:
    _validate_design(arguments.design.resolve())
    manifest = _validate_manifest(arguments.manifest.resolve())
    if _git("status", "--porcelain"):
        raise ValueError("Round 61 source ingestion requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    database = arguments.database.resolve()
    output = arguments.output.resolve()
    with MarketDataStore(database) as store:
        filtered = _ingest_filtered_archives(
            store,
            manifest=manifest,
            output=output,
            implementation_commit=implementation_commit,
            timeout=arguments.timeout,
        )
        futures_archives, futures_series = _audit_existing_futures(store, manifest)
        series = _filtered_series_evidence(store, manifest) + futures_series
    payload = _certificate_payload(
        implementation_commit=implementation_commit,
        database_file=database.name,
        complete=True,
        archive_evidence=filtered,
        futures_evidence=futures_archives,
        series_evidence=sorted(series, key=lambda row: (row["symbol"], row["kind"])),
    )
    write_json_atomic(output, payload, indent=2)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-061-carry-economic-replay-design.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-061-carry-event-manifest.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    certificate = run(_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "round": certificate["round"],
                "complete": certificate["complete"],
                "filtered_archives": len(certificate["filtered_archive_evidence"]),
                "source_certificate_sha256": certificate["source_certificate_sha256"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
