"""Transactional DuckDB cache for verified tape/depth forecast datasets."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Mapping

import numpy as np

from .assets import normalize_symbol
from .microstructure_warehouse import MicrostructureWarehouse
from .tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_FEATURE_VERSION,
    TAPE_DEPTH_TARGET_MODE,
    TapeDepthForecastDataset,
    tape_depth_dataset_fingerprint,
)


TAPE_DEPTH_CACHE_SCHEMA_VERSION = "tape-depth-dataset-cache-v1"
_MANIFEST_TABLE = "tape_depth_dataset_cache_manifest"
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_ROWS_TABLE = "tape_depth_dataset_cache_rows_" + hashlib.sha256(
    TAPE_DEPTH_FEATURE_VERSION.encode("ascii")
).hexdigest()[:12]

_MANIFEST_COLUMNS = (
    ("cache_key", "VARCHAR"),
    ("schema_version", "VARCHAR"),
    ("feature_version", "VARCHAR"),
    ("symbol", "VARCHAR"),
    ("requested_start_ms", "BIGINT"),
    ("requested_end_ms", "BIGINT"),
    ("horizon_seconds", "INTEGER"),
    ("total_latency_ms", "INTEGER"),
    ("decision_cadence_seconds", "INTEGER"),
    ("maximum_depth_age_ms", "INTEGER"),
    ("row_count", "BIGINT"),
    ("first_decision_time_ms", "BIGINT"),
    ("last_decision_time_ms", "BIGINT"),
    ("source_manifest_fingerprint", "VARCHAR"),
    ("source_evidence_json", "VARCHAR"),
    ("dataset_fingerprint", "VARCHAR"),
    ("rows_table", "VARCHAR"),
    ("created_at_ms", "BIGINT"),
)
_BASE_ROW_COLUMNS = (
    ("cache_key", "VARCHAR"),
    ("decision_time_ms", "BIGINT"),
    ("target_entry_time_ms", "BIGINT"),
    ("target_exit_time_ms", "BIGINT"),
    ("target_entry_price", "DOUBLE"),
    ("target_exit_price", "DOUBLE"),
    ("gross_return_bps", "DOUBLE"),
)
_BASE_ROW_NAMES = frozenset(name for name, _data_type in _BASE_ROW_COLUMNS)
_ROW_COLUMNS = (*_BASE_ROW_COLUMNS, *((name, "FLOAT") for name in TAPE_DEPTH_FEATURE_NAMES))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_identifiers() -> None:
    identifiers = (_MANIFEST_TABLE, _ROWS_TABLE, *(name for name, _ in _ROW_COLUMNS))
    if any(_SAFE_IDENTIFIER.fullmatch(name) is None for name in identifiers):
        raise ValueError("tape/depth cache schema contains an unsafe identifier")


def _table_schema(warehouse: MicrostructureWarehouse, table: str) -> tuple[tuple[str, str], ...]:
    rows = warehouse.connect().execute(f"PRAGMA table_info('{table}')").fetchall()
    return tuple((str(row[1]), str(row[2]).upper()) for row in rows)


def _ensure_cache_schema(warehouse: MicrostructureWarehouse) -> None:
    _validate_identifiers()
    manifest_definitions = ",\n".join(
        f"{name} {data_type}{' PRIMARY KEY' if name == 'cache_key' else ' NOT NULL'}"
        for name, data_type in _MANIFEST_COLUMNS
    )
    row_definitions = ",\n".join(
        f"{name} {data_type}{' NOT NULL' if name in _BASE_ROW_NAMES else ''}"
        for name, data_type in _ROW_COLUMNS
    )
    connection = warehouse.connect()
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {_MANIFEST_TABLE} ({manifest_definitions})"
    )
    connection.execute(f"CREATE TABLE IF NOT EXISTS {_ROWS_TABLE} ({row_definitions})")
    if _table_schema(warehouse, _MANIFEST_TABLE) != _MANIFEST_COLUMNS:
        raise ValueError("tape/depth cache manifest schema drifted")
    if _table_schema(warehouse, _ROWS_TABLE) != _ROW_COLUMNS:
        raise ValueError("tape/depth cache row schema drifted")


def tape_depth_dataset_cache_key(
    *,
    symbol: str,
    requested_start_ms: int,
    requested_end_ms: int,
    horizon_seconds: int,
    total_latency_ms: int,
    decision_cadence_seconds: int,
    maximum_depth_age_ms: int,
    source_evidence: Mapping[str, object],
) -> str:
    if not bool(source_evidence.get("verified")) or not _is_sha256(
        source_evidence.get("manifest_fingerprint")
    ):
        raise ValueError("tape/depth cache requires verified source evidence")
    start_ms = int(requested_start_ms)
    end_ms = int(requested_end_ms)
    if start_ms > end_ms:
        raise ValueError("tape/depth cache interval is empty")
    contract = {
        "schema_version": TAPE_DEPTH_CACHE_SCHEMA_VERSION,
        "feature_version": TAPE_DEPTH_FEATURE_VERSION,
        "feature_names": list(TAPE_DEPTH_FEATURE_NAMES),
        "target_mode": TAPE_DEPTH_TARGET_MODE,
        "symbol": normalize_symbol(symbol),
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "horizon_seconds": int(horizon_seconds),
        "total_latency_ms": int(total_latency_ms),
        "decision_cadence_seconds": int(decision_cadence_seconds),
        "maximum_depth_age_ms": int(maximum_depth_age_ms),
        "source_evidence": dict(source_evidence),
    }
    return hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest()


def _cache_key_for_dataset(dataset: TapeDepthForecastDataset) -> str:
    if dataset.rows < 1:
        raise ValueError("cannot cache an empty tape/depth dataset")
    return tape_depth_dataset_cache_key(
        symbol=dataset.symbol,
        requested_start_ms=int(dataset.decision_time_ms[0]),
        requested_end_ms=int(dataset.decision_time_ms[-1]),
        horizon_seconds=dataset.horizon_seconds,
        total_latency_ms=dataset.total_latency_ms,
        decision_cadence_seconds=dataset.decision_cadence_seconds,
        maximum_depth_age_ms=dataset.maximum_depth_age_ms,
        source_evidence=dataset.source_evidence,
    )


def save_tape_depth_dataset_cache(
    warehouse: MicrostructureWarehouse,
    dataset: TapeDepthForecastDataset,
) -> str:
    if (
        dataset.feature_version != TAPE_DEPTH_FEATURE_VERSION
        or dataset.feature_names != TAPE_DEPTH_FEATURE_NAMES
        or dataset.target_mode != TAPE_DEPTH_TARGET_MODE
        or dataset.features.shape != (dataset.rows, len(TAPE_DEPTH_FEATURE_NAMES))
        or dataset.features.dtype != np.float32
    ):
        raise ValueError("tape/depth dataset cannot be stored under this cache contract")
    _ensure_cache_schema(warehouse)
    cache_key = _cache_key_for_dataset(dataset)
    dataset_fingerprint = tape_depth_dataset_fingerprint(dataset)
    connection = warehouse.connect()
    existing = connection.execute(
        f"SELECT dataset_fingerprint, row_count FROM {_MANIFEST_TABLE} WHERE cache_key = ?",
        [cache_key],
    ).fetchone()
    if existing is not None:
        if str(existing[0]) != dataset_fingerprint or int(existing[1]) != dataset.rows:
            raise ValueError("tape/depth cache key collides with different dataset evidence")
        return cache_key

    registered_name = f"cache_input_{cache_key[:16]}"
    mapping: dict[str, np.ndarray] = {
        "decision_time_ms": np.asarray(dataset.decision_time_ms, dtype=np.int64),
        "target_entry_time_ms": np.asarray(dataset.target_entry_time_ms, dtype=np.int64),
        "target_exit_time_ms": np.asarray(dataset.target_exit_time_ms, dtype=np.int64),
        "target_entry_price": np.asarray(dataset.target_entry_price, dtype=np.float64),
        "target_exit_price": np.asarray(dataset.target_exit_price, dtype=np.float64),
        "gross_return_bps": np.asarray(dataset.gross_return_bps, dtype=np.float64),
    }
    mapping.update(
        {
            name: np.asarray(dataset.features[:, index], dtype=np.float32)
            for index, name in enumerate(TAPE_DEPTH_FEATURE_NAMES)
        }
    )
    connection.register(registered_name, mapping)
    source_evidence_json = _canonical_json(dict(dataset.source_evidence))
    row_names = ", ".join(name for name, _ in _ROW_COLUMNS)
    projected_names = ", ".join(name for name, _ in _ROW_COLUMNS[1:])
    manifest_names = ", ".join(name for name, _ in _MANIFEST_COLUMNS)
    manifest_values = [
        cache_key,
        TAPE_DEPTH_CACHE_SCHEMA_VERSION,
        TAPE_DEPTH_FEATURE_VERSION,
        dataset.symbol,
        int(dataset.decision_time_ms[0]),
        int(dataset.decision_time_ms[-1]),
        int(dataset.horizon_seconds),
        int(dataset.total_latency_ms),
        int(dataset.decision_cadence_seconds),
        int(dataset.maximum_depth_age_ms),
        int(dataset.rows),
        int(dataset.decision_time_ms[0]),
        int(dataset.decision_time_ms[-1]),
        str(dataset.source_evidence["manifest_fingerprint"]),
        source_evidence_json,
        dataset_fingerprint,
        _ROWS_TABLE,
        int(datetime.now(tz=UTC).timestamp() * 1_000),
    ]
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            f"INSERT INTO {_ROWS_TABLE} ({row_names}) "
            f"SELECT ? AS cache_key, {projected_names} FROM {registered_name}",
            [cache_key],
        )
        inserted = connection.execute(
            f"SELECT count(*), min(decision_time_ms), max(decision_time_ms) "
            f"FROM {_ROWS_TABLE} WHERE cache_key = ?",
            [cache_key],
        ).fetchone()
        if inserted != (
            dataset.rows,
            int(dataset.decision_time_ms[0]),
            int(dataset.decision_time_ms[-1]),
        ):
            raise ValueError("tape/depth cache row write was incomplete")
        placeholders = ", ".join("?" for _ in _MANIFEST_COLUMNS)
        connection.execute(
            f"INSERT INTO {_MANIFEST_TABLE} ({manifest_names}) VALUES ({placeholders})",
            manifest_values,
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.unregister(registered_name)
    return cache_key


def _as_float_array(value: object) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        return np.asarray(
            np.ma.asarray(value, dtype=np.float64).filled(np.nan),
            dtype=np.float64,
        )
    return np.asarray(value, dtype=np.float64)


def load_tape_depth_dataset_cache(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    requested_start_ms: int,
    requested_end_ms: int,
    horizon_seconds: int,
    total_latency_ms: int,
    decision_cadence_seconds: int,
    maximum_depth_age_ms: int,
    source_evidence: Mapping[str, object],
) -> TapeDepthForecastDataset | None:
    _ensure_cache_schema(warehouse)
    cache_key = tape_depth_dataset_cache_key(
        symbol=symbol,
        requested_start_ms=requested_start_ms,
        requested_end_ms=requested_end_ms,
        horizon_seconds=horizon_seconds,
        total_latency_ms=total_latency_ms,
        decision_cadence_seconds=decision_cadence_seconds,
        maximum_depth_age_ms=maximum_depth_age_ms,
        source_evidence=source_evidence,
    )
    connection = warehouse.connect()
    manifest = connection.execute(
        f"SELECT * FROM {_MANIFEST_TABLE} WHERE cache_key = ?",
        [cache_key],
    ).fetchone()
    if manifest is None:
        return None
    metadata = {
        name: manifest[index] for index, (name, _data_type) in enumerate(_MANIFEST_COLUMNS)
    }
    expected_source_json = _canonical_json(dict(source_evidence))
    if (
        metadata["schema_version"] != TAPE_DEPTH_CACHE_SCHEMA_VERSION
        or metadata["feature_version"] != TAPE_DEPTH_FEATURE_VERSION
        or metadata["symbol"] != normalize_symbol(symbol)
        or int(metadata["requested_start_ms"]) != int(requested_start_ms)
        or int(metadata["requested_end_ms"]) != int(requested_end_ms)
        or int(metadata["horizon_seconds"]) != int(horizon_seconds)
        or int(metadata["total_latency_ms"]) != int(total_latency_ms)
        or int(metadata["decision_cadence_seconds"]) != int(decision_cadence_seconds)
        or int(metadata["maximum_depth_age_ms"]) != int(maximum_depth_age_ms)
        or metadata["source_manifest_fingerprint"]
        != source_evidence.get("manifest_fingerprint")
        or metadata["source_evidence_json"] != expected_source_json
        or metadata["rows_table"] != _ROWS_TABLE
        or not _is_sha256(metadata["dataset_fingerprint"])
    ):
        raise ValueError("tape/depth cache manifest failed its evidence contract")

    projected_names = ", ".join(name for name, _ in _ROW_COLUMNS[1:])
    values = connection.execute(
        f"SELECT {projected_names} FROM {_ROWS_TABLE} "
        "WHERE cache_key = ? ORDER BY decision_time_ms",
        [cache_key],
    ).fetchnumpy()
    decision_times = np.asarray(values.pop("decision_time_ms"), dtype=np.int64)
    target_entry_times = np.asarray(values.pop("target_entry_time_ms"), dtype=np.int64)
    target_exit_times = np.asarray(values.pop("target_exit_time_ms"), dtype=np.int64)
    entry_prices = _as_float_array(values.pop("target_entry_price"))
    exit_prices = _as_float_array(values.pop("target_exit_price"))
    targets = _as_float_array(values.pop("gross_return_bps"))
    feature_matrix = np.empty(
        (len(decision_times), len(TAPE_DEPTH_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for index, name in enumerate(TAPE_DEPTH_FEATURE_NAMES):
        feature_matrix[:, index] = _as_float_array(values.pop(name))
    if values:
        raise ValueError("tape/depth cache returned unexpected columns")
    if (
        len(decision_times) != int(metadata["row_count"])
        or len(decision_times) < 1
        or int(decision_times[0]) != int(metadata["first_decision_time_ms"])
        or int(decision_times[-1]) != int(metadata["last_decision_time_ms"])
    ):
        raise ValueError("tape/depth cache rows are incomplete")
    cached_source_evidence = json.loads(expected_source_json)
    dataset = TapeDepthForecastDataset(
        symbol=normalize_symbol(symbol),
        feature_version=TAPE_DEPTH_FEATURE_VERSION,
        feature_names=TAPE_DEPTH_FEATURE_NAMES,
        target_mode=TAPE_DEPTH_TARGET_MODE,
        horizon_seconds=int(horizon_seconds),
        total_latency_ms=int(total_latency_ms),
        decision_cadence_seconds=int(decision_cadence_seconds),
        maximum_depth_age_ms=int(maximum_depth_age_ms),
        decision_time_ms=decision_times,
        target_entry_time_ms=target_entry_times,
        target_exit_time_ms=target_exit_times,
        target_entry_price=entry_prices,
        target_exit_price=exit_prices,
        gross_return_bps=targets,
        features=feature_matrix,
        source_evidence=cached_source_evidence,
    )
    if tape_depth_dataset_fingerprint(dataset) != metadata["dataset_fingerprint"]:
        raise ValueError("tape/depth cache dataset fingerprint differs")
    return dataset


__all__ = [
    "TAPE_DEPTH_CACHE_SCHEMA_VERSION",
    "load_tape_depth_dataset_cache",
    "save_tape_depth_dataset_cache",
    "tape_depth_dataset_cache_key",
]
