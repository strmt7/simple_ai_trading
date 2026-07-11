"""Transactional DuckDB cache for source-bound microstructure datasets."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Mapping

import numpy as np

from .assets import normalize_symbol
from .microstructure_features import (
    AGGREGATE_DEPTH_FEATURE_VERSION,
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MICROSTRUCTURE_TRADE_EMBARGO_MS,
    MicrostructureDataset,
    microstructure_feature_names,
    validate_microstructure_dataset,
)
from .microstructure_warehouse import MicrostructureWarehouse


MICROSTRUCTURE_CACHE_SCHEMA_VERSION = "exact-bbo-dataset-cache-v1"
AGGREGATE_DEPTH_CACHE_SCHEMA_VERSION = "aggregate-depth-dataset-cache-v1"
_MANIFEST_TABLE = "microstructure_dataset_cache_manifest"
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_ROWS_TABLE = (
    "microstructure_dataset_cache_rows_"
    + hashlib.sha256(MICROSTRUCTURE_FEATURE_VERSION.encode("ascii")).hexdigest()[:12]
)

_MANIFEST_COLUMNS = (
    ("cache_key", "VARCHAR"),
    ("schema_version", "VARCHAR"),
    ("feature_version", "VARCHAR"),
    ("symbol", "VARCHAR"),
    ("requested_start_ms", "BIGINT"),
    ("requested_end_ms", "BIGINT"),
    ("horizon_seconds", "INTEGER"),
    ("total_latency_ms", "INTEGER"),
    ("taker_fee_bps", "DOUBLE"),
    ("additional_slippage_bps_per_side", "DOUBLE"),
    ("reference_order_notional_quote", "DOUBLE"),
    ("max_l1_participation", "DOUBLE"),
    ("max_quote_age_ms", "INTEGER"),
    ("decision_cadence_seconds", "INTEGER"),
    ("require_full_history_inventory", "BOOLEAN"),
    ("row_count", "BIGINT"),
    ("first_decision_time_ms", "BIGINT"),
    ("last_decision_time_ms", "BIGINT"),
    ("source_manifest_fingerprint", "VARCHAR"),
    ("source_evidence_json", "VARCHAR"),
    ("dataset_fingerprint", "VARCHAR"),
    ("rows_table", "VARCHAR"),
    ("created_at_ms", "BIGINT"),
)
_ARRAY_FIELDS = (
    ("decision_time_ms", "decision_time_ms", "BIGINT", "<i8"),
    ("long_exit_time_ms", "long_exit_time_ms", "BIGINT", "<i8"),
    ("short_exit_time_ms", "short_exit_time_ms", "BIGINT", "<i8"),
    ("long_net_bps", "long_net_bps", "DOUBLE", "<f8"),
    ("short_net_bps", "short_net_bps", "DOUBLE", "<f8"),
    ("entry_spread_bps", "entry_spread_bps", "DOUBLE", "<f8"),
    ("exit_spread_bps", "exit_spread_bps", "DOUBLE", "<f8"),
    ("entry_quote_age_ms", "entry_quote_age_ms", "BIGINT", "<i8"),
    ("exit_quote_age_ms", "exit_quote_age_ms", "BIGINT", "<i8"),
    ("entry_bid_price", "entry_bid_price", "DOUBLE", "<f8"),
    ("entry_ask_price", "entry_ask_price", "DOUBLE", "<f8"),
    ("fixed_exit_bid_price", "fixed_exit_bid_price", "DOUBLE", "<f8"),
    ("fixed_exit_ask_price", "fixed_exit_ask_price", "DOUBLE", "<f8"),
    ("entry_bid_qty", "entry_bid_qty", "DOUBLE", "<f8"),
    ("entry_ask_qty", "entry_ask_qty", "DOUBLE", "<f8"),
    ("fixed_exit_bid_qty", "fixed_exit_bid_qty", "DOUBLE", "<f8"),
    ("fixed_exit_ask_qty", "fixed_exit_ask_qty", "DOUBLE", "<f8"),
    ("long_l1_participation", "long_l1_participation", "DOUBLE", "<f8"),
    ("short_l1_participation", "short_l1_participation", "DOUBLE", "<f8"),
    ("long_liquidity_eligible", "long_liquidity_eligible", "BOOLEAN", "|b1"),
    ("short_liquidity_eligible", "short_liquidity_eligible", "BOOLEAN", "|b1"),
)
_ROW_COLUMNS = (
    ("cache_key", "VARCHAR"),
    *((name, data_type) for name, _attribute, data_type, _dtype in _ARRAY_FIELDS),
    *((name, "FLOAT") for name in MICROSTRUCTURE_FEATURE_NAMES),
)


def _cache_schema_version(feature_version: str) -> str:
    return (
        AGGREGATE_DEPTH_CACHE_SCHEMA_VERSION
        if feature_version == AGGREGATE_DEPTH_FEATURE_VERSION
        else MICROSTRUCTURE_CACHE_SCHEMA_VERSION
    )


def _rows_table(feature_version: str) -> str:
    if feature_version == MICROSTRUCTURE_FEATURE_VERSION:
        return _ROWS_TABLE
    return (
        "microstructure_dataset_cache_rows_"
        + hashlib.sha256(feature_version.encode("ascii")).hexdigest()[:12]
    )


def _row_columns(feature_version: str) -> tuple[tuple[str, str], ...]:
    if feature_version == MICROSTRUCTURE_FEATURE_VERSION:
        return _ROW_COLUMNS
    return (
        ("cache_key", "VARCHAR"),
        *((name, data_type) for name, _attribute, data_type, _dtype in _ARRAY_FIELDS),
        *((name, "FLOAT") for name in microstructure_feature_names(feature_version)),
    )


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
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validated_source_evidence(value: Mapping[str, object]) -> dict[str, object]:
    evidence = dict(value)
    certificate = evidence.get("corpus_certificate")
    if (
        not evidence.get("verified")
        or not evidence.get("is_current")
        or not _is_sha256(evidence.get("manifest_fingerprint"))
        or not isinstance(certificate, Mapping)
        or certificate.get("status") != "pass"
        or not certificate.get("verified")
        or not _is_sha256(certificate.get("certificate_sha256"))
    ):
        raise ValueError(
            "microstructure cache requires current verified source evidence"
        )
    return evidence


def _validate_identifiers(
    feature_version: str = MICROSTRUCTURE_FEATURE_VERSION,
) -> None:
    rows_table = _rows_table(feature_version)
    row_columns = _row_columns(feature_version)
    identifiers = (_MANIFEST_TABLE, rows_table, *(name for name, _ in row_columns))
    if any(_SAFE_IDENTIFIER.fullmatch(name) is None for name in identifiers):
        raise ValueError("microstructure cache schema contains an unsafe identifier")


def _table_schema(
    warehouse: MicrostructureWarehouse,
    table: str,
) -> tuple[tuple[str, str], ...]:
    rows = warehouse.connect().execute(f"PRAGMA table_info('{table}')").fetchall()
    return tuple((str(row[1]), str(row[2]).upper()) for row in rows)


def _ensure_cache_schema(
    warehouse: MicrostructureWarehouse,
    feature_version: str = MICROSTRUCTURE_FEATURE_VERSION,
) -> None:
    _validate_identifiers(feature_version)
    rows_table = _rows_table(feature_version)
    row_columns = _row_columns(feature_version)
    manifest_definitions = ",\n".join(
        f"{name} {data_type}{' PRIMARY KEY' if name == 'cache_key' else ' NOT NULL'}"
        for name, data_type in _MANIFEST_COLUMNS
    )
    row_definitions = ",\n".join(
        f"{name} {data_type} NOT NULL" for name, data_type in row_columns
    )
    connection = warehouse.connect()
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS {_MANIFEST_TABLE} ({manifest_definitions})"
    )
    connection.execute(f"CREATE TABLE IF NOT EXISTS {rows_table} ({row_definitions})")
    if _table_schema(warehouse, _MANIFEST_TABLE) != _MANIFEST_COLUMNS:
        raise ValueError("microstructure cache manifest schema drifted")
    if _table_schema(warehouse, rows_table) != row_columns:
        raise ValueError("microstructure cache row schema drifted")


def _dataset_contract(dataset: MicrostructureDataset) -> dict[str, object]:
    return {
        "symbol": dataset.symbol,
        "feature_version": dataset.feature_version,
        "feature_names": list(dataset.feature_names),
        "horizon_seconds": dataset.horizon_seconds,
        "total_latency_ms": dataset.total_latency_ms,
        "taker_fee_bps": dataset.taker_fee_bps,
        "additional_slippage_bps_per_side": (dataset.additional_slippage_bps_per_side),
        "reference_order_notional_quote": dataset.reference_order_notional_quote,
        "max_l1_participation": dataset.max_l1_participation,
        "max_quote_age_ms": dataset.max_quote_age_ms,
        "decision_cadence_seconds": dataset.decision_cadence_seconds,
        "target_mode": dataset.target_mode,
        "trade_feature_embargo_ms": dataset.trade_feature_embargo_ms,
        "source_evidence": dict(dataset.source_evidence or {}),
    }


def microstructure_dataset_fingerprint(dataset: MicrostructureDataset) -> str:
    """Hash every exact feature, label, timing, and provenance field."""

    validate_microstructure_dataset(dataset)
    digest = hashlib.sha256(_canonical_json(_dataset_contract(dataset)).encode("ascii"))
    for name, attribute, _data_type, dtype in _ARRAY_FIELDS:
        values = np.ascontiguousarray(
            getattr(dataset, attribute), dtype=np.dtype(dtype)
        )
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    features = np.ascontiguousarray(dataset.features, dtype="<f4")
    digest.update(b"features\x00")
    digest.update(np.asarray(features.shape, dtype="<i8").tobytes())
    digest.update(features.tobytes())
    return digest.hexdigest()


def microstructure_dataset_cache_key(
    *,
    symbol: str,
    requested_start_ms: int,
    requested_end_ms: int,
    horizon_seconds: int,
    total_latency_ms: int,
    taker_fee_bps: float,
    additional_slippage_bps_per_side: float,
    reference_order_notional_quote: float,
    max_l1_participation: float,
    max_quote_age_ms: int,
    decision_cadence_seconds: int,
    require_full_history_inventory: bool,
    source_evidence: Mapping[str, object],
    feature_version: str = MICROSTRUCTURE_FEATURE_VERSION,
) -> str:
    evidence = _validated_source_evidence(source_evidence)
    start_ms = int(requested_start_ms)
    end_ms = int(requested_end_ms)
    if start_ms > end_ms:
        raise ValueError("microstructure cache interval is empty")
    selected_feature_version = str(feature_version)
    selected_feature_names = microstructure_feature_names(selected_feature_version)
    contract = {
        "schema_version": _cache_schema_version(selected_feature_version),
        "feature_version": selected_feature_version,
        "feature_names": list(selected_feature_names),
        "target_mode": "fixed_horizon",
        "symbol": normalize_symbol(symbol),
        "requested_start_ms": start_ms,
        "requested_end_ms": end_ms,
        "horizon_seconds": int(horizon_seconds),
        "total_latency_ms": int(total_latency_ms),
        "taker_fee_bps": float(taker_fee_bps),
        "additional_slippage_bps_per_side": float(additional_slippage_bps_per_side),
        "reference_order_notional_quote": float(reference_order_notional_quote),
        "max_l1_participation": float(max_l1_participation),
        "max_quote_age_ms": int(max_quote_age_ms),
        "decision_cadence_seconds": int(decision_cadence_seconds),
        "require_full_history_inventory": bool(require_full_history_inventory),
        "source_evidence": evidence,
    }
    return hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest()


def _cache_parameters(
    dataset: MicrostructureDataset,
    *,
    requested_start_ms: int,
    requested_end_ms: int,
    require_full_history_inventory: bool,
) -> dict[str, object]:
    if dataset.source_evidence is None:
        raise ValueError("microstructure cache dataset has no source evidence")
    return {
        "symbol": dataset.symbol,
        "requested_start_ms": int(requested_start_ms),
        "requested_end_ms": int(requested_end_ms),
        "horizon_seconds": dataset.horizon_seconds,
        "total_latency_ms": dataset.total_latency_ms,
        "taker_fee_bps": dataset.taker_fee_bps,
        "additional_slippage_bps_per_side": (dataset.additional_slippage_bps_per_side),
        "reference_order_notional_quote": dataset.reference_order_notional_quote,
        "max_l1_participation": dataset.max_l1_participation,
        "max_quote_age_ms": dataset.max_quote_age_ms,
        "decision_cadence_seconds": dataset.decision_cadence_seconds,
        "require_full_history_inventory": bool(require_full_history_inventory),
        "source_evidence": dataset.source_evidence,
        "feature_version": dataset.feature_version,
    }


def save_microstructure_dataset_cache(
    warehouse: MicrostructureWarehouse,
    dataset: MicrostructureDataset,
    *,
    requested_start_ms: int,
    requested_end_ms: int,
    require_full_history_inventory: bool,
) -> str:
    """Atomically cache one immutable fixed-horizon microstructure dataset."""

    expected_feature_names = microstructure_feature_names(dataset.feature_version)
    if (
        dataset.rows < 1
        or dataset.feature_names != expected_feature_names
        or dataset.target_mode != "fixed_horizon"
        or dataset.stop_loss_bps is not None
        or dataset.take_profit_bps is not None
        or dataset.path_resolution_ms is not None
        or dataset.features.dtype != np.float32
    ):
        raise ValueError("microstructure dataset cannot be cached under this contract")
    validate_microstructure_dataset(dataset)
    parameters = _cache_parameters(
        dataset,
        requested_start_ms=requested_start_ms,
        requested_end_ms=requested_end_ms,
        require_full_history_inventory=require_full_history_inventory,
    )
    cache_key = microstructure_dataset_cache_key(**parameters)
    dataset_fingerprint = microstructure_dataset_fingerprint(dataset)
    _ensure_cache_schema(warehouse, dataset.feature_version)
    rows_table = _rows_table(dataset.feature_version)
    row_columns = _row_columns(dataset.feature_version)
    connection = warehouse.connect()
    existing = connection.execute(
        f"SELECT dataset_fingerprint, row_count FROM {_MANIFEST_TABLE} "
        "WHERE cache_key = ?",
        [cache_key],
    ).fetchone()
    if existing is not None:
        if str(existing[0]) != dataset_fingerprint or int(existing[1]) != dataset.rows:
            raise ValueError(
                "microstructure cache key collides with different evidence"
            )
        return cache_key

    mapping = {
        name: np.asarray(getattr(dataset, attribute), dtype=np.dtype(dtype))
        for name, attribute, _data_type, dtype in _ARRAY_FIELDS
    }
    mapping.update(
        {
            name: np.asarray(dataset.features[:, index], dtype=np.float32)
            for index, name in enumerate(expected_feature_names)
        }
    )
    registered_name = f"microstructure_cache_input_{cache_key[:12]}"
    connection.register(registered_name, mapping)
    evidence = _validated_source_evidence(dataset.source_evidence or {})
    source_json = _canonical_json(evidence)
    row_names = ", ".join(name for name, _data_type in row_columns)
    projected_names = ", ".join(name for name, _data_type in row_columns[1:])
    manifest_names = ", ".join(name for name, _data_type in _MANIFEST_COLUMNS)
    manifest_values = [
        cache_key,
        _cache_schema_version(dataset.feature_version),
        dataset.feature_version,
        dataset.symbol,
        int(requested_start_ms),
        int(requested_end_ms),
        dataset.horizon_seconds,
        dataset.total_latency_ms,
        dataset.taker_fee_bps,
        dataset.additional_slippage_bps_per_side,
        dataset.reference_order_notional_quote,
        dataset.max_l1_participation,
        dataset.max_quote_age_ms,
        dataset.decision_cadence_seconds,
        bool(require_full_history_inventory),
        dataset.rows,
        int(dataset.decision_time_ms[0]),
        int(dataset.decision_time_ms[-1]),
        str(evidence["manifest_fingerprint"]),
        source_json,
        dataset_fingerprint,
        rows_table,
        int(datetime.now(tz=UTC).timestamp() * 1_000),
    ]
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.execute(
            f"INSERT INTO {rows_table} ({row_names}) "
            f"SELECT ? AS cache_key, {projected_names} FROM {registered_name}",
            [cache_key],
        )
        inserted = connection.execute(
            f"SELECT count(*), min(decision_time_ms), max(decision_time_ms) "
            f"FROM {rows_table} WHERE cache_key = ?",
            [cache_key],
        ).fetchone()
        if inserted != (
            dataset.rows,
            int(dataset.decision_time_ms[0]),
            int(dataset.decision_time_ms[-1]),
        ):
            raise ValueError("microstructure cache row write was incomplete")
        placeholders = ", ".join("?" for _name, _data_type in _MANIFEST_COLUMNS)
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


def _numeric_array(value: object, dtype: str) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        return np.asarray(np.ma.asarray(value).filled(np.nan), dtype=np.dtype(dtype))
    return np.asarray(value, dtype=np.dtype(dtype))


def load_microstructure_dataset_cache(
    warehouse: MicrostructureWarehouse,
    **parameters: object,
) -> MicrostructureDataset | None:
    """Load and re-hash one exact cache match, or return ``None`` on a miss."""

    cache_key = microstructure_dataset_cache_key(**parameters)  # type: ignore[arg-type]
    selected_feature_version = str(
        parameters.get("feature_version", MICROSTRUCTURE_FEATURE_VERSION)
    )
    selected_feature_names = microstructure_feature_names(selected_feature_version)
    _ensure_cache_schema(warehouse, selected_feature_version)
    rows_table = _rows_table(selected_feature_version)
    row_columns = _row_columns(selected_feature_version)
    connection = warehouse.connect()
    manifest = connection.execute(
        f"SELECT * FROM {_MANIFEST_TABLE} WHERE cache_key = ?",
        [cache_key],
    ).fetchone()
    if manifest is None:
        return None
    metadata = {
        name: manifest[index]
        for index, (name, _data_type) in enumerate(_MANIFEST_COLUMNS)
    }
    evidence = _validated_source_evidence(parameters["source_evidence"])  # type: ignore[arg-type]
    expected_source_json = _canonical_json(evidence)
    expected = {
        "feature_version": selected_feature_version,
        "symbol": normalize_symbol(str(parameters["symbol"])),
        "requested_start_ms": int(parameters["requested_start_ms"]),
        "requested_end_ms": int(parameters["requested_end_ms"]),
        "horizon_seconds": int(parameters["horizon_seconds"]),
        "total_latency_ms": int(parameters["total_latency_ms"]),
        "taker_fee_bps": float(parameters["taker_fee_bps"]),
        "additional_slippage_bps_per_side": float(
            parameters["additional_slippage_bps_per_side"]
        ),
        "reference_order_notional_quote": float(
            parameters["reference_order_notional_quote"]
        ),
        "max_l1_participation": float(parameters["max_l1_participation"]),
        "max_quote_age_ms": int(parameters["max_quote_age_ms"]),
        "decision_cadence_seconds": int(parameters["decision_cadence_seconds"]),
        "require_full_history_inventory": bool(
            parameters["require_full_history_inventory"]
        ),
    }
    if (
        metadata["schema_version"] != _cache_schema_version(selected_feature_version)
        or any(metadata[name] != value for name, value in expected.items())
        or metadata["source_manifest_fingerprint"] != evidence["manifest_fingerprint"]
        or metadata["source_evidence_json"] != expected_source_json
        or metadata["rows_table"] != rows_table
        or not _is_sha256(metadata["dataset_fingerprint"])
    ):
        raise ValueError("microstructure cache manifest failed its evidence contract")

    projected_names = ", ".join(name for name, _data_type in row_columns[1:])
    values = connection.execute(
        f"SELECT {projected_names} FROM {rows_table} "
        "WHERE cache_key = ? ORDER BY decision_time_ms",
        [cache_key],
    ).fetchnumpy()
    arrays = {
        attribute: _numeric_array(values.pop(name), dtype)
        for name, attribute, _data_type, dtype in _ARRAY_FIELDS
    }
    features = np.empty(
        (len(arrays["decision_time_ms"]), len(selected_feature_names)),
        dtype=np.float32,
    )
    for index, name in enumerate(selected_feature_names):
        features[:, index] = _numeric_array(values.pop(name), "<f4")
    if values:
        raise ValueError("microstructure cache returned unexpected columns")
    rows = len(arrays["decision_time_ms"])
    if (
        rows != int(metadata["row_count"])
        or rows < 1
        or int(arrays["decision_time_ms"][0]) != int(metadata["first_decision_time_ms"])
        or int(arrays["decision_time_ms"][-1]) != int(metadata["last_decision_time_ms"])
    ):
        raise ValueError("microstructure cache rows are incomplete")
    dataset = MicrostructureDataset(
        symbol=expected["symbol"],
        feature_version=selected_feature_version,
        feature_names=selected_feature_names,
        horizon_seconds=expected["horizon_seconds"],
        total_latency_ms=expected["total_latency_ms"],
        taker_fee_bps=expected["taker_fee_bps"],
        additional_slippage_bps_per_side=expected["additional_slippage_bps_per_side"],
        reference_order_notional_quote=expected["reference_order_notional_quote"],
        max_l1_participation=expected["max_l1_participation"],
        max_quote_age_ms=expected["max_quote_age_ms"],
        decision_cadence_seconds=expected["decision_cadence_seconds"],
        target_mode="fixed_horizon",
        stop_loss_bps=None,
        take_profit_bps=None,
        trigger_execution_slippage_bps=None,
        path_resolution_ms=None,
        features=features,
        source_evidence=evidence,
        trade_feature_embargo_ms=MICROSTRUCTURE_TRADE_EMBARGO_MS,
        **arrays,
    )
    validate_microstructure_dataset(dataset)
    if microstructure_dataset_fingerprint(dataset) != metadata["dataset_fingerprint"]:
        raise ValueError("microstructure cached dataset fingerprint differs")
    return dataset


__all__ = [
    "AGGREGATE_DEPTH_CACHE_SCHEMA_VERSION",
    "MICROSTRUCTURE_CACHE_SCHEMA_VERSION",
    "load_microstructure_dataset_cache",
    "microstructure_dataset_cache_key",
    "microstructure_dataset_fingerprint",
    "save_microstructure_dataset_cache",
]
