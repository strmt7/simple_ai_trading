"""Append-only prospective shadow evidence for independent Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock
import time
import zlib

from .polymarket_round21_prospective import (
    Round21ProspectivePrediction,
    validate_round21_prospective_prediction,
)


POLYMARKET_ROUND21_SHADOW_STORE_SCHEMA_VERSION = (
    "polymarket-round21-prospective-shadow-store-v1"
)
POLYMARKET_ROUND21_SHADOW_RUN_SCHEMA_VERSION = (
    "polymarket-round21-prospective-shadow-run-v1"
)
POLYMARKET_ROUND21_SHADOW_RECORD_SCHEMA_VERSION = (
    "polymarket-round21-prospective-shadow-record-v1"
)
POLYMARKET_ROUND21_SHADOW_TERMINAL_SCHEMA_VERSION = (
    "polymarket-round21-prospective-shadow-terminal-v1"
)

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_LAYERS = frozenset(("core", "core_spot", "core_spot_usdm"))
_TERMINAL_STATUSES = frozenset(("complete", "interrupted", "failed"))
_MAX_RAW_PREDICTION_BYTES = 256 * 1024
_MAX_COMPRESSED_PREDICTION_BYTES = 64 * 1024
_MAX_TERMINAL_REASON_CHARS = 512


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_bytes(value))


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 shadow {name} digest is invalid")
    return selected


def _run_id(value: object) -> str:
    selected = str(value or "").strip().lower()
    if _RUN_ID.fullmatch(selected) is None:
        raise ValueError("Round 21 shadow run ID is invalid")
    return selected


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Round 21 shadow {name} is invalid")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 21 shadow prediction contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 shadow prediction contains {value}")


def _compress_prediction(
    prediction: Round21ProspectivePrediction,
) -> tuple[bytes, bytes]:
    raw = _canonical_bytes(prediction.validated().asdict())
    if not 1 <= len(raw) <= _MAX_RAW_PREDICTION_BYTES:
        raise ValueError("Round 21 shadow prediction payload is too large")
    compressed = zlib.compress(raw, level=6)
    if not 1 <= len(compressed) <= _MAX_COMPRESSED_PREDICTION_BYTES:
        raise ValueError("Round 21 shadow compressed prediction is too large")
    return raw, compressed


def _decode_prediction(
    payload: bytes,
    *,
    raw_byte_count: int,
    payload_sha256: str,
    compressed_payload_sha256: str,
) -> Round21ProspectivePrediction:
    if (
        not isinstance(payload, bytes)
        or not 1 <= len(payload) <= _MAX_COMPRESSED_PREDICTION_BYTES
        or _sha256(payload) != compressed_payload_sha256
        or not 1 <= raw_byte_count <= _MAX_RAW_PREDICTION_BYTES
    ):
        raise ValueError("Round 21 shadow compressed prediction differs")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(payload, _MAX_RAW_PREDICTION_BYTES + 1)
    except zlib.error as exc:
        raise ValueError("Round 21 shadow prediction compression is invalid") from exc
    if (
        len(raw) > _MAX_RAW_PREDICTION_BYTES
        or decoder.unconsumed_tail
        or decoder.unused_data
        or not decoder.eof
        or len(raw) != raw_byte_count
        or _sha256(raw) != payload_sha256
    ):
        raise ValueError("Round 21 shadow prediction payload differs")
    try:
        parsed = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 21 shadow prediction JSON is invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("Round 21 shadow prediction JSON is not an object")
    prediction = validate_round21_prospective_prediction(parsed)
    if _canonical_bytes(prediction.asdict()) != raw:
        raise ValueError("Round 21 shadow prediction serialization differs")
    return prediction


@dataclass(frozen=True, slots=True)
class Round21ShadowRun:
    run_id: str
    source_model_artifact_sha256: str
    sealed_result_sha256: str
    population_layer: str
    started_at_ms: int
    run_sha256: str
    target_accessed: bool = False
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    grants_execution_authority: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        source_model_artifact_sha256: str,
        sealed_result_sha256: str,
        population_layer: str,
        started_at_ms: int,
    ) -> Round21ShadowRun:
        selected_id = _run_id(run_id)
        model_sha = _digest(source_model_artifact_sha256, name="model artifact")
        sealed_sha = _digest(sealed_result_sha256, name="sealed result")
        layer = str(population_layer or "").strip()
        started = _positive_int(started_at_ms, name="start timestamp")
        if layer not in _LAYERS:
            raise ValueError("Round 21 shadow population layer is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_SHADOW_RUN_SCHEMA_VERSION,
            "run_id": selected_id,
            "source_model_artifact_sha256": model_sha,
            "sealed_result_sha256": sealed_sha,
            "population_layer": layer,
            "started_at_ms": started,
            "target_accessed": False,
            "credentials_used": False,
            "account_connected": False,
            "binance_execution_connected": False,
            "grants_execution_authority": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        return cls(
            run_id=selected_id,
            source_model_artifact_sha256=model_sha,
            sealed_result_sha256=sealed_sha,
            population_layer=layer,
            started_at_ms=started,
            run_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21ShadowRun:
        rebuilt = self.create(
            run_id=self.run_id,
            source_model_artifact_sha256=self.source_model_artifact_sha256,
            sealed_result_sha256=self.sealed_result_sha256,
            population_layer=self.population_layer,
            started_at_ms=self.started_at_ms,
        )
        if self != rebuilt:
            raise ValueError("Round 21 shadow run identity differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21StoredShadowPrediction:
    run_id: str
    sequence_number: int
    condition_id: str
    decision_time_ms: int
    prediction_sha256: str
    previous_record_sha256: str
    record_sha256: str
    recorded_at_ms: int
    prediction: Round21ProspectivePrediction


@dataclass(frozen=True, slots=True)
class Round21ShadowTerminal:
    run_id: str
    sequence_number: int
    status: str
    reason: str
    previous_record_sha256: str
    finished_at_ms: int
    terminal_sha256: str
    grants_execution_authority: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False


@dataclass(frozen=True, slots=True)
class Round21ShadowAudit:
    run: Round21ShadowRun
    prediction_count: int
    observed_count: int
    abstention_count: int
    last_record_sha256: str
    terminal: Round21ShadowTerminal | None
    integrity_passed: bool = True
    target_accessed: bool = False
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    grants_execution_authority: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False


class Round21ProspectiveShadowStore:
    """Durable evidence ledger with no account, order, or promotion authority."""

    credentials_used = False
    account_connected = False
    binance_execution_connected = False
    grants_execution_authority = False
    trading_authority = False
    paper_trading_authority = False
    live_trading_authority = False

    def __init__(
        self,
        path: str | Path,
        *,
        wall_time_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        selected = Path(path)
        if selected.exists() and selected.is_symlink():
            raise ValueError("Round 21 shadow database may not be a symlink")
        if not callable(wall_time_ms):
            raise TypeError("Round 21 shadow wall clock is invalid")
        selected.parent.mkdir(parents=True, exist_ok=True)
        self.path = selected
        self._wall_time_ms = wall_time_ms
        self._lock = Lock()
        self._connection = sqlite3.connect(
            selected,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._configure_and_initialize()
        except Exception:
            self._connection.close()
            raise

    def __enter__(self) -> Round21ProspectiveShadowStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _configure_and_initialize(self) -> None:
        connection = self._connection
        connection.execute("PRAGMA busy_timeout=30000")
        mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0])
        if mode.lower() != "delete":
            raise RuntimeError("Round 21 shadow journal mode differs")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-8192")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS round21_shadow_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round21_shadow_run (
                run_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                source_model_artifact_sha256 TEXT NOT NULL,
                sealed_result_sha256 TEXT NOT NULL,
                population_layer TEXT NOT NULL,
                started_at_ms INTEGER NOT NULL,
                run_sha256 TEXT NOT NULL UNIQUE,
                CHECK (length(run_id) = 32),
                CHECK (length(source_model_artifact_sha256) = 64),
                CHECK (length(sealed_result_sha256) = 64),
                CHECK (population_layer IN ('core', 'core_spot', 'core_spot_usdm')),
                CHECK (started_at_ms > 0),
                CHECK (length(run_sha256) = 64)
            );
            CREATE TABLE IF NOT EXISTS round21_shadow_prediction (
                run_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                condition_id TEXT NOT NULL,
                decision_time_ms INTEGER NOT NULL,
                prediction_sha256 TEXT NOT NULL,
                previous_record_sha256 TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                compressed_payload_sha256 TEXT NOT NULL,
                raw_byte_count INTEGER NOT NULL,
                compressed_byte_count INTEGER NOT NULL,
                payload BLOB NOT NULL,
                recorded_at_ms INTEGER NOT NULL,
                record_sha256 TEXT NOT NULL UNIQUE,
                PRIMARY KEY (run_id, sequence_number),
                UNIQUE (run_id, condition_id, decision_time_ms),
                FOREIGN KEY (run_id) REFERENCES round21_shadow_run(run_id),
                CHECK (sequence_number > 0),
                CHECK (length(condition_id) = 66),
                CHECK (decision_time_ms > 0),
                CHECK (length(prediction_sha256) = 64),
                CHECK (length(previous_record_sha256) = 64),
                CHECK (length(payload_sha256) = 64),
                CHECK (length(compressed_payload_sha256) = 64),
                CHECK (raw_byte_count > 0 AND raw_byte_count <= 262144),
                CHECK (compressed_byte_count > 0 AND compressed_byte_count <= 65536),
                CHECK (length(payload) = compressed_byte_count),
                CHECK (recorded_at_ms > 0),
                CHECK (length(record_sha256) = 64)
            );
            CREATE TABLE IF NOT EXISTS round21_shadow_terminal (
                run_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                previous_record_sha256 TEXT NOT NULL,
                finished_at_ms INTEGER NOT NULL,
                terminal_sha256 TEXT NOT NULL UNIQUE,
                FOREIGN KEY (run_id) REFERENCES round21_shadow_run(run_id),
                CHECK (sequence_number > 0),
                CHECK (status IN ('complete', 'interrupted', 'failed')),
                CHECK (length(reason) <= 512),
                CHECK (length(previous_record_sha256) = 64),
                CHECK (finished_at_ms > 0),
                CHECK (length(terminal_sha256) = 64)
            );
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO round21_shadow_metadata (key, value) VALUES (?, ?)",
            ("schema_version", POLYMARKET_ROUND21_SHADOW_STORE_SCHEMA_VERSION),
        )
        row = connection.execute(
            "SELECT value FROM round21_shadow_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or str(row[0]) != POLYMARKET_ROUND21_SHADOW_STORE_SCHEMA_VERSION:
            raise RuntimeError("Round 21 shadow store schema differs")
        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS round21_shadow_metadata_no_update
            BEFORE UPDATE ON round21_shadow_metadata BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow metadata is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_metadata_no_delete
            BEFORE DELETE ON round21_shadow_metadata BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow metadata is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_run_no_update
            BEFORE UPDATE ON round21_shadow_run BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow run is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_run_no_delete
            BEFORE DELETE ON round21_shadow_run BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow run is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_prediction_no_update
            BEFORE UPDATE ON round21_shadow_prediction BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow prediction is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_prediction_no_delete
            BEFORE DELETE ON round21_shadow_prediction BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow prediction is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_terminal_no_update
            BEFORE UPDATE ON round21_shadow_terminal BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow terminal is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS round21_shadow_terminal_no_delete
            BEFORE DELETE ON round21_shadow_terminal BEGIN
                SELECT RAISE(ABORT, 'Round 21 shadow terminal is immutable');
            END;
            """
        )
        expected_columns = {
            "round21_shadow_metadata": ("key", "value"),
            "round21_shadow_run": (
                "run_id",
                "schema_version",
                "source_model_artifact_sha256",
                "sealed_result_sha256",
                "population_layer",
                "started_at_ms",
                "run_sha256",
            ),
            "round21_shadow_prediction": (
                "run_id",
                "sequence_number",
                "condition_id",
                "decision_time_ms",
                "prediction_sha256",
                "previous_record_sha256",
                "payload_sha256",
                "compressed_payload_sha256",
                "raw_byte_count",
                "compressed_byte_count",
                "payload",
                "recorded_at_ms",
                "record_sha256",
            ),
            "round21_shadow_terminal": (
                "run_id",
                "schema_version",
                "sequence_number",
                "status",
                "reason",
                "previous_record_sha256",
                "finished_at_ms",
                "terminal_sha256",
            ),
        }
        for table_name, columns in expected_columns.items():
            observed = tuple(
                str(row[1])
                for row in connection.execute(
                    f"PRAGMA table_info('{table_name}')"
                ).fetchall()
            )
            if observed != columns:
                raise RuntimeError(
                    f"Round 21 shadow table schema differs: {table_name}"
                )
        check = connection.execute("PRAGMA quick_check").fetchone()
        if check is None or str(check[0]).lower() != "ok":
            raise RuntimeError("Round 21 shadow store integrity check failed")

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> Round21ShadowRun:
        if str(row["schema_version"]) != POLYMARKET_ROUND21_SHADOW_RUN_SCHEMA_VERSION:
            raise ValueError("Round 21 shadow run schema differs")
        run = Round21ShadowRun.create(
            run_id=str(row["run_id"]),
            source_model_artifact_sha256=str(row["source_model_artifact_sha256"]),
            sealed_result_sha256=str(row["sealed_result_sha256"]),
            population_layer=str(row["population_layer"]),
            started_at_ms=int(row["started_at_ms"]),
        )
        if run.run_sha256 != str(row["run_sha256"]):
            raise ValueError("Round 21 shadow run hash differs")
        return run.validated()

    def start_run(
        self,
        *,
        run_id: str,
        source_model_artifact_sha256: str,
        sealed_result_sha256: str,
        population_layer: str,
        started_at_ms: int | None = None,
    ) -> Round21ShadowRun:
        started = int(self._wall_time_ms()) if started_at_ms is None else started_at_ms
        run = Round21ShadowRun.create(
            run_id=run_id,
            source_model_artifact_sha256=source_model_artifact_sha256,
            sealed_result_sha256=sealed_result_sha256,
            population_layer=population_layer,
            started_at_ms=started,
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM round21_shadow_run WHERE run_id = ?",
                    (run.run_id,),
                ).fetchone()
                if row is not None:
                    existing = self._run_from_row(row)
                    if existing != run:
                        raise ValueError("Round 21 shadow run ID is immutable")
                    self._connection.execute("COMMIT")
                    return existing
                self._connection.execute(
                    """
                    INSERT INTO round21_shadow_run VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        POLYMARKET_ROUND21_SHADOW_RUN_SCHEMA_VERSION,
                        run.source_model_artifact_sha256,
                        run.sealed_result_sha256,
                        run.population_layer,
                        run.started_at_ms,
                        run.run_sha256,
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise
        return run

    def _run(self, run_id: str) -> Round21ShadowRun:
        row = self._connection.execute(
            "SELECT * FROM round21_shadow_run WHERE run_id = ?",
            (_run_id(run_id),),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Round 21 shadow run: {run_id}")
        return self._run_from_row(row)

    @staticmethod
    def _prediction_record_payload(
        *,
        run_id: str,
        sequence_number: int,
        prediction: Round21ProspectivePrediction,
        previous_record_sha256: str,
        payload_sha256: str,
        compressed_payload_sha256: str,
        raw_byte_count: int,
        compressed_byte_count: int,
        recorded_at_ms: int,
    ) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_SHADOW_RECORD_SCHEMA_VERSION,
            "run_id": run_id,
            "sequence_number": sequence_number,
            "condition_id": prediction.condition_id,
            "decision_time_ms": prediction.decision_time_ms,
            "prediction_sha256": prediction.prediction_sha256,
            "previous_record_sha256": previous_record_sha256,
            "payload_sha256": payload_sha256,
            "compressed_payload_sha256": compressed_payload_sha256,
            "raw_byte_count": raw_byte_count,
            "compressed_byte_count": compressed_byte_count,
            "recorded_at_ms": recorded_at_ms,
            "target_accessed": False,
            "credentials_used": False,
            "account_connected": False,
            "binance_execution_connected": False,
            "grants_execution_authority": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    @classmethod
    def _stored_prediction_from_row(
        cls,
        row: sqlite3.Row,
    ) -> Round21StoredShadowPrediction:
        payload = bytes(row["payload"])
        prediction = _decode_prediction(
            payload,
            raw_byte_count=int(row["raw_byte_count"]),
            payload_sha256=str(row["payload_sha256"]),
            compressed_payload_sha256=str(row["compressed_payload_sha256"]),
        )
        record_payload = cls._prediction_record_payload(
            run_id=str(row["run_id"]),
            sequence_number=int(row["sequence_number"]),
            prediction=prediction,
            previous_record_sha256=str(row["previous_record_sha256"]),
            payload_sha256=str(row["payload_sha256"]),
            compressed_payload_sha256=str(row["compressed_payload_sha256"]),
            raw_byte_count=int(row["raw_byte_count"]),
            compressed_byte_count=int(row["compressed_byte_count"]),
            recorded_at_ms=int(row["recorded_at_ms"]),
        )
        record_sha = _canonical_sha256(record_payload)
        if (
            len(payload) != int(row["compressed_byte_count"])
            or prediction.condition_id != str(row["condition_id"])
            or prediction.decision_time_ms != int(row["decision_time_ms"])
            or prediction.prediction_sha256 != str(row["prediction_sha256"])
            or record_sha != str(row["record_sha256"])
        ):
            raise ValueError("Round 21 shadow prediction record differs")
        return Round21StoredShadowPrediction(
            run_id=str(row["run_id"]),
            sequence_number=int(row["sequence_number"]),
            condition_id=prediction.condition_id,
            decision_time_ms=prediction.decision_time_ms,
            prediction_sha256=prediction.prediction_sha256,
            previous_record_sha256=str(row["previous_record_sha256"]),
            record_sha256=record_sha,
            recorded_at_ms=int(row["recorded_at_ms"]),
            prediction=prediction,
        )

    def append_prediction(
        self,
        run_id: str,
        prediction: Round21ProspectivePrediction,
        *,
        recorded_at_ms: int | None = None,
    ) -> Round21StoredShadowPrediction:
        if not isinstance(prediction, Round21ProspectivePrediction):
            raise TypeError("Round 21 shadow prediction type differs")
        selected_prediction = prediction.validated()
        raw, compressed = _compress_prediction(selected_prediction)
        payload_sha = _sha256(raw)
        compressed_sha = _sha256(compressed)
        recorded = (
            int(self._wall_time_ms()) if recorded_at_ms is None else recorded_at_ms
        )
        _positive_int(recorded, name="record timestamp")
        if recorded < selected_prediction.observed_at_ms:
            raise ValueError("Round 21 shadow record precedes observation")
        selected_id = _run_id(run_id)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._run(selected_id)
                if (
                    selected_prediction.source_model_artifact_sha256
                    != run.source_model_artifact_sha256
                    or selected_prediction.sealed_result_sha256
                    != run.sealed_result_sha256
                    or selected_prediction.population_layer != run.population_layer
                ):
                    raise ValueError("Round 21 shadow prediction identity differs")
                if selected_prediction.observed_at_ms < run.started_at_ms:
                    raise ValueError("Round 21 shadow prediction precedes run")
                existing_row = self._connection.execute(
                    """
                    SELECT * FROM round21_shadow_prediction
                    WHERE run_id = ? AND condition_id = ? AND decision_time_ms = ?
                    """,
                    (
                        selected_id,
                        selected_prediction.condition_id,
                        selected_prediction.decision_time_ms,
                    ),
                ).fetchone()
                if existing_row is not None:
                    existing = self._stored_prediction_from_row(existing_row)
                    if (
                        existing.prediction_sha256
                        != selected_prediction.prediction_sha256
                        or str(existing_row["payload_sha256"]) != payload_sha
                    ):
                        raise ValueError("Round 21 shadow prediction key is immutable")
                    self._connection.execute("COMMIT")
                    return existing
                if (
                    self._connection.execute(
                        "SELECT 1 FROM round21_shadow_terminal WHERE run_id = ?",
                        (selected_id,),
                    ).fetchone()
                    is not None
                ):
                    raise ValueError("Round 21 shadow run is terminal")
                previous = self._connection.execute(
                    """
                    SELECT sequence_number, record_sha256, recorded_at_ms,
                           decision_time_ms
                    FROM round21_shadow_prediction
                    WHERE run_id = ? ORDER BY sequence_number DESC LIMIT 1
                    """,
                    (selected_id,),
                ).fetchone()
                sequence = 1 if previous is None else int(previous[0]) + 1
                previous_sha = run.run_sha256 if previous is None else str(previous[1])
                if previous is not None and (
                    recorded < int(previous[2])
                    or selected_prediction.decision_time_ms <= int(previous[3])
                ):
                    raise ValueError("Round 21 shadow prediction chronology differs")
                record_payload = self._prediction_record_payload(
                    run_id=selected_id,
                    sequence_number=sequence,
                    prediction=selected_prediction,
                    previous_record_sha256=previous_sha,
                    payload_sha256=payload_sha,
                    compressed_payload_sha256=compressed_sha,
                    raw_byte_count=len(raw),
                    compressed_byte_count=len(compressed),
                    recorded_at_ms=recorded,
                )
                record_sha = _canonical_sha256(record_payload)
                self._connection.execute(
                    """
                    INSERT INTO round21_shadow_prediction VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        selected_id,
                        sequence,
                        selected_prediction.condition_id,
                        selected_prediction.decision_time_ms,
                        selected_prediction.prediction_sha256,
                        previous_sha,
                        payload_sha,
                        compressed_sha,
                        len(raw),
                        len(compressed),
                        compressed,
                        recorded,
                        record_sha,
                    ),
                )
                row = self._connection.execute(
                    """
                    SELECT * FROM round21_shadow_prediction
                    WHERE run_id = ? AND sequence_number = ?
                    """,
                    (selected_id, sequence),
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "Round 21 shadow prediction insert was not durable"
                    )
                stored = self._stored_prediction_from_row(row)
                self._connection.execute("COMMIT")
                return stored
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _terminal_payload(
        *,
        run_id: str,
        sequence_number: int,
        status: str,
        reason: str,
        previous_record_sha256: str,
        finished_at_ms: int,
    ) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_SHADOW_TERMINAL_SCHEMA_VERSION,
            "run_id": run_id,
            "sequence_number": sequence_number,
            "status": status,
            "reason": reason,
            "previous_record_sha256": previous_record_sha256,
            "finished_at_ms": finished_at_ms,
            "grants_execution_authority": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    @classmethod
    def _terminal_from_row(cls, row: sqlite3.Row) -> Round21ShadowTerminal:
        if (
            str(row["schema_version"])
            != POLYMARKET_ROUND21_SHADOW_TERMINAL_SCHEMA_VERSION
        ):
            raise ValueError("Round 21 shadow terminal schema differs")
        payload = cls._terminal_payload(
            run_id=str(row["run_id"]),
            sequence_number=int(row["sequence_number"]),
            status=str(row["status"]),
            reason=str(row["reason"]),
            previous_record_sha256=str(row["previous_record_sha256"]),
            finished_at_ms=int(row["finished_at_ms"]),
        )
        terminal_sha = _canonical_sha256(payload)
        if terminal_sha != str(row["terminal_sha256"]):
            raise ValueError("Round 21 shadow terminal hash differs")
        return Round21ShadowTerminal(
            run_id=str(row["run_id"]),
            sequence_number=int(row["sequence_number"]),
            status=str(row["status"]),
            reason=str(row["reason"]),
            previous_record_sha256=str(row["previous_record_sha256"]),
            finished_at_ms=int(row["finished_at_ms"]),
            terminal_sha256=terminal_sha,
        )

    def terminate_run(
        self,
        run_id: str,
        *,
        status: str,
        reason: str = "",
        finished_at_ms: int | None = None,
    ) -> Round21ShadowTerminal:
        selected_id = _run_id(run_id)
        selected_status = str(status or "").strip().lower()
        selected_reason = str(reason or "").strip()
        finished = (
            int(self._wall_time_ms()) if finished_at_ms is None else finished_at_ms
        )
        _positive_int(finished, name="terminal timestamp")
        if (
            selected_status not in _TERMINAL_STATUSES
            or len(selected_reason) > _MAX_TERMINAL_REASON_CHARS
            or (selected_status == "complete" and selected_reason)
            or (selected_status != "complete" and not selected_reason)
        ):
            raise ValueError("Round 21 shadow terminal state is invalid")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                run = self._run(selected_id)
                if finished < run.started_at_ms:
                    raise ValueError("Round 21 shadow terminal precedes run")
                existing_row = self._connection.execute(
                    "SELECT * FROM round21_shadow_terminal WHERE run_id = ?",
                    (selected_id,),
                ).fetchone()
                if existing_row is not None:
                    existing = self._terminal_from_row(existing_row)
                    if (
                        existing.status != selected_status
                        or existing.reason != selected_reason
                        or existing.finished_at_ms != finished
                    ):
                        raise ValueError("Round 21 shadow terminal is immutable")
                    self._connection.execute("COMMIT")
                    return existing
                previous = self._connection.execute(
                    """
                    SELECT sequence_number, record_sha256, recorded_at_ms
                    FROM round21_shadow_prediction
                    WHERE run_id = ? ORDER BY sequence_number DESC LIMIT 1
                    """,
                    (selected_id,),
                ).fetchone()
                sequence = 1 if previous is None else int(previous[0]) + 1
                previous_sha = run.run_sha256 if previous is None else str(previous[1])
                if previous is not None and finished < int(previous[2]):
                    raise ValueError("Round 21 shadow terminal precedes last record")
                payload = self._terminal_payload(
                    run_id=selected_id,
                    sequence_number=sequence,
                    status=selected_status,
                    reason=selected_reason,
                    previous_record_sha256=previous_sha,
                    finished_at_ms=finished,
                )
                terminal_sha = _canonical_sha256(payload)
                self._connection.execute(
                    "INSERT INTO round21_shadow_terminal VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        selected_id,
                        POLYMARKET_ROUND21_SHADOW_TERMINAL_SCHEMA_VERSION,
                        sequence,
                        selected_status,
                        selected_reason,
                        previous_sha,
                        finished,
                        terminal_sha,
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM round21_shadow_terminal WHERE run_id = ?",
                    (selected_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "Round 21 shadow terminal insert was not durable"
                    )
                terminal = self._terminal_from_row(row)
                self._connection.execute("COMMIT")
                return terminal
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise

    def predictions(self, run_id: str) -> tuple[Round21StoredShadowPrediction, ...]:
        """Load and verify every prediction in sequence order."""

        selected_id = _run_id(run_id)
        with self._lock:
            self._run(selected_id)
            rows = self._connection.execute(
                """
                SELECT * FROM round21_shadow_prediction
                WHERE run_id = ? ORDER BY sequence_number
                """,
                (selected_id,),
            ).fetchall()
        return tuple(self._stored_prediction_from_row(row) for row in rows)

    def run_ids(self) -> tuple[str, ...]:
        """List verified run identities without decoding prediction payloads."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM round21_shadow_run ORDER BY started_at_ms, run_id"
            ).fetchall()
        return tuple(self._run_from_row(row).run_id for row in rows)

    def audit_run(self, run_id: str) -> Round21ShadowAudit:
        """Replay the complete semantic and byte chain without granting authority."""

        selected_id = _run_id(run_id)
        with self._lock:
            self._connection.execute("BEGIN")
            try:
                check = self._connection.execute("PRAGMA quick_check").fetchone()
                if check is None or str(check[0]).lower() != "ok":
                    raise ValueError("Round 21 shadow database integrity differs")
                run = self._run(selected_id)
                terminal_row = self._connection.execute(
                    "SELECT * FROM round21_shadow_terminal WHERE run_id = ?",
                    (selected_id,),
                ).fetchone()
                if terminal_row is None:
                    raise ValueError("Round 21 shadow run is not terminal")
                terminal = self._terminal_from_row(terminal_row)
                rows = self._connection.execute(
                    """
                    SELECT * FROM round21_shadow_prediction
                    WHERE run_id = ? ORDER BY sequence_number
                    """,
                    (selected_id,),
                ).fetchall()
                previous_sha = run.run_sha256
                previous_decision_ms = 0
                previous_recorded_ms = run.started_at_ms
                observed_count = 0
                abstention_count = 0
                for expected_sequence, row in enumerate(rows, start=1):
                    stored = self._stored_prediction_from_row(row)
                    prediction = stored.prediction
                    if (
                        stored.sequence_number != expected_sequence
                        or stored.previous_record_sha256 != previous_sha
                        or stored.recorded_at_ms < previous_recorded_ms
                        or prediction.decision_time_ms <= previous_decision_ms
                        or prediction.source_model_artifact_sha256
                        != run.source_model_artifact_sha256
                        or prediction.sealed_result_sha256 != run.sealed_result_sha256
                        or prediction.population_layer != run.population_layer
                    ):
                        raise ValueError("Round 21 shadow prediction chain differs")
                    previous_sha = stored.record_sha256
                    previous_decision_ms = prediction.decision_time_ms
                    previous_recorded_ms = stored.recorded_at_ms
                    observed_count += prediction.status == "observed"
                    abstention_count += prediction.status == "abstain"
                if (
                    terminal.sequence_number != len(rows) + 1
                    or terminal.previous_record_sha256 != previous_sha
                    or terminal.finished_at_ms < previous_recorded_ms
                ):
                    raise ValueError("Round 21 shadow terminal chain differs")
                audit = Round21ShadowAudit(
                    run=run,
                    prediction_count=len(rows),
                    observed_count=observed_count,
                    abstention_count=abstention_count,
                    last_record_sha256=previous_sha,
                    terminal=terminal,
                )
                self._connection.execute("COMMIT")
                return audit
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise


credentials_used = False
account_connected = False
binance_execution_connected = False
grants_execution_authority = False
profitability_claim = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_SHADOW_RECORD_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SHADOW_RUN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SHADOW_STORE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SHADOW_TERMINAL_SCHEMA_VERSION",
    "Round21ProspectiveShadowStore",
    "Round21ShadowAudit",
    "Round21ShadowRun",
    "Round21ShadowTerminal",
    "Round21StoredShadowPrediction",
]
