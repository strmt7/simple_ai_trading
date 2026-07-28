"""Durable one-use governance for the Round 74 sealed event evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

from .impact_absorption_ai_uplift import Round74AIPretestQualificationPanel
from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionPolicySelection,
)
from .impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS,
)
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .types import config_paths


ROUND74_SEALED_LEDGER_SCHEMA_VERSION = "round-074-sealed-ledger-v3"
ROUND74_SEALED_LEDGER_PRETEST_SCHEMA_VERSION = "round-074-sealed-ledger-v2"
ROUND74_SEALED_LEDGER_LEGACY_SCHEMA_VERSION = "round-074-sealed-ledger-v1"
ROUND74_SEALED_CLAIM_SCHEMA_VERSION = "round-074-sealed-claim-v3"
ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION = "round-074-sealed-dataset-identity-v2"
ROUND74_SEALED_TEST_POPULATION_SCHEMA_VERSION = "round-074-sealed-test-population-v1"
ROUND74_SEALED_OPTIMIZATION_POPULATIONS = ("capture_run", "eligible_target")
ROUND74_SEALED_RESULT_OUTCOMES = (
    "candidate_passed_predeclared_gates",
    "candidate_failed_predeclared_gates",
    "evaluation_error",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_CLAIM_STATUSES = ("reserved", "complete", "failed")


class Round74SealedLedgerError(RuntimeError):
    """Base error for durable Round 74 sealed-evaluation governance."""


class Round74SealedReuseError(Round74SealedLedgerError):
    """Raised before evaluation when the sealed access was already claimed."""


def default_round74_sealed_ledger_path() -> Path:
    override = str(
        os.environ.get("SIMPLE_AI_TRADING_ROUND74_SEALED_LEDGER") or ""
    ).strip()
    if override:
        return Path(override).expanduser()
    return config_paths()["base"] / "round74_sealed_evaluations.sqlite3"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 sealed {label} digest differs")
    return selected


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round 74 sealed {label} differs")
    return int(value)


def _valid_test_run_count(count: object, optimization_population: str) -> bool:
    expected = ROUND74_EVENT_COHORT_DEFAULT_ROLE_COUNTS["test"]
    if isinstance(count, bool) or not isinstance(count, int):
        return False
    if optimization_population == "capture_run":
        return count == expected
    return optimization_population == "eligible_target" and count >= expected


def _capture_run_test_population_sha256(run_ids: Sequence[str]) -> str:
    return _canonical_sha256(
        {
            "schema_version": ROUND74_SEALED_TEST_POPULATION_SCHEMA_VERSION,
            "optimization_population": "capture_run",
            "assignment_basis": "fixed_legacy_test_role",
            "test_run_ids": list(run_ids),
            "all_admitted_test_runs_required": True,
        }
    )


@dataclass(frozen=True)
class Round74SealedEvaluationClaim:
    """One immutable reservation that consumes a test-access identity."""

    reservation_id: str
    ledger_id: str
    test_access_sha256: str
    dataset_sha256: str
    partition_sha256: str
    scaler_sha256: str
    pretest_policy_sha256: str
    probability_calibration_sha256: str
    action_selection_sha256: str
    ai_pretest_qualification_sha256: str
    ai_pretest_qualification_required: bool
    ai_manifest_sha256: tuple[str, ...]
    profile: str
    optimization_population: str
    test_population_sha256: str
    test_run_ids: tuple[str, ...]
    batch_sha256: tuple[str, ...]
    rows: int
    first_wall_ns: int
    last_wall_ns: int
    status: str
    result_outcome: str
    result_sha256: str
    error: str
    reserved_at_ns: int
    completed_at_ns: int | None
    schema_version: str = ROUND74_SEALED_CLAIM_SCHEMA_VERSION

    def validate(self) -> None:
        observed_runs = len(self.test_run_ids)
        if (
            self.schema_version != ROUND74_SEALED_CLAIM_SCHEMA_VERSION
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.reservation_id,
                    self.ledger_id,
                    self.test_access_sha256,
                    self.dataset_sha256,
                    self.partition_sha256,
                    self.scaler_sha256,
                    self.pretest_policy_sha256,
                    self.probability_calibration_sha256,
                    self.action_selection_sha256,
                    self.test_population_sha256,
                    *self.ai_manifest_sha256,
                    *self.batch_sha256,
                )
            )
            or not self.ai_manifest_sha256
            or len(self.ai_manifest_sha256) > 2
            or len(set(self.ai_manifest_sha256)) != len(self.ai_manifest_sha256)
            or not isinstance(self.ai_pretest_qualification_required, bool)
            or (
                self.ai_pretest_qualification_required
                and _SHA256.fullmatch(self.ai_pretest_qualification_sha256) is None
            )
            or (
                not self.ai_pretest_qualification_required
                and self.ai_pretest_qualification_sha256
            )
            or self.optimization_population
            not in ROUND74_SEALED_OPTIMIZATION_POPULATIONS
            or not _valid_test_run_count(
                observed_runs,
                self.optimization_population,
            )
            or len(set(self.test_run_ids)) != observed_runs
            or any(_RUN_ID.fullmatch(value) is None for value in self.test_run_ids)
            or not self.batch_sha256
            or len(set(self.batch_sha256)) != len(self.batch_sha256)
            or self.profile not in ROUND74_ACTION_PROFILES
            or isinstance(self.rows, bool)
            or self.rows < observed_runs
            or isinstance(self.first_wall_ns, bool)
            or isinstance(self.last_wall_ns, bool)
            or not 0 < self.first_wall_ns <= self.last_wall_ns
            or self.status not in _CLAIM_STATUSES
            or isinstance(self.reserved_at_ns, bool)
            or self.reserved_at_ns <= 0
            or self.error != self.error.strip()[:2_000]
        ):
            raise ValueError("Round 74 sealed claim differs")
        if self.status == "reserved":
            valid_completion = (
                not self.result_outcome
                and not self.result_sha256
                and not self.error
                and self.completed_at_ns is None
            )
        else:
            valid_completion = (
                self.result_outcome in ROUND74_SEALED_RESULT_OUTCOMES
                and _SHA256.fullmatch(self.result_sha256) is not None
                and isinstance(self.completed_at_ns, int)
                and not isinstance(self.completed_at_ns, bool)
                and self.completed_at_ns >= self.reserved_at_ns
                and (
                    self.status == "failed"
                    if self.result_outcome == "evaluation_error"
                    else self.status == "complete"
                )
                and (
                    bool(self.error)
                    if self.result_outcome == "evaluation_error"
                    else not self.error
                )
            )
        if not valid_completion:
            raise ValueError("Round 74 sealed claim completion differs")

    @property
    def claim_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "reservation_id": self.reservation_id,
            "ledger_id": self.ledger_id,
            "test_access_sha256": self.test_access_sha256,
            "dataset_sha256": self.dataset_sha256,
            "partition_sha256": self.partition_sha256,
            "scaler_sha256": self.scaler_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "action_selection_sha256": self.action_selection_sha256,
            "ai_pretest_qualification_sha256": (self.ai_pretest_qualification_sha256),
            "ai_pretest_qualification_required": (
                self.ai_pretest_qualification_required
            ),
            "ai_manifest_sha256": list(self.ai_manifest_sha256),
            "profile": self.profile,
            "optimization_population": self.optimization_population,
            "test_population_sha256": self.test_population_sha256,
            "test_run_ids": list(self.test_run_ids),
            "batch_sha256": list(self.batch_sha256),
            "rows": self.rows,
            "first_wall_ns": self.first_wall_ns,
            "last_wall_ns": self.last_wall_ns,
            "status": self.status,
            "result_outcome": self.result_outcome,
            "result_sha256": self.result_sha256,
            "error": self.error,
            "reserved_at_ns": self.reserved_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "test_access_consumed_at_reservation": True,
            "reservation_reset_api_available": False,
        }
        if include_sha256:
            value["claim_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> Round74SealedEvaluationClaim:
        payload = dict(value)
        claimed = str(payload.pop("claim_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 sealed claim digest differs")
        if (
            payload.pop("test_access_consumed_at_reservation", None) is not True
            or payload.pop("reservation_reset_api_available", None) is not False
        ):
            raise ValueError("Round 74 sealed claim policy differs")
        try:
            selected = cls(
                reservation_id=str(payload["reservation_id"]),
                ledger_id=str(payload["ledger_id"]),
                test_access_sha256=str(payload["test_access_sha256"]),
                dataset_sha256=str(payload["dataset_sha256"]),
                partition_sha256=str(payload["partition_sha256"]),
                scaler_sha256=str(payload["scaler_sha256"]),
                pretest_policy_sha256=str(payload["pretest_policy_sha256"]),
                probability_calibration_sha256=str(
                    payload["probability_calibration_sha256"]
                ),
                action_selection_sha256=str(payload["action_selection_sha256"]),
                ai_pretest_qualification_sha256=str(
                    payload["ai_pretest_qualification_sha256"]
                ),
                ai_pretest_qualification_required=payload[
                    "ai_pretest_qualification_required"
                ],
                ai_manifest_sha256=tuple(
                    str(item) for item in payload["ai_manifest_sha256"]
                ),
                profile=str(payload["profile"]),
                optimization_population=str(payload["optimization_population"]),
                test_population_sha256=str(payload["test_population_sha256"]),
                test_run_ids=tuple(str(item) for item in payload["test_run_ids"]),
                batch_sha256=tuple(str(item) for item in payload["batch_sha256"]),
                rows=_strict_int(payload["rows"], "rows"),
                first_wall_ns=_strict_int(
                    payload["first_wall_ns"],
                    "first wall",
                ),
                last_wall_ns=_strict_int(
                    payload["last_wall_ns"],
                    "last wall",
                ),
                status=str(payload["status"]),
                result_outcome=str(payload["result_outcome"]),
                result_sha256=str(payload["result_sha256"]),
                error=str(payload["error"]),
                reserved_at_ns=_strict_int(
                    payload["reserved_at_ns"],
                    "reserved time",
                ),
                completed_at_ns=(
                    _strict_int(payload["completed_at_ns"], "completed time")
                    if payload["completed_at_ns"] is not None
                    else None
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 sealed claim payload differs") from exc
        selected.validate()
        if selected.as_dict() != dict(value):
            raise ValueError("Round 74 sealed claim payload differs")
        return selected


@dataclass(frozen=True)
class Round74SealedDatasetIdentity:
    """Metadata-only identity that can be reserved before loading test targets."""

    test_access_sha256: str
    partition_sha256: str
    scaler_sha256: str
    optimization_population: str
    test_population_sha256: str
    test_run_ids: tuple[str, ...]
    batch_sha256: tuple[str, ...]
    rows: int
    first_wall_ns: int
    last_wall_ns: int
    schema_version: str = ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION

    def validate(self) -> None:
        observed_runs = len(self.test_run_ids)
        if (
            self.schema_version != ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION
            or _SHA256.fullmatch(self.test_access_sha256) is None
            or _SHA256.fullmatch(self.partition_sha256) is None
            or _SHA256.fullmatch(self.scaler_sha256) is None
            or _SHA256.fullmatch(self.test_population_sha256) is None
            or self.optimization_population
            not in ROUND74_SEALED_OPTIMIZATION_POPULATIONS
            or not _valid_test_run_count(
                observed_runs,
                self.optimization_population,
            )
            or len(set(self.test_run_ids)) != observed_runs
            or any(_RUN_ID.fullmatch(value) is None for value in self.test_run_ids)
            or not self.batch_sha256
            or len(set(self.batch_sha256)) != len(self.batch_sha256)
            or any(_SHA256.fullmatch(value) is None for value in self.batch_sha256)
            or isinstance(self.rows, bool)
            or not isinstance(self.rows, int)
            or self.rows < 1
            or isinstance(self.first_wall_ns, bool)
            or not isinstance(self.first_wall_ns, int)
            or self.first_wall_ns <= 0
            or isinstance(self.last_wall_ns, bool)
            or not isinstance(self.last_wall_ns, int)
            or self.last_wall_ns < self.first_wall_ns
        ):
            raise ValueError("Round 74 sealed dataset identity differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "test_access_sha256": self.test_access_sha256,
            "partition_sha256": self.partition_sha256,
            "scaler_sha256": self.scaler_sha256,
            "optimization_population": self.optimization_population,
            "test_population_sha256": self.test_population_sha256,
            "test_run_ids": list(self.test_run_ids),
            "batch_sha256": list(self.batch_sha256),
            "rows": self.rows,
            "first_wall_ns": self.first_wall_ns,
            "last_wall_ns": self.last_wall_ns,
        }

    @property
    def dataset_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


def build_round74_sealed_dataset_identity(
    batches: Sequence[Round74EventTrainingBatch],
    *,
    optimization_population: str = "capture_run",
    expected_test_run_ids: Sequence[str] | None = None,
    test_population_sha256: str | None = None,
) -> Round74SealedDatasetIdentity:
    selected = tuple(batches)
    if not selected:
        raise ValueError("Round 74 sealed test batches are missing")
    prior_key: tuple[object, ...] | None = None
    partitions: set[str] = set()
    scalers: set[str] = set()
    access: set[str] = set()
    run_ids: list[str] = []
    samples: set[str] = set()
    rows = 0
    for batch in selected:
        batch.validate()
        if batch.role != "test":
            raise ValueError("Round 74 sealed ledger rejects development data")
        first_key = (
            int(batch.decision_wall_ns[0]),
            batch.run_id[0],
            int(batch.decision_monotonic_ns[0]),
        )
        last_key = (
            int(batch.decision_wall_ns[-1]),
            batch.run_id[-1],
            int(batch.decision_monotonic_ns[-1]),
        )
        if prior_key is not None and first_key <= prior_key:
            raise ValueError("Round 74 sealed batch order regressed")
        prior_key = last_key
        partitions.add(batch.partition_sha256)
        scalers.add(batch.scaler_sha256)
        access.update(batch.test_access_sha256)
        for run_id in batch.run_id:
            if not run_ids or run_ids[-1] != run_id:
                run_ids.append(run_id)
        for sample in batch.sample_sha256:
            if sample in samples:
                raise ValueError("Round 74 sealed sample is duplicated")
            samples.add(sample)
        rows += batch.rows
    selected_population = str(optimization_population)
    expected = (
        None
        if expected_test_run_ids is None
        else tuple(str(value) for value in expected_test_run_ids)
    )
    if (
        len(partitions) != 1
        or len(scalers) != 1
        or len(access) != 1
        or not _valid_test_run_count(len(run_ids), selected_population)
        or len(set(run_ids)) != len(run_ids)
    ):
        raise ValueError("Round 74 sealed dataset identity differs")
    if selected_population == "capture_run":
        population_sha256 = _capture_run_test_population_sha256(run_ids)
        if (
            expected is not None
            and expected != tuple(run_ids)
            or test_population_sha256 is not None
            and str(test_population_sha256) != population_sha256
        ):
            raise ValueError("Round 74 sealed test population differs")
    else:
        population_sha256 = str(test_population_sha256 or "")
        if expected != tuple(run_ids) or _SHA256.fullmatch(population_sha256) is None:
            raise ValueError("Round 74 sealed test population differs")
    batch_hashes = tuple(batch.batch_sha256 for batch in selected)
    identity = Round74SealedDatasetIdentity(
        test_access_sha256=next(iter(access)),
        partition_sha256=next(iter(partitions)),
        scaler_sha256=next(iter(scalers)),
        optimization_population=selected_population,
        test_population_sha256=population_sha256,
        test_run_ids=tuple(run_ids),
        batch_sha256=batch_hashes,
        rows=rows,
        first_wall_ns=int(selected[0].decision_wall_ns[0]),
        last_wall_ns=int(selected[-1].decision_wall_ns[-1]),
    )
    identity.validate()
    return identity


class Round74SealedEvaluationLedger:
    """SQLite ledger with reserve-before-read and no reset operation."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path) if path is not None else default_round74_sealed_ledger_path()
        )

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if mode != "delete":
                mode = str(
                    connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
                ).lower()
            if mode != "delete":
                raise Round74SealedLedgerError(
                    f"Round 74 sealed ledger journal mode differs: {mode}"
                )
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            self._initialize(connection)
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise Round74SealedLedgerError(
                    "Round 74 sealed ledger integrity check failed"
                )
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _initialize(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS round74_governance_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round74_sealed_claims (
                reservation_id TEXT PRIMARY KEY,
                ledger_id TEXT NOT NULL,
                test_access_sha256 TEXT NOT NULL UNIQUE,
                dataset_sha256 TEXT NOT NULL UNIQUE,
                partition_sha256 TEXT NOT NULL,
                scaler_sha256 TEXT NOT NULL,
                pretest_policy_sha256 TEXT NOT NULL,
                probability_calibration_sha256 TEXT NOT NULL,
                action_selection_sha256 TEXT NOT NULL,
                ai_pretest_qualification_sha256 TEXT NOT NULL,
                ai_pretest_qualification_required INTEGER NOT NULL,
                ai_manifest_sha256_json TEXT NOT NULL,
                profile TEXT NOT NULL,
                optimization_population TEXT NOT NULL,
                test_population_sha256 TEXT NOT NULL,
                test_run_ids_json TEXT NOT NULL,
                batch_sha256_json TEXT NOT NULL,
                rows INTEGER NOT NULL,
                first_wall_ns INTEGER NOT NULL,
                last_wall_ns INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_outcome TEXT NOT NULL DEFAULT '',
                result_sha256 TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                reserved_at_ns INTEGER NOT NULL,
                completed_at_ns INTEGER,
                CHECK (profile IN ('conservative', 'regular', 'aggressive')),
                CHECK (optimization_population IN ('capture_run', 'eligible_target')),
                CHECK (rows > 0),
                CHECK (first_wall_ns > 0 AND last_wall_ns >= first_wall_ns),
                CHECK (status IN ('reserved', 'complete', 'failed'))
            );
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(round74_sealed_claims)"
            ).fetchall()
        }
        if "optimization_population" not in columns:
            connection.execute(
                """
                ALTER TABLE round74_sealed_claims
                ADD COLUMN optimization_population TEXT NOT NULL
                DEFAULT 'capture_run'
                """
            )
        if "test_population_sha256" not in columns:
            connection.execute(
                """
                ALTER TABLE round74_sealed_claims
                ADD COLUMN test_population_sha256 TEXT NOT NULL
                DEFAULT ''
                """
            )
        if "ai_pretest_qualification_sha256" not in columns:
            connection.execute(
                """
                ALTER TABLE round74_sealed_claims
                ADD COLUMN ai_pretest_qualification_sha256 TEXT NOT NULL
                DEFAULT ''
                """
            )
        if "ai_pretest_qualification_required" not in columns:
            connection.execute(
                """
                ALTER TABLE round74_sealed_claims
                ADD COLUMN ai_pretest_qualification_required INTEGER NOT NULL
                DEFAULT 0
                """
            )
        population_rows = connection.execute(
            """
            SELECT reservation_id, optimization_population,
                   test_run_ids_json, test_population_sha256
            FROM round74_sealed_claims
            """
        ).fetchall()
        for row in population_rows:
            if str(row["test_population_sha256"]):
                continue
            try:
                run_ids = json.loads(str(row["test_run_ids_json"]))
            except json.JSONDecodeError as exc:
                raise Round74SealedLedgerError(
                    "Round 74 sealed legacy population JSON differs"
                ) from exc
            if (
                str(row["optimization_population"]) != "capture_run"
                or not isinstance(run_ids, list)
                or any(not isinstance(value, str) for value in run_ids)
                or not _valid_test_run_count(len(run_ids), "capture_run")
            ):
                raise Round74SealedLedgerError(
                    "Round 74 sealed legacy population differs"
                )
            connection.execute(
                """
                UPDATE round74_sealed_claims
                SET test_population_sha256 = ?
                WHERE reservation_id = ?
                """,
                [
                    _capture_run_test_population_sha256(run_ids),
                    str(row["reservation_id"]),
                ],
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO round74_governance_metadata (key, value)
            VALUES ('schema_version', ?)
            """,
            [ROUND74_SEALED_LEDGER_SCHEMA_VERSION],
        )
        schema = connection.execute(
            """
            SELECT value FROM round74_governance_metadata
            WHERE key = 'schema_version'
            """
        ).fetchone()
        if schema is not None and str(schema[0]) in {
            ROUND74_SEALED_LEDGER_LEGACY_SCHEMA_VERSION,
            ROUND74_SEALED_LEDGER_PRETEST_SCHEMA_VERSION,
        }:
            prior_schema = str(schema[0])
            connection.execute(
                """
                UPDATE round74_governance_metadata
                SET value = ?
                WHERE key = 'schema_version' AND value = ?
                """,
                [
                    ROUND74_SEALED_LEDGER_SCHEMA_VERSION,
                    prior_schema,
                ],
            )
            schema = (ROUND74_SEALED_LEDGER_SCHEMA_VERSION,)
        if schema is None or str(schema[0]) != ROUND74_SEALED_LEDGER_SCHEMA_VERSION:
            raise Round74SealedLedgerError("Round 74 sealed ledger schema differs")
        ledger = connection.execute(
            """
            SELECT value FROM round74_governance_metadata
            WHERE key = 'ledger_id'
            """
        ).fetchone()
        if ledger is None:
            connection.execute(
                """
                INSERT OR IGNORE INTO round74_governance_metadata (key, value)
                VALUES ('ledger_id', ?)
                """,
                [hashlib.sha256(os.urandom(32)).hexdigest()],
            )
            ledger = connection.execute(
                """
                SELECT value FROM round74_governance_metadata
                WHERE key = 'ledger_id'
                """
            ).fetchone()
        if ledger is None:
            raise Round74SealedLedgerError("Round 74 sealed ledger identity is missing")
        _require_sha256(ledger[0], "ledger")

    @staticmethod
    def _claim(row: sqlite3.Row) -> Round74SealedEvaluationClaim:
        try:
            manifests = json.loads(str(row["ai_manifest_sha256_json"]))
            run_ids = json.loads(str(row["test_run_ids_json"]))
            batches = json.loads(str(row["batch_sha256_json"]))
        except json.JSONDecodeError as exc:
            raise Round74SealedLedgerError(
                "Round 74 sealed ledger JSON differs"
            ) from exc
        if not all(isinstance(value, list) for value in (manifests, run_ids, batches)):
            raise Round74SealedLedgerError(
                "Round 74 sealed ledger list payload differs"
            )
        qualification_required = row["ai_pretest_qualification_required"]
        if qualification_required not in (0, 1):
            raise Round74SealedLedgerError(
                "Round 74 sealed AI qualification flag differs"
            )
        claim = Round74SealedEvaluationClaim(
            reservation_id=str(row["reservation_id"]),
            ledger_id=str(row["ledger_id"]),
            test_access_sha256=str(row["test_access_sha256"]),
            dataset_sha256=str(row["dataset_sha256"]),
            partition_sha256=str(row["partition_sha256"]),
            scaler_sha256=str(row["scaler_sha256"]),
            pretest_policy_sha256=str(row["pretest_policy_sha256"]),
            probability_calibration_sha256=str(row["probability_calibration_sha256"]),
            action_selection_sha256=str(row["action_selection_sha256"]),
            ai_pretest_qualification_sha256=str(row["ai_pretest_qualification_sha256"]),
            ai_pretest_qualification_required=bool(qualification_required),
            ai_manifest_sha256=tuple(str(value) for value in manifests),
            profile=str(row["profile"]),
            optimization_population=str(row["optimization_population"]),
            test_population_sha256=str(row["test_population_sha256"]),
            test_run_ids=tuple(str(value) for value in run_ids),
            batch_sha256=tuple(str(value) for value in batches),
            rows=int(row["rows"]),
            first_wall_ns=int(row["first_wall_ns"]),
            last_wall_ns=int(row["last_wall_ns"]),
            status=str(row["status"]),
            result_outcome=str(row["result_outcome"]),
            result_sha256=str(row["result_sha256"]),
            error=str(row["error"]),
            reserved_at_ns=int(row["reserved_at_ns"]),
            completed_at_ns=(
                int(row["completed_at_ns"])
                if row["completed_at_ns"] is not None
                else None
            ),
        )
        claim.validate()
        return claim

    def reserve(
        self,
        *,
        test_batches: Sequence[Round74EventTrainingBatch],
        action_selection: Round74ActionPolicySelection,
        ai_pretest_qualification: Round74AIPretestQualificationPanel,
    ) -> Round74SealedEvaluationClaim:
        """Reserve an already loaded panel for low-level callers and tests."""

        return self.reserve_identity(
            test_identity=build_round74_sealed_dataset_identity(
                test_batches,
                optimization_population=action_selection.optimization_population,
            ),
            action_selection=action_selection,
            ai_pretest_qualification=ai_pretest_qualification,
        )

    def reserve_identity(
        self,
        *,
        test_identity: Round74SealedDatasetIdentity,
        action_selection: Round74ActionPolicySelection,
        ai_pretest_qualification: Round74AIPretestQualificationPanel,
    ) -> Round74SealedEvaluationClaim:
        """Atomically consume metadata-only test access before target loading."""

        action_selection.validate()
        if not action_selection.accepted:
            raise ValueError("Round 74 sealed action policy is not accepted")
        test_identity.validate()
        identity = test_identity
        if identity.optimization_population != action_selection.optimization_population:
            raise ValueError("Round 74 sealed optimization population differs")
        ai_pretest_qualification.validate()
        manifests = ai_pretest_qualification.model_manifest_sha256
        if not ai_pretest_qualification.qualification_passed:
            raise ValueError("Round 74 sealed AI pretest qualification did not pass")
        if (
            ai_pretest_qualification.action_selection_sha256
            != action_selection.selection_sha256
            or ai_pretest_qualification.pretest_policy_sha256
            != action_selection.pretest_policy_sha256
            or ai_pretest_qualification.probability_calibration_sha256
            != action_selection.probability_calibration_sha256
            or ai_pretest_qualification.profile != action_selection.profile
        ):
            raise ValueError(
                "Round 74 sealed AI pretest qualification identity differs"
            )
        if len(manifests) != 2 or len(set(manifests)) != 2:
            raise ValueError("Round 74 sealed AI manifest panel differs")
        qualification_sha256 = ai_pretest_qualification.qualification_sha256
        now_ns = time.time_ns()
        contract = {
            "test_access_sha256": identity.test_access_sha256,
            "dataset_sha256": identity.dataset_sha256,
            "partition_sha256": identity.partition_sha256,
            "scaler_sha256": identity.scaler_sha256,
            "pretest_policy_sha256": action_selection.pretest_policy_sha256,
            "probability_calibration_sha256": (
                action_selection.probability_calibration_sha256
            ),
            "action_selection_sha256": action_selection.selection_sha256,
            "ai_pretest_qualification_sha256": qualification_sha256,
            "ai_pretest_qualification_required": True,
            "ai_manifest_sha256": manifests,
            "profile": action_selection.profile,
            "optimization_population": identity.optimization_population,
            "test_population_sha256": identity.test_population_sha256,
            "test_run_ids": identity.test_run_ids,
            "batch_sha256": identity.batch_sha256,
            "rows": identity.rows,
            "first_wall_ns": identity.first_wall_ns,
            "last_wall_ns": identity.last_wall_ns,
        }
        reservation_id = hashlib.sha256(
            _canonical_json(contract).encode("ascii")
            + os.urandom(32)
            + str(now_ns).encode("ascii")
        ).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            ledger_row = connection.execute(
                """
                SELECT value FROM round74_governance_metadata
                WHERE key = 'ledger_id'
                """
            ).fetchone()
            if ledger_row is None:
                raise Round74SealedLedgerError(
                    "Round 74 sealed ledger identity disappeared"
                )
            ledger_id = _require_sha256(ledger_row[0], "ledger")
            prior = connection.execute(
                """
                SELECT reservation_id, status FROM round74_sealed_claims
                WHERE test_access_sha256 = ? OR dataset_sha256 = ?
                LIMIT 1
                """,
                [identity.test_access_sha256, identity.dataset_sha256],
            ).fetchone()
            if prior is not None:
                raise Round74SealedReuseError(
                    "Round 74 sealed test was already reserved: "
                    f"reservation={prior['reservation_id']} "
                    f"status={prior['status']}"
                )
            connection.execute(
                """
                INSERT INTO round74_sealed_claims (
                    reservation_id, ledger_id, test_access_sha256,
                    dataset_sha256, partition_sha256, scaler_sha256,
                    pretest_policy_sha256, probability_calibration_sha256,
                    action_selection_sha256,
                    ai_pretest_qualification_sha256,
                    ai_pretest_qualification_required,
                    ai_manifest_sha256_json,
                    profile, optimization_population, test_population_sha256,
                    test_run_ids_json, batch_sha256_json, rows,
                    first_wall_ns, last_wall_ns, status, reserved_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, 'reserved', ?)
                """,
                [
                    reservation_id,
                    ledger_id,
                    identity.test_access_sha256,
                    identity.dataset_sha256,
                    identity.partition_sha256,
                    identity.scaler_sha256,
                    action_selection.pretest_policy_sha256,
                    action_selection.probability_calibration_sha256,
                    action_selection.selection_sha256,
                    qualification_sha256,
                    1,
                    _canonical_json(list(manifests)),
                    action_selection.profile,
                    identity.optimization_population,
                    identity.test_population_sha256,
                    _canonical_json(list(identity.test_run_ids)),
                    _canonical_json(list(identity.batch_sha256)),
                    identity.rows,
                    identity.first_wall_ns,
                    identity.last_wall_ns,
                    now_ns,
                ],
            )
            row = connection.execute(
                """
                SELECT * FROM round74_sealed_claims
                WHERE reservation_id = ?
                """,
                [reservation_id],
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise Round74SealedLedgerError(
                    "Round 74 sealed reservation is unreadable"
                )
            return self._claim(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finalize(
        self,
        reservation_id: str,
        *,
        result_outcome: str,
        result_sha256: str,
        error: str = "",
    ) -> Round74SealedEvaluationClaim:
        reservation = _require_sha256(reservation_id, "reservation")
        outcome = str(result_outcome)
        if outcome not in ROUND74_SEALED_RESULT_OUTCOMES:
            raise ValueError("Round 74 sealed result outcome differs")
        result = _require_sha256(result_sha256, "result")
        detail = " ".join(str(error).split())[:2_000]
        if outcome == "evaluation_error" and not detail:
            raise ValueError("Round 74 sealed evaluation error is missing")
        if outcome != "evaluation_error" and detail:
            raise ValueError("Round 74 sealed successful result has an error")
        status = "failed" if outcome == "evaluation_error" else "complete"
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """
                SELECT status, reserved_at_ns FROM round74_sealed_claims
                WHERE reservation_id = ?
                """,
                [reservation],
            ).fetchone()
            if current is None:
                raise Round74SealedLedgerError(
                    "Round 74 sealed reservation does not exist"
                )
            if str(current["status"]) != "reserved":
                raise Round74SealedLedgerError(
                    "Round 74 sealed reservation was already finalized"
                )
            completed = max(time.time_ns(), int(current["reserved_at_ns"]))
            connection.execute(
                """
                UPDATE round74_sealed_claims
                SET status = ?, result_outcome = ?, result_sha256 = ?,
                    error = ?, completed_at_ns = ?
                WHERE reservation_id = ? AND status = 'reserved'
                """,
                [status, outcome, result, detail, completed, reservation],
            )
            row = connection.execute(
                """
                SELECT * FROM round74_sealed_claims
                WHERE reservation_id = ?
                """,
                [reservation],
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise Round74SealedLedgerError(
                    "Round 74 sealed reservation disappeared"
                )
            return self._claim(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def claim(
        self,
        reservation_id: str,
    ) -> Round74SealedEvaluationClaim | None:
        reservation = _require_sha256(reservation_id, "reservation")
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM round74_sealed_claims
                WHERE reservation_id = ?
                """,
                [reservation],
            ).fetchone()
            return self._claim(row) if row is not None else None
        finally:
            connection.close()

    def claim_matches(
        self,
        claim: Round74SealedEvaluationClaim,
        *,
        required_status: str,
    ) -> bool:
        claim.validate()
        if required_status not in _CLAIM_STATUSES or claim.status != required_status:
            return False
        try:
            stored = self.claim(claim.reservation_id)
        except (OSError, sqlite3.Error, Round74SealedLedgerError, ValueError):
            return False
        return stored is not None and stored.as_dict() == claim.as_dict()


__all__ = [
    "ROUND74_SEALED_CLAIM_SCHEMA_VERSION",
    "ROUND74_SEALED_DATASET_IDENTITY_SCHEMA_VERSION",
    "ROUND74_SEALED_LEDGER_SCHEMA_VERSION",
    "ROUND74_SEALED_RESULT_OUTCOMES",
    "Round74SealedDatasetIdentity",
    "Round74SealedEvaluationClaim",
    "Round74SealedEvaluationLedger",
    "Round74SealedLedgerError",
    "Round74SealedReuseError",
    "build_round74_sealed_dataset_identity",
    "default_round74_sealed_ledger_path",
]
