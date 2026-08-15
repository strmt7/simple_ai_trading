"""Irreversible pre-access governance for the Round 74 sealed test."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time


ROUND74_TERMINAL_PREACCESS_SCHEMA_VERSION = "round-074-terminal-preaccess-v3"
ROUND74_TERMINAL_ACCESS_CLAIM_SCHEMA_VERSION = "round-074-terminal-access-claim-v1"
ROUND74_TERMINAL_RESULT_BUNDLE_SCHEMA_VERSION = "round-074-terminal-result-bundle-v1"
ROUND74_TERMINAL_ONE_USE_STORE_SCHEMA_VERSION = "round-074-terminal-one-use-store-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_PROFILES = ("conservative", "regular", "aggressive")
_POPULATIONS = ("capture_run", "eligible_target")
_STATUSES = ("reserved", "complete", "failed")
_MAXIMUM_JSON_BYTES = 256 * 1024 * 1024


class Round74TerminalReuseError(RuntimeError):
    """Raised when any process attempts to reuse the terminal test access."""


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
        raise ValueError(f"Round 74 terminal {label} digest differs")
    return selected


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Round 74 terminal {label} integer differs")
    return value


def _strict_json_object(raw: str, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Round 74 terminal {label} has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"Round 74 terminal {label} contains {item}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Round 74 terminal {label} JSON differs") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Round 74 terminal {label} root differs")
    return value


@dataclass(frozen=True)
class Round74TerminalPreaccessIdentity:
    """Target-free identity frozen before any sealed target artifact is read."""

    plan_sha256: str
    coverage_sha256: str
    partition_sha256: str
    test_population_sha256: str
    test_run_ids: tuple[str, ...]
    database_route_sha256: str
    optimization_population: str
    development_bundle_sha256: str
    pretest_policy_sha256: str
    feature_scaler_sha256: str
    probability_calibration_sha256: str
    action_selection_sha256: str
    final_action_configuration_sha256: str
    ai_pretest_qualification_sha256: str
    ai_manifest_sha256: tuple[str, ...]
    profile: str
    backend_preflight_sha256: str
    model_provenance_sha256: tuple[str, ...]
    terminal_observed_wall_ns: int
    schema_version: str = ROUND74_TERMINAL_PREACCESS_SCHEMA_VERSION
    sealed_targets_read: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def validate(self) -> None:
        digests = (
            self.plan_sha256,
            self.coverage_sha256,
            self.partition_sha256,
            self.test_population_sha256,
            self.database_route_sha256,
            self.development_bundle_sha256,
            self.pretest_policy_sha256,
            self.feature_scaler_sha256,
            self.probability_calibration_sha256,
            self.action_selection_sha256,
            self.final_action_configuration_sha256,
            self.ai_pretest_qualification_sha256,
            self.backend_preflight_sha256,
            *self.ai_manifest_sha256,
            *self.model_provenance_sha256,
        )
        if (
            self.schema_version != ROUND74_TERMINAL_PREACCESS_SCHEMA_VERSION
            or any(_SHA256.fullmatch(value) is None for value in digests)
            or len(self.test_run_ids) < 24
            or len(set(self.test_run_ids)) != len(self.test_run_ids)
            or any(_RUN_ID.fullmatch(value) is None for value in self.test_run_ids)
            or self.optimization_population not in _POPULATIONS
            or not 1 <= len(self.ai_manifest_sha256) <= 2
            or len(set(self.ai_manifest_sha256)) != len(self.ai_manifest_sha256)
            or tuple(sorted(self.ai_manifest_sha256)) != self.ai_manifest_sha256
            or len(self.model_provenance_sha256) != len(self.ai_manifest_sha256)
            or len(set(self.model_provenance_sha256))
            != len(self.model_provenance_sha256)
            or self.profile not in _PROFILES
            or isinstance(self.terminal_observed_wall_ns, bool)
            or not isinstance(self.terminal_observed_wall_ns, int)
            or self.terminal_observed_wall_ns <= 0
            or any(
                (
                    self.sealed_targets_read,
                    self.trading_authority,
                    self.profitability_claim,
                )
            )
        ):
            raise ValueError("Round 74 terminal preaccess identity differs")

    @property
    def preaccess_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "plan_sha256": self.plan_sha256,
            "coverage_sha256": self.coverage_sha256,
            "partition_sha256": self.partition_sha256,
            "test_population_sha256": self.test_population_sha256,
            "test_run_ids": list(self.test_run_ids),
            "database_route_sha256": self.database_route_sha256,
            "optimization_population": self.optimization_population,
            "development_bundle_sha256": self.development_bundle_sha256,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "feature_scaler_sha256": self.feature_scaler_sha256,
            "probability_calibration_sha256": (self.probability_calibration_sha256),
            "action_selection_sha256": self.action_selection_sha256,
            "final_action_configuration_sha256": (
                self.final_action_configuration_sha256
            ),
            "ai_pretest_qualification_sha256": (self.ai_pretest_qualification_sha256),
            "ai_manifest_sha256": list(self.ai_manifest_sha256),
            "profile": self.profile,
            "backend_preflight_sha256": self.backend_preflight_sha256,
            "model_provenance_sha256": list(self.model_provenance_sha256),
            "terminal_observed_wall_ns": self.terminal_observed_wall_ns,
            "sealed_targets_read": False,
            "trading_authority": False,
            "profitability_claim": False,
        }
        if include_sha256:
            value["preaccess_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> Round74TerminalPreaccessIdentity:
        original = dict(value)
        payload = dict(original)
        claimed = str(payload.pop("preaccess_sha256", ""))
        expected_keys = {
            "schema_version",
            "plan_sha256",
            "coverage_sha256",
            "partition_sha256",
            "test_population_sha256",
            "test_run_ids",
            "database_route_sha256",
            "optimization_population",
            "development_bundle_sha256",
            "pretest_policy_sha256",
            "feature_scaler_sha256",
            "probability_calibration_sha256",
            "action_selection_sha256",
            "final_action_configuration_sha256",
            "ai_pretest_qualification_sha256",
            "ai_manifest_sha256",
            "profile",
            "backend_preflight_sha256",
            "model_provenance_sha256",
            "terminal_observed_wall_ns",
            "sealed_targets_read",
            "trading_authority",
            "profitability_claim",
        }
        test_runs = payload.get("test_run_ids")
        manifests = payload.get("ai_manifest_sha256")
        provenance = payload.get("model_provenance_sha256")
        if (
            set(payload) != expected_keys
            or _SHA256.fullmatch(claimed) is None
            or claimed != _canonical_sha256(payload)
            or not isinstance(test_runs, list)
            or not isinstance(manifests, list)
            or not isinstance(provenance, list)
            or payload.get("sealed_targets_read") is not False
            or payload.get("trading_authority") is not False
            or payload.get("profitability_claim") is not False
        ):
            raise ValueError("Round 74 terminal preaccess payload differs")
        try:
            selected = cls(
                plan_sha256=str(payload["plan_sha256"]),
                coverage_sha256=str(payload["coverage_sha256"]),
                partition_sha256=str(payload["partition_sha256"]),
                test_population_sha256=str(payload["test_population_sha256"]),
                test_run_ids=tuple(str(item) for item in test_runs),
                database_route_sha256=str(payload["database_route_sha256"]),
                optimization_population=str(payload["optimization_population"]),
                development_bundle_sha256=str(payload["development_bundle_sha256"]),
                pretest_policy_sha256=str(payload["pretest_policy_sha256"]),
                feature_scaler_sha256=str(payload["feature_scaler_sha256"]),
                probability_calibration_sha256=str(
                    payload["probability_calibration_sha256"]
                ),
                action_selection_sha256=str(payload["action_selection_sha256"]),
                final_action_configuration_sha256=str(
                    payload["final_action_configuration_sha256"]
                ),
                ai_pretest_qualification_sha256=str(
                    payload["ai_pretest_qualification_sha256"]
                ),
                ai_manifest_sha256=tuple(str(item) for item in manifests),
                profile=str(payload["profile"]),
                backend_preflight_sha256=str(payload["backend_preflight_sha256"]),
                model_provenance_sha256=tuple(str(item) for item in provenance),
                terminal_observed_wall_ns=_strict_int(
                    payload["terminal_observed_wall_ns"],
                    "observed wall",
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 terminal preaccess payload differs") from exc
        selected.validate()
        if selected.as_dict() != original:
            raise ValueError("Round 74 terminal preaccess identity differs")
        return selected


@dataclass(frozen=True)
class Round74TerminalAccessClaim:
    """Durable proof that sealed access was consumed before target loading."""

    reservation_id: str
    preaccess_sha256: str
    test_unlock_sha256: str
    preaccess: Round74TerminalPreaccessIdentity
    status: str
    result_sha256: str
    error: str
    reserved_at_ns: int
    completed_at_ns: int | None
    schema_version: str = ROUND74_TERMINAL_ACCESS_CLAIM_SCHEMA_VERSION

    def validate(self) -> None:
        self.preaccess.validate()
        if (
            self.schema_version != ROUND74_TERMINAL_ACCESS_CLAIM_SCHEMA_VERSION
            or _SHA256.fullmatch(self.reservation_id) is None
            or self.preaccess_sha256 != self.preaccess.preaccess_sha256
            or _SHA256.fullmatch(self.test_unlock_sha256) is None
            or self.status not in _STATUSES
            or isinstance(self.reserved_at_ns, bool)
            or not isinstance(self.reserved_at_ns, int)
            or self.reserved_at_ns <= 0
            or self.error != " ".join(self.error.split())[:2_000]
        ):
            raise ValueError("Round 74 terminal access claim differs")
        if self.status == "reserved":
            valid_completion = (
                not self.result_sha256
                and not self.error
                and self.completed_at_ns is None
            )
        else:
            valid_completion = (
                _SHA256.fullmatch(self.result_sha256) is not None
                and isinstance(self.completed_at_ns, int)
                and not isinstance(self.completed_at_ns, bool)
                and self.completed_at_ns >= self.reserved_at_ns
                and (bool(self.error) if self.status == "failed" else not self.error)
            )
        if not valid_completion:
            raise ValueError("Round 74 terminal access completion differs")

    @property
    def claim_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "reservation_id": self.reservation_id,
            "preaccess_sha256": self.preaccess_sha256,
            "test_unlock_sha256": self.test_unlock_sha256,
            "preaccess": self.preaccess.as_dict(),
            "status": self.status,
            "result_sha256": self.result_sha256,
            "error": self.error,
            "reserved_at_ns": self.reserved_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "access_consumed_before_target_loading": True,
            "reservation_reset_api_available": False,
        }
        if include_sha256:
            value["claim_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> Round74TerminalAccessClaim:
        original = dict(value)
        payload = dict(original)
        claimed = str(payload.pop("claim_sha256", ""))
        preaccess = payload.get("preaccess")
        if (
            _SHA256.fullmatch(claimed) is None
            or claimed != _canonical_sha256(payload)
            or not isinstance(preaccess, Mapping)
            or payload.pop("access_consumed_before_target_loading", None) is not True
            or payload.pop("reservation_reset_api_available", None) is not False
        ):
            raise ValueError("Round 74 terminal access claim payload differs")
        expected_keys = {
            "schema_version",
            "reservation_id",
            "preaccess_sha256",
            "test_unlock_sha256",
            "preaccess",
            "status",
            "result_sha256",
            "error",
            "reserved_at_ns",
            "completed_at_ns",
        }
        if set(payload) != expected_keys:
            raise ValueError("Round 74 terminal access claim fields differ")
        try:
            selected = cls(
                reservation_id=str(payload["reservation_id"]),
                preaccess_sha256=str(payload["preaccess_sha256"]),
                test_unlock_sha256=str(payload["test_unlock_sha256"]),
                preaccess=Round74TerminalPreaccessIdentity.from_mapping(preaccess),
                status=str(payload["status"]),
                result_sha256=str(payload["result_sha256"]),
                error=str(payload["error"]),
                reserved_at_ns=_strict_int(payload["reserved_at_ns"], "reserved"),
                completed_at_ns=(
                    None
                    if payload["completed_at_ns"] is None
                    else _strict_int(payload["completed_at_ns"], "completed")
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 terminal access claim payload differs") from exc
        selected.validate()
        if selected.claim_sha256 != claimed or selected.as_dict() != original:
            raise ValueError("Round 74 terminal access claim identity differs")
        return selected


def _validate_terminal_cross_bindings(
    access_claim: Round74TerminalAccessClaim,
    *,
    dataset: Mapping[str, object],
    report: Mapping[str, object],
    sealed_claim: Mapping[str, object],
) -> None:
    access_claim.validate()
    preaccess = access_claim.preaccess
    expected_test_access_sha256 = _canonical_sha256(
        {
            "pretest_model_policy_sha256": preaccess.pretest_policy_sha256,
            "test_unlock_sha256": access_claim.test_unlock_sha256,
        }
    )
    dataset_bindings = {
        "partition_sha256": preaccess.partition_sha256,
        "scaler_sha256": preaccess.feature_scaler_sha256,
        "optimization_population": preaccess.optimization_population,
        "test_population_sha256": preaccess.test_population_sha256,
        "test_run_ids": list(preaccess.test_run_ids),
        "test_access_sha256": expected_test_access_sha256,
    }
    expected_report = {
        "pretest_policy_sha256": preaccess.pretest_policy_sha256,
        "probability_calibration_sha256": (preaccess.probability_calibration_sha256),
        "action_selection_sha256": preaccess.action_selection_sha256,
        "final_action_configuration_sha256": (
            preaccess.final_action_configuration_sha256
        ),
        "ai_pretest_qualification_sha256": (preaccess.ai_pretest_qualification_sha256),
        "profile": preaccess.profile,
        "optimization_population": preaccess.optimization_population,
        "test_access_sha256": expected_test_access_sha256,
    }
    expected_claim = {
        **expected_report,
        "partition_sha256": preaccess.partition_sha256,
        "scaler_sha256": preaccess.feature_scaler_sha256,
        "test_population_sha256": preaccess.test_population_sha256,
        "test_run_ids": list(preaccess.test_run_ids),
    }
    if (
        any(dataset.get(key) != expected for key, expected in dataset_bindings.items())
        or any(report.get(key) != expected for key, expected in expected_report.items())
        or any(
            sealed_claim.get(key) != expected
            for key, expected in expected_claim.items()
        )
        or report.get("reservation_id") != sealed_claim.get("reservation_id")
    ):
        raise ValueError("Round 74 terminal result cross-binding differs")


def build_round74_terminal_result_bundle(
    *,
    access_claim: Round74TerminalAccessClaim,
    dataset_identity: Mapping[str, object],
    sealed_report: Mapping[str, object],
    finalized_sealed_claim: Mapping[str, object],
) -> dict[str, object]:
    """Bind the complete successful result to the irreversible access claim."""

    access_claim.validate()
    if access_claim.status != "reserved":
        raise ValueError("Round 74 terminal access claim is not live")
    dataset = dict(dataset_identity)
    report = dict(sealed_report)
    sealed_claim = dict(finalized_sealed_claim)
    dataset_sha256 = _canonical_sha256(dataset)
    report_sha256 = str(report.get("report_sha256", ""))
    sealed_claim_sha256 = str(sealed_claim.get("claim_sha256", ""))
    if (
        _SHA256.fullmatch(report_sha256) is None
        or report_sha256
        != _canonical_sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        or _SHA256.fullmatch(sealed_claim_sha256) is None
        or sealed_claim_sha256
        != _canonical_sha256(
            {key: value for key, value in sealed_claim.items() if key != "claim_sha256"}
        )
        or report.get("dataset_sha256") != dataset_sha256
        or sealed_claim.get("dataset_sha256") != dataset_sha256
        or report.get("test_access_sha256") != sealed_claim.get("test_access_sha256")
        or sealed_claim.get("result_sha256") != report_sha256
        or sealed_claim.get("status") != "complete"
    ):
        raise ValueError("Round 74 terminal sealed result identity differs")
    _validate_terminal_cross_bindings(
        access_claim,
        dataset=dataset,
        report=report,
        sealed_claim=sealed_claim,
    )
    value: dict[str, object] = {
        "schema_version": ROUND74_TERMINAL_RESULT_BUNDLE_SCHEMA_VERSION,
        "access_claim": access_claim.as_dict(),
        "dataset_identity": dataset,
        "sealed_report": report,
        "finalized_sealed_claim": sealed_claim,
        "sealed_test_accessed": True,
        "orders_submitted": False,
        "promotion_authority": False,
        "paper_trading_authority": False,
        "testnet_trading_authority": False,
        "live_trading_authority": False,
        "profitability_claim": False,
        "leverage_applied": False,
    }
    value["bundle_sha256"] = _canonical_sha256(value)
    return value


def validate_round74_terminal_result_bundle(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate a persisted terminal bundle without reopening sealed data."""

    payload = dict(value)
    claimed = str(payload.pop("bundle_sha256", ""))
    expected_keys = {
        "schema_version",
        "access_claim",
        "dataset_identity",
        "sealed_report",
        "finalized_sealed_claim",
        "sealed_test_accessed",
        "orders_submitted",
        "promotion_authority",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
        "profitability_claim",
        "leverage_applied",
    }
    access = payload.get("access_claim")
    dataset = payload.get("dataset_identity")
    report = payload.get("sealed_report")
    sealed_claim = payload.get("finalized_sealed_claim")
    false_fields = (
        "orders_submitted",
        "promotion_authority",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
        "profitability_claim",
        "leverage_applied",
    )
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != ROUND74_TERMINAL_RESULT_BUNDLE_SCHEMA_VERSION
        or _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
        or payload.get("sealed_test_accessed") is not True
        or any(payload.get(field) is not False for field in false_fields)
        or not isinstance(access, Mapping)
        or not isinstance(dataset, Mapping)
        or not isinstance(report, Mapping)
        or not isinstance(sealed_claim, Mapping)
    ):
        raise ValueError("Round 74 terminal result bundle differs")
    access_claim = Round74TerminalAccessClaim.from_mapping(access)
    if access_claim.status != "reserved":
        raise ValueError("Round 74 terminal result access claim differs")
    dataset_sha256 = _canonical_sha256(dict(dataset))
    report_payload = dict(report)
    report_claimed = str(report_payload.pop("report_sha256", ""))
    sealed_payload = dict(sealed_claim)
    sealed_claimed = str(sealed_payload.pop("claim_sha256", ""))
    if (
        _SHA256.fullmatch(report_claimed) is None
        or report_claimed != _canonical_sha256(report_payload)
        or _SHA256.fullmatch(sealed_claimed) is None
        or sealed_claimed != _canonical_sha256(sealed_payload)
        or report.get("dataset_sha256") != dataset_sha256
        or sealed_claim.get("dataset_sha256") != dataset_sha256
        or sealed_claim.get("status") != "complete"
        or sealed_claim.get("result_sha256") != report_claimed
    ):
        raise ValueError("Round 74 terminal persisted sealed result differs")
    _validate_terminal_cross_bindings(
        access_claim,
        dataset=dataset,
        report=report,
        sealed_claim=sealed_claim,
    )
    restored = dict(value)
    if restored.get("bundle_sha256") != claimed:
        raise ValueError("Round 74 terminal result bundle identity differs")
    return restored


class Round74TerminalOneUseStore:
    """SQLite one-use store with full terminal-result recovery."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise ValueError("Round 74 terminal one-use store path differs")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            mode = str(connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0])
            if mode.lower() != "delete":
                raise RuntimeError("Round 74 terminal journal mode differs")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS governance (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS terminal_access (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    reservation_id TEXT NOT NULL UNIQUE,
                    preaccess_sha256 TEXT NOT NULL UNIQUE,
                    test_unlock_sha256 TEXT NOT NULL UNIQUE,
                    preaccess_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    reserved_at_ns INTEGER NOT NULL,
                    completed_at_ns INTEGER,
                    CHECK (status IN ('reserved', 'complete', 'failed'))
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO governance (key, value) VALUES (?, ?)",
                ("schema_version", ROUND74_TERMINAL_ONE_USE_STORE_SCHEMA_VERSION),
            )
            schema = connection.execute(
                "SELECT value FROM governance WHERE key = 'schema_version'"
            ).fetchone()
            if (
                schema is None
                or str(schema[0]) != ROUND74_TERMINAL_ONE_USE_STORE_SCHEMA_VERSION
            ):
                raise RuntimeError("Round 74 terminal one-use schema differs")
            check = connection.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise RuntimeError("Round 74 terminal one-use integrity differs")
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _claim(row: sqlite3.Row) -> Round74TerminalAccessClaim:
        preaccess = Round74TerminalPreaccessIdentity.from_mapping(
            _strict_json_object(str(row["preaccess_json"]), "stored preaccess")
        )
        claim = Round74TerminalAccessClaim(
            reservation_id=str(row["reservation_id"]),
            preaccess_sha256=str(row["preaccess_sha256"]),
            test_unlock_sha256=str(row["test_unlock_sha256"]),
            preaccess=preaccess,
            status=str(row["status"]),
            result_sha256=str(row["result_sha256"]),
            error=str(row["error"]),
            reserved_at_ns=int(row["reserved_at_ns"]),
            completed_at_ns=(
                None if row["completed_at_ns"] is None else int(row["completed_at_ns"])
            ),
        )
        claim.validate()
        return claim

    def reserve(
        self,
        preaccess: Round74TerminalPreaccessIdentity,
    ) -> Round74TerminalAccessClaim:
        """Consume the single access before a caller may read test targets."""

        preaccess.validate()
        now_ns = time.time_ns()
        entropy = os.urandom(32)
        reservation_id = hashlib.sha256(
            preaccess.preaccess_sha256.encode("ascii")
            + str(now_ns).encode("ascii")
            + entropy
        ).hexdigest()
        test_unlock_sha256 = _canonical_sha256(
            {
                "schema_version": ROUND74_TERMINAL_ACCESS_CLAIM_SCHEMA_VERSION,
                "reservation_id": reservation_id,
                "preaccess_sha256": preaccess.preaccess_sha256,
                "purpose": "round74_sealed_test_unlock",
            }
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT reservation_id, status FROM terminal_access WHERE singleton = 1"
            ).fetchone()
            if prior is not None:
                raise Round74TerminalReuseError(
                    "Round 74 terminal test access was already consumed: "
                    f"reservation={prior['reservation_id']} status={prior['status']}"
                )
            connection.execute(
                """
                INSERT INTO terminal_access (
                    singleton, reservation_id, preaccess_sha256,
                    test_unlock_sha256, preaccess_json, status, reserved_at_ns
                ) VALUES (1, ?, ?, ?, ?, 'reserved', ?)
                """,
                (
                    reservation_id,
                    preaccess.preaccess_sha256,
                    test_unlock_sha256,
                    _canonical_json(preaccess.as_dict()),
                    now_ns,
                ),
            )
            row = connection.execute(
                "SELECT * FROM terminal_access WHERE singleton = 1"
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise RuntimeError("Round 74 terminal reservation disappeared")
            return self._claim(row)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def finalize_success(
        self,
        claim: Round74TerminalAccessClaim,
        bundle: Mapping[str, object],
    ) -> Round74TerminalAccessClaim:
        claim.validate()
        validated = validate_round74_terminal_result_bundle(bundle)
        result_sha256 = _require_sha256(
            validated.get("bundle_sha256"),
            "result bundle",
        )
        return self._finalize(
            claim,
            status="complete",
            result_sha256=result_sha256,
            result_json=_canonical_json(validated),
            error="",
        )

    def finalize_failure(
        self,
        claim: Round74TerminalAccessClaim,
        error: BaseException,
    ) -> Round74TerminalAccessClaim:
        claim.validate()
        detail = " ".join(f"{error.__class__.__name__}: {error}".split())[:2_000]
        if not detail:
            detail = "RuntimeError: Round 74 terminal evaluation failed"
        result_sha256 = _canonical_sha256(
            {
                "schema_version": ROUND74_TERMINAL_RESULT_BUNDLE_SCHEMA_VERSION,
                "reservation_id": claim.reservation_id,
                "status": "failed",
                "error": detail,
            }
        )
        return self._finalize(
            claim,
            status="failed",
            result_sha256=result_sha256,
            result_json="",
            error=detail,
        )

    def _finalize(
        self,
        claim: Round74TerminalAccessClaim,
        *,
        status: str,
        result_sha256: str,
        result_json: str,
        error: str,
    ) -> Round74TerminalAccessClaim:
        if claim.status != "reserved" or status not in ("complete", "failed"):
            raise ValueError("Round 74 terminal finalization differs")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM terminal_access WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise RuntimeError("Round 74 terminal reservation is missing")
            current = self._claim(row)
            if (
                current.status != "reserved"
                or current.claim_sha256 != claim.claim_sha256
            ):
                raise Round74TerminalReuseError(
                    "Round 74 terminal reservation was already finalized"
                )
            completed_at_ns = max(time.time_ns(), claim.reserved_at_ns)
            connection.execute(
                """
                UPDATE terminal_access
                SET status = ?, result_sha256 = ?, result_json = ?,
                    error = ?, completed_at_ns = ?
                WHERE singleton = 1 AND status = 'reserved'
                """,
                (
                    status,
                    result_sha256,
                    result_json,
                    error,
                    completed_at_ns,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM terminal_access WHERE singleton = 1"
            ).fetchone()
            connection.execute("COMMIT")
            if updated is None:
                raise RuntimeError("Round 74 terminal completion disappeared")
            return self._claim(updated)
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def claim(self) -> Round74TerminalAccessClaim | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM terminal_access WHERE singleton = 1"
            ).fetchone()
            return None if row is None else self._claim(row)
        finally:
            connection.close()

    def load_completed_bundle(self) -> dict[str, object]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT status, result_sha256, result_json FROM terminal_access "
                "WHERE singleton = 1"
            ).fetchone()
            if row is None or str(row["status"]) != "complete":
                raise ValueError("Round 74 terminal completed result is unavailable")
            raw = str(row["result_json"])
            if not raw or len(raw.encode("utf-8")) > _MAXIMUM_JSON_BYTES:
                raise ValueError("Round 74 terminal stored result size differs")
            result = validate_round74_terminal_result_bundle(
                _strict_json_object(raw, "stored result")
            )
            if result.get("bundle_sha256") != str(row["result_sha256"]):
                raise ValueError("Round 74 terminal stored result digest differs")
            return result
        finally:
            connection.close()


__all__ = [
    "ROUND74_TERMINAL_ACCESS_CLAIM_SCHEMA_VERSION",
    "ROUND74_TERMINAL_ONE_USE_STORE_SCHEMA_VERSION",
    "ROUND74_TERMINAL_PREACCESS_SCHEMA_VERSION",
    "ROUND74_TERMINAL_RESULT_BUNDLE_SCHEMA_VERSION",
    "Round74TerminalAccessClaim",
    "Round74TerminalOneUseStore",
    "Round74TerminalPreaccessIdentity",
    "Round74TerminalReuseError",
    "build_round74_terminal_result_bundle",
    "validate_round74_terminal_result_bundle",
]
