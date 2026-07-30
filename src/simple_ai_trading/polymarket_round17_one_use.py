"""Append-only one-use claim boundary for Polymarket Round 17 held-out data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import subprocess
import time

from .polymarket_round17_campaign_operator import (
    Round17CampaignOperatorConfig,
    inspect_round17_campaign_readiness,
)
from .polymarket_round17_development_operator import (
    validate_round17_development_result,
)


POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256 = (
    "e7e4471cd71882beee8837ac6251e52d407c5be57f716d703a816f70b9abf5dd"
)
POLYMARKET_ROUND17_ONE_USE_CLAIM_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-one-use-claim-v1"
)
POLYMARKET_ROUND17_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-implementation-manifest-v1"
)
POLYMARKET_ROUND17_ONE_USE_STORE_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-one-use-store-v1"
)
POLYMARKET_ROUND17_TEST_ACCESS_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-test-access-v1"
)
POLYMARKET_ROUND17_TEST_FIRST_SLOT = 1012
POLYMARKET_ROUND17_TEST_LAST_SLOT = 1439
POLYMARKET_ROUND17_TEST_START_MS = 1_787_166_000_000
POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE = 1_787_936_400_000
_IMPLEMENTATION_PATHS = (
    "docs/model-research/polymarket/"
    "round-017-btc-5m-one-use-evaluation-contract-v1.json",
    "src/simple_ai_trading/polymarket_round17_campaign_operator.py",
    "src/simple_ai_trading/polymarket_round17_cohort.py",
    "src/simple_ai_trading/polymarket_round17_dataset.py",
    "src/simple_ai_trading/polymarket_round17_development_operator.py",
    "src/simple_ai_trading/polymarket_round17_economic.py",
    "src/simple_ai_trading/polymarket_round17_evaluation.py",
    "src/simple_ai_trading/polymarket_round17_execution.py",
    "src/simple_ai_trading/polymarket_round17_features.py",
    "src/simple_ai_trading/polymarket_round17_model.py",
    "src/simple_ai_trading/polymarket_round17_outcomes.py",
    "src/simple_ai_trading/polymarket_round17_resolution.py",
    "src/simple_ai_trading/polymarket_round17_uncertainty.py",
    "src/simple_ai_trading/polymarket_round17_one_use.py",
    "tools/run_round17_polymarket_one_use.py",
)
_CLAIM_STATUSES = frozenset(
    {"open", "access_open", "resolution_pending", "completed", "failed"}
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_MAXIMUM_CONTRACT_BYTES = 256 * 1024
_MAXIMUM_IMPLEMENTATION_FILE_BYTES = 4 * 1024 * 1024
_MAXIMUM_RESULT_BYTES = 512 * 1024 * 1024


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 one-use artifact contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 one-use artifact contains {value}")


def _strict_mapping_json(value: str, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not isinstance(payload, Mapping) or _canonical_json(payload) != value:
        raise ValueError(f"{name} is not a canonical object")
    return dict(payload)


def load_round17_one_use_contract(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAXIMUM_CONTRACT_BYTES
    ):
        raise ValueError("Round 17 one-use contract is unavailable")
    try:
        payload = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 17 one-use contract is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 17 one-use contract is not an object")
    contract = dict(payload)
    claimed = str(contract.pop("contract_sha256", "")).strip().lower()
    parents = contract.get("parents")
    partition = contract.get("test_partition")
    endpoint = contract.get("endpoint_evaluation")
    economic = contract.get("economic_evaluation")
    authority = contract.get("authority")
    if (
        claimed != POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
        or claimed != _canonical_sha256(contract)
        or contract.get("schema_version")
        != "polymarket-round17-btc-5m-one-use-evaluation-contract-v1"
        or contract.get("round") != 17
        or contract.get("status")
        != "preregistered_before_test_feature_or_outcome_access"
        or not isinstance(parents, Mapping)
        or parents.get("required_development_status") != "development_accepted"
        or not isinstance(partition, Mapping)
        or partition.get("role") != "test"
        or partition.get("first_slot") != POLYMARKET_ROUND17_TEST_FIRST_SLOT
        or partition.get("last_slot") != POLYMARKET_ROUND17_TEST_LAST_SLOT
        or partition.get("start_ms") != POLYMARKET_ROUND17_TEST_START_MS
        or partition.get("end_ms_exclusive")
        != POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE
        or partition.get("minimum_resolved_conditions") != 1800
        or partition.get("minimum_calendar_days") != 7
        or not isinstance(endpoint, Mapping)
        or endpoint.get("paired_bootstrap_unit") != "condition"
        or endpoint.get("paired_bootstrap_samples") != 2000
        or endpoint.get("paired_bootstrap_seed") != 17017
        or not isinstance(economic, Mapping)
        or economic.get("minimum_executed_actions_per_profile_per_scenario")
        != 300
        or contract.get("implementation_manifest_paths")
        != list(_IMPLEMENTATION_PATHS)
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
    ):
        raise ValueError("Round 17 one-use contract identity differs")
    return {**contract, "contract_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round17ImplementationFile:
    path: str
    bytes: int
    sha256: str

    def asdict(self) -> dict[str, object]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    def validated(self) -> Round17ImplementationFile:
        selected = PurePosixPath(self.path)
        if (
            selected.is_absolute()
            or ".." in selected.parts
            or selected.as_posix() != self.path
            or self.bytes < 1
            or self.bytes > _MAXIMUM_IMPLEMENTATION_FILE_BYTES
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("Round 17 implementation file identity differs")
        return self


@dataclass(frozen=True, slots=True)
class Round17ImplementationManifest:
    repository_commit_sha: str
    files: tuple[Round17ImplementationFile, ...]
    manifest_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND17_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION
            ),
            "evaluation_contract_sha256": (
                POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
            ),
            "repository_commit_sha": self.repository_commit_sha,
            "files": [item.asdict() for item in self.files],
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "manifest_sha256": self.manifest_sha256}

    def validated(self) -> Round17ImplementationManifest:
        if (
            _COMMIT.fullmatch(self.repository_commit_sha) is None
            or tuple(item.path for item in self.files) != _IMPLEMENTATION_PATHS
            or any(item.validated() is not item for item in self.files)
            or _SHA256.fullmatch(self.manifest_sha256) is None
            or self.manifest_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 implementation manifest differs")
        return self


def build_round17_implementation_manifest(
    repository: str | Path,
    *,
    repository_commit_sha: str,
) -> Round17ImplementationManifest:
    root = Path(repository).resolve()
    commit = str(repository_commit_sha or "").strip().lower()
    if not root.is_dir() or root.is_symlink() or _COMMIT.fullmatch(commit) is None:
        raise ValueError("Round 17 implementation repository identity differs")
    try:
        observed_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip().lower()
        dirty = subprocess.run(
            ["git", "status", "--porcelain=v1", "--", *_IMPLEMENTATION_PATHS],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Round 17 implementation Git identity is unavailable") from exc
    if observed_commit != commit or dirty.strip():
        raise ValueError("Round 17 implementation files are not clean at the commit")
    files: list[Round17ImplementationFile] = []
    for relative in _IMPLEMENTATION_PATHS:
        path = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if (
            root not in path.parents
            or path.is_symlink()
            or not path.is_file()
            or not 1 <= path.stat().st_size <= _MAXIMUM_IMPLEMENTATION_FILE_BYTES
        ):
            raise ValueError("Round 17 implementation file is unavailable")
        payload = path.read_bytes()
        files.append(
            Round17ImplementationFile(
                path=relative,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ).validated()
        )
    provisional = Round17ImplementationManifest(
        repository_commit_sha=commit,
        files=tuple(files),
    )
    return replace(
        provisional,
        manifest_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round17TestAccessClaim:
    development_result_sha256: str
    campaign_readiness_sha256: str
    campaign_development_index_sha256: str
    cohort_manifest_sha256: str
    target_manifest_sha256: str
    model_pretest_sha256: str
    probability_calibration_sha256: str
    economic_pretest_sha256: str
    implementation_manifest_sha256: str
    repository_commit_sha: str
    opened_at_ms: int
    claim_sha256: str = ""

    def binding_payload(self) -> dict[str, object]:
        return {
            "evaluation_contract_sha256": (
                POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
            ),
            "development_result_sha256": self.development_result_sha256,
            "campaign_readiness_sha256": self.campaign_readiness_sha256,
            "campaign_development_index_sha256": (
                self.campaign_development_index_sha256
            ),
            "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "model_pretest_sha256": self.model_pretest_sha256,
            "probability_calibration_sha256": (
                self.probability_calibration_sha256
            ),
            "economic_pretest_sha256": self.economic_pretest_sha256,
            "implementation_manifest_sha256": self.implementation_manifest_sha256,
            "repository_commit_sha": self.repository_commit_sha,
            "test_first_slot": POLYMARKET_ROUND17_TEST_FIRST_SLOT,
            "test_last_slot": POLYMARKET_ROUND17_TEST_LAST_SLOT,
            "test_start_ms": POLYMARKET_ROUND17_TEST_START_MS,
            "test_end_ms_exclusive": POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND17_ONE_USE_CLAIM_SCHEMA_VERSION,
            **self.binding_payload(),
            "opened_at_ms": self.opened_at_ms,
            "test_features_accessed": False,
            "test_targets_accessed": False,
            "test_execution_accessed": False,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "binance_credentials_used": False,
            "binance_execution_connected": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "claim_sha256": self.claim_sha256}

    def validated(self) -> Round17TestAccessClaim:
        hashes = (
            self.development_result_sha256,
            self.campaign_readiness_sha256,
            self.campaign_development_index_sha256,
            self.cohort_manifest_sha256,
            self.target_manifest_sha256,
            self.model_pretest_sha256,
            self.probability_calibration_sha256,
            self.economic_pretest_sha256,
            self.implementation_manifest_sha256,
            self.claim_sha256,
        )
        if (
            any(_SHA256.fullmatch(value) is None for value in hashes)
            or _COMMIT.fullmatch(self.repository_commit_sha) is None
            or self.opened_at_ms <= 0
            or self.claim_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 one-use claim differs")
        return self


def _claim_from_mapping(value: Mapping[str, object]) -> Round17TestAccessClaim:
    expected = {
        "schema_version",
        "evaluation_contract_sha256",
        "development_result_sha256",
        "campaign_readiness_sha256",
        "campaign_development_index_sha256",
        "cohort_manifest_sha256",
        "target_manifest_sha256",
        "model_pretest_sha256",
        "probability_calibration_sha256",
        "economic_pretest_sha256",
        "implementation_manifest_sha256",
        "repository_commit_sha",
        "test_first_slot",
        "test_last_slot",
        "test_start_ms",
        "test_end_ms_exclusive",
        "opened_at_ms",
        "test_features_accessed",
        "test_targets_accessed",
        "test_execution_accessed",
        "automatic_promotion",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
        "binance_credentials_used",
        "binance_execution_connected",
        "claim_sha256",
    }
    if (
        set(value) != expected
        or value.get("schema_version")
        != POLYMARKET_ROUND17_ONE_USE_CLAIM_SCHEMA_VERSION
        or value.get("evaluation_contract_sha256")
        != POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256
        or value.get("test_first_slot") != POLYMARKET_ROUND17_TEST_FIRST_SLOT
        or value.get("test_last_slot") != POLYMARKET_ROUND17_TEST_LAST_SLOT
        or value.get("test_start_ms") != POLYMARKET_ROUND17_TEST_START_MS
        or value.get("test_end_ms_exclusive")
        != POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE
        or any(
            value.get(name) is not False
            for name in (
                "test_features_accessed",
                "test_targets_accessed",
                "test_execution_accessed",
                "automatic_promotion",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
                "binance_credentials_used",
                "binance_execution_connected",
            )
        )
    ):
        raise ValueError("Round 17 one-use claim schema differs")
    return Round17TestAccessClaim(
        development_result_sha256=str(value["development_result_sha256"]),
        campaign_readiness_sha256=str(value["campaign_readiness_sha256"]),
        campaign_development_index_sha256=str(
            value["campaign_development_index_sha256"]
        ),
        cohort_manifest_sha256=str(value["cohort_manifest_sha256"]),
        target_manifest_sha256=str(value["target_manifest_sha256"]),
        model_pretest_sha256=str(value["model_pretest_sha256"]),
        probability_calibration_sha256=str(
            value["probability_calibration_sha256"]
        ),
        economic_pretest_sha256=str(value["economic_pretest_sha256"]),
        implementation_manifest_sha256=str(
            value["implementation_manifest_sha256"]
        ),
        repository_commit_sha=str(value["repository_commit_sha"]),
        opened_at_ms=int(value["opened_at_ms"]),
        claim_sha256=str(value["claim_sha256"]),
    ).validated()


class Round17OneUseClaimStore:
    """One-row claim state with a hash-chained append-only transition log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.is_symlink() or (
            self.path.exists() and not self.path.is_file()
        ):
            raise ValueError("Round 17 one-use store path is invalid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA temp_store = MEMORY")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._initialize()

    def __enter__(self) -> Round17OneUseClaimStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS round17_one_use_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS round17_one_use_claim (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                claim_sha256 TEXT NOT NULL UNIQUE CHECK (length(claim_sha256) = 64),
                claim_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'open', 'access_open', 'resolution_pending',
                        'completed', 'failed'
                    )
                ),
                test_access_sha256 TEXT UNIQUE,
                access_started_ms INTEGER,
                result_sha256 TEXT UNIQUE,
                result_json TEXT,
                failure_json TEXT,
                CHECK (
                    (test_access_sha256 IS NULL AND access_started_ms IS NULL)
                    OR (
                        length(test_access_sha256) = 64
                        AND access_started_ms > 0
                    )
                ),
                CHECK (
                    (result_sha256 IS NULL AND result_json IS NULL)
                    OR (length(result_sha256) = 64 AND result_json IS NOT NULL)
                )
            );
            CREATE TABLE IF NOT EXISTS round17_one_use_event (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_sha256 TEXT NOT NULL UNIQUE CHECK (length(event_sha256) = 64),
                previous_event_sha256 TEXT NOT NULL,
                event_json TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            """
            INSERT OR IGNORE INTO round17_one_use_metadata(singleton, schema_version)
            VALUES (1, ?)
            """,
            (POLYMARKET_ROUND17_ONE_USE_STORE_SCHEMA_VERSION,),
        )
        row = self.connection.execute(
            "SELECT schema_version FROM round17_one_use_metadata WHERE singleton = 1"
        ).fetchone()
        if (
            row is None
            or row["schema_version"]
            != POLYMARKET_ROUND17_ONE_USE_STORE_SCHEMA_VERSION
        ):
            raise ValueError("Round 17 one-use store schema differs")

    def _append_event(
        self,
        *,
        event_type: str,
        claim_sha256: str,
        observed_at_ms: int,
        details: Mapping[str, object],
    ) -> None:
        prior = self.connection.execute(
            """
            SELECT event_sha256 FROM round17_one_use_event
            ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        previous = "" if prior is None else str(prior["event_sha256"])
        payload = {
            "schema_version": "polymarket-round17-one-use-event-v1",
            "event_type": event_type,
            "claim_sha256": claim_sha256,
            "observed_at_ms": int(observed_at_ms),
            "previous_event_sha256": previous,
            "details": dict(details),
        }
        digest = _canonical_sha256(payload)
        self.connection.execute(
            """
            INSERT INTO round17_one_use_event(
                event_sha256, previous_event_sha256, event_json
            ) VALUES (?, ?, ?)
            """,
            (digest, previous, _canonical_json(payload)),
        )

    def open_claim(self, candidate: Round17TestAccessClaim) -> Round17TestAccessClaim:
        selected = candidate.validated()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                """
                SELECT claim_json FROM round17_one_use_claim
                WHERE singleton = 1
                """
            ).fetchone()
            if row is None:
                self.connection.execute(
                    """
                    INSERT INTO round17_one_use_claim(
                        singleton, claim_sha256, claim_json, status
                    ) VALUES (1, ?, ?, 'open')
                    """,
                    (selected.claim_sha256, _canonical_json(selected.asdict())),
                )
                self._append_event(
                    event_type="claim_opened",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=selected.opened_at_ms,
                    details={"binding_sha256": _canonical_sha256(selected.binding_payload())},
                )
                output = selected
            else:
                existing = _claim_from_mapping(
                    _strict_mapping_json(
                        str(row["claim_json"]),
                        name="Round 17 stored claim",
                    )
                )
                if existing.binding_payload() != selected.binding_payload():
                    raise RuntimeError(
                        "Round 17 one-use claim already binds different evidence"
                    )
                output = existing
            self.connection.execute("COMMIT")
            return output
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def _claim_row(self, claim_sha256: str) -> sqlite3.Row:
        row = self.connection.execute(
            """
            SELECT * FROM round17_one_use_claim
            WHERE singleton = 1 AND claim_sha256 = ?
            """,
            (claim_sha256,),
        ).fetchone()
        if row is None:
            raise ValueError("Round 17 one-use claim is unavailable")
        return row

    def consume_test_access(
        self,
        claim: Round17TestAccessClaim,
        *,
        observed_at_ms: int | None = None,
    ) -> str:
        selected = claim.validated()
        now = (
            int(time.time_ns() // 1_000_000)
            if observed_at_ms is None
            else int(observed_at_ms)
        )
        if now <= 0:
            raise ValueError("Round 17 test access time is invalid")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._claim_row(selected.claim_sha256)
            status = str(row["status"])
            if status in {"completed", "failed"}:
                raise RuntimeError("Round 17 one-use claim is terminal")
            existing = row["test_access_sha256"]
            if existing is None:
                access_payload = {
                    "schema_version": POLYMARKET_ROUND17_TEST_ACCESS_SCHEMA_VERSION,
                    "claim_sha256": selected.claim_sha256,
                    "access_started_ms": now,
                    "test_first_slot": POLYMARKET_ROUND17_TEST_FIRST_SLOT,
                    "test_last_slot": POLYMARKET_ROUND17_TEST_LAST_SLOT,
                }
                access_sha256 = _canonical_sha256(access_payload)
                self.connection.execute(
                    """
                    UPDATE round17_one_use_claim
                    SET status = 'access_open',
                        test_access_sha256 = ?,
                        access_started_ms = ?
                    WHERE singleton = 1
                    """,
                    (access_sha256, now),
                )
                self._append_event(
                    event_type="test_access_consumed",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=now,
                    details={"test_access_sha256": access_sha256},
                )
            else:
                access_sha256 = str(existing)
            self.connection.execute("COMMIT")
            return access_sha256
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def mark_resolution_pending(
        self,
        claim: Round17TestAccessClaim,
        *,
        pending_condition_count: int,
        observed_at_ms: int | None = None,
    ) -> None:
        selected = claim.validated()
        count = int(pending_condition_count)
        now = (
            int(time.time_ns() // 1_000_000)
            if observed_at_ms is None
            else int(observed_at_ms)
        )
        if count < 1 or now <= 0:
            raise ValueError("Round 17 pending resolution state is invalid")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._claim_row(selected.claim_sha256)
            if row["test_access_sha256"] is None:
                raise RuntimeError("Round 17 test access was not consumed")
            status = str(row["status"])
            if status in {"completed", "failed"}:
                raise RuntimeError("Round 17 one-use claim is terminal")
            if status != "resolution_pending":
                self.connection.execute(
                    """
                    UPDATE round17_one_use_claim
                    SET status = 'resolution_pending'
                    WHERE singleton = 1
                    """
                )
                self._append_event(
                    event_type="resolution_pending",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=now,
                    details={"pending_condition_count": count},
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def complete(
        self,
        claim: Round17TestAccessClaim,
        result: Mapping[str, object],
        *,
        observed_at_ms: int | None = None,
    ) -> dict[str, object]:
        selected = claim.validated()
        payload = dict(result)
        claimed = str(payload.pop("result_sha256", "")).strip().lower()
        now = (
            int(time.time_ns() // 1_000_000)
            if observed_at_ms is None
            else int(observed_at_ms)
        )
        if (
            now <= 0
            or claimed != _canonical_sha256(payload)
            or payload.get("claim_sha256") != selected.claim_sha256
            or payload.get("test_access_consumed") is not True
            or any(
                payload.get(name) is not False
                for name in (
                    "automatic_promotion",
                    "profitability_claim",
                    "paper_trading_authority",
                    "live_trading_authority",
                    "binance_credentials_used",
                    "binance_execution_connected",
                )
            )
        ):
            raise ValueError("Round 17 one-use result differs")
        complete_payload = {**payload, "result_sha256": claimed}
        raw = _canonical_json(complete_payload)
        if len(raw.encode("ascii")) > _MAXIMUM_RESULT_BYTES:
            raise ValueError("Round 17 one-use result is too large")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._claim_row(selected.claim_sha256)
            status = str(row["status"])
            if status == "failed":
                raise RuntimeError("Round 17 one-use claim already failed")
            if row["test_access_sha256"] is None:
                raise RuntimeError("Round 17 test access was not consumed")
            if payload.get("test_access_sha256") != row["test_access_sha256"]:
                raise ValueError("Round 17 result test access differs")
            if status == "completed":
                if (
                    row["result_sha256"] != claimed
                    or row["result_json"] != raw
                ):
                    raise RuntimeError("Round 17 completed result differs")
            else:
                self.connection.execute(
                    """
                    UPDATE round17_one_use_claim
                    SET status = 'completed', result_sha256 = ?, result_json = ?
                    WHERE singleton = 1
                    """,
                    (claimed, raw),
                )
                self._append_event(
                    event_type="evaluation_completed",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=now,
                    details={"result_sha256": claimed},
                )
            self.connection.execute("COMMIT")
            return complete_payload
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def fail(
        self,
        claim: Round17TestAccessClaim,
        *,
        reason: str,
        observed_at_ms: int | None = None,
    ) -> None:
        selected = claim.validated()
        selected_reason = str(reason or "").strip()
        now = (
            int(time.time_ns() // 1_000_000)
            if observed_at_ms is None
            else int(observed_at_ms)
        )
        if not selected_reason or len(selected_reason) > 500 or now <= 0:
            raise ValueError("Round 17 failure reason is invalid")
        failure = {
            "schema_version": "polymarket-round17-one-use-failure-v1",
            "claim_sha256": selected.claim_sha256,
            "failed_at_ms": now,
            "reason": selected_reason,
            "return_to_development": False,
        }
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._claim_row(selected.claim_sha256)
            status = str(row["status"])
            if status == "completed":
                raise RuntimeError("Round 17 one-use claim already completed")
            if status != "failed":
                self.connection.execute(
                    """
                    UPDATE round17_one_use_claim
                    SET status = 'failed', failure_json = ?
                    WHERE singleton = 1
                    """,
                    (_canonical_json(failure),),
                )
                self._append_event(
                    event_type="evaluation_failed",
                    claim_sha256=selected.claim_sha256,
                    observed_at_ms=now,
                    details={"failure_sha256": _canonical_sha256(failure)},
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def snapshot(self) -> dict[str, object]:
        row = self.connection.execute(
            "SELECT * FROM round17_one_use_claim WHERE singleton = 1"
        ).fetchone()
        if row is None:
            return {
                "schema_version": POLYMARKET_ROUND17_ONE_USE_STORE_SCHEMA_VERSION,
                "claim": None,
                "status": "empty",
                "test_access_consumed": False,
                "event_count": 0,
            }
        claim = _claim_from_mapping(
            _strict_mapping_json(
                str(row["claim_json"]),
                name="Round 17 stored claim",
            )
        )
        status = str(row["status"])
        if (
            status not in _CLAIM_STATUSES
            or row["claim_sha256"] != claim.claim_sha256
            or (
                status == "open"
                and (
                    row["test_access_sha256"] is not None
                    or row["access_started_ms"] is not None
                )
            )
            or (
                status in {"access_open", "resolution_pending", "completed"}
                and row["test_access_sha256"] is None
            )
            or (
                status == "completed"
                and (
                    row["result_sha256"] is None
                    or row["result_json"] is None
                    or row["failure_json"] is not None
                )
            )
            or (
                status == "failed"
                and (
                    row["failure_json"] is None
                    or row["result_sha256"] is not None
                    or row["result_json"] is not None
                )
            )
        ):
            raise ValueError("Round 17 one-use status differs")
        events = self.connection.execute(
            """
            SELECT event_sha256, previous_event_sha256, event_json
            FROM round17_one_use_event ORDER BY sequence
            """
        ).fetchall()
        previous = ""
        for event in events:
            payload = _strict_mapping_json(
                str(event["event_json"]),
                name="Round 17 stored event",
            )
            if (
                event["previous_event_sha256"] != previous
                or payload.get("previous_event_sha256") != previous
                or event["event_sha256"] != _canonical_sha256(payload)
                or payload.get("claim_sha256") != claim.claim_sha256
            ):
                raise ValueError("Round 17 one-use event chain differs")
            previous = str(event["event_sha256"])
        result = (
            None
            if row["result_json"] is None
            else _strict_mapping_json(
                str(row["result_json"]),
                name="Round 17 stored result",
            )
        )
        failure = (
            None
            if row["failure_json"] is None
            else _strict_mapping_json(
                str(row["failure_json"]),
                name="Round 17 stored failure",
            )
        )
        if result is not None:
            result_body = dict(result)
            result_claimed = str(result_body.pop("result_sha256", "")).lower()
            if (
                result_claimed != row["result_sha256"]
                or result_claimed != _canonical_sha256(result_body)
                or result.get("claim_sha256") != claim.claim_sha256
                or result.get("test_access_sha256") != row["test_access_sha256"]
            ):
                raise ValueError("Round 17 stored result differs")
        if failure is not None and (
            failure.get("schema_version")
            != "polymarket-round17-one-use-failure-v1"
            or failure.get("claim_sha256") != claim.claim_sha256
            or failure.get("return_to_development") is not False
            or not str(failure.get("reason") or "").strip()
        ):
            raise ValueError("Round 17 stored failure differs")
        return {
            "schema_version": POLYMARKET_ROUND17_ONE_USE_STORE_SCHEMA_VERSION,
            "claim": claim.asdict(),
            "status": status,
            "test_access_sha256": row["test_access_sha256"],
            "test_access_consumed": row["test_access_sha256"] is not None,
            "result": result,
            "failure": failure,
            "event_count": len(events),
            "event_chain_head_sha256": previous,
        }


def stage_round17_one_use_claim(
    *,
    store_path: str | Path,
    repository: str | Path,
    repository_commit_sha: str,
    contract_path: str | Path,
    development_result: Mapping[str, object],
    campaign: Round17CampaignOperatorConfig,
    observed_at_ms: int | None = None,
) -> Round17TestAccessClaim:
    """Persist the immutable claim without opening test data or the capture DB."""

    load_round17_one_use_contract(contract_path)
    result = validate_round17_development_result(development_result)
    if result["status"] != "development_accepted":
        raise RuntimeError("Round 17 development result was not accepted")
    readiness = inspect_round17_campaign_readiness(campaign)
    if not readiness.ready:
        raise RuntimeError("Round 17 campaign is not terminal")
    artifacts = result["artifacts"]
    parents = result["parents"]
    if not isinstance(artifacts, Mapping) or not isinstance(parents, Mapping):
        raise ValueError("Round 17 development result parents differ")
    index = artifacts.get("development_index")
    if (
        not isinstance(index, Mapping)
        or index.get("readiness_sha256") != readiness.readiness_sha256
    ):
        raise ValueError("Round 17 claim campaign readiness differs")
    implementation = build_round17_implementation_manifest(
        repository,
        repository_commit_sha=repository_commit_sha,
    )
    now = (
        int(time.time_ns() // 1_000_000)
        if observed_at_ms is None
        else int(observed_at_ms)
    )
    provisional = Round17TestAccessClaim(
        development_result_sha256=str(result["result_sha256"]),
        campaign_readiness_sha256=readiness.readiness_sha256,
        campaign_development_index_sha256=str(
            parents["campaign_development_index_sha256"]
        ),
        cohort_manifest_sha256=str(parents["cohort_manifest_sha256"]),
        target_manifest_sha256=str(parents["target_manifest_sha256"]),
        model_pretest_sha256=str(parents["model_pretest_sha256"]),
        probability_calibration_sha256=str(
            parents["probability_calibration_sha256"]
        ),
        economic_pretest_sha256=str(parents["economic_pretest_sha256"]),
        implementation_manifest_sha256=implementation.manifest_sha256,
        repository_commit_sha=implementation.repository_commit_sha,
        opened_at_ms=now,
    )
    candidate = replace(
        provisional,
        claim_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()
    store = Path(store_path).resolve()
    database = campaign.database_path.resolve()
    state_root = campaign.state_root.resolve()
    if (
        store == database
        or store == state_root
        or state_root in store.parents
        or store in state_root.parents
    ):
        raise ValueError("Round 17 one-use store overlaps capture state")
    with Round17OneUseClaimStore(store) as claim_store:
        return claim_store.open_claim(candidate)


__all__ = [
    "POLYMARKET_ROUND17_IMPLEMENTATION_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_ONE_USE_CLAIM_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256",
    "POLYMARKET_ROUND17_ONE_USE_STORE_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_TEST_ACCESS_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_TEST_END_MS_EXCLUSIVE",
    "POLYMARKET_ROUND17_TEST_FIRST_SLOT",
    "POLYMARKET_ROUND17_TEST_LAST_SLOT",
    "POLYMARKET_ROUND17_TEST_START_MS",
    "Round17ImplementationFile",
    "Round17ImplementationManifest",
    "Round17OneUseClaimStore",
    "Round17TestAccessClaim",
    "build_round17_implementation_manifest",
    "load_round17_one_use_contract",
    "stage_round17_one_use_claim",
]
