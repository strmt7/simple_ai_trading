"""Crash-resumable optional Binance predictor campaign for Polymarket Round 21."""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import BinaryIO, Callable, Mapping

from .polymarket_recorder import (
    POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS,
    PolymarketEvidenceStore,
)
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_sidecar import (
    POLYMARKET_ROUND21_CAMPAIGN_PLAN_SHA256,
    POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES,
    POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256,
    POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES,
    POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
    create_round21_sidecar_recorder,
    validate_round21_sidecar_manifest,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-campaign-design-v2"
)
POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256 = (
    "f29f272c7e1d072f5723b029d8c867c9ceb26b9a70eef3603e81360faae5c27c"
)
POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_PLAN_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-campaign-plan-v2"
)
POLYMARKET_ROUND21_SIDECAR_SEGMENT_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-segment-manifest-v2"
)
POLYMARKET_ROUND21_SIDECAR_SEGMENT_RESULT_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-segment-result-v2"
)
POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_STATE_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-campaign-state-v2"
)
POLYMARKET_ROUND21_SIDECAR_MAXIMUM_CONSECUTIVE_FAILURES = 3
POLYMARKET_ROUND21_SIDECAR_FAILURE_BACKOFF_SECONDS = 30
_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-021-binance-sidecar-campaign-design-v2.json"
)
_REQUIRED_FILES = (
    _DESIGN_RELATIVE,
    "docs/model-research/polymarket/"
    "round-021-binance-sidecar-capture-design-v1.json",
    "docs/model-research/polymarket/"
    "round-021-independent-matched-edge-contract-v1.json",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_round21_contract.py",
    "src/simple_ai_trading/polymarket_round21_sidecar.py",
    "src/simple_ai_trading/polymarket_round21_sidecar_campaign.py",
    "tests/test_polymarket_round21_sidecar_campaign.py",
    "tools/run_polymarket_round21_sidecar_campaign.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_MAXIMUM_JSON_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 21 sidecar campaign JSON has duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 sidecar campaign JSON contains {value}")


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
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_strict_json(path: Path, *, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unavailable")
    size = path.stat().st_size
    if not 2 <= size <= _MAXIMUM_JSON_BYTES:
        raise ValueError(f"{label} size is invalid")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an object")
    return value


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode:
        raise ValueError(
            "Round 21 sidecar campaign Git operation failed: "
            + (completed.stderr.strip() or completed.stdout.strip())[:500]
        )
    return completed.stdout.strip()


def _repository_attestation(
    repository: Path,
) -> tuple[str, str, dict[str, str]]:
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Round 21 sidecar campaign requires a clean worktree")
    commit = _git(repository, "rev-parse", "HEAD").lower()
    tree = _git(repository, "rev-parse", "HEAD^{tree}").lower()
    files: dict[str, str] = {}
    for relative in _REQUIRED_FILES:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"Round 21 sidecar campaign file is unavailable: {relative}"
            )
        files[relative] = _file_sha256(path)
    return commit, tree, files


def _validate_design(repository: Path) -> None:
    payload = dict(
        _read_strict_json(
            repository / _DESIGN_RELATIVE,
            label="Round 21 sidecar campaign design",
        )
    )
    claimed = str(payload.pop("design_sha256", "")).strip().lower()
    if (
        claimed != POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SCHEMA_VERSION
        or payload.get("round") != 21
        or payload.get("status")
        != "preregistered_after_host_reboot_before_target_or_model_access"
    ):
        raise ValueError("Round 21 sidecar campaign design differs")


def validate_round21_legacy_sidecar_state(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the final target-blind state written by the interrupted v1 run."""

    state = dict(value)
    claimed = str(state.pop("artifact_sha256", "")).strip().lower()
    details = state.get("details")
    expected_keys = {
        "schema_version",
        "phase",
        "observed_at_ms",
        "round21_contract_sha256",
        "sidecar_design_sha256",
        "database_bytes",
        "wal_bytes",
        "free_bytes",
        "details",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    if (
        set(state) != expected_keys
        or claimed != _canonical_sha256(state)
        or _SHA256.fullmatch(claimed) is None
        or state.get("schema_version")
        != "polymarket-round21-binance-sidecar-state-v1"
        or state.get("phase") not in {"capture-started", "capturing"}
        or state.get("round21_contract_sha256")
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or state.get("sidecar_design_sha256")
        != POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256
        or type(state.get("observed_at_ms")) is not int
        or int(state["observed_at_ms"]) <= 0
        or any(
            type(state.get(name)) is not int or int(state[name]) < 0
            for name in ("database_bytes", "wal_bytes", "free_bytes")
        )
        or not isinstance(details, Mapping)
        or _RUN_ID.fullmatch(str(details.get("run_id") or "")) is None
        or details.get("phase") != state.get("phase")
        or type(details.get("duration_seconds")) is not int
        or not 5 <= int(details["duration_seconds"]) <= 30 * 86_400
        or type(details.get("error_count")) is not int
        or int(details["error_count"]) < 0
        or any(
            state.get(name) is not False
            for name in (
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 21 legacy sidecar state differs")
    return {**state, "artifact_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round21SidecarCampaignPlan:
    created_at_ms: int
    scheduled_start_ms: int
    scheduled_end_ms: int
    legacy_run_id: str
    legacy_state_observed_at_ms: int
    legacy_state_artifact_sha256: str
    repository_commit_oid: str
    repository_tree_oid: str
    repository_file_sha256: Mapping[str, str]
    plan_sha256: str


def create_round21_sidecar_campaign_plan(
    *,
    created_at_ms: int,
    scheduled_start_ms: int,
    legacy_state: Mapping[str, object],
    repository_commit_oid: str,
    repository_tree_oid: str,
    repository_file_sha256: Mapping[str, str],
) -> dict[str, object]:
    legacy = validate_round21_legacy_sidecar_state(legacy_state)
    details = legacy["details"]
    if not isinstance(details, Mapping):
        raise AssertionError("validated legacy sidecar details are unavailable")
    payload: dict[str, object] = {
        "schema_version": (
            POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_PLAN_SCHEMA_VERSION
        ),
        "created_at_ms": int(created_at_ms),
        "scheduled_start_ms": int(scheduled_start_ms),
        "scheduled_end_ms": POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
        "round21_contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
        "round20_campaign_plan_sha256": (
            POLYMARKET_ROUND21_CAMPAIGN_PLAN_SHA256
        ),
        "sidecar_v1_design_sha256": POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256,
        "sidecar_campaign_design_sha256": (
            POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256
        ),
        "legacy_run_id": str(details["run_id"]),
        "legacy_state_observed_at_ms": int(legacy["observed_at_ms"]),
        "legacy_state_artifact_sha256": legacy["artifact_sha256"],
        "legacy_interruption": "host_reboot",
        "repository_commit_oid": str(repository_commit_oid).strip().lower(),
        "repository_tree_oid": str(repository_tree_oid).strip().lower(),
        "repository_file_sha256": dict(sorted(repository_file_sha256.items())),
        "required_streams": ["binance_futures", "binance_spot"],
        "credentials_used": False,
        "account_state_accessed": False,
        "execution_connected": False,
        "outcomes_accessed": False,
        "model_scores_accessed": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    validate_round21_sidecar_campaign_plan(payload)
    return payload


def validate_round21_sidecar_campaign_plan(
    value: Mapping[str, object],
) -> Round21SidecarCampaignPlan:
    payload = dict(value)
    claimed = str(payload.pop("plan_sha256", "")).strip().lower()
    files = payload.get("repository_file_sha256")
    false_fields = (
        "credentials_used",
        "account_state_accessed",
        "execution_connected",
        "outcomes_accessed",
        "model_scores_accessed",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    expected_keys = {
        "schema_version",
        "created_at_ms",
        "scheduled_start_ms",
        "scheduled_end_ms",
        "round21_contract_sha256",
        "round20_campaign_plan_sha256",
        "sidecar_v1_design_sha256",
        "sidecar_campaign_design_sha256",
        "legacy_run_id",
        "legacy_state_observed_at_ms",
        "legacy_state_artifact_sha256",
        "legacy_interruption",
        "repository_commit_oid",
        "repository_tree_oid",
        "repository_file_sha256",
        "required_streams",
        *false_fields,
    }
    created = payload.get("created_at_ms")
    start = payload.get("scheduled_start_ms")
    end = payload.get("scheduled_end_ms")
    observed = payload.get("legacy_state_observed_at_ms")
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_PLAN_SCHEMA_VERSION
        or type(created) is not int
        or type(start) is not int
        or type(end) is not int
        or type(observed) is not int
        or not 0 < start <= observed < created < end
        or end != POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS
        or payload.get("round21_contract_sha256")
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or payload.get("round20_campaign_plan_sha256")
        != POLYMARKET_ROUND21_CAMPAIGN_PLAN_SHA256
        or payload.get("sidecar_v1_design_sha256")
        != POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256
        or payload.get("sidecar_campaign_design_sha256")
        != POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256
        or _RUN_ID.fullmatch(str(payload.get("legacy_run_id") or "")) is None
        or _SHA256.fullmatch(
            str(payload.get("legacy_state_artifact_sha256") or "")
        )
        is None
        or payload.get("legacy_interruption") != "host_reboot"
        or _GIT_OID.fullmatch(str(payload.get("repository_commit_oid") or ""))
        is None
        or _GIT_OID.fullmatch(str(payload.get("repository_tree_oid") or ""))
        is None
        or not isinstance(files, Mapping)
        or set(files) != set(_REQUIRED_FILES)
        or any(_SHA256.fullmatch(str(item)) is None for item in files.values())
        or payload.get("required_streams")
        != ["binance_futures", "binance_spot"]
        or any(payload.get(name) is not False for name in false_fields)
    ):
        raise ValueError("Round 21 sidecar campaign plan differs")
    return Round21SidecarCampaignPlan(
        created_at_ms=created,
        scheduled_start_ms=start,
        scheduled_end_ms=end,
        legacy_run_id=str(payload["legacy_run_id"]),
        legacy_state_observed_at_ms=observed,
        legacy_state_artifact_sha256=str(
            payload["legacy_state_artifact_sha256"]
        ),
        repository_commit_oid=str(payload["repository_commit_oid"]),
        repository_tree_oid=str(payload["repository_tree_oid"]),
        repository_file_sha256=dict(files),
        plan_sha256=claimed,
    )


def load_round21_sidecar_campaign_plan(
    path: str | Path,
) -> Round21SidecarCampaignPlan:
    return validate_round21_sidecar_campaign_plan(
        _read_strict_json(Path(path), label="Round 21 sidecar campaign plan")
    )


def build_round21_sidecar_campaign_plan(
    repository: str | Path,
    *,
    legacy_state_path: str | Path,
    scheduled_start_ms: int,
) -> dict[str, object]:
    root = Path(repository).resolve()
    _validate_design(root)
    legacy = _read_strict_json(
        Path(legacy_state_path),
        label="Round 21 legacy sidecar state",
    )
    commit, tree, files = _repository_attestation(root)
    return create_round21_sidecar_campaign_plan(
        created_at_ms=time.time_ns() // 1_000_000,
        scheduled_start_ms=scheduled_start_ms,
        legacy_state=legacy,
        repository_commit_oid=commit,
        repository_tree_oid=tree,
        repository_file_sha256=files,
    )


def write_round21_sidecar_campaign_plan(
    path: str | Path,
    value: Mapping[str, object],
) -> None:
    validate_round21_sidecar_campaign_plan(value)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(Path(path), encoded)


def build_round21_sidecar_segment_manifest(
    plan: Round21SidecarCampaignPlan,
    *,
    run_id: str,
    created_at_ms: int,
    duration_seconds: int,
    segment_index: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": (
            POLYMARKET_ROUND21_SIDECAR_SEGMENT_MANIFEST_SCHEMA_VERSION
        ),
        "plan_sha256": plan.plan_sha256,
        "round21_contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
        "sidecar_campaign_design_sha256": (
            POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256
        ),
        "run_id": str(run_id).strip().lower(),
        "created_at_ms": int(created_at_ms),
        "capture_duration_seconds": int(duration_seconds),
        "segment_index": int(segment_index),
        "scheduled_campaign_start_ms": plan.scheduled_start_ms,
        "scheduled_campaign_end_ms": plan.scheduled_end_ms,
        "purpose": "round21_optional_predictor_sidecar_segment",
        "required_assets": [],
        "required_streams": ["binance_futures", "binance_spot"],
        "spot_streams": ["btcusdt@bookTicker", "btcusdt@trade"],
        "usdm_streams": ["btcusdt@bookTicker", "btcusdt@trade"],
        "repository_commit_oid": plan.repository_commit_oid,
        "repository_tree_oid": plan.repository_tree_oid,
        "cross_segment_state_carry": False,
        "outcomes_accessed": False,
        "model_scores_accessed": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    validate_round21_sidecar_segment_manifest(payload, plan)
    return payload


def validate_round21_sidecar_segment_manifest(
    value: Mapping[str, object],
    plan: Round21SidecarCampaignPlan,
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    created = payload.get("created_at_ms")
    duration = payload.get("capture_duration_seconds")
    segment = payload.get("segment_index")
    false_fields = (
        "cross_segment_state_carry",
        "outcomes_accessed",
        "model_scores_accessed",
        "binance_credentials_used",
        "binance_execution_connected",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    expected_keys = {
        "schema_version",
        "plan_sha256",
        "round21_contract_sha256",
        "sidecar_campaign_design_sha256",
        "run_id",
        "created_at_ms",
        "capture_duration_seconds",
        "segment_index",
        "scheduled_campaign_start_ms",
        "scheduled_campaign_end_ms",
        "purpose",
        "required_assets",
        "required_streams",
        "spot_streams",
        "usdm_streams",
        "repository_commit_oid",
        "repository_tree_oid",
        *false_fields,
    }
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_SIDECAR_SEGMENT_MANIFEST_SCHEMA_VERSION
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("round21_contract_sha256")
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or payload.get("sidecar_campaign_design_sha256")
        != POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256
        or _RUN_ID.fullmatch(str(payload.get("run_id") or "")) is None
        or type(created) is not int
        or not plan.created_at_ms <= created < plan.scheduled_end_ms
        or type(duration) is not int
        or not 5 <= duration <= POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS
        or created + duration * 1_000 > plan.scheduled_end_ms + 999
        or type(segment) is not int
        or segment < 1
        or payload.get("scheduled_campaign_start_ms") != plan.scheduled_start_ms
        or payload.get("scheduled_campaign_end_ms") != plan.scheduled_end_ms
        or payload.get("purpose")
        != "round21_optional_predictor_sidecar_segment"
        or payload.get("required_assets") != []
        or payload.get("required_streams")
        != ["binance_futures", "binance_spot"]
        or payload.get("spot_streams")
        != ["btcusdt@bookTicker", "btcusdt@trade"]
        or payload.get("usdm_streams")
        != ["btcusdt@bookTicker", "btcusdt@trade"]
        or payload.get("repository_commit_oid") != plan.repository_commit_oid
        or payload.get("repository_tree_oid") != plan.repository_tree_oid
        or any(payload.get(name) is not False for name in false_fields)
    ):
        raise ValueError("Round 21 sidecar segment manifest differs")
    return {**payload, "manifest_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round21SidecarCampaignConfig:
    repository: Path
    plan_path: Path
    database_path: Path
    state_root: Path

    def validate(self) -> None:
        root = self.repository.resolve()
        plan = self.plan_path.resolve()
        database = self.database_path.resolve()
        state = self.state_root.resolve()
        if (
            not root.is_dir()
            or not plan.is_file()
            or root not in plan.parents
            or database == plan
            or state == plan
            or database == state
            or database in state.parents
            or state in database.parents
        ):
            raise ValueError("Round 21 sidecar campaign configuration differs")


class _CampaignFileLock(AbstractContextManager["_CampaignFileLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: BinaryIO | None = None

    def __enter__(self) -> _CampaignFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(
                "Round 21 sidecar campaign is already running"
            ) from exc
        self.handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        handle = self.handle
        if handle is None:
            return
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self.handle = None


def _write_hashed_json(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    body = dict(payload)
    body["artifact_sha256"] = _canonical_sha256(body)
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(path, encoded)
    return body


def _segment_result_path(state_root: Path, segment_index: int) -> Path:
    return state_root / "segments" / f"segment-{segment_index:04d}.json"


def _write_segment_result(
    state_root: Path,
    *,
    plan: Round21SidecarCampaignPlan,
    segment_index: int,
    status: str,
    details: Mapping[str, object],
) -> dict[str, object]:
    if status not in {"complete", "degraded", "failed", "interrupted"}:
        raise ValueError("Round 21 sidecar segment status differs")
    payload: dict[str, object] = {
        "schema_version": (
            POLYMARKET_ROUND21_SIDECAR_SEGMENT_RESULT_SCHEMA_VERSION
        ),
        "plan_sha256": plan.plan_sha256,
        "segment_index": int(segment_index),
        "status": status,
        "observed_at_ms": time.time_ns() // 1_000_000,
        "optional_feature_admission_pending": status in {"complete", "degraded"},
        "details": dict(details),
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    path = _segment_result_path(state_root, segment_index)
    if path.exists():
        raise FileExistsError(
            f"Round 21 sidecar segment result already exists: {path}"
        )
    return _write_hashed_json(path, payload)


def _segment_results(
    state_root: Path,
    plan: Round21SidecarCampaignPlan,
) -> tuple[dict[str, object], ...]:
    root = state_root / "segments"
    if not root.exists():
        return ()
    paths = tuple(sorted(root.glob("segment-*.json")))
    results: list[dict[str, object]] = []
    for expected_index, path in enumerate(paths):
        value = dict(_read_strict_json(path, label="Round 21 sidecar segment result"))
        claimed = str(value.pop("artifact_sha256", "")).strip().lower()
        status = value.get("status")
        if (
            set(value)
            != {
                "schema_version",
                "plan_sha256",
                "segment_index",
                "status",
                "observed_at_ms",
                "optional_feature_admission_pending",
                "details",
                "model_data_eligible",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            }
            or path != _segment_result_path(state_root, expected_index)
            or claimed != _canonical_sha256(value)
            or _SHA256.fullmatch(claimed) is None
            or value.get("schema_version")
            != POLYMARKET_ROUND21_SIDECAR_SEGMENT_RESULT_SCHEMA_VERSION
            or value.get("plan_sha256") != plan.plan_sha256
            or value.get("segment_index") != expected_index
            or status not in {"complete", "degraded", "failed", "interrupted"}
            or type(value.get("observed_at_ms")) is not int
            or int(value["observed_at_ms"]) <= 0
            or not isinstance(value.get("details"), Mapping)
            or value.get("optional_feature_admission_pending")
            is not (status in {"complete", "degraded"})
            or any(
                value.get(name) is not False
                for name in (
                    "model_data_eligible",
                    "profitability_claim",
                    "paper_trading_authority",
                    "live_trading_authority",
                )
            )
        ):
            raise ValueError("Round 21 sidecar segment result set differs")
        results.append({**value, "artifact_sha256": claimed})
    return tuple(results)


def _resource_block(config: Round21SidecarCampaignConfig) -> str | None:
    database_bytes = (
        config.database_path.stat().st_size if config.database_path.is_file() else 0
    )
    wal = Path(f"{config.database_path}.wal")
    database_bytes += wal.stat().st_size if wal.is_file() else 0
    if database_bytes >= POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES:
        return "database_and_wal_cap_exceeded"
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(config.database_path.parent).free < (
        POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES
    ):
        return "minimum_free_space_crossed"
    return None


def _verify_repository(
    repository: Path,
    plan: Round21SidecarCampaignPlan,
) -> None:
    commit, tree, files = _repository_attestation(repository)
    if (
        not hmac.compare_digest(commit, plan.repository_commit_oid)
        or not hmac.compare_digest(tree, plan.repository_tree_oid)
        or files != dict(plan.repository_file_sha256)
    ):
        raise ValueError("Round 21 sidecar campaign repository attestation differs")


def _load_and_verify(
    config: Round21SidecarCampaignConfig,
) -> Round21SidecarCampaignPlan:
    config.validate()
    plan = load_round21_sidecar_campaign_plan(config.plan_path)
    _validate_design(config.repository.resolve())
    _verify_repository(config.repository.resolve(), plan)
    return plan


def _recover_orphaned_segments(
    config: Round21SidecarCampaignConfig,
    plan: Round21SidecarCampaignPlan,
    *,
    first_segment_index: int,
) -> int:
    if not config.database_path.is_file():
        raise ValueError("Round 21 legacy sidecar database is unavailable")
    recovered = 0
    now_ms = time.time_ns() // 1_000_000
    with PolymarketEvidenceStore(
        config.database_path,
        memory_limit="512MB",
        threads=1,
    ) as store:
        rows = (
            store.connect()
            .execute(
                """
                SELECT r.run_id, r.started_at_ms, m.manifest_json
                FROM polymarket_recorder_run AS r
                JOIN polymarket_preregistration_manifest AS m USING (run_id)
                WHERE r.status = 'running'
                ORDER BY r.started_at_ms, r.run_id
                """
            )
            .fetchall()
        )
        if first_segment_index == 0 and len(rows) != 1:
            raise ValueError("Round 21 legacy sidecar orphan was not found")
        if first_segment_index > 0 and len(rows) > 1:
            raise ValueError("Round 21 sidecar has multiple active recorder runs")
        validated_rows: list[tuple[str, int]] = []
        for offset, (run_id, started_at_ms, manifest_json) in enumerate(rows):
            expected_index = first_segment_index + offset
            manifest = json.loads(
                str(manifest_json),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
            if not isinstance(manifest, Mapping):
                raise ValueError("Round 21 orphan sidecar manifest differs")
            if expected_index == 0:
                validated = validate_round21_sidecar_manifest(manifest)
                if (
                    str(run_id) != plan.legacy_run_id
                    or validated["run_id"] != plan.legacy_run_id
                ):
                    raise ValueError("Round 21 legacy sidecar run differs")
            else:
                validated = validate_round21_sidecar_segment_manifest(
                    manifest,
                    plan,
                )
                if validated["segment_index"] != expected_index:
                    raise ValueError("Round 21 orphan sidecar segment differs")
            validated_rows.append((str(run_id), int(started_at_ms)))
        for offset, (run_id, started_at_ms) in enumerate(validated_rows):
            expected_index = first_segment_index + offset
            report = store.fail_run(
                run_id,
                started_at_ms=started_at_ms,
                ended_at_ms=max(started_at_ms, now_ms),
                database=str(config.database_path),
                errors=("host_or_process_restart_interrupted_segment",),
            )
            _write_segment_result(
                config.state_root,
                plan=plan,
                segment_index=expected_index,
                status="interrupted",
                details={
                    "run_id": report.run_id,
                    "report_sha256": report.report_sha256,
                    "started_at_ms": report.started_at_ms,
                    "ended_at_ms": report.ended_at_ms,
                    "raw_message_count": report.raw_message_count,
                    "stream_gap_count": report.stream_gap_count,
                    "stream_counts": dict(report.stream_counts),
                    "integrity_errors": list(report.integrity_errors),
                    "errors": list(report.errors),
                },
            )
            recovered += 1
    return recovered


def _reconcile_failed_capture(
    config: Round21SidecarCampaignConfig,
    plan: Round21SidecarCampaignPlan,
    *,
    segment_index: int,
    error: Exception,
) -> dict[str, object]:
    recovered = _recover_orphaned_segments(
        config,
        plan,
        first_segment_index=segment_index,
    )
    if recovered == 1:
        results = _segment_results(config.state_root, plan)
        if len(results) != segment_index + 1:
            raise RuntimeError(
                "Round 21 sidecar recovered result set is not contiguous"
            )
        return dict(results[segment_index])
    if recovered != 0:
        raise RuntimeError("Round 21 sidecar recovered an impossible run count")
    return _write_segment_result(
        config.state_root,
        plan=plan,
        segment_index=segment_index,
        status="failed",
        details={
            "failure_type": type(error).__name__,
            "failure": str(error)[:2_000],
        },
    )


def inspect_round21_sidecar_campaign(
    config: Round21SidecarCampaignConfig,
    *,
    now_ms: int | None = None,
) -> dict[str, object]:
    plan = _load_and_verify(config)
    observed = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
    results = _segment_results(config.state_root, plan)
    return {
        "schema_version": (
            POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_STATE_SCHEMA_VERSION
        ),
        "plan_sha256": plan.plan_sha256,
        "observed_at_ms": observed,
        "relation": "after_campaign" if observed >= plan.scheduled_end_ms else "open",
        "terminal_segment_count": len(results),
        "resource_block": _resource_block(config),
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


async def run_round21_sidecar_campaign(
    config: Round21SidecarCampaignConfig,
    *,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    plan = _load_and_verify(config)
    config.state_root.mkdir(parents=True, exist_ok=True)

    def notify(value: Mapping[str, object]) -> None:
        if progress is None:
            return
        try:
            progress(value)
        except Exception:
            return

    with _CampaignFileLock(config.state_root / "campaign.lock"):
        results = list(_segment_results(config.state_root, plan))
        recovered = _recover_orphaned_segments(
            config,
            plan,
            first_segment_index=len(results),
        )
        results = list(_segment_results(config.state_root, plan))
        consecutive_failures = 0
        while True:
            now_ms = time.time_ns() // 1_000_000
            remaining_ms = plan.scheduled_end_ms - now_ms
            if remaining_ms < 5_000:
                break
            resource = _resource_block(config)
            if resource is not None:
                return {
                    **inspect_round21_sidecar_campaign(config, now_ms=now_ms),
                    "status": "resource_blocked",
                    "resource_block": resource,
                }
            segment_index = len(results)
            duration_seconds = min(
                POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS,
                max(5, math.floor(remaining_ms / 1_000)),
            )
            recorder = create_round21_sidecar_recorder(config.database_path)
            manifest_holder: dict[str, object] = {}

            def manifest_factory(
                run_id: str,
                created_at_ms: int,
            ) -> Mapping[str, object]:
                manifest = build_round21_sidecar_segment_manifest(
                    plan,
                    run_id=run_id,
                    created_at_ms=created_at_ms,
                    duration_seconds=duration_seconds,
                    segment_index=segment_index,
                )
                manifest_holder.update(manifest)
                return manifest

            def capture_progress(
                phase: str,
                details: Mapping[str, object],
            ) -> None:
                state = {
                    "schema_version": (
                        POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_STATE_SCHEMA_VERSION
                    ),
                    "plan_sha256": plan.plan_sha256,
                    "observed_at_ms": time.time_ns() // 1_000_000,
                    "phase": phase,
                    "segment_index": segment_index,
                    "details": dict(details),
                    "model_data_eligible": False,
                    "profitability_claim": False,
                    "paper_trading_authority": False,
                    "live_trading_authority": False,
                }
                persisted = _write_hashed_json(
                    config.state_root / "campaign-state.json",
                    state,
                )
                notify(persisted)

            try:
                report = await recorder.run(
                    duration_seconds=duration_seconds,
                    progress=capture_progress,
                    progress_interval_seconds=30,
                    stop_requested=lambda: _resource_block(config),
                    preregistration_manifest_factory=manifest_factory,
                )
                if not manifest_holder:
                    raise RuntimeError(
                        "Round 21 sidecar segment manifest was not created"
                    )
                _verify_repository(config.repository.resolve(), plan)
                status = (
                    report.status
                    if report.status in {"complete", "degraded"}
                    else "failed"
                )
                result = _write_segment_result(
                    config.state_root,
                    plan=plan,
                    segment_index=segment_index,
                    status=status,
                    details={
                        "run_id": report.run_id,
                        "manifest_sha256": manifest_holder["manifest_sha256"],
                        "report_sha256": report.report_sha256,
                        "started_at_ms": report.started_at_ms,
                        "ended_at_ms": report.ended_at_ms,
                        "duration_seconds": report.duration_seconds,
                        "raw_message_count": report.raw_message_count,
                        "stream_gap_count": report.stream_gap_count,
                        "stream_counts": dict(report.stream_counts),
                        "integrity_errors": list(report.integrity_errors),
                        "errors": list(report.errors),
                    },
                )
            except Exception as exc:
                result = _reconcile_failed_capture(
                    config,
                    plan,
                    segment_index=segment_index,
                    error=exc,
                )
            results.append(result)
            notify(result)
            if result["status"] in {"complete", "degraded"}:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= (
                    POLYMARKET_ROUND21_SIDECAR_MAXIMUM_CONSECUTIVE_FAILURES
                ):
                    break
                await asyncio.sleep(
                    POLYMARKET_ROUND21_SIDECAR_FAILURE_BACKOFF_SECONDS
                )
        status_counts: dict[str, int] = {}
        for result in results:
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        terminal = {
            "schema_version": (
                POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_STATE_SCHEMA_VERSION
            ),
            "plan_sha256": plan.plan_sha256,
            "status": (
                "campaign_window_ended"
                if time.time_ns() // 1_000_000 >= plan.scheduled_end_ms
                else "campaign_failed"
            ),
            "terminal_segment_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "recovered_interrupted_segment_count": recovered,
            "optional_feature_admission_pending": True,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        notify(terminal)
        return terminal


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256",
    "POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_PLAN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SIDECAR_SEGMENT_MANIFEST_SCHEMA_VERSION",
    "Round21SidecarCampaignConfig",
    "Round21SidecarCampaignPlan",
    "build_round21_sidecar_campaign_plan",
    "build_round21_sidecar_segment_manifest",
    "create_round21_sidecar_campaign_plan",
    "inspect_round21_sidecar_campaign",
    "load_round21_sidecar_campaign_plan",
    "run_round21_sidecar_campaign",
    "validate_round21_legacy_sidecar_state",
    "validate_round21_sidecar_campaign_plan",
    "validate_round21_sidecar_segment_manifest",
    "write_round21_sidecar_campaign_plan",
]
