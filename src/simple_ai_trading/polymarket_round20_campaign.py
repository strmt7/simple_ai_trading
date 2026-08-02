"""Crash-resumable continuous corpus campaign for Polymarket Round 20."""

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
from .polymarket_round20_capture import (
    POLYMARKET_ROUND20_QUALIFICATION_SCHEMA_VERSION,
    create_round20_recorder,
    validate_round20_qualification,
)
from .polymarket_round20_contract import (
    POLYMARKET_ROUND20_CONTRACT_SHA256,
    load_round20_contract,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SCHEMA_VERSION = (
    "polymarket-round20-continuous-campaign-design-v1"
)
POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256 = (
    "1616da48a7a63477e03328f6a1c032b611981a58f2d0c801f06f95462471757f"
)
POLYMARKET_ROUND20_CAMPAIGN_PLAN_SCHEMA_VERSION = (
    "polymarket-round20-continuous-campaign-plan-v1"
)
POLYMARKET_ROUND20_SEGMENT_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round20-continuous-segment-manifest-v1"
)
POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION = (
    "polymarket-round20-continuous-segment-result-v1"
)
POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION = (
    "polymarket-round20-continuous-campaign-state-v1"
)
POLYMARKET_ROUND20_QUALIFICATION_RESULT_SHA256 = (
    "5260a5b6c11e8acfb1343d25c593a1af21d3a239e6cf81d1430296f4a63ee05d"
)
POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS = 1_200
POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT = 2_160
POLYMARKET_ROUND20_CAMPAIGN_SECONDS = (
    POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS * POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT
)
POLYMARKET_ROUND20_DATABASE_CAP_BYTES = 384 * 1024**3
POLYMARKET_ROUND20_MINIMUM_FREE_BYTES = 256 * 1024**3
POLYMARKET_ROUND20_FAILURE_BACKOFF_SECONDS = 30
POLYMARKET_ROUND20_MAXIMUM_CONSECUTIVE_FAILURES = 3
_CONTRACT_RELATIVE = (
    "docs/model-research/polymarket/"
    "round-020-independent-redundant-corpus-contract-v1.json"
)
_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/round-020-continuous-campaign-design-v1.json"
)
_QUALIFICATION_RELATIVE = (
    "docs/model-research/polymarket/evidence/round-020-capture-qualification-v1.json"
)
_REQUIRED_FILES = (
    _CONTRACT_RELATIVE,
    _DESIGN_RELATIVE,
    _QUALIFICATION_RELATIVE,
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_redundant_union.py",
    "src/simple_ai_trading/polymarket_round20_campaign.py",
    "src/simple_ai_trading/polymarket_round20_capture.py",
    "src/simple_ai_trading/polymarket_round20_contract.py",
    "tests/test_polymarket_round20_campaign.py",
    "tools/run_polymarket_round20_campaign.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")
_RUN_ID = re.compile(r"^[0-9a-f]{32}$")
_MAX_JSON_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 20 campaign JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 20 campaign JSON contains {value}")


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
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_strict_json(path: Path, *, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is unavailable")
    size = path.stat().st_size
    if not 2 <= size <= _MAX_JSON_BYTES:
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
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return completed.stdout.strip()


def _repository_attestation(
    repository: Path,
) -> tuple[str, str, dict[str, str]]:
    if _git(repository, "status", "--porcelain=v1"):
        raise ValueError("Round 20 campaign requires a clean worktree")
    commit_oid = _git(repository, "rev-parse", "HEAD").lower()
    tree_oid = _git(repository, "rev-parse", "HEAD^{tree}").lower()
    files: dict[str, str] = {}
    for relative in _REQUIRED_FILES:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Round 20 campaign file is unavailable: {relative}")
        files[relative] = _file_sha256(path)
    return commit_oid, tree_oid, files


def _validate_campaign_design(repository: Path) -> None:
    path = repository / _DESIGN_RELATIVE
    payload = dict(_read_strict_json(path, label="Round 20 campaign design"))
    claimed = str(payload.pop("design_sha256", "")).strip().lower()
    if (
        claimed != POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("schema_version")
        != POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SCHEMA_VERSION
        or payload.get("round") != 20
    ):
        raise ValueError("Round 20 campaign design differs")


@dataclass(frozen=True, slots=True)
class PolymarketRound20CampaignPlan:
    created_at_ms: int
    scheduled_start_ms: int
    scheduled_end_ms: int
    repository_commit_oid: str
    repository_tree_oid: str
    repository_file_sha256: Mapping[str, str]
    plan_sha256: str

    def logical_unit_index(self, observed_at_ms: int) -> int | None:
        observed = int(observed_at_ms)
        if observed < self.scheduled_start_ms or observed >= self.scheduled_end_ms:
            return None
        return (observed - self.scheduled_start_ms) // (
            POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS * 1_000
        )


def create_round20_campaign_plan(
    *,
    created_at_ms: int,
    scheduled_start_ms: int,
    repository_commit_oid: str,
    repository_tree_oid: str,
    repository_file_sha256: Mapping[str, str],
) -> dict[str, object]:
    created = int(created_at_ms)
    start = int(scheduled_start_ms)
    files = dict(sorted(repository_file_sha256.items()))
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND20_CAMPAIGN_PLAN_SCHEMA_VERSION,
        "created_at_ms": created,
        "scheduled_start_ms": start,
        "scheduled_end_ms": start + POLYMARKET_ROUND20_CAMPAIGN_SECONDS * 1_000,
        "logical_unit_seconds": POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS,
        "logical_unit_count": POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT,
        "round20_contract_sha256": POLYMARKET_ROUND20_CONTRACT_SHA256,
        "campaign_design_sha256": POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256,
        "qualification_result_sha256": (POLYMARKET_ROUND20_QUALIFICATION_RESULT_SHA256),
        "required_assets": ["BTC"],
        "required_streams": ["clob_market", "polymarket_rtds"],
        "required_clob_lanes": ["clob-a", "clob-b"],
        "required_rtds_topics": ["crypto_prices_chainlink"],
        "planned_transport_restarts": False,
        "maximum_concurrent_capture_segments": 1,
        "database_cap_bytes": POLYMARKET_ROUND20_DATABASE_CAP_BYTES,
        "minimum_free_bytes": POLYMARKET_ROUND20_MINIMUM_FREE_BYTES,
        "repository_commit_oid": str(repository_commit_oid).strip().lower(),
        "repository_tree_oid": str(repository_tree_oid).strip().lower(),
        "repository_file_sha256": files,
        "binance_required": False,
        "binance_captured": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    validate_round20_campaign_plan(payload)
    return payload


def validate_round20_campaign_plan(
    value: Mapping[str, object],
) -> PolymarketRound20CampaignPlan:
    payload = dict(value)
    claimed = str(payload.pop("plan_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "created_at_ms",
        "scheduled_start_ms",
        "scheduled_end_ms",
        "logical_unit_seconds",
        "logical_unit_count",
        "round20_contract_sha256",
        "campaign_design_sha256",
        "qualification_result_sha256",
        "required_assets",
        "required_streams",
        "required_clob_lanes",
        "required_rtds_topics",
        "planned_transport_restarts",
        "maximum_concurrent_capture_segments",
        "database_cap_bytes",
        "minimum_free_bytes",
        "repository_commit_oid",
        "repository_tree_oid",
        "repository_file_sha256",
        "binance_required",
        "binance_captured",
        "binance_credentials_used",
        "binance_execution_connected",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    created = payload.get("created_at_ms")
    start = payload.get("scheduled_start_ms")
    end = payload.get("scheduled_end_ms")
    files = payload.get("repository_file_sha256")
    false_fields = (
        "planned_transport_restarts",
        "binance_required",
        "binance_captured",
        "binance_credentials_used",
        "binance_execution_connected",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    unit_ms = POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS * 1_000
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload["schema_version"] != POLYMARKET_ROUND20_CAMPAIGN_PLAN_SCHEMA_VERSION
        or type(created) is not int
        or type(start) is not int
        or type(end) is not int
        or created <= 0
        or start <= created
        or start % unit_ms
        or end != start + POLYMARKET_ROUND20_CAMPAIGN_SECONDS * 1_000
        or payload["logical_unit_seconds"] != POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS
        or payload["logical_unit_count"] != POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT
        or payload["round20_contract_sha256"] != POLYMARKET_ROUND20_CONTRACT_SHA256
        or payload["campaign_design_sha256"]
        != POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256
        or payload["qualification_result_sha256"]
        != POLYMARKET_ROUND20_QUALIFICATION_RESULT_SHA256
        or payload["required_assets"] != ["BTC"]
        or payload["required_streams"] != ["clob_market", "polymarket_rtds"]
        or payload["required_clob_lanes"] != ["clob-a", "clob-b"]
        or payload["required_rtds_topics"] != ["crypto_prices_chainlink"]
        or payload["maximum_concurrent_capture_segments"] != 1
        or payload["database_cap_bytes"] != POLYMARKET_ROUND20_DATABASE_CAP_BYTES
        or payload["minimum_free_bytes"] != POLYMARKET_ROUND20_MINIMUM_FREE_BYTES
        or _GIT_OID.fullmatch(str(payload["repository_commit_oid"])) is None
        or _GIT_OID.fullmatch(str(payload["repository_tree_oid"])) is None
        or not isinstance(files, Mapping)
        or set(files) != set(_REQUIRED_FILES)
        or any(_SHA256.fullmatch(str(item)) is None for item in files.values())
        or any(payload[name] is not False for name in false_fields)
    ):
        raise ValueError("Round 20 campaign plan differs")
    return PolymarketRound20CampaignPlan(
        created_at_ms=created,
        scheduled_start_ms=start,
        scheduled_end_ms=end,
        repository_commit_oid=str(payload["repository_commit_oid"]),
        repository_tree_oid=str(payload["repository_tree_oid"]),
        repository_file_sha256=dict(files),
        plan_sha256=claimed,
    )


def load_round20_campaign_plan(path: str | Path) -> PolymarketRound20CampaignPlan:
    return validate_round20_campaign_plan(
        _read_strict_json(Path(path), label="Round 20 campaign plan")
    )


def build_round20_campaign_plan(
    repository: str | Path,
    *,
    scheduled_start_ms: int,
) -> dict[str, object]:
    root = Path(repository).resolve()
    _validate_campaign_design(root)
    program = load_round20_contract(root / _CONTRACT_RELATIVE)
    qualification = validate_round20_qualification(
        _read_strict_json(
            root / _QUALIFICATION_RELATIVE,
            label="Round 20 qualification",
        )
    )
    if (
        program.contract_sha256 != POLYMARKET_ROUND20_CONTRACT_SHA256
        or qualification["schema_version"]
        != POLYMARKET_ROUND20_QUALIFICATION_SCHEMA_VERSION
        or qualification["result_sha256"]
        != POLYMARKET_ROUND20_QUALIFICATION_RESULT_SHA256
        or qualification["qualified"] is not True
    ):
        raise ValueError("Round 20 campaign parent evidence differs")
    commit, tree, files = _repository_attestation(root)
    return create_round20_campaign_plan(
        created_at_ms=time.time_ns() // 1_000_000,
        scheduled_start_ms=scheduled_start_ms,
        repository_commit_oid=commit,
        repository_tree_oid=tree,
        repository_file_sha256=files,
    )


def write_round20_campaign_plan(path: str | Path, value: Mapping[str, object]) -> None:
    validate_round20_campaign_plan(value)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(Path(path), encoded)


def _verify_repository(
    repository: Path,
    plan: PolymarketRound20CampaignPlan,
) -> None:
    commit, tree, files = _repository_attestation(repository)
    if (
        not hmac.compare_digest(commit, plan.repository_commit_oid)
        or not hmac.compare_digest(tree, plan.repository_tree_oid)
        or files != dict(plan.repository_file_sha256)
    ):
        raise ValueError("Round 20 campaign repository attestation differs")


def build_round20_segment_manifest(
    plan: PolymarketRound20CampaignPlan,
    *,
    run_id: str,
    created_at_ms: int,
    duration_seconds: int,
    segment_index: int,
) -> dict[str, object]:
    created = int(created_at_ms)
    duration = int(duration_seconds)
    first_unit = plan.logical_unit_index(max(created, plan.scheduled_start_ms))
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND20_SEGMENT_MANIFEST_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "round20_contract_sha256": POLYMARKET_ROUND20_CONTRACT_SHA256,
        "campaign_design_sha256": POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256,
        "qualification_result_sha256": (POLYMARKET_ROUND20_QUALIFICATION_RESULT_SHA256),
        "run_id": str(run_id).strip().lower(),
        "created_at_ms": created,
        "capture_duration_seconds": duration,
        "segment_index": int(segment_index),
        "scheduled_campaign_start_ms": plan.scheduled_start_ms,
        "scheduled_campaign_end_ms": plan.scheduled_end_ms,
        "first_logical_unit_index": first_unit,
        "purpose": "prospective_corpus",
        "required_assets": ["BTC"],
        "required_streams": ["clob_market", "polymarket_rtds"],
        "required_clob_lanes": ["clob-a", "clob-b"],
        "required_rtds_topics": ["crypto_prices_chainlink"],
        "optional_predictor_sources_captured": [],
        "repository_commit_oid": plan.repository_commit_oid,
        "repository_tree_oid": plan.repository_tree_oid,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    validate_round20_segment_manifest(payload, plan)
    return payload


def validate_round20_segment_manifest(
    value: Mapping[str, object],
    plan: PolymarketRound20CampaignPlan,
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).strip().lower()
    false_fields = (
        "binance_credentials_used",
        "binance_execution_connected",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    created = payload.get("created_at_ms")
    duration = payload.get("capture_duration_seconds")
    segment = payload.get("segment_index")
    expected_unit = (
        None
        if type(created) is not int
        else plan.logical_unit_index(max(created, plan.scheduled_start_ms))
    )
    if (
        claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND20_SEGMENT_MANIFEST_SCHEMA_VERSION
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("round20_contract_sha256") != POLYMARKET_ROUND20_CONTRACT_SHA256
        or payload.get("campaign_design_sha256")
        != POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256
        or payload.get("qualification_result_sha256")
        != POLYMARKET_ROUND20_QUALIFICATION_RESULT_SHA256
        or _RUN_ID.fullmatch(str(payload.get("run_id"))) is None
        or type(created) is not int
        or not plan.created_at_ms <= created < plan.scheduled_end_ms
        or type(duration) is not int
        or not 5 <= duration <= POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS
        or type(segment) is not int
        or segment < 0
        or payload.get("scheduled_campaign_start_ms") != plan.scheduled_start_ms
        or payload.get("scheduled_campaign_end_ms") != plan.scheduled_end_ms
        or payload.get("first_logical_unit_index") != expected_unit
        or payload.get("purpose") != "prospective_corpus"
        or payload.get("required_assets") != ["BTC"]
        or payload.get("required_streams") != ["clob_market", "polymarket_rtds"]
        or payload.get("required_clob_lanes") != ["clob-a", "clob-b"]
        or payload.get("required_rtds_topics") != ["crypto_prices_chainlink"]
        or payload.get("optional_predictor_sources_captured") != []
        or payload.get("repository_commit_oid") != plan.repository_commit_oid
        or payload.get("repository_tree_oid") != plan.repository_tree_oid
        or any(payload.get(name) is not False for name in false_fields)
    ):
        raise ValueError("Round 20 campaign segment manifest differs")
    return {**payload, "manifest_sha256": claimed}


@dataclass(frozen=True, slots=True)
class PolymarketRound20CampaignConfig:
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
            raise ValueError("Round 20 campaign configuration differs")


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
            raise RuntimeError("Round 20 campaign is already running") from exc
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


def _write_hashed_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    hash_name: str = "artifact_sha256",
) -> dict[str, object]:
    body = dict(payload)
    body[hash_name] = _canonical_sha256(body)
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("ascii")
    write_bytes_atomic(path, encoded)
    return body


def _segment_result_path(state_root: Path, segment_index: int) -> Path:
    return state_root / "segments" / f"segment-{segment_index:04d}.json"


def _write_segment_result(
    state_root: Path,
    *,
    plan: PolymarketRound20CampaignPlan,
    segment_index: int,
    status: str,
    details: Mapping[str, object],
) -> dict[str, object]:
    if status not in {"complete", "degraded", "failed", "interrupted"}:
        raise ValueError("Round 20 campaign segment status differs")
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "segment_index": int(segment_index),
        "status": status,
        "observed_at_ms": time.time_ns() // 1_000_000,
        "condition_admission_pending": status in {"complete", "degraded"},
        "details": dict(details),
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    path = _segment_result_path(state_root, segment_index)
    if path.exists():
        raise FileExistsError(f"Round 20 segment result already exists: {path}")
    return _write_hashed_json(path, payload)


def _segment_results(
    state_root: Path,
    plan: PolymarketRound20CampaignPlan,
) -> tuple[dict[str, object], ...]:
    root = state_root / "segments"
    if not root.exists():
        return ()
    results: list[dict[str, object]] = []
    for expected_index, path in enumerate(sorted(root.glob("segment-*.json"))):
        value = dict(_read_strict_json(path, label="Round 20 segment result"))
        claimed = str(value.pop("artifact_sha256", "")).strip().lower()
        if (
            claimed != _canonical_sha256(value)
            or value.get("schema_version")
            != POLYMARKET_ROUND20_SEGMENT_RESULT_SCHEMA_VERSION
            or value.get("plan_sha256") != plan.plan_sha256
            or value.get("segment_index") != expected_index
            or path != _segment_result_path(state_root, expected_index)
            or value.get("status")
            not in {"complete", "degraded", "failed", "interrupted"}
            or value.get("model_data_eligible") is not False
            or value.get("profitability_claim") is not False
            or value.get("paper_trading_authority") is not False
            or value.get("live_trading_authority") is not False
        ):
            raise ValueError("Round 20 segment result set differs")
        results.append({**value, "artifact_sha256": claimed})
    return tuple(results)


def _resource_block(config: PolymarketRound20CampaignConfig) -> str | None:
    database_bytes = (
        config.database_path.stat().st_size if config.database_path.is_file() else 0
    )
    wal = Path(f"{config.database_path}.wal")
    database_bytes += wal.stat().st_size if wal.is_file() else 0
    if database_bytes >= POLYMARKET_ROUND20_DATABASE_CAP_BYTES:
        return "database_cap_reached"
    config.database_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(config.database_path.parent).free < (
        POLYMARKET_ROUND20_MINIMUM_FREE_BYTES
    ):
        return "minimum_free_space_not_met"
    return None


def _load_and_verify(
    config: PolymarketRound20CampaignConfig,
) -> PolymarketRound20CampaignPlan:
    config.validate()
    plan = load_round20_campaign_plan(config.plan_path)
    _validate_campaign_design(config.repository.resolve())
    _verify_repository(config.repository.resolve(), plan)
    return plan


def _recover_orphaned_segments(
    config: PolymarketRound20CampaignConfig,
    plan: PolymarketRound20CampaignPlan,
    *,
    first_segment_index: int,
) -> int:
    if not config.database_path.is_file():
        return 0
    recovered = 0
    now_ms = time.time_ns() // 1_000_000
    with PolymarketEvidenceStore(
        config.database_path,
        memory_limit="1GB",
        threads=2,
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
        for offset, (run_id, started_at_ms, manifest_json) in enumerate(rows):
            manifest = json.loads(
                str(manifest_json),
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
            if not isinstance(manifest, Mapping):
                raise ValueError("Round 20 orphan manifest differs")
            validate_round20_segment_manifest(manifest, plan)
            segment_index = manifest.get("segment_index")
            expected_index = first_segment_index + offset
            if segment_index != expected_index:
                raise ValueError("Round 20 orphan segment index differs")
            report = store.fail_run(
                str(run_id),
                started_at_ms=int(started_at_ms),
                ended_at_ms=max(int(started_at_ms), now_ms),
                database=str(config.database_path),
                errors=("campaign_restart_interrupted_segment",),
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
                    "integrity_errors": list(report.integrity_errors),
                    "errors": list(report.errors),
                },
            )
            recovered += 1
    return recovered


def inspect_round20_campaign(
    config: PolymarketRound20CampaignConfig,
    *,
    now_ms: int | None = None,
) -> dict[str, object]:
    plan = _load_and_verify(config)
    observed = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
    results = _segment_results(config.state_root, plan)
    relation = (
        "before_campaign"
        if observed < plan.scheduled_start_ms
        else "after_campaign"
        if observed >= plan.scheduled_end_ms
        else "open"
    )
    return {
        "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "observed_at_ms": observed,
        "relation": relation,
        "current_logical_unit_index": plan.logical_unit_index(observed),
        "terminal_segment_count": len(results),
        "resource_block": _resource_block(config),
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


async def run_round20_campaign(
    config: PolymarketRound20CampaignConfig,
    *,
    poll_interval_seconds: float = 1.0,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    plan = _load_and_verify(config)
    interval = float(poll_interval_seconds)
    if not 0.1 <= interval <= 30.0:
        raise ValueError("Round 20 campaign poll interval differs")
    config.state_root.mkdir(parents=True, exist_ok=True)

    def notify(value: Mapping[str, object]) -> None:
        if progress is None:
            return
        try:
            progress(value)
        except Exception:
            return

    with _CampaignFileLock(config.state_root / "campaign.lock"):
        next_waiting_heartbeat = 0.0
        while time.time_ns() // 1_000_000 < plan.scheduled_start_ms:
            monotonic_now = time.monotonic()
            if monotonic_now >= next_waiting_heartbeat:
                observed_at_ms = time.time_ns() // 1_000_000
                state = {
                    "schema_version": (
                        POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION
                    ),
                    "plan_sha256": plan.plan_sha256,
                    "observed_at_ms": observed_at_ms,
                    "phase": "waiting-for-campaign-start",
                    "scheduled_start_ms": plan.scheduled_start_ms,
                    "remaining_seconds": max(
                        0.0,
                        (plan.scheduled_start_ms - observed_at_ms) / 1_000.0,
                    ),
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
                next_waiting_heartbeat = monotonic_now + 30.0
            await asyncio.sleep(interval)
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
                    **inspect_round20_campaign(config, now_ms=now_ms),
                    "status": "resource_blocked",
                    "resource_block": resource,
                }
            segment_index = len(results)
            duration_seconds = min(
                POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS,
                max(5, math.ceil(remaining_ms / 1_000)),
            )
            recorder = create_round20_recorder(config.database_path)
            manifest_holder: dict[str, object] = {}

            def manifest_factory(
                run_id: str,
                created_at_ms: int,
            ) -> Mapping[str, object]:
                manifest = build_round20_segment_manifest(
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
                observed_at_ms = time.time_ns() // 1_000_000
                state = {
                    "schema_version": (
                        POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION
                    ),
                    "plan_sha256": plan.plan_sha256,
                    "observed_at_ms": observed_at_ms,
                    "phase": phase,
                    "segment_index": segment_index,
                    "current_logical_unit_index": plan.logical_unit_index(
                        observed_at_ms
                    ),
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
                    raise RuntimeError("Round 20 segment manifest was not created")
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
                        "condition_count": len(report.conditions),
                        "integrity_errors": list(report.integrity_errors),
                        "errors": list(report.errors),
                    },
                )
            except Exception as exc:
                result = _write_segment_result(
                    config.state_root,
                    plan=plan,
                    segment_index=segment_index,
                    status="failed",
                    details={
                        "failure_type": type(exc).__name__,
                        "failure": str(exc)[:2_000],
                    },
                )
            results.append(result)
            notify(result)
            if result["status"] in {"complete", "degraded"}:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if (
                    consecutive_failures
                    >= POLYMARKET_ROUND20_MAXIMUM_CONSECUTIVE_FAILURES
                ):
                    break
                await asyncio.sleep(POLYMARKET_ROUND20_FAILURE_BACKOFF_SECONDS)
        status_counts: dict[str, int] = {}
        for result in results:
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        terminal = {
            "schema_version": POLYMARKET_ROUND20_CAMPAIGN_STATE_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "status": (
                "campaign_window_ended"
                if time.time_ns() // 1_000_000 >= plan.scheduled_end_ms
                else "campaign_failed"
            ),
            "terminal_segment_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "recovered_interrupted_segment_count": recovered,
            "condition_admission_pending": True,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        persisted_terminal = _write_hashed_json(
            config.state_root / "campaign-state.json",
            terminal,
        )
        notify(persisted_terminal)
        return persisted_terminal


__all__ = [
    "POLYMARKET_ROUND20_CAMPAIGN_DESIGN_SHA256",
    "POLYMARKET_ROUND20_CAMPAIGN_PLAN_SCHEMA_VERSION",
    "POLYMARKET_ROUND20_LOGICAL_UNIT_COUNT",
    "POLYMARKET_ROUND20_LOGICAL_UNIT_SECONDS",
    "PolymarketRound20CampaignConfig",
    "PolymarketRound20CampaignPlan",
    "build_round20_campaign_plan",
    "build_round20_segment_manifest",
    "create_round20_campaign_plan",
    "inspect_round20_campaign",
    "load_round20_campaign_plan",
    "run_round20_campaign",
    "validate_round20_campaign_plan",
    "validate_round20_segment_manifest",
    "write_round20_campaign_plan",
]
