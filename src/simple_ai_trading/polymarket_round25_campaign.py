"""Source-pinned prospective core capture for Polymarket Round 25."""

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

from .polymarket import (
    POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION,
    PolymarketPublicClient,
    validate_clob_market_info,
)
from .polymarket_recorder import (
    POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS,
    PolymarketPublicRecorder,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND25_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/round-025-twap-core-capture-design-v1.json"
)
POLYMARKET_ROUND25_DESIGN_SHA256 = (
    "b5c130622514b2b82855f0e1cc011b29a81bf0583e6bd37fa3e4d4b702d6a113"
)
POLYMARKET_ROUND25_PLAN_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-campaign-plan-v1"
)
POLYMARKET_ROUND25_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-segment-manifest-v1"
)
POLYMARKET_ROUND25_STATE_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-campaign-state-v1"
)
POLYMARKET_ROUND25_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-twap-core-segment-result-v1"
)
POLYMARKET_ROUND25_RESOLUTION_SOURCE = (
    "https://data.chain.link/streams/btc-usd-twap-30s-streams"
)
POLYMARKET_ROUND25_START_MS = 1_786_406_400_000
POLYMARKET_ROUND25_END_MS = 1_788_046_800_000
POLYMARKET_ROUND25_DATABASE_CAP_BYTES = 200 * 1024**3
POLYMARKET_ROUND25_MINIMUM_FREE_BYTES = 512 * 1024**3
POLYMARKET_ROUND25_FAILURE_BACKOFF_SECONDS = 60
POLYMARKET_ROUND25_MAXIMUM_CONSECUTIVE_FAILURES = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_FILES = (
    "docs/model-research/polymarket/round-025-twap-core-capture-design-v1.json",
    "docs/model-research/polymarket/round-025-twap-source-qualification-2026-08-10.json",
    "src/simple_ai_trading/polymarket.py",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_round25_campaign.py",
    "tests/test_polymarket.py",
    "tests/test_polymarket_round25_campaign.py",
    "tools/qualify_polymarket_round25_source.py",
    "tools/run_polymarket_round25_campaign.py",
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 JSON contains {value}")


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or not 2 <= path.stat().st_size <= 2**20:
        raise ValueError(f"Round 25 {label} is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 {label} is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 25 {label} is not an object")
    return dict(value)


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if completed.returncode:
        raise ValueError("Round 25 repository identity is unavailable")
    return completed.stdout.strip()


def _repository_attestation(
    repository: Path,
    *,
    require_clean: bool,
) -> tuple[str, str, dict[str, str]]:
    root = repository.resolve()
    if require_clean and _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Round 25 campaign requires a clean worktree")
    commit = _git(root, "rev-parse", "HEAD").lower()
    tree = _git(root, "rev-parse", "HEAD^{tree}").lower()
    files: dict[str, str] = {}
    for relative in _REQUIRED_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Round 25 campaign file is unavailable: {relative}")
        files[relative] = _file_sha256(path)
    if _GIT_OID.fullmatch(commit) is None or _GIT_OID.fullmatch(tree) is None:
        raise ValueError("Round 25 repository identity differs")
    return commit, tree, files


def load_round25_design(repository: str | Path) -> dict[str, object]:
    payload = _read_json(
        Path(repository).resolve() / POLYMARKET_ROUND25_DESIGN_RELATIVE,
        label="capture design",
    )
    claimed = str(payload.pop("design_sha256", "")).strip().lower()
    source = payload.get("source_regime")
    schedule = payload.get("schedule")
    authority = payload.get("authority")
    if (
        claimed != POLYMARKET_ROUND25_DESIGN_SHA256
        or claimed != _canonical_sha256(payload)
        or payload.get("status")
        != "frozen_after_public_source_drift_observation_before_successor_capture"
        or not isinstance(source, Mapping)
        or source.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or source.get("legacy_and_twap_conditions_may_be_pooled") is not False
        or not isinstance(schedule, Mapping)
        or schedule.get("scheduled_start_ms") != POLYMARKET_ROUND25_START_MS
        or schedule.get("scheduled_end_ms") != POLYMARKET_ROUND25_END_MS
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
    ):
        raise ValueError("Round 25 capture design differs")
    return {**payload, "design_sha256": claimed}


def qualify_round25_source(
    client: PolymarketPublicClient,
    *,
    observed_at_ms: int,
) -> dict[str, object]:
    observed = int(observed_at_ms)
    if observed <= 0:
        raise ValueError("Round 25 source observation time differs")
    markets = client.discover_five_minute_markets(
        now_ms=observed,
        include_next=True,
        require_all_assets=True,
        assets=("BTC",),
    )
    expected_epoch_ms = observed // 300_000 * 300_000
    if (
        len(markets) != 2
        or tuple(market.event_start_ms for market in markets)
        != (expected_epoch_ms, expected_epoch_ms + 300_000)
        or any(
            market.resolution_source.rstrip("/")
            != POLYMARKET_ROUND25_RESOLUTION_SOURCE
            for market in markets
        )
    ):
        raise ValueError("Round 25 consecutive source markets are unavailable")
    protocol_version = client.protocol_version()
    if protocol_version != POLYMARKET_REQUIRED_CLOB_PROTOCOL_VERSION:
        raise ValueError("Round 25 CLOB protocol version differs")
    evidence: list[dict[str, object]] = []
    for market in markets:
        info = client.clob_market_info(market.condition_id)
        clob = validate_clob_market_info(market, info)
        clob_json = _canonical_json(dict(info))
        evidence.append(
            {
                "slug": market.slug,
                "market_id": market.market_id,
                "condition_id": market.condition_id,
                "event_start_ms": market.event_start_ms,
                "end_ms": market.end_ms,
                "resolution_source": market.resolution_source,
                "gamma_payload_sha256": market.gamma_payload_sha256,
                "gamma_payload_json": market.gamma_payload_json,
                "clob_info_sha256": hashlib.sha256(
                    clob_json.encode("ascii")
                ).hexdigest(),
                "clob_info_json": clob_json,
                "minimum_order_age_seconds": clob["minimum_order_age_seconds"],
                "taker_order_delay_enabled": clob["taker_order_delay_enabled"],
            }
        )
    payload: dict[str, object] = {
        "schema_version": "polymarket-round25-twap-source-qualification-v1",
        "status": "passed",
        "observed_at_ms": observed,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "clob_protocol_version": protocol_version,
        "market_count": len(evidence),
        "markets": evidence,
        "public_gamma_endpoint": "https://gamma-api.polymarket.com/markets",
        "public_clob_endpoint": "https://clob.polymarket.com",
        "credentials_used": False,
        "execution_connected": False,
        "account_state_accessed": False,
        "outcomes_accessed": False,
        "model_scores_accessed": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["qualification_sha256"] = _canonical_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class PolymarketRound25CampaignPlan:
    created_at_ms: int
    repository_commit_oid: str
    repository_tree_oid: str
    repository_file_sha256: Mapping[str, str]
    source_qualification_sha256: str
    plan_sha256: str

    @property
    def scheduled_start_ms(self) -> int:
        return POLYMARKET_ROUND25_START_MS

    @property
    def scheduled_end_ms(self) -> int:
        return POLYMARKET_ROUND25_END_MS


def create_round25_campaign_plan(
    *,
    created_at_ms: int,
    repository_commit_oid: str,
    repository_tree_oid: str,
    repository_file_sha256: Mapping[str, str],
    source_qualification_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND25_PLAN_SCHEMA_VERSION,
        "created_at_ms": int(created_at_ms),
        "scheduled_start_ms": POLYMARKET_ROUND25_START_MS,
        "scheduled_end_ms": POLYMARKET_ROUND25_END_MS,
        "design_sha256": POLYMARKET_ROUND25_DESIGN_SHA256,
        "source_qualification_sha256": str(source_qualification_sha256).lower(),
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "required_assets": ["BTC"],
        "required_streams": ["clob_market", "polymarket_rtds"],
        "required_clob_lanes": ["clob"],
        "required_rtds_topics": ["crypto_prices_chainlink"],
        "database_cap_bytes": POLYMARKET_ROUND25_DATABASE_CAP_BYTES,
        "minimum_free_bytes": POLYMARKET_ROUND25_MINIMUM_FREE_BYTES,
        "repository_commit_oid": str(repository_commit_oid).lower(),
        "repository_tree_oid": str(repository_tree_oid).lower(),
        "repository_file_sha256": dict(sorted(repository_file_sha256.items())),
        "binance_captured": False,
        "credentials_used": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["plan_sha256"] = _canonical_sha256(payload)
    validate_round25_campaign_plan(payload)
    return payload


def validate_round25_campaign_plan(
    value: Mapping[str, object],
) -> PolymarketRound25CampaignPlan:
    payload = dict(value)
    claimed = str(payload.pop("plan_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "created_at_ms",
        "scheduled_start_ms",
        "scheduled_end_ms",
        "design_sha256",
        "source_qualification_sha256",
        "resolution_source",
        "required_assets",
        "required_streams",
        "required_clob_lanes",
        "required_rtds_topics",
        "database_cap_bytes",
        "minimum_free_bytes",
        "repository_commit_oid",
        "repository_tree_oid",
        "repository_file_sha256",
        "binance_captured",
        "credentials_used",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    files = payload.get("repository_file_sha256")
    false_fields = (
        "binance_captured",
        "credentials_used",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        set(payload) != expected_keys
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version") != POLYMARKET_ROUND25_PLAN_SCHEMA_VERSION
        or type(payload.get("created_at_ms")) is not int
        or not 0 < int(payload["created_at_ms"]) < POLYMARKET_ROUND25_START_MS
        or payload.get("scheduled_start_ms") != POLYMARKET_ROUND25_START_MS
        or payload.get("scheduled_end_ms") != POLYMARKET_ROUND25_END_MS
        or payload.get("design_sha256") != POLYMARKET_ROUND25_DESIGN_SHA256
        or _SHA256.fullmatch(str(payload.get("source_qualification_sha256"))) is None
        or payload.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or payload.get("required_assets") != ["BTC"]
        or payload.get("required_streams") != ["clob_market", "polymarket_rtds"]
        or payload.get("required_clob_lanes") != ["clob"]
        or payload.get("required_rtds_topics") != ["crypto_prices_chainlink"]
        or payload.get("database_cap_bytes")
        != POLYMARKET_ROUND25_DATABASE_CAP_BYTES
        or payload.get("minimum_free_bytes")
        != POLYMARKET_ROUND25_MINIMUM_FREE_BYTES
        or _GIT_OID.fullmatch(str(payload.get("repository_commit_oid"))) is None
        or _GIT_OID.fullmatch(str(payload.get("repository_tree_oid"))) is None
        or not isinstance(files, Mapping)
        or set(files) != set(_REQUIRED_FILES)
        or any(_SHA256.fullmatch(str(item)) is None for item in files.values())
        or any(payload.get(field) is not False for field in false_fields)
    ):
        raise ValueError("Round 25 campaign plan differs")
    return PolymarketRound25CampaignPlan(
        created_at_ms=int(payload["created_at_ms"]),
        repository_commit_oid=str(payload["repository_commit_oid"]),
        repository_tree_oid=str(payload["repository_tree_oid"]),
        repository_file_sha256=dict(files),
        source_qualification_sha256=str(payload["source_qualification_sha256"]),
        plan_sha256=claimed,
    )


def load_round25_campaign_plan(path: str | Path) -> PolymarketRound25CampaignPlan:
    return validate_round25_campaign_plan(
        _read_json(Path(path), label="campaign plan")
    )


def build_round25_campaign_plan(
    *,
    repository: str | Path,
    source_qualification: str | Path,
    created_at_ms: int | None = None,
) -> dict[str, object]:
    root = Path(repository).resolve()
    load_round25_design(root)
    qualification = _read_json(Path(source_qualification), label="source qualification")
    qualification_sha = str(qualification.pop("qualification_sha256", "")).lower()
    if (
        qualification_sha != _canonical_sha256(qualification)
        or _SHA256.fullmatch(qualification_sha) is None
        or qualification.get("status") != "passed"
        or qualification.get("resolution_source")
        != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or qualification.get("credentials_used") is not False
        or qualification.get("execution_connected") is not False
    ):
        raise ValueError("Round 25 source qualification differs")
    commit, tree, files = _repository_attestation(root, require_clean=True)
    return create_round25_campaign_plan(
        created_at_ms=(
            time.time_ns() // 1_000_000
            if created_at_ms is None
            else int(created_at_ms)
        ),
        repository_commit_oid=commit,
        repository_tree_oid=tree,
        repository_file_sha256=files,
        source_qualification_sha256=qualification_sha,
    )


def write_round25_campaign_plan(path: str | Path, value: Mapping[str, object]) -> None:
    validate_round25_campaign_plan(value)
    write_bytes_atomic(
        Path(path),
        (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode("ascii"),
    )


def _verify_repository(repository: Path, plan: PolymarketRound25CampaignPlan) -> None:
    commit, tree, files = _repository_attestation(repository, require_clean=False)
    if (
        not hmac.compare_digest(commit, plan.repository_commit_oid)
        or not hmac.compare_digest(tree, plan.repository_tree_oid)
        or files != dict(plan.repository_file_sha256)
    ):
        raise ValueError("Round 25 campaign repository attestation differs")


def build_round25_segment_manifest(
    plan: PolymarketRound25CampaignPlan,
    *,
    run_id: str,
    created_at_ms: int,
    capture_duration_seconds: int,
    segment_index: int,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND25_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_ms": int(created_at_ms),
        "capture_duration_seconds": int(capture_duration_seconds),
        "segment_index": int(segment_index),
        "plan_sha256": plan.plan_sha256,
        "design_sha256": POLYMARKET_ROUND25_DESIGN_SHA256,
        "source_qualification_sha256": plan.source_qualification_sha256,
        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
        "scheduled_campaign_start_ms": POLYMARKET_ROUND25_START_MS,
        "scheduled_campaign_end_ms": POLYMARKET_ROUND25_END_MS,
        "required_assets": ["BTC"],
        "required_streams": ["clob_market", "polymarket_rtds"],
        "required_clob_lanes": ["clob"],
        "required_rtds_topics": ["crypto_prices_chainlink"],
        "repository_commit_oid": plan.repository_commit_oid,
        "repository_tree_oid": plan.repository_tree_oid,
        "binance_captured": False,
        "credentials_used": False,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    validate_round25_segment_manifest(payload, plan)
    return payload


def validate_round25_segment_manifest(
    value: Mapping[str, object],
    plan: PolymarketRound25CampaignPlan,
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("manifest_sha256", "")).lower()
    created = payload.get("created_at_ms")
    duration = payload.get("capture_duration_seconds")
    false_fields = (
        "binance_captured",
        "credentials_used",
        "outcomes_consulted",
        "model_scores_consulted",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or payload.get("schema_version")
        != POLYMARKET_ROUND25_MANIFEST_SCHEMA_VERSION
        or not isinstance(payload.get("run_id"), str)
        or not payload["run_id"]
        or type(created) is not int
        or not POLYMARKET_ROUND25_START_MS <= int(created) < POLYMARKET_ROUND25_END_MS
        or type(duration) is not int
        or not 5 <= int(duration) <= POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS
        or int(created) + int(duration) * 1_000
        > POLYMARKET_ROUND25_END_MS + 1_000
        or type(payload.get("segment_index")) is not int
        or int(payload["segment_index"]) < 0
        or payload.get("plan_sha256") != plan.plan_sha256
        or payload.get("design_sha256") != POLYMARKET_ROUND25_DESIGN_SHA256
        or payload.get("source_qualification_sha256")
        != plan.source_qualification_sha256
        or payload.get("resolution_source") != POLYMARKET_ROUND25_RESOLUTION_SOURCE
        or payload.get("scheduled_campaign_start_ms")
        != POLYMARKET_ROUND25_START_MS
        or payload.get("scheduled_campaign_end_ms") != POLYMARKET_ROUND25_END_MS
        or payload.get("required_assets") != ["BTC"]
        or payload.get("required_streams") != ["clob_market", "polymarket_rtds"]
        or payload.get("required_clob_lanes") != ["clob"]
        or payload.get("required_rtds_topics") != ["crypto_prices_chainlink"]
        or payload.get("repository_commit_oid") != plan.repository_commit_oid
        or payload.get("repository_tree_oid") != plan.repository_tree_oid
        or any(payload.get(field) is not False for field in false_fields)
    ):
        raise ValueError("Round 25 segment manifest differs")
    return {**payload, "manifest_sha256": claimed}


@dataclass(frozen=True, slots=True)
class PolymarketRound25CampaignConfig:
    repository: Path
    plan_path: Path
    database_path: Path
    state_root: Path

    def validated(self) -> PolymarketRound25CampaignConfig:
        root = self.repository.resolve()
        plan = self.plan_path.resolve()
        database = self.database_path.resolve()
        state = self.state_root.resolve()
        if (
            not root.is_dir()
            or not plan.is_file()
            or root not in plan.parents
            or root not in database.parents
            or root not in state.parents
            or database == plan
            or state == plan
            or state == database
            or state in database.parents
            or database in state.parents
        ):
            raise ValueError("Round 25 campaign configuration differs")
        return PolymarketRound25CampaignConfig(root, plan, database, state)


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
            raise RuntimeError("Round 25 campaign is already running") from exc
        self.handle = handle
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self.handle is None:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


def _write_hashed_json(path: Path, value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["artifact_sha256"] = _canonical_sha256(payload)
    write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    return payload


def _manifest_path(state_root: Path, segment_index: int) -> Path:
    return state_root / f"segment-{segment_index:04d}-manifest.json"


def _result_path(state_root: Path, segment_index: int) -> Path:
    return state_root / f"segment-{segment_index:04d}-result.json"


def _write_segment_result(
    state_root: Path,
    *,
    plan: PolymarketRound25CampaignPlan,
    segment_index: int,
    status: str,
    details: Mapping[str, object],
) -> dict[str, object]:
    if status not in {"complete", "degraded", "failed", "interrupted"}:
        raise ValueError("Round 25 segment status differs")
    return _write_hashed_json(
        _result_path(state_root, segment_index),
        {
            "schema_version": POLYMARKET_ROUND25_RESULT_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "segment_index": int(segment_index),
            "status": status,
            "observed_at_ms": time.time_ns() // 1_000_000,
            "details": dict(details),
            "condition_admission_pending": True,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
    )


def _segment_results(
    state_root: Path,
    plan: PolymarketRound25CampaignPlan,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, path in enumerate(sorted(state_root.glob("segment-*-result.json"))):
        value = _read_json(path, label="segment result")
        claimed = str(value.pop("artifact_sha256", "")).lower()
        if (
            claimed != _canonical_sha256(value)
            or value.get("schema_version") != POLYMARKET_ROUND25_RESULT_SCHEMA_VERSION
            or value.get("plan_sha256") != plan.plan_sha256
            or value.get("segment_index") != index
        ):
            raise ValueError("Round 25 segment result differs")
        results.append({**value, "artifact_sha256": claimed})
    return results


def _recover_orphans(
    config: PolymarketRound25CampaignConfig,
    plan: PolymarketRound25CampaignPlan,
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    manifests = sorted(config.state_root.glob("segment-*-manifest.json"))
    for index in range(len(results), len(manifests)):
        manifest = _read_json(manifests[index], label="segment manifest")
        validated = validate_round25_segment_manifest(manifest, plan)
        if int(validated["segment_index"]) != index:
            raise ValueError("Round 25 orphaned segment index differs")
        results.append(
            _write_segment_result(
                config.state_root,
                plan=plan,
                segment_index=index,
                status="interrupted",
                details={
                    "run_id": validated["run_id"],
                    "manifest_sha256": validated["manifest_sha256"],
                    "reason": "campaign_process_interrupted_before_terminal_report",
                },
            )
        )
    if len(manifests) != len(results):
        raise ValueError("Round 25 manifest and result sets differ")
    return results


def _resource_block(config: PolymarketRound25CampaignConfig) -> str | None:
    database_bytes = (
        config.database_path.stat().st_size if config.database_path.exists() else 0
    )
    free_bytes = shutil.disk_usage(config.database_path.parent).free
    if database_bytes >= POLYMARKET_ROUND25_DATABASE_CAP_BYTES:
        return "database_cap_reached"
    if free_bytes < POLYMARKET_ROUND25_MINIMUM_FREE_BYTES:
        return "minimum_free_space_breached"
    return None


def _create_recorder(database: Path) -> PolymarketPublicRecorder:
    client = PolymarketPublicClient(
        required_five_minute_resolution_sources={
            "BTC": POLYMARKET_ROUND25_RESOLUTION_SOURCE
        }
    )
    return PolymarketPublicRecorder(
        database,
        client=client,
        queue_capacity=100_000,
        discovery_interval_seconds=30,
        memory_limit="1GB",
        database_threads=2,
        assets=("BTC",),
        include_binance_futures=False,
        include_binance_spot=False,
        include_rtds_binance=False,
        clob_lane_ids=("clob",),
    )


def inspect_round25_campaign(
    config: PolymarketRound25CampaignConfig,
    *,
    now_ms: int | None = None,
) -> dict[str, object]:
    selected = config.validated()
    plan = load_round25_campaign_plan(selected.plan_path)
    _verify_repository(selected.repository, plan)
    results = _segment_results(selected.state_root, plan) if selected.state_root.exists() else []
    observed = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
    return {
        "schema_version": POLYMARKET_ROUND25_STATE_SCHEMA_VERSION,
        "plan_sha256": plan.plan_sha256,
        "observed_at_ms": observed,
        "phase": (
            "before_campaign"
            if observed < POLYMARKET_ROUND25_START_MS
            else "after_campaign"
            if observed >= POLYMARKET_ROUND25_END_MS
            else "campaign_window"
        ),
        "terminal_segment_count": len(results),
        "database_bytes": (
            selected.database_path.stat().st_size
            if selected.database_path.exists()
            else 0
        ),
        "free_bytes": shutil.disk_usage(selected.database_path.parent).free,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }


def _permanent_source_failure(result: Mapping[str, object]) -> bool:
    details = result.get("details")
    text = _canonical_json(details).lower()
    return "resolution source" in text or "configured exact source" in text


async def run_round25_campaign(
    config: PolymarketRound25CampaignConfig,
    *,
    poll_interval_seconds: float = 1.0,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    recorder_factory: Callable[[Path], PolymarketPublicRecorder] = _create_recorder,
) -> dict[str, object]:
    selected = config.validated()
    plan = load_round25_campaign_plan(selected.plan_path)
    load_round25_design(selected.repository)
    _verify_repository(selected.repository, plan)
    interval = float(poll_interval_seconds)
    if not 0.1 <= interval <= 30.0:
        raise ValueError("Round 25 campaign poll interval differs")
    selected.state_root.mkdir(parents=True, exist_ok=True)
    selected.database_path.parent.mkdir(parents=True, exist_ok=True)

    def notify(value: Mapping[str, object]) -> None:
        if progress is None:
            return
        try:
            progress(value)
        except Exception:
            return

    with _CampaignFileLock(selected.state_root / "campaign.lock"):
        while time.time_ns() // 1_000_000 < POLYMARKET_ROUND25_START_MS:
            observed = time.time_ns() // 1_000_000
            state = _write_hashed_json(
                selected.state_root / "campaign-state.json",
                {
                    "schema_version": POLYMARKET_ROUND25_STATE_SCHEMA_VERSION,
                    "plan_sha256": plan.plan_sha256,
                    "observed_at_ms": observed,
                    "phase": "waiting_for_campaign_start",
                    "remaining_seconds": max(
                        0.0, (POLYMARKET_ROUND25_START_MS - observed) / 1_000.0
                    ),
                    "model_data_eligible": False,
                    "profitability_claim": False,
                    "paper_trading_authority": False,
                    "live_trading_authority": False,
                },
            )
            notify(state)
            await asyncio.sleep(min(30.0, interval))
        results = _recover_orphans(
            selected,
            plan,
            _segment_results(selected.state_root, plan),
        )
        consecutive_failures = 0
        source_failure = False
        while True:
            now = time.time_ns() // 1_000_000
            remaining_ms = POLYMARKET_ROUND25_END_MS - now
            if remaining_ms < 5_000:
                break
            resource = _resource_block(selected)
            if resource is not None:
                terminal = {
                    **inspect_round25_campaign(selected, now_ms=now),
                    "status": "resource_blocked",
                    "resource_block": resource,
                    "condition_admission_pending": True,
                }
                persisted = _write_hashed_json(
                    selected.state_root / "campaign-state.json", terminal
                )
                notify(persisted)
                return persisted
            segment_index = len(results)
            duration = min(
                POLYMARKET_MAXIMUM_CAPTURE_DURATION_SECONDS,
                max(5, math.ceil(remaining_ms / 1_000)),
            )
            manifest_holder: dict[str, object] = {}

            def manifest_factory(run_id: str, created_at_ms: int) -> Mapping[str, object]:
                manifest = build_round25_segment_manifest(
                    plan,
                    run_id=run_id,
                    created_at_ms=created_at_ms,
                    capture_duration_seconds=duration,
                    segment_index=segment_index,
                )
                write_bytes_atomic(
                    _manifest_path(selected.state_root, segment_index),
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
                        "ascii"
                    ),
                )
                manifest_holder.update(manifest)
                return manifest

            def capture_progress(
                phase: str,
                details: Mapping[str, object],
            ) -> None:
                observed = time.time_ns() // 1_000_000
                state = _write_hashed_json(
                    selected.state_root / "campaign-state.json",
                    {
                        "schema_version": POLYMARKET_ROUND25_STATE_SCHEMA_VERSION,
                        "plan_sha256": plan.plan_sha256,
                        "observed_at_ms": observed,
                        "phase": phase,
                        "segment_index": segment_index,
                        "details": dict(details),
                        "database_bytes": (
                            selected.database_path.stat().st_size
                            if selected.database_path.exists()
                            else 0
                        ),
                        "free_bytes": shutil.disk_usage(
                            selected.database_path.parent
                        ).free,
                        "model_data_eligible": False,
                        "profitability_claim": False,
                        "paper_trading_authority": False,
                        "live_trading_authority": False,
                    },
                )
                notify(state)

            try:
                report = await recorder_factory(selected.database_path).run(
                    duration_seconds=duration,
                    progress=capture_progress,
                    progress_interval_seconds=30,
                    stop_requested=lambda: _resource_block(selected),
                    preregistration_manifest_factory=manifest_factory,
                )
                if not manifest_holder:
                    raise RuntimeError("Round 25 segment manifest was not created")
                _verify_repository(selected.repository, plan)
                status = (
                    report.status
                    if report.status in {"complete", "degraded"}
                    else "failed"
                )
                result = _write_segment_result(
                    selected.state_root,
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
                        "resolution_source": POLYMARKET_ROUND25_RESOLUTION_SOURCE,
                    },
                )
            except Exception as exc:
                result = _write_segment_result(
                    selected.state_root,
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
            source_failure = _permanent_source_failure(result)
            if source_failure:
                break
            if result["status"] in {"complete", "degraded"}:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= POLYMARKET_ROUND25_MAXIMUM_CONSECUTIVE_FAILURES:
                    break
                await asyncio.sleep(POLYMARKET_ROUND25_FAILURE_BACKOFF_SECONDS)
        status_counts: dict[str, int] = {}
        for result in results:
            status = str(result["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        terminal = {
            "schema_version": POLYMARKET_ROUND25_STATE_SCHEMA_VERSION,
            "plan_sha256": plan.plan_sha256,
            "status": (
                "campaign_window_ended"
                if time.time_ns() // 1_000_000 >= POLYMARKET_ROUND25_END_MS
                else "source_regime_changed"
                if source_failure
                else "campaign_failed"
            ),
            "terminal_segment_count": len(results),
            "status_counts": dict(sorted(status_counts.items())),
            "condition_admission_pending": True,
            "model_data_eligible": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
        persisted = _write_hashed_json(
            selected.state_root / "campaign-state.json", terminal
        )
        notify(persisted)
        return persisted


__all__ = [
    "POLYMARKET_ROUND25_DESIGN_SHA256",
    "POLYMARKET_ROUND25_END_MS",
    "POLYMARKET_ROUND25_RESOLUTION_SOURCE",
    "POLYMARKET_ROUND25_START_MS",
    "PolymarketRound25CampaignConfig",
    "PolymarketRound25CampaignPlan",
    "build_round25_campaign_plan",
    "build_round25_segment_manifest",
    "create_round25_campaign_plan",
    "inspect_round25_campaign",
    "load_round25_campaign_plan",
    "load_round25_design",
    "qualify_round25_source",
    "run_round25_campaign",
    "validate_round25_campaign_plan",
    "validate_round25_segment_manifest",
    "write_round25_campaign_plan",
]
