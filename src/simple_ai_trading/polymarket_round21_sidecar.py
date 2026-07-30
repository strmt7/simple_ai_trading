"""Independent public Binance predictor sidecar for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from .polymarket_recorder import PolymarketPublicRecorder
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256


POLYMARKET_ROUND21_SIDECAR_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-capture-design-v1"
)
POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256 = (
    "c802b13e169f868c7a37619669cdc957862a1cb58c6d3299c0aae63ff0d86d4a"
)
POLYMARKET_ROUND21_SIDECAR_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round21-binance-sidecar-manifest-v1"
)
POLYMARKET_ROUND21_CAMPAIGN_PLAN_SHA256 = (
    "2c1d87577de566bd4934c9678bcbded5bf156b671a413b83fa6d463372db1d71"
)
POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS = 1_788_046_800_000
POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES = 64 * 1024**3
POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES = 256 * 1024**3
_REQUIRED_FILES = (
    "docs/model-research/polymarket/"
    "round-021-independent-matched-edge-contract-v1.json",
    "docs/model-research/polymarket/"
    "round-021-binance-sidecar-capture-design-v1.json",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_round21_contract.py",
    "src/simple_ai_trading/polymarket_round21_sidecar.py",
    "tests/test_polymarket_recorder.py",
    "tests/test_polymarket_round21_contract.py",
    "tests/test_polymarket_round21_sidecar.py",
    "tools/run_polymarket_round21_sidecar.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID = re.compile(r"^[0-9a-f]{40,64}$")


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


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    if result.returncode:
        raise ValueError(
            "Round 21 sidecar Git operation failed: "
            + (result.stderr.strip() or result.stdout.strip())[:500]
        )
    return result.stdout.strip()


def _repository_attestation(
    repository: str | Path,
) -> tuple[str, str, dict[str, str]]:
    root = Path(repository).resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Round 21 sidecar requires a clean Git worktree")
    commit_oid = _git(root, "rev-parse", "HEAD").lower()
    tree_oid = _git(root, "rev-parse", "HEAD^{tree}").lower()
    files: dict[str, str] = {}
    for relative in _REQUIRED_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Round 21 sidecar file is unavailable: {relative}")
        files[relative] = _file_sha256(path)
    return commit_oid, tree_oid, files


def create_round21_sidecar_manifest(
    *,
    run_id: str,
    created_at_ms: int,
    capture_duration_seconds: int,
    scheduled_end_ms: int,
    repository_commit_oid: str,
    repository_tree_oid: str,
    repository_file_sha256: Mapping[str, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND21_SIDECAR_MANIFEST_SCHEMA_VERSION,
        "round21_contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
        "sidecar_design_sha256": POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256,
        "parent_round20_plan_sha256": POLYMARKET_ROUND21_CAMPAIGN_PLAN_SHA256,
        "purpose": "round21_optional_predictor_sidecar",
        "run_id": str(run_id),
        "created_at_ms": int(created_at_ms),
        "capture_duration_seconds": int(capture_duration_seconds),
        "scheduled_end_ms": int(scheduled_end_ms),
        "required_assets": [],
        "required_streams": ["binance_futures", "binance_spot"],
        "spot_streams": ["btcusdt@bookTicker", "btcusdt@trade"],
        "usdm_streams": ["btcusdt@bookTicker", "btcusdt@trade"],
        "repository_commit_oid": str(repository_commit_oid).lower(),
        "repository_tree_oid": str(repository_tree_oid).lower(),
        "repository_file_sha256": dict(sorted(repository_file_sha256.items())),
        "clean_worktree_before_capture": True,
        "outcomes_consulted": False,
        "model_scores_consulted": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round21_sidecar_manifest(payload)


def validate_round21_sidecar_manifest(
    value: Mapping[str, object],
) -> dict[str, object]:
    manifest = dict(value)
    claimed = str(manifest.pop("manifest_sha256", "")).strip().lower()
    files = manifest.get("repository_file_sha256")
    expected_keys = {
        "schema_version",
        "round21_contract_sha256",
        "sidecar_design_sha256",
        "parent_round20_plan_sha256",
        "purpose",
        "run_id",
        "created_at_ms",
        "capture_duration_seconds",
        "scheduled_end_ms",
        "required_assets",
        "required_streams",
        "spot_streams",
        "usdm_streams",
        "repository_commit_oid",
        "repository_tree_oid",
        "repository_file_sha256",
        "clean_worktree_before_capture",
        "outcomes_consulted",
        "model_scores_consulted",
        "binance_credentials_used",
        "binance_execution_connected",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    }
    false_fields = (
        "outcomes_consulted",
        "model_scores_consulted",
        "binance_credentials_used",
        "binance_execution_connected",
        "model_data_eligible",
        "profitability_claim",
        "paper_trading_authority",
        "live_trading_authority",
    )
    if (
        set(manifest) != expected_keys
        or claimed != _canonical_sha256(manifest)
        or _SHA256.fullmatch(claimed) is None
        or manifest["schema_version"]
        != POLYMARKET_ROUND21_SIDECAR_MANIFEST_SCHEMA_VERSION
        or manifest["round21_contract_sha256"]
        != POLYMARKET_ROUND21_CONTRACT_SHA256
        or manifest["sidecar_design_sha256"]
        != POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256
        or manifest["parent_round20_plan_sha256"]
        != POLYMARKET_ROUND21_CAMPAIGN_PLAN_SHA256
        or manifest["purpose"] != "round21_optional_predictor_sidecar"
        or re.fullmatch(r"[0-9a-f]{32}", str(manifest["run_id"])) is None
        or type(manifest["created_at_ms"]) is not int
        or int(manifest["created_at_ms"]) <= 0
        or type(manifest["capture_duration_seconds"]) is not int
        or not 5 <= int(manifest["capture_duration_seconds"]) <= 30 * 86_400
        or type(manifest["scheduled_end_ms"]) is not int
        or int(manifest["scheduled_end_ms"]) != POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS
        or manifest["required_assets"] != []
        or manifest["required_streams"] != ["binance_futures", "binance_spot"]
        or manifest["spot_streams"]
        != ["btcusdt@bookTicker", "btcusdt@trade"]
        or manifest["usdm_streams"]
        != ["btcusdt@bookTicker", "btcusdt@trade"]
        or manifest["clean_worktree_before_capture"] is not True
        or any(manifest[field] is not False for field in false_fields)
        or _GIT_OID.fullmatch(str(manifest["repository_commit_oid"])) is None
        or _GIT_OID.fullmatch(str(manifest["repository_tree_oid"])) is None
        or not isinstance(files, Mapping)
        or set(files) != set(_REQUIRED_FILES)
        or any(_SHA256.fullmatch(str(item)) is None for item in files.values())
    ):
        raise ValueError("Round 21 sidecar manifest differs")
    return {**manifest, "manifest_sha256": claimed}


def build_round21_sidecar_manifest(
    repository: str | Path,
    *,
    run_id: str,
    created_at_ms: int,
    capture_duration_seconds: int,
    scheduled_end_ms: int = POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
) -> dict[str, object]:
    commit_oid, tree_oid, files = _repository_attestation(repository)
    return create_round21_sidecar_manifest(
        run_id=run_id,
        created_at_ms=created_at_ms,
        capture_duration_seconds=capture_duration_seconds,
        scheduled_end_ms=scheduled_end_ms,
        repository_commit_oid=commit_oid,
        repository_tree_oid=tree_oid,
        repository_file_sha256=files,
    )


def verify_round21_sidecar_attestation(
    repository: str | Path,
    value: Mapping[str, object],
) -> None:
    manifest = validate_round21_sidecar_manifest(value)
    root = Path(repository).resolve()
    if _git(root, "rev-parse", "HEAD").lower() != manifest["repository_commit_oid"]:
        raise ValueError("Round 21 sidecar commit differs")
    if _git(root, "rev-parse", "HEAD^{tree}").lower() != manifest[
        "repository_tree_oid"
    ]:
        raise ValueError("Round 21 sidecar tree differs")
    files = manifest["repository_file_sha256"]
    if not isinstance(files, Mapping):
        raise AssertionError("validated Round 21 sidecar files are unavailable")
    for relative, expected in files.items():
        path = root / str(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or not hmac.compare_digest(_file_sha256(path), str(expected))
        ):
            raise ValueError(f"Round 21 sidecar file bytes differ: {relative}")


def create_round21_sidecar_recorder(
    database: str | Path,
) -> PolymarketPublicRecorder:
    return PolymarketPublicRecorder(
        database,
        queue_capacity=50_000,
        discovery_interval_seconds=30,
        memory_limit="512MB",
        database_threads=1,
        assets=("BTC",),
        include_binance_futures=True,
        include_binance_spot=True,
        include_rtds_binance=False,
        include_polymarket_core=False,
        binance_book_ticker_profile=True,
    )


def round21_sidecar_state(
    *,
    phase: str,
    observed_at_ms: int,
    database_bytes: int,
    wal_bytes: int,
    free_bytes: int,
    details: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "polymarket-round21-binance-sidecar-state-v1",
        "phase": str(phase),
        "observed_at_ms": int(observed_at_ms),
        "round21_contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
        "sidecar_design_sha256": POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256,
        "database_bytes": int(database_bytes),
        "wal_bytes": int(wal_bytes),
        "free_bytes": int(free_bytes),
        "details": dict(details or {}),
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "POLYMARKET_ROUND21_SIDECAR_DATABASE_CAP_BYTES",
    "POLYMARKET_ROUND21_SIDECAR_DESIGN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SIDECAR_DESIGN_SHA256",
    "POLYMARKET_ROUND21_SIDECAR_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SIDECAR_MINIMUM_FREE_BYTES",
    "POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS",
    "build_round21_sidecar_manifest",
    "create_round21_sidecar_manifest",
    "create_round21_sidecar_recorder",
    "round21_sidecar_state",
    "validate_round21_sidecar_manifest",
    "verify_round21_sidecar_attestation",
]
