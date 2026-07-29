"""Pre-message repository attestation for Round 14 BTC-only capture units."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Mapping, Sequence

from .polymarket_round14_contract import (
    PolymarketRound14Program,
    load_round14_contract,
)


POLYMARKET_ROUND14_CAPTURE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round14-capture-manifest-v1"
)
POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS = 1_800
POLYMARKET_ROUND14_QUALIFICATION_MINIMUM_SECONDS = 60
POLYMARKET_ROUND14_QUALIFICATION_MAXIMUM_SECONDS = 300
POLYMARKET_ROUND14_REQUIRED_ASSETS = ("BTC",)
POLYMARKET_ROUND14_REQUIRED_STREAMS = (
    "binance_futures",
    "binance_spot",
    "clob_market",
    "polymarket_rtds",
)
_REQUIRED_REPOSITORY_FILES = (
    "src/simple_ai_trading/polymarket.py",
    "src/simple_ai_trading/polymarket_capture_frame.py",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_btc_reference.py",
    "src/simple_ai_trading/polymarket_round14_contract.py",
    "src/simple_ai_trading/polymarket_round14_capture.py",
    "src/simple_ai_trading/polymarket_round14_features.py",
    "tests/test_polymarket.py",
    "tests/test_polymarket_capture_frame.py",
    "tests/test_polymarket_recorder.py",
    "tests/test_polymarket_round14_contract.py",
    "tests/test_polymarket_round14_capture.py",
    "tests/test_polymarket_round14_features.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT = re.compile(r"^[0-9a-f]{40,64}$")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _safe_repository_path(value: object) -> bool:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    return bool(
        text
        and "\\" not in text
        and not path.is_absolute()
        and "." not in path.parts
        and ".." not in path.parts
    )


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        raise ValueError(
            "Round 14 capture Git operation failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _repository_root() -> Path:
    candidate = Path(__file__).resolve().parent
    root = Path(
        os.fsdecode(_git_bytes(candidate, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if not root.is_dir():
        raise ValueError("Round 14 repository root is unavailable")
    return root


def create_round14_capture_manifest(
    *,
    run_id: str,
    created_at_ms: int,
    capture_duration_seconds: int,
    purpose: str,
    repository_commit: str,
    repository_tree: str,
    contract_repository_path: str,
    contract_sha256: str,
    required_file_sha256: Mapping[str, str],
    slot_index: int | None = None,
    scheduled_start_ms: int | None = None,
    campaign_plan_sha256: str | None = None,
) -> dict[str, object]:
    normalized_purpose = str(purpose or "").strip().lower()
    model_data_eligible = normalized_purpose == "prospective"
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND14_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id or "").strip(),
        "created_at_ms": int(created_at_ms),
        "capture_duration_seconds": int(capture_duration_seconds),
        "purpose": normalized_purpose,
        "model_data_eligible": model_data_eligible,
        "slot_index": slot_index,
        "scheduled_start_ms": scheduled_start_ms,
        "campaign_plan_sha256": campaign_plan_sha256,
        "repository_commit": str(repository_commit or "").strip().lower(),
        "repository_tree": str(repository_tree or "").strip().lower(),
        "contract_repository_path": str(contract_repository_path or "").strip(),
        "contract_sha256": str(contract_sha256 or "").strip().lower(),
        "required_file_sha256": dict(
            sorted(
                (str(path), str(digest).strip().lower())
                for path, digest in required_file_sha256.items()
            )
        ),
        "required_assets": list(POLYMARKET_ROUND14_REQUIRED_ASSETS),
        "required_streams": list(POLYMARKET_ROUND14_REQUIRED_STREAMS),
        "source_hash_algorithm": "sha256_raw_committed_bytes",
        "capture_started_before_manifest": False,
        "outcome_endpoints_queried": False,
        "labels_consulted": False,
        "model_scores_consulted": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round14_capture_manifest(payload)


def validate_round14_capture_manifest(
    value: Mapping[str, object],
    *,
    expected_run_id: str | None = None,
    expected_program: PolymarketRound14Program | None = None,
) -> dict[str, object]:
    manifest = dict(value)
    claimed = str(manifest.pop("manifest_sha256", "")).strip().lower()
    expected_keys = {
        "schema_version",
        "run_id",
        "created_at_ms",
        "capture_duration_seconds",
        "purpose",
        "model_data_eligible",
        "slot_index",
        "scheduled_start_ms",
        "campaign_plan_sha256",
        "repository_commit",
        "repository_tree",
        "contract_repository_path",
        "contract_sha256",
        "required_file_sha256",
        "required_assets",
        "required_streams",
        "source_hash_algorithm",
        "capture_started_before_manifest",
        "outcome_endpoints_queried",
        "labels_consulted",
        "model_scores_consulted",
        "paper_trading_authority",
        "live_trading_authority",
    }
    if set(manifest) != expected_keys:
        raise ValueError("Round 14 capture manifest schema is invalid")
    purpose = manifest["purpose"]
    duration = manifest["capture_duration_seconds"]
    slot_index = manifest["slot_index"]
    scheduled = manifest["scheduled_start_ms"]
    plan_sha = manifest["campaign_plan_sha256"]
    qualification = purpose == "qualification"
    prospective = purpose == "prospective"
    files = manifest["required_file_sha256"]
    common_valid = (
        manifest["schema_version"]
        == POLYMARKET_ROUND14_CAPTURE_MANIFEST_SCHEMA_VERSION
        and bool(str(manifest["run_id"]))
        and type(manifest["created_at_ms"]) is int
        and manifest["created_at_ms"] > 0
        and type(duration) is int
        and (qualification or prospective)
        and manifest["model_data_eligible"] is prospective
        and _GIT_OBJECT.fullmatch(str(manifest["repository_commit"])) is not None
        and _GIT_OBJECT.fullmatch(str(manifest["repository_tree"])) is not None
        and _safe_repository_path(manifest["contract_repository_path"])
        and _SHA256.fullmatch(str(manifest["contract_sha256"])) is not None
        and isinstance(files, Mapping)
        and bool(files)
        and all(
            _safe_repository_path(path)
            and _SHA256.fullmatch(str(digest)) is not None
            for path, digest in files.items()
        )
        and set(_REQUIRED_REPOSITORY_FILES).issubset(files)
        and manifest["contract_repository_path"] in files
        and manifest["required_assets"]
        == list(POLYMARKET_ROUND14_REQUIRED_ASSETS)
        and manifest["required_streams"]
        == list(POLYMARKET_ROUND14_REQUIRED_STREAMS)
        and manifest["source_hash_algorithm"] == "sha256_raw_committed_bytes"
        and manifest["capture_started_before_manifest"] is False
        and manifest["outcome_endpoints_queried"] is False
        and manifest["labels_consulted"] is False
        and manifest["model_scores_consulted"] is False
        and manifest["paper_trading_authority"] is False
        and manifest["live_trading_authority"] is False
        and _SHA256.fullmatch(claimed) is not None
        and _canonical_sha256(manifest) == claimed
    )
    qualification_valid = qualification and (
        POLYMARKET_ROUND14_QUALIFICATION_MINIMUM_SECONDS
        <= duration
        <= POLYMARKET_ROUND14_QUALIFICATION_MAXIMUM_SECONDS
        and slot_index is None
        and scheduled is None
        and plan_sha is None
    )
    prospective_valid = prospective and (
        duration == POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS
        and type(slot_index) is int
        and 0 <= slot_index < 1_440
        and type(scheduled) is int
        and scheduled > 0
        and scheduled % 1_800_000 == 0
        and 0 <= manifest["created_at_ms"] - scheduled <= 60_000
        and _SHA256.fullmatch(str(plan_sha)) is not None
    )
    if not common_valid or not (qualification_valid or prospective_valid):
        raise ValueError("Round 14 capture manifest is invalid")
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("Round 14 capture manifest binds another run")
    if (
        expected_program is not None
        and manifest["contract_sha256"] != expected_program.contract_sha256
    ):
        raise ValueError("Round 14 capture manifest binds another contract")
    return {**manifest, "manifest_sha256": claimed}


def build_round14_capture_manifest(
    contract_path: str | Path,
    *,
    run_id: str,
    created_at_ms: int,
    capture_duration_seconds: int,
    purpose: str,
    slot_index: int | None = None,
    scheduled_start_ms: int | None = None,
    campaign_plan_sha256: str | None = None,
    additional_required_files: Sequence[str] = (),
) -> dict[str, object]:
    program = load_round14_contract(contract_path)
    root = _repository_root()
    if _git_bytes(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("Round 14 capture requires a clean Git worktree")
    commit = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip().lower()
    tree = (
        _git_bytes(root, "rev-parse", "HEAD^{tree}")
        .decode("ascii")
        .strip()
        .lower()
    )
    selected_contract = Path(contract_path).resolve()
    try:
        contract_relative = selected_contract.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Round 14 contract is outside the repository") from exc
    required = tuple(
        dict.fromkeys(
            (
                *_REQUIRED_REPOSITORY_FILES,
                contract_relative,
                *(str(value) for value in additional_required_files),
            )
        )
    )
    file_hashes: dict[str, str] = {}
    for relative in required:
        if not _safe_repository_path(relative):
            raise ValueError("Round 14 required repository path is unsafe")
        committed = _git_bytes(root, "show", f"{commit}:{relative}")
        file_hashes[relative] = hashlib.sha256(committed).hexdigest()
    return create_round14_capture_manifest(
        run_id=run_id,
        created_at_ms=created_at_ms,
        capture_duration_seconds=capture_duration_seconds,
        purpose=purpose,
        repository_commit=commit,
        repository_tree=tree,
        contract_repository_path=contract_relative,
        contract_sha256=program.contract_sha256,
        required_file_sha256=file_hashes,
        slot_index=slot_index,
        scheduled_start_ms=scheduled_start_ms,
        campaign_plan_sha256=campaign_plan_sha256,
    )


def verify_round14_repository_attestation(
    value: Mapping[str, object],
) -> dict[str, object]:
    manifest = validate_round14_capture_manifest(value)
    root = _repository_root()
    commit = str(manifest["repository_commit"])
    actual_tree = (
        _git_bytes(root, "rev-parse", f"{commit}^{{tree}}")
        .decode("ascii")
        .strip()
        .lower()
    )
    if actual_tree != manifest["repository_tree"]:
        raise ValueError("Round 14 captured Git tree differs")
    files = manifest["required_file_sha256"]
    if not isinstance(files, Mapping):
        raise ValueError("Round 14 capture file attestation is unavailable")
    for relative, expected_sha256 in sorted(files.items()):
        committed = _git_bytes(root, "show", f"{commit}:{relative}")
        if hashlib.sha256(committed).hexdigest() != expected_sha256:
            raise ValueError(f"Round 14 captured file bytes differ: {relative}")
    return manifest


__all__ = [
    "POLYMARKET_ROUND14_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_PROSPECTIVE_UNIT_SECONDS",
    "POLYMARKET_ROUND14_QUALIFICATION_MAXIMUM_SECONDS",
    "POLYMARKET_ROUND14_QUALIFICATION_MINIMUM_SECONDS",
    "POLYMARKET_ROUND14_REQUIRED_ASSETS",
    "POLYMARKET_ROUND14_REQUIRED_STREAMS",
    "build_round14_capture_manifest",
    "create_round14_capture_manifest",
    "validate_round14_capture_manifest",
    "verify_round14_repository_attestation",
]
