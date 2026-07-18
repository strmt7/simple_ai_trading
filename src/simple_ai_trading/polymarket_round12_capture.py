"""Git- and hash-bound capture attestation for Polymarket Round 12."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_ROUND12_CAPTURE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round12-capture-preregistration-manifest-v1"
)
_REQUIRED_REPOSITORY_FILES = (
    "src/simple_ai_trading/cli.py",
    "src/simple_ai_trading/polymarket_action_pipeline.py",
    "src/simple_ai_trading/polymarket_action_value.py",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_round12_admission.py",
    "src/simple_ai_trading/polymarket_round12_capture.py",
    "src/simple_ai_trading/polymarket_round12_reference.py",
    "tests/test_polymarket_replay.py",
    "tests/test_polymarket_round12_admission.py",
    "tests/test_polymarket_round12_capture.py",
    "tests/test_polymarket_round12_reference.py",
)


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


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Round 12 capture requires an accessible Git repository"
        ) from exc


def _repository_root() -> Path:
    candidate = Path(__file__).resolve().parent
    root = Path(
        os.fsdecode(_git_bytes(candidate, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if not root.is_dir():
        raise ValueError("Round 12 repository root is unavailable")
    return root


def create_round12_capture_manifest(
    *,
    run_id: str,
    started_at_ms: int,
    repository_commit: str,
    contract_sha256: str,
    model_sha256: str,
    policy_sha256: str,
    reference_implementation_sha256: str,
    action_pipeline_implementation_sha256: str,
    required_git_blob_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Create the canonical manifest after repository evidence is verified."""

    normalized_files = {
        str(path): str(digest) for path, digest in required_git_blob_sha256.items()
    }
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND12_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_ms": int(started_at_ms),
        "repository_commit": str(repository_commit).lower(),
        "contract_sha256": str(contract_sha256),
        "model_sha256": str(model_sha256),
        "policy_sha256": str(policy_sha256),
        "reference_implementation_sha256": str(reference_implementation_sha256),
        "action_pipeline_implementation_sha256": str(
            action_pipeline_implementation_sha256
        ),
        "required_git_blob_sha256": dict(sorted(normalized_files.items())),
        "capture_started_before_manifest": False,
        "labels_consulted": False,
        "model_scores_consulted": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round12_capture_manifest_payload(payload)


def validate_round12_capture_manifest_payload(
    payload: Mapping[str, object],
    *,
    expected_run_id: str | None = None,
    expected_contract: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest = dict(payload)
    claimed = manifest.pop("manifest_sha256", None)
    files = manifest.get("required_git_blob_sha256")
    commit = str(manifest.get("repository_commit") or "")
    if (
        manifest.get("schema_version")
        != POLYMARKET_ROUND12_CAPTURE_MANIFEST_SCHEMA_VERSION
        or not str(manifest.get("run_id") or "")
        or type(manifest.get("created_at_ms")) is not int
        or int(manifest["created_at_ms"]) <= 0
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or not all(
            _is_sha256(manifest.get(field))
            for field in (
                "contract_sha256",
                "model_sha256",
                "policy_sha256",
                "reference_implementation_sha256",
                "action_pipeline_implementation_sha256",
            )
        )
        or not isinstance(files, Mapping)
        or not files
        or any(
            not str(path)
            or Path(str(path)).is_absolute()
            or ".." in Path(str(path)).parts
            or not _is_sha256(digest)
            for path, digest in files.items()
        )
        or manifest.get("capture_started_before_manifest") is not False
        or manifest.get("labels_consulted") is not False
        or manifest.get("model_scores_consulted") is not False
        or manifest.get("paper_trading_authority") is not False
        or manifest.get("live_trading_authority") is not False
        or not _is_sha256(claimed)
        or _canonical_sha256(manifest) != claimed
    ):
        raise ValueError("Round 12 capture manifest is invalid")
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("Round 12 capture manifest binds another recorder run")
    if expected_contract is not None:
        implementation = expected_contract.get("implementation")
        model = expected_contract.get("model_contract")
        policy = expected_contract.get("primary_policy")
        if not all(
            isinstance(value, Mapping) for value in (implementation, model, policy)
        ):
            raise ValueError("Round 12 contract sections are unavailable")
        if (
            manifest["contract_sha256"] != expected_contract.get("contract_sha256")
            or manifest["model_sha256"] != model.get("model_sha256")
            or manifest["policy_sha256"] != policy.get("policy_sha256")
            or manifest["reference_implementation_sha256"]
            != implementation.get("reference_implementation_sha256")
            or manifest["action_pipeline_implementation_sha256"]
            != implementation.get("action_pipeline_implementation_sha256")
        ):
            raise ValueError("Round 12 capture manifest differs from the contract")
    return {**manifest, "manifest_sha256": claimed}


def build_round12_capture_manifest(
    contract_path: str | Path,
    *,
    run_id: str,
    started_at_ms: int,
    additional_required_files: Sequence[str] = (),
) -> dict[str, object]:
    """Attest a clean committed repository before a recorder starts."""

    from .polymarket_action_pipeline import (
        polymarket_action_pipeline_implementation_sha256,
    )
    from .polymarket_round12_reference import load_round12_confirmation_contract

    selected_contract_path = Path(contract_path).resolve()
    contract = load_round12_confirmation_contract(
        selected_contract_path,
        require_current_implementation=True,
    )
    root = _repository_root()
    status = _git_bytes(root, "status", "--porcelain", "--untracked-files=all")
    if status.strip():
        raise ValueError("Round 12 capture requires a clean Git worktree")
    commit = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip().lower()
    try:
        contract_relative = selected_contract_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Round 12 contract is outside the repository") from exc
    predecessor = contract.get("predecessor_evidence")
    if not isinstance(predecessor, Mapping):
        raise ValueError("Round 12 predecessor evidence is unavailable")
    predecessor_relative = (
        (
            selected_contract_path.parent
            / str(predecessor.get("artifact_filename") or "")
        )
        .relative_to(root)
        .as_posix()
    )
    required = tuple(
        dict.fromkeys(
            (
                *_REQUIRED_REPOSITORY_FILES,
                contract_relative,
                predecessor_relative,
                *(str(value) for value in additional_required_files),
            )
        )
    )
    file_hashes: dict[str, str] = {}
    for relative in required:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Round 12 required repository path is unsafe")
        if not (root / path).is_file():
            raise ValueError(
                f"Round 12 required file is unavailable: {path.as_posix()}"
            )
        committed = _git_bytes(root, "show", f"{commit}:{path.as_posix()}")
        file_hashes[path.as_posix()] = hashlib.sha256(committed).hexdigest()
    implementation = contract["implementation"]
    model = contract["model_contract"]
    policy = contract["primary_policy"]
    return create_round12_capture_manifest(
        run_id=run_id,
        started_at_ms=started_at_ms,
        repository_commit=commit,
        contract_sha256=str(contract["contract_sha256"]),
        model_sha256=str(model["model_sha256"]),
        policy_sha256=str(policy["policy_sha256"]),
        reference_implementation_sha256=str(
            implementation["reference_implementation_sha256"]
        ),
        action_pipeline_implementation_sha256=(
            polymarket_action_pipeline_implementation_sha256()
        ),
        required_git_blob_sha256=file_hashes,
    )


def load_round12_capture_manifest(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    contract: Mapping[str, object],
) -> dict[str, object]:
    row = (
        store.connect()
        .execute(
            """
        SELECT r.started_at_ms, m.manifest_json, m.manifest_sha256
        FROM polymarket_recorder_run AS r
        JOIN polymarket_preregistration_manifest AS m USING (run_id)
        WHERE r.run_id = ?
        """,
            [str(run_id)],
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("Round 12 recorder preregistration manifest is missing")
    try:
        decoded = json.loads(
            str(row[1]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Round 12 recorder manifest JSON is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 12 recorder manifest is not an object")
    manifest = validate_round12_capture_manifest_payload(
        decoded,
        expected_run_id=str(run_id),
        expected_contract=contract,
    )
    if (
        manifest["created_at_ms"] != int(row[0])
        or manifest["manifest_sha256"] != str(row[2])
        or _canonical_json(manifest) != str(row[1])
    ):
        raise ValueError("Round 12 recorder manifest persistence differs")
    return manifest


__all__ = [
    "POLYMARKET_ROUND12_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "build_round12_capture_manifest",
    "create_round12_capture_manifest",
    "load_round12_capture_manifest",
    "validate_round12_capture_manifest_payload",
]
