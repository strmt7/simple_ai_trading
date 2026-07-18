"""Git- and hash-bound capture attestation for Polymarket Round 13."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import TYPE_CHECKING

from .polymarket_round13 import PolymarketRound13Program

if TYPE_CHECKING:
    from .polymarket_recorder import PolymarketEvidenceStore


POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION = (
    "polymarket-round13-capture-preregistration-manifest-v1"
)
POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS = 86_400
_REQUIRED_REPOSITORY_FILES = (
    "native/windows/generated/command_contract.hpp",
    "native/windows/src/main.cpp",
    "src/simple_ai_trading/cli.py",
    "src/simple_ai_trading/command_contract.py",
    "src/simple_ai_trading/duckdb_batch.py",
    "src/simple_ai_trading/paper_execution.py",
    "src/simple_ai_trading/polymarket.py",
    "src/simple_ai_trading/polymarket_action_pipeline.py",
    "src/simple_ai_trading/polymarket_action_value.py",
    "src/simple_ai_trading/polymarket_continuity.py",
    "src/simple_ai_trading/polymarket_coverage.py",
    "src/simple_ai_trading/polymarket_features.py",
    "src/simple_ai_trading/polymarket_recorder.py",
    "src/simple_ai_trading/polymarket_replay.py",
    "src/simple_ai_trading/polymarket_repricing.py",
    "src/simple_ai_trading/polymarket_resolution.py",
    "src/simple_ai_trading/polymarket_round12_admission.py",
    "src/simple_ai_trading/polymarket_round12_reference.py",
    "src/simple_ai_trading/polymarket_round13.py",
    "src/simple_ai_trading/polymarket_round13_capture.py",
    "src/simple_ai_trading/polymarket_round13_evaluation.py",
    "src/simple_ai_trading/polymarket_round13_publication.py",
    "tests/test_polymarket_round13.py",
    "tests/test_polymarket_round13_capture.py",
    "tests/test_polymarket_round13_evaluation.py",
    "tests/test_polymarket_round13_publication.py",
    "tests/test_polymarket_recorder.py",
    "tests/test_polymarket_replay.py",
    "tests/test_polymarket_resolution.py",
    "tests/test_polymarket.py",
    "tests/test_ai_runtime_and_parity.py",
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


def _is_git_object_id(value: object) -> bool:
    text = str(value)
    return len(text) in {40, 64} and all(
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


def _safe_repository_path(value: object) -> bool:
    text = str(value)
    path = PurePosixPath(text)
    return bool(
        text
        and "\\" not in text
        and ":" not in text
        and not path.is_absolute()
        and path.as_posix() == text
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Round 13 capture requires an accessible Git repository"
        ) from exc


def _repository_root() -> Path:
    candidate = Path(__file__).resolve().parent
    root = Path(
        os.fsdecode(_git_bytes(candidate, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if not root.is_dir():
        raise ValueError("Round 13 repository root is unavailable")
    return root


def create_round13_capture_manifest(
    *,
    run_id: str,
    started_at_ms: int,
    capture_duration_seconds: int,
    repository_commit: str,
    repository_tree: str,
    contract_repository_path: str,
    predecessor_repository_path: str,
    contract_sha256: str,
    model_sha256: str,
    policy_sha256: str,
    reference_implementation_sha256: str,
    action_pipeline_implementation_sha256: str,
    round13_program_implementation_sha256: str,
    required_file_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Create one canonical pre-message capture attestation."""

    normalized_files = {
        str(path): str(digest).lower() for path, digest in required_file_sha256.items()
    }
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_ms": int(started_at_ms),
        "capture_duration_seconds": int(capture_duration_seconds),
        "repository_commit": str(repository_commit).lower(),
        "repository_tree": str(repository_tree).lower(),
        "contract_repository_path": str(contract_repository_path),
        "predecessor_repository_path": str(predecessor_repository_path),
        "contract_sha256": str(contract_sha256),
        "model_sha256": str(model_sha256),
        "policy_sha256": str(policy_sha256),
        "reference_implementation_sha256": str(reference_implementation_sha256),
        "action_pipeline_implementation_sha256": str(
            action_pipeline_implementation_sha256
        ),
        "round13_program_implementation_sha256": str(
            round13_program_implementation_sha256
        ),
        "required_file_sha256": dict(sorted(normalized_files.items())),
        "source_hash_algorithm": "sha256_raw_committed_bytes",
        "capture_started_before_manifest": False,
        "outcome_endpoints_queried": False,
        "labels_consulted": False,
        "model_scores_consulted": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return validate_round13_capture_manifest_payload(payload)


def validate_round13_capture_manifest_payload(
    payload: Mapping[str, object],
    *,
    expected_run_id: str | None = None,
    expected_program: PolymarketRound13Program | None = None,
) -> dict[str, object]:
    manifest = dict(payload)
    claimed = manifest.pop("manifest_sha256", None)
    files = manifest.get("required_file_sha256")
    contract_path = manifest.get("contract_repository_path")
    predecessor_path = manifest.get("predecessor_repository_path")
    if (
        manifest.get("schema_version")
        != POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION
        or not str(manifest.get("run_id") or "")
        or type(manifest.get("created_at_ms")) is not int
        or int(manifest["created_at_ms"]) <= 0
        or type(manifest.get("capture_duration_seconds")) is not int
        or manifest.get("capture_duration_seconds")
        != POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS
        or not _is_git_object_id(manifest.get("repository_commit"))
        or not _is_git_object_id(manifest.get("repository_tree"))
        or not _safe_repository_path(contract_path)
        or not _safe_repository_path(predecessor_path)
        or not all(
            _is_sha256(manifest.get(field))
            for field in (
                "contract_sha256",
                "model_sha256",
                "policy_sha256",
                "reference_implementation_sha256",
                "action_pipeline_implementation_sha256",
                "round13_program_implementation_sha256",
            )
        )
        or not isinstance(files, Mapping)
        or not files
        or any(
            not _safe_repository_path(path) or not _is_sha256(digest)
            for path, digest in files.items()
        )
        or not set(_REQUIRED_REPOSITORY_FILES).issubset(files)
        or contract_path not in files
        or predecessor_path not in files
        or manifest.get("source_hash_algorithm") != "sha256_raw_committed_bytes"
        or manifest.get("capture_started_before_manifest") is not False
        or manifest.get("outcome_endpoints_queried") is not False
        or manifest.get("labels_consulted") is not False
        or manifest.get("model_scores_consulted") is not False
        or manifest.get("paper_trading_authority") is not False
        or manifest.get("live_trading_authority") is not False
        or not _is_sha256(claimed)
        or _canonical_sha256(manifest) != claimed
    ):
        raise ValueError("Round 13 capture manifest is invalid")
    if expected_run_id is not None and manifest["run_id"] != expected_run_id:
        raise ValueError("Round 13 capture manifest binds another recorder run")
    if expected_program is not None:
        program = expected_program.validated()
        implementation = program.contract.get("implementation")
        predecessor = program.contract.get("predecessor_evidence")
        if not isinstance(implementation, Mapping) or not isinstance(
            predecessor, Mapping
        ):
            raise ValueError("Round 13 program attestation sections are unavailable")
        if (
            manifest["contract_sha256"] != program.contract_sha256
            or manifest["model_sha256"] != program.model.model_sha256
            or manifest["policy_sha256"] != program.policy.policy_sha256
            or manifest["reference_implementation_sha256"]
            != implementation.get("reference_implementation_sha256")
            or manifest["action_pipeline_implementation_sha256"]
            != implementation.get("action_pipeline_implementation_sha256")
            or manifest["round13_program_implementation_sha256"]
            != implementation.get("round13_program_implementation_sha256")
            or PurePosixPath(str(predecessor_path)).name
            != predecessor.get("artifact_filename")
        ):
            raise ValueError("Round 13 capture manifest differs from the program")
    return {**manifest, "manifest_sha256": claimed}


def build_round13_capture_manifest(
    contract_path: str | Path,
    *,
    run_id: str,
    started_at_ms: int,
    capture_duration_seconds: int,
    additional_required_files: Sequence[str] = (),
) -> dict[str, object]:
    """Attest a clean committed implementation before recorder startup."""

    from .polymarket_action_pipeline import (
        polymarket_action_pipeline_implementation_sha256,
    )
    from .polymarket_round12_reference import (
        polymarket_round12_reference_implementation_sha256,
    )
    from .polymarket_round13 import (
        load_round13_confirmation_contract,
        polymarket_round13_program_implementation_sha256,
    )

    selected_contract_path = Path(contract_path).resolve()
    program = load_round13_confirmation_contract(selected_contract_path)
    root = _repository_root()
    if _git_bytes(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ValueError("Round 13 capture requires a clean Git worktree")
    commit = _git_bytes(root, "rev-parse", "HEAD").decode("ascii").strip().lower()
    tree = _git_bytes(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip().lower()
    try:
        contract_relative = selected_contract_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Round 13 contract is outside the repository") from exc
    predecessor = program.contract.get("predecessor_evidence")
    if not isinstance(predecessor, Mapping):
        raise ValueError("Round 13 predecessor evidence is unavailable")
    predecessor_path = selected_contract_path.parent / str(
        predecessor.get("artifact_filename") or ""
    )
    try:
        predecessor_relative = predecessor_path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Round 13 predecessor is outside the repository") from exc
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
        if not _safe_repository_path(relative):
            raise ValueError("Round 13 required repository path is unsafe")
        path = root / PurePosixPath(relative)
        if not path.is_file():
            raise ValueError(f"Round 13 required file is unavailable: {relative}")
        committed = _git_bytes(root, "show", f"{commit}:{relative}")
        file_hashes[relative] = hashlib.sha256(committed).hexdigest()
    implementation = program.contract["implementation"]
    if not isinstance(implementation, Mapping):
        raise ValueError("Round 13 implementation contract is unavailable")
    return create_round13_capture_manifest(
        run_id=run_id,
        started_at_ms=started_at_ms,
        capture_duration_seconds=capture_duration_seconds,
        repository_commit=commit,
        repository_tree=tree,
        contract_repository_path=contract_relative,
        predecessor_repository_path=predecessor_relative,
        contract_sha256=program.contract_sha256,
        model_sha256=program.model.model_sha256,
        policy_sha256=program.policy.policy_sha256,
        reference_implementation_sha256=(
            polymarket_round12_reference_implementation_sha256()
        ),
        action_pipeline_implementation_sha256=(
            polymarket_action_pipeline_implementation_sha256()
        ),
        round13_program_implementation_sha256=(
            polymarket_round13_program_implementation_sha256()
        ),
        required_file_sha256=file_hashes,
    )


def verify_round13_repository_attestation(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Re-read the captured Git tree and every required committed file."""

    manifest = validate_round13_capture_manifest_payload(payload)
    root = _repository_root()
    commit = str(manifest["repository_commit"])
    expected_tree = str(manifest["repository_tree"])
    actual_tree = (
        _git_bytes(root, "rev-parse", f"{commit}^{{tree}}")
        .decode("ascii")
        .strip()
        .lower()
    )
    if actual_tree != expected_tree:
        raise ValueError("Round 13 captured Git tree differs")
    files = manifest["required_file_sha256"]
    if not isinstance(files, Mapping):
        raise ValueError("Round 13 required file attestation is unavailable")
    for relative, expected_sha256 in sorted(files.items()):
        committed = _git_bytes(root, "show", f"{commit}:{relative}")
        if hashlib.sha256(committed).hexdigest() != expected_sha256:
            raise ValueError(f"Round 13 captured file bytes differ: {relative}")
    return manifest


def load_round13_capture_manifest(
    store: PolymarketEvidenceStore,
    *,
    run_id: str,
    program: PolymarketRound13Program,
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
        raise ValueError("Round 13 recorder preregistration manifest is missing")
    try:
        decoded = json.loads(
            str(row[1]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("Round 13 recorder manifest JSON is invalid") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 13 recorder manifest is not an object")
    manifest = validate_round13_capture_manifest_payload(
        decoded,
        expected_run_id=str(run_id),
        expected_program=program,
    )
    verify_round13_repository_attestation(manifest)
    if (
        manifest["created_at_ms"] != int(row[0])
        or manifest["manifest_sha256"] != str(row[2])
        or _canonical_json(manifest) != str(row[1])
    ):
        raise ValueError("Round 13 recorder manifest persistence differs")
    return manifest


__all__ = [
    "POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS",
    "POLYMARKET_ROUND13_CAPTURE_MANIFEST_SCHEMA_VERSION",
    "build_round13_capture_manifest",
    "create_round13_capture_manifest",
    "load_round13_capture_manifest",
    "validate_round13_capture_manifest_payload",
    "verify_round13_repository_attestation",
]
