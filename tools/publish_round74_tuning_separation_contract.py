"""Publish the Round 74 disjoint tuning-role contract correction."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess  # nosec B404
import tempfile
from typing import Any


RESEARCH = Path("docs/model-research/action-value")
PREVIOUS = RESEARCH / "round-074-event-sequence-model-design-v61.json"
OUTPUT = RESEARCH / "round-074-event-sequence-model-design-v62.json"
OPERATOR_PATH = "src/simple_ai_trading/round74_event_model_operator.py"
PUBLISHER_PATH = "tools/publish_round74_tuning_separation_contract.py"
NORMALIZATION = "text_bytes_crlf_and_cr_normalized_to_lf_before_sha256"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalized_file_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _strict_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Round 74 artifact has duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"Round 74 artifact contains {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("Round 74 artifact root differs")
    return value


def _load_hash_bound(path: Path) -> dict[str, Any]:
    value = _strict_object(path)
    claimed = str(value.get("design_sha256", ""))
    canonical = dict(value)
    canonical.pop("design_sha256", None)
    if claimed != _canonical_sha256(canonical):
        raise ValueError("Round 74 prior design digest differs")
    return value


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
    )
    return completed.stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    if _git(repository, "status", "--porcelain"):
        raise RuntimeError("Round 74 publisher requires a clean repository")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Round 74 publisher commit identity differs")
    return commit


def _durable_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable Round 74 artifact already exists: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _build(repository: Path, commit: str) -> dict[str, Any]:
    previous = _load_hash_bound(repository / PREVIOUS)
    if previous.get("schema_version") != "round-074-event-sequence-model-design-v61":
        raise ValueError("Round 74 prior model design schema differs")
    dataset = previous.get("dataset_assembly_contract")
    training = previous.get("development_training_contract")
    source = previous.get("source_binding")
    if not all(isinstance(value, dict) for value in (dataset, training, source)):
        raise ValueError("Round 74 prior model contract sections differ")
    assert isinstance(dataset, dict)
    assert isinstance(training, dict)
    assert isinstance(source, dict)
    tuning = dataset.get("tuning_subpartition")
    if (
        not isinstance(tuning, dict)
        or tuning.get("expected_tuning_runs") != 24
        or tuning.get("model_selection_runs") != 12
        or tuning.get("probability_calibration_runs") != 6
        or tuning.get("action_policy_selection_runs") != 6
        or tuning.get("run_reuse_permitted") is not False
        or training.get("complexity_promotion_required_paired_capture_runs") != 24
    ):
        raise ValueError("Round 74 prior tuning-role inconsistency differs")

    value = deepcopy(previous)
    value.pop("design_sha256")
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    value.update(
        {
            "schema_version": "round-074-event-sequence-model-design-v62",
            "frozen_at_utc": now,
            "implementation_git_commit": commit,
            "research_git_basis_commit": commit,
            "supersedes_design_sha256": previous["design_sha256"],
        }
    )
    source = value["source_binding"]
    source.update(
        {
            "event_model_operator_path": OPERATOR_PATH,
            "event_model_operator_sha256": _normalized_file_sha256(
                repository / OPERATOR_PATH
            ),
            "event_model_operator_schema_version": (
                "round-074-event-model-operator-v2"
            ),
            "tuning_role_contract_generator_path": PUBLISHER_PATH,
            "tuning_role_contract_generator_sha256": _normalized_file_sha256(
                repository / PUBLISHER_PATH
            ),
            "file_sha256_normalization": NORMALIZATION,
        }
    )
    dataset = value["dataset_assembly_contract"]
    dataset.update(
        {
            "disjoint_tuning_role_adapter_implemented_now": True,
            "tuning_batch_order_must_equal_subpartition_order": True,
            "tuning_parent_partition_digest_reconciled": True,
            "tuning_scaler_digest_uniformity_required": True,
            "tuning_role_assignment_hash_bound": True,
            "model_selection_receives_only_first_12_tuning_runs": True,
            "probability_calibration_receives_only_next_6_tuning_runs": True,
            "action_policy_selection_receives_only_final_6_tuning_runs": True,
            "test_role_accessed_by_tuning_assignment": False,
        }
    )
    training = value["development_training_contract"]
    training.update(
        {
            "tuning_role_adapter_required_before_candidate_training": True,
            "early_stopping_objective": (
                "mean masked proper probabilistic loss across the 12 "
                "model-selection capture runs with equal run weights"
            ),
            "candidate_selection": (
                "fixed parameter-count order; each more complex challenger "
                "requires all 12 model-selection capture runs, must strictly "
                "exceed the numerical mean-loss floor, and may not degrade "
                "any paired run beyond that same numerical floor"
            ),
            "complexity_promotion_required_paired_capture_runs": 12,
            "candidate_training_may_receive_calibration_runs": False,
            "candidate_training_may_receive_policy_selection_runs": False,
            "model_selection_probability_calibration_or_policy_run_reuse_permitted": (
                False
            ),
            "calibration_and_policy_runs_retained_outside_model_early_stopping": True,
        }
    )
    value["tuning_role_correction"] = {
        "detected_issue": (
            "v61 declared a disjoint 12/6/6 tuning subpartition but still "
            "described candidate selection over all 24 tuning runs"
        ),
        "corrected_policy": (
            "candidate early stopping and complexity promotion use only the "
            "12 model-selection runs; calibration and policy runs remain disjoint"
        ),
        "raw_capture_contract_changed": False,
        "cohort_schedule_changed": False,
        "capture_roles_or_slot_times_changed": False,
        "selected_from_market_targets_model_results_or_profitability": False,
        "representative_market_training_completed": False,
        "financial_edge_established": False,
        "profitability_claim": False,
    }
    authority = value["authority"]
    authority["disjoint_tuning_role_adapter_implementation"] = True
    authority["representative_market_training"] = False
    authority["model_selection"] = False
    authority["profitability_claim"] = False
    value["design_sha256"] = _canonical_sha256(value)
    return value


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    commit = _require_clean_repository(repository)
    output = repository / OUTPUT
    value = _build(repository, commit)
    _durable_exclusive_json(output, value)
    print(
        json.dumps(
            {
                "output": str(OUTPUT).replace("\\", "/"),
                "design_sha256": value["design_sha256"],
                "implementation_git_commit": commit,
                "model_selection_runs": 12,
                "calibration_runs": 6,
                "policy_selection_runs": 6,
                "profitability_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
