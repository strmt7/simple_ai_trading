"""Publish Round 74 multi-timescale model and cohort contract supersessions."""

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

from simple_ai_trading.impact_absorption_event_action_policy import (
    ROUND74_ACTION_CONTEXT_SCHEMA_VERSION,
    ROUND74_ACTION_POLICY_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_cohort import (
    ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    ROUND74_EVENT_DATASET_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_scaling import (
    ROUND74_EVENT_SCALER_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
    ROUND74_EVENT_STATE_HALF_LIVES_SECONDS,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)


RESEARCH = Path("docs/model-research/action-value")
PREVIOUS_MODEL_DESIGN = RESEARCH / "round-074-event-sequence-model-design-v58.json"
PREVIOUS_AI_DESIGN = RESEARCH / "round-074-local-ai-review-design-v44.json"
PREVIOUS_OPERATOR = RESEARCH / "round-074-event-cohort-operator-v4.json"
TRAINING_PREFLIGHT = (
    RESEARCH
    / "round-074-event-training-directml-preflight-multi-timescale-v9-2026-07-27.json"
)
MODEL_DESIGN_OUTPUT = RESEARCH / "round-074-event-sequence-model-design-v59.json"
AI_DESIGN_OUTPUT = RESEARCH / "round-074-local-ai-review-design-v45.json"
OPERATOR_OUTPUT = RESEARCH / "round-074-event-cohort-operator-v5.json"
OPERATOR_SUPERSESSION_OUTPUT = (
    RESEARCH / "round-074-event-cohort-operator-v4-supersession-2026-07-27.json"
)
GENERATOR_PATH = "tools/publish_round74_multiscale_contracts.py"
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


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key rejected in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant rejected: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _load_hash_bound(path: Path, field: str) -> dict[str, Any]:
    value = _strict_json(path)
    claimed = value.pop(field, None)
    if claimed != _canonical_sha256(value):
        raise ValueError(f"artifact digest differs: {path}")
    value[field] = claimed
    return value


def _file_sha256(path: Path, *, normalize_text: bool = True) -> str:
    payload = path.read_bytes()
    if normalize_text:
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(  # nosec B603
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
        errors="strict",
    )
    return result.stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("contract publication requires a clean repository")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != 40 or any(value not in "0123456789abcdef" for value in commit):
        raise RuntimeError("contract publication Git identity differs")
    return commit


def _write_new(path: Path, value: dict[str, Any], hash_field: str) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace immutable artifact: {path}")
    value[hash_field] = _canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (
                    json.dumps(
                        value,
                        allow_nan=False,
                        ensure_ascii=True,
                        indent=2,
                    )
                    + "\n"
                ).encode("ascii")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _load_hash_bound(path, hash_field)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _source_hash(repository: Path, relative_path: str) -> str:
    return _file_sha256(repository / relative_path)


def _parameter_counts(preflight: dict[str, Any]) -> dict[str, int]:
    inputs = preflight["input_contract"]
    counts = inputs["candidate_parameter_counts"]
    candidate_ids = list(ROUND74_EVENT_MODEL_CANDIDATES)
    if (
        inputs["candidate_ids"] != candidate_ids
        or inputs["feature_count"] != len(ROUND74_EVENT_FEATURE_NAMES)
        or inputs["feature_names_sha256"] != ROUND74_EVENT_FEATURE_NAMES_SHA256
        or inputs["state_half_lives_seconds"]
        != list(ROUND74_EVENT_STATE_HALF_LIVES_SECONDS)
        or not isinstance(counts, dict)
        or set(counts) != set(candidate_ids)
    ):
        raise ValueError("multi-timescale preflight input contract differs")
    selected = {
        candidate_id: int(counts[candidate_id]) for candidate_id in candidate_ids
    }
    if any(
        later <= earlier
        for earlier, later in zip(
            selected.values(),
            tuple(selected.values())[1:],
        )
    ):
        raise ValueError("candidate parameter complexity order differs")
    return selected


def _operator_contract(
    repository: Path,
    *,
    commit: str,
    now: str,
    previous: dict[str, Any],
) -> dict[str, Any]:
    value = deepcopy(previous)
    previous_sha256 = str(value.pop("artifact_sha256"))
    value["schema_version"] = "round-074-event-cohort-operator-contract-v5"
    value["created_at_utc"] = now
    value["source_parent_git_commit"] = commit
    value["superseded_operator_contract_artifact_sha256"] = previous_sha256
    value["source_binding"] = {
        "operator_path": "src/simple_ai_trading/round74_event_cohort_operator.py",
        "operator_sha256": _file_sha256(
            repository / "src/simple_ai_trading/round74_event_cohort_operator.py",
            normalize_text=False,
        ),
        "wrapper_path": "tools/run_round74_event_cohort_slot.py",
        "wrapper_sha256": _file_sha256(
            repository / "tools/run_round74_event_cohort_slot.py",
            normalize_text=False,
        ),
    }
    partition = value["partition_contract"]
    partition["dataset_schema_version"] = ROUND74_EVENT_DATASET_SCHEMA_VERSION
    partition["event_sequence_schema_version"] = ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION
    partition["event_scaler_schema_version"] = ROUND74_EVENT_SCALER_SCHEMA_VERSION
    partition["feature_count"] = len(ROUND74_EVENT_FEATURE_NAMES)
    partition["feature_names_sha256"] = ROUND74_EVENT_FEATURE_NAMES_SHA256
    partition["state_half_lives_seconds"] = list(ROUND74_EVENT_STATE_HALF_LIVES_SECONDS)
    value["supersession_scope"] = {
        "raw_capture_plan_changed": False,
        "slot_schedule_or_roles_changed": False,
        "capture_executable_changed": False,
        "downstream_feature_and_dataset_contract_changed": True,
        "selected_from_market_model_or_target_outcome": False,
        "slot_zero_started_when_superseded": False,
        "cohort_market_data_collected_when_superseded": False,
    }
    return value


def _operator_supersession(
    *,
    now: str,
    previous: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "round-074-event-cohort-operator-supersession-v4",
        "observed_at_utc": now,
        "cohort_plan_sha256": replacement["cohort_plan_sha256"],
        "superseded_operator": {
            "path": str(PREVIOUS_OPERATOR).replace("\\", "/"),
            "artifact_sha256": previous["artifact_sha256"],
            "dataset_schema_version": previous["partition_contract"][
                "dataset_schema_version"
            ],
        },
        "replacement_operator": {
            "path": str(OPERATOR_OUTPUT).replace("\\", "/"),
            "artifact_sha256": replacement["artifact_sha256"],
            "dataset_schema_version": replacement["partition_contract"][
                "dataset_schema_version"
            ],
        },
        "correction_basis": {
            "reason": (
                "Bind the prospective post-capture pipeline to the causal "
                "multi-timescale feature and dataset schemas before slot zero."
            ),
            "raw_capture_plan_changed": False,
            "task_schedule_changed": False,
            "roles_or_slot_times_changed": False,
            "selected_from_market_model_or_target_outcome": False,
            "slot_zero_started": False,
            "cohort_market_data_collected": False,
        },
        "authority": {
            "credentials_used": False,
            "orders_submitted": False,
            "model_training_or_evaluation_performed": False,
            "profitability_or_edge_claim": False,
        },
    }


def _update_model_design(
    repository: Path,
    *,
    commit: str,
    now: str,
    previous: dict[str, Any],
    preflight: dict[str, Any],
    operator: dict[str, Any],
    operator_supersession: dict[str, Any],
    counts: dict[str, int],
) -> dict[str, Any]:
    value = deepcopy(previous)
    value.pop("design_sha256")
    value["schema_version"] = "round-074-event-sequence-model-design-v59"
    value["frozen_at_utc"] = now
    value["implementation_git_commit"] = commit
    value["research_git_basis_commit"] = commit
    value["supersedes_design_sha256"] = previous["design_sha256"]
    source = value["source_binding"]
    changed_paths = {
        "event_sequence": "src/simple_ai_trading/impact_absorption_event_sequence.py",
        "event_model": "src/simple_ai_trading/impact_absorption_event_model.py",
        "event_scaler": "src/simple_ai_trading/impact_absorption_event_scaling.py",
        "event_dataset": "src/simple_ai_trading/impact_absorption_event_dataset.py",
        "event_action_policy": (
            "src/simple_ai_trading/impact_absorption_event_action_policy.py"
        ),
        "event_training": "src/simple_ai_trading/impact_absorption_event_training.py",
    }
    for name, path in changed_paths.items():
        source[f"{name}_path"] = path
        source[f"{name}_sha256"] = _source_hash(repository, path)
    source.update(
        {
            "event_sequence_schema_version": ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
            "event_model_schema_version": ROUND74_EVENT_MODEL_SCHEMA_VERSION,
            "event_scaler_schema_version": ROUND74_EVENT_SCALER_SCHEMA_VERSION,
            "event_dataset_schema_version": ROUND74_EVENT_DATASET_SCHEMA_VERSION,
            "event_action_context_schema_version": (
                ROUND74_ACTION_CONTEXT_SCHEMA_VERSION
            ),
            "event_action_policy_schema_version": ROUND74_ACTION_POLICY_SCHEMA_VERSION,
            "event_cohort_plan_schema_version": (
                ROUND74_EVENT_COHORT_PLAN_SCHEMA_VERSION
            ),
            "event_training_schema_version": ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
            "pretest_policy_schema_version": (
                ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
            ),
            "feature_count": len(ROUND74_EVENT_FEATURE_NAMES),
            "feature_names_sha256": ROUND74_EVENT_FEATURE_NAMES_SHA256,
            "contract_generator_path": GENERATOR_PATH,
            "contract_generator_sha256": _source_hash(repository, GENERATOR_PATH),
        }
    )
    host = value["host_evidence_binding"]
    preflight_path = str(TRAINING_PREFLIGHT).replace("\\", "/")
    host.update(
        {
            "event_sequence_replay_current_source_bound": False,
            "event_sequence_replay_reuse_scope": (
                "Historical exact-wire and read-only replay evidence for "
                "sequence-v2 only; it does not exercise or validate the "
                "current sequence-v3 multi-timescale feature state."
            ),
            "event_training_directml_path": preflight_path,
            "event_training_directml_file_sha256": _source_hash(
                repository, preflight_path
            ),
            "event_training_directml_artifact_sha256": preflight["artifact_sha256"],
            "event_training_directml_execution_git_commit": preflight[
                "execution_git_commit"
            ],
            "event_training_directml_model_source_bound": True,
            "event_training_directml_training_source_bound": True,
            "event_training_directml_feature_count": len(ROUND74_EVENT_FEATURE_NAMES),
            "event_training_directml_feature_names_sha256": (
                ROUND74_EVENT_FEATURE_NAMES_SHA256
            ),
            "event_training_directml_state_half_lives_seconds": list(
                ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
            ),
            "event_training_directml_candidate_parameter_counts": counts,
            "event_training_directml_runner_sha256": preflight["source_binding"][
                "preflight_runner_sha256"
            ],
            "event_training_directml_publisher_sha256": preflight["source_binding"][
                "publisher_sha256"
            ],
            "event_training_directml_dataset_legacy_checkout_sha256": preflight[
                "source_binding"
            ]["event_dataset_sha256"],
            "event_training_directml_dataset_canonical_sha256": source[
                "event_dataset_sha256"
            ],
            "event_training_directml_reuse_scope": (
                "Two fresh-process AMD DirectML executions bind sequence-v3, "
                "66 causal features, 5/30/300-second state, model-v5, "
                "training-v11, pretest-policy-v10, dataset-v8, and all four "
                "candidates. The constructed unequal training runs prove equal "
                "capture-run gradient weight; this is not market fit, edge, AI "
                "uplift, or profitability evidence."
            ),
        }
    )
    features = value["causal_feature_contract"]
    features.update(
        {
            "absolute_top_20_bid_and_ask_quote_depth_retained": True,
            "continuous_time_state_half_lives_seconds": list(
                ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
            ),
            "continuous_time_decay_uses_receipt_nanoseconds": True,
            "fixed_event_rate_assumption_permitted": False,
            "state_warmed_by_pre_feature_ready_receipts": True,
            "state_inputs": [
                "mid log return",
                "signed aggressive trade pressure",
                "signed displayed-depth pressure",
                "signed liquidation pressure",
                "spread",
                "L1 imbalance",
            ],
            "state_future_receipt_or_target_access": False,
        }
    )
    for candidate_id, parameter_count in counts.items():
        value["candidate_panel"][candidate_id]["parameter_count"] = parameter_count
    cohort = value["cohort_admission_contract"]
    cohort.update(
        {
            "plan_path": (
                "docs/model-research/action-value/round-074-event-cohort-plan-v4.json"
            ),
            "plan_sha256": operator["cohort_plan_sha256"],
            "plan_implementation_git_commit": "6efe56902524941c35654902b889ad85cd50f1b1",
            "operator_contract_artifact_sha256": operator["artifact_sha256"],
            "operator_contract_path": str(OPERATOR_OUTPUT).replace("\\", "/"),
            "operator_supersession_artifact_sha256": operator_supersession[
                "artifact_sha256"
            ],
            "operator_supersession_path": str(OPERATOR_SUPERSESSION_OUTPUT).replace(
                "\\", "/"
            ),
            "raw_capture_schedule_changed_for_multiscale_features": False,
            "post_capture_dataset_schema_version": (
                ROUND74_EVENT_DATASET_SCHEMA_VERSION
            ),
        }
    )
    value["authority"]["continuous_time_multiscale_feature_implementation"] = True
    value["authority"]["representative_market_training"] = False
    return value


def _update_ai_design(
    repository: Path,
    *,
    commit: str,
    now: str,
    previous: dict[str, Any],
    model_design: dict[str, Any],
    preflight: dict[str, Any],
    counts: dict[str, int],
) -> dict[str, Any]:
    value = deepcopy(previous)
    value.pop("artifact_sha256")
    value["schema_version"] = "round-074-local-ai-review-design-v45"
    value["researched_at_utc"] = now
    value["implementation_git_commit"] = commit
    value["research_git_basis_commit"] = commit
    value["supersedes_artifact_sha256"] = previous["artifact_sha256"]
    source = value["source_binding"]
    model_design_path = str(MODEL_DESIGN_OUTPUT).replace("\\", "/")
    source.update(
        {
            "event_sequence_path": (
                "src/simple_ai_trading/impact_absorption_event_sequence.py"
            ),
            "event_sequence_sha256": _source_hash(
                repository,
                "src/simple_ai_trading/impact_absorption_event_sequence.py",
            ),
            "event_sequence_schema_version": ROUND74_EVENT_SEQUENCE_SCHEMA_VERSION,
            "event_scaler_path": (
                "src/simple_ai_trading/impact_absorption_event_scaling.py"
            ),
            "event_scaler_sha256": _source_hash(
                repository,
                "src/simple_ai_trading/impact_absorption_event_scaling.py",
            ),
            "event_scaler_schema_version": ROUND74_EVENT_SCALER_SCHEMA_VERSION,
            "event_dataset_path": (
                "src/simple_ai_trading/impact_absorption_event_dataset.py"
            ),
            "event_dataset_sha256": _source_hash(
                repository,
                "src/simple_ai_trading/impact_absorption_event_dataset.py",
            ),
            "event_dataset_schema_version": ROUND74_EVENT_DATASET_SCHEMA_VERSION,
            "event_model_sha256": _source_hash(
                repository,
                "src/simple_ai_trading/impact_absorption_event_model.py",
            ),
            "event_training_sha256": _source_hash(
                repository,
                "src/simple_ai_trading/impact_absorption_event_training.py",
            ),
            "action_policy_sha256": _source_hash(
                repository,
                "src/simple_ai_trading/impact_absorption_event_action_policy.py",
            ),
            "event_model_design_path": model_design_path,
            "event_model_design_file_sha256": _source_hash(
                repository, model_design_path
            ),
            "event_model_design_sha256": model_design["design_sha256"],
            "event_model_schema_version": ROUND74_EVENT_MODEL_SCHEMA_VERSION,
            "event_training_schema_version": ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
            "pretest_policy_schema_version": (
                ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
            ),
            "action_context_schema_version": ROUND74_ACTION_CONTEXT_SCHEMA_VERSION,
            "action_policy_schema_version": ROUND74_ACTION_POLICY_SCHEMA_VERSION,
            "feature_count": len(ROUND74_EVENT_FEATURE_NAMES),
            "feature_names_sha256": ROUND74_EVENT_FEATURE_NAMES_SHA256,
            "state_half_lives_seconds": list(ROUND74_EVENT_STATE_HALF_LIVES_SECONDS),
            "contract_generator_path": GENERATOR_PATH,
            "contract_generator_sha256": _source_hash(repository, GENERATOR_PATH),
        }
    )
    compute = value["model_compute_evidence_binding"]
    preflight_path = str(TRAINING_PREFLIGHT).replace("\\", "/")
    compute.update(
        {
            "path": preflight_path,
            "file_sha256": _source_hash(repository, preflight_path),
            "artifact_sha256": preflight["artifact_sha256"],
            "execution_git_commit": preflight["execution_git_commit"],
            "candidate_parameter_counts": counts,
            "feature_count": len(ROUND74_EVENT_FEATURE_NAMES),
            "feature_names_sha256": ROUND74_EVENT_FEATURE_NAMES_SHA256,
            "state_half_lives_seconds": list(ROUND74_EVENT_STATE_HALF_LIVES_SECONDS),
        }
    )
    architecture = value["architecture"]
    architecture.update(
        {
            "ml_attention_candidate_parameter_count": counts["causal_event_attention"],
            "ml_continuous_time_multiscale_state_implemented": True,
            "ml_multiscale_state_half_lives_seconds": list(
                ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
            ),
            "ml_multiscale_state_future_or_target_access": False,
            "ai_receives_target_free_multiscale_candidate_context_only": True,
            "multiscale_directml_preflight_completed": True,
        }
    )
    value["status"]["representative_market_ai_evaluation_completed"] = False
    value["status"]["ai_uplift_established"] = False
    value["status"]["profitability_claim"] = False
    return value


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    commit = _require_clean_repository(repository)
    now = _now()
    previous_operator = _load_hash_bound(
        repository / PREVIOUS_OPERATOR,
        "artifact_sha256",
    )
    previous_model = _load_hash_bound(
        repository / PREVIOUS_MODEL_DESIGN,
        "design_sha256",
    )
    previous_ai = _load_hash_bound(
        repository / PREVIOUS_AI_DESIGN,
        "artifact_sha256",
    )
    preflight = _load_hash_bound(
        repository / TRAINING_PREFLIGHT,
        "artifact_sha256",
    )
    counts = _parameter_counts(preflight)
    operator = _operator_contract(
        repository,
        commit=commit,
        now=now,
        previous=previous_operator,
    )
    _write_new(repository / OPERATOR_OUTPUT, operator, "artifact_sha256")
    supersession = _operator_supersession(
        now=now,
        previous=previous_operator,
        replacement=operator,
    )
    _write_new(
        repository / OPERATOR_SUPERSESSION_OUTPUT,
        supersession,
        "artifact_sha256",
    )
    model_design = _update_model_design(
        repository,
        commit=commit,
        now=now,
        previous=previous_model,
        preflight=preflight,
        operator=operator,
        operator_supersession=supersession,
        counts=counts,
    )
    _write_new(repository / MODEL_DESIGN_OUTPUT, model_design, "design_sha256")
    ai_design = _update_ai_design(
        repository,
        commit=commit,
        now=now,
        previous=previous_ai,
        model_design=model_design,
        preflight=preflight,
        counts=counts,
    )
    _write_new(repository / AI_DESIGN_OUTPUT, ai_design, "artifact_sha256")
    print(
        json.dumps(
            {
                "ai_design_sha256": ai_design["artifact_sha256"],
                "execution_git_commit": commit,
                "model_design_sha256": model_design["design_sha256"],
                "operator_contract_sha256": operator["artifact_sha256"],
                "operator_supersession_sha256": supersession["artifact_sha256"],
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
