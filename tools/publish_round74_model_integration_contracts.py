"""Publish source-bound Round 74 model and AI integration contracts."""

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


REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH = Path("docs/model-research/action-value")
PREVIOUS_MODEL = RESEARCH / "round-074-event-sequence-model-design-v62.json"
MODEL_OUTPUT = RESEARCH / "round-074-event-sequence-model-design-v63.json"
PREVIOUS_AI = RESEARCH / "round-074-local-ai-review-design-v48.json"
AI_OUTPUT = RESEARCH / "round-074-local-ai-review-design-v49.json"
OPERATOR_PATH = "src/simple_ai_trading/round74_event_model_operator.py"
MODEL_PATH = "src/simple_ai_trading/impact_absorption_event_model.py"
TRAINING_PATH = "src/simple_ai_trading/impact_absorption_event_training.py"
PUBLISHER_PATH = "tools/publish_round74_model_integration_contracts.py"
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
                raise ValueError(f"{path.name} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"{path.name} contains {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} root differs")
    return value


def _load_hash_bound(path: Path, digest_key: str) -> dict[str, Any]:
    value = _strict_object(path)
    claimed = value.pop(digest_key, None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(value):
        raise ValueError(f"{path.name} hash binding differs")
    value[digest_key] = claimed
    return value


def _git(*arguments: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", *arguments],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
        errors="strict",
        timeout=30,
    )
    return completed.stdout.strip()


def _require_clean_repository() -> str:
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Round 74 integration publication requires a clean worktree")
    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Round 74 integration commit identity differs")
    return commit


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True).encode("ascii") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable Round 74 artifact exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _source_binding() -> dict[str, object]:
    return {
        "event_model_operator_path": OPERATOR_PATH,
        "event_model_operator_sha256": _normalized_file_sha256(
            REPOSITORY / OPERATOR_PATH
        ),
        "event_model_operator_schema_version": "round-074-event-model-operator-v3",
        "event_model_path": MODEL_PATH,
        "event_model_sha256": _normalized_file_sha256(REPOSITORY / MODEL_PATH),
        "event_training_path": TRAINING_PATH,
        "event_training_sha256": _normalized_file_sha256(REPOSITORY / TRAINING_PATH),
        "event_training_schema_version": "round-074-event-training-v13",
        "pretest_policy_schema_version": "round-074-event-pretest-policy-v12",
        "model_integration_contract_generator_path": PUBLISHER_PATH,
        "model_integration_contract_generator_sha256": _normalized_file_sha256(
            REPOSITORY / PUBLISHER_PATH
        ),
        "file_sha256_normalization": NORMALIZATION,
    }


def _build_model(commit: str) -> dict[str, Any]:
    previous = _load_hash_bound(REPOSITORY / PREVIOUS_MODEL, "design_sha256")
    if previous.get("schema_version") != "round-074-event-sequence-model-design-v62":
        raise ValueError("Round 74 prior model design schema differs")
    value = deepcopy(previous)
    value.pop("design_sha256")
    value.update(
        {
            "schema_version": "round-074-event-sequence-model-design-v63",
            "frozen_at_utc": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "implementation_git_commit": commit,
            "research_git_basis_commit": commit,
            "supersedes_design_sha256": previous["design_sha256"],
        }
    )
    value["source_binding"].update(_source_binding())
    value["representative_window_selection_contract"] = {
        "schema_version": "round-074-target-blind-window-selection-v1",
        "implemented_now": True,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "temporal_strata_per_symbol": 16,
        "windows_per_symbol_per_stratum": 16,
        "windows_per_symbol_per_capture_run": 256,
        "windows_per_capture_run": 768,
        "selection": "exact bottom SHA-256 rank within each symbol and time stratum",
        "ranking_inputs": [
            "dataset schema",
            "capture run identity",
            "symbol",
            "temporal stratum",
            "decision monotonic nanoseconds",
            "feature window SHA-256",
        ],
        "realized_targets_model_outputs_or_profitability_used": False,
        "underfilled_symbol_or_stratum_policy": "reject",
        "output_order": "chronological after selection",
        "full_window_materialization_permitted": False,
    }
    value["device_execution_contract"] = {
        "implemented_now": True,
        "default_device_run_group_size": 8,
        "maximum_device_run_group_size": 32,
        "training_capture_runs": 120,
        "model_selection_capture_runs": 12,
        "probability_calibration_capture_runs": 6,
        "action_policy_selection_capture_runs": 6,
        "candidate_fit_may_access_calibration_or_policy_runs": False,
        "optimizer_forwards_per_step_at_default_group_size": 15,
        "ungrouped_optimizer_forwards_per_step": 120,
        "optimizer_forward_reduction_fraction": 0.875,
        "equal_capture_run_loss_normalization_preserved": True,
        "gradient_divisor": "training_capture_run_count",
        "host_to_device_tensor_transfers_per_group": 7,
        "per_parameter_gradient_sync_before_clip_permitted": False,
        "gradient_nonfinite_boundary": "single clipped gradient norm per optimizer step",
        "parameter_nonfinite_boundary": "single stacked device check per epoch",
        "dropout_rng_is_bound_by_fixed_group_size": True,
        "cross_platform_bitwise_reproducibility_claim": False,
    }
    value["development_memory_contract"] = {
        "development_capture_runs": 144,
        "representative_windows": 110592,
        "feature_sequence_length": 128,
        "feature_count": 66,
        "feature_dtype": "float32",
        "feature_tensor_upper_bound_bytes": 3737124864,
        "feature_tensor_upper_bound_gib": 3.48046875,
        "overlapping_feature_or_target_cache_written_to_disk": False,
        "all_window_feature_tensor_materialization_permitted": False,
        "target_and_runtime_overhead_excluded_from_feature_only_bound": True,
    }
    training = value["development_training_contract"]
    training.update(
        {
            "training_capture_runs": 120,
            "model_selection_capture_runs": 12,
            "probability_calibration_capture_runs": 6,
            "action_policy_selection_capture_runs": 6,
            "representative_window_policy_required_in_cohort_mode": True,
            "partial_cohort_training_permitted": False,
            "preflight_mode_may_claim_representative_market_evidence": False,
            "directml_run_group_size": 8,
            "run_group_gradient_equivalence_control_tested": True,
        }
    )
    value["model_integration_correction"] = {
        "target_blind_representative_sampler_implemented": True,
        "cohort_population_enforced_in_production_mode": True,
        "disjoint_tuning_role_adapter_connected_to_training": True,
        "directml_grouped_training_and_evaluation_implemented": True,
        "synthetic_directml_preflight_passed_on_amd": True,
        "representative_market_training_completed": False,
        "sealed_test_evaluated": False,
        "financial_edge_established": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
    }
    authority = value["authority"]
    authority["representative_window_sampler_implementation"] = True
    authority["grouped_directml_training_implementation"] = True
    authority["representative_market_training"] = False
    authority["model_selection"] = False
    authority["profitability_claim"] = False
    value["design_sha256"] = _canonical_sha256(value)
    return value


def _build_ai(commit: str, model: dict[str, Any]) -> dict[str, Any]:
    previous = _load_hash_bound(REPOSITORY / PREVIOUS_AI, "artifact_sha256")
    if previous.get("schema_version") != "round-074-local-ai-review-design-v48":
        raise ValueError("Round 74 prior AI design schema differs")
    value = deepcopy(previous)
    value.pop("artifact_sha256")
    value.update(
        {
            "schema_version": "round-074-local-ai-review-design-v49",
            "researched_at_utc": datetime.now(UTC)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "implementation_git_commit": commit,
            "research_git_basis_commit": commit,
            "supersedes_artifact_sha256": previous["artifact_sha256"],
        }
    )
    model_path = str(MODEL_OUTPUT).replace("\\", "/")
    source = value["source_binding"]
    source.update(
        {
            "event_model_design_path": model_path,
            "event_model_design_file_sha256": _normalized_file_sha256(
                REPOSITORY / MODEL_OUTPUT
            ),
            "event_model_design_sha256": model["design_sha256"],
            **_source_binding(),
        }
    )
    architecture = value["architecture"]
    architecture.pop(
        "ml_candidate_selection_requires_complete_24_run_tuning_panel",
        None,
    )
    architecture.update(
        {
            "ml_candidate_selection_requires_complete_12_run_model_selection_panel": (
                True
            ),
            "ml_probability_calibration_uses_disjoint_6_run_panel": True,
            "ml_action_policy_selection_uses_disjoint_6_run_panel": True,
            "ml_target_blind_representative_window_sampler_implemented": True,
            "ml_representative_windows_per_capture_run": 768,
            "ml_default_directml_run_group_size": 8,
            "ml_grouped_forward_preserves_equal_capture_run_objective": True,
            "ml_candidate_fit_may_access_ai_outputs": False,
        }
    )
    value["model_integration"] = {
        "event_model_design_sha256": model["design_sha256"],
        "model_selection_precedes_ai_review": True,
        "ai_receives_target_free_preselected_ml_candidates_only": True,
        "ai_may_increase_risk_select_side_set_leverage_or_submit_orders": False,
        "representative_market_ml_training_completed": False,
        "representative_market_ai_evaluation_completed": False,
        "financial_edge_established": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
    }
    value["authority"]["representative_market_ai_evaluation"] = False
    value["authority"]["financial_edge_established"] = False
    value["authority"]["profitability_claim"] = False
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def main() -> int:
    commit = _require_clean_repository()
    model = _build_model(commit)
    _write_immutable(REPOSITORY / MODEL_OUTPUT, model)
    ai = _build_ai(commit, model)
    _write_immutable(REPOSITORY / AI_OUTPUT, ai)
    print(
        json.dumps(
            {
                "model_output": str(MODEL_OUTPUT).replace("\\", "/"),
                "model_design_sha256": model["design_sha256"],
                "ai_output": str(AI_OUTPUT).replace("\\", "/"),
                "ai_artifact_sha256": ai["artifact_sha256"],
                "implementation_git_commit": commit,
                "representative_market_training_completed": False,
                "profitability_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
