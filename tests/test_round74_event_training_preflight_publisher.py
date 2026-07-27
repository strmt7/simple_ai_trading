from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
TOOL_PATH = REPOSITORY / "tools/publish_round74_event_training_preflight.py"
SPEC = importlib.util.spec_from_file_location(
    "publish_round74_event_training_preflight",
    TOOL_PATH,
)
assert SPEC is not None and SPEC.loader is not None
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)
assert isinstance(PUBLISHER, ModuleType)


def _run() -> dict[str, object]:
    candidate_ids = [
        "event_pooling_linear",
        "event_pooling_mlp",
        "causal_event_tcn",
        "causal_event_attention",
    ]
    losses = dict.fromkeys(candidate_ids, 0.5)
    peers = {candidate_id: [0.4, 0.5, 0.6] for candidate_id in candidate_ids}
    return {
        "schema_version": PUBLISHER.RUN_SCHEMA_VERSION,
        "execution_git_commit": "a" * 40,
        "backend": {
            "requested": "directml",
            "kind": "directml",
            "device": "privateuseone:0",
            "vendor": "test AMD",
            "selection": "runtime_current_device",
            "accelerated": True,
            "torch_version": "test",
            "torch_directml_version": "test",
            "safetensors_version": "test",
            "deterministic_algorithms_requested": True,
            "warning_count": 0,
            "cpu_fallback_warning_count": 0,
        },
        "input_contract": {
            "training_batch_sha256": ["1" * 64, "6" * 64],
            "tuning_batch_sha256": "2" * 64,
            "training_rows": 7,
            "tuning_rows": 2,
            "training_capture_runs": 2,
            "tuning_capture_runs": 1,
            "training_eligible_action_targets": 54,
            "tuning_eligible_action_targets": 15,
            "training_eligible_regime_targets": 28,
            "tuning_eligible_regime_targets": 8,
            "candidate_ids": candidate_ids,
            "candidate_parameter_counts": {
                "event_pooling_linear": 19_900,
                "event_pooling_mlp": 40_228,
                "causal_event_tcn": 130_532,
                "causal_event_attention": 153_532,
            },
            "feature_count": len(PUBLISHER.ROUND74_EVENT_FEATURE_NAMES),
            "feature_names_sha256": (
                PUBLISHER.ROUND74_EVENT_FEATURE_NAMES_SHA256
            ),
            "state_half_lives_seconds": list(
                PUBLISHER.ROUND74_EVENT_STATE_HALF_LIVES_SECONDS
            ),
            "seeds": [7411, 7423, 7433],
            "epochs": 1,
            "minibatch_rows": 2,
            "optimization_population": {
                "unit": "capture_run",
                "optimizer_step": (
                    "one eligible minibatch per training capture run with "
                    "gradient accumulation"
                ),
                "gradient_divisor": "training_capture_run_count",
                "shorter_run_policy": (
                    "deterministic epoch-rotated cycling of eligible minibatches"
                ),
                "fully_censored_minibatches_contribute_gradients": False,
                "fully_censored_capture_run_policy": "reject",
                "row_pooled_optimizer_steps_permitted": False,
            },
        },
        "result": {
            "policy_sha256": "3" * 64,
            "model_sha256": "4" * 64,
            "model_byte_count": 1024,
            "prediction_sha256": "5" * 64,
            "selected_candidate_id": "event_pooling_linear",
            "candidate_run_balanced_tuning_proper_loss": losses,
            "candidate_worst_run_tuning_proper_loss": losses,
            "candidate_pooled_tuning_proper_loss": losses,
            "peer_best_run_balanced_tuning_proper_loss": peers,
            "peer_run_balanced_optimization_schedule": {
                candidate_id: [
                    {
                        "run_count": 2.0,
                        "optimizer_steps": 3.0,
                        "run_contributions_per_optimizer_step": 2.0,
                        "minimum_run_minibatch_contributions": 3.0,
                        "maximum_run_minibatch_contributions": 3.0,
                        "minimum_eligible_minibatches_per_run": 1.0,
                        "maximum_eligible_minibatches_per_run": 3.0,
                    }
                    for _seed in (7411, 7423, 7433)
                ]
                for candidate_id in candidate_ids
            },
            "selection": {
                "criterion": "test complexity gate",
                "selected_candidate_id": "event_pooling_linear",
                "selected_tuning_metrics": {"run_balanced_loss": 0.5},
                "ordered_candidate_ids": candidate_ids,
                "planned_comparison_count": 3,
                "required_paired_capture_run_count": 24,
                "minimum_mean_proper_loss_improvement": 1e-5,
                "maximum_permitted_paired_run_loss_degradation": 1e-5,
                "statistical_independence_or_significance_claim": False,
                "promotion_reports": [
                    {
                        "paired_capture_run_count": 1,
                        "complete_tuning_panel": False,
                        "promoted": False,
                    },
                    {
                        "paired_capture_run_count": 1,
                        "complete_tuning_panel": False,
                        "promoted": False,
                    },
                    {
                        "paired_capture_run_count": 1,
                        "complete_tuning_panel": False,
                        "promoted": False,
                    },
                ],
                "complexity_promotion_privilege": False,
                "backtest_metric_used_for_selection": False,
            },
        },
        "temporary_artifact_count_before_cleanup": 2,
        "temporary_directory_removed": True,
    }


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        PUBLISHER._strict_json('{"value":1,"value":2}')
    with pytest.raises(ValueError, match="non-finite JSON"):
        PUBLISHER._strict_json('{"value":NaN}')


def test_validate_run_accepts_complete_four_candidate_panel() -> None:
    run = json.loads(json.dumps(_run(), sort_keys=True))
    PUBLISHER._validate_run(run, commit="a" * 40)
    assert run["input_contract"]["training_capture_runs"] == 2
    assert run["input_contract"]["training_rows"] == 7


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("kind", "cpu", "backend"),
        ("warning_count", 1, "backend"),
        ("cpu_fallback_warning_count", 1, "backend"),
    ),
)
def test_validate_run_rejects_backend_downgrades(
    field: str,
    value: object,
    message: str,
) -> None:
    run = _run()
    run["backend"][field] = value
    with pytest.raises(RuntimeError, match=message):
        PUBLISHER._validate_run(run, commit="a" * 40)


def test_validate_run_rejects_incomplete_candidate_metrics() -> None:
    run = _run()
    del run["result"]["candidate_pooled_tuning_proper_loss"]["causal_event_tcn"]
    with pytest.raises(RuntimeError, match="panel differs"):
        PUBLISHER._validate_run(run, commit="a" * 40)


def test_validate_run_rejects_unearned_complexity_promotion() -> None:
    run = _run()
    run["result"]["selection"]["promotion_reports"][0]["promoted"] = True
    with pytest.raises(RuntimeError, match="complexity-promotion"):
        PUBLISHER._validate_run(run, commit="a" * 40)


def test_build_evidence_rejects_repeated_run_drift() -> None:
    first = _run()
    second = deepcopy(first)
    second["result"]["policy_sha256"] = "6" * 64
    with pytest.raises(RuntimeError, match="results differ"):
        PUBLISHER._build_evidence(
            REPOSITORY,
            commit="a" * 40,
            first=first,
            second=second,
            supersedes_artifact_sha256="7" * 64,
        )


def test_persisted_artifact_is_hash_bound_and_never_replaced(
    tmp_path: Path,
) -> None:
    run = _run()
    evidence = PUBLISHER._build_evidence(
        REPOSITORY,
        commit="a" * 40,
        first=run,
        second=deepcopy(run),
        supersedes_artifact_sha256="7" * 64,
    )
    output = tmp_path / "evidence.json"
    PUBLISHER._write_new_artifact(output, evidence)

    persisted = json.loads(output.read_text(encoding="ascii"))
    claimed = persisted.pop("artifact_sha256")
    assert claimed == PUBLISHER._canonical_sha256(persisted)
    with pytest.raises(RuntimeError, match="refusing to replace"):
        PUBLISHER._write_new_artifact(output, evidence)
