from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping

from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_pretraining import (
    ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)


SCHEMA_VERSION = "round-074-event-sequence-model-design-v136"
PREDECESSOR_SCHEMA_VERSION = "round-074-event-sequence-model-design-v135"
EXPECTED_PRETRAINING_SCHEMA = "round-074-causal-next-event-pretraining-v5"
EXPECTED_TRAINING_SCHEMA = "round-074-event-training-v39"
EXPECTED_POLICY_SCHEMA = "round-074-event-pretest-policy-v38"
EXPECTED_MODEL_SCHEMA = "round-074-event-payoff-model-v13"


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    selected: dict[str, object] = {}
    for key, value in pairs:
        if key in selected:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        selected[key] = value
    return selected


def _load_json(path: Path) -> Mapping[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON value is forbidden: {value}")
        ),
    )
    if not isinstance(value, Mapping):
        raise ValueError("model design payload must be an object")
    return value


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def _write_immutable(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable model design differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    selected = completed.stdout.strip().lower()
    if len(selected) != 40 or any(
        value not in "0123456789abcdef" for value in selected
    ):
        raise ValueError("source Git base commit differs")
    return selected


def _source_binding(
    repository: Path, relative_paths: tuple[str, ...]
) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for relative in relative_paths:
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(f"required model source is unavailable: {path}")
        values.append({"path": relative, "sha256": _file_sha256(path)})
    return values


def publish(repository: Path, output: Path) -> Mapping[str, object]:
    if (
        ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION != EXPECTED_PRETRAINING_SCHEMA
        or ROUND74_EVENT_TRAINING_SCHEMA_VERSION != EXPECTED_TRAINING_SCHEMA
        or ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION != EXPECTED_POLICY_SCHEMA
        or ROUND74_EVENT_MODEL_SCHEMA_VERSION != EXPECTED_MODEL_SCHEMA
    ):
        raise RuntimeError("pretraining objective schema contract differs")
    predecessor_path = (
        repository
        / "docs"
        / "model-research"
        / "action-value"
        / "round-074-event-sequence-model-design-v135.json"
    )
    predecessor = dict(_load_json(predecessor_path))
    predecessor_sha256 = str(predecessor.pop("design_sha256", ""))
    if predecessor.get(
        "schema_version"
    ) != PREDECESSOR_SCHEMA_VERSION or predecessor_sha256 != _canonical_sha256(
        predecessor
    ):
        raise ValueError("predecessor model design differs")
    implementation_files = _source_binding(
        repository,
        (
            "src/simple_ai_trading/impact_absorption_event_pretraining.py",
            "src/simple_ai_trading/impact_absorption_event_training.py",
            "tests/test_impact_absorption_event_training.py",
            "tools/publish_round74_pretraining_mask_contract.py",
        ),
    )
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "target_free_pretraining_mask_objective_correction",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "round": 74,
        "source_git_base_commit": _git_head(repository),
        "source_files_are_authoritative_by_sha256": True,
        "source_git_clean_required": False,
        "supersedes_artifact_sha256": predecessor_sha256,
        "source_binding": {
            "file_sha256_normalization": "canonical_lf_bytes",
            "implementation_files": implementation_files,
            "predecessor_design": {
                "path": predecessor_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha256(predecessor_path),
                "design_sha256": predecessor_sha256,
            },
        },
        "schema_contract": {
            "causal_next_event_pretraining": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
            "event_training": ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
            "pretest_policy": ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
            "event_model": ROUND74_EVENT_MODEL_SCHEMA_VERSION,
        },
        "defect": {
            "masked_continuous_features_were_zeroed_before_target_delta_construction": True,
            "masked_dimensions_remained_in_continuous_loss_denominator": True,
            "trivial_zero_targets_diluted_admissible_feature_gradients": True,
            "supervised_market_target_or_execution_objective_defect": False,
        },
        "objective_contract": {
            "model_input_for_masked_feature": "zero_at_every_event",
            "next_event_type_target": "original_observed_event_type",
            "continuous_target": "next_scaled_feature_delta",
            "continuous_loss_dimensions": "unmasked_continuous_features_only",
            "masked_continuous_target_policy": "exclude_from_loss_not_zero_impute",
            "continuous_loss_denominator": "active_event_transition_and_unmasked_dimension_mean",
            "invalid_duplicate_or_out_of_range_mask_policy": "fail_closed_before_tensor_indexing",
            "all_continuous_features_masked_policy": "fail_closed",
            "temporary_pretraining_head_persisted": False,
            "pretraining_uses_supervised_payoff_path_risk_or_execution_targets": False,
        },
        "unchanged_contract": {
            "candidate_panel_changed": False,
            "feature_views_changed": False,
            "capture_population_or_roles_changed": False,
            "training_tuning_calibration_or_test_roles_changed": False,
            "supervised_distributional_loss_changed": False,
            "cost_latency_funding_or_l2_execution_gate_changed": False,
            "complexity_promotion_gate_changed": False,
            "sealed_one_use_test_changed": False,
            "ai_role_changed": False,
            "trading_runtime_or_authority_changed": False,
        },
        "research_basis": [
            {
                "title": "LOBERT: Generative AI Foundation Model for Limit Order Book Messages",
                "url": "https://arxiv.org/abs/2511.12563",
                "use": "Retain message-level next-event self-supervision while preserving continuous price, volume, and time representations.",
            },
            {
                "title": "Self-Supervised Pretext Tasks for Event Sequence Data from Detecting Misalignment",
                "url": "https://openreview.net/forum?id=zc101QzXtw",
                "use": "Keep event-time and event-type self-supervision target-free and separate from downstream financial outcomes.",
            },
            {
                "title": "TLOB: A Novel Transformer Model with Dual Attention for Stock Price Trend Prediction with Limit Order Book Data",
                "url": "https://arxiv.org/abs/2502.15757",
                "use": "Retain simple controls and after-cost complexity promotion because predictive architecture gains can deteriorate after spread costs.",
            },
        ],
        "verification": {
            "focused_command": (
                "python -m pytest tests/test_impact_absorption_event_training.py "
                "-k 'causal_next_event_pretraining or market_state_pretraining or "
                "pretraining_continuous_loss or pretraining_rejects_invalid' -q"
            ),
            "focused_tests_passed": 9,
            "ruff_affected_files_passed": True,
            "representative_market_training_run": False,
            "sealed_test_accessed": False,
        },
        "evidence_boundary": {
            "representative_training_completed": False,
            "predictive_accuracy_established": False,
            "financial_edge_established": False,
            "profitability_established": False,
            "ai_uplift_established": False,
        },
        "authority": {
            "paper_trading": False,
            "testnet_trading": False,
            "live_trading": False,
        },
    }
    value["design_sha256"] = _canonical_sha256(value)
    _write_immutable(output, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/round-074-event-sequence-model-design-v136.json"
        ),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output
    if not output.is_absolute():
        output = repository / output
    value = publish(repository, output.resolve())
    print(json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
