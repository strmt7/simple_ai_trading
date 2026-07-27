"""Publish repeated Round 74 DirectML trainer preflight evidence.

The publisher runs the constructed-tensor preflight twice in fresh processes.
It proves only compute, serialization, cleanup, and deterministic-governance
properties. It does not use market data and cannot establish financial edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simple_ai_trading.impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
)
from simple_ai_trading.impact_absorption_event_training import (
    ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
    ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
)


EVIDENCE_SCHEMA_VERSION = "round-074-event-training-directml-preflight-evidence-v14"
RUN_SCHEMA_VERSION = "round-074-event-training-preflight-run-v3"
SOURCE_PATHS = {
    "event_sequence": "src/simple_ai_trading/impact_absorption_event_sequence.py",
    "event_scaling": "src/simple_ai_trading/impact_absorption_event_scaling.py",
    "event_targets": "src/simple_ai_trading/impact_absorption_event_targets.py",
    "event_dataset": "src/simple_ai_trading/impact_absorption_event_dataset.py",
    "event_model": "src/simple_ai_trading/impact_absorption_event_model.py",
    "event_training": "src/simple_ai_trading/impact_absorption_event_training.py",
    "event_cohort": "src/simple_ai_trading/impact_absorption_event_cohort.py",
    "storage": "src/simple_ai_trading/storage.py",
    "preflight_runner": "tools/run_round74_event_training_preflight.py",
    "publisher": "tools/publish_round74_event_training_preflight.py",
}


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant rejected: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _strict_json(raw: str) -> dict[str, Any]:
    value = json.loads(
        raw,
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("preflight output must be a JSON object")
    return value


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
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("Round 74 evidence publication requires a clean worktree")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Round 74 evidence git identity differs")
    return commit


def _run_preflight(repository: Path) -> dict[str, Any]:
    runner = repository / SOURCE_PATHS["preflight_runner"]
    completed = subprocess.run(  # nosec B603
        [sys.executable, str(runner), "--repository", str(repository)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        encoding="ascii",
        errors="strict",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Round 74 DirectML preflight failed with exit "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    if completed.stderr.strip():
        raise RuntimeError("Round 74 DirectML preflight emitted stderr")
    return _strict_json(completed.stdout)


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"{field} is not a lowercase SHA-256 digest")
    return value


def _validate_run(run: dict[str, Any], *, commit: str) -> None:
    if run.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RuntimeError("Round 74 preflight run schema differs")
    if run.get("execution_git_commit") != commit:
        raise RuntimeError("Round 74 preflight run commit differs")
    backend = run.get("backend")
    inputs = run.get("input_contract")
    result = run.get("result")
    if not isinstance(backend, dict) or not isinstance(inputs, dict):
        raise RuntimeError("Round 74 preflight backend or input contract is absent")
    if not isinstance(result, dict):
        raise RuntimeError("Round 74 preflight result is absent")
    if (
        backend.get("requested") != "directml"
        or backend.get("kind") != "directml"
        or backend.get("accelerated") is not True
        or backend.get("warning_count") != 0
        or backend.get("cpu_fallback_warning_count") != 0
    ):
        raise RuntimeError("Round 74 DirectML backend did not pass its strict gate")
    candidate_ids = inputs.get("candidate_ids")
    seeds = inputs.get("seeds")
    if (
        not isinstance(candidate_ids, list)
        or len(candidate_ids) < 2
        or len(candidate_ids) != len(set(candidate_ids))
        or not all(isinstance(value, str) and value for value in candidate_ids)
        or not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(value, int) and value > 0 for value in seeds)
    ):
        raise RuntimeError("Round 74 preflight candidate or seed panel differs")
    metric_fields = (
        "candidate_run_balanced_tuning_proper_loss",
        "candidate_worst_run_tuning_proper_loss",
        "candidate_pooled_tuning_proper_loss",
        "peer_best_run_balanced_tuning_proper_loss",
    )
    for field in metric_fields:
        values = result.get(field)
        if not isinstance(values, dict) or set(values) != set(candidate_ids):
            raise RuntimeError(f"Round 74 preflight {field} panel differs")
    peer_metrics = result["peer_best_run_balanced_tuning_proper_loss"]
    if any(
        not isinstance(peer_metrics[candidate_id], list)
        or len(peer_metrics[candidate_id]) != len(seeds)
        for candidate_id in candidate_ids
    ):
        raise RuntimeError("Round 74 preflight peer update count differs")
    for field in ("policy_sha256", "model_sha256", "prediction_sha256"):
        _require_sha256(result.get(field), field)
    selection = result.get("selection")
    if (
        not isinstance(selection, dict)
        or selection.get("selected_candidate_id")
        != result.get("selected_candidate_id")
        or selection.get("ordered_candidate_ids") != candidate_ids
        or selection.get("planned_comparison_count") != len(candidate_ids) - 1
        or selection.get("required_paired_capture_run_count") != 24
        or selection.get("statistical_independence_or_significance_claim")
        is not False
        or selection.get("complexity_promotion_privilege") is not False
        or selection.get("backtest_metric_used_for_selection") is not False
        or not isinstance(selection.get("promotion_reports"), list)
        or len(selection["promotion_reports"]) != len(candidate_ids) - 1
        or any(
            not isinstance(report, dict)
            or report.get("promoted") is not False
            or report.get("paired_capture_run_count") != 1
            or report.get("complete_tuning_panel") is not False
            for report in selection["promotion_reports"]
        )
        or result.get("selected_candidate_id") != candidate_ids[0]
    ):
        raise RuntimeError("Round 74 complexity-promotion preflight gate failed")
    if (
        not isinstance(result.get("model_byte_count"), int)
        or result["model_byte_count"] <= 0
        or result.get("selected_candidate_id") not in candidate_ids
        or run.get("temporary_artifact_count_before_cleanup") != 2
        or run.get("temporary_directory_removed") is not True
    ):
        raise RuntimeError("Round 74 preflight serialization or cleanup gate failed")


def _source_binding(repository: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative_path in SOURCE_PATHS.items():
        path = repository / relative_path
        if not path.is_file():
            raise RuntimeError(f"Round 74 source is absent: {relative_path}")
        result[f"{name}_path"] = relative_path
        result[f"{name}_sha256"] = _file_sha256(path)
    result["event_model_schema_version"] = ROUND74_EVENT_MODEL_SCHEMA_VERSION
    result["event_training_schema_version"] = ROUND74_EVENT_TRAINING_SCHEMA_VERSION
    result["pretest_policy_schema_version"] = (
        ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
    )
    return result


def _build_evidence(
    repository: Path,
    *,
    commit: str,
    first: dict[str, Any],
    second: dict[str, Any],
    supersedes_artifact_sha256: str,
) -> dict[str, Any]:
    if first != second:
        raise RuntimeError("Round 74 repeated DirectML preflight results differ")
    inputs = first["input_contract"]
    backend = dict(first["backend"])
    backend["warning_count_per_execution"] = backend.pop("warning_count")
    backend["cpu_fallback_warning_count_per_execution"] = backend.pop(
        "cpu_fallback_warning_count"
    )
    candidate_ids = inputs["candidate_ids"]
    seeds = inputs["seeds"]
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "supersedes_artifact_sha256": _require_sha256(
            supersedes_artifact_sha256,
            "supersedes_artifact_sha256",
        ),
        "round": 74,
        "executed_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "execution_git_commit": commit,
        "source_binding": _source_binding(repository),
        "backend": backend,
        "input_contract": {
            "source": (
                "deterministic constructed tensor contract from the hash-bound "
                "preflight runner"
            ),
            "real_market_events_used": False,
            "real_market_targets_used": False,
            "training_rows": 2,
            "tuning_rows": 2,
            **inputs,
            "test_batches_consumed": 0,
        },
        "verification": {
            "operator_process_invocation_count": 2,
            "fresh_process_execution_count": 2,
            "successful_execution_count": 2,
            "candidate_count": len(candidate_ids),
            "all_candidates_trained": True,
            "seed_count_per_candidate": len(seeds),
            "peer_update_count": len(candidate_ids) * len(seeds),
            "safe_tensor_state_reload_verified": True,
            "atomic_policy_reload_verified": True,
            "temporary_artifacts_removed_after_each_execution": True,
            "cross_execution_complete_result_equal": True,
            "cross_execution_policy_sha256_equal": True,
            "cross_execution_model_sha256_equal": True,
            "cross_execution_prediction_sha256_equal": True,
            "cross_execution_candidate_metrics_equal": True,
        },
        "repeated_result": first["result"],
        "interpretation": {
            "result_type": "compute serialization and governance preflight only",
            "candidate_loss_has_financial_meaning": False,
            "representative_market_training_performed": False,
            "sealed_test_evaluated": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
            "cross_platform_bitwise_reproducibility_claim": False,
        },
    }
    evidence["artifact_sha256"] = _canonical_sha256(evidence)
    return evidence


def _write_new_artifact(path: Path, evidence: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to replace existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            evidence,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
        )
        + "\n"
    ).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    persisted = _strict_json(path.read_text(encoding="ascii"))
    claimed = persisted.pop("artifact_sha256", None)
    if claimed != _canonical_sha256(persisted):
        path.unlink(missing_ok=True)
        raise RuntimeError("persisted Round 74 artifact hash differs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supersedes", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output
    if not output.is_absolute():
        output = repository / output
    supersedes = arguments.supersedes
    if not supersedes.is_absolute():
        supersedes = repository / supersedes
    superseded = _strict_json(supersedes.read_text(encoding="ascii"))
    supersedes_hash = _require_sha256(
        superseded.get("artifact_sha256"),
        "superseded artifact_sha256",
    )
    commit = _require_clean_repository(repository)
    first = _run_preflight(repository)
    second = _run_preflight(repository)
    _validate_run(first, commit=commit)
    _validate_run(second, commit=commit)
    evidence = _build_evidence(
        repository,
        commit=commit,
        first=first,
        second=second,
        supersedes_artifact_sha256=supersedes_hash,
    )
    _write_new_artifact(output, evidence)
    print(
        json.dumps(
            {
                "artifact_sha256": evidence["artifact_sha256"],
                "candidate_ids": evidence["input_contract"]["candidate_ids"],
                "execution_git_commit": commit,
                "output": str(output),
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
