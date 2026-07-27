"""Publish the hash-bound Round 74 warm-residency AI design revision."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
RESEARCH = Path("docs/model-research/action-value")
PREVIOUS_DESIGN = RESEARCH / "round-074-local-ai-review-design-v47.json"
RUNTIME_PREFLIGHT = RESEARCH / "round-074-local-ai-runtime-preflight-v2-2026-07-27.json"
OUTPUT = RESEARCH / "round-074-local-ai-review-design-v48.json"
GENERATOR_PATH = "tools/publish_round74_warm_ai_contract.py"
RUNTIME_PATH = "src/simple_ai_trading/impact_absorption_ai_runtime.py"
PREPARATION_PATH = "src/simple_ai_trading/impact_absorption_ai_review_preparation.py"
RUNTIME_SCHEMA_VERSION = "round-074-ai-runtime-outcome-v2"
PANEL_SCHEMA_VERSION = "round-074-ai-review-panel-v4"


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
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path.name} contains duplicate keys")
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
    persisted = value.pop(digest_key, None)
    if not isinstance(persisted, str) or persisted != _canonical_sha256(value):
        raise ValueError(f"{path.name} hash binding differs")
    value[digest_key] = persisted
    return value


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(  # nosec B603
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
        errors="strict",
    )
    return completed.stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Round 74 AI design publication requires a clean worktree")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Round 74 AI design commit identity differs")
    return commit


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 74 {label} differs")
    return value


def _validated_outcomes(
    preflight: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw = preflight.get("model_outcomes")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("Round 74 warm AI outcome coverage differs")
    outcomes = tuple(_mapping(value, "AI runtime outcome") for value in raw)
    expected = (
        ("fino1:8b", "cold"),
        ("fino1:8b", "warm"),
        ("qwen3:8b", "cold"),
        ("qwen3:8b", "warm"),
    )
    if tuple((value.get("model_name"), value.get("phase")) for value in outcomes) != (
        expected
    ):
        raise ValueError("Round 74 warm AI phase order differs")
    for value in outcomes:
        outcome = _mapping(value.get("outcome"), "AI parent outcome")
        worker = _mapping(outcome.get("worker_result"), "AI worker result")
        capability = _mapping(outcome.get("capability"), "AI capability")
        residency = _mapping(worker.get("residency"), "AI residency")
        decision = _mapping(worker.get("decision"), "AI decision")
        if (
            outcome.get("status") != "accepted"
            or outcome.get("remote_inference_used") is not False
            or outcome.get("execution_authority") is not False
            or residency.get("status") != "gpu_resident"
            or residency.get("vram_to_model_ratio") != 1.0
            or decision.get("may_increase_risk") is not False
            or decision.get("may_select_side") is not False
            or decision.get("may_set_leverage") is not False
            or decision.get("may_submit_or_cancel_orders") is not False
        ):
            raise ValueError("Round 74 warm AI accepted evidence differs")
        if value["phase"] == "warm" and (
            capability.get("pre_inference_exact_model_fully_gpu_resident") is not True
            or capability.get("pre_inference_warm_ram_headroom_passed") is not True
        ):
            raise ValueError("Round 74 warm AI residency gate differs")
    for cold, warm in ((outcomes[0], outcomes[1]), (outcomes[2], outcomes[3])):
        cold_worker = _mapping(
            _mapping(cold["outcome"], "cold outcome").get("worker_result"),
            "cold worker",
        )
        warm_worker = _mapping(
            _mapping(warm["outcome"], "warm outcome").get("worker_result"),
            "warm worker",
        )
        if not (
            isinstance(cold_worker.get("load_duration_ns"), int)
            and isinstance(warm_worker.get("load_duration_ns"), int)
            and warm_worker["load_duration_ns"] < cold_worker["load_duration_ns"]
        ):
            raise ValueError("Round 74 warm AI load amortization differs")
    return outcomes


def _phase_values(
    outcomes: tuple[Mapping[str, Any], ...],
    *,
    phase: str,
    field: str,
) -> list[float]:
    values: list[float] = []
    for value in outcomes:
        if value["phase"] != phase:
            continue
        capability = _mapping(
            _mapping(value["outcome"], "phase outcome").get("capability"),
            "phase capability",
        )
        selected = capability.get(field)
        if isinstance(selected, bool) or not isinstance(selected, (int, float)):
            raise ValueError(f"Round 74 AI {phase} {field} differs")
        values.append(float(selected))
    if len(values) != 2:
        raise ValueError(f"Round 74 AI {phase} capability coverage differs")
    return values


def _duration_map(
    outcomes: tuple[Mapping[str, Any], ...],
    field: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in outcomes:
        outcome = _mapping(value["outcome"], "duration outcome")
        source = (
            _mapping(outcome.get("worker_result"), "duration worker")
            if field != "elapsed_ns"
            else outcome
        )
        duration = source.get(field)
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
            raise ValueError(f"Round 74 AI {field} differs")
        result.setdefault(str(value["model_name"]), {})[str(value["phase"])] = duration
    return result


def _build(repository: Path, commit: str) -> dict[str, Any]:
    previous_path = repository / PREVIOUS_DESIGN
    preflight_path = repository / RUNTIME_PREFLIGHT
    previous = _load_hash_bound(previous_path, "artifact_sha256")
    preflight = _load_hash_bound(preflight_path, "artifact_sha256")
    if preflight.get("schema_version") != "round-074-local-ai-runtime-preflight-v2":
        raise ValueError("Round 74 warm AI preflight schema differs")
    outcomes = _validated_outcomes(preflight)
    source_binding = _mapping(preflight.get("source_binding"), "source binding")
    verification = _mapping(preflight.get("verification"), "verification")
    isolation = _mapping(preflight.get("runtime_isolation"), "runtime isolation")
    interpretation = _mapping(preflight.get("interpretation"), "interpretation")
    if (
        verification.get("all_warm_requests_reused_exact_gpu_residency") is not True
        or verification.get("all_warm_load_durations_below_cold_load_durations")
        is not True
        or isolation.get("resident_models_after") != []
        or interpretation.get("representative_market_ai_evaluation_completed")
        is not False
        or interpretation.get("ai_uplift_established") is not False
        or interpretation.get("financial_edge_established") is not False
        or interpretation.get("profitability_claim") is not False
    ):
        raise ValueError("Round 74 warm AI preflight interpretation differs")

    value = deepcopy(previous)
    value.pop("artifact_sha256")
    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    value.update(
        {
            "schema_version": "round-074-local-ai-review-design-v48",
            "researched_at_utc": now,
            "implementation_git_commit": commit,
            "research_git_basis_commit": commit,
            "supersedes_artifact_sha256": previous["artifact_sha256"],
        }
    )
    source = value["source_binding"]
    source.update(
        {
            "runtime_path": RUNTIME_PATH,
            "runtime_sha256": _normalized_file_sha256(repository / RUNTIME_PATH),
            "runtime_outcome_schema_version": RUNTIME_SCHEMA_VERSION,
            "review_preparation_path": PREPARATION_PATH,
            "review_preparation_sha256": _normalized_file_sha256(
                repository / PREPARATION_PATH
            ),
            "review_panel_schema_version": PANEL_SCHEMA_VERSION,
            "contract_generator_path": GENERATOR_PATH,
            "contract_generator_sha256": _normalized_file_sha256(
                repository / GENERATOR_PATH
            ),
        }
    )
    value["architecture"].update(
        {
            "cold_start_headroom_rechecked_before_model_load": True,
            "warm_request_requires_exact_preinference_model_digest": True,
            "warm_request_requires_preinference_full_gpu_residency": True,
            "warm_request_rechecks_residual_system_ram": True,
            "warm_request_worker_rechecks_postinference_full_gpu_residency": True,
            "model_major_provider_batching_implemented": True,
            "declared_model_unload_verified_after_each_real_model_batch": True,
            "unrelated_models_targeted_for_unload": False,
            "provider_load_duration_bound_into_worker_evidence": True,
        }
    )
    value["resource_policy"].update(
        {
            "minimum_cold_free_vram_bytes": 8 * 1024**3,
            "minimum_cold_free_system_ram_bytes": 16 * 1024**3,
            "minimum_warm_free_system_ram_bytes": 4 * 1024**3,
            "cold_headroom_not_reapplied_to_exact_resident_model_bytes": True,
            "warm_residual_ram_floor_may_be_bypassed": False,
            "one_large_model_resident_at_a_time": True,
            "declared_model_only_unload": True,
            "verified_absent_before_next_model_batch": True,
        }
    )
    relative_preflight = str(RUNTIME_PREFLIGHT).replace("\\", "/")
    cold_ram = _phase_values(outcomes, phase="cold", field="free_ram_gb")
    cold_vram = _phase_values(outcomes, phase="cold", field="free_vram_gb")
    warm_ram = _phase_values(outcomes, phase="warm", field="free_ram_gb")
    warm_vram = _phase_values(outcomes, phase="warm", field="free_vram_gb")
    value["host_preflight"].update(
        {
            "observed_at_utc": preflight["executed_at_utc"],
            "evidence_path": relative_preflight,
            "evidence_artifact_sha256": preflight["artifact_sha256"],
            "minimum_observed_free_vram_gib": min(cold_vram),
            "minimum_required_free_vram_gib": 8.0,
            "minimum_observed_free_system_ram_gib": min(cold_ram),
            "minimum_required_free_system_ram_gib": 16.0,
            "minimum_warm_free_vram_gib": min(warm_vram),
            "minimum_warm_free_system_ram_gib": min(warm_ram),
            "minimum_required_warm_free_system_ram_gib": 4.0,
            "cold_and_warm_request_count": len(outcomes),
            "cold_and_warm_runtime_preflight_passed": True,
            "runtime_outcome_schema_version": RUNTIME_SCHEMA_VERSION,
            "review_panel_schema_version": PANEL_SCHEMA_VERSION,
            "elapsed_ns_by_model_and_phase": _duration_map(outcomes, "elapsed_ns"),
            "load_duration_ns_by_model_and_phase": _duration_map(
                outcomes, "load_duration_ns"
            ),
            "representative_market_ai_evaluation_completed": False,
            "ai_uplift_established": False,
            "financial_edge_established": False,
            "profitability_claim": False,
        }
    )
    value["latest_capability_recheck"].update(
        {
            "observed_at_utc": preflight["executed_at_utc"],
            "minimum_free_vram_gib_observed": min(cold_vram),
            "minimum_free_vram_gib_required": 8.0,
            "minimum_free_system_ram_gib_observed": min(cold_ram),
            "minimum_free_system_ram_gib_required": 16.0,
            "minimum_warm_free_system_ram_gib_observed": min(warm_ram),
            "minimum_warm_free_system_ram_gib_required": 4.0,
            "all_cold_and_warm_models_fully_gpu_resident": True,
            "all_warm_requests_reused_exact_gpu_residency": True,
            "all_warm_load_durations_below_cold_load_durations": True,
        }
    )
    value["runtime_preflight_evidence_binding"].update(
        {
            "path": relative_preflight,
            "file_sha256": _normalized_file_sha256(preflight_path),
            "artifact_sha256": preflight["artifact_sha256"],
            "schema_version": preflight["schema_version"],
            "execution_git_commit": preflight["execution_git_commit"],
            "publisher_path": source_binding["publisher_path"],
            "publisher_sha256": source_binding["publisher_sha256"],
            "protocol_sha256": source_binding["ai_protocol_sha256"],
            "model_count": verification["model_count"],
            "request_count": verification["request_count"],
            "model_names": ["fino1:8b", "qwen3:8b"],
            "runtime_outcome_schema_versions": [
                _mapping(value["outcome"], "bound outcome")["schema_version"]
                for value in outcomes
            ],
            "all_models_accepted_by_protocol": True,
            "all_models_fully_gpu_resident": True,
            "all_warm_requests_reused_exact_gpu_residency": True,
            "all_warm_load_durations_below_cold_load_durations": True,
            "declared_models_unloaded_between_model_batches": True,
            "resident_models_before": isolation["resident_models_before"],
            "resident_models_after": isolation["resident_models_after"],
            "representative_market_ai_evaluation_completed": False,
            "ai_uplift_established": False,
            "financial_edge_established": False,
            "profitability_claim": False,
        }
    )
    value["status"].update(
        {
            "representative_market_ai_evaluation_completed": False,
            "ai_uplift_established": False,
            "financial_edge_established": False,
            "profitability_claim": False,
        }
    )
    value["artifact_sha256"] = _canonical_sha256(value)
    return value


def main() -> int:
    repository = REPOSITORY.resolve()
    commit = _require_clean_repository(repository)
    output = repository / OUTPUT
    if output.exists():
        raise FileExistsError(f"Round 74 warm AI design already exists: {output}")
    value = _build(repository, commit)
    output.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2) + "\n",
        encoding="ascii",
        newline="\n",
    )
    persisted = _load_hash_bound(output, "artifact_sha256")
    if persisted != value:
        raise RuntimeError("Round 74 warm AI design persistence differs")
    print(
        json.dumps(
            {
                "artifact_sha256": value["artifact_sha256"],
                "execution_git_commit": commit,
                "output": str(OUTPUT).replace("\\", "/"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
