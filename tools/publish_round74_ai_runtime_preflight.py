"""Publish an isolated, nonfinancial Round 74 local-AI runtime preflight."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess  # nosec B404
import sys
import time
from typing import Mapping
from urllib import error as urllib_error
from urllib import request as urllib_request


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_ai_protocol import (  # noqa: E402
    ROUND74_AI_PROMPT_PAYLOAD_SCHEMA_VERSION,
    ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION,
    ROUND74_AI_SYSTEM_PROMPT_SCHEMA_VERSION,
    ROUND74_AI_TEMPORAL_BLOCK_COUNT,
    ROUND74_AI_TEMPORAL_FEATURE_NAMES,
    Round74AIReviewRequest,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (  # noqa: E402
    ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_runtime import (  # noqa: E402
    review_round74_ai_candidate,
)
from simple_ai_trading.impact_absorption_event_sequence import (  # noqa: E402
    ROUND74_EVENT_FEATURE_NAMES,
)
from simple_ai_trading.storage import write_bytes_atomic  # noqa: E402


SCHEMA_VERSION = "round-074-local-ai-runtime-preflight-v4"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
MAXIMUM_RESPONSE_BYTES = 1_000_000
UNLOAD_TIMEOUT_SECONDS = 10.0
SOURCE_PATHS = {
    "ai_protocol": "src/simple_ai_trading/impact_absorption_ai_protocol.py",
    "ai_review_preparation": (
        "src/simple_ai_trading/impact_absorption_ai_review_preparation.py"
    ),
    "ai_runtime": "src/simple_ai_trading/impact_absorption_ai_runtime.py",
    "ai_worker": "src/simple_ai_trading/impact_absorption_ai_worker.py",
    "event_sequence": "src/simple_ai_trading/impact_absorption_event_sequence.py",
    "publisher": "tools/publish_round74_ai_runtime_preflight.py",
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


def _progress(stage: str, **values: object) -> None:
    detail = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[round74-ai-preflight] {stage} {detail}".rstrip(), flush=True)


def _normalized_file_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate keys")
            result[key] = value
        return result

    if len(raw) > MAXIMUM_RESPONSE_BYTES:
        raise ValueError(f"{label} exceeds its byte limit")
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"{label} contains {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{label} root differs")
    return value


def _ollama_json(
    path: str,
    *,
    payload: Mapping[str, object] | None = None,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    if not path.startswith("/api/"):
        raise ValueError("Round 74 Ollama path differs")
    body = None if payload is None else _canonical_bytes(dict(payload))
    selected = urllib_request.Request(
        f"{OLLAMA_ENDPOINT}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib_request.urlopen(  # nosec B310 - fixed loopback endpoint
            selected,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read(MAXIMUM_RESPONSE_BYTES + 1)
    except (OSError, urllib_error.URLError) as exc:
        raise ValueError("Round 74 Ollama control request failed") from exc
    return _strict_json(raw, label="Round 74 Ollama control response")


def _loaded_models() -> dict[str, dict[str, object]]:
    response = _ollama_json("/api/ps")
    models = response.get("models")
    if not isinstance(models, list):
        raise ValueError("Round 74 Ollama running-model inventory differs")
    result: dict[str, dict[str, object]] = {}
    for value in models:
        if not isinstance(value, Mapping):
            raise ValueError("Round 74 Ollama running-model entry differs")
        name = value.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in result
            or not isinstance(value.get("size"), int)
            or not isinstance(value.get("size_vram"), int)
        ):
            raise ValueError("Round 74 Ollama running-model identity differs")
        result[name] = dict(value)
    return result


def _unload_declared_model(model_name: str) -> None:
    if model_name not in _loaded_models():
        return
    response = _ollama_json(
        "/api/generate",
        payload={
            "model": model_name,
            "keep_alive": 0,
            "stream": False,
        },
        timeout_seconds=UNLOAD_TIMEOUT_SECONDS,
    )
    if response.get("done") is not True:
        raise ValueError("Round 74 Ollama unload response differs")
    deadline = time.monotonic() + UNLOAD_TIMEOUT_SECONDS
    while model_name in _loaded_models():
        if time.monotonic() >= deadline:
            raise TimeoutError("Round 74 Ollama model unload timed out")
        time.sleep(0.25)


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
        raise RuntimeError("Round 74 AI preflight requires a clean worktree")
    commit = _git(repository, "rev-parse", "HEAD")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Round 74 AI preflight commit identity differs")
    return commit


def _synthetic_request() -> Round74AIReviewRequest:
    feature_count = len(ROUND74_EVENT_FEATURE_NAMES)
    event_fractions = (0.40, 0.20, 0.30, 0.09, 0.01)
    feature_last = tuple(
        1.0 if index in (0, 5) else 0.0
        for index in range(feature_count)
    )
    feature_mean = tuple(
        (
            event_fractions[index]
            if index < len(event_fractions)
            else 1.0 if index == 5 else 0.0
        )
        for index in range(feature_count)
    )
    feature_standard_deviation = tuple(
        (
            (event_fractions[index] * (1.0 - event_fractions[index])) ** 0.5
            if index < len(event_fractions)
            else 0.0 if index < 8 else 1.0
        )
        for index in range(feature_count)
    )
    now = time.time_ns()
    request = Round74AIReviewRequest(
        pretest_policy_sha256="a" * 64,
        probability_calibration_sha256="b" * 64,
        sample_sha256="c" * 64,
        deterministic_risk_state_sha256="d" * 64,
        risk_profile="conservative",
        asset_slot=0,
        side="long",
        horizon_seconds=30,
        requested_wall_ns=now,
        expires_wall_ns=now + 30_000_000_000,
        proposed_risk_size_bps=50,
        feature_last=feature_last,
        feature_mean=feature_mean,
        feature_standard_deviation=feature_standard_deviation,
        feature_recent_change=(0.0,) * feature_count,
        feature_recent_block_means=tuple(
            tuple(
                (
                    0.25 * (block_index + 1)
                    if feature_index in (0, 9, 10, 13)
                    else 0.0
                )
                for feature_index in range(len(ROUND74_AI_TEMPORAL_FEATURE_NAMES))
            )
            for block_index in range(ROUND74_AI_TEMPORAL_BLOCK_COUNT)
        ),
        payoff_quantiles_bps=(-2.0, -1.0, 0.0, 1.0, 2.0),
        maximum_adverse_excursion_quantiles_bps=(1.0, 2.0, 3.0, 4.0, 5.0),
        positive_payoff_probability=0.70,
        adverse_selection_probability=0.85,
        regime_unpredictability_probability=0.90,
    )
    request.validate()
    return request


def _source_binding(repository: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative_path in SOURCE_PATHS.items():
        path = repository / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"Round 74 AI preflight source is absent: {relative_path}"
            )
        result[f"{name}_path"] = relative_path
        result[f"{name}_sha256"] = _normalized_file_sha256(path)
    return result


def _run(repository: Path) -> dict[str, object]:
    commit = _require_clean_repository(repository)
    panel = round74_default_ai_review_model_panel()
    model_names = tuple(binding.model_name for binding in panel)
    loaded_before = _loaded_models()
    unrelated = sorted(set(loaded_before) - set(model_names))
    if unrelated:
        raise RuntimeError(
            f"Round 74 AI preflight found unrelated resident models: {unrelated}"
        )
    outcomes: list[dict[str, object]] = []
    try:
        for binding in panel:
            for model_name in model_names:
                _unload_declared_model(model_name)
            for phase in ("cold", "warm"):
                _progress(
                    "review-start",
                    model=binding.model_name,
                    phase=phase,
                )
                request = _synthetic_request()
                outcome = review_round74_ai_candidate(
                    binding.runtime,
                    binding.manifest,
                    request,
                    deterministic_risk_gate_passed=True,
                    observed_wall_ns=time.time_ns(),
                )
                outcome.validate()
                worker = outcome.worker_result
                capability = outcome.capability or {}
                _progress(
                    "review-result",
                    model=binding.model_name,
                    phase=phase,
                    status=outcome.status,
                    failure_class=outcome.failure_class,
                    free_ram_gb=capability.get("free_ram_gb"),
                    free_vram_gb=capability.get("free_vram_gb"),
                    warm_exact_gpu=capability.get(
                        "pre_inference_exact_model_fully_gpu_resident"
                    ),
                )
                if (
                    outcome.status != "accepted"
                    or worker is None
                    or outcome.approved_risk_size_bps > outcome.proposed_risk_size_bps
                    or worker["residency"]["status"] != "gpu_resident"
                    or worker["residency"]["vram_to_model_ratio"] != 1.0
                    or worker["decision"]["may_increase_risk"] is not False
                    or worker["decision"]["may_select_side"] is not False
                    or worker["decision"]["may_set_leverage"] is not False
                    or worker["decision"]["may_submit_or_cancel_orders"] is not False
                ):
                    raise RuntimeError(
                        f"Round 74 AI preflight model failed: "
                        f"{binding.model_name} {phase}"
                    )
                outcomes.append(
                    {
                        "role": binding.role,
                        "model_name": binding.model_name,
                        "phase": phase,
                        "manifest_sha256": binding.manifest.manifest_sha256,
                        "request_sha256": request.request_sha256,
                        "outcome": outcome.as_dict(),
                    }
                )
                _progress(
                    "review-complete",
                    model=binding.model_name,
                    phase=phase,
                    elapsed_ns=outcome.elapsed_ns,
                    load_duration_ns=worker["load_duration_ns"],
                )
            cold, warm = outcomes[-2:]
            cold_worker = cold["outcome"]["worker_result"]
            warm_worker = warm["outcome"]["worker_result"]
            if (
                cold["phase"] != "cold"
                or warm["phase"] != "warm"
                or warm_worker["load_duration_ns"] >= cold_worker["load_duration_ns"]
                or warm["outcome"]["capability"][
                    "pre_inference_exact_model_fully_gpu_resident"
                ]
                is not True
            ):
                raise RuntimeError(
                    f"Round 74 AI warm residency proof failed: {binding.model_name}"
                )
            _unload_declared_model(binding.model_name)
    finally:
        for model_name in model_names:
            _unload_declared_model(model_name)
    loaded_after = _loaded_models()
    if loaded_after:
        raise RuntimeError("Round 74 AI preflight left a model resident")
    evidence: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "round": 74,
        "executed_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "execution_git_commit": commit,
        "source_binding": _source_binding(repository),
        "runtime_isolation": {
            "ollama_endpoint": OLLAMA_ENDPOINT,
            "resident_models_before": sorted(loaded_before),
            "unrelated_resident_models_permitted": False,
            "declared_models_unloaded_between_model_batches": True,
            "declared_model_retained_between_cold_and_warm_requests": True,
            "resident_models_after": sorted(loaded_after),
            "model_process_terminated": False,
            "official_keep_alive_zero_api_used": True,
        },
        "input_contract": {
            "source": "deterministic constructed high-risk numeric packet",
            "review_request_schema_version": (
                ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
            ),
            "prompt_payload_schema_version": (
                ROUND74_AI_PROMPT_PAYLOAD_SCHEMA_VERSION
            ),
            "system_prompt_schema_version": (
                ROUND74_AI_SYSTEM_PROMPT_SCHEMA_VERSION
            ),
            "review_panel_schema_version": ROUND74_AI_REVIEW_PANEL_SCHEMA_VERSION,
            "risk_profile": "conservative",
            "temporal_block_count": ROUND74_AI_TEMPORAL_BLOCK_COUNT,
            "temporal_feature_count": len(ROUND74_AI_TEMPORAL_FEATURE_NAMES),
            "temporal_order": "oldest_to_newest",
            "temporal_path_exercised": True,
            "real_market_events_used": False,
            "real_market_targets_used": False,
            "absolute_market_date_exposed_to_model": False,
            "real_symbol_exposed_to_model": False,
            "test_partition_accessed": False,
        },
        "model_outcomes": outcomes,
        "verification": {
            "model_count": len(panel),
            "request_count": len(outcomes),
            "cold_request_count": sum(value["phase"] == "cold" for value in outcomes),
            "warm_request_count": sum(value["phase"] == "warm" for value in outcomes),
            "all_models_accepted_by_protocol": True,
            "all_models_fully_gpu_resident": True,
            "all_warm_requests_reused_exact_gpu_residency": True,
            "all_warm_load_durations_below_cold_load_durations": True,
            "all_models_remote_inference_used": False,
            "all_models_execution_authority": False,
            "all_models_may_increase_risk": False,
            "all_models_may_select_side": False,
            "all_models_may_set_leverage": False,
            "all_models_may_submit_or_cancel_orders": False,
        },
        "interpretation": {
            "result_type": (
                "local model protocol, latency, provenance, and GPU-residency "
                "preflight only"
            ),
            "representative_market_ai_evaluation_completed": False,
            "ai_uplift_established": False,
            "financial_edge_established": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
        "research_sources": [
            {
                "title": "Ollama FAQ: model keep-alive and unloading",
                "url": "https://docs.ollama.com/faq",
                "used_for": (
                    "retaining each model for its warm request and isolating model "
                    "batches with the documented keep_alive API"
                ),
            },
            {
                "title": "Ollama API: list running models",
                "url": "https://docs.ollama.com/api/ps",
                "used_for": "fail-closed resident-model inventory checks",
            },
        ],
    }
    evidence["artifact_sha256"] = _canonical_sha256(evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=REPOSITORY,
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output
    if not output.is_absolute():
        output = repository / output
    if output.exists():
        raise FileExistsError(f"Round 74 AI preflight output exists: {output}")
    evidence = _run(repository)
    payload = (
        json.dumps(
            evidence,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
        ).encode("ascii")
        + b"\n"
    )
    write_bytes_atomic(output, payload)
    persisted = _strict_json(
        output.read_bytes(),
        label="Round 74 persisted AI preflight",
    )
    claimed = persisted.pop("artifact_sha256", None)
    if claimed != _canonical_sha256(persisted):
        raise RuntimeError("Round 74 persisted AI preflight digest differs")
    print(
        json.dumps(
            {
                "artifact_sha256": evidence["artifact_sha256"],
                "execution_git_commit": evidence["execution_git_commit"],
                "models": [value["model_name"] for value in evidence["model_outcomes"]],
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
