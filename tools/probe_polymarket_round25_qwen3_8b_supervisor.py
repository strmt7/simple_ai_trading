"""Screen Qwen3 8B on the frozen Round 25 target-free supervisor battery."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Mapping

from probe_polymarket_round25_ai_supervisor import (
    _evaluate_checks,
    _file_sha256,
    _scenario_packet,
    _windows_gpu_names,
)
from simple_ai_trading.ai_runtime import inspect_ollama_model_residency
from simple_ai_trading.polymarket_round25_ai import (
    _canonical_json,
    _canonical_sha256,
    _get_json,
    _normalized_model,
    _post_json,
    _provider_usage,
)
from simple_ai_trading.polymarket_round25_ai_supervisor import (
    _RESPONSE_SCHEMA,
    _parse_response,
    _prompt,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-qwen3-8b-regime-supervisor-scenario-contract-v1.json"
)
SCENARIO_LEDGER = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-fin-r1-regime-supervisor-scenario-contract-v1.json"
)
PRIOR_RISK_BENCHMARK = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "latest"
    / "ai-risk-models-rejected.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-qwen3-8b-regime-supervisor-host-probe-v1-2026-08-10.json"
)
CONTRACT_SHA256 = "8e5e1a210441069e3f71bec42005c725e2c52b521263342b78a82e7bfda3eaf8"
SCENARIO_LEDGER_SHA256 = "9828aaf05deafe776c09a26c4e4cc0578762b9c53efa6c75ef83d4c3dc14dac4"
PRIOR_RISK_BENCHMARK_FILE_SHA256 = (
    "78afdd2f3438056da9a062e1e3d00428c69b045f7e57d2c8e268d74e9200b5bd"
)
MODEL = "qwen3:8b"
MODEL_DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
PARAMETER_SIZE = "8.2B"
QUANTIZATION = "Q4_K_M"
BASE_URL = "http://127.0.0.1:11434"
TIMEOUT_SECONDS = 30.0
PRELOAD_SECONDS = 60.0
CONTEXT_TOKENS = 4096
OUTPUT_TOKENS = 24
SEED = 25_027
KEEP_ALIVE = "2m"
_OLLAMA_VERSION_PREFIX = "0.32."


def _validated_contract(path: Path, expected_sha256: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    claimed = value.pop("contract_sha256", None)
    if claimed != expected_sha256 or _canonical_sha256(value) != claimed:
        raise ValueError("Round 25 Qwen3 8B supervisor contract differs")
    value["contract_sha256"] = claimed
    return value


def _load_inputs() -> tuple[dict[str, object], dict[str, object]]:
    contract = _validated_contract(CONTRACT, CONTRACT_SHA256)
    scenarios = _validated_contract(SCENARIO_LEDGER, SCENARIO_LEDGER_SHA256)
    if (
        contract.get("packet_and_prompt", {}).get("scenario_ledger_source_sha256")
        != SCENARIO_LEDGER_SHA256
        or _file_sha256(PRIOR_RISK_BENCHMARK) != PRIOR_RISK_BENCHMARK_FILE_SHA256
    ):
        raise ValueError("Round 25 Qwen3 8B supervisor source lineage differs")
    return contract, scenarios


def _preflight() -> tuple[str, str]:
    version_payload = _get_json(f"{BASE_URL}/api/version", TIMEOUT_SECONDS)
    if not isinstance(version_payload, Mapping) or set(version_payload) != {"version"}:
        raise ValueError("Round 25 Qwen3 8B Ollama version differs")
    version = str(version_payload["version"] or "")
    if not version.startswith(_OLLAMA_VERSION_PREFIX):
        raise ValueError("Round 25 Qwen3 8B Ollama compatibility differs")
    patch = version.removeprefix(_OLLAMA_VERSION_PREFIX)
    if not patch.isdigit() or int(patch) < 4:
        raise ValueError("Round 25 Qwen3 8B Ollama patch differs")
    tags = _get_json(f"{BASE_URL}/api/tags", TIMEOUT_SECONDS)
    if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
        raise ValueError("Round 25 Qwen3 8B inventory differs")
    matches = []
    for raw in tags["models"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Round 25 Qwen3 8B inventory entry differs")
        names = {_normalized_model(raw.get("name")), _normalized_model(raw.get("model"))}
        if _normalized_model(MODEL) in names:
            matches.append(raw)
    if len(matches) != 1 or matches[0].get("digest") != MODEL_DIGEST:
        raise ValueError("Round 25 Qwen3 8B digest differs")
    details = matches[0].get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("format") != "gguf"
        or details.get("parameter_size") != PARAMETER_SIZE
        or details.get("quantization_level") != QUANTIZATION
    ):
        raise ValueError("Round 25 Qwen3 8B inventory metadata differs")
    show = _post_json(
        f"{BASE_URL}/api/show",
        {"model": MODEL, "verbose": False},
        TIMEOUT_SECONDS,
    )
    if not isinstance(show, Mapping):
        raise ValueError("Round 25 Qwen3 8B model details differ")
    show_details = show.get("details")
    model_info = show.get("model_info")
    if (
        not isinstance(show_details, Mapping)
        or show_details.get("format") != "gguf"
        or show_details.get("parameter_size") != PARAMETER_SIZE
        or show_details.get("quantization_level") != QUANTIZATION
        or not isinstance(model_info, Mapping)
        or int(model_info.get("general.parameter_count") or 0) < 8_000_000_000
    ):
        raise ValueError("Round 25 Qwen3 8B model details metadata differs")
    return version, _canonical_sha256(show)


def _preload() -> dict[str, object]:
    response = _post_json(
        f"{BASE_URL}/api/generate",
        {
            "model": MODEL,
            "prompt": "",
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": CONTEXT_TOKENS},
        },
        PRELOAD_SECONDS,
    )
    if not isinstance(response, Mapping) or response.get("done") is not True:
        raise ValueError("Round 25 Qwen3 8B preload failed")
    residency = inspect_ollama_model_residency(
        BASE_URL,
        MODEL,
        2.0,
        expected_digest=MODEL_DIGEST,
    ).validated()
    if (
        not residency.fully_gpu_resident
        or residency.vram_to_model_ratio is None
        or residency.vram_to_model_ratio < 0.99
    ):
        raise ValueError("Round 25 Qwen3 8B preload residency differs")
    return residency.asdict()


def _unload() -> None:
    response = _post_json(
        f"{BASE_URL}/api/generate",
        {"model": MODEL, "keep_alive": 0, "stream": False},
        TIMEOUT_SECONDS,
    )
    if (
        not isinstance(response, Mapping)
        or response.get("done") is not True
        or response.get("done_reason") != "unload"
    ):
        raise ValueError("Round 25 Qwen3 8B unload failed")
    residency = inspect_ollama_model_residency(
        BASE_URL,
        MODEL,
        2.0,
        expected_digest=MODEL_DIGEST,
    ).validated()
    if residency.loaded:
        raise ValueError("Round 25 Qwen3 8B remained loaded")


def _review(packet: object, *, version: str, show_sha256: str) -> dict[str, object]:
    prompt = _prompt(packet)
    started_ns = time.perf_counter_ns()
    provider = _post_json(
        f"{BASE_URL}/api/generate",
        {
            "model": MODEL,
            "prompt": prompt,
            "system": (
                "You are a conservative financial risk supervisor. You can only "
                "reduce or halt future entries and can never affect exits."
            ),
            "format": _RESPONSE_SCHEMA,
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "seed": SEED,
                "num_predict": OUTPUT_TOKENS,
                "num_ctx": CONTEXT_TOKENS,
            },
        },
        TIMEOUT_SECONDS,
    )
    latency_seconds = max(0.0, (time.perf_counter_ns() - started_ns) / 1_000_000_000)
    usage = _provider_usage(provider)
    if (
        _normalized_model(usage["model"]) != _normalized_model(MODEL)
        or not math.isfinite(latency_seconds)
        or latency_seconds > TIMEOUT_SECONDS
        or int(usage["total_duration"]) / 1_000_000_000.0 > latency_seconds + 1.0
    ):
        raise ValueError("Round 25 Qwen3 8B provider telemetry differs")
    if not isinstance(provider, Mapping):
        raise ValueError("Round 25 Qwen3 8B provider response differs")
    action, multiplier, cooldown_ms, _reasons = _parse_response(provider.get("response"))
    residency = inspect_ollama_model_residency(
        BASE_URL,
        MODEL,
        2.0,
        expected_digest=MODEL_DIGEST,
    ).validated()
    if (
        not residency.fully_gpu_resident
        or residency.vram_to_model_ratio is None
        or residency.vram_to_model_ratio < 0.99
    ):
        raise ValueError("Round 25 Qwen3 8B response residency differs")
    response_text = str(provider["response"])
    return {
        "action": action,
        "maximum_size_multiplier": multiplier,
        "cooldown_ms": cooldown_ms,
        "valid_model_response": True,
        "failure_code": None,
        "latency_seconds": latency_seconds,
        "prompt_token_count": int(usage["prompt_eval_count"]),
        "output_token_count": int(usage["eval_count"]),
        "prompt_sha256": hashlib.sha256(prompt.encode("ascii")).hexdigest(),
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "gpu_residency_ratio": residency.vram_to_model_ratio,
        "ollama_version": version,
        "show_metadata_sha256": show_sha256,
    }


def run_probe() -> dict[str, object]:
    contract, scenario_ledger = _load_inputs()
    order = scenario_ledger.get("scenario_order")
    overrides = scenario_ledger.get("scenario_overrides")
    if not isinstance(order, list) or len(order) != 7 or not isinstance(overrides, Mapping):
        raise ValueError("Round 25 Qwen3 8B scenario ledger differs")
    results: list[dict[str, object]] = []
    preload = None
    unloaded = False
    failure = None
    try:
        print("preloading qwen3:8b", flush=True)
        preload = _preload()
        version, show_sha256 = _preflight()
        for index, raw_id in enumerate(order, start=1):
            scenario_id = str(raw_id)
            changes = overrides.get(scenario_id)
            if not isinstance(changes, Mapping):
                raise ValueError("Round 25 Qwen3 8B scenario changes differ")
            packet = _scenario_packet(scenario_id=scenario_id, overrides=changes)
            try:
                result = _review(packet, version=version, show_sha256=show_sha256)
            except Exception as exc:  # noqa: BLE001 - preserve exact bounded case failure
                result = {
                    "action": None,
                    "maximum_size_multiplier": 0.0,
                    "cooldown_ms": 0,
                    "valid_model_response": False,
                    "failure_code": f"{type(exc).__name__}:{str(exc)[:160]}",
                    "latency_seconds": None,
                    "prompt_token_count": None,
                    "output_token_count": None,
                    "prompt_sha256": None,
                    "response_sha256": None,
                    "gpu_residency_ratio": None,
                    "ollama_version": version,
                    "show_metadata_sha256": show_sha256,
                }
            results.append({
                "scenario_id": scenario_id,
                "changed_fields": dict(changes),
                "packet_sha256": packet.packet_sha256,
                **result,
            })
            print(
                f"scenario {index}/7 {scenario_id}: "
                f"action={result['action']} failure={result['failure_code']}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - preserve bounded host failure evidence
        failure = {"type": type(exc).__name__, "message": str(exc)[:240]}
    finally:
        try:
            print("unloading qwen3:8b", flush=True)
            _unload()
            unloaded = True
        except Exception as exc:  # noqa: BLE001 - unload failure must remain visible
            failure = failure or {"type": type(exc).__name__, "message": str(exc)[:240]}
    checks = _evaluate_checks(results, unloaded=unloaded) if len(results) == 7 else {
        key: False for key in contract["checks"]
    }
    module = ROOT / "src" / "simple_ai_trading" / "polymarket_round25_ai_supervisor.py"
    tool = Path(__file__).resolve()
    return {
        "schema_version": "polymarket-round25-qwen3-8b-regime-supervisor-host-probe-v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "target_free_supervisor_behavior_verified"
        if all(checks.values()) and failure is None
        else "target_free_supervisor_behavior_failed",
        "host": {
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "gpu_names": list(_windows_gpu_names()),
        },
        "candidate": {
            "candidate_id": "qwen3-8b-regime-supervisor-v1",
            "model": MODEL,
            "digest": MODEL_DIGEST,
            "preload_residency": preload,
        },
        "source": {
            "candidate_contract_path": CONTRACT.relative_to(ROOT).as_posix(),
            "candidate_contract_sha256": CONTRACT_SHA256,
            "candidate_contract_file_sha256": _file_sha256(CONTRACT),
            "scenario_ledger_path": SCENARIO_LEDGER.relative_to(ROOT).as_posix(),
            "scenario_ledger_sha256": SCENARIO_LEDGER_SHA256,
            "scenario_ledger_file_sha256": _file_sha256(SCENARIO_LEDGER),
            "prior_risk_benchmark_path": PRIOR_RISK_BENCHMARK.relative_to(ROOT).as_posix(),
            "prior_risk_benchmark_file_sha256": PRIOR_RISK_BENCHMARK_FILE_SHA256,
            "packet_prompt_module_path": module.relative_to(ROOT).as_posix(),
            "packet_prompt_module_sha256": _file_sha256(module),
            "probe_tool_path": tool.relative_to(ROOT).as_posix(),
            "probe_tool_sha256": _file_sha256(tool),
        },
        "scenario_results": results,
        "checks": checks,
        "failure": failure,
        "claims": {
            "market_data_used": False,
            "model_fitted": False,
            "predictive_accuracy_verified": False,
            "predictive_edge_verified": False,
            "profitability_verified": False,
            "ai_uplift_verified": False,
            "paper_authority": False,
            "live_authority": False,
            "orders_submitted": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = run_probe()
    result["evidence_sha256"] = _canonical_sha256(result)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(_canonical_json({
        "checks": result["checks"],
        "evidence_sha256": result["evidence_sha256"],
        "output": str(output),
        "status": result["status"],
    }), flush=True)
    return 0 if result["status"] == "target_free_supervisor_behavior_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
