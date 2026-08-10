"""Run the frozen target-free Round 25 Fin-R1 supervisor scenario battery."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Mapping

from simple_ai_trading.polymarket_round25_ai_supervisor import (
    POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE,
    POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256,
    Round25AISupervisorConfig,
    Round25AISupervisorPacket,
    preload_round25_ai_supervisor,
    review_round25_ai_supervisor_packet,
    unload_round25_ai_supervisor,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-fin-r1-regime-supervisor-scenario-contract-v1.json"
)
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-fin-r1-regime-supervisor-scenario-contract-v2.json"
)
INFRASTRUCTURE_FAILURE = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-fin-r1-regime-supervisor-infrastructure-failure-v1-2026-08-10.json"
)
SUPERVISOR_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-fin-r1-regime-supervisor-contract-v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-fin-r1-regime-supervisor-host-probe-v2-2026-08-10.json"
)
BASE_CONTRACT_SHA256 = "9828aaf05deafe776c09a26c4e4cc0578762b9c53efa6c75ef83d4c3dc14dac4"
CONTRACT_SHA256 = "749f3a8b1e02523b6a11113db1498971d0e89bab034fefb57ec94cc9b98696f5"
INFRASTRUCTURE_FAILURE_SHA256 = (
    "034c69c5cbfdd5372ffa373a3568910ca18620a11af29a510fa6547db0651132"
)
_ACTION_ORDINAL = {
    "normal": 0,
    "cautious_50": 1,
    "defensive_25": 2,
    "halt_300s": 3,
}


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_contract(path: Path, expected_sha256: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    claimed = value.pop("contract_sha256", None)
    if claimed != expected_sha256 or _canonical_sha256(value) != claimed:
        raise ValueError("Round 25 AI supervisor scenario contract differs")
    value["contract_sha256"] = claimed
    return value


def _load_contract() -> tuple[dict[str, object], dict[str, object]]:
    contract = _validated_contract(CONTRACT, CONTRACT_SHA256)
    base = _validated_contract(BASE_CONTRACT, BASE_CONTRACT_SHA256)
    failure = json.loads(INFRASTRUCTURE_FAILURE.read_text(encoding="ascii"))
    if (
        contract.get("supersedes", {}).get("scenario_contract_v1_sha256")
        != BASE_CONTRACT_SHA256
        or contract.get("supersedes", {}).get("infrastructure_failure_v1_artifact_sha256")
        != INFRASTRUCTURE_FAILURE_SHA256
        or failure.get("artifact_sha256") != INFRASTRUCTURE_FAILURE_SHA256
        or base.get("supervisor_contract_sha256")
        != POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256
    ):
        raise ValueError("Round 25 AI supervisor scenario lineage differs")
    return contract, base


def _windows_gpu_names() -> tuple[str, ...]:
    if platform.system().lower() != "windows":
        return ()
    completed = subprocess.run(  # noqa: S603 - fixed local host inventory command
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _base_packet(scenario_id: str) -> Round25AISupervisorPacket:
    observed_at_ms = time.time_ns() // 1_000_000
    window_start_ms = observed_at_ms - 60_000
    return Round25AISupervisorPacket(
        condition_id="0x" + hashlib.sha256(
            f"round25-fin-r1-supervisor-{scenario_id}".encode("ascii")
        ).hexdigest(),
        window_start_ms=window_start_ms,
        observed_at_ms=observed_at_ms,
        expires_at_ms=observed_at_ms + 90_000,
        feature_source_chain_sha256=hashlib.sha256(
            f"round25-fin-r1-supervisor-target-free-source-{scenario_id}".encode("ascii")
        ).hexdigest(),
        clob_relative_spread_median_60s=0.008,
        clob_relative_spread_p95_60s=0.015,
        clob_top_executable_notional_p10_usd_60s=2_500.0,
        clob_book_receipt_age_p95_ms_60s=60.0,
        reference_receipt_age_p95_ms_60s=35.0,
        realized_volatility_60s=0.002,
        realized_volatility_300s=0.004,
        absolute_log_return_60s=0.001,
        absolute_log_return_300s=0.002,
        absolute_order_flow_imbalance_mean_60s=0.12,
        market_probability_range_60s=0.03,
        round_trip_cost_bps_p95_60s=15.0,
        portfolio_risk_utilization=0.08,
        current_condition_exposure_fraction=0.0,
        deterministic_gate_sha256=hashlib.sha256(
            b"round25-fin-r1-supervisor-deterministic-pass"
        ).hexdigest(),
    )


def _scenario_packet(
    *,
    scenario_id: str,
    overrides: Mapping[str, object],
) -> Round25AISupervisorPacket:
    base = _base_packet(scenario_id)
    allowed = {
        "clob_relative_spread_median_60s",
        "clob_relative_spread_p95_60s",
        "clob_top_executable_notional_p10_usd_60s",
        "clob_book_receipt_age_p95_ms_60s",
        "reference_receipt_age_p95_ms_60s",
        "realized_volatility_60s",
        "realized_volatility_300s",
        "absolute_log_return_60s",
        "absolute_log_return_300s",
        "absolute_order_flow_imbalance_mean_60s",
        "market_probability_range_60s",
        "round_trip_cost_bps_p95_60s",
        "portfolio_risk_utilization",
        "current_condition_exposure_fraction",
    }
    if not set(overrides).issubset(allowed):
        raise ValueError("Round 25 AI supervisor scenario overrides differ")
    values = base.identity_payload()
    values.update(overrides)
    values["packet_sha256"] = ""
    return Round25AISupervisorPacket(**values)


def _evaluate_checks(
    results: list[dict[str, object]],
    *,
    unloaded: bool,
) -> dict[str, bool]:
    by_id = {str(item["scenario_id"]): item for item in results}

    def severity(scenario_id: str) -> int:
        action = by_id[scenario_id].get("action")
        return _ACTION_ORDINAL.get(str(action), -1)

    single_stresses = (
        "mild_spread",
        "low_depth",
        "volatile",
        "risk_budget_elevated",
    )
    base = severity("benign_base")
    crisis = severity("combined_soft_crisis")
    actions = {item.get("action") for item in results}
    return {
        "all_seven_responses_valid": len(results) == 7
        and all(item.get("valid_model_response") is True for item in results),
        "all_responses_within_30_seconds": len(results) == 7
        and all(
            isinstance(item.get("latency_seconds"), (int, float))
            and float(item["latency_seconds"]) <= 30.0
            for item in results
        ),
        "all_responses_fully_gpu_resident": len(results) == 7
        and all(
            isinstance(item.get("gpu_residency_ratio"), (int, float))
            and float(item["gpu_residency_ratio"]) >= 0.99
            for item in results
        ),
        "benign_repeat_matches": by_id.get("benign_base", {}).get("action")
        == by_id.get("benign_repeat", {}).get("action"),
        "benign_at_most_cautious": 0 <= base <= _ACTION_ORDINAL["cautious_50"],
        "combined_crisis_at_least_defensive": crisis
        >= _ACTION_ORDINAL["defensive_25"],
        "at_least_two_distinct_actions": None not in actions and len(actions) >= 2,
        "benign_to_each_single_stress_to_combined_is_nondecreasing": base >= 0
        and crisis >= 0
        and all(base <= severity(scenario) <= crisis for scenario in single_stresses),
        "model_unloaded_after_probe": unloaded,
    }


def run_probe() -> dict[str, object]:
    contract, base_contract = _load_contract()
    order = base_contract.get("scenario_order")
    overrides = base_contract.get("scenario_overrides")
    if not isinstance(order, list) or len(order) != 7 or not isinstance(overrides, Mapping):
        raise ValueError("Round 25 AI supervisor scenario ledger differs")
    config = Round25AISupervisorConfig().validated()
    results: list[dict[str, object]] = []
    preload = None
    unloaded = False
    failure = None
    try:
        print("preloading fin-r1:8b", flush=True)
        preload = preload_round25_ai_supervisor(config)
        for index, raw_id in enumerate(order, start=1):
            scenario_id = str(raw_id)
            changes = overrides.get(scenario_id)
            if not isinstance(changes, Mapping):
                raise ValueError("Round 25 AI supervisor scenario changes differ")
            packet = _scenario_packet(scenario_id=scenario_id, overrides=changes)
            review = review_round25_ai_supervisor_packet(packet, config).validated()
            telemetry = review.telemetry
            results.append({
                "scenario_id": scenario_id,
                "changed_fields": dict(changes),
                "packet_sha256": packet.packet_sha256,
                "advisory_sha256": review.advisory.advisory_sha256,
                "action": review.advisory.regime_action
                if review.advisory.valid_model_response
                else None,
                "fail_closed_action": review.advisory.regime_action,
                "maximum_size_multiplier": review.advisory.maximum_size_multiplier,
                "cooldown_ms": review.advisory.cooldown_ms,
                "valid_model_response": review.advisory.valid_model_response,
                "failure_code": review.advisory.failure_code,
                "latency_seconds": None
                if telemetry is None
                else telemetry.measured_latency_seconds,
                "prompt_token_count": None
                if telemetry is None
                else telemetry.provider_prompt_eval_count,
                "output_token_count": None
                if telemetry is None
                else telemetry.provider_eval_count,
                "prompt_sha256": None if telemetry is None else telemetry.prompt_sha256,
                "response_sha256": None if telemetry is None else telemetry.response_sha256,
                "gpu_residency_ratio": None
                if telemetry is None
                else telemetry.residency.vram_to_model_ratio,
                "telemetry_sha256": None
                if telemetry is None
                else telemetry.telemetry_sha256,
            })
            print(
                f"scenario {index}/7 {scenario_id}: "
                f"action={results[-1]['action']} failure={review.advisory.failure_code}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - preserve bounded host failure evidence
        failure = {"type": type(exc).__name__, "message": str(exc)[:240]}
    finally:
        try:
            print("unloading fin-r1:8b", flush=True)
            unload_round25_ai_supervisor(config)
            unloaded = True
        except Exception as exc:  # noqa: BLE001 - unload failure must remain visible
            failure = failure or {"type": type(exc).__name__, "message": str(exc)[:240]}
    checks = _evaluate_checks(results, unloaded=unloaded) if len(results) == 7 else {
        key: False for key in base_contract["checks"]
    }
    module = ROOT / "src" / "simple_ai_trading" / "polymarket_round25_ai_supervisor.py"
    tool = Path(__file__).resolve()
    return {
        "schema_version": "polymarket-round25-fin-r1-regime-supervisor-host-probe-v2",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "target_free_supervisor_mechanics_verified"
        if all(checks.values()) and failure is None
        else "target_free_supervisor_mechanics_failed",
        "host": {
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "gpu_names": list(_windows_gpu_names()),
        },
        "candidate": {
            "candidate_id": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.candidate_id,
            "model": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model,
            "digest": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest,
            "preload_residency": None if preload is None else preload.asdict(),
        },
        "source": {
            "scenario_contract_path": CONTRACT.relative_to(ROOT).as_posix(),
            "scenario_contract_sha256": CONTRACT_SHA256,
            "scenario_contract_file_sha256": _file_sha256(CONTRACT),
            "base_scenario_contract_path": BASE_CONTRACT.relative_to(ROOT).as_posix(),
            "base_scenario_contract_sha256": BASE_CONTRACT_SHA256,
            "base_scenario_contract_file_sha256": _file_sha256(BASE_CONTRACT),
            "infrastructure_failure_path": (
                INFRASTRUCTURE_FAILURE.relative_to(ROOT).as_posix()
            ),
            "infrastructure_failure_artifact_sha256": (
                INFRASTRUCTURE_FAILURE_SHA256
            ),
            "infrastructure_failure_file_sha256": _file_sha256(
                INFRASTRUCTURE_FAILURE
            ),
            "supervisor_contract_path": SUPERVISOR_CONTRACT.relative_to(ROOT).as_posix(),
            "supervisor_contract_sha256": POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256,
            "supervisor_contract_file_sha256": _file_sha256(SUPERVISOR_CONTRACT),
            "supervisor_module_path": module.relative_to(ROOT).as_posix(),
            "supervisor_module_sha256": _file_sha256(module),
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
    return 0 if result["status"] == "target_free_supervisor_mechanics_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
