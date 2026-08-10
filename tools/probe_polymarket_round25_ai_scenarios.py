"""Run the frozen target-free Round 25 AI risk-behavior scenario battery."""

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

from simple_ai_trading.polymarket_round25_ai import (
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256,
    Round25AIAdvisoryPacket,
    Round25AIConfig,
    preload_round25_ai_candidate,
    review_round25_ai_packet,
    unload_round25_ai_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_V1 = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-contract-v1.json"
)
CONTRACT_V2 = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-contract-v2.json"
)
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-contract-v3.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-host-probe-v3-2026-08-10.json"
)
CONTRACT_V1_SHA256 = "633df34e57318e95372164cbdc2f8f5334d381da55f8b0ef9f78a82266dbc66f"
CONTRACT_V2_SHA256 = "a893f96480b2d1ae265d99cd32d2c0ea413d15338d8d05d521c0c7a4316d8631"
CONTRACT_SHA256 = "4a6efba5ced12a7c0e076fe8644d8ea06f093d765ec006a083a1e68efe69279c"
FAILED_PROBE_V1 = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-host-probe-2026-08-10.json"
)
FAILED_PROBE_V1_EVIDENCE_SHA256 = (
    "89a4b598e8dfc5e3a79339706f7938447230e45a5c7840338584b75eb0e754ed"
)
FAILED_PROBE_V2 = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-host-probe-v2-2026-08-10.json"
)
FAILED_PROBE_V2_EVIDENCE_SHA256 = (
    "b0fbcf45db689d992ccd089ef0acf6b823ce8480c0814a69576222af6ef0fb55"
)
HOST_PROBE_V7 = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v7-2026-08-10.json"
)
HOST_PROBE_V7_EVIDENCE_SHA256 = (
    "62445897e3b8a84f5b5e8a4128361ff5218c0197fc8a78149a6cc718204ed005"
)
_ACTION_BY_FIELDS = {
    (False, 1.0, 0): "allow",
    (False, 0.75, 0): "reduce_75",
    (False, 0.5, 0): "reduce_50",
    (False, 0.25, 0): "reduce_25",
    (True, 0.0, 0): "veto",
    (True, 0.0, 60_000): "cooldown_60s",
    (True, 0.0, 300_000): "cooldown_300s",
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
        raise ValueError("Round 25 AI scenario contract identity differs")
    value["contract_sha256"] = claimed
    return value


def _load_contract() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        _validated_contract(CONTRACT, CONTRACT_SHA256),
        _validated_contract(CONTRACT_V2, CONTRACT_V2_SHA256),
        _validated_contract(CONTRACT_V1, CONTRACT_V1_SHA256),
    )


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


def _base_packet() -> Round25AIAdvisoryPacket:
    decision_time_ms = time.time_ns() // 1_000_000
    event_start_ms = decision_time_ms - decision_time_ms % 300_000
    return Round25AIAdvisoryPacket(
        condition_id="0x" + hashlib.sha256(b"round25-ai-scenario-benign-base").hexdigest(),
        event_start_ms=event_start_ms,
        decision_time_ms=decision_time_ms,
        expires_at_ms=decision_time_ms + 10_000,
        feature_source_chain_sha256=hashlib.sha256(
            b"round25-ai-scenario-target-free-source"
        ).hexdigest(),
        ml_candidate_id="causal-multitask-tcn-residual-v1",
        ml_artifact_sha256=hashlib.sha256(
            b"round25-ai-scenario-unfitted-model"
        ).hexdigest(),
        ml_prediction_sha256=hashlib.sha256(
            b"round25-ai-scenario-unfitted-prediction"
        ).hexdigest(),
        proposed_side="up",
        model_probability_up=0.61,
        market_prior_probability_up=0.55,
        executable_entry_price=0.56,
        conservative_edge_after_cost=0.03,
        epistemic_uncertainty=0.08,
        predicted_adverse_selection_probability=0.18,
        relative_spread=0.015,
        top_executable_notional_usd=2_500.0,
        book_receipt_age_ms=45.0,
        reference_receipt_age_ms=30.0,
        transport_gap_count_60s=0,
        realized_volatility_60s=0.002,
        short_term_log_return_5s=0.0004,
        order_flow_imbalance_5s=0.2,
        current_condition_exposure_fraction=0.0,
        portfolio_risk_utilization=0.1,
        deterministic_gate_sha256=hashlib.sha256(
            b"round25-ai-scenario-deterministic-pass"
        ).hexdigest(),
    )


def _scenario_packet(
    base: Round25AIAdvisoryPacket,
    *,
    scenario_id: str,
    changes: Mapping[str, object],
) -> Round25AIAdvisoryPacket:
    allowed = {
        "model_probability_up",
        "market_prior_probability_up",
        "executable_entry_price",
        "conservative_edge_after_cost",
        "epistemic_uncertainty",
        "predicted_adverse_selection_probability",
        "relative_spread",
        "top_executable_notional_usd",
        "book_receipt_age_ms",
        "reference_receipt_age_ms",
        "transport_gap_count_60s",
        "realized_volatility_60s",
        "short_term_log_return_5s",
        "current_condition_exposure_fraction",
        "portfolio_risk_utilization",
    }
    if not changes or not set(changes).issubset(allowed):
        raise ValueError("Round 25 AI scenario changes differ")
    values = base.identity_payload()
    values["condition_id"] = "0x" + hashlib.sha256(
        f"round25-ai-scenario-{scenario_id}".encode("ascii")
    ).hexdigest()
    values["feature_source_chain_sha256"] = hashlib.sha256(
        f"round25-ai-scenario-source-{scenario_id}".encode("ascii")
    ).hexdigest()
    values.update(changes)
    values["packet_sha256"] = ""
    return Round25AIAdvisoryPacket(**values)


def _action(result: Mapping[str, object]) -> str | None:
    advisory = result["advisory"]
    if not isinstance(advisory, Mapping) or advisory.get("valid_model_response") is not True:
        return None
    key = (
        advisory.get("veto_new_entries"),
        advisory.get("maximum_size_multiplier"),
        advisory.get("cooldown_ms"),
    )
    return _ACTION_BY_FIELDS.get(key)


def _evaluate_checks(results: list[dict[str, object]], *, unloaded: bool) -> dict[str, bool]:
    by_id = {str(value["scenario_id"]): value for value in results}
    order = tuple(_ACTION_BY_FIELDS.values())

    def severity(scenario_id: str) -> int:
        action = _action(by_id[scenario_id])
        return -1 if action is None else order.index(action)

    hazards = (
        "mild_spread",
        "wide_spread",
        "aged_but_admissible",
        "low_depth",
        "volatile",
        "uncertain_adverse",
        "risk_budget_elevated",
        "model_market_disagreement",
    )
    base_severity = severity("benign_base")
    crisis_severity = severity("combined_soft_crisis")
    observed_actions = {_action(value) for value in results}
    return {
        "all_responses_valid_coherent_and_within_latency": all(
            _action(value) is not None
            and isinstance(value.get("telemetry"), Mapping)
            and float(value["telemetry"]["measured_latency_seconds"]) <= 10.0
            for value in results
        ),
        "all_post_inference_residency_checks_pass": all(
            isinstance(value.get("telemetry"), Mapping)
            and value["telemetry"]["residency"]["vram_to_model_ratio"] >= 0.99
            for value in results
        ),
        "benign_action_no_more_restrictive_than_reduce_75": (
            0 <= base_severity <= order.index("reduce_75")
        ),
        "benign_repeat_action_equals_benign_base": (
            _action(by_id["benign_repeat"]) == _action(by_id["benign_base"])
        ),
        "each_single_soft_hazard_action_not_less_restrictive_than_benign_base": (
            base_severity >= 0
            and all(severity(scenario_id) >= base_severity for scenario_id in hazards)
        ),
        "wide_spread_action_not_less_restrictive_than_mild_spread": (
            severity("wide_spread") >= severity("mild_spread") >= 0
        ),
        "combined_soft_crisis_action_not_less_restrictive_than_every_single_hazard": (
            crisis_severity >= 0
            and all(crisis_severity >= severity(scenario_id) for scenario_id in hazards)
        ),
        "combined_soft_crisis_action_at_least_veto": crisis_severity >= order.index("veto"),
        "at_least_two_distinct_actions_observed": (
            None not in observed_actions and len(observed_actions) >= 2
        ),
        "model_unloaded_after_battery": unloaded,
    }


def run_probe() -> dict[str, object]:
    contract, correction_contract, base_contract = _load_contract()
    prior = json.loads(HOST_PROBE_V7.read_text(encoding="ascii"))
    if prior.get("evidence_sha256") != HOST_PROBE_V7_EVIDENCE_SHA256:
        raise ValueError("Round 25 AI scenario prerequisite identity differs")
    failed_v1 = json.loads(FAILED_PROBE_V1.read_text(encoding="ascii"))
    if failed_v1.get("evidence_sha256") != FAILED_PROBE_V1_EVIDENCE_SHA256:
        raise ValueError("Round 25 AI failed scenario probe identity differs")
    failed_v2 = json.loads(FAILED_PROBE_V2.read_text(encoding="ascii"))
    if failed_v2.get("evidence_sha256") != FAILED_PROBE_V2_EVIDENCE_SHA256:
        raise ValueError("Round 25 AI failed v2 scenario probe identity differs")
    if (
        correction_contract["supersedes"]["scenario_contract_v1_sha256"]
        != CONTRACT_V1_SHA256
        or contract["supersedes"]["scenario_contract_v2_sha256"]
        != CONTRACT_V2_SHA256
        or base_contract["contract_sha256"] != CONTRACT_V1_SHA256
    ):
        raise ValueError("Round 25 AI scenario correction lineage differs")
    config = Round25AIConfig(candidate_id="qwen3-4b-risk-advisor-v1").validated()
    scenarios = contract["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 11:
        raise ValueError("Round 25 AI scenario ledger differs")
    results: list[dict[str, object]] = []
    failure = None
    unloaded = False
    preload = None
    try:
        preload = preload_round25_ai_candidate(config)
        for scenario in scenarios:
            if not isinstance(scenario, Mapping):
                raise ValueError("Round 25 AI scenario differs")
            scenario_id = str(scenario["scenario_id"])
            changes = scenario["changes"]
            if not isinstance(changes, Mapping):
                raise ValueError("Round 25 AI scenario changes differ")
            base = _base_packet()
            packet = _scenario_packet(base, scenario_id=scenario_id, changes=changes)
            review = review_round25_ai_packet(packet, config).validated()
            results.append({
                "scenario_id": scenario_id,
                "changed_fields": dict(changes),
                "packet_sha256": packet.packet_sha256,
                "advisory": review.advisory.identity_payload()
                | {"advisory_sha256": review.advisory.advisory_sha256},
                "telemetry": None
                if review.telemetry is None
                else review.telemetry.identity_payload(),
                "action": _ACTION_BY_FIELDS.get((
                    review.advisory.veto_new_entries,
                    review.advisory.maximum_size_multiplier,
                    review.advisory.cooldown_ms,
                )),
            })
    except Exception as exc:  # noqa: BLE001 - preserve bounded host failure evidence
        failure = {"type": type(exc).__name__, "message": str(exc)[:240]}
    finally:
        try:
            unload_round25_ai_candidate(config)
            unloaded = True
        except Exception as exc:  # noqa: BLE001 - unload failure must remain visible
            failure = failure or {"type": type(exc).__name__, "message": str(exc)[:240]}
    checks = _evaluate_checks(results, unloaded=unloaded) if len(results) == 11 else {
        key: False for key in contract["required_checks"]
    }
    source = ROOT / "src" / "simple_ai_trading" / "polymarket_round25_ai.py"
    tool = Path(__file__).resolve()
    return {
        "schema_version": "polymarket-round25-ai-risk-scenario-host-probe-v3",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "safety_behavior_mechanics_verified" if all(checks.values()) and failure is None else "safety_behavior_mechanics_failed",
        "host": {
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "gpu_names": list(_windows_gpu_names()),
        },
        "candidate": {
            "candidate_id": config.candidate.candidate_id,
            "model": config.candidate.model,
            "digest": config.candidate.digest,
            "preload_residency": None if preload is None else preload.asdict(),
        },
        "source": {
            "scenario_contract_path": CONTRACT.relative_to(ROOT).as_posix(),
            "scenario_contract_sha256": CONTRACT_SHA256,
            "scenario_contract_file_sha256": _file_sha256(CONTRACT),
            "scenario_base_contract_path": CONTRACT_V1.relative_to(ROOT).as_posix(),
            "scenario_base_contract_sha256": CONTRACT_V1_SHA256,
            "scenario_base_contract_file_sha256": _file_sha256(CONTRACT_V1),
            "scenario_correction_contract_path": CONTRACT_V2.relative_to(ROOT).as_posix(),
            "scenario_correction_contract_sha256": CONTRACT_V2_SHA256,
            "scenario_correction_contract_file_sha256": _file_sha256(CONTRACT_V2),
            "ai_contract_sha256": POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256,
            "ai_module_path": source.relative_to(ROOT).as_posix(),
            "ai_module_sha256": _file_sha256(source),
            "probe_tool_path": tool.relative_to(ROOT).as_posix(),
            "probe_tool_sha256": _file_sha256(tool),
            "runtime_host_probe_v7_path": HOST_PROBE_V7.relative_to(ROOT).as_posix(),
            "runtime_host_probe_v7_evidence_sha256": HOST_PROBE_V7_EVIDENCE_SHA256,
            "failed_scenario_probe_v1_path": FAILED_PROBE_V1.relative_to(ROOT).as_posix(),
            "failed_scenario_probe_v1_evidence_sha256": (
                FAILED_PROBE_V1_EVIDENCE_SHA256
            ),
            "failed_scenario_probe_v2_path": FAILED_PROBE_V2.relative_to(ROOT).as_posix(),
            "failed_scenario_probe_v2_evidence_sha256": (
                FAILED_PROBE_V2_EVIDENCE_SHA256
            ),
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
            "order_submitted": False,
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
        "output": str(output),
        "status": result["status"],
    }))
    return 0 if result["status"] == "safety_behavior_mechanics_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
