"""Correct the Round 25 AI v3 scenario report without rerunning inference."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_round25_ai import (
    Round25AIConfig,
    unload_round25_ai_candidate,
)
from probe_polymarket_round25_ai_scenarios import _evaluate_checks


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROBE = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-host-probe-v3-2026-08-10.json"
)
SOURCE_EVIDENCE_SHA256 = (
    "2ee0f848829a362a4b1957809a8c4ca86f7d103cd726057f21f40e6eb1a7f246"
)
SCENARIO_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-contract-v3.json"
)
SCENARIO_CONTRACT_SHA256 = (
    "4a6efba5ced12a7c0e076fe8644d8ea06f093d765ec006a083a1e68efe69279c"
)
CORRECTION_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-correction-contract-v1.json"
)
CORRECTION_CONTRACT_SHA256 = (
    "271c9d964224cea77ed7df1a68a39ae077f20de868849640aac24f9a4e80dc2d"
)
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-scenario-host-probe-v3-correction-2026-08-10.json"
)


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


def _load_self_hashed(path: Path, expected_sha256: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    claimed = value.pop("contract_sha256", None)
    if claimed != expected_sha256 or _canonical_sha256(value) != claimed:
        raise ValueError("Round 25 AI correction contract identity differs")
    value["contract_sha256"] = claimed
    return value


def build_correction() -> dict[str, object]:
    source = json.loads(SOURCE_PROBE.read_text(encoding="ascii"))
    claimed = source.pop("evidence_sha256", None)
    if claimed != SOURCE_EVIDENCE_SHA256 or _canonical_sha256(source) != claimed:
        raise ValueError("Round 25 AI source scenario evidence differs")
    source["evidence_sha256"] = claimed
    scenario_contract = _load_self_hashed(
        SCENARIO_CONTRACT,
        SCENARIO_CONTRACT_SHA256,
    )
    correction_contract = _load_self_hashed(
        CORRECTION_CONTRACT,
        CORRECTION_CONTRACT_SHA256,
    )
    results = source.get("scenario_results")
    expected_ids = [value["scenario_id"] for value in scenario_contract["scenarios"]]
    if (
        source.get("failure") is not None
        or not isinstance(results, list)
        or len(results) != 11
        or [value.get("scenario_id") for value in results] != expected_ids
        or correction_contract["source"]["host_probe_v3_evidence_sha256"]
        != SOURCE_EVIDENCE_SHA256
    ):
        raise ValueError("Round 25 AI correction population differs")
    config = Round25AIConfig(candidate_id="qwen3-4b-risk-advisor-v1").validated()
    unload_round25_ai_candidate(config)
    checks = _evaluate_checks(results, unloaded=True)
    identities = [
        {
            "advisory_sha256": value["advisory"]["advisory_sha256"],
            "packet_sha256": value["packet_sha256"],
            "scenario_id": value["scenario_id"],
            "telemetry_sha256": _canonical_sha256(value["telemetry"]),
        }
        for value in results
    ]
    tool = Path(__file__).resolve()
    evaluator = ROOT / "tools" / "probe_polymarket_round25_ai_scenarios.py"
    return {
        "schema_version": "polymarket-round25-ai-risk-scenario-host-probe-v3-correction-v1",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": (
            "safety_behavior_mechanics_verified"
            if all(checks.values())
            else "safety_behavior_mechanics_failed"
        ),
        "source_probe_path": SOURCE_PROBE.relative_to(ROOT).as_posix(),
        "source_probe_evidence_sha256": SOURCE_EVIDENCE_SHA256,
        "source_probe_file_sha256": _file_sha256(SOURCE_PROBE),
        "scenario_contract_path": SCENARIO_CONTRACT.relative_to(ROOT).as_posix(),
        "scenario_contract_sha256": SCENARIO_CONTRACT_SHA256,
        "correction_contract_path": CORRECTION_CONTRACT.relative_to(ROOT).as_posix(),
        "correction_contract_sha256": CORRECTION_CONTRACT_SHA256,
        "scenario_result_identities": identities,
        "scenario_result_identity_root_sha256": _canonical_sha256(identities),
        "actions": [
            {"scenario_id": value["scenario_id"], "action": value["action"]}
            for value in results
        ],
        "maximum_measured_latency_seconds": max(
            float(value["telemetry"]["measured_latency_seconds"])
            for value in results
        ),
        "checks": checks,
        "model_inference_repeated": False,
        "exact_digest_confirmed_unloaded_after_correction": True,
        "source": {
            "correction_tool_path": tool.relative_to(ROOT).as_posix(),
            "correction_tool_sha256": _file_sha256(tool),
            "check_evaluator_path": evaluator.relative_to(ROOT).as_posix(),
            "check_evaluator_sha256": _file_sha256(evaluator),
        },
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
    result = build_correction()
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
        "model_inference_repeated": result["model_inference_repeated"],
        "output": str(output),
        "status": result["status"],
    }))
    return 0 if result["status"] == "safety_behavior_mechanics_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
