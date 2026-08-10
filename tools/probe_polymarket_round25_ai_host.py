"""Probe the frozen Round 25 AI candidates without market data or outcomes."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

from simple_ai_trading.polymarket_round25_ai import (
    POLYMARKET_ROUND25_AI_CANDIDATES,
    POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256,
    Round25AIAdvisoryPacket,
    Round25AIConfig,
    preload_round25_ai_candidate,
    review_round25_ai_packet,
    unload_round25_ai_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v7-2026-08-10.json"
)
FAILED_V1_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-2026-08-10.json"
)
FAILED_V1_SHA256 = "882549d5d9ab8df97ca31c28842c583016ce6bb3511cb7a182cd4ade89e5b2d0"
FAILED_V2_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v2-2026-08-10.json"
)
FAILED_V2_SHA256 = "68c1b00b8659fd45c3f1a14667ff94944e18673eb635c1d8540b5565c41e045a"
FAILED_V3_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v3-2026-08-10.json"
)
FAILED_V3_SHA256 = "853afaefc3af5f2f36a68f9c5bc4eec24f70880ae7aff917ff26659f49af8242"
FAILED_V4_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v4-2026-08-10.json"
)
FAILED_V4_SHA256 = "8be591f9bdf20930bf2d611eae065e7efba8bb93bb78917a8500f1421dc8c370"
FAILED_V5_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v5-2026-08-10.json"
)
FAILED_V5_SHA256 = "fda0a5a9fc8a61402e64849163d9c16657303d13b7fdff1b6161d76d6fe2f4d3"
PRIOR_V6_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-ai-risk-advisory-host-probe-v6-2026-08-10.json"
)
PRIOR_V6_SHA256 = "febf68717042b2d69c410c34e86c0703feb56bacae5163430b8e226e3eb1b540"


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
    return tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )


def _fixture_packet() -> Round25AIAdvisoryPacket:
    decision_time_ms = time.time_ns() // 1_000_000
    event_start_ms = decision_time_ms - decision_time_ms % 300_000
    fixture_identity = hashlib.sha256(
        b"round25-ai-host-probe-no-market-data-v1"
    ).hexdigest()
    return Round25AIAdvisoryPacket(
        condition_id=f"0x{fixture_identity}",
        event_start_ms=event_start_ms,
        decision_time_ms=decision_time_ms,
        expires_at_ms=decision_time_ms + 10_000,
        feature_source_chain_sha256=hashlib.sha256(
            b"fixture-feature-source-no-market-data"
        ).hexdigest(),
        ml_candidate_id="causal-multitask-tcn-residual-v1",
        ml_artifact_sha256=hashlib.sha256(
            b"fixture-unfitted-model-artifact"
        ).hexdigest(),
        ml_prediction_sha256=hashlib.sha256(
            b"fixture-unfitted-model-prediction"
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
            b"fixture-deterministic-gate-pass"
        ).hexdigest(),
    )


def _candidate_probe(candidate_id: str) -> dict[str, object]:
    config = Round25AIConfig(candidate_id=candidate_id).validated()
    preloaded = None
    result = None
    failure = None
    unload_failure = None
    try:
        preloaded = preload_round25_ai_candidate(config)
        packet = _fixture_packet()
        result = review_round25_ai_packet(packet, config)
    except Exception as exc:  # noqa: BLE001 - preserve bounded host failure evidence
        failure = {"type": type(exc).__name__, "message": str(exc)[:240]}
    finally:
        try:
            unload_round25_ai_candidate(config)
        except Exception as exc:  # noqa: BLE001 - unload failure must remain visible
            unload_failure = {"type": type(exc).__name__, "message": str(exc)[:240]}
    telemetry = None if result is None else result.telemetry
    checks = {
        "preload_full_gpu_residency": bool(
            preloaded is not None and preloaded.fully_gpu_resident
        ),
        "valid_coherent_typed_response": bool(
            result is not None and result.advisory.valid_model_response
        ),
        "post_inference_exact_digest_full_gpu_residency": bool(
            telemetry is not None
            and telemetry.model_digest == config.candidate.digest
            and telemetry.residency.fully_gpu_resident
        ),
        "provider_latency_within_contract": bool(
            telemetry is not None and telemetry.measured_latency_seconds <= 10.0
        ),
        "model_unloaded_after_probe": unload_failure is None,
        "target_outcome_resolution_and_pnl_absent": True,
        "trading_authority_absent": bool(
            result is not None
            and result.live_authority is False
            and result.paper_authority is False
            and result.order_submitted is False
        ),
    }
    return {
        "candidate": {
            "candidate_id": config.candidate.candidate_id,
            "model": config.candidate.model,
            "digest": config.candidate.digest,
            "parameter_size": config.candidate.parameter_size,
            "quantization": config.candidate.quantization,
            "upstream_revision": config.candidate.upstream_revision,
        },
        "fixture": {
            "kind": "deterministic_numerical_runtime_fixture_no_market_data",
            "conditions": 1,
            "targets": 0,
            "outcomes": 0,
            "resolutions": 0,
            "pnl_observations": 0,
        },
        "checks": checks,
        "passed": all(checks.values()) and failure is None,
        "preload_residency": None if preloaded is None else preloaded.asdict(),
        "review_result": None if result is None else {
            **result.identity_payload(),
            "result_sha256": result.result_sha256,
        },
        "failure": failure,
        "unload_failure": unload_failure,
    }


def run_probe(*, candidate_ids: tuple[str, ...]) -> dict[str, object]:
    frozen_ids = tuple(candidate.candidate_id for candidate in POLYMARKET_ROUND25_AI_CANDIDATES)
    if not candidate_ids or any(candidate_id not in frozen_ids for candidate_id in candidate_ids):
        raise ValueError("Round 25 AI probe candidate set differs")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Round 25 AI probe candidates are duplicated")
    candidates = tuple(_candidate_probe(candidate_id) for candidate_id in candidate_ids)
    source = ROOT / "src" / "simple_ai_trading" / "polymarket_round25_ai.py"
    contract = (
        ROOT
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-risk-advisory-contract-v6.json"
    )
    tool = Path(__file__).resolve()
    failed_v1 = json.loads(FAILED_V1_PATH.read_text(encoding="ascii"))
    if failed_v1.get("evidence_sha256") != FAILED_V1_SHA256:
        raise ValueError("Round 25 AI failed v1 probe identity differs")
    failed_v2 = json.loads(FAILED_V2_PATH.read_text(encoding="ascii"))
    if failed_v2.get("evidence_sha256") != FAILED_V2_SHA256:
        raise ValueError("Round 25 AI failed v2 probe identity differs")
    failed_v3 = json.loads(FAILED_V3_PATH.read_text(encoding="ascii"))
    if failed_v3.get("evidence_sha256") != FAILED_V3_SHA256:
        raise ValueError("Round 25 AI failed v3 probe identity differs")
    failed_v4 = json.loads(FAILED_V4_PATH.read_text(encoding="ascii"))
    if failed_v4.get("evidence_sha256") != FAILED_V4_SHA256:
        raise ValueError("Round 25 AI failed v4 probe identity differs")
    failed_v5 = json.loads(FAILED_V5_PATH.read_text(encoding="ascii"))
    if failed_v5.get("evidence_sha256") != FAILED_V5_SHA256:
        raise ValueError("Round 25 AI failed v5 probe identity differs")
    prior_v6 = json.loads(PRIOR_V6_PATH.read_text(encoding="ascii"))
    if prior_v6.get("evidence_sha256") != PRIOR_V6_SHA256:
        raise ValueError("Round 25 AI prior v6 probe identity differs")
    return {
        "schema_version": "polymarket-round25-ai-risk-advisory-host-probe-v7",
        "captured_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": (
            "runtime_mechanics_verified"
            if all(bool(candidate["passed"]) for candidate in candidates)
            else "runtime_mechanics_failed"
        ),
        "host": {
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "gpu_names": list(_windows_gpu_names()),
        },
        "supersedes_runtime_probe": {
            "path": PRIOR_V6_PATH.relative_to(ROOT).as_posix(),
            "evidence_sha256": PRIOR_V6_SHA256,
            "prior_failure_paths": [
                FAILED_V1_PATH.relative_to(ROOT).as_posix(),
                FAILED_V2_PATH.relative_to(ROOT).as_posix(),
                FAILED_V3_PATH.relative_to(ROOT).as_posix(),
                FAILED_V4_PATH.relative_to(ROOT).as_posix(),
                FAILED_V5_PATH.relative_to(ROOT).as_posix(),
            ],
            "prior_failure_evidence_sha256": [
                FAILED_V1_SHA256,
                FAILED_V2_SHA256,
                FAILED_V3_SHA256,
                FAILED_V4_SHA256,
                FAILED_V5_SHA256,
            ],
            "failure_boundary": "qwen3_4b_returned_allow_for_all_ten_target_free_v2_safety_scenarios",
            "correction": "add_non_numeric_monotonic_risk_dimension_instruction_without_changing_model_authority_or_deadline",
            "model_response_observed": True,
        },
        "source": {
            "ai_contract_sha256": POLYMARKET_ROUND25_AI_RISK_ADVISORY_CONTRACT_SHA256,
            "contract_path": contract.relative_to(ROOT).as_posix(),
            "contract_file_sha256": _file_sha256(contract),
            "ai_module_path": source.relative_to(ROOT).as_posix(),
            "ai_module_sha256": _file_sha256(source),
            "probe_tool_path": tool.relative_to(ROOT).as_posix(),
            "probe_tool_sha256": _file_sha256(tool),
        },
        "candidates": list(candidates),
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
    parser.add_argument(
        "--candidate",
        action="append",
        choices=tuple(
            candidate.candidate_id
            for candidate in POLYMARKET_ROUND25_AI_CANDIDATES
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    candidate_ids = tuple(args.candidate or (
        candidate.candidate_id for candidate in POLYMARKET_ROUND25_AI_CANDIDATES
    ))
    result = run_probe(candidate_ids=candidate_ids)
    result["evidence_sha256"] = _canonical_sha256(result)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing probe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(_canonical_json({
        "output": str(output),
        "status": result["status"],
        "candidates": [
            {
                "candidate_id": candidate["candidate"]["candidate_id"],
                "passed": candidate["passed"],
            }
            for candidate in result["candidates"]
        ],
    }))
    return 0 if result["status"] == "runtime_mechanics_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
