"""Run one pinned local model through the frozen Round 74 AI safety screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.impact_absorption_ai_contract_screen import (  # noqa: E402
    ROUND74_AI_CONTRACT_SCREEN_SCHEMA_VERSION,
    evaluate_round74_ai_contract_outcome,
    round74_ai_contract_cases,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (  # noqa: E402
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_runtime import (  # noqa: E402
    preload_round74_ai_model,
    review_round74_ai_candidate,
    unload_round74_ai_model,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--hard-stop-wall-ns", type=int)
    parser.add_argument("--unload-reserve-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    bindings = {
        binding.runtime.model_name: binding
        for binding in round74_default_ai_review_model_panel()
    }
    if arguments.model_name not in bindings:
        raise ValueError("model name is not in the frozen default panel")
    binding = bindings[arguments.model_name]
    cases = round74_ai_contract_cases()
    started_wall_ns = time.time_ns()
    results: list[dict[str, object]] = []
    preload = None
    unloaded = False
    incomplete_reason = ""
    print(
        f"round74-ai-screen: preloading {arguments.model_name}",
        file=sys.stderr,
        flush=True,
    )
    try:
        preload = preload_round74_ai_model(binding.runtime, binding.manifest)
        print(
            "round74-ai-screen: preload passed with full GPU residency",
            file=sys.stderr,
            flush=True,
        )
        for index, case in enumerate(cases, start=1):
            if arguments.hard_stop_wall_ns is not None:
                reserve_ns = int(
                    (
                        binding.runtime.timeout_seconds
                        + arguments.unload_reserve_seconds
                    )
                    * 1_000_000_000
                )
                if time.time_ns() + reserve_ns >= arguments.hard_stop_wall_ns:
                    incomplete_reason = "hard_stop_unload_reserve_reached"
                    break
            request_wall_ns = time.time_ns()
            request = case.build_request(request_wall_ns)
            print(
                f"round74-ai-screen: case {index}/{len(cases)} {case.case_id}",
                file=sys.stderr,
                flush=True,
            )
            outcome = review_round74_ai_candidate(
                binding.runtime,
                binding.manifest,
                request,
                deterministic_risk_gate_passed=True,
                observed_wall_ns=request_wall_ns,
            )
            evidence = evaluate_round74_ai_contract_outcome(case, outcome)
            results.append(evidence)
            print(
                "round74-ai-screen: "
                f"{case.case_id} runtime={evidence['runtime_status']} "
                f"semantic_passed={evidence['semantic_passed']}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        print(
            f"round74-ai-screen: unloading {arguments.model_name}",
            file=sys.stderr,
            flush=True,
        )
        unloaded = unload_round74_ai_model(binding.runtime, binding.manifest)
    completed = len(results) == len(cases)
    report: dict[str, object] = {
        "schema_version": ROUND74_AI_CONTRACT_SCREEN_SCHEMA_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": binding.runtime.model_name,
        "model_manifest_sha256": binding.manifest.manifest_sha256,
        "model_artifact_sha256": binding.manifest.model_artifact_sha256,
        "preload_residency": None if preload is None else preload.asdict(),
        "full_gpu_residency_verified": bool(
            preload is not None and preload.fully_gpu_resident
        ),
        "case_count_required": len(cases),
        "case_count_completed": len(results),
        "complete": completed,
        "incomplete_reason": incomplete_reason,
        "all_runtime_accepted": completed
        and all(result["runtime_accepted"] for result in results),
        "all_runtime_failures_closed": all(result["fail_closed"] for result in results),
        "all_semantic_expectations_passed": completed
        and all(result["semantic_passed"] for result in results),
        "results": results,
        "model_unload_requested_and_verified": unloaded,
        "elapsed_wall_ns": time.time_ns() - started_wall_ns,
        "synthetic_non_market_contract_packets_only": True,
        "real_market_data_used": False,
        "financial_edge_tested": False,
        "profitability_claim": False,
        "drawdown_claim": False,
        "trading_authority": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
