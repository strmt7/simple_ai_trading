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
    evaluate_round74_ai_mirror_consistency,
    round74_ai_contract_cases,
)
from simple_ai_trading.impact_absorption_ai_review_preparation import (  # noqa: E402
    round74_default_ai_review_model_panel,
)
from simple_ai_trading.impact_absorption_ai_runtime import (  # noqa: E402
    ROUND74_AI_RUNTIME_PRELOAD_TIMEOUT_SECONDS,
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
    unload_action_performed = False
    model_absent_after_cleanup_verified = False
    incomplete_reason = ""
    failure_class = ""
    failure_message = ""
    print(
        f"round74-ai-screen: preloading {arguments.model_name}",
        file=sys.stderr,
        flush=True,
    )
    try:
        preload_timeout = ROUND74_AI_RUNTIME_PRELOAD_TIMEOUT_SECONDS
        if arguments.hard_stop_wall_ns is not None:
            available_seconds = (
                arguments.hard_stop_wall_ns - time.time_ns()
            ) / 1_000_000_000 - arguments.unload_reserve_seconds
            if available_seconds <= 0.0:
                incomplete_reason = "hard_stop_preload_reserve_reached"
                raise TimeoutError(incomplete_reason)
            preload_timeout = min(preload_timeout, available_seconds)
        preload = preload_round74_ai_model(
            binding.runtime,
            binding.manifest,
            timeout_seconds=preload_timeout,
        )
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
    except Exception as exc:  # The report must survive every fail-closed path.
        failure_class = type(exc).__name__
        failure_message = str(exc)[:500]
        if not incomplete_reason:
            incomplete_reason = "screen_infrastructure_failure"
        print(
            f"round74-ai-screen: blocked {failure_class}: {failure_message}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        print(
            f"round74-ai-screen: unloading {arguments.model_name}",
            file=sys.stderr,
            flush=True,
        )
        try:
            unload_action_performed = unload_round74_ai_model(
                binding.runtime,
                binding.manifest,
            )
            model_absent_after_cleanup_verified = True
        except Exception as exc:
            cleanup_message = str(exc)[:500]
            failure_class = failure_class or type(exc).__name__
            failure_message = (
                f"{failure_message}; cleanup: {cleanup_message}"
                if failure_message
                else f"cleanup: {cleanup_message}"
            )
            incomplete_reason = "model_cleanup_failure"
    completed = len(results) == len(cases)
    mirror_checks = evaluate_round74_ai_mirror_consistency(results)
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
        "failure_class": failure_class,
        "failure_message": failure_message,
        "all_runtime_accepted": completed
        and all(result["runtime_accepted"] for result in results),
        "all_runtime_failures_closed": all(result["fail_closed"] for result in results),
        "all_semantic_expectations_passed": completed
        and all(result["semantic_passed"] for result in results),
        "mirror_checks": list(mirror_checks),
        "all_mirror_checks_passed": completed
        and all(check["passed"] for check in mirror_checks),
        "results": results,
        "model_unload_action_performed": unload_action_performed,
        "model_absent_after_cleanup_verified": (
            model_absent_after_cleanup_verified
        ),
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
    if failure_class:
        return 1
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
