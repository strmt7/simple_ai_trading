"""Run the hash-bound Round 39 rolling-refit, utility, and local-AI ablation."""

from __future__ import annotations

import argparse
import ctypes
from datetime import UTC, datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.ai_trade_veto import AI_MODELS  # noqa: E402
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.rolling_refit_ai_veto import (  # noqa: E402
    BATCH_SIZE,
    MAX_CASES,
    benchmark_rolling_ai_model,
    build_rolling_ai_cases,
    rolling_case_set_sha256,
)
from simple_ai_trading.rolling_refit_model import (  # noqa: E402
    EVALUATION_MONTHS,
    run_rolling_refit_screen,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 39
DESIGN_SCHEMA = "causal-refit-utility-ai-ablation-design-v3"
BINDING_SCHEMA = "round-039-causal-refit-utility-ai-execution-binding-v1"
REPORT_SCHEMA = "causal-refit-utility-ai-ablation-report-v1"
SOURCE_CERTIFICATE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 39 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 39 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 39 design identity is invalid")
    governance = design.get("governance")
    source = design.get("source_and_accounting_contract")
    walk = design.get("causal_walk_forward_contract")
    model = design.get("model_contract")
    thresholds = design.get("monthly_action_threshold_contract")
    ai = design.get("ai_ablation_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (governance, source, walk, model, thresholds, ai)
    ):
        raise ValueError("Round 39 design sections are incomplete")
    for field in (
        "unregistered_hyperparameter_search_permitted",
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "oracle_feature_or_runtime_label_use_permitted",
        "ai_can_create_reverse_or_increase_trades",
        "ai_model_selection_on_round39_evaluation_permitted",
        "profitability_target_override_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 39 governance must deny {field}")
    if (
        source.get("symbols") != list(SYMBOLS)
        or source.get("round_trip_execution_charge_bps") != 12.0
        or source.get("derivatives_directional_features_permitted") is not False
        or walk.get("evaluation_months") != list(EVALUATION_MONTHS)
        or model.get("candidate_count") != 4
        or model.get("expected_model_artifact_count") != 60
        or thresholds.get("maximum_action_probability_grid")
        != [0.40, 0.45, 0.50, 0.55, 0.60]
        or thresholds.get("direction_probability_margin_grid")
        != [0.05, 0.10, 0.15, 0.20]
        or ai.get("models") != list(AI_MODELS)
        or ai.get("batch_size") != BATCH_SIZE
        or ai.get("maximum_cases") != MAX_CASES
    ):
        raise ValueError("Round 39 implementation and design contracts differ")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 39 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 39 execution binding identity is invalid")
    source = binding.get("source_certificate")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_sha256")
        != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 39 source certificate binding is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 39 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 39 execution requires a clean worktree")
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "merge-base",
                "--is-ancestor",
                implementation_commit,
                "HEAD",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Round 39 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 39 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 39 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        bound_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        current_oid = _git("rev-parse", f"HEAD:{relative_path}")
        if bound_oid != expected_oid or current_oid != expected_oid:
            raise ValueError(f"Round 39 bound blob changed: {relative_path}")
    return binding, claimed, implementation_commit


def _memory_evidence() -> dict[str, object]:
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            return {
                "source": "windows_process_memory_counters",
                "current_working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "current_pagefile_bytes": int(counters.PagefileUsage),
                "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
            }
    return {"source": "unavailable"}


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.last_write = 0.0
        self.sequence = 0

    def __call__(self, phase: str, detail: Mapping[str, object]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "round-039-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "phase": phase,
            "detail": dict(detail),
            "elapsed_seconds": time.perf_counter() - self.started,
            "memory": _memory_evidence(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        print(_canonical_json(payload), flush=True)
        now = time.monotonic()
        status_changed = str(detail.get("status") or "") in {"started", "complete"}
        if status_changed or now - self.last_write >= 30.0:
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            self.last_write = now


def _utility_uplift(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (str(item["architecture"]), str(item["weighting"])): item
        for item in candidates
    }
    output: list[dict[str, object]] = []
    for architecture in (
        "shared_two_stage_hurdle_lightgbm",
        "per_symbol_direct_multiclass_lightgbm",
    ):
        equal = lookup[(architecture, "equal")]
        utility = lookup[(architecture, "bounded_economic_utility")]
        equal_replay = equal["aggregate_replay"]
        utility_replay = utility["aggregate_replay"]
        equal_losses = [
            float(month["evaluation_classification"]["multiclass_log_loss"])
            for month in equal["monthly_results"]
        ]
        utility_losses = [
            float(month["evaluation_classification"]["multiclass_log_loss"])
            for month in utility["monthly_results"]
        ]
        output.append(
            {
                "architecture": architecture,
                "horizon_minutes": equal["horizon_minutes"],
                "mean_monthly_log_loss_delta_utility_minus_equal": (
                    sum(utility_losses) / len(utility_losses)
                    - sum(equal_losses) / len(equal_losses)
                ),
                "aggregate_trade_delta_utility_minus_equal": (
                    int(utility_replay["total_trades"])
                    - int(equal_replay["total_trades"])
                ),
                "aggregate_mean_net_bps_delta_utility_minus_equal": (
                    float(utility_replay["mean_net_bps"])
                    - float(equal_replay["mean_net_bps"])
                ),
                "equal_ai_entry_support_passed": equal[
                    "ai_entry_support_passed"
                ],
                "utility_ai_entry_support_passed": utility[
                    "ai_entry_support_passed"
                ],
                "equal_ml_gate_passed": equal["aggregate_ml_gate_passed"],
                "utility_ml_gate_passed": utility["aggregate_ml_gate_passed"],
            }
        )
    return output


def _round38_static_comparison(failure_path: Path) -> dict[str, object]:
    failure = _read_object(failure_path, "Round 38 failure analysis")
    canonical = dict(failure)
    claimed = str(canonical.pop("analysis_sha256", ""))
    if (
        failure.get("round") != 38
        or claimed
        != "f5b693fd00891ae8af9f93eec6925837c16c019aa246864ec9af29100496917c"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 38 failure analysis identity is invalid")
    return {
        "analysis_sha256": claimed,
        "shared_hurdle_h120_static": failure["best_viability_candidate"],
        "per_symbol_direct_h30_static": failure["calibration_decay_case"][
            "viability"
        ],
    }


def run(arguments: argparse.Namespace) -> dict[str, object]:
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root / "status.json")
    started = time.perf_counter()
    design, design_sha = _validate_design(arguments.design.resolve())
    binding, binding_sha, implementation_commit = _validate_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        source_certificate_path=arguments.source_certificate.resolve(),
    )
    progress(
        "binding",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": implementation_commit,
        },
    )
    panel, price_source = load_verified_minute_panel(
        arguments.database.resolve(), progress=progress
    )
    premium, funding, derivatives_source = load_derivatives_state(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.source_certificate.resolve(),
        progress=progress,
    )
    dataset = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=progress,
    )
    progress(
        "dataset",
        {
            "status": "complete",
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "model_feature_count": dataset.price_flow_feature_count,
            "matrix_bytes": int(dataset.features.nbytes),
            "source_exclusions": dict(dataset.source_exclusions),
        },
    )
    del panel, premium, funding
    screen = run_rolling_refit_screen(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    cases = build_rolling_ai_cases(dataset, screen.support_candidates)
    case_payload = {
        "schema_version": "causal-rolling-ai-veto-case-set-v1",
        "case_set_sha256": rolling_case_set_sha256(cases),
        "support_candidate_ids": [
            candidate.candidate_id for candidate in screen.support_candidates
        ],
        "cases": [case.evidence_payload() for case in cases],
    }
    write_json_atomic(
        evidence_root / "ai-cases.json", case_payload, indent=2, sort_keys=True
    )
    ai_reports: list[dict[str, object]] = []
    if cases:
        for model in AI_MODELS:
            ai_reports.append(
                benchmark_rolling_ai_model(
                    cases,
                    model=model,
                    base_url=arguments.ollama_url,
                    timeout_seconds=arguments.ai_timeout_seconds,
                    progress=progress,
                ).asdict()
            )
    support_ids = [
        candidate.candidate_id for candidate in screen.support_candidates
    ]
    passed_ids = [candidate.candidate_id for candidate in screen.passed_candidates]
    ai_passed = [
        str(item["model"]) for item in ai_reports if item["uplift_gate_passed"]
    ]
    status = "rejected"
    if passed_ids:
        status = "ml_gate_observed_consumed_research_only"
    if ai_passed:
        status = "ai_uplift_observed_consumed_research_only"
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": status,
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "backend": {
            "kind": screen.backend_kind,
            "device": screen.backend_device,
            "gpu_first_requested": True,
            "python_dependencies": {
                package: metadata.version(package)
                for package in ("lightgbm", "numpy", "scipy")
            },
        },
        "source_evidence": dataset.source_evidence.asdict(),
        "dataset": {
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "model_feature_count": dataset.price_flow_feature_count,
            "derivatives_risk_and_accounting_feature_count": (
                len(dataset.feature_names) - dataset.price_flow_feature_count
            ),
            "features_dtype": str(dataset.features.dtype),
            "features_bytes": int(dataset.features.nbytes),
            "source_exclusions": dict(dataset.source_exclusions),
            "horizons_minutes": [30, 120],
            "persistent_feature_copy_created": False,
        },
        "monthly_schedules": list(screen.schedules),
        "candidate_results": list(screen.candidate_results),
        "model_artifacts": [artifact.asdict() for artifact in screen.model_artifacts],
        "utility_weighting_uplift": _utility_uplift(screen.candidate_results),
        "round38_static_comparison": _round38_static_comparison(
            arguments.round38_failure.resolve()
        ),
        "superseded_execution_attempts": [
            {
                "evidence_root": "external://round39-causal-refit-utility-ai-20260712-v1",
                "last_status_file_sha256": "b9cfd3acc6db8a0ef8941d5ea5c2060ee2dfbf60db06ff5348c3bd4bc4dde0e5",
                "ml_artifacts_completed": 60,
                "ai_batches_attempted": 2,
                "ai_batch_latency_seconds": [91.7, 85.679],
                "ai_batch_schema_valid": [True, False],
                "decision_content_or_outcomes_inspected": False,
                "report_created": False,
                "reason_stopped": "unbounded AI runtime and verbose-schema failure",
            },
            {
                "evidence_root": "external://round39-causal-refit-utility-ai-20260712-v2",
                "last_status_file_sha256": "cdd723a44bb9dedfb5785c7d1aa5c79492ef955769655dd2238a0c8eda286e11",
                "ml_artifacts_completed": 60,
                "ai_batches_attempted": 2,
                "ai_batch_latency_seconds": [51.679, 42.926],
                "ai_batch_schema_valid": [False, False],
                "decision_content_or_outcomes_inspected": False,
                "report_created": False,
                "reason_stopped": "JSON Schema did not constrain array cardinality or case-ID enum",
            },
        ],
        "ai_entry_support_candidates": support_ids,
        "aggregate_ml_gate_passed_candidates": passed_ids,
        "ai_case_set": {
            "path": "ai-cases.json",
            "cases": len(cases),
            "case_set_sha256": case_payload["case_set_sha256"],
            "batch_size": BATCH_SIZE,
        },
        "ai_reports": ai_reports,
        "ai_uplift_gate_passed_models": ai_passed,
        "model_selection_on_evaluation_permitted": False,
        "ai_model_selection_on_evaluation_permitted": False,
        "selection_confirmation_accessed": False,
        "terminal_2026_accessed": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "runtime_evidence": {
            "elapsed_seconds": time.perf_counter() - started,
            "logical_cpu_count": os.cpu_count(),
            "memory": _memory_evidence(),
        },
        "report_canonical_sha256": "PENDING",
    }
    canonical = dict(report)
    canonical.pop("report_canonical_sha256")
    report["report_canonical_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(evidence_root / "report.json", report, indent=2, sort_keys=True)
    progress(
        "report",
        {
            "status": "complete",
            "report_canonical_sha256": report["report_canonical_sha256"],
            "support_candidates": len(support_ids),
            "ml_gate_candidates": len(passed_ids),
            "ai_cases": len(cases),
            "ai_uplift_models": len(ai_passed),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-039-causal-refit-utility-ai-ablation-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-039-causal-refit-utility-ai-execution-binding.json",
    )
    parser.add_argument(
        "--round38-failure",
        type=Path,
        default=research / "round-038-failure-analysis.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ai-timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
