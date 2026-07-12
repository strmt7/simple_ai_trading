"""Run the hash-bound Round 38 derivatives hurdle and local-AI ablation."""

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

from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    HORIZONS_MINUTES,
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_ai_veto import (  # noqa: E402
    AI_MODELS,
    benchmark_derivatives_ai_model,
    build_derivatives_ai_cases,
    derivatives_case_set_sha256,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.derivatives_hurdle_model import (  # noqa: E402
    ACTION_PROBABILITY_GRID,
    ARCHITECTURES,
    DIRECTION_MARGIN_GRID,
    FEATURE_SETS,
    run_fixed_model_screen,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 38
DESIGN_SCHEMA = "derivatives-hurdle-ai-ablation-design-v2"
BINDING_SCHEMA = "round-038-derivatives-hurdle-ai-execution-binding-v1"
REPORT_SCHEMA = "derivatives-hurdle-ai-ablation-report-v1"


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
        raise ValueError("Round 38 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 38 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 38 design identity is invalid")
    governance = design.get("governance")
    source = design.get("source_contract")
    model = design.get("model_contract")
    feature = design.get("feature_contract")
    thresholds = design.get("action_threshold_contract")
    ai = design.get("ai_ablation_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (governance, source, feature, model, thresholds, ai)
    ):
        raise ValueError("Round 38 design sections are incomplete")
    for field in (
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "ai_can_create_reverse_or_increase_trades",
        "oracle_feature_or_runtime_label_use_permitted",
        "profitability_target_override_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 38 governance must deny {field}")
    if (
        source.get("symbols") != list(SYMBOLS)
        or feature.get("ablation_feature_sets") != list(FEATURE_SETS)
        or model.get("architectures") != list(ARCHITECTURES)
        or model.get("fixed_ablation_count") != 32
        or thresholds.get("maximum_action_probability_grid")
        != list(ACTION_PROBABILITY_GRID)
        or thresholds.get("direction_probability_margin_grid")
        != list(DIRECTION_MARGIN_GRID)
        or ai.get("models") != list(AI_MODELS)
    ):
        raise ValueError("Round 38 implementation and design contracts differ")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 38 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 38 execution binding identity is invalid")
    source = binding.get("source_certificate")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_sha256")
        != "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 38 source certificate binding is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 38 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 38 execution requires a clean worktree")
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
        raise ValueError("Round 38 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 38 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 38 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        bound_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        current_oid = _git("rev-parse", f"HEAD:{relative_path}")
        if bound_oid != expected_oid or current_oid != expected_oid:
            raise ValueError(f"Round 38 bound blob changed: {relative_path}")
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
            "schema_version": "round-038-progress-v1",
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


def _feature_uplift(
    candidates: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (
            str(item["architecture"]),
            str(item["feature_set"]),
            int(item["horizon_minutes"]),
        ): item
        for item in candidates
    }
    output: list[dict[str, object]] = []
    for architecture in ARCHITECTURES:
        for horizon in HORIZONS_MINUTES:
            baseline = lookup[(architecture, "price_flow_only", horizon)]
            augmented = lookup[
                (architecture, "price_flow_plus_premium_and_funding", horizon)
            ]
            baseline_replay = baseline["viability_replay"]
            augmented_replay = augmented["viability_replay"]
            output.append(
                {
                    "architecture": architecture,
                    "horizon_minutes": horizon,
                    "viability_log_loss_delta_augmented_minus_price": (
                        float(augmented["viability_classification"]["multiclass_log_loss"])
                        - float(baseline["viability_classification"]["multiclass_log_loss"])
                    ),
                    "viability_brier_delta_augmented_minus_price": (
                        float(augmented["viability_classification"]["multiclass_brier_score"])
                        - float(baseline["viability_classification"]["multiclass_brier_score"])
                    ),
                    "both_action_thresholds_selected": (
                        baseline["selected_action_threshold"] is not None
                        and augmented["selected_action_threshold"] is not None
                    ),
                    "viability_mean_net_bps_delta": (
                        float(augmented_replay["mean_net_bps"])
                        - float(baseline_replay["mean_net_bps"])
                        if baseline["selected_action_threshold"] is not None
                        and augmented["selected_action_threshold"] is not None
                        else None
                    ),
                    "price_flow_viability_gate_passed": baseline[
                        "viability_gate_passed"
                    ],
                    "augmented_viability_gate_passed": augmented[
                        "viability_gate_passed"
                    ],
                }
            )
    return output


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
            "price_flow_feature_count": dataset.price_flow_feature_count,
            "matrix_bytes": int(dataset.features.nbytes),
            "source_exclusions": dict(dataset.source_exclusions),
        },
    )
    del panel, premium, funding
    screen = run_fixed_model_screen(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    cases = build_derivatives_ai_cases(dataset, screen.passed_candidates)
    case_payload = {
        "schema_version": "causal-derivatives-ai-veto-case-set-v1",
        "case_set_sha256": derivatives_case_set_sha256(cases),
        "cases": [item.evidence_payload() for item in cases],
    }
    write_json_atomic(
        evidence_root / "ai-cases.json", case_payload, indent=2, sort_keys=True
    )
    ai_reports = []
    if cases:
        for model in AI_MODELS:
            ai_reports.append(
                benchmark_derivatives_ai_model(
                    cases,
                    model=model,
                    base_url=arguments.ollama_url,
                    timeout_seconds=arguments.ai_timeout_seconds,
                    progress=progress,
                ).asdict()
            )
    passed_ids = [item.candidate_id for item in screen.passed_candidates]
    ai_passed = [
        str(item["model"]) for item in ai_reports if item["uplift_gate_passed"]
    ]
    status = "rejected"
    if passed_ids:
        status = "ml_viability_candidates_research_only"
    if ai_passed:
        status = "ai_uplift_observed_research_only"
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
            "price_flow_feature_count": dataset.price_flow_feature_count,
            "derivatives_feature_count": (
                len(dataset.feature_names) - dataset.price_flow_feature_count
            ),
            "features_dtype": str(dataset.features.dtype),
            "features_bytes": int(dataset.features.nbytes),
            "source_exclusions": dict(dataset.source_exclusions),
            "horizons_minutes": list(HORIZONS_MINUTES),
            "persistent_feature_copy_created": False,
        },
        "candidate_results": list(screen.candidate_results),
        "model_artifacts": [item.asdict() for item in screen.model_artifacts],
        "derivatives_feature_uplift": _feature_uplift(screen.candidate_results),
        "viability_gate_passed_candidates": passed_ids,
        "ai_case_set": {
            "path": "ai-cases.json",
            "cases": len(cases),
            "case_set_sha256": case_payload["case_set_sha256"],
        },
        "ai_reports": ai_reports,
        "ai_uplift_gate_passed_models": ai_passed,
        "model_selection_on_viability_permitted": False,
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
            "viability_candidates": len(passed_ids),
            "ai_cases": len(cases),
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-038-derivatives-hurdle-ai-ablation-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-038-derivatives-hurdle-ai-execution-binding.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ai-timeout-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
