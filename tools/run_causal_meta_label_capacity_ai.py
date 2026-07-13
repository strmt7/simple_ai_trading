"""Run the hash-bound Round 40 causal meta-label and finance-AI screen."""

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

from simple_ai_trading.causal_meta_label_ai_veto import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_MODEL,
    HASH_SAMPLE_MODULUS,
    MAX_CASES,
    benchmark_meta_label_ai_model,
    build_meta_label_ai_cases,
    meta_case_set_sha256,
)
from simple_ai_trading.causal_meta_label_model import (  # noqa: E402
    EVALUATION_MONTHS,
    MAXIMUM_ENTRIES_PER_SYMBOL_DAY,
    META_FEATURE_COUNT,
    META_PROBABILITY_GRID,
    PRIMARY_MARGIN_GRID,
    run_causal_meta_label_screen,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 40
DESIGN_SCHEMA = "causal-meta-label-capacity-ai-design-v1"
BINDING_SCHEMA = "round-040-causal-meta-label-execution-binding-v1"
REPORT_SCHEMA = "causal-meta-label-capacity-ai-report-v1"
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
        raise ValueError("Round 40 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 40 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 40 design identity is invalid")
    governance = design.get("governance")
    source = design.get("source_contract")
    schedule = design.get("development_schedule")
    primary = design.get("primary_model")
    meta_model = design.get("meta_label_model")
    capacity = design.get("causal_capacity_and_threshold_contract")
    ai = design.get("ai_ablation_contract")
    if not all(
        isinstance(item, Mapping)
        for item in (
            governance,
            source,
            schedule,
            primary,
            meta_model,
            capacity,
            ai,
        )
    ):
        raise ValueError("Round 40 design sections are incomplete")
    for field in (
        "unregistered_hyperparameter_search_permitted",
        "2025_h2_selection_confirmation_access_permitted",
        "2026_terminal_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "future_opportunity_set_ranking_permitted",
        "oracle_feature_or_runtime_label_use_permitted",
        "profitability_target_override_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 40 governance must deny {field}")
    if (
        source.get("symbols") != list(SYMBOLS)
        or source.get("dataset_rows") != 1_098_105
        or schedule.get("evaluation_months") != list(EVALUATION_MONTHS)
        or primary.get("models") != 18
        or meta_model.get("models") != 6
        or meta_model.get("feature_count") != META_FEATURE_COUNT
        or capacity.get("meta_probability_grid") != list(META_PROBABILITY_GRID)
        or capacity.get("primary_margin_grid") != list(PRIMARY_MARGIN_GRID)
        or capacity.get("threshold_cells_total") != 216
        or capacity.get("maximum_entries_per_symbol_per_utc_day")
        != MAXIMUM_ENTRIES_PER_SYMBOL_DAY
        or ai.get("maximum_cases") != MAX_CASES
    ):
        raise ValueError("Round 40 implementation and design contracts differ")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 40 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 40 execution binding identity is invalid")
    source = binding.get("source_certificate")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_sha256")
        != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 40 source certificate binding is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 40 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 40 execution requires a clean worktree")
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
        raise ValueError("Round 40 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 40 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 40 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        bound_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        current_oid = _git("rev-parse", f"HEAD:{relative_path}")
        if bound_oid != expected_oid or current_oid != expected_oid:
            raise ValueError(f"Round 40 bound blob changed: {relative_path}")
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
            "schema_version": "round-040-progress-v1",
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
        status_changed = str(detail.get("status") or "") in {
            "started",
            "complete",
        }
        if status_changed or now - self.last_write >= 30.0:
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            self.last_write = now


def run(arguments: argparse.Namespace) -> dict[str, object]:
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root / "status.json")
    started = time.perf_counter()
    design, design_sha = _validate_design(arguments.design.resolve())
    _, binding_sha, implementation_commit = _validate_binding(
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
    screen = run_causal_meta_label_screen(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    cases = build_meta_label_ai_cases(dataset, screen.passed_candidate)
    case_payload = {
        "schema_version": "causal-meta-label-ai-veto-case-set-v1",
        "case_set_sha256": meta_case_set_sha256(cases),
        "candidate_id": (
            screen.passed_candidate.candidate_id
            if screen.passed_candidate is not None
            else None
        ),
        "hash_sample_modulus": HASH_SAMPLE_MODULUS,
        "maximum_cases": MAX_CASES,
        "cases": [case.evidence_payload() for case in cases],
    }
    write_json_atomic(
        evidence_root / "ai-cases.json", case_payload, indent=2, sort_keys=True
    )
    ai_report: dict[str, object] | None = None
    ai_error: str | None = None
    if cases:
        try:
            ai_report = benchmark_meta_label_ai_model(
                cases,
                model=arguments.ai_model,
                base_url=arguments.ollama_url,
                timeout_seconds=arguments.ai_timeout_seconds,
                progress=progress,
            ).asdict()
        except Exception as exc:
            ai_error = f"{type(exc).__name__}: {exc}"
            progress(
                "round40_ai_veto",
                {"status": "complete", "model": arguments.ai_model, "error": ai_error},
            )
    ml_passed = screen.passed_candidate is not None
    ai_passed = bool(ai_report and ai_report["uplift_gate_passed"])
    status = "rejected"
    if ml_passed:
        status = "ml_gate_observed_consumed_development_only"
    if ai_passed:
        status = "ai_uplift_observed_consumed_development_only"
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
            "gpu_first_requested": arguments.compute_backend != "cpu",
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
            "meta_feature_count": META_FEATURE_COUNT,
            "features_dtype": str(dataset.features.dtype),
            "features_bytes": int(dataset.features.nbytes),
            "source_exclusions": dict(dataset.source_exclusions),
            "horizon_minutes": 30,
            "persistent_feature_copy_created": False,
        },
        "monthly_schedules": list(screen.schedules),
        "candidate_result": dict(screen.candidate_result),
        "model_artifacts": [
            artifact.asdict() for artifact in screen.model_artifacts
        ],
        "aggregate_ml_gate_passed": ml_passed,
        "ai_case_set": {
            "path": "ai-cases.json",
            "cases": len(cases),
            "case_set_sha256": case_payload["case_set_sha256"],
            "hash_sample_modulus": HASH_SAMPLE_MODULUS,
            "batch_size": BATCH_SIZE,
        },
        "ai_candidate_model": {
            "design_identity": "DianJin/DianJin-R1-7B",
            "runtime_identity": arguments.ai_model,
        },
        "ai_report": ai_report,
        "ai_error": ai_error,
        "ai_uplift_gate_passed": ai_passed,
        "selection_contaminated": True,
        "development_only": True,
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
            "ml_gate_passed": ml_passed,
            "ai_cases": len(cases),
            "ai_uplift_passed": ai_passed,
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-040-causal-meta-label-capacity-ai-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-040-causal-meta-label-execution-binding.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--ai-model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ai-timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
