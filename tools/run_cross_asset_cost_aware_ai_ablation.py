"""Run the hash-bound Round 37 cross-asset ML and local-AI ablation."""

from __future__ import annotations

import argparse
import ctypes
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.ai_trade_veto import (  # noqa: E402
    AI_MODELS,
    benchmark_ai_veto_model,
    build_ai_trade_cases,
    case_set_sha256,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    HORIZONS_MINUTES,
    SYMBOLS,
    build_cross_asset_dataset,
    load_verified_minute_panel,
)
from simple_ai_trading.cross_asset_cost_model import (  # noqa: E402
    evaluate_candidates,
    train_fixed_candidates,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


DESIGN_SCHEMA = "cross-asset-cost-aware-ai-ablation-design-v2"
BINDING_SCHEMA = "round-037-cross-asset-ai-execution-binding-v1"
REPORT_SCHEMA = "cross-asset-cost-aware-ai-ablation-report-v1"
ROUND = 37


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
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root is not an object")
    return payload


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
        raise ValueError("Round 37 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 37 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError("Round 37 design canonical hash is invalid")
    if design.get("schema_version") != DESIGN_SCHEMA or design.get("round") != ROUND:
        raise ValueError("Round 37 design identity is invalid")
    governance = design.get("governance")
    if not isinstance(governance, Mapping):
        raise ValueError("Round 37 governance is missing")
    for field in (
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "ai_can_create_or_reverse_trades",
        "oracle_feature_or_runtime_label_use_permitted",
        "profitability_target_override_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 37 governance must deny {field}")
    target = design.get("decision_and_target_contract")
    if not isinstance(target, Mapping):
        raise ValueError("Round 37 target contract is missing")
    if target.get("horizons_minutes") != list(HORIZONS_MINUTES):
        raise ValueError("Round 37 horizons do not match implementation")
    if target.get("round_trip_execution_charge_bps") != 12.0:
        raise ValueError("Round 37 execution charge does not match implementation")
    ai = design.get("ai_ablation_contract")
    if not isinstance(ai, Mapping) or ai.get("models") != list(AI_MODELS):
        raise ValueError("Round 37 AI models do not match implementation")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 37 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if claimed != _canonical_sha256(canonical):
        raise ValueError("Round 37 binding canonical hash is invalid")
    if binding.get("schema_version") != BINDING_SCHEMA or binding.get("round") != ROUND:
        raise ValueError("Round 37 binding identity is invalid")
    if binding.get("design_sha256") != design_sha256:
        raise ValueError("Round 37 binding design hash differs")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 37 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 37 runner requires a clean worktree")
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
        raise ValueError("Round 37 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 37 binding blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 37 binding blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        actual_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        if actual_oid != expected_oid:
            raise ValueError(f"Round 37 bound blob changed: {relative_path}")
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
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
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
        elapsed = time.perf_counter() - self.started
        payload = {
            "schema_version": "round-037-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "phase": phase,
            "detail": dict(detail),
            "elapsed_seconds": elapsed,
            "memory": _memory_evidence(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        print(_canonical_json(payload), flush=True)
        now = time.monotonic()
        status_changed = str(detail.get("status") or "") in {"started", "complete"}
        if status_changed or now - self.last_write >= 30.0:
            write_json_atomic(self.path, payload, indent=2, sort_keys=True)
            self.last_write = now


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-037-cross-asset-cost-aware-ai-ablation-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=ROOT
        / "docs/model-research/action-value/round-037-cross-asset-ai-execution-binding.json",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "data/market_data.sqlite",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ai-timeout-seconds", type=float, default=60.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    evidence_root = args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root / "status.json")
    started = time.perf_counter()
    design, design_sha = _validate_design(args.design.resolve())
    binding, binding_sha, implementation_commit = _validate_binding(
        args.binding.resolve(),
        design_sha256=design_sha,
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
    panel, source_evidence = load_verified_minute_panel(
        args.database,
        progress=progress,
    )
    progress("source_load", {"status": "complete", "symbols": list(SYMBOLS)})
    dataset = build_cross_asset_dataset(
        panel,
        source_evidence,
        progress=progress,
    )
    progress(
        "dataset",
        {
            "status": "complete",
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "matrix_bytes": int(dataset.features.nbytes),
        },
    )
    del panel
    trained = train_fixed_candidates(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=args.compute_backend,
        progress=progress,
    )
    candidates = evaluate_candidates(dataset, trained, progress=progress)
    ai_cases = build_ai_trade_cases(dataset, trained, candidates)
    cases_payload = {
        "schema_version": "causal-ai-trade-veto-case-set-v1",
        "case_set_sha256": case_set_sha256(ai_cases),
        "cases": [item.evidence_payload() for item in ai_cases],
    }
    write_json_atomic(
        evidence_root / "ai-cases.json",
        cases_payload,
        indent=2,
        sort_keys=True,
    )
    ai_reports = []
    if ai_cases:
        for model in AI_MODELS:
            ai_reports.append(
                benchmark_ai_veto_model(
                    ai_cases,
                    model=model,
                    base_url=args.ollama_url,
                    timeout_seconds=args.ai_timeout_seconds,
                    progress=progress,
                )
            )
    else:
        progress(
            "ai_veto",
            {
                "status": "complete",
                "reason": "no_shared_ml_cases_met_frozen_calibration_support",
            },
        )
    viable = [item for item in candidates if item.viability_gate_passed]
    ai_uplift = [item for item in ai_reports if item.uplift_gate_passed]
    status = (
        "diagnostic_complete_candidates_require_confirmation_no_authority"
        if viable
        else "rejected"
    )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": status,
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "source_evidence": source_evidence.asdict(),
        "dataset": {
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
            "symbols": list(SYMBOLS),
            "horizons_minutes": list(HORIZONS_MINUTES),
            "features_dtype": str(dataset.features.dtype),
            "features_bytes": int(dataset.features.nbytes),
            "persistent_feature_copy_created": False,
        },
        "backend": {
            "kind": trained.backend_kind,
            "device": trained.backend_device,
            "gpu_first_requested": args.compute_backend != "cpu",
        },
        "model_artifacts": [item.asdict() for item in trained.model_artifacts],
        "candidate_results": [item.asdict() for item in candidates],
        "viability_gate_passed_candidates": [
            {"family": item.family, "horizon_minutes": item.horizon_minutes}
            for item in viable
        ],
        "ai_case_set": {
            "path": "ai-cases.json",
            "case_set_sha256": cases_payload["case_set_sha256"],
            "cases": len(ai_cases),
        },
        "ai_reports": [item.asdict() for item in ai_reports],
        "ai_uplift_gate_passed_models": [item.model for item in ai_uplift],
        "runtime_evidence": {
            "elapsed_seconds": time.perf_counter() - started,
            "memory": _memory_evidence(),
            "logical_cpu_count": os.cpu_count(),
        },
        "selection_confirmation_accessed": False,
        "terminal_2026_accessed": False,
        "model_selection_on_viability_permitted": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    report["report_canonical_sha256"] = _canonical_sha256(report)
    write_json_atomic(
        evidence_root / "report.json",
        report,
        indent=2,
        sort_keys=True,
    )
    progress(
        "complete",
        {
            "status": status,
            "report_canonical_sha256": report["report_canonical_sha256"],
            "viability_candidates": len(viable),
            "ai_uplift_models": len(ai_uplift),
        },
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(f"Round 37 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        _canonical_json(
            {
                "status": report["status"],
                "report_canonical_sha256": report["report_canonical_sha256"],
                "viability_candidates": len(
                    report["viability_gate_passed_candidates"]
                ),
                "ai_uplift_models": len(report["ai_uplift_gate_passed_models"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
