"""Run the hash-bound Round 50 path-bounded competing-risk experiment."""

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

from simple_ai_trading.action_hurdle_tcn_model import (  # noqa: E402
    build_action_hurdle_temporal_dataset,
)
from simple_ai_trading.barrier_competing_risk_analysis import (  # noqa: E402
    candidate_diagnostics,
    economic_gate,
    leverage_sensitivity,
    mechanism_ablation_gate,
    replay_fixed_trades,
    select_fixed_policy_trades,
)
from simple_ai_trading.barrier_competing_risk_tcn_model import (  # noqa: E402
    CANDIDATES,
    EVENT_CLASSES,
    HORIZON_MINUTES,
    RECEPTIVE_FIELD_STEPS,
    SEEDS,
    barrier_event_classes,
    train_barrier_competing_risk_candidates,
)
from simple_ai_trading.barrier_payoff_data import (  # noqa: E402
    BarrierSpecification,
    build_barrier_payoff_dataset,
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


ROUND = 50
DESIGN_SCHEMA = "path-bounded-competing-risk-tcn-design-v1"
BINDING_SCHEMA = "round-050-path-bounded-competing-risk-execution-binding-v1"
REPORT_SCHEMA = "path-bounded-competing-risk-tcn-report-v1"
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
EXPECTED_PREDECESSOR_DATASET_SHA256 = (
    "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
)
EXPECTED_BARRIER_DATASET_SHA256 = (
    "31c7713339cff9ad12f3bae02475743d09b2248bfc1b85e02e1f3306a699e774"
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
        raise ValueError("Round 50 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 50 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 50 design identity is invalid")
    data = design.get("data_contract")
    target = design.get("target_contract")
    model = design.get("model_contract")
    claims = design.get("claims")
    if not all(isinstance(item, Mapping) for item in (data, target, model, claims)):
        raise ValueError("Round 50 design sections are incomplete")
    candidates = model.get("candidates")
    encoder = model.get("shared_encoder")
    if (
        data.get("symbols") != list(SYMBOLS)
        or target.get("horizon_minutes") != HORIZON_MINUTES
        or target.get("event_classes_per_side") != EVENT_CLASSES
        or not isinstance(candidates, list)
        or [item.get("id") for item in candidates if isinstance(item, Mapping)]
        != list(CANDIDATES)
        or model.get("seeds") != list(SEEDS)
        or not isinstance(encoder, Mapping)
        or encoder.get("receptive_field_decision_steps") != RECEPTIVE_FIELD_STEPS
    ):
        raise ValueError("Round 50 implementation and design contracts differ")
    if (
        claims.get("profitability_claim_permitted") is not False
        or claims.get("trading_authority_permitted") is not False
        or claims.get("selection_contaminated") is not True
    ):
        raise ValueError("Round 50 frozen claims are invalid")
    ai_contract = design.get("ai_contract")
    if (
        not isinstance(ai_contract, Mapping)
        or ai_contract.get("order_authority") is not False
        or ai_contract.get("financial_edge_tested_by_safety_benchmark") is not False
        or ai_contract.get("ai_can_repair_failed_numerical_or_economic_gate")
        is not False
    ):
        raise ValueError("Round 50 AI authority contract is invalid")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 50 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 50 binding identity is invalid")
    source = binding.get("source_certificate")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_sha256") != SOURCE_CANONICAL_SHA256
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 50 source certificate binding is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 50 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 50 execution requires a clean worktree")
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
        raise ValueError("Round 50 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 50 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 50 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        bound_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        current_oid = _git("rev-parse", f"HEAD:{relative_path}")
        if bound_oid != expected_oid or current_oid != expected_oid:
            raise ValueError(f"Round 50 bound blob changed: {relative_path}")
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
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            return {
                "working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                "pagefile_bytes": int(counters.PagefileUsage),
                "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
            }
    return {
        "working_set_bytes": None,
        "peak_working_set_bytes": None,
        "pagefile_bytes": None,
        "peak_pagefile_bytes": None,
    }


class ProgressWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()
        self.sequence = 0

    def __call__(self, event: str, details: Mapping[str, object]) -> None:
        self.sequence += 1
        payload = {
            "schema_version": "round-050-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "event": event,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "memory": _memory_evidence(),
            "details": dict(details),
        }
        write_json_atomic(self.path, payload, indent=2, sort_keys=True)
        print(_canonical_json(payload), flush=True)


def _write_failure_status(path: Path, error: Exception, elapsed_seconds: float) -> None:
    payload = {
        "schema_version": "round-050-progress-v1",
        "round": ROUND,
        "event": "round50_failed",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "memory": _memory_evidence(),
        "details": {"error_type": type(error).__name__, "error": str(error)},
    }
    write_json_atomic(path, payload, indent=2, sort_keys=True)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "torch", "torch-directml"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _prepare_evidence_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError("Round 50 evidence root must be new or empty")
    path.mkdir(parents=True, exist_ok=True)


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    evidence_root = arguments.evidence_root.resolve()
    _prepare_evidence_root(evidence_root)
    progress = ProgressWriter(evidence_root / "status.json")
    try:
        design, design_sha = _validate_design(arguments.design.resolve())
        binding, binding_sha, implementation_commit = _validate_binding(
            arguments.binding.resolve(),
            design_sha256=design_sha,
            source_certificate_path=arguments.source_certificate.resolve(),
        )
        progress(
            "round50_contracts_validated",
            {
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
        source = build_derivatives_hurdle_dataset(
            panel,
            premium,
            funding,
            derivatives_source,
            progress=progress,
        )
        temporal = build_action_hurdle_temporal_dataset(source)
        if temporal.dataset_sha256 != EXPECTED_PREDECESSOR_DATASET_SHA256:
            raise ValueError("Round 50 reconstructed a different predecessor corpus")
        specification = BarrierSpecification(
            horizon_minutes=60,
            stop_volatility_multiple=1.0,
            take_profit_to_stop_ratio=2.0,
            minimum_stop_bps=24.0,
            maximum_stop_bps=80.0,
            round_trip_execution_charge_bps=12.0,
        )
        barrier = build_barrier_payoff_dataset(
            panel, funding, source, temporal, specification
        )
        if barrier.dataset_sha256 != EXPECTED_BARRIER_DATASET_SHA256:
            raise ValueError("Round 50 reconstructed a different barrier corpus")
        event_classes = barrier_event_classes(barrier)
        progress(
            "round50_dataset_validated",
            {
                "predecessor_dataset_sha256": temporal.dataset_sha256,
                "barrier_dataset_sha256": barrier.dataset_sha256,
                "timestamps": temporal.timestamps,
                "rows": barrier.rows,
            },
        )
        bundles, preflight = train_barrier_competing_risk_candidates(
            temporal,
            barrier,
            model_dir=evidence_root / "models",
            prediction_dir=evidence_root / "predictions",
            compute_backend=arguments.compute_backend,
            progress=progress,
        )
        diagnostics = {
            candidate_id: candidate_diagnostics(
                bundle, temporal, barrier, event_classes
            )
            for candidate_id, bundle in bundles.items()
        }
        mechanism_gate = mechanism_ablation_gate(diagnostics)
        policy: dict[str, object] = {}
        gates: dict[str, object] = {}
        leverage: dict[str, object] = {}
        for candidate_index, candidate_id in enumerate(CANDIDATES):
            trades = select_fixed_policy_trades(
                bundles[candidate_id], temporal, barrier
            )
            replay = replay_fixed_trades(
                trades,
                temporal,
                barrier,
                candidate_index=candidate_index,
            )
            gate = economic_gate(
                replay,
                quality_gate_passed=bool(
                    diagnostics[candidate_id]["quality_gate"]["passed"]
                ),
                mechanism_gate_passed=bool(mechanism_gate["passed"]),
                candidate_id=candidate_id,
            )
            policy[candidate_id] = replay
            gates[candidate_id] = gate
            leverage[candidate_id] = leverage_sensitivity(replay, gate)
            progress(
                "round50_candidate_analyzed",
                {
                    "candidate_id": candidate_id,
                    "quality_gate_passed": diagnostics[candidate_id]["quality_gate"][
                        "passed"
                    ],
                    "economic_gate_passed": gate["passed"],
                    "closed_trades": replay["scenarios"]["base"]["closed_trades"],
                    "base_return_fraction": replay["scenarios"]["base"][
                        "total_return_fraction"
                    ],
                },
            )
        elapsed = time.perf_counter() - started
        report: dict[str, object] = {
            "schema_version": REPORT_SCHEMA,
            "round": ROUND,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": implementation_commit,
            "source_certificate_sha256": SOURCE_CANONICAL_SHA256,
            "dataset": {
                "symbols": list(SYMBOLS),
                "source_resolution_seconds": 60,
                "decision_interval_seconds": 300,
                "predecessor_dataset_sha256": temporal.dataset_sha256,
                "barrier_dataset_sha256": barrier.dataset_sha256,
                "timestamps": temporal.timestamps,
                "rows": barrier.rows,
                "specification": specification.asdict(),
                "synthetic_rows": 0,
                "selection_confirmation_or_terminal_rows_read": False,
            },
            "backend": preflight,
            "artifacts": {
                candidate_id: [artifact.asdict() for artifact in bundle.artifacts]
                for candidate_id, bundle in bundles.items()
            },
            "target_baselines": {
                candidate_id: bundle.target_baselines.asdict()
                for candidate_id, bundle in bundles.items()
            },
            "training_history": {
                candidate_id: list(bundle.training_history)
                for candidate_id, bundle in bundles.items()
            },
            "diagnostics": diagnostics,
            "mechanism_gate": mechanism_gate,
            "fixed_policy": policy,
            "economic_gates": gates,
            "leverage_sensitivity": leverage,
            "ai": {
                "risk_reviewer": design["ai_contract"],
                "paired_uplift_run": False,
                "paired_uplift_not_run_reason": (
                    "No deterministic candidate passed every mandatory gate."
                    if not any(gate["passed"] for gate in gates.values())
                    else "A separate hash-bound paired AI-veto experiment is required."
                ),
                "trading_authority": False,
            },
            "claims": {
                "profitability_claim": False,
                "trading_authority": False,
                "selection_contaminated": True,
                "beta_research_only": True,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "memory": _memory_evidence(),
                "package_versions": _package_versions(),
                "persistent_feature_or_target_copy_created": False,
            },
        }
        report["report_canonical_sha256"] = _canonical_sha256(report)
        report_path = evidence_root / "report.json"
        write_json_atomic(report_path, report, indent=2, sort_keys=True)
        progress(
            "round50_complete",
            {
                "report_path": str(report_path),
                "report_file_sha256": _file_sha256(report_path),
                "report_canonical_sha256": report["report_canonical_sha256"],
                "elapsed_seconds": elapsed,
            },
        )
        return report
    except Exception as exc:
        _write_failure_status(
            evidence_root / "status.json", exc, time.perf_counter() - started
        )
        raise


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-050-path-bounded-competing-risk-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-050-path-bounded-competing-risk-tcn-binding.json",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend", choices=("directml", "cpu"), default="directml"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    summary = {
        "report_canonical_sha256": report["report_canonical_sha256"],
        "mechanism_gate": report["mechanism_gate"],
        "economic_gates": report["economic_gates"],
        "profitability_claim": False,
        "trading_authority": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
