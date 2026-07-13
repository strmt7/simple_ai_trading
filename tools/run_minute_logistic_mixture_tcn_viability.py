"""Run the hash-bound Round 48 minute logistic-mixture TCN experiment."""

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
    SYMBOLS,
    load_verified_minute_panel,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    build_derivatives_hurdle_dataset,
    load_derivatives_state,
)
from simple_ai_trading.minute_logistic_mixture_analysis import (  # noqa: E402
    STRESS_EXECUTION_CHARGE_BPS,
    candidate_diagnostics,
    economic_gate,
    mixture_ablation_gate,
    replay_fixed_trades,
    select_fixed_policy_trades,
)
from simple_ai_trading.minute_logistic_mixture_tcn_model import (  # noqa: E402
    CANDIDATE_COMPONENTS,
    HORIZONS_MINUTES,
    RECEPTIVE_FIELD_STEPS,
    SEEDS,
    build_minute_temporal_dataset,
    train_minute_mixture_candidates,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 48
DESIGN_SCHEMA = "minute-logistic-mixture-tcn-design-v1"
BINDING_SCHEMA = "round-048-minute-logistic-mixture-execution-binding-v1"
REPORT_SCHEMA = "minute-logistic-mixture-tcn-report-v1"


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
        raise ValueError("Round 48 Git binding command failed") from exc


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 48 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 48 design identity is invalid")
    model = design.get("model_contract")
    data = design.get("data_contract")
    governance = design.get("governance")
    if not all(isinstance(item, Mapping) for item in (model, data, governance)):
        raise ValueError("Round 48 design sections are incomplete")
    candidates = model.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Round 48 design candidates are missing")
    design_components = {
        str(item["id"]): int(item["distribution_components"])
        for item in candidates
        if isinstance(item, Mapping)
    }
    shared_encoder = model.get("shared_encoder")
    if (
        data.get("symbols") != list(SYMBOLS)
        or data.get("forecast_horizons_minutes") != list(HORIZONS_MINUTES)
        or design_components != CANDIDATE_COMPONENTS
        or not isinstance(shared_encoder, Mapping)
        or shared_encoder.get("receptive_field_decision_steps")
        != RECEPTIVE_FIELD_STEPS
        or model.get("seeds") != list(SEEDS)
    ):
        raise ValueError("Round 48 implementation and design contracts differ")
    for field in (
        "selection_confirmation_2025_h2_access_permitted",
        "terminal_2026_access_permitted",
        "testnet_or_live_execution_permitted",
        "promotion_permitted",
        "leverage_permitted",
        "fee_or_slippage_reduction_permitted",
        "risk_gate_relaxation_permitted",
        "post_outcome_parameter_seed_candidate_threshold_or_policy_selection_permitted",
        "manual_graph_or_result_editing_permitted",
        "profitability_ai_or_mixture_uplift_claim_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 48 governance must deny {field}")
    return design, claimed


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 48 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 48 binding identity is invalid")
    source = binding.get("source_certificate")
    if (
        not isinstance(source, Mapping)
        or source.get("canonical_sha256")
        != "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 48 source certificate binding is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40:
        raise ValueError("Round 48 implementation commit is invalid")
    if _git("status", "--porcelain"):
        raise ValueError("Round 48 execution requires a clean worktree")
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
        raise ValueError("Round 48 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 48 bound blobs are missing")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 48 bound blob is invalid")
        relative_path = str(item.get("path") or "")
        expected_oid = str(item.get("git_blob_oid") or "")
        bound_oid = _git("rev-parse", f"{implementation_commit}:{relative_path}")
        current_oid = _git("rev-parse", f"HEAD:{relative_path}")
        if bound_oid != expected_oid or current_oid != expected_oid:
            raise ValueError(f"Round 48 bound blob changed: {relative_path}")
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
        success = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if success:
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
            "schema_version": "round-048-progress-v1",
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


def _package_versions() -> dict[str, str]:
    names = ("numpy", "scipy", "torch", "torch-directml")
    output: dict[str, str] = {}
    for name in names:
        try:
            output[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def _candidate_result(
    *,
    dataset: object,
    bundle: object,
    diagnostics: Mapping[str, object],
    ablation: Mapping[str, object] | None,
) -> dict[str, object]:
    trades = select_fixed_policy_trades(dataset, bundle)
    base = replay_fixed_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="base",
        execution_charge_bps=12.0,
    )
    stress = replay_fixed_trades(
        dataset,
        trades,
        candidate_id=bundle.candidate_id,
        scenario="stress",
        execution_charge_bps=STRESS_EXECUTION_CHARGE_BPS,
    )
    distribution_gate = diagnostics["distribution_gate"]
    action_gate = diagnostics["action_gate"]
    if not isinstance(distribution_gate, Mapping) or not isinstance(
        action_gate, Mapping
    ):
        raise ValueError("Round 48 candidate gates are invalid")
    quality_passed = bool(distribution_gate["passed"]) and bool(
        action_gate["passed"]
    )
    if ablation is not None:
        quality_passed = quality_passed and bool(ablation["passed"])
    economics = economic_gate(
        base,
        stress,
        quality_passed=quality_passed,
    )
    return {
        "candidate_id": bundle.candidate_id,
        "components": bundle.components,
        "artifacts": [item.asdict() for item in bundle.artifacts],
        "feature_scaler": bundle.feature_scaler.asdict(),
        "target_scaler": bundle.target_scaler.asdict(),
        "training_history": list(bundle.training_history),
        "diagnostics": dict(diagnostics),
        "mixture_ablation_gate": dict(ablation) if ablation is not None else None,
        "combined_quality_gate_passed": quality_passed,
        "base": {
            "metrics": dict(base.metrics),
            "monthly": list(base.monthly),
            "daily_equity": list(base.daily_equity),
            "trades": [item.asdict() for item in base.trades],
            "trade_outcomes": list(base.trade_outcomes),
        },
        "stress": {
            "metrics": dict(stress.metrics),
            "monthly": list(stress.monthly),
            "daily_equity": list(stress.daily_equity),
            "trade_outcomes": list(stress.trade_outcomes),
        },
        "economic_gate": economics,
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
    hurdle_dataset = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=progress,
    )
    dataset = build_minute_temporal_dataset(hurdle_dataset)
    del panel, premium, funding, hurdle_dataset
    gap_count = int(
        (
            dataset.timestamps_ms[-1] - dataset.timestamps_ms[0]
        )
        // (5 * 60_000)
        - (dataset.timestamps_ms.size - 1)
    )
    progress(
        "round48_dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "feature_bytes_view": int(dataset.features.nbytes),
            "target_bytes": int(dataset.signed_target_bps.nbytes),
            "dataset_sha256": dataset.dataset_sha256,
            "source_gap_net_count": gap_count,
            "role_timestamp_counts": {
                role: int(mask.sum()) for role, mask in dataset.role_masks.items()
            },
        },
    )
    bundles, preflight = train_minute_mixture_candidates(
        dataset,
        model_dir=evidence_root / "models",
        prediction_dir=evidence_root / "predictions",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    diagnostics = {
        candidate_id: candidate_diagnostics(dataset, bundle)
        for candidate_id, bundle in bundles.items()
    }
    ablation = mixture_ablation_gate(
        diagnostics["single_logistic_tcn"],
        diagnostics["state_mixture_logistic_tcn"],
    )
    candidate_results = []
    for candidate_id, bundle in bundles.items():
        candidate_results.append(
            _candidate_result(
                dataset=dataset,
                bundle=bundle,
                diagnostics=diagnostics[candidate_id],
                ablation=(
                    ablation
                    if candidate_id == "state_mixture_logistic_tcn"
                    else None
                ),
            )
        )
    all_gates = [
        bool(result["combined_quality_gate_passed"])
        and bool(result["economic_gate"]["passed"])
        for result in candidate_results
    ]
    ai_eligible = any(all_gates)
    external_artifacts = []
    for bundle in bundles.values():
        for artifact in bundle.artifacts:
            for kind, path_string, digest, size in (
                (
                    "model",
                    artifact.model_path,
                    artifact.model_sha256,
                    artifact.model_bytes,
                ),
                (
                    "predictions",
                    artifact.prediction_path,
                    artifact.prediction_sha256,
                    artifact.prediction_bytes,
                ),
            ):
                external_artifacts.append(
                    {
                        "kind": kind,
                        "candidate_id": artifact.candidate_id,
                        "seed": artifact.seed,
                        "path": path_string,
                        "bytes": size,
                        "sha256": digest,
                    }
                )
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": (
            "deterministic_candidate_mechanical_gates_passed_ai_ablation_required"
            if ai_eligible
            else "quality_or_economic_gate_rejected"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "report_canonical_sha256": "PENDING",
        "dataset": {
            "dataset_sha256": dataset.dataset_sha256,
            "feature_stream_sha256": dataset.feature_stream_sha256,
            "target_stream_sha256": dataset.target_stream_sha256,
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "symbols": list(SYMBOLS),
            "horizons_minutes": list(HORIZONS_MINUTES),
            "feature_count": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
            "first_timestamp_ms": int(dataset.timestamps_ms[0]),
            "last_timestamp_ms": int(dataset.timestamps_ms[-1]),
            "role_timestamp_counts": {
                role: int(mask.sum()) for role, mask in dataset.role_masks.items()
            },
            "source_evidence": dict(dataset.source_evidence),
            "persistent_feature_copy_created": False,
        },
        "backend": preflight,
        "candidate_results": candidate_results,
        "mixture_ablation_gate": ablation,
        "ai_decision": {
            "paired_veto_only_ablation_eligible": ai_eligible,
            "executed": False,
            "reason": (
                "A deterministic candidate cleared all preregistered mechanical gates; a separately hash-bound paired AI veto ablation is required before any AI uplift statement."
                if ai_eligible
                else "No deterministic candidate cleared distribution, action-quality, mixture, and economic gates. AI cannot repair or obscure a failed numerical model."
            ),
            "ai_uplift_claim": False,
            "language_model_order_authority": False,
        },
        "runtime_evidence": {
            "elapsed_seconds": time.perf_counter() - started,
            "memory": _memory_evidence(),
            "package_versions": _package_versions(),
        },
        "external_artifacts": external_artifacts,
        "selection_confirmation_accessed": False,
        "terminal_2026_accessed": False,
        "leverage_applied": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "trading_authority": False,
        "promotion_permitted": False,
    }
    canonical = dict(report)
    canonical.pop("report_canonical_sha256")
    report["report_canonical_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(
        evidence_root / "report.json", report, indent=2, sort_keys=True
    )
    progress(
        "round48_complete",
        {
            "status": report["status"],
            "report_canonical_sha256": report["report_canonical_sha256"],
            "report_file_sha256": _file_sha256(evidence_root / "report.json"),
            "ai_ablation_eligible": ai_eligible,
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend",
        choices=("directml", "cpu"),
        default="directml",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    run(_parser().parse_args(arguments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
