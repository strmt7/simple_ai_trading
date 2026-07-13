"""Run the hash-bound Round 43 stateful turnover and AI-factor ablation."""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import UTC, datetime
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence


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
    load_derivatives_state,
)
from simple_ai_trading.stateful_turnover_model import (  # noqa: E402
    AI_FACTOR_NAMES,
    BASE_ONE_WAY_COST_BPS,
    BOOTSTRAP_BLOCK_HOURS,
    BOOTSTRAP_SAMPLES,
    COST_FILTER_LAMBDA,
    FAMILYWISE_LOWER_QUANTILE,
    FEATURE_SETS,
    MAXIMUM_HOLDING_HOURS,
    MODES,
    SCHEDULES,
    SEED,
    STRESS_ONE_WAY_COST_BPS,
    ReplayResult,
    build_stateful_hourly_dataset,
    evaluate_ai_uplift,
    evaluate_stateful_gate,
    replay_always_long,
    replay_independent_hourly,
    replay_stateful_policy,
    train_stateful_forecasts,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 43
DESIGN_SCHEMA = "stateful-turnover-ai-factor-ablation-design-v2"
AUDIT_SCHEMA = "round-043-ai-factor-audit-v1"
BINDING_SCHEMA = "round-043-stateful-turnover-ai-factor-execution-binding-v2"
REPORT_SCHEMA = "stateful-turnover-ai-factor-ablation-report-v2"
REASON_LABELS = {
    0: "none",
    1: "entry",
    2: "exit",
    3: "reversal",
    4: "maximum_hold_exit",
}


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
        raise ValueError("Round 43 Git binding command failed") from exc


def _validate_hashed_object(
    value: Mapping[str, object],
    *,
    field: str,
    schema: str,
    label: str,
) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field, ""))
    if (
        value.get("schema_version") != schema
        or value.get("round") != ROUND
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError(f"{label} identity is invalid")
    return claimed


def _validate_design_and_audit(
    design_path: Path,
    audit_path: Path,
) -> tuple[dict[str, object], str, dict[str, object], str]:
    design = _read_object(design_path, "Round 43 design")
    design_sha = _validate_hashed_object(
        design,
        field="design_sha256",
        schema=DESIGN_SCHEMA,
        label="Round 43 design",
    )
    audit = _read_object(audit_path, "Round 43 AI factor audit")
    audit_sha = _validate_hashed_object(
        audit,
        field="audit_sha256",
        schema=AUDIT_SCHEMA,
        label="Round 43 AI factor audit",
    )
    governance = design.get("governance")
    feature = design.get("feature_ablation")
    model = design.get("model_contract")
    position = design.get("position_and_cost_contract")
    evaluation = design.get("evaluation_contract")
    audit_governance = audit.get("governance")
    admitted = audit.get("admitted_factor_programs")
    if not all(
        isinstance(item, Mapping)
        for item in (
            governance,
            feature,
            model,
            position,
            evaluation,
            audit_governance,
        )
    ) or not isinstance(admitted, list):
        raise ValueError("Round 43 design or audit sections are incomplete")
    denied_fields = (
        "selection_confirmation_2025_h2_or_terminal_2026_access_permitted",
        "fee_or_slippage_reduction_permitted",
        "maker_execution_assumption_permitted",
        "risk_gate_relaxation_permitted",
        "ai_text_or_direct_action_in_execution_loop_permitted",
        "ai_factor_rewrite_after_outcome_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "leverage_permitted",
    )
    for field in denied_fields:
        if governance.get(field) is not False:
            raise ValueError(f"Round 43 governance must deny {field}")
    for field in (
        "market_features_or_targets_supplied_to_models",
        "historical_outcomes_supplied_to_models",
        "model_generated_text_executed_as_code",
        "model_generated_trading_actions_permitted",
    ):
        if audit_governance.get(field) is not False:
            raise ValueError(f"Round 43 AI audit must deny {field}")
    if (
        feature.get("ai_factor_audit_sha256") != audit_sha
        or feature.get("factor_program_count") != len(AI_FACTOR_NAMES)
        or [item.get("name") for item in admitted] != list(AI_FACTOR_NAMES)
        or model.get("feature_sets") != list(FEATURE_SETS)
        or model.get("total_models") != 12
        or position.get("modes") != list(MODES)
        or position.get("cost_filter_lambda") != COST_FILTER_LAMBDA
        or position.get("base_one_way_cost_bps_per_unit_position_change")
        != BASE_ONE_WAY_COST_BPS
        or position.get("stress_one_way_cost_bps_per_unit_position_change")
        != STRESS_ONE_WAY_COST_BPS
        or position.get("maximum_continuous_holding_hours") != MAXIMUM_HOLDING_HOURS
        or evaluation.get("bootstrap_replicates") != BOOTSTRAP_SAMPLES
        or evaluation.get("bootstrap_block_hours") != BOOTSTRAP_BLOCK_HOURS
        or evaluation.get("familywise_one_sided_alpha_for_four_candidates")
        != FAMILYWISE_LOWER_QUANTILE
    ):
        raise ValueError("Round 43 implementation and frozen contracts differ")
    return design, design_sha, audit, audit_sha


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    audit_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 43 execution binding")
    binding_sha = _validate_hashed_object(
        binding,
        field="binding_sha256",
        schema=BINDING_SCHEMA,
        label="Round 43 execution binding",
    )
    source = binding.get("source_certificate")
    if (
        binding.get("design_sha256") != design_sha256
        or binding.get("ai_factor_audit_sha256") != audit_sha256
        or not isinstance(source, Mapping)
        or source.get("canonical_sha256")
        != "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 43 binding inputs differ")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40 or _git("status", "--porcelain"):
        raise ValueError("Round 43 requires a bound commit and clean worktree")
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
        raise ValueError("Round 43 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 43 binding has no blobs")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 43 binding blob is invalid")
        relative_path = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{implementation_commit}:{relative_path}") != expected
            or _git("rev-parse", f"HEAD:{relative_path}") != expected
        ):
            raise ValueError(f"Round 43 bound blob changed: {relative_path}")
    return binding, binding_sha, implementation_commit


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
            "schema_version": "round-043-progress-v1",
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


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Round 43 CSV has no rows: {path.name}")
    fieldnames: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def _write_ledger(path: Path, results: Sequence[ReplayResult]) -> int:
    rows = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=9) as stream:
        fieldnames = [
            "candidate_id",
            "feature_set",
            "mode",
            "cost_scenario",
            "one_way_cost_bps",
            "decision_time_utc",
            "symbol",
            "forecast_bps",
            "signed_pre_transition_utility_bps",
            "position",
            "position_age_hours",
            "transition_units",
            "transition_reason",
            "weighted_symbol_net_bps",
            "portfolio_net_bps",
            "final_boundary_exit",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            final_mask = result.positions[-1] != 0
            for hour, timestamp_ms in enumerate(result.timestamps_ms):
                timestamp = datetime.fromtimestamp(
                    timestamp_ms / 1000.0, UTC
                ).isoformat()
                for symbol_index, symbol in enumerate(SYMBOLS):
                    writer.writerow(
                        {
                            "candidate_id": result.candidate_id,
                            "feature_set": result.feature_set,
                            "mode": result.mode,
                            "cost_scenario": result.cost_scenario,
                            "one_way_cost_bps": result.cost_bps,
                            "decision_time_utc": timestamp,
                            "symbol": symbol,
                            "forecast_bps": result.forecasts_bps[hour, symbol_index],
                            "signed_pre_transition_utility_bps": result.target_bps[
                                hour, symbol_index
                            ],
                            "position": int(result.positions[hour, symbol_index]),
                            "position_age_hours": int(
                                result.position_age_hours[hour, symbol_index]
                            ),
                            "transition_units": result.transition_units[
                                hour, symbol_index
                            ],
                            "transition_reason": REASON_LABELS[
                                int(result.transition_reasons[hour, symbol_index])
                            ],
                            "weighted_symbol_net_bps": result.symbol_net_bps[
                                hour, symbol_index
                            ],
                            "portfolio_net_bps": result.portfolio_return_bps[hour],
                            "final_boundary_exit": bool(
                                hour == result.timestamps_ms.size - 1
                                and final_mask[symbol_index]
                            ),
                        }
                    )
                    rows += 1
    return rows


def _summary_rows(results: Sequence[ReplayResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        metrics = result.metrics
        interval = metrics["bootstrap_mean_hourly_net_bps"]
        if not isinstance(interval, Mapping):
            raise ValueError("Round 43 bootstrap summary is invalid")
        rows.append(
            {
                key: value
                for key, value in metrics.items()
                if key not in {"monthly", "symbols", "bootstrap_mean_hourly_net_bps"}
            }
            | {
                "bootstrap_lower_bps": interval["lower_bps"],
                "bootstrap_median_bps": interval["median_bps"],
                "bootstrap_upper_bps": interval["upper_bps"],
            }
        )
    return rows


def _monthly_rows(results: Sequence[ReplayResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for row in result.metrics["monthly"]:  # type: ignore[union-attr]
            rows.append(
                {
                    "candidate_id": result.candidate_id,
                    "cost_scenario": result.cost_scenario,
                    **row,
                }
            )
    return rows


def _symbol_rows(results: Sequence[ReplayResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        for row in result.metrics["symbols"]:  # type: ignore[union-attr]
            rows.append(
                {
                    "candidate_id": result.candidate_id,
                    "cost_scenario": result.cost_scenario,
                    **row,
                }
            )
    return rows


def _artifact_manifest(paths: Sequence[Path]) -> list[dict[str, object]]:
    return [
        {
            "name": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def run(arguments: argparse.Namespace) -> dict[str, object]:
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root / "status.json")
    started = time.perf_counter()
    design_path = arguments.design.resolve()
    audit_path = arguments.ai_factor_audit.resolve()
    source_path = arguments.source_certificate.resolve()
    design, design_sha, audit, audit_sha = _validate_design_and_audit(
        design_path, audit_path
    )
    binding, binding_sha, implementation_commit = _validate_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        audit_sha256=audit_sha,
        source_certificate_path=source_path,
    )
    progress(
        "binding",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "ai_factor_audit_sha256": audit_sha,
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
        source_certificate_path=source_path,
        progress=progress,
    )
    del premium
    dataset = build_stateful_hourly_dataset(
        panel,
        funding,
        derivatives_source,
        progress=progress,
    )
    progress(
        "dataset",
        {
            "status": "complete",
            "rows": dataset.rows,
            "baseline_features": dataset.baseline_features.shape[1],
            "augmented_features": dataset.augmented_features.shape[1],
            "matrix_bytes": int(
                dataset.baseline_features.nbytes
                + dataset.augmented_features.nbytes
                + dataset.signed_pre_transition_utility_bps.nbytes
            ),
            "dataset_sha256": dataset.dataset_sha256,
        },
    )
    del panel, funding
    forecasts = train_stateful_forecasts(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    stateful_results: list[ReplayResult] = []
    comparator_results: list[ReplayResult] = []
    costs = (
        ("base", BASE_ONE_WAY_COST_BPS),
        ("stress", STRESS_ONE_WAY_COST_BPS),
    )
    for cost_index, (cost_scenario, cost_bps) in enumerate(costs):
        for feature_index, feature_set in enumerate(FEATURE_SETS):
            for mode_index, mode in enumerate(MODES):
                seed = SEED + 1_000 + cost_index * 100 + feature_index * 10 + mode_index
                result = replay_stateful_policy(
                    dataset,
                    forecasts.predictions[feature_set],
                    feature_set=feature_set,
                    mode=mode,
                    cost_scenario=cost_scenario,
                    cost_bps=cost_bps,
                    seed=seed,
                )
                stateful_results.append(result)
                comparator_results.append(
                    replay_independent_hourly(
                        dataset,
                        forecasts.predictions[feature_set],
                        feature_set=feature_set,
                        mode=mode,
                        cost_scenario=cost_scenario,
                        cost_bps=cost_bps,
                        seed=seed + 500,
                    )
                )
                progress(
                    "round43_replay",
                    {
                        "status": "complete",
                        "candidate_id": result.candidate_id,
                        "cost_scenario": cost_scenario,
                        "total_net_return_fraction": result.metrics[
                            "total_net_return_fraction"
                        ],
                        "maximum_drawdown_fraction": result.metrics[
                            "maximum_drawdown_fraction"
                        ],
                        "transition_events": result.metrics["transition_events"],
                    },
                )
        comparator_results.append(
            replay_always_long(
                dataset,
                cost_scenario=cost_scenario,
                cost_bps=cost_bps,
                seed=SEED + 2_000 + cost_index,
            )
        )
    all_reload = all(
        artifact.reload_max_abs_prediction_error <= 1e-12
        for artifact in forecasts.artifacts
    )
    indexed = {
        (result.feature_set, result.mode, result.cost_scenario): result
        for result in stateful_results
    }
    gates: list[dict[str, object]] = []
    for feature_set in FEATURE_SETS:
        for mode in MODES:
            base = indexed[(feature_set, mode, "base")]
            stress = indexed[(feature_set, mode, "stress")]
            gates.append(
                {
                    "candidate_id": base.candidate_id,
                    **evaluate_stateful_gate(
                        base,
                        stress,
                        all_models_exact_reload=all_reload,
                    ),
                }
            )
    ai_uplift = evaluate_ai_uplift(
        indexed[(FEATURE_SETS[0], "long_only", "base")],
        indexed[(FEATURE_SETS[1], "long_only", "base")],
        indexed[(FEATURE_SETS[0], "long_only", "stress")],
        indexed[(FEATURE_SETS[1], "long_only", "stress")],
    )
    diagnostics_path = evidence_root / "forecast_diagnostics.csv"
    model_path = evidence_root / "models.csv"
    replay_path = evidence_root / "replays.csv"
    monthly_path = evidence_root / "monthly.csv"
    symbol_path = evidence_root / "symbols.csv"
    ledger_path = evidence_root / "position_ledger.csv.gz"
    _write_csv(diagnostics_path, forecasts.diagnostics)
    _write_csv(
        model_path,
        (
            artifact.asdict()
            | {
                "top_feature_gain": _canonical_json(
                    artifact.asdict()["top_feature_gain"]
                )
            }
            for artifact in forecasts.artifacts
        ),
    )
    all_results = stateful_results + comparator_results
    _write_csv(replay_path, _summary_rows(all_results))
    _write_csv(monthly_path, _monthly_rows(all_results))
    _write_csv(symbol_path, _symbol_rows(all_results))
    ledger_rows = _write_ledger(ledger_path, all_results)
    output_paths = (
        diagnostics_path,
        model_path,
        replay_path,
        monthly_path,
        symbol_path,
        ledger_path,
    )
    runtime = time.perf_counter() - started
    source_evidence = dataset.source_evidence.asdict()
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": design_sha,
        "ai_factor_audit_sha256": audit_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "design": design,
        "ai_factor_audit": audit,
        "binding": binding,
        "source_evidence": source_evidence,
        "dataset": {
            "rows": dataset.rows,
            "hourly_timestamps": dataset.rows // len(SYMBOLS),
            "symbols": list(SYMBOLS),
            "first_decision_time_ms": int(dataset.decision_time_ms[0]),
            "last_decision_time_ms": int(dataset.decision_time_ms[-1]),
            "baseline_feature_count": dataset.baseline_features.shape[1],
            "augmented_feature_count": dataset.augmented_features.shape[1],
            "ai_factor_names": list(AI_FACTOR_NAMES),
            "dataset_sha256": dataset.dataset_sha256,
            "memory_bytes": int(
                dataset.baseline_features.nbytes
                + dataset.augmented_features.nbytes
                + dataset.signed_pre_transition_utility_bps.nbytes
            ),
        },
        "schedules": [schedule.asdict() for schedule in SCHEDULES],
        "compute": {
            "requested_backend": arguments.compute_backend,
            "backend_kind": forecasts.backend_kind,
            "backend_device": forecasts.backend_device,
            "lightgbm_version": metadata.version("lightgbm"),
            "numpy_version": metadata.version("numpy"),
            "model_artifacts": len(forecasts.artifacts),
            "all_artifacts_exact_reload": all_reload,
        },
        "models": [artifact.asdict() for artifact in forecasts.artifacts],
        "forecast_diagnostics": list(forecasts.diagnostics),
        "stateful_replays": [result.summary() for result in stateful_results],
        "comparators": [result.summary() for result in comparator_results],
        "stateful_viability_gates": gates,
        "ai_uplift_gate": ai_uplift,
        "outputs": _artifact_manifest(output_paths),
        "position_ledger_rows": ledger_rows,
        "claims": {
            "any_stateful_viability_gate_passed": any(
                bool(gate["passed"]) for gate in gates
            ),
            "ai_uplift_gate_passed": bool(ai_uplift["passed"]),
            "profitability_established": False,
            "ai_improvement_established": bool(ai_uplift["passed"]),
            "selection_confirmation_established": False,
            "promotion_authorized": False,
            "testnet_or_live_trading_authorized": False,
            "leverage_authorized": False,
            "reason": "The complete 2025 H1 evaluation window is consumed development evidence. Passing can authorize only a separately frozen confirmation experiment.",
        },
        "runtime": {
            "elapsed_seconds": runtime,
            "memory": _memory_evidence(),
        },
    }
    canonical = dict(report)
    report["report_canonical_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(evidence_root / "report.json", report, indent=2, sort_keys=True)
    progress(
        "report",
        {
            "status": "complete",
            "report_canonical_sha256": report["report_canonical_sha256"],
            "stateful_gate_pass_count": sum(bool(gate["passed"]) for gate in gates),
            "ai_uplift_passed": ai_uplift["passed"],
            "ledger_rows": ledger_rows,
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-043-stateful-turnover-ai-factor-design.json",
    )
    parser.add_argument(
        "--ai-factor-audit",
        type=Path,
        default=research / "round-043-ai-factor-audit.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-043-stateful-turnover-ai-factor-binding.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
