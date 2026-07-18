"""Run the hash-bound Round 44 distributional TCN viability experiment."""

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

import numpy as np


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
from simple_ai_trading.distributional_tcn_model import (  # noqa: E402
    BASE_ONE_WAY_COST_BPS,
    BOOTSTRAP_BLOCK_HOURS,
    BOOTSTRAP_SAMPLES,
    DILATIONS,
    FAMILYWISE_LOWER_QUANTILE,
    HIDDEN_CHANNELS,
    HORIZONS,
    KERNEL_SIZE,
    QUANTILES,
    RECEPTIVE_FIELD,
    ROLES,
    SEEDS,
    STRESS_ONE_WAY_COST_BPS,
    WINDOW_HOURS,
    build_distributional_dataset,
    economic_gate,
    forecast_diagnostics,
    replay_planned_trades,
    role_mask,
    select_planned_trades,
    train_distributional_tcn_ensemble,
)
from simple_ai_trading.compute import SUPPORTED_COMPUTE_BACKENDS  # noqa: E402
from simple_ai_trading.stateful_turnover_model import (  # noqa: E402
    build_stateful_hourly_dataset,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 44
DESIGN_SCHEMA = "distributional-tcn-viability-design-v1"
BINDING_SCHEMA = "round-044-distributional-tcn-execution-binding-v1"
REPORT_SCHEMA = "distributional-tcn-viability-report-v1"
SOURCE_CANONICAL_SHA256 = (
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
        raise ValueError("Round 44 Git binding command failed") from exc


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


def _require_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 44 design section is missing: {key}")
    return value


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 44 design")
    design_sha = _validate_hashed_object(
        design,
        field="design_sha256",
        schema=DESIGN_SCHEMA,
        label="Round 44 design",
    )
    data = _require_mapping(design, "data_contract")
    model = _require_mapping(design, "model_contract")
    windows = _require_mapping(model, "training_windows")
    policy = _require_mapping(design, "descriptive_policy_contract")
    economics = _require_mapping(design, "economic_gate")
    governance = _require_mapping(design, "governance")
    ai_contract = _require_mapping(design, "ai_contract")
    if (
        data.get("source_certificate_canonical_sha256")
        != SOURCE_CANONICAL_SHA256
        or data.get("symbols") != list(SYMBOLS)
        or data.get("causal_feature_count") != 71
        or data.get("target_horizons_hours") != list(HORIZONS)
        or data.get("target_quantiles") != list(QUANTILES)
        or model.get("input_channels") != 71
        or model.get("hidden_channels") != HIDDEN_CHANNELS
        or model.get("kernel_size") != KERNEL_SIZE
        or model.get("residual_block_dilations") != list(DILATIONS)
        or model.get("receptive_field_hours") != RECEPTIVE_FIELD
        or model.get("seeds") != list(SEEDS)
        or windows.get("total_hours") != WINDOW_HOURS
        or policy.get("base_one_way_transition_cost_bps")
        != BASE_ONE_WAY_COST_BPS
        or policy.get("stress_one_way_transition_cost_bps")
        != STRESS_ONE_WAY_COST_BPS
        or economics.get("bootstrap_replicates") != BOOTSTRAP_SAMPLES
        or economics.get("familywise_circular_block_bootstrap_hours")
        != BOOTSTRAP_BLOCK_HOURS
        or economics.get("one_sided_familywise_lower_quantile")
        != FAMILYWISE_LOWER_QUANTILE
    ):
        raise ValueError("Round 44 implementation and frozen design differ")
    for field in (
        "selection_confirmation_2025_h2_access_permitted",
        "terminal_2026_access_permitted",
        "testnet_or_live_execution_permitted",
        "promotion_permitted",
        "leverage_permitted",
        "fee_or_slippage_reduction_permitted",
        "risk_gate_relaxation_permitted",
        "post_outcome_parameter_or_seed_selection_permitted",
        "manual_graph_or_result_editing_permitted",
        "profitability_or_ai_uplift_claim_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 44 governance must deny {field}")
    for field in (
        "market_features_or_future_outcomes_supplied_to_language_models",
        "language_model_numerical_forecasts_or_orders_permitted",
        "ai_trade_ablation_permitted",
    ):
        if ai_contract.get(field) is not False:
            raise ValueError(f"Round 44 AI contract must deny {field}")
    return design, design_sha


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 44 execution binding")
    binding_sha = _validate_hashed_object(
        binding,
        field="binding_sha256",
        schema=BINDING_SCHEMA,
        label="Round 44 execution binding",
    )
    source = binding.get("source_certificate")
    if (
        binding.get("design_sha256") != design_sha256
        or not isinstance(source, Mapping)
        or source.get("canonical_sha256") != SOURCE_CANONICAL_SHA256
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 44 binding inputs differ")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40 or _git("status", "--porcelain"):
        raise ValueError("Round 44 requires a bound commit and clean worktree")
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
        raise ValueError("Round 44 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 44 binding has no blobs")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 44 binding blob is invalid")
        relative_path = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{implementation_commit}:{relative_path}") != expected
            or _git("rev-parse", f"HEAD:{relative_path}") != expected
        ):
            raise ValueError(f"Round 44 bound blob changed: {relative_path}")
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
    def __init__(self, root: Path) -> None:
        self.status_path = root / "status.json"
        self.events_path = root / "progress_events.jsonl"
        self.started = time.perf_counter()
        self.sequence = 0
        self.frozen = False

    def __call__(self, phase: str, detail: Mapping[str, object]) -> None:
        if self.frozen:
            raise RuntimeError("Round 44 progress stream is already frozen")
        self.sequence += 1
        payload = {
            "schema_version": "round-044-progress-v1",
            "round": ROUND,
            "sequence": self.sequence,
            "phase": phase,
            "detail": dict(detail),
            "elapsed_seconds": time.perf_counter() - self.started,
            "memory": _memory_evidence(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        encoded = _canonical_json(payload)
        print(encoded, flush=True)
        with self.events_path.open("a", encoding="ascii", newline="\n") as stream:
            stream.write(encoded + "\n")
        write_json_atomic(self.status_path, payload, indent=2, sort_keys=True)

    def freeze(self, detail: Mapping[str, object]) -> None:
        self("finalization", {"status": "complete", **detail})
        self.frozen = True


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    if not materialized and fieldnames is None:
        raise ValueError(f"Round 44 CSV has no rows: {path.name}")
    resolved_fields = list(fieldnames or ())
    for row in materialized:
        for field in row:
            if field not in resolved_fields:
                resolved_fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(materialized)


def _iso_timestamp(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000.0, UTC).isoformat()


def _write_hourly_ledger(path: Path, replays: Sequence[object]) -> int:
    fieldnames = [
        "scenario",
        "one_way_cost_bps",
        "decision_time_utc",
        "portfolio_return_bps",
    ]
    for symbol in SYMBOLS:
        fieldnames.extend((f"{symbol}_position", f"{symbol}_return_bps"))
    rows = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=9) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for replay in replays:
            for index, timestamp in enumerate(replay.timestamps_ms):
                row: dict[str, object] = {
                    "scenario": replay.scenario,
                    "one_way_cost_bps": replay.one_way_cost_bps,
                    "decision_time_utc": _iso_timestamp(int(timestamp)),
                    "portfolio_return_bps": float(replay.portfolio_return_bps[index]),
                }
                for symbol_index, symbol in enumerate(SYMBOLS):
                    row[f"{symbol}_position"] = int(
                        replay.positions[index, symbol_index]
                    )
                    row[f"{symbol}_return_bps"] = float(
                        replay.symbol_return_bps[index, symbol_index]
                    )
                writer.writerow(row)
                rows += 1
    return rows


def _trade_rows(dataset: object, trades: Sequence[object]) -> Iterable[dict[str, object]]:
    for trade in trades:
        raw = (
            trade.side
            * dataset.hourly_return_bps[
                trade.decision_index : trade.decision_index + trade.horizon_hours,
                trade.symbol_index,
            ].astype(float)
        )
        yield {
            **trade.asdict(),
            "decision_time_utc": _iso_timestamp(trade.decision_time_ms),
            "base_realized_net_bps": float(
                raw.sum() - 2.0 * BASE_ONE_WAY_COST_BPS
            ),
            "stress_realized_net_bps": float(
                raw.sum() - 2.0 * STRESS_ONE_WAY_COST_BPS
            ),
        }


def _artifact_manifest(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "name": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in paths
    ]


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _role_summaries(dataset: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ROLES:
        mask = role_mask(dataset, role.name)
        selected = dataset.forward_return_bps[mask]
        indexes = dataset.timestamps_ms[mask]
        rows.append(
            {
                "role": role.name,
                "start": role.start,
                "end": role.end,
                "timestamps": int(mask.sum()),
                "symbol_rows": int(mask.sum() * len(SYMBOLS)),
                "target_values": int(selected.size),
                "finite_target_values": int(np.count_nonzero(np.isfinite(selected))),
                "first_timestamp_utc": _iso_timestamp(int(indexes[0])),
                "last_timestamp_utc": _iso_timestamp(int(indexes[-1])),
            }
        )
    return rows


def run(arguments: argparse.Namespace) -> dict[str, object]:
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    progress = ProgressWriter(evidence_root)
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
    del premium
    source_dataset = build_stateful_hourly_dataset(
        panel,
        funding,
        derivatives_source,
        progress=progress,
    )
    dataset = build_distributional_dataset(source_dataset)
    roles = _role_summaries(dataset)
    progress(
        "round44_dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "features": len(dataset.feature_names),
            "dataset_sha256": dataset.dataset_sha256,
        },
    )
    del panel, funding
    bundle, preflight = train_distributional_tcn_ensemble(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    monthly_diagnostics, seed_stability, diagnostic_summary = forecast_diagnostics(
        dataset, bundle
    )
    forecast_gate = diagnostic_summary["gate"]
    trades = select_planned_trades(dataset, bundle.ensemble_predictions_bps)
    base = replay_planned_trades(
        dataset,
        trades,
        scenario="base",
        one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
    )
    stress = replay_planned_trades(
        dataset,
        trades,
        scenario="stress",
        one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
    )
    if not (
        tuple(item.trade_id for item in base.trades)
        == tuple(item.trade_id for item in stress.trades)
        and np.array_equal(base.positions, stress.positions)
    ):
        raise RuntimeError("Round 44 stress replay changed the frozen base ledger")
    economics = economic_gate(
        forecast_gate_passed=bool(forecast_gate["passed"]),
        stress=stress,
    )
    progress(
        "round44_gates",
        {
            "status": "complete",
            "forecast_gate_passed": forecast_gate["passed"],
            "economic_gate_passed": economics["passed"],
            "trades": len(trades),
            "base_total_net_return_fraction": base.metrics[
                "total_net_return_fraction"
            ],
            "stress_total_net_return_fraction": stress.metrics[
                "total_net_return_fraction"
            ],
        },
    )

    forecast_path = evidence_root / "forecast_diagnostics.csv"
    horizon_path = evidence_root / "horizon_summary.csv"
    stability_path = evidence_root / "seed_stability.csv"
    model_path = evidence_root / "models.csv"
    role_path = evidence_root / "roles.csv"
    trade_path = evidence_root / "trades.csv"
    replay_path = evidence_root / "replays.csv"
    monthly_path = evidence_root / "monthly_economics.csv"
    symbol_path = evidence_root / "symbol_economics.csv"
    ledger_path = evidence_root / "hourly_ledger.csv.gz"
    scaler_path = evidence_root / "scalers.json"
    _write_csv(forecast_path, monthly_diagnostics)
    _write_csv(horizon_path, diagnostic_summary["horizons"])
    _write_csv(stability_path, seed_stability)
    _write_csv(model_path, (item.asdict() for item in bundle.artifacts))
    _write_csv(role_path, roles)
    _write_csv(
        trade_path,
        _trade_rows(dataset, trades),
        fieldnames=(
            "trade_id",
            "symbol",
            "symbol_index",
            "decision_index",
            "decision_time_ms",
            "side",
            "horizon_hours",
            "selected_lower_quartile_bps",
            "expected_after_cost_bps",
            "decision_time_utc",
            "base_realized_net_bps",
            "stress_realized_net_bps",
        ),
    )
    replays = (base, stress)
    _write_csv(
        replay_path,
        (
            {
                **{
                    key: value
                    for key, value in replay.metrics.items()
                    if key
                    not in {
                        "monthly",
                        "trades_by_symbol",
                        "symbol_net_bps",
                        "bootstrap_mean_hourly_portfolio_bps",
                    }
                },
                "trades_by_symbol": _canonical_json(
                    replay.metrics["trades_by_symbol"]
                ),
                "symbol_net_bps": _canonical_json(replay.metrics["symbol_net_bps"]),
                "bootstrap_mean_hourly_portfolio_bps": _canonical_json(
                    replay.metrics["bootstrap_mean_hourly_portfolio_bps"]
                ),
            }
            for replay in replays
        ),
    )
    _write_csv(
        monthly_path,
        (
            {"scenario": replay.scenario, **row}
            for replay in replays
            for row in replay.metrics["monthly"]
        ),
    )
    _write_csv(
        symbol_path,
        (
            {
                "scenario": replay.scenario,
                "symbol": symbol,
                "trades": replay.metrics["trades_by_symbol"][symbol],
                "net_bps": replay.metrics["symbol_net_bps"][symbol],
            }
            for replay in replays
            for symbol in SYMBOLS
        ),
    )
    ledger_rows = _write_hourly_ledger(ledger_path, replays)
    write_json_atomic(
        scaler_path,
        {
            "schema_version": "round-044-training-scalers-v1",
            "feature_names": list(dataset.feature_names),
            "feature_scaler": bundle.feature_scaler.asdict(),
            "target_scaler": bundle.target_scaler.asdict(),
        },
        indent=2,
        sort_keys=True,
    )
    progress.freeze(
        {
            "forecast_gate_passed": forecast_gate["passed"],
            "economic_gate_passed": economics["passed"],
            "ledger_rows": ledger_rows,
        }
    )
    output_paths = (
        forecast_path,
        horizon_path,
        stability_path,
        model_path,
        role_path,
        trade_path,
        replay_path,
        monthly_path,
        symbol_path,
        ledger_path,
        scaler_path,
        progress.events_path,
        progress.status_path,
        *(Path(item.path) for item in bundle.artifacts),
    )
    source_evidence = source_dataset.source_evidence.asdict()
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": implementation_commit,
        "design": design,
        "binding": binding,
        "source_evidence": source_evidence,
        "dataset": {
            "dataset_sha256": dataset.dataset_sha256,
            "predecessor_dataset_sha256": source_dataset.dataset_sha256,
            "symbols": list(SYMBOLS),
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
            "target_horizons_hours": list(HORIZONS),
            "target_quantiles": list(QUANTILES),
            "first_timestamp_utc": _iso_timestamp(int(dataset.timestamps_ms[0])),
            "last_timestamp_utc": _iso_timestamp(int(dataset.timestamps_ms[-1])),
            "roles": roles,
            "matrix_bytes": int(
                dataset.features.nbytes
                + dataset.hourly_return_bps.nbytes
                + dataset.forward_return_bps.nbytes
            ),
        },
        "compute": {
            "requested_backend": arguments.compute_backend,
            "backend_kind": bundle.backend_kind,
            "backend_device": bundle.backend_device,
            "preflight": preflight,
            "torch_version": _version("torch"),
            "torch_directml_version": _version("torch-directml"),
            "numpy_version": _version("numpy"),
            "scipy_version": _version("scipy"),
            "model_artifacts": len(bundle.artifacts),
            "all_artifacts_exact_reload": all(
                item.reload_max_abs_prediction_error <= 1e-6
                for item in bundle.artifacts
            ),
        },
        "models": [item.asdict() for item in bundle.artifacts],
        "forecast_diagnostics": diagnostic_summary,
        "seed_stability": seed_stability,
        "descriptive_policy": {
            "trade_count": len(trades),
            "trade_ledger_sha256": _canonical_sha256(
                [item.asdict() for item in trades]
            ),
            "fixed_ledger_under_stress": True,
            "base": dict(base.metrics),
            "stress": dict(stress.metrics),
        },
        "economic_gate": economics,
        "ai_evidence": _require_mapping(design, "ai_contract"),
        "outputs": _artifact_manifest(output_paths),
        "hourly_ledger_rows": ledger_rows,
        "progress_event_count": progress.sequence,
        "claims": {
            "forecast_gate_passed": bool(forecast_gate["passed"]),
            "economic_gate_passed": bool(economics["passed"]),
            "profitability_established": False,
            "ai_improvement_established": False,
            "selection_confirmation_established": False,
            "promotion_authorized": False,
            "testnet_or_live_trading_authorized": False,
            "leverage_authorized": False,
            "reason": "All Round 44 roles are consumed development evidence. Passing cannot authorize trading or profitability claims.",
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "memory": _memory_evidence(),
        },
    }
    canonical = dict(report)
    report["report_canonical_sha256"] = _canonical_sha256(canonical)
    write_json_atomic(evidence_root / "report.json", report, indent=2, sort_keys=True)
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-044-distributional-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-044-distributional-tcn-binding.json",
    )
    parser.add_argument(
        "--database", type=Path, default=ROOT / "data/market_data.sqlite"
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_COMPUTE_BACKENDS, default="auto"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
