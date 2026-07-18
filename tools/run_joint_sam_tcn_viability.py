"""Run the hash-bound Round 45 joint cross-asset TCN and SAM screen."""

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
    DILATIONS,
    HIDDEN_CHANNELS,
    HORIZONS,
    KERNEL_SIZE,
    QUANTILES,
    RECEPTIVE_FIELD,
    ROLES,
    STRESS_ONE_WAY_COST_BPS,
    build_distributional_dataset,
    role_mask,
)
from simple_ai_trading.compute import SUPPORTED_COMPUTE_BACKENDS  # noqa: E402
from simple_ai_trading.joint_distributional_tcn_model import (  # noqa: E402
    BATCH_SIZE,
    BOOTSTRAP_BLOCK_HOURS,
    BOOTSTRAP_SAMPLES,
    CANDIDATES,
    FAMILYWISE_LOWER_QUANTILE,
    SAM_RHO,
    SEEDS,
    JointForecastBundle,
    joint_preflight,
    joint_economic_gate,
    joint_forecast_diagnostics,
    optimizer_ablation_gate,
    replay_consensus_trades,
    select_consensus_trades,
    train_joint_candidate,
)
from simple_ai_trading.stateful_turnover_model import (  # noqa: E402
    build_stateful_hourly_dataset,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 45
DESIGN_SCHEMA = "joint-sam-distributional-tcn-design-v1"
BINDING_SCHEMA = "round-045-joint-sam-tcn-execution-binding-v1"
REPORT_SCHEMA = "joint-sam-distributional-tcn-report-v1"
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
PREDECESSOR_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
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
        raise ValueError("Round 45 Git binding command failed") from exc


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
        raise ValueError(f"Round 45 design section is missing: {key}")
    return value


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 45 design")
    design_sha = _validate_hashed_object(
        design,
        field="design_sha256",
        schema=DESIGN_SCHEMA,
        label="Round 45 design",
    )
    predecessor = _require_mapping(design, "predecessor_evidence")
    data = _require_mapping(design, "data_contract")
    model = _require_mapping(design, "model_contract")
    adamw = _require_mapping(model, "adamw")
    sam = _require_mapping(model, "sam")
    windows = _require_mapping(model, "training_windows")
    policy = _require_mapping(design, "descriptive_policy_contract")
    economics = _require_mapping(design, "economic_gate")
    governance = _require_mapping(design, "governance")
    ai_contract = _require_mapping(design, "ai_contract")
    candidates = model.get("candidates")
    if (
        predecessor.get("round_44_dataset_sha256")
        != PREDECESSOR_DATASET_SHA256
        or data.get("source_certificate_canonical_sha256")
        != SOURCE_CANONICAL_SHA256
        or data.get("predecessor_dataset_sha256") != PREDECESSOR_DATASET_SHA256
        or data.get("symbols") != list(SYMBOLS)
        or data.get("causal_features_per_symbol") != 71
        or data.get("joint_input_channels") != 213
        or data.get("target_horizons_hours") != list(HORIZONS)
        or data.get("target_quantiles") != list(QUANTILES)
        or not isinstance(candidates, list)
        or [item.get("id") for item in candidates] != list(CANDIDATES)
        or model.get("hidden_channels") != HIDDEN_CHANNELS
        or model.get("kernel_size") != KERNEL_SIZE
        or model.get("residual_block_dilations") != list(DILATIONS)
        or model.get("receptive_field_hours") != RECEPTIVE_FIELD
        or model.get("seeds") != list(SEEDS)
        or adamw.get("learning_rate") != 1e-3
        or adamw.get("weight_decay") != 1e-4
        or sam.get("rho") != SAM_RHO
        or windows.get("total_hours") != 384
        or windows.get("batch_size") != BATCH_SIZE
        or policy.get("base_one_way_transition_cost_bps")
        != BASE_ONE_WAY_COST_BPS
        or policy.get("stress_one_way_transition_cost_bps")
        != STRESS_ONE_WAY_COST_BPS
        or economics.get("bootstrap_replicates") != BOOTSTRAP_SAMPLES
        or economics.get("familywise_circular_block_bootstrap_hours")
        != BOOTSTRAP_BLOCK_HOURS
        or economics.get("one_sided_familywise_lower_quantile")
        != FAMILYWISE_LOWER_QUANTILE
        or economics.get("minimum_closed_trades") != 180
    ):
        raise ValueError("Round 45 implementation and frozen design differ")
    for field in (
        "selection_confirmation_2025_h2_access_permitted",
        "terminal_2026_access_permitted",
        "testnet_or_live_execution_permitted",
        "promotion_permitted",
        "leverage_permitted",
        "fee_or_slippage_reduction_permitted",
        "risk_gate_relaxation_permitted",
        "post_outcome_parameter_seed_or_candidate_selection_permitted",
        "manual_graph_or_result_editing_permitted",
        "profitability_ai_or_sam_uplift_claim_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 45 governance must deny {field}")
    for field in (
        "fin_r1_round_44_benchmark_passed",
        "language_model_architecture_votes_binding",
        "market_features_or_future_outcomes_supplied_to_language_models",
        "language_model_numerical_forecasts_or_orders_permitted",
        "ai_trade_ablation_permitted",
    ):
        if ai_contract.get(field) is not False:
            raise ValueError(f"Round 45 AI contract must deny {field}")
    return design, design_sha


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 45 execution binding")
    binding_sha = _validate_hashed_object(
        binding,
        field="binding_sha256",
        schema=BINDING_SCHEMA,
        label="Round 45 execution binding",
    )
    source = binding.get("source_certificate")
    if (
        binding.get("design_sha256") != design_sha256
        or not isinstance(source, Mapping)
        or source.get("canonical_sha256") != SOURCE_CANONICAL_SHA256
        or source.get("file_sha256") != _file_sha256(source_certificate_path)
    ):
        raise ValueError("Round 45 binding inputs differ")
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40 or _git("status", "--porcelain"):
        raise ValueError("Round 45 requires a bound commit and clean worktree")
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
        raise ValueError("Round 45 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 45 binding has no blobs")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 45 binding blob is invalid")
        relative_path = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{implementation_commit}:{relative_path}") != expected
            or _git("rev-parse", f"HEAD:{relative_path}") != expected
        ):
            raise ValueError(f"Round 45 bound blob changed: {relative_path}")
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
            raise RuntimeError("Round 45 progress stream is already frozen")
        self.sequence += 1
        payload = {
            "schema_version": "round-045-progress-v1",
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
        raise ValueError(f"Round 45 CSV has no rows: {path.name}")
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


def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)


def _write_derived_cache(root: Path, dataset: object) -> tuple[Path, ...]:
    cache = root / "derived_dataset"
    paths = (
        cache / "timestamps_ms.npy",
        cache / "features.npy",
        cache / "hourly_return_bps.npy",
        cache / "forward_return_bps.npy",
        cache / "metadata.json",
    )
    _write_npy(paths[0], dataset.timestamps_ms)
    _write_npy(paths[1], dataset.features)
    _write_npy(paths[2], dataset.hourly_return_bps)
    _write_npy(paths[3], dataset.forward_return_bps)
    write_json_atomic(
        paths[4],
        {
            "schema_version": "round-045-derived-dataset-cache-v1",
            "dataset_sha256": dataset.dataset_sha256,
            "symbols": list(SYMBOLS),
            "feature_names": list(dataset.feature_names),
            "target_horizons_hours": list(HORIZONS),
            "dtypes": {
                "timestamps_ms": str(dataset.timestamps_ms.dtype),
                "features": str(dataset.features.dtype),
                "hourly_return_bps": str(dataset.hourly_return_bps.dtype),
                "forward_return_bps": str(dataset.forward_return_bps.dtype),
            },
            "shapes": {
                "timestamps_ms": list(dataset.timestamps_ms.shape),
                "features": list(dataset.features.shape),
                "hourly_return_bps": list(dataset.hourly_return_bps.shape),
                "forward_return_bps": list(dataset.forward_return_bps.shape),
            },
        },
        indent=2,
        sort_keys=True,
    )
    return paths


def _write_predictions(root: Path, bundle: JointForecastBundle) -> tuple[Path, Path]:
    prediction_root = root / "predictions"
    seed_path = prediction_root / f"{bundle.candidate_id}_seed_predictions_bps.npy"
    ensemble_path = (
        prediction_root / f"{bundle.candidate_id}_ensemble_predictions_bps.npy"
    )
    _write_npy(seed_path, bundle.seed_predictions_bps)
    _write_npy(ensemble_path, bundle.ensemble_predictions_bps)
    return seed_path, ensemble_path


def _write_hourly_ledger(path: Path, replays: Sequence[object]) -> int:
    fieldnames = [
        "candidate_id",
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
                    "candidate_id": replay.metrics["candidate_id"],
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


def _trade_rows(
    dataset: object,
    trades: Sequence[object],
) -> Iterable[dict[str, object]]:
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
    if dataset.dataset_sha256 != PREDECESSOR_DATASET_SHA256:
        raise ValueError("Round 45 derived dataset differs from Round 44")
    roles = _role_summaries(dataset)
    cache_paths = _write_derived_cache(evidence_root, dataset)
    progress(
        "round45_dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "features_per_symbol": len(dataset.feature_names),
            "dataset_sha256": dataset.dataset_sha256,
            "derived_cache_bytes": sum(path.stat().st_size for path in cache_paths),
        },
    )
    del panel, funding
    device, preflight = joint_preflight(arguments.compute_backend)
    bundles: dict[str, JointForecastBundle] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    monthly_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    prediction_paths: list[Path] = []
    trade_map: dict[str, tuple[object, ...]] = {}
    replays: list[object] = []
    economic_gates: dict[str, dict[str, object]] = {}
    for candidate_id in CANDIDATES:
        bundle = train_joint_candidate(
            dataset,
            candidate_id=candidate_id,
            model_dir=evidence_root / "models",
            device=device,
            preflight=preflight,
            progress=progress,
        )
        bundles[candidate_id] = bundle
        prediction_paths.extend(_write_predictions(evidence_root, bundle))
        candidate_monthly, candidate_stability, candidate_diagnostics = (
            joint_forecast_diagnostics(dataset, bundle)
        )
        monthly_rows.extend(candidate_monthly)
        stability_rows.extend(candidate_stability)
        diagnostics[candidate_id] = candidate_diagnostics
        trades = select_consensus_trades(dataset, bundle)
        trade_map[candidate_id] = trades
        base = replay_consensus_trades(
            dataset,
            trades,
            candidate_id=candidate_id,
            scenario="base",
            one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
        )
        stress = replay_consensus_trades(
            dataset,
            trades,
            candidate_id=candidate_id,
            scenario="stress",
            one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
        )
        if not (
            tuple(item.trade_id for item in base.trades)
            == tuple(item.trade_id for item in stress.trades)
            and np.array_equal(base.positions, stress.positions)
        ):
            raise RuntimeError("Round 45 stress replay changed the base ledger")
        replays.extend((base, stress))
        gate = joint_economic_gate(
            forecast_gate_passed=bool(candidate_diagnostics["gate"]["passed"]),
            stress=stress,
        )
        economic_gates[candidate_id] = gate
        progress(
            "round45_candidate",
            {
                "status": "complete",
                "candidate_id": candidate_id,
                "forecast_gate_passed": candidate_diagnostics["gate"]["passed"],
                "economic_gate_passed": gate["passed"],
                "trades": len(trades),
                "base_total_net_return_fraction": base.metrics[
                    "total_net_return_fraction"
                ],
                "stress_total_net_return_fraction": stress.metrics[
                    "total_net_return_fraction"
                ],
            },
        )
    optimizer_gate = optimizer_ablation_gate(
        bundles["joint_adamw"],
        diagnostics["joint_adamw"],
        bundles["joint_sam"],
        diagnostics["joint_sam"],
    )

    diagnostics_path = evidence_root / "forecast_diagnostics.csv"
    horizon_path = evidence_root / "horizon_summary.csv"
    symbol_horizon_path = evidence_root / "symbol_horizon_summary.csv"
    stability_path = evidence_root / "seed_stability.csv"
    model_path = evidence_root / "models.csv"
    role_path = evidence_root / "roles.csv"
    trade_path = evidence_root / "trades.csv"
    replay_path = evidence_root / "replays.csv"
    monthly_path = evidence_root / "monthly_economics.csv"
    symbol_path = evidence_root / "symbol_economics.csv"
    ledger_path = evidence_root / "hourly_ledger.csv.gz"
    scaler_path = evidence_root / "scalers.json"
    _write_csv(diagnostics_path, monthly_rows)
    _write_csv(
        horizon_path,
        (
            row
            for candidate_id in CANDIDATES
            for row in diagnostics[candidate_id]["horizons"]
        ),
    )
    _write_csv(
        symbol_horizon_path,
        (
            row
            for candidate_id in CANDIDATES
            for row in diagnostics[candidate_id]["symbol_horizons"]
        ),
    )
    _write_csv(stability_path, stability_rows)
    _write_csv(
        model_path,
        (
            artifact.asdict()
            for candidate_id in CANDIDATES
            for artifact in bundles[candidate_id].artifacts
        ),
    )
    _write_csv(role_path, roles)
    _write_csv(
        trade_path,
        (
            row
            for candidate_id in CANDIDATES
            for row in _trade_rows(dataset, trade_map[candidate_id])
        ),
        fieldnames=(
            "trade_id",
            "candidate_id",
            "symbol",
            "symbol_index",
            "decision_index",
            "decision_time_ms",
            "side",
            "horizon_hours",
            "worst_seed_median_bps",
            "expected_after_cost_bps",
            "decision_time_utc",
            "base_realized_net_bps",
            "stress_realized_net_bps",
        ),
    )
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
            {
                "candidate_id": replay.metrics["candidate_id"],
                "scenario": replay.scenario,
                **row,
            }
            for replay in replays
            for row in replay.metrics["monthly"]
        ),
    )
    _write_csv(
        symbol_path,
        (
            {
                "candidate_id": replay.metrics["candidate_id"],
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
            "schema_version": "round-045-training-scalers-v1",
            "feature_names": list(dataset.feature_names),
            "feature_scaler": bundles[CANDIDATES[0]].feature_scaler.asdict(),
            "target_scaler": bundles[CANDIDATES[0]].target_scaler.asdict(),
            "identical_across_candidates": (
                bundles[CANDIDATES[0]].feature_scaler.asdict()
                == bundles[CANDIDATES[1]].feature_scaler.asdict()
                and bundles[CANDIDATES[0]].target_scaler.asdict()
                == bundles[CANDIDATES[1]].target_scaler.asdict()
            ),
        },
        indent=2,
        sort_keys=True,
    )
    progress.freeze(
        {
            "candidate_forecast_gate_pass_count": sum(
                bool(diagnostics[candidate_id]["gate"]["passed"])
                for candidate_id in CANDIDATES
            ),
            "candidate_economic_gate_pass_count": sum(
                bool(economic_gates[candidate_id]["passed"])
                for candidate_id in CANDIDATES
            ),
            "optimizer_ablation_gate_passed": optimizer_gate["passed"],
            "ledger_rows": ledger_rows,
        }
    )
    model_paths = [
        Path(artifact.path)
        for candidate_id in CANDIDATES
        for artifact in bundles[candidate_id].artifacts
    ]
    output_paths = (
        diagnostics_path,
        horizon_path,
        symbol_horizon_path,
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
        *cache_paths,
        *prediction_paths,
        *model_paths,
    )
    replay_index = {
        (str(replay.metrics["candidate_id"]), replay.scenario): replay
        for replay in replays
    }
    source_evidence = source_dataset.source_evidence.asdict()
    candidate_reports: list[dict[str, object]] = []
    for candidate_id in CANDIDATES:
        base = replay_index[(candidate_id, "base")]
        stress = replay_index[(candidate_id, "stress")]
        candidate_reports.append(
            {
                "candidate_id": candidate_id,
                "models": [
                    artifact.asdict()
                    for artifact in bundles[candidate_id].artifacts
                ],
                "forecast_diagnostics": diagnostics[candidate_id],
                "trade_count": len(trade_map[candidate_id]),
                "trade_ledger_sha256": _canonical_sha256(
                    [item.asdict() for item in trade_map[candidate_id]]
                ),
                "fixed_ledger_under_stress": True,
                "base": dict(base.metrics),
                "stress": dict(stress.metrics),
                "economic_gate": economic_gates[candidate_id],
                "prediction_files": _artifact_manifest(
                    [
                        path
                        for path in prediction_paths
                        if path.name.startswith(candidate_id)
                    ]
                ),
            }
        )
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
            "source_hourly_dataset_sha256": source_dataset.dataset_sha256,
            "round44_equivalent_dataset_sha256": PREDECESSOR_DATASET_SHA256,
            "symbols": list(SYMBOLS),
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count_per_symbol": len(dataset.feature_names),
            "joint_input_channels": len(dataset.feature_names) * len(SYMBOLS),
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
            "derived_cache": _artifact_manifest(cache_paths),
        },
        "compute": {
            "requested_backend": arguments.compute_backend,
            "backend_kind": preflight["backend_kind"],
            "backend_device": preflight["backend_device"],
            "preflight": preflight,
            "torch_version": _version("torch"),
            "torch_directml_version": _version("torch-directml"),
            "numpy_version": _version("numpy"),
            "scipy_version": _version("scipy"),
            "model_artifacts": len(model_paths),
            "all_artifacts_exact_reload": all(
                artifact.reload_max_abs_prediction_error <= 1e-6
                for candidate_id in CANDIDATES
                for artifact in bundles[candidate_id].artifacts
            ),
        },
        "candidates": candidate_reports,
        "optimizer_ablation_gate": optimizer_gate,
        "ai_evidence": _require_mapping(design, "ai_contract"),
        "outputs": _artifact_manifest(output_paths),
        "hourly_ledger_rows": ledger_rows,
        "progress_event_count": progress.sequence,
        "claims": {
            "candidate_forecast_gate_pass_count": sum(
                bool(diagnostics[candidate_id]["gate"]["passed"])
                for candidate_id in CANDIDATES
            ),
            "candidate_economic_gate_pass_count": sum(
                bool(economic_gates[candidate_id]["passed"])
                for candidate_id in CANDIDATES
            ),
            "optimizer_ablation_gate_passed": bool(optimizer_gate["passed"]),
            "profitability_established": False,
            "ai_improvement_established": False,
            "sam_improvement_established": False,
            "selection_confirmation_established": False,
            "promotion_authorized": False,
            "testnet_or_live_trading_authorized": False,
            "leverage_authorized": False,
            "reason": "Every Round 45 role is consumed development evidence. No result can authorize trading or a profitability, AI, or SAM improvement claim.",
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
        default=research / "round-045-joint-sam-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-045-joint-sam-tcn-binding.json",
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
