"""Run the hash-bound Round 46 stability-regularized TCN screen."""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import UTC, datetime
import gzip
import hashlib
from importlib import metadata
import json
import math
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

from simple_ai_trading.cross_asset_cost_data import SYMBOLS  # noqa: E402
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
    DistributionalDataset,
    compounded_forward_returns,
    role_mask,
)
from simple_ai_trading.joint_distributional_tcn_model import (  # noqa: E402
    BOOTSTRAP_BLOCK_HOURS,
    BOOTSTRAP_SAMPLES,
    FAMILYWISE_LOWER_QUANTILE,
    joint_economic_gate,
    joint_forecast_diagnostics,
    replay_consensus_trades,
    select_consensus_trades,
)
from simple_ai_trading.stable_distributional_tcn_model import (  # noqa: E402
    BATCH_SIZE,
    CANDIDATES,
    CONSISTENCY_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    EMA_DECAY,
    MAXIMUM_EPOCHS,
    MAXIMUM_MEDIAN_EARLY_STOP_PINBALL,
    MINIMUM_IMPROVEMENT,
    MINIMUM_STABILITY,
    MINIMUM_STABILITY_IMPROVEMENT,
    PREDECESSOR_MEDIAN_EARLY_STOP_PINBALL,
    PREDECESSOR_MINIMUM_STABILITY,
    SEEDS,
    WAVEBOUND_EPSILON,
    WAVEBOUND_WARMUP_UPDATES,
    StabilityForecastBundle,
    stability_mechanism_gate,
    train_stability_candidates,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 46
DESIGN_SCHEMA = "stability-regularized-distributional-tcn-design-v1"
BINDING_SCHEMA = "round-046-stability-tcn-execution-binding-v1"
REPORT_SCHEMA = "stability-regularized-distributional-tcn-report-v1"
PREDECESSOR_REPORT_SCHEMA = "joint-sam-distributional-tcn-report-v1"
SOURCE_SCHEMA = "round-038-derivatives-source-certificate-v1"
SOURCE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
PREDECESSOR_REPORT_CANONICAL_SHA256 = (
    "4f3e7f80e2ba2fb08f7523b02a7548b9b4ff3a34e49121b048d4c6484a09375d"
)
PREDECESSOR_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
)
CACHE_METADATA_SHA256 = (
    "033480cd3b5669a060f297e7e477c2543a551602834914803bfd1127608d1135"
)
CACHE_FILES = (
    "timestamps_ms.npy",
    "features.npy",
    "hourly_return_bps.npy",
    "forward_return_bps.npy",
    "metadata.json",
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
        raise ValueError("Round 46 Git binding command failed") from exc


def _validate_hashed_object(
    value: Mapping[str, object],
    *,
    field: str,
    schema: str,
    round_number: int,
    label: str,
) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field, ""))
    if (
        value.get("schema_version") != schema
        or value.get("round") != round_number
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError(f"{label} identity is invalid")
    return claimed


def _require_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 46 design section is missing: {key}")
    return value


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 46 design")
    design_sha = _validate_hashed_object(
        design,
        field="design_sha256",
        schema=DESIGN_SCHEMA,
        round_number=ROUND,
        label="Round 46 design",
    )
    predecessor = _require_mapping(design, "predecessor_evidence")
    data = _require_mapping(design, "data_contract")
    model = _require_mapping(design, "model_contract")
    adamw = _require_mapping(model, "adamw")
    windows = _require_mapping(model, "training_windows")
    mechanism = _require_mapping(design, "mechanism_gate")
    policy = _require_mapping(design, "descriptive_policy_contract")
    economics = _require_mapping(design, "economic_gate")
    governance = _require_mapping(design, "governance")
    ai_contract = _require_mapping(design, "ai_contract")
    candidates = model.get("candidates")
    if (
        predecessor.get("round_45_report_canonical_sha256")
        != PREDECESSOR_REPORT_CANONICAL_SHA256
        or predecessor.get("dataset_sha256") != PREDECESSOR_DATASET_SHA256
        or data.get("source_certificate_canonical_sha256") != SOURCE_CANONICAL_SHA256
        or data.get("predecessor_dataset_sha256") != PREDECESSOR_DATASET_SHA256
        or data.get("round_45_derived_cache_metadata_sha256") != CACHE_METADATA_SHA256
        or data.get("symbols") != list(SYMBOLS)
        or data.get("causal_features_per_symbol") != 71
        or data.get("target_horizons_hours") != list(HORIZONS)
        or data.get("target_quantiles") != list(QUANTILES)
        or not isinstance(candidates, list)
        or not all(isinstance(item, Mapping) for item in candidates)
        or [item.get("id") for item in candidates] != list(CANDIDATES)
        or model.get("hidden_channels") != HIDDEN_CHANNELS
        or model.get("kernel_size") != KERNEL_SIZE
        or model.get("residual_block_dilations") != list(DILATIONS)
        or model.get("receptive_field_hours") != RECEPTIVE_FIELD
        or model.get("seeds") != list(SEEDS)
        or adamw.get("learning_rate") != 1e-3
        or adamw.get("weight_decay") != 1e-4
        or windows.get("total_hours") != 384
        or windows.get("batch_size") != BATCH_SIZE
        or model.get("maximum_epochs") != MAXIMUM_EPOCHS
        or model.get("early_stopping_patience_epochs") != EARLY_STOPPING_PATIENCE
        or model.get("minimum_improvement") != MINIMUM_IMPROVEMENT
        or mechanism.get("predecessor_minimum_seed_stability")
        != PREDECESSOR_MINIMUM_STABILITY
        or mechanism.get("minimum_required_seed_stability") != MINIMUM_STABILITY
        or mechanism.get("minimum_absolute_seed_stability_improvement")
        != MINIMUM_STABILITY_IMPROVEMENT
        or mechanism.get("predecessor_median_best_early_stop_pinball")
        != PREDECESSOR_MEDIAN_EARLY_STOP_PINBALL
        or mechanism.get("maximum_permitted_median_best_early_stop_pinball")
        != MAXIMUM_MEDIAN_EARLY_STOP_PINBALL
        or policy.get("base_one_way_transition_cost_bps") != BASE_ONE_WAY_COST_BPS
        or policy.get("stress_one_way_transition_cost_bps") != STRESS_ONE_WAY_COST_BPS
        or policy.get("leverage") != 1.0
        or policy.get("funding_cash_flows_included") is not True
        or economics.get("bootstrap_replicates") != BOOTSTRAP_SAMPLES
        or economics.get("familywise_circular_block_bootstrap_hours")
        != BOOTSTRAP_BLOCK_HOURS
        or economics.get("one_sided_familywise_lower_quantile")
        != FAMILYWISE_LOWER_QUANTILE
        or economics.get("minimum_closed_trades") != 180
    ):
        raise ValueError("Round 46 implementation and frozen design differ")
    candidate_by_id = {
        str(item.get("id")): item for item in candidates if isinstance(item, Mapping)
    }
    wavebound = candidate_by_id.get("wavebound_ema")
    mutual = candidate_by_id.get("mutual_median_consistency")
    if (
        not isinstance(wavebound, Mapping)
        or wavebound.get("ema_decay") != EMA_DECAY
        or wavebound.get("warmup_optimizer_updates") != WAVEBOUND_WARMUP_UPDATES
        or wavebound.get("epsilon_normalized_pinball") != WAVEBOUND_EPSILON
        or not isinstance(mutual, Mapping)
        or mutual.get("consistency_weight") != CONSISTENCY_WEIGHT
        or mutual.get("tail_quantile_consistency_penalty") is not False
    ):
        raise ValueError("Round 46 stability mechanism differs from the design")
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
        "profitability_ai_or_stability_uplift_claim_permitted",
    ):
        if governance.get(field) is not False:
            raise ValueError(f"Round 46 governance must deny {field}")
    if (
        ai_contract.get("risk_review_benchmark_passed") is not True
        or ai_contract.get("benchmark_is_financial_edge_evidence") is not False
        or ai_contract.get(
            "market_features_or_future_outcomes_supplied_to_language_models"
        )
        is not False
        or ai_contract.get("language_model_numerical_forecasts_or_orders_permitted")
        is not False
        or ai_contract.get("ai_trade_ablation_permitted") is not False
    ):
        raise ValueError("Round 46 AI safety boundary differs from the design")
    return design, design_sha


def _validate_external_input(
    path: Path,
    binding_entry: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not path.is_file() or binding_entry.get("file_sha256") != _file_sha256(path):
        raise ValueError(f"{label} differs from the execution binding")


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    source_certificate_path: Path,
    predecessor_report_path: Path,
) -> tuple[dict[str, object], str, str]:
    binding = _read_object(path, "Round 46 execution binding")
    binding_sha = _validate_hashed_object(
        binding,
        field="binding_sha256",
        schema=BINDING_SCHEMA,
        round_number=ROUND,
        label="Round 46 execution binding",
    )
    source = binding.get("source_certificate")
    predecessor = binding.get("predecessor_report")
    if (
        binding.get("design_sha256") != design_sha256
        or not isinstance(source, Mapping)
        or source.get("canonical_sha256") != SOURCE_CANONICAL_SHA256
        or not isinstance(predecessor, Mapping)
        or predecessor.get("canonical_sha256") != PREDECESSOR_REPORT_CANONICAL_SHA256
    ):
        raise ValueError("Round 46 binding inputs differ")
    _validate_external_input(
        source_certificate_path, source, label="Round 46 source certificate"
    )
    _validate_external_input(
        predecessor_report_path, predecessor, label="Round 46 predecessor report"
    )
    implementation_commit = str(binding.get("implementation_commit") or "")
    if len(implementation_commit) != 40 or _git("status", "--porcelain"):
        raise ValueError("Round 46 requires a bound commit and clean worktree")
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
        raise ValueError("Round 46 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 46 binding has no blobs")
    for item in blobs:
        if not isinstance(item, Mapping):
            raise ValueError("Round 46 binding blob is invalid")
        relative_path = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            _git("rev-parse", f"{implementation_commit}:{relative_path}") != expected
            or _git("rev-parse", f"HEAD:{relative_path}") != expected
        ):
            raise ValueError(f"Round 46 bound blob changed: {relative_path}")
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
            raise RuntimeError("Round 46 progress stream is already frozen")
        self.sequence += 1
        payload = {
            "schema_version": "round-046-progress-v1",
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


def _dataset_identity(
    feature_names: Sequence[str],
    timestamps_ms: np.ndarray,
    features: np.ndarray,
    hourly_return_bps: np.ndarray,
    forward_return_bps: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    names = tuple(feature_names) + tuple(
        f"forward_return_{value}h_bps" for value in HORIZONS
    )
    for name in names:
        digest.update(str(name).encode("ascii"))
        digest.update(b"\0")
    for values in (
        timestamps_ms,
        features,
        hourly_return_bps,
        forward_return_bps,
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _load_verified_cache(
    cache_root: Path,
    binding: Mapping[str, object],
) -> tuple[DistributionalDataset, list[dict[str, object]], dict[str, object]]:
    bound_manifest = binding.get("derived_cache")
    if not isinstance(bound_manifest, list):
        raise ValueError("Round 46 binding has no cache manifest")
    bound_by_name = {
        str(item.get("name")): item
        for item in bound_manifest
        if isinstance(item, Mapping)
    }
    manifest: list[dict[str, object]] = []
    for name in CACHE_FILES:
        path = cache_root / name
        expected = bound_by_name.get(name)
        if not path.is_file() or expected is None:
            raise ValueError(f"Round 46 cache input is missing: {name}")
        item = {
            "name": name,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if item["bytes"] != expected.get("bytes") or item["sha256"] != expected.get(
            "sha256"
        ):
            raise ValueError(f"Round 46 cache input changed: {name}")
        manifest.append(item)
    metadata_path = cache_root / "metadata.json"
    metadata_value = _read_object(metadata_path, "Round 46 cache metadata")
    if (
        _file_sha256(metadata_path) != CACHE_METADATA_SHA256
        or metadata_value.get("schema_version") != "round-045-derived-dataset-cache-v1"
        or metadata_value.get("dataset_sha256") != PREDECESSOR_DATASET_SHA256
        or metadata_value.get("symbols") != list(SYMBOLS)
        or metadata_value.get("target_horizons_hours") != list(HORIZONS)
    ):
        raise ValueError("Round 46 cache metadata is invalid")
    arrays = {
        name: np.load(cache_root / f"{name}.npy", allow_pickle=False)
        for name in (
            "timestamps_ms",
            "features",
            "hourly_return_bps",
            "forward_return_bps",
        )
    }
    dtypes = metadata_value.get("dtypes")
    shapes = metadata_value.get("shapes")
    if not isinstance(dtypes, Mapping) or not isinstance(shapes, Mapping):
        raise ValueError("Round 46 cache array contract is missing")
    for name, values in arrays.items():
        if str(values.dtype) != dtypes.get(name) or list(values.shape) != shapes.get(
            name
        ):
            raise ValueError(f"Round 46 cache array contract differs: {name}")
    timestamps = arrays["timestamps_ms"]
    features = arrays["features"]
    hourly = arrays["hourly_return_bps"]
    forward = arrays["forward_return_bps"]
    if (
        timestamps.ndim != 1
        or timestamps.size < 2
        or not np.all(np.diff(timestamps) == 3_600_000)
        or features.shape != (timestamps.size, len(SYMBOLS), 71)
        or hourly.shape != (timestamps.size, len(SYMBOLS))
        or forward.shape != (timestamps.size, len(SYMBOLS), len(HORIZONS))
        or not np.isfinite(features).all()
        or not np.isfinite(hourly).all()
    ):
        raise ValueError("Round 46 cache arrays are structurally invalid")
    reproduced_forward = compounded_forward_returns(hourly)
    if not np.array_equal(reproduced_forward, forward, equal_nan=True):
        raise ValueError("Round 46 cached targets do not reproduce from hourly returns")
    feature_names = tuple(str(value) for value in metadata_value["feature_names"])
    identity = _dataset_identity(feature_names, timestamps, features, hourly, forward)
    if identity != PREDECESSOR_DATASET_SHA256:
        raise ValueError("Round 46 cache aggregate dataset identity differs")
    dataset = DistributionalDataset(
        feature_names=feature_names,
        timestamps_ms=timestamps,
        features=features,
        hourly_return_bps=hourly,
        forward_return_bps=forward,
        dataset_sha256=identity,
    )
    verification = {
        "individual_file_hashes_verified": True,
        "metadata_contract_verified": True,
        "hourly_grid_contiguous": True,
        "forward_targets_exactly_reproduced": True,
        "aggregate_dataset_identity_reproduced": True,
        "cache_copied_to_evidence": False,
    }
    return dataset, manifest, verification


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    materialized = list(rows)
    if not materialized and fieldnames is None:
        raise ValueError(f"Round 46 CSV has no rows: {path.name}")
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


def _write_predictions(
    root: Path,
    bundle: StabilityForecastBundle,
) -> tuple[Path, Path]:
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
    dataset: DistributionalDataset,
    trades: Sequence[object],
) -> Iterable[dict[str, object]]:
    for trade in trades:
        raw = trade.side * dataset.hourly_return_bps[
            trade.decision_index : trade.decision_index + trade.horizon_hours,
            trade.symbol_index,
        ].astype(float)
        yield {
            **trade.asdict(),
            "decision_time_utc": _iso_timestamp(trade.decision_time_ms),
            "base_realized_net_bps": float(raw.sum() - 2.0 * BASE_ONE_WAY_COST_BPS),
            "stress_realized_net_bps": float(raw.sum() - 2.0 * STRESS_ONE_WAY_COST_BPS),
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


def _role_summaries(dataset: DistributionalDataset) -> list[dict[str, object]]:
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


def _economic_gate(
    *,
    forecast_gate_passed: bool,
    mechanism_gate_passed: bool,
    stress: object,
) -> dict[str, object]:
    gate = joint_economic_gate(
        forecast_gate_passed=forecast_gate_passed and mechanism_gate_passed,
        stress=stress,
    )
    reasons = list(gate["reasons"])
    if mechanism_gate_passed is False:
        reasons.append("mechanism_gate_failed")
    if forecast_gate_passed and not mechanism_gate_passed:
        reasons = [item for item in reasons if item != "forecast_gate_failed"]
    return {**gate, "passed": not reasons, "reasons": list(dict.fromkeys(reasons))}


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
        predecessor_report_path=arguments.predecessor_report.resolve(),
    )
    source = _read_object(
        arguments.source_certificate.resolve(), "Round 46 source certificate"
    )
    source_sha = _validate_hashed_object(
        source,
        field="source_certificate_sha256",
        schema=SOURCE_SCHEMA,
        round_number=38,
        label="Round 46 source certificate",
    )
    predecessor = _read_object(
        arguments.predecessor_report.resolve(), "Round 46 predecessor report"
    )
    predecessor_sha = _validate_hashed_object(
        predecessor,
        field="report_canonical_sha256",
        schema=PREDECESSOR_REPORT_SCHEMA,
        round_number=45,
        label="Round 46 predecessor report",
    )
    progress(
        "binding",
        {
            "status": "complete",
            "design_sha256": design_sha,
            "binding_sha256": binding_sha,
            "implementation_commit": implementation_commit,
            "source_certificate_canonical_sha256": source_sha,
            "predecessor_report_canonical_sha256": predecessor_sha,
        },
    )
    dataset, cache_manifest, cache_verification = _load_verified_cache(
        arguments.derived_cache.resolve(), binding
    )
    roles = _role_summaries(dataset)
    progress(
        "round46_dataset",
        {
            "status": "complete",
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "features_per_symbol": len(dataset.feature_names),
            "dataset_sha256": dataset.dataset_sha256,
            "cache_bytes_read_once": sum(int(item["bytes"]) for item in cache_manifest),
            "cache_copied": False,
        },
    )
    bundles, preflight = train_stability_candidates(
        dataset,
        model_dir=evidence_root / "models",
        compute_backend=arguments.compute_backend,
        progress=progress,
    )
    diagnostics: dict[str, dict[str, object]] = {}
    mechanism_gates: dict[str, dict[str, object]] = {}
    economic_gates: dict[str, dict[str, object]] = {}
    monthly_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    prediction_paths: list[Path] = []
    trade_map: dict[str, tuple[object, ...]] = {}
    replays: list[object] = []
    for candidate_index, candidate_id in enumerate(CANDIDATES):
        bundle = bundles[candidate_id]
        prediction_paths.extend(_write_predictions(evidence_root, bundle))
        candidate_monthly, candidate_stability, candidate_diagnostics = (
            joint_forecast_diagnostics(dataset, bundle)  # type: ignore[arg-type]
        )
        monthly_rows.extend(candidate_monthly)
        stability_rows.extend(candidate_stability)
        diagnostics[candidate_id] = candidate_diagnostics
        mechanism = stability_mechanism_gate(bundle, candidate_diagnostics)
        mechanism_gates[candidate_id] = mechanism
        trades = select_consensus_trades(dataset, bundle)  # type: ignore[arg-type]
        trade_map[candidate_id] = trades
        base = replay_consensus_trades(
            dataset,
            trades,
            candidate_id=candidate_id,
            scenario="base",
            one_way_cost_bps=BASE_ONE_WAY_COST_BPS,
            bootstrap_seed=SEEDS[0] + candidate_index * 1_000,
        )
        stress = replay_consensus_trades(
            dataset,
            trades,
            candidate_id=candidate_id,
            scenario="stress",
            one_way_cost_bps=STRESS_ONE_WAY_COST_BPS,
            bootstrap_seed=SEEDS[0] + candidate_index * 1_000 + 100,
        )
        if not (
            tuple(item.trade_id for item in base.trades)
            == tuple(item.trade_id for item in stress.trades)
            and np.array_equal(base.positions, stress.positions)
        ):
            raise RuntimeError("Round 46 stress replay changed the base ledger")
        replays.extend((base, stress))
        economic = _economic_gate(
            forecast_gate_passed=bool(candidate_diagnostics["gate"]["passed"]),
            mechanism_gate_passed=bool(mechanism["passed"]),
            stress=stress,
        )
        economic_gates[candidate_id] = economic
        progress(
            "round46_candidate",
            {
                "status": "complete",
                "candidate_id": candidate_id,
                "forecast_gate_passed": candidate_diagnostics["gate"]["passed"],
                "mechanism_gate_passed": mechanism["passed"],
                "economic_gate_passed": economic["passed"],
                "trades": len(trades),
                "base_total_net_return_fraction": base.metrics[
                    "total_net_return_fraction"
                ],
                "stress_total_net_return_fraction": stress.metrics[
                    "total_net_return_fraction"
                ],
            },
        )

    diagnostics_path = evidence_root / "forecast_diagnostics.csv"
    horizon_path = evidence_root / "horizon_summary.csv"
    symbol_horizon_path = evidence_root / "symbol_horizon_summary.csv"
    stability_path = evidence_root / "seed_stability.csv"
    mechanism_path = evidence_root / "mechanism_gates.json"
    training_path = evidence_root / "training_history.json"
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
    write_json_atomic(mechanism_path, mechanism_gates, indent=2, sort_keys=True)
    write_json_atomic(
        training_path,
        {
            "schema_version": "round-046-training-history-v1",
            "round": ROUND,
            "candidates": {
                candidate_id: list(bundles[candidate_id].training_history)
                for candidate_id in CANDIDATES
            },
        },
        indent=2,
        sort_keys=True,
    )
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
                "trades_by_symbol": _canonical_json(replay.metrics["trades_by_symbol"]),
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
            "schema_version": "round-046-training-scalers-v1",
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
            "candidate_mechanism_gate_pass_count": sum(
                bool(mechanism_gates[candidate_id]["passed"])
                for candidate_id in CANDIDATES
            ),
            "candidate_economic_gate_pass_count": sum(
                bool(economic_gates[candidate_id]["passed"])
                for candidate_id in CANDIDATES
            ),
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
        mechanism_path,
        training_path,
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
        *prediction_paths,
        *model_paths,
    )
    replay_index = {
        (str(replay.metrics["candidate_id"]), replay.scenario): replay
        for replay in replays
    }
    candidate_reports: list[dict[str, object]] = []
    for candidate_id in CANDIDATES:
        base = replay_index[(candidate_id, "base")]
        stress = replay_index[(candidate_id, "stress")]
        candidate_reports.append(
            {
                "candidate_id": candidate_id,
                "models": [
                    artifact.asdict() for artifact in bundles[candidate_id].artifacts
                ],
                "training_history_file": next(
                    item
                    for item in _artifact_manifest((training_path,))
                    if item["name"] == training_path.name
                ),
                "forecast_diagnostics": diagnostics[candidate_id],
                "mechanism_gate": mechanism_gates[candidate_id],
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
        "source_lineage": {
            "source_certificate_canonical_sha256": source_sha,
            "source_certificate_file_sha256": _file_sha256(
                arguments.source_certificate.resolve()
            ),
            "predecessor_report_canonical_sha256": predecessor_sha,
            "predecessor_report_file_sha256": _file_sha256(
                arguments.predecessor_report.resolve()
            ),
        },
        "dataset": {
            "dataset_sha256": dataset.dataset_sha256,
            "round44_and_round45_equivalent_dataset_sha256": (
                PREDECESSOR_DATASET_SHA256
            ),
            "symbols": list(SYMBOLS),
            "timestamps": dataset.timestamps,
            "rows": dataset.rows,
            "feature_count_per_symbol": len(dataset.feature_names),
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
            "derived_cache_inputs": cache_manifest,
            "cache_verification": cache_verification,
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
        "ai_evidence": _require_mapping(design, "ai_contract"),
        "outputs": _artifact_manifest(output_paths),
        "hourly_ledger_rows": ledger_rows,
        "progress_event_count": progress.sequence,
        "claims": {
            "candidate_forecast_gate_pass_count": sum(
                bool(diagnostics[candidate_id]["gate"]["passed"])
                for candidate_id in CANDIDATES
            ),
            "candidate_mechanism_gate_pass_count": sum(
                bool(mechanism_gates[candidate_id]["passed"])
                for candidate_id in CANDIDATES
            ),
            "candidate_economic_gate_pass_count": sum(
                bool(economic_gates[candidate_id]["passed"])
                for candidate_id in CANDIDATES
            ),
            "profitability_established": False,
            "ai_improvement_established": False,
            "stability_uplift_established": False,
            "selection_confirmation_established": False,
            "promotion_authorized": False,
            "testnet_or_live_trading_authorized": False,
            "leverage_authorized": False,
            "reason": "Every Round 46 role is consumed development evidence. No result can authorize trading or a profitability or AI-edge claim.",
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "memory": _memory_evidence(),
        },
    }
    if not all(
        math.isfinite(float(candidate["base"]["total_net_return_fraction"]))
        and math.isfinite(float(candidate["stress"]["total_net_return_fraction"]))
        for candidate in candidate_reports
    ):
        raise RuntimeError("Round 46 report contains nonfinite economics")
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
        default=research / "round-046-stability-regularized-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-046-stability-regularized-tcn-binding.json",
    )
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--predecessor-report", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend", choices=("directml", "cpu"), default="directml"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run(arguments)
    print(
        _canonical_json(
            {
                "evidence_root": str(arguments.evidence_root.resolve()),
                "report_canonical_sha256": report["report_canonical_sha256"],
                "status": report["status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
