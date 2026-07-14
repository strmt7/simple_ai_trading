"""Run the hash-bound Round 55 bounded alpha and governed AI ablation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import UTC, datetime
import gc
import hashlib
from importlib import metadata as package_metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.ai_factor_programs import (  # noqa: E402
    FactorProgram,
    fit_factor_transform,
    validate_factor_program,
)
from simple_ai_trading.bounded_alpha_lightgbm import (  # noqa: E402
    VIEW_IDS,
    BoundedAlphaSpec,
    block_bootstrap_mean_bps,
    build_trade_plan,
    consensus_decisions,
    load_bounded_alpha_model,
    predict_bounded_alpha_model,
    replay_trade_plan,
    save_bounded_alpha_model,
    train_bounded_alpha_model,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    MINUTE_MS,
    SYMBOLS,
    load_verified_minute_panel_window,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    load_derivatives_state,
)
from simple_ai_trading.stop_time_payoff_data import (  # noqa: E402
    EVENT_NAMES,
    StopTimeSpecification,
    build_stop_time_payoff_dataset,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 55
DESIGN_SCHEMA = "round-055-bounded-alpha-lightgbm-ai-design-v1"
BINDING_SCHEMA = "round-055-bounded-alpha-execution-binding-v1"
AI_REPORT_SCHEMA = "round-055-ai-factor-research-report-v1"
AI_LEDGER_SCHEMA = "round-055-ai-factor-program-ledger-v1"
REPORT_SCHEMA = "round-055-bounded-alpha-report-v1"
EXPECTED_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
)
EXPECTED_SOURCE_CERTIFICATE_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
SEEDS = (5501, 5502, 5503)
TREATMENTS = ("baseline_71", "ai_program_augmented")
BASE_COST_BPS = 12.0
STRESS_COST_BPS = 16.0
TRADE_FIELDS = (
    "treatment",
    "interval",
    "decision_time_utc",
    "exit_time_utc",
    "symbol",
    "side",
    "size_fraction",
    "score_bps",
    "stop_bps",
    "event",
    "stress_net_payoff_bps",
    "stress_initial_capital_return_fraction",
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
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Round 55 Git binding command failed") from exc


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1000)


class ProgressWriter:
    def __init__(self, root: Path) -> None:
        self.path = root / "progress_events.jsonl"
        self.status_path = root / "status.json"
        self.started = time.perf_counter()

    def __call__(self, stage: str, values: Mapping[str, object]) -> None:
        payload = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "stage": stage,
            **dict(values),
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(payload) + "\n")
        write_json_atomic(self.status_path, payload)
        detail = " ".join(f"{key}={value}" for key, value in values.items())
        print(f"round55 {stage} {detail}".rstrip(), flush=True)


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 55 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen_development_only"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 55 design identity is invalid")
    source = design.get("source_contract")
    model = design.get("model_contract")
    payoff = design.get("payoff_contract")
    if not all(isinstance(item, Mapping) for item in (source, model, payoff)):
        raise ValueError("Round 55 design sections are absent")
    if (
        source.get("symbols") != list(SYMBOLS)
        or source.get("feature_dataset_sha256") != EXPECTED_DATASET_SHA256
        or source.get("source_certificate_canonical_sha256")
        != EXPECTED_SOURCE_CERTIFICATE_SHA256
        or model.get("views") is None
        or model.get("seeds") != list(SEEDS)
        or payoff.get("training_round_trip_execution_charge_bps")
        != STRESS_COST_BPS
    ):
        raise ValueError("Round 55 design and implementation constants differ")
    return design, claimed


def _validate_file_manifest(
    row: Mapping[str, object],
    path: Path,
    label: str,
) -> None:
    if (
        int(row.get("bytes", -1)) != path.stat().st_size
        or str(row.get("sha256", "")) != _file_sha256(path)
    ):
        raise ValueError(f"Round 55 {label} file identity differs")


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    derived_cache: Path,
    source_certificate: Path,
    ai_report: Path,
    ai_ledger: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    binding = _read_object(path, "Round 55 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 55 execution binding identity is invalid")
    implementation_commit = str(binding.get("implementation_commit", ""))
    if not implementation_commit or subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise ValueError("Round 55 implementation commit is not an ancestor of HEAD")
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 55 execution binding has no blobs")
    for row in blobs:
        if not isinstance(row, Mapping):
            raise ValueError("Round 55 blob row is invalid")
        source_path = str(row.get("path", ""))
        expected = str(row.get("git_blob_oid", ""))
        if _git("rev-parse", f"{implementation_commit}:{source_path}") != expected:
            raise ValueError(f"Round 55 implementation blob differs: {source_path}")

    external = binding.get("external_evidence")
    if not isinstance(external, Mapping):
        raise ValueError("Round 55 external evidence binding is absent")
    for key, candidate in (
        ("source_certificate", source_certificate),
        ("ai_report", ai_report),
        ("ai_ledger", ai_ledger),
    ):
        row = external.get(key)
        if not isinstance(row, Mapping):
            raise ValueError(f"Round 55 {key} binding is absent")
        _validate_file_manifest(row, candidate, key)

    rows = binding.get("derived_cache")
    if not isinstance(rows, list):
        raise ValueError("Round 55 derived-cache binding is absent")
    paths: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Round 55 derived-cache row is invalid")
        name = str(row.get("name", ""))
        candidate = (derived_cache / name).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Round 55 cache file is absent: {candidate}")
        _validate_file_manifest(row, candidate, f"cache {name}")
        paths[name] = candidate
    required = {
        "features.npy",
        "forward_return_bps.npy",
        "hourly_return_bps.npy",
        "metadata.json",
        "timestamps_ms.npy",
    }
    if set(paths) != required:
        raise ValueError("Round 55 derived-cache file set differs")
    return binding, paths


def _load_development_cache(
    paths: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, object]]:
    metadata = _read_object(paths["metadata.json"], "Round 55 cache metadata")
    feature_names = tuple(str(value) for value in metadata.get("feature_names", []))
    if (
        metadata.get("schema_version") != "round-045-derived-dataset-cache-v1"
        or metadata.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or metadata.get("symbols") != list(SYMBOLS)
        or len(feature_names) != 71
    ):
        raise ValueError("Round 55 cache metadata differs")
    timestamps_source = np.load(paths["timestamps_ms.npy"], mmap_mode="r", allow_pickle=False)
    feature_source = np.load(paths["features.npy"], mmap_mode="r", allow_pickle=False)
    if timestamps_source.shape != (30_647,) or feature_source.shape != (30_647, 3, 71):
        raise ValueError("Round 55 cache array shapes differ")
    forbidden_start = int(np.searchsorted(timestamps_source, _date_ms("2024-10-01")))
    if forbidden_start <= 0 or forbidden_start >= timestamps_source.size:
        raise ValueError("Round 55 forbidden-row boundary is invalid")
    timestamps = np.asarray(timestamps_source[:forbidden_start], dtype=np.int64).copy()
    features = np.asarray(feature_source[:forbidden_start], dtype=np.float32).copy()
    if (
        np.any(np.diff(timestamps) <= 0)
        or timestamps[-1] >= _date_ms("2024-10-01")
        or not np.isfinite(features).all()
    ):
        raise ValueError("Round 55 development cache is invalid")
    return timestamps, features, feature_names, {
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "source_timestamps": int(timestamps_source.size),
        "development_timestamps": int(timestamps.size),
        "first_development_timestamp_ms": int(timestamps[0]),
        "last_development_timestamp_ms": int(timestamps[-1]),
        "forbidden_existing_rows_loaded": False,
        "forbidden_existing_rows": int(timestamps_source.size - forbidden_start),
    }


def _validate_ai_evidence(
    report_path: Path,
    ledger_path: Path,
    feature_names: Sequence[str],
    *,
    design_sha256: str,
    implementation_commit: str,
) -> tuple[tuple[FactorProgram, ...], dict[str, object], dict[str, object]]:
    report = _read_object(report_path, "Round 55 AI report")
    report_canonical = dict(report)
    report_sha = str(report_canonical.pop("report_sha256", ""))
    if (
        report.get("schema_version") != AI_REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "complete"
        or report.get("design_sha256") != design_sha256
        or report.get("implementation_commit") != implementation_commit
        or report.get("market_values_read") is not False
        or report.get("outcomes_read") is not False
        or report_sha != _canonical_sha256(report_canonical)
    ):
        raise ValueError("Round 55 AI report identity is invalid")
    ledger = _read_object(ledger_path, "Round 55 AI factor ledger")
    ledger_canonical = dict(ledger)
    ledger_sha = str(ledger_canonical.pop("ledger_sha256", ""))
    rows = ledger.get("programs")
    if (
        ledger.get("schema_version") != AI_LEDGER_SCHEMA
        or ledger.get("round") != ROUND
        or ledger.get("design_sha256") != design_sha256
        or ledger.get("source_dataset_sha256") != EXPECTED_DATASET_SHA256
        or ledger.get("feature_names") != list(feature_names)
        or ledger.get("outcomes_or_market_values_read") is not False
        or ledger.get("order_authority") is not False
        or ledger_sha != _canonical_sha256(ledger_canonical)
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Round 55 AI factor ledger identity is invalid")
    programs: list[FactorProgram] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Round 55 AI factor program row is invalid")
        input_mapping = {
            key: row[key]
            for key in (
                "name",
                "expression",
                "mechanism",
                "failure_mode",
                "expected_horizon",
            )
        }
        validated = validate_factor_program(
            input_mapping,
            model=str(row.get("model", "")),
            feature_names=feature_names,
        )
        if validated.asdict() != dict(row):
            raise ValueError("Round 55 AI factor program canonical form differs")
        programs.append(validated)
    return tuple(programs), report, ledger


def _role_mask(
    timestamps_ms: np.ndarray,
    start: str,
    end_exclusive: str,
) -> np.ndarray:
    start_ms = _date_ms(start)
    end_ms = _date_ms(end_exclusive)
    exit_time = timestamps_ms + 61 * MINUTE_MS
    return (
        (timestamps_ms >= start_ms)
        & (timestamps_ms < end_ms)
        & (exit_time < end_ms)
    )


def _role_masks(timestamps_ms: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "iteration_training": _role_mask(
            timestamps_ms, "2022-01-01", "2024-01-01"
        ),
        "iteration_selection": _role_mask(
            timestamps_ms, "2024-01-01", "2024-04-01"
        ),
        "final_refit": _role_mask(timestamps_ms, "2022-01-01", "2024-07-01"),
        "policy_development": _role_mask(
            timestamps_ms, "2024-07-01", "2024-10-01"
        ),
    }
    if any(np.count_nonzero(mask) == 0 for mask in masks.values()):
        raise ValueError("Round 55 chronological role is empty")
    return masks


def _interval_masks(timestamps_ms: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "policy_development": _role_mask(
            timestamps_ms, "2024-07-01", "2024-09-01"
        ),
        "development_holdout": _role_mask(
            timestamps_ms, "2024-09-01", "2024-10-01"
        ),
    }


def _payoff_manifest(
    root: Path,
    payoff: object,
    source_evidence: Mapping[str, object],
) -> dict[str, object]:
    arrays = {
        "timestamps_ms": payoff.timestamps_ms,
        "stop_bps": payoff.stop_bps,
        "long_event_code": payoff.long_event_code,
        "short_event_code": payoff.short_event_code,
        "long_event_minute": payoff.long_event_minute,
        "short_event_minute": payoff.short_event_minute,
        "long_exit_time_ms": payoff.long_exit_time_ms,
        "short_exit_time_ms": payoff.short_exit_time_ms,
        "long_net_payoff_bps": payoff.long_net_payoff_bps,
        "short_net_payoff_bps": payoff.short_net_payoff_bps,
        "long_gap_through_slippage_bps": payoff.long_gap_through_slippage_bps,
        "short_gap_through_slippage_bps": payoff.short_gap_through_slippage_bps,
    }
    path = root / "stop_time_payoff_dataset.npz"
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as reloaded:
        if set(reloaded.files) != set(arrays) or any(
            not np.array_equal(reloaded[name], values)
            for name, values in arrays.items()
        ):
            raise RuntimeError("Round 55 payoff cache reload differs")
    manifest: dict[str, object] = {
        "schema_version": "round-055-stop-time-payoff-manifest-v1",
        "dataset_sha256": payoff.dataset_sha256,
        "source_dataset_sha256": payoff.source_dataset_sha256,
        "specification": payoff.specification.asdict(),
        "timestamps": payoff.timestamps,
        "rows": payoff.rows,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "source_evidence": dict(source_evidence),
        "synthetic_rows": 0,
        "forbidden_existing_rows_read": False,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = root / "stop_time_payoff_manifest.json"
    write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_file_sha256"] = _file_sha256(manifest_path)
    return manifest


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _predictive_diagnostics(
    predictions: np.ndarray,
    payoff: object,
    intervals: Mapping[str, np.ndarray],
) -> dict[str, object]:
    truth = np.stack(
        (payoff.long_net_payoff_bps, payoff.short_net_payoff_bps),
        axis=-1,
    ).astype(np.float64)
    roles: dict[str, object] = {}
    for role, mask in intervals.items():
        by_view: dict[str, object] = {}
        for view_index, view_id in enumerate(VIEW_IDS):
            seed_rows = []
            for seed_index, seed in enumerate(SEEDS):
                side_values = {
                    side: _spearman(
                        predictions[view_index, seed_index, mask, :, side_index].reshape(-1),
                        truth[mask, :, side_index].reshape(-1),
                    )
                    for side_index, side in enumerate(("long", "short"))
                }
                seed_rows.append({"seed": seed, "spearman": side_values})
            by_view[view_id] = seed_rows
        roles[role] = by_view
    return roles


def _stability_diagnostics(
    predictions: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    medians = np.median(predictions[:, :, mask], axis=1)
    pairwise: dict[str, float | None] = {}
    values = []
    for left_index, left in enumerate(VIEW_IDS):
        for right_index in range(left_index + 1, len(VIEW_IDS)):
            right = VIEW_IDS[right_index]
            correlation = _spearman(
                medians[left_index].reshape(-1),
                medians[right_index].reshape(-1),
            )
            pairwise[f"{left}__{right}"] = correlation
            if correlation is not None:
                values.append(correlation)
    signs = np.sign(predictions[:, :, mask])
    all_seed_same = np.all(signs == signs[:, :1], axis=1)
    return {
        "pairwise_view_median_spearman": pairwise,
        "minimum_pairwise_view_median_spearman": min(values) if values else None,
        "all_seed_sign_agreement_fraction": float(np.mean(all_seed_same)),
    }


def _gate(
    design: Mapping[str, object],
    *,
    development: Mapping[str, object],
    holdout: Mapping[str, object],
    holdout_bootstrap: Mapping[str, object],
    stability: Mapping[str, object],
    maximum_reload_error: float,
) -> dict[str, object]:
    frozen = design["development_gates"]
    assert isinstance(frozen, Mapping)
    failures: list[str] = []

    def metric(source: Mapping[str, object], name: str, fallback: float) -> float:
        value = source.get(name)
        return fallback if value is None else float(value)

    checks = (
        (
            int(development["closed_trades"])
            >= int(frozen["minimum_development_closed_trades"]),
            "development_closed_trades",
        ),
        (
            int(development["active_days"])
            >= int(frozen["minimum_development_active_days"]),
            "development_active_days",
        ),
        (
            metric(development, "profit_factor", -math.inf)
            >= float(frozen["minimum_development_stress_profit_factor"]),
            "development_profit_factor",
        ),
        (
            float(development["maximum_drawdown_fraction"])
            <= float(frozen["maximum_development_stress_drawdown_fraction"]),
            "development_drawdown",
        ),
        (
            float(development["total_return_fraction"])
            > float(frozen["minimum_development_stress_return_fraction"]),
            "development_return",
        ),
        (
            int(holdout["closed_trades"])
            >= int(frozen["minimum_holdout_closed_trades"]),
            "holdout_closed_trades",
        ),
        (
            int(holdout["active_days"])
            >= int(frozen["minimum_holdout_active_days"]),
            "holdout_active_days",
        ),
        (
            metric(holdout, "profit_factor", -math.inf)
            >= float(frozen["minimum_holdout_stress_profit_factor"]),
            "holdout_profit_factor",
        ),
        (
            float(holdout["maximum_drawdown_fraction"])
            <= float(frozen["maximum_holdout_stress_drawdown_fraction"]),
            "holdout_drawdown",
        ),
        (
            float(holdout["total_return_fraction"])
            > float(frozen["minimum_holdout_stress_return_fraction"]),
            "holdout_return",
        ),
        (
            int(holdout["symbols_with_trades"])
            >= int(frozen["required_symbol_activity"]),
            "holdout_symbol_activity",
        ),
        (
            float(holdout["maximum_single_symbol_fraction_of_absolute_net_pnl"])
            <= float(frozen["maximum_single_symbol_fraction_of_absolute_net_pnl"]),
            "holdout_symbol_concentration",
        ),
        (
            float(stability.get("minimum_pairwise_view_median_spearman") or -math.inf)
            >= float(frozen["minimum_view_median_pairwise_spearman"]),
            "view_stability",
        ),
        (
            float(stability["all_seed_sign_agreement_fraction"])
            >= float(frozen["minimum_seed_sign_agreement_fraction"]),
            "seed_sign_stability",
        ),
        (
            float(holdout_bootstrap["lower_bps"])
            > float(frozen["minimum_168h_block_bootstrap_lower_mean_hourly_bps"]),
            "holdout_familywise_bootstrap",
        ),
        (
            maximum_reload_error
            <= float(frozen["maximum_reload_prediction_error_bps"]),
            "model_reload",
        ),
    )
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return {"passed": not failures, "failures": failures, "checks": len(checks)}


def _paired_ai_gate(
    design: Mapping[str, object],
    baseline_gate: Mapping[str, object],
    ai_gate: Mapping[str, object],
    baseline_holdout: object,
    ai_holdout: object,
) -> dict[str, object]:
    frozen = design["ai_uplift_gate"]
    assert isinstance(frozen, Mapping)
    delta = ai_holdout.hourly_return_fraction - baseline_holdout.hourly_return_fraction
    bootstrap = block_bootstrap_mean_bps(delta, seed=55_092)
    baseline_metrics = baseline_holdout.metrics
    ai_metrics = ai_holdout.metrics
    baseline_drawdown = float(baseline_metrics["maximum_drawdown_fraction"])
    ai_drawdown = float(ai_metrics["maximum_drawdown_fraction"])
    drawdown_ratio = (
        ai_drawdown / baseline_drawdown
        if baseline_drawdown > 0.0
        else (0.0 if ai_drawdown == 0.0 else math.inf)
    )
    baseline_trades = int(baseline_metrics["closed_trades"])
    trade_ratio = (
        int(ai_metrics["closed_trades"]) / baseline_trades
        if baseline_trades
        else 0.0
    )
    return_delta = float(
        ai_metrics["total_return_fraction"]
        - baseline_metrics["total_return_fraction"]
    )
    checks = {
        "baseline_gate": bool(baseline_gate["passed"]),
        "ai_gate": bool(ai_gate["passed"]),
        "positive_return_delta": return_delta
        > float(frozen["minimum_paired_holdout_stress_return_delta_fraction"]),
        "positive_familywise_bootstrap": float(bootstrap["lower_bps"])
        > float(frozen["minimum_familywise_168h_block_bootstrap_lower_paired_delta_bps"]),
        "drawdown_noninferiority": drawdown_ratio
        <= float(frozen["maximum_ai_to_baseline_drawdown_ratio"]),
        "activity_retention": trade_ratio
        >= float(frozen["minimum_ai_to_baseline_closed_trade_ratio"]),
    }
    return {
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
        "checks": checks,
        "holdout_stress_return_delta_fraction": return_delta,
        "drawdown_ratio": drawdown_ratio if math.isfinite(drawdown_ratio) else None,
        "closed_trade_ratio": trade_ratio,
        "paired_hourly_bootstrap": bootstrap,
    }


def _monthly_rows(
    treatment: str,
    interval: str,
    replay: object,
) -> list[dict[str, object]]:
    timestamps = replay.interval_timestamps_ms
    months = np.asarray(
        [datetime.fromtimestamp(value / 1000.0, UTC).strftime("%Y-%m") for value in timestamps]
    )
    rows = []
    for month in sorted(set(months)):
        selected = months == month
        values = replay.hourly_return_fraction[selected]
        equity = 1.0 + np.cumsum(values)
        peak = np.maximum.accumulate(equity)
        drawdown = float(np.max((peak - equity) / np.maximum(peak, 1e-12)))
        rows.append(
            {
                "treatment": treatment,
                "interval": interval,
                "scenario": replay.scenario,
                "month": month,
                "hours": int(np.count_nonzero(selected)),
                "return_fraction": float(np.sum(values)),
                "maximum_drawdown_fraction": drawdown,
            }
        )
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    empty_fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows and not empty_fieldnames:
        raise ValueError(f"Round 55 CSV would be empty: {path.name}")
    fieldnames = list(rows[0]) if rows else list(empty_fieldnames or ())
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"Round 55 CSV fields differ: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _trade_rows(treatment: str, interval: str, plan: object) -> list[dict[str, object]]:
    rows = []
    for index in range(plan.closed_trades):
        rows.append(
            {
                "treatment": treatment,
                "interval": interval,
                "decision_time_utc": datetime.fromtimestamp(
                    int(plan.decision_time_ms[index]) / 1000.0, UTC
                ).isoformat(),
                "exit_time_utc": datetime.fromtimestamp(
                    int(plan.exit_time_ms[index]) / 1000.0, UTC
                ).isoformat(),
                "symbol": SYMBOLS[int(plan.symbol_index[index])],
                "side": "long" if int(plan.side[index]) == 1 else "short",
                "size_fraction": float(plan.size_fraction[index]),
                "score_bps": float(plan.score_bps[index]),
                "stop_bps": float(plan.stop_bps[index]),
                "event": EVENT_NAMES[int(plan.event_code[index])],
                "stress_net_payoff_bps": float(plan.stress_net_payoff_bps[index]),
                "stress_initial_capital_return_fraction": float(
                    plan.size_fraction[index]
                    * plan.stress_net_payoff_bps[index]
                    / 10_000.0
                ),
            }
        )
    return rows


def run(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    evidence_root = arguments.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=False)
    model_root = evidence_root / "models"
    model_root.mkdir()
    progress = ProgressWriter(evidence_root)
    design, design_sha = _validate_design(arguments.design.resolve())
    binding, cache_paths = _validate_binding(
        arguments.binding.resolve(),
        design_sha256=design_sha,
        derived_cache=arguments.derived_cache.resolve(),
        source_certificate=arguments.source_certificate.resolve(),
        ai_report=arguments.ai_report.resolve(),
        ai_ledger=arguments.ai_ledger.resolve(),
    )
    implementation_commit = str(binding["implementation_commit"])
    timestamps, baseline_features, feature_names, cache_evidence = (
        _load_development_cache(cache_paths)
    )
    programs, ai_report, ai_ledger = _validate_ai_evidence(
        arguments.ai_report.resolve(),
        arguments.ai_ledger.resolve(),
        feature_names,
        design_sha256=design_sha,
        implementation_commit=implementation_commit,
    )
    roles = _role_masks(timestamps)
    intervals = _interval_masks(timestamps)
    progress(
        "contracts_validated",
        {
            "design_sha256": design_sha,
            "binding_sha256": binding["binding_sha256"],
            "implementation_commit": implementation_commit,
            "development_timestamps": len(timestamps),
            "syntax_valid_ai_programs": len(programs),
        },
    )

    panel, price_source = load_verified_minute_panel_window(
        arguments.database.resolve(),
        materialization_end="2024-10-01",
        progress=progress,
    )
    premium, funding, derivatives_source = load_derivatives_state(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.source_certificate.resolve(),
        progress=progress,
    )
    del premium
    volatility_index = feature_names.index("target_realized_volatility_60m_bps")
    payoff = build_stop_time_payoff_dataset(
        panel,
        funding,
        timestamps,
        baseline_features[..., volatility_index],
        source_dataset_sha256=EXPECTED_DATASET_SHA256,
        specification=StopTimeSpecification(
            horizon_minutes=60,
            stop_volatility_multiple=1.5,
            minimum_stop_bps=40.0,
            maximum_stop_bps=250.0,
            round_trip_execution_charge_bps=STRESS_COST_BPS,
        ),
    )
    source_evidence = derivatives_source.asdict()
    del panel, funding, price_source, derivatives_source
    gc.collect()
    payoff_evidence = _payoff_manifest(evidence_root, payoff, source_evidence)
    progress(
        "payoff_dataset_built",
        {
            "dataset_sha256": payoff.dataset_sha256,
            "rows": payoff.rows,
            "cache_bytes": payoff_evidence["bytes"],
        },
    )

    flat_baseline = baseline_features.reshape(-1, baseline_features.shape[-1])
    transform, factor_matrix, factor_rejections = fit_factor_transform(
        programs,
        flat_baseline,
        feature_names,
        np.repeat(roles["final_refit"], len(SYMBOLS)),
    )
    augmented_features = np.concatenate(
        (
            baseline_features,
            factor_matrix.reshape(
                baseline_features.shape[0], len(SYMBOLS), factor_matrix.shape[-1]
            ),
        ),
        axis=-1,
    ).astype(np.float32)
    transform_path = evidence_root / "ai_factor_transform.json"
    transform_payload = transform.asdict()
    transform_payload["runtime_rejections"] = list(factor_rejections)
    write_json_atomic(transform_path, transform_payload)
    progress(
        "ai_factor_transform_fitted",
        {
            "accepted_programs": len(transform.programs),
            "runtime_rejections": len(factor_rejections),
            "transform_sha256": transform.transform_sha256,
        },
    )

    treatment_features = {
        TREATMENTS[0]: (baseline_features, feature_names),
        TREATMENTS[1]: (
            augmented_features,
            feature_names + transform.output_feature_names,
        ),
    }
    model_spec = BoundedAlphaSpec()
    treatment_results: dict[str, object] = {}
    prediction_diagnostics: dict[str, object] = {}
    stability_rows: dict[str, object] = {}
    model_manifests: dict[str, object] = {}
    replay_objects: dict[str, dict[str, dict[str, object]]] = {}
    trade_plans: dict[str, dict[str, object]] = {}
    all_trade_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for treatment_id, (features, names) in treatment_features.items():
        predictions = np.empty(
            (
                len(VIEW_IDS),
                len(SEEDS),
                len(timestamps),
                len(SYMBOLS),
                2,
            ),
            dtype=np.float64,
        )
        artifacts = []
        for view_index, view_id in enumerate(VIEW_IDS):
            for seed_index, seed in enumerate(SEEDS):
                model = train_bounded_alpha_model(
                    treatment_id=treatment_id,
                    view_id=view_id,
                    seed=seed,
                    features=features,
                    feature_names=names,
                    timestamps_ms=timestamps,
                    stop_bps=payoff.stop_bps,
                    long_target_bps=payoff.long_net_payoff_bps,
                    short_target_bps=payoff.short_net_payoff_bps,
                    role_masks=roles,
                    long_exit_time_ms=payoff.long_exit_time_ms,
                    short_exit_time_ms=payoff.short_exit_time_ms,
                    source_dataset_sha256=EXPECTED_DATASET_SHA256,
                    payoff_dataset_sha256=payoff.dataset_sha256,
                    spec=model_spec,
                    compute_backend=arguments.compute_backend,
                    progress=progress,
                )
                path = model_root / f"{treatment_id}-{view_id}-seed-{seed}.json"
                save_bounded_alpha_model(path, model)
                reloaded = load_bounded_alpha_model(path)
                predictions[view_index, seed_index] = predict_bounded_alpha_model(
                    reloaded, features, payoff.stop_bps
                )
                artifacts.append(
                    {
                        "treatment_id": treatment_id,
                        "view_id": view_id,
                        "seed": seed,
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "file_sha256": _file_sha256(path),
                        "model_sha256": model.model_sha256,
                        "best_iterations": dict(model.best_iterations),
                        "iteration_selection_mae_skill": dict(
                            model.iteration_selection_mae_skill
                        ),
                        "reload_max_abs_prediction_error_bps": (
                            model.reload_max_abs_prediction_error_bps
                        ),
                        "backend_kind": model.backend_kind,
                        "backend_device": model.backend_device,
                    }
                )
                progress(
                    "model_artifact_complete",
                    {
                        "treatment_id": treatment_id,
                        "view_id": view_id,
                        "seed": seed,
                        "bytes": path.stat().st_size,
                    },
                )
        prediction_path = evidence_root / f"{treatment_id}-predictions.npy"
        np.save(prediction_path, predictions, allow_pickle=False)
        reloaded_predictions = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(reloaded_predictions, predictions):
            raise RuntimeError("Round 55 prediction cache reload differs")
        model_manifests[treatment_id] = {
            "artifacts": artifacts,
            "prediction_path": str(prediction_path.resolve()),
            "prediction_bytes": prediction_path.stat().st_size,
            "prediction_sha256": _file_sha256(prediction_path),
        }
        prediction_diagnostics[treatment_id] = _predictive_diagnostics(
            predictions, payoff, intervals
        )
        stability = _stability_diagnostics(
            predictions, intervals["development_holdout"]
        )
        stability_rows[treatment_id] = stability
        decisions = consensus_decisions(predictions, features, names)
        treatment_replays: dict[str, dict[str, object]] = {}
        treatment_plans: dict[str, object] = {}
        interval_report: dict[str, object] = {}
        for interval_name, interval_mask in intervals.items():
            plan = build_trade_plan(payoff, decisions, interval_mask)
            stress = replay_trade_plan(
                payoff,
                plan,
                interval_mask,
                scenario="stress",
                round_trip_execution_charge_bps=STRESS_COST_BPS,
            )
            base = replay_trade_plan(
                payoff,
                plan,
                interval_mask,
                scenario="base",
                round_trip_execution_charge_bps=BASE_COST_BPS,
            )
            if not np.array_equal(stress.interval_timestamps_ms, base.interval_timestamps_ms):
                raise RuntimeError("Round 55 base and stress timestamps differ")
            bootstrap = block_bootstrap_mean_bps(
                stress.hourly_return_fraction,
                seed=55_100 + len(treatment_replays),
            )
            interval_report[interval_name] = {
                "base": dict(base.metrics),
                "stress": dict(stress.metrics),
                "stress_block_bootstrap_mean_hourly_bps": bootstrap,
            }
            treatment_replays[interval_name] = {"base": base, "stress": stress}
            treatment_plans[interval_name] = plan
            all_trade_rows.extend(_trade_rows(treatment_id, interval_name, plan))
            for scenario_replay in (base, stress):
                monthly_rows.extend(
                    _monthly_rows(treatment_id, interval_name, scenario_replay)
                )
                for timestamp, value in zip(
                    scenario_replay.interval_timestamps_ms,
                    scenario_replay.hourly_return_fraction,
                    strict=True,
                ):
                    hourly_rows.append(
                        {
                            "treatment": treatment_id,
                            "interval": interval_name,
                            "scenario": scenario_replay.scenario,
                            "timestamp_utc": datetime.fromtimestamp(
                                int(timestamp) / 1000.0, UTC
                            ).isoformat(),
                            "initial_capital_return_fraction": float(value),
                        }
                    )
        maximum_reload = max(
            float(item["reload_max_abs_prediction_error_bps"])
            for item in artifacts
        )
        gate = _gate(
            design,
            development=interval_report["policy_development"]["stress"],
            holdout=interval_report["development_holdout"]["stress"],
            holdout_bootstrap=interval_report["development_holdout"][
                "stress_block_bootstrap_mean_hourly_bps"
            ],
            stability=stability,
            maximum_reload_error=maximum_reload,
        )
        treatment_results[treatment_id] = {
            "features": len(names),
            "intervals": interval_report,
            "stability": stability,
            "gate": gate,
        }
        replay_objects[treatment_id] = treatment_replays
        trade_plans[treatment_id] = treatment_plans
        progress(
            "treatment_complete",
            {
                "treatment_id": treatment_id,
                "gate_passed": gate["passed"],
                "gate_failures": len(gate["failures"]),
                "holdout_stress_return": interval_report["development_holdout"][
                    "stress"
                ]["total_return_fraction"],
            },
        )

    baseline_result = treatment_results[TREATMENTS[0]]
    ai_result = treatment_results[TREATMENTS[1]]
    assert isinstance(baseline_result, Mapping) and isinstance(ai_result, Mapping)
    paired_gate = _paired_ai_gate(
        design,
        baseline_result["gate"],
        ai_result["gate"],
        replay_objects[TREATMENTS[0]]["development_holdout"]["stress"],
        replay_objects[TREATMENTS[1]]["development_holdout"]["stress"],
    )
    trade_path = evidence_root / "trades.csv"
    hourly_path = evidence_root / "hourly_ledger.csv"
    monthly_path = evidence_root / "monthly_economics.csv"
    _write_csv(trade_path, all_trade_rows, empty_fieldnames=TRADE_FIELDS)
    _write_csv(hourly_path, hourly_rows)
    _write_csv(monthly_path, monthly_rows)

    retained = [
        treatment
        for treatment, result in treatment_results.items()
        if isinstance(result, Mapping)
        and isinstance(result.get("gate"), Mapping)
        and result["gate"].get("passed") is True
    ]
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA,
        "round": ROUND,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "development_gate_passed" if retained else "rejected",
        "design_sha256": design_sha,
        "binding_sha256": binding["binding_sha256"],
        "implementation_commit": implementation_commit,
        "claims": {
            "profitability": False,
            "untouched_confirmation": False,
            "ai_uplift": bool(paired_gate["passed"]),
            "testnet_readiness": False,
            "live_trading_readiness": False,
            "leverage_readiness": False,
        },
        "data": {
            **cache_evidence,
            "symbols": list(SYMBOLS),
            "feature_count": len(feature_names),
            "decision_resolution_seconds": 3600,
            "execution_path_resolution_seconds": 60,
            "synthetic_rows": 0,
            "selection_contaminated": True,
            "selection_confirmation_or_terminal_rows_read": False,
            "payoff": payoff_evidence,
        },
        "chronology": {
            role: {
                "timestamps": int(np.count_nonzero(mask)),
                "rows": int(np.count_nonzero(mask) * len(SYMBOLS)),
            }
            for role, mask in {**roles, **intervals}.items()
        },
        "ai_factor_research": {
            "report_sha256": ai_report["report_sha256"],
            "ledger_sha256": ai_ledger["ledger_sha256"],
            "syntax_valid_programs": len(programs),
            "runtime_accepted_programs": len(transform.programs),
            "runtime_rejections": list(factor_rejections),
            "transform_sha256": transform.transform_sha256,
            "order_authority": False,
        },
        "model": {
            "views": list(VIEW_IDS),
            "seeds": list(SEEDS),
            "specification": asdict(model_spec),
            "backend_requested": arguments.compute_backend,
            "lightgbm_version": package_metadata.version("lightgbm"),
            "artifacts": model_manifests,
            "predictive_diagnostics": prediction_diagnostics,
        },
        "treatments": treatment_results,
        "ai_uplift_gate": paired_gate,
        "retained_for_separately_frozen_untouched_design": retained,
        "artifacts": {
            "trades_csv": {
                "path": str(trade_path.resolve()),
                "bytes": trade_path.stat().st_size,
                "sha256": _file_sha256(trade_path),
            },
            "hourly_ledger_csv": {
                "path": str(hourly_path.resolve()),
                "bytes": hourly_path.stat().st_size,
                "sha256": _file_sha256(hourly_path),
            },
            "monthly_economics_csv": {
                "path": str(monthly_path.resolve()),
                "bytes": monthly_path.stat().st_size,
                "sha256": _file_sha256(monthly_path),
            },
            "ai_factor_transform": {
                "path": str(transform_path.resolve()),
                "bytes": transform_path.stat().st_size,
                "sha256": _file_sha256(transform_path),
            },
        },
        "trial_accounting": design["trial_accounting"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256"] = _canonical_sha256(report)
    report_path = evidence_root / "report.json"
    write_json_atomic(report_path, report)
    progress(
        "complete",
        {
            "status": report["status"],
            "retained_treatments": len(retained),
            "ai_uplift_gate_passed": paired_gate["passed"],
            "report_sha256": report["report_sha256"],
        },
    )
    return report


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-055-bounded-alpha-lightgbm-ai-design.json",
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=ROOT / "data/market_data.sqlite")
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--ai-report", type=Path, required=True)
    parser.add_argument("--ai-ledger", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend",
        choices=("directml", "cpu"),
        default="directml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
