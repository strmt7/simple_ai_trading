"""Run the hash-bound Round 56 paired-action and governed AI ablation."""

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
import re
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
    ActionConditionedFactorProgram,
    evaluate_factor_program,
    validate_action_conditioned_factor_program,
)
from simple_ai_trading.bounded_alpha_lightgbm import (  # noqa: E402
    block_bootstrap_mean_bps,
    build_trade_plan,
    replay_trade_plan,
)
from simple_ai_trading.cross_asset_cost_data import (  # noqa: E402
    SYMBOLS,
    load_verified_minute_panel_window,
)
from simple_ai_trading.derivatives_hurdle_data import (  # noqa: E402
    load_verified_funding_states,
)
from simple_ai_trading.paired_action_lightgbm import (  # noqa: E402
    ACTION_NAMES,
    OBJECTIVE_IDS,
    VIEW_IDS,
    PairedActionSpec,
    action_conditioned_feature_names,
    apply_paired_action_calibration,
    build_monthly_outer_folds,
    build_paired_action_panel,
    embargoed_interval_mask,
    fit_paired_action_calibration,
    load_paired_action_calibration,
    load_paired_action_model,
    paired_action_decisions,
    pinball_loss,
    predict_paired_action_model,
    save_paired_action_calibration,
    save_paired_action_model,
    train_paired_action_model,
)
from simple_ai_trading.stop_time_payoff_data import (  # noqa: E402
    EVENT_NAMES,
    StopTimeSpecification,
    build_stop_time_payoff_dataset,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


ROUND = 56
DESIGN_SCHEMA = "round-056-paired-action-distributional-design-v1"
BINDING_SCHEMA = "round-056-paired-action-execution-binding-v1"
AI_REPORT_SCHEMA = "round-056-ai-factor-research-report-v1"
AI_LEDGER_SCHEMA = "round-056-action-conditioned-factor-program-ledger-v1"
REPORT_SCHEMA = "round-056-paired-action-report-v1"
EXPECTED_DATASET_SHA256 = (
    "13086282510f69862552dfc7d85839d6910bb5cfd3e67b69f6c879ccd1c8837f"
)
EXPECTED_SOURCE_CERTIFICATE_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
SEEDS = (5601, 5602, 5603)
TREATMENTS = ("baseline_72", "ai_program_augmented")
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
    "lower_tail_size_multiplier",
    "score_bps",
    "lower_tail_bps",
    "stop_bps",
    "event",
    "stress_net_payoff_bps",
    "stress_initial_capital_return_fraction",
)
HOURLY_FIELDS = (
    "treatment",
    "interval",
    "scenario",
    "timestamp_utc",
    "initial_capital_return_fraction",
)
MONTHLY_FIELDS = (
    "treatment",
    "interval",
    "scenario",
    "month",
    "hours",
    "return_fraction",
    "maximum_drawdown_fraction",
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
        raise ValueError("Round 56 Git binding command failed") from exc


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
        print(f"round56 {stage} {detail}".rstrip(), flush=True)


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 56 design")
    canonical = dict(design)
    claimed = str(canonical.pop("design_sha256", ""))
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen_development_only"
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 56 design identity is invalid")
    source = design.get("source_contract")
    model = design.get("model_contract")
    payoff = design.get("payoff_contract")
    action = design.get("action_representation")
    if not all(isinstance(item, Mapping) for item in (source, model, payoff, action)):
        raise ValueError("Round 56 design sections are absent")
    if (
        source.get("symbols") != list(SYMBOLS)
        or source.get("feature_dataset_sha256") != EXPECTED_DATASET_SHA256
        or source.get("source_certificate_canonical_sha256")
        != EXPECTED_SOURCE_CERTIFICATE_SHA256
        or model.get("seeds") != list(SEEDS)
        or [row.get("id") for row in model.get("views", [])] != list(VIEW_IDS)
        or [row.get("id") for row in model.get("objectives", [])]
        != list(OBJECTIVE_IDS)
        or action.get("model_feature_count") != 72
        or payoff.get("training_round_trip_execution_charge_bps")
        != STRESS_COST_BPS
    ):
        raise ValueError("Round 56 design and implementation constants differ")
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
        raise ValueError(f"Round 56 {label} file identity differs")


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
    derived_cache: Path,
    source_certificate: Path,
    ai_report: Path,
    ai_ledger: Path,
    rejected_ai_report: Path,
) -> tuple[dict[str, object], dict[str, Path]]:
    binding = _read_object(path, "Round 56 execution binding")
    canonical = dict(binding)
    claimed = str(canonical.pop("binding_sha256", ""))
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 56 execution binding identity is invalid")
    implementation_commit = str(binding.get("implementation_commit", ""))
    if not implementation_commit or subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            implementation_commit,
            "HEAD",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise ValueError("Round 56 implementation commit is not an ancestor of HEAD")
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 56 execution binding has no blobs")
    for row in blobs:
        if not isinstance(row, Mapping):
            raise ValueError("Round 56 blob row is invalid")
        source_path = str(row.get("path", ""))
        expected = str(row.get("git_blob_oid", ""))
        if _git("rev-parse", f"{implementation_commit}:{source_path}") != expected:
            raise ValueError(f"Round 56 implementation blob differs: {source_path}")

    external = binding.get("external_evidence")
    if not isinstance(external, Mapping):
        raise ValueError("Round 56 external evidence binding is absent")
    for key, candidate in (
        ("source_certificate", source_certificate),
        ("ai_report", ai_report),
        ("ai_ledger", ai_ledger),
        ("rejected_ai_report", rejected_ai_report),
    ):
        row = external.get(key)
        if not isinstance(row, Mapping):
            raise ValueError(f"Round 56 {key} binding is absent")
        _validate_file_manifest(row, candidate, key)

    rows = binding.get("derived_cache")
    if not isinstance(rows, list):
        raise ValueError("Round 56 derived-cache binding is absent")
    paths: dict[str, Path] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Round 56 derived-cache row is invalid")
        name = str(row.get("name", ""))
        candidate = (derived_cache / name).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Round 56 cache file is absent: {candidate}")
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
        raise ValueError("Round 56 derived-cache file set differs")
    return binding, paths


def _load_development_cache(
    paths: Mapping[str, Path],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, object]]:
    metadata = _read_object(paths["metadata.json"], "Round 56 cache metadata")
    feature_names = tuple(str(value) for value in metadata.get("feature_names", []))
    if (
        metadata.get("schema_version") != "round-045-derived-dataset-cache-v1"
        or metadata.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or metadata.get("symbols") != list(SYMBOLS)
        or len(feature_names) != 71
        or len(action_conditioned_feature_names(feature_names)) != 72
    ):
        raise ValueError("Round 56 cache metadata differs")
    timestamps_source = np.load(
        paths["timestamps_ms.npy"], mmap_mode="r", allow_pickle=False
    )
    feature_source = np.load(paths["features.npy"], mmap_mode="r", allow_pickle=False)
    if timestamps_source.shape != (30_647,) or feature_source.shape != (30_647, 3, 71):
        raise ValueError("Round 56 cache array shapes differ")
    forbidden_start = int(np.searchsorted(timestamps_source, _date_ms("2024-10-01")))
    if forbidden_start <= 0 or forbidden_start >= timestamps_source.size:
        raise ValueError("Round 56 forbidden-row boundary is invalid")
    timestamps = np.asarray(timestamps_source[:forbidden_start], dtype=np.int64).copy()
    features = np.asarray(feature_source[:forbidden_start], dtype=np.float32).copy()
    if (
        np.any(np.diff(timestamps) <= 0)
        or timestamps[-1] >= _date_ms("2024-10-01")
        or not np.isfinite(features).all()
    ):
        raise ValueError("Round 56 development cache is invalid")
    return timestamps, features, feature_names, {
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "source_timestamps": int(timestamps_source.size),
        "development_timestamps": int(timestamps.size),
        "first_development_timestamp_ms": int(timestamps[0]),
        "last_development_timestamp_ms": int(timestamps[-1]),
        "forbidden_existing_rows_loaded": False,
        "forbidden_existing_rows": int(timestamps_source.size - forbidden_start),
    }


def _validate_rejected_ai_trial(
    path: Path,
    *,
    design_sha256: str,
) -> dict[str, object]:
    report = _read_object(path, "Round 56 rejected AI report")
    canonical = dict(report)
    claimed = str(canonical.pop("report_sha256", ""))
    if (
        report.get("schema_version") != AI_REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("design_sha256") != design_sha256
        or report.get("market_values_read") is not False
        or report.get("timestamps_read") is not False
        or report.get("outcomes_read") is not False
        or report.get("trading_authority") is not False
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 56 rejected AI trial identity is invalid")
    return {
        "report_sha256": claimed,
        "file_sha256": _file_sha256(path),
        "reason": "prompt mentioned the excluded side-identity feature name; no values, timestamps, outcomes, or trading authority were exposed",
        "programs_used": 0,
    }


def _validate_ai_evidence(
    report_path: Path,
    ledger_path: Path,
    visible_feature_names: Sequence[str],
    *,
    design_sha256: str,
    implementation_commit: str,
) -> tuple[
    tuple[ActionConditionedFactorProgram, ...],
    dict[str, object],
    dict[str, object],
]:
    report = _read_object(report_path, "Round 56 AI report")
    report_canonical = dict(report)
    report_sha = str(report_canonical.pop("report_sha256", ""))
    ai_implementation_commit = str(report.get("implementation_commit", ""))
    if (
        report.get("schema_version") != AI_REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "complete"
        or report.get("design_sha256") != design_sha256
        or not ai_implementation_commit
        or report.get("market_values_read") is not False
        or report.get("timestamps_read") is not False
        or report.get("outcomes_read") is not False
        or report.get("trading_authority") is not False
        or report_sha != _canonical_sha256(report_canonical)
    ):
        raise ValueError("Round 56 AI report identity is invalid")
    if subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            ai_implementation_commit,
            implementation_commit,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise ValueError("Round 56 AI implementation is not an ancestor")
    implementation_blobs = report.get("implementation_blobs")
    if not isinstance(implementation_blobs, Mapping) or not implementation_blobs:
        raise ValueError("Round 56 AI implementation blobs are absent")
    for source_path, expected_oid in implementation_blobs.items():
        path_text = str(source_path)
        oid_text = str(expected_oid)
        if (
            _git("rev-parse", f"{ai_implementation_commit}:{path_text}") != oid_text
            or _git("rev-parse", f"{implementation_commit}:{path_text}") != oid_text
        ):
            raise ValueError(f"Round 56 AI implementation blob changed: {path_text}")

    ledger = _read_object(ledger_path, "Round 56 AI factor ledger")
    ledger_canonical = dict(ledger)
    ledger_sha = str(ledger_canonical.pop("ledger_sha256", ""))
    rows = ledger.get("programs")
    if (
        ledger.get("schema_version") != AI_LEDGER_SCHEMA
        or ledger.get("round") != ROUND
        or ledger.get("design_sha256") != design_sha256
        or ledger.get("source_dataset_sha256") != EXPECTED_DATASET_SHA256
        or ledger.get("language_model_visible_feature_names")
        != list(visible_feature_names)
        or "action_sign" in visible_feature_names
        or ledger.get("market_values_timestamps_or_outcomes_read") is not False
        or ledger.get("action_sign_visible_to_language_model") is not False
        or ledger.get("order_authority") is not False
        or ledger.get("position_sizing_authority") is not False
        or ledger.get("risk_gate_override") is not False
        or ledger_sha != _canonical_sha256(ledger_canonical)
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("Round 56 AI factor ledger identity is invalid")
    programs: list[ActionConditionedFactorProgram] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Round 56 AI factor program row is invalid")
        input_mapping = {
            key: row[key]
            for key in (
                "name",
                "expression",
                "mechanism",
                "failure_mode",
                "expected_horizon",
                "action_symmetry",
            )
        }
        validated = validate_action_conditioned_factor_program(
            input_mapping,
            model=str(row.get("model", "")),
            feature_names=visible_feature_names,
        )
        if validated.asdict() != dict(row):
            raise ValueError("Round 56 AI factor program canonical form differs")
        if validated.canonical_expression in seen:
            raise ValueError("Round 56 AI factor ledger contains a duplicate expression")
        seen.add(validated.canonical_expression)
        programs.append(validated)
    return tuple(programs), report, ledger


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
            raise RuntimeError("Round 56 payoff cache reload differs")
    manifest: dict[str, object] = {
        "schema_version": "round-056-stop-time-payoff-manifest-v1",
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


def _factor_output_name(program: ActionConditionedFactorProgram) -> str:
    model = re.sub(r"[^a-z0-9]+", "_", program.model.lower()).strip("_")
    return f"ai_{model}_{program.name}"


def _evaluate_ai_factors(
    programs: Sequence[ActionConditionedFactorProgram],
    paired_panel: object,
    first_training_mask: np.ndarray,
    final_training_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], list[dict[str, object]]]:
    visible_names = paired_panel.feature_names[:-1]
    matrix = paired_panel.features[..., :-1].reshape(-1, len(visible_names))
    first_rows = np.repeat(first_training_mask, len(SYMBOLS) * len(ACTION_NAMES))
    final_rows = np.repeat(final_training_mask, len(SYMBOLS) * len(ACTION_NAMES))
    columns: list[np.ndarray] = []
    names: list[str] = []
    rejections: list[dict[str, object]] = []
    for program in programs:
        try:
            values = evaluate_factor_program(
                program.as_factor_program(), matrix, visible_names
            )
            first_span = float(
                np.subtract(*np.quantile(values[first_rows], (0.995, 0.005)))
            )
            final_span = float(
                np.subtract(*np.quantile(values[final_rows], (0.995, 0.005)))
            )
            if first_span <= 1e-12 or final_span <= 1e-12:
                raise ValueError("degenerate_training_distribution")
        except ValueError as exc:
            rejections.append(
                {
                    "model": program.model,
                    "name": program.name,
                    "program_sha256": program.program_sha256,
                    "reason": str(exc),
                }
            )
            continue
        columns.append(values.astype(np.float32))
        names.append(_factor_output_name(program))
    if not columns:
        raise ValueError("Round 56 no AI factor survived feature-only safety checks")
    values = np.column_stack(columns).reshape(
        *paired_panel.target_bps.shape, len(columns)
    )
    return values.astype(np.float32), tuple(names), rejections


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return None
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else None


def _skill(loss: float, baseline_loss: float) -> float:
    return 1.0 - loss / baseline_loss if baseline_loss > 0.0 else -math.inf


def _predictive_diagnostics(
    design: Mapping[str, object],
    *,
    calibration: object,
    point_predictions: np.ndarray,
    q20_predictions: np.ndarray,
    point_baselines: np.ndarray,
    q20_baselines: np.ndarray,
    truth_bps: np.ndarray,
    timestamps_ms: np.ndarray,
    maximum_reload_error_bps: float,
) -> tuple[dict[str, object], dict[str, object]]:
    calibrated_point, calibrated_q20 = apply_paired_action_calibration(
        calibration, point_predictions, q20_predictions
    )
    point_score = np.mean(calibrated_point, axis=0)
    q20_score = np.min(calibrated_q20, axis=0)
    point_baseline = np.mean(np.median(point_baselines, axis=1), axis=0)
    q20_baseline = np.min(np.median(q20_baselines, axis=1), axis=0)
    truth = np.asarray(truth_bps, dtype=np.float64)
    finite = all(
        np.isfinite(value).all()
        for value in (
            point_score,
            q20_score,
            point_baseline,
            q20_baseline,
            truth,
        )
    )
    if not finite:
        raise ValueError("Round 56 held-forward prediction is nonfinite")

    pooled_model_mse = float(np.mean(np.square(truth - point_score)))
    pooled_baseline_mse = float(np.mean(np.square(truth - point_baseline)))
    pooled_model_pinball = pinball_loss(truth, q20_score)
    pooled_baseline_pinball = pinball_loss(truth, q20_baseline)
    point_action_skill = {}
    q20_action_skill = {}
    q20_action_coverage = {}
    for action_index, action in enumerate(ACTION_NAMES):
        action_truth = truth[..., action_index]
        action_point = point_score[..., action_index]
        action_point_baseline = point_baseline[..., action_index]
        action_q20 = q20_score[..., action_index]
        action_q20_baseline = q20_baseline[..., action_index]
        point_action_skill[action] = _skill(
            float(np.mean(np.square(action_truth - action_point))),
            float(np.mean(np.square(action_truth - action_point_baseline))),
        )
        q20_action_skill[action] = _skill(
            pinball_loss(action_truth, action_q20),
            pinball_loss(action_truth, action_q20_baseline),
        )
        q20_action_coverage[action] = float(np.mean(action_truth <= action_q20))

    flattened_score = point_score.reshape(-1)
    flattened_truth = truth.reshape(-1)
    pooled_spearman = _spearman(flattened_score, flattened_truth)
    months = np.asarray(
        [
            datetime.fromtimestamp(int(value) / 1000.0, UTC).strftime("%Y-%m")
            for value in timestamps_ms
        ]
    )
    monthly_spearman: dict[str, float | None] = {}
    for month in sorted(set(months)):
        selected = months == month
        monthly_spearman[month] = _spearman(
            point_score[selected].reshape(-1), truth[selected].reshape(-1)
        )
    positive_months = sum(
        value is not None and value > 0.0 for value in monthly_spearman.values()
    )
    order = np.argsort(flattened_score, kind="mergesort")
    quintile_rows = max(1, flattened_score.size // 5)
    bottom_mean = float(np.mean(flattened_truth[order[:quintile_rows]]))
    top_mean = float(np.mean(flattened_truth[order[-quintile_rows:]]))
    coverage = float(np.mean(truth <= q20_score))
    diagnostics: dict[str, object] = {
        "role": "2024-01-01 through 2024-06-30 held-forward calibration validation",
        "rows": int(truth.size),
        "timestamps": int(truth.shape[0]),
        "finite": finite,
        "point": {
            "model_mse_bps_squared": pooled_model_mse,
            "causal_constant_mse_bps_squared": pooled_baseline_mse,
            "pooled_mse_skill": _skill(pooled_model_mse, pooled_baseline_mse),
            "action_mse_skill": point_action_skill,
            "pooled_spearman": pooled_spearman,
            "monthly_spearman": monthly_spearman,
            "positive_spearman_months": positive_months,
            "top_score_quintile_realized_mean_bps": top_mean,
            "bottom_score_quintile_realized_mean_bps": bottom_mean,
        },
        "lower_tail": {
            "model_pinball_bps": pooled_model_pinball,
            "causal_constant_pinball_bps": pooled_baseline_pinball,
            "pooled_pinball_skill": _skill(
                pooled_model_pinball, pooled_baseline_pinball
            ),
            "action_pinball_skill": q20_action_skill,
            "pooled_coverage": coverage,
            "action_coverage": q20_action_coverage,
            "conformal_guarantee_claimed": False,
        },
        "maximum_reload_prediction_error_bps": maximum_reload_error_bps,
    }

    frozen = design["predictive_gates"]
    assert isinstance(frozen, Mapping)
    checks = {
        "finite_predictions": finite,
        "pooled_point_mse_skill": diagnostics["point"]["pooled_mse_skill"]
        >= float(frozen["minimum_pooled_point_mse_skill_vs_causal_constant"]),
        "long_point_mse_skill": point_action_skill["long"]
        >= float(frozen["minimum_each_action_point_mse_skill_vs_causal_constant"]),
        "short_point_mse_skill": point_action_skill["short"]
        >= float(frozen["minimum_each_action_point_mse_skill_vs_causal_constant"]),
        "pooled_q20_pinball_skill": diagnostics["lower_tail"][
            "pooled_pinball_skill"
        ]
        >= float(frozen["minimum_pooled_q20_pinball_skill_vs_causal_constant"]),
        "long_q20_pinball_skill": q20_action_skill["long"]
        >= float(frozen["minimum_each_action_q20_pinball_skill_vs_causal_constant"]),
        "short_q20_pinball_skill": q20_action_skill["short"]
        >= float(frozen["minimum_each_action_q20_pinball_skill_vs_causal_constant"]),
        "pooled_spearman": pooled_spearman is not None
        and pooled_spearman > float(frozen["minimum_pooled_spearman"]),
        "positive_spearman_months": positive_months
        >= int(frozen["minimum_months_with_positive_spearman"]),
        "score_quintile_order": top_mean > bottom_mean,
        "top_score_quintile_positive": top_mean
        > float(frozen["minimum_top_score_quintile_realized_mean_bps"]),
        "pooled_q20_coverage": float(frozen["minimum_pooled_q20_coverage"])
        <= coverage
        <= float(frozen["maximum_pooled_q20_coverage"]),
        "long_q20_coverage": float(frozen["minimum_each_action_q20_coverage"])
        <= q20_action_coverage["long"]
        <= float(frozen["maximum_each_action_q20_coverage"]),
        "short_q20_coverage": float(frozen["minimum_each_action_q20_coverage"])
        <= q20_action_coverage["short"]
        <= float(frozen["maximum_each_action_q20_coverage"]),
        "model_reload": maximum_reload_error_bps
        <= float(frozen["maximum_reload_prediction_error_bps"]),
    }
    gate = {
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
        "checks": checks,
    }
    return diagnostics, gate


def _development_gate(
    design: Mapping[str, object],
    *,
    development: Mapping[str, object],
    holdout: Mapping[str, object],
    holdout_bootstrap: Mapping[str, object],
) -> dict[str, object]:
    frozen = design["development_gates"]
    assert isinstance(frozen, Mapping)

    def metric(source: Mapping[str, object], name: str, fallback: float) -> float:
        value = source.get(name)
        return fallback if value is None else float(value)

    checks = {
        "development_closed_trades": int(development["closed_trades"])
        >= int(frozen["minimum_development_closed_trades"]),
        "development_active_days": int(development["active_days"])
        >= int(frozen["minimum_development_active_days"]),
        "development_profit_factor": metric(development, "profit_factor", -math.inf)
        >= float(frozen["minimum_development_stress_profit_factor"]),
        "development_drawdown": float(development["maximum_drawdown_fraction"])
        <= float(frozen["maximum_development_stress_drawdown_fraction"]),
        "development_return": float(development["total_return_fraction"])
        > float(frozen["minimum_development_stress_return_fraction"]),
        "holdout_closed_trades": int(holdout["closed_trades"])
        >= int(frozen["minimum_holdout_closed_trades"]),
        "holdout_active_days": int(holdout["active_days"])
        >= int(frozen["minimum_holdout_active_days"]),
        "holdout_profit_factor": metric(holdout, "profit_factor", -math.inf)
        >= float(frozen["minimum_holdout_stress_profit_factor"]),
        "holdout_drawdown": float(holdout["maximum_drawdown_fraction"])
        <= float(frozen["maximum_holdout_stress_drawdown_fraction"]),
        "holdout_return": float(holdout["total_return_fraction"])
        > float(frozen["minimum_holdout_stress_return_fraction"]),
        "holdout_symbol_activity": int(holdout["symbols_with_trades"])
        >= int(frozen["required_symbol_activity"]),
        "holdout_symbol_concentration": float(
            holdout["maximum_single_symbol_fraction_of_absolute_net_pnl"]
        )
        <= float(frozen["maximum_single_symbol_fraction_of_absolute_net_pnl"]),
        "holdout_familywise_bootstrap": float(holdout_bootstrap["lower_bps"])
        > float(frozen["minimum_168h_block_bootstrap_lower_mean_hourly_bps"]),
    }
    return {
        "passed": all(checks.values()),
        "failures": [name for name, passed in checks.items() if not passed],
        "checks": checks,
    }


def _paired_ai_gate(
    design: Mapping[str, object],
    baseline_result: Mapping[str, object],
    ai_result: Mapping[str, object],
    replay_objects: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    frozen = design["ai_uplift_gate"]
    assert isinstance(frozen, Mapping)
    prerequisite = {
        "baseline_predictive_gate": bool(baseline_result["predictive_gate"]["passed"]),
        "ai_predictive_gate": bool(ai_result["predictive_gate"]["passed"]),
        "baseline_economic_gate": bool(
            isinstance(baseline_result.get("development_gate"), Mapping)
            and baseline_result["development_gate"].get("passed") is True
        ),
        "ai_economic_gate": bool(
            isinstance(ai_result.get("development_gate"), Mapping)
            and ai_result["development_gate"].get("passed") is True
        ),
    }
    if not all(prerequisite.values()):
        return {
            "passed": False,
            "failures": [name for name, value in prerequisite.items() if not value],
            "checks": prerequisite,
            "paired_hourly_bootstrap": None,
        }
    baseline = replay_objects[TREATMENTS[0]]["development_holdout"]
    ai = replay_objects[TREATMENTS[1]]["development_holdout"]
    delta = ai.hourly_return_fraction - baseline.hourly_return_fraction
    bootstrap = block_bootstrap_mean_bps(delta, seed=56_092)
    baseline_metrics = baseline.metrics
    ai_metrics = ai.metrics
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
        **prerequisite,
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
        [
            datetime.fromtimestamp(value / 1000.0, UTC).strftime("%Y-%m")
            for value in timestamps
        ]
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
    empty_fieldnames: Sequence[str],
) -> None:
    fieldnames = list(rows[0]) if rows else list(empty_fieldnames)
    if not fieldnames or any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"Round 56 CSV fields differ: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _trade_rows(
    treatment: str,
    interval: str,
    plan: object,
    decisions: object,
) -> list[dict[str, object]]:
    rows = []
    for index in range(plan.closed_trades):
        decision_index = int(plan.decision_index[index])
        symbol_index = int(plan.symbol_index[index])
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
                "symbol": SYMBOLS[symbol_index],
                "side": "long" if int(plan.side[index]) == 1 else "short",
                "size_fraction": float(plan.size_fraction[index]),
                "lower_tail_size_multiplier": float(
                    decisions.size_multiplier[decision_index, symbol_index]
                ),
                "score_bps": float(plan.score_bps[index]),
                "lower_tail_bps": float(
                    decisions.lower_tail_bps[decision_index, symbol_index]
                ),
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


def _cache_predictions(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, object]:
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as reloaded:
        if set(reloaded.files) != set(arrays):
            raise RuntimeError("Round 56 prediction cache fields differ")
        for name, value in arrays.items():
            if not np.array_equal(reloaded[name], value, equal_nan=True):
                raise RuntimeError(f"Round 56 prediction cache differs: {name}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }


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
        rejected_ai_report=arguments.rejected_ai_report.resolve(),
    )
    implementation_commit = str(binding["implementation_commit"])
    timestamps, source_features, source_feature_names, cache_evidence = (
        _load_development_cache(cache_paths)
    )
    paired_names = action_conditioned_feature_names(source_feature_names)
    programs, ai_report, ai_ledger = _validate_ai_evidence(
        arguments.ai_report.resolve(),
        arguments.ai_ledger.resolve(),
        paired_names[:-1],
        design_sha256=design_sha,
        implementation_commit=implementation_commit,
    )
    rejected_ai_trial = _validate_rejected_ai_trial(
        arguments.rejected_ai_report.resolve(), design_sha256=design_sha
    )
    progress(
        "contracts_validated",
        {
            "design_sha256": design_sha,
            "binding_sha256": binding["binding_sha256"],
            "implementation_commit": implementation_commit,
            "development_timestamps": len(timestamps),
            "syntax_valid_ai_programs": len(programs),
            "rejected_ai_prompt_trials": 1,
        },
    )

    panel, price_source = load_verified_minute_panel_window(
        arguments.database.resolve(),
        materialization_end="2024-10-01",
        progress=progress,
    )
    funding, funding_source = load_verified_funding_states(
        arguments.database.resolve(),
        panel,
        price_source,
        source_certificate_path=arguments.source_certificate.resolve(),
        progress=progress,
    )
    volatility_index = source_feature_names.index(
        "target_realized_volatility_60m_bps"
    )
    payoff = build_stop_time_payoff_dataset(
        panel,
        funding,
        timestamps,
        source_features[..., volatility_index],
        source_dataset_sha256=EXPECTED_DATASET_SHA256,
        specification=StopTimeSpecification(
            horizon_minutes=60,
            stop_volatility_multiple=1.5,
            minimum_stop_bps=40.0,
            maximum_stop_bps=250.0,
            round_trip_execution_charge_bps=STRESS_COST_BPS,
        ),
    )
    source_evidence = funding_source.asdict()
    del panel, funding, price_source, funding_source
    gc.collect()
    payoff_evidence = _payoff_manifest(evidence_root, payoff, source_evidence)
    paired = build_paired_action_panel(
        features=source_features,
        feature_names=source_feature_names,
        timestamps_ms=timestamps,
        stop_bps=payoff.stop_bps,
        long_target_bps=payoff.long_net_payoff_bps,
        short_target_bps=payoff.short_net_payoff_bps,
        long_exit_time_ms=payoff.long_exit_time_ms,
        short_exit_time_ms=payoff.short_exit_time_ms,
    )
    if paired.feature_names != paired_names or paired.rows != payoff.rows * 2:
        raise RuntimeError("Round 56 paired-action panel identity differs")
    outer_folds = build_monthly_outer_folds(paired)
    calibration_fit_mask = embargoed_interval_mask(
        paired, start="2023-07-01", end="2024-01-01"
    )
    calibration_validation_mask = embargoed_interval_mask(
        paired, start="2024-01-01", end="2024-07-01"
    )
    all_oof_calibration_mask = embargoed_interval_mask(
        paired, start="2023-07-01", end="2024-07-01"
    )
    final_refit_mask = embargoed_interval_mask(
        paired, start="2022-01-01", end="2024-07-01"
    )
    policy_probe_mask = embargoed_interval_mask(
        paired, start="2024-07-01", end="2024-10-01"
    )
    intervals = {
        "policy_development": embargoed_interval_mask(
            paired, start="2024-07-01", end="2024-09-01"
        ),
        "development_holdout": embargoed_interval_mask(
            paired, start="2024-09-01", end="2024-10-01"
        ),
    }
    factor_values, factor_names, factor_runtime_rejections = _evaluate_ai_factors(
        programs,
        paired,
        outer_folds[0].training_mask,
        final_refit_mask,
    )
    factor_evidence = {
        "programs_from_ledger": len(programs),
        "runtime_accepted_programs": len(factor_names),
        "runtime_rejections": factor_runtime_rejections,
        "factor_feature_names": list(factor_names),
        "factor_values_sha256": hashlib.sha256(
            np.ascontiguousarray(factor_values).view(np.uint8)
        ).hexdigest(),
        "market_outcomes_used_for_factor_selection": False,
    }
    write_json_atomic(evidence_root / "ai_factor_runtime.json", factor_evidence)
    progress(
        "paired_dataset_ready",
        {
            "payoff_dataset_sha256": payoff.dataset_sha256,
            "paired_rows": paired.rows,
            "base_features": len(paired.feature_names),
            "ai_factors": len(factor_names),
            "outer_folds": len(outer_folds),
        },
    )

    model_spec = PairedActionSpec()
    treatment_inputs = {
        TREATMENTS[0]: (None, ()),
        TREATMENTS[1]: (factor_values, factor_names),
    }
    treatment_results: dict[str, dict[str, object]] = {}
    replay_objects: dict[str, dict[str, object]] = {}
    all_trade_rows: list[dict[str, object]] = []
    hourly_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    model_manifests: dict[str, object] = {}
    for treatment_id, (treatment_factors, treatment_factor_names) in treatment_inputs.items():
        shape = (
            len(VIEW_IDS),
            len(SEEDS),
            paired.timestamps,
            len(SYMBOLS),
            len(ACTION_NAMES),
        )
        point_final = np.empty(shape, dtype=np.float64)
        q20_final = np.empty(shape, dtype=np.float64)
        point_oof = np.full(shape, np.nan, dtype=np.float64)
        q20_oof = np.full(shape, np.nan, dtype=np.float64)
        point_baseline = np.full(shape, np.nan, dtype=np.float64)
        q20_baseline = np.full(shape, np.nan, dtype=np.float64)
        artifacts: list[dict[str, object]] = []
        for view_index, view_id in enumerate(VIEW_IDS):
            for objective_id in OBJECTIVE_IDS:
                for seed_index, seed in enumerate(SEEDS):
                    result = train_paired_action_model(
                        treatment_id=treatment_id,
                        view_id=view_id,
                        objective_id=objective_id,
                        seed=seed,
                        panel=paired,
                        outer_folds=outer_folds,
                        final_refit_mask=final_refit_mask,
                        prediction_probe_mask=policy_probe_mask,
                        factor_values=treatment_factors,
                        factor_feature_names=treatment_factor_names,
                        source_dataset_sha256=EXPECTED_DATASET_SHA256,
                        payoff_dataset_sha256=payoff.dataset_sha256,
                        design_sha256=design_sha,
                        spec=model_spec,
                        compute_backend=arguments.compute_backend,
                        progress=progress,
                    )
                    path = model_root / (
                        f"{treatment_id}-{view_id}-{objective_id}-seed-{seed}.json"
                    )
                    save_paired_action_model(path, result.model)
                    reloaded = load_paired_action_model(path)
                    final_prediction = predict_paired_action_model(
                        reloaded, paired, treatment_factors
                    )
                    target_final = (
                        point_final
                        if objective_id == OBJECTIVE_IDS[0]
                        else q20_final
                    )
                    target_oof = (
                        point_oof if objective_id == OBJECTIVE_IDS[0] else q20_oof
                    )
                    target_baseline = (
                        point_baseline
                        if objective_id == OBJECTIVE_IDS[0]
                        else q20_baseline
                    )
                    target_final[view_index, seed_index] = final_prediction
                    target_oof[view_index, seed_index] = result.oof_prediction_bps
                    target_baseline[view_index, seed_index] = (
                        result.oof_causal_baseline_bps
                    )
                    artifacts.append(
                        {
                            "treatment_id": treatment_id,
                            "view_id": view_id,
                            "objective_id": objective_id,
                            "seed": seed,
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "file_sha256": _file_sha256(path),
                            "model_sha256": result.model.model_sha256,
                            "final_iterations": result.model.final_iterations,
                            "outer_fold_loss_skill": [
                                row["loss_skill"]
                                for row in result.model.outer_fold_diagnostics
                            ],
                            "reload_max_abs_prediction_error_bps": (
                                result.model.reload_max_abs_prediction_error_bps
                            ),
                            "backend_kind": result.model.backend_kind,
                            "backend_device": result.model.backend_device,
                        }
                    )
                    progress(
                        "model_artifact_complete",
                        {
                            "treatment_id": treatment_id,
                            "view_id": view_id,
                            "objective_id": objective_id,
                            "seed": seed,
                            "bytes": path.stat().st_size,
                        },
                    )

        for mask in (calibration_fit_mask, calibration_validation_mask):
            if not (
                np.isfinite(point_oof[:, :, mask]).all()
                and np.isfinite(q20_oof[:, :, mask]).all()
                and np.isfinite(point_baseline[:, :, mask]).all()
                and np.isfinite(q20_baseline[:, :, mask]).all()
            ):
                raise RuntimeError("Round 56 OOF coverage is incomplete")
        preliminary_calibration = fit_paired_action_calibration(
            treatment_id=treatment_id,
            point_predictions_bps=point_oof,
            q20_predictions_bps=q20_oof,
            truth_bps=paired.target_bps,
            timestamp_mask=calibration_fit_mask,
            timestamps_ms=timestamps,
        )
        validation_diagnostics, predictive_gate = _predictive_diagnostics(
            design,
            calibration=preliminary_calibration,
            point_predictions=point_oof[:, :, calibration_validation_mask],
            q20_predictions=q20_oof[:, :, calibration_validation_mask],
            point_baselines=point_baseline[:, :, calibration_validation_mask],
            q20_baselines=q20_baseline[:, :, calibration_validation_mask],
            truth_bps=paired.target_bps[calibration_validation_mask],
            timestamps_ms=timestamps[calibration_validation_mask],
            maximum_reload_error_bps=max(
                float(row["reload_max_abs_prediction_error_bps"])
                for row in artifacts
            ),
        )
        final_calibration = fit_paired_action_calibration(
            treatment_id=treatment_id,
            point_predictions_bps=point_oof,
            q20_predictions_bps=q20_oof,
            truth_bps=paired.target_bps,
            timestamp_mask=all_oof_calibration_mask,
            timestamps_ms=timestamps,
        )
        calibration_path = evidence_root / f"{treatment_id}-calibration.json"
        save_paired_action_calibration(calibration_path, final_calibration)
        final_calibration = load_paired_action_calibration(calibration_path)
        prediction_path = evidence_root / f"{treatment_id}-predictions.npz"
        prediction_manifest = _cache_predictions(
            prediction_path,
            {
                "point_final_bps": point_final,
                "q20_final_bps": q20_final,
                "point_oof_bps": point_oof,
                "q20_oof_bps": q20_oof,
                "point_causal_baseline_bps": point_baseline,
                "q20_causal_baseline_bps": q20_baseline,
            },
        )
        model_manifests[treatment_id] = {
            "artifacts": artifacts,
            "prediction_cache": prediction_manifest,
            "calibration": {
                "path": str(calibration_path.resolve()),
                "bytes": calibration_path.stat().st_size,
                "file_sha256": _file_sha256(calibration_path),
                "calibration_sha256": final_calibration.calibration_sha256,
            },
        }
        result_report: dict[str, object] = {
            "features": len(paired.feature_names) + len(treatment_factor_names),
            "predictive_validation": validation_diagnostics,
            "predictive_gate": predictive_gate,
            "economic_status": "not_evaluated_due_predictive_failure",
            "development_gate": None,
            "intervals": {},
        }
        if predictive_gate["passed"]:
            decisions = paired_action_decisions(
                point_predictions_bps=point_final,
                q20_predictions_bps=q20_final,
                calibration=final_calibration,
                source_features=source_features,
                source_feature_names=source_feature_names,
                stop_bps=payoff.stop_bps,
            )
            selected_actions = decisions.actions[decisions.actions != 0]
            long_fraction = (
                float(np.mean(selected_actions == 1))
                if selected_actions.size
                else None
            )
            warning_threshold = float(
                design["predictive_gates"]["side_preference_warning_fraction"]
            )
            side_warning = bool(
                long_fraction is not None
                and max(long_fraction, 1.0 - long_fraction) > warning_threshold
            )
            interval_report: dict[str, object] = {}
            treatment_replays: dict[str, object] = {}
            for interval_index, (interval_name, interval_mask) in enumerate(
                intervals.items()
            ):
                plan = build_trade_plan(
                    payoff,
                    decisions,
                    interval_mask,
                    size_multiplier=decisions.size_multiplier,
                )
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
                bootstrap = block_bootstrap_mean_bps(
                    stress.hourly_return_fraction,
                    seed=56_100 + interval_index,
                )
                interval_report[interval_name] = {
                    "base": dict(base.metrics),
                    "stress": dict(stress.metrics),
                    "stress_block_bootstrap_mean_hourly_bps": bootstrap,
                }
                treatment_replays[interval_name] = stress
                all_trade_rows.extend(
                    _trade_rows(treatment_id, interval_name, plan, decisions)
                )
                for replay in (base, stress):
                    monthly_rows.extend(
                        _monthly_rows(treatment_id, interval_name, replay)
                    )
                    for timestamp, value in zip(
                        replay.interval_timestamps_ms,
                        replay.hourly_return_fraction,
                        strict=True,
                    ):
                        hourly_rows.append(
                            {
                                "treatment": treatment_id,
                                "interval": interval_name,
                                "scenario": replay.scenario,
                                "timestamp_utc": datetime.fromtimestamp(
                                    int(timestamp) / 1000.0, UTC
                                ).isoformat(),
                                "initial_capital_return_fraction": float(value),
                            }
                        )
            development_gate = _development_gate(
                design,
                development=interval_report["policy_development"]["stress"],
                holdout=interval_report["development_holdout"]["stress"],
                holdout_bootstrap=interval_report["development_holdout"][
                    "stress_block_bootstrap_mean_hourly_bps"
                ],
            )
            result_report.update(
                {
                    "economic_status": "evaluated",
                    "development_gate": development_gate,
                    "intervals": interval_report,
                    "action_diagnostics": {
                        "eligible_actions": int(selected_actions.size),
                        "long_fraction": long_fraction,
                        "side_preference_warning": side_warning,
                        "warning_threshold": warning_threshold,
                        "mean_lower_tail_size_multiplier": float(
                            np.mean(
                                decisions.size_multiplier[
                                    decisions.size_multiplier > 0.0
                                ]
                            )
                        )
                        if np.any(decisions.size_multiplier > 0.0)
                        else None,
                    },
                }
            )
            replay_objects[treatment_id] = treatment_replays
        treatment_results[treatment_id] = result_report
        progress(
            "treatment_complete",
            {
                "treatment_id": treatment_id,
                "predictive_gate_passed": predictive_gate["passed"],
                "predictive_failures": len(predictive_gate["failures"]),
                "economic_status": result_report["economic_status"],
            },
        )

    paired_gate = _paired_ai_gate(
        design,
        treatment_results[TREATMENTS[0]],
        treatment_results[TREATMENTS[1]],
        replay_objects,
    )
    trade_path = evidence_root / "trades.csv"
    hourly_path = evidence_root / "hourly_ledger.csv"
    monthly_path = evidence_root / "monthly_economics.csv"
    _write_csv(trade_path, all_trade_rows, empty_fieldnames=TRADE_FIELDS)
    _write_csv(hourly_path, hourly_rows, empty_fieldnames=HOURLY_FIELDS)
    _write_csv(monthly_path, monthly_rows, empty_fieldnames=MONTHLY_FIELDS)
    retained = [
        treatment
        for treatment, result in treatment_results.items()
        if result["predictive_gate"]["passed"]
        and isinstance(result.get("development_gate"), Mapping)
        and result["development_gate"].get("passed") is True
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
            "source_feature_count": len(source_feature_names),
            "paired_base_feature_count": len(paired.feature_names),
            "decision_resolution_seconds": 3600,
            "execution_path_resolution_seconds": 60,
            "synthetic_rows": 0,
            "selection_contaminated": True,
            "forbidden_existing_rows_read": False,
            "payoff": payoff_evidence,
        },
        "chronology": {
            "outer_folds": [fold.evidence() for fold in outer_folds],
            "calibration_fit_timestamps": int(
                np.count_nonzero(calibration_fit_mask)
            ),
            "calibration_validation_timestamps": int(
                np.count_nonzero(calibration_validation_mask)
            ),
            "final_refit_timestamps": int(np.count_nonzero(final_refit_mask)),
            "forbidden_existing_rows_loaded": False,
        },
        "ai_factor_research": {
            "report_sha256": ai_report["report_sha256"],
            "ledger_sha256": ai_ledger["ledger_sha256"],
            "syntax_valid_programs": len(programs),
            **factor_evidence,
            "rejected_prompt_trial": rejected_ai_trial,
            "order_authority": False,
        },
        "model": {
            "views": list(VIEW_IDS),
            "objectives": list(OBJECTIVE_IDS),
            "seeds": list(SEEDS),
            "specification": asdict(model_spec),
            "backend_requested": arguments.compute_backend,
            "lightgbm_version": package_metadata.version("lightgbm"),
            "artifacts": model_manifests,
        },
        "treatments": treatment_results,
        "ai_uplift_gate": paired_gate,
        "retained_for_separately_frozen_next_design": retained,
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
            "ai_factor_runtime": {
                "path": str((evidence_root / "ai_factor_runtime.json").resolve()),
                "bytes": (evidence_root / "ai_factor_runtime.json").stat().st_size,
                "sha256": _file_sha256(evidence_root / "ai_factor_runtime.json"),
            },
        },
        "trial_accounting": {
            **design["trial_accounting"],
            "invalid_ai_prompt_trials": 1,
            "invalid_ai_prompt_programs_used": 0,
        },
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
        default=research / "round-056-paired-action-distributional-design.json",
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=ROOT / "data/market_data.sqlite")
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--ai-report", type=Path, required=True)
    parser.add_argument("--ai-ledger", type=Path, required=True)
    parser.add_argument("--rejected-ai-report", type=Path, required=True)
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
