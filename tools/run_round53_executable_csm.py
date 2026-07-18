"""Run the frozen Round 53 executable CSM mechanism screen."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.compute import SUPPORTED_COMPUTE_BACKENDS  # noqa: E402

from simple_ai_trading.executable_csm_lightgbm import (  # noqa: E402
    ExecutableCsmPredictionBatch,
    ExecutableCsmSpec,
    TrainedExecutableCsmModel,
    load_executable_csm_model,
    predict_executable_csm_model,
    save_executable_csm_model,
    train_executable_csm_model,
)
from simple_ai_trading.executable_payoff_lightgbm import (  # noqa: E402
    ExecutablePayoffPredictionBatch,
    build_executable_payoff_dataset,
    load_executable_payoff_model,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ActionScoreBatch,
    BarrierActionTrace,
)
from simple_ai_trading.payoff_distribution_analysis import (  # noqa: E402
    base_and_paired_stress_traces,
    finite_spearman,
    pairwise_seed_spearman,
    portfolio_trace_metrics,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round51_categorical_payoff_fincast import (  # noqa: E402
    _load_real_symbol_data,
)
from tools.run_round52_executable_support_hurdle import (  # noqa: E402
    _load_fincast_features,
    _role_indexes,
)


ROUND = 53
SCHEMA = "round-053-executable-csm-fincast-report-v1"
BINDING_SCHEMA = "round-053-executable-csm-fincast-execution-binding-v1"
EXPECTED_DESIGN_SHA256 = (
    "58a6df6f34bc4d2cbb660be2c84f80a352a708e315197e45f4ab9922ef7504e4"
)
EXPECTED_ROUND52_REPORT_FILE_SHA256 = (
    "c5b728161535372d934ff9087a24b81c2490246cd17cb56beb3e29a3052d73fa"
)
EXPECTED_ROUND52_REPORT_CANONICAL_SHA256 = (
    "ace44ebc33dc0601306841b4c353b43a184b2aa604b49f73d2301257f86d2f7f"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SEEDS = (5201, 5202, 5203)
SIDES = ("long", "short")
CANDIDATES = (
    "executable_direct_mean_lightgbm",
    "executable_csm_lightgbm",
    "executable_csm_lightgbm_fincast",
)
CSM_CANDIDATES = CANDIDATES[1:]
POLICY_ROLES = ("policy_calibration", "evaluation")
DAY_MS = 86_400_000


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
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _progress(stage: str, **values: object) -> None:
    detail = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"round53 {stage}{(' ' + detail) if detail else ''}", flush=True)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 53 design")
    canonical = dict(design)
    claimed = canonical.pop("design_sha256", None)
    actual = _canonical_sha256(canonical)
    if (
        design.get("schema_version") != "executable-csm-fincast-screen-design-v1"
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or claimed != actual
        or actual != EXPECTED_DESIGN_SHA256
        or design.get("claims", {}).get("selection_contaminated") is not True
        or design.get("claims", {}).get("profitability_claim_permitted") is not False
        or design.get("claims", {}).get("leverage_permitted") is not False
        or design.get("economic_screen", {}).get("leverage") != 1.0
    ):
        raise ValueError("Round 53 design identity or fail-closed claims drifted")
    return design, actual


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
) -> tuple[dict[str, object], str]:
    binding = _read_object(path, "Round 53 binding")
    canonical = dict(binding)
    claimed = canonical.pop("binding_sha256", None)
    actual = _canonical_sha256(canonical)
    contract = binding.get("execution_contract")
    if (
        binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha256
        or claimed != actual
        or not isinstance(contract, dict)
        or contract.get("selection_contaminated") is not True
        or contract.get("trading_authority") is not False
        or contract.get("leverage_applied") is not False
    ):
        raise ValueError("Round 53 binding identity or authority contract drifted")
    implementation_commit = str(binding.get("implementation_commit", ""))
    if len(implementation_commit) != 40:
        raise ValueError("Round 53 implementation commit is invalid")
    try:
        _git("merge-base", "--is-ancestor", implementation_commit, "HEAD")
    except subprocess.CalledProcessError as exc:
        raise ValueError("Round 53 implementation is not an ancestor of HEAD") from exc
    blobs = binding.get("blobs")
    if not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 53 binding blobs are absent")
    for item in blobs:
        if not isinstance(item, dict):
            raise ValueError("Round 53 binding blob entry is invalid")
        relative = str(item.get("path", ""))
        expected = str(item.get("git_blob", ""))
        if not relative or len(expected) != 40:
            raise ValueError("Round 53 binding blob identity is invalid")
        actual_blob = _git("rev-parse", f"{implementation_commit}:{relative}")
        if actual_blob != expected:
            raise ValueError(f"Round 53 bound blob drifted: {relative}")
    return binding, actual


def _validate_round52_report(path: Path) -> dict[str, object]:
    if _file_sha256(path) != EXPECTED_ROUND52_REPORT_FILE_SHA256:
        raise ValueError("Round 52 source report file drifted")
    report = _read_object(path, "Round 52 source report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("schema_version")
        != "round-052-executable-support-hurdle-fincast-report-v1"
        or report.get("round") != 52
        or claimed != EXPECTED_ROUND52_REPORT_CANONICAL_SHA256
        or _canonical_sha256(canonical) != claimed
        or report.get("claims", {}).get("selection_contaminated") is not True
        or report.get("claims", {}).get("trading_authority") is not False
    ):
        raise ValueError("Round 52 source report identity drifted")
    return report


def _specifications(design: Mapping[str, object]) -> dict[str, ExecutableCsmSpec]:
    contract = design["model_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("Round 53 model contract is invalid")
    lightgbm = contract["lightgbm"]
    factorization = contract["csm_factorization"]
    if not isinstance(lightgbm, Mapping) or not isinstance(factorization, Mapping):
        raise ValueError("Round 53 model parameters are invalid")
    if lightgbm.get("minimum_leaf_rows") != (
        "max(64, min(512, ceil(0.002 * side-specific training rows)))"
    ):
        raise ValueError("Round 53 leaf-size contract drifted")
    common = {
        "family": "side_specific_executable_csm",
        "magnitude_edge_quantiles": (0.10, 0.30, 0.50, 0.70, 0.90),
        "learning_rate": float(lightgbm["learning_rate"]),
        "num_leaves": int(lightgbm["num_leaves"]),
        "max_depth": int(lightgbm["max_depth"]),
        "minimum_leaf_fraction": 0.002,
        "minimum_leaf_rows": 64,
        "maximum_leaf_rows": 512,
        "feature_fraction": float(lightgbm["feature_fraction"]),
        "bagging_fraction": float(lightgbm["bagging_fraction"]),
        "bagging_freq": int(lightgbm["bagging_freq"]),
        "lambda_l1": float(lightgbm["lambda_l1"]),
        "lambda_l2": float(lightgbm["lambda_l2"]),
        "max_bin": int(lightgbm["max_bin"]),
        "num_boost_round": int(lightgbm["num_boost_round"]),
        "early_stopping_rounds": int(lightgbm["early_stopping_rounds"]),
        "gpu_use_dp_required": True,
    }
    return {
        candidate: ExecutableCsmSpec(candidate_id=candidate, **common)
        for candidate in CSM_CANDIDATES
    }


def _prediction_from_npz(
    path: Path,
    *,
    architecture: str,
) -> ExecutablePayoffPredictionBatch:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return ExecutablePayoffPredictionBatch(
        architecture=architecture,
        magnitude_floor_count=0,
        **arrays,
    )


def _save_csm_prediction(
    path: Path,
    prediction: ExecutableCsmPredictionBatch,
) -> dict[str, object]:
    arrays = {
        field: np.asarray(getattr(prediction, field))
        for field in (
            "endpoint_indexes",
            "long_expected_net_bps",
            "short_expected_net_bps",
            "long_executable",
            "short_executable",
            "long_profitable_probability",
            "short_profitable_probability",
            "long_q10_net_bps",
            "short_q10_net_bps",
            "long_q90_net_bps",
            "short_q90_net_bps",
            "long_cvar10_net_bps",
            "short_cvar10_net_bps",
            "long_magnitude_probabilities",
            "short_magnitude_probabilities",
            "long_positive_probability_by_magnitude",
            "short_positive_probability_by_magnitude",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        **_artifact(path),
        "rows": prediction.rows,
        "magnitude_classes": prediction.magnitude_classes,
    }


def _binary_log_loss(probability: np.ndarray, outcome: np.ndarray) -> float:
    predicted = np.clip(np.asarray(probability, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    actual = np.asarray(outcome, dtype=np.float64)
    return float(
        -np.mean(actual * np.log(predicted) + (1.0 - actual) * np.log1p(-predicted))
    )


def _multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    chosen = np.clip(values[np.arange(len(values)), target], 1e-15, 1.0)
    return float(-np.mean(np.log(chosen)))


def _ranked_probability_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    cumulative = np.cumsum(values, axis=1)[:, :-1]
    thresholds = np.arange(values.shape[1] - 1, dtype=np.int64)
    observed = (target[:, None] <= thresholds[None, :]).astype(np.float64)
    return float(np.mean(np.sum(np.square(cumulative - observed), axis=1)))


def _csm_prediction_metrics(
    *,
    model: TrainedExecutableCsmModel,
    dataset,
    indexes: np.ndarray,
    prediction: ExecutableCsmPredictionBatch,
    side: str,
) -> dict[str, object]:
    selected = np.asarray(indexes, dtype=np.int64)
    executable = np.asarray(
        prediction.long_executable if side == "long" else prediction.short_executable,
        dtype=np.bool_,
    )
    expected = np.asarray(
        prediction.long_expected_net_bps
        if side == "long"
        else prediction.short_expected_net_bps,
        dtype=np.float64,
    )[executable]
    target_all = np.asarray(
        dataset.payoff.long_net_bps
        if side == "long"
        else dataset.payoff.short_net_bps,
        dtype=np.float64,
    )[selected]
    actual = target_all[executable]
    stop_width = np.asarray(dataset.payoff.stop_width_bps, dtype=np.float64)[
        selected
    ][executable]
    probability = np.asarray(
        prediction.long_profitable_probability
        if side == "long"
        else prediction.short_profitable_probability,
        dtype=np.float64,
    )[executable]
    magnitude_probability = np.asarray(
        prediction.long_magnitude_probabilities
        if side == "long"
        else prediction.short_magnitude_probabilities,
        dtype=np.float64,
    )[executable]
    positive_by_magnitude = np.asarray(
        prediction.long_positive_probability_by_magnitude
        if side == "long"
        else prediction.short_positive_probability_by_magnitude,
        dtype=np.float64,
    )[executable]
    if len(actual) < 32:
        raise ValueError(f"Round 53 {side} prediction metrics lack support")
    classes = model.spec.magnitude_classes
    magnitude = np.abs(actual) / stop_width
    edges = np.asarray(model.magnitude_edges_risk_units[side], dtype=np.float64)
    magnitude_labels = np.searchsorted(edges, magnitude, side="right").astype(np.int64)
    profitable = actual > 0.0
    joint_labels = np.where(
        profitable,
        classes + magnitude_labels,
        classes - 1 - magnitude_labels,
    ).astype(np.int64)
    positive_mass = magnitude_probability * positive_by_magnitude
    negative_mass = magnitude_probability * (1.0 - positive_by_magnitude)
    joint_probability = np.concatenate(
        (negative_mass[:, ::-1], positive_mass), axis=1
    )
    training_joint = np.asarray(
        model.training_joint_probabilities[side], dtype=np.float64
    )
    joint_baseline = np.broadcast_to(training_joint, joint_probability.shape)
    magnitude_baseline_vector = (
        training_joint[:classes][::-1] + training_joint[classes:]
    )
    magnitude_baseline = np.broadcast_to(
        magnitude_baseline_vector, magnitude_probability.shape
    )
    joint_loss = _multiclass_log_loss(joint_probability, joint_labels)
    joint_baseline_loss = _multiclass_log_loss(joint_baseline, joint_labels)
    magnitude_loss = _multiclass_log_loss(magnitude_probability, magnitude_labels)
    magnitude_baseline_loss = _multiclass_log_loss(
        magnitude_baseline, magnitude_labels
    )
    magnitude_rps = _ranked_probability_score(
        magnitude_probability, magnitude_labels
    )
    magnitude_baseline_rps = _ranked_probability_score(
        magnitude_baseline, magnitude_labels
    )
    baseline_mean = float(model.training_target_mean_bps[side])
    mse = float(np.mean(np.square(expected - actual)))
    baseline_mse = float(np.mean(np.square(baseline_mean - actual)))
    probability_loss = _binary_log_loss(probability, profitable)
    baseline_probability = np.full(
        len(actual), model.training_profitable_prevalence[side], dtype=np.float64
    )
    baseline_probability_loss = _binary_log_loss(
        baseline_probability, profitable
    )
    return {
        "role_rows": len(selected),
        "executable_rows": len(actual),
        "rejected_rows": int(len(selected) - len(actual)),
        "executable_ratio": float(np.mean(executable)),
        "actual_mean_net_bps": float(np.mean(actual)),
        "predicted_mean_net_bps": float(np.mean(expected)),
        "expected_payoff_mse_bps2": mse,
        "training_mean_baseline_mse_bps2": baseline_mse,
        "expected_payoff_mse_skill": 1.0 - mse / max(baseline_mse, 1e-15),
        "expected_payoff_spearman": finite_spearman(actual, expected),
        "profitable_event_rate": float(np.mean(profitable)),
        "predicted_profitable_probability_mean": float(np.mean(probability)),
        "probability_log_loss": probability_loss,
        "training_prevalence_log_loss": baseline_probability_loss,
        "probability_log_loss_skill": 1.0
        - probability_loss / max(baseline_probability_loss, 1e-15),
        "magnitude_log_loss": magnitude_loss,
        "training_magnitude_log_loss": magnitude_baseline_loss,
        "magnitude_log_loss_skill": 1.0
        - magnitude_loss / max(magnitude_baseline_loss, 1e-15),
        "magnitude_ranked_probability_score": magnitude_rps,
        "training_magnitude_ranked_probability_score": magnitude_baseline_rps,
        "magnitude_ranked_probability_skill": 1.0
        - magnitude_rps / max(magnitude_baseline_rps, 1e-15),
        "joint_log_loss": joint_loss,
        "training_joint_log_loss": joint_baseline_loss,
        "joint_log_loss_skill": 1.0
        - joint_loss / max(joint_baseline_loss, 1e-15),
        "prediction_sha256": _array_sha256(expected),
        "actual_sha256": _array_sha256(actual),
        "magnitude_probability_sha256": _array_sha256(magnitude_probability),
        "joint_probability_sha256": _array_sha256(joint_probability),
    }


@dataclass(frozen=True)
class _ExpectedPrediction:
    endpoint_indexes: np.ndarray
    long_expected_net_bps: np.ndarray
    short_expected_net_bps: np.ndarray
    long_executable: np.ndarray
    short_executable: np.ndarray


@dataclass(frozen=True)
class _EnsembleScore:
    endpoint_indexes: np.ndarray
    side: np.ndarray
    strength_bps: np.ndarray
    eligible: np.ndarray


def _expected_prediction(prediction) -> _ExpectedPrediction:
    return _ExpectedPrediction(
        endpoint_indexes=np.asarray(prediction.endpoint_indexes, dtype=np.int64),
        long_expected_net_bps=np.asarray(
            prediction.long_expected_net_bps, dtype=np.float64
        ),
        short_expected_net_bps=np.asarray(
            prediction.short_expected_net_bps, dtype=np.float64
        ),
        long_executable=np.asarray(prediction.long_executable, dtype=np.bool_),
        short_executable=np.asarray(prediction.short_executable, dtype=np.bool_),
    )


def _slice_prediction(
    prediction: _ExpectedPrediction,
    selected: np.ndarray,
) -> _ExpectedPrediction:
    mask = np.asarray(selected, dtype=np.bool_)
    return _ExpectedPrediction(
        endpoint_indexes=prediction.endpoint_indexes[mask],
        long_expected_net_bps=prediction.long_expected_net_bps[mask],
        short_expected_net_bps=prediction.short_expected_net_bps[mask],
        long_executable=prediction.long_executable[mask],
        short_executable=prediction.short_executable[mask],
    )


def _ensemble_score(predictions: Sequence[_ExpectedPrediction]) -> _EnsembleScore:
    if len(predictions) != len(SEEDS):
        raise ValueError("Round 53 ensemble member count drifted")
    endpoints = np.asarray(predictions[0].endpoint_indexes, dtype=np.int64)
    if any(
        not np.array_equal(prediction.endpoint_indexes, endpoints)
        or not np.array_equal(
            prediction.long_executable, predictions[0].long_executable
        )
        or not np.array_equal(
            prediction.short_executable, predictions[0].short_executable
        )
        for prediction in predictions[1:]
    ):
        raise ValueError("Round 53 ensemble member contracts differ")
    long_stack = np.stack(
        [prediction.long_expected_net_bps for prediction in predictions]
    )
    short_stack = np.stack(
        [prediction.short_expected_net_bps for prediction in predictions]
    )
    negative_infinity = np.full(len(endpoints), -np.inf, dtype=np.float64)
    long_rank = np.where(
        predictions[0].long_executable, np.min(long_stack, axis=0), negative_infinity
    )
    short_rank = np.where(
        predictions[0].short_executable,
        np.min(short_stack, axis=0),
        negative_infinity,
    )
    choose_long = long_rank > short_rank
    choose_short = short_rank > long_rank
    side = np.zeros(len(endpoints), dtype=np.int8)
    side[choose_long] = 1
    side[choose_short] = -1
    strength = np.where(
        choose_long, long_rank, np.where(choose_short, short_rank, 0.0)
    )
    eligible = (side != 0) & (strength > 0.0)
    return _EnsembleScore(
        endpoint_indexes=endpoints,
        side=side,
        strength_bps=strength,
        eligible=eligible,
    )


def _threshold(score: _EnsembleScore, coverage: float) -> float | None:
    supported = (score.side != 0) & np.isfinite(score.strength_bps)
    values = np.asarray(score.strength_bps[supported], dtype=np.float64)
    if len(values) < 32:
        return None
    threshold = float(np.quantile(values, 1.0 - float(coverage), method="higher"))
    return max(float(np.nextafter(0.0, 1.0)), threshold)


def _action_score(
    ensemble: _EnsembleScore,
    threshold: float | None,
) -> ActionScoreBatch:
    selected = (
        np.zeros(len(ensemble.endpoint_indexes), dtype=np.bool_)
        if threshold is None
        else ensemble.eligible & (ensemble.strength_bps >= float(threshold))
    )
    side = np.where(selected, ensemble.side, 0).astype(np.int8)
    strength = np.where(selected, ensemble.strength_bps, 0.0).astype(np.float64)
    return ActionScoreBatch(
        endpoint_indexes=ensemble.endpoint_indexes,
        side=side,
        strength_bps=strength,
        eligible=side != 0,
        profile="conservative",
    )


def _trace_summary(trace: BarrierActionTrace) -> dict[str, object]:
    return {
        "metrics": trace.asdict()["metrics"],
        "source_endpoint_indexes_sha256": _array_sha256(
            np.asarray(trace.source_endpoint_indexes, dtype=np.int64)
        ),
        "net_bps_sha256": _array_sha256(np.asarray(trace.net_bps, dtype=np.float64)),
    }


def _policy_result(
    *,
    symbol_state: Mapping[str, Mapping[str, object]],
    candidate: str,
    role: str,
    thresholds: Mapping[str, float | None],
) -> dict[str, object]:
    base_traces: dict[str, BarrierActionTrace] = {}
    stress_traces: dict[str, BarrierActionTrace] = {}
    symbols: dict[str, object] = {}
    overlap_violations = 0
    for symbol in SYMBOLS:
        state = symbol_state[symbol]
        predictions = state["predictions"][candidate][role]
        ensemble = _ensemble_score(predictions)
        score = _action_score(ensemble, thresholds[symbol])
        base, stress, overlap = base_and_paired_stress_traces(
            state["dataset"],
            state["targets"],
            score,
            extra_stress_slippage_bps_per_side=2.0,
        )
        base_traces[symbol] = base
        stress_traces[symbol] = stress
        overlap_violations += overlap
        symbols[symbol] = {
            "threshold_bps": thresholds[symbol],
            "eligible_rows_before_threshold": int(np.sum(ensemble.eligible)),
            "selected_rows_before_non_overlap": int(np.sum(score.eligible)),
            "base": _trace_summary(base),
            "paired_stress": _trace_summary(stress),
            "paired_stress_overlap_violations": overlap,
        }
    return {
        "candidate": candidate,
        "role": role,
        "thresholds_bps": dict(thresholds),
        "symbols": symbols,
        "base": portfolio_trace_metrics(base_traces, symbol_weight=1.0 / len(SYMBOLS)),
        "paired_stress": portfolio_trace_metrics(
            stress_traces, symbol_weight=1.0 / len(SYMBOLS)
        ),
        "paired_stress_overlap_violations": overlap_violations,
    }


def _scenario_gate_reasons(
    result: Mapping[str, object],
    *,
    minimum_trades: int,
    maximum_drawdown_bps: float | None,
    require_symbol_breadth: bool,
) -> list[str]:
    reasons: list[str] = []
    for scenario in ("base", "paired_stress"):
        scenario_result = result[scenario]
        metrics = scenario_result["metrics"]
        if int(metrics["trades"]) < int(minimum_trades):
            reasons.append(f"{scenario}_trades_below_{minimum_trades}")
        if float(metrics["total_net_bps"]) <= 0.0:
            reasons.append(f"{scenario}_total_net_bps_not_positive")
        profit_factor = metrics.get("profit_factor")
        if profit_factor is None or float(profit_factor) <= 1.0:
            reasons.append(f"{scenario}_profit_factor_not_above_one")
        if require_symbol_breadth:
            symbol_pnl = scenario_result["symbol_net_bps"]
            if sum(float(value) > 0.0 for value in symbol_pnl.values()) < 2:
                reasons.append(f"{scenario}_positive_symbols_below_two")
            if float(
                scenario_result["maximum_single_symbol_positive_pnl_share"]
            ) > 0.7:
                reasons.append(f"{scenario}_positive_pnl_concentration_above_0.70")
        if maximum_drawdown_bps is not None and float(
            metrics["max_drawdown_bps"]
        ) > float(maximum_drawdown_bps):
            reasons.append(f"{scenario}_drawdown_above_{maximum_drawdown_bps:g}_bps")
    if int(result["paired_stress_overlap_violations"]) != 0:
        reasons.append("paired_stress_overlap_violations_nonzero")
    return reasons


def _calibration_gate_reasons(
    aggregate: Mapping[str, object],
    days: Sequence[Mapping[str, object]],
) -> list[str]:
    reasons = _scenario_gate_reasons(
        aggregate,
        minimum_trades=8,
        maximum_drawdown_bps=None,
        require_symbol_breadth=True,
    )
    for index, day in enumerate(days, start=1):
        day_reasons = _scenario_gate_reasons(
            day,
            minimum_trades=3,
            maximum_drawdown_bps=None,
            require_symbol_breadth=False,
        )
        reasons.extend(f"day_{index}_{reason}" for reason in day_reasons)
    return sorted(set(reasons))


def _predictive_gate(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    candidate: str,
) -> dict[str, object]:
    reasons: list[str] = []
    expected_by_symbol_side: dict[tuple[str, str], list[np.ndarray]] = {}
    for symbol in SYMBOLS:
        for seed in SEEDS:
            record = metrics[symbol][candidate][str(seed)]
            for side in SIDES:
                values = record["evaluation"][side]
                if float(values["expected_payoff_mse_skill"]) <= 0.0:
                    reasons.append(f"{symbol}_{seed}_{side}_mse_skill_not_positive")
                if float(values["expected_payoff_spearman"]) < 0.03:
                    reasons.append(f"{symbol}_{seed}_{side}_spearman_below_0.03")
                if candidate in CSM_CANDIDATES:
                    if float(values["joint_log_loss_skill"]) <= 0.0:
                        reasons.append(
                            f"{symbol}_{seed}_{side}_joint_log_loss_skill_not_positive"
                        )
                    if float(values["magnitude_ranked_probability_skill"]) <= 0.0:
                        reasons.append(
                            f"{symbol}_{seed}_{side}_magnitude_rps_skill_not_positive"
                        )
                expected = np.asarray(
                    record["prediction_arrays"]["evaluation"][
                        f"{side}_expected_net_bps"
                    ],
                    dtype=np.float64,
                )
                executable = np.asarray(
                    record["prediction_arrays"]["evaluation"][
                        f"{side}_executable"
                    ],
                    dtype=np.bool_,
                )
                expected_by_symbol_side.setdefault((symbol, side), []).append(
                    expected[executable]
                )
    seed_stability: dict[str, object] = {}
    for (symbol, side), arrays in expected_by_symbol_side.items():
        stability = pairwise_seed_spearman(arrays)
        seed_stability[f"{symbol}_{side}"] = stability
        if float(stability["minimum_spearman"]) < 0.50:
            reasons.append(f"{symbol}_{side}_seed_spearman_below_0.50")
    return {
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "seed_stability": seed_stability,
    }


def _mean_metric(
    metrics: Mapping[str, Mapping[str, object]],
    candidate: str,
    name: str,
) -> float:
    return float(
        np.mean(
            [
                float(metrics[symbol][candidate][str(seed)]["evaluation"][side][name])
                for symbol in SYMBOLS
                for seed in SEEDS
                for side in SIDES
            ]
        )
    )


def run_round53(
    *,
    design_path: Path,
    binding_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    round52_report_path: Path,
    evidence_root: Path,
    compute_backend: str,
    memory_limit: str,
    threads: int,
) -> dict[str, object]:
    started = time.perf_counter()
    design, design_sha = _validate_design(design_path)
    binding, binding_sha = _validate_binding(
        binding_path, design_sha256=design_sha
    )
    round52 = _validate_round52_report(round52_report_path)
    round51_path = Path(str(round52["source_round_51"]["report_path"]))
    round51 = _read_object(round51_path, "Round 51 source report")
    specifications = _specifications(design)
    evidence_root.mkdir(parents=True, exist_ok=True)
    _progress("start", design=design_sha, binding=binding_sha)

    data_contract = design["data_contract"]
    execution = design["execution_target"]
    source_states: dict[str, dict[str, object]] = {}
    model_paths: dict[str, dict[str, dict[int, Path]]] = {}
    model_evidence: dict[str, object] = {}
    data_evidence: dict[str, object] = {}

    for symbol in SYMBOLS:
        _progress("symbol-load", symbol=symbol)
        source = _load_real_symbol_data(
            symbol=symbol,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            memory_limit=memory_limit,
            threads=threads,
            data_contract=data_contract,
            execution=execution,
        )
        dataset = source["dataset"]
        targets = source["targets"]
        fincast_names, fincast_matrix, fincast_evidence = _load_fincast_features(
            round51,
            symbol=symbol,
            expected_file_sha256=str(
                data_contract["fincast_feature_artifacts"][symbol]["file_sha256"]
            ),
            expected_bytes=int(
                data_contract["fincast_feature_artifacts"][symbol]["bytes"]
            ),
            dataset_rows=dataset.rows,
            decision_time_ms=dataset.decision_time_ms,
        )
        deterministic = build_executable_payoff_dataset(
            dataset, targets, target_scenario="base"
        )
        ai = build_executable_payoff_dataset(
            dataset,
            targets,
            target_scenario="base",
            extra_feature_names=fincast_names,
            extra_features=fincast_matrix,
        )
        roles = _role_indexes(deterministic.payoff.decision_time_ms)
        if (
            not np.array_equal(
                deterministic.payoff.decision_time_ms, ai.payoff.decision_time_ms
            )
            or not np.array_equal(deterministic.long_executable, ai.long_executable)
            or not np.array_equal(deterministic.short_executable, ai.short_executable)
        ):
            raise ValueError(f"Round 53 {symbol} AI dataset changed row support")
        source_states[symbol] = {
            "dataset": dataset,
            "targets": targets,
            "datasets": {
                CANDIDATES[0]: deterministic,
                CANDIDATES[1]: deterministic,
                CANDIDATES[2]: ai,
            },
            "roles": roles,
            "predictions": {candidate: {} for candidate in CANDIDATES},
        }
        data_evidence[symbol] = {
            "microstructure_rows": dataset.rows,
            "deterministic_rows": deterministic.rows,
            "deterministic_dataset_sha256": deterministic.dataset_sha256,
            "ai_dataset_sha256": ai.dataset_sha256,
            "deterministic_feature_count": len(deterministic.payoff.feature_names),
            "ai_feature_count": len(ai.payoff.feature_names),
            "synthetic_rows": 0,
            "roles": {name: len(indexes) for name, indexes in roles.items()},
            "support": {
                "long_executable_rows": int(np.sum(deterministic.long_executable)),
                "short_executable_rows": int(np.sum(deterministic.short_executable)),
                "long_executable_ratio": float(np.mean(deterministic.long_executable)),
                "short_executable_ratio": float(
                    np.mean(deterministic.short_executable)
                ),
                "long_mask_sha256": _array_sha256(deterministic.long_executable),
                "short_mask_sha256": _array_sha256(deterministic.short_executable),
            },
            "fincast": fincast_evidence,
            "source_evidence": source["source_evidence"],
        }
        model_paths[symbol] = {}
        model_evidence[symbol] = {}
        for candidate in CSM_CANDIDATES:
            candidate_dataset = source_states[symbol]["datasets"][candidate]
            model_paths[symbol][candidate] = {}
            model_evidence[symbol][candidate] = {}
            for seed in SEEDS:
                path = evidence_root / "models" / symbol / candidate / f"seed-{seed}.json"
                state = "loaded"
                if path.is_file():
                    model = load_executable_csm_model(path)
                    if (
                        model.spec != specifications[candidate]
                        or model.source_dataset_sha256 != candidate_dataset.dataset_sha256
                        or model.seed != seed
                        or model.backend_requested != compute_backend
                    ):
                        raise ValueError(
                            f"Round 53 cached model drifted: {symbol} {candidate} {seed}"
                        )
                else:
                    state = "trained"
                    _progress("train", symbol=symbol, candidate=candidate, seed=seed)
                    model = train_executable_csm_model(
                        candidate_dataset,
                        train_indexes=roles["train"],
                        early_stop_indexes=roles["early_stop"],
                        probability_calibration_indexes=roles[
                            "probability_calibration"
                        ],
                        probability_calibration_end_exclusive_ms=int(
                            candidate_dataset.payoff.decision_time_ms[
                                roles["policy_calibration"][0]
                            ]
                        ),
                        spec=specifications[candidate],
                        target_scenario="base",
                        compute_backend=compute_backend,
                        seed=seed,
                        progress=lambda name, side, step, total, s=symbol, c=candidate, d=seed: _progress(
                            "head",
                            symbol=s,
                            candidate=c,
                            seed=d,
                            head=name,
                            side=side,
                            step=f"{step}/{total}",
                        ),
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_executable_csm_model(path, model)
                if compute_backend == "directml" and model.backend_kind != "opencl":
                    raise RuntimeError("Round 53 LightGBM silently left OpenCL")
                model_paths[symbol][candidate][seed] = path
                model_evidence[symbol][candidate][str(seed)] = {
                    "cache_state": state,
                    "artifact": _artifact(path),
                    "model_sha256": model.model_sha256,
                    "backend_kind": model.backend_kind,
                    "backend_device": model.backend_device,
                    "role_rows": model.role_rows,
                    "rejected_role_rows": model.rejected_role_rows,
                    "role_mask_sha256": model.role_mask_sha256,
                    "magnitude_class_support": model.magnitude_class_support,
                    "sign_class_support": model.sign_class_support,
                    "minimum_leaf_rows": model.minimum_leaf_rows,
                    "magnitude_edges_risk_units": model.magnitude_edges_risk_units,
                    "magnitude_representatives_risk_units": (
                        model.magnitude_representatives_risk_units
                    ),
                    "magnitude_temperature": model.magnitude_temperature,
                    "sign_probability_calibration": (
                        model.sign_probability_calibration
                    ),
                    "best_iterations": model.best_iterations,
                }
                del model
        del fincast_matrix
        gc.collect()
        _progress("symbol-models-complete", symbol=symbol)

    _progress("all-models-trained", models=len(SYMBOLS) * len(CSM_CANDIDATES) * len(SEEDS))

    metrics: dict[str, dict[str, object]] = {}
    prediction_evidence: dict[str, object] = {}
    for symbol in SYMBOLS:
        state = source_states[symbol]
        roles = state["roles"]
        metrics[symbol] = {candidate: {} for candidate in CANDIDATES}
        prediction_evidence[symbol] = {candidate: {} for candidate in CANDIDATES}
        for seed in SEEDS:
            direct_model_path = Path(
                str(
                    round52["models"][symbol][CANDIDATES[0]][str(seed)]["artifact"][
                        "path"
                    ]
                )
            )
            direct_model = load_executable_payoff_model(direct_model_path)
            direct_record: dict[str, object] = {"prediction_arrays": {}}
            prediction_evidence[symbol][CANDIDATES[0]][str(seed)] = {}
            for role in POLICY_ROLES:
                source_artifact = round52["prediction_artifacts"][symbol][
                    CANDIDATES[0]
                ][str(seed)][role]
                path = Path(str(source_artifact["path"]))
                if _file_sha256(path) != str(source_artifact["sha256"]):
                    raise ValueError("Round 53 direct-control prediction drifted")
                prediction = _prediction_from_npz(path, architecture="direct_mean")
                state["predictions"][CANDIDATES[0]].setdefault(role, []).append(
                    _expected_prediction(prediction)
                )
                selected = np.asarray(roles[role], dtype=np.int64)
                direct_record[role] = {}
                for side in SIDES:
                    executable = np.asarray(
                        prediction.long_executable
                        if side == "long"
                        else prediction.short_executable,
                        dtype=np.bool_,
                    )
                    expected = np.asarray(
                        prediction.long_expected_net_bps
                        if side == "long"
                        else prediction.short_expected_net_bps,
                        dtype=np.float64,
                    )[executable]
                    actual_all = np.asarray(
                        state["datasets"][CANDIDATES[0]].payoff.long_net_bps
                        if side == "long"
                        else state["datasets"][CANDIDATES[0]].payoff.short_net_bps,
                        dtype=np.float64,
                    )[selected]
                    actual = actual_all[executable]
                    baseline = float(direct_model.training_target_mean_bps[side])
                    mse = float(np.mean(np.square(expected - actual)))
                    baseline_mse = float(np.mean(np.square(baseline - actual)))
                    direct_record[role][side] = {
                        "role_rows": len(selected),
                        "executable_rows": len(actual),
                        "rejected_rows": int(len(selected) - len(actual)),
                        "executable_ratio": float(np.mean(executable)),
                        "actual_mean_net_bps": float(np.mean(actual)),
                        "predicted_mean_net_bps": float(np.mean(expected)),
                        "expected_payoff_mse_bps2": mse,
                        "training_mean_baseline_mse_bps2": baseline_mse,
                        "expected_payoff_mse_skill": 1.0
                        - mse / max(baseline_mse, 1e-15),
                        "expected_payoff_spearman": finite_spearman(actual, expected),
                        "prediction_sha256": _array_sha256(expected),
                        "actual_sha256": _array_sha256(actual),
                    }
                direct_record["prediction_arrays"][role] = {
                    "long_expected_net_bps": prediction.long_expected_net_bps,
                    "short_expected_net_bps": prediction.short_expected_net_bps,
                    "long_executable": prediction.long_executable,
                    "short_executable": prediction.short_executable,
                }
                prediction_evidence[symbol][CANDIDATES[0]][str(seed)][role] = {
                    **dict(source_artifact),
                    "reused_from_round": 52,
                }
            metrics[symbol][CANDIDATES[0]][str(seed)] = direct_record
            model_evidence[symbol].setdefault(CANDIDATES[0], {})[str(seed)] = {
                "reused_from_round": 52,
                "artifact": _artifact(direct_model_path),
                "model_sha256": direct_model.model_sha256,
                "backend_kind": direct_model.backend_kind,
                "backend_device": direct_model.backend_device,
            }
            del direct_model

        for candidate in CSM_CANDIDATES:
            candidate_dataset = state["datasets"][candidate]
            for seed in SEEDS:
                model = load_executable_csm_model(model_paths[symbol][candidate][seed])
                record: dict[str, object] = {"prediction_arrays": {}}
                prediction_evidence[symbol][candidate][str(seed)] = {}
                for role in POLICY_ROLES:
                    prediction = predict_executable_csm_model(
                        model, candidate_dataset, roles[role]
                    )
                    state["predictions"][candidate].setdefault(role, []).append(
                        _expected_prediction(prediction)
                    )
                    path = (
                        evidence_root
                        / "predictions"
                        / symbol
                        / candidate
                        / f"seed-{seed}-{role}.npz"
                    )
                    prediction_evidence[symbol][candidate][str(seed)][role] = (
                        _save_csm_prediction(path, prediction)
                    )
                    record[role] = {
                        side: _csm_prediction_metrics(
                            model=model,
                            dataset=candidate_dataset,
                            indexes=roles[role],
                            prediction=prediction,
                            side=side,
                        )
                        for side in SIDES
                    }
                    record["prediction_arrays"][role] = {
                        "long_expected_net_bps": prediction.long_expected_net_bps,
                        "short_expected_net_bps": prediction.short_expected_net_bps,
                        "long_executable": prediction.long_executable,
                        "short_executable": prediction.short_executable,
                    }
                metrics[symbol][candidate][str(seed)] = record
                del model
        _progress("symbol-predictions-complete", symbol=symbol)

    for symbol in SYMBOLS:
        roles = source_states[symbol]["roles"]
        times = np.asarray(
            source_states[symbol]["datasets"][CANDIDATES[0]].payoff.decision_time_ms[
                roles["policy_calibration"]
            ],
            dtype=np.int64,
        )
        day_numbers = times // DAY_MS
        unique_days = np.unique(day_numbers)
        if len(unique_days) != 2:
            raise ValueError(f"Round 53 {symbol} policy calibration day count drifted")
        for candidate in CANDIDATES:
            full = source_states[symbol]["predictions"][candidate][
                "policy_calibration"
            ]
            for day_index, day in enumerate(unique_days, start=1):
                role = f"policy_calibration_day_{day_index}"
                mask = day_numbers == day
                source_states[symbol]["predictions"][candidate][role] = [
                    _slice_prediction(prediction, mask) for prediction in full
                ]

    coverages = tuple(
        float(value) for value in design["economic_screen"]["threshold_coverages"]
    )
    policy_grid: dict[str, object] = {}
    selected_policy: dict[str, object] = {}
    for candidate in CANDIDATES:
        rows: list[dict[str, object]] = []
        for coverage in coverages:
            thresholds = {
                symbol: _threshold(
                    _ensemble_score(
                        source_states[symbol]["predictions"][candidate][
                            "policy_calibration"
                        ]
                    ),
                    coverage,
                )
                for symbol in SYMBOLS
            }
            aggregate = _policy_result(
                symbol_state=source_states,
                candidate=candidate,
                role="policy_calibration",
                thresholds=thresholds,
            )
            days = [
                _policy_result(
                    symbol_state=source_states,
                    candidate=candidate,
                    role=f"policy_calibration_day_{index}",
                    thresholds=thresholds,
                )
                for index in (1, 2)
            ]
            calibration_reasons = _calibration_gate_reasons(aggregate, days)
            evaluation = _policy_result(
                symbol_state=source_states,
                candidate=candidate,
                role="evaluation",
                thresholds=thresholds,
            )
            evaluation_reasons = _scenario_gate_reasons(
                evaluation,
                minimum_trades=30,
                maximum_drawdown_bps=120.0,
                require_symbol_breadth=True,
            )
            rows.append(
                {
                    "coverage": coverage,
                    "thresholds_bps": thresholds,
                    "policy_calibration": aggregate,
                    "policy_calibration_days": days,
                    "policy_calibration_gate": {
                        "passed": not calibration_reasons,
                        "reasons": calibration_reasons,
                    },
                    "consumed_evaluation_diagnostic": evaluation,
                    "consumed_evaluation_gate": {
                        "passed": not evaluation_reasons,
                        "reasons": evaluation_reasons,
                    },
                }
            )
        passing = [row for row in rows if row["policy_calibration_gate"]["passed"]]
        passing.sort(
            key=lambda row: (
                -float(
                    row["policy_calibration"]["paired_stress"]["metrics"][
                        "mean_net_bps"
                    ]
                ),
                float(
                    row["policy_calibration"]["paired_stress"]["metrics"][
                        "max_drawdown_bps"
                    ]
                ),
                float(row["coverage"]),
            )
        )
        selected = passing[0] if passing else None
        selected_policy[candidate] = {
            "selected_coverage": selected["coverage"] if selected else None,
            "selection_passed": selected is not None,
            "evaluation_gate_passed": (
                bool(selected["consumed_evaluation_gate"]["passed"])
                if selected
                else False
            ),
            "policy_calibration": selected["policy_calibration"] if selected else None,
            "policy_calibration_days": (
                selected["policy_calibration_days"] if selected else None
            ),
            "consumed_evaluation": (
                selected["consumed_evaluation_diagnostic"] if selected else None
            ),
            "evaluation_reasons": (
                selected["consumed_evaluation_gate"]["reasons"]
                if selected
                else ["no_temporally_stable_policy_calibration_variant_passed"]
            ),
        }
        policy_grid[candidate] = rows

    predictive_gates = {
        candidate: _predictive_gate(metrics, candidate=candidate)
        for candidate in CANDIDATES
    }
    control_joint = _mean_metric(metrics, CANDIDATES[1], "joint_log_loss")
    ai_joint = _mean_metric(metrics, CANDIDATES[2], "joint_log_loss")
    control_spearman = _mean_metric(
        metrics, CANDIDATES[1], "expected_payoff_spearman"
    )
    ai_spearman = _mean_metric(metrics, CANDIDATES[2], "expected_payoff_spearman")
    ai_reasons: list[str] = []
    if control_joint - ai_joint < 0.005:
        ai_reasons.append("average_joint_log_loss_improvement_below_0.005")
    if ai_spearman - control_spearman < 0.005:
        ai_reasons.append("average_expected_payoff_spearman_improvement_below_0.005")
    if not predictive_gates[CANDIDATES[1]]["passed"]:
        ai_reasons.append("deterministic_csm_predictive_gate_failed")
    if not selected_policy[CANDIDATES[1]]["evaluation_gate_passed"]:
        ai_reasons.append("deterministic_csm_economic_gate_failed")
    if not selected_policy[CANDIDATES[2]]["evaluation_gate_passed"]:
        ai_reasons.append("ai_csm_economic_gate_failed")
    control_evaluation = selected_policy[CANDIDATES[1]]["consumed_evaluation"]
    ai_evaluation = selected_policy[CANDIDATES[2]]["consumed_evaluation"]
    if isinstance(control_evaluation, Mapping) and isinstance(ai_evaluation, Mapping):
        for scenario in ("base", "paired_stress"):
            control_scenario = control_evaluation[scenario]
            ai_scenario = ai_evaluation[scenario]
            control_values = control_scenario["metrics"]
            ai_values = ai_scenario["metrics"]
            if float(ai_values["mean_net_bps"]) <= float(
                control_values["mean_net_bps"]
            ):
                ai_reasons.append(f"{scenario}_mean_net_bps_not_improved")
            if float(ai_values["max_drawdown_bps"]) > float(
                control_values["max_drawdown_bps"]
            ):
                ai_reasons.append(f"{scenario}_drawdown_worsened")
            if int(ai_values["trades"]) < int(control_values["trades"]):
                ai_reasons.append(f"{scenario}_trade_count_worsened")
            if float(ai_scenario["maximum_single_symbol_positive_pnl_share"]) > float(
                control_scenario["maximum_single_symbol_positive_pnl_share"]
            ):
                ai_reasons.append(f"{scenario}_concentration_worsened")
        control_days = selected_policy[CANDIDATES[1]]["policy_calibration_days"]
        ai_days = selected_policy[CANDIDATES[2]]["policy_calibration_days"]
        for index, (control_day, ai_day) in enumerate(
            zip(control_days, ai_days, strict=True), start=1
        ):
            for scenario in ("base", "paired_stress"):
                if float(ai_day[scenario]["metrics"]["mean_net_bps"]) <= float(
                    control_day[scenario]["metrics"]["mean_net_bps"]
                ):
                    ai_reasons.append(
                        f"day_{index}_{scenario}_mean_net_bps_not_improved"
                    )
    ai_uplift = {
        "passed": not ai_reasons,
        "reasons": sorted(set(ai_reasons)),
        "control_average_joint_log_loss": control_joint,
        "ai_average_joint_log_loss": ai_joint,
        "joint_log_loss_improvement": control_joint - ai_joint,
        "control_average_expected_payoff_spearman": control_spearman,
        "ai_average_expected_payoff_spearman": ai_spearman,
        "expected_payoff_spearman_improvement": ai_spearman - control_spearman,
    }
    mechanism_candidates = [
        candidate
        for candidate in CSM_CANDIDATES
        if predictive_gates[candidate]["passed"]
        and selected_policy[candidate]["evaluation_gate_passed"]
        and (candidate != CANDIDATES[2] or ai_uplift["passed"])
    ]

    report_metrics: dict[str, object] = {}
    for symbol in SYMBOLS:
        report_metrics[symbol] = {}
        for candidate in CANDIDATES:
            report_metrics[symbol][candidate] = {}
            for seed in SEEDS:
                record = dict(metrics[symbol][candidate][str(seed)])
                record.pop("prediction_arrays")
                report_metrics[symbol][candidate][str(seed)] = record

    round_reasons = ["selection_contaminated_consumed_development_interval"]
    if not mechanism_candidates:
        round_reasons.append("no_candidate_passed_predictive_temporal_and_economic_gates")
    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "round": ROUND,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "design_sha256": design_sha,
        "binding_sha256": binding_sha,
        "implementation_commit": binding["implementation_commit"],
        "claims": {
            "selection_contaminated": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
        },
        "runtime_resources": {
            "warehouse": str(warehouse_path.resolve()),
            "cache_root": str(cache_root.resolve()),
            "compute_backend_requested": compute_backend,
            "duckdb_memory_limit": memory_limit,
            "duckdb_threads": threads,
            "all_models_trained_before_evaluation": True,
            "round52_direct_control_reused": True,
            "round51_fincast_features_reused": True,
        },
        "source_round_52": {
            "report_path": str(round52_report_path.resolve()),
            "report_canonical_sha256": EXPECTED_ROUND52_REPORT_CANONICAL_SHA256,
            "report_file_sha256": EXPECTED_ROUND52_REPORT_FILE_SHA256,
        },
        "data": data_evidence,
        "models": model_evidence,
        "prediction_artifacts": prediction_evidence,
        "predictive_metrics": report_metrics,
        "predictive_gates": predictive_gates,
        "policy_grid": policy_grid,
        "selected_policy": selected_policy,
        "ai_uplift_gate": ai_uplift,
        "mechanism_screen": {
            "passed_candidates": mechanism_candidates,
            "untouched_data_expansion_authorized": bool(mechanism_candidates),
            "trading_or_promotion_authorized": False,
        },
        "round_gate": {
            "passed": False,
            "reasons": round_reasons,
        },
    }
    report["report_canonical_sha256"] = _canonical_sha256(report)
    report_path = evidence_root / "report.json"
    write_json_atomic(report_path, report, indent=2, sort_keys=True)
    _progress(
        "complete",
        seconds=f"{report['runtime_seconds']:.1f}",
        mechanisms=len(mechanism_candidates),
        report=report_path,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-053-executable-csm-fincast-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-053-executable-csm-fincast-binding.json",
    )
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--round52-report", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--compute-backend", choices=SUPPORTED_COMPUTE_BACKENDS, default="auto"
    )
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    run_round53(
        design_path=arguments.design,
        binding_path=arguments.binding,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        round52_report_path=arguments.round52_report,
        evidence_root=arguments.evidence_root,
        compute_backend=arguments.compute_backend,
        memory_limit=arguments.memory_limit,
        threads=arguments.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
