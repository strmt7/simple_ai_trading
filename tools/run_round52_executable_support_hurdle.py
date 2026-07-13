"""Run the frozen Round 52 executable-support payoff mechanism screen."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, time as datetime_time
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

from simple_ai_trading.executable_payoff_lightgbm import (  # noqa: E402
    ExecutablePayoffDataset,
    ExecutablePayoffPredictionBatch,
    ExecutablePayoffSpec,
    TrainedExecutablePayoffModel,
    build_executable_payoff_dataset,
    load_executable_payoff_model,
    predict_executable_payoff_model,
    save_executable_payoff_model,
    train_executable_payoff_model,
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


ROUND = 52
SCHEMA = "round-052-executable-support-hurdle-fincast-report-v1"
BINDING_SCHEMA = "round-052-executable-support-hurdle-execution-binding-v1"
EXPECTED_DESIGN_SHA256 = (
    "af95d80a3adc21b72d6809d43afb3f2446213fe0a4e089b10366691465a0c669"
)
EXPECTED_ROUND51_REPORT_CANONICAL_SHA256 = (
    "b97a12764256680402d526fd17ee56999c7f88335d66570a196aae3d0e9d0201"
)
EXPECTED_ROUND51_REPORT_FILE_SHA256 = (
    "d2e6c2e1a8ba0a48293f124d148359e4015f9c25aa61fb11b9dc7578d7975a80"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SEEDS = (5201, 5202, 5203)
CANDIDATES = (
    "executable_direct_mean_lightgbm",
    "executable_hurdle_lightgbm",
    "executable_hurdle_lightgbm_fincast",
)
HURDLE_CANDIDATES = CANDIDATES[1:]
SIDES = ("long", "short")
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
    print(f"round52 {stage}{(' ' + detail) if detail else ''}", flush=True)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _validate_design(path: Path) -> tuple[dict[str, object], str]:
    design = _read_object(path, "Round 52 design")
    canonical = dict(design)
    claimed = canonical.pop("design_sha256", None)
    actual = _canonical_sha256(canonical)
    if (
        design.get("round") != ROUND
        or design.get("status") != "frozen"
        or design.get("schema_version")
        != "executable-support-hurdle-fincast-screen-design-v1"
        or claimed != actual
        or actual != EXPECTED_DESIGN_SHA256
        or design.get("claims", {}).get("selection_contaminated") is not True
        or design.get("claims", {}).get("profitability_claim_permitted") is not False
        or design.get("economic_screen", {}).get("leverage") != 1.0
    ):
        raise ValueError("Round 52 design identity or fail-closed claims drifted")
    return design, actual


def _validate_binding(
    path: Path,
    *,
    design_sha256: str,
) -> tuple[dict[str, object], str]:
    binding = _read_object(path, "Round 52 binding")
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
        or contract.get("all_models_trained_before_evaluation") is not True
        or contract.get("support_alignment_required") is not True
        or contract.get("selection_contaminated") is not True
        or contract.get("profitability_claim_permitted") is not False
        or contract.get("trading_authority_permitted") is not False
        or contract.get("leverage_applied") is not False
        or _git("status", "--porcelain")
    ):
        raise ValueError("Round 52 binding or clean-worktree contract is invalid")
    implementation_commit = str(binding.get("implementation_commit") or "")
    blobs = binding.get("blobs")
    if not implementation_commit or not isinstance(blobs, list) or not blobs:
        raise ValueError("Round 52 binding implementation evidence is missing")
    for item in blobs:
        if not isinstance(item, dict):
            raise ValueError("Round 52 binding blob record is invalid")
        relative = str(item.get("path") or "")
        expected = str(item.get("git_blob_oid") or "")
        if (
            not relative
            or not expected
            or _git("rev-parse", f"{implementation_commit}:{relative}") != expected
            or _git("rev-parse", f"HEAD:{relative}") != expected
        ):
            raise ValueError(f"Round 52 bound implementation drifted: {relative}")
    return binding, actual


def _validate_round51_report(path: Path) -> dict[str, object]:
    if _file_sha256(path) != EXPECTED_ROUND51_REPORT_FILE_SHA256:
        raise ValueError("Round 51 source report file hash drifted")
    report = _read_object(path, "Round 51 source report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("round") != 51
        or claimed != EXPECTED_ROUND51_REPORT_CANONICAL_SHA256
        or _canonical_sha256(canonical) != claimed
        or report.get("round_gate", {}).get("passed") is not False
    ):
        raise ValueError("Round 51 source report identity drifted")
    return report


def _utc_bounds(first: str, last: str) -> tuple[int, int]:
    start = datetime.combine(date.fromisoformat(first), datetime_time(), tzinfo=UTC)
    end = datetime.combine(date.fromisoformat(last), datetime_time(), tzinfo=UTC)
    return int(start.timestamp() * 1_000), int(end.timestamp() * 1_000) + DAY_MS - 1


def _role_indexes(decision_time_ms: np.ndarray) -> dict[str, np.ndarray]:
    boundaries = {
        "train": ("2023-05-16", "2023-05-31"),
        "early_stop": ("2023-06-01", "2023-06-04"),
        "probability_calibration": ("2023-06-05", "2023-06-06"),
        "policy_calibration": ("2023-06-07", "2023-06-08"),
        "evaluation": ("2023-06-09", "2023-06-14"),
    }
    output: dict[str, np.ndarray] = {}
    times = np.asarray(decision_time_ms, dtype=np.int64)
    for role, (first, last) in boundaries.items():
        lower, upper = _utc_bounds(first, last)
        selected = np.flatnonzero((times >= lower) & (times <= upper)).astype(np.int64)
        if len(selected) < 256:
            raise ValueError(f"Round 52 {role} role has insufficient rows")
        output[role] = selected
    ordered = tuple(output[name] for name in boundaries)
    if any(left[-1] >= right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Round 52 chronological roles overlap")
    return output


def _specifications(design: Mapping[str, object]) -> dict[str, ExecutablePayoffSpec]:
    contract = design["model_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("Round 52 model contract is invalid")
    lightgbm = contract["lightgbm"]
    if not isinstance(lightgbm, Mapping):
        raise ValueError("Round 52 LightGBM contract is invalid")
    if lightgbm.get("minimum_leaf_rows") != (
        "max(64, min(512, ceil(0.002 * side-specific training rows)))"
    ):
        raise ValueError("Round 52 dynamic leaf-size contract drifted")
    common = {
        "family": "side_specific_executable_payoff",
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
        CANDIDATES[0]: ExecutablePayoffSpec(
            candidate_id=CANDIDATES[0],
            architecture="direct_mean",
            **common,
        ),
        CANDIDATES[1]: ExecutablePayoffSpec(
            candidate_id=CANDIDATES[1],
            architecture="sign_magnitude_hurdle",
            **common,
        ),
        CANDIDATES[2]: ExecutablePayoffSpec(
            candidate_id=CANDIDATES[2],
            architecture="sign_magnitude_hurdle",
            **common,
        ),
    }


def _load_fincast_features(
    report: Mapping[str, object],
    *,
    symbol: str,
    expected_file_sha256: str,
    expected_bytes: int,
    dataset_rows: int,
    decision_time_ms: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, object]]:
    data = report.get("data")
    if not isinstance(data, Mapping) or not isinstance(data.get(symbol), Mapping):
        raise ValueError(f"Round 51 {symbol} FinCast evidence is missing")
    symbol_data = data[symbol]
    artifact = symbol_data.get("fincast_feature_artifact")
    fincast = symbol_data.get("fincast")
    if not isinstance(artifact, Mapping) or not isinstance(fincast, Mapping):
        raise ValueError(f"Round 51 {symbol} FinCast artifact is invalid")
    path = Path(str(artifact.get("path") or ""))
    expected_sha = str(artifact.get("sha256") or "")
    if (
        not path.is_file()
        or path.stat().st_size != int(expected_bytes)
        or int(artifact.get("bytes", -1)) != int(expected_bytes)
        or expected_sha != expected_file_sha256
        or _file_sha256(path) != expected_sha
        or int(fincast.get("rows", -1)) != dataset_rows
        or str(fincast.get("decision_times_sha256") or "")
        != _array_sha256(np.asarray(decision_time_ms, dtype=np.int64))
    ):
        raise ValueError(f"Round 51 {symbol} cached FinCast identity drifted")
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    names = tuple(str(value) for value in fincast.get("feature_names", ()))
    if (
        matrix.shape != (dataset_rows, len(names))
        or len(names) != 30
        or matrix.dtype != np.float32
        or not np.all(np.isfinite(matrix))
        or _array_sha256(np.asarray(matrix))
        != str(fincast.get("features_sha256") or "")
    ):
        raise ValueError(f"Round 51 {symbol} cached FinCast matrix is invalid")
    return (
        names,
        matrix,
        {
            "source_round": 51,
            "artifact": _artifact(path),
            "features_sha256": str(fincast["features_sha256"]),
            "decision_times_sha256": str(fincast["decision_times_sha256"]),
            "feature_count": len(names),
            "rows": dataset_rows,
            "rerun": False,
        },
    )


def _save_prediction(
    path: Path,
    prediction: ExecutablePayoffPredictionBatch,
) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {
        "endpoint_indexes": prediction.endpoint_indexes,
        "long_expected_net_bps": prediction.long_expected_net_bps,
        "short_expected_net_bps": prediction.short_expected_net_bps,
        "long_executable": prediction.long_executable,
        "short_executable": prediction.short_executable,
    }
    optional = {
        "long_profitable_probability": prediction.long_profitable_probability,
        "short_profitable_probability": prediction.short_profitable_probability,
        "long_conditional_gain_bps": prediction.long_conditional_gain_bps,
        "short_conditional_gain_bps": prediction.short_conditional_gain_bps,
        "long_conditional_loss_bps": prediction.long_conditional_loss_bps,
        "short_conditional_loss_bps": prediction.short_conditional_loss_bps,
    }
    arrays.update(
        {
            name: np.asarray(value)
            for name, value in optional.items()
            if value is not None
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        **_artifact(path),
        "architecture": prediction.architecture,
        "rows": prediction.rows,
        "magnitude_floor_count": prediction.magnitude_floor_count,
    }


def _binary_log_loss(probability: np.ndarray, outcome: np.ndarray) -> float:
    predicted = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    actual = np.asarray(outcome, dtype=np.float64)
    return float(
        -np.mean(actual * np.log(predicted) + (1.0 - actual) * np.log1p(-predicted))
    )


def _prediction_metrics(
    *,
    model: TrainedExecutablePayoffModel,
    dataset: ExecutablePayoffDataset,
    indexes: np.ndarray,
    prediction: ExecutablePayoffPredictionBatch,
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
        dataset.payoff.long_net_bps if side == "long" else dataset.payoff.short_net_bps,
        dtype=np.float64,
    )[selected]
    actual = target_all[executable]
    if len(actual) < 32:
        raise ValueError(f"Round 52 {side} prediction metrics lack support")
    baseline_mean = float(model.training_target_mean_bps[side])
    mse = float(np.mean(np.square(expected - actual)))
    baseline_mse = float(np.mean(np.square(baseline_mean - actual)))
    result: dict[str, object] = {
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
        "prediction_sha256": _array_sha256(expected),
        "actual_sha256": _array_sha256(actual),
    }
    if model.spec.architecture == "direct_mean":
        return result
    probability = np.asarray(
        prediction.long_profitable_probability
        if side == "long"
        else prediction.short_profitable_probability,
        dtype=np.float64,
    )[executable]
    outcome = actual > 0.0
    prevalence = float(model.training_profitable_prevalence[side])
    baseline_probability = np.full(len(outcome), prevalence, dtype=np.float64)
    log_loss = _binary_log_loss(probability, outcome)
    baseline_log_loss = _binary_log_loss(baseline_probability, outcome)
    brier = float(np.mean(np.square(probability - outcome)))
    baseline_brier = float(np.mean(np.square(baseline_probability - outcome)))
    result.update(
        {
            "profitable_event_rate": float(np.mean(outcome)),
            "predicted_profitable_probability_mean": float(np.mean(probability)),
            "probability_log_loss": log_loss,
            "training_prevalence_log_loss": baseline_log_loss,
            "probability_log_loss_skill": 1.0
            - log_loss / max(baseline_log_loss, 1e-15),
            "probability_brier_score": brier,
            "training_prevalence_brier_score": baseline_brier,
            "probability_brier_skill": 1.0 - brier / max(baseline_brier, 1e-15),
        }
    )
    return result


@dataclass(frozen=True)
class _EnsembleScore:
    endpoint_indexes: np.ndarray
    side: np.ndarray
    strength_bps: np.ndarray
    eligible: np.ndarray
    long_executable: np.ndarray
    short_executable: np.ndarray


def _ensemble_score(
    predictions: Sequence[ExecutablePayoffPredictionBatch],
) -> _EnsembleScore:
    if len(predictions) != len(SEEDS):
        raise ValueError("Round 52 ensemble member count drifted")
    endpoints = np.asarray(predictions[0].endpoint_indexes, dtype=np.int64)
    architecture = predictions[0].architecture
    if any(
        prediction.architecture != architecture
        or not np.array_equal(prediction.endpoint_indexes, endpoints)
        or not np.array_equal(
            prediction.long_executable, predictions[0].long_executable
        )
        or not np.array_equal(
            prediction.short_executable, predictions[0].short_executable
        )
        for prediction in predictions[1:]
    ):
        raise ValueError("Round 52 ensemble member contracts differ")
    long_stack = np.stack(
        [prediction.long_expected_net_bps for prediction in predictions]
    ).astype(np.float64)
    short_stack = np.stack(
        [prediction.short_expected_net_bps for prediction in predictions]
    ).astype(np.float64)
    long_executable = np.asarray(predictions[0].long_executable, dtype=np.bool_)
    short_executable = np.asarray(predictions[0].short_executable, dtype=np.bool_)
    long_worst = np.min(long_stack, axis=0)
    short_worst = np.min(short_stack, axis=0)
    negative_infinity = np.full(len(endpoints), -np.inf, dtype=np.float64)
    long_rank = np.where(long_executable, long_worst, negative_infinity)
    short_rank = np.where(short_executable, short_worst, negative_infinity)
    choose_long = long_rank > short_rank
    choose_short = short_rank > long_rank
    side = np.zeros(len(endpoints), dtype=np.int8)
    side[choose_long] = 1
    side[choose_short] = -1
    strength = np.where(
        choose_long, long_worst, np.where(choose_short, short_worst, 0.0)
    )
    eligible = (side != 0) & (strength > 0.0)
    if architecture == "sign_magnitude_hurdle":
        long_probability = np.mean(
            np.stack(
                [prediction.long_profitable_probability for prediction in predictions]
            ),
            axis=0,
        )
        short_probability = np.mean(
            np.stack(
                [prediction.short_profitable_probability for prediction in predictions]
            ),
            axis=0,
        )
        eligible &= np.where(choose_long, long_probability, short_probability) > 0.50
    return _EnsembleScore(
        endpoint_indexes=endpoints,
        side=side,
        strength_bps=strength,
        eligible=eligible,
        long_executable=long_executable,
        short_executable=short_executable,
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
        predictions = state["predictions"]
        if not isinstance(predictions, Mapping):
            raise ValueError("Round 52 prediction state is invalid")
        candidate_predictions = predictions[candidate]
        if not isinstance(candidate_predictions, Mapping):
            raise ValueError("Round 52 candidate prediction state is invalid")
        ensemble = _ensemble_score(candidate_predictions[role])
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
            stress_traces,
            symbol_weight=1.0 / len(SYMBOLS),
        ),
        "paired_stress_overlap_violations": overlap_violations,
    }


def _policy_gate_reasons(
    result: Mapping[str, object],
    *,
    minimum_trades: int,
    maximum_drawdown_bps: float | None,
) -> list[str]:
    reasons: list[str] = []
    for scenario in ("base", "paired_stress"):
        scenario_result = result[scenario]
        if not isinstance(scenario_result, Mapping):
            raise ValueError("Round 52 policy result scenario is invalid")
        metrics = scenario_result["metrics"]
        symbol_pnl = scenario_result["symbol_net_bps"]
        if not isinstance(metrics, Mapping) or not isinstance(symbol_pnl, Mapping):
            raise ValueError("Round 52 policy result metrics are invalid")
        if int(metrics["trades"]) < int(minimum_trades):
            reasons.append(f"{scenario}_trades_below_{minimum_trades}")
        if float(metrics["total_net_bps"]) <= 0.0:
            reasons.append(f"{scenario}_total_net_bps_not_positive")
        profit_factor = metrics.get("profit_factor")
        if profit_factor is None or float(profit_factor) <= 1.0:
            reasons.append(f"{scenario}_profit_factor_not_above_one")
        if sum(float(value) > 0.0 for value in symbol_pnl.values()) < 2:
            reasons.append(f"{scenario}_positive_symbols_below_two")
        if float(scenario_result["maximum_single_symbol_positive_pnl_share"]) > 0.7:
            reasons.append(f"{scenario}_positive_pnl_concentration_above_0.70")
        if maximum_drawdown_bps is not None and float(
            metrics["max_drawdown_bps"]
        ) > float(maximum_drawdown_bps):
            reasons.append(f"{scenario}_drawdown_above_{maximum_drawdown_bps:g}_bps")
    if int(result["paired_stress_overlap_violations"]) != 0:
        reasons.append("paired_stress_overlap_violations_nonzero")
    return reasons


def _predictive_gate(
    metrics: Mapping[str, Mapping[str, object]],
    *,
    candidate: str,
) -> dict[str, object]:
    reasons: list[str] = []
    expected_by_symbol_side: dict[tuple[str, str], list[np.ndarray]] = {}
    for symbol in SYMBOLS:
        candidate_metrics = metrics[symbol][candidate]
        for seed in SEEDS:
            seed_metrics = candidate_metrics[str(seed)]["evaluation"]
            if candidate in HURDLE_CANDIDATES:
                diagnostics = candidate_metrics[str(seed)]["prediction_diagnostics"]
                for role in POLICY_ROLES:
                    if int(diagnostics[role]["magnitude_floor_count"]) != 0:
                        reasons.append(
                            f"{symbol}_{seed}_{role}_negative_magnitude_predictions"
                        )
            for side in SIDES:
                values = seed_metrics[side]
                if float(values["expected_payoff_mse_skill"]) <= 0.0:
                    reasons.append(f"{symbol}_{seed}_{side}_mse_skill_not_positive")
                if float(values["expected_payoff_spearman"]) < 0.03:
                    reasons.append(f"{symbol}_{seed}_{side}_spearman_below_0.03")
                if candidate in HURDLE_CANDIDATES:
                    if float(values["probability_log_loss_skill"]) <= 0.0:
                        reasons.append(
                            f"{symbol}_{seed}_{side}_log_loss_skill_not_positive"
                        )
                    if float(values["probability_brier_skill"]) <= 0.0:
                        reasons.append(
                            f"{symbol}_{seed}_{side}_brier_skill_not_positive"
                        )
                key = (symbol, side)
                expected_by_symbol_side.setdefault(key, []).append(
                    np.asarray(
                        metrics[symbol][candidate][str(seed)]["prediction_arrays"][
                            "evaluation"
                        ][f"{side}_expected_net_bps"],
                        dtype=np.float64,
                    )[
                        np.asarray(
                            metrics[symbol][candidate][str(seed)]["prediction_arrays"][
                                "evaluation"
                            ][f"{side}_executable"],
                            dtype=np.bool_,
                        )
                    ]
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
    values = [
        float(metrics[symbol][candidate][str(seed)]["evaluation"][side][name])
        for symbol in SYMBOLS
        for seed in SEEDS
        for side in SIDES
    ]
    return float(np.mean(values))


def run_round52(
    *,
    design_path: Path,
    binding_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    round51_report_path: Path,
    evidence_root: Path,
    compute_backend: str,
    memory_limit: str,
    threads: int,
) -> dict[str, object]:
    started = time.perf_counter()
    design, design_sha = _validate_design(design_path)
    binding, binding_sha = _validate_binding(binding_path, design_sha256=design_sha)
    source_report = _validate_round51_report(round51_report_path)
    specifications = _specifications(design)
    evidence_root.mkdir(parents=True, exist_ok=True)
    _progress("start", design=design_sha, binding=binding_sha)

    source_states: dict[str, dict[str, object]] = {}
    model_paths: dict[str, dict[str, dict[int, Path]]] = {}
    model_evidence: dict[str, object] = {}
    data_evidence: dict[str, object] = {}
    data_contract = design["data_contract"]
    execution = design["execution_target"]
    if not isinstance(data_contract, Mapping) or not isinstance(execution, Mapping):
        raise ValueError("Round 52 data or execution contract is invalid")

    # Phase one deliberately finishes every model before any evaluation prediction.
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
            source_report,
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
            dataset,
            targets,
            target_scenario="base",
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
                deterministic.payoff.decision_time_ms,
                ai.payoff.decision_time_ms,
            )
            or not np.array_equal(deterministic.long_executable, ai.long_executable)
            or not np.array_equal(deterministic.short_executable, ai.short_executable)
        ):
            raise ValueError(f"Round 52 {symbol} AI dataset changed row support")
        source_states[symbol] = {
            "dataset": dataset,
            "targets": targets,
            "datasets": {
                CANDIDATES[0]: deterministic,
                CANDIDATES[1]: deterministic,
                CANDIDATES[2]: ai,
            },
            "roles": roles,
            "predictions": {},
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
        for candidate in CANDIDATES:
            candidate_dataset = source_states[symbol]["datasets"][candidate]
            model_paths[symbol][candidate] = {}
            model_evidence[symbol][candidate] = {}
            for seed in SEEDS:
                path = (
                    evidence_root / "models" / symbol / candidate / f"seed-{seed}.json"
                )
                state = "loaded"
                if path.is_file():
                    model = load_executable_payoff_model(path)
                    if (
                        model.spec != specifications[candidate]
                        or model.source_dataset_sha256
                        != candidate_dataset.dataset_sha256
                        or model.seed != seed
                        or model.backend_requested != compute_backend
                    ):
                        raise ValueError(
                            f"Round 52 cached model drifted: {symbol} {candidate} {seed}"
                        )
                else:
                    state = "trained"
                    _progress("train", symbol=symbol, candidate=candidate, seed=seed)
                    model = train_executable_payoff_model(
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
                        progress=lambda name, side, step, total, s=symbol, c=candidate, d=seed: (
                            _progress(
                                "head",
                                symbol=s,
                                candidate=c,
                                seed=d,
                                head=name,
                                side=side,
                                step=f"{step}/{total}",
                            )
                        ),
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_executable_payoff_model(path, model)
                if compute_backend == "directml" and model.backend_kind != "opencl":
                    raise RuntimeError("Round 52 LightGBM silently left OpenCL")
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
                    "class_support": model.class_support,
                    "minimum_leaf_rows": model.minimum_leaf_rows,
                    "best_iterations": model.best_iterations,
                    "probability_calibration": model.probability_calibration,
                }
                del model
        del fincast_matrix
        gc.collect()
        _progress("symbol-models-complete", symbol=symbol)

    _progress("all-models-trained", models=len(SYMBOLS) * len(CANDIDATES) * len(SEEDS))

    metrics: dict[str, dict[str, object]] = {}
    prediction_evidence: dict[str, object] = {}
    for symbol in SYMBOLS:
        state = source_states[symbol]
        roles = state["roles"]
        metrics[symbol] = {}
        prediction_evidence[symbol] = {}
        state_predictions: dict[str, object] = {}
        for candidate in CANDIDATES:
            candidate_dataset = state["datasets"][candidate]
            metrics[symbol][candidate] = {}
            prediction_evidence[symbol][candidate] = {}
            candidate_predictions: dict[str, list[ExecutablePayoffPredictionBatch]] = {
                role: [] for role in POLICY_ROLES
            }
            for seed in SEEDS:
                model = load_executable_payoff_model(
                    model_paths[symbol][candidate][seed]
                )
                seed_metrics: dict[str, object] = {
                    "prediction_arrays": {},
                    "prediction_diagnostics": {},
                }
                prediction_evidence[symbol][candidate][str(seed)] = {}
                for role in POLICY_ROLES:
                    prediction = predict_executable_payoff_model(
                        model,
                        candidate_dataset,
                        roles[role],
                    )
                    candidate_predictions[role].append(prediction)
                    path = (
                        evidence_root
                        / "predictions"
                        / symbol
                        / candidate
                        / f"seed-{seed}-{role}.npz"
                    )
                    prediction_evidence[symbol][candidate][str(seed)][role] = (
                        _save_prediction(path, prediction)
                    )
                    seed_metrics[role] = {
                        side: _prediction_metrics(
                            model=model,
                            dataset=candidate_dataset,
                            indexes=roles[role],
                            prediction=prediction,
                            side=side,
                        )
                        for side in SIDES
                    }
                    seed_metrics["prediction_diagnostics"][role] = {
                        "magnitude_floor_count": prediction.magnitude_floor_count,
                    }
                    seed_metrics["prediction_arrays"][role] = {
                        "long_expected_net_bps": prediction.long_expected_net_bps,
                        "short_expected_net_bps": prediction.short_expected_net_bps,
                        "long_executable": prediction.long_executable,
                        "short_executable": prediction.short_executable,
                    }
                metrics[symbol][candidate][str(seed)] = seed_metrics
                del model
            state_predictions[candidate] = candidate_predictions
        state["predictions"] = state_predictions
        _progress("symbol-predictions-complete", symbol=symbol)

    coverages = tuple(
        float(value) for value in design["economic_screen"]["threshold_coverages"]
    )
    policy_results: dict[str, object] = {}
    selected_policy: dict[str, object] = {}
    for candidate in CANDIDATES:
        candidate_rows: list[dict[str, object]] = []
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
            calibration = _policy_result(
                symbol_state=source_states,
                candidate=candidate,
                role="policy_calibration",
                thresholds=thresholds,
            )
            calibration_reasons = _policy_gate_reasons(
                calibration,
                minimum_trades=6,
                maximum_drawdown_bps=None,
            )
            evaluation = _policy_result(
                symbol_state=source_states,
                candidate=candidate,
                role="evaluation",
                thresholds=thresholds,
            )
            evaluation_reasons = _policy_gate_reasons(
                evaluation,
                minimum_trades=30,
                maximum_drawdown_bps=120.0,
            )
            candidate_rows.append(
                {
                    "coverage": coverage,
                    "thresholds_bps": thresholds,
                    "policy_calibration": calibration,
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
        passing = [
            row for row in candidate_rows if row["policy_calibration_gate"]["passed"]
        ]
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
            "consumed_evaluation": (
                selected["consumed_evaluation_diagnostic"] if selected else None
            ),
            "evaluation_reasons": (
                selected["consumed_evaluation_gate"]["reasons"]
                if selected
                else ["no_policy_calibration_variant_passed"]
            ),
        }
        policy_results[candidate] = candidate_rows

    predictive_gates = {
        candidate: _predictive_gate(metrics, candidate=candidate)
        for candidate in CANDIDATES
    }
    control_log_loss = _mean_metric(metrics, CANDIDATES[1], "probability_log_loss")
    ai_log_loss = _mean_metric(metrics, CANDIDATES[2], "probability_log_loss")
    control_spearman = _mean_metric(metrics, CANDIDATES[1], "expected_payoff_spearman")
    ai_spearman = _mean_metric(metrics, CANDIDATES[2], "expected_payoff_spearman")
    ai_reasons: list[str] = []
    if control_log_loss - ai_log_loss < 0.005:
        ai_reasons.append("average_probability_log_loss_improvement_below_0.005")
    if ai_spearman - control_spearman < 0.005:
        ai_reasons.append("average_expected_payoff_spearman_improvement_below_0.005")
    if not predictive_gates[CANDIDATES[1]]["passed"]:
        ai_reasons.append("deterministic_hurdle_predictive_gate_failed")
    if not selected_policy[CANDIDATES[2]]["evaluation_gate_passed"]:
        ai_reasons.append("ai_consumed_evaluation_economic_gate_failed")
    if not selected_policy[CANDIDATES[1]]["evaluation_gate_passed"]:
        ai_reasons.append("deterministic_hurdle_consumed_evaluation_gate_failed")
    control_evaluation = selected_policy[CANDIDATES[1]]["consumed_evaluation"]
    ai_evaluation = selected_policy[CANDIDATES[2]]["consumed_evaluation"]
    if isinstance(control_evaluation, Mapping) and isinstance(ai_evaluation, Mapping):
        for scenario in ("base", "paired_stress"):
            control_scenario = control_evaluation[scenario]
            ai_scenario = ai_evaluation[scenario]
            control_metrics = control_scenario["metrics"]
            ai_metrics = ai_scenario["metrics"]
            if float(ai_metrics["mean_net_bps"]) <= float(
                control_metrics["mean_net_bps"]
            ):
                ai_reasons.append(f"{scenario}_mean_net_bps_not_improved")
            if float(ai_metrics["max_drawdown_bps"]) > float(
                control_metrics["max_drawdown_bps"]
            ):
                ai_reasons.append(f"{scenario}_drawdown_worsened")
            if int(ai_metrics["trades"]) < int(control_metrics["trades"]):
                ai_reasons.append(f"{scenario}_trade_count_worsened")
            if float(ai_scenario["maximum_single_symbol_positive_pnl_share"]) > float(
                control_scenario["maximum_single_symbol_positive_pnl_share"]
            ):
                ai_reasons.append(f"{scenario}_concentration_worsened")
    ai_uplift = {
        "passed": not ai_reasons,
        "reasons": ai_reasons,
        "control_average_probability_log_loss": control_log_loss,
        "ai_average_probability_log_loss": ai_log_loss,
        "probability_log_loss_improvement": control_log_loss - ai_log_loss,
        "control_average_expected_payoff_spearman": control_spearman,
        "ai_average_expected_payoff_spearman": ai_spearman,
        "expected_payoff_spearman_improvement": ai_spearman - control_spearman,
    }
    mechanism_candidates = [
        candidate
        for candidate in CANDIDATES
        if predictive_gates[candidate]["passed"]
        and selected_policy[candidate]["evaluation_gate_passed"]
        and (candidate != CANDIDATES[2] or ai_uplift["passed"])
    ]
    round_reasons = ["selection_contaminated_consumed_development_interval"]
    if not mechanism_candidates:
        round_reasons.append("no_candidate_passed_predictive_and_economic_gates")

    # Remove in-memory arrays from the JSON-facing metric records.
    report_metrics: dict[str, object] = {}
    for symbol in SYMBOLS:
        report_metrics[symbol] = {}
        for candidate in CANDIDATES:
            report_metrics[symbol][candidate] = {}
            for seed in SEEDS:
                record = dict(metrics[symbol][candidate][str(seed)])
                record.pop("prediction_arrays")
                report_metrics[symbol][candidate][str(seed)] = record

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
        },
        "source_round_51": {
            "report_path": str(round51_report_path.resolve()),
            "report_canonical_sha256": EXPECTED_ROUND51_REPORT_CANONICAL_SHA256,
            "report_file_sha256": EXPECTED_ROUND51_REPORT_FILE_SHA256,
        },
        "data": data_evidence,
        "models": model_evidence,
        "prediction_artifacts": prediction_evidence,
        "predictive_metrics": report_metrics,
        "predictive_gates": predictive_gates,
        "policy_grid": policy_results,
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
        default=research / "round-052-executable-support-hurdle-fincast-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-052-execution-binding.json",
    )
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--round51-report", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--compute-backend", default="directml")
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument(
        "--threads", type=int, default=max(1, min(12, os.cpu_count() or 1))
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.threads < 1 or args.threads > 64:
        raise ValueError("Round 52 DuckDB thread count is invalid")
    report = run_round52(
        design_path=args.design.resolve(),
        binding_path=args.binding.resolve(),
        warehouse_path=args.warehouse.resolve(),
        cache_root=args.cache_root.resolve(),
        round51_report_path=args.round51_report.resolve(),
        evidence_root=args.evidence_root.resolve(),
        compute_backend=str(args.compute_backend),
        memory_limit=str(args.memory_limit),
        threads=int(args.threads),
    )
    if args.json:
        print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
