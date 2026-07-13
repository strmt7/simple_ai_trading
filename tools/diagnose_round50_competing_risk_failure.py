"""Diagnose Round 50 path-value failure without training or outcome tuning."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.action_hurdle_tcn_model import (  # noqa: E402
    build_action_hurdle_temporal_dataset,
)
from simple_ai_trading.barrier_competing_risk_analysis import (  # noqa: E402
    replay_fixed_trades,
)
from simple_ai_trading.barrier_payoff_data import (  # noqa: E402
    BarrierPayoffDataset,
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
CANDIDATES = ("direct_barrier_mean_tcn", "competing_risk_barrier_tcn")
SIDES = ("short", "long")
SEEDS = (5001, 5002, 5003)
EXPECTED_REPORT_CANONICAL_SHA256 = (
    "8629a07940c0d8b4b16b35be4d7b651c1625807f8abae82a9e7fa7bfe73b6850"
)
EXPECTED_REPORT_FILE_SHA256 = (
    "47385351b7faf6bf1feb19d84f1c6200c5b6d5552e735877ae95f4f2c62245e8"
)
EXPECTED_BARRIER_DATASET_SHA256 = (
    "31c7713339cff9ad12f3bae02475743d09b2248bfc1b85e02e1f3306a699e774"
)
EXPECTED_PREDECESSOR_DATASET_SHA256 = (
    "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
)
MINUTE_MS = 60_000


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("Round 50 report must be an object")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        claimed != EXPECTED_REPORT_CANONICAL_SHA256
        or _canonical_sha256(canonical) != claimed
        or _file_sha256(path) != EXPECTED_REPORT_FILE_SHA256
        or report.get("round") != ROUND
        or report.get("schema_version") != "path-bounded-competing-risk-tcn-report-v1"
        or report.get("claims", {}).get("profitability_claim") is not False
        or report.get("claims", {}).get("trading_authority") is not False
    ):
        raise ValueError("Round 50 report identity or claims are invalid")
    for candidate in CANDIDATES:
        artifacts = report["artifacts"][candidate]
        if len(artifacts) != len(SEEDS):
            raise ValueError("Round 50 artifact set is incomplete")
        for artifact in artifacts:
            for path_key, bytes_key, hash_key in (
                ("model_path", "model_bytes", "model_sha256"),
                ("prediction_path", "prediction_bytes", "prediction_sha256"),
            ):
                artifact_path = Path(artifact[path_key])
                if (
                    not artifact_path.is_file()
                    or artifact_path.stat().st_size != int(artifact[bytes_key])
                    or _file_sha256(artifact_path) != artifact[hash_key]
                ):
                    raise ValueError(f"Round 50 artifact drifted: {artifact_path}")
    return report


def _progress(event: str, details: Mapping[str, object]) -> None:
    print(
        json.dumps(
            {"event": event, "details": dict(details)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _load_corpus(
    database: Path, source_certificate: Path
) -> tuple[object, BarrierPayoffDataset]:
    panel, price_source = load_verified_minute_panel(database, progress=_progress)
    premium, funding, derivatives_source = load_derivatives_state(
        database,
        panel,
        price_source,
        source_certificate_path=source_certificate,
        progress=_progress,
    )
    source = build_derivatives_hurdle_dataset(
        panel,
        premium,
        funding,
        derivatives_source,
        progress=_progress,
    )
    temporal = build_action_hurdle_temporal_dataset(source)
    if temporal.dataset_sha256 != EXPECTED_PREDECESSOR_DATASET_SHA256:
        raise ValueError("Round 50 predecessor dataset changed")
    barrier = build_barrier_payoff_dataset(
        panel,
        funding,
        source,
        temporal,
        BarrierSpecification(
            horizon_minutes=60,
            stop_volatility_multiple=1.0,
            take_profit_to_stop_ratio=2.0,
            minimum_stop_bps=24.0,
            maximum_stop_bps=80.0,
            round_trip_execution_charge_bps=12.0,
        ),
    )
    if barrier.dataset_sha256 != EXPECTED_BARRIER_DATASET_SHA256:
        raise ValueError("Round 50 barrier dataset changed")
    return temporal, barrier


def _load_predictions(
    report: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    common_indices: np.ndarray | None = None
    candidates: dict[str, dict[str, np.ndarray]] = {}
    for candidate in CANDIDATES:
        arrays: dict[str, list[np.ndarray]] = {
            "action": [],
            "groups": [],
            "timeout_mean": [],
            "event_minutes": [],
        }
        for artifact in sorted(
            report["artifacts"][candidate], key=lambda item: int(item["seed"])
        ):
            with np.load(Path(artifact["prediction_path"]), allow_pickle=False) as item:
                indices = item["global_indices"].astype(np.int64, copy=False)
                if common_indices is None:
                    common_indices = indices.copy()
                elif not np.array_equal(common_indices, indices):
                    raise ValueError("Round 50 prediction indices differ")
                arrays["action"].append(item["action_values_bps"].copy())
                arrays["groups"].append(item["event_group_probability"].copy())
                arrays["timeout_mean"].append(item["timeout_mean_risk_units"].copy())
                arrays["event_minutes"].append(item["event_expected_minutes"].copy())
        candidates[candidate] = {
            name: np.stack(values).astype(np.float64, copy=False)
            for name, values in arrays.items()
        }
    if common_indices is None:
        raise ValueError("Round 50 prediction artifacts are empty")
    return common_indices, candidates


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _flatten_role(
    value: np.ndarray, role_mask: np.ndarray, side_index: int
) -> np.ndarray:
    return value[role_mask, :, side_index].reshape(-1).astype(np.float64)


def _prediction_summary(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    actual: np.ndarray,
    events: np.ndarray,
    roles: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary: list[dict[str, object]] = []
    quantiles: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        seeds = predictions[candidate]["action"]
        mean = np.mean(seeds, axis=0)
        worst = np.min(seeds, axis=0)
        disagreement = np.std(seeds, axis=0)
        for role, role_mask in roles.items():
            for side_index, side in enumerate(SIDES):
                predicted = _flatten_role(mean, role_mask, side_index)
                predicted_worst = _flatten_role(worst, role_mask, side_index)
                predicted_std = _flatten_role(disagreement, role_mask, side_index)
                realized = _flatten_role(actual, role_mask, side_index)
                selected = predicted_worst > 0.0
                summary.append(
                    {
                        "candidate_id": candidate,
                        "role": role,
                        "side": side,
                        "rows": int(realized.size),
                        "prediction_mean_bps": float(np.mean(predicted)),
                        "prediction_std_bps": float(np.std(predicted)),
                        "actual_mean_bps": float(np.mean(realized)),
                        "actual_std_bps": float(np.std(realized)),
                        "mean_bias_bps": float(np.mean(predicted - realized)),
                        "mse_bps2": float(np.mean((predicted - realized) ** 2)),
                        "spearman": _safe_spearman(predicted, realized),
                        "pearson": _safe_pearson(predicted, realized),
                        "all_seed_positive_rows": int(np.count_nonzero(selected)),
                        "all_seed_positive_fraction": float(np.mean(selected)),
                        "selected_prediction_mean_bps": (
                            float(np.mean(predicted[selected]))
                            if np.any(selected)
                            else 0.0
                        ),
                        "selected_actual_mean_bps": (
                            float(np.mean(realized[selected]))
                            if np.any(selected)
                            else 0.0
                        ),
                        "selected_actual_positive_rate": (
                            float(np.mean(realized[selected] > 0.0))
                            if np.any(selected)
                            else 0.0
                        ),
                        "selected_seed_disagreement_mean_bps": (
                            float(np.mean(predicted_std[selected]))
                            if np.any(selected)
                            else 0.0
                        ),
                    }
                )
                role_events = _flatten_role(events, role_mask, side_index).astype(
                    np.int8
                )
                order = np.argsort(predicted, kind="stable")
                for quantile_index, quantile_rows in enumerate(
                    np.array_split(order, 10), start=1
                ):
                    quantile_events = role_events[quantile_rows]
                    quantiles.append(
                        {
                            "candidate_id": candidate,
                            "role": role,
                            "side": side,
                            "prediction_decile": quantile_index,
                            "rows": int(quantile_rows.size),
                            "prediction_mean_bps": float(
                                np.mean(predicted[quantile_rows])
                            ),
                            "worst_seed_prediction_mean_bps": float(
                                np.mean(predicted_worst[quantile_rows])
                            ),
                            "actual_mean_bps": float(np.mean(realized[quantile_rows])),
                            "actual_positive_rate": float(
                                np.mean(realized[quantile_rows] > 0.0)
                            ),
                            "stop_rate": float(np.mean(quantile_events == 0)),
                            "timeout_rate": float(np.mean(quantile_events == 1)),
                            "take_profit_rate": float(np.mean(quantile_events == 2)),
                        }
                    )
    return summary, quantiles


def _path_decomposition(
    report: Mapping[str, object],
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    barrier: BarrierPayoffDataset,
    indices: np.ndarray,
    actual: np.ndarray,
    events: np.ndarray,
    roles: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], float]:
    candidate = "competing_risk_barrier_tcn"
    values = predictions[candidate]
    groups = values["groups"]
    timeout_mean = values["timeout_mean"]
    stop = barrier.stop_bps[indices].astype(np.float64)
    baselines = report["target_baselines"][candidate]
    stop_residual = np.asarray(
        baselines["stop_residual_mean_risk_units"], dtype=np.float64
    )
    take_residual = np.asarray(
        baselines["take_residual_mean_risk_units"], dtype=np.float64
    )
    stop_value = np.empty((indices.size, len(SYMBOLS), len(SIDES)))
    take_value = np.empty_like(stop_value)
    for symbol_index in range(len(SYMBOLS)):
        for side_index in range(len(SIDES)):
            stop_value[:, symbol_index, side_index] = (
                -stop[:, symbol_index]
                - 12.0
                + stop_residual[symbol_index, side_index] * stop[:, symbol_index]
            )
            take_value[:, symbol_index, side_index] = (
                2.0 * stop[:, symbol_index]
                - 12.0
                + take_residual[symbol_index, side_index] * stop[:, symbol_index]
            )
    stop_component = groups[..., 0] * stop_value[None, ...]
    timeout_component = groups[..., 1] * timeout_mean * stop[None, :, :, None]
    take_component = groups[..., 2] * take_value[None, ...]
    reconstructed = stop_component + take_component + timeout_component
    reconstruction_error = float(np.max(np.abs(reconstructed - values["action"])))
    if reconstruction_error > 1e-4:
        raise ValueError("Round 50 structured value reconstruction failed")
    rows: list[dict[str, object]] = []
    components = {
        "stop": np.mean(stop_component, axis=0),
        "take_profit": np.mean(take_component, axis=0),
        "timeout": np.mean(timeout_component, axis=0),
    }
    action_mean = np.mean(values["action"], axis=0)
    action_worst = np.min(values["action"], axis=0)
    group_mean = np.mean(groups, axis=0)
    duration_mean = np.mean(values["event_minutes"], axis=0)
    for role, role_mask in roles.items():
        for side_index, side in enumerate(SIDES):
            realized = _flatten_role(actual, role_mask, side_index)
            action = _flatten_role(action_mean, role_mask, side_index)
            selected = _flatten_role(action_worst, role_mask, side_index) > 0.0
            role_events = _flatten_role(events, role_mask, side_index).astype(np.int8)
            row: dict[str, object] = {
                "role": role,
                "side": side,
                "rows": int(realized.size),
                "selected_rows": int(np.count_nonzero(selected)),
                "action_actual_spearman": _safe_spearman(action, realized),
                "expected_event_minute_mean": float(
                    np.mean(_flatten_role(duration_mean, role_mask, side_index))
                ),
            }
            for name, component in components.items():
                flat = _flatten_role(component, role_mask, side_index)
                row[f"{name}_component_mean_bps"] = float(np.mean(flat))
                row[f"{name}_component_actual_spearman"] = _safe_spearman(
                    flat, realized
                )
                row[f"selected_{name}_component_mean_bps"] = (
                    float(np.mean(flat[selected])) if np.any(selected) else 0.0
                )
            for group_index, name in enumerate(("stop", "timeout", "take_profit")):
                predicted_group = _flatten_role(
                    group_mean[..., group_index], role_mask, side_index
                )
                row[f"selected_predicted_{name}_rate"] = (
                    float(np.mean(predicted_group[selected]))
                    if np.any(selected)
                    else 0.0
                )
                row[f"selected_actual_{name}_rate"] = (
                    float(np.mean(role_events[selected] == group_index))
                    if np.any(selected)
                    else 0.0
                )
            rows.append(row)
    return rows, reconstruction_error


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    centered = x - np.mean(x)
    denominator = float(np.dot(centered, centered))
    slope = (
        max(0.0, float(np.dot(centered, y - np.mean(y)) / denominator))
        if denominator > 0.0
        else 0.0
    )
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def _calibrate_candidate(
    method: str,
    predictions: np.ndarray,
    actual: np.ndarray,
    calibration_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    calibrated = predictions.copy()
    coefficients: list[dict[str, object]] = []
    for seed_index, seed in enumerate(SEEDS):
        for side_index, side in enumerate(SIDES):
            if method.startswith("pooled"):
                groups = [("pooled", tuple(range(len(SYMBOLS))))]
            else:
                groups = [(symbol, (index,)) for index, symbol in enumerate(SYMBOLS)]
            for group, symbol_indices in groups:
                x = predictions[
                    seed_index,
                    calibration_mask,
                    :,
                    side_index,
                ][:, symbol_indices].reshape(-1)
                y = actual[calibration_mask, :, side_index][:, symbol_indices].reshape(
                    -1
                )
                if method.endswith("offset"):
                    slope = 1.0
                    intercept = float(np.mean(y - x))
                elif method.endswith("affine"):
                    intercept, slope = _fit_affine(x, y)
                else:
                    raise KeyError(method)
                for symbol_index in symbol_indices:
                    calibrated[seed_index, :, symbol_index, side_index] = (
                        intercept
                        + slope * predictions[seed_index, :, symbol_index, side_index]
                    )
                coefficients.append(
                    {
                        "method": method,
                        "seed": seed,
                        "side": side,
                        "group": group,
                        "rows": int(x.size),
                        "intercept_bps": intercept,
                        "slope": slope,
                        "raw_calibration_bias_bps": float(np.mean(x - y)),
                        "calibrated_calibration_bias_bps": float(
                            np.mean(intercept + slope * x - y)
                        ),
                        "raw_calibration_mse_bps2": float(np.mean((x - y) ** 2)),
                        "calibrated_calibration_mse_bps2": float(
                            np.mean((intercept + slope * x - y) ** 2)
                        ),
                    }
                )
    return calibrated, coefficients


def _convex_blend(
    direct: np.ndarray,
    path: np.ndarray,
    actual: np.ndarray,
    calibration_mask: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = np.empty_like(direct)
    coefficients: list[dict[str, object]] = []
    for seed_index, seed in enumerate(SEEDS):
        for side_index, side in enumerate(SIDES):
            direct_cal = direct[seed_index, calibration_mask, :, side_index].reshape(-1)
            path_cal = path[seed_index, calibration_mask, :, side_index].reshape(-1)
            y = actual[calibration_mask, :, side_index].reshape(-1)
            difference = direct_cal - path_cal
            centered = difference - np.mean(difference)
            denominator = float(np.dot(centered, centered))
            direct_weight = (
                float(
                    np.clip(
                        np.dot(centered, y - path_cal - np.mean(y - path_cal))
                        / denominator,
                        0.0,
                        1.0,
                    )
                )
                if denominator > 0.0
                else 0.5
            )
            raw_blend = (
                direct_weight * direct[seed_index, :, :, side_index]
                + (1.0 - direct_weight) * path[seed_index, :, :, side_index]
            )
            raw_cal = direct_weight * direct_cal + (1.0 - direct_weight) * path_cal
            intercept = float(np.mean(y - raw_cal))
            output[seed_index, :, :, side_index] = intercept + raw_blend
            coefficients.append(
                {
                    "method": "pooled_side_convex_blend_offset",
                    "seed": seed,
                    "side": side,
                    "direct_weight": direct_weight,
                    "path_weight": 1.0 - direct_weight,
                    "intercept_bps": intercept,
                    "rows": int(y.size),
                    "raw_calibration_mse_bps2": float(np.mean((raw_cal - y) ** 2)),
                    "calibrated_calibration_mse_bps2": float(
                        np.mean((intercept + raw_cal - y) ** 2)
                    ),
                }
            )
    return output, coefficients


def _select_trades(
    candidate_id: str,
    seed_values: np.ndarray,
    indices: np.ndarray,
    evaluation_mask: np.ndarray,
    temporal: object,
    barrier: BarrierPayoffDataset,
) -> list[dict[str, object]]:
    trades: list[dict[str, object]] = []
    evaluation_local = np.flatnonzero(evaluation_mask)
    for symbol_index, symbol in enumerate(SYMBOLS):
        available_after_ms = -1
        for local_index in evaluation_local:
            global_index = int(indices[local_index])
            decision_time_ms = int(temporal.timestamps_ms[global_index])
            if decision_time_ms < available_after_ms:
                continue
            values = seed_values[:, local_index, symbol_index]
            worst = np.min(values, axis=0)
            eligible = worst > 0.0
            if not np.any(eligible):
                continue
            if eligible[0] and eligible[1]:
                if worst[0] == worst[1]:
                    continue
                side_index = int(np.argmax(worst))
            else:
                side_index = int(np.flatnonzero(eligible)[0])
            event_minute = int(
                barrier.event_minute[global_index, symbol_index, side_index]
            )
            entry_time_ms = decision_time_ms + MINUTE_MS
            exit_time_ms = entry_time_ms + event_minute * MINUTE_MS
            available_after_ms = exit_time_ms
            base_payoff = float(
                barrier.net_payoff_bps[global_index, symbol_index, side_index]
            )
            trades.append(
                {
                    "trade_id": f"{candidate_id}:{symbol}:{decision_time_ms}:{(-1, 1)[side_index]}",
                    "candidate_id": candidate_id,
                    "symbol": symbol,
                    "symbol_index": symbol_index,
                    "decision_index": global_index,
                    "decision_time_ms": decision_time_ms,
                    "entry_time_ms": entry_time_ms,
                    "exit_time_ms": exit_time_ms,
                    "side": (-1, 1)[side_index],
                    "side_name": SIDES[side_index],
                    "event_code": int(
                        barrier.event_code[global_index, symbol_index, side_index]
                    ),
                    "event_name": ("stop_loss", "timeout", "take_profit")[
                        int(barrier.event_code[global_index, symbol_index, side_index])
                    ],
                    "holding_minutes": event_minute,
                    "stop_bps": float(barrier.stop_bps[global_index, symbol_index]),
                    "take_profit_bps": float(
                        barrier.take_profit_bps[global_index, symbol_index]
                    ),
                    "worst_seed_expected_payoff_bps": float(worst[side_index]),
                    "mean_seed_expected_payoff_bps": float(
                        np.mean(values[:, side_index])
                    ),
                    "base_net_payoff_bps": base_payoff,
                    "stress_net_payoff_bps": base_payoff - 4.0,
                }
            )
    return sorted(trades, key=lambda item: (item["exit_time_ms"], item["trade_id"]))


def _compact_replay(
    candidate_id: str,
    method: str,
    seed_values: np.ndarray,
    indices: np.ndarray,
    evaluation_mask: np.ndarray,
    temporal: object,
    barrier: BarrierPayoffDataset,
    *,
    bootstrap_candidate_index: int,
) -> dict[str, object]:
    trades = _select_trades(
        f"{candidate_id}:{method}",
        seed_values,
        indices,
        evaluation_mask,
        temporal,
        barrier,
    )
    replay = replay_fixed_trades(
        trades,
        temporal,
        barrier,
        candidate_index=bootstrap_candidate_index,
    )
    side_counts = {
        side: sum(item["side_name"] == side for item in trades) for side in SIDES
    }
    scenarios = {}
    for scenario in ("base", "stress"):
        result = replay["scenarios"][scenario]
        scenarios[scenario] = {
            key: value for key, value in result.items() if key != "daily"
        }
    return {
        "candidate_id": candidate_id,
        "method": method,
        "side_closed_trades": side_counts,
        "scenarios": scenarios,
    }


def _calibration_probes(
    predictions: Mapping[str, Mapping[str, np.ndarray]],
    actual: np.ndarray,
    indices: np.ndarray,
    roles: Mapping[str, np.ndarray],
    temporal: object,
    barrier: BarrierPayoffDataset,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    calibration_mask = roles["calibration"]
    evaluation_mask = roles["evaluation"]
    replays: list[dict[str, object]] = []
    coefficients: list[dict[str, object]] = []
    replay_index = 0
    for candidate in CANDIDATES:
        raw = predictions[candidate]["action"]
        candidate_index = CANDIDATES.index(candidate)
        replays.append(
            _compact_replay(
                candidate,
                "raw",
                raw,
                indices,
                evaluation_mask,
                temporal,
                barrier,
                bootstrap_candidate_index=candidate_index,
            )
        )
        replay_index += 1
        for method in (
            "pooled_side_offset",
            "pooled_side_affine",
            "symbol_side_offset",
            "symbol_side_affine",
        ):
            calibrated, fitted = _calibrate_candidate(
                method, raw, actual, calibration_mask
            )
            coefficients.extend({"candidate_id": candidate, **row} for row in fitted)
            replays.append(
                _compact_replay(
                    candidate,
                    method,
                    calibrated,
                    indices,
                    evaluation_mask,
                    temporal,
                    barrier,
                    bootstrap_candidate_index=100 + replay_index,
                )
            )
            replay_index += 1
    blended, fitted = _convex_blend(
        predictions["direct_barrier_mean_tcn"]["action"],
        predictions["competing_risk_barrier_tcn"]["action"],
        actual,
        calibration_mask,
    )
    coefficients.extend(
        {"candidate_id": "direct_path_convex_blend", **row} for row in fitted
    )
    replays.append(
        _compact_replay(
            "direct_path_convex_blend",
            "pooled_side_convex_blend_offset",
            blended,
            indices,
            evaluation_mask,
            temporal,
            barrier,
            bootstrap_candidate_index=100 + replay_index,
        )
    )
    return replays, coefficients


def _raw_replay_invariants(
    report: Mapping[str, object], replays: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    invariants: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        observed = next(
            row
            for row in replays
            if row["candidate_id"] == candidate and row["method"] == "raw"
        )
        expected_scenarios = {
            scenario: {
                key: value
                for key, value in report["fixed_policy"][candidate]["scenarios"][
                    scenario
                ].items()
                if key != "daily"
            }
            for scenario in ("base", "stress")
        }
        expected_hash = _canonical_sha256(expected_scenarios)
        observed_hash = _canonical_sha256(observed["scenarios"])
        if observed_hash != expected_hash:
            raise ValueError(f"Raw policy replay drifted for {candidate}")
        invariants.append(
            {
                "candidate_id": candidate,
                "passed": True,
                "scenario_canonical_sha256": observed_hash,
            }
        )
    return invariants


def _fixed_policy_side_breakdown(
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        trades = report["fixed_policy"][candidate]["trades"]
        for side in SIDES:
            selected = [item for item in trades if item["side_name"] == side]
            payoff = np.asarray(
                [float(item["base_net_payoff_bps"]) for item in selected],
                dtype=np.float64,
            )
            positive = float(np.sum(payoff[payoff > 0.0]))
            negative = float(-np.sum(payoff[payoff < 0.0]))
            rows.append(
                {
                    "candidate_id": candidate,
                    "side": side,
                    "closed_trades": len(selected),
                    "mean_net_payoff_bps": float(np.mean(payoff))
                    if payoff.size
                    else 0.0,
                    "portfolio_return_fraction": float(np.sum(payoff) / 10_000.0 / 3.0),
                    "win_rate": float(np.mean(payoff > 0.0)) if payoff.size else 0.0,
                    "profit_factor": positive / negative if negative > 0.0 else 0.0,
                    "stop_loss_trades": sum(
                        item["event_name"] == "stop_loss" for item in selected
                    ),
                    "take_profit_trades": sum(
                        item["event_name"] == "take_profit" for item in selected
                    ),
                    "timeout_trades": sum(
                        item["event_name"] == "timeout" for item in selected
                    ),
                }
            )
    return rows


def diagnose(arguments: argparse.Namespace) -> dict[str, object]:
    report_path = arguments.evidence_root.resolve() / "report.json"
    report = _read_report(report_path)
    temporal, barrier = _load_corpus(
        arguments.database.resolve(), arguments.source_certificate.resolve()
    )
    indices, predictions = _load_predictions(report)
    actual = barrier.net_payoff_bps[indices].astype(np.float64)
    events = barrier.event_code[indices]
    roles = {
        "calibration": barrier.role_masks["calibration"][indices],
        "evaluation": barrier.role_masks["viability"][indices],
    }
    if (
        np.count_nonzero(roles["calibration"]) != 26_484
        or np.count_nonzero(roles["evaluation"]) != 52_104
        or np.any(roles["calibration"] & roles["evaluation"])
    ):
        raise ValueError("Round 50 diagnosis roles drifted")
    summary, quantiles = _prediction_summary(predictions, actual, events, roles)
    decomposition, reconstruction_error = _path_decomposition(
        report,
        predictions,
        barrier,
        indices,
        actual,
        events,
        roles,
    )
    calibration_replays, calibration_coefficients = _calibration_probes(
        predictions,
        actual,
        indices,
        roles,
        temporal,
        barrier,
    )
    raw_replay_invariants = _raw_replay_invariants(report, calibration_replays)
    result: dict[str, object] = {
        "schema_version": "round-050-competing-risk-failure-diagnosis-v1",
        "round": ROUND,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_report_canonical_sha256": EXPECTED_REPORT_CANONICAL_SHA256,
        "source_report_file_sha256": EXPECTED_REPORT_FILE_SHA256,
        "dataset": {
            "barrier_dataset_sha256": barrier.dataset_sha256,
            "predecessor_dataset_sha256": temporal.dataset_sha256,
            "source_resolution_seconds": 60,
            "decision_interval_seconds": 300,
            "calibration_timestamps": int(np.count_nonzero(roles["calibration"])),
            "evaluation_timestamps": int(np.count_nonzero(roles["evaluation"])),
            "symbols": list(SYMBOLS),
            "synthetic_rows": 0,
        },
        "research_basis": [
            {
                "source": "https://proceedings.mlr.press/v80/imani18a.html",
                "finding_used": "Soft categorical targets can stabilize noisy regression compared with a scalar point target.",
            },
            {
                "source": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/39717429762da92201a750dd03386920-Abstract-Conference.html",
                "finding_used": "Distributional regression should be evaluated with a proper distribution score such as CRPS, including model aggregation and selection.",
            },
            {
                "source": "https://proceedings.mlr.press/v286/deshpande25a.html",
                "finding_used": "Post-hoc regression recalibration can address distribution shift while retaining a baseline-regret objective.",
            },
            {
                "source": "https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html",
                "finding_used": "Conflicting auxiliary-task gradients can cause negative transfer and should be measured before sharing an encoder.",
            },
            {
                "source": "https://proceedings.mlr.press/v162/shah22a.html",
                "finding_used": "Abstention can worsen subgroup performance, so coverage reduction requires symbol, side, and month breadth checks.",
            },
            {
                "source": "https://arxiv.org/abs/2606.00060",
                "finding_used": "Recent non-peer-reviewed BTC evidence emphasizes that cost-aware forecast-to-trade conversion can matter more than small loss-function differences.",
                "limitation": "Hourly BTC results and reported profitability are not imported as evidence for this repository.",
            },
        ],
        "prediction_summary": summary,
        "prediction_deciles": quantiles,
        "path_decomposition": decomposition,
        "structured_value_reconstruction_max_abs_error_bps": reconstruction_error,
        "fixed_policy_side_breakdown": _fixed_policy_side_breakdown(report),
        "raw_policy_replay_invariants": raw_replay_invariants,
        "calibration_only_probe": {
            "fit_role": "2024-Q4 calibration only",
            "evaluation_role": "consumed 2025-H1; diagnostic only",
            "threshold_search_performed": False,
            "methods": calibration_replays,
            "coefficients": calibration_coefficients,
        },
        "claims": {
            "selection_contaminated": True,
            "development_only": True,
            "profitability_claim": False,
            "trading_authority": False,
            "leverage_applied": False,
            "ai_uplift_claim": False,
        },
    }
    result["canonical_sha256"] = _canonical_sha256(result)
    write_json_atomic(arguments.output.resolve(), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-certificate", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = diagnose(arguments)
    print(
        json.dumps(
            {
                "canonical_sha256": result["canonical_sha256"],
                "output": str(arguments.output.resolve()),
                "calibration_methods": len(result["calibration_only_probe"]["methods"]),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
