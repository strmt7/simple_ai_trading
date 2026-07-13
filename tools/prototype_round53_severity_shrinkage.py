"""Probe calibration-only severity shrinkage against sealed Round 52 artifacts.

This is a development experiment, not a trading or promotion authority. It reuses
the frozen Round 52 data roles and model artifacts, estimates shrinkage only on
the probability-calibration role, and consumes no new market interval.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.executable_payoff_lightgbm import (  # noqa: E402
    ExecutablePayoffPredictionBatch,
    TrainedExecutablePayoffModel,
    build_executable_payoff_dataset,
    load_executable_payoff_model,
    predict_executable_payoff_model,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round51_categorical_payoff_fincast import (  # noqa: E402
    _load_real_symbol_data,
)
from tools.run_round52_executable_support_hurdle import (  # noqa: E402
    CANDIDATES,
    SEEDS,
    SIDES,
    SYMBOLS,
    _ensemble_score,
    _policy_result,
    _prediction_metrics,
    _role_indexes,
    _threshold,
)


CONTROL = CANDIDATES[1]
VARIANTS = ("raw", "shrink_gain", "shrink_loss", "shrink_both")


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_from_npz(path: Path) -> ExecutablePayoffPredictionBatch:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return ExecutablePayoffPredictionBatch(
        architecture="sign_magnitude_hurdle",
        magnitude_floor_count=0,
        **arrays,
    )


def _shrinkage_coefficient(
    prediction: np.ndarray,
    actual: np.ndarray,
    *,
    anchor: float,
) -> float:
    centered_prediction = np.asarray(prediction, dtype=np.float64) - float(anchor)
    centered_actual = np.asarray(actual, dtype=np.float64) - float(anchor)
    denominator = float(np.dot(centered_prediction, centered_prediction))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    coefficient = float(np.dot(centered_prediction, centered_actual) / denominator)
    return float(np.clip(coefficient, 0.0, 1.0))


def _fit_side_shrinkage(
    *,
    model: TrainedExecutablePayoffModel,
    prediction: ExecutablePayoffPredictionBatch,
    actual: np.ndarray,
    side: str,
) -> dict[str, object]:
    executable = np.asarray(
        prediction.long_executable if side == "long" else prediction.short_executable,
        dtype=np.bool_,
    )
    payoff = np.asarray(actual, dtype=np.float64)
    gain_prediction = np.asarray(
        prediction.long_conditional_gain_bps
        if side == "long"
        else prediction.short_conditional_gain_bps,
        dtype=np.float64,
    )
    loss_prediction = np.asarray(
        prediction.long_conditional_loss_bps
        if side == "long"
        else prediction.short_conditional_loss_bps,
        dtype=np.float64,
    )
    positive = executable & (payoff > 0.0)
    non_positive = executable & ~positive
    if min(int(np.sum(positive)), int(np.sum(non_positive))) < 32:
        raise ValueError(f"{model.symbol} {side} shrinkage support is insufficient")
    gain_anchor = float(model.training_conditional_gain_mean_bps[side])
    loss_anchor = float(model.training_conditional_loss_mean_bps[side])
    return {
        "gain_alpha": _shrinkage_coefficient(
            gain_prediction[positive], payoff[positive], anchor=gain_anchor
        ),
        "loss_alpha": _shrinkage_coefficient(
            loss_prediction[non_positive],
            -payoff[non_positive],
            anchor=loss_anchor,
        ),
        "gain_anchor_bps": gain_anchor,
        "loss_anchor_bps": loss_anchor,
        "profitable_rows": int(np.sum(positive)),
        "non_profitable_rows": int(np.sum(non_positive)),
    }


def _transform_prediction(
    prediction: ExecutablePayoffPredictionBatch,
    *,
    calibration: Mapping[str, Mapping[str, object]],
    variant: str,
) -> ExecutablePayoffPredictionBatch:
    values: dict[str, np.ndarray] = {}
    for side in SIDES:
        probability = np.asarray(
            prediction.long_profitable_probability
            if side == "long"
            else prediction.short_profitable_probability,
            dtype=np.float64,
        )
        raw_gain = np.asarray(
            prediction.long_conditional_gain_bps
            if side == "long"
            else prediction.short_conditional_gain_bps,
            dtype=np.float64,
        )
        raw_loss = np.asarray(
            prediction.long_conditional_loss_bps
            if side == "long"
            else prediction.short_conditional_loss_bps,
            dtype=np.float64,
        )
        side_calibration = calibration[side]
        gain_alpha = (
            float(side_calibration["gain_alpha"])
            if variant in {"shrink_gain", "shrink_both"}
            else 1.0
        )
        loss_alpha = (
            float(side_calibration["loss_alpha"])
            if variant in {"shrink_loss", "shrink_both"}
            else 1.0
        )
        gain_anchor = float(side_calibration["gain_anchor_bps"])
        loss_anchor = float(side_calibration["loss_anchor_bps"])
        gain = np.maximum(gain_anchor + gain_alpha * (raw_gain - gain_anchor), 0.0)
        loss = np.maximum(loss_anchor + loss_alpha * (raw_loss - loss_anchor), 0.0)
        values[f"{side}_probability"] = probability
        values[f"{side}_gain"] = gain
        values[f"{side}_loss"] = loss
        values[f"{side}_expected"] = probability * gain - (1.0 - probability) * loss
    return ExecutablePayoffPredictionBatch(
        architecture=prediction.architecture,
        endpoint_indexes=np.asarray(prediction.endpoint_indexes, dtype=np.int64),
        long_expected_net_bps=values["long_expected"],
        short_expected_net_bps=values["short_expected"],
        long_executable=np.asarray(prediction.long_executable, dtype=np.bool_),
        short_executable=np.asarray(prediction.short_executable, dtype=np.bool_),
        long_profitable_probability=values["long_probability"],
        short_profitable_probability=values["short_probability"],
        long_conditional_gain_bps=values["long_gain"],
        short_conditional_gain_bps=values["short_gain"],
        long_conditional_loss_bps=values["long_loss"],
        short_conditional_loss_bps=values["short_loss"],
        magnitude_floor_count=0,
    )


def _slice_prediction(
    prediction: ExecutablePayoffPredictionBatch,
    selected: np.ndarray,
) -> ExecutablePayoffPredictionBatch:
    mask = np.asarray(selected, dtype=np.bool_)
    fields = {
        "endpoint_indexes": prediction.endpoint_indexes[mask],
        "long_expected_net_bps": prediction.long_expected_net_bps[mask],
        "short_expected_net_bps": prediction.short_expected_net_bps[mask],
        "long_executable": prediction.long_executable[mask],
        "short_executable": prediction.short_executable[mask],
        "long_profitable_probability": prediction.long_profitable_probability[mask],
        "short_profitable_probability": prediction.short_profitable_probability[mask],
        "long_conditional_gain_bps": prediction.long_conditional_gain_bps[mask],
        "short_conditional_gain_bps": prediction.short_conditional_gain_bps[mask],
        "long_conditional_loss_bps": prediction.long_conditional_loss_bps[mask],
        "short_conditional_loss_bps": prediction.short_conditional_loss_bps[mask],
    }
    return ExecutablePayoffPredictionBatch(
        architecture=prediction.architecture,
        magnitude_floor_count=0,
        **fields,
    )


def _policy_metrics(result: Mapping[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for scenario in ("base", "paired_stress"):
        scenario_result = result[scenario]
        output[scenario] = {
            "metrics": scenario_result["metrics"],
            "symbol_net_bps": scenario_result["symbol_net_bps"],
            "maximum_single_symbol_positive_pnl_share": scenario_result[
                "maximum_single_symbol_positive_pnl_share"
            ],
        }
    output["paired_stress_overlap_violations"] = result[
        "paired_stress_overlap_violations"
    ]
    return output


def run_probe(
    *,
    design_path: Path,
    round52_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    design = _read_object(design_path)
    round52 = _read_object(round52_report_path)
    resources = round52["runtime_resources"]
    data_contract = design["data_contract"]
    execution = design["execution_target"]
    source_states: dict[str, dict[str, object]] = {}
    calibrations: dict[str, object] = {}
    predictive: dict[str, dict[str, list[float]]] = {
        variant: {"mse_skill": [], "spearman": []} for variant in VARIANTS
    }

    for symbol in SYMBOLS:
        print(f"round53-probe source-load symbol={symbol}", flush=True)
        source = _load_real_symbol_data(
            symbol=symbol,
            warehouse_path=Path(str(resources["warehouse"])),
            cache_root=Path(str(resources["cache_root"])),
            memory_limit=str(resources["duckdb_memory_limit"]),
            threads=int(resources["duckdb_threads"]),
            data_contract=data_contract,
            execution=execution,
        )
        dataset = source["dataset"]
        targets = source["targets"]
        executable_dataset = build_executable_payoff_dataset(
            dataset, targets, target_scenario="base"
        )
        roles = _role_indexes(executable_dataset.payoff.decision_time_ms)
        source_states[symbol] = {
            "dataset": dataset,
            "targets": targets,
            "roles": roles,
            "predictions": {variant: {} for variant in VARIANTS},
        }
        calibrations[symbol] = {}

        for seed in SEEDS:
            model_path = Path(
                str(round52["models"][symbol][CONTROL][str(seed)]["artifact"]["path"])
            )
            model = load_executable_payoff_model(model_path)
            probability_prediction = predict_executable_payoff_model(
                model, executable_dataset, roles["probability_calibration"]
            )
            calibration: dict[str, Mapping[str, object]] = {}
            for side in SIDES:
                actual = np.asarray(
                    executable_dataset.payoff.long_net_bps
                    if side == "long"
                    else executable_dataset.payoff.short_net_bps,
                    dtype=np.float64,
                )[roles["probability_calibration"]]
                calibration[side] = _fit_side_shrinkage(
                    model=model,
                    prediction=probability_prediction,
                    actual=actual,
                    side=side,
                )
            calibrations[symbol][str(seed)] = calibration

            for role in ("policy_calibration", "evaluation"):
                artifact = round52["prediction_artifacts"][symbol][CONTROL][str(seed)][
                    role
                ]
                raw = _prediction_from_npz(Path(str(artifact["path"])))
                for variant in VARIANTS:
                    transformed = _transform_prediction(
                        raw, calibration=calibration, variant=variant
                    )
                    source_states[symbol]["predictions"][variant].setdefault(
                        role, []
                    ).append(transformed)
                    if role == "evaluation":
                        for side in SIDES:
                            metric = _prediction_metrics(
                                model=model,
                                dataset=executable_dataset,
                                indexes=roles[role],
                                prediction=transformed,
                                side=side,
                            )
                            predictive[variant]["mse_skill"].append(
                                float(metric["expected_payoff_mse_skill"])
                            )
                            predictive[variant]["spearman"].append(
                                float(metric["expected_payoff_spearman"])
                            )

        calibration_times = np.asarray(
            executable_dataset.payoff.decision_time_ms[roles["policy_calibration"]],
            dtype=np.int64,
        )
        day_number = calibration_times // 86_400_000
        unique_days = np.unique(day_number)
        if len(unique_days) != 2:
            raise ValueError(f"{symbol} policy calibration does not span two days")
        for variant in VARIANTS:
            full = source_states[symbol]["predictions"][variant][
                "policy_calibration"
            ]
            for day_index, day in enumerate(unique_days, start=1):
                role = f"policy_calibration_day_{day_index}"
                selected = day_number == day
                source_states[symbol]["predictions"][variant][role] = [
                    _slice_prediction(prediction, selected) for prediction in full
                ]

    coverages = tuple(
        float(value) for value in design["economic_screen"]["threshold_coverages"]
    )
    policies: dict[str, object] = {}
    for variant in VARIANTS:
        rows: list[dict[str, object]] = []
        for coverage in coverages:
            thresholds = {
                symbol: _threshold(
                    _ensemble_score(
                        source_states[symbol]["predictions"][variant][
                            "policy_calibration"
                        ]
                    ),
                    coverage,
                )
                for symbol in SYMBOLS
            }
            row = {
                "coverage": coverage,
                "thresholds_bps": thresholds,
                "policy_calibration": _policy_metrics(
                    _policy_result(
                        symbol_state=source_states,
                        candidate=variant,
                        role="policy_calibration",
                        thresholds=thresholds,
                    )
                ),
                "policy_calibration_day_1": _policy_metrics(
                    _policy_result(
                        symbol_state=source_states,
                        candidate=variant,
                        role="policy_calibration_day_1",
                        thresholds=thresholds,
                    )
                ),
                "policy_calibration_day_2": _policy_metrics(
                    _policy_result(
                        symbol_state=source_states,
                        candidate=variant,
                        role="policy_calibration_day_2",
                        thresholds=thresholds,
                    )
                ),
                "consumed_evaluation": _policy_metrics(
                    _policy_result(
                        symbol_state=source_states,
                        candidate=variant,
                        role="evaluation",
                        thresholds=thresholds,
                    )
                ),
            }
            rows.append(row)
        policies[variant] = rows

    predictive_summary = {
        variant: {
            name: {
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "observations": len(values),
            }
            for name, values in metrics.items()
        }
        for variant, metrics in predictive.items()
    }
    result: dict[str, object] = {
        "schema_version": "round-053-severity-shrinkage-prototype-v1",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "claims": {
            "development_only": True,
            "selection_contaminated": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "leverage_applied": False,
        },
        "source_round52_report": {
            "path": str(round52_report_path.resolve()),
            "file_sha256": _file_sha256(round52_report_path),
            "canonical_sha256": round52["report_canonical_sha256"],
        },
        "calibration": calibrations,
        "predictive_summary": predictive_summary,
        "policy_grid": policies,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    print(f"round53-probe complete output={output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--round52-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_probe(
        design_path=arguments.design,
        round52_report_path=arguments.round52_report,
        output_path=arguments.output,
    )
    print(json.dumps(result["predictive_summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
