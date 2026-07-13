"""Diagnose whether Round 53 CSM rank survives in the executable top tail."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from pathlib import Path
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
    build_executable_payoff_dataset,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ActionScoreBatch,
    BarrierActionTrace,
)
from simple_ai_trading.payoff_distribution_analysis import (  # noqa: E402
    base_and_paired_stress_traces,
    portfolio_trace_metrics,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round51_categorical_payoff_fincast import (  # noqa: E402
    _load_real_symbol_data,
)
from tools.run_round52_executable_support_hurdle import (  # noqa: E402
    _role_indexes,
)
from tools.run_round53_executable_csm import (  # noqa: E402
    CSM_CANDIDATES,
    DAY_MS,
    SEEDS,
    SYMBOLS,
    _canonical_sha256,
    _file_sha256,
    _read_object,
)


SCHEMA = "round-053-csm-rank-tail-diagnostic-v1"
EXPECTED_REPORT_FILE_SHA256 = (
    "5619ab8cd012c69465d8bfde04a0bc106927957f7c67d4b107587c1c5199501f"
)
EXPECTED_REPORT_CANONICAL_SHA256 = (
    "e21d35c332841ed53797f41ac111adf58e04e17d022d0e307b1d0ed620a5a658"
)
COVERAGES = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
AGGREGATIONS = ("worst_seed", "mean_seed", "mean_minus_one_seed_std")
ROLES = (
    "policy_calibration",
    "policy_calibration_day_1",
    "policy_calibration_day_2",
    "evaluation",
)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _progress(stage: str, **values: object) -> None:
    fields = " ".join(f"{name}={value}" for name, value in values.items())
    print(f"[round53-rank-tail] {stage} {fields}".rstrip(), flush=True)


def _validate_report(path: Path) -> dict[str, object]:
    if _file_sha256(path) != EXPECTED_REPORT_FILE_SHA256:
        raise ValueError("Round 53 report file drifted")
    report = _read_object(path, "Round 53 report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("schema_version")
        != "round-053-executable-csm-fincast-report-v1"
        or report.get("round") != 53
        or claimed != EXPECTED_REPORT_CANONICAL_SHA256
        or _canonical_sha256(canonical) != claimed
        or report.get("claims", {}).get("selection_contaminated") is not True
        or report.get("claims", {}).get("trading_authority") is not False
    ):
        raise ValueError("Round 53 report identity or claims drifted")
    return report


@dataclass(frozen=True)
class _Prediction:
    endpoint_indexes: np.ndarray
    long_expected_net_bps: np.ndarray
    short_expected_net_bps: np.ndarray
    long_executable: np.ndarray
    short_executable: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))


@dataclass(frozen=True)
class _RankScore:
    endpoint_indexes: np.ndarray
    side: np.ndarray
    raw_score_bps: np.ndarray
    supported: np.ndarray


def _load_prediction(
    artifact: Mapping[str, object],
    *,
    expected_endpoints: np.ndarray,
) -> _Prediction:
    path = Path(str(artifact["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(artifact["bytes"])
        or _file_sha256(path) != str(artifact["sha256"])
    ):
        raise ValueError(f"Round 53 prediction artifact drifted: {path}")
    with np.load(path, allow_pickle=False) as archive:
        prediction = _Prediction(
            endpoint_indexes=np.asarray(archive["endpoint_indexes"], dtype=np.int64),
            long_expected_net_bps=np.asarray(
                archive["long_expected_net_bps"], dtype=np.float64
            ),
            short_expected_net_bps=np.asarray(
                archive["short_expected_net_bps"], dtype=np.float64
            ),
            long_executable=np.asarray(archive["long_executable"], dtype=np.bool_),
            short_executable=np.asarray(archive["short_executable"], dtype=np.bool_),
        )
    arrays = (
        prediction.long_expected_net_bps,
        prediction.short_expected_net_bps,
        prediction.long_executable,
        prediction.short_executable,
    )
    if (
        prediction.rows == 0
        or not np.array_equal(prediction.endpoint_indexes, expected_endpoints)
        or any(array.shape != (prediction.rows,) for array in arrays)
        or not np.all(np.isfinite(prediction.long_expected_net_bps))
        or not np.all(np.isfinite(prediction.short_expected_net_bps))
    ):
        raise ValueError(f"Round 53 prediction contract drifted: {path}")
    return prediction


def _slice_prediction(prediction: _Prediction, mask: np.ndarray) -> _Prediction:
    selected = np.asarray(mask, dtype=np.bool_)
    return _Prediction(
        endpoint_indexes=prediction.endpoint_indexes[selected],
        long_expected_net_bps=prediction.long_expected_net_bps[selected],
        short_expected_net_bps=prediction.short_expected_net_bps[selected],
        long_executable=prediction.long_executable[selected],
        short_executable=prediction.short_executable[selected],
    )


def _aggregate(predictions: Sequence[_Prediction], method: str) -> _RankScore:
    if len(predictions) != len(SEEDS) or method not in AGGREGATIONS:
        raise ValueError("Round 53 rank aggregation contract is invalid")
    first = predictions[0]
    for prediction in predictions[1:]:
        if (
            not np.array_equal(prediction.endpoint_indexes, first.endpoint_indexes)
            or not np.array_equal(prediction.long_executable, first.long_executable)
            or not np.array_equal(prediction.short_executable, first.short_executable)
        ):
            raise ValueError("Round 53 ensemble prediction supports differ")
    long_stack = np.stack(
        [prediction.long_expected_net_bps for prediction in predictions]
    )
    short_stack = np.stack(
        [prediction.short_expected_net_bps for prediction in predictions]
    )
    if method == "worst_seed":
        long_score = np.min(long_stack, axis=0)
        short_score = np.min(short_stack, axis=0)
    elif method == "mean_seed":
        long_score = np.mean(long_stack, axis=0)
        short_score = np.mean(short_stack, axis=0)
    else:
        long_score = np.mean(long_stack, axis=0) - np.std(long_stack, axis=0)
        short_score = np.mean(short_stack, axis=0) - np.std(short_stack, axis=0)
    negative_infinity = np.full(first.rows, -np.inf, dtype=np.float64)
    executable_long = np.where(
        first.long_executable, long_score, negative_infinity
    )
    executable_short = np.where(
        first.short_executable, short_score, negative_infinity
    )
    choose_long = executable_long > executable_short
    choose_short = executable_short > executable_long
    side = np.zeros(first.rows, dtype=np.int8)
    side[choose_long] = 1
    side[choose_short] = -1
    raw_score = np.where(
        choose_long,
        executable_long,
        np.where(choose_short, executable_short, 0.0),
    )
    supported = (side != 0) & np.isfinite(raw_score)
    return _RankScore(
        endpoint_indexes=first.endpoint_indexes,
        side=side,
        raw_score_bps=raw_score,
        supported=supported,
    )


def _threshold(score: _RankScore, coverage: float) -> float:
    values = np.asarray(score.raw_score_bps[score.supported], dtype=np.float64)
    if len(values) < 32:
        raise ValueError("Round 53 rank score has insufficient supported rows")
    return float(np.quantile(values, 1.0 - coverage, method="higher"))


def _action_score(score: _RankScore, threshold_bps: float) -> ActionScoreBatch:
    selected = score.supported & (score.raw_score_bps >= threshold_bps)
    side = np.where(selected, score.side, 0).astype(np.int8)
    rank_margin = np.where(
        selected,
        1.0 + np.maximum(score.raw_score_bps - threshold_bps, 0.0),
        0.0,
    )
    return ActionScoreBatch(
        endpoint_indexes=score.endpoint_indexes,
        side=side,
        strength_bps=rank_margin.astype(np.float64),
        eligible=side != 0,
        profile="conservative",
    )


def _trace_summary(trace: BarrierActionTrace) -> dict[str, object]:
    return {
        "metrics": trace.asdict()["metrics"],
        "source_endpoint_indexes_sha256": _array_sha256(
            np.asarray(trace.source_endpoint_indexes, dtype=np.int64)
        ),
        "net_bps_sha256": _array_sha256(
            np.asarray(trace.net_bps, dtype=np.float64)
        ),
    }


def _opportunity_summary(
    score: _RankScore,
    threshold_bps: float,
    targets,
) -> dict[str, object]:
    selected = score.supported & (score.raw_score_bps >= threshold_bps)
    positions = np.searchsorted(targets.source_indexes, score.endpoint_indexes)
    if (
        np.any(positions >= targets.rows)
        or not np.array_equal(
            targets.source_indexes[positions], score.endpoint_indexes
        )
        or not np.all(targets.valid[positions])
    ):
        raise ValueError("Round 53 diagnostic target support drifted")
    actual = np.where(
        score.side == 1,
        targets.base_long_net_bps[positions],
        targets.base_short_net_bps[positions],
    )
    supported_actual = np.asarray(actual[score.supported], dtype=np.float64)
    selected_actual = np.asarray(actual[selected], dtype=np.float64)
    selected_score = np.asarray(score.raw_score_bps[selected], dtype=np.float64)
    return {
        "supported_rows": int(np.sum(score.supported)),
        "selected_rows": int(np.sum(selected)),
        "selected_raw_score_mean_bps": (
            float(np.mean(selected_score)) if len(selected_score) else None
        ),
        "selected_realized_base_mean_bps": (
            float(np.mean(selected_actual)) if len(selected_actual) else None
        ),
        "selected_realized_profitable_ratio": (
            float(np.mean(selected_actual > 0.0)) if len(selected_actual) else None
        ),
        "supported_realized_base_mean_bps": float(np.mean(supported_actual)),
        "selected_realized_lift_bps": (
            float(np.mean(selected_actual) - np.mean(supported_actual))
            if len(selected_actual)
            else None
        ),
        "selected_mean_calibration_error_bps": (
            float(np.mean(selected_score) - np.mean(selected_actual))
            if len(selected_actual)
            else None
        ),
        "selected_endpoint_indexes_sha256": _array_sha256(
            score.endpoint_indexes[selected]
        ),
    }


def _policy_result(
    states: Mapping[str, Mapping[str, object]],
    *,
    candidate: str,
    aggregation: str,
    role: str,
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    base_traces: dict[str, BarrierActionTrace] = {}
    stress_traces: dict[str, BarrierActionTrace] = {}
    symbols: dict[str, object] = {}
    overlap_violations = 0
    for symbol in SYMBOLS:
        state = states[symbol]
        score = state["scores"][candidate][aggregation][role]
        threshold = float(thresholds[symbol])
        action = _action_score(score, threshold)
        base, stress, overlap = base_and_paired_stress_traces(
            state["dataset"],
            state["targets"],
            action,
            extra_stress_slippage_bps_per_side=2.0,
        )
        base_traces[symbol] = base
        stress_traces[symbol] = stress
        overlap_violations += overlap
        symbols[symbol] = {
            "raw_score_threshold_bps": threshold,
            "opportunity_rows_are_overlapping_not_trades": True,
            "opportunities": _opportunity_summary(
                score, threshold, state["targets"]
            ),
            "base": _trace_summary(base),
            "paired_stress": _trace_summary(stress),
            "paired_stress_overlap_violations": overlap,
        }
    return {
        "candidate": candidate,
        "aggregation": aggregation,
        "role": role,
        "thresholds_bps": dict(thresholds),
        "symbols": symbols,
        "base": portfolio_trace_metrics(
            base_traces, symbol_weight=1.0 / len(SYMBOLS)
        ),
        "paired_stress": portfolio_trace_metrics(
            stress_traces, symbol_weight=1.0 / len(SYMBOLS)
        ),
        "paired_stress_overlap_violations": overlap_violations,
    }


def _positive_economics(result: Mapping[str, object]) -> bool:
    for scenario in ("base", "paired_stress"):
        metrics = result[scenario]["metrics"]
        if (
            int(metrics["trades"]) == 0
            or float(metrics["total_net_bps"]) <= 0.0
            or metrics["profit_factor"] is None
            or float(metrics["profit_factor"]) <= 1.0
        ):
            return False
    return True


def _calibration_screen(
    aggregate: Mapping[str, object],
    days: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    reasons: list[str] = []
    if int(aggregate["paired_stress"]["metrics"]["trades"]) < 8:
        reasons.append("aggregate_paired_stress_trades_below_8")
    if not _positive_economics(aggregate):
        reasons.append("aggregate_base_or_stress_economics_not_positive")
    symbol_pnl = aggregate["paired_stress"]["symbol_net_bps"]
    if sum(float(value) > 0.0 for value in symbol_pnl.values()) < 2:
        reasons.append("aggregate_positive_symbols_below_2")
    if float(
        aggregate["paired_stress"]["maximum_single_symbol_positive_pnl_share"]
    ) > 0.7:
        reasons.append("aggregate_positive_pnl_concentration_above_0.7")
    for index, day in enumerate(days, start=1):
        if int(day["paired_stress"]["metrics"]["trades"]) < 3:
            reasons.append(f"day_{index}_paired_stress_trades_below_3")
        if not _positive_economics(day):
            reasons.append(f"day_{index}_base_or_stress_economics_not_positive")
    return {"passed": not reasons, "reasons": reasons}


def diagnose(
    *,
    report_path: Path,
    design_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_path: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, object]:
    started = time.perf_counter()
    report = _validate_report(report_path)
    design = _read_object(design_path, "Round 53 frozen design")
    if design.get("design_sha256") != report.get("design_sha256"):
        raise ValueError("Round 53 diagnostic design identity drifted")
    data_contract = design["data_contract"]
    execution = design["execution_target"]
    states: dict[str, dict[str, object]] = {}

    for symbol in SYMBOLS:
        _progress("load", symbol=symbol)
        source = _load_real_symbol_data(
            symbol=symbol,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            memory_limit=memory_limit,
            threads=threads,
            data_contract=data_contract,
            execution=execution,
        )
        executable = build_executable_payoff_dataset(
            source["dataset"], source["targets"], target_scenario="base"
        )
        role_indexes = _role_indexes(executable.payoff.decision_time_ms)
        role_predictions: dict[str, dict[str, list[_Prediction]]] = {
            candidate: {} for candidate in CSM_CANDIDATES
        }
        for candidate in CSM_CANDIDATES:
            for role in ("policy_calibration", "evaluation"):
                expected_endpoints = np.asarray(
                    executable.payoff.source_row_indexes[role_indexes[role]],
                    dtype=np.int64,
                )
                role_predictions[candidate][role] = [
                    _load_prediction(
                        report["prediction_artifacts"][symbol][candidate][str(seed)][
                            role
                        ],
                        expected_endpoints=expected_endpoints,
                    )
                    for seed in SEEDS
                ]
            calibration = role_predictions[candidate]["policy_calibration"]
            times = source["dataset"].decision_time_ms[
                calibration[0].endpoint_indexes
            ]
            unique_days = np.unique(times // DAY_MS)
            if len(unique_days) != 2:
                raise ValueError(f"{symbol} diagnostic calibration days drifted")
            for day_index, day in enumerate(unique_days, start=1):
                mask = times // DAY_MS == day
                role_predictions[candidate][
                    f"policy_calibration_day_{day_index}"
                ] = [_slice_prediction(value, mask) for value in calibration]
        scores = {
            candidate: {
                aggregation: {
                    role: _aggregate(role_predictions[candidate][role], aggregation)
                    for role in ROLES
                }
                for aggregation in AGGREGATIONS
            }
            for candidate in CSM_CANDIDATES
        }
        states[symbol] = {
            "dataset": source["dataset"],
            "targets": source["targets"],
            "scores": scores,
            "dataset_sha256": source["dataset_sha256"],
            "executable_dataset_sha256": executable.dataset_sha256,
        }

    grid: list[dict[str, object]] = []
    for candidate in CSM_CANDIDATES:
        for aggregation in AGGREGATIONS:
            for coverage in COVERAGES:
                thresholds = {
                    symbol: _threshold(
                        states[symbol]["scores"][candidate][aggregation][
                            "policy_calibration"
                        ],
                        coverage,
                    )
                    for symbol in SYMBOLS
                }
                calibration = _policy_result(
                    states,
                    candidate=candidate,
                    aggregation=aggregation,
                    role="policy_calibration",
                    thresholds=thresholds,
                )
                days = [
                    _policy_result(
                        states,
                        candidate=candidate,
                        aggregation=aggregation,
                        role=f"policy_calibration_day_{index}",
                        thresholds=thresholds,
                    )
                    for index in (1, 2)
                ]
                screen = _calibration_screen(calibration, days)
                evaluation = _policy_result(
                    states,
                    candidate=candidate,
                    aggregation=aggregation,
                    role="evaluation",
                    thresholds=thresholds,
                )
                row = {
                    "candidate": candidate,
                    "aggregation": aggregation,
                    "coverage": coverage,
                    "thresholds_bps": thresholds,
                    "policy_calibration": calibration,
                    "policy_calibration_days": days,
                    "calibration_screen": screen,
                    "consumed_evaluation": evaluation,
                    "consumed_evaluation_positive_base_and_stress": (
                        _positive_economics(evaluation)
                    ),
                }
                grid.append(row)
                _progress(
                    "screen",
                    candidate=candidate,
                    aggregation=aggregation,
                    coverage=coverage,
                    calibration_passed=screen["passed"],
                    calibration_stress_trades=calibration["paired_stress"][
                        "metrics"
                    ]["trades"],
                    calibration_stress_mean=calibration["paired_stress"][
                        "metrics"
                    ]["mean_net_bps"],
                    evaluation_stress_mean=evaluation["paired_stress"]["metrics"][
                        "mean_net_bps"
                    ],
                )

    passing_calibration = [row for row in grid if row["calibration_screen"]["passed"]]
    stable_rows = [
        row
        for row in passing_calibration
        if row["consumed_evaluation_positive_base_and_stress"]
    ]
    result: dict[str, object] = {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "source_round_53": {
            "report_path": str(report_path.resolve()),
            "report_file_sha256": EXPECTED_REPORT_FILE_SHA256,
            "report_canonical_sha256": EXPECTED_REPORT_CANONICAL_SHA256,
        },
        "claims": {
            "selection_contaminated": True,
            "development_only": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
        },
        "method": {
            "purpose": (
                "Test whether Round 53 global rank correlation survives in the "
                "executable top tail after removing the positive-EV requirement."
            ),
            "threshold_source": "complete policy-calibration role only",
            "evaluation_status": "already consumed; diagnostic only",
            "aggregations": list(AGGREGATIONS),
            "coverages": list(COVERAGES),
            "raw_negative_scores_permitted_for_ranking": True,
            "replay_score_transform": (
                "Selected raw scores are converted to positive distance above the "
                "calibration threshold solely to satisfy the replay API contract."
            ),
            "opportunity_rows_are_overlapping_not_trades": True,
            "trades_are_exact_non_overlapping_100ms_barrier_replays": True,
        },
        "data": {
            symbol: {
                "dataset_sha256": states[symbol]["dataset_sha256"],
                "executable_dataset_sha256": states[symbol][
                    "executable_dataset_sha256"
                ],
            }
            for symbol in SYMBOLS
        },
        "grid": grid,
        "diagnosis": {
            "calibration_rows_passed": len(passing_calibration),
            "calibration_and_consumed_evaluation_rows_positive": len(stable_rows),
            "ordinal_followup_supported": bool(stable_rows),
            "next_round_authorized": False,
            "reason": (
                "A rank-tail follow-up is diagnostically supported only when at "
                "least one fixed rule passes the prior calibration screen and has "
                "positive base and paired-stress economics on the consumed interval."
            ),
        },
    }
    result["report_canonical_sha256"] = _canonical_sha256(result)
    write_json_atomic(output_path, result, indent=2, sort_keys=True)
    _progress(
        "complete",
        seconds=f"{result['runtime_seconds']:.1f}",
        calibration_passes=len(passing_calibration),
        stable=len(stable_rows),
        output=output_path,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "model-research"
            / "action-value"
            / "round-053-executable-csm-fincast-design.json"
        ),
    )
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    diagnose(
        report_path=arguments.report,
        design_path=arguments.design,
        warehouse_path=arguments.warehouse,
        cache_root=arguments.cache_root,
        output_path=arguments.output,
        memory_limit=arguments.memory_limit,
        threads=arguments.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
