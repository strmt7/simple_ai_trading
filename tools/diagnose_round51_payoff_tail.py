"""Diagnose Round 51 score tails against exact frozen replay outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
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

from simple_ai_trading.categorical_payoff_lightgbm import (  # noqa: E402
    build_categorical_payoff_dataset,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ActionScoreBatch,
    BarrierActionTrace,
)
from simple_ai_trading.payoff_distribution_analysis import (  # noqa: E402
    base_and_paired_stress_traces,
    portfolio_trace_metrics,
)
from tools.run_round51_categorical_payoff_fincast import (  # noqa: E402
    _load_real_symbol_data,
    _role_indexes,
)


ROUND = 51
SCHEMA = "round-051-payoff-tail-failure-analysis-v1"
EXPECTED_REPORT_CANONICAL_SHA256 = (
    "b97a12764256680402d526fd17ee56999c7f88335d66570a196aae3d0e9d0201"
)
EXPECTED_REPORT_FILE_SHA256 = (
    "d2e6c2e1a8ba0a48293f124d148359e4015f9c25aa61fb11b9dc7578d7975a80"
)
EXPECTED_DESIGN_SHA256 = (
    "42f9afbda8755807e898fa8bb54ad4039d1f9f6b2f4d6c825afc0b1d02bcfba3"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CANDIDATES = (
    "direct_mean_lightgbm",
    "categorical_payoff_lightgbm",
    "categorical_payoff_lightgbm_fincast",
)
SEEDS = (5101, 5102, 5103)
SIDES = ("long", "short")
RANK_DEPTHS = (25, 50, 100, 250, 500, 1_000, 2_500, 5_000)
OUTCOME_NAMES = {
    0: "horizon",
    1: "stop",
    2: "take",
    3: "ambiguous_stop",
    4: "protection_gap_stop",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


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
        raise ValueError(f"{label} must be an object")
    return value


def _canonical_identity(value: Mapping[str, object], field: str, label: str) -> str:
    canonical = dict(value)
    claimed = canonical.pop(field, None)
    if not isinstance(claimed, str) or _canonical_sha256(canonical) != claimed:
        raise ValueError(f"{label} canonical identity is invalid")
    return claimed


def _progress(stage: str, **fields: object) -> None:
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"round51-diagnosis {stage}{(' ' + suffix) if suffix else ''}", flush=True)


def _validated_inputs(
    *, design_path: Path, report_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    design = _read_object(design_path, "Round 51 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 51 design")
    report = _read_object(report_path, "Round 51 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 51 report"
    )
    if (
        design_sha != EXPECTED_DESIGN_SHA256
        or report_sha != EXPECTED_REPORT_CANONICAL_SHA256
        or _file_sha256(report_path) != EXPECTED_REPORT_FILE_SHA256
        or report.get("design_sha256") != design_sha
        or report.get("round") != ROUND
        or report.get("round_gate", {}).get("passed") is not False
        or report.get("claims", {}).get("selection_contaminated") is not True
        or report.get("claims", {}).get("profitability_claim") is not False
        or report.get("claims", {}).get("trading_authority") is not False
        or report.get("claims", {}).get("leverage_applied") is not False
    ):
        raise ValueError("Round 51 diagnosis lineage or claims are invalid")
    return design, report


def _verified_prediction(item: Mapping[str, object]) -> dict[str, np.ndarray]:
    path = Path(str(item.get("path") or ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(item.get("bytes") or -1)
        or _file_sha256(path) != item.get("sha256")
    ):
        raise ValueError(f"Round 51 prediction artifact drifted: {path}")
    with np.load(path, allow_pickle=False) as source:
        values = {name: np.asarray(source[name]) for name in source.files}
    if not values or any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("Round 51 prediction artifact contains invalid arrays")
    return values


def _candidate_scores(
    candidate: str,
    models: Sequence[Mapping[str, object]],
    expected_endpoints: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if [int(model["seed"]) for model in models] != list(SEEDS):
        raise ValueError("Round 51 diagnosis seed order drifted")
    means: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    probabilities: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    for model in models:
        values = _verified_prediction(model["prediction"])
        endpoints = np.asarray(values["endpoint_indexes"], dtype=np.int64)
        if not np.array_equal(endpoints, expected_endpoints):
            raise ValueError("Round 51 diagnosis prediction endpoints drifted")
        for side in SIDES:
            means[side].append(np.asarray(values[f"{side}_mean_bps"], dtype=np.float64))
            probability_name = f"{side}_profitable_probability"
            if probability_name in values:
                probabilities[side].append(
                    np.asarray(values[probability_name], dtype=np.float64)
                )
    output = {
        "expected_mean": (
            np.mean(np.stack(means["long"]), axis=0),
            np.mean(np.stack(means["short"]), axis=0),
        ),
        "expected_worst_seed": (
            np.min(np.stack(means["long"]), axis=0),
            np.min(np.stack(means["short"]), axis=0),
        ),
    }
    if candidate != "direct_mean_lightgbm":
        if len(probabilities["long"]) != len(SEEDS):
            raise ValueError(
                "Round 51 categorical probability artifacts are incomplete"
            )
        output["profitable_probability_mean"] = (
            np.mean(np.stack(probabilities["long"]), axis=0),
            np.mean(np.stack(probabilities["short"]), axis=0),
        )
    return output


def _ranked_action(
    long_score: np.ndarray,
    short_score: np.ndarray,
    endpoint_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    long_values = np.asarray(long_score, dtype=np.float64)
    short_values = np.asarray(short_score, dtype=np.float64)
    endpoints = np.asarray(endpoint_indexes, dtype=np.int64)
    if (
        long_values.shape != short_values.shape
        or long_values.shape != endpoints.shape
        or long_values.ndim != 1
        or len(long_values) == 0
        or not np.all(np.isfinite(long_values))
        or not np.all(np.isfinite(short_values))
        or np.any(np.diff(endpoints) <= 0)
    ):
        raise ValueError("Round 51 diagnosis scores are invalid")
    side = np.zeros(len(endpoints), dtype=np.int8)
    side[long_values > short_values] = 1
    side[short_values > long_values] = -1
    score = np.maximum(long_values, short_values)
    eligible = side != 0
    positions = np.flatnonzero(eligible)
    order = positions[np.lexsort((endpoints[positions], -score[positions]))]
    return side, score, order


def _profit_factor(values: np.ndarray) -> float | None:
    positive = float(np.sum(values[values > 0.0]))
    negative = float(-np.sum(values[values < 0.0]))
    if negative <= 0.0:
        return None
    return positive / negative


def _maximum_additive_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    cumulative = np.cumsum(values, dtype=np.float64)
    peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
    return float(np.max(peak - cumulative, initial=0.0))


def _raw_rank_metrics(
    *,
    selected_positions: np.ndarray,
    side: np.ndarray,
    base_long: np.ndarray,
    base_short: np.ndarray,
    base_long_outcome: np.ndarray,
    base_short_outcome: np.ndarray,
    long_liquidity_eligible: np.ndarray,
    short_liquidity_eligible: np.ndarray,
    decision_time_ms: np.ndarray,
    explicit_round_trip_cost_bps: float,
) -> dict[str, object]:
    selected = np.sort(np.asarray(selected_positions, dtype=np.int64))
    chosen_side = side[selected]
    values = np.where(
        chosen_side == 1,
        base_long[selected],
        base_short[selected],
    ).astype(np.float64)
    outcomes = np.where(
        chosen_side == 1,
        base_long_outcome[selected],
        base_short_outcome[selected],
    ).astype(np.int8)
    executable = np.where(
        chosen_side == 1,
        long_liquidity_eligible[selected],
        short_liquidity_eligible[selected],
    ).astype(bool)
    day_ids = decision_time_ms[selected] // 86_400_000
    daily = [
        {
            "utc_day_id": int(day),
            "rows": int(np.sum(day_ids == day)),
            "mean_net_bps": float(np.mean(values[day_ids == day])),
            "total_net_bps": float(np.sum(values[day_ids == day])),
        }
        for day in np.unique(day_ids)
    ]
    counts = Counter(int(value) for value in outcomes)
    return {
        "rows": len(selected),
        "long_rows": int(np.sum(chosen_side == 1)),
        "short_rows": int(np.sum(chosen_side == -1)),
        "executable_rows": int(np.sum(executable)),
        "executable_ratio": float(np.mean(executable)),
        "executable_mean_net_bps": (
            float(np.mean(values[executable])) if np.any(executable) else None
        ),
        "executable_positive_ratio": (
            float(np.mean(values[executable] > 0.0)) if np.any(executable) else None
        ),
        "positive_rows": int(np.sum(values > 0.0)),
        "positive_ratio": float(np.mean(values > 0.0)),
        "mean_net_bps": float(np.mean(values)),
        "median_net_bps": float(np.median(values)),
        "total_net_bps": float(np.sum(values)),
        "profit_factor": _profit_factor(values),
        "maximum_additive_drawdown_bps": _maximum_additive_drawdown(values),
        "mean_after_adding_frozen_explicit_cost_bps": float(
            np.mean(values) + explicit_round_trip_cost_bps
        ),
        "break_even_explicit_round_trip_cost_bps": float(
            np.mean(values) + explicit_round_trip_cost_bps
        ),
        "outcomes": {name: counts.get(code, 0) for code, name in OUTCOME_NAMES.items()},
        "daily": daily,
    }


def _trace_summary(trace: BarrierActionTrace) -> dict[str, object]:
    values = np.asarray(trace.net_bps, dtype=np.float64)
    timestamps = np.asarray(trace.exit_times_ms, dtype=np.int64)
    daily = [
        {
            "utc_day_id": int(day),
            "trades": int(np.sum(timestamps // 86_400_000 == day)),
            "total_net_bps": float(np.sum(values[timestamps // 86_400_000 == day])),
        }
        for day in np.unique(timestamps // 86_400_000)
    ]
    return {
        "metrics": trace.asdict()["metrics"],
        "daily": daily,
        "source_endpoint_indexes_sha256": hashlib.sha256(
            np.asarray(trace.source_endpoint_indexes, dtype=np.int64).tobytes()
        ).hexdigest(),
    }


def _rank_bins(
    *,
    score: np.ndarray,
    order: np.ndarray,
    side: np.ndarray,
    base_long: np.ndarray,
    base_short: np.ndarray,
    long_liquidity_eligible: np.ndarray,
    short_liquidity_eligible: np.ndarray,
) -> list[dict[str, object]]:
    bins = np.array_split(order, 10)
    rows: list[dict[str, object]] = []
    for rank, selected in enumerate(bins, start=1):
        values = np.where(
            side[selected] == 1,
            base_long[selected],
            base_short[selected],
        )
        executable = np.where(
            side[selected] == 1,
            long_liquidity_eligible[selected],
            short_liquidity_eligible[selected],
        )
        rows.append(
            {
                "rank_bin": rank,
                "rank_direction": "best_to_worst",
                "rows": len(selected),
                "minimum_score": float(np.min(score[selected])),
                "maximum_score": float(np.max(score[selected])),
                "mean_score": float(np.mean(score[selected])),
                "mean_net_bps": float(np.mean(values)),
                "positive_ratio": float(np.mean(values > 0.0)),
                "executable_ratio": float(np.mean(executable)),
                "executable_mean_net_bps": (
                    float(np.mean(values[executable])) if np.any(executable) else None
                ),
            }
        )
    return rows


def _score_batch(
    *, endpoint_indexes: np.ndarray, side: np.ndarray, selected: np.ndarray
) -> ActionScoreBatch:
    selected_side = np.zeros(len(endpoint_indexes), dtype=np.int8)
    strength = np.zeros(len(endpoint_indexes), dtype=np.float64)
    selected_side[selected] = side[selected]
    strength[selected] = 1.0
    return ActionScoreBatch(
        endpoint_indexes=np.asarray(endpoint_indexes, dtype=np.int64),
        side=selected_side,
        strength_bps=strength,
        eligible=selected_side != 0,
        profile="conservative",
    )


def diagnose(
    *,
    design_path: Path,
    report_path: Path,
    warehouse_path: Path,
    cache_root: Path,
    output_path: Path,
    memory_limit: str,
    threads: int,
) -> dict[str, object]:
    started = time.perf_counter()
    design, report = _validated_inputs(
        design_path=design_path,
        report_path=report_path,
    )
    data_contract = design["data_contract"]
    execution = design["execution_target"]
    roles_contract = design["chronological_roles"]
    explicit_round_trip_cost_bps = 2.0 * (
        float(execution["taker_fee_bps_per_side"])
        + float(execution["additional_slippage_bps_per_side"])
    )
    symbol_results: dict[str, object] = {}
    traces: dict[tuple[str, str, int, str], BarrierActionTrace] = {}

    for symbol in SYMBOLS:
        symbol_started = time.perf_counter()
        _progress("symbol-start", symbol=symbol)
        real = _load_real_symbol_data(
            symbol=symbol,
            warehouse_path=warehouse_path,
            cache_root=cache_root,
            memory_limit=memory_limit,
            threads=threads,
            data_contract=data_contract,
            execution=execution,
        )
        dataset = real["dataset"]
        targets = real["targets"]
        expected = report["data"][symbol]
        if (
            real["dataset_sha256"] != expected["microstructure_dataset_sha256"]
            or targets.valid_rows != expected["valid_barrier_rows"]
            or _canonical_sha256(targets.summary())
            != _canonical_sha256(expected["barrier_summary"])
        ):
            raise ValueError(f"Round 51 {symbol} reconstructed evidence drifted")
        payoff = build_categorical_payoff_dataset(
            dataset,
            targets,
            target_scenario="base",
        )
        if payoff.dataset_sha256 != expected["deterministic_dataset_sha256"]:
            raise ValueError(f"Round 51 {symbol} payoff dataset drifted")
        roles = _role_indexes(payoff.decision_time_ms, roles_contract)
        evaluation = np.asarray(roles["evaluation"], dtype=np.int64)
        endpoints = np.asarray(payoff.source_row_indexes[evaluation], dtype=np.int64)
        base_long = np.asarray(payoff.long_net_bps[evaluation], dtype=np.float64)
        base_short = np.asarray(payoff.short_net_bps[evaluation], dtype=np.float64)
        target_positions = np.searchsorted(targets.source_indexes, endpoints)
        if np.any(target_positions >= targets.rows) or not np.array_equal(
            targets.source_indexes[target_positions], endpoints
        ):
            raise ValueError("Round 51 diagnosis target alignment failed")
        base_long_outcome = np.asarray(
            targets.base_long_outcome[target_positions], dtype=np.int8
        )
        base_short_outcome = np.asarray(
            targets.base_short_outcome[target_positions], dtype=np.int8
        )
        long_liquidity_eligible = np.asarray(
            dataset.long_liquidity_eligible[endpoints], dtype=bool
        )
        short_liquidity_eligible = np.asarray(
            dataset.short_liquidity_eligible[endpoints], dtype=bool
        )
        decision_times = np.asarray(payoff.decision_time_ms[evaluation], dtype=np.int64)
        candidates: dict[str, object] = {}
        for candidate in CANDIDATES:
            _progress("candidate", symbol=symbol, candidate=candidate)
            models = report["symbol_results"][symbol][candidate]["models"]
            variants = _candidate_scores(candidate, models, endpoints)
            variant_reports: dict[str, object] = {}
            for variant, (long_score, short_score) in variants.items():
                side, combined_score, order = _ranked_action(
                    long_score,
                    short_score,
                    endpoints,
                )
                curves: list[dict[str, object]] = []
                for depth in RANK_DEPTHS:
                    selected = order[: min(depth, len(order))]
                    score = _score_batch(
                        endpoint_indexes=endpoints,
                        side=side,
                        selected=selected,
                    )
                    base_trace, stress_trace, overlaps = base_and_paired_stress_traces(
                        dataset,
                        targets,
                        score,
                        extra_stress_slippage_bps_per_side=2.0,
                    )
                    traces[(candidate, variant, depth, symbol)] = base_trace
                    traces[(candidate, variant + "__stress", depth, symbol)] = (
                        stress_trace
                    )
                    curves.append(
                        {
                            "rank_depth": depth,
                            "raw_ranked": _raw_rank_metrics(
                                selected_positions=selected,
                                side=side,
                                base_long=base_long,
                                base_short=base_short,
                                base_long_outcome=base_long_outcome,
                                base_short_outcome=base_short_outcome,
                                long_liquidity_eligible=long_liquidity_eligible,
                                short_liquidity_eligible=short_liquidity_eligible,
                                decision_time_ms=decision_times,
                                explicit_round_trip_cost_bps=explicit_round_trip_cost_bps,
                            ),
                            "base_nonoverlapping": _trace_summary(base_trace),
                            "paired_stress_nonoverlapping": _trace_summary(
                                stress_trace
                            ),
                            "paired_stress_overlap_violations": overlaps,
                        }
                    )
                variant_reports[variant] = {
                    "rows": len(order),
                    "best_score": float(combined_score[order[0]]),
                    "worst_score": float(combined_score[order[-1]]),
                    "rank_bins": _rank_bins(
                        score=combined_score,
                        order=order,
                        side=side,
                        base_long=base_long,
                        base_short=base_short,
                        long_liquidity_eligible=long_liquidity_eligible,
                        short_liquidity_eligible=short_liquidity_eligible,
                    ),
                    "curves": curves,
                }
            candidates[candidate] = variant_reports
        symbol_results[symbol] = {
            "evaluation_rows": len(evaluation),
            "dataset_sha256": real["dataset_sha256"],
            "payoff_dataset_sha256": payoff.dataset_sha256,
            "evaluation_liquidity_support": {
                "long_eligible_rows": int(np.sum(long_liquidity_eligible)),
                "long_eligible_ratio": float(np.mean(long_liquidity_eligible)),
                "short_eligible_rows": int(np.sum(short_liquidity_eligible)),
                "short_eligible_ratio": float(np.mean(short_liquidity_eligible)),
                "either_side_eligible_rows": int(
                    np.sum(long_liquidity_eligible | short_liquidity_eligible)
                ),
                "either_side_eligible_ratio": float(
                    np.mean(long_liquidity_eligible | short_liquidity_eligible)
                ),
            },
            "candidates": candidates,
            "runtime_seconds": time.perf_counter() - symbol_started,
        }
        _progress(
            "symbol-complete",
            symbol=symbol,
            seconds=f"{time.perf_counter() - symbol_started:.1f}",
        )

    portfolios: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        variants = symbol_results[SYMBOLS[0]]["candidates"][candidate]
        for variant in variants:
            for depth in RANK_DEPTHS:
                base = portfolio_trace_metrics(
                    {
                        symbol: traces[(candidate, variant, depth, symbol)]
                        for symbol in SYMBOLS
                    },
                    symbol_weight=1.0 / len(SYMBOLS),
                )
                stress = portfolio_trace_metrics(
                    {
                        symbol: traces[(candidate, variant + "__stress", depth, symbol)]
                        for symbol in SYMBOLS
                    },
                    symbol_weight=1.0 / len(SYMBOLS),
                )
                portfolios.append(
                    {
                        "candidate": candidate,
                        "ranking_variant": variant,
                        "rank_depth_per_symbol": depth,
                        "base": base,
                        "paired_stress": stress,
                    }
                )
    best = max(
        portfolios,
        key=lambda row: float(row["base"]["metrics"]["total_net_bps"]),
    )
    positive_base = sum(
        float(row["base"]["metrics"]["total_net_bps"]) > 0.0 for row in portfolios
    )
    positive_stress = sum(
        float(row["paired_stress"]["metrics"]["total_net_bps"]) > 0.0
        for row in portfolios
    )
    best_base_metrics = best["base"]["metrics"]
    best_stress_metrics = best["paired_stress"]["metrics"]
    sol_support = symbol_results["SOLUSDT"]["evaluation_liquidity_support"]
    analysis: dict[str, object] = {
        "schema_version": SCHEMA,
        "round": ROUND,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "source_report_canonical_sha256": EXPECTED_REPORT_CANONICAL_SHA256,
        "source_report_file_sha256": EXPECTED_REPORT_FILE_SHA256,
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "diagnostic_grid": {
            "rank_depths_per_symbol": list(RANK_DEPTHS),
            "ranking_uses_outcomes": False,
            "best_row_selection_uses_consumed_evaluation_outcomes": True,
            "multiple_comparisons": len(portfolios),
            "explicit_round_trip_cost_bps": explicit_round_trip_cost_bps,
            "stress_extra_slippage_bps_per_side": 2.0,
        },
        "symbol_results": symbol_results,
        "portfolio_curves": portfolios,
        "summary": {
            "portfolio_variants": len(portfolios),
            "positive_base_variants": positive_base,
            "positive_paired_stress_variants": positive_stress,
            "best_consumed_base_variant": best,
            "strict_round51_selected_trades": 0,
            "findings": {
                "payoff_training_was_not_action_support_masked": True,
                "sol_either_side_liquidity_eligible_ratio": sol_support[
                    "either_side_eligible_ratio"
                ],
                "best_grid_base_trades": best_base_metrics["trades"],
                "best_grid_base_mean_net_bps": best_base_metrics["mean_net_bps"],
                "best_grid_base_profit_factor": best_base_metrics["profit_factor"],
                "best_grid_base_max_drawdown_bps": best_base_metrics[
                    "max_drawdown_bps"
                ],
                "best_grid_stress_mean_net_bps": best_stress_metrics["mean_net_bps"],
                "best_grid_stress_profit_factor": best_stress_metrics["profit_factor"],
                "best_grid_single_symbol_positive_pnl_share": best["base"][
                    "maximum_single_symbol_positive_pnl_share"
                ],
                "best_grid_is_stress_positive": bool(
                    float(best_stress_metrics["total_net_bps"]) > 0.0
                ),
                "best_grid_meets_round51_minimum_trades": bool(
                    int(best_base_metrics["trades"]) >= 30
                ),
                "best_grid_meets_round51_diversification": bool(
                    float(best["base"]["maximum_single_symbol_positive_pnl_share"])
                    <= 0.70
                ),
            },
            "next_model_requirements": [
                "Apply side-specific liquidity support masks to training, early stopping, calibration, scoring, and every proper-score baseline.",
                "Separate profitable-event probability from conditional gain and loss magnitude, then recombine them into exact after-cost expected utility.",
                "Select any rank depth or abstention threshold only on a chronological calibration role; evaluation may never choose it.",
                "Require positive paired-stress economics, at least 30 non-overlapping trades, two positive symbols, and bounded concentration before expansion.",
                "Treat FinCast as an ablation until it improves both proper scores and stress economics on executable rows.",
                "Do not infer passive-order economics until full-depth queue and fill evidence exists.",
            ],
            "interpretation": (
                "Post-failure diagnostics on a consumed evaluation interval cannot "
                "promote, tune, or establish a profitable strategy."
            ),
        },
        "selection_contaminated": True,
        "development_only": True,
        "promotion_permitted": False,
        "profitability_claim": False,
        "trading_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "leverage_applied": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _progress(
        "complete",
        portfolios=len(portfolios),
        positive_base=positive_base,
        positive_stress=positive_stress,
        seconds=f"{time.perf_counter() - started:.1f}",
        output=output_path,
    )
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path(
            "docs/model-research/action-value/round-051-categorical-payoff-fincast-design.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round51-categorical-payoff-fincast-20260713-v2\report.json"
        ),
    )
    parser.add_argument(
        "--warehouse",
        type=Path,
        default=Path(r"E:\SimpleAITradingData\tick-warehouse\round51-screen.duckdb"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(r"E:\SimpleAITradingData\tick-warehouse\archive-cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/model-research/action-value/round-051-failure-analysis.json"
        ),
    )
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.threads < 1 or args.threads > 64:
        raise ValueError("Round 51 diagnosis DuckDB threads must lie in [1, 64]")
    diagnose(
        design_path=args.design,
        report_path=args.report,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_path=args.output,
        memory_limit=str(args.memory_limit).upper(),
        threads=args.threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
