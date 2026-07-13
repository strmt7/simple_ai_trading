#!/usr/bin/env python3
"""Diagnose Round 54 without reading its withheld evaluation role."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
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

from simple_ai_trading.distributional_tcn_model import (  # noqa: E402
    HORIZONS,
    role_mask,
)
from simple_ai_trading.sequential_action_value_model import (  # noqa: E402
    ACTIONS,
    POLICY_IDS,
    SequentialQSpec,
    consensus_policy_actions,
    policy_score,
    replay_consensus_actions,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round54_sequential_action_value_prototype import (  # noqa: E402
    _canonical_sha256,
    _file_sha256,
    _load_dataset,
    _read_object,
    _validate_source_binding,
)


ROUND = 54
SOURCE_SCHEMA = "round-054-sequential-action-value-prototype-report-v1"
DIAGNOSTIC_SCHEMA = "round-054-sequential-failure-diagnostic-v1"
SOURCE_REPORT_CANONICAL_SHA256 = (
    "8aff454b63201ccab8a53d7a9cf4a9f46bad04a284074c7e7a22e001fb9385ff"
)
SOURCE_REPORT_FILE_SHA256 = (
    "e1a6e53d4be1d21ef101f7172083740af5ad332ec8933ca445625b32f5085642"
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _validate_report(path: Path) -> dict[str, object]:
    if _file_sha256(path) != SOURCE_REPORT_FILE_SHA256:
        raise ValueError("Round 54 source report file hash drifted")
    report = _read_object(path, "Round 54 source report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    actual = _canonical_sha256(canonical)
    if (
        report.get("schema_version") != SOURCE_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected_before_evaluation"
        or report.get("evaluation_replay_performed") is not False
        or claimed != actual
        or actual != SOURCE_REPORT_CANONICAL_SHA256
    ):
        raise ValueError("Round 54 source report identity or boundary is invalid")
    return report


def _verified_prediction_path(
    report: Mapping[str, object],
    source_root: Path,
) -> Path:
    rows = report.get("evidence_artifacts")
    if not isinstance(rows, list):
        raise ValueError("Round 54 source report has no artifact manifest")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and Path(str(row.get("path", ""))).name == "seed_q_bps.npy"
    ]
    if len(matches) != 1:
        raise ValueError("Round 54 source prediction artifact is not unique")
    row = matches[0]
    path = source_root / "seed_q_bps.npy"
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or _file_sha256(path) != str(row.get("sha256", ""))
    ):
        raise ValueError("Round 54 source prediction artifact drifted")
    return path


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    result = spearmanr(
        np.asarray(left, dtype=np.float64).ravel(),
        np.asarray(right, dtype=np.float64).ravel(),
    )
    value = float(result.statistic)
    return value if math.isfinite(value) else 0.0


def _direction_rows(
    seed_q_bps: np.ndarray,
    dataset,
    calibration_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy_id in POLICY_IDS:
        scores = np.median(policy_score(seed_q_bps, policy_id), axis=0)
        directional = scores[:, :, 1, 2] - scores[:, :, 1, 0]
        for horizon_index, horizon in enumerate(HORIZONS):
            actual = dataset.forward_return_bps[:, :, horizon_index]
            rows.append(
                {
                    "policy_id": policy_id,
                    "horizon_hours": horizon,
                    "symbol": "ALL",
                    "rows": int(np.count_nonzero(calibration_mask)) * 3,
                    "directional_spearman": _finite_spearman(
                        directional[calibration_mask],
                        actual[calibration_mask],
                    ),
                }
            )
            for symbol_index, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
                rows.append(
                    {
                        "policy_id": policy_id,
                        "horizon_hours": horizon,
                        "symbol": symbol,
                        "rows": int(np.count_nonzero(calibration_mask)),
                        "directional_spearman": _finite_spearman(
                            directional[calibration_mask, symbol_index],
                            actual[calibration_mask, symbol_index],
                        ),
                    }
                )
    return rows


def _position_runs(
    actions: np.ndarray,
    timestamps_ms: np.ndarray,
    calibration_mask: np.ndarray,
    policy_id: str,
) -> list[dict[str, object]]:
    indexes = np.flatnonzero(calibration_mask)
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        values = actions[indexes, symbol_index]
        start = 0
        for end in range(1, values.size + 1):
            if end < values.size and values[end] == values[start]:
                continue
            side = int(values[start])
            if side != 0:
                rows.append(
                    {
                        "policy_id": policy_id,
                        "symbol": symbol,
                        "side": side,
                        "hours": end - start,
                        "start_utc": datetime.fromtimestamp(
                            int(timestamps_ms[indexes[start]]) / 1_000,
                            tz=UTC,
                        ).isoformat(),
                        "end_utc": datetime.fromtimestamp(
                            int(timestamps_ms[indexes[end - 1]]) / 1_000,
                            tz=UTC,
                        ).isoformat(),
                    }
                )
            start = end
    return rows


def _age_capped_actions(
    seed_q_bps: np.ndarray,
    calibration_mask: np.ndarray,
    policy_id: str,
    maximum_holding_hours: int,
) -> np.ndarray:
    scores = policy_score(seed_q_bps, policy_id)
    indexes = np.flatnonzero(calibration_mask)
    actions = np.zeros((calibration_mask.size, 3), dtype=np.int8)
    positions = np.zeros(3, dtype=np.int8)
    ages = np.zeros(3, dtype=np.int16)
    for index in indexes:
        for symbol_index in range(3):
            if positions[symbol_index] != 0 and ages[symbol_index] >= maximum_holding_hours:
                positions[symbol_index] = 0
                ages[symbol_index] = 0
                actions[index, symbol_index] = 0
                continue
            previous_position = int(positions[symbol_index])
            previous_index = previous_position + 1
            seed_scores = scores[:, index, symbol_index, previous_index, :]
            seed_choices = np.argmax(seed_scores, axis=1)
            ensemble_choice = int(np.argmax(np.median(seed_scores, axis=0)))
            advantages = (
                seed_scores[:, ensemble_choice] - seed_scores[:, previous_index]
            )
            if np.all(seed_choices == ensemble_choice) and float(np.min(advantages)) > 0.0:
                positions[symbol_index] = ACTIONS[ensemble_choice]
            if positions[symbol_index] == 0:
                ages[symbol_index] = 0
            elif int(positions[symbol_index]) != previous_position:
                ages[symbol_index] = 1
            else:
                ages[symbol_index] += 1
            actions[index, symbol_index] = positions[symbol_index]
    return actions


def _finite_hold_rows(
    seed_q_bps: np.ndarray,
    dataset,
    calibration_mask: np.ndarray,
) -> list[dict[str, object]]:
    diagnostic_spec = SequentialQSpec(bootstrap_samples=2_000)
    rows: list[dict[str, object]] = []
    for policy_index, policy_id in enumerate(POLICY_IDS):
        for maximum_holding_hours in (4, 8, 12, 24):
            actions = _age_capped_actions(
                seed_q_bps,
                calibration_mask,
                policy_id,
                maximum_holding_hours,
            )
            replay = replay_consensus_actions(
                dataset,
                actions,
                calibration_mask,
                policy_id=f"{policy_id}_cap_{maximum_holding_hours}h",
                role="calibration_diagnostic",
                scenario="stress",
                one_way_cost_bps=diagnostic_spec.stress_one_way_cost_bps,
                bootstrap_seed=(
                    diagnostic_spec.seeds[0]
                    + policy_index * 1_000
                    + maximum_holding_hours
                ),
                spec=diagnostic_spec,
            )
            metrics = replay.metrics
            bootstrap = metrics["bootstrap_mean_hourly_portfolio_bps"]
            rows.append(
                {
                    "policy_id": policy_id,
                    "maximum_holding_hours": maximum_holding_hours,
                    "stress_total_net_return_fraction": metrics[
                        "total_net_return_fraction"
                    ],
                    "stress_maximum_drawdown_fraction": metrics[
                        "maximum_drawdown_fraction"
                    ],
                    "stress_profit_factor": metrics["profit_factor"],
                    "closed_trades": metrics["closed_trades"],
                    "active_days": metrics["active_days"],
                    "bootstrap_lower_mean_hourly_bps": bootstrap["lower_bps"],
                    "symbol_net_bps": metrics["symbol_net_bps"],
                }
            )
    return rows


def diagnose(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    report_path = arguments.source_report.resolve()
    report = _validate_report(report_path)
    dirty = _git("status", "--porcelain")
    if dirty:
        raise RuntimeError("Round 54 diagnosis requires a clean committed implementation")
    implementation_commit = _git("rev-parse", "HEAD")
    _binding, cache_paths = _validate_source_binding(
        arguments.dataset_binding.resolve(),
        arguments.derived_cache.resolve(),
    )
    dataset = _load_dataset(cache_paths)
    prediction_path = _verified_prediction_path(report, report_path.parent)
    seed_q_bps = np.load(prediction_path, allow_pickle=False)
    if seed_q_bps.shape != (3, dataset.timestamps, 3, 3, 3, 5):
        raise ValueError("Round 54 prediction tensor dimensions drifted")
    calibration_mask = role_mask(dataset, "calibration")
    direction_rows = _direction_rows(seed_q_bps, dataset, calibration_mask)
    position_runs: list[dict[str, object]] = []
    original_policy_rows: list[dict[str, object]] = []
    for policy_id in POLICY_IDS:
        actions, diagnostics = consensus_policy_actions(
            seed_q_bps,
            calibration_mask,
            policy_id,
        )
        runs = _position_runs(
            actions,
            dataset.timestamps_ms,
            calibration_mask,
            policy_id,
        )
        position_runs.extend(runs)
        hours = [int(row["hours"]) for row in runs]
        original_policy_rows.append(
            {
                "policy_id": policy_id,
                **diagnostics,
                "nonflat_runs": len(runs),
                "maximum_holding_hours": max(hours, default=0),
                "median_holding_hours": float(np.median(hours)) if hours else 0.0,
            }
        )
    finite_hold_rows = _finite_hold_rows(
        seed_q_bps,
        dataset,
        calibration_mask,
    )
    best_finite = max(
        finite_hold_rows,
        key=lambda row: float(row["stress_total_net_return_fraction"]),
    )
    pooled_direction = [
        row
        for row in direction_rows
        if row["symbol"] == "ALL"
    ]
    maximum_directional_spearman = max(
        float(row["directional_spearman"]) for row in pooled_direction
    )
    output: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "round": ROUND,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "rejected",
        "development_only": True,
        "evaluation_role_read": False,
        "profitability_claim": False,
        "trading_authority": False,
        "source_report_canonical_sha256": SOURCE_REPORT_CANONICAL_SHA256,
        "source_report_file_sha256": SOURCE_REPORT_FILE_SHA256,
        "source_prediction_sha256": _file_sha256(prediction_path),
        "dataset_sha256": dataset.dataset_sha256,
        "implementation_commit": implementation_commit,
        "directional_rank": direction_rows,
        "original_policy_summary": original_policy_rows,
        "position_runs": position_runs,
        "finite_holding_screen": finite_hold_rows,
        "findings": [
            (
                "The Bellman model learned a numerically stable representation, but its best "
                f"pooled calibration directional Spearman was only {maximum_directional_spearman:.6f}."
            ),
            (
                "The median-Q controller converted weak action differences into position runs as "
                f"long as {max(int(row['hours']) for row in position_runs)} hours, which is not "
                "a day-trading or stale-position-safe policy."
            ),
            (
                "Mandatory finite holding reduced the stale-position defect but did not establish "
                "robust economics. The best diagnostic row was "
                f"{best_finite['policy_id']} at {best_finite['maximum_holding_hours']} hours with "
                f"stress return {100.0 * float(best_finite['stress_total_net_return_fraction']):.4f}%, "
                f"drawdown {100.0 * float(best_finite['stress_maximum_drawdown_fraction']):.4f}%, "
                f"profit factor {float(best_finite['stress_profit_factor']):.4f}, and bootstrap "
                f"lower mean {float(best_finite['bootstrap_lower_mean_hourly_bps']):.6f} bps/hour."
            ),
            (
                "Positive Bellman-residual skill is not evidence of forecast alpha or tradable "
                "value; future designs must gate directional and economic skill separately."
            ),
        ],
        "decision": {
            "retain": [
                "DirectML-compatible distributional tensors",
                "explicit prior-position/action cost accounting",
                "all-seed stability diagnostics",
            ],
            "reject": [
                "unbounded rolling greedy Q policy",
                "Bellman residual as a sufficient model-quality gate",
                "any controller without a mandatory maximum holding period",
            ],
            "next_model_requirements": [
                "Use a supervised finite-horizon alpha distribution as the primary learned quantity and keep dynamic programming in a bounded controller.",
                "Make maximum holding time, per-trade stop loss, forced reconciliation, and cooldown state explicit and non-bypassable.",
                "Require directional rank, proper distributional score, net action value, bootstrap, drawdown, activity, and symbol-breadth gates before evaluation.",
                "Preserve the withheld evaluation boundary until a calibration candidate passes every gate.",
            ],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output["diagnostic_canonical_sha256"] = _canonical_sha256(output)
    write_json_atomic(
        arguments.output.resolve(),
        output,
        indent=2,
        sort_keys=True,
    )
    return output


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument(
        "--dataset-binding",
        type=Path,
        default=research / "round-047-cost-aware-utility-tcn-binding.json",
    )
    parser.add_argument("--derived-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output = diagnose(_parser().parse_args(argv))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
