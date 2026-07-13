"""Publish truthful Round 53 CSM evidence, diagnostics, tables, and charts."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

ROUND = 53
REPORT_SCHEMA = "round-053-executable-csm-fincast-report-v1"
DIAGNOSTIC_SCHEMA = "round-053-csm-rank-tail-diagnostic-v1"
DESIGN_SHA256 = "58a6df6f34bc4d2cbb660be2c84f80a352a708e315197e45f4ab9922ef7504e4"
BINDING_SHA256 = "580b081827562f1fcff383245cedff82bda89a1873476e951622d5335b0eaf92"
SOURCE_REPORT_CANONICAL_SHA256 = (
    "e21d35c332841ed53797f41ac111adf58e04e17d022d0e307b1d0ed620a5a658"
)
SOURCE_REPORT_FILE_SHA256 = (
    "5619ab8cd012c69465d8bfde04a0bc106927957f7c67d4b107587c1c5199501f"
)
DIAGNOSTIC_CANONICAL_SHA256 = (
    "0e9a181e228760042af272bf56733ad5bf2883abe662e129ce57bbcad91459dd"
)
DIAGNOSTIC_FILE_SHA256 = (
    "9195f1757541c11df0d09367fca792f506b887a04bc1bade4ee53dd987fde8f5"
)
DIAGNOSTIC_IMPLEMENTATION_COMMIT = (
    "7e15ab84210799a3ef7764f7923eb5d03f253dfd"
)
DIAGNOSTIC_IMPLEMENTATION_BLOB = "96d539abecce9a4156004d30c3fe90b643618f66"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SEEDS = (5201, 5202, 5203)
CANDIDATES = (
    "executable_direct_mean_lightgbm",
    "executable_csm_lightgbm",
    "executable_csm_lightgbm_fincast",
)
DISPLAY = {
    CANDIDATES[0]: "Direct mean",
    CANDIDATES[1]: "Executable CSM",
    CANDIDATES[2]: "CSM + FinCast",
}


from tools.publish_round52_executable_support_hurdle import (  # noqa: E402
    COLORS,
    _artifact,
    _bar_svg,
    _canonical_json,
    _canonical_sha256,
    _file_sha256,
    _line_svg,
    _read_object,
    _validate_finite,
    _verified_artifact,
    _write_csv,
    _write_text,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source(
    *,
    report_path: Path,
    diagnostic_path: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    if _file_sha256(report_path) != SOURCE_REPORT_FILE_SHA256:
        raise ValueError("Round 53 source report file hash drifted")
    if _file_sha256(diagnostic_path) != DIAGNOSTIC_FILE_SHA256:
        raise ValueError("Round 53 rank-tail diagnostic file hash drifted")
    report = _read_object(report_path, "Round 53 report")
    diagnostic = _read_object(diagnostic_path, "Round 53 rank-tail diagnostic")
    design = _read_object(design_path, "Round 53 design")
    binding = _read_object(binding_path, "Round 53 binding")
    report_canonical = dict(report)
    report_claimed = report_canonical.pop("report_canonical_sha256", None)
    diagnostic_canonical = dict(diagnostic)
    diagnostic_claimed = diagnostic_canonical.pop("report_canonical_sha256", None)
    design_canonical = dict(design)
    design_claimed = design_canonical.pop("design_sha256", None)
    binding_canonical = dict(binding)
    binding_claimed = binding_canonical.pop("binding_sha256", None)
    claims = report.get("claims")
    mechanism = report.get("mechanism_screen")
    diagnosis = diagnostic.get("diagnosis")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report_claimed != SOURCE_REPORT_CANONICAL_SHA256
        or _canonical_sha256(report_canonical) != report_claimed
        or diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA
        or diagnostic_claimed != DIAGNOSTIC_CANONICAL_SHA256
        or _canonical_sha256(diagnostic_canonical) != diagnostic_claimed
        or diagnostic.get("source_round_53", {}).get("report_file_sha256")
        != SOURCE_REPORT_FILE_SHA256
        or design_claimed != DESIGN_SHA256
        or _canonical_sha256(design_canonical) != design_claimed
        or binding_claimed != BINDING_SHA256
        or _canonical_sha256(binding_canonical) != binding_claimed
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("binding_sha256") != BINDING_SHA256
        or report.get("round_gate", {}).get("passed") is not False
        or not isinstance(claims, Mapping)
        or claims.get("selection_contaminated") is not True
        or any(
            claims.get(name) is not False
            for name in (
                "profitability_claim",
                "ai_uplift_claim",
                "trading_authority",
                "testnet_authority",
                "live_authority",
                "leverage_applied",
            )
        )
        or not isinstance(mechanism, Mapping)
        or mechanism.get("passed_candidates") != []
        or mechanism.get("untouched_data_expansion_authorized") is not False
        or mechanism.get("trading_or_promotion_authorized") is not False
        or not isinstance(diagnosis, Mapping)
        or diagnosis.get("calibration_rows_passed") != 0
        or diagnosis.get("ordinal_followup_supported") is not False
        or diagnosis.get("next_round_authorized") is not False
        or diagnostic.get("claims", {}).get("trading_authority") is not False
        or _git(
            "rev-parse",
            f"{DIAGNOSTIC_IMPLEMENTATION_COMMIT}:tools/diagnose_round53_csm_rank_tail.py",
        )
        != DIAGNOSTIC_IMPLEMENTATION_BLOB
    ):
        raise ValueError("Round 53 source contracts or claims drifted")
    _validate_finite(report)
    _validate_finite(diagnostic)
    model_count = 0
    prediction_count = 0
    for symbol in SYMBOLS:
        if int(report["data"][symbol]["synthetic_rows"]) != 0:
            raise ValueError(f"Round 53 {symbol} contains synthetic rows")
        _verified_artifact(
            report["data"][symbol]["fincast"]["artifact"], f"{symbol} FinCast"
        )
        for candidate in CANDIDATES:
            for seed in SEEDS:
                _verified_artifact(
                    report["models"][symbol][candidate][str(seed)]["artifact"],
                    f"{symbol} {candidate} {seed} model",
                )
                model_count += 1
                for role in ("policy_calibration", "evaluation"):
                    _verified_artifact(
                        report["prediction_artifacts"][symbol][candidate][str(seed)][
                            role
                        ],
                        f"{symbol} {candidate} {seed} {role} prediction",
                    )
                    prediction_count += 1
    if model_count != 27 or prediction_count != 54:
        raise ValueError("Round 53 artifact inventory is incomplete")
    return report, diagnostic, design, binding


def _support_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        data = report["data"][symbol]
        support = data["support"]
        rows.append(
            {
                "round": ROUND,
                "symbol": symbol,
                "source_rows": data["microstructure_rows"],
                "valid_barrier_rows": data["deterministic_rows"],
                "long_executable_rows": support["long_executable_rows"],
                "long_executable_ratio": support["long_executable_ratio"],
                "short_executable_rows": support["short_executable_rows"],
                "short_executable_ratio": support["short_executable_ratio"],
                "long_mask_sha256": support["long_mask_sha256"],
                "short_mask_sha256": support["short_mask_sha256"],
                "synthetic_rows": data["synthetic_rows"],
            }
        )
    return rows


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            for seed in SEEDS:
                record = report["predictive_metrics"][symbol][candidate][str(seed)]
                for role in ("policy_calibration", "evaluation"):
                    for side in ("long", "short"):
                        rows.append(
                            {
                                "round": ROUND,
                                "symbol": symbol,
                                "candidate": candidate,
                                "seed": seed,
                                "role": role,
                                "side": side,
                                **record[role][side],
                            }
                        )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            for seed in SEEDS:
                model = report["models"][symbol][candidate][str(seed)]
                rows.append(
                    {
                        "round": ROUND,
                        "symbol": symbol,
                        "candidate": candidate,
                        "seed": seed,
                        "reused_from_round": model.get("reused_from_round", ""),
                        "cache_state": model.get("cache_state", ""),
                        "backend_kind": model["backend_kind"],
                        "backend_device": model["backend_device"],
                        "model_sha256": model["model_sha256"],
                        "artifact_sha256": model["artifact"]["sha256"],
                        "artifact_bytes": model["artifact"]["bytes"],
                        "role_rows": model.get("role_rows", ""),
                        "rejected_role_rows": model.get("rejected_role_rows", ""),
                        "role_mask_sha256": model.get("role_mask_sha256", ""),
                        "minimum_leaf_rows": model.get("minimum_leaf_rows", ""),
                        "best_iterations": model.get("best_iterations", ""),
                        "magnitude_temperature": model.get(
                            "magnitude_temperature", ""
                        ),
                        "sign_probability_calibration": model.get(
                            "sign_probability_calibration", ""
                        ),
                    }
                )
    return rows


def _policy_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for item in report["policy_grid"][candidate]:
            row: dict[str, object] = {
                "round": ROUND,
                "candidate": candidate,
                "coverage": item["coverage"],
                "thresholds_bps": item["thresholds_bps"],
                "policy_calibration_passed": item["policy_calibration_gate"][
                    "passed"
                ],
                "policy_calibration_reasons": item["policy_calibration_gate"][
                    "reasons"
                ],
                "consumed_evaluation_passed": item["consumed_evaluation_gate"][
                    "passed"
                ],
                "consumed_evaluation_reasons": item["consumed_evaluation_gate"][
                    "reasons"
                ],
                "policy_calibration_days": item["policy_calibration_days"],
                "formally_selected": False,
                "selection_contaminated": True,
            }
            for role_key, result_key in (
                ("policy_calibration", "policy_calibration"),
                ("consumed_evaluation", "consumed_evaluation_diagnostic"),
            ):
                result = item[result_key]
                for scenario in ("base", "paired_stress"):
                    metrics = result[scenario]["metrics"]
                    prefix = f"{role_key}_{scenario}"
                    for metric in (
                        "trades",
                        "total_net_bps",
                        "mean_net_bps",
                        "profit_factor",
                        "max_drawdown_bps",
                        "active_days",
                        "trades_per_active_day",
                    ):
                        row[f"{prefix}_{metric}"] = metrics[metric]
                    row[f"{prefix}_symbol_net_bps"] = result[scenario][
                        "symbol_net_bps"
                    ]
            rows.append(row)
    return rows


def _rank_tail_rows(diagnostic: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in diagnostic["grid"]:
        row: dict[str, object] = {
            "round": ROUND,
            "candidate": item["candidate"],
            "aggregation": item["aggregation"],
            "coverage": item["coverage"],
            "thresholds_bps": item["thresholds_bps"],
            "calibration_passed": item["calibration_screen"]["passed"],
            "calibration_reasons": item["calibration_screen"]["reasons"],
            "consumed_evaluation_positive_base_and_stress": item[
                "consumed_evaluation_positive_base_and_stress"
            ],
            "formally_selected": False,
            "selection_contaminated": True,
        }
        for role_key, result_key in (
            ("policy_calibration", "policy_calibration"),
            ("consumed_evaluation", "consumed_evaluation"),
        ):
            result = item[result_key]
            for scenario in ("base", "paired_stress"):
                prefix = f"{role_key}_{scenario}"
                for metric, value in result[scenario]["metrics"].items():
                    row[f"{prefix}_{metric}"] = value
                row[f"{prefix}_symbol_net_bps"] = result[scenario][
                    "symbol_net_bps"
                ]
            for symbol in SYMBOLS:
                opportunities = result["symbols"][symbol]["opportunities"]
                for metric, value in opportunities.items():
                    row[f"{role_key}_{symbol}_{metric}"] = value
        rows.append(row)
    return rows


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        predictive = report["predictive_gates"][candidate]
        selected = report["selected_policy"][candidate]
        rows.extend(
            (
                {
                    "round": ROUND,
                    "candidate": candidate,
                    "gate": "predictive",
                    "passed": predictive["passed"],
                    "reasons": predictive["reasons"],
                },
                {
                    "round": ROUND,
                    "candidate": candidate,
                    "gate": "policy_calibration",
                    "passed": selected["selection_passed"],
                    "reasons": (
                        []
                        if selected["selection_passed"]
                        else ["no_temporally_stable_policy_calibration_variant_passed"]
                    ),
                },
                {
                    "round": ROUND,
                    "candidate": candidate,
                    "gate": "consumed_evaluation",
                    "passed": selected["evaluation_gate_passed"],
                    "reasons": selected["evaluation_reasons"],
                },
            )
        )
    rows.append(
        {
            "round": ROUND,
            "candidate": CANDIDATES[2],
            "gate": "ai_uplift",
            "passed": report["ai_uplift_gate"]["passed"],
            "reasons": report["ai_uplift_gate"]["reasons"],
        }
    )
    return rows


def _ai_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    ai = report["ai_uplift_gate"]
    return [
        {
            "round": ROUND,
            "control": CANDIDATES[1],
            "treatment": CANDIDATES[2],
            "control_average_joint_log_loss": ai[
                "control_average_joint_log_loss"
            ],
            "treatment_average_joint_log_loss": ai["ai_average_joint_log_loss"],
            "joint_log_loss_improvement": ai["joint_log_loss_improvement"],
            "required_joint_log_loss_improvement": 0.005,
            "control_average_expected_payoff_spearman": ai[
                "control_average_expected_payoff_spearman"
            ],
            "treatment_average_expected_payoff_spearman": ai[
                "ai_average_expected_payoff_spearman"
            ],
            "expected_payoff_spearman_improvement": ai[
                "expected_payoff_spearman_improvement"
            ],
            "required_expected_payoff_spearman_improvement": 0.005,
            "passed": ai["passed"],
            "reasons": ai["reasons"],
            "parameter_count": 991437160,
            "runtime": "DirectML cached causal FinCast features",
        }
    ]


def _fixed_policy(report: Mapping[str, object]) -> Mapping[str, object]:
    for row in report["policy_grid"][CANDIDATES[1]]:
        if math.isclose(float(row["coverage"]), 0.001, abs_tol=1e-12):
            return row
    raise ValueError("Round 53 fixed 0.001 policy is absent")


def _best_calibration_rank_tail(
    diagnostic: Mapping[str, object],
) -> Mapping[str, object]:
    return max(
        diagnostic["grid"],
        key=lambda row: float(
            row["policy_calibration"]["paired_stress"]["metrics"]["mean_net_bps"]
        ),
    )


def _daily_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    fixed = _fixed_policy(report)
    rows: list[dict[str, object]] = []
    for period, result_key in (
        ("policy_calibration", "policy_calibration"),
        ("consumed_evaluation", "consumed_evaluation_diagnostic"),
    ):
        result = fixed[result_key]
        for scenario in ("base", "paired_stress"):
            cumulative = 0.0
            observed = {
                int(item["utc_day_id"]): float(item["net_bps"])
                for item in result[scenario]["daily_net_bps"]
            }
            if period == "policy_calibration":
                days = range(19515, 19517)
            else:
                days = range(19517, 19523)
            for day_id in days:
                value = observed.get(day_id, 0.0)
                cumulative += value
                rows.append(
                    {
                        "round": ROUND,
                        "candidate": CANDIDATES[1],
                        "coverage": 0.001,
                        "period": period,
                        "scenario": scenario,
                        "utc_date": datetime.fromtimestamp(
                            day_id * 86_400, tz=UTC
                        ).date().isoformat(),
                        "daily_weighted_net_bps": value,
                        "cumulative_weighted_net_bps": cumulative,
                        "formally_selected": False,
                        "selection_contaminated": True,
                    }
                )
    return rows


def _progress_rows(
    previous_path: Path,
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    with previous_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    observed = [int(row["round"]) for row in rows]
    if observed == list(range(1, ROUND + 1)):
        rows = rows[:-1]
        observed = observed[:-1]
    if observed != list(range(1, ROUND)):
        raise ValueError("research progress must contain exactly Rounds 1 through 52")
    fixed = _fixed_policy(report)
    evaluation = fixed["consumed_evaluation_diagnostic"]["paired_stress"]["metrics"]
    calibration = fixed["policy_calibration"]["paired_stress"]["metrics"]
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "executable conditional sign-magnitude + FinCast ablation",
            "periods": "2023-05-16..2023-06-14; eval 2023-06-09..2023-06-14",
            "selection_contaminated": "True",
            "horizon_seconds": "300",
            "feature_set": "118 causal microstructure; +30 cached FinCast treatment",
            "risk_level": "consumed development only; unlevered",
            "spearman_ic": str(
                report["ai_uplift_gate"][
                    "control_average_expected_payoff_spearman"
                ]
            ),
            "selected_signals": "0",
            "executable_trades": "0",
            "status": "rejected",
            "source_file": (
                "verified Round 53 report and rank-tail diagnostic; no calibration "
                "rule passed"
            ),
            "best_policy_trades": str(evaluation["trades"]),
            "best_policy_total_net_bps": str(evaluation["total_net_bps"]),
            "best_policy_mean_net_bps": str(evaluation["mean_net_bps"]),
            "best_policy_max_drawdown_bps": str(evaluation["max_drawdown_bps"]),
            "best_policy_profit_factor": str(evaluation["profit_factor"]),
            "best_model_id": "executable_csm_consumed_diagnostic_not_selected",
            "ensemble_models": "27",
            "valid_barrier_rows": str(
                sum(
                    int(report["data"][symbol]["deterministic_rows"])
                    for symbol in SYMBOLS
                )
            ),
            "calibration_eligible_rows": str(calibration["trades"]),
            "policy_eligible_rows": str(evaluation["trades"]),
            "development_consumed": "True",
            "architecture_gates_passed": "0",
            "architecture_gate_count": "3",
        }
    )
    rows.append(new)
    return rows, fields


def _support_svg(rows: Sequence[Mapping[str, object]]) -> str:
    groups = [
        (
            str(row["symbol"]),
            (
                (
                    "Long executable",
                    100.0 * float(row["long_executable_ratio"]),
                    COLORS["teal"],
                ),
                (
                    "Short executable",
                    100.0 * float(row["short_executable_ratio"]),
                    COLORS["blue"],
                ),
            ),
        )
        for row in rows
    ]
    return _bar_svg(
        title="Replay-executable action support",
        subtitle="Exact side-specific support shared by fitting, calibration, and replay",
        groups=groups,
        y_min=0.0,
        y_max=100.0,
        y_label="Executable valid-barrier rows (%)",
    )


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    groups = []
    for candidate in CANDIDATES:
        selected = [
            row
            for row in rows
            if row["candidate"] == candidate and row["role"] == "evaluation"
        ]
        mse = 100.0 * sum(
            float(row["expected_payoff_mse_skill"]) for row in selected
        ) / len(selected)
        rank = 100.0 * sum(
            float(row["expected_payoff_spearman"]) for row in selected
        ) / len(selected)
        proper = [
            float(row["joint_log_loss_skill"])
            for row in selected
            if row.get("joint_log_loss_skill") is not None
        ]
        groups.append(
            (
                DISPLAY[candidate],
                (
                    ("Expected-payoff MSE skill", mse, COLORS["red"]),
                    ("Expected-payoff Spearman x100", rank, COLORS["blue"]),
                    (
                        "Joint log-loss skill",
                        100.0 * sum(proper) / len(proper) if proper else None,
                        COLORS["teal"],
                    ),
                ),
            )
        )
    return _bar_svg(
        title="Distribution skill did not become expected-value accuracy",
        subtitle="Mean consumed-development score over three assets, both sides, and three seeds",
        groups=groups,
        y_min=-4.0,
        y_max=9.0,
        y_label="Percent or rank correlation x100",
    )


def _policy_svg(report: Mapping[str, object]) -> str:
    groups = []
    for candidate in CANDIDATES:
        row = next(
            item
            for item in report["policy_grid"][candidate]
            if math.isclose(float(item["coverage"]), 0.001, abs_tol=1e-12)
        )
        calibration = row["policy_calibration"]
        evaluation = row["consumed_evaluation_diagnostic"]
        groups.append(
            (
                DISPLAY[candidate],
                (
                    (
                        "Calibration base",
                        calibration["base"]["metrics"]["mean_net_bps"],
                        COLORS["cyan"],
                    ),
                    (
                        "Calibration stress",
                        calibration["paired_stress"]["metrics"]["mean_net_bps"],
                        COLORS["amber"],
                    ),
                    (
                        "Consumed eval base",
                        evaluation["base"]["metrics"]["mean_net_bps"],
                        COLORS["blue"],
                    ),
                    (
                        "Consumed eval stress",
                        evaluation["paired_stress"]["metrics"]["mean_net_bps"],
                        COLORS["red"],
                    ),
                ),
            )
        )
    return _bar_svg(
        title="Frozen positive-EV policy failed calibration",
        subtitle="Fixed 0.10% coverage; zero-height direct bars are empty ledgers",
        groups=groups,
        y_min=-16.0,
        y_max=22.0,
        y_label="Weighted mean net return (bps/trade)",
    )


def _rank_tail_svg(diagnostic: Mapping[str, object]) -> str:
    rows = [
        row
        for row in diagnostic["grid"]
        if row["candidate"] == CANDIDATES[1]
        and row["aggregation"] == "worst_seed"
    ]
    points_calibration = [
        (
            float(index),
            float(
                row["policy_calibration"]["paired_stress"]["metrics"][
                    "mean_net_bps"
                ]
            ),
        )
        for index, row in enumerate(rows)
    ]
    points_evaluation = [
        (
            float(index),
            float(
                row["consumed_evaluation"]["paired_stress"]["metrics"][
                    "mean_net_bps"
                ]
            ),
        )
        for index, row in enumerate(rows)
    ]
    labels = {
        float(index): f"{100.0 * float(row['coverage']):g}%"
        for index, row in enumerate(rows)
        if index in {0, 2, 4, 6, 8}
    }
    return _line_svg(
        title="Global rank correlation failed in the executable tail",
        subtitle="Worst-seed CSM; thresholds fit on calibration, all 9 calibration rules rejected",
        series=(
            ("Calibration paired stress", points_calibration, COLORS["amber"]),
            ("Consumed evaluation paired stress", points_evaluation, COLORS["red"]),
        ),
        x_labels=labels,
        y_label="Weighted mean net bps per non-overlapping trade",
    )


def _ai_svg(report: Mapping[str, object]) -> str:
    ai = report["ai_uplift_gate"]
    return _bar_svg(
        title="FinCast reduced control quality",
        subtitle="Matched causal ablation; higher improvement is better, frozen requirement is +0.005",
        groups=(
            (
                "Joint log loss",
                (
                    (
                        "Observed improvement",
                        ai["joint_log_loss_improvement"],
                        COLORS["teal"],
                    ),
                    ("Required improvement", 0.005, COLORS["muted"]),
                ),
            ),
            (
                "Expected-payoff rank",
                (
                    (
                        "Observed improvement",
                        ai["expected_payoff_spearman_improvement"],
                        COLORS["blue"],
                    ),
                    ("Required improvement", 0.005, COLORS["muted"]),
                ),
            ),
        ),
        y_min=-0.004,
        y_max=0.006,
        y_label="Absolute improvement",
        tick_decimals=3,
        value_decimals=6,
    )


def _daily_svg(rows: Sequence[Mapping[str, object]]) -> str:
    dates = sorted({str(row["utc_date"]) for row in rows})
    date_index = {value: float(index) for index, value in enumerate(dates)}
    series = []
    for name, period, scenario, color in (
        ("Calibration base", "policy_calibration", "base", COLORS["cyan"]),
        (
            "Calibration stress",
            "policy_calibration",
            "paired_stress",
            COLORS["amber"],
        ),
        ("Consumed eval base", "consumed_evaluation", "base", COLORS["blue"]),
        (
            "Consumed eval stress",
            "consumed_evaluation",
            "paired_stress",
            COLORS["red"],
        ),
    ):
        points = [
            (
                date_index[str(row["utc_date"])],
                float(row["cumulative_weighted_net_bps"]),
            )
            for row in rows
            if row["period"] == period and row["scenario"] == scenario
        ]
        series.append((name, points, color))
    labels = {
        float(index): value[5:]
        for index, value in enumerate(dates)
        if index in {0, len(dates) - 1} or index % 2 == 0
    }
    return _line_svg(
        title="Frozen-policy daily diagnostic",
        subtitle="Executable CSM at 0.10% coverage; roles reset to zero and no line was selected",
        series=series,
        x_labels=labels,
        y_label="Cumulative weighted net bps",
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points: list[tuple[float, float]] = []
    for row in rows:
        raw = str(row.get("best_policy_mean_net_bps") or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            points.append((float(row["round"]), value))
    labels = {points[0][0]: str(int(points[0][0])), points[-1][0]: str(ROUND)}
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        if points[0][0] <= value <= points[-1][0]:
            labels[value] = str(int(value))
    return _line_svg(
        title="Optimization research progression",
        subtitle="Descriptive best bps/trade by round; datasets differ and every point is non-promotable",
        series=(("Descriptive best policy", points, COLORS["teal"]),),
        x_labels=labels,
        y_label="Net bps per trade",
    )


def _clean_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    parent = (ROOT / "docs" / "model-research" / "action-value").resolve()
    if not resolved.is_relative_to(parent) or resolved.name != "latest":
        raise ValueError("publication output must be action-value/latest")
    if resolved.exists():
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _readme(
    report: Mapping[str, object], diagnostic: Mapping[str, object]
) -> str:
    fixed = _fixed_policy(report)
    calibration = fixed["policy_calibration"]["paired_stress"]["metrics"]
    evaluation = fixed["consumed_evaluation_diagnostic"]["paired_stress"]["metrics"]
    best_tail = _best_calibration_rank_tail(diagnostic)
    tail_calibration = best_tail["policy_calibration"]["paired_stress"]["metrics"]
    tail_evaluation = best_tail["consumed_evaluation"]["paired_stress"]["metrics"]
    ai = report["ai_uplift_gate"]
    return f"""# Round 53: Executable Conditional Sign-Magnitude

> **Rejected consumed-development screen.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 53 factorized each side's executable payoff into six magnitude states and a magnitude-conditioned sign model. Eighteen new OpenCL LightGBM models used verified Binance USD-M BTC, ETH, and SOL event data; nine direct controls and sealed causal FinCast features were reused.

The CSM improved average expected-payoff rank to `{float(ai['control_average_expected_payoff_spearman']):.6f}` and every joint proper-score comparison, but mean calibration remained wrong. Its frozen 0.10% policy had {int(calibration['trades'])} calibration trade at `{float(calibration['mean_net_bps']):.6f}` stressed bps/trade. The {int(evaluation['trades'])} later trades averaged `+{float(evaluation['mean_net_bps']):.6f}` bps, but calibration did not authorize them.

A separate fixed 54-rule rank-tail diagnostic removed the positive-EV requirement. Zero rules passed calibration. The least-bad calibration rule was `{best_tail['aggregation']}` at `{100.0 * float(best_tail['coverage']):g}%` coverage: {int(tail_calibration['trades'])} trades at `{float(tail_calibration['mean_net_bps']):.6f}` stressed bps/trade; its consumed result was `{float(tail_evaluation['mean_net_bps']):.6f}`. Global correlation therefore did not establish executable top-tail alpha.

FinCast worsened joint log loss by `{abs(float(ai['joint_log_loss_improvement'])):.6f}` and expected-payoff rank by `{abs(float(ai['expected_payoff_spearman_improvement'])):.6f}` versus the matched CSM control.

## Evidence

| View | Graph | Source |
|---|---|---|
| Executable support | [SVG](charts/executable-support.svg) | [CSV](support.csv) |
| Forecast quality | [SVG](charts/forecast-quality.svg) | [CSV](forecast.csv) |
| Frozen policy | [SVG](charts/policy-economics.svg) | [CSV](policy-grid.csv) |
| Rank-tail falsification | [SVG](charts/rank-tail.svg) | [CSV](rank-tail.csv) |
| Fixed-policy daily path | [SVG](charts/daily-equity.svg) | [CSV](daily-policy.csv) |
| FinCast uplift | [SVG](charts/ai-uplift.svg) | [CSV](ai-uplift.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`screen.json` and `rank-tail-screen.json` preserve the complete sources. `report.json` binds every publication file to the frozen design, execution binding, external reports, models, and predictions.
"""


def publish(
    *,
    report_path: Path,
    diagnostic_path: Path,
    design_path: Path,
    binding_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, diagnostic, _design, binding = _validate_source(
        report_path=report_path,
        diagnostic_path=diagnostic_path,
        design_path=design_path,
        binding_path=binding_path,
    )
    progress_rows, progress_fields = _progress_rows(previous_progress_path, report)
    support_rows = _support_rows(report)
    forecast_rows = _forecast_rows(report)
    model_rows = _model_rows(report)
    policy_rows = _policy_rows(report)
    rank_tail_rows = _rank_tail_rows(diagnostic)
    gate_rows = _gate_rows(report)
    ai_rows = _ai_rows(report)
    daily_rows = _daily_rows(report)
    _clean_output(output_dir)
    charts = output_dir / "charts"

    _write_csv(output_dir / "support.csv", support_rows)
    _write_csv(output_dir / "forecast.csv", forecast_rows)
    _write_csv(output_dir / "models.csv", model_rows)
    _write_csv(output_dir / "policy-grid.csv", policy_rows)
    _write_csv(output_dir / "rank-tail.csv", rank_tail_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(output_dir / "ai-uplift.csv", ai_rows)
    _write_csv(output_dir / "daily-policy.csv", daily_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "executable-support.svg", _support_svg(support_rows))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(forecast_rows))
    _write_text(charts / "policy-economics.svg", _policy_svg(report))
    _write_text(charts / "rank-tail.svg", _rank_tail_svg(diagnostic))
    _write_text(charts / "daily-equity.svg", _daily_svg(daily_rows))
    _write_text(charts / "ai-uplift.svg", _ai_svg(report))
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))
    _write_text(output_dir / "README.md", _readme(report, diagnostic))
    write_json_atomic(output_dir / "screen.json", report, indent=2, sort_keys=True)
    write_json_atomic(
        output_dir / "rank-tail-screen.json", diagnostic, indent=2, sort_keys=True
    )

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    fixed = _fixed_policy(report)
    calibration = fixed["policy_calibration"]["paired_stress"]["metrics"]
    evaluation = fixed["consumed_evaluation_diagnostic"]["paired_stress"]["metrics"]
    best_tail = _best_calibration_rank_tail(diagnostic)
    publication: dict[str, object] = {
        "schema_version": "round-053-action-value-publication-v1",
        "round": ROUND,
        "published_at": report["generated_at"],
        "publisher_path": "tools/publish_round53_executable_csm.py",
        "source": {
            "report_path": str(report_path.resolve()),
            "report_file_sha256": SOURCE_REPORT_FILE_SHA256,
            "report_canonical_sha256": SOURCE_REPORT_CANONICAL_SHA256,
            "rank_tail_diagnostic_path": str(diagnostic_path.resolve()),
            "rank_tail_diagnostic_file_sha256": DIAGNOSTIC_FILE_SHA256,
            "rank_tail_diagnostic_canonical_sha256": DIAGNOSTIC_CANONICAL_SHA256,
            "rank_tail_implementation_commit": DIAGNOSTIC_IMPLEMENTATION_COMMIT,
            "rank_tail_implementation_blob": DIAGNOSTIC_IMPLEMENTATION_BLOB,
            "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
            "design_sha256": DESIGN_SHA256,
            "binding_path": str(binding_path.relative_to(ROOT)).replace("\\", "/"),
            "binding_sha256": BINDING_SHA256,
            "implementation_commit": binding["implementation_commit"],
        },
        "claims": {
            "status": "rejected",
            "selection_contaminated": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
            "untouched_data_expansion_authorized": False,
        },
        "result": {
            "models": 27,
            "prediction_artifacts": 54,
            "formally_selected_policies": 0,
            "passed_mechanisms": 0,
            "csm_calibration_trades": calibration["trades"],
            "csm_calibration_stress_mean_net_bps": calibration["mean_net_bps"],
            "csm_consumed_evaluation_trades": evaluation["trades"],
            "csm_consumed_evaluation_stress_mean_net_bps": evaluation[
                "mean_net_bps"
            ],
            "rank_tail_rules_tested": len(diagnostic["grid"]),
            "rank_tail_calibration_rules_passed": diagnostic["diagnosis"][
                "calibration_rows_passed"
            ],
            "least_bad_rank_tail_rule": {
                "candidate": best_tail["candidate"],
                "aggregation": best_tail["aggregation"],
                "coverage": best_tail["coverage"],
                "calibration_stress_metrics": best_tail["policy_calibration"][
                    "paired_stress"
                ]["metrics"],
            },
            "ai_joint_log_loss_improvement": report["ai_uplift_gate"][
                "joint_log_loss_improvement"
            ],
            "ai_expected_payoff_spearman_improvement": report["ai_uplift_gate"][
                "expected_payoff_spearman_improvement"
            ],
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2, sort_keys=True)
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round53-executable-csm-20260714-v1\report.json"
        ),
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round53-csm-rank-tail-diagnostic-20260714-v1.json"
        ),
    )
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
    parser.add_argument("--progress", type=Path, default=research / "latest" / "progress.csv")
    parser.add_argument("--output", type=Path, default=research / "latest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = publish(
        report_path=args.report.resolve(),
        diagnostic_path=args.diagnostic.resolve(),
        design_path=args.design.resolve(),
        binding_path=args.binding.resolve(),
        previous_progress_path=args.progress.resolve(),
        output_dir=args.output.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": publication["round"],
                "status": publication["claims"]["status"],
                "publication_canonical_sha256": publication[
                    "publication_canonical_sha256"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
