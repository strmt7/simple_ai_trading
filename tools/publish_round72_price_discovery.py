"""Publish hash-verified Round 72 price-discovery rejection evidence."""

from __future__ import annotations

import argparse
import csv
import html
import itertools
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

from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.publish_round52_executable_support_hurdle import (  # noqa: E402
    COLORS,
    _artifact,
    _bar_svg,
    _canonical_json,
    _canonical_sha256,
    _file_sha256,
    _read_object,
    _svg_start,
    _validate_finite,
    _write_csv,
    _write_text,
)
from tools.publish_round59_funding_persistence_feasibility import (  # noqa: E402
    _clean_output,
)


ROUND = 72
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS = (30, 60, 300)
LAYERS = ("perpetual_only", "spot_perpetual")
HEADS = ("binary_direction", "continuous_return_bps")
FOLDS = tuple(range(1, 7))
REPORT_SCHEMA = "round-072-price-discovery-evaluation-v1"
PUBLICATION_SCHEMA = "round-072-price-discovery-publication-v1"
REPORT_FILE_SHA256 = "eb0723856bcc905698ba39da7e381e67eb61bd6551e1374962abc25d663acec0"
REPORT_SHA256 = "65900fa58299d56fffa04206dcff83a343e9b005ca71f654efe4e939508d3e3d"
METRICS_FILE_SHA256 = "42bccfc5cf5a6408e8909695eac060306a281c37ab3f57d928a2a048333ac3bf"
CORPUS_FILE_SHA256 = "72b7a72a422e8df6d8f01f950a53129e9d4ebf69ca09736317e3238289b72d3f"
CORPUS_SHA256 = "1d7791db923f1d1a7eddc8189934424795246ea01250f6dbef26a59483605adb"
DESIGN_FILE_SHA256 = "23b443ec6e30431034dd3156e2bfb40f86638183ced8e28a2d856d0713ef4a20"
DESIGN_SHA256 = "505818f74cdd9484f66b1a504de821d4d366ec35c7d7bf978ffa15d613104812"
INVENTORY_FILE_SHA256 = "055b5c5a3d975a092bae0c27253757c3be39535945d0aa6caaecdf31e9343b1b"
INVENTORY_SHA256 = "e8c505132716c68ad753cbdd93b23094b778d9067c8a6c9381fad0e20cdd662c"
IMPLEMENTATION_FILE_SHA256 = "e16fef96b033ac1c127c01bed55d56cc3e19ebbcdfe0d358f1c72ce58d129adb"
IMPLEMENTATION_SHA256 = "d8679606e75ec7fa2bf00032b34489218085f7c7f5159419e192f3ee351dfad9"
EXECUTION_COMMIT = "43e465865a288110e5b01adc6cfc5d22df82eecc"
SOURCE_BLOBS = {
    "tools/run_round72_price_discovery_screen.py": "4358b198960890e45a9fb9faf63d7fc07a0a3fa0",
    "src/simple_ai_trading/price_discovery_dataset.py": "ff819f7fabf2a5e01255eda7008c8e5729b3d4e0",
    "src/simple_ai_trading/price_discovery_model.py": "d6385a231f2a59d1ec9be6ef1eff2ce27d664aa3",
    "src/simple_ai_trading/price_discovery_evaluation.py": "03b77a97113ce110d636e6cde436e29d1642229a",
}


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _canonical_value(value: Mapping[str, object], digest_key: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(digest_key, ""))
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError(f"{digest_key} does not match canonical content")
    return actual


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {path.name}")
    return rows


def _validate_sources(
    *,
    report_path: Path,
    metrics_path: Path,
    corpus_path: Path,
    design_path: Path,
    inventory_path: Path,
    implementation_path: Path,
) -> dict[str, object]:
    expected_files = (
        (report_path, REPORT_FILE_SHA256),
        (metrics_path, METRICS_FILE_SHA256),
        (corpus_path, CORPUS_FILE_SHA256),
        (design_path, DESIGN_FILE_SHA256),
        (inventory_path, INVENTORY_FILE_SHA256),
        (implementation_path, IMPLEMENTATION_FILE_SHA256),
    )
    for path, expected in expected_files:
        if _file_sha256(path) != expected:
            raise ValueError(f"Round 72 source file hash drifted: {path.name}")
    report = _read_object(report_path, "Round 72 evaluation")
    corpus = _read_object(corpus_path, "Round 72 corpus report")
    design = _read_object(design_path, "Round 72 design")
    inventory = _read_object(inventory_path, "Round 72 inventory")
    implementation = _read_object(implementation_path, "Round 72 implementation")
    metrics = _read_csv(metrics_path)
    expected_models = set(
        itertools.product(SYMBOLS, HORIZONS, LAYERS, HEADS, FOLDS)
    )
    models = report.get("models")
    observed_models = {
        (
            row.get("symbol"),
            row.get("horizon_seconds"),
            row.get("feature_layer"),
            row.get("head"),
            row.get("fold"),
        )
        for row in models
    } if isinstance(models, list) else set()
    scope = report.get("scope")
    backend = report.get("backend")
    comparisons = report.get("feature_comparisons")
    components = report.get("symbol_horizon_components")
    layers = report.get("layer_reports")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or _canonical_value(report, "report_sha256") != REPORT_SHA256
        or report.get("decision") != "reject_round_072_price_discovery"
        or report.get("primary_gate_passed") is not False
        or report.get("feature_increment_gate_passed") is not False
        or any(
            report.get(key) is not False
            for key in (
                "profitability_claim",
                "execution_or_fill_claim",
                "trading_authority",
                "leverage_authority",
            )
        )
        or not isinstance(scope, dict)
        or scope.get("symbols") != list(SYMBOLS)
        or scope.get("horizons_seconds") != list(HORIZONS)
        or scope.get("feature_layers") != list(LAYERS)
        or scope.get("heads") != list(HEADS)
        or scope.get("development_last_month") != "2026-03"
        or scope.get("terminal_holdout_read") is not False
        or scope.get("terminal_months_excluded")
        != ["2026-04", "2026-05", "2026-06"]
        or scope.get("profit_or_execution_target") is not False
        or backend != {
            "requested": "auto",
            "kind": "opencl",
            "device": "opencl:auto",
            "lightgbm_version": "4.6.0",
        }
        or report.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or observed_models != expected_models
        or not isinstance(models, list)
        or len(models) != 216
        or any(
            row.get("reload_max_absolute_prediction_difference") != 0.0
            or int(row.get("training_rows", 0)) <= 0
            or int(row.get("tuning_rows", 0)) <= 0
            or int(row.get("test_rows", 0)) <= 0
            for row in models
        )
        or not isinstance(layers, list)
        or len(layers) != 36
        or not isinstance(comparisons, list)
        or len(comparisons) != 36
        or any(row.get("passed") is not False for row in comparisons)
        or not isinstance(components, list)
        or len(components) != 9
        or any(row.get("passed") is not False for row in components)
        or len(metrics) != 108
        or any(row.get("evaluation_report_sha256") != REPORT_SHA256 for row in metrics)
        or _canonical_value(corpus, "report_sha256") != CORPUS_SHA256
        or corpus.get("status") != "complete"
        or corpus.get("completed_days") != 69
        or corpus.get("completed_files") != 414
        or corpus.get("completed_compressed_bytes") != 5_964_131_852
        or corpus.get("raw_aggregate_trades_retained") is not False
        or corpus.get("selected_archives_retained") is not False
        or _canonical_value(design, "design_sha256") != DESIGN_SHA256
        or _canonical_value(inventory, "inventory_sha256") != INVENTORY_SHA256
        or _canonical_value(implementation, "implementation_sha256")
        != IMPLEMENTATION_SHA256
        or implementation.get("design_sha256") != DESIGN_SHA256
        or implementation.get("inventory_sha256") != INVENTORY_SHA256
    ):
        raise ValueError("Round 72 publication source identity drifted")
    certificate = report.get("corpus_certificate")
    if (
        certificate != corpus.get("corpus_certificate")
        or report.get("corpus_certificate_sha256")
        != _canonical_sha256(certificate)
    ):
        raise ValueError("Round 72 corpus certificate differs from evaluation")
    for path, expected_blob in SOURCE_BLOBS.items():
        if (
            _git("rev-parse", f"{EXECUTION_COMMIT}:{path}") != expected_blob
            or _git("rev-parse", f"HEAD:{path}") != expected_blob
        ):
            raise ValueError(f"Round 72 execution source drifted: {path}")
    _validate_finite(report)
    _validate_finite(corpus)
    return report


def _layer_index(report: Mapping[str, object]) -> dict[tuple[object, ...], dict]:
    return {
        (
            row["symbol"],
            row["horizon_seconds"],
            row["feature_layer"],
            row["head"],
        ): row
        for row in report["layer_reports"]
    }


def _comparison_index(
    report: Mapping[str, object],
) -> dict[tuple[object, ...], dict]:
    return {
        (row["symbol"], row["horizon_seconds"], row["head"], row["metric"]): row
        for row in report["feature_comparisons"]
    }


def _component_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    layers = _layer_index(report)
    comparisons = _comparison_index(report)
    components = {
        (row["symbol"], row["horizon_seconds"]): row
        for row in report["symbol_horizon_components"]
    }
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for horizon in HORIZONS:
            perpetual_binary = layers[
                symbol, horizon, "perpetual_only", "binary_direction"
            ]
            combined_binary = layers[
                symbol, horizon, "spot_perpetual", "binary_direction"
            ]
            perpetual_return = layers[
                symbol, horizon, "perpetual_only", "continuous_return_bps"
            ]
            combined_return = layers[
                symbol, horizon, "spot_perpetual", "continuous_return_bps"
            ]
            component = components[symbol, horizon]
            log_increment = comparisons[
                symbol, horizon, "binary_direction", "log_loss"
            ]
            mse_increment = comparisons[
                symbol, horizon, "continuous_return_bps", "mean_squared_error"
            ]
            rows.append(
                {
                    "round": ROUND,
                    "symbol": symbol,
                    "horizon_seconds": horizon,
                    "out_of_sample_rows": combined_binary["rows"],
                    "utc_days": combined_binary["utc_days"],
                    "perpetual_only_binary_log_loss_improvement": (
                        perpetual_binary["prevalence_comparison"]["log_loss"][
                            "relative_improvement"
                        ]
                    ),
                    "spot_perpetual_binary_log_loss_improvement": (
                        combined_binary["prevalence_comparison"]["log_loss"][
                            "relative_improvement"
                        ]
                    ),
                    "spot_perpetual_binary_brier_improvement": (
                        combined_binary["prevalence_comparison"]["brier_score"][
                            "relative_improvement"
                        ]
                    ),
                    "spot_perpetual_stress_log_loss_skill": (
                        combined_binary["stress_comparison"]["log_loss"]["skill"]
                    ),
                    "spot_perpetual_stress_brier_skill": (
                        combined_binary["stress_comparison"]["brier_score"]["skill"]
                    ),
                    "balanced_accuracy": combined_binary["metrics"][
                        "balanced_accuracy"
                    ],
                    "balanced_accuracy_day_lower_95": combined_binary[
                        "day_bootstrap"
                    ]["balanced_accuracy"]["lower_95"],
                    "MCC": combined_binary["metrics"]["MCC"],
                    "MCC_day_lower_95": combined_binary["day_bootstrap"]["MCC"][
                        "lower_95"
                    ],
                    "perpetual_only_continuous_MSE_skill_vs_zero": (
                        perpetual_return["controls"]["mean_squared_error"][
                            "skill_vs_zero"
                        ]
                    ),
                    "spot_perpetual_continuous_MSE_skill_vs_zero": (
                        combined_return["controls"]["mean_squared_error"][
                            "skill_vs_zero"
                        ]
                    ),
                    "spot_perpetual_continuous_MAE_skill_vs_zero": (
                        combined_return["controls"]["mean_absolute_error"][
                            "skill_vs_zero"
                        ]
                    ),
                    "spot_perpetual_Spearman": combined_return["metrics"][
                        "Spearman"
                    ],
                    "spot_perpetual_Spearman_day_lower_95": combined_return[
                        "day_bootstrap"
                    ]["Spearman"]["lower_95"],
                    "spot_perpetual_increment_log_loss_improvement": log_increment[
                        "relative_improvement"
                    ],
                    "spot_perpetual_increment_log_loss_q_value": log_increment[
                        "q_value"
                    ],
                    "spot_perpetual_increment_MSE_improvement": mse_increment[
                        "relative_improvement"
                    ],
                    "spot_perpetual_increment_MSE_q_value": mse_increment["q_value"],
                    "component_passed": component["passed"],
                    "failure_reasons": ";".join(component["reasons"]),
                    "evaluation_report_sha256": REPORT_SHA256,
                }
            )
    return rows


def _comparison_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "round": ROUND,
            **dict(row),
            "evaluation_report_sha256": REPORT_SHA256,
        }
        for row in report["feature_comparisons"]
    ]


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "round": ROUND,
            **dict(row),
            "evaluation_report_sha256": REPORT_SHA256,
        }
        for row in report["models"]
    ]


def _progress_rows(previous_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    with previous_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    rows = [row for row in rows if int(row["round"]) <= 61]
    if [int(row["round"]) for row in rows] != list(range(1, 62)):
        raise ValueError("research progress must contain exactly Rounds 1 through 61")

    def add(round_number: int, **values: str) -> None:
        row = {field: "" for field in fields}
        row.update({"round": str(round_number), **values})
        rows.append(row)

    add(
        62,
        stage="coarse depth-stress transition forecast",
        periods="42 months 2023-01..2026-06; 35 walk-forward OOS months",
        selection_contaminated="True",
        horizon_seconds="60;300",
        feature_set="causal Binance bookDepth state-transition features; BTC ETH SOL",
        risk_level="predictive gate only; no execution, P&L, AI, or leverage",
        status="passed_predictive_gate",
        source_file="verified Round 62 report; 24/24 proper-score comparisons passed",
        best_model_id="coarse_depth_stress_transition",
        ensemble_models="210",
        development_consumed="True",
        architecture_gates_passed="24",
        architecture_gate_count="24",
    )
    implementation_rounds = {
        63: "cost-aware labels and AI evidence binding",
        64: "positive-expectancy meta-label execution",
        65: "live payoff evidence for AI review",
        66: "dependence-aware meta-label confidence",
        67: "purged chronological meta-label validation",
        68: "bounded AI context",
        69: "complete AI-uplift cohort",
        70: "exact-case AI cadence admission",
    }
    for round_number, stage in implementation_rounds.items():
        add(
            round_number,
            stage=stage,
            selection_contaminated="False",
            risk_level="implementation and contract validation only",
            status="implementation_only_no_backtest",
            source_file=f"round-{round_number:03d} research document; no new model or economic result",
            development_consumed="False",
        )
    add(
        71,
        stage="institutional microstructure and AI research decision",
        selection_contaminated="False",
        risk_level="research and architecture decision only",
        status="research_only_no_model",
        source_file="Round 71 research decision; no model or economic replay",
        development_consumed="False",
    )
    add(
        72,
        stage="spot-perpetual price-discovery predictive screen",
        periods="sampled 2020-10..2026-03 development; OOS 2023-04..2026-03; terminal 2026-04..06 sealed",
        selection_contaminated="True",
        horizon_seconds="30;60;300",
        feature_set="123 perpetual-only and 287 spot-perpetual causal one-second flow features",
        risk_level="predictive gate only; unlevered; no economic replay",
        selected_signals="0",
        executable_trades="0",
        status="rejected",
        source_file="verified Round 72 report; 0/36 incremental comparisons and 0/9 components passed",
        best_model_id="spot_perpetual_increment_rejected",
        ensemble_models="216",
        development_consumed="True",
        architecture_gates_passed="0",
        architecture_gate_count="45",
    )
    if [int(row["round"]) for row in rows] != list(range(1, ROUND + 1)):
        raise ValueError("research progress publication is incomplete")
    return rows, fields


def _labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [f"{str(row['symbol'])[:3]} {row['horizon_seconds']}s" for row in rows]


def _binary_skill_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Directional proper-score skill stayed below the frozen hurdle",
        subtitle="Relative log-loss improvement versus each fold's training prevalence; required >= 0.200% for every symbol and horizon",
        groups=tuple(
            (
                label,
                (
                    (
                        "Perpetual only",
                        100.0 * float(row["perpetual_only_binary_log_loss_improvement"]),
                        COLORS["blue"],
                    ),
                    (
                        "Spot + perpetual",
                        100.0 * float(row["spot_perpetual_binary_log_loss_improvement"]),
                        COLORS["teal"],
                    ),
                    ("Required", 0.2, COLORS["amber"]),
                ),
            )
            for label, row in zip(_labels(rows), rows, strict=True)
        ),
        y_min=-0.10,
        y_max=0.25,
        y_label="Relative log-loss improvement (%)",
        tick_decimals=2,
        value_decimals=3,
    )


def _continuous_skill_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = [
        100.0 * float(row[key])
        for row in rows
        for key in (
            "perpetual_only_continuous_MSE_skill_vs_zero",
            "spot_perpetual_continuous_MSE_skill_vs_zero",
        )
    ]
    return _bar_svg(
        title="Return regression failed the zero-return control",
        subtitle="Relative MSE skill versus predicting zero; required >= 0.100% for every symbol and horizon",
        groups=tuple(
            (
                label,
                (
                    (
                        "Perpetual only",
                        100.0 * float(row["perpetual_only_continuous_MSE_skill_vs_zero"]),
                        COLORS["blue"],
                    ),
                    (
                        "Spot + perpetual",
                        100.0 * float(row["spot_perpetual_continuous_MSE_skill_vs_zero"]),
                        COLORS["red"],
                    ),
                    ("Required", 0.1, COLORS["amber"]),
                ),
            )
            for label, row in zip(_labels(rows), rows, strict=True)
        ),
        y_min=min(-0.18, min(values) * 1.2),
        y_max=0.14,
        y_label="Relative MSE skill vs zero (%)",
        tick_decimals=2,
        value_decimals=3,
    )


def _increment_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Spot and basis features added no reliable incremental skill",
        subtitle="Spot+perpetual loss improvement versus perpetual-only; required >= 0.100% and BH q <= 0.05, observed q 0.9695-0.9814",
        groups=tuple(
            (
                label,
                (
                    (
                        "Binary log loss",
                        100.0 * float(row["spot_perpetual_increment_log_loss_improvement"]),
                        COLORS["teal"],
                    ),
                    (
                        "Return MSE",
                        100.0 * float(row["spot_perpetual_increment_MSE_improvement"]),
                        COLORS["red"],
                    ),
                    ("Required", 0.1, COLORS["amber"]),
                ),
            )
            for label, row in zip(_labels(rows), rows, strict=True)
        ),
        y_min=-0.08,
        y_max=0.12,
        y_label="Incremental loss improvement (%)",
        tick_decimals=2,
        value_decimals=3,
    )


def _confidence_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = [
        value
        for row in rows
        for value in (
            100.0 * (float(row["balanced_accuracy_day_lower_95"]) - 0.5),
            100.0 * float(row["MCC_day_lower_95"]),
        )
    ]
    return _bar_svg(
        title="Day-block confidence was inconsistent across assets",
        subtitle="Lower 95% bootstrap margins above the null; both must be strictly positive for every symbol and horizon",
        groups=tuple(
            (
                label,
                (
                    (
                        "Balanced accuracy margin",
                        100.0 * (float(row["balanced_accuracy_day_lower_95"]) - 0.5),
                        COLORS["blue"],
                    ),
                    (
                        "MCC lower bound",
                        100.0 * float(row["MCC_day_lower_95"]),
                        COLORS["teal"],
                    ),
                    ("Required margin", 0.0, COLORS["amber"]),
                ),
            )
            for label, row in zip(_labels(rows), rows, strict=True)
        ),
        y_min=min(-1.0, min(values) * 1.2),
        y_max=max(3.5, max(values) * 1.15),
        y_label="Lower-bound margin (percentage points)",
        tick_decimals=1,
        value_decimals=2,
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1200, 760
    left, right, top = 105, 55, 135
    metric_bottom = 500
    timeline_y = 635
    plot_w = width - left - right
    points = [
        (int(row["round"]), 100.0 * float(row["spearman_ic"]))
        for row in rows
        if row.get("spearman_ic") not in ("", None)
    ]
    y_values = [value for _round, value in points] + [0.0]
    y_min, y_max = min(y_values) - 1.0, max(y_values) + 1.0

    def x(round_number: float) -> float:
        return left + (round_number - 1.0) / (ROUND - 1.0) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (metric_bottom - top)

    lines = _svg_start(
        "Research progression: comparable statistics and gate outcomes",
        "Top: only recorded Spearman values. Bottom: recent gate status; implementation-only rounds are not backtests.",
        width=width,
        height=height,
    )
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        yy = y(value)
        lines.extend(
            (
                f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="{COLORS["grid"]}"/>',
                f'<text x="{left-14}" y="{yy+5:.1f}" text-anchor="end" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.1f}</text>',
            )
        )
    path = " ".join(
        ("M" if index == 0 else "L") + f" {x(px):.1f} {y(py):.1f}"
        for index, (px, py) in enumerate(points)
    )
    lines.append(
        f'<path d="{path}" fill="none" stroke="{COLORS["blue"]}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for px, py in points:
        lines.append(
            f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="4" fill="{COLORS["blue"]}"/>'
        )
    lines.extend(
        (
            f'<text transform="translate(25 {(top+metric_bottom)/2:.1f}) rotate(-90)" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Recorded Spearman x 100</text>',
            f'<line x1="{x(54):.1f}" y1="{timeline_y}" x2="{x(72):.1f}" y2="{timeline_y}" stroke="{COLORS["grid"]}" stroke-width="3"/>',
            f'<text x="{left}" y="{timeline_y-80}" fill="{COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="16" font-weight="600">Rounds 54-72: gate status</text>',
        )
    )
    recent = {int(row["round"]): row for row in rows if int(row["round"]) >= 54}
    for round_number, row in recent.items():
        status = str(row.get("status") or "")
        color = (
            COLORS["green"]
            if "passed" in status
            else COLORS["red"]
            if "rejected" in status
            else COLORS["muted"]
        )
        lines.extend(
            (
                f'<circle cx="{x(round_number):.1f}" cy="{timeline_y}" r="8" fill="{color}"/>',
                f'<text x="{x(round_number):.1f}" y="{timeline_y+28}" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="11">{round_number}</text>',
            )
        )
    outcome_x = (365, 565, 755, 945)
    for xx, round_number, label, color in zip(
        outcome_x,
        (60, 61, 62, 72),
        ("structural pass", "economic reject", "predictive pass", "predictive reject"),
        (COLORS["green"], COLORS["red"], COLORS["green"], COLORS["red"]),
        strict=True,
    ):
        lines.extend(
            (
                f'<circle cx="{xx}" cy="{timeline_y-47}" r="5" fill="{color}"/>',
                f'<text x="{xx+11}" y="{timeline_y-43}" fill="{COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="11"><tspan font-weight="700">R{round_number}</tspan><tspan fill="{COLORS["subtext"]}"> {html.escape(label)}</tspan></text>',
            )
        )
    lines.extend(
        (
            f'<circle cx="{left}" cy="{height-35}" r="6" fill="{COLORS["green"]}"/><text x="{left+14}" y="{height-30}" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">Passed frozen gate</text>',
            f'<circle cx="{left+190}" cy="{height-35}" r="6" fill="{COLORS["red"]}"/><text x="{left+204}" y="{height-30}" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">Rejected</text>',
            f'<circle cx="{left+320}" cy="{height-35}" r="6" fill="{COLORS["muted"]}"/><text x="{left+334}" y="{height-30}" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">Implementation or research only</text>',
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def _decision_analysis(
    report: Mapping[str, object],
    components: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    comparisons = list(report["feature_comparisons"])
    analysis: dict[str, object] = {
        "schema_version": "round-072-price-discovery-decision-v1",
        "round": ROUND,
        "decision": report["decision"],
        "primary_gate_passed": False,
        "component_pass_count": sum(bool(row["component_passed"]) for row in components),
        "component_count": len(components),
        "feature_comparison_pass_count": sum(bool(row["passed"]) for row in comparisons),
        "feature_comparison_count": len(comparisons),
        "positive_increment_count": sum(
            float(row["relative_improvement"]) > 0.0 for row in comparisons
        ),
        "feature_increment_q_value_range": [
            min(float(row["q_value"]) for row in comparisons),
            max(float(row["q_value"]) for row in comparisons),
        ],
        "critical_findings": [
            "No symbol-horizon component passed the preregistered primary gate.",
            "No spot_perpetual comparison passed both the 0.1% incremental-loss hurdle and BH q <= 0.05.",
            "The continuous-return head failed the zero-return MSE hurdle for all nine symbol-horizon components.",
            "The sealed 2026-04 through 2026-06 terminal months were not read.",
        ],
        "authorized_next_step": (
            "A separately numbered and prospectively frozen hypothesis using materially "
            "new information; Round 72 terminal, PnL, leverage, and larger-model branches remain closed."
        ),
        "prohibited": [
            "opening the Round 72 terminal months",
            "post hoc threshold or feature tuning against this consumed result",
            "PnL optimization, leverage, testnet, live, or profitability claims from Round 72",
            "claiming that predictive accuracy or AI uplift was established",
        ],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(rows: Sequence[Mapping[str, object]]) -> str:
    table = "\n".join(
        "| {symbol} | {horizon}s | {log:+.3f}% | {mse:+.3f}% | {increment:+.3f}% | {q:.4f} | Rejected |".format(
            symbol=row["symbol"],
            horizon=row["horizon_seconds"],
            log=100.0 * float(row["spot_perpetual_binary_log_loss_improvement"]),
            mse=100.0 * float(row["spot_perpetual_continuous_MSE_skill_vs_zero"]),
            increment=100.0 * float(row["spot_perpetual_increment_log_loss_improvement"]),
            q=float(row["spot_perpetual_increment_log_loss_q_value"]),
        )
        for row in rows
    )
    return f"""# Round 72: Spot-Perpetual Price Discovery

> **Rejected. No profitability or trading claim.** None of the nine BTC, ETH, and SOL symbol-horizon components passed the frozen predictive gate. Terminal data, P&L replay, leverage, testnet, and live trading remain closed.

Round 72 tested whether causal one-second spot flow, perpetual flow, basis, and lead-lag features improved autonomous crypto day-trading forecasts at 30, 60, and 300 seconds. The official Binance corpus contains 17,884,800 one-second rows from one deterministic UTC day per month across October 2020 through June 2026. Six rolling out-of-sample folds covered April 2023 through March 2026; April-June 2026 remained sealed.

All 216 shallow LightGBM models ran through OpenCL and reproduced predictions exactly after serialization. The spot-perpetual layer improved only 15 of 36 primary losses, by amounts far below the frozen hurdle; every adjusted q-value was 0.9695-0.9814. The return-regression head failed to beat a zero-return forecast on MSE for every symbol and horizon.

| Symbol | Horizon | Direction log-loss skill | Return MSE skill vs zero | Spot incremental log-loss | BH q | Decision |
|---|---:|---:|---:|---:|---:|---|
{table}

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Directional proper-score skill | [SVG](charts/primary-binary-skill.svg) | [CSV](components.csv) |
| Return skill versus zero | [SVG](charts/primary-continuous-skill.svg) | [CSV](components.csv) |
| Increment from spot and basis features | [SVG](charts/spot-perpetual-increment.svg) | [CSV](components.csv) |
| Day-block confidence | [SVG](charts/day-block-confidence.svg) | [CSV](components.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

The exact [evaluation](evaluation.json), [108-row metric table](metrics.csv), [36 paired comparisons](feature-comparisons.csv), [216 model records](models.csv), corpus ingestion certificate, design, inventory, and implementation freeze are tracked beside the graphs. Every SVG is regenerated from these numeric files.

## Limits

- This is a predictive screen, not an execution or after-cost backtest.
- One full UTC day per month is representative sampling, not continuous tick coverage of every day.
- Binance spot and perpetual crypto trade continuously; UTC dates are sampling blocks, not formal closes. Listed ETFs and futures follow their own venue calendars and were excluded.
- Aggregate trades do not expose historical quotes, queue position, spread, impact, or receive latency.
- The consumed development result cannot be rescued by post hoc tuning. A materially new hypothesis requires a new preregistered round.
"""


def publish(
    *,
    report_path: Path,
    metrics_path: Path,
    corpus_path: Path,
    design_path: Path,
    inventory_path: Path,
    implementation_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report = _validate_sources(
        report_path=report_path,
        metrics_path=metrics_path,
        corpus_path=corpus_path,
        design_path=design_path,
        inventory_path=inventory_path,
        implementation_path=implementation_path,
    )
    components = _component_rows(report)
    comparisons = _comparison_rows(report)
    models = _model_rows(report)
    progress, progress_fields = _progress_rows(previous_progress_path)
    decision = _decision_analysis(report, components)
    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "components.csv", components)
    _write_csv(output_dir / "feature-comparisons.csv", comparisons)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(
        output_dir / "progress.csv",
        [{field: row.get(field, "") for field in progress_fields} for row in progress],
    )
    _write_text(charts / "primary-binary-skill.svg", _binary_skill_svg(components))
    _write_text(
        charts / "primary-continuous-skill.svg",
        _continuous_skill_svg(components),
    )
    _write_text(
        charts / "spot-perpetual-increment.svg",
        _increment_svg(components),
    )
    _write_text(charts / "day-block-confidence.svg", _confidence_svg(components))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    _write_text(output_dir / "README.md", _readme(components))
    write_json_atomic(output_dir / "decision-analysis.json", decision, indent=2)
    shutil.copyfile(report_path, output_dir / "evaluation.json")
    shutil.copyfile(metrics_path, output_dir / "metrics.csv")
    shutil.copyfile(corpus_path, output_dir / "corpus-ingestion.json")
    shutil.copyfile(design_path, output_dir / "design.json")
    shutil.copyfile(inventory_path, output_dir / "inventory.json")
    shutil.copyfile(implementation_path, output_dir / "implementation.json")
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "publisher_path": "tools/publish_round72_price_discovery.py",
        "publisher_git_blob_oid": _git(
            "rev-parse", "HEAD:tools/publish_round72_price_discovery.py"
        ),
        "source": {
            "execution_commit": EXECUTION_COMMIT,
            "evaluation_file_sha256": REPORT_FILE_SHA256,
            "evaluation_report_sha256": REPORT_SHA256,
            "metrics_file_sha256": METRICS_FILE_SHA256,
            "corpus_file_sha256": CORPUS_FILE_SHA256,
            "corpus_report_sha256": CORPUS_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "design_sha256": DESIGN_SHA256,
            "inventory_file_sha256": INVENTORY_FILE_SHA256,
            "inventory_sha256": INVENTORY_SHA256,
            "implementation_file_sha256": IMPLEMENTATION_FILE_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "execution_source_blob_oids": SOURCE_BLOBS,
        },
        "claims": {
            "status": "rejected",
            "terminal_holdout_read": False,
            "profitability_claim": False,
            "execution_or_fill_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_authority": False,
        },
        "result": {
            "decision": report["decision"],
            "primary_gate_passed": False,
            "feature_comparison_passes": 0,
            "feature_comparisons": 36,
            "component_passes": 0,
            "components": 9,
            "models": 216,
            "backend": report["backend"],
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2)
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / "data" / "round72-price-discovery-evaluation.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "data" / "round72-price-discovery-metrics.csv",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "round72-spot-perpetual-corpus-ingestion.json",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-072-spot-perpetual-price-discovery-design.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=research / "round-072-spot-perpetual-inventory.json",
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=research / "round-072-price-discovery-implementation.json",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=research / "latest" / "progress.csv",
    )
    parser.add_argument("--output", type=Path, default=research / "latest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        report_path=arguments.evaluation.resolve(),
        metrics_path=arguments.metrics.resolve(),
        corpus_path=arguments.corpus.resolve(),
        design_path=arguments.design.resolve(),
        inventory_path=arguments.inventory.resolve(),
        implementation_path=arguments.implementation.resolve(),
        previous_progress_path=arguments.progress.resolve(),
        output_dir=arguments.output.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": publication["round"],
                "publication_canonical_sha256": publication[
                    "publication_canonical_sha256"
                ],
                "artifacts": len(publication["artifacts"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
