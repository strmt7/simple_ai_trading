"""Publish verified Round 44 distributional TCN evidence and charts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import html
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.publish_cross_asset_cost_aware_ai_ablation import (  # noqa: E402
    _research_progress_svg,
    _svg_start,
)
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_distributional_tcn_viability import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 44
PUBLICATION_SCHEMA = "distributional-tcn-publication-v1"
HORIZON_COLORS = {
    1: "#2563a6",
    4: "#0f766e",
    12: "#b7791f",
    24: "#7b559c",
}


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_identity(value: Mapping[str, object], field: str, label: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field, ""))
    if len(claimed) != 64 or _canonical_sha256(canonical) != claimed:
        raise ValueError(f"{label} canonical identity is invalid")
    return claimed


def _validated_source(
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], str, str]:
    design = _read_object(design_path, "Round 44 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 44 design")
    binding = _read_object(binding_path, "Round 44 binding")
    binding_sha = _canonical_identity(
        binding, "binding_sha256", "Round 44 binding"
    )
    report = _read_object(evidence_root / "report.json", "Round 44 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 44 report"
    )
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or binding.get("schema_version") != BINDING_SCHEMA
        or report.get("schema_version") != REPORT_SCHEMA
        or any(item.get("round") != ROUND for item in (design, binding, report))
        or binding.get("design_sha256") != design_sha
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
    ):
        raise ValueError("Round 44 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    compute = report.get("compute")
    models = report.get("models")
    forecast = report.get("forecast_diagnostics")
    policy = report.get("descriptive_policy")
    economics = report.get("economic_gate")
    claims = report.get("claims")
    outputs = report.get("outputs")
    if (
        report.get("status") != "complete"
        or not isinstance(dataset, Mapping)
        or dataset.get("timestamps") != 30_647
        or dataset.get("rows") != 91_941
        or dataset.get("feature_count") != 71
        or not isinstance(compute, Mapping)
        or compute.get("backend_kind") != "directml"
        or compute.get("model_artifacts") != 3
        or compute.get("all_artifacts_exact_reload") is not True
        or compute.get("preflight", {}).get("cpu_fallback_warning_count") != 0
        or not isinstance(models, list)
        or len(models) != 3
        or any(model.get("reload_max_abs_prediction_error") != 0.0 for model in models)
        or any(model.get("warning_count") != 0 for model in models)
        or not isinstance(forecast, Mapping)
        or forecast.get("gate", {}).get("passed") is not False
        or forecast.get("gate", {}).get("reasons")
        != ["seed_prediction_stability_below_0_5"]
        or len(forecast.get("horizons", [])) != 4
        or not isinstance(policy, Mapping)
        or policy.get("trade_count") != 1
        or policy.get("fixed_ledger_under_stress") is not True
        or not isinstance(economics, Mapping)
        or economics.get("passed") is not False
        or not isinstance(claims, Mapping)
        or any(
            claims.get(field) is not False
            for field in (
                "forecast_gate_passed",
                "economic_gate_passed",
                "profitability_established",
                "ai_improvement_established",
                "selection_confirmation_established",
                "promotion_authorized",
                "testnet_or_live_trading_authorized",
                "leverage_authorized",
            )
        )
        or report.get("hourly_ledger_rows") != 13_056
        or not isinstance(outputs, list)
    ):
        raise ValueError("Round 44 source, model, or rejection evidence drifted")
    for item in outputs:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 44 evidence output drifted: {path}")
    return report, report_sha, binding_sha


def _csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    source = report["source_evidence"]
    price = source["price_flow"]
    derivatives = {item["symbol"]: item for item in source["derivatives_series"]}
    archives = {item["symbol"]: item for item in price["archive_evidence"]}
    rows: list[dict[str, object]] = []
    for item in price["series_evidence"]:
        symbol = item["symbol"]
        rows.append(
            {
                **item,
                "complete_verified_archives": archives[symbol][
                    "complete_verified_archives"
                ],
                "archive_first_period": archives[symbol]["first_period"],
                "archive_last_period": archives[symbol]["last_period"],
                **derivatives[symbol],
                "price_panel_sha256": price["panel_stream_sha256"],
                "derivatives_panel_sha256": source["derivatives_panel_sha256"],
                "source_certificate_sha256": source["source_certificate_sha256"],
            }
        )
    return rows


def _daily_equity_rows(evidence_root: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    with gzip.open(
        evidence_root / "hourly_ledger.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        for row in csv.DictReader(stream):
            grouped[row["scenario"]].append(
                (row["decision_time_utc"][:10], float(row["portfolio_return_bps"]))
            )
    output: list[dict[str, object]] = []
    for scenario, observations in sorted(grouped.items()):
        equity = 1.0
        current_date = ""
        day_last_equity = 1.0
        day_net_bps = 0.0
        day_hours = 0
        for date, return_bps in observations:
            if current_date and date != current_date:
                output.append(
                    {
                        "scenario": scenario,
                        "date_utc": current_date,
                        "equity": day_last_equity,
                        "cumulative_return_fraction": day_last_equity - 1.0,
                        "day_additive_net_bps": day_net_bps,
                        "hours": day_hours,
                    }
                )
                day_net_bps = 0.0
                day_hours = 0
            current_date = date
            equity *= 1.0 + return_bps / 10_000.0
            day_last_equity = equity
            day_net_bps += return_bps
            day_hours += 1
        if current_date:
            output.append(
                {
                    "scenario": scenario,
                    "date_utc": current_date,
                    "equity": day_last_equity,
                    "cumulative_return_fraction": day_last_equity - 1.0,
                    "day_additive_net_bps": day_net_bps,
                    "hours": day_hours,
                }
            )
    return output


def _progress_rows(
    prior_path: Path,
    report: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    with prior_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    rows = [row for row in rows if int(row["round"]) != ROUND]
    base = report["descriptive_policy"]["base"]
    one_hour = next(
        item
        for item in report["forecast_diagnostics"]["horizons"]
        if item["horizon_hours"] == 1
    )
    evaluation = next(
        item for item in report["dataset"]["roles"] if item["role"] == "evaluation"
    )
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "causal multi-horizon distributional TCN",
            "periods": "2022-01-01..2025-06-30 roles; eval 2024-10-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "3600;14400;43200;86400",
            "feature_set": "71 causal features; 127-hour temporal receptive field",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "spearman_ic": one_hour["pooled_median_spearman"],
            "selected_signals": base["trades"],
            "executable_trades": base["trades"],
            "mean_net_bps": base["mean_hourly_portfolio_bps"],
            "status": "rejected",
            "source_file": "verified Round 44 distributional TCN report; mean is hourly portfolio bps",
            "best_policy_trades": base["trades"],
            "best_policy_total_net_bps": 10_000.0
            * float(base["total_net_return_fraction"]),
            "best_policy_mean_net_bps": base["mean_hourly_portfolio_bps"],
            "best_policy_max_drawdown_bps": 10_000.0
            * float(base["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": base["profit_factor"],
            "best_model_id": "distributional_tcn_three_seed_median",
            "daily_model_fits": 3,
            "accepted_thresholds": 0,
            "ensemble_models": 3,
            "valid_barrier_rows": evaluation["symbol_rows"],
            "policy_eligible_rows": evaluation["symbol_rows"],
            "development_consumed": True,
            "architecture_gates_passed": 0,
            "architecture_gate_count": 1,
        }
    )
    rows.append(row)
    rows.sort(key=lambda item: int(item["round"]))
    return fields, rows


def _forecast_quality_svg(report: Mapping[str, object]) -> str:
    horizons = report["forecast_diagnostics"]["horizons"]
    width, height = 1500, 760
    left, right, top, panel_height, gap = 120, 70, 145, 205, 100
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Round 44 learned measurable multi-horizon forecast information",
        "Consumed development evaluation, 2024-10-01 through 2025-06-30 UTC; skill is relative to training-role unconditional quantiles.",
    )
    for panel, (title, key, lower, upper, threshold) in enumerate(
        (
            ("Pinball skill versus unconditional baseline", "pinball_skill", 0.0, 0.05, 0.01),
            ("Pooled median-return Spearman coefficient", "pooled_median_spearman", 0.0, 0.07, 0.0),
        )
    ):
        panel_top = top + panel * (panel_height + gap)
        lines.append(
            f'<text x="{left}" y="{panel_top - 20}" class="label">{title}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{panel_top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
        )

        def y(value: float) -> float:
            return panel_top + panel_height * (upper - value) / (upper - lower)

        for tick_index in range(6):
            tick = lower + tick_index * (upper - lower) / 5
            py = y(tick)
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
            )
        threshold_y = y(threshold)
        lines.append(
            f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left + chart_width}" y2="{threshold_y:.1f}" stroke="#b42318" stroke-width="2" stroke-dasharray="8 6"/>'
        )
        group = chart_width / len(horizons)
        for index, item in enumerate(horizons):
            value = float(item[key])
            center = left + group * (index + 0.5)
            py = y(value)
            base_y = y(0.0)
            color = HORIZON_COLORS[int(item["horizon_hours"])]
            lines.append(
                f'<rect x="{center - 52:.1f}" y="{py:.1f}" width="104" height="{max(2.0, base_y - py):.1f}" fill="{color}" fill-opacity="0.86"/>'
            )
            lines.append(
                f'<text x="{center:.1f}" y="{py - 10:.1f}" text-anchor="middle" class="value">{value:.4f}</text>'
            )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{item["horizon_hours"]} h</text>'
                )
    lines.extend(
        [
            f'<text x="{left}" y="{height - 25}" class="note">All four horizons passed skill, rank-association, monthly-sign, and interval-coverage checks; the aggregate gate failed only seed stability.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _seed_stability_svg(report: Mapping[str, object]) -> str:
    rows = report["seed_stability"]
    pairs = sorted({(int(row["left_seed"]), int(row["right_seed"])) for row in rows})
    horizons = (1, 4, 12, 24)
    colors = ("#2563a6", "#0f766e", "#b7791f")
    width, height = 1500, 690
    left, right, top, chart_height = 120, 70, 145, 360
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Longer-horizon seed stability did not meet the frozen floor",
        "Pairwise Spearman correlation of seed median predictions on 2024-10-01 through 2025-06-30 UTC; required minimum 0.500.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )

    def x(index: int) -> float:
        return left + chart_width * index / (len(horizons) - 1)

    def y(value: float) -> float:
        return top + chart_height * (0.65 - value) / 0.25

    for tick in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        py = y(tick)
        style = 'stroke="#b42318" stroke-width="2" stroke-dasharray="8 6"' if tick == 0.50 else 'class="grid"'
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" {style}/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    indexed = {
        (int(row["left_seed"]), int(row["right_seed"]), int(row["horizon_hours"])): float(
            row["median_prediction_spearman"]
        )
        for row in rows
    }
    for pair, color in zip(pairs, colors, strict=True):
        points = " ".join(
            f"{x(index):.1f},{y(indexed[(*pair, horizon)]):.1f}"
            for index, horizon in enumerate(horizons)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for index, horizon in enumerate(horizons):
            value = indexed[(*pair, horizon)]
            lines.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="6" fill="{color}"/>'
            )
    for index, horizon in enumerate(horizons):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{horizon} h</text>'
        )
    for index, (pair, color) in enumerate(zip(pairs, colors, strict=True)):
        legend_x = left + index * 300
        lines.append(
            f'<line x1="{legend_x}" y1="{height - 82}" x2="{legend_x + 30}" y2="{height - 82}" stroke="{color}" stroke-width="4"/><text x="{legend_x + 40}" y="{height - 77}" class="note">seed {pair[0]} / {pair[1]}</text>'
        )
    minimum = min(float(row["median_prediction_spearman"]) for row in rows)
    lines.extend(
        [
            f'<text x="{left}" y="{height - 28}" class="note">Observed minimum {minimum:.3f}; four of twelve pair-horizon comparisons were below 0.500. No post-outcome exemption was applied.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _monthly_rank_svg(diagnostics: Sequence[Mapping[str, object]]) -> str:
    months = sorted({str(row["period"]) for row in diagnostics})
    horizons = (1, 4, 12, 24)
    indexed = {
        (str(row["period"]), int(row["horizon_hours"])): float(
            row["median_spearman"]
        )
        for row in diagnostics
    }
    width, height = 1500, 690
    left, right, top, chart_height = 120, 70, 145, 360
    chart_width = width - left - right
    lower, upper = -0.07, 0.23
    lines = _svg_start(
        width,
        height,
        "Monthly rank association varied by horizon and regime",
        "Pooled BTCUSDT, ETHUSDT, and SOLUSDT median-return Spearman coefficient; every month shown, 2024-10 through 2025-06 UTC.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )

    def x(index: int) -> float:
        return left + chart_width * index / (len(months) - 1)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    for tick in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0.0 else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.2f}</text>'
        )
    for horizon in horizons:
        color = HORIZON_COLORS[horizon]
        points = " ".join(
            f"{x(index):.1f},{y(indexed[(month, horizon)]):.1f}"
            for index, month in enumerate(months)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for index, month in enumerate(months):
            lines.append(
                f'<circle cx="{x(index):.1f}" cy="{y(indexed[(month, horizon)]):.1f}" r="5" fill="{color}"/>'
            )
    for index, month in enumerate(months):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + chart_height + 27}" text-anchor="middle" class="axis">{html.escape(month)}</text>'
        )
    for index, horizon in enumerate(horizons):
        legend_x = left + index * 190
        color = HORIZON_COLORS[horizon]
        lines.append(
            f'<line x1="{legend_x}" y1="{height - 79}" x2="{legend_x + 28}" y2="{height - 79}" stroke="{color}" stroke-width="4"/><text x="{legend_x + 38}" y="{height - 74}" class="note">{horizon} h</text>'
        )
    lines.extend(
        [
            f'<text x="{left}" y="{height - 26}" class="note">Positive-month counts were 9/9, 5/9, 8/9, and 6/9 for 1 h, 4 h, 12 h, and 24 h respectively.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _economics_svg(report: Mapping[str, object]) -> str:
    policy = report["descriptive_policy"]
    base = policy["base"]
    stress = policy["stress"]
    values = (
        100.0 * float(base["total_net_return_fraction"]),
        100.0 * float(stress["total_net_return_fraction"]),
    )
    width, height = 1500, 650
    left, top, chart_width, chart_height = 150, 150, 700, 300
    lower, upper = -0.40, 0.05
    lines = _svg_start(
        width,
        height,
        "The frozen lower-quartile policy produced one losing trade",
        "Compounded net return over 2024-10-01 through 2025-06-30 UTC; stress reprices the identical action ledger at 8 bps versus 6 bps one-way.",
    )

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    for tick in (-0.4, -0.3, -0.2, -0.1, 0.0):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0.0 else "grid"}"/><text x="{left - 15}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    for index, (label, value, color) in enumerate(
        zip(("base 6 bps", "stress 8 bps"), values, ("#2563a6", "#b42318"), strict=True)
    ):
        center = left + 240 + index * 290
        py = y(value)
        zero = y(0.0)
        lines.append(
            f'<rect x="{center - 65}" y="{zero:.1f}" width="130" height="{max(2.0, py - zero):.1f}" fill="{color}" fill-opacity="0.86"/>'
        )
        lines.append(
            f'<text x="{center}" y="{py + 24:.1f}" text-anchor="middle" class="value">{value:+.3f}%</text><text x="{center}" y="{top + chart_height + 30}" text-anchor="middle" class="axis">{label}</text>'
        )
    panel_x = 980
    lines.extend(
        [
            f'<text x="{panel_x}" y="165" class="label">Policy diagnostics</text>',
            f'<text x="{panel_x}" y="215" class="value">1 trade</text>',
            f'<text x="{panel_x}" y="255" class="note">BTCUSDT short, 12 h</text>',
            f'<text x="{panel_x}" y="300" class="value">{100 * float(base["maximum_drawdown_fraction"]):.3f}% max drawdown</text>',
            f'<text x="{panel_x}" y="345" class="value">{float(base["profit_factor"]):.3f} profit factor</text>',
            f'<text x="{panel_x}" y="390" class="value">{base["active_days"]} active day</text>',
            f'<text x="{panel_x}" y="435" class="note">Stress bootstrap lower: {float(stress["bootstrap_mean_hourly_portfolio_bps"]["lower_bps"]):+.4f} bps/hour</text>',
            '<text x="150" y="560" class="note">This descriptive policy has no capacity or profitability evidence. Its failure does not erase the separate forecast-quality result.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _equity_svg(daily_rows: Sequence[Mapping[str, object]]) -> str:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in daily_rows:
        grouped[str(row["scenario"])].append(row)
    dates = sorted({str(row["date_utc"]) for row in daily_rows})
    width, height = 1500, 650
    left, right, top, chart_height = 120, 70, 145, 330
    chart_width = width - left - right
    lower, upper = -0.40, 0.05
    lines = _svg_start(
        width,
        height,
        "Dated equity confirms near-total abstention and one loss",
        "Daily close of the full hourly portfolio ledger, 2024-10-01 through 2025-06-29 UTC; flat sections are genuine zero exposure.",
    )

    def x(date: str) -> float:
        return left + chart_width * dates.index(date) / (len(dates) - 1)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    for tick in (-0.4, -0.3, -0.2, -0.1, 0.0):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0.0 else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    for scenario, color in (("base", "#2563a6"), ("stress", "#b42318")):
        points = " ".join(
            f"{x(str(row['date_utc'])):.1f},{y(100 * float(row['cumulative_return_fraction'])):.1f}"
            for row in grouped[scenario]
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    for date in (
        "2024-10-01",
        "2024-12-01",
        "2025-02-01",
        "2025-04-01",
        "2025-06-01",
    ):
        if date in dates:
            px = x(date)
            lines.append(
                f'<text x="{px:.1f}" y="{top + chart_height + 27}" text-anchor="middle" class="axis">{date}</text>'
            )
    lines.extend(
        [
            '<line x1="120" y1="570" x2="150" y2="570" stroke="#2563a6" stroke-width="4"/><text x="160" y="575" class="note">base 6 bps</text>',
            '<line x1="330" y1="570" x2="360" y2="570" stroke="#b42318" stroke-width="4"/><text x="370" y="575" class="note">stress 8 bps, same ledger</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _failure_analysis(
    report: Mapping[str, object],
    report_sha: str,
    binding_sha: str,
    evidence_root: Path,
) -> dict[str, object]:
    forecast = report["forecast_diagnostics"]
    base = report["descriptive_policy"]["base"]
    stress = report["descriptive_policy"]["stress"]
    analysis: dict[str, object] = {
        "schema_version": "round-044-failure-analysis-v1",
        "round": ROUND,
        "status": "rejected",
        "evidence": {
            "report_canonical_sha256": report_sha,
            "report_file_sha256": _file_sha256(evidence_root / "report.json"),
            "design_sha256": report["design_sha256"],
            "binding_sha256": binding_sha,
            "implementation_commit": report["implementation_commit"],
            "dataset_sha256": report["dataset"]["dataset_sha256"],
            "hourly_rows": report["dataset"]["rows"],
            "evaluation_timestamps": next(
                item["timestamps"]
                for item in report["dataset"]["roles"]
                if item["role"] == "evaluation"
            ),
            "directml_model_artifacts": len(report["models"]),
            "exact_reload_failures": sum(
                item["reload_max_abs_prediction_error"] != 0.0
                for item in report["models"]
            ),
            "cpu_fallback_warnings": report["compute"]["preflight"][
                "cpu_fallback_warning_count"
            ],
        },
        "horizon_results": forecast["horizons"],
        "forecast_gate": forecast["gate"],
        "economic_results": {
            "trade_count": report["descriptive_policy"]["trade_count"],
            "base": base,
            "stress": stress,
            "gate": report["economic_gate"],
        },
        "critical_interpretation": [
            "All four horizons beat the training-role unconditional quantile baseline by more than the frozen one-percent pinball-skill floor, and all four pooled median-return Spearman coefficients were positive.",
            "The forecast gate failed only stochastic stability: the minimum pairwise seed median-prediction correlation was below 0.500, concentrated in 4-hour, 12-hour, and 24-hour comparisons.",
            "Central 80-percent and 50-percent interval coverage stayed inside every frozen calibration bound, so interval calibration was not the rejection cause.",
            "The lower-quartile-after-cost policy admitted one BTCUSDT short in nine evaluation months. It lost 95.618 bps at base cost and cannot establish activity, capacity, or profitability.",
            "The fixed stress replay changed only one-way cost, so its lower return is monotonic and directly attributable to the additional four basis points of round-trip cost.",
            "The local Fin-R1 model failed its frozen reconnect-risk benchmark and no language model received features, outcomes, numerical forecast authority, or order authority. AI uplift remains false.",
        ],
        "decisions": [
            "Do not promote, leverage, testnet-run, or represent the Round 44 model or policy as profitable.",
            "Retain causal distributional sequence modeling as the next research family because forecast skill and calibration passed; do not retain the one-trade lower-quartile policy as an economic candidate.",
            "Do not lower the 0.500 seed-stability floor or the 12 bps base round-trip cost after observing the evaluation outcomes.",
            "Improve stochastic stability through a separately frozen training design, then derive any action mapping from training and calibration roles only before replaying consumed evaluation data.",
            "Persist future prediction tensors and per-symbol diagnostics as hashed external evidence to avoid unnecessary source reconstruction and to expose cross-asset concentration.",
        ],
        "next_model_requirements": [
            "Pre-register a stability-focused ablation that changes optimization or regularization without searching the consumed evaluation period.",
            "Measure per-symbol, per-horizon pinball skill, median rank association, directional calibration, and tail coverage before portfolio conversion.",
            "Replace the all-or-nothing lower-quartile entry condition with a calibration-only expected-utility policy that still pays the full fixed action ledger costs and permits abstention.",
            "Keep the same BTCUSDT, ETHUSDT, and SOLUSDT source, fixed one-third sleeves, unlevered exposure, and matched base/stress ledger until an unlevered economic edge passes.",
            "Keep language models outside the real-time execution loop; any AI contribution requires a separate paired uplift contract after the deterministic model passes.",
        ],
        "generated_at_utc": report["generated_at_utc"],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    horizons = report["forecast_diagnostics"]["horizons"]
    rows = "\n".join(
        f"| {item['horizon_hours']} h | {100 * float(item['pinball_skill']):.2f}% | {float(item['pooled_median_spearman']):.4f} | {item['positive_monthly_median_spearman_count']}/9 | {float(item['coverage_80']):.3f} | {float(item['coverage_50']):.3f} |"
        for item in horizons
    )
    forecast = report["forecast_diagnostics"]["gate"]
    base = report["descriptive_policy"]["base"]
    stress = report["descriptive_policy"]["stress"]
    return f"""# Round 44: Causal Distributional TCN

> **Beta research warning:** rejected, selection-contaminated development evidence. No model is approved for testnet, live day trading, leverage, or autonomous execution.

Round 44 replaced one-hour point forecasts with a 127-hour causal temporal convolutional network that predicts calibrated 1, 4, 12, and 24-hour return distributions. Three seeds trained on the AMD GPU through DirectML and reloaded exactly with zero fallback warnings.

![Forecast quality](charts/forecast-quality.svg)

| Horizon | Pinball skill | Median Spearman | Positive months | 80% coverage | 50% coverage |
|---:|---:|---:|---:|---:|---:|
{rows}

Forecast learning improved materially, but the frozen gate failed because minimum pairwise seed stability was `{float(forecast['minimum_pairwise_seed_median_prediction_spearman']):.3f}`, below `0.500`.

![Seed stability](charts/seed-stability.svg)

![Monthly rank association](charts/monthly-rank.svg)

The descriptive lower-quartile policy admitted one BTCUSDT short. It returned `{100 * float(base['total_net_return_fraction']):+.3f}%` at 6 bps one-way and `{100 * float(stress['total_net_return_fraction']):+.3f}%` when the identical ledger was repriced at 8 bps. This is not viable economic evidence.

![Policy economics](charts/policy-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Research progress](charts/research-progress.svg)

Data: [horizons](horizons.csv) | [monthly forecast diagnostics](diagnostics.csv) | [seed stability](seed-stability.csv) | [models](models.csv) | [roles](roles.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbols](symbols.csv) | [daily equity](daily-equity.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-044-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
"""


def _clean_output(output_dir: Path, expected: set[Path]) -> None:
    if not output_dir.exists():
        return
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_file() and path not in expected:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def publish(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
    failure_path: Path,
) -> dict[str, object]:
    report, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path, binding_path
    )
    diagnostics = _csv_rows(evidence_root / "forecast_diagnostics.csv")
    trades = _csv_rows(evidence_root / "trades.csv")
    replays = _csv_rows(evidence_root / "replays.csv")
    monthly = _csv_rows(evidence_root / "monthly_economics.csv")
    symbols = _csv_rows(evidence_root / "symbol_economics.csv")
    daily_rows = _daily_equity_rows(evidence_root)
    sources = _source_rows(report)
    progress_fields, progress_rows = _progress_rows(prior_progress_path, report)
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "horizons.csv",
        output_dir / "diagnostics.csv",
        output_dir / "seed-stability.csv",
        output_dir / "models.csv",
        output_dir / "roles.csv",
        output_dir / "trades.csv",
        output_dir / "replays.csv",
        output_dir / "monthly.csv",
        output_dir / "symbols.csv",
        output_dir / "daily-equity.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "forecast-quality.svg",
        charts / "seed-stability.svg",
        charts / "monthly-rank.svg",
        charts / "policy-economics.svg",
        charts / "daily-equity.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "horizons.csv", report["forecast_diagnostics"]["horizons"])
    _write_csv(output_dir / "diagnostics.csv", diagnostics)
    _write_csv(output_dir / "seed-stability.csv", report["seed_stability"])
    _write_csv(output_dir / "models.csv", report["models"])
    _write_csv(output_dir / "roles.csv", report["dataset"]["roles"])
    _write_csv(output_dir / "trades.csv", trades)
    _write_csv(output_dir / "replays.csv", replays)
    _write_csv(output_dir / "monthly.csv", monthly)
    _write_csv(output_dir / "symbols.csv", symbols)
    _write_csv(output_dir / "daily-equity.csv", daily_rows)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress_rows, fields=progress_fields)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "forecast-quality.svg", _forecast_quality_svg(report))
    _write_text(charts / "seed-stability.svg", _seed_stability_svg(report))
    _write_text(charts / "monthly-rank.svg", _monthly_rank_svg(diagnostics))
    _write_text(charts / "policy-economics.svg", _economics_svg(report))
    _write_text(charts / "daily-equity.svg", _equity_svg(daily_rows))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress_rows))
    failure = _failure_analysis(
        report, source_report_sha, binding_sha, evidence_root
    )
    _write_text(
        failure_path,
        json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "distributional_tcn_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "hourly_dataset_rows": report["dataset"]["rows"],
        "evaluation_timestamps": next(
            item["timestamps"]
            for item in report["dataset"]["roles"]
            if item["role"] == "evaluation"
        ),
        "directml_model_artifact_count": len(report["models"]),
        "hourly_ledger_rows": report["hourly_ledger_rows"],
        "forecast_gate_passed": False,
        "economic_gate_passed": False,
        "selection_contaminated": True,
        "development_only": True,
        "trading_authority": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "leverage_applied": False,
        "failure_analysis_sha256": failure["analysis_sha256"],
        "failure_analysis_file_sha256": _file_sha256(failure_path),
        "artifact_integrity": [_artifact(path, output_dir) for path in artifact_paths],
        "publication_sha256": "PENDING",
    }
    canonical = dict(publication)
    canonical.pop("publication_sha256")
    publication["publication_sha256"] = _canonical_sha256(canonical)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-044-distributional-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-044-distributional-tcn-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-analysis",
        type=Path,
        default=research / "round-044-failure-analysis.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        evidence_root=arguments.evidence_root.resolve(),
        design_path=arguments.design.resolve(),
        binding_path=arguments.binding.resolve(),
        prior_progress_path=arguments.prior_progress.resolve(),
        output_dir=arguments.output_dir.resolve(),
        failure_path=arguments.failure_analysis.resolve(),
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
