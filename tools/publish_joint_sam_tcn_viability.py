"""Publish verified Round 45 joint AdamW/SAM TCN evidence and charts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gzip
import html
import json
import math
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
from tools.publish_distributional_tcn_viability import (  # noqa: E402
    _canonical_identity,
    _clean_output,
    _csv_rows,
    _file_sha256,
    _read_object,
    _source_rows,
)
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_joint_sam_tcn_viability import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 45
PUBLICATION_SCHEMA = "joint-sam-distributional-tcn-publication-v1"
CANDIDATE_COLORS = {"joint_adamw": "#2563a6", "joint_sam": "#0f766e"}
CANDIDATE_LABELS = {"joint_adamw": "Joint AdamW", "joint_sam": "Joint SAM"}
SCENARIO_COLORS = {"base": "#2563a6", "stress": "#b42318"}
HORIZONS = (1, 4, 12, 24)


def _validated_source(
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], str, str]:
    design = _read_object(design_path, "Round 45 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 45 design")
    binding = _read_object(binding_path, "Round 45 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 45 binding")
    report = _read_object(evidence_root / "report.json", "Round 45 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 45 report"
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
        raise ValueError("Round 45 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    compute = report.get("compute")
    candidates = report.get("candidates")
    optimizer_gate = report.get("optimizer_ablation_gate")
    claims = report.get("claims")
    outputs = report.get("outputs")
    if (
        report.get("status") != "complete"
        or not isinstance(dataset, Mapping)
        or dataset.get("timestamps") != 30_647
        or dataset.get("rows") != 91_941
        or dataset.get("feature_count_per_symbol") != 71
        or dataset.get("joint_input_channels") != 213
        or dataset.get("symbols") != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        or not isinstance(compute, Mapping)
        or compute.get("backend_kind") != "directml"
        or compute.get("model_artifacts") != 6
        or compute.get("all_artifacts_exact_reload") is not True
        or compute.get("preflight", {}).get("cpu_fallback_warning_count") != 0
        or not isinstance(candidates, list)
        or {item.get("candidate_id") for item in candidates}
        != {"joint_adamw", "joint_sam"}
        or any(
            item.get("forecast_diagnostics", {}).get("gate", {}).get("passed")
            is not False
            for item in candidates
        )
        or any(
            item.get("economic_gate", {}).get("passed") is not False
            for item in candidates
        )
        or any(item.get("fixed_ledger_under_stress") is not True for item in candidates)
        or any(len(item.get("models", [])) != 3 for item in candidates)
        or any(
            model.get("reload_max_abs_prediction_error") != 0.0
            or model.get("warning_count") != 0
            for item in candidates
            for model in item.get("models", [])
        )
        or not isinstance(optimizer_gate, Mapping)
        or optimizer_gate.get("passed") is not False
        or not isinstance(claims, Mapping)
        or any(
            claims.get(field) is not False
            for field in (
                "optimizer_ablation_gate_passed",
                "sam_improvement_established",
                "profitability_established",
                "ai_improvement_established",
                "selection_confirmation_established",
                "promotion_authorized",
                "testnet_or_live_trading_authorized",
                "leverage_authorized",
            )
        )
        or report.get("hourly_ledger_rows") != 26_112
        or not isinstance(outputs, list)
        or len(outputs) != 29
    ):
        raise ValueError("Round 45 source, model, or rejection evidence drifted")
    for item in outputs:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 45 evidence output drifted: {path}")
    return report, report_sha, binding_sha


def _candidate(report: Mapping[str, object], candidate_id: str) -> Mapping[str, object]:
    return next(
        item for item in report["candidates"] if item["candidate_id"] == candidate_id
    )


def _daily_equity_rows(evidence_root: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    with gzip.open(
        evidence_root / "hourly_ledger.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        for row in csv.DictReader(stream):
            grouped[(row["candidate_id"], row["scenario"])].append(
                (row["decision_time_utc"][:10], float(row["portfolio_return_bps"]))
            )
    output: list[dict[str, object]] = []
    for (candidate_id, scenario), observations in sorted(grouped.items()):
        equity = 1.0
        current_date = ""
        day_last_equity = 1.0
        day_net_bps = 0.0
        day_hours = 0
        for date, return_bps in observations:
            if current_date and date != current_date:
                output.append(
                    {
                        "candidate_id": candidate_id,
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
                    "candidate_id": candidate_id,
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
    sam = _candidate(report, "joint_sam")
    base = sam["base"]
    one_hour = next(
        item
        for item in sam["forecast_diagnostics"]["horizons"]
        if item["horizon_hours"] == 1
    )
    evaluation = next(
        item for item in report["dataset"]["roles"] if item["role"] == "evaluation"
    )
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "joint cross-asset TCN; AdamW versus SAM",
            "periods": "2022-01-01..2025-06-30 roles; eval 2024-10-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "3600;14400;43200;86400",
            "feature_set": "213 joint causal channels; 71 per symbol",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "spearman_ic": one_hour["pooled_median_spearman"],
            "selected_signals": base["trades"],
            "executable_trades": base["trades"],
            "mean_net_bps": base["mean_hourly_portfolio_bps"],
            "status": "rejected",
            "source_file": "verified Round 45 joint SAM TCN report; SAM point estimate is not validated",
            "best_policy_trades": base["trades"],
            "best_policy_total_net_bps": 10_000.0
            * float(base["total_net_return_fraction"]),
            "best_policy_mean_net_bps": base["mean_hourly_portfolio_bps"],
            "best_policy_max_drawdown_bps": 10_000.0
            * float(base["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": base["profit_factor"],
            "best_model_id": "joint_sam_descriptive_only",
            "daily_model_fits": 6,
            "accepted_thresholds": 0,
            "ensemble_models": 3,
            "valid_barrier_rows": evaluation["symbol_rows"],
            "policy_eligible_rows": evaluation["symbol_rows"],
            "development_consumed": True,
            "architecture_gates_passed": 0,
            "architecture_gate_count": 2,
        }
    )
    rows.append(row)
    rows.sort(key=lambda item: int(item["round"]))
    return fields, rows


def _forecast_quality_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 760
    left, right, top, panel_height, gap = 120, 70, 145, 205, 100
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Round 45 preserved weak forecast skill but SAM did not improve it",
        "Consumed development evaluation, 2024-10-01 through 2025-06-30 UTC; pinball skill is relative to training-role unconditional quantiles.",
    )
    for panel, (title, key, lower, upper, threshold) in enumerate(
        (
            (
                "Pinball skill versus unconditional baseline",
                "pinball_skill",
                0.0,
                0.045,
                0.01,
            ),
            (
                "Pooled median-return Spearman coefficient",
                "pooled_median_spearman",
                0.0,
                0.075,
                0.0,
            ),
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
        group_width = chart_width / len(HORIZONS)
        for horizon_index, horizon in enumerate(HORIZONS):
            center = left + group_width * (horizon_index + 0.5)
            for candidate_index, candidate_id in enumerate(CANDIDATE_COLORS):
                candidate = _candidate(report, candidate_id)
                item = next(
                    row
                    for row in candidate["forecast_diagnostics"]["horizons"]
                    if row["horizon_hours"] == horizon
                )
                value = float(item[key])
                bar_x = center - 58 + candidate_index * 62
                py = y(value)
                base_y = y(0.0)
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{py:.1f}" width="54" height="{max(2.0, base_y - py):.1f}" fill="{CANDIDATE_COLORS[candidate_id]}" fill-opacity="0.88"/>'
                )
                lines.append(
                    f'<text x="{bar_x + 27:.1f}" y="{py - 9:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{horizon} h</text>'
                )
    for index, candidate_id in enumerate(CANDIDATE_COLORS):
        x = left + index * 230
        lines.append(
            f'<rect x="{x}" y="{height - 42}" width="24" height="12" fill="{CANDIDATE_COLORS[candidate_id]}"/><text x="{x + 34}" y="{height - 31}" class="note">{CANDIDATE_LABELS[candidate_id]}</text>'
        )
    lines.append(
        f'<text x="{width - right}" y="{height - 31}" text-anchor="end" class="note">Both candidates failed the separate seed-stability gate.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _seed_stability_svg(rows: Sequence[Mapping[str, object]]) -> str:
    minima: dict[tuple[str, int], float] = {}
    for candidate_id in CANDIDATE_COLORS:
        for horizon in HORIZONS:
            minima[(candidate_id, horizon)] = min(
                float(row["median_prediction_spearman"])
                for row in rows
                if row["candidate_id"] == candidate_id
                and int(row["horizon_hours"]) == horizon
            )
    width, height = 1500, 650
    left, right, top, chart_height = 120, 70, 145, 330
    chart_width = width - left - right
    lower, upper = 0.0, 0.55
    lines = _svg_start(
        width,
        height,
        "Joint training sharply degraded cross-seed stability",
        "Minimum pairwise Spearman correlation of seed median predictions by horizon; the frozen floor was 0.500.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )

    def x(index: int) -> float:
        return left + chart_width * index / (len(HORIZONS) - 1)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    for tick in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        py = y(tick)
        style = (
            'stroke="#b42318" stroke-width="2" stroke-dasharray="8 6"'
            if tick == 0.5
            else 'class="grid"'
        )
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" {style}/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>'
        )
    for candidate_index, (candidate_id, color) in enumerate(CANDIDATE_COLORS.items()):
        points = []
        for index, horizon in enumerate(HORIZONS):
            value = minima[(candidate_id, horizon)]
            points.append((x(index), y(value)))
            label_y = y(value) - 12 if candidate_index == 0 else y(value) + 22
            lines.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="6" fill="{color}"/><text x="{x(index):.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
            )
        lines.append(
            '<polyline points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + f'" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    for index, horizon in enumerate(HORIZONS):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{horizon} h</text>'
        )
    for index, candidate_id in enumerate(CANDIDATE_COLORS):
        legend_x = left + index * 240
        lines.append(
            f'<line x1="{legend_x}" y1="{height - 68}" x2="{legend_x + 30}" y2="{height - 68}" stroke="{CANDIDATE_COLORS[candidate_id]}" stroke-width="4"/><text x="{legend_x + 40}" y="{height - 63}" class="note">{CANDIDATE_LABELS[candidate_id]}</text>'
        )
    overall = {
        candidate_id: min(
            value for (name, _), value in minima.items() if name == candidate_id
        )
        for candidate_id in CANDIDATE_COLORS
    }
    lines.append(
        f'<text x="{left}" y="{height - 25}" class="note">Overall minima: AdamW {overall["joint_adamw"]:.3f}; SAM {overall["joint_sam"]:.3f}. SAM changed the minimum by {overall["joint_sam"] - overall["joint_adamw"]:+.3f}, so the optimizer-ablation gate failed.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _symbol_forecast_svg(report: Mapping[str, object]) -> str:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    width, height = 1500, 760
    left, right, top, panel_height, gap = 120, 70, 145, 205, 100
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Per-symbol diagnostics expose a weak SOL 24-hour quantile forecast",
        "SAM ensemble on consumed development evaluation; positive rank association does not imply after-cost profitability.",
    )
    sam = _candidate(report, "joint_sam")
    rows = sam["forecast_diagnostics"]["symbol_horizons"]
    for panel, (title, key, lower, upper) in enumerate(
        (
            (
                "Pinball skill versus unconditional baseline",
                "pinball_skill",
                -0.012,
                0.04,
            ),
            ("Median-return Spearman coefficient", "pooled_median_spearman", 0.0, 0.09),
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
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{("zero" if abs(tick) < 1e-9 else "grid")}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.3f}</text>'
            )
        group_width = chart_width / len(HORIZONS)
        for horizon_index, horizon in enumerate(HORIZONS):
            center = left + group_width * (horizon_index + 0.5)
            for symbol_index, symbol in enumerate(symbols):
                item = next(
                    row
                    for row in rows
                    if row["horizon_hours"] == horizon and row["symbol"] == symbol
                )
                value = float(item[key])
                bar_x = center - 75 + symbol_index * 52
                zero_y = y(0.0)
                value_y = y(value)
                color = ("#2563a6", "#0f766e", "#b7791f")[symbol_index]
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="44" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{color}" fill-opacity="0.88"/>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{horizon} h</text>'
                )
    for index, (symbol, color) in enumerate(
        zip(symbols, ("#2563a6", "#0f766e", "#b7791f"), strict=True)
    ):
        x = left + index * 210
        lines.append(
            f'<rect x="{x}" y="{height - 42}" width="24" height="12" fill="{color}"/><text x="{x + 34}" y="{height - 31}" class="note">{symbol}</text>'
        )
    lines.append(
        f'<text x="{width - right}" y="{height - 31}" text-anchor="end" class="note">SOLUSDT 24 h pinball skill: -0.851%.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _economics_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 720
    left, right, top, chart_height = 130, 590, 155, 360
    chart_width = width - left - right
    lower, upper = -50.0, 30.0
    lines = _svg_start(
        width,
        height,
        "SAM's positive point estimate failed every robustness test",
        "Compounded net return on the fixed 2024-10-01 through 2025-06-30 UTC action ledgers; base is 6 bps and stress is 8 bps one-way.",
    )

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    for tick in (-40, -20, 0, 20):
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{("zero" if tick == 0 else "grid")}"/><text x="{left - 15}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
        )
    group_width = chart_width / 2
    for candidate_index, candidate_id in enumerate(CANDIDATE_COLORS):
        candidate = _candidate(report, candidate_id)
        center = left + group_width * (candidate_index + 0.5)
        for scenario_index, scenario in enumerate(("base", "stress")):
            value = 100.0 * float(candidate[scenario]["total_net_return_fraction"])
            bar_x = center - 72 + scenario_index * 78
            zero_y = y(0.0)
            value_y = y(value)
            lines.append(
                f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="66" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{SCENARIO_COLORS[scenario]}" fill-opacity="0.88"/>'
            )
            text_y = value_y - 10 if value >= 0 else value_y + 20
            lines.append(
                f'<text x="{bar_x + 33:.1f}" y="{text_y:.1f}" text-anchor="middle" class="value">{value:+.1f}%</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{CANDIDATE_LABELS[candidate_id]}</text>'
        )
    panel_x = 995
    sam = _candidate(report, "joint_sam")
    adamw = _candidate(report, "joint_adamw")
    lines.extend(
        [
            f'<text x="{panel_x}" y="165" class="label">Rejected diagnostics</text>',
            f'<text x="{panel_x}" y="215" class="value">SAM: {sam["base"]["trades"]} trades / {sam["base"]["active_days"]} active days</text>',
            f'<text x="{panel_x}" y="255" class="value">SAM max drawdown: {100 * float(sam["base"]["maximum_drawdown_fraction"]):.2f}%</text>',
            f'<text x="{panel_x}" y="295" class="value">SAM profit factor: {float(sam["base"]["profit_factor"]):.3f}</text>',
            f'<text x="{panel_x}" y="335" class="value">SAM positive months: {sam["base"]["positive_months"]}/9</text>',
            f'<text x="{panel_x}" y="375" class="value">SAM stress CI lower: {float(sam["stress"]["bootstrap_mean_hourly_portfolio_bps"]["lower_bps"]):+.3f} bps/hour</text>',
            f'<text x="{panel_x}" y="430" class="note">AdamW max drawdown: {100 * float(adamw["base"]["maximum_drawdown_fraction"]):.2f}%</text>',
            f'<text x="{panel_x}" y="462" class="note">AdamW profit factor: {float(adamw["base"]["profit_factor"]):.3f}</text>',
            f'<text x="{panel_x}" y="494" class="note">AdamW stress CI lower: {float(adamw["stress"]["bootstrap_mean_hourly_portfolio_bps"]["lower_bps"]):+.3f} bps/hour</text>',
            '<rect x="130" y="613" width="24" height="12" fill="#2563a6"/><text x="164" y="624" class="note">base 6 bps</text>',
            '<rect x="315" y="613" width="24" height="12" fill="#b42318"/><text x="349" y="624" class="note">stress 8 bps, same ledger</text>',
            '<text x="130" y="671" class="note">The positive SAM point estimate is selection-contaminated and accompanied by excessive drawdown, weak profit factor, regime concentration, and a confidence interval crossing zero.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _monthly_economics_svg(monthly: Sequence[Mapping[str, object]]) -> str:
    months = sorted({str(row["month"]) for row in monthly})
    indexed = {
        (str(row["candidate_id"]), str(row["scenario"]), str(row["month"])): 100.0
        * float(row["total_net_return_fraction"])
        for row in monthly
    }
    width, height = 1500, 780
    left, right, top, panel_height, gap = 120, 70, 145, 220, 105
    chart_width = width - left - right
    lower, upper = -25.0, 35.0
    lines = _svg_start(
        width,
        height,
        "Monthly returns reveal regime concentration and reversal",
        "All nine consumed development months are shown; base and stress use identical decisions for each candidate.",
    )
    for panel, candidate_id in enumerate(CANDIDATE_COLORS):
        panel_top = top + panel * (panel_height + gap)
        lines.append(
            f'<text x="{left}" y="{panel_top - 20}" class="label">{CANDIDATE_LABELS[candidate_id]}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{panel_top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
        )

        def y(value: float) -> float:
            return panel_top + panel_height * (upper - value) / (upper - lower)

        for tick in (-20, 0, 20):
            py = y(float(tick))
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{("zero" if tick == 0 else "grid")}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
            )
        group_width = chart_width / len(months)
        for month_index, month in enumerate(months):
            center = left + group_width * (month_index + 0.5)
            for scenario_index, scenario in enumerate(("base", "stress")):
                value = indexed[(candidate_id, scenario, month)]
                zero_y = y(0.0)
                value_y = y(value)
                bar_x = center - 25 + scenario_index * 27
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="23" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{SCENARIO_COLORS[scenario]}" fill-opacity="0.88"/>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{html.escape(month)}</text>'
                )
    lines.extend(
        [
            '<rect x="120" y="742" width="24" height="12" fill="#2563a6"/><text x="154" y="753" class="note">base</text>',
            '<rect x="245" y="742" width="24" height="12" fill="#b42318"/><text x="279" y="753" class="note">stress</text>',
            '<text x="1430" y="753" text-anchor="end" class="note">SAM was positive in only 4/9 months; February 2025 dominated its cumulative point estimate.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _equity_svg(daily_rows: Sequence[Mapping[str, object]]) -> str:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in daily_rows:
        grouped[(str(row["candidate_id"]), str(row["scenario"]))].append(row)
    dates = sorted({str(row["date_utc"]) for row in daily_rows})
    width, height = 1500, 720
    left, right, top, chart_height = 120, 70, 145, 390
    chart_width = width - left - right
    plotted_values = [
        100.0 * float(row["cumulative_return_fraction"]) for row in daily_rows
    ]
    lower = min(-5.0, 10.0 * math.floor(min(plotted_values) / 10.0) - 5.0)
    upper = max(5.0, 10.0 * math.ceil(max(plotted_values) / 10.0) + 5.0)
    lines = _svg_start(
        width,
        height,
        "Dated equity confirms unstable, high-drawdown economics",
        "Daily close of every hourly portfolio ledger, 2024-10-01 through 2025-06-29 UTC; no missing months or synthetic interpolation.",
    )

    def x(date: str) -> float:
        return left + chart_width * dates.index(date) / (len(dates) - 1)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    first_tick = 20 * math.ceil(lower / 20)
    last_tick = 20 * math.floor(upper / 20)
    for tick in range(first_tick, last_tick + 1, 20):
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{("zero" if tick == 0 else "grid")}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
        )
    styles = {
        ("joint_adamw", "base"): ("#2563a6", ""),
        ("joint_adamw", "stress"): ("#2563a6", "8 6"),
        ("joint_sam", "base"): ("#0f766e", ""),
        ("joint_sam", "stress"): ("#b42318", "8 6"),
    }
    for key, (color, dash) in styles.items():
        points = " ".join(
            f"{x(str(row['date_utc'])):.1f},{y(100 * float(row['cumulative_return_fraction'])):.1f}"
            for row in grouped[key]
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash_attr}/>'
        )
    for date in ("2024-10-01", "2024-12-01", "2025-02-01", "2025-04-01", "2025-06-01"):
        if date in dates:
            lines.append(
                f'<text x="{x(date):.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{date}</text>'
            )
    legend = (
        ("AdamW base", "#2563a6", ""),
        ("AdamW stress", "#2563a6", "8 6"),
        ("SAM base", "#0f766e", ""),
        ("SAM stress", "#b42318", "8 6"),
    )
    for index, (label, color, dash) in enumerate(legend):
        lx = left + index * 245
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{lx}" y1="{height - 78}" x2="{lx + 30}" y2="{height - 78}" stroke="{color}" stroke-width="4"{dash_attr}/><text x="{lx + 40}" y="{height - 73}" class="note">{label}</text>'
        )
    lines.append(
        f'<text x="{left}" y="{height - 27}" class="note">SAM base ended positive, but its 29.48% drawdown and negative familywise bootstrap lower bound reject the policy.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _failure_analysis(
    report: Mapping[str, object],
    report_sha: str,
    binding_sha: str,
    evidence_root: Path,
) -> dict[str, object]:
    analysis: dict[str, object] = {
        "schema_version": "round-045-failure-analysis-v1",
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
            "directml_model_artifacts": report["compute"]["model_artifacts"],
            "exact_reload_failures": sum(
                model["reload_max_abs_prediction_error"] != 0.0
                for candidate in report["candidates"]
                for model in candidate["models"]
            ),
            "cpu_fallback_warnings": report["compute"]["preflight"][
                "cpu_fallback_warning_count"
            ],
            "declared_output_count": len(report["outputs"]),
        },
        "optimizer_ablation_gate": report["optimizer_ablation_gate"],
        "candidate_results": {
            candidate["candidate_id"]: {
                "forecast_gate": candidate["forecast_diagnostics"]["gate"],
                "horizons": candidate["forecast_diagnostics"]["horizons"],
                "symbol_horizons": candidate["forecast_diagnostics"]["symbol_horizons"],
                "base": candidate["base"],
                "stress": candidate["stress"],
                "economic_gate": candidate["economic_gate"],
            }
            for candidate in report["candidates"]
        },
        "critical_interpretation": [
            "Both joint candidates beat the unconditional quantile baseline and had positive pooled rank association at all four horizons, so the architecture retained weak forecast information.",
            "Joint cross-asset training sharply reduced the minimum pairwise seed prediction correlation to 0.189 for AdamW and 0.181 for SAM, far below the frozen 0.500 floor and below Round 44's 0.452 minimum.",
            "SAM marginally reduced median early-stop pinball loss but worsened minimum seed stability by 0.008, so the pre-registered optimizer-ablation gate rejected it.",
            "The consensus policy generated 898 AdamW and 947 SAM trades over 272 active days, proving that Round 44's one-trade behavior was an action-mapping defect rather than lack of forecast variation.",
            "SAM's base point estimate was positive 22.264%, but it had a 29.477% maximum drawdown, 1.029 profit factor, only four positive months, and a -0.741 bps/hour base bootstrap lower bound. The fixed stress replay remained positive 7.761% but its lower bound was -1.043 bps/hour.",
            "February 2025 supplied most of SAM's cumulative gain, while later months reversed materially. This is regime concentration, not robust evidence of a persistent edge.",
            "SOLUSDT 24-hour pinball skill was negative for both candidates even though rank association was positive, exposing a horizon-specific calibration weakness hidden by pooled metrics.",
            "No language model received market features, outcomes, forecasts, or order authority. Fin-R1 remained outside execution after failing the separate reconnect-risk benchmark, so AI uplift remains false.",
        ],
        "decisions": [
            "Do not promote, leverage, testnet-run, or describe either Round 45 candidate as profitable.",
            "Reject SAM as an optimizer improvement under the frozen contract; do not reinterpret its positive point estimate after observing evaluation outcomes.",
            "Do not retain the joint all-symbol backbone as the next stability candidate because its seed disagreement is materially worse than the predecessor.",
            "Retain the cost-aware seed-consensus action concept for future calibration-only work because it restored realistic activity, but require drawdown, bootstrap, month-count, and profit-factor gates before any authority.",
            "Use deterministic or explicitly stability-regularized model aggregation next, and pre-register regime-aware exposure controls using training and calibration roles only.",
        ],
        "next_model_requirements": [
            "Return to per-symbol/shared-backbone causal sequence features and isolate stochastic variation in the prediction head instead of coupling all 213 channels in one unstable backbone.",
            "Test weight averaging or a deterministic convex quantile readout under a frozen multi-seed agreement contract before touching the consumed evaluation ledger.",
            "Use ensemble disagreement as an abstention and sizing input, with thresholds learned only from training and calibration roles.",
            "Add a stateful volatility, liquidity, and forecast-dispersion risk governor that caps drawdown without using evaluation outcomes or reducing fixed execution costs.",
            "Keep BTCUSDT, ETHUSDT, and SOLUSDT, unlevered one-third sleeves, exact base/stress ledgers, and the existing no-authority AI boundary until economics pass.",
            "Improve the local finance-language-model risk-review benchmark independently; only run a paired AI uplift experiment after deterministic numerical gates pass.",
        ],
        "generated_at_utc": report["generated_at_utc"],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    optimizer = report["optimizer_ablation_gate"]
    adamw = _candidate(report, "joint_adamw")
    sam = _candidate(report, "joint_sam")
    horizon_rows = "\n".join(
        f"| {item['horizon_hours']} h | {100 * float(item['pinball_skill']):.2f}% | {float(item['pooled_median_spearman']):.4f} | {100 * float(next(row for row in sam['forecast_diagnostics']['horizons'] if row['horizon_hours'] == item['horizon_hours'])['pinball_skill']):.2f}% | {float(next(row for row in sam['forecast_diagnostics']['horizons'] if row['horizon_hours'] == item['horizon_hours'])['pooled_median_spearman']):.4f} |"
        for item in adamw["forecast_diagnostics"]["horizons"]
    )
    return f"""# Round 45: Joint TCN and SAM

> **Beta research warning:** rejected, selection-contaminated development evidence. No model is approved for testnet, live day trading, leverage, or autonomous execution.

Round 45 compared a joint 213-channel cross-asset distributional TCN trained with AdamW and sharpness-aware minimization (SAM). Six three-seed artifacts trained on the AMD GPU through DirectML, reloaded exactly, and emitted zero fallback warnings.

![Forecast quality](charts/forecast-quality.svg)

| Horizon | AdamW skill | AdamW Spearman | SAM skill | SAM Spearman |
|---:|---:|---:|---:|---:|
{horizon_rows}

Both candidates preserved weak forecast information, but joint training made seed agreement materially worse. AdamW reached only `{float(optimizer["adamw_minimum_seed_stability"]):.3f}` and SAM `{float(optimizer["sam_minimum_seed_stability"]):.3f}` against the frozen `0.500` floor. SAM therefore did **not** establish an optimizer improvement.

![Seed stability](charts/seed-stability.svg)

![Per-symbol forecast quality](charts/symbol-forecast.svg)

The consensus mapping restored activity: AdamW made `{adamw["base"]["trades"]}` trades and SAM `{sam["base"]["trades"]}` across `{sam["base"]["active_days"]}` active days. AdamW lost `{100 * float(adamw["base"]["total_net_return_fraction"]):.2f}%`. SAM's `{100 * float(sam["base"]["total_net_return_fraction"]):+.2f}%` base point estimate is **not validated**: maximum drawdown was `{100 * float(sam["base"]["maximum_drawdown_fraction"]):.2f}%`, profit factor `{float(sam["base"]["profit_factor"]):.3f}`, only `{sam["base"]["positive_months"]}/9` months were positive, and the stress bootstrap lower bound was `{float(sam["stress"]["bootstrap_mean_hourly_portfolio_bps"]["lower_bps"]):+.3f}` bps/hour.

![Policy economics](charts/policy-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Research progress](charts/research-progress.svg)

Data: [horizons](horizons.csv) | [symbol horizons](symbol-horizons.csv) | [forecast diagnostics](diagnostics.csv) | [seed stability](seed-stability.csv) | [models](models.csv) | [roles](roles.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-045-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
"""


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
    seed_stability = _csv_rows(evidence_root / "seed_stability.csv")
    models = _csv_rows(evidence_root / "models.csv")
    roles = _csv_rows(evidence_root / "roles.csv")
    trades = _csv_rows(evidence_root / "trades.csv")
    replays = _csv_rows(evidence_root / "replays.csv")
    monthly = _csv_rows(evidence_root / "monthly_economics.csv")
    symbols = _csv_rows(evidence_root / "symbol_economics.csv")
    daily_rows = _daily_equity_rows(evidence_root)
    sources = _source_rows(report)
    progress_fields, progress_rows = _progress_rows(prior_progress_path, report)
    horizons = [
        {"candidate_id": candidate["candidate_id"], **row}
        for candidate in report["candidates"]
        for row in candidate["forecast_diagnostics"]["horizons"]
    ]
    symbol_horizons = [
        {"candidate_id": candidate["candidate_id"], **row}
        for candidate in report["candidates"]
        for row in candidate["forecast_diagnostics"]["symbol_horizons"]
    ]
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "horizons.csv",
        output_dir / "symbol-horizons.csv",
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
        charts / "symbol-forecast.svg",
        charts / "policy-economics.svg",
        charts / "monthly-economics.svg",
        charts / "daily-equity.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "horizons.csv", horizons)
    _write_csv(output_dir / "symbol-horizons.csv", symbol_horizons)
    _write_csv(output_dir / "diagnostics.csv", diagnostics)
    _write_csv(output_dir / "seed-stability.csv", seed_stability)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "roles.csv", roles)
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
    _write_text(charts / "seed-stability.svg", _seed_stability_svg(seed_stability))
    _write_text(charts / "symbol-forecast.svg", _symbol_forecast_svg(report))
    _write_text(charts / "policy-economics.svg", _economics_svg(report))
    _write_text(charts / "monthly-economics.svg", _monthly_economics_svg(monthly))
    _write_text(charts / "daily-equity.svg", _equity_svg(daily_rows))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress_rows))
    failure = _failure_analysis(report, source_report_sha, binding_sha, evidence_root)
    _write_text(
        failure_path,
        json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "joint_sam_distributional_tcn_graph_data",
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
        "directml_model_artifact_count": report["compute"]["model_artifacts"],
        "hourly_ledger_rows": report["hourly_ledger_rows"],
        "candidate_forecast_gate_pass_count": 0,
        "candidate_economic_gate_pass_count": 0,
        "optimizer_ablation_gate_passed": False,
        "sam_improvement_claim": False,
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
        "--design", type=Path, default=research / "round-045-joint-sam-tcn-design.json"
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-045-joint-sam-tcn-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-analysis",
        type=Path,
        default=research / "round-045-failure-analysis.json",
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
