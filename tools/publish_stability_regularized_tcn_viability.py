"""Publish verified Round 46 stability-regularized TCN evidence and charts."""

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
)
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_stability_regularized_tcn_viability import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 46
PUBLICATION_SCHEMA = "stability-regularized-distributional-tcn-publication-v1"
CANDIDATES = ("wavebound_ema", "mutual_median_consistency")
CANDIDATE_COLORS = {
    "wavebound_ema": "#a33a32",
    "mutual_median_consistency": "#0f766e",
}
CANDIDATE_LABELS = {
    "wavebound_ema": "WaveBound EMA",
    "mutual_median_consistency": "Mutual consistency",
}
SCENARIO_COLORS = {"base": "#2563a6", "stress": "#b42318"}
SYMBOL_COLORS = {"BTCUSDT": "#2563a6", "ETHUSDT": "#0f766e", "SOLUSDT": "#b7791f"}
HORIZONS = (1, 4, 12, 24)


def _candidate(report: Mapping[str, object], candidate_id: str) -> Mapping[str, object]:
    return next(
        item for item in report["candidates"] if item["candidate_id"] == candidate_id
    )


def _validated_source(
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], str, str]:
    design = _read_object(design_path, "Round 46 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 46 design")
    binding = _read_object(binding_path, "Round 46 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 46 binding")
    report = _read_object(evidence_root / "report.json", "Round 46 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 46 report"
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
        raise ValueError("Round 46 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    compute = report.get("compute")
    candidates = report.get("candidates")
    claims = report.get("claims")
    outputs = report.get("outputs")
    if (
        report.get("status") != "complete"
        or not isinstance(dataset, Mapping)
        or dataset.get("timestamps") != 30_647
        or dataset.get("rows") != 91_941
        or dataset.get("feature_count_per_symbol") != 71
        or dataset.get("symbols") != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        or not isinstance(compute, Mapping)
        or compute.get("backend_kind") != "directml"
        or compute.get("model_artifacts") != 6
        or compute.get("all_artifacts_exact_reload") is not True
        or compute.get("preflight", {}).get("cpu_fallback_warning_count") != 0
        or not isinstance(candidates, list)
        or {item.get("candidate_id") for item in candidates} != set(CANDIDATES)
        or any(item.get("fixed_ledger_under_stress") is not True for item in candidates)
        or any(len(item.get("models", [])) != 3 for item in candidates)
        or any(
            model.get("reload_max_abs_prediction_error") != 0.0
            or model.get("warning_count") != 0
            for item in candidates
            for model in item.get("models", [])
        )
        or _candidate(report, "wavebound_ema")["forecast_diagnostics"]["gate"]["passed"]
        is not False
        or _candidate(report, "wavebound_ema")["mechanism_gate"]["passed"] is not False
        or _candidate(report, "mutual_median_consistency")["forecast_diagnostics"][
            "gate"
        ]["passed"]
        is not True
        or _candidate(report, "mutual_median_consistency")["mechanism_gate"]["passed"]
        is not True
        or any(
            item.get("economic_gate", {}).get("passed") is not False
            for item in candidates
        )
        or not isinstance(claims, Mapping)
        or claims.get("candidate_forecast_gate_pass_count") != 1
        or claims.get("candidate_mechanism_gate_pass_count") != 1
        or claims.get("candidate_economic_gate_pass_count") != 0
        or any(
            claims.get(field) is not False
            for field in (
                "profitability_established",
                "ai_improvement_established",
                "stability_uplift_established",
                "selection_confirmation_established",
                "promotion_authorized",
                "testnet_or_live_trading_authorized",
                "leverage_authorized",
            )
        )
        or report.get("hourly_ledger_rows") != 26_112
        or not isinstance(outputs, list)
        or len(outputs) != 26
    ):
        raise ValueError("Round 46 source, model, or rejection evidence drifted")
    for item in outputs:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 46 evidence output drifted: {path}")
    for item in dataset["derived_cache_inputs"]:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 46 cache input drifted: {path}")
    return report, report_sha, binding_sha


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


def _training_rows(evidence_root: Path) -> list[dict[str, object]]:
    payload = _read_object(evidence_root / "training_history.json", "training history")
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("Round 46 training history has no candidates")
    rows: list[dict[str, object]] = []
    fields: list[str] = []
    for candidate_id in CANDIDATES:
        history = candidates.get(candidate_id)
        if not isinstance(history, list):
            raise ValueError(f"Round 46 training history is missing {candidate_id}")
        for source in history:
            row = dict(source)
            for key, value in tuple(row.items()):
                if isinstance(value, (dict, list)):
                    row[key] = json.dumps(
                        value,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                if key not in fields:
                    fields.append(key)
            rows.append(row)
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    lineage = report["source_lineage"]
    rows = [
        {
            "artifact_class": "source_certificate",
            "name": "round38_derivatives_source_certificate",
            "path": report["binding"]["source_certificate"]["path"],
            "bytes": "",
            "file_sha256": lineage["source_certificate_file_sha256"],
            "canonical_sha256": lineage["source_certificate_canonical_sha256"],
        },
        {
            "artifact_class": "predecessor_report",
            "name": "round45_joint_sam_report",
            "path": report["binding"]["predecessor_report"]["path"],
            "bytes": "",
            "file_sha256": lineage["predecessor_report_file_sha256"],
            "canonical_sha256": lineage["predecessor_report_canonical_sha256"],
        },
    ]
    for item in report["dataset"]["derived_cache_inputs"]:
        rows.append(
            {
                "artifact_class": "verified_derived_cache",
                "name": item["name"],
                "path": item["path"],
                "bytes": item["bytes"],
                "file_sha256": item["sha256"],
                "canonical_sha256": "",
            }
        )
    return rows


def _progress_rows(
    prior_path: Path,
    report: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    with prior_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields or int(rows[-1]["round"]) not in (45, 46):
        raise ValueError("Round 46 prior progress history is invalid")
    rows = [row for row in rows if int(row["round"]) != ROUND]
    mutual = _candidate(report, "mutual_median_consistency")
    base = mutual["base"]
    one_hour = next(
        item
        for item in mutual["forecast_diagnostics"]["horizons"]
        if item["horizon_hours"] == 1
    )
    evaluation = next(
        item for item in report["dataset"]["roles"] if item["role"] == "evaluation"
    )
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "stability-regularized TCN; WaveBound versus mutual consistency",
            "periods": "2022-01-01..2025-06-30 roles; eval 2024-10-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "3600;14400;43200;86400",
            "feature_set": "71 causal features; per-symbol shared TCN",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "spearman_ic": one_hour["pooled_median_spearman"],
            "selected_signals": base["trades"],
            "executable_trades": base["trades"],
            "mean_net_bps": base["mean_hourly_portfolio_bps"],
            "status": "rejected",
            "source_file": "verified Round 46 mutual-consistency TCN report; economic gate failed",
            "best_policy_trades": base["trades"],
            "best_policy_total_net_bps": 10_000.0
            * float(base["total_net_return_fraction"]),
            "best_policy_mean_net_bps": base["mean_hourly_portfolio_bps"],
            "best_policy_max_drawdown_bps": 10_000.0
            * float(base["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": base["profit_factor"],
            "best_model_id": "mutual_median_consistency_descriptive_only",
            "daily_model_fits": 6,
            "accepted_thresholds": 0,
            "ensemble_models": 3,
            "valid_barrier_rows": evaluation["symbol_rows"],
            "policy_eligible_rows": evaluation["symbol_rows"],
            "development_consumed": True,
            "architecture_gates_passed": 1,
            "architecture_gate_count": 2,
        }
    )
    rows.append(row)
    rows.sort(key=lambda item: int(item["round"]))
    return fields, rows


def _forecast_quality_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 780
    left, right, top, panel_height, gap = 120, 70, 145, 220, 105
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Mutual consistency passed the frozen forecast screen",
        "Consumed development evaluation, 2024-10-01 through 2025-06-30 UTC; skill is relative to training-role unconditional quantiles.",
    )
    panels = (
        (
            "Pinball skill versus unconditional baseline",
            "pinball_skill",
            -14.0,
            6.0,
            1.0,
            100.0,
        ),
        (
            "Pooled median-return Spearman coefficient",
            "pooled_median_spearman",
            0.0,
            0.08,
            0.0,
            1.0,
        ),
    )
    for panel, (title, key, lower, upper, threshold, scale) in enumerate(panels):
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
            label = f"{tick:+.0f}%" if scale == 100.0 else f"{tick:.3f}"
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{label}</text>'
            )
        threshold_y = y(threshold)
        lines.append(
            f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left + chart_width}" y2="{threshold_y:.1f}" stroke="#6b7280" stroke-width="2" stroke-dasharray="8 6"/>'
        )
        group_width = chart_width / len(HORIZONS)
        for horizon_index, horizon in enumerate(HORIZONS):
            center = left + group_width * (horizon_index + 0.5)
            for candidate_index, candidate_id in enumerate(CANDIDATES):
                candidate = _candidate(report, candidate_id)
                item = next(
                    row
                    for row in candidate["forecast_diagnostics"]["horizons"]
                    if row["horizon_hours"] == horizon
                )
                value = float(item[key]) * scale
                bar_x = center - 58 + candidate_index * 62
                zero_y = y(0.0)
                value_y = y(value)
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="54" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{CANDIDATE_COLORS[candidate_id]}" fill-opacity="0.88"/>'
                )
                label_y = value_y - 9 if value >= 0 else value_y + 18
                label = f"{value:+.1f}%" if scale == 100.0 else f"{value:.3f}"
                lines.append(
                    f'<text x="{bar_x + 27:.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{label}</text>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{horizon} h</text>'
                )
    for index, candidate_id in enumerate(CANDIDATES):
        x = left + index * 260
        lines.append(
            f'<rect x="{x}" y="{height - 42}" width="24" height="12" fill="{CANDIDATE_COLORS[candidate_id]}"/><text x="{x + 34}" y="{height - 31}" class="note">{CANDIDATE_LABELS[candidate_id]}</text>'
        )
    lines.append(
        f'<text x="{width - right}" y="{height - 31}" text-anchor="end" class="note">Mutual consistency passed all four horizon skill and rank checks; this is not an economic pass.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _seed_stability_svg(rows: Sequence[Mapping[str, object]]) -> str:
    minima = {
        (candidate_id, horizon): min(
            float(row["median_prediction_spearman"])
            for row in rows
            if row["candidate_id"] == candidate_id
            and int(row["horizon_hours"]) == horizon
        )
        for candidate_id in CANDIDATES
        for horizon in HORIZONS
    }
    width, height = 1500, 660
    left, right, top, chart_height = 120, 70, 145, 350
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Peer consistency raised cross-seed agreement above the frozen floor",
        "Minimum pairwise Spearman correlation of seed median predictions by horizon; results remain selection-contaminated development evidence.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )

    def x(index: int) -> float:
        return left + chart_width * index / (len(HORIZONS) - 1)

    def y(value: float) -> float:
        return top + chart_height * (1.0 - value)

    for tick in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0):
        py = y(tick)
        style = (
            'stroke="#b42318" stroke-width="2" stroke-dasharray="8 6"'
            if tick == 0.5
            else 'class="grid"'
        )
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" {style}/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>'
        )
    for candidate_index, candidate_id in enumerate(CANDIDATES):
        points = []
        for index, horizon in enumerate(HORIZONS):
            value = minima[(candidate_id, horizon)]
            points.append((x(index), y(value)))
            label_y = y(value) - 12 if candidate_index == 1 else y(value) + 22
            lines.append(
                f'<circle cx="{x(index):.1f}" cy="{y(value):.1f}" r="6" fill="{CANDIDATE_COLORS[candidate_id]}"/><text x="{x(index):.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
            )
        lines.append(
            '<polyline points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + f'" fill="none" stroke="{CANDIDATE_COLORS[candidate_id]}" stroke-width="3"/>'
        )
    for index, horizon in enumerate(HORIZONS):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{horizon} h</text>'
        )
    for index, candidate_id in enumerate(CANDIDATES):
        legend_x = left + index * 270
        lines.append(
            f'<line x1="{legend_x}" y1="{height - 70}" x2="{legend_x + 30}" y2="{height - 70}" stroke="{CANDIDATE_COLORS[candidate_id]}" stroke-width="4"/><text x="{legend_x + 40}" y="{height - 65}" class="note">{CANDIDATE_LABELS[candidate_id]}</text>'
        )
    lines.append(
        f'<text x="{left}" y="{height - 27}" class="note">Overall minima: WaveBound {min(minima[("wavebound_ema", h)] for h in HORIZONS):.3f}; mutual consistency {min(minima[("mutual_median_consistency", h)] for h in HORIZONS):.3f}; frozen floor 0.500.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _training_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 760
    left, right, top, panel_height, gap = 120, 70, 145, 270, 105
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Training dynamics expose WaveBound seed dispersion",
        "Chronological early-stop pinball is normalized by training-role robust target scales; mutual peers share one selected epoch.",
    )
    wave_rows = [row for row in rows if row["candidate_id"] == "wavebound_ema"]
    mutual_rows = [
        row for row in rows if row["candidate_id"] == "mutual_median_consistency"
    ]
    values = [float(row["ema_early_stop_pinball"]) for row in wave_rows] + [
        float(row["ensemble_early_stop_pinball"]) for row in mutual_rows
    ]
    lower = 0.35
    upper = max(0.46, math.ceil(max(values) * 20.0) / 20.0)
    lines.append(
        f'<text x="{left}" y="{top - 20}" class="label">Early-stop pinball by epoch</text><rect x="{left}" y="{top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )

    def x(epoch: int) -> float:
        return left + chart_width * (epoch - 1) / 49.0

    def y(value: float) -> float:
        return top + panel_height * (upper - value) / (upper - lower)

    for tick_index in range(6):
        tick = lower + tick_index * (upper - lower) / 5
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
        )
    seed_colors = {4601: "#9b2c2c", 4602: "#c06c2b", 4603: "#7c3f96"}
    for seed, color in seed_colors.items():
        selected = [row for row in wave_rows if int(row["seed"]) == seed]
        points = " ".join(
            f"{x(int(row['epoch'])):.1f},{y(float(row['ema_early_stop_pinball'])):.1f}"
            for row in selected
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
    mutual_points = " ".join(
        f"{x(int(row['epoch'])):.1f},{y(float(row['ensemble_early_stop_pinball'])):.1f}"
        for row in mutual_rows
    )
    lines.append(
        f'<polyline points="{mutual_points}" fill="none" stroke="#0f766e" stroke-width="4"/>'
    )
    for epoch in (1, 10, 20, 30, 40, 50):
        lines.append(
            f'<text x="{x(epoch):.1f}" y="{top + panel_height + 26}" text-anchor="middle" class="axis">{epoch}</text>'
        )
    second_top = top + panel_height + gap
    second_height = 125
    lines.append(
        f'<text x="{left}" y="{second_top - 20}" class="label">Mutual standardized-median consistency penalty</text><rect x="{left}" y="{second_top}" width="{chart_width}" height="{second_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    max_consistency = max(float(row["training_consistency"]) for row in mutual_rows)

    def consistency_y(value: float) -> float:
        return second_top + second_height * (max_consistency - value) / max_consistency

    for tick in (0.0, max_consistency / 2.0, max_consistency):
        py = consistency_y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
        )
    consistency_points = " ".join(
        f"{x(int(row['epoch'])):.1f},{consistency_y(float(row['training_consistency'])):.1f}"
        for row in mutual_rows
    )
    lines.append(
        f'<polyline points="{consistency_points}" fill="none" stroke="#0f766e" stroke-width="3"/>'
    )
    legend = (
        ("Wave 4601", seed_colors[4601]),
        ("Wave 4602", seed_colors[4602]),
        ("Wave 4603", seed_colors[4603]),
        ("Mutual ensemble", "#0f766e"),
    )
    for index, (label, color) in enumerate(legend):
        lx = left + index * 245
        lines.append(
            f'<line x1="{lx}" y1="{height - 32}" x2="{lx + 30}" y2="{height - 32}" stroke="{color}" stroke-width="4"/><text x="{lx + 40}" y="{height - 27}" class="note">{label}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _symbol_forecast_svg(report: Mapping[str, object]) -> str:
    rows = _candidate(report, "mutual_median_consistency")["forecast_diagnostics"][
        "symbol_horizons"
    ]
    width, height = 1500, 760
    left, right, top, panel_height, gap = 120, 70, 145, 205, 100
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Mutual forecast quality is broad but SOL 24-hour skill is negative",
        "Per-symbol consumed development diagnostics; positive rank association does not establish after-cost profitability.",
    )
    panels = (
        ("Pinball skill versus symbol baseline", "pinball_skill", -1.0, 4.5, 100.0),
        (
            "Median-return Spearman coefficient",
            "pooled_median_spearman",
            -0.025,
            0.10,
            1.0,
        ),
    )
    symbols = tuple(SYMBOL_COLORS)
    for panel, (title, key, lower, upper, scale) in enumerate(panels):
        panel_top = top + panel * (panel_height + gap)
        lines.append(
            f'<text x="{left}" y="{panel_top - 20}" class="label">{title}</text><rect x="{left}" y="{panel_top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
        )

        def y(value: float) -> float:
            return panel_top + panel_height * (upper - value) / (upper - lower)

        for tick_index in range(6):
            tick = lower + tick_index * (upper - lower) / 5
            py = y(tick)
            label = f"{tick:+.1f}%" if scale == 100.0 else f"{tick:+.3f}"
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{label}</text>'
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
                value = float(item[key]) * scale
                bar_x = center - 75 + symbol_index * 52
                zero_y = y(0.0)
                value_y = y(value)
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="44" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{SYMBOL_COLORS[symbol]}" fill-opacity="0.88"/>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{horizon} h</text>'
                )
    for index, symbol in enumerate(symbols):
        x = left + index * 210
        lines.append(
            f'<rect x="{x}" y="{height - 42}" width="24" height="12" fill="{SYMBOL_COLORS[symbol]}"/><text x="{x + 34}" y="{height - 31}" class="note">{symbol}</text>'
        )
    lines.append(
        f'<text x="{width - right}" y="{height - 31}" text-anchor="end" class="note">SOLUSDT 24 h: -0.15% pinball skill and -0.0156 Spearman.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _economics_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 760
    left, right, top, panel_height, gap = 130, 70, 145, 210, 105
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Positive mutual point estimates failed institutional risk gates",
        "Fixed 2024-10-01 through 2025-06-30 UTC ledgers; base is 6 bps and stress is 8 bps one-way with no leverage.",
    )
    panels = (
        ("Compounded net return", "total_net_return_fraction", -40.0, 40.0),
        ("Maximum drawdown", "maximum_drawdown_fraction", 0.0, 55.0),
    )
    for panel, (title, key, lower, upper) in enumerate(panels):
        panel_top = top + panel * (panel_height + gap)
        lines.append(
            f'<text x="{left}" y="{panel_top - 20}" class="label">{title}</text><rect x="{left}" y="{panel_top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
        )

        def y(value: float) -> float:
            return panel_top + panel_height * (upper - value) / (upper - lower)

        for tick_index in range(5):
            tick = lower + tick_index * (upper - lower) / 4
            py = y(tick)
            tick_label = f"{tick:+.0f}%" if panel == 0 else f"{tick:.0f}%"
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick_label}</text>'
            )
        group_width = chart_width / len(CANDIDATES)
        for candidate_index, candidate_id in enumerate(CANDIDATES):
            center = left + group_width * (candidate_index + 0.5)
            candidate = _candidate(report, candidate_id)
            for scenario_index, scenario in enumerate(("base", "stress")):
                value = 100.0 * float(candidate[scenario][key])
                bar_x = center - 72 + scenario_index * 78
                zero_y = y(0.0)
                value_y = y(value)
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="66" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{SCENARIO_COLORS[scenario]}" fill-opacity="0.88"/>'
                )
                label_y = value_y - 9 if value >= 0 else value_y + 18
                if key == "maximum_drawdown_fraction":
                    label_y = value_y - 9
                lines.append(
                    f'<text x="{bar_x + 33:.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{value:.1f}%</text>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{CANDIDATE_LABELS[candidate_id]}</text>'
                )
    mutual = _candidate(report, "mutual_median_consistency")
    lines.extend(
        [
            '<rect x="130" y="722" width="24" height="12" fill="#2563a6"/><text x="164" y="733" class="note">base</text>',
            '<rect x="270" y="722" width="24" height="12" fill="#b42318"/><text x="304" y="733" class="note">stress, same ledger</text>',
            f'<text x="1430" y="733" text-anchor="end" class="note">Mutual: PF {float(mutual["base"]["profit_factor"]):.3f}; stress lower bound {float(mutual["stress"]["bootstrap_mean_hourly_portfolio_bps"]["lower_bps"]):+.3f} bps/hour; ETH P&amp;L share {100 * float(mutual["base"]["maximum_single_symbol_fraction_of_absolute_net_pnl"]):.1f}%.</text>',
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
    width, height = 1500, 790
    left, right, top, panel_height, gap = 120, 70, 145, 225, 105
    chart_width = width - left - right
    lower, upper = -30.0, 25.0
    lines = _svg_start(
        width,
        height,
        "Monthly economics remain regime-sensitive",
        "All nine consumed development months; base and stress use identical decisions for each candidate.",
    )
    for panel, candidate_id in enumerate(CANDIDATES):
        panel_top = top + panel * (panel_height + gap)
        lines.append(
            f'<text x="{left}" y="{panel_top - 20}" class="label">{CANDIDATE_LABELS[candidate_id]}</text><rect x="{left}" y="{panel_top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
        )

        def y(value: float) -> float:
            return panel_top + panel_height * (upper - value) / (upper - lower)

        for tick in (-20, 0, 20):
            py = y(float(tick))
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
            )
        group_width = chart_width / len(months)
        for month_index, month in enumerate(months):
            center = left + group_width * (month_index + 0.5)
            for scenario_index, scenario in enumerate(("base", "stress")):
                value = indexed[(candidate_id, scenario, month)]
                bar_x = center - 26 + scenario_index * 28
                zero_y = y(0.0)
                value_y = y(value)
                lines.append(
                    f'<rect x="{bar_x:.1f}" y="{min(zero_y, value_y):.1f}" width="23" height="{max(2.0, abs(zero_y - value_y)):.1f}" fill="{SCENARIO_COLORS[scenario]}" fill-opacity="0.88"/>'
                )
            if panel == 1:
                lines.append(
                    f'<text x="{center:.1f}" y="{panel_top + panel_height + 27}" text-anchor="middle" class="axis">{html.escape(month)}</text>'
                )
    lines.extend(
        [
            '<rect x="120" y="752" width="24" height="12" fill="#2563a6"/><text x="154" y="763" class="note">base</text>',
            '<rect x="245" y="752" width="24" height="12" fill="#b42318"/><text x="279" y="763" class="note">stress</text>',
            '<text x="1430" y="763" text-anchor="end" class="note">Mutual was positive in 6/9 months, but November and January losses remained material.</text>',
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
    plotted = [100.0 * float(row["cumulative_return_fraction"]) for row in daily_rows]
    lower = 10.0 * math.floor(min(plotted) / 10.0) - 5.0
    upper = 10.0 * math.ceil(max(plotted) / 10.0) + 5.0
    lines = _svg_start(
        width,
        height,
        "Dated equity confirms positive drift with excessive drawdown",
        "Daily close of every hourly portfolio ledger, 2024-10-01 through 2025-06-29 UTC; no interpolation or synthetic observations.",
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
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
        )
    styles = {
        ("wavebound_ema", "base"): ("#a33a32", ""),
        ("wavebound_ema", "stress"): ("#a33a32", "8 6"),
        ("mutual_median_consistency", "base"): ("#0f766e", ""),
        ("mutual_median_consistency", "stress"): ("#2563a6", "8 6"),
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
        ("Wave base", "#a33a32", ""),
        ("Wave stress", "#a33a32", "8 6"),
        ("Mutual base", "#0f766e", ""),
        ("Mutual stress", "#2563a6", "8 6"),
    )
    for index, (label, color, dash) in enumerate(legend):
        lx = left + index * 245
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{lx}" y1="{height - 78}" x2="{lx + 30}" y2="{height - 78}" stroke="{color}" stroke-width="4"{dash_attr}/><text x="{lx + 40}" y="{height - 73}" class="note">{label}</text>'
        )
    lines.append(
        f'<text x="{left}" y="{height - 27}" class="note">Mutual base ended +33.87%, but its 26.31% maximum drawdown and confidence interval crossing zero reject the policy.</text>'
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
        "schema_version": "round-046-failure-analysis-v1",
        "round": ROUND,
        "status": "economic_gate_rejected",
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
        "candidate_results": {
            candidate["candidate_id"]: {
                "forecast_gate": candidate["forecast_diagnostics"]["gate"],
                "mechanism_gate": candidate["mechanism_gate"],
                "horizons": candidate["forecast_diagnostics"]["horizons"],
                "symbol_horizons": candidate["forecast_diagnostics"]["symbol_horizons"],
                "base": candidate["base"],
                "stress": candidate["stress"],
                "economic_gate": candidate["economic_gate"],
            }
            for candidate in report["candidates"]
        },
        "critical_interpretation": [
            "WaveBound did not stabilize this quantile TCN adaptation. Its minimum pairwise seed correlation fell to 0.211, median early-stop pinball worsened 15.76% versus Round 44, and all four horizon pinball skills were negative.",
            "Mutual median consistency met the pre-registered mechanism gate: minimum pairwise seed correlation was 0.867, up 0.416 from Round 44, while common-epoch early-stop pinball improved 0.44%.",
            "The mutual ensemble passed every pooled forecast requirement: all four horizons beat unconditional pinball by 2.08% to 4.41%, all four pooled rank associations were positive, interval coverage stayed inside frozen bounds, and quantiles never crossed.",
            "The mutual fixed consensus ledger generated 935 trades over 272 active days. Its base point estimate was +33.868% and its stress point estimate +18.173%, but these are consumed development results and do not establish profitability.",
            "Mutual economics failed the frozen gate: base maximum drawdown was 26.306%, hourly profit factor was 1.036, ETH supplied 72.38% of absolute symbol net P&L, and the stress familywise bootstrap lower bound was -1.076 bps/hour.",
            "Six of nine mutual months were positive, but November 2024 and January 2025 each lost more than 10% at base cost. The dated ledger therefore remains regime-sensitive.",
            "SOLUSDT 24-hour forecast quality remained weak: -0.15% pinball skill and -0.0156 rank association. Pooled success does not remove this symbol-horizon defect.",
            "The local Qwen3 8B benchmark remains safety-reasoning evidence only. No language model received market features, future outcomes, numerical forecast authority, or order authority, so AI edge remains untested and false.",
        ],
        "decisions": [
            "Retain mutual median consistency as a development backbone candidate; reject WaveBound for this architecture and objective.",
            "Do not promote, leverage, testnet-run, or describe the mutual point estimates as validated profitability.",
            "Do not lower drawdown, bootstrap, concentration, or profit-factor gates after observing the positive cumulative return.",
            "Treat the next bottleneck as conditional exposure and trade-quality control, not raw activity or cross-seed forecast agreement.",
        ],
        "next_model_requirements": [
            "Freeze a risk-aware candidate using only training and calibration roles: volatility targeting, ensemble-disagreement abstention, symbol concentration limits, and stateful drawdown/cooldown controls.",
            "Use the mutual ensemble forecasts as immutable inputs and pre-register any trade-quality meta-label or exposure rule before accessing a new confirmation period.",
            "Address SOLUSDT 24-hour weakness with horizon-specific calibration or abstention without post-hoc evaluation tuning.",
            "Acquire and certify a genuinely untouched confirmation period with matched funding, premium, price, volume, and execution-cost lineage before any promotion claim.",
            "Only after deterministic forecast, risk, and economic gates pass, run a paired local-language-model risk-review ablation with identical actions and costs; deny direct numerical forecast or order authority.",
        ],
        "generated_at_utc": report["generated_at_utc"],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    wavebound = _candidate(report, "wavebound_ema")
    mutual = _candidate(report, "mutual_median_consistency")
    horizons = "\n".join(
        f"| {item['horizon_hours']} h | {100 * float(item['pinball_skill']):+.2f}% | {float(item['pooled_median_spearman']):.4f} | {100 * float(item['coverage_50']):.1f}% | {100 * float(item['coverage_80']):.1f}% |"
        for item in mutual["forecast_diagnostics"]["horizons"]
    )
    return f"""# Round 46: Stability-Regularized TCN

> **Beta research warning:** the economic gate failed. No model is approved for testnet, live day trading, leverage, or autonomous execution.

Round 46 compared WaveBound EMA error bounds with three co-trained distributional TCN peers. Six artifacts trained through DirectML on the AMD GPU, reloaded exactly, and emitted zero fallback warnings. The source dataset and every cached forward target were independently re-hashed and reproduced before training.

![Forecast quality](charts/forecast-quality.svg)

Mutual consistency passed the frozen forecast and mechanism screens. Minimum seed agreement rose from `0.452` in Round 44 to `{float(mutual["mechanism_gate"]["minimum_seed_stability"]):.3f}`; WaveBound fell to `{float(wavebound["mechanism_gate"]["minimum_seed_stability"]):.3f}`.

| Horizon | Pinball skill | Spearman | 50% coverage | 80% coverage |
|---:|---:|---:|---:|---:|
{horizons}

![Seed stability](charts/seed-stability.svg)

![Training dynamics](charts/training-dynamics.svg)

![Per-symbol forecast quality](charts/symbol-forecast.svg)

The mutual ledger made `{mutual["base"]["trades"]}` trades over `{mutual["base"]["active_days"]}` active days. Its base and stress point estimates were `{100 * float(mutual["base"]["total_net_return_fraction"]):+.2f}%` and `{100 * float(mutual["stress"]["total_net_return_fraction"]):+.2f}%`. They are **not validated profitability**: base drawdown was `{100 * float(mutual["base"]["maximum_drawdown_fraction"]):.2f}%`, hourly profit factor `{float(mutual["base"]["profit_factor"]):.3f}`, ETH represented `{100 * float(mutual["base"]["maximum_single_symbol_fraction_of_absolute_net_pnl"]):.1f}%` of absolute symbol P&L, and the stress bootstrap lower bound was `{float(mutual["stress"]["bootstrap_mean_hourly_portfolio_bps"]["lower_bps"]):+.3f}` bps/hour.

![Policy economics](charts/policy-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Research progress](charts/research-progress.svg)

Data: [horizons](horizons.csv) | [symbol horizons](symbol-horizons.csv) | [forecast diagnostics](diagnostics.csv) | [seed stability](seed-stability.csv) | [training](training.csv) | [models](models.csv) | [roles](roles.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-046-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
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
    training = _training_rows(evidence_root)
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
        output_dir / "training.csv",
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
        charts / "training-dynamics.svg",
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
    _write_csv(output_dir / "training.csv", training)
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
    _write_text(charts / "training-dynamics.svg", _training_svg(training))
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
        "artifact_class": "stability_regularized_distributional_tcn_graph_data",
        "round": ROUND,
        "status": "economic_gate_rejected",
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
        "candidate_forecast_gate_pass_count": 1,
        "candidate_mechanism_gate_pass_count": 1,
        "candidate_economic_gate_pass_count": 0,
        "selection_contaminated": True,
        "development_only": True,
        "trading_authority": False,
        "profitability_claim": False,
        "stability_uplift_claim": False,
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
        default=research / "round-046-stability-regularized-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-046-stability-regularized-tcn-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-path",
        type=Path,
        default=research / "round-046-failure-analysis.json",
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
        failure_path=arguments.failure_path.resolve(),
    )
    print(json.dumps(publication, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
