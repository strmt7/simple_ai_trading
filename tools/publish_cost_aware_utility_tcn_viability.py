"""Publish verified Round 47 utility-TCN evidence, data, and charts."""

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
from tools.run_cost_aware_utility_tcn_viability import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 47
PUBLICATION_SCHEMA = "cost-aware-utility-distributional-tcn-publication-v1"
CANDIDATES = ("cost_aware_utility", "cost_aware_utility_rank")
LABELS = {
    "cost_aware_utility": "Proper utility multitask",
    "cost_aware_utility_rank": "Utility + pairwise rank",
}
COLORS = {
    "cost_aware_utility": "#2563a6",
    "cost_aware_utility_rank": "#0f766e",
}
SCENARIO_DASH = {"base": "", "stress": "8 6"}
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
    design = _read_object(design_path, "Round 47 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 47 design")
    binding = _read_object(binding_path, "Round 47 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 47 binding")
    report = _read_object(evidence_root / "report.json", "Round 47 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 47 report"
    )
    candidates = report.get("candidates")
    compute = report.get("compute")
    dataset = report.get("dataset")
    claims = report.get("claims")
    outputs = report.get("outputs")
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or binding.get("schema_version") != BINDING_SCHEMA
        or report.get("schema_version") != REPORT_SCHEMA
        or any(item.get("round") != ROUND for item in (design, binding, report))
        or binding.get("design_sha256") != design_sha
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
        or report.get("status") != "complete"
        or not isinstance(dataset, Mapping)
        or dataset.get("timestamps") != 30_647
        or dataset.get("rows") != 91_941
        or dataset.get("feature_count_per_symbol") != 71
        or dataset.get("symbols") != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        or dataset.get("replay_aligned_utility_target_finite_in_all_roles") is not True
        or not isinstance(compute, Mapping)
        or compute.get("backend_kind") != "directml"
        or compute.get("model_artifacts") != 6
        or compute.get("all_artifacts_exact_reload") is not True
        or compute.get("all_temperature_fits_nonworsening") is not True
        or compute.get("preflight", {}).get("cpu_fallback_warning_count") != 0
        or not isinstance(candidates, list)
        or {item.get("candidate_id") for item in candidates} != set(CANDIDATES)
        or any(len(item.get("models", [])) != 3 for item in candidates)
        or any(
            item.get("fixed_ledger_under_stress") is not True
            or item.get("exact_replay_aligned_target_identity") is not True
            for item in candidates
        )
        or not isinstance(claims, Mapping)
        or any(
            claims.get(field) is not False
            for field in (
                "profitability_established",
                "ai_improvement_established",
                "selection_confirmation_established",
                "promotion_authorized",
                "testnet_or_live_trading_authorized",
                "leverage_authorized",
            )
        )
        or not isinstance(outputs, list)
        or not outputs
    ):
        raise ValueError("Round 47 source, model, or governance evidence drifted")
    _validate_tree(report)
    for item in outputs:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 47 evidence output drifted: {path}")
    for item in dataset["derived_cache_inputs"]:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 47 cache input drifted: {path}")
    return report, report_sha, binding_sha


def _daily_equity_rows(evidence_root: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    with gzip.open(
        evidence_root / "hourly_ledger.csv.gz", "rt", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            grouped[(row["candidate_id"], row["scenario"])].append(
                (row["decision_time_utc"][:10], float(row["portfolio_return_bps"]))
            )
    output: list[dict[str, object]] = []
    for (candidate_id, scenario), observations in sorted(grouped.items()):
        equity = 1.0
        current = ""
        daily_bps = 0.0
        hours = 0
        for date, return_bps in observations:
            if current and date != current:
                output.append(
                    {
                        "candidate_id": candidate_id,
                        "scenario": scenario,
                        "date_utc": current,
                        "equity": equity,
                        "cumulative_return_fraction": equity - 1.0,
                        "day_additive_net_bps": daily_bps,
                        "hours": hours,
                    }
                )
                daily_bps = 0.0
                hours = 0
            current = date
            equity *= 1.0 + return_bps / 10_000.0
            daily_bps += return_bps
            hours += 1
        if current:
            output.append(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "date_utc": current,
                    "equity": equity,
                    "cumulative_return_fraction": equity - 1.0,
                    "day_additive_net_bps": daily_bps,
                    "hours": hours,
                }
            )
    return output


def _training_rows(evidence_root: Path) -> list[dict[str, object]]:
    payload = _read_object(evidence_root / "training_history.json", "training history")
    candidates = payload.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("Round 47 training history has no candidates")
    rows: list[dict[str, object]] = []
    fields: list[str] = []
    for candidate_id in CANDIDATES:
        history = candidates.get(candidate_id)
        if not isinstance(history, list):
            raise ValueError(f"Round 47 training history is missing {candidate_id}")
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


def _source_rows(
    report: Mapping[str, object], evidence_root: Path
) -> list[dict[str, object]]:
    lineage = report["source_lineage"]
    return [
        {
            "artifact_class": "source_certificate",
            "canonical_sha256": lineage["source_certificate_canonical_sha256"],
            "file_sha256": lineage["source_certificate_file_sha256"],
        },
        {
            "artifact_class": "predecessor_report",
            "canonical_sha256": lineage["predecessor_report_canonical_sha256"],
            "file_sha256": lineage["predecessor_report_file_sha256"],
        },
        {
            "artifact_class": "round47_report",
            "canonical_sha256": report["report_canonical_sha256"],
            "file_sha256": _file_sha256(evidence_root / "report.json"),
        },
    ]


def _progress_rows(
    path: Path,
    report: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed = [int(row["round"]) for row in rows]
    prior_rounds = list(range(1, ROUND))
    if observed == [*prior_rounds, ROUND]:
        rows.pop()
        observed.pop()
    if not fields or observed != prior_rounds:
        raise ValueError("Round 47 prior progress history is invalid")
    best = max(
        report["candidates"],
        key=lambda item: float(item["base"]["mean_hourly_portfolio_bps"]),
    )
    utility_rows = best["action_diagnostics"]["utility_horizons"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "replay-aligned cost-aware utility TCN",
            "periods": "2022-01-01..2025-06-30 roles; eval 2024-10-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "3600;14400;43200;86400",
            "feature_set": "71 causal features; quantile mean-utility and side-probability heads",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "spearman_ic": max(float(item["spearman"]) for item in utility_rows),
            "selected_signals": best["trade_count"],
            "executable_trades": best["trade_count"],
            "mean_net_bps": best["base"]["mean_hourly_portfolio_bps"],
            "status": "rejected",
            "source_file": "verified Round 47 utility-TCN report; no profitability claim",
            "best_policy_trades": best["base"]["trades"],
            "best_policy_total_net_bps": 10_000
            * float(best["base"]["total_net_return_fraction"]),
            "best_policy_mean_net_bps": best["base"]["mean_hourly_portfolio_bps"],
            "best_policy_max_drawdown_bps": 10_000
            * float(best["base"]["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": best["base"]["profit_factor"],
            "best_model_id": best["candidate_id"] + "_descriptive_only",
            "ensemble_models": 6,
            "development_consumed": True,
            "architecture_gates_passed": report["claims"][
                "candidate_action_gate_pass_count"
            ],
            "architecture_gate_count": 2,
        }
    )
    rows.append(row)
    return fields, rows


def _bounds(values: Sequence[float], *, reference: float = 0.0) -> tuple[float, float]:
    low = min(*values, reference)
    high = max(*values, reference)
    span = max(high - low, 0.02)
    return low - 0.12 * span, high + 0.12 * span


def _two_panel_horizon_svg(
    *,
    title: str,
    description: str,
    series: Mapping[str, Sequence[Mapping[str, object]]],
    upper_key: str,
    upper_label: str,
    lower_key: str,
    lower_label: str,
    upper_reference: float = 0.0,
    lower_reference: float = 0.0,
) -> str:
    width, height = 1500, 760
    left, right, panel_height = 135, 70, 205
    chart_width = width - left - right
    panel_tops = (155, 465)
    lines = _svg_start(width, height, title, description)
    for index, (candidate_id, color) in enumerate(COLORS.items()):
        x = 900 + index * 270
        lines.append(
            f'<line x1="{x}" y1="53" x2="{x + 34}" y2="53" stroke="{color}" stroke-width="4"/><text x="{x + 44}" y="58" class="note">{html.escape(LABELS[candidate_id])}</text>'
        )
    x_positions = {
        horizon: left + index * chart_width / (len(HORIZONS) - 1)
        for index, horizon in enumerate(HORIZONS)
    }
    for panel_index, (key, label, reference) in enumerate(
        (
            (upper_key, upper_label, upper_reference),
            (lower_key, lower_label, lower_reference),
        )
    ):
        top = panel_tops[panel_index]
        values = [float(row[key]) for rows in series.values() for row in rows]
        low, high = _bounds(values, reference=reference)

        def y(value: float) -> float:
            return top + panel_height * (high - value) / (high - low)

        lines.append(f'<text x="{left}" y="{top - 20}" class="label">{label}</text>')
        for tick in (low, reference, high):
            py = y(tick)
            css = "zero" if math.isclose(tick, reference) else "grid"
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{css}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.3f}</text>'
            )
        for candidate_id, rows in series.items():
            ordered = sorted(rows, key=lambda row: int(row["horizon_hours"]))
            points = [
                (
                    x_positions[int(row["horizon_hours"])],
                    y(float(row[key])),
                )
                for row in ordered
            ]
            color = COLORS[candidate_id]
            lines.append(
                '<polyline points="'
                + " ".join(f"{x:.1f},{py:.1f}" for x, py in points)
                + f'" fill="none" stroke="{color}" stroke-width="4"/>'
            )
            for x, py in points:
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>'
                )
        for horizon, x in x_positions.items():
            lines.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 28}" text-anchor="middle" class="axis">{horizon}h</text>'
            )
    lines.append(
        '<text x="56" y="730" class="note">All values are from the frozen 2024-10 through 2025-06 consumed evaluation role; zero-reference lines are not acceptance thresholds.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _forecast_svg(report: Mapping[str, object]) -> str:
    return _two_panel_horizon_svg(
        title="Round 47 probabilistic forecast quality",
        description="Pinball skill versus the training-only unconditional baseline and pooled median rank association.",
        series={
            candidate_id: _candidate(report, candidate_id)["forecast_diagnostics"][
                "horizons"
            ]
            for candidate_id in CANDIDATES
        },
        upper_key="pinball_skill",
        upper_label="Pinball skill (fraction; higher is better)",
        lower_key="pooled_median_spearman",
        lower_label="Median forecast Spearman",
    )


def _utility_svg(report: Mapping[str, object]) -> str:
    return _two_panel_horizon_svg(
        title="Replay-aligned conditional-mean utility quality",
        description="The new head is measured against exact additive holding-period utility used by the replay ledger.",
        series={
            candidate_id: _candidate(report, candidate_id)["action_diagnostics"][
                "utility_horizons"
            ]
            for candidate_id in CANDIDATES
        },
        upper_key="mse_skill",
        upper_label="Conditional-mean MSE skill versus training mean",
        lower_key="spearman",
        lower_label="Predicted versus realized utility Spearman",
    )


def _action_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 760
    left, right, chart_width = 130, 65, 1305
    tops = (155, 465)
    panel_height = 205
    lines = _svg_start(
        width,
        height,
        "Cost-covering action probability quality",
        "Calibrated probability discrimination and log-loss skill for exact long and short profitability labels.",
    )
    categories = [(horizon, side) for horizon in HORIZONS for side in ("short", "long")]
    x_positions = {
        category: left + index * chart_width / (len(categories) - 1)
        for index, category in enumerate(categories)
    }
    all_rows = {
        candidate_id: _candidate(report, candidate_id)["action_diagnostics"][
            "action_side_horizons"
        ]
        for candidate_id in CANDIDATES
    }
    for panel_index, (key, label, reference) in enumerate(
        (("roc_auc", "ROC AUC", 0.5), ("log_loss_skill", "Log-loss skill", 0.0))
    ):
        top = tops[panel_index]
        values = [float(row[key]) for rows in all_rows.values() for row in rows]
        low, high = _bounds(values, reference=reference)

        def y(value: float) -> float:
            return top + panel_height * (high - value) / (high - low)

        lines.append(f'<text x="{left}" y="{top - 20}" class="label">{label}</text>')
        for tick in (low, reference, high):
            py = y(tick)
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if math.isclose(tick, reference) else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
            )
        for candidate_id, rows in all_rows.items():
            by_side = {side: [] for side in ("short", "long")}
            for row in rows:
                by_side[str(row["side"])].append(row)
            for side, side_rows in by_side.items():
                ordered = sorted(side_rows, key=lambda row: int(row["horizon_hours"]))
                points = [
                    (
                        x_positions[(int(row["horizon_hours"]), side)],
                        y(float(row[key])),
                    )
                    for row in ordered
                ]
                dash = ' stroke-dasharray="7 5"' if side == "short" else ""
                color = COLORS[candidate_id]
                lines.append(
                    '<polyline points="'
                    + " ".join(f"{x:.1f},{py:.1f}" for x, py in points)
                    + f'" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
                )
                for x, py in points:
                    shape = (
                        f'<rect x="{x - 5:.1f}" y="{py - 5:.1f}" width="10" height="10" fill="{color}"/>'
                        if side == "short"
                        else f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>'
                    )
                    lines.append(shape)
        for (horizon, side), x in x_positions.items():
            lines.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 28}" text-anchor="middle" class="axis">{horizon}h {side[0].upper()}</text>'
            )
    lines.extend(
        [
            f'<line x1="900" y1="52" x2="934" y2="52" stroke="{COLORS[CANDIDATES[0]]}" stroke-width="4"/><text x="944" y="57" class="note">{LABELS[CANDIDATES[0]]}</text>',
            f'<line x1="1170" y1="52" x2="1204" y2="52" stroke="{COLORS[CANDIDATES[1]]}" stroke-width="4"/><text x="1214" y="57" class="note">{LABELS[CANDIDATES[1]]}</text>',
            '<rect x="955" y="73" width="11" height="11" fill="#52606d"/><text x="974" y="84" class="note">short</text>',
            '<circle cx="1045" cy="79" r="6" fill="#52606d"/><text x="1060" y="84" class="note">long</text>',
            '<text x="56" y="730" class="note">AUC above 0.5 and log-loss skill above zero are necessary action-quality checks, not evidence of positive after-cost expectancy.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _stability_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 620
    left, right, top, chart_height = 150, 80, 145, 330
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Independent-seed output stability",
        "Minimum pairwise Spearman across all peer pairs; every output family must exceed the frozen 0.50 floor.",
    )
    metrics = ("Quantile median", "Mean utility", "Action logits")
    values: dict[str, tuple[float, float, float]] = {}
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        values[candidate_id] = (
            float(
                candidate["forecast_diagnostics"]["gate"][
                    "minimum_pairwise_seed_median_prediction_spearman"
                ]
            ),
            float(
                candidate["action_diagnostics"]["gate"][
                    "minimum_pairwise_seed_conditional_mean_spearman"
                ]
            ),
            float(
                candidate["action_diagnostics"]["gate"][
                    "minimum_pairwise_seed_action_logit_spearman"
                ]
            ),
        )
    low = min(0.0, *(value for row in values.values() for value in row))
    high = max(1.0, *(value for row in values.values() for value in row))

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    for tick in (low, 0.5, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if tick == 0.5 else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    group_width = chart_width / len(metrics)
    bar_width = 115
    for metric_index, metric in enumerate(metrics):
        center = left + group_width * (metric_index + 0.5)
        for candidate_index, candidate_id in enumerate(CANDIDATES):
            value = values[candidate_id][metric_index]
            x = center + (candidate_index - 0.5) * 145 - bar_width / 2
            py = y(value)
            lines.append(
                f'<rect x="{x:.1f}" y="{py:.1f}" width="{bar_width}" height="{top + chart_height - py:.1f}" fill="{COLORS[candidate_id]}"/><text x="{x + bar_width / 2:.1f}" y="{py - 9:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 34}" text-anchor="middle" class="label">{metric}</text>'
        )
    for index, candidate_id in enumerate(CANDIDATES):
        x = 900 + index * 270
        lines.append(
            f'<rect x="{x}" y="46" width="14" height="14" fill="{COLORS[candidate_id]}"/><text x="{x + 24}" y="58" class="note">{html.escape(LABELS[candidate_id])}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _training_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 740
    left, right, chart_width = 130, 70, 1300
    tops = (150, 445)
    panel_height = 190
    lines = _svg_start(
        width,
        height,
        "Round 47 training dynamics",
        "Chronological early-stop composite and calibrated-action training loss; epochs after each minimum remain diagnostic only.",
    )
    by_candidate = {
        candidate_id: [row for row in rows if row["candidate_id"] == candidate_id]
        for candidate_id in CANDIDATES
    }
    for panel_index, (key, label) in enumerate(
        (
            ("early_stop_composite", "Early-stop composite"),
            ("training_action_bce", "Training side-profitability BCE"),
        )
    ):
        top = tops[panel_index]
        numeric = [
            float(row[key]) for values in by_candidate.values() for row in values
        ]
        low, high = _bounds(numeric, reference=min(numeric))
        max_epoch = max(
            int(row["epoch"]) for values in by_candidate.values() for row in values
        )

        def x(epoch: int) -> float:
            return left + chart_width * (epoch - 1) / max(max_epoch - 1, 1)

        def y(value: float) -> float:
            return top + panel_height * (high - value) / (high - low)

        lines.append(f'<text x="{left}" y="{top - 18}" class="label">{label}</text>')
        for tick in (low, (low + high) / 2.0, high):
            py = y(tick)
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.4f}</text>'
            )
        for candidate_id, values in by_candidate.items():
            ordered = sorted(values, key=lambda row: int(row["epoch"]))
            points = [(x(int(row["epoch"])), y(float(row[key]))) for row in ordered]
            color = COLORS[candidate_id]
            lines.append(
                '<polyline points="'
                + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + f'" fill="none" stroke="{color}" stroke-width="4"/>'
            )
            best_epoch = int(ordered[-1]["best_epoch"])
            best_row = next(row for row in ordered if int(row["epoch"]) == best_epoch)
            lines.append(
                f'<circle cx="{x(best_epoch):.1f}" cy="{y(float(best_row[key])):.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
            )
        for epoch in sorted({1, max_epoch // 2 or 1, max_epoch}):
            lines.append(
                f'<text x="{x(epoch):.1f}" y="{top + panel_height + 26}" text-anchor="middle" class="axis">epoch {epoch}</text>'
            )
    for index, candidate_id in enumerate(CANDIDATES):
        x = 900 + index * 270
        lines.append(
            f'<line x1="{x}" y1="52" x2="{x + 34}" y2="52" stroke="{COLORS[candidate_id]}" stroke-width="4"/><text x="{x + 44}" y="57" class="note">{html.escape(LABELS[candidate_id])}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _economics_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 700
    left, right, top, chart_height = 135, 75, 150, 360
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Fixed-ledger after-cost economics",
        "Base and stress point estimates are shown with drawdown and familywise block-bootstrap uncertainty; all data are consumed development evidence.",
    )
    groups: list[tuple[str, str, Mapping[str, object]]] = []
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        for scenario in ("base", "stress"):
            groups.append((candidate_id, scenario, candidate[scenario]))
    lower = min(
        -10.0,
        *(
            100.0 * float(metrics["total_net_return_fraction"])
            for _, _, metrics in groups
        ),
        *(
            -100.0 * float(metrics["maximum_drawdown_fraction"])
            for _, _, metrics in groups
        ),
    )
    upper = max(
        10.0,
        *(
            100.0 * float(metrics["total_net_return_fraction"])
            for _, _, metrics in groups
        ),
    )

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    for tick in (lower, 0.0, upper):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    group_width = chart_width / len(groups)
    for index, (candidate_id, scenario, metrics) in enumerate(groups):
        center = left + group_width * (index + 0.5)
        total = 100.0 * float(metrics["total_net_return_fraction"])
        drawdown = -100.0 * float(metrics["maximum_drawdown_fraction"])
        color = COLORS[candidate_id]
        opacity = "1" if scenario == "base" else "0.55"
        for offset, value in ((-38, total), (38, drawdown)):
            zero_y = y(0.0)
            value_y = y(value)
            lines.append(
                f'<rect x="{center + offset - 28:.1f}" y="{min(zero_y, value_y):.1f}" width="56" height="{abs(zero_y - value_y):.1f}" fill="{color}" opacity="{opacity}"/>'
            )
        bootstrap = metrics["bootstrap_mean_hourly_portfolio_bps"]
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 32}" text-anchor="middle" class="label">{html.escape(LABELS[candidate_id].replace(" utility", ""))}</text><text x="{center:.1f}" y="{top + chart_height + 52}" text-anchor="middle" class="axis">{scenario}; n={metrics["trades"]}</text>'
        )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 78}" text-anchor="middle" class="note">PF {float(metrics["profit_factor"] or 0):.3f}; bootstrap L {float(bootstrap["lower_bps"]):+.2f}</text>'
        )
    lines.extend(
        [
            '<rect x="930" y="45" width="15" height="15" fill="#52606d"/><text x="955" y="58" class="note">total return (up)</text>',
            '<rect x="1120" y="45" width="15" height="15" fill="#52606d" opacity="0.65"/><text x="1145" y="58" class="note">maximum drawdown (down)</text>',
            '<text x="56" y="672" class="note">Bootstrap L is the one-sided familywise lower bound in mean hourly portfolio bps. Positive cumulative return alone cannot pass the gate.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _monthly_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["scenario"] == "base"]
    months = sorted({str(row["month"]) for row in selected})
    width, height = 1500, 650
    left, right, top, chart_height = 120, 60, 145, 330
    chart_width = width - left - right
    values = [100.0 * float(row["total_net_return_fraction"]) for row in selected]
    low, high = _bounds(values)
    lines = _svg_start(
        width,
        height,
        "Base-cost monthly portfolio returns",
        "Dated fixed-ledger returns expose regime dependence hidden by aggregate cumulative performance.",
    )

    def x(index: int) -> float:
        return left + chart_width * index / max(len(months) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    for tick in (low, 0.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    for candidate_id in CANDIDATES:
        candidate_rows = {
            row["month"]: row for row in selected if row["candidate_id"] == candidate_id
        }
        points = [
            (
                x(index),
                y(100.0 * float(candidate_rows[month]["total_net_return_fraction"])),
            )
            for index, month in enumerate(months)
        ]
        lines.append(
            '<polyline points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + f'" fill="none" stroke="{COLORS[candidate_id]}" stroke-width="4"/>'
        )
        lines.extend(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{COLORS[candidate_id]}"/>'
            for px, py in points
        )
    for index, month in enumerate(months):
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + chart_height + 30}" text-anchor="middle" class="axis">{month}</text>'
        )
    lines.extend(
        [
            f'<line x1="900" y1="52" x2="934" y2="52" stroke="{COLORS[CANDIDATES[0]]}" stroke-width="4"/><text x="944" y="57" class="note">{LABELS[CANDIDATES[0]]}</text>',
            f'<line x1="1170" y1="52" x2="1204" y2="52" stroke="{COLORS[CANDIDATES[1]]}" stroke-width="4"/><text x="1214" y="57" class="note">{LABELS[CANDIDATES[1]]}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _equity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 680
    left, right, top, chart_height = 115, 70, 145, 390
    chart_width = width - left - right
    dates = sorted({str(row["date_utc"]) for row in rows})
    values = [float(row["equity"]) for row in rows]
    low, high = _bounds(values, reference=1.0)
    lines = _svg_start(
        width,
        height,
        "Dated fixed-ledger equity",
        "Daily close of hourly compounded one-third sleeves; dashed lines reprice the identical base ledger at stress costs.",
    )

    def x(date: str) -> float:
        return left + chart_width * dates.index(date) / max(len(dates) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    for tick in (low, 1.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if tick == 1 else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    for candidate_id in CANDIDATES:
        for scenario in ("base", "stress"):
            selected = [
                row
                for row in rows
                if row["candidate_id"] == candidate_id and row["scenario"] == scenario
            ]
            points = [
                (x(str(row["date_utc"])), y(float(row["equity"]))) for row in selected
            ]
            dash = SCENARIO_DASH[scenario]
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                '<polyline points="'
                + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + f'" fill="none" stroke="{COLORS[candidate_id]}" stroke-width="3"{dash_attr}/>'
            )
    for date in (dates[0], dates[len(dates) // 2], dates[-1]):
        lines.append(
            f'<text x="{x(date):.1f}" y="{top + chart_height + 32}" text-anchor="middle" class="axis">{date}</text>'
        )
    lines.extend(
        [
            f'<line x1="900" y1="52" x2="934" y2="52" stroke="{COLORS[CANDIDATES[0]]}" stroke-width="4"/><text x="944" y="57" class="note">{LABELS[CANDIDATES[0]]}</text>',
            f'<line x1="1170" y1="52" x2="1204" y2="52" stroke="{COLORS[CANDIDATES[1]]}" stroke-width="4"/><text x="1214" y="57" class="note">{LABELS[CANDIDATES[1]]}</text>',
            '<line x1="955" y1="79" x2="989" y2="79" stroke="#52606d" stroke-width="3"/><text x="999" y="84" class="note">base cost</text>',
            '<line x1="1095" y1="79" x2="1129" y2="79" stroke="#52606d" stroke-width="3" stroke-dasharray="7 5"/><text x="1139" y="84" class="note">stress cost</text>',
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
    candidate_results = {
        candidate["candidate_id"]: {
            "forecast_gate": candidate["forecast_diagnostics"]["gate"],
            "action_gate": candidate["action_diagnostics"]["gate"],
            "combined_action_gate_passed": candidate["combined_action_gate_passed"],
            "rank_ablation_gate": candidate["rank_ablation_gate"],
            "utility_horizons": candidate["action_diagnostics"]["utility_horizons"],
            "action_side_horizons": candidate["action_diagnostics"][
                "action_side_horizons"
            ],
            "base": candidate["base"],
            "stress": candidate["stress"],
            "economic_gate": candidate["economic_gate"],
        }
        for candidate in report["candidates"]
    }
    economic_passes = int(report["claims"]["candidate_economic_gate_pass_count"])
    interpretations: list[str] = [
        "Round 46's median-based action score had almost no rank relationship with realized net utility. Round 47 replaced that mapping with an affine conditional-mean target that exactly matches the additive replay ledger before costs.",
        "Long and short profitability heads were trained against the unchanged 12 bps round-trip hurdle and calibrated with one scalar temperature per seed on the earlier calibration role. No evaluation outcome selected a threshold or temperature.",
        "The pairwise candidate changed one frozen term only. Its mechanism result is reported independently of policy economics because a favorable cumulative return cannot retroactively validate the ranking objective.",
    ]
    for candidate_id in CANDIDATES:
        result = candidate_results[candidate_id]
        utility = result["utility_horizons"]
        action = result["action_side_horizons"]
        interpretations.append(
            f"{LABELS[candidate_id]} produced {result['base']['trades']} trades; "
            f"base return {100 * float(result['base']['total_net_return_fraction']):+.3f}%, "
            f"stress return {100 * float(result['stress']['total_net_return_fraction']):+.3f}%, "
            f"base drawdown {100 * float(result['base']['maximum_drawdown_fraction']):.3f}%, "
            f"best utility Spearman {max(float(row['spearman']) for row in utility):+.4f}, "
            f"and best action AUC {max(float(row['roc_auc']) for row in action):.4f}. "
            f"Forecast/action/economic gates were {result['forecast_gate']['passed']}/"
            f"{result['combined_action_gate_passed']}/{result['economic_gate']['passed']}."
        )
    interpretations.extend(
        [
            "Every point estimate remains selection-contaminated development evidence. A positive aggregate result, if present, is not a profitability claim and cannot authorize testnet, live day trading, or leverage.",
            "The Qwen3 8B evidence remains a schema-constrained risk-reasoning benchmark. No language model received market outcomes or numerical order authority, so AI trading uplift remains untested.",
        ]
    )
    analysis: dict[str, object] = {
        "schema_version": "round-047-failure-analysis-v1",
        "round": ROUND,
        "status": (
            "development_gate_passed_without_promotion"
            if economic_passes
            else "quality_or_economic_gate_rejected"
        ),
        "evidence": {
            "report_canonical_sha256": report_sha,
            "report_file_sha256": _file_sha256(evidence_root / "report.json"),
            "design_sha256": report["design_sha256"],
            "binding_sha256": binding_sha,
            "implementation_commit": report["implementation_commit"],
            "dataset_sha256": report["dataset"]["dataset_sha256"],
            "directml_model_artifacts": report["compute"]["model_artifacts"],
            "exact_reload_verified": report["compute"]["all_artifacts_exact_reload"],
            "temperature_nonworsening_verified": report["compute"][
                "all_temperature_fits_nonworsening"
            ],
            "cpu_fallback_warnings": report["compute"]["preflight"][
                "cpu_fallback_warning_count"
            ],
            "declared_output_count": len(report["outputs"]),
        },
        "rank_ablation_gate": report["rank_ablation_gate"],
        "candidate_results": candidate_results,
        "critical_interpretation": interpretations,
        "decisions": [
            "Do not promote, apply leverage, or authorize testnet/live execution from this consumed-development round.",
            "Retain a candidate mechanism only if its own forecast and action-quality gates pass; do not select it from cumulative return alone.",
            "Do not lower the probability floor, costs, drawdown ceiling, bootstrap bound, or diversification gate after seeing the evaluation ledger.",
            "Keep language-model authority limited to fail-closed risk review until a deterministic candidate passes all numerical gates on untouched confirmation data.",
        ],
        "next_model_requirements": [
            "If conditional-mean skill remains weak, test a pre-registered robust mean-distribution model or state-conditioned mixture while preserving exact replay-aligned targets.",
            "If probability skill is positive but economics fail, diagnose calibration drift, side/horizon concentration, and volatility-conditioned utility without post-hoc evaluation thresholds.",
            "Acquire an untouched chronological confirmation period only after the mechanism is fixed; preserve matched funding, volume, price, and execution-cost lineage.",
            "Run any local finance-language-model ablation as a paired veto-only test with identical deterministic candidate actions and no future outcomes in prompts.",
        ],
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "trading_authority": False,
        "leverage_applied": False,
        "generated_at_utc": report["generated_at_utc"],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    rows = []
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        utility = candidate["action_diagnostics"]["utility_horizons"]
        rows.append(
            "| "
            + LABELS[candidate_id]
            + f" | {max(float(row['spearman']) for row in utility):+.4f}"
            + f" | {candidate['base']['trades']}"
            + f" | {100 * float(candidate['base']['total_net_return_fraction']):+.2f}%"
            + f" | {100 * float(candidate['stress']['total_net_return_fraction']):+.2f}%"
            + f" | {100 * float(candidate['base']['maximum_drawdown_fraction']):.2f}%"
            + f" | {float(candidate['base']['profit_factor'] or 0):.3f}"
            + f" | {candidate['forecast_diagnostics']['gate']['passed']}/"
            + f"{candidate['combined_action_gate_passed']}/"
            + f"{candidate['economic_gate']['passed']} |"
        )
    runtime = report["runtime"]
    return f"""# Round 47: Replay-Aligned Utility TCN

> **Beta research warning:** no model is approved for testnet, live day trading, leverage, or autonomous execution. All results use a consumed development period.

Round 47 fixes a model-policy mismatch: a conditional median is not expected P&L. The stable causal TCN now learns exact additive holding-period mean utility plus calibrated long/short probabilities, while retaining monotone return quantiles. The second candidate adds one bounded pairwise ranking term.

![Forecast quality](charts/forecast-quality.svg)

![Utility quality](charts/utility-quality.svg)

![Action probability quality](charts/action-quality.svg)

![Seed stability](charts/seed-stability.svg)

| Candidate | Best utility Spearman | Trades | Base return | Stress return | Base drawdown | Profit factor | Forecast/action/economic gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

Point estimates are not validated profitability. The fixed ledger charges 6 bps per side at base and 8 bps per side under stress, uses one-third sleeves, and forbids overlapping positions within a symbol.

![Training dynamics](charts/training-dynamics.svg)

![Policy economics](charts/policy-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Research progress](charts/research-progress.svg)

DirectML trained six AMD-GPU artifacts in `{float(runtime["elapsed_seconds"]):.1f}s`; all three output heads reloaded exactly and the warning-fatal preflight recorded zero CPU fallbacks. The local 8B language model remains a risk-review component only; AI trading uplift is not established.

Data: [forecast horizons](horizons.csv) | [utility horizons](utility-horizons.csv) | [action horizons](action-horizons.csv) | [seed stability](seed-stability.csv) | [training](training.csv) | [models](models.csv) | [roles](roles.csv) | [label prevalence](labels.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-047-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
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
    forecast_rows = _csv_rows(evidence_root / "forecast_diagnostics.csv")
    quantile_stability = _csv_rows(evidence_root / "quantile_seed_stability.csv")
    action_stability = _csv_rows(evidence_root / "action_seed_stability.csv")
    combined_stability = [
        {
            "candidate_id": row["candidate_id"],
            "left_seed": row["left_seed"],
            "right_seed": row["right_seed"],
            "horizon_hours": row["horizon_hours"],
            "output": "quantile_median",
            "side": "signed",
            "spearman": row["median_prediction_spearman"],
        }
        for row in quantile_stability
    ] + action_stability
    utility_rows = _csv_rows(evidence_root / "utility_horizon_summary.csv")
    action_rows = _csv_rows(evidence_root / "action_side_horizon_summary.csv")
    labels = _csv_rows(evidence_root / "role_label_prevalence.csv")
    training = _training_rows(evidence_root)
    models = _csv_rows(evidence_root / "models.csv")
    roles = _csv_rows(evidence_root / "roles.csv")
    trades = _csv_rows(evidence_root / "trades.csv")
    replays = _csv_rows(evidence_root / "replays.csv")
    monthly = _csv_rows(evidence_root / "monthly_economics.csv")
    symbols = _csv_rows(evidence_root / "symbol_economics.csv")
    daily = _daily_equity_rows(evidence_root)
    sources = _source_rows(report, evidence_root)
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
        output_dir / "forecast-diagnostics.csv",
        output_dir / "utility-horizons.csv",
        output_dir / "action-horizons.csv",
        output_dir / "seed-stability.csv",
        output_dir / "labels.csv",
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
        charts / "utility-quality.svg",
        charts / "action-quality.svg",
        charts / "seed-stability.svg",
        charts / "training-dynamics.svg",
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
    _write_csv(output_dir / "forecast-diagnostics.csv", forecast_rows)
    _write_csv(output_dir / "utility-horizons.csv", utility_rows)
    _write_csv(output_dir / "action-horizons.csv", action_rows)
    _write_csv(output_dir / "seed-stability.csv", combined_stability)
    _write_csv(output_dir / "labels.csv", labels)
    _write_csv(output_dir / "training.csv", training)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "roles.csv", roles)
    _write_csv(output_dir / "trades.csv", trades)
    _write_csv(output_dir / "replays.csv", replays)
    _write_csv(output_dir / "monthly.csv", monthly)
    _write_csv(output_dir / "symbols.csv", symbols)
    _write_csv(output_dir / "daily-equity.csv", daily)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress_rows, fields=progress_fields)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(report))
    _write_text(charts / "utility-quality.svg", _utility_svg(report))
    _write_text(charts / "action-quality.svg", _action_svg(report))
    _write_text(charts / "seed-stability.svg", _stability_svg(report))
    _write_text(charts / "training-dynamics.svg", _training_svg(training))
    _write_text(charts / "policy-economics.svg", _economics_svg(report))
    _write_text(charts / "monthly-economics.svg", _monthly_svg(monthly))
    _write_text(charts / "daily-equity.svg", _equity_svg(daily))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress_rows))
    failure = _failure_analysis(report, source_report_sha, binding_sha, evidence_root)
    _write_text(
        failure_path,
        json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    economic_passes = int(report["claims"]["candidate_economic_gate_pass_count"])
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "cost_aware_utility_tcn_graph_data",
        "round": ROUND,
        "status": (
            "development_gate_passed_without_promotion"
            if economic_passes
            else "quality_or_economic_gate_rejected"
        ),
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
        "candidate_forecast_gate_pass_count": report["claims"][
            "candidate_forecast_gate_pass_count"
        ],
        "candidate_action_gate_pass_count": report["claims"][
            "candidate_action_gate_pass_count"
        ],
        "candidate_economic_gate_pass_count": economic_passes,
        "rank_ablation_passed": report["claims"]["rank_ablation_passed"],
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
        default=research / "round-047-cost-aware-utility-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-047-cost-aware-utility-tcn-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-path",
        type=Path,
        default=research / "round-047-failure-analysis.json",
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
