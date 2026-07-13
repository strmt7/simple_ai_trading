"""Publish verified Round 48 minute logistic-mixture evidence and charts."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import html
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


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
    _file_sha256,
    _read_object,
)
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_minute_logistic_mixture_tcn_viability import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 48
PUBLICATION_SCHEMA = "minute-logistic-mixture-tcn-publication-v1"
CANDIDATES = ("single_logistic_tcn", "state_mixture_logistic_tcn")
LABELS = {
    "single_logistic_tcn": "Single logistic control",
    "state_mixture_logistic_tcn": "State-conditioned mixture",
}
COLORS = {
    "single_logistic_tcn": "#2563a6",
    "state_mixture_logistic_tcn": "#0f766e",
}
HORIZONS = (15, 30, 60, 120)


def _candidate(
    report: Mapping[str, object], candidate_id: str
) -> Mapping[str, object]:
    return next(
        item
        for item in report["candidate_results"]
        if item["candidate_id"] == candidate_id
    )


def _validated_source(
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design = _read_object(design_path, "Round 48 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 48 design")
    binding = _read_object(binding_path, "Round 48 binding")
    binding_sha = _canonical_identity(
        binding, "binding_sha256", "Round 48 binding"
    )
    report = _read_object(evidence_root / "report.json", "Round 48 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 48 report"
    )
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("status") != "frozen"
        or binding.get("schema_version") != BINDING_SCHEMA
        or report.get("schema_version") != REPORT_SCHEMA
        or any(item.get("round") != ROUND for item in (design, binding, report))
        or binding.get("design_sha256") != design_sha
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
        or report.get("status") != "quality_or_economic_gate_rejected"
    ):
        raise ValueError("Round 48 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    backend = report.get("backend")
    results = report.get("candidate_results")
    ai = report.get("ai_decision")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("dataset_sha256")
        != "6969a3134049a326024939d5f9c46a99c37a4932e4a1f146a542a77427bba92b"
        or dataset.get("timestamps") != 366_035
        or dataset.get("rows") != 1_098_105
        or dataset.get("feature_count") != 71
        or dataset.get("symbols") != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        or dataset.get("persistent_feature_copy_created") is not False
        or not isinstance(backend, Mapping)
        or backend.get("backend_kind") != "directml"
        or backend.get("cpu_fallback_warnings") != 0
        or backend.get("warning_count") != 0
        or not isinstance(results, list)
        or {item.get("candidate_id") for item in results} != set(CANDIDATES)
        or any(len(item.get("artifacts", [])) != 3 for item in results)
        or any(item.get("combined_quality_gate_passed") is not False for item in results)
        or any(item.get("economic_gate", {}).get("passed") is not False for item in results)
        or report.get("mixture_ablation_gate", {}).get("passed") is not True
        or not isinstance(ai, Mapping)
        or ai.get("executed") is not False
        or ai.get("ai_uplift_claim") is not False
        or report.get("selection_confirmation_accessed") is not False
        or report.get("terminal_2026_accessed") is not False
        or report.get("profitability_claim") is not False
        or report.get("ai_uplift_claim") is not False
        or report.get("trading_authority") is not False
        or report.get("leverage_applied") is not False
    ):
        raise ValueError("Round 48 model, data, or governance evidence drifted")
    for candidate in results:
        for artifact in candidate["artifacts"]:
            if (
                artifact.get("backend_kind") != "directml"
                or artifact.get("warning_count") != 0
                or any(
                    float(artifact[field]) != 0.0
                    for field in (
                        "reload_max_abs_weight_error",
                        "reload_max_abs_location_error",
                        "reload_max_abs_scale_error",
                    )
                )
            ):
                raise ValueError("Round 48 model reload evidence drifted")
    declared = report.get("external_artifacts")
    if not isinstance(declared, list) or len(declared) != 12:
        raise ValueError("Round 48 external artifact declaration is incomplete")
    for item in declared:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 48 external artifact drifted: {path}")
    return report, design, report_sha, binding_sha


def _timestamp_iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1_000.0, tz=UTC).isoformat()


def _csv_ready(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    fields: list[str] = []
    output: list[dict[str, object]] = []
    for source in rows:
        row: dict[str, object] = {}
        for key, value in source.items():
            if key not in fields:
                fields.append(key)
            row[key] = (
                json.dumps(
                    value,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
                if isinstance(value, (dict, list))
                else value
            )
        output.append(row)
    return fields, [{field: row.get(field, "") for field in fields} for row in output]


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields, ready = _csv_ready(rows)
    _write_csv(path, ready, fields=fields)


def _table_rows(
    report: Mapping[str, object], design: Mapping[str, object]
) -> dict[str, list[dict[str, object]]]:
    horizons: list[dict[str, object]] = []
    symbol_horizons: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    monthly_forecasts: list[dict[str, object]] = []
    pit: list[dict[str, object]] = []
    routing: list[dict[str, object]] = []
    prediction_summary: list[dict[str, object]] = []
    training: list[dict[str, object]] = []
    models: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    replays: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    symbols: list[dict[str, object]] = []
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        diagnostics = candidate["diagnostics"]
        horizons.extend(dict(row) for row in diagnostics["horizons"])
        symbol_horizons.extend(dict(row) for row in diagnostics["symbols"])
        actions.extend(dict(row) for row in diagnostics["actions"])
        stability.extend(dict(row) for row in diagnostics["seed_stability"])
        monthly_forecasts.extend(dict(row) for row in diagnostics["monthly"])
        pit.extend(dict(row) for row in diagnostics["pit_histogram"])
        routing.extend(dict(row) for row in diagnostics["routing"])
        prediction_summary.append(
            {
                "candidate_id": candidate_id,
                **diagnostics["prediction_summary"],
            }
        )
        training.extend(dict(row) for row in candidate["training_history"])
        models.extend(dict(row) for row in candidate["artifacts"])
        for trade in candidate["base"]["trades"]:
            trades.append(
                {
                    **trade,
                    "decision_time_utc": _timestamp_iso(trade["decision_time_ms"]),
                    "exit_time_utc": _timestamp_iso(trade["exit_time_ms"]),
                }
            )
        for scenario in ("base", "stress"):
            replay = candidate[scenario]
            replays.extend(
                {
                    **row,
                    "decision_time_utc": _timestamp_iso(row["decision_time_ms"]),
                    "exit_time_utc": _timestamp_iso(row["exit_time_ms"]),
                    "booked_time_utc": _timestamp_iso(row["booked_time_ms"]),
                }
                for row in replay["trade_outcomes"]
            )
            monthly.extend(dict(row) for row in replay["monthly"])
            daily.extend(dict(row) for row in replay["daily_equity"])
            metrics = replay["metrics"]
            symbols.extend(
                {
                    "candidate_id": candidate_id,
                    "scenario": scenario,
                    "symbol": symbol,
                    "trades": metrics["trades_by_symbol"][symbol],
                    "net_bps": metrics["symbol_net_bps"][symbol],
                }
                for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
            )
    roles = [
        {
            "role": role,
            "start_utc": contract["start"],
            "end_inclusive_utc": contract["end_inclusive"],
            "timestamps": report["dataset"]["role_timestamp_counts"][
                "viability" if role == "evaluation" else role
            ],
            "uses": contract["uses"],
            "selection_contaminated": role == "evaluation",
        }
        for role, contract in design["chronological_roles"].items()
        if isinstance(contract, Mapping)
        and "start" in contract
        and "end_inclusive" in contract
    ]
    source = report["dataset"]["source_evidence"]
    price = source["price_flow"]
    derivatives = {row["symbol"]: row for row in source["derivatives_series"]}
    archives = {row["symbol"]: row for row in price["archive_evidence"]}
    sources = [
        {
            **row,
            **{
                f"derivatives_{key}": value
                for key, value in derivatives[row["symbol"]].items()
                if key != "symbol"
            },
            **{
                f"archive_{key}": value
                for key, value in archives[row["symbol"]].items()
                if key != "symbol"
            },
            "price_panel_sha256": price["panel_stream_sha256"],
            "derivatives_panel_sha256": source["derivatives_panel_sha256"],
            "source_certificate_sha256": source["source_certificate_sha256"],
        }
        for row in price["series_evidence"]
    ]
    return {
        "horizons": horizons,
        "symbol-horizons": symbol_horizons,
        "action-horizons": actions,
        "seed-stability": stability,
        "monthly-forecast": monthly_forecasts,
        "pit-histogram": pit,
        "routing": routing,
        "prediction-summary": prediction_summary,
        "training": training,
        "models": models,
        "roles": roles,
        "trades": trades,
        "replays": replays,
        "monthly": monthly,
        "symbols": symbols,
        "daily-equity": daily,
        "sources": sources,
    }


def _progress_rows(
    path: Path, report: Mapping[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    rows = [row for row in rows if int(row["round"]) != ROUND]
    if not fields or [int(row["round"]) for row in rows] != list(range(1, ROUND)):
        raise ValueError("Round 48 prior progress history is invalid")
    best = max(
        report["candidate_results"],
        key=lambda item: float(item["base"]["metrics"]["total_net_return_fraction"]),
    )
    horizons = best["diagnostics"]["horizons"]
    actions = best["diagnostics"]["actions"]
    metrics = best["base"]["metrics"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "minute state-conditioned logistic-mixture TCN",
            "periods": "2022-01-01..2025-06-30 roles; eval 2025-01-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "900;1800;3600;7200",
            "feature_set": "71 causal features; 5-minute large-kernel TCN",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "direction_auc": max(float(item["roc_auc"]) for item in actions),
            "spearman_ic": max(
                float(item["distribution_mean_spearman"]) for item in horizons
            ),
            "selected_signals": metrics["trades"],
            "executable_trades": metrics["trades"],
            "mean_net_bps": metrics["mean_five_minute_portfolio_bps"],
            "status": "rejected",
            "source_file": "verified Round 48 logistic-mixture report; no profitability claim",
            "best_policy_trades": metrics["trades"],
            "best_policy_total_net_bps": 10_000
            * float(metrics["total_net_return_fraction"]),
            "best_policy_mean_net_bps": metrics["mean_five_minute_portfolio_bps"],
            "best_policy_max_drawdown_bps": 10_000
            * float(metrics["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": metrics["profit_factor"],
            "best_model_id": best["candidate_id"] + "_descriptive_only",
            "ensemble_models": 6,
            "development_consumed": True,
            "architecture_gates_passed": 0,
            "architecture_gate_count": 2,
        }
    )
    rows.append(row)
    return fields, rows


def _bounds(
    values: Sequence[float], *, reference: float = 0.0, padding: float = 0.12
) -> tuple[float, float]:
    low = min(*values, reference)
    high = max(*values, reference)
    span = max(high - low, 0.02)
    return low - padding * span, high + padding * span


def _append_candidate_legend(
    lines: list[str], *, x_start: int = 900, y: int = 53
) -> None:
    for index, candidate_id in enumerate(CANDIDATES):
        x = x_start + index * 270
        lines.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 34}" y2="{y}" stroke="{COLORS[candidate_id]}" stroke-width="4"/><text x="{x + 44}" y="{y + 5}" class="note">{html.escape(LABELS[candidate_id])}</text>'
        )


def _two_panel_horizon_svg(
    *,
    title: str,
    description: str,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
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
    _append_candidate_legend(lines)
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
        values = [float(row[key]) for candidate in rows.values() for row in candidate]
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
        for candidate_id, candidate_rows in rows.items():
            ordered = sorted(candidate_rows, key=lambda row: int(row["horizon_minutes"]))
            points = [
                (x_positions[int(row["horizon_minutes"])], y(float(row[key])))
                for row in ordered
            ]
            color = COLORS[candidate_id]
            lines.append(
                '<polyline points="'
                + " ".join(f"{x:.1f},{py:.1f}" for x, py in points)
                + f'" fill="none" stroke="{color}" stroke-width="4"/>'
            )
            for x, py in points:
                lines.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
        for horizon, x in x_positions.items():
            lines.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 28}" text-anchor="middle" class="axis">{horizon}m</text>'
            )
    lines.append(
        '<text x="56" y="730" class="note">All values are regenerated from the hash-bound 2025-H1 consumed evaluation report.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _forecast_svg(report: Mapping[str, object]) -> str:
    return _two_panel_horizon_svg(
        title="Round 48 probabilistic forecast quality",
        description="Likelihood skill was positive, but conditional-mean rank association remained weak.",
        rows={
            candidate_id: _candidate(report, candidate_id)["diagnostics"]["horizons"]
            for candidate_id in CANDIDATES
        },
        upper_key="negative_log_likelihood_skill",
        upper_label="Negative-log-likelihood skill (higher is better)",
        lower_key="distribution_mean_spearman",
        lower_label="Conditional distribution-mean Spearman",
    )


def _action_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 760
    left, right, panel_height = 135, 70, 205
    chart_width = width - left - right
    panel_tops = (155, 465)
    lines = _svg_start(
        width,
        height,
        "Cost-covering action probability quality",
        "Long uses solid markers; short uses dashed lines and square markers.",
    )
    _append_candidate_legend(lines)
    x_positions = {
        horizon: left + index * chart_width / (len(HORIZONS) - 1)
        for index, horizon in enumerate(HORIZONS)
    }
    all_rows = {
        candidate_id: _candidate(report, candidate_id)["diagnostics"]["actions"]
        for candidate_id in CANDIDATES
    }
    for panel_index, (key, label, reference) in enumerate(
        (("roc_auc", "ROC AUC", 0.5), ("log_loss_skill", "Log-loss skill", 0.0))
    ):
        top = panel_tops[panel_index]
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
            for side in ("long", "short"):
                selected = sorted(
                    (row for row in rows if row["side"] == side),
                    key=lambda row: int(row["horizon_minutes"]),
                )
                points = [
                    (x_positions[int(row["horizon_minutes"])], y(float(row[key])))
                    for row in selected
                ]
                dash = ' stroke-dasharray="7 5"' if side == "short" else ""
                color = COLORS[candidate_id]
                lines.append(
                    '<polyline points="'
                    + " ".join(f"{x:.1f},{py:.1f}" for x, py in points)
                    + f'" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
                )
                for x, py in points:
                    if side == "short":
                        lines.append(
                            f'<rect x="{x - 5:.1f}" y="{py - 5:.1f}" width="10" height="10" fill="{color}"/>'
                        )
                    else:
                        lines.append(
                            f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>'
                        )
        for horizon, x in x_positions.items():
            lines.append(
                f'<text x="{x:.1f}" y="{top + panel_height + 28}" text-anchor="middle" class="axis">{horizon}m</text>'
            )
    lines.append(
        '<text x="56" y="730" class="note">The 15-minute action signal discriminated direction, but the frozen expected-value rule admitted no 15-minute trades.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _seed_stability_svg(report: Mapping[str, object]) -> str:
    rows: dict[str, list[dict[str, object]]] = {}
    for candidate_id in CANDIDATES:
        source = _candidate(report, candidate_id)["diagnostics"]["seed_stability"]
        reduced = []
        for horizon in HORIZONS:
            selected = [row for row in source if int(row["horizon_minutes"]) == horizon]
            reduced.append(
                {
                    "horizon_minutes": horizon,
                    "mean_stability": min(
                        float(row["distribution_mean_spearman"]) for row in selected
                    ),
                    "probability_stability": min(
                        min(
                            float(row["long_probability_spearman"]),
                            float(row["short_probability_spearman"]),
                        )
                        for row in selected
                    ),
                }
            )
        rows[candidate_id] = reduced
    return _two_panel_horizon_svg(
        title="Independent-seed forecast stability",
        description="Minimum pairwise rank agreement across the three independently trained peers.",
        rows=rows,
        upper_key="mean_stability",
        upper_label="Distribution-mean Spearman",
        lower_key="probability_stability",
        lower_label="Minimum long/short probability Spearman",
        upper_reference=0.5,
        lower_reference=0.5,
    )


def _training_svg(report: Mapping[str, object]) -> str:
    rows = {
        candidate_id: _candidate(report, candidate_id)["training_history"]
        for candidate_id in CANDIDATES
    }
    width, height = 1500, 740
    left, right, top, panel_height = 135, 70, 145, 430
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "DirectML training and early stopping",
        "Common three-seed ensemble diagnostics; lower composite loss is better.",
    )
    values = [
        float(row[key])
        for candidate_rows in rows.values()
        for row in candidate_rows
        for key in ("training_composite", "early_stop_composite")
    ]
    low, high = _bounds(values, reference=min(values), padding=0.08)
    maximum_epoch = max(int(row["epoch"]) for candidate_rows in rows.values() for row in candidate_rows)

    def x(epoch: int) -> float:
        return left + (epoch - 1) * chart_width / max(maximum_epoch - 1, 1)

    def y(value: float) -> float:
        return top + panel_height * (high - value) / (high - low)

    for tick in (low, (low + high) / 2.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
        )
    for candidate_id, candidate_rows in rows.items():
        color = COLORS[candidate_id]
        for key, dash in (("early_stop_composite", ""), ("training_composite", ' stroke-dasharray="8 6"')):
            points = [(x(int(row["epoch"])), y(float(row[key]))) for row in candidate_rows]
            lines.append(
                '<polyline points="'
                + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + f'" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
            )
        best = min(candidate_rows, key=lambda row: float(row["best_early_stop_composite"]))
        lines.append(
            f'<circle cx="{x(int(best["best_epoch"])):.1f}" cy="{y(float(best["best_early_stop_composite"])):.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        )
    for epoch in (1, 5, 10, 15, maximum_epoch):
        lines.append(
            f'<text x="{x(epoch):.1f}" y="{top + panel_height + 30}" text-anchor="middle" class="axis">{epoch}</text>'
        )
    lines.append(
        '<text x="56" y="660" class="note">Solid: early-stop role. Dashed: training role. Large markers: retained checkpoints.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _allocation_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 680
    left, right, top, chart_height = 150, 70, 150, 390
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Frozen policy horizon concentration",
        "Share of base-ledger trades by holding horizon; no quota was forced.",
    )
    _append_candidate_legend(lines)
    group_width = chart_width / len(HORIZONS)
    bar_width = 92
    for tick in (0.0, 0.5, 1.0):
        py = top + chart_height * (1.0 - tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="grid"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{100 * tick:.0f}%</text>'
        )
    for h_index, horizon in enumerate(HORIZONS):
        center = left + (h_index + 0.5) * group_width
        for c_index, candidate_id in enumerate(CANDIDATES):
            metrics = _candidate(report, candidate_id)["base"]["metrics"]
            fraction = int(metrics["trades_by_horizon"][str(horizon)]) / int(metrics["trades"])
            x = center + (c_index - 0.5) * (bar_width + 18) - bar_width / 2
            y = top + chart_height * (1.0 - fraction)
            height_value = chart_height * fraction
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{height_value:.1f}" fill="{COLORS[candidate_id]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{max(y - 8, top - 8):.1f}" text-anchor="middle" class="value">{100 * fraction:.1f}%</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 30}" text-anchor="middle" class="axis">{horizon}m</text>'
        )
    lines.append(
        '<text x="56" y="625" class="note">The control placed 97.8% and the mixture 99.7% of trades at 120 minutes, despite the strongest action AUC occurring at 15 minutes.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _policy_svg(report: Mapping[str, object]) -> str:
    width, height = 1500, 930
    left, right, panel_height = 170, 70, 185
    panel_tops = (145, 400, 655)
    lines = _svg_start(
        width,
        height,
        "After-cost policy economics",
        "Identical action ledgers repriced at 12 bps base and 16 bps stress roundtrip charges.",
    )
    categories = [(candidate_id, scenario) for candidate_id in CANDIDATES for scenario in ("base", "stress")]
    chart_width = width - left - right
    x_positions = {
        category: left + (index + 0.5) * chart_width / len(categories)
        for index, category in enumerate(categories)
    }
    panels = (
        ("total_net_return_fraction", "Total net return", 0.0, 100.0, "%"),
        ("maximum_drawdown_fraction", "Maximum drawdown", 0.0, 100.0, "%"),
        ("profit_factor", "Profit factor", 1.05, 1.0, ""),
    )
    for panel_index, (key, label, reference, multiplier, suffix) in enumerate(panels):
        top = panel_tops[panel_index]
        values = [
            multiplier * float(_candidate(report, cid)[scenario]["metrics"][key])
            for cid, scenario in categories
        ]
        scaled_reference = multiplier * reference
        low, high = _bounds(values, reference=scaled_reference)

        def y(value: float) -> float:
            return top + panel_height * (high - value) / (high - low)

        lines.append(f'<text x="{left}" y="{top - 18}" class="label">{label}</text>')
        for tick in (low, scaled_reference, high):
            py = y(tick)
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if math.isclose(tick, scaled_reference) else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.2f}{suffix}</text>'
            )
        for candidate_id, scenario in categories:
            value = multiplier * float(_candidate(report, candidate_id)[scenario]["metrics"][key])
            px = x_positions[(candidate_id, scenario)]
            zero_y = y(0.0 if key != "profit_factor" else scaled_reference)
            value_y = y(value)
            bar_top = min(zero_y, value_y)
            bar_height = max(abs(zero_y - value_y), 2.0)
            opacity = "1.0" if scenario == "base" else "0.55"
            lines.append(
                f'<rect x="{px - 58:.1f}" y="{bar_top:.1f}" width="116" height="{bar_height:.1f}" fill="{COLORS[candidate_id]}" opacity="{opacity}"/>'
            )
            lines.append(
                f'<text x="{px:.1f}" y="{min(value_y, zero_y) - 7:.1f}" text-anchor="middle" class="value">{value:+.2f}{suffix}</text>'
            )
        if panel_index == 2:
            for candidate_id, scenario in categories:
                px = x_positions[(candidate_id, scenario)]
                label_text = ("Control" if candidate_id == CANDIDATES[0] else "Mixture") + " " + scenario
                lines.append(
                    f'<text x="{px:.1f}" y="{top + panel_height + 28}" text-anchor="middle" class="axis">{html.escape(label_text)}</text>'
                )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _monthly_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 700
    left, right, top, chart_height = 145, 70, 145, 390
    chart_width = width - left - right
    months = sorted({str(row["month"]) for row in rows})
    lines = _svg_start(
        width,
        height,
        "Monthly after-cost return",
        "Base is solid and stress is dashed; every series comes from the fixed replay ledger.",
    )
    _append_candidate_legend(lines)
    values = [100.0 * float(row["total_net_return_fraction"]) for row in rows]
    low, high = _bounds(values, reference=0.0)

    def x(month: str) -> float:
        return left + months.index(month) * chart_width / max(len(months) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    for tick in (low, 0.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if math.isclose(tick, 0.0) else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    for candidate_id in CANDIDATES:
        for scenario in ("base", "stress"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["candidate_id"] == candidate_id and row["scenario"] == scenario
                ),
                key=lambda row: str(row["month"]),
            )
            points = [(x(str(row["month"])), y(100.0 * float(row["total_net_return_fraction"]))) for row in selected]
            dash = ' stroke-dasharray="8 6"' if scenario == "stress" else ""
            lines.append(
                '<polyline points="'
                + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + f'" fill="none" stroke="{COLORS[candidate_id]}" stroke-width="3"{dash}/>'
            )
    for month in months:
        lines.append(
            f'<text x="{x(month):.1f}" y="{top + chart_height + 30}" text-anchor="middle" class="axis">{month}</text>'
        )
    lines.append(
        '<text x="56" y="640" class="note">The mixture had one positive base month; all other candidate/scenario months were negative.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _equity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 700
    left, right, top, chart_height = 145, 70, 145, 390
    chart_width = width - left - right
    dates = sorted({str(row["date"]) for row in rows})
    lines = _svg_start(
        width,
        height,
        "Dated portfolio equity",
        "One-third symbol sleeves, no leverage, no overlapping position within a symbol.",
    )
    _append_candidate_legend(lines)
    values = [float(row["equity"]) for row in rows]
    low, high = _bounds(values, reference=1.0)

    def x(date: str) -> float:
        return left + dates.index(date) * chart_width / max(len(dates) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    for tick in (low, 1.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if math.isclose(tick, 1.0) else "grid"}"/><text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    for candidate_id in CANDIDATES:
        for scenario in ("base", "stress"):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["candidate_id"] == candidate_id and row["scenario"] == scenario
                ),
                key=lambda row: str(row["date"]),
            )
            points = [(x(str(row["date"])), y(float(row["equity"]))) for row in selected]
            dash = ' stroke-dasharray="8 6"' if scenario == "stress" else ""
            lines.append(
                '<polyline points="'
                + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
                + f'" fill="none" stroke="{COLORS[candidate_id]}" stroke-width="3"{dash}/>'
            )
    for date in (dates[0], dates[len(dates) // 2], dates[-1]):
        lines.append(
            f'<text x="{x(date):.1f}" y="{top + chart_height + 30}" text-anchor="middle" class="axis">{date}</text>'
        )
    lines.append(
        '<text x="56" y="640" class="note">Solid: 12 bps roundtrip. Dashed: identical trades repriced at 16 bps roundtrip.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _failure_analysis(
    report: Mapping[str, object],
    source_report_sha: str,
    binding_sha: str,
    evidence_root: Path,
) -> dict[str, object]:
    candidates: dict[str, object] = {}
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        metrics = candidate["base"]["metrics"]
        candidates[candidate_id] = {
            "best_epoch": candidate["artifacts"][0]["best_epoch"],
            "distribution_gate": candidate["diagnostics"]["distribution_gate"],
            "action_gate": candidate["diagnostics"]["action_gate"],
            "economic_gate": candidate["economic_gate"],
            "base_metrics": metrics,
            "stress_metrics": candidate["stress"]["metrics"],
            "fifteen_minute_action_auc": {
                row["side"]: row["roc_auc"]
                for row in candidate["diagnostics"]["actions"]
                if row["horizon_minutes"] == 15
            },
            "fifteen_minute_trades": metrics["trades_by_horizon"]["15"],
            "one_hundred_twenty_minute_trade_fraction": metrics[
                "trades_by_horizon"
            ]["120"]
            / metrics["trades"],
        }
    analysis: dict[str, object] = {
        "schema_version": "round-048-failure-analysis-v1",
        "round": ROUND,
        "status": "rejected",
        "evidence": {
            "report_canonical_sha256": source_report_sha,
            "report_file_sha256": _file_sha256(evidence_root / "report.json"),
            "design_sha256": report["design_sha256"],
            "binding_sha256": binding_sha,
            "implementation_commit": report["implementation_commit"],
            "dataset_sha256": report["dataset"]["dataset_sha256"],
            "timestamps": report["dataset"]["timestamps"],
            "rows": report["dataset"]["rows"],
            "evaluation_timestamps": report["dataset"]["role_timestamp_counts"][
                "viability"
            ],
            "directml_models": 6,
            "exact_reload_failures": 0,
            "cpu_fallback_warnings": report["backend"]["cpu_fallback_warnings"],
        },
        "mixture_ablation_gate": report["mixture_ablation_gate"],
        "candidate_results": candidates,
        "critical_interpretation": [
            "The state-conditioned mixture improved average negative log likelihood by 0.7381 percent and used more than two effective components, but this distributional improvement did not improve conditional-mean ranking or after-cost economics.",
            "Both candidates learned useful 15-minute cost-covering direction probabilities with ROC AUC near 0.618, yet the frozen policy opened zero 15-minute trades because all three seeds also had to predict positive expected net value.",
            "The policy instead concentrated 97.84 percent of control trades and 99.75 percent of mixture trades at 120 minutes, where direction and conditional-mean skill were weakest. This is a model-policy alignment failure, not evidence that more leverage or lower costs would help.",
            "Every bootstrap interval was negative, profit factors were below one, and drawdowns exceeded 41 percent. No point estimate supports profitability, testnet execution, AI uplift, or leverage.",
            "The probabilistic likelihood learned return scale and tail shape substantially better than unconditional baselines, while predicted conditional means remained too compressed relative to realized volatility. A likelihood-only mean is therefore not an adequate action-value estimator for this dataset.",
        ],
        "decisions": [
            "Reject both Round 48 candidates and preserve the 2025-H1 period as consumed development evidence.",
            "Do not tune the probability floor, expected-value threshold, costs, drawdown limit, or horizon choice against these outcomes.",
            "Do not run the local finance language model: deterministic candidates failed mandatory numerical and economic gates.",
            "Retain the tail-stable DirectML logistic likelihood, exact reload checks, hash-bound evidence, and independent-seed supervisor infrastructure.",
        ],
        "next_model_requirements": [
            "Make 15 minutes the only executable primary horizon; use 30 minutes only as an auxiliary representation target so horizon base rates cannot route actions toward weak 120-minute forecasts.",
            "Replace the distribution mean as action value with a three-action hurdle decomposition: calibrated action-profit probability, conditional gain severity, conditional loss severity, and a frozen expected-net-utility identity after 12 bps costs.",
            "Train the utility decomposition with robust conditional-magnitude losses plus bounded pairwise decision ranking; measure each component against training-only baselines and require seed agreement before replay.",
            "Use uncertainty as abstention, never as forced activity. Require positive fixed-ledger stress bootstrap, profit factor, drawdown, monthly breadth, and symbol-diversification gates before opening untouched confirmation data.",
            "Treat the available one-second taker-flow corpus as a later execution-veto/timing input, not as standalone direction alpha. Expand it chronologically only after the slower 15-minute candidate passes consumed-development gates.",
            "Keep any finance-language-model contribution asynchronous and veto-only. A paired AI-versus-ML ablation is eligible only after the deterministic action-value model clears every gate.",
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
        metrics = candidate["base"]["metrics"]
        stress = candidate["stress"]["metrics"]
        max_auc = max(float(row["roc_auc"]) for row in candidate["diagnostics"]["actions"])
        rows.append(
            f"| {LABELS[candidate_id]} | {max_auc:.3f} | {metrics['trades']} | "
            f"{100 * float(metrics['total_net_return_fraction']):+.2f}% | "
            f"{100 * float(stress['total_net_return_fraction']):+.2f}% | "
            f"{100 * float(metrics['maximum_drawdown_fraction']):.2f}% | "
            f"{float(metrics['profit_factor']):.3f} | False/True/False |"
        )
    runtime = report["runtime_evidence"]
    return f"""# Round 48: Minute Logistic-Mixture TCN

> **Beta research warning:** neither model is approved for testnet, live day trading, leverage, or autonomous execution. The 2025-H1 result is consumed development evidence.

Round 48 trained six causal large-kernel TCNs on 366,035 five-minute timestamps derived from verified Binance one-minute archives. The mixture improved return-density likelihood, but both policies lost money after fixed costs and were rejected.

![Forecast quality](charts/forecast-quality.svg)

![Action quality](charts/action-quality.svg)

| Candidate | Best action AUC | Trades | Base return | Stress return | Base drawdown | Profit factor | Distribution/action/economic gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

The strongest measured signal was 15-minute direction classification, but the expected-value rule admitted no 15-minute trades. Instead, both ledgers concentrated almost entirely at 120 minutes and failed every economic gate.

![Horizon allocation](charts/horizon-allocation.svg)

![Policy economics](charts/policy-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Seed stability](charts/seed-stability.svg)

![Training dynamics](charts/training-dynamics.svg)

![Research progress](charts/research-progress.svg)

DirectML completed in `{float(runtime['elapsed_seconds']):.1f}s`, peaked at `{float(runtime['memory']['peak_working_set_bytes']) / 2**30:.2f} GiB` working set, recorded zero CPU fallbacks, and reloaded all six models exactly. AI was correctly withheld because no deterministic candidate passed.

Data: [forecast horizons](horizons.csv) | [action horizons](action-horizons.csv) | [symbol horizons](symbol-horizons.csv) | [seed stability](seed-stability.csv) | [training](training.csv) | [models](models.csv) | [roles](roles.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [source lineage](sources.csv) | [progress](progress.csv) | [failure analysis](../round-048-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
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
    report, design, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path, binding_path
    )
    tables = _table_rows(report, design)
    progress_fields, progress_rows = _progress_rows(prior_progress_path, report)
    tables["progress"] = progress_rows
    charts = output_dir / "charts"
    table_paths = {output_dir / f"{name}.csv" for name in tables}
    chart_paths = {
        charts / "forecast-quality.svg",
        charts / "action-quality.svg",
        charts / "seed-stability.svg",
        charts / "training-dynamics.svg",
        charts / "horizon-allocation.svg",
        charts / "policy-economics.svg",
        charts / "monthly-economics.svg",
        charts / "daily-equity.svg",
        charts / "research-progress.svg",
    }
    expected = table_paths | chart_paths | {
        output_dir / "README.md",
        output_dir / "screen.json",
        output_dir / "report.json",
    }
    _clean_output(output_dir, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if name == "progress":
            _write_csv(output_dir / "progress.csv", rows, fields=progress_fields)
        else:
            _write_rows(output_dir / f"{name}.csv", rows)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(report))
    _write_text(charts / "action-quality.svg", _action_svg(report))
    _write_text(charts / "seed-stability.svg", _seed_stability_svg(report))
    _write_text(charts / "training-dynamics.svg", _training_svg(report))
    _write_text(charts / "horizon-allocation.svg", _allocation_svg(report))
    _write_text(charts / "policy-economics.svg", _policy_svg(report))
    _write_text(charts / "monthly-economics.svg", _monthly_svg(tables["monthly"]))
    _write_text(charts / "daily-equity.svg", _equity_svg(tables["daily-equity"]))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress_rows))
    failure = _failure_analysis(report, source_report_sha, binding_sha, evidence_root)
    _write_text(
        failure_path,
        json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "minute_logistic_mixture_tcn_graph_data",
        "round": ROUND,
        "status": "quality_or_economic_gate_rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "dataset_sha256": report["dataset"]["dataset_sha256"],
        "dataset_rows": report["dataset"]["rows"],
        "evaluation_timestamps": report["dataset"]["role_timestamp_counts"][
            "viability"
        ],
        "directml_model_artifact_count": 6,
        "candidate_distribution_gate_pass_count": 0,
        "candidate_action_gate_pass_count": 2,
        "candidate_economic_gate_pass_count": 0,
        "mixture_ablation_passed": report["mixture_ablation_gate"]["passed"],
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
        default=research / "round-048-minute-logistic-mixture-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-048-minute-logistic-mixture-tcn-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-path",
        type=Path,
        default=research / "round-048-failure-analysis.json",
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
