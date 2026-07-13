"""Publish verified Round 43 stateful turnover and AI-factor evidence."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
import gzip
import hashlib
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
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_stateful_turnover_ai_factor_ablation import (  # noqa: E402
    AUDIT_SCHEMA,
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 43
PUBLICATION_SCHEMA = "stateful-turnover-ai-factor-publication-v1"
STATEFUL_CANDIDATES = (
    "baseline_71_long_only",
    "baseline_71_long_short",
    "ai_research_augmented_77_long_only",
    "ai_research_augmented_77_long_short",
)
SHORT_LABELS = {
    "baseline_71_long_only": "ML long-only",
    "baseline_71_long_short": "ML long-short",
    "ai_research_augmented_77_long_only": "AI-factor long-only",
    "ai_research_augmented_77_long_short": "AI-factor long-short",
}
COLORS = {
    "baseline_71_long_only": "#2563a6",
    "baseline_71_long_short": "#0f766e",
    "ai_research_augmented_77_long_only": "#b7791f",
    "ai_research_augmented_77_long_short": "#7b559c",
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
    audit_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], str, str]:
    design = _read_object(design_path, "Round 43 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 43 design")
    audit = _read_object(audit_path, "Round 43 AI audit")
    audit_sha = _canonical_identity(audit, "audit_sha256", "Round 43 AI audit")
    binding = _read_object(binding_path, "Round 43 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 43 binding")
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, "Round 43 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 43 report"
    )
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or audit.get("schema_version") != AUDIT_SCHEMA
        or binding.get("schema_version") != BINDING_SCHEMA
        or report.get("schema_version") != REPORT_SCHEMA
        or any(item.get("round") != ROUND for item in (design, audit, binding, report))
        or binding.get("design_sha256") != design_sha
        or binding.get("ai_factor_audit_sha256") != audit_sha
        or report.get("design_sha256") != design_sha
        or report.get("ai_factor_audit_sha256") != audit_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
    ):
        raise ValueError("Round 43 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    compute = report.get("compute")
    models = report.get("models")
    stateful = report.get("stateful_replays")
    comparators = report.get("comparators")
    gates = report.get("stateful_viability_gates")
    claims = report.get("claims")
    outputs = report.get("outputs")
    if (
        report.get("status") != "complete"
        or not isinstance(dataset, Mapping)
        or dataset.get("rows") != 91_941
        or dataset.get("hourly_timestamps") != 30_647
        or dataset.get("baseline_feature_count") != 71
        or dataset.get("augmented_feature_count") != 77
        or not isinstance(compute, Mapping)
        or compute.get("backend_kind") != "opencl"
        or compute.get("model_artifacts") != 12
        or compute.get("all_artifacts_exact_reload") is not True
        or not isinstance(models, list)
        or len(models) != 12
        or any(model.get("reload_max_abs_prediction_error") != 0.0 for model in models)
        or not isinstance(stateful, list)
        or len(stateful) != 8
        or not isinstance(comparators, list)
        or len(comparators) != 10
        or not isinstance(gates, list)
        or len(gates) != 4
        or any(gate.get("passed") is not False for gate in gates)
        or report.get("ai_uplift_gate", {}).get("passed") is not False
        or not isinstance(claims, Mapping)
        or any(
            claims.get(field) is not False
            for field in (
                "any_stateful_viability_gate_passed",
                "ai_uplift_gate_passed",
                "profitability_established",
                "ai_improvement_established",
                "selection_confirmation_established",
                "promotion_authorized",
                "testnet_or_live_trading_authorized",
                "leverage_authorized",
            )
        )
        or report.get("position_ledger_rows") != 234_522
        or not isinstance(outputs, list)
    ):
        raise ValueError("Round 43 source, model, or rejection evidence drifted")
    for item in outputs:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 43 evidence output drifted: {path}")
    return report, report_sha, binding_sha


def _replay_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for family, values in (
        ("stateful", report["stateful_replays"]),
        ("comparator", report["comparators"]),
    ):
        for metrics in values:
            interval = metrics["bootstrap_mean_hourly_net_bps"]
            output.append(
                {
                    "replay_family": family,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key
                        not in {"monthly", "symbols", "bootstrap_mean_hourly_net_bps"}
                    },
                    "bootstrap_lower_bps": interval["lower_bps"],
                    "bootstrap_median_bps": interval["median_bps"],
                    "bootstrap_upper_bps": interval["upper_bps"],
                    "bootstrap_lower_quantile": interval["lower_quantile"],
                }
            )
    return output


def _monthly_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for family, values in (
        ("stateful", report["stateful_replays"]),
        ("comparator", report["comparators"]),
    ):
        for metrics in values:
            for row in metrics["monthly"]:
                output.append(
                    {
                        "replay_family": family,
                        "candidate_id": metrics["candidate_id"],
                        "feature_set": metrics["feature_set"],
                        "mode": metrics["mode"],
                        "cost_scenario": metrics["cost_scenario"],
                        **row,
                    }
                )
    return output


def _symbol_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for family, values in (
        ("stateful", report["stateful_replays"]),
        ("comparator", report["comparators"]),
    ):
        for metrics in values:
            for row in metrics["symbols"]:
                output.append(
                    {
                        "replay_family": family,
                        "candidate_id": metrics["candidate_id"],
                        "feature_set": metrics["feature_set"],
                        "mode": metrics["mode"],
                        "cost_scenario": metrics["cost_scenario"],
                        **row,
                    }
                )
    return output


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            **model,
            "top_feature_gain": json.dumps(
                model["top_feature_gain"], separators=(",", ":"), sort_keys=True
            ),
        }
        for model in report["models"]
    ]


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for gate in report["stateful_viability_gates"]:
        output.append(
            {
                "gate": "stateful_viability",
                "candidate_id": gate["candidate_id"],
                "passed": gate["passed"],
                "failed_checks": ";".join(gate["reasons"]),
                "checks_json": json.dumps(
                    gate["checks"], separators=(",", ":"), sort_keys=True
                ),
            }
        )
    ai = report["ai_uplift_gate"]
    output.append(
        {
            "gate": "ai_uplift_primary_long_only",
            "candidate_id": "ai_research_augmented_77_long_only_minus_baseline_71_long_only",
            "passed": ai["passed"],
            "failed_checks": ";".join(ai["reasons"]),
            "checks_json": json.dumps(
                ai["checks"], separators=(",", ":"), sort_keys=True
            ),
        }
    )
    return output


def _ai_uplift_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    gate = report["ai_uplift_gate"]
    interval = gate["paired_stress_hourly_delta_bps"]
    return [
        {
            "pair": "long_only_augmented_minus_baseline",
            "cost_scenario": "stress",
            "observed_mean_hourly_delta_bps": interval["observed_mean"],
            "bootstrap_lower_bps": interval["lower_bps"],
            "bootstrap_median_bps": interval["median_bps"],
            "bootstrap_upper_bps": interval["upper_bps"],
            "bootstrap_samples": interval["samples"],
            "bootstrap_block_hours": interval["block_hours"],
            "gate_passed": gate["passed"],
            "failed_checks": ";".join(gate["reasons"]),
        }
    ]


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
    ledger_path = evidence_root / "position_ledger.csv.gz"
    series: dict[tuple[str, str, str, str], list[tuple[str, float]]] = defaultdict(list)
    with gzip.open(ledger_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["symbol"] != "BTCUSDT":
                continue
            key = (
                row["candidate_id"],
                row["feature_set"],
                row["mode"],
                row["cost_scenario"],
            )
            series[key].append(
                (row["decision_time_utc"][:10], float(row["portfolio_net_bps"]))
            )
    output: list[dict[str, object]] = []
    for key, observations in sorted(series.items()):
        equity = 1.0
        current_date = ""
        day_last_equity = 1.0
        day_net_bps = 0.0
        day_hours = 0
        for date, net_bps in observations:
            if current_date and date != current_date:
                output.append(
                    {
                        "candidate_id": key[0],
                        "feature_set": key[1],
                        "mode": key[2],
                        "cost_scenario": key[3],
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
            equity *= 1.0 + net_bps / 10_000.0
            day_last_equity = equity
            day_net_bps += net_bps
            day_hours += 1
        if current_date:
            output.append(
                {
                    "candidate_id": key[0],
                    "feature_set": key[1],
                    "mode": key[2],
                    "cost_scenario": key[3],
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
    best = next(
        item
        for item in report["stateful_replays"]
        if item["candidate_id"] == "ai_research_augmented_77_long_short"
        and item["cost_scenario"] == "base"
    )
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "stateful turnover-aware hourly ML and AI-factor ablation",
            "periods": "2022-10-01..2025-06-30 rolling roles; eval 2025-H1",
            "selection_contaminated": True,
            "horizon_seconds": 3600,
            "feature_set": "71 causal price/flow features + 6 AI-research interactions",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "spearman_ic": 0.03223722099967383,
            "selected_signals": best["transition_events"],
            "executable_trades": best["transition_events"],
            "mean_net_bps": best["mean_hourly_net_bps"],
            "status": "rejected",
            "source_file": "verified Round 43 stateful turnover/AI-factor report; mean is hourly portfolio bps",
            "best_policy_trades": best["transition_events"],
            "best_policy_total_net_bps": 10_000.0
            * float(best["total_net_return_fraction"]),
            "best_policy_mean_net_bps": best["mean_hourly_net_bps"],
            "best_policy_max_drawdown_bps": 10_000.0
            * float(best["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": best["profit_factor"],
            "best_model_id": best["candidate_id"],
            "daily_model_fits": 12,
            "accepted_thresholds": 0,
            "ensemble_models": 12,
            "valid_barrier_rows": 13_029,
            "policy_eligible_rows": 13_029,
            "development_consumed": True,
        }
    )
    rows.append(row)
    rows.sort(key=lambda item: int(item["round"]))
    return fields, rows


def _economics_svg(report: Mapping[str, object]) -> str:
    values = {
        (item["candidate_id"], item["cost_scenario"]): item
        for item in report["stateful_replays"]
    }
    width, height = 1500, 760
    left, right, top, chart_height = 115, 65, 150, 390
    chart_width = width - left - right
    returns = [
        100.0 * float(values[(candidate, scenario)]["total_net_return_fraction"])
        for candidate in STATEFUL_CANDIDATES
        for scenario in ("base", "stress")
    ]
    lower = min(-11.0, min(returns) - 1.0)
    upper = max(8.0, max(returns) + 1.0)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines = _svg_start(
        width,
        height,
        "Round 43 stateful returns were not stable across cost-aware policies",
        "Compounded 2025-H1 net return; stress also raises the transition hurdle, so the ledgers are not matched cost sensitivity.",
    )
    lines.extend(
        [
            f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
            '<rect x="1040" y="47" width="16" height="16" fill="#197a55"/><text x="1066" y="60" class="note">positive</text>',
            '<rect x="1160" y="47" width="16" height="16" fill="#b42318"/><text x="1186" y="60" class="note">negative</text>',
            '<rect x="1275" y="47" width="16" height="16" fill="none" stroke="#17212b" stroke-width="2"/><text x="1301" y="60" class="note">base</text>',
            '<rect x="1370" y="47" width="16" height="16" fill="none" stroke="#17212b" stroke-width="2" stroke-dasharray="4 3"/><text x="1396" y="60" class="note">stress</text>',
        ]
    )
    for tick in range(math.floor(lower / 2) * 2, math.ceil(upper / 2) * 2 + 1, 2):
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left - 13}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
        )
    group_width = chart_width / len(STATEFUL_CANDIDATES)
    for index, candidate in enumerate(STATEFUL_CANDIDATES):
        center = left + group_width * (index + 0.5)
        for offset, scenario in ((-44, "base"), (44, "stress")):
            metrics = values[(candidate, scenario)]
            value = 100.0 * float(metrics["total_net_return_fraction"])
            py = y(value)
            zero = y(0.0)
            fill = "#197a55" if value >= 0.0 else "#b42318"
            dash = "" if scenario == "base" else ' stroke-dasharray="5 3"'
            lines.append(
                f'<rect x="{center + offset - 31:.1f}" y="{min(py, zero):.1f}" width="62" height="{max(2.0, abs(zero - py)):.1f}" fill="{fill}" fill-opacity="0.82" stroke="#17212b" stroke-width="2"{dash}/>'
            )
            lines.append(
                f'<text x="{center + offset:.1f}" y="{py - 10 if value >= 0 else py + 20:.1f}" text-anchor="middle" class="value">{value:+.2f}%</text>'
            )
            lines.append(
                f'<text x="{center + offset:.1f}" y="{top + chart_height + 25}" text-anchor="middle" class="axis">{scenario}</text>'
            )
        base = values[(candidate, "base")]
        stress = values[(candidate, "stress")]
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 58}" text-anchor="middle" class="label">{html.escape(SHORT_LABELS[candidate])}</text>'
        )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 82}" text-anchor="middle" class="note">DD {100 * float(base["maximum_drawdown_fraction"]):.1f}% / {100 * float(stress["maximum_drawdown_fraction"]):.1f}%</text>'
        )
    lines.extend(
        [
            f'<text x="{left}" y="{height - 46}" class="note">No candidate passed: every familywise 168-hour bootstrap lower bound was negative and activity covered only 20-32 days.</text>',
            f'<text x="{left}" y="{height - 24}" class="note">Unlevered fixed one-third sleeves; exact funding; 6 bps base and 8 bps stress one-way charges per unit transition.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(report: Mapping[str, object]) -> str:
    diagnostics = {
        (row["feature_set"], row["evaluation_month"]): row
        for row in report["forecast_diagnostics"]
        if row["role"] == "evaluation" and row["symbol"] == "ALL"
    }
    models = {
        (row["feature_set"], row["evaluation_month"]): row for row in report["models"]
    }
    months = [f"2025-{month:02d}" for month in range(1, 7)]
    width, height = 1500, 800
    left, right, top, panel_height, gap = 115, 65, 150, 220, 100
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Round 43 forecasts lost association after January",
        "Pooled out-of-sample Spearman IC and prior-month nonnegative amplitude calibration, by evaluation month.",
    )
    labels = (
        ("baseline_71", "ML 71", "#2563a6"),
        ("ai_research_augmented_77", "AI-factor 77", "#b7791f"),
    )
    for legend_index, (_, label, color) in enumerate(labels):
        x = 1090 + legend_index * 180
        lines.append(
            f'<line x1="{x}" y1="52" x2="{x + 28}" y2="52" stroke="{color}" stroke-width="4"/><circle cx="{x + 14}" cy="52" r="5" fill="{color}"/><text x="{x + 38}" y="57" class="note">{label}</text>'
        )
    x_positions = {
        month: left + chart_width * index / (len(months) - 1)
        for index, month in enumerate(months)
    }
    for panel, (title, lower, upper) in enumerate(
        (
            ("Spearman information coefficient", -0.02, 0.05),
            ("Amplitude slope", 0.0, 1.7),
        )
    ):
        panel_top = top + panel * (panel_height + gap)
        lines.append(
            f'<text x="{left}" y="{panel_top - 22}" class="label">{title}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{panel_top}" width="{chart_width}" height="{panel_height}" fill="#ffffff" stroke="#d8e0e7"/>'
        )

        def y(value: float) -> float:
            return panel_top + panel_height * (upper - value) / (upper - lower)

        for tick in np_ticks(lower, upper, 4):
            py = y(tick)
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if abs(tick) < 1e-12 else "grid"}"/>'
            )
            lines.append(
                f'<text x="{left - 13}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
            )
        for feature_set, _, color in labels:
            points: list[str] = []
            for month in months:
                value = (
                    float(
                        diagnostics[(feature_set, month)][
                            "spearman_information_coefficient"
                        ]
                    )
                    if panel == 0
                    else float(models[(feature_set, month)]["amplitude_slope"])
                )
                points.append(f"{x_positions[month]:.1f},{y(value):.1f}")
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
            )
            for month, point in zip(months, points, strict=True):
                px, py = point.split(",")
                lines.append(f'<circle cx="{px}" cy="{py}" r="5" fill="{color}"/>')
        if panel == 1:
            for month in months:
                lines.append(
                    f'<text x="{x_positions[month]:.1f}" y="{panel_top + panel_height + 24}" text-anchor="middle" class="axis">{month}</text>'
                )
    lines.extend(
        [
            f'<text x="{left}" y="{height - 28}" class="note">A zero slope is a fail-closed abstention: the immediately prior calibration relation was nonpositive, so the forecast was set to zero.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def np_ticks(lower: float, upper: float, count: int) -> list[float]:
    return [lower + index * (upper - lower) / count for index in range(count + 1)]


def _equity_svg(daily_rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row
        for row in daily_rows
        if row["candidate_id"] in STATEFUL_CANDIDATES and row["cost_scenario"] == "base"
    ]
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[str(row["candidate_id"])].append(row)
    dates = sorted({str(row["date_utc"]) for row in selected})
    values = [100.0 * float(row["cumulative_return_fraction"]) for row in selected]
    lower = min(-12.0, min(values) - 1.0)
    upper = max(6.0, max(values) + 1.0)
    width, height = 1500, 700
    left, right, top, chart_height = 115, 65, 145, 410
    chart_width = width - left - right

    def x(date: str) -> float:
        return left + chart_width * dates.index(date) / (len(dates) - 1)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines = _svg_start(
        width,
        height,
        "Round 43 daily equity paths show concentration, not a durable edge",
        "Base cost-aware stateful policies, 2025-01-01 through 2025-06-30 UTC; daily points are derived from the full hourly ledger.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    for tick in range(math.floor(lower / 4) * 4, math.ceil(upper / 4) * 4 + 1, 4):
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left - 13}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d}%</text>'
        )
    for candidate in STATEFUL_CANDIDATES:
        points = " ".join(
            f"{x(str(row['date_utc'])):.1f},{y(100 * float(row['cumulative_return_fraction'])):.1f}"
            for row in grouped[candidate]
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{COLORS[candidate]}" stroke-width="3"/>'
        )
    tick_dates = (
        "2025-01-01",
        "2025-02-01",
        "2025-03-01",
        "2025-04-01",
        "2025-05-01",
        "2025-06-01",
        "2025-06-30",
    )
    for date in tick_dates:
        if date in dates:
            px = x(date)
            lines.append(
                f'<line x1="{px:.1f}" y1="{top + chart_height}" x2="{px:.1f}" y2="{top + chart_height + 7}" stroke="#60717f"/>'
            )
            lines.append(
                f'<text x="{px:.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{date[5:]}</text>'
            )
    for index, candidate in enumerate(STATEFUL_CANDIDATES):
        x_legend = left + index * 330
        lines.append(
            f'<line x1="{x_legend}" y1="{height - 74}" x2="{x_legend + 28}" y2="{height - 74}" stroke="{COLORS[candidate]}" stroke-width="4"/><text x="{x_legend + 38}" y="{height - 69}" class="note">{html.escape(SHORT_LABELS[candidate])}</text>'
        )
    lines.extend(
        [
            f'<text x="{left}" y="{height - 28}" class="note">Flat sections are genuine abstention after zero calibration slopes; they are not missing data.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _ai_uplift_svg(report: Mapping[str, object]) -> str:
    indexed = {
        (item["candidate_id"], item["cost_scenario"]): item
        for item in report["stateful_replays"]
    }
    ai = report["ai_uplift_gate"]
    interval = ai["paired_stress_hourly_delta_bps"]
    width, height = 1500, 650
    lines = _svg_start(
        width,
        height,
        "AI-assisted factors changed point estimates but did not establish uplift",
        "Pre-registered long-only comparison; the paired 168-hour block-bootstrap interval crosses zero and drawdown worsened.",
    )
    panel_x = (130, 650, 1060)
    panel_width = (390, 300, 330)
    titles = ("Compounded return", "Maximum drawdown", "Paired stress delta")
    for x_value, width_value, title in zip(panel_x, panel_width, titles, strict=True):
        lines.append(
            f'<text x="{x_value}" y="132" class="label">{title}</text><rect x="{x_value}" y="150" width="{width_value}" height="300" fill="#ffffff" stroke="#d8e0e7"/>'
        )
    return_values = [
        100.0 * float(indexed[(candidate, scenario)]["total_net_return_fraction"])
        for scenario in ("base", "stress")
        for candidate in ("baseline_71_long_only", "ai_research_augmented_77_long_only")
    ]
    lower, upper = -5.5, 5.5

    def return_y(value: float) -> float:
        return 150 + 300 * (upper - value) / (upper - lower)

    lines.append(
        f'<line x1="130" y1="{return_y(0):.1f}" x2="520" y2="{return_y(0):.1f}" class="zero"/>'
    )
    positions = (190, 270, 380, 460)
    for index, (position, value) in enumerate(
        zip(positions, return_values, strict=True)
    ):
        fill = "#2563a6" if index % 2 == 0 else "#b7791f"
        py = return_y(value)
        zero = return_y(0.0)
        lines.append(
            f'<rect x="{position - 24}" y="{min(py, zero):.1f}" width="48" height="{max(2, abs(zero - py)):.1f}" fill="{fill}"/>'
        )
        lines.append(
            f'<text x="{position}" y="{py - 9 if value >= 0 else py + 19:.1f}" text-anchor="middle" class="value">{value:+.2f}%</text>'
        )
        lines.append(
            f'<text x="{position}" y="477" text-anchor="middle" class="axis">{("base", "stress")[index // 2]}</text>'
        )
    drawdowns = [
        100.0 * float(indexed[(candidate, "stress")]["maximum_drawdown_fraction"])
        for candidate in (
            "baseline_71_long_only",
            "ai_research_augmented_77_long_only",
        )
    ]
    for position, value, color in zip(
        (740, 860), drawdowns, ("#2563a6", "#b7791f"), strict=True
    ):
        height_value = 250.0 * value / 10.0
        lines.append(
            f'<rect x="{position - 34}" y="{430 - height_value:.1f}" width="68" height="{height_value:.1f}" fill="{color}"/><text x="{position}" y="{420 - height_value:.1f}" text-anchor="middle" class="value">{value:.2f}%</text>'
        )
    lines.append(
        '<text x="740" y="477" text-anchor="middle" class="axis">ML</text><text x="860" y="477" text-anchor="middle" class="axis">AI-factor</text>'
    )
    delta_lower, delta_upper = -0.6, 0.7

    def delta_x(value: float) -> float:
        return 1090 + 270 * (value - delta_lower) / (delta_upper - delta_lower)

    zero_x = delta_x(0.0)
    lines.append(
        f'<line x1="{zero_x:.1f}" y1="185" x2="{zero_x:.1f}" y2="420" class="zero"/>'
    )
    lines.append(
        f'<line x1="{delta_x(float(interval["lower_bps"])):.1f}" y1="300" x2="{delta_x(float(interval["upper_bps"])):.1f}" y2="300" stroke="#7b559c" stroke-width="6"/><circle cx="{delta_x(float(interval["observed_mean"])):.1f}" cy="300" r="9" fill="#7b559c"/>'
    )
    lines.append(
        f'<text x="1225" y="350" text-anchor="middle" class="value">95% [{float(interval["lower_bps"]):+.3f}, {float(interval["upper_bps"]):+.3f}] bps/hour</text>'
    )
    lines.extend(
        [
            '<rect x="170" y="520" width="16" height="16" fill="#2563a6"/><text x="196" y="533" class="note">ML baseline</text>',
            '<rect x="340" y="520" width="16" height="16" fill="#b7791f"/><text x="366" y="533" class="note">AI-research factors</text>',
            f'<text x="650" y="533" class="note">Gate: rejected ({html.escape("; ".join(ai["reasons"]))})</text>',
            '<text x="130" y="592" class="note">Positive point estimates are insufficient: matched resampling and non-degrading drawdown were mandatory.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _readme(report: Mapping[str, object]) -> str:
    indexed = {
        (item["candidate_id"], item["cost_scenario"]): item
        for item in report["stateful_replays"]
    }
    rows = []
    for candidate in STATEFUL_CANDIDATES:
        base = indexed[(candidate, "base")]
        stress = indexed[(candidate, "stress")]
        rows.append(
            f"| {SHORT_LABELS[candidate]} | {100 * float(base['total_net_return_fraction']):+.2f}% | {100 * float(stress['total_net_return_fraction']):+.2f}% | {100 * float(base['maximum_drawdown_fraction']):.2f}% | {float(stress['bootstrap_mean_hourly_net_bps']['lower_bps']):+.3f} | {stress['active_days']} |"
        )
    return f"""# Round 43: Stateful Turnover and AI-Factor Ablation

> **Beta research warning:** rejected, selection-contaminated development evidence. No model is approved for testnet, live day trading, leverage, or autonomous execution.

Round 43 replaced fictitious hourly close/reopen cycles with persistent positions and actual transition costs. It also tested six bounded factors proposed through a local 8B AI research workflow. All 12 monthly LightGBM models ran on OpenCL and reloaded exactly; no candidate passed.

![Stateful economics](charts/stateful-economics.svg)

| Candidate | Base return | Stress-policy return | Base max DD | Stress bootstrap lower bps/hour | Active days |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The stress ledger is **not** a matched cost-only sensitivity: its 8 bps one-way charge also raises the transition hurdle. This explains why some stress point estimates exceed base. Future stress tests must reprice one fixed action ledger.

![Forecast stability](charts/forecast-stability.svg)

![Daily equity](charts/daily-equity.svg)

![AI uplift](charts/ai-uplift.svg)

The primary AI-factor long-only pair improved point estimates, but its paired stress delta was `{float(report["ai_uplift_gate"]["paired_stress_hourly_delta_bps"]["observed_mean"]):+.3f}` bps/hour with a 95% block-bootstrap interval of `[{float(report["ai_uplift_gate"]["paired_stress_hourly_delta_bps"]["lower_bps"]):+.3f}, {float(report["ai_uplift_gate"]["paired_stress_hourly_delta_bps"]["upper_bps"]):+.3f}]`; drawdown also worsened. AI uplift is not established.

![Research progress](charts/research-progress.svg)

Data: [replays](replays.csv) | [monthly](monthly.csv) | [symbols](symbols.csv) | [forecast diagnostics](diagnostics.csv) | [models](models.csv) | [gates](gates.csv) | [AI uplift](ai-uplift.csv) | [daily equity](daily-equity.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-043-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
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
    audit_path: Path,
    binding_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path, audit_path, binding_path
    )
    replay_rows = _replay_rows(report)
    monthly_rows = _monthly_rows(report)
    symbol_rows = _symbol_rows(report)
    model_rows = _model_rows(report)
    gate_rows = _gate_rows(report)
    ai_rows = _ai_uplift_rows(report)
    source_rows = _source_rows(report)
    daily_rows = _daily_equity_rows(evidence_root)
    progress_fields, progress_rows = _progress_rows(prior_progress_path, report)
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "replays.csv",
        output_dir / "monthly.csv",
        output_dir / "symbols.csv",
        output_dir / "diagnostics.csv",
        output_dir / "models.csv",
        output_dir / "gates.csv",
        output_dir / "ai-uplift.csv",
        output_dir / "daily-equity.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "stateful-economics.svg",
        charts / "forecast-stability.svg",
        charts / "daily-equity.svg",
        charts / "ai-uplift.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "replays.csv", replay_rows)
    _write_csv(output_dir / "monthly.csv", monthly_rows)
    _write_csv(output_dir / "symbols.csv", symbol_rows)
    _write_csv(output_dir / "diagnostics.csv", report["forecast_diagnostics"])
    _write_csv(output_dir / "models.csv", model_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(output_dir / "ai-uplift.csv", ai_rows)
    _write_csv(output_dir / "daily-equity.csv", daily_rows)
    _write_csv(output_dir / "sources.csv", source_rows)
    _write_csv(output_dir / "progress.csv", progress_rows, fields=progress_fields)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "stateful-economics.svg", _economics_svg(report))
    _write_text(charts / "forecast-stability.svg", _forecast_svg(report))
    _write_text(charts / "daily-equity.svg", _equity_svg(daily_rows))
    _write_text(charts / "ai-uplift.svg", _ai_uplift_svg(report))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress_rows))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "stateful_turnover_ai_factor_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "ai_factor_audit_sha256": report["ai_factor_audit_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "hourly_dataset_rows": report["dataset"]["rows"],
        "evaluation_hours": report["stateful_replays"][0]["hours"],
        "gpu_model_artifact_count": len(model_rows),
        "position_ledger_rows": report["position_ledger_rows"],
        "stateful_candidate_count": len(STATEFUL_CANDIDATES),
        "stateful_gate_pass_count": 0,
        "ai_uplift_gate_passed": False,
        "selection_contaminated": True,
        "development_only": True,
        "trading_authority": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "leverage_applied": False,
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
        default=research / "round-043-stateful-turnover-ai-factor-design.json",
    )
    parser.add_argument(
        "--ai-factor-audit",
        type=Path,
        default=research / "round-043-ai-factor-audit.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-043-stateful-turnover-ai-factor-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        evidence_root=arguments.evidence_root,
        design_path=arguments.design,
        audit_path=arguments.ai_factor_audit,
        binding_path=arguments.binding,
        prior_progress_path=arguments.prior_progress,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
