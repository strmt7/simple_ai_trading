"""Publish verified Round 49 action-hurdle evidence and deterministic charts."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import html
import json
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
from tools.run_action_hurdle_tcn_viability import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


ROUND = 49
PUBLICATION_SCHEMA = "cost-aware-action-hurdle-tcn-publication-v1"
CANDIDATES = ("direct_action_mean_tcn", "hurdle_action_value_tcn")
LABELS = {
    "direct_action_mean_tcn": "Direct mean control",
    "hurdle_action_value_tcn": "Conditional gain/loss hurdle",
}
COLORS = {
    "direct_action_mean_tcn": "#2563a6",
    "hurdle_action_value_tcn": "#0f766e",
}
SIDES = ("short", "long")
EXPECTED_DATASET_SHA256 = (
    "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
)
EXPECTED_PREDECESSOR_SHA256 = (
    "6969a3134049a326024939d5f9c46a99c37a4932e4a1f146a542a77427bba92b"
)


def _candidate(report: Mapping[str, object], candidate_id: str) -> Mapping[str, object]:
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
    design = _read_object(design_path, "Round 49 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 49 design")
    binding = _read_object(binding_path, "Round 49 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 49 binding")
    report = _read_object(evidence_root / "report.json", "Round 49 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 49 report"
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
        raise ValueError("Round 49 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    backend = report.get("backend")
    results = report.get("candidate_results")
    ai = report.get("ai_decision")
    mechanism = report.get("mechanism_ablation_gate")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("dataset_sha256") != EXPECTED_DATASET_SHA256
        or dataset.get("predecessor_dataset_sha256") != EXPECTED_PREDECESSOR_SHA256
        or dataset.get("timestamps") != 366_035
        or dataset.get("rows") != 1_098_105
        or dataset.get("feature_count") != 71
        or dataset.get("symbols") != ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        or dataset.get("horizons_minutes") != [15, 30]
        or dataset.get("persistent_feature_copy_created") is not False
        or not isinstance(backend, Mapping)
        or backend.get("backend_kind") != "directml"
        or backend.get("backend_device") != "privateuseone:0"
        or backend.get("cpu_fallback_warnings") != 0
        or backend.get("warning_count") != 0
        or not isinstance(results, list)
        or {item.get("candidate_id") for item in results} != set(CANDIDATES)
        or any(len(item.get("artifacts", [])) != 3 for item in results)
        or any(
            item.get("combined_quality_gate_passed") is not False for item in results
        )
        or any(
            item.get("economic_gate", {}).get("passed") is not False for item in results
        )
        or not isinstance(mechanism, Mapping)
        or mechanism.get("passed") is not False
        or not isinstance(ai, Mapping)
        or ai.get("executed") is not False
        or ai.get("paired_veto_only_ablation_eligible") is not False
        or ai.get("ai_uplift_claim") is not False
        or report.get("selection_confirmation_accessed") is not False
        or report.get("terminal_2026_accessed") is not False
        or report.get("profitability_claim") is not False
        or report.get("ai_uplift_claim") is not False
        or report.get("trading_authority") is not False
        or report.get("leverage_applied") is not False
    ):
        raise ValueError("Round 49 model, data, or governance evidence drifted")
    for candidate in results:
        if candidate["numerical_quality_gate"]["passed"] is not True:
            raise ValueError("Round 49 numerical quality evidence drifted")
        for artifact in candidate["artifacts"]:
            if (
                artifact.get("backend_kind") != "directml"
                or max(
                    float(artifact[field])
                    for field in (
                        "reload_max_abs_logit_error",
                        "reload_max_abs_primary_error",
                        "reload_max_abs_secondary_error",
                        "reload_max_abs_auxiliary_error",
                    )
                )
                != 0.0
            ):
                raise ValueError("Round 49 model reload evidence drifted")
    declared = report.get("external_artifacts")
    if not isinstance(declared, list) or len(declared) != 12:
        raise ValueError("Round 49 external artifact declaration is incomplete")
    for item in declared:
        path = Path(str(item["path"]))
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or _file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"Round 49 external artifact drifted: {path}")
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
    probability: list[dict[str, object]] = []
    expected_net: list[dict[str, object]] = []
    severity: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    training: list[dict[str, object]] = []
    models: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    replays: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    symbols: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        diagnostics = candidate["diagnostics"]
        probability.extend(dict(row) for row in diagnostics["probability"])
        expected_net.extend(dict(row) for row in diagnostics["expected_net"])
        severity.extend(dict(row) for row in diagnostics["severity"])
        stability.extend(dict(row) for row in diagnostics["seed_stability"])
        training.extend(dict(row) for row in candidate["training_history"])
        models.extend(dict(row) for row in candidate["artifacts"])
        gates.append(
            {
                "candidate_id": candidate_id,
                "numerical_quality_gate_passed": candidate["numerical_quality_gate"][
                    "passed"
                ],
                "action_quality_gate_passed": diagnostics["action_quality_gate"][
                    "passed"
                ],
                "action_quality_gate_reasons": diagnostics["action_quality_gate"][
                    "reasons"
                ],
                "mechanism_gate_applicable": (
                    candidate["mechanism_ablation_gate"] is not None
                ),
                "mechanism_gate_passed": (
                    candidate["mechanism_ablation_gate"]["passed"]
                    if candidate["mechanism_ablation_gate"] is not None
                    else None
                ),
                "combined_quality_gate_passed": candidate[
                    "combined_quality_gate_passed"
                ],
                "economic_gate_passed": candidate["economic_gate"]["passed"],
                "economic_gate_reasons": candidate["economic_gate"]["reasons"],
            }
        )
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
    mechanism = [{**report["mechanism_ablation_gate"]}]
    return {
        "probability": probability,
        "expected-net": expected_net,
        "severity": severity,
        "seed-stability": stability,
        "training": training,
        "models": models,
        "roles": roles,
        "target-geometry": [dict(row) for row in report["dataset"]["target_geometry"]],
        "trades": trades,
        "replays": replays,
        "monthly": monthly,
        "symbols": symbols,
        "daily-equity": daily,
        "gates": gates,
        "mechanism": mechanism,
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
        raise ValueError("Round 49 prior progress history is invalid")
    best = _candidate(report, "hurdle_action_value_tcn")
    pooled_probability = [
        row for row in best["diagnostics"]["probability"] if row["scope"] == "pooled"
    ]
    pooled_action = [
        row for row in best["diagnostics"]["expected_net"] if row["scope"] == "pooled"
    ]
    metrics = best["base"]["metrics"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "15-minute cost-aware action-hurdle TCN",
            "periods": "2022-01-01..2025-06-30 roles; eval 2025-01-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "900;1800 auxiliary",
            "feature_set": "71 causal features; 5-minute large-kernel TCN",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "direction_auc": max(float(item["roc_auc"]) for item in pooled_probability),
            "spearman_ic": max(
                float(item["expected_net_spearman"]) for item in pooled_action
            ),
            "selected_signals": metrics["trades"],
            "executable_trades": metrics["trades"],
            "mean_net_bps": metrics["mean_five_minute_portfolio_bps"],
            "status": "rejected",
            "source_file": "verified Round 49 action-hurdle report; positive point estimate is not an edge claim",
            "best_policy_trades": metrics["trades"],
            "best_policy_total_net_bps": 10_000
            * float(metrics["total_net_return_fraction"]),
            "best_policy_mean_net_bps": metrics["mean_five_minute_portfolio_bps"],
            "best_policy_max_drawdown_bps": 10_000
            * float(metrics["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": metrics["profit_factor"],
            "best_model_id": "hurdle_action_value_tcn_descriptive_only",
            "ensemble_models": 6,
            "development_consumed": True,
            "architecture_gates_passed": 0,
            "architecture_gate_count": 3,
        }
    )
    rows.append(row)
    return fields, rows


def _bounds(
    values: Sequence[float], *, reference: float = 0.0, padding: float = 0.15
) -> tuple[float, float]:
    low = min(*values, reference)
    high = max(*values, reference)
    span = max(high - low, 0.01)
    return low - padding * span, high + padding * span


def _legend(lines: list[str], labels: Sequence[tuple[str, str]]) -> None:
    start = 930
    for index, (label, color) in enumerate(labels):
        x = start + index * 240
        lines.append(
            f'<rect x="{x}" y="42" width="18" height="18" fill="{color}"/>'
            f'<text x="{x + 28}" y="56" class="note">{html.escape(label)}</text>'
        )


def _bar_panel(
    lines: list[str],
    *,
    values: Sequence[tuple[str, str, float]],
    top: float,
    left: float,
    width: float,
    height: float,
    low: float,
    high: float,
    label: str,
    reference: float = 0.0,
    value_format: str = ".3f",
) -> None:
    lines.append(
        f'<text x="{left}" y="{top - 18}" class="label">{html.escape(label)}</text>'
    )

    def y(value: float) -> float:
        return top + height * (high - value) / (high - low)

    for tick in (low, (low + high) / 2.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + width}" y2="{py:.1f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:{value_format}}</text>'
        )
    if low <= reference <= high:
        py = y(reference)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + width}" y2="{py:.1f}" class="zero"/>'
        )
    slot = width / len(values)
    baseline = y(reference if low <= reference <= high else low)
    for index, (name, color, value) in enumerate(values):
        x = left + index * slot + slot * 0.18
        bar_width = slot * 0.64
        py = y(value)
        rect_y = min(py, baseline)
        rect_height = max(abs(baseline - py), 1.0)
        lines.append(
            f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" height="{rect_height:.1f}" fill="{color}"/>'
        )
        label_y = py - 8 if value >= reference else py + 18
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{value:{value_format}}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{top + height + 22:.1f}" text-anchor="middle" class="axis">{html.escape(name)}</text>'
        )


def _forecast_svg(report: Mapping[str, object]) -> str:
    pooled = {
        (candidate_id, row["side"]): row
        for candidate_id in CANDIDATES
        for row in _candidate(report, candidate_id)["diagnostics"]["probability"]
        if row["scope"] == "pooled"
    }
    entries = [
        (
            f"{side.title()} / {'control' if candidate == CANDIDATES[0] else 'hurdle'}",
            COLORS[candidate],
            candidate,
            side,
        )
        for side in SIDES
        for candidate in CANDIDATES
    ]
    width, height = 1500, 760
    lines = _svg_start(
        width,
        height,
        "Profit-probability quality remained stable",
        "Pooled 15-minute evaluation metrics on consumed 2025-H1; classification quality is not an after-cost edge.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    _bar_panel(
        lines,
        values=[
            (label, color, float(pooled[(candidate, side)]["roc_auc"]))
            for label, color, candidate, side in entries
        ],
        top=145,
        left=130,
        width=1300,
        height=205,
        low=0.55,
        high=0.65,
        reference=0.55,
        label="ROC AUC (frozen minimum 0.55)",
    )
    _bar_panel(
        lines,
        values=[
            (label, color, float(pooled[(candidate, side)]["log_loss_skill"]))
            for label, color, candidate, side in entries
        ],
        top=470,
        left=130,
        width=1300,
        height=180,
        low=0.0,
        high=0.05,
        label="Binary log-loss skill versus training prevalence",
        value_format=".4f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _action_value_svg(report: Mapping[str, object]) -> str:
    pooled = {
        (candidate_id, row["side"]): row
        for candidate_id in CANDIDATES
        for row in _candidate(report, candidate_id)["diagnostics"]["expected_net"]
        if row["scope"] == "pooled"
    }
    entries = [
        (
            f"{side.title()} / {'control' if candidate == CANDIDATES[0] else 'hurdle'}",
            COLORS[candidate],
            candidate,
            side,
        )
        for side in SIDES
        for candidate in CANDIDATES
    ]
    spearman = [
        float(pooled[(candidate, side)]["expected_net_spearman"])
        for _, _, candidate, side in entries
    ]
    mse_skill = [
        float(pooled[(candidate, side)]["expected_net_mse_skill"])
        for _, _, candidate, side in entries
    ]
    spearman_low, spearman_high = _bounds(spearman, reference=0.03)
    mse_low, mse_high = _bounds(mse_skill)
    width, height = 1500, 760
    lines = _svg_start(
        width,
        height,
        "Expected-net prediction failed the action-quality gate",
        "Both architectures preserved probability AUC but failed to rank or improve 15-minute net utility.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    _bar_panel(
        lines,
        values=[
            (label, color, float(pooled[(candidate, side)]["expected_net_spearman"]))
            for label, color, candidate, side in entries
        ],
        top=145,
        left=130,
        width=1300,
        height=205,
        low=spearman_low,
        high=spearman_high,
        reference=0.03,
        label="Expected-net Spearman (frozen minimum 0.03)",
        value_format=".4f",
    )
    _bar_panel(
        lines,
        values=[
            (label, color, float(pooled[(candidate, side)]["expected_net_mse_skill"]))
            for label, color, candidate, side in entries
        ],
        top=470,
        left=130,
        width=1300,
        height=180,
        low=mse_low,
        high=mse_high,
        label="Expected-net MSE skill versus training side mean",
        value_format=".4f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _severity_svg(report: Mapping[str, object]) -> str:
    rows = [
        row
        for row in _candidate(report, "hurdle_action_value_tcn")["diagnostics"][
            "severity"
        ]
        if row["scope"] == "pooled"
    ]
    values: list[tuple[str, str, float]] = []
    for row in rows:
        side = str(row["side"]).title()
        values.extend(
            (
                (
                    f"{side} gain",
                    "#0f766e",
                    float(row["conditional_gain_gamma_score_skill"]),
                ),
                (
                    f"{side} loss",
                    "#b45309",
                    float(row["conditional_loss_gamma_score_skill"]),
                ),
            )
        )
    width, height = 1500, 520
    lines = _svg_start(
        width,
        height,
        "Conditional severity improved, but did not combine into edge",
        "Gamma mean-score skill versus training conditional means; positive component skill did not yield expected-net rank skill.",
    )
    _legend(lines, [("conditional gain", "#0f766e"), ("conditional loss", "#b45309")])
    _bar_panel(
        lines,
        values=values,
        top=150,
        left=130,
        width=1300,
        height=245,
        low=0.0,
        high=0.04,
        label="Conditional Gamma mean-score skill",
        value_format=".4f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _seed_stability_svg(report: Mapping[str, object]) -> str:
    values: list[tuple[str, str, float]] = []
    for candidate_id in CANDIDATES:
        gate = _candidate(report, candidate_id)["diagnostics"]["action_quality_gate"]
        suffix = "control" if candidate_id == CANDIDATES[0] else "hurdle"
        values.extend(
            (
                (
                    f"Probability / {suffix}",
                    COLORS[candidate_id],
                    float(gate["minimum_pairwise_seed_probability_spearman"]),
                ),
                (
                    f"Net value / {suffix}",
                    COLORS[candidate_id],
                    float(gate["minimum_pairwise_seed_expected_net_spearman"]),
                ),
            )
        )
    width, height = 1500, 520
    lines = _svg_start(
        width,
        height,
        "Probability was stable; action value was not",
        "Minimum pairwise seed Spearman on consumed 2025-H1; frozen minimum is 0.50 for both outputs.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    _bar_panel(
        lines,
        values=values,
        top=150,
        left=130,
        width=1300,
        height=245,
        low=0.0,
        high=1.0,
        reference=0.5,
        label="Minimum pairwise seed Spearman",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _policy_svg(report: Mapping[str, object]) -> str:
    returns: list[tuple[str, str, float]] = []
    drawdowns: list[tuple[str, str, float]] = []
    for candidate_id in CANDIDATES:
        candidate = _candidate(report, candidate_id)
        suffix = "control" if candidate_id == CANDIDATES[0] else "hurdle"
        for scenario, color in (("base", COLORS[candidate_id]), ("stress", "#b45309")):
            metrics = candidate[scenario]["metrics"]
            returns.append(
                (
                    f"{scenario.title()} / {suffix}",
                    color,
                    100.0 * float(metrics["total_net_return_fraction"]),
                )
            )
            drawdowns.append(
                (
                    f"{scenario.title()} / {suffix}",
                    color,
                    100.0 * float(metrics["maximum_drawdown_fraction"]),
                )
            )
    return_low, return_high = _bounds([item[2] for item in returns])
    drawdown_high = max(8.0, max(item[2] for item in drawdowns) * 1.15)
    width, height = 1500, 760
    lines = _svg_start(
        width,
        height,
        "Positive point estimate failed economic validation",
        "Exact fixed trade ledger at 12 bps base and 16 bps stress; no trade reselection, leverage, or fee improvement.",
    )
    _legend(lines, [("candidate base charge", "#0f766e"), ("16 bps stress", "#b45309")])
    _bar_panel(
        lines,
        values=returns,
        top=145,
        left=130,
        width=1300,
        height=205,
        low=return_low,
        high=return_high,
        label="Total net return (%)",
        value_format="+.2f",
    )
    _bar_panel(
        lines,
        values=drawdowns,
        top=470,
        left=130,
        width=1300,
        height=180,
        low=0.0,
        high=drawdown_high,
        label="Maximum drawdown (%)",
        value_format=".2f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _monthly_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["candidate_id"] == "hurdle_action_value_tcn"]
    months = sorted({str(row["month"]) for row in selected})
    values = [100.0 * float(row["total_net_return_fraction"]) for row in selected]
    low, high = _bounds(values)
    width, height = 1500, 610
    left, right, top, chart_height = 120, 70, 145, 330
    chart_width = width - left - right

    def x(index: int) -> float:
        return left + chart_width * index / max(len(months) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Hurdle economics were concentrated early in 2025",
        "Monthly compounded return from the same 165 trades; January-April contain 157 trades and both losing months.",
    )
    _legend(lines, [("12 bps base", "#0f766e"), ("16 bps stress", "#b45309")])
    for tick in (low, 0.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{("zero" if tick == 0.0 else "grid")}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    by_scenario = {
        scenario: {
            str(row["month"]): 100.0 * float(row["total_net_return_fraction"])
            for row in selected
            if row["scenario"] == scenario
        }
        for scenario in ("base", "stress")
    }
    for scenario, color in (("base", "#0f766e"), ("stress", "#b45309")):
        points = [
            (x(index), y(by_scenario[scenario][month]))
            for index, month in enumerate(months)
        ]
        lines.append(
            '<polyline points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + f'" fill="none" stroke="{color}" stroke-width="4"/>'
        )
        for index, (px, py) in enumerate(points):
            value = by_scenario[scenario][months[index]]
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}"/>')
            if scenario == "base":
                lines.append(
                    f'<text x="{px:.1f}" y="{py - 12:.1f}" text-anchor="middle" class="value">{value:+.2f}%</text>'
                )
    trade_counts = {
        str(row["month"]): int(row["trades"])
        for row in selected
        if row["scenario"] == "base"
    }
    for index, month in enumerate(months):
        px = x(index)
        lines.append(
            f'<text x="{px:.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{month}</text>'
        )
        lines.append(
            f'<text x="{px:.1f}" y="{top + chart_height + 48}" text-anchor="middle" class="note">{trade_counts[month]} trades</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _equity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["candidate_id"] == "hurdle_action_value_tcn"]
    dates = sorted({str(row["date"]) for row in selected})
    by_scenario = {
        scenario: {
            str(row["date"]): float(row["equity"])
            for row in selected
            if row["scenario"] == scenario
        }
        for scenario in ("base", "stress")
    }
    all_values = [
        value for mapping in by_scenario.values() for value in mapping.values()
    ]
    low, high = _bounds(all_values, reference=1.0, padding=0.08)
    width, height = 1500, 610
    left, right, top, chart_height = 120, 70, 145, 330
    chart_width = width - left - right

    def x(index: int) -> float:
        return left + chart_width * index / max(len(dates) - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Dated hurdle equity path",
        "One-third unlevered sleeves and exact close-time booking; the path is consumed development evidence, not deployable performance.",
    )
    _legend(lines, [("12 bps base", "#0f766e"), ("16 bps stress", "#b45309")])
    for tick in (low, 1.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{("zero" if tick == 1.0 else "grid")}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
        )
    for scenario, color in (("base", "#0f766e"), ("stress", "#b45309")):
        points = [
            (x(index), y(by_scenario[scenario][date]))
            for index, date in enumerate(dates)
        ]
        lines.append(
            '<polyline points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + f'" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    tick_indices = sorted({0, len(dates) // 3, 2 * len(dates) // 3, len(dates) - 1})
    for index in tick_indices:
        lines.append(
            f'<text x="{x(index):.1f}" y="{top + chart_height + 30}" text-anchor="middle" class="axis">{dates[index]}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _training_svg(report: Mapping[str, object]) -> str:
    histories = {
        candidate_id: _candidate(report, candidate_id)["training_history"]
        for candidate_id in CANDIDATES
    }
    values = [
        float(row["early_stop_composite"])
        for history in histories.values()
        for row in history
    ]
    low, high = _bounds(values, reference=min(values), padding=0.12)
    width, height = 1500, 610
    left, right, top, chart_height = 120, 70, 145, 330
    chart_width = width - left - right
    maximum_epoch = max(
        int(row["epoch"]) for history in histories.values() for row in history
    )

    def x(epoch: int) -> float:
        return left + chart_width * (epoch - 1) / max(maximum_epoch - 1, 1)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Early stopping rejected later overfit",
        "Mean three-seed early-stop composite by epoch; selected epochs were 2 for direct control and 4 for hurdle.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    for tick in (low, (low + high) / 2.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="grid"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
        )
    for candidate_id in CANDIDATES:
        points = [
            (x(int(row["epoch"])), y(float(row["early_stop_composite"])))
            for row in histories[candidate_id]
        ]
        lines.append(
            '<polyline points="'
            + " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            + f'" fill="none" stroke="{COLORS[candidate_id]}" stroke-width="4"/>'
        )
        best_epoch = int(histories[candidate_id][-1]["best_epoch"])
        best = next(
            row for row in histories[candidate_id] if int(row["epoch"]) == best_epoch
        )
        lines.append(
            f'<circle cx="{x(best_epoch):.1f}" cy="{y(float(best["early_stop_composite"])):.1f}" r="8" fill="{COLORS[candidate_id]}" stroke="#ffffff" stroke-width="3"/>'
        )
    for epoch in range(1, maximum_epoch + 1):
        if epoch == 1 or epoch == maximum_epoch or epoch % 2 == 0:
            lines.append(
                f'<text x="{x(epoch):.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{epoch}</text>'
            )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _failure_analysis(
    report: Mapping[str, object],
    source_report_sha: str,
    binding_sha: str,
    evidence_root: Path,
) -> dict[str, object]:
    direct = _candidate(report, "direct_action_mean_tcn")
    hurdle = _candidate(report, "hurdle_action_value_tcn")
    direct_action = {
        row["side"]: row
        for row in direct["diagnostics"]["expected_net"]
        if row["scope"] == "pooled"
    }
    hurdle_action = {
        row["side"]: row
        for row in hurdle["diagnostics"]["expected_net"]
        if row["scope"] == "pooled"
    }
    severity = {
        row["side"]: row
        for row in hurdle["diagnostics"]["severity"]
        if row["scope"] == "pooled"
    }
    analysis: dict[str, object] = {
        "schema_version": "round-049-action-hurdle-failure-analysis-v1",
        "round": ROUND,
        "status": "rejected",
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "binding_sha256": binding_sha,
        "dataset_sha256": report["dataset"]["dataset_sha256"],
        "predecessor_dataset_sha256": report["dataset"]["predecessor_dataset_sha256"],
        "observed_result": {
            "direct_control": {
                "trades": direct["base"]["metrics"]["trades"],
                "short_expected_net_spearman": direct_action["short"][
                    "expected_net_spearman"
                ],
                "long_expected_net_spearman": direct_action["long"][
                    "expected_net_spearman"
                ],
                "action_quality_gate_reasons": direct["diagnostics"][
                    "action_quality_gate"
                ]["reasons"],
            },
            "hurdle": {
                "trades": hurdle["base"]["metrics"]["trades"],
                "active_days": hurdle["base"]["metrics"]["active_days"],
                "base_total_net_return_fraction": hurdle["base"]["metrics"][
                    "total_net_return_fraction"
                ],
                "stress_total_net_return_fraction": hurdle["stress"]["metrics"][
                    "total_net_return_fraction"
                ],
                "base_maximum_drawdown_fraction": hurdle["base"]["metrics"][
                    "maximum_drawdown_fraction"
                ],
                "base_profit_factor": hurdle["base"]["metrics"]["profit_factor"],
                "stress_profit_factor": hurdle["stress"]["metrics"]["profit_factor"],
                "stress_bootstrap_lower_mean_five_minute_bps": hurdle["stress"][
                    "metrics"
                ]["bootstrap_mean_five_minute_portfolio_bps"]["lower_bps"],
                "trades_by_symbol": hurdle["base"]["metrics"]["trades_by_symbol"],
                "symbol_net_bps": hurdle["base"]["metrics"]["symbol_net_bps"],
                "short_expected_net_spearman": hurdle_action["short"][
                    "expected_net_spearman"
                ],
                "long_expected_net_spearman": hurdle_action["long"][
                    "expected_net_spearman"
                ],
                "short_gain_gamma_skill": severity["short"][
                    "conditional_gain_gamma_score_skill"
                ],
                "short_loss_gamma_skill": severity["short"][
                    "conditional_loss_gamma_score_skill"
                ],
                "long_gain_gamma_skill": severity["long"][
                    "conditional_gain_gamma_score_skill"
                ],
                "long_loss_gamma_skill": severity["long"][
                    "conditional_loss_gamma_score_skill"
                ],
                "action_quality_gate_reasons": hurdle["diagnostics"][
                    "action_quality_gate"
                ]["reasons"],
                "economic_gate_reasons": hurdle["economic_gate"]["reasons"],
            },
            "mechanism_ablation": report["mechanism_ablation_gate"],
        },
        "critical_interpretation": [
            "The positive 12 bps point estimate is not evidence of a repeatable edge because expected-net Spearman and MSE skill failed for both sides, seed action-value stability failed, and the familywise stress bootstrap lower bound was negative.",
            "The hurdle heads learned conditional gain and loss severity better than static training means, but their exact probability-weighted combination did not improve expected-net ranking over the direct control.",
            "The ledger was sparse and unstable: 165 trades occurred on only 35 days, 130 trades occurred in January-February, and ETHUSDT supplied all positive symbol net PnL while BTCUSDT and SOLUSDT lost.",
            "The 16 bps stress ledger retained the same timestamps and sides but reduced total return to 0.30% and profit factor to 1.018, which is too close to break-even for execution uncertainty.",
            "No threshold, seed, side, month, symbol, fee, or leverage choice may be selected from this consumed evaluation result.",
        ],
        "next_model_requirements": [
            "Retain the calibrated probability trunk because pooled 15-minute AUC and proper probability scores remained useful and stable.",
            "Replace independently accurate severity components with an objective that must add stable out-of-sample expected-net ordering across symbols and months, while preserving proper conditional-distribution scoring.",
            "Measure and penalize symbol and month concentration using only training and early-stop roles; do not tune a trade threshold on consumed 2025-H1 outcomes.",
            "Require seed-stable positive action value and exact fixed-ledger base/stress replay with no forced activity, leverage, or fee reduction.",
            "Reject isolated positive returns unless probability, action-value, mechanism, bootstrap, activity, drawdown, profit-factor, and concentration gates all pass together.",
            "Keep local language-model AI disabled until a deterministic candidate clears every gate; AI remains a paired asynchronous veto, never an order generator or loss-recovery mechanism.",
        ],
        "selection_contaminated": True,
        "development_only": True,
        "profitability_claim": False,
        "trading_authority": False,
        "leverage_applied": False,
        "ai_uplift_claim": False,
        "generated_at_utc": report["generated_at_utc"],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    direct = _candidate(report, "direct_action_mean_tcn")
    hurdle = _candidate(report, "hurdle_action_value_tcn")
    rows: list[str] = []
    for candidate in (direct, hurdle):
        candidate_id = str(candidate["candidate_id"])
        base = candidate["base"]["metrics"]
        stress = candidate["stress"]["metrics"]
        probability = [
            row
            for row in candidate["diagnostics"]["probability"]
            if row["scope"] == "pooled"
        ]
        profit_factor = (
            f"{float(base['profit_factor']):.3f}"
            if base["profit_factor"] is not None
            else "n/a"
        )
        rows.append(
            f"| {LABELS[candidate_id]} | {max(float(row['roc_auc']) for row in probability):.3f} | "
            f"{base['trades']} | {100 * float(base['total_net_return_fraction']):+.2f}% | "
            f"{100 * float(stress['total_net_return_fraction']):+.2f}% | "
            f"{100 * float(base['maximum_drawdown_fraction']):.2f}% | {profit_factor} | "
            f"{candidate['numerical_quality_gate']['passed']}/{candidate['diagnostics']['action_quality_gate']['passed']}/{candidate['economic_gate']['passed']} |"
        )
    runtime = report["runtime_evidence"]
    return f"""# Round 49: Cost-Aware Action-Hurdle TCN

> **Beta research warning:** neither candidate is approved for testnet, live day trading, leverage, or autonomous execution. The 2025-H1 result is consumed development evidence.

Round 49 trained six causal large-kernel TCNs on verified Binance BTCUSDT, ETHUSDT, and SOLUSDT data. Explicit conditional gain/loss modeling created a positive point estimate, but expected-net prediction, temporal breadth, diversification, and stress confidence failed. The candidate was rejected.

![Profit-probability quality](charts/forecast-quality.svg)

![Expected-net quality](charts/action-value-quality.svg)

![Conditional severity](charts/severity-quality.svg)

| Candidate | Best 15m AUC | Trades | Base return | Stress return | Base drawdown | Profit factor | Numerical/action/economic gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

The hurdle ledger made 165 trades on 35 days. Its base return was `+2.53%`, but the 16 bps stress return fell to `+0.30%`; the familywise bootstrap lower bound was negative. ETHUSDT supplied `+1,457.49` net bps while BTCUSDT and SOLUSDT lost, and 130 trades occurred in January-February. This is not a profitability claim.

![Policy economics](charts/policy-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Dated equity](charts/daily-equity.svg)

![Seed stability](charts/seed-stability.svg)

![Training dynamics](charts/training-dynamics.svg)

![Research progress](charts/research-progress.svg)

DirectML completed in `{float(runtime["elapsed_seconds"]):.1f}s`, peaked at `{float(runtime["memory"]["peak_working_set_bytes"]) / 2**30:.2f} GiB` working set, recorded zero CPU fallbacks, and reloaded all six models and prediction arrays exactly. AI was withheld because no deterministic candidate passed.

Data: [probability](probability.csv) | [expected net](expected-net.csv) | [conditional severity](severity.csv) | [seed stability](seed-stability.csv) | [training](training.csv) | [models](models.csv) | [roles](roles.csv) | [target geometry](target-geometry.csv) | [trades](trades.csv) | [replays](replays.csv) | [monthly economics](monthly.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [gates](gates.csv) | [mechanism](mechanism.csv) | [source lineage](sources.csv) | [progress](progress.csv) | [failure analysis](../round-049-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
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
        charts / "action-value-quality.svg",
        charts / "severity-quality.svg",
        charts / "seed-stability.svg",
        charts / "training-dynamics.svg",
        charts / "policy-economics.svg",
        charts / "monthly-economics.svg",
        charts / "daily-equity.svg",
        charts / "research-progress.svg",
    }
    expected = (
        table_paths
        | chart_paths
        | {
            output_dir / "README.md",
            output_dir / "screen.json",
            output_dir / "report.json",
        }
    )
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
    _write_text(charts / "action-value-quality.svg", _action_value_svg(report))
    _write_text(charts / "severity-quality.svg", _severity_svg(report))
    _write_text(charts / "seed-stability.svg", _seed_stability_svg(report))
    _write_text(charts / "training-dynamics.svg", _training_svg(report))
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
        "artifact_class": "cost_aware_action_hurdle_tcn_graph_data",
        "round": ROUND,
        "status": "quality_or_economic_gate_rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "dataset_sha256": report["dataset"]["dataset_sha256"],
        "predecessor_dataset_sha256": report["dataset"]["predecessor_dataset_sha256"],
        "dataset_rows": report["dataset"]["rows"],
        "evaluation_timestamps": report["dataset"]["role_timestamp_counts"][
            "viability"
        ],
        "directml_model_artifact_count": 6,
        "candidate_numerical_gate_pass_count": 2,
        "candidate_action_gate_pass_count": 0,
        "candidate_economic_gate_pass_count": 0,
        "mechanism_ablation_passed": report["mechanism_ablation_gate"]["passed"],
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
        default=research / "round-049-cost-aware-action-hurdle-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-049-cost-aware-action-hurdle-tcn-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-path",
        type=Path,
        default=research / "round-049-failure-analysis.json",
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
