"""Publish verified Round 50 path-bounded model evidence and charts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import UTC, datetime
import hashlib
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
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)


ROUND = 50
DESIGN_SCHEMA = "path-bounded-competing-risk-tcn-design-v1"
BINDING_SCHEMA = "round-050-path-bounded-competing-risk-execution-binding-v1"
REPORT_SCHEMA = "path-bounded-competing-risk-tcn-report-v1"
PUBLICATION_SCHEMA = "path-bounded-competing-risk-tcn-publication-v1"
CANDIDATES = ("direct_barrier_mean_tcn", "competing_risk_barrier_tcn")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SIDES = ("short", "long")
LABELS = {
    "direct_barrier_mean_tcn": "Direct mean control",
    "competing_risk_barrier_tcn": "Competing-risk model",
}
SHORT_LABELS = {
    "direct_barrier_mean_tcn": "Direct",
    "competing_risk_barrier_tcn": "Path",
}
COLORS = {
    "direct_barrier_mean_tcn": "#2563a6",
    "competing_risk_barrier_tcn": "#0f766e",
}
EXPECTED_DATASET_SHA256 = (
    "31c7713339cff9ad12f3bae02475743d09b2248bfc1b85e02e1f3306a699e774"
)
EXPECTED_PREDECESSOR_SHA256 = (
    "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
)
EXPECTED_SOURCE_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


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
    ai_benchmark_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design = _read_object(design_path, "Round 50 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 50 design")
    binding = _read_object(binding_path, "Round 50 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 50 binding")
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, "Round 50 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 50 report"
    )
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or binding.get("design_sha256") != design_sha
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
    ):
        raise ValueError("Round 50 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    backend = report.get("backend")
    claims = report.get("claims")
    diagnostics = report.get("diagnostics")
    gates = report.get("economic_gates")
    artifacts = report.get("artifacts")
    mechanism = report.get("mechanism_gate")
    leverage = report.get("leverage_sensitivity")
    ai = report.get("ai")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("barrier_dataset_sha256") != EXPECTED_DATASET_SHA256
        or dataset.get("predecessor_dataset_sha256") != EXPECTED_PREDECESSOR_SHA256
        or dataset.get("source_resolution_seconds") != 60
        or dataset.get("decision_interval_seconds") != 300
        or dataset.get("rows") != 1_098_105
        or dataset.get("timestamps") != 366_035
        or dataset.get("symbols") != list(SYMBOLS)
        or dataset.get("synthetic_rows") != 0
        or dataset.get("selection_confirmation_or_terminal_rows_read") is not False
        or report.get("source_certificate_sha256") != EXPECTED_SOURCE_SHA256
        or not isinstance(backend, Mapping)
        or backend.get("backend_kind") != "directml"
        or backend.get("backend_device") != "privateuseone:0"
        or backend.get("cpu_fallback_warnings") != 0
        or backend.get("warning_count") != 0
        or not isinstance(claims, Mapping)
        or claims.get("selection_contaminated") is not True
        or claims.get("beta_research_only") is not True
        or claims.get("profitability_claim") is not False
        or claims.get("trading_authority") is not False
        or not isinstance(diagnostics, Mapping)
        or set(diagnostics) != set(CANDIDATES)
        or not isinstance(gates, Mapping)
        or set(gates) != set(CANDIDATES)
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != set(CANDIDATES)
        or not isinstance(mechanism, Mapping)
        or mechanism.get("passed") is not False
        or not isinstance(leverage, Mapping)
        or not isinstance(ai, Mapping)
        or ai.get("paired_uplift_run") is not False
        or ai.get("trading_authority") is not False
    ):
        raise ValueError("Round 50 model, data, or governance evidence drifted")
    if any(
        diagnostics[candidate]["quality_gate"]["passed"] is not False
        or gates[candidate]["passed"] is not False
        or leverage[candidate]["run"] is not False
        for candidate in CANDIDATES
    ):
        raise ValueError("Round 50 failed gates were not preserved")
    for candidate in CANDIDATES:
        candidate_artifacts = artifacts[candidate]
        if not isinstance(candidate_artifacts, list) or len(candidate_artifacts) != 3:
            raise ValueError(f"Round 50 {candidate} artifact set is incomplete")
        if {item.get("seed") for item in candidate_artifacts} != {5001, 5002, 5003}:
            raise ValueError(f"Round 50 {candidate} seed set is incomplete")
        for item in candidate_artifacts:
            model_path = Path(str(item.get("model_path") or ""))
            prediction_path = Path(str(item.get("prediction_path") or ""))
            if (
                not model_path.is_file()
                or model_path.stat().st_size != int(item.get("model_bytes") or -1)
                or _file_sha256(model_path) != item.get("model_sha256")
                or not prediction_path.is_file()
                or prediction_path.stat().st_size
                != int(item.get("prediction_bytes") or -1)
                or _file_sha256(prediction_path) != item.get("prediction_sha256")
            ):
                raise ValueError(f"Round 50 external artifact drifted: {item}")
            reload_fields = [
                key for key in item if str(key).startswith("reload_max_abs_")
            ]
            runtime_fields = [
                key for key in item if str(key).startswith("runtime_repeat_max_abs_")
            ]
            if (
                not reload_fields
                or any(float(item[key]) != 0.0 for key in reload_fields)
                or not runtime_fields
                or any(float(item[key]) > 1e-4 for key in runtime_fields)
                or item.get("backend_kind") != "directml"
                or item.get("warning_count") != 0
            ):
                raise ValueError("Round 50 checkpoint evidence is invalid")
    source_binding = binding.get("source_certificate")
    if not isinstance(source_binding, Mapping):
        raise ValueError("Round 50 source certificate binding is absent")
    source_path = Path(str(source_binding.get("path") or ""))
    if (
        source_binding.get("canonical_sha256") != EXPECTED_SOURCE_SHA256
        or not source_path.is_file()
        or _file_sha256(source_path) != source_binding.get("file_sha256")
    ):
        raise ValueError("Round 50 source certificate drifted")
    reviewer = ai.get("risk_reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or reviewer.get("live_benchmark_file_sha256") != _file_sha256(ai_benchmark_path)
        or reviewer.get("financial_edge_tested_by_safety_benchmark") is not False
        or reviewer.get("order_authority") is not False
        or reviewer.get("ai_can_repair_failed_numerical_or_economic_gate") is not False
    ):
        raise ValueError("Round 50 AI safety evidence drifted")
    return report, design, report_sha, binding_sha


def _timestamp_iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=UTC).isoformat()


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


def _maximum_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        equity += float(value)
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _table_rows(
    report: Mapping[str, object], design: Mapping[str, object]
) -> dict[str, list[dict[str, object]]]:
    forecast: list[dict[str, object]] = []
    monthly_forecast: list[dict[str, object]] = []
    symbol_forecast: list[dict[str, object]] = []
    seed_stability: list[dict[str, object]] = []
    training: list[dict[str, object]] = []
    models: list[dict[str, object]] = []
    scenarios: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    monthly_performance: list[dict[str, object]] = []
    symbols: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    baselines: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        diagnostic = report["diagnostics"][candidate]
        quality = diagnostic["quality_gate"]
        for row in diagnostic["sides"]:
            ece = row["event_group_ece"]
            forecast.append(
                {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "event_group_ece"
                    },
                    "event_ece_stop_loss": ece["stop_loss"],
                    "event_ece_take_profit": ece["take_profit"],
                    "event_ece_timeout": ece["timeout"],
                    "quality_gate_passed": quality["passed"],
                    "quality_gate_reasons": quality["reasons"],
                }
            )
        monthly_forecast.extend(dict(row) for row in diagnostic["monthly"])
        symbol_forecast.extend(dict(row) for row in diagnostic["symbols"])
        seed_stability.extend(
            {"candidate_id": candidate, **row} for row in diagnostic["seed_stability"]
        )
        training.extend(dict(row) for row in report["training_history"][candidate])
        models.extend(dict(row) for row in report["artifacts"][candidate])
        fixed = report["fixed_policy"][candidate]
        candidate_trades: list[dict[str, object]] = []
        for item in fixed["trades"]:
            trade = {
                **item,
                "decision_time_utc": _timestamp_iso(item["decision_time_ms"]),
                "entry_time_utc": _timestamp_iso(item["entry_time_ms"]),
                "exit_time_utc": _timestamp_iso(item["exit_time_ms"]),
            }
            candidate_trades.append(trade)
            trades.append(trade)
        for scenario in ("base", "stress"):
            result = fixed["scenarios"][scenario]
            scenarios.append(
                {
                    "candidate_id": candidate,
                    "scenario": scenario,
                    **{
                        key: value
                        for key, value in result.items()
                        if key
                        not in {
                            "daily",
                            "bootstrap",
                            "symbol_closed_trades",
                            "symbol_net_return_fraction",
                        }
                    },
                    "bootstrap_lower_return_fraction": result["bootstrap"][
                        "lower_return_fraction"
                    ],
                    "bootstrap_median_return_fraction": result["bootstrap"][
                        "median_return_fraction"
                    ],
                    "symbol_closed_trades": result["symbol_closed_trades"],
                    "symbol_net_return_fraction": result["symbol_net_return_fraction"],
                }
            )
            scenario_daily = [
                {"candidate_id": candidate, "scenario": scenario, **row}
                for row in result["daily"]
            ]
            daily.extend(scenario_daily)
            months = sorted({str(row["date"])[:7] for row in result["daily"]})
            for month in months:
                month_daily = [
                    row for row in result["daily"] if str(row["date"]).startswith(month)
                ]
                payoff_field = f"{scenario}_net_payoff_bps"
                month_trades = [
                    row
                    for row in candidate_trades
                    if str(row["exit_time_utc"]).startswith(month)
                ]
                monthly_performance.append(
                    {
                        "candidate_id": candidate,
                        "scenario": scenario,
                        "month": month,
                        "return_fraction": sum(
                            float(row["return_fraction"]) for row in month_daily
                        ),
                        "maximum_drawdown_fraction": _maximum_drawdown(
                            [float(row["return_fraction"]) for row in month_daily]
                        ),
                        "closed_trades": len(month_trades),
                        "active_days": sum(
                            float(row["return_fraction"]) != 0.0 for row in month_daily
                        ),
                        "mean_net_payoff_bps": (
                            sum(float(row[payoff_field]) for row in month_trades)
                            / len(month_trades)
                            if month_trades
                            else 0.0
                        ),
                    }
                )
            payoff_field = f"{scenario}_net_payoff_bps"
            for symbol in SYMBOLS:
                symbol_trades = [
                    row for row in candidate_trades if row["symbol"] == symbol
                ]
                daily_returns: defaultdict[str, float] = defaultdict(float)
                for row in symbol_trades:
                    daily_returns[str(row["exit_time_utc"])[:10]] += (
                        float(row[payoff_field]) / 10_000.0 / len(SYMBOLS)
                    )
                payoff = [float(row[payoff_field]) for row in symbol_trades]
                symbols.append(
                    {
                        "candidate_id": candidate,
                        "scenario": scenario,
                        "symbol": symbol,
                        "closed_trades": len(symbol_trades),
                        "total_return_fraction": sum(payoff) / 10_000.0 / len(SYMBOLS),
                        "maximum_drawdown_fraction": _maximum_drawdown(
                            [daily_returns[date] for date in sorted(daily_returns)]
                        ),
                        "mean_net_payoff_bps": sum(payoff) / len(payoff)
                        if payoff
                        else 0.0,
                        "win_rate": (
                            sum(value > 0.0 for value in payoff) / len(payoff)
                            if payoff
                            else 0.0
                        ),
                        "stop_loss_trades": sum(
                            row["event_name"] == "stop_loss" for row in symbol_trades
                        ),
                        "take_profit_trades": sum(
                            row["event_name"] == "take_profit" for row in symbol_trades
                        ),
                        "timeout_trades": sum(
                            row["event_name"] == "timeout" for row in symbol_trades
                        ),
                    }
                )
        economic = report["economic_gates"][candidate]
        gates.append(
            {
                "candidate_id": candidate,
                "quality_gate_passed": quality["passed"],
                "quality_gate_reasons": quality["reasons"],
                "mechanism_gate_applicable": candidate == "competing_risk_barrier_tcn",
                "mechanism_gate_passed": (
                    report["mechanism_gate"]["passed"]
                    if candidate == "competing_risk_barrier_tcn"
                    else None
                ),
                "economic_gate_passed": economic["passed"],
                "economic_gate_reasons": economic["reasons"],
                "leverage_sensitivity_run": report["leverage_sensitivity"][candidate][
                    "run"
                ],
                "ai_paired_uplift_run": report["ai"]["paired_uplift_run"],
            }
        )
        target = report["target_baselines"][candidate]
        for symbol_index, symbol in enumerate(SYMBOLS):
            for side_index, side in enumerate(SIDES):
                event = target["event_class_probability"][symbol_index][side_index]
                baselines.append(
                    {
                        "candidate_id": candidate,
                        "symbol": symbol,
                        "side": side,
                        "direct_mean_risk_units": target["direct_mean_risk_units"][
                            symbol_index
                        ][side_index],
                        "stop_residual_mean_risk_units": target[
                            "stop_residual_mean_risk_units"
                        ][symbol_index][side_index],
                        "take_residual_mean_risk_units": target[
                            "take_residual_mean_risk_units"
                        ][symbol_index][side_index],
                        "training_stop_probability": sum(event[:60]),
                        "training_take_probability": sum(event[60:120]),
                        "training_timeout_probability": event[120],
                        "pooled_timeout_profit_probability": target[
                            "pooled_timeout_profit_probability"
                        ][side_index],
                        "pooled_timeout_mean_risk_units": target[
                            "pooled_timeout_mean_risk_units"
                        ][side_index],
                    }
                )
    roles = [
        {"role": role, "period": period, "selection_contaminated": role == "evaluation"}
        for role, period in design["chronological_roles"].items()
        if role
        in {
            "training",
            "early_stop",
            "calibration",
            "evaluation",
        }
    ]
    sources = [
        {
            "source_type": "research",
            "source": row["source"],
            "finding_used": row["finding_used"],
            "limitation": row["limitation"],
        }
        for row in design["research_basis"]
    ]
    sources.extend(
        (
            {
                "source_type": "dataset",
                "source": "verified Binance USD-M minute panel",
                "finding_used": report["dataset"]["barrier_dataset_sha256"],
                "limitation": "One-minute source resolution; no multi-year second-level claim.",
            },
            {
                "source_type": "AI safety benchmark",
                "source": report["ai"]["risk_reviewer"]["live_benchmark_external_file"],
                "finding_used": report["ai"]["risk_reviewer"]["selected_risk_reviewer"],
                "limitation": "Safety reasoning only; financial edge was not tested.",
            },
        )
    )
    return {
        "forecast": forecast,
        "monthly-forecast": monthly_forecast,
        "symbol-forecast": symbol_forecast,
        "seed-stability": seed_stability,
        "training": training,
        "models": models,
        "scenarios": scenarios,
        "trades": trades,
        "daily-equity": daily,
        "monthly-performance": monthly_performance,
        "symbols": symbols,
        "gates": gates,
        "mechanism": [dict(report["mechanism_gate"])],
        "target-baselines": baselines,
        "roles": roles,
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
        raise ValueError("Round 50 prior progress history is invalid")
    candidate = "competing_risk_barrier_tcn"
    side_rows = report["diagnostics"][candidate]["sides"]
    base = report["fixed_policy"][candidate]["scenarios"]["base"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "60-minute path-bounded competing-risk TCN",
            "periods": "2022-01-01..2025-06-30 roles; eval 2025-01-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": 3600,
            "feature_set": "71 causal features; 5-minute large-kernel TCN; 121 path classes/side",
            "risk_level": "consumed development only; unlevered fixed sleeves",
            "spearman_ic": sum(
                float(item["expected_payoff_spearman"]) for item in side_rows
            )
            / len(side_rows),
            "selected_signals": base["closed_trades"],
            "executable_trades": base["closed_trades"],
            "mean_net_bps": base["mean_net_payoff_bps"],
            "status": "rejected",
            "source_file": "verified Round 50 path-bounded report; event skill did not become after-cost edge",
            "best_policy_trades": base["closed_trades"],
            "best_policy_total_net_bps": 10_000 * float(base["total_return_fraction"]),
            "best_policy_mean_net_bps": base["mean_net_payoff_bps"],
            "best_policy_max_drawdown_bps": 10_000
            * float(base["maximum_drawdown_fraction"]),
            "best_policy_profit_factor": base["profit_factor"],
            "best_model_id": "competing_risk_barrier_tcn_rejected",
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
    start = 830
    for index, (label, color) in enumerate(labels):
        x = start + index * 290
        lines.append(
            f'<rect x="{x}" y="43" width="18" height="18" fill="{color}"/>'
            f'<text x="{x + 28}" y="57" class="note">{html.escape(label)}</text>'
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
        x = left + index * slot + slot * 0.2
        bar_width = slot * 0.6
        py = y(value)
        rect_y = min(py, baseline)
        rect_height = max(abs(baseline - py), 1.0)
        lines.append(
            f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" height="{rect_height:.1f}" fill="{color}"/>'
            f'<text x="{x + bar_width / 2:.1f}" y="{py - 9 if value >= reference else py + 19:.1f}" text-anchor="middle" class="value">{value:{value_format}}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{top + height + 24:.1f}" text-anchor="middle" class="axis">{html.escape(name)}</text>'
        )


def _event_quality_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = [
        (
            f"{SHORT_LABELS[str(row['candidate_id'])]} {str(row['side']).title()}",
            COLORS[str(row["candidate_id"])],
            float(row["event_log_loss_skill"]),
        )
        for row in rows
    ]
    ece = [
        (
            f"{SHORT_LABELS[str(row['candidate_id'])]} {str(row['side']).title()}",
            COLORS[str(row["candidate_id"])],
            float(row["maximum_event_group_ece"]),
        )
        for row in rows
    ]
    lines = _svg_start(
        1500,
        760,
        "Path-event probabilities improved proper scores",
        "Consumed 2025-H1 evaluation; calibration skill is not an after-cost trading edge.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    _bar_panel(
        lines,
        values=values,
        top=145,
        left=130,
        width=1300,
        height=205,
        low=0.0,
        high=0.04,
        label="Event log-loss skill versus training prevalence",
        value_format=".4f",
    )
    _bar_panel(
        lines,
        values=ece,
        top=470,
        left=130,
        width=1300,
        height=180,
        low=0.0,
        high=0.05,
        reference=0.05,
        label="Maximum event-group ECE (frozen ceiling 0.05; lower is better)",
        value_format=".4f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _payoff_quality_svg(
    forecast: Sequence[Mapping[str, object]], scenarios: Sequence[Mapping[str, object]]
) -> str:
    entries = [
        (
            f"{SHORT_LABELS[str(row['candidate_id'])]} {str(row['side']).title()}",
            COLORS[str(row["candidate_id"])],
            float(row["expected_payoff_spearman"]),
        )
        for row in forecast
    ]
    mse = [
        (
            f"{SHORT_LABELS[str(row['candidate_id'])]} {str(row['side']).title()}",
            COLORS[str(row["candidate_id"])],
            float(row["expected_payoff_mse_skill"]),
        )
        for row in forecast
    ]
    tail = [
        (
            SHORT_LABELS[str(row["candidate_id"])],
            COLORS[str(row["candidate_id"])],
            float(row["mean_net_payoff_bps"]),
        )
        for row in scenarios
        if row["scenario"] == "base"
    ]
    rank_low, rank_high = _bounds([item[2] for item in entries], reference=0.03)
    mse_low, mse_high = _bounds([item[2] for item in mse])
    tail_low, tail_high = _bounds([item[2] for item in tail])
    lines = _svg_start(
        1500,
        1010,
        "Expected-payoff rank and selected tail failed",
        "Proper event forecasts did not rank exact after-cost payoff or produce a profitable positive-prediction tail.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    _bar_panel(
        lines,
        values=entries,
        top=145,
        left=130,
        width=1300,
        height=185,
        low=rank_low,
        high=rank_high,
        reference=0.03,
        label="Expected-payoff Spearman (frozen minimum 0.03)",
        value_format=".4f",
    )
    _bar_panel(
        lines,
        values=mse,
        top=445,
        left=130,
        width=1300,
        height=175,
        low=mse_low,
        high=mse_high,
        label="Expected-payoff MSE skill versus training mean",
        value_format=".4f",
    )
    _bar_panel(
        lines,
        values=tail,
        top=745,
        left=130,
        width=1300,
        height=155,
        low=tail_low,
        high=tail_high,
        label="Mean realized payoff of all-seed-positive selected trades (bps, 12 bps charge)",
        value_format="+.2f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _seed_stability_svg(
    forecast: Sequence[Mapping[str, object]],
    stability: Sequence[Mapping[str, object]],
) -> str:
    values = []
    for candidate in CANDIDATES:
        candidate_values = [
            float(row["spearman"])
            for row in stability
            if row["candidate_id"] == candidate
        ]
        values.append(
            (SHORT_LABELS[candidate], COLORS[candidate], min(candidate_values))
        )
    lines = _svg_start(
        1500,
        520,
        "Seed agreement passed while payoff quality failed",
        "Minimum pairwise expected-payoff Spearman across three independently trained seeds; frozen minimum 0.50.",
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


def _line_panel(
    lines: list[str],
    *,
    series: Sequence[tuple[str, str, Sequence[tuple[int, float]]]],
    top: float,
    left: float,
    width: float,
    height: float,
    label: str,
) -> None:
    values = [value for _name, _color, points in series for _x, value in points]
    low, high = _bounds(values, reference=min(values), padding=0.12)
    x_low = min(x for _name, _color, points in series for x, _value in points)
    x_high = max(x for _name, _color, points in series for x, _value in points)
    lines.append(
        f'<text x="{left}" y="{top - 18}" class="label">{html.escape(label)}</text>'
    )
    for tick in (low, (low + high) / 2.0, high):
        py = top + height * (high - tick) / (high - low)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + width}" y2="{py:.1f}" class="grid"/>'
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.4f}</text>'
        )
    for _name, color, points in series:
        coordinates = []
        for epoch, value in points:
            px = left + width * (epoch - x_low) / max(x_high - x_low, 1)
            py = top + height * (high - value) / (high - low)
            coordinates.append(f"{px:.1f},{py:.1f}")
        lines.append(
            f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    for epoch in (x_low, (x_low + x_high) // 2, x_high):
        px = left + width * (epoch - x_low) / max(x_high - x_low, 1)
        lines.append(
            f'<text x="{px:.1f}" y="{top + height + 22}" text-anchor="middle" class="axis">epoch {epoch}</text>'
        )


def _training_svg(rows: Sequence[Mapping[str, object]]) -> str:
    series_event = []
    series_payoff = []
    for candidate in CANDIDATES:
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
        series_event.append(
            (
                LABELS[candidate],
                COLORS[candidate],
                [
                    (int(row["epoch"]), float(row["early_stop_event_log_loss"]))
                    for row in candidate_rows
                ],
            )
        )
        series_payoff.append(
            (
                LABELS[candidate],
                COLORS[candidate],
                [
                    (int(row["epoch"]), float(row["early_stop_primary_mse"]))
                    for row in candidate_rows
                ],
            )
        )
    lines = _svg_start(
        1500,
        760,
        "Validation plateaued under frozen early stopping",
        "Training ended at epoch 14 for the direct control and 15 for the competing-risk model.",
    )
    _legend(lines, [(LABELS[item], COLORS[item]) for item in CANDIDATES])
    _line_panel(
        lines,
        series=series_event,
        top=145,
        left=130,
        width=1300,
        height=205,
        label="Early-stop event log loss",
    )
    _line_panel(
        lines,
        series=series_payoff,
        top=470,
        left=130,
        width=1300,
        height=180,
        label="Early-stop expected-payoff MSE (risk units)",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _policy_svg(rows: Sequence[Mapping[str, object]]) -> str:
    returns = []
    drawdowns = []
    for row in rows:
        candidate = str(row["candidate_id"])
        scenario = str(row["scenario"])
        label = f"{SHORT_LABELS[candidate]} {scenario.title()}"
        color = COLORS[candidate] if scenario == "base" else "#b45309"
        returns.append((label, color, 100.0 * float(row["total_return_fraction"])))
        drawdowns.append(
            (label, color, 100.0 * float(row["maximum_drawdown_fraction"]))
        )
    return_low, return_high = _bounds([item[2] for item in returns])
    lines = _svg_start(
        1500,
        760,
        "Both fixed policies lost after costs",
        "Exact no-overlap ledger at 12 bps base and the identical trades repriced at 16 bps stress; unlevered fixed sleeves.",
    )
    _legend(lines, [("12 bps base", "#0f766e"), ("16 bps stress", "#b45309")])
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
        high=max(item[2] for item in drawdowns) * 1.18,
        label="Maximum drawdown (%)",
        value_format=".2f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _equity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row for row in rows if row["candidate_id"] == "competing_risk_barrier_tcn"
    ]
    scenarios = {
        name: sorted(
            (row for row in selected if row["scenario"] == name),
            key=lambda row: str(row["date"]),
        )
        for name in ("base", "stress")
    }
    dates = [str(row["date"]) for row in scenarios["base"]]
    colors = {"base": "#0f766e", "stress": "#b45309"}
    equity_values = [
        float(row["equity_fraction"])
        for scenario in scenarios.values()
        for row in scenario
    ] + [1.0]
    low, high = _bounds(equity_values, reference=1.0, padding=0.08)
    width, height = 1500, 800
    left, right, top, panel_height = 130, 70, 145, 220
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Dated competing-risk equity and drawdown",
        "Consumed 2025-H1 additive fixed-sleeve replay; cash is the 1.0 baseline and leverage is absent.",
    )
    _legend(
        lines, [("12 bps base", colors["base"]), ("16 bps stress", colors["stress"])]
    )

    def x(index: int) -> float:
        return left + chart_width * index / max(len(dates) - 1, 1)

    def equity_y(value: float) -> float:
        return top + panel_height * (high - value) / (high - low)

    lines.append(
        f'<text x="{left}" y="{top - 18}" class="label">Equity fraction versus cash</text>'
    )
    for tick in (low, 1.0, high):
        py = equity_y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 1.0 else "grid"}"/>'
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    for scenario, scenario_rows in scenarios.items():
        points = " ".join(
            f"{x(index):.1f},{equity_y(float(row['equity_fraction'])):.1f}"
            for index, row in enumerate(scenario_rows)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[scenario]}" stroke-width="3"/>'
        )
    draw_top, draw_height = 500, 170
    drawdowns: dict[str, list[float]] = {}
    for scenario, scenario_rows in scenarios.items():
        peak = 1.0
        values = []
        for row in scenario_rows:
            equity = float(row["equity_fraction"])
            peak = max(peak, equity)
            values.append((peak - equity) / peak)
        drawdowns[scenario] = values
    draw_high = max(value for values in drawdowns.values() for value in values) * 1.12
    lines.append(
        f'<text x="{left}" y="{draw_top - 18}" class="label">Drawdown fraction</text>'
    )
    for tick in (0.0, draw_high / 2.0, draw_high):
        py = draw_top + draw_height * (draw_high - tick) / draw_high
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="grid"/>'
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{100 * tick:.1f}%</text>'
        )
    for scenario, values in drawdowns.items():
        points = " ".join(
            f"{x(index):.1f},{draw_top + draw_height * (draw_high - value) / draw_high:.1f}"
            for index, value in enumerate(values)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[scenario]}" stroke-width="3"/>'
        )
    for index in (0, len(dates) // 2, len(dates) - 1):
        lines.append(
            f'<text x="{x(index):.1f}" y="{height - 48}" text-anchor="middle" class="axis">{dates[index]}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _monthly_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row for row in rows if row["candidate_id"] == "competing_risk_barrier_tcn"
    ]
    months = sorted({str(row["month"]) for row in selected})
    base = {str(row["month"]): row for row in selected if row["scenario"] == "base"}
    stress = {str(row["month"]): row for row in selected if row["scenario"] == "stress"}
    returns = [
        100.0 * float(source[month]["return_fraction"])
        for source in (base, stress)
        for month in months
    ]
    low, high = _bounds(returns)
    width, height = 1500, 720
    left, right, top, panel_height = 130, 70, 145, 245
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "Monthly losses were broad rather than one isolated event",
        "Competing-risk fixed policy, 2025-H1; monthly bars use the exact daily ledger and trade counts use exit month.",
    )
    _legend(lines, [("12 bps base", "#0f766e"), ("16 bps stress", "#b45309")])

    def y(value: float) -> float:
        return top + panel_height * (high - value) / (high - low)

    for tick in (low, 0.0, high):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}%</text>'
        )
    slot = chart_width / len(months)
    baseline = y(0.0)
    for index, month in enumerate(months):
        for offset, (source, color) in enumerate(
            ((base, "#0f766e"), (stress, "#b45309"))
        ):
            value = 100.0 * float(source[month]["return_fraction"])
            px = left + index * slot + slot * (0.18 + 0.34 * offset)
            py = y(value)
            lines.append(
                f'<rect x="{px:.1f}" y="{min(py, baseline):.1f}" width="{slot * 0.28:.1f}" height="{max(abs(baseline - py), 1):.1f}" fill="{color}"/>'
            )
        lines.append(
            f'<text x="{left + (index + 0.5) * slot:.1f}" y="{top + panel_height + 24}" text-anchor="middle" class="axis">{month}</text>'
        )
    count_top, count_height = 520, 105
    max_count = max(int(base[month]["closed_trades"]) for month in months)
    lines.append(
        f'<text x="{left}" y="{count_top - 18}" class="label">Closed trades by exit month</text>'
    )
    for index, month in enumerate(months):
        count = int(base[month]["closed_trades"])
        bar_height = count_height * count / max_count
        px = left + index * slot + slot * 0.27
        lines.append(
            f'<rect x="{px:.1f}" y="{count_top + count_height - bar_height:.1f}" width="{slot * 0.46:.1f}" height="{bar_height:.1f}" fill="#526674"/>'
            f'<text x="{px + slot * 0.23:.1f}" y="{count_top + count_height - bar_height - 8:.1f}" text-anchor="middle" class="value">{count}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _symbol_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row for row in rows if row["candidate_id"] == "competing_risk_barrier_tcn"
    ]
    base = {str(row["symbol"]): row for row in selected if row["scenario"] == "base"}
    stress = {
        str(row["symbol"]): row for row in selected if row["scenario"] == "stress"
    }
    returns = [
        (
            f"{symbol} Base",
            "#0f766e",
            100.0 * float(base[symbol]["total_return_fraction"]),
        )
        for symbol in SYMBOLS
    ] + [
        (
            f"{symbol} Stress",
            "#b45309",
            100.0 * float(stress[symbol]["total_return_fraction"]),
        )
        for symbol in SYMBOLS
    ]
    drawdowns = [
        (symbol, "#2563a6", 100.0 * float(base[symbol]["maximum_drawdown_fraction"]))
        for symbol in SYMBOLS
    ]
    low, high = _bounds([item[2] for item in returns])
    lines = _svg_start(
        1500,
        760,
        "Every symbol lost under the competing-risk policy",
        "Fixed one-third sleeves; symbol drawdown is reconstructed from the same exit-dated trade ledger.",
    )
    _legend(lines, [("12 bps base", "#0f766e"), ("16 bps stress", "#b45309")])
    _bar_panel(
        lines,
        values=returns,
        top=145,
        left=130,
        width=1300,
        height=205,
        low=low,
        high=high,
        label="Net portfolio return contribution (%)",
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
        high=max(item[2] for item in drawdowns) * 1.18,
        label="Base symbol-sleeve maximum drawdown (%)",
        value_format=".2f",
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _failure_analysis(
    report: Mapping[str, object],
    *,
    report_sha: str,
    binding_sha: str,
    evidence_root: Path,
) -> dict[str, object]:
    direct = report["fixed_policy"]["direct_barrier_mean_tcn"]["scenarios"]["base"]
    path = report["fixed_policy"]["competing_risk_barrier_tcn"]["scenarios"]["base"]
    side_rows = {
        row["side"]: row
        for row in report["diagnostics"]["competing_risk_barrier_tcn"]["sides"]
    }
    analysis: dict[str, object] = {
        "schema_version": "round-050-path-bounded-failure-analysis-v1",
        "round": ROUND,
        "status": "rejected",
        "source_report_canonical_sha256": report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "binding_sha256": binding_sha,
        "observed_result": {
            "direct_control": {
                "trades": direct["closed_trades"],
                "return_fraction": direct["total_return_fraction"],
                "maximum_drawdown_fraction": direct["maximum_drawdown_fraction"],
                "profit_factor": direct["profit_factor"],
            },
            "competing_risk": {
                "trades": path["closed_trades"],
                "return_fraction": path["total_return_fraction"],
                "maximum_drawdown_fraction": path["maximum_drawdown_fraction"],
                "profit_factor": path["profit_factor"],
                "short_expected_payoff_spearman": side_rows["short"][
                    "expected_payoff_spearman"
                ],
                "long_expected_payoff_spearman": side_rows["long"][
                    "expected_payoff_spearman"
                ],
            },
            "mechanism_gate": report["mechanism_gate"],
        },
        "findings": [
            "Event-time log loss and grouped Brier score improved for both sides, so the path classifier learned real distributional structure.",
            "That probability structure did not identify after-cost action value: the competing-risk average side Spearman improvement was negative and short-side rank was negative in every symbol and month.",
            "The all-seed-positive rule exposed rather than repaired the error: it selected 610 trades, 525 short, and all three symbol sleeves lost after 12 bps costs.",
            "Long-side ranking improved, but a mixed long/short policy cannot be promoted while the short mechanism fails breadth, economics, and drawdown gates.",
            "The direct control also lost and did not meet activity, breadth, confidence, or expected-payoff quality requirements.",
            "Leverage and AI were correctly withheld; neither can convert a failed deterministic unlevered model into valid evidence.",
        ],
        "next_model_requirements": [
            "Freeze side-specific model and policy gates so a failed short lane cannot dominate a viable long lane.",
            "Test a lower-variance categorical payoff distribution against the direct mean using proper scoring and calibration-only transforms.",
            "Preserve exact next-open barrier cash flows, funding, gap-through stops, costs, and no-overlap replay without outcome reweighting.",
            "Diagnose event-probability-to-payoff reconstruction by event group, duration, symbol, month, and predicted-value quantile before another full run.",
            "Require positive unlevered breadth and familywise block-bootstrap evidence before leverage or an AI veto ablation is allowed.",
            "Keep Qwen3 8B in asynchronous veto-only scope; its safety benchmark is not alpha evidence.",
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
    forecast = {
        (candidate, row["side"]): row
        for candidate in CANDIDATES
        for row in report["diagnostics"][candidate]["sides"]
    }
    rows = []
    for candidate in CANDIDATES:
        base = report["fixed_policy"][candidate]["scenarios"]["base"]
        stress = report["fixed_policy"][candidate]["scenarios"]["stress"]
        rows.append(
            f"| {LABELS[candidate]} | {float(forecast[(candidate, 'short')]['expected_payoff_spearman']):+.4f} | "
            f"{float(forecast[(candidate, 'long')]['expected_payoff_spearman']):+.4f} | {base['closed_trades']} | "
            f"{100 * float(base['total_return_fraction']):+.2f}% | {100 * float(stress['total_return_fraction']):+.2f}% | "
            f"{100 * float(base['maximum_drawdown_fraction']):.2f}% | {float(base['profit_factor']):.3f} | false/false |"
        )
    runtime = report["runtime"]
    reviewer = report["ai"]["risk_reviewer"]
    return f"""# Round 50: Path-Bounded Competing-Risk TCN

> **Beta research warning:** neither model is approved for testnet, live day trading, leverage, autonomous execution, or a profitability claim. The 2025-H1 interval is consumed development evidence.

Round 50 tested whether exact stop-loss, take-profit, and timeout probabilities produce better after-cost action values than a matched direct-mean control. Event distributions improved, but payoff ranking and fixed-policy economics failed. The candidate was rejected.

![Event quality](charts/event-quality.svg)

![Expected-payoff quality](charts/expected-payoff-quality.svg)

| Candidate | Short payoff rank | Long payoff rank | Trades | Base return | Stress return | Base drawdown | Profit factor | Quality/economic gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

The competing-risk model made `610` non-overlapping trades on `108` active days, including `525` shorts. It lost `35.73%` at 12 bps and `43.86%` at 16 bps, with `37.41%` base drawdown; BTCUSDT, ETHUSDT, and SOLUSDT all lost. The direct control made `90` trades and lost `4.30%`. These are additive fixed-sleeve research returns, not deployable portfolio estimates.

![Policy economics](charts/policy-economics.svg)

![Dated equity and drawdown](charts/daily-equity-drawdown.svg)

![Monthly performance](charts/monthly-performance.svg)

![Symbol performance](charts/symbol-performance.svg)

![Seed stability](charts/seed-stability.svg)

![Training dynamics](charts/training-dynamics.svg)

![Research progress](charts/research-progress.svg)

The source is verified Binance USD-M **one-minute** BTCUSDT, ETHUSDT, and SOLUSDT data from 2022 through 2025-H1. Decisions occur every five minutes and each target follows the next-minute open for up to 60 minutes. This is not a multi-year second-level dataset claim.

DirectML trained six models on the AMD GPU in `{float(runtime["elapsed_seconds"]):.1f}s`, peaked at `{float(runtime["memory"]["peak_working_set_bytes"]) / 2**30:.2f} GiB` working set, recorded zero CPU fallbacks, and hash-verified every checkpoint and prediction artifact. `{reviewer["selected_risk_reviewer"]}` passed the separate safety benchmark, but AI uplift was not run because no deterministic model passed. Leverage was likewise withheld.

Data: [forecast quality](forecast.csv) | [monthly forecast](monthly-forecast.csv) | [symbol forecast](symbol-forecast.csv) | [seed stability](seed-stability.csv) | [training](training.csv) | [models](models.csv) | [target baselines](target-baselines.csv) | [trades](trades.csv) | [scenarios](scenarios.csv) | [monthly performance](monthly-performance.csv) | [symbol economics](symbols.csv) | [daily equity](daily-equity.csv) | [gates](gates.csv) | [mechanism](mechanism.csv) | [roles](roles.csv) | [sources](sources.csv) | [progress](progress.csv) | [failure analysis](../round-050-failure-analysis.json) | [validated source report](screen.json) | [integrity report](report.json)
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
    ai_benchmark_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
    failure_path: Path,
) -> dict[str, object]:
    report, design, report_sha, binding_sha = _validated_source(
        evidence_root, design_path, binding_path, ai_benchmark_path
    )
    tables = _table_rows(report, design)
    progress_fields, progress_rows = _progress_rows(prior_progress_path, report)
    tables["progress"] = progress_rows
    charts = output_dir / "charts"
    chart_sources = {
        "charts/event-quality.svg": ["forecast.csv"],
        "charts/expected-payoff-quality.svg": ["forecast.csv", "scenarios.csv"],
        "charts/seed-stability.svg": ["seed-stability.csv"],
        "charts/training-dynamics.svg": ["training.csv"],
        "charts/policy-economics.svg": ["scenarios.csv"],
        "charts/daily-equity-drawdown.svg": ["daily-equity.csv"],
        "charts/monthly-performance.svg": ["monthly-performance.csv"],
        "charts/symbol-performance.svg": ["symbols.csv"],
        "charts/research-progress.svg": ["progress.csv"],
    }
    table_paths = {output_dir / f"{name}.csv" for name in tables}
    chart_paths = {output_dir / path for path in chart_sources}
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
        (evidence_root / "report.json").read_text(encoding="utf-8"),
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "event-quality.svg", _event_quality_svg(tables["forecast"]))
    _write_text(
        charts / "expected-payoff-quality.svg",
        _payoff_quality_svg(tables["forecast"], tables["scenarios"]),
    )
    _write_text(
        charts / "seed-stability.svg",
        _seed_stability_svg(tables["forecast"], tables["seed-stability"]),
    )
    _write_text(charts / "training-dynamics.svg", _training_svg(tables["training"]))
    _write_text(charts / "policy-economics.svg", _policy_svg(tables["scenarios"]))
    _write_text(
        charts / "daily-equity-drawdown.svg", _equity_svg(tables["daily-equity"])
    )
    _write_text(
        charts / "monthly-performance.svg",
        _monthly_svg(tables["monthly-performance"]),
    )
    _write_text(charts / "symbol-performance.svg", _symbol_svg(tables["symbols"]))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress_rows))
    failure = _failure_analysis(
        report,
        report_sha=report_sha,
        binding_sha=binding_sha,
        evidence_root=evidence_root,
    )
    _write_text(
        failure_path,
        json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "path_bounded_competing_risk_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "dataset_sha256": report["dataset"]["barrier_dataset_sha256"],
        "predecessor_dataset_sha256": report["dataset"]["predecessor_dataset_sha256"],
        "dataset_rows": report["dataset"]["rows"],
        "evaluation_timestamps": report["diagnostics"][CANDIDATES[0]]["evaluation_rows"]
        // (len(SYMBOLS) * len(SIDES)),
        "source_resolution_seconds": report["dataset"]["source_resolution_seconds"],
        "decision_interval_seconds": report["dataset"]["decision_interval_seconds"],
        "directml_model_artifact_count": 6,
        "external_artifacts_hash_verified": True,
        "candidate_quality_gate_pass_count": 0,
        "candidate_economic_gate_pass_count": 0,
        "mechanism_gate_passed": False,
        "leverage_sensitivity_run": False,
        "ai_paired_uplift_run": False,
        "selection_contaminated": True,
        "development_only": True,
        "trading_authority": False,
        "profitability_claim": False,
        "ai_uplift_claim": False,
        "leverage_applied": False,
        "failure_analysis_sha256": failure["analysis_sha256"],
        "failure_analysis_file_sha256": _file_sha256(failure_path),
        "graph_sources": chart_sources,
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
        default=research / "round-050-path-bounded-competing-risk-tcn-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-050-path-bounded-competing-risk-tcn-binding.json",
    )
    parser.add_argument("--ai-benchmark", type=Path, required=True)
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
    parser.add_argument(
        "--failure-path",
        type=Path,
        default=research / "round-050-failure-analysis.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        evidence_root=arguments.evidence_root.resolve(),
        design_path=arguments.design.resolve(),
        binding_path=arguments.binding.resolve(),
        ai_benchmark_path=arguments.ai_benchmark.resolve(),
        prior_progress_path=arguments.prior_progress.resolve(),
        output_dir=arguments.output_dir.resolve(),
        failure_path=arguments.failure_path.resolve(),
    )
    print(json.dumps(publication, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
