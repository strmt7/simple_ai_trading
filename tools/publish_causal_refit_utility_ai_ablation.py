"""Publish verified Round 39 rolling-refit, utility, and AI evidence."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.derivatives_hurdle_model import (  # noqa: E402
    _stationary_bootstrap_mean_net,
)
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
from tools.run_causal_refit_utility_ai_ablation import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


PUBLICATION_SCHEMA = "causal-refit-utility-ai-ablation-publication-v1"
ROUND = 39
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 7))


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
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design = _read_object(design_path, "Round 39 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 39 design")
    binding = _read_object(binding_path, "Round 39 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 39 binding")
    report = _read_object(evidence_root / "report.json", "Round 39 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 39 report"
    )
    cases = _read_object(evidence_root / "ai-cases.json", "Round 39 AI case set")
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit")
        != binding.get("implementation_commit")
    ):
        raise ValueError("Round 39 evidence lineage is invalid")
    _validate_tree(report)
    source = report.get("source_evidence")
    price = source.get("price_flow") if isinstance(source, Mapping) else None
    dataset = report.get("dataset")
    candidates = report.get("candidate_results")
    artifacts = report.get("model_artifacts")
    ai_reports = report.get("ai_reports")
    if (
        not isinstance(source, Mapping)
        or source.get("selection_confirmation_or_terminal_rows_read") is not False
        or source.get("derivatives_panel_sha256")
        != "3e7a57184d690afc7061e9f48e437716ff73b72f01302d2eb151112351e8cf7e"
        or not isinstance(price, Mapping)
        or price.get("panel_stream_sha256")
        != "40ef0a76d57fa844fb01ccd90d7c768f5fbbf613467303a2dc1dcdd39b100a3d"
        or price.get("selection_confirmation_or_terminal_rows_read") is not False
        or not isinstance(dataset, Mapping)
        or dataset.get("rows") != 1_098_105
        or dataset.get("feature_count") != 103
        or dataset.get("model_feature_count") != 71
        or dataset.get("features_dtype") != "float32"
        or dataset.get("persistent_feature_copy_created") is not False
        or not isinstance(candidates, list)
        or len(candidates) != 4
        or sum(len(item.get("monthly_results", [])) for item in candidates) != 24
        or sum(
            len(month.get("threshold_trace", []))
            for item in candidates
            for month in item.get("monthly_results", [])
        )
        != 480
        or not isinstance(artifacts, list)
        or len(artifacts) != 60
        or report.get("ai_entry_support_candidates")
        != [item["candidate_id"] for item in candidates]
        or report.get("aggregate_ml_gate_passed_candidates") != []
        or report.get("ai_uplift_gate_passed_models") != []
        or not isinstance(ai_reports, list)
        or len(ai_reports) != 2
        or {item.get("model") for item in ai_reports if isinstance(item, Mapping)}
        != {"qwen3:8b", "fino1:8b"}
        or any(
            item.get("cases") != 180
            or item.get("valid_responses") != 180
            or item.get("provider_failures") != 0
            or item.get("uplift_gate_passed") is not False
            for item in ai_reports
        )
        or report.get("selection_confirmation_accessed") is not False
        or report.get("terminal_2026_accessed") is not False
        or cases.get("schema_version") != "causal-rolling-ai-veto-case-set-v1"
        or cases.get("case_set_sha256")
        != "fb94623fc8c5906bfb8f18a951def9ca19bae7d66ca3d2e91ef4d3cb93e5985f"
        or not isinstance(cases.get("cases"), list)
        or len(cases["cases"]) != 180
        or report.get("ai_case_set", {}).get("case_set_sha256")
        != cases.get("case_set_sha256")
    ):
        raise ValueError("Round 39 source, model, or AI evidence drifted")
    for candidate in candidates:
        if (
            candidate.get("selected_threshold_months") != 6
            or candidate.get("months_with_trades") != 6
            or candidate.get("ai_entry_support_passed") is not True
            or candidate.get("aggregate_ml_gate_passed") is not False
            or float(candidate["aggregate_replay"]["mean_net_bps"]) >= 0.0
        ):
            raise ValueError("Round 39 candidate gate evidence drifted")
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Round 39 unexpectedly grants {field}")
    return report, cases, report_sha, binding_sha


def _candidate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        replay = candidate["aggregate_replay"]
        row = {
            "candidate_id": candidate["candidate_id"],
            "architecture": candidate["architecture"],
            "weighting": candidate["weighting"],
            "horizon_minutes": candidate["horizon_minutes"],
            "selected_threshold_months": candidate["selected_threshold_months"],
            "months_with_trades": candidate["months_with_trades"],
            **{key: replay[key] for key in replay if key != "trades_by_symbol"},
            **{
                f"{symbol.lower()}_trades": replay["trades_by_symbol"][symbol]
                for symbol in SYMBOLS
            },
            "maximum_single_month_fraction_of_positive_net_bps": candidate[
                "maximum_single_month_fraction_of_positive_net_bps"
            ],
            "ai_entry_support_passed": candidate["ai_entry_support_passed"],
            "aggregate_ml_gate_passed": candidate["aggregate_ml_gate_passed"],
            "aggregate_ml_gate_reasons": ";".join(
                candidate["aggregate_ml_gate_reasons"]
            ),
        }
        rows.append(row)
    return rows


def _monthly_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        for month in candidate["monthly_results"]:
            selected = month["selected_action_threshold"]
            replay = month["evaluation_replay"]
            classification = month["evaluation_classification"]
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "architecture": candidate["architecture"],
                    "weighting": candidate["weighting"],
                    "horizon_minutes": candidate["horizon_minutes"],
                    "evaluation_month": month["evaluation_month"],
                    "maximum_action_probability": selected[
                        "maximum_action_probability"
                    ],
                    "direction_probability_margin": selected[
                        "direction_probability_margin"
                    ],
                    "evaluation_rows": classification["rows"],
                    "multiclass_log_loss": classification["multiclass_log_loss"],
                    "balanced_accuracy": classification["balanced_accuracy"],
                    "total_trades": replay["total_trades"],
                    "btcusdt_trades": replay["trades_by_symbol"]["BTCUSDT"],
                    "ethusdt_trades": replay["trades_by_symbol"]["ETHUSDT"],
                    "solusdt_trades": replay["trades_by_symbol"]["SOLUSDT"],
                    "mean_net_bps": replay["mean_net_bps"],
                    "profit_factor": replay["profit_factor"],
                    "day_block_lower_95_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_lower_95"
                    ],
                    "day_block_upper_95_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_upper_95"
                    ],
                }
            )
    return rows


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        for month in candidate["monthly_results"]:
            selected = month["selected_action_threshold"]
            for trace in month["threshold_trace"]:
                replay = trace["replay"]
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "evaluation_month": month["evaluation_month"],
                        "maximum_action_probability": trace[
                            "maximum_action_probability"
                        ],
                        "direction_probability_margin": trace[
                            "direction_probability_margin"
                        ],
                        "support_passed": trace["support_passed"],
                        "selected": (
                            trace["maximum_action_probability"]
                            == selected["maximum_action_probability"]
                            and trace["direction_probability_margin"]
                            == selected["direction_probability_margin"]
                        ),
                        "total_trades": replay["total_trades"],
                        "btcusdt_trades": replay["trades_by_symbol"]["BTCUSDT"],
                        "ethusdt_trades": replay["trades_by_symbol"]["ETHUSDT"],
                        "solusdt_trades": replay["trades_by_symbol"]["SOLUSDT"],
                        "active_utc_days": replay["active_utc_days"],
                        "mean_net_bps": replay["mean_net_bps"],
                        "profit_factor": replay["profit_factor"],
                        "day_block_lower_95_net_bps": replay[
                            "day_block_bootstrap_mean_net_bps_lower_95"
                        ],
                        "day_block_upper_95_net_bps": replay[
                            "day_block_bootstrap_mean_net_bps_upper_95"
                        ],
                    }
                )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    fields = (
        "model_id",
        "candidate_id",
        "evaluation_month",
        "target_head",
        "architecture",
        "weighting",
        "horizon_minutes",
        "symbol",
        "feature_count",
        "training_rows",
        "early_stop_rows",
        "training_weight_minimum",
        "training_weight_mean",
        "training_weight_maximum",
        "utility_weight_normalizer_bps",
        "best_iteration",
        "backend_kind",
        "backend_device",
        "reload_max_abs_prediction_error",
        "sha256",
        "bytes",
    )
    return [{field: item[field] for field in fields} for item in report["model_artifacts"]]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    source = report["source_evidence"]
    price_by_symbol = {
        item["symbol"]: item for item in source["price_flow"]["series_evidence"]
    }
    rows: list[dict[str, object]] = []
    for derivatives in source["derivatives_series"]:
        price = price_by_symbol[derivatives["symbol"]]
        rows.append(
            {
                "symbol": derivatives["symbol"],
                "price_rows": price["rows"],
                "price_gap_count": price["gap_count"],
                "price_stream_sha256": price["stream_sha256"],
                "premium_rows": derivatives["premium_rows"],
                "premium_gap_events": derivatives["premium_gap_events"],
                "premium_missing_minutes": derivatives["premium_missing_minutes"],
                "premium_stream_sha256": derivatives["premium_stream_sha256"],
                "funding_rows": derivatives["funding_rows"],
                "funding_stream_sha256": derivatives["funding_stream_sha256"],
            }
        )
    return rows


def _ai_case_rows(cases: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "case_id": item["case_id"],
            "candidate_id": item["candidate_id"],
            "relative_day_index": item["relative_day_index"],
            "evaluation_month": MONTHS[int(item["refit_sequence_index"])],
            "symbol": item["symbol"],
            "horizon_minutes": item["horizon_minutes"],
            "direction": item["direction"],
            "action_probability": item["action_probability"],
            "direction_probability_margin": item[
                "direction_probability_margin"
            ],
            "outcome_net_bps": item["outcome_net_bps"],
        }
        for item in cases["cases"]
    ]


def _ai_rows(report: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    summary_fields = (
        "model",
        "model_digest",
        "model_metadata_sha256",
        "case_set_sha256",
        "cases",
        "batches",
        "valid_responses",
        "approvals",
        "vetoes",
        "cooldowns",
        "provider_failures",
        "average_batch_latency_seconds",
        "retained_active_days",
        "maximum_retained_symbol_fraction",
        "baseline_total_net_bps",
        "ai_total_net_bps",
        "baseline_mean_case_net_bps",
        "ai_mean_retained_case_net_bps",
        "ai_profit_factor",
        "ai_median_monthly_net_bps",
        "ai_negative_month_fraction",
        "baseline_max_drawdown_bps",
        "ai_max_drawdown_bps",
        "matched_days",
        "positive_daily_delta_rate",
        "exact_sign_test_p_value",
        "mean_daily_delta_bps",
        "bootstrap_delta_lower_95_bps",
        "bootstrap_delta_upper_95_bps",
        "uplift_gate_passed",
    )
    for model in report["ai_reports"]:
        summary = {field: model[field] for field in summary_fields}
        for symbol in SYMBOLS:
            summary[f"{symbol.lower()}_retained"] = model[
                "retained_trades_by_symbol"
            ][symbol]
        summary["uplift_gate_reasons"] = ";".join(model["uplift_gate_reasons"])
        summaries.append(summary)
        for result in model["results"]:
            decisions.append(
                {
                    "model": model["model"],
                    "case_id": result["case_id"],
                    "batch_index": result["batch_index"],
                    "batch_latency_seconds": result["batch_latency_seconds"],
                    "prompt_sha256": result["prompt_sha256"],
                    "response_sha256": result["response_sha256"],
                    "action": result["decision"]["action"],
                    "risk_multiplier": result["decision"]["risk_multiplier"],
                    "confidence": result["decision"]["confidence"],
                    "reason_codes": ";".join(result["decision"]["reason_codes"]),
                    "valid": result["decision"]["valid"],
                    "baseline_net_bps": result["baseline_net_bps"],
                    "ai_net_bps": result["ai_net_bps"],
                }
            )
    return summaries, decisions


def _capacity_rows(cases: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in cases:
        value = float(case["outcome_net_bps"])
        groups[("symbol", str(case["symbol"]))].append(value)
        groups[("month", str(case["evaluation_month"]))].append(value)
        groups[("candidate", str(case["candidate_id"]))].append(value)
    outcomes = np.asarray([float(case["outcome_net_bps"]) for case in cases])
    daily_net = np.zeros(181, dtype=np.float64)
    daily_trades = np.zeros(181, dtype=np.int64)
    for case in cases:
        index = int(case["relative_day_index"])
        daily_net[index] += float(case["outcome_net_bps"])
        daily_trades[index] += 1
    interval = _stationary_bootstrap_mean_net(
        daily_net,
        daily_trades,
        samples=50_000,
        seed=3939,
    )

    def metrics(scope: str, member: str, values: Sequence[float]) -> dict[str, object]:
        current = np.asarray(values, dtype=np.float64)
        positive = float(np.sum(current[current > 0.0]))
        negative = float(np.sum(current[current < 0.0]))
        return {
            "scope": scope,
            "member": member,
            "trades": int(current.size),
            "total_net_bps": float(np.sum(current)),
            "mean_net_bps": float(np.mean(current)),
            "median_net_bps": float(np.median(current)),
            "positive_rate": float(np.mean(current > 0.0)),
            "profit_factor": positive / abs(negative) if negative < 0.0 else "",
            "day_block_lower_95_mean_net_bps": "",
            "day_block_median_mean_net_bps": "",
            "day_block_upper_95_mean_net_bps": "",
            "bootstrap_samples": "",
            "bootstrap_seed": "",
        }

    rows = [metrics("overall", "all", outcomes)]
    rows[0].update(
        {
            "day_block_lower_95_mean_net_bps": interval[0],
            "day_block_median_mean_net_bps": interval[1],
            "day_block_upper_95_mean_net_bps": interval[2],
            "bootstrap_samples": 50_000,
            "bootstrap_seed": 3939,
        }
    )
    rows.extend(
        metrics(scope, member, values)
        for (scope, member), values in sorted(groups.items())
    )
    return rows


def _progress_rows(path: Path, candidates: Sequence[Mapping[str, object]]) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed = [int(row["round"]) for row in rows]
    if not fields or observed not in (list(range(1, 39)), list(range(1, 40))):
        raise ValueError("Round 39 prior progress history is invalid")
    rows = [row for row in rows if int(row["round"]) != ROUND]
    best = max(candidates, key=lambda item: float(item["mean_net_bps"]))
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "causal monthly refit, utility weighting, and local-AI veto",
            "periods": "2022-10-01..2025-06-30 rolling roles",
            "selection_contaminated": True,
            "horizon_seconds": "1800;7200",
            "feature_set": "71 causal price/flow features; funding cash-flow accounting",
            "risk_level": "research-only; no policy",
            "selected_signals": best["total_trades"],
            "executable_trades": best["total_trades"],
            "mean_net_bps": best["mean_net_bps"],
            "status": "rejected",
            "source_file": "verified Round 39 rolling-refit/AI report",
            "best_model_id": best["candidate_id"],
            "daily_model_fits": 60,
            "calibration_threshold_traces": 480,
            "accepted_thresholds": 0,
            "ensemble_models": 60,
            "development_consumed": True,
        }
    )
    rows.append(row)
    return fields, rows


def _candidate_label(row: Mapping[str, object]) -> str:
    architecture = str(row["architecture"])
    model = "shared hurdle" if architecture.startswith("shared") else "symbol direct"
    weighting = "utility" if str(row["weighting"]).startswith("bounded") else "equal"
    return f"{model} / {weighting} / {row['horizon_minutes']}m"


def _candidate_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = sorted(rows, key=lambda item: float(item["mean_net_bps"]), reverse=True)
    width, height = 1480, 520
    left, right, top, row_height = 360, 240, 140, 64
    chart_width = width - left - right
    low = min(float(row["day_block_bootstrap_mean_net_bps_lower_95"]) for row in selected) - 2
    high = max(2.0, max(float(row["day_block_bootstrap_mean_net_bps_upper_95"]) for row in selected) + 2)

    def x(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Rolling refits did not recover after-cost expectancy",
        "Aggregate 2025-H1 day-block intervals; all four activity-complete candidates failed economics.",
    )
    for tick in range(-25, 6, 5):
        if low <= tick <= high:
            px = x(float(tick))
            lines.append(f'<line x1="{px:.1f}" y1="{top-24}" x2="{px:.1f}" y2="{top+row_height*len(selected)}" class="grid"/>')
            lines.append(f'<text x="{px:.1f}" y="{top+row_height*len(selected)+25}" text-anchor="middle" class="axis">{tick:+d}</text>')
    lines.append(f'<line x1="{x(0):.1f}" y1="{top-24}" x2="{x(0):.1f}" y2="{top+row_height*len(selected)}" class="zero"/>')
    for index, row in enumerate(selected):
        y = top + index * row_height
        lo = float(row["day_block_bootstrap_mean_net_bps_lower_95"])
        mean = float(row["mean_net_bps"])
        hi = float(row["day_block_bootstrap_mean_net_bps_upper_95"])
        lines.append(f'<text x="{left-18}" y="{y+5}" text-anchor="end" class="label">{html.escape(_candidate_label(row))}</text>')
        lines.append(f'<line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" stroke="#94a3b8" stroke-width="8" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="8" fill="#b42318"/>')
        lines.append(f'<text x="{width-70}" y="{y+5}" text-anchor="end" class="value">{mean:+.2f} bps; n={int(row["total_trades"]):,}</text>')
    lines.append(f'<text x="{left+chart_width/2:.1f}" y="{height-34}" text-anchor="middle" class="label">Mean net basis points per trade</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _monthly_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1480, 720
    left, right, top, chart_height = 110, 70, 140, 430
    chart_width = width - left - right
    low, high = -25.0, 15.0

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    colors = ("#2563eb", "#0f766e", "#b45309", "#7b559c")
    candidates = sorted({str(row["candidate_id"]) for row in rows})
    lines = _svg_start(
        width,
        height,
        "Monthly refits still failed after regime change",
        "Evaluation-month mean net bps after the frozen 12 bps charge and exact funding cash flows.",
    )
    for tick in range(-20, 11, 10):
        py = y(float(tick))
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left+chart_width}" y2="{py:.1f}" class="{("zero" if tick == 0 else "grid")}"/>')
        lines.append(f'<text x="{left-14}" y="{py+4:.1f}" text-anchor="end" class="axis">{tick:+d}</text>')
    for candidate_index, candidate in enumerate(candidates):
        current = sorted(
            (row for row in rows if row["candidate_id"] == candidate),
            key=lambda item: str(item["evaluation_month"]),
        )
        points = []
        for month_index, row in enumerate(current):
            x = left + chart_width * (month_index + 0.5) / 6
            points.append((x, y(float(row["mean_net_bps"]))))
        color = colors[candidate_index]
        lines.append('<polyline points="' + " ".join(f"{x:.1f},{py:.1f}" for x, py in points) + f'" fill="none" stroke="{color}" stroke-width="4"/>')
        lines.extend(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="6" fill="{color}"/>' for x, py in points)
    for month_index, month in enumerate(MONTHS):
        x = left + chart_width * (month_index + 0.5) / 6
        lines.append(f'<text x="{x:.1f}" y="{top+chart_height+30}" text-anchor="middle" class="axis">{month}</text>')
    for index, candidate in enumerate(candidates):
        sample = next(row for row in rows if row["candidate_id"] == candidate)
        x = 120 + (index % 2) * 530
        y_legend = 610 + (index // 2) * 34
        lines.append(f'<line x1="{x}" y1="{y_legend}" x2="{x+28}" y2="{y_legend}" stroke="{colors[index]}" stroke-width="4"/><text x="{x+38}" y="{y_legend+5}" class="note">{html.escape(_candidate_label(sample))}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _capacity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    lookup = {
        (str(row["member"]), str(row["scope"])): row for row in rows
    }
    values = [float(lookup[(symbol, "symbol")]["mean_net_bps"]) for symbol in SYMBOLS]
    width, height = 1280, 560
    left, right, top, row_height = 260, 190, 145, 78
    chart_width = width - left - right
    low, high = -10.0, 55.0

    def x(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Confidence-ranked capacity was not diversified economically",
        "The outcome-blind offline sample had 60 cases per symbol, but SOL supplied most net gains.",
    )
    for tick in range(-10, 51, 10):
        px = x(float(tick))
        lines.append(f'<line x1="{px:.1f}" y1="{top-25}" x2="{px:.1f}" y2="{top+row_height*3}" class="grid"/>')
        lines.append(f'<text x="{px:.1f}" y="{top+row_height*3+24}" text-anchor="middle" class="axis">{tick:+d}</text>')
    lines.append(f'<line x1="{x(0):.1f}" y1="{top-25}" x2="{x(0):.1f}" y2="{top+row_height*3}" class="zero"/>')
    colors = ("#2563eb", "#0f766e", "#b45309")
    for index, (symbol, value) in enumerate(zip(SYMBOLS, values, strict=True)):
        y = top + index * row_height
        start, end = x(min(0.0, value)), x(max(0.0, value))
        lines.append(f'<text x="{left-20}" y="{y+20}" text-anchor="end" class="label">{symbol}</text>')
        lines.append(f'<rect x="{start:.1f}" y="{y}" width="{max(end-start, 2):.1f}" height="32" fill="{colors[index]}"/>')
        total = float(lookup[(symbol, "symbol")]["total_net_bps"])
        lines.append(f'<text x="{width-54}" y="{y+21}" text-anchor="end" class="value">{value:+.2f} mean; {total:+.1f} total bps</text>')
    overall = lookup[("all", "overall")]
    lines.append(f'<text x="{left}" y="{height-78}" class="note">Overall: {float(overall["mean_net_bps"]):+.2f} mean bps, PF {float(overall["profit_factor"]):.3f}; 50,000-sample day-block lower 95% = {float(overall["day_block_lower_95_mean_net_bps"]):+.3f} bps.</text>')
    lines.append(f'<text x="{left}" y="{height-47}" class="note">Completed-month confidence ranks make this an offline AI sample, not a deployable ML policy.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _ai_svg(rows: Sequence[Mapping[str, object]]) -> str:
    by_model = {str(row["model"]): row for row in rows}
    baseline = float(rows[0]["baseline_total_net_bps"])
    values = (
        ("ML case baseline", baseline, "#2563eb"),
        ("Qwen3 retained", float(by_model["qwen3:8b"]["ai_total_net_bps"]), "#7b559c"),
        ("Fino1 retained", float(by_model["fino1:8b"]["ai_total_net_bps"]), "#0f766e"),
    )
    width, height = 1280, 600
    left, right, top, row_height = 290, 120, 150, 88
    chart_width = width - left - right
    high = baseline * 1.08

    def x(value: float) -> float:
        return left + chart_width * value / high

    lines = _svg_start(
        width,
        height,
        "Local AI reduced a stronger matched baseline",
        "Same 180 outcome-blind offline cases; totals are summed net bps, not portfolio ROI or an equity curve.",
    )
    for index, (label, value, color) in enumerate(values):
        y = top + index * row_height
        lines.append(f'<text x="{left-20}" y="{y+25}" text-anchor="end" class="label">{label}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{max(x(value)-left, 2):.1f}" height="38" fill="{color}"/>')
        lines.append(f'<text x="{min(x(value)+12, width-80):.1f}" y="{y+25}" class="value">{value:+.1f}</text>')
    fino = by_model["fino1:8b"]
    qwen = by_model["qwen3:8b"]
    lines.append(f'<text x="{left}" y="{height-92}" class="note">Qwen: {qwen["approvals"]} approvals; matched delta lower 95% {float(qwen["bootstrap_delta_lower_95_bps"]):+.2f} daily bps.</text>')
    lines.append(f'<text x="{left}" y="{height-62}" class="note">Fino: {fino["approvals"]} approvals; matched delta lower 95% {float(fino["bootstrap_delta_lower_95_bps"]):+.2f} daily bps; sign-test p={float(fino["exact_sign_test_p_value"]):.3f}.</text>')
    lines.append(f'<text x="{left}" y="{height-32}" class="note">Neither AI model passed the frozen uplift gate.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _readme(
    report: Mapping[str, object],
    candidates: Sequence[Mapping[str, object]],
    capacity: Sequence[Mapping[str, object]],
    ai: Sequence[Mapping[str, object]],
) -> str:
    best = max(candidates, key=lambda item: float(item["mean_net_bps"]))
    overall = next(row for row in capacity if row["scope"] == "overall")
    fino = next(row for row in ai if row["model"] == "fino1:8b")
    qwen = next(row for row in ai if row["model"] == "qwen3:8b")
    runtime = report["runtime_evidence"]
    return f"""# Round 39: rolling ML and local AI rejected

**Monthly refits restored support, not after-cost edge.** All four BTC/ETH/SOL candidates traded across all six 2025-H1 months, but their aggregate means remained negative. Both local 8B models processed the same 180 outcome-blind offline cases without provider failures; neither improved the matched baseline.

| Evidence | Verified result |
| --- | ---: |
| Source / evaluation span | Binance USD-M 1m / 2025-01-01 to 2025-06-30 UTC |
| Rolling candidates / monthly refits / model artifacts | {len(candidates)} / 24 / {len(report["model_artifacts"])} |
| Threshold cells / selected month thresholds | 480 / 24 |
| Best full candidate | {best["candidate_id"]} |
| Best full-candidate result | {int(best["total_trades"]):,} trades; {float(best["mean_net_bps"]):+.3f} mean net bps; PF {float(best["profit_factor"]):.3f} |
| Confidence-capacity diagnostic | {int(overall["trades"])} cases; {float(overall["mean_net_bps"]):+.3f} mean; PF {float(overall["profit_factor"]):.3f}; lower 95% {float(overall["day_block_lower_95_mean_net_bps"]):+.3f} |
| Qwen3 | {qwen["approvals"]} approvals; uplift gate failed |
| Fino1 | {fino["approvals"]} approvals; {float(fino["ai_total_net_bps"]):+.1f} retained net bps; uplift lower 95% {float(fino["bootstrap_delta_lower_95_bps"]):+.2f}; failed |
| Compute / runtime / peak working set | {report["backend"]["device"]} / {runtime["elapsed_seconds"]:.1f}s / {runtime["memory"]["peak_working_set_bytes"] / 1024**3:.2f} GiB |
| Trading authority / leverage | none / none |

![Candidate economics](charts/rolling-candidate-economics.svg)

![Monthly economics](charts/monthly-economics.svg)

![Capacity concentration](charts/confidence-capacity.svg)

![AI ablation](charts/ai-ablation.svg)

![Research progress](charts/research-progress.svg)

The confidence-ranked subset is an offline diagnostic, not an accepted policy. Features and outcomes remain causal, but selecting the top cases across each completed evaluation month uses a future opportunity set that is unavailable to a live controller. Its 50,000-sample lower bound is below zero, ETH lost money, and SOL supplied most gains. Fino retained positive total economics but underperformed the stronger matched baseline; Qwen vetoed every case. No AI, ML, ROI, portfolio, leverage, execution, or profitability claim passed.

Data: [candidates.csv](candidates.csv) | [monthly.csv](monthly.csv) | [thresholds.csv](thresholds.csv) | [models.csv](models.csv) | [ai-cases.csv](ai-cases.csv) | [ai-models.csv](ai-models.csv) | [ai-decisions.csv](ai-decisions.csv) | [capacity-summary.csv](capacity-summary.csv) | [utility-uplift.csv](utility-uplift.csv) | [sources.csv](sources.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
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
) -> dict[str, object]:
    """Publish validated latest-only Round 39 tables, charts, and manifest."""

    evidence_root = evidence_root.resolve()
    report, case_set, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path.resolve(), binding_path.resolve()
    )
    candidates = _candidate_rows(report)
    monthly = _monthly_rows(report)
    thresholds = _threshold_rows(report)
    models = _model_rows(report)
    sources = _source_rows(report)
    ai_cases = _ai_case_rows(case_set)
    ai_models, ai_decisions = _ai_rows(report)
    capacity = _capacity_rows(ai_cases)
    utility = [dict(item) for item in report["utility_weighting_uplift"]]
    progress_fields, progress = _progress_rows(prior_progress_path, candidates)
    output_dir = output_dir.resolve()
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "candidates.csv",
        output_dir / "monthly.csv",
        output_dir / "thresholds.csv",
        output_dir / "models.csv",
        output_dir / "ai-cases.csv",
        output_dir / "ai-models.csv",
        output_dir / "ai-decisions.csv",
        output_dir / "capacity-summary.csv",
        output_dir / "utility-uplift.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "rolling-candidate-economics.svg",
        charts / "monthly-economics.svg",
        charts / "confidence-capacity.svg",
        charts / "ai-ablation.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    _write_csv(output_dir / "candidates.csv", candidates)
    _write_csv(output_dir / "monthly.csv", monthly)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "ai-cases.csv", ai_cases)
    _write_csv(output_dir / "ai-models.csv", ai_models)
    _write_csv(output_dir / "ai-decisions.csv", ai_decisions)
    _write_csv(output_dir / "capacity-summary.csv", capacity)
    _write_csv(output_dir / "utility-uplift.csv", utility)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "screen.json",
        (evidence_root / "report.json").read_text(encoding="utf-8"),
    )
    _write_text(output_dir / "README.md", _readme(report, candidates, capacity, ai_models))
    _write_text(charts / "rolling-candidate-economics.svg", _candidate_svg(candidates))
    _write_text(charts / "monthly-economics.svg", _monthly_svg(monthly))
    _write_text(charts / "confidence-capacity.svg", _capacity_svg(capacity))
    _write_text(charts / "ai-ablation.svg", _ai_svg(ai_models))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    overall = next(row for row in capacity if row["scope"] == "overall")
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "causal_refit_utility_ai_ablation_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_ai_case_file_sha256": _file_sha256(evidence_root / "ai-cases.json"),
        "source_implementation_commit": report["implementation_commit"],
        "dataset_rows": report["dataset"]["rows"],
        "model_feature_count": report["dataset"]["model_feature_count"],
        "gpu_model_artifact_count": len(models),
        "candidate_count": len(candidates),
        "monthly_result_count": len(monthly),
        "threshold_cell_count": len(thresholds),
        "ai_case_count": len(ai_cases),
        "ai_decision_count": len(ai_decisions),
        "ai_provider_failure_count": sum(
            int(row["provider_failures"]) for row in ai_models
        ),
        "confidence_capacity_bootstrap_samples": 50_000,
        "confidence_capacity_bootstrap_seed": 3939,
        "confidence_capacity_lower_95_mean_net_bps": overall[
            "day_block_lower_95_mean_net_bps"
        ],
        "aggregate_ml_gate_passed_candidate_count": 0,
        "ai_uplift_gate_passed_model_count": 0,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
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
        default=research / "round-039-causal-refit-utility-ai-ablation-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-039-causal-refit-utility-ai-execution-binding.json",
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
        binding_path=arguments.binding,
        prior_progress_path=arguments.prior_progress,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
