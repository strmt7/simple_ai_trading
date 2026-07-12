"""Publish verified Round 38 derivatives-hurdle ML and AI-ablation evidence."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
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
from tools.run_derivatives_hurdle_ai_ablation import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


PUBLICATION_SCHEMA = "derivatives-hurdle-ai-ablation-publication-v1"
ROUND = 38
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


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
    design = _read_object(design_path, "Round 38 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 38 design")
    binding = _read_object(binding_path, "Round 38 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 38 binding")
    report = _read_object(evidence_root / "report.json", "Round 38 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 38 report"
    )
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
        raise ValueError("Round 38 evidence lineage is invalid")
    _validate_tree(report)
    source = report.get("source_evidence")
    price = source.get("price_flow") if isinstance(source, Mapping) else None
    price_series = price.get("series_evidence") if isinstance(price, Mapping) else None
    derivatives = (
        source.get("derivatives_series") if isinstance(source, Mapping) else None
    )
    dataset = report.get("dataset")
    runtime = report.get("runtime_evidence")
    candidates = report.get("candidate_results")
    artifacts = report.get("model_artifacts")
    uplift = report.get("derivatives_feature_uplift")
    ai_cases = report.get("ai_case_set")
    if (
        not isinstance(source, Mapping)
        or source.get("selection_confirmation_or_terminal_rows_read") is not False
        or source.get("source_certificate_sha256")
        != "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
        or source.get("derivatives_panel_sha256")
        != "3e7a57184d690afc7061e9f48e437716ff73b72f01302d2eb151112351e8cf7e"
        or not isinstance(price, Mapping)
        or price.get("materialized_start") != "2021-12-01"
        or price.get("materialized_end") != "2025-06-30"
        or price.get("selection_confirmation_or_terminal_rows_read") is not False
        or price.get("panel_stream_sha256")
        != "40ef0a76d57fa844fb01ccd90d7c768f5fbbf613467303a2dc1dcdd39b100a3d"
        or not isinstance(price_series, list)
        or len(price_series) != 3
        or not isinstance(derivatives, list)
        or len(derivatives) != 3
        or {item.get("symbol") for item in derivatives if isinstance(item, Mapping)}
        != set(SYMBOLS)
        or not isinstance(dataset, Mapping)
        or dataset.get("rows") != 1_098_105
        or dataset.get("feature_count") != 103
        or dataset.get("price_flow_feature_count") != 71
        or dataset.get("derivatives_feature_count") != 32
        or dataset.get("features_dtype") != "float32"
        or dataset.get("persistent_feature_copy_created") is not False
        or not isinstance(candidates, list)
        or len(candidates) != 32
        or sum(len(item.get("threshold_trace", [])) for item in candidates) != 640
        or not isinstance(artifacts, list)
        or len(artifacts) != 96
        or not isinstance(uplift, list)
        or len(uplift) != 16
        or any(
            float(item["viability_log_loss_delta_augmented_minus_price"]) <= 0
            for item in uplift
        )
        or not isinstance(runtime, Mapping)
        or runtime.get("elapsed_seconds") != 594.3303373000235
        or runtime.get("memory", {}).get("peak_working_set_bytes")
        != 4_662_194_176
        or not isinstance(ai_cases, Mapping)
        or ai_cases.get("cases") != 0
        or report.get("viability_gate_passed_candidates") != []
        or report.get("ai_uplift_gate_passed_models") != []
        or report.get("model_selection_on_viability_permitted") is not False
        or report.get("selection_confirmation_accessed") is not False
        or report.get("terminal_2026_accessed") is not False
    ):
        raise ValueError("Round 38 source, model, or runtime evidence drifted")
    if any(
        not isinstance(candidate, Mapping)
        or candidate.get("selected_action_threshold") is None
        or int(candidate.get("viability_replay", {}).get("total_trades", 0)) < 300
        or candidate.get("viability_gate_passed") is not False
        for candidate in candidates
    ):
        raise ValueError("Round 38 candidate activity or gate evidence drifted")
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Round 38 unexpectedly grants {field}")
    return report, report_sha, binding_sha


def _candidate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        selected = candidate["selected_action_threshold"]
        cal_cls = candidate["calibration_classification"]
        val_cls = candidate["viability_classification"]
        cal = candidate["calibration_replay"]
        val = candidate["viability_replay"]
        row: dict[str, object] = {
            "candidate_id": candidate["candidate_id"],
            "architecture": candidate["architecture"],
            "feature_set": candidate["feature_set"],
            "horizon_minutes": candidate["horizon_minutes"],
            "selected_maximum_action_probability": selected[
                "maximum_action_probability"
            ],
            "selected_direction_probability_margin": selected[
                "direction_probability_margin"
            ],
            "temperature": candidate["temperature_calibration"]["temperature"],
        }
        for prefix, metrics in (("calibration", cal_cls), ("viability", val_cls)):
            for field in (
                "rows",
                "accuracy",
                "balanced_accuracy",
                "multiclass_log_loss",
                "multiclass_brier_score",
                "expected_calibration_error",
                "maximum_probability_mean",
            ):
                row[f"{prefix}_{field}"] = metrics[field]
        for prefix, replay in (("calibration", cal), ("viability", val)):
            for field in (
                "total_trades",
                "active_utc_days",
                "mean_net_bps",
                "median_monthly_net_bps",
                "profit_factor",
                "negative_month_fraction",
                "day_block_bootstrap_mean_net_bps_lower_95",
                "day_block_bootstrap_mean_net_bps_median",
                "day_block_bootstrap_mean_net_bps_upper_95",
                "maximum_peak_to_trough_drawdown_bps",
                "maximum_single_symbol_fraction",
                "mean_funding_cash_flow_bps",
            ):
                row[f"{prefix}_{field}"] = replay[field]
            for symbol in SYMBOLS:
                row[f"{prefix}_{symbol.lower()}_trades"] = replay[
                    "trades_by_symbol"
                ][symbol]
        row["viability_gate_passed"] = candidate["viability_gate_passed"]
        row["viability_gate_reasons"] = ";".join(
            str(item) for item in candidate["viability_gate_reasons"]
        )
        rows.append(row)
    return rows


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        selected = candidate["selected_action_threshold"]
        for trace in candidate["threshold_trace"]:
            replay = trace["replay"]
            row = {
                "candidate_id": candidate["candidate_id"],
                "architecture": candidate["architecture"],
                "feature_set": candidate["feature_set"],
                "horizon_minutes": candidate["horizon_minutes"],
                "maximum_action_probability": trace["maximum_action_probability"],
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
            }
            for field in (
                "candidate_rows",
                "total_trades",
                "active_utc_days",
                "mean_net_bps",
                "median_monthly_net_bps",
                "profit_factor",
                "negative_month_fraction",
                "day_block_bootstrap_mean_net_bps_lower_95",
                "day_block_bootstrap_mean_net_bps_median",
                "day_block_bootstrap_mean_net_bps_upper_95",
                "maximum_peak_to_trough_drawdown_bps",
                "maximum_single_symbol_fraction",
                "mean_funding_cash_flow_bps",
            ):
                row[field] = replay[field]
            for symbol in SYMBOLS:
                row[f"{symbol.lower()}_trades"] = replay["trades_by_symbol"][symbol]
            rows.append(row)
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    fields = (
        "model_id",
        "candidate_id",
        "architecture",
        "target_head",
        "symbol",
        "feature_set",
        "horizon_minutes",
        "training_rows",
        "early_stop_rows",
        "feature_count",
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
                "premium_maximum_gap_minutes": derivatives[
                    "premium_maximum_gap_minutes"
                ],
                "premium_stream_sha256": derivatives["premium_stream_sha256"],
                "funding_rows": derivatives["funding_rows"],
                "funding_minimum_interval_hours": derivatives[
                    "funding_minimum_interval_hours"
                ],
                "funding_maximum_interval_hours": derivatives[
                    "funding_maximum_interval_hours"
                ],
                "funding_stream_sha256": derivatives["funding_stream_sha256"],
            }
        )
    return rows


def _progress_rows(path: Path, candidates: Sequence[Mapping[str, object]]) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed = [int(row["round"]) for row in rows]
    if not fields or observed not in (list(range(1, 38)), list(range(1, 39))):
        raise ValueError("Round 38 prior progress history is invalid")
    rows = [row for row in rows if int(row["round"]) != ROUND]
    best = max(candidates, key=lambda item: float(item["viability_mean_net_bps"]))
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "derivatives-aware direct and hurdle classification",
            "periods": "2022-01-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "900;1800;3600;7200",
            "feature_set": "71 price/flow plus 32 premium/funding features",
            "risk_level": "research-only; no policy",
            "selected_signals": best["viability_total_trades"],
            "executable_trades": best["viability_total_trades"],
            "mean_net_bps": best["viability_mean_net_bps"],
            "status": "rejected",
            "source_file": "verified Round 38 derivatives-hurdle report",
            "best_model_id": best["candidate_id"],
            "daily_model_fits": 96,
            "calibration_threshold_traces": 640,
            "accepted_thresholds": 0,
            "ensemble_models": 96,
            "calibration_eligible_rows": 79_479,
            "development_consumed": True,
        }
    )
    rows.append(row)
    return fields, rows


def _short_label(row: Mapping[str, object]) -> str:
    architecture = str(row["architecture"])
    scope = "shared" if architecture.startswith("shared") else "symbol"
    model = "hurdle" if "hurdle" in architecture else "direct"
    features = "price+derivatives" if "plus" in str(row["feature_set"]) else "price"
    return f"{scope} {model} / {features} / {row['horizon_minutes']}m"


def _viability_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = sorted(rows, key=lambda item: float(item["viability_mean_net_bps"]), reverse=True)[:16]
    width, height = 1560, 1020
    left, right, top, row_height = 430, 240, 132, 46
    chart_width = width - left - right
    lower = min(float(row["viability_day_block_bootstrap_mean_net_bps_lower_95"]) for row in selected)
    upper = max(float(row["viability_day_block_bootstrap_mean_net_bps_upper_95"]) for row in selected)
    low, high = min(-28.0, lower - 2.0), max(5.0, upper + 2.0)

    def x(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Round 38 restored activity but not after-cost edge",
        "Best 16 of 32 candidates on unseen 2025-H1; interval is the day-block bootstrap mean net bps.",
    )
    zero = x(0.0)
    tick_start = int(low // 5) * 5
    tick_end = int(high // 5) * 5
    for tick in range(tick_start, tick_end + 1, 5):
        if tick < low or tick > high:
            continue
        px = x(float(tick))
        lines.append(
            f'<line x1="{px:.1f}" y1="{top-18}" x2="{px:.1f}" '
            f'y2="{top+row_height*len(selected)}" class="grid"/>'
        )
        lines.append(
            f'<text x="{px:.1f}" y="{top+row_height*len(selected)+27}" '
            f'text-anchor="middle" class="axis">{tick:+d}</text>'
        )
    lines.append(f'<line x1="{zero:.1f}" y1="{top-18}" x2="{zero:.1f}" y2="{top+row_height*len(selected)}" class="zero"/>')
    for index, row in enumerate(selected):
        y = top + index * row_height
        lo = float(row["viability_day_block_bootstrap_mean_net_bps_lower_95"])
        mean = float(row["viability_mean_net_bps"])
        hi = float(row["viability_day_block_bootstrap_mean_net_bps_upper_95"])
        lines.append(f'<text x="{left-18}" y="{y+5}" text-anchor="end" class="label">{html.escape(_short_label(row))}</text>')
        lines.append(f'<line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" stroke="#94a3b8" stroke-width="7" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="7" fill="#b42318"/>')
        lines.append(f'<text x="{width-64}" y="{y+5}" text-anchor="end" class="value">{mean:+.2f} bps; n={int(row["viability_total_trades"]):,}</text>')
    lines.append(f'<text x="{left+chart_width/2:.1f}" y="{height-72}" text-anchor="middle" class="label">Mean net basis points per trade (2025-H1)</text>')
    lines.append(f'<text x="{left}" y="{height-38}" class="note">Dot: mean net bps after the frozen 12 bps charge and exact funding cash flow. All 32 candidates failed viability.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _calibration_shift_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1180, 860
    left, right, top, bottom = 120, 80, 130, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    values = [
        float(row[field])
        for row in rows
        for field in ("calibration_mean_net_bps", "viability_mean_net_bps")
    ]
    low, high = min(values) - 2.0, max(values) + 2.0

    def x(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Calibration gains did not persist",
        "Each point is one selected candidate threshold: 2024-Q4 calibration versus unseen 2025-H1 mean net bps.",
    )
    lines.append(f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>')
    tick_start = int(low // 5) * 5
    tick_end = int(high // 5) * 5
    for tick in range(tick_start, tick_end + 1, 5):
        if tick < low or tick > high:
            continue
        px, py = x(float(tick)), y(float(tick))
        lines.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top+chart_height}" class="grid"/>')
        lines.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left+chart_width}" y2="{py:.1f}" class="grid"/>')
        lines.append(f'<text x="{px:.1f}" y="{top+chart_height+25}" text-anchor="middle" class="axis">{tick:+d}</text>')
        lines.append(f'<text x="{left-14}" y="{py+4:.1f}" text-anchor="end" class="axis">{tick:+d}</text>')
    lines.append(f'<line x1="{x(low):.1f}" y1="{y(low):.1f}" x2="{x(high):.1f}" y2="{y(high):.1f}" stroke="#64748b" stroke-width="2" stroke-dasharray="7 6"/>')
    lines.append(f'<line x1="{x(0):.1f}" y1="{top}" x2="{x(0):.1f}" y2="{top+chart_height}" class="zero"/>')
    lines.append(f'<line x1="{left}" y1="{y(0):.1f}" x2="{left+chart_width}" y2="{y(0):.1f}" class="zero"/>')
    colors = {"price_flow_only": "#2563eb", "price_flow_plus_premium_and_funding": "#b45309"}
    for row in rows:
        cx = x(float(row["calibration_mean_net_bps"]))
        cy = y(float(row["viability_mean_net_bps"]))
        color = colors[str(row["feature_set"])]
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{color}" fill-opacity="0.78" stroke="#ffffff" stroke-width="1.5"><title>{html.escape(_short_label(row))}</title></circle>')
    lines.append(f'<text x="{left+chart_width/2:.1f}" y="{height-38}" text-anchor="middle" class="label">Calibration mean net bps (2024-Q4)</text>')
    lines.append(f'<text x="34" y="{top+chart_height/2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 34 {top+chart_height/2:.1f})">Viability mean net bps (2025-H1)</text>')
    lines.append('<circle cx="780" cy="52" r="7" fill="#2563eb"/><text x="796" y="57" class="note">price/flow</text>')
    lines.append('<circle cx="900" cy="52" r="7" fill="#b45309"/><text x="916" y="57" class="note">price/flow + derivatives</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _uplift_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = sorted(rows, key=lambda item: (str(item["architecture"]), int(item["horizon_minutes"])))
    width, height = 1480, 890
    left, right, top, row_height = 420, 150, 128, 43
    chart_width = width - left - right
    high = max(float(row["viability_log_loss_delta_augmented_minus_price"]) for row in selected) * 1.15
    lines = _svg_start(
        width,
        height,
        "Premium and funding features worsened predictive loss",
        "Matched 2025-H1 multiclass log-loss delta: augmented minus price/flow; positive is worse.",
    )
    for index, row in enumerate(selected):
        y = top + index * row_height
        value = float(row["viability_log_loss_delta_augmented_minus_price"])
        bar = chart_width * value / high
        architecture = str(row["architecture"]).replace("_lightgbm", "").replace("_", " ")
        label = f"{architecture} / {row['horizon_minutes']}m"
        lines.append(f'<text x="{left-18}" y="{y+18}" text-anchor="end" class="label">{html.escape(label)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="25" fill="#b45309"/>')
        lines.append(f'<text x="{left+bar+10:.1f}" y="{y+18}" class="value">+{value:.6f}</text>')
    lines.append(f'<text x="{left}" y="{height-34}" class="note">All 16 matched deltas are positive. Derivatives data remains retained for funding cash flows and risk state, not accepted directional alpha.</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _readme(report: Mapping[str, object], candidates: Sequence[Mapping[str, object]]) -> str:
    best = max(candidates, key=lambda item: float(item["viability_mean_net_bps"]))
    runtime = report["runtime_evidence"]
    return f"""# Round 38: activity restored, economic edge rejected

**The derivatives-hurdle lane traded often enough, but did not earn permission to trade.** Every frozen candidate lost money after the 12 bps execution charge and exact funding cash flows on unseen 2025-H1. Premium and funding features worsened multiclass log loss in all 16 matched ablations.

| Evidence | Verified result |
| --- | ---: |
| Source / target span | Binance USD-M 1m / 2022-01-01 to 2025-06-30 UTC |
| Decision rows / causal features | {report["dataset"]["rows"]:,} / {report["dataset"]["feature_count"]} |
| GPU artifacts / candidates / threshold cells | {len(report["model_artifacts"])} / {len(candidates)} / {sum(len(item["threshold_trace"]) for item in report["candidate_results"])} |
| Viability activity range | {min(int(item["viability_total_trades"]) for item in candidates):,} to {max(int(item["viability_total_trades"]) for item in candidates):,} trades |
| Best 2025-H1 candidate | {best["candidate_id"]} |
| Best viability result | {int(best["viability_total_trades"]):,} trades; {float(best["viability_mean_net_bps"]):+.3f} mean net bps; PF {float(best["viability_profit_factor"]):.3f} |
| Best day-block lower 95% bound | {float(best["viability_day_block_bootstrap_mean_net_bps_lower_95"]):+.3f} net bps |
| Derivatives log-loss improvements | 0 / 16 |
| AI cases / local models called | {report["ai_case_set"]["cases"]} / {len(report["ai_reports"])} |
| Compute / runtime / peak working set | {report["backend"]["device"]} / {runtime["elapsed_seconds"]:.1f}s / {runtime["memory"]["peak_working_set_bytes"] / 1024**3:.2f} GiB |
| Trading authority / leverage | none / none |

![Viability economics](charts/viability-economics.svg)

![Calibration to viability](charts/calibration-to-viability.svg)

![Derivatives feature ablation](charts/derivatives-feature-ablation.svg)

![Research progress](charts/research-progress.svg)

These are independent, non-overlapping trade replays, not a capital-constrained portfolio, ROI series, or equity curve. No ROI graph is published because the experiment did not produce one. Qwen3 and Fino1 were not called because no ML candidate passed the frozen viability gate; inventing AI cases would invalidate the ablation.

The next experiment addresses the observed regime decay with causal monthly refits. It does not weaken cost, support, diversification, or confidence gates. Selection-confirmation 2025-H2 and terminal 2026 data remain sealed.

Data: [candidates.csv](candidates.csv) | [thresholds.csv](thresholds.csv) | [models.csv](models.csv) | [derivatives-uplift.csv](derivatives-uplift.csv) | [sources.csv](sources.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
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
    """Publish validated latest-only Round 38 tables, charts, and manifest."""

    evidence_root = evidence_root.resolve()
    report, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path.resolve(), binding_path.resolve()
    )
    candidates = _candidate_rows(report)
    thresholds = _threshold_rows(report)
    models = _model_rows(report)
    sources = _source_rows(report)
    uplift = [dict(item) for item in report["derivatives_feature_uplift"]]
    progress_fields, progress = _progress_rows(prior_progress_path, candidates)
    output_dir = output_dir.resolve()
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "candidates.csv",
        output_dir / "thresholds.csv",
        output_dir / "models.csv",
        output_dir / "derivatives-uplift.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "viability-economics.svg",
        charts / "calibration-to-viability.svg",
        charts / "derivatives-feature-ablation.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    _write_csv(output_dir / "candidates.csv", candidates)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "derivatives-uplift.csv", uplift)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "screen.json",
        (evidence_root / "report.json").read_text(encoding="utf-8"),
    )
    _write_text(output_dir / "README.md", _readme(report, candidates))
    _write_text(charts / "viability-economics.svg", _viability_svg(candidates))
    _write_text(
        charts / "calibration-to-viability.svg", _calibration_shift_svg(candidates)
    )
    _write_text(
        charts / "derivatives-feature-ablation.svg", _uplift_svg(uplift)
    )
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "derivatives_hurdle_ai_ablation_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "price_panel_stream_sha256": report["source_evidence"]["price_flow"][
            "panel_stream_sha256"
        ],
        "derivatives_panel_sha256": report["source_evidence"][
            "derivatives_panel_sha256"
        ],
        "dataset_rows": report["dataset"]["rows"],
        "feature_count": report["dataset"]["feature_count"],
        "gpu_model_artifact_count": len(models),
        "candidate_count": len(candidates),
        "threshold_cell_count": len(thresholds),
        "supported_threshold_cell_count": sum(
            bool(item["support_passed"]) for item in thresholds
        ),
        "selected_threshold_count": len(candidates),
        "viability_gate_passed_candidate_count": 0,
        "ai_case_count": 0,
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
        default=research / "round-038-derivatives-hurdle-ai-ablation-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-038-derivatives-hurdle-ai-execution-binding.json",
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
