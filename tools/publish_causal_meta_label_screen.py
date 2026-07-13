"""Publish verified Round 40 causal meta-label development evidence."""

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
from tools.run_causal_meta_label_capacity_ai import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


PUBLICATION_SCHEMA = "causal-meta-label-capacity-publication-v1"
ROUND = 40
MONTHS = tuple(f"2024-{month:02d}" for month in range(7, 13))
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


def _canonical_identity(
    value: Mapping[str, object], field: str, label: str
) -> str:
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
    design = _read_object(design_path, "Round 40 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 40 design")
    binding = _read_object(binding_path, "Round 40 binding")
    binding_sha = _canonical_identity(
        binding, "binding_sha256", "Round 40 binding"
    )
    report = _read_object(evidence_root / "report.json", "Round 40 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 40 report"
    )
    cases = _read_object(evidence_root / "ai-cases.json", "Round 40 AI case set")
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
        raise ValueError("Round 40 evidence lineage is invalid")
    _validate_tree(report)
    source = report.get("source_evidence")
    price = source.get("price_flow") if isinstance(source, Mapping) else None
    dataset = report.get("dataset")
    candidate = report.get("candidate_result")
    models = report.get("model_artifacts")
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
        or dataset.get("meta_feature_count") != 81
        or dataset.get("features_dtype") != "float32"
        or dataset.get("persistent_feature_copy_created") is not False
        or not isinstance(candidate, Mapping)
        or candidate.get("selected_threshold_months") != 1
        or candidate.get("months_with_trades") != 1
        or candidate.get("support_passed") is not False
        or candidate.get("aggregate_gate_passed") is not False
        or len(candidate.get("monthly_results", [])) != 6
        or sum(
            len(month.get("threshold_trace", []))
            for month in candidate.get("monthly_results", [])
        )
        != 216
        or not isinstance(models, list)
        or len(models) != 24
        or any(model.get("reload_max_abs_prediction_error") != 0.0 for model in models)
        or report.get("aggregate_ml_gate_passed") is not False
        or report.get("ai_report") is not None
        or report.get("ai_error") is not None
        or report.get("ai_uplift_gate_passed") is not False
        or report.get("selection_contaminated") is not True
        or report.get("development_only") is not True
        or report.get("selection_confirmation_accessed") is not False
        or report.get("terminal_2026_accessed") is not False
        or cases.get("schema_version")
        != "causal-meta-label-ai-veto-case-set-v1"
        or cases.get("cases") != []
    ):
        raise ValueError("Round 40 source or model evidence drifted")
    aggregate = candidate["aggregate_replay"]
    if (
        aggregate.get("total_trades") != 70
        or abs(float(aggregate["mean_net_bps"]) - 36.060967821734295) > 1e-12
        or abs(float(aggregate["profit_factor"]) - 2.008035067071759) > 1e-12
        or abs(
            float(aggregate["day_block_bootstrap_mean_net_bps_lower_95"])
            - (-4.307041800475944)
        )
        > 1e-12
    ):
        raise ValueError("Round 40 aggregate economics drifted")
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Round 40 report unexpectedly claims {field}")
    return report, report_sha, binding_sha


def _best_supported(
    traces: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    supported = [item for item in traces if item.get("support_passed") is True]
    if not supported:
        return None

    def key(item: Mapping[str, object]) -> tuple[float, float]:
        replay = item["capacity_replay"]["replay"]
        lower = replay["day_block_bootstrap_mean_net_bps_lower_95"]
        return (
            float(lower) if lower is not None else float("-inf"),
            float(replay["mean_net_bps"]),
        )

    return max(supported, key=key)


def _monthly_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in report["candidate_result"]["monthly_results"]:
        best = _best_supported(month["threshold_trace"])
        best_replay = best["capacity_replay"]["replay"] if best else {}
        evaluation = month["evaluation_replay"]["replay"]
        selected = month["selected_threshold"]
        rows.append(
            {
                "evaluation_month": month["evaluation_month"],
                "threshold_selected": selected is not None,
                "selected_meta_probability": (
                    selected["meta_probability_threshold"] if selected else None
                ),
                "selected_primary_margin": (
                    selected["primary_margin_threshold"] if selected else None
                ),
                "supported_threshold_cells": sum(
                    item["support_passed"] for item in month["threshold_trace"]
                ),
                "economic_threshold_cells": sum(
                    item["economic_gate_passed"]
                    for item in month["threshold_trace"]
                ),
                "best_supported_calibration_trades": best_replay.get("total_trades"),
                "best_supported_calibration_mean_net_bps": best_replay.get(
                    "mean_net_bps"
                ),
                "best_supported_calibration_profit_factor": best_replay.get(
                    "profit_factor"
                ),
                "best_supported_calibration_lower_95_mean_net_bps": (
                    best_replay.get("day_block_bootstrap_mean_net_bps_lower_95")
                ),
                "meta_fit_roc_auc": month["meta_classification"]["fit"]["roc_auc"],
                "meta_early_stop_roc_auc": month["meta_classification"][
                    "early_stop"
                ]["roc_auc"],
                "meta_calibration_roc_auc": month["meta_classification"][
                    "threshold_calibration"
                ]["roc_auc"],
                "meta_evaluation_roc_auc": month["meta_classification"][
                    "evaluation"
                ]["roc_auc"],
                "evaluation_trades": evaluation["total_trades"],
                "evaluation_active_days": evaluation["active_utc_days"],
                "evaluation_mean_net_bps": evaluation["mean_net_bps"],
                "evaluation_profit_factor": evaluation["profit_factor"],
                "evaluation_lower_95_mean_net_bps": evaluation[
                    "day_block_bootstrap_mean_net_bps_lower_95"
                ],
                "btcusdt_trades": evaluation["trades_by_symbol"]["BTCUSDT"],
                "ethusdt_trades": evaluation["trades_by_symbol"]["ETHUSDT"],
                "solusdt_trades": evaluation["trades_by_symbol"]["SOLUSDT"],
                "capacity_rejections": month["evaluation_replay"][
                    "capacity_rejections"
                ],
                "overlap_rejections": month["evaluation_replay"][
                    "overlap_rejections"
                ],
            }
        )
    return rows


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for month in report["candidate_result"]["monthly_results"]:
        selected = month["selected_threshold"]
        for item in month["threshold_trace"]:
            replay = item["capacity_replay"]["replay"]
            rows.append(
                {
                    "evaluation_month": month["evaluation_month"],
                    "meta_probability_threshold": item[
                        "meta_probability_threshold"
                    ],
                    "primary_margin_threshold": item["primary_margin_threshold"],
                    "selected": bool(
                        selected
                        and selected["meta_probability_threshold"]
                        == item["meta_probability_threshold"]
                        and selected["primary_margin_threshold"]
                        == item["primary_margin_threshold"]
                    ),
                    "support_passed": item["support_passed"],
                    "economic_gate_passed": item["economic_gate_passed"],
                    "threshold_candidate_rows": item["capacity_replay"][
                        "threshold_candidate_rows"
                    ],
                    "overlap_rejections": item["capacity_replay"][
                        "overlap_rejections"
                    ],
                    "capacity_rejections": item["capacity_replay"][
                        "capacity_rejections"
                    ],
                    "total_trades": replay["total_trades"],
                    "active_utc_days": replay["active_utc_days"],
                    "btcusdt_trades": replay["trades_by_symbol"]["BTCUSDT"],
                    "ethusdt_trades": replay["trades_by_symbol"]["ETHUSDT"],
                    "solusdt_trades": replay["trades_by_symbol"]["SOLUSDT"],
                    "mean_net_bps": replay["mean_net_bps"],
                    "profit_factor": replay["profit_factor"],
                    "day_block_lower_95_mean_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_lower_95"
                    ],
                    "day_block_median_mean_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_median"
                    ],
                    "day_block_upper_95_mean_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_upper_95"
                    ],
                }
            )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            key: value
            for key, value in model.items()
            if key not in {"path", "top_feature_gain"}
        }
        for model in report["model_artifacts"]
    ]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    price_by_symbol = {
        item["symbol"]: item
        for item in report["source_evidence"]["price_flow"]["series_evidence"]
    }
    rows: list[dict[str, object]] = []
    for derivative in report["source_evidence"]["derivatives_series"]:
        price = price_by_symbol[derivative["symbol"]]
        rows.append(
            {
                "symbol": derivative["symbol"],
                "price_rows": price["rows"],
                "price_gap_count": price["gap_count"],
                "price_stream_sha256": price["stream_sha256"],
                "premium_rows": derivative["premium_rows"],
                "premium_gap_events": derivative["premium_gap_events"],
                "premium_missing_minutes": derivative["premium_missing_minutes"],
                "premium_stream_sha256": derivative["premium_stream_sha256"],
                "funding_rows": derivative["funding_rows"],
                "funding_stream_sha256": derivative["funding_stream_sha256"],
            }
        )
    return rows


def _progress_rows(
    path: Path, report: Mapping[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed = [int(row["round"]) for row in rows]
    if not fields or observed not in (list(range(1, 40)), list(range(1, 41))):
        raise ValueError("Round 40 prior progress history is invalid")
    rows = [row for row in rows if int(row["round"]) != ROUND]
    aggregate = report["candidate_result"]["aggregate_replay"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "causal stacked profitability meta-label and capacity control",
            "periods": "2023-01-01..2024-12-31 rolling roles",
            "selection_contaminated": True,
            "horizon_seconds": "1800",
            "feature_set": "71 primary + 10 stacked causal meta features",
            "risk_level": "consumed development only; no policy",
            "selected_signals": aggregate["total_trades"],
            "executable_trades": aggregate["total_trades"],
            "mean_net_bps": aggregate["mean_net_bps"],
            "status": "rejected",
            "source_file": "verified Round 40 causal meta-label report",
            "best_model_id": report["candidate_result"]["candidate_id"],
            "daily_model_fits": 24,
            "calibration_threshold_traces": 216,
            "accepted_thresholds": 0,
            "ensemble_models": 24,
            "development_consumed": True,
        }
    )
    rows.append(row)
    return fields, rows


def _auc_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1480, 650
    left, right, top, chart_height = 110, 70, 140, 360
    chart_width = width - left - right
    low, high = 0.50, 0.80
    series = (
        ("meta_fit_roc_auc", "meta fit (in-sample)", "#b45309", "8 6"),
        ("meta_early_stop_roc_auc", "meta early stop", "#7b559c", ""),
        ("meta_calibration_roc_auc", "threshold calibration", "#0f766e", ""),
        ("meta_evaluation_roc_auc", "evaluation", "#2563eb", ""),
    )

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Meta-label discrimination persisted, but one-month fitting overfit",
        "Binary profitability ROC AUC by causal role; fit is in-sample to the meta model while later roles are unseen.",
    )
    for tick in (0.50, 0.60, 0.70, 0.80):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left+chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0.5 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left-14}" y="{py+4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    for key, _, color, dash in series:
        points = []
        for index, row in enumerate(rows):
            x = left + chart_width * (index + 0.5) / len(rows)
            points.append((x, y(float(row[key]))))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{py:.1f}" for x, py in points)
            + f'" fill="none" stroke="{color}" stroke-width="4"{dash_attr}/>'
        )
        lines.extend(
            f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>'
            for x, py in points
        )
    for index, month in enumerate(MONTHS):
        x = left + chart_width * (index + 0.5) / len(rows)
        lines.append(
            f'<text x="{x:.1f}" y="{top+chart_height+30}" text-anchor="middle" class="axis">{month}</text>'
        )
    for index, (_, label, color, dash) in enumerate(series):
        x = 120 + index * 310
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{x}" y1="570" x2="{x+28}" y2="570" stroke="{color}" stroke-width="4"{dash_attr}/><text x="{x+38}" y="575" class="note">{html.escape(label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _calibration_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1480, 610
    left, right, top, chart_height = 110, 70, 145, 300
    chart_width = width - left - right
    low, high = -25.0, 55.0

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Only December calibration cleared uncertainty",
        "Best support-complete threshold per month; intervals exist only when positive mean and PF triggered bootstrap evaluation.",
    )
    for tick in (-20, 0, 20, 40):
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left+chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left-14}" y="{py+4:.1f}" text-anchor="end" class="axis">{tick:+d}</text>'
        )
    for index, row in enumerate(rows):
        x = left + chart_width * (index + 0.5) / len(rows)
        mean = float(row["best_supported_calibration_mean_net_bps"])
        lower = row["best_supported_calibration_lower_95_mean_net_bps"]
        if lower is not None:
            lines.append(
                f'<line x1="{x:.1f}" y1="{y(float(lower)):.1f}" x2="{x:.1f}" y2="{y(mean):.1f}" stroke="#94a3b8" stroke-width="7" stroke-linecap="round"/>'
            )
        color = "#0f766e" if int(row["economic_threshold_cells"]) else "#b42318"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y(mean):.1f}" r="8" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y(mean)-15:.1f}" text-anchor="middle" class="value">{mean:+.1f}</text>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top+chart_height+30}" text-anchor="middle" class="axis">{row["evaluation_month"]}</text>'
        )
    lines.append(
        '<text x="110" y="535" class="note">Red: no threshold passed the frozen economic gate. Teal: at least one passed. Values are calibration mean net bps, not evaluation ROI.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _evaluation_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1480, 560
    left, right, top, chart_height = 110, 70, 145, 250
    chart_width = width - left - right
    low, high = -10.0, 50.0

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Five months remained flat; December was positive but unsupported",
        "Evaluation mean net bps after the frozen 12 bps charge; crosses denote no threshold and therefore no trade series.",
    )
    zero = y(0.0)
    lines.append(
        f'<line x1="{left}" y1="{zero:.1f}" x2="{left+chart_width}" y2="{zero:.1f}" class="zero"/>'
    )
    for index, row in enumerate(rows):
        x = left + chart_width * (index + 0.5) / len(rows)
        if int(row["evaluation_trades"]) == 0:
            lines.append(
                f'<line x1="{x-8:.1f}" y1="{zero-8:.1f}" x2="{x+8:.1f}" y2="{zero+8:.1f}" stroke="#60717f" stroke-width="3"/><line x1="{x-8:.1f}" y1="{zero+8:.1f}" x2="{x+8:.1f}" y2="{zero-8:.1f}" stroke="#60717f" stroke-width="3"/>'
            )
        else:
            value = float(row["evaluation_mean_net_bps"])
            lines.append(
                f'<rect x="{x-42:.1f}" y="{y(value):.1f}" width="84" height="{zero-y(value):.1f}" fill="#0f766e"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{y(value)-15:.1f}" text-anchor="middle" class="value">{value:+.2f} bps; n={int(row["evaluation_trades"])}</text>'
            )
        lines.append(
            f'<text x="{x:.1f}" y="{top+chart_height+32}" text-anchor="middle" class="axis">{row["evaluation_month"]}</text>'
        )
    lines.append(
        '<text x="110" y="505" class="note">Aggregate across selected actions: +36.061 mean bps, PF 2.008, day-block lower 95% -4.307. No portfolio, ROI, or leverage simulation.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _readme(report: Mapping[str, object], monthly: Sequence[Mapping[str, object]]) -> str:
    aggregate = report["candidate_result"]["aggregate_replay"]
    auc = [float(row["meta_evaluation_roc_auc"]) for row in monthly]
    runtime = report["runtime_evidence"]
    return f"""# Round 40: causal meta-label screen rejected

**The stacked model found weak repeatable discrimination, but not a six-month policy.** All 24 OpenCL LightGBM artifacts reloaded exactly. Meta-label evaluation AUC stayed above chance in every month, yet only December's prior-month threshold cleared the frozen economic and uncertainty gates. Five months correctly remained flat.

| Evidence | Verified result |
| --- | ---: |
| Source / evaluation span | Binance USD-M 1m / 2024-07-01 to 2024-12-31 UTC |
| Primary / meta GPU artifacts | 18 / 6 |
| Threshold cells / months selected | 216 / 1 of 6 |
| Meta evaluation AUC range | {min(auc):.3f} to {max(auc):.3f} |
| Selected evaluation actions | {int(aggregate["total_trades"])} ({aggregate["trades_by_symbol"]["BTCUSDT"]}/{aggregate["trades_by_symbol"]["ETHUSDT"]}/{aggregate["trades_by_symbol"]["SOLUSDT"]} BTC/ETH/SOL) |
| Conditional action result | {float(aggregate["mean_net_bps"]):+.3f} mean net bps; PF {float(aggregate["profit_factor"]):.3f} |
| Six-month day-block lower 95% | {float(aggregate["day_block_bootstrap_mean_net_bps_lower_95"]):+.3f} bps |
| AI cases / AI models run | 0 / 0; ML gate failed first |
| Compute / runtime / peak working set | {report["backend"]["device"]} / {runtime["elapsed_seconds"]:.1f}s / {runtime["memory"]["peak_working_set_bytes"] / 1024**3:.2f} GiB |
| Trading authority / leverage | none / none |

![Meta-label AUC](charts/meta-label-auc.svg)

![Calibration economics](charts/calibration-economics.svg)

![Evaluation activity](charts/evaluation-activity.svg)

![Research progress](charts/research-progress.svg)

The `+36.061` bps action mean is not a profitability or ROI claim. It comes from 70 December actions after five zero-action months, its stationary day-block lower bound is negative, and the repeated development period is selection-contaminated. The in-sample meta-fit AUC was materially higher than every later role, which identifies short meta training as the next defect to address. Selection-confirmation 2025-H2 and terminal 2026 remain sealed.

Data: [candidate.csv](candidate.csv) | [monthly.csv](monthly.csv) | [thresholds.csv](thresholds.csv) | [models.csv](models.csv) | [sources.csv](sources.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
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
    report, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path, binding_path
    )
    monthly = _monthly_rows(report)
    thresholds = _threshold_rows(report)
    models = _model_rows(report)
    sources = _source_rows(report)
    progress_fields, progress = _progress_rows(prior_progress_path, report)
    candidate = [{
        key: value
        for key, value in report["candidate_result"].items()
        if key != "monthly_results"
    }]
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "candidate.csv",
        output_dir / "monthly.csv",
        output_dir / "thresholds.csv",
        output_dir / "models.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "meta-label-auc.svg",
        charts / "calibration-economics.svg",
        charts / "evaluation-activity.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate.csv", candidate)
    _write_csv(output_dir / "monthly.csv", monthly)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress, fields=progress_fields)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report, monthly))
    _write_text(charts / "meta-label-auc.svg", _auc_svg(monthly))
    _write_text(charts / "calibration-economics.svg", _calibration_svg(monthly))
    _write_text(charts / "evaluation-activity.svg", _evaluation_svg(monthly))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "causal_meta_label_capacity_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "dataset_rows": report["dataset"]["rows"],
        "primary_feature_count": report["dataset"]["model_feature_count"],
        "meta_feature_count": report["dataset"]["meta_feature_count"],
        "gpu_model_artifact_count": len(models),
        "threshold_cell_count": len(thresholds),
        "monthly_result_count": len(monthly),
        "selected_threshold_month_count": sum(
            bool(row["threshold_selected"]) for row in monthly
        ),
        "aggregate_trade_count": report["candidate_result"]["aggregate_replay"][
            "total_trades"
        ],
        "aggregate_mean_net_bps": report["candidate_result"]["aggregate_replay"][
            "mean_net_bps"
        ],
        "aggregate_day_block_lower_95_mean_net_bps": report[
            "candidate_result"
        ]["aggregate_replay"]["day_block_bootstrap_mean_net_bps_lower_95"],
        "ai_case_count": 0,
        "selection_contaminated": True,
        "development_only": True,
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
        default=research / "round-040-causal-meta-label-capacity-ai-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-040-causal-meta-label-execution-binding.json",
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
