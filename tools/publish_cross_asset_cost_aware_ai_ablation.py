"""Publish verified Round 37 cross-asset ML and AI-ablation evidence."""

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

from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_cross_asset_cost_aware_ai_ablation import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


PUBLICATION_SCHEMA = "cross-asset-cost-aware-ai-ablation-publication-v1"
ROUND = 37
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS = (15, 30, 60, 120)


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
    design = _read_object(design_path, "Round 37 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 37 design")
    binding = _read_object(binding_path, "Round 37 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 37 binding")
    report = _read_object(evidence_root / "report.json", "Round 37 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 37 report"
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
        or report.get("implementation_commit") != binding.get("implementation_commit")
    ):
        raise ValueError("Round 37 evidence lineage is invalid")
    _validate_tree(report)
    source = report.get("source_evidence")
    dataset = report.get("dataset")
    runtime = report.get("runtime_evidence")
    memory = runtime.get("memory") if isinstance(runtime, Mapping) else None
    series = source.get("series_evidence") if isinstance(source, Mapping) else None
    candidates = report.get("candidate_results")
    artifacts = report.get("model_artifacts")
    ai_cases = report.get("ai_case_set")
    if (
        not isinstance(source, Mapping)
        or source.get("materialized_start") != "2021-12-01"
        or source.get("materialized_end") != "2025-06-30"
        or source.get("selection_confirmation_or_terminal_rows_read") is not False
        or source.get("panel_stream_sha256")
        != "40ef0a76d57fa844fb01ccd90d7c768f5fbbf613467303a2dc1dcdd39b100a3d"
        or not isinstance(series, list)
        or len(series) != 3
        or {item.get("symbol") for item in series if isinstance(item, Mapping)}
        != set(SYMBOLS)
        or any(
            not isinstance(item, Mapping)
            or item.get("rows") != 1_883_520
            or item.get("gap_count") != 0
            or item.get("duplicate_or_regressed_time_count") != 0
            or item.get("nonfinite_numeric_rows") != 0
            or item.get("invalid_ohlc_rows") != 0
            for item in series
        )
        or not isinstance(dataset, Mapping)
        or dataset.get("rows") != 1_103_328
        or dataset.get("feature_count") != 71
        or dataset.get("features_dtype") != "float32"
        or dataset.get("persistent_feature_copy_created") is not False
        or not isinstance(candidates, list)
        or len(candidates) != 20
        or sum(len(item.get("threshold_trace", [])) for item in candidates) != 100
        or not isinstance(artifacts, list)
        or len(artifacts) != 16
        or not isinstance(runtime, Mapping)
        or not isinstance(memory, Mapping)
        or memory.get("peak_working_set_bytes") != 3_557_158_912
        or not isinstance(ai_cases, Mapping)
        or ai_cases.get("cases") != 0
        or report.get("viability_gate_passed_candidates") != []
        or report.get("ai_uplift_gate_passed_models") != []
        or report.get("model_selection_on_viability_permitted") is not False
        or report.get("selection_confirmation_accessed") is not False
        or report.get("terminal_2026_accessed") is not False
    ):
        raise ValueError("Round 37 source, model, or runtime evidence drifted")
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Round 37 unexpectedly grants {field}")
    for candidate in candidates:
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("selected_threshold_bps") is not None
            or candidate.get("viability_gate_passed") is not False
            or candidate.get("calibration_replay", {}).get("total_trades") != 0
            or candidate.get("viability_replay", {}).get("total_trades") != 0
        ):
            raise ValueError("Round 37 candidate unexpectedly passed selection")
    return report, report_sha, binding_sha


def _candidate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        calibration = candidate["calibration_prediction_metrics"]
        viability = candidate["viability_prediction_metrics"]
        rows.append(
            {
                "family": candidate["family"],
                "horizon_minutes": candidate["horizon_minutes"],
                "calibration_rows": calibration["rows"],
                "calibration_pearson_ic": calibration[
                    "pearson_information_coefficient"
                ],
                "calibration_spearman_ic": calibration[
                    "spearman_information_coefficient"
                ],
                "calibration_prediction_std_bps": calibration[
                    "prediction_standard_deviation_bps"
                ],
                "calibration_actual_std_bps": calibration[
                    "actual_standard_deviation_bps"
                ],
                "viability_rows": viability["rows"],
                "viability_pearson_ic": viability["pearson_information_coefficient"],
                "viability_spearman_ic": viability["spearman_information_coefficient"],
                "viability_prediction_std_bps": viability[
                    "prediction_standard_deviation_bps"
                ],
                "viability_actual_std_bps": viability["actual_standard_deviation_bps"],
                "selected_threshold_bps": "",
                "viability_gate_passed": False,
                "viability_gate_reasons": ";".join(
                    str(item) for item in candidate["viability_gate_reasons"]
                ),
            }
        )
    return rows


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in report["candidate_results"]:
        for trace in candidate["threshold_trace"]:
            replay = trace["replay"]
            rows.append(
                {
                    "family": candidate["family"],
                    "horizon_minutes": candidate["horizon_minutes"],
                    "threshold_bps": trace["threshold_bps"],
                    "support_passed": trace["support_passed"],
                    "candidate_rows": replay["candidate_rows"],
                    "nonoverlapping_trades": replay["total_trades"],
                    "active_utc_days": replay["active_utc_days"],
                    "btc_trades": replay["trades_by_symbol"]["BTCUSDT"],
                    "eth_trades": replay["trades_by_symbol"]["ETHUSDT"],
                    "sol_trades": replay["trades_by_symbol"]["SOLUSDT"],
                    "mean_net_bps": replay["mean_net_bps"],
                    "median_monthly_net_bps": replay["median_monthly_net_bps"],
                    "positive_rate": replay["positive_rate"],
                    "profit_factor": replay["profit_factor"],
                    "negative_month_fraction": replay["negative_month_fraction"],
                    "day_block_lower_95_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_lower_95"
                    ],
                    "day_block_median_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_median"
                    ],
                    "day_block_upper_95_net_bps": replay[
                        "day_block_bootstrap_mean_net_bps_upper_95"
                    ],
                }
            )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            key: item[key]
            for key in (
                "model_id",
                "family",
                "symbol",
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
        }
        for item in report["model_artifacts"]
    ]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [dict(item) for item in report["source_evidence"]["series_evidence"]]


def _progress_rows(
    path: Path, candidates: Sequence[Mapping[str, object]]
) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed = [int(row["round"]) for row in rows]
    if not fields or observed not in (list(range(1, 37)), list(range(1, 38))):
        raise ValueError("Round 37 prior progress history is invalid")
    rows = [row for row in rows if int(row["round"]) != ROUND]
    best = max(candidates, key=lambda item: float(item["viability_pearson_ic"]))
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "cross-asset cost-aware ML and local-AI ablation",
            "periods": "2022-01-01..2025-06-30",
            "selection_contaminated": True,
            "horizon_seconds": "900;1800;3600;7200",
            "feature_set": "71 causal cross-asset minute features",
            "risk_level": "research-only; no policy",
            "direction_auc": "",
            "spearman_ic": best["viability_spearman_ic"],
            "selected_signals": 0,
            "executable_trades": 0,
            "mean_gross_bps": "",
            "mean_net_bps": "",
            "status": "rejected",
            "source_file": "verified Round 37 cross-asset ML/AI report",
            "best_model_id": (
                f"{best['family']} h{best['horizon_minutes']}; no selection"
            ),
            "daily_model_fits": 16,
            "calibration_threshold_traces": 100,
            "accepted_thresholds": 0,
            "ensemble_models": 16,
            "calibration_eligible_rows": 79_479,
            "development_consumed": True,
        }
    )
    rows.append(row)
    return fields, rows


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Segoe UI",Arial,sans-serif;letter-spacing:0}.title{font-size:27px;font-weight:700;fill:#17212b}.subtitle{font-size:14px;fill:#52606d}.label{font-size:13px;fill:#263746}.axis{font-size:12px;fill:#60717f}.value{font-size:12px;font-weight:650;fill:#263746}.note{font-size:12px;fill:#65727d}.grid{stroke:#e1e8ed;stroke-width:1}.zero{stroke:#526674;stroke-width:2;stroke-dasharray:6 5}</style>',
        f'<text x="56" y="52" class="title">{html.escape(title)}</text>',
        f'<text x="56" y="80" class="subtitle">{html.escape(description)}</text>',
    ]


def _prediction_svg(rows: Sequence[Mapping[str, object]]) -> str:
    learned = [
        row
        for row in rows
        if row["family"] in {"shared_cross_asset_lightgbm", "per_symbol_lightgbm"}
    ]
    width, height = 1480, 760
    left, right, top, panel_height = 130, 70, 150, 205
    chart_width = width - left - right
    colors = {
        "shared_cross_asset_lightgbm": "#0f766e",
        "per_symbol_lightgbm": "#2563eb",
    }
    lines = _svg_start(
        width,
        height,
        "Round 37 learned weak association but compressed trade forecasts",
        "Unseen 2025-H1 prediction metrics; association is not after-cost profitability.",
    )
    x_positions = {
        horizon: left + index * chart_width / (len(HORIZONS) - 1)
        for index, horizon in enumerate(HORIZONS)
    }

    def panel(key: str, label: str, y_top: float, low: float, high: float) -> None:
        lines.append(
            f'<text x="{left}" y="{y_top - 18}" class="label">{html.escape(label)}</text>'
        )
        for tick in (low, (low + high) / 2, high):
            y = y_top + panel_height * (high - tick) / (high - low)
            lines.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
            )
            lines.append(
                f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
            )
        for family, color in colors.items():
            family_rows = sorted(
                (row for row in learned if row["family"] == family),
                key=lambda item: int(item["horizon_minutes"]),
            )
            points = []
            for row in family_rows:
                value = float(row[key])
                x = x_positions[int(row["horizon_minutes"])]
                y = y_top + panel_height * (high - value) / (high - low)
                points.append((x, y))
            lines.append(
                '<polyline points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                + f'" fill="none" stroke="{color}" stroke-width="4"/>'
            )
            lines.extend(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
                for x, y in points
            )

    panel("viability_pearson_ic", "Pearson information coefficient", top, -0.015, 0.025)
    panel(
        "viability_prediction_std_bps",
        "Prediction standard deviation (bps; realized volatility is 40-168 bps)",
        top + panel_height + 120,
        0.0,
        6.0,
    )
    for horizon, x in x_positions.items():
        lines.append(
            f'<text x="{x:.1f}" y="{height - 58}" text-anchor="middle" class="axis">{horizon}m</text>'
        )
    for index, (family, color) in enumerate(colors.items()):
        label = (
            "shared LightGBM" if family.startswith("shared") else "per-symbol LightGBM"
        )
        x = left + index * 250
        lines.append(
            f'<line x1="{x}" y1="{height - 24}" x2="{x + 28}" y2="{height - 24}" stroke="{color}" stroke-width="4"/><text x="{x + 38}" y="{height - 19}" class="note">{label}</text>'
        )
    lines.append(
        f'<text x="{width - right}" y="{height - 19}" text-anchor="end" class="note">Best Pearson IC: 0.0178 at 120m. No model passed economic selection.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _support_svg(rows: Sequence[Mapping[str, object]]) -> str:
    active = [
        row
        for row in rows
        if row["family"]
        in {
            "linear_ridge",
            "shared_cross_asset_lightgbm",
            "per_symbol_lightgbm",
        }
    ]
    largest: list[Mapping[str, object]] = []
    for family in sorted({str(row["family"]) for row in active}):
        for horizon in HORIZONS:
            group = [
                row
                for row in active
                if row["family"] == family and int(row["horizon_minutes"]) == horizon
            ]
            largest.append(
                max(group, key=lambda item: int(item["nonoverlapping_trades"]))
            )
    width, height = 1480, 800
    left, right, top, row_height = 390, 120, 126, 48
    chart_width = width - left - right
    max_trades = max(int(row["nonoverlapping_trades"]) for row in largest)
    colors = (
        ("btc_trades", "#2563eb"),
        ("eth_trades", "#0f766e"),
        ("sol_trades", "#b45309"),
    )
    lines = _svg_start(
        width,
        height,
        "No calibration threshold had cross-symbol support",
        "Largest-support threshold per family and horizon; every row failed the frozen minimum of 15 trades per symbol.",
    )
    for index, row in enumerate(largest):
        y = top + index * row_height
        family = str(row["family"]).replace("_", " ")
        label = f"{family} / {row['horizon_minutes']}m / {float(row['threshold_bps']):.0f}bps"
        lines.append(
            f'<text x="{left - 18}" y="{y + 18}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        x = left
        for key, color in colors:
            value = int(row[key])
            bar_width = chart_width * value / max_trades
            lines.append(
                f'<rect x="{x:.1f}" y="{y}" width="{bar_width:.1f}" height="26" fill="{color}"/>'
            )
            x += bar_width
        lines.append(
            f'<text x="{min(x + 9, width - right + 8):.1f}" y="{y + 18}" class="value">{row["btc_trades"]}/{row["eth_trades"]}/{row["sol_trades"]}</text>'
        )
    for index, (key, color) in enumerate(colors):
        x = left + index * 160
        lines.append(
            f'<rect x="{x}" y="{height - 38}" width="16" height="16" fill="{color}"/><text x="{x + 24}" y="{height - 25}" class="note">{key[:3].upper()} trades</text>'
        )
    lines.append(
        f'<text x="{width - right}" y="{height - 25}" text-anchor="end" class="note">Values are descriptive calibration counts, not selected or viability trades.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _economics_svg(rows: Sequence[Mapping[str, object]]) -> str:
    eligible = [
        row
        for row in rows
        if int(row["nonoverlapping_trades"]) >= 30
        and row["day_block_lower_95_net_bps"] is not None
    ]
    selected = sorted(
        eligible,
        key=lambda item: float(item["day_block_lower_95_net_bps"]),
        reverse=True,
    )[:12]
    width, height = 1480, 800
    left, right, top, row_height = 420, 280, 128, 46
    chart_width = width - left - right
    low = min(-25.0, min(float(row["day_block_lower_95_net_bps"]) for row in selected))
    high = max(55.0, max(float(row["day_block_upper_95_net_bps"]) for row in selected))

    def x(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Rare profitable calibration tails were not diversified",
        "Top day-block intervals among cells with at least 30 trades; all cells failed frozen support and remained ineligible.",
    )
    zero = x(0.0)
    lines.append(
        f'<line x1="{zero:.1f}" y1="{top - 12}" x2="{zero:.1f}" y2="{top + row_height * len(selected)}" class="zero"/>'
    )
    for index, row in enumerate(selected):
        y = top + index * row_height
        family = str(row["family"]).replace("_", " ")
        label = f"{family} / {row['horizon_minutes']}m / {float(row['threshold_bps']):.0f}bps"
        lower = float(row["day_block_lower_95_net_bps"])
        median = float(row["day_block_median_net_bps"])
        upper = float(row["day_block_upper_95_net_bps"])
        lines.append(
            f'<text x="{left - 18}" y="{y + 6}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        lines.append(
            f'<line x1="{x(lower):.1f}" y1="{y}" x2="{x(upper):.1f}" y2="{y}" stroke="#94a3b8" stroke-width="7" stroke-linecap="round"/>'
        )
        lines.append(f'<circle cx="{x(median):.1f}" cy="{y}" r="6" fill="#0f766e"/>')
        lines.append(
            f'<text x="{width - 70}" y="{y + 6}" text-anchor="end" class="value">n={row["nonoverlapping_trades"]}; {row["btc_trades"]}/{row["eth_trades"]}/{row["sol_trades"]}</text>'
        )
    lines.append(
        f'<text x="{left}" y="{height - 34}" class="note">Line: 95% day-block bootstrap interval; dot: median bootstrap mean net bps. Counts: total and BTC/ETH/SOL.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _research_progress_svg(progress: Sequence[Mapping[str, object]]) -> str:
    rows = [row for row in progress if int(row["round"]) >= 7]
    plotted: list[tuple[Mapping[str, object], str, float | None]] = []
    numeric: list[float] = []
    for row in rows:
        net = str(row.get("mean_net_bps") or "").strip()
        diagnostic = str(row.get("best_top_500_exact_after_cost_bps") or "").strip()
        if int(row.get("executable_trades") or 0) > 0 and net:
            kind, value = "simulated", float(net)
            numeric.append(value)
        elif diagnostic:
            kind, value = "diagnostic", float(diagnostic)
            numeric.append(value)
        else:
            kind, value = "absent", None
        plotted.append((row, kind, value))
    lower = min(-2.0, min(numeric) - 1.5)
    upper = max(2.0, max(numeric) + 1.5)
    width, height = 1500, 660
    left, right, top, chart_height = 120, 70, 150, 330
    chart_width = width - left - right

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines = _svg_start(
        width,
        height,
        "After-cost evidence by research round",
        "Heterogeneous simulations, overlap diagnostics, and rounds without a trade series use separate marks.",
    )
    lines.extend(
        [
            '<circle cx="870" cy="52" r="7" fill="#b42318"/><text x="887" y="57" class="note">simulated-trade mean</text>',
            '<rect x="1050" y="45" width="14" height="14" fill="#7b559c"/><text x="1074" y="57" class="note">overlap diagnostic</text>',
            '<rect x="1250" y="45" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 1257 52)"/><text x="1276" y="57" class="note">no trade series</text>',
            f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
        ]
    )
    ticks = (lower, lower + (upper - lower) / 2, 0.0, upper)
    for tick in sorted(set(ticks)):
        py = y(tick)
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.1f}</text>'
        )
    absent_y = top + chart_height + 72
    lines.append(
        f'<text x="{left - 16}" y="{absent_y + 4}" text-anchor="end" class="axis">no series</text>'
    )
    for index, (row, kind, value) in enumerate(plotted):
        x = left + chart_width * (index + 0.5) / len(plotted)
        if kind == "simulated" and value is not None:
            py = y(value)
            lines.append(
                f'<circle cx="{x:.1f}" cy="{py:.1f}" r="8" fill="#b42318" stroke="#ffffff" stroke-width="2"/>'
            )
            label_offset = 15 + 17 * (index % 2)
            display_value = 0.0 if abs(value) < 0.005 else value
            lines.append(
                f'<text x="{x:.1f}" y="{py - label_offset:.1f}" text-anchor="middle" class="value">{display_value:+.2f}</text>'
            )
        elif kind == "diagnostic" and value is not None:
            py = y(value)
            lines.append(
                f'<rect x="{x - 7:.1f}" y="{py - 7:.1f}" width="14" height="14" fill="#7b559c"/>'
            )
        else:
            lines.append(
                f'<rect x="{x - 7:.1f}" y="{absent_y - 7:.1f}" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 {x:.1f} {absent_y:.1f})"/>'
            )
        lines.append(
            f'<text x="{x:.1f}" y="{absent_y + 37}" text-anchor="middle" class="axis">R{row["round"]}</text>'
        )
    lines.append(
        '<text x="56" y="632" class="note">Windows and units differ by round. This is evidence lineage, not an equity curve or portfolio return series.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _readme(
    report: Mapping[str, object], candidates: Sequence[Mapping[str, object]]
) -> str:
    best = max(candidates, key=lambda item: float(item["viability_pearson_ic"]))
    runtime = report["runtime_evidence"]
    memory = runtime["memory"]
    return f"""# Round 37: diversified candidate selection rejected

**The cross-asset regression lane did not earn permission to trade.** GPU-trained LightGBM found weak out-of-period association, but forecasts were too compressed and the rare profitable calibration tails were concentrated in SOLUSDT. The frozen BTC/ETH/SOL support gate rejected every threshold before viability replay or AI review.

| Evidence | Verified result |
| --- | ---: |
| Source / target span | Binance USD-M 1m / 2022-01-01 to 2025-06-30 UTC |
| Decision rows / causal features | {report["dataset"]["rows"]:,} / {report["dataset"]["feature_count"]} |
| GPU models / candidates / threshold cells | {len(report["model_artifacts"])} / {len(candidates)} / {sum(len(item["threshold_trace"]) for item in report["candidate_results"])} |
| Best 2025-H1 Pearson / Spearman IC | {float(best["viability_pearson_ic"]):.4f} / {float(best["viability_spearman_ic"]):.4f} ({str(best["family"]).replace("_", " ")}, {best["horizon_minutes"]}m) |
| Selected thresholds / viability trades | 0 / 0 |
| AI cases / AI models evaluated | {report["ai_case_set"]["cases"]} / {len(report["ai_reports"])} |
| Compute / runtime / peak working set | {report["backend"]["device"]} / {runtime["elapsed_seconds"]:.1f}s / {memory["peak_working_set_bytes"] / 1024**3:.2f} GiB |
| Trading authority / leverage | none / none |

![Prediction quality](charts/prediction-quality.svg)

![Calibration support](charts/calibration-support.svg)

![Calibration economics](charts/calibration-economics.svg)

![Research progress](charts/research-progress.svg)

The positive tail observations are calibration diagnostics, not selected trades, ROI, an equity curve, or profitability evidence. No ROI graph exists because no portfolio or executable trade series was produced. Qwen3 and Fino1 were intentionally not invoked: without a diversified ML candidate set, an AI veto ablation would have no causal cases to review.

The next lane changes the target rather than weakening risk controls: cost-aware long/abstain/short classification with real futures premium and funding features. Selection-confirmation 2025-H2 and terminal 2026 data remain sealed.

Data: [candidates.csv](candidates.csv) | [thresholds.csv](thresholds.csv) | [models.csv](models.csv) | [sources.csv](sources.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
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
    """Publish validated latest-only Round 37 tables, charts, and manifest."""

    evidence_root = evidence_root.resolve()
    report, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path.resolve(), binding_path.resolve()
    )
    candidates = _candidate_rows(report)
    thresholds = _threshold_rows(report)
    models = _model_rows(report)
    sources = _source_rows(report)
    progress_fields, progress = _progress_rows(prior_progress_path, candidates)
    output_dir = output_dir.resolve()
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "candidates.csv",
        output_dir / "thresholds.csv",
        output_dir / "models.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "prediction-quality.svg",
        charts / "calibration-support.svg",
        charts / "calibration-economics.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    _write_csv(output_dir / "candidates.csv", candidates)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "screen.json",
        (evidence_root / "report.json").read_text(encoding="utf-8"),
    )
    _write_text(output_dir / "README.md", _readme(report, candidates))
    _write_text(charts / "prediction-quality.svg", _prediction_svg(candidates))
    _write_text(charts / "calibration-support.svg", _support_svg(thresholds))
    _write_text(charts / "calibration-economics.svg", _economics_svg(thresholds))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "cross_asset_cost_aware_ai_ablation_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "source_panel_stream_sha256": report["source_evidence"]["panel_stream_sha256"],
        "dataset_rows": report["dataset"]["rows"],
        "feature_count": report["dataset"]["feature_count"],
        "gpu_model_count": len(models),
        "candidate_count": len(candidates),
        "threshold_cell_count": len(thresholds),
        "selected_threshold_count": 0,
        "viability_trade_count": 0,
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
        default=research / "round-037-cross-asset-cost-aware-ai-ablation-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-037-cross-asset-ai-execution-binding.json",
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
