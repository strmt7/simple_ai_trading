from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from simple_ai_trading.storage import write_json_atomic
from simple_ai_trading.tape_depth_execution import (
    load_tape_depth_execution_confirmation_design,
)
from simple_ai_trading.tape_depth_execution_confirmation import (
    _canonical_sha256,
    _validate_checkpoint,
)


PERIOD_FIELDS = (
    "period",
    "forecast_status",
    "direction_auc",
    "spearman_ic",
    "mae_bps",
    "zero_baseline_mae_bps",
    "selected_signals",
    "long_signals",
    "short_signals",
    "overlap_suppressed",
    "scheduled_signals",
    "participation_rejections",
    "quote_rejections",
    "executable_trades",
    "mean_quote_gross_bps",
    "mean_net_bps",
    "positive_net_rate",
    "model_sha256",
    "predictions_sha256",
    "period_fingerprint",
)
TRADE_FIELDS = (
    "period",
    "signal_index",
    "side",
    "decision_time_utc",
    "entry_time_utc",
    "exit_time_utc",
    "entry_bid",
    "entry_ask",
    "exit_bid",
    "exit_ask",
    "entry_quote_age_ms",
    "exit_quote_age_ms",
    "maximum_l1_participation",
    "trade_reference_gross_bps",
    "quote_path_gross_bps",
    "spread_cost_bps",
    "fee_cost_bps",
    "slippage_cost_bps",
    "net_return_bps",
    "status",
    "rejection_reason",
)
PROGRESS_FIELDS = (
    "round",
    "stage",
    "periods",
    "selection_contaminated",
    "horizon_seconds",
    "feature_set",
    "risk_level",
    "direction_auc",
    "spearman_ic",
    "selected_signals",
    "executable_trades",
    "mean_gross_bps",
    "mean_net_bps",
    "status",
    "source_file",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish hash-manifested real tape/depth confirmation evidence."
    )
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/model-research/tape-depth/latest"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def _timestamp_utc(value: object) -> str:
    return datetime.fromtimestamp(int(value) / 1_000.0, tz=UTC).isoformat()


def _finite(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite financial evidence")
    return parsed


def _load_confirmation(
    root: Path,
    *,
    design_path: Path,
    availability_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object], str]:
    design, design_sha256 = load_tape_depth_execution_confirmation_design(
        design_path,
        availability_path=availability_path,
    )
    report = _read_json(root / "report.json")
    if report.get("design_sha256") != design_sha256 or report.get(
        "confirmation_fingerprint"
    ) != _canonical_sha256(report, omit="confirmation_fingerprint"):
        raise ValueError("confirmation report failed fingerprint validation")
    periods = [str(value) for value in design["confirmation_periods"]]
    period_reports = [
        _validate_checkpoint(
            root / "periods" / period / "report.json",
            output_dir=root,
            period=period,
            design_sha256=design_sha256,
        )
        for period in periods
    ]
    final_fingerprints = [
        str(item.get("period_fingerprint"))
        for item in report.get("periods", [])
        if isinstance(item, Mapping)
    ]
    if final_fingerprints != [
        str(item["period_fingerprint"]) for item in period_reports
    ]:
        raise ValueError("confirmation report differs from period checkpoints")
    return report, period_reports, design, design_sha256


def _period_and_trade_rows(
    reports: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    period_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for report in reports:
        period = str(report["period"])
        forecast = report["forecast"]
        execution = report["execution"]
        evidence = report["evidence"]
        if not all(
            isinstance(value, Mapping) for value in (forecast, execution, evidence)
        ):
            raise ValueError("confirmation period sections are invalid")
        forecast = dict(forecast)  # type: ignore[arg-type]
        execution = dict(execution)  # type: ignore[arg-type]
        evidence = dict(evidence)  # type: ignore[arg-type]
        forecast_metrics = dict(forecast["metrics"])
        execution_metrics = dict(execution["metrics"])
        period_rows.append(
            {
                "period": period,
                "forecast_status": forecast["status"],
                "direction_auc": forecast_metrics["direction_auc"],
                "spearman_ic": forecast_metrics["spearman_information_coefficient"],
                "mae_bps": forecast_metrics["mean_absolute_error_bps"],
                "zero_baseline_mae_bps": forecast_metrics["zero_baseline_mae_bps"],
                "selected_signals": execution_metrics["selected_signal_rows"],
                "long_signals": forecast_metrics["calibration_threshold_long_rows"],
                "short_signals": forecast_metrics["calibration_threshold_short_rows"],
                "overlap_suppressed": execution_metrics["overlap_suppressed_rows"],
                "scheduled_signals": execution_metrics["scheduled_signal_rows"],
                "participation_rejections": execution_metrics[
                    "rejected_participation_rows"
                ],
                "quote_rejections": execution_metrics["rejected_quote_rows"],
                "executable_trades": execution_metrics["executable_rows"],
                "mean_quote_gross_bps": execution_metrics["mean_quote_path_gross_bps"],
                "mean_net_bps": execution_metrics["mean_net_return_bps"],
                "positive_net_rate": execution_metrics["positive_net_rate"],
                "model_sha256": evidence["model_sha256"],
                "predictions_sha256": evidence["predictions_sha256"],
                "period_fingerprint": report["period_fingerprint"],
            }
        )
        raw_rows = report.get("execution_rows")
        if not isinstance(raw_rows, list):
            raise ValueError("confirmation execution rows are invalid")
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise ValueError("confirmation execution row is invalid")
            row = dict(raw)
            trade_rows.append(
                {
                    "period": period,
                    "signal_index": row["signal_index"],
                    "side": "long" if int(row["side"]) == 1 else "short",
                    "decision_time_utc": _timestamp_utc(row["decision_time_ms"]),
                    "entry_time_utc": _timestamp_utc(row["target_entry_time_ms"]),
                    "exit_time_utc": _timestamp_utc(row["target_exit_time_ms"]),
                    **{field: row.get(field) for field in TRADE_FIELDS[6:]},
                }
            )
    return period_rows, trade_rows


def _find_result(
    payload: Mapping[str, object],
    **expected: object,
) -> dict[str, object]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("discovery screen lacks results")
    matches = [
        dict(item)
        for item in results
        if isinstance(item, Mapping)
        and all(item.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"discovery screen match is ambiguous: {expected}")
    return matches[0]


def _metric_row(
    round_number: int,
    *,
    stage: str,
    source_file: str,
    result: Mapping[str, object],
    horizon_seconds: int,
    feature_set: str,
    risk_level: str,
) -> dict[str, object]:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("discovery result lacks metrics")
    return {
        "round": round_number,
        "stage": stage,
        "periods": "2024-03-15",
        "selection_contaminated": True,
        "horizon_seconds": horizon_seconds,
        "feature_set": feature_set,
        "risk_level": risk_level,
        "direction_auc": metrics["direction_auc"],
        "spearman_ic": metrics["spearman_information_coefficient"],
        "selected_signals": metrics["calibration_threshold_rows"],
        "executable_trades": "",
        "mean_gross_bps": metrics["calibration_threshold_mean_signed_gross_bps"],
        "mean_net_bps": "",
        "status": result["status"],
        "source_file": source_file,
    }


def _progress_rows(
    discovery_root: Path,
    confirmation_report: Mapping[str, object],
    period_reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    diagnostic = _read_json(discovery_root / "diagnostic.json")
    horizon = _read_json(discovery_root / "horizon-screen.json")
    features = _read_json(discovery_root / "feature-risk-screen.json")
    conviction = _read_json(discovery_root / "conviction-capacity-screen.json")
    intermediate = _read_json(discovery_root / "intermediate-horizon-screen.json")
    calibrated = _read_json(discovery_root / "calibrated-confidence-screen.json")
    after_cost = _read_json(discovery_root / "after-cost-taker-diagnostic.json")
    conservative = dict(diagnostic["models"])["conservative"]
    if not isinstance(conservative, Mapping):
        raise ValueError("initial discovery model is invalid")
    rows = [
        _metric_row(
            1,
            stage="initial full-feature forecast",
            source_file="diagnostic.json",
            result=conservative,
            horizon_seconds=60,
            feature_set="full",
            risk_level="conservative",
        ),
        _metric_row(
            2,
            stage="horizon screen",
            source_file="horizon-screen.json",
            result=_find_result(horizon, horizon_seconds=5),
            horizon_seconds=5,
            feature_set="full",
            risk_level="regular",
        ),
        _metric_row(
            3,
            stage="feature ablation",
            source_file="feature-risk-screen.json",
            result=_find_result(
                features,
                horizon_seconds=5,
                risk_level="regular",
                feature_set="cross_asset",
                model_profile="regularized",
            ),
            horizon_seconds=5,
            feature_set="cross_asset",
            risk_level="regular",
        ),
        _metric_row(
            4,
            stage="risk conviction",
            source_file="conviction-capacity-screen.json",
            result=_find_result(
                conviction,
                risk_level="conservative",
                feature_set="cross_asset",
                model_profile="regularized",
            ),
            horizon_seconds=5,
            feature_set="cross_asset",
            risk_level="conservative",
        ),
        _metric_row(
            5,
            stage="intermediate horizon",
            source_file="intermediate-horizon-screen.json",
            result=_find_result(
                intermediate,
                horizon_seconds=20,
                feature_set="cross_asset",
            ),
            horizon_seconds=20,
            feature_set="cross_asset",
            risk_level="conservative",
        ),
        _metric_row(
            6,
            stage="calibrated confidence",
            source_file="calibrated-confidence-screen.json",
            result=_find_result(
                calibrated,
                horizon_seconds=20,
                risk_level="conservative",
            ),
            horizon_seconds=20,
            feature_set="cross_asset",
            risk_level="conservative",
        ),
    ]
    forecast = after_cost.get("forecast_artifact")
    scenarios = after_cost.get("scenarios")
    if not isinstance(forecast, Mapping) or not isinstance(scenarios, Mapping):
        raise ValueError("after-cost discovery evidence is invalid")
    stressed = scenarios.get("observed_bbo_plus_1bps_each_side")
    if not isinstance(stressed, Mapping) or not isinstance(
        stressed.get("report"), Mapping
    ):
        raise ValueError("stressed discovery execution evidence is invalid")
    stressed_report = dict(stressed["report"])
    stressed_metrics = dict(stressed_report["metrics"])
    forecast_metrics = dict(forecast["metrics"])
    rows.append(
        {
            "round": 7,
            "stage": "exact-BBO discovery",
            "periods": "2024-03-15",
            "selection_contaminated": True,
            "horizon_seconds": 20,
            "feature_set": "cross_asset",
            "risk_level": "conservative",
            "direction_auc": forecast_metrics["direction_auc"],
            "spearman_ic": forecast_metrics["spearman_information_coefficient"],
            "selected_signals": stressed_metrics["selected_signal_rows"],
            "executable_trades": stressed_metrics["executable_rows"],
            "mean_gross_bps": stressed_metrics["mean_quote_path_gross_bps"],
            "mean_net_bps": stressed_metrics["mean_net_return_bps"],
            "status": stressed_report["status"],
            "source_file": "after-cost-taker-diagnostic.json",
        }
    )
    weights = [
        int(dict(dict(report["forecast"])["metrics"])["rows"])
        for report in period_reports
    ]
    total_weight = sum(weights)
    weighted_auc = (
        sum(
            weight * _finite(dict(dict(report["forecast"])["metrics"])["direction_auc"])
            for weight, report in zip(weights, period_reports, strict=True)
        )
        / total_weight
    )
    weighted_ic = (
        sum(
            weight
            * _finite(
                dict(dict(report["forecast"])["metrics"])[
                    "spearman_information_coefficient"
                ]
            )
            for weight, report in zip(weights, period_reports, strict=True)
        )
        / total_weight
    )
    all_execution_rows = [
        row
        for report in period_reports
        for row in report["execution_rows"]  # type: ignore[union-attr]
        if isinstance(row, Mapping) and row.get("status") == "executable"
    ]
    mean_gross = sum(
        _finite(row["quote_path_gross_bps"]) for row in all_execution_rows
    ) / len(all_execution_rows)
    actual = dict(confirmation_report["actual"])
    rows.append(
        {
            "round": 8,
            "stage": "untouched exact-BBO confirmation",
            "periods": ";".join(str(report["period"]) for report in period_reports),
            "selection_contaminated": False,
            "horizon_seconds": 20,
            "feature_set": "cross_asset",
            "risk_level": "conservative",
            "direction_auc": weighted_auc,
            "spearman_ic": weighted_ic,
            "selected_signals": sum(
                int(dict(dict(report["execution"])["metrics"])["selected_signal_rows"])
                for report in period_reports
            ),
            "executable_trades": actual["combined_executable_rows"],
            "mean_gross_bps": mean_gross,
            "mean_net_bps": actual["combined_mean_net_return_bps"],
            "status": confirmation_report["status"],
            "source_file": "confirmation report + period checkpoints",
        }
    )
    return rows


def _svg_header(width: int, height: int, title: str, subtitle: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="52" y="46" font-family="Segoe UI, Arial, sans-serif" font-size="24" fill="#111827">{html.escape(title)}</text>
  <text x="52" y="73" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#475569">{html.escape(subtitle)}</text>
'''


def _bar_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    first_field: str,
    second_field: str,
    first_label: str,
    second_label: str,
    title: str,
    subtitle: str,
    suffix: str,
) -> str:
    width, height = 1100, 470
    left, top, chart_width, chart_height = 110, 110, 920, 270
    values = [_finite(row[first_field]) for row in rows] + [
        _finite(row[second_field]) for row in rows
    ]
    low, high = min(values + [0.0]), max(values + [0.0])
    padding = max(1.0, (high - low) * 0.15)
    low -= padding
    high += padding

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * chart_height

    output = [_svg_header(width, height, title, subtitle)]
    zero_y = y(0.0)
    output.append(
        f'  <line x1="{left}" y1="{zero_y:.2f}" x2="{left + chart_width}" y2="{zero_y:.2f}" stroke="#64748b" stroke-width="1.5"/>'
    )
    group_width = chart_width / len(rows)
    bar_width = 58
    colors = ("#0f766e", "#b91c1c")
    for index, row in enumerate(rows):
        center = left + group_width * (index + 0.5)
        for offset, field in enumerate((first_field, second_field)):
            value = _finite(row[field])
            x = center + (-bar_width - 6 if offset == 0 else 6)
            value_y = y(value)
            rect_y = min(zero_y, value_y)
            rect_height = max(1.0, abs(zero_y - value_y))
            output.append(
                f'  <rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_width}" height="{rect_height:.2f}" fill="{colors[offset]}"/>'
            )
            label_y = value_y - 8 if value >= 0 else value_y + 18
            output.append(
                f'  <text x="{x + bar_width / 2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#111827">{value:+.3f}{suffix}</text>'
            )
        output.append(
            f'  <text x="{center:.2f}" y="414" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{html.escape(str(row["period"]))}</text>'
        )
    output.extend(
        (
            f'  <rect x="{left}" y="440" width="14" height="14" fill="{colors[0]}"/><text x="{left + 22}" y="452" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{html.escape(first_label)}</text>',
            f'  <rect x="{left + 220}" y="440" width="14" height="14" fill="{colors[1]}"/><text x="{left + 242}" y="452" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{html.escape(second_label)}</text>',
            "</svg>\n",
        )
    )
    return "\n".join(output)


def _funnel_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1100, 460
    left, top, chart_width, chart_height = 90, 105, 950, 265
    maximum = max(1, max(int(row["selected_signals"]) for row in rows))
    colors = ("#0f766e", "#2563eb", "#b91c1c")
    fields = ("selected_signals", "scheduled_signals", "executable_trades")
    labels = ("selected", "non-overlapping", "executable")
    output = [
        _svg_header(
            width,
            height,
            "Round 8 signal selection",
            "Untouched UTC dates; no minimum trade quota is forced.",
        )
    ]
    group_width = chart_width / len(rows)
    bar_width = 52
    baseline = top + chart_height
    output.append(
        f'  <line x1="{left}" y1="{baseline}" x2="{left + chart_width}" y2="{baseline}" stroke="#94a3b8"/>'
    )
    for index, row in enumerate(rows):
        center = left + group_width * (index + 0.5)
        for bar_index, field in enumerate(fields):
            value = int(row[field])
            bar_height = chart_height * value / maximum
            x = center + (bar_index - 1) * (bar_width + 8) - bar_width / 2
            output.append(
                f'  <rect x="{x:.2f}" y="{baseline - bar_height:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{colors[bar_index]}"/>'
            )
            output.append(
                f'  <text x="{x + bar_width / 2:.2f}" y="{baseline - bar_height - 7:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#111827">{value}</text>'
            )
        output.append(
            f'  <text x="{center:.2f}" y="400" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{row["period"]}</text>'
        )
    for index, label in enumerate(labels):
        x = left + index * 180
        output.append(
            f'  <rect x="{x}" y="430" width="14" height="14" fill="{colors[index]}"/><text x="{x + 22}" y="442" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{label}</text>'
        )
    output.append("</svg>\n")
    return "\n".join(output)


def _quality_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1100, 470
    output = [
        _svg_header(
            width,
            height,
            "Round 8 forecast quality",
            "AUC baseline 0.5; Spearman information-coefficient baseline 0.0.",
        )
    ]
    panels = (
        ("direction_auc", 0.48, 0.59, 0.5, "Direction AUC", "#2563eb"),
        ("spearman_ic", 0.0, 0.14, 0.0, "Spearman IC", "#0f766e"),
    )
    for panel_index, (field, low, high, baseline_value, label, color) in enumerate(
        panels
    ):
        x0 = 70 + panel_index * 530
        y0, panel_width, panel_height = 120, 470, 260

        def y(value: float) -> float:
            return y0 + (high - value) / (high - low) * panel_height

        output.append(
            f'  <text x="{x0}" y="105" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#111827">{label}</text>'
        )
        baseline_y = y(float(baseline_value))
        output.append(
            f'  <line x1="{x0}" y1="{baseline_y:.2f}" x2="{x0 + panel_width}" y2="{baseline_y:.2f}" stroke="#94a3b8" stroke-dasharray="6 5"/>'
        )
        points = []
        for index, row in enumerate(rows):
            x = x0 + panel_width * (index + 0.5) / len(rows)
            value = _finite(row[field])
            points.append(f"{x:.2f},{y(value):.2f}")
            output.append(
                f'  <circle cx="{x:.2f}" cy="{y(value):.2f}" r="6" fill="{color}"/><text x="{x:.2f}" y="{y(value) - 12:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#111827">{value:.4f}</text><text x="{x:.2f}" y="410" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">{row["period"]}</text>'
            )
        output.append(
            f'  <polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
    output.append("</svg>\n")
    return "\n".join(output)


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1160, 560
    left, chart_width = 80, 1010
    output = [
        _svg_header(
            width,
            height,
            "Tape/depth research progress: rounds 1-8",
            "Rounds 1-7 used the contaminated discovery date; Round 8 is untouched confirmation.",
        )
    ]
    x_values = [
        left + chart_width * index / (len(rows) - 1) for index in range(len(rows))
    ]
    auc_top, auc_height = 105, 175
    econ_top, econ_height = 330, 150

    def auc_y(value: float) -> float:
        return auc_top + (0.60 - value) / 0.15 * auc_height

    output.append(
        f'  <line x1="{left}" y1="{auc_y(0.5):.2f}" x2="{left + chart_width}" y2="{auc_y(0.5):.2f}" stroke="#94a3b8" stroke-dasharray="6 5"/><text x="{left}" y="95" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">Direction AUC</text>'
    )
    auc_points = " ".join(
        f"{x:.2f},{auc_y(_finite(row['direction_auc'])):.2f}"
        for x, row in zip(x_values, rows, strict=True)
    )
    output.append(
        f'  <polyline points="{auc_points}" fill="none" stroke="#2563eb" stroke-width="3"/>'
    )
    economic_values = [
        _finite(row["mean_net_bps"])
        if row["mean_net_bps"] != ""
        else _finite(row["mean_gross_bps"])
        for row in rows
    ]
    low, high = min(economic_values + [0.0]), max(economic_values + [0.0])
    padding = max(1.0, (high - low) * 0.12)
    low -= padding
    high += padding

    def econ_y(value: float) -> float:
        return econ_top + (high - value) / (high - low) * econ_height

    output.append(
        f'  <line x1="{left}" y1="{econ_y(0.0):.2f}" x2="{left + chart_width}" y2="{econ_y(0.0):.2f}" stroke="#64748b"/><text x="{left}" y="320" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">Selected mean return (gross before Round 7; net for Rounds 7-8), bps</text>'
    )
    for x, row, value in zip(x_values, rows, economic_values, strict=True):
        contaminated = bool(row["selection_contaminated"])
        color = "#b91c1c" if row["mean_net_bps"] != "" else "#0f766e"
        value_label_y = econ_y(value) - 11 if value >= 0.0 else econ_y(value) + 22
        output.append(
            f'  <circle cx="{x:.2f}" cy="{econ_y(value):.2f}" r="6" fill="{color}"/><text x="{x:.2f}" y="{value_label_y:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#111827">{value:+.3f}</text><text x="{x:.2f}" y="520" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334155">R{row["round"]}</text><text x="{x:.2f}" y="540" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#{"64748b" if contaminated else "111827"}">{"discovery" if contaminated else "confirmation"}</text>'
        )
    output.append("</svg>\n")
    return "\n".join(output)


def _artifact_entry(path: Path, *, repo_root: Path) -> dict[str, object]:
    entry: dict[str, object] = {
        "path": path.relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            entry["columns"] = columns
            entry["row_count"] = sum(1 for _ in reader)
    return entry


def _validate_publication(report_path: Path, *, repo_root: Path) -> None:
    report = _read_json(report_path)
    manifest = report.get("artifact_integrity")
    tracked = report.get("tracked_artifacts")
    if (
        report.get("artifact_class") != "exchange_sourced_model_confirmation_graph_data"
        or report.get("tracked_repo_artifact") is not True
        or report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
        or not isinstance(manifest, list)
        or not isinstance(tracked, list)
    ):
        raise ValueError("published confirmation report contract is invalid")
    report_relative = report_path.relative_to(repo_root).as_posix()
    if report_relative not in tracked:
        raise ValueError("published confirmation report is not tracked by its manifest")
    entries = {
        str(entry.get("path")): entry
        for entry in manifest
        if isinstance(entry, Mapping)
    }
    for raw_path in tracked:
        relative = str(raw_path)
        if relative == report_relative:
            continue
        entry = entries.get(relative)
        path = repo_root / relative
        if (
            entry is None
            or not path.is_file()
            or int(entry.get("bytes", -1)) != path.stat().st_size
            or entry.get("sha256") != _sha256(path)
        ):
            raise ValueError(f"published confirmation artifact differs: {relative}")
        if path.suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                columns = next(reader)
                rows = sum(1 for _ in reader)
            if (
                entry.get("columns") != columns
                or int(entry.get("row_count", -1)) != rows
            ):
                raise ValueError(
                    f"published confirmation CSV shape differs: {relative}"
                )


def main() -> int:
    args = _arguments()
    repo_root = Path(__file__).resolve().parents[1]
    confirmation_root = args.confirmation_root.resolve()
    discovery_root = args.discovery_root.resolve()
    output_dir = args.output_dir.resolve()
    report, period_reports, design, design_sha256 = _load_confirmation(
        confirmation_root,
        design_path=args.design.resolve(),
        availability_path=args.availability.resolve(),
    )
    period_rows, trade_rows = _period_and_trade_rows(period_reports)
    progress_rows = _progress_rows(discovery_root, report, period_reports)
    periods_path = output_dir / "periods.csv"
    trades_path = output_dir / "trades.csv"
    progress_path = output_dir / "progress.csv"
    after_cost_path = output_dir / "charts" / "after-cost-performance.svg"
    quality_path = output_dir / "charts" / "forecast-quality.svg"
    funnel_path = output_dir / "charts" / "signal-selection.svg"
    progress_chart_path = output_dir / "charts" / "research-progress.svg"
    readme_path = output_dir / "README.md"
    report_path = output_dir / "report.json"
    _write_csv(periods_path, period_rows, PERIOD_FIELDS)
    _write_csv(trades_path, trade_rows, TRADE_FIELDS)
    _write_csv(progress_path, progress_rows, PROGRESS_FIELDS)
    _write_text(
        after_cost_path,
        _bar_chart(
            period_rows,
            first_field="mean_quote_gross_bps",
            second_field="mean_net_bps",
            first_label="observed quote-path gross",
            second_label="net after fees + 1 bps slippage per leg",
            title="Round 8 exact-BBO after-cost confirmation",
            subtitle="BTCUSDT, 20-second horizon, 100 ms BBO, $1,000 reference notional, leverage 1x.",
            suffix="",
        ),
    )
    _write_text(quality_path, _quality_svg(period_rows))
    _write_text(funnel_path, _funnel_svg(period_rows))
    _write_text(progress_chart_path, _progress_svg(progress_rows))
    actual = dict(report["actual"])
    _write_text(
        readme_path,
        f"""# Tape/Depth Round 8 Evidence

Status: **rejected**. This is real, checksummed Binance USD-M research evidence,
not a profitability or execution claim.

- Untouched UTC dates: {", ".join(str(value) for value in design["confirmation_periods"])}
- Executable trades: {actual["combined_executable_rows"]}
- Weighted mean net return: {float(actual["combined_mean_net_return_bps"]):+.6f} bps
- Positive net rate: {float(actual["combined_positive_net_rate"]):.2%}
- Quote-path rejections: {actual["quote_rejection_rows"]}
- Liquidations: {actual["liquidation_events"]}
- Design SHA-256: `{design_sha256}`
- Confirmation SHA-256: `{report["confirmation_fingerprint"]}`

The fixed candidate failed five precommitted gates: {", ".join(str(value) for value in report["rejection_reasons"])}.
It must not be promoted or traded. These three dates are consumed confirmation
evidence and cannot be reused for model selection.

## Charts

![After-cost performance](charts/after-cost-performance.svg)

![Forecast quality](charts/forecast-quality.svg)

![Signal selection](charts/signal-selection.svg)

![Research progress](charts/research-progress.svg)

The source tables are [periods.csv](periods.csv), [trades.csv](trades.csv), and
[progress.csv](progress.csv). Independent fixed-horizon trades do not form a
portfolio equity curve, so this evidence intentionally reports no ROI or
drawdown. Regenerate with `python tools/publish_tape_depth_confirmation.py` and
the arguments recorded in `report.json`.
""",
    )
    artifacts = [
        readme_path,
        periods_path,
        trades_path,
        progress_path,
        after_cost_path,
        quality_path,
        funnel_path,
        progress_chart_path,
    ]
    publication = {
        "schema_version": "tape-depth-research-publication-v1",
        "artifact_class": "exchange_sourced_model_confirmation_graph_data",
        "tracked_repo_artifact": True,
        "status": report["status"],
        "rejection_reasons": report["rejection_reasons"],
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "roi_claim": False,
        "drawdown_claim": False,
        "symbol": "BTCUSDT",
        "peer_symbols": ["ETHUSDT", "SOLUSDT"],
        "market_type": "Binance USD-M futures",
        "feature_interval": "1s",
        "execution_quote_interval": "100ms",
        "confirmation_periods": design["confirmation_periods"],
        "design_sha256": design_sha256,
        "availability_sha256": design["availability_sha256"],
        "confirmation_fingerprint": report["confirmation_fingerprint"],
        "actual": actual,
        "assumptions": design["execution"],
        "source": {
            "provider": "Binance public Data Vision S3",
            "checksums": "official per-archive SHA-256 sidecars",
            "period_fingerprints": [
                value["period_fingerprint"] for value in period_reports
            ],
            "model_sha256": [
                value["evidence"]["model_sha256"] for value in period_reports
            ],  # type: ignore[index]
            "predictions_sha256": [
                value["evidence"]["predictions_sha256"]
                for value in period_reports  # type: ignore[index]
            ],
        },
        "regeneration_command": (
            "python tools/publish_tape_depth_confirmation.py "
            "--confirmation-root <local-confirmation-root> "
            "--discovery-root <local-discovery-root> "
            "--design docs/model-research/tape-depth/confirmation-design.json "
            "--availability docs/microstructure/availability.json "
            "--output-dir docs/model-research/tape-depth/latest"
        ),
        "tracked_artifacts": [
            report_path.relative_to(repo_root).as_posix(),
            *(path.relative_to(repo_root).as_posix() for path in artifacts),
        ],
        "artifact_integrity": [
            _artifact_entry(path, repo_root=repo_root) for path in artifacts
        ],
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    write_json_atomic(report_path, publication, indent=2, sort_keys=True)
    _validate_publication(report_path, repo_root=repo_root)
    print(
        "published tape/depth confirmation: "
        f"status={publication['status']} periods={len(period_rows)} "
        f"trades={len(trade_rows)} output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
