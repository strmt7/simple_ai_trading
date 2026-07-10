"""Latest-iteration and cross-round optimization progress artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .optimization_evidence import _artifact_integrity_manifest, _utc_now, _write_csv
from .storage import write_json_atomic


_FIELDS = (
    "round_id",
    "generated_at_utc",
    "market_type",
    "interval",
    "objective",
    "critical_verdict",
    "promotion_grade",
    "promotion_status",
    "symbol_count_completed",
    "accepted_symbol_count",
    "mean_roi_pct",
    "median_roi_pct",
    "mean_baseline_roi_pct",
    "worst_max_drawdown_pct",
    "total_closed_trades",
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _read_report(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _progress_row(payload: Mapping[str, object]) -> dict[str, object]:
    progress = payload.get("progress")
    progress = progress if isinstance(progress, Mapping) else {}
    critical = payload.get("critical_analysis")
    critical = critical if isinstance(critical, Mapping) else {}
    promotion = payload.get("promotion_grade_contract")
    promotion = promotion if isinstance(promotion, Mapping) else {}
    return {
        "round_id": str(payload.get("round_id") or ""),
        "generated_at_utc": str(payload.get("generated_at_utc") or progress.get("generated_at_utc") or ""),
        "market_type": str(payload.get("market_type") or ""),
        "interval": str(payload.get("interval") or ""),
        "objective": str(payload.get("objective") or ""),
        "critical_verdict": str(critical.get("verdict") or payload.get("evidence_verdict") or "unknown"),
        "promotion_grade": bool(payload.get("promotion_grade")),
        "promotion_status": str(promotion.get("status") or "not_requested"),
        "symbol_count_completed": int(payload.get("symbol_count_completed") or progress.get("symbol_count") or 0),
        "accepted_symbol_count": int(progress.get("accepted_symbol_count") or 0),
        "mean_roi_pct": _finite(progress.get("mean_roi_pct")),
        "median_roi_pct": _finite(progress.get("median_roi_pct")),
        "mean_baseline_roi_pct": _finite(progress.get("mean_baseline_roi_pct")),
        "worst_max_drawdown_pct": _finite(progress.get("worst_max_drawdown_pct")),
        "total_closed_trades": int(progress.get("total_closed_trades") or 0),
    }


def _scan_round_reports(docs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report_path in sorted(docs_root.glob("*/data/report.json")):
        if report_path.parts[-3] == "iteration-progress":
            continue
        payload = _read_report(report_path)
        if not payload or payload.get("artifact_class") != "exchange_sourced_backtest_graph_data":
            continue
        rows.append(_progress_row(payload))
    rows.sort(key=lambda row: (str(row.get("generated_at_utc") or ""), str(row.get("round_id") or "")))
    return rows


def _polyline(values: Sequence[float], *, x0: float, y0: float, width: float, height: float) -> str:
    if not values:
        return ""
    min_value = min(values)
    max_value = max(values)
    span = max(1e-9, max_value - min_value)
    count = max(1, len(values) - 1)
    points: list[str] = []
    for index, value in enumerate(values):
        x = x0 + (width * index / count)
        y = y0 + height - ((value - min_value) / span * height)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _render_progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width = 980
    height = 360
    left = 80
    top = 48
    chart_w = 820
    chart_h = 210
    roi_values = [_finite(row.get("mean_roi_pct")) for row in rows]
    drawdown_values = [_finite(row.get("worst_max_drawdown_pct")) for row in rows]
    closed_trades = [int(row.get("total_closed_trades") or 0) for row in rows]
    if not rows:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="220" viewBox="0 0 980 220">'
            '<rect width="980" height="220" fill="#ffffff"/>'
            '<text x="32" y="58" font-family="Segoe UI, Arial, sans-serif" font-size="24" fill="#111827">'
            "Optimization progress</text>"
            '<text x="32" y="104" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#475569">'
            "No exchange-sourced round reports are tracked yet.</text></svg>"
        )
    roi_line = _polyline(roi_values, x0=left, y0=top, width=chart_w, height=chart_h)
    drawdown_line = _polyline(drawdown_values, x0=left, y0=top, width=chart_w, height=chart_h)
    latest = rows[-1]
    latest_label = str(latest.get("round_id") or "")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  <text x="32" y="36" font-family="Segoe UI, Arial, sans-serif" font-size="24" fill="#111827">Optimization progress</text>
  <text x="32" y="64" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#475569">Source: tracked exchange-sourced round reports. Latest: {latest_label}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#cbd5e1"/>
  <line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#cbd5e1"/>
  <polyline points="{roi_line}" fill="none" stroke="#0f766e" stroke-width="3"/>
  <polyline points="{drawdown_line}" fill="none" stroke="#b91c1c" stroke-width="3" stroke-dasharray="8 5"/>
  <text x="32" y="150" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#0f766e">mean ROI %</text>
  <text x="32" y="174" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#b91c1c">worst drawdown %</text>
  <text x="{left}" y="{top + chart_h + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{rows[0].get("round_id")}</text>
  <text x="{left + chart_w - 120}" y="{top + chart_h + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">{latest_label}</text>
  <text x="32" y="318" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111827">Latest closed trades: {closed_trades[-1]} | latest accepted symbols: {latest.get("accepted_symbol_count")}/{latest.get("symbol_count_completed")} | promotion: {latest.get("promotion_status")}</text>
</svg>
'''


def build_optimization_progress_artifacts(
    docs_root: Path = Path("docs/optimization"),
    *,
    output_round: str = "iteration-progress",
) -> dict[str, object]:
    docs_root = Path(docs_root)
    rows = _scan_round_reports(docs_root)
    output_dir = docs_root / output_round
    data_dir = output_dir / "data"
    charts_dir = output_dir / "charts"
    report_path = data_dir / "report.json"
    csv_path = data_dir / "progress.csv"
    chart_path = charts_dir / "progress.svg"
    data_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, rows, _FIELDS)
    chart_path.write_text(_render_progress_svg(rows), encoding="utf-8", newline="\n")
    tracked_artifacts = [
        str(report_path).replace("\\", "/"),
        str(csv_path).replace("\\", "/"),
        str(chart_path).replace("\\", "/"),
    ]
    report = {
        "round_id": output_round,
        "generated_at_utc": _utc_now(),
        "artifact_class": "exchange_sourced_backtest_graph_data",
        "tracked_repo_artifact": True,
        "data_source": "derived_from_tracked_exchange_sourced_optimization_reports",
        "source_report_count": len(rows),
        "latest_round_id": str(rows[-1].get("round_id") or "") if rows else None,
        "tracked_artifacts": tracked_artifacts,
        "artifact_integrity": _artifact_integrity_manifest(tracked_artifacts, report_path=report_path),
    }
    write_json_atomic(report_path, report, indent=2, sort_keys=True)
    return report


__all__ = ["build_optimization_progress_artifacts"]
