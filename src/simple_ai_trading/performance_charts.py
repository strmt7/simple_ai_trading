"""Dependency-light financial chart rendering for backtest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EquityPoint:
    index: int
    equity: float
    drawdown: float = 0.0
    timestamp_ms: int | None = None


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _scale(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high <= low:
        return (out_low + out_high) / 2.0
    ratio = (value - low) / (high - low)
    return out_low + ratio * (out_high - out_low)


def _date_label(ts_ms: int | None) -> str:
    if ts_ms is None:
        return ""
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def _timescale_label(data: list[EquityPoint]) -> tuple[str, str, str]:
    timestamps = [int(point.timestamp_ms) for point in data if point.timestamp_ms is not None]
    if len(timestamps) < 2:
        return "sample index", "", ""
    start = min(timestamps)
    end = max(timestamps)
    days = max(0.0, (end - start) / (24 * 60 * 60 * 1000))
    years = days / 365.25
    return (
        f"{_date_label(start)} to {_date_label(end)} ({days:.1f} days / {years:.2f} years)",
        _date_label(start),
        _date_label(end),
    )


def render_equity_svg(points: Iterable[EquityPoint], *, title: str = "Backtest performance") -> str:
    data = list(points)
    if not data:
        data = [EquityPoint(0, 0.0, 0.0)]
    width = 960
    height = 540
    left = 64
    right = 24
    top = 52
    bottom = 58
    chart_w = width - left - right
    chart_h = height - top - bottom
    equities = [point.equity for point in data]
    drawdowns = [point.drawdown for point in data]
    min_equity = min(equities)
    max_equity = max(equities)
    max_drawdown = max(0.01, max(drawdowns))
    count = max(1, len(data) - 1)
    timescale, start_label, end_label = _timescale_label(data)
    equity_points = []
    drawdown_points = []
    for idx, point in enumerate(data):
        x = left + (idx / count) * chart_w
        y_equity = _scale(point.equity, min_equity, max_equity, top + chart_h, top)
        y_dd = _scale(point.drawdown, 0.0, max_drawdown, top + chart_h, top + chart_h * 0.62)
        equity_points.append((x, y_equity))
        drawdown_points.append((x, y_dd))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <text x="{left}" y="32" font-family="Segoe UI, Arial, sans-serif" font-size="22" fill="#111827">{title}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#94a3b8"/>
  <line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#94a3b8"/>
  <text x="{left}" y="{height - 20}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">day-trading simulation timeline: {timescale}</text>
  <text x="{left + chart_w - 170}" y="{height - 20}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334155">equity and drawdown</text>
  <text x="{left}" y="{top + chart_h + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{start_label}</text>
  <text x="{left + chart_w - 82}" y="{top + chart_h + 14}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#475569">{end_label}</text>
  <polyline points="{_polyline(drawdown_points)}" fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="5 4"/>
  <polyline points="{_polyline(equity_points)}" fill="none" stroke="#0f766e" stroke-width="3"/>
  <circle cx="{equity_points[-1][0]:.2f}" cy="{equity_points[-1][1]:.2f}" r="4" fill="#0f766e"/>
  <text x="{left}" y="{top + chart_h + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#111827">start {equities[0]:.2f} | end {equities[-1]:.2f} | max drawdown {max_drawdown:.2%}</text>
</svg>
"""


def write_equity_svg(points: Iterable[EquityPoint], path: str | Path, *, title: str = "Backtest performance") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_equity_svg(points, title=title), encoding="utf-8")
    return output
