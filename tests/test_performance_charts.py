from __future__ import annotations

from simple_ai_trading.performance_charts import EquityPoint, render_equity_svg, write_equity_svg


def test_render_equity_svg_contains_day_trading_performance_labels() -> None:
    svg = render_equity_svg(
        [
            EquityPoint(0, 1000.0, 0.0, 1704067200000),
            EquityPoint(1, 980.0, 0.02, 1704153600000),
            EquityPoint(2, 1030.0, 0.01, 1704240000000),
        ]
    )

    assert svg.startswith("<svg")
    assert "day-trading simulation timeline" in svg
    assert "2024-01-01" in svg
    assert "years" in svg
    assert "max drawdown" in svg


def test_write_equity_svg_creates_parent_directory(tmp_path) -> None:
    path = write_equity_svg([EquityPoint(0, 1.0, 0.0)], tmp_path / "charts" / "backtest.svg")

    assert path.exists()
    assert "<polyline" in path.read_text(encoding="utf-8")
