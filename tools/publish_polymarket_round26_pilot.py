"""Publish hash-bound Round 26 development evidence without upgrading its claims."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCHEMA_VERSION = "polymarket-round26-twap60-analysis-v2"
MANIFEST_SCHEMA_VERSION = "polymarket-round26-publication-manifest-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_analysis(path: Path) -> dict[str, object]:
    try:
        result = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Round 26 analysis is not strict JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("Round 26 analysis must be an object")
    body = dict(result)
    claimed = str(body.pop("analysis_sha256", "")).lower()
    if len(claimed) != 64 or claimed != _canonical_sha256(body):
        raise ValueError("Round 26 analysis hash differs")
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("data_role") != "development_only"
        or result.get("edge_claim") is not False
        or result.get("profitability_claim") is not False
        or result.get("paper_trading_authority") is not False
        or result.get("live_trading_authority") is not False
    ):
        raise ValueError("Round 26 analysis claim boundary differs")
    if not isinstance(result.get("taker_results"), list) or not isinstance(
        result.get("best_in_sample_trades"), list
    ):
        raise ValueError("Round 26 analysis tables differ")
    return result


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="ascii", newline="\n")
    temporary.replace(path)


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    lines: list[str] = []

    class _Sink:
        def write(self, value: str) -> int:
            lines.append(value)
            return len(value)

    writer = csv.DictWriter(_Sink(), fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in columns})
    return "".join(lines)


def _utc(milliseconds: int) -> datetime:
    return datetime.fromtimestamp(milliseconds / 1_000, tz=UTC)


def _trade_rows(result: Mapping[str, object]) -> list[dict[str, object]]:
    trades = result["best_in_sample_trades"]
    assert isinstance(trades, list)
    cumulative = 0.0
    rows: list[dict[str, object]] = []
    for index, trade in enumerate(trades, start=1):
        assert isinstance(trade, Mapping)
        cumulative += float(trade["net_pnl_quote"])
        rows.append(
            {
                "trade_index": index,
                **trade,
                "decision_utc": _utc(int(trade["decision_wall_ms"])).isoformat(),
                "entry_utc": _utc(int(trade["entry_wall_ms"])).isoformat(),
                "exit_utc": _utc(int(trade["exit_wall_ms"])).isoformat(),
                "cumulative_net_pnl_quote": f"{cumulative:.12g}",
            }
        )
    return rows


def _render(result: Mapping[str, object], destination: Path) -> None:
    started = _utc(int(result["capture_started_at_ms"]))
    ended = _utc(int(result["capture_ended_at_ms"]))
    trade_rows = _trade_rows(result)
    configurations = result["taker_results"]
    assert isinstance(configurations, list)
    plt.rcParams.update(
        {
            "axes.edgecolor": "#25333d",
            "axes.labelcolor": "#25333d",
            "axes.titlecolor": "#15232d",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": str(result["analysis_sha256"]),
            "xtick.color": "#52616b",
            "ytick.color": "#52616b",
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(11.4, 7.7), constrained_layout=True)
    figure.patch.set_facecolor("#ffffff")
    figure.suptitle(
        "Round 26 TWAP60 development pilot",
        fontsize=17,
        fontweight="bold",
        color="#15232d",
    )

    pnl_axis = axes[0]
    pnl_axis.axhline(0.0, color="#94a1aa", linewidth=1.0, zorder=0)
    pnl_axis.set_xlim(started, ended)
    if trade_rows:
        times = [_utc(int(row["exit_wall_ms"])) for row in trade_rows]
        cumulative = np.asarray(
            [float(row["cumulative_net_pnl_quote"]) for row in trade_rows]
        )
        pnl_axis.step(times, cumulative, where="post", color="#126e82", linewidth=2.2)
        pnl_axis.scatter(times, cumulative, color="#126e82", s=18, zorder=3)
        pnl_axis.annotate(
            f"Final {cumulative[-1]:+.4f} USDC",
            (times[-1], cumulative[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            color="#126e82",
            fontweight="bold",
        )
    else:
        pnl_axis.text(
            0.5,
            0.5,
            "No executable taker trades in the selected configuration",
            transform=pnl_axis.transAxes,
            ha="center",
            va="center",
            color="#65737c",
            fontweight="bold",
        )
    pnl_axis.set_title(
        "In-sample selected configuration: cumulative after-cost P&L",
        loc="left",
        fontweight="bold",
    )
    pnl_axis.set_ylabel("Net P&L (USDC)")
    pnl_axis.set_xlabel("Observed exit time (UTC)")
    pnl_axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    pnl_axis.grid(axis="y", color="#dce3e7", linewidth=0.8)
    pnl_axis.spines[["top", "right"]].set_visible(False)

    robustness_axis = axes[1]
    trade_counts = np.asarray([int(item["trade_count"]) for item in configurations])
    net_pnl = np.asarray([float(item["net_pnl_quote"]) for item in configurations])
    drawdown = np.asarray(
        [float(item["maximum_drawdown_quote"]) for item in configurations]
    )
    sizes = 18.0 + np.minimum(drawdown, 5.0) * 12.0
    colors = np.where(net_pnl > 0.0, "#2a7f62", "#c34a55")
    robustness_axis.axhline(0.0, color="#94a1aa", linewidth=1.0, zorder=0)
    robustness_axis.scatter(
        trade_counts,
        net_pnl,
        c=colors,
        s=sizes,
        alpha=0.58,
        edgecolors="none",
    )
    best = result.get("best_in_sample_taker_configuration")
    if isinstance(best, Mapping):
        robustness_axis.scatter(
            [int(best["trade_count"])],
            [float(best["net_pnl_quote"])],
            marker="*",
            color="#15232d",
            s=150,
            zorder=4,
            label="Selected in sample",
        )
        robustness_axis.legend(frameon=False, loc="best")
    robustness_axis.set_title(
        "Frozen 960-configuration robustness scan",
        loc="left",
        fontweight="bold",
    )
    robustness_axis.set_xlabel("Executed round trips")
    robustness_axis.set_ylabel("After-cost net P&L (USDC)")
    robustness_axis.grid(color="#dce3e7", linewidth=0.8)
    robustness_axis.spines[["top", "right"]].set_visible(False)

    gap_count = int(result["capture_stream_gap_count"])
    figure.text(
        0.01,
        0.005,
        (
            f"Development only | {started:%Y-%m-%d %H:%M:%S} to "
            f"{ended:%H:%M:%S} UTC | exact recorded books and fees | "
            f"stream gaps: {gap_count} | no edge or profitability claim"
        ),
        color="#52616b",
        fontsize=8.5,
    )
    figure.savefig(
        destination,
        format="svg",
        facecolor="white",
        metadata={"Creator": "simple-ai-trading evidence publisher", "Date": None},
    )
    plt.close(figure)
    svg = destination.read_text(encoding="utf-8")
    _write_text(destination, "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def _readme(result: Mapping[str, object]) -> str:
    started = _utc(int(result["capture_started_at_ms"]))
    ended = _utc(int(result["capture_ended_at_ms"]))
    best = result.get("best_in_sample_taker_configuration")
    if not isinstance(best, Mapping):
        best = {}
    pilot_passed = bool(result["pilot_passed"])
    gap_count = int(result["capture_stream_gap_count"])
    return f"""# Round 26 TWAP60 pilot

> Development evidence only. No edge, profitability, paper-trading, or live-trading authority.

![Round 26 development result](round26-development-result.svg)

| Evidence | Result |
|---|---:|
| Exact UTC capture | {started:%Y-%m-%d %H:%M:%S} to {ended:%Y-%m-%d %H:%M:%S} |
| Capture duration | {int(result['capture_duration_seconds'])} seconds |
| Resolved BTC 5-minute markets | {int(result['resolved_market_count'])} |
| Frozen configurations | {int(result['configuration_count'])} |
| Selected round trips | {int(best.get('trade_count', 0))} |
| Selected after-cost P&L | {float(best.get('net_pnl_quote', 0.0)):+.6f} USDC |
| Selected maximum drawdown | {float(best.get('maximum_drawdown_quote', 0.0)):.6f} USDC |
| Recorded stream gaps | {gap_count} |
| Pilot gate | {'Passed' if pilot_passed else 'Failed'} |

The configuration was selected and measured on the same one-hour development
sample. It is hypothesis generation, not out-of-sample evidence. A stream gap
also invalidates the pilot whenever the recorded gap count is nonzero. Exact
numeric data are in `round26-analysis.json`, `round26-configurations.csv`, and
`round26-selected-trades.csv`; the SVG is never the source of truth.
"""


def publish(analysis_path: Path, output_dir: Path) -> dict[str, object]:
    result = _load_analysis(analysis_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_copy = output_dir / "round26-analysis.json"
    shutil.copyfile(analysis_path, analysis_copy)

    configurations = result["taker_results"]
    assert isinstance(configurations, list)
    configuration_columns = (
        "lookback_ms",
        "threshold_bps",
        "signal_mode",
        "taker_delay_ms",
        "hold_ms",
        "eligible_signal_count",
        "trade_count",
        "unique_condition_count",
        "gross_pnl_quote",
        "fees_quote",
        "net_pnl_quote",
        "mean_net_pnl_quote",
        "win_rate",
        "maximum_drawdown_quote",
    )
    configuration_path = output_dir / "round26-configurations.csv"
    _write_text(
        configuration_path,
        _csv_text(configuration_columns, configurations),
    )
    trade_columns = (
        "trade_index",
        "condition_id",
        "outcome",
        "decision_wall_ms",
        "decision_utc",
        "entry_wall_ms",
        "entry_utc",
        "exit_wall_ms",
        "exit_utc",
        "entry_price",
        "exit_price",
        "gross_pnl_quote",
        "fees_quote",
        "net_pnl_quote",
        "cumulative_net_pnl_quote",
    )
    trade_path = output_dir / "round26-selected-trades.csv"
    _write_text(trade_path, _csv_text(trade_columns, _trade_rows(result)))
    chart_path = output_dir / "round26-development-result.svg"
    _render(result, chart_path)
    readme_path = output_dir / "README.md"
    _write_text(readme_path, _readme(result))

    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            analysis_copy,
            configuration_path,
            trade_path,
            chart_path,
            readme_path,
        )
    }
    body: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "analysis_sha256": result["analysis_sha256"],
        "capture_result_sha256": result["capture_result_sha256"],
        "capture_started_at_ms": result["capture_started_at_ms"],
        "capture_ended_at_ms": result["capture_ended_at_ms"],
        "diagnostic_only": True,
        "edge_claim": False,
        "profitability_claim": False,
        "files": files,
    }
    manifest = {**body, "manifest_sha256": _canonical_sha256(body)}
    _write_text(
        output_dir / "publication-manifest.json",
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            publish(args.analysis, args.output_dir),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
