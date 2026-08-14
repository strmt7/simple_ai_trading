"""Render truthful Round 25 diagnostic evidence from its hashed result."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simple_ai_trading.polymarket_round25_forensic_model import (  # noqa: E402
    validate_round25_forensic_result,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="ascii", newline="\n")
    temporary.replace(path)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, Mapping):
        raise ValueError("Round 25 forensic result is not an object")
    return validate_round25_forensic_result(value)


def _trade_csv(result: Mapping[str, object]) -> str:
    columns = (
        "trade_index",
        "condition_id",
        "event_start_ms",
        "event_start_utc",
        "outcome",
        "resolved_up",
        "entry_cost_quote",
        "fee_quote",
        "net_pnl_quote",
        "cumulative_net_pnl_quote",
    )
    lines: list[str] = []

    class _Sink:
        def write(self, value: str) -> int:
            lines.append(value)
            return len(value)

    writer = csv.DictWriter(_Sink(), fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    trades = result["trade_results"]
    assert isinstance(trades, list)
    for index, trade in enumerate(trades, start=1):
        assert isinstance(trade, Mapping)
        event_start = datetime.fromtimestamp(int(trade["event_start_ms"]) / 1_000, tz=UTC)
        writer.writerow(
            {
                "trade_index": index,
                **trade,
                "event_start_utc": event_start.isoformat(),
            }
        )
    return "".join(lines)


def _metrics_csv(result: Mapping[str, object]) -> str:
    prior = result["market_prior_metrics"]
    selected = result["selected_metrics"]
    assert isinstance(prior, Mapping) and isinstance(selected, Mapping)
    rows = ["metric,market_prior,selected_model\n"]
    for metric in sorted(prior):
        rows.append(f"{metric},{float(prior[metric]):.12g},{float(selected[metric]):.12g}\n")
    return "".join(rows)


def _render(result: Mapping[str, object], destination: Path) -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": "#24323d",
            "axes.labelcolor": "#24323d",
            "axes.titlecolor": "#15232d",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": str(result["result_sha256"]),
            "xtick.color": "#52616b",
            "ytick.color": "#52616b",
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(11.2, 7.2), constrained_layout=True)
    figure.patch.set_facecolor("#ffffff")
    figure.suptitle(
        "Round 25 forensic diagnostic: sealed selection evidence",
        fontsize=17,
        fontweight="bold",
        color="#15232d",
    )
    trades = result["trade_results"]
    assert isinstance(trades, list)
    pnl_axis = axes[0]
    pnl_axis.axhline(0.0, color="#9aa7b0", linewidth=1.0, zorder=0)
    if trades:
        x = [
            datetime.fromtimestamp(int(trade["event_start_ms"]) / 1_000, tz=UTC)
            for trade in trades
        ]
        cumulative = np.asarray(
            [float(trade["cumulative_net_pnl_quote"]) for trade in trades]
        )
        pnl_axis.step(x, cumulative, where="post", color="#16697a", linewidth=2.3)
        pnl_axis.scatter(x, cumulative, color="#16697a", s=24, zorder=3)
        pnl_axis.annotate(
            f"Final {cumulative[-1]:+.3f} USDC",
            (x[-1], cumulative[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            color="#16697a",
            fontweight="bold",
        )
        pnl_axis.tick_params(axis="x", rotation=20)
    else:
        pnl_axis.text(
            0.5,
            0.5,
            "No entry cleared the frozen after-fee edge and depth gates",
            transform=pnl_axis.transAxes,
            ha="center",
            va="center",
            color="#6b7780",
            fontweight="bold",
        )
        pnl_axis.set_xticks([])
    pnl_axis.set_title("After-cost cumulative P&L", loc="left", fontweight="bold")
    pnl_axis.set_xlabel("Market start (UTC)")
    pnl_axis.set_ylabel("Net P&L (USDC)")
    pnl_axis.grid(axis="y", color="#dce3e7", linewidth=0.8)
    pnl_axis.spines[["top", "right"]].set_visible(False)

    metric_axis = axes[1]
    prior = result["market_prior_metrics"]
    selected = result["selected_metrics"]
    assert isinstance(prior, Mapping) and isinstance(selected, Mapping)
    labels = ("Log loss", "Brier score")
    keys = ("condition_equal_log_loss", "condition_equal_brier_score")
    positions = np.arange(2)
    width = 0.34
    prior_values = np.asarray([float(prior[key]) for key in keys])
    selected_values = np.asarray([float(selected[key]) for key in keys])
    prior_bars = metric_axis.bar(
        positions - width / 2,
        prior_values,
        width,
        label="Market prior",
        color="#7b8790",
    )
    selected_bars = metric_axis.bar(
        positions + width / 2,
        selected_values,
        width,
        label=str(result["selected_candidate_id"]),
        color="#d1495b",
    )
    metric_axis.bar_label(prior_bars, fmt="%.4f", padding=3, fontsize=9)
    metric_axis.bar_label(selected_bars, fmt="%.4f", padding=3, fontsize=9)
    metric_axis.set_xticks(positions, labels)
    metric_axis.set_title(
        "Condition-equal predictive loss (lower is better)",
        loc="left",
        fontweight="bold",
    )
    metric_axis.legend(frameon=False, loc="upper right")
    metric_axis.grid(axis="y", color="#dce3e7", linewidth=0.8)
    metric_axis.spines[["top", "right"]].set_visible(False)
    observed = datetime.fromtimestamp(int(result["created_at_ms"]) / 1_000, tz=UTC)
    figure.text(
        0.01,
        0.005,
        (
            f"Diagnostic only | {result['selection_condition_count']} BTC 5-minute markets | "
            f"+1 tick adverse entry stress | exact fee curve | {observed:%Y-%m-%d %H:%M UTC}"
        ),
        color="#52616b",
        fontsize=8.5,
    )
    figure.savefig(
        destination,
        format="svg",
        facecolor="white",
        metadata={"Creator": "simple-ai-trading evidence renderer", "Date": None},
    )
    plt.close(figure)
    svg = destination.read_text(encoding="utf-8")
    _write_text(
        destination,
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _load(args.result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trade_path = args.output_dir / "round25-selection-trades.csv"
    metric_path = args.output_dir / "round25-selection-metrics.csv"
    chart_path = args.output_dir / "round25-selection-diagnostic.svg"
    _write_text(trade_path, _trade_csv(result))
    _write_text(metric_path, _metrics_csv(result))
    _render(result, chart_path)
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (trade_path, metric_path, chart_path)
    }
    body = {
        "chart_family": "time_change_and_baseline_comparison",
        "diagnostic_only": True,
        "files": files,
        "result_sha256": result["result_sha256"],
        "schema_version": "polymarket-round25-forensic-figure-manifest-v1",
    }
    manifest = {**body, "manifest_sha256": hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()}
    _write_text(
        args.output_dir / "round25-selection-figure-manifest.json",
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
