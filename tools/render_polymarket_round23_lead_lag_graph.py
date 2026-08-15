from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "simple-ai-trading-round23-lead-lag-v1"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).parents[1]
RESULT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-023-lead-lag-results-v1.json"
)
OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-023-lead-lag-performance.svg"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load_result() -> dict[str, object]:
    decoded = json.loads(RESULT.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("Round 23 lead-lag result is not an object")
    result = dict(decoded)
    claimed = str(result.pop("result_sha256", ""))
    actual = hashlib.sha256(_canonical_json(result).encode("ascii")).hexdigest()
    if claimed != actual or result.get("mechanism_gate_passed") is not True:
        raise ValueError("Round 23 lead-lag graph source differs")
    return {**result, "result_sha256": claimed}


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#d7dde3", linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the audited Round 23 graph.")
    parser.add_argument("--qa-png", type=Path)
    args = parser.parse_args()
    result = _load_result()
    calibration = result["calibration"]
    selection = result["selection"]
    bootstrap = result["bootstrap"]
    assert isinstance(calibration, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(bootstrap, Mapping)
    calibration_baseline = calibration["baseline_metrics"]
    calibration_candidate = calibration["candidate_metrics"]
    selection_baseline = selection["baseline_metrics"]
    selection_candidate = selection["candidate_metrics"]
    improvements = selection["condition_mse_improvements"]
    assert isinstance(calibration_baseline, Mapping)
    assert isinstance(calibration_candidate, Mapping)
    assert isinstance(selection_baseline, Mapping)
    assert isinstance(selection_candidate, Mapping)
    assert isinstance(improvements, list)

    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), dpi=120)
    figure.patch.set_facecolor("#ffffff")
    baseline_color = "#374151"
    candidate_color = "#087f5b"
    negative_color = "#c2413b"

    x = np.arange(2)
    width = 0.34
    baseline_mse = [
        float(calibration_baseline["condition_equal_mse"]),
        float(selection_baseline["condition_equal_mse"]),
    ]
    candidate_mse = [
        float(calibration_candidate["condition_equal_mse"]),
        float(selection_candidate["condition_equal_mse"]),
    ]
    baseline_bars = axes[0].bar(
        x - width / 2,
        baseline_mse,
        width,
        color=baseline_color,
        label="Polymarket only",
    )
    candidate_bars = axes[0].bar(
        x + width / 2,
        candidate_mse,
        width,
        color=candidate_color,
        label="+ Binance public data",
    )
    axes[0].bar_label(baseline_bars, fmt="%.4f", padding=3, fontsize=8)
    axes[0].bar_label(candidate_bars, fmt="%.4f", padding=3, fontsize=8)
    axes[0].set_title("One-second forecast error", loc="left", fontweight="bold")
    axes[0].set_ylabel("Condition-equal MSE (lower is better)")
    axes[0].set_xticks(x, ("Calibration", "Exploratory selection"))
    axes[0].set_ylim(0, max((*baseline_mse, *candidate_mse)) * 1.25)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    _style_axis(axes[0])

    metric_names = ("Direction accuracy", "Correlation")
    baseline_quality = [
        float(selection_baseline["changed_row_direction_accuracy"]),
        float(selection_baseline["pearson_correlation"]),
    ]
    candidate_quality = [
        float(selection_candidate["changed_row_direction_accuracy"]),
        float(selection_candidate["pearson_correlation"]),
    ]
    baseline_quality_bars = axes[1].bar(
        x - width / 2,
        baseline_quality,
        width,
        color=baseline_color,
    )
    candidate_quality_bars = axes[1].bar(
        x + width / 2,
        candidate_quality,
        width,
        color=candidate_color,
    )
    axes[1].bar_label(baseline_quality_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1].bar_label(candidate_quality_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[1].set_title("Exploratory selection quality", loc="left", fontweight="bold")
    axes[1].set_xticks(x, metric_names)
    axes[1].set_ylim(0, 0.75)
    _style_axis(axes[1])

    condition_values = sorted(
        float(item["mse_improvement"])
        for item in improvements
        if isinstance(item, Mapping)
    )
    colors = [
        candidate_color if value > 0 else negative_color for value in condition_values
    ]
    axes[2].bar(np.arange(len(condition_values)), condition_values, color=colors)
    axes[2].axhline(0, color="#59636e", linewidth=0.9)
    axes[2].set_title("MSE improvement by condition", loc="left", fontweight="bold")
    axes[2].set_xlabel("12 conditions, sorted")
    axes[2].set_ylabel("Baseline MSE - candidate MSE")
    axes[2].set_xticks((0, len(condition_values) - 1), ("lowest", "highest"))
    _style_axis(axes[2])

    relative_improvement = (
        100.0
        * float(selection["mse_improvement"])
        / float(selection_baseline["condition_equal_mse"])
    )
    interval = bootstrap["mse_improvement_95_interval"]
    assert isinstance(interval, list)
    figure.suptitle(
        "Round 23: public BTC flow predicts the next Polymarket probability move",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.925,
        f"Exploratory MSE reduction {relative_improvement:.2f}% | "
        f"clustered 95% improvement interval [{float(interval[0]):.6f}, "
        f"{float(interval[1]):.6f}] | all 12 conditions improved",
        ha="left",
        fontsize=9.5,
        color="#4f5963",
    )
    figure.text(
        0.055,
        0.012,
        "Predictive event-time diagnostic only. No spread, queue, fill, receipt-latency, PnL, profitability, or trading-authority claim; fresh receipt-time holdout required.",
        ha="left",
        fontsize=8.2,
        color="#4f5963",
    )
    figure.tight_layout(rect=(0.04, 0.065, 0.99, 0.88), w_pad=2.25)
    figure.savefig(
        OUTPUT,
        format="svg",
        metadata={
            "Creator": "simple-ai-trading Round 23 renderer",
            "Date": None,
            "Description": f"source-result-sha256:{result['result_sha256']}",
        },
    )
    if args.qa_png is not None:
        qa_output = args.qa_png.resolve()
        qa_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            qa_output,
            format="png",
            dpi=160,
            metadata={"Software": "simple-ai-trading Round 23 renderer"},
        )
    plt.close(figure)
    print(
        _canonical_json(
            {
                "output": str(OUTPUT),
                "qa_png": str(args.qa_png.resolve()) if args.qa_png else None,
                "result_sha256": result["result_sha256"],
                "svg_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
