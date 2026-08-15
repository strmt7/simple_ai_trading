from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib


matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "simple-ai-trading-round22-diagnostic-v1"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).parents[1]
RESULT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-022-diagnostic-results-v1.json"
)
OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-022-diagnostic-performance.svg"
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
        raise ValueError("Round 22 diagnostic result is not an object")
    result = dict(decoded)
    claimed = str(result.pop("result_sha256", ""))
    actual = hashlib.sha256(_canonical_json(result).encode("ascii")).hexdigest()
    if claimed != actual or result.get("diagnostic_pass") is not False:
        raise ValueError("Round 22 diagnostic graph source differs")
    return {**result, "result_sha256": claimed}


def _metric_values(
    result: Mapping[str, object],
    metric: str,
) -> tuple[list[float], list[float]]:
    calibration = result["calibration_partition"]
    selection = result["selection"]
    assert isinstance(calibration, Mapping)
    assert isinstance(selection, Mapping)
    calibration_report = calibration["calibrated"]
    selection_report = selection["calibrated"]
    assert isinstance(calibration_report, Mapping)
    assert isinstance(selection_report, Mapping)
    return (
        [
            float(calibration_report["baseline"][metric]),
            float(selection_report["baseline"][metric]),
        ],
        [
            float(calibration_report["model"][metric]),
            float(selection_report["model"][metric]),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the audited Round 22 graph.")
    parser.add_argument("--qa-png", type=Path)
    args = parser.parse_args()
    result = _load_result()
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), dpi=120)
    figure.patch.set_facecolor("#ffffff")
    baseline_color = "#343a40"
    model_color = "#16867a"
    for axis, metric, title in zip(
        axes[:2],
        ("brier_score", "log_loss"),
        ("Brier score", "Log loss"),
        strict=True,
    ):
        baseline, model = _metric_values(result, metric)
        x = np.arange(2)
        width = 0.34
        baseline_bars = axis.bar(
            x - width / 2,
            baseline,
            width,
            color=baseline_color,
            label="Market prior",
        )
        model_bars = axis.bar(
            x + width / 2,
            model,
            width,
            color=model_color,
            label="L2 residual",
        )
        axis.bar_label(baseline_bars, fmt="%.4f", padding=3, fontsize=8)
        axis.bar_label(model_bars, fmt="%.4f", padding=3, fontsize=8)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.set_xticks(x, ("Calibration", "Selection"))
        axis.set_ylim(0, max((*baseline, *model)) * 1.24)
        axis.grid(axis="y", color="#d9dee3", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    selection = result["selection"]
    assert isinstance(selection, Mapping)
    horizons = selection["by_elapsed_horizon"]
    assert isinstance(horizons, list)
    labels: list[str] = []
    brier_delta: list[float] = []
    log_delta: list[float] = []
    for item in horizons:
        assert isinstance(item, Mapping)
        elapsed = item["elapsed_seconds"]
        baseline = item["baseline"]
        model = item["model"]
        assert isinstance(elapsed, list)
        assert isinstance(baseline, Mapping)
        assert isinstance(model, Mapping)
        labels.append(f"{elapsed[0]}-{elapsed[1]}")
        brier_delta.append(float(baseline["brier_score"]) - float(model["brier_score"]))
        log_delta.append(float(baseline["log_loss"]) - float(model["log_loss"]))
    x = np.arange(len(labels))
    axes[2].axhline(0, color="#59636e", linewidth=0.9)
    axes[2].plot(
        x,
        brier_delta,
        color="#16867a",
        marker="o",
        linewidth=2,
        label="Brier delta",
    )
    axes[2].plot(
        x,
        log_delta,
        color="#c43d3d",
        marker="s",
        linewidth=2,
        label="Log-loss delta",
    )
    axes[2].set_title(
        "Selection improvement by elapsed time",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_xticks(x, labels)
    axes[2].set_xlabel("Elapsed seconds")
    axes[2].grid(axis="y", color="#d9dee3", linewidth=0.7)
    axes[2].set_axisbelow(True)
    axes[2].spines[["top", "right", "left"]].set_visible(False)
    axes[2].tick_params(axis="y", length=0)
    axes[2].legend(frameon=False, fontsize=9, loc="lower left")

    bootstrap = selection["bootstrap"]
    assert isinstance(bootstrap, Mapping)
    subtitle = (
        "Diagnostic failed | selection improvement probability: "
        f"Brier {100 * float(bootstrap['brier']['improvement_probability']):.1f}% | "
        f"log loss {100 * float(bootstrap['log_loss']['improvement_probability']):.1f}%"
    )
    figure.suptitle(
        "Round 22: executable market prior vs causal L2 residual",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(0.055, 0.925, subtitle, ha="left", fontsize=10, color="#4f5963")
    figure.text(
        0.055,
        0.015,
        "Lower scores are better. Horizon delta = prior loss - model loss; positive favors the model. Diagnostic only; no PnL or edge claim.",
        ha="left",
        fontsize=8.5,
        color="#4f5963",
    )
    figure.tight_layout(rect=(0.04, 0.07, 0.99, 0.88), w_pad=2.2)
    figure.savefig(
        OUTPUT,
        format="svg",
        metadata={"Creator": "simple-ai-trading Round 22 renderer", "Date": None},
    )
    if args.qa_png is not None:
        qa_output = args.qa_png.resolve()
        qa_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            qa_output,
            format="png",
            dpi=160,
            metadata={"Software": "simple-ai-trading Round 22 renderer"},
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
