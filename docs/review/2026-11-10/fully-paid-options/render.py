"""Render constructed payout laws; exact rational results remain authoritative."""

from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simple_ai_trading.option_long_floor import (
    CappedExerciseFee,
    evaluate_fully_paid_long_option_floor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    fee = CappedExerciseFee(Decimal("0.1"), Decimal("0.5"))
    result = evaluate_fully_paid_long_option_floor(
        lower_call_strike=Decimal("100"),
        upper_put_strike=Decimal("300"),
        matched_base_quantity=Decimal("1"),
        call_exercise_fee=fee,
        put_exercise_fee=fee,
        terminal_cost_reserve_quote=Decimal("0"),
    )
    # All kinks plus one tail point: connecting these is exact on this interval.
    prices = sorted(
        {point.settlement_price for point in result.critical_points} | {Fraction(400)}
    )
    gross, net = [], []
    for price in prices:
        call, put = max(price - 100, 0), max(300 - price, 0)
        gross.append(float(call + put))
        net.append(
            float(call + put - min(price / 10, call / 2) - min(price / 10, put / 2))
        )
    with plt.rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": "supplied-long-option-floor"}
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.6))
        try:
            axis.axvspan(100, 300, color="#edf2f7", zorder=0)
            axis.plot(
                [float(price) for price in prices],
                gross,
                color="#245c8c",
                label="Gross payout",
            )
            axis.plot(
                [float(price) for price in prices],
                net,
                color="#ba4a14",
                label="After supplied exercise fees",
            )
            axis.axhline(
                178.5,
                color="#666666",
                linestyle="--",
                linewidth=1,
                label="Hypothetical cash alternative: 178.5",
            )
            minimum_price = float(result.minimizing_breakpoints[0])
            minimum_payout = float(result.terminal_net_payout_floor_quote)
            axis.scatter([minimum_price], [minimum_payout], color="#ba4a14", zorder=3)
            axis.annotate(
                f"Exact minimum: {minimum_payout:g} at S = {minimum_price:g}",
                xy=(minimum_price, minimum_payout),
                xytext=(160, 95),
                arrowprops={"arrowstyle": "->", "color": "#ba4a14"},
                fontsize=10,
            )
            axis.set(
                xlim=(0, 400),
                ylim=(0, 330),
                xlabel="Common settlement price S",
                ylabel="Payout in quote units",
                title="Constructed two-long-option payout",
            )
            axis.set_xticks([0, 100, 125, 250, 300, 400])
            axis.grid(axis="y", alpha=0.2)
            axis.legend(loc="lower left", fontsize=9)
            figure.text(
                0.12,
                0.02,
                "Supplied inputs, not venue fees or market prices. No qualified edge.",
                fontsize=9,
            )
            figure.tight_layout(rect=(0, 0.06, 1, 1))
            with BytesIO() as buffer:
                figure.savefig(
                    buffer,
                    format="svg",
                    metadata={"Date": None, "Creator": "Supplied-law research"},
                )
                # Normalize generated path-line padding, retaining line breaks
                # as SVG whitespace separators and leaving geometry unchanged.
                svg = (
                    "\n".join(
                        line.rstrip()
                        for line in buffer.getvalue().decode("utf-8").splitlines()
                    )
                    + "\n"
                )
            with args.output.open("xb") as stream:
                stream.write(svg.encode("utf-8"))
            if args.preview is not None:
                with args.preview.open("xb") as stream:
                    figure.savefig(stream, format="png", dpi=150)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    main()
