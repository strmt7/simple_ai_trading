"""Conditional inverse-short algebra; no venue adapter or profitability gate.

All rates/prices are caller-supplied scenarios, not current Binance terms.
The residual delta applies only along the common-price path spot == mark.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


def _finite(value: Decimal, name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class InverseShortScenario:
    """One isolated, fixed-contract-count accounting scenario before execution.

    collateral_coin is the starting gross coin balance. net_coin_cashflow
    includes every subsequent signed coin debit/credit exactly once, including
    any coin converted out. Converted proceeds may enter outside_quote_cash,
    which is not margin collateral. No other positions or borrowing are modeled.
    maintenance_rate is an illustrative proportional requirement, NOT an
    exchange tier, liquidation threshold, or proof of account solvency.
    """

    contract_count: int
    contract_size_quote: Decimal
    entry_future: Decimal
    collateral_coin: Decimal
    mark_future: Decimal
    spot_conversion: Decimal
    net_coin_cashflow: Decimal
    outside_quote_cash: Decimal
    maintenance_rate: Decimal

    def evaluate(self) -> dict[str, Decimal]:
        if (
            isinstance(self.contract_count, bool)
            or not isinstance(self.contract_count, int)
            or self.contract_count <= 0
        ):
            raise ValueError("contract_count must be a positive integer")
        for name in (
            "contract_size_quote",
            "entry_future",
            "mark_future",
            "spot_conversion",
        ):
            _finite(getattr(self, name), name, positive=True)
        for name in (
            "collateral_coin",
            "net_coin_cashflow",
            "outside_quote_cash",
            "maintenance_rate",
        ):
            _finite(getattr(self, name), name)
        if self.collateral_coin < 0 or not 0 <= self.maintenance_rate < 1:
            raise ValueError(
                "nonnegative collateral and maintenance in [0, 1) required"
            )
        with localcontext() as context:
            context.prec = 50
            notional = self.contract_count * self.contract_size_quote
            target = notional / self.entry_future
            residual = self.collateral_coin - target + self.net_coin_cashflow
            pnl = notional / self.mark_future - target
            equity = residual + notional / self.mark_future
            maintenance = self.maintenance_rate * notional / self.mark_future
            return {
                "notional_quote": notional,
                "matched_collateral_coin": target,
                "short_pnl_coin": pnl,
                "equity_coin": equity,
                "equity_at_spot_quote": equity * self.spot_conversion,
                "total_wealth_quote": (
                    equity * self.spot_conversion + self.outside_quote_cash
                ),
                "common_price_residual_delta_coin": residual,
                "scenario_maintenance_coin": maintenance,
                "scenario_margin_surplus_coin": equity - maintenance,
                "inverse_component_at_spot_quote": (
                    notional * self.spot_conversion / self.mark_future
                ),
            }


def illustrative_scenarios() -> list[dict]:
    """Fixed synthetic counterexamples; not selected market data or benchmarks."""
    d = Decimal
    rows = []
    for label, coin_cashflow, quote_cash in (
        ("matched_no_cashflow", d("0"), d("0")),
        ("coin_fee_debit", d("-0.0001"), d("0")),
        ("coin_funding_retained", d("0.0001"), d("0")),
        ("coin_funding_converted_at_100", d("0"), d("0.01")),
    ):
        for price in (d("25"), d("100"), d("400")):
            scenario = InverseShortScenario(
                contract_count=10,
                contract_size_quote=d("10"),
                entry_future=d("100"),
                collateral_coin=d("1"),
                mark_future=price,
                spot_conversion=price,
                net_coin_cashflow=coin_cashflow,
                outside_quote_cash=quote_cash,
                maintenance_rate=d("0.005"),
            )
            rows.append(
                {
                    "label": label,
                    "inputs": {
                        key: str(value) for key, value in asdict(scenario).items()
                    },
                    "outputs": {
                        key: str(value) for key, value in scenario.evaluate().items()
                    },
                }
            )
    return rows


def write_review(output: Path) -> None:
    """Retain deterministic scenarios and exact source/implementation bindings."""
    root = Path(__file__).resolve().parents[1]
    sources = [
        "tools/review_inverse_collateral.py",
        "docs/review/2026-09-04/inverse-collateral-source-plan.json",
        "docs/review/2026-09-04/inverse-collateral-source-extraction.json",
    ]
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "conditional_synthetic_accounting_only",
        "official_mechanics_verified": False,
        "source_result": "All three FAQ opens redirected to the regional landing page.",
        "market_data_requests": 0,
        "account_requests": 0,
        "accepted_edge": False,
        "profitability_claim": False,
        "bindings": [
            {
                "path": path,
                "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
            }
            for path in sources
        ],
        "scenarios": illustrative_scenarios(),
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["result_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    write_review(parser.parse_args().output)
