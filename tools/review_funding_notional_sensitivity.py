"""Offline exploratory sensitivity, not a rerun/promotion of the funding study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools.adjudicate_binance_backpack_funding_v2 import _timestamp_ms

ROOT = Path(__file__).resolve().parents[1]
BASE = "docs/model-research/action-value/"
STEM = "binance-backpack-btc-eth-sol-funding-adjudication-"
D = Decimal
HOUR = 3_600_000


def canonical(value: dict, field: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in value.items() if k != field},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def sensitivity(coefficients: list[Decimal], hurdle: Decimal) -> dict:
    """Independent positive notional weights: an outer bound, not a price path."""
    if not coefficients or any(not x.is_finite() for x in coefficients):
        raise ValueError("nonempty finite settlement coefficients required")
    if not hurdle.is_finite() or hurdle < 0:
        raise ValueError("finite nonnegative fixed hurdle required")
    gross = sum(coefficients, D(0)) * 10_000
    absolute = sum((abs(x) for x in coefficients), D(0)) * 10_000
    net = gross - hurdle
    return {
        "settlement_cashflow_count": len(coefficients),
        "unit_weight_gross_bips": str(gross),
        "absolute_settlement_coefficients_bips": str(absolute),
        "unchanged_frozen_hurdle_bips": str(hurdle),
        "unit_weight_net_bips": str(net),
        "symmetric_weight_radius_to_zero_net": (
            str(abs(net) / absolute) if absolute else None
        ),
        "illustrative_outer_bounds": [
            {
                "radius": str(radius),
                "lower_net_bips": str(net - radius * absolute),
                "upper_net_bips": str(net + radius * absolute),
            }
            for radius in (D("0.10"), D("0.20"))
        ],
    }


def run(output: Path) -> None:
    if output.exists():
        raise RuntimeError("review output already exists")
    bindings = []

    def read(path: str, expected: str | None = None) -> bytes:
        resolved = (ROOT / path).resolve()
        resolved.relative_to(ROOT)
        raw = resolved.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise RuntimeError(f"retained input binding changed: {path}")
        bindings.append({"path": path, "file_sha256": digest})
        return raw

    journal_path = output.with_suffix(".journal.jsonl")
    with journal_path.open("x", encoding="utf-8") as journal:

        def record(phase: str) -> None:
            journal.write(
                json.dumps(
                    {
                        "phase": phase,
                        "time_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
            journal.flush()

        record("started_offline_exploratory_review")
        try:
            contract = json.loads(read(BASE + STEM + "contract-v2-2026-09-01.json"))
            prior = json.loads(read(BASE + STEM + "result-v2-2026-09-01.json"))
            for obj, key in ((contract, "contract_sha256"), (prior, "result_sha256")):
                if canonical(obj, key) != obj[key]:
                    raise RuntimeError("retained canonical hash mismatch")
            if prior["contract"]["sha256"] != contract["contract_sha256"]:
                raise RuntimeError("prior result refers to another contract")
            for path, digest in (
                (
                    "tools/adjudicate_binance_backpack_funding.py",
                    "d20cad18fb7a21409921e445273db4094094adb49c6c3acf8a32a4e0267ff623",
                ),
                (
                    "tools/adjudicate_binance_backpack_funding_v2.py",
                    "9ba7d2db3a6fb41318b8f5358424f88c8a1d26f7d3a693305d4c4b4707586f83",
                ),
                (
                    "docs/model-research/binance/raw/backpack-btc-eth-sol-funding-v1-2026-09-01/07-backpack-hourly-utc-anchor.raw.md",
                    "64ef1f561e6d095d6a732a770b07dac8460d543ea592e148a517e1a7239c45b2",
                ),
            ):
                read(path, digest)
            results = {}
            for source in contract["sources"]:
                rows = {
                    venue: json.loads(
                        read(source[venue]["path"], source[venue]["file_sha256"])
                    )
                    for venue in ("backpack", "binance")
                }
                series = {}
                for venue, items in rows.items():
                    series[venue] = {}
                    for item in items:
                        if item["symbol"] != source[venue]["symbol"]:
                            raise RuntimeError("symbol mismatch")
                        timestamp = (
                            _timestamp_ms(item["intervalEndTimestamp"])
                            if venue == "backpack"
                            else int(item["fundingTime"])
                        )
                        if venue == "binance":
                            snapped = ((timestamp + 4 * HOUR) // (8 * HOUR)) * 8 * HOUR
                            if (
                                abs(snapped - timestamp)
                                > contract["population"][
                                    "maximum_binance_schedule_jitter_ms"
                                ]
                            ):
                                raise RuntimeError("schedule mismatch")
                            timestamp = snapped
                        if timestamp in series[venue]:
                            raise RuntimeError("duplicate settlement")
                        series[venue][timestamp] = D(str(item["fundingRate"]))
                asset_prior = prior["asset_results"][source["asset"]]
                orientation = asset_prior["orientation_selected_from_training_only"]
                signs = {
                    "short_backpack_USDC_long_binance_USDT": D(1),
                    "long_backpack_USDC_short_binance_USDT": D(-1),
                }
                sign = signs[orientation]
                asset_result = {"unchanged_orientation": orientation, "roles": {}}
                for role, bounds in contract["roles"].items():
                    coefficients = []
                    for i in range(bounds["start"], bounds["stop"]):
                        t = (
                            contract["population"]["first_aligned_bucket_ms"]
                            + i * 8 * HOUR
                        )
                        coefficients.extend(
                            sign * series["backpack"][t - offset * HOUR]
                            for offset in range(7, -1, -1)
                        )
                        coefficients.append(-sign * series["binance"][t])
                    old = asset_prior["roles"][role]
                    hurdle = sum(
                        (
                            D(old[key])
                            for key in (
                                "round_trip_execution_bips",
                                "two_leg_capital_hurdle_bips",
                                "usdc_usdt_quote_unit_stress_bips",
                                "custody_latency_failure_stress_bips",
                            )
                        ),
                        D(0),
                    )
                    result = sensitivity(coefficients, hurdle)
                    if D(result["unit_weight_gross_bips"]) != D(
                        old["gross_funding_spread_bips"]
                    ):
                        raise RuntimeError("retained role arithmetic not reconstructed")
                    result["retained_role_gross_reconstructed"] = True
                    asset_result["roles"][role] = result
                asset_result["raw_field_sets"] = {
                    venue: sorted({key for row in items for key in row})
                    for venue, items in rows.items()
                }
                results[source["asset"]] = asset_result
            result = {
                "schema_version": "funding-notional-sensitivity-review-v1",
                "study_kind": "retrospective_exploratory_sensitivity_not_validation",
                "promotion_eligible": False,
                "market_data_requests": 0,
                "historical_results_changed": False,
                "interpretation": "Weights are settlement quote notionals converted to a common numeraire divided by initial reference notional. The 10/20 percent ranges are hypothetical independent-coordinate outer bounds, not observed ranges or feasible jointly delta-neutral price paths. Frozen costs stay fixed solely to isolate this sensitivity; they are not actual weighted strategy costs. A zero-crossing is not evidence of a profitable path or satisfaction of any retry trigger.",
                "inputs": bindings,
                "implementation_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "assets": results,
            }
            result["result_sha256"] = canonical(result, "result_sha256")
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            record("completed")
            print(
                json.dumps(
                    {asset: row["roles"]["test"] for asset, row in results.items()}
                )
            )
        except Exception:
            record("failed_do_not_overwrite_or_refetch")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
