"""Publish hash-verified Round 58 two-sided maker feasibility evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.publish_round52_executable_support_hurdle import (  # noqa: E402
    COLORS,
    _artifact,
    _bar_svg,
    _canonical_json,
    _canonical_sha256,
    _file_sha256,
    _line_svg,
    _read_object,
    _validate_finite,
    _write_csv,
    _write_text,
)


ROUND = 58
REPORT_SCHEMA = "round-058-two-sided-maker-support-v1"
PUBLICATION_SCHEMA = "round-058-two-sided-maker-feasibility-publication-v1"
REPORT_CANONICAL_SHA256 = (
    "46da40b3ddd8818ebfd45ea0d8a1cec260e88354d6b952fbdc9b2e0d1a2c2bdd"
)
REPORT_FILE_SHA256 = "0471f049129ba58556c50412c0681f425faaf25696657ac45f92afd94810d18e"
IMPLEMENTATION_COMMIT = "58aa5a52b8fd8bde2a329fbf8d49841d96405a0a"
PROBE_PATH = "tools/probe_round58_two_sided_maker_support.py"
PROBE_BLOB_OID = "adef23780ae6a682a4434e30f21b8e82ab55857c"
COST_CONTRACT_SCHEMA = "round-057-queue-censored-make-take-execution-contract-v1"
COST_CONTRACT_SHA256 = (
    "ef42dcd1fcf003838a34c78a3d87a49b45d78f16b7be47b596fc9eece9841dd6"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
STATE_LABELS = ("none", "bid_only", "ask_only", "both")
FILL_BUCKETS = {
    0: "unfilled_by_15s",
    1: "filled_by_5s",
    2: "filled_5s_to_10s",
    3: "filled_10s_to_15s",
}


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _canonical_value(value: Mapping[str, object], digest_key: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(digest_key, ""))
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError(f"{digest_key} does not match canonical content")
    return actual


def _validate_source(
    report_path: Path, cost_contract_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if _file_sha256(report_path) != REPORT_FILE_SHA256:
        raise ValueError("Round 58 report file hash drifted")
    report = _read_object(report_path, "Round 58 support report")
    contract = _read_object(cost_contract_path, "Round 57 cost reference")
    expected_contract = {
        "decision_cadence_seconds": 10,
        "placement_latency_ms": 750,
        "maximum_quote_age_ms": 1000,
        "order_notional_quote_per_side": 500.0,
        "maker_order_expiry_ms": 15000,
        "full_displayed_l1_queue_ahead": True,
        "own_order_quantity_included": True,
        "cancellation_fill_credit": False,
        "matching_exact_price_prints_only": True,
        "full_fill_required": True,
    }
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("purpose") != "two_sided_structural_fill_support_only"
        or _canonical_value(report, "report_sha256") != REPORT_CANONICAL_SHA256
        or tuple(report.get("symbols", ())) != SYMBOLS
        or report.get("day_utc") != "2023-06-01"
        or report.get("market_source")
        != "official Binance Data Vision USD-M daily archives"
        or report.get("contract") != expected_contract
        or any(
            report.get(name) is not False
            for name in (
                "strategy_outcomes_read",
                "price_returns_read",
                "costs_read",
                "profit_and_loss_read",
                "policy_thresholds_selected",
                "trading_authority",
                "profitability_claim",
                "leverage_applied",
            )
        )
        or contract.get("schema_version") != COST_CONTRACT_SCHEMA
        or _canonical_value(contract, "contract_sha256") != COST_CONTRACT_SHA256
        or contract.get("feature_spec", {}).get("maker_entry_fee_bps") != 2.0
        or contract.get("feature_spec", {}).get("additional_slippage_bps_per_side")
        != 1.0
        or _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{PROBE_PATH}") != PROBE_BLOB_OID
    ):
        raise ValueError("Round 58 source identity or non-authority claims drifted")
    _validate_finite(report)

    results = report.get("symbol_results")
    if (
        not isinstance(results, list)
        or tuple(row.get("symbol") for row in results) != SYMBOLS
    ):
        raise ValueError("Round 58 symbol inventory drifted")
    for row in results:
        symbol = str(row["symbol"])
        source = row["source"]
        manifests = source["manifests"]
        if (
            source["manifest_sha256"] != _canonical_sha256(manifests)
            or len(manifests) != 2
            or {item["data_type"] for item in manifests} != {"bookTicker", "trades"}
        ):
            raise ValueError(f"Round 58 {symbol} source manifest drifted")
        for manifest in manifests:
            if (
                manifest.get("provider") != "binance"
                or manifest.get("market_type") != "futures"
                or manifest.get("symbol") != symbol
                or manifest.get("period") != report["day_utc"]
                or manifest.get("status") != "complete"
                or manifest.get("is_current") is not True
                or manifest.get("checksum_status") != "verified"
                or manifest.get("source_sha256") != manifest.get("expected_sha256")
                or any(
                    int(manifest.get(name, -1)) != 0
                    for name in (
                        "invalid_rows",
                        "duplicate_ids",
                        "update_id_regressions",
                        "event_time_regressions",
                        "out_of_order_rows",
                        "crossed_books",
                    )
                )
            ):
                raise ValueError(f"Round 58 {symbol} archive integrity drifted")

        eligible = int(row["decision_rows_quote_age_eligible"])
        planned = int(row["decision_rows_planned"])
        rejected = int(row["decision_rows_quote_age_rejected"])
        states = row["joint_fill_state"]
        state_rows = {name: int(states[name]["rows"]) for name in STATE_LABELS}
        if (
            planned != eligible + rejected
            or int(states["total_rows"]) != eligible
            or sum(state_rows.values()) != eligible
            or any(
                not math.isclose(
                    float(states[name]["ratio"]),
                    state_rows[name] / eligible,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for name in STATE_LABELS
            )
            or sum(int(value) for value in row["joint_fill_sequencing"].values())
            != state_rows["both"]
            or int(row["bid_fill"]["rows"]) != eligible
            or int(row["ask_fill"]["rows"]) != eligible
            or int(row["bid_fill"]["filled_rows"])
            != state_rows["bid_only"] + state_rows["both"]
            or int(row["ask_fill"]["filled_rows"])
            != state_rows["ask_only"] + state_rows["both"]
            or len(row["fill_bucket_cross_table"]) != 16
            or sum(int(item["rows"]) for item in row["fill_bucket_cross_table"])
            != eligible
        ):
            raise ValueError(f"Round 58 {symbol} joint-fill reconciliation failed")
        result_identity = {
            "bid_fill_result_sha256": row["bid_fill"]["result_sha256"],
            "ask_fill_result_sha256": row["ask_fill"]["result_sha256"],
            "joint_state": states,
            "sequencing": row["joint_fill_sequencing"],
            "placement_spread_bps": row["placement_spread_bps"],
            "both_fill_placement_spread_bps": row["both_fill_placement_spread_bps"],
            "singleton_placement_spread_bps": row["singleton_placement_spread_bps"],
            "exposure_both_ms": row["both_fill_inventory_exposure_ms"],
            "singleton_remaining_ms": row[
                "singleton_maximum_inventory_exposure_before_expiry_ms"
            ],
            "cross_table": row["fill_bucket_cross_table"],
        }
        if _canonical_sha256(result_identity) != row["result_sha256"]:
            raise ValueError(f"Round 58 {symbol} result identity drifted")
    return report, contract


def _cost_reference(contract: Mapping[str, object]) -> tuple[float, float]:
    features = contract["feature_spec"]
    maker_fee = float(features["maker_entry_fee_bps"])
    slippage = float(features["additional_slippage_bps_per_side"])
    return 2.0 * maker_fee, 2.0 * (maker_fee + slippage)


def _joint_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        state = result["joint_fill_state"]
        singleton_rows = int(state["bid_only"]["rows"]) + int(state["ask_only"]["rows"])
        rows.append(
            {
                "round": ROUND,
                "day_utc": report["day_utc"],
                "symbol": result["symbol"],
                "eligible_decisions": state["total_rows"],
                "none_rows": state["none"]["rows"],
                "none_ratio": state["none"]["ratio"],
                "bid_only_rows": state["bid_only"]["rows"],
                "bid_only_ratio": state["bid_only"]["ratio"],
                "ask_only_rows": state["ask_only"]["rows"],
                "ask_only_ratio": state["ask_only"]["ratio"],
                "singleton_rows": singleton_rows,
                "singleton_ratio": singleton_rows / int(state["total_rows"]),
                "both_rows": state["both"]["rows"],
                "both_ratio": state["both"]["ratio"],
            }
        )
    return rows


def _spread_rows(
    report: Mapping[str, object], contract: Mapping[str, object]
) -> list[dict[str, object]]:
    fee_reference, base_reference = _cost_reference(contract)
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        spread = result["both_fill_placement_spread_bps"]
        rows.append(
            {
                "round": ROUND,
                "day_utc": report["day_utc"],
                "symbol": result["symbol"],
                "both_fill_rows": result["joint_fill_state"]["both"]["rows"],
                "spread_p50_bps": spread["p50"],
                "spread_p90_bps": spread["p90"],
                "spread_p99_bps": spread["p99"],
                "spread_max_bps": spread["max"],
                "prior_frozen_fee_reference_bps": fee_reference,
                "prior_frozen_base_cost_reference_bps": base_reference,
                "p99_clears_fee_reference": float(spread["p99"]) > fee_reference,
                "p99_clears_base_cost_reference": float(spread["p99"]) > base_reference,
                "max_clears_base_cost_reference": float(spread["max"]) > base_reference,
            }
        )
    return rows


def _exposure_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        both = result["both_fill_inventory_exposure_ms"]
        singleton = result["singleton_maximum_inventory_exposure_before_expiry_ms"]
        rows.append(
            {
                "round": ROUND,
                "day_utc": report["day_utc"],
                "symbol": result["symbol"],
                "both_fill_p50_ms": both["p50"],
                "both_fill_p90_ms": both["p90"],
                "both_fill_p99_ms": both["p99"],
                "both_fill_max_ms": both["max"],
                "singleton_p50_ms": singleton["p50"],
                "singleton_p90_ms": singleton["p90"],
                "singleton_p99_ms": singleton["p99"],
                "singleton_max_ms": singleton["max"],
                "maker_order_expiry_ms": report["contract"]["maker_order_expiry_ms"],
            }
        )
    return rows


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        for manifest in result["source"]["manifests"]:
            rows.append(
                {
                    "round": ROUND,
                    "symbol": result["symbol"],
                    "day_utc": report["day_utc"],
                    "provider": manifest["provider"],
                    "market_type": manifest["market_type"],
                    "data_type": manifest["data_type"],
                    "rows_read": manifest["rows_read"],
                    "derived_rows": manifest["derived_rows"],
                    "first_exchange_time_ms": manifest["first_exchange_time_ms"],
                    "last_exchange_time_ms": manifest["last_exchange_time_ms"],
                    "checksum_status": manifest["checksum_status"],
                    "source_sha256": manifest["source_sha256"],
                    "url": manifest["url"],
                }
            )
    return rows


def _cross_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "round": ROUND,
            "day_utc": report["day_utc"],
            "symbol": result["symbol"],
            "bid_bucket": item["bid_bucket"],
            "bid_bucket_label": FILL_BUCKETS[int(item["bid_bucket"])],
            "ask_bucket": item["ask_bucket"],
            "ask_bucket_label": FILL_BUCKETS[int(item["ask_bucket"])],
            "rows": item["rows"],
        }
        for result in report["symbol_results"]
        for item in result["fill_bucket_cross_table"]
    ]


def _progress_rows(previous_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    with previous_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    observed = [int(row["round"]) for row in rows]
    if observed == list(range(1, ROUND + 1)):
        rows = rows[:-1]
        observed = observed[:-1]
    if observed != list(range(1, ROUND)):
        raise ValueError("research progress must contain exactly Rounds 1 through 57")
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "value-blind two-sided maker support feasibility",
            "periods": "official Binance USD-M 2023-06-01; one consumed UTC day",
            "selection_contaminated": "True",
            "horizon_seconds": "15",
            "feature_set": (
                "official L1 BBO and exact-price trades; full displayed queue; "
                "10-second decisions"
            ),
            "risk_level": (
                "posthoc structural feasibility only; no returns, P&L, policy, "
                "model, AI, or leverage"
            ),
            "selected_signals": "0",
            "status": "rejected",
            "source_file": (
                "verified Round 58 value-blind support report; symmetric touch "
                "making rejected before training"
            ),
            "best_model_id": "two_sided_touch_maker_structurally_rejected",
            "ensemble_models": "0",
            "development_consumed": "True",
        }
    )
    rows.append(new)
    return rows, fields


def _joint_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Singleton fills dominated two-sided completions",
        subtitle=(
            "1 June 2023 UTC; 750 ms placement, full displayed queue ahead, "
            "15-second expiry; value-blind support diagnostic"
        ),
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    ("No fill", 100.0 * float(row["none_ratio"]), COLORS["muted"]),
                    (
                        "One side only",
                        100.0 * float(row["singleton_ratio"]),
                        COLORS["amber"],
                    ),
                    (
                        "Both sides",
                        100.0 * float(row["both_ratio"]),
                        COLORS["teal"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=75.0,
        y_label="Eligible decisions (%)",
        tick_decimals=0,
        value_decimals=2,
    )


def _spread_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Observed two-fill spreads did not clear the frozen cost reference",
        subtitle=(
            "Value-blind placement spread only; 4 bps is the prior 2 bps/side fee "
            "reference and 6 bps adds the prior 1 bps/side slippage reference"
        ),
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    ("p50 spread", float(row["spread_p50_bps"]), COLORS["teal"]),
                    ("p99 spread", float(row["spread_p99_bps"]), COLORS["blue"]),
                    ("max spread", float(row["spread_max_bps"]), COLORS["cyan"]),
                    (
                        "Fee ref.",
                        float(row["prior_frozen_fee_reference_bps"]),
                        COLORS["amber"],
                    ),
                    (
                        "Base cost ref.",
                        float(row["prior_frozen_base_cost_reference_bps"]),
                        COLORS["red"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=8.0,
        y_label="Basis points",
        tick_decimals=1,
        value_decimals=2,
    )


def _exposure_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="One-sided inventory persisted for most of the quote lifetime",
        subtitle=(
            "Elapsed milliseconds between two fills, or remaining exposure after "
            "a singleton fill until the frozen 15-second expiry"
        ),
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    (
                        "Both p50",
                        float(row["both_fill_p50_ms"]),
                        COLORS["teal"],
                    ),
                    (
                        "Both p99",
                        float(row["both_fill_p99_ms"]),
                        COLORS["blue"],
                    ),
                    (
                        "Singleton p50",
                        float(row["singleton_p50_ms"]),
                        COLORS["amber"],
                    ),
                    (
                        "Singleton p99",
                        float(row["singleton_p99_ms"]),
                        COLORS["red"],
                    ),
                    (
                        "Quote expiry",
                        float(row["maker_order_expiry_ms"]),
                        COLORS["muted"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=16500.0,
        y_label="Milliseconds",
        tick_decimals=0,
        value_decimals=0,
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points: list[tuple[float, float]] = []
    for row in rows:
        raw = str(row.get("spearman_ic", "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            points.append((float(row["round"]), 100.0 * value))
    labels = {value: str(int(value)) for value in (points[0][0], points[-1][0])}
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        if points[0][0] <= value <= points[-1][0]:
            labels[value] = str(int(value))
    return _line_svg(
        title="Optimization research record through Round 58",
        subtitle=(
            "Recorded rank statistics use differing targets; Round 58 generated "
            "no comparable forecast statistic and is represented only in progress.csv"
        ),
        series=(("Recorded Spearman", points, COLORS["teal"]),),
        x_labels=labels,
        y_label="Recorded Spearman x 100",
    )


def _clean_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    parent = (ROOT / "docs" / "model-research" / "action-value").resolve()
    if not resolved.is_relative_to(parent) or resolved.name != "latest":
        raise ValueError("publication output must be action-value/latest")
    if resolved.exists():
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _failure_analysis(
    report: Mapping[str, object],
    joint_rows: Sequence[Mapping[str, object]],
    spread_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    analysis: dict[str, object] = {
        "schema_version": "round-058-two-sided-maker-feasibility-analysis-v1",
        "round": ROUND,
        "source_report_sha256": REPORT_CANONICAL_SHA256,
        "status": "rejected_posthoc_feasibility",
        "selection_contaminated": True,
        "facts": {
            "observed_utc_days": 1,
            "eligible_decisions": sum(
                int(row["eligible_decisions"]) for row in joint_rows
            ),
            "both_fill_rows": sum(int(row["both_rows"]) for row in joint_rows),
            "singleton_rows": sum(int(row["singleton_rows"]) for row in joint_rows),
            "symbols_with_p99_spread_above_prior_fee_reference": sum(
                bool(row["p99_clears_fee_reference"]) for row in spread_rows
            ),
            "symbols_with_p99_spread_above_prior_base_cost_reference": sum(
                bool(row["p99_clears_base_cost_reference"]) for row in spread_rows
            ),
            "returns_read": False,
            "costs_read_by_probe": False,
            "pnl_read": False,
            "model_trained": False,
            "policy_selected": False,
            "ai_evaluated": False,
        },
        "decision": [
            "Do not train or promote a symmetric best-bid/best-ask touch-making policy under the prior frozen retail cost reference.",
            "All three symbols had a 99th-percentile two-fill placement spread below the prior 4 bps round-trip fee reference.",
            "One-sided fills were materially more common than two-sided completions, leaving unresolved inventory exposure.",
        ],
        "retained_infrastructure": [
            "Checksum-bound official event ingestion and the queue-censored fill simulator remain useful execution infrastructure.",
            "Any future passive strategy needs a separately frozen economic mechanism and untouched evidence.",
        ],
        "prohibited_inferences": [
            "profitability",
            "AI uplift",
            "testnet readiness",
            "live trading readiness",
            "leverage readiness",
            "performance across other dates or regimes",
        ],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(
    report: Mapping[str, object],
    joint_rows: Sequence[Mapping[str, object]],
    spread_rows: Sequence[Mapping[str, object]],
) -> str:
    joint_table = "\n".join(
        "| {symbol} | {rows:,} | {both:.2f}% | {single:.2f}% | {none:.2f}% |".format(
            symbol=row["symbol"],
            rows=int(row["eligible_decisions"]),
            both=100.0 * float(row["both_ratio"]),
            single=100.0 * float(row["singleton_ratio"]),
            none=100.0 * float(row["none_ratio"]),
        )
        for row in joint_rows
    )
    spread_table = "\n".join(
        "| {symbol} | {p50:.4f} | {p90:.4f} | {p99:.4f} | {maximum:.4f} |".format(
            symbol=row["symbol"],
            p50=float(row["spread_p50_bps"]),
            p90=float(row["spread_p90_bps"]),
            p99=float(row["spread_p99_bps"]),
            maximum=float(row["spread_max_bps"]),
        )
        for row in spread_rows
    )
    return f"""# Round 58: Two-Sided Maker Feasibility

> **Rejected post-hoc structural diagnostic.** This is consumed development evidence, not a pre-registered profitability test. It grants no trading, testnet, live, leverage, AI-uplift, or performance authority.

Round 58 asked a narrow question before spending more GPU time: can simultaneous best-bid and best-ask orders complete often enough, and capture enough observed spread, to justify training a symmetric touch-making model? The value-blind probe used checksum-verified official Binance USD-M BTCUSDT, ETHUSDT, and SOLUSDT events from **{report["day_utc"]} UTC**. It read no returns, costs, P&L, strategy outcomes, or policy thresholds.

The answer is no under the prior frozen research cost reference. Two-sided fills occurred in only 2.36-3.18% of eligible decisions, while one-sided fills occurred in 28.03-47.19%. Every symbol's 99th-percentile two-fill placement spread was below 1 bps. The earlier frozen contract models 2 bps maker fee per side, or 4 bps round trip before its additional 1 bps per-side slippage reference. Account-specific production fees must still be queried from the venue; these values are a pinned research comparison, not a universal Binance fee claim.

| Joint fill support | Eligible decisions | Both sides | One side only | No fill |
|---|---:|---:|---:|---:|
{joint_table}

| Two-fill placement spread (bps) | p50 | p90 | p99 | Maximum |
|---|---:|---:|---:|---:|
{spread_table}

No model was trained, no trades were replayed, and ROI, drawdown, profit factor, leverage, and AI uplift were not computed. This early rejection is intentional: a model cannot manufacture gross spread that is absent from the observed mechanism. The next candidate must use a structurally different source of edge and a newly frozen design.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Joint fill support | [SVG](charts/joint-fill-support.svg) | [CSV](joint-fill-support.csv) |
| Spread versus prior cost reference | [SVG](charts/spread-feasibility.svg) | [CSV](spread-feasibility.csv) |
| Inventory exposure duration | [SVG](charts/inventory-exposure.svg) | [CSV](inventory-exposure.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `fill-bucket-cross.csv`, `failure-analysis.json`, and the exact `screen.json` preserve the underlying evidence. Every chart is regenerated from a tracked CSV.

## Research basis

- [Binance official public market-data archives](https://data.binance.vision/)
- [Avellaneda and Stoikov: High-frequency trading in a limit order book](https://doi.org/10.1080/14697680701381228)
- [Huang, Lehalle, and Rosenbaum: the queue-reactive model](https://arxiv.org/abs/1312.0563)
- [The Market Maker's Dilemma: fill probability and post-fill returns](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5074873)
"""


def publish(
    *,
    report_path: Path,
    cost_contract_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, contract = _validate_source(report_path, cost_contract_path)
    joint_rows = _joint_rows(report)
    spread_rows = _spread_rows(report, contract)
    exposure_rows = _exposure_rows(report)
    source_rows = _source_rows(report)
    cross_rows = _cross_rows(report)
    progress_rows, progress_fields = _progress_rows(previous_progress_path)
    failure = _failure_analysis(report, joint_rows, spread_rows)

    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "joint-fill-support.csv", joint_rows)
    _write_csv(output_dir / "spread-feasibility.csv", spread_rows)
    _write_csv(output_dir / "inventory-exposure.csv", exposure_rows)
    _write_csv(output_dir / "source-coverage.csv", source_rows)
    _write_csv(output_dir / "fill-bucket-cross.csv", cross_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "joint-fill-support.svg", _joint_svg(joint_rows))
    _write_text(charts / "spread-feasibility.svg", _spread_svg(spread_rows))
    _write_text(charts / "inventory-exposure.svg", _exposure_svg(exposure_rows))
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))
    _write_text(output_dir / "README.md", _readme(report, joint_rows, spread_rows))
    write_json_atomic(output_dir / "failure-analysis.json", failure, indent=2)
    shutil.copyfile(report_path, output_dir / "screen.json")

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    fee_reference, base_reference = _cost_reference(contract)
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "publisher_path": "tools/publish_round58_two_sided_maker_feasibility.py",
        "source": {
            "report_file": report_path.name,
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "cost_reference_path": str(cost_contract_path.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "cost_reference_sha256": COST_CONTRACT_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "probe_path": PROBE_PATH,
            "probe_git_blob_oid": PROBE_BLOB_OID,
        },
        "claims": {
            "status": "rejected_posthoc_feasibility",
            "selection_contaminated": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
            "performance_claim": False,
        },
        "result": {
            "observed_utc_days": 1,
            "eligible_decisions": sum(
                int(row["eligible_decisions"]) for row in joint_rows
            ),
            "both_fill_rows": sum(int(row["both_rows"]) for row in joint_rows),
            "singleton_rows": sum(int(row["singleton_rows"]) for row in joint_rows),
            "symbols_with_p99_spread_above_prior_fee_reference": sum(
                bool(row["p99_clears_fee_reference"]) for row in spread_rows
            ),
            "symbols_with_p99_spread_above_prior_base_cost_reference": sum(
                bool(row["p99_clears_base_cost_reference"]) for row in spread_rows
            ),
            "prior_frozen_fee_reference_bps": fee_reference,
            "prior_frozen_base_cost_reference_bps": base_reference,
            "model_trained": False,
            "policy_selected": False,
            "economic_replay_performed": False,
            "ai_ablation_performed": False,
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2)
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\evidence\round58-two-sided-support-20260715.json"
        ),
    )
    parser.add_argument(
        "--cost-contract",
        type=Path,
        default=research / "round-057-queue-censored-make-take-execution-contract.json",
    )
    parser.add_argument(
        "--progress", type=Path, default=research / "latest" / "progress.csv"
    )
    parser.add_argument("--output", type=Path, default=research / "latest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        report_path=arguments.report.resolve(),
        cost_contract_path=arguments.cost_contract.resolve(),
        previous_progress_path=arguments.progress.resolve(),
        output_dir=arguments.output.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": publication["round"],
                "status": publication["claims"]["status"],
                "publication_canonical_sha256": publication[
                    "publication_canonical_sha256"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
