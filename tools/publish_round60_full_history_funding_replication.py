"""Publish hash-verified Round 60 funding replication evidence."""

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
from tools.publish_round59_funding_persistence_feasibility import (  # noqa: E402
    _breadth_rows as _round59_breadth_rows,
    _cell_rows as _round59_cell_rows,
    _clean_output,
    _source_rows as _round59_source_rows,
    _transition_rows as _round59_transition_rows,
)


ROUND = 60
REPORT_SCHEMA = "round-060-full-history-funding-replication-report-v1"
PUBLICATION_SCHEMA = "round-060-full-history-funding-replication-publication-v1"
DESIGN_SCHEMA = "round-060-full-history-funding-replication-design-v1"
CERTIFICATE_SCHEMA = "round-060-full-history-funding-source-certificate-v1"
REPORT_CANONICAL_SHA256 = (
    "a020bd2f26280b82705ffa4bda83b37d439dfa09377eda64ffdfcbf17c9e9ba4"
)
REPORT_FILE_SHA256 = "0bf9dc26b6bc53a9bfebedba9f6ae43cca879f14ced85bc7b4b64de70f388b5d"
DESIGN_SHA256 = "965609f682595efd07dad719c958fe2086bbd29c5e3c6b2a98ba8b37261806c2"
DESIGN_FILE_SHA256 = "47b651d98c83f78215ca2e8a080120d9dec9b331b5d1894c296cb968a5d6302f"
CERTIFICATE_CANONICAL_SHA256 = (
    "e3fe53ba87d728eed85efdaa93350b3c76ab7adcb7216cac690c5029818a9736"
)
CERTIFICATE_FILE_SHA256 = (
    "45262aea2ca244c9b1323370c220cbf7ddc8c2e4956f33eac4278ed5b2d6b373"
)
PRIOR_REPORT_CANONICAL_SHA256 = (
    "268e0a4734ae10ad2f413ca77e75de3cdc55ae98a0ae07bd5c3a944499be03d0"
)
PRIOR_REPORT_FILE_SHA256 = (
    "f99843b7998a9bc473f7c8d8c80c52a8e718e7729e5d9896263bdfe01538d14e"
)
IMPLEMENTATION_COMMIT = "35e10f45a4b4525bd9be3ad4746d43cc8f61de60"
RUNNER_PATH = "tools/run_round60_full_history_funding_replication.py"
RUNNER_BLOB_OID = "c8a7dbafadb22a467b323a2238b6c7cf5366e12b"
INGEST_PATH = "tools/ingest_round60_full_funding_archives.py"
INGEST_BLOB_OID = "5c8128b6a10a385d6b0c2e98f8d28b61c8705f99"
DESIGN_PATH = "docs/model-research/action-value/round-060-full-history-funding-replication-design.json"
DESIGN_BLOB_OID = "3c6e6a94f42c31377fcd1eaf1b2ba0181247b5b1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
TRIGGERS = ("positive", "at_least_1bp", "at_least_2bp")
HORIZONS = (24, 72, 168)


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


def _validate_sources(
    *, report_path: Path, design_path: Path, certificate_path: Path, prior_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    expected_files = (
        (report_path, REPORT_FILE_SHA256, "Round 60 report"),
        (design_path, DESIGN_FILE_SHA256, "Round 60 design"),
        (certificate_path, CERTIFICATE_FILE_SHA256, "Round 60 certificate"),
        (prior_path, PRIOR_REPORT_FILE_SHA256, "Round 59 report"),
    )
    for path, expected, label in expected_files:
        if _file_sha256(path) != expected:
            raise ValueError(f"{label} file hash drifted")
    report = _read_object(report_path, "Round 60 report")
    design = _read_object(design_path, "Round 60 design")
    certificate = _read_object(certificate_path, "Round 60 source certificate")
    prior = _read_object(prior_path, "Round 59 report")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status")
        != "full_history_support_passed_spot_ingestion_authorized"
        or _canonical_value(report, "report_sha256") != REPORT_CANONICAL_SHA256
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or report.get("result", {}).get("symbol_cells") != 27
        or report.get("result", {}).get("breadth_cells") != 9
        or report.get("result", {}).get("passing_breadth_cells") != 1
        or report.get("spot_history_ingestion_authorized") is not True
        or report.get("selection_contaminated") is not True
        or any(
            report.get(name) is not False
            for name in (
                "price_rows_read",
                "premium_index_rows_read",
                "spot_rows_read",
                "model_trained",
                "ai_evaluated",
                "profitability_claim",
                "trading_authority",
                "testnet_or_live_authority",
                "leverage_applied",
            )
        )
        or design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or _canonical_value(design, "design_sha256") != DESIGN_SHA256
        or certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or _canonical_value(certificate, "source_certificate_sha256")
        != CERTIFICATE_CANONICAL_SHA256
        or _canonical_value(prior, "report_sha256") != PRIOR_REPORT_CANONICAL_SHA256
        or prior.get("status") != "rejected_funding_persistence_support"
        or _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{RUNNER_PATH}")
        != RUNNER_BLOB_OID
        or _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{INGEST_PATH}")
        != INGEST_BLOB_OID
        or _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{DESIGN_PATH}")
        != DESIGN_BLOB_OID
    ):
        raise ValueError("Round 60 publication source identity drifted")
    _validate_finite(report)
    _validate_finite(certificate)

    results = report.get("symbol_results")
    if (
        not isinstance(results, list)
        or tuple(row.get("symbol") for row in results) != SYMBOLS
    ):
        raise ValueError("Round 60 symbol inventory drifted")
    observed: list[tuple[str, str, int]] = []
    passing: list[tuple[str, str, int]] = []
    for result in results:
        canonical = dict(result)
        claimed = str(canonical.pop("result_sha256", ""))
        if _canonical_sha256(canonical) != claimed:
            raise ValueError(f"Round 60 {result['symbol']} result identity drifted")
        for cell in result["cells"]:
            key = (
                str(result["symbol"]),
                str(cell["trigger_id"]),
                int(cell["horizon_hours"]),
            )
            observed.append(key)
            if cell.get("symbol_gate_passed") is True:
                passing.append(key)
    expected = [
        (symbol, trigger, horizon)
        for symbol in SYMBOLS
        for trigger in TRIGGERS
        for horizon in HORIZONS
    ]
    expected_passing = [(symbol, "at_least_2bp", 168) for symbol in SYMBOLS]
    passing_breadth = [row for row in report["breadth_gates"] if row.get("passed")]
    if (
        observed != expected
        or passing != expected_passing
        or len(passing_breadth) != 1
        or passing_breadth[0].get("trigger_id") != "at_least_2bp"
        or passing_breadth[0].get("horizon_hours") != 168
    ):
        raise ValueError("Round 60 passing-cell inventory drifted")
    return report, prior


def _cell_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = _round59_cell_rows(report)
    for row in rows:
        row["round"] = ROUND
    return rows


def _transition_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = _round59_transition_rows(report)
    for row in rows:
        row["round"] = ROUND
    return rows


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = _round59_source_rows(report)
    for row in rows:
        row["round"] = ROUND
    return rows


def _breadth_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = _round59_breadth_rows(report)
    for row in rows:
        row["round"] = ROUND
    return rows


def _high_cells(report: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {
        result["symbol"]: next(
            cell
            for cell in result["cells"]
            if cell["trigger_id"] == "at_least_2bp"
            and int(cell["horizon_hours"]) == 168
        )
        for result in report["symbol_results"]
    }


def _comparison_rows(
    report: Mapping[str, object], prior: Mapping[str, object]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for round_number, period, source in (
        (59, "2021-12..2025-06", prior),
        (60, "2020-01/09..2026-06", report),
    ):
        for symbol, cell in _high_cells(source).items():
            stress = cell["cost_comparisons"]["stress_four_leg"]
            rows.append(
                {
                    "round": round_number,
                    "period": period,
                    "symbol": symbol,
                    "trigger_id": cell["trigger_id"],
                    "horizon_hours": cell["horizon_hours"],
                    "episodes": cell["episodes"],
                    "mean_gross_funding_bps": cell["mean_gross_funding_bps"],
                    "mean_after_stress_reference_bps": stress["mean_net_reference_bps"],
                    "median_after_stress_reference_bps": stress[
                        "median_net_reference_bps"
                    ],
                    "positive_after_stress_reference_fraction": stress[
                        "positive_net_reference_fraction"
                    ],
                    "bootstrap_lower_95_mean_after_stress_reference_bps": stress[
                        "bootstrap_lower_95_mean_net_reference_bps"
                    ],
                    "bootstrap_upper_95_mean_after_stress_reference_bps": stress[
                        "bootstrap_upper_95_mean_net_reference_bps"
                    ],
                    "symbol_gate_passed": cell["symbol_gate_passed"],
                }
            )
    return rows


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
        raise ValueError("research progress must contain exactly Rounds 1 through 59")
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "unchanged full-history funding-persistence replication",
            "periods": "BTC/ETH 2020-01..2026-06; SOL 2020-09..2026-06",
            "selection_contaminated": "True",
            "horizon_seconds": "604800",
            "feature_set": "settled funding only; Round 59 triggers, windows, costs, uncertainty, and gates unchanged",
            "risk_level": "structural data-build gate; no prices, P&L, model, AI, or leverage",
            "selected_signals": "1",
            "status": "passed_structural_gate",
            "source_file": "verified Round 60 report; >=2 bps/7-day cell passed BTC/ETH/SOL breadth under the 32 bps reference",
            "best_model_id": "funding_persistence_spot_replay_authorized",
            "ensemble_models": "0",
            "development_consumed": "True",
        }
    )
    rows.append(new)
    return rows, fields


def _gross_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if int(row["horizon_hours"]) == 168]
    by_key = {(row["symbol"], row["trigger_id"]): row for row in selected}
    return _bar_svg(
        title="Full-history seven-day funding-persistence screen",
        subtitle="BTC/ETH Jan 2020-Jun 2026; SOL Sep 2020-Jun 2026; gross funding references are not realized P&L",
        groups=tuple(
            (
                symbol[:3],
                (
                    (
                        "Positive trigger",
                        float(by_key[(symbol, "positive")]["mean_gross_funding_bps"]),
                        COLORS["teal"],
                    ),
                    (
                        ">=1 bps trigger",
                        float(
                            by_key[(symbol, "at_least_1bp")]["mean_gross_funding_bps"]
                        ),
                        COLORS["blue"],
                    ),
                    (
                        ">=2 bps trigger",
                        float(
                            by_key[(symbol, "at_least_2bp")]["mean_gross_funding_bps"]
                        ),
                        COLORS["cyan"],
                    ),
                    ("Four-leg taker ref.", 28.0, COLORS["amber"]),
                    ("Stress ref.", 32.0, COLORS["red"]),
                ),
            )
            for symbol in SYMBOLS
        ),
        y_min=-10.0,
        y_max=100.0,
        y_label="Basis points",
        tick_decimals=0,
        value_decimals=2,
    )


def _support_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row
        for row in rows
        if row["trigger_id"] == "at_least_2bp" and int(row["horizon_hours"]) == 168
    ]
    return _bar_svg(
        title="The precommitted elevated-funding cell passed all symbol gates",
        subtitle="Non-overlapping seven-day episodes; values subtract the 32 bps stress reference but exclude basis and execution dynamics",
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    (
                        "Mean after reference",
                        float(row["stress_four_leg_mean_net_reference_bps"]),
                        COLORS["teal"],
                    ),
                    (
                        "Median after reference",
                        float(row["stress_four_leg_median_net_reference_bps"]),
                        COLORS["blue"],
                    ),
                    (
                        "Lower 95% mean",
                        float(
                            row[
                                "stress_four_leg_bootstrap_lower_95_mean_net_reference_bps"
                            ]
                        ),
                        COLORS["cyan"],
                    ),
                ),
            )
            for row in selected
        ),
        y_min=0.0,
        y_max=60.0,
        y_label="Funding carry minus 32 bps reference",
        tick_decimals=0,
        value_decimals=2,
    )


def _comparison_svg(rows: Sequence[Mapping[str, object]]) -> str:
    by_key = {(int(row["round"]), row["symbol"]): row for row in rows}
    return _bar_svg(
        title="Full-history replication increased elevated-funding support",
        subtitle="Same >=2 bps trigger and seven-day non-overlapping episode rule; Round 59 thresholds were not changed",
        groups=tuple(
            (
                symbol[:3],
                (
                    (
                        "Round 59 episodes",
                        float(by_key[(59, symbol)]["episodes"]),
                        COLORS["amber"],
                    ),
                    (
                        "Round 60 episodes",
                        float(by_key[(60, symbol)]["episodes"]),
                        COLORS["teal"],
                    ),
                ),
            )
            for symbol in SYMBOLS
        ),
        y_min=0.0,
        y_max=80.0,
        y_label="Non-overlapping episodes",
        tick_decimals=0,
        value_decimals=0,
    )


def _persistence_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Funding-sign persistence across the complete frozen range",
        subtitle="Observed settled-funding transitions; stored 1-8 hour intervals are read rather than assumed",
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    (
                        "Next positive after positive",
                        100.0 * float(row["next_positive_given_positive"]),
                        COLORS["teal"],
                    ),
                    (
                        "Next positive after nonpositive",
                        100.0 * float(row["next_positive_given_nonpositive"]),
                        COLORS["blue"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=100.0,
        y_label="Conditional probability (%)",
        tick_decimals=0,
        value_decimals=2,
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
        title="Optimization research record through Round 60",
        subtitle="Rank statistics use differing targets; structural Rounds 58-60 add no forecast statistic and remain in progress.csv",
        series=(("Recorded Spearman", points, COLORS["teal"]),),
        x_labels=labels,
        y_label="Recorded Spearman x 100",
    )


def _decision_analysis(
    report: Mapping[str, object], comparison: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    current = [row for row in comparison if int(row["round"]) == ROUND]
    analysis: dict[str, object] = {
        "schema_version": "round-060-full-history-funding-replication-decision-v1",
        "round": ROUND,
        "source_report_sha256": REPORT_CANONICAL_SHA256,
        "status": report["status"],
        "selection_contaminated": True,
        "facts": {
            "verified_monthly_archives": 226,
            "funding_rows": {
                row["symbol"]: row["source"]["rows"] for row in report["symbol_results"]
            },
            "passing_breadth_cell": {
                "trigger_id": "at_least_2bp",
                "horizon_hours": 168,
            },
            "episode_counts": {row["symbol"]: row["episodes"] for row in current},
            "lower_95_mean_after_stress_reference_bps": {
                row["symbol"]: row["bootstrap_lower_95_mean_after_stress_reference_bps"]
                for row in current
            },
            "spot_history_ingestion_authorized": True,
            "price_or_basis_rows_read": False,
            "model_or_ai_evaluated": False,
        },
        "decision": [
            "Proceed only to a separately frozen synchronized spot-perpetual economic replay.",
            "Require causal two-leg prices, spreads, depth, legging latency, basis change, fees, margin, and liquidation controls before any return claim.",
            "Do not train a model or AI system until the deterministic economic replay clears its own precommitted gate.",
        ],
        "prohibited_inferences": [
            "profitability",
            "basis neutrality",
            "AI uplift",
            "testnet readiness",
            "live trading readiness",
            "leverage readiness",
        ],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(
    report: Mapping[str, object],
    cells: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]],
) -> str:
    high = {
        row["symbol"]: row
        for row in cells
        if row["trigger_id"] == "at_least_2bp" and int(row["horizon_hours"]) == 168
    }
    transition_by_symbol = {row["symbol"]: row for row in transitions}
    table = "\n".join(
        "| {symbol} | {rows:,} | {persist:.2f}% | {episodes} | {mean:+.2f} | {median:+.2f} | {positive:.2f}% | {lower:+.2f} |".format(
            symbol=symbol,
            rows=next(
                row["source"]["rows"]
                for row in report["symbol_results"]
                if row["symbol"] == symbol
            ),
            persist=100.0
            * float(transition_by_symbol[symbol]["next_positive_given_positive"]),
            episodes=int(high[symbol]["episodes"]),
            mean=float(high[symbol]["stress_four_leg_mean_net_reference_bps"]),
            median=float(high[symbol]["stress_four_leg_median_net_reference_bps"]),
            positive=100.0
            * float(high[symbol]["stress_four_leg_positive_net_reference_fraction"]),
            lower=float(
                high[symbol][
                    "stress_four_leg_bootstrap_lower_95_mean_net_reference_bps"
                ]
            ),
        )
        for symbol in SYMBOLS
    )
    return f"""# Round 60: Full-History Funding Replication

> **Structural gate passed; no profitability claim.** This consumed funding-only study authorizes one separately frozen spot-perpetual replay. It does not authorize a model, AI, leverage, testnet, or live trading.

Round 60 kept every Round 59 trigger, horizon, cost reference, bootstrap seed, and breadth threshold unchanged, then expanded to every frozen complete official monthly funding archive: **January 2020-June 2026** for BTC and ETH and **September 2020-June 2026** for SOL. All 226 archives passed Binance SHA-256 sidecar checks, and every database month was re-hashed against its certified row stream.

Exactly one of nine breadth cells passed: a causally observed settled funding rate of at least `2` bps followed by a non-overlapping seven-day window. All three symbols exceeded the precommitted 40-episode, 55% positive, positive-median, and positive lower-95%-mean gates after the 32 bps research reference.

| Seven-day `>=2` bps evidence | Funding rows | P(next positive \| positive) | Episodes | Mean after 32 bps | Median after 32 bps | Positive after 32 bps | Lower 95% mean after 32 bps |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

These are funding-carry references, not trade returns. The replay still has to price both legs and model spot-perpetual basis change, synchronized spread and depth, queueing, market impact, legging latency, account-specific commissions, margin, liquidation, outages, and unwind behavior. No price, basis, P&L, model, or AI row was read in this round.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Seven-day gross funding versus references | [SVG](charts/seven-day-gross-carry.svg) | [CSV](funding-cells.csv) |
| Passing cell confidence and support | [SVG](charts/elevated-funding-support.svg) | [CSV](funding-cells.csv) |
| Round 59-to-60 sample comparison | [SVG](charts/replication-comparison.svg) | [CSV](round59-comparison.csv) |
| Funding-sign persistence | [SVG](charts/funding-sign-persistence.svg) | [CSV](sign-persistence.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `breadth-gates.csv`, `source-certificate.json`, `decision-analysis.json`, and the exact `screen.json` preserve the remaining source-bound evidence. Every graph is regenerated from a tracked table.

## Research basis

- [Binance USD-M funding history and interval metadata](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-history)
- [Binance spot account commissions](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/account#query-commission-rates-user_data)
- [Binance USD-M user commissions](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account#user-commission-rate)
- [Fundamentals of Perpetual Futures](https://arxiv.org/abs/2212.06888)
"""


def publish(
    *,
    report_path: Path,
    design_path: Path,
    certificate_path: Path,
    prior_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, prior = _validate_sources(
        report_path=report_path,
        design_path=design_path,
        certificate_path=certificate_path,
        prior_path=prior_path,
    )
    cells = _cell_rows(report)
    transitions = _transition_rows(report)
    sources = _source_rows(report)
    breadth = _breadth_rows(report)
    comparison = _comparison_rows(report, prior)
    progress, progress_fields = _progress_rows(previous_progress_path)
    decision = _decision_analysis(report, comparison)

    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "funding-cells.csv", cells)
    _write_csv(output_dir / "sign-persistence.csv", transitions)
    _write_csv(output_dir / "source-coverage.csv", sources)
    _write_csv(output_dir / "breadth-gates.csv", breadth)
    _write_csv(output_dir / "round59-comparison.csv", comparison)
    _write_csv(
        output_dir / "progress.csv",
        [{field: row.get(field, "") for field in progress_fields} for row in progress],
    )
    _write_text(charts / "seven-day-gross-carry.svg", _gross_svg(cells))
    _write_text(charts / "elevated-funding-support.svg", _support_svg(cells))
    _write_text(charts / "replication-comparison.svg", _comparison_svg(comparison))
    _write_text(charts / "funding-sign-persistence.svg", _persistence_svg(transitions))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    _write_text(output_dir / "README.md", _readme(report, cells, transitions))
    write_json_atomic(output_dir / "decision-analysis.json", decision, indent=2)
    shutil.copyfile(report_path, output_dir / "screen.json")
    shutil.copyfile(certificate_path, output_dir / "source-certificate.json")

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "publisher_path": "tools/publish_round60_full_history_funding_replication.py",
        "source": {
            "report_file": report_path.name,
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "design_path": DESIGN_PATH,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "design_sha256": DESIGN_SHA256,
            "source_certificate_file_sha256": CERTIFICATE_FILE_SHA256,
            "source_certificate_canonical_sha256": CERTIFICATE_CANONICAL_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "runner_path": RUNNER_PATH,
            "runner_git_blob_oid": RUNNER_BLOB_OID,
            "ingest_path": INGEST_PATH,
            "ingest_git_blob_oid": INGEST_BLOB_OID,
        },
        "claims": {
            "status": report["status"],
            "selection_contaminated": True,
            "spot_history_ingestion_authorized": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
            "performance_claim": False,
        },
        "result": report["result"],
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
            r"E:\SimpleAITradingData\evidence\round60-full-history-funding-replication-20260715-v1.json"
        ),
    )
    parser.add_argument("--design", type=Path, default=ROOT / DESIGN_PATH)
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round60-full-funding-source-20260715-v1\certificate.json"
        ),
    )
    parser.add_argument(
        "--prior-report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\evidence\round59-funding-persistence-20260715-v1.json"
        ),
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
        design_path=arguments.design.resolve(),
        certificate_path=arguments.certificate.resolve(),
        prior_path=arguments.prior_report.resolve(),
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
