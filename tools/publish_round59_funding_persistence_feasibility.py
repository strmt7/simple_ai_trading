"""Publish hash-verified Round 59 funding-persistence feasibility evidence."""

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


ROUND = 59
REPORT_SCHEMA = "round-059-funding-persistence-feasibility-report-v1"
PUBLICATION_SCHEMA = "round-059-funding-persistence-feasibility-publication-v1"
DESIGN_SCHEMA = "round-059-funding-persistence-feasibility-design-v1"
REPORT_CANONICAL_SHA256 = (
    "268e0a4734ae10ad2f413ca77e75de3cdc55ae98a0ae07bd5c3a944499be03d0"
)
REPORT_FILE_SHA256 = "f99843b7998a9bc473f7c8d8c80c52a8e718e7729e5d9896263bdfe01538d14e"
DESIGN_SHA256 = "cf5d17873d9b269adc3aebcb9f1237d8854dbc80b9ef098a8d8e6e1f5d1d6f95"
SOURCE_CERTIFICATE_FILE_SHA256 = (
    "e2fe434d7c290f09913160506c52fce30849a6bd319465390c4b4d22dad482a7"
)
SOURCE_CERTIFICATE_CANONICAL_SHA256 = (
    "8bf4c9404edbdb80285bbd472a856430873c77de97b5c7fbf24f6c8f86eaab39"
)
IMPLEMENTATION_COMMIT = "c263fbd525f85a6cef86c1e8930d09844927a3d6"
RUNNER_PATH = "tools/run_round59_funding_persistence_feasibility.py"
RUNNER_BLOB_OID = "78a8dde241e6c6568de75e3d46a6f99cceff4263"
DESIGN_PATH = (
    "docs/model-research/action-value/"
    "round-059-funding-persistence-feasibility-design.json"
)
DESIGN_BLOB_OID = "7f6702a168a6b928e342a7c934fbdc5ae45f5eac"
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
    *, report_path: Path, design_path: Path, certificate_path: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if _file_sha256(report_path) != REPORT_FILE_SHA256:
        raise ValueError("Round 59 report file hash drifted")
    if _file_sha256(certificate_path) != SOURCE_CERTIFICATE_FILE_SHA256:
        raise ValueError("Round 59 source certificate file hash drifted")
    report = _read_object(report_path, "Round 59 report")
    design = _read_object(design_path, "Round 59 design")
    certificate = _read_object(certificate_path, "Round 38 source certificate")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected_funding_persistence_support"
        or _canonical_value(report, "report_sha256") != REPORT_CANONICAL_SHA256
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or tuple(report.get("source", {}).get("symbols", ())) != SYMBOLS
        or report.get("spot_history_ingestion_authorized") is not False
        or report.get("selection_contaminated") is not True
        or report.get("result", {}).get("symbol_cells") != 27
        or report.get("result", {}).get("breadth_cells") != 9
        or report.get("result", {}).get("passing_breadth_cells") != 0
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
        or _canonical_value(certificate, "source_certificate_sha256")
        != SOURCE_CERTIFICATE_CANONICAL_SHA256
        or _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{RUNNER_PATH}")
        != RUNNER_BLOB_OID
        or _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{DESIGN_PATH}")
        != DESIGN_BLOB_OID
    ):
        raise ValueError("Round 59 source identity or non-authority claims drifted")
    _validate_finite(report)

    results = report.get("symbol_results")
    if (
        not isinstance(results, list)
        or tuple(row.get("symbol") for row in results) != SYMBOLS
    ):
        raise ValueError("Round 59 symbol inventory drifted")
    observed_cells: list[tuple[str, str, int]] = []
    for result in results:
        canonical = dict(result)
        claimed = str(canonical.pop("result_sha256", ""))
        if _canonical_sha256(canonical) != claimed:
            raise ValueError(f"Round 59 {result['symbol']} result identity drifted")
        for cell in result["cells"]:
            observed_cells.append(
                (
                    str(result["symbol"]),
                    str(cell["trigger_id"]),
                    int(cell["horizon_hours"]),
                )
            )
            if cell.get("symbol_gate_passed") is not False:
                raise ValueError("Round 59 unexpectedly contains a passing symbol cell")
    expected_cells = [
        (symbol, trigger, horizon)
        for symbol in SYMBOLS
        for trigger in TRIGGERS
        for horizon in HORIZONS
    ]
    if observed_cells != expected_cells or any(
        gate.get("passed") is not False for gate in report["breadth_gates"]
    ):
        raise ValueError("Round 59 cell or breadth-gate inventory drifted")
    return report, design, certificate


def _cell_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        for cell in result["cells"]:
            row: dict[str, object] = {
                "round": ROUND,
                "symbol": result["symbol"],
                "trigger_id": cell["trigger_id"],
                "trigger_operator": cell["trigger_operator"],
                "trigger_value_bps": cell["trigger_value_bps"],
                "horizon_hours": cell["horizon_hours"],
                "episodes": cell["episodes"],
                "first_decision_time_ms": cell["first_decision_time_ms"],
                "last_decision_time_ms": cell["last_decision_time_ms"],
                "mean_future_settlements": cell["mean_future_settlements"],
                "mean_gross_funding_bps": cell["mean_gross_funding_bps"],
                "median_gross_funding_bps": cell["median_gross_funding_bps"],
                "p10_gross_funding_bps": cell["p10_gross_funding_bps"],
                "p90_gross_funding_bps": cell["p90_gross_funding_bps"],
                "bootstrap_lower_95_mean_gross_bps": cell[
                    "bootstrap_lower_95_mean_gross_bps"
                ],
                "bootstrap_upper_95_mean_gross_bps": cell[
                    "bootstrap_upper_95_mean_gross_bps"
                ],
                "symbol_gate_passed": cell["symbol_gate_passed"],
            }
            for reference, values in cell["cost_comparisons"].items():
                for name, value in values.items():
                    row[f"{reference}_{name}"] = value
            rows.append(row)
    return rows


def _transition_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "round": ROUND,
            "symbol": result["symbol"],
            **result["sign_transition"],
        }
        for result in report["symbol_results"]
    ]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"round": ROUND, **result["source"]} for result in report["symbol_results"]]


def _breadth_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"round": ROUND, **row} for row in report["breadth_gates"]]


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
        raise ValueError("research progress must contain exactly Rounds 1 through 58")
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "consumed funding-persistence structural feasibility",
            "periods": "2021-12-01..2025-06-30; previously consumed funding archive",
            "selection_contaminated": "True",
            "horizon_seconds": "604800",
            "feature_set": (
                "settled funding only; causal positive/1bp/2bp triggers; "
                "non-overlapping 1/3/7-day windows"
            ),
            "risk_level": (
                "structural data-build gate; no prices, P&L, model, AI, or leverage"
            ),
            "selected_signals": "0",
            "status": "rejected",
            "source_file": (
                "verified Round 59 report; no same-trigger/horizon cell passed "
                "BTC/ETH/SOL breadth under the 32 bps reference"
            ),
            "best_model_id": "funding_persistence_no_spot_ingestion",
            "ensemble_models": "0",
            "development_consumed": "True",
        }
    )
    rows.append(new)
    return rows, fields


def _seven_day_carry_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if int(row["horizon_hours"]) == 168]
    by_key = {(row["symbol"], row["trigger_id"]): row for row in selected}
    return _bar_svg(
        title="Seven-day funding carry required rare elevated settlements",
        subtitle=(
            "December 2021-June 2025; non-overlapping outcomes after a causally "
            "observed settlement; cost bars are references, not realized P&L"
        ),
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
        y_max=55.0,
        y_label="Basis points",
        tick_decimals=0,
        value_decimals=2,
    )


def _high_trigger_support_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row
        for row in rows
        if row["trigger_id"] == "at_least_2bp" and int(row["horizon_hours"]) == 168
    ]
    return _bar_svg(
        title="Elevated-funding support remained statistically insufficient",
        subtitle=(
            "Seven-day >=2 bps trigger; only 20 BTC, 20 ETH, and 25 SOL "
            "non-overlapping episodes versus 40 required per symbol"
        ),
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    (
                        "Mean after stress ref.",
                        float(row["stress_four_leg_mean_net_reference_bps"]),
                        COLORS["teal"],
                    ),
                    (
                        "Median after stress ref.",
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
                        COLORS["red"],
                    ),
                ),
            )
            for row in selected
        ),
        y_min=-5.0,
        y_max=20.0,
        y_label="Funding carry minus 32 bps reference",
        tick_decimals=0,
        value_decimals=2,
    )


def _persistence_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Positive funding was persistent but usually too small",
        subtitle=(
            "December 2021-June 2025 settled funding transitions; the screen "
            "does not assume a fixed settlement interval"
        ),
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
        title="Optimization research record through Round 59",
        subtitle=(
            "Recorded rank statistics use differing targets; Round 59 generated "
            "no forecast statistic and is represented only in progress.csv"
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
    report: Mapping[str, object], cells: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    high = [
        row
        for row in cells
        if row["trigger_id"] == "at_least_2bp" and int(row["horizon_hours"]) == 168
    ]
    ordinary = [
        row
        for row in cells
        if row["trigger_id"] == "positive" and int(row["horizon_hours"]) == 168
    ]
    analysis: dict[str, object] = {
        "schema_version": "round-059-funding-persistence-failure-analysis-v1",
        "round": ROUND,
        "source_report_sha256": REPORT_CANONICAL_SHA256,
        "status": "rejected_funding_persistence_support",
        "selection_contaminated": True,
        "facts": {
            "symbol_cells": 27,
            "passing_symbol_cells": 0,
            "breadth_cells": 9,
            "passing_breadth_cells": 0,
            "ordinary_positive_trigger_7d_mean_gross_bps": {
                row["symbol"]: row["mean_gross_funding_bps"] for row in ordinary
            },
            "elevated_trigger_7d_episode_counts": {
                row["symbol"]: row["episodes"] for row in high
            },
            "elevated_trigger_7d_lower_95_stress_reference_bps": {
                row["symbol"]: row[
                    "stress_four_leg_bootstrap_lower_95_mean_net_reference_bps"
                ]
                for row in high
            },
            "minimum_required_episodes_per_symbol": 40,
            "spot_history_ingestion_authorized": False,
            "price_or_basis_rows_read": False,
            "model_or_ai_evaluated": False,
        },
        "decision": [
            "Do not ingest the multi-year spot corpus or train a basis/funding model under this evidence.",
            "Ordinary positive funding did not amortize the frozen four-leg cost references over seven days.",
            "The >=2 bps trigger had positive mean seven-day support after the stress reference, but only 20/20/25 episodes and nonpositive lower confidence support for BTC and SOL.",
        ],
        "retained_hypothesis": [
            "Elevated funding may warrant a separately frozen full-available-history funding-only replication with the same thresholds and gates.",
            "No gate may be relaxed using these consumed outcomes.",
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
    ordinary = {
        row["symbol"]: row
        for row in cells
        if row["trigger_id"] == "positive" and int(row["horizon_hours"]) == 168
    }
    transition_by_symbol = {row["symbol"]: row for row in transitions}
    table = "\n".join(
        "| {symbol} | {persist:.2f}% | {ordinary:+.2f} | {episodes} | {high_mean:+.2f} | {high_median:+.2f} | {lower:+.2f} |".format(
            symbol=symbol,
            persist=100.0
            * float(transition_by_symbol[symbol]["next_positive_given_positive"]),
            ordinary=float(ordinary[symbol]["mean_gross_funding_bps"]),
            episodes=int(high[symbol]["episodes"]),
            high_mean=float(high[symbol]["stress_four_leg_mean_net_reference_bps"]),
            high_median=float(high[symbol]["stress_four_leg_median_net_reference_bps"]),
            lower=float(
                high[symbol][
                    "stress_four_leg_bootstrap_lower_95_mean_net_reference_bps"
                ]
            ),
        )
        for symbol in SYMBOLS
    )
    return f"""# Round 59: Funding-Persistence Feasibility

> **Rejected consumed structural evidence.** No profitability, basis-neutrality, AI-uplift, leverage, testnet, live-trading, or execution claim is made.

Round 59 tested whether causally observed positive BTCUSDT, ETHUSDT, and SOLUSDT funding persisted strongly enough to justify downloading the missing synchronized spot history. The runner reconstructed and re-hashed every monthly funding row against 129 checksum-certified Binance archive streams from **December 2021 through June 2025**. It read no price, premium-index, spot, basis, P&L, model, or AI rows.

Positive funding usually remained positive at the next settlement, but ordinary positive-funding episodes did not clear four-leg costs. The rare `>=2` bps trigger produced positive mean seven-day carry after the 32 bps stress reference, yet only 20 BTC, 20 ETH, and 25 SOL non-overlapping episodes existed versus 40 required. BTC and SOL lower 95% mean bounds remained below zero. All 27 symbol cells and all nine BTC/ETH/SOL breadth cells failed, so spot-history ingestion is not authorized.

| Seven-day evidence | P(next positive \| positive) | Ordinary trigger gross mean (bps) | `>=2` bps episodes | `>=2` mean after 32 bps | Median after 32 bps | Lower 95% mean after 32 bps |
|---|---:|---:|---:|---:|---:|---:|
{table}

The 4/28/32 bps values are pinned research references, not account-specific realized fees. The 28 bps reference is two spot fills at 10 bps plus two futures fills at 4 bps; the 32 bps stress reference adds 1 bps to each of four fills. A production path must query both signed commission endpoints and still model synchronized spreads, depth, legging latency, basis change, margin, and liquidation.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Seven-day gross carry versus references | [SVG](charts/seven-day-gross-carry.svg) | [CSV](funding-cells.csv) |
| Elevated-funding confidence and support | [SVG](charts/elevated-funding-support.svg) | [CSV](funding-cells.csv) |
| Funding-sign persistence | [SVG](charts/funding-sign-persistence.svg) | [CSV](sign-persistence.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `breadth-gates.csv`, `source-certificate.json`, `failure-analysis.json`, and the exact `screen.json` preserve the remaining source-bound evidence. Every chart is regenerated from a tracked CSV.

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
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, _design, _certificate = _validate_sources(
        report_path=report_path,
        design_path=design_path,
        certificate_path=certificate_path,
    )
    cells = _cell_rows(report)
    transitions = _transition_rows(report)
    sources = _source_rows(report)
    breadth = _breadth_rows(report)
    progress, progress_fields = _progress_rows(previous_progress_path)
    failure = _failure_analysis(report, cells)

    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "funding-cells.csv", cells)
    _write_csv(output_dir / "sign-persistence.csv", transitions)
    _write_csv(output_dir / "source-coverage.csv", sources)
    _write_csv(output_dir / "breadth-gates.csv", breadth)
    _write_csv(
        output_dir / "progress.csv",
        [{field: row.get(field, "") for field in progress_fields} for row in progress],
    )
    _write_text(charts / "seven-day-gross-carry.svg", _seven_day_carry_svg(cells))
    _write_text(
        charts / "elevated-funding-support.svg", _high_trigger_support_svg(cells)
    )
    _write_text(charts / "funding-sign-persistence.svg", _persistence_svg(transitions))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    _write_text(output_dir / "README.md", _readme(report, cells, transitions))
    write_json_atomic(output_dir / "failure-analysis.json", failure, indent=2)
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
        "publisher_path": "tools/publish_round59_funding_persistence_feasibility.py",
        "source": {
            "report_file": report_path.name,
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "design_path": DESIGN_PATH,
            "design_sha256": DESIGN_SHA256,
            "source_certificate_file_sha256": SOURCE_CERTIFICATE_FILE_SHA256,
            "source_certificate_canonical_sha256": SOURCE_CERTIFICATE_CANONICAL_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "runner_path": RUNNER_PATH,
            "runner_git_blob_oid": RUNNER_BLOB_OID,
        },
        "claims": {
            "status": "rejected_funding_persistence_support",
            "selection_contaminated": True,
            "spot_history_ingestion_authorized": False,
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
            r"E:\SimpleAITradingData\evidence\round59-funding-persistence-20260715-v1.json"
        ),
    )
    parser.add_argument("--design", type=Path, default=ROOT / DESIGN_PATH)
    parser.add_argument(
        "--certificate",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round38-derivatives-source-20260712-v2\certificate.json"
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
