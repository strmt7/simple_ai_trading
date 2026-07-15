"""Publish hash-verified Round 61 carry replay evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
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
    _svg_start,
    _validate_finite,
    _write_csv,
    _write_text,
)
from tools.publish_round59_funding_persistence_feasibility import (  # noqa: E402
    _clean_output,
)


ROUND = 61
REPORT_SCHEMA = "round-061-carry-economic-replay-report-v1"
PUBLICATION_SCHEMA = "round-061-carry-economic-replay-publication-v1"
DESIGN_SCHEMA = "round-061-carry-economic-replay-design-v4"
CERTIFICATE_SCHEMA = "round-061-carry-event-source-certificate-v1"
MANIFEST_SCHEMA = "round-061-carry-event-manifest-v2"
REPORT_FILE_SHA256 = "dc9fa604257db59a1b2d1766c70fa8131aafc5cb132238a2f536f13ad2b08908"
REPORT_CANONICAL_SHA256 = (
    "e2f6275232b7f6b7b511211b26a697536a401e14a118393502bcfda96ae4d6e4"
)
DESIGN_FILE_SHA256 = "f04422d726045170a7e1221e4fa06a3ed83d094c5bf3a629edda10e6c66d0d6c"
DESIGN_SHA256 = "eadab11fea709aa562f11557e08887a155a8d446cefa09c77153c2b1967159cc"
CERTIFICATE_FILE_SHA256 = (
    "a419767b6e04ab6b97aa26f9b526a3c5ac80e37303237af3cd39348f78ca912a"
)
CERTIFICATE_CANONICAL_SHA256 = (
    "579c27e3575b46f07231bf510787195048eeedf2b4ca00413b55271ad14a2d30"
)
MANIFEST_FILE_SHA256 = (
    "65a5c20b2ad8a85add95d49f5ea94260d36062c8fed004dfd5ee7e310814700f"
)
MANIFEST_SHA256 = "8b5a8037176c5e37af2c261c0ab79dd9f43f6e0d9024e78f6306694293126594"
IMPLEMENTATION_COMMIT = "7d8d16c7194b241c10ef98ef9be94f9b16f9bd9b"
RUNNER_PATH = "tools/run_round61_carry_economic_replay.py"
RUNNER_BLOB_OID = "fb4399d96806befdecf83845c02c120d40fd9cd2"
SOURCE_IMPLEMENTATION_COMMIT = "3f7d1f4dfe9de4c7af4236720a5cf671963bfc37"
SOURCE_LOADER_PATH = "tools/ingest_round61_carry_event_sources.py"
SOURCE_LOADER_BLOB_OID = "05d6bada41acb28b628cab0bf883e2c7d529f433"
DESIGN_PATH = (
    "docs/model-research/action-value/round-061-carry-economic-replay-design.json"
)
DESIGN_BLOB_OID = "85d61e6c6ec3e8c1ce0aa42a99205389c126185f"
MANIFEST_PATH = "docs/model-research/action-value/round-061-carry-event-manifest.json"
MANIFEST_BLOB_OID = "a5ef08b300b0509b1eeddd09d453b895185ef0a2"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


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
    *,
    report_path: Path,
    design_path: Path,
    certificate_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    expected_files = (
        (report_path, REPORT_FILE_SHA256, "Round 61 report"),
        (design_path, DESIGN_FILE_SHA256, "Round 61 design"),
        (certificate_path, CERTIFICATE_FILE_SHA256, "Round 61 certificate"),
        (manifest_path, MANIFEST_FILE_SHA256, "Round 61 manifest"),
    )
    for path, expected, label in expected_files:
        if _file_sha256(path) != expected:
            raise ValueError(f"{label} file hash drifted")
    report = _read_object(report_path, "Round 61 report")
    design = _read_object(design_path, "Round 61 design")
    certificate = _read_object(certificate_path, "Round 61 source certificate")
    manifest = _read_object(manifest_path, "Round 61 event manifest")
    bound_blobs = (
        (IMPLEMENTATION_COMMIT, RUNNER_PATH, RUNNER_BLOB_OID),
        (SOURCE_IMPLEMENTATION_COMMIT, SOURCE_LOADER_PATH, SOURCE_LOADER_BLOB_OID),
        (SOURCE_IMPLEMENTATION_COMMIT, DESIGN_PATH, DESIGN_BLOB_OID),
        ("80d96fa507b69f38e1869a08a1db6c9f93ca1534", MANIFEST_PATH, MANIFEST_BLOB_OID),
    )
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected_elevated_funding_carry"
        or _canonical_value(report, "report_sha256") != REPORT_CANONICAL_SHA256
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("manifest_sha256") != MANIFEST_SHA256
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or report.get("tick_execution_replay_authorized") is not False
        or report.get("synthetic_or_filled_source_rows") is not False
        or any(
            report.get(name) is not False
            for name in (
                "model_training_authorized",
                "ai_evaluation_authorized",
                "trading_authority",
                "testnet_or_live_authority",
                "profitability_claim",
                "leverage_applied",
            )
        )
        or report.get("result", {}).get("all_symbols_passed") is not False
        or report.get("result", {}).get("passed_symbols") != []
        or design.get("schema_version") != DESIGN_SCHEMA
        or _canonical_value(design, "design_sha256") != DESIGN_SHA256
        or certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("complete") is not True
        or _canonical_value(certificate, "source_certificate_sha256")
        != CERTIFICATE_CANONICAL_SHA256
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or _canonical_value(manifest, "manifest_sha256") != MANIFEST_SHA256
        or any(
            _git("rev-parse", f"{commit}:{path}") != expected
            or _git("rev-parse", f"HEAD:{path}") != expected
            for commit, path, expected in bound_blobs
        )
    ):
        raise ValueError("Round 61 publication source identity drifted")
    results = report.get("symbol_results")
    expected_counts = {
        "BTCUSDT": (72, 72, 30),
        "ETHUSDT": (76, 76, 20),
        "SOLUSDT": (62, 61, 0),
    }
    if (
        not isinstance(results, list)
        or tuple(row.get("symbol") for row in results) != SYMBOLS
    ):
        raise ValueError("Round 61 symbol inventory drifted")
    for result in results:
        canonical = dict(result)
        claimed = str(canonical.pop("result_sha256", ""))
        symbol = str(result["symbol"])
        summary = result["summary"]
        observed = (
            int(summary["manifest_episodes"]),
            int(summary["source_eligible_episodes"]),
            int(summary["capacity_eligible_episodes"]),
        )
        if (
            claimed != _canonical_sha256(canonical)
            or observed != expected_counts[symbol]
            or result.get("gate", {}).get("passed") is not False
        ):
            raise ValueError(f"Round 61 {symbol} result identity drifted")
    _validate_finite(report)
    _validate_finite(certificate)
    return report


def _utc(timestamp_ms: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _summary_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        summary = result["summary"]
        metrics = result["metrics"]
        rows.append(
            {
                "round": ROUND,
                "symbol": result["symbol"],
                **summary,
                "mean_stress_net_committed_capital_bps": metrics[
                    "mean_stress_net_committed_capital_bps"
                ],
                "median_stress_net_committed_capital_bps": metrics[
                    "median_stress_net_committed_capital_bps"
                ],
                "positive_stress_net_fraction": metrics["positive_stress_net_fraction"],
                "bootstrap_lower_95_mean_stress_net_committed_capital_bps": metrics[
                    "bootstrap_lower_95_mean_stress_net_committed_capital_bps"
                ],
                "maximum_sequential_drawdown_committed_capital_bps": metrics[
                    "maximum_sequential_drawdown_committed_capital_bps"
                ],
                "worst_episode_committed_capital_bps": metrics[
                    "worst_episode_committed_capital_bps"
                ],
                "expected_shortfall_10pct_committed_capital_bps": metrics[
                    "expected_shortfall_10pct_committed_capital_bps"
                ],
                "distinct_calendar_years": metrics["distinct_calendar_years"],
                "positive_calendar_year_fraction": metrics[
                    "positive_calendar_year_fraction"
                ],
                "gate_passed": result["gate"]["passed"],
            }
        )
    return rows


def _episode_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        for episode in result["episodes"]:
            rows.append(
                {
                    "round": ROUND,
                    "symbol": result["symbol"],
                    "episode_id": episode["episode_id"],
                    "decision_time_utc": _utc(int(episode["decision_time_ms"])),
                    "end_time_utc": _utc(int(episode["end_time_ms"])),
                    "current_funding_bps": episode["current_funding_bps"],
                    "future_funding_settlements": episode["future_funding_settlements"],
                    "source_eligible": episode["source_eligible"],
                    "source_ineligible_reasons": ";".join(
                        episode["source_ineligible_reasons"]
                    ),
                    "capacity_eligible": episode["capacity_eligible"],
                    "economically_scored": episode["economically_scored"],
                    "spot_pnl_usdt": episode.get("spot_pnl_usdt"),
                    "perpetual_pnl_usdt": episode.get("perpetual_pnl_usdt"),
                    "basis_pnl_usdt": episode.get("basis_pnl_usdt"),
                    "short_funding_pnl_usdt": episode.get("short_funding_pnl_usdt"),
                    "gross_pnl_usdt": episode.get("gross_pnl_usdt"),
                    "exchange_taker_fees_usdt": episode.get("exchange_taker_fees_usdt"),
                    "additional_operational_slippage_usdt": episode.get(
                        "additional_operational_slippage_usdt"
                    ),
                    "stress_net_pnl_usdt": episode.get("stress_net_pnl_usdt"),
                    "stress_net_committed_capital_bps": episode.get(
                        "stress_net_committed_capital_bps"
                    ),
                }
            )
    return rows


def _capacity_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        for episode in result["episodes"]:
            for fill in episode.get("fill_capacity", []):
                rows.append(
                    {
                        "round": ROUND,
                        "symbol": result["symbol"],
                        "episode_id": episode["episode_id"],
                        "decision_time_utc": _utc(int(episode["decision_time_ms"])),
                        **fill,
                    }
                )
    return rows


def _yearly_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {"round": ROUND, "symbol": result["symbol"], **row}
        for result in report["symbol_results"]
        for row in result["metrics"]["yearly_results"]
    ]


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {"round": ROUND, "symbol": result["symbol"], **row}
        for result in report["symbol_results"]
        for row in result["gate"]["checks"]
    ]


def _cumulative_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in report["symbol_results"]:
        cumulative = 0.0
        for episode in result["episodes"]:
            if not episode["economically_scored"]:
                continue
            cumulative += float(episode["stress_net_committed_capital_bps"])
            rows.append(
                {
                    "round": ROUND,
                    "symbol": result["symbol"],
                    "episode_id": episode["episode_id"],
                    "decision_time_ms": episode["decision_time_ms"],
                    "decision_time_utc": _utc(int(episode["decision_time_ms"])),
                    "episode_stress_net_committed_capital_bps": episode[
                        "stress_net_committed_capital_bps"
                    ],
                    "cumulative_stress_net_committed_capital_bps": cumulative,
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
        raise ValueError("research progress must contain exactly Rounds 1 through 60")
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "matched spot-perpetual minute-stress economic replay",
            "periods": "event-specific 2020-2025; non-overlapping seven-day episodes",
            "selection_contaminated": "True",
            "horizon_seconds": "604800",
            "feature_set": "settled funding, synchronized spot/perpetual 1m execution, mark prices, same-side taker flow",
            "risk_level": "unlevered; adverse minute bounds; actual-notional fees; 1% taker-flow participation cap",
            "selected_signals": "0",
            "after_cost_diagnostic_rows": "50",
            "status": "rejected",
            "source_file": "verified Round 61 report; capacity and after-cost economic gates failed",
            "best_model_id": "elevated_funding_carry_rejected",
            "ensemble_models": "0",
            "development_consumed": "True",
        }
    )
    rows.append(new)
    return rows, fields


def _eligibility_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Round 61 source and executable-capacity support",
        subtitle="Capacity denominator is source-eligible episodes; every fill must stay within 1% of same-side one-minute taker flow",
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    (
                        "Source eligible",
                        100.0 * float(row["source_eligible_fraction"]),
                        COLORS["teal"],
                    ),
                    (
                        "Capacity eligible",
                        100.0 * float(row["capacity_eligible_fraction"]),
                        COLORS["red"],
                    ),
                    ("Required", 90.0, COLORS["amber"]),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=100.0,
        y_label="Eligible episodes (%)",
        tick_decimals=0,
        value_decimals=2,
    )


def _economics_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="After-cost stress economics failed on the admitted subset",
        subtitle="Committed-capital bps per seven-day episode; SOL has no capacity-admitted episodes and is intentionally blank",
        groups=tuple(
            (
                str(row["symbol"])[:3],
                (
                    (
                        "Mean",
                        row["mean_stress_net_committed_capital_bps"],
                        COLORS["teal"],
                    ),
                    (
                        "Median",
                        row["median_stress_net_committed_capital_bps"],
                        COLORS["blue"],
                    ),
                    (
                        "Bootstrap lower 95% mean",
                        row["bootstrap_lower_95_mean_stress_net_committed_capital_bps"],
                        COLORS["red"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=-15.0,
        y_max=5.0,
        y_label="Stress-net committed-capital return (bps)",
        tick_decimals=0,
        value_decimals=2,
    )


def _decomposition_svg(report: Mapping[str, object]) -> str:
    groups = []
    for result in report["symbol_results"]:
        metrics = result["metrics"]
        count = int(metrics["episodes"])
        if count == 0:
            continue
        groups.append(
            (
                str(result["symbol"])[:3],
                (
                    ("Basis P&L", metrics["mean_basis_pnl_usdt"], COLORS["blue"]),
                    (
                        "Funding P&L",
                        metrics["mean_short_funding_pnl_usdt"],
                        COLORS["teal"],
                    ),
                    (
                        "Exchange fees",
                        -float(metrics["mean_exchange_taker_fees_usdt"]),
                        COLORS["amber"],
                    ),
                    (
                        "Operational slippage",
                        -float(metrics["mean_additional_operational_slippage_usdt"]),
                        COLORS["red"],
                    ),
                    (
                        "Stress net",
                        float(metrics["total_stress_net_pnl_usdt"]) / count,
                        COLORS["cyan"],
                    ),
                ),
            )
        )
    return _bar_svg(
        title="Mean P&L decomposition of capacity-admitted episodes",
        subtitle="USDT per 20,000 USDT committed-capital episode; positive funding did not reliably overcome basis movement and costs",
        groups=tuple(groups),
        y_min=-80.0,
        y_max=120.0,
        y_label="Mean USDT per episode",
        tick_decimals=0,
        value_decimals=2,
    )


def _cumulative_svg(rows: Sequence[Mapping[str, object]]) -> str:
    by_symbol = {
        symbol: [row for row in rows if row["symbol"] == symbol] for symbol in SYMBOLS
    }
    series = tuple(
        (
            symbol[:3],
            tuple(
                (
                    float(row["decision_time_ms"]),
                    float(row["cumulative_stress_net_committed_capital_bps"]),
                )
                for row in by_symbol[symbol]
            ),
            color,
        )
        for symbol, color in (
            ("BTCUSDT", COLORS["teal"]),
            ("ETHUSDT", COLORS["blue"]),
        )
        if by_symbol[symbol]
    )
    all_times = [point[0] for _name, points, _color in series for point in points]
    start, end = min(all_times), max(all_times)
    start_year = datetime.fromtimestamp(start / 1000, tz=UTC).year
    end_year = datetime.fromtimestamp(end / 1000, tz=UTC).year
    labels = {
        timestamp: str(year)
        for year in range(start_year, end_year + 1)
        if start
        <= (timestamp := datetime(year, 1, 1, tzinfo=UTC).timestamp() * 1000)
        <= end
    }
    return _line_svg(
        title="Sequential stress-net path on capacity-admitted episodes",
        subtitle="Consumed event subset only; gaps are inactive periods, not zero-return holdings; SOL has no admitted path",
        series=series,
        x_labels=labels,
        y_label="Cumulative committed-capital bps",
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points = [
        (float(row["round"]), 100.0 * float(row["spearman_ic"]))
        for row in rows
        if row.get("spearman_ic") not in ("", None)
    ]
    width, height = 1200, 700
    left, right, top, bottom = 105, 55, 135, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    y_values = [point[1] for point in points] + [0.0]
    y_min, y_max = min(y_values), max(y_values)
    padding = max(1.0, (y_max - y_min) * 0.12)
    y_min -= padding
    y_max += padding

    def x(value: float) -> float:
        return left + (value - 1.0) / 60.0 * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    lines = _svg_start(
        "Action-value research progression",
        "Recorded directional rank statistics only; Rounds 58-61 add no forecast statistic",
        width=width,
        height=height,
    )
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        yy = y(value)
        lines.extend(
            (
                f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{COLORS["grid"]}"/>',
                f'<text x="{left - 14}" y="{yy + 5:.1f}" text-anchor="end" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.1f}</text>',
            )
        )
    path = " ".join(
        ("M" if index == 0 else "L") + f" {x(px):.1f} {y(py):.1f}"
        for index, (px, py) in enumerate(points)
    )
    lines.append(
        f'<path d="{path}" fill="none" stroke="{COLORS["teal"]}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for px, py in points:
        lines.append(
            f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="4" fill="{COLORS["teal"]}"/>'
        )
    for round_number in (1, 10, 20, 30, 40, 50, 61):
        lines.append(
            f'<text x="{x(round_number):.1f}" y="{height - bottom + 34}" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{round_number}</text>'
        )
    lines.extend(
        (
            f'<line x1="{x(61):.1f}" y1="{top}" x2="{x(61):.1f}" y2="{height - bottom}" stroke="{COLORS["amber"]}" stroke-width="2" stroke-dasharray="6 6"/>',
            f'<text x="{x(61) - 8:.1f}" y="{top + 20}" text-anchor="end" fill="{COLORS["amber"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">Latest evidence</text>',
            f'<rect x="{left}" y="{height - 43}" width="16" height="4" fill="{COLORS["teal"]}"/>',
            f'<text x="{left + 24}" y="{height - 35}" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">Recorded Spearman</text>',
            f'<text transform="translate(25 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Recorded Spearman x 100</text>',
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def _decision_analysis(
    report: Mapping[str, object],
    summaries: Sequence[Mapping[str, object]],
    capacity_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    fill_support = {
        symbol: {
            fill: sum(
                row["symbol"] == symbol and row["fill"] == fill and bool(row["passed"])
                for row in capacity_rows
            )
            for fill in (
                "spot_entry",
                "spot_exit",
                "perpetual_entry",
                "perpetual_exit",
            )
        }
        for symbol in SYMBOLS
    }
    analysis: dict[str, object] = {
        "schema_version": "round-061-carry-economic-replay-decision-v1",
        "round": ROUND,
        "status": report["status"],
        "decision": "reject elevated-funding seven-day carry family",
        "source_and_capacity": list(summaries),
        "fill_pass_counts": fill_support,
        "economic_findings": {
            result["symbol"]: {
                "economically_scored_episodes": result["metrics"]["episodes"],
                "mean_stress_net_committed_capital_bps": result["metrics"][
                    "mean_stress_net_committed_capital_bps"
                ],
                "median_stress_net_committed_capital_bps": result["metrics"][
                    "median_stress_net_committed_capital_bps"
                ],
                "bootstrap_lower_95_mean_stress_net_committed_capital_bps": result[
                    "metrics"
                ]["bootstrap_lower_95_mean_stress_net_committed_capital_bps"],
            }
            for result in report["symbol_results"]
        },
        "authorized_next_step": None,
        "prohibited": [
            "relaxing the frozen capacity or risk thresholds after observing results",
            "tick-level confirmation for this rejected family",
            "model or AI training for this rejected family",
            "profitability, ROI, testnet, live, leverage, or trading-readiness claims",
        ],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _display(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    numeric = float(value) * (100.0 if percent else 1.0)
    suffix = "%" if percent else ""
    return f"{numeric:+.2f}{suffix}"


def _readme(rows: Sequence[Mapping[str, object]]) -> str:
    table = "\n".join(
        "| {symbol} | {source}/{total} ({source_pct:.2f}%) | {capacity} ({capacity_pct:.2f}%) | {mean} | {median} | {positive} | {lower} | {drawdown} | Rejected |".format(
            symbol=row["symbol"],
            source=int(row["source_eligible_episodes"]),
            total=int(row["manifest_episodes"]),
            source_pct=100.0 * float(row["source_eligible_fraction"]),
            capacity=int(row["capacity_eligible_episodes"]),
            capacity_pct=100.0 * float(row["capacity_eligible_fraction"]),
            mean=_display(row["mean_stress_net_committed_capital_bps"]),
            median=_display(row["median_stress_net_committed_capital_bps"]),
            positive=_display(row["positive_stress_net_fraction"], percent=True),
            lower=_display(
                row["bootstrap_lower_95_mean_stress_net_committed_capital_bps"]
            ),
            drawdown=_display(row["maximum_sequential_drawdown_committed_capital_bps"]),
        )
        for row in rows
    )
    return f"""# Round 61: Matched Spot-Perpetual Economic Replay

> **Rejected. No profitability or trading claim.** The elevated-funding seven-day carry family failed executable-capacity and after-cost economic gates. Tick replay, model training, AI evaluation, leverage, testnet, and live trading remain unauthorized.

Round 61 matched a long spot leg with a 1x short USD-M perpetual leg at the same base quantity. It used official checksum-verified Binance minute bars and settled funding, adverse minute high/low execution bounds, actual fill notionals, four taker fees, one extra basis point per fill, and a maximum 1% share of same-side one-minute taker flow. No missing price was interpolated or filled.

| Symbol | Source eligible | Capacity eligible | Mean net bps | Median net bps | Positive | Lower 95% mean | Max drawdown bps | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{table}

Capacity is the first decisive failure: only 30 BTC, 20 ETH, and zero SOL episodes could support every modeled fill. On those admitted BTC/ETH subsets, median stress-net returns and bootstrap lower means were still negative. Positive mean values were not sufficient to pass the precommitted distribution, tail, year-stability, concentration, and breadth gates.

## Evidence

| View | Graph | Tracked source |
|---|---|---|
| Source and executable-capacity support | [SVG](charts/source-capacity-eligibility.svg) | [CSV](summary.csv) |
| After-cost stress economics | [SVG](charts/stress-net-economics.svg) | [CSV](summary.csv) |
| Mean P&L decomposition | [SVG](charts/pnl-decomposition.svg) | [CSV](episodes.csv) |
| Sequential event-time path | [SVG](charts/cumulative-stress-net.svg) | [CSV](cumulative.csv) |
| Research progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

capacity.csv, yearly.csv, gates.csv, decision-analysis.json, design.json, event-manifest.json, source-certificate.json, and the exact screen.json preserve the remaining source-bound evidence. Every graph is regenerated from tracked numeric data.

## Limits

- Minute extremes are conservative bounds, not historical order-book fills.
- Same-side taker flow is a capacity proxy, not displayed depth.
- The event set is consumed development evidence.
- This seven-day carry screen is separate from the platform's intraday directional day-trading objective.
"""


def publish(
    *,
    report_path: Path,
    design_path: Path,
    certificate_path: Path,
    manifest_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report = _validate_sources(
        report_path=report_path,
        design_path=design_path,
        certificate_path=certificate_path,
        manifest_path=manifest_path,
    )
    summaries = _summary_rows(report)
    episodes = _episode_rows(report)
    capacity = _capacity_rows(report)
    yearly = _yearly_rows(report)
    gates = _gate_rows(report)
    cumulative = _cumulative_rows(report)
    progress, progress_fields = _progress_rows(previous_progress_path)
    decision = _decision_analysis(report, summaries, capacity)
    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(output_dir / "episodes.csv", episodes)
    _write_csv(output_dir / "capacity.csv", capacity)
    _write_csv(output_dir / "yearly.csv", yearly)
    _write_csv(output_dir / "gates.csv", gates)
    _write_csv(output_dir / "cumulative.csv", cumulative)
    _write_csv(
        output_dir / "progress.csv",
        [{field: row.get(field, "") for field in progress_fields} for row in progress],
    )
    _write_text(charts / "source-capacity-eligibility.svg", _eligibility_svg(summaries))
    _write_text(charts / "stress-net-economics.svg", _economics_svg(summaries))
    _write_text(charts / "pnl-decomposition.svg", _decomposition_svg(report))
    _write_text(charts / "cumulative-stress-net.svg", _cumulative_svg(cumulative))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    _write_text(output_dir / "README.md", _readme(summaries))
    write_json_atomic(output_dir / "decision-analysis.json", decision, indent=2)
    shutil.copyfile(report_path, output_dir / "screen.json")
    shutil.copyfile(design_path, output_dir / "design.json")
    shutil.copyfile(manifest_path, output_dir / "event-manifest.json")
    shutil.copyfile(certificate_path, output_dir / "source-certificate.json")
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "publisher_path": "tools/publish_round61_carry_economic_replay.py",
        "source": {
            "report_file": report_path.name,
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "design_path": DESIGN_PATH,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "design_sha256": DESIGN_SHA256,
            "manifest_path": MANIFEST_PATH,
            "manifest_file_sha256": MANIFEST_FILE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "source_certificate_file_sha256": CERTIFICATE_FILE_SHA256,
            "source_certificate_canonical_sha256": CERTIFICATE_CANONICAL_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "runner_path": RUNNER_PATH,
            "runner_git_blob_oid": RUNNER_BLOB_OID,
            "source_loader_path": SOURCE_LOADER_PATH,
            "source_loader_git_blob_oid": SOURCE_LOADER_BLOB_OID,
        },
        "claims": {
            "status": report["status"],
            "selection_contaminated": True,
            "tick_execution_replay_authorized": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "model_training_authorized": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=ROOT / DESIGN_PATH)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST_PATH)
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
        manifest_path=arguments.manifest.resolve(),
        previous_progress_path=arguments.progress.resolve(),
        output_dir=arguments.output.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": publication["round"],
                "publication_canonical_sha256": publication[
                    "publication_canonical_sha256"
                ],
                "artifacts": len(publication["artifacts"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
