"""Deterministic publication for the Round 8 repricing mechanism ceiling."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from .polymarket_repricing import (
    POLYMARKET_REPRICING_CONTRACT_SHA256,
    POLYMARKET_REPRICING_REPORT_SCHEMA_VERSION,
)


_PUBLICATION_SCHEMA = "polymarket-repricing-publication-v1"
_ASSETS = ("BTC", "ETH", "SOL")
_LATENCIES_MS = (100, 250, 500, 1_000)
_HOLDING_PERIODS_MS = (250, 500, 1_000, 2_000, 5_000)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _rows(value: object, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return tuple(value)


def _decimal(value: object, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite")
    return parsed


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not readable JSON") from exc
    return dict(_mapping(payload, name))


def _validated_report(path: Path) -> dict[str, Any]:
    report = _load_json(path, "repricing report")
    report_sha256 = str(report.get("report_sha256", ""))
    identity = dict(report)
    identity.pop("report_sha256", None)
    market_counts = _mapping(report.get("market_counts"), "market counts")
    opportunities = _rows(report.get("opportunities"), "opportunities")
    cells = _rows(report.get("cells"), "cells")
    config = _mapping(report.get("config"), "repricing config")
    latencies = tuple(config.get("per_leg_submission_latencies_ms", ()))
    holding_periods = tuple(config.get("holding_periods_ms", ()))
    expected_market_counts = {asset: int(market_counts.get(asset, -1)) for asset in _ASSETS}
    expected_opportunities = sum(expected_market_counts.values()) * 2 * len(latencies) * len(holding_periods)
    reason_keys: set[str] | None = None
    for row in opportunities:
        reasons = _mapping(row.get("terminal_reason_counts"), "terminal reasons")
        current_keys = {str(key) for key in reasons}
        reason_keys = current_keys if reason_keys is None else reason_keys
        if (
            current_keys != reason_keys
            or sum(int(value) for value in reasons.values()) != int(row.get("decision_count", -1))
            or int(reasons.get("complete_round_trip", -1))
            != int(row.get("complete_round_trip_count", -2))
        ):
            raise ValueError("repricing opportunity terminal arithmetic is invalid")
    if (
        report.get("schema_version") != POLYMARKET_REPRICING_REPORT_SCHEMA_VERSION
        or report.get("contract_sha256") != POLYMARKET_REPRICING_CONTRACT_SHA256
        or not report_sha256
        or _canonical_sha256(identity) != report_sha256
        or not str(report.get("source_run_id", ""))
        or latencies != _LATENCIES_MS
        or holding_periods != _HOLDING_PERIODS_MS
        or any(count < 0 for count in expected_market_counts.values())
        or len(opportunities) != expected_opportunities
        or len(cells) != len(_ASSETS) * len(_LATENCIES_MS) * len(_HOLDING_PERIODS_MS)
        or not report.get("confirmation_eligible")
        or not report.get("noncausal_oracle_upper_bound")
        or report.get("training_authority")
        or report.get("trading_authority")
        or report.get("profitability_claim")
    ):
        raise ValueError("repricing report failed publication validation")
    return report


def _validated_capture(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    capture = _load_json(path, "capture evidence")
    recorder = _mapping(capture.get("recorder"), "capture recorder")
    if (
        recorder.get("run_id") != report.get("source_run_id")
        or recorder.get("status") != "complete"
        or int(recorder.get("stream_gap_count", -1)) != 0
        or recorder.get("integrity_errors") != []
        or int(recorder.get("market_snapshot_count", -1))
        != sum(int(value) for value in _mapping(report["market_counts"], "market counts").values())
        or not str(recorder.get("started_at_utc", ""))
        or not str(recorder.get("ended_at_utc", ""))
    ):
        raise ValueError("capture evidence does not authorize this publication")
    return capture


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if not rows:
        raise ValueError("publication CSV cannot be empty")
    columns = tuple(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return columns


def _cell_rows(
    report: Mapping[str, Any], capture: Mapping[str, Any]
) -> tuple[dict[str, object], ...]:
    recorder = _mapping(capture["recorder"], "capture recorder")
    result = []
    for row in _rows(report["cells"], "cells"):
        result.append(
            {
                "capture_start_utc": recorder["started_at_utc"],
                "capture_end_utc": recorder["ended_at_utc"],
                "asset": row["asset"],
                "per_leg_submission_latency_ms": row["per_leg_submission_latency_ms"],
                "holding_period_ms": row["holding_period_ms"],
                "source_market_count": row["market_count"],
                "complete_market_count": row["complete_market_count"],
                "positive_market_count": row["positive_market_count"],
                "positive_market_fraction": row["positive_market_fraction"],
                "median_market_outcome_best_net_bps": row[
                    "median_market_outcome_best_net_bps"
                ],
                "maximum_market_outcome_best_net_bps": row[
                    "maximum_market_outcome_best_net_bps"
                ],
                "confirmation_eligible": str(report["confirmation_eligible"]).lower(),
                "noncausal_oracle_upper_bound": "true",
            }
        )
    return tuple(result)


def _primary_rows(
    report: Mapping[str, Any], capture: Mapping[str, Any]
) -> tuple[dict[str, object], ...]:
    config = _mapping(report["config"], "repricing config")
    recorder = _mapping(capture["recorder"], "capture recorder")
    latency = int(config["primary_per_leg_submission_latency_ms"])
    holding = int(config["primary_holding_period_ms"])
    result = []
    for row in _rows(report["opportunities"], "opportunities"):
        if (
            int(row["per_leg_submission_latency_ms"]) != latency
            or int(row["holding_period_ms"]) != holding
        ):
            continue
        result.append(
            {
                "capture_start_utc": recorder["started_at_utc"],
                "capture_end_utc": recorder["ended_at_utc"],
                "condition_id": row["condition_id"],
                "market_id": row["market_id"],
                "asset": row["asset"],
                "outcome": row["outcome"],
                "per_leg_submission_latency_ms": latency,
                "holding_period_ms": holding,
                "decision_count": row["decision_count"],
                "complete_round_trip_count": row["complete_round_trip_count"],
                "quantity": row["quantity"],
                "best_entry_cost_quote": row["best_entry_cost_quote"],
                "best_exit_proceeds_quote": row["best_exit_proceeds_quote"],
                "best_net_quote": row["best_net_quote"],
                "best_net_bps_on_entry_cost": row["best_net_bps_on_entry_cost"],
                "positive": str(row["positive"]).lower(),
                "opportunity_sha256": row["opportunity_sha256"],
            }
        )
    return tuple(result)


def _terminal_rows(report: Mapping[str, Any]) -> tuple[dict[str, object], ...]:
    counts: dict[str, int] = {}
    for row in _rows(report["opportunities"], "opportunities"):
        for reason, value in _mapping(
            row["terminal_reason_counts"], "terminal reasons"
        ).items():
            counts[str(reason)] = counts.get(str(reason), 0) + int(value)
    total = sum(counts.values())
    return tuple(
        {
            "terminal_reason": reason,
            "decision_count": count,
            "decision_fraction": format(Decimal(count) / Decimal(total), "f"),
        }
        for reason, count in sorted(counts.items())
    )


def _heat_color(value: Decimal | None, maximum: Decimal) -> str:
    if value is None:
        return "#e5e7eb"
    if value < 0:
        return "#fecaca"
    ratio = float(min(Decimal("1"), value / maximum)) if maximum > 0 else 0.0
    start = (224, 242, 241)
    end = (15, 118, 110)
    red, green, blue = (
        round(start[index] + (end[index] - start[index]) * ratio)
        for index in range(3)
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def _ceiling_svg(
    cells: Sequence[Mapping[str, object]],
    *,
    start_utc: str,
    end_utc: str,
) -> str:
    lookup = {
        (
            str(row["asset"]),
            int(row["per_leg_submission_latency_ms"]),
            int(row["holding_period_ms"]),
        ): (
            None
            if row["median_market_outcome_best_net_bps"] is None
            else _decimal(
                row["median_market_outcome_best_net_bps"],
                "median oracle bps",
            )
        )
        for row in cells
    }
    finite = [value for value in lookup.values() if value is not None and value >= 0]
    maximum = max(finite, default=Decimal("1"))
    width, height = 1320, 720
    cell_width, cell_height = 64, 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Noncausal executable repricing ceiling</title>',
        '<desc id="desc">Median best future-timed after-cost basis points by asset, submission latency, and holding period. This is not a strategy return.</desc>',
        '<rect width="1320" height="720" fill="#f8fafc"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}.title{font-size:28px;font-weight:700;fill:#111827}.sub{font-size:15px;fill:#475569}.label{font-size:13px;fill:#334155}.cell{font-size:13px;font-weight:650}.asset{font-size:19px;font-weight:700;fill:#111827}.warning{font-size:14px;font-weight:700;fill:#991b1b}</style>',
        '<text class="title" x="44" y="48">Round 8: noncausal executable repricing ceiling</text>',
        f'<text class="sub" x="44" y="76">Median best future-timed after-cost bps per market/outcome | {escape(start_utc)} to {escape(end_utc)}</text>',
        '<text class="warning" x="44" y="104">ORACLE MECHANISM SCREEN - NOT ROI, NOT A CAUSAL STRATEGY, NO TRAINING OR TRADING AUTHORITY</text>',
    ]
    hold_labels = {250: "0.25s", 500: "0.5s", 1000: "1s", 2000: "2s", 5000: "5s"}
    for panel_index, asset in enumerate(_ASSETS):
        panel_x = 52 + panel_index * 422
        grid_x = panel_x + 58
        grid_y = 190
        parts.append(f'<text class="asset" x="{panel_x}" y="150">{asset}</text>')
        parts.append(f'<text class="label" x="{panel_x}" y="171">local latency / hold</text>')
        for column, holding in enumerate(_HOLDING_PERIODS_MS):
            x = grid_x + column * cell_width + cell_width / 2
            parts.append(f'<text class="label" x="{x}" y="181" text-anchor="middle">{hold_labels[holding]}</text>')
        for row_index, latency in enumerate(_LATENCIES_MS):
            y = grid_y + row_index * cell_height
            parts.append(f'<text class="label" x="{grid_x - 10}" y="{y + 35}" text-anchor="end">{latency}ms</text>')
            for column, holding in enumerate(_HOLDING_PERIODS_MS):
                value = lookup.get((asset, latency, holding))
                x = grid_x + column * cell_width
                fill = _heat_color(value, maximum)
                display = "N/A" if value is None else f"{value:,.0f}"
                text_fill = "#ffffff" if value is not None and value / maximum > Decimal("0.52") else "#0f172a"
                parts.extend(
                    (
                        f'<rect x="{x}" y="{y}" width="{cell_width - 4}" height="{cell_height - 4}" rx="4" fill="{fill}" stroke="#cbd5e1"/>',
                        f'<text class="cell" x="{x + (cell_width - 4) / 2}" y="{y + 32}" text-anchor="middle" fill="{text_fill}">{display}</text>',
                    )
                )
    parts.extend(
        (
            '<rect x="44" y="468" width="1232" height="1" fill="#cbd5e1"/>',
            '<text class="asset" x="44" y="510">What this establishes</text>',
            '<text class="sub" x="44" y="540">Displayed depth and recorded fees still permit some short-horizon round trips after two taker legs in this small capture.</text>',
            '<text class="asset" x="44" y="584">What this does not establish</text>',
            '<text class="sub" x="44" y="614">The best decision is chosen with future books. Twelve markets cannot satisfy the frozen 30-markets-per-asset breadth gate.</text>',
            '<text class="sub" x="44" y="642">A causal predictor, independent holdout, mark-to-market risk, inventory funding, and live order evidence remain required.</text>',
            '<text class="label" x="44" y="688">Color is scaled to the largest displayed median; exact unrounded values are in repricing-cells.csv.</text>',
            '</svg>',
        )
    )
    return "\n".join(parts) + "\n"


@dataclass(frozen=True)
class PolymarketRepricingPublicationResult:
    report_sha256: str
    manifest_sha256: str
    generated_files: tuple[str, ...]


def publish_polymarket_repricing_report(
    report_path: str | Path,
    capture_evidence_path: str | Path,
    research_root: str | Path,
    *,
    round_number: int = 8,
) -> PolymarketRepricingPublicationResult:
    """Publish an integrity-bound oracle ceiling without performance claims."""

    if round_number != 8:
        raise ValueError("repricing publisher only accepts frozen research round 8")
    source = Path(report_path).resolve()
    capture_source = Path(capture_evidence_path).resolve()
    root = Path(research_root).resolve()
    report = _validated_report(source)
    capture = _validated_capture(capture_source, report)
    recorder = _mapping(capture["recorder"], "capture recorder")
    cells = _cell_rows(report, capture)
    primary = _primary_rows(report, capture)
    terminal = _terminal_rows(report)

    latest = root / "latest"
    charts = latest / "charts"
    tables = latest / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for old_chart in charts.glob("*.svg"):
        old_chart.unlink()
    for old_table in tables.glob("*.csv"):
        old_table.unlink()

    source_name = "round-008-executable-repricing-ceiling-report.json"
    source_target = root / source_name
    source_target.parent.mkdir(parents=True, exist_ok=True)
    temporary = source_target.with_name(f".{source_target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(source_target)

    table_payloads = {
        "repricing-cells.csv": cells,
        "repricing-primary-markets.csv": primary,
        "repricing-terminal-reasons.csv": terminal,
    }
    table_columns = {
        name: _write_csv(tables / name, rows)
        for name, rows in table_payloads.items()
    }
    chart = charts / "repricing-ceiling.svg"
    _write_text(
        chart,
        _ceiling_svg(
            cells,
            start_utc=str(recorder["started_at_utc"]),
            end_utc=str(recorder["ended_at_utc"]),
        ),
    )

    opportunities = _rows(report["opportunities"], "opportunities")
    complete_decisions = sum(
        int(row["complete_round_trip_count"]) for row in opportunities
    )
    positive_rows = sum(bool(row["positive"]) for row in opportunities)
    readme = f"""# Polymarket research round 8

![Noncausal executable repricing ceiling](charts/repricing-ceiling.svg)

The gap-free `{recorder['started_at_utc']}` to `{recorder['ended_at_utc']}`
capture covers 12 BTC/ETH/SOL five-minute markets and 612,522 reconstructed
books. Round 8 found {complete_decisions:,} complete two-taker oracle paths;
{positive_rows} of 480 market/outcome/grid rows had a positive best future-timed
path after displayed depth and both fee legs.

This is a **noncausal mechanism ceiling**, not ROI or a trading strategy. The
frozen primary gate has only three complete markets per asset versus 30
required. No model training, AI-edge, profitability, drawdown, or trading claim
is authorized. The next valid experiment is a preregistered causal predictor on
independent prospective evidence.

Inspect the [full signed report](../{source_name}),
[exact chart data](tables/repricing-cells.csv),
[primary market rows](tables/repricing-primary-markets.csv), and
[integrity manifest](publication-integrity.json).
"""
    readme_path = latest / "README.md"
    _write_text(readme_path, readme)

    generated = (
        source_target,
        readme_path,
        chart,
        *(tables / name for name in sorted(table_payloads)),
    )
    entries = []
    for path in generated:
        entry: dict[str, object] = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if path.suffix == ".csv":
            entry["row_count"] = len(table_payloads[path.name])
            entry["columns"] = list(table_columns[path.name])
        entries.append(entry)
    capture_relative = capture_source.relative_to(root).as_posix()
    manifest_body = {
        "schema_version": _PUBLICATION_SCHEMA,
        "round": round_number,
        "artifact_class": "exchange_sourced_noncausal_mechanism_ceiling",
        "source_report": source_name,
        "source_report_sha256": report["report_sha256"],
        "source_capture_evidence": capture_relative,
        "source_capture_evidence_file_sha256": _file_sha256(capture_source),
        "source_run_id": report["source_run_id"],
        "capture_start_utc": recorder["started_at_utc"],
        "capture_end_utc": recorder["ended_at_utc"],
        "generated_artifacts": entries,
        "reproduction_command": (
            "python -m simple_ai_trading.polymarket_repricing_publication "
            f"--report docs/model-research/polymarket/{source_name} "
            "--capture-evidence "
            f"docs/model-research/polymarket/{capture_relative} "
            "--research-root docs/model-research/polymarket"
        ),
        "claims": {
            "noncausal_oracle_upper_bound": True,
            "training_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
            "roi_claim": False,
            "drawdown_claim": False,
            "ai_edge_evaluated": False,
        },
    }
    manifest_sha256 = _canonical_sha256(manifest_body)
    manifest = {**manifest_body, "manifest_sha256": manifest_sha256}
    manifest_path = latest / "publication-integrity.json"
    _write_json(manifest_path, manifest)
    return PolymarketRepricingPublicationResult(
        report_sha256=str(report["report_sha256"]),
        manifest_sha256=manifest_sha256,
        generated_files=tuple(str(path) for path in (*generated, manifest_path)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the frozen Polymarket Round 8 repricing ceiling."
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--capture-evidence", required=True)
    parser.add_argument("--research-root", required=True)
    args = parser.parse_args(argv)
    result = publish_polymarket_repricing_report(
        args.report,
        args.capture_evidence,
        args.research_root,
    )
    print(
        json.dumps(
            {
                "report_sha256": result.report_sha256,
                "manifest_sha256": result.manifest_sha256,
                "generated_files": list(result.generated_files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PolymarketRepricingPublicationResult",
    "publish_polymarket_repricing_report",
]
