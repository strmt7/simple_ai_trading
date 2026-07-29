from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
from html import escape
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_EVALUATION = Path(
    "docs/model-research/polymarket/evidence/"
    "round-014-btc-5m-shadow-hour-evaluation-v1.json"
)
DEFAULT_OUTPUT = Path("docs/model-research/polymarket/latest/shadow")


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
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load_evaluation(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("shadow evaluation must be an object")
    claimed = str(payload.pop("artifact_sha256", "")).lower()
    if len(claimed) != 64 or _canonical_sha256(payload) != claimed:
        raise ValueError("shadow evaluation artifact hash differs")
    payload["artifact_sha256"] = claimed
    if (
        payload.get("schema_version")
        != "polymarket-btc-shadow-hour-evaluation-v1"
        or payload.get("profitability_claim") is not False
        or payload.get("trading_authority") is not False
    ):
        raise ValueError("shadow evaluation authority contract differs")
    counterfactual = _mapping(
        payload.get("first_candidate_counterfactual"),
        name="first-candidate counterfactual",
    )
    if (
        int(counterfactual.get("real_orders_submitted", -1)) != 0
        or int(counterfactual.get("real_fills_observed", -1)) != 0
        or counterfactual.get("displayed_depth_fill_counterfactual_only") is not True
    ):
        raise ValueError("shadow evaluation contains real execution evidence")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("shadow evaluation has no events")
    return payload


def _utc(milliseconds: int) -> str:
    return datetime.fromtimestamp(
        int(milliseconds) / 1000,
        tz=timezone.utc,
    ).isoformat(timespec="seconds").replace("+00:00", "Z")


def _event_rows(evaluation: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cumulative = Decimal("0")
    events = evaluation["events"]
    assert isinstance(events, list)
    for ordinal, raw_event in enumerate(events, start=1):
        event = _mapping(raw_event, name="shadow event")
        entry = _mapping(event.get("selected_entry"), name="selected entry")
        metrics = _mapping(event.get("prediction_metrics"), name="event metrics")
        pnl = Decimal(str(entry["counterfactual_net_pnl_quote"]))
        cumulative += pnl
        rows.append(
            {
                "ordinal": ordinal,
                "event_start_utc": _utc(int(event["event_start_ms"])),
                "event_end_utc": _utc(int(event["event_end_ms"])),
                "condition_id": str(event["condition_id"]),
                "winner": str(event["winner"]),
                "selected_outcome": str(entry["selected_outcome"]),
                "selected_outcome_won": bool(entry["selected_outcome_won"]),
                "counterfactual_net_pnl_quote": format(pnl, "f"),
                "cumulative_counterfactual_net_pnl_quote": format(cumulative, "f"),
                "observed_prediction_rows": int(event["observed_prediction_rows"]),
                "directional_accuracy": float(metrics["directional_accuracy"]),
                "mean_log_loss": float(metrics["mean_log_loss"]),
                "mean_brier_score": float(metrics["mean_brier_score"]),
            }
        )
    expected = Decimal(
        str(
            _mapping(
                evaluation["first_candidate_counterfactual"],
                name="counterfactual",
            )["net_pnl_quote"]
        )
    )
    if cumulative != expected:
        raise ValueError("shadow event PnL does not reconcile to summary")
    return rows


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        raise ValueError("cannot publish an empty shadow table")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(rows[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _pnl_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1200, 640
    left, right, top, bottom = 105, 55, 95, 105
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = [
        Decimal(str(row["cumulative_counterfactual_net_pnl_quote"]))
        for row in rows
    ]
    minimum = min(min(values), Decimal("0"))
    maximum = max(max(values), Decimal("0"))
    span = maximum - minimum
    padding = max(Decimal("1"), span * Decimal("0.12"))
    y_min = minimum - padding
    y_max = maximum + padding

    def x_at(index: int) -> float:
        if len(rows) == 1:
            return left + chart_width / 2
        return left + chart_width * index / (len(rows) - 1)

    def y_at(value: Decimal) -> float:
        return top + chart_height * float((y_max - value) / (y_max - y_min))

    points = " ".join(
        f"{x_at(index):.2f},{y_at(value):.2f}"
        for index, value in enumerate(values)
    )
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title description">'
        ),
        '<title id="title">Round 14 one-hour cumulative shadow P&amp;L</title>',
        (
            '<desc id="description">Counterfactual after-cost quote P&amp;L '
            f'across {len(rows)} resolved BTC five-minute events. No real '
            "orders or fills occurred.</desc>"
        ),
        '<rect width="1200" height="640" fill="#f8fafc"/>',
        (
            '<text x="105" y="42" fill="#111827" font-family="Segoe UI, sans-serif" '
            'font-size="28" font-weight="700">Round 14 one-hour shadow</text>'
        ),
        (
            '<text x="105" y="72" fill="#475569" font-family="Segoe UI, sans-serif" '
            'font-size="17">Cumulative counterfactual after-cost quote P&amp;L; '
            "zero real orders and fills</text>"
        ),
    ]
    for index in range(6):
        fraction = Decimal(index) / Decimal(5)
        value = y_max - (y_max - y_min) * fraction
        y = top + chart_height * index / 5
        lines.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" '
                    f'y2="{y:.2f}" stroke="#dbe2ea" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 16}" y="{y + 6:.2f}" text-anchor="end" '
                    'fill="#475569" font-family="Segoe UI, sans-serif" '
                    f'font-size="14">{escape(format(value, ".2f"))}</text>'
                ),
            ]
        )
    zero_y = y_at(Decimal("0"))
    lines.append(
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" '
        f'y2="{zero_y:.2f}" stroke="#64748b" stroke-width="2"/>'
    )
    lines.append(
        f'<polyline points="{points}" fill="none" stroke="#b42318" '
        'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        color = "#067647" if Decimal(
            str(row["counterfactual_net_pnl_quote"])
        ) > 0 else "#b42318"
        lines.append(
            f'<circle cx="{x_at(index):.2f}" cy="{y_at(value):.2f}" r="5" '
            f'fill="{color}" stroke="#ffffff" stroke-width="2"/>'
        )
    tick_indices = sorted({0, len(rows) // 2, len(rows) - 1})
    for index in tick_indices:
        label = str(rows[index]["event_start_utc"])[11:16]
        lines.append(
            f'<text x="{x_at(index):.2f}" y="{height - 62}" text-anchor="middle" '
            'fill="#475569" font-family="Segoe UI, sans-serif" '
            f'font-size="14">{escape(label)} UTC</text>'
        )
    lines.extend(
        [
            (
                f'<text x="30" y="{top + chart_height / 2:.2f}" '
                'transform="rotate(-90 30 '
                f'{top + chart_height / 2:.2f})" text-anchor="middle" '
                'fill="#334155" font-family="Segoe UI, sans-serif" '
                'font-size="15">Quote P&amp;L</text>'
            ),
            (
                f'<text x="{width / 2:.2f}" y="{height - 22}" text-anchor="middle" '
                'fill="#64748b" font-family="Segoe UI, sans-serif" '
                f'font-size="14">Source: {len(rows)} jointly resolved official '
                "CLOB/Gamma conditions</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _readme(evaluation: Mapping[str, object]) -> str:
    counterfactual = _mapping(
        evaluation["first_candidate_counterfactual"],
        name="counterfactual",
    )
    metrics = _mapping(
        evaluation["condition_balanced_prediction_metrics"],
        name="prediction metrics",
    )
    population = _mapping(evaluation["population"], name="population")
    return f"""# Round 14 one-hour shadow

> **Rejected. Counterfactual only. No orders, fills, profitability claim, or trading authority.**

![Cumulative shadow P&L](cumulative-pnl.svg)

The frozen first-candidate policy won
`{int(counterfactual["winning_events"])}` of
`{int(counterfactual["selected_events"])}` resolved events. Counterfactual net
P&L after observed displayed-depth costs was
`{counterfactual["net_pnl_quote"]}` quote. Profit factor was
`{float(counterfactual["profit_factor"]):.4f}` and maximum drawdown was
`{counterfactual["maximum_drawdown_quote"]}` quote. Condition-balanced row
accuracy was `{float(metrics["directional_accuracy"]):.4f}`, but it did not
translate into an executable event-selection edge.

The run observed `{int(population["observed_prediction_rows"])}` prediction
rows and `{int(population["opportunity_errors"])}` opportunity errors. It
submitted zero real orders and observed zero real fills.

## Audit

- [Event-level source table](event-outcomes.csv)
- [Publication integrity](publication-integrity.json)
- [Hash-bound evaluation](../../evidence/round-014-btc-5m-shadow-hour-evaluation-v1.json)
- [Exact source log](../../evidence/round-014-btc-5m-shadow-hour-1785350261587.jsonl)

Regenerate with `python tools/publish_polymarket_shadow_hour.py`.
"""


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="ascii", newline="\n")
    os.replace(temporary, path)


def publish(evaluation_path: Path, output: Path) -> dict[str, object]:
    evaluation = _load_evaluation(evaluation_path)
    rows = _event_rows(evaluation)
    files = {
        "README.md": _readme(evaluation),
        "event-outcomes.csv": _csv_text(rows),
        "cumulative-pnl.svg": _pnl_svg(rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        _write_atomic(output / name, text)
    artifacts = []
    for name in sorted(files):
        path = output / name
        entry: dict[str, object] = {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if name == "event-outcomes.csv":
            entry["row_count"] = len(rows)
        if name == "cumulative-pnl.svg":
            entry["source_table"] = "event-outcomes.csv"
        artifacts.append(entry)
    manifest: dict[str, object] = {
        "schema_version": "polymarket-shadow-hour-publication-v1",
        "source_evaluation_path": evaluation_path.as_posix(),
        "source_evaluation_file_sha256": _file_sha256(evaluation_path),
        "source_evaluation_artifact_sha256": evaluation["artifact_sha256"],
        "asset": "BTC",
        "market_variant": "fiveminute",
        "resolved_events": len(rows),
        "real_orders_submitted": 0,
        "real_fills_observed": 0,
        "profitability_claim": False,
        "trading_authority": False,
        "manual_chart_edits_permitted": False,
        "artifacts": artifacts,
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_atomic(
        output / "publication-integrity.json",
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish hash-verified Polymarket shadow-hour tables and SVG."
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = publish(args.evaluation, args.output)
    print(_canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
