#!/usr/bin/env python3
"""Publish the canonical Round 27 mechanics diagnostic and graph."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from simple_ai_trading.polymarket_round27_mechanics import (
    ROUND27_MECHANICS_SCHEMA_VERSION,
)
from simple_ai_trading.storage import write_bytes_atomic


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_result(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 27 mechanics result is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 mechanics result must be an object")
    body = dict(value)
    claimed = str(body.pop("mechanics_sha256", "")).lower()
    if (
        value.get("schema_version") != ROUND27_MECHANICS_SCHEMA_VERSION
        or len(claimed) != 64
        or claimed != _sha256(_canonical_json(body).encode("ascii"))
        or value.get("interpretation", {}).get("edge_claim") is not False
        or value.get("interpretation", {}).get("profitability_claim") is not False
    ):
        raise ValueError("Round 27 mechanics result claim differs")
    return value


def _csv_bytes(rows: list[dict[str, object]], fields: tuple[str, ...]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _best_cost(segments: list[dict[str, object]], key: str) -> str | None:
    values = [
        str(item[key])
        for item in segments
        if item.get(key) is not None
    ]
    return None if not values else format(min(map(float, values)), ".6f")


def _svg(result: dict[str, object]) -> bytes:
    candidates = result["candidate_counts"]
    complete = result["complete_set_latency"]
    coverage = result["coverage"]
    rows = [
        ("Extreme settlement value", int(candidates["extreme_settlement_value"]["state_count"]), "#0f766e"),
        ("Late strong favorite", int(candidates["late_strong_favorite"]["state_count"]), "#2563eb"),
        ("Complete set after fee", int(candidates["complete_set_after_fee"]["state_count"]), "#d97706"),
        ("Split-sell after fee", int(candidates["split_sell_after_fee"]["state_count"]), "#7c3aed"),
    ]
    maximum = max(
        1.0,
        *(math.log10(value + 1) for _label, value, _color in rows),
    )
    bars: list[str] = []
    for index, (label, value, color) in enumerate(rows):
        y = 204 + index * 82
        width = 430 * math.log10(value + 1) / maximum
        bars.extend(
            [
                f'<text x="70" y="{y}" class="label">{escape(label)}</text>',
                f'<rect x="70" y="{y + 14}" width="430" height="22" rx="3" fill="#e5e7eb"/>',
                f'<rect x="70" y="{y + 14}" width="{width:.2f}" height="22" rx="3" fill="{color}"/>',
                f'<text x="515" y="{y + 31}" class="value">{value:,}</text>',
            ]
        )
    stages = [
        ("Same-state episodes", int(complete["same_state_episode_count"]), "#d97706"),
        ("After 250 ms delay", int(complete["venue_delay_survivor_count"]), "#dc2626"),
        ("Sequential 500 ms", int(complete["minimum_sequential_survivor_count"]), "#991b1b"),
    ]
    funnel: list[str] = []
    for index, (label, value, color) in enumerate(stages):
        y = 216 + index * 90
        width = 230 if value else 4
        funnel.extend(
            [
                f'<text x="720" y="{y}" class="label">{escape(label)}</text>',
                f'<rect x="720" y="{y + 14}" width="{width}" height="28" rx="3" fill="{color}"/>',
                f'<text x="966" y="{y + 35}" class="value">{value}</text>',
            ]
        )
    segments = complete["segment_benchmarks"]
    best_same = _best_cost(segments, "best_same_state_cost")
    best_delay = _best_cost(segments, "best_venue_delay_cost")
    best_sequence = _best_cost(segments, "best_minimum_sequential_cost")
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="680" viewBox="0 0 1120 680">
<style>
  .title {{ font: 700 30px "Segoe UI", Arial, sans-serif; fill: #111827; }}
  .subtitle {{ font: 400 15px "Segoe UI", Arial, sans-serif; fill: #4b5563; }}
  .heading {{ font: 700 18px "Segoe UI", Arial, sans-serif; fill: #111827; }}
  .label {{ font: 600 14px "Segoe UI", Arial, sans-serif; fill: #374151; }}
  .value {{ font: 700 14px Consolas, monospace; fill: #111827; }}
  .note {{ font: 400 13px "Segoe UI", Arial, sans-serif; fill: #4b5563; }}
</style>
<rect width="1120" height="680" fill="#ffffff"/>
<text x="56" y="58" class="title">Round 27 mechanics diagnostic</text>
<text x="56" y="86" class="subtitle">{int(coverage['eligible_market_count'])} BTC five-minute markets | {int(coverage['paired_quote_state_count']):,} paired states | exact fees | recorded venue delay</text>
<line x1="56" y1="112" x2="1064" y2="112" stroke="#d1d5db"/>
<text x="70" y="158" class="heading">Candidate quote states</text>
<text x="70" y="181" class="note">Log10 bar scale; exact state count at right</text>
{''.join(bars)}
<line x1="660" y1="148" x2="660" y2="532" stroke="#e5e7eb"/>
<text x="720" y="158" class="heading">Complete-set survival</text>
<text x="720" y="181" class="note">Five-share FOK mechanics; no network latency</text>
{''.join(funnel)}
<text x="720" y="510" class="label">Best cost per complete set</text>
<text x="720" y="540" class="note">Same state</text><text x="966" y="540" class="value">{best_same}</text>
<text x="720" y="566" class="note">After venue delay</text><text x="966" y="566" class="value">{best_delay}</text>
<text x="720" y="592" class="note">Optimistic sequential</text><text x="966" y="592" class="value">{best_sequence}</text>
<line x1="56" y1="620" x2="1064" y2="620" stroke="#d1d5db"/>
<text x="56" y="649" class="note">Preregistered Stage 0 mechanics only. No edge, profitability, paper-trading, or live-trading claim.</text>
</svg>'''
    return content.encode("ascii")


def publish(result_path: Path, output_directory: Path) -> dict[str, object]:
    result = _load_result(result_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    segments = result["complete_set_latency"]["segment_benchmarks"]
    segment_fields = (
        "condition_id",
        "segment_id",
        "same_state_episode_count",
        "venue_delay_survivor_count",
        "minimum_sequential_survivor_count",
        "best_same_state_cost",
        "best_venue_delay_cost",
        "best_minimum_sequential_cost",
    )
    files: dict[str, bytes] = {
        "mechanics-diagnostic.json": (
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        ).encode("ascii"),
        "complete-set-latency.csv": _csv_bytes(segments, segment_fields),
        "mechanics-diagnostic.svg": _svg(result),
    }
    complete = result["complete_set_latency"]
    same_state = int(complete["same_state_episode_count"])
    venue_survivors = int(complete["venue_delay_survivor_count"])
    sequential_survivors = int(complete["minimum_sequential_survivor_count"])
    readme = f"""# Round 27 mechanics diagnostic

This target-free screen used {result['coverage']['eligible_market_count']} BTC five-minute markets and {result['coverage']['paired_quote_state_count']:,} synchronized quote states from the preregistered Stage 0 cohort. Exact message batches were fully applied before evaluating Up and Down together, and each condition was replayed independently with bounded memory.

The same-state screen found {same_state} after-fee complete-set episodes. {venue_survivors} survived the recorded venue delay and {sequential_survivors} survived the optimistic two-delay sequential floor. The best delayed cost was {_best_cost(complete['segment_benchmarks'], 'best_venue_delay_cost')} pUSD; the best sequential cost was {_best_cost(complete['segment_benchmarks'], 'best_minimum_sequential_cost')} pUSD per complete set, before network or order-response latency.

Extreme-price states occurred in {result['candidate_counts']['extreme_settlement_value']['market_count']} markets and late-favorite states in {result['candidate_counts']['late_strong_favorite']['market_count']} markets. These are candidate observations, not trades or an edge. Public quotes cannot prove maker fills, queue position, settlement value, or profitability.

![Round 27 mechanics diagnostic](mechanics-diagnostic.svg)

The [canonical JSON](mechanics-diagnostic.json) and [latency table](complete-set-latency.csv) are hash-bound source data. Stage 0 permits mechanics screening only and is not promotion eligible. No market edge or after-cost profitability is claimed.
"""
    files["README.md"] = readme.encode("ascii")
    for name, value in files.items():
        write_bytes_atomic(output_directory / name, value)
    manifest: dict[str, object] = {
        "schema_version": "polymarket-round27-mechanics-publication-v2",
        "mechanics_sha256": result["mechanics_sha256"],
        "diagnostic_only": True,
        "edge_claim": False,
        "profitability_claim": False,
        "files": {name: _sha256(value) for name, value in sorted(files.items())},
    }
    manifest["manifest_sha256"] = _sha256(
        _canonical_json(manifest).encode("ascii")
    )
    write_bytes_atomic(
        output_directory / "publication-manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            publish(arguments.result.resolve(), arguments.output_directory.resolve()),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
