#!/usr/bin/env python3
"""Publish the canonical Round 27 settlement-mechanics evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

import duckdb

from simple_ai_trading.polymarket_round27_resolution import (
    POLYMARKET_ROUND27_RESOLUTION_AUDIT_SCHEMA_VERSION,
    audit_round27_resolution_collection,
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


def _load_audit(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 27 resolution audit is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Round 27 resolution audit must be an object")
    body = dict(value)
    claimed = str(body.pop("audit_sha256", "")).lower()
    if (
        value.get("schema_version")
        != POLYMARKET_ROUND27_RESOLUTION_AUDIT_SCHEMA_VERSION
        or claimed != _sha256(_canonical_json(body).encode("ascii"))
        or value.get("mechanics_validation_complete") is not True
        or value.get("edge_claim") is not False
        or value.get("profitability_claim") is not False
    ):
        raise ValueError("Round 27 resolution audit claim differs")
    return value


def _utc(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1_000, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _labels(database: Path) -> list[dict[str, object]]:
    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT c.event_start_ms, c.condition_id, e.winning_outcome,
                   e.clob_payload_sha256, e.gamma_payload_sha256,
                   e.evidence_sha256
            FROM round27_resolution_condition c
            JOIN round27_resolution_evidence e USING (condition_id)
            ORDER BY c.event_start_ms, c.condition_id
            """
        ).fetchall()
    return [
        {
            "event_start_utc": _utc(int(row[0])),
            "condition_id": str(row[1]),
            "winning_outcome": str(row[2]),
            "clob_payload_sha256": str(row[3]),
            "gamma_payload_sha256": str(row[4]),
            "evidence_sha256": str(row[5]),
        }
        for row in rows
    ]


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    fields = (
        "event_start_utc",
        "condition_id",
        "winning_outcome",
        "clob_payload_sha256",
        "gamma_payload_sha256",
        "evidence_sha256",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _svg(audit: dict[str, object], labels: list[dict[str, object]]) -> bytes:
    count = int(audit["condition_count"])
    agreements = int(audit["dual_source_agreement_count"])
    up = int(audit["winner_counts"].get("Up", 0))
    down = int(audit["winner_counts"].get("Down", 0))
    scale = 500 / max(1, count)
    start = str(labels[0]["event_start_utc"])
    end = str(labels[-1]["event_start_utc"])
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500">
<style>
  .title {{ font: 700 30px "Segoe UI", Arial, sans-serif; fill: #111827; }}
  .subtitle {{ font: 400 15px "Segoe UI", Arial, sans-serif; fill: #4b5563; }}
  .heading {{ font: 700 18px "Segoe UI", Arial, sans-serif; fill: #111827; }}
  .label {{ font: 600 14px "Segoe UI", Arial, sans-serif; fill: #374151; }}
  .value {{ font: 700 16px Consolas, monospace; fill: #111827; }}
  .note {{ font: 400 13px "Segoe UI", Arial, sans-serif; fill: #4b5563; }}
</style>
<rect width="1120" height="500" fill="#ffffff"/>
<text x="56" y="58" class="title">Round 27 settlement mechanics</text>
<text x="56" y="86" class="subtitle">{escape(start)} to {escape(end)} | BTC five-minute markets</text>
<line x1="56" y1="112" x2="1064" y2="112" stroke="#d1d5db"/>
<text x="70" y="158" class="heading">Official source validation</text>
<text x="70" y="194" class="label">Eligible markets</text>
<rect x="70" y="210" width="500" height="30" rx="3" fill="#e5e7eb"/>
<rect x="70" y="210" width="{count * scale:.2f}" height="30" rx="3" fill="#2563eb"/>
<text x="590" y="232" class="value">{count}</text>
<text x="70" y="284" class="label">Gamma and CLOB agree</text>
<rect x="70" y="300" width="500" height="30" rx="3" fill="#e5e7eb"/>
<rect x="70" y="300" width="{agreements * scale:.2f}" height="30" rx="3" fill="#0f766e"/>
<text x="590" y="322" class="value">{agreements}</text>
<line x1="660" y1="148" x2="660" y2="370" stroke="#e5e7eb"/>
<text x="720" y="158" class="heading">Observed settlement labels</text>
<text x="720" y="194" class="label">Up</text>
<rect x="720" y="210" width="{up * scale:.2f}" height="30" rx="3" fill="#0f766e"/>
<text x="{740 + up * scale:.2f}" y="232" class="value">{up}</text>
<text x="720" y="284" class="label">Down</text>
<rect x="720" y="300" width="{down * scale:.2f}" height="30" rx="3" fill="#d97706"/>
<text x="{740 + down * scale:.2f}" y="322" class="value">{down}</text>
<line x1="56" y1="402" x2="1064" y2="402" stroke="#d1d5db"/>
<text x="56" y="432" class="note">Dual-source terminal-state and winner validation; raw receipts are compressed and hash-bound.</text>
<text x="56" y="458" class="note">Settlement mechanics only. No model, trade, edge, profitability, paper-trading, or live-trading claim.</text>
</svg>'''
    return content.encode("ascii")


def publish(
    *,
    source_database: Path,
    resolution_database: Path,
    audit_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    supplied = _load_audit(audit_path)
    audited = audit_round27_resolution_collection(
        resolution_database,
        source_database=source_database,
    )
    if supplied != audited:
        raise ValueError("Round 27 supplied and database resolution audits differ")
    labels = _labels(resolution_database)
    if len(labels) != audited["resolution_count"]:
        raise ValueError("Round 27 published label population differs")
    output_directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        "settlement-mechanics-audit.json": (
            json.dumps(audited, indent=2, sort_keys=True) + "\n"
        ).encode("ascii"),
        "settlement-labels.csv": _csv_bytes(labels),
        "settlement-mechanics.svg": _svg(audited, labels),
    }
    readme = f"""# Round 27 settlement mechanics

The target-access claim was persisted before querying any result. All {audited["resolution_count"]} eligible BTC five-minute markets were terminal on both official public sources, and Gamma and CLOB agreed on all {audited["dual_source_agreement_count"]} winners. The observed labels were {audited["winner_counts"]["Up"]} Up and {audited["winner_counts"]["Down"]} Down.

![Round 27 settlement mechanics](settlement-mechanics.svg)

The [canonical audit](settlement-mechanics-audit.json) and [per-market labels](settlement-labels.csv) bind every row to the two raw payload hashes and an evidence hash. Raw public receipts remain in the compact local evidence database. This validates settlement mechanics only: Stage 0 is not model-fitting data and makes no edge or profitability claim.
"""
    files["README.md"] = readme.encode("ascii")
    for name, value in files.items():
        write_bytes_atomic(output_directory / name, value)
    manifest: dict[str, object] = {
        "schema_version": "polymarket-round27-resolution-publication-v1",
        "audit_sha256": audited["audit_sha256"],
        "diagnostic_only": True,
        "edge_claim": False,
        "profitability_claim": False,
        "files": {name: _sha256(value) for name, value in sorted(files.items())},
    }
    manifest["manifest_sha256"] = _sha256(_canonical_json(manifest).encode("ascii"))
    write_bytes_atomic(
        output_directory / "publication-manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--resolution-database", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            publish(
                source_database=arguments.source_database.resolve(),
                resolution_database=arguments.resolution_database.resolve(),
                audit_path=arguments.audit.resolve(),
                output_directory=arguments.output_directory.resolve(),
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
