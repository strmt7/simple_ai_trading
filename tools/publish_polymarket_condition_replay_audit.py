"""Publish hash-bound condition replay eligibility without economic claims."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


AUDIT_SCHEMA_VERSION = "polymarket-condition-replay-audit-v1"
MANIFEST_SCHEMA_VERSION = "polymarket-condition-replay-publication-v2"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _load(path: Path) -> dict[str, object]:
    try:
        report = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("condition replay audit is not strict JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("condition replay audit must be an object")
    body = dict(report)
    claimed = str(body.pop("audit_sha256", "")).lower()
    if len(claimed) != 64 or claimed != _sha256(body):
        raise ValueError("condition replay audit hash differs")
    if (
        report.get("schema_version") != AUDIT_SCHEMA_VERSION
        or report.get("target_free") is not True
        or report.get("model_data_eligible") is not False
        or report.get("edge_claim") is not False
        or report.get("profitability_claim") is not False
        or not isinstance(report.get("conditions"), list)
    ):
        raise ValueError("condition replay audit claim boundary differs")
    return report


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="ascii", newline="\n")
    temporary.replace(path)


def _csv(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str:
    chunks: list[str] = []

    class Sink:
        def write(self, value: str) -> int:
            chunks.append(value)
            return len(value)

    writer = csv.DictWriter(Sink(), fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return "".join(chunks)


def _utc(milliseconds: int) -> datetime:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def _sparse_ticks(labels: Sequence[str], *, maximum: int = 12) -> tuple[list[int], list[str]]:
    if maximum < 2:
        raise ValueError("sparse tick maximum is invalid")
    if len(labels) <= maximum:
        positions = list(range(len(labels)))
    else:
        stride = math.ceil((len(labels) - 1) / (maximum - 1))
        positions = list(range(0, len(labels), stride))
        if positions[-1] != len(labels) - 1:
            positions.append(len(labels) - 1)
    return positions, [labels[index] for index in positions]


def _tables(
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_conditions = report["conditions"]
    assert isinstance(raw_conditions, list)
    conditions: list[dict[str, object]] = []
    segments: list[dict[str, object]] = []
    for raw in raw_conditions:
        if not isinstance(raw, Mapping):
            raise ValueError("condition replay audit row differs")
        raw_segments = raw.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("condition replay segment table differs")
        eligible_duration = sum(
            int(item["interval_duration_ms"])
            for item in raw_segments
            if isinstance(item, Mapping) and bool(item.get("eligible"))
        )
        market_duration = int(raw["end_ms"]) - int(raw["event_start_ms"])
        conditions.append(
            {
                "condition_id": raw["condition_id"],
                "slug": raw["slug"],
                "event_start_ms": raw["event_start_ms"],
                "event_start_utc": _utc(int(raw["event_start_ms"])).isoformat(),
                "end_ms": raw["end_ms"],
                "end_utc": _utc(int(raw["end_ms"])).isoformat(),
                "eligible": raw["eligible"],
                "eligible_duration_ms": eligible_duration,
                "eligible_duration_fraction": (
                    eligible_duration / market_duration if market_duration else 0.0
                ),
                "failure_class": raw.get("failure_class"),
                "failure": raw.get("failure"),
            }
        )
        for segment in raw_segments:
            if not isinstance(segment, Mapping):
                raise ValueError("condition replay segment row differs")
            segments.append(
                {
                    "condition_id": raw["condition_id"],
                    "slug": raw["slug"],
                    **segment,
                    "interval_start_utc": (
                        ""
                        if segment.get("interval_start_ms") is None
                        else _utc(int(segment["interval_start_ms"])).isoformat()
                    ),
                    "interval_end_utc": (
                        ""
                        if segment.get("interval_end_ms") is None
                        else _utc(int(segment["interval_end_ms"])).isoformat()
                    ),
                }
            )
    return conditions, segments


def _render(report: Mapping[str, object], path: Path, *, title: str) -> None:
    conditions, segments = _tables(report)
    started = _utc(int(report["run_started_at_ms"]))
    ended = _utc(int(report["run_ended_at_ms"]))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": str(report["audit_sha256"]),
        }
    )
    figure, (timeline, coverage) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": (2.2, 1)},
    )
    figure.subplots_adjust(
        bottom=0.16,
        hspace=0.38,
        left=0.09,
        right=0.98,
        top=0.9,
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(
        f"{title} target-free replay eligibility",
        fontsize=16,
        fontweight="bold",
        color="#17242d",
    )
    labels = [
        _utc(int(item["event_start_ms"])).strftime("%H:%M") for item in conditions
    ]
    y_by_condition = {
        str(item["condition_id"]): index for index, item in enumerate(conditions)
    }
    eligible_label_available = True
    for segment in segments:
        if not bool(segment["eligible"]):
            continue
        start = _utc(int(segment["interval_start_ms"]))
        end = _utc(int(segment["interval_end_ms"]))
        timeline.barh(
            y_by_condition[str(segment["condition_id"])],
            (end - start).total_seconds() / 86400,
            left=mdates.date2num(start),
            height=0.55,
            color="#23866f",
            label="Eligible interval" if eligible_label_available else None,
        )
        eligible_label_available = False
    excluded_label_available = True
    for item in conditions:
        if bool(item["eligible"]):
            continue
        timeline.scatter(
            _utc(int(item["event_start_ms"])),
            y_by_condition[str(item["condition_id"])],
            marker="x",
            s=55,
            linewidths=2,
            color="#bd3d4a",
            zorder=3,
            label="Excluded condition" if excluded_label_available else None,
        )
        excluded_label_available = False
    timeline.set_xlim(started, ended)
    timeline_ticks, timeline_labels = _sparse_ticks(labels)
    timeline.set_yticks(timeline_ticks, timeline_labels)
    timeline.invert_yaxis()
    timeline.set_title("Checksum-valid executable intervals", loc="left", fontweight="bold")
    timeline.set_xlabel("Receipt time (UTC)")
    timeline.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    timeline.grid(axis="x", color="#dce3e7", linewidth=0.8)
    timeline.legend(frameon=False, loc="upper right")
    timeline.spines[["top", "right"]].set_visible(False)

    percentages = [100 * float(item["eligible_duration_fraction"]) for item in conditions]
    colors = ["#23866f" if value > 0 else "#bd3d4a" for value in percentages]
    coverage.bar(range(len(percentages)), percentages, color=colors, width=0.7)
    coverage.axhline(100, color="#8f9aa2", linewidth=0.8)
    coverage.set_ylim(0, 105)
    coverage_ticks, coverage_labels = _sparse_ticks(labels)
    coverage.set_xticks(
        coverage_ticks,
        coverage_labels,
        rotation=45,
        ha="right",
    )
    coverage.set_ylabel("Eligible market time (%)")
    coverage.set_title("Condition-isolated coverage", loc="left", fontweight="bold")
    coverage.grid(axis="y", color="#dce3e7", linewidth=0.8)
    coverage.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.01,
        0.025,
        (
            f"{started:%Y-%m-%d %H:%M:%S} to {ended:%H:%M:%S} UTC | "
            f"{report['eligible_condition_count']}/{report['condition_count']} "
            "conditions eligible | no P&L, edge, or profitability claim"
        ),
        color="#53616a",
        fontsize=8.5,
    )
    figure.savefig(
        path,
        format="svg",
        facecolor="white",
        metadata={"Creator": "simple-ai-trading evidence publisher", "Date": None},
    )
    plt.close(figure)
    svg = path.read_text(encoding="utf-8")
    _write(path, "\n".join(line.rstrip() for line in svg.splitlines()) + "\n")


def publish(
    source: Path,
    output_dir: Path,
    *,
    title: str = "Polymarket",
) -> dict[str, object]:
    selected_title = str(title or "").strip()
    if not selected_title or len(selected_title) > 80:
        raise ValueError("condition replay publication title is invalid")
    report = _load(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "condition-replay-audit.json"
    shutil.copyfile(source, report_path)
    conditions, segments = _tables(report)
    conditions_path = output_dir / "condition-replay-conditions.csv"
    _write(
        conditions_path,
        _csv(
            tuple(conditions[0]),
            conditions,
        ),
    )
    segments_path = output_dir / "condition-replay-segments.csv"
    segment_columns = tuple(segments[0]) if segments else ()
    _write(segments_path, _csv(segment_columns, segments))
    chart_path = output_dir / "condition-replay-eligibility.svg"
    _render(report, chart_path, title=selected_title)
    readme_path = output_dir / "README.md"
    _write(
        readme_path,
        f"""# {selected_title} replay eligibility

> Target-free data-integrity evidence only. No P&L, edge, profitability, paper-trading, or live-trading claim.

![Condition replay eligibility](condition-replay-eligibility.svg)

| Evidence | Result |
|---|---:|
| Exact UTC capture | {_utc(int(report['run_started_at_ms'])):%Y-%m-%d %H:%M:%S} to {_utc(int(report['run_ended_at_ms'])):%Y-%m-%d %H:%M:%S} |
| Recorded conditions | {int(report['condition_count'])} |
| Condition-isolated eligible | {int(report['eligible_condition_count'])} |
| Excluded | {int(report['failed_condition_count'])} |
| Recorded stream gaps | {int(report['stream_gap_count'])} |
| Minimum executable interval | {int(report['minimum_executable_interval_ms'])} ms |

Eligibility means that a condition had a fresh two-outcome baseline and a
checksum-valid replay interval bounded by the connection's recorded lifetime.
It does not make the full capture model-eligible. The JSON and CSV files are
the numeric sources of truth; the SVG is derived from them.
""",
    )
    files = {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in (
            report_path,
            conditions_path,
            segments_path,
            chart_path,
            readme_path,
        )
    }
    body: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "title": selected_title,
        "audit_sha256": report["audit_sha256"],
        "diagnostic_only": True,
        "edge_claim": False,
        "profitability_claim": False,
        "files": files,
    }
    manifest = {**body, "manifest_sha256": _sha256(body)}
    _write(
        output_dir / "publication-manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Polymarket")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            publish(args.audit, args.output_dir, title=args.title),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
