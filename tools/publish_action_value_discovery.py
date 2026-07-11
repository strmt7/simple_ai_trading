from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

try:
    from tools.run_action_value_discovery import (
        discovery_candidates,
        load_discovery_design,
    )
except ModuleNotFoundError:
    from run_action_value_discovery import (
        discovery_candidates,
        load_discovery_design,
    )


_LEGACY_IMPLEMENTATION_COMMITS = {
    9: "8a0eec2f56b8a4a727a5dacdea098ed51b9ba917",
}
CANDIDATE_FIELDS = (
    "candidate_id",
    "horizon_seconds",
    "risk_level",
    "stop_loss_bps",
    "take_profit_bps",
    "max_l1_participation",
    "fit_status",
    "artifact_status",
    "error",
    "rows",
    "positive_long_ratio",
    "positive_short_ratio",
    "mean_long_net_bps",
    "mean_short_net_bps",
    "selection_long_auc",
    "selection_short_auc",
    "policy_long_positive_edge_rows",
    "policy_short_positive_edge_rows",
    "policy_trades",
    "selection_trades",
    "selection_total_net_bps",
    "selection_mean_net_bps",
    "selection_max_drawdown_bps",
    "selection_profit_factor",
    "artifact_sha256",
    "rejection_reasons",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish truthful, hash-bound action-value discovery evidence"
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, default=None)
    parser.add_argument(
        "--prior-progress",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/model-research/action-value/latest"),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _round_number(design: Mapping[str, object]) -> int:
    value = int(design.get("round") or 0)
    if value <= 0:
        raise ValueError("action-value design round must be positive")
    return value


def _implementation_commit(design: Mapping[str, object]) -> str:
    round_number = _round_number(design)
    change_control = design.get("change_control")
    commit = (
        str(change_control.get("implementation_commit") or "")
        if isinstance(change_control, Mapping)
        else ""
    )
    commit = commit or _LEGACY_IMPLEMENTATION_COMMITS.get(round_number, "")
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError(
            "action-value design does not pin a full implementation commit"
        )
    return commit


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def _finite(value: object) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite action-value evidence")
    return parsed


def _candidate_rows(
    evidence_root: Path,
    design: Mapping[str, object],
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    completed = report.get("ranked_results")
    errors = report.get("errors")
    if not isinstance(completed, list) or not isinstance(errors, list):
        raise ValueError("discovery report candidate evidence is invalid")
    by_completed = {
        str(value.get("candidate_id")): value
        for value in completed
        if isinstance(value, Mapping)
    }
    by_error = {
        str(value.get("candidate_id")): value
        for value in errors
        if isinstance(value, Mapping)
    }
    rows: list[dict[str, object]] = []
    for candidate in discovery_candidates(design):
        candidate_id = str(candidate["candidate_id"])
        completed_value = by_completed.get(candidate_id)
        error_value = by_error.get(candidate_id)
        if (completed_value is None) == (error_value is None):
            raise ValueError(
                f"candidate outcome is missing or ambiguous: {candidate_id}"
            )
        row = {**candidate}
        if error_value is not None:
            row.update(
                {
                    "fit_status": "fit_error",
                    "artifact_status": "",
                    "error": str(error_value.get("error") or ""),
                }
            )
            rows.append(row)
            continue
        assert completed_value is not None
        artifact_path = evidence_root / f"{candidate_id}.json"
        artifact = _read_json(artifact_path)
        artifact_sha256 = _sha256(artifact_path)
        if (
            artifact_sha256 != completed_value.get("artifact_sha256")
            or artifact.get("status") != completed_value.get("status")
            or artifact.get("terminal_evaluated_at") is not None
            or artifact.get("terminal_metrics") is not None
        ):
            raise ValueError(f"candidate artifact binding failed: {candidate_id}")
        summary = artifact.get("dataset_summary")
        policy_search = artifact.get("policy_search")
        selection_metrics = artifact.get("selection_metrics")
        if not all(
            isinstance(value, Mapping)
            for value in (summary, policy_search, selection_metrics)
        ):
            raise ValueError(f"candidate financial evidence is invalid: {candidate_id}")
        assert isinstance(summary, Mapping)
        assert isinstance(policy_search, Mapping)
        assert isinstance(selection_metrics, Mapping)
        auc = completed_value.get("selection_auc")
        policy = completed_value.get("policy_metrics")
        if not isinstance(auc, Mapping) or not isinstance(policy, Mapping):
            raise ValueError(f"candidate policy evidence is invalid: {candidate_id}")
        selection_trades = int(selection_metrics["trades"])
        row.update(
            {
                "fit_status": "trained",
                "artifact_status": str(artifact["status"]),
                "error": "",
                "rows": int(summary["rows"]),
                "positive_long_ratio": _finite(summary["positive_long_ratio"]),
                "positive_short_ratio": _finite(summary["positive_short_ratio"]),
                "mean_long_net_bps": _finite(summary["mean_long_net_bps"]),
                "mean_short_net_bps": _finite(summary["mean_short_net_bps"]),
                "selection_long_auc": _finite(auc["long"]),
                "selection_short_auc": _finite(auc["short"]),
                "policy_long_positive_edge_rows": int(
                    policy_search["long_positive_edge_rows"]
                ),
                "policy_short_positive_edge_rows": int(
                    policy_search["short_positive_edge_rows"]
                ),
                "policy_trades": int(policy["trades"]),
                "selection_trades": selection_trades,
                "selection_total_net_bps": _finite(selection_metrics["total_net_bps"]),
                "selection_mean_net_bps": (
                    ""
                    if selection_trades == 0
                    else _finite(selection_metrics["mean_net_bps"])
                ),
                "selection_max_drawdown_bps": _finite(
                    selection_metrics["max_drawdown_bps"]
                ),
                "selection_profit_factor": (
                    ""
                    if selection_metrics.get("profit_factor") is None
                    else _finite(selection_metrics["profit_factor"])
                ),
                "artifact_sha256": artifact_sha256,
                "rejection_reasons": ";".join(
                    str(value) for value in artifact.get("rejection_reasons") or ()
                ),
            }
        )
        rows.append(row)
    return rows


def _svg_header(
    title: str, subtitle: str, *, width: int = 1120, height: int = 620
) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        f'<text x="52" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="26" font-weight="700" fill="#18242f">{html.escape(title)}</text>',
        f'<text x="52" y="82" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#51606d">{html.escape(subtitle)}</text>',
    ]


def _compact_candidate_label(candidate_id: object) -> str:
    value = str(candidate_id)
    risk, separator, horizon = value.rpartition("-h")
    risk_label = {
        "conservative": "C",
        "regular": "R",
        "aggressive": "A",
    }.get(risk, risk[:1].upper())
    return f"{risk_label} {horizon}s" if separator and horizon.isdigit() else value


def _forecast_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    round_number: int,
) -> str:
    trained = [row for row in rows if row["fit_status"] == "trained"]
    lines = _svg_header(
        f"Round {round_number} forecast quality",
        "Selection AUC only; C/R/A denote conservative, regular, and aggressive risk profiles.",
    )
    left, top, chart_width, chart_height = 90, 120, 960, 380
    auc_values = [
        _finite(row[field])
        for row in trained
        for field in ("selection_long_auc", "selection_short_auc")
    ]
    axis_upper = min(
        1.0,
        max(0.7, math.ceil((max(auc_values, default=0.7) + 0.03) * 10.0) / 10.0),
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    for tick in range(0, int(round(axis_upper * 10.0)) + 1):
        value = tick / 10.0
        y = top + chart_height - value / axis_upper * chart_height
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#e5ebf0"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value:.1f}</text>'
        )
    y_half = top + chart_height - 0.5 / axis_upper * chart_height
    lines.append(
        f'<line x1="{left}" y1="{y_half:.1f}" x2="{left + chart_width}" y2="{y_half:.1f}" stroke="#b44b4b" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    group = chart_width / max(1, len(trained))
    for index, row in enumerate(trained):
        center = left + group * (index + 0.5)
        for offset, field, color in (
            (-14, "selection_long_auc", "#218c8c"),
            (14, "selection_short_auc", "#d59b2d"),
        ):
            value = _finite(row[field])
            height = value / axis_upper * chart_height
            lines.append(
                f'<rect x="{center + offset - 10:.1f}" y="{top + chart_height - height:.1f}" width="20" height="{height:.1f}" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{center + offset:.1f}" y="{top + chart_height - height - 7:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#263744">{value:.3f}</text>'
            )
        label = _compact_candidate_label(row["candidate_id"])
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334653">{html.escape(label)}</text>'
        )
    lines.extend(
        [
            '<rect x="390" y="560" width="16" height="16" fill="#218c8c"/><text x="414" y="573" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Long AUC</text>',
            '<rect x="525" y="560" width="16" height="16" fill="#d59b2d"/><text x="549" y="573" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Short AUC</text>',
            '<line x1="675" y1="568" x2="705" y2="568" stroke="#b44b4b" stroke-width="2" stroke-dasharray="7 5"/><text x="713" y="573" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Random</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _economics_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    round_number: int,
) -> str:
    trained = [row for row in rows if row["fit_status"] == "trained"]
    values = [
        _finite(row[field])
        for row in trained
        for field in ("mean_long_net_bps", "mean_short_net_bps")
    ]
    lower = min(-1.0, min(values, default=-1.0) * 1.15)
    upper = max(2.0, max(values, default=0.0) * 1.15)
    lines = _svg_header(
        f"Round {round_number} after-cost economics",
        "Mean executable labels are negative; selected-trade mean is undefined because every policy abstained.",
    )
    left, top, chart_width, chart_height = 90, 120, 960, 380
    scale = chart_height / (upper - lower)
    zero_y = top + upper * scale
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{left + chart_width}" y2="{zero_y:.1f}" stroke="#526674" stroke-width="2"/>'
    )
    group = chart_width / max(1, len(trained))
    bars = (
        (-24, "mean_long_net_bps", "#218c8c"),
        (24, "mean_short_net_bps", "#d59b2d"),
    )
    for index, row in enumerate(trained):
        center = left + group * (index + 0.5)
        for offset, field, color in bars:
            value = _finite(row[field])
            value_y = zero_y - value * scale
            y = min(zero_y, value_y)
            height = max(1.0, abs(value_y - zero_y))
            lines.append(
                f'<rect x="{center + offset - 11:.1f}" y="{y:.1f}" width="22" height="{height:.1f}" fill="{color}"/>'
            )
            label_y = value_y - 7 if value >= 0 else value_y + 17
            lines.append(
                f'<text x="{center + offset:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#263744">{value:.2f}</text>'
            )
        label = _compact_candidate_label(row["candidate_id"])
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334653">{html.escape(label)}</text>'
        )
    lines.extend(
        [
            '<text x="52" y="330" transform="rotate(-90 52 330)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">Basis points per trade</text>',
            '<rect x="390" y="560" width="16" height="16" fill="#218c8c"/><text x="414" y="573" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Long executable label</text>',
            '<rect x="610" y="560" width="16" height="16" fill="#d59b2d"/><text x="634" y="573" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Short executable label</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _funnel_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    round_number: int,
) -> str:
    positive_edge_rows = sum(
        int(row.get("policy_long_positive_edge_rows") or 0)
        + int(row.get("policy_short_positive_edge_rows") or 0)
        for row in rows
    )
    stages = (
        ("Precommitted candidates", len(rows), "#287f9e"),
        (
            "Statistically trainable",
            sum(row["fit_status"] == "trained" for row in rows),
            "#218c8c",
        ),
        (
            "Policy-viable candidates",
            sum(int(row.get("policy_trades") or 0) > 0 for row in rows),
            "#d59b2d",
        ),
        (
            "Selection-active candidates",
            sum(int(row.get("selection_trades") or 0) > 0 for row in rows),
            "#b44b4b",
        ),
        (
            "Unrejected candidates",
            sum(row.get("artifact_status") == "candidate" for row in rows),
            "#7b559c",
        ),
    )
    lines = _svg_header(
        f"Round {round_number} signal selection",
        f"Candidate-level stages only; artifacts separately record {positive_edge_rows} positive predicted-edge rows.",
    )
    maximum = max(value for _label, value, _color in stages)
    for index, (label, value, color) in enumerate(stages):
        y = 130 + index * 82
        width = 760 * value / max(1, maximum)
        lines.append(
            f'<text x="52" y="{y + 27}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#334653">{html.escape(label)}</text>'
        )
        lines.append(f'<rect x="300" y="{y}" width="760" height="44" fill="#e6ebef"/>')
        if width > 0:
            lines.append(
                f'<rect x="300" y="{y}" width="{width:.1f}" height="44" fill="{color}"/>'
            )
        lines.append(
            f'<text x="{min(1045, 318 + width):.1f}" y="{y + 29}" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="#18242f">{value}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _progress_svg(progress: Sequence[Mapping[str, object]]) -> str:
    after_cost = [row for row in progress if int(row["round"]) >= 7]
    measured = [
        row
        for row in after_cost
        if str(row.get("mean_net_bps", "")).strip()
        and int(row.get("executable_trades") or 0) > 0
    ]
    lines = _svg_header(
        "After-cost research progress",
        "Mean net return is plotted only for rounds with executable trades; abstaining rounds are marked.",
    )
    left, top, chart_width, chart_height = 100, 120, 930, 380
    values = [_finite(row["mean_net_bps"]) for row in measured]
    lower = min(-14.0, min(values, default=-14.0) - 1.0)
    upper = max(2.0, max(values, default=0.0) + 1.0)
    scale = chart_height / (upper - lower)
    zero_y = top + upper * scale
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    for value in (-15.0, -10.0, -5.0, 0.0):
        if lower <= value <= upper:
            y = zero_y - value * scale
            color = "#526674" if value == 0.0 else "#e5ebf0"
            width = 2 if value == 0.0 else 1
            lines.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="{color}" stroke-width="{width}"/>'
            )
            lines.append(
                f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value:.0f}</text>'
            )
    measured_points: list[tuple[float, float, Mapping[str, object]]] = []
    for index, row in enumerate(after_cost):
        x = left + chart_width * (index + 0.5) / max(1, len(after_cost))
        text_value = str(row.get("mean_net_bps", "")).strip()
        if not text_value or int(row.get("executable_trades") or 0) <= 0:
            lines.append(
                f'<line x1="{x:.1f}" y1="{top + 35}" x2="{x:.1f}" y2="{top + chart_height - 35}" stroke="#a7b2bb" stroke-width="2" stroke-dasharray="5 6"/>'
            )
            lines.append(
                f'<rect x="{x - 7:.1f}" y="{top + chart_height / 2 - 7:.1f}" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 {x:.1f} {top + chart_height / 2:.1f})"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{top + chart_height / 2 - 24:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263744">No executable trades</text>'
            )
            continue
        value = _finite(row["mean_net_bps"])
        y = zero_y - value * scale
        measured_points.append((x, y, row))
    if len(measured_points) > 1:
        lines.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y, _row in measured_points)
            + '" fill="none" stroke="#287f9e" stroke-width="4"/>'
        )
    for x, y, row in measured_points:
        value = _finite(row["mean_net_bps"])
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#b44b4b" stroke="#ffffff" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263744">{value:+.2f} bps</text>'
        )
    for index, row in enumerate(after_cost):
        x = left + chart_width * (index + 0.5) / max(1, len(after_cost))
        lines.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 30}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Round {row["round"]}</text>'
        )
    lines.append(
        '<text x="42" y="310" transform="rotate(-90 42 310)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">Mean net basis points per trade</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def publish(
    evidence_root: Path,
    design_path: Path,
    prior_progress_path: Path | None,
    output_dir: Path,
    diagnostics_path: Path | None = None,
) -> dict[str, object]:
    design = load_discovery_design(design_path)
    round_number = _round_number(design)
    implementation_commit = _implementation_commit(design)
    report_path = evidence_root / "report.json"
    report = _read_json(report_path)
    if (
        report.get("design_sha256") != design["design_sha256"]
        or report.get("status") not in {"completed", "failed"}
        or report.get("terminal_holdout_accessed") is not False
        or report.get("trading_authority") is not False
        or report.get("profitability_claim") is not False
        or int(report.get("candidate_count") or 0) != int(design["candidate_count"])
    ):
        raise ValueError(f"Round {round_number} report contract is invalid")
    rows = _candidate_rows(evidence_root, design, report)
    trained = [row for row in rows if row["fit_status"] == "trained"]
    fit_error_count = sum(row["fit_status"] == "fit_error" for row in rows)
    if len(trained) + fit_error_count != len(rows):
        raise ValueError("action-value candidate outcome count is inconsistent")
    unrejected = [row for row in trained if row["artifact_status"] == "candidate"]
    semantic_status = "candidate" if unrejected else "rejected"
    ranked_results = report.get("ranked_results")
    if not isinstance(ranked_results, list) or len(ranked_results) != len(trained):
        raise ValueError("action-value ranked result count is inconsistent")
    diagnostic: dict[str, object] | None = None
    if diagnostics_path is not None:
        diagnostic = _read_json(diagnostics_path)
        claimed_diagnostic_sha256 = str(diagnostic.get("diagnostic_sha256") or "")
        canonical_diagnostic = dict(diagnostic)
        canonical_diagnostic.pop("diagnostic_sha256", None)
        if (
            diagnostic.get("design_sha256") != design["design_sha256"]
            or diagnostic.get("source_report_sha256") != _sha256(report_path)
            or diagnostic.get("terminal_holdout_accessed") is not False
            or diagnostic.get("trading_authority") is not False
            or diagnostic.get("profitability_claim") is not False
            or claimed_diagnostic_sha256 != _canonical_sha256(canonical_diagnostic)
            or not isinstance(diagnostic.get("candidates"), list)
            or len(diagnostic["candidates"]) != len(rows)
        ):
            raise ValueError("action-value diagnostic binding is invalid")
    elif round_number >= 10:
        raise ValueError("Round 10 and later publications require diagnostics")

    progress_source = prior_progress_path
    if progress_source is None:
        progress_source = (
            Path("docs/model-research/tape-depth/latest/progress.csv")
            if round_number == 9
            else output_dir / "progress.csv"
        )
    with progress_source.open("r", encoding="utf-8", newline="") as handle:
        progress = list(csv.DictReader(handle))
    if progress and int(progress[-1]["round"]) == round_number:
        progress = progress[:-1]
    if not progress or int(progress[-1]["round"]) != round_number - 1:
        raise ValueError(
            f"prior progress table does not end at Round {round_number - 1}"
        )
    best = ranked_results[0] if ranked_results else None
    best_selection = (
        best.get("selection_metrics") if isinstance(best, Mapping) else None
    )
    best_policy = best.get("policy_metrics") if isinstance(best, Mapping) else None
    best_auc = best.get("selection_auc") if isinstance(best, Mapping) else None
    if not all(
        value is None or isinstance(value, Mapping)
        for value in (best_selection, best_policy, best_auc)
    ):
        raise ValueError("best action-value candidate metrics are invalid")
    best_selection_trades = (
        int(best_selection.get("trades") or 0)
        if isinstance(best_selection, Mapping)
        else 0
    )
    best_mean_net = (
        _finite(best_selection["mean_net_bps"])
        if isinstance(best_selection, Mapping) and best_selection_trades > 0
        else ""
    )
    data = design["data"]
    training = design["training"]
    horizons = design["horizon_seconds"]
    risk_profiles = design["risk_profiles"]
    assert isinstance(data, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(horizons, list)
    assert isinstance(risk_profiles, Mapping)
    progress.append(
        {
            "round": round_number,
            "stage": "exact-BBO action-value discovery",
            "periods": f"{data['start_date']}..{data['end_date']}",
            "selection_contaminated": True,
            "horizon_seconds": ";".join(str(value) for value in horizons),
            "feature_set": str(training["feature_version"]),
            "risk_level": ";".join(str(value) for value in risk_profiles),
            "direction_auc": (
                (_finite(best_auc["long"]) + _finite(best_auc["short"])) / 2.0
                if isinstance(best_auc, Mapping)
                else ""
            ),
            "spearman_ic": "",
            "selected_signals": (
                int(best_policy.get("trades") or 0)
                if isinstance(best_policy, Mapping)
                else 0
            ),
            "executable_trades": best_selection_trades,
            "mean_gross_bps": "",
            "mean_net_bps": best_mean_net,
            "status": semantic_status,
            "source_file": f"action-value Round {round_number} report",
        }
    )

    charts = output_dir / "charts"
    if diagnostic is not None:
        _write_text(
            output_dir / "diagnostics.json",
            json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    _write_csv(output_dir / "candidates.csv", rows, CANDIDATE_FIELDS)
    _write_csv(output_dir / "progress.csv", progress, tuple(progress[0]))
    _write_text(
        charts / "forecast-quality.svg",
        _forecast_svg(rows, round_number=round_number),
    )
    _write_text(
        charts / "after-cost-performance.svg",
        _economics_svg(rows, round_number=round_number),
    )
    _write_text(
        charts / "signal-selection.svg",
        _funnel_svg(rows, round_number=round_number),
    )
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    policy_trades = sum(int(row.get("policy_trades") or 0) for row in trained)
    selection_trades = sum(int(row.get("selection_trades") or 0) for row in trained)
    positive_edge_rows = sum(
        int(row.get("policy_long_positive_edge_rows") or 0)
        + int(row.get("policy_short_positive_edge_rows") or 0)
        for row in trained
    )
    certificate_values = {
        str(value.get("corpus_certificate_sha256") or "")
        for value in ranked_results
        if isinstance(value, Mapping)
        and str(value.get("corpus_certificate_sha256") or "")
    }
    if len(certificate_values) > 1:
        raise ValueError("trained candidates bind different corpus certificates")
    certificate = next(iter(certificate_values), "unavailable-no-trained-artifact")
    no_trade_note = (
        "No trained candidate selected an executable trade; per-trade mean return\n"
        "is undefined and no equity curve is generated."
        if selection_trades == 0
        else (
            f"Across trained research candidates, {selection_trades} selection "
            "trades were observed; candidate-level economics remain separate in "
            "`candidates.csv`."
        )
    )
    readme = f"""# Action-Value Round {round_number} Evidence

Status: **{semantic_status}**. This is checksummed Binance USD-M discovery evidence, not
a profitability, execution, or trading-authority claim.

- UTC window: {data["start_date"]} through {data["end_date"]} (now consumed for selection)
- Precommitted candidates: {len(rows)}
- Statistical fit failures: {fit_error_count}
- Trained candidates: {len(trained)}
- Unrejected candidates: {len(unrejected)}
- Policy trades across trained candidates: {policy_trades}
- Selection trades across trained candidates: {selection_trades}
- Positive predicted-edge policy rows: {positive_edge_rows}
- Design SHA-256: `{design["design_sha256"]}`
- Corpus certificate SHA-256: `{certificate}`
- Implementation commit: `{implementation_commit}`

{no_trade_note}

Fit errors and every trained artifact hash are retained verbatim
in the source table. A failed fit is not counted as a zero-return model, and an
abstaining model is not presented as profitable.

## Charts

![After-cost performance](charts/after-cost-performance.svg)

![Forecast quality](charts/forecast-quality.svg)

![Signal selection](charts/signal-selection.svg)

![Research progress](charts/research-progress.svg)

The source tables are [candidates.csv](candidates.csv) and
[progress.csv](progress.csv); reconstructed class support and top-score outcomes
are in [diagnostics.json](diagnostics.json) when required by the round. Every
trained artifact SHA-256 and every fit error is retained in `candidates.csv`;
no zero-trade equity curve is fabricated.
Regenerate by passing this round's `--design`, `--evidence-root`, and required
`--diagnostics` to `python tools/publish_action_value_discovery.py`.
"""
    _write_text(output_dir / "README.md", readme)
    generated = [
        output_dir / "README.md",
        output_dir / "candidates.csv",
        output_dir / "progress.csv",
        charts / "after-cost-performance.svg",
        charts / "forecast-quality.svg",
        charts / "signal-selection.svg",
        charts / "research-progress.svg",
    ]
    if diagnostic is not None:
        generated.append(output_dir / "diagnostics.json")
    publication = {
        "schema_version": "action-value-discovery-publication-v1",
        "artifact_class": "exchange_sourced_model_discovery_graph_data",
        "round": round_number,
        "status": semantic_status,
        "runner_status": report["status"],
        "trading_authority": False,
        "profitability_claim": False,
        "terminal_holdout_accessed": False,
        "selection_window_is_consumed": True,
        "design_sha256": design["design_sha256"],
        "source_report_sha256": _sha256(report_path),
        "implementation_commit": implementation_commit,
        "corpus_certificate_sha256": certificate,
        "diagnostic_sha256": (
            diagnostic.get("diagnostic_sha256") if diagnostic is not None else None
        ),
        "actual": {
            "candidate_count": len(rows),
            "fit_error_count": fit_error_count,
            "trained_candidate_count": len(trained),
            "unrejected_candidate_count": len(unrejected),
            "policy_positive_edge_rows": positive_edge_rows,
            "policy_trades": policy_trades,
            "selection_trades": selection_trades,
        },
        "artifact_integrity": [
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in generated
        ],
        "candidates": rows,
    }
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return publication


def main() -> int:
    args = _arguments()
    try:
        publication = publish(
            args.evidence_root,
            args.design,
            args.prior_progress,
            args.output_dir,
            args.diagnostics,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"publish-action-value-discovery failed: {type(exc).__name__}: {exc}")
        return 2
    print(
        "publish-action-value-discovery: "
        f"status={publication['status']} artifacts={len(publication['artifact_integrity'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
