"""Publish the deterministic Polymarket status through failed Round 13."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = ROOT / "docs/model-research/polymarket"
LATEST_DIR = POLYMARKET_DIR / "latest"
TABLE_DIR = LATEST_DIR / "tables"
CHART_DIR = LATEST_DIR / "charts"
INTEGRITY_PATH = LATEST_DIR / "publication-integrity.json"
ROUND9_REPORT = POLYMARKET_DIR / "round-009-causal-action-value-failure-report.json"
ROUND10_CONTRACT = (
    POLYMARKET_DIR / "round-010-observability-utility-hurdle-contract.json"
)
ROUND10_REPORT = POLYMARKET_DIR / "round-010-development-hurdle-report.json"
ROUND11_CONTRACT = (
    POLYMARKET_DIR / "round-011-single-leg-directional-value-contract.json"
)
ROUND11_REPORT = POLYMARKET_DIR / "round-011-single-leg-directional-value-report.json"
ROUND11_ARTIFACT = (
    POLYMARKET_DIR / "round-011-single-leg-directional-value-artifact.json"
)
ROUND12_INVALIDATION = POLYMARKET_DIR / "round-012-invalidated-capture-evidence.json"
ROUND13_CONTRACT = POLYMARKET_DIR / "round-013-sealed-confirmation-contract.json"
ROUND13_INVALIDATION = POLYMARKET_DIR / "round-013-invalidated-capture-evidence.json"
PUBLICATION_SCHEMA_VERSION = "polymarket-round13-failed-capture-publication-v1"

COLORS = {
    "ink": "#0F172A",
    "muted": "#475569",
    "grid": "#CBD5E1",
    "background": "#F8FAFC",
    "model": "#0F766E",
    "prior": "#64748B",
    "negative": "#B42318",
    "positive": "#15803D",
    "accent": "#2563EB",
    "warning": "#D97706",
}


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json_object(path: Path) -> dict[str, object]:
    decoded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON source is not an object: {path.name}")
    return decoded


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_hashed_json(path: Path, hash_field: str) -> dict[str, object]:
    value = _read_json_object(path)
    claimed = str(value.pop(hash_field, ""))
    if len(claimed) != 64 or _canonical_sha256(value) != claimed:
        raise ValueError(f"canonical hash mismatch: {path.name}")
    value[hash_field] = claimed
    return value


def _read_round10_report() -> dict[str, object]:
    value = _read_json_object(ROUND10_REPORT)
    artifact_sha256 = str(value.pop("artifact_sha256", ""))
    if _canonical_sha256(value) != artifact_sha256:
        raise ValueError("Round 10 artifact hash differs")
    value["artifact_sha256"] = artifact_sha256
    identity_keys = (
        "schema_version",
        "contract_sha256",
        "dataset_sha256",
        "pipeline_report_sha256",
        "split_sha256",
        "model_sha256",
        "head_candidates",
        "validation_head_metrics",
        "policy_candidates",
        "selected_policy",
        "development_passed",
        "nonlinear_challenger_authorized",
        "confirmation_authorized",
        "profitability_claim",
    )
    identity = {key: value[key] for key in identity_keys}
    if _canonical_sha256(identity) != value["report_sha256"]:
        raise ValueError("Round 10 report hash differs")
    return value


def _verify_sources() -> tuple[dict[str, object], ...]:
    round9 = _read_json_object(ROUND9_REPORT)
    round9_claimed = str(round9.pop("report_canonical_sha256", ""))
    if _canonical_sha256(round9) != round9_claimed:
        raise ValueError("Round 9 failure report hash differs")
    round9["report_canonical_sha256"] = round9_claimed
    round10_contract = _read_hashed_json(ROUND10_CONTRACT, "contract_sha256")
    round10 = _read_round10_report()
    round11_contract = _read_hashed_json(ROUND11_CONTRACT, "contract_sha256")
    round11 = _read_hashed_json(ROUND11_REPORT, "report_sha256")
    artifact = _read_hashed_json(ROUND11_ARTIFACT, "artifact_sha256")
    if (
        round10["contract_sha256"] != round10_contract["contract_sha256"]
        or round11["contract_sha256"] != round11_contract["contract_sha256"]
        or round11["artifact_sha256"] != artifact["artifact_sha256"]
        or round10["development_passed"] is not False
        or round11["development_passed"] is not False
        or round10["profitability_claim"] is not False
        or round11["profitability_claim"] is not False
        or round11["trading_authority"] is not False
    ):
        raise ValueError("Round 10/11 publication truth constraints differ")
    selected = round11["selected_policy"]
    candidates = round11["policy_candidates"]
    if (
        not isinstance(selected, dict)
        or not isinstance(candidates, list)
        or len(candidates) != 30
        or int(selected["filled_conditions"]) != 42
        or int(selected["unknown_entries"]) != 0
        or bool(selected["gate_passed"])
    ):
        raise ValueError("Round 11 policy population differs")
    return round9, round10, round11, artifact


def _svg_start(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{COLORS["background"]}"/>',
        f'<text x="48" y="55" font-family="Segoe UI,Arial,sans-serif" font-size="28" font-weight="700" fill="{COLORS["ink"]}">{escape(title)}</text>',
        f'<text x="48" y="88" font-family="Segoe UI,Arial,sans-serif" font-size="15" fill="{COLORS["muted"]}">{escape(subtitle)}</text>',
    ]


def _direction_chart(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1280, 650
    left, right, top, bottom = 110, 1210, 150, 530
    maximum = max(float(row["market_prior_log_loss"]) for row in rows) * 1.15
    body = _svg_start(
        width,
        height,
        "Round 11 directional calibration",
        "Condition-weighted validation log loss; lower is better | 42 independent resolved markets",
    )
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - (bottom - top) * tick / 5
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["muted"]}">{value:.2f}</text>'
        )
    group_width = (right - left) / len(rows)
    bar_width = 54
    for index, row in enumerate(rows):
        center = left + group_width * (index + 0.5)
        for offset, key, color in (
            (-bar_width / 2, "model_log_loss", COLORS["model"]),
            (bar_width / 2, "market_prior_log_loss", COLORS["prior"]),
        ):
            value = float(row[key])
            bar_height = (bottom - top) * value / maximum
            x = center + offset - bar_width / 2
            body.append(
                f'<rect x="{x:.1f}" y="{bottom - bar_height:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="3" fill="{color}"/>'
            )
            body.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{bottom - bar_height - 9:.1f}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["ink"]}">{value:.3f}</text>'
            )
        body.append(
            f'<text x="{center:.1f}" y="{bottom + 32}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="15" font-weight="600" fill="{COLORS["ink"]}">{escape(str(row["scope"]))}</text>'
        )
    body.extend(
        [
            f'<rect x="430" y="588" width="18" height="18" rx="2" fill="{COLORS["model"]}"/><text x="458" y="602" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="{COLORS["muted"]}">Calibrated residual model</text>',
            f'<rect x="700" y="588" width="18" height="18" rx="2" fill="{COLORS["prior"]}"/><text x="728" y="602" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="{COLORS["muted"]}">Raw market midpoint prior</text>',
            f'<text x="48" y="632" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["negative"]}">Development only. The learned residual norm is near zero; most uplift is probability sharpening, not an established external-data edge.</text>',
            "</svg>",
        ]
    )
    return "\n".join(body) + "\n"


def _equity_chart(
    rows: Sequence[Mapping[str, object]], span: Mapping[str, object]
) -> str:
    width, height = 1280, 690
    left, right, top, bottom = 110, 1210, 150, 545
    equity = [float(row["cumulative_utility_quote"]) for row in rows]
    drawdown = [-float(row["drawdown_quote"]) for row in rows]
    minimum = min(0.0, min(equity), min(drawdown))
    maximum = max(0.0, max(equity))
    padding = max(1.0, (maximum - minimum) * 0.12)
    minimum -= padding
    maximum += padding

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (right - left) * index / max(1, len(rows) - 1)
        y = bottom - (bottom - top) * (value - minimum) / (maximum - minimum)
        return x, y

    body = _svg_start(
        width,
        height,
        "Round 11 sequential settlement utility",
        f"Real replay labels | {span['start']} to {span['end']} | 42 simulated fills | exact entry fees | redemption overhead unavailable",
    )
    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        _, y = point(0, value)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{left - 16}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["muted"]}">{value:.1f}</text>'
        )
    equity_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (point(i, value) for i, value in enumerate(equity))
    )
    drawdown_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (point(i, value) for i, value in enumerate(drawdown))
    )
    body.append(
        f'<polyline points="{equity_points}" fill="none" stroke="{COLORS["model"]}" stroke-width="4" stroke-linejoin="round"/>'
    )
    body.append(
        f'<polyline points="{drawdown_points}" fill="none" stroke="{COLORS["negative"]}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for index, row in enumerate(rows):
        x, y = point(index, equity[index])
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS["model"]}"/>'
        )
        if index in {0, 3, 6, 9, 13}:
            label = datetime.fromtimestamp(
                int(row["event_start_ms"]) / 1000, tz=timezone.utc
            ).strftime("%m-%d %H:%M")
            body.append(
                f'<text x="{x:.1f}" y="{bottom + 30}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="12" fill="{COLORS["muted"]}">{label}</text>'
            )
    body.extend(
        [
            f'<line x1="395" y1="600" x2="435" y2="600" stroke="{COLORS["model"]}" stroke-width="4"/><text x="447" y="605" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="{COLORS["muted"]}">Cumulative utility</text>',
            f'<line x1="650" y1="600" x2="690" y2="600" stroke="{COLORS["negative"]}" stroke-width="3"/><text x="702" y="605" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="{COLORS["muted"]}">Drawdown shown below zero</text>',
            f'<text x="48" y="654" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["negative"]}">Point estimate +22.44105 quote; maximum drawdown 12.36399; 95% bootstrap lower mean-group utility -1.38152. Gate failed.</text>',
            "</svg>",
        ]
    )
    return "\n".join(body) + "\n"


def _admission_chart(selected: Mapping[str, object]) -> str:
    width, height = 1280, 410
    left, right = 130, 1190
    lower = float(selected["bootstrap"]["lower_95_mean_group_utility_quote"])
    median = float(selected["bootstrap"]["median_mean_group_utility_quote"])
    upper = float(selected["bootstrap"]["upper_95_mean_group_utility_quote"])
    axis_min = math.floor(min(-2.0, lower - 0.5))
    axis_max = math.ceil(max(5.0, upper + 0.5))

    def x(value: float) -> float:
        return left + (right - left) * (value - axis_min) / (axis_max - axis_min)

    body = _svg_start(
        width,
        height,
        "Round 11 admission result",
        "Moving-block bootstrap of mean utility per synchronized five-minute group | 2,000 samples | block length 3",
    )
    y = 225
    body.append(
        f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="{COLORS["grid"]}" stroke-width="3"/>'
    )
    for value in range(axis_min, axis_max + 1):
        body.append(
            f'<line x1="{x(value):.1f}" y1="{y - 9}" x2="{x(value):.1f}" y2="{y + 9}" stroke="{COLORS["muted"]}"/>'
        )
        body.append(
            f'<text x="{x(value):.1f}" y="{y + 34}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["muted"]}">{value}</text>'
        )
    body.append(
        f'<line x1="{x(0):.1f}" y1="140" x2="{x(0):.1f}" y2="{y + 15}" stroke="{COLORS["negative"]}" stroke-width="2" stroke-dasharray="6 5"/>'
    )
    body.append(
        f'<line x1="{x(lower):.1f}" y1="{y}" x2="{x(upper):.1f}" y2="{y}" stroke="{COLORS["warning"]}" stroke-width="12" stroke-linecap="round"/>'
    )
    body.append(
        f'<circle cx="{x(median):.1f}" cy="{y}" r="10" fill="{COLORS["model"]}" stroke="#FFFFFF" stroke-width="3"/>'
    )
    body.extend(
        [
            f'<text x="{x(lower):.1f}" y="190" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="{COLORS["negative"]}">lower {lower:.3f}</text>',
            f'<text x="{x(median):.1f}" y="170" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="700" fill="{COLORS["model"]}">median {median:.3f}</text>',
            f'<text x="{x(upper):.1f}" y="190" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="14" fill="{COLORS["warning"]}">upper {upper:.3f}</text>',
            f'<text x="48" y="350" font-family="Segoe UI,Arial,sans-serif" font-size="15" font-weight="700" fill="{COLORS["negative"]}">FAILED: the 95% interval crosses zero, and its lower bound does not beat the market-prior policy.</text>',
            f'<text x="48" y="378" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["muted"]}">No profitability, ROI, drawdown, paper-trading, live-trading, nonlinear-model, or AI-uplift authority follows from this result.</text>',
            "</svg>",
        ]
    )
    return "\n".join(body) + "\n"


def _progress_chart(rows: Sequence[Mapping[str, object]]) -> str:
    width = 1280
    top = 135
    row_height = 82
    height = top + len(rows) * row_height + 90
    body = _svg_start(
        width,
        height,
        "Optimization evidence progression",
        "Exact status; unavailable simulated-fill counts are not zero, and counts are not ROI",
    )
    known_counts = [
        int(row["selected_filled_conditions"])
        for row in rows
        if row.get("selected_filled_conditions") is not None
    ]
    maximum = max(1, max(known_counts, default=0))
    for index, row in enumerate(rows):
        y = top + index * row_height
        raw_count = row.get("selected_filled_conditions")
        count = None if raw_count is None else int(raw_count)
        bar = 700 * (0 if count is None else count) / maximum
        color = COLORS["muted"] if count is None else COLORS["negative"]
        count_label = "N/A" if count is None else str(count)
        body.extend(
            [
                f'<text x="48" y="{y + 27}" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="700" fill="{COLORS["ink"]}">Round {row["round"]}</text>',
                f'<text x="155" y="{y + 27}" font-family="Segoe UI,Arial,sans-serif" font-size="15" fill="{COLORS["muted"]}">{escape(str(row["action"]))}</text>',
                f'<rect x="455" y="{y}" width="700" height="38" rx="4" fill="#E2E8F0"/>',
                f'<rect x="455" y="{y}" width="{bar:.1f}" height="38" rx="4" fill="{color}"/>',
                f'<text x="470" y="{y + 26}" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="700" fill="{COLORS["ink"]}">{count_label} simulated fills | {escape(str(row["status"]))}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="48" y="{height - 34}" font-family="Segoe UI,Arial,sans-serif" font-size="13" fill="{COLORS["muted"]}">Rounds 12 and 13 were invalidated before outcomes. Neither has performance metrics.</text>',
            "</svg>",
        ]
    )
    return "\n".join(body) + "\n"


def _remove_stale(expected: set[Path]) -> None:
    for directory in (TABLE_DIR, CHART_DIR):
        resolved = directory.resolve()
        if resolved.parent != LATEST_DIR.resolve():
            raise ValueError("latest publication path escaped its root")
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() and path not in expected:
                path.unlink()


def publish() -> str:
    round9, round10, round11, artifact = _verify_sources()
    round12 = _read_hashed_json(ROUND12_INVALIDATION, "artifact_sha256")
    round13_contract = _read_hashed_json(ROUND13_CONTRACT, "contract_sha256")
    round13_failure = _read_hashed_json(
        ROUND13_INVALIDATION, "artifact_sha256"
    )
    outcome_access = round12.get("outcome_access_evidence")
    persisted_counts = round12.get("persisted_table_counts")
    raw_chunk_evidence = round12.get("raw_chunk_evidence")
    freshness = round13_contract.get("freshness")
    round13_outcome_access = round13_failure.get("outcome_access_evidence")
    round13_requirement = round13_failure.get("frozen_capture_requirement")
    round13_authority = round13_failure.get("authority")
    if (
        round12.get("round") != 12
        or round12.get("status") != "invalidated_before_outcome_access"
        or not isinstance(outcome_access, Mapping)
        or outcome_access.get("performance_labels_opened") is not False
        or not isinstance(persisted_counts, Mapping)
        or not isinstance(persisted_counts.get("polymarket_market_snapshot"), int)
        or not isinstance(raw_chunk_evidence, Mapping)
        or not isinstance(raw_chunk_evidence.get("message_count"), int)
        or round13_contract.get("round") != 13
        or round13_contract.get("status") != "frozen_before_fresh_capture"
        or not isinstance(freshness, Mapping)
        or freshness.get("capture_started") is not False
        or round13_failure.get("round") != 13
        or round13_failure.get("status")
        != "failed_capture_ineligible_before_outcome_access"
        or not isinstance(round13_outcome_access, Mapping)
        or round13_outcome_access.get("performance_labels_opened") is not False
        or not isinstance(round13_requirement, Mapping)
        or round13_requirement.get("required_one_shot_capture_completed") is not False
        or not isinstance(round13_authority, Mapping)
        or any(bool(value) for value in round13_authority.values())
    ):
        raise ValueError("Round 12/13 status evidence differs")
    selected = round11["selected_policy"]
    direction = round11["direction_validation"]
    if not isinstance(selected, Mapping):
        raise ValueError("Round 11 selected policy is not an object")
    selected_bootstrap = selected.get("bootstrap")
    if not isinstance(selected_bootstrap, Mapping):
        raise ValueError("Round 11 bootstrap evidence is not an object")
    try:
        round11_conditions = int(selected["filled_conditions"])
        round11_total = float(selected["total_utility_quote"])
        round11_drawdown = float(selected["maximum_drawdown_quote"])
        round11_bootstrap_lower = float(
            selected_bootstrap["lower_95_mean_group_utility_quote"]
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 11 selected metrics are invalid") from exc
    if not all(
        math.isfinite(value)
        for value in (round11_total, round11_drawdown, round11_bootstrap_lower)
    ):
        raise ValueError("Round 11 selected metrics are non-finite")
    direction_rows: list[dict[str, object]] = []
    pooled = direction["pooled"]
    for scope, value in (("Pooled", pooled), *direction["per_asset"].items()):
        direction_rows.append(
            {
                "scope": scope,
                "conditions": int(value["conditions"]),
                "model_log_loss": float(value["model_log_loss"]),
                "market_prior_log_loss": float(value["market_prior_log_loss"]),
                "log_loss_improvement": float(value["market_prior_log_loss"])
                - float(value["model_log_loss"]),
                "model_brier_score": float(value["model_brier_score"]),
                "market_prior_brier_score": float(value["market_prior_brier_score"]),
                "brier_improvement": float(value["market_prior_brier_score"])
                - float(value["model_brier_score"]),
            }
        )
    policy_rows: list[dict[str, object]] = []
    for candidate in round11["policy_candidates"]:
        prior = candidate["market_prior_policy"]
        policy_rows.append(
            {
                "minimum_remaining_seconds": candidate["minimum_remaining_seconds"],
                "margin_quote": candidate["margin_quote"],
                "attempts": candidate["attempts"],
                "filled_conditions": candidate["filled_conditions"],
                "wins": candidate["wins"],
                "losses": candidate["losses"],
                "unknown_entries": candidate["unknown_entries"],
                "total_utility_quote": candidate["total_utility_quote"],
                "mean_condition_utility_quote": candidate[
                    "mean_condition_utility_quote"
                ],
                "median_condition_utility_quote": candidate[
                    "median_condition_utility_quote"
                ],
                "maximum_drawdown_quote": candidate["maximum_drawdown_quote"],
                "bootstrap_lower_mean_group_utility_quote": candidate["bootstrap"][
                    "lower_95_mean_group_utility_quote"
                ],
                "bootstrap_upper_mean_group_utility_quote": candidate["bootstrap"][
                    "upper_95_mean_group_utility_quote"
                ],
                "market_prior_total_utility_quote": prior["total_utility_quote"],
                "market_prior_bootstrap_lower_mean_group_utility_quote": prior[
                    "bootstrap"
                ]["lower_95_mean_group_utility_quote"],
                "gate_passed": candidate["gate_passed"],
                "gate_reasons": "|".join(candidate["gate_reasons"]),
                "candidate_sha256": candidate["candidate_sha256"],
            }
        )
    decision_by_condition = {
        row["condition_id"]: row
        for row in selected["decision_results"]
        if row["result"] in {"filled_hold_to_resolution", "unknown_blocked"}
    }
    condition_rows: list[dict[str, object]] = []
    for condition in selected["condition_results"]:
        decision = decision_by_condition.get(condition["condition_id"], {})
        condition_rows.append(
            {
                **condition,
                "decision_monotonic_ns": decision.get("decision_monotonic_ns"),
                "remaining_seconds": decision.get("remaining_seconds"),
                "p_win": decision.get("p_win"),
                "p_observable": decision.get("p_observable"),
                "p_fill": decision.get("p_fill"),
                "upper_entry_cost_quote": decision.get("upper_entry_cost_quote"),
                "exact_entry_cost_quote": decision.get("entry_cost_quote"),
                "score_quote": decision.get("score_quote"),
                "won": decision.get("won"),
                "terminal_reason": decision.get("terminal_reason"),
            }
        )
    group_utility = [float(value) for value in selected["group_utility_quote"]]
    cumulative = 0.0
    peak = 0.0
    equity_rows: list[dict[str, object]] = []
    for event_start_ms, utility in zip(
        round11["split"]["validation_groups"], group_utility, strict=True
    ):
        cumulative += utility
        peak = max(peak, cumulative)
        equity_rows.append(
            {
                "event_start_ms": int(event_start_ms),
                "event_start_utc": datetime.fromtimestamp(
                    int(event_start_ms) / 1000, tz=timezone.utc
                ).isoformat(),
                "group_utility_quote": utility,
                "cumulative_utility_quote": cumulative,
                "running_peak_quote": peak,
                "drawdown_quote": peak - cumulative,
            }
        )
    if (
        abs(cumulative - float(selected["total_utility_quote"])) > 1e-9
        or abs(
            max(float(row["drawdown_quote"]) for row in equity_rows)
            - float(selected["maximum_drawdown_quote"])
        )
        > 1e-9
    ):
        raise ValueError("Round 11 equity reconstruction differs")
    execution_rows = []
    for head, values in round11["execution_validation"].items():
        execution_rows.append({"head": head, **values})
    progression_rows = [
        {
            "round": 9,
            "action": "1 s taker scalp admission",
            "status": "failed before fit",
            "independent_groups": 47,
            "conditions": 141,
            "selected_filled_conditions": 0,
            "total_utility_quote": None,
            "maximum_drawdown_quote": None,
            "bootstrap_lower_mean_group_utility_quote": None,
            "profitability_claim": False,
        },
        {
            "round": 10,
            "action": "1 s taker scalp hurdle",
            "status": "no positive score",
            "independent_groups": 47,
            "conditions": 141,
            "selected_filled_conditions": int(
                round10["selected_policy"]["complete_count"]
            ),
            "total_utility_quote": float(
                round10["selected_policy"]["aggregate_stress_utility_quote"]
            ),
            "maximum_drawdown_quote": float(
                round10["selected_policy"]["maximum_realized_drawdown_quote"]
            ),
            "bootstrap_lower_mean_group_utility_quote": float(
                round10["selected_policy"]["bootstrap_lower_mean_group_utility_quote"]
            ),
            "profitability_claim": False,
        },
        {
            "round": 11,
            "action": "single-leg settlement hold",
            "status": "failed uncertainty gate",
            "independent_groups": 14,
            "conditions": 42,
            "selected_filled_conditions": int(selected["filled_conditions"]),
            "total_utility_quote": float(selected["total_utility_quote"]),
            "maximum_drawdown_quote": float(selected["maximum_drawdown_quote"]),
            "bootstrap_lower_mean_group_utility_quote": float(
                selected["bootstrap"]["lower_95_mean_group_utility_quote"]
            ),
            "profitability_claim": False,
        },
        {
            "round": 12,
            "action": "sealed calibration confirmation",
            "status": "invalidated before outcome access",
            "independent_groups": None,
            "conditions": None,
            "selected_filled_conditions": None,
            "total_utility_quote": None,
            "maximum_drawdown_quote": None,
            "bootstrap_lower_mean_group_utility_quote": None,
            "profitability_claim": False,
        },
        {
            "round": 13,
            "action": "slippage-limited sealed confirmation",
            "status": "failed: 1,921.322 s of 86,400 s; four stream gaps",
            "independent_groups": None,
            "conditions": None,
            "selected_filled_conditions": None,
            "total_utility_quote": None,
            "maximum_drawdown_quote": None,
            "bootstrap_lower_mean_group_utility_quote": None,
            "profitability_claim": False,
        },
    ]
    table_paths = {
        TABLE_DIR / "round11-direction-validation.csv",
        TABLE_DIR / "round11-policy-candidates.csv",
        TABLE_DIR / "round11-selected-conditions.csv",
        TABLE_DIR / "round11-equity.csv",
        TABLE_DIR / "round11-execution-validation.csv",
        TABLE_DIR / "optimization-progress.csv",
    }
    chart_paths = {
        CHART_DIR / "round11-direction-quality.svg",
        CHART_DIR / "round11-equity-drawdown.svg",
        CHART_DIR / "round11-admission.svg",
        CHART_DIR / "optimization-progress.svg",
    }
    _remove_stale(table_paths | chart_paths)
    _write_csv(
        TABLE_DIR / "round11-direction-validation.csv",
        tuple(direction_rows[0]),
        direction_rows,
    )
    _write_csv(
        TABLE_DIR / "round11-policy-candidates.csv",
        tuple(policy_rows[0]),
        policy_rows,
    )
    _write_csv(
        TABLE_DIR / "round11-selected-conditions.csv",
        tuple(condition_rows[0]),
        condition_rows,
    )
    _write_csv(TABLE_DIR / "round11-equity.csv", tuple(equity_rows[0]), equity_rows)
    execution_fields = sorted({key for row in execution_rows for key in row})
    _write_csv(
        TABLE_DIR / "round11-execution-validation.csv",
        execution_fields,
        execution_rows,
    )
    _write_csv(
        TABLE_DIR / "optimization-progress.csv",
        tuple(progression_rows[0]),
        progression_rows,
    )
    _atomic_text(
        CHART_DIR / "round11-direction-quality.svg", _direction_chart(direction_rows)
    )
    _atomic_text(
        CHART_DIR / "round11-equity-drawdown.svg",
        _equity_chart(equity_rows, round11["utc_span"]),
    )
    _atomic_text(CHART_DIR / "round11-admission.svg", _admission_chart(selected))
    _atomic_text(
        CHART_DIR / "optimization-progress.svg", _progress_chart(progression_rows)
    )
    latest_readme = f"""# Polymarket model status

![Optimization evidence progression](charts/optimization-progress.svg)

## Current boundary

Round 13 failed before outcome access. Its one-use BTC/ETH/SOL five-minute
capture stopped at `{round13_requirement["observed_duration_seconds"]}` seconds
of the required `{round13_requirement["required_duration_seconds"]}` seconds,
with `{round13_failure["raw_chunk_evidence"]["message_count"]}` persisted source
messages and `{round13_requirement["stream_gap_count"]}` stream gaps. It never
reached the frozen evaluation boundary, so every return, drawdown, fill, and
model-comparison field is unavailable, not zero.

Round 12 is not performance evidence. Its recorder captured
`{raw_chunk_evidence["message_count"]}` messages, but the evaluator
and publication chain had not been preregistered. It was invalidated before
outcome access; every return, drawdown, and fill field is therefore unavailable,
not zero.

Round 11 remains the latest scored result. Its simulated after-cost utility was
`{round11_total:+.5f}` quote on {round11_conditions} development conditions, but
maximum drawdown was
`{round11_drawdown:.5f}` and the 95% moving-block-bootstrap lower mean-group
utility was `{round11_bootstrap_lower:.5f}`. It failed uncertainty
and raw-market-prior gates. No profitability, ROI, acceptable-drawdown, paper,
AI-uplift, or trading claim exists.

## Evidence

- [Round 13 frozen contract](../round-013-sealed-confirmation-contract.json)
- [Round 13 invalidation](../round-013-invalidated-capture-evidence.json)
- [Round 12 invalidation](../round-012-invalidated-capture-evidence.json)
- [Round 11 contract](../round-011-single-leg-directional-value-contract.json)
- [Round 11 report](../round-011-single-leg-directional-value-report.json)
- [Round 11 model artifact](../round-011-single-leg-directional-value-artifact.json)
- [Optimization data](tables/optimization-progress.csv)
- [Publication integrity](publication-integrity.json)

Regenerate these exact tables, charts, and hashes with
`python tools/publish_polymarket_round11.py`. Round 13 cannot acquire paper or
live authority. Any successor requires a new prospective contract and untouched
capture, followed by separate proof of authenticated order lifecycle, balance
ownership, settlement delay, and redemption overhead.
"""
    _atomic_text(LATEST_DIR / "README.md", latest_readme)
    source_paths = (
        ROUND9_REPORT,
        ROUND10_CONTRACT,
        ROUND10_REPORT,
        ROUND11_CONTRACT,
        ROUND11_REPORT,
        ROUND11_ARTIFACT,
        ROUND12_INVALIDATION,
        ROUND13_CONTRACT,
        ROUND13_INVALIDATION,
    )
    artifact_paths = tuple(
        sorted(
            table_paths | chart_paths | set(source_paths) | {LATEST_DIR / "README.md"},
            key=lambda path: path.relative_to(ROOT).as_posix(),
        )
    )
    integrity: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "latest_round": 13,
        "status": "round13_capture_failed_ineligible",
        "source_report_sha256": round11["report_sha256"],
        "source_artifact_sha256": artifact["artifact_sha256"],
        "source_invalidation_sha256": round12["artifact_sha256"],
        "source_round13_contract_sha256": round13_contract["contract_sha256"],
        "source_round13_invalidation_sha256": round13_failure[
            "artifact_sha256"
        ],
        "artifacts": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in artifact_paths
        ],
        "profitability_claim": False,
        "roi_claim": False,
        "drawdown_claim": False,
        "paper_authority": False,
        "trading_authority": False,
    }
    integrity["publication_sha256"] = _canonical_sha256(integrity)
    _atomic_text(
        INTEGRITY_PATH,
        json.dumps(integrity, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    return str(integrity["publication_sha256"])


if __name__ == "__main__":
    print(f"Round 13 failed-capture publication: {publish()}")
