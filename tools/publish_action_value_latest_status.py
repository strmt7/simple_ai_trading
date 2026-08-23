"""Publish the current action-value research status without rewriting old results."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
BASE_PROGRESS = RESEARCH / "latest" / "progress.csv"
OUTPUT = RESEARCH / "latest-status"
LATEST_ROUND = 76
SOURCE_SPECS = {
    "round73": (
        RESEARCH / "round-073-v9-corpus-invalidation-2026-07-25.json",
        "artifact_sha256",
    ),
    "round74": (
        RESEARCH / "round-074-terminal-campaign-outcome-v2-2026-08-10.json",
        "artifact_sha256",
    ),
    "round75": (
        RESEARCH / "round-075-terminal-campaign-audit-2026-08-23.json",
        "artifact_sha256",
    ),
    "round76_preregistration": (
        RESEARCH / "round-076-mark-conditioned-duration-preregistration-v1.json",
        "preregistration_sha256",
    ),
    "round76_adjudication": (
        RESEARCH / "round-076-round75-source-gate-adjudication-v1.json",
        "artifact_sha256",
    ),
    "cross_regime": (
        ROOT
        / "docs"
        / "model-research"
        / "cross-regime-edge-acceptance-contract-v1.json",
        "contract_sha256",
    ),
}
STATUS_BY_ROUND = {
    73: "invalidated_before_model",
    74: "terminal_campaign_quota_failed_no_model",
    75: "rejected_incomplete_campaign",
    76: "blocked_by_round75_source_gate",
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(path: Path, digest_field: str) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="ascii"))
    claimed = str(value.pop(digest_field))
    if claimed != _canonical_sha256(value):
        raise ValueError(f"{path.name} canonical digest differs")
    value[digest_field] = claimed
    return value, claimed


def _validate_sources() -> dict[str, dict[str, Any]]:
    loaded = {
        name: _load_bound(path, digest_field)[0]
        for name, (path, digest_field) in SOURCE_SPECS.items()
    }
    if loaded["round73"]["decision"]["round_073_v9_campaign_status"] != (
        "invalidated_before_target_or_model_access"
    ):
        raise ValueError("Round 73 status differs")
    round74_decision = loaded["round74"]["decision"]
    if (
        round74_decision.get("reason") != "training_eligible_anchor_quota_failed"
        or round74_decision.get("status") != "campaign_cannot_qualify_model"
    ):
        raise ValueError("Round 74 status differs")
    if loaded["round75"]["status"] != "rejected_incomplete_campaign" or any(
        loaded["round75"]["gates"].values()
    ):
        raise ValueError("Round 75 terminal gate differs")
    if loaded["round76_preregistration"]["implementation_gate"][
        "implementation_permitted_now"
    ]:
        raise ValueError("Round 76 preregistration unexpectedly permits implementation")
    if loaded["round76_adjudication"]["decision"]["status"] != (
        "blocked_before_implementation_by_round75_source_gate"
    ):
        raise ValueError("Round 76 adjudication differs")
    if loaded["cross_regime"]["claim_boundary"][
        "profitable_in_every_future_market_is_guaranteed"
    ]:
        raise ValueError("cross-regime contract promises future profit")
    return loaded


def _read_progress() -> tuple[list[dict[str, str]], list[str]]:
    with BASE_PROGRESS.open("r", encoding="ascii", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    if [int(row["round"]) for row in rows] != list(range(1, 73)):
        raise ValueError("base progress must contain exact Rounds 1 through 72")
    return rows, fields


def _append_status_rows(
    rows: list[dict[str, str]], fields: Sequence[str]
) -> list[dict[str, str]]:
    values = {
        73: {
            "stage": "prospective event-corpus capture",
            "periods": "2026-07-23 prospective capture",
            "feature_set": "target-free BTC ETH SOL event corpus",
            "risk_level": "source qualification only; no model or P&L",
            "source_file": "round-073-v9-corpus-invalidation-2026-07-25.json",
        },
        74: {
            "stage": "segmented event-sequence campaign",
            "periods": "2026-07-28..2026-08-10 prospective campaign",
            "feature_set": "causal multi-lane event sequence",
            "risk_level": "source population gate only; no model or P&L",
            "source_file": "round-074-terminal-campaign-outcome-v2-2026-08-10.json",
        },
        75: {
            "stage": "continuous event-sequence replacement campaign",
            "periods": "2026-08-11..2026-08-23 prospective campaign",
            "feature_set": "causal one-second BTC ETH SOL event capture",
            "risk_level": "source continuity gate only; no model or P&L",
            "source_file": "round-075-terminal-campaign-audit-2026-08-23.json",
        },
        76: {
            "stage": "mark-conditioned censored-duration pretraining",
            "periods": "not run",
            "feature_set": "preregistered candidate; no implementation or training",
            "risk_level": "blocked by rejected Round 75 source campaign",
            "source_file": "round-076-round75-source-gate-adjudication-v1.json",
        },
    }
    for round_number in range(73, LATEST_ROUND + 1):
        row = {field: "" for field in fields}
        row.update(
            {
                "round": str(round_number),
                "selection_contaminated": "False",
                "selected_signals": "0",
                "executable_trades": "0",
                "status": STATUS_BY_ROUND[round_number],
                "development_consumed": "False",
                **values[round_number],
            }
        )
        rows.append(row)
    if [int(row["round"]) for row in rows] != list(range(1, LATEST_ROUND + 1)):
        raise ValueError("published research status is incomplete")
    return rows


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="ascii", newline="\n")
    temporary.replace(path)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _x(round_number: int, *, start: int, end: int, left: float, right: float) -> float:
    return left + (round_number - start) * (right - left) / (end - start)


def _progress_svg(rows: Sequence[Mapping[str, str]]) -> str:
    left, right = 88.0, 1140.0
    points = [
        (int(row["round"]), 100.0 * float(row["spearman_ic"]))
        for row in rows
        if row.get("spearman_ic")
    ]
    y_values = [value for _, value in points] + [0.0]
    y_min, y_max = min(y_values) - 1.0, max(y_values) + 1.0

    def metric_x(round_number: int) -> float:
        return _x(round_number, start=1, end=LATEST_ROUND, left=left, right=right)

    def metric_y(value: float) -> float:
        return 110.0 + (y_max - value) * 250.0 / (y_max - y_min)

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img">',
        "<title>Action-value research status through Round 76</title>",
        "<desc>Comparable recorded Spearman statistics and source-bound gate outcomes. Rounds 73 through 76 produced no model or profitability result.</desc>",
        '<rect width="1200" height="760" fill="#0b1220"/>',
        '<text x="60" y="50" fill="#f8fafc" font-family="Segoe UI,Arial,sans-serif" font-size="27" font-weight="700">Action-value research status through Round 76</text>',
        '<text x="60" y="78" fill="#cbd5e1" font-family="Segoe UI,Arial,sans-serif" font-size="15">Only comparable recorded statistics are plotted. Later source-gate rounds add no model, trade, ROI, or profitability metric.</text>',
    ]
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4.0
        yy = metric_y(value)
        lines.extend(
            [
                f'<line x1="{left:.1f}" y1="{yy:.1f}" x2="{right:.1f}" y2="{yy:.1f}" stroke="#334155"/>',
                f'<text x="{left - 12:.1f}" y="{yy + 5:.1f}" text-anchor="end" fill="#cbd5e1" font-family="Segoe UI,Arial,sans-serif" font-size="12">{value:.1f}</text>',
            ]
        )
    segments: list[list[tuple[int, float]]] = []
    for round_number, value in points:
        if not segments or round_number != segments[-1][-1][0] + 1:
            segments.append([])
        segments[-1].append((round_number, value))
    for segment in segments:
        if len(segment) < 2:
            continue
        path = " ".join(
            ("M" if index == 0 else "L")
            + f" {metric_x(round_number):.1f} {metric_y(value):.1f}"
            for index, (round_number, value) in enumerate(segment)
        )
        lines.append(
            f'<path d="{path}" fill="none" stroke="#60a5fa" stroke-width="3" stroke-linejoin="round"/>'
        )
    for round_number, value in points:
        lines.append(
            f'<circle cx="{metric_x(round_number):.1f}" cy="{metric_y(value):.1f}" r="3" fill="#60a5fa"/>'
        )
    lines.append(
        '<text transform="translate(24 235) rotate(-90)" text-anchor="middle" fill="#cbd5e1" font-family="Segoe UI,Arial,sans-serif" font-size="13">Recorded Spearman x 100</text>'
    )
    timeline_y = 470.0
    lines.extend(
        [
            '<text x="60" y="420" fill="#f8fafc" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="600">Rounds 54-76: frozen-gate outcome</text>',
            f'<line x1="{_x(54, start=54, end=LATEST_ROUND, left=left, right=right):.1f}" y1="{timeline_y:.1f}" x2="{right:.1f}" y2="{timeline_y:.1f}" stroke="#334155" stroke-width="3"/>',
        ]
    )
    for row in rows:
        round_number = int(row["round"])
        if round_number < 54:
            continue
        status = row.get("status", "")
        color = (
            "#22c55e"
            if "passed" in status
            else "#ef4444"
            if any(
                word in status
                for word in ("rejected", "failed", "invalidated", "blocked")
            )
            else "#94a3b8"
        )
        xx = _x(round_number, start=54, end=LATEST_ROUND, left=left, right=right)
        lines.extend(
            [
                f'<circle cx="{xx:.1f}" cy="{timeline_y:.1f}" r="8" fill="{color}"/>',
                f'<text x="{xx:.1f}" y="{timeline_y + 27:.1f}" text-anchor="middle" fill="#cbd5e1" font-family="Segoe UI,Arial,sans-serif" font-size="11">{round_number}</text>',
            ]
        )
    labels = (
        (72, "predictive candidate rejected"),
        (73, "corpus invalidated before model"),
        (74, "training population quota failed"),
        (75, "replacement campaign incomplete"),
        (76, "candidate blocked before implementation"),
    )
    for index, (round_number, label) in enumerate(labels):
        column, row_number = index % 2, index // 2
        xx, yy = 90 + column * 550, 555 + row_number * 48
        lines.extend(
            [
                f'<circle cx="{xx}" cy="{yy - 5}" r="6" fill="#ef4444"/>',
                f'<text x="{xx + 15}" y="{yy}" fill="#f8fafc" font-family="Segoe UI,Arial,sans-serif" font-size="14"><tspan font-weight="700">R{round_number}</tspan><tspan dx="5" fill="#cbd5e1">{html.escape(label)}</tspan></text>',
            ]
        )
    lines.extend(
        [
            '<text x="60" y="725" fill="#cbd5e1" font-family="Segoe UI,Arial,sans-serif" font-size="13">Unsupported bull, bear, sideways, choppy, volatility, liquidity, or latency slices require rejection or abstention; no future no-loss claim is valid.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _readme(status_sha256: str) -> str:
    return f"""# Current Action-Value Research Status

> **No accepted market edge or trading authority.** Round 72 remains the latest completed predictive model evaluation and was rejected. Rounds 73-76 produced no model, backtest, trade, ROI, or profitability result.

| Round | Outcome | What it means |
| ---: | --- | --- |
| 72 | Predictive candidate rejected | No symbol-horizon component passed the frozen gate. |
| 73 | Corpus invalidated | Stopped before target or model access. |
| 74 | Training population quota failed | No model was trained. |
| 75 | Replacement campaign incomplete | Train/tune/test and source-continuity gates failed. |
| 76 | Blocked before implementation | Its required Round 75 source population is inadmissible. |

[Progress graph](research-progress.svg) | [Canonical progress data](progress.csv) | [Machine status](status.json) | [Round 72 metrics and graphs](../latest/README.md)

Every future candidate must pass separate after-cost bull, bear, sideways, choppy, volatility, liquidity, and latency slices. An unsupported slice means no new exposure. This is a rejection/abstention rule, not a promise that a strategy can profit or avoid every loss in every future market.

Status SHA-256: `{status_sha256}`
"""


def publish() -> dict[str, Any]:
    _validate_sources()
    rows, fields = _read_progress()
    rows = _append_status_rows(rows, fields)
    publisher_path = Path(__file__).resolve()
    source_records = {}
    for name, (path, digest_field) in SOURCE_SPECS.items():
        value = json.loads(path.read_text(encoding="ascii"))
        source_records[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "file_sha256": _file_sha256(path),
            digest_field: value[digest_field],
        }
    status: dict[str, Any] = {
        "schema_version": "action-value-latest-status-v1",
        "current_round": LATEST_ROUND,
        "latest_completed_model_evaluation": {
            "round": 72,
            "status": "rejected",
            "accepted_market_edge": False,
        },
        "later_round_status": {
            str(round_number): status_value
            for round_number, status_value in STATUS_BY_ROUND.items()
        },
        "claims": {
            "ai_uplift_established": False,
            "financial_edge_established": False,
            "profitability_established": False,
            "trading_authority": False,
        },
        "cross_regime_policy": {
            "future_profit_or_no_loss_guaranteed": False,
            "unsupported_slice_action": "abstain_from_new_exposure",
        },
        "publisher": {
            "path": publisher_path.relative_to(ROOT).as_posix(),
            "file_sha256": _file_sha256(publisher_path),
        },
        "sources": source_records,
    }
    status["status_sha256"] = _canonical_sha256(status)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT / "progress.csv", rows, fields)
    _write_text(OUTPUT / "research-progress.svg", _progress_svg(rows))
    _write_text(
        OUTPUT / "status.json",
        json.dumps(status, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    _write_text(OUTPUT / "README.md", _readme(status["status_sha256"]))
    return status


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=True, sort_keys=True))
