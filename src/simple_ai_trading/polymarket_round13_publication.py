"""Deterministic, table-backed publication for Polymarket Round 13."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
from io import StringIO
import json
import math
import os
from pathlib import Path
import shutil

from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_round13 import polymarket_round13_scenarios
from .polymarket_round13_evaluation import (
    POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION,
    load_round13_evaluation_report,
)


POLYMARKET_ROUND13_PUBLICATION_SCHEMA_VERSION = (
    "polymarket-round13-latest-publication-v1"
)

_ASSETS = ("BTC", "ETH", "SOL")
_POLICIES = ("calibrated", "raw_market_prior")
_COLORS = {
    "ink": "#111827",
    "muted": "#475569",
    "grid": "#CBD5E1",
    "background": "#F8FAFC",
    "treatment": "#0F766E",
    "control": "#4F46E5",
    "positive": "#15803D",
    "negative": "#B42318",
    "warning": "#B45309",
    "btc": "#D97706",
    "eth": "#2563EB",
    "sol": "#7C3AED",
}
_PROGRESS_COLUMNS = (
    "round",
    "action",
    "status",
    "independent_groups",
    "conditions",
    "selected_filled_conditions",
    "total_utility_quote",
    "maximum_drawdown_quote",
    "bootstrap_lower_mean_group_utility_quote",
    "confirmation_passed",
    "profitability_claim",
    "source_report_sha256",
)


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 13 {name} is not an object")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _rows(value: object, *, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"Round 13 {name} is not an object list")
    return list(value)


def _utc(milliseconds: int) -> str:
    instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        milliseconds=int(milliseconds)
    )
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Round 13 CSV contains a non-finite value")
        return format(value, ".17g")
    return value


def _render_csv(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return output.getvalue()


def _validated_report(report: Mapping[str, object]) -> Mapping[str, object]:
    scenarios = _mapping(report.get("scenarios"), name="scenario report")
    expected_scenarios = tuple(item.name for item in polymarket_round13_scenarios())
    scenario_config = _rows(
        report.get("execution_scenarios"), name="execution scenarios"
    )
    if (
        report.get("schema_version") != POLYMARKET_ROUND13_EVALUATION_SCHEMA_VERSION
        or report.get("round") != 13
        or set(scenarios) != set(expected_scenarios)
        or len(scenarios) != len(expected_scenarios)
        or scenario_config != [item.asdict() for item in polymarket_round13_scenarios()]
        or report.get("profitability_claim") is not False
        or report.get("paper_authority") is not False
        or report.get("live_trading_authority") is not False
        or report.get("annualized_roi_available") is not False
    ):
        raise ValueError("Round 13 publication truth contract differs")
    for name in expected_scenarios:
        scenario = _mapping(scenarios[name], name=f"scenario {name}")
        for key in ("calibrated_treatment", "raw_market_prior_control"):
            metrics = _mapping(scenario.get(key), name=f"{name} {key}")
            if metrics.get("scenario") != name:
                raise ValueError("Round 13 scenario metric identity differs")
    return report


def _verified_progress_history(latest: Path) -> list[dict[str, object]]:
    table = latest / "tables" / "optimization-progress.csv"
    manifest_path = latest / "publication-integrity.json"
    if not table.is_file() or not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "existing optimization history manifest is unreadable"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("existing optimization history manifest is invalid")
    claimed_field = (
        "publication_sha256" if "publication_sha256" in manifest else "manifest_sha256"
    )
    unhashed = dict(manifest)
    claimed = unhashed.pop(claimed_field, None)
    if claimed != _canonical_sha256(unhashed):
        raise ValueError("existing optimization history manifest hash differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("existing optimization history artifacts are unavailable")
    matching = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and str(item.get("path") or "").endswith(
            "latest/tables/optimization-progress.csv"
        )
    ]
    if len(matching) != 1 or matching[0].get("sha256") != _file_sha256(table):
        raise ValueError("existing optimization history table hash differs")
    with table.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows or any(not str(row.get("round") or "").isdigit() for row in rows):
        raise ValueError("existing optimization history rows are invalid")
    rounds = [int(str(row["round"])) for row in rows if int(str(row["round"])) < 13]
    if len(rounds) != len(set(rounds)):
        raise ValueError("existing optimization history has duplicate rounds")
    return [
        {column: row.get(column, "") for column in _PROGRESS_COLUMNS}
        for row in rows
        if int(str(row["round"])) < 13
    ]


def _publication_rows(
    report: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Sequence[str], list[dict[str, object]]]]:
    report_sha = str(report["report_sha256"])
    contract_sha = str(report["contract_sha256"])
    run_id = str(report["run_id"])
    span = _mapping(report["utc_span_ms"], name="UTC span")
    start_ms, end_ms = int(span["start"]), int(span["end"])
    scenario_config = {
        str(item["name"]): item
        for item in _rows(report["execution_scenarios"], name="execution scenarios")
    }
    scenarios = _mapping(report["scenarios"], name="scenario report")
    common = {
        "report_sha256": report_sha,
        "contract_sha256": contract_sha,
        "run_id": run_id,
        "start_utc": _utc(start_ms),
        "end_utc": _utc(end_ms),
        "assets": "/".join(_ASSETS),
    }
    summary: list[dict[str, object]] = []
    equity: list[dict[str, object]] = []
    per_asset: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    admission: list[dict[str, object]] = []
    for scenario_name in (item.name for item in polymarket_round13_scenarios()):
        scenario_value = scenarios[scenario_name]
        scenario = _mapping(scenario_value, name=f"scenario {scenario_name}")
        config = scenario_config[str(scenario_name)]
        for policy, report_key in (
            ("calibrated", "calibrated_treatment"),
            ("raw_market_prior", "raw_market_prior_control"),
        ):
            metrics = _mapping(scenario[report_key], name=f"{scenario_name} {policy}")
            bootstrap = _mapping(metrics["bootstrap"], name="bootstrap")
            comparison = (
                _mapping(metrics["control_comparison"], name="control comparison")
                if policy == "calibrated"
                else None
            )
            comparison_bootstrap = (
                _mapping(
                    comparison["treatment_minus_control_bootstrap"],
                    name="control bootstrap",
                )
                if comparison is not None
                else None
            )
            summary.append(
                {
                    **common,
                    "scenario": scenario_name,
                    "policy": policy,
                    "submission_latency_ms": config["submission_latency_ms"],
                    "fee_multiplier": config["fee_multiplier"],
                    "adverse_ticks": config["adverse_ticks"],
                    "displayed_depth_fraction": (
                        float(config["depth_numerator"])
                        / float(config["depth_denominator"])
                    ),
                    "condition_count": metrics["condition_count"],
                    "attempt_count": metrics["attempt_count"],
                    "simulated_filled_conditions": metrics[
                        "simulated_filled_conditions"
                    ],
                    "simulated_no_fill_attempts": metrics["simulated_no_fill_attempts"],
                    "unknown_after_submit_conditions": metrics[
                        "unknown_after_submit_conditions"
                    ],
                    "wins": metrics["wins"],
                    "losses": metrics["losses"],
                    "total_utility_quote": metrics["total_utility_quote"],
                    "mean_condition_utility_quote": metrics[
                        "mean_condition_utility_quote"
                    ],
                    "median_condition_utility_quote": metrics[
                        "median_condition_utility_quote"
                    ],
                    "median_simulated_filled_condition_utility_quote": metrics[
                        "median_simulated_filled_condition_utility_quote"
                    ],
                    "maximum_group_entry_exposure_quote": metrics[
                        "maximum_group_entry_exposure_quote"
                    ],
                    "capital_deployed_quote": metrics["capital_deployed_quote"],
                    "market_horizon_capital_time_quote_seconds": metrics[
                        "market_horizon_capital_time_quote_seconds"
                    ],
                    "turnover_quote": metrics["turnover_quote"],
                    "maximum_drawdown_quote": metrics["maximum_drawdown_quote"],
                    "drawdown_limit_quote": metrics["drawdown_limit_quote"],
                    "bootstrap_lower_mean_group_utility_quote": bootstrap[
                        "lower_95_mean_group_utility_quote"
                    ],
                    "bootstrap_median_mean_group_utility_quote": bootstrap[
                        "median_mean_group_utility_quote"
                    ],
                    "bootstrap_upper_mean_group_utility_quote": bootstrap[
                        "upper_95_mean_group_utility_quote"
                    ],
                    "treatment_minus_control_total_utility_quote": (
                        None
                        if comparison is None
                        else comparison["treatment_minus_control_total_utility_quote"]
                    ),
                    "non_tied_treatment_control_conditions": (
                        None
                        if comparison is None
                        else comparison["non_tied_condition_count"]
                    ),
                    "treatment_control_bootstrap_lower_mean_group_utility_quote": (
                        None
                        if comparison_bootstrap is None
                        else comparison_bootstrap["lower_95_mean_group_utility_quote"]
                    ),
                    "treatment_control_bootstrap_median_mean_group_utility_quote": (
                        None
                        if comparison_bootstrap is None
                        else comparison_bootstrap["median_mean_group_utility_quote"]
                    ),
                    "treatment_control_bootstrap_upper_mean_group_utility_quote": (
                        None
                        if comparison_bootstrap is None
                        else comparison_bootstrap["upper_95_mean_group_utility_quote"]
                    ),
                    "gate_passed": metrics.get("gate_passed", False),
                    "gate_reasons": "|".join(
                        str(value)
                        for value in metrics.get(
                            "gate_reasons", metrics["gate_reasons_without_control"]
                        )
                    ),
                }
            )
            for row in _rows(metrics["equity"], name="equity rows"):
                equity.append(
                    {
                        **common,
                        "scenario": scenario_name,
                        "policy": policy,
                        "event_start_ms": row["event_start_ms"],
                        "event_start_utc": _utc(int(row["event_start_ms"])),
                        "group_utility_quote": row["group_utility_quote"],
                        "cumulative_utility_quote": row["cumulative_utility_quote"],
                        "drawdown_quote": row["drawdown_quote"],
                    }
                )
            utilities = _mapping(
                metrics["per_asset_utility_quote"], name="asset utility"
            )
            fills = _mapping(
                metrics["simulated_fills_per_asset"], name="simulated asset fills"
            )
            for asset in _ASSETS:
                per_asset.append(
                    {
                        **common,
                        "scenario": scenario_name,
                        "policy": policy,
                        "asset": asset,
                        "simulated_filled_conditions": fills[asset],
                        "utility_quote": utilities[asset],
                    }
                )
            for state, count in _mapping(
                metrics["attempt_states"], name="attempt states"
            ).items():
                admission.append(
                    {
                        **common,
                        "scenario": scenario_name,
                        "policy": policy,
                        "category": "attempt_state",
                        "reason": state,
                        "count": count,
                    }
                )
            for reason, count in _mapping(
                metrics["abstentions_by_reason"], name="abstentions"
            ).items():
                admission.append(
                    {
                        **common,
                        "scenario": scenario_name,
                        "policy": policy,
                        "category": "abstention",
                        "reason": reason,
                        "count": count,
                    }
                )
        treatment = _mapping(
            scenario["calibrated_treatment"], name="calibrated treatment"
        )
        comparison = _mapping(
            treatment["control_comparison"], name="control comparison"
        )
        for row in _rows(comparison["per_condition"], name="paired conditions"):
            paired.append(
                {
                    **common,
                    "scenario": scenario_name,
                    "condition_id": row["condition_id"],
                    "asset": row["asset"],
                    "event_start_ms": row["event_start_ms"],
                    "event_start_utc": _utc(int(row["event_start_ms"])),
                    "treatment_utility_quote": row["treatment_utility_quote"],
                    "control_utility_quote": row["control_utility_quote"],
                    "difference_quote": row["difference_quote"],
                }
            )

    reliability: list[dict[str, object]] = []
    proper = _mapping(report["proper_scores"], name="proper scores")
    for policy in _POLICIES:
        policy_scores = _mapping(proper[policy], name=f"{policy} scores")
        scopes = {"pooled": policy_scores["pooled"]}
        scopes.update(_mapping(policy_scores["per_asset"], name="per-asset scores"))
        for scope, score_value in scopes.items():
            score = _mapping(score_value, name=f"{scope} score")
            calibration = _mapping(score["calibration"], name="calibration")
            for row in _rows(score["reliability_bins"], name="reliability bins"):
                reliability.append(
                    {
                        **common,
                        "policy": policy,
                        "scope": scope,
                        "count": score["count"],
                        "log_loss": score["log_loss"],
                        "brier_score": score["brier_score"],
                        "calibration_available": calibration["available"],
                        "calibration_intercept": calibration["intercept"],
                        "calibration_slope": calibration["slope"],
                        "bin": row["bin"],
                        "lower_probability": row["lower_probability"],
                        "upper_probability": row["upper_probability"],
                        "bin_count": row["count"],
                        "mean_probability": row["mean_probability"],
                        "observed_frequency": row["observed_frequency"],
                    }
                )

    primary = _mapping(
        _mapping(scenarios["primary"], name="primary scenario")["calibrated_treatment"],
        name="primary treatment",
    )
    primary_bootstrap = _mapping(primary["bootstrap"], name="primary bootstrap")
    progress = [dict(row) for row in history]
    progress.append(
        {
            "round": 13,
            "action": "frozen calibrated settlement hold",
            "status": (
                "confirmation passed; no trading authority"
                if report["confirmation_passed"]
                else "confirmation gate failed"
            ),
            "independent_groups": _mapping(report["data"], name="data")[
                "independent_synchronized_groups"
            ],
            "conditions": primary["condition_count"],
            "selected_filled_conditions": primary["simulated_filled_conditions"],
            "total_utility_quote": primary["total_utility_quote"],
            "maximum_drawdown_quote": primary["maximum_drawdown_quote"],
            "bootstrap_lower_mean_group_utility_quote": primary_bootstrap[
                "lower_95_mean_group_utility_quote"
            ],
            "confirmation_passed": report["confirmation_passed"],
            "profitability_claim": False,
            "source_report_sha256": report_sha,
        }
    )
    tables = {
        "round13-scenario-summary.csv": summary,
        "round13-equity.csv": equity,
        "round13-per-asset.csv": per_asset,
        "round13-reliability.csv": reliability,
        "round13-treatment-control.csv": paired,
        "round13-admission-states.csv": admission,
        "optimization-progress.csv": progress,
    }
    return {
        name: (tuple(rows[0].keys()) if rows else (), rows)
        if name != "optimization-progress.csv"
        else (_PROGRESS_COLUMNS, rows)
        for name, rows in tables.items()
    }


def _svg_start(
    width: int, height: int, title: str, subtitle: str, description: str
) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        f'<rect width="{width}" height="{height}" fill="{_COLORS["background"]}"/>',
        "<style>text{font-family:Segoe UI,Arial,sans-serif;letter-spacing:0}.title{font-size:27px;font-weight:700}.sub{font-size:14px}.axis{font-size:12px}.label{font-size:13px;font-weight:600}</style>",
        f'<text class="title" x="48" y="52" fill="{_COLORS["ink"]}">{escape(title)}</text>',
        f'<text class="sub" x="48" y="80" fill="{_COLORS["muted"]}">{escape(subtitle)}</text>',
    ]


def _line_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    subtitle: str,
) -> str:
    width, height = 1320, 650
    left, right, top, bottom = 105, 1260, 125, 530
    values = [float(row["cumulative_utility_quote"]) for row in rows] + [
        -float(row["drawdown_quote"]) for row in rows
    ]
    lower, upper = min(0.0, min(values)), max(0.0, max(values))
    pad = max(0.5, (upper - lower) * 0.1)
    lower, upper = lower - pad, upper + pad

    def point(index: int, value: float) -> tuple[float, float]:
        x = left + (right - left) * index / max(1, len(rows) - 1)
        y = bottom - (bottom - top) * (value - lower) / (upper - lower)
        return x, y

    body = _svg_start(
        width,
        height,
        title,
        subtitle,
        "Chronological cumulative after-cost utility and drawdown from the exact equity table.",
    )
    for index in range(6):
        value = lower + (upper - lower) * index / 5
        y = point(0, value)[1]
        body.extend(
            (
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{_COLORS["grid"]}"/>',
                f'<text class="axis" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" fill="{_COLORS["muted"]}">{value:.3g}</text>',
            )
        )
    for key, sign, color in (
        ("cumulative_utility_quote", 1.0, _COLORS["treatment"]),
        ("drawdown_quote", -1.0, _COLORS["negative"]),
    ):
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (
                point(index, sign * float(row[key])) for index, row in enumerate(rows)
            )
        )
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
    tick_indexes = sorted({round(index * (len(rows) - 1) / 5) for index in range(6)})
    for index in tick_indexes:
        x = point(index, 0)[0]
        label = str(rows[index]["event_start_utc"])[5:16].replace("T", " ")
        body.append(
            f'<text class="axis" x="{x:.2f}" y="{bottom + 28}" text-anchor="middle" fill="{_COLORS["muted"]}">{escape(label)}</text>'
        )
    body.extend(
        (
            f'<line x1="430" y1="585" x2="470" y2="585" stroke="{_COLORS["treatment"]}" stroke-width="3"/><text class="sub" x="480" y="590" fill="{_COLORS["muted"]}">Cumulative utility</text>',
            f'<line x1="700" y1="585" x2="740" y2="585" stroke="{_COLORS["negative"]}" stroke-width="3"/><text class="sub" x="750" y="590" fill="{_COLORS["muted"]}">Drawdown below zero</text>',
            "</svg>",
        )
    )
    return "\n".join(body) + "\n"


def _bar_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    title: str,
    subtitle: str,
    category_key: str,
    value_key: str,
    color_key: str | None = None,
) -> str:
    width, height = 1320, 590
    left, right, top, bottom = 105, 1260, 125, 485
    values = [float(row[value_key]) for row in rows]
    lower, upper = min(0.0, min(values)), max(0.0, max(values))
    pad = max(0.5, (upper - lower) * 0.12)
    lower, upper = lower - pad, upper + pad

    def y(value: float) -> float:
        return bottom - (bottom - top) * (value - lower) / (upper - lower)

    body = _svg_start(
        width,
        height,
        title,
        subtitle,
        "Bar values are generated directly from the corresponding publication table.",
    )
    zero = y(0.0)
    body.append(
        f'<line x1="{left}" y1="{zero:.2f}" x2="{right}" y2="{zero:.2f}" stroke="{_COLORS["ink"]}"/>'
    )
    slot = (right - left) / max(1, len(rows))
    bar_width = min(92.0, slot * 0.62)
    for index, row in enumerate(rows):
        value = float(row[value_key])
        x = left + slot * (index + 0.5) - bar_width / 2
        top_y, bottom_y = min(zero, y(value)), max(zero, y(value))
        color_name = str(row[color_key]) if color_key else ""
        color = _COLORS.get(color_name.lower(), _COLORS["treatment"])
        body.extend(
            (
                f'<rect x="{x:.2f}" y="{top_y:.2f}" width="{bar_width:.2f}" height="{max(1.0, bottom_y - top_y):.2f}" rx="3" fill="{color}"/>',
                f'<text class="label" x="{x + bar_width / 2:.2f}" y="{top_y - 8:.2f}" text-anchor="middle" fill="{_COLORS["ink"]}">{value:.3g}</text>',
                f'<text class="axis" x="{x + bar_width / 2:.2f}" y="{bottom + 25}" text-anchor="middle" fill="{_COLORS["muted"]}">{escape(str(row[category_key]))}</text>',
            )
        )
    body.append("</svg>")
    return "\n".join(body) + "\n"


def _reliability_chart(rows: Sequence[Mapping[str, object]], subtitle: str) -> str:
    selected = [
        row for row in rows if row["scope"] == "pooled" and int(row["bin_count"]) > 0
    ]
    width, height = 900, 700
    left, right, top, bottom = 110, 830, 125, 605

    def point(x: float, y: float) -> tuple[float, float]:
        return left + (right - left) * x, bottom - (bottom - top) * y

    body = _svg_start(
        width,
        height,
        "Round 13 pooled reliability",
        subtitle,
        "Observed outcome frequency versus mean forecast probability in nonempty decile bins.",
    )
    for index in range(6):
        value = index / 5
        x, y = point(value, value)
        body.extend(
            (
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{_COLORS["grid"]}"/>',
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="{_COLORS["grid"]}"/>',
                f'<text class="axis" x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" fill="{_COLORS["muted"]}">{value:.1f}</text>',
                f'<text class="axis" x="{x:.2f}" y="{bottom + 24}" text-anchor="middle" fill="{_COLORS["muted"]}">{value:.1f}</text>',
            )
        )
    x1, y1 = point(0, 0)
    x2, y2 = point(1, 1)
    body.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{_COLORS["muted"]}" stroke-dasharray="7 6"/>'
    )
    for policy, color in (
        ("calibrated", _COLORS["treatment"]),
        ("raw_market_prior", _COLORS["control"]),
    ):
        policy_rows = [row for row in selected if row["policy"] == policy]
        points = []
        for row in policy_rows:
            x, y = point(
                float(row["mean_probability"]), float(row["observed_frequency"])
            )
            points.append((x, y))
            radius = 4 + min(8, math.sqrt(int(row["bin_count"])))
            body.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.82"/>'
            )
        if points:
            body.append(
                f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
    body.extend(
        (
            f'<text class="sub" x="{left}" y="670" fill="{_COLORS["muted"]}">x: mean forecast probability   y: observed Up frequency   point area: bin count</text>',
            "</svg>",
        )
    )
    return "\n".join(body) + "\n"


def _comparison_chart(summary: Sequence[Mapping[str, object]], subtitle: str) -> str:
    treatment = [row for row in summary if row["policy"] == "calibrated"]
    rows = [
        {
            "scenario": row["scenario"],
            "utility": row["total_utility_quote"],
            "drawdown": row["maximum_drawdown_quote"],
            "passed": row["gate_passed"],
        }
        for row in treatment
    ]
    width, height = 1320, 650
    left, top = 360, 130
    maximum = max(
        1.0,
        max(abs(float(row["utility"])) for row in rows),
        max(float(row["drawdown"]) for row in rows),
    )
    body = _svg_start(
        width,
        height,
        "Round 13 execution stress-test acceptance criteria",
        subtitle,
        "Treatment total utility and maximum drawdown for every preregistered execution scenario.",
    )
    for index, row in enumerate(rows):
        y = top + index * 67
        utility = float(row["utility"])
        drawdown = float(row["drawdown"])
        utility_width = 650 * abs(utility) / maximum
        drawdown_width = 650 * drawdown / maximum
        body.extend(
            (
                f'<text class="label" x="48" y="{y + 18}" fill="{_COLORS["ink"]}">{escape(str(row["scenario"]))}</text>',
                f'<rect x="{left}" y="{y}" width="650" height="18" rx="3" fill="#E2E8F0"/>',
                f'<rect x="{left}" y="{y}" width="{utility_width:.2f}" height="18" rx="3" fill="{_COLORS["positive"] if utility > 0 else _COLORS["negative"]}"/>',
                f'<text class="axis" x="{left + 665}" y="{y + 14}" fill="{_COLORS["muted"]}">utility {utility:.4g}</text>',
                f'<rect x="{left}" y="{y + 25}" width="650" height="10" rx="2" fill="#E2E8F0"/>',
                f'<rect x="{left}" y="{y + 25}" width="{drawdown_width:.2f}" height="10" rx="2" fill="{_COLORS["warning"]}"/>',
                f'<text class="axis" x="{left + 665}" y="{y + 35}" fill="{_COLORS["muted"]}">drawdown {drawdown:.4g}</text>',
            )
        )
    body.append("</svg>")
    return "\n".join(body) + "\n"


def _control_interval_chart(
    summary: Sequence[Mapping[str, object]], subtitle: str
) -> str:
    rows = [row for row in summary if row["policy"] == "calibrated"]
    intervals = [
        (
            float(row["treatment_control_bootstrap_lower_mean_group_utility_quote"]),
            float(row["treatment_control_bootstrap_median_mean_group_utility_quote"]),
            float(row["treatment_control_bootstrap_upper_mean_group_utility_quote"]),
        )
        for row in rows
    ]
    width, height = 1320, 640
    left, right, top = 335, 1000, 145
    lower = min(0.0, min(value[0] for value in intervals))
    upper = max(0.0, max(value[2] for value in intervals))
    pad = max(0.05, (upper - lower) * 0.12)
    lower, upper = lower - pad, upper + pad

    def x(value: float) -> float:
        return left + (right - left) * (value - lower) / (upper - lower)

    body = _svg_start(
        width,
        height,
        "Round 13 calibrated minus raw-prior control",
        subtitle,
        "Deterministic moving-block bootstrap interval for paired mean utility per synchronized event group.",
    )
    zero = x(0.0)
    body.append(
        f'<line x1="{zero:.2f}" y1="115" x2="{zero:.2f}" y2="560" stroke="{_COLORS["negative"]}" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    for index, (row, interval) in enumerate(zip(rows, intervals, strict=True)):
        low, median, high = interval
        y = top + index * 61
        body.extend(
            (
                f'<text class="label" x="48" y="{y + 5}" fill="{_COLORS["ink"]}">{escape(str(row["scenario"]))}</text>',
                f'<line x1="{x(low):.2f}" y1="{y}" x2="{x(high):.2f}" y2="{y}" stroke="{_COLORS["warning"]}" stroke-width="10" stroke-linecap="round"/>',
                f'<circle cx="{x(median):.2f}" cy="{y}" r="8" fill="{_COLORS["treatment"]}" stroke="#FFFFFF" stroke-width="2"/>',
                f'<text class="axis" x="1240" y="{y + 5}" text-anchor="end" fill="{_COLORS["muted"]}">[{low:.4g}, {median:.4g}, {high:.4g}]</text>',
            )
        )
    body.extend(
        (
            f'<text class="axis" x="{left}" y="602" fill="{_COLORS["muted"]}">Whisker: 95% interval   dot: median bootstrap mean   dashed line: zero</text>',
            "</svg>",
        )
    )
    return "\n".join(body) + "\n"


def _admission_chart(rows: Sequence[Mapping[str, object]], subtitle: str) -> str:
    primary = [
        row
        for row in rows
        if row["scenario"] == "primary" and row["policy"] == "calibrated"
    ]
    aggregated: dict[str, int] = {}
    for row in primary:
        key = f"{row['category']}:{row['reason']}"
        aggregated[key] = aggregated.get(key, 0) + int(row["count"])
    selected = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))[:10]
    width, height = 1320, 650
    body = _svg_start(
        width,
        height,
        "Round 13 primary admission states",
        subtitle,
        "Ten largest exact attempt-state and abstention counts for the calibrated primary policy.",
    )
    maximum = max((value for _, value in selected), default=1)
    for index, (label, value) in enumerate(selected):
        y = 120 + index * 48
        bar = 780 * value / maximum
        body.extend(
            (
                f'<text class="axis" x="48" y="{y + 19}" fill="{_COLORS["ink"]}">{escape(label)}</text>',
                f'<rect x="410" y="{y}" width="780" height="25" rx="3" fill="#E2E8F0"/>',
                f'<rect x="410" y="{y}" width="{bar:.2f}" height="25" rx="3" fill="{_COLORS["control"]}"/>',
                f'<text class="label" x="1205" y="{y + 19}" fill="{_COLORS["ink"]}">{value:,}</text>',
            )
        )
    body.append("</svg>")
    return "\n".join(body) + "\n"


def _progress_chart(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1320, 180 + 86 * len(rows)
    body = _svg_start(
        width,
        height,
        "Optimization evidence progression",
        "Round-specific evidence; utility and fill counts are not comparable annualized ROI",
        "Historical frozen rounds and the latest confirmation outcome, sourced from the exact progress table.",
    )
    utilities = [
        abs(float(row["total_utility_quote"]))
        for row in rows
        if str(row.get("total_utility_quote") or "")
    ]
    maximum = max(utilities, default=1.0)
    for index, row in enumerate(rows):
        y = 120 + index * 86
        raw = str(row.get("total_utility_quote") or "")
        utility = None if not raw else float(raw)
        bar = 460 * (0.0 if utility is None else abs(utility) / maximum)
        color = (
            _COLORS["muted"]
            if utility is None
            else _COLORS["positive"]
            if utility > 0
            else _COLORS["negative"]
        )
        body.extend(
            (
                f'<text class="label" x="48" y="{y + 19}" fill="{_COLORS["ink"]}">Round {escape(str(row["round"]))}</text>',
                f'<text class="axis" x="135" y="{y + 19}" fill="{_COLORS["muted"]}">{escape(str(row["action"]))}</text>',
                f'<rect x="520" y="{y}" width="460" height="24" rx="3" fill="#E2E8F0"/>',
                f'<rect x="520" y="{y}" width="{bar:.2f}" height="24" rx="3" fill="{color}"/>',
                f'<text class="axis" x="995" y="{y + 18}" fill="{_COLORS["ink"]}">utility {"N/A" if utility is None else format(utility, ".4g")}</text>',
                f'<text class="axis" x="520" y="{y + 49}" fill="{_COLORS["muted"]}">{escape(str(row["status"]))}</text>',
            )
        )
    body.append("</svg>")
    return "\n".join(body) + "\n"


def _remove_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != root.resolve() or not resolved.name.startswith("."):
        raise ValueError("Round 13 publication cleanup path escaped its root")
    if resolved.exists():
        shutil.rmtree(resolved)


def _replace_latest(root: Path, staging: Path, latest: Path) -> Path | None:
    backup = root / ".round13-latest-backup"
    if backup.exists() and not latest.exists():
        os.replace(backup, latest)
    elif backup.exists():
        raise ValueError("Round 13 publication has an ambiguous prior backup")
    if latest.exists():
        os.replace(latest, backup)
    try:
        os.replace(staging, latest)
    except Exception:
        if backup.exists() and not latest.exists():
            os.replace(backup, latest)
        raise
    return backup if backup.exists() else None


@dataclass(frozen=True)
class PolymarketRound13PublicationResult:
    report_sha256: str
    manifest_sha256: str
    generated_files: tuple[str, ...]


def publish_round13_evaluation(
    store: PolymarketEvidenceStore,
    *,
    report_sha256: str,
    research_root: str | Path,
) -> PolymarketRound13PublicationResult:
    """Publish through a staged, crash-recoverable directory replacement."""

    report = _validated_report(
        load_round13_evaluation_report(store, report_sha256=report_sha256)
    )
    root = Path(research_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    history = _verified_progress_history(latest)
    tables = _publication_rows(report, history)
    span = _mapping(report["utc_span_ms"], name="UTC span")
    subtitle = (
        f"BTC / ETH / SOL | {_utc(int(span['start']))} to "
        f"{_utc(int(span['end']))} | real prospective evidence"
    )
    summary_rows = tables["round13-scenario-summary.csv"][1]
    equity_rows = tables["round13-equity.csv"][1]
    primary_equity = [
        row
        for row in equity_rows
        if row["scenario"] == "primary" and row["policy"] == "calibrated"
    ]
    asset_rows = [
        row
        for row in tables["round13-per-asset.csv"][1]
        if row["scenario"] == "primary" and row["policy"] == "calibrated"
    ]
    reliability_rows = tables["round13-reliability.csv"][1]
    admission_rows = tables["round13-admission-states.csv"][1]
    progress_rows = tables["optimization-progress.csv"][1]
    charts = {
        "round13-equity-drawdown.svg": _line_chart(
            primary_equity,
            title="Round 13 primary equity and drawdown",
            subtitle=subtitle,
        ),
        "round13-per-asset-utility.svg": _bar_chart(
            asset_rows,
            title="Round 13 primary utility by asset",
            subtitle=subtitle,
            category_key="asset",
            value_key="utility_quote",
            color_key="asset",
        ),
        "round13-reliability.svg": _reliability_chart(reliability_rows, subtitle),
        "round13-treatment-control.svg": _control_interval_chart(
            summary_rows, subtitle
        ),
        "round13-stress.svg": _comparison_chart(summary_rows, subtitle),
        "round13-admission.svg": _admission_chart(admission_rows, subtitle),
        "optimization-progress.svg": _progress_chart(progress_rows),
    }
    staging = root / f".round13-latest-{report_sha256[:12]}"
    _remove_tree(staging, root)
    (staging / "tables").mkdir(parents=True)
    (staging / "charts").mkdir(parents=True)
    generated_rows: dict[str, int] = {}
    backup: Path | None = None
    swapped = False
    try:
        for name, (columns, rows) in tables.items():
            path = staging / "tables" / name
            path.write_text(_render_csv(columns, rows), encoding="utf-8", newline="\n")
            generated_rows[f"latest/tables/{name}"] = len(rows)
        for name, content in charts.items():
            (staging / "charts" / name).write_text(
                content, encoding="utf-8", newline="\n"
            )
        report_name = "round-013-sealed-confirmation-report.json"
        report_path = staging / report_name
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        confirmation = bool(report["confirmation_passed"])
        primary = _mapping(
            _mapping(
                _mapping(report["scenarios"], name="scenarios")["primary"],
                name="primary scenario",
            )["calibrated_treatment"],
            name="primary treatment",
        )
        readme = f"""# Polymarket research round 13

![Primary equity and drawdown](charts/round13-equity-drawdown.svg)

Round 13 used one sealed prospective BTC/ETH/SOL five-minute capture from
`{_utc(int(span["start"]))}` to `{_utc(int(span["end"]))}`. The primary policy
produced displayed-book simulated fills in
`{primary["simulated_filled_conditions"]}` independent conditions and produced
`{float(primary["total_utility_quote"]):.6g}` quote after modeled entry fees.
The conjunctive confirmation gate **{"passed" if confirmation else "failed"}**.

This remains research evidence, not a profitability or ROI claim. Settlement
overhead and authenticated order lifecycle are unmeasured, so paper and live
trading authority remain disabled.

Use the [scenario table](tables/round13-scenario-summary.csv),
[exact equity rows](tables/round13-equity.csv),
[paired control rows](tables/round13-treatment-control.csv), and
[integrity manifest](publication-integrity.json) to audit every chart.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        artifacts: list[dict[str, object]] = []
        for path in sorted(
            (item for item in staging.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(staging).as_posix(),
        ):
            relative = path.relative_to(staging).as_posix()
            entry: dict[str, object] = {
                "path": f"latest/{relative}",
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            if path.suffix == ".csv":
                entry["row_count"] = generated_rows[f"latest/{relative}"]
            if path.name == report_name:
                entry["source_report_sha256"] = report_sha256
            if path.suffix == ".svg":
                entry["source_tables"] = {
                    "round13-equity-drawdown.svg": ["round13-equity.csv"],
                    "round13-per-asset-utility.svg": ["round13-per-asset.csv"],
                    "round13-reliability.svg": ["round13-reliability.csv"],
                    "round13-treatment-control.svg": [
                        "round13-scenario-summary.csv",
                        "round13-treatment-control.csv",
                    ],
                    "round13-stress.svg": ["round13-scenario-summary.csv"],
                    "round13-admission.svg": ["round13-admission-states.csv"],
                    "optimization-progress.svg": ["optimization-progress.csv"],
                }[path.name]
            artifacts.append(entry)
        manifest_body = {
            "schema_version": POLYMARKET_ROUND13_PUBLICATION_SCHEMA_VERSION,
            "latest_round": 13,
            "source_report_sha256": report_sha256,
            "source_contract_sha256": report["contract_sha256"],
            "source_run_id": report["run_id"],
            "capture_start_utc": _utc(int(span["start"])),
            "capture_end_utc": _utc(int(span["end"])),
            "assets": list(_ASSETS),
            "artifacts": artifacts,
            "confirmation_passed": confirmation,
            "profitability_claim": False,
            "roi_claim": False,
            "paper_authority": False,
            "live_trading_authority": False,
            "manual_chart_edits_permitted": False,
        }
        manifest_sha256 = _canonical_sha256(manifest_body)
        manifest = {**manifest_body, "manifest_sha256": manifest_sha256}
        (staging / "publication-integrity.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        backup = _replace_latest(root, staging, latest)
        swapped = True
        for entry in artifacts:
            path = root / str(entry["path"])
            if not path.is_file() or _file_sha256(path) != entry["sha256"]:
                raise ValueError("Round 13 published artifact verification failed")
        if backup is not None:
            _remove_tree(backup, root)
    except Exception:
        if swapped and latest.exists():
            invalid = root / ".round13-invalid-latest"
            _remove_tree(invalid, root)
            os.replace(latest, invalid)
            if backup is not None and backup.exists():
                os.replace(backup, latest)
            _remove_tree(invalid, root)
        _remove_tree(staging, root)
        raise
    generated = tuple(str(root / str(item["path"])) for item in artifacts) + (
        str(latest / "publication-integrity.json"),
    )
    return PolymarketRound13PublicationResult(
        report_sha256=report_sha256,
        manifest_sha256=manifest_sha256,
        generated_files=generated,
    )


__all__ = [
    "POLYMARKET_ROUND13_PUBLICATION_SCHEMA_VERSION",
    "PolymarketRound13PublicationResult",
    "publish_round13_evaluation",
]
