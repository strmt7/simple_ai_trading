"""Deterministic publication of a validated Polymarket Round 17 result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from datetime import UTC, datetime
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
from typing import Any

from .polymarket_round17_evaluation import (
    POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS,
    validate_round17_one_use_result,
)
from .storage import write_bytes_atomic


POLYMARKET_ROUND17_PUBLICATION_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-publication-v1"
)
_MANIFEST_NAME = "publication-manifest.json"
_PROFILES = ("conservative", "regular", "aggressive")
_SCENARIOS = (
    "primary",
    "latency_250ms",
    "latency_750ms",
    "latency_1000ms",
    "half_depth",
    "quarter_depth",
    "one_tick_adverse",
    "combined",
)
_COLORS = (
    "#006D77",
    "#D1495B",
    "#3A5A40",
    "#7A5195",
    "#E07A5F",
    "#2F6690",
    "#8A6D1D",
    "#555B6E",
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


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 publication input contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 publication input contains {value}")


def load_round17_one_use_result(path: str | Path) -> dict[str, object]:
    """Load and validate one bounded terminal-result JSON artifact."""

    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= 512 * 1024 * 1024
    ):
        raise ValueError("Round 17 terminal result is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 17 terminal result is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 17 terminal result is not an object")
    return validate_round17_one_use_result(value)


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 17 publication {name} is unavailable")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Round 17 publication {name} is non-finite")
    try:
        selected = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Round 17 publication {name} is non-finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"Round 17 publication {name} is non-finite")
    return selected


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"Round 17 publication {name} is invalid")
    return value


def _date(day_start_ms: object) -> str:
    if isinstance(day_start_ms, bool) or not isinstance(day_start_ms, int):
        raise ValueError("Round 17 publication UTC day is invalid")
    return datetime.fromtimestamp(day_start_ms / 1000, tz=UTC).date().isoformat()


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _metric_rows(
    endpoint: Mapping[str, object],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    candidate_id = str(endpoint.get("selected_candidate_id") or "").strip()
    candidate = _object(
        endpoint.get("candidate_metrics"),
        name="candidate metrics",
    )
    controls = _object(endpoint.get("control_metrics"), name="control metrics")
    if (
        not candidate_id
        or candidate_id in controls
        or tuple(controls) != POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
    ):
        raise ValueError("Round 17 publication endpoint identities differ")
    identities = (candidate_id, *POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS)
    metrics_by_id = {
        candidate_id: candidate,
        **{
            control_id: _object(
                controls[control_id],
                name=f"{control_id} metrics",
            )
            for control_id in POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
        },
    }
    rows: list[dict[str, object]] = []
    required = (
        "condition_balanced_log_loss",
        "condition_balanced_brier",
        "condition_weighted_balanced_accuracy",
        "condition_weighted_matthews_correlation",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
    )
    for identity in identities:
        metrics = metrics_by_id[identity]
        parsed = {
            key: _finite(metrics.get(key), name=f"{identity} {key}") for key in required
        }
        if (
            parsed["condition_balanced_log_loss"] < 0
            or not 0 <= parsed["condition_balanced_brier"] <= 1
            or not 0 <= parsed["condition_weighted_balanced_accuracy"] <= 1
            or not -1 <= parsed["condition_weighted_matthews_correlation"] <= 1
            or not 0 <= parsed["expected_calibration_error"] <= 1
        ):
            raise ValueError("Round 17 publication endpoint metric range differs")
        rows.append(
            {
                "series_id": identity,
                "role": "candidate" if identity == candidate_id else "control",
                "log_loss": metrics["condition_balanced_log_loss"],
                "brier_score": metrics["condition_balanced_brier"],
                "balanced_accuracy": metrics["condition_weighted_balanced_accuracy"],
                "matthews_correlation": metrics[
                    "condition_weighted_matthews_correlation"
                ],
                "calibration_intercept": metrics["calibration_intercept"],
                "calibration_slope": metrics["calibration_slope"],
                "expected_calibration_error": metrics["expected_calibration_error"],
            }
        )
    return rows, identities


def _daily_endpoint_rows(
    endpoint: Mapping[str, object],
    identities: tuple[str, ...],
) -> list[dict[str, object]]:
    daily = endpoint.get("daily_metrics")
    if not isinstance(daily, list):
        raise ValueError("Round 17 publication daily endpoint metrics differ")
    rows: list[dict[str, object]] = []
    for item in daily:
        selected = _object(item, name="daily endpoint row")
        candidate = _object(selected["candidate"], name="daily candidate metrics")
        controls = _object(selected["controls"], name="daily control metrics")
        metrics_by_id = {
            identities[0]: candidate,
            **{
                control_id: _object(
                    controls[control_id],
                    name=f"daily {control_id} metrics",
                )
                for control_id in POLYMARKET_ROUND17_ENDPOINT_CONTROL_IDS
            },
        }
        for identity in identities:
            metrics = metrics_by_id[identity]
            rows.append(
                {
                    "date_utc": _date(selected["day_start_ms"]),
                    "day_start_ms": selected["day_start_ms"],
                    "condition_count": selected["condition_count"],
                    "series_id": identity,
                    "role": ("candidate" if identity == identities[0] else "control"),
                    "log_loss": metrics["condition_balanced_log_loss"],
                    "brier_score": metrics["condition_balanced_brier"],
                    "balanced_accuracy": metrics["final_condition_balanced_accuracy"],
                }
            )
    return rows


def _economic_rows(
    economic: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scenario_metrics = _object(
        economic.get("scenario_metrics"),
        name="scenario metrics",
    )
    scenario_gates = _object(
        economic.get("scenario_gates"),
        name="scenario gates",
    )
    scenario_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for profile in _PROFILES:
        by_scenario = _object(
            scenario_metrics[profile],
            name=f"{profile} scenario metrics",
        )
        gates = _object(scenario_gates[profile], name=f"{profile} scenario gates")
        for scenario in _SCENARIOS:
            metrics = _object(
                by_scenario[scenario],
                name=f"{profile} {scenario} metrics",
            )
            condition_count = _integer(
                metrics.get("condition_count"),
                name=f"{profile} {scenario} condition count",
                minimum=1,
            )
            calendar_day_count = _integer(
                metrics.get("calendar_day_count"),
                name=f"{profile} {scenario} calendar day count",
                minimum=1,
            )
            executed = _integer(
                metrics.get("executed_action_count"),
                name=f"{profile} {scenario} action count",
            )
            wins = _integer(
                metrics.get("win_count"),
                name=f"{profile} {scenario} win count",
            )
            win_rate = _finite(
                metrics.get("win_rate"),
                name=f"{profile} {scenario} win rate",
            )
            drawdown = _finite(
                metrics.get("maximum_drawdown_fraction"),
                name=f"{profile} {scenario} maximum drawdown",
            )
            worst_day = _finite(
                metrics.get("worst_daily_loss_fraction"),
                name=f"{profile} {scenario} worst daily loss",
            )
            for field in (
                "net_pnl_quote",
                "mean_event_utility_quote",
                "median_daily_pnl_quote",
                "daily_block_bootstrap_mean_event_utility_lower_95",
            ):
                _finite(
                    metrics.get(field),
                    name=f"{profile} {scenario} {field}",
                )
            profit_factor = metrics.get("profit_factor")
            if (
                profit_factor is not None
                and _finite(
                    profit_factor,
                    name=f"{profile} {scenario} profit factor",
                )
                < 0
            ):
                raise ValueError("Round 17 publication profit factor differs")
            unbounded = metrics.get("profit_factor_unbounded_without_observed_loss")
            gate = gates[scenario]
            if (
                wins > executed
                or not 0 <= win_rate <= 1
                or not 0 <= drawdown <= 1
                or not 0 <= worst_day <= 1
                or not isinstance(unbounded, bool)
                or not isinstance(gate, bool)
                or calendar_day_count > condition_count
            ):
                raise ValueError("Round 17 publication economic metric differs")
            scenario_rows.append(
                {
                    "risk_profile": profile,
                    "scenario": scenario,
                    "condition_count": condition_count,
                    "calendar_day_count": calendar_day_count,
                    "executed_action_count": executed,
                    "win_count": wins,
                    "win_rate": metrics["win_rate"],
                    "net_pnl_quote": metrics["net_pnl_quote"],
                    "mean_event_utility_quote": metrics["mean_event_utility_quote"],
                    "median_daily_pnl_quote": metrics["median_daily_pnl_quote"],
                    "profit_factor": (
                        ""
                        if metrics["profit_factor"] is None
                        else metrics["profit_factor"]
                    ),
                    "profit_factor_unbounded_without_observed_loss": unbounded,
                    "maximum_drawdown_fraction": metrics["maximum_drawdown_fraction"],
                    "worst_daily_loss_fraction": metrics["worst_daily_loss_fraction"],
                    "daily_block_bootstrap_mean_event_utility_lower_95": metrics[
                        "daily_block_bootstrap_mean_event_utility_lower_95"
                    ],
                    "scenario_gate_passed": gate,
                }
            )
            series = metrics["daily_pnl_series"]
            if not isinstance(series, list):
                raise ValueError("Round 17 publication daily equity differs")
            for item in series:
                daily = _object(item, name="daily economic row")
                daily_rows.append(
                    {
                        "date_utc": _date(daily["day_start_ms"]),
                        "day_start_ms": daily["day_start_ms"],
                        "risk_profile": profile,
                        "scenario": scenario,
                        "daily_net_pnl_quote": daily["net_pnl_quote"],
                        "ending_equity_quote": daily["ending_equity_quote"],
                    }
                )
    return scenario_rows, daily_rows


def _gate_rows(
    endpoint: Mapping[str, object],
    economic: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [
        {
            "domain": "endpoint",
            "risk_profile": "",
            "scenario": "",
            "gate": gate,
            "passed": passed,
        }
        for gate, passed in _object(endpoint["gates"], name="endpoint gates").items()
    ]
    scenario_gates = _object(economic["scenario_gates"], name="scenario gates")
    profile_gates = _object(economic["profile_gates"], name="profile gates")
    for profile in _PROFILES:
        gates = _object(scenario_gates[profile], name=f"{profile} scenario gates")
        rows.extend(
            {
                "domain": "economic_scenario",
                "risk_profile": profile,
                "scenario": scenario,
                "gate": "all_preregistered_economic_gates",
                "passed": gates[scenario],
            }
            for scenario in _SCENARIOS
        )
        rows.append(
            {
                "domain": "economic_profile",
                "risk_profile": profile,
                "scenario": "",
                "gate": "all_scenarios",
                "passed": profile_gates[profile],
            }
        )
    return rows


def _line_svg(
    *,
    title: str,
    description: str,
    y_label: str,
    dates: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
) -> bytes:
    if (
        not dates
        or not series
        or any(len(values) != len(dates) for _, values in series)
    ):
        raise ValueError("Round 17 publication chart series differ")
    width, height = 1200, 640
    left, right, top, bottom = 105, 40, 145, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for _, values in series for value in values]
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * 0.08, max(abs(low), abs(high), 1.0) * 0.01)
    low -= padding
    high += padding

    def x(index: int) -> float:
        return left + (
            plot_width / 2 if len(dates) == 1 else plot_width * index / (len(dates) - 1)
        )

    def y(value: float) -> float:
        return top + plot_height * (high - value) / (high - low)

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(description)}</desc>',
        '<rect width="1200" height="640" fill="#FFFFFF"/>',
        (
            f'<text x="{left}" y="42" fill="#17202A" font-family="Segoe UI,Arial,'
            f'sans-serif" font-size="24" font-weight="600">{escape(title)}</text>'
        ),
        (
            f'<text x="{left}" y="68" fill="#475467" font-family="Segoe UI,Arial,'
            f'sans-serif" font-size="13">{escape(description)}</text>'
        ),
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        vertical = y(value)
        parts.extend(
            (
                (
                    f'<line x1="{left}" y1="{vertical:.2f}" x2="{width - right}" '
                    f'y2="{vertical:.2f}" stroke="#D0D5DD" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 12}" y="{vertical + 4:.2f}" text-anchor="end" '
                    'fill="#475467" font-family="Segoe UI,Arial,sans-serif" '
                    f'font-size="11">{value:.6g}</text>'
                ),
            )
        )
    tick_indexes = (
        [0]
        if len(dates) == 1
        else sorted(
            {
                round(index * (len(dates) - 1) / min(5, len(dates) - 1))
                for index in range(min(5, len(dates) - 1) + 1)
            }
        )
    )
    for index in tick_indexes:
        horizontal = x(index)
        parts.append(
            (
                f'<text x="{horizontal:.2f}" y="{height - bottom + 28}" '
                'text-anchor="middle" fill="#475467" '
                'font-family="Segoe UI,Arial,sans-serif" font-size="11">'
                f"{escape(dates[index])}</text>"
            )
        )
    for index, (label, values) in enumerate(series):
        color = _COLORS[index % len(_COLORS)]
        points = " ".join(
            f"{x(point):.2f},{y(value):.2f}" for point, value in enumerate(values)
        )
        legend_x = left + (index % 4) * 260
        legend_y = 92 + (index // 4) * 24
        parts.extend(
            (
                (
                    f'<line x1="{legend_x}" y1="{legend_y}" '
                    f'x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" '
                    'stroke-width="3"/>'
                ),
                (
                    f'<text x="{legend_x + 36}" y="{legend_y + 4}" fill="#344054" '
                    'font-family="Segoe UI,Arial,sans-serif" font-size="11">'
                    f"{escape(label)}</text>"
                ),
                (
                    f'<polyline points="{points}" fill="none" stroke="{color}" '
                    'stroke-width="2.5" stroke-linejoin="round" '
                    'stroke-linecap="round"/>'
                ),
            )
        )
        parts.extend(
            (
                f'<circle cx="{x(point):.2f}" cy="{y(value):.2f}" r="2.5" '
                f'fill="{color}"/>'
            )
            for point, value in enumerate(values)
        )
    parts.extend(
        (
            (
                f'<text x="25" y="{top + plot_height / 2:.2f}" '
                f'transform="rotate(-90 25 {top + plot_height / 2:.2f})" '
                'text-anchor="middle" fill="#344054" '
                'font-family="Segoe UI,Arial,sans-serif" font-size="12">'
                f"{escape(y_label)}</text>"
            ),
            (
                f'<text x="{width - right}" y="{height - 18}" text-anchor="end" '
                'fill="#667085" font-family="Segoe UI,Arial,sans-serif" '
                'font-size="10">UTC dates; source: terminal-result.json</text>'
            ),
            "</svg>",
        )
    )
    return ("\n".join(parts) + "\n").encode("ascii")


def _endpoint_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    field: str,
    title: str,
    y_label: str,
) -> bytes:
    dates = tuple(dict.fromkeys(str(row["date_utc"]) for row in rows))
    identities = tuple(dict.fromkeys(str(row["series_id"]) for row in rows))
    values = {
        identity: [
            _finite(
                next(
                    row[field]
                    for row in rows
                    if row["series_id"] == identity and row["date_utc"] == day
                ),
                name=f"{identity} {field}",
            )
            for day in dates
        ]
        for identity in identities
    }
    return _line_svg(
        title=title,
        description=(
            "Condition-equal-weighted held-out values by UTC day; "
            "candidate and all preregistered controls."
        ),
        y_label=y_label,
        dates=dates,
        series=tuple((identity, values[identity]) for identity in identities),
    )


def _equity_chart(
    rows: Sequence[Mapping[str, object]],
    *,
    profile: str,
) -> bytes:
    selected = [row for row in rows if row["risk_profile"] == profile]
    dates = tuple(dict.fromkeys(str(row["date_utc"]) for row in selected))
    values = {
        scenario: [
            _finite(
                next(
                    row["ending_equity_quote"]
                    for row in selected
                    if row["scenario"] == scenario and row["date_utc"] == day
                ),
                name=f"{profile} {scenario} equity",
            )
            for day in dates
        ]
        for scenario in _SCENARIOS
    }
    return _line_svg(
        title=f"{profile.title()} after-cost equity by execution scenario",
        description=(
            "Frozen policy replay with no reinvestment; all eight "
            "preregistered liquidity and latency scenarios."
        ),
        y_label="Equity (quote currency)",
        dates=dates,
        series=tuple((scenario, values[scenario]) for scenario in _SCENARIOS),
    )


def _readme(
    result: Mapping[str, object],
    endpoint_rows: Sequence[Mapping[str, object]],
    scenario_rows: Sequence[Mapping[str, object]],
    daily_rows: Sequence[Mapping[str, object]],
) -> bytes:
    endpoint = _object(result["endpoint_holdout"], name="endpoint holdout")
    dates = tuple(dict.fromkeys(str(row["date_utc"]) for row in daily_rows))
    candidate = endpoint_rows[0]
    primary = {
        str(row["risk_profile"]): row
        for row in scenario_rows
        if row["scenario"] == "primary"
    }
    lines = [
        "# Polymarket Round 17 one-use result",
        "",
        "> **Beta research evidence. No paper or live trading authority exists.**",
        "",
        "![Daily held-out log loss](endpoint-daily-log-loss.svg)",
        "",
        (
            f"Status: **{str(result['status']).replace('_', ' ')}**. "
            f"The one-use BTC five-minute test covered "
            f"`{endpoint['condition_count']}` conditions from "
            f"`{dates[0]}` through `{dates[-1]}` UTC. "
            "A failed gate cannot return to development or trigger promotion."
        ),
        "",
        "## Predictive result",
        "",
        "| Series | Role | Log loss | Brier | Balanced accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        lines.append(
            f"| `{row['series_id']}` | {row['role']} | "
            f"{row['log_loss']} | {row['brier_score']} | "
            f"{row['balanced_accuracy']} |"
        )
    lines.extend(
        (
            "",
            "![Daily held-out Brier score](endpoint-daily-brier.svg)",
            "",
            "## After-cost primary scenario",
            "",
            "| Risk profile | Actions | Net PnL | Profit factor | Max drawdown | Gate |",
            "|---|---:|---:|---:|---:|---|",
        )
    )
    for profile in _PROFILES:
        row = primary[profile]
        lines.append(
            f"| {profile.title()} | {row['executed_action_count']} | "
            f"{row['net_pnl_quote']} | {row['profit_factor'] or 'unbounded'} | "
            f"{row['maximum_drawdown_fraction']} | "
            f"{'pass' if row['scenario_gate_passed'] else 'fail'} |"
        )
    lines.extend(
        (
            "",
            "![Conservative scenario equity](equity-conservative.svg)",
            "",
            "![Regular scenario equity](equity-regular.svg)",
            "",
            "![Aggressive scenario equity](equity-aggressive.svg)",
            "",
            "All eight frozen latency, depth, adverse-price, and combined stress "
            "scenarios are retained in the numeric tables. PnL is simulated from "
            "captured public books and the preregistered execution model; it is "
            "not authenticated fill evidence or a live-profit claim.",
            "",
            "## Audit data",
            "",
            "- [Exact terminal result](terminal-result.json)",
            "- [Endpoint summary](endpoint-summary.csv)",
            "- [Daily endpoint metrics](endpoint-daily.csv)",
            "- [Economic scenarios](economic-scenarios.csv)",
            "- [Daily PnL and equity](daily-equity.csv)",
            "- [All gates](gates.csv)",
            "- [Publication integrity](publication-manifest.json)",
            "",
            f"Result SHA-256: `{result['result_sha256']}`",
            f"Selected candidate: `{candidate['series_id']}`",
            "",
        )
    )
    return "\n".join(lines).encode("ascii")


def _validated_existing_publication(
    output_dir: Path,
    result_sha256: str,
) -> dict[str, object] | None:
    manifest_path = output_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return None
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Round 17 publication manifest is invalid")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 17 publication manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Round 17 publication manifest is invalid")
    body = dict(manifest)
    claimed = str(body.pop("publication_sha256", "")).lower()
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != POLYMARKET_ROUND17_PUBLICATION_SCHEMA_VERSION
        or manifest.get("result_sha256") != result_sha256
        or claimed != _canonical_sha256(body)
        or not isinstance(artifacts, list)
    ):
        raise ValueError("Round 17 sealed publication differs")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("Round 17 publication artifact manifest differs")
        path = output_dir / str(artifact.get("path") or "")
        if (
            path.parent != output_dir
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != artifact.get("bytes")
            or _sha256(path.read_bytes()) != artifact.get("sha256")
        ):
            raise ValueError("Round 17 sealed publication artifact differs")
    return dict(manifest)


def publish_round17_one_use_result(
    result: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, object]:
    """Publish numeric authority first and seal derived artifacts last."""

    verified = validate_round17_one_use_result(result)
    requested_destination = Path(output_dir)
    if requested_destination.is_symlink():
        raise ValueError("Round 17 publication output directory is invalid")
    destination = requested_destination.resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError("Round 17 publication output directory is invalid")
    destination.mkdir(parents=True, exist_ok=True)
    existing = _validated_existing_publication(
        destination,
        str(verified["result_sha256"]),
    )
    if existing is not None:
        return existing

    endpoint = _object(verified["endpoint_holdout"], name="endpoint holdout")
    economic = _object(verified["economic_holdout"], name="economic holdout")
    endpoint_summary, identities = _metric_rows(endpoint)
    endpoint_daily = _daily_endpoint_rows(endpoint, identities)
    scenario_rows, daily_rows = _economic_rows(economic)
    gate_rows = _gate_rows(endpoint, economic)
    files = {
        "terminal-result.json": _json_bytes(verified),
        "endpoint-summary.csv": _csv_bytes(
            (
                "series_id",
                "role",
                "log_loss",
                "brier_score",
                "balanced_accuracy",
                "matthews_correlation",
                "calibration_intercept",
                "calibration_slope",
                "expected_calibration_error",
            ),
            endpoint_summary,
        ),
        "endpoint-daily.csv": _csv_bytes(
            (
                "date_utc",
                "day_start_ms",
                "condition_count",
                "series_id",
                "role",
                "log_loss",
                "brier_score",
                "balanced_accuracy",
            ),
            endpoint_daily,
        ),
        "economic-scenarios.csv": _csv_bytes(
            (
                "risk_profile",
                "scenario",
                "condition_count",
                "calendar_day_count",
                "executed_action_count",
                "win_count",
                "win_rate",
                "net_pnl_quote",
                "mean_event_utility_quote",
                "median_daily_pnl_quote",
                "profit_factor",
                "profit_factor_unbounded_without_observed_loss",
                "maximum_drawdown_fraction",
                "worst_daily_loss_fraction",
                "daily_block_bootstrap_mean_event_utility_lower_95",
                "scenario_gate_passed",
            ),
            scenario_rows,
        ),
        "daily-equity.csv": _csv_bytes(
            (
                "date_utc",
                "day_start_ms",
                "risk_profile",
                "scenario",
                "daily_net_pnl_quote",
                "ending_equity_quote",
            ),
            daily_rows,
        ),
        "gates.csv": _csv_bytes(
            ("domain", "risk_profile", "scenario", "gate", "passed"),
            gate_rows,
        ),
        "endpoint-daily-log-loss.svg": _endpoint_chart(
            endpoint_daily,
            field="log_loss",
            title="Round 17 daily held-out log loss",
            y_label="Binary log loss (lower is better)",
        ),
        "endpoint-daily-brier.svg": _endpoint_chart(
            endpoint_daily,
            field="brier_score",
            title="Round 17 daily held-out Brier score",
            y_label="Brier score (lower is better)",
        ),
        **{
            f"equity-{profile}.svg": _equity_chart(
                daily_rows,
                profile=profile,
            )
            for profile in _PROFILES
        },
    }
    files["README.md"] = _readme(
        verified,
        endpoint_summary,
        scenario_rows,
        daily_rows,
    )
    artifacts: list[dict[str, object]] = []
    numeric = {
        "terminal-result.json",
        "endpoint-summary.csv",
        "endpoint-daily.csv",
        "economic-scenarios.csv",
        "daily-equity.csv",
        "gates.csv",
    }
    for name, payload in files.items():
        target = destination / name
        if target.is_symlink():
            raise ValueError("Round 17 publication target is a symlink")
        write_bytes_atomic(target, payload)
        artifacts.append(
            {
                "path": name,
                "bytes": len(payload),
                "sha256": _sha256(payload),
                "role": (
                    "numeric_authority"
                    if name in numeric
                    else "derived_visual_or_narrative"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": POLYMARKET_ROUND17_PUBLICATION_SCHEMA_VERSION,
        "result_sha256": verified["result_sha256"],
        "status": verified["status"],
        "heldout_accepted": verified["heldout_accepted"],
        "generated_from_validated_terminal_result_only": True,
        "automatic_promotion": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "artifacts": artifacts,
        "chart_sources": {
            "endpoint-daily-log-loss.svg": ["endpoint-daily.csv"],
            "endpoint-daily-brier.svg": ["endpoint-daily.csv"],
            **{f"equity-{profile}.svg": ["daily-equity.csv"] for profile in _PROFILES},
        },
    }
    manifest["publication_sha256"] = _canonical_sha256(manifest)
    write_bytes_atomic(destination / _MANIFEST_NAME, _json_bytes(manifest))
    return manifest


__all__ = [
    "POLYMARKET_ROUND17_PUBLICATION_SCHEMA_VERSION",
    "load_round17_one_use_result",
    "publish_round17_one_use_result",
]
