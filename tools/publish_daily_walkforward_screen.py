from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

try:
    from tools.publish_action_value_discovery import (
        _canonical_sha256,
        _sha256,
        _write_csv,
        _write_text,
    )
    from tools.run_daily_walkforward_screen import load_daily_walkforward_design
except ModuleNotFoundError:  # pragma: no cover - direct tools directory execution
    from publish_action_value_discovery import (
        _canonical_sha256,
        _sha256,
        _write_csv,
        _write_text,
    )
    from run_daily_walkforward_screen import load_daily_walkforward_design


_REPORT_SCHEMA = "daily-walk-forward-screen-report-v1"
_CANDIDATE_FIELDS = (
    "phase",
    "candidate_id",
    "evaluation_day",
    "day_status",
    "training_rows",
    "early_stop_rows",
    "calibration_rows",
    "evaluation_rows",
    "backend_kind",
    "backend_device",
    "best_epoch",
    "threshold_candidate_count",
    "accepted_threshold_count",
    "selected_threshold_accepted",
    "least_bad_quantile",
    "least_bad_threshold",
    "least_bad_calibration_trades",
    "least_bad_calibration_total_net_bps",
    "least_bad_calibration_mean_net_bps",
    "least_bad_calibration_max_drawdown_bps",
    "least_bad_calibration_positive_day_ratio",
    "least_bad_calibration_worst_trade_bps",
    "least_bad_calibration_rejection_reasons",
    "evaluation_direction_auc",
    "evaluation_spearman_ic",
    "evaluation_mae_bps",
    "evaluation_zero_mae_bps",
    "evaluation_trades",
    "evaluation_total_net_bps",
    "evaluation_max_drawdown_bps",
    "model_sha256",
    "artifact_path",
    "artifact_bytes",
    "artifact_sha256",
)
_FALSE_CLAIMS = (
    "trading_authority",
    "execution_claim",
    "profitability_claim",
    "portfolio_claim",
    "leverage_applied",
)
_SERIES = (
    ("policy", "rolling-10d", "#147d8f", "rolling 10d policy"),
    ("policy", "rolling-25d", "#315d9b", "rolling 25d policy"),
    (
        "policy",
        "expanding-half-life-7d",
        "#b66b16",
        "expanding 7d half-life policy",
    ),
    ("development", "rolling-10d", "#a23b55", "selected development"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish hash-bound Round 15 daily walk-forward evidence",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument(
        "--prior-progress",
        type=Path,
        default=Path("docs/model-research/action-value/latest/progress.csv"),
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
        raise ValueError(f"unreadable JSON evidence: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {path.name}")
    return payload


def _finite(value: object, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite daily walk-forward {label}")
    return parsed


def _require_false_claims(payload: Mapping[str, object], *, label: str) -> None:
    for key in _FALSE_CLAIMS:
        if payload.get(key) is not False:
            raise ValueError(f"daily walk-forward {label} carries {key}")


def _date_range(start: object, end: object) -> list[str]:
    first = date.fromisoformat(str(start))
    last = date.fromisoformat(str(end))
    if first > last:
        raise ValueError("daily walk-forward date range is reversed")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _validate_report_identity(
    report: Mapping[str, object],
    *,
    design_sha256: str,
) -> str:
    if (
        report.get("schema_version") != _REPORT_SCHEMA
        or int(report.get("round") or 0) != 15
    ):
        raise ValueError("daily walk-forward report identity is invalid")
    claimed = str(report.get("report_sha256") or "")
    canonical = dict(report)
    canonical.pop("report_sha256", None)
    if claimed != _canonical_sha256(canonical):
        raise ValueError("daily walk-forward report canonical SHA-256 mismatch")
    if report.get("design_sha256") != design_sha256:
        raise ValueError("daily walk-forward report design binding differs")
    if report.get("status") != "rejected":
        raise ValueError("Round 15 publication expects the rejected source result")
    _require_false_claims(report, label="report")
    if (
        report.get("terminal_holdout_accessed") is not False
        or report.get("development_window_is_consumed") is not True
    ):
        raise ValueError("daily walk-forward holdout contract is invalid")
    return claimed


def _validate_artifact(evidence_root: Path, raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("daily walk-forward model artifact is missing")
    relative = Path(str(raw.get("path") or ""))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "models"
        or relative.suffix != ".safetensors"
    ):
        raise ValueError("daily walk-forward model artifact path is unsafe")
    root = evidence_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("daily walk-forward model artifact escapes evidence root")
    if (
        not path.is_file()
        or path.stat().st_size != int(raw.get("bytes") or -1)
        or _sha256(path) != str(raw.get("sha256") or "")
    ):
        raise ValueError("daily walk-forward model artifact integrity failed")
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_metrics(raw: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"daily walk-forward {label} metrics are missing")
    trades = int(raw.get("trades") or 0)
    long_trades = int(raw.get("long_trades") or 0)
    short_trades = int(raw.get("short_trades") or 0)
    if (
        trades < 0
        or long_trades < 0
        or short_trades < 0
        or long_trades + short_trades != trades
    ):
        raise ValueError(f"daily walk-forward {label} trade counts are inconsistent")
    for key in (
        "total_net_bps",
        "mean_net_bps",
        "median_net_bps",
        "max_drawdown_bps",
        "win_rate",
        "worst_trade_bps",
        "best_trade_bps",
        "trades_per_active_day",
    ):
        _finite(raw.get(key), label=f"{label} {key}")
    if _finite(raw.get("max_drawdown_bps"), label=label) < 0.0:
        raise ValueError(f"daily walk-forward {label} drawdown is negative")
    if trades == 0 and any(
        abs(_finite(raw.get(key), label=f"{label} empty {key}")) > 1e-12
        for key in ("total_net_bps", "mean_net_bps", "max_drawdown_bps")
    ):
        raise ValueError(f"daily walk-forward {label} empty trace is non-zero")
    return raw


def _validate_trace(raw: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"daily walk-forward {label} trace is missing")
    if (
        raw.get("trading_authority") is not False
        or raw.get("portfolio_claim") is not False
    ):
        raise ValueError(f"daily walk-forward {label} trace carries authority")
    metrics = _validate_metrics(raw.get("metrics"), label=label)
    trades = int(metrics["trades"])
    for key in (
        "timestamps_ms",
        "source_endpoint_indexes",
        "sides",
        "gross_bps",
        "net_bps",
    ):
        values = raw.get(key)
        if not isinstance(values, list) or len(values) != trades:
            raise ValueError(
                f"daily walk-forward {label} trace array differs from trades"
            )
    total_net = sum(_finite(value, label=f"{label} net") for value in raw["net_bps"])
    total_gross = sum(
        _finite(value, label=f"{label} gross") for value in raw["gross_bps"]
    )
    if not math.isclose(total_net, float(metrics["total_net_bps"]), abs_tol=1e-8):
        raise ValueError(f"daily walk-forward {label} trace net total differs")
    if not math.isclose(
        total_gross, float(raw.get("total_gross_bps") or 0.0), abs_tol=1e-8
    ):
        raise ValueError(f"daily walk-forward {label} trace gross total differs")
    return raw


def _validate_forecast(raw: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or int(raw.get("rows") or 0) <= 0:
        raise ValueError(f"daily walk-forward {label} forecast metrics are invalid")
    for key in (
        "direction_auc",
        "spearman_information_coefficient",
        "mean_absolute_error_bps",
        "zero_baseline_mae_bps",
    ):
        _finite(raw.get(key), label=f"{label} {key}")
    return raw


def _least_bad_threshold(
    selection: Mapping[str, object],
    *,
    quantiles: Sequence[float],
    label: str,
) -> tuple[Mapping[str, object], int]:
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(quantiles):
        raise ValueError(f"daily walk-forward {label} threshold grid is incomplete")
    found: list[Mapping[str, object]] = []
    seen: set[float] = set()
    accepted = 0
    for raw in candidates:
        if not isinstance(raw, Mapping):
            raise ValueError(f"daily walk-forward {label} threshold row is invalid")
        quantile = _finite(raw.get("quantile"), label=f"{label} quantile")
        if quantile in seen:
            raise ValueError(
                f"daily walk-forward {label} threshold quantile is duplicated"
            )
        seen.add(quantile)
        _finite(raw.get("threshold"), label=f"{label} threshold")
        _finite(raw.get("utility_bps"), label=f"{label} utility")
        _finite(raw.get("positive_day_ratio"), label=f"{label} positive-day ratio")
        _validate_metrics(raw.get("metrics"), label=f"{label} calibration")
        if not isinstance(raw.get("rejection_reasons"), list):
            raise ValueError(
                f"daily walk-forward {label} threshold reasons are invalid"
            )
        if raw.get("trading_authority") is not False:
            raise ValueError(f"daily walk-forward {label} threshold carries authority")
        accepted += raw.get("accepted") is True
        found.append(raw)
    if sorted(seen) != sorted(float(value) for value in quantiles):
        raise ValueError(
            f"daily walk-forward {label} threshold quantiles differ from design"
        )
    # This is descriptive evidence only: the least-negative calibration trace is
    # not retroactively selected and cannot authorize an evaluation-day trade.
    least_bad = max(
        found,
        key=lambda row: (
            float(row["metrics"]["total_net_bps"]),
            -float(row["metrics"]["max_drawdown_bps"]),
            float(row["quantile"]),
        ),
    )
    return least_bad, accepted


def _validate_day(
    raw: object,
    *,
    phase: str,
    candidate_id: str,
    expected_day: str,
    quantiles: Sequence[float],
    evidence_root: Path,
) -> dict[str, object]:
    label = f"{phase} {candidate_id} {expected_day}"
    if not isinstance(raw, Mapping) or raw.get("evaluation_day") != expected_day:
        raise ValueError(f"daily walk-forward {label} day identity differs")
    if raw.get("candidate_id") != candidate_id:
        raise ValueError(f"daily walk-forward {label} candidate identity differs")
    _require_false_claims(raw, label=label)
    plan = raw.get("plan")
    model = raw.get("model")
    selection = raw.get("threshold_selection")
    if (
        not isinstance(plan, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(selection, Mapping)
    ):
        raise ValueError(f"daily walk-forward {label} model plan is incomplete")
    for claim in ("trading_authority", "execution_claim", "profitability_claim"):
        if model.get(claim) is not False:
            raise ValueError(f"daily walk-forward {label} model carries {claim}")
    if model.get("schema_version") != "exact-bbo-gross-architecture-v2":
        raise ValueError(f"daily walk-forward {label} model schema differs")
    model_sha256 = str(model.get("model_sha256") or "")
    if len(model_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in model_sha256
    ):
        raise ValueError(f"daily walk-forward {label} model hash is invalid")
    artifact = _validate_artifact(evidence_root, raw.get("model_artifact"))
    forecast = _validate_forecast(raw.get("evaluation_forecast_metrics"), label=label)
    evaluation_trace = _validate_trace(
        raw.get("evaluation_trace"), label=f"{label} evaluation"
    )
    least_bad, accepted_count = _least_bad_threshold(
        selection,
        quantiles=quantiles,
        label=label,
    )
    selection_accepted = selection.get("accepted") is True
    if selection.get("trading_authority") is not False:
        raise ValueError(f"daily walk-forward {label} selection carries authority")
    if selection_accepted != (accepted_count > 0):
        raise ValueError(f"daily walk-forward {label} accepted threshold count differs")
    if not selection_accepted:
        if (
            selection.get("quantile") is not None
            or selection.get("threshold") is not None
        ):
            raise ValueError(
                f"daily walk-forward {label} rejected threshold was selected"
            )
        selected_trace = _validate_trace(
            selection.get("selected_trace"),
            label=f"{label} abstention",
        )
        if int(selected_trace["metrics"]["trades"]) != 0:
            raise ValueError(
                f"daily walk-forward {label} abstention trace contains trades"
            )
        if int(evaluation_trace["metrics"]["trades"]) != 0:
            raise ValueError(
                f"daily walk-forward {label} evaluation bypassed abstention"
            )
    least_metrics = least_bad["metrics"]
    eval_metrics = evaluation_trace["metrics"]
    return {
        "phase": phase,
        "candidate_id": candidate_id,
        "evaluation_day": expected_day,
        "day_status": "threshold_selected" if selection_accepted else "abstained",
        "training_rows": int(plan.get("training_rows") or 0),
        "early_stop_rows": int(plan.get("early_stop_rows") or 0),
        "calibration_rows": int(plan.get("calibration_rows") or 0),
        "evaluation_rows": int(plan.get("evaluation_rows") or 0),
        "backend_kind": str(model.get("backend_kind") or ""),
        "backend_device": str(model.get("backend_device") or ""),
        "best_epoch": int(model.get("best_epoch") or 0),
        "threshold_candidate_count": len(selection["candidates"]),
        "accepted_threshold_count": accepted_count,
        "selected_threshold_accepted": selection_accepted,
        "least_bad_quantile": least_bad["quantile"],
        "least_bad_threshold": least_bad["threshold"],
        "least_bad_calibration_trades": least_metrics["trades"],
        "least_bad_calibration_total_net_bps": least_metrics["total_net_bps"],
        "least_bad_calibration_mean_net_bps": least_metrics["mean_net_bps"],
        "least_bad_calibration_max_drawdown_bps": least_metrics["max_drawdown_bps"],
        "least_bad_calibration_positive_day_ratio": least_bad["positive_day_ratio"],
        "least_bad_calibration_worst_trade_bps": least_metrics["worst_trade_bps"],
        "least_bad_calibration_rejection_reasons": ";".join(
            str(value) for value in least_bad["rejection_reasons"]
        ),
        "evaluation_direction_auc": forecast["direction_auc"],
        "evaluation_spearman_ic": forecast["spearman_information_coefficient"],
        "evaluation_mae_bps": forecast["mean_absolute_error_bps"],
        "evaluation_zero_mae_bps": forecast["zero_baseline_mae_bps"],
        "evaluation_trades": eval_metrics["trades"],
        "evaluation_total_net_bps": eval_metrics["total_net_bps"],
        "evaluation_max_drawdown_bps": eval_metrics["max_drawdown_bps"],
        "model_sha256": model_sha256,
        "artifact_path": artifact["path"],
        "artifact_bytes": artifact["bytes"],
        "artifact_sha256": artifact["sha256"],
    }


def _validated_evidence(
    evidence_root: Path,
    design: Mapping[str, object],
    design_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    report = _read_json(evidence_root / "report.json")
    _validate_report_identity(report, design_sha256=design_sha256)
    quantiles = [float(value) for value in design["threshold_policy"]["quantiles"]]
    evaluation = design["evaluation"]
    policy_days = _date_range(evaluation["policy_start"], evaluation["policy_end"])
    development_days = _date_range(
        evaluation["development_start"],
        evaluation["development_end"],
    )
    fit_ids = [str(value["candidate_id"]) for value in design["fit_candidates"]]
    policy = report.get("policy_results")
    if not isinstance(policy, list) or len(policy) != len(fit_ids):
        raise ValueError("daily walk-forward policy candidate set differs from design")
    blocks: dict[str, Mapping[str, object]] = {}
    for block in policy:
        if not isinstance(block, Mapping) or not isinstance(
            block.get("candidate"), Mapping
        ):
            raise ValueError("daily walk-forward policy candidate is invalid")
        candidate_id = str(block["candidate"].get("candidate_id") or "")
        if candidate_id in blocks:
            raise ValueError("daily walk-forward policy candidate is duplicated")
        blocks[candidate_id] = block
    if set(blocks) != set(fit_ids):
        raise ValueError("daily walk-forward policy candidate identities differ")
    rows: list[dict[str, object]] = []
    for candidate_id in fit_ids:
        block = blocks[candidate_id]
        _require_false_claims(block, label=f"policy {candidate_id}")
        daily = block.get("daily_results")
        if not isinstance(daily, list) or len(daily) != len(policy_days):
            raise ValueError(f"daily walk-forward policy days differ: {candidate_id}")
        for expected_day, raw in zip(policy_days, daily, strict=True):
            rows.append(
                _validate_day(
                    raw,
                    phase="policy",
                    candidate_id=candidate_id,
                    expected_day=expected_day,
                    quantiles=quantiles,
                    evidence_root=evidence_root,
                )
            )
    development = report.get("development_result")
    selected_id = str(report.get("selected_policy_candidate_id") or "")
    if (
        not isinstance(development, Mapping)
        or not isinstance(development.get("candidate"), Mapping)
        or development["candidate"].get("candidate_id") != selected_id
    ):
        raise ValueError(
            "daily walk-forward development candidate differs from policy selection"
        )
    _require_false_claims(development, label="development")
    daily = development.get("daily_results")
    if not isinstance(daily, list) or len(daily) != len(development_days):
        raise ValueError("daily walk-forward development days differ from design")
    for expected_day, raw in zip(development_days, daily, strict=True):
        rows.append(
            _validate_day(
                raw,
                phase="development",
                candidate_id=selected_id,
                expected_day=expected_day,
                quantiles=quantiles,
                evidence_root=evidence_root,
            )
        )
    if len(rows) != 21:
        raise ValueError("Round 15 publication requires exactly 21 causal daily fits")
    return report, rows


def _series_rows(
    rows: Sequence[Mapping[str, object]],
    phase: str,
    candidate_id: str,
) -> list[Mapping[str, object]]:
    return sorted(
        (
            row
            for row in rows
            if row["phase"] == phase and row["candidate_id"] == candidate_id
        ),
        key=lambda row: str(row["evaluation_day"]),
    )


def _economics_svg(rows: Sequence[Mapping[str, object]]) -> str:
    ordered = [
        row
        for phase, candidate, _, _ in _SERIES
        for row in _series_rows(rows, phase, candidate)
    ]
    width, height = 1600, 1060
    left, right, top = 360, 130, 128
    row_gap, bar_height = 38, 20
    chart_width = width - left - right
    values = [
        _finite(
            row["least_bad_calibration_total_net_bps"], label="chart calibration net"
        )
        for row in ordered
    ]
    lower = min(-50.0, math.floor(min(values, default=-50.0) / 50.0) * 50.0)

    def x_position(value: float) -> float:
        return left + chart_width * (value - lower) / (0.0 - lower)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 15 prior-only calibration economics</title>',
        '<desc id="desc">Least-negative after-cost calibration trace for each causal daily fit. Every value is negative and no threshold was selected.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Prior-only calibration economics</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Least-negative threshold per fit; these are two-day calibration traces, not evaluation trades, ROI, or an equity curve.</text>',
    ]
    for tick in range(int(lower), 1, 50):
        x = x_position(float(tick))
        lines.append(
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{top + row_gap * len(ordered)}" stroke="#{"637785" if tick == 0 else "e1e8ed"}" stroke-width="{2 if tick == 0 else 1}"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top - 20}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{tick}</text>'
        )
    for index, row in enumerate(ordered):
        value = float(row["least_bad_calibration_total_net_bps"])
        y = top + index * row_gap
        label = html.escape(
            f"{row['phase']} / {row['candidate_id']} / {row['evaluation_day']}"
        )
        lines.append(
            f'<text x="{left - 16}" y="{y + 15}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#334653">{label}</text>'
        )
        x = x_position(value)
        zero = x_position(0.0)
        lines.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(1.0, zero - x):.1f}" height="{bar_height}" rx="3" fill="#b44a45"/>'
        )
        lines.append(
            f'<text x="{x - 8:.1f}" y="{y + 15}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#7d2929">{value:+.2f}</text>'
        )
    lines.extend(
        [
            f'<text x="{left + chart_width / 2:.1f}" y="{height - 42}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Total exact after-cost basis points on the preceding calibration window</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1600, 930
    left, right = 120, 70
    chart_width = width - left - right
    days = sorted({str(row["evaluation_day"]) for row in rows})
    x_by_day = {
        day: left + chart_width * index / max(1, len(days) - 1)
        for index, day in enumerate(days)
    }

    def panel(
        *,
        key: str,
        title: str,
        top: float,
        bottom: float,
        lower: float,
        upper: float,
        baseline: float,
    ) -> list[str]:
        def y_position(value: float) -> float:
            return top + (bottom - top) * (upper - value) / (upper - lower)

        output = [
            f'<text x="{left}" y="{top - 22}" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="#263746">{html.escape(title)}</text>',
            f'<rect x="{left}" y="{top}" width="{chart_width}" height="{bottom - top}" fill="#ffffff" stroke="#d8e0e7"/>',
        ]
        for tick in (lower, (lower + upper) / 2.0, upper):
            y = y_position(tick)
            output.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#e4eaef"/>'
            )
            output.append(
                f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{tick:.2f}</text>'
            )
        baseline_y = y_position(baseline)
        output.append(
            f'<line x1="{left}" y1="{baseline_y:.1f}" x2="{left + chart_width}" y2="{baseline_y:.1f}" stroke="#6b7780" stroke-width="2" stroke-dasharray="6 6"/>'
        )
        for phase, candidate, color, _ in _SERIES:
            matching = _series_rows(rows, phase, candidate)
            points = [
                (
                    x_by_day[str(row["evaluation_day"])],
                    y_position(_finite(row[key], label=f"chart {key}")),
                )
                for row in matching
            ]
            if points:
                output.append(
                    '<polyline points="'
                    + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                    + f'" fill="none" stroke="{color}" stroke-width="3"/>'
                )
                output.extend(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>'
                    for x, y in points
                )
        return output

    auc_values = [float(row["evaluation_direction_auc"]) for row in rows]
    ic_values = [float(row["evaluation_spearman_ic"]) for row in rows]
    auc_lower = min(0.48, math.floor(min(auc_values) * 100.0) / 100.0 - 0.01)
    auc_upper = max(0.62, math.ceil(max(auc_values) * 100.0) / 100.0 + 0.01)
    ic_lower = min(-0.03, math.floor(min(ic_values) * 100.0) / 100.0 - 0.01)
    ic_upper = max(0.20, math.ceil(max(ic_values) * 100.0) / 100.0 + 0.01)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 15 daily forecast quality</title>',
        '<desc id="desc">Daily direction AUC and Spearman information coefficient for three policy refit schedules and the selected development schedule.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Daily forecast quality did not become tradable</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Forecast association remained positive on many days, but every threshold-selection candidate failed the precommitted net-of-cost acceptance criteria.</text>',
    ]
    for index, (_, _, color, label) in enumerate(_SERIES):
        x = 120 + index * 355
        y = 120
        lines.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="4"/>'
        )
        lines.append(
            f'<text x="{x + 38}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{html.escape(label)}</text>'
        )
    lines.extend(
        panel(
            key="evaluation_direction_auc",
            title="Direction AUC (0.50 is chance)",
            top=180,
            bottom=440,
            lower=auc_lower,
            upper=auc_upper,
            baseline=0.5,
        )
    )
    lines.extend(
        panel(
            key="evaluation_spearman_ic",
            title="Spearman information coefficient (0.00 is no rank association)",
            top=540,
            bottom=815,
            lower=ic_lower,
            upper=ic_upper,
            baseline=0.0,
        )
    )
    for index, day in enumerate(days):
        x = x_by_day[day]
        lines.append(
            f'<text x="{x:.1f}" y="850" transform="rotate(35 {x:.1f} 850)" text-anchor="start" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#60717f">{html.escape(day)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _funnel_svg(
    *,
    model_fits: int,
    threshold_traces: int,
    accepted_thresholds: int,
    evaluation_trades: int,
    research_candidates: int,
) -> str:
    stages = (
        (model_fits, "causal daily model fits"),
        (threshold_traces, "prior-only threshold traces"),
        (accepted_thresholds, "accepted thresholds"),
        (evaluation_trades, "evaluation trades"),
        (research_candidates, "research candidates"),
    )
    width, height = 1600, 470
    gap, box_width = 20, 274
    start = (width - (box_width * len(stages) + gap * (len(stages) - 1))) / 2
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 15 signal selection</title>',
        '<desc id="desc">Twenty-one model fits produced eighty-four threshold traces. None passed, so evaluation remained in abstention and no candidate was created.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Mandatory abstention stopped failed thresholds</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">The selection sequence reports operational evidence; zero evaluation trades is an abstention result, not profitability evidence.</text>',
    ]
    for index, (value, label) in enumerate(stages):
        x = start + index * (box_width + gap)
        fill = "#edf6f7" if value else "#fff5f4"
        stroke = "#147d8f" if value else "#b44a45"
        lines.append(
            f'<rect x="{x:.1f}" y="145" width="{box_width}" height="150" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="215" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="40" font-weight="700" fill="{stroke}">{value}</text>'
        )
        words = label.split()
        split = max(1, len(words) // 2)
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="252" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#263746"><tspan x="{x + box_width / 2:.1f}">{html.escape(" ".join(words[:split]))}</tspan><tspan x="{x + box_width / 2:.1f}" dy="19">{html.escape(" ".join(words[split:]))}</tspan></text>'
        )
    lines.extend(
        [
            '<text x="56" y="387" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">No trade, leverage, portfolio return, ROI, or profitability authority was produced.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _progress_svg(progress: Sequence[Mapping[str, object]]) -> str:
    rows = [row for row in progress if int(row["round"]) >= 7]
    compact_labels = len(rows) > 18
    width, height = 1500, 620
    left, right, top, chart_height = 120, 70, 158, 330
    chart_width = width - left - right
    lower, upper = -14.0, 2.0

    def y_position(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">After-cost evidence by research round</title>',
        '<desc id="desc">Executed means, overlapping-forecast diagnostics, and rounds with no executable series use distinct markers. Round fifteen produced no evaluation trades.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">After-cost evidence by research round</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Distinct markers prevent diagnostics or abstention from being presented as an executed trade mean.</text>',
        '<circle cx="900" cy="55" r="7" fill="#b42318"/><text x="917" y="60" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">executed mean</text>',
        '<rect x="1052" y="48" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 1059 55)"/><text x="1076" y="60" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">no executable series</text>',
        '<rect x="1277" y="48" width="14" height="14" fill="#7b559c"/><text x="1301" y="60" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">overlap diagnostic</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in (-14.0, -10.0, -5.0, 0.0):
        y = y_position(value)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#{"526674" if value == 0.0 else "e5ebf0"}" stroke-width="{2 if value == 0.0 else 1}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value:.0f}</text>'
        )
    executed_points: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        x = left + chart_width * (index + 0.5) / max(1, len(rows))
        net_text = str(row.get("mean_net_bps") or "").strip()
        diagnostic_text = str(
            row.get("best_top_500_exact_after_cost_bps") or ""
        ).strip()
        if int(row.get("executable_trades") or 0) > 0 and net_text:
            value = _finite(net_text, label="progress executed mean")
            y = y_position(value)
            executed_points.append((x, y))
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#b42318" stroke="#ffffff" stroke-width="3"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#263744">{value:+.2f}</text>'
            )
        elif diagnostic_text:
            value = _finite(diagnostic_text, label="progress diagnostic")
            y = y_position(value)
            lines.append(
                f'<rect x="{x - 8:.1f}" y="{y - 8:.1f}" width="16" height="16" fill="#7b559c" stroke="#ffffff" stroke-width="2"/>'
            )
        else:
            y = y_position(-6.0)
            lines.append(
                f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 {x:.1f} {y:.1f})"/>'
            )
        round_label = f'R{row["round"]}' if compact_labels else f'Round {row["round"]}'
        lines.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="{11 if compact_labels else 13}" fill="#334653">{round_label}</text>'
        )
    if len(executed_points) > 1:
        lines.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in executed_points)
            + '" fill="none" stroke="#287f9e" stroke-width="4"/>'
        )
    lines.extend(
        [
            '<text x="42" y="323" transform="rotate(-90 42 323)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">After-cost basis points</text>',
            f'<text x="56" y="574" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">{"R denotes research round. " if compact_labels else ""}Windows and units differ by round. This is evidence lineage, not a continuous equity curve or portfolio return series.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _progress_rows(
    path: Path,
    design: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        progress = [dict(row) for row in csv.DictReader(handle)]
    if progress and int(progress[-1]["round"]) == 15:
        progress.pop()
    if not progress or int(progress[-1]["round"]) != 14:
        raise ValueError("prior progress table does not end at Round 14")
    development = [row for row in rows if row["phase"] == "development"]
    execution = design["execution"]
    data = design["data"]
    progress.append(
        {
            "round": 15,
            "stage": "causal daily refit and mandatory abstention screen",
            "periods": f"{data['start_date']}..{data['end_date']}",
            "selection_contaminated": True,
            "horizon_seconds": execution["horizon_seconds"],
            "feature_set": "l1-tape-causal-v7",
            "risk_level": "research-only",
            "direction_auc": statistics.fmean(
                float(row["evaluation_direction_auc"]) for row in development
            ),
            "spearman_ic": statistics.fmean(
                float(row["evaluation_spearman_ic"]) for row in development
            ),
            "selected_signals": 0,
            "executable_trades": sum(int(row["evaluation_trades"]) for row in rows),
            "mean_gross_bps": "",
            "mean_net_bps": "",
            "status": str(report["status"]),
            "source_file": "daily walk-forward Round 15 report",
            "best_model_id": report["selected_policy_candidate_id"],
            "daily_model_fits": len(rows),
            "calibration_threshold_traces": sum(
                int(row["threshold_candidate_count"]) for row in rows
            ),
            "accepted_thresholds": sum(
                int(row["accepted_threshold_count"]) for row in rows
            ),
            "least_bad_calibration_total_net_bps": max(
                float(row["least_bad_calibration_total_net_bps"])
                for row in rows
                if row["phase"] == "policy"
            ),
        }
    )
    return progress


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def publish(
    evidence_root: Path,
    design_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    design, design_sha256 = load_daily_walkforward_design(
        design_path,
        require_current=False,
    )
    report, rows = _validated_evidence(evidence_root, design, design_sha256)
    progress = _progress_rows(prior_progress_path, design, rows, report)
    progress_fields = tuple(progress[0]) + tuple(
        key for key in progress[-1] if key not in progress[0]
    )
    threshold_traces = sum(int(row["threshold_candidate_count"]) for row in rows)
    accepted_thresholds = sum(int(row["accepted_threshold_count"]) for row in rows)
    evaluation_trades = sum(int(row["evaluation_trades"]) for row in rows)
    least_bad_policy = max(
        (row for row in rows if row["phase"] == "policy"),
        key=lambda row: float(row["least_bad_calibration_total_net_bps"]),
    )
    charts = output_dir / "charts"
    diagnostics: dict[str, object] = {
        "schema_version": "daily-walk-forward-publication-diagnostics-v1",
        "artifact_class": "consumed_daily_walk_forward_evidence_no_trading_authority",
        "design_sha256": design_sha256,
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": True,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "publication_rows": rows,
        "source_report": report,
    }
    diagnostics["diagnostic_sha256"] = _canonical_sha256(diagnostics)
    _write_csv(output_dir / "candidates.csv", rows, _CANDIDATE_FIELDS)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "diagnostics.json",
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(charts / "after-cost-performance.svg", _economics_svg(rows))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(rows))
    _write_text(
        charts / "signal-selection.svg",
        _funnel_svg(
            model_fits=len(rows),
            threshold_traces=threshold_traces,
            accepted_thresholds=accepted_thresholds,
            evaluation_trades=evaluation_trades,
            research_candidates=0,
        ),
    )
    _write_text(charts / "research-progress.svg", _progress_svg(progress))

    execution = design["execution"]
    round_trip_cost = 2.0 * (
        float(execution["taker_fee_bps_per_side"])
        + float(execution["additional_slippage_bps_per_side"])
    )
    readme = f"""# Round 15: daily refits abstained

**Rejected without trading authority.** All **{threshold_traces}** prior-only threshold traces from **{len(rows)}** causal daily model fits failed the precommitted net-of-cost acceptance criteria. No threshold was allowed to trade an evaluation day.

| Evidence | Result |
| --- | ---: |
| Least-negative policy calibration trace | {float(least_bad_policy["least_bad_calibration_total_net_bps"]):+.2f} bps over {int(least_bad_policy["least_bad_calibration_trades"])} trades |
| Its maximum drawdown | {float(least_bad_policy["least_bad_calibration_max_drawdown_bps"]):.2f} bps |
| Accepted thresholds | {accepted_thresholds} / {threshold_traces} |
| Evaluation trades | {evaluation_trades} |
| Research candidates | 0 |

![Prior-only calibration economics](charts/after-cost-performance.svg)

![Daily forecast quality](charts/forecast-quality.svg)

![Signal selection](charts/signal-selection.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, {design["data"]["start_date"]} through {design["data"]["end_date"]} UTC; {int(report["dataset"]["event_rows"]):,} causal events from {int(report["dataset"]["rows"]):,} exact-BBO rows. Traces use {int(execution["total_latency_ms"])} ms latency and {round_trip_cost:.0f} bps configured taker round-trip cost. The development window is consumed; 2023-07-07 remains untouched.

No ROI or equity curve is shown because no evaluation trade occurred. Fixed-horizon traces still lack intrahorizon stop-loss paths, so this result cannot authorize trading, leverage, or a profitability claim.

Data: [candidates.csv](candidates.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
"""
    _write_text(output_dir / "README.md", readme)
    generated = [
        output_dir / "README.md",
        output_dir / "candidates.csv",
        output_dir / "progress.csv",
        output_dir / "diagnostics.json",
        charts / "after-cost-performance.svg",
        charts / "forecast-quality.svg",
        charts / "signal-selection.svg",
        charts / "research-progress.svg",
    ]
    implementation = design["implementation"]
    publication: dict[str, object] = {
        "schema_version": "daily-walk-forward-screen-publication-v1",
        "artifact_class": "exchange_sourced_daily_walk_forward_graph_data",
        "round": 15,
        "design_revision": int(design["design_revision"]),
        "status": str(report["status"]),
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": True,
        "design_sha256": design_sha256,
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "implementation_commit": implementation["commit"],
        "corpus_certificate_sha256": report["corpus_certificate_sha256"],
        "diagnostic_sha256": diagnostics["diagnostic_sha256"],
        "actual": {
            "daily_model_fits": len(rows),
            "policy_model_fits": sum(row["phase"] == "policy" for row in rows),
            "development_model_fits": sum(
                row["phase"] == "development" for row in rows
            ),
            "calibration_threshold_traces": threshold_traces,
            "accepted_thresholds": accepted_thresholds,
            "evaluation_trades": evaluation_trades,
            "research_candidates": 0,
            "least_bad_policy_calibration_total_net_bps": least_bad_policy[
                "least_bad_calibration_total_net_bps"
            ],
            "least_bad_policy_calibration_trades": least_bad_policy[
                "least_bad_calibration_trades"
            ],
            "least_bad_policy_calibration_max_drawdown_bps": least_bad_policy[
                "least_bad_calibration_max_drawdown_bps"
            ],
            "selected_policy_candidate_id": report["selected_policy_candidate_id"],
        },
        "source_artifacts": [
            {
                "path": row["artifact_path"],
                "sha256": row["artifact_sha256"],
                "bytes": row["artifact_bytes"],
            }
            for row in rows
        ],
        "artifact_integrity": [
            {
                "path": _portable_path(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in generated
        ],
    }
    publication["publication_sha256"] = _canonical_sha256(publication)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return publication


def main() -> int:
    args = _arguments()
    publication = publish(
        args.evidence_root,
        args.design,
        args.prior_progress,
        args.output_dir,
    )
    print(
        "daily-walk-forward-publication: "
        f"status={publication['status']} "
        f"fits={publication['actual']['daily_model_fits']} "
        f"trades={publication['actual']['evaluation_trades']} "
        f"sha256={publication['publication_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
