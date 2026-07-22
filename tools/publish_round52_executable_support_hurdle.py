"""Publish truthful Round 52 model evidence and source tables."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROUND = 52
REPORT_SCHEMA = "round-052-executable-support-hurdle-fincast-report-v1"
DESIGN_SHA256 = "af95d80a3adc21b72d6809d43afb3f2446213fe0a4e089b10366691465a0c669"
BINDING_SHA256 = "e14f6e4b742e0da6d838621355a16fbc030ae2a941b6eef0ee2dd1ab9344568a"
SOURCE_REPORT_CANONICAL_SHA256 = (
    "ace44ebc33dc0601306841b4c353b43a184b2aa604b49f73d2301257f86d2f7f"
)
SOURCE_REPORT_FILE_SHA256 = (
    "c5b728161535372d934ff9087a24b81c2490246cd17cb56beb3e29a3052d73fa"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SEEDS = (5201, 5202, 5203)
CANDIDATES = (
    "executable_direct_mean_lightgbm",
    "executable_hurdle_lightgbm",
    "executable_hurdle_lightgbm_fincast",
)
DISPLAY = {
    CANDIDATES[0]: "Direct mean",
    CANDIDATES[1]: "Executable hurdle",
    CANDIDATES[2]: "Hurdle + FinCast",
}
COLORS = {
    "teal": "#14b8a6",
    "cyan": "#22d3ee",
    "blue": "#60a5fa",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "green": "#22c55e",
    "muted": "#94a3b8",
    "grid": "#334155",
    "panel": "#111827",
    "background": "#0b1220",
    "text": "#f8fafc",
    "subtext": "#cbd5e1",
}


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
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _validate_finite(value: object, label: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{label}[{index}]")


def _verified_artifact(item: Mapping[str, object], label: str) -> None:
    path = Path(str(item.get("path") or ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(item.get("bytes", -1))
        or _file_sha256(path) != str(item.get("sha256") or "")
    ):
        raise ValueError(f"{label} artifact identity drifted")


def _validate_source(
    *,
    report_path: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if _file_sha256(report_path) != SOURCE_REPORT_FILE_SHA256:
        raise ValueError("Round 52 source report file hash drifted")
    report = _read_object(report_path, "Round 52 report")
    design = _read_object(design_path, "Round 52 design")
    binding = _read_object(binding_path, "Round 52 binding")
    report_canonical = dict(report)
    report_claimed = report_canonical.pop("report_canonical_sha256", None)
    design_canonical = dict(design)
    design_claimed = design_canonical.pop("design_sha256", None)
    binding_canonical = dict(binding)
    binding_claimed = binding_canonical.pop("binding_sha256", None)
    claims = report.get("claims")
    mechanism = report.get("mechanism_screen")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report_claimed != SOURCE_REPORT_CANONICAL_SHA256
        or _canonical_sha256(report_canonical) != report_claimed
        or design_claimed != DESIGN_SHA256
        or _canonical_sha256(design_canonical) != design_claimed
        or binding_claimed != BINDING_SHA256
        or _canonical_sha256(binding_canonical) != binding_claimed
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("binding_sha256") != BINDING_SHA256
        or report.get("round_gate", {}).get("passed") is not False
        or not isinstance(claims, Mapping)
        or claims.get("selection_contaminated") is not True
        or any(
            claims.get(name) is not False
            for name in (
                "profitability_claim",
                "ai_uplift_claim",
                "trading_authority",
                "testnet_authority",
                "live_authority",
                "leverage_applied",
            )
        )
        or not isinstance(mechanism, Mapping)
        or mechanism.get("passed_candidates") != []
        or mechanism.get("untouched_data_expansion_authorized") is not False
        or mechanism.get("trading_or_promotion_authorized") is not False
    ):
        raise ValueError("Round 52 source contracts or claims drifted")
    _validate_finite(report)
    model_count = 0
    prediction_count = 0
    for symbol in SYMBOLS:
        if int(report["data"][symbol]["synthetic_rows"]) != 0:
            raise ValueError(f"Round 52 {symbol} contains synthetic rows")
        _verified_artifact(
            report["data"][symbol]["fincast"]["artifact"], f"{symbol} FinCast"
        )
        for candidate in CANDIDATES:
            for seed in SEEDS:
                _verified_artifact(
                    report["models"][symbol][candidate][str(seed)]["artifact"],
                    f"{symbol} {candidate} {seed} model",
                )
                model_count += 1
                for role in ("policy_calibration", "evaluation"):
                    _verified_artifact(
                        report["prediction_artifacts"][symbol][candidate][str(seed)][
                            role
                        ],
                        f"{symbol} {candidate} {seed} {role} prediction",
                    )
                    prediction_count += 1
    if model_count != 27 or prediction_count != 54:
        raise ValueError("Round 52 artifact inventory is incomplete")
    return report, design, binding


def _json_cell(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"publication table would be empty: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_cell(row.get(key, "")) for key in fields})


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _support_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        data = report["data"][symbol]
        support = data["support"]
        rows.append(
            {
                "round": ROUND,
                "symbol": symbol,
                "source_rows": data["microstructure_rows"],
                "valid_barrier_rows": data["deterministic_rows"],
                "long_executable_rows": support["long_executable_rows"],
                "long_executable_ratio": support["long_executable_ratio"],
                "short_executable_rows": support["short_executable_rows"],
                "short_executable_ratio": support["short_executable_ratio"],
                "long_mask_sha256": support["long_mask_sha256"],
                "short_mask_sha256": support["short_mask_sha256"],
                "synthetic_rows": data["synthetic_rows"],
            }
        )
    return rows


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            for seed in SEEDS:
                record = report["predictive_metrics"][symbol][candidate][str(seed)]
                for role in ("policy_calibration", "evaluation"):
                    diagnostics = record["prediction_diagnostics"][role]
                    for side in ("long", "short"):
                        metric = record[role][side]
                        rows.append(
                            {
                                "round": ROUND,
                                "symbol": symbol,
                                "candidate": candidate,
                                "seed": seed,
                                "role": role,
                                "side": side,
                                **metric,
                                "magnitude_floor_count": diagnostics[
                                    "magnitude_floor_count"
                                ],
                            }
                        )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            for seed in SEEDS:
                model = report["models"][symbol][candidate][str(seed)]
                rows.append(
                    {
                        "round": ROUND,
                        "symbol": symbol,
                        "candidate": candidate,
                        "seed": seed,
                        "cache_state": model["cache_state"],
                        "backend_kind": model["backend_kind"],
                        "backend_device": model["backend_device"],
                        "model_sha256": model["model_sha256"],
                        "artifact_sha256": model["artifact"]["sha256"],
                        "artifact_bytes": model["artifact"]["bytes"],
                        "role_rows": model["role_rows"],
                        "rejected_role_rows": model["rejected_role_rows"],
                        "role_mask_sha256": model["role_mask_sha256"],
                        "minimum_leaf_rows": model["minimum_leaf_rows"],
                        "best_iterations": model["best_iterations"],
                    }
                )
    return rows


def _policy_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for item in report["policy_grid"][candidate]:
            row: dict[str, object] = {
                "round": ROUND,
                "candidate": candidate,
                "coverage": item["coverage"],
                "thresholds_bps": item["thresholds_bps"],
                "policy_calibration_passed": item["policy_calibration_gate"]["passed"],
                "policy_calibration_reasons": item["policy_calibration_gate"][
                    "reasons"
                ],
                "consumed_evaluation_passed": item["consumed_evaluation_gate"][
                    "passed"
                ],
                "consumed_evaluation_reasons": item["consumed_evaluation_gate"][
                    "reasons"
                ],
                "formally_selected": False,
                "selection_contaminated": True,
            }
            for role_key, result_key in (
                ("policy_calibration", "policy_calibration"),
                ("consumed_evaluation", "consumed_evaluation_diagnostic"),
            ):
                result = item[result_key]
                for scenario in ("base", "paired_stress"):
                    metrics = result[scenario]["metrics"]
                    prefix = f"{role_key}_{scenario}"
                    for metric in (
                        "trades",
                        "total_net_bps",
                        "mean_net_bps",
                        "profit_factor",
                        "max_drawdown_bps",
                        "active_days",
                        "trades_per_active_day",
                    ):
                        row[f"{prefix}_{metric}"] = metrics[metric]
                    row[f"{prefix}_symbol_net_bps"] = result[scenario]["symbol_net_bps"]
                    row[f"{prefix}_maximum_positive_pnl_share"] = result[scenario][
                        "maximum_single_symbol_positive_pnl_share"
                    ]
            rows.append(row)
    return rows


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        predictive = report["predictive_gates"][candidate]
        selected = report["selected_policy"][candidate]
        rows.extend(
            (
                {
                    "round": ROUND,
                    "candidate": candidate,
                    "gate": "predictive",
                    "passed": predictive["passed"],
                    "reasons": predictive["reasons"],
                },
                {
                    "round": ROUND,
                    "candidate": candidate,
                    "gate": "policy_calibration",
                    "passed": selected["selection_passed"],
                    "reasons": (
                        []
                        if selected["selection_passed"]
                        else ["no_policy_calibration_variant_passed"]
                    ),
                },
                {
                    "round": ROUND,
                    "candidate": candidate,
                    "gate": "consumed_evaluation",
                    "passed": selected["evaluation_gate_passed"],
                    "reasons": selected["evaluation_reasons"],
                },
            )
        )
    rows.append(
        {
            "round": ROUND,
            "candidate": CANDIDATES[2],
            "gate": "ai_uplift",
            "passed": report["ai_uplift_gate"]["passed"],
            "reasons": report["ai_uplift_gate"]["reasons"],
        }
    )
    return rows


def _ai_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    ai = report["ai_uplift_gate"]
    return [
        {
            "round": ROUND,
            "control": CANDIDATES[1],
            "treatment": CANDIDATES[2],
            "control_average_probability_log_loss": ai[
                "control_average_probability_log_loss"
            ],
            "treatment_average_probability_log_loss": ai[
                "ai_average_probability_log_loss"
            ],
            "probability_log_loss_improvement": ai["probability_log_loss_improvement"],
            "required_probability_log_loss_improvement": 0.005,
            "control_average_expected_payoff_spearman": ai[
                "control_average_expected_payoff_spearman"
            ],
            "treatment_average_expected_payoff_spearman": ai[
                "ai_average_expected_payoff_spearman"
            ],
            "expected_payoff_spearman_improvement": ai[
                "expected_payoff_spearman_improvement"
            ],
            "required_expected_payoff_spearman_improvement": 0.005,
            "passed": ai["passed"],
            "reasons": ai["reasons"],
            "parameter_count": 991437160,
            "runtime": "DirectML cached causal features; no Round 52 rerun",
        }
    ]


def _diagnostic_policy(report: Mapping[str, object]) -> Mapping[str, object]:
    for row in report["policy_grid"][CANDIDATES[1]]:
        if math.isclose(float(row["coverage"]), 0.0025, abs_tol=1e-12):
            return row
    raise ValueError("Round 52 fixed 0.0025 diagnostic policy is absent")


def _daily_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    diagnostic = _diagnostic_policy(report)
    rows: list[dict[str, object]] = []
    for period, result_key in (
        ("policy_calibration", "policy_calibration"),
        ("consumed_evaluation", "consumed_evaluation_diagnostic"),
    ):
        result = diagnostic[result_key]
        for scenario in ("base", "paired_stress"):
            cumulative = 0.0
            for item in result[scenario]["daily_net_bps"]:
                cumulative += float(item["net_bps"])
                day_id = int(item["utc_day_id"])
                date_utc = datetime.fromtimestamp(
                    day_id * 86_400,
                    tz=UTC,
                ).date()
                rows.append(
                    {
                        "round": ROUND,
                        "candidate": CANDIDATES[1],
                        "coverage": 0.0025,
                        "period": period,
                        "scenario": scenario,
                        "utc_date": date_utc.isoformat(),
                        "daily_weighted_net_bps": item["net_bps"],
                        "cumulative_weighted_net_bps": cumulative,
                        "formally_selected": False,
                        "selection_contaminated": True,
                    }
                )
    if not rows:
        raise ValueError("Round 52 daily diagnostic rows are empty")
    return rows


def _progress_rows(
    previous_path: Path,
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[str]]:
    with previous_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    observed = [int(row["round"]) for row in rows]
    if observed == list(range(1, ROUND + 1)):
        rows = rows[:-1]
        observed = observed[:-1]
    if observed != list(range(1, ROUND)):
        raise ValueError("research progress must contain exactly Rounds 1 through 51")
    diagnostic = _diagnostic_policy(report)["consumed_evaluation_diagnostic"]
    stress = diagnostic["paired_stress"]["metrics"]
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "executable-support hurdle + FinCast ablation",
            "periods": "2023-05-16..2023-06-14; eval 2023-06-09..2023-06-14",
            "selection_contaminated": "True",
            "horizon_seconds": "300",
            "feature_set": "118 causal microstructure; +30 cached FinCast treatment",
            "risk_level": "consumed development only; unlevered",
            "spearman_ic": str(
                report["ai_uplift_gate"]["control_average_expected_payoff_spearman"]
            ),
            "selected_signals": "0",
            "executable_trades": "0",
            "status": "rejected",
            "source_file": (
                "verified Round 52 report; positive consumed-evaluation diagnostic "
                "was not selected by calibration"
            ),
            "best_policy_trades": str(stress["trades"]),
            "best_policy_total_net_bps": str(stress["total_net_bps"]),
            "best_policy_mean_net_bps": str(stress["mean_net_bps"]),
            "best_policy_max_drawdown_bps": str(stress["max_drawdown_bps"]),
            "best_policy_profit_factor": str(stress["profit_factor"]),
            "best_model_id": (
                "executable_hurdle_lightgbm_consumed_diagnostic_not_selected"
            ),
            "ensemble_models": "27",
            "valid_barrier_rows": str(
                sum(
                    int(report["data"][symbol]["deterministic_rows"])
                    for symbol in SYMBOLS
                )
            ),
            "calibration_eligible_rows": str(
                _diagnostic_policy(report)["policy_calibration"]["base"]["metrics"][
                    "trades"
                ]
            ),
            "policy_eligible_rows": str(stress["trades"]),
            "development_consumed": "True",
            "architecture_gates_passed": "0",
            "architecture_gate_count": "3",
        }
    )
    rows.append(new)
    return rows, fields


def _svg_start(
    title: str, subtitle: str, *, width: int = 1200, height: int = 700
) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(subtitle)}</desc>",
        f'<rect width="{width}" height="{height}" fill="{COLORS["background"]}"/>',
        f'<text x="64" y="58" fill="{COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="28" font-weight="700">{html.escape(title)}</text>',
        f'<text x="64" y="88" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="15">{html.escape(subtitle)}</text>',
    ]


def _bar_svg(
    *,
    title: str,
    subtitle: str,
    groups: Sequence[tuple[str, Sequence[tuple[str, float | None, str]]]],
    y_min: float,
    y_max: float,
    y_label: str,
    tick_decimals: int = 2,
    value_decimals: int = 3,
    avoid_value_label_collisions: bool = False,
) -> str:
    width, height = 1200, 700
    left, right, top, bottom = 100, 50, 135, 125
    plot_w, plot_h = width - left - right, height - top - bottom
    lines = _svg_start(title, subtitle, width=width, height=height)

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        yy = y(value)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{COLORS["grid"]}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{yy + 5:.1f}" text-anchor="end" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.{tick_decimals}f}</text>'
        )
    if y_min < 0.0 < y_max:
        lines.append(
            f'<line x1="{left}" y1="{y(0):.1f}" x2="{width - right}" y2="{y(0):.1f}" stroke="{COLORS["muted"]}" stroke-width="2"/>'
        )
    group_width = plot_w / len(groups)
    for group_index, (label, bars) in enumerate(groups):
        slot = group_width / (len(bars) + 1)
        placed_labels: list[tuple[float, float, float]] = []
        for bar_index, (_series, value, color) in enumerate(bars, start=1):
            if value is None:
                continue
            numeric_value = float(value)
            x = left + group_index * group_width + bar_index * slot - slot * 0.34
            zero_y = y(0.0)
            value_y = y(numeric_value)
            height_px = abs(zero_y - value_y)
            top_y = min(zero_y, value_y)
            lines.append(
                f'<rect x="{x:.1f}" y="{top_y:.1f}" width="{slot * 0.68:.1f}" height="{max(1.0, height_px):.1f}" rx="3" fill="{color}"/>'
            )
            label_text = f"{numeric_value:.{value_decimals}f}"
            label_x = x + slot * 0.34
            label_y = value_y - 8 if numeric_value >= 0 else value_y + 18
            label_width = max(24.0, len(label_text) * 6.8)
            if avoid_value_label_collisions:
                direction = -1.0 if numeric_value >= 0 else 1.0
                for _attempt in range(len(placed_labels) + 1):
                    collision = any(
                        abs(label_x - prior_x)
                        < (label_width + prior_width) / 2.0 + 4.0
                        and abs(label_y - prior_y) < 14.0
                        for prior_x, prior_y, prior_width in placed_labels
                    )
                    if not collision:
                        break
                    label_y += direction * 14.0
                label_y = min(max(label_y, top + 12.0), top + plot_h - 4.0)
            placed_labels.append((label_x, label_y, label_width))
            lines.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" fill="{COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{label_text}</text>'
            )
        center = left + group_index * group_width + group_width / 2
        lines.append(
            f'<text x="{center:.1f}" y="{height - bottom + 34}" text-anchor="middle" fill="{COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{html.escape(label)}</text>'
        )
    lines.append(
        f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{html.escape(y_label)}</text>'
    )
    legends = groups[0][1]
    legend_slot = plot_w / len(legends)
    for legend_index, (series, _value, color) in enumerate(legends):
        legend_x = left + legend_index * legend_slot
        lines.append(
            f'<rect x="{legend_x}" y="{height - 45}" width="14" height="14" rx="2" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 22}" y="{height - 33}" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{html.escape(series)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    groups = []
    for candidate in CANDIDATES:
        selected = [
            row
            for row in rows
            if row["candidate"] == candidate and row["role"] == "evaluation"
        ]
        mse = (
            100.0
            * sum(float(row["expected_payoff_mse_skill"]) for row in selected)
            / len(selected)
        )
        spearman = (
            100.0
            * sum(float(row["expected_payoff_spearman"]) for row in selected)
            / len(selected)
        )
        probability_rows = [
            row
            for row in selected
            if row.get("probability_log_loss_skill") != ""
            and row.get("probability_log_loss_skill") is not None
        ]
        log_skill = (
            100.0
            * sum(float(row["probability_log_loss_skill"]) for row in probability_rows)
            / len(probability_rows)
            if probability_rows
            else None
        )
        groups.append(
            (
                DISPLAY[candidate],
                (
                    ("Expected-payoff MSE skill", mse, COLORS["red"]),
                    ("Expected-payoff Spearman x100", spearman, COLORS["blue"]),
                    ("Probability log-loss skill", log_skill, COLORS["teal"]),
                ),
            )
        )
    return _bar_svg(
        title="Executable-row forecast quality",
        subtitle="Mean over BTC, ETH, SOL, both sides, and three seeds; evaluation is consumed development data",
        groups=groups,
        y_min=-3.0,
        y_max=10.0,
        y_label="Percent or rank correlation x100",
    )


def _support_svg(rows: Sequence[Mapping[str, object]]) -> str:
    groups = [
        (
            str(row["symbol"]),
            (
                (
                    "Long executable",
                    100.0 * float(row["long_executable_ratio"]),
                    COLORS["teal"],
                ),
                (
                    "Short executable",
                    100.0 * float(row["short_executable_ratio"]),
                    COLORS["blue"],
                ),
            ),
        )
        for row in rows
    ]
    return _bar_svg(
        title="Replay-executable action support",
        subtitle="Exact side-specific mask used in fitting, calibration, scoring, thresholding, and replay",
        groups=groups,
        y_min=0.0,
        y_max=100.0,
        y_label="Executable valid-barrier rows (%)",
    )


def _policy_svg(report: Mapping[str, object]) -> str:
    groups = []
    for candidate in CANDIDATES:
        row = next(
            item
            for item in report["policy_grid"][candidate]
            if math.isclose(float(item["coverage"]), 0.0025, abs_tol=1e-12)
        )
        calibration = row["policy_calibration"]
        evaluation = row["consumed_evaluation_diagnostic"]
        groups.append(
            (
                DISPLAY[candidate],
                (
                    (
                        "Calibration base",
                        calibration["base"]["metrics"]["mean_net_bps"],
                        COLORS["cyan"],
                    ),
                    (
                        "Calibration stress",
                        calibration["paired_stress"]["metrics"]["mean_net_bps"],
                        COLORS["amber"],
                    ),
                    (
                        "Consumed eval base",
                        evaluation["base"]["metrics"]["mean_net_bps"],
                        COLORS["blue"],
                    ),
                    (
                        "Consumed eval stress",
                        evaluation["paired_stress"]["metrics"]["mean_net_bps"],
                        COLORS["red"],
                    ),
                ),
            )
        )
    return _bar_svg(
        title="Calibration did not authorize the later reversal",
        subtitle="Fixed 0.25% coverage; zero-height direct bars are empty ledgers, not zero-return results",
        groups=groups,
        y_min=-11.0,
        y_max=7.0,
        y_label="Weighted mean net return (bps/trade)",
    )


def _ai_svg(report: Mapping[str, object]) -> str:
    ai = report["ai_uplift_gate"]
    groups = [
        (
            "Probability log loss",
            (
                (
                    "Observed improvement",
                    ai["probability_log_loss_improvement"],
                    COLORS["teal"],
                ),
                ("Required improvement", 0.005, COLORS["muted"]),
            ),
        ),
        (
            "Expected-payoff Spearman",
            (
                (
                    "Observed improvement",
                    ai["expected_payoff_spearman_improvement"],
                    COLORS["blue"],
                ),
                ("Required improvement", 0.005, COLORS["muted"]),
            ),
        ),
    ]
    return _bar_svg(
        title="FinCast failed the matched uplift gate",
        subtitle="Causal cached features from the 991,437,160-parameter model; higher is better",
        groups=groups,
        y_min=0.0,
        y_max=0.006,
        y_label="Absolute improvement",
        tick_decimals=4,
        value_decimals=6,
    )


def _line_svg(
    *,
    title: str,
    subtitle: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]],
    x_labels: Mapping[float, str],
    y_label: str,
) -> str:
    width, height = 1200, 700
    left, right, top, bottom = 105, 55, 135, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    all_points = [point for _name, points, _color in series for point in points]
    x_min, x_max = (
        min(point[0] for point in all_points),
        max(point[0] for point in all_points),
    )
    y_values = [point[1] for point in all_points] + [0.0]
    y_min, y_max = min(y_values), max(y_values)
    padding = max(1.0, (y_max - y_min) * 0.12)
    y_min -= padding
    y_max += padding
    lines = _svg_start(title, subtitle, width=width, height=height)

    def x(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1.0) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * plot_h

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        yy = y(value)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" stroke="{COLORS["grid"]}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{yy + 5:.1f}" text-anchor="end" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.1f}</text>'
        )
    for value, label in x_labels.items():
        xx = x(value)
        lines.append(
            f'<text x="{xx:.1f}" y="{height - bottom + 34}" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{html.escape(label)}</text>'
        )
    legend_x = left
    for name, points, color in series:
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x(px):.1f} {y(py):.1f}"
            for index, (px, py) in enumerate(points)
        )
        lines.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>'
        )
        for px, py in points:
            lines.append(
                f'<circle cx="{x(px):.1f}" cy="{y(py):.1f}" r="4" fill="{color}"/>'
            )
        lines.append(
            f'<rect x="{legend_x}" y="{height - 43}" width="16" height="4" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{legend_x + 24}" y="{height - 35}" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{html.escape(name)}</text>'
        )
        legend_x += 250
    lines.append(
        f'<text transform="translate(25 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" fill="{COLORS["subtext"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{html.escape(y_label)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _daily_svg(rows: Sequence[Mapping[str, object]]) -> str:
    dates = sorted({str(row["utc_date"]) for row in rows})
    date_index = {value: float(index) for index, value in enumerate(dates)}
    series = []
    config = (
        ("Calibration base", "policy_calibration", "base", COLORS["cyan"]),
        ("Calibration stress", "policy_calibration", "paired_stress", COLORS["amber"]),
        ("Consumed eval base", "consumed_evaluation", "base", COLORS["blue"]),
        ("Consumed eval stress", "consumed_evaluation", "paired_stress", COLORS["red"]),
    )
    for name, period, scenario, color in config:
        points = [
            (
                date_index[str(row["utc_date"])],
                float(row["cumulative_weighted_net_bps"]),
            )
            for row in rows
            if row["period"] == period and row["scenario"] == scenario
        ]
        series.append((name, points, color))
    labels = {
        float(index): value[5:]
        for index, value in enumerate(dates)
        if index in {0, len(dates) - 1} or index % 2 == 0
    }
    return _line_svg(
        title="Fixed-policy cumulative diagnostic",
        subtitle="Executable hurdle at frozen 0.25% coverage; each role resets to zero and no line was formally selected",
        series=series,
        x_labels=labels,
        y_label="Cumulative weighted net bps",
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points: list[tuple[float, float]] = []
    for row in rows:
        raw = str(row.get("best_policy_mean_net_bps") or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            points.append((float(row["round"]), value))
    labels = {value: str(int(value)) for value in (points[0][0], points[-1][0])}
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        if points[0][0] <= value <= points[-1][0]:
            labels[value] = str(int(value))
    return _line_svg(
        title="Optimization research progression",
        subtitle="Best descriptive net bps/trade by round; datasets and policies differ, all shown values remain non-promotable",
        series=(("Descriptive best policy", points, COLORS["teal"]),),
        x_labels=labels,
        y_label="Net bps per trade",
    )


def _clean_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    expected_parent = (ROOT / "docs" / "model-research" / "action-value").resolve()
    if not resolved.is_relative_to(expected_parent) or resolved.name != "latest":
        raise ValueError("publication output must be action-value/latest")
    if resolved.exists():
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _readme(report: Mapping[str, object]) -> str:
    ai = report["ai_uplift_gate"]
    diagnostic = _diagnostic_policy(report)
    calibration = diagnostic["policy_calibration"]
    evaluation = diagnostic["consumed_evaluation_diagnostic"]
    calibration_base = float(calibration["base"]["metrics"]["mean_net_bps"])
    calibration_stress = float(calibration["paired_stress"]["metrics"]["mean_net_bps"])
    evaluation_base = float(evaluation["base"]["metrics"]["mean_net_bps"])
    evaluation_stress = float(evaluation["paired_stress"]["metrics"]["mean_net_bps"])
    return f"""# Round 52: Executable-Support Hurdle

> **Rejected consumed-development screen.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 52 corrected the measured Round 51 target-policy mismatch: training, early stopping, calibration, thresholds, scoring, and replay now use one hash-bound side-specific executable predicate. It trained 27 OpenCL LightGBM models on verified Binance USD-M BTC, ETH, and SOL tick data and reused the sealed causal FinCast feature matrices.

The correction restored activity, but calibration rejected every policy. The deterministic hurdle produced {int(calibration["base"]["metrics"]["trades"])} calibration trades at `{calibration_base:.6f}` base and `{calibration_stress:.6f}` stressed bps/trade. The later consumed interval showed {int(evaluation["base"]["metrics"]["trades"])} trades at `+{evaluation_base:.6f}` base and `+{evaluation_stress:.6f}` stressed bps/trade, but that reversal was not authorized by calibration and cannot be selected. The direct model remained flat. FinCast produced 3 negative calibration trades and 7 negative consumed-evaluation trades.

Profitable-event classification improved over training prevalence, while expected-payoff magnitude did not: mean evaluation expected-payoff MSE skill was negative for every architecture. FinCast improved average probability log loss by `{float(ai["probability_log_loss_improvement"]):.6f}` and expected-payoff Spearman by `{float(ai["expected_payoff_spearman_improvement"]):.6f}`, below both frozen `0.005` gates.

## Evidence

| View | Graph | Source |
|---|---|---|
| Executable support | [SVG](charts/executable-support.svg) | [CSV](support.csv) |
| Forecast quality | [SVG](charts/forecast-quality.svg) | [CSV](forecast.csv) |
| Policy economics | [SVG](charts/policy-economics.svg) | [CSV](policy-grid.csv) |
| Fixed-policy daily path | [SVG](charts/daily-equity.svg) | [CSV](daily-policy.csv) |
| FinCast uplift | [SVG](charts/ai-uplift.svg) | [CSV](ai-uplift.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`screen.json` is the complete source report. `report.json` binds this publication to the frozen design, execution binding, external report, and every published file. Model and prediction artifact hashes are recorded in [models.csv](models.csv) and `screen.json`; flattened gate outcomes are in [gates.csv](gates.csv).
"""


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def publish(
    *,
    report_path: Path,
    design_path: Path,
    binding_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, design, binding = _validate_source(
        report_path=report_path,
        design_path=design_path,
        binding_path=binding_path,
    )
    progress_rows, progress_fields = _progress_rows(previous_progress_path, report)
    support_rows = _support_rows(report)
    forecast_rows = _forecast_rows(report)
    model_rows = _model_rows(report)
    policy_rows = _policy_rows(report)
    gate_rows = _gate_rows(report)
    ai_rows = _ai_rows(report)
    daily_rows = _daily_rows(report)
    _clean_output(output_dir)
    charts = output_dir / "charts"

    _write_csv(output_dir / "support.csv", support_rows)
    _write_csv(output_dir / "forecast.csv", forecast_rows)
    _write_csv(output_dir / "models.csv", model_rows)
    _write_csv(output_dir / "policy-grid.csv", policy_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(output_dir / "ai-uplift.csv", ai_rows)
    _write_csv(output_dir / "daily-policy.csv", daily_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "executable-support.svg", _support_svg(support_rows))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(forecast_rows))
    _write_text(charts / "policy-economics.svg", _policy_svg(report))
    _write_text(charts / "daily-equity.svg", _daily_svg(daily_rows))
    _write_text(charts / "ai-uplift.svg", _ai_svg(report))
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))
    _write_text(output_dir / "README.md", _readme(report))
    from simple_ai_trading.storage import write_json_atomic

    write_json_atomic(output_dir / "screen.json", report, indent=2, sort_keys=True)

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    diagnostic = _diagnostic_policy(report)
    calibration = diagnostic["policy_calibration"]
    evaluation = diagnostic["consumed_evaluation_diagnostic"]
    publication: dict[str, object] = {
        "schema_version": "round-052-action-value-publication-v1",
        "round": ROUND,
        "published_at": report["generated_at"],
        "publisher_path": "tools/publish_round52_executable_support_hurdle.py",
        "source": {
            "report_path": str(report_path.resolve()),
            "report_file_sha256": SOURCE_REPORT_FILE_SHA256,
            "report_canonical_sha256": SOURCE_REPORT_CANONICAL_SHA256,
            "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
            "design_sha256": DESIGN_SHA256,
            "binding_path": str(binding_path.relative_to(ROOT)).replace("\\", "/"),
            "binding_sha256": BINDING_SHA256,
            "implementation_commit": binding["implementation_commit"],
        },
        "claims": {
            "status": "rejected",
            "selection_contaminated": True,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
            "untouched_data_expansion_authorized": False,
        },
        "result": {
            "models": 27,
            "prediction_artifacts": 54,
            "formally_selected_policies": 0,
            "passed_mechanisms": 0,
            "deterministic_hurdle_calibration_trades": calibration["base"]["metrics"][
                "trades"
            ],
            "deterministic_hurdle_calibration_base_mean_net_bps": calibration["base"][
                "metrics"
            ]["mean_net_bps"],
            "deterministic_hurdle_calibration_stress_mean_net_bps": calibration[
                "paired_stress"
            ]["metrics"]["mean_net_bps"],
            "deterministic_hurdle_consumed_evaluation_trades": evaluation["base"][
                "metrics"
            ]["trades"],
            "deterministic_hurdle_consumed_evaluation_base_mean_net_bps": evaluation[
                "base"
            ]["metrics"]["mean_net_bps"],
            "deterministic_hurdle_consumed_evaluation_stress_mean_net_bps": evaluation[
                "paired_stress"
            ]["metrics"]["mean_net_bps"],
            "ai_probability_log_loss_improvement": report["ai_uplift_gate"][
                "probability_log_loss_improvement"
            ],
            "ai_expected_payoff_spearman_improvement": report["ai_uplift_gate"][
                "expected_payoff_spearman_improvement"
            ],
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2, sort_keys=True)
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round52-executable-support-hurdle-20260713-v1\report.json"
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-052-executable-support-hurdle-fincast-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-052-execution-binding.json",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=research / "latest" / "progress.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=research / "latest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = publish(
        report_path=args.report.resolve(),
        design_path=args.design.resolve(),
        binding_path=args.binding.resolve(),
        previous_progress_path=args.progress.resolve(),
        output_dir=args.output.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": publication["round"],
                "publication_canonical_sha256": publication[
                    "publication_canonical_sha256"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
