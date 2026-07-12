"""Publish verified Round 32 evidence, tables, and deterministic charts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from tools.publish_daily_walkforward_screen import _progress_svg
    from tools.run_shared_action_viability import (
        BINDING_SCHEMA_VERSION,
        REPORT_SCHEMA_VERSION,
        _canonical_sha256,
        _is_sha256,
        _sha256_file,
        load_round32_design,
    )
except ModuleNotFoundError:  # pragma: no cover - direct tools execution
    from publish_daily_walkforward_screen import _progress_svg
    from run_shared_action_viability import (
        BINDING_SCHEMA_VERSION,
        REPORT_SCHEMA_VERSION,
        _canonical_sha256,
        _is_sha256,
        _sha256_file,
        load_round32_design,
    )


PUBLICATION_SCHEMA_VERSION = "shared-action-viability-publication-v1"
_FALSE_CLAIMS = (
    "trading_authority",
    "execution_claim",
    "profitability_claim",
    "portfolio_claim",
    "leverage_applied",
)
_STAGES = ("calibration", "policy", "development", "distant_confirmation")


def _read_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _validate_tree(value: object, *, label: str = "report") -> None:
    if isinstance(value, Mapping):
        for name in _FALSE_CLAIMS:
            if name in value and value[name] is not False:
                raise ValueError(f"{label} contains an unauthorized {name}")
        for name, child in value.items():
            _validate_tree(child, label=f"{label}.{name}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_tree(child, label=f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")


def _validated_source(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], str]:
    design, design_sha, _profiles = load_round32_design(design_path)
    binding = _read_object(binding_path, label="Round 32 execution binding")
    binding_canonical = dict(binding)
    binding_sha = binding_canonical.pop("binding_sha256", None)
    if (
        binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or not _is_sha256(binding_sha)
        or _canonical_sha256(binding_canonical) != binding_sha
    ):
        raise ValueError("Round 32 execution binding hash is invalid")
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, label="Round 32 report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 32
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or not _is_sha256(claimed)
        or _canonical_sha256(canonical) != claimed
        or report.get("all_target_dates_previously_consumed") is not True
        or report.get("untouched_policy_windows_accessed") is not False
    ):
        raise ValueError("Round 32 report identity is invalid")
    _validate_tree(report)
    access = report.get("stage_access")
    survivors = report.get("stage_survivors")
    if (
        not isinstance(access, Mapping)
        or access.get("calibration_prediction") is not True
        or access.get("policy_prediction") is not False
        or access.get("development_prediction") is not False
        or access.get("distant_confirmation_prediction") is not False
        or access.get("later_stage_predictions_withheld_until_prior_stage_passes")
        is not True
        or not isinstance(survivors, Mapping)
        or any(survivors.get(name) != [] for name in survivors)
        or report.get("distant_corpus") is not None
    ):
        raise ValueError("Round 32 nested stage access is invalid")
    implementation = report.get("implementation_gates")
    models = report.get("ensemble_models")
    profiles = report.get("profile_results")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("status") != "pass"
        or implementation.get("artifact_reload_equivalence_passed") is not True
        or implementation.get("nonfinite_prediction_count") != 0
        or not isinstance(models, list)
        or len(models) != 3
        or not isinstance(profiles, list)
        or [item.get("profile") for item in profiles if isinstance(item, Mapping)]
        != ["conservative", "regular", "aggressive"]
    ):
        raise ValueError("Round 32 implementation or profile evidence is incomplete")
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("Round 32 model evidence is invalid")
        relative = Path(str(model.get("artifact_path") or ""))
        artifact = evidence_root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not artifact.is_file()
            or artifact.stat().st_size != int(model.get("artifact_bytes") or -1)
            or _sha256_file(artifact) != model.get("artifact_sha256")
            or model.get("backend_kind") != "opencl"
            or model.get("artifact_reload_max_abs_prediction_error") != 0.0
            or model.get("action_swap_equivariance_max_abs_error") != 0.0
        ):
            raise ValueError("Round 32 model artifact evidence differs")
    eligible = []
    candidate_count = 0
    for profile in profiles:
        assert isinstance(profile, Mapping)
        calibration = profile.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError("Round 32 calibration result is missing")
        eligible.append(int(calibration.get("eligible_rows") or 0))
        threshold = calibration.get("threshold_selection")
        if not isinstance(threshold, Mapping) or threshold.get("accepted") is not False:
            raise ValueError("Round 32 threshold acceptance drifted")
        candidates = threshold.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Round 32 threshold candidates are missing")
        candidate_count += len(candidates)
        if profile.get("final_status") != "rejected":
            raise ValueError("Round 32 final profile status changed")
    if eligible != [0, 0, 10] or candidate_count != 4:
        raise ValueError("Round 32 calibration eligibility differs")
    return design, report, str(claimed)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _stage_rows(
    design: Mapping[str, object], report: Mapping[str, object]
) -> list[dict[str, object]]:
    roles = design["data"]["roles"]
    role_evidence = report["training_corpus"]["roles"]
    profiles = report["profile_results"]
    assert isinstance(roles, Mapping)
    assert isinstance(role_evidence, Mapping)
    assert isinstance(profiles, list)
    stage_survivors = report["stage_survivors"]
    assert isinstance(stage_survivors, Mapping)
    output: list[dict[str, object]] = []
    for stage in _STAGES:
        role = roles[stage]
        assert isinstance(role, Mapping)
        evidence = role_evidence.get(stage)
        opened = stage == "calibration"
        output.append(
            {
                "stage": stage,
                "evaluation_start": role["start"],
                "evaluation_end": role["end"],
                "prediction_opened": opened,
                "profile_count": len(profiles) if opened else 0,
                "survivor_count": len(stage_survivors.get(stage) or []),
                "valid_role_rows": (
                    int(evidence["rows"])
                    if isinstance(evidence, Mapping)
                    else 0
                ),
                "withheld_reason": (
                    "" if opened else "no_calibration_profile_passed"
                ),
            }
        )
    return output


def _profile_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for profile in report["profile_results"]:
        assert isinstance(profile, Mapping)
        calibration = profile["calibration"]
        assert isinstance(calibration, Mapping)
        threshold = calibration["threshold_selection"]
        assert isinstance(threshold, Mapping)
        output.append(
            {
                "profile": profile["profile"],
                "calibration_rows": 28_581,
                "eligible_rows": calibration["eligible_rows"],
                "threshold_candidate_count": len(threshold["candidates"]),
                "accepted_threshold_count": int(bool(threshold["accepted"])),
                "calibration_status": calibration["status"],
                "policy_prediction_opened": False,
                "development_prediction_opened": False,
                "distant_confirmation_opened": False,
                "final_status": profile["final_status"],
            }
        )
    return output


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for profile in report["profile_results"]:
        assert isinstance(profile, Mapping)
        calibration = profile["calibration"]
        assert isinstance(calibration, Mapping)
        selection = calibration["threshold_selection"]
        assert isinstance(selection, Mapping)
        for candidate in selection["candidates"]:
            assert isinstance(candidate, Mapping)
            stress = candidate["stress_metrics"]
            base = candidate["base_metrics"]
            assert isinstance(stress, Mapping)
            assert isinstance(base, Mapping)
            output.append(
                {
                    "profile": profile["profile"],
                    "quantile": candidate["quantile"],
                    "threshold_bps": candidate["threshold_bps"],
                    "stress_trades": stress["trades"],
                    "stress_total_net_bps": stress["total_net_bps"],
                    "stress_mean_net_bps": stress["mean_net_bps"],
                    "stress_max_drawdown_bps": stress["max_drawdown_bps"],
                    "stress_worst_trade_bps": stress["worst_trade_bps"],
                    "stress_profit_factor": stress["profit_factor"],
                    "stress_positive_day_ratio": candidate[
                        "stress_positive_day_ratio"
                    ],
                    "base_total_net_bps": base["total_net_bps"],
                    "accepted": candidate["accepted"],
                    "rejection_reasons": ";".join(candidate["rejection_reasons"]),
                }
            )
    return output


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    diagnostics = report["forecast_diagnostics_by_role"]["calibration"]
    stress = diagnostics["stress"]["sides"]
    selected = diagnostics["selected_action"]
    assert isinstance(stress, Mapping)
    assert isinstance(selected, Mapping)
    output: list[dict[str, object]] = []
    for side in ("long", "short"):
        value = stress[side]
        assert isinstance(value, Mapping)
        top = value["top_rows"]
        assert isinstance(top, list)
        output.append(
            {
                "stage": "calibration",
                "evaluation_start": "2023-06-21",
                "evaluation_end": "2023-06-25",
                "diagnostic": side,
                "rows": value["rows"],
                "profitable_auc": value["profitable_auc"],
                "pearson_ic": value["pearson_information_coefficient"],
                "spearman_ic": value["spearman_information_coefficient"],
                "mean_actual_stress_net_bps": value["mean_actual_net_bps"],
                "top_100_mean_stress_net_bps": top[0]["mean_actual_net_bps"],
                "top_500_mean_stress_net_bps": top[1]["mean_actual_net_bps"],
                "long_share_top_500": "",
            }
        )
    selected_top = selected["top_rows"]
    assert isinstance(selected_top, Mapping)
    output.append(
        {
            "stage": "calibration",
            "evaluation_start": "2023-06-21",
            "evaluation_end": "2023-06-25",
            "diagnostic": "selected_action",
            "rows": selected["rows"],
            "profitable_auc": selected["side_choice_auc"],
            "pearson_ic": selected[
                "selected_action_pearson_information_coefficient"
            ],
            "spearman_ic": selected[
                "selected_action_spearman_information_coefficient"
            ],
            "mean_actual_stress_net_bps": selected[
                "mean_selected_stress_net_bps"
            ],
            "top_100_mean_stress_net_bps": selected_top["100"][
                "mean_stress_net_bps"
            ],
            "top_500_mean_stress_net_bps": selected_top["500"][
                "mean_stress_net_bps"
            ],
            "long_share_top_500": selected_top["500"]["long_share"],
        }
    )
    return output


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "seed": model["seed"],
            "model_sha256": model["model_sha256"],
            "artifact_sha256": model["artifact_sha256"],
            "artifact_bytes": model["artifact_bytes"],
            "training_backend": model["backend_kind"],
            "training_event_rows": model["training_event_rows"],
            "early_stop_event_rows": model["early_stop_event_rows"],
            "probability_calibration_event_rows": model[
                "probability_calibration_event_rows"
            ],
            "internal_purged_event_rows": model["internal_purged_event_rows"],
            "positive_class_prevalence": model["positive_class_prevalence"],
            "advantage_directional_loss": model[
                "advantage_validation_directional_loss"
            ],
            "reload_max_abs_error": model[
                "artifact_reload_max_abs_prediction_error"
            ],
            "equivariance_max_abs_error": model[
                "action_swap_equivariance_max_abs_error"
            ],
        }
        for model in report["ensemble_models"]
    ]


def _progress_rows(
    path: Path,
    report: Mapping[str, object],
    forecasts: Sequence[Mapping[str, object]],
    thresholds: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields or not rows:
        raise ValueError("prior action-value progress is empty")
    rows = [row for row in rows if int(row["round"]) != 32]
    if [int(row["round"]) for row in rows] != list(range(1, 32)):
        raise ValueError("prior action-value progress sequence changed")
    selected = next(row for row in forecasts if row["diagnostic"] == "selected_action")
    least_bad = max(
        thresholds, key=lambda row: float(row["stress_total_net_bps"])
    )
    update = {name: "" for name in fields}
    update.update(
        {
            "round": 32,
            "stage": "shared symmetric action-value calibration",
            "periods": "2023-05-16..2023-07-06",
            "selection_contaminated": True,
            "horizon_seconds": 900,
            "feature_set": "l1-tape-causal-v8 shared long/short canonicalization",
            "risk_level": "conservative;regular;aggressive",
            "direction_auc": selected["profitable_auc"],
            "spearman_ic": selected["spearman_ic"],
            "selected_signals": 10,
            "executable_trades": least_bad["stress_trades"],
            "status": "rejected",
            "source_file": "verified Round 32 report",
            "best_model_id": "three-seed shared-action LightGBM hurdle ensemble",
            "best_top_500_exact_after_cost_bps": selected[
                "top_500_mean_stress_net_bps"
            ],
            "after_cost_diagnostic_rows": selected["rows"],
            "calibration_threshold_traces": len(thresholds),
            "accepted_thresholds": 0,
            "least_bad_calibration_total_net_bps": least_bad[
                "stress_total_net_bps"
            ],
            "ensemble_models": 3,
            "valid_barrier_rows": report["training_corpus"][
                "valid_barrier_rows"
            ],
            "calibration_eligible_rows": 10,
            "development_consumed": False,
        }
    )
    rows.append(update)
    return rows, fields


def _stage_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1320, 430
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 32 chronological stage access</title>',
        '<desc id="desc">Calibration opened; policy, development, and distant confirmation remained withheld after all profiles failed.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Later predictions remained behind risk gates</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53636f">Green means predicted and evaluated; red means withheld. All dates are UTC.</text>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#8998a3"/></marker></defs>',
    ]
    for index, row in enumerate(rows):
        x = 48 + index * 315
        opened = bool(row["prediction_opened"])
        fill = "#e7f4ef" if opened else "#fff0ed"
        stroke = "#147d70" if opened else "#bc493e"
        lines.extend(
            [
                f'<rect x="{x}" y="125" width="275" height="190" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                f'<text x="{x + 20}" y="164" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#24343e">{html.escape(str(row["stage"]).replace("_", " ").title())}</text>',
                f'<text x="{x + 20}" y="194" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="{stroke}">{"opened" if opened else "withheld"}</text>',
                f'<text x="{x + 20}" y="227" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#455763">{row["evaluation_start"]} to {row["evaluation_end"]}</text>',
                f'<text x="{x + 20}" y="258" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#455763">{int(row["survivor_count"])} profiles survived</text>',
                f'<text x="{x + 20}" y="287" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#667681">{int(row["valid_role_rows"]):,} valid role rows</text>',
            ]
        )
        if index < len(rows) - 1:
            lines.append(
                f'<path d="M {x + 275} 220 L {x + 305} 220" stroke="#8998a3" stroke-width="3" marker-end="url(#arrow)"/>'
            )
    lines.extend(
        [
            '<text x="48" y="382" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#62727d">No untouched date, leverage, live execution, or trading authority was accessed.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _eligibility_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 980, 500
    maximum = max(20, max(int(row["eligible_rows"]) for row in rows))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 32 calibration eligibility</title>',
        '<desc id="desc">Eligible event counts after probability, uncertainty, lower-tail, ensemble, and independent direction-consensus gates.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Risk controls suppressed unsupported actions</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Calibration: 2023-06-21 to 2023-06-25 UTC; 28,581 event rows.</text>',
        '<rect x="130" y="125" width="760" height="270" fill="#ffffff" stroke="#d7e0e6"/>',
    ]
    colors = ("#147d70", "#3d6f99", "#bd6645")
    slot = 760 / len(rows)
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        value = int(row["eligible_rows"])
        x = 130 + slot * (index + 0.5)
        bar_height = 230 * value / maximum
        y = 365 - bar_height
        lines.extend(
            [
                f'<rect x="{x - 58:.1f}" y="{y:.1f}" width="116" height="{max(2.0, bar_height):.1f}" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#263640">{value}</text>',
                f'<text x="{x:.1f}" y="426" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#344750">{html.escape(str(row["profile"]).title())}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="48" y="470" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">Eligibility is not a trade count; all profiles still failed threshold-selection risk gates.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _threshold_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1140, 590
    left, top, chart_width, chart_height = 105, 135, 950, 300
    values = [float(row["stress_total_net_bps"]) for row in rows]
    lower, upper = min(-80.0, min(values) - 10.0), max(80.0, max(values) + 10.0)

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    zero = y(0.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 32 calibration threshold economics</title>',
        '<desc id="desc">Adverse-stress total net basis points for all four aggressive-profile threshold candidates, including trade count and drawdown.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Sparse positive tails did not establish an edge</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Adverse-stress net P&amp;L after spread, latency, fees, slippage, and barrier execution.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d7e0e6"/>',
    ]
    for tick in (-60, -30, 0, 30, 60):
        yy = y(float(tick))
        lines.extend(
            [
                f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#{"687985" if tick == 0 else "e5ebef"}" stroke-width="{2 if tick == 0 else 1}"/>',
                f'<text x="{left - 14}" y="{yy + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#61717c">{tick:+d}</text>',
            ]
        )
    slot = chart_width / len(rows)
    for index, row in enumerate(rows):
        value = float(row["stress_total_net_bps"])
        x = left + slot * (index + 0.5)
        yy = y(value)
        lines.extend(
            [
                f'<rect x="{x - 43:.1f}" y="{min(yy, zero):.1f}" width="86" height="{max(2.0, abs(yy - zero)):.1f}" fill="#bc493e"/>',
                f'<text x="{x:.1f}" y="{yy - 10 if value >= 0 else yy + 20:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263640">{value:+.2f}</text>',
                f'<text x="{x:.1f}" y="468" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#344750">q{int(round(float(row["quantile"]) * 100))}</text>',
                f'<text x="{x:.1f}" y="491" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#5d6d78">{int(row["stress_trades"])} trades</text>',
                f'<text x="{x:.1f}" y="511" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#5d6d78">DD {float(row["stress_max_drawdown_bps"]):.2f} bps</text>',
            ]
        )
    lines.extend(
        [
            '<text x="35" y="290" transform="rotate(-90 35 290)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Total net basis points</text>',
            '<text x="48" y="560" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">q85 and q95 were positive only over two and one simulated trades; both failed minimum-support and positive-day gates.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1260, 590
    selected = next(row for row in rows if row["diagnostic"] == "selected_action")
    sides = [row for row in rows if row["diagnostic"] in {"long", "short"}]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 32 calibration forecast quality</title>',
        '<desc id="desc">Probability discrimination and selected-action ranked-tail economics on the calibration period.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Side classifiers did not produce tradable ranked tails</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Calibration: 2023-06-21 to 2023-06-25 UTC. Forecast metrics are not profitability claims.</text>',
        '<rect x="70" y="125" width="500" height="310" fill="#ffffff" stroke="#d7e0e6"/>',
        '<rect x="650" y="125" width="540" height="310" fill="#ffffff" stroke="#d7e0e6"/>',
        '<text x="90" y="158" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">Probability-of-profit ROC AUC</text>',
        '<text x="670" y="158" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">Selected-action stress mean</text>',
    ]
    auc_rows = [*sides, selected]
    colors = ("#147d70", "#75569a", "#bd6645")
    for index, (row, color) in enumerate(zip(auc_rows, colors, strict=True)):
        value = float(row["profitable_auc"])
        x = 115 + index * 145
        bar_height = max(0.0, (value - 0.45) / 0.25 * 215)
        y = 390 - bar_height
        label = str(row["diagnostic"]).replace("_", " ").title()
        lines.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="86" height="{max(2.0, bar_height):.1f}" fill="{color}"/>',
                f'<text x="{x + 43}" y="{y - 9:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263640">{value:.3f}</text>',
                f'<text x="{x + 43}" y="418" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#344750">{html.escape(label)}</text>',
            ]
        )
    tail_values = (
        ("Top 100", float(selected["top_100_mean_stress_net_bps"])),
        ("Top 500", float(selected["top_500_mean_stress_net_bps"])),
        ("All", float(selected["mean_actual_stress_net_bps"])),
    )
    for index, (label, value) in enumerate(tail_values):
        y = 205 + index * 78
        width_value = min(420.0, abs(value) / 15.0 * 420.0)
        lines.extend(
            [
                f'<text x="670" y="{y + 19}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#344750">{label}</text>',
                f'<rect x="760" y="{y}" width="{width_value:.1f}" height="30" fill="#bc493e"/>',
                f'<text x="{770 + width_value:.1f}" y="{y + 21}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263640">{value:+.2f} bps</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="70" y="500" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Selected side AUC {float(selected["profitable_auc"]):.4f}; Pearson IC {float(selected["pearson_ic"]):+.4f}; Spearman IC {float(selected["spearman_ic"]):+.4f}.</text>',
            '<text x="70" y="532" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">Positive long classification did not offset weak side choice, negative rank correlation, or after-cost tail losses.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _artifact(path: Path, root: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            value["row_count"] = sum(1 for _row in csv.DictReader(handle))
    return value


def publish(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    design, report, source_report_sha = _validated_source(
        evidence_root=evidence_root,
        design_path=design_path,
        binding_path=binding_path,
    )
    stages = _stage_rows(design, report)
    profiles = _profile_rows(report)
    thresholds = _threshold_rows(report)
    forecasts = _forecast_rows(report)
    models = _model_rows(report)
    progress, progress_fields = _progress_rows(
        prior_progress_path, report, forecasts, thresholds
    )
    charts = output_dir / "charts"
    _write_csv(output_dir / "stages.csv", stages)
    _write_csv(output_dir / "profiles.csv", profiles)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "forecast.csv", forecasts)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(charts / "stage-access.svg", _stage_svg(stages))
    _write_text(charts / "eligibility.svg", _eligibility_svg(profiles))
    _write_text(charts / "threshold-economics.svg", _threshold_svg(thresholds))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(forecasts))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    stale_candidate_file = output_dir / "candidates.csv"
    if stale_candidate_file.exists():
        stale_candidate_file.unlink()
    stale_chart = charts / "candidate-economics.svg"
    if stale_chart.exists():
        stale_chart.unlink()
    selected = next(row for row in forecasts if row["diagnostic"] == "selected_action")
    least_bad = max(
        thresholds, key=lambda row: float(row["stress_total_net_bps"])
    )
    readme = f"""# Round 32: shared-action calibration rejected

**Rejected without trading authority.** A three-seed, symmetric long/short LightGBM ensemble trained on official BTCUSDT top-of-book and trade events. All profiles failed the first economic gate, so policy, development, and distant-confirmation predictions stayed withheld.

| Evidence | Verified result |
| --- | ---: |
| Source window | 2023-05-16 to 2023-07-06 UTC |
| Causal one-second rows | {int(report['training_corpus']['dataset']['rows']):,} |
| CUSUM events / valid barrier outcomes | {int(report['training_corpus']['event_rows']):,} / {int(report['training_corpus']['valid_barrier_rows']):,} |
| Train / early-stop / calibration rows | 128,307 / 21,934 / 28,581 |
| Eligible rows: conservative / regular / aggressive | 0 / 0 / 10 |
| Selected-side AUC / Spearman IC | {float(selected['profitable_auc']):.4f} / {float(selected['spearman_ic']):+.4f} |
| Top-100 / top-500 stress mean | {float(selected['top_100_mean_stress_net_bps']):+.2f} / {float(selected['top_500_mean_stress_net_bps']):+.2f} bps |
| Highest calibration total (insufficient support) | q{int(round(float(least_bad['quantile']) * 100))}: {float(least_bad['stress_total_net_bps']):+.2f} bps over {int(least_bad['stress_trades'])} trade |
| Final profiles | none |

![Stage access](charts/stage-access.svg)

![Calibration eligibility](charts/eligibility.svg)

![Threshold economics](charts/threshold-economics.svg)

![Forecast quality](charts/forecast-quality.svg)

![Research progress](charts/research-progress.svg)

The positive q85 and q95 totals came from only two and one simulated trades. They failed minimum-support and positive-day gates and are not evidence of profitability. DirectML tensor execution and OpenCL FP64 LightGBM training were attested; LightGBM prediction used its CPU path. No leverage, live execution, portfolio claim, or untouched-period claim is permitted.

Data: [stages.csv](stages.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [forecast.csv](forecast.csv) | [models.csv](models.csv) | [progress.csv](progress.csv) | [integrity report](report.json)
"""
    _write_text(output_dir / "README.md", readme)
    artifact_paths = [
        output_dir / "README.md",
        output_dir / "stages.csv",
        output_dir / "profiles.csv",
        output_dir / "thresholds.csv",
        output_dir / "forecast.csv",
        output_dir / "models.csv",
        output_dir / "progress.csv",
        charts / "stage-access.svg",
        charts / "eligibility.svg",
        charts / "threshold-economics.svg",
        charts / "forecast-quality.svg",
        charts / "research-progress.svg",
    ]
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact_class": "shared_action_viability_graph_data",
        "round": 32,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": report["binding_sha256"],
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "stage_access": {
            "calibration": True,
            "policy": False,
            "development": False,
            "distant_confirmation": False,
        },
        "final_profiles": [],
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "artifact_integrity": [_artifact(path, output_dir) for path in artifact_paths],
        "publication_sha256": "PENDING",
    }
    canonical = dict(publication)
    canonical.pop("publication_sha256")
    publication["publication_sha256"] = _canonical_sha256(canonical)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish verified Round 32 evidence.")
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-032-shared-action-value-viability-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-032-execution-binding.json",
    )
    parser.add_argument(
        "--prior-progress",
        type=Path,
        default=research / "latest" / "progress.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=research / "latest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        evidence_root=arguments.evidence_root,
        design_path=arguments.design,
        binding_path=arguments.binding,
        prior_progress_path=arguments.prior_progress,
        output_dir=arguments.output_dir,
    )
    print(
        f"Published Round 32 status={publication['status']} "
        f"publication_sha256={publication['publication_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
