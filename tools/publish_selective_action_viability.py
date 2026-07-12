"""Publish verified Round 33 tables and deterministic financial charts."""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.publish_daily_walkforward_screen import _progress_svg  # noqa: E402
from tools.run_selective_action_viability import (  # noqa: E402
    BINDING_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    _canonical_sha256,
    _is_sha256,
    _sha256_file,
    load_round33_design,
)


PUBLICATION_SCHEMA_VERSION = "selective-action-viability-publication-v1"
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
                raise ValueError(f"{label} contains unauthorized {name}")
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
    design, design_sha, _profiles = load_round33_design(design_path)
    binding = _read_object(binding_path, label="Round 33 execution binding")
    binding_canonical = dict(binding)
    binding_sha = binding_canonical.pop("binding_sha256", None)
    if (
        binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or not _is_sha256(binding_sha)
        or _canonical_sha256(binding_canonical) != binding_sha
    ):
        raise ValueError("Round 33 execution binding hash is invalid")
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, label="Round 33 report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 33
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or not _is_sha256(claimed)
        or _canonical_sha256(canonical) != claimed
        or report.get("all_target_dates_previously_consumed") is not True
        or report.get("untouched_policy_windows_accessed") is not False
    ):
        raise ValueError("Round 33 report identity is invalid")
    _validate_tree(report)
    access = report.get("stage_access")
    survivors = report.get("stage_survivors")
    if (
        access
        != {
            "calibration_prediction": True,
            "calibration_threshold_selection": False,
            "policy_prediction": False,
            "development_prediction": False,
            "distant_confirmation_prediction": False,
            "later_stage_predictions_withheld_until_prior_stage_passes": True,
        }
        or not isinstance(survivors, Mapping)
        or survivors.get("calibration_architecture") is not False
        or any(
            survivors.get(name) != []
            for name in (
                "calibration",
                "policy",
                "development",
                "distant_trace",
                "final",
            )
        )
        or report.get("distant_corpus") is not None
    ):
        raise ValueError("Round 33 nested-stage access is invalid")
    implementation = report.get("implementation_gates")
    models = report.get("ensemble_models")
    profiles = report.get("profile_results")
    architecture = report.get("calibration_architecture")
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
        or not isinstance(architecture, Mapping)
        or architecture.get("status") != "rejected"
        or architecture.get("rejection_reasons")
        != [
            "conditional_direction_auc_gate_failed",
            "selected_top_100_mean_stress_net_gate_failed",
            "selected_top_500_mean_stress_net_gate_failed",
        ]
    ):
        raise ValueError("Round 33 implementation or rejection evidence is incomplete")
    symmetry_limits = {
        "opportunity_mirror_invariance_max_abs_error": float(
            implementation[
                "maximum_permitted_opportunity_mirror_invariance_error"
            ]
        ),
        "direction_mirror_complementarity_max_abs_error": float(
            implementation[
                "maximum_permitted_direction_mirror_complementarity_error"
            ]
        ),
        "action_swap_equivariance_max_abs_error": float(
            implementation["maximum_permitted_action_swap_equivariance_error"]
        ),
    }
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("Round 33 model evidence is invalid")
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
            or any(
                not math.isfinite(float(model[name]))
                or float(model[name]) > limit
                for name, limit in symmetry_limits.items()
            )
        ):
            raise ValueError("Round 33 model artifact evidence differs")
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("Round 33 profile evidence is invalid")
        calibration = profile.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or calibration.get("eligible_rows") != 0
            or calibration.get("architecture_gate_passed") is not False
            or calibration.get("threshold_selection") is not None
            or calibration.get("threshold_withheld_reason")
            != "calibration_architecture_rejected"
            or profile.get("final_status") != "rejected"
        ):
            raise ValueError("Round 33 profile rejection evidence changed")
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
    design: Mapping[str, object],
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    roles = design["data"]["roles"]
    evidence = report["training_corpus"]["roles"]
    access = report["stage_access"]
    assert isinstance(roles, Mapping)
    assert isinstance(evidence, Mapping)
    assert isinstance(access, Mapping)
    output: list[dict[str, object]] = []
    for stage in _STAGES:
        role = roles[stage]
        role_evidence = evidence.get(stage)
        assert isinstance(role, Mapping)
        opened = bool(access[f"{stage}_prediction"])
        output.append(
            {
                "stage": stage,
                "evaluation_start": role["start"],
                "evaluation_end": role["end"],
                "prediction_opened": opened,
                "survivor_count": 0,
                "valid_role_rows": (
                    int(role_evidence["rows"])
                    if isinstance(role_evidence, Mapping)
                    else 0
                ),
                "withheld_reason": (
                    ""
                    if opened
                    else "calibration_architecture_rejected"
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
        output.append(
            {
                "profile": profile["profile"],
                "calibration_rows": 28_581,
                "eligible_rows": calibration["eligible_rows"],
                "architecture_gate_passed": calibration[
                    "architecture_gate_passed"
                ],
                "threshold_selection_evaluated": False,
                "calibration_status": calibration["status"],
                "policy_prediction_opened": False,
                "development_prediction_opened": False,
                "distant_confirmation_opened": False,
                "final_status": profile["final_status"],
            }
        )
    return output


def _architecture_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    architecture = report["calibration_architecture"]
    diagnostics = architecture["diagnostics"]
    gates = architecture["gates"]
    selected = diagnostics["selected_action"]
    top = selected["top_rows"]
    assert isinstance(gates, Mapping)
    return [
        {
            "metric": "opportunity_roc_auc",
            "observed": diagnostics["opportunity_auc"],
            "acceptance_operator": ">=",
            "acceptance_gate": gates["minimum_opportunity_auc"],
            "passed": True,
            "population_rows": diagnostics["rows"],
            "unit": "auc",
        },
        {
            "metric": "conditional_direction_roc_auc",
            "observed": diagnostics["conditional_direction_auc"],
            "acceptance_operator": ">=",
            "acceptance_gate": gates["minimum_conditional_direction_auc"],
            "passed": False,
            "population_rows": diagnostics["conditional_direction_rows"],
            "unit": "auc",
        },
        {
            "metric": "selected_top_100_mean_stress_net_bps",
            "observed": top["100"]["mean_stress_net_bps"],
            "acceptance_operator": ">",
            "acceptance_gate": gates[
                "minimum_selected_top_100_mean_stress_net_bps"
            ],
            "passed": False,
            "population_rows": 100,
            "unit": "bps",
        },
        {
            "metric": "selected_top_500_mean_stress_net_bps",
            "observed": top["500"]["mean_stress_net_bps"],
            "acceptance_operator": ">",
            "acceptance_gate": gates[
                "minimum_selected_top_500_mean_stress_net_bps"
            ],
            "passed": False,
            "population_rows": 500,
            "unit": "bps",
        },
    ]


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    architecture = report["calibration_architecture"]["diagnostics"]
    selected = architecture["selected_action"]
    stress = report["forecast_diagnostics_by_role"]["calibration"]["stress"][
        "sides"
    ]
    output = [
        {
            "diagnostic": "opportunity",
            "population": "either_side_positive_after_costs",
            "rows": architecture["rows"],
            "roc_auc": architecture["opportunity_auc"],
            "pearson_ic": "",
            "spearman_ic": "",
            "mean_stress_net_bps": "",
            "top_100_mean_stress_net_bps": "",
            "top_500_mean_stress_net_bps": "",
            "long_share_top_500": "",
        },
        {
            "diagnostic": "conditional_direction",
            "population": "profitable_opportunity_rows_only",
            "rows": architecture["conditional_direction_rows"],
            "roc_auc": architecture["conditional_direction_auc"],
            "pearson_ic": "",
            "spearman_ic": "",
            "mean_stress_net_bps": "",
            "top_100_mean_stress_net_bps": "",
            "top_500_mean_stress_net_bps": "",
            "long_share_top_500": "",
        },
    ]
    selected_top = selected["top_rows"]
    output.append(
        {
            "diagnostic": "selected_action",
            "population": "all_calibration_events",
            "rows": selected["rows"],
            "roc_auc": selected["side_choice_auc"],
            "pearson_ic": selected[
                "selected_action_pearson_information_coefficient"
            ],
            "spearman_ic": selected[
                "selected_action_spearman_information_coefficient"
            ],
            "mean_stress_net_bps": selected["mean_selected_stress_net_bps"],
            "top_100_mean_stress_net_bps": selected_top["100"][
                "mean_stress_net_bps"
            ],
            "top_500_mean_stress_net_bps": selected_top["500"][
                "mean_stress_net_bps"
            ],
            "long_share_top_500": selected_top["500"]["long_share"],
        }
    )
    for side in ("long", "short"):
        value = stress[side]
        output.append(
            {
                "diagnostic": f"raw_{side}_action",
                "population": "all_calibration_events",
                "rows": value["rows"],
                "roc_auc": value["profitable_auc"],
                "pearson_ic": value["pearson_information_coefficient"],
                "spearman_ic": value["spearman_information_coefficient"],
                "mean_stress_net_bps": value["mean_actual_net_bps"],
                "top_100_mean_stress_net_bps": value["top_rows"][0][
                    "mean_actual_net_bps"
                ],
                "top_500_mean_stress_net_bps": value["top_rows"][1][
                    "mean_actual_net_bps"
                ],
                "long_share_top_500": "",
            }
        )
    return output


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for model in report["ensemble_models"]:
        output.append(
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
                "conditional_direction_temperature": model[
                    "conditional_direction_temperature"
                ],
                "opportunity_calibration_slope": model[
                    "opportunity_probability_calibration"
                ][0],
                "opportunity_calibration_intercept": model[
                    "opportunity_probability_calibration"
                ][1],
                "reload_max_abs_error": model[
                    "artifact_reload_max_abs_prediction_error"
                ],
                "opportunity_invariance_max_abs_error": model[
                    "opportunity_mirror_invariance_max_abs_error"
                ],
                "direction_complementarity_max_abs_error": model[
                    "direction_mirror_complementarity_max_abs_error"
                ],
                "action_swap_max_abs_error": model[
                    "action_swap_equivariance_max_abs_error"
                ],
            }
        )
    return output


def _progress_rows(
    path: Path,
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields or [int(row["round"]) for row in rows] != list(range(1, 33)):
        raise ValueError("prior action-value progress sequence changed")
    diagnostics = report["calibration_architecture"]["diagnostics"]
    selected = diagnostics["selected_action"]
    update = {name: "" for name in fields}
    update.update(
        {
            "round": 33,
            "stage": "factorized selective opportunity-direction calibration",
            "periods": "2023-05-16..2023-07-06",
            "selection_contaminated": True,
            "horizon_seconds": 900,
            "feature_set": "l1-tape-causal-v8 factorized opportunity and conditional direction",
            "risk_level": "conservative;regular;aggressive",
            "direction_auc": diagnostics["conditional_direction_auc"],
            "spearman_ic": selected[
                "selected_action_spearman_information_coefficient"
            ],
            "selected_signals": 0,
            "executable_trades": 0,
            "status": "rejected",
            "source_file": "verified Round 33 report",
            "best_model_id": "three-seed factorized selective-action LightGBM ensemble",
            "best_top_500_exact_after_cost_bps": selected["top_rows"]["500"][
                "mean_stress_net_bps"
            ],
            "after_cost_diagnostic_rows": selected["rows"],
            "calibration_threshold_traces": 0,
            "accepted_thresholds": 0,
            "ensemble_models": 3,
            "valid_barrier_rows": report["training_corpus"][
                "valid_barrier_rows"
            ],
            "calibration_eligible_rows": 0,
            "development_consumed": False,
        }
    )
    rows.append(update)
    return rows, fields


def _stage_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1320, 430
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 33 chronological stage access</title>',
        '<desc id="desc">Calibration opened; threshold selection and all later predictions remained withheld after the architecture gate failed.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Nested evaluation stopped at calibration</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53636f">Green means predicted; red means withheld. All dates are UTC.</text>',
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
                f'<text x="{x + 20}" y="258" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#455763">0 profiles survived</text>',
                f'<text x="{x + 20}" y="287" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#667681">{int(row["valid_role_rows"]):,} valid role rows</text>',
            ]
        )
        if index < len(rows) - 1:
            lines.append(
                f'<path d="M {x + 275} 220 L {x + 305} 220" stroke="#8998a3" stroke-width="3" marker-end="url(#arrow)"/>'
            )
    lines.extend(
        [
            '<text x="48" y="382" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#62727d">Threshold selection, later roles, untouched dates, leverage, and execution authority remained inaccessible.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _eligibility_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 980, 500
    colors = ("#147d70", "#3d6f99", "#bd6645")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 33 calibration eligibility</title>',
        '<desc id="desc">No risk profile reached policy eligibility because the architecture gate failed before threshold selection.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">No profile received a tradable signal</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Calibration: 2023-06-21 to 2023-06-25 UTC; 28,581 event rows.</text>',
        '<rect x="110" y="125" width="780" height="260" fill="#ffffff" stroke="#d7e0e6"/>',
    ]
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        x = 240 + index * 250
        lines.extend(
            [
                f'<circle cx="{x}" cy="250" r="52" fill="#ffffff" stroke="{color}" stroke-width="8"/>',
                f'<text x="{x}" y="264" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="38" font-weight="700" fill="#263640">0</text>',
                f'<text x="{x}" y="337" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#344750">{html.escape(str(row["profile"]).title())}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="48" y="450" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">Zero is the fail-closed result, not evidence that no market opportunities existed.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _architecture_svg(rows: Sequence[Mapping[str, object]]) -> str:
    by_metric = {str(row["metric"]): row for row in rows}
    opportunity = by_metric["opportunity_roc_auc"]
    direction = by_metric["conditional_direction_roc_auc"]
    top100 = by_metric["selected_top_100_mean_stress_net_bps"]
    top500 = by_metric["selected_top_500_mean_stress_net_bps"]
    width, height = 1260, 590
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 33 calibration architecture gates</title>',
        '<desc id="desc">Opportunity AUC passed, while conditional direction AUC and selected top-tail after-cost returns failed.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Opportunity detection survived; action selection did not</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Precommitted calibration gates; adverse-stress net returns include modeled execution costs.</text>',
        '<rect x="70" y="125" width="520" height="330" fill="#ffffff" stroke="#d7e0e6"/>',
        '<rect x="650" y="125" width="540" height="330" fill="#ffffff" stroke="#d7e0e6"/>',
        '<text x="92" y="160" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">ROC AUC</text>',
        '<text x="672" y="160" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">Selected-action stress mean</text>',
    ]
    for index, (label, row) in enumerate(
        (("Opportunity", opportunity), ("Direction | opportunity", direction))
    ):
        y = 215 + index * 125
        value = float(row["observed"])
        gate = float(row["acceptance_gate"])
        observed_width = 390 * (value - 0.45) / 0.25
        gate_x = 160 + 390 * (gate - 0.45) / 0.25
        color = "#147d70" if bool(row["passed"]) else "#bc493e"
        lines.extend(
            [
                f'<text x="92" y="{y - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#344750">{label}</text>',
                f'<rect x="160" y="{y}" width="390" height="28" fill="#edf1f4"/>',
                f'<rect x="160" y="{y}" width="{max(0.0, observed_width):.1f}" height="28" fill="{color}"/>',
                f'<line x1="{gate_x:.1f}" y1="{y - 8}" x2="{gate_x:.1f}" y2="{y + 36}" stroke="#18242d" stroke-width="2"/>',
                f'<text x="160" y="{y + 55}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#53636f">observed {value:.4f}; gate {gate:.3f}</text>',
            ]
        )
    zero_x = 1030
    scale = 18.0
    lines.append(
        f'<line x1="{zero_x}" y1="185" x2="{zero_x}" y2="395" stroke="#18242d" stroke-width="2"/>'
    )
    for index, (label, row) in enumerate((("Top 100", top100), ("Top 500", top500))):
        y = 225 + index * 115
        value = float(row["observed"])
        bar_width = min(350.0, abs(value) * scale)
        lines.extend(
            [
                f'<text x="672" y="{y - 18}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#344750">{label}</text>',
                f'<rect x="{zero_x - bar_width:.1f}" y="{y}" width="{bar_width:.1f}" height="34" fill="#bc493e"/>',
                f'<text x="672" y="{y + 58}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263640">{value:+.2f} bps; required &gt; 0</text>',
            ]
        )
    lines.extend(
        [
            '<text x="70" y="512" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Result: 1 of 4 architecture gates passed. Threshold selection was not opened.</text>',
            '<text x="70" y="542" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">AUC measures ranking discrimination; it is not an economic return or profitability claim.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = {str(row["diagnostic"]): row for row in rows}
    selected = values["selected_action"]
    width, height = 1260, 560
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 33 factorized forecast diagnostics</title>',
        '<desc id="desc">Opportunity and conditional direction AUC beside selected-action information coefficients and ranked-tail returns.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Directional ranking remained economically adverse</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Calibration: 2023-06-21 to 2023-06-25 UTC; no threshold or later role was evaluated.</text>',
        '<rect x="70" y="125" width="500" height="300" fill="#ffffff" stroke="#d7e0e6"/>',
        '<rect x="650" y="125" width="540" height="300" fill="#ffffff" stroke="#d7e0e6"/>',
        '<text x="92" y="160" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">Discrimination</text>',
        '<text x="672" y="160" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">Selected action</text>',
    ]
    for index, name in enumerate(("opportunity", "conditional_direction", "selected_action")):
        row = values[name]
        value = float(row["roc_auc"])
        y = 205 + index * 68
        width_value = max(0.0, (value - 0.45) / 0.25 * 340)
        color = "#147d70" if name == "opportunity" else "#bd6645"
        lines.extend(
            [
                f'<text x="92" y="{y + 20}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#344750">{name.replace("_", " ").title()}</text>',
                f'<rect x="235" y="{y}" width="300" height="28" fill="#edf1f4"/>',
                f'<rect x="235" y="{y}" width="{min(300.0, width_value):.1f}" height="28" fill="{color}"/>',
                f'<text x="545" y="{y + 20}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#263640">{value:.4f}</text>',
            ]
        )
    metrics = (
        ("Pearson IC", float(selected["pearson_ic"])),
        ("Spearman IC", float(selected["spearman_ic"])),
        ("Top 100", float(selected["top_100_mean_stress_net_bps"])),
        ("Top 500", float(selected["top_500_mean_stress_net_bps"])),
    )
    for index, (label, value) in enumerate(metrics):
        y = 205 + index * 55
        suffix = "" if "IC" in label else " bps"
        lines.extend(
            [
                f'<text x="672" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#344750">{label}</text>',
                f'<text x="1135" y="{y}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" fill="#bc493e">{value:+.4f}{suffix}</text>',
                f'<line x1="672" y1="{y + 13}" x2="1135" y2="{y + 13}" stroke="#e5ebef"/>',
            ]
        )
    lines.extend(
        [
            '<text x="70" y="486" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">The factorization improved opportunity AUC but did not create positive after-cost action ranking.</text>',
            '<text x="70" y="516" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">All values come from the hash-bound report; chart geometry is regenerated deterministically from the CSV data.</text>',
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
    architecture = _architecture_rows(report)
    forecasts = _forecast_rows(report)
    models = _model_rows(report)
    progress, progress_fields = _progress_rows(prior_progress_path, report)
    charts = output_dir / "charts"
    _write_csv(output_dir / "stages.csv", stages)
    _write_csv(output_dir / "profiles.csv", profiles)
    _write_csv(output_dir / "architecture.csv", architecture)
    _write_csv(output_dir / "forecast.csv", forecasts)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(charts / "stage-access.svg", _stage_svg(stages))
    _write_text(charts / "eligibility.svg", _eligibility_svg(profiles))
    _write_text(charts / "architecture-gates.svg", _architecture_svg(architecture))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(forecasts))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    for stale in (
        output_dir / "thresholds.csv",
        charts / "threshold-economics.svg",
        output_dir / "candidates.csv",
        charts / "candidate-economics.svg",
    ):
        if stale.exists():
            stale.unlink()
    diagnostics = report["calibration_architecture"]["diagnostics"]
    selected = diagnostics["selected_action"]
    readme = f"""# Round 33: selective-action calibration rejected

**Rejected without trading authority.** The factorized model separated opportunity detection from direction conditional on an after-cost opportunity. Opportunity discrimination passed, but direction and selected-action economics failed their frozen calibration gates. Threshold selection and every later role remained withheld.

| Evidence | Verified result |
| --- | ---: |
| Source window | 2023-05-16 to 2023-07-06 UTC |
| Causal one-second rows | {int(report['training_corpus']['dataset']['rows']):,} |
| CUSUM events / valid barrier outcomes | {int(report['training_corpus']['event_rows']):,} / {int(report['training_corpus']['valid_barrier_rows']):,} |
| Train / early-stop / calibration rows | 128,307 / 21,934 / 28,581 |
| Opportunity ROC AUC / gate | {float(diagnostics['opportunity_auc']):.4f} / 0.6500 |
| Conditional direction ROC AUC / gate | {float(diagnostics['conditional_direction_auc']):.4f} / 0.5500 |
| Selected-action side AUC / Spearman IC | {float(selected['side_choice_auc']):.4f} / {float(selected['selected_action_spearman_information_coefficient']):+.4f} |
| Top-100 / top-500 stress mean | {float(selected['top_rows']['100']['mean_stress_net_bps']):+.2f} / {float(selected['top_rows']['500']['mean_stress_net_bps']):+.2f} bps |
| Eligible rows: conservative / regular / aggressive | 0 / 0 / 0 |
| Final profiles | none |

![Stage access](charts/stage-access.svg)

![Calibration architecture gates](charts/architecture-gates.svg)

![Calibration eligibility](charts/eligibility.svg)

![Forecast diagnostics](charts/forecast-quality.svg)

![Research progress](charts/research-progress.svg)

All three direction calibrators reached temperature `54.598`, the configured search boundary, which is recorded in `models.csv`. This is evidence of weak confidence calibration, not a reason to loosen risk controls. DirectML tensor execution and OpenCL FP64 LightGBM training were attested. No leverage, testnet or live execution, untouched-period claim, or profitability claim is permitted.

Data: [stages.csv](stages.csv) | [profiles.csv](profiles.csv) | [architecture.csv](architecture.csv) | [forecast.csv](forecast.csv) | [models.csv](models.csv) | [progress.csv](progress.csv) | [integrity report](report.json)
"""
    _write_text(output_dir / "README.md", readme)
    artifact_paths = [
        output_dir / "README.md",
        output_dir / "stages.csv",
        output_dir / "profiles.csv",
        output_dir / "architecture.csv",
        output_dir / "forecast.csv",
        output_dir / "models.csv",
        output_dir / "progress.csv",
        charts / "stage-access.svg",
        charts / "eligibility.svg",
        charts / "architecture-gates.svg",
        charts / "forecast-quality.svg",
        charts / "research-progress.svg",
    ]
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact_class": "selective_action_viability_graph_data",
        "round": 33,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": report["binding_sha256"],
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "stage_access": {
            "calibration_prediction": True,
            "calibration_threshold_selection": False,
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
        "artifact_integrity": [
            _artifact(path, output_dir) for path in artifact_paths
        ],
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
    parser = argparse.ArgumentParser(description="Publish verified Round 33 evidence.")
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-033-selective-action-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-033-execution-binding.json",
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
    print(json.dumps(publication, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
