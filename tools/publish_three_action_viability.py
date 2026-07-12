"""Publish verified Round 34 evidence as latest-only tables and SVG charts."""

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
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _read_object,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_three_action_viability import (  # noqa: E402
    BINDING_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    _canonical_sha256,
    _is_sha256,
    _sha256_file,
    load_round34_design,
)


PUBLICATION_SCHEMA_VERSION = "three-action-utility-viability-publication-v1"
_STAGES = ("calibration", "policy", "development", "distant_confirmation")


def _validated_source(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], str]:
    design, design_sha, _profiles = load_round34_design(design_path)
    binding = _read_object(binding_path, label="Round 34 execution binding")
    binding_canonical = dict(binding)
    binding_sha = binding_canonical.pop("binding_sha256", None)
    binding_implementation = binding.get("implementation")
    if (
        binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("round") != 34
        or not _is_sha256(binding_sha)
        or _canonical_sha256(binding_canonical) != binding_sha
        or not isinstance(binding_implementation, Mapping)
    ):
        raise ValueError("Round 34 execution binding hash is invalid")
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, label="Round 34 report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 34
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding_implementation.get("commit")
        or not _is_sha256(claimed)
        or _canonical_sha256(canonical) != claimed
        or report.get("all_target_dates_previously_consumed") is not True
        or report.get("untouched_policy_windows_accessed") is not False
    ):
        raise ValueError("Round 34 report identity is invalid")
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
        raise ValueError("Round 34 nested-stage access is invalid")
    semantics = report.get("probability_semantics")
    implementation = report.get("implementation_gates")
    models = report.get("ensemble_models")
    profiles = report.get("profile_results")
    architecture = report.get("calibration_architecture")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("semantic_aliasing_permitted") is not False
        or not isinstance(implementation, Mapping)
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
        raise ValueError("Round 34 implementation or rejection evidence is incomplete")
    corpus = report.get("training_corpus")
    if (
        not isinstance(corpus, Mapping)
        or corpus.get("corpus_certificate_sha256")
        != "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
        or corpus.get("barrier_targets_sha256")
        != "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
        or corpus.get("cache_key")
        != "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
        or corpus.get("event_rows") != 230_941
        or corpus.get("valid_barrier_rows") != 229_000
    ):
        raise ValueError("Round 34 source-corpus identity changed")
    symmetry_limits = {
        "class_probability_mirror_swap_max_abs_error": float(
            implementation["maximum_permitted_class_probability_mirror_swap_error"]
        ),
        "action_swap_equivariance_max_abs_error": float(
            implementation["maximum_permitted_action_swap_equivariance_error"]
        ),
        "probability_sum_max_abs_error": float(
            implementation["maximum_permitted_probability_sum_error"]
        ),
    }
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("Round 34 model evidence is invalid")
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
                not math.isfinite(float(model[name])) or float(model[name]) > limit
                for name, limit in symmetry_limits.items()
            )
        ):
            raise ValueError("Round 34 model artifact evidence differs")
    expected_eligibility = {
        "conservative": 0,
        "regular": 10,
        "aggressive": 39,
    }
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise ValueError("Round 34 profile evidence is invalid")
        calibration = profile.get("calibration")
        if (
            not isinstance(calibration, Mapping)
            or not isinstance(calibration.get("eligible_rows"), int)
            or calibration.get("eligible_rows")
            != expected_eligibility.get(str(profile.get("profile") or ""))
            or calibration.get("architecture_gate_passed") is not False
            or calibration.get("threshold_selection") is not None
            or calibration.get("threshold_withheld_reason")
            != "calibration_architecture_rejected"
            or profile.get("final_status") != "rejected"
        ):
            raise ValueError("Round 34 profile rejection evidence changed")
    return design, report, str(claimed)


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
                    "" if opened else "calibration_architecture_rejected"
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
                "architecture_gate_passed": False,
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
    definitions = (
        (
            "opportunity_roc_auc",
            diagnostics["opportunity_auc"],
            ">=",
            gates["minimum_opportunity_auc"],
            diagnostics["rows"],
            "auc",
        ),
        (
            "conditional_direction_roc_auc",
            diagnostics["conditional_direction_auc"],
            ">=",
            gates["minimum_conditional_direction_auc"],
            diagnostics["conditional_direction_rows"],
            "auc",
        ),
        (
            "side_profit_roc_auc",
            diagnostics["side_profit_auc"],
            ">=",
            gates["minimum_side_profit_auc"],
            diagnostics["side_profit_rows"],
            "auc",
        ),
        (
            "side_profit_brier_to_base_rate_ratio",
            diagnostics["side_profit_brier_to_base_rate_ratio"],
            "<=",
            gates["maximum_side_profit_brier_to_base_rate_ratio"],
            diagnostics["side_profit_rows"],
            "ratio",
        ),
        (
            "multiclass_log_loss_to_class_prior_ratio",
            diagnostics["multiclass_log_loss_to_class_prior_ratio"],
            "<=",
            gates["maximum_multiclass_log_loss_to_class_prior_ratio"],
            diagnostics["rows"],
            "ratio",
        ),
        (
            "selected_top_100_mean_stress_net_bps",
            top["100"]["mean_stress_net_bps"],
            ">",
            gates["minimum_selected_top_100_mean_stress_net_bps"],
            100,
            "bps",
        ),
        (
            "selected_top_500_mean_stress_net_bps",
            top["500"]["mean_stress_net_bps"],
            ">",
            gates["minimum_selected_top_500_mean_stress_net_bps"],
            500,
            "bps",
        ),
    )
    rows: list[dict[str, object]] = []
    for metric, observed, operator, gate, population, unit in definitions:
        actual = float(observed)
        limit = float(gate)
        passed = actual >= limit if operator == ">=" else actual <= limit
        if operator == ">":
            passed = actual > limit
        rows.append(
            {
                "metric": metric,
                "observed": actual,
                "acceptance_operator": operator,
                "acceptance_gate": limit,
                "passed": passed,
                "population_rows": population,
                "unit": unit,
            }
        )
    return rows


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    diagnostics = report["calibration_architecture"]["diagnostics"]
    selected = diagnostics["selected_action"]
    stress = report["forecast_diagnostics_by_role"]["calibration"]["stress"]["sides"]
    rows = [
        {
            "diagnostic": "action_opportunity",
            "population": "non_abstain_action_class",
            "rows": diagnostics["rows"],
            "roc_auc": diagnostics["opportunity_auc"],
            "pearson_ic": "",
            "spearman_ic": "",
            "mean_stress_net_bps": "",
            "top_100_mean_stress_net_bps": "",
            "top_500_mean_stress_net_bps": "",
            "long_share_top_500": "",
        },
        {
            "diagnostic": "conditional_action_direction",
            "population": "non_abstain_action_class_only",
            "rows": diagnostics["conditional_direction_rows"],
            "roc_auc": diagnostics["conditional_direction_auc"],
            "pearson_ic": "",
            "spearman_ic": "",
            "mean_stress_net_bps": "",
            "top_100_mean_stress_net_bps": "",
            "top_500_mean_stress_net_bps": "",
            "long_share_top_500": "",
        },
        {
            "diagnostic": "side_profit",
            "population": "paired_long_short_action_rows",
            "rows": diagnostics["side_profit_rows"],
            "roc_auc": diagnostics["side_profit_auc"],
            "pearson_ic": "",
            "spearman_ic": "",
            "mean_stress_net_bps": "",
            "top_100_mean_stress_net_bps": "",
            "top_500_mean_stress_net_bps": "",
            "long_share_top_500": "",
        },
    ]
    selected_top = selected["top_rows"]
    rows.append(
        {
            "diagnostic": "selected_action",
            "population": "all_calibration_events",
            "rows": selected["rows"],
            "roc_auc": selected["side_choice_auc"],
            "pearson_ic": selected["selected_action_pearson_information_coefficient"],
            "spearman_ic": selected["selected_action_spearman_information_coefficient"],
            "mean_stress_net_bps": selected["mean_selected_stress_net_bps"],
            "top_100_mean_stress_net_bps": selected_top["100"]["mean_stress_net_bps"],
            "top_500_mean_stress_net_bps": selected_top["500"]["mean_stress_net_bps"],
            "long_share_top_500": selected_top["500"]["long_share"],
        }
    )
    for side in ("long", "short"):
        value = stress[side]
        rows.append(
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
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for model in report["ensemble_models"]:
        calibration = model["side_profit_probability_calibration"]
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
                "multiclass_temperature": model["multiclass_calibration_temperature"],
                "multiclass_abstain_logit_bias": model[
                    "multiclass_calibration_abstain_logit_bias"
                ],
                "multiclass_calibration_log_loss": model[
                    "multiclass_calibration_log_loss"
                ],
                "multiclass_class_prior_log_loss": model[
                    "multiclass_class_prior_log_loss"
                ],
                "side_profit_platt_slope": calibration[0],
                "side_profit_platt_intercept": calibration[1],
                "regret_scale_bps": model["regret_scale_bps"],
                "reload_max_abs_error": model[
                    "artifact_reload_max_abs_prediction_error"
                ],
                "class_mirror_swap_max_abs_error": model[
                    "class_probability_mirror_swap_max_abs_error"
                ],
                "action_swap_max_abs_error": model[
                    "action_swap_equivariance_max_abs_error"
                ],
                "probability_sum_max_abs_error": model["probability_sum_max_abs_error"],
            }
        )
    return output


def _progress_rows(
    path: Path,
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not original_fields:
        raise ValueError("prior action-value progress fields are missing")
    rounds = [int(row["round"]) for row in rows]
    if rounds == list(range(1, 35)):
        if rows[-1].get("source_file") != "verified Round 34 report":
            raise ValueError("existing Round 34 progress row is not reproducible")
        rows = rows[:-1]
    elif rounds != list(range(1, 34)):
        raise ValueError("prior action-value progress sequence changed")
    added_fields = (
        "opportunity_auc",
        "side_profit_auc",
        "multiclass_log_loss_ratio",
        "top_100_exact_after_cost_bps",
        "architecture_gates_passed",
        "architecture_gate_count",
    )
    fields = tuple(
        (
            *original_fields,
            *(name for name in added_fields if name not in original_fields),
        )
    )
    for row in rows:
        for name in fields:
            row.setdefault(name, "")
    diagnostics = report["calibration_architecture"]["diagnostics"]
    selected = diagnostics["selected_action"]
    eligible = max(
        int(profile["calibration"]["eligible_rows"])
        for profile in report["profile_results"]
    )
    update = {name: "" for name in fields}
    update.update(
        {
            "round": 34,
            "stage": "utility-weighted symmetric three-action calibration",
            "periods": "2023-05-16..2023-07-06",
            "selection_contaminated": True,
            "horizon_seconds": 900,
            "feature_set": (
                "l1-tape-causal-v8 separate action-class and side-profit probabilities"
            ),
            "risk_level": "conservative;regular;aggressive",
            "direction_auc": diagnostics["conditional_direction_auc"],
            "spearman_ic": selected["selected_action_spearman_information_coefficient"],
            "selected_signals": eligible,
            "executable_trades": 0,
            "status": "rejected",
            "source_file": "verified Round 34 report",
            "best_model_id": (
                "three-seed utility-weighted symmetric three-action LightGBM ensemble"
            ),
            "best_top_500_exact_after_cost_bps": selected["top_rows"]["500"][
                "mean_stress_net_bps"
            ],
            "after_cost_diagnostic_rows": selected["rows"],
            "calibration_threshold_traces": 0,
            "accepted_thresholds": 0,
            "ensemble_models": 3,
            "valid_barrier_rows": report["training_corpus"]["valid_barrier_rows"],
            "calibration_eligible_rows": eligible,
            "development_consumed": False,
            "opportunity_auc": diagnostics["opportunity_auc"],
            "side_profit_auc": diagnostics["side_profit_auc"],
            "multiclass_log_loss_ratio": diagnostics[
                "multiclass_log_loss_to_class_prior_ratio"
            ],
            "top_100_exact_after_cost_bps": selected["top_rows"]["100"][
                "mean_stress_net_bps"
            ],
            "architecture_gates_passed": 4,
            "architecture_gate_count": 7,
        }
    )
    rows.append(update)
    return rows, fields


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _stage_svg(rows: Sequence[Mapping[str, object]]) -> str:
    lines = _svg_start(
        1320,
        430,
        "Round 34 chronological stage access",
        "Calibration prediction opened; threshold selection and all later stages remained withheld after the architecture rejection.",
    )
    lines.extend(
        [
            '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Nested evaluation stopped at calibration</text>',
            '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53636f">Green means predicted; red means withheld. All dates are UTC.</text>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#8998a3"/></marker></defs>',
        ]
    )
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
    maximum = max(1, *(int(row["eligible_rows"]) for row in rows))
    lines = _svg_start(
        1080,
        520,
        "Round 34 calibration policy eligibility",
        "Conservative had zero eligible rows, regular had ten, and aggressive had thirty-nine; architecture rejection prevented threshold selection.",
    )
    lines.extend(
        [
            '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Some actions cleared policy gates, but none were authorized</text>',
            '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Calibration: 2023-06-21 to 2023-06-25 UTC; architecture gates run before threshold economics.</text>',
            '<line x1="260" y1="125" x2="260" y2="400" stroke="#9aa8b2"/>',
        ]
    )
    colors = ("#147d70", "#3d6f99", "#bd6645")
    for index, (row, color) in enumerate(zip(rows, colors, strict=True)):
        y = 150 + index * 90
        value = int(row["eligible_rows"])
        width = 650.0 * value / maximum
        lines.extend(
            [
                f'<text x="48" y="{y + 27}" font-family="Segoe UI, Arial, sans-serif" font-size="15" font-weight="700" fill="#344750">{html.escape(str(row["profile"]).title())}</text>',
                f'<rect x="260" y="{y}" width="650" height="42" fill="#edf1f4"/>',
                f'<rect x="260" y="{y}" width="{width:.1f}" height="42" fill="{color}"/>',
                f'<text x="930" y="{y + 29}" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#263640">{value:,}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="48" y="452" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Eligible rows are diagnostics only: the negative ranked tails stopped threshold selection.</text>',
            '<text x="48" y="482" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">No threshold trace, backtest trade, leverage, testnet order, or live order was produced.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _architecture_svg(rows: Sequence[Mapping[str, object]]) -> str:
    lines = _svg_start(
        1320,
        720,
        "Round 34 calibration architecture gates",
        "Four of seven frozen architecture gates passed; conditional direction and both selected-action ranked-tail return gates failed.",
    )
    lines.extend(
        [
            '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Calibration quality improved; action economics remained adverse</text>',
            '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Observed values and precommitted gates; stress returns include modeled fees, latency, spread, and slippage.</text>',
            '<text x="70" y="123" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#62727d">METRIC</text>',
            '<text x="770" y="123" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#62727d">OBSERVED</text>',
            '<text x="930" y="123" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#62727d">GATE</text>',
            '<text x="1140" y="123" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#62727d">RESULT</text>',
        ]
    )
    for index, row in enumerate(rows):
        y = 145 + index * 67
        passed = bool(row["passed"])
        color = "#147d70" if passed else "#bc493e"
        fill = "#f5f8fa" if index % 2 == 0 else "#ffffff"
        observed = float(row["observed"])
        gate = float(row["acceptance_gate"])
        unit = str(row["unit"])
        observed_text = f"{observed:+.4f}" if unit == "bps" else f"{observed:.4f}"
        gate_text = f"{row['acceptance_operator']} {gate:.4f}"
        lines.extend(
            [
                f'<rect x="48" y="{y}" width="1224" height="56" fill="{fill}"/>',
                f'<text x="70" y="{y + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#2b3d47">{html.escape(str(row["metric"]).replace("_", " "))}</text>',
                f'<text x="770" y="{y + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" fill="#263640">{observed_text} {html.escape(unit)}</text>',
                f'<text x="930" y="{y + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">{html.escape(gate_text)}</text>',
                f'<rect x="1132" y="{y + 12}" width="112" height="32" rx="4" fill="{color}"/>',
                f'<text x="1188" y="{y + 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#ffffff">{"PASS" if passed else "FAIL"}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="48" y="666" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Result: 4 of 7 gates passed. Passing probability calibration does not override negative selected-action returns.</text>',
            '<text x="48" y="696" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">AUC and loss ratios measure forecast quality; only after-cost stress bps measure ranked economic outcome.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = {str(row["diagnostic"]): row for row in rows}
    selected = values["selected_action"]
    auc_rows = (
        ("Action opportunity", values["action_opportunity"]),
        ("Direction | action", values["conditional_action_direction"]),
        ("Side profit", values["side_profit"]),
    )
    lines = _svg_start(
        1260,
        570,
        "Round 34 three-action forecast diagnostics",
        "Action-opportunity and side-profit AUC passed, while conditional direction and selected-action ranked returns remained weak or negative.",
    )
    lines.extend(
        [
            '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#18242d">Probability quality did not translate into positive action ranking</text>',
            '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">Calibration: 2023-06-21 to 2023-06-25 UTC; separate class and side-profit probabilities.</text>',
            '<rect x="70" y="125" width="500" height="315" fill="#ffffff" stroke="#d7e0e6"/>',
            '<rect x="650" y="125" width="540" height="315" fill="#ffffff" stroke="#d7e0e6"/>',
            '<text x="92" y="160" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">ROC AUC</text>',
            '<text x="672" y="160" font-family="Segoe UI, Arial, sans-serif" font-size="16" font-weight="700" fill="#2b3d47">Selected action</text>',
        ]
    )
    for index, (label, row) in enumerate(auc_rows):
        y = 198 + index * 75
        value = float(row["roc_auc"])
        bar_width = min(300.0, max(0.0, (value - 0.45) / 0.25 * 300.0))
        color = "#147d70" if value >= 0.55 else "#bd6645"
        lines.extend(
            [
                f'<text x="92" y="{y + 20}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#344750">{html.escape(label)}</text>',
                f'<rect x="245" y="{y}" width="270" height="28" fill="#edf1f4"/>',
                f'<rect x="245" y="{y}" width="{min(270.0, bar_width):.1f}" height="28" fill="{color}"/>',
                f'<text x="545" y="{y + 20}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#263640">{value:.4f}</text>',
            ]
        )
    metrics = (
        ("Side-choice AUC", float(selected["roc_auc"]), ""),
        ("Pearson IC", float(selected["pearson_ic"]), ""),
        ("Spearman IC", float(selected["spearman_ic"]), ""),
        ("Top 100", float(selected["top_100_mean_stress_net_bps"]), " bps"),
        ("Top 500", float(selected["top_500_mean_stress_net_bps"]), " bps"),
    )
    for index, (label, value, suffix) in enumerate(metrics):
        y = 198 + index * 48
        color = "#147d70" if value > 0.0 and not suffix else "#bc493e"
        if label == "Side-choice AUC":
            color = "#bd6645"
        lines.extend(
            [
                f'<text x="672" y="{y}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#344750">{html.escape(label)}</text>',
                f'<text x="1135" y="{y}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" fill="{color}">{value:+.4f}{suffix}</text>',
                f'<line x1="672" y1="{y + 13}" x2="1135" y2="{y + 13}" stroke="#e5ebef"/>',
            ]
        )
    lines.extend(
        [
            '<text x="70" y="500" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53636f">The top-100 tail improved versus Round 33 but remained negative after modeled execution costs.</text>',
            '<text x="70" y="530" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#667681">Chart values are regenerated deterministically from forecast.csv and the hash-bound source report.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


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
    eligibility = {str(row["profile"]): int(row["eligible_rows"]) for row in profiles}
    readme = f"""# Round 34: three-action calibration rejected

**Rejected without trading authority.** The model now keeps action-class probabilities separate from side-profit probabilities. Opportunity, side-profit, multiclass log-loss, and side-profit Brier gates passed; conditional direction and both selected-action economic tails failed. Threshold selection and every later role remained withheld.

| Evidence | Verified result |
| --- | ---: |
| Source window | 2023-05-16 to 2023-07-06 UTC |
| Causal one-second rows | {int(report["training_corpus"]["dataset"]["rows"]):,} |
| CUSUM events / valid barrier outcomes | {int(report["training_corpus"]["event_rows"]):,} / {int(report["training_corpus"]["valid_barrier_rows"]):,} |
| Train / early-stop / calibration rows | 128,307 / 21,934 / 28,581 |
| Opportunity ROC AUC / gate | {float(diagnostics["opportunity_auc"]):.4f} / 0.6500 |
| Conditional direction ROC AUC / gate | {float(diagnostics["conditional_direction_auc"]):.4f} / 0.5500 |
| Side-profit ROC AUC / gate | {float(diagnostics["side_profit_auc"]):.4f} / 0.5500 |
| Multiclass log-loss / prior ratio | {float(diagnostics["multiclass_log_loss_to_class_prior_ratio"]):.4f} |
| Side-profit Brier / prior ratio | {float(diagnostics["side_profit_brier_to_base_rate_ratio"]):.4f} |
| Selected-action side AUC / Spearman IC | {float(selected["side_choice_auc"]):.4f} / {float(selected["selected_action_spearman_information_coefficient"]):+.4f} |
| Top-100 / top-500 stress mean | {float(selected["top_rows"]["100"]["mean_stress_net_bps"]):+.2f} / {float(selected["top_rows"]["500"]["mean_stress_net_bps"]):+.2f} bps |
| Eligible rows: conservative / regular / aggressive | {eligibility["conservative"]} / {eligibility["regular"]} / {eligibility["aggressive"]} |
| Architecture gates passed | 4 / 7 |
| Final profiles | none |

![Stage access](charts/stage-access.svg)

![Calibration architecture gates](charts/architecture-gates.svg)

![Calibration eligibility](charts/eligibility.svg)

![Forecast diagnostics](charts/forecast-quality.svg)

![Research progress](charts/research-progress.svg)

The nonzero regular and aggressive eligibility counts show that the corrected expected-return semantics removed Round 33's universal policy bottleneck. They do not rescue the negative ranked tails and are not a reason to loosen risk controls. DirectML tensor execution and OpenCL FP64 LightGBM training were attested. No leverage, testnet or live execution, untouched-period claim, or profitability claim is permitted.

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
        "artifact_class": "three_action_utility_viability_graph_data",
        "round": 34,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": report["binding_sha256"],
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "source_corpus_certificate_sha256": report["training_corpus"][
            "corpus_certificate_sha256"
        ],
        "source_barrier_targets_sha256": report["training_corpus"][
            "barrier_targets_sha256"
        ],
        "source_cache_key": report["training_corpus"]["cache_key"],
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
    parser = argparse.ArgumentParser(description="Publish verified Round 34 evidence.")
    research = ROOT / "docs" / "model-research" / "action-value"
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-034-three-action-utility-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-034-execution-binding.json",
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
