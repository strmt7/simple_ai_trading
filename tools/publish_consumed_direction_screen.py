"""Publish verified Round 35 direction-screen evidence and static SVG charts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import html
import json
import math
from pathlib import Path
import sys


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
from tools.run_consumed_direction_screen import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    _canonical_sha256,
    _eligibility_reasons,
    _is_sha256,
    _sha256_file,
    load_direction_screen_binding,
    load_direction_screen_design,
)


PUBLICATION_SCHEMA_VERSION = "consumed-direction-screen-publication-v1"
_VARIANT_ORDER = (
    "full_uniqueness",
    "full_utility_margin",
    "noncycle_uniqueness",
    "noncycle_utility_margin",
    "compact_uniqueness",
    "compact_utility_margin",
)
_SHORT_LABELS = {
    "full_uniqueness": "Full / uniqueness",
    "full_utility_margin": "Full / utility margin",
    "noncycle_uniqueness": "No cycles / uniqueness",
    "noncycle_utility_margin": "No cycles / utility margin",
    "compact_uniqueness": "Compact / uniqueness",
    "compact_utility_margin": "Compact / utility margin",
}


def _validated_source(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design, design_sha = load_direction_screen_design(design_path)
    binding, binding_sha = load_direction_screen_binding(
        binding_path,
        design_path=design_path,
        design_sha256=design_sha,
    )
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, label="Round 35 direction-screen report")
    canonical = dict(report)
    claimed = canonical.pop("report_canonical_sha256", None)
    implementation = binding.get("implementation")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 35
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or not isinstance(implementation, Mapping)
        or report.get("implementation_commit") != implementation.get("commit")
        or not _is_sha256(claimed)
        or _canonical_sha256(canonical) != claimed
        or report.get("architecture_freeze_candidate") is not None
        or report.get("architecture_freeze_eligible_variants") != []
    ):
        raise ValueError("Round 35 source report identity is invalid")
    _validate_tree(report)
    expected_access = {
        "certified_source_materialized_through_development": True,
        "train_used_for_fit": True,
        "early_stop_used_for_early_stopping": True,
        "calibration_prediction_and_metrics": True,
        "policy_prediction_or_metrics": False,
        "development_prediction_or_metrics": False,
        "distant_confirmation_source_materialization": False,
        "distant_confirmation_prediction_or_metrics": False,
    }
    source = report.get("source_evidence")
    runtime = report.get("runtime")
    calibration = report.get("calibration")
    variants = report.get("variant_results")
    design_variants = design.get("variants")
    if (
        report.get("stage_access") != expected_access
        or not isinstance(source, Mapping)
        or source.get("corpus_certificate_sha256")
        != "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
        or source.get("barrier_targets_sha256")
        != "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
        or source.get("cache_key")
        != "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
        or source.get("dataset_rows") != 877_894
        or source.get("event_rows") != 230_941
        or source.get("valid_barrier_rows") != 229_000
        or not isinstance(runtime, Mapping)
        or runtime.get("lightgbm_training_backend_required") != "opencl"
        or runtime.get("lightgbm_gpu_use_dp_required") is not True
        or runtime.get("cpu_fallback_permitted") is not False
        or not isinstance(runtime.get("directml_attestation"), Mapping)
        or runtime["directml_attestation"].get("status") != "pass"
        or not isinstance(calibration, Mapping)
        or calibration.get("event_rows") != 28_581
        or calibration.get("positive_opportunity_rows") != 15_222
        or not isinstance(variants, list)
        or not isinstance(design_variants, list)
        or [item.get("variant") for item in variants if isinstance(item, Mapping)]
        != list(_VARIANT_ORDER)
        or [
            item.get("variant") for item in design_variants if isinstance(item, Mapping)
        ]
        != list(_VARIANT_ORDER)
    ):
        raise ValueError("Round 35 source, runtime, or variant evidence drifted")
    gates = design["architecture_freeze_eligibility"]
    assert isinstance(gates, Mapping)
    for variant in variants:
        if not isinstance(variant, Mapping):
            raise ValueError("Round 35 variant result is invalid")
        model = variant.get("model")
        metrics = variant.get("metrics")
        gain = variant.get("feature_gain")
        relative = (
            Path(str(model.get("artifact_path") or ""))
            if isinstance(model, Mapping)
            else Path()
        )
        artifact_path = evidence_root / relative
        if (
            not isinstance(model, Mapping)
            or not isinstance(metrics, Mapping)
            or not isinstance(gain, list)
            or len(gain) != int(variant.get("feature_count") or -1)
            or variant.get("architecture_freeze_eligible") is not False
            or not isinstance(variant.get("rejection_reasons"), list)
            or not variant["rejection_reasons"]
            or relative.is_absolute()
            or ".." in relative.parts
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != int(model.get("artifact_bytes") or -1)
            or _sha256_file(artifact_path) != model.get("artifact_sha256")
            or model.get("backend_requested") != "directml"
            or model.get("backend_kind") != "opencl"
            or model.get("artifact_reload_max_abs_prediction_error") != 0.0
            or not math.isfinite(
                float(model.get("mirror_swap_max_abs_prediction_error"))
            )
            or float(model["mirror_swap_max_abs_prediction_error"]) > 1e-12
            or model.get("nonfinite_prediction_count") != 0
            or variant["rejection_reasons"]
            != _eligibility_reasons(metrics, gates, nonfinite_predictions=0)
        ):
            raise ValueError("Round 35 model artifact or rejection evidence differs")
        normalized_gain = [float(item["normalized_gain"]) for item in gain]
        if any(
            not math.isfinite(value) or value < 0.0 for value in normalized_gain
        ) or not math.isclose(sum(normalized_gain), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Round 35 feature-gain evidence is invalid")
    return design, report, str(claimed), binding_sha


def _variant_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for item in report["variant_results"]:
        model = item["model"]
        metrics = item["metrics"]
        frozen = metrics["frozen_opportunity_ranked"]
        confidence = metrics["candidate_confidence_ranked"]
        output.append(
            {
                "variant": item["variant"],
                "feature_set": item["feature_set"],
                "weighting": item["weighting"],
                "feature_count": item["feature_count"],
                "pooled_direction_auc": metrics["pooled_direction_auc"],
                "daily_auc_minimum": metrics["daily_auc_minimum"],
                "daily_auc_median": metrics["daily_auc_median"],
                "daily_auc_standard_deviation": metrics["daily_auc_standard_deviation"],
                "days_above_chance": metrics["days_above_chance"],
                "direction_accuracy": metrics["direction_accuracy"],
                "brier_score": metrics["conditional_long_probability_brier_score"],
                "all_routed_mean_stress_net_bps": metrics[
                    "all_routed_mean_stress_net_bps"
                ],
                "frozen_top_100_mean_stress_net_bps": frozen["100"][
                    "mean_stress_net_bps"
                ],
                "frozen_top_500_mean_stress_net_bps": frozen["500"][
                    "mean_stress_net_bps"
                ],
                "frozen_top_1000_mean_stress_net_bps": frozen["1000"][
                    "mean_stress_net_bps"
                ],
                "confidence_top_100_mean_stress_net_bps": confidence["100"][
                    "mean_stress_net_bps"
                ],
                "confidence_top_500_mean_stress_net_bps": confidence["500"][
                    "mean_stress_net_bps"
                ],
                "confidence_top_1000_mean_stress_net_bps": confidence["1000"][
                    "mean_stress_net_bps"
                ],
                "best_iteration": model["best_iteration"],
                "training_runtime_seconds": model["training_runtime_seconds"],
                "model_sha256": model["model_sha256"],
                "artifact_sha256": model["artifact_sha256"],
                "backend": model["backend_kind"],
                "architecture_freeze_eligible": False,
                "rejection_reasons": ";".join(item["rejection_reasons"]),
            }
        )
    return output


def _daily_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report["variant_results"]:
        for day in item["metrics"]["daily"]:
            rows.append(
                {
                    "variant": item["variant"],
                    "date": day["date"],
                    "rows": day["rows"],
                    "routed_rows": day["routed_rows"],
                    "direction_auc": day["direction_auc"],
                    "direction_accuracy": day["direction_accuracy"],
                    "brier_score": day["brier_score"],
                }
            )
    return rows


def _feature_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "variant": item["variant"],
            "feature_set": item["feature_set"],
            "weighting": item["weighting"],
            "rank": feature["rank"],
            "feature": feature["feature"],
            "normalized_gain": feature["normalized_gain"],
        }
        for item in report["variant_results"]
        for feature in item["feature_gain"]
    ]


def _gate_rows(
    design: Mapping[str, object],
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    gates = design["architecture_freeze_eligibility"]
    definitions = (
        (
            "pooled_direction_auc",
            "pooled_direction_auc",
            "minimum_pooled_direction_auc",
            ">=",
            "pooled_direction_auc_gate_failed",
        ),
        (
            "minimum_daily_direction_auc",
            "daily_auc_minimum",
            "minimum_daily_direction_auc",
            ">=",
            "minimum_daily_direction_auc_gate_failed",
        ),
        (
            "median_daily_direction_auc",
            "daily_auc_median",
            "minimum_median_daily_direction_auc",
            ">=",
            "median_daily_direction_auc_gate_failed",
        ),
        (
            "days_above_chance",
            "days_above_chance",
            "minimum_days_above_chance",
            ">=",
            "days_above_chance_gate_failed",
        ),
    )
    rows: list[dict[str, object]] = []
    for item in report["variant_results"]:
        metrics = item["metrics"]
        reasons = set(item["rejection_reasons"])
        for gate, metric_name, threshold_name, comparator, reason in definitions:
            rows.append(
                {
                    "variant": item["variant"],
                    "gate": gate,
                    "observed": metrics[metric_name],
                    "threshold": gates[threshold_name],
                    "comparator": comparator,
                    "passed": reason not in reasons,
                    "failure_reason": reason if reason in reasons else "",
                }
            )
        ranked = (
            (
                "frozen_opportunity_top_100_stress_net",
                metrics["frozen_opportunity_ranked"]["100"]["mean_stress_net_bps"],
                "minimum_frozen_opportunity_top_100_mean_stress_net_bps",
                "frozen_opportunity_top_100_stress_net_gate_failed",
            ),
            (
                "frozen_opportunity_top_500_stress_net",
                metrics["frozen_opportunity_ranked"]["500"]["mean_stress_net_bps"],
                "minimum_frozen_opportunity_top_500_mean_stress_net_bps",
                "frozen_opportunity_top_500_stress_net_gate_failed",
            ),
            (
                "candidate_confidence_top_500_stress_net",
                metrics["candidate_confidence_ranked"]["500"]["mean_stress_net_bps"],
                "minimum_candidate_confidence_top_500_mean_stress_net_bps",
                "candidate_confidence_top_500_stress_net_gate_failed",
            ),
        )
        for gate, observed, threshold_name, reason in ranked:
            rows.append(
                {
                    "variant": item["variant"],
                    "gate": gate,
                    "observed": observed,
                    "threshold": gates[threshold_name],
                    "comparator": ">",
                    "passed": reason not in reasons,
                    "failure_reason": reason if reason in reasons else "",
                }
            )
        rows.append(
            {
                "variant": item["variant"],
                "gate": "nonfinite_predictions",
                "observed": item["model"]["nonfinite_prediction_count"],
                "threshold": gates["maximum_nonfinite_predictions"],
                "comparator": "<=",
                "passed": "nonfinite_prediction_gate_failed" not in reasons,
                "failure_reason": (
                    "nonfinite_prediction_gate_failed"
                    if "nonfinite_prediction_gate_failed" in reasons
                    else ""
                ),
            }
        )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    fields = (
        "model_sha256",
        "artifact_sha256",
        "artifact_bytes",
        "backend_requested",
        "backend_kind",
        "backend_device",
        "lightgbm_version",
        "best_iteration",
        "train_role_rows",
        "train_opportunity_rows",
        "early_stop_role_rows",
        "early_stop_opportunity_rows",
        "utility_margin_scale_bps",
        "train_weight_multiplier_mean",
        "early_stop_weight_multiplier_mean",
        "artifact_reload_max_abs_prediction_error",
        "mirror_swap_max_abs_prediction_error",
        "nonfinite_prediction_count",
        "training_runtime_seconds",
    )
    return [
        {
            "variant": item["variant"],
            **{field: item["model"][field] for field in fields},
        }
        for item in report["variant_results"]
    ]


def _read_progress(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields or not rows or rows[-1].get("round") not in {"34", "35"}:
        raise ValueError("Round 35 prior progress history is invalid")
    rows = [row for row in rows if row.get("round") != "35"]
    if rows[-1].get("round") != "34":
        raise ValueError("Round 35 progress predecessor is not Round 34")
    return fields, rows


def _progress_rows(
    prior_path: Path,
    report: Mapping[str, object],
) -> tuple[list[str], list[dict[str, object]]]:
    fields, rows = _read_progress(prior_path)
    variants = report["variant_results"]
    best_auc = max(variants, key=lambda item: item["metrics"]["pooled_direction_auc"])
    best_top500 = max(
        variants,
        key=lambda item: item["metrics"]["frozen_opportunity_ranked"]["500"][
            "mean_stress_net_bps"
        ],
    )
    best_top100 = max(
        variants,
        key=lambda item: item["metrics"]["frozen_opportunity_ranked"]["100"][
            "mean_stress_net_bps"
        ],
    )
    maximum_passed = max(8 - len(item["rejection_reasons"]) for item in variants)
    row = {field: "" for field in fields}
    row.update(
        {
            "round": 35,
            "stage": "consumed-data conditional direction architecture screen",
            "periods": "2023-05-16..2023-07-06",
            "selection_contaminated": True,
            "horizon_seconds": 900,
            "feature_set": "l1-tape-causal-v8 full/noncycle/compact comparison",
            "risk_level": "research-only; no policy",
            "direction_auc": best_auc["metrics"]["pooled_direction_auc"],
            "selected_signals": 0,
            "executable_trades": 0,
            "status": "rejected",
            "source_file": "verified Round 35 direction-screen report",
            "best_model_id": best_auc["variant"],
            "best_top_500_exact_after_cost_bps": best_top500["metrics"][
                "frozen_opportunity_ranked"
            ]["500"]["mean_stress_net_bps"],
            "after_cost_diagnostic_rows": report["calibration"][
                "positive_opportunity_rows"
            ],
            "valid_barrier_rows": report["source_evidence"]["valid_barrier_rows"],
            "calibration_eligible_rows": 0,
            "development_consumed": False,
            "opportunity_auc": 0.6636738763493204,
            "top_100_exact_after_cost_bps": best_top100["metrics"][
                "frozen_opportunity_ranked"
            ]["100"]["mean_stress_net_bps"],
            "architecture_gates_passed": maximum_passed,
            "architecture_gate_count": 8,
        }
    )
    rows.append(row)
    return fields, rows


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(description)}</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Segoe UI",Arial,sans-serif;letter-spacing:0}.title{font-size:24px;font-weight:700;fill:#17202a}.subtitle{font-size:14px;fill:#4b5563}.label{font-size:13px;fill:#263238}.value{font-size:13px;font-weight:650;fill:#17202a}.axis{font-size:12px;fill:#64748b}.note{font-size:12px;fill:#6b7280}.grid{stroke:#d8dee7;stroke-width:1}.zero{stroke:#9f1239;stroke-width:2}.gate{stroke:#b91c1c;stroke-width:2;stroke-dasharray:7 5}</style>',
        f'<text x="48" y="42" class="title">{html.escape(title)}</text>',
        f'<text x="48" y="66" class="subtitle">{html.escape(description)}</text>',
    ]


def _auc_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1320, 600
    left, right, top, bottom = 300, 70, 112, 70
    chart_width = width - left - right
    minimum, maximum = 0.45, 0.60

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * chart_width

    svg = _svg_start(
        width,
        height,
        "Direct direction improved, but no variant reached the frozen gate",
        "Pooled ROC AUC with daily minimum and median; consumed BTCUSDT calibration dates only.",
    )
    for tick in (0.45, 0.50, 0.55, 0.60):
        px = x(tick)
        svg.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height - bottom}" class="grid"/>'
        )
        svg.append(
            f'<text x="{px:.1f}" y="{height - bottom + 24}" text-anchor="middle" class="axis">{tick:.2f}</text>'
        )
    gate_x = x(0.55)
    svg.append(
        f'<line x1="{gate_x:.1f}" y1="{top - 8}" x2="{gate_x:.1f}" y2="{height - bottom}" class="gate"/>'
    )
    svg.append(
        f'<text x="{gate_x + 8:.1f}" y="{top - 14}" class="axis" fill="#b91c1c">pooled gate 0.55</text>'
    )
    for index, row in enumerate(rows):
        y = top + 34 + index * 68
        pooled = float(row["pooled_direction_auc"])
        daily_min = float(row["daily_auc_minimum"])
        daily_median = float(row["daily_auc_median"])
        label = _SHORT_LABELS[str(row["variant"])]
        svg.append(
            f'<text x="{left - 18}" y="{y + 5}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        svg.append(
            f'<line x1="{x(daily_min):.1f}" y1="{y}" x2="{x(daily_median):.1f}" y2="{y}" stroke="#64748b" stroke-width="5" stroke-linecap="round"/>'
        )
        svg.append(f'<circle cx="{x(daily_min):.1f}" cy="{y}" r="6" fill="#b45309"/>')
        svg.append(
            f'<circle cx="{x(daily_median):.1f}" cy="{y}" r="6" fill="#2563eb"/>'
        )
        color = (
            "#0f766e"
            if pooled == max(float(item["pooled_direction_auc"]) for item in rows)
            else "#334155"
        )
        svg.append(
            f'<rect x="{x(pooled) - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" fill="{color}"/>'
        )
        svg.append(
            f'<text x="{x(pooled):.1f}" y="{y - 14}" text-anchor="middle" class="value">{pooled:.4f}</text>'
        )
    svg.extend(
        [
            f'<circle cx="{left}" cy="{height - 28}" r="5" fill="#b45309"/><text x="{left + 12}" y="{height - 24}" class="note">daily minimum</text>',
            f'<circle cx="{left + 150}" cy="{height - 28}" r="5" fill="#2563eb"/><text x="{left + 162}" y="{height - 24}" class="note">daily median</text>',
            f'<rect x="{left + 300}" y="{height - 34}" width="11" height="11" fill="#334155"/><text x="{left + 318}" y="{height - 24}" class="note">pooled AUC</text>',
            f'<text x="{width - 48}" y="{height - 24}" text-anchor="end" class="note">No result is out-of-sample or executable.</text>',
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def _tails_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1380, 650
    left, right, top, bottom = 310, 80, 118, 72
    chart_width = width - left - right
    minimum, maximum = -30.0, 10.0

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * chart_width

    svg = _svg_start(
        width,
        height,
        "Ranked after-cost tails remained economically negative",
        "Mean stress net return in basis points; the zero line is the minimum strict pass boundary.",
    )
    for tick in (-30, -20, -10, 0, 10):
        px = x(float(tick))
        css = "zero" if tick == 0 else "grid"
        svg.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height - bottom}" class="{css}"/>'
        )
        svg.append(
            f'<text x="{px:.1f}" y="{height - bottom + 24}" text-anchor="middle" class="axis">{tick:+d}</text>'
        )
    for index, row in enumerate(rows):
        y = top + 34 + index * 70
        label = _SHORT_LABELS[str(row["variant"])]
        values = (
            (
                "frozen 100",
                float(row["frozen_top_100_mean_stress_net_bps"]),
                "#b45309",
                -12,
            ),
            (
                "frozen 500",
                float(row["frozen_top_500_mean_stress_net_bps"]),
                "#0f766e",
                0,
            ),
            (
                "confidence 500",
                float(row["confidence_top_500_mean_stress_net_bps"]),
                "#2563eb",
                12,
            ),
        )
        svg.append(
            f'<text x="{left - 18}" y="{y + 5}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        for _name, value, color, offset in values:
            svg.append(
                f'<circle cx="{x(value):.1f}" cy="{y + offset}" r="6" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{x(value) + 10:.1f}" y="{y + offset + 4}" class="value">{value:+.2f}</text>'
            )
    legend_y = height - 26
    for offset, (name, color) in enumerate(
        (
            ("frozen rank top 100", "#b45309"),
            ("frozen rank top 500", "#0f766e"),
            ("candidate confidence top 500", "#2563eb"),
        )
    ):
        lx = left + offset * 235
        svg.append(
            f'<circle cx="{lx}" cy="{legend_y - 4}" r="5" fill="{color}"/><text x="{lx + 12}" y="{legend_y}" class="note">{name}</text>'
        )
    svg.append(
        f'<text x="{width - 48}" y="{legend_y}" text-anchor="end" class="note">Stress targets include fees, slippage, latency, and barrier execution.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _daily_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1320, 650
    left, top = 300, 130
    cell_width, cell_height = 170, 62
    dates = sorted({str(row["date"]) for row in rows})
    lookup = {
        (str(row["variant"]), str(row["date"])): float(row["direction_auc"])
        for row in rows
    }
    svg = _svg_start(
        width,
        height,
        "Daily direction quality was not stable across variants",
        "ROC AUC by UTC date; green is above chance, amber near chance, red below chance.",
    )
    for column, date in enumerate(dates):
        x = left + column * cell_width
        svg.append(
            f'<text x="{x + cell_width / 2:.1f}" y="{top - 18}" text-anchor="middle" class="axis">{date}</text>'
        )
    for row_index, variant in enumerate(_VARIANT_ORDER):
        y = top + row_index * cell_height
        svg.append(
            f'<text x="{left - 18}" y="{y + cell_height / 2 + 5:.1f}" text-anchor="end" class="label">{html.escape(_SHORT_LABELS[variant])}</text>'
        )
        for column, date in enumerate(dates):
            value = lookup[(variant, date)]
            if value >= 0.53:
                color = "#b7e4c7"
            elif value > 0.50:
                color = "#d8f3dc"
            elif value >= 0.48:
                color = "#fde68a"
            else:
                color = "#fecaca"
            x = left + column * cell_width
            svg.append(
                f'<rect x="{x + 4}" y="{y + 4}" width="{cell_width - 8}" height="{cell_height - 8}" rx="4" fill="{color}" stroke="#cbd5e1"/>'
            )
            svg.append(
                f'<text x="{x + cell_width / 2:.1f}" y="{y + cell_height / 2 + 5:.1f}" text-anchor="middle" class="value">{value:.4f}</text>'
            )
    svg.append(
        f'<text x="{left}" y="{height - 32}" class="note">All five dates were already consumed. Cell color encodes discrimination only, not return or trading authority.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _features_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 900
    columns, panel_width, panel_height = 2, 700, 250
    left, top = 48, 112
    by_variant: dict[str, list[Mapping[str, object]]] = {
        name: [] for name in _VARIANT_ORDER
    }
    for row in rows:
        if int(row["rank"]) <= 5:
            by_variant[str(row["variant"])].append(row)
    svg = _svg_start(
        width,
        height,
        "Feature gain shifted, but no representation established an edge",
        "Top five normalized LightGBM gain shares per prespecified variant; gain is diagnostic, not causal evidence.",
    )
    for index, variant in enumerate(_VARIANT_ORDER):
        column = index % columns
        row_index = index // columns
        px = left + column * (panel_width + 24)
        py = top + row_index * panel_height
        svg.append(
            f'<text x="{px}" y="{py}" class="value">{html.escape(_SHORT_LABELS[variant])}</text>'
        )
        entries = sorted(by_variant[variant], key=lambda item: int(item["rank"]))
        maximum = max(float(item["normalized_gain"]) for item in entries)
        for item_index, item in enumerate(entries):
            y = py + 30 + item_index * 38
            value = float(item["normalized_gain"])
            bar = 260 * value / maximum if maximum > 0 else 0
            feature = str(item["feature"])
            svg.append(
                f'<text x="{px}" y="{y + 14}" class="label">{html.escape(feature)}</text>'
            )
            svg.append(
                f'<rect x="{px + 330}" y="{y}" width="{bar:.1f}" height="20" fill="#0f766e"/>'
            )
            svg.append(
                f'<text x="{px + 600}" y="{y + 15}" class="value">{value:.3f}</text>'
            )
    svg.append(
        f'<text x="{left}" y="{height - 30}" class="note">OFI = order-flow imbalance. Deterministic cycle removal improved one tail but did not pass the complete gate set.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _readme(
    report: Mapping[str, object],
    variant_rows: Sequence[Mapping[str, object]],
) -> str:
    best_auc = max(variant_rows, key=lambda row: float(row["pooled_direction_auc"]))
    best_tail = max(
        variant_rows,
        key=lambda row: float(row["frozen_top_500_mean_stress_net_bps"]),
    )
    return f"""# Round 35: direct direction screen rejected

**Rejected without a viability candidate or trading authority.** Six prespecified mirror-equivariant LightGBM variants were trained sequentially on the same consumed BTCUSDT roles. Direct binary side-superiority learning improved discrimination over Round 34, but no variant passed the complete stability and after-cost gate set.

| Evidence | Verified result |
| --- | ---: |
| Source materialization | 2023-05-16 to 2023-07-06 UTC |
| Calibration metrics | 2023-06-21 to 2023-06-25 UTC |
| Causal one-second rows | {report["source_evidence"]["dataset_rows"]:,} |
| CUSUM events / valid barrier outcomes | {report["source_evidence"]["event_rows"]:,} / {report["source_evidence"]["valid_barrier_rows"]:,} |
| Calibration opportunity rows | {report["calibration"]["positive_opportunity_rows"]:,} |
| Variants / eligible variants | 6 / 0 |
| Best pooled direction ROC AUC | {float(best_auc["pooled_direction_auc"]):.4f} ({_SHORT_LABELS[str(best_auc["variant"])]}) |
| Best daily minimum / median AUC | {float(best_auc["daily_auc_minimum"]):.4f} / {float(best_auc["daily_auc_median"]):.4f} |
| Best frozen top-500 stress mean | {float(best_tail["frozen_top_500_mean_stress_net_bps"]):+.2f} bps ({_SHORT_LABELS[str(best_tail["variant"])]}) |
| Best frozen top-100 stress mean | {max(float(row["frozen_top_100_mean_stress_net_bps"]) for row in variant_rows):+.2f} bps |
| DirectML / LightGBM training | pass / OpenCL FP64 |
| Architecture-freeze candidate | none |

![Direction AUC comparison](charts/direction-auc.svg)

![After-cost ranked tails](charts/after-cost-tails.svg)

![Daily direction AUC](charts/daily-direction-auc.svg)

![Feature gain](charts/feature-gain.svg)

![Research progress](charts/research-progress.svg)

The isolated positive `+0.52 bps` top-500 result belongs to the noncycle utility-margin variant, whose top-100 and confidence-ranked top-500 means were negative and which failed six of eight gates. It is not evidence of profitability. Full feature gain remained concentrated in OFI and deterministic cycle variables; removing cycles improved one ranked tail but reduced pooled and worst-day stability. Utility-margin weighting did not provide a consistent improvement.

This screen is post-hoc discovery on already-consumed BTCUSDT dates. No ETHUSDT or SOLUSDT result, out-of-sample result, leverage, portfolio return, testnet/live execution, or profitability claim is permitted.

Data: [variants.csv](variants.csv) | [daily.csv](daily.csv) | [features.csv](features.csv) | [gates.csv](gates.csv) | [models.csv](models.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
"""


def _clean_output(output_dir: Path, expected: set[Path]) -> None:
    if not output_dir.exists():
        return
    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_file() and path not in expected:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def publish(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Validate source evidence and atomically regenerate the latest-only tree."""

    design, report, source_report_sha, binding_sha = _validated_source(
        evidence_root=evidence_root,
        design_path=design_path,
        binding_path=binding_path,
    )
    variants = _variant_rows(report)
    daily = _daily_rows(report)
    features = _feature_rows(report)
    gates = _gate_rows(design, report)
    models = _model_rows(report)
    progress_fields, progress = _progress_rows(prior_progress_path, report)
    output_dir = output_dir.resolve()
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "variants.csv",
        output_dir / "daily.csv",
        output_dir / "features.csv",
        output_dir / "gates.csv",
        output_dir / "models.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "direction-auc.svg",
        charts / "after-cost-tails.svg",
        charts / "daily-direction-auc.svg",
        charts / "feature-gain.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    _write_csv(output_dir / "variants.csv", variants)
    _write_csv(output_dir / "daily.csv", daily)
    _write_csv(output_dir / "features.csv", features)
    _write_csv(output_dir / "gates.csv", gates)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report, variants))
    _write_text(charts / "direction-auc.svg", _auc_svg(variants))
    _write_text(charts / "after-cost-tails.svg", _tails_svg(variants))
    _write_text(charts / "daily-direction-auc.svg", _daily_svg(daily))
    _write_text(charts / "feature-gain.svg", _features_svg(features))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact_class": "consumed_direction_screen_graph_data",
        "round": 35,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "source_corpus_certificate_sha256": report["source_evidence"][
            "corpus_certificate_sha256"
        ],
        "source_barrier_targets_sha256": report["source_evidence"][
            "barrier_targets_sha256"
        ],
        "source_cache_key": report["source_evidence"]["cache_key"],
        "variant_count": 6,
        "architecture_freeze_eligible_variants": [],
        "architecture_freeze_candidate": None,
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
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(
        description="Publish verified Round 35 direction-screen evidence.",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-035-consumed-direction-screen-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-035-direction-screen-execution-binding.json",
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
