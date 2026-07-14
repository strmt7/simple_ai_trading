"""Publish hash-verified Round 56 predictive evidence and diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.publish_round52_executable_support_hurdle import (  # noqa: E402
    COLORS,
    _artifact,
    _bar_svg,
    _canonical_json,
    _canonical_sha256,
    _file_sha256,
    _line_svg,
    _read_object,
    _validate_finite,
    _write_csv,
    _write_text,
)


ROUND = 56
REPORT_SCHEMA = "round-056-paired-action-report-v1"
PUBLICATION_SCHEMA = "round-056-paired-action-publication-v1"
DESIGN_SCHEMA = "round-056-paired-action-distributional-design-v1"
BINDING_SCHEMA = "round-056-paired-action-execution-binding-v1"
REPORT_CANONICAL_SHA256 = (
    "8fb9cdca77608554e187830d5ff636ea7013a7bce9a4a3a1805afd76e5a8f764"
)
REPORT_FILE_SHA256 = (
    "08fa37b8a5d896786e07cd90262574556531796df48c33373a3057eb2f99b6a7"
)
DESIGN_SHA256 = "893cbc7e24b1125d16e36affa23329a3e18fc74f08131231619870ab831d99d4"
BINDING_SHA256 = "ffdcd1226fe55af6edaf331d4908ec0945b5871fa021035fd2702d0795aeaa9a"
IMPLEMENTATION_COMMIT = "244902e1604f664c1693db8620a782ab88ca7da7"
TREATMENTS = ("baseline_72", "ai_program_augmented")
DISPLAY = {
    "baseline_72": "Baseline",
    "ai_program_augmented": "Governed AI factors",
}
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ACTIONS = ("long", "short")
STRESS_CHARGE_BPS = 16.0


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def _canonical_value(value: Mapping[str, object], digest_key: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(digest_key, ""))
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError(f"{digest_key} does not match canonical content")
    return actual


def _verify_artifact(
    row: Mapping[str, object],
    *,
    digest_key: str = "sha256",
) -> Path:
    path = Path(str(row.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or _file_sha256(path) != str(row.get(digest_key, ""))
    ):
        raise ValueError(f"Round 56 artifact drifted: {path}")
    return path


def _validate_sources(
    report_path: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if _file_sha256(report_path) != REPORT_FILE_SHA256:
        raise ValueError("Round 56 report file hash drifted")
    report = _read_object(report_path, "Round 56 report")
    design = _read_object(design_path, "Round 56 design")
    binding = _read_object(binding_path, "Round 56 binding")
    claims = report.get("claims")
    data = report.get("data")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected"
        or _canonical_value(report, "report_sha256") != REPORT_CANONICAL_SHA256
        or design.get("schema_version") != DESIGN_SCHEMA
        or _canonical_value(design, "design_sha256") != DESIGN_SHA256
        or binding.get("schema_version") != BINDING_SCHEMA
        or _canonical_value(binding, "binding_sha256") != BINDING_SHA256
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("binding_sha256") != BINDING_SHA256
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or binding.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or not isinstance(claims, Mapping)
        or any(value is not False for value in claims.values())
        or not isinstance(data, Mapping)
        or data.get("synthetic_rows") != 0
        or data.get("forbidden_existing_rows_read") is not False
        or report.get("retained_for_separately_frozen_next_design") != []
    ):
        raise ValueError("Round 56 source contracts or claims drifted")
    _validate_finite(report)

    expected_blobs = {
        str(row["path"]): str(row["git_blob_oid"])
        for row in binding["blobs"]
    }
    for source_path, expected_oid in expected_blobs.items():
        if _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{source_path}") != expected_oid:
            raise ValueError(f"Round 56 implementation blob drifted: {source_path}")

    for treatment in TREATMENTS:
        manifest = report["model"]["artifacts"][treatment]
        _verify_artifact(manifest["prediction_cache"])
        _verify_artifact(manifest["calibration"], digest_key="file_sha256")
        for row in manifest["artifacts"]:
            model_path = _verify_artifact(row, digest_key="file_sha256")
            model = _read_object(model_path, "Round 56 model")
            if (
                _canonical_value(model, "model_sha256") != row["model_sha256"]
                or model.get("treatment_id") != treatment
                or model.get("view_id") != row["view_id"]
                or model.get("objective_id") != row["objective_id"]
                or model.get("seed") != row["seed"]
                or row.get("backend_kind") != "opencl"
                or row.get("reload_max_abs_prediction_error_bps") != 0.0
            ):
                raise ValueError(f"Round 56 model identity drifted: {model_path}")

    for row in report["artifacts"].values():
        _verify_artifact(row)
    payoff = report["data"]["payoff"]
    _verify_artifact(
        {
            "path": payoff["path"],
            "bytes": payoff["bytes"],
            "sha256": payoff["file_sha256"],
        }
    )
    _verify_artifact(
        {
            "path": payoff["manifest_path"],
            "bytes": Path(payoff["manifest_path"]).stat().st_size,
            "sha256": payoff["manifest_file_sha256"],
        }
    )
    for row in binding["external_evidence"].values():
        _verify_artifact(row)
    return report, design, binding


def _summary_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for treatment in TREATMENTS:
        result = report["treatments"][treatment]
        point = result["predictive_validation"]["point"]
        lower = result["predictive_validation"]["lower_tail"]
        rows.append(
            {
                "round": ROUND,
                "treatment": treatment,
                "held_forward_start": "2024-01-01",
                "held_forward_end_exclusive": "2024-07-01",
                "rows": result["predictive_validation"]["rows"],
                "timestamps": result["predictive_validation"]["timestamps"],
                "pooled_point_mse_skill": point["pooled_mse_skill"],
                "long_point_mse_skill": point["action_mse_skill"]["long"],
                "short_point_mse_skill": point["action_mse_skill"]["short"],
                "pooled_spearman": point["pooled_spearman"],
                "positive_spearman_months": point["positive_spearman_months"],
                "top_score_quintile_realized_stress_bps": point[
                    "top_score_quintile_realized_mean_bps"
                ],
                "bottom_score_quintile_realized_stress_bps": point[
                    "bottom_score_quintile_realized_mean_bps"
                ],
                "pooled_q20_pinball_skill": lower["pooled_pinball_skill"],
                "long_q20_pinball_skill": lower["action_pinball_skill"]["long"],
                "short_q20_pinball_skill": lower["action_pinball_skill"]["short"],
                "pooled_q20_coverage": lower["pooled_coverage"],
                "long_q20_coverage": lower["action_coverage"]["long"],
                "short_q20_coverage": lower["action_coverage"]["short"],
                "predictive_gate_passed": result["predictive_gate"]["passed"],
                "predictive_gate_failures": ";".join(
                    result["predictive_gate"]["failures"]
                ),
                "economic_status": result["economic_status"],
            }
        )
    return rows


def _monthly_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for treatment in TREATMENTS:
        point = report["treatments"][treatment]["predictive_validation"]["point"]
        for month, value in point["monthly_spearman"].items():
            rows.append(
                {
                    "round": ROUND,
                    "treatment": treatment,
                    "month": month,
                    "spearman": value,
                    "role": "held_forward_predictive_validation",
                }
            )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    fold_ids = [row["fold_id"] for row in report["chronology"]["outer_folds"]]
    rows = []
    for treatment in TREATMENTS:
        artifacts = report["model"]["artifacts"][treatment]["artifacts"]
        for artifact in artifacts:
            skills = artifact["outer_fold_loss_skill"]
            if len(skills) != len(fold_ids):
                raise ValueError("Round 56 fold skill count differs")
            for fold_id, skill in zip(fold_ids, skills, strict=True):
                rows.append(
                    {
                        "round": ROUND,
                        "treatment": treatment,
                        "view": artifact["view_id"],
                        "objective": artifact["objective_id"],
                        "seed": artifact["seed"],
                        "fold": fold_id,
                        "loss_skill": skill,
                        "final_iterations": artifact["final_iterations"],
                        "backend": artifact["backend_kind"],
                        "device": artifact["backend_device"],
                        "reload_max_abs_prediction_error_bps": artifact[
                            "reload_max_abs_prediction_error_bps"
                        ],
                        "model_sha256": artifact["model_sha256"],
                        "file_sha256": artifact["file_sha256"],
                    }
                )
    return rows


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for treatment in TREATMENTS:
        gate = report["treatments"][treatment]["predictive_gate"]
        rows.extend(
            {
                "round": ROUND,
                "candidate": treatment,
                "gate": name,
                "passed": passed,
            }
            for name, passed in gate["checks"].items()
        )
    uplift = report["ai_uplift_gate"]
    rows.extend(
        {
            "round": ROUND,
            "candidate": "ai_uplift",
            "gate": name,
            "passed": passed,
        }
        for name, passed in uplift["checks"].items()
    )
    return rows


def _ai_rows(binding: Mapping[str, object]) -> list[dict[str, object]]:
    ledger_path = Path(binding["external_evidence"]["ai_ledger"]["path"])
    ledger = _read_object(ledger_path, "Round 56 AI ledger")
    rows = []
    for program in ledger["programs"]:
        rows.append(
            {
                "round": ROUND,
                "status": "accepted",
                "model": program["model"],
                "name": program["name"],
                "expression": program["canonical_expression"],
                "mechanism": program["mechanism"],
                "failure_mode": program["failure_mode"],
                "reason": "",
                "program_sha256": program["program_sha256"],
            }
        )
    for rejected in ledger["rejections"]:
        rows.append(
            {
                "round": ROUND,
                "status": "rejected",
                "model": rejected["model"],
                "name": rejected.get("name") or "",
                "expression": "",
                "mechanism": "",
                "failure_mode": "",
                "reason": rejected["reason"],
                "program_sha256": "",
            }
        )
    return rows


def _score_diagnostics(
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payoff_path = Path(report["data"]["payoff"]["path"])
    with np.load(payoff_path, allow_pickle=False) as payoff:
        timestamps = np.asarray(payoff["timestamps_ms"], dtype=np.int64)
        truth = np.stack(
            [payoff["long_net_payoff_bps"], payoff["short_net_payoff_bps"]],
            axis=-1,
        ).astype(np.float64)
        maximum_exit_ms = np.maximum(
            np.max(payoff["long_exit_time_ms"], axis=1),
            np.max(payoff["short_exit_time_ms"], axis=1),
        )
    start_ms = int(np.datetime64("2024-01-01", "ms").astype(np.int64))
    end_ms = int(np.datetime64("2024-07-01", "ms").astype(np.int64))
    mask = (
        (timestamps >= start_ms)
        & (timestamps < end_ms)
        & (maximum_exit_ms < end_ms)
    )
    if np.count_nonzero(mask) != 4367:
        raise ValueError("Round 56 diagnostic interval differs")

    percentile_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    for treatment in TREATMENTS:
        path = Path(
            report["model"]["artifacts"][treatment]["prediction_cache"]["path"]
        )
        with np.load(path, allow_pickle=False) as predictions:
            raw = np.asarray(predictions["point_oof_bps"][:, :, mask], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("Round 56 diagnostic prediction is nonfinite")
        score = np.mean(np.median(raw, axis=1), axis=0)
        target = truth[mask]
        flat_score = score.reshape(-1)
        flat_target = target.reshape(-1)
        selected_action = np.argmax(score, axis=-1)
        best_score = np.take_along_axis(
            score, selected_action[..., None], axis=-1
        )[..., 0]
        best_target = np.take_along_axis(
            target, selected_action[..., None], axis=-1
        )[..., 0]
        for scope, scope_score, scope_target in (
            ("all_action_rows", flat_score, flat_target),
            ("model_preferred_action", best_score.reshape(-1), best_target.reshape(-1)),
        ):
            for fraction in (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01):
                threshold = float(np.quantile(scope_score, 1.0 - fraction))
                selected = scope_score >= threshold
                mean_net = float(np.mean(scope_target[selected]))
                percentile_rows.append(
                    {
                        "round": ROUND,
                        "treatment": treatment,
                        "scope": scope,
                        "top_fraction": fraction,
                        "rows": int(np.count_nonzero(selected)),
                        "raw_score_threshold_bps": threshold,
                        "realized_stress_mean_bps": mean_net,
                        "realized_before_frozen_charge_mean_bps": (
                            mean_net + STRESS_CHARGE_BPS
                        ),
                        "positive_stress_payoff_fraction": float(
                            np.mean(scope_target[selected] > 0.0)
                        ),
                        "post_hoc_consumed_diagnostic": True,
                        "promotion_authority": False,
                    }
                )
        for symbol_index, symbol in enumerate(SYMBOLS):
            source_score = score[:, symbol_index].reshape(-1)
            source_target = target[:, symbol_index].reshape(-1)
            decomposition_rows.append(
                {
                    "round": ROUND,
                    "treatment": treatment,
                    "dimension": "symbol",
                    "value": symbol,
                    "rows": source_target.size,
                    "raw_spearman": float(
                        spearmanr(source_score, source_target).statistic
                    ),
                    "realized_stress_mean_bps": float(np.mean(source_target)),
                    "post_hoc_consumed_diagnostic": True,
                }
            )
        for action_index, action in enumerate(ACTIONS):
            source_score = score[..., action_index].reshape(-1)
            source_target = target[..., action_index].reshape(-1)
            decomposition_rows.append(
                {
                    "round": ROUND,
                    "treatment": treatment,
                    "dimension": "action",
                    "value": action,
                    "rows": source_target.size,
                    "raw_spearman": float(
                        spearmanr(source_score, source_target).statistic
                    ),
                    "realized_stress_mean_bps": float(np.mean(source_target)),
                    "post_hoc_consumed_diagnostic": True,
                }
            )
    return percentile_rows, decomposition_rows


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
        raise ValueError("research progress must contain exactly Rounds 1 through 55")
    ai = report["treatments"]["ai_program_augmented"]
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "paired-action distributional LightGBM + governed AI factors",
            "periods": (
                "rolling OOF 2023-07..2024-06; held-forward validation "
                "2024-01..2024-06; later economics gated"
            ),
            "selection_contaminated": "True",
            "horizon_seconds": "3600",
            "feature_set": "72 paired-action causal features; +2 AI programs",
            "risk_level": "predictive gate only; unlevered; no economic replay",
            "spearman_ic": str(
                ai["predictive_validation"]["point"]["pooled_spearman"]
            ),
            "selected_signals": "0",
            "executable_trades": "0",
            "status": "rejected",
            "source_file": (
                "verified Round 56 report; real 1m Binance futures paths; "
                "16 bps stress charge"
            ),
            "best_model_id": "ai_program_augmented_predictive_gate_failed",
            "ensemble_models": "24",
            "valid_barrier_rows": str(report["data"]["payoff"]["rows"]),
            "calibration_eligible_rows": str(
                report["chronology"]["calibration_validation_timestamps"]
            ),
            "policy_eligible_rows": "0",
            "development_consumed": "True",
            "architecture_gates_passed": "14",
            "architecture_gate_count": "15",
        }
    )
    rows.append(new)
    return rows, fields


def _skill_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Held-forward forecast skill remained small but positive",
        subtitle=(
            "January-June 2024; causal constants are zero-skill baselines; "
            "economics were not evaluated"
        ),
        groups=tuple(
            (
                DISPLAY[str(row["treatment"])],
                (
                    (
                        "Point MSE skill",
                        100.0 * float(row["pooled_point_mse_skill"]),
                        COLORS["blue"],
                    ),
                    (
                        "q20 pinball skill",
                        100.0 * float(row["pooled_q20_pinball_skill"]),
                        COLORS["teal"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=7.0,
        y_label="Skill versus causal constant (%)",
    )


def _monthly_svg(rows: Sequence[Mapping[str, object]]) -> str:
    months = sorted({str(row["month"]) for row in rows})
    positions = {month: float(index) for index, month in enumerate(months)}
    series = []
    for treatment, color in zip(
        TREATMENTS, (COLORS["blue"], COLORS["teal"]), strict=True
    ):
        points = [
            (positions[str(row["month"])], 100.0 * float(row["spearman"]))
            for row in rows
            if row["treatment"] == treatment
        ]
        series.append((DISPLAY[treatment], points, color))
    return _line_svg(
        title="AI factors improved monthly rank consistency",
        subtitle=(
            "Held-forward action-payoff Spearman; AI was positive in 5/6 months, "
            "but still failed the payoff hurdle"
        ),
        series=tuple(series),
        x_labels={positions[month]: month[5:] for month in months},
        y_label="Spearman x 100",
    )


def _quintile_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Rank separation did not overcome realistic costs",
        subtitle=(
            "Held-forward calibrated score quintiles; targets include the frozen "
            "16 bps round-trip stress charge"
        ),
        groups=tuple(
            (
                DISPLAY[str(row["treatment"])],
                (
                    (
                        "Bottom score quintile",
                        float(row["bottom_score_quintile_realized_stress_bps"]),
                        COLORS["red"],
                    ),
                    (
                        "Top score quintile",
                        float(row["top_score_quintile_realized_stress_bps"]),
                        COLORS["amber"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=-20.0,
        y_max=2.0,
        y_label="Mean realized stress payoff (bps)",
        tick_decimals=1,
        value_decimals=2,
    )


def _percentile_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["scope"] == "model_preferred_action"]
    fractions = (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01)
    x_by_fraction = {value: float(index) for index, value in enumerate(fractions)}
    series = []
    for treatment, color in zip(
        TREATMENTS, (COLORS["blue"], COLORS["teal"]), strict=True
    ):
        points = [
            (
                x_by_fraction[fraction],
                float(
                    next(
                        row["realized_stress_mean_bps"]
                        for row in selected
                        if row["treatment"] == treatment
                        and math.isclose(float(row["top_fraction"]), fraction)
                    )
                ),
            )
            for fraction in fractions
        ]
        series.append((DISPLAY[treatment], points, color))
    return _line_svg(
        title="Extreme raw scores were not monotonically profitable",
        subtitle=(
            "Post-hoc consumed-data diagnosis of the model-preferred side; "
            "not a selected policy or promotion result"
        ),
        series=tuple(series),
        x_labels={x_by_fraction[value]: f"top {100 * value:.0f}%" for value in fractions},
        y_label="Mean realized stress payoff (bps)",
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points = []
    for row in rows:
        raw = str(row.get("spearman_ic", "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            points.append((float(row["round"]), 100.0 * value))
    labels = {points[0][0]: str(int(points[0][0])), points[-1][0]: str(ROUND)}
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        if points[0][0] <= value <= points[-1][0]:
            labels[value] = str(int(value))
    return _line_svg(
        title="Optimization research progression",
        subtitle=(
            "Recorded rank statistic by round; targets, horizons, and datasets differ, "
            "so the line is diagnostic rather than a performance claim"
        ),
        series=(("Recorded Spearman", points, COLORS["teal"]),),
        x_labels=labels,
        y_label="Recorded Spearman x 100",
    )


def _clean_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    parent = (ROOT / "docs" / "model-research" / "action-value").resolve()
    if not resolved.is_relative_to(parent) or resolved.name != "latest":
        raise ValueError("publication output must be action-value/latest")
    if resolved.exists():
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _failure_analysis(
    report: Mapping[str, object],
    percentile_rows: Sequence[Mapping[str, object]],
    decomposition_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ai = report["treatments"]["ai_program_augmented"]
    baseline = report["treatments"]["baseline_72"]
    top_two = next(
        row
        for row in percentile_rows
        if row["treatment"] == "ai_program_augmented"
        and row["scope"] == "all_action_rows"
        and math.isclose(float(row["top_fraction"]), 0.02)
    )
    sol = next(
        row
        for row in decomposition_rows
        if row["treatment"] == "ai_program_augmented"
        and row["dimension"] == "symbol"
        and row["value"] == "SOLUSDT"
    )
    analysis: dict[str, object] = {
        "schema_version": "round-056-paired-action-failure-analysis-v1",
        "round": ROUND,
        "source_report_sha256": REPORT_CANONICAL_SHA256,
        "status": "rejected_post_hoc_diagnosis",
        "facts": {
            "baseline_pooled_spearman": baseline["predictive_validation"]["point"][
                "pooled_spearman"
            ],
            "ai_pooled_spearman": ai["predictive_validation"]["point"][
                "pooled_spearman"
            ],
            "baseline_positive_months": baseline["predictive_validation"]["point"][
                "positive_spearman_months"
            ],
            "ai_positive_months": ai["predictive_validation"]["point"][
                "positive_spearman_months"
            ],
            "ai_top_quintile_stress_bps": ai["predictive_validation"]["point"][
                "top_score_quintile_realized_mean_bps"
            ],
            "ai_post_hoc_top_two_percent_stress_bps": top_two[
                "realized_stress_mean_bps"
            ],
            "ai_post_hoc_top_two_percent_before_frozen_charge_bps": top_two[
                "realized_before_frozen_charge_mean_bps"
            ],
            "ai_sol_raw_spearman": sol["raw_spearman"],
            "economic_replay_performed": False,
        },
        "diagnosis": [
            "The paired target and explicit action identity improved structural correctness but did not identify positive after-cost action values.",
            "The two governed AI factors improved pooled and monthly rank consistency but did not establish positive top-quintile payoff or AI uplift.",
            "Post-hoc extreme-score means were non-monotonic, so a tighter threshold would be outcome-driven curve fitting rather than a justified repair.",
            "Hourly state resolution is too coarse to support the requested intraday cadence or exploit short-lived order-flow effects already present in the verified tick warehouse.",
            "SOLUSDT remained the weakest raw rank component; future work must retain all three symbols but model symbol-specific microstructure and costs.",
        ],
        "prohibited_inferences": [
            "profitability",
            "AI uplift",
            "testnet readiness",
            "live trading readiness",
            "leverage readiness",
            "threshold promotion from the post-hoc percentile table",
        ],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    baseline = report["treatments"]["baseline_72"]["predictive_validation"]
    ai = report["treatments"]["ai_program_augmented"]["predictive_validation"]
    return f"""# Round 56: Paired Action Values

> **Rejected development evidence.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 56 trained `24` AMD OpenCL LightGBM artifacts on BTCUSDT, ETHUSDT, and SOLUSDT. Long and short were explicit paired actions. Labels used real one-minute futures paths, gap-through stops, settled funding, and a frozen `16 bps` round-trip stress charge. Model reload error was exactly zero.

| Held-forward metric (Jan-Jun 2024) | Baseline | Governed AI factors |
|---|---:|---:|
| Point MSE skill vs causal constant | {100 * float(baseline['point']['pooled_mse_skill']):.3f}% | {100 * float(ai['point']['pooled_mse_skill']):.3f}% |
| Pooled Spearman | {float(baseline['point']['pooled_spearman']):.5f} | {float(ai['point']['pooled_spearman']):.5f} |
| Positive-Spearman months | {baseline['point']['positive_spearman_months']}/6 | {ai['point']['positive_spearman_months']}/6 |
| Top score quintile, realized stress payoff | {float(baseline['point']['top_score_quintile_realized_mean_bps']):+.2f} bps | {float(ai['point']['top_score_quintile_realized_mean_bps']):+.2f} bps |
| q20 pinball skill | {100 * float(baseline['lower_tail']['pooled_pinball_skill']):.3f}% | {100 * float(ai['lower_tail']['pooled_pinball_skill']):.3f}% |
| q20 coverage | {100 * float(baseline['lower_tail']['pooled_coverage']):.2f}% | {100 * float(ai['lower_tail']['pooled_coverage']):.2f}% |

The baseline failed monthly rank consistency and positive top-quintile payoff. The two accepted Fino1 factor programs improved rank consistency to `5/6` positive months and pooled Spearman to `{float(ai['point']['pooled_spearman']):.5f}`, but the top quintile still lost `{abs(float(ai['point']['top_score_quintile_realized_mean_bps'])):.2f} bps` per action row after stress costs. The AI treatment therefore failed before economic replay. Trades, ROI, drawdown, and leverage were not evaluated.

The run used `24,096` hourly timestamps and `144,576` paired action rows derived from real minute paths. It generated no synthetic rows and did not read October 2024 or later observations. The percentile analysis is explicitly post-hoc and cannot select a policy.

## Evidence

| View | Graph | Source |
|---|---|---|
| Forecast skill | [SVG](charts/predictive-skill.svg) | [CSV](predictive-summary.csv) |
| Monthly rank | [SVG](charts/monthly-rank.svg) | [CSV](monthly-rank.csv) |
| Payoff stratification | [SVG](charts/payoff-stratification.svg) | [CSV](predictive-summary.csv) |
| Extreme-score diagnosis | [SVG](charts/score-percentiles.svg) | [CSV](score-percentiles.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`model-fold-skill.csv`, `decomposition.csv`, `gates.csv`, `ai-factors.csv`, `failure-analysis.json`, and `screen.json` preserve the remaining evidence. Every chart is regenerated from tracked tabular data.
"""


def publish(
    *,
    report_path: Path,
    design_path: Path,
    binding_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, _design, binding = _validate_sources(
        report_path, design_path, binding_path
    )
    summary_rows = _summary_rows(report)
    monthly_rows = _monthly_rows(report)
    model_rows = _model_rows(report)
    gate_rows = _gate_rows(report)
    ai_rows = _ai_rows(binding)
    percentile_rows, decomposition_rows = _score_diagnostics(report)
    progress_rows, progress_fields = _progress_rows(previous_progress_path, report)
    failure = _failure_analysis(report, percentile_rows, decomposition_rows)

    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "predictive-summary.csv", summary_rows)
    _write_csv(output_dir / "monthly-rank.csv", monthly_rows)
    _write_csv(output_dir / "model-fold-skill.csv", model_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(output_dir / "ai-factors.csv", ai_rows)
    _write_csv(output_dir / "score-percentiles.csv", percentile_rows)
    _write_csv(output_dir / "decomposition.csv", decomposition_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "predictive-skill.svg", _skill_svg(summary_rows))
    _write_text(charts / "monthly-rank.svg", _monthly_svg(monthly_rows))
    _write_text(charts / "payoff-stratification.svg", _quintile_svg(summary_rows))
    _write_text(charts / "score-percentiles.svg", _percentile_svg(percentile_rows))
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))
    _write_text(output_dir / "README.md", _readme(report))
    write_json_atomic(output_dir / "failure-analysis.json", failure, indent=2)
    write_json_atomic(output_dir / "screen.json", report, indent=2)

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "published_at_utc": report["generated_at_utc"],
        "publisher_path": "tools/publish_round56_paired_action.py",
        "source": {
            "report_path": str(report_path),
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
            "design_sha256": DESIGN_SHA256,
            "binding_path": str(binding_path.relative_to(ROOT)).replace("\\", "/"),
            "binding_sha256": BINDING_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "dataset_sha256": report["data"]["dataset_sha256"],
            "payoff_dataset_sha256": report["data"]["payoff"]["dataset_sha256"],
            "ai_ledger_sha256": report["ai_factor_research"]["ledger_sha256"],
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
            "post_hoc_threshold_authority": False,
        },
        "result": {
            "treatments": len(TREATMENTS),
            "model_artifacts": 24,
            "outer_model_fits": 288,
            "accepted_ai_factor_programs": report["ai_factor_research"][
                "runtime_accepted_programs"
            ],
            "passed_treatments": 0,
            "ai_uplift_gate_passed": False,
            "forbidden_existing_rows_read": False,
            "synthetic_rows": 0,
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2)
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round56-paired-action-20260714-v1\report.json"
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-056-paired-action-distributional-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-056-paired-action-execution-binding.json",
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
    arguments = _parser().parse_args(argv)
    publication = publish(
        report_path=arguments.report.resolve(),
        design_path=arguments.design.resolve(),
        binding_path=arguments.binding.resolve(),
        previous_progress_path=arguments.progress.resolve(),
        output_dir=arguments.output.resolve(),
    )
    print(
        _canonical_json(
            {
                "round": publication["round"],
                "status": publication["claims"]["status"],
                "publication_canonical_sha256": publication[
                    "publication_canonical_sha256"
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
