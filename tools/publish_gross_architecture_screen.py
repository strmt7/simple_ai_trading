from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

try:
    from tools.publish_action_value_discovery import (
        _canonical_sha256,
        _sha256,
        _write_csv,
        _write_text,
    )
    from tools.run_gross_architecture_screen import load_gross_architecture_design
except ModuleNotFoundError:
    from publish_action_value_discovery import (
        _canonical_sha256,
        _sha256,
        _write_csv,
        _write_text,
    )
    from run_gross_architecture_screen import load_gross_architecture_design


_FINAL_BASELINE_ID = "lightgbm-gross-baseline"
_TOP_ROW_COUNTS = (100, 500, 1_000)
_CANDIDATE_FIELDS = (
    "candidate_id",
    "model_family",
    "selection_stage",
    "selected_for_stage_two",
    "status",
    "backend_kind",
    "backend_device",
    "best_epoch",
    "calibration_direction_auc",
    "calibration_spearman_ic",
    "calibration_mae_bps",
    "calibration_zero_mae_bps",
    "calibration_top_500_gross_bps",
    "calibration_top_500_exact_after_cost_bps",
    "policy_direction_auc",
    "policy_spearman_ic",
    "policy_mae_bps",
    "policy_zero_mae_bps",
    "policy_top_500_gross_bps",
    "policy_top_500_exact_after_cost_bps",
    "development_direction_auc",
    "development_spearman_ic",
    "development_mae_bps",
    "development_zero_mae_bps",
    "development_top_100_gross_bps",
    "development_top_100_exact_after_cost_bps",
    "development_top_500_gross_bps",
    "development_top_500_exact_after_cost_bps",
    "development_top_500_exact_positive_rate",
    "development_top_1000_gross_bps",
    "development_top_1000_exact_after_cost_bps",
    "model_sha256",
    "artifact_path",
    "artifact_sha256",
    "rejection_reasons",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish hash-bound Round 13 gross-architecture evidence",
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
        raise ValueError(f"non-finite gross-architecture {label}")
    return parsed


def _canonical_payload_hash(payload: Mapping[str, object], field: str) -> str:
    claimed = str(payload.get(field) or "")
    canonical = dict(payload)
    canonical.pop(field, None)
    if len(claimed) != 64 or claimed != _canonical_sha256(canonical):
        raise ValueError(f"gross-architecture {field} binding is invalid")
    return claimed


def _top_row(metrics: Mapping[str, object], requested: int) -> Mapping[str, object]:
    rows = metrics.get("top_rows")
    if not isinstance(rows, list):
        raise ValueError("gross-architecture top-row evidence is missing")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and int(row.get("requested_rows") or 0) == requested
    ]
    if len(matches) != 1:
        raise ValueError("gross-architecture top-row evidence is ambiguous")
    row = matches[0]
    if (
        row.get("portfolio_claim") is not False
        or int(row.get("rows") or 0) != requested
    ):
        raise ValueError("gross-architecture top-row contract is invalid")
    for key in (
        "mean_signed_gross_bps",
        "mean_exact_after_cost_bps",
        "signed_gross_positive_rate",
        "exact_after_cost_positive_rate",
    ):
        _finite(row.get(key), label=key)
    return row


def _validate_metrics(metrics: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"gross-architecture {label} metrics are missing")
    rows = int(metrics.get("rows") or 0)
    eligible = int(metrics.get("exact_after_cost_eligible_rows") or 0)
    ratio = _finite(
        metrics.get("exact_after_cost_eligible_ratio"),
        label=f"{label} eligible ratio",
    )
    if rows <= 0 or eligible <= 0 or eligible > rows or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"gross-architecture {label} row counts are invalid")
    if abs(ratio - eligible / rows) > 1e-12:
        raise ValueError(f"gross-architecture {label} eligible ratio drifted")
    for key in (
        "direction_auc",
        "spearman_information_coefficient",
        "mean_absolute_error_bps",
        "zero_baseline_mae_bps",
    ):
        _finite(metrics.get(key), label=f"{label} {key}")
    if not 0.0 <= float(metrics["direction_auc"]) <= 1.0:
        raise ValueError(f"gross-architecture {label} AUC is invalid")
    for requested in _TOP_ROW_COUNTS:
        _top_row(metrics, requested)
    return metrics


def _validate_artifact(
    evidence_root: Path,
    artifact: object,
    *,
    candidate_id: str,
) -> dict[str, object]:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"gross-architecture artifact is missing: {candidate_id}")
    relative = Path(str(artifact.get("path") or ""))
    if (
        not relative.name
        or relative.is_absolute()
        or len(relative.parts) != 1
        or ".." in relative.parts
    ):
        raise ValueError(f"gross-architecture artifact path is unsafe: {candidate_id}")
    path = evidence_root / relative
    if (
        not path.is_file()
        or int(artifact.get("bytes") or -1) != path.stat().st_size
        or str(artifact.get("sha256") or "") != _sha256(path)
    ):
        raise ValueError(
            f"gross-architecture artifact integrity failed: {candidate_id}"
        )
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _metric_columns(prefix: str, metrics: Mapping[str, object]) -> dict[str, object]:
    top_500 = _top_row(metrics, 500)
    return {
        f"{prefix}_direction_auc": metrics["direction_auc"],
        f"{prefix}_spearman_ic": metrics["spearman_information_coefficient"],
        f"{prefix}_mae_bps": metrics["mean_absolute_error_bps"],
        f"{prefix}_zero_mae_bps": metrics["zero_baseline_mae_bps"],
        f"{prefix}_top_500_gross_bps": top_500["mean_signed_gross_bps"],
        f"{prefix}_top_500_exact_after_cost_bps": top_500["mean_exact_after_cost_bps"],
    }


def _validated_evidence(
    evidence_root: Path,
    design: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    report_path = evidence_root / "report.json"
    report = _read_json(report_path)
    _canonical_payload_hash(report, "report_sha256")
    if (
        report.get("schema_version") != "gross-architecture-screen-report-v1"
        or report.get("status") not in {"rejected", "research_candidate"}
        or report.get("round") != 13
        or report.get("design_sha256") != design["design_sha256"]
        or report.get("terminal_holdout_accessed") is not False
        or report.get("development_window_is_consumed") is not True
        or report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
        or report.get("portfolio_claim") is not False
        or report.get("leverage_applied") is not False
    ):
        raise ValueError("gross-architecture report contract is invalid")

    successive = report.get("successive_halving")
    final_results = report.get("final_results")
    dataset = report.get("dataset")
    if (
        not isinstance(successive, Mapping)
        or not isinstance(final_results, list)
        or not isinstance(dataset, Mapping)
    ):
        raise ValueError("gross-architecture report sections are incomplete")
    stage_one = successive.get("stage_one_results")
    selected = successive.get("selected_candidate_ids")
    if not isinstance(stage_one, list) or not isinstance(selected, list):
        raise ValueError("gross-architecture successive-halving evidence is incomplete")
    selected_ids = tuple(str(value) for value in selected)
    designed_ids = tuple(
        str(value["candidate_id"])
        for value in design["neural_candidates"]
        if isinstance(value, Mapping)
    )
    stage_one_ids = tuple(
        str(value.get("candidate_id") or "")
        for value in stage_one
        if isinstance(value, Mapping)
    )
    if (
        len(stage_one_ids) != len(stage_one)
        or set(stage_one_ids) != set(designed_ids)
        or len(selected_ids) != 2
        or len(set(selected_ids)) != len(selected_ids)
        or not set(selected_ids).issubset(stage_one_ids)
    ):
        raise ValueError("gross-architecture candidate selection is inconsistent")

    stage_rows: list[dict[str, object]] = []
    stage_by_id: dict[str, Mapping[str, object]] = {}
    for raw in stage_one:
        assert isinstance(raw, Mapping)
        candidate_id = str(raw["candidate_id"])
        artifact = raw.get("artifact")
        metrics = _validate_metrics(
            raw.get("calibration_metrics"),
            label=f"{candidate_id} calibration",
        )
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("trading_authority") is not False
            or artifact.get("execution_claim") is not False
            or artifact.get("profitability_claim") is not False
            or len(str(artifact.get("model_sha256") or "")) != 64
        ):
            raise ValueError(
                f"gross-architecture stage-one artifact is invalid: {candidate_id}"
            )
        stage_by_id[candidate_id] = raw
        top_500 = _top_row(metrics, 500)
        stage_rows.append(
            {
                "candidate_id": candidate_id,
                "model_family": str(artifact.get("model_family") or ""),
                "selection_stage": (
                    "stage_two_final"
                    if candidate_id in selected_ids
                    else "eliminated_stage_one"
                ),
                "selected_for_stage_two": candidate_id in selected_ids,
                "status": (
                    "advanced" if candidate_id in selected_ids else "eliminated"
                ),
                "backend_kind": str(artifact.get("backend_kind") or ""),
                "backend_device": str(artifact.get("backend_device") or ""),
                "best_epoch": int(artifact.get("best_epoch") or 0),
                **_metric_columns("calibration", metrics),
                "model_sha256": str(artifact["model_sha256"]),
                "artifact_path": "",
                "artifact_sha256": "",
                "rejection_reasons": (
                    ""
                    if candidate_id in selected_ids
                    else "successive_halving_elimination"
                ),
                "calibration_top_500_gross_bps": top_500["mean_signed_gross_bps"],
                "calibration_top_500_exact_after_cost_bps": top_500[
                    "mean_exact_after_cost_bps"
                ],
            }
        )

    final_ids = tuple(
        str(value.get("candidate_id") or "")
        for value in final_results
        if isinstance(value, Mapping)
    )
    expected_final_ids = set(selected_ids) | {_FINAL_BASELINE_ID}
    if (
        len(final_ids) != len(final_results)
        or len(set(final_ids)) != len(final_ids)
        or set(final_ids) != expected_final_ids
    ):
        raise ValueError("gross-architecture final candidate set is inconsistent")

    final_rows: list[dict[str, object]] = []
    for raw in final_results:
        assert isinstance(raw, Mapping)
        candidate_id = str(raw["candidate_id"])
        artifact = raw.get("artifact")
        policy = _validate_metrics(
            raw.get("policy_metrics"),
            label=f"{candidate_id} policy",
        )
        development = _validate_metrics(
            raw.get("development_metrics"),
            label=f"{candidate_id} development",
        )
        artifact_file = _validate_artifact(
            evidence_root,
            raw.get("artifact_file"),
            candidate_id=candidate_id,
        )
        reasons = raw.get("rejection_reasons")
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(reasons, list)
            or raw.get("status") not in {"rejected", "research_candidate"}
            or raw.get("trading_authority") is not False
            or raw.get("execution_claim") is not False
            or raw.get("profitability_claim") is not False
            or artifact.get("trading_authority") is not False
            or artifact.get("execution_claim") is not False
            or artifact.get("profitability_claim") is not False
            or len(str(artifact.get("model_sha256") or "")) != 64
        ):
            raise ValueError(
                f"gross-architecture final artifact is invalid: {candidate_id}"
            )
        if (raw["status"] == "rejected") != bool(reasons):
            raise ValueError(f"gross-architecture status/reason drift: {candidate_id}")
        if candidate_id != _FINAL_BASELINE_ID:
            stage = stage_by_id[candidate_id]
            if artifact.get("spec") != stage["artifact"].get("spec"):
                raise ValueError(
                    f"gross-architecture candidate specification drift: {candidate_id}"
                )
            calibration = _validate_metrics(
                stage.get("calibration_metrics"),
                label=f"{candidate_id} calibration",
            )
            calibration_columns = _metric_columns("calibration", calibration)
        else:
            calibration_columns = {}
        top_100 = _top_row(development, 100)
        top_500 = _top_row(development, 500)
        top_1000 = _top_row(development, 1_000)
        final_rows.append(
            {
                "candidate_id": candidate_id,
                "model_family": str(artifact.get("model_family") or ""),
                "selection_stage": (
                    "baseline_final"
                    if candidate_id == _FINAL_BASELINE_ID
                    else "stage_two_final"
                ),
                "selected_for_stage_two": candidate_id != _FINAL_BASELINE_ID,
                "status": str(raw["status"]),
                "backend_kind": str(artifact.get("backend_kind") or ""),
                "backend_device": str(artifact.get("backend_device") or ""),
                "best_epoch": (
                    ""
                    if artifact.get("best_epoch") is None
                    else int(artifact["best_epoch"])
                ),
                **calibration_columns,
                **_metric_columns("policy", policy),
                **_metric_columns("development", development),
                "development_top_100_gross_bps": top_100["mean_signed_gross_bps"],
                "development_top_100_exact_after_cost_bps": top_100[
                    "mean_exact_after_cost_bps"
                ],
                "development_top_500_exact_positive_rate": top_500[
                    "exact_after_cost_positive_rate"
                ],
                "development_top_1000_gross_bps": top_1000["mean_signed_gross_bps"],
                "development_top_1000_exact_after_cost_bps": top_1000[
                    "mean_exact_after_cost_bps"
                ],
                "model_sha256": str(artifact["model_sha256"]),
                "artifact_path": artifact_file["path"],
                "artifact_sha256": artifact_file["sha256"],
                "rejection_reasons": ";".join(str(reason) for reason in reasons),
            }
        )

    expected_report_status = (
        "research_candidate"
        if any(row["status"] == "research_candidate" for row in final_results)
        else "rejected"
    )
    if report["status"] != expected_report_status:
        raise ValueError("gross-architecture report status drifted")
    return report, stage_rows, final_rows


def _candidate_label(candidate_id: object) -> str:
    labels = {
        "mlp-bounded-gmadl": "MLP + bounded GMADL",
        "mlp-huber-direction": "MLP + Huber/direction",
        _FINAL_BASELINE_ID: "LightGBM baseline",
    }
    return labels.get(str(candidate_id), str(candidate_id))


def _after_cost_svg(rows: Sequence[Mapping[str, object]]) -> str:
    values = [
        _finite(row[key], label=key)
        for row in rows
        for key in (
            "development_top_500_gross_bps",
            "development_top_500_exact_after_cost_bps",
        )
    ]
    bound = max(10.0, math.ceil(max(abs(value) for value in values) / 2.0) * 2.0)
    lower, upper = -bound, bound
    width, height = 1500, 520
    left, right, top = 330, 90, 150
    chart_width = width - left - right
    zero_x = left + chart_width * (0.0 - lower) / (upper - lower)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Gross signal did not clear taker costs</title>',
        '<desc id="desc">Development-window top 500 signed gross and exact after-cost means for three models. Every gross mean was positive and every after-cost mean was negative.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Gross signal did not clear taker costs</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Consumed development window; top 500 overlapping forecasts per model, not trades or portfolio ROI.</text>',
    ]
    for index in range(5):
        value = lower + (upper - lower) * index / 4.0
        x = left + chart_width * index / 4.0
        lines.append(
            f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 80}" stroke="#dce3e8" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">{value:+.0f}</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * 92
        lines.append(
            f'<text x="{left - 24}" y="{y + 38}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#263746">{html.escape(_candidate_label(row["candidate_id"]))}</text>'
        )
        for offset, key, color, label in (
            (8, "development_top_500_gross_bps", "#287f9e", "gross"),
            (42, "development_top_500_exact_after_cost_bps", "#b42318", "exact net"),
        ):
            value = _finite(row[key], label=key)
            x = left + chart_width * (value - lower) / (upper - lower)
            start = min(x, zero_x)
            bar_width = max(2.0, abs(x - zero_x))
            lines.append(
                f'<rect x="{start:.1f}" y="{y + offset}" width="{bar_width:.1f}" height="25" rx="3" fill="{color}"/>'
            )
            anchor = "start" if value >= 0.0 else "end"
            text_x = x + 9 if value >= 0.0 else x - 9
            lines.append(
                f'<text x="{text_x:.1f}" y="{y + offset + 18}" text-anchor="{anchor}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="{color}">{label} {value:+.2f}</text>'
            )
    lines.extend(
        [
            f'<line x1="{zero_x:.1f}" y1="{top - 18}" x2="{zero_x:.1f}" y2="{height - 80}" stroke="#17212b" stroke-width="2"/>',
            '<text x="750" y="476" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Basis points; exact net includes 750 ms latency, 5 bps fee and 1 bps slippage per side.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 520
    panels = (
        ("Direction AUC", "development_direction_auc", 0.5, 0.58, 90, 650),
        ("Spearman IC", "development_spearman_ic", 0.0, 0.14, 790, 650),
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">All final models carried measurable gross forecast information</title>',
        '<desc id="desc">Development direction AUC and Spearman information coefficient for two MLP candidates and the LightGBM benchmark.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">All final models carried measurable gross forecast information</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Forecast quality alone is insufficient: the separate execution-cost gate rejected every model.</text>',
    ]
    colors = ("#287f9e", "#4c7b52", "#6e7781")
    for title, key, lower, upper, left, chart_width in panels:
        top, chart_height = 146, 250
        lines.append(
            f'<text x="{left}" y="124" font-family="Segoe UI, Arial, sans-serif" font-size="18" font-weight="700" fill="#263746">{title}</text>'
        )
        for index in range(5):
            value = lower + (upper - lower) * index / 4.0
            y = top + chart_height - chart_height * (value - lower) / (upper - lower)
            lines.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#dce3e8" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#65727d">{value:.3f}</text>'
            )
        bar_width = 110
        gap = (chart_width - len(rows) * bar_width) / (len(rows) + 1)
        for index, row in enumerate(rows):
            value = _finite(row[key], label=key)
            x = left + gap + index * (bar_width + gap)
            bar_height = chart_height * (value - lower) / (upper - lower)
            y = top + chart_height - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="3" fill="{colors[index % len(colors)]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263746">{value:.3f}</text>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{html.escape(_candidate_label(row["candidate_id"]))}</text>'
            )
    lines.extend(
        [
            '<text x="56" y="486" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Window: 2023-07-01 through 2023-07-06 UTC. It was not used for model selection, but is now consumed.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _funnel_svg(*, neural_screened: int, final_models: int, predictive: int) -> str:
    stages = (
        ("Neural architectures screened", neural_screened),
        ("Final models with baseline", final_models),
        ("Passed forecast gates", predictive),
        ("Passed top-500 cost gate", 0),
        ("Trading candidates", 0),
    )
    width, height = 1500, 420
    box_width, gap, start_x = 240, 46, 56
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">The execution-cost gate stopped promotion</title>',
        '<desc id="desc">Three neural architectures were screened. Two advanced and were compared with LightGBM. All final models passed forecast-quality gates and none passed the after-cost gate.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">The execution-cost gate stopped promotion</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">A positive forecast was never allowed to become trading authority without positive exact after-cost evidence.</text>',
    ]
    for index, (label, value) in enumerate(stages):
        x = start_x + index * (box_width + gap)
        fill = "#f4f7f9" if value else "#fff1f0"
        stroke = "#60717f" if value else "#b42318"
        if index:
            lines.append(
                f'<line x1="{x - gap + 8}" y1="220" x2="{x - 9}" y2="220" stroke="#7b8994" stroke-width="2"/>'
            )
            lines.append(
                f'<path d="M {x - 9} 214 L {x} 220 L {x - 9} 226 Z" fill="#7b8994"/>'
            )
        lines.append(
            f'<rect x="{x}" y="146" width="{box_width}" height="148" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="213" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="38" font-weight="700" fill="{stroke}">{value}</text>'
        )
        words = label.split(" ")
        split = max(1, len(words) // 2)
        first = html.escape(" ".join(words[:split]))
        second = html.escape(" ".join(words[split:]))
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="254" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#263746"><tspan x="{x + box_width / 2:.1f}" dy="0">{first}</tspan><tspan x="{x + box_width / 2:.1f}" dy="19">{second}</tspan></text>'
        )
    lines.extend(
        [
            '<text x="56" y="366" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">No trade, leverage, portfolio return, or profitability claim was produced.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _research_progress_svg(progress: Sequence[Mapping[str, object]]) -> str:
    rows = [row for row in progress if int(row["round"]) >= 7]
    width, height = 1500, 620
    left, right, top, chart_height = 120, 70, 158, 330
    chart_width = width - left - right
    lower, upper = -14.0, 2.0

    def y_position(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">After-cost evidence by research round</title>',
        '<desc id="desc">Rounds seven and eight show means from executable-trade simulations. Rounds nine through twelve produced no executable series. Round thirteen shows a top-500 overlapping-forecast diagnostic and is not a trade result.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">After-cost evidence by research round</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Distinct markers prevent an overlapping-forecast diagnostic from being presented as an executed trade mean.</text>',
        '<circle cx="920" cy="55" r="7" fill="#b42318"/><text x="937" y="60" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">executed mean</text>',
        '<rect x="1077" y="48" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 1084 55)"/><text x="1101" y="60" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">no executable series</text>',
        '<rect x="1290" y="48" width="14" height="14" fill="#7b559c"/><text x="1314" y="60" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">overlap diagnostic</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in (-14.0, -10.0, -5.0, 0.0):
        y = y_position(value)
        color = "#526674" if value == 0.0 else "#e5ebf0"
        stroke_width = 2 if value == 0.0 else 1
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="{color}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value:.0f}</text>'
        )

    executed_points: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        x = left + chart_width * (index + 0.5) / max(1, len(rows))
        executed = int(row.get("executable_trades") or 0) > 0
        net_text = str(row.get("mean_net_bps") or "").strip()
        diagnostic_text = str(
            row.get("best_top_500_exact_after_cost_bps") or ""
        ).strip()
        if executed and net_text:
            value = _finite(net_text, label="progress executed mean")
            y = y_position(value)
            executed_points.append((x, y))
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#b42318" stroke="#ffffff" stroke-width="3"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#263744">{value:+.2f} bps</text>'
            )
        elif diagnostic_text:
            value = _finite(diagnostic_text, label="progress overlap diagnostic")
            y = y_position(value)
            lines.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" stroke="#bca8cc" stroke-width="2" stroke-dasharray="5 6"/>'
            )
            lines.append(
                f'<rect x="{x - 8:.1f}" y="{y - 8:.1f}" width="16" height="16" fill="#7b559c" stroke="#ffffff" stroke-width="2"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{y - 17:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#5f3e78">{value:+.2f} bps diagnostic</text>'
            )
        else:
            y = y_position(-6.0)
            lines.append(
                f'<line x1="{x:.1f}" y1="{top + 32}" x2="{x:.1f}" y2="{top + chart_height - 32}" stroke="#a7b2bb" stroke-width="2" stroke-dasharray="5 6"/>'
            )
            lines.append(
                f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" fill="#ffffff" stroke="#60717f" stroke-width="2" transform="rotate(45 {x:.1f} {y:.1f})"/>'
            )
        lines.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">Round {row["round"]}</text>'
        )
    if len(executed_points) > 1:
        lines.append(
            '<polyline points="'
            + " ".join(f"{x:.1f},{y:.1f}" for x, y in executed_points)
            + '" fill="none" stroke="#287f9e" stroke-width="4"/>'
        )
        for x, y in executed_points:
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#b42318" stroke="#ffffff" stroke-width="3"/>'
            )
    lines.extend(
        [
            '<text x="42" y="323" transform="rotate(-90 42 323)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">After-cost basis points</text>',
            '<text x="56" y="574" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Windows and units differ by round. Values are source-bound evidence, not a continuous equity curve or directly comparable portfolio returns.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _progress_rows(
    path: Path,
    design: Mapping[str, object],
    best: Mapping[str, object],
) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if rows and int(rows[-1]["round"]) == int(design["round"]):
        rows.pop()
    if not rows or int(rows[-1]["round"]) != int(design["round"]) - 1:
        raise ValueError("prior progress table does not end at the preceding round")
    data = design["data"]
    assert isinstance(data, Mapping)
    rows.append(
        {
            "round": int(design["round"]),
            "stage": "gross architecture and loss screen",
            "periods": f"{data['start_date']}..{data['end_date']}",
            "selection_contaminated": True,
            "horizon_seconds": int(design["execution"]["horizon_seconds"]),
            "feature_set": "l1-tape-causal-v7",
            "risk_level": "research-only",
            "direction_auc": best["development_direction_auc"],
            "spearman_ic": best["development_spearman_ic"],
            "selected_signals": 0,
            "executable_trades": 0,
            "mean_gross_bps": "",
            "mean_net_bps": "",
            "status": "rejected",
            "source_file": "gross architecture Round 13 v2 report",
            "best_model_id": best["candidate_id"],
            "best_top_500_gross_bps": best["development_top_500_gross_bps"],
            "best_top_500_exact_after_cost_bps": best[
                "development_top_500_exact_after_cost_bps"
            ],
            "after_cost_diagnostic_rows": 500,
        }
    )
    return rows


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
    design, design_sha256 = load_gross_architecture_design(
        design_path,
        require_current=False,
    )
    report, stage_rows, final_rows = _validated_evidence(evidence_root, design)
    ranked = sorted(
        final_rows,
        key=lambda row: (
            float(row["development_top_500_exact_after_cost_bps"]),
            float(row["development_spearman_ic"]),
            float(row["development_direction_auc"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    all_candidates = [
        row for row in stage_rows if not bool(row["selected_for_stage_two"])
    ] + final_rows
    progress = _progress_rows(prior_progress_path, design, best)
    progress_fields = tuple(progress[0]) + tuple(
        key for key in progress[-1] if key not in progress[0]
    )
    charts = output_dir / "charts"

    diagnostics: dict[str, object] = {
        "schema_version": "gross-architecture-publication-diagnostics-v1",
        "artifact_class": "consumed_development_diagnostic_no_trading_authority",
        "design_sha256": design_sha256,
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "corpus_certificate_sha256": report["corpus_certificate_sha256"],
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": True,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "runtime_resources": report["runtime_resources"],
        "dataset": report["dataset"],
        "successive_halving": report["successive_halving"],
        "final_results": report["final_results"],
        "publication_rows": all_candidates,
        "limitations": report["limitations"],
    }
    diagnostics["diagnostic_sha256"] = _canonical_sha256(diagnostics)

    _write_csv(output_dir / "candidates.csv", all_candidates, _CANDIDATE_FIELDS)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "diagnostics.json",
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(charts / "after-cost-performance.svg", _after_cost_svg(ranked))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(ranked))
    predictive = sum(
        not [
            reason
            for reason in str(row["rejection_reasons"]).split(";")
            if reason and reason != "top_500_exact_after_cost_gate_failed"
        ]
        for row in final_rows
    )
    _write_text(
        charts / "signal-selection.svg",
        _funnel_svg(
            neural_screened=len(stage_rows),
            final_models=len(final_rows),
            predictive=predictive,
        ),
    )
    _write_text(
        charts / "research-progress.svg",
        _research_progress_svg(progress),
    )

    data = design["data"]
    terminal = design["reserved_terminal"]
    implementation = design["implementation"]
    assert isinstance(data, Mapping)
    assert isinstance(terminal, Mapping)
    assert isinstance(implementation, Mapping)
    readme = f"""# Round 13: gross edge, no taker viability

**Rejected.** The best model produced **{float(best["development_top_500_gross_bps"]):+.2f} bps gross**, but only **{float(best["development_top_500_exact_after_cost_bps"]):+.2f} bps after measured taker costs** across its 500 strongest development forecasts.

| Model | AUC | Spearman IC | Top 500 gross | Top 500 exact net |
| --- | ---: | ---: | ---: | ---: |
"""
    for row in ranked:
        readme += (
            f"| {_candidate_label(row['candidate_id'])} "
            f"| {float(row['development_direction_auc']):.3f} "
            f"| {float(row['development_spearman_ic']):.3f} "
            f"| {float(row['development_top_500_gross_bps']):+.2f} bps "
            f"| {float(row['development_top_500_exact_after_cost_bps']):+.2f} bps |\n"
        )
    readme += f"""

![After-cost performance](charts/after-cost-performance.svg)

![Forecast quality](charts/forecast-quality.svg)

![Model selection](charts/signal-selection.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, {data["start_date"]} through {data["end_date"]} UTC; {int(report["dataset"]["event_rows"]):,} causal events from {int(report["dataset"]["rows"]):,} exact-BBO rows. The development window is consumed and the {terminal["date"]} terminal day remains untouched. Top-row forecasts overlap, so they are not trades, an equity curve, ROI, or trading authority.

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
    publication: dict[str, object] = {
        "schema_version": "gross-architecture-screen-publication-v1",
        "artifact_class": "exchange_sourced_gross_model_graph_data",
        "round": int(design["round"]),
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
        "runtime_resources": report["runtime_resources"],
        "actual": {
            "dataset_rows": report["dataset"]["rows"],
            "event_rows": report["dataset"]["event_rows"],
            "stage_one_neural_candidates": len(stage_rows),
            "stage_two_neural_candidates": len(
                report["successive_halving"]["selected_candidate_ids"]
            ),
            "final_model_count": len(final_rows),
            "unrejected_candidate_count": sum(
                row["status"] == "research_candidate" for row in final_rows
            ),
            "best_model_id": best["candidate_id"],
            "best_development_direction_auc": best["development_direction_auc"],
            "best_development_spearman_ic": best["development_spearman_ic"],
            "best_top_500_gross_bps": best["development_top_500_gross_bps"],
            "best_top_500_exact_after_cost_bps": best[
                "development_top_500_exact_after_cost_bps"
            ],
            "executed_trades": 0,
        },
        "source_artifacts": [
            {
                "path": str(row["artifact_path"]),
                "sha256": str(row["artifact_sha256"]),
                "bytes": (evidence_root / str(row["artifact_path"])).stat().st_size,
            }
            for row in final_rows
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
        "gross-architecture-publication: "
        f"status={publication['status']} "
        f"models={publication['actual']['final_model_count']} "
        f"sha256={publication['publication_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
