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
    from tools.publish_gross_architecture_screen import (
        _canonical_payload_hash,
        _research_progress_svg,
    )
    from tools.run_head_coherence_screen import load_head_coherence_design
except ModuleNotFoundError:
    from publish_action_value_discovery import (
        _canonical_sha256,
        _sha256,
        _write_csv,
        _write_text,
    )
    from publish_gross_architecture_screen import (
        _canonical_payload_hash,
        _research_progress_svg,
    )
    from run_head_coherence_screen import load_head_coherence_design


_MODEL_LABELS = {
    "mlp-huber-direction": "MLP Huber/direction",
    "mlp-gmadl-coherence-025": "MLP GMADL + coherence 0.25",
    "lightgbm-gross-baseline": "LightGBM baseline",
}
_METHOD_LABELS = {
    "mean": "mean",
    "direction_confidence": "direction",
    "direction_magnitude": "direction x magnitude",
    "head_consensus": "head consensus",
    "conservative_quantile": "conservative quantile",
}
_METHOD_ORDER = tuple(_METHOD_LABELS)
_CANDIDATE_FIELDS = (
    "candidate_id",
    "model_id",
    "model_family",
    "score_method",
    "status",
    "backend_kind",
    "backend_device",
    "best_epoch",
    "training_data_mode",
    "training_preload_bytes",
    "development_direction_auc",
    "development_spearman_ic",
    "development_mae_bps",
    "development_zero_mae_bps",
    "policy_active_rows",
    "policy_head_agreement_ratio",
    "policy_top_100_gross_bps",
    "policy_top_100_exact_after_cost_bps",
    "policy_top_500_gross_bps",
    "policy_top_500_exact_after_cost_bps",
    "policy_top_500_exact_positive_rate",
    "development_active_rows",
    "development_head_agreement_ratio",
    "development_top_100_gross_bps",
    "development_top_100_exact_after_cost_bps",
    "development_top_500_gross_bps",
    "development_top_500_exact_after_cost_bps",
    "development_top_500_exact_positive_rate",
    "model_sha256",
    "artifact_path",
    "artifact_sha256",
    "policy_evaluation_error",
    "development_evaluation_error",
    "rejection_reasons",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish hash-bound Round 14 head-coherence evidence",
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON evidence: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path.name}")
    return value


def _finite(value: object, *, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite head-coherence {label}")
    return parsed


def _top_row(metrics: Mapping[str, object], requested: int) -> Mapping[str, object]:
    rows = metrics.get("top_rows")
    if not isinstance(rows, list):
        raise ValueError("head-coherence top-row evidence is missing")
    matched = [
        row
        for row in rows
        if isinstance(row, Mapping) and int(row.get("requested_rows") or 0) == requested
    ]
    if len(matched) != 1:
        raise ValueError("head-coherence top-row evidence is ambiguous")
    row = matched[0]
    if (
        row.get("portfolio_claim") is not False
        or row.get("overlapping_forecasts") is not True
        or int(row.get("rows") or 0) > requested
        or int(row.get("rows") or 0) <= 0
    ):
        raise ValueError("head-coherence top-row contract is invalid")
    for key in (
        "mean_signed_gross_bps",
        "mean_exact_after_cost_bps",
        "exact_after_cost_positive_rate",
    ):
        _finite(row.get(key), label=key)
    return row


def _validate_action_metrics(
    metrics: object,
    *,
    label: str,
) -> Mapping[str, object] | None:
    if metrics is None:
        return None
    if not isinstance(metrics, Mapping):
        raise ValueError(f"head-coherence {label} action metrics are invalid")
    rows = int(metrics.get("rows") or 0)
    active = int(metrics.get("active_rows") or 0)
    eligible = int(metrics.get("exact_after_cost_eligible_rows") or 0)
    if (
        rows <= 0
        or active <= 0
        or eligible <= 0
        or eligible > active
        or active > rows
        or metrics.get("trading_authority") is not False
        or metrics.get("execution_claim") is not False
        or metrics.get("profitability_claim") is not False
        or metrics.get("portfolio_claim") is not False
    ):
        raise ValueError(f"head-coherence {label} action contract is invalid")
    _finite(metrics.get("mean_direction_head_agreement_ratio"), label=label)
    for requested in (100, 500, 1_000):
        _top_row(metrics, requested)
    return metrics


def _validate_forecast(metrics: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(metrics, Mapping) or int(metrics.get("rows") or 0) <= 0:
        raise ValueError(f"head-coherence {label} forecast metrics are invalid")
    for key in (
        "direction_auc",
        "spearman_information_coefficient",
        "mean_absolute_error_bps",
        "zero_baseline_mae_bps",
    ):
        _finite(metrics.get(key), label=f"{label} {key}")
    return metrics


def _validate_artifact(
    evidence_root: Path,
    raw: object,
    *,
    model_id: str,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"head-coherence artifact is missing: {model_id}")
    relative = Path(str(raw.get("path") or ""))
    if relative.is_absolute() or len(relative.parts) != 1 or not relative.name:
        raise ValueError(f"head-coherence artifact path is unsafe: {model_id}")
    path = evidence_root / relative
    if (
        not path.is_file()
        or path.stat().st_size != int(raw.get("bytes") or -1)
        or _sha256(path) != str(raw.get("sha256") or "")
    ):
        raise ValueError(f"head-coherence artifact integrity failed: {model_id}")
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _metric_columns(prefix: str, metrics: Mapping[str, object]) -> dict[str, object]:
    top_100 = _top_row(metrics, 100)
    top_500 = _top_row(metrics, 500)
    return {
        f"{prefix}_active_rows": metrics["active_rows"],
        f"{prefix}_head_agreement_ratio": metrics[
            "mean_direction_head_agreement_ratio"
        ],
        f"{prefix}_top_100_gross_bps": top_100["mean_signed_gross_bps"],
        f"{prefix}_top_100_exact_after_cost_bps": top_100["mean_exact_after_cost_bps"],
        f"{prefix}_top_500_gross_bps": top_500["mean_signed_gross_bps"],
        f"{prefix}_top_500_exact_after_cost_bps": top_500["mean_exact_after_cost_bps"],
        f"{prefix}_top_500_exact_positive_rate": top_500[
            "exact_after_cost_positive_rate"
        ],
    }


def _validated_evidence(
    evidence_root: Path,
    design: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    report_path = evidence_root / "report.json"
    report = _read_json(report_path)
    _canonical_payload_hash(report, "report_sha256")
    if (
        report.get("schema_version") != "head-coherence-screen-report-v1"
        or report.get("status") not in {"rejected", "research_candidate"}
        or report.get("round") != 14
        or report.get("design_sha256") != design["design_sha256"]
        or report.get("terminal_holdout_accessed") is not False
        or report.get("development_window_is_consumed") is not True
        or report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
        or report.get("portfolio_claim") is not False
        or report.get("leverage_applied") is not False
    ):
        raise ValueError("head-coherence report contract is invalid")
    successive = report.get("successive_halving")
    final = report.get("final_results")
    if not isinstance(successive, Mapping) or not isinstance(final, list):
        raise ValueError("head-coherence report sections are incomplete")
    stage_one = successive.get("stage_one_results")
    selected = successive.get("selected_candidate_ids")
    if (
        not isinstance(stage_one, list)
        or len(stage_one) != 4
        or not isinstance(selected, list)
        or len(selected) != 2
    ):
        raise ValueError("head-coherence successive-halving evidence is incomplete")
    final_ids = {
        str(value.get("candidate_id") or "")
        for value in final
        if isinstance(value, Mapping)
    }
    if final_ids != set(str(value) for value in selected) | {"lightgbm-gross-baseline"}:
        raise ValueError("head-coherence final model set is inconsistent")

    rows: list[dict[str, object]] = []
    any_candidate = False
    for model in final:
        if not isinstance(model, Mapping):
            raise ValueError("head-coherence final model is invalid")
        model_id = str(model["candidate_id"])
        artifact = model.get("artifact")
        actions = model.get("action_results")
        _validate_forecast(
            model.get("policy_forecast_metrics"),
            label=f"{model_id} policy",
        )
        development_forecast = _validate_forecast(
            model.get("development_forecast_metrics"),
            label=f"{model_id} development",
        )
        artifact_file = _validate_artifact(
            evidence_root,
            model.get("artifact_file"),
            model_id=model_id,
        )
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(actions, list)
            or len(actions) != len(design["action_score_methods"])
            or model.get("trading_authority") is not False
            or model.get("execution_claim") is not False
            or model.get("profitability_claim") is not False
            or model.get("portfolio_claim") is not False
            or artifact.get("trading_authority") is not False
            or artifact.get("execution_claim") is not False
            or artifact.get("profitability_claim") is not False
            or len(str(artifact.get("model_sha256") or "")) != 64
        ):
            raise ValueError(f"head-coherence final model contract failed: {model_id}")
        methods: set[str] = set()
        model_has_candidate = False
        for action in actions:
            if not isinstance(action, Mapping):
                raise ValueError(f"head-coherence action result is invalid: {model_id}")
            method = str(action.get("score_method") or "")
            methods.add(method)
            policy = _validate_action_metrics(
                action.get("policy_metrics"),
                label=f"{model_id} {method} policy",
            )
            development = _validate_action_metrics(
                action.get("development_metrics"),
                label=f"{model_id} {method} development",
            )
            policy_error = action.get("policy_evaluation_error")
            development_error = action.get("development_evaluation_error")
            reasons = action.get("rejection_reasons")
            if (
                not isinstance(reasons, list)
                or action.get("status") not in {"rejected", "research_candidate"}
                or action.get("trading_authority") is not False
                or action.get("execution_claim") is not False
                or action.get("profitability_claim") is not False
                or action.get("portfolio_claim") is not False
                or (action["status"] == "rejected") != bool(reasons)
                or (policy is None) != bool(policy_error)
                or (development is None) != bool(development_error)
            ):
                raise ValueError(
                    f"head-coherence action contract failed: {model_id}/{method}"
                )
            row: dict[str, object] = {
                "candidate_id": f"{model_id}/{method}",
                "model_id": model_id,
                "model_family": str(artifact.get("model_family") or ""),
                "score_method": method,
                "status": str(action["status"]),
                "backend_kind": str(artifact.get("backend_kind") or ""),
                "backend_device": str(artifact.get("backend_device") or ""),
                "best_epoch": (
                    ""
                    if artifact.get("best_epoch") is None
                    else int(artifact["best_epoch"])
                ),
                "training_data_mode": str(
                    artifact.get("training_data_mode") or "not_applicable"
                ),
                "training_preload_bytes": int(
                    artifact.get("training_preload_bytes") or 0
                ),
                "development_direction_auc": development_forecast["direction_auc"],
                "development_spearman_ic": development_forecast[
                    "spearman_information_coefficient"
                ],
                "development_mae_bps": development_forecast["mean_absolute_error_bps"],
                "development_zero_mae_bps": development_forecast[
                    "zero_baseline_mae_bps"
                ],
                "model_sha256": str(artifact["model_sha256"]),
                "artifact_path": artifact_file["path"],
                "artifact_sha256": artifact_file["sha256"],
                "policy_evaluation_error": str(policy_error or ""),
                "development_evaluation_error": str(development_error or ""),
                "rejection_reasons": ";".join(str(value) for value in reasons),
            }
            if policy is not None:
                row.update(_metric_columns("policy", policy))
            if development is not None:
                row.update(_metric_columns("development", development))
            rows.append(row)
            if action["status"] == "research_candidate":
                any_candidate = True
                model_has_candidate = True
        if methods != set(design["action_score_methods"]):
            raise ValueError(f"head-coherence action methods drifted: {model_id}")
        if (model.get("status") == "research_candidate") != model_has_candidate:
            raise ValueError(f"head-coherence model status drifted: {model_id}")
    expected_status = "research_candidate" if any_candidate else "rejected"
    if report["status"] != expected_status:
        raise ValueError("head-coherence report status drifted")
    rows.sort(
        key=lambda row: (
            tuple(_MODEL_LABELS).index(str(row["model_id"])),
            _METHOD_ORDER.index(str(row["score_method"])),
        )
    )
    return report, rows


def _chart_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if int(row.get("policy_active_rows") or 0) >= 500
        and int(row.get("development_active_rows") or 0) >= 500
    ]


def _economics_svg(rows: Sequence[Mapping[str, object]]) -> str:
    plotted = _chart_rows(rows)
    values = [
        _finite(row[key], label=key)
        for row in plotted
        for key in (
            "policy_top_500_exact_after_cost_bps",
            "development_top_500_exact_after_cost_bps",
        )
    ]
    bound = max(24.0, math.ceil(max(abs(value) for value in values) / 2.0) * 2.0)
    lower, upper = -bound, 4.0
    width, left, right, top, row_height = 1500, 430, 80, 150, 50
    chart_width = width - left - right
    height = top + len(plotted) * row_height + 100
    zero_x = left + chart_width * (0.0 - lower) / (upper - lower)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">No high-activity mapping cleared taker costs</title>',
        '<desc id="desc">Policy and development top-500 exact after-cost means for every action mapping with active rows. All values are negative.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">No high-activity mapping cleared taker costs</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Top 500 overlapping forecasts per role; these are diagnostics, not executed trades or portfolio ROI.</text>',
    ]
    for index in range(8):
        value = lower + (upper - lower) * index / 7.0
        x = left + chart_width * index / 7.0
        lines.append(
            f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 72}" stroke="#dce3e8" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#65727d">{value:+.0f}</text>'
        )
    prior_model = None
    for index, row in enumerate(plotted):
        y = top + index * row_height
        model_id = str(row["model_id"])
        if prior_model is not None and model_id != prior_model:
            lines.append(
                f'<line x1="56" y1="{y - 5}" x2="{width - 56}" y2="{y - 5}" stroke="#9aa8b3" stroke-width="2"/>'
            )
        prior_model = model_id
        label = (
            f"{_MODEL_LABELS[model_id]} / {_METHOD_LABELS[str(row['score_method'])]}"
        )
        lines.append(
            f'<text x="{left - 18}" y="{y + 27}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#263746">{html.escape(label)}</text>'
        )
        for offset, key, color, prefix in (
            (8, "policy_top_500_exact_after_cost_bps", "#b42318", "policy"),
            (28, "development_top_500_exact_after_cost_bps", "#7b559c", "dev"),
        ):
            value = _finite(row[key], label=key)
            x = left + chart_width * (value - lower) / (upper - lower)
            start = min(x, zero_x)
            lines.append(
                f'<rect x="{start:.1f}" y="{y + offset}" width="{max(2.0, abs(x - zero_x)):.1f}" height="16" rx="2" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{x - 7:.1f}" y="{y + offset + 13}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" fill="{color}">{prefix} {value:+.2f}</text>'
            )
    lines.extend(
        [
            f'<line x1="{zero_x:.1f}" y1="{top - 18}" x2="{zero_x:.1f}" y2="{height - 72}" stroke="#17212b" stroke-width="2"/>',
            f'<text x="{left + chart_width / 2:.1f}" y="{height - 30}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Exact after-cost basis points; 750 ms latency and 12 bps configured round trip.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    models: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        model_id = str(row["model_id"])
        if model_id not in seen:
            models.append(row)
            seen.add(model_id)
    panels = (
        ("Direction AUC", "development_direction_auc", 0.5, 0.58),
        ("Spearman IC", "development_spearman_ic", 0.0, 0.14),
        ("Head agreement", "development_head_agreement_ratio", 0.70, 0.94),
    )
    width, height = 1500, 520
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Coherence loss did not improve final head agreement</title>',
        '<desc id="desc">Development AUC, Spearman information coefficient and mean-direction head agreement for the two neural finalists and LightGBM baseline.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Coherence loss did not improve final head agreement</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">The 0.25 coherence model also missed the zero-MAE baseline; complexity was not promoted on intent.</text>',
    ]
    colors = ("#287f9e", "#4c7b52", "#6e7781")
    panel_width, gap = 410, 65
    for panel_index, (title, key, lower, upper) in enumerate(panels):
        left = 80 + panel_index * (panel_width + gap)
        top, chart_height = 148, 240
        lines.append(
            f'<text x="{left}" y="125" font-family="Segoe UI, Arial, sans-serif" font-size="17" font-weight="700" fill="#263746">{title}</text>'
        )
        for tick in range(4):
            value = lower + (upper - lower) * tick / 3.0
            y = top + chart_height * (upper - value) / (upper - lower)
            lines.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + panel_width}" y2="{y:.1f}" stroke="#dce3e8" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#65727d">{value:.3f}</text>'
            )
        bar_width = 82
        bar_gap = (panel_width - len(models) * bar_width) / (len(models) + 1)
        for index, row in enumerate(models):
            value = _finite(row.get(key), label=key)
            x = left + bar_gap + index * (bar_width + bar_gap)
            bar_height = chart_height * (value - lower) / (upper - lower)
            y = top + chart_height - bar_height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="3" fill="{colors[index]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#263746">{value:.3f}</text>'
            )
            short_label = ("Huber", "Coherence", "LightGBM")[index]
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#52606d">{short_label}</text>'
            )
    lines.extend(
        [
            '<text x="56" y="482" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Consumed development window: 2023-07-01 through 2023-07-06 UTC. It was not used for selection.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _funnel_svg(*, action_rows: Sequence[Mapping[str, object]]) -> str:
    active = sum(int(row.get("policy_active_rows") or 0) >= 500 for row in action_rows)
    positive_policy_gross = sum(
        int(row.get("policy_active_rows") or 0) >= 500
        and float(row.get("policy_top_500_gross_bps") or -1.0) > 0.0
        for row in action_rows
    )
    stages = (
        ("Neural architectures screened", 4),
        ("Final models with baseline", 3),
        ("Action mappings evaluated", len(action_rows)),
        ("Activity-qualified mappings", active),
        ("Positive policy gross", positive_policy_gross),
        ("Research candidates", 0),
    )
    width, height = 1500, 420
    box_width, gap, start_x = 205, 38, 56
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">The policy regime stopped every mapping</title>',
        f'<desc id="desc">Four neural architectures were screened, two advanced and were compared with LightGBM, {len(action_rows)} mappings were evaluated, {active} had sufficient activity, {positive_policy_gross} had positive policy gross returns, and none became a research candidate.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">The policy regime stopped every mapping</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">No later positive development tail was allowed to erase the earlier policy failure.</text>',
    ]
    for index, (label, value) in enumerate(stages):
        x = start_x + index * (box_width + gap)
        fill = "#f4f7f9" if value else "#fff1f0"
        stroke = "#60717f" if value else "#b42318"
        if index:
            lines.append(
                f'<line x1="{x - gap + 7}" y1="218" x2="{x - 9}" y2="218" stroke="#7b8994" stroke-width="2"/><path d="M {x - 9} 212 L {x} 218 L {x - 9} 224 Z" fill="#7b8994"/>'
            )
        words = label.split()
        split = max(1, len(words) // 2)
        lines.append(
            f'<rect x="{x}" y="145" width="{box_width}" height="146" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="210" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="36" font-weight="700" fill="{stroke}">{value}</text>'
        )
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="251" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#263746"><tspan x="{x + box_width / 2:.1f}">{html.escape(" ".join(words[:split]))}</tspan><tspan x="{x + box_width / 2:.1f}" dy="18">{html.escape(" ".join(words[split:]))}</tspan></text>'
        )
    lines.extend(
        [
            '<text x="56" y="365" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Zero candidates means no trade, leverage, execution, portfolio, or profitability authority.</text>',
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
    if rows and int(rows[-1]["round"]) == 14:
        rows.pop()
    if not rows or int(rows[-1]["round"]) != 13:
        raise ValueError("prior progress table does not end at Round 13")
    data = design["data"]
    assert isinstance(data, Mapping)
    rows.append(
        {
            "round": 14,
            "stage": "head coherence and action mapping screen",
            "periods": f"{data['start_date']}..{data['end_date']}",
            "selection_contaminated": True,
            "horizon_seconds": 300,
            "feature_set": "l1-tape-causal-v7",
            "risk_level": "research-only",
            "direction_auc": best["development_direction_auc"],
            "spearman_ic": best["development_spearman_ic"],
            "selected_signals": 0,
            "executable_trades": 0,
            "mean_gross_bps": "",
            "mean_net_bps": "",
            "status": "rejected",
            "source_file": "head coherence Round 14 report",
            "best_model_id": best["candidate_id"],
            "best_top_500_gross_bps": best["development_top_500_gross_bps"],
            "best_top_500_exact_after_cost_bps": best[
                "development_top_500_exact_after_cost_bps"
            ],
            "best_policy_top_500_gross_bps": best["policy_top_500_gross_bps"],
            "best_policy_top_500_exact_after_cost_bps": best[
                "policy_top_500_exact_after_cost_bps"
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
    design, design_sha256 = load_head_coherence_design(
        design_path,
        require_current=False,
    )
    report, candidates = _validated_evidence(evidence_root, design)
    activity_qualified = [
        row
        for row in candidates
        if int(row.get("policy_active_rows") or 0) >= 500
        and int(row.get("development_active_rows") or 0) >= 500
    ]
    if not activity_qualified:
        raise ValueError("head-coherence publication has no activity-qualified row")
    best = max(
        activity_qualified,
        key=lambda row: (
            float(row["policy_top_500_exact_after_cost_bps"]),
            float(row["policy_top_500_gross_bps"]),
            -_METHOD_ORDER.index(str(row["score_method"])),
        ),
    )
    progress = _progress_rows(prior_progress_path, design, best)
    progress_fields = tuple(progress[0]) + tuple(
        key for key in progress[-1] if key not in progress[0]
    )
    charts = output_dir / "charts"
    diagnostics: dict[str, object] = {
        "schema_version": "head-coherence-publication-diagnostics-v1",
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
        "publication_rows": candidates,
        "limitations": report["limitations"],
    }
    diagnostics["diagnostic_sha256"] = _canonical_sha256(diagnostics)
    _write_csv(output_dir / "candidates.csv", candidates, _CANDIDATE_FIELDS)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "diagnostics.json",
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(charts / "after-cost-performance.svg", _economics_svg(candidates))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(candidates))
    _write_text(
        charts / "action-funnel.svg",
        _funnel_svg(action_rows=candidates),
    )
    _write_text(
        charts / "research-progress.svg",
        _research_progress_svg(progress),
    )

    huber_direction = next(
        row
        for row in candidates
        if row["model_id"] == "mlp-huber-direction"
        and row["score_method"] == "direction_magnitude"
    )
    readme = f"""# Round 14: action mapping still failed policy

**Rejected.** Direction x magnitude improved the Huber model over mean-head selection, but its top-500 policy result was **{float(huber_direction["policy_top_500_gross_bps"]):+.2f} bps gross** and **{float(huber_direction["policy_top_500_exact_after_cost_bps"]):+.2f} bps exact net**. Its later development result improved to **{float(huber_direction["development_top_500_gross_bps"]):+.2f} bps gross** but remained **{float(huber_direction["development_top_500_exact_after_cost_bps"]):+.2f} bps exact net**.

| Model | Best activity-qualified policy mapping | Policy gross | Policy exact net | Development exact net |
| --- | --- | ---: | ---: | ---: |
"""
    model_best: list[Mapping[str, object]] = []
    for model_id in _MODEL_LABELS:
        model_rows = [row for row in activity_qualified if row["model_id"] == model_id]
        row = max(
            model_rows,
            key=lambda value: float(value["policy_top_500_exact_after_cost_bps"]),
        )
        model_best.append(row)
        readme += (
            f"| {_MODEL_LABELS[model_id]} | {_METHOD_LABELS[str(row['score_method'])]} "
            f"| {float(row['policy_top_500_gross_bps']):+.2f} bps "
            f"| {float(row['policy_top_500_exact_after_cost_bps']):+.2f} bps "
            f"| {float(row['development_top_500_exact_after_cost_bps']):+.2f} bps |\n"
        )
    readme += f"""

![After-cost mapping results](charts/after-cost-performance.svg)

![Forecast and coherence quality](charts/forecast-quality.svg)

![Action funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, {design["data"]["start_date"]} through {design["data"]["end_date"]} UTC; {int(report["dataset"]["event_rows"]):,} causal events from {int(report["dataset"]["rows"]):,} exact-BBO rows. The development role is consumed and 2023-07-07 remains untouched. Rows overlap, so these are not trades, an equity curve, ROI, or trading authority.

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
        charts / "action-funnel.svg",
        charts / "research-progress.svg",
    ]
    implementation = design["implementation"]
    assert isinstance(implementation, Mapping)
    publication: dict[str, object] = {
        "schema_version": "head-coherence-screen-publication-v1",
        "artifact_class": "exchange_sourced_head_coherence_graph_data",
        "round": 14,
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
            "neural_architectures_screened": 4,
            "neural_architectures_advanced": 2,
            "final_model_count": 3,
            "action_candidate_count": len(candidates),
            "activity_qualified_candidate_count": len(activity_qualified),
            "positive_policy_gross_candidate_count": sum(
                float(row["policy_top_500_gross_bps"]) > 0.0
                for row in activity_qualified
            ),
            "unrejected_candidate_count": sum(
                row["status"] == "research_candidate" for row in candidates
            ),
            "best_activity_qualified_candidate_id": best["candidate_id"],
            "best_policy_top_500_gross_bps": best["policy_top_500_gross_bps"],
            "best_policy_top_500_exact_after_cost_bps": best[
                "policy_top_500_exact_after_cost_bps"
            ],
            "best_development_top_500_gross_bps": best["development_top_500_gross_bps"],
            "best_development_top_500_exact_after_cost_bps": best[
                "development_top_500_exact_after_cost_bps"
            ],
            "executed_trades": 0,
        },
        "source_artifacts": [
            {
                "path": row["artifact_path"],
                "sha256": row["artifact_sha256"],
                "bytes": (evidence_root / str(row["artifact_path"])).stat().st_size,
            }
            for row in model_best
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
        "head-coherence-publication: "
        f"status={publication['status']} "
        f"candidates={publication['actual']['action_candidate_count']} "
        f"sha256={publication['publication_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
