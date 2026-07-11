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
        _progress_svg,
        _sha256,
        _write_csv,
        _write_text,
    )
    from tools.run_selective_event_discovery import load_selective_event_design
except ModuleNotFoundError:
    from publish_action_value_discovery import (
        _canonical_sha256,
        _progress_svg,
        _sha256,
        _write_csv,
        _write_text,
    )
    from run_selective_event_discovery import load_selective_event_design


_RISK_ORDER = {"conservative": 0, "regular": 1, "aggressive": 2}
_METHOD_LABELS = {
    "event_direct_mean": "mean",
    "event_upper_quantile": "upper q",
    "event_distributional_value": "distribution",
}
_CANDIDATE_FIELDS = (
    "candidate_id",
    "horizon_seconds",
    "risk_level",
    "score_method",
    "status",
    "policy_accepted",
    "minimum_policy_trades",
    "policy_trades",
    "policy_total_net_bps",
    "policy_mean_net_bps",
    "policy_max_drawdown_bps",
    "policy_profit_factor",
    "policy_threshold",
    "selection_trades",
    "selection_total_net_bps",
    "selection_max_drawdown_bps",
    "selection_profit_factor",
    "selection_daily_ci_lower_bps",
    "selection_top_100_mean_actual_net_bps",
    "selection_top_100_profitable_ratio",
    "selection_event_rows",
    "model_backend",
    "model_device",
    "parameter_profile",
    "model_sha256",
    "artifact_sha256",
    "rejection_reasons",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish hash-bound selective-event discovery evidence",
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
        raise ValueError(f"non-finite selective-event {label}")
    return parsed


def _canonical_payload_hash(payload: Mapping[str, object], field: str) -> str:
    claimed = str(payload.get(field) or "")
    canonical = dict(payload)
    canonical.pop(field, None)
    if len(claimed) != 64 or claimed != _canonical_sha256(canonical):
        raise ValueError(f"selective-event {field} binding is invalid")
    return claimed


def _top_score_row(rows: object, requested: int = 100) -> Mapping[str, object]:
    if not isinstance(rows, list):
        raise ValueError("selective-event top-score evidence is missing")
    matched = [
        row
        for row in rows
        if isinstance(row, Mapping) and int(row.get("requested_rows") or 0) == requested
    ]
    if len(matched) != 1:
        raise ValueError("selective-event top-score evidence is ambiguous")
    return matched[0]


def _validated_evidence(
    evidence_root: Path,
    design: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    report_path = evidence_root / "report.json"
    report = _read_json(report_path)
    _canonical_payload_hash(report, "report_sha256")
    if (
        report.get("status") != "completed"
        or report.get("design_sha256") != design["design_sha256"]
        or report.get("terminal_holdout_accessed") is not False
        or report.get("trading_authority") is not False
        or report.get("profitability_claim") is not False
        or int(report.get("completed_candidate_count") or 0)
        != int(design["candidate_count"])
        or int(report.get("failed_model_fit_count") or 0) != 0
    ):
        raise ValueError("selective-event report contract is invalid")
    ranked = report.get("ranked_results")
    if not isinstance(ranked, list) or len(ranked) != int(design["candidate_count"]):
        raise ValueError("selective-event ranked results are incomplete")
    report_outcomes = {
        str(value.get("candidate_id")): value
        for value in ranked
        if isinstance(value, Mapping)
    }
    if len(report_outcomes) != len(ranked):
        raise ValueError("selective-event report candidate IDs are not unique")

    risk_profiles = design["risk_profiles"]
    horizons = design["horizon_seconds"]
    training = design["training"]
    assert isinstance(risk_profiles, Mapping)
    assert isinstance(horizons, list)
    assert isinstance(training, Mapping)
    artifacts: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for horizon in horizons:
        for risk_level in risk_profiles:
            fit_id = f"{risk_level}-h{int(horizon)}"
            path = evidence_root / f"{fit_id}.json"
            artifact = _read_json(path)
            _canonical_payload_hash(artifact, "artifact_sha256")
            model_evidence = artifact.get("model_evidence")
            model_strings = artifact.get("model_strings")
            outcomes = artifact.get("outcomes")
            if (
                artifact.get("model_fit_id") != fit_id
                or artifact.get("design_sha256") != design["design_sha256"]
                or artifact.get("corpus_certificate_sha256")
                != report.get("corpus_certificate_sha256")
                or artifact.get("terminal_holdout_accessed") is not False
                or artifact.get("trading_authority") is not False
                or artifact.get("profitability_claim") is not False
                or artifact.get("runtime_resources") != report.get("runtime_resources")
                or not isinstance(model_evidence, Mapping)
                or not isinstance(model_strings, Mapping)
                or not isinstance(outcomes, list)
                or len(outcomes) != len(training["score_methods"])
                or model_evidence.get("model_sha256")
                != _canonical_sha256(dict(sorted(model_strings.items())))
            ):
                raise ValueError(f"selective-event fit artifact is invalid: {fit_id}")
            for raw in outcomes:
                if not isinstance(raw, Mapping):
                    raise ValueError(f"selective-event outcome is invalid: {fit_id}")
                candidate_id = str(raw.get("candidate_id") or "")
                report_outcome = report_outcomes.get(candidate_id)
                if (
                    report_outcome is None
                    or _canonical_sha256(dict(raw))
                    != _canonical_sha256(dict(report_outcome))
                ):
                    raise ValueError(f"selective-event report/artifact drift: {candidate_id}")
                policy = raw.get("policy")
                selection = raw.get("selection_metrics")
                confidence = raw.get("selection_confidence")
                event_rows = artifact.get("event_rows")
                if not all(
                    isinstance(value, Mapping)
                    for value in (policy, selection, confidence, event_rows)
                ):
                    raise ValueError(f"selective-event metrics are incomplete: {candidate_id}")
                assert isinstance(policy, Mapping)
                assert isinstance(selection, Mapping)
                assert isinstance(confidence, Mapping)
                assert isinstance(event_rows, Mapping)
                policy_metrics = policy.get("best_observed_metrics")
                if not isinstance(policy_metrics, Mapping):
                    raise ValueError(f"selective-event policy evidence is missing: {candidate_id}")
                policy_trades = int(policy_metrics.get("trades") or 0)
                policy_total = _finite(
                    policy_metrics.get("total_net_bps"),
                    label="policy return",
                )
                top_100 = _top_score_row(raw.get("selection_top_score_rows"))
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "horizon_seconds": int(raw["horizon_seconds"]),
                        "risk_level": str(raw["risk_level"]),
                        "score_method": str(raw["score_method"]),
                        "status": str(raw["status"]),
                        "policy_accepted": bool(policy["accepted"]),
                        "minimum_policy_trades": int(policy["minimum_trades"]),
                        "policy_trades": policy_trades,
                        "policy_total_net_bps": policy_total,
                        "policy_mean_net_bps": policy_total / policy_trades,
                        "policy_max_drawdown_bps": _finite(
                            policy_metrics.get("max_drawdown_bps"),
                            label="policy drawdown",
                        ),
                        "policy_profit_factor": _finite(
                            policy_metrics.get("profit_factor"),
                            label="policy profit factor",
                        ),
                        "policy_threshold": _finite(
                            policy.get("best_observed_threshold"),
                            label="policy threshold",
                        ),
                        "selection_trades": int(selection.get("trades") or 0),
                        "selection_total_net_bps": _finite(
                            selection.get("total_net_bps"),
                            label="selection return",
                        ),
                        "selection_max_drawdown_bps": _finite(
                            selection.get("max_drawdown_bps"),
                            label="selection drawdown",
                        ),
                        "selection_profit_factor": (
                            ""
                            if selection.get("profit_factor") is None
                            else _finite(
                                selection["profit_factor"],
                                label="selection profit factor",
                            )
                        ),
                        "selection_daily_ci_lower_bps": _finite(
                            confidence.get("mean_daily_net_bps_ci_lower"),
                            label="selection daily lower bound",
                        ),
                        "selection_top_100_mean_actual_net_bps": _finite(
                            top_100.get("mean_actual_net_bps"),
                            label="top-100 return",
                        ),
                        "selection_top_100_profitable_ratio": _finite(
                            top_100.get("actual_profitable_ratio"),
                            label="top-100 profitable ratio",
                        ),
                        "selection_event_rows": int(event_rows["selection"]),
                        "model_backend": str(model_evidence.get("backend_kind") or ""),
                        "model_device": str(model_evidence.get("backend_device") or ""),
                        "parameter_profile": str(
                            model_evidence.get("parameter_profile") or ""
                        ),
                        "model_sha256": str(model_evidence.get("model_sha256") or ""),
                        "artifact_sha256": str(artifact["artifact_sha256"]),
                        "rejection_reasons": ";".join(
                            str(value) for value in raw.get("rejection_reasons") or ()
                        ),
                    }
                )
            artifacts.append(artifact)
    if set(report_outcomes) != {row["candidate_id"] for row in candidates}:
        raise ValueError("selective-event candidate set is inconsistent")
    candidates.sort(
        key=lambda row: (
            int(row["horizon_seconds"]),
            _RISK_ORDER[str(row["risk_level"])],
            tuple(_METHOD_LABELS).index(str(row["score_method"])),
        )
    )
    return report, artifacts, candidates


def _nice_negative_bound(values: Sequence[float]) -> float:
    magnitude = abs(min(values))
    if magnitude <= 0.0:
        return -1.0
    order = 10.0 ** math.floor(math.log10(magnitude / 4.0))
    step = max(order, math.ceil((magnitude / 4.0) / order) * order)
    return -step * math.ceil(magnitude / step)


def _candidate_label(row: Mapping[str, object]) -> str:
    risk = str(row["risk_level"])
    method = _METHOD_LABELS[str(row["score_method"])]
    return f"{int(row['horizon_seconds'])}s  {risk}  {method}"


def _negative_bar_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    title: str,
    subtitle: str,
    footer: str,
) -> str:
    values = [_finite(row[value_key], label=value_key) for row in rows]
    lower = _nice_negative_bound(values)
    width = 1500
    left = 390
    right = 80
    top = 150
    row_height = 34
    chart_width = width - left - right
    chart_height = row_height * len(rows)
    height = top + chart_height + 105
    zero_x = left + chart_width
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{html.escape(title)}</title>",
        f"<desc id=\"desc\">{html.escape(subtitle)} {html.escape(footer)}</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">{html.escape(title)}</text>',
        f'<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">{html.escape(subtitle)}</text>',
    ]
    for index in range(5):
        value = lower + (0.0 - lower) * index / 4.0
        x = left + chart_width * index / 4.0
        lines.append(
            f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{top + chart_height}" stroke="#dce3e8" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">{value:.0f}</text>'
        )
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        y = top + index * row_height
        if index and int(row["horizon_seconds"]) != int(rows[index - 1]["horizon_seconds"]):
            lines.append(
                f'<line x1="56" y1="{y - 6}" x2="{width - 56}" y2="{y - 6}" stroke="#8d9aa5" stroke-width="2"/>'
            )
        center = y + 16
        x = left + chart_width * (value - lower) / (0.0 - lower)
        bar_width = max(2.0, zero_x - x)
        lines.append(
            f'<text x="{left - 18}" y="{center + 5}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#263746">{html.escape(_candidate_label(row))}</text>'
        )
        lines.append(
            f'<rect x="{x:.1f}" y="{y + 5}" width="{bar_width:.1f}" height="22" rx="3" fill="#b42318"/>'
        )
        lines.append(
            f'<text x="{max(left + 4, x - 8):.1f}" y="{center + 5}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#7a271a">{value:+.2f}</text>'
        )
    lines.extend(
        [
            f'<line x1="{zero_x:.1f}" y1="{top - 18}" x2="{zero_x:.1f}" y2="{top + chart_height}" stroke="#17212b" stroke-width="2"/>',
            f'<text x="{left + chart_width / 2:.1f}" y="{top + chart_height + 42}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#52606d">Net basis points</text>',
            f'<text x="56" y="{height - 30}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">{html.escape(footer)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _funnel_svg(*, fits: int, candidates: int, evaluable: int, accepted: int) -> str:
    stages = (
        ("Verified model fits", fits),
        ("Scored candidates", candidates),
        ("Policy-evaluable", evaluable),
        ("Positive policy utility", accepted),
        ("Selection trades", 0),
        ("Unrejected", 0),
    )
    width, height = 1500, 410
    box_width, gap, start_x = 200, 38, 56
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Prediction never cleared the policy economics gate</title>',
        '<desc id="desc">Six verified fits produced eighteen scored candidates. All were policy-evaluable, none had positive after-cost policy utility, no selection trade was permitted, and no candidate survived.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="54" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Prediction never cleared the policy economics gate</text>',
        '<text x="56" y="84" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">Counts are exact Round 12 v6 artifacts; the terminal holdout was not accessed.</text>',
    ]
    for index, (label, value) in enumerate(stages):
        x = start_x + index * (box_width + gap)
        fill = "#f4f7f9" if value else "#fff1f0"
        stroke = "#9aa8b3" if value else "#b42318"
        if index:
            prior_right = x - gap
            lines.append(
                f'<line x1="{prior_right + 7}" y1="210" x2="{x - 9}" y2="210" stroke="#7b8994" stroke-width="2"/>'
            )
            lines.append(
                f'<path d="M {x - 9} 204 L {x} 210 L {x - 9} 216 Z" fill="#7b8994"/>'
            )
        lines.append(
            f'<rect x="{x}" y="142" width="{box_width}" height="136" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="198" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="34" font-weight="700" fill="{stroke}">{value}</text>'
        )
        lines.append(
            f'<text x="{x + box_width / 2:.1f}" y="236" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#263746">{html.escape(label)}</text>'
        )
    lines.extend(
        [
            '<text x="56" y="348" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Activity floors are evidence requirements, not trade quotas. Zero selection trades is the correct result after a negative policy screen.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _progress_rows(path: Path, design: Mapping[str, object], best: Mapping[str, object]) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if rows and int(rows[-1]["round"]) == int(design["round"]):
        rows.pop()
    if not rows or int(rows[-1]["round"]) != int(design["round"]) - 1:
        raise ValueError("prior progress table does not end at the preceding round")
    data = design["data"]
    training = design["training"]
    assert isinstance(data, Mapping)
    assert isinstance(training, Mapping)
    rows.append(
        {
            "round": int(design["round"]),
            "stage": "bounded selective-event viability",
            "periods": f"{data['start_date']}..{data['end_date']}",
            "selection_contaminated": True,
            "horizon_seconds": ";".join(str(v) for v in design["horizon_seconds"]),
            "feature_set": str(training["feature_version"]),
            "risk_level": ";".join(str(v) for v in design["risk_profiles"]),
            "direction_auc": "",
            "spearman_ic": "",
            "selected_signals": int(best["policy_trades"]),
            "executable_trades": 0,
            "mean_gross_bps": "",
            "mean_net_bps": "",
            "status": "rejected",
            "source_file": "selective-event Round 12 v6 report",
            "best_policy_trades": int(best["policy_trades"]),
            "best_policy_total_net_bps": best["policy_total_net_bps"],
            "best_policy_mean_net_bps": best["policy_mean_net_bps"],
            "best_policy_max_drawdown_bps": best["policy_max_drawdown_bps"],
            "best_policy_profit_factor": best["policy_profit_factor"],
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
    design = load_selective_event_design(design_path, require_current=True)
    report, artifacts, candidates = _validated_evidence(evidence_root, design)
    best = max(candidates, key=lambda row: float(row["policy_total_net_bps"]))
    progress = _progress_rows(prior_progress_path, design, best)
    progress_fields = tuple(progress[0]) + tuple(
        key for key in progress[-1] if key not in progress[0]
    )
    charts = output_dir / "charts"
    diagnostics: dict[str, object] = {
        "schema_version": "selective-event-publication-diagnostics-v1",
        "artifact_class": "consumed_selection_diagnostic_no_trading_authority",
        "design_sha256": design["design_sha256"],
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "corpus_certificate_sha256": report["corpus_certificate_sha256"],
        "terminal_holdout_accessed": False,
        "trading_authority": False,
        "profitability_claim": False,
        "runtime_resources": report["runtime_resources"],
        "fits": [
            {
                "model_fit_id": artifact["model_fit_id"],
                "artifact_sha256": artifact["artifact_sha256"],
                "source_evidence": artifact["source_evidence"],
                "path_target_evidence": artifact["path_target_evidence"],
                "role_evidence": artifact["role_evidence"],
                "event_rows": artifact["event_rows"],
                "model_evidence": artifact["model_evidence"],
                "best_iterations": artifact["best_iterations"],
                "probability_calibration": artifact["probability_calibration"],
            }
            for artifact in artifacts
        ],
        "candidates": candidates,
    }
    diagnostics["diagnostic_sha256"] = _canonical_sha256(diagnostics)

    _write_csv(output_dir / "candidates.csv", candidates, _CANDIDATE_FIELDS)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "diagnostics.json",
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(
        charts / "after-cost-performance.svg",
        _negative_bar_svg(
            candidates,
            value_key="policy_total_net_bps",
            title="Every policy tail lost after costs",
            subtitle="Five-day policy window; each bar is a non-overlapping trade trace selected from a fixed score threshold search.",
            footer="Fees: 5 bps/side; added slippage: 1 bps/side; latency: 750 ms. Total net bps is not portfolio ROI.",
        ),
    )
    _write_text(
        charts / "forecast-quality.svg",
        _negative_bar_svg(
            candidates,
            value_key="selection_top_100_mean_actual_net_bps",
            title="Top-ranked selection rows still had negative mean returns",
            subtitle="Consumed six-day selection window; mean realized after-cost outcome among each candidate's 100 strongest scored rows.",
            footer="Diagnostic ranking only: rows can overlap and do not form an equity curve or executable trade series.",
        ),
    )
    evaluable = sum(bool(row["policy_trades"]) for row in candidates)
    accepted = sum(bool(row["policy_accepted"]) for row in candidates)
    _write_text(
        charts / "action-funnel.svg",
        _funnel_svg(
            fits=len(artifacts),
            candidates=len(candidates),
            evaluable=evaluable,
            accepted=accepted,
        ),
    )
    _write_text(charts / "research-progress.svg", _progress_svg(progress))

    data = design["data"]
    reserved = design["reserved_terminal"]
    change_control = design["change_control"]
    assert isinstance(data, Mapping)
    assert isinstance(reserved, Mapping)
    assert isinstance(change_control, Mapping)
    readme = f"""# Round 12: bounded viability

**Rejected.** All 18 candidates completed; none had positive after-cost policy utility. The best policy trace lost **{float(best['policy_total_net_bps']):.2f} bps** over **{int(best['policy_trades'])} trades** with **{float(best['policy_max_drawdown_bps']):.2f} bps** max drawdown.

![Policy economics](charts/after-cost-performance.svg)

![Top-score reality check](charts/forecast-quality.svg)

![Action funnel](charts/action-funnel.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, {data['start_date']} through {data['end_date']} UTC. The window is consumed; the {reserved['start_date']} terminal day remains untouched. This is research evidence, not profitability or trading authority.

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
    publication: dict[str, object] = {
        "schema_version": "selective-event-discovery-publication-v1",
        "artifact_class": "exchange_sourced_model_discovery_graph_data",
        "round": int(design["round"]),
        "design_revision": int(design["design_revision"]),
        "status": "rejected",
        "runner_status": report["status"],
        "trading_authority": False,
        "profitability_claim": False,
        "terminal_holdout_accessed": False,
        "selection_window_is_consumed": True,
        "design_sha256": design["design_sha256"],
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "implementation_commit": change_control["implementation_commit"],
        "corpus_certificate_sha256": report["corpus_certificate_sha256"],
        "diagnostic_sha256": diagnostics["diagnostic_sha256"],
        "runtime_resources": report["runtime_resources"],
        "actual": {
            "model_fit_count": len(artifacts),
            "candidate_count": len(candidates),
            "fit_error_count": int(report["failed_model_fit_count"]),
            "unrejected_candidate_count": int(report["unrejected_candidate_count"]),
            "policy_evaluable_candidate_count": evaluable,
            "positive_policy_candidate_count": accepted,
            "selection_trades": sum(int(row["selection_trades"]) for row in candidates),
            "best_policy_candidate_id": best["candidate_id"],
            "best_policy_trades": best["policy_trades"],
            "best_policy_total_net_bps": best["policy_total_net_bps"],
            "best_policy_max_drawdown_bps": best["policy_max_drawdown_bps"],
            "best_policy_profit_factor": best["policy_profit_factor"],
        },
        "source_artifacts": [
            {
                "name": f"{artifact['model_fit_id']}.json",
                "sha256": artifact["artifact_sha256"],
                "bytes": (evidence_root / f"{artifact['model_fit_id']}.json").stat().st_size,
            }
            for artifact in artifacts
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
        "selective-event-publication: "
        f"status={publication['status']} "
        f"candidates={publication['actual']['candidate_count']} "
        f"sha256={publication['publication_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
