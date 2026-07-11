"""Publish verified adaptive-action evidence and graph data."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    from tools.run_adaptive_action_screen import (
        REPORT_SCHEMA_VERSION,
        load_adaptive_action_design,
    )
    from tools.run_gross_architecture_screen import _canonical_sha256, _is_sha256
    from tools.run_outcome_mixture_screen import load_outcome_mixture_design
except ModuleNotFoundError:  # pragma: no cover - direct tools directory execution
    from publish_daily_walkforward_screen import _progress_svg
    from run_adaptive_action_screen import (
        REPORT_SCHEMA_VERSION,
        load_adaptive_action_design,
    )
    from run_gross_architecture_screen import _canonical_sha256, _is_sha256
    from run_outcome_mixture_screen import load_outcome_mixture_design


PUBLICATION_SCHEMA_VERSION = "adaptive-action-screen-publication-v1"
_FALSE_CLAIMS = (
    "trading_authority",
    "execution_claim",
    "profitability_claim",
    "portfolio_claim",
    "leverage_applied",
)
_PROGRESS_IDENTITIES = {
    16: (
        "adaptive 100 ms barrier action-value ensemble",
        "three-seed adaptive-barrier-shared-residual",
    ),
    17: (
        "conditional win/loss outcome-mixture ensemble",
        "three-seed conditional-outcome-mixture-shared-residual",
    ),
    18: (
        "rank-regularized conditional outcome-mixture ensemble",
        "three-seed rank-regularized-outcome-mixture-shared-residual",
    ),
    19: (
        "depth-normalized order-flow conditional distribution model",
        "three-seed depth-normalized-order-flow outcome-mixture",
    ),
    20: (
        "parameter-matched direction-specific representation ablation",
        "three-seed independent-long-short outcome-mixture",
    ),
    21: (
        "sampled pairwise net-return ranking ablation",
        "three-seed independent-long-short pairwise-ranked outcome-mixture",
    ),
    22: (
        "additive pairwise net-return regularization",
        "three-seed calibration-preserving additive-pairwise outcome-mixture",
    ),
    23: (
        "bounded causal temporal-attention ablation",
        "three-seed causal-temporal-attention outcome-mixture",
    ),
}


def _progress_identity(round_number: int) -> tuple[str, str]:
    try:
        return _PROGRESS_IDENTITIES[round_number]
    except KeyError as exc:
        raise ValueError(
            f"adaptive action publication copy is undefined for Round {round_number}"
        ) from exc


def _publication_narrative(
    round_number: int,
    *,
    all_candidate_stress_nets_negative: bool,
) -> tuple[str, str, str]:
    if round_number == 16:
        return (
            "action-value ensemble abstained",
            "The three-seed DirectML ensemble improved forecast error and probability-of-profit discrimination in places, but the highest-ranked signals remained negative net of costs.",
            "The next model change must estimate conditional profit/loss outcomes rather than relax the risk controls.",
        )
    if round_number == 17:
        return (
            "conditional net-return distribution model abstained",
            "The conditional profit/loss decomposition improved point-error metrics versus the zero-return benchmark, but probability calibration was mostly worse than the prevalence benchmark and the highest-ranked signals remained negative net of costs.",
            "The next precommitted model change must target regime-conditioned net-return ranking and probability calibration rather than relax the risk controls.",
        )
    if round_number == 18:
        summary = (
            "The added ranking objective produced a small aggressive-profile candidate set, but every threshold-selection simulation lost money after stress costs."
            if all_candidate_stress_nets_negative
            else "The added ranking objective produced threshold-selection candidates, but none passed the precommitted stress-test acceptance criteria."
        )
        return (
            "rank-regularized net-return model abstained",
            summary,
            "The next precommitted model change must target regime-conditioned net-return ranking and probability calibration rather than relax the risk controls.",
        )
    if round_number == 19:
        return (
            "depth-normalized order-flow model abstained",
            "Depth-normalized aggressive order-flow inputs increased signal counts under the aggressive risk profile, but every threshold-selection simulation lost money after stress costs.",
            "The next precommitted model change must address direction-specific net-return ranking and probability calibration rather than add further depth-normalized inputs or relax the risk controls.",
        )
    if round_number == 20:
        return (
            "direction-specific outcome model abstained",
            "Parameter-matched independent long and short representations increased signals meeting pre-threshold controls, but every threshold-selection simulation remained negative net of stress costs.",
            "The next precommitted change must improve decision-objective alignment with realized net returns rather than add capacity or relax the risk controls.",
        )
    if round_number == 21:
        return (
            "pairwise net-return ranking model abstained",
            "Sampled pairwise net-return ranking improved several discrimination and short-side tail diagnostics, but worsened the best out-of-sample tail and eliminated threshold-selection eligibility across all risk profiles.",
            "The next precommitted change must restore calibrated positive expected-return separation while retaining net-return ordering and the existing risk controls.",
        )
    if round_number == 22:
        return (
            "additive net-return ranking model abstained",
            "The bounded additive pairwise term restored Round 20-like signal eligibility but did not improve the best out-of-sample tail, and every threshold-selection simulation remained negative net of stress costs.",
            "Further ranking-loss tuning is not justified; the next precommitted change must address target horizon or regime conditioning without relaxing the risk controls.",
        )
    if round_number == 23:
        return (
            "causal temporal-attention outcome model abstained",
            "The bounded 30-second context improved the policy-window long top-100 mean net return under stress, but the gain did not persist in the calibration window or broader ranked tails, signal eligibility fell sharply, and every nonempty threshold-selection simulation lost money after stress costs.",
            "The next precommitted change must test regime- or horizon-conditioned target formation and ranking stability without relaxing any risk control; the isolated positive policy tail is insufficient evidence of an edge.",
        )
    raise ValueError(
        f"adaptive action publication narrative is undefined for Round {round_number}"
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish verified adaptive-action evidence"
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--prior-progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"adaptive action publication cannot read {path.name}"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"adaptive action publication {path.name} is not an object")
    return value


def _finite(value: object, *, label: str) -> float:
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"adaptive action publication {label} is non-finite")
    return output


def _require_false_claims(value: Mapping[str, object], *, label: str) -> None:
    if any(value.get(name) is not False for name in _FALSE_CLAIMS):
        raise ValueError(f"adaptive action publication {label} authority drifted")


def _safe_artifact(evidence_root: Path, raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError("adaptive action model artifact is invalid")
    relative = Path(str(raw.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("adaptive action model artifact path is unsafe")
    path = evidence_root / relative
    if (
        not path.is_file()
        or not _is_sha256(raw.get("sha256"))
        or _sha256(path) != raw["sha256"]
        or path.stat().st_size != int(raw.get("bytes") or -1)
    ):
        raise ValueError("adaptive action model artifact integrity failed")
    return {
        "path": relative.as_posix(),
        "sha256": str(raw["sha256"]),
        "bytes": path.stat().st_size,
    }


def _validated_evidence(
    evidence_root: Path,
    design: Mapping[str, object],
    design_sha256: str,
) -> dict[str, object]:
    report_path = evidence_root / "report.json"
    report = _read_json(report_path)
    claimed = report.get("report_sha256")
    canonical = dict(report)
    canonical.pop("report_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(canonical):
        raise ValueError("adaptive action source report hash is invalid")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != design.get("round")
        or report.get("design_sha256") != design_sha256
        or report.get("status") != "rejected"
        or report.get("terminal_holdout_accessed") is not False
        or report.get("development_window_is_consumed") is not False
        or report.get("policy_survivors") != []
        or report.get("final_profiles") != []
    ):
        raise ValueError("adaptive action source report contract is invalid")
    _require_false_claims(report, label="source report")
    dataset = report.get("dataset")
    forecasts = report.get("forecast_diagnostics")
    profiles = report.get("profile_results")
    models = report.get("ensemble_models")
    if (
        not isinstance(dataset, Mapping)
        or not isinstance(forecasts, Mapping)
        or not isinstance(profiles, list)
        or not isinstance(models, list)
        or len(models) != 3
        or tuple(
            value.get("profile") if isinstance(value, Mapping) else None
            for value in profiles
        )
        != ("conservative", "regular", "aggressive")
    ):
        raise ValueError("adaptive action source report sections are incomplete")
    if (
        dataset.get("barrier_targets_sha256") is None
        or dataset.get("barrier_summary") is None
        or int(dataset.get("valid_barrier_rows") or 0) <= 0
    ):
        raise ValueError("adaptive action barrier evidence is incomplete")
    artifacts = []
    for member in models:
        if not isinstance(member, Mapping) or not isinstance(
            member.get("model"), Mapping
        ):
            raise ValueError("adaptive action model member is invalid")
        _require_false_claims(member["model"], label="model member")
        artifact = member.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError("adaptive action model artifact is invalid")
        if int(report["round"]) >= 17 and artifact.get("reload_verified") is not True:
            raise ValueError("adaptive action model artifact reload was not verified")
        artifacts.append(_safe_artifact(evidence_root, artifact))
    report["_validated_artifacts"] = artifacts
    return report


def _forecast_rows(
    report: Mapping[str, object], design: Mapping[str, object]
) -> list[dict[str, object]]:
    forecasts = report["forecast_diagnostics"]
    data = design["data"]
    assert isinstance(forecasts, Mapping)
    assert isinstance(data, Mapping)
    roles = data["roles"]
    assert isinstance(roles, Mapping)
    output: list[dict[str, object]] = []
    for key in (
        "calibration_base",
        "calibration_stress",
        "policy_base",
        "policy_stress",
    ):
        raw = forecasts.get(key)
        if not isinstance(raw, Mapping):
            raise ValueError(f"adaptive action {key} forecast diagnostics are missing")
        _require_false_claims(raw, label=f"{key} forecast")
        role, scenario = key.split("_", 1)
        role_dates = roles[role]
        if not isinstance(role_dates, Mapping):
            raise ValueError("adaptive action forecast role dates are invalid")
        sides = raw.get("sides")
        if not isinstance(sides, Mapping):
            raise ValueError("adaptive action forecast sides are missing")
        for side in ("long", "short"):
            metrics = sides.get(side)
            if not isinstance(metrics, Mapping):
                raise ValueError("adaptive action side forecast is missing")
            top = metrics.get("top_rows")
            if not isinstance(top, list) or len(top) != 3:
                raise ValueError("adaptive action top-row diagnostics are incomplete")
            top_by_request = {
                int(value["requested_rows"]): value
                for value in top
                if isinstance(value, Mapping)
            }
            if set(top_by_request) != {100, 500, 1_000}:
                raise ValueError("adaptive action top-row diagnostics drifted")
            output.append(
                {
                    "role": role,
                    "scenario": scenario,
                    "side": side,
                    "start_date": role_dates["start"],
                    "end_date": role_dates["end"],
                    "rows": int(metrics["rows"]),
                    "positive_ratio": _finite(
                        metrics["actual_positive_ratio"], label="positive ratio"
                    ),
                    "auc": _finite(metrics["profitable_auc"], label="AUC"),
                    "pearson_ic": _finite(
                        metrics["pearson_information_coefficient"], label="IC"
                    ),
                    "mae_bps": _finite(metrics["mean_absolute_error_bps"], label="MAE"),
                    "zero_mae_bps": _finite(
                        metrics["zero_baseline_mae_bps"], label="zero MAE"
                    ),
                    "brier": _finite(metrics["profitable_brier"], label="Brier"),
                    "prevalence_brier": _finite(
                        metrics["prevalence_brier"], label="prevalence Brier"
                    ),
                    "interval_80_coverage": _finite(
                        metrics["interval_80_coverage"], label="interval coverage"
                    ),
                    "mean_epistemic_std_bps": _finite(
                        metrics["mean_epistemic_std_bps"], label="epistemic std"
                    ),
                    "top_100_mean_net_bps": _finite(
                        top_by_request[100]["mean_actual_net_bps"],
                        label="top 100 net",
                    ),
                    "top_500_mean_net_bps": _finite(
                        top_by_request[500]["mean_actual_net_bps"],
                        label="top 500 net",
                    ),
                    "top_1000_mean_net_bps": _finite(
                        top_by_request[1_000]["mean_actual_net_bps"],
                        label="top 1000 net",
                    ),
                }
            )
    return output


def _profile_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in report["profile_results"]:
        if not isinstance(raw, Mapping):
            raise ValueError("adaptive action profile result is invalid")
        _require_false_claims(raw, label="profile result")
        selection = raw.get("threshold_selection")
        if not isinstance(selection, Mapping):
            raise ValueError("adaptive action threshold selection is missing")
        _require_false_claims(selection, label="threshold selection")
        candidates = selection.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("adaptive action threshold candidates are invalid")
        output.append(
            {
                "profile": raw["profile"],
                "calibration_eligible_rows": int(raw["calibration_eligible_rows"]),
                "calibration_threshold_candidates": len(candidates),
                "calibration_threshold_accepted": bool(selection["accepted"]),
                "calibration_rejection_reasons": ";".join(
                    str(value) for value in selection["rejection_reasons"]
                ),
                "policy_eligible_rows": int(raw["policy_eligible_rows"]),
                "policy_status": raw["policy_status"],
                "policy_trades": int(raw["policy_stress_trace"]["metrics"]["trades"]),
                "policy_total_net_bps": _finite(
                    raw["policy_stress_trace"]["metrics"]["total_net_bps"],
                    label="out-of-sample net",
                ),
                "development_evaluated": bool(raw["development_evaluated"]),
            }
        )
    return output


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in report["profile_results"]:
        if not isinstance(raw, Mapping):
            raise ValueError("adaptive action profile result is invalid")
        selection = raw.get("threshold_selection")
        if not isinstance(selection, Mapping):
            raise ValueError("adaptive action threshold selection is missing")
        candidates = selection.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("adaptive action threshold candidates are invalid")
        if not candidates:
            output.append(
                {
                    "profile": raw["profile"],
                    "candidate_available": False,
                    "quantile": "",
                    "threshold_bps": "",
                    "accepted": False,
                    "stress_trades": 0,
                    "stress_total_net_bps": "",
                    "stress_max_drawdown_bps": "",
                    "stress_profit_factor": "",
                    "stress_win_rate": "",
                    "rejection_reasons": ";".join(
                        str(value) for value in selection["rejection_reasons"]
                    ),
                }
            )
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("adaptive action threshold candidate is invalid")
            _require_false_claims(candidate, label="threshold candidate")
            stress = candidate.get("stress_metrics")
            if not isinstance(stress, Mapping):
                raise ValueError("adaptive action threshold stress metrics are missing")
            profit_factor = stress.get("profit_factor")
            output.append(
                {
                    "profile": raw["profile"],
                    "candidate_available": True,
                    "quantile": _finite(candidate["quantile"], label="quantile"),
                    "threshold_bps": _finite(
                        candidate["threshold_bps"], label="threshold"
                    ),
                    "accepted": bool(candidate["accepted"]),
                    "stress_trades": int(stress["trades"]),
                    "stress_total_net_bps": _finite(
                        stress["total_net_bps"], label="threshold net"
                    ),
                    "stress_max_drawdown_bps": _finite(
                        stress["max_drawdown_bps"], label="threshold drawdown"
                    ),
                    "stress_profit_factor": (
                        ""
                        if profit_factor is None
                        else _finite(profit_factor, label="threshold profit factor")
                    ),
                    "stress_win_rate": _finite(
                        stress["win_rate"], label="threshold win rate"
                    ),
                    "rejection_reasons": ";".join(
                        str(value) for value in candidate["rejection_reasons"]
                    ),
                }
            )
    return output


def _barrier_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    dataset = report["dataset"]
    assert isinstance(dataset, Mapping)
    summary = dataset["barrier_summary"]
    assert isinstance(summary, Mapping)
    output: list[dict[str, object]] = []
    for scenario in ("base", "stress"):
        scenario_data = summary[scenario]
        assert isinstance(scenario_data, Mapping)
        for side in ("long", "short"):
            metrics = scenario_data[side]
            outcomes = scenario_data[f"{side}_outcomes"]
            assert isinstance(metrics, Mapping)
            assert isinstance(outcomes, Mapping)
            output.append(
                {
                    "scenario": scenario,
                    "side": side,
                    "rows": int(metrics["rows"]),
                    "positive_rows": int(metrics["positive_rows"]),
                    "positive_ratio": _finite(
                        metrics["positive_ratio"], label="barrier positive ratio"
                    ),
                    "mean_net_bps": _finite(
                        metrics["mean_net_bps"], label="barrier mean net"
                    ),
                    "horizon": int(outcomes["horizon"]),
                    "stop": int(outcomes["stop"]),
                    "take": int(outcomes["take"]),
                    "ambiguous_stop": int(outcomes["ambiguous_stop"]),
                    "protection_gap_stop": int(outcomes["protection_gap_stop"]),
                }
            )
    return output


def _read_progress(path: Path, *, target_round: int = 16) -> list[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(value) for value in csv.DictReader(stream)]
    except OSError as exc:
        raise ValueError("adaptive action prior progress is unreadable") from exc
    if (
        not rows
        or int(rows[-1].get("round") or 0) != target_round - 1
        or any(int(value.get("round") or 0) == target_round for value in rows)
    ):
        raise ValueError("adaptive action prior progress lineage is invalid")
    return rows


def _progress_rows(
    prior_path: Path,
    report: Mapping[str, object],
    forecast_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    round_number = int(report["round"])
    rows = _read_progress(prior_path, target_round=round_number)
    policy_stress = [
        row
        for row in forecast_rows
        if row["role"] == "policy" and row["scenario"] == "stress"
    ]
    profiles = report["profile_results"]
    stage, best_model_id = _progress_identity(round_number)
    rows.append(
        {
            "round": round_number,
            "stage": stage,
            "periods": "2023-05-16..2023-07-06",
            "selection_contaminated": True,
            "horizon_seconds": 900,
            "feature_set": (
                "l1-tape-causal-v8" if round_number >= 19 else "l1-tape-causal-v7"
            ),
            "risk_level": "conservative;regular;aggressive",
            "direction_auc": max(float(value["auc"]) for value in policy_stress),
            "spearman_ic": "",
            "selected_signals": 0,
            "executable_trades": 0,
            "mean_gross_bps": "",
            "mean_net_bps": "",
            "status": "rejected",
            "source_file": f"adaptive action-value Round {round_number} report",
            "best_model_id": best_model_id,
            "best_top_500_exact_after_cost_bps": max(
                float(value["top_500_mean_net_bps"]) for value in policy_stress
            ),
            "after_cost_diagnostic_rows": 500,
            "ensemble_models": len(report["ensemble_models"]),
            "valid_barrier_rows": report["dataset"]["valid_barrier_rows"],
            "calibration_eligible_rows": max(
                int(value["calibration_eligible_rows"]) for value in profiles
            ),
            "policy_eligible_rows": max(
                int(value["policy_eligible_rows"]) for value in profiles
            ),
            "development_consumed": False,
        }
    )
    return rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or ())
    if not names:
        for row in rows:
            for name in row:
                if name not in names:
                    names.append(name)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _forecast_svg(
    rows: Sequence[Mapping[str, object]], *, round_number: int = 16
) -> str:
    selected = [row for row in rows if row["scenario"] == "stress"]
    width, height = 1240, 560
    left, top, chart_width, chart_height = 90, 130, 1080, 300
    bar_width = 88
    colors = {"long": "#16827a", "short": "#8d5aa7"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Round {round_number} probability-of-profit discrimination by evaluation window</title>',
        '<desc id="desc">Long and short adverse-stress ROC AUC for threshold selection from June 21 to 25 and out-of-sample validation from June 26 to 30, 2023. A dashed line marks random discrimination at 0.5.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#17212b">Probability-of-profit discrimination</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53616d">Adverse-stress outcomes; ROC AUC above 0.5 indicates discrimination better than chance.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_height * (1.0 - value)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" stroke="#{"6e7d88" if value == 0.5 else "e6ebef"}" stroke-width="{2 if value == 0.5 else 1}" stroke-dasharray="{("7 6" if value == 0.5 else "none")}"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value:.2f}</text>'
        )
    for index, row in enumerate(selected):
        x = left + 135 + index * 240
        value = float(row["auc"])
        role_label = {
            "calibration": "Threshold selection",
            "policy": "Out-of-sample",
        }[str(row["role"])]
        y = top + chart_height * (1.0 - value)
        h = top + chart_height - y
        color = colors[str(row["side"])]
        lines.extend(
            [
                f'<rect x="{x - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#253744">{value:.3f}</text>',
                f'<text x="{x:.1f}" y="{top + chart_height + 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">{role_label} {html.escape(str(row["side"]))}</text>',
                f'<text x="{x:.1f}" y="{top + chart_height + 48}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7882">{html.escape(str(row["start_date"]))} to {html.escape(str(row["end_date"]))}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="34" y="285" transform="rotate(-90 34 285)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">ROC AUC</text>',
            '<text x="48" y="532" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">ROC AUC alone does not establish a net-of-cost edge; all precommitted risk controls remained binding.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _tail_svg(rows: Sequence[Mapping[str, object]], *, round_number: int = 16) -> str:
    selected = [row for row in rows if row["scenario"] == "stress"]
    width, height = 1240, 580
    left, top, chart_width, chart_height = 90, 130, 1080, 310
    tail_values = [
        float(row[field])
        for row in selected
        for field in ("top_100_mean_net_bps", "top_500_mean_net_bps")
    ]
    minimum, maximum = min(tail_values), max(tail_values)
    lower = min(-22.0, math.floor(minimum) - 3.0)
    upper = max(2.0, math.ceil(maximum) + 1.0)
    all_negative = maximum < 0.0
    headline = (
        "Highest-ranked signals remained negative net of costs"
        if all_negative
        else "Ranked-tail economics were mixed and not stable"
    )
    description = (
        "Every displayed mean is negative."
        if all_negative
        else "One or more displayed means are positive, but no threshold passed the precommitted stress controls."
    )

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Round {round_number} net returns for highest-ranked signals</title>',
        f'<desc id="desc">Mean adverse-stress net return in basis points for the 100 and 500 highest-ranked signals in threshold-selection and out-of-sample validation windows during June 2023. {description}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#17212b">{headline}</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53616d">Realized 100 ms BBO-path outcomes include fees, latency, slippage and delayed stop execution.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    tick_step = max(5.0, math.ceil(((upper - lower) / 6.0) / 5.0) * 5.0)
    first_tick = math.ceil(lower / tick_step) * tick_step
    tick_count = int(math.floor((upper - first_tick) / tick_step)) + 1
    for value in (first_tick + index * tick_step for index in range(tick_count)):
        yy = y(value)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#{"536674" if value == 0 else "e6ebef"}" stroke-width="{2 if value == 0 else 1}"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value:.0f}</text>'
        )
    colors = ("#c64c3f", "#e3a229")
    for index, row in enumerate(selected):
        center = left + 140 + index * 240
        role_label = {
            "calibration": "Threshold selection",
            "policy": "Out-of-sample",
        }[str(row["role"])]
        for offset, (field, label, color) in enumerate(
            zip(
                ("top_100_mean_net_bps", "top_500_mean_net_bps"),
                ("top 100", "top 500"),
                colors,
                strict=True,
            )
        ):
            value = float(row[field])
            xx = center + (offset - 0.5) * 72
            yy = y(value)
            zero_y = y(0.0)
            bar_y = min(yy, zero_y)
            bar_height = abs(yy - zero_y)
            label_y = yy - 8.0 if value >= 0.0 else yy + 18.0
            bar_color = "#16827a" if value > 0.0 else color
            lines.extend(
                [
                    f'<rect x="{xx - 29:.1f}" y="{bar_y:.1f}" width="58" height="{bar_height:.1f}" fill="{bar_color}"/>',
                    f'<text x="{xx:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" fill="#344652">{value:+.2f}</text>',
                    f'<text x="{xx:.1f}" y="{top + chart_height + 48}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#6b7882">{label}</text>',
                ]
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 27}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#334653">{role_label} {html.escape(str(row["side"]))}</text>'
        )
    lines.extend(
        [
            '<text x="34" y="295" transform="rotate(-90 34 295)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">Mean net basis points</text>',
            '<text x="48" y="550" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Bars are measured outcomes, not forecasts. No threshold was relaxed after seeing them.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _funnel_svg(rows: Sequence[Mapping[str, object]], *, round_number: int = 16) -> str:
    width, height = 1120, 540
    max_value = max(1, max(int(row["policy_eligible_rows"]) for row in rows))
    left, top, chart_width, chart_height = 100, 130, 940, 270
    colors = {"calibration": "#16827a", "policy": "#8d5aa7"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Round {round_number} signals passing pre-trade risk controls</title>',
        '<desc id="desc">Signal counts passing pre-threshold controls in threshold-selection and out-of-sample validation windows for conservative, regular and aggressive risk profiles. No simulated trade is permitted without an accepted threshold.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#17212b">No candidate threshold passed all pre-trade risk controls</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53616d">Signals require positive uncertainty-adjusted expected return, probability of profit, ensemble agreement and lower-tail controls.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in range(0, max_value + 1, max(1, math.ceil(max_value / 5))):
        yy = top + chart_height * (1.0 - value / max_value)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#e6ebef"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value}</text>'
        )
    for index, row in enumerate(rows):
        center = left + 170 + index * 300
        for offset, (field, color_key, label) in enumerate(
            (
                ("calibration_eligible_rows", "calibration", "selection"),
                ("policy_eligible_rows", "policy", "validation"),
            )
        ):
            value = int(row[field])
            xx = center + (offset - 0.5) * 86
            bar_height = chart_height * value / max_value
            yy = top + chart_height - bar_height
            lines.extend(
                [
                    f'<rect x="{xx - 34:.1f}" y="{yy:.1f}" width="68" height="{bar_height:.1f}" fill="{colors[color_key]}"/>',
                    f'<text x="{xx:.1f}" y="{yy - 10:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#253744">{value}</text>',
                    f'<text x="{xx:.1f}" y="{top + chart_height + 46}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="#6b7882">{label}</text>',
                ]
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 25}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" fill="#334653">{html.escape(str(row["profile"]).title())}</text>'
        )
    lines.extend(
        [
            '<text x="48" y="510" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">Threshold selection: 2023-06-21 to 2023-06-25 UTC. Out-of-sample validation: 2023-06-26 to 2023-06-30 UTC.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _barrier_svg(
    rows: Sequence[Mapping[str, object]], *, round_number: int = 16
) -> str:
    width, height = 1200, 560
    left, top, chart_width, chart_height = 100, 130, 1020, 280
    colors = {
        "horizon": "#59788e",
        "stop": "#c64c3f",
        "take": "#16827a",
        "ambiguous_stop": "#e3a229",
        "protection_gap_stop": "#8d5aa7",
    }
    outcomes = tuple(colors)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">Round {round_number} adaptive barrier outcome composition</title>',
        '<desc id="desc">Stacked percentages of horizon, stop, take, ambiguous stop and protection-gap stop outcomes for base and adverse-stress long and short paths from May 16 to July 6, 2023.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="48" font-family="Segoe UI, Arial, sans-serif" font-size="27" font-weight="700" fill="#17212b">Path outcomes were predominantly horizon exits</text>',
        '<text x="48" y="78" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53616d">Real Binance Futures 100 ms BBO paths; each stack totals the valid event rows.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in (0, 25, 50, 75, 100):
        yy = top + chart_height * (1.0 - value / 100.0)
        lines.append(
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#e6ebef"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#60717f">{value}%</text>'
        )
    for index, row in enumerate(rows):
        x = left + 125 + index * 250
        total = sum(int(row[name]) for name in outcomes)
        cursor = top + chart_height
        for name in outcomes:
            share = int(row[name]) / total
            segment = chart_height * share
            cursor -= segment
            lines.append(
                f'<rect x="{x - 55:.1f}" y="{cursor:.1f}" width="110" height="{segment:.1f}" fill="{colors[name]}"/>'
            )
        lines.extend(
            [
                f'<text x="{x:.1f}" y="{top + chart_height + 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" fill="#334653">{html.escape(str(row["scenario"]).title())} {html.escape(str(row["side"]))}</text>',
                f'<text x="{x:.1f}" y="{top + chart_height + 49}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7882">positive {100 * float(row["positive_ratio"]):.1f}%</text>',
            ]
        )
    legend_x = 115
    for name in outcomes:
        lines.extend(
            [
                f'<rect x="{legend_x}" y="493" width="14" height="14" fill="{colors[name]}"/>',
                f'<text x="{legend_x + 21}" y="505" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#53616d">{html.escape(name.replace("_", " "))}</text>',
            ]
        )
        legend_x += 190
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _research_progress_svg(
    rows: Sequence[Mapping[str, object]], *, round_number: int = 16
) -> str:
    return _progress_svg(rows).replace(
        "Round fifteen produced no evaluation trades.",
        f"Rounds 15 through {round_number} produced no evaluation trades.",
    )


def _gate_summary(
    profile_rows: Sequence[Mapping[str, object]],
    threshold_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    if not profile_rows:
        raise ValueError("adaptive action publication has no profile rows")
    highest = max(profile_rows, key=lambda row: int(row["calibration_eligible_rows"]))
    eligible = [
        row for row in profile_rows if int(row["calibration_eligible_rows"]) > 0
    ]
    empty = [
        str(row["profile"])
        for row in profile_rows
        if int(row["calibration_eligible_rows"]) == 0
    ]
    candidate_count = sum(
        int(row["calibration_threshold_candidates"]) for row in profile_rows
    )
    accepted_count = sum(
        int(bool(row["calibration_threshold_accepted"])) for row in profile_rows
    )
    policy_trades = sum(int(row["policy_trades"]) for row in profile_rows)
    development_evaluated = any(
        bool(row["development_evaluated"]) for row in profile_rows
    )
    candidate_rows = [
        row for row in threshold_rows if bool(row.get("candidate_available"))
    ]
    all_candidate_stress_nets_negative = bool(candidate_rows) and all(
        float(row["stress_total_net_bps"]) < 0.0 for row in candidate_rows
    )
    if eligible:
        eligible_text = ", ".join(
            f"{str(row['profile']).capitalize()} ({int(row['calibration_eligible_rows']):,})"
            for row in eligible
        )
        empty_clause = (
            f"{' and '.join(name.capitalize() for name in empty)} produced none. "
            if empty
            else ""
        )
        candidate_clause = (
            f"The {candidate_count} resulting threshold candidates all failed the "
            "stress-test acceptance criteria"
            if candidate_count
            else "No threshold candidate could be constructed"
        )
        sentence = (
            f"Signals meeting pre-threshold controls appeared only for {eligible_text}; "
            f"{empty_clause}{candidate_clause}, so no out-of-sample simulated trade, development access, "
            "leverage, or trading authority was permitted."
        )
    else:
        sentence = (
            "All three risk profiles had zero signals meeting pre-threshold controls, so no "
            "threshold, out-of-sample simulated trade, development access, leverage, or trading "
            "authority was permitted."
        )
    return {
        "highest_eligible_rows": int(highest["calibration_eligible_rows"]),
        "highest_eligible_profile": (
            str(highest["profile"])
            if int(highest["calibration_eligible_rows"]) > 0
            else "none"
        ),
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "policy_trades": policy_trades,
        "development_evaluated": development_evaluated,
        "all_candidate_stress_nets_negative": all_candidate_stress_nets_negative,
        "sentence": sentence,
    }


def publish(
    evidence_root: Path,
    design_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    raw_design = _read_json(design_path)
    if raw_design.get("schema_version") == "outcome-mixture-screen-design-v1":
        design, design_sha256 = load_outcome_mixture_design(
            design_path, require_current=False
        )
    else:
        design, design_sha256 = load_adaptive_action_design(
            design_path, require_current=False
        )
    report = _validated_evidence(evidence_root, design, design_sha256)
    forecast_rows = _forecast_rows(report, design)
    profile_rows = _profile_rows(report)
    thresholds = _threshold_rows(report)
    gate_summary = _gate_summary(profile_rows, thresholds)
    barrier_rows = _barrier_rows(report)
    progress = _progress_rows(prior_progress_path, report, forecast_rows)
    progress_fields: list[str] = []
    for row in progress:
        for name in row:
            if name not in progress_fields:
                progress_fields.append(name)
    diagnostics: dict[str, object] = {
        "schema_version": "adaptive-action-publication-diagnostics-v1",
        "artifact_class": "exchange_sourced_adaptive_action_evidence_no_authority",
        "design_sha256": design_sha256,
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "forecast_rows": forecast_rows,
        "profile_rows": profile_rows,
        "barrier_rows": barrier_rows,
        "source_report": {
            name: value
            for name, value in report.items()
            if name != "_validated_artifacts"
        },
    }
    diagnostics["diagnostic_sha256"] = _canonical_sha256(diagnostics)
    charts = output_dir / "charts"
    _write_csv(output_dir / "forecast.csv", forecast_rows)
    _write_csv(output_dir / "profiles.csv", profile_rows)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "barrier-outcomes.csv", barrier_rows)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "diagnostics.json",
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(
        charts / "forecast-quality.svg",
        _forecast_svg(forecast_rows, round_number=int(report["round"])),
    )
    _write_text(
        charts / "ranked-tail-economics.svg",
        _tail_svg(forecast_rows, round_number=int(report["round"])),
    )
    _write_text(
        charts / "pre-trade-risk-controls.svg",
        _funnel_svg(profile_rows, round_number=int(report["round"])),
    )
    round_number = int(report["round"])
    _write_text(
        charts / "barrier-outcomes.svg",
        _barrier_svg(barrier_rows, round_number=round_number),
    )
    _write_text(
        charts / "research-progress.svg",
        _research_progress_svg(progress, round_number=round_number),
    )
    calibration_stress = [
        row
        for row in forecast_rows
        if row["role"] == "calibration" and row["scenario"] == "stress"
    ]
    policy_stress = [
        row
        for row in forecast_rows
        if row["role"] == "policy" and row["scenario"] == "stress"
    ]
    best_calibration_auc = max(calibration_stress, key=lambda row: float(row["auc"]))
    best_policy_auc = max(policy_stress, key=lambda row: float(row["auc"]))
    best_policy_tail = max(
        policy_stress, key=lambda row: float(row["top_100_mean_net_bps"])
    )
    displayed_tail_values = [
        float(row[field])
        for row in calibration_stress + policy_stress
        for field in ("top_100_mean_net_bps", "top_500_mean_net_bps")
    ]
    negative_tail_count = sum(value < 0.0 for value in displayed_tail_values)
    if negative_tail_count == len(displayed_tail_values):
        tail_evidence = (
            "every displayed top-100 and top-500 realized mean net return remained "
            "negative"
        )
    else:
        tail_evidence = (
            f'the best out-of-sample top-100 mean was {float(best_policy_tail["top_100_mean_net_bps"]):+.3f} bps, '
            f"but {negative_tail_count} of {len(displayed_tail_values)} displayed top-100/top-500 means were negative and no threshold was accepted"
        )
    execution = design["execution"]
    assert isinstance(execution, Mapping)
    title, summary, next_step = _publication_narrative(
        round_number,
        all_candidate_stress_nets_negative=bool(
            gate_summary["all_candidate_stress_nets_negative"]
        ),
    )
    readme = f"""# Round {round_number}: {title}

**Rejected without trading authority.** {summary} {gate_summary["sentence"]}

| Evidence | Result |
| --- | ---: |
| Best threshold-selection stress ROC AUC | {float(best_calibration_auc["auc"]):.3f} ({best_calibration_auc["side"]}) |
| Best out-of-sample stress ROC AUC | {float(best_policy_auc["auc"]):.3f} ({best_policy_auc["side"]}) |
| Best out-of-sample top-100 mean net return | {float(best_policy_tail["top_100_mean_net_bps"]):+.2f} bps ({best_policy_tail["side"]}) |
| Largest pre-threshold eligible signal set | {int(gate_summary["highest_eligible_rows"]):,} / {int(best_calibration_auc["rows"]):,} ({gate_summary["highest_eligible_profile"]}) |
| Thresholds evaluated / accepted | {int(gate_summary["candidate_count"]):,} / {int(gate_summary["accepted_count"]):,} |
| Out-of-sample simulated trades | {int(gate_summary["policy_trades"]):,} |
| Authorized / live-executed trades | 0 / 0 |

![Forecast quality](charts/forecast-quality.svg)

![Net returns for highest-ranked signals](charts/ranked-tail-economics.svg)

![Signals passing pre-trade risk controls](charts/pre-trade-risk-controls.svg)

![Barrier outcomes](charts/barrier-outcomes.svg)

![Research progress](charts/research-progress.svg)

BTCUSDT, {design["data"]["start_date"]} through {design["data"]["end_date"]} UTC; {int(report["dataset"]["valid_barrier_rows"]):,} valid event labels from {int(report["dataset"]["rows"]):,} exact-BBO rows. The simulation uses {int(execution["horizon_seconds"])} s positions, 100 ms paths, {int(execution["total_latency_ms"])} ms total latency, and {2 * (float(execution["taker_fee_bps_per_side"]) + float(execution["additional_slippage_bps_per_side"])):.0f} bps configured taker round-trip cost.

Probability-of-profit discrimination did not translate into an economically usable net-return ranking: threshold-selection stress ROC AUC reached {float(best_calibration_auc["auc"]):.3f}, and {tail_evidence}. {next_step} The development window and reserved 2023-07-07 terminal day remain untouched.

Data: [forecast.csv](forecast.csv) | [profiles.csv](profiles.csv) | [thresholds.csv](thresholds.csv) | [barrier-outcomes.csv](barrier-outcomes.csv) | [progress.csv](progress.csv) | [diagnostics.json](diagnostics.json) | [integrity report](report.json)
"""
    _write_text(output_dir / "README.md", readme)
    generated = [
        output_dir / "README.md",
        output_dir / "forecast.csv",
        output_dir / "profiles.csv",
        output_dir / "thresholds.csv",
        output_dir / "barrier-outcomes.csv",
        output_dir / "progress.csv",
        output_dir / "diagnostics.json",
        charts / "forecast-quality.svg",
        charts / "ranked-tail-economics.svg",
        charts / "pre-trade-risk-controls.svg",
        charts / "barrier-outcomes.svg",
        charts / "research-progress.svg",
    ]
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact_class": "exchange_sourced_adaptive_action_graph_data",
        "round": round_number,
        "design_revision": int(design["design_revision"]),
        "status": "rejected",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": False,
        "design_sha256": design_sha256,
        "source_report_sha256": _sha256(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "implementation_commit": design["implementation"]["commit"],
        "corpus_certificate_sha256": report["corpus_certificate_sha256"],
        "barrier_targets_sha256": report["dataset"]["barrier_targets_sha256"],
        "diagnostic_sha256": diagnostics["diagnostic_sha256"],
        "actual": {
            "ensemble_models": len(report["ensemble_models"]),
            "valid_barrier_rows": report["dataset"]["valid_barrier_rows"],
            "best_calibration_stress_auc": best_calibration_auc["auc"],
            "best_policy_stress_auc": best_policy_auc["auc"],
            "best_policy_top_100_mean_net_bps": best_policy_tail[
                "top_100_mean_net_bps"
            ],
            "calibration_eligible_rows": gate_summary["highest_eligible_rows"],
            "accepted_thresholds": gate_summary["accepted_count"],
            "policy_trades": gate_summary["policy_trades"],
            "development_evaluated": gate_summary["development_evaluated"],
            "research_candidates": 0,
        },
        "source_artifacts": report["_validated_artifacts"],
        "artifact_integrity": [
            {
                "path": path.relative_to(output_dir).as_posix(),
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
        "adaptive-action-publication: "
        f"status={publication['status']} sha256={publication['publication_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
