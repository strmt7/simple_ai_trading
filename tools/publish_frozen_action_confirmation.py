"""Publish verified Round 31 frozen-confirmation evidence and graph data."""

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
    from tools.run_frozen_action_confirmation import (
        REPORT_SCHEMA_VERSION,
        _STAGE_NAMES,
        load_frozen_confirmation_design,
    )
    from tools.run_gross_architecture_screen import (
        _canonical_sha256,
        _is_sha256,
        _sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - direct tools execution
    from publish_daily_walkforward_screen import _progress_svg
    from run_frozen_action_confirmation import (
        REPORT_SCHEMA_VERSION,
        _STAGE_NAMES,
        load_frozen_confirmation_design,
    )
    from run_gross_architecture_screen import (
        _canonical_sha256,
        _is_sha256,
        _sha256_file,
    )


PUBLICATION_SCHEMA_VERSION = "frozen-action-confirmation-publication-v1"
_FALSE_CLAIMS = (
    "trading_authority",
    "execution_claim",
    "profitability_claim",
    "portfolio_claim",
    "leverage_applied",
)


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _require_false_claims(value: Mapping[str, object], *, label: str) -> None:
    if any(value.get(name) is not False for name in _FALSE_CLAIMS):
        raise ValueError(f"frozen confirmation {label} authority drifted")


def _validated_report(
    evidence_root: Path,
    design: Mapping[str, object],
    design_sha256: str,
) -> dict[str, object]:
    report_path = evidence_root / "report.json"
    report = _read_json(report_path, label="frozen confirmation report")
    claimed = report.get("report_sha256")
    canonical = dict(report)
    canonical.pop("report_sha256", None)
    if (
        not _is_sha256(claimed)
        or _canonical_sha256(canonical) != claimed
        or report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 31
        or report.get("design_sha256") != design_sha256
        or report.get("status") not in {"rejected", "research_candidate"}
        or report.get("terminal_holdout_accessed") is not False
    ):
        raise ValueError("frozen confirmation report identity is invalid")
    _require_false_claims(report, label="report")
    access = report.get("stage_access")
    stages = report.get("stages")
    artifacts = report.get("stage_artifacts")
    if (
        not isinstance(access, Mapping)
        or set(access) != set(_STAGE_NAMES)
        or not isinstance(stages, Mapping)
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError("frozen confirmation stage evidence is incomplete")
    expected_profiles = tuple(
        str(profile["profile"]) for profile in design["risk_profiles"]
    )
    prior_survivors = expected_profiles
    should_open = True
    for name in _STAGE_NAMES:
        value = access[name]
        if not isinstance(value, bool) or value is not should_open:
            raise ValueError("frozen confirmation stage access is non-nested")
        if value:
            stage = stages.get(name)
            artifact = artifacts.get(name)
            if not isinstance(stage, Mapping) or not isinstance(artifact, Mapping):
                raise ValueError("frozen confirmation opened stage is missing")
            relative = Path(str(artifact.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("frozen confirmation stage artifact path is unsafe")
            path = evidence_root / relative
            if (
                not path.is_file()
                or _sha256_file(path) != artifact.get("sha256")
                or path.stat().st_size != int(artifact.get("bytes") or -1)
                or _read_json(path, label=f"{name} stage artifact") != stage
                or stage.get("stage") != name
            ):
                raise ValueError("frozen confirmation stage artifact differs")
            _require_false_claims(stage, label=f"{name} stage")
            diagnostics = stage.get("forecast_diagnostics")
            results = stage.get("profile_results")
            survivors = stage.get("surviving_profiles")
            if (
                not isinstance(diagnostics, Mapping)
                or set(diagnostics) != {"base", "stress"}
                or not isinstance(results, list)
                or not isinstance(survivors, list)
            ):
                raise ValueError("frozen confirmation stage sections are incomplete")
            for scenario in ("base", "stress"):
                diagnostic = diagnostics[scenario]
                if not isinstance(diagnostic, Mapping):
                    raise ValueError("frozen confirmation forecast evidence is invalid")
                _require_false_claims(
                    diagnostic,
                    label=f"{name} {scenario} forecast",
                )
            result_profiles = tuple(
                result.get("profile") if isinstance(result, Mapping) else None
                for result in results
            )
            if result_profiles != prior_survivors:
                raise ValueError("frozen confirmation evaluated profiles drifted")
            passed_profiles: list[str] = []
            for result in results:
                assert isinstance(result, Mapping)
                _require_false_claims(result, label=f"{name} profile")
                candidates = result.get("candidates")
                expected_count = 4 if name == "confirmation" else 1
                if not isinstance(candidates, list) or len(candidates) != expected_count:
                    raise ValueError("frozen confirmation candidate count drifted")
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        raise ValueError("frozen confirmation candidate is invalid")
                    _require_false_claims(candidate, label=f"{name} candidate")
                    reasons = candidate.get("gate_reasons")
                    if (
                        not isinstance(reasons, list)
                        or bool(candidate.get("passed")) == bool(reasons)
                    ):
                        raise ValueError("frozen confirmation candidate gate drifted")
                passed = bool(result.get("passed"))
                selected_quantile = result.get("selected_quantile")
                selected_threshold = result.get("selected_threshold_bps")
                if passed:
                    if selected_quantile is None or selected_threshold is None:
                        raise ValueError("frozen confirmation selection is incomplete")
                    passed_profiles.append(str(result["profile"]))
                elif selected_quantile is not None or selected_threshold is not None:
                    raise ValueError("frozen confirmation rejected selection drifted")
            if survivors != passed_profiles:
                raise ValueError("frozen confirmation survivors drifted")
            prior_survivors = tuple(passed_profiles)
            should_open = bool(prior_survivors)
        elif name in stages or name in artifacts:
            raise ValueError("frozen confirmation withheld stage has evidence")
    if bool(report.get("policy_window_is_consumed")) != bool(access["policy"]):
        raise ValueError("frozen confirmation policy consumption differs")
    if bool(report.get("development_window_is_consumed")) != bool(
        access["development"]
    ):
        raise ValueError("frozen confirmation development consumption differs")
    governance = report.get("consumed_period_governance")
    if (
        not isinstance(governance, Mapping)
        or governance.get("all_consumed_dates_excluded_from_targets") is not True
        or governance.get("excluded_target_dates")
        != design["data"]["excluded_target_dates"]
    ):
        raise ValueError("frozen confirmation consumed-period evidence differs")
    expected_final = list(prior_survivors) if access["development"] else []
    if (
        report.get("final_profiles") != expected_final
        or (report.get("status") == "research_candidate") != bool(expected_final)
    ):
        raise ValueError("frozen confirmation final profile status drifted")
    return report


def _stage_rows(
    design: Mapping[str, object], report: Mapping[str, object]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in _STAGE_NAMES:
        contract = design["data"]["stages"][name]
        opened = bool(report["stage_access"][name])
        stage = report["stages"].get(name) if opened else None
        rows.append(
            {
                "stage": name,
                "opened": opened,
                "context_start": contract["context_start"],
                "evaluation_start": contract["evaluation_start"],
                "evaluation_end": contract["evaluation_end"],
                "next_unopened_date": contract["next_unopened_date"],
                "valid_target_rows": (
                    int(stage["data"]["valid_target_rows"]) if opened else 0
                ),
                "surviving_profiles": (
                    ";".join(str(value) for value in stage["surviving_profiles"])
                    if opened
                    else ""
                ),
                "survivor_count": (
                    len(stage["surviving_profiles"]) if opened else 0
                ),
                "withheld_reason": (
                    "" if opened else "prior_stage_rejected"
                ),
                "terminal_holdout_accessed": False,
            }
        )
    return rows


def _candidate_rows(
    design: Mapping[str, object], report: Mapping[str, object]
) -> list[dict[str, object]]:
    profiles = {
        str(item["profile"]): item for item in design["risk_profiles"]
    }
    gate_name = {
        "confirmation": "calibration_gates",
        "policy": "policy_gates",
        "development": "development_gates",
    }
    rows: list[dict[str, object]] = []
    for stage_name in _STAGE_NAMES:
        stage = report["stages"].get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        start = stage["data"]["evaluation_start"]
        end = stage["data"]["evaluation_end"]
        for result in stage["profile_results"]:
            profile = str(result["profile"])
            required = int(profiles[profile][gate_name[stage_name]]["minimum_trades"])
            for candidate in result["candidates"]:
                base = candidate["base_trace"]["metrics"]
                stress = candidate["stress_trace"]["metrics"]
                rows.append(
                    {
                        "stage": stage_name,
                        "evaluation_start": start,
                        "evaluation_end": end,
                        "profile": profile,
                        "quantile": float(candidate["quantile"]),
                        "threshold_bps": float(candidate["threshold_bps"]),
                        "eligible_rows": int(result["eligible_rows"]),
                        "passed": bool(candidate["passed"]),
                        "selected": (
                            bool(result["passed"])
                            and float(result["selected_threshold_bps"])
                            == float(candidate["threshold_bps"])
                        ),
                        "gate_reasons": ";".join(candidate["gate_reasons"]),
                        "required_minimum_trades": required,
                        "base_trades": int(base["trades"]),
                        "base_total_net_bps": float(base["total_net_bps"]),
                        "stress_trades": int(stress["trades"]),
                        "stress_total_net_bps": float(stress["total_net_bps"]),
                        "stress_mean_net_bps": float(stress["mean_net_bps"]),
                        "stress_max_drawdown_bps": float(
                            stress["max_drawdown_bps"]
                        ),
                        "stress_profit_factor": stress["profit_factor"],
                        "stress_win_rate": float(stress["win_rate"]),
                        "stress_worst_trade_bps": float(stress["worst_trade_bps"]),
                        "drawdown_adjusted_utility_bps": float(
                            candidate["drawdown_adjusted_utility_bps"]
                        ),
                    }
                )
    return rows


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage_name in _STAGE_NAMES:
        stage = report["stages"].get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        for scenario in ("base", "stress"):
            diagnostic = stage["forecast_diagnostics"][scenario]
            for side in ("long", "short"):
                values = diagnostic["sides"][side]
                top = {int(item["requested_rows"]): item for item in values["top_rows"]}
                rows.append(
                    {
                        "stage": stage_name,
                        "evaluation_start": stage["data"]["evaluation_start"],
                        "evaluation_end": stage["data"]["evaluation_end"],
                        "scenario": scenario,
                        "side": side,
                        "rows": int(values["rows"]),
                        "profitable_auc": float(values["profitable_auc"]),
                        "profitable_brier": float(values["profitable_brier"]),
                        "prevalence_brier": float(values["prevalence_brier"]),
                        "information_coefficient": float(
                            values["pearson_information_coefficient"]
                        ),
                        "mean_actual_net_bps": float(values["mean_actual_net_bps"]),
                        "top_100_mean_net_bps": float(
                            top[100]["mean_actual_net_bps"]
                        ),
                        "top_500_mean_net_bps": float(
                            top[500]["mean_actual_net_bps"]
                        ),
                    }
                )
    return rows


def _read_progress(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    if not rows or not fields or int(rows[-1]["round"]) != 30:
        raise ValueError("Round 31 prior progress lineage is invalid")
    return rows, fields


def _progress_rows(
    path: Path,
    report: Mapping[str, object],
    forecast_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    rows, fields = _read_progress(path)
    deepest = next(
        name for name in reversed(_STAGE_NAMES) if report["stage_access"][name]
    )
    stress = [
        row
        for row in forecast_rows
        if row["stage"] == deepest and row["scenario"] == "stress"
    ]
    best_top_500 = max(float(row["top_500_mean_net_bps"]) for row in stress)
    policy_candidates = [
        row for row in candidate_rows if row["stage"] == "policy"
    ]
    best_policy_candidate = (
        max(policy_candidates, key=lambda row: float(row["stress_total_net_bps"]))
        if policy_candidates
        else None
    )
    source = report["stage_artifacts"][deepest]["path"]
    update: dict[str, object] = {name: "" for name in fields}
    update.update(
        {
            "round": 31,
            "stage": "frozen Round 30 chronological confirmation",
            "periods": "2024-01-01..2024-03-29",
            "selection_contaminated": False,
            "horizon_seconds": 900,
            "feature_set": "l1-tape-causal-v8",
            "risk_level": "conservative;regular;aggressive",
            "selected_signals": 0,
            "executable_trades": 0,
            "status": report["status"],
            "source_file": source,
            "best_model_id": "frozen Round 30 LightGBM hurdle ensemble",
            "best_top_500_exact_after_cost_bps": best_top_500,
            "best_policy_trades": (
                int(best_policy_candidate["stress_trades"])
                if best_policy_candidate is not None
                else ""
            ),
            "best_policy_total_net_bps": (
                float(best_policy_candidate["stress_total_net_bps"])
                if best_policy_candidate is not None
                else ""
            ),
            "best_policy_mean_net_bps": (
                float(best_policy_candidate["stress_mean_net_bps"])
                if best_policy_candidate is not None
                else ""
            ),
            "best_policy_max_drawdown_bps": (
                float(best_policy_candidate["stress_max_drawdown_bps"])
                if best_policy_candidate is not None
                else ""
            ),
            "best_policy_profit_factor": (
                best_policy_candidate["stress_profit_factor"]
                if best_policy_candidate is not None
                else ""
            ),
            "ensemble_models": 3,
            "calibration_threshold_traces": 12,
            "accepted_thresholds": len(report["final_profiles"]),
            "development_consumed": bool(
                report["development_window_is_consumed"]
            ),
        }
    )
    return [*rows, update], fields


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _stage_access_svg(
    rows: Sequence[Mapping[str, object]], *, terminal_date: str
) -> str:
    width, height = 1260, 450
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 31 nested chronological access</title>',
        '<desc id="desc">Confirmation, policy, and development windows open only after the prior stage passes. The terminal date remains sealed.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="50" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Later dates stayed behind binding evidence gates</text>',
        '<text x="48" y="82" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53616d">Green means opened; red means withheld. Raw archive integrity checks do not open targets, predictions, or financial metrics.</text>',
    ]
    for index, row in enumerate(rows):
        x = 70 + index * 390
        opened = bool(row["opened"])
        fill = "#e8f5f2" if opened else "#fff0ee"
        stroke = "#16827a" if opened else "#c64c3f"
        status = "opened" if opened else "withheld"
        survivors = (
            f"{int(row['survivor_count'])} profiles survived"
            if opened
            else "prior stage rejected"
        )
        lines.extend(
            [
                f'<rect x="{x}" y="135" width="340" height="190" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
                f'<text x="{x + 24}" y="176" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="700" fill="#22333d">{html.escape(str(row["stage"]).title())}</text>',
                f'<text x="{x + 24}" y="207" font-family="Segoe UI, Arial, sans-serif" font-size="14" font-weight="700" fill="{stroke}">{status}</text>',
                f'<text x="{x + 24}" y="240" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#40525e">{html.escape(str(row["evaluation_start"]))} to {html.escape(str(row["evaluation_end"]))} UTC</text>',
                f'<text x="{x + 24}" y="270" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#40525e">{html.escape(survivors)}</text>',
                f'<text x="{x + 24}" y="299" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#64727d">{int(row["valid_target_rows"]):,} valid target rows</text>',
            ]
        )
        if index < len(rows) - 1:
            lines.append(
                f'<path d="M {x + 340} 230 L {x + 376} 230" stroke="#8a98a3" stroke-width="3" marker-end="url(#arrow)"/>'
            )
    lines.insert(
        4,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#8a98a3"/></marker></defs>',
    )
    lines.extend(
        [
            f'<text x="48" y="395" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#52606d">Reserved terminal: {html.escape(terminal_date)} UTC; not queried, labeled, predicted, or evaluated.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _candidate_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1800, 700
    left, top, chart_width, chart_height = 95, 150, 1620, 330
    if not rows:
        raise ValueError("candidate chart requires opened candidates")
    values = [float(row["stress_total_net_bps"]) for row in rows]
    minimum, maximum = min(0.0, min(values)), max(0.0, max(values))
    span = max(1.0, maximum - minimum)
    padding = max(10.0, span * 0.1)
    lower, upper = minimum - padding, maximum + padding

    def y(value: float) -> float:
        return top + chart_height * (upper - value) / (upper - lower)

    periods = "; ".join(
        f"{name.title()} {next(row['evaluation_start'] for row in rows if row['stage'] == name)} to {next(row['evaluation_end'] for row in rows if row['stage'] == name)}"
        for name in _STAGE_NAMES
        if any(row["stage"] == name for row in rows)
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 31 after-cost candidate economics</title>',
        '<desc id="desc">Adverse-stress total net basis points for every opened frozen threshold, with trades versus required support and maximum drawdown.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="50" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Frozen thresholds faced fresh after-cost evidence</text>',
        f'<text x="48" y="82" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#53616d">{html.escape(periods)} UTC.</text>',
        '<text x="48" y="106" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#64727d">Labels show simulated trades / required minimum and maximum drawdown; no result carries trading authority.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in np_ticks(lower, upper, 6):
        yy = y(value)
        lines.extend(
            [
                f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#{"536674" if abs(value) < 1e-9 else "e6ebef"}" stroke-width="{2 if abs(value) < 1e-9 else 1}"/>',
                f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#60717f">{value:+.0f}</text>',
            ]
        )
    slot = chart_width / len(rows)
    zero = y(0.0)
    for index, row in enumerate(rows):
        value = float(row["stress_total_net_bps"])
        xx = left + slot * (index + 0.5)
        yy = y(value)
        color = "#16827a" if bool(row["passed"]) else "#c64c3f"
        label = (
            f"{str(row['stage'])[0].upper()}-{str(row['profile'])[0].upper()}"
            f" q{int(round(100 * float(row['quantile'])))}"
        )
        lines.extend(
            [
                f'<rect x="{xx - min(25, slot * 0.32):.1f}" y="{min(yy, zero):.1f}" width="{2 * min(25, slot * 0.32):.1f}" height="{max(1.0, abs(yy - zero)):.1f}" fill="{color}"/>',
                f'<text x="{xx:.1f}" y="{yy - 8 if value >= 0 else yy + 17:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" font-weight="700" fill="#263744">{value:+.1f}</text>',
                f'<text x="{xx:.1f}" y="510" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" font-weight="700" fill="#334653">{html.escape(label)}</text>',
                f'<text x="{xx:.1f}" y="530" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="#53616d">{int(row["stress_trades"])}/{int(row["required_minimum_trades"])} trades</text>',
                f'<text x="{xx:.1f}" y="548" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="#6b7882">DD {float(row["stress_max_drawdown_bps"]):.1f}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="34" y="315" transform="rotate(-90 34 315)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">Total net basis points</text>',
            '<rect x="48" y="620" width="14" height="14" fill="#16827a"/><text x="70" y="632" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#53616d">passed stage gates</text>',
            '<rect x="230" y="620" width="14" height="14" fill="#c64c3f"/><text x="252" y="632" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#53616d">rejected</text>',
            '<text x="48" y="668" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">C/P/D denote confirmation, policy, and development; C/R/A denote Conservative, Regular, and Aggressive.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def np_ticks(lower: float, upper: float, target: int) -> tuple[float, ...]:
    span = max(1.0, upper - lower)
    rough = span / max(1, target - 1)
    magnitude = 10.0 ** math.floor(math.log10(rough))
    step = min((1.0, 2.0, 5.0, 10.0), key=lambda value: abs(value * magnitude - rough)) * magnitude
    first = math.ceil(lower / step) * step
    return tuple(
        first + index * step
        for index in range(int(math.floor((upper - first) / step)) + 1)
    )


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["scenario"] == "stress"]
    width, height = 1300, 560
    left, top, chart_width, chart_height = 90, 140, 1140, 280
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Round 31 probability-of-profit discrimination</title>',
        '<desc id="desc">Adverse-stress ROC AUC by opened stage and side, with exact UTC windows and a chance reference at 0.5.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="48" y="50" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#17212b">Forecast discrimination across opened stages</text>',
        '<text x="48" y="82" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#53616d">ROC AUC is diagnostic evidence, not an after-cost profitability result.</text>',
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>',
    ]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = top + chart_height * (1.0 - value)
        lines.extend(
            [
                f'<line x1="{left}" y1="{yy:.1f}" x2="{left + chart_width}" y2="{yy:.1f}" stroke="#{"687985" if value == 0.5 else "e6ebef"}" stroke-width="{2 if value == 0.5 else 1}" stroke-dasharray="{"7 6" if value == 0.5 else "none"}"/>',
                f'<text x="{left - 12}" y="{yy + 5:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#60717f">{value:.2f}</text>',
            ]
        )
    slot = chart_width / len(selected)
    for index, row in enumerate(selected):
        value = float(row["profitable_auc"])
        xx = left + slot * (index + 0.5)
        yy = top + chart_height * (1.0 - value)
        color = "#16827a" if row["side"] == "long" else "#8d5aa7"
        label = f"{str(row['stage']).title()} {row['side']}"
        lines.extend(
            [
                f'<rect x="{xx - min(42, slot * 0.3):.1f}" y="{yy:.1f}" width="{2 * min(42, slot * 0.3):.1f}" height="{top + chart_height - yy:.1f}" fill="{color}"/>',
                f'<text x="{xx:.1f}" y="{yy - 9:.1f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="12" font-weight="700" fill="#263744">{value:.3f}</text>',
                f'<text x="{xx:.1f}" y="449" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#334653">{html.escape(label)}</text>',
                f'<text x="{xx:.1f}" y="468" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="#6b7882">{html.escape(str(row["evaluation_start"]))} to {html.escape(str(row["evaluation_end"]))}</text>',
            ]
        )
    lines.extend(
        [
            '<text x="34" y="280" transform="rotate(-90 34 280)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#51606d">ROC AUC</text>',
            '<text x="48" y="530" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#65727d">All dates are UTC. Brier score and ranked-tail data remain available in forecast.csv.</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _artifact(path: Path, root: Path) -> dict[str, object]:
    item = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            item["row_count"] = sum(1 for _ in csv.DictReader(handle))
    return item


def publish(
    *,
    evidence_root: Path,
    design_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    design, design_sha256 = load_frozen_confirmation_design(design_path)
    report = _validated_report(evidence_root, design, design_sha256)
    stages = _stage_rows(design, report)
    candidates = _candidate_rows(design, report)
    forecasts = _forecast_rows(report)
    progress, progress_fields = _progress_rows(
        prior_progress_path, report, forecasts, candidates
    )
    charts = output_dir / "charts"
    _write_csv(output_dir / "stages.csv", stages)
    _write_csv(output_dir / "candidates.csv", candidates)
    _write_csv(output_dir / "forecast.csv", forecasts)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        charts / "stage-access.svg",
        _stage_access_svg(
            stages, terminal_date=str(design["reserved_terminal"]["date"])
        ),
    )
    _write_text(charts / "candidate-economics.svg", _candidate_svg(candidates))
    _write_text(charts / "forecast-quality.svg", _forecast_svg(forecasts))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    deepest = next(name for name in reversed(_STAGE_NAMES) if report["stage_access"][name])
    deepest_rows = [row for row in candidates if row["stage"] == deepest]
    passing = [row for row in deepest_rows if row["passed"]]
    best = (
        max(deepest_rows, key=lambda row: float(row["stress_total_net_bps"]))
        if deepest_rows
        else None
    )
    status_text = (
        "research candidate without trading authority"
        if report["status"] == "research_candidate"
        else "rejected without trading authority"
    )
    withheld = [row["stage"] for row in stages if not row["opened"]]
    withheld_text = ", ".join(withheld) if withheld else "none"
    readme = f"""# Round 31: frozen chronological confirmation {report['status']}

**{status_text.capitalize()}.** The exact Round 30 models and twelve thresholds were evaluated without retraining or recalibration. The deepest opened stage was **{deepest}**; withheld stages: **{withheld_text}**. {len(passing)} candidate(s) passed that stage. No leverage or trading authority was permitted.

| Evidence | Result |
| --- | ---: |
| Exact-BBO archive availability | 2023-05-16 to 2024-03-30 UTC (320 gap-free days) |
| Confirmation window | 2024-01-01 to 2024-02-04 UTC |
| Policy window | 2024-02-06 to 2024-03-05 UTC |
| Development window | 2024-03-06 to 2024-03-29 UTC; 2024-03-15 excluded |
| Deepest opened stage | {deepest} |
| Candidates in deepest stage / passed | {len(deepest_rows)} / {len(passing)} |
| Best deepest-stage stress net return | {float(best['stress_total_net_bps']):+.2f} bps from {int(best['stress_trades'])} trades |
| Best deepest-stage maximum drawdown | {float(best['stress_max_drawdown_bps']):.2f} bps |
| Final research profiles | {', '.join(report['final_profiles']) or 'none'} |
| Authorized / live-executed trades | 0 / 0 |

![Nested stage access](charts/stage-access.svg)

![After-cost candidate economics](charts/candidate-economics.svg)

![Forecast quality](charts/forecast-quality.svg)

![Research progress](charts/research-progress.svg)

The terminal date, **2024-03-30**, was not ingested, queried, labeled, predicted, or evaluated. Dates already consumed by earlier rounds were excluded from targets. Official archive ingestion and deterministic causal-feature materialization occurred before model evaluation, but later-stage target, prediction, and metric construction remained gated.

This is single-symbol BTCUSDT research evidence. It cannot satisfy portfolio-diversification requirements and is not a profitability, execution, leverage, or deployment claim. Binance publishes years of trade data, but its public exact `bookTicker` history begins on 2023-05-16; the 320-day BBO limit is reported rather than extrapolated.

Data: [stages.csv](stages.csv) | [candidates.csv](candidates.csv) | [forecast.csv](forecast.csv) | [progress.csv](progress.csv) | [integrity report](report.json)
"""
    _write_text(output_dir / "README.md", readme)
    generated = [
        output_dir / "README.md",
        output_dir / "stages.csv",
        output_dir / "candidates.csv",
        output_dir / "forecast.csv",
        output_dir / "progress.csv",
        charts / "stage-access.svg",
        charts / "candidate-economics.svg",
        charts / "forecast-quality.svg",
        charts / "research-progress.svg",
    ]
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact_class": "frozen_chronological_confirmation_graph_data",
        "round": 31,
        "status": report["status"],
        "design_sha256": design_sha256,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "source_report_canonical_sha256": report["report_sha256"],
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "stage_access": report["stage_access"],
        "final_profiles": report["final_profiles"],
        "deepest_opened_stage": deepest,
        "deepest_stage_candidate_count": len(deepest_rows),
        "deepest_stage_passing_count": len(passing),
        "artifact_integrity": [_artifact(path, output_dir) for path in generated],
    }
    publication["publication_sha256"] = _canonical_sha256(publication)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish verified frozen chronological confirmation evidence"
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--prior-progress", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    publication = publish(
        evidence_root=args.evidence_root,
        design_path=args.design,
        prior_progress_path=args.prior_progress,
        output_dir=args.output_dir,
    )
    print(
        f"frozen-confirmation-publication: status={publication['status']} "
        f"sha256={publication['publication_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
