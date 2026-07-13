"""Publish hash-bound Round 54 model and controller failure evidence."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping, Sequence


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


ROUND = 54
REPORT_SCHEMA = "round-054-sequential-action-value-prototype-report-v1"
DIAGNOSTIC_SCHEMA = "round-054-sequential-failure-diagnostic-v1"
PUBLICATION_SCHEMA = "round-054-action-value-publication-v1"
DESIGN_SHA256 = "bc6bc8ea8e4fb8f2d44234c5a987201530786754e2e821e804e7fdb1b3a8a5c4"
DESIGN_FILE_SHA256 = "fbb92bf3cae13144f61f7db5488af72ea04e78aeacf0def47246ff4b870e6e3a"
REPORT_CANONICAL_SHA256 = "8aff454b63201ccab8a53d7a9cf4a9f46bad04a284074c7e7a22e001fb9385ff"
REPORT_FILE_SHA256 = "e1a6e53d4be1d21ef101f7172083740af5ad332ec8933ca445625b32f5085642"
DIAGNOSTIC_CANONICAL_SHA256 = "dd392f0fa1278860424bc27e5764ad84ac943c98ad992f4ec707ff6a94d7369f"
DIAGNOSTIC_FILE_SHA256 = "c2de85558288bbd9befa9e4cc7ecd5966b9d7db8e281bfd4090562ca445b8bec"
DIAGNOSTIC_IMPLEMENTATION_COMMIT = "3b2500c7c44532bec51e374c6577b75773a0a54a"
DIAGNOSTIC_IMPLEMENTATION_BLOB = "85a825f2695b601e63470ea9d4724fee708c70b9"
POLICIES = (
    "median_q_all_seed_consensus",
    "lower_tail_q_all_seed_consensus",
)
DISPLAY = {
    POLICIES[0]: "Median Q",
    POLICIES[1]: "Lower-tail Q",
}


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
    claimed = canonical.pop(digest_key, None)
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError(f"{digest_key} does not match canonical content")
    return actual


def _verify_artifact(row: Mapping[str, object]) -> None:
    path = Path(str(row.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or _file_sha256(path) != str(row.get("sha256", ""))
    ):
        raise ValueError(f"Round 54 artifact drifted: {path}")


def _validate_sources(
    report_path: Path,
    diagnostic_path: Path,
    design_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if _file_sha256(report_path) != REPORT_FILE_SHA256:
        raise ValueError("Round 54 report file hash drifted")
    if _file_sha256(diagnostic_path) != DIAGNOSTIC_FILE_SHA256:
        raise ValueError("Round 54 diagnostic file hash drifted")
    if _file_sha256(design_path) != DESIGN_FILE_SHA256:
        raise ValueError("Round 54 design file hash drifted")
    report = _read_object(report_path, "Round 54 report")
    diagnostic = _read_object(diagnostic_path, "Round 54 diagnostic")
    design = _read_object(design_path, "Round 54 design")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or _canonical_value(report, "report_canonical_sha256")
        != REPORT_CANONICAL_SHA256
        or diagnostic.get("schema_version") != DIAGNOSTIC_SCHEMA
        or diagnostic.get("round") != ROUND
        or _canonical_value(diagnostic, "diagnostic_canonical_sha256")
        != DIAGNOSTIC_CANONICAL_SHA256
        or _canonical_value(design, "design_sha256") != DESIGN_SHA256
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("status") != "rejected_before_evaluation"
        or report.get("evaluation_replay_performed") is not False
        or diagnostic.get("status") != "rejected"
        or diagnostic.get("evaluation_role_read") is not False
        or diagnostic.get("source_report_file_sha256") != REPORT_FILE_SHA256
        or diagnostic.get("source_report_canonical_sha256")
        != REPORT_CANONICAL_SHA256
        or diagnostic.get("implementation_commit")
        != DIAGNOSTIC_IMPLEMENTATION_COMMIT
        or any(
            value is not False
            for value in (
                report.get("profitability_claim"),
                report.get("ai_uplift_claim"),
                report.get("trading_authority"),
                report.get("leverage_applied"),
                diagnostic.get("profitability_claim"),
                diagnostic.get("trading_authority"),
            )
        )
        or _git(
            "rev-parse",
            f"{DIAGNOSTIC_IMPLEMENTATION_COMMIT}:tools/diagnose_round54_sequential_failure.py",
        )
        != DIAGNOSTIC_IMPLEMENTATION_BLOB
    ):
        raise ValueError("Round 54 source contracts or claims drifted")
    _validate_finite(report)
    _validate_finite(diagnostic)
    artifacts = report.get("evidence_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 10:
        raise ValueError("Round 54 evidence manifest is incomplete")
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise ValueError("Round 54 evidence manifest row is invalid")
        _verify_artifact(row)
    return report, diagnostic, design


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"round": ROUND, **dict(row)} for row in report["models"]]


def _direction_rows(diagnostic: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"round": ROUND, **dict(row)} for row in diagnostic["directional_rank"]]


def _policy_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy_id in POLICIES:
        policy = report["calibration"][policy_id]
        row: dict[str, object] = {
            "round": ROUND,
            "policy_id": policy_id,
            "minimum_pairwise_seed_score_spearman": policy[
                "minimum_pairwise_seed_score_spearman"
            ],
            "position_changes": policy["action_diagnostics"]["position_changes"],
            "unanimous_fraction": policy["action_diagnostics"]["unanimous_fraction"],
            "gate_passed": policy["gate"]["passed"],
            "gate_reasons": policy["gate"]["reasons"],
        }
        for scenario in ("base", "stress"):
            metrics = policy[scenario]
            for key in (
                "total_net_return_fraction",
                "maximum_drawdown_fraction",
                "profit_factor",
                "closed_trades",
                "active_days",
                "mean_hourly_portfolio_bps",
                "maximum_single_symbol_fraction_of_absolute_net_pnl",
                "symbol_net_bps",
            ):
                row[f"{scenario}_{key}"] = metrics[key]
            row[f"{scenario}_bootstrap_lower_mean_hourly_bps"] = metrics[
                "bootstrap_mean_hourly_portfolio_bps"
            ]["lower_bps"]
        rows.append(row)
    return rows


def _holding_rows(diagnostic: Mapping[str, object]) -> list[dict[str, object]]:
    return [{"round": ROUND, **dict(row)} for row in diagnostic["position_runs"]]


def _holding_summary_rows(
    diagnostic: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        {"round": ROUND, **dict(row)}
        for row in diagnostic["original_policy_summary"]
    ]


def _finite_hold_rows(diagnostic: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {"round": ROUND, "selection_contaminated": True, **dict(row)}
        for row in diagnostic["finite_holding_screen"]
    ]


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for policy_id in POLICIES:
        gate = report["calibration"][policy_id]["gate"]
        rows.append(
            {
                "round": ROUND,
                "candidate": policy_id,
                "gate": "calibration_policy",
                "passed": gate["passed"],
                "reasons": gate["reasons"],
            }
        )
    rows.append(
        {
            "round": ROUND,
            "candidate": "all",
            "gate": "evaluation_authorized",
            "passed": False,
            "reasons": ["no_calibration_policy_passed"],
        }
    )
    return rows


def _progress_rows(
    previous_path: Path,
    diagnostic: Mapping[str, object],
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
        raise ValueError("research progress must contain exactly Rounds 1 through 53")
    finite = max(
        diagnostic["finite_holding_screen"],
        key=lambda row: float(row["stress_total_net_return_fraction"]),
    )
    return_fraction = float(finite["stress_total_net_return_fraction"])
    closed_trades = int(finite["closed_trades"])
    approximate_net_bps_per_trade = (
        10_000.0 * math.log1p(return_fraction) / closed_trades
        if closed_trades and return_fraction > -1.0
        else ""
    )
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "sequential distributional action-value prototype",
            "periods": "train 2022-01-01..2024-03-31; calibration 2024-07-01..2024-09-30; evaluation sealed",
            "selection_contaminated": "True",
            "horizon_seconds": "3600",
            "feature_set": "71 causal hourly features; 3 assets; 5 quantiles",
            "risk_level": "development only; unlevered",
            "spearman_ic": str(
                max(
                    float(item["directional_spearman"])
                    for item in diagnostic["directional_rank"]
                    if item["symbol"] == "ALL"
                )
            ),
            "selected_signals": "0",
            "executable_trades": "0",
            "status": "rejected",
            "source_file": "verified Round 54 report and sealed calibration-only failure diagnostic",
            "best_policy_trades": str(closed_trades),
            "best_policy_total_net_bps": str(10_000.0 * return_fraction),
            "best_policy_mean_net_bps": str(approximate_net_bps_per_trade),
            "best_policy_max_drawdown_bps": str(
                10_000.0 * float(finite["stress_maximum_drawdown_fraction"])
            ),
            "best_policy_profit_factor": str(finite["stress_profit_factor"]),
            "best_model_id": "posthoc_8h_cap_diagnostic_not_selected",
            "ensemble_models": "3",
            "calibration_eligible_rows": "6552",
            "policy_eligible_rows": str(closed_trades),
            "development_consumed": "True",
            "architecture_gates_passed": "0",
            "architecture_gate_count": "2",
        }
    )
    rows.append(row)
    return rows, fields


def _model_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Bellman residual skill was positive but insufficient",
        subtitle="Early-stop TD loss improvement over the matched zero-Q Bellman baseline",
        groups=tuple(
            (
                f"Seed {row['seed']}",
                (("TD residual skill", 100.0 * float(row["early_stop_td_skill"]), COLORS["teal"]),),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=30.0,
        y_label="TD residual skill (%)",
    )


def _direction_svg(rows: Sequence[Mapping[str, object]]) -> str:
    pooled = [row for row in rows if row["symbol"] == "ALL"]
    groups = []
    for horizon in (1, 4, 12, 24):
        values = {
            str(row["policy_id"]): 100.0 * float(row["directional_spearman"])
            for row in pooled
            if int(row["horizon_hours"]) == horizon
        }
        groups.append(
            (
                f"{horizon}h",
                (
                    ("Median Q", values[POLICIES[0]], COLORS["blue"]),
                    ("Lower-tail Q", values[POLICIES[1]], COLORS["teal"]),
                ),
            )
        )
    return _bar_svg(
        title="Weak directional rank survived the failed controller",
        subtitle="Pooled calibration-only Spearman rank; evaluation remained sealed",
        groups=groups,
        y_min=0.0,
        y_max=5.5,
        y_label="Directional Spearman x100",
    )


def _policy_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="Unbounded sequential policies failed risk and economics",
        subtitle="Calibration stress replay with 8 bps one-way cost; no leverage",
        groups=tuple(
            (
                DISPLAY[str(row["policy_id"])],
                (
                    (
                        "Net return",
                        100.0 * float(row["stress_total_net_return_fraction"]),
                        COLORS["red"],
                    ),
                    (
                        "Maximum drawdown",
                        100.0 * float(row["stress_maximum_drawdown_fraction"]),
                        COLORS["amber"],
                    ),
                ),
            )
            for row in rows
        ),
        y_min=-35.0,
        y_max=50.0,
        y_label="Percent of unlevered capital",
    )


def _holding_svg(rows: Sequence[Mapping[str, object]]) -> str:
    return _bar_svg(
        title="The learned controller created stale positions",
        subtitle="Consecutive non-flat calibration hours; day-trading safety requires a hard bound",
        groups=tuple(
            (
                DISPLAY[str(row["policy_id"])],
                (
                    ("Maximum hold", float(row["maximum_holding_hours"]), COLORS["red"]),
                    ("Median hold", float(row["median_holding_hours"]), COLORS["amber"]),
                ),
            )
            for row in rows
        ),
        y_min=0.0,
        y_max=850.0,
        y_label="Consecutive hours",
    )


def _finite_hold_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["policy_id"] == POLICIES[0]]
    return _bar_svg(
        title="Finite holding fixed persistence, not robustness",
        subtitle="Post-hoc calibration diagnostic; all bootstrap lower bounds remained negative",
        groups=tuple(
            (
                f"{row['maximum_holding_hours']}h cap",
                (
                    (
                        "Stress return",
                        100.0 * float(row["stress_total_net_return_fraction"]),
                        COLORS["blue"],
                    ),
                    (
                        "Maximum drawdown",
                        100.0 * float(row["stress_maximum_drawdown_fraction"]),
                        COLORS["amber"],
                    ),
                ),
            )
            for row in selected
        ),
        y_min=-10.0,
        y_max=27.0,
        y_label="Percent of unlevered capital",
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points = []
    for row in rows:
        raw = str(row.get("best_policy_mean_net_bps", "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            points.append((float(row["round"]), value))
    labels = {points[0][0]: str(int(points[0][0])), points[-1][0]: str(ROUND)}
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        if points[0][0] <= value <= points[-1][0]:
            labels[value] = str(int(value))
    return _line_svg(
        title="Optimization research progression",
        subtitle="Descriptive best bps/trade by round; datasets differ and every point is non-promotable",
        series=(("Descriptive best policy", points, COLORS["teal"]),),
        x_labels=labels,
        y_label="Approximate net bps per closed trade",
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


def _readme(
    report: Mapping[str, object],
    diagnostic: Mapping[str, object],
) -> str:
    median = report["calibration"][POLICIES[0]]["stress"]
    finite = max(
        diagnostic["finite_holding_screen"],
        key=lambda row: float(row["stress_total_net_return_fraction"]),
    )
    rank = max(
        float(row["directional_spearman"])
        for row in diagnostic["directional_rank"]
        if row["symbol"] == "ALL"
    )
    return f"""# Round 54: Sequential Distributional Action Value

> **Rejected before evaluation.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Three DirectML-trained dueling causal TCNs reduced matched early-stop Bellman residual loss by `25.76%` to `26.20%`, but that was not directional or economic proof. Calibration directional rank peaked at `{rank:.6f}`. The median-Q controller held a position for up to `791` hours and returned `{100.0 * float(median['total_net_return_fraction']):.2f}%` under stress with `{100.0 * float(median['maximum_drawdown_fraction']):.2f}%` drawdown.

A post-hoc finite-hold diagnostic found its least-bad point at `{finite['maximum_holding_hours']}h`: `{100.0 * float(finite['stress_total_net_return_fraction']):+.2f}%` stress return across `{finite['closed_trades']}` trades, `{100.0 * float(finite['stress_maximum_drawdown_fraction']):.2f}%` drawdown, and a `{float(finite['bootstrap_lower_mean_hourly_bps']):.4f}` bps/hour bootstrap lower bound. It remains rejected and selection-contaminated.

Round 55 must forecast finite-horizon return distributions directly, keep the controller bounded, and gate directional skill, proper scores, path risk, net action value, bootstrap evidence, activity, and asset breadth separately. The evaluation interval remains unread.

## Evidence

| View | Graph | Source |
|---|---|---|
| Bellman fit | [SVG](charts/model-skill.svg) | [CSV](models.csv) |
| Directional rank | [SVG](charts/directional-rank.svg) | [CSV](directional-rank.csv) |
| Policy economics | [SVG](charts/policy-economics.svg) | [CSV](policies.csv) |
| Holding duration | [SVG](charts/holding-duration.svg) | [CSV](holding-summary.csv) |
| Bounded-hold screen | [SVG](charts/finite-hold.svg) | [CSV](finite-hold.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`screen.json`, `failure-diagnostic.json`, and `holding-runs.csv` preserve the full verified evidence. `report.json` binds every publication artifact to the exact external reports, design, dataset, model artifacts, and diagnostic implementation.
"""


def publish(
    *,
    report_path: Path,
    diagnostic_path: Path,
    design_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, diagnostic, _design = _validate_sources(
        report_path,
        diagnostic_path,
        design_path,
    )
    model_rows = _model_rows(report)
    direction_rows = _direction_rows(diagnostic)
    policy_rows = _policy_rows(report)
    holding_rows = _holding_rows(diagnostic)
    holding_summary_rows = _holding_summary_rows(diagnostic)
    finite_hold_rows = _finite_hold_rows(diagnostic)
    gate_rows = _gate_rows(report)
    progress_rows, progress_fields = _progress_rows(
        previous_progress_path,
        diagnostic,
    )
    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "models.csv", model_rows)
    _write_csv(output_dir / "directional-rank.csv", direction_rows)
    _write_csv(output_dir / "policies.csv", policy_rows)
    _write_csv(output_dir / "holding-runs.csv", holding_rows)
    _write_csv(output_dir / "holding-summary.csv", holding_summary_rows)
    _write_csv(output_dir / "finite-hold.csv", finite_hold_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "model-skill.svg", _model_svg(model_rows))
    _write_text(charts / "directional-rank.svg", _direction_svg(direction_rows))
    _write_text(charts / "policy-economics.svg", _policy_svg(policy_rows))
    _write_text(
        charts / "holding-duration.svg",
        _holding_svg(holding_summary_rows),
    )
    _write_text(charts / "finite-hold.svg", _finite_hold_svg(finite_hold_rows))
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))
    _write_text(output_dir / "README.md", _readme(report, diagnostic))
    write_json_atomic(output_dir / "screen.json", report, indent=2, sort_keys=True)
    write_json_atomic(
        output_dir / "failure-diagnostic.json",
        diagnostic,
        indent=2,
        sort_keys=True,
    )
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "published_at_utc": report["generated_at_utc"],
        "publisher_path": "tools/publish_round54_sequential_failure.py",
        "source": {
            "report_path": str(report_path),
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "diagnostic_path": str(diagnostic_path),
            "diagnostic_file_sha256": DIAGNOSTIC_FILE_SHA256,
            "diagnostic_canonical_sha256": DIAGNOSTIC_CANONICAL_SHA256,
            "diagnostic_implementation_commit": DIAGNOSTIC_IMPLEMENTATION_COMMIT,
            "diagnostic_implementation_blob": DIAGNOSTIC_IMPLEMENTATION_BLOB,
            "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
            "design_file_sha256": DESIGN_FILE_SHA256,
            "design_sha256": DESIGN_SHA256,
            "dataset_sha256": report["dataset_sha256"],
            "training_implementation_commit": report["implementation_commit"],
        },
        "claims": {
            "status": "rejected",
            "selection_contaminated": True,
            "evaluation_role_read": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "trading_authority": False,
            "testnet_authority": False,
            "live_authority": False,
            "leverage_applied": False,
            "untouched_data_expansion_authorized": False,
        },
        "result": {
            "models": len(model_rows),
            "calibration_policies": len(policy_rows),
            "passed_calibration_policies": 0,
            "maximum_observed_holding_hours": max(
                int(row["maximum_holding_hours"])
                for row in holding_summary_rows
            ),
            "finite_hold_rules_tested": len(finite_hold_rows),
            "evaluation_replayed": False,
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2, sort_keys=True)
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round54-sequential-action-value-20260714-v2\report.json"
        ),
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round54-sequential-failure-diagnostic-20260714-v1.json"
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=research
        / "round-054-sequential-distributional-action-value-prototype-design.json",
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
        diagnostic_path=arguments.diagnostic.resolve(),
        design_path=arguments.design.resolve(),
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
