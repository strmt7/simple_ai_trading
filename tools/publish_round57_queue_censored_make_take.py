"""Publish hash-verified Round 57 make/take predictive evidence."""

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


ROUND = 57
REPORT_SCHEMA = "round-057-queue-censored-make-take-report-v1"
PUBLICATION_SCHEMA = "round-057-queue-censored-make-take-publication-v1"
DESIGN_SCHEMA = "round-057-queue-censored-make-take-design-v1"
CONTRACT_SCHEMA = "round-057-queue-censored-make-take-execution-contract-v1"
BINDING_SCHEMA = "round-057-queue-censored-make-take-execution-binding-v1"
STATE_SCHEMA = "round-057-run-state-v1"
REPORT_CANONICAL_SHA256 = (
    "c026542c7073496317f02a3f30dbf4274e7c7155a5026d79d8d76883ed924f27"
)
REPORT_FILE_SHA256 = "1e549d21b6aa96f78e5f7e740457af0116e161ce67e9b600a62d28ee2d12e712"
DESIGN_SHA256 = "61165c1150f5349ff1b15f2243e7520ecbd7af44578edda7fc2be8bb56b66cdf"
CONTRACT_SHA256 = "ef42dcd1fcf003838a34c78a3d87a49b45d78f16b7be47b596fc9eece9841dd6"
BINDING_SHA256 = "04e3900823bf60d7a93ab8d734aa85375b5a71e1b8d3854db55daddd05f98bb4"
IMPLEMENTATION_COMMIT = "57017244837d6c94f03403393253d823e6c62c7a"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ROLES = ("policy_calibration", "evaluation")
ACTIONS = (
    "passive_long",
    "passive_short",
    "aggressive_long",
    "aggressive_short",
)
ACTION_DISPLAY = {
    "passive_long": "Passive long",
    "passive_short": "Passive short",
    "aggressive_long": "Aggressive long",
    "aggressive_short": "Aggressive short",
}
ACTION_COLORS = {
    "passive_long": COLORS["teal"],
    "passive_short": COLORS["blue"],
    "aggressive_long": COLORS["amber"],
    "aggressive_short": COLORS["red"],
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
    claimed = str(canonical.pop(digest_key, ""))
    actual = _canonical_sha256(canonical)
    if claimed != actual:
        raise ValueError(f"{digest_key} does not match canonical content")
    return actual


def _validate_sources(
    report_path: Path,
    design_path: Path,
    contract_path: Path,
    binding_path: Path,
    state_path: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    if _file_sha256(report_path) != REPORT_FILE_SHA256:
        raise ValueError("Round 57 report file hash drifted")
    report = _read_object(report_path, "Round 57 report")
    design = _read_object(design_path, "Round 57 design")
    contract = _read_object(contract_path, "Round 57 contract")
    binding = _read_object(binding_path, "Round 57 binding")
    state = _read_object(state_path, "Round 57 run state")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected_policy_predictive"
        or _canonical_value(report, "report_sha256") != REPORT_CANONICAL_SHA256
        or design.get("schema_version") != DESIGN_SCHEMA
        or _canonical_value(design, "design_sha256") != DESIGN_SHA256
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or _canonical_value(contract, "contract_sha256") != CONTRACT_SHA256
        or binding.get("schema_version") != BINDING_SCHEMA
        or _canonical_value(binding, "binding_sha256") != BINDING_SHA256
        or state.get("schema_version") != STATE_SCHEMA
        or _canonical_value(state, "state_sha256")
        != "b549de9926935be5699c0ab73611a481a409947f1096ab5425d38448ee28b983"
        or report.get("design_sha256") != DESIGN_SHA256
        or report.get("contract_sha256") != CONTRACT_SHA256
        or report.get("binding_sha256") != BINDING_SHA256
        or report.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or binding.get("implementation_commit") != IMPLEMENTATION_COMMIT
        or state.get("report_sha256") != REPORT_CANONICAL_SHA256
        or state.get("result_status") != report.get("status")
        or report.get("policy_selection") is not None
        or report.get("economic_evaluation") is not None
        or report.get("selection_contaminated") is not True
        or report.get("trading_authority") is not False
        or report.get("execution_claim") is not False
        or report.get("profitability_claim") is not False
        or report.get("portfolio_claim") is not False
        or report.get("leverage_applied") is not False
        or report.get("ai_uplift_claim") is not False
        or tuple(row["symbol"] for row in report["source"]) != SYMBOLS
    ):
        raise ValueError("Round 57 source contracts or claims drifted")
    _validate_finite(report)

    for row in binding["blobs"]:
        source_path = str(row["path"])
        expected_oid = str(row["git_blob_oid"])
        if _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:{source_path}") != expected_oid:
            raise ValueError(f"Round 57 implementation blob drifted: {source_path}")

    model_root = report_path.parent / "models"
    expected_seeds = (5701, 5702, 5703)
    for family in ("queue_fill", "payoff"):
        manifests = report["models"]["artifacts"][family]
        if tuple(int(row["seed"]) for row in manifests) != expected_seeds:
            raise ValueError(f"Round 57 {family} seeds drifted")
        for row in manifests:
            path = model_root / str(row["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(row["bytes"])
                or _file_sha256(path) != str(row["file_sha256"])
            ):
                raise ValueError(f"Round 57 model artifact drifted: {path}")
            model = _read_object(path, f"Round 57 {family} model")
            if (
                model.get("model_sha256") != row["model_sha256"]
                or model.get("seed") != row["seed"]
                or model.get("backend_kind") != "opencl"
                or model.get("backend_device") != "opencl:auto"
                or model.get("trading_authority") is not False
                or model.get("profitability_claim") is not False
                or model.get("leverage_applied") is not False
            ):
                raise ValueError(f"Round 57 model identity drifted: {path}")
    return report, design, contract, binding, state


def _fill_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ROLES:
        for item in report["predictive"][role]["fill_metrics"]:
            rows.append({"round": ROUND, "role": role, **item})
    return rows


def _payoff_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ROLES:
        for item in report["predictive"][role]["payoff_metrics"]:
            rows.append({"round": ROUND, "role": role, **item})
    return rows


def _action_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ROLES:
        for symbol_row in report["action_values"][role]:
            for action, values in symbol_row["by_action"].items():
                rows.append(
                    {
                        "round": ROUND,
                        "role": role,
                        "symbol": symbol_row["symbol"],
                        "event_rows": symbol_row["event_rows"],
                        "action": action,
                        **values,
                    }
                )
    return rows


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    source_by_symbol = {row["symbol"]: row for row in report["source"]}
    return [
        {
            "round": ROUND,
            "symbol": row["symbol"],
            "role": row["role"],
            "source_rows": source_by_symbol[row["symbol"]]["rows"],
            "source_dataset_fingerprint": source_by_symbol[row["symbol"]][
                "dataset_fingerprint"
            ],
            "candidate_rows_before_stress_quote_gate": row[
                "candidate_rows_before_stress_quote_gate"
            ],
            "stress_quote_valid_rows": row["stress_quote_valid_rows"],
            "stress_quote_invalid_rows": row["stress_quote_invalid_rows"],
            "trade_rows": row["trade_rows"],
            "trade_source_sha256": row["trade_source_sha256"],
        }
        for row in report["role_evidence"]
    ]


def _model_rows(
    report: Mapping[str, object], report_path: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in ("queue_fill", "payoff"):
        for item in report["models"]["artifacts"][family]:
            model = _read_object(
                report_path.parent / "models" / str(item["path"]),
                f"Round 57 {family} model",
            )
            rows.append(
                {
                    "round": ROUND,
                    "family": family,
                    "seed": item["seed"],
                    "backend_kind": model["backend_kind"],
                    "backend_device": model["backend_device"],
                    "lightgbm_version": model["lightgbm_version"],
                    "best_iterations": ";".join(
                        str(value) for value in model["best_iterations"]
                    ),
                    "model_sha256": item["model_sha256"],
                    "file_sha256": item["file_sha256"],
                    "bytes": item["bytes"],
                }
            )
    return rows


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ROLES:
        predictive = report["predictive"][role]
        rows.append(
            {
                "round": ROUND,
                "role": role,
                "scope": "payoff_early_quality",
                "symbol": "ALL",
                "action_or_side": "ALL",
                "passed": predictive["payoff_early_quality_gate_passed"],
            }
        )
        for item in predictive["fill_metrics"]:
            rows.append(
                {
                    "round": ROUND,
                    "role": role,
                    "scope": "queue_fill",
                    "symbol": item["symbol"],
                    "action_or_side": item["side"],
                    "passed": item["passed"],
                }
            )
        for item in predictive["payoff_metrics"]:
            rows.append(
                {
                    "round": ROUND,
                    "role": role,
                    "scope": "conditional_payoff",
                    "symbol": item["symbol"],
                    "action_or_side": item["action_name"],
                    "passed": item["passed"],
                }
            )
        rows.append(
            {
                "round": ROUND,
                "role": role,
                "scope": "predictive_gate",
                "symbol": "ALL",
                "action_or_side": "ALL",
                "passed": predictive["predictive_gate_passed"],
            }
        )
    return rows


def _progress_rows(
    previous_path: Path, report: Mapping[str, object]
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
        raise ValueError("research progress must contain exactly Rounds 1 through 56")
    evaluation = report["predictive"]["evaluation"]
    mean_spearman = sum(
        float(item["spearman"]) for item in evaluation["payoff_metrics"]
    ) / len(evaluation["payoff_metrics"])
    calibration_events = sum(
        int(item["event_rows"])
        for item in report["action_values"]["policy_calibration"]
    )
    new = {field: "" for field in fields}
    new.update(
        {
            "round": ROUND,
            "stage": "queue-censored make/take LightGBM seed ensembles",
            "periods": (
                "train 2023-05-16..05-31; calibration 2023-06-05..06-08; "
                "consumed evaluation 2023-06-09..06-14"
            ),
            "selection_contaminated": "True",
            "horizon_seconds": "300",
            "feature_set": (
                "causal L1, tape, aggregate depth, queue, exponential flow; "
                "10-second decisions"
            ),
            "risk_level": "predictive mechanism screen; unlevered; no policy replay",
            "spearman_ic": str(mean_spearman),
            "selected_signals": "0",
            "executable_trades": "0",
            "status": "rejected",
            "source_file": (
                "verified Round 57 report; Spearman is unweighted mean over 12 "
                "evaluation symbol-action cells"
            ),
            "best_model_id": "queue_fill_retained_payoff_rejected",
            "ensemble_models": "6",
            "calibration_eligible_rows": str(calibration_events),
            "policy_eligible_rows": "0",
            "development_consumed": "True",
            "architecture_gates_passed": "12",
            "architecture_gate_count": "19",
        }
    )
    rows.append(new)
    return rows, fields


def _fill_skill_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["role"] == "evaluation"]
    return _bar_svg(
        title="Queue-fill survival generalized across the consumed evaluation",
        subtitle=(
            "9-14 June 2023 UTC; exact 15-second fill outcomes; positive values "
            "beat causal training-prevalence baselines"
        ),
        groups=tuple(
            (
                f"{str(row['symbol'])[:3]} {str(row['side'])[0].upper()}",
                (
                    (
                        "Log-loss skill",
                        100.0 * float(row["log_loss_skill"]),
                        COLORS["teal"],
                    ),
                    (
                        "Integrated Brier skill",
                        100.0 * float(row["integrated_brier_skill"]),
                        COLORS["blue"],
                    ),
                ),
            )
            for row in selected
        ),
        y_min=0.0,
        y_max=25.0,
        y_label="Proper-score skill (%)",
        tick_decimals=0,
        value_decimals=2,
    )


def _top_quintile_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["role"] == "evaluation"]
    by_key = {(row["symbol"], row["action_name"]): row for row in selected}
    return _bar_svg(
        title="Every highest-scored payoff quintile remained negative after costs",
        subtitle=(
            "9-14 June 2023 UTC consumed evaluation; 750 ms placement and protection "
            "latency, observed spread, fees, slippage, and path exits"
        ),
        groups=tuple(
            (
                symbol[:3],
                tuple(
                    (
                        ACTION_DISPLAY[action],
                        float(by_key[(symbol, action)]["top_quintile_mean_net_bps"]),
                        ACTION_COLORS[action],
                    )
                    for action in ACTIONS
                ),
            )
            for symbol in SYMBOLS
        ),
        y_min=-18.0,
        y_max=1.0,
        y_label="Top-quintile realized net payoff (bps)",
        tick_decimals=1,
        value_decimals=2,
    )


def _action_value_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["role"] == "evaluation"]
    by_key = {(row["symbol"], row["action"]): row for row in selected}
    return _bar_svg(
        title="Predicted opportunity-weighted action value was negative",
        subtitle=(
            "9-14 June 2023 UTC consumed evaluation; passive values include zero "
            "for ineligible and unfilled opportunities"
        ),
        groups=tuple(
            (
                symbol[:3],
                tuple(
                    (
                        ACTION_DISPLAY[action],
                        float(by_key[(symbol, action)]["mean_expected_value_bps"]),
                        ACTION_COLORS[action],
                    )
                    for action in ACTIONS
                ),
            )
            for symbol in SYMBOLS
        ),
        y_min=-15.0,
        y_max=1.0,
        y_label="Mean predicted value per opportunity (bps)",
        tick_decimals=1,
        value_decimals=2,
    )


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    points: list[tuple[float, float]] = []
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
    labels = {value: str(int(value)) for value in (points[0][0], points[-1][0])}
    for value in (10.0, 20.0, 30.0, 40.0, 50.0):
        if points[0][0] <= value <= points[-1][0]:
            labels[value] = str(int(value))
    return _line_svg(
        title="Optimization research record through Round 57",
        subtitle=(
            "Recorded rank statistic by round; Round 57 is the transparent mean "
            "of 12 action cells, and differing targets make this diagnostic only"
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
    fill_rows: Sequence[Mapping[str, object]],
    payoff_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evaluation_fill = [row for row in fill_rows if row["role"] == "evaluation"]
    evaluation_payoff = [row for row in payoff_rows if row["role"] == "evaluation"]
    top_values = [float(row["top_quintile_mean_net_bps"]) for row in evaluation_payoff]
    analysis: dict[str, object] = {
        "schema_version": "round-057-queue-censored-make-take-failure-analysis-v1",
        "round": ROUND,
        "source_report_sha256": REPORT_CANONICAL_SHA256,
        "status": "rejected_policy_predictive",
        "facts": {
            "evaluation_fill_cells_passed": sum(
                bool(row["passed"]) for row in evaluation_fill
            ),
            "evaluation_fill_cells_total": len(evaluation_fill),
            "evaluation_payoff_cells_passed": sum(
                bool(row["passed"]) for row in evaluation_payoff
            ),
            "evaluation_payoff_cells_total": len(evaluation_payoff),
            "evaluation_positive_top_quintile_cells": sum(
                value > 0.0 for value in top_values
            ),
            "evaluation_top_quintile_best_net_bps": max(top_values),
            "evaluation_top_quintile_worst_net_bps": min(top_values),
            "policy_selection_performed": False,
            "economic_replay_performed": False,
            "ai_ablation_performed": False,
        },
        "retained_mechanism": [
            "The queue-fill survival ensemble passed every policy-calibration and consumed-evaluation symbol-side cell on both proper scores.",
            "Future passive lifecycle research may reuse the fill model only under a separately frozen design and new held-forward evidence.",
        ],
        "rejected_mechanism": [
            "Conditional directional payoff failed the frozen predictive gate.",
            "All 12 consumed-evaluation top-score quintiles had negative realized net payoff after the frozen execution costs.",
            "The current lifecycle always pays a taker exit, so a separately frozen passive-exit mechanism is the next cost hypothesis; it is not an edge claim.",
        ],
        "prohibited_inferences": [
            "profitability",
            "AI uplift",
            "testnet readiness",
            "live trading readiness",
            "leverage readiness",
            "threshold repair from consumed outcomes",
        ],
    }
    analysis["analysis_sha256"] = _canonical_sha256(analysis)
    return analysis


def _readme(report: Mapping[str, object]) -> str:
    fill = {
        (row["symbol"], row["side"]): row
        for row in report["predictive"]["evaluation"]["fill_metrics"]
    }
    payoff = {
        (row["symbol"], row["action_name"]): row
        for row in report["predictive"]["evaluation"]["payoff_metrics"]
    }
    fill_table = "\n".join(
        "| {symbol} | {ll:.2f}% | {lb:.2f}% | {sl:.2f}% | {sb:.2f}% |".format(
            symbol=symbol,
            ll=100 * float(fill[(symbol, "long")]["log_loss_skill"]),
            lb=100 * float(fill[(symbol, "long")]["integrated_brier_skill"]),
            sl=100 * float(fill[(symbol, "short")]["log_loss_skill"]),
            sb=100 * float(fill[(symbol, "short")]["integrated_brier_skill"]),
        )
        for symbol in SYMBOLS
    )
    payoff_table = "\n".join(
        "| {symbol} | {pl:+.2f} | {ps:+.2f} | {al:+.2f} | {ass:+.2f} |".format(
            symbol=symbol,
            pl=float(payoff[(symbol, "passive_long")]["top_quintile_mean_net_bps"]),
            ps=float(payoff[(symbol, "passive_short")]["top_quintile_mean_net_bps"]),
            al=float(payoff[(symbol, "aggressive_long")]["top_quintile_mean_net_bps"]),
            ass=float(
                payoff[(symbol, "aggressive_short")]["top_quintile_mean_net_bps"]
            ),
        )
        for symbol in SYMBOLS
    )
    return f"""# Round 57: Queue-Censored Make/Take

> **Rejected development evidence.** No profitability, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 57 trained fixed three-seed queue-fill and payoff ensembles with AMD OpenCL LightGBM on real official Binance USD-M BTCUSDT, ETHUSDT, and SOLUSDT events. Decisions were spaced 10 seconds apart. Labels used an explicit 750 ms placement delay, a 15-second queue-censored passive-order lifetime, observed spread and queue, a 100 ms path grid, fees, slippage, protection latency, and a five-minute post-fill lifecycle.

The queue-fill mechanism generalized. The directional payoff mechanism did not. Every fill cell passed in policy calibration and consumed evaluation, but only 3/12 evaluation payoff cells passed and every evaluation top-score quintile remained negative after costs. The run therefore stopped before policy selection and economic replay. Trades, ROI, drawdown, leverage, and AI uplift were not evaluated.

| Evaluation fill skill | Long log loss | Long Brier | Short log loss | Short Brier |
|---|---:|---:|---:|---:|
{fill_table}

| Top-quintile realized net payoff (bps) | Passive long | Passive short | Aggressive long | Aggressive short |
|---|---:|---:|---:|---:|
{payoff_table}

The retained hypothesis is narrow: queue-fill survival is useful execution infrastructure. The rejected hypothesis is that this L1/tape directional model can clear the frozen maker-entry/taker-exit or taker-entry/taker-exit costs. No threshold, leverage, or language model is allowed to repair that negative mechanism on consumed outcomes.

## Evidence

| View | Graph | Source |
|---|---|---|
| Queue-fill proper-score skill | [SVG](charts/fill-survival-skill.svg) | [CSV](fill-survival.csv) |
| Realized top-quintile payoff | [SVG](charts/top-quintile-net-payoff.svg) | [CSV](conditional-payoff.csv) |
| Opportunity-weighted action value | [SVG](charts/expected-action-value.svg) | [CSV](action-values.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`source-coverage.csv`, `model-artifacts.csv`, `gates.csv`, `failure-analysis.json`, and `screen.json` preserve the remaining source-bound evidence. Every chart is regenerated from tracked tabular data.
"""


def publish(
    *,
    report_path: Path,
    design_path: Path,
    contract_path: Path,
    binding_path: Path,
    state_path: Path,
    previous_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, _design, _contract, _binding, state = _validate_sources(
        report_path, design_path, contract_path, binding_path, state_path
    )
    fill_rows = _fill_rows(report)
    payoff_rows = _payoff_rows(report)
    action_rows = _action_rows(report)
    source_rows = _source_rows(report)
    model_rows = _model_rows(report, report_path)
    gate_rows = _gate_rows(report)
    progress_rows, progress_fields = _progress_rows(previous_progress_path, report)
    failure = _failure_analysis(report, fill_rows, payoff_rows)

    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "fill-survival.csv", fill_rows)
    _write_csv(output_dir / "conditional-payoff.csv", payoff_rows)
    _write_csv(output_dir / "action-values.csv", action_rows)
    _write_csv(output_dir / "source-coverage.csv", source_rows)
    _write_csv(output_dir / "model-artifacts.csv", model_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "fill-survival-skill.svg", _fill_skill_svg(fill_rows))
    _write_text(charts / "top-quintile-net-payoff.svg", _top_quintile_svg(payoff_rows))
    _write_text(charts / "expected-action-value.svg", _action_value_svg(action_rows))
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
        "published_at_ms": state["observed_at_ms"],
        "publisher_path": "tools/publish_round57_queue_censored_make_take.py",
        "source": {
            "report_path": str(report_path),
            "report_file_sha256": REPORT_FILE_SHA256,
            "report_canonical_sha256": REPORT_CANONICAL_SHA256,
            "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
            "design_sha256": DESIGN_SHA256,
            "contract_path": str(contract_path.relative_to(ROOT)).replace("\\", "/"),
            "contract_sha256": CONTRACT_SHA256,
            "binding_path": str(binding_path.relative_to(ROOT)).replace("\\", "/"),
            "binding_sha256": BINDING_SHA256,
            "implementation_commit": IMPLEMENTATION_COMMIT,
            "source_dataset_fingerprints": {
                row["symbol"]: row["dataset_fingerprint"] for row in report["source"]
            },
        },
        "claims": {
            "status": "rejected_policy_predictive",
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
            "queue_fill_seed_members": 3,
            "payoff_seed_members": 3,
            "policy_calibration_fill_cells_passed": 6,
            "policy_calibration_fill_cells_total": 6,
            "policy_calibration_payoff_cells_passed": 5,
            "policy_calibration_payoff_cells_total": 12,
            "evaluation_fill_cells_passed": 6,
            "evaluation_fill_cells_total": 6,
            "evaluation_payoff_cells_passed": 3,
            "evaluation_payoff_cells_total": 12,
            "positive_evaluation_top_quintile_cells": 0,
            "policy_selected": False,
            "economic_replay_performed": False,
            "ai_ablation_performed": False,
        },
        "artifacts": [_artifact(path, output_dir) for path in artifact_paths],
    }
    publication["publication_canonical_sha256"] = _canonical_sha256(publication)
    write_json_atomic(output_dir / "report.json", publication, indent=2)
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    evidence = Path(r"E:\SimpleAITradingData\evidence\round57-20260715-1509-v2")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=evidence / "round57-report.json")
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-057-queue-censored-make-take-design.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=research / "round-057-queue-censored-make-take-execution-contract.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-057-queue-censored-make-take-execution-binding.json",
    )
    parser.add_argument("--state", type=Path, default=evidence / "run-state.json")
    parser.add_argument(
        "--progress", type=Path, default=research / "latest" / "progress.csv"
    )
    parser.add_argument("--output", type=Path, default=research / "latest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    publication = publish(
        report_path=arguments.report.resolve(),
        design_path=arguments.design.resolve(),
        contract_path=arguments.contract.resolve(),
        binding_path=arguments.binding.resolve(),
        state_path=arguments.state.resolve(),
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
