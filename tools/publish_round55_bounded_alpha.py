"""Publish hash-bound Round 55 bounded-alpha and AI-ablation evidence."""

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


ROUND = 55
REPORT_SCHEMA = "round-055-bounded-alpha-report-v1"
PUBLICATION_SCHEMA = "round-055-bounded-alpha-publication-v1"
DESIGN_SCHEMA = "round-055-bounded-alpha-lightgbm-ai-design-v1"
BINDING_SCHEMA = "round-055-bounded-alpha-execution-binding-v1"
REPORT_CANONICAL_SHA256 = (
    "47dc22e987fff9cb508ff09fed2222e80391d18797496d4f2f9e476aee887919"
)
REPORT_FILE_SHA256 = (
    "b556ff0302d230ef620a7bfb11ad49ad35f5d083ebc390d5174657a4f466fda2"
)
DESIGN_SHA256 = "e6746db669ffc2633f197cf61dd73b369c2889381900f4e73a375192922f2827"
BINDING_SHA256 = "731c0216b53226376bb465c3fce0d3d32cb6bcecf063d98e3cd2ac9e9c9315ed"
IMPLEMENTATION_COMMIT = "e1bb011385a326602ac73c9c86b017c48178b20d"
TREATMENTS = ("baseline_71", "ai_program_augmented")
INTERVALS = ("policy_development", "development_holdout")
DISPLAY = {
    "baseline_71": "Baseline",
    "ai_program_augmented": "8B AI factors",
    "policy_development": "Jul-Aug 2024",
    "development_holdout": "Sep 2024",
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


def _verify_artifact(row: Mapping[str, object]) -> Path:
    path = Path(str(row.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(row.get("bytes", -1))
        or _file_sha256(path) != str(row.get("sha256", ""))
    ):
        raise ValueError(f"Round 55 artifact drifted: {path}")
    return path


def _validate_sources(
    report_path: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if _file_sha256(report_path) != REPORT_FILE_SHA256:
        raise ValueError("Round 55 report file hash drifted")
    report = _read_object(report_path, "Round 55 report")
    design = _read_object(design_path, "Round 55 design")
    binding = _read_object(binding_path, "Round 55 binding")
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
        or not isinstance(data, Mapping)
        or any(value is not False for value in claims.values())
        or data.get("synthetic_rows") != 0
        or data.get("forbidden_existing_rows_loaded") is not False
        or data.get("selection_confirmation_or_terminal_rows_read") is not False
        or report.get("retained_for_separately_frozen_untouched_design") != []
    ):
        raise ValueError("Round 55 source contracts or claims drifted")
    _validate_finite(report)
    if _git("rev-parse", f"{IMPLEMENTATION_COMMIT}:tools/run_round55_bounded_alpha.py") != next(
        str(row["git_blob_oid"])
        for row in binding["blobs"]
        if row["path"] == "tools/run_round55_bounded_alpha.py"
    ):
        raise ValueError("Round 55 implementation blob drifted")

    for treatment in TREATMENTS:
        models = report["model"]["artifacts"][treatment]
        prediction = {
            "path": models["prediction_path"],
            "bytes": models["prediction_bytes"],
            "sha256": models["prediction_sha256"],
        }
        _verify_artifact(prediction)
        for row in models["artifacts"]:
            model_path = _verify_artifact(
                {
                    "path": row["path"],
                    "bytes": row["bytes"],
                    "sha256": row["file_sha256"],
                }
            )
            model = _read_object(model_path, "Round 55 model artifact")
            if (
                _canonical_value(model, "model_sha256") != row["model_sha256"]
                or model.get("treatment_id") != treatment
                or model.get("view_id") != row["view_id"]
                or model.get("seed") != row["seed"]
            ):
                raise ValueError(f"Round 55 model identity drifted: {model_path}")
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
    external = binding["external_evidence"]
    for row in external.values():
        _verify_artifact(row)
    return report, design, binding


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _treatment_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for treatment in TREATMENTS:
        result = report["treatments"][treatment]
        for interval in INTERVALS:
            interval_result = result["intervals"][interval]
            for scenario in ("base", "stress"):
                metrics = interval_result[scenario]
                rows.append(
                    {
                        "round": ROUND,
                        "treatment": treatment,
                        "interval": interval,
                        "scenario": scenario,
                        **dict(metrics),
                        "bootstrap_lower_mean_hourly_bps": interval_result[
                            "stress_block_bootstrap_mean_hourly_bps"
                        ]["lower_bps"]
                        if scenario == "stress"
                        else "",
                        "gate_passed": result["gate"]["passed"],
                        "gate_failures": ";".join(result["gate"]["failures"]),
                    }
                )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for treatment in TREATMENTS:
        for row in report["model"]["artifacts"][treatment]["artifacts"]:
            for side in ("long", "short"):
                rows.append(
                    {
                        "round": ROUND,
                        "treatment": treatment,
                        "view": row["view_id"],
                        "seed": row["seed"],
                        "side": side,
                        "best_iteration": row["best_iterations"][side],
                        "iteration_selection_mae_skill": row[
                            "iteration_selection_mae_skill"
                        ][side],
                        "backend": row["backend_kind"],
                        "device": row["backend_device"],
                        "reload_max_abs_prediction_error_bps": row[
                            "reload_max_abs_prediction_error_bps"
                        ],
                        "model_sha256": row["model_sha256"],
                        "file_sha256": row["file_sha256"],
                    }
                )
    return rows


def _rank_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    diagnostics = report["model"]["predictive_diagnostics"]
    for treatment in TREATMENTS:
        for interval in INTERVALS:
            for view, seed_rows in diagnostics[treatment][interval].items():
                for seed_row in seed_rows:
                    for side, value in seed_row["spearman"].items():
                        rows.append(
                            {
                                "round": ROUND,
                                "treatment": treatment,
                                "interval": interval,
                                "view": view,
                                "seed": seed_row["seed"],
                                "side": side,
                                "spearman": value,
                            }
                        )
    return rows


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for treatment in TREATMENTS:
        gate = report["treatments"][treatment]["gate"]
        failures = set(gate["failures"])
        for name in (
            "development_closed_trades",
            "development_active_days",
            "development_profit_factor",
            "development_drawdown",
            "development_return",
            "holdout_closed_trades",
            "holdout_active_days",
            "holdout_profit_factor",
            "holdout_drawdown",
            "holdout_return",
            "holdout_symbol_activity",
            "holdout_symbol_concentration",
            "view_stability",
            "seed_sign_stability",
            "holdout_familywise_bootstrap",
            "model_reload",
        ):
            rows.append(
                {
                    "round": ROUND,
                    "candidate": treatment,
                    "gate": name,
                    "passed": name not in failures,
                }
            )
    uplift = report["ai_uplift_gate"]
    for name, passed in uplift["checks"].items():
        rows.append(
            {
                "round": ROUND,
                "candidate": "ai_uplift",
                "gate": name,
                "passed": passed,
            }
        )
    return rows


def _ai_rows(binding: Mapping[str, object]) -> list[dict[str, object]]:
    ledger_path = Path(binding["external_evidence"]["ai_ledger"]["path"])
    ledger = _read_object(ledger_path, "Round 55 AI ledger")
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


def _equity_rows(hourly_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    output = []
    state: dict[tuple[str, str, str], float] = {}
    for row in hourly_rows:
        key = (row["treatment"], row["interval"], row["scenario"])
        state[key] = state.get(key, 1.0) + float(
            row["initial_capital_return_fraction"]
        )
        output.append(
            {
                **dict(row),
                "fixed_initial_capital_equity": state[key],
                "cumulative_return_fraction": state[key] - 1.0,
            }
        )
    return output


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
        raise ValueError("research progress must contain exactly Rounds 1 through 54")
    best = report["treatments"]["ai_program_augmented"]["intervals"][
        "policy_development"
    ]["stress"]
    gate = report["treatments"]["ai_program_augmented"]["gate"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "stop-bounded multi-view LightGBM with 8B AI-factor ablation",
            "periods": "train 2022-01-01..2024-06-30; development 2024-07-01..2024-09-30; later rows excluded",
            "selection_contaminated": "True",
            "horizon_seconds": "3600",
            "feature_set": "71 causal hourly features plus 7 governed AI programs",
            "risk_level": "conservative research controller; unlevered; fixed capital",
            "spearman_ic": str(
                max(
                    float(row["spearman"])
                    for row in _rank_rows(report)
                    if row["spearman"] is not None
                )
            ),
            "selected_signals": str(best["signals_before_cooldowns"]),
            "executable_trades": str(best["closed_trades"]),
            "status": "rejected",
            "source_file": "verified Round 55 report; exact 1m Binance OHLC and funding paths",
            "best_policy_trades": str(best["closed_trades"]),
            "best_policy_total_net_bps": str(
                10_000.0 * float(best["total_return_fraction"])
            ),
            "best_policy_mean_net_bps": str(
                best["mean_trade_initial_capital_bps"]
            ),
            "best_policy_max_drawdown_bps": str(
                10_000.0 * float(best["maximum_drawdown_fraction"])
            ),
            "best_policy_profit_factor": str(best["profit_factor"]),
            "best_model_id": "ai_program_augmented_development_only",
            "ensemble_models": "9",
            "calibration_eligible_rows": str(
                report["chronology"]["policy_development"]["rows"]
            ),
            "policy_eligible_rows": str(best["closed_trades"]),
            "development_consumed": "True",
            "architecture_gates_passed": str(16 - len(gate["failures"])),
            "architecture_gate_count": "16",
        }
    )
    rows.append(row)
    return rows, fields


def _model_skill_svg(rows: Sequence[Mapping[str, object]]) -> str:
    groups = []
    views = (
        ("raw_uniform", "raw uniform", COLORS["blue"]),
        ("risk_normalized_uniform", "risk uniform", COLORS["teal"]),
        (
            "risk_normalized_recency_180d",
            "risk recency 180d",
            COLORS["amber"],
        ),
    )
    for treatment in TREATMENTS:
        values = []
        for view, label, color in views:
            selected = [
                float(row["iteration_selection_mae_skill"])
                for row in rows
                if row["treatment"] == treatment and row["view"] == view
            ]
            values.append(
                (
                    label,
                    100.0 * sum(selected) / len(selected),
                    color,
                )
            )
        groups.append((DISPLAY[treatment], tuple(values)))
    return _bar_svg(
        title="Fresh refits beat constant payoff baselines only marginally",
        subtitle="Q1 2024 MAE skill averaged across long/short and three seeds",
        groups=tuple(groups),
        y_min=0.0,
        y_max=2.5,
        y_label="MAE skill (%)",
    )


def _economics_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["scenario"] == "stress"]
    return _bar_svg(
        title="Positive descriptive returns did not pass evidence gates",
        subtitle="Fixed initial capital; 16 bps round trip; no leverage; all intervals consumed",
        groups=tuple(
            (
                f"{DISPLAY[str(row['treatment'])]} {DISPLAY[str(row['interval'])]}",
                (
                    (
                        "Net return",
                        100.0 * float(row["total_return_fraction"]),
                        COLORS["teal"],
                    ),
                    (
                        "Maximum drawdown",
                        100.0 * float(row["maximum_drawdown_fraction"]),
                        COLORS["amber"],
                    ),
                ),
            )
            for row in selected
        ),
        y_min=0.0,
        y_max=0.45,
        y_label="Percent of initial capital",
    )


def _activity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [row for row in rows if row["scenario"] == "stress"]
    return _bar_svg(
        title="The bounded controller remained too selective",
        subtitle="Frozen minimums were 30/20 trades/days in Jul-Aug and 12/8 in September",
        groups=tuple(
            (
                f"{DISPLAY[str(row['treatment'])]} {DISPLAY[str(row['interval'])]}",
                (
                    ("Closed trades", float(row["closed_trades"]), COLORS["blue"]),
                    ("Active days", float(row["active_days"]), COLORS["teal"]),
                ),
            )
            for row in selected
        ),
        y_min=0.0,
        y_max=32.0,
        y_label="Count",
    )


def _ai_uplift_svg(report: Mapping[str, object]) -> str:
    groups = []
    for interval in INTERVALS:
        baseline = report["treatments"][TREATMENTS[0]]["intervals"][interval][
            "stress"
        ]
        ai = report["treatments"][TREATMENTS[1]]["intervals"][interval]["stress"]
        delta_bps = 10_000.0 * (
            float(ai["total_return_fraction"])
            - float(baseline["total_return_fraction"])
        )
        groups.append(
            (
                DISPLAY[interval],
                (("AI minus baseline", delta_bps, COLORS["teal"] if delta_bps >= 0 else COLORS["red"]),),
            )
        )
    return _bar_svg(
        title="AI factors did not demonstrate matched uplift",
        subtitle="Initial-capital return delta; September paired hourly lower bound was also negative",
        groups=tuple(groups),
        y_min=-5.0,
        y_max=15.0,
        y_label="Initial-capital basis points",
    )


def _equity_svg(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        row
        for row in rows
        if row["interval"] == "development_holdout" and row["scenario"] == "stress"
    ]
    timestamps = sorted({row["timestamp_utc"] for row in selected})
    x_by_time = {value: float(index) for index, value in enumerate(timestamps)}
    series = []
    for treatment, color in ((TREATMENTS[0], COLORS["blue"]), (TREATMENTS[1], COLORS["teal"])):
        points = [
            (
                x_by_time[row["timestamp_utc"]],
                10_000.0 * float(row["cumulative_return_fraction"]),
            )
            for row in selected
            if row["treatment"] == treatment
        ]
        series.append((DISPLAY[treatment], points, color))
    labels = {}
    for index in (0, len(timestamps) // 2, len(timestamps) - 1):
        labels[float(index)] = timestamps[index][5:10]
    return _line_svg(
        title="September fixed-capital stress equity",
        subtitle="Hourly one-minute-path realization; fixed initial capital, no leverage",
        series=tuple(series),
        x_labels=labels,
        y_label="Cumulative initial-capital bps",
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


def _readme(report: Mapping[str, object]) -> str:
    baseline_dev = report["treatments"][TREATMENTS[0]]["intervals"][INTERVALS[0]][
        "stress"
    ]
    ai_dev = report["treatments"][TREATMENTS[1]]["intervals"][INTERVALS[0]]["stress"]
    baseline_hold = report["treatments"][TREATMENTS[0]]["intervals"][INTERVALS[1]][
        "stress"
    ]
    ai_hold = report["treatments"][TREATMENTS[1]]["intervals"][INTERVALS[1]]["stress"]
    paired = report["ai_uplift_gate"]
    return f"""# Round 55: Stop-Bounded Payoff Models

> **Rejected development evidence.** No profitability, untouched-confirmation, AI-uplift, leverage, testnet, live-trading, or promotion claim is made.

Round 55 trained `18` OpenCL LightGBM artifacts (`36` side models) on BTCUSDT, ETHUSDT, and SOLUSDT. Targets used real one-minute Binance futures paths, exact gap-through stops, settled funding, and a `16 bps` round-trip stress charge. Every position stopped or timed out within `60 minutes`; notional used fixed initial capital with no reinvestment and no leverage.

| Treatment | Period | Trades | Stress return | Max drawdown | Profit factor |
|---|---:|---:|---:|---:|---:|
| Baseline | Jul-Aug 2024 | {baseline_dev['closed_trades']} | {100 * float(baseline_dev['total_return_fraction']):+.4f}% | {100 * float(baseline_dev['maximum_drawdown_fraction']):.4f}% | {float(baseline_dev['profit_factor']):.3f} |
| 8B AI factors | Jul-Aug 2024 | {ai_dev['closed_trades']} | {100 * float(ai_dev['total_return_fraction']):+.4f}% | {100 * float(ai_dev['maximum_drawdown_fraction']):.4f}% | {float(ai_dev['profit_factor']):.3f} |
| Baseline | Sep 2024 | {baseline_hold['closed_trades']} | {100 * float(baseline_hold['total_return_fraction']):+.4f}% | {100 * float(baseline_hold['maximum_drawdown_fraction']):.4f}% | {float(baseline_hold['profit_factor']):.3f} |
| 8B AI factors | Sep 2024 | {ai_hold['closed_trades']} | {100 * float(ai_hold['total_return_fraction']):+.4f}% | {100 * float(ai_hold['maximum_drawdown_fraction']):.4f}% | {float(ai_hold['profit_factor']):.3f} |

Both treatments failed six frozen gates: development and September trade/day counts, September P&L concentration, and the familywise block-bootstrap lower bound. The seven Fino1/Qwen3 factor programs improved July-August descriptively but reduced September stress return by `{100 * float(paired['holdout_stress_return_delta_fraction']):.4f}%`; the paired lower bound was `{float(paired['paired_hourly_bootstrap']['lower_bps']):.5f} bps/hour`. AI uplift therefore failed.

The run read `24,096` hourly timestamps and `72,288` symbol paths through September 2024. It generated no synthetic rows and did not load the `6,551` excluded October 2024-June 2025 timestamps. A future interval remains untouched, but Round 55 authorized no access to it.

## Evidence

| View | Graph | Source |
|---|---|---|
| Model skill | [SVG](charts/model-skill.svg) | [CSV](models.csv) |
| Path economics | [SVG](charts/economics.svg) | [CSV](treatments.csv) |
| Trading activity | [SVG](charts/activity.svg) | [CSV](treatments.csv) |
| Matched AI effect | [SVG](charts/ai-uplift.svg) | [CSV](treatments.csv) |
| September equity | [SVG](charts/september-equity.svg) | [CSV](equity.csv) |
| Round progression | [SVG](charts/research-progress.svg) | [CSV](progress.csv) |

`trades.csv`, `hourly-ledger.csv`, `monthly-economics.csv`, `predictive-rank.csv`, `ai-factors.csv`, `gates.csv`, and `screen.json` preserve the underlying evidence. Every chart is regenerated from tracked tabular data.
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
    treatment_rows = _treatment_rows(report)
    model_rows = _model_rows(report)
    rank_rows = _rank_rows(report)
    gate_rows = _gate_rows(report)
    ai_rows = _ai_rows(binding)
    source_artifacts = report["artifacts"]
    trade_rows = _read_csv(Path(source_artifacts["trades_csv"]["path"]))
    hourly_rows = _read_csv(Path(source_artifacts["hourly_ledger_csv"]["path"]))
    monthly_rows = _read_csv(Path(source_artifacts["monthly_economics_csv"]["path"]))
    equity_rows = _equity_rows(hourly_rows)
    progress_rows, progress_fields = _progress_rows(previous_progress_path, report)
    _clean_output(output_dir)
    charts = output_dir / "charts"
    _write_csv(output_dir / "treatments.csv", treatment_rows)
    _write_csv(output_dir / "models.csv", model_rows)
    _write_csv(output_dir / "predictive-rank.csv", rank_rows)
    _write_csv(output_dir / "gates.csv", gate_rows)
    _write_csv(output_dir / "ai-factors.csv", ai_rows)
    _write_csv(output_dir / "trades.csv", trade_rows)
    _write_csv(output_dir / "hourly-ledger.csv", hourly_rows)
    _write_csv(output_dir / "monthly-economics.csv", monthly_rows)
    _write_csv(output_dir / "equity.csv", equity_rows)
    _write_csv(
        output_dir / "progress.csv",
        [
            {field: row.get(field, "") for field in progress_fields}
            for row in progress_rows
        ],
    )
    _write_text(charts / "model-skill.svg", _model_skill_svg(model_rows))
    _write_text(charts / "economics.svg", _economics_svg(treatment_rows))
    _write_text(charts / "activity.svg", _activity_svg(treatment_rows))
    _write_text(charts / "ai-uplift.svg", _ai_uplift_svg(report))
    _write_text(charts / "september-equity.svg", _equity_svg(equity_rows))
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))
    _write_text(output_dir / "README.md", _readme(report))
    write_json_atomic(output_dir / "screen.json", report, indent=2, sort_keys=True)
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "published_at_utc": report["generated_at_utc"],
        "publisher_path": "tools/publish_round55_bounded_alpha.py",
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
            "treatments": len(TREATMENTS),
            "model_artifacts": 18,
            "side_models": 36,
            "accepted_ai_factor_programs": report["ai_factor_research"][
                "runtime_accepted_programs"
            ],
            "passed_treatments": 0,
            "ai_uplift_gate_passed": False,
            "forbidden_existing_rows_loaded": False,
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
            r"E:\SimpleAITradingData\round55-bounded-alpha-20260714-v2\report.json"
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-055-bounded-alpha-lightgbm-ai-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-055-bounded-alpha-execution-binding.json",
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
