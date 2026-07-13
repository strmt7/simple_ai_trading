"""Publish verified Round 42 second-flow execution evidence and charts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.publish_cross_asset_cost_aware_ai_ablation import (  # noqa: E402
    _research_progress_svg,
    _svg_start,
)
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_second_flow_execution_overlay import (  # noqa: E402
    BINDING_SCHEMA,
    DESIGN_SCHEMA,
    REPORT_SCHEMA,
    _canonical_sha256,
)


PUBLICATION_SCHEMA = "second-flow-execution-overlay-publication-v1"
ROUND = 42
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
COMPARATORS = (
    "immediate_base",
    "veto_only_base",
    "timing_base",
    "timing_stress",
    "oracle_best_delay_diagnostic",
)
DIAGNOSTIC_FIELDS = (
    "fold",
    "head",
    "role",
    "rows",
    "positive_rows",
    "positive_fraction",
    "roc_auc",
    "brier_score",
    "log_loss",
    "mean_actual_net_bps",
    "mean_predicted_net_bps",
    "mean_absolute_error_bps",
    "root_mean_squared_error_bps",
    "pearson_information_coefficient",
    "spearman_information_coefficient",
    "target_quantile",
    "empirical_below_prediction_fraction",
    "pinball_loss_bps",
    "mean_prediction_bps",
)


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root is not an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_identity(value: Mapping[str, object], field: str, label: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field, ""))
    if len(claimed) != 64 or _canonical_sha256(canonical) != claimed:
        raise ValueError(f"{label} canonical identity is invalid")
    return claimed


def _validated_source(
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], str, str]:
    design = _read_object(design_path, "Round 42 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 42 design")
    binding = _read_object(binding_path, "Round 42 binding")
    binding_sha = _canonical_identity(binding, "binding_sha256", "Round 42 binding")
    report = _read_object(evidence_root / "report.json", "Round 42 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 42 report"
    )
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("status") != "rejected"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
    ):
        raise ValueError("Round 42 evidence lineage is invalid")
    _validate_tree(report)
    dataset = report.get("dataset")
    source = report.get("source_evidence")
    second = source.get("second_flow") if isinstance(source, Mapping) else None
    models = report.get("model_artifacts")
    folds = report.get("folds")
    aggregate = report.get("aggregate")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("proposals") != 703
        or dataset.get("option_rows") != 2_812
        or dataset.get("feature_count") != 251
        or dataset.get("features_dtype") != "float32"
        or dataset.get("features_bytes") != 2_823_248
        or dataset.get("base_round_trip_charge_bps") != 12.0
        or dataset.get("stress_round_trip_charge_bps") != 16.0
        or dataset.get("persistent_feature_prediction_or_raw_trade_copy_created")
        is not False
        or not isinstance(second, Mapping)
        or second.get("certificate_sha256")
        != "e06c6c2894b0a1c5e2370a45c8cb0a7910fba62b4ea61f1cf973d4a06f59d1f5"
        or second.get("rows_total") != 1_814_400
        or second.get("raw_aggregate_trade_rows_total") != 15_475_296
        or not isinstance(models, list)
        or len(models) != 6
        or any(model.get("backend_kind") != "opencl" for model in models)
        or any(model.get("reload_max_abs_prediction_error") != 0.0 for model in models)
        or not isinstance(folds, list)
        or len(folds) != 2
        or any(len(fold.get("threshold_trace", [])) != 27 for fold in folds)
        or any(fold.get("selected_threshold") is not None for fold in folds)
        or not isinstance(aggregate, Mapping)
        or report.get("pilot_gate_passed") is not False
        or report.get("ai_ablation", {}).get("cases") != 0
        or report.get("selection_contaminated") is not True
        or report.get("development_only") is not True
    ):
        raise ValueError("Round 42 source or model evidence drifted")
    immediate = aggregate["immediate_base"]
    oracle = aggregate["oracle_best_delay_diagnostic"]
    if (
        immediate.get("total_trades") != 41
        or abs(float(immediate["mean_net_bps"]) - (-15.281501421114294)) > 1e-12
        or abs(float(immediate["profit_factor"]) - 0.36029789364433235) > 1e-12
        or oracle.get("total_trades") != 40
        or abs(float(oracle["mean_net_bps"]) - (-14.279364039376379)) > 1e-12
        or abs(
            float(oracle["mean_increment_over_matched_immediate_bps"])
            - 5.723640344664455
        )
        > 1e-12
        or any(
            aggregate[name]["total_trades"] != 0
            for name in ("timing_base", "timing_stress", "veto_only_base")
        )
    ):
        raise ValueError("Round 42 comparator economics drifted")
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "roi_claim",
        "drawdown_claim",
        "leverage_applied",
        "ai_uplift_claim",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Round 42 report unexpectedly claims {field}")
    return report, report_sha, binding_sha


def _comparator_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in report["folds"]:
        for name in COMPARATORS:
            evidence = fold["evaluation"][name]
            metrics = evidence["metrics"]
            rows.append(
                {
                    "scope": "evaluation_fold",
                    "fold": fold["fold_id"],
                    "comparator": name,
                    **metrics,
                    "role_proposals": evidence["role_proposals"],
                    "eligible_proposals": evidence["eligible_proposals"],
                    "vetoed_proposals": evidence["vetoed_proposals"],
                    "veto_fraction": evidence["veto_fraction"],
                    "overlap_rejections": evidence["overlap_rejections"],
                    "capacity_rejections": evidence["capacity_rejections"],
                }
            )
    for name in COMPARATORS:
        rows.append(
            {
                "scope": "aggregate_evaluation",
                "fold": "2024-06-06..2024-06-07",
                "comparator": name,
                **report["aggregate"][name],
                "role_proposals": None,
                "eligible_proposals": None,
                "vetoed_proposals": None,
                "veto_fraction": None,
                "overlap_rejections": None,
                "capacity_rejections": None,
            }
        )
    return rows


def _diagnostic_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in report["folds"]:
        for head, roles in fold["model_diagnostics"].items():
            for role, metrics in roles.items():
                rows.append(
                    {
                        "fold": fold["fold_id"],
                        "head": head,
                        "role": role,
                        **metrics,
                    }
                )
    return rows


def _threshold_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in report["folds"]:
        for cell in fold["threshold_trace"]:
            base = cell["timing_base"]
            stress = cell["timing_stress"]
            rows.append(
                {
                    "fold": fold["fold_id"],
                    "positive_probability_threshold": cell[
                        "positive_probability_threshold"
                    ],
                    "expected_net_bps_threshold": cell["expected_net_bps_threshold"],
                    "lower_quartile_bps_threshold": cell[
                        "lower_quartile_bps_threshold"
                    ],
                    "support_passed": cell["support_passed"],
                    "economic_gate_passed": cell["economic_gate_passed"],
                    "base_eligible_proposals": base["eligible_proposals"],
                    "base_veto_fraction": base["veto_fraction"],
                    "base_overlap_rejections": base["overlap_rejections"],
                    "base_capacity_rejections": base["capacity_rejections"],
                    **{f"base_{key}": value for key, value in base["metrics"].items()},
                    **{
                        f"stress_{key}": value
                        for key, value in stress["metrics"].items()
                    },
                }
            )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            key: value
            for key, value in model.items()
            if key not in {"path", "top_feature_gain"}
        }
        for model in report["model_artifacts"]
    ]


def _proposal_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "utc_day": item["utc_day"],
            "total": item["total"],
            **{
                f"{symbol.lower()}_proposals": item["by_symbol"][symbol]
                for symbol in SYMBOLS
            },
        }
        for item in report["dataset"]["proposal_counts_by_utc_day"]
    ]


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "symbol": item["symbol"],
            "second_rows": item["rows"],
            "first_open_time_ms": item["first_open_time_ms"],
            "last_open_time_ms": item["last_open_time_ms"],
            "gap_count": item["gap_count"],
            "zero_trade_seconds": item["zero_trade_seconds"],
            "raw_aggregate_trade_rows": item["raw_aggregate_trades"]["rows"],
            "active_trade_seconds": item["raw_aggregate_trades"]["active_seconds"],
            "archive_files": len(item["archives"]),
            "stream_sha256": item["stream_sha256"],
            "archive_manifest_sha256": item["archive_manifest_sha256"],
        }
        for item in report["source_evidence"]["second_flow"]["symbols"]
    ]


def _progress_rows(
    path: Path, report: Mapping[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed = [int(row["round"]) for row in rows]
    if not fields or observed not in (list(range(1, 42)), list(range(1, 43))):
        raise ValueError("Round 42 prior progress history is invalid")
    rows = [row for row in rows if int(row["round"]) != ROUND]
    immediate = report["aggregate"]["immediate_base"]
    row = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "one-second taker-flow execution timing overlay",
            "periods": "2024-06-01..2024-06-07; eval 2024-06-06..07",
            "selection_contaminated": True,
            "horizon_seconds": 1_800,
            "feature_set": "Round 41 primary + 251 option-conditional flow features",
            "risk_level": "consumed development only; no policy",
            "selected_signals": report["dataset"]["proposals"],
            "executable_trades": 0,
            "mean_gross_bps": float(immediate["mean_net_bps"]) + 12.0,
            "mean_net_bps": immediate["mean_net_bps"],
            "status": "rejected",
            "source_file": "verified Round 42 second-flow report; immediate comparator",
            "best_model_id": "round42_second_flow_execution_overlay",
            "daily_model_fits": 6,
            "calibration_threshold_traces": 54,
            "accepted_thresholds": 0,
            "ensemble_models": 6,
            "development_consumed": True,
        }
    )
    rows.append(row)
    return fields, rows


def _economics_svg(report: Mapping[str, object]) -> str:
    groups: list[tuple[str, Mapping[str, object], Mapping[str, object]]] = []
    for fold in report["folds"]:
        groups.append(
            (
                fold["schedule"]["evaluation_day"],
                fold["evaluation"]["immediate_base"]["metrics"],
                fold["evaluation"]["oracle_best_delay_diagnostic"]["metrics"],
            )
        )
    groups.append(
        (
            "aggregate",
            report["aggregate"]["immediate_base"],
            report["aggregate"]["oracle_best_delay_diagnostic"],
        )
    )
    width, height = 1480, 650
    left, right, top, chart_height = 120, 70, 150, 330
    chart_width = width - left - right
    low, high = -30.0, 5.0

    def y(value: float) -> float:
        return top + chart_height * (high - value) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Even the post-outcome best delay remained negative after and before charge",
        "Mean proposal-side utility; oracle delay is a prohibited post-outcome upper-bound diagnostic, not a strategy.",
    )
    for tick in (-30, -20, -10, 0):
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d} bps</text>'
        )
    zero = y(0.0)
    for index, (label, immediate, oracle) in enumerate(groups):
        center = left + chart_width * (index + 0.5) / len(groups)
        for offset, metrics, color, name in (
            (-54, immediate, "#b42318", "immediate"),
            (54, oracle, "#7b559c", "oracle delay"),
        ):
            value = float(metrics["mean_net_bps"])
            value_y = y(value)
            lines.append(
                f'<rect x="{center + offset - 38:.1f}" y="{zero:.1f}" width="76" height="{value_y - zero:.1f}" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{center + offset:.1f}" y="{value_y + 22:.1f}" text-anchor="middle" class="value">{value:+.2f}; n={int(metrics["total_trades"])}</text>'
            )
            lines.append(
                f'<text x="{center + offset:.1f}" y="{top + chart_height + 48:.1f}" text-anchor="middle" class="note">{html.escape(name)}</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 82:.1f}" text-anchor="middle" class="label">{html.escape(label)}</text>'
        )
    lines.append(
        '<text x="120" y="610" class="note">Aggregate immediate: -15.282 net bps (-3.282 after adding back the fixed 12 bps charge). Oracle: -14.279 (-2.279 before fixed charge). No threshold executed.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _diagnostics_svg(report: Mapping[str, object]) -> str:
    folds = report["folds"]
    panels = (
        (
            "positive_net_probability",
            "roc_auc",
            "Binary profitability ROC AUC",
            0.40,
            0.65,
            0.50,
        ),
        (
            "robust_expected_net_utility",
            "spearman_information_coefficient",
            "Expected-utility Spearman IC",
            -0.15,
            0.35,
            0.0,
        ),
        (
            "lower_quartile_net_utility",
            "empirical_below_prediction_fraction",
            "Lower-quartile empirical coverage",
            0.0,
            0.50,
            0.25,
        ),
    )
    width, height = 1480, 920
    left, right, top, panel_height, gap = 150, 70, 145, 165, 78
    chart_width = width - left - right
    lines = _svg_start(
        width,
        height,
        "The timing heads did not establish stable predictive utility",
        "Calibration and untouched evaluation diagnostics for the two causal folds; reference lines show chance, zero IC, and target quantile coverage.",
    )
    colors = {"threshold_calibration": "#0f766e", "evaluation": "#2563eb"}
    for panel_index, (head, metric, label, low, high, reference) in enumerate(panels):
        panel_top = top + panel_index * (panel_height + gap)

        def y(value: float) -> float:
            return panel_top + panel_height * (high - value) / (high - low)

        lines.append(
            f'<text x="{left}" y="{panel_top - 18}" class="label">{html.escape(label)}</text>'
        )
        for tick in (low, reference, high):
            py = y(tick)
            lines.append(
                f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == reference else "grid"}"/>'
            )
            lines.append(
                f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+.2f}</text>'
            )
        for fold_index, fold in enumerate(folds):
            center = left + chart_width * (fold_index + 0.5) / len(folds)
            for role_index, role in enumerate(("threshold_calibration", "evaluation")):
                value = float(fold["model_diagnostics"][head][role][metric])
                x = center + (-50 if role_index == 0 else 50)
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y(value):.1f}" r="8" fill="{colors[role]}"/>'
                )
                lines.append(
                    f'<text x="{x:.1f}" y="{y(value) - 14:.1f}" text-anchor="middle" class="value">{value:.3f}</text>'
                )
            lines.append(
                f'<text x="{center:.1f}" y="{panel_top + panel_height + 27:.1f}" text-anchor="middle" class="axis">eval {fold["schedule"]["evaluation_day"]}</text>'
            )
    lines.append(
        '<circle cx="1050" cy="885" r="7" fill="#0f766e"/><text x="1068" y="890" class="note">threshold calibration</text><circle cx="1260" cy="885" r="7" fill="#2563eb"/><text x="1278" y="890" class="note">evaluation</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _coverage_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1480, 650
    left, right, top, chart_height = 120, 80, 140, 330
    chart_width = width - left - right
    high = max(int(row["total"]) for row in rows) * 1.12
    colors = (
        ("btcusdt_proposals", "#2563eb", "BTCUSDT"),
        ("ethusdt_proposals", "#0f766e", "ETHUSDT"),
        ("solusdt_proposals", "#b45309", "SOLUSDT"),
    )

    def y(value: float) -> float:
        return top + chart_height * (high - value) / high

    lines = _svg_start(
        width,
        height,
        "Primary proposal coverage was sparse and materially unbalanced",
        "Frozen Round 41 margin proposals by UTC day; counts are opportunities before overlay veto, overlap, or daily capacity.",
    )
    for tick in (0, 50, 100, 150):
        if tick > high:
            continue
        py = y(float(tick))
        lines.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{left + chart_width}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick}</text>'
        )
    bar_width = chart_width / len(rows) * 0.58
    for index, row in enumerate(rows):
        center = left + chart_width * (index + 0.5) / len(rows)
        cumulative = 0
        for key, color, _ in colors:
            value = int(row[key])
            y_top = y(cumulative + value)
            y_bottom = y(cumulative)
            lines.append(
                f'<rect x="{center - bar_width / 2:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" height="{y_bottom - y_top:.1f}" fill="{color}"/>'
            )
            cumulative += value
        lines.append(
            f'<text x="{center:.1f}" y="{y(cumulative) - 12:.1f}" text-anchor="middle" class="value">{cumulative}</text>'
        )
        lines.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 29:.1f}" text-anchor="middle" class="axis">{html.escape(str(row["utc_day"])[5:])}</text>'
        )
    for index, (_, color, label) in enumerate(colors):
        x = 120 + index * 180
        lines.append(
            f'<rect x="{x}" y="570" width="16" height="16" fill="{color}"/><text x="{x + 25}" y="583" class="note">{label}</text>'
        )
    lines.append(
        '<text x="1410" y="583" text-anchor="end" class="note">Seven-day totals: BTC 294, ETH 359, SOL 50. This is not diversified action evidence.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _readme(report: Mapping[str, object]) -> str:
    immediate = report["aggregate"]["immediate_base"]
    oracle = report["aggregate"]["oracle_best_delay_diagnostic"]
    runtime = report["runtime_evidence"]
    return f"""# Round 42: one-second execution overlay rejected

**Second-level taker flow did not rescue the frozen primary signal.** The verified source contains 1,814,400 contiguous one-second bars derived from 15,475,296 checksum-backed Binance USD-M aggregate trades. Six OpenCL LightGBM heads reloaded exactly, but no calibration cell admitted an action.

| Evidence | Verified result |
| --- | ---: |
| Source / evaluation span | Binance USD-M 1s / 2024-06-06 to 2024-06-07 UTC |
| Frozen proposals / delay options | {report["dataset"]["proposals"]} / {report["dataset"]["option_rows"]} |
| Features / GPU artifacts / threshold cells | {report["dataset"]["feature_count"]} / {len(report["model_artifacts"])} / 54 |
| Immediate comparator | {immediate["total_trades"]} trades; {float(immediate["mean_net_bps"]):+.3f} net bps; PF {float(immediate["profit_factor"]):.3f} |
| Post-outcome best-delay diagnostic | {oracle["total_trades"]} trades; {float(oracle["mean_net_bps"]):+.3f} net bps; +{float(oracle["mean_increment_over_matched_immediate_bps"]):.3f} bps vs matched immediate |
| Timing-overlay actions / selected folds | 0 / 0 of 2 |
| AI cases / AI models run | 0 / 0; seconds-loop AI prohibited by design |
| Compute / runtime / peak working set | {report["backend"]["device"]} / {runtime["elapsed_seconds"]:.1f}s / {runtime["memory"]["peak_working_set_bytes"] / 1024**3:.2f} GiB |
| Trading authority / leverage | none / none |

![Execution economics](charts/execution-economics.svg)

![Model diagnostics](charts/model-diagnostics.svg)

![Proposal coverage](charts/proposal-coverage.svg)

![Research progress](charts/research-progress.svg)

Zero overlay trades means the frozen hurdle rejected every option, not that proposals disappeared: 703 primary opportunities were evaluated. Adding back the fixed 12 bps charge leaves the immediate comparator at `{float(immediate["mean_net_bps"]) + 12.0:+.3f}` bps and the prohibited best-delay diagnostic at `{float(oracle["mean_net_bps"]) + 12.0:+.3f}` bps on their respective capacity-selected sets. The primary direction signal was therefore negative even before that fixed charge; more execution tuning is not justified.

This seven-day, repeatedly consumed development pilot is not multi-year evidence, an equity curve, ROI, portfolio drawdown, execution quality, AI uplift, or profitability. Larger second-level acquisition was not authorized.

Data: [comparators.csv](comparators.csv) | [diagnostics.csv](diagnostics.csv) | [thresholds.csv](thresholds.csv) | [proposals.csv](proposals.csv) | [models.csv](models.csv) | [sources.csv](sources.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
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
    report, source_report_sha, binding_sha = _validated_source(
        evidence_root, design_path, binding_path
    )
    comparators = _comparator_rows(report)
    diagnostics = _diagnostic_rows(report)
    thresholds = _threshold_rows(report)
    proposals = _proposal_rows(report)
    models = _model_rows(report)
    sources = _source_rows(report)
    progress_fields, progress = _progress_rows(prior_progress_path, report)
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "comparators.csv",
        output_dir / "diagnostics.csv",
        output_dir / "thresholds.csv",
        output_dir / "proposals.csv",
        output_dir / "models.csv",
        output_dir / "sources.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "execution-economics.svg",
        charts / "model-diagnostics.svg",
        charts / "proposal-coverage.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    output_dir.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "comparators.csv", comparators)
    _write_csv(output_dir / "diagnostics.csv", diagnostics, fields=DIAGNOSTIC_FIELDS)
    _write_csv(output_dir / "thresholds.csv", thresholds)
    _write_csv(output_dir / "proposals.csv", proposals)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "progress.csv", progress, fields=progress_fields)
    _write_text(
        output_dir / "screen.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "execution-economics.svg", _economics_svg(report))
    _write_text(charts / "model-diagnostics.svg", _diagnostics_svg(report))
    _write_text(charts / "proposal-coverage.svg", _coverage_svg(proposals))
    _write_text(charts / "research-progress.svg", _research_progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "artifact_class": "second_flow_execution_graph_data",
        "round": ROUND,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _file_sha256(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "second_rows": report["source_evidence"]["second_flow"]["rows_total"],
        "raw_aggregate_trade_rows": report["source_evidence"]["second_flow"][
            "raw_aggregate_trade_rows_total"
        ],
        "proposal_count": report["dataset"]["proposals"],
        "option_row_count": report["dataset"]["option_rows"],
        "feature_count": report["dataset"]["feature_count"],
        "gpu_model_artifact_count": len(models),
        "threshold_cell_count": len(thresholds),
        "selected_threshold_fold_count": 0,
        "overlay_trade_count": report["aggregate"]["timing_base"]["total_trades"],
        "immediate_comparator_trade_count": report["aggregate"]["immediate_base"][
            "total_trades"
        ],
        "immediate_comparator_mean_net_bps": report["aggregate"]["immediate_base"][
            "mean_net_bps"
        ],
        "oracle_diagnostic_mean_net_bps": report["aggregate"][
            "oracle_best_delay_diagnostic"
        ]["mean_net_bps"],
        "ai_case_count": 0,
        "selection_contaminated": True,
        "development_only": True,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "roi_claim": False,
        "drawdown_claim": False,
        "leverage_applied": False,
        "ai_uplift_claim": False,
        "artifact_integrity": [_artifact(path, output_dir) for path in artifact_paths],
        "publication_sha256": "PENDING",
    }
    canonical = dict(publication)
    canonical.pop("publication_sha256")
    publication["publication_sha256"] = _canonical_sha256(canonical)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs/model-research/action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-042-second-flow-execution-overlay-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-042-second-flow-execution-binding.json",
    )
    parser.add_argument(
        "--prior-progress", type=Path, default=research / "latest/progress.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=research / "latest")
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
