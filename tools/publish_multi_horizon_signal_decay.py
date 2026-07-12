"""Publish verified Round 36 signal-decay evidence and static SVG charts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
import html
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tools.publish_daily_walkforward_screen import _progress_svg  # noqa: E402
from tools.publish_selective_action_viability import (  # noqa: E402
    _artifact,
    _read_object,
    _validate_tree,
    _write_csv,
    _write_text,
)
from tools.run_multi_horizon_signal_decay import (  # noqa: E402
    REPORT_SCHEMA_VERSION,
    _EXPECTED_HORIZONS,
    _EXPECTED_SIGNALS,
    _canonical_sha256,
    _is_sha256,
    _sha256_file,
    load_signal_decay_binding,
    load_signal_decay_design,
)


PUBLICATION_SCHEMA_VERSION = "multi-horizon-signal-decay-publication-v1"
_COLORS = {
    "l1_imbalance": "#0f766e",
    "microprice_offset_bps": "#2563eb",
    "normalized_ofi": "#b45309",
    "ofi_10s_mean": "#7c3aed",
}
_LABELS = {
    "l1_imbalance": "L1 imbalance",
    "microprice_offset_bps": "Microprice offset",
    "normalized_ofi": "Normalized OFI",
    "ofi_10s_mean": "OFI 10s mean",
    "ofi_60s_mean": "OFI 60s mean",
    "ofi_300s_mean": "OFI 300s mean",
    "trade_imbalance": "Trade imbalance",
    "trade_imbalance_10s_mean": "Trade imbalance 10s",
    "trade_imbalance_60s_mean": "Trade imbalance 60s",
    "trade_imbalance_300s_mean": "Trade imbalance 300s",
    "signed_pressure_to_opposing_depth_10s": "Depth-normalized flow 10s",
    "signed_pressure_to_opposing_depth_60s": "Depth-normalized flow 60s",
    "signed_pressure_to_opposing_depth_300s": "Depth-normalized flow 300s",
}


def _validated_source(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
) -> tuple[dict[str, object], dict[str, object], str, str]:
    design, design_sha = load_signal_decay_design(design_path)
    binding, binding_sha = load_signal_decay_binding(
        binding_path,
        design_path=design_path,
        design_sha256=design_sha,
    )
    report_path = evidence_root / "report.json"
    report = _read_object(report_path, label="Round 36 signal-decay report")
    canonical = dict(report)
    report_sha = canonical.pop("report_canonical_sha256", None)
    implementation = binding.get("implementation")
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("round") != 36
        or report.get("status") != "diagnostic_complete_no_authority"
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or not isinstance(implementation, Mapping)
        or report.get("implementation_commit") != implementation.get("commit")
        or not _is_sha256(report_sha)
        or _canonical_sha256(canonical) != report_sha
    ):
        raise ValueError("Round 36 source report identity is invalid")
    _validate_tree(report)
    source = report.get("source_evidence")
    access = report.get("stage_access")
    quote_evidence = report.get("zero_latency_quote_evidence")
    completeness = report.get("completeness")
    runtime = report.get("runtime_evidence")
    memory = runtime.get("memory") if isinstance(runtime, Mapping) else None
    results = report.get("signal_horizon_results")
    if (
        not isinstance(source, Mapping)
        or source.get("corpus_certificate_sha256")
        != "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
        or source.get("barrier_targets_sha256")
        != "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
        or source.get("cache_key")
        != "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
        or source.get("cache_state") != "hit"
        or source.get("dataset_rows") != 877_894
        or source.get("event_rows") != 230_941
        or source.get("metric_event_rows") != 28_845
        or not isinstance(access, Mapping)
        or access.get("calibration_metrics") is not True
        or any(
            access.get(field) is not False
            for field in (
                "train_prediction_or_metrics",
                "early_stop_prediction_or_metrics",
                "policy_prediction_or_metrics",
                "development_prediction_or_metrics",
                "distant_confirmation_source_materialized",
                "distant_confirmation_prediction_or_metrics",
            )
        )
        or not isinstance(quote_evidence, Mapping)
        or quote_evidence.get("requested_timestamps") != 80_307
        or quote_evidence.get("valid_timestamps") != 80_307
        or quote_evidence.get("invalid_timestamps") != 0
        or not isinstance(completeness, Mapping)
        or completeness.get("all_cells_complete") is not True
        or completeness.get("reported_signal_horizon_cells") != 91
        or completeness.get("reported_daily_records") != 455
        or completeness.get("reported_regime_records") != 819
        or completeness.get("placebo_replicates_per_cell") != 200
        or not isinstance(runtime, Mapping)
        or not isinstance(memory, Mapping)
        or memory.get("source") != "windows_process_memory_counters"
        or not 0 < int(memory.get("peak_working_set_bytes") or 0) <= 12 * 1024**3
        or runtime.get("gpu_training_used") is not False
        or runtime.get("persistent_duplicate_dataset_or_quote_archive_created")
        is not False
        or not isinstance(results, list)
        or len(results) != 91
    ):
        raise ValueError("Round 36 source, access, completeness, or runtime drifted")
    expected_order = [
        (signal, horizon)
        for horizon in _EXPECTED_HORIZONS
        for signal in _EXPECTED_SIGNALS
    ]
    observed_order = [
        (str(item.get("signal")), int(item.get("horizon_seconds") or 0))
        for item in results
        if isinstance(item, Mapping)
    ]
    if observed_order != expected_order:
        raise ValueError("Round 36 signal-horizon ordering drifted")
    positive_regimes = 0
    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("Round 36 result cell is invalid")
        direction = item.get("direction")
        cost = item.get("cost_decomposition")
        robustness = item.get("nonoverlapping_robustness")
        tails = item.get("ranked_event_outcomes")
        regimes = item.get("regime_metrics")
        placebo = item.get("placebo")
        if (
            not isinstance(direction, Mapping)
            or not isinstance(cost, Mapping)
            or not isinstance(robustness, Mapping)
            or not isinstance(robustness.get("cost"), Mapping)
            or not isinstance(tails, list)
            or len(tails) != 3
            or not isinstance(regimes, list)
            or len(regimes) != 9
            or not isinstance(placebo, Mapping)
            or item.get("nonfinite_signal_rows") != 0
            or cost.get("mean_delayed_net_return_bps") is None
            or float(cost["mean_delayed_net_return_bps"]) >= 0.0
            or robustness["cost"].get("mean_delayed_net_return_bps") is None
            or float(robustness["cost"]["mean_delayed_net_return_bps"]) >= 0.0
            or any(
                tail.get("mean_delayed_net_return_bps") is None
                or float(tail["mean_delayed_net_return_bps"]) >= 0.0
                for tail in tails
            )
            or placebo.get("formal_multiple_testing_significance_claim") is not False
            or any(
                item.get(field) is not False
                for field in (
                    "model_candidate",
                    "trading_authority",
                    "execution_claim",
                    "profitability_claim",
                    "portfolio_claim",
                    "leverage_applied",
                )
            )
        ):
            raise ValueError("Round 36 result economics or authority drifted")
        for regime in regimes:
            if not isinstance(regime, Mapping):
                raise ValueError("Round 36 regime record is invalid")
            regime_cost = regime.get("cost")
            if (
                isinstance(regime_cost, Mapping)
                and float(regime_cost["mean_delayed_net_return_bps"]) >= 0.0
            ):
                positive_regimes += 1
    if positive_regimes != 0:
        raise ValueError("Round 36 contains an unexpected positive regime result")
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
        "model_trained",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Round 36 unexpectedly grants {field}")
    if report.get("model_candidate") is not None:
        raise ValueError("Round 36 unexpectedly creates a model candidate")
    return design, report, str(report_sha), binding_sha


def _signal_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report["signal_horizon_results"]:
        direction = item["direction"]
        daily = item["daily_summary"]
        cost = item["cost_decomposition"]
        robustness = item["nonoverlapping_robustness"]
        placebo = item["placebo"]
        rows.append(
            {
                "signal": item["signal"],
                "family": item["family"],
                "horizon_seconds": item["horizon_seconds"],
                "direction_rows": direction["rows"],
                "weighted_roc_auc": direction["weighted_roc_auc"],
                "unweighted_roc_auc": direction["unweighted_roc_auc"],
                "spearman_information_coefficient": direction[
                    "spearman_information_coefficient"
                ],
                "weighted_direction_accuracy": direction["weighted_direction_accuracy"],
                "daily_auc_minimum": daily["weighted_auc_minimum"],
                "daily_auc_median": daily["weighted_auc_median"],
                "daily_auc_standard_deviation": daily[
                    "weighted_auc_standard_deviation"
                ],
                "days_above_chance": daily["days_above_chance"],
                "routed_rows": cost["routed_rows"],
                "delayed_l1_eligible_rows": cost["delayed_l1_eligible_rows"],
                "mean_signal_aligned_gross_midquote_return_bps": cost[
                    "mean_signal_aligned_gross_midquote_return_bps"
                ],
                "mean_cross_spread_gross_return_bps": cost[
                    "mean_cross_spread_gross_return_bps"
                ],
                "mean_spread_crossing_cost_bps": cost["mean_spread_crossing_cost_bps"],
                "mean_fee_and_slippage_cost_bps": cost[
                    "mean_fee_and_slippage_cost_bps"
                ],
                "mean_historical_latency_drag_bps": cost[
                    "mean_historical_latency_drag_bps"
                ],
                "mean_delayed_net_return_bps": cost["mean_delayed_net_return_bps"],
                "delayed_net_positive_rate": cost["delayed_net_positive_rate"],
                "nonoverlapping_rows": robustness["selected_rows"],
                "nonoverlapping_weighted_roc_auc": robustness["direction"][
                    "weighted_roc_auc"
                ],
                "nonoverlapping_mean_delayed_net_return_bps": robustness["cost"][
                    "mean_delayed_net_return_bps"
                ],
                "placebo_observed_rank_descending": placebo["observed_rank_descending"],
                "placebo_exceedance_fraction": placebo[
                    "one_sided_empirical_exceedance_fraction"
                ],
                "placebo_mean_auc": placebo["placebo_mean"],
                "placebo_auc_standard_deviation": placebo["placebo_standard_deviation"],
                "placebo_auc_95_low": placebo["placebo_95_percent_interval"][0],
                "placebo_auc_95_high": placebo["placebo_95_percent_interval"][1],
                "nonfinite_signal_rows": item["nonfinite_signal_rows"],
            }
        )
    return rows


def _daily_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report["signal_horizon_results"]:
        for daily in item["daily_direction"]:
            rows.append(
                {
                    "signal": item["signal"],
                    "horizon_seconds": item["horizon_seconds"],
                    "date": daily["utc_date"],
                    "rows": daily["rows"],
                    "weighted_roc_auc": daily["weighted_roc_auc"],
                    "unweighted_roc_auc": daily["unweighted_roc_auc"],
                    "spearman_information_coefficient": daily[
                        "spearman_information_coefficient"
                    ],
                    "weighted_direction_accuracy": daily["weighted_direction_accuracy"],
                }
            )
    return rows


def _regime_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report["signal_horizon_results"]:
        for regime in item["regime_metrics"]:
            direction = regime.get("direction") or {}
            cost = regime.get("cost") or {}
            rows.append(
                {
                    "signal": item["signal"],
                    "horizon_seconds": item["horizon_seconds"],
                    "regime": regime["regime"],
                    "band": regime["band"],
                    "support_rows": regime["support_rows"],
                    "metrics_reported": regime["metrics_reported"],
                    "weighted_roc_auc": direction.get("weighted_roc_auc", ""),
                    "spearman_information_coefficient": direction.get(
                        "spearman_information_coefficient", ""
                    ),
                    "mean_signal_aligned_gross_midquote_return_bps": cost.get(
                        "mean_signal_aligned_gross_midquote_return_bps", ""
                    ),
                    "mean_delayed_net_return_bps": cost.get(
                        "mean_delayed_net_return_bps", ""
                    ),
                }
            )
    return rows


def _ranked_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report["signal_horizon_results"]:
        for tail in item["ranked_event_outcomes"]:
            rows.append(
                {
                    "signal": item["signal"],
                    "horizon_seconds": item["horizon_seconds"],
                    **tail,
                }
            )
    return rows


def _decay_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in report["signal_decay_summary"]:
        for position, horizon in enumerate(item["horizons_seconds"]):
            rows.append(
                {
                    "signal": item["signal"],
                    "horizon_seconds": horizon,
                    "weighted_roc_auc": item["weighted_auc"][position],
                    "weighted_auc_minus_chance": item["weighted_auc_minus_chance"][
                        position
                    ],
                    "spearman_information_coefficient": item[
                        "spearman_information_coefficient"
                    ][position],
                    "earliest_peak_horizon_seconds": item[
                        "earliest_peak_horizon_seconds"
                    ],
                    "earliest_peak_weighted_auc": item["earliest_peak_weighted_auc"],
                    "half_life_status": item["half_life_status"],
                    "half_life_seconds": item["half_life_seconds"],
                }
            )
    return rows


def _support_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for item in report["horizon_support"]:
        rows.append(
            {
                "horizon_seconds": item["horizon_seconds"],
                "rows": item["rows"],
                "first_decision_time_ms": item["first_decision_time_ms"],
                "last_decision_time_ms": item["last_decision_time_ms"],
                **item["exclusion_counts"],
            }
        )
    return rows


def _progress_rows(
    path: Path,
    signal_rows: Sequence[Mapping[str, object]],
    ranked_rows: Sequence[Mapping[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields or [int(row["round"]) for row in rows] != list(range(1, 36)):
        raise ValueError("Round 36 prior progress history is invalid")
    best_auc = max(signal_rows, key=lambda row: float(row["weighted_roc_auc"]))
    best_net = max(
        signal_rows,
        key=lambda row: float(row["mean_delayed_net_return_bps"]),
    )
    best_100 = max(
        (row for row in ranked_rows if int(row["requested_rows"]) == 100),
        key=lambda row: float(row["mean_delayed_net_return_bps"]),
    )
    best_500 = max(
        (row for row in ranked_rows if int(row["requested_rows"]) == 500),
        key=lambda row: float(row["mean_delayed_net_return_bps"]),
    )
    row = {field: "" for field in fields}
    row.update(
        {
            "round": 36,
            "stage": "consumed-data multi-horizon signal decay and cost coverage",
            "periods": "2023-06-21..2023-06-25",
            "selection_contaminated": True,
            "horizon_seconds": "5;15;30;60;120;300;900",
            "feature_set": "13 prespecified l1-tape-causal-v8 signals",
            "risk_level": "research-only; no policy",
            "direction_auc": best_auc["weighted_roc_auc"],
            "spearman_ic": best_auc["spearman_information_coefficient"],
            "selected_signals": 0,
            "executable_trades": 0,
            "mean_gross_bps": max(
                float(item["mean_signal_aligned_gross_midquote_return_bps"])
                for item in signal_rows
            ),
            "mean_net_bps": best_net["mean_delayed_net_return_bps"],
            "status": "rejected",
            "source_file": "verified Round 36 signal-decay report",
            "best_model_id": "none; strongest raw signal l1_imbalance at 5 seconds",
            "best_top_500_exact_after_cost_bps": best_500[
                "mean_delayed_net_return_bps"
            ],
            "after_cost_diagnostic_rows": max(
                int(item["delayed_l1_eligible_rows"]) for item in signal_rows
            ),
            "valid_barrier_rows": 229_000,
            "calibration_eligible_rows": 0,
            "development_consumed": False,
            "top_100_exact_after_cost_bps": best_100["mean_delayed_net_return_bps"],
        }
    )
    rows.append(row)
    return fields, rows


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(description)}</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Segoe UI",Arial,sans-serif;letter-spacing:0}.title{font-size:24px;font-weight:700;fill:#17202a}.subtitle{font-size:14px;fill:#4b5563}.label{font-size:13px;fill:#263238}.value{font-size:13px;font-weight:650;fill:#17202a}.axis{font-size:12px;fill:#64748b}.note{font-size:12px;fill:#6b7280}.grid{stroke:#d8dee7;stroke-width:1}.zero{stroke:#9f1239;stroke-width:2}.chance{stroke:#334155;stroke-width:2;stroke-dasharray:7 5}</style>',
        f'<text x="48" y="42" class="title">{html.escape(title)}</text>',
        f'<text x="48" y="66" class="subtitle">{html.escape(description)}</text>',
    ]


def _decay_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1420, 720
    left, right, top, bottom = 100, 80, 112, 88
    horizons = list(_EXPECTED_HORIZONS)
    x_positions = [
        left + index * (width - left - right) / (len(horizons) - 1)
        for index in range(len(horizons))
    ]
    minimum, maximum = 0.47, 0.62

    def y(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * (height - top - bottom)

    svg = _svg_start(
        width,
        height,
        "Short-horizon direction exists, then decays rapidly",
        "Average-uniqueness-weighted ROC AUC on consumed BTCUSDT dates; no line is a trading policy.",
    )
    for tick in (0.48, 0.50, 0.54, 0.58, 0.62):
        py = y(tick)
        svg.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"chance" if tick == 0.50 else "grid"}"/>'
        )
        svg.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>'
        )
    lookup = {
        (str(row["signal"]), int(row["horizon_seconds"])): float(
            row["weighted_roc_auc"]
        )
        for row in rows
    }
    for signal in _EXPECTED_SIGNALS:
        points = [
            (x_positions[index], y(lookup[(signal, horizon)]))
            for index, horizon in enumerate(horizons)
        ]
        color = _COLORS.get(signal, "#cbd5e1")
        width_value = 4 if signal in _COLORS else 1.5
        opacity = 1.0 if signal in _COLORS else 0.75
        svg.append(
            f'<polyline points="{" ".join(f"{x:.1f},{py:.1f}" for x, py in points)}" fill="none" stroke="{color}" stroke-width="{width_value}" opacity="{opacity}"/>'
        )
        if signal in _COLORS:
            for x, py in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
    for index, horizon in enumerate(horizons):
        svg.append(
            f'<text x="{x_positions[index]:.1f}" y="{height - bottom + 28}" text-anchor="middle" class="axis">{horizon}s</text>'
        )
    for index, signal in enumerate(_COLORS):
        lx = left + index * 250
        svg.append(
            f'<line x1="{lx}" y1="{height - 30}" x2="{lx + 28}" y2="{height - 30}" stroke="{_COLORS[signal]}" stroke-width="4"/><text x="{lx + 38}" y="{height - 25}" class="note">{html.escape(_LABELS[signal])}</text>'
        )
    svg.append(
        f'<text x="{width - right}" y="{height - 25}" text-anchor="end" class="note">Gray lines: remaining prespecified signals.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _cost_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1400, 650
    left, right, top, bottom = 105, 70, 112, 88
    horizons = list(_EXPECTED_HORIZONS)
    minimum, maximum = -13.0, 2.0

    def y(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * (height - top - bottom)

    svg = _svg_start(
        width,
        height,
        "Predictive direction did not cover taker costs",
        "Best descriptive gross and delayed net means at each horizon; no horizon selection is permitted.",
    )
    for tick in (-12, -9, -6, -3, 0):
        py = y(float(tick))
        svg.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        svg.append(
            f'<text x="{left - 14}" y="{py + 4:.1f}" text-anchor="end" class="axis">{tick:+d} bps</text>'
        )
    group_width = (width - left - right) / len(horizons)
    for index, horizon in enumerate(horizons):
        horizon_rows = [row for row in rows if int(row["horizon_seconds"]) == horizon]
        gross = max(
            float(row["mean_signal_aligned_gross_midquote_return_bps"])
            for row in horizon_rows
        )
        net = max(float(row["mean_delayed_net_return_bps"]) for row in horizon_rows)
        center = left + (index + 0.5) * group_width
        for value, offset, color in ((gross, -18, "#0f766e"), (net, 18, "#b91c1c")):
            zero_y = y(0.0)
            value_y = y(value)
            svg.append(
                f'<rect x="{center + offset - 12:.1f}" y="{min(zero_y, value_y):.1f}" width="24" height="{abs(zero_y - value_y):.1f}" fill="{color}"/>'
            )
        svg.append(
            f'<text x="{center:.1f}" y="{height - bottom + 26}" text-anchor="middle" class="axis">{horizon}s</text>'
        )
        svg.append(
            f'<text x="{center - 18:.1f}" y="{y(gross) - 8:.1f}" text-anchor="middle" class="value">{gross:+.2f}</text>'
        )
        svg.append(
            f'<text x="{center + 18:.1f}" y="{y(net) + 18:.1f}" text-anchor="middle" class="value">{net:+.2f}</text>'
        )
    svg.extend(
        [
            f'<rect x="{left}" y="{height - 36}" width="16" height="16" fill="#0f766e"/><text x="{left + 24}" y="{height - 23}" class="note">best signal-aligned gross midquote mean</text>',
            f'<rect x="{left + 350}" y="{height - 36}" width="16" height="16" fill="#b91c1c"/><text x="{left + 374}" y="{height - 23}" class="note">best delayed taker net mean</text>',
            f'<text x="{width - right}" y="{height - 23}" text-anchor="end" class="note">Fee + slippage cost was approximately 12 bps round trip.</text>',
            "</svg>",
        ]
    )
    return "\n".join(svg) + "\n"


def _tails_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1400, 660
    left, right, top, bottom = 105, 70, 112, 90
    horizons = list(_EXPECTED_HORIZONS)
    minimum, maximum = -14.0, 1.0

    def y(value: float) -> float:
        return top + (maximum - value) / (maximum - minimum) * (height - top - bottom)

    svg = _svg_start(
        width,
        height,
        "Every absolute-signal ranked tail remained negative",
        "Least-negative mean delayed taker outcome across signals at each frozen horizon.",
    )
    for tick in (-12, -9, -6, -3, 0):
        py = y(float(tick))
        svg.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
    x_positions = [
        left + index * (width - left - right) / (len(horizons) - 1)
        for index in range(len(horizons))
    ]
    for count, color in ((100, "#b45309"), (500, "#2563eb"), (1000, "#7c3aed")):
        points = []
        for index, horizon in enumerate(horizons):
            candidates = [
                float(row["mean_delayed_net_return_bps"])
                for row in rows
                if int(row["horizon_seconds"]) == horizon
                and int(row["requested_rows"]) == count
            ]
            points.append((x_positions[index], y(max(candidates))))
        svg.append(
            f'<polyline points="{" ".join(f"{x:.1f},{py:.1f}" for x, py in points)}" fill="none" stroke="{color}" stroke-width="4"/>'
        )
        for x, py in points:
            svg.append(f'<circle cx="{x:.1f}" cy="{py:.1f}" r="5" fill="{color}"/>')
    for index, horizon in enumerate(horizons):
        svg.append(
            f'<text x="{x_positions[index]:.1f}" y="{height - bottom + 28}" text-anchor="middle" class="axis">{horizon}s</text>'
        )
    for index, (count, color) in enumerate(
        ((100, "#b45309"), (500, "#2563eb"), (1000, "#7c3aed"))
    ):
        lx = left + index * 190
        svg.append(
            f'<line x1="{lx}" y1="{height - 30}" x2="{lx + 28}" y2="{height - 30}" stroke="{color}" stroke-width="4"/><text x="{lx + 38}" y="{height - 25}" class="note">top {count}</text>'
        )
    svg.append(
        f'<text x="{width - right}" y="{height - 25}" text-anchor="end" class="note">Event outcomes are not executable trades or portfolio returns.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _daily_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 960
    left, top = 330, 130
    dates = sorted({str(row["date"]) for row in rows})
    selected = [row for row in rows if int(row["horizon_seconds"]) == 5]
    lookup = {
        (str(row["signal"]), str(row["date"])): float(row["weighted_roc_auc"])
        for row in selected
    }
    cell_width, cell_height = 210, 56
    svg = _svg_start(
        width,
        height,
        "Five-second direction was stable for the strongest L1 signals",
        "Weighted ROC AUC by UTC date; this heatmap measures direction, not after-cost profitability.",
    )
    for column, date in enumerate(dates):
        x = left + column * cell_width
        svg.append(
            f'<text x="{x + cell_width / 2:.1f}" y="{top - 18}" text-anchor="middle" class="axis">{date}</text>'
        )
    for row_index, signal in enumerate(_EXPECTED_SIGNALS):
        y = top + row_index * cell_height
        svg.append(
            f'<text x="{left - 18}" y="{y + cell_height / 2 + 5:.1f}" text-anchor="end" class="label">{html.escape(_LABELS[signal])}</text>'
        )
        for column, date in enumerate(dates):
            value = lookup[(signal, date)]
            color = (
                "#b7e4c7"
                if value >= 0.55
                else "#d8f3dc"
                if value > 0.50
                else "#fde68a"
                if value >= 0.48
                else "#fecaca"
            )
            x = left + column * cell_width
            svg.append(
                f'<rect x="{x + 4}" y="{y + 4}" width="{cell_width - 8}" height="{cell_height - 8}" rx="4" fill="{color}" stroke="#cbd5e1"/>'
            )
            svg.append(
                f'<text x="{x + cell_width / 2:.1f}" y="{y + cell_height / 2 + 5:.1f}" text-anchor="middle" class="value">{value:.4f}</text>'
            )
    svg.append(
        f'<text x="{left}" y="{height - 28}" class="note">All five dates were already consumed; no cell can be used for policy or regime selection.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _placebo_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1500, 920
    left, right, top, bottom = 330, 70, 112, 70
    selected = [row for row in rows if int(row["horizon_seconds"]) == 5]
    minimum, maximum = 0.47, 0.62

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * (width - left - right)

    svg = _svg_start(
        width,
        height,
        "Observed five-second AUC separated from within-day placebos",
        "Observed weighted AUC versus 200 within-day signal permutations; descriptive only, no multiplicity claim.",
    )
    for tick in (0.48, 0.50, 0.54, 0.58, 0.62):
        px = x(tick)
        svg.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height - bottom}" class="{"chance" if tick == 0.50 else "grid"}"/>'
        )
    for index, row in enumerate(selected):
        y = top + 28 + index * 54
        low = float(row["placebo_auc_95_low"])
        high = float(row["placebo_auc_95_high"])
        mean = float(row["placebo_mean_auc"])
        observed = float(row["weighted_roc_auc"])
        svg.append(
            f'<text x="{left - 18}" y="{y + 5}" text-anchor="end" class="label">{html.escape(_LABELS[str(row["signal"])])}</text>'
        )
        svg.append(
            f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="#94a3b8" stroke-width="6" stroke-linecap="round"/>'
        )
        svg.append(f'<circle cx="{x(mean):.1f}" cy="{y}" r="5" fill="#475569"/>')
        svg.append(
            f'<rect x="{x(observed) - 6:.1f}" y="{y - 6}" width="12" height="12" fill="#0f766e"/>'
        )
        svg.append(
            f'<text x="{x(observed) + 12:.1f}" y="{y + 5}" class="value">{observed:.4f}</text>'
        )
    svg.append(
        f'<text x="{left}" y="{height - 24}" class="note">Gray interval: placebo 95% range; gray dot: placebo mean; green square: observed AUC.</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _readme(report: Mapping[str, object]) -> str:
    results = report["signal_horizon_results"]
    best_auc = max(results, key=lambda item: item["direction"]["weighted_roc_auc"])
    best_net = max(
        results,
        key=lambda item: item["cost_decomposition"]["mean_delayed_net_return_bps"],
    )
    best_tails = {}
    for count in (100, 500, 1000):
        entries = [
            (
                tail["mean_delayed_net_return_bps"],
                item["signal"],
                item["horizon_seconds"],
            )
            for item in results
            for tail in item["ranked_event_outcomes"]
            if tail["requested_rows"] == count
        ]
        best_tails[count] = max(entries)
    memory = report["runtime_evidence"]["memory"]
    return f"""# Round 36: direction exists, taker edge rejected

**The consumed BTCUSDT L1/tape lane is rejected for taker trading.** Short-horizon direction is measurable, but no prespecified signal, horizon, non-overlapping sample, ranked tail, or causal regime slice covered observed spread plus the frozen fee/slippage model.

| Evidence | Verified result |
| --- | ---: |
| Metric dates | 2023-06-21 to 2023-06-25 UTC |
| Causal one-second rows / calibration events | {report["source_evidence"]["dataset_rows"]:,} / {report["source_evidence"]["metric_event_rows"]:,} |
| Signals x horizons | 13 x 7 = 91 |
| Best weighted direction ROC AUC | {best_auc["direction"]["weighted_roc_auc"]:.4f} ({_LABELS[best_auc["signal"]]}, {best_auc["horizon_seconds"]}s) |
| Daily minimum / median AUC | {best_auc["daily_summary"]["weighted_auc_minimum"]:.4f} / {best_auc["daily_summary"]["weighted_auc_median"]:.4f} |
| Measured half-life | {next(item["half_life_seconds"] for item in report["signal_decay_summary"] if item["signal"] == "l1_imbalance"):.2f}s (consumed role only) |
| Best all-routed delayed taker mean | {best_net["cost_decomposition"]["mean_delayed_net_return_bps"]:+.2f} bps |
| Best top-100 / 500 / 1000 means | {best_tails[100][0]:+.2f} / {best_tails[500][0]:+.2f} / {best_tails[1000][0]:+.2f} bps |
| Positive regime slices | 0 / 819 |
| Peak / final working set | {memory["peak_working_set_bytes"] / 1024**3:.2f} / {memory["current_working_set_bytes"] / 1024**3:.2f} GiB |
| Model candidate / trading authority | none / none |

![Signal decay](charts/signal-decay.svg)

![Cost coverage](charts/cost-coverage.svg)

![After-cost tails](charts/after-cost-tails.svg)

![Daily direction](charts/daily-direction-auc.svg)

![Placebo comparison](charts/placebo-comparison.svg)

![Research progress](charts/research-progress.svg)

The strongest L1 imbalance and microprice effects decay to roughly half their excess AUC within 15 seconds. The largest signal-aligned gross mean across all 91 cells was only `+0.46` bps, while the frozen round-trip fee and adverse-slippage charge was approximately `12` bps. Leverage cannot repair negative unlevered expectancy.

This is post-hoc evidence on five already-consumed BTCUSDT dates. It contains no out-of-sample, ETHUSDT, SOLUSDT, model-training, portfolio-return, testnet/live-execution, or profitability claim.

Data: [signals.csv](signals.csv) | [daily.csv](daily.csv) | [regimes.csv](regimes.csv) | [ranked-event-outcomes.csv](ranked-event-outcomes.csv) | [decay.csv](decay.csv) | [horizon-support.csv](horizon-support.csv) | [progress.csv](progress.csv) | [validated source report](screen.json) | [integrity report](report.json)
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
    """Publish the validated report, complete graph data, and latest-only charts."""

    _design, report, source_report_sha, binding_sha = _validated_source(
        evidence_root=evidence_root.resolve(),
        design_path=design_path,
        binding_path=binding_path,
    )
    signals = _signal_rows(report)
    daily = _daily_rows(report)
    regimes = _regime_rows(report)
    ranked = _ranked_rows(report)
    decay = _decay_rows(report)
    support = _support_rows(report)
    progress_fields, progress = _progress_rows(prior_progress_path, signals, ranked)
    output_dir = output_dir.resolve()
    charts = output_dir / "charts"
    expected = {
        output_dir / "README.md",
        output_dir / "signals.csv",
        output_dir / "daily.csv",
        output_dir / "regimes.csv",
        output_dir / "ranked-event-outcomes.csv",
        output_dir / "decay.csv",
        output_dir / "horizon-support.csv",
        output_dir / "progress.csv",
        output_dir / "screen.json",
        output_dir / "report.json",
        charts / "signal-decay.svg",
        charts / "cost-coverage.svg",
        charts / "after-cost-tails.svg",
        charts / "daily-direction-auc.svg",
        charts / "placebo-comparison.svg",
        charts / "research-progress.svg",
    }
    _clean_output(output_dir, expected)
    _write_csv(output_dir / "signals.csv", signals)
    _write_csv(output_dir / "daily.csv", daily)
    _write_csv(output_dir / "regimes.csv", regimes)
    _write_csv(output_dir / "ranked-event-outcomes.csv", ranked)
    _write_csv(output_dir / "decay.csv", decay)
    _write_csv(output_dir / "horizon-support.csv", support)
    _write_csv(output_dir / "progress.csv", progress, progress_fields)
    _write_text(
        output_dir / "screen.json",
        (evidence_root / "report.json").read_text(encoding="utf-8"),
    )
    _write_text(output_dir / "README.md", _readme(report))
    _write_text(charts / "signal-decay.svg", _decay_svg(signals))
    _write_text(charts / "cost-coverage.svg", _cost_svg(signals))
    _write_text(charts / "after-cost-tails.svg", _tails_svg(ranked))
    _write_text(charts / "daily-direction-auc.svg", _daily_svg(daily))
    _write_text(charts / "placebo-comparison.svg", _placebo_svg(signals))
    _write_text(charts / "research-progress.svg", _progress_svg(progress))
    artifact_paths = sorted(expected - {output_dir / "report.json"})
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "artifact_class": "multi_horizon_signal_decay_graph_data",
        "round": 36,
        "status": "rejected",
        "design_sha256": report["design_sha256"],
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": source_report_sha,
        "source_report_file_sha256": _sha256_file(evidence_root / "report.json"),
        "source_implementation_commit": report["implementation_commit"],
        "source_corpus_certificate_sha256": report["source_evidence"][
            "corpus_certificate_sha256"
        ],
        "source_barrier_targets_sha256": report["source_evidence"][
            "barrier_targets_sha256"
        ],
        "source_cache_key": report["source_evidence"]["cache_key"],
        "signal_count": 13,
        "horizon_count": 7,
        "signal_horizon_cells": 91,
        "model_candidate": None,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "model_trained": False,
        "artifact_integrity": [_artifact(path, output_dir) for path in artifact_paths],
        "publication_sha256": "PENDING",
    }
    canonical = dict(publication)
    canonical.pop("publication_sha256")
    publication["publication_sha256"] = _canonical_sha256(canonical)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(
        description="Publish verified Round 36 signal-decay evidence.",
    )
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-036-multi-horizon-signal-decay-design.json",
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=research / "round-036-signal-decay-execution-binding.json",
    )
    parser.add_argument(
        "--prior-progress",
        type=Path,
        default=research / "latest" / "progress.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=research / "latest",
    )
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
    summary = {
        "status": publication["status"],
        "publication_sha256": publication["publication_sha256"],
        "signal_horizon_cells": publication["signal_horizon_cells"],
        "trading_authority": publication["trading_authority"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
