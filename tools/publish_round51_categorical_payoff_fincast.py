"""Publish hash-verified Round 51 payoff-distribution and FinCast evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime, timedelta
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.publish_cross_asset_cost_aware_ai_ablation import (  # noqa: E402
    _research_progress_svg,
)


ROUND = 51
DESIGN_SCHEMA = "categorical-payoff-fincast-screen-design-v1"
BINDING_SCHEMA = "round-051-categorical-payoff-fincast-execution-binding-v1"
REPORT_SCHEMA = "categorical-payoff-fincast-screen-report-v1"
PUBLICATION_SCHEMA = "categorical-payoff-fincast-publication-v1"
EXPECTED_DESIGN_SHA256 = (
    "42f9afbda8755807e898fa8bb54ad4039d1f9f6b2f4d6c825afc0b1d02bcfba3"
)
EXPECTED_BINDING_SHA256 = (
    "4e4f17914bb27411675ffd4dfa7a64c56b3f4515025a95ffa675edcab031b5b1"
)
EXPECTED_REPORT_SHA256 = (
    "b97a12764256680402d526fd17ee56999c7f88335d66570a196aae3d0e9d0201"
)
EXPECTED_REPORT_FILE_SHA256 = (
    "d2e6c2e1a8ba0a48293f124d148359e4015f9c25aa61fb11b9dc7578d7975a80"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CANDIDATES = (
    "direct_mean_lightgbm",
    "categorical_payoff_lightgbm",
    "categorical_payoff_lightgbm_fincast",
)
SIDES = ("long", "short")
SCENARIOS = ("base", "stress")
LABELS = {
    "direct_mean_lightgbm": "Direct mean",
    "categorical_payoff_lightgbm": "Categorical",
    "categorical_payoff_lightgbm_fincast": "Categorical + FinCast",
}
SHORT_LABELS = {
    "direct_mean_lightgbm": "Direct",
    "categorical_payoff_lightgbm": "Categorical",
    "categorical_payoff_lightgbm_fincast": "FinCast",
}
COLORS = {
    "direct_mean_lightgbm": "#2563a6",
    "categorical_payoff_lightgbm": "#0f766e",
    "categorical_payoff_lightgbm_fincast": "#b54708",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _canonical_identity(value: Mapping[str, object], field: str, label: str) -> str:
    canonical = dict(value)
    claimed = canonical.pop(field, None)
    if not isinstance(claimed, str) or _canonical_sha256(canonical) != claimed:
        raise ValueError(f"{label} canonical identity is invalid")
    return claimed


def _validate_finite_tree(value: object, label: str = "report") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")


def _verified_artifact(item: Mapping[str, object], label: str) -> int:
    path = Path(str(item.get("path") or ""))
    expected_bytes = int(item.get("bytes") or -1)
    expected_hash = str(item.get("sha256") or "")
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or _file_sha256(path) != expected_hash
    ):
        raise ValueError(f"{label} artifact drifted: {path}")
    return expected_bytes


def _validate_prediction(item: Mapping[str, object], *, categorical: bool) -> int:
    size = _verified_artifact(item, "prediction")
    with np.load(Path(str(item["path"])), allow_pickle=False) as values:
        if not values.files or any(
            not np.all(np.isfinite(values[name])) for name in values.files
        ):
            raise ValueError("prediction artifact contains invalid arrays")
        required = {"endpoint_indexes", "long_mean_bps", "short_mean_bps"}
        if not required.issubset(values.files):
            raise ValueError("prediction artifact is incomplete")
        if categorical:
            required_categorical = {
                "long_probabilities",
                "short_probabilities",
                "long_profitable_probability",
                "short_profitable_probability",
            }
            if not required_categorical.issubset(values.files):
                raise ValueError("categorical prediction artifact is incomplete")
            for name in ("long_probabilities", "short_probabilities"):
                probabilities = np.asarray(values[name], dtype=np.float64)
                if (
                    probabilities.ndim != 2
                    or np.any(probabilities < 0.0)
                    or np.any(probabilities > 1.0)
                    or np.max(np.abs(np.sum(probabilities, axis=1) - 1.0)) > 1e-6
                ):
                    raise ValueError("categorical probabilities are invalid")
    return size


def _validated_source(
    *, evidence_root: Path, design_path: Path, binding_path: Path
) -> tuple[dict[str, object], dict[str, object], str, str, int]:
    design = _read_object(design_path, "Round 51 design")
    design_sha = _canonical_identity(design, "design_sha256", "Round 51 design")
    binding = _read_object(binding_path, "Round 51 binding")
    binding_sha = _canonical_identity(
        binding, "binding_sha256", "Round 51 binding"
    )
    source_path = evidence_root / "report.json"
    report = _read_object(source_path, "Round 51 report")
    report_sha = _canonical_identity(
        report, "report_canonical_sha256", "Round 51 report"
    )
    if (
        design_sha != EXPECTED_DESIGN_SHA256
        or binding_sha != EXPECTED_BINDING_SHA256
        or report_sha != EXPECTED_REPORT_SHA256
        or _file_sha256(source_path) != EXPECTED_REPORT_FILE_SHA256
        or design.get("schema_version") != DESIGN_SCHEMA
        or design.get("round") != ROUND
        or design.get("status") != "frozen"
        or binding.get("schema_version") != BINDING_SCHEMA
        or binding.get("round") != ROUND
        or binding.get("design_sha256") != design_sha
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("round") != ROUND
        or report.get("design_sha256") != design_sha
        or report.get("binding_sha256") != binding_sha
        or report.get("implementation_commit") != binding.get("implementation_commit")
    ):
        raise ValueError("Round 51 evidence lineage is invalid")
    _validate_finite_tree(report)
    claims = report.get("claims")
    data = report.get("data")
    symbol_results = report.get("symbol_results")
    portfolios = report.get("portfolio_results")
    distribution = report.get("distribution_gates")
    economics = report.get("economic_gates")
    ai = report.get("ai_uplift_gate")
    if (
        claims
        != {
            "ai_uplift_claim": False,
            "beta_research_only": True,
            "leverage_applied": False,
            "live_authority": False,
            "profitability_claim": False,
            "selection_contaminated": True,
            "source_market_rows_synthetic": 0,
            "testnet_authority": False,
            "trading_authority": False,
        }
        or report.get("round_gate")
        != {"passed": False, "promotion_permitted": False}
        or not isinstance(data, Mapping)
        or tuple(data) != SYMBOLS
        or not isinstance(symbol_results, Mapping)
        or tuple(symbol_results) != SYMBOLS
        or not isinstance(portfolios, Mapping)
        or set(portfolios) != set(CANDIDATES)
        or not isinstance(distribution, Mapping)
        or set(distribution)
        != {
            "categorical_payoff_lightgbm",
            "categorical_payoff_lightgbm_fincast",
        }
        or any(gate.get("passed") is not False for gate in distribution.values())
        or not isinstance(economics, Mapping)
        or set(economics) != set(CANDIDATES)
        or any(gate.get("passed") is not False for gate in economics.values())
        or not isinstance(ai, Mapping)
        or ai.get("passed") is not False
    ):
        raise ValueError("Round 51 gates or governance claims drifted")

    verified_bytes = 0
    model_count = 0
    prediction_count = 0
    for symbol in SYMBOLS:
        evidence = data[symbol]
        source = evidence.get("source_evidence")
        fincast = evidence.get("fincast")
        if (
            evidence.get("synthetic_rows") != 0
            or not isinstance(source, Mapping)
            or source.get("verified") is not True
            or not isinstance(fincast, Mapping)
            or fincast.get("runtime", {}).get("backend_kind") != "directml"
            or fincast.get("runtime", {}).get("backend_device")
            != "privateuseone:0"
            or fincast.get("warning_count") != 0
            or fincast.get("cpu_fallback_warning_count") != 0
            or fincast.get("runtime", {}).get("parameter_count") != 991_437_160
        ):
            raise ValueError(f"Round 51 {symbol} source or FinCast evidence drifted")
        verified_bytes += _verified_artifact(
            evidence["fincast_feature_artifact"], "FinCast feature"
        )
        reports = symbol_results[symbol]
        if set(reports) != set(CANDIDATES):
            raise ValueError(f"Round 51 {symbol} candidate set drifted")
        for candidate in CANDIDATES:
            candidate_report = reports[candidate]
            models = candidate_report.get("models")
            if (
                not isinstance(models, list)
                or len(models) != 3
                or {model.get("seed") for model in models} != {5101, 5102, 5103}
                or candidate_report.get("selection", {}).get("eligible_rows") != 0
                or candidate_report.get("base_trace", {}).get("metrics", {}).get(
                    "trades"
                )
                != 0
                or candidate_report.get("stress_trace", {}).get("metrics", {}).get(
                    "trades"
                )
                != 0
            ):
                raise ValueError(f"Round 51 {symbol} {candidate} evidence drifted")
            for model in models:
                if (
                    model.get("backend_kind") != "opencl"
                    or not str(model.get("backend_device") or "").startswith("opencl:")
                ):
                    raise ValueError("Round 51 LightGBM GPU evidence drifted")
                verified_bytes += _verified_artifact(model["model"], "model")
                verified_bytes += _validate_prediction(
                    model["prediction"],
                    categorical=candidate != "direct_mean_lightgbm",
                )
                model_count += 1
                prediction_count += 1
    if model_count != 27 or prediction_count != 27:
        raise ValueError("Round 51 artifact set is incomplete")
    for candidate in CANDIDATES:
        for scenario in SCENARIOS:
            if portfolios[candidate][scenario]["metrics"]["trades"] != 0:
                raise ValueError("Round 51 zero-trade result changed")
    return report, design, report_sha, binding_sha, verified_bytes


def _json_cell(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            writer.writerow({field: _json_cell(source.get(field, "")) for field in fields})


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _forecast_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            models = report["symbol_results"][symbol][candidate]["models"]
            for model in models:
                for side in SIDES:
                    metrics = model["forecast_metrics"][side]
                    row: dict[str, object] = {
                        "symbol": symbol,
                        "candidate_id": candidate,
                        "candidate": LABELS[candidate],
                        "seed": model["seed"],
                        "side": side,
                    }
                    row.update(metrics)
                    row.pop("calibration_bins", None)
                    row.pop("daily_brier_skill", None)
                    rows.append(row)
    return rows


def _prediction_tail_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            result = report["symbol_results"][symbol][candidate]
            arrays: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
            probabilities: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
            positive_by_seed: dict[str, list[int]] = {side: [] for side in SIDES}
            for model in result["models"]:
                with np.load(model["prediction"]["path"], allow_pickle=False) as values:
                    for side in SIDES:
                        prediction = np.asarray(
                            values[f"{side}_mean_bps"], dtype=np.float64
                        )
                        arrays[side].append(prediction)
                        positive_by_seed[side].append(int(np.sum(prediction > 0.0)))
                        probability_name = f"{side}_profitable_probability"
                        if probability_name in values.files:
                            probabilities[side].append(
                                np.asarray(values[probability_name], dtype=np.float64)
                            )
            for side in SIDES:
                stack = np.stack(arrays[side])
                mean = np.mean(stack, axis=0)
                worst = np.min(stack, axis=0)
                row = {
                    "symbol": symbol,
                    "candidate_id": candidate,
                    "candidate": LABELS[candidate],
                    "side": side,
                    "rows": len(mean),
                    "mean_prediction_min_bps": float(np.min(mean)),
                    "mean_prediction_p50_bps": float(np.quantile(mean, 0.50)),
                    "mean_prediction_p90_bps": float(np.quantile(mean, 0.90)),
                    "mean_prediction_p99_bps": float(np.quantile(mean, 0.99)),
                    "mean_prediction_p999_bps": float(np.quantile(mean, 0.999)),
                    "mean_prediction_max_bps": float(np.max(mean)),
                    "worst_seed_prediction_max_bps": float(np.max(worst)),
                    "all_seed_positive_rows": int(np.sum(worst > 0.0)),
                    "seed_5101_positive_rows": positive_by_seed[side][0],
                    "seed_5102_positive_rows": positive_by_seed[side][1],
                    "seed_5103_positive_rows": positive_by_seed[side][2],
                    "selected_rows": int(
                        result["selection"][f"{side}_eligible_rows"]
                    ),
                }
                if probabilities[side]:
                    mean_probability = np.mean(np.stack(probabilities[side]), axis=0)
                    row.update(
                        {
                            "mean_profitable_probability_max": float(
                                np.max(mean_probability)
                            ),
                            "mean_profitable_probability_above_0_5_rows": int(
                                np.sum(mean_probability > 0.5)
                            ),
                        }
                    )
                rows.append(row)
    return rows


def _scenario_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for scenario in SCENARIOS:
            result = report["portfolio_results"][candidate][scenario]
            row = {
                "candidate_id": candidate,
                "candidate": LABELS[candidate],
                "scenario": scenario,
                **result["metrics"],
                "maximum_single_symbol_positive_pnl_share": result[
                    "maximum_single_symbol_positive_pnl_share"
                ],
            }
            rows.append(row)
    return rows


def _symbol_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for scenario in SCENARIOS:
            portfolio = report["portfolio_results"][candidate][scenario]
            for symbol in SYMBOLS:
                trace = report["symbol_results"][symbol][candidate][
                    f"{scenario}_trace"
                ]
                rows.append(
                    {
                        "candidate_id": candidate,
                        "candidate": LABELS[candidate],
                        "scenario": scenario,
                        "symbol": symbol,
                        **trace["metrics"],
                        "portfolio_weighted_net_bps": portfolio["symbol_net_bps"][
                            symbol
                        ],
                        "positive_pnl_share": portfolio["positive_pnl_share"][symbol],
                    }
                )
    return rows


def _model_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for candidate in CANDIDATES:
            for model in report["symbol_results"][symbol][candidate]["models"]:
                row = {
                    "symbol": symbol,
                    "candidate_id": candidate,
                    "candidate": LABELS[candidate],
                    "seed": model["seed"],
                    "backend_kind": model["backend_kind"],
                    "backend_device": model["backend_device"],
                    "training_seconds": model["training_seconds"],
                    "cache_state": model["cache_state"],
                    "cache_load_seconds": model["cache_load_seconds"],
                    "best_iterations_long": model["best_iterations"]["long"],
                    "best_iterations_short": model["best_iterations"]["short"],
                    "model_bytes": model["model"]["bytes"],
                    "model_sha256": model["model"]["sha256"],
                    "prediction_bytes": model["prediction"]["bytes"],
                    "prediction_sha256": model["prediction"]["sha256"],
                }
                if candidate != "direct_mean_lightgbm":
                    row.update(
                        {
                            "temperature_long": model["temperature"]["long"],
                            "temperature_short": model["temperature"]["short"],
                            "calibration_log_loss_before_long": model[
                                "calibration_log_loss_before"
                            ]["long"],
                            "calibration_log_loss_after_long": model[
                                "calibration_log_loss_after"
                            ]["long"],
                            "calibration_log_loss_before_short": model[
                                "calibration_log_loss_before"
                            ]["short"],
                            "calibration_log_loss_after_short": model[
                                "calibration_log_loss_after"
                            ]["short"],
                        }
                    )
                rows.append(row)
    return rows


def _barrier_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        summary = report["data"][symbol]["barrier_summary"]
        for scenario in SCENARIOS:
            for side in SIDES:
                result = summary[scenario][side]
                outcomes = summary[scenario][f"{side}_outcomes"]
                rows.append(
                    {
                        "symbol": symbol,
                        "scenario": scenario,
                        "side": side,
                        **result,
                        **{f"outcome_{key}": value for key, value in outcomes.items()},
                    }
                )
    return rows


def _role_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for role, result in report["data"][symbol]["roles"].items():
            rows.append(
                {
                    "symbol": symbol,
                    "role": role,
                    "rows": result["rows"],
                    "first_decision_time_ms": result["first_decision_time_ms"],
                    "first_decision_time_utc": datetime.fromtimestamp(
                        result["first_decision_time_ms"] / 1000.0, tz=UTC
                    ).isoformat(),
                    "last_decision_time_ms": result["last_decision_time_ms"],
                    "last_decision_time_utc": datetime.fromtimestamp(
                        result["last_decision_time_ms"] / 1000.0, tz=UTC
                    ).isoformat(),
                }
            )
    return rows


def _source_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        evidence = report["data"][symbol]
        source = evidence["source_evidence"]
        rows.append(
            {
                "symbol": symbol,
                "provider": source["corpus_certificate"]["provider"],
                "market_type": source["corpus_certificate"]["market_type"],
                "first_event_time_ms": source["first_event_time_ms"],
                "last_event_time_ms": source["last_event_time_ms"],
                "source_archive_count": source["source_archive_count"],
                "source_raw_rows": source["source_raw_rows"],
                "microstructure_rows": evidence["microstructure_rows"],
                "valid_barrier_rows": evidence["valid_barrier_rows"],
                "synthetic_rows": evidence["synthetic_rows"],
                "verified": source["verified"],
                "certificate_sha256": source["corpus_certificate"][
                    "certificate_sha256"
                ],
                "fincast_rows": evidence["fincast"]["rows"],
                "fincast_inference_seconds": evidence["fincast"][
                    "inference_seconds"
                ],
                "fincast_warning_count": evidence["fincast"]["warning_count"],
                "fincast_cpu_fallback_warning_count": evidence["fincast"][
                    "cpu_fallback_warning_count"
                ],
                "fincast_feature_sha256": evidence["fincast"]["features_sha256"],
            }
        )
    return rows


def _gate_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate, gate in report["distribution_gates"].items():
        for reason in gate["reasons"]:
            rows.append(
                {
                    "gate_type": "distribution",
                    "candidate_id": candidate,
                    "candidate": LABELS[candidate],
                    "passed": gate["passed"],
                    "reason": reason,
                }
            )
    for candidate, gate in report["economic_gates"].items():
        for reason in gate["reasons"]:
            rows.append(
                {
                    "gate_type": "economic",
                    "candidate_id": candidate,
                    "candidate": LABELS[candidate],
                    "passed": gate["passed"],
                    "reason": reason,
                }
            )
    for reason in report["ai_uplift_gate"]["reasons"]:
        rows.append(
            {
                "gate_type": "ai_uplift",
                "candidate_id": "categorical_payoff_lightgbm_fincast",
                "candidate": LABELS["categorical_payoff_lightgbm_fincast"],
                "passed": report["ai_uplift_gate"]["passed"],
                "reason": reason,
            }
        )
    return rows


def _ai_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    gate = report["ai_uplift_gate"]
    rows = [
        {
            "scope": "forecast",
            "scenario": "all",
            "control_ranked_probability_skill": gate[
                "control_average_ranked_probability_skill"
            ],
            "treatment_ranked_probability_skill": gate[
                "treatment_average_ranked_probability_skill"
            ],
            "ranked_probability_skill_improvement": gate[
                "ranked_probability_skill_improvement"
            ],
            "control_expected_payoff_spearman": gate[
                "control_average_expected_payoff_spearman"
            ],
            "treatment_expected_payoff_spearman": gate[
                "treatment_average_expected_payoff_spearman"
            ],
            "expected_payoff_spearman_improvement": gate[
                "expected_payoff_spearman_improvement"
            ],
            "control_expected_payoff_mse_bps2": gate[
                "control_average_expected_payoff_mse_bps2"
            ],
            "treatment_expected_payoff_mse_bps2": gate[
                "treatment_average_expected_payoff_mse_bps2"
            ],
            "expected_payoff_mse_ratio": gate["expected_payoff_mse_ratio"],
            "passed": gate["passed"],
        }
    ]
    for scenario in SCENARIOS:
        economics = gate["economics"][scenario]
        rows.append(
            {
                "scope": "economics",
                "scenario": scenario,
                **{key: value for key, value in economics.items() if key != "daily"},
                "passed": gate["passed"],
            }
        )
    return rows


def _daily_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    start = datetime(2023, 6, 9, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        for scenario in SCENARIOS:
            observed = {
                int(item["utc_day_id"]): float(item["net_bps"])
                for item in report["portfolio_results"][candidate][scenario][
                    "daily_net_bps"
                ]
            }
            cumulative = 0.0
            for offset in range(6):
                day = start + timedelta(days=offset)
                day_id = int(day.timestamp() * 1000) // 86_400_000
                net = observed.get(day_id, 0.0)
                cumulative += net
                rows.append(
                    {
                        "candidate_id": candidate,
                        "candidate": LABELS[candidate],
                        "scenario": scenario,
                        "date_utc": day.date().isoformat(),
                        "selected_trades": 0,
                        "net_bps": net,
                        "cumulative_net_bps": cumulative,
                        "ledger_present": day_id in observed,
                        "zero_return_reason": "no selected trades",
                    }
                )
    return rows


def _progress_rows(
    prior_path: Path, report: Mapping[str, object]
) -> tuple[list[str], list[dict[str, object]]]:
    with prior_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    observed_rounds = [int(row["round"]) for row in rows]
    if observed_rounds == list(range(1, ROUND + 1)):
        rows = rows[:-1]
        observed_rounds = observed_rounds[:-1]
    if not fields or observed_rounds != list(range(1, ROUND)):
        raise ValueError("Round 51 prior progress history is invalid")
    row: dict[str, object] = {field: "" for field in fields}
    row.update(
        {
            "round": ROUND,
            "stage": "categorical payoff + FinCast uplift screen",
            "periods": "2023-05-16..2023-06-14; eval 2023-06-09..2023-06-14",
            "selection_contaminated": True,
            "horizon_seconds": 300,
            "feature_set": "118 causal microstructure; +30 FinCast treatment",
            "risk_level": "consumed development only; unlevered",
            "spearman_ic": report["ai_uplift_gate"][
                "treatment_average_expected_payoff_spearman"
            ],
            "selected_signals": 0,
            "executable_trades": 0,
            "status": "rejected",
            "source_file": (
                "verified Round 51 real-tick report; no candidate predicted a "
                "positive after-cost action"
            ),
            "best_policy_trades": 0,
            "best_policy_total_net_bps": 0.0,
            "best_policy_max_drawdown_bps": 0.0,
            "best_model_id": "none",
            "ensemble_models": 27,
            "valid_barrier_rows": sum(
                int(report["data"][symbol]["valid_barrier_rows"])
                for symbol in SYMBOLS
            ),
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 0,
            "development_consumed": True,
            "architecture_gates_passed": 0,
            "architecture_gate_count": 5,
        }
    )
    rows.append(row)
    return fields, rows


def _svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:"Segoe UI",Arial,sans-serif;letter-spacing:0}.title{font-size:28px;font-weight:700;fill:#17212b}.subtitle{font-size:14px;fill:#52606d}.label{font-size:13px;fill:#263746}.axis{font-size:12px;fill:#60717f}.value{font-size:12px;font-weight:650;fill:#263746}.note{font-size:12px;fill:#65727d}.grid{stroke:#e1e8ed;stroke-width:1}.zero{stroke:#526674;stroke-width:2;stroke-dasharray:6 5}.gate{stroke:#b42318;stroke-width:2;stroke-dasharray:5 4}</style>',
        f'<text x="56" y="52" class="title">{html.escape(title)}</text>',
        f'<text x="56" y="80" class="subtitle">{html.escape(description)}</text>',
    ]


def _forecast_svg(rows: Sequence[Mapping[str, object]]) -> str:
    aggregates: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        subset = [row for row in rows if row["candidate_id"] == candidate]
        aggregates[candidate] = {
            "spearman": float(
                np.mean([float(row["expected_payoff_spearman"]) for row in subset])
            ),
            "mse_skill": float(
                np.mean([float(row["expected_payoff_mse_skill"]) for row in subset])
            ),
            "rps_skill": (
                float(
                    np.mean(
                        [
                            float(row["ranked_probability_skill"])
                            for row in subset
                        ]
                    )
                )
                if candidate != "direct_mean_lightgbm"
                else 0.0
            ),
        }
    width, height = 1420, 660
    lines = _svg_start(
        width,
        height,
        "Round 51 forecast quality",
        "Seed, side, and symbol averages on 2023-06-09 through 2023-06-14; positive skill does not establish tradable edge.",
    )
    panels = (
        ("spearman", "Expected-payoff Spearman", -0.01, 0.06),
        ("mse_skill", "Expected-payoff MSE skill", -0.02, 0.20),
        ("rps_skill", "Ranked-probability skill", -0.01, 0.07),
    )
    panel_width = 390
    for panel_index, (key, label, low, high) in enumerate(panels):
        x0 = 80 + panel_index * 450
        y0, chart_height = 150, 350
        lines.append(f'<text x="{x0}" y="125" class="label">{label}</text>')
        for tick in (low, 0.0, high):
            y = y0 + chart_height * (high - tick) / (high - low)
            lines.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}" class="{"zero" if tick == 0 else "grid"}"/>'
            )
            lines.append(
                f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:+.2f}</text>'
            )
        bar_width = 82
        for index, candidate in enumerate(CANDIDATES):
            value = aggregates[candidate][key]
            zero_y = y0 + chart_height * high / (high - low)
            value_y = y0 + chart_height * (high - value) / (high - low)
            top = min(zero_y, value_y)
            bar_height = max(1.0, abs(zero_y - value_y))
            x = x0 + 35 + index * 118
            lines.append(
                f'<rect x="{x}" y="{top:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{COLORS[candidate]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{top - 9:.1f}" text-anchor="middle" class="value">{value:+.4f}</text>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y0 + chart_height + 27}" text-anchor="middle" class="axis">{SHORT_LABELS[candidate]}</text>'
            )
    lines.append(
        '<text x="56" y="610" class="note">The FinCast treatment improved mean payoff rank by +0.0020 but reduced ranked-probability skill by 0.000074 and increased MSE.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _barrier_svg(rows: Sequence[Mapping[str, object]]) -> str:
    base = [row for row in rows if row["scenario"] == "base"]
    width, height = 1420, 690
    left, right, top, row_height = 270, 90, 145, 72
    chart_width = width - left - right
    low, high = -14.0, 2.0

    def x(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    lines = _svg_start(
        width,
        height,
        "Unconditional five-minute action payoffs were deeply negative",
        "Exact after-cost barrier targets over 2023-05-16 through 2023-06-14; no model prediction is involved.",
    )
    for tick in (-14.0, -10.0, -6.0, -2.0, 0.0, 2.0):
        px = x(tick)
        lines.append(
            f'<line x1="{px:.1f}" y1="{top - 20}" x2="{px:.1f}" y2="{top + row_height * len(base)}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{px:.1f}" y="{top - 31}" text-anchor="middle" class="axis">{tick:+.0f} bps</text>'
        )
    for index, row in enumerate(base):
        y = top + index * row_height
        value = float(row["mean_net_bps"])
        x0, x1 = x(value), x(0.0)
        lines.append(
            f'<text x="{left - 18}" y="{y + 16}" text-anchor="end" class="label">{row["symbol"]} {row["side"]}</text>'
        )
        lines.append(
            f'<rect x="{min(x0, x1):.1f}" y="{y}" width="{max(1.0, abs(x1 - x0)):.1f}" height="25" fill="#b42318"/>'
        )
        lines.append(
            f'<text x="{x0 + 9:.1f}" y="{y + 18}" class="value" style="fill:#ffffff">{value:+.2f}</text>'
        )
        lines.append(
            f'<text x="{x1 + 12:.1f}" y="{y + 18}" class="note">positive {100 * float(row["positive_ratio"]):.1f}%</text>'
        )
    lines.append(
        '<text x="56" y="650" class="note">Costs include contemporaneous BBO crossing, 5 bps taker fee per side, latency, and trigger slippage under the frozen replay contract.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _prediction_tail_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1480, 1080
    left, right, top, row_height = 390, 90, 135, 47
    low, high = -12.5, 1.0
    chart_width = width - left - right

    def x(value: float) -> float:
        clipped = min(high, max(low, value))
        return left + chart_width * (clipped - low) / (high - low)

    lines = _svg_start(
        width,
        height,
        "No ensemble found a positive expected-value action",
        "Best worst-seed expected payoff anywhere in each evaluation series; the frozen selector required this value above zero.",
    )
    for tick in (-12.0, -9.0, -6.0, -3.0, 0.0):
        px = x(tick)
        lines.append(
            f'<line x1="{px:.1f}" y1="{top - 20}" x2="{px:.1f}" y2="{top + row_height * len(rows)}" class="{"zero" if tick == 0 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{px:.1f}" y="{top - 31}" text-anchor="middle" class="axis">{tick:+.0f} bps</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * row_height
        value = float(row["worst_seed_prediction_max_bps"])
        x0, x1 = x(value), x(0.0)
        label = f'{row["symbol"]} {row["side"]} | {SHORT_LABELS[str(row["candidate_id"])]}'
        lines.append(
            f'<text x="{left - 18}" y="{y + 14}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{min(x0, x1):.1f}" y="{y}" width="{max(1.0, abs(x1 - x0)):.1f}" height="21" fill="{COLORS[str(row["candidate_id"])]}"/>'
        )
        lines.append(
            f'<text x="{x0 - 9:.1f}" y="{y + 15}" text-anchor="end" class="value">{value:+.2f}</text>'
        )
    lines.append(
        '<text x="56" y="1042" class="note">Zero selected trades is a failed viability result, not zero-risk or flat-performance evidence.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _calibration_svg(rows: Sequence[Mapping[str, object]]) -> str:
    categorical = [
        row
        for row in rows
        if row["candidate_id"] != "direct_mean_lightgbm"
    ]
    grouped: dict[tuple[str, str, str], float] = {}
    for row in categorical:
        key = (str(row["candidate_id"]), str(row["symbol"]), str(row["side"]))
        grouped[key] = max(
            grouped.get(key, 0.0), float(row["maximum_10_bin_calibration_error"])
        )
    entries = [
        (candidate, symbol, side, grouped[(candidate, symbol, side)])
        for symbol in SYMBOLS
        for side in SIDES
        for candidate in CANDIDATES[1:]
    ]
    width, height = 1450, 850
    left, right, top, row_height = 390, 100, 140, 52
    chart_width = width - left - right
    high = 0.15

    def x(value: float) -> float:
        return left + chart_width * value / high

    lines = _svg_start(
        width,
        height,
        "Profitable-action probability calibration failed",
        "Maximum ten-bin calibration error across seeds; the frozen per-model limit was 0.05.",
    )
    for tick in (0.0, 0.05, 0.10, 0.15):
        px = x(tick)
        lines.append(
            f'<line x1="{px:.1f}" y1="{top - 20}" x2="{px:.1f}" y2="{top + row_height * len(entries)}" class="{"gate" if tick == 0.05 else "grid"}"/>'
        )
        lines.append(
            f'<text x="{px:.1f}" y="{top - 31}" text-anchor="middle" class="axis">{tick:.2f}</text>'
        )
    for index, (candidate, symbol, side, value) in enumerate(entries):
        y = top + index * row_height
        label = f"{symbol} {side} | {SHORT_LABELS[candidate]}"
        lines.append(
            f'<text x="{left - 18}" y="{y + 16}" text-anchor="end" class="label">{html.escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{max(1.0, x(value) - left):.1f}" height="25" fill="{COLORS[candidate]}"/>'
        )
        lines.append(
            f'<text x="{x(value) + 9:.1f}" y="{y + 18}" class="value">{value:.4f}</text>'
        )
    lines.append(
        '<text x="56" y="812" class="note">Short-side calibration exceeded the limit for every symbol and seed; FinCast did not repair it.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _ai_svg(report: Mapping[str, object]) -> str:
    gate = report["ai_uplift_gate"]
    metrics = (
        (
            "Ranked-probability skill",
            float(gate["control_average_ranked_probability_skill"]),
            float(gate["treatment_average_ranked_probability_skill"]),
            0.005,
            "higher is better; required delta +0.005",
        ),
        (
            "Expected-payoff Spearman",
            float(gate["control_average_expected_payoff_spearman"]),
            float(gate["treatment_average_expected_payoff_spearman"]),
            0.005,
            "higher is better; required delta +0.005",
        ),
        (
            "Expected-payoff MSE (bps2)",
            float(gate["control_average_expected_payoff_mse_bps2"]),
            float(gate["treatment_average_expected_payoff_mse_bps2"]),
            0.0,
            "lower is better; treatment/control = 1.00117",
        ),
    )
    width, height = 1420, 760
    lines = _svg_start(
        width,
        height,
        "FinCast did not establish AI uplift",
        "Matched deterministic control and 991M-parameter FinCast treatment on identical evaluation endpoints.",
    )
    for index, (name, control, treatment, required, note) in enumerate(metrics):
        x0 = 80 + index * 450
        y0, panel_height, panel_width = 165, 360, 360
        low = 0.0
        high = max(control, treatment) * 1.18
        if index < 2:
            high = max(0.06, high)
        lines.append(f'<text x="{x0}" y="130" class="label">{html.escape(name)}</text>')
        for tick in (low, high / 2.0, high):
            y = y0 + panel_height * (high - tick) / (high - low)
            lines.append(
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" y2="{y:.1f}" class="grid"/>'
            )
            lines.append(
                f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.3f}</text>'
            )
        for bar_index, (label, value, color) in enumerate(
            (("Control", control, COLORS[CANDIDATES[1]]), ("FinCast", treatment, COLORS[CANDIDATES[2]]))
        ):
            x = x0 + 55 + bar_index * 145
            y = y0 + panel_height * (high - value) / (high - low)
            lines.append(
                f'<rect x="{x}" y="{y:.1f}" width="95" height="{y0 + panel_height - y:.1f}" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{x + 47.5:.1f}" y="{y - 9:.1f}" text-anchor="middle" class="value">{value:.5f}</text>'
            )
            lines.append(
                f'<text x="{x + 47.5:.1f}" y="{y0 + panel_height + 28}" text-anchor="middle" class="axis">{label}</text>'
            )
        delta = treatment - control
        lines.append(
            f'<text x="{x0}" y="{y0 + panel_height + 62}" class="value">Delta {delta:+.6f}</text>'
        )
        lines.append(
            f'<text x="{x0}" y="{y0 + panel_height + 84}" class="note">{html.escape(note)}</text>'
        )
        if required:
            lines.append(
                f'<text x="{x0}" y="{y0 + panel_height + 104}" class="note">Observed delta missed by {required - delta:+.6f}</text>'
            )
    lines.append(
        '<text x="56" y="720" class="note">Economics could not show uplift because both matched selectors produced zero trades; the AI gate therefore failed closed.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _daily_svg(rows: Sequence[Mapping[str, object]]) -> str:
    base = [row for row in rows if row["scenario"] == "base"]
    dates = sorted({str(row["date_utc"]) for row in base})
    width, height = 1420, 570
    left, right, top, chart_height = 110, 80, 150, 270
    chart_width = width - left - right
    zero_y = top + chart_height / 2
    lines = _svg_start(
        width,
        height,
        "Evaluation ledger: no selected trades",
        "Dated base-scenario cumulative net bps, 2023-06-09 through 2023-06-14; flat lines denote abstention, not validated performance.",
    )
    lines.append(
        f'<rect x="{left}" y="{top}" width="{chart_width}" height="{chart_height}" fill="#ffffff" stroke="#d8e0e7"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{zero_y}" x2="{left + chart_width}" y2="{zero_y}" class="zero"/>'
    )
    lines.append(
        f'<text x="{left - 15}" y="{zero_y + 4}" text-anchor="end" class="axis">0 bps</text>'
    )
    for index, date in enumerate(dates):
        x = left + chart_width * index / (len(dates) - 1)
        lines.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" class="grid"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 28}" text-anchor="middle" class="axis">{date[5:]}</text>'
        )
    lines.append(
        f'<line x1="{left}" y1="{zero_y}" x2="{left + chart_width}" y2="{zero_y}" stroke="#263746" stroke-width="4"/>'
    )
    lines.append(
        f'<text x="{left + chart_width - 12}" y="{zero_y - 12}" text-anchor="end" class="value">All three candidates: 0 trades / 0 bps</text>'
    )
    lines.append(
        '<text x="56" y="525" class="note">No ROI, profit factor, or drawdown inference can be made from an empty ledger. All economic gates failed.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _artifact(path: Path, root: Path) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            value["row_count"] = sum(1 for _ in csv.DictReader(handle))
    return value


def _clean_output(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    expected_parent = (ROOT / "docs" / "model-research" / "action-value").resolve()
    if not resolved.is_relative_to(expected_parent) or resolved.name != "latest":
        raise ValueError("publication output must be the repository action-value/latest path")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    for path in sorted(output_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def _readme(report: Mapping[str, object]) -> str:
    ai = report["ai_uplift_gate"]
    return f"""# Round 51: Categorical Payoff + FinCast

> **Beta research warning:** this consumed-development screen is not approved for testnet, live day trading, leverage, autonomous execution, or a profitability claim.

Round 51 tested direct after-cost regression, a categorical payoff model, and the same categorical model with causal features from the 991,437,160-parameter FinCast foundation model. All three selectors produced **zero eligible actions**. The round was rejected.

![Prediction tail](charts/prediction-tail.svg)

The result is not a threshold accident. Across BTCUSDT, ETHUSDT, and SOLUSDT, even the best worst-seed expected payoff remained below zero after spread, fees, latency, and slippage. An empty ledger has no meaningful ROI, profit factor, or drawdown.

![Barrier baselines](charts/barrier-baselines.svg)

![Evaluation ledger](charts/daily-equity.svg)

| Candidate | Selected trades | Base net | Stress net | Economic gate |
|---|---:|---:|---:|:---:|
| Direct mean | 0 | 0 bps | 0 bps | false |
| Categorical | 0 | 0 bps | 0 bps | false |
| Categorical + FinCast | 0 | 0 bps | 0 bps | false |

The categorical models showed positive aggregate proper-score skill, but short-side profitable-probability calibration failed the frozen `0.05` limit. SOL short payoff rank also fell below `0.03`.

![Forecast quality](charts/forecast-quality.svg)

![Calibration](charts/calibration.svg)

FinCast changed average ranked-probability skill by `{float(ai['ranked_probability_skill_improvement']):+.6f}`, expected-payoff Spearman by `{float(ai['expected_payoff_spearman_improvement']):+.6f}`, and expected-payoff MSE by a ratio of `{float(ai['expected_payoff_mse_ratio']):.6f}`. It missed both precommitted `+0.005` uplift gates and did not establish economic uplift.

![AI uplift](charts/ai-uplift.svg)

![Research progress](charts/research-progress.svg)

The source is verified, checksummed Binance USD-M `bookTicker`, trades, and sampled aggregate depth for **2023-05-16 through 2023-06-14**. Decisions are ten seconds apart; each target follows exact 100 ms BBO paths for up to five minutes. Evaluation covers **2023-06-09 through 2023-06-14**. This is real tick evidence, but it is only 30 days and is not a multi-year claim.

FinCast ran through DirectML on the AMD GPU with zero warnings or CPU fallback. All 27 LightGBM models used OpenCL. The publisher independently verified 27 model files, 27 prediction files, three FinCast matrices, finite arrays, and probability normalization.

Data: [forecast](forecast.csv) | [prediction tails](prediction-tails.csv) | [barrier baselines](barrier-baselines.csv) | [scenarios](scenarios.csv) | [symbols](symbols.csv) | [daily ledger](daily-policy.csv) | [AI uplift](ai-uplift.csv) | [models](models.csv) | [roles](roles.csv) | [sources](sources.csv) | [gates](gates.csv) | [progress](progress.csv) | [source report](screen.json) | [publication integrity](report.json)
"""


def publish(
    *,
    evidence_root: Path,
    design_path: Path,
    binding_path: Path,
    prior_progress_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    report, _design, report_sha, binding_sha, verified_bytes = _validated_source(
        evidence_root=evidence_root,
        design_path=design_path,
        binding_path=binding_path,
    )
    forecast = _forecast_rows(report)
    tails = _prediction_tail_rows(report)
    barriers = _barrier_rows(report)
    scenarios = _scenario_rows(report)
    symbols = _symbol_rows(report)
    models = _model_rows(report)
    roles = _role_rows(report)
    sources = _source_rows(report)
    gates = _gate_rows(report)
    ai = _ai_rows(report)
    daily = _daily_rows(report)
    progress_fields, progress = _progress_rows(prior_progress_path, report)

    _clean_output(output_dir)
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(evidence_root / "report.json", output_dir / "screen.json")
    _write_csv(output_dir / "forecast.csv", forecast)
    _write_csv(output_dir / "prediction-tails.csv", tails)
    _write_csv(output_dir / "barrier-baselines.csv", barriers)
    _write_csv(output_dir / "scenarios.csv", scenarios)
    _write_csv(output_dir / "symbols.csv", symbols)
    _write_csv(output_dir / "models.csv", models)
    _write_csv(output_dir / "roles.csv", roles)
    _write_csv(output_dir / "sources.csv", sources)
    _write_csv(output_dir / "gates.csv", gates)
    _write_csv(output_dir / "ai-uplift.csv", ai)
    _write_csv(output_dir / "daily-policy.csv", daily)
    _write_csv(
        output_dir / "progress.csv",
        [{field: row.get(field, "") for field in progress_fields} for row in progress],
    )
    _write_text(chart_dir / "forecast-quality.svg", _forecast_svg(forecast))
    _write_text(chart_dir / "prediction-tail.svg", _prediction_tail_svg(tails))
    _write_text(chart_dir / "barrier-baselines.svg", _barrier_svg(barriers))
    _write_text(chart_dir / "calibration.svg", _calibration_svg(forecast))
    _write_text(chart_dir / "ai-uplift.svg", _ai_svg(report))
    _write_text(chart_dir / "daily-equity.svg", _daily_svg(daily))
    _write_text(chart_dir / "research-progress.svg", _research_progress_svg(progress))
    _write_text(output_dir / "README.md", _readme(report))

    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "report.json"
    )
    artifact_integrity = [_artifact(path, output_dir) for path in artifact_paths]
    publication: dict[str, object] = {
        "schema_version": PUBLICATION_SCHEMA,
        "round": ROUND,
        "status": "rejected",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "publisher_path": "tools/publish_round51_categorical_payoff_fincast.py",
        "publisher_sha256": _file_sha256(Path(__file__)),
        "source_implementation_commit": report["implementation_commit"],
        "design_sha256": EXPECTED_DESIGN_SHA256,
        "binding_sha256": binding_sha,
        "source_report_canonical_sha256": report_sha,
        "source_report_file_sha256": EXPECTED_REPORT_FILE_SHA256,
        "symbols": list(SYMBOLS),
        "source_period": "2023-05-16..2023-06-14",
        "evaluation_period": "2023-06-09..2023-06-14",
        "decision_cadence_seconds": 10,
        "target_path_resolution_ms": 100,
        "target_horizon_seconds": 300,
        "source_market_rows_synthetic": 0,
        "model_artifact_count": 27,
        "prediction_artifact_count": 27,
        "fincast_feature_artifact_count": 3,
        "external_artifacts_hash_verified": True,
        "external_artifacts_verified_bytes": verified_bytes,
        "fincast_parameter_count": 991_437_160,
        "fincast_backend_kind": "directml",
        "lightgbm_backend_kind": "opencl",
        "selected_trades": 0,
        "round_gate_passed": False,
        "distribution_gate_pass_count": 0,
        "economic_gate_pass_count": 0,
        "ai_uplift_gate_passed": False,
        "selection_contaminated": True,
        "development_only": True,
        "trading_authority": False,
        "testnet_authority": False,
        "live_authority": False,
        "profitability_claim": False,
        "leverage_applied": False,
        "ai_uplift_claim": False,
        "graph_sources": {
            "charts/forecast-quality.svg": ["forecast.csv"],
            "charts/prediction-tail.svg": ["prediction-tails.csv"],
            "charts/barrier-baselines.svg": ["barrier-baselines.csv"],
            "charts/calibration.svg": ["forecast.csv"],
            "charts/ai-uplift.svg": ["ai-uplift.csv"],
            "charts/daily-equity.svg": ["daily-policy.csv"],
            "charts/research-progress.svg": ["progress.csv"],
        },
        "artifact_integrity": artifact_integrity,
    }
    publication["publication_sha256"] = _canonical_sha256(publication)
    _write_text(
        output_dir / "report.json",
        json.dumps(publication, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            r"E:\SimpleAITradingData\round51-categorical-payoff-fincast-20260713-v2"
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=Path(
            "docs/model-research/action-value/round-051-categorical-payoff-fincast-design.json"
        ),
    )
    parser.add_argument(
        "--binding",
        type=Path,
        default=Path(
            "docs/model-research/action-value/round-051-execution-binding.json"
        ),
    )
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    publication = publish(
        evidence_root=args.evidence_root,
        design_path=args.design,
        binding_path=args.binding,
        prior_progress_path=args.prior_progress,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "round": publication["round"],
                "status": publication["status"],
                "selected_trades": publication["selected_trades"],
                "publication_sha256": publication["publication_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
