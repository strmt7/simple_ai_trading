"""Deterministic publication of prospective Polymarket model evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html import escape
import io
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any

from .ai_model_benchmark import AI_MODEL_BENCHMARK_CONTRACT
from .ai_runtime import ollama_residency_from_mapping
from .ai_uplift import assess_ai_uplift
from .polymarket_model import (
    POLYMARKET_MODEL_FEATURE_NAMES,
    POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION,
    POLYMARKET_PROFILE_CONTRACT_SHA256,
    POLYMARKET_PROFILE_FEATURES,
    POLYMARKET_PROFILE_L2_CANDIDATES,
    POLYMARKET_PROFILE_MODEL_SCHEMA_VERSION,
    POLYMARKET_PROFILE_REPORT_SCHEMA_VERSION,
)
from .polymarket_model_execution import (
    POLYMARKET_CAUSAL_SETTLEMENT_CONTRACT_SHA256,
    POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION,
    POLYMARKET_RETRY_CHALLENGER_SCHEMA_VERSION,
    POLYMARKET_RETRY_CONTRACT_SHA256,
    POLYMARKET_RETRY_EXECUTION_REPORT_SCHEMA_VERSION,
)
from .polymarket_source_verification import (
    validate_polymarket_source_verification,
)


POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION = "polymarket-prospective-model-experiment-v3"
_PREDICTION_SCHEMA = "polymarket-held-out-predictions-v2"
_PROFILE_PREDICTION_SCHEMA = "polymarket-profile-held-out-predictions-v1"
_MODEL_SAMPLE_SCHEMA = "polymarket-model-sample-v4"
_PUBLICATION_SCHEMA = "polymarket-model-publication-v1"
_MODEL_SCHEMA = "polymarket-market-anchored-logit-v4"
_PROBABILITY_SCHEMA = "polymarket-probability-report-v2"
_AI_CASE_SCHEMA = "polymarket-ai-veto-case-v3"
_AI_REPORT_SCHEMA = "polymarket-ai-veto-report-v5"
_ASSETS = ("BTC", "ETH", "SOL")
_POLICIES = ("baseline", "model", "profile_model", "model_retry", "ai")
_AI_MICROSTRUCTURE_FIELDS = (
    "direct_distance_from_chainlink_open_bps",
    "direct_chainlink_basis_bps",
    "direct_return_100ms_bps",
    "direct_return_250ms_bps",
    "direct_return_1000ms_bps",
    "direct_return_5000ms_bps",
    "direct_realized_volatility_100ms_bps",
    "direct_realized_volatility_1000ms_bps",
    "direct_realized_volatility_5000ms_bps",
    "direct_diffusion_market_logit_gap",
    "chainlink_diffusion_market_logit_gap",
    "direct_trade_imbalance_100ms",
    "direct_trade_imbalance_250ms",
    "direct_trade_imbalance_1000ms",
    "direct_trade_imbalance_5000ms",
    "direct_top_imbalance",
    "direct_spread_bps",
    "up_microprice_deviation_bps",
    "down_microprice_deviation_bps",
    "up_top_imbalance",
    "down_top_imbalance",
    "outcome_midpoint_sum_error_bps",
    "executable_ask_pair_premium_bps",
    "executable_bid_pair_discount_bps",
)
_AI_FRESHNESS_FIELDS = (
    "up_book_age_ms",
    "down_book_age_ms",
    "direct_binance_age_ms",
    "chainlink_source_age_ms",
    "chainlink_arrival_age_ms",
    "chainlink_anchor_gap_ms",
)
_AI_REASON_CODES = {
    "edge_after_fees",
    "weak_probability_uplift",
    "market_disagreement",
    "liquidity_stress",
    "latency_risk",
    "source_staleness",
    "volatile_regime",
    "orderbook_imbalance",
    "model_calibration_risk",
    "insufficient_evidence",
    "cooldown_required",
}
_COLORS = {
    "background": "#0b1220",
    "panel": "#111c2e",
    "grid": "#26364d",
    "text": "#f8fafc",
    "muted": "#a9b8cc",
    "baseline": "#94a3b8",
    "model": "#22c55e",
    "profile_model": "#f97316",
    "model_retry": "#eab308",
    "ai": "#38bdf8",
    "BTC": "#f59e0b",
    "ETH": "#8b5cf6",
    "SOL": "#14b8a6",
    "negative": "#fb7185",
}
_POLICY_DASH = {
    "baseline": "",
    "model": "12 5",
    "profile_model": "7 4",
    "model_retry": "3 5",
    "ai": "14 4 3 4",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    _atomic_write(path, value.replace("\r\n", "\n").encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"publication table has no rows: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"publication table columns drifted: {path.name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, stream.getvalue())


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _as_rows(value: object, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return [_as_mapping(item, f"{name} row") for item in value]


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _decimal(value: object, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _utc(timestamp_ms: object) -> str:
    timestamp = int(timestamp_ms)
    return (
        datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _verify_claims(value: Mapping[str, Any], *, name: str) -> None:
    for key in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        if key in value and value[key] is not False:
            raise ValueError(f"{name} makes a forbidden {key} claim")


def _verify_embedded_digest(
    value: Mapping[str, Any],
    digest_key: str,
    *,
    name: str,
) -> None:
    canonical = dict(value)
    claimed = str(canonical.pop(digest_key, ""))
    if len(claimed) != 64 or claimed != _canonical_sha256(canonical):
        raise ValueError(f"{name} {digest_key} is invalid")


def _named_losses(value: object, name: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    rows: list[tuple[str, float]] = []
    for raw in value:
        if not isinstance(raw, list) or len(raw) != 2 or not str(raw[0]):
            raise ValueError(f"{name} row is malformed")
        rows.append((str(raw[0]), _finite_float(raw[1], f"{name} loss")))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"{name} candidate names are not unique")
    return tuple(rows)


def _probability_metrics_from_rows(
    rows: Sequence[Mapping[str, Any]],
    probability_key: str,
) -> dict[str, float | int]:
    total_weight = 0.0
    log_loss = 0.0
    brier = 0.0
    accuracy = 0.0
    sharpness = 0.0
    bins = [[0.0, 0.0, 0.0] for _ in range(10)]
    for row in rows:
        probability = _finite_float(row.get(probability_key), probability_key)
        weight = _finite_float(row.get("market_weight"), "market weight")
        label = 1.0 if row.get("official_up") is True else 0.0
        total_weight += weight
        log_loss += weight * -(
            label * math.log(probability) + (1.0 - label) * math.log1p(-probability)
        )
        brier += weight * (probability - label) ** 2
        accuracy += weight * float((probability >= 0.5) == bool(label))
        sharpness += weight * abs(probability - 0.5)
        bin_index = min(int(probability * 10), 9)
        bins[bin_index][0] += weight
        bins[bin_index][1] += weight * label
        bins[bin_index][2] += weight * probability
    if total_weight <= 0.0:
        raise ValueError("held-out probability evidence has no effective weight")
    calibration_error = 0.0
    for bin_weight, observed_sum, predicted_sum in bins:
        if bin_weight > 0.0:
            calibration_error += (bin_weight / total_weight) * abs(
                observed_sum / bin_weight - predicted_sum / bin_weight
            )
    return {
        "row_count": len(rows),
        "market_count": len({str(row["condition_id"]) for row in rows}),
        "time_group_count": len({int(row["event_start_ms"]) for row in rows}),
        "effective_market_weight": total_weight,
        "weighted_log_loss": log_loss / total_weight,
        "weighted_brier_score": brier / total_weight,
        "weighted_calibration_error": calibration_error,
        "weighted_accuracy": accuracy / total_weight,
        "weighted_sharpness": sharpness / total_weight,
    }


def _validate_probability_metrics(
    reported: Mapping[str, Any],
    expected: Mapping[str, float | int],
    *,
    name: str,
) -> None:
    for key in ("row_count", "market_count", "time_group_count"):
        if int(reported.get(key, -1)) != int(expected[key]):
            raise ValueError(f"{name} {key} does not reconcile")
    for key in (
        "effective_market_weight",
        "weighted_log_loss",
        "weighted_brier_score",
        "weighted_calibration_error",
        "weighted_accuracy",
        "weighted_sharpness",
    ):
        if not math.isclose(
            _finite_float(reported.get(key), f"{name} {key}"),
            float(expected[key]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} {key} does not reconcile")


def _validate_nested_model_selection(
    model: Mapping[str, Any],
    probability: Mapping[str, Any],
    split: Mapping[str, Any],
) -> None:
    if (
        model.get("schema_version") != _MODEL_SCHEMA
        or probability.get("schema_version") != _PROBABILITY_SCHEMA
    ):
        raise ValueError("unsupported nested Polymarket model schema")
    if model.get("model_sha256") != probability.get("model_sha256") or model.get(
        "selected_candidate"
    ) != probability.get("selected_candidate"):
        raise ValueError("nested Polymarket model report is inconsistent")

    config = _as_mapping(model.get("config"), "model config")
    raw_l2 = config.get("l2_candidates")
    if not isinstance(raw_l2, list) or not raw_l2:
        raise ValueError("nested model L2 candidates are missing")
    l2_candidates = tuple(_finite_float(value, "L2 candidate") for value in raw_l2)
    if l2_candidates != tuple(sorted(set(l2_candidates))) or min(l2_candidates) <= 0:
        raise ValueError("nested model L2 candidates are invalid")
    candidate_names = tuple(
        f"offset_l2_{format(value, '.17g')}" for value in l2_candidates
    )
    inner_losses = _named_losses(
        model.get("candidate_inner_log_losses"),
        "inner model losses",
    )
    if tuple(name for name, _loss in inner_losses) != (
        "market_baseline",
        *candidate_names,
    ):
        raise ValueError("nested model inner candidate set drifted")
    expected_inner_name, _expected_inner_loss = min(
        inner_losses[1:],
        key=lambda item: (
            item[1],
            -float(item[0].removeprefix("offset_l2_")),
            item[0],
        ),
    )
    inner_selected = str(model.get("inner_selected_candidate", ""))
    if inner_selected != expected_inner_name:
        raise ValueError("nested model inner selection is inconsistent")

    validation_losses = _named_losses(
        model.get("validation_gate_log_losses"),
        "outer validation gate losses",
    )
    if tuple(name for name, _loss in validation_losses) != (
        "market_baseline",
        inner_selected,
    ):
        raise ValueError("nested model validation gate candidate set drifted")
    required_improvement = _finite_float(
        config.get("minimum_validation_log_loss_improvement"),
        "minimum validation improvement",
    )
    gate_passed = validation_losses[1][1] <= (
        validation_losses[0][1] - required_improvement
    )
    expected_selected = inner_selected if gate_passed else "market_baseline"
    selected = str(model.get("selected_candidate", ""))
    selected_l2 = model.get("selected_l2")
    if (
        selected != expected_selected
        or (selected == "market_baseline" and selected_l2 is not None)
        or (
            selected != "market_baseline"
            and _finite_float(selected_l2, "selected L2")
            != float(selected.removeprefix("offset_l2_"))
        )
    ):
        raise ValueError("nested model promotion gate is inconsistent")
    if selected == "market_baseline" and any(
        _finite_float(value, "fallback coefficient") != 0.0
        for value in model.get("coefficients", ())
    ):
        raise ValueError("nested model fallback coefficients are not zero")

    train_groups = tuple(int(value) for value in split.get("train_group_starts_ms", ()))
    if not train_groups or train_groups != tuple(sorted(set(train_groups))):
        raise ValueError("nested model training groups are invalid")
    fold_count = int(config.get("inner_fold_count", -1))
    validation_size = int(config.get("inner_validation_time_groups", -1))
    purge_size = int(config.get("inner_purge_time_groups", -1))
    minimum_train = int(config.get("minimum_inner_train_time_groups", -1))
    raw_boundaries = model.get("inner_fold_boundaries_ms")
    if (
        not isinstance(raw_boundaries, list)
        or int(model.get("inner_fold_count", -1)) != fold_count
        or len(raw_boundaries) != fold_count
        or int(model.get("training_time_group_count", -1)) != len(train_groups)
    ):
        raise ValueError("nested model fold count is inconsistent")
    first_validation = len(train_groups) - fold_count * validation_size
    expected_boundaries: list[list[int]] = []
    for fold_index in range(fold_count):
        validation_start = first_validation + fold_index * validation_size
        train_end = validation_start - purge_size
        if (
            validation_start < 0
            or train_end < minimum_train
            or validation_start + validation_size > len(train_groups)
        ):
            raise ValueError("nested model fold configuration exceeds training data")
        expected_boundaries.append(
            [
                train_groups[0],
                train_groups[train_end - 1],
                train_groups[validation_start],
                train_groups[validation_start + validation_size - 1],
            ]
        )
    if raw_boundaries != expected_boundaries:
        raise ValueError("nested model fold boundaries are not chronological")


def _profile_candidate_details(name: str) -> tuple[str, tuple[str, ...], float]:
    for profile, feature_names in POLYMARKET_PROFILE_FEATURES:
        for l2 in POLYMARKET_PROFILE_L2_CANDIDATES:
            candidate = f"offset_profile_{profile}_l2_{format(l2, '.17g')}"
            if name == candidate:
                return profile, feature_names, l2
    raise ValueError("unknown frozen Polymarket profile candidate")


def _validate_profile_model_selection(
    profile_model: Mapping[str, Any],
    profile_probability: Mapping[str, Any],
    control_model: Mapping[str, Any],
    control_probability: Mapping[str, Any],
) -> None:
    _verify_claims(profile_model, name="profile model")
    _verify_claims(profile_probability, name="profile probability report")
    _verify_embedded_digest(profile_model, "model_sha256", name="profile model")
    _verify_embedded_digest(
        profile_probability,
        "report_sha256",
        name="profile probability report",
    )
    if (
        profile_model.get("schema_version") != POLYMARKET_PROFILE_MODEL_SCHEMA_VERSION
        or profile_probability.get("schema_version")
        != POLYMARKET_PROFILE_REPORT_SCHEMA_VERSION
        or profile_model.get("contract_sha256") != POLYMARKET_PROFILE_CONTRACT_SHA256
        or profile_probability.get("contract_sha256")
        != POLYMARKET_PROFILE_CONTRACT_SHA256
        or profile_model.get("control_model_sha256")
        != control_model.get("model_sha256")
        or profile_probability.get("control_model_sha256")
        != control_model.get("model_sha256")
        or profile_probability.get("challenger_model_sha256")
        != profile_model.get("model_sha256")
        or profile_probability.get("selected_candidate")
        != profile_model.get("selected_candidate")
        or _canonical_json(profile_model.get("config"))
        != _canonical_json(control_model.get("config"))
        or tuple(profile_model.get("feature_names", ()))
        != POLYMARKET_MODEL_FEATURE_NAMES
        or profile_model.get("inner_fold_boundaries_ms")
        != control_model.get("inner_fold_boundaries_ms")
    ):
        raise ValueError("Polymarket profile model provenance is inconsistent")
    if _canonical_json(profile_probability.get("baseline_metrics")) != _canonical_json(
        control_probability.get("baseline_metrics")
    ) or _canonical_json(profile_probability.get("control_metrics")) != _canonical_json(
        control_probability.get("model_metrics")
    ):
        raise ValueError("Polymarket profile control metrics drifted")

    candidate_names = tuple(
        f"offset_profile_{profile}_l2_{format(l2, '.17g')}"
        for profile, _feature_names in POLYMARKET_PROFILE_FEATURES
        for l2 in POLYMARKET_PROFILE_L2_CANDIDATES
    )
    inner_losses = _named_losses(
        profile_model.get("candidate_inner_log_losses"),
        "profile inner model losses",
    )
    if tuple(name for name, _loss in inner_losses) != (
        "market_baseline",
        *candidate_names,
    ):
        raise ValueError("Polymarket profile candidate grid drifted")

    def selection_key(item: tuple[str, float]) -> tuple[float, int, float, str]:
        profile, feature_names, l2 = _profile_candidate_details(item[0])
        return item[1], len(feature_names), -l2, profile

    inner_selected, _inner_loss = min(inner_losses[1:], key=selection_key)
    if profile_model.get("inner_selected_candidate") != inner_selected:
        raise ValueError("Polymarket profile inner selection is inconsistent")
    validation_losses = _named_losses(
        profile_model.get("validation_gate_log_losses"),
        "profile validation gate losses",
    )
    if tuple(name for name, _loss in validation_losses) != (
        "market_baseline",
        inner_selected,
    ):
        raise ValueError("Polymarket profile validation gate drifted")
    config = _as_mapping(profile_model.get("config"), "profile model config")
    required = _finite_float(
        config.get("minimum_validation_log_loss_improvement"),
        "profile minimum validation improvement",
    )
    if (
        tuple(_finite_float(value, "profile L2") for value in config["l2_candidates"])
        != POLYMARKET_PROFILE_L2_CANDIDATES
        or int(config.get("inner_fold_count", -1)) != 3
        or not math.isclose(required, 0.0001, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ValueError("Polymarket profile frozen grid is inconsistent")
    gate_passed = validation_losses[1][1] <= validation_losses[0][1] - required
    expected_selected = inner_selected if gate_passed else "market_baseline"
    selected = str(profile_model.get("selected_candidate", ""))
    coefficients = tuple(
        _finite_float(value, "profile coefficient")
        for value in profile_model.get("coefficients", ())
    )
    width = len(POLYMARKET_MODEL_FEATURE_NAMES)
    parameter_arrays = tuple(
        tuple(
            _finite_float(value, f"profile {name}")
            for value in profile_model.get(name, ())
        )
        for name in (
            "winsor_lower",
            "winsor_upper",
            "robust_center",
            "robust_scale",
        )
    )
    if (
        selected != expected_selected
        or len(coefficients) != width + 1
        or any(len(values) != width for values in parameter_arrays)
    ):
        raise ValueError("Polymarket profile selected parameters are inconsistent")
    if selected == "market_baseline":
        if (
            profile_model.get("selected_profile") is not None
            or profile_model.get("selected_feature_names") != []
            or profile_model.get("selected_l2") is not None
            or any(value != 0.0 for value in coefficients)
        ):
            raise ValueError("Polymarket profile fallback is inconsistent")
    else:
        profile, feature_names, l2 = _profile_candidate_details(selected)
        selected_names = tuple(profile_model.get("selected_feature_names", ()))
        if (
            profile_model.get("selected_profile") != profile
            or selected_names != feature_names
            or _finite_float(profile_model.get("selected_l2"), "profile selected L2")
            != l2
        ):
            raise ValueError("Polymarket profile accepted candidate is inconsistent")
        active = set(feature_names)
        if any(
            coefficients[index + 1] != 0.0
            for index, name in enumerate(POLYMARKET_MODEL_FEATURE_NAMES)
            if name not in active
        ):
            raise ValueError("Polymarket excluded profile coefficients are not zero")

    baseline_roles = _as_mapping(
        profile_probability.get("baseline_metrics"),
        "profile baseline metrics",
    )
    control_roles = _as_mapping(
        profile_probability.get("control_metrics"),
        "profile control metrics",
    )
    challenger_roles = _as_mapping(
        profile_probability.get("challenger_metrics"),
        "profile challenger metrics",
    )
    if set(baseline_roles) != {"train", "validation", "test"} or set(
        challenger_roles
    ) != {"train", "validation", "test"}:
        raise ValueError("Polymarket profile probability roles are incomplete")
    selected_validation_loss = (
        validation_losses[1][1] if gate_passed else validation_losses[0][1]
    )
    comparisons = (
        (
            _finite_float(
                _as_mapping(
                    baseline_roles["validation"],
                    "profile baseline validation metrics",
                )["weighted_log_loss"],
                "profile baseline validation log loss",
            ),
            validation_losses[0][1],
        ),
        (
            _finite_float(
                _as_mapping(
                    challenger_roles["validation"],
                    "profile challenger validation metrics",
                )["weighted_log_loss"],
                "profile challenger validation log loss",
            ),
            selected_validation_loss,
        ),
        (
            _finite_float(
                profile_probability.get("validation_log_loss_delta_vs_control"),
                "profile validation delta",
            ),
            selected_validation_loss
            - _finite_float(
                _as_mapping(
                    control_roles["validation"],
                    "profile control validation metrics",
                )["weighted_log_loss"],
                "profile control validation log loss",
            ),
        ),
    )
    if any(
        not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        for actual, expected in comparisons
    ):
        raise ValueError("Polymarket profile validation scores do not reconcile")


@dataclass(frozen=True)
class ValidatedPolymarketArtifact:
    payload: Mapping[str, Any]
    artifact_sha256: str
    predictions: tuple[Mapping[str, Any], ...]
    profile_predictions: tuple[Mapping[str, Any], ...]
    executions: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PolymarketPublicationResult:
    round_number: int
    research_root: str
    source_artifact: str
    artifact_sha256: str
    generated_files: tuple[str, ...]
    manifest_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": _PUBLICATION_SCHEMA,
            "round": self.round_number,
            "research_root": self.research_root,
            "source_artifact": self.source_artifact,
            "artifact_sha256": self.artifact_sha256,
            "generated_files": list(self.generated_files),
            "manifest_sha256": self.manifest_sha256,
        }


def _validate_execution_report(
    report: Mapping[str, Any],
    *,
    conditions: set[str],
    expected_time_group_count: int,
    name: str,
    expected_schema: str = POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION,
) -> None:
    if report.get("schema_version") != expected_schema:
        raise ValueError(f"{name} uses an unsupported execution schema")
    if (
        report.get("causal_settlement_contract_sha256")
        != POLYMARKET_CAUSAL_SETTLEMENT_CONTRACT_SHA256
    ):
        raise ValueError(f"{name} does not bind the causal settlement contract")
    _verify_claims(report, name=name)
    _verify_embedded_digest(report, "report_sha256", name=name)
    trades_value = report.get("trades")
    equity_value = report.get("equity_curve")
    if not isinstance(trades_value, list) or not isinstance(equity_value, list):
        raise ValueError(f"{name} execution ledger is malformed")
    trades = [_as_mapping(row, f"{name} trade") for row in trades_value]
    equity = [_as_mapping(row, f"{name} equity point") for row in equity_value]
    permissions = _as_mapping(
        report.get("market_permissions"),
        f"{name} market permissions",
    )
    delays = _as_mapping(
        report.get("decision_delay_ms_by_condition"),
        f"{name} decision delays",
    )
    if (
        set(permissions) != conditions
        or any(not isinstance(value, bool) for value in permissions.values())
        or set(delays) != conditions
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 300_000
            for value in delays.values()
        )
        or report.get("market_permission_sha256")
        != _canonical_sha256(
            {
                "schema_version": "polymarket-market-permission-v1",
                "permissions": dict(sorted(permissions.items())),
            }
        )
        or report.get("decision_delay_input_sha256")
        != _canonical_sha256(
            {
                "schema_version": "polymarket-decision-delay-input-v1",
                "decision_delay_ms_by_condition": dict(sorted(delays.items())),
            }
        )
    ):
        raise ValueError(f"{name} permission or latency map is invalid")
    config = _as_mapping(report.get("config"), f"{name} execution config")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "submission_latency_ms",
            "maximum_execution_observation_delay_ms",
            "maximum_book_age_ms",
            "order_ttl_ms",
            "minimum_expected_edge_per_contract",
            "initial_capital_quote",
            "maximum_loss_fraction_per_market",
            "maximum_loss_fraction_per_time_group",
        },
        name=f"{name} execution config",
    )
    submission_latency = int(config["submission_latency_ms"])
    maximum_observation_delay = int(config["maximum_execution_observation_delay_ms"])
    maximum_book_age = int(config["maximum_book_age_ms"])
    order_ttl = int(config["order_ttl_ms"])
    minimum_edge = _decimal(
        config["minimum_expected_edge_per_contract"],
        f"{name} minimum edge",
    )
    initial_capital = _decimal(config["initial_capital_quote"], f"{name} capital")
    per_market_risk = _decimal(
        config["maximum_loss_fraction_per_market"],
        f"{name} per-market risk",
    )
    per_group_risk = _decimal(
        config["maximum_loss_fraction_per_time_group"],
        f"{name} per-group risk",
    )
    if (
        config.get("schema_version") != "polymarket-execution-config-v2"
        or not 1 <= submission_latency <= 60_000
        or not 0 <= maximum_observation_delay <= 60_000
        or not 0 <= maximum_book_age <= 60_000
        or not 1_000 <= order_ttl <= 300_000
        or not Decimal("0") <= minimum_edge <= Decimal("0.25")
        or not Decimal("10") <= initial_capital <= Decimal("1000000000")
        or not Decimal("0") < per_market_risk <= Decimal("0.10")
        or not per_market_risk <= per_group_risk <= Decimal("0.30")
    ):
        raise ValueError(f"{name} execution configuration is invalid")
    if (
        int(report.get("evaluated_market_count", -1)) != len(conditions)
        or int(report.get("attempted_order_count", -1)) != len(trades)
        or len(equity) != expected_time_group_count
    ):
        raise ValueError(f"{name} execution coverage is inconsistent")

    filled = [row for row in trades if row.get("execution_state") == "FILLED"]
    settled = [
        row for row in filled if str(row.get("official_resolution_event_id", ""))
    ]
    wins = [row for row in filled if _decimal(row["realized_pnl_quote"], "PnL") > 0]
    losses = [row for row in filled if _decimal(row["realized_pnl_quote"], "PnL") <= 0]
    if (
        int(report.get("filled_order_count", -1)) != len(filled)
        or len(filled) != len(settled)
        or int(report.get("winning_order_count", -1)) != len(wins)
        or int(report.get("losing_order_count", -1)) != len(losses)
    ):
        raise ValueError(f"{name} contains unsettled or miscounted filled trades")

    deployed = Decimal("0")
    payouts = Decimal("0")
    fees = Decimal("0")
    realized = Decimal("0")
    trade_ids: set[str] = set()
    trade_pnl_by_group: dict[int, Decimal] = {}
    for trade in trades:
        condition_id = str(trade.get("condition_id", ""))
        trade_id = str(trade.get("trade_id", ""))
        trade_sha256 = str(trade.get("trade_sha256", ""))
        decision_delay = int(trade.get("decision_delay_ms", -1))
        requested_latency = decision_delay + submission_latency
        effective_latency = int(trade.get("effective_latency_ms", -1))
        trade_identity = dict(trade)
        trade_identity.pop("trade_sha256", None)
        if (
            str(trade.get("asset", "")) not in _ASSETS
            or condition_id not in conditions
            or len(trade_id) != 64
            or trade_id in trade_ids
            or len(trade_sha256) != 64
            or trade_sha256 != _canonical_sha256(trade_identity)
            or decision_delay != delays[condition_id]
            or int(trade.get("submission_latency_ms", -1)) != submission_latency
            or not (
                requested_latency
                <= effective_latency
                <= requested_latency + maximum_observation_delay
            )
        ):
            raise ValueError(f"{name} trade identity or latency binding is invalid")
        trade_ids.add(trade_id)
        quantity = _decimal(trade.get("quantity"), f"{name} trade quantity")
        filled_quantity = _decimal(
            trade.get("filled_quantity"),
            f"{name} filled quantity",
        )
        average = _decimal(
            trade.get("average_fill_price"),
            f"{name} fill price",
        )
        fee = _decimal(trade.get("fee_quote"), f"{name} trade fee")
        payout = _decimal(
            trade.get("gross_payout_quote"),
            f"{name} trade payout",
        )
        pnl = _decimal(
            trade.get("realized_pnl_quote"),
            f"{name} trade PnL",
        )
        if min(quantity, filled_quantity, average, fee, payout) < 0:
            raise ValueError(f"{name} trade contains a negative accounting value")
        if trade.get("execution_state") == "FILLED":
            if filled_quantity != quantity or pnl != payout - average * quantity - fee:
                raise ValueError(f"{name} filled-trade accounting is inconsistent")
            deployed += average * quantity + fee
            payouts += payout
            fees += fee
            realized += pnl
            event_start = int(trade.get("event_start_ms", -1))
            trade_pnl_by_group[event_start] = (
                trade_pnl_by_group.get(event_start, Decimal("0")) + pnl
            )
        elif any(value != 0 for value in (filled_quantity, average, fee, payout, pnl)):
            raise ValueError(f"{name} non-fill carries economic credit")

    initial = _decimal(report.get("initial_capital_quote"), f"{name} initial equity")
    final = _decimal(report.get("final_equity_quote"), f"{name} final equity")
    reported_net = _decimal(report.get("net_realized_pnl_quote"), f"{name} net PnL")
    if (
        deployed
        != _decimal(report.get("gross_deployed_capital_quote"), f"{name} deployed")
        or payouts != _decimal(report.get("gross_payout_quote"), f"{name} payouts")
        or fees != _decimal(report.get("total_fees_quote"), f"{name} fees")
        or realized != reported_net
        or final != initial + reported_net
    ):
        raise ValueError(f"{name} aggregate accounting does not reconcile")
    expected_initial_return = reported_net / initial
    expected_deployed_return = (
        reported_net / deployed if deployed > 0 else Decimal("0")
    )
    if (
        initial != initial_capital
        or _decimal(
            report.get("return_on_initial_capital"),
            f"{name} return on initial capital",
        )
        != expected_initial_return
        or _decimal(
            report.get("return_on_deployed_capital"),
            f"{name} return on deployed capital",
        )
        != expected_deployed_return
    ):
        raise ValueError(f"{name} return accounting does not reconcile")

    running = initial
    peak = initial
    maximum_drawdown = Decimal("0")
    maximum_drawdown_fraction = Decimal("0")
    chronology: list[tuple[int, int]] = []
    event_starts: list[int] = []
    for point in equity:
        event_start = int(point.get("event_start_ms", -1))
        settled_at = int(point.get("settled_at_ms", -1))
        group_pnl = _decimal(point.get("group_realized_pnl_quote"), "group PnL")
        running += group_pnl
        peak = max(peak, running)
        drawdown = peak - running
        drawdown_fraction = drawdown / peak if peak > 0 else Decimal("0")
        if (
            event_start < 0
            or settled_at < event_start
            or group_pnl != trade_pnl_by_group.get(event_start, Decimal("0"))
            or running != _decimal(point.get("equity_quote"), "equity")
            or peak != _decimal(point.get("peak_equity_quote"), "peak equity")
            or drawdown != _decimal(point.get("drawdown_quote"), "drawdown")
            or drawdown_fraction
            != _decimal(point.get("drawdown_fraction"), "drawdown fraction")
        ):
            raise ValueError(f"{name} equity curve does not reconcile")
        maximum_drawdown = max(maximum_drawdown, drawdown)
        maximum_drawdown_fraction = max(
            maximum_drawdown_fraction,
            drawdown_fraction,
        )
        chronology.append((settled_at, event_start))
        event_starts.append(event_start)
    if (
        chronology != sorted(chronology)
        or len(event_starts) != len(set(event_starts))
        or len(event_starts) != expected_time_group_count
        or running != final
        or maximum_drawdown
        != _decimal(report.get("maximum_drawdown_quote"), "maximum drawdown")
        or maximum_drawdown_fraction
        != _decimal(
            report.get("maximum_drawdown_fraction"),
            "maximum drawdown fraction",
        )
    ):
        raise ValueError(f"{name} equity path summary does not reconcile")


def _validate_execution_probability_binding(
    report: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    name: str,
) -> None:
    expected = _canonical_sha256(
        {
            "schema_version": "polymarket-probability-input-v1",
            "sample_ids": [str(row["sample_id"]) for row in predictions],
            "probabilities": [
                format(
                    _finite_float(row[probability_key], f"{name} probability"),
                    ".17g",
                )
                for row in predictions
            ],
            "market_permission_sha256": report["market_permission_sha256"],
            "decision_delay_input_sha256": report["decision_delay_input_sha256"],
        }
    )
    if report.get("probability_input_sha256") != expected:
        raise ValueError(f"{name} probability input does not reconstruct")


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields do not match the frozen schema")


def _validate_ai_prompt_shape(prompt: Mapping[str, Any]) -> None:
    _require_exact_keys(
        prompt,
        {
            "schema_version",
            "task",
            "asset",
            "five_minute_market",
            "remaining_seconds",
            "proposed_outcome",
            "model_probability",
            "market_implied_probability",
            "model_probability_uplift",
            "decision_best_ask",
            "protective_limit_price",
            "expected_edge_per_contract_after_fee",
            "minimum_required_edge_per_contract",
            "maximum_loss_fraction_per_market",
            "maximum_loss_fraction_per_time_group",
            "assumed_submission_latency_ms",
            "microstructure",
            "source_freshness_ms",
            "liquidity_context",
            "validation_only_model_evidence",
            "hard_constraints",
        },
        name="AI prompt",
    )
    microstructure = _as_mapping(prompt["microstructure"], "AI microstructure")
    freshness = _as_mapping(prompt["source_freshness_ms"], "AI freshness")
    liquidity = _as_mapping(prompt["liquidity_context"], "AI liquidity")
    validation = _as_mapping(
        prompt["validation_only_model_evidence"],
        "AI validation-only evidence",
    )
    constraints = _as_mapping(prompt["hard_constraints"], "AI hard constraints")
    _require_exact_keys(
        microstructure,
        set(_AI_MICROSTRUCTURE_FIELDS),
        name="AI microstructure",
    )
    _require_exact_keys(freshness, set(_AI_FRESHNESS_FIELDS), name="AI freshness")
    _require_exact_keys(
        liquidity,
        {
            "proposed_outcome_ask_depth_3_contracts",
            "market_liquidity_quote",
            "market_volume_quote",
        },
        name="AI liquidity",
    )
    _require_exact_keys(
        validation,
        {
            "market_baseline_log_loss",
            "residual_model_log_loss",
            "log_loss_delta",
            "validation_market_count",
        },
        name="AI validation-only evidence",
    )
    expected_constraints = {
        "cannot_create_or_reverse_trade": True,
        "cannot_increase_size_or_limit": True,
        "invalid_or_uncertain_response_means_veto": True,
    }
    if dict(constraints) != expected_constraints:
        raise ValueError("AI prompt hard constraints were weakened")
    if (
        prompt.get("schema_version") != _AI_CASE_SCHEMA
        or prompt.get("task") != "veto_only_review_of_frozen_ml_proposal"
        or prompt.get("asset") not in _ASSETS
        or prompt.get("five_minute_market") is not True
        or prompt.get("proposed_outcome") not in {"Up", "Down"}
        or isinstance(prompt.get("remaining_seconds"), bool)
        or int(prompt.get("remaining_seconds", -1)) not in {30, 60, 120, 180, 240}
        or isinstance(prompt.get("assumed_submission_latency_ms"), bool)
        or not 1 <= int(prompt.get("assumed_submission_latency_ms", -1)) <= 60_000
    ):
        raise ValueError("AI prompt contract is invalid")
    model_probability = _finite_float(
        prompt.get("model_probability"),
        "AI model probability",
    )
    market_probability = _finite_float(
        prompt.get("market_implied_probability"),
        "AI market probability",
    )
    uplift = _finite_float(prompt.get("model_probability_uplift"), "AI uplift")
    if not (
        0.0 < model_probability < 1.0
        and 0.0 < market_probability < 1.0
        and math.isclose(
            uplift,
            model_probability - market_probability,
            rel_tol=0.0,
            abs_tol=2e-8,
        )
    ):
        raise ValueError("AI prompt probabilities are inconsistent")
    for field in (
        "decision_best_ask",
        "protective_limit_price",
        "expected_edge_per_contract_after_fee",
        "minimum_required_edge_per_contract",
        "maximum_loss_fraction_per_market",
        "maximum_loss_fraction_per_time_group",
    ):
        _decimal(prompt.get(field), f"AI prompt {field}")
    if any(
        not math.isfinite(_finite_float(value, f"AI microstructure {name}"))
        for name, value in microstructure.items()
    ):
        raise ValueError("AI prompt microstructure is not finite")
    if any(
        _finite_float(value, f"AI freshness {name}") < 0.0
        for name, value in freshness.items()
    ) or any(
        _finite_float(value, f"AI liquidity {name}") < 0.0
        for name, value in liquidity.items()
    ):
        raise ValueError("AI prompt freshness or liquidity is outside its domain")
    for name in (
        "market_baseline_log_loss",
        "residual_model_log_loss",
        "log_loss_delta",
    ):
        _finite_float(validation.get(name), f"AI validation evidence {name}")
    if (
        isinstance(validation.get("validation_market_count"), bool)
        or int(validation.get("validation_market_count", -1)) < 1
    ):
        raise ValueError("AI validation market count is invalid")


def _bounded_profit_factor(gains: float, losses: float) -> float:
    if losses > 0.0:
        return min(999.0, max(0.0, gains / losses))
    return 999.0 if gains > 0.0 else 0.0


def _bounded_downside_return_risk_ratio(
    return_fraction: float,
    drawdown: float,
) -> float:
    if drawdown > 0.0:
        return min(999.0, max(-999.0, return_fraction / drawdown))
    if return_fraction > 0.0:
        return 999.0
    return -999.0 if return_fraction < 0.0 else 0.0


def _execution_uplift_metrics(
    report: Mapping[str, Any],
    *,
    dataset_fingerprint: str,
) -> dict[str, object]:
    trades = [
        _as_mapping(item, "AI uplift trade")
        for item in report.get("trades", ())
        if isinstance(item, Mapping) and item.get("execution_state") == "FILLED"
    ]
    values = [
        float(_decimal(item["realized_pnl_quote"], "AI uplift trade PnL"))
        for item in trades
    ]
    gains = sum(value for value in values if value > 0.0)
    losses = abs(sum(value for value in values if value < 0.0))
    loss_streak = 0
    maximum_loss_streak = 0
    for value in values:
        loss_streak = loss_streak + 1 if value < 0.0 else 0
        maximum_loss_streak = max(maximum_loss_streak, loss_streak)
    drawdown = _finite_float(
        report.get("maximum_drawdown_fraction"),
        "AI uplift drawdown",
    )
    net = _finite_float(report.get("net_realized_pnl_quote"), "AI uplift PnL")
    return_fraction = _finite_float(
        report.get("return_on_initial_capital"),
        "AI uplift return",
    )
    return {
        "realized_pnl": net,
        "roi_pct": 100.0 * return_fraction,
        "max_drawdown": drawdown,
        "expectancy": net / len(values) if values else 0.0,
        "profit_factor": _bounded_profit_factor(gains, losses),
        "closed_trades": len(values),
        "win_rate": (
            sum(value > 0.0 for value in values) / len(values) if values else 0.0
        ),
        "liquidation_events": 0,
        "max_consecutive_losses": maximum_loss_streak,
        "downside_return_risk_ratio": _bounded_downside_return_risk_ratio(
            return_fraction,
            drawdown,
        ),
        "dataset_fingerprint": dataset_fingerprint,
        "evidence_sha256": str(report.get("report_sha256", "")),
    }


def _matched_ai_uplift_periods(
    predictions: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    ai: Mapping[str, Any],
) -> list[dict[str, object]]:
    starts = tuple(sorted({int(row["event_start_ms"]) for row in predictions}))
    if not starts:
        raise ValueError("AI uplift prediction periods are empty")
    baseline_capital = _decimal(
        baseline.get("initial_capital_quote"),
        "baseline initial capital",
    )
    ai_capital = _decimal(ai.get("initial_capital_quote"), "AI initial capital")
    if baseline_capital <= 0 or ai_capital != baseline_capital:
        raise ValueError("AI uplift initial capital differs")

    def returns_by_start(
        report: Mapping[str, Any],
        label: str,
    ) -> dict[int, float]:
        result: dict[int, float] = {}
        curve = report.get("equity_curve")
        if not isinstance(curve, list):
            raise ValueError(f"{label} equity curve is invalid")
        for raw in curve:
            item = _as_mapping(raw, f"{label} equity point")
            start_ms = int(item["event_start_ms"])
            if start_ms in result:
                raise ValueError(f"{label} equity periods are duplicated")
            group_pnl = _decimal(
                item["group_realized_pnl_quote"],
                f"{label} group PnL",
            )
            result[start_ms] = float(group_pnl / baseline_capital)
        if set(result) != set(starts):
            raise ValueError(f"{label} equity periods differ")
        return result

    baseline_by_start = returns_by_start(baseline, "baseline")
    ai_by_start = returns_by_start(ai, "AI")
    return [
        {
            "scope": "polymarket_btc_eth_sol_five_minute_test",
            "period_start_ms": start_ms,
            "period_end_ms": start_ms + 300_000,
            "baseline_return": baseline_by_start[start_ms],
            "ai_return": ai_by_start[start_ms],
        }
        for start_ms in starts
    ]


def _strict_ai_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("duplicate AI response JSON key")
        parsed[key] = value
    return parsed


def _ai_action_reasons_coherent(action: str, codes: Sequence[str]) -> bool:
    selected = set(codes)
    if action == "approve":
        return selected == {"edge_after_fees"}
    if action == "cooldown":
        return "cooldown_required" in selected
    if action == "veto":
        return bool(selected - {"edge_after_fees"})
    return False


def _parsed_valid_ai_response(response: object) -> dict[str, object] | None:
    if not isinstance(response, Mapping):
        return None
    message = response.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        return None
    content = message["content"]
    if len(content) > 4_096:
        return None
    try:
        parsed = json.loads(content, object_pairs_hook=_strict_ai_json_object)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "action",
        "confidence",
        "reason_codes",
        "summary",
    }:
        return None
    raw_action = parsed.get("action")
    raw_confidence = parsed.get("confidence")
    raw_codes = parsed.get("reason_codes")
    raw_summary = parsed.get("summary")
    if (
        not isinstance(raw_action, str)
        or isinstance(raw_confidence, bool)
        or not isinstance(raw_confidence, (int, float))
        or not isinstance(raw_codes, list)
        or any(not isinstance(value, str) for value in raw_codes)
        or not isinstance(raw_summary, str)
        or not 1 <= len(raw_codes) <= 4
    ):
        return None
    action = raw_action
    confidence = float(raw_confidence)
    codes = tuple(dict.fromkeys(raw_codes))
    summary = raw_summary.strip()
    if (
        action not in {"approve", "veto", "cooldown"}
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
        or len(codes) != len(raw_codes)
        or any(code not in _AI_REASON_CODES for code in codes)
        or not _ai_action_reasons_coherent(action, codes)
        or not summary
        or len(summary) > 180
    ):
        return None
    return {
        "action": action,
        "confidence": confidence,
        "reason_codes": list(codes),
        "summary": summary,
        "valid": True,
        "failure_reason": "",
        "permits_entry": action == "approve",
    }


def _validate_ai_evidence(
    ai: Mapping[str, Any],
    *,
    predictions: Sequence[Mapping[str, Any]],
    probability: Mapping[str, Any],
    model_execution: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if ai.get("enabled") is False:
        _verify_claims(ai, name="AI-disabled evidence")
        return None
    if ai.get("enabled") is not True:
        raise ValueError("AI evidence enabled state is invalid")
    _require_exact_keys(
        ai,
        {
            "enabled",
            "risk_benchmark",
            "policy_selection",
            "prompt_cases",
            "veto_report",
            "execution",
            "uplift",
        },
        name="AI evidence",
    )
    selection = _as_mapping(ai["policy_selection"], "AI policy selection")
    _require_exact_keys(
        selection,
        {
            "evaluated_market_count",
            "candidate_count",
            "candidates",
            "reason_counts",
            "selection_sha256",
        },
        name="AI policy selection",
    )
    conditions = {str(row["condition_id"]) for row in predictions}
    prediction_by_sample = {str(row["sample_id"]): row for row in predictions}
    if (
        int(selection.get("evaluated_market_count", -1)) != len(conditions)
        or not _is_sha256(selection.get("selection_sha256"))
        or not isinstance(selection.get("candidates"), list)
    ):
        raise ValueError("AI policy selection identity is invalid")
    candidates = [
        _as_mapping(item, "AI policy candidate") for item in selection["candidates"]
    ]
    if int(selection.get("candidate_count", -1)) != len(candidates):
        raise ValueError("AI policy candidate count is inconsistent")
    reason_counts = _as_mapping(selection.get("reason_counts"), "AI selection reasons")
    if any(
        not str(name)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in reason_counts.items()
    ) or sum(reason_counts.values()) + len(candidates) != len(conditions):
        raise ValueError("AI policy selection coverage is inconsistent")
    candidate_by_sample: dict[str, Mapping[str, Any]] = {}
    candidate_conditions: set[str] = set()
    for candidate in candidates:
        _require_exact_keys(
            candidate,
            {
                "sample_id",
                "condition_id",
                "asset",
                "event_start_ms",
                "decision_received_wall_ms",
                "outcome",
                "predicted_probability",
                "expected_edge_per_contract",
                "decision_best_ask",
                "limit_price",
            },
            name="AI policy candidate",
        )
        sample_id = str(candidate["sample_id"])
        condition_id = str(candidate["condition_id"])
        prediction = prediction_by_sample.get(sample_id)
        outcome = str(candidate["outcome"])
        if (
            prediction is None
            or condition_id != str(prediction["condition_id"])
            or condition_id in candidate_conditions
            or sample_id in candidate_by_sample
            or candidate["asset"] != prediction["asset"]
            or int(candidate["event_start_ms"]) != int(prediction["event_start_ms"])
            or int(candidate["decision_received_wall_ms"])
            != int(prediction["decision_received_wall_ms"])
            or outcome not in {"Up", "Down"}
        ):
            raise ValueError("AI policy candidate does not bind held-out evidence")
        model_up = _finite_float(
            prediction["model_up_probability"],
            "held-out model probability",
        )
        expected_probability = model_up if outcome == "Up" else 1.0 - model_up
        if str(candidate["predicted_probability"]) != format(
            expected_probability,
            ".17g",
        ):
            raise ValueError("AI policy candidate probability is inconsistent")
        best_ask = _decimal(candidate["decision_best_ask"], "candidate best ask")
        limit = _decimal(candidate["limit_price"], "candidate limit")
        edge = _decimal(candidate["expected_edge_per_contract"], "candidate edge")
        if not (
            Decimal("0") < best_ask < Decimal("1")
            and best_ask <= limit < 1
            and edge >= 0
        ):
            raise ValueError("AI policy candidate economics are invalid")
        candidate_by_sample[sample_id] = candidate
        candidate_conditions.add(condition_id)
    expected_selection_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-policy-selection-v1",
            "config": model_execution["config"],
            "sample_ids": [str(row["sample_id"]) for row in predictions],
            "probabilities": [
                format(
                    _finite_float(
                        row["model_up_probability"],
                        "held-out model probability",
                    ),
                    ".17g",
                )
                for row in predictions
            ],
            "candidates": [dict(candidate) for candidate in candidates],
            "reason_counts": dict(sorted(reason_counts.items())),
        }
    )
    if selection.get("selection_sha256") != expected_selection_sha256:
        raise ValueError("AI policy selection does not reconstruct")
    raw_cases = ai.get("prompt_cases")
    if not isinstance(raw_cases, list):
        raise ValueError("AI prompt cases must be an array")
    cases = [_as_mapping(item, "AI prompt case") for item in raw_cases]
    if len(cases) != len(candidates):
        raise ValueError("AI prompt case count does not match frozen proposals")
    validation_baseline = _as_mapping(
        _as_mapping(probability["baseline_metrics"], "baseline metrics")["validation"],
        "baseline validation metrics",
    )
    validation_model = _as_mapping(
        _as_mapping(probability["model_metrics"], "model metrics")["validation"],
        "model validation metrics",
    )
    execution_config = _as_mapping(
        model_execution["config"],
        "model execution config",
    )
    case_ids: set[str] = set()
    case_sha256_values: list[str] = []
    for case in cases:
        _require_exact_keys(
            case,
            {
                "schema_version",
                "case_id",
                "condition_id",
                "sample_id",
                "asset",
                "event_start_ms",
                "decision_received_wall_ms",
                "decision_received_monotonic_ns",
                "prompt_payload",
                "case_sha256",
            },
            name="AI prompt case",
        )
        identity = dict(case)
        case_sha256 = str(identity.pop("case_sha256"))
        prompt = _as_mapping(case["prompt_payload"], "AI prompt")
        _validate_ai_prompt_shape(prompt)
        sample_id = str(case["sample_id"])
        candidate = candidate_by_sample.get(sample_id)
        prediction = prediction_by_sample.get(sample_id)
        case_id = str(case["case_id"])
        if (
            case.get("schema_version") != _AI_CASE_SCHEMA
            or candidate is None
            or prediction is None
            or case_id in case_ids
            or case_sha256 != _canonical_sha256(identity)
            or str(case["condition_id"]) != str(candidate["condition_id"])
            or case["asset"] != candidate["asset"]
            or int(case["event_start_ms"]) != int(candidate["event_start_ms"])
            or int(case["decision_received_wall_ms"])
            != int(candidate["decision_received_wall_ms"])
            or int(case["decision_received_monotonic_ns"])
            != int(prediction["decision_received_monotonic_ns"])
        ):
            raise ValueError("AI prompt case identity is invalid")
        expected_case_id = _canonical_sha256(
            {
                "selection_sha256": selection["selection_sha256"],
                "model_report_sha256": probability["report_sha256"],
                "sample_id": sample_id,
                "prompt_payload": prompt,
            }
        )
        outcome = str(candidate["outcome"])
        baseline_up = _finite_float(
            prediction["baseline_up_probability"],
            "held-out market probability",
        )
        outcome_prior = baseline_up if outcome == "Up" else 1.0 - baseline_up
        candidate_probability = _finite_float(
            candidate["predicted_probability"],
            "candidate probability",
        )
        expected_prompt_values = {
            "asset": candidate["asset"],
            "remaining_seconds": int(prediction["horizon_seconds"]),
            "proposed_outcome": outcome,
            "model_probability": round(candidate_probability, 8),
            "market_implied_probability": round(outcome_prior, 8),
            "model_probability_uplift": round(
                candidate_probability - outcome_prior,
                8,
            ),
            "decision_best_ask": str(candidate["decision_best_ask"]),
            "protective_limit_price": str(candidate["limit_price"]),
            "expected_edge_per_contract_after_fee": str(
                candidate["expected_edge_per_contract"]
            ),
            "minimum_required_edge_per_contract": str(
                execution_config["minimum_expected_edge_per_contract"]
            ),
            "maximum_loss_fraction_per_market": str(
                execution_config["maximum_loss_fraction_per_market"]
            ),
            "maximum_loss_fraction_per_time_group": str(
                execution_config["maximum_loss_fraction_per_time_group"]
            ),
            "assumed_submission_latency_ms": int(
                execution_config["submission_latency_ms"]
            ),
        }
        if case_id != expected_case_id or any(
            prompt.get(name) != value for name, value in expected_prompt_values.items()
        ):
            raise ValueError("AI prompt does not reconstruct from frozen evidence")
        feature_map = {
            str(name): _finite_float(value, f"AI source feature {name}")
            for name, value in zip(
                prediction["feature_names"],
                prediction["feature_values"],
                strict=True,
            )
        }
        risk_map = {
            str(name): _finite_float(value, f"AI source risk context {name}")
            for name, value in zip(
                prediction["risk_context_names"],
                prediction["risk_context_values"],
                strict=True,
            )
        }
        expected_microstructure = {
            name: round(feature_map[name], 8) for name in _AI_MICROSTRUCTURE_FIELDS
        }
        expected_freshness = {
            name: round(risk_map[name], 3) for name in _AI_FRESHNESS_FIELDS
        }
        expected_liquidity = {
            "proposed_outcome_ask_depth_3_contracts": round(
                risk_map[
                    "up_ask_depth_3_contracts"
                    if outcome == "Up"
                    else "down_ask_depth_3_contracts"
                ],
                8,
            ),
            "market_liquidity_quote": round(
                math.expm1(risk_map["log1p_market_liquidity_quote"]),
                2,
            ),
            "market_volume_quote": round(
                math.expm1(risk_map["log1p_market_volume_quote"]),
                2,
            ),
        }
        if (
            dict(prompt["microstructure"]) != expected_microstructure
            or dict(prompt["source_freshness_ms"]) != expected_freshness
            or dict(prompt["liquidity_context"]) != expected_liquidity
        ):
            raise ValueError("AI prompt does not match its causal model sample")
        expected_validation = {
            "market_baseline_log_loss": round(
                _finite_float(
                    validation_baseline["weighted_log_loss"],
                    "baseline validation log loss",
                ),
                10,
            ),
            "residual_model_log_loss": round(
                _finite_float(
                    validation_model["weighted_log_loss"],
                    "model validation log loss",
                ),
                10,
            ),
            "log_loss_delta": round(
                _finite_float(
                    probability["validation_log_loss_delta"],
                    "validation log-loss delta",
                ),
                10,
            ),
            "validation_market_count": int(validation_model["market_count"]),
        }
        if dict(prompt["validation_only_model_evidence"]) != expected_validation:
            raise ValueError("AI prompt contains non-frozen validation evidence")
        case_ids.add(case_id)
        case_sha256_values.append(case_sha256)
    expected_case_order = sorted(
        cases,
        key=lambda item: (
            int(item["decision_received_monotonic_ns"]),
            int(item["decision_received_wall_ms"]),
            str(item["asset"]),
            str(item["condition_id"]),
        ),
    )
    if cases != expected_case_order:
        raise ValueError("AI prompt cases are not chronological")

    veto = _as_mapping(ai["veto_report"], "AI veto report")
    _verify_claims(veto, name="AI veto report")
    _verify_embedded_digest(veto, "report_sha256", name="AI veto report")
    config = _as_mapping(veto.get("config"), "AI veto config")
    benchmark = _as_mapping(ai["risk_benchmark"], "AI risk benchmark")
    model_provenance = _as_mapping(
        benchmark.get("model_provenance"),
        "AI model provenance",
    )
    raw_model_size = model_provenance.get("size_bytes")
    if isinstance(raw_model_size, bool):
        raise ValueError("AI model provenance size is invalid")
    try:
        model_size = int(raw_model_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("AI model provenance size is invalid") from exc
    _require_exact_keys(
        config,
        {
            "model",
            "base_url",
            "timeout_seconds",
            "minimum_approval_confidence",
            "maximum_advisory_latency_seconds",
            "seed",
        },
        name="AI veto config",
    )
    model_name = str(config.get("model", ""))
    base_url = str(config.get("base_url", "")).rstrip("/")
    timeout_seconds = _finite_float(config.get("timeout_seconds"), "AI timeout")
    minimum_confidence = _finite_float(
        config.get("minimum_approval_confidence"),
        "AI approval confidence floor",
    )
    maximum_latency = _finite_float(
        config.get("maximum_advisory_latency_seconds"),
        "AI advisory latency ceiling",
    )
    expected_case_set_sha256 = _canonical_sha256(
        {
            "schema_version": _AI_CASE_SCHEMA,
            "selection_sha256": selection["selection_sha256"],
            "case_sha256": case_sha256_values,
        }
    )
    if (
        veto.get("schema_version") != _AI_REPORT_SCHEMA
        or veto.get("advisory_only") is not True
        or veto.get("selection_sha256") != selection["selection_sha256"]
        or veto.get("case_set_sha256") != expected_case_set_sha256
        or int(veto.get("case_count", -1)) != len(cases)
        or not model_name
        or not (
            base_url.startswith("http://127.0.0.1:")
            or base_url.startswith("http://localhost:")
        )
        or not 1.0 <= timeout_seconds <= 300.0
        or not 0.5 <= minimum_confidence <= 1.0
        or not 0.1 <= maximum_latency <= 60.0
        or isinstance(config.get("seed"), bool)
        or int(config.get("seed", -1)) < 0
        or _finite_float(veto.get("model_parameters_b"), "AI model size") < 2.0
        or not _is_sha256(veto.get("model_digest"))
        or not _is_sha256(veto.get("model_metadata_sha256"))
        or not _is_sha256(veto.get("risk_benchmark_evidence_sha256"))
        or veto.get("risk_benchmark_evidence_sha256") != benchmark.get("sha256")
        or benchmark.get("selected_model") != model_name
        or benchmark.get("contract") != AI_MODEL_BENCHMARK_CONTRACT
        or not str(benchmark.get("path", ""))
        or model_provenance.get("model") != model_name
        or model_provenance.get("benchmark_sha256") != benchmark.get("sha256")
        or model_provenance.get("benchmark_contract") != benchmark.get("contract")
        or model_provenance.get("ollama_manifest_digest") != veto.get("model_digest")
        or not _is_sha256(model_provenance.get("provenance_sha256"))
        or not _is_sha256(model_provenance.get("base_blob_sha256"))
        or model_size <= 2_000_000_000
        or not str(model_provenance.get("path", ""))
    ):
        raise ValueError("AI veto report provenance is inconsistent")
    _finite_float(benchmark.get("score"), "AI risk benchmark score")
    results_value = veto.get("results")
    if not isinstance(results_value, list) or len(results_value) != len(cases):
        raise ValueError("AI veto result count is inconsistent")
    results = [_as_mapping(item, "AI veto result") for item in results_value]
    expected_permissions = {condition: False for condition in conditions}
    inference_latencies: list[float] = []
    queue_delays: list[float] = []
    latencies: list[float] = []
    worker_available_ns: int | None = None
    valid_count = approval_count = veto_count = cooldown_count = failure_count = 0
    for case, result in zip(cases, results, strict=True):
        _require_exact_keys(
            result,
            {
                "case_id",
                "condition_id",
                "model",
                "inference_latency_seconds",
                "queue_delay_seconds",
                "latency_seconds",
                "response_sha256",
                "response_payload",
                "provider_runtime",
                "decision",
            },
            name="AI veto result",
        )
        decision = _as_mapping(result["decision"], "AI veto decision")
        _require_exact_keys(
            decision,
            {
                "action",
                "confidence",
                "reason_codes",
                "summary",
                "valid",
                "failure_reason",
                "permits_entry",
            },
            name="AI veto decision",
        )
        inference_latency = _finite_float(
            result["inference_latency_seconds"],
            "AI veto inference latency",
        )
        queue_delay = _finite_float(
            result["queue_delay_seconds"],
            "AI veto queue delay",
        )
        latency = _finite_float(result["latency_seconds"], "AI veto latency")
        arrival_ns = int(case["decision_received_monotonic_ns"])
        inference_duration_ns = max(
            0,
            math.ceil(inference_latency * 1_000_000_000),
        )
        service_start_ns = max(arrival_ns, worker_available_ns or arrival_ns)
        expected_queue_delay_ns = service_start_ns - arrival_ns
        worker_available_ns = service_start_ns + inference_duration_ns
        expected_queue_delay = expected_queue_delay_ns / 1_000_000_000.0
        expected_latency = (
            expected_queue_delay_ns + inference_duration_ns
        ) / 1_000_000_000.0
        action = str(decision.get("action", ""))
        confidence = _finite_float(decision.get("confidence"), "AI confidence")
        reason_codes = decision.get("reason_codes")
        valid = decision.get("valid")
        permits = decision.get("permits_entry")
        failure_reason = str(decision.get("failure_reason", ""))
        parsed_response = _parsed_valid_ai_response(result.get("response_payload"))
        runtime_value = result.get("provider_runtime")
        runtime = None
        if runtime_value is not None:
            try:
                runtime = ollama_residency_from_mapping(runtime_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("AI veto provider runtime is invalid") from exc
            if runtime.requested_model != model_name or (
                runtime.loaded and runtime.digest != veto.get("model_digest")
            ):
                raise ValueError("AI veto provider runtime provenance differs")
        gpu_runtime_proved = bool(
            runtime is not None
            and runtime.loaded
            and runtime.gpu_resident
            and runtime.digest == veto.get("model_digest")
        )
        valid_response_was_overridden = latency > maximum_latency or (
            parsed_response is not None
            and parsed_response["action"] == "approve"
            and float(parsed_response["confidence"]) < minimum_confidence
        )
        if (
            result.get("case_id") != case["case_id"]
            or result.get("condition_id") != case["condition_id"]
            or result.get("model") != model_name
            or isinstance(result.get("inference_latency_seconds"), bool)
            or isinstance(result.get("queue_delay_seconds"), bool)
            or isinstance(result.get("latency_seconds"), bool)
            or inference_latency < 0.0
            or queue_delay < 0.0
            or latency < 0.0
            or not math.isclose(
                queue_delay,
                expected_queue_delay,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                latency,
                expected_latency,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not _is_sha256(result.get("response_sha256"))
            or result.get("response_sha256")
            != _canonical_sha256(result.get("response_payload"))
            or action not in {"approve", "veto", "cooldown"}
            or not 0.0 <= confidence <= 1.0
            or not isinstance(reason_codes, list)
            or not 1 <= len(reason_codes) <= 4
            or any(not isinstance(code, str) for code in reason_codes)
            or len(set(reason_codes)) != len(reason_codes)
            or any(code not in _AI_REASON_CODES for code in reason_codes)
            or (valid and not _ai_action_reasons_coherent(action, reason_codes))
            or not isinstance(valid, bool)
            or not isinstance(permits, bool)
            or permits is not (valid and action == "approve")
            or not isinstance(decision.get("summary"), str)
            or len(str(decision["summary"])) > 180
            or (valid and dict(decision) != parsed_response)
            or (valid and not gpu_runtime_proved)
            or (
                not valid
                and parsed_response is not None
                and not valid_response_was_overridden
            )
            or (valid and failure_reason)
            or (valid and latency > maximum_latency)
            or (valid and action == "approve" and confidence < minimum_confidence)
            or (
                not valid
                and (action != "veto" or confidence != 0.0 or not failure_reason)
            )
        ):
            raise ValueError("AI veto result is malformed or not fail-closed")
        expected_permissions[str(case["condition_id"])] = permits
        inference_latencies.append(inference_latency)
        queue_delays.append(queue_delay)
        latencies.append(latency)
        valid_count += int(valid)
        approval_count += int(action == "approve")
        veto_count += int(action == "veto")
        cooldown_count += int(action == "cooldown")
        failure_count += int(not valid)
    permissions = _as_mapping(veto.get("market_permissions"), "AI permissions")
    expected_permission_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-market-permission-v1",
            "permissions": dict(sorted(expected_permissions.items())),
        }
    )
    reported_average = _finite_float(
        veto.get("average_latency_seconds"),
        "AI average latency",
    )
    reported_maximum = _finite_float(
        veto.get("maximum_latency_seconds"),
        "AI maximum latency",
    )
    reported_average_inference = _finite_float(
        veto.get("average_inference_latency_seconds"),
        "AI average inference latency",
    )
    reported_maximum_inference = _finite_float(
        veto.get("maximum_inference_latency_seconds"),
        "AI maximum inference latency",
    )
    reported_average_queue = _finite_float(
        veto.get("average_queue_delay_seconds"),
        "AI average queue delay",
    )
    reported_maximum_queue = _finite_float(
        veto.get("maximum_queue_delay_seconds"),
        "AI maximum queue delay",
    )
    expected_average_inference = (
        sum(inference_latencies) / len(inference_latencies)
        if inference_latencies
        else 0.0
    )
    expected_maximum_inference = max(inference_latencies, default=0.0)
    expected_average_queue = (
        sum(queue_delays) / len(queue_delays) if queue_delays else 0.0
    )
    expected_maximum_queue = max(queue_delays, default=0.0)
    expected_average = sum(latencies) / len(latencies) if latencies else 0.0
    expected_maximum = max(latencies, default=0.0)
    if (
        dict(permissions) != dict(sorted(expected_permissions.items()))
        or veto.get("market_permission_sha256") != expected_permission_sha256
        or int(veto.get("valid_response_count", -1)) != valid_count
        or int(veto.get("approval_count", -1)) != approval_count
        or int(veto.get("veto_count", -1)) != veto_count
        or int(veto.get("cooldown_count", -1)) != cooldown_count
        or int(veto.get("provider_failure_count", -1)) != failure_count
        or not math.isclose(
            reported_average_inference,
            expected_average_inference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reported_maximum_inference,
            expected_maximum_inference,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reported_average_queue,
            expected_average_queue,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reported_maximum_queue,
            expected_maximum_queue,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reported_average, expected_average, rel_tol=1e-12, abs_tol=1e-12
        )
        or not math.isclose(
            reported_maximum, expected_maximum, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ValueError("AI veto aggregate evidence does not reconcile")

    ai_execution = _as_mapping(ai["execution"], "AI execution")
    expected_delays = {condition: 0 for condition in conditions}
    for result in results:
        expected_delays[str(result["condition_id"])] = int(
            math.ceil(max(0.0, float(result["latency_seconds"])) * 1_000.0)
        )
    expected_delay_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-decision-delay-input-v1",
            "decision_delay_ms_by_condition": dict(sorted(expected_delays.items())),
        }
    )
    expected_probability_input_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-probability-input-v1",
            "sample_ids": [str(row["sample_id"]) for row in predictions],
            "probabilities": [
                format(
                    _finite_float(
                        row["model_up_probability"],
                        "held-out model probability",
                    ),
                    ".17g",
                )
                for row in predictions
            ],
            "market_permission_sha256": expected_permission_sha256,
            "decision_delay_input_sha256": expected_delay_sha256,
        }
    )
    if (
        ai_execution.get("market_permissions")
        != dict(sorted(expected_permissions.items()))
        or ai_execution.get("market_permission_sha256") != expected_permission_sha256
        or ai_execution.get("decision_delay_ms_by_condition")
        != dict(sorted(expected_delays.items()))
        or ai_execution.get("decision_delay_input_sha256") != expected_delay_sha256
        or ai_execution.get("probability_input_sha256")
        != expected_probability_input_sha256
        or ai_execution.get("config") != model_execution.get("config")
    ):
        raise ValueError("AI execution is not bound to veto-only decisions")
    uplift = _as_mapping(ai["uplift"], "AI uplift")
    _verify_claims(uplift, name="AI uplift")
    dataset_fingerprint = str(probability.get("source_dataset_sha256", ""))
    expected_uplift = assess_ai_uplift(
        _execution_uplift_metrics(
            model_execution,
            dataset_fingerprint=dataset_fingerprint,
        ),
        _execution_uplift_metrics(
            ai_execution,
            dataset_fingerprint=dataset_fingerprint,
        ),
        model_name=model_name,
        model_parameters_b=_finite_float(
            veto.get("model_parameters_b"),
            "AI veto model size",
        ),
        model_artifact_sha256=str(veto["report_sha256"]),
        matched_periods=_matched_ai_uplift_periods(
            predictions,
            model_execution,
            ai_execution,
        ),
    ).asdict()
    if _canonical_json(uplift) != _canonical_json(expected_uplift):
        raise ValueError("AI uplift evidence is not model-bound")
    return ai_execution


def validate_polymarket_model_artifact(path: str | Path) -> ValidatedPolymarketArtifact:
    """Fail closed unless the experiment and every publication input are coherent."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        payload = _as_mapping(json.loads(raw.decode("utf-8")), "model artifact")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Polymarket model artifact is not valid UTF-8 JSON") from exc
    if payload.get("schema_version") != POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported Polymarket model artifact schema")
    canonical = dict(payload)
    claimed_artifact_sha256 = str(canonical.pop("artifact_sha256", ""))
    if len(
        claimed_artifact_sha256
    ) != 64 or claimed_artifact_sha256 != _canonical_sha256(canonical):
        raise ValueError("Polymarket model artifact identity is invalid")
    _verify_claims(payload, name="model artifact")

    model = _as_mapping(payload.get("model"), "model")
    probability = _as_mapping(payload.get("probability_report"), "probability report")
    profile_model = _as_mapping(payload.get("profile_model"), "profile model")
    profile_probability = _as_mapping(
        payload.get("profile_probability_report"),
        "profile probability report",
    )
    split = _as_mapping(payload.get("split"), "split")
    model_dataset = _as_mapping(payload.get("model_dataset"), "model dataset")
    feature_dataset = _as_mapping(payload.get("feature_dataset"), "feature dataset")
    for name, value, digest_key in (
        ("model", model, "model_sha256"),
        ("probability report", probability, "report_sha256"),
    ):
        _verify_claims(value, name=name)
        _verify_embedded_digest(value, digest_key, name=name)
    if not (
        model.get("source_dataset_sha256") == model_dataset.get("dataset_sha256")
        and model.get("source_split_sha256") == split.get("split_sha256")
        and probability.get("source_dataset_sha256")
        == model_dataset.get("dataset_sha256")
        and probability.get("source_split_sha256") == split.get("split_sha256")
        and profile_model.get("source_dataset_sha256")
        == model_dataset.get("dataset_sha256")
        and profile_model.get("source_split_sha256") == split.get("split_sha256")
        and profile_probability.get("source_dataset_sha256")
        == model_dataset.get("dataset_sha256")
        and profile_probability.get("source_split_sha256") == split.get("split_sha256")
        and model_dataset.get("source_dataset_sha256")
        == feature_dataset.get("dataset_sha256")
        and payload.get("run_id") == feature_dataset.get("run_id")
    ):
        raise ValueError("Polymarket model provenance chain is inconsistent")
    _validate_nested_model_selection(model, probability, split)
    _validate_profile_model_selection(
        profile_model,
        profile_probability,
        model,
        probability,
    )
    if model_dataset.get("training_ready") is not True:
        raise ValueError("Polymarket model dataset was not training-ready")

    evidence = _as_mapping(
        payload.get("held_out_prediction_evidence"),
        "held-out prediction evidence",
    )
    if (
        evidence.get("schema_version") != _PREDICTION_SCHEMA
        or evidence.get("role") != "untouched_chronological_test"
    ):
        raise ValueError("held-out prediction evidence contract is invalid")
    predictions = _as_rows(evidence.get("rows"), "held-out predictions")
    if (
        evidence.get("row_count") != len(predictions)
        or evidence.get("rows_sha256") != _canonical_sha256(predictions)
        or tuple(evidence.get("assets", ())) != _ASSETS
    ):
        raise ValueError("held-out prediction evidence identity is invalid")
    sample_ids: set[str] = set()
    conditions: set[str] = set()
    time_groups: set[int] = set()
    condition_labels: dict[str, bool] = {}
    condition_assets: dict[str, str] = {}
    expected_feature_names = tuple(model_dataset.get("model_feature_names", ()))
    expected_risk_names = tuple(model_dataset.get("risk_context_names", ()))
    if (
        not expected_feature_names
        or expected_feature_names != tuple(model.get("feature_names", ()))
        or not expected_risk_names
    ):
        raise ValueError("held-out sample feature contracts are inconsistent")
    for row in predictions:
        _require_exact_keys(
            row,
            {
                "schema_version",
                "sample_id",
                "source_run_id",
                "source_feature_id",
                "condition_id",
                "market_id",
                "asset",
                "event_start_ms",
                "end_ms",
                "decision_received_wall_ms",
                "decision_received_monotonic_ns",
                "decision_event_id",
                "horizon_seconds",
                "feature_names",
                "feature_values",
                "risk_context_names",
                "risk_context_values",
                "baseline_up_probability",
                "up_best_bid",
                "up_best_ask",
                "down_best_bid",
                "down_best_ask",
                "official_up",
                "resolution_event_id",
                "market_weight",
                "input_provenance_sha256",
                "sample_sha256",
                "model_up_probability",
            },
            name="held-out prediction row",
        )
        sample_id = str(row.get("sample_id", ""))
        condition_id = str(row.get("condition_id", ""))
        asset = str(row.get("asset", ""))
        event_start = int(row.get("event_start_ms", -1))
        end_ms = int(row.get("end_ms", -1))
        decision_ms = int(row.get("decision_received_wall_ms", -1))
        horizon = int(row.get("horizon_seconds", -1))
        label = row.get("official_up")
        feature_names = tuple(row.get("feature_names", ()))
        risk_names = tuple(row.get("risk_context_names", ()))
        feature_values_raw = row.get("feature_values")
        risk_values_raw = row.get("risk_context_values")
        if not isinstance(feature_values_raw, list) or not isinstance(
            risk_values_raw,
            list,
        ):
            raise ValueError("held-out model sample vectors are malformed")
        feature_values = tuple(
            _finite_float(value, "held-out model feature")
            for value in feature_values_raw
        )
        risk_values = tuple(
            _finite_float(value, "held-out risk context") for value in risk_values_raw
        )
        sample_identity = dict(row)
        sample_identity.pop("model_up_probability")
        claimed_sample_sha256 = str(sample_identity.pop("sample_sha256"))
        expected_sample_id = _canonical_sha256(
            {
                "source_dataset_sha256": model_dataset["source_dataset_sha256"],
                "source_feature_id": row["source_feature_id"],
                "horizon_seconds": horizon,
                "config": model_dataset["config"],
            }
        )
        if (
            row.get("schema_version") != _MODEL_SAMPLE_SCHEMA
            or len(sample_id) != 64
            or sample_id in sample_ids
            or sample_id != expected_sample_id
            or not _is_sha256(claimed_sample_sha256)
            or claimed_sample_sha256 != _canonical_sha256(sample_identity)
            or row.get("source_run_id") != payload.get("run_id")
            or not _is_sha256(row.get("source_feature_id"))
            or not str(row.get("decision_event_id", ""))
            or not str(row.get("resolution_event_id", ""))
            or int(row.get("decision_received_monotonic_ns", -1)) < 0
            or not condition_id
            or not str(row.get("market_id", ""))
            or asset not in _ASSETS
            or end_ms != event_start + 300_000
            or decision_ms != end_ms - horizon * 1_000
            or horizon not in {30, 60, 120, 180, 240}
            or not isinstance(label, bool)
            or not _is_sha256(row.get("input_provenance_sha256"))
            or feature_names != expected_feature_names
            or risk_names != expected_risk_names
            or len(feature_values) != len(expected_feature_names)
            or len(risk_values) != len(expected_risk_names)
            or any(value < 0.0 for value in risk_values)
            or any(
                str(raw) != format(value, ".17g")
                for raw, value in zip(
                    feature_values_raw,
                    feature_values,
                    strict=True,
                )
            )
            or any(
                str(raw) != format(value, ".17g")
                for raw, value in zip(
                    risk_values_raw,
                    risk_values,
                    strict=True,
                )
            )
        ):
            raise ValueError("held-out prediction row is malformed")
        baseline = _finite_float(
            row.get("baseline_up_probability"), "baseline probability"
        )
        model_probability = _finite_float(
            row.get("model_up_probability"), "model probability"
        )
        weight = _finite_float(row.get("market_weight"), "market weight")
        up_bid = _finite_float(row.get("up_best_bid"), "held-out Up bid")
        up_ask = _finite_float(row.get("up_best_ask"), "held-out Up ask")
        down_bid = _finite_float(row.get("down_best_bid"), "held-out Down bid")
        down_ask = _finite_float(row.get("down_best_ask"), "held-out Down ask")
        up_midpoint = (up_bid + up_ask) / 2.0
        down_midpoint = (down_bid + down_ask) / 2.0
        reconstructed_baseline = up_midpoint / (up_midpoint + down_midpoint)
        if not (
            0.0 < baseline < 1.0
            and 0.0 < model_probability < 1.0
            and weight > 0.0
            and 0.0 < up_bid < up_ask < 1.0
            and 0.0 < down_bid < down_ask < 1.0
            and math.isclose(
                baseline,
                reconstructed_baseline,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "held-out probability or market weight is outside its domain"
            )
        if (
            condition_id in condition_labels
            and condition_labels[condition_id] is not label
        ):
            raise ValueError("held-out market has inconsistent official labels")
        if condition_id in condition_assets and condition_assets[condition_id] != asset:
            raise ValueError("held-out market has inconsistent assets")
        sample_ids.add(sample_id)
        conditions.add(condition_id)
        time_groups.add(event_start)
        condition_labels[condition_id] = label
        condition_assets[condition_id] = asset
    expected_groups = tuple(
        int(value) for value in split.get("test_group_starts_ms", ())
    )
    baseline_roles = _as_mapping(
        probability.get("baseline_metrics"),
        "baseline metrics",
    )
    model_roles = _as_mapping(
        probability.get("model_metrics"),
        "model metrics",
    )
    baseline_test_metrics = _as_mapping(
        baseline_roles.get("test"),
        "baseline test metrics",
    )
    model_test_metrics = _as_mapping(
        model_roles.get("test"),
        "model test metrics",
    )
    if (
        set(expected_groups) != time_groups
        or evidence.get("market_count") != len(conditions)
        or evidence.get("time_group_count") != len(time_groups)
        or int(
            _as_mapping(split.get("sample_counts"), "split sample counts").get(
                "test",
                -1,
            )
        )
        != len(predictions)
    ):
        raise ValueError("held-out predictions do not match the frozen split")
    expected_baseline_test = _probability_metrics_from_rows(
        predictions,
        "baseline_up_probability",
    )
    expected_model_test = _probability_metrics_from_rows(
        predictions,
        "model_up_probability",
    )
    _validate_probability_metrics(
        baseline_test_metrics,
        expected_baseline_test,
        name="baseline test metrics",
    )
    _validate_probability_metrics(
        model_test_metrics,
        expected_model_test,
        name="model test metrics",
    )

    profile_evidence = _as_mapping(
        payload.get("profile_held_out_prediction_evidence"),
        "profile held-out prediction evidence",
    )
    _verify_claims(profile_evidence, name="profile held-out prediction evidence")
    _require_exact_keys(
        profile_evidence,
        {
            "schema_version",
            "role",
            "control_rows_sha256",
            "row_count",
            "rows_sha256",
            "rows",
            "trading_authority",
            "execution_claim",
            "profitability_claim",
        },
        name="profile held-out prediction evidence",
    )
    profile_rows = _as_rows(
        profile_evidence.get("rows"),
        "profile held-out prediction rows",
    )
    if (
        profile_evidence.get("schema_version") != _PROFILE_PREDICTION_SCHEMA
        or profile_evidence.get("role") != "untouched_chronological_test"
        or profile_evidence.get("control_rows_sha256") != evidence.get("rows_sha256")
        or int(profile_evidence.get("row_count", -1)) != len(predictions)
        or len(profile_rows) != len(predictions)
        or profile_evidence.get("rows_sha256") != _canonical_sha256(profile_rows)
    ):
        raise ValueError("profile held-out prediction evidence is invalid")
    profile_predictions: list[Mapping[str, Any]] = []
    for control_row, profile_row in zip(predictions, profile_rows, strict=True):
        _require_exact_keys(
            profile_row,
            {"sample_id", "profile_model_up_probability"},
            name="profile held-out prediction row",
        )
        profile_value = _finite_float(
            profile_row.get("profile_model_up_probability"),
            "profile model probability",
        )
        if (
            profile_row.get("sample_id") != control_row.get("sample_id")
            or not 0.0 < profile_value < 1.0
            or str(profile_row.get("profile_model_up_probability"))
            != format(profile_value, ".17g")
        ):
            raise ValueError("profile held-out prediction row is malformed")
        profile_predictions.append(
            {
                **dict(control_row),
                "profile_model_up_probability": profile_row[
                    "profile_model_up_probability"
                ],
            }
        )
    profile_test_metrics = _as_mapping(
        _as_mapping(
            profile_probability.get("challenger_metrics"),
            "profile challenger metrics",
        ).get("test"),
        "profile challenger test metrics",
    )
    expected_profile_test = _probability_metrics_from_rows(
        profile_predictions,
        "profile_model_up_probability",
    )
    _validate_probability_metrics(
        profile_test_metrics,
        expected_profile_test,
        name="profile challenger test metrics",
    )
    for reported, expected, name in (
        (
            profile_probability.get("test_log_loss_delta_vs_control"),
            float(expected_profile_test["weighted_log_loss"])
            - float(expected_model_test["weighted_log_loss"]),
            "profile test log-loss delta",
        ),
        (
            profile_probability.get("test_brier_delta_vs_control"),
            float(expected_profile_test["weighted_brier_score"])
            - float(expected_model_test["weighted_brier_score"]),
            "profile test Brier delta",
        ),
    ):
        if not math.isclose(
            _finite_float(reported, name),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} does not reconcile")

    validation_losses = _named_losses(
        model["validation_gate_log_losses"],
        "outer validation gate losses",
    )
    selected_model_validation_loss = (
        validation_losses[0][1]
        if model["selected_candidate"] == "market_baseline"
        else validation_losses[1][1]
    )
    baseline_validation_metrics = _as_mapping(
        baseline_roles.get("validation"),
        "baseline validation metrics",
    )
    model_validation_metrics = _as_mapping(
        model_roles.get("validation"),
        "model validation metrics",
    )
    comparisons = (
        (
            _finite_float(
                baseline_validation_metrics.get("weighted_log_loss"),
                "baseline validation log loss",
            ),
            validation_losses[0][1],
            "baseline validation log loss",
        ),
        (
            _finite_float(
                model_validation_metrics.get("weighted_log_loss"),
                "model validation log loss",
            ),
            selected_model_validation_loss,
            "model validation log loss",
        ),
        (
            _finite_float(
                probability.get("validation_log_loss_delta"),
                "validation log-loss delta",
            ),
            selected_model_validation_loss - validation_losses[0][1],
            "validation log-loss delta",
        ),
        (
            _finite_float(
                probability.get("test_log_loss_delta"),
                "test log-loss delta",
            ),
            float(expected_model_test["weighted_log_loss"])
            - float(expected_baseline_test["weighted_log_loss"]),
            "test log-loss delta",
        ),
        (
            _finite_float(
                probability.get("test_brier_delta"),
                "test Brier delta",
            ),
            float(expected_model_test["weighted_brier_score"])
            - float(expected_baseline_test["weighted_brier_score"]),
            "test Brier delta",
        ),
    )
    for reported_value, expected_value, name in comparisons:
        if not math.isclose(
            reported_value,
            expected_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{name} does not reconcile")
    confirmation = _as_mapping(
        payload.get("confirmatory_evidence_contract"),
        "confirmatory evidence contract",
    )
    _verify_claims(confirmation, name="confirmatory evidence contract")
    if (
        confirmation.get("independent_unit")
        != "shared_btc_eth_sol_five_minute_time_group"
        or int(confirmation.get("minimum_untouched_test_time_groups", -1)) != 30
        or int(confirmation.get("observed_untouched_test_time_groups", -1))
        != len(time_groups)
        or confirmation.get("confirmatory_ready") is not (len(time_groups) >= 30)
    ):
        raise ValueError("confirmatory evidence contract is inconsistent")

    executions: dict[str, Mapping[str, Any]] = {}
    for policy, key in (
        ("baseline", "baseline_execution"),
        ("model", "model_execution"),
        ("profile_model", "profile_model_execution"),
        ("model_retry", "model_retry_execution"),
    ):
        report = _as_mapping(payload.get(key), f"{policy} execution")
        _verify_claims(report, name=f"{policy} execution")
        _verify_embedded_digest(report, "report_sha256", name=f"{policy} execution")
        executions[policy] = report
    ai = _as_mapping(payload.get("ai"), "AI evidence")
    ai_execution = _validate_ai_evidence(
        ai,
        predictions=predictions,
        probability=probability,
        model_execution=executions["model"],
    )
    if ai_execution is not None:
        _verify_claims(ai_execution, name="AI execution")
        _verify_embedded_digest(ai_execution, "report_sha256", name="AI execution")
        executions["ai"] = ai_execution

    for policy, report in executions.items():
        probability_rows = (
            profile_predictions if policy == "profile_model" else predictions
        )
        _validate_execution_report(
            report,
            conditions=conditions,
            expected_time_group_count=len(time_groups),
            name=f"{policy} primary execution",
            expected_schema=(
                POLYMARKET_RETRY_EXECUTION_REPORT_SCHEMA_VERSION
                if policy == "model_retry"
                else POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION
            ),
        )
        _validate_execution_probability_binding(
            report,
            probability_rows,
            probability_key=(
                "baseline_up_probability"
                if policy == "baseline"
                else (
                    "profile_model_up_probability"
                    if policy == "profile_model"
                    else "model_up_probability"
                )
            ),
            name=f"{policy} primary execution",
        )

    sensitivity = _as_mapping(
        payload.get("execution_latency_sensitivity"),
        "execution latency sensitivity",
    )
    _verify_claims(sensitivity, name="execution latency sensitivity")
    latency_values = sensitivity.get("network_latencies_ms")
    if (
        sensitivity.get("schema_version")
        != "polymarket-execution-latency-sensitivity-v1"
        or not isinstance(latency_values, list)
        or not latency_values
        or latency_values != sorted(set(latency_values))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 60_000
            for value in latency_values
        )
        or int(sensitivity.get("primary_network_latency_ms", -1)) not in latency_values
    ):
        raise ValueError("execution latency sensitivity contract is invalid")
    sensitivity_policies = _as_mapping(
        sensitivity.get("policies"),
        "execution latency policies",
    )
    if set(sensitivity_policies) != set(executions):
        raise ValueError("execution latency sensitivity policy set is invalid")
    primary_latency = int(sensitivity["primary_network_latency_ms"])
    for policy, raw_reports in sensitivity_policies.items():
        reports = _as_mapping(raw_reports, f"{policy} latency reports")
        if set(reports) != {str(value) for value in latency_values}:
            raise ValueError(f"{policy} latency scenario set is incomplete")
        for latency in latency_values:
            scenario = _as_mapping(
                reports[str(latency)],
                f"{policy} {latency}ms execution",
            )
            if (
                int(
                    _as_mapping(scenario["config"], "scenario config")[
                        "submission_latency_ms"
                    ]
                )
                != latency
            ):
                raise ValueError(f"{policy} latency scenario config drifted")
            _validate_execution_report(
                scenario,
                conditions=conditions,
                expected_time_group_count=len(time_groups),
                name=f"{policy} {latency}ms execution",
                expected_schema=(
                    POLYMARKET_RETRY_EXECUTION_REPORT_SCHEMA_VERSION
                    if policy == "model_retry"
                    else POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION
                ),
            )
            _validate_execution_probability_binding(
                scenario,
                (profile_predictions if policy == "profile_model" else predictions),
                probability_key=(
                    "baseline_up_probability"
                    if policy == "baseline"
                    else (
                        "profile_model_up_probability"
                        if policy == "profile_model"
                        else "model_up_probability"
                    )
                ),
                name=f"{policy} {latency}ms execution",
            )
            if (
                scenario.get("market_permissions")
                != executions[str(policy)].get("market_permissions")
                or scenario.get("decision_delay_ms_by_condition")
                != executions[str(policy)].get("decision_delay_ms_by_condition")
                or scenario.get("probability_input_sha256")
                != executions[str(policy)].get("probability_input_sha256")
            ):
                raise ValueError(
                    f"{policy} latency scenario changed frozen policy inputs"
                )
            if latency == primary_latency and scenario.get(
                "report_sha256"
            ) != executions[str(policy)].get("report_sha256"):
                raise ValueError(f"{policy} primary latency report does not match")
    all_latency_reports = [
        _as_mapping(report, f"{policy} latency execution")
        for policy_reports in sensitivity_policies.values()
        for policy, report in _as_mapping(
            policy_reports,
            "latency policy reports",
        ).items()
    ]
    model_latency_reports = _as_mapping(
        sensitivity_policies["model"],
        "model latency reports",
    )
    retry_latency_reports = _as_mapping(
        sensitivity_policies["model_retry"],
        "model retry latency reports",
    )
    expected_retry_gates = {
        "probability_model_gates_passed": (
            _finite_float(
                probability["validation_log_loss_delta"],
                "validation log-loss delta",
            )
            < 0.0
            and _finite_float(
                probability["test_log_loss_delta"],
                "test log-loss delta",
            )
            < 0.0
            and _finite_float(
                probability["test_brier_delta"],
                "test Brier delta",
            )
            < 0.0
        ),
        "minimum_confirmatory_test_time_groups_met": len(time_groups) >= 30,
        "positive_after_cost_at_every_latency": all(
            _decimal(
                _as_mapping(
                    retry_latency_reports[str(latency)],
                    "model retry latency report",
                )["net_realized_pnl_quote"],
                "model retry latency PnL",
            )
            > 0
            for latency in latency_values
        ),
        "improved_after_cost_at_every_latency": all(
            _decimal(
                _as_mapping(
                    retry_latency_reports[str(latency)],
                    "model retry latency report",
                )["net_realized_pnl_quote"],
                "model retry latency PnL",
            )
            > _decimal(
                _as_mapping(
                    model_latency_reports[str(latency)],
                    "model latency report",
                )["net_realized_pnl_quote"],
                "model latency PnL",
            )
            for latency in latency_values
        ),
        "return_on_deployed_not_worse_at_every_latency": all(
            _decimal(
                _as_mapping(
                    retry_latency_reports[str(latency)],
                    "model retry latency report",
                )["return_on_deployed_capital"],
                "model retry deployed-capital return",
            )
            >= _decimal(
                _as_mapping(
                    model_latency_reports[str(latency)],
                    "model latency report",
                )["return_on_deployed_capital"],
                "model deployed-capital return",
            )
            for latency in latency_values
        ),
        "maximum_drawdown_not_worse_at_every_latency": all(
            _decimal(
                _as_mapping(
                    retry_latency_reports[str(latency)],
                    "model retry latency report",
                )["maximum_drawdown_fraction"],
                "model retry drawdown",
            )
            <= _decimal(
                _as_mapping(
                    model_latency_reports[str(latency)],
                    "model latency report",
                )["maximum_drawdown_fraction"],
                "model drawdown",
            )
            for latency in latency_values
        ),
        "all_order_outcomes_terminal": all(
            trade.get("execution_state") != "UNKNOWN"
            for report in retry_latency_reports.values()
            for trade in _as_rows(
                _as_mapping(report, "model retry latency report")["trades"],
                "model retry latency trades",
            )
        ),
    }
    retry_challenger = _as_mapping(
        payload.get("retry_challenger"),
        "model retry challenger",
    )
    _require_exact_keys(
        retry_challenger,
        {
            "schema_version",
            "contract_sha256",
            "control_policy",
            "challenger_policy",
            "gates",
            "accepted",
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
        },
        name="model retry challenger",
    )
    _verify_claims(retry_challenger, name="model retry challenger")
    retry_gates = _as_mapping(retry_challenger["gates"], "model retry gates")
    if (
        retry_challenger.get("schema_version")
        != POLYMARKET_RETRY_CHALLENGER_SCHEMA_VERSION
        or retry_challenger.get("contract_sha256") != POLYMARKET_RETRY_CONTRACT_SHA256
        or retry_challenger.get("control_policy") != "model"
        or retry_challenger.get("challenger_policy") != "model_retry"
        or dict(retry_gates) != expected_retry_gates
        or retry_challenger.get("accepted") is not all(expected_retry_gates.values())
    ):
        raise ValueError("model retry challenger does not reconstruct")
    profile_latency_reports = _as_mapping(
        sensitivity_policies["profile_model"],
        "profile model latency reports",
    )
    profile_validation_delta = _finite_float(
        profile_probability["validation_log_loss_delta_vs_control"],
        "profile validation log-loss delta",
    )
    profile_test_log_loss_delta = _finite_float(
        profile_probability["test_log_loss_delta_vs_control"],
        "profile test log-loss delta",
    )
    profile_test_brier_delta = _finite_float(
        profile_probability["test_brier_delta_vs_control"],
        "profile test Brier delta",
    )
    expected_profile_gates = {
        "validation_log_loss_not_worse_than_control": (profile_validation_delta <= 0.0),
        "test_log_loss_strictly_better_than_control": (
            profile_test_log_loss_delta < 0.0
        ),
        "test_brier_score_strictly_better_than_control": (
            profile_test_brier_delta < 0.0
        ),
        "minimum_untouched_test_time_groups_met": len(time_groups) >= 30,
        "positive_after_cost_at_every_latency": all(
            _decimal(
                _as_mapping(
                    profile_latency_reports[str(latency)],
                    "profile latency report",
                )["net_realized_pnl_quote"],
                "profile latency PnL",
            )
            > 0
            for latency in latency_values
        ),
        "improved_after_cost_at_every_latency": all(
            _decimal(
                _as_mapping(
                    profile_latency_reports[str(latency)],
                    "profile latency report",
                )["net_realized_pnl_quote"],
                "profile latency PnL",
            )
            > _decimal(
                _as_mapping(
                    model_latency_reports[str(latency)],
                    "model latency report",
                )["net_realized_pnl_quote"],
                "model latency PnL",
            )
            for latency in latency_values
        ),
        "return_on_deployed_not_worse_at_every_latency": all(
            _decimal(
                _as_mapping(
                    profile_latency_reports[str(latency)],
                    "profile latency report",
                )["return_on_deployed_capital"],
                "profile deployed-capital return",
            )
            >= _decimal(
                _as_mapping(
                    model_latency_reports[str(latency)],
                    "model latency report",
                )["return_on_deployed_capital"],
                "model deployed-capital return",
            )
            for latency in latency_values
        ),
        "maximum_drawdown_not_worse_at_every_latency": all(
            _decimal(
                _as_mapping(
                    profile_latency_reports[str(latency)],
                    "profile latency report",
                )["maximum_drawdown_fraction"],
                "profile drawdown",
            )
            <= _decimal(
                _as_mapping(
                    model_latency_reports[str(latency)],
                    "model latency report",
                )["maximum_drawdown_fraction"],
                "model drawdown",
            )
            for latency in latency_values
        ),
        "all_order_outcomes_terminal": all(
            trade.get("execution_state") != "UNKNOWN"
            for raw_report in profile_latency_reports.values()
            for trade in _as_mapping(
                raw_report,
                "profile latency report",
            )["trades"]
        ),
    }
    profile_challenger = _as_mapping(
        payload.get("profile_challenger"),
        "profile model challenger",
    )
    _require_exact_keys(
        profile_challenger,
        {
            "schema_version",
            "contract_sha256",
            "control_policy",
            "challenger_policy",
            "gates",
            "promotion_gates_passed",
            "accepted",
            "status",
            "requires_later_prospective_confirmation",
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
        },
        name="profile model challenger",
    )
    _verify_claims(profile_challenger, name="profile model challenger")
    profile_promotion_gates_passed = all(expected_profile_gates.values())
    expected_profile_status = (
        "awaiting_later_prospective_confirmation"
        if profile_promotion_gates_passed
        else "exploratory_gates_failed"
    )
    if (
        profile_challenger.get("schema_version")
        != POLYMARKET_PROFILE_CHALLENGER_SCHEMA_VERSION
        or profile_challenger.get("contract_sha256")
        != POLYMARKET_PROFILE_CONTRACT_SHA256
        or profile_challenger.get("control_policy") != "model"
        or profile_challenger.get("challenger_policy") != "profile_model"
        or dict(_as_mapping(profile_challenger["gates"], "profile gates"))
        != expected_profile_gates
        or profile_challenger.get("promotion_gates_passed")
        is not profile_promotion_gates_passed
        or profile_challenger.get("accepted") is not False
        or profile_challenger.get("status") != expected_profile_status
        or profile_challenger.get("requires_later_prospective_confirmation") is not True
    ):
        raise ValueError("profile model challenger does not reconstruct")
    expected_gates = {
        "validation_probability_improved": (
            _finite_float(
                probability["validation_log_loss_delta"],
                "validation log-loss delta",
            )
            < 0.0
        ),
        "untouched_test_probability_improved": (
            _finite_float(
                probability["test_log_loss_delta"],
                "test log-loss delta",
            )
            < 0.0
            and _finite_float(
                probability["test_brier_delta"],
                "test Brier delta",
            )
            < 0.0
        ),
        "minimum_confirmatory_test_time_groups_met": len(time_groups) >= 30,
        "after_cost_execution_improved": (
            _decimal(
                executions["model"]["net_realized_pnl_quote"],
                "model net PnL",
            )
            > _decimal(
                executions["baseline"]["net_realized_pnl_quote"],
                "baseline net PnL",
            )
        ),
        "after_cost_model_improved_at_every_stress_latency": all(
            _decimal(
                _as_mapping(
                    _as_mapping(
                        sensitivity_policies["model"],
                        "model latency reports",
                    )[str(latency)],
                    "model latency report",
                )["net_realized_pnl_quote"],
                "model latency PnL",
            )
            > _decimal(
                _as_mapping(
                    _as_mapping(
                        sensitivity_policies["baseline"],
                        "baseline latency reports",
                    )[str(latency)],
                    "baseline latency report",
                )["net_realized_pnl_quote"],
                "baseline latency PnL",
            )
            for latency in latency_values
        ),
        "retry_challenger_accepted": all(expected_retry_gates.values()),
        "profile_challenger_promotion_gates_passed": (profile_promotion_gates_passed),
        "all_positions_officially_settled": all(
            int(report["filled_order_count"])
            == int(report["winning_order_count"]) + int(report["losing_order_count"])
            and all(
                trade.get("execution_state") != "FILLED"
                or bool(str(trade.get("official_resolution_event_id", "")))
                for trade in report["trades"]
            )
            for report in all_latency_reports
        ),
        "all_order_outcomes_terminal": all(
            trade.get("execution_state") != "UNKNOWN"
            for report in all_latency_reports
            for trade in report["trades"]
        ),
        "ai_enabled": ai.get("enabled") is True,
        "ai_uplift_accepted": bool(
            ai.get("enabled") is True
            and _as_mapping(ai["uplift"], "AI uplift").get("accepted") is True
        ),
        "live_trading_authority": False,
        "profitability_claim": False,
    }
    gates = _as_mapping(payload.get("evidence_gates"), "evidence gates")
    if dict(gates) != expected_gates:
        raise ValueError("Polymarket evidence gates do not reconstruct")
    return ValidatedPolymarketArtifact(
        payload=payload,
        artifact_sha256=claimed_artifact_sha256,
        predictions=tuple(predictions),
        profile_predictions=tuple(profile_predictions),
        executions=executions,
    )


def _probability_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    report = _as_mapping(payload["probability_report"], "probability report")
    profile_report = _as_mapping(
        payload["profile_probability_report"],
        "profile probability report",
    )
    rows: list[dict[str, object]] = []
    for treatment, source, key in (
        ("market_implied", report, "baseline_metrics"),
        ("residual_model", report, "model_metrics"),
        ("profile_challenger", profile_report, "challenger_metrics"),
    ):
        roles = _as_mapping(source[key], f"{treatment} probability metrics")
        for role in ("train", "validation", "test"):
            metric = _as_mapping(roles[role], f"{treatment} {role} metrics")
            rows.append(
                {
                    "role": role,
                    "treatment": treatment,
                    "rows": metric["row_count"],
                    "markets": metric["market_count"],
                    "time_groups": metric["time_group_count"],
                    "effective_market_weight": metric["effective_market_weight"],
                    "weighted_log_loss": metric["weighted_log_loss"],
                    "weighted_brier_score": metric["weighted_brier_score"],
                    "weighted_calibration_error": metric["weighted_calibration_error"],
                    "weighted_accuracy": metric["weighted_accuracy"],
                    "weighted_sharpness": metric["weighted_sharpness"],
                }
            )
    return rows


def _model_selection_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    model = _as_mapping(payload["model"], "model")
    inner = _named_losses(
        model["candidate_inner_log_losses"],
        "inner model losses",
    )
    validation = _named_losses(
        model["validation_gate_log_losses"],
        "outer validation gate losses",
    )
    inner_baseline = inner[0][1]
    validation_baseline = validation[0][1]
    selected_inner = str(model["inner_selected_candidate"])
    accepted = str(model["selected_candidate"])
    rows: list[dict[str, object]] = []
    for candidate, loss in inner:
        rows.append(
            {
                "stage": "inner_selection",
                "evidence_unit": "purged_rolling_fold_weighted_mean",
                "candidate": candidate,
                "l2": (
                    ""
                    if candidate == "market_baseline"
                    else candidate.removeprefix("offset_l2_")
                ),
                "weighted_log_loss": loss,
                "baseline_log_loss": inner_baseline,
                "delta_vs_stage_baseline": loss - inner_baseline,
                "selected_for_outer_gate": candidate == selected_inner,
                "accepted_after_outer_gate": candidate == accepted,
                "fold_count": int(model["inner_fold_count"]),
            }
        )
    for candidate, loss in validation:
        rows.append(
            {
                "stage": "outer_promotion_gate",
                "evidence_unit": "chronological_validation_tail",
                "candidate": candidate,
                "l2": (
                    ""
                    if candidate == "market_baseline"
                    else candidate.removeprefix("offset_l2_")
                ),
                "weighted_log_loss": loss,
                "baseline_log_loss": validation_baseline,
                "delta_vs_stage_baseline": loss - validation_baseline,
                "selected_for_outer_gate": candidate == selected_inner,
                "accepted_after_outer_gate": candidate == accepted,
                "fold_count": 1,
            }
        )
    return rows


def _profile_model_selection_rows(
    payload: Mapping[str, Any],
) -> list[dict[str, object]]:
    model = _as_mapping(payload["profile_model"], "profile model")
    inner = _named_losses(
        model["candidate_inner_log_losses"],
        "profile inner model losses",
    )
    validation = _named_losses(
        model["validation_gate_log_losses"],
        "profile outer validation losses",
    )
    inner_baseline = inner[0][1]
    validation_baseline = validation[0][1]
    selected_inner = str(model["inner_selected_candidate"])
    accepted = str(model["selected_candidate"])
    rows: list[dict[str, object]] = []
    for stage, evidence_unit, values, baseline, fold_count in (
        (
            "inner_selection",
            "purged_rolling_fold_weighted_mean",
            inner,
            inner_baseline,
            int(model["inner_fold_count"]),
        ),
        (
            "outer_promotion_gate",
            "chronological_validation_tail",
            validation,
            validation_baseline,
            1,
        ),
    ):
        for candidate, loss in values:
            if candidate == "market_baseline":
                profile = ""
                active_feature_count: int | str = ""
                l2: float | str = ""
            else:
                profile, feature_names, l2 = _profile_candidate_details(candidate)
                active_feature_count = len(feature_names)
            rows.append(
                {
                    "stage": stage,
                    "evidence_unit": evidence_unit,
                    "candidate": candidate,
                    "profile": profile,
                    "active_feature_count": active_feature_count,
                    "l2": l2,
                    "weighted_log_loss": loss,
                    "baseline_log_loss": baseline,
                    "delta_vs_stage_baseline": loss - baseline,
                    "selected_for_outer_gate": candidate == selected_inner,
                    "accepted_after_outer_gate": candidate == accepted,
                    "fold_count": fold_count,
                }
            )
    return rows


def _prediction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "feature_names": _canonical_json(row["feature_names"]),
            "feature_values": _canonical_json(row["feature_values"]),
            "risk_context_names": _canonical_json(row["risk_context_names"]),
            "risk_context_values": _canonical_json(row["risk_context_values"]),
            "event_start_utc": _utc(row["event_start_ms"]),
            "end_utc": _utc(row["end_ms"]),
            "decision_received_utc": _utc(row["decision_received_wall_ms"]),
        }
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["event_start_ms"]),
                str(item["asset"]),
                -int(item["horizon_seconds"]),
            ),
        )
    ]


def _held_out_group_score_rows(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope in ("ALL", *_ASSETS):
        selected = [
            row for row in predictions if scope == "ALL" or row["asset"] == scope
        ]
        for event_start in sorted({int(row["event_start_ms"]) for row in selected}):
            group = [
                row for row in selected if int(row["event_start_ms"]) == event_start
            ]
            weight = sum(
                _finite_float(row["market_weight"], "market weight") for row in group
            )
            if weight <= 0:
                raise ValueError("held-out group has no effective weight")
            baseline_log_loss = 0.0
            model_log_loss = 0.0
            profile_log_loss = 0.0
            baseline_brier = 0.0
            model_brier = 0.0
            profile_brier = 0.0
            for row in group:
                label = 1.0 if row["official_up"] else 0.0
                row_weight = _finite_float(row["market_weight"], "market weight")
                baseline = _finite_float(
                    row["baseline_up_probability"],
                    "baseline probability",
                )
                model = _finite_float(
                    row["model_up_probability"],
                    "model probability",
                )
                profile = _finite_float(
                    row["profile_model_up_probability"],
                    "profile model probability",
                )
                baseline_log_loss += row_weight * -(
                    label * math.log(baseline) + (1.0 - label) * math.log1p(-baseline)
                )
                model_log_loss += row_weight * -(
                    label * math.log(model) + (1.0 - label) * math.log1p(-model)
                )
                profile_log_loss += row_weight * -(
                    label * math.log(profile) + (1.0 - label) * math.log1p(-profile)
                )
                baseline_brier += row_weight * (baseline - label) ** 2
                model_brier += row_weight * (model - label) ** 2
                profile_brier += row_weight * (profile - label) ** 2
            baseline_log_loss /= weight
            model_log_loss /= weight
            profile_log_loss /= weight
            baseline_brier /= weight
            model_brier /= weight
            profile_brier /= weight
            end_ms = max(int(row["end_ms"]) for row in group)
            rows.append(
                {
                    "scope": scope,
                    "event_start_ms": event_start,
                    "event_start_utc": _utc(event_start),
                    "end_ms": end_ms,
                    "end_utc": _utc(end_ms),
                    "rows": len(group),
                    "markets": len({str(row["condition_id"]) for row in group}),
                    "effective_market_weight": weight,
                    "baseline_log_loss": baseline_log_loss,
                    "model_log_loss": model_log_loss,
                    "profile_model_log_loss": profile_log_loss,
                    "log_loss_delta": model_log_loss - baseline_log_loss,
                    "profile_vs_model_log_loss_delta": (
                        profile_log_loss - model_log_loss
                    ),
                    "baseline_brier_score": baseline_brier,
                    "model_brier_score": model_brier,
                    "profile_model_brier_score": profile_brier,
                    "brier_delta": model_brier - baseline_brier,
                    "profile_vs_model_brier_delta": profile_brier - model_brier,
                }
            )
    return rows


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires values")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _group_score_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    artifact_sha256: str,
) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for scope in ("ALL", *_ASSETS):
        deltas = [
            _finite_float(row["log_loss_delta"], "group log-loss delta")
            for row in rows
            if row["scope"] == scope
        ]
        if not deltas:
            raise ValueError(f"held-out group scores have no {scope} rows")
        block_length = max(2, int(math.ceil(math.sqrt(len(deltas)))))
        circular_blocks = [
            [deltas[(start + offset) % len(deltas)] for offset in range(block_length)]
            for start in range(len(deltas))
        ]
        seed = int(
            hashlib.sha256(f"{artifact_sha256}:{scope}".encode("ascii")).hexdigest()[
                :16
            ],
            16,
        )
        generator = random.Random(seed)
        bootstrap_means: list[float] = []
        for _ in range(10_000):
            sample: list[float] = []
            while len(sample) < len(deltas):
                sample.extend(
                    circular_blocks[generator.randrange(len(circular_blocks))]
                )
            bootstrap_means.append(sum(sample[: len(deltas)]) / len(deltas))
        lower = _quantile(bootstrap_means, 0.025)
        upper = _quantile(bootstrap_means, 0.975)
        summaries[scope] = {
            "time_group_count": len(deltas),
            "mean_log_loss_delta": sum(deltas) / len(deltas),
            "median_log_loss_delta": _quantile(deltas, 0.5),
            "minimum_log_loss_delta": min(deltas),
            "maximum_log_loss_delta": max(deltas),
            "improved_time_groups": sum(value < 0.0 for value in deltas),
            "unchanged_time_groups": sum(value == 0.0 for value in deltas),
            "degraded_time_groups": sum(value > 0.0 for value in deltas),
            "moving_block_bootstrap_95pct_lower": lower,
            "moving_block_bootstrap_95pct_upper": upper,
            "block_length_time_groups": block_length,
            "bootstrap_resamples": 10_000,
            "minimum_confirmatory_time_groups": 30,
            "confirmatory_ready": len(deltas) >= 30 and upper < 0.0,
        }
    body = {
        "schema_version": "polymarket-held-out-group-score-summary-v1",
        "independent_unit": "shared_btc_eth_sol_five_minute_time_group",
        "method": "deterministic_circular_moving_block_bootstrap",
        "artifact_sha256": artifact_sha256,
        "scopes": summaries,
        "trading_authority": False,
        "profitability_claim": False,
    }
    return {**body, "summary_sha256": _canonical_sha256(body)}


def _execution_summary_rows(
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    fields = (
        "evaluated_market_count",
        "signal_market_count",
        "attempted_order_count",
        "filled_order_count",
        "winning_order_count",
        "losing_order_count",
        "abstained_market_count",
        "gross_deployed_capital_quote",
        "gross_payout_quote",
        "total_fees_quote",
        "net_realized_pnl_quote",
        "initial_capital_quote",
        "final_equity_quote",
        "return_on_initial_capital",
        "return_on_deployed_capital",
        "maximum_drawdown_quote",
        "maximum_drawdown_fraction",
        "report_sha256",
    )
    return [
        {"policy": policy, **{field: report[field] for field in fields}}
        for policy, report in executions.items()
    ]


def _latency_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    sensitivity = _as_mapping(
        payload["execution_latency_sensitivity"],
        "execution latency sensitivity",
    )
    latencies = [int(value) for value in sensitivity["network_latencies_ms"]]
    policies = _as_mapping(sensitivity["policies"], "latency policies")
    rows: list[dict[str, object]] = []
    for policy, raw_reports in policies.items():
        reports = _as_mapping(raw_reports, f"{policy} latency reports")
        for latency in latencies:
            report = _as_mapping(reports[str(latency)], "latency report")
            delays = [
                int(value)
                for value in _as_mapping(
                    report["decision_delay_ms_by_condition"],
                    "decision delays",
                ).values()
            ]
            trades = report["trades"]
            rows.append(
                {
                    "policy": policy,
                    "network_latency_ms": latency,
                    "mean_decision_delay_ms": (
                        sum(delays) / len(delays) if delays else 0.0
                    ),
                    "maximum_decision_delay_ms": max(delays, default=0),
                    "evaluated_markets": report["evaluated_market_count"],
                    "signals": report["signal_market_count"],
                    "attempted_orders": report["attempted_order_count"],
                    "fills": report["filled_order_count"],
                    "indeterminate_orders": sum(
                        row["execution_state"] == "UNKNOWN" for row in trades
                    ),
                    "net_realized_pnl_quote": report["net_realized_pnl_quote"],
                    "return_on_initial_capital": report["return_on_initial_capital"],
                    "maximum_drawdown_fraction": report["maximum_drawdown_fraction"],
                    "report_sha256": report["report_sha256"],
                }
            )
    return rows


def _equity_rows(
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy, report in executions.items():
        for point in report["equity_curve"]:
            rows.append(
                {
                    "policy": policy,
                    "event_start_ms": point["event_start_ms"],
                    "event_start_utc": _utc(point["event_start_ms"]),
                    "settled_at_ms": point["settled_at_ms"],
                    "settled_at_utc": _utc(point["settled_at_ms"]),
                    "group_realized_pnl_quote": point["group_realized_pnl_quote"],
                    "equity_quote": point["equity_quote"],
                    "peak_equity_quote": point["peak_equity_quote"],
                    "drawdown_quote": point["drawdown_quote"],
                    "drawdown_fraction": point["drawdown_fraction"],
                }
            )
    if not rows:
        raise ValueError("execution reports contain no equity points")
    return rows


def _trade_rows(
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fields: list[str] | None = None
    for policy, report in executions.items():
        for trade in report["trades"]:
            row = {
                "policy": policy,
                **dict(trade),
                "event_start_utc": _utc(trade["event_start_ms"]),
                "end_utc": _utc(trade["end_ms"]),
                "decision_received_utc": _utc(trade["decision_received_wall_ms"]),
            }
            if fields is None:
                fields = list(row)
            elif set(row) != set(fields):
                raise ValueError("execution trade schemas differ between policies")
            rows.append({key: row[key] for key in fields})
    if rows:
        return rows
    return [
        {
            "policy": "none",
            "execution_state": "NO_PROPOSALS",
            "execution_reason": "no positive after-fee proposal",
        }
    ]


def _per_asset_rows(
    predictions: Sequence[Mapping[str, Any]],
    executions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, object]]:
    market_counts = {
        asset: len(
            {str(row["condition_id"]) for row in predictions if row["asset"] == asset}
        )
        for asset in _ASSETS
    }
    rows: list[dict[str, object]] = []
    for policy, report in executions.items():
        trades = report["trades"]
        for asset in _ASSETS:
            selected = [row for row in trades if row["asset"] == asset]
            filled = [row for row in selected if row["execution_state"] == "FILLED"]
            pnl = sum(
                (_decimal(row["realized_pnl_quote"], "trade PnL") for row in filled),
                Decimal("0"),
            )
            fees = sum(
                (_decimal(row["fee_quote"], "trade fee") for row in filled),
                Decimal("0"),
            )
            deployed = sum(
                (
                    _decimal(row["average_fill_price"], "fill price")
                    * _decimal(row["filled_quantity"], "fill quantity")
                    for row in filled
                ),
                Decimal("0"),
            )
            rows.append(
                {
                    "policy": policy,
                    "asset": asset,
                    "evaluated_markets": market_counts[asset],
                    "signals": len(selected),
                    "fills": len(filled),
                    "wins": sum(
                        _decimal(row["realized_pnl_quote"], "trade PnL") > 0
                        for row in filled
                    ),
                    "losses": sum(
                        _decimal(row["realized_pnl_quote"], "trade PnL") < 0
                        for row in filled
                    ),
                    "fees_quote": str(fees),
                    "deployed_capital_quote": str(deployed),
                    "net_realized_pnl_quote": str(pnl),
                    "return_on_deployed_capital": str(
                        pnl / deployed if deployed > 0 else Decimal("0")
                    ),
                }
            )
    return rows


def _ai_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    ai = _as_mapping(payload["ai"], "AI evidence")
    if ai.get("enabled") is not True:
        return [{"enabled": False, "reason": ai.get("reason", "operator_disabled")}]
    veto = _as_mapping(ai["veto_report"], "AI veto report")
    results = veto.get("results")
    if not isinstance(results, list) or not results:
        return [{"enabled": True, "reason": "no_positive_after_fee_proposals"}]
    rows: list[dict[str, object]] = []
    for result in results:
        decision = _as_mapping(result["decision"], "AI decision")
        rows.append(
            {
                "enabled": True,
                "case_id": result["case_id"],
                "condition_id": result["condition_id"],
                "model": result["model"],
                "latency_seconds": result["latency_seconds"],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "valid": decision["valid"],
                "permits_entry": decision["permits_entry"],
                "reason_codes": ";".join(decision["reason_codes"]),
                "summary": decision["summary"],
                "failure_reason": decision["failure_reason"],
                "response_sha256": result["response_sha256"],
            }
        )
    return rows


def _ai_case_rows(payload: Mapping[str, Any]) -> list[dict[str, object]]:
    ai = _as_mapping(payload["ai"], "AI evidence")
    base = {
        "enabled": bool(ai.get("enabled")),
        "case_id": "",
        "case_sha256": "",
        "sample_id": "",
        "condition_id": "",
        "asset": "",
        "event_start_utc": "",
        "decision_received_utc": "",
        "proposed_outcome": "",
        "prompt_schema_version": "",
        "prompt_payload_sha256": "",
        "prompt_payload_json": "",
        "reason": "",
    }
    if ai.get("enabled") is not True:
        return [{**base, "reason": ai.get("reason", "operator_disabled")}]
    raw_cases = ai.get("prompt_cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        return [{**base, "reason": "no_positive_after_fee_proposals"}]
    rows: list[dict[str, object]] = []
    for raw_case in raw_cases:
        case = _as_mapping(raw_case, "AI prompt case")
        prompt = _as_mapping(case["prompt_payload"], "AI prompt")
        rows.append(
            {
                **base,
                "case_id": case["case_id"],
                "case_sha256": case["case_sha256"],
                "sample_id": case["sample_id"],
                "condition_id": case["condition_id"],
                "asset": case["asset"],
                "event_start_utc": _utc(case["event_start_ms"]),
                "decision_received_utc": _utc(case["decision_received_wall_ms"]),
                "proposed_outcome": prompt["proposed_outcome"],
                "prompt_schema_version": prompt["schema_version"],
                "prompt_payload_sha256": _canonical_sha256(prompt),
                "prompt_payload_json": _canonical_json(prompt),
            }
        )
    return rows


def _svg_base(
    title: str, subtitle: str, description: str, *, height: int = 700
) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(description)}</desc>",
        f'<rect width="1200" height="{height}" fill="{_COLORS["background"]}"/>',
        f'<text x="64" y="58" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="28" font-weight="700">{escape(title)}</text>',
        f'<text x="64" y="88" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="15">{escape(subtitle)}</text>',
    ]


def _probability_svg(
    rows: Sequence[Mapping[str, object]], *, start: str, end: str
) -> str:
    selected = {
        (str(row["role"]), str(row["treatment"])): row
        for row in rows
        if row["role"] in {"validation", "test"}
    }
    categories = (
        ("Validation log loss", "validation", "weighted_log_loss"),
        ("Test log loss", "test", "weighted_log_loss"),
        ("Test Brier score", "test", "weighted_brier_score"),
    )
    values = [
        _finite_float(selected[(role, treatment)][metric], metric)
        for _label, role, metric in categories
        for treatment in (
            "market_implied",
            "residual_model",
            "profile_challenger",
        )
    ]
    maximum = max(values) * 1.15 if max(values) > 0 else 1.0
    lines = _svg_base(
        "Proper-score comparison",
        f"Chronological validation and untouched test; held-out window {start} to {end}",
        "Weighted proper probability scores from probability-metrics.csv. Lower is better.",
    )
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    for index in range(5):
        value = maximum * index / 4
        y = bottom - (bottom - top) * index / 4
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}"/>'
        )
        lines.append(
            f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.3f}</text>'
        )
    group_width = width / len(categories)
    for index, (label, role, metric) in enumerate(categories):
        center = left + group_width * (index + 0.5)
        for offset, treatment in (
            (-72.0, "market_implied"),
            (0.0, "residual_model"),
            (72.0, "profile_challenger"),
        ):
            value = _finite_float(selected[(role, treatment)][metric], metric)
            height = (bottom - top) * value / maximum
            x = center + offset - 27.0
            y = bottom - height
            color = _COLORS[
                "baseline"
                if treatment == "market_implied"
                else "profile_model"
                if treatment == "profile_challenger"
                else "model"
            ]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="54" height="{height:.1f}" rx="3" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{x + 27:.1f}" y="{y - 10:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12" font-weight="700">{value:.4f}</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="608" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="15">{escape(label)}</text>'
        )
    lines.extend(
        (
            f'<rect x="235" y="645" width="18" height="18" rx="2" fill="{_COLORS["baseline"]}"/><text x="263" y="659" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Market-implied prior</text>',
            f'<rect x="500" y="645" width="18" height="18" rx="2" fill="{_COLORS["model"]}"/><text x="528" y="659" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Round-3 control</text>',
            f'<rect x="750" y="645" width="18" height="18" rx="2" fill="{_COLORS["profile_model"]}"/><text x="778" y="659" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">Profile challenger</text>',
            "</svg>",
        )
    )
    return "\n".join(lines) + "\n"


def _model_selection_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    start: str,
    end: str,
) -> str:
    inner = [row for row in rows if row["stage"] == "inner_selection"]
    outer = [row for row in rows if row["stage"] == "outer_promotion_gate"]
    ordered = [*inner, *outer]
    deltas = [
        _finite_float(row["delta_vs_stage_baseline"], "selection delta")
        for row in ordered
    ]
    extent = max(max((abs(value) for value in deltas), default=0.0) * 1.15, 0.0001)
    plot_left, plot_center, plot_right = 460.0, 700.0, 940.0
    row_centers = [180.0 + index * 54.0 for index in range(len(inner))]
    row_centers.extend(540.0 + index * 58.0 for index in range(len(outer)))
    lines = _svg_base(
        "Nested model selection",
        f"Inner purged rolling selection, then one outer promotion gate; {start} to {end}",
        "Log-loss deltas derived from model-selection.csv. Negative values improve on the market-implied baseline.",
        height=740,
    )
    for value, x in ((-extent, plot_left), (0.0, plot_center), (extent, plot_right)):
        lines.append(
            f'<line x1="{x:.1f}" y1="132" x2="{x:.1f}" y2="650" '
            f'stroke="{_COLORS["grid"]}" stroke-width="{2 if value == 0 else 1}"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="122" text-anchor="middle" '
            f'fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" '
            f'font-size="12">{value:+.5f}</text>'
        )
    lines.append(
        f'<text x="64" y="151" fill="{_COLORS["muted"]}" '
        'font-family="Segoe UI,Arial,sans-serif" font-size="13" font-weight="700">'
        "INNER TRAINING FOLDS</text>"
    )
    lines.append(
        f'<line x1="64" y1="505" x2="1136" y2="505" stroke="{_COLORS["grid"]}"/>'
    )
    lines.append(
        f'<text x="64" y="530" fill="{_COLORS["muted"]}" '
        'font-family="Segoe UI,Arial,sans-serif" font-size="13" font-weight="700">'
        "OUTER VALIDATION GATE</text>"
    )
    for row, y, delta in zip(ordered, row_centers, deltas, strict=True):
        candidate = str(row["candidate"])
        stage = str(row["stage"])
        label = (
            "Market-implied prior"
            if candidate == "market_baseline"
            else f"L2 = {_finite_float(row['l2'], 'L2 label'):.6g}"
        )
        value_x = plot_center + 240.0 * delta / extent
        bar_x = min(plot_center, value_x)
        bar_width = max(abs(value_x - plot_center), 2.0)
        color = (
            _COLORS["baseline"]
            if delta == 0.0
            else _COLORS["model"]
            if delta < 0.0
            else _COLORS["negative"]
        )
        accepted = bool(row["accepted_after_outer_gate"])
        inner_winner = bool(row["selected_for_outer_gate"])
        rejected = (
            stage == "outer_promotion_gate"
            and candidate != "market_baseline"
            and inner_winner
            and not accepted
        )
        stroke = (
            _COLORS["negative"]
            if rejected
            else _COLORS["model"]
            if accepted
            else _COLORS["ai"]
            if inner_winner
            else color
        )
        stroke_width = 3 if accepted or rejected else 2 if inner_winner else 0
        lines.append(
            f'<text x="438" y="{y + 5:.1f}" text-anchor="end" '
            f'fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" '
            f'font-size="14">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{bar_x:.1f}" y="{y - 13:.1f}" width="{bar_width:.1f}" '
            f'height="26" rx="2" fill="{color}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text x="960" y="{y + 5:.1f}" fill="{_COLORS["text"]}" '
            f'font-family="Consolas,monospace" font-size="13">{delta:+.6f}</text>'
        )
        if stage == "inner_selection" and inner_winner:
            status = "INNER WINNER"
        elif rejected:
            status = "REJECTED"
        elif accepted and candidate == "market_baseline":
            status = "FALLBACK"
        elif accepted:
            status = "ACCEPTED"
        else:
            status = ""
        if status:
            lines.append(
                f'<text x="1060" y="{y + 5:.1f}" fill="{stroke}" '
                'font-family="Segoe UI,Arial,sans-serif" font-size="11" '
                f'font-weight="700">{status}</text>'
            )
    lines.append(
        f'<text x="64" y="704" fill="{_COLORS["muted"]}" '
        'font-family="Segoe UI,Arial,sans-serif" font-size="13">'
        "Selection uses training folds only. The outer tail can reject, but cannot retune, the frozen candidate.</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _profile_model_selection_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    start: str,
    end: str,
) -> str:
    inner = [
        row
        for row in rows
        if row["stage"] == "inner_selection" and row["candidate"] != "market_baseline"
    ]
    outer = [
        row
        for row in rows
        if row["stage"] == "outer_promotion_gate"
        and row["candidate"] != "market_baseline"
    ]
    ordered = [*inner, *outer]
    deltas = [
        _finite_float(row["delta_vs_stage_baseline"], "profile selection delta")
        for row in ordered
    ]
    extent = max(max((abs(value) for value in deltas), default=0.0) * 1.15, 0.0001)
    outer_spacing = 18.0
    height = int(180 + 43 * len(ordered) + outer_spacing + 90)
    top = 150.0
    plot_center = 720.0
    half_width = 205.0
    lines = _svg_base(
        "Frozen profile challenger grid",
        f"Four declared feature profiles by five L2 values; {start} to {end}",
        "Training-fold weighted log-loss deltas from profile-model-selection.csv. Lower is better; test outcomes are excluded from selection.",
        height=height,
    )
    bottom = top + 43.0 * len(ordered) + outer_spacing
    for value, x in (
        (-extent, plot_center - half_width),
        (0.0, plot_center),
        (extent, plot_center + half_width),
    ):
        lines.append(
            f'<line x1="{x:.1f}" y1="125" x2="{x:.1f}" y2="{bottom:.1f}" stroke="{_COLORS["grid"]}" stroke-width="{2 if value == 0 else 1}"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="116" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{value:+.6f}</text>'
        )
    prior_profile = ""
    for index, (row, delta) in enumerate(zip(ordered, deltas, strict=True)):
        y = top + index * 43.0 + (outer_spacing if index >= len(inner) else 0.0)
        if index == len(inner):
            lines.append(
                f'<line x1="64" y1="{y - 24:.1f}" x2="1136" y2="{y - 24:.1f}" stroke="{_COLORS["grid"]}"/>'
            )
            lines.append(
                f'<text x="64" y="{y - 7:.1f}" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="11" font-weight="700">OUTER VALIDATION GATE</text>'
            )
            prior_profile = ""
        profile = str(row["profile"])
        if profile != prior_profile:
            lines.append(
                f'<text x="64" y="{y + 5:.1f}" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12" font-weight="700">{escape(profile.upper() or "OUTER GATE")}</text>'
            )
            prior_profile = profile
        label = (
            f"L2 {_finite_float(row['l2'], 'profile L2'):.6g} / "
            f"{row['active_feature_count']} features"
        )
        value_x = plot_center + half_width * delta / extent
        x = min(plot_center, value_x)
        width = max(abs(value_x - plot_center), 2.0)
        outer_gate = row["stage"] == "outer_promotion_gate"
        accepted = bool(row["accepted_after_outer_gate"]) and outer_gate
        winner = bool(row["selected_for_outer_gate"])
        rejected = outer_gate and winner and not accepted
        color = _COLORS["profile_model"] if delta < 0.0 else _COLORS["negative"]
        stroke = (
            _COLORS["text"]
            if accepted
            else _COLORS["negative"]
            if rejected
            else _COLORS["ai"]
            if winner
            else color
        )
        lines.append(
            f'<text x="478" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{x:.1f}" y="{y - 11:.1f}" width="{width:.1f}" height="22" rx="2" fill="{color}" stroke="{stroke}" stroke-width="{3 if accepted else 2 if winner else 0}"/>'
        )
        lines.append(
            f'<text x="950" y="{y + 5:.1f}" fill="{_COLORS["text"]}" font-family="Consolas,monospace" font-size="12">{delta:+.7f}</text>'
        )
        status = (
            "ACCEPTED"
            if accepted
            else "REJECTED"
            if rejected
            else "INNER WINNER"
            if winner
            else ""
        )
        if status:
            lines.append(
                f'<text x="1080" y="{y + 5:.1f}" fill="{stroke}" font-family="Segoe UI,Arial,sans-serif" font-size="10" font-weight="700">{status}</text>'
            )
    lines.append(
        f'<text x="64" y="{height - 35}" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">A later prospective capture is required even if every exploratory promotion gate passes.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _policy_label(policy: str) -> str:
    return {
        "baseline": "Market prior",
        "model": "Round-3 control",
        "profile_model": "Profile challenger",
        "model_retry": "Control retry",
        "ai": "AI veto",
    }.get(policy, policy.replace("_", " ").title())


def _group_score_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    start: str,
    end: str,
    value_key: str = "log_loss_delta",
    title: str = "Untouched-test log-loss delta by time group",
    comparison: str = "residual-model minus market-prior",
    negative_label: str = "Negative: residual model improved",
    positive_label: str = "Positive: residual model degraded",
    negative_color: str | None = None,
) -> str:
    selected = sorted(
        (row for row in rows if row["scope"] == "ALL"),
        key=lambda row: int(row["event_start_ms"]),
    )
    values = [_finite_float(row[value_key], "group log-loss delta") for row in selected]
    extent = max(max((abs(value) for value in values), default=0.0) * 1.2, 0.01)
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    zero = (top + bottom) / 2
    lines = _svg_base(
        title,
        f"Shared BTC/ETH/SOL five-minute groups; {start} to {end}; lower is better",
        f"Market-equal weighted {comparison} log loss from held-out-group-scores.csv. Negative bars improve on the comparator.",
    )
    for value in (-extent, -extent / 2, 0.0, extent / 2, extent):
        y = zero - (bottom - top) * 0.5 * value / extent
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}" stroke-width="{2 if value == 0 else 1}"/>'
        )
        lines.append(
            f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:.4f}</text>'
        )
    slot = width / max(1, len(selected))
    bar_width = max(3.0, min(70.0, slot * 0.72))
    for index, (row, value) in enumerate(zip(selected, values, strict=True)):
        center = left + slot * (index + 0.5)
        y_value = zero - (bottom - top) * 0.5 * value / extent
        y = min(zero, y_value)
        height = max(abs(y_value - zero), 1.0)
        color = (
            negative_color or _COLORS["model"]
            if value < 0.0
            else _COLORS["negative"]
            if value > 0.0
            else _COLORS["baseline"]
        )
        lines.append(
            f'<rect x="{center - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="2" fill="{color}"/>'
        )
        if len(selected) <= 12:
            label_y = y + height + 16 if value < 0.0 else y - 8
            lines.append(
                f'<text x="{center:.1f}" y="{label_y:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="11">{value:+.4f}</text>'
            )
    tick_indexes = (
        sorted(
            {
                round(index * (len(selected) - 1) / min(5, len(selected) - 1))
                for index in range(min(5, len(selected) - 1) + 1)
            }
        )
        if len(selected) > 1
        else [0]
    )
    for index in tick_indexes:
        center = left + slot * (index + 0.5)
        timestamp = int(selected[index]["event_start_ms"])
        label = datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc).strftime(
            "%H:%MZ"
        )
        lines.append(
            f'<text x="{center:.1f}" y="610" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{label}</text>'
        )
    lines.append(
        f'<text x="120" y="660" fill="{negative_color or _COLORS["model"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{escape(negative_label)}</text>'
    )
    lines.append(
        f'<text x="880" y="660" fill="{_COLORS["negative"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{escape(positive_label)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _equity_svg(rows: Sequence[Mapping[str, object]], *, start: str, end: str) -> str:
    by_policy: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(
            (int(row["settled_at_ms"]), _finite_float(row["equity_quote"], "equity"))
        )
    for values in by_policy.values():
        values.sort()
    all_points = [point for values in by_policy.values() for point in values]
    min_time = min(point[0] for point in all_points)
    max_time = max(point[0] for point in all_points)
    min_value = min(point[1] for point in all_points)
    max_value = max(point[1] for point in all_points)
    padding = max((max_value - min_value) * 0.15, max(abs(max_value), 1.0) * 0.001)
    low, high = min_value - padding, max_value + padding
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    lines = _svg_base(
        "Held-out settled equity",
        f"Full-depth FOK replay with measured feed arrival and fixed latency; {start} to {end}",
        "Settled equity from equity-curves.csv. Every filled order is tied to an official resolution.",
    )
    for index in range(5):
        value = low + (high - low) * index / 4
        y = bottom - (bottom - top) * index / 4
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}"/>'
        )
        lines.append(
            f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:,.2f}</text>'
        )
    for policy in _POLICIES:
        values = by_policy.get(policy)
        if not values:
            continue
        points: list[str] = []
        for timestamp, equity in values:
            x = left + width * (timestamp - min_time) / max(1, max_time - min_time)
            y = bottom - (bottom - top) * (equity - low) / (high - low)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLORS[policy]}" stroke-width="3" stroke-dasharray="{_POLICY_DASH[policy]}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    tick_count = 5
    for index in range(tick_count):
        timestamp = min_time + (max_time - min_time) * index // (tick_count - 1)
        x = left + width * index / (tick_count - 1)
        label = datetime.fromtimestamp(timestamp / 1_000.0, tz=timezone.utc).strftime(
            "%H:%MZ"
        )
        lines.append(
            f'<text x="{x:.1f}" y="608" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{label}</text>'
        )
    present_policies = tuple(name for name in _POLICIES if name in by_policy)
    legend_slot = 1080.0 / len(present_policies)
    for index, policy in enumerate(present_policies):
        x = 60.0 + index * legend_slot
        label = _policy_label(policy)
        lines.append(
            f'<line x1="{x:.1f}" y1="652" x2="{x + 24:.1f}" y2="652" stroke="{_COLORS[policy]}" stroke-width="4"/><text x="{x + 32:.1f}" y="657" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{escape(label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _asset_svg(rows: Sequence[Mapping[str, object]], *, start: str, end: str) -> str:
    policies = tuple(dict.fromkeys(str(row["policy"]) for row in rows))
    values = [_finite_float(row["net_realized_pnl_quote"], "asset PnL") for row in rows]
    extent = max(max((abs(value) for value in values), default=0.0) * 1.25, 1.0)
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    zero = (top + bottom) / 2
    lines = _svg_base(
        "After-fee realized PnL by asset",
        f"BTC, ETH, and SOL five-minute markets; held-out window {start} to {end}",
        "Net settled quote-currency PnL and fill counts from per-asset-execution.csv. No leverage.",
    )
    for value in (-extent, -extent / 2, 0.0, extent / 2, extent):
        y = zero - (bottom - top) * 0.5 * value / extent
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}" stroke-width="{2 if value == 0 else 1}"/>'
        )
        lines.append(
            f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:,.2f}</text>'
        )
    group_width = width / len(_ASSETS)
    bar_width = min(68.0, group_width / (len(policies) + 1))
    for asset_index, asset in enumerate(_ASSETS):
        center = left + group_width * (asset_index + 0.5)
        for policy_index, policy in enumerate(policies):
            row = next(
                item
                for item in rows
                if item["asset"] == asset and item["policy"] == policy
            )
            value = _finite_float(row["net_realized_pnl_quote"], "asset PnL")
            x = (
                center
                + (policy_index - (len(policies) - 1) / 2) * (bar_width + 10)
                - bar_width / 2
            )
            y_value = zero - (bottom - top) * 0.5 * value / extent
            y = min(zero, y_value)
            height = max(abs(y_value - zero), 1.0)
            color = _COLORS[policy]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{color}"/>'
            )
            label_y = y - 9 if value >= 0 else y + height + 19
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="{10 if len(policies) > 4 else 12}">{value:.2f} / {row["fills"]}</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="610" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="700">{asset}</text>'
        )
    legend_slot = 1080.0 / len(policies)
    for index, policy in enumerate(policies):
        x = 60.0 + index * legend_slot
        lines.append(
            f'<rect x="{x:.1f}" y="642" width="16" height="16" rx="2" fill="{_COLORS[policy]}"/><text x="{x + 24:.1f}" y="655" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{escape(_policy_label(policy))}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _latency_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    start: str,
    end: str,
) -> str:
    policies = tuple(dict.fromkeys(str(row["policy"]) for row in rows))
    latencies = sorted({int(row["network_latency_ms"]) for row in rows})
    values = [
        _finite_float(row["net_realized_pnl_quote"], "latency PnL") for row in rows
    ]
    low = min(values + [0.0])
    high = max(values + [0.0])
    padding = max((high - low) * 0.15, 1.0)
    low -= padding
    high += padding
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    lines = _svg_base(
        "Network-latency sensitivity",
        f"Causal full-depth FOK replay; held-out window {start} to {end}",
        "Settled after-fee PnL by predeclared network latency from latency-sensitivity.csv. AI lines also include measured model decision delay.",
    )
    for index in range(5):
        value = low + (high - low) * index / 4
        y = bottom - (bottom - top) * index / 4
        lines.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + width:.1f}" y2="{y:.1f}" stroke="{_COLORS["grid"]}"/>'
        )
        lines.append(
            f'<text x="{left - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{value:,.2f}</text>'
        )
    x_by_latency = {
        latency: left + width * index / max(1, len(latencies) - 1)
        for index, latency in enumerate(latencies)
    }
    for latency, x in x_by_latency.items():
        lines.append(
            f'<text x="{x:.1f}" y="608" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{latency} ms</text>'
        )
    for policy_index, policy in enumerate(policies):
        policy_rows = {
            int(row["network_latency_ms"]): row
            for row in rows
            if row["policy"] == policy
        }
        points: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        for latency in latencies:
            value = _finite_float(
                policy_rows[latency]["net_realized_pnl_quote"],
                "latency PnL",
            )
            x = x_by_latency[latency]
            y = bottom - (bottom - top) * (value - low) / (high - low)
            points.append(f"{x:.2f},{y:.2f}")
            coordinates.append((x, y, value))
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{_COLORS[policy]}" stroke-width="3" stroke-dasharray="{_POLICY_DASH[policy]}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, y, value in coordinates:
            direction = 1.0 if x < 600.0 else -1.0
            label_x = x + direction * (12.0 + policy_index * 48.0)
            anchor = "start" if direction > 0.0 else "end"
            lines.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{label_x:.1f}" y2="{y - 10:.1f}" stroke="{_COLORS[policy]}" stroke-width="1"/><circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{_COLORS[policy]}"/><text x="{label_x:.1f}" y="{y - 12:.1f}" text-anchor="{anchor}" fill="{_COLORS[policy]}" font-family="Segoe UI,Arial,sans-serif" font-size="11" font-weight="700">{value:.2f}</text>'
            )
    legend_slot = 1080.0 / len(policies)
    for index, policy in enumerate(policies):
        x = 60.0 + index * legend_slot
        label = _policy_label(policy)
        lines.append(
            f'<line x1="{x:.1f}" y1="652" x2="{x + 24:.1f}" y2="652" stroke="{_COLORS[policy]}" stroke-width="4"/><text x="{x + 32:.1f}" y="657" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="12">{escape(label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _progress_rows(
    artifact: ValidatedPolymarketArtifact,
    prior_round_path: Path | None,
    round_number: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if prior_round_path is not None and prior_round_path.is_file():
        prior = _as_mapping(
            json.loads(prior_round_path.read_text(encoding="utf-8")), "prior round"
        )
        counts = _as_mapping(
            _as_mapping(prior.get("dataset"), "prior dataset").get(
                "labeled_market_counts"
            ),
            "prior market counts",
        )
        rows.append(
            {
                "round": int(prior.get("round", 2)),
                "status": prior.get("status", "pipeline_evidence"),
                "BTC_markets": counts.get("BTC", 0),
                "ETH_markets": counts.get("ETH", 0),
                "SOL_markets": counts.get("SOL", 0),
                "feature_rows": _as_mapping(prior["dataset"], "prior dataset").get(
                    "row_count", 0
                ),
                "model_evaluated": False,
                "test_log_loss_delta": "",
                "model_net_realized_pnl_quote": "",
                "profitability_claim": False,
            }
        )
    payload = artifact.payload
    counts = _as_mapping(
        _as_mapping(payload["model_dataset"], "model dataset")["market_counts"],
        "market counts",
    )
    report = _as_mapping(payload["probability_report"], "probability report")
    model_execution = artifact.executions["model"]
    rows.append(
        {
            "round": round_number,
            "status": "prospective_model_evaluated",
            "BTC_markets": counts["BTC"],
            "ETH_markets": counts["ETH"],
            "SOL_markets": counts["SOL"],
            "feature_rows": _as_mapping(payload["feature_dataset"], "feature dataset")[
                "row_count"
            ],
            "model_evaluated": True,
            "test_log_loss_delta": report["test_log_loss_delta"],
            "model_net_realized_pnl_quote": model_execution["net_realized_pnl_quote"],
            "profitability_claim": False,
        }
    )
    return rows


def _progress_svg(rows: Sequence[Mapping[str, object]]) -> str:
    maximum = max(int(row[f"{asset}_markets"]) for row in rows for asset in _ASSETS)
    maximum = max(maximum, 30)
    lines = _svg_base(
        "Prospective evidence progression",
        "Resolved market coverage by research round; production gate is 30 markets per asset",
        "Real prospective market counts from research-progress.csv. This is an evidence-coverage chart, not a profitability chart.",
    )
    left, top, bottom, width = 120.0, 150.0, 570.0, 1010.0
    gate_y = bottom - (bottom - top) * 30 / maximum
    lines.append(
        f'<line x1="{left:.1f}" y1="{gate_y:.1f}" x2="{left + width:.1f}" y2="{gate_y:.1f}" stroke="#f8fafc" stroke-width="2" stroke-dasharray="8 7"/>'
    )
    lines.append(
        f'<text x="{left + width - 5:.1f}" y="{gate_y - 9:.1f}" text-anchor="end" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">30-market research gate</text>'
    )
    group_width = width / len(rows)
    for round_index, row in enumerate(rows):
        center = left + group_width * (round_index + 0.5)
        for asset_index, asset in enumerate(_ASSETS):
            value = int(row[f"{asset}_markets"])
            bar_width = min(76.0, group_width / 5)
            x = center + (asset_index - 1) * (bar_width + 14) - bar_width / 2
            height = (bottom - top) * value / maximum
            y = bottom - height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="3" fill="{_COLORS[asset]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 9:.1f}" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14" font-weight="700">{value}</text>'
            )
        lines.append(
            f'<text x="{center:.1f}" y="610" text-anchor="middle" fill="{_COLORS["text"]}" font-family="Segoe UI,Arial,sans-serif" font-size="17" font-weight="700">Round {row["round"]}</text>'
        )
        lines.append(
            f'<text x="{center:.1f}" y="634" text-anchor="middle" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="13">{escape(str(row["status"]))}</text>'
        )
    for index, asset in enumerate(_ASSETS):
        x = 410 + index * 145
        lines.append(
            f'<rect x="{x}" y="656" width="18" height="18" rx="2" fill="{_COLORS[asset]}"/><text x="{x + 28}" y="671" fill="{_COLORS["muted"]}" font-family="Segoe UI,Arial,sans-serif" font-size="14">{asset}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _results_markdown(
    artifact: ValidatedPolymarketArtifact,
    round_number: int,
    start: str,
    end: str,
    group_score_summary: Mapping[str, object],
) -> str:
    payload = artifact.payload
    report = _as_mapping(payload["probability_report"], "probability report")
    trained_model = _as_mapping(payload["model"], "model")
    profile_report = _as_mapping(
        payload["profile_probability_report"],
        "profile probability report",
    )
    profile_model = _as_mapping(payload["profile_model"], "profile model")
    baseline = artifact.executions["baseline"]
    model = artifact.executions["model"]
    profile_execution = artifact.executions["profile_model"]
    model_retry = artifact.executions["model_retry"]
    retry_challenger = _as_mapping(
        payload["retry_challenger"],
        "model retry challenger",
    )
    profile_challenger = _as_mapping(
        payload["profile_challenger"],
        "profile model challenger",
    )
    ai = _as_mapping(payload["ai"], "AI evidence")
    ai_line = "AI was disabled for this experiment."
    if ai.get("enabled") is True:
        ai_execution = artifact.executions["ai"]
        uplift = _as_mapping(ai["uplift"], "AI uplift")
        ai_line = (
            f"The veto-only AI path filled {ai_execution['filled_order_count']} orders, "
            f"settled {ai_execution['net_realized_pnl_quote']} quote PnL, and its governed "
            f"uplift gate was **{'accepted' if uplift.get('accepted') else 'not accepted'}**."
        )
    gates = _as_mapping(payload["evidence_gates"], "evidence gates")
    all_groups = _as_mapping(
        _as_mapping(group_score_summary["scopes"], "score scopes")["ALL"],
        "combined group score summary",
    )
    gate_rows = "\n".join(
        f"| `{key}` | `{str(value).lower()}` |" for key, value in gates.items()
    )
    return f"""# Polymarket research round {round_number}

This report is generated from one hash-verified prospective artifact. It covers
BTC, ETH, and SOL five-minute markets from `{start}` through `{end}`. It is
research evidence only: no live trading, portfolio, leverage, or profitability
claim is made.

![Nested model selection](latest/charts/model-selection.svg)

The purged inner folds selected `{trained_model["inner_selected_candidate"]}`.
The independent outer validation gate then
**{"accepted" if trained_model["selected_candidate"] != "market_baseline" else "rejected"}**
that frozen candidate; it was not used to retune regularization.

![Proper-score comparison](latest/charts/probability-quality.svg)

The selected candidate was `{report["selected_candidate"]}`. Relative to the
market-implied prior, validation log-loss changed by
`{_finite_float(report["validation_log_loss_delta"], "validation delta"):.8f}`
and untouched-test log-loss changed by
`{_finite_float(report["test_log_loss_delta"], "test delta"):.8f}`. Lower is
better; the test Brier-score change was
`{_finite_float(report["test_brier_delta"], "Brier delta"):.8f}`.

![Frozen profile grid](latest/charts/profile-model-selection.svg)

The preregistered reduced-profile grid selected
`{profile_model["inner_selected_candidate"]}` inside the same purged training
folds. Its outer gate selected `{profile_model["selected_candidate"]}`. Against
the unchanged round-3 control, untouched-test log loss changed by
`{_finite_float(profile_report["test_log_loss_delta_vs_control"], "profile test delta"):.8f}`
and test Brier score changed by
`{_finite_float(profile_report["test_brier_delta_vs_control"], "profile Brier delta"):.8f}`.
Even if all exploratory gates pass, this challenger remains non-selectable until
a later prospective capture confirms it.

![Time-group score deltas](latest/charts/held-out-group-scores.svg)

![Profile-vs-control time-group deltas](latest/charts/profile-held-out-group-scores.svg)

The untouched tail contains `{all_groups["time_group_count"]}` shared five-minute
time groups. Its deterministic moving-block bootstrap interval for mean log-loss
delta is
`[{_finite_float(all_groups["moving_block_bootstrap_95pct_lower"], "lower interval"):.8f},
{_finite_float(all_groups["moving_block_bootstrap_95pct_upper"], "upper interval"):.8f}]`.
Confirmatory status is **{"ready" if all_groups["confirmatory_ready"] else "not ready"}**;
the frozen minimum is 30 untouched groups and the entire interval must be below
zero. This interval is an exploratory dependence-aware diagnostic, not a p-value.

![Held-out settled equity](latest/charts/held-out-equity.svg)

The baseline replay filled `{baseline["filled_order_count"]}` orders and settled
`{baseline["net_realized_pnl_quote"]}` quote PnL. The residual-model replay filled
`{model["filled_order_count"]}` orders and settled
`{model["net_realized_pnl_quote"]}` quote PnL. The terminal-zero-fill retry
challenger filled `{model_retry["filled_order_count"]}` orders, settled
`{model_retry["net_realized_pnl_quote"]}` quote PnL, and was
**{"accepted" if retry_challenger["accepted"] else "not accepted"}** by its
precommitted gates. These are short prospective diagnostics after modeled
dynamic fees and recorded depth, not evidence of a durable edge. The profile
challenger filled `{profile_execution["filled_order_count"]}` orders, settled
`{profile_execution["net_realized_pnl_quote"]}` quote PnL, and its exploratory
gates **{"passed" if profile_challenger["promotion_gates_passed"] else "did not pass"}**.
It was not accepted for paper trading. {ai_line}

![After-fee PnL by asset](latest/charts/per-asset-execution.svg)

![Network-latency sensitivity](latest/charts/latency-sensitivity.svg)

## Evidence gates

| Gate | Result |
|---|---:|
{gate_rows}

## Reproduction data

- [Experiment artifact](round-{round_number:03d}-prospective-model-experiment.json)
- [Held-out prediction rows](latest/tables/held-out-predictions.csv)
- [Nested model selection](latest/tables/model-selection.csv)
- [Frozen profile selection](latest/tables/profile-model-selection.csv)
- [Probability metrics](latest/tables/probability-metrics.csv)
- [Held-out time-group scores](latest/tables/held-out-group-scores.csv)
- [Time-group uncertainty summary](latest/held-out-group-score-summary.json)
- [Execution summary](latest/tables/execution-summary.csv)
- [Settled equity curves](latest/tables/equity-curves.csv)
- [Execution ledger](latest/tables/trades.csv)
- [Per-asset execution](latest/tables/per-asset-execution.csv)
- [Latency sensitivity](latest/tables/latency-sensitivity.csv)
- [Immutable AI prompt cases](latest/tables/ai-prompt-cases.csv)
- [AI decisions](latest/tables/ai-decisions.csv)
- [Round progression](latest/tables/research-progress.csv)
- [Integrity manifest](latest/publication-integrity.json)
"""


def publish_polymarket_model_artifact(
    artifact_path: str | Path,
    research_root: str | Path,
    *,
    source_verification: Mapping[str, Any],
    round_number: int = 3,
    prior_round_path: str | Path | None = None,
) -> PolymarketPublicationResult:
    """Publish charts only after independent source reconstruction succeeds."""

    if round_number < 1:
        raise ValueError("publication round must be positive")
    source = Path(artifact_path).resolve()
    root = Path(research_root).resolve()
    validated = validate_polymarket_model_artifact(source)
    validated_source_verification = dict(
        validate_polymarket_source_verification(
            source_verification,
            artifact_sha256=validated.artifact_sha256,
            run_id=str(validated.payload["run_id"]),
        )
    )
    sensitivity = _as_mapping(
        validated.payload["execution_latency_sensitivity"],
        "execution latency sensitivity",
    )
    sensitivity_policies = _as_mapping(
        sensitivity["policies"],
        "execution latency policies",
    )
    expected_source_hashes = {
        str(policy): {
            str(latency): str(
                _as_mapping(
                    report,
                    f"{policy} {latency}ms execution",
                )["report_sha256"]
            )
            for latency, report in _as_mapping(
                reports,
                f"{policy} latency reports",
            ).items()
        }
        for policy, reports in sensitivity_policies.items()
    }
    expected_source_identities = {
        "feature_dataset_sha256": _as_mapping(
            validated.payload["feature_dataset"],
            "feature dataset",
        )["dataset_sha256"],
        "model_dataset_sha256": _as_mapping(
            validated.payload["model_dataset"],
            "model dataset",
        )["dataset_sha256"],
        "split_sha256": _as_mapping(
            validated.payload["split"],
            "model split",
        )["split_sha256"],
        "model_sha256": _as_mapping(
            validated.payload["model"],
            "model",
        )["model_sha256"],
        "probability_report_sha256": _as_mapping(
            validated.payload["probability_report"],
            "probability report",
        )["report_sha256"],
        "profile_model_sha256": _as_mapping(
            validated.payload["profile_model"],
            "profile model",
        )["model_sha256"],
        "profile_probability_report_sha256": _as_mapping(
            validated.payload["profile_probability_report"],
            "profile probability report",
        )["report_sha256"],
        "held_out_rows_sha256": _as_mapping(
            validated.payload["held_out_prediction_evidence"],
            "held-out prediction evidence",
        )["rows_sha256"],
        "profile_held_out_rows_sha256": _as_mapping(
            validated.payload["profile_held_out_prediction_evidence"],
            "profile held-out prediction evidence",
        )["rows_sha256"],
    }
    if (
        any(
            validated_source_verification.get(name) != value
            for name, value in expected_source_identities.items()
        )
        or validated_source_verification[
            "execution_report_sha256_by_policy_and_latency"
        ]
        != expected_source_hashes
    ):
        raise ValueError(
            "Polymarket source verification does not reconstruct every "
            "artifact execution scenario"
        )
    predictions = _prediction_rows(validated.profile_predictions)
    start = _utc(min(int(row["event_start_ms"]) for row in validated.predictions))
    end = _utc(max(int(row["end_ms"]) for row in validated.predictions))
    probability_rows = _probability_rows(validated.payload)
    model_selection_rows = _model_selection_rows(validated.payload)
    profile_model_selection_rows = _profile_model_selection_rows(validated.payload)
    group_score_rows = _held_out_group_score_rows(validated.profile_predictions)
    group_score_summary = _group_score_summary(
        group_score_rows,
        artifact_sha256=validated.artifact_sha256,
    )
    summary_rows = _execution_summary_rows(validated.executions)
    latency_rows = _latency_rows(validated.payload)
    equity_rows = _equity_rows(validated.executions)
    trade_rows = _trade_rows(validated.executions)
    per_asset_rows = _per_asset_rows(validated.predictions, validated.executions)
    ai_case_rows = _ai_case_rows(validated.payload)
    ai_rows = _ai_rows(validated.payload)
    prior = Path(prior_round_path).resolve() if prior_round_path is not None else None
    progress_rows = _progress_rows(validated, prior, round_number)

    latest = root / "latest"
    charts = latest / "charts"
    tables = latest / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    current_chart_names = {
        "model-selection.svg",
        "profile-model-selection.svg",
        "probability-quality.svg",
        "held-out-group-scores.svg",
        "profile-held-out-group-scores.svg",
        "held-out-equity.svg",
        "per-asset-execution.svg",
        "latency-sensitivity.svg",
        "research-progress.svg",
    }
    for old_chart in charts.glob("*.svg"):
        if old_chart.name not in current_chart_names:
            old_chart.unlink()

    source_name = f"round-{round_number:03d}-prospective-model-experiment.json"
    source_target = root / source_name
    if source != source_target:
        source_target.parent.mkdir(parents=True, exist_ok=True)
        temporary = source_target.with_name(f".{source_target.name}.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(source_target)

    table_payloads = {
        "held-out-predictions.csv": predictions,
        "model-selection.csv": model_selection_rows,
        "profile-model-selection.csv": profile_model_selection_rows,
        "probability-metrics.csv": probability_rows,
        "held-out-group-scores.csv": group_score_rows,
        "execution-summary.csv": summary_rows,
        "equity-curves.csv": equity_rows,
        "trades.csv": trade_rows,
        "per-asset-execution.csv": per_asset_rows,
        "latency-sensitivity.csv": latency_rows,
        "ai-prompt-cases.csv": ai_case_rows,
        "ai-decisions.csv": ai_rows,
        "research-progress.csv": progress_rows,
    }
    for name, rows in table_payloads.items():
        _write_csv(tables / name, rows)
    _write_json(latest / "held-out-group-score-summary.json", group_score_summary)
    source_verification_path = latest / "source-verification.json"
    _write_json(source_verification_path, validated_source_verification)
    _write_text(
        charts / "model-selection.svg",
        _model_selection_svg(model_selection_rows, start=start, end=end),
    )
    _write_text(
        charts / "profile-model-selection.svg",
        _profile_model_selection_svg(
            profile_model_selection_rows,
            start=start,
            end=end,
        ),
    )
    _write_text(
        charts / "probability-quality.svg",
        _probability_svg(probability_rows, start=start, end=end),
    )
    _write_text(
        charts / "held-out-group-scores.svg",
        _group_score_svg(group_score_rows, start=start, end=end),
    )
    _write_text(
        charts / "profile-held-out-group-scores.svg",
        _group_score_svg(
            group_score_rows,
            start=start,
            end=end,
            value_key="profile_vs_model_log_loss_delta",
            title="Profile-vs-control log-loss delta by time group",
            comparison="profile-challenger minus round-3-control",
            negative_label="Negative: profile challenger improved",
            positive_label="Positive: profile challenger degraded",
            negative_color=_COLORS["profile_model"],
        ),
    )
    _write_text(
        charts / "held-out-equity.svg", _equity_svg(equity_rows, start=start, end=end)
    )
    _write_text(
        charts / "per-asset-execution.svg",
        _asset_svg(per_asset_rows, start=start, end=end),
    )
    _write_text(
        charts / "latency-sensitivity.svg",
        _latency_svg(latency_rows, start=start, end=end),
    )
    _write_text(charts / "research-progress.svg", _progress_svg(progress_rows))

    results_name = f"round-{round_number:03d}-prospective-model-results.md"
    results = _results_markdown(
        validated,
        round_number,
        start,
        end,
        group_score_summary,
    )
    _write_text(root / results_name, results)
    source_verification_note = (
        "The complete source database was independently reconstructed before "
        "publication. Inspect the [source-verification report]"
        "(source-verification.json).\n\n"
    )
    latest_readme = f"""# Polymarket research round {round_number}

![Held-out settled equity](charts/held-out-equity.svg)

The current publication is generated from prospective BTC/ETH/SOL evidence for
`{start}` through `{end}`. It includes the market prior, unchanged round-3
control, frozen reduced-profile challenger, retry challenger, and governed
AI-veto diagnostics where available. The profile challenger cannot be selected
for paper trading from this exploratory capture. No live-trading or durable
profitability claim is made.

{source_verification_note}
[Read the measured results](../{results_name}) or inspect the
[integrity manifest](publication-integrity.json) and [source tables](tables/).

![Research progression](charts/research-progress.svg)
"""
    _write_text(latest / "README.md", latest_readme)

    generated = [
        source_target,
        root / results_name,
        latest / "README.md",
        latest / "held-out-group-score-summary.json",
        source_verification_path,
        *(tables / name for name in sorted(table_payloads)),
        *(charts / name for name in sorted(current_chart_names)),
    ]
    entries = []
    for path in generated:
        relative = path.relative_to(root).as_posix()
        entry: dict[str, object] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if path.suffix == ".csv":
            entry["row_count"] = len(table_payloads[path.name])
            entry["columns"] = list(table_payloads[path.name][0])
        entries.append(entry)
    manifest_body = {
        "schema_version": _PUBLICATION_SCHEMA,
        "round": round_number,
        "source_artifact": source_name,
        "source_artifact_sha256": validated.artifact_sha256,
        "held_out_start_utc": start,
        "held_out_end_utc": end,
        "assets": list(_ASSETS),
        "source_reconstruction_verified": True,
        "source_verification_report_sha256": validated_source_verification[
            "report_sha256"
        ],
        "generated_artifacts": entries,
        "claims": {
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        },
    }
    manifest_sha256 = _canonical_sha256(manifest_body)
    manifest = {**manifest_body, "manifest_sha256": manifest_sha256}
    _write_json(latest / "publication-integrity.json", manifest)
    return PolymarketPublicationResult(
        round_number=round_number,
        research_root=str(root),
        source_artifact=str(source_target),
        artifact_sha256=validated.artifact_sha256,
        generated_files=tuple(str(path) for path in generated),
        manifest_sha256=manifest_sha256,
    )


__all__ = [
    "POLYMARKET_MODEL_ARTIFACT_SCHEMA_VERSION",
    "PolymarketPublicationResult",
    "ValidatedPolymarketArtifact",
    "publish_polymarket_model_artifact",
    "validate_polymarket_model_artifact",
]
