"""Target-free Round 21 cases and delayed permissions for the local AI veto."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import math
import re
from typing import Mapping, Sequence

from .ai_runtime import estimate_model_parameters_b, ollama_residency_from_mapping
from .polymarket_ai_veto import (
    POLYMARKET_AI_CASE_SCHEMA_VERSION,
    POLYMARKET_AI_REPORT_SCHEMA_VERSION,
    PolymarketAIVetoCase,
    PolymarketAIVetoReport,
    PolymarketAIVetoResult,
)
from .polymarket_round21_core_features import POLYMARKET_ROUND21_FEATURE_SCHEMA
from .polymarket_round21_model import (
    Round21InferencePanel,
    Round21ProbabilityBatch,
)
from .polymarket_round21_replay import (
    Round21DirectionalPermission,
    Round21ReplayCondition,
)


POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256 = (
    "b84e49cbc382e4211a2e83f2b889c06506cd8ae477a7cdd35e1cd4258b0cce84"
)
POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION = "polymarket-round21-ai-veto-case-v1"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_CASE_KEYS = frozenset(
    {
        "future_book",
        "future_books",
        "future_pnl",
        "label",
        "matched_population_sha256",
        "outcome",
        "outcome_sha256",
        "pnl",
        "profit",
        "resolution",
        "resolved_up",
        "target",
    }
)
_AI_REASON_CODES = frozenset(
    {
        "cooldown_required",
        "edge_after_fees",
        "insufficient_evidence",
        "latency_risk",
        "liquidity_stress",
        "market_disagreement",
        "model_calibration_risk",
        "orderbook_imbalance",
        "source_staleness",
        "volatile_regime",
        "weak_probability_uplift",
    }
)
_CORE_AI_NAMES = (
    "core.elapsed_fraction",
    "core.remaining_seconds",
    "core.chainlink_log_distance_from_open",
    "core.chainlink_variance_rate_per_second",
    "core.chainlink_receipt_age_ms",
    "core.structural_probability_up",
    "core.up_best_bid",
    "core.up_best_ask",
    "core.up_relative_spread",
    "core.up_microprice",
    "core.up_top_quantity_imbalance",
    "core.up_bid_depth_5",
    "core.up_ask_depth_5",
    "core.down_best_bid",
    "core.down_best_ask",
    "core.down_relative_spread",
    "core.down_microprice",
    "core.down_top_quantity_imbalance",
    "core.down_bid_depth_5",
    "core.down_ask_depth_5",
    "core.normalized_market_prior_up",
    "core.structural_minus_market_prior",
    "core.complement_buy_overround",
    "core.complement_sell_underround",
    "core.book_receipt_skew_ms",
    *tuple(
        f"core.chainlink_{metric}_{window}ms"
        for window in (250, 1000, 5000, 15000, 60000, 120000)
        for metric in (
            "log_return",
            "realized_variance",
            "jump_fraction",
        )
    ),
    *tuple(
        f"core.{outcome}_{metric}_{window}ms"
        for outcome in ("up", "down")
        for window in (250, 1000, 5000, 15000)
        for metric in (
            "top_order_flow_imbalance",
            "mean_top_quantity_imbalance",
            "microprice_log_return",
            "level_quantity_flow_pressure",
        )
    ),
)
_VENUE_AI_WINDOWS = (250, 1000, 5000, 15000, 60000, 120000)
_SPOT_AI_NAMES = tuple(
    f"spot.{source}_{metric}_{window}ms"
    for source, metrics in (
        (
            "trade",
            ("log_return", "realized_variance", "signed_quote_imbalance"),
        ),
        (
            "book",
            (
                "mid_log_return",
                "mean_relative_spread",
                "mean_top_quantity_imbalance",
            ),
        ),
    )
    for window in _VENUE_AI_WINDOWS
    for metric in metrics
)
_USDM_AI_NAMES = (
    *tuple(
        f"usdm.{source}_{metric}_{window}ms"
        for source, metrics in (
            (
                "trade",
                ("log_return", "realized_variance", "signed_quote_imbalance"),
            ),
            (
                "book",
                (
                    "mid_log_return",
                    "mean_relative_spread",
                    "mean_top_quantity_imbalance",
                ),
            ),
        )
        for window in _VENUE_AI_WINDOWS
        for metric in metrics
    ),
    "usdm.log_mid_basis",
    "usdm.log_microprice_basis",
    *tuple(
        f"usdm.spot_minus_usdm_mid_log_return_{window}ms"
        for window in _VENUE_AI_WINDOWS
    ),
)


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


def round21_ai_case_source_evidence_sha256(
    *,
    condition_id: str,
    decision_time_ms: int,
    feature_batch_sha256: str,
    feature_row_sha256: str,
    probability_batch_sha256: str,
    model_artifact_sha256: str,
    causal_market_path_sha256: str,
) -> str:
    """Bind a case receipt to the exact target-free inputs available then."""

    condition = str(condition_id or "").strip().lower()
    if isinstance(decision_time_ms, bool):
        raise ValueError("Round 21 AI case source evidence is invalid")
    decision = int(decision_time_ms)
    digests = {
        "feature_batch_sha256": str(feature_batch_sha256 or "").strip().lower(),
        "feature_row_sha256": str(feature_row_sha256 or "").strip().lower(),
        "probability_batch_sha256": (
            str(probability_batch_sha256 or "").strip().lower()
        ),
        "model_artifact_sha256": str(model_artifact_sha256 or "").strip().lower(),
        "causal_market_path_sha256": (
            str(causal_market_path_sha256 or "").strip().lower()
        ),
    }
    if (
        not condition.startswith("0x")
        or len(condition) != 66
        or any(value not in "0123456789abcdef" for value in condition[2:])
        or decision <= 0
        or any(
            _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
            for value in digests.values()
        )
    ):
        raise ValueError("Round 21 AI case source evidence is invalid")
    return _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION,
            "ai_veto_design_sha256": POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
            "condition_id": condition,
            "decision_time_ms": decision,
            **digests,
            "target_accessed": False,
            "future_books_accessed": False,
            "outcome_accessed": False,
        }
    )


def _all_mapping_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_mapping_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.extend(_all_mapping_keys(item))
    return tuple(keys)


def _validated_round21_ai_case(
    case: PolymarketAIVetoCase,
) -> PolymarketAIVetoCase:
    payload = case.prompt_payload
    if not isinstance(payload, Mapping):
        raise ValueError("Round 21 AI veto case payload differs")
    identity = payload.get("identity")
    hard_constraints = payload.get("hard_constraints")
    probability = payload.get("probability")
    expected_top_level = {
        "ai_veto_design_sha256",
        "asset",
        "condition_id",
        "core",
        "decision_time_ms",
        "display_precision_significant_digits",
        "event_start_ms",
        "execution",
        "hard_constraints",
        "identity",
        "market",
        "missingness",
        "probability",
        "schema_version",
        "spot",
        "task",
        "usdm",
    }
    expected_identity = {
        "case_receipt_sha256",
        "causal_market_path_sha256",
        "feature_batch_sha256",
        "feature_row_sha256",
        "model_artifact_sha256",
        "probability_batch_sha256",
    }
    expected_constraints = {
        "cannot_block_reduction_lock_stop_close_or_recovery": True,
        "cannot_create_reverse_or_increase_risk": True,
        "invalid_unavailable_or_late_preserves_deterministic_policy": True,
        "no_outcome_resolution_future_book_or_future_pnl": True,
    }
    forbidden = _FORBIDDEN_CASE_KEYS.intersection(_all_mapping_keys(payload))
    decision_time = payload.get("decision_time_ms")
    if (
        set(payload) != expected_top_level
        or forbidden
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION
        or payload.get("ai_veto_design_sha256")
        != POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256
        or payload.get("task") != "condition_level_directional_entry_risk_veto"
        or payload.get("asset") != case.asset
        or case.asset != "BTC"
        or payload.get("market") != "polymarket_crypto_up_down_5m"
        or payload.get("condition_id") != case.condition_id
        or payload.get("event_start_ms") != case.event_start_ms
        or isinstance(decision_time, bool)
        or not isinstance(decision_time, int)
        or not decision_time <= case.decision_received_wall_ms <= decision_time + 250
        or case.decision_received_monotonic_ns <= 0
        or not isinstance(identity, Mapping)
        or set(identity) != expected_identity
        or not isinstance(probability, Mapping)
        or hard_constraints != expected_constraints
        or payload.get("display_precision_significant_digits") != 10
        or case.case_id != _canonical_sha256(payload)
        or case.case_sha256 != _canonical_sha256(case.identity_payload())
        or case.sample_id != identity.get("feature_row_sha256")
        or _SHA256.fullmatch(case.case_id) is None
        or _SHA256.fullmatch(case.case_sha256) is None
        or _SHA256.fullmatch(case.sample_id) is None
        or any(
            _SHA256.fullmatch(str(identity.get(name) or "")) is None
            or str(identity.get(name)) == _EMPTY_SHA256
            for name in expected_identity
        )
    ):
        raise ValueError("Round 21 AI veto case differs")
    return case


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError("Round 21 AI response contains duplicate keys")
        parsed[key] = value
    return parsed


def _parsed_provider_decision(
    payload: object,
) -> tuple[str, float, tuple[str, ...], str] | None:
    if not isinstance(payload, Mapping):
        return None
    message = payload.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        return None
    try:
        parsed = json.loads(
            str(message["content"]),
            object_pairs_hook=_strict_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, Mapping) or set(parsed) != {
        "action",
        "confidence",
        "reason_codes",
        "summary",
    }:
        return None
    action = parsed.get("action")
    confidence = parsed.get("confidence")
    reason_codes = parsed.get("reason_codes")
    summary = parsed.get("summary")
    if (
        not isinstance(action, str)
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not isinstance(reason_codes, list)
        or any(not isinstance(value, str) for value in reason_codes)
        or not isinstance(summary, str)
    ):
        return None
    confidence_float = float(confidence)
    selected_codes = tuple(reason_codes)
    selected_summary = summary.strip()
    if (
        action not in {"approve", "cooldown", "veto"}
        or not math.isfinite(confidence_float)
        or not 0.0 <= confidence_float <= 1.0
        or not 1 <= len(selected_codes) <= 4
        or len(set(selected_codes)) != len(selected_codes)
        or any(value not in _AI_REASON_CODES for value in selected_codes)
        or not selected_summary
        or len(selected_summary) > 180
        or action == "approve"
        and set(selected_codes) != {"edge_after_fees"}
        or action == "cooldown"
        and "cooldown_required" not in selected_codes
        or action == "veto"
        and not (set(selected_codes) - {"edge_after_fees"})
    ):
        return None
    return action, confidence_float, selected_codes, selected_summary


def _provider_usage(payload: object, *, model: str) -> Mapping[str, int] | None:
    if (
        not isinstance(payload, Mapping)
        or payload.get("model") != model
        or payload.get("done") is not True
        or payload.get("done_reason") != "stop"
    ):
        return None
    fields = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    if any(
        isinstance(payload.get(name), bool)
        or not isinstance(payload.get(name), int)
        for name in fields
    ):
        return None
    usage = {name: int(payload[name]) for name in fields}
    if (
        usage["total_duration"] <= 0
        or usage["load_duration"] < 0
        or usage["prompt_eval_count"] <= 0
        or usage["prompt_eval_duration"] <= 0
        or usage["eval_count"] <= 0
        or usage["eval_duration"] <= 0
        or usage["load_duration"] > usage["total_duration"]
        or usage["prompt_eval_duration"] > usage["total_duration"]
        or usage["eval_duration"] > usage["total_duration"]
    ):
        return None
    return usage


def _valid_decision_shape(
    result: PolymarketAIVetoResult,
    *,
    config_model: str,
    minimum_approval_confidence: float,
    maximum_advisory_latency_seconds: float,
) -> bool:
    decision = result.decision
    parsed = _parsed_provider_decision(result.response_payload)
    if decision.valid:
        if parsed is None:
            return False
        action, confidence, reason_codes, summary = parsed
        return (
            decision.action == action
            and decision.confidence == confidence
            and decision.reason_codes == reason_codes
            and decision.summary == summary
            and not decision.failure_reason
            and (
                action != "approve"
                or confidence >= minimum_approval_confidence
            )
            and result.latency_seconds <= maximum_advisory_latency_seconds
            and _provider_usage(result.response_payload, model=config_model)
            is not None
        )
    exact_failure = (
        decision.action == "veto"
        and decision.confidence == 0.0
        and decision.reason_codes == ("insufficient_evidence",)
        and bool(decision.failure_reason)
    )
    if not exact_failure:
        return False
    if parsed is None:
        return (
            isinstance(result.response_payload, Mapping)
            and str(result.response_payload.get("schema_version") or "").startswith(
                "polymarket-ai-veto-failure-v"
            )
        )
    action, confidence, _reason_codes, _summary = parsed
    return (
        action == "approve" and confidence < minimum_approval_confidence
    ) or result.latency_seconds > maximum_advisory_latency_seconds


def _same_float(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_round21_ai_report(
    *,
    cases: Sequence[PolymarketAIVetoCase],
    report: PolymarketAIVetoReport,
) -> str:
    config = report.config.validated()
    payload = report.asdict()
    claimed = str(payload.pop("report_sha256", "")).strip().lower()
    expected_parameters = estimate_model_parameters_b(config.model)
    if (
        report.schema_version != POLYMARKET_AI_REPORT_SCHEMA_VERSION
        or config != report.config
        or claimed != _canonical_sha256(payload)
        or _SHA256.fullmatch(claimed) is None
        or claimed == _EMPTY_SHA256
        or expected_parameters is None
        or expected_parameters < 2.0
        or not _same_float(report.model_parameters_b, expected_parameters)
        or any(
            _SHA256.fullmatch(str(value or "")) is None
            or str(value) == _EMPTY_SHA256
            for value in (
                report.model_digest,
                report.model_metadata_sha256,
                report.risk_benchmark_evidence_sha256,
                report.selection_sha256,
            )
        )
        or report.case_set_sha256
        != _case_set_sha256(
            cases,
            selection_sha256=report.selection_sha256,
        )
        or report.case_count != len(cases)
        or len(report.results) != len(cases)
        or not report.advisory_only
        or report.trading_authority
        or report.profitability_claim
    ):
        raise ValueError("Round 21 AI veto report differs")
    if tuple(result.case_id for result in report.results) != tuple(
        case.case_id for case in cases
    ):
        raise ValueError("Round 21 AI veto result order differs")

    worker_available_ns: int | None = None
    usages: list[Mapping[str, int]] = []
    for case, result in zip(cases, report.results, strict=True):
        inference = float(result.inference_latency_seconds)
        queue = float(result.queue_delay_seconds)
        latency = float(result.latency_seconds)
        if (
            result.condition_id != case.condition_id
            or result.model != config.model
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (inference, queue, latency)
            )
            or result.response_sha256
            != _canonical_sha256(result.response_payload)
            or _SHA256.fullmatch(result.response_sha256) is None
            or result.response_sha256 == _EMPTY_SHA256
        ):
            raise ValueError("Round 21 AI veto result differs")
        inference_ns = max(0, math.ceil(inference * 1_000_000_000))
        arrival_ns = case.decision_received_monotonic_ns
        service_start_ns = max(arrival_ns, worker_available_ns or arrival_ns)
        queue_ns = service_start_ns - arrival_ns
        expected_queue = queue_ns / 1_000_000_000.0
        expected_latency = (queue_ns + inference_ns) / 1_000_000_000.0
        worker_available_ns = service_start_ns + inference_ns
        if (
            not _same_float(queue, expected_queue)
            or not _same_float(latency, expected_latency)
            or not _valid_decision_shape(
                result,
                config_model=config.model,
                minimum_approval_confidence=config.minimum_approval_confidence,
                maximum_advisory_latency_seconds=(
                    config.maximum_advisory_latency_seconds
                ),
            )
        ):
            raise ValueError("Round 21 AI veto result evidence differs")
        usage = _provider_usage(result.response_payload, model=config.model)
        if usage is not None:
            usages.append(usage)
        if result.decision.valid:
            if result.provider_runtime is None:
                raise ValueError("Round 21 AI provider residency is unavailable")
            runtime = ollama_residency_from_mapping(result.provider_runtime)
            if (
                runtime.requested_model != config.model
                or runtime.digest != report.model_digest
                or not runtime.loaded
                or not runtime.fully_gpu_resident
            ):
                raise ValueError("Round 21 AI provider residency differs")

    valid_count = sum(result.decision.valid for result in report.results)
    permissions = {
        case.condition_id: (
            result.decision.valid and result.decision.action == "approve"
        )
        for case, result in zip(cases, report.results, strict=True)
    }
    permission_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-market-permission-v1",
            "permissions": dict(sorted(permissions.items())),
        }
    )
    inference_latencies = [
        float(result.inference_latency_seconds) for result in report.results
    ]
    queue_delays = [float(result.queue_delay_seconds) for result in report.results]
    latencies = [float(result.latency_seconds) for result in report.results]
    if (
        report.valid_response_count != valid_count
        or report.approval_count
        != sum(result.decision.action == "approve" for result in report.results)
        or report.veto_count
        != sum(result.decision.action == "veto" for result in report.results)
        or report.cooldown_count
        != sum(result.decision.action == "cooldown" for result in report.results)
        or report.provider_failure_count != len(report.results) - valid_count
        or report.provider_telemetry_count != len(usages)
        or report.total_prompt_token_count
        != sum(value["prompt_eval_count"] for value in usages)
        or report.total_output_token_count
        != sum(value["eval_count"] for value in usages)
        or report.maximum_prompt_token_count
        != max((value["prompt_eval_count"] for value in usages), default=0)
        or report.maximum_output_token_count
        != max((value["eval_count"] for value in usages), default=0)
        or dict(report.market_permissions) != dict(sorted(permissions.items()))
        or report.market_permission_sha256 != permission_sha256
        or not _same_float(
            report.average_inference_latency_seconds,
            sum(inference_latencies) / len(inference_latencies),
        )
        or not _same_float(
            report.maximum_inference_latency_seconds,
            max(inference_latencies),
        )
        or not _same_float(
            report.average_queue_delay_seconds,
            sum(queue_delays) / len(queue_delays),
        )
        or not _same_float(report.maximum_queue_delay_seconds, max(queue_delays))
        or not _same_float(
            report.average_latency_seconds,
            sum(latencies) / len(latencies),
        )
        or not _same_float(report.maximum_latency_seconds, max(latencies))
    ):
        raise ValueError("Round 21 AI veto report accounting differs")
    return claimed


@dataclass(frozen=True, slots=True)
class Round21AICaseReceipt:
    condition_id: str
    decision_time_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    source_evidence_sha256: str
    receipt_sha256: str

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        decision_time_ms: int,
        received_wall_ms: int,
        received_monotonic_ns: int,
        source_evidence_sha256: str,
    ) -> Round21AICaseReceipt:
        condition = str(condition_id or "").strip().lower()
        decision = int(decision_time_ms)
        wall = int(received_wall_ms)
        monotonic = int(received_monotonic_ns)
        source_sha = str(source_evidence_sha256 or "").strip().lower()
        if (
            not condition.startswith("0x")
            or len(condition) != 66
            or any(value not in "0123456789abcdef" for value in condition[2:])
            or decision <= 0
            or not decision <= wall <= decision + 250
            or monotonic <= 0
            or _SHA256.fullmatch(source_sha) is None
            or source_sha == _EMPTY_SHA256
        ):
            raise ValueError("Round 21 AI case receipt is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION,
            "ai_veto_design_sha256": POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
            "condition_id": condition,
            "decision_time_ms": decision,
            "received_wall_ms": wall,
            "received_monotonic_ns": monotonic,
            "source_evidence_sha256": source_sha,
        }
        return cls(
            condition_id=condition,
            decision_time_ms=decision,
            received_wall_ms=wall,
            received_monotonic_ns=monotonic,
            source_evidence_sha256=source_sha,
            receipt_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21AICaseReceipt:
        rebuilt = self.create(
            condition_id=self.condition_id,
            decision_time_ms=self.decision_time_ms,
            received_wall_ms=self.received_wall_ms,
            received_monotonic_ns=self.received_monotonic_ns,
            source_evidence_sha256=self.source_evidence_sha256,
        )
        if self != rebuilt:
            raise ValueError("Round 21 AI case receipt differs")
        return self


def _selected_features(
    names: Sequence[str],
    values: Sequence[float],
    selected_names: Sequence[str],
) -> dict[str, float]:
    feature_map = dict(zip(names, values, strict=True))
    if any(name not in feature_map for name in selected_names):
        raise ValueError("Round 21 AI feature contract differs")
    return {
        name.split(".", 1)[1]: float(format(feature_map[name], ".10g"))
        for name in selected_names
    }


def _case_set_sha256(
    cases: Sequence[PolymarketAIVetoCase],
    *,
    selection_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": POLYMARKET_AI_CASE_SCHEMA_VERSION,
            "selection_sha256": selection_sha256,
            "case_sha256": [case.case_sha256 for case in cases],
        }
    )


def build_round21_ai_veto_cases(
    *,
    conditions: Sequence[Round21ReplayCondition],
    panel: Round21InferencePanel,
    probability_batch: Round21ProbabilityBatch,
    case_receipts: Sequence[Round21AICaseReceipt],
) -> tuple[PolymarketAIVetoCase, ...]:
    """Build one target-free condition review from exact immutable model inputs."""

    selected_conditions = tuple(value.validated() for value in conditions)
    selected_panel = panel.validate()
    selected_batch = probability_batch.validated()
    receipts = tuple(value.validated() for value in case_receipts)
    receipt_map = {value.condition_id: value for value in receipts}
    schema = POLYMARKET_ROUND21_FEATURE_SCHEMA.validated()
    if (
        not selected_conditions
        or selected_batch.feature_batch_sha256
        != selected_panel.feature_batch_sha256
        or selected_panel.core_feature_names_sha256 != schema.core_names_sha256
        or selected_panel.spot_feature_names_sha256 != schema.spot_names_sha256
        or selected_panel.usdm_feature_names_sha256 != schema.usdm_names_sha256
        or len(receipt_map) != len(receipts)
        or set(receipt_map)
        != {value.market.condition_id for value in selected_conditions}
    ):
        raise ValueError("Round 21 AI case population differs")
    batch_positions = {
        int(panel_index): position
        for position, panel_index in enumerate(selected_batch.indices.tolist())
    }
    cases: list[PolymarketAIVetoCase] = []
    for condition in selected_conditions:
        first_envelope = condition.envelopes[0]
        receipt = receipt_map[condition.market.condition_id]
        matches = [
            index
            for index, (condition_id, decision_time) in enumerate(
                zip(
                    selected_panel.condition_ids,
                    selected_panel.decision_time_ms,
                    strict=True,
                )
            )
            if str(condition_id) == condition.market.condition_id
            and int(decision_time) == first_envelope.decision_time_ms
            and index in batch_positions
        ]
        if len(matches) != 1:
            raise ValueError("Round 21 AI case row is unavailable")
        index = matches[0]
        row_probability, row_lower, row_upper = selected_batch.row(index)
        causal_market_path_sha256 = condition.causal_market_path_sha256(
            decision_time_ms=first_envelope.decision_time_ms,
        )
        expected_receipt_source_sha256 = (
            round21_ai_case_source_evidence_sha256(
                condition_id=condition.market.condition_id,
                decision_time_ms=first_envelope.decision_time_ms,
                feature_batch_sha256=selected_panel.feature_batch_sha256,
                feature_row_sha256=selected_panel.row_sha256(index),
                probability_batch_sha256=selected_batch.prediction_sha256,
                model_artifact_sha256=(
                    selected_batch.source_model_artifact_sha256
                ),
                causal_market_path_sha256=causal_market_path_sha256,
            )
        )
        if (
            first_envelope.model_layer != selected_batch.population_layer
            or first_envelope.source_model_artifact_sha256
            != selected_batch.source_model_artifact_sha256
            or first_envelope.source_probability_batch_sha256
            != selected_batch.prediction_sha256
            or first_envelope.feature_row_sha256
            != selected_panel.row_sha256(index)
            or Decimal(format(row_probability, ".17g"))
            != first_envelope.probability_up
            or Decimal(format(row_lower, ".17g")) != first_envelope.lower_up
            or Decimal(format(row_upper, ".17g")) != first_envelope.upper_up
            or receipt.decision_time_ms != first_envelope.decision_time_ms
            or receipt.source_evidence_sha256
            != expected_receipt_source_sha256
        ):
            raise ValueError("Round 21 AI probability evidence differs")
        up_book = condition.creation_book(
            outcome="Up",
            decision_time_ms=first_envelope.decision_time_ms,
        )
        down_book = condition.creation_book(
            outcome="Down",
            decision_time_ms=first_envelope.decision_time_ms,
        )
        if up_book is None or down_book is None:
            raise ValueError("Round 21 AI causal book evidence is unavailable")
        payload: dict[str, object] = {
            "schema_version": POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION,
            "ai_veto_design_sha256": POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
            "task": "condition_level_directional_entry_risk_veto",
            "asset": "BTC",
            "market": "polymarket_crypto_up_down_5m",
            "condition_id": condition.market.condition_id,
            "event_start_ms": condition.market.event_start_ms,
            "decision_time_ms": int(selected_panel.decision_time_ms[index]),
            "probability": {
                "up_point": format(first_envelope.probability_up, "f"),
                "up_lower": format(first_envelope.lower_up, "f"),
                "up_upper": format(first_envelope.upper_up, "f"),
                "candidate_disagreement_width": format(
                    first_envelope.upper_up - first_envelope.lower_up,
                    "f",
                ),
                "market_prior_up": format(
                    float(selected_panel.market_prior_probability[index]),
                    ".10g",
                ),
                "structural_up": format(
                    float(selected_panel.structural_probability[index]),
                    ".10g",
                ),
                "model_layer": first_envelope.model_layer,
            },
            "core": _selected_features(
                schema.core_names,
                selected_panel.core_features[index],
                _CORE_AI_NAMES,
            ),
            "spot": (
                _selected_features(
                    schema.spot_names,
                    selected_panel.spot_features[index],
                    _SPOT_AI_NAMES,
                )
                if bool(selected_panel.spot_available[index])
                else None
            ),
            "usdm": (
                _selected_features(
                    schema.usdm_names,
                    selected_panel.usdm_features[index],
                    _USDM_AI_NAMES,
                )
                if bool(selected_panel.usdm_available[index])
                else None
            ),
            "missingness": {
                "spot_available": bool(selected_panel.spot_available[index]),
                "usdm_available": bool(selected_panel.usdm_available[index]),
            },
            "execution": {
                "fee_rate": format(condition.market.fee_schedule.rate, "f"),
                "fee_exponent": condition.market.fee_schedule.exponent,
                "tick_size": format(condition.market.tick_size, "f"),
                "minimum_order_size": format(
                    condition.market.minimum_order_size,
                    "f",
                ),
                "creation_book_maximum_age_ms": 500,
                "taker_order_delay_enabled": (
                    condition.market_evidence.taker_order_delay_enabled
                ),
            },
            "identity": {
                "model_artifact_sha256": (
                    selected_batch.source_model_artifact_sha256
                ),
                "probability_batch_sha256": selected_batch.prediction_sha256,
                "feature_batch_sha256": selected_panel.feature_batch_sha256,
                "feature_row_sha256": selected_panel.row_sha256(index),
                "causal_market_path_sha256": causal_market_path_sha256,
                "case_receipt_sha256": receipt.receipt_sha256,
            },
            "hard_constraints": {
                "cannot_create_reverse_or_increase_risk": True,
                "cannot_block_reduction_lock_stop_close_or_recovery": True,
                "no_outcome_resolution_future_book_or_future_pnl": True,
                "invalid_unavailable_or_late_preserves_deterministic_policy": True,
            },
            "display_precision_significant_digits": 10,
        }
        case_id = _canonical_sha256(payload)
        case = PolymarketAIVetoCase(
            case_id=case_id,
            condition_id=condition.market.condition_id,
            sample_id=selected_panel.row_sha256(index),
            asset="BTC",
            event_start_ms=condition.market.event_start_ms,
            decision_received_wall_ms=receipt.received_wall_ms,
            decision_received_monotonic_ns=receipt.received_monotonic_ns,
            prompt_payload=payload,
            case_sha256="",
        )
        cases.append(
            replace(
                case,
                case_sha256=_canonical_sha256(case.identity_payload()),
            )
        )
    ordered = tuple(
        sorted(
            cases,
            key=lambda value: (
                value.decision_received_monotonic_ns,
                value.decision_received_wall_ms,
                value.condition_id,
            ),
        )
    )
    if len({value.condition_id for value in ordered}) != len(ordered):
        raise ValueError("Round 21 AI cases contain duplicate conditions")
    return ordered


def round21_permissions_from_ai_report(
    *,
    cases: Sequence[PolymarketAIVetoCase],
    report: PolymarketAIVetoReport,
) -> tuple[Round21DirectionalPermission, ...]:
    """Translate valid timely AI results into delayed veto-only permissions."""

    selected_cases = tuple(_validated_round21_ai_case(value) for value in cases)
    if not selected_cases:
        raise ValueError("Round 21 AI veto cases are empty")
    claimed = _validate_round21_ai_report(cases=selected_cases, report=report)
    case_map = {value.case_id: value for value in selected_cases}
    if len(case_map) != len(selected_cases):
        raise ValueError("Round 21 AI veto cases differ")
    permissions: list[Round21DirectionalPermission] = []
    seen_conditions: set[str] = set()
    for result in report.results:
        case = case_map.get(result.case_id)
        latency = float(result.latency_seconds)
        decision = result.decision
        if (
            case is None
            or result.condition_id != case.condition_id
            or result.model != report.config.model
            or not math.isfinite(latency)
            or latency < 0.0
            or result.condition_id in seen_conditions
            or _SHA256.fullmatch(result.response_sha256) is None
            or result.response_sha256 == _EMPTY_SHA256
            or decision.action not in {"approve", "veto", "cooldown"}
            or type(decision.valid) is not bool
        ):
            raise ValueError("Round 21 AI veto result differs")
        seen_conditions.add(result.condition_id)
        allowed = not decision.valid or decision.action == "approve"
        effective_at_ms = (
            case.decision_received_wall_ms + math.ceil(latency * 1_000.0)
        )
        source_sha = _canonical_sha256(
            {
                "schema_version": POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION,
                "ai_veto_design_sha256": (
                    POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256
                ),
                "report_sha256": claimed,
                "case_sha256": case.case_sha256,
                "response_sha256": result.response_sha256,
                "model": result.model,
                "model_digest": report.model_digest,
                "latency_seconds": format(latency, ".17g"),
                "decision": decision.asdict(),
                "directional_entry_allowed": allowed,
            }
        )
        permissions.append(
            Round21DirectionalPermission.create(
                condition_id=result.condition_id,
                effective_at_ms=effective_at_ms,
                directional_entry_allowed=allowed,
                source_evidence_sha256=source_sha,
            )
        )
    if seen_conditions != {value.condition_id for value in selected_cases}:
        raise ValueError("Round 21 AI veto result population differs")
    return tuple(sorted(permissions, key=lambda value: value.condition_id))


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_AI_CASE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256",
    "Round21AICaseReceipt",
    "build_round21_ai_veto_cases",
    "round21_ai_case_source_evidence_sha256",
    "round21_permissions_from_ai_report",
]
