"""Frozen prospective contract for the BTC five-minute Polymarket edge study."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping


POLYMARKET_ROUND14_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round14-btc-5m-prospective-contract-v1"
)
POLYMARKET_ROUND14_CAPTURE_DURATION_SECONDS = 2_592_000
POLYMARKET_ROUND14_DECISION_CADENCE_MS = 250
_MAX_CONTRACT_BYTES = 256 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "round",
        "status",
        "scope",
        "predecessor",
        "hypotheses",
        "resolution_target",
        "prospective_capture",
        "partition",
        "causal_features",
        "model_candidates",
        "ai_review",
        "actions",
        "risk_profiles",
        "execution_scenarios",
        "evaluation",
        "promotion",
        "research_basis",
        "freshness",
        "contract_sha256",
    }
)
_CONTROL_MODELS = (
    "raw_executable_polymarket_prior",
    "chainlink_structural_endpoint_probability",
    "polymarket_prior_monotone_calibration",
)
_ENDPOINT_MODELS = (
    "elastic_net_logistic_residual",
    "shallow_monotone_lightgbm_residual",
    "causal_multiscale_tcn_endpoint",
)
_ACTION_MODELS = (
    "distributional_executable_action_value_lightgbm",
    "causal_multitask_tcn_action_value",
)
_MANDATORY_SCENARIOS = (
    ("latency_250ms", 250, 0, Decimal("1"), Decimal("1")),
    ("latency_750ms", 750, 0, Decimal("1"), Decimal("1")),
    ("latency_1000ms", 1000, 0, Decimal("1"), Decimal("1")),
    ("half_depth", 500, 0, Decimal("0.5"), Decimal("1")),
    ("quarter_depth", 500, 0, Decimal("0.25"), Decimal("1")),
    ("one_tick_adverse", 500, 1, Decimal("1"), Decimal("1")),
    ("combined", 1000, 2, Decimal("0.5"), Decimal("2")),
)
_PROFILE_LIMITS = {
    "conservative": (
        Decimal("0.001"),
        Decimal("0.005"),
        Decimal("0.02"),
        30,
    ),
    "regular": (
        Decimal("0.002"),
        Decimal("0.01"),
        Decimal("0.04"),
        15,
    ),
    "aggressive": (
        Decimal("0.0035"),
        Decimal("0.015"),
        Decimal("0.06"),
        5,
    ),
}


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 14 contract contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 14 contract contains {value}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 14 {name} must be an object")
    return value


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 14 {name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 14 {name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"Round 14 {name} must be a finite decimal")
    return parsed


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _exact_strings(value: object, expected: tuple[str, ...], *, name: str) -> None:
    if not isinstance(value, list) or tuple(value) != expected:
        raise ValueError(f"Round 14 {name} differs from the frozen contract")


@dataclass(frozen=True, slots=True)
class PolymarketRound14ExecutionScenario:
    name: str
    submission_latency_ms: int
    adverse_ticks: int
    displayed_depth_fraction: Decimal
    fee_multiplier: Decimal


@dataclass(frozen=True, slots=True)
class PolymarketRound14RiskProfile:
    name: str
    maximum_event_loss_capital_fraction: Decimal
    maximum_daily_loss_capital_fraction: Decimal
    maximum_drawdown_capital_fraction: Decimal
    cooldown_minutes_after_loss_cluster: int


@dataclass(frozen=True, slots=True)
class PolymarketRound14Program:
    contract_sha256: str
    capture_duration_seconds: int
    minimum_complete_markets: int
    minimum_untouched_test_markets: int
    controls: tuple[str, ...]
    endpoint_models: tuple[str, ...]
    action_models: tuple[str, ...]
    scenarios: tuple[PolymarketRound14ExecutionScenario, ...]
    risk_profiles: tuple[PolymarketRound14RiskProfile, ...]
    paper_authority: bool = False
    live_authority: bool = False


def _scenario(
    value: object,
    *,
    expected: tuple[str, int, int, Decimal, Decimal],
) -> PolymarketRound14ExecutionScenario:
    payload = _mapping(value, name="execution scenario")
    if set(payload) != {
        "name",
        "submission_latency_ms",
        "adverse_ticks",
        "displayed_depth_fraction",
        "fee_multiplier",
    }:
        raise ValueError("Round 14 execution scenario schema is invalid")
    scenario = PolymarketRound14ExecutionScenario(
        name=str(payload["name"]),
        submission_latency_ms=int(payload["submission_latency_ms"]),
        adverse_ticks=int(payload["adverse_ticks"]),
        displayed_depth_fraction=_decimal(
            payload["displayed_depth_fraction"],
            name="displayed depth fraction",
        ),
        fee_multiplier=_decimal(
            payload["fee_multiplier"],
            name="fee multiplier",
        ),
    )
    if (
        scenario.name,
        scenario.submission_latency_ms,
        scenario.adverse_ticks,
        scenario.displayed_depth_fraction,
        scenario.fee_multiplier,
    ) != expected:
        raise ValueError("Round 14 execution scenario differs")
    return scenario


def _risk_profiles(value: object) -> tuple[PolymarketRound14RiskProfile, ...]:
    payload = _mapping(value, name="risk profiles")
    if set(payload) != {"default", "reinvestment_default", *_PROFILE_LIMITS, "global"}:
        raise ValueError("Round 14 risk profile schema is invalid")
    if payload["default"] != "conservative" or payload["reinvestment_default"] is not False:
        raise ValueError("Round 14 risk defaults differ")
    profiles: list[PolymarketRound14RiskProfile] = []
    for name, expected in _PROFILE_LIMITS.items():
        raw = _mapping(payload[name], name=f"{name} risk profile")
        if set(raw) != {
            "maximum_event_loss_capital_fraction",
            "maximum_daily_loss_capital_fraction",
            "maximum_drawdown_capital_fraction",
            "uncertainty_action",
            "cooldown_minutes_after_loss_cluster",
        }:
            raise ValueError(f"Round 14 {name} risk profile schema is invalid")
        profile = PolymarketRound14RiskProfile(
            name=name,
            maximum_event_loss_capital_fraction=_decimal(
                raw["maximum_event_loss_capital_fraction"],
                name=f"{name} event loss",
            ),
            maximum_daily_loss_capital_fraction=_decimal(
                raw["maximum_daily_loss_capital_fraction"],
                name=f"{name} daily loss",
            ),
            maximum_drawdown_capital_fraction=_decimal(
                raw["maximum_drawdown_capital_fraction"],
                name=f"{name} drawdown",
            ),
            cooldown_minutes_after_loss_cluster=int(
                raw["cooldown_minutes_after_loss_cluster"]
            ),
        )
        if (
            profile.maximum_event_loss_capital_fraction,
            profile.maximum_daily_loss_capital_fraction,
            profile.maximum_drawdown_capital_fraction,
            profile.cooldown_minutes_after_loss_cluster,
        ) != expected:
            raise ValueError(f"Round 14 {name} risk limits differ")
        profiles.append(profile)
    return tuple(profiles)


def validate_round14_contract(
    value: Mapping[str, object],
) -> PolymarketRound14Program:
    contract = dict(value)
    if set(contract) != _EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError("Round 14 contract schema is invalid")
    claimed_hash = str(contract["contract_sha256"] or "").strip().lower()
    if _SHA256.fullmatch(claimed_hash) is None:
        raise ValueError("Round 14 contract hash is invalid")
    body = dict(contract)
    body.pop("contract_sha256")
    if _canonical_sha256(body) != claimed_hash:
        raise ValueError("Round 14 contract hash differs")
    if (
        contract["schema_version"] != POLYMARKET_ROUND14_CONTRACT_SCHEMA_VERSION
        or contract["round"] != 14
        or contract["status"] != "preregistered_not_started"
    ):
        raise ValueError("Round 14 contract identity differs")
    scope = _mapping(contract["scope"], name="scope")
    if (
        scope.get("venue") != "polymarket"
        or scope.get("protocol_version") != 2
        or scope.get("asset") != "BTC"
        or scope.get("market_variant") != "fiveminute"
        or scope.get("environment") != "research"
    ):
        raise ValueError("Round 14 scope differs")
    predecessor = _mapping(contract["predecessor"], name="predecessor")
    if (
        predecessor.get("round13_status") != "failed_before_outcome_access"
        or predecessor.get("reuse_allowed") is not False
        or predecessor.get("pooling_allowed") is not False
        or predecessor.get("round11_status") != "rejected"
    ):
        raise ValueError("Round 14 predecessor exclusion differs")
    target = _mapping(contract["resolution_target"], name="resolution target")
    if (
        target.get("primary_endpoint")
        != "https://polymarket.com/api/crypto/crypto-price"
        or target.get("endpoint_status")
        != "first_party_frontend_endpoint_not_publicly_documented"
    ):
        raise ValueError("Round 14 resolution source differs")
    _exact_strings(
        target.get("primary_response_fields"),
        (
            "openPrice",
            "closePrice",
            "timestamp",
            "completed",
            "incomplete",
            "cached",
        ),
        name="resolution fields",
    )
    capture = _mapping(contract["prospective_capture"], name="capture")
    if (
        capture.get("duration_seconds")
        != POLYMARKET_ROUND14_CAPTURE_DURATION_SECONDS
        or capture.get("calendar_days") != 30
        or capture.get("decision_cadence_ms")
        != POLYMARKET_ROUND14_DECISION_CADENCE_MS
        or capture.get("capture_unit_minutes") != 30
        or capture.get("event_duration_seconds") != 300
        or capture.get("minimum_complete_markets") != 7000
        or capture.get("minimum_untouched_test_markets") != 1800
        or capture.get("minimum_test_calendar_days") != 7
    ):
        raise ValueError("Round 14 prospective capture differs")
    partition = _mapping(contract["partition"], name="partition")
    if (
        partition.get("method") != "contiguous_event_start_time"
        or partition.get("train_days") != 14
        or partition.get("train_tune_embargo_seconds") != 3600
        or partition.get("tune_days") != 7
        or partition.get("tune_test_embargo_seconds") != 3600
        or partition.get("test_days_minimum") != 7
        or partition.get("test_access")
        != "one_use_after_immutable_pretest_manifest"
    ):
        raise ValueError("Round 14 partition differs")
    models = _mapping(contract["model_candidates"], name="model candidates")
    _exact_strings(models.get("controls"), _CONTROL_MODELS, name="control models")
    _exact_strings(
        models.get("endpoint_models"),
        _ENDPOINT_MODELS,
        name="endpoint models",
    )
    _exact_strings(
        models.get("action_models"),
        _ACTION_MODELS,
        name="action models",
    )
    ai = _mapping(contract["ai_review"], name="AI review")
    if (
        ai.get("default_enabled_in_product") is not True
        or ai.get("research_authority") != "veto_or_reduce_only"
        or ai.get("direct_entry_authority") is not False
        or ai.get("candidate_models") != ["Fino1-8B", "Qwen3-8B"]
        or ai.get("cadence_seconds") != 5
    ):
        raise ValueError("Round 14 AI boundary differs")
    execution = _mapping(contract["execution_scenarios"], name="execution scenarios")
    primary = execution.get("primary")
    primary_scenario = _scenario(
        {"name": "primary", **dict(_mapping(primary, name="primary scenario"))},
        expected=("primary", 500, 0, Decimal("1"), Decimal("1")),
    )
    raw_scenarios = execution.get("mandatory_stress")
    if not isinstance(raw_scenarios, list) or len(raw_scenarios) != len(
        _MANDATORY_SCENARIOS
    ):
        raise ValueError("Round 14 mandatory scenario count differs")
    scenarios = (primary_scenario,) + tuple(
        _scenario(raw, expected=expected)
        for raw, expected in zip(raw_scenarios, _MANDATORY_SCENARIOS, strict=True)
    )
    evaluation = _mapping(contract["evaluation"], name="evaluation")
    power = _mapping(evaluation.get("minimum_power"), name="minimum power")
    if (
        power.get("resolved_test_markets") != 1800
        or power.get("non_tied_endpoint_predictions") != 300
        or power.get("executed_test_actions") != 300
        or power.get("executed_test_calendar_days") != 7
        or power.get("both_outcomes") is not True
    ):
        raise ValueError("Round 14 minimum power differs")
    promotion = _mapping(contract["promotion"], name="promotion")
    if (
        promotion.get("paper_authority") is not False
        or promotion.get("live_authority") is not False
        or promotion.get("profitability_claim") is not False
        or promotion.get("ai_edge_claim") is not False
        or promotion.get("automatic_promotion") is not False
        or promotion.get("required_schema") != "polymarket-live-promotion-v1"
    ):
        raise ValueError("Round 14 authority differs")
    return PolymarketRound14Program(
        contract_sha256=claimed_hash,
        capture_duration_seconds=int(capture["duration_seconds"]),
        minimum_complete_markets=int(capture["minimum_complete_markets"]),
        minimum_untouched_test_markets=int(
            capture["minimum_untouched_test_markets"]
        ),
        controls=_CONTROL_MODELS,
        endpoint_models=_ENDPOINT_MODELS,
        action_models=_ACTION_MODELS,
        scenarios=scenarios,
        risk_profiles=_risk_profiles(contract["risk_profiles"]),
    )


def load_round14_contract(path: str | Path) -> PolymarketRound14Program:
    contract_path = Path(path)
    if contract_path.is_symlink():
        raise ValueError("Round 14 contract cannot be a symlink")
    raw = contract_path.read_bytes()
    if not raw or len(raw) > _MAX_CONTRACT_BYTES:
        raise ValueError("Round 14 contract size is invalid")
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 14 contract is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 14 contract must be an object")
    return validate_round14_contract(payload)


__all__ = [
    "POLYMARKET_ROUND14_CAPTURE_DURATION_SECONDS",
    "POLYMARKET_ROUND14_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND14_DECISION_CADENCE_MS",
    "PolymarketRound14ExecutionScenario",
    "PolymarketRound14Program",
    "PolymarketRound14RiskProfile",
    "load_round14_contract",
    "validate_round14_contract",
]
