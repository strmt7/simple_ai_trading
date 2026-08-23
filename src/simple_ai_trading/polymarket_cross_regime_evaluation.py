"""Semantic cross-regime evidence required before Polymarket promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from .polymarket_live_risk import polymarket_live_risk_profile


POLYMARKET_CROSS_REGIME_EVALUATION_SCHEMA_VERSION = (
    "polymarket-cross-regime-evaluation-v1"
)
CROSS_REGIME_EDGE_ACCEPTANCE_CONTRACT_SHA256 = (
    "06068c3e5e8003dce4d3e5b47391a64c987941e6aa25b841f9e1079dcb6ecf23"
)
POLYMARKET_REQUIRED_REGIME_SLICES = (
    "direction:bullish",
    "direction:bearish",
    "direction:sideways",
    "path:directional",
    "path:choppy",
    "volatility:low_or_normal_volatility",
    "volatility:high_volatility",
    "volatility:stress_volatility",
    "liquidity:normal_liquidity",
    "liquidity:stressed_spread",
    "liquidity:stressed_depth",
    "liquidity:stale_or_missing_book",
    "execution:nominal_latency",
    "execution:elevated_latency",
    "execution:severe_latency",
    "execution:no_fill_or_unknown_state",
)
_MANDATORY_ABSTENTION_SLICES = frozenset(
    {
        "liquidity:stale_or_missing_book",
        "execution:no_fill_or_unknown_state",
    }
)
_MARKET_VARIANTS = frozenset({"fiveminute", "fifteenminute"})
_MAX_EVALUATION_BYTES = 512 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("cross-regime evaluation JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"cross-regime evaluation JSON contains {value}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed != value or parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return parsed


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class PolymarketRegimeSliceEvidence:
    """One target-blind regime or execution-stress evaluation slice."""

    slice_id: str
    classification_count: int
    decision_count: int
    policy_action: str
    opened_position_count: int
    closed_trade_count: int
    after_cost_net_pnl_quote: Decimal
    paired_delta_lower_bound_quote: Decimal
    maximum_drawdown_capital_fraction: Decimal
    expected_shortfall_capital_fraction: Decimal
    unknown_execution_state_count: int
    untracked_inventory_count: int
    abstention_replay_passed: bool
    passed: bool

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCrossRegimeAggregate:
    """Portfolio-level result across the exact required evaluation slices."""

    classification_count: int
    decision_count: int
    opened_position_count: int
    closed_trade_count: int
    after_cost_net_pnl_quote: Decimal
    block_bootstrap_lower_bound_quote: Decimal
    maximum_drawdown_capital_fraction: Decimal
    expected_shortfall_capital_fraction: Decimal
    maximum_profit_concentration_fraction: Decimal
    unknown_execution_state_count: int
    untracked_inventory_count: int
    passed: bool

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolymarketCrossRegimeEvaluation:
    """Strict model-bound evidence with no trading authority by itself."""

    evaluation_id: str
    evaluation_sha256: str
    created_at_ms: int
    source_commit: str
    market_variant: str
    risk_profile: str
    model_artifact_sha256: str
    data_manifest_sha256: str
    cost_model_sha256: str
    regime_definition_sha256: str
    selection_policy_sha256: str
    train_role_sha256: str
    tune_role_sha256: str
    test_role_sha256: str
    minimum_classifications_per_slice: int
    minimum_closed_trades_per_traded_slice: int
    bootstrap_confidence: Decimal
    maximum_profit_concentration_fraction: Decimal
    slices: tuple[PolymarketRegimeSliceEvidence, ...]
    aggregate: PolymarketCrossRegimeAggregate

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _validate_role_evidence(value: object) -> tuple[str, str, str]:
    role = _mapping(value, name="cross-regime role evidence")
    if set(role) != {
        "train_sha256",
        "tune_sha256",
        "test_sha256",
        "roles_nonoverlapping",
        "test_sealed_before_selection",
        "selection_frozen_before_test",
    }:
        raise ValueError("cross-regime role evidence schema is invalid")
    hashes = (
        _sha(role["train_sha256"], name="training role"),
        _sha(role["tune_sha256"], name="tuning role"),
        _sha(role["test_sha256"], name="test role"),
    )
    if len(set(hashes)) != 3 or not all(
        _boolean(role[name], name=name)
        for name in (
            "roles_nonoverlapping",
            "test_sealed_before_selection",
            "selection_frozen_before_test",
        )
    ):
        raise ValueError("cross-regime train/tune/test isolation failed")
    return hashes


def _validate_thresholds(
    value: object,
) -> tuple[int, int, Decimal, Decimal]:
    thresholds = _mapping(value, name="cross-regime thresholds")
    if set(thresholds) != {
        "minimum_classifications_per_slice",
        "minimum_closed_trades_per_traded_slice",
        "bootstrap_confidence",
        "maximum_profit_concentration_fraction",
    }:
        raise ValueError("cross-regime threshold schema is invalid")
    minimum_classifications = _integer(
        thresholds["minimum_classifications_per_slice"],
        name="minimum classifications per slice",
        minimum=30,
    )
    minimum_trades = _integer(
        thresholds["minimum_closed_trades_per_traded_slice"],
        name="minimum closed trades per traded slice",
        minimum=30,
    )
    confidence = _decimal(
        thresholds["bootstrap_confidence"],
        name="bootstrap confidence",
    )
    concentration = _decimal(
        thresholds["maximum_profit_concentration_fraction"],
        name="maximum profit concentration",
    )
    if not Decimal("0.95") <= confidence <= Decimal("0.999"):
        raise ValueError("bootstrap confidence is outside promotion bounds")
    if not Decimal("0") < concentration <= Decimal("0.60"):
        raise ValueError("profit concentration threshold is outside promotion bounds")
    return minimum_classifications, minimum_trades, confidence, concentration


def _validate_slice(
    value: object,
    *,
    minimum_classifications: int,
    minimum_trades: int,
    maximum_drawdown: Decimal,
    maximum_expected_shortfall: Decimal,
) -> PolymarketRegimeSliceEvidence:
    raw = _mapping(value, name="cross-regime slice")
    if set(raw) != {
        "slice_id",
        "classification_count",
        "decision_count",
        "policy_action",
        "opened_position_count",
        "closed_trade_count",
        "after_cost_net_pnl_quote",
        "paired_delta_lower_bound_quote",
        "maximum_drawdown_capital_fraction",
        "expected_shortfall_capital_fraction",
        "unknown_execution_state_count",
        "untracked_inventory_count",
        "abstention_replay_passed",
        "passed",
    }:
        raise ValueError("cross-regime slice schema is invalid")
    slice_id = str(raw["slice_id"] or "").strip().lower()
    if slice_id not in POLYMARKET_REQUIRED_REGIME_SLICES:
        raise ValueError("cross-regime slice identifier is invalid")
    classifications = _integer(
        raw["classification_count"],
        name=f"{slice_id} classification count",
        minimum=minimum_classifications,
    )
    decisions = _integer(
        raw["decision_count"],
        name=f"{slice_id} decision count",
        minimum=minimum_classifications,
    )
    openings = _integer(
        raw["opened_position_count"],
        name=f"{slice_id} opening count",
    )
    trades = _integer(
        raw["closed_trade_count"],
        name=f"{slice_id} closed trade count",
    )
    unknown = _integer(
        raw["unknown_execution_state_count"],
        name=f"{slice_id} unknown execution state count",
    )
    untracked = _integer(
        raw["untracked_inventory_count"],
        name=f"{slice_id} untracked inventory count",
    )
    pnl = _decimal(
        raw["after_cost_net_pnl_quote"],
        name=f"{slice_id} after-cost net P&L",
    )
    lower_bound = _decimal(
        raw["paired_delta_lower_bound_quote"],
        name=f"{slice_id} paired lower bound",
    )
    drawdown = _decimal(
        raw["maximum_drawdown_capital_fraction"],
        name=f"{slice_id} maximum drawdown",
    )
    expected_shortfall = _decimal(
        raw["expected_shortfall_capital_fraction"],
        name=f"{slice_id} expected shortfall",
    )
    action = str(raw["policy_action"] or "").strip().lower()
    abstention_passed = _boolean(
        raw["abstention_replay_passed"],
        name=f"{slice_id} abstention replay",
    )
    declared_passed = _boolean(raw["passed"], name=f"{slice_id} passed")
    common_passed = (
        decisions == classifications
        and unknown == 0
        and untracked == 0
        and Decimal("0") <= drawdown <= maximum_drawdown
        and Decimal("0") <= expected_shortfall <= maximum_expected_shortfall
    )
    if action == "trade":
        semantic_passed = (
            common_passed
            and slice_id not in _MANDATORY_ABSTENTION_SLICES
            and openings == trades
            and trades >= minimum_trades
            and pnl >= Decimal("0")
            and lower_bound >= Decimal("0")
            and abstention_passed is False
        )
    elif action == "abstain":
        semantic_passed = (
            common_passed
            and openings == 0
            and trades == 0
            and pnl == Decimal("0")
            and lower_bound == Decimal("0")
            and drawdown == Decimal("0")
            and expected_shortfall == Decimal("0")
            and abstention_passed is True
        )
    else:
        raise ValueError("cross-regime policy action is invalid")
    if declared_passed is not semantic_passed or not semantic_passed:
        raise ValueError(f"cross-regime slice failed: {slice_id}")
    return PolymarketRegimeSliceEvidence(
        slice_id=slice_id,
        classification_count=classifications,
        decision_count=decisions,
        policy_action=action,
        opened_position_count=openings,
        closed_trade_count=trades,
        after_cost_net_pnl_quote=pnl,
        paired_delta_lower_bound_quote=lower_bound,
        maximum_drawdown_capital_fraction=drawdown,
        expected_shortfall_capital_fraction=expected_shortfall,
        unknown_execution_state_count=unknown,
        untracked_inventory_count=untracked,
        abstention_replay_passed=abstention_passed,
        passed=True,
    )


def _validate_aggregate(
    value: object,
    *,
    slices: tuple[PolymarketRegimeSliceEvidence, ...],
    maximum_drawdown: Decimal,
    maximum_expected_shortfall: Decimal,
    maximum_concentration: Decimal,
) -> PolymarketCrossRegimeAggregate:
    raw = _mapping(value, name="cross-regime aggregate")
    if set(raw) != {
        "classification_count",
        "decision_count",
        "opened_position_count",
        "closed_trade_count",
        "after_cost_net_pnl_quote",
        "block_bootstrap_lower_bound_quote",
        "maximum_drawdown_capital_fraction",
        "expected_shortfall_capital_fraction",
        "maximum_profit_concentration_fraction",
        "unknown_execution_state_count",
        "untracked_inventory_count",
        "passed",
    }:
        raise ValueError("cross-regime aggregate schema is invalid")
    classifications = _integer(
        raw["classification_count"],
        name="aggregate classification count",
    )
    decisions = _integer(raw["decision_count"], name="aggregate decision count")
    openings = _integer(
        raw["opened_position_count"],
        name="aggregate opening count",
    )
    trades = _integer(raw["closed_trade_count"], name="aggregate trade count")
    unknown = _integer(
        raw["unknown_execution_state_count"],
        name="aggregate unknown execution state count",
    )
    untracked = _integer(
        raw["untracked_inventory_count"],
        name="aggregate untracked inventory count",
    )
    pnl = _decimal(raw["after_cost_net_pnl_quote"], name="aggregate net P&L")
    lower_bound = _decimal(
        raw["block_bootstrap_lower_bound_quote"],
        name="aggregate block bootstrap lower bound",
    )
    drawdown = _decimal(
        raw["maximum_drawdown_capital_fraction"],
        name="aggregate maximum drawdown",
    )
    expected_shortfall = _decimal(
        raw["expected_shortfall_capital_fraction"],
        name="aggregate expected shortfall",
    )
    concentration = _decimal(
        raw["maximum_profit_concentration_fraction"],
        name="aggregate profit concentration",
    )
    declared_passed = _boolean(raw["passed"], name="aggregate passed")
    positive_pnls = tuple(
        item.after_cost_net_pnl_quote
        for item in slices
        if item.after_cost_net_pnl_quote > Decimal("0")
    )
    computed_pnl = sum(
        (item.after_cost_net_pnl_quote for item in slices),
        Decimal("0"),
    )
    computed_concentration = (
        max(positive_pnls) / sum(positive_pnls, Decimal("0"))
        if positive_pnls
        else Decimal("1")
    )
    semantic_passed = (
        classifications == sum(item.classification_count for item in slices)
        and decisions == sum(item.decision_count for item in slices)
        and openings == sum(item.opened_position_count for item in slices)
        and trades == sum(item.closed_trade_count for item in slices)
        and pnl == computed_pnl
        and pnl > Decimal("0")
        and lower_bound > Decimal("0")
        and Decimal("0") <= drawdown <= maximum_drawdown
        and Decimal("0") <= expected_shortfall <= maximum_expected_shortfall
        and concentration == computed_concentration
        and concentration <= maximum_concentration
        and unknown == 0
        and untracked == 0
        and all(item.passed for item in slices)
    )
    if declared_passed is not semantic_passed or not semantic_passed:
        raise ValueError("cross-regime aggregate failed")
    return PolymarketCrossRegimeAggregate(
        classification_count=classifications,
        decision_count=decisions,
        opened_position_count=openings,
        closed_trade_count=trades,
        after_cost_net_pnl_quote=pnl,
        block_bootstrap_lower_bound_quote=lower_bound,
        maximum_drawdown_capital_fraction=drawdown,
        expected_shortfall_capital_fraction=expected_shortfall,
        maximum_profit_concentration_fraction=concentration,
        unknown_execution_state_count=unknown,
        untracked_inventory_count=untracked,
        passed=True,
    )


def validate_polymarket_cross_regime_evaluation(
    value: Mapping[str, object],
    *,
    expected_model_artifact_sha256: str,
    expected_source_commit: str,
    expected_market_variant: str,
    expected_risk_profile: str,
) -> PolymarketCrossRegimeEvaluation:
    """Validate exact model, role, cost, risk, and regime evidence."""

    payload = dict(value)
    if set(payload) != {
        "schema_version",
        "evaluation_id",
        "evaluation_sha256",
        "created_at_ms",
        "source_commit",
        "venue",
        "asset",
        "market_variant",
        "risk_profile",
        "model_artifact_sha256",
        "cross_regime_contract_sha256",
        "data_manifest_sha256",
        "cost_model_sha256",
        "regime_definition_sha256",
        "selection_policy_sha256",
        "role_evidence",
        "thresholds",
        "slices",
        "aggregate",
        "authority",
    }:
        raise ValueError("cross-regime evaluation schema is invalid")
    market_variant = str(payload["market_variant"] or "").strip().lower()
    risk_profile = str(payload["risk_profile"] or "").strip().lower()
    source_commit = str(payload["source_commit"] or "").strip().lower()
    if (
        payload["schema_version"] != POLYMARKET_CROSS_REGIME_EVALUATION_SCHEMA_VERSION
        or payload["venue"] != "polymarket"
        or payload["asset"] != "BTC"
        or market_variant not in _MARKET_VARIANTS
        or market_variant != str(expected_market_variant).strip().lower()
        or source_commit != str(expected_source_commit).strip().lower()
        or _GIT_COMMIT.fullmatch(source_commit) is None
        or risk_profile != str(expected_risk_profile).strip().lower()
    ):
        raise ValueError("cross-regime evaluation scope differs")
    profile = polymarket_live_risk_profile(risk_profile)
    model_sha = _sha(payload["model_artifact_sha256"], name="evaluated model")
    if model_sha != _sha(
        expected_model_artifact_sha256,
        name="expected evaluated model",
    ):
        raise ValueError("cross-regime evaluation model hash differs")
    if (
        _sha(
            payload["cross_regime_contract_sha256"],
            name="cross-regime contract",
        )
        != CROSS_REGIME_EDGE_ACCEPTANCE_CONTRACT_SHA256
    ):
        raise ValueError("cross-regime contract hash differs")
    evidence_hashes = tuple(
        _sha(payload[name], name=name)
        for name in (
            "data_manifest_sha256",
            "cost_model_sha256",
            "regime_definition_sha256",
            "selection_policy_sha256",
        )
    )
    train_sha, tune_sha, test_sha = _validate_role_evidence(payload["role_evidence"])
    minimum_classifications, minimum_trades, confidence, max_concentration = (
        _validate_thresholds(payload["thresholds"])
    )
    raw_slices = payload["slices"]
    if not isinstance(raw_slices, list):
        raise ValueError("cross-regime slices must be a list")
    slices = tuple(
        _validate_slice(
            item,
            minimum_classifications=minimum_classifications,
            minimum_trades=minimum_trades,
            maximum_drawdown=profile.maximum_drawdown_capital_fraction,
            maximum_expected_shortfall=(profile.maximum_daily_loss_capital_fraction),
        )
        for item in raw_slices
    )
    if tuple(item.slice_id for item in slices) != POLYMARKET_REQUIRED_REGIME_SLICES:
        raise ValueError(
            "required cross-regime slices are missing, duplicated, or reordered"
        )
    aggregate = _validate_aggregate(
        payload["aggregate"],
        slices=slices,
        maximum_drawdown=profile.maximum_drawdown_capital_fraction,
        maximum_expected_shortfall=profile.maximum_daily_loss_capital_fraction,
        maximum_concentration=max_concentration,
    )
    authority = _mapping(payload["authority"], name="evaluation authority")
    if dict(authority) != {
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }:
        raise ValueError("cross-regime evaluation cannot grant trading authority")
    created_at_ms = _integer(
        payload["created_at_ms"],
        name="evaluation creation time",
        minimum=1,
    )
    evaluation_id = _sha(payload["evaluation_id"], name="evaluation identifier")
    claimed = _sha(payload["evaluation_sha256"], name="evaluation hash")
    body = dict(payload)
    body.pop("evaluation_sha256")
    if _canonical_sha256(body) != claimed:
        raise ValueError("cross-regime evaluation hash differs")
    return PolymarketCrossRegimeEvaluation(
        evaluation_id=evaluation_id,
        evaluation_sha256=claimed,
        created_at_ms=created_at_ms,
        source_commit=source_commit,
        market_variant=market_variant,
        risk_profile=risk_profile,
        model_artifact_sha256=model_sha,
        data_manifest_sha256=evidence_hashes[0],
        cost_model_sha256=evidence_hashes[1],
        regime_definition_sha256=evidence_hashes[2],
        selection_policy_sha256=evidence_hashes[3],
        train_role_sha256=train_sha,
        tune_role_sha256=tune_sha,
        test_role_sha256=test_sha,
        minimum_classifications_per_slice=minimum_classifications,
        minimum_closed_trades_per_traded_slice=minimum_trades,
        bootstrap_confidence=confidence,
        maximum_profit_concentration_fraction=max_concentration,
        slices=slices,
        aggregate=aggregate,
    )


def load_polymarket_cross_regime_evaluation(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_model_artifact_sha256: str,
    expected_source_commit: str,
    expected_market_variant: str,
    expected_risk_profile: str,
) -> PolymarketCrossRegimeEvaluation:
    """Load strict, bounded JSON and validate every promotion-bearing field."""

    evaluation_path = Path(path)
    if evaluation_path.is_symlink():
        raise ValueError("cross-regime evaluation cannot be a symlink")
    raw = evaluation_path.read_bytes()
    if not raw or len(raw) > _MAX_EVALUATION_BYTES:
        raise ValueError("cross-regime evaluation file size is invalid")
    if hashlib.sha256(raw).hexdigest() != _sha(
        expected_file_sha256,
        name="expected evaluation file",
    ):
        raise ValueError("cross-regime evaluation file hash differs")
    try:
        payload = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cross-regime evaluation is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("cross-regime evaluation is not an object")
    return validate_polymarket_cross_regime_evaluation(
        payload,
        expected_model_artifact_sha256=expected_model_artifact_sha256,
        expected_source_commit=expected_source_commit,
        expected_market_variant=expected_market_variant,
        expected_risk_profile=expected_risk_profile,
    )


def polymarket_cross_regime_evaluation_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical body hash used by evaluation publishers."""

    body = dict(value)
    body.pop("evaluation_sha256", None)
    return _canonical_sha256(body)


__all__ = [
    "CROSS_REGIME_EDGE_ACCEPTANCE_CONTRACT_SHA256",
    "POLYMARKET_CROSS_REGIME_EVALUATION_SCHEMA_VERSION",
    "POLYMARKET_REQUIRED_REGIME_SLICES",
    "PolymarketCrossRegimeAggregate",
    "PolymarketCrossRegimeEvaluation",
    "PolymarketRegimeSliceEvidence",
    "load_polymarket_cross_regime_evaluation",
    "polymarket_cross_regime_evaluation_sha256",
    "validate_polymarket_cross_regime_evaluation",
]
