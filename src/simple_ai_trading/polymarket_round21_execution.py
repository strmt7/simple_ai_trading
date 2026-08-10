"""Independent delayed Polymarket execution mechanics for Round 21."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

from .paper_execution import (
    BookLevel,
    PaperBookSnapshot,
    PaperOrderIntent,
    PolymarketFeeModel,
    simulate_aggressive_order,
)
from .polymarket import (
    POLYMARKET_TAKER_ORDER_DELAY_MS,
    PolymarketFiveMinuteMarket,
)
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_POLICY_SHA256,
)
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
)


POLYMARKET_ROUND21_EXECUTION_POLICY_SCHEMA_VERSION = (
    "polymarket-round21-executable-action-policy-v3"
)
POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256 = (
    "979788e5fa16a9d9c3aa3f3639286e4da659e074c29fd8019f6e3e5977dc51e0"
)
POLYMARKET_ROUND21_EXECUTION_SCHEMA_VERSION = (
    "polymarket-round21-delayed-fak-execution-v1"
)
POLYMARKET_ROUND21_MAXIMUM_EXECUTION_OBSERVATION_LATENESS_MS = 500
POLYMARKET_ROUND21_SHARE_QUANTUM = Decimal("0.000001")
POLYMARKET_ROUND21_PLATFORM_FEE_QUANTUM = Decimal("0.00001")
POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM = Decimal("0.000001")
POLYMARKET_ROUND21_MAXIMUM_BUILDER_TAKER_FEE_BPS = Decimal("100")
_SUBMISSION_LATENCIES_MS = (250, 500, 1_000)
_DISPLAYED_DEPTH_FRACTIONS = (
    Decimal("1"),
    Decimal("0.5"),
    Decimal("0.25"),
)
_ADVERSE_TICKS = (0, 1, 2)
_PRIMARY_SCENARIO_IDENTITY = (500, Decimal("1"), 0)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_INVENTORY_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAX_POLICY_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 21 execution policy has duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 execution policy contains {value}")


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


def _decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 {name} is not a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 21 {name} is not a finite decimal") from exc
    if not selected.is_finite() or (positive and selected <= 0):
        raise ValueError(f"Round 21 {name} is not a valid decimal")
    return selected


def _sha256(value: object, *, name: str, allow_empty: bool = False) -> str:
    selected = str(value or "").strip().lower()
    if (
        _SHA256.fullmatch(selected) is None
        or not allow_empty
        and selected == _EMPTY_SHA256
    ):
        raise ValueError(f"Round 21 {name} identity is invalid")
    return selected


def _ceil_quote(value: Decimal, quantum: Decimal) -> Decimal:
    if value < 0:
        raise ValueError("Round 21 fee cannot be negative")
    if value == 0:
        return Decimal("0")
    return value.quantize(quantum, rounding=ROUND_CEILING)


def _maximum_executable_price_levels(
    *,
    side: str,
    limit_price: Decimal,
    tick_size: Decimal,
) -> int:
    span = limit_price if side == "BUY" else Decimal("1") - limit_price
    levels = int((span / tick_size).to_integral_value(rounding=ROUND_DOWN))
    if levels <= 0:
        raise ValueError("Round 21 executable price lattice is invalid")
    return levels


def _is_tick_aligned(value: Decimal, tick_size: Decimal) -> bool:
    return value / tick_size == (value / tick_size).to_integral_value()


@dataclass(frozen=True, slots=True)
class Round21ExecutionScenario:
    name: str
    submission_latency_ms: int
    displayed_depth_fraction: Decimal
    adverse_ticks: int

    def validated(self) -> Round21ExecutionScenario:
        name = str(self.name or "").strip().lower()
        latency = int(self.submission_latency_ms)
        fraction = _decimal(
            self.displayed_depth_fraction,
            name="displayed depth fraction",
            positive=True,
        )
        adverse = int(self.adverse_ticks)
        identity = (latency, fraction, adverse)
        expected_name = (
            "primary"
            if identity == _PRIMARY_SCENARIO_IDENTITY
            else (
                f"latency_{latency}ms_depth_"
                f"{int(fraction * 100)}pct_adverse_{adverse}ticks"
            )
        )
        if (
            latency not in _SUBMISSION_LATENCIES_MS
            or fraction not in _DISPLAYED_DEPTH_FRACTIONS
            or adverse not in _ADVERSE_TICKS
            or name != expected_name
        ):
            raise ValueError("Round 21 execution scenario differs")
        return replace(
            self,
            name=name,
            submission_latency_ms=latency,
            displayed_depth_fraction=fraction,
            adverse_ticks=adverse,
        )


def _create_scenarios() -> tuple[Round21ExecutionScenario, ...]:
    output: list[Round21ExecutionScenario] = []
    for latency in _SUBMISSION_LATENCIES_MS:
        for fraction in _DISPLAYED_DEPTH_FRACTIONS:
            for adverse in _ADVERSE_TICKS:
                identity = (latency, fraction, adverse)
                name = (
                    "primary"
                    if identity == _PRIMARY_SCENARIO_IDENTITY
                    else (
                        f"latency_{latency}ms_depth_"
                        f"{int(fraction * 100)}pct_adverse_{adverse}ticks"
                    )
                )
                output.append(
                    Round21ExecutionScenario(
                        name=name,
                        submission_latency_ms=latency,
                        displayed_depth_fraction=fraction,
                        adverse_ticks=adverse,
                    ).validated()
                )
    selected = tuple(output)
    if len(selected) != 27 or len({item.name for item in selected}) != 27:
        raise RuntimeError("Round 21 execution scenario matrix differs")
    return selected


POLYMARKET_ROUND21_EXECUTION_SCENARIOS = _create_scenarios()


def round21_execution_scenario(name: str) -> Round21ExecutionScenario:
    selected_name = str(name or "").strip().lower()
    matches = tuple(
        item
        for item in POLYMARKET_ROUND21_EXECUTION_SCENARIOS
        if item.name == selected_name
    )
    if len(matches) != 1:
        raise ValueError("Round 21 execution scenario is unknown")
    return matches[0]


def validate_round21_execution_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Reject rehashed changes to latency, fills, fees, or independence."""

    policy = dict(value)
    claimed = str(policy.pop("design_sha256", "")).strip().lower()
    parents = policy.get("parents")
    protocol = policy.get("protocol_basis")
    scenarios = policy.get("scenarios")
    timing = policy.get("timing")
    order_book = policy.get("order_and_book")
    fees = policy.get("fees")
    unknown = policy.get("unknown_state")
    independence = policy.get("independence")
    authority = policy.get("authority")
    supersession = policy.get("supersession")
    if (
        set(policy)
        != {
            "schema_version",
            "round",
            "status",
            "parents",
            "protocol_basis",
            "scenarios",
            "timing",
            "order_and_book",
            "fees",
            "unknown_state",
            "independence",
            "authority",
            "supersedes",
            "supersession",
        }
        or claimed != POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256
        or claimed != _canonical_sha256(policy)
        or policy.get("schema_version")
        != POLYMARKET_ROUND21_EXECUTION_POLICY_SCHEMA_VERSION
        or policy.get("round") != 21
        or policy.get("status") != "preregistered_during_target_and_model_blind_capture"
        or policy.get("supersedes") != "polymarket-round21-executable-action-policy-v2"
        or parents
        != {
            "round21_contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
            "round21_dataset_design_sha256": (POLYMARKET_ROUND21_DATASET_DESIGN_SHA256),
            "round21_feature_policy_sha256": (POLYMARKET_ROUND21_FEATURE_POLICY_SHA256),
        }
        or not isinstance(protocol, Mapping)
        or protocol.get("protocol") != "polymarket_clob_v2"
        or protocol.get("public_market_data_only_for_replay") is not True
        or protocol.get("credentials") is not False
        or protocol.get("account_access") is not False
        or protocol.get("orders_submitted") is not False
        or not isinstance(scenarios, Mapping)
        or scenarios.get("construction") != "full_cartesian_product"
        or scenarios.get("primary")
        != {
            "submission_latency_ms": 500,
            "displayed_depth_fraction": "1",
            "adverse_ticks": 0,
        }
        or scenarios.get("submission_latency_ms") != list(_SUBMISSION_LATENCIES_MS)
        or scenarios.get("displayed_depth_fraction")
        != [format(value, "f") for value in _DISPLAYED_DEPTH_FRACTIONS]
        or scenarios.get("adverse_ticks") != list(_ADVERSE_TICKS)
        or scenarios.get("scenario_count") != 27
        or scenarios.get("all_scenarios_required_for_economic_gate") is not True
        or not isinstance(timing, Mapping)
        or timing.get("decision_plan_fixed_before_execution_book") is not True
        or timing.get("taker_order_delay_ms_when_enabled")
        != POLYMARKET_TAKER_ORDER_DELAY_MS
        or timing.get("maximum_execution_observation_lateness_ms")
        != POLYMARKET_ROUND21_MAXIMUM_EXECUTION_OBSERVATION_LATENESS_MS
        or timing.get("source_timestamps") != "audit_only"
        or timing.get("receipt_time") != "execution_authority"
        or timing.get("future_execution_book_used_for_decision") is not False
        or not isinstance(order_book, Mapping)
        or order_book.get("order_type") != "FAK"
        or order_book.get("partial_fills") is not True
        or order_book.get("hidden_liquidity_credit") is not False
        or order_book.get("queue_fill_inference") is not False
        or order_book.get("share_quantum")
        != format(POLYMARKET_ROUND21_SHARE_QUANTUM, "f")
        or not isinstance(fees, Mapping)
        or fees.get("maker_rebate_credit") is not False
        or fees.get("fee_free_assumption_without_captured_rule") is not False
        or fees.get("fee_calculated_per_fill_level") is not True
        or not isinstance(unknown, Mapping)
        or unknown.get("missing_execution_book_after_submit")
        != "maximum_bound_loss_and_block_new_exposure"
        or unknown.get("connection_or_gap_unproven")
        != "maximum_bound_loss_and_block_new_exposure"
        or unknown.get("malformed_or_wrong_market_evidence")
        != "reject_condition_not_no_fill"
        or not isinstance(independence, Mapping)
        or independence.get("execution_venue") != "polymarket_only"
        or independence.get("binance_execution") is not False
        or independence.get("binance_account") is not False
        or independence.get("binance_risk_or_stop_dependency") is not False
        or supersession
        != {
            "round21_executable_action_policy_v2_sha256": (
                "fce75edb69b20dfe593337036a95299312262ef468bc9df5d811ccbbc99017f7"
            ),
            "change": ("bind_unchanged_execution_matrix_to_receipt_age_features"),
            "scenario_matrix_changed": False,
            "fees_or_latency_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
        }
        or authority
        != {
            "model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 21 executable action policy differs")
    return {**policy, "design_sha256": claimed}


def load_round21_execution_policy(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAX_POLICY_BYTES
    ):
        raise ValueError("Round 21 executable action policy is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 executable action policy is unavailable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 executable action policy is not an object")
    return validate_round21_execution_policy(value)


@dataclass(frozen=True, slots=True)
class Round21MarketExecutionEvidence:
    condition_id: str
    observed_wall_ms: int
    observed_monotonic_ns: int
    maker_base_fee: int
    taker_base_fee: int
    taker_order_delay_enabled: bool
    general_order_delay_seconds: int
    minimum_order_age_seconds: int
    clob_info_sha256: str
    up_fee_rate_sha256: str
    down_fee_rate_sha256: str
    snapshot_sha256: str
    evidence_sha256: str
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        observed_wall_ms: int,
        observed_monotonic_ns: int,
        maker_base_fee: int,
        taker_base_fee: int,
        taker_order_delay_enabled: bool,
        general_order_delay_seconds: int,
        minimum_order_age_seconds: int,
        clob_info_sha256: str,
        up_fee_rate_sha256: str,
        down_fee_rate_sha256: str,
        snapshot_sha256: str,
    ) -> Round21MarketExecutionEvidence:
        condition = str(condition_id or "").strip().lower()
        observed = int(observed_wall_ms)
        monotonic = int(observed_monotonic_ns)
        maker_fee = int(maker_base_fee)
        taker_fee = int(taker_base_fee)
        general_delay = int(general_order_delay_seconds)
        minimum_age = int(minimum_order_age_seconds)
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or min(
                maker_fee,
                taker_fee,
                general_delay,
                minimum_age,
            )
            < 0
            or observed <= 0
            or monotonic <= 0
            or type(taker_order_delay_enabled) is not bool
        ):
            raise ValueError("Round 21 market execution evidence is invalid")
        hashes = {
            "clob_info_sha256": _sha256(
                clob_info_sha256,
                name="CLOB market information",
            ),
            "up_fee_rate_sha256": _sha256(
                up_fee_rate_sha256,
                name="Up fee-rate",
            ),
            "down_fee_rate_sha256": _sha256(
                down_fee_rate_sha256,
                name="Down fee-rate",
            ),
            "snapshot_sha256": _sha256(
                snapshot_sha256,
                name="market snapshot",
            ),
        }
        payload = {
            "schema_version": POLYMARKET_ROUND21_EXECUTION_SCHEMA_VERSION,
            "execution_policy_sha256": (POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256),
            "condition_id": condition,
            "observed_wall_ms": observed,
            "observed_monotonic_ns": monotonic,
            "maker_base_fee": maker_fee,
            "taker_base_fee": taker_fee,
            "taker_order_delay_enabled": taker_order_delay_enabled,
            "general_order_delay_seconds": general_delay,
            "minimum_order_age_seconds": minimum_age,
            **hashes,
            "trading_authority": False,
        }
        return cls(
            condition_id=condition,
            observed_wall_ms=observed,
            observed_monotonic_ns=monotonic,
            maker_base_fee=maker_fee,
            taker_base_fee=taker_fee,
            taker_order_delay_enabled=taker_order_delay_enabled,
            general_order_delay_seconds=general_delay,
            minimum_order_age_seconds=minimum_age,
            clob_info_sha256=hashes["clob_info_sha256"],
            up_fee_rate_sha256=hashes["up_fee_rate_sha256"],
            down_fee_rate_sha256=hashes["down_fee_rate_sha256"],
            snapshot_sha256=hashes["snapshot_sha256"],
            evidence_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21MarketExecutionEvidence:
        rebuilt = self.create(
            condition_id=self.condition_id,
            observed_wall_ms=self.observed_wall_ms,
            observed_monotonic_ns=self.observed_monotonic_ns,
            maker_base_fee=self.maker_base_fee,
            taker_base_fee=self.taker_base_fee,
            taker_order_delay_enabled=self.taker_order_delay_enabled,
            general_order_delay_seconds=self.general_order_delay_seconds,
            minimum_order_age_seconds=self.minimum_order_age_seconds,
            clob_info_sha256=self.clob_info_sha256,
            up_fee_rate_sha256=self.up_fee_rate_sha256,
            down_fee_rate_sha256=self.down_fee_rate_sha256,
            snapshot_sha256=self.snapshot_sha256,
        )
        if self != rebuilt or self.trading_authority:
            raise ValueError("Round 21 market execution evidence differs")
        return self

    @property
    def taker_order_delay_ms(self) -> int:
        return POLYMARKET_TAKER_ORDER_DELAY_MS if self.taker_order_delay_enabled else 0


@dataclass(frozen=True, slots=True)
class Round21AggressiveOrderPlan:
    condition_id: str
    token_id: str
    event_end_ms: int
    decision_time_ms: int
    effective_execution_target_ms: int
    outcome: str
    side: str
    quantity: Decimal
    limit_price: Decimal
    tick_size: Decimal
    minimum_order_size: Decimal
    scenario: Round21ExecutionScenario
    fee_enabled: bool
    fee_rate: Decimal
    fee_exponent: int
    fee_taker_only: bool
    builder_taker_fee_bps: Decimal
    maximum_loss_quote: Decimal
    parent_inventory_id: str
    predictor_evidence_sha256: str
    reconciliation_sha256: str
    market_execution_evidence_sha256: str
    plan_sha256: str
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        market: PolymarketFiveMinuteMarket,
        market_evidence: Round21MarketExecutionEvidence,
        scenario_name: str,
        decision_time_ms: int,
        outcome: str,
        side: str,
        quantity: Decimal,
        limit_price: Decimal,
        effective_tick_size: Decimal,
        builder_taker_fee_bps: Decimal,
        owned_cost_basis_quote: Decimal = Decimal("0"),
        parent_inventory_id: str = "",
        predictor_evidence_sha256: str,
        reconciliation_sha256: str,
    ) -> Round21AggressiveOrderPlan:
        evidence = market_evidence.validated()
        scenario = round21_execution_scenario(scenario_name)
        decision = int(decision_time_ms)
        selected_outcome = str(outcome or "").strip().title()
        selected_side = str(side or "").strip().upper()
        selected_quantity = _decimal(quantity, name="quantity", positive=True)
        selected_limit = _decimal(limit_price, name="limit price", positive=True)
        tick = _decimal(
            effective_tick_size,
            name="effective tick size",
            positive=True,
        )
        minimum_order = _decimal(
            market.minimum_order_size,
            name="minimum order size",
            positive=True,
        )
        builder_fee_bps = _decimal(
            builder_taker_fee_bps,
            name="builder taker fee",
        )
        owned_cost = _decimal(owned_cost_basis_quote, name="owned cost basis")
        parent = str(parent_inventory_id or "").strip()
        if (
            market.asset != "BTC"
            or market.horizon_minutes != 5
            or market.condition_id != evidence.condition_id
            or not market.event_start_ms <= decision < market.end_ms
            or evidence.observed_wall_ms > decision
            or selected_outcome not in {"Up", "Down"}
            or selected_side not in {"BUY", "SELL"}
            or selected_quantity < minimum_order
            or not _is_tick_aligned(
                selected_quantity,
                POLYMARKET_ROUND21_SHARE_QUANTUM,
            )
            or not Decimal("0") < tick < Decimal("1")
            or not _is_tick_aligned(selected_limit, tick)
            or not tick <= selected_limit <= Decimal("1") - tick
            or not Decimal("0")
            <= builder_fee_bps
            <= POLYMARKET_ROUND21_MAXIMUM_BUILDER_TAKER_FEE_BPS
            or owned_cost < 0
            or (selected_side == "BUY" and (parent or owned_cost != 0))
            or (
                selected_side == "SELL"
                and (_INVENTORY_ID.fullmatch(parent) is None or owned_cost <= 0)
            )
        ):
            raise ValueError("Round 21 aggressive order plan is invalid")
        target = (
            decision
            + scenario.submission_latency_ms
            + evidence.taker_order_delay_ms
            + evidence.general_order_delay_seconds * 1_000
        )
        if target >= market.end_ms:
            raise ValueError("Round 21 aggressive order cannot execute before expiry")
        token_id = (
            market.up_token_id if selected_outcome == "Up" else market.down_token_id
        )
        fee_model = market.fee_schedule.fee_model()
        maximum_platform_fee = fee_model(
            Decimal("0.5"),
            selected_quantity,
            "taker",
        )
        maximum_price_levels = _maximum_executable_price_levels(
            side=selected_side,
            limit_price=selected_limit,
            tick_size=tick,
        )
        if maximum_platform_fee > 0:
            maximum_platform_fee += (
                Decimal(maximum_price_levels - 1)
                * POLYMARKET_ROUND21_PLATFORM_FEE_QUANTUM
            )
        maximum_builder_fee = _ceil_quote(
            selected_quantity * selected_limit * builder_fee_bps / Decimal("10000"),
            POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM,
        )
        if builder_fee_bps > 0:
            maximum_builder_fee += (
                Decimal(maximum_price_levels - 1)
                * POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM
            )
        maximum_loss = (
            selected_quantity * selected_limit
            + maximum_platform_fee
            + maximum_builder_fee
            if selected_side == "BUY"
            else owned_cost
        )
        predictor_sha = _sha256(
            predictor_evidence_sha256,
            name="predictor evidence",
        )
        reconciliation = _sha256(
            reconciliation_sha256,
            name="reconciliation",
        )
        provisional = cls(
            condition_id=market.condition_id,
            token_id=token_id,
            event_end_ms=market.end_ms,
            decision_time_ms=decision,
            effective_execution_target_ms=target,
            outcome=selected_outcome,
            side=selected_side,
            quantity=selected_quantity,
            limit_price=selected_limit,
            tick_size=tick,
            minimum_order_size=minimum_order,
            scenario=scenario,
            fee_enabled=market.fee_schedule.enabled,
            fee_rate=market.fee_schedule.rate,
            fee_exponent=market.fee_schedule.exponent,
            fee_taker_only=market.fee_schedule.taker_only,
            builder_taker_fee_bps=builder_fee_bps,
            maximum_loss_quote=maximum_loss,
            parent_inventory_id=parent,
            predictor_evidence_sha256=predictor_sha,
            reconciliation_sha256=reconciliation,
            market_execution_evidence_sha256=evidence.evidence_sha256,
            plan_sha256=_EMPTY_SHA256,
        )
        return replace(
            provisional,
            plan_sha256=_canonical_sha256(provisional.identity_payload()),
        ).validated()

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_EXECUTION_SCHEMA_VERSION,
            "execution_policy_sha256": (POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256),
            "condition_id": self.condition_id,
            "token_id": self.token_id,
            "event_end_ms": self.event_end_ms,
            "decision_time_ms": self.decision_time_ms,
            "effective_execution_target_ms": self.effective_execution_target_ms,
            "outcome": self.outcome,
            "side": self.side,
            "quantity": format(self.quantity, "f"),
            "limit_price": format(self.limit_price, "f"),
            "tick_size": format(self.tick_size, "f"),
            "minimum_order_size": format(self.minimum_order_size, "f"),
            "scenario": {
                "name": self.scenario.name,
                "submission_latency_ms": self.scenario.submission_latency_ms,
                "displayed_depth_fraction": format(
                    self.scenario.displayed_depth_fraction,
                    "f",
                ),
                "adverse_ticks": self.scenario.adverse_ticks,
            },
            "fee": {
                "enabled": self.fee_enabled,
                "rate": format(self.fee_rate, "f"),
                "exponent": self.fee_exponent,
                "taker_only": self.fee_taker_only,
                "builder_taker_fee_bps": format(
                    self.builder_taker_fee_bps,
                    "f",
                ),
                "maker_rebate_credit": False,
            },
            "maximum_loss_quote": format(self.maximum_loss_quote, "f"),
            "parent_inventory_id": self.parent_inventory_id,
            "predictor_evidence_sha256": self.predictor_evidence_sha256,
            "reconciliation_sha256": self.reconciliation_sha256,
            "market_execution_evidence_sha256": (self.market_execution_evidence_sha256),
            "binance_credentials_used": False,
            "binance_execution_connected": False,
            "trading_authority": False,
        }

    def validated(self) -> Round21AggressiveOrderPlan:
        try:
            scenario = self.scenario.validated()
            token_valid = _TOKEN_ID.fullmatch(self.token_id) is not None
            hashes_valid = all(
                _SHA256.fullmatch(value) is not None and value != _EMPTY_SHA256
                for value in (
                    self.predictor_evidence_sha256,
                    self.reconciliation_sha256,
                    self.market_execution_evidence_sha256,
                )
            )
            decimals_valid = all(
                value.is_finite()
                for value in (
                    self.quantity,
                    self.limit_price,
                    self.tick_size,
                    self.minimum_order_size,
                    self.fee_rate,
                    self.builder_taker_fee_bps,
                    self.maximum_loss_quote,
                )
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Round 21 aggressive order plan differs") from None
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or not token_valid
            or self.event_end_ms <= self.effective_execution_target_ms
            or not self.decision_time_ms < self.effective_execution_target_ms
            or self.outcome not in {"Up", "Down"}
            or self.side not in {"BUY", "SELL"}
            or self.quantity < self.minimum_order_size
            or not _is_tick_aligned(
                self.quantity,
                POLYMARKET_ROUND21_SHARE_QUANTUM,
            )
            or not _is_tick_aligned(self.limit_price, self.tick_size)
            or not self.tick_size <= self.limit_price <= Decimal("1") - self.tick_size
            or self.fee_rate < 0
            or self.fee_rate > 1
            or self.fee_exponent <= 0
            or type(self.fee_enabled) is not bool
            or type(self.fee_taker_only) is not bool
            or not Decimal("0")
            <= self.builder_taker_fee_bps
            <= POLYMARKET_ROUND21_MAXIMUM_BUILDER_TAKER_FEE_BPS
            or self.maximum_loss_quote <= 0
            or (self.side == "BUY" and self.parent_inventory_id)
            or (
                self.side == "SELL"
                and _INVENTORY_ID.fullmatch(self.parent_inventory_id) is None
            )
            or scenario != self.scenario
            or not hashes_valid
            or not decimals_valid
            or self.plan_sha256 != _canonical_sha256(self.identity_payload())
            or self.trading_authority
        ):
            raise ValueError("Round 21 aggressive order plan differs")
        return self


def _builder_fee(
    price: Decimal,
    quantity: Decimal,
    builder_fee_bps: Decimal,
) -> Decimal:
    return _ceil_quote(
        price * quantity * builder_fee_bps / Decimal("10000"),
        POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM,
    )


def _transformed_book(
    plan: Round21AggressiveOrderPlan,
    book: PaperBookSnapshot,
) -> PaperBookSnapshot:
    source = book.validated()
    adverse = plan.tick_size * plan.scenario.adverse_ticks
    fraction = plan.scenario.displayed_depth_fraction

    def transform(
        values: tuple[BookLevel, ...],
        *,
        side: str,
    ) -> tuple[BookLevel, ...]:
        output: list[BookLevel] = []
        for level in values:
            if not _is_tick_aligned(level.price, plan.tick_size):
                raise ValueError("Round 21 execution book price is off tick")
            price = level.price
            if plan.side == "BUY" and side == "ask":
                price += adverse
            elif plan.side == "SELL" and side == "bid":
                price -= adverse
            quantity = level.quantity * fraction
            if Decimal("0") < price < Decimal("1") and quantity > 0:
                output.append(BookLevel(price=price, quantity=quantity))
        return tuple(output)

    bids = transform(source.bids, side="bid")
    asks = transform(source.asks, side="ask")
    identity = {
        "schema_version": POLYMARKET_ROUND21_EXECUTION_SCHEMA_VERSION,
        "execution_policy_sha256": POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256,
        "plan_sha256": plan.plan_sha256,
        "source_payload_sha256": source.source_payload_sha256,
        "received_wall_ms": source.received_wall_ms,
        "scenario": plan.scenario.name,
        "side": plan.side,
        "bids": [
            [format(level.price, "f"), format(level.quantity, "f")] for level in bids
        ],
        "asks": [
            [format(level.price, "f"), format(level.quantity, "f")] for level in asks
        ],
    }
    return replace(
        source,
        bids=bids,
        asks=asks,
        source_payload_sha256=_canonical_sha256(identity),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round21AggressiveExecutionObservation:
    plan_sha256: str
    state: str
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    average_fill_price: Decimal
    platform_fee_quote: Decimal
    builder_fee_quote: Decimal
    total_fee_quote: Decimal
    execution_cash_flow_quote: Decimal
    conservative_utility_bound_quote: Decimal
    blocks_new_exposure: bool
    execution_book_sha256: str
    transformed_execution_book_sha256: str
    reason: str
    observation_sha256: str
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_EXECUTION_SCHEMA_VERSION,
            "execution_policy_sha256": (POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256),
            "plan_sha256": self.plan_sha256,
            "state": self.state,
            "filled_quantity": format(self.filled_quantity, "f"),
            "unfilled_quantity": format(self.unfilled_quantity, "f"),
            "average_fill_price": format(self.average_fill_price, "f"),
            "platform_fee_quote": format(self.platform_fee_quote, "f"),
            "builder_fee_quote": format(self.builder_fee_quote, "f"),
            "total_fee_quote": format(self.total_fee_quote, "f"),
            "execution_cash_flow_quote": format(
                self.execution_cash_flow_quote,
                "f",
            ),
            "conservative_utility_bound_quote": format(
                self.conservative_utility_bound_quote,
                "f",
            ),
            "blocks_new_exposure": self.blocks_new_exposure,
            "execution_book_sha256": self.execution_book_sha256,
            "transformed_execution_book_sha256": (
                self.transformed_execution_book_sha256
            ),
            "reason": self.reason,
            "trading_authority": False,
        }

    def validated(self) -> Round21AggressiveExecutionObservation:
        decimals = (
            self.filled_quantity,
            self.unfilled_quantity,
            self.average_fill_price,
            self.platform_fee_quote,
            self.builder_fee_quote,
            self.total_fee_quote,
            self.execution_cash_flow_quote,
            self.conservative_utility_bound_quote,
        )
        if (
            _SHA256.fullmatch(self.plan_sha256) is None
            or self.state
            not in {
                "filled",
                "partial_fill",
                "known_no_fill",
                "unknown_after_submit",
            }
            or any(not value.is_finite() for value in decimals)
            or min(
                self.filled_quantity,
                self.unfilled_quantity,
                self.average_fill_price,
                self.platform_fee_quote,
                self.builder_fee_quote,
                self.total_fee_quote,
            )
            < 0
            or self.total_fee_quote != self.platform_fee_quote + self.builder_fee_quote
            or type(self.blocks_new_exposure) is not bool
            or (
                self.state == "unknown_after_submit"
                and (
                    self.filled_quantity != 0
                    or self.average_fill_price != 0
                    or self.total_fee_quote != 0
                    or self.execution_cash_flow_quote != 0
                    or not self.blocks_new_exposure
                    or self.transformed_execution_book_sha256
                    or (
                        self.execution_book_sha256
                        and _SHA256.fullmatch(self.execution_book_sha256) is None
                    )
                )
            )
            or (
                self.state != "unknown_after_submit"
                and (
                    self.blocks_new_exposure
                    or _SHA256.fullmatch(self.execution_book_sha256) is None
                    or _SHA256.fullmatch(self.transformed_execution_book_sha256) is None
                )
            )
            or (
                self.state == "filled"
                and (self.filled_quantity <= 0 or self.unfilled_quantity != 0)
            )
            or (
                self.state == "partial_fill"
                and (self.filled_quantity <= 0 or self.unfilled_quantity <= 0)
            )
            or (
                self.state == "known_no_fill"
                and (
                    self.filled_quantity != 0
                    or self.average_fill_price != 0
                    or self.total_fee_quote != 0
                    or self.execution_cash_flow_quote != 0
                )
            )
            or not self.reason
            or self.observation_sha256 != _canonical_sha256(self.identity_payload())
            or self.trading_authority
        ):
            raise ValueError("Round 21 aggressive execution observation differs")
        return self


def _unknown_observation(
    plan: Round21AggressiveOrderPlan,
    *,
    reason: str,
    execution_book_sha256: str = "",
) -> Round21AggressiveExecutionObservation:
    provisional = Round21AggressiveExecutionObservation(
        plan_sha256=plan.plan_sha256,
        state="unknown_after_submit",
        filled_quantity=Decimal("0"),
        unfilled_quantity=plan.quantity,
        average_fill_price=Decimal("0"),
        platform_fee_quote=Decimal("0"),
        builder_fee_quote=Decimal("0"),
        total_fee_quote=Decimal("0"),
        execution_cash_flow_quote=Decimal("0"),
        conservative_utility_bound_quote=-plan.maximum_loss_quote,
        blocks_new_exposure=True,
        execution_book_sha256=execution_book_sha256,
        transformed_execution_book_sha256="",
        reason=reason,
        observation_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        observation_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def observe_round21_aggressive_execution(
    plan: Round21AggressiveOrderPlan,
    book: PaperBookSnapshot | None,
) -> Round21AggressiveExecutionObservation:
    """Replay one fixed Polymarket FAK plan on its delayed observed book."""

    selected = plan.validated()
    if book is None:
        return _unknown_observation(
            selected,
            reason="missing_execution_book_after_submit",
        )
    source = book.validated()
    if (
        source.venue != "polymarket"
        or source.market_id != selected.condition_id
        or source.asset_id != selected.token_id
        or source.received_wall_ms < selected.effective_execution_target_ms
        or source.received_wall_ms
        > selected.effective_execution_target_ms
        + POLYMARKET_ROUND21_MAXIMUM_EXECUTION_OBSERVATION_LATENESS_MS
        or source.received_wall_ms >= selected.event_end_ms
    ):
        raise ValueError("Round 21 delayed execution book is invalid")
    if not source.connected or not source.gap_free:
        return _unknown_observation(
            selected,
            reason="execution_book_connection_or_gap_unproven",
            execution_book_sha256=source.source_payload_sha256,
        )
    transformed = _transformed_book(selected, source)
    platform_model = PolymarketFeeModel(
        enabled=selected.fee_enabled,
        rate=selected.fee_rate,
        exponent=selected.fee_exponent,
        taker_only=selected.fee_taker_only,
    )

    def combined_fee(
        price: Decimal,
        quantity: Decimal,
        role: str,
    ) -> Decimal:
        return platform_model(price, quantity, role) + _builder_fee(
            price,
            quantity,
            selected.builder_taker_fee_bps,
        )

    intent = PaperOrderIntent(
        intent_id="round21-" + selected.plan_sha256[:40],
        venue="polymarket",
        market_id=selected.condition_id,
        asset_id=selected.token_id,
        symbol="BTC",
        outcome=selected.outcome,
        side=selected.side,
        order_type="FAK",
        limit_price=selected.limit_price,
        quantity=selected.quantity,
        created_at_ms=selected.decision_time_ms,
        expires_at_ms=selected.event_end_ms,
        parent_inventory_id=selected.parent_inventory_id,
    )
    result = simulate_aggressive_order(
        intent,
        transformed,
        execution_time_ms=transformed.received_wall_ms,
        submission_latency_ms=(
            selected.effective_execution_target_ms - selected.decision_time_ms
        ),
        maximum_book_age_ms=0,
        fee=combined_fee,
        owned_quantity=(selected.quantity if selected.side == "SELL" else None),
        closing_position=False,
    )
    if result.state == "UNKNOWN":
        return _unknown_observation(selected, reason=result.reason)
    if result.state not in {"FILLED", "CANCELLED", "EXPIRED"}:
        raise RuntimeError("Round 21 aggressive execution state differs")
    platform_fee = sum(
        (
            platform_model(fill.price, fill.quantity, fill.liquidity_role)
            for fill in result.fills
        ),
        start=Decimal("0"),
    )
    builder_fee = sum(
        (
            _builder_fee(
                fill.price,
                fill.quantity,
                selected.builder_taker_fee_bps,
            )
            for fill in result.fills
        ),
        start=Decimal("0"),
    )
    if result.fee_quote != platform_fee + builder_fee:
        raise RuntimeError("Round 21 execution fee reconciliation differs")
    notional = result.average_fill_price * result.filled_quantity
    cash_flow = (
        -(notional + result.fee_quote)
        if selected.side == "BUY"
        else notional - result.fee_quote
    )
    if result.filled_quantity == selected.quantity:
        state = "filled"
    elif result.filled_quantity > 0:
        state = "partial_fill"
    else:
        state = "known_no_fill"
    provisional = Round21AggressiveExecutionObservation(
        plan_sha256=selected.plan_sha256,
        state=state,
        filled_quantity=result.filled_quantity,
        unfilled_quantity=result.remaining_quantity,
        average_fill_price=result.average_fill_price,
        platform_fee_quote=platform_fee,
        builder_fee_quote=builder_fee,
        total_fee_quote=result.fee_quote,
        execution_cash_flow_quote=cash_flow,
        conservative_utility_bound_quote=cash_flow,
        blocks_new_exposure=False,
        execution_book_sha256=source.source_payload_sha256,
        transformed_execution_book_sha256=(transformed.source_payload_sha256),
        reason=result.reason,
        observation_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        observation_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_EXECUTION_POLICY_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256",
    "POLYMARKET_ROUND21_EXECUTION_SCENARIOS",
    "POLYMARKET_ROUND21_EXECUTION_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_MAXIMUM_EXECUTION_OBSERVATION_LATENESS_MS",
    "POLYMARKET_ROUND21_SHARE_QUANTUM",
    "Round21AggressiveExecutionObservation",
    "Round21AggressiveOrderPlan",
    "Round21ExecutionScenario",
    "Round21MarketExecutionEvidence",
    "load_round21_execution_policy",
    "observe_round21_aggressive_execution",
    "round21_execution_scenario",
    "validate_round21_execution_policy",
]
