"""Exact one-condition Round 17 economic outcomes from resolved replay evidence."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from typing import Mapping, Sequence

from .paper_execution import BOT_OWNER, PaperExecutionResult
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook, PolymarketResolutionEvidence
from .polymarket_round14_contract import PolymarketRound14Program
from .polymarket_round17_dataset import PolymarketRound17ConditionDataset
from .polymarket_round17_economic import (
    POLYMARKET_ROUND17_ECONOMIC_PATHS,
    POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS,
    Round17ConditionEconomicOutcome,
    build_round17_condition_economic_outcome,
)
from .polymarket_round17_execution import (
    POLYMARKET_ROUND17_MAXIMUM_CREATION_BOOK_AGE_MS,
    POLYMARKET_ROUND17_MAXIMUM_EXECUTION_OBSERVATION_DELAY_MS,
    Round17EntryDecision,
    Round17EntryObservation,
    Round17OwnedLot,
    Round17ProbabilityEnvelope,
    observe_round17_complement_lock,
    observe_round17_entry,
    observe_round17_owned_close,
    plan_round17_complement_lock,
    plan_round17_owned_close,
    select_round17_entry,
)
from .polymarket_round17_features import (
    POLYMARKET_ROUND17_CONTRACT_SHA256,
    PolymarketRound17FeatureRow,
)
from .polymarket_round17_uncertainty import Round17CalibratedEnvelope


POLYMARKET_ROUND17_OUTCOME_MATERIALIZATION_SCHEMA_VERSION = (
    "polymarket-round17-condition-economic-materialization-v1"
)
POLYMARKET_ROUND17_RETRY_INTERVAL_MS = 1_000
POLYMARKET_ROUND17_MINIMUM_HOLD_MS = 1_000
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTCOMES = ("Up", "Down")
_SCENARIOS = (
    "primary",
    "latency_250ms",
    "latency_750ms",
    "latency_1000ms",
    "half_depth",
    "quarter_depth",
    "one_tick_adverse",
    "combined",
)
_PROFILES = ("conservative", "regular", "aggressive")
_SOURCE_PARTITIONS = frozenset({"tune_economic", "test"})
_NUMERIC_GUARD = Decimal("0.000000000001")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not selected.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return selected


@dataclass(frozen=True, slots=True)
class Round17DecisionProbability:
    condition_id: str
    decision_time_ms: int
    feature_input_sha256: str
    feature_values_sha256: str
    model_pretest_sha256: str
    calibration_sha256: str
    calibration_supported: bool
    calibration_support_condition_count: int
    envelope: Round17ProbabilityEnvelope
    prediction_sha256: str
    source_partition: str = "tune_economic"
    test_access_sha256: str | None = None

    def identity_payload(self) -> dict[str, object]:
        selected = self.envelope.validated()
        return {
            "schema_version": "polymarket-round17-decision-probability-v2",
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "feature_input_sha256": self.feature_input_sha256,
            "feature_values_sha256": self.feature_values_sha256,
            "model_pretest_sha256": self.model_pretest_sha256,
            "calibration_sha256": self.calibration_sha256,
            "calibration_supported": self.calibration_supported,
            "calibration_support_condition_count": (
                self.calibration_support_condition_count
            ),
            "source_partition": self.source_partition,
            "test_access_sha256": self.test_access_sha256,
            "probability_up": format(selected.probability_up, "f"),
            "lower_up": format(selected.lower_up, "f"),
            "upper_up": format(selected.upper_up, "f"),
        }

    def validated(self) -> "Round17DecisionProbability":
        self.envelope.validated()
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.decision_time_ms <= 0
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.feature_input_sha256,
                    self.feature_values_sha256,
                    self.model_pretest_sha256,
                    self.calibration_sha256,
                    self.prediction_sha256,
                )
            )
            or type(self.calibration_supported) is not bool
            or self.calibration_support_condition_count < 0
            or (
                self.calibration_supported
                and self.calibration_support_condition_count < 1
            )
            or self.source_partition not in _SOURCE_PARTITIONS
            or (
                self.source_partition == "tune_economic"
                and self.test_access_sha256 is not None
            )
            or (
                self.source_partition == "test"
                and (
                    self.test_access_sha256 is None
                    or _SHA256.fullmatch(self.test_access_sha256) is None
                )
            )
            or self.envelope.evidence_sha256 != self.prediction_sha256
            or self.prediction_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 decision probability is invalid")
        return self


def build_round17_decision_probability(
    row: PolymarketRound17FeatureRow,
    *,
    probability_up: Decimal,
    lower_up: Decimal,
    upper_up: Decimal,
    model_pretest_sha256: str,
    calibration_sha256: str,
    calibration_supported: bool,
    calibration_support_condition_count: int,
    source_partition: str = "tune_economic",
    test_access_sha256: str | None = None,
) -> Round17DecisionProbability:
    if not isinstance(row, PolymarketRound17FeatureRow):
        raise TypeError("Round 17 probability source row type differs")
    provisional_envelope = Round17ProbabilityEnvelope(
        probability_up=_decimal(probability_up, name="probability_up"),
        lower_up=_decimal(lower_up, name="lower_up"),
        upper_up=_decimal(upper_up, name="upper_up"),
        evidence_sha256="0" * 64,
    )
    payload = {
        "schema_version": "polymarket-round17-decision-probability-v2",
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "condition_id": row.condition_id,
        "decision_time_ms": row.decision_time_ms,
        "feature_input_sha256": row.input_sha256,
        "feature_values_sha256": row.values_sha256,
        "model_pretest_sha256": str(model_pretest_sha256),
        "calibration_sha256": str(calibration_sha256),
        "calibration_supported": bool(calibration_supported),
        "calibration_support_condition_count": int(calibration_support_condition_count),
        "source_partition": str(source_partition),
        "test_access_sha256": test_access_sha256,
        "probability_up": format(provisional_envelope.probability_up, "f"),
        "lower_up": format(provisional_envelope.lower_up, "f"),
        "upper_up": format(provisional_envelope.upper_up, "f"),
    }
    digest = _canonical_sha256(payload)
    return Round17DecisionProbability(
        condition_id=row.condition_id,
        decision_time_ms=row.decision_time_ms,
        feature_input_sha256=row.input_sha256,
        feature_values_sha256=row.values_sha256,
        model_pretest_sha256=str(model_pretest_sha256),
        calibration_sha256=str(calibration_sha256),
        calibration_supported=bool(calibration_supported),
        calibration_support_condition_count=int(calibration_support_condition_count),
        envelope=Round17ProbabilityEnvelope(
            probability_up=provisional_envelope.probability_up,
            lower_up=provisional_envelope.lower_up,
            upper_up=provisional_envelope.upper_up,
            evidence_sha256=digest,
        ),
        prediction_sha256=digest,
        source_partition=str(source_partition),
        test_access_sha256=test_access_sha256,
    ).validated()


def build_round17_calibrated_decision_probability(
    row: PolymarketRound17FeatureRow,
    calibrated: Round17CalibratedEnvelope,
) -> Round17DecisionProbability:
    if not isinstance(calibrated, Round17CalibratedEnvelope):
        raise TypeError("Round 17 calibrated envelope type differs")
    if (
        calibrated.source_role not in _SOURCE_PARTITIONS
        or calibrated.condition_id != row.condition_id
        or calibrated.decision_time_ms != row.decision_time_ms
        or calibrated.feature_input_sha256 != row.input_sha256
        or calibrated.feature_values_sha256 != row.values_sha256
    ):
        raise ValueError("Round 17 calibrated feature evidence differs")
    envelope = calibrated.envelope.validated()
    return build_round17_decision_probability(
        row,
        probability_up=envelope.probability_up,
        lower_up=envelope.lower_up,
        upper_up=envelope.upper_up,
        model_pretest_sha256=calibrated.model_pretest_sha256,
        calibration_sha256=calibrated.calibration_sha256,
        calibration_supported=calibrated.supported,
        calibration_support_condition_count=(calibrated.support_condition_count),
        source_partition=calibrated.source_role,
        test_access_sha256=calibrated.test_access_sha256,
    )


class _BookIndex:
    def __init__(
        self,
        market: PolymarketFiveMinuteMarket,
        books: Sequence[PolymarketRecordedBook],
        *,
        run_id: str,
    ) -> None:
        by_outcome: dict[str, list[PolymarketRecordedBook]] = {
            "Up": [],
            "Down": [],
        }
        for book in books:
            if (
                book.run_id != run_id
                or book.market.condition_id != market.condition_id
                or book.market.event_start_ms != market.event_start_ms
                or book.market.end_ms != market.end_ms
                or book.outcome not in by_outcome
                or book.token_id
                != (
                    market.up_token_id if book.outcome == "Up" else market.down_token_id
                )
                or not book.snapshot.connected
                or not book.snapshot.gap_free
            ):
                raise ValueError("Round 17 economic replay book identity differs")
            by_outcome[book.outcome].append(book)
        self.books = {
            outcome: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.received_wall_ms,
                        item.received_monotonic_ns,
                        item.event_id,
                    ),
                )
            )
            for outcome, values in by_outcome.items()
        }
        if any(not values for values in self.books.values()):
            raise ValueError("Round 17 economic replay lacks paired books")
        self.times = {
            outcome: tuple(item.received_wall_ms for item in values)
            for outcome, values in self.books.items()
        }

    def creation(
        self,
        outcome: str,
        *,
        decision_time_ms: int,
    ) -> PolymarketRecordedBook | None:
        values = self.books[outcome]
        index = bisect_right(self.times[outcome], decision_time_ms) - 1
        if index < 0:
            return None
        selected = values[index]
        if (
            decision_time_ms - selected.received_wall_ms
            > POLYMARKET_ROUND17_MAXIMUM_CREATION_BOOK_AGE_MS
        ):
            return None
        return selected

    def execution(
        self,
        outcome: str,
        *,
        target_time_ms: int,
        segment_id: str,
    ) -> PolymarketRecordedBook | None:
        values = self.books[outcome]
        index = bisect_left(self.times[outcome], target_time_ms)
        while index < len(values):
            selected = values[index]
            if (
                selected.received_wall_ms - target_time_ms
                > POLYMARKET_ROUND17_MAXIMUM_EXECUTION_OBSERVATION_DELAY_MS
            ):
                return None
            if selected.segment_id == segment_id:
                return selected
            index += 1
        return None


def _validate_resolution(
    market: PolymarketFiveMinuteMarket,
    dataset: PolymarketRound17ConditionDataset,
    resolution: PolymarketResolutionEvidence,
) -> None:
    expected_token = (
        market.up_token_id
        if resolution.winning_outcome == "Up"
        else market.down_token_id
        if resolution.winning_outcome == "Down"
        else ""
    )
    if (
        resolution.run_id != dataset.run_id
        or resolution.condition_id != market.condition_id
        or resolution.winning_asset_id != expected_token
        or resolution.resolved_at_ms < market.end_ms
        or resolution.received_wall_ms < market.end_ms
        or _SHA256.fullmatch(resolution.event_sha256) is None
        or not resolution.event_id
        or not resolution.source
    ):
        raise ValueError("Round 17 official resolution identity differs")


def _result_payload(result: PaperExecutionResult) -> dict[str, object]:
    return {
        "state": result.state,
        "filled_quantity": format(result.filled_quantity, "f"),
        "remaining_quantity": format(result.remaining_quantity, "f"),
        "average_fill_price": format(result.average_fill_price, "f"),
        "fee_quote": format(result.fee_quote, "f"),
        "reason": result.reason,
        "source_payload_sha256": result.source_payload_sha256,
        "fills": [
            [
                format(fill.price, "f"),
                format(fill.quantity, "f"),
                format(fill.fee_quote, "f"),
                fill.liquidity_role,
            ]
            for fill in result.fills
        ],
    }


def _source_evidence_sha256(
    *,
    path: str,
    decision: Round17EntryDecision,
    entry: Round17EntryObservation,
    resolution: PolymarketResolutionEvidence,
    extra: object,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "polymarket-round17-economic-source-evidence-v1",
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "path": path,
            "decision_sha256": decision.decision_sha256,
            "entry_observation_sha256": entry.observation_sha256,
            "resolution_event_sha256": resolution.event_sha256,
            "extra": extra,
        }
    )


def _settlement_pnl(
    decision: Round17EntryDecision,
    entry: Round17EntryObservation,
    resolution: PolymarketResolutionEvidence,
) -> Decimal:
    candidate = decision.candidate
    if (
        candidate is None
        or entry.state != "filled"
        or entry.actual_entry_cost_quote is None
    ):
        raise ValueError("Round 17 settlement path lacks a filled entry")
    payout = (
        candidate.quantity
        if resolution.winning_outcome == candidate.outcome
        else Decimal("0")
    )
    return payout - entry.actual_entry_cost_quote


def _future_probabilities(
    predictions: Sequence[Round17DecisionProbability],
    *,
    after_ms: int,
) -> tuple[Round17DecisionProbability, ...]:
    return tuple(item for item in predictions if item.decision_time_ms >= after_ms)


def _intrawindow_outcome(
    *,
    market: PolymarketFiveMinuteMarket,
    predictions: Sequence[Round17DecisionProbability],
    index: _BookIndex,
    program: PolymarketRound14Program,
    scenario: str,
    decision: Round17EntryDecision,
    entry: Round17EntryObservation,
    resolution: PolymarketResolutionEvidence,
) -> tuple[Decimal, bool, bool, object]:
    candidate = decision.candidate
    if candidate is None or entry.actual_entry_cost_quote is None:
        raise ValueError("Round 17 intrawindow path lacks a filled entry")
    lot = Round17OwnedLot(
        owner=BOT_OWNER,
        parent_intent_id="round17-parent-" + decision.decision_sha256[:32],
        market_id=market.condition_id,
        token_id=candidate.token_id,
        outcome=candidate.outcome,
        quantity=candidate.quantity,
        entry_cost_quote=entry.actual_entry_cost_quote,
    ).validated()
    scenario_value = next(item for item in program.scenarios if item.name == scenario)
    for probability in _future_probabilities(
        predictions,
        after_ms=decision.decision_time_ms + POLYMARKET_ROUND17_MINIMUM_HOLD_MS,
    ):
        if (
            probability.decision_time_ms + scenario_value.submission_latency_ms
            >= market.end_ms
        ):
            break
        creation = index.creation(
            lot.outcome,
            decision_time_ms=probability.decision_time_ms,
        )
        if creation is None:
            continue
        plan = plan_round17_owned_close(
            market,
            lot,
            creation.snapshot,
            program,
            decision_time_ms=probability.decision_time_ms,
            scenario_name=scenario,
        )
        if plan is None:
            continue
        hold_lower = (
            probability.envelope.selected_lower_bound(lot.outcome) * lot.quantity
        )
        if plan.minimum_exit_proceeds_quote <= hold_lower + _NUMERIC_GUARD:
            continue
        delayed = index.execution(
            lot.outcome,
            target_time_ms=(
                probability.decision_time_ms + scenario_value.submission_latency_ms
            ),
            segment_id=creation.segment_id,
        )
        observed = observe_round17_owned_close(
            plan,
            market,
            None if delayed is None else delayed.snapshot,
            program,
        )
        evidence = {
            "close_plan_sha256": plan.plan_sha256,
            "close_result": _result_payload(observed),
            "trigger_prediction_sha256": probability.prediction_sha256,
            "hold_lower_quote": format(hold_lower, "f"),
        }
        if observed.state == "FILLED":
            proceeds = (
                observed.average_fill_price * observed.filled_quantity
                - observed.fee_quote
            )
            return proceeds - lot.entry_cost_quote, False, False, evidence
        if observed.state in {"UNKNOWN", "CLOSE_PENDING"}:
            return -candidate.maximum_entry_loss_quote, True, True, evidence
    return (
        _settlement_pnl(decision, entry, resolution),
        False,
        False,
        {"terminal": "official_settlement_without_owned_close"},
    )


def _complement_outcome(
    *,
    market: PolymarketFiveMinuteMarket,
    predictions: Sequence[Round17DecisionProbability],
    index: _BookIndex,
    program: PolymarketRound14Program,
    scenario: str,
    decision: Round17EntryDecision,
    entry: Round17EntryObservation,
    resolution: PolymarketResolutionEvidence,
) -> tuple[Decimal, bool, bool, object]:
    candidate = decision.candidate
    if candidate is None or entry.actual_entry_cost_quote is None:
        raise ValueError("Round 17 complement path lacks a filled entry")
    lot = Round17OwnedLot(
        owner=BOT_OWNER,
        parent_intent_id="round17-parent-" + decision.decision_sha256[:32],
        market_id=market.condition_id,
        token_id=candidate.token_id,
        outcome=candidate.outcome,
        quantity=candidate.quantity,
        entry_cost_quote=entry.actual_entry_cost_quote,
    ).validated()
    complement = "Down" if lot.outcome == "Up" else "Up"
    scenario_value = next(item for item in program.scenarios if item.name == scenario)
    for probability in _future_probabilities(
        predictions,
        after_ms=decision.decision_time_ms + POLYMARKET_ROUND17_MINIMUM_HOLD_MS,
    ):
        if (
            probability.decision_time_ms + scenario_value.submission_latency_ms
            >= market.end_ms
        ):
            break
        creation = index.creation(
            complement,
            decision_time_ms=probability.decision_time_ms,
        )
        if creation is None:
            continue
        plan = plan_round17_complement_lock(
            market,
            lot,
            creation.snapshot,
            program,
            decision_time_ms=probability.decision_time_ms,
            scenario_name=scenario,
        )
        if plan is None:
            continue
        delayed = index.execution(
            complement,
            target_time_ms=(
                probability.decision_time_ms + scenario_value.submission_latency_ms
            ),
            segment_id=creation.segment_id,
        )
        observed = observe_round17_complement_lock(
            plan,
            market,
            None if delayed is None else delayed.snapshot,
            program,
        )
        evidence = {
            "complement_plan_sha256": plan.plan_sha256,
            "complement_observation_sha256": observed.observation_sha256,
            "trigger_prediction_sha256": probability.prediction_sha256,
        }
        if observed.state == "locked" and observed.guaranteed_net_quote is not None:
            return observed.guaranteed_net_quote, False, False, evidence
        if observed.state == "unknown_after_submit":
            return -candidate.maximum_entry_loss_quote, True, True, evidence
    return (
        _settlement_pnl(decision, entry, resolution),
        False,
        False,
        {"terminal": "official_settlement_without_complement_lock"},
    )


def _no_entry_outcomes(
    *,
    market: PolymarketFiveMinuteMarket,
    profile: str,
    scenario: str,
    threshold: Decimal,
    capital: Decimal,
    evidence_sha256: str,
    paths: Sequence[str],
    source_partition: str,
) -> tuple[Round17ConditionEconomicOutcome, ...]:
    return tuple(
        build_round17_condition_economic_outcome(
            condition_id=market.condition_id,
            event_start_ms=market.event_start_ms,
            path=path,
            risk_profile=profile,
            scenario=scenario,
            minimum_edge_quote_per_share=threshold,
            risk_capital_quote=capital,
            entry_executed=False,
            realized_net_quote=Decimal("0"),
            maximum_loss_quote=Decimal("0"),
            unknown_state=False,
            lifecycle_violation=False,
            ownership_violation=False,
            decision_sha256=evidence_sha256,
            source_evidence_sha256=evidence_sha256,
            source_partition=source_partition,
        )
        for path in paths
    )


def materialize_round17_condition_economic_outcomes(
    *,
    market: PolymarketFiveMinuteMarket,
    dataset: PolymarketRound17ConditionDataset,
    predictions: Sequence[Round17DecisionProbability],
    books: Sequence[PolymarketRecordedBook],
    resolution: PolymarketResolutionEvidence,
    program: PolymarketRound14Program,
    risk_capital_quote: Decimal,
    source_partition: str = "tune_economic",
    selected_policy_by_profile: Mapping[str, tuple[str, Decimal]] | None = None,
) -> tuple[Round17ConditionEconomicOutcome, ...]:
    """Build all frozen policy cells while retaining only one condition in memory."""

    source = dataset.validated()
    capital = _decimal(risk_capital_quote, name="risk capital")
    partition = str(source_partition or "").strip()
    if selected_policy_by_profile is None:
        selected_policies = {
            profile: (
                POLYMARKET_ROUND17_ECONOMIC_PATHS,
                POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS,
            )
            for profile in _PROFILES
        }
    else:
        if set(selected_policy_by_profile) != set(_PROFILES):
            raise ValueError("Round 17 selected economic policy profiles differ")
        selected_policies = {}
        for profile in _PROFILES:
            path, threshold = selected_policy_by_profile[profile]
            selected_threshold = _decimal(threshold, name="minimum edge")
            if (
                path not in POLYMARKET_ROUND17_ECONOMIC_PATHS
                or selected_threshold not in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS
            ):
                raise ValueError("Round 17 selected economic policy differs")
            selected_policies[profile] = ((path,), (selected_threshold,))
    if (
        capital <= 0
        or partition not in _SOURCE_PARTITIONS
        or market.condition_id != source.condition_id
        or market.event_start_ms != source.event_start_ms
        or market.end_ms != source.event_end_ms
        or market.asset != "BTC"
        or tuple(item.name for item in program.risk_profiles) != _PROFILES
        or tuple(item.name for item in program.scenarios) != _SCENARIOS
    ):
        raise ValueError("Round 17 economic materialization identity differs")
    _validate_resolution(market, source, resolution)
    probability_rows = tuple(item.validated() for item in predictions)
    if (
        len(probability_rows) != len(source.rows)
        or tuple(item.decision_time_ms for item in probability_rows)
        != tuple(item.decision_time_ms for item in source.rows)
        or any(
            item.condition_id != source.condition_id
            or item.feature_input_sha256 != row.input_sha256
            or item.feature_values_sha256 != row.values_sha256
            for item, row in zip(probability_rows, source.rows, strict=True)
        )
        or len({item.model_pretest_sha256 for item in probability_rows}) != 1
        or len({item.calibration_sha256 for item in probability_rows}) != 1
        or {item.source_partition for item in probability_rows} != {partition}
        or (
            partition == "test"
            and len({item.test_access_sha256 for item in probability_rows}) != 1
        )
    ):
        raise ValueError("Round 17 probability panel differs from feature rows")
    index = _BookIndex(market, books, run_id=source.run_id)
    outputs: list[Round17ConditionEconomicOutcome] = []

    for profile in _PROFILES:
        selected_paths, selected_thresholds = selected_policies[profile]
        for threshold in selected_thresholds:
            for scenario_name in _SCENARIOS:
                scenario = next(
                    item for item in program.scenarios if item.name == scenario_name
                )
                selected_decision: Round17EntryDecision | None = None
                selected_entry: Round17EntryObservation | None = None
                last_attempt_ms = -POLYMARKET_ROUND17_RETRY_INTERVAL_MS
                attempted_decision_hashes: list[str] = []
                for probability in probability_rows:
                    if (
                        probability.decision_time_ms - last_attempt_ms
                        < POLYMARKET_ROUND17_RETRY_INTERVAL_MS
                    ):
                        continue
                    creation = {
                        outcome: index.creation(
                            outcome,
                            decision_time_ms=probability.decision_time_ms,
                        )
                        for outcome in _OUTCOMES
                    }
                    if any(item is None for item in creation.values()):
                        continue
                    typed_creation = {
                        outcome: item
                        for outcome, item in creation.items()
                        if item is not None
                    }
                    if (
                        len(
                            {
                                (item.connection_id, item.segment_id)
                                for item in typed_creation.values()
                            }
                        )
                        != 1
                    ):
                        continue
                    decision = select_round17_entry(
                        market,
                        {
                            outcome: item.snapshot
                            for outcome, item in typed_creation.items()
                        },
                        probability.envelope,
                        program,
                        decision_time_ms=probability.decision_time_ms,
                        risk_profile=profile,
                        scenario_name=scenario_name,
                        risk_capital_quote=capital,
                        minimum_expected_edge_quote_per_share=threshold,
                        reconciliation_ok=True,
                        existing_owned_exposure=False,
                    )
                    attempted_decision_hashes.append(decision.decision_sha256)
                    if decision.candidate is None:
                        continue
                    last_attempt_ms = probability.decision_time_ms
                    outcome = decision.candidate.outcome
                    creation_book = typed_creation[outcome]
                    execution_book = index.execution(
                        outcome,
                        target_time_ms=(
                            probability.decision_time_ms
                            + scenario.submission_latency_ms
                        ),
                        segment_id=creation_book.segment_id,
                    )
                    entry = observe_round17_entry(
                        decision,
                        market,
                        (None if execution_book is None else execution_book.snapshot),
                        program,
                    )
                    if entry.state == "known_no_fill":
                        continue
                    selected_decision = decision
                    selected_entry = entry
                    break

                if selected_decision is None or selected_entry is None:
                    evidence = _canonical_sha256(
                        {
                            "schema_version": (
                                POLYMARKET_ROUND17_OUTCOME_MATERIALIZATION_SCHEMA_VERSION
                            ),
                            "condition_id": market.condition_id,
                            "profile": profile,
                            "threshold": format(threshold, "f"),
                            "scenario": scenario_name,
                            "attempted_decision_sha256": attempted_decision_hashes,
                            "terminal": "no_filled_or_unknown_entry",
                        }
                    )
                    outputs.extend(
                        _no_entry_outcomes(
                            market=market,
                            profile=profile,
                            scenario=scenario_name,
                            threshold=threshold,
                            capital=capital,
                            evidence_sha256=evidence,
                            paths=selected_paths,
                            source_partition=partition,
                        )
                    )
                    continue

                candidate = selected_decision.candidate
                assert candidate is not None
                if selected_entry.state == "unknown_after_submit":
                    for path in selected_paths:
                        evidence = _source_evidence_sha256(
                            path=path,
                            decision=selected_decision,
                            entry=selected_entry,
                            resolution=resolution,
                            extra={"terminal": "unknown_entry_state"},
                        )
                        outputs.append(
                            build_round17_condition_economic_outcome(
                                condition_id=market.condition_id,
                                event_start_ms=market.event_start_ms,
                                path=path,
                                risk_profile=profile,
                                scenario=scenario_name,
                                minimum_edge_quote_per_share=threshold,
                                risk_capital_quote=capital,
                                entry_executed=True,
                                realized_net_quote=(
                                    -candidate.maximum_entry_loss_quote
                                ),
                                maximum_loss_quote=(candidate.maximum_entry_loss_quote),
                                unknown_state=True,
                                lifecycle_violation=True,
                                ownership_violation=False,
                                decision_sha256=(selected_decision.decision_sha256),
                                source_evidence_sha256=evidence,
                                source_partition=partition,
                            )
                        )
                    continue

                settlement = _settlement_pnl(
                    selected_decision,
                    selected_entry,
                    resolution,
                )
                path_values: Mapping[
                    str,
                    tuple[Decimal, bool, bool, object],
                ] = {
                    "settlement_directional": (
                        settlement,
                        False,
                        False,
                        {"terminal": "official_settlement"},
                    ),
                    "intrawindow_owned_reprice": _intrawindow_outcome(
                        market=market,
                        predictions=probability_rows,
                        index=index,
                        program=program,
                        scenario=scenario_name,
                        decision=selected_decision,
                        entry=selected_entry,
                        resolution=resolution,
                    ),
                    "complement_lock": _complement_outcome(
                        market=market,
                        predictions=probability_rows,
                        index=index,
                        program=program,
                        scenario=scenario_name,
                        decision=selected_decision,
                        entry=selected_entry,
                        resolution=resolution,
                    ),
                }
                for path, (
                    pnl,
                    unknown,
                    lifecycle,
                    extra,
                ) in path_values.items():
                    if path not in selected_paths:
                        continue
                    evidence = _source_evidence_sha256(
                        path=path,
                        decision=selected_decision,
                        entry=selected_entry,
                        resolution=resolution,
                        extra=extra,
                    )
                    outputs.append(
                        build_round17_condition_economic_outcome(
                            condition_id=market.condition_id,
                            event_start_ms=market.event_start_ms,
                            path=path,
                            risk_profile=profile,
                            scenario=scenario_name,
                            minimum_edge_quote_per_share=threshold,
                            risk_capital_quote=capital,
                            entry_executed=True,
                            realized_net_quote=pnl,
                            maximum_loss_quote=(candidate.maximum_entry_loss_quote),
                            unknown_state=unknown,
                            lifecycle_violation=lifecycle,
                            ownership_violation=False,
                            decision_sha256=selected_decision.decision_sha256,
                            source_evidence_sha256=evidence,
                            source_partition=partition,
                        )
                    )

    expected_count = (
        sum(
            len(paths) * len(thresholds)
            for paths, thresholds in selected_policies.values()
        )
        * len(_SCENARIOS)
    )
    if len(outputs) != expected_count:
        raise RuntimeError("Round 17 condition economic output grid differs")
    return tuple(outputs)


__all__ = [
    "POLYMARKET_ROUND17_OUTCOME_MATERIALIZATION_SCHEMA_VERSION",
    "Round17DecisionProbability",
    "build_round17_calibrated_decision_probability",
    "build_round17_decision_probability",
    "materialize_round17_condition_economic_outcomes",
]
