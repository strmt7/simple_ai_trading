"""Target-free AI veto cases for the Round 28 augmented model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import math

import numpy as np

from . import polymarket_round27_economics as _economics
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
)
from .polymarket_round28_ai_contract import (
    POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
)
from .polymarket_round28_book_ticker import (
    POLYMARKET_ROUND28_FEATURE_NAMES,
    POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
    Round28FeatureRow,
)
from .polymarket_round28_model import Round28ProbabilityModel


POLYMARKET_ROUND28_AI_CASE_SCHEMA_VERSION = "polymarket-round28-ai-case-v1"
POLYMARKET_ROUND28_AI_CASE_PANEL_SCHEMA_VERSION = (
    "polymarket-round28-ai-case-panel-v1"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_NON_AUTHORITY_FLAGS = (
    "target_accessed",
    "outcome_accessed",
    "future_books_accessed",
    "pnl_accessed",
    "credentials_used",
    "orders_submitted",
    "trading_authority",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 AI {name} SHA-256 differs")
    return selected


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class _TargetFreeSample:
    run_id: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    market_prior_probability: float
    values: tuple[float, ...]
    feature_row_sha256: str
    source_chain_sha256: str


@dataclass(frozen=True, slots=True)
class _TargetFreePartition:
    role: str
    samples: tuple[_TargetFreeSample, ...]


@dataclass(frozen=True, slots=True)
class Round28AICase:
    partition_role: str
    condition_id: str
    event_start_ms: int
    market_end_ms: int
    decision_time_ms: int
    proposed_side: str
    token_id: str
    predicted_probability: float
    market_prior_probability_up: float
    quantity: str
    limit_price: str
    decision_tick_size: str
    decision_average_price: str
    decision_fee_quote: str
    expected_edge_per_contract: str
    segment_id: str
    connection_id: str
    decision_book_event_id: str
    decision_source_payload_sha256: str
    feature_row_sha256: str
    feature_source_chain_sha256: str
    selection_claim_sha256: str
    model_name: str
    model_feature_view: str
    model_sha256: str
    causal_features: tuple[tuple[str, float], ...]
    source_evidence_sha256: str
    case_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND28_AI_CASE_SCHEMA_VERSION,
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "partition_role": self.partition_role,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "market_end_ms": self.market_end_ms,
            "decision_time_ms": self.decision_time_ms,
            "proposed_side": self.proposed_side,
            "token_id": self.token_id,
            "predicted_probability": self.predicted_probability,
            "market_prior_probability_up": self.market_prior_probability_up,
            "quantity": self.quantity,
            "limit_price": self.limit_price,
            "decision_tick_size": self.decision_tick_size,
            "decision_average_price": self.decision_average_price,
            "decision_fee_quote": self.decision_fee_quote,
            "expected_edge_per_contract": self.expected_edge_per_contract,
            "segment_id": self.segment_id,
            "connection_id": self.connection_id,
            "decision_book_event_id": self.decision_book_event_id,
            "decision_source_payload_sha256": self.decision_source_payload_sha256,
            "feature_row_sha256": self.feature_row_sha256,
            "feature_source_chain_sha256": self.feature_source_chain_sha256,
            "feature_names_sha256": POLYMARKET_ROUND28_FEATURE_NAMES_SHA256,
            "selection_claim_sha256": self.selection_claim_sha256,
            "model_name": self.model_name,
            "model_feature_view": self.model_feature_view,
            "model_sha256": self.model_sha256,
            "causal_features": [
                {"name": name, "value": value}
                for name, value in self.causal_features
            ],
            **{field: False for field in _NON_AUTHORITY_FLAGS},
            "source_evidence_sha256": self.source_evidence_sha256,
        }

    def validated(self) -> "Round28AICase":
        names = tuple(name for name, _value in self.causal_features)
        values = tuple(value for _name, value in self.causal_features)
        decimals = tuple(
            Decimal(value)
            for value in (
                self.quantity,
                self.limit_price,
                self.decision_tick_size,
                self.decision_average_price,
                self.decision_fee_quote,
                self.expected_edge_per_contract,
            )
        )
        if (
            self.partition_role not in {"selection", "sealed"}
            or not self.condition_id.startswith("0x")
            or len(self.condition_id) != 66
            or self.market_end_ms - self.event_start_ms != 300_000
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.market_end_ms
            or self.proposed_side not in {"Up", "Down"}
            or not self.token_id
            or not 0.0 < self.predicted_probability < 1.0
            or not 0.0 < self.market_prior_probability_up < 1.0
            or any(not value.is_finite() for value in decimals)
            or Decimal(self.quantity) <= 0
            or Decimal(self.limit_price) <= 0
            or not 0 < Decimal(self.decision_tick_size) < 1
            or Decimal(self.limit_price) % Decimal(self.decision_tick_size) != 0
            or Decimal(self.decision_average_price) <= 0
            or Decimal(self.decision_fee_quote) < 0
            or Decimal(self.expected_edge_per_contract) <= 0
            or not self.segment_id
            or not self.connection_id
            or names != POLYMARKET_ROUND28_FEATURE_NAMES
            or any(not math.isfinite(value) for value in values)
            or self.model_feature_view != "round28_bbo_augmented"
            or any(
                _sha256(value, name=name) != value
                for value, name in (
                    (self.decision_source_payload_sha256, "decision source"),
                    (self.feature_row_sha256, "feature row"),
                    (self.feature_source_chain_sha256, "feature source chain"),
                    (self.selection_claim_sha256, "selection claim"),
                    (self.model_sha256, "model"),
                    (self.source_evidence_sha256, "source evidence"),
                )
            )
            or self.case_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 28 AI case differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "case_sha256": self.case_sha256}


@dataclass(frozen=True, slots=True)
class Round28AICasePanel:
    partition_role: str
    source_run_id: str
    model_name: str
    model_sha256: str
    selection_claim_sha256: str
    source_audit_sha256: str
    economic_config: dict[str, object]
    evaluated_condition_count: int
    evaluated_condition_ids_sha256: str
    baseline_candidate_population_sha256: str
    selection_reason_counts: dict[str, int]
    cases: tuple[Round28AICase, ...]
    panel_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND28_AI_CASE_PANEL_SCHEMA_VERSION,
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "partition_role": self.partition_role,
            "source_run_id": self.source_run_id,
            "model_name": self.model_name,
            "model_feature_view": "round28_bbo_augmented",
            "model_sha256": self.model_sha256,
            "selection_claim_sha256": self.selection_claim_sha256,
            "source_audit_sha256": self.source_audit_sha256,
            "economic_config": dict(self.economic_config),
            "evaluated_condition_count": self.evaluated_condition_count,
            "evaluated_condition_ids_sha256": (
                self.evaluated_condition_ids_sha256
            ),
            "baseline_candidate_population_sha256": (
                self.baseline_candidate_population_sha256
            ),
            "selection_reason_counts": dict(self.selection_reason_counts),
            "case_sha256": [case.case_sha256 for case in self.cases],
            "case_count": len(self.cases),
            "cases": [case.asdict() for case in self.cases],
            "prompt_population_sha256": _canonical_sha256(
                [round28_ai_case_prompt(case) for case in self.cases]
            ),
            **{field: False for field in _NON_AUTHORITY_FLAGS},
        }

    def validated(self) -> "Round28AICasePanel":
        cases = tuple(case.validated() for case in self.cases)
        ordered = tuple(
            sorted(
                cases,
                key=lambda case: (
                    case.event_start_ms,
                    case.condition_id,
                    case.decision_time_ms,
                ),
            )
        )
        condition_ids = tuple(case.condition_id for case in cases)
        if (
            self.partition_role not in {"selection", "sealed"}
            or not self.source_run_id
            or not self.model_name
            or self.evaluated_condition_count < len(cases)
            or len(condition_ids) != len(set(condition_ids))
            or cases != ordered
            or any(case.partition_role != self.partition_role for case in cases)
            or any(
                (
                    case.model_name,
                    case.model_sha256,
                    case.selection_claim_sha256,
                )
                != (
                    self.model_name,
                    self.model_sha256,
                    self.selection_claim_sha256,
                )
                for case in cases
            )
            or any(
                _sha256(value, name=name) != value
                for value, name in (
                    (self.model_sha256, "model"),
                    (self.selection_claim_sha256, "selection claim"),
                    (self.source_audit_sha256, "source audit"),
                    (
                        self.evaluated_condition_ids_sha256,
                        "evaluated condition population",
                    ),
                    (
                        self.baseline_candidate_population_sha256,
                        "candidate population",
                    ),
                )
            )
            or self.panel_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 28 AI case panel differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "panel_sha256": self.panel_sha256}


def round28_ai_case_prompt(case: Round28AICase) -> str:
    selected = case.validated()
    packet = {
        "decision_time_ms": selected.decision_time_ms,
        "event_start_ms": selected.event_start_ms,
        "market_end_ms": selected.market_end_ms,
        "proposed_side": selected.proposed_side,
        "predicted_probability": selected.predicted_probability,
        "market_prior_probability_up": selected.market_prior_probability_up,
        "quantity": selected.quantity,
        "limit_price": selected.limit_price,
        "decision_average_price": selected.decision_average_price,
        "decision_fee_quote": selected.decision_fee_quote,
        "expected_edge_per_contract": selected.expected_edge_per_contract,
        "causal_features": {
            name: value for name, value in selected.causal_features
        },
    }
    return (
        "You are a pre-trade risk veto for one BTC five-minute Polymarket "
        "minimum-size FOK proposal. You cannot create a trade, increase size, "
        "change side, or override deterministic controls. Use only this numeric "
        "causal packet. Reject or reduce for material liquidity, spread/cost, "
        "source freshness/gap, spot-futures disagreement, cross-venue "
        "disagreement, volatility/jump, model-market disagreement, or late "
        "horizon risk. The proposal is already at venue minimum size, so reduce "
        "is executed as abstain. Return unchanged only when no material risk is "
        "present. Return only the required JSON object. case="
        + _canonical_json(packet)
    )


def _target_free_samples(
    rows: Sequence[Round28FeatureRow],
) -> tuple[_TargetFreeSample, ...]:
    selected_rows = tuple(row.validated() for row in rows)
    if not selected_rows:
        raise ValueError("Round 28 AI feature population is empty")
    keys = tuple((row.condition_id, row.decision_time_ms) for row in selected_rows)
    if len(keys) != len(set(keys)):
        raise ValueError("Round 28 AI feature population is duplicated")
    return tuple(
        _TargetFreeSample(
            run_id=row.run_id,
            condition_id=row.condition_id,
            event_start_ms=row.event_start_ms,
            decision_time_ms=row.decision_time_ms,
            market_prior_probability=row.market_prior_probability,
            values=row.values,
            feature_row_sha256=row.row_sha256,
            source_chain_sha256=row.source_chain_sha256,
        )
        for row in selected_rows
    )


def _probabilities(
    samples: Sequence[_TargetFreeSample],
    selected_model: Round28ProbabilityModel,
) -> np.ndarray:
    features = np.asarray([sample.values for sample in samples], dtype=np.float64)
    priors = np.asarray(
        [sample.market_prior_probability for sample in samples],
        dtype=np.float64,
    )
    offsets = np.log(priors / (1.0 - priors))
    probability = np.asarray(
        selected_model.predict(features, offsets),
        dtype=np.float64,
    )
    if (
        probability.shape != priors.shape
        or not np.all(np.isfinite(probability))
        or np.any((probability <= 0.0) | (probability >= 1.0))
    ):
        raise ValueError("Round 28 AI probability population differs")
    return probability


def _case_from_candidate(
    *,
    candidate: object,
    sample: _TargetFreeSample,
    role: str,
    selection_claim_sha256: str,
    model_name: str,
    model_sha256: str,
) -> Round28AICase:
    source_evidence = _canonical_sha256(
        {
            "ai_contract_sha256": POLYMARKET_ROUND28_AI_CONTRACT_SHA256,
            "condition_id": candidate.condition_id,
            "decision_time_ms": candidate.decision_time_ms,
            "decision_source_payload_sha256": (
                candidate.decision_source_payload_sha256
            ),
            "feature_row_sha256": sample.feature_row_sha256,
            "feature_source_chain_sha256": sample.source_chain_sha256,
            "selection_claim_sha256": selection_claim_sha256,
            "model_sha256": model_sha256,
            **{field: False for field in _NON_AUTHORITY_FLAGS[:4]},
        }
    )
    provisional = Round28AICase(
        partition_role=role,
        condition_id=candidate.condition_id,
        event_start_ms=candidate.event_start_ms,
        market_end_ms=candidate.market_end_ms,
        decision_time_ms=candidate.decision_time_ms,
        proposed_side=candidate.outcome,
        token_id=candidate.token_id,
        predicted_probability=candidate.predicted_probability,
        market_prior_probability_up=sample.market_prior_probability,
        quantity=_decimal_text(candidate.quantity),
        limit_price=_decimal_text(candidate.limit_price),
        decision_tick_size=_decimal_text(candidate.decision_tick_size),
        decision_average_price=_decimal_text(candidate.decision_average_price),
        decision_fee_quote=_decimal_text(candidate.decision_fee_quote),
        expected_edge_per_contract=_decimal_text(
            candidate.expected_edge_per_contract
        ),
        segment_id=candidate.segment_id,
        connection_id=candidate.connection_id,
        decision_book_event_id=candidate.decision_book_event_id,
        decision_source_payload_sha256=(
            candidate.decision_source_payload_sha256
        ),
        feature_row_sha256=sample.feature_row_sha256,
        feature_source_chain_sha256=sample.source_chain_sha256,
        selection_claim_sha256=selection_claim_sha256,
        model_name=model_name,
        model_feature_view="round28_bbo_augmented",
        model_sha256=model_sha256,
        causal_features=tuple(zip(POLYMARKET_ROUND28_FEATURE_NAMES, sample.values)),
        source_evidence_sha256=source_evidence,
        case_sha256="",
    )
    return replace(
        provisional,
        case_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def materialize_round28_ai_cases(
    *,
    role: str,
    rows: Sequence[Round28FeatureRow],
    selected_model: Round28ProbabilityModel,
    selection_claim_sha256: str,
    markets: Sequence[PolymarketFiveMinuteMarket],
    source_audit_sha256: str,
    config: Round27EconomicConfig,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batches: Iterable[Round27EconomicBookBatch] | None = None,
) -> Round28AICasePanel:
    """Freeze the exact augmented candidate cases without outcomes or targets."""

    if role not in {"selection", "sealed"}:
        raise ValueError("Round 28 AI case role differs")
    if selected_model.feature_view != "round28_bbo_augmented":
        raise ValueError("Round 28 AI requires the selected augmented model")
    model_payload = selected_model.asdict()
    model_sha256 = _sha256(model_payload.get("model_sha256"), name="model")
    selection_sha256 = _sha256(selection_claim_sha256, name="selection claim")
    cfg = config.validated()
    samples = _target_free_samples(rows)
    run_ids = {sample.run_id for sample in samples}
    if len(run_ids) != 1:
        raise ValueError("Round 28 AI feature source run differs")
    probability = _probabilities(samples, selected_model)
    conditions = {sample.condition_id for sample in samples}
    market_by_condition = {
        market.condition_id: market
        for market in markets
        if market.condition_id in conditions
    }
    if set(market_by_condition) != conditions or any(
        market_by_condition[sample.condition_id].asset != "BTC"
        or market_by_condition[sample.condition_id].event_start_ms
        != sample.event_start_ms
        for sample in samples
    ):
        raise ValueError("Round 28 AI market population differs")
    if (books is None) == (book_batches is None):
        raise ValueError("Round 28 AI cases require exactly one book source")
    batches: Iterable[Round27EconomicBookBatch] = (
        (
            Round27EconomicBookBatch(
                condition_ids=tuple(sorted(conditions)),
                books=tuple(
                    book
                    for book in books or ()
                    if book.market.condition_id in conditions
                ),
            ),
        )
        if books is not None
        else book_batches or ()
    )
    sample_indices_by_condition: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(samples):
        sample_indices_by_condition.setdefault(sample.condition_id, []).append(
            sample_index
        )
    seen_conditions: set[str] = set()
    cases: list[Round28AICase] = []
    reasons: dict[str, int] = {}
    candidate_population: list[dict[str, object]] = []
    for raw_batch in batches:
        if not isinstance(raw_batch, Round27EconomicBookBatch):
            raise ValueError("Round 28 AI book batch type differs")
        batch = raw_batch.validated()
        batch_conditions = set(batch.condition_ids)
        if (
            not batch_conditions <= conditions
            or batch_conditions & seen_conditions
            or len(batch_conditions) > cfg.maximum_conditions_per_book_batch
        ):
            raise ValueError("Round 28 AI book batch scope differs")
        seen_conditions.update(batch_conditions)
        sample_indices = sorted(
            sample_index
            for condition_id in batch.condition_ids
            for sample_index in sample_indices_by_condition[condition_id]
        )
        batch_samples = tuple(samples[index] for index in sample_indices)
        partition = _TargetFreePartition(role=role, samples=batch_samples)
        batch_probability = probability[np.asarray(sample_indices, dtype=np.int64)]
        index = _economics._BookIndex(batch.books)  # noqa: SLF001
        candidates, batch_reasons = _economics._build_candidates(  # noqa: SLF001
            partition,
            batch_probability,
            {
                condition_id: market_by_condition[condition_id]
                for condition_id in batch.condition_ids
            },
            index,
            cfg,
        )
        for reason, count in batch_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + count
        for candidate in candidates:
            sample = batch_samples[candidate.sample_index]
            cases.append(
                _case_from_candidate(
                    candidate=candidate,
                    sample=sample,
                    role=role,
                    selection_claim_sha256=selection_sha256,
                    model_name=selected_model.model_name,
                    model_sha256=model_sha256,
                )
            )
            candidate_population.append(
                {
                    "condition_id": candidate.condition_id,
                    "decision_time_ms": candidate.decision_time_ms,
                    "outcome": candidate.outcome,
                    "limit_price": _decimal_text(candidate.limit_price),
                    "quantity": _decimal_text(candidate.quantity),
                }
            )
    if seen_conditions != conditions:
        raise ValueError("Round 28 AI book batches do not cover the role")
    ordered_cases = tuple(
        sorted(
            cases,
            key=lambda case: (
                case.event_start_ms,
                case.condition_id,
                case.decision_time_ms,
            ),
        )
    )
    event_start_by_condition = {
        case.condition_id: case.event_start_ms for case in ordered_cases
    }
    candidate_population.sort(
        key=lambda item: (
            event_start_by_condition[str(item["condition_id"])],
            str(item["condition_id"]),
            int(item["decision_time_ms"]),
        )
    )
    provisional = Round28AICasePanel(
        partition_role=role,
        source_run_id=next(iter(run_ids)),
        model_name=selected_model.model_name,
        model_sha256=model_sha256,
        selection_claim_sha256=selection_sha256,
        source_audit_sha256=_sha256(source_audit_sha256, name="source audit"),
        economic_config=cfg.asdict(),
        evaluated_condition_count=len(conditions),
        evaluated_condition_ids_sha256=_canonical_sha256(sorted(conditions)),
        baseline_candidate_population_sha256=_canonical_sha256(
            candidate_population
        ),
        selection_reason_counts=dict(sorted(reasons.items())),
        cases=ordered_cases,
        panel_sha256="",
    )
    return replace(
        provisional,
        panel_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def round28_ai_case_from_mapping(value: Mapping[str, object]) -> Round28AICase:
    payload = dict(value)
    expected = {
        *Round28AICase.__dataclass_fields__,
        "schema_version",
        "ai_contract_sha256",
        "feature_names_sha256",
        *_NON_AUTHORITY_FLAGS,
    }
    features = payload.get("causal_features")
    if (
        set(payload) != expected
        or payload.get("schema_version") != POLYMARKET_ROUND28_AI_CASE_SCHEMA_VERSION
        or payload.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or payload.get("feature_names_sha256")
        != POLYMARKET_ROUND28_FEATURE_NAMES_SHA256
        or any(payload.get(field) is not False for field in _NON_AUTHORITY_FLAGS)
        or not isinstance(features, list)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "value"}
            or not isinstance(item.get("name"), str)
            or isinstance(item.get("value"), bool)
            or not isinstance(item.get("value"), (int, float))
            for item in features
        )
    ):
        raise ValueError("Round 28 persisted AI case differs")
    constructor = {
        key: payload[key]
        for key in Round28AICase.__dataclass_fields__
        if key != "causal_features"
    }
    return Round28AICase(
        **constructor,
        causal_features=tuple(
            (str(item["name"]), float(item["value"])) for item in features
        ),
    ).validated()


def round28_ai_case_panel_from_mapping(
    value: Mapping[str, object],
) -> Round28AICasePanel:
    payload = dict(value)
    expected = {
        *Round28AICasePanel.__dataclass_fields__,
        "schema_version",
        "ai_contract_sha256",
        "model_feature_view",
        "case_sha256",
        "case_count",
        "prompt_population_sha256",
        *_NON_AUTHORITY_FLAGS,
    }
    raw_cases = payload.get("cases")
    if (
        set(payload) != expected
        or payload.get("schema_version")
        != POLYMARKET_ROUND28_AI_CASE_PANEL_SCHEMA_VERSION
        or payload.get("ai_contract_sha256")
        != POLYMARKET_ROUND28_AI_CONTRACT_SHA256
        or payload.get("model_feature_view") != "round28_bbo_augmented"
        or any(payload.get(field) is not False for field in _NON_AUTHORITY_FLAGS)
        or not isinstance(raw_cases, list)
        or not isinstance(payload.get("economic_config"), Mapping)
        or not isinstance(payload.get("selection_reason_counts"), Mapping)
        or type(payload.get("evaluated_condition_count")) is not int
        or type(payload.get("case_count")) is not int
        or not isinstance(payload.get("case_sha256"), list)
    ):
        raise ValueError("Round 28 persisted AI case panel differs")
    cases = tuple(
        round28_ai_case_from_mapping(item)
        for item in raw_cases
        if isinstance(item, Mapping)
    )
    if (
        len(cases) != len(raw_cases)
        or payload["case_count"] != len(cases)
        or payload["case_sha256"] != [case.case_sha256 for case in cases]
    ):
        raise ValueError("Round 28 persisted AI case population differs")
    panel = Round28AICasePanel(
        partition_role=str(payload["partition_role"]),
        source_run_id=str(payload["source_run_id"]),
        model_name=str(payload["model_name"]),
        model_sha256=str(payload["model_sha256"]),
        selection_claim_sha256=str(payload["selection_claim_sha256"]),
        source_audit_sha256=str(payload["source_audit_sha256"]),
        economic_config=dict(payload["economic_config"]),
        evaluated_condition_count=int(payload["evaluated_condition_count"]),
        evaluated_condition_ids_sha256=str(
            payload["evaluated_condition_ids_sha256"]
        ),
        baseline_candidate_population_sha256=str(
            payload["baseline_candidate_population_sha256"]
        ),
        selection_reason_counts={
            str(key): int(count)
            for key, count in payload["selection_reason_counts"].items()
        },
        cases=cases,
        panel_sha256=str(payload["panel_sha256"]),
    ).validated()
    if (
        payload["prompt_population_sha256"]
        != panel.identity_payload()["prompt_population_sha256"]
    ):
        raise ValueError("Round 28 persisted AI prompt population differs")
    return panel


__all__ = [
    "POLYMARKET_ROUND28_AI_CASE_PANEL_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_AI_CASE_SCHEMA_VERSION",
    "Round28AICase",
    "Round28AICasePanel",
    "materialize_round28_ai_cases",
    "round28_ai_case_from_mapping",
    "round28_ai_case_panel_from_mapping",
    "round28_ai_case_prompt",
]
