"""Target-free Round 27 AI case materialization from frozen baseline candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np

from . import polymarket_round27_economics as _economics
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_ai_ablation_contract import (
    POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256,
)
from .polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
)
from .polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    Round27FeatureRow,
)
from .polymarket_round27_model import Round27ProbabilityModel


POLYMARKET_ROUND27_AI_CASE_SCHEMA_VERSION = "polymarket-round27-ai-case-v1"
POLYMARKET_ROUND27_AI_CASE_PANEL_SCHEMA_VERSION = (
    "polymarket-round27-ai-case-panel-v1"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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
        raise ValueError(f"Round 27 AI {name} SHA-256 differs")
    return selected


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class _TargetFreeSample:
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
class Round27AICase:
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
    decision_average_price: str
    decision_fee_quote: str
    expected_edge_per_contract: str
    segment_id: str
    connection_id: str
    decision_book_event_id: str
    decision_source_payload_sha256: str
    feature_row_sha256: str
    feature_source_chain_sha256: str
    model_name: str
    model_sha256: str
    causal_features: tuple[tuple[str, float], ...]
    source_evidence_sha256: str
    case_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND27_AI_CASE_SCHEMA_VERSION,
            "ablation_contract_sha256": (
                POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
            ),
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
            "decision_average_price": self.decision_average_price,
            "decision_fee_quote": self.decision_fee_quote,
            "expected_edge_per_contract": self.expected_edge_per_contract,
            "segment_id": self.segment_id,
            "connection_id": self.connection_id,
            "decision_book_event_id": self.decision_book_event_id,
            "decision_source_payload_sha256": self.decision_source_payload_sha256,
            "feature_row_sha256": self.feature_row_sha256,
            "feature_source_chain_sha256": self.feature_source_chain_sha256,
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "causal_features": [
                {"name": name, "value": value}
                for name, value in self.causal_features
            ],
            "target_accessed": False,
            "outcome_accessed": False,
            "future_books_accessed": False,
            "pnl_accessed": False,
            "credentials_used": False,
            "orders_submitted": False,
            "trading_authority": False,
            "source_evidence_sha256": self.source_evidence_sha256,
        }

    def validated(self) -> "Round27AICase":
        names = tuple(name for name, _value in self.causal_features)
        values = tuple(value for _name, value in self.causal_features)
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
            or any(not Decimal(value).is_finite() for value in (
                self.quantity,
                self.limit_price,
                self.decision_average_price,
                self.decision_fee_quote,
                self.expected_edge_per_contract,
            ))
            or Decimal(self.quantity) <= 0
            or Decimal(self.limit_price) <= 0
            or Decimal(self.decision_average_price) <= 0
            or Decimal(self.decision_fee_quote) < 0
            or Decimal(self.expected_edge_per_contract) <= 0
            or not self.segment_id
            or not self.connection_id
            or names != POLYMARKET_ROUND27_FEATURE_NAMES
            or any(not math.isfinite(value) for value in values)
            or _sha256(self.decision_source_payload_sha256, name="decision source")
            != self.decision_source_payload_sha256
            or _sha256(self.feature_row_sha256, name="feature row")
            != self.feature_row_sha256
            or _sha256(self.feature_source_chain_sha256, name="feature source chain")
            != self.feature_source_chain_sha256
            or _sha256(self.model_sha256, name="model") != self.model_sha256
            or _sha256(self.source_evidence_sha256, name="source evidence")
            != self.source_evidence_sha256
            or self.case_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 27 AI case differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "case_sha256": self.case_sha256}


@dataclass(frozen=True, slots=True)
class Round27AICasePanel:
    partition_role: str
    model_name: str
    model_sha256: str
    source_audit_sha256: str
    economic_config: dict[str, object]
    evaluated_condition_count: int
    evaluated_condition_ids_sha256: str
    baseline_candidate_population_sha256: str
    selection_reason_counts: dict[str, int]
    cases: tuple[Round27AICase, ...]
    panel_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND27_AI_CASE_PANEL_SCHEMA_VERSION,
            "ablation_contract_sha256": (
                POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
            ),
            "partition_role": self.partition_role,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
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
            "prompt_population_sha256": _canonical_sha256(
                [round27_ai_case_prompt(case) for case in self.cases]
            ),
            "target_accessed": False,
            "outcome_accessed": False,
            "future_books_accessed": False,
            "pnl_accessed": False,
            "credentials_used": False,
            "orders_submitted": False,
            "trading_authority": False,
        }

    def validated(self) -> "Round27AICasePanel":
        cases = tuple(case.validated() for case in self.cases)
        condition_ids = tuple(case.condition_id for case in cases)
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
        if (
            self.partition_role not in {"selection", "sealed"}
            or not self.model_name
            or _sha256(self.model_sha256, name="model") != self.model_sha256
            or _sha256(self.source_audit_sha256, name="source audit")
            != self.source_audit_sha256
            or self.evaluated_condition_count < len(cases)
            or _sha256(
                self.evaluated_condition_ids_sha256,
                name="evaluated condition population",
            )
            != self.evaluated_condition_ids_sha256
            or _sha256(
                self.baseline_candidate_population_sha256,
                name="candidate population",
            )
            != self.baseline_candidate_population_sha256
            or len(condition_ids) != len(set(condition_ids))
            or cases != ordered
            or any(case.partition_role != self.partition_role for case in cases)
            or any(
                (case.model_name, case.model_sha256)
                != (self.model_name, self.model_sha256)
                for case in cases
            )
            or self.panel_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 27 AI case panel differs")
        return self

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "cases": [case.asdict() for case in self.cases],
            "panel_sha256": self.panel_sha256,
        }


def round27_ai_case_prompt(case: Round27AICase) -> str:
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
        "minimum-size FOK proposal. You cannot predict price, create a trade, "
        "increase size, change side, or override deterministic controls. Use "
        "only the causal packet. Reject or reduce when liquidity, spread/cost, "
        "staleness/gaps, cross-venue disagreement, volatility/jumps, model "
        "disagreement, late horizon, or insufficient evidence creates material "
        "risk. The proposal is already at venue minimum size, so reduce is "
        "executed as abstain. Return unchanged only when no material risk is "
        "present. Return only the required JSON object. case="
        + _canonical_json(packet)
    )


def _target_free_samples(
    rows: Sequence[Round27FeatureRow],
) -> tuple[_TargetFreeSample, ...]:
    selected_rows = tuple(row.validated() for row in rows)
    if not selected_rows:
        raise ValueError("Round 27 AI feature population is empty")
    keys = tuple((row.condition_id, row.decision_time_ms) for row in selected_rows)
    if len(keys) != len(set(keys)):
        raise ValueError("Round 27 AI feature population is duplicated")
    return tuple(
        _TargetFreeSample(
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
    selected_model: Round27ProbabilityModel | None,
) -> np.ndarray:
    features = np.asarray([sample.values for sample in samples], dtype=np.float64)
    priors = np.asarray(
        [sample.market_prior_probability for sample in samples],
        dtype=np.float64,
    )
    offsets = np.log(priors / (1.0 - priors))
    probability = (
        priors
        if selected_model is None
        else np.asarray(selected_model.predict(features, offsets), dtype=np.float64)
    )
    if (
        probability.shape != priors.shape
        or not np.all(np.isfinite(probability))
        or np.any((probability <= 0.0) | (probability >= 1.0))
    ):
        raise ValueError("Round 27 AI probability population differs")
    return probability


def _case_from_candidate(
    *,
    candidate: object,
    sample: _TargetFreeSample,
    role: str,
    model_name: str,
    model_sha256: str,
) -> Round27AICase:
    source_evidence = _canonical_sha256(
        {
            "ablation_contract_sha256": (
                POLYMARKET_ROUND27_AI_ABLATION_CONTRACT_SHA256
            ),
            "condition_id": candidate.condition_id,
            "decision_time_ms": candidate.decision_time_ms,
            "decision_source_payload_sha256": (
                candidate.decision_source_payload_sha256
            ),
            "feature_row_sha256": sample.feature_row_sha256,
            "feature_source_chain_sha256": sample.source_chain_sha256,
            "model_sha256": model_sha256,
            "target_accessed": False,
            "outcome_accessed": False,
            "future_books_accessed": False,
        }
    )
    provisional = Round27AICase(
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
        model_name=model_name,
        model_sha256=model_sha256,
        causal_features=tuple(zip(POLYMARKET_ROUND27_FEATURE_NAMES, sample.values)),
        source_evidence_sha256=source_evidence,
        case_sha256="",
    )
    return replace(
        provisional,
        case_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def materialize_round27_ai_cases(
    *,
    role: str,
    rows: Sequence[Round27FeatureRow],
    selected_model: Round27ProbabilityModel | None,
    model_name: str,
    model_sha256: str,
    markets: Sequence[PolymarketFiveMinuteMarket],
    source_audit_sha256: str,
    config: Round27EconomicConfig,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batches: Iterable[Round27EconomicBookBatch] | None = None,
) -> Round27AICasePanel:
    """Materialize one target-free case for every frozen baseline candidate."""

    if role not in {"selection", "sealed"}:
        raise ValueError("Round 27 AI case role differs")
    cfg = config.validated()
    samples = _target_free_samples(rows)
    if selected_model is None:
        if model_name != "market_prior":
            raise ValueError("Round 27 AI model identity differs")
    else:
        model_payload = selected_model.asdict()
        if (
            model_name != selected_model.model_name
            or model_payload.get("model_sha256") != model_sha256
        ):
            raise ValueError("Round 27 AI model identity differs")
    probability = _probabilities(samples, selected_model)
    conditions = {sample.condition_id for sample in samples}
    market_by_condition = {
        market.condition_id: market
        for market in markets
        if market.condition_id in conditions
    }
    if set(market_by_condition) != conditions:
        raise ValueError("Round 27 AI market population differs")
    if any(
        market_by_condition[sample.condition_id].asset != "BTC"
        or market_by_condition[sample.condition_id].event_start_ms
        != sample.event_start_ms
        for sample in samples
    ):
        raise ValueError("Round 27 AI market metadata differs")
    if (books is None) == (book_batches is None):
        raise ValueError("Round 27 AI cases require exactly one book source")
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
    for index, sample in enumerate(samples):
        sample_indices_by_condition.setdefault(sample.condition_id, []).append(index)
    seen_conditions: set[str] = set()
    cases: list[Round27AICase] = []
    reasons: dict[str, int] = {}
    candidate_population: list[dict[str, object]] = []
    for raw_batch in batches:
        if not isinstance(raw_batch, Round27EconomicBookBatch):
            raise ValueError("Round 27 AI book batch type differs")
        batch = raw_batch.validated()
        batch_conditions = set(batch.condition_ids)
        if (
            not batch_conditions <= conditions
            or batch_conditions & seen_conditions
            or len(batch_conditions) > cfg.maximum_conditions_per_book_batch
        ):
            raise ValueError("Round 27 AI book batch scope differs")
        seen_conditions.update(batch_conditions)
        sample_indices = sorted(
            index
            for condition_id in batch.condition_ids
            for index in sample_indices_by_condition[condition_id]
        )
        batch_samples = tuple(samples[index] for index in sample_indices)
        batch_partition = _TargetFreePartition(role=role, samples=batch_samples)
        batch_probability = probability[np.asarray(sample_indices, dtype=np.int64)]
        batch_markets = {
            condition_id: market_by_condition[condition_id]
            for condition_id in batch.condition_ids
        }
        index = _economics._BookIndex(batch.books)  # noqa: SLF001
        candidates, batch_reasons = _economics._build_candidates(  # noqa: SLF001
            batch_partition,
            batch_probability,
            batch_markets,
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
                    model_name=model_name,
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
        raise ValueError("Round 27 AI book batches do not cover the role")
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
    provisional = Round27AICasePanel(
        partition_role=role,
        model_name=str(model_name),
        model_sha256=_sha256(model_sha256, name="model"),
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


__all__ = [
    "POLYMARKET_ROUND27_AI_CASE_PANEL_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_AI_CASE_SCHEMA_VERSION",
    "Round27AICase",
    "Round27AICasePanel",
    "materialize_round27_ai_cases",
    "round27_ai_case_prompt",
]
