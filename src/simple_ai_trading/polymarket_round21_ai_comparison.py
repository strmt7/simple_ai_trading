"""Matched 81-ledger economic gate for the Round 21 local AI veto."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import re
from typing import Sequence

from .polymarket_ai_veto import PolymarketAIVetoCase, PolymarketAIVetoReport
from .polymarket_round21_ai import (
    POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
    round21_permissions_from_ai_report,
)
from .polymarket_round21_comparison import (
    Round21MatchedReplayDelta,
    paired_round21_replay_delta,
    round21_replay_matrix_sha256,
)
from .polymarket_round21_replay import (
    POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
    Round21EconomicReplay,
    Round21ReplayCondition,
    replay_round21_full_matrix,
)


POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION = (
    "polymarket-round21-ai-matched-economic-comparison-v1"
)
POLYMARKET_ROUND21_AI_MINIMUM_MATCHED_DECISIONS = 300
POLYMARKET_ROUND21_AI_MINIMUM_NON_TIED_ACTIONS = 30
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


@dataclass(frozen=True, slots=True)
class Round21AIMatchedComparison:
    model: str
    model_digest: str
    ai_report_sha256: str
    matched_population_sha256: str
    baseline_matrix_sha256: str
    ai_matrix_sha256: str
    ai_permission_root_sha256: str
    matched_decision_count: int
    non_tied_primary_action_count: int
    deltas: tuple[Round21MatchedReplayDelta, ...]
    all_replays_accepted: bool
    development_qualified: bool
    ai_model_selected: bool
    comparison_sha256: str
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
            "ai_veto_design_sha256": POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
            "economic_replay_design_sha256": (
                POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256
            ),
            "model": self.model,
            "model_digest": self.model_digest,
            "ai_report_sha256": self.ai_report_sha256,
            "matched_population_sha256": self.matched_population_sha256,
            "baseline_matrix_sha256": self.baseline_matrix_sha256,
            "ai_matrix_sha256": self.ai_matrix_sha256,
            "ai_permission_root_sha256": self.ai_permission_root_sha256,
            "matched_decision_count": self.matched_decision_count,
            "minimum_matched_decisions": (
                POLYMARKET_ROUND21_AI_MINIMUM_MATCHED_DECISIONS
            ),
            "non_tied_primary_action_count": (
                self.non_tied_primary_action_count
            ),
            "minimum_non_tied_primary_actions": (
                POLYMARKET_ROUND21_AI_MINIMUM_NON_TIED_ACTIONS
            ),
            "delta_sha256": [value.delta_sha256 for value in self.deltas],
            "all_replays_accepted": self.all_replays_accepted,
            "development_qualified": self.development_qualified,
            "ai_model_selected": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self) -> Round21AIMatchedComparison:
        deltas = tuple(value.validated() for value in self.deltas)
        expected_qualified = (
            self.all_replays_accepted
            and self.matched_decision_count
            >= POLYMARKET_ROUND21_AI_MINIMUM_MATCHED_DECISIONS
            and self.non_tied_primary_action_count
            >= POLYMARKET_ROUND21_AI_MINIMUM_NON_TIED_ACTIONS
        )
        if (
            not self.model
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.model_digest,
                    self.ai_report_sha256,
                    self.matched_population_sha256,
                    self.baseline_matrix_sha256,
                    self.ai_matrix_sha256,
                    self.ai_permission_root_sha256,
                )
            )
            or self.matched_decision_count < 1
            or not 0
            <= self.non_tied_primary_action_count
            <= self.matched_decision_count
            or len(deltas) != 81
            or len({(value.profile, value.scenario) for value in deltas}) != 81
            or self.all_replays_accepted != all(value.accepted for value in deltas)
            or self.development_qualified != expected_qualified
            or self.ai_model_selected
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.comparison_sha256
            != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 AI matched comparison differs")
        return self


def _primary_actions(
    matrix: Sequence[Round21EconomicReplay],
) -> dict[tuple[str, int], str]:
    matches = tuple(
        value.validated()
        for value in matrix
        if value.profile == "conservative" and value.scenario == "primary"
    )
    if len(matches) != 1:
        raise ValueError("Round 21 AI primary replay is unavailable")
    actions: dict[tuple[str, int], str] = {}
    for condition in matches[0].conditions:
        for step in condition.steps:
            key = (condition.condition_id, step.decision_time_ms)
            if key in actions:
                raise ValueError("Round 21 AI primary action is duplicated")
            actions[key] = step.action
    return actions


def compare_round21_ai_full_matrix(
    *,
    conditions: Sequence[Round21ReplayCondition],
    cases: Sequence[PolymarketAIVetoCase],
    report: PolymarketAIVetoReport,
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
) -> Round21AIMatchedComparison:
    """Compare the deterministic policy with delayed veto-only AI permissions."""

    selected_conditions = tuple(value.validated() for value in conditions)
    if not selected_conditions:
        raise ValueError("Round 21 AI matched population is empty")
    condition_ids = tuple(
        value.market.condition_id for value in selected_conditions
    )
    if (
        len(set(condition_ids)) != len(condition_ids)
        or set(condition_ids) != {value.condition_id for value in cases}
    ):
        raise ValueError("Round 21 AI matched population differs")
    decision_keys = tuple(
        (condition.market.condition_id, envelope.decision_time_ms)
        for condition in selected_conditions
        for envelope in condition.envelopes
    )
    if len(set(decision_keys)) != len(decision_keys):
        raise ValueError("Round 21 AI matched decisions are duplicated")
    permissions = round21_permissions_from_ai_report(cases=cases, report=report)
    baseline_matrix = replay_round21_full_matrix(
        selected_conditions,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
    )
    ai_matrix = replay_round21_full_matrix(
        selected_conditions,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        directional_permissions=permissions,
    )
    if len(baseline_matrix) != 81 or len(ai_matrix) != 81:
        raise ValueError("Round 21 AI replay matrix differs")
    permission_roots = {
        value.directional_permission_root_sha256 for value in ai_matrix
    }
    if len(permission_roots) != 1:
        raise ValueError("Round 21 AI permission root differs")
    deltas = tuple(
        paired_round21_replay_delta(left, right)
        for left, right in zip(baseline_matrix, ai_matrix, strict=True)
    )
    baseline_actions = _primary_actions(baseline_matrix)
    ai_actions = _primary_actions(ai_matrix)
    if set(baseline_actions) != set(ai_actions):
        raise ValueError("Round 21 AI primary action population differs")
    non_tied_actions = sum(
        baseline_actions[key] != ai_actions[key] for key in baseline_actions
    )
    all_replays_accepted = all(value.accepted for value in deltas)
    development_qualified = (
        all_replays_accepted
        and len(decision_keys) >= POLYMARKET_ROUND21_AI_MINIMUM_MATCHED_DECISIONS
        and non_tied_actions
        >= POLYMARKET_ROUND21_AI_MINIMUM_NON_TIED_ACTIONS
    )
    matched_population_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION,
            "condition_sha256": [
                value.matched_population_sha256()
                for value in selected_conditions
            ],
        }
    )
    provisional = Round21AIMatchedComparison(
        model=report.config.model,
        model_digest=report.model_digest,
        ai_report_sha256=report.report_sha256,
        matched_population_sha256=matched_population_sha256,
        baseline_matrix_sha256=round21_replay_matrix_sha256(baseline_matrix),
        ai_matrix_sha256=round21_replay_matrix_sha256(ai_matrix),
        ai_permission_root_sha256=next(iter(permission_roots)),
        matched_decision_count=len(decision_keys),
        non_tied_primary_action_count=non_tied_actions,
        deltas=deltas,
        all_replays_accepted=all_replays_accepted,
        development_qualified=development_qualified,
        ai_model_selected=False,
        comparison_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        comparison_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_AI_COMPARISON_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_AI_MINIMUM_MATCHED_DECISIONS",
    "POLYMARKET_ROUND21_AI_MINIMUM_NON_TIED_ACTIONS",
    "Round21AIMatchedComparison",
    "compare_round21_ai_full_matrix",
]
