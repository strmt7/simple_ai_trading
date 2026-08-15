"""Finite development selection for the Round 21 local AI veto candidates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import re
from typing import Mapping, Sequence

from .polymarket_round21_ai_comparison import Round21AIMatchedComparison
from .polymarket_round21_execution import POLYMARKET_ROUND21_EXECUTION_SCENARIOS
from .polymarket_round21_policy import POLYMARKET_ROUND21_RISK_PROFILES


POLYMARKET_ROUND21_AI_SELECTION_SCHEMA_VERSION = (
    "polymarket-round21-ai-candidate-selection-v1"
)
POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-ai-candidate-selection-design-v7"
)
POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256 = (
    "5a5b9e38322cdb7031e3eff2d5b3b4e5d00218c20d838efcaf68fde7461ce81a"
)
POLYMARKET_ROUND21_AI_CANDIDATES = (
    "qwen3.5:9b",
    "fin-r1:8b",
    "fino1:8b",
)
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


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 AI {name} is invalid")
    return selected


def validate_round21_ai_candidate_selection_design(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Reject changes to the target-blind AI candidate-selection contract."""

    design = dict(value)
    claimed = _digest(design.pop("design_sha256", ""), name="selection design")
    parents = design.get("parents")
    program = design.get("candidate_program")
    supersession = design.get("supersession")
    admission = design.get("admission")
    ranking = design.get("ranking")
    semantics = design.get("semantics")
    authority = design.get("authority")
    if (
        set(design)
        != {
            "schema_version",
            "round",
            "status",
            "parents",
            "supersession",
            "candidate_program",
            "admission",
            "ranking",
            "semantics",
            "authority",
        }
        or claimed != POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256
        or claimed != _canonical_sha256(design)
        or design.get("schema_version")
        != POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SCHEMA_VERSION
        or design.get("round") != 21
        or design.get("status")
        != (
            "preregistered_during_capture_before_capture_target_or_model_outcome_access"
        )
        or parents
        != {
            "round21_ai_candidate_selection_design_v6_sha256": (
                "0ca13da7e3210b219357897bb2c3e50ded9745ca8aa748fe890ba90cf8cda33a"
            ),
            "round21_contract_sha256": (
                "6aadbce31c175438c40c6a1204383d828fd78ddef93b280aa2f999f347669116"
            ),
            "round21_ai_veto_design_sha256": (
                "65ebb61934cac403af4369e862d851db09126efbe5d35e4b091d851ee251d7c5"
            ),
            "round21_model_design_sha256": (
                "ee81fb773028ffeedb62f34a56e1741075f22d68429c94d1e50e55c3cdc5563e"
            ),
        }
        or supersession
        != {
            "change": ("bind_unchanged_finite_ai_selection_to_receipt_age_features"),
            "candidate_count_changed": False,
            "candidate_identities_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
            "selection_or_ranking_changed": False,
        }
        or program
        != {
            "models": list(POLYMARKET_ROUND21_AI_CANDIDATES),
            "exactly_one_report_per_model": True,
            "immutable_weight_digest_required": True,
            "duplicate_weight_digest_rejected": True,
            "same_matched_population_required": True,
            "same_deterministic_baseline_matrix_required": True,
            "same_81_profile_scenario_ledgers_required": True,
        }
        or admission
        != {
            "development_qualified_required": True,
            "all_81_after_cost_ledgers_must_pass": True,
            "minimum_matched_decisions": 300,
            "minimum_non_tied_primary_actions": 30,
            "failed_candidates_remain_in_audit_artifact": True,
        }
        or ranking
        != {
            "scope": "development_only_before_one_use_sealed_test",
            "primary": (
                "maximize_minimum_daily_mean_delta_lower_95_quote_across_81_ledgers"
            ),
            "secondary": (
                "maximize_minimum_mean_condition_utility_delta_quote_across_81_ledgers"
            ),
            "tertiary": ("minimize_maximum_drawdown_delta_fraction_across_81_ledgers"),
            "quaternary": ("maximize_minimum_tail_mean_delta_quote_across_81_ledgers"),
            "final_tie_break": "ascending_exact_model_name",
            "post_nomination_development_retuning": False,
        }
        or semantics
        != {
            "output": ("one_development_nominated_sealed_test_challenger_or_none"),
            "nomination_is_model_selection": False,
            "nomination_is_ai_edge_claim": False,
            "nomination_is_profitability_claim": False,
            "nomination_grants_execution_authority": False,
        }
        or authority
        != {
            "capture_data_accessed": False,
            "test_targets_accessed": False,
            "ai_model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 21 AI candidate-selection design differs")
    return {**design, "design_sha256": claimed}


@dataclass(frozen=True, slots=True)
class Round21AICandidateScore:
    model: str
    model_digest: str
    comparison_sha256: str
    development_qualified: bool
    rejection_reasons: tuple[str, ...]
    minimum_daily_mean_delta_lower_95_quote: Decimal | None
    minimum_mean_condition_utility_delta_quote: Decimal
    maximum_drawdown_delta_fraction: Decimal
    minimum_tail_mean_delta_quote: Decimal
    score_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_AI_SELECTION_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256,
            "model": self.model,
            "model_digest": self.model_digest,
            "comparison_sha256": self.comparison_sha256,
            "development_qualified": self.development_qualified,
            "rejection_reasons": list(self.rejection_reasons),
            "minimum_daily_mean_delta_lower_95_quote": (
                None
                if self.minimum_daily_mean_delta_lower_95_quote is None
                else format(
                    self.minimum_daily_mean_delta_lower_95_quote,
                    "f",
                )
            ),
            "minimum_mean_condition_utility_delta_quote": format(
                self.minimum_mean_condition_utility_delta_quote,
                "f",
            ),
            "maximum_drawdown_delta_fraction": format(
                self.maximum_drawdown_delta_fraction,
                "f",
            ),
            "minimum_tail_mean_delta_quote": format(
                self.minimum_tail_mean_delta_quote,
                "f",
            ),
        }

    def validated(self) -> Round21AICandidateScore:
        lower = self.minimum_daily_mean_delta_lower_95_quote
        decimals = (
            self.minimum_mean_condition_utility_delta_quote,
            self.maximum_drawdown_delta_fraction,
            self.minimum_tail_mean_delta_quote,
        )
        if (
            self.model not in POLYMARKET_ROUND21_AI_CANDIDATES
            or _digest(self.model_digest, name="model digest") != self.model_digest
            or _digest(self.comparison_sha256, name="comparison digest")
            != self.comparison_sha256
            or any(not value.is_finite() for value in decimals)
            or (lower is not None and not lower.is_finite())
            or len(set(self.rejection_reasons)) != len(self.rejection_reasons)
            or self.development_qualified != (not self.rejection_reasons)
            or (
                self.development_qualified
                and (
                    lower is None
                    or lower <= 0
                    or self.minimum_mean_condition_utility_delta_quote <= 0
                    or self.maximum_drawdown_delta_fraction > 0
                    or self.minimum_tail_mean_delta_quote < 0
                )
            )
            or self.score_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 AI candidate score differs")
        return self


def _candidate_score(
    comparison: Round21AIMatchedComparison,
) -> Round21AICandidateScore:
    selected = comparison.validated()
    lower_values = tuple(
        delta.daily_mean_delta_lower_95_quote for delta in selected.deltas
    )
    lower = (
        None
        if any(value is None for value in lower_values)
        else min(value for value in lower_values if value is not None)
    )
    reasons: list[str] = []
    if not selected.all_replays_accepted:
        reasons.append("one_or_more_after_cost_ledgers_rejected")
    if selected.matched_decision_count < 300:
        reasons.append("matched_decision_count_below_300")
    if selected.non_tied_primary_action_count < 30:
        reasons.append("non_tied_primary_action_count_below_30")
    if selected.development_qualified != (not reasons):
        raise ValueError("Round 21 AI development qualification differs")
    provisional = Round21AICandidateScore(
        model=selected.model,
        model_digest=selected.model_digest,
        comparison_sha256=selected.comparison_sha256,
        development_qualified=selected.development_qualified,
        rejection_reasons=tuple(reasons),
        minimum_daily_mean_delta_lower_95_quote=lower,
        minimum_mean_condition_utility_delta_quote=min(
            value.mean_condition_utility_delta_quote for value in selected.deltas
        ),
        maximum_drawdown_delta_fraction=max(
            value.maximum_drawdown_delta_fraction for value in selected.deltas
        ),
        minimum_tail_mean_delta_quote=min(
            value.tail_mean_delta_quote for value in selected.deltas
        ),
        score_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        score_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _ranking_key(
    score: Round21AICandidateScore,
) -> tuple[Decimal, Decimal, Decimal, Decimal, str]:
    selected = score.validated()
    lower = selected.minimum_daily_mean_delta_lower_95_quote
    if not selected.development_qualified or lower is None:
        raise ValueError("Round 21 AI candidate is not rankable")
    return (
        -lower,
        -selected.minimum_mean_condition_utility_delta_quote,
        selected.maximum_drawdown_delta_fraction,
        -selected.minimum_tail_mean_delta_quote,
        selected.model,
    )


@dataclass(frozen=True, slots=True)
class Round21AICandidateSelection:
    matched_population_sha256: str
    baseline_matrix_sha256: str
    scores: tuple[Round21AICandidateScore, ...]
    qualified_candidate_count: int
    nominated_model: str | None
    nominated_model_digest: str | None
    nominated_comparison_sha256: str | None
    selection_sha256: str
    ai_model_selected: bool = False
    ai_edge_claim: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_AI_SELECTION_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256,
            "matched_population_sha256": self.matched_population_sha256,
            "baseline_matrix_sha256": self.baseline_matrix_sha256,
            "score_sha256": [value.score_sha256 for value in self.scores],
            "qualified_candidate_count": self.qualified_candidate_count,
            "nominated_model": self.nominated_model,
            "nominated_model_digest": self.nominated_model_digest,
            "nominated_comparison_sha256": self.nominated_comparison_sha256,
            "ai_model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self) -> Round21AICandidateSelection:
        scores = tuple(value.validated() for value in self.scores)
        qualified = tuple(value for value in scores if value.development_qualified)
        nominated = None if not qualified else min(qualified, key=_ranking_key)
        expected = (
            (None, None, None)
            if nominated is None
            else (
                nominated.model,
                nominated.model_digest,
                nominated.comparison_sha256,
            )
        )
        actual = (
            self.nominated_model,
            self.nominated_model_digest,
            self.nominated_comparison_sha256,
        )
        if (
            _digest(self.matched_population_sha256, name="population digest")
            != self.matched_population_sha256
            or _digest(self.baseline_matrix_sha256, name="baseline digest")
            != self.baseline_matrix_sha256
            or tuple(value.model for value in scores)
            != POLYMARKET_ROUND21_AI_CANDIDATES
            or len({value.model_digest for value in scores}) != len(scores)
            or self.qualified_candidate_count != len(qualified)
            or actual != expected
            or self.ai_model_selected
            or self.ai_edge_claim
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.selection_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 AI candidate selection differs")
        return self


def select_round21_ai_candidate(
    comparisons: Sequence[Round21AIMatchedComparison],
) -> Round21AICandidateSelection:
    """Nominate one development-qualified model for the one-use sealed test."""

    selected = tuple(value.validated() for value in comparisons)
    by_model = {value.model: value for value in selected}
    if (
        len(selected) != len(POLYMARKET_ROUND21_AI_CANDIDATES)
        or len(by_model) != len(selected)
        or set(by_model) != set(POLYMARKET_ROUND21_AI_CANDIDATES)
        or len({value.model_digest for value in selected}) != len(selected)
        or len({value.matched_population_sha256 for value in selected}) != 1
        or len({value.baseline_matrix_sha256 for value in selected}) != 1
    ):
        raise ValueError("Round 21 AI candidate population differs")
    expected_ledgers = {
        (profile.name, scenario.name)
        for profile in POLYMARKET_ROUND21_RISK_PROFILES
        for scenario in POLYMARKET_ROUND21_EXECUTION_SCENARIOS
    }
    if any(
        {(delta.profile, delta.scenario) for delta in value.deltas} != expected_ledgers
        for value in selected
    ):
        raise ValueError("Round 21 AI candidate ledger population differs")
    scores = tuple(
        _candidate_score(by_model[model]) for model in POLYMARKET_ROUND21_AI_CANDIDATES
    )
    qualified = tuple(value for value in scores if value.development_qualified)
    nominated = None if not qualified else min(qualified, key=_ranking_key)
    provisional = Round21AICandidateSelection(
        matched_population_sha256=selected[0].matched_population_sha256,
        baseline_matrix_sha256=selected[0].baseline_matrix_sha256,
        scores=scores,
        qualified_candidate_count=len(qualified),
        nominated_model=None if nominated is None else nominated.model,
        nominated_model_digest=(None if nominated is None else nominated.model_digest),
        nominated_comparison_sha256=(
            None if nominated is None else nominated.comparison_sha256
        ),
        selection_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        selection_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_AI_CANDIDATES",
    "POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256",
    "POLYMARKET_ROUND21_AI_SELECTION_SCHEMA_VERSION",
    "Round21AICandidateScore",
    "Round21AICandidateSelection",
    "select_round21_ai_candidate",
    "validate_round21_ai_candidate_selection_design",
]
