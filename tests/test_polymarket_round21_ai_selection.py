from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round21_ai_comparison import (
    Round21AIMatchedComparison,
)
from simple_ai_trading.polymarket_round21_ai_selection import (
    POLYMARKET_ROUND21_AI_CANDIDATES,
    POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256,
    select_round21_ai_candidate,
    validate_round21_ai_candidate_selection_design,
)
from simple_ai_trading.polymarket_round21_comparison import (
    Round21MatchedReplayDelta,
)
from simple_ai_trading.polymarket_round21_execution import (
    POLYMARKET_ROUND21_EXECUTION_SCENARIOS,
)
from simple_ai_trading.polymarket_round21_policy import (
    POLYMARKET_ROUND21_RISK_PROFILES,
)


DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-ai-candidate-selection-design-v1.json"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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


def _delta(
    *,
    model: str,
    profile: str,
    scenario: str,
    lower: Decimal,
    drawdown_delta: Decimal,
    accepted: bool,
) -> Round21MatchedReplayDelta:
    provisional = Round21MatchedReplayDelta(
        profile=profile,
        scenario=scenario,
        baseline_replay_sha256=_sha(f"baseline:{profile}:{scenario}"),
        challenger_replay_sha256=_sha(
            f"challenger:{model}:{profile}:{scenario}"
        ),
        matched_condition_count=400,
        net_pnl_delta_quote=Decimal("2"),
        mean_condition_utility_delta_quote=Decimal("0.01"),
        daily_mean_delta_lower_95_quote=lower,
        maximum_drawdown_delta_fraction=drawdown_delta,
        tail_mean_delta_quote=Decimal("0.02"),
        accepted=accepted,
        reasons=() if accepted else ("forced_test_rejection",),
        delta_sha256=EMPTY_SHA256,
    )
    return replace(
        provisional,
        delta_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _comparison(
    model: str,
    *,
    lower: str,
    drawdown_delta: str = "-0.001",
    accepted: bool = True,
    matched_population_sha256: str | None = None,
    model_digest: str | None = None,
) -> Round21AIMatchedComparison:
    deltas = tuple(
        _delta(
            model=model,
            profile=profile.name,
            scenario=scenario.name,
            lower=Decimal(lower),
            drawdown_delta=Decimal(drawdown_delta),
            accepted=accepted,
        )
        for profile in POLYMARKET_ROUND21_RISK_PROFILES
        for scenario in POLYMARKET_ROUND21_EXECUTION_SCENARIOS
    )
    provisional = Round21AIMatchedComparison(
        model=model,
        model_digest=model_digest or _sha(f"weights:{model}"),
        ai_report_sha256=_sha(f"report:{model}"),
        matched_population_sha256=(
            matched_population_sha256 or _sha("matched-population")
        ),
        baseline_matrix_sha256=_sha("baseline-matrix"),
        ai_matrix_sha256=_sha(f"ai-matrix:{model}"),
        ai_permission_root_sha256=_sha(f"permissions:{model}"),
        matched_decision_count=400,
        non_tied_primary_action_count=40,
        deltas=deltas,
        all_replays_accepted=accepted,
        development_qualified=accepted,
        ai_model_selected=False,
        comparison_sha256=EMPTY_SHA256,
    )
    return replace(
        provisional,
        comparison_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _comparisons() -> tuple[Round21AIMatchedComparison, ...]:
    return (
        _comparison("qwen3:8b", lower="0.01"),
        _comparison("fin-r1:8b", lower="0.02"),
        _comparison("fino1:8b", lower="0.03"),
    )


def test_round21_ai_selection_design_is_canonical_and_non_authoritative() -> None:
    value = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))

    validated = validate_round21_ai_candidate_selection_design(value)

    assert (
        validated["design_sha256"]
        == POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256
    )
    assert validated["candidate_program"]["models"] == list(
        POLYMARKET_ROUND21_AI_CANDIDATES
    )
    assert not any(validated["authority"].values())


def test_round21_ai_selection_design_rejects_rehashed_ranking_drift() -> None:
    value = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    value["ranking"]["primary"] = "maximize_development_pnl"
    body = dict(value)
    body.pop("design_sha256")
    value["design_sha256"] = _canonical_sha256(body)

    with pytest.raises(ValueError, match="selection design differs"):
        validate_round21_ai_candidate_selection_design(value)


def test_round21_ai_selection_nominates_worst_case_after_cost_winner_only() -> None:
    selection = select_round21_ai_candidate(_comparisons())

    assert selection.qualified_candidate_count == 3
    assert selection.nominated_model == "fino1:8b"
    assert selection.nominated_model_digest == _sha("weights:fino1:8b")
    assert not selection.ai_model_selected
    assert not selection.ai_edge_claim
    assert not selection.profitability_claim
    assert not selection.paper_trading_authority
    assert not selection.live_trading_authority


def test_round21_ai_selection_uses_drawdown_only_after_primary_tie() -> None:
    comparisons = (
        _comparison("qwen3:8b", lower="0.02", drawdown_delta="-0.002"),
        _comparison("fin-r1:8b", lower="0.02", drawdown_delta="-0.004"),
        _comparison("fino1:8b", lower="0.01", drawdown_delta="-0.010"),
    )

    selection = select_round21_ai_candidate(comparisons)

    assert selection.nominated_model == "fin-r1:8b"


def test_round21_ai_selection_keeps_rejections_and_nominates_none() -> None:
    comparisons = tuple(
        _comparison(model, lower="0.01", accepted=False)
        for model in POLYMARKET_ROUND21_AI_CANDIDATES
    )

    selection = select_round21_ai_candidate(comparisons)

    assert selection.qualified_candidate_count == 0
    assert selection.nominated_model is None
    assert all(
        score.rejection_reasons
        == ("one_or_more_after_cost_ledgers_rejected",)
        for score in selection.scores
    )


def test_round21_ai_selection_rejects_population_or_weight_aliasing() -> None:
    comparisons = list(_comparisons())
    comparisons[2] = _comparison(
        "fino1:8b",
        lower="0.03",
        matched_population_sha256=_sha("different-population"),
    )
    with pytest.raises(ValueError, match="candidate population differs"):
        select_round21_ai_candidate(comparisons)

    comparisons = list(_comparisons())
    comparisons[2] = _comparison(
        "fino1:8b",
        lower="0.03",
        model_digest=comparisons[1].model_digest,
    )
    with pytest.raises(ValueError, match="candidate population differs"):
        select_round21_ai_candidate(comparisons)


def test_round21_ai_selection_rejects_unregistered_ledger_names() -> None:
    comparisons = list(_comparisons())
    original = comparisons[0]
    changed_delta = replace(
        original.deltas[0],
        scenario="unregistered-scenario",
        delta_sha256=EMPTY_SHA256,
    )
    changed_delta = replace(
        changed_delta,
        delta_sha256=_canonical_sha256(changed_delta.identity_payload()),
    ).validated()
    changed = replace(
        original,
        deltas=(changed_delta, *original.deltas[1:]),
        comparison_sha256=EMPTY_SHA256,
    )
    changed = replace(
        changed,
        comparison_sha256=_canonical_sha256(changed.identity_payload()),
    ).validated()
    comparisons[0] = changed

    with pytest.raises(ValueError, match="ledger population differs"):
        select_round21_ai_candidate(comparisons)


def test_round21_ai_selection_rejects_nomination_tampering() -> None:
    selection = select_round21_ai_candidate(_comparisons())

    with pytest.raises(ValueError, match="candidate selection differs"):
        replace(selection, nominated_model="qwen3:8b").validated()
