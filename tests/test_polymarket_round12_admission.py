from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from simple_ai_trading.polymarket_round12_admission import (
    PolymarketRound12ActionLocalAdmission,
    build_round12_action_local_admission,
)


SEGMENT_A = "a" * 64
SEGMENT_B = "b" * 64
FEATURE_SHA = "c" * 64
DECISION_NS = 10_000_000_000
TARGET_NS = 10_500_000_000


@dataclass(frozen=True)
class _Feature:
    action_feature_sha256: str = FEATURE_SHA
    condition_id: str = "condition"
    token_id: str = "token"
    outcome: str = "Up"
    decision_event_id: str = "decision"
    decision_received_wall_ms: int = 1_000
    decision_received_monotonic_ns: int = DECISION_NS

    def validated(self):
        return self


@dataclass(frozen=True)
class _Book:
    event_id: str
    segment_id: str
    received_monotonic_ns: int


@dataclass(frozen=True)
class _Decision:
    creation_book: _Book
    condition_id: str = "condition"
    token_id: str = "token"
    outcome: str = "Up"
    event_id: str = "decision"
    segment_id: str = SEGMENT_A
    received_wall_ms: int = 1_000
    received_monotonic_ns: int = DECISION_NS

    def validated(self, *, maximum_creation_book_age_ms: int):
        age = self.received_monotonic_ns - self.creation_book.received_monotonic_ns
        if not 0 <= age <= maximum_creation_book_age_ms * 1_000_000:
            raise ValueError("invalid creation age")
        return self


def _execution(
    *,
    terminal_reason: str = "complete_round_trip",
    entry_filled: bool = True,
    entry_result: object | None = None,
    entry_book: _Book | None = None,
    target_ns: int | None = TARGET_NS,
    decision: _Decision | None = None,
):
    selected_decision = decision or _Decision(
        creation_book=_Book("creation", SEGMENT_A, DECISION_NS - 100_000_000)
    )
    selected_entry_book = entry_book
    if selected_entry_book is None and terminal_reason not in {
        "missing_entry_execution_book",
        "entry_enters_excluded_close_window",
        "entry_tick_drift",
        "missing_entry_execution_parameters",
        "unsupported_entry_minimum_order_age",
    }:
        selected_entry_book = _Book("entry", SEGMENT_A, TARGET_NS + 20_000_000)
    selected_result = entry_result
    if selected_result is None and terminal_reason == "entry_not_filled":
        selected_result = SimpleNamespace(state="REJECTED")
    return SimpleNamespace(
        terminal_reason=terminal_reason,
        decision=selected_decision,
        entry_execution_target_monotonic_ns=target_ns,
        entry_book=selected_entry_book,
        entry_result=selected_result,
        entry_filled=entry_filled,
    )


def test_action_local_admission_records_known_fill() -> None:
    admission = build_round12_action_local_admission(_Feature(), _execution())

    assert admission.decision_admissible is True
    assert admission.submission_attempted is True
    assert admission.observation_state == "known_fill"
    assert admission.condition_blocked is True
    assert admission.decision_segment_id == admission.entry_book_segment_id
    assert len(admission.admission_sha256) == 64


def test_action_local_admission_records_definite_no_fill_without_blocking() -> None:
    admission = build_round12_action_local_admission(
        _Feature(),
        _execution(
            terminal_reason="entry_not_filled",
            entry_filled=False,
        ),
    )

    assert admission.observation_state == "known_no_fill"
    assert admission.submission_attempted is True
    assert admission.condition_blocked is False


def test_action_local_admission_never_turns_missing_observation_into_no_fill() -> None:
    admission = build_round12_action_local_admission(
        _Feature(),
        _execution(
            terminal_reason="missing_entry_execution_book",
            entry_filled=False,
            entry_book=None,
        ),
    )

    assert admission.observation_state == "unknown_after_submit"
    assert admission.condition_blocked is True
    assert admission.entry_book_event_id == ""
    assert admission.reasons == ("missing_entry_execution_book",)


def test_action_local_admission_treats_pre_submit_tick_drift_as_abstention() -> None:
    admission = build_round12_action_local_admission(
        _Feature(),
        _execution(
            terminal_reason="entry_tick_drift",
            entry_filled=False,
        ),
    )

    assert admission.observation_state == "not_submitted"
    assert admission.submission_attempted is False
    assert admission.entry_execution_target_monotonic_ns is None
    assert admission.condition_blocked is False


def test_action_local_admission_missing_creation_is_not_a_submission() -> None:
    admission = build_round12_action_local_admission(_Feature(), None)

    assert admission.decision_admissible is False
    assert admission.submission_attempted is False
    assert admission.observation_state == "not_submitted"
    assert admission.reasons == ("missing_entry_creation_book",)


def test_action_local_admission_does_not_require_post_entry_feed_continuity() -> None:
    admission = build_round12_action_local_admission(
        _Feature(),
        _execution(terminal_reason="missing_exit_execution_book"),
    )

    assert admission.observation_state == "known_fill"
    assert (
        admission.identity_payload()[
            "post_entry_feed_continuity_required_for_resolution_label"
        ]
        is False
    )


def test_action_local_admission_rejects_cross_segment_entry() -> None:
    with pytest.raises(ValueError, match="crossed continuity segments"):
        build_round12_action_local_admission(
            _Feature(),
            _execution(entry_book=_Book("entry", SEGMENT_B, TARGET_NS + 1)),
        )


def test_action_local_admission_rejects_entry_outside_observation_horizon() -> None:
    with pytest.raises(ValueError, match="observed entry continuity"):
        build_round12_action_local_admission(
            _Feature(),
            _execution(
                entry_book=_Book("entry", SEGMENT_A, TARGET_NS + 500_000_001)
            ),
        )


def test_action_local_admission_hash_detects_mutation() -> None:
    admission = build_round12_action_local_admission(_Feature(), _execution())
    tampered: PolymarketRound12ActionLocalAdmission = replace(
        admission,
        condition_blocked=False,
    )

    with pytest.raises(ValueError, match="admission is invalid"):
        tampered.validated()
