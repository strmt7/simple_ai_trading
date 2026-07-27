from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.impact_absorption_event_action_policy import (
    Round74ActionPolicySelection,
    Round74ActionThresholdEvaluation,
    Round74ActionTrace,
    Round74ActionTraceMetrics,
    round74_action_profile,
)
from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_sealed_ledger import (
    Round74SealedEvaluationClaim,
    Round74SealedEvaluationLedger,
    Round74SealedLedgerError,
    Round74SealedReuseError,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
    ROUND74_EVENT_SYMBOLS,
)


TEST_RUNS = tuple(f"{index:032x}" for index in range(100, 124))
POLICY_RUNS = tuple(f"{index:032x}" for index in range(200, 206))


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _test_batch(*, role: str = "test") -> Round74EventTrainingBatch:
    rows = 24
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    wall = np.arange(rows, dtype=np.int64) * 4_000_000_000_000
    wall += 1_800_000_000_000_000_000
    features = np.zeros(
        (
            rows,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        ),
        dtype=np.float32,
    )
    entry = np.full(action_shape, 10, dtype=np.int64)
    exit_value = np.full(action_shape, 20, dtype=np.int64)
    result = Round74EventTrainingBatch(
        role=role,
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=TEST_RUNS,
        symbol=tuple(
            ROUND74_EVENT_SYMBOLS[index % len(ROUND74_EVENT_SYMBOLS)]
            for index in range(rows)
        ),
        decision_monotonic_ns=_readonly(np.full(rows, 1_000_000_000, dtype=np.int64)),
        decision_wall_ns=_readonly(wall),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.arange(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{300 + index:064x}" for index in range(rows)),
        target_context_sha256=tuple("3" * 64 for _ in range(rows)),
        test_access_sha256=tuple(
            "4" * 64 if role == "test" else "" for _ in range(rows)
        ),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(entry),
        actual_exit_monotonic_ns=_readonly(exit_value),
        net_payoff_bps=_readonly(np.ones(action_shape, dtype=np.float32)),
        maximum_adverse_excursion_bps=_readonly(
            np.ones(action_shape, dtype=np.float32)
        ),
        adverse_selection=_readonly(np.zeros(action_shape, dtype=np.float32)),
        regime_unpredictability=_readonly(np.zeros(regime_shape, dtype=np.float32)),
        action_eligibility=_readonly(np.ones(action_shape, dtype=np.float32)),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    result.validate()
    return result


def _selection() -> Round74ActionPolicySelection:
    metrics = Round74ActionTraceMetrics(
        trades=6,
        active_runs=6,
        distinct_symbols=3,
        total_net_bps=6.0,
        mean_net_bps=1.0,
        median_net_bps=1.0,
        win_rate=1.0,
        profit_factor=None,
        maximum_drawdown_bps=0.0,
        gross_profit_bps=6.0,
        gross_loss_bps=0.0,
        worst_trade_bps=1.0,
        mean_maximum_adverse_excursion_bps=1.0,
        adverse_selection_rate=0.0,
        profitable_run_ratio=1.0,
        maximum_symbol_trade_share=1.0 / 3.0,
    )
    trace = Round74ActionTrace(
        threshold_score=1.0,
        expected_run_ids=POLICY_RUNS,
        row_index=tuple(range(6)),
        run_id=POLICY_RUNS,
        symbol=ROUND74_EVENT_SYMBOLS * 2,
        feature_row_sha256=tuple(f"{400 + index:064x}" for index in range(6)),
        horizon_seconds=(30,) * 6,
        side=(1,) * 6,
        entry_monotonic_ns=(10,) * 6,
        exit_monotonic_ns=(20,) * 6,
        net_payoff_bps=(1.0,) * 6,
        maximum_adverse_excursion_bps=(1.0,) * 6,
        adverse_selection=(0,) * 6,
        skipped_target_ineligible=0,
        skipped_same_symbol_overlap=0,
        metrics=metrics,
    )
    quantiles = round74_action_profile("aggressive").threshold_quantiles
    evaluations = tuple(
        Round74ActionThresholdEvaluation(
            quantile=quantile,
            threshold_score=1.0,
            objective_bps=6.0,
            accepted=True,
            rejection_reasons=(),
            trace=trace,
        )
        for quantile in quantiles
    )
    result = Round74ActionPolicySelection(
        profile="aggressive",
        pretest_policy_sha256="5" * 64,
        probability_calibration_sha256="6" * 64,
        tuning_subpartition_sha256="7" * 64,
        target_batch_sha256="8" * 64,
        candidate_sha256="9" * 64,
        accepted=True,
        selected_quantile=quantiles[0],
        selected_threshold_score=1.0,
        evaluations=evaluations,
        rejection_reasons=(),
    )
    result.validate()
    return result


def test_reservation_consumes_test_access_before_evaluation(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64, "b" * 64),
    )

    assert claim.status == "reserved"
    assert claim.rows == 24
    assert claim.test_run_ids == TEST_RUNS
    assert claim.ai_manifest_sha256 == ("a" * 64, "b" * 64)
    assert ledger.claim_matches(claim, required_status="reserved")
    assert len(claim.claim_sha256) == 64
    assert Round74SealedEvaluationClaim.from_mapping(claim.as_dict()) == claim
    assert not Path(f"{ledger.path}-wal").exists()
    with pytest.raises(Round74SealedReuseError, match="already reserved"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64, "b" * 64),
        )


def test_completed_or_failed_reservation_cannot_be_reset_or_finalized_twice(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64,),
    )
    completed = ledger.finalize(
        claim.reservation_id,
        result_outcome="candidate_failed_predeclared_gates",
        result_sha256="c" * 64,
    )

    assert completed.status == "complete"
    assert ledger.claim_matches(completed, required_status="complete")
    with pytest.raises(Round74SealedLedgerError, match="already finalized"):
        ledger.finalize(
            claim.reservation_id,
            result_outcome="candidate_passed_predeclared_gates",
            result_sha256="d" * 64,
        )
    with pytest.raises(Round74SealedReuseError):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
        )


def test_evaluation_error_is_durable_and_still_consumes_test(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64,),
    )
    failed = ledger.finalize(
        claim.reservation_id,
        result_outcome="evaluation_error",
        result_sha256="e" * 64,
        error="worker interrupted after reservation",
    )

    assert failed.status == "failed"
    assert failed.error == "worker interrupted after reservation"
    assert ledger.claim_matches(failed, required_status="failed")
    with pytest.raises(Round74SealedReuseError):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
        )


def test_development_data_is_rejected_before_ledger_creation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.sqlite3"
    ledger = Round74SealedEvaluationLedger(path)

    with pytest.raises(ValueError, match="rejects development data"):
        ledger.reserve(
            test_batches=(_test_batch(role="tuning"),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64,),
        )

    assert not path.exists()


def test_tampered_claim_or_duplicate_manifest_panel_is_rejected(
    tmp_path: Path,
) -> None:
    ledger = Round74SealedEvaluationLedger(tmp_path / "sealed.sqlite3")
    with pytest.raises(ValueError, match="manifest panel differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64, "a" * 64),
        )
    with pytest.raises(ValueError, match="manifest panel differs"):
        ledger.reserve(
            test_batches=(_test_batch(),),
            action_selection=_selection(),
            ai_manifest_sha256=("a" * 64, "b" * 64, "c" * 64),
        )
    claim = ledger.reserve(
        test_batches=(_test_batch(),),
        action_selection=_selection(),
        ai_manifest_sha256=("a" * 64,),
    )
    tampered = replace(claim, dataset_sha256="f" * 64)
    tampered.validate()

    assert not ledger.claim_matches(tampered, required_status="reserved")
    tampered_payload = claim.as_dict()
    tampered_payload["rows"] = 25
    with pytest.raises(ValueError, match="digest differs"):
        Round74SealedEvaluationClaim.from_mapping(tampered_payload)
