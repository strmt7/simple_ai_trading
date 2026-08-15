from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest

import simple_ai_trading.polymarket_round21_shadow as shadow_module
from simple_ai_trading.polymarket_round21_live_features import (
    Round21CoordinatedPrediction,
)
from simple_ai_trading.polymarket_round21_session import (
    Round21RollingPublicDataService,
)
from simple_ai_trading.polymarket_round21_shadow import (
    Round21ProspectiveShadowRunner,
)
from simple_ai_trading.polymarket_round21_shadow_store import (
    Round21ProspectiveShadowStore,
)

from polymarket_round21_support import (
    SHADOW_CONDITION_ID,
    SHADOW_MODEL_SHA,
    SHADOW_SEALED_SHA,
    START_MS,
    round21_replay_condition,
    round21_shadow_prediction,
    sha,
)


MARKET = replace(
    round21_replay_condition().market,
    condition_id=SHADOW_CONDITION_ID,
    event_start_ms=START_MS,
    end_ms=START_MS + 300_000,
)
RUN_ID = "1" * 32
MODEL_SHA = SHADOW_MODEL_SHA
SEALED_SHA = SHADOW_SEALED_SHA
_prediction = round21_shadow_prediction


def _clock(*values: int):
    remaining = iter(values)
    last = values[-1]

    def read() -> int:
        nonlocal last
        last = next(remaining, last)
        return last

    return read


def _service(*, prediction=None, failure: Exception | None = None):
    service = Mock(spec=Round21RollingPublicDataService)
    service.scorer = Mock()
    service.scorer.source_model_artifact_sha256 = MODEL_SHA
    service.scorer.sealed_result.result_sha256 = SEALED_SHA
    service.scorer.population_layer = "core"
    service.current_market.return_value = MARKET
    if prediction is None:
        service.evaluate.return_value = None
    else:
        service.evaluate.return_value = Round21CoordinatedPrediction.create(
            status=prediction.status,
            reasons=() if prediction.status == "observed" else (prediction.reason,),
            market=MARKET,
            decision_time_ms=prediction.decision_time_ms,
            observed_at_ms=prediction.observed_at_ms,
            core_source_healthy=True,
            optional_source_healthy=True,
            core_gap_sha256=(),
            prediction=prediction,
        ).validated()

    async def run(stop: asyncio.Event) -> None:
        if failure is not None:
            raise failure
        await stop.wait()

    service.run.side_effect = run
    return service


def test_shadow_runner_records_one_target_free_prediction_and_completes(
    tmp_path,
) -> None:
    async def exercise():
        prediction = _prediction()
        service = _service(prediction=prediction)
        with Round21ProspectiveShadowStore(tmp_path / "complete.sqlite3") as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                poll_interval_seconds=0.01,
                wall_time_ms=_clock(
                    START_MS,
                    prediction.observed_at_ms,
                    START_MS + 2_000,
                    START_MS + 2_000,
                ),
            )
            audit = await runner.run(
                asyncio.Event(),
                scheduled_end_ms=START_MS + 2_000,
            )
            assert audit.prediction_count == 1
            assert audit.observed_count == 1
            assert audit.terminal is not None
            assert audit.terminal.status == "complete"
            assert store.predictions(RUN_ID)[0].prediction == prediction
            with pytest.raises(RuntimeError, match="cannot be restarted"):
                await runner.run(
                    asyncio.Event(),
                    scheduled_end_ms=START_MS + 3_000,
                )
        return service, runner

    service, runner = asyncio.run(exercise())
    service.run.assert_awaited_once()
    assert not any(
        (
            runner.credentials_used,
            runner.account_connected,
            runner.binance_execution_connected,
            runner.grants_execution_authority,
            runner.trading_authority,
            runner.paper_trading_authority,
            runner.live_trading_authority,
        )
    )


def test_shadow_runner_stop_is_terminal_interruption_without_prediction(
    tmp_path,
) -> None:
    async def exercise():
        service = _service()
        stop = asyncio.Event()
        stop.set()
        with Round21ProspectiveShadowStore(tmp_path / "stop.sqlite3") as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                poll_interval_seconds=0.01,
                wall_time_ms=_clock(START_MS, START_MS + 1, START_MS + 2),
            )
            audit = await runner.run(
                stop,
                scheduled_end_ms=START_MS + 2_000,
            )
            assert audit.prediction_count == 0
            assert audit.terminal is not None
            assert audit.terminal.status == "interrupted"
            assert audit.terminal.reason == "operator_stop"

    asyncio.run(exercise())


def test_shadow_runner_persists_failed_terminal_before_propagating_service_error(
    tmp_path,
) -> None:
    async def exercise(path):
        service = _service(failure=RuntimeError("feed failed"))
        with Round21ProspectiveShadowStore(path) as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                poll_interval_seconds=0.01,
                wall_time_ms=_clock(START_MS, START_MS + 1, START_MS + 2),
            )
            with pytest.raises(RuntimeError, match="feed failed"):
                await runner.run(
                    asyncio.Event(),
                    scheduled_end_ms=START_MS + 2_000,
                )

    path = tmp_path / "failed.sqlite3"
    asyncio.run(exercise(path))
    with Round21ProspectiveShadowStore(path) as restarted:
        audit = restarted.audit_run(RUN_ID)
        assert audit.terminal is not None
        assert audit.terminal.status == "failed"
        assert audit.terminal.reason == "runtime_RuntimeError"


def test_shadow_runner_rejects_invalid_boundaries(tmp_path) -> None:
    service = _service()
    with Round21ProspectiveShadowStore(tmp_path / "invalid.sqlite3") as store:
        with pytest.raises(TypeError, match="data service"):
            Round21ProspectiveShadowRunner(data_service=object(), store=store)
        with pytest.raises(TypeError, match="store differs"):
            Round21ProspectiveShadowRunner(data_service=service, store=object())
        with pytest.raises(ValueError, match="polling interval"):
            Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                poll_interval_seconds=1.0,
            )
        with pytest.raises(TypeError, match="wall clock"):
            Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                wall_time_ms=0,
            )

        runner = Round21ProspectiveShadowRunner(
            data_service=service,
            store=store,
            run_id=RUN_ID,
            wall_time_ms=lambda: START_MS,
        )
        with pytest.raises(TypeError, match="Stop type"):
            asyncio.run(runner.run(object(), scheduled_end_ms=START_MS + 1))
        with pytest.raises(ValueError, match="not prospective"):
            asyncio.run(runner.run(asyncio.Event(), scheduled_end_ms=START_MS))

        invalid_clock = Round21ProspectiveShadowRunner(
            data_service=service,
            store=store,
            run_id="2" * 32,
            wall_time_ms=lambda: 0,
        )
        with pytest.raises(ValueError, match="start timestamp"):
            asyncio.run(
                invalid_clock.run(
                    asyncio.Event(),
                    scheduled_end_ms=START_MS + 1,
                )
            )


@pytest.mark.parametrize("mode", ("no_market", "no_decision", "gap_abstention"))
def test_shadow_runner_preserves_empty_coverage_without_inventing_predictions(
    tmp_path,
    mode: str,
) -> None:
    async def exercise():
        service = _service()
        if mode == "no_market":
            service.current_market.return_value = None
        elif mode == "gap_abstention":
            service.evaluate.return_value = Round21CoordinatedPrediction.create(
                status="abstain",
                reasons=("public_source_gap",),
                market=MARKET,
                decision_time_ms=START_MS,
                observed_at_ms=START_MS + 1,
                core_source_healthy=False,
                optional_source_healthy=True,
                core_gap_sha256=(sha("gap"),),
                prediction=None,
            ).validated()
        with Round21ProspectiveShadowStore(tmp_path / f"{mode}.sqlite3") as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                poll_interval_seconds=0.01,
                wall_time_ms=_clock(
                    START_MS,
                    START_MS + 1,
                    START_MS + 20,
                    START_MS + 20,
                ),
            )
            audit = await runner.run(
                asyncio.Event(),
                scheduled_end_ms=START_MS + 20,
            )
            assert audit.prediction_count == 0
            assert audit.terminal is not None
            assert audit.terminal.status == "complete"
        return service

    service = asyncio.run(exercise())
    if mode == "no_market":
        service.evaluate.assert_not_called()
    else:
        service.evaluate.assert_called_once()


@pytest.mark.parametrize("failure_mode", ("clean_return", "clock", "decision_type"))
def test_shadow_runner_latches_runtime_fault_before_terminal_audit(
    tmp_path,
    failure_mode: str,
) -> None:
    async def exercise(path):
        service = _service()
        clock = _clock(START_MS, START_MS + 1, START_MS + 2, START_MS + 3)
        if failure_mode == "clean_return":

            async def clean_run(_stop: asyncio.Event) -> None:
                return

            service.run.side_effect = clean_run
            expected = "returned unexpectedly"
        elif failure_mode == "clock":
            clock = _clock(START_MS, START_MS - 1, START_MS + 2)
            expected = "clock regressed"
        else:
            service.evaluate.return_value = object()
            expected = "coordinated prediction differs"
        with Round21ProspectiveShadowStore(path) as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                poll_interval_seconds=0.01,
                wall_time_ms=clock,
            )
            with pytest.raises((RuntimeError, TypeError), match=expected):
                await runner.run(
                    asyncio.Event(),
                    scheduled_end_ms=START_MS + 100,
                )

    path = tmp_path / f"fault-{failure_mode}.sqlite3"
    asyncio.run(exercise(path))
    with Round21ProspectiveShadowStore(path) as store:
        audit = store.audit_run(RUN_ID)
        assert audit.terminal is not None
        assert audit.terminal.status == "failed"


@pytest.mark.parametrize("service_cancelled", (False, True))
def test_shadow_runner_propagates_shutdown_failure_after_terminalizing(
    tmp_path,
    service_cancelled: bool,
) -> None:
    async def exercise(path):
        service = _service()

        async def fail_on_stop(stop: asyncio.Event) -> None:
            await stop.wait()
            if service_cancelled:
                raise asyncio.CancelledError
            raise RuntimeError("shutdown failure")

        service.run.side_effect = fail_on_stop
        stop = asyncio.Event()
        stop.set()
        with Round21ProspectiveShadowStore(path) as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                wall_time_ms=_clock(START_MS, START_MS + 1, START_MS + 2),
            )
            with pytest.raises(
                asyncio.CancelledError if service_cancelled else RuntimeError
            ):
                await runner.run(stop, scheduled_end_ms=START_MS + 100)

    path = tmp_path / f"shutdown-{service_cancelled}.sqlite3"
    asyncio.run(exercise(path))
    with Round21ProspectiveShadowStore(path) as store:
        audit = store.audit_run(RUN_ID)
        assert audit.terminal is not None
        assert audit.terminal.status == (
            "interrupted" if service_cancelled else "failed"
        )


def test_shadow_runner_cancellation_is_durable_and_propagated(tmp_path) -> None:
    async def exercise(path):
        service = _service()
        with Round21ProspectiveShadowStore(path) as store:
            runner = Round21ProspectiveShadowRunner(
                data_service=service,
                store=store,
                run_id=RUN_ID,
                poll_interval_seconds=0.25,
                wall_time_ms=_clock(START_MS, START_MS + 1, START_MS + 2),
            )
            task = asyncio.create_task(
                runner.run(
                    asyncio.Event(),
                    scheduled_end_ms=START_MS + 100,
                )
            )
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    path = tmp_path / "cancelled.sqlite3"
    asyncio.run(exercise(path))
    with Round21ProspectiveShadowStore(path) as store:
        audit = store.audit_run(RUN_ID)
        assert audit.terminal is not None
        assert audit.terminal.status == "interrupted"
        assert audit.terminal.reason == "runner_cancelled"


def test_shadow_runner_module_has_no_execution_authority() -> None:
    assert not any(
        (
            shadow_module.credentials_used,
            shadow_module.account_connected,
            shadow_module.binance_execution_connected,
            shadow_module.grants_execution_authority,
            shadow_module.profitability_claim,
            shadow_module.paper_trading_authority,
            shadow_module.live_trading_authority,
        )
    )
