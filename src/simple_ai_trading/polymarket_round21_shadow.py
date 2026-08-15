"""No-order prospective shadow runner for independent Polymarket Round 21."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import secrets
from threading import Lock
import time

from .polymarket_round21_live_features import Round21CoordinatedPrediction
from .polymarket_round21_session import Round21RollingPublicDataService
from .polymarket_round21_shadow_store import (
    Round21ProspectiveShadowStore,
    Round21ShadowAudit,
)


POLYMARKET_ROUND21_SHADOW_RUNNER_SCHEMA_VERSION = (
    "polymarket-round21-prospective-shadow-runner-v1"
)


class Round21ProspectiveShadowRunner:
    """Coordinate public scoring and durable evidence without order authority."""

    credentials_used = False
    account_connected = False
    binance_execution_connected = False
    grants_execution_authority = False
    trading_authority = False
    paper_trading_authority = False
    live_trading_authority = False

    def __init__(
        self,
        *,
        data_service: Round21RollingPublicDataService,
        store: Round21ProspectiveShadowStore,
        run_id: str | None = None,
        poll_interval_seconds: float = 0.05,
        wall_time_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        if not isinstance(data_service, Round21RollingPublicDataService):
            raise TypeError("Round 21 shadow data service differs")
        if not isinstance(store, Round21ProspectiveShadowStore):
            raise TypeError("Round 21 shadow store differs")
        selected_run_id = secrets.token_hex(16) if run_id is None else str(run_id)
        interval = float(poll_interval_seconds)
        if not 0.01 <= interval <= 0.25:
            raise ValueError("Round 21 shadow polling interval is invalid")
        if not callable(wall_time_ms):
            raise TypeError("Round 21 shadow runner wall clock is invalid")
        self.data_service = data_service
        self.store = store
        self.run_id = selected_run_id
        self.poll_interval_seconds = interval
        self._wall_time_ms = wall_time_ms
        self._lock = Lock()
        self._started = False

    @staticmethod
    def _timestamp(value: object, *, name: str) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError(f"Round 21 shadow runner {name} is invalid")
        return value

    async def _wait(self, stop: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=self.poll_interval_seconds,
            )
        except TimeoutError:
            return

    async def run(
        self,
        stop: asyncio.Event,
        *,
        scheduled_end_ms: int,
    ) -> Round21ShadowAudit:
        """Run once to a fixed end or Stop and return a non-authoritative audit."""

        if not isinstance(stop, asyncio.Event):
            raise TypeError("Round 21 shadow runner Stop type differs")
        started_at_ms = self._timestamp(
            self._wall_time_ms(),
            name="start timestamp",
        )
        end_ms = self._timestamp(scheduled_end_ms, name="scheduled end")
        if end_ms <= started_at_ms:
            raise ValueError("Round 21 shadow scheduled end is not prospective")
        with self._lock:
            if self._started:
                raise RuntimeError("Round 21 shadow runner cannot be restarted")
            self._started = True

        scorer = self.data_service.scorer
        run = self.store.start_run(
            run_id=self.run_id,
            source_model_artifact_sha256=scorer.source_model_artifact_sha256,
            sealed_result_sha256=scorer.sealed_result.result_sha256,
            population_layer=scorer.population_layer,
            started_at_ms=started_at_ms,
        )
        service_stop = asyncio.Event()
        service_task = asyncio.create_task(self.data_service.run(service_stop))
        status = "interrupted"
        reason = "operator_stop"
        failure: BaseException | None = None
        last_wall_ms = started_at_ms
        last_recorded_ms = started_at_ms
        try:
            await asyncio.sleep(0)
            while True:
                if service_task.done():
                    await service_task
                    raise RuntimeError(
                        "Round 21 shadow public data service returned unexpectedly"
                    )
                now_ms = self._timestamp(
                    self._wall_time_ms(),
                    name="wall timestamp",
                )
                if now_ms < last_wall_ms:
                    raise RuntimeError("Round 21 shadow wall clock regressed")
                last_wall_ms = now_ms
                if now_ms >= end_ms:
                    status = "complete"
                    reason = ""
                    break
                if stop.is_set():
                    break
                market = self.data_service.current_market()
                if market is not None:
                    coordinated = self.data_service.evaluate(
                        market,
                        observed_at_ms=now_ms,
                    )
                    if coordinated is not None:
                        if not isinstance(coordinated, Round21CoordinatedPrediction):
                            raise TypeError(
                                "Round 21 shadow coordinated prediction differs"
                            )
                        decision = coordinated.validated()
                        if decision.prediction is not None:
                            stored = self.store.append_prediction(
                                run.run_id,
                                decision.prediction,
                                recorded_at_ms=now_ms,
                            )
                            last_recorded_ms = max(
                                last_recorded_ms,
                                stored.recorded_at_ms,
                            )
                await self._wait(stop)
        except asyncio.CancelledError as exc:
            status = "interrupted"
            reason = "runner_cancelled"
            failure = exc
        except Exception as exc:
            status = "failed"
            reason = f"runtime_{exc.__class__.__name__}"
            failure = exc
        finally:
            service_stop.set()
            try:
                await service_task
            except asyncio.CancelledError as exc:
                if failure is None:
                    status = "interrupted"
                    reason = "public_data_service_cancelled"
                    failure = exc
            except Exception as exc:
                if failure is None:
                    status = "failed"
                    reason = f"public_data_service_{exc.__class__.__name__}"
                    failure = exc

        finished_at_ms = max(
            started_at_ms,
            last_wall_ms,
            last_recorded_ms,
            int(self._wall_time_ms()),
        )
        self.store.terminate_run(
            run.run_id,
            status=status,
            reason=reason,
            finished_at_ms=finished_at_ms,
        )
        audit = self.store.audit_run(run.run_id)
        if failure is not None:
            raise failure
        return audit


credentials_used = False
account_connected = False
binance_execution_connected = False
grants_execution_authority = False
profitability_claim = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_SHADOW_RUNNER_SCHEMA_VERSION",
    "Round21ProspectiveShadowRunner",
]
