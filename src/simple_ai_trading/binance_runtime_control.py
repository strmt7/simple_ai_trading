"""Binance adapter for shared durable lease, pause and Stop ordering.

Construction alone does not enroll a trading account or wire native dispatch.
Callers must separately bind execution scope, owned assets and durable intents.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .runtime_control import (
    RuntimeControl,
    RuntimeControlBlocked,
    RuntimeControlError,
    RuntimeControlNamespace,
    RuntimeControlService,
    RuntimeLeaseInterlock,
)


BINANCE_RUNTIME_CONTROL_SCHEMA_VERSION = "binance-runtime-control-v1"
BINANCE_RUNTIME_LEASE_ENV = "SIMPLE_AI_TRADING_BINANCE_RUNTIME_LEASE"


class BinanceRuntimeControl(RuntimeControl):
    """Keep Binance authority in its own durable control namespace."""

    namespace = RuntimeControlNamespace(
        table_name="binance_runtime_control",
        schema_version=BINANCE_RUNTIME_CONTROL_SCHEMA_VERSION,
        label="Binance",
    )

    @contextmanager
    def stopped_recovery_guard(self, *, timeout_seconds: float = 1.0) -> Iterator[None]:
        """Fence rearm and owned closes while applying retained execution evidence."""
        with self._interlock(), self.close_guard(timeout_seconds=timeout_seconds):
            snapshot = self.snapshot()
            if (
                snapshot.state != "stop_requested"
                or not snapshot.paused
                or snapshot.lease_id
                or snapshot.owner_process_id != 0
                or snapshot.started_at_ms != 0
                or snapshot.heartbeat_at_ms != 0
            ):
                raise self.blocked_error(
                    "recovery requires ownerless, Stop-latched control"
                )
            yield


__all__ = [
    "BINANCE_RUNTIME_CONTROL_SCHEMA_VERSION",
    "BINANCE_RUNTIME_LEASE_ENV",
    "BinanceRuntimeControl",
    "RuntimeControlBlocked",
    "RuntimeControlError",
    "RuntimeControlService",
    "RuntimeLeaseInterlock",
]
