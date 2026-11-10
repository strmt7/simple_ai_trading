"""Binance adapter for shared durable lease, pause and Stop ordering.

Construction alone does not enroll a trading account or wire native dispatch.
Callers must separately bind execution scope, owned assets and durable intents.
"""

from __future__ import annotations

from .runtime_control import (
    RuntimeControl,
    RuntimeControlBlocked,
    RuntimeControlError,
    RuntimeControlNamespace,
    RuntimeControlService,
    RuntimeLeaseInterlock,
)


BINANCE_RUNTIME_CONTROL_SCHEMA_VERSION = "binance-runtime-control-v1"


class BinanceRuntimeControl(RuntimeControl):
    """Keep Binance authority in its own durable control namespace."""

    namespace = RuntimeControlNamespace(
        table_name="binance_runtime_control",
        schema_version=BINANCE_RUNTIME_CONTROL_SCHEMA_VERSION,
        label="Binance",
    )


__all__ = [
    "BINANCE_RUNTIME_CONTROL_SCHEMA_VERSION",
    "BinanceRuntimeControl",
    "RuntimeControlBlocked",
    "RuntimeControlError",
    "RuntimeControlService",
    "RuntimeLeaseInterlock",
]
