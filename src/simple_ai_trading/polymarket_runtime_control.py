"""Polymarket adapter for shared durable lease and Stop control.

The original table, record digests, schema versions, exceptions and public types
remain compatible with existing Polymarket ledgers and callers.
"""

from __future__ import annotations

from dataclasses import dataclass

from .polymarket_live import PolymarketLiveBlocked, PolymarketLiveError
from .runtime_control import (
    RuntimeControl,
    RuntimeControlNamespace,
    RuntimeControlService,
    RuntimeControlSnapshot,
    RuntimeLeaseInterlock,
    _CrossProcessFileLock as _SharedCrossProcessFileLock,
)


POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION = "polymarket-runtime-control-v2"


class _CrossProcessFileLock(_SharedCrossProcessFileLock):
    blocked_error = PolymarketLiveBlocked
    label = "Polymarket"


@dataclass(frozen=True, slots=True)
class PolymarketRuntimeControlSnapshot(RuntimeControlSnapshot):
    """Compatibility type for the original Polymarket control record."""


class PolymarketRuntimeControl(RuntimeControl):
    """Use the shared control with the existing Polymarket storage namespace."""

    namespace = RuntimeControlNamespace(
        table_name="polymarket_runtime_control",
        schema_version=POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION,
        legacy_schema_version="polymarket-runtime-control-v1",
        label="Polymarket",
    )
    error = PolymarketLiveError
    blocked_error = PolymarketLiveBlocked
    snapshot_type = PolymarketRuntimeControlSnapshot
    lock_type = _CrossProcessFileLock


@dataclass(frozen=True, slots=True)
class PolymarketRuntimeLeaseInterlock(RuntimeLeaseInterlock):
    """Bind one Polymarket lease to the shared submission interlock."""


class PolymarketRuntimeControlService(RuntimeControlService):
    """Preserve the session-owned Polymarket heartbeat service interface."""


__all__ = [
    "POLYMARKET_RUNTIME_CONTROL_SCHEMA_VERSION",
    "PolymarketRuntimeControl",
    "PolymarketRuntimeControlService",
    "PolymarketRuntimeControlSnapshot",
    "PolymarketRuntimeLeaseInterlock",
]
