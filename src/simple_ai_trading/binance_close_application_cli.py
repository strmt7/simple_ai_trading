"""Local operator inspection and application of retained owned Spot close evidence."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .binance_close_application import (
    apply_spot_closing_recovery,
    inspect_closing_recovery,
)
from .binance_close_intents import validate_closing_client_id
from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import OpenIntentError
from .binance_runtime_control import BinanceRuntimeControl
from .positions import PositionsStore


def _original_scope(store: PositionsStore) -> tuple[BinanceExecutionScope, str]:
    with closing(store.opening_intents._connect()) as connection:
        rows = connection.execute(
            "SELECT id, scope_json FROM execution_scope LIMIT 2"
        ).fetchall()
        if (
            len(rows) != 1
            or rows[0][0] != 1
            or not isinstance(rows[0][1], str)
            or not 0 < len(rows[0][1]) <= 1024
        ):
            raise OpenIntentError("original execution scope is missing or unreadable")
        scope = BinanceExecutionScope(**json.loads(rows[0][1]))
        store.opening_intents._bind_scope(connection, scope)
    canonical = json.dumps(
        asdict(scope), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return scope, hashlib.sha256(canonical).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def run_close_recovery(args: argparse.Namespace) -> tuple[int, dict]:
    """Use original local scope; never load runtime credentials or construct a client."""
    action = getattr(args, "action", "inspect")
    payload = {
        "schema_version": "owned-close-operator-v1",
        "action": action,
        "account_queried": False,
        "order_submitted": False,
        "rearmed": False,
        "application_outcome": "not_requested"
        if action == "inspect"
        else "not_started",
    }
    try:
        if action not in {"inspect", "apply"}:
            raise OpenIntentError("choose inspect or apply explicitly")
        client_id = validate_closing_client_id(getattr(args, "client_order_id", None))
        scope_digest = getattr(args, "expected_scope_sha256", None)
        recovery_digest = getattr(args, "expected_recovery_sha256", None)
        if action == "apply" and not (
            _digest(scope_digest) and _digest(recovery_digest)
        ):
            raise OpenIntentError(
                "apply requires both exact digests from local inspection"
            )
        if action == "inspect" and (
            scope_digest is not None or recovery_digest is not None
        ):
            raise OpenIntentError(
                "inspection does not accept apply-only expected digests"
            )
        directory = getattr(args, "data_dir", None)
        if not isinstance(directory, (str, Path)) or not str(directory):
            raise OpenIntentError("select an existing owned execution directory")
        store = PositionsStore(Path(directory))
        scope, observed_scope_digest = _original_scope(store)
        inspection = inspect_closing_recovery(store, scope=scope, client_id=client_id)
        observed_recovery_digest = inspection.recovery_sha256
        payload.update(
            client_order_id=client_id,
            venue_origin=scope.origin,
            market_type=scope.market_type,
            execution_scope_sha256=observed_scope_digest,
            recovery_sha256=observed_recovery_digest,
            local_intent_state=inspection.local_intent_state,
            application_recorded=inspection.application_recorded,
        )
        if action == "inspect":
            payload.update(
                status="inspected",
                application_admission_checked=False,
                message=(
                    "Local application receipt committed; ledger projections were not inspected. No account state was queried."
                    if inspection.application_recorded
                    else "Retained evidence validated; application checks remain pending. No account state was queried."
                ),
            )
            return 0, payload
        if (
            scope_digest != observed_scope_digest
            or recovery_digest != observed_recovery_digest
        ):
            raise OpenIntentError(
                "selected scope or retained evidence differs from the inspected digests"
            )
        control_path = (
            getattr(args, "runtime_control", None)
            or store.root / "binance_runtime_control.sqlite3"
        )
        control = BinanceRuntimeControl(
            control_path, enrollment_path=getattr(args, "runtime_enrollment", None)
        )
        payload["application_outcome"] = "unknown"
        result = apply_spot_closing_recovery(
            store,
            client_id=client_id,
            scope=scope,
            expected_recovery_sha256=recovery_digest,
            control=control,
            execution_enrollment_path=getattr(args, "execution_enrollment", None),
        )
        payload.update(
            status="applied",
            application_outcome="applied",
            application_recorded=True,
            local_intent_state="RECORDED",
            result=asdict(result),
            message="Applied retained closing evidence under Stop; rearm was not requested.",
        )
        return 0, payload
    except OpenIntentError as exc:
        payload.update(status="blocked", message=str(exc))
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        # Local paths and malformed retained payload values do not enter diagnostics.
        payload.update(
            status="blocked",
            message="Local recovery storage or scope is unavailable or malformed. Preserve the original files.",
        )
    return 2, payload
