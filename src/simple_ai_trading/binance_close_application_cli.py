"""Explicit inspection, bounded collection and application of owned close evidence."""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .api import BinanceAPIError, _default_base_url
from .binance_close_application import (
    apply_spot_closing_recovery,
    inspect_closing_recovery,
)
from .binance_close_intents import validate_closing_client_id
from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import OpenIntentError
from .binance_runtime_control import BinanceRuntimeControl, RuntimeControlError
from .binance_recovery_reads import (
    RecoveryReadProgress,
    _validated_timeout,
    collect_stopped_closing_recovery,
)
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


def _configured_credentials(scope, load_runtime_fn):
    """Resolve current configuration only after local collection admission."""
    if load_runtime_fn is None:
        from .config import load_runtime

        load_runtime_fn = load_runtime
    runtime = load_runtime_fn()
    origin, _ = _default_base_url(
        runtime.testnet, runtime.market_type, demo=getattr(runtime, "demo", False)
    )
    if (
        runtime.market_type != scope.market_type
        or origin != scope.origin
        or BinanceExecutionScope.from_api_key(
            origin, runtime.market_type, runtime.api_key
        )
        != scope
    ):
        raise OpenIntentError("configured credentials differ from original scope")
    return runtime.api_key, runtime.api_secret


def _read_payload(payload, progress):
    states = (progress.order_state, progress.trades_state)
    payload.update(
        account_queried=True
        if "response_received" in states
        else None
        if "attempted" in states
        else False,
        order_read_state=progress.order_state,
        trades_read_state=progress.trades_state,
    )


def run_close_recovery(
    args: argparse.Namespace, *, load_runtime_fn=None
) -> tuple[int, dict]:
    """Keep inspection/application offline; collect only under explicit admission."""
    action = getattr(args, "action", "inspect")
    payload = {
        "schema_version": "owned-close-operator-v2",
        "action": action,
        "account_queried": False,
        "order_submitted": False,
        "rearmed": False,
        "application_outcome": "not_requested" if action != "apply" else "not_started",
        "collection_outcome": "not_started" if action == "collect" else "not_requested",
    }
    progress = RecoveryReadProgress()
    _read_payload(payload, progress)
    try:
        if action not in {"inspect", "collect", "apply"}:
            raise OpenIntentError("choose inspect, collect or apply explicitly")
        client_id = validate_closing_client_id(getattr(args, "client_order_id", None))
        scope_digest = getattr(args, "expected_scope_sha256", None)
        recovery_digest = getattr(args, "expected_recovery_sha256", None)
        request_digest = getattr(args, "expected_request_sha256", None)
        timeout = getattr(args, "read_timeout_seconds", None)
        if action == "apply" and not (
            _digest(scope_digest) and _digest(recovery_digest)
        ):
            raise OpenIntentError(
                "apply requires both exact digests from local inspection"
            )
        if action == "collect" and not (
            _digest(scope_digest) and _digest(request_digest)
        ):
            raise OpenIntentError(
                "collect requires exact scope and request inspection digests"
            )
        if action == "inspect" and any(
            value is not None
            for value in (scope_digest, recovery_digest, request_digest, timeout)
        ):
            raise OpenIntentError(
                "inspection does not accept collection/application-only options"
            )
        if (
            action == "apply" and (request_digest is not None or timeout is not None)
        ) or (action == "collect" and recovery_digest is not None):
            raise OpenIntentError("selected recovery action has incompatible options")
        if action == "collect":
            timeout = _validated_timeout(20.0 if timeout is None else timeout)
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
            request_sha256=inspection.request_sha256,
            symbol=inspection.symbol,
            retained_evidence=observed_recovery_digest is not None,
            retained_evidence_at_inspection=observed_recovery_digest is not None,
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
                    if observed_recovery_digest is not None
                    else "Unresolved request inspected; terminal evidence is absent. No account state was queried."
                ),
            )
            return 0, payload
        if (
            scope_digest != observed_scope_digest
            or (action == "apply" and recovery_digest != observed_recovery_digest)
            or (action == "collect" and request_digest != inspection.request_sha256)
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
        if action == "collect":
            payload["collection_outcome"] = "unknown"
            if not payload["retained_evidence"]:
                payload["retained_evidence"] = None
            collect_stopped_closing_recovery(
                store,
                scope=scope,
                client_id=client_id,
                expected_request_sha256=request_digest,
                control=control,
                credential_provider=lambda: _configured_credentials(
                    scope, load_runtime_fn
                ),
                execution_enrollment_path=getattr(args, "execution_enrollment", None),
                timeout_seconds=timeout,
                progress=progress,
            )
            _read_payload(payload, progress)
            retained = inspect_closing_recovery(store, scope=scope, client_id=client_id)
            if (
                retained.recovery_sha256 is None
                or retained.request_sha256 != request_digest
            ):
                raise OpenIntentError(
                    "collection returned without exact request-bound terminal evidence"
                )
            payload.update(
                status="collected",
                collection_outcome="cached" if progress.retained_reused else "retained",
                retained_evidence=True,
                recovery_sha256=retained.recovery_sha256,
                local_intent_state=retained.local_intent_state,
                application_recorded=retained.application_recorded,
                message="Retained terminal evidence is available; inventory was not applied and Stop remains latched.",
            )
            return 0, payload
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
    except BinanceAPIError:
        payload.update(
            status="blocked",
            message="Recovery read failed or expired. Inspect retained state; no order was submitted.",
        )
    except RuntimeControlError:
        payload.update(
            status="blocked",
            message="Recovery requires available enrolled, ownerless, Stop-latched control. Preserve unresolved ownership and do not rearm.",
        )
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        # Local paths and malformed retained payload values do not enter diagnostics.
        payload.update(
            status="blocked",
            message="Local recovery storage or scope is unavailable or malformed. Preserve the original files.",
        )
    _read_payload(payload, progress)
    return 2, payload
