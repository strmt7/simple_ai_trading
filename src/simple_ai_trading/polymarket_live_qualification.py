"""Evidence-bound qualification for the authenticated Polymarket lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .polymarket_live import PolymarketLiveBlocked
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion
from .polymarket_live_v2 import PolymarketLiveCredentials


POLYMARKET_LIFECYCLE_QUALIFICATION_SCHEMA_VERSION = (
    "polymarket-authenticated-lifecycle-qualification-v1"
)
_MAXIMUM_QUALIFICATION_BYTES = 2 * 1024 * 1024
_MAXIMUM_VALIDITY_MS = 2_592_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ORDER_ID = re.compile(r"^0x[0-9a-f]{64}$")
_VERIFIED_CAPABILITY = object()
_ZERO_SHA256 = "0" * 64
_UNFILLED_EVENTS = (
    "geoblock_preflight",
    "authenticated_account_snapshot_before",
    "user_stream_authenticated",
    "order_submitted",
    "order_cancelled",
    "order_terminal_read",
    "user_stream_reconnected",
    "authenticated_account_snapshot_after_reconnect",
    "final_reconciliation",
)
_FILLED_EVENTS = (
    "geoblock_preflight",
    "authenticated_account_snapshot_before",
    "user_stream_authenticated",
    "order_submitted",
    "fill_confirmed",
    "close_submitted",
    "close_fill_confirmed",
    "user_stream_reconnected",
    "authenticated_account_snapshot_after_reconnect",
    "final_reconciliation",
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Polymarket lifecycle JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket lifecycle JSON contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _evidence_sha(value: object, *, name: str) -> str:
    normalized = _sha(value, name=name)
    if normalized == _ZERO_SHA256:
        raise ValueError(f"{name} cannot be the zero digest")
    return normalized


def _decimal(value: object, *, name: str, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (nonnegative and parsed < 0):
        raise ValueError(f"{name} must be a finite nonnegative decimal")
    return parsed


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} schema is invalid")


def _string_list(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicates")
    return result


def polymarket_account_fingerprint(credentials: PolymarketLiveCredentials) -> str:
    """Return a non-secret binding for the dedicated wallet configuration."""

    if not isinstance(credentials, PolymarketLiveCredentials):
        raise TypeError("credentials must be PolymarketLiveCredentials")
    return _canonical_sha256(
        {
            "domain": "polymarket-account-fingerprint-v1",
            "funder_address": credentials.funder_address,
            "signature_type": credentials.signature_type,
        }
    )


def polymarket_credential_fingerprint(credentials: PolymarketLiveCredentials) -> str:
    """Bind exact credentials without retaining or combining their values."""

    if not isinstance(credentials, PolymarketLiveCredentials):
        raise TypeError("credentials must be PolymarketLiveCredentials")
    digest = hashlib.sha256(b"polymarket-credential-fingerprint-v1\0")
    for label, value in (
        ("private_key", credentials.private_key),
        ("api_key", credentials.api_key),
        ("api_secret", credentials.api_secret),
        ("api_passphrase", credentials.api_passphrase),
        ("funder_address", credentials.funder_address),
        ("signature_type", str(credentials.signature_type)),
    ):
        encoded = value.encode("utf-8", errors="strict")
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PolymarketLifecycleEvidenceEvent:
    sequence: int
    event_type: str
    observed_at_ms: int
    request_evidence_sha256: str
    response_evidence_sha256: str
    payload: Mapping[str, object]
    previous_event_sha256: str
    event_sha256: str


@dataclass(frozen=True, slots=True)
class PolymarketLifecycleQualification:
    qualification_id: str
    qualification_sha256: str
    source_commit: str
    bot_id: str
    market_variant: str
    account_fingerprint_sha256: str
    credential_fingerprint_sha256: str
    created_at_ms: int
    expires_at_ms: int
    execution_path: str
    opening_order_id: str
    closing_order_id: str
    maximum_at_risk_quote: Decimal
    events: tuple[PolymarketLifecycleEvidenceEvent, ...]
    qualified: bool
    secret_material_recorded: bool
    trading_authority: bool

    def assert_current(self, *, observed_at_ms: int) -> None:
        now = int(observed_at_ms)
        if now < self.created_at_ms:
            raise PolymarketLiveBlocked(
                "Polymarket lifecycle qualification is not yet valid"
            )
        if now >= self.expires_at_ms:
            raise PolymarketLiveBlocked("Polymarket lifecycle qualification expired")
        if not self.qualified:
            raise PolymarketLiveBlocked(
                "Polymarket authenticated lifecycle is not qualified"
            )


@dataclass(frozen=True, slots=True)
class VerifiedPolymarketLifecycleQualification:
    """Loader-only capability proving the sanitized lifecycle report parsed."""

    qualification: PolymarketLifecycleQualification
    path: Path
    file_sha256: str
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_CAPABILITY:
            raise ValueError("Polymarket lifecycle evidence was not verified")

    def assert_runtime_binding(
        self,
        *,
        credentials: PolymarketLiveCredentials,
        promotion: VerifiedPolymarketLivePromotion,
        observed_at_ms: int,
    ) -> None:
        if not isinstance(promotion, VerifiedPolymarketLivePromotion):
            raise PolymarketLiveBlocked(
                "Polymarket lifecycle requires verified promotion evidence"
            )
        self.assert_promotion_binding(
            promotion=promotion,
            observed_at_ms=observed_at_ms,
        )
        report = self.qualification
        if report.account_fingerprint_sha256 != polymarket_account_fingerprint(
            credentials
        ) or report.credential_fingerprint_sha256 != polymarket_credential_fingerprint(
            credentials
        ):
            raise PolymarketLiveBlocked(
                "Polymarket lifecycle evidence differs from this runtime"
            )

    def assert_promotion_binding(
        self,
        *,
        promotion: VerifiedPolymarketLivePromotion,
        observed_at_ms: int,
    ) -> None:
        if not isinstance(promotion, VerifiedPolymarketLivePromotion):
            raise PolymarketLiveBlocked(
                "Polymarket lifecycle requires verified promotion evidence"
            )
        report = self.qualification
        policy = promotion.promotion
        report.assert_current(observed_at_ms=observed_at_ms)
        if (
            report.source_commit != policy.source_commit
            or report.bot_id != policy.bot_id
            or report.market_variant != policy.market_variant
        ):
            raise PolymarketLiveBlocked(
                "Polymarket lifecycle evidence differs from this promotion"
            )
        if report.execution_path != "filled_and_closed":
            raise PolymarketLiveBlocked(
                "Polymarket lifecycle qualification does not prove a confirmed "
                "fill and parent-bound close"
            )


def _validate_event_chain(
    values: object,
) -> tuple[PolymarketLifecycleEvidenceEvent, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Polymarket lifecycle events must be an array")
    if not values or len(values) > 32:
        raise ValueError("Polymarket lifecycle event count is invalid")
    events: list[PolymarketLifecycleEvidenceEvent] = []
    previous = _ZERO_SHA256
    last_time = 0
    for expected_sequence, value in enumerate(values, start=1):
        event = dict(_mapping(value, name="lifecycle event"))
        _exact_keys(
            event,
            {
                "sequence",
                "event_type",
                "observed_at_ms",
                "request_evidence_sha256",
                "response_evidence_sha256",
                "payload",
                "previous_event_sha256",
                "event_sha256",
            },
            name="lifecycle event",
        )
        sequence = int(event["sequence"])
        observed = int(event["observed_at_ms"])
        if sequence != expected_sequence or observed <= 0 or observed < last_time:
            raise ValueError("Polymarket lifecycle event chronology is invalid")
        event_type = str(event["event_type"] or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", event_type):
            raise ValueError("Polymarket lifecycle event type is invalid")
        payload = dict(_mapping(event["payload"], name="lifecycle event payload"))
        claimed_previous = _sha(
            event["previous_event_sha256"],
            name="previous lifecycle event hash",
        )
        claimed_event = _sha(event["event_sha256"], name="lifecycle event hash")
        if claimed_previous != previous:
            raise ValueError("Polymarket lifecycle event chain differs")
        body = dict(event)
        body.pop("event_sha256")
        if _canonical_sha256(body) != claimed_event:
            raise ValueError("Polymarket lifecycle event hash differs")
        parsed = PolymarketLifecycleEvidenceEvent(
            sequence=sequence,
            event_type=event_type,
            observed_at_ms=observed,
            request_evidence_sha256=_evidence_sha(
                event["request_evidence_sha256"],
                name="lifecycle request evidence hash",
            ),
            response_evidence_sha256=_evidence_sha(
                event["response_evidence_sha256"],
                name="lifecycle response evidence hash",
            ),
            payload=payload,
            previous_event_sha256=claimed_previous,
            event_sha256=claimed_event,
        )
        events.append(parsed)
        previous = claimed_event
        last_time = observed
    return tuple(events)


def _event_payload(
    events: Mapping[str, PolymarketLifecycleEvidenceEvent],
    event_type: str,
    expected_keys: set[str],
) -> Mapping[str, object]:
    payload = events[event_type].payload
    _exact_keys(payload, expected_keys, name=f"{event_type} payload")
    return payload


def _validate_common_events(
    events: tuple[PolymarketLifecycleEvidenceEvent, ...],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    by_type = {event.event_type: event for event in events}
    if len(by_type) != len(events):
        raise ValueError("Polymarket lifecycle event types are duplicated")
    geoblock = _event_payload(
        by_type,
        "geoblock_preflight",
        {"allowed", "protocol_version", "server_time_skew_ms"},
    )
    if (
        geoblock["allowed"] is not True
        or geoblock["protocol_version"] != 2
        or not 0 <= int(geoblock["server_time_skew_ms"]) <= 5_000
    ):
        raise ValueError("Polymarket lifecycle preflight did not pass")
    before = _event_payload(
        by_type,
        "authenticated_account_snapshot_before",
        {
            "authenticated",
            "bot_open_order_ids",
            "bot_position_quantity",
            "foreign_order_ids_sha256",
            "foreign_position_ids_sha256",
            "unknown_state",
        },
    )
    if (
        before["authenticated"] is not True
        or _string_list(before["bot_open_order_ids"], name="initial bot open order IDs")
        or _decimal(
            before["bot_position_quantity"],
            name="initial bot position quantity",
            nonnegative=True,
        )
        != 0
        or before["unknown_state"] is not False
    ):
        raise ValueError("Polymarket lifecycle did not start flat and known")
    _evidence_sha(before["foreign_order_ids_sha256"], name="foreign order set hash")
    _evidence_sha(
        before["foreign_position_ids_sha256"], name="foreign position set hash"
    )
    stream = _event_payload(
        by_type,
        "user_stream_authenticated",
        {"authenticated", "first_inbound_frame_sha256"},
    )
    if stream["authenticated"] is not True:
        raise ValueError("Polymarket lifecycle user stream was not authenticated")
    _evidence_sha(stream["first_inbound_frame_sha256"], name="first inbound frame hash")
    reconnect = _event_payload(
        by_type,
        "user_stream_reconnected",
        {
            "forced_disconnect",
            "reauthenticated",
            "downtime_ms",
            "first_inbound_frame_sha256",
        },
    )
    if (
        reconnect["forced_disconnect"] is not True
        or reconnect["reauthenticated"] is not True
        or not 1 <= int(reconnect["downtime_ms"]) <= 60_000
    ):
        raise ValueError("Polymarket lifecycle reconnect was not proven")
    _evidence_sha(
        reconnect["first_inbound_frame_sha256"],
        name="reconnected inbound frame hash",
    )
    after = _event_payload(
        by_type,
        "authenticated_account_snapshot_after_reconnect",
        {
            "authenticated",
            "bot_open_order_ids",
            "bot_position_quantity",
            "foreign_order_ids_sha256",
            "foreign_position_ids_sha256",
            "unknown_state",
        },
    )
    final = _event_payload(
        by_type,
        "final_reconciliation",
        {
            "authenticated",
            "bot_open_order_ids",
            "bot_position_quantity",
            "foreign_order_ids_sha256",
            "foreign_position_ids_sha256",
            "unknown_order_state",
            "unknown_fill_state",
            "only_exact_owned_ids_modified",
        },
    )
    for payload, label in ((after, "post-reconnect"), (final, "final")):
        if (
            payload["authenticated"] is not True
            or _string_list(
                payload["bot_open_order_ids"],
                name=f"{label} bot open order IDs",
            )
            or _decimal(
                payload["bot_position_quantity"],
                name=f"{label} bot position quantity",
                nonnegative=True,
            )
            != 0
            or payload["foreign_order_ids_sha256"] != before["foreign_order_ids_sha256"]
            or payload["foreign_position_ids_sha256"]
            != before["foreign_position_ids_sha256"]
        ):
            raise ValueError(
                "Polymarket lifecycle final account state is not flat and unchanged"
            )
    if (
        after["unknown_state"] is not False
        or final["unknown_order_state"] is not False
        or final["unknown_fill_state"] is not False
        or final["only_exact_owned_ids_modified"] is not True
    ):
        raise ValueError("Polymarket lifecycle retained unknown or foreign state")
    return before, final


def _validate_opening_order(
    events: Mapping[str, PolymarketLifecycleEvidenceEvent],
    *,
    expected_order_id: str,
    maximum_at_risk_quote: Decimal,
) -> tuple[Decimal, str]:
    submitted = _event_payload(
        events,
        "order_submitted",
        {
            "intent_sha256",
            "expected_order_id",
            "observed_order_id",
            "side",
            "order_type",
            "quantity",
            "limit_price",
            "maximum_at_risk_quote",
            "status",
            "matched_quantity",
            "submission_accepted",
        },
    )
    _evidence_sha(submitted["intent_sha256"], name="opening intent hash")
    quantity = _decimal(submitted["quantity"], name="opening quantity")
    limit_price = _decimal(submitted["limit_price"], name="opening limit price")
    submitted_risk = _decimal(
        submitted["maximum_at_risk_quote"],
        name="submitted maximum at-risk quote",
        nonnegative=True,
    )
    if (
        submitted["expected_order_id"] != expected_order_id
        or submitted["observed_order_id"] != expected_order_id
        or submitted["side"] != "BUY"
        or submitted["order_type"] not in {"FAK", "FOK", "GTD"}
        or quantity <= 0
        or not Decimal("0") < limit_price < Decimal("1")
        or submitted_risk != maximum_at_risk_quote
        or submitted_risk <= 0
        or submitted_risk > Decimal("10")
        or submitted["status"] not in {"LIVE", "MATCHED"}
        or submitted["submission_accepted"] is not True
    ):
        raise ValueError("Polymarket lifecycle opening order evidence differs")
    matched = _decimal(
        submitted["matched_quantity"],
        name="opening matched quantity",
        nonnegative=True,
    )
    if matched > quantity:
        raise ValueError("Polymarket lifecycle opening fill exceeds quantity")
    return quantity, str(submitted["status"])


def _validate_unfilled_path(
    events: Mapping[str, PolymarketLifecycleEvidenceEvent],
    *,
    order_id: str,
) -> None:
    cancelled = _event_payload(
        events,
        "order_cancelled",
        {
            "requested_order_id",
            "cancelled_order_ids",
            "not_cancelled",
            "exact_owned_id_only",
        },
    )
    not_cancelled = _mapping(cancelled["not_cancelled"], name="not-cancelled map")
    if (
        cancelled["requested_order_id"] != order_id
        or _string_list(cancelled["cancelled_order_ids"], name="cancelled order IDs")
        != (order_id,)
        or not not_cancelled == {}
        or cancelled["exact_owned_id_only"] is not True
    ):
        raise ValueError("Polymarket lifecycle exact cancellation differs")
    terminal = _event_payload(
        events,
        "order_terminal_read",
        {"order_id", "status", "matched_quantity", "authenticated"},
    )
    if (
        terminal["order_id"] != order_id
        or terminal["status"] not in {"CANCELED", "CANCELLED"}
        or _decimal(
            terminal["matched_quantity"],
            name="terminal matched quantity",
            nonnegative=True,
        )
        != 0
        or terminal["authenticated"] is not True
    ):
        raise ValueError("Polymarket lifecycle cancellation was not terminal")


def _validate_confirmed_fill(
    payload: Mapping[str, object],
    *,
    event_name: str,
    order_id: str,
    quantity: Decimal,
) -> None:
    if (
        payload["order_id"] != order_id
        or not _string_list(payload["trade_ids"], name=f"{event_name} trade IDs")
        or _decimal(
            payload["total_quantity"],
            name=f"{event_name} quantity",
            nonnegative=True,
        )
        != quantity
        or payload["status"] != "CONFIRMED"
        or payload["fee_accounting_verified"] is not True
    ):
        raise ValueError(f"Polymarket lifecycle {event_name} differs")


def _validate_filled_path(
    events: Mapping[str, PolymarketLifecycleEvidenceEvent],
    *,
    opening_order_id: str,
    closing_order_id: str,
    quantity: Decimal,
) -> None:
    fill_keys = {
        "order_id",
        "trade_ids",
        "total_quantity",
        "status",
        "fee_accounting_verified",
    }
    opening_fill = _event_payload(events, "fill_confirmed", fill_keys)
    _validate_confirmed_fill(
        opening_fill,
        event_name="opening fill",
        order_id=opening_order_id,
        quantity=quantity,
    )
    close = _event_payload(
        events,
        "close_submitted",
        {
            "parent_order_id",
            "intent_sha256",
            "expected_order_id",
            "observed_order_id",
            "side",
            "quantity",
            "status",
            "submission_accepted",
        },
    )
    _evidence_sha(close["intent_sha256"], name="closing intent hash")
    if (
        close["parent_order_id"] != opening_order_id
        or close["expected_order_id"] != closing_order_id
        or close["observed_order_id"] != closing_order_id
        or close["side"] != "SELL"
        or _decimal(close["quantity"], name="closing quantity") != quantity
        or close["status"] != "MATCHED"
        or close["submission_accepted"] is not True
    ):
        raise ValueError("Polymarket lifecycle close order differs")
    closing_fill = _event_payload(events, "close_fill_confirmed", fill_keys)
    _validate_confirmed_fill(
        closing_fill,
        event_name="closing fill",
        order_id=closing_order_id,
        quantity=quantity,
    )


def validate_polymarket_lifecycle_qualification(
    value: Mapping[str, object],
) -> PolymarketLifecycleQualification:
    payload = dict(value)
    _exact_keys(
        payload,
        {
            "schema_version",
            "qualification_id",
            "qualification_sha256",
            "source_commit",
            "venue",
            "protocol_version",
            "asset",
            "market_variant",
            "environment",
            "bot_id",
            "account_fingerprint_sha256",
            "credential_fingerprint_sha256",
            "created_at_ms",
            "expires_at_ms",
            "events",
            "result",
        },
        name="Polymarket lifecycle qualification",
    )
    if (
        payload["schema_version"] != POLYMARKET_LIFECYCLE_QUALIFICATION_SCHEMA_VERSION
        or payload["venue"] != "polymarket"
        or payload["protocol_version"] != 2
        or payload["asset"] != "BTC"
        or payload["market_variant"] not in {"fiveminute", "fifteenminute"}
        or payload["environment"] != "live"
    ):
        raise ValueError("Polymarket lifecycle qualification scope is invalid")
    claimed = _sha(payload["qualification_sha256"], name="qualification hash")
    body = dict(payload)
    body.pop("qualification_sha256")
    if _canonical_sha256(body) != claimed:
        raise ValueError("Polymarket lifecycle qualification hash differs")
    qualification_id = _evidence_sha(
        payload["qualification_id"], name="qualification ID"
    )
    source_commit = str(payload["source_commit"] or "").strip().lower()
    if _GIT_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("Polymarket lifecycle source commit is invalid")
    bot_id = str(payload["bot_id"] or "").strip()
    if not 8 <= len(bot_id) <= 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", bot_id):
        raise ValueError("Polymarket lifecycle bot ID is invalid")
    created = int(payload["created_at_ms"])
    expires = int(payload["expires_at_ms"])
    if created <= 0 or expires <= created or expires - created > _MAXIMUM_VALIDITY_MS:
        raise ValueError("Polymarket lifecycle validity interval is invalid")
    events = _validate_event_chain(payload["events"])
    if events[-1].observed_at_ms > created:
        raise ValueError("Polymarket lifecycle report creation chronology is invalid")
    result = dict(_mapping(payload["result"], name="lifecycle result"))
    _exact_keys(
        result,
        {
            "qualified",
            "execution_path",
            "opening_order_id",
            "closing_order_id",
            "maximum_at_risk_quote",
            "secret_material_recorded",
            "trading_authority",
        },
        name="Polymarket lifecycle result",
    )
    execution_path = str(result["execution_path"])
    expected_events = (
        _UNFILLED_EVENTS
        if execution_path == "cancelled_unfilled"
        else _FILLED_EVENTS
        if execution_path == "filled_and_closed"
        else ()
    )
    if tuple(event.event_type for event in events) != expected_events:
        raise ValueError("Polymarket lifecycle execution path is invalid")
    if (
        result["qualified"] is not True
        or result["secret_material_recorded"] is not False
        or result["trading_authority"] is not False
    ):
        raise ValueError("Polymarket lifecycle result does not qualify")
    opening_order_id = str(result["opening_order_id"] or "").strip().lower()
    closing_order_id = str(result["closing_order_id"] or "").strip().lower()
    if _ORDER_ID.fullmatch(opening_order_id) is None:
        raise ValueError("Polymarket lifecycle opening order ID is invalid")
    if execution_path == "filled_and_closed":
        if (
            _ORDER_ID.fullmatch(closing_order_id) is None
            or closing_order_id == opening_order_id
        ):
            raise ValueError("Polymarket lifecycle closing order ID is invalid")
    elif closing_order_id:
        raise ValueError("Unfilled lifecycle cannot claim a closing order")
    maximum_at_risk = _decimal(
        result["maximum_at_risk_quote"],
        name="lifecycle maximum at-risk quote",
        nonnegative=True,
    )
    _validate_common_events(events)
    event_map = {event.event_type: event for event in events}
    quantity, opening_status = _validate_opening_order(
        event_map,
        expected_order_id=opening_order_id,
        maximum_at_risk_quote=maximum_at_risk,
    )
    if execution_path == "cancelled_unfilled":
        if (
            opening_status != "LIVE"
            or _decimal(
                event_map["order_submitted"].payload["matched_quantity"],
                name="unfilled opening matched quantity",
                nonnegative=True,
            )
            != 0
        ):
            raise ValueError("Unfilled lifecycle opening order was not unfilled")
        _validate_unfilled_path(event_map, order_id=opening_order_id)
    else:
        if (
            opening_status != "MATCHED"
            or _decimal(
                event_map["order_submitted"].payload["matched_quantity"],
                name="opening matched quantity",
                nonnegative=True,
            )
            != quantity
        ):
            raise ValueError("Filled lifecycle opening order was not fully matched")
        _validate_filled_path(
            event_map,
            opening_order_id=opening_order_id,
            closing_order_id=closing_order_id,
            quantity=quantity,
        )
    return PolymarketLifecycleQualification(
        qualification_id=qualification_id,
        qualification_sha256=claimed,
        source_commit=source_commit,
        bot_id=bot_id,
        market_variant=str(payload["market_variant"]),
        account_fingerprint_sha256=_evidence_sha(
            payload["account_fingerprint_sha256"],
            name="account fingerprint",
        ),
        credential_fingerprint_sha256=_evidence_sha(
            payload["credential_fingerprint_sha256"],
            name="credential fingerprint",
        ),
        created_at_ms=created,
        expires_at_ms=expires,
        execution_path=execution_path,
        opening_order_id=opening_order_id,
        closing_order_id=closing_order_id,
        maximum_at_risk_quote=maximum_at_risk,
        events=events,
        qualified=True,
        secret_material_recorded=False,
        trading_authority=False,
    )


def load_polymarket_lifecycle_qualification(
    path: str | Path,
    *,
    expected_file_sha256: str,
    observed_at_ms: int,
) -> VerifiedPolymarketLifecycleQualification:
    qualification_path = Path(path)
    if qualification_path.is_symlink():
        raise ValueError("Polymarket lifecycle qualification cannot be a symlink")
    resolved = qualification_path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Polymarket lifecycle qualification is not a file")
    with resolved.open("rb") as handle:
        raw = handle.read(_MAXIMUM_QUALIFICATION_BYTES + 1)
    if not raw or len(raw) > _MAXIMUM_QUALIFICATION_BYTES:
        raise ValueError("Polymarket lifecycle qualification file size is invalid")
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != _sha(expected_file_sha256, name="qualification file hash"):
        raise ValueError("Polymarket lifecycle qualification file hash differs")
    try:
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Polymarket lifecycle qualification is not strict JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise ValueError("Polymarket lifecycle qualification is not an object")
    canonical = (_canonical_json(value) + "\n").encode("ascii")
    if raw != canonical:
        raise ValueError("Polymarket lifecycle qualification is not canonical")
    qualification = validate_polymarket_lifecycle_qualification(value)
    qualification.assert_current(observed_at_ms=observed_at_ms)
    return VerifiedPolymarketLifecycleQualification(
        qualification=qualification,
        path=resolved,
        file_sha256=file_sha,
        _capability=_VERIFIED_CAPABILITY,
    )


__all__ = [
    "POLYMARKET_LIFECYCLE_QUALIFICATION_SCHEMA_VERSION",
    "PolymarketLifecycleEvidenceEvent",
    "PolymarketLifecycleQualification",
    "VerifiedPolymarketLifecycleQualification",
    "load_polymarket_lifecycle_qualification",
    "polymarket_account_fingerprint",
    "polymarket_credential_fingerprint",
    "validate_polymarket_lifecycle_qualification",
]
