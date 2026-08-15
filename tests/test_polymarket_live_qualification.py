from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from dataclasses import replace

import pytest

from simple_ai_trading.polymarket_live import PolymarketLiveBlocked
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
)
from simple_ai_trading.polymarket_live_qualification import (
    VerifiedPolymarketLifecycleQualification,
    load_polymarket_lifecycle_qualification,
    polymarket_account_fingerprint,
    polymarket_credential_fingerprint,
    validate_polymarket_lifecycle_qualification,
)
from simple_ai_trading.polymarket_live_v2 import PolymarketLiveCredentials


NOW_MS = 1_800_000_000_000
OPEN_ORDER_ID = "0x" + "1" * 64
CLOSE_ORDER_ID = "0x" + "2" * 64
FOREIGN_ORDERS_SHA = "3" * 64
FOREIGN_POSITIONS_SHA = "4" * 64
SOURCE_COMMIT = "5" * 40
BOT_ID = "simple-ai-trading-polymarket-btc"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _credentials(*, api_key: str = "api-key-12345678") -> PolymarketLiveCredentials:
    return PolymarketLiveCredentials(
        private_key="0x" + "a" * 64,
        api_key=api_key,
        api_secret="api-secret-12345678",
        api_passphrase="api-passphrase-12345678",
        funder_address="0x" + "b" * 40,
        signature_type=2,
    )


def _event(
    sequence: int,
    event_type: str,
    payload: dict[str, object],
    previous: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "sequence": sequence,
        "event_type": event_type,
        "observed_at_ms": NOW_MS - 20_000 + sequence * 1_000,
        "request_evidence_sha256": _sha(["request", sequence]),
        "response_evidence_sha256": _sha(["response", sequence]),
        "payload": payload,
        "previous_event_sha256": previous,
    }
    return {**body, "event_sha256": _sha(body)}


def _events(*, execution_path: str) -> list[dict[str, object]]:
    payloads: list[tuple[str, dict[str, object]]] = [
        (
            "geoblock_preflight",
            {
                "allowed": True,
                "protocol_version": 2,
                "server_time_skew_ms": 20,
            },
        ),
        (
            "authenticated_account_snapshot_before",
            {
                "authenticated": True,
                "bot_open_order_ids": [],
                "bot_position_quantity": "0",
                "foreign_order_ids_sha256": FOREIGN_ORDERS_SHA,
                "foreign_position_ids_sha256": FOREIGN_POSITIONS_SHA,
                "unknown_state": False,
            },
        ),
        (
            "user_stream_authenticated",
            {
                "authenticated": True,
                "first_inbound_frame_sha256": _sha("first-frame"),
            },
        ),
        (
            "order_submitted",
            {
                "intent_sha256": _sha("open-intent"),
                "expected_order_id": OPEN_ORDER_ID,
                "observed_order_id": OPEN_ORDER_ID,
                "side": "BUY",
                "order_type": "FOK" if execution_path == "filled_and_closed" else "GTD",
                "quantity": "5",
                "limit_price": "0.50",
                "maximum_at_risk_quote": "2.5",
                "status": "MATCHED"
                if execution_path == "filled_and_closed"
                else "LIVE",
                "matched_quantity": "5"
                if execution_path == "filled_and_closed"
                else "0",
                "submission_accepted": True,
            },
        ),
    ]
    if execution_path == "cancelled_unfilled":
        payloads.extend(
            [
                (
                    "order_cancelled",
                    {
                        "requested_order_id": OPEN_ORDER_ID,
                        "cancelled_order_ids": [OPEN_ORDER_ID],
                        "not_cancelled": {},
                        "exact_owned_id_only": True,
                    },
                ),
                (
                    "order_terminal_read",
                    {
                        "order_id": OPEN_ORDER_ID,
                        "status": "CANCELLED",
                        "matched_quantity": "0",
                        "authenticated": True,
                    },
                ),
            ]
        )
    else:
        fill = {
            "order_id": OPEN_ORDER_ID,
            "trade_ids": ["open-trade"],
            "total_quantity": "5",
            "status": "CONFIRMED",
            "fee_accounting_verified": True,
        }
        payloads.extend(
            [
                ("fill_confirmed", fill),
                (
                    "close_submitted",
                    {
                        "parent_order_id": OPEN_ORDER_ID,
                        "intent_sha256": _sha("close-intent"),
                        "expected_order_id": CLOSE_ORDER_ID,
                        "observed_order_id": CLOSE_ORDER_ID,
                        "side": "SELL",
                        "quantity": "5",
                        "status": "MATCHED",
                        "submission_accepted": True,
                    },
                ),
                (
                    "close_fill_confirmed",
                    {
                        **fill,
                        "order_id": CLOSE_ORDER_ID,
                        "trade_ids": ["close-trade"],
                    },
                ),
            ]
        )
    payloads.extend(
        [
            (
                "user_stream_reconnected",
                {
                    "forced_disconnect": True,
                    "reauthenticated": True,
                    "downtime_ms": 1_000,
                    "first_inbound_frame_sha256": _sha("reconnected-frame"),
                },
            ),
            (
                "authenticated_account_snapshot_after_reconnect",
                {
                    "authenticated": True,
                    "bot_open_order_ids": [],
                    "bot_position_quantity": "0",
                    "foreign_order_ids_sha256": FOREIGN_ORDERS_SHA,
                    "foreign_position_ids_sha256": FOREIGN_POSITIONS_SHA,
                    "unknown_state": False,
                },
            ),
            (
                "final_reconciliation",
                {
                    "authenticated": True,
                    "bot_open_order_ids": [],
                    "bot_position_quantity": "0",
                    "foreign_order_ids_sha256": FOREIGN_ORDERS_SHA,
                    "foreign_position_ids_sha256": FOREIGN_POSITIONS_SHA,
                    "unknown_order_state": False,
                    "unknown_fill_state": False,
                    "only_exact_owned_ids_modified": True,
                },
            ),
        ]
    )
    events: list[dict[str, object]] = []
    previous = "0" * 64
    for sequence, (event_type, payload) in enumerate(payloads, start=1):
        event = _event(sequence, event_type, payload, previous)
        events.append(event)
        previous = str(event["event_sha256"])
    return events


def _report(
    *,
    execution_path: str = "cancelled_unfilled",
    credentials: PolymarketLiveCredentials | None = None,
) -> dict[str, object]:
    selected_credentials = credentials or _credentials()
    body: dict[str, object] = {
        "schema_version": "polymarket-authenticated-lifecycle-qualification-v1",
        "qualification_id": _sha("qualification-id"),
        "source_commit": SOURCE_COMMIT,
        "venue": "polymarket",
        "protocol_version": 2,
        "asset": "BTC",
        "market_variant": "fiveminute",
        "environment": "live",
        "bot_id": BOT_ID,
        "account_fingerprint_sha256": polymarket_account_fingerprint(
            selected_credentials
        ),
        "credential_fingerprint_sha256": polymarket_credential_fingerprint(
            selected_credentials
        ),
        "created_at_ms": NOW_MS,
        "expires_at_ms": NOW_MS + 86_400_000,
        "events": _events(execution_path=execution_path),
        "result": {
            "qualified": True,
            "execution_path": execution_path,
            "opening_order_id": OPEN_ORDER_ID,
            "closing_order_id": (
                CLOSE_ORDER_ID if execution_path == "filled_and_closed" else ""
            ),
            "maximum_at_risk_quote": "2.5",
            "secret_material_recorded": False,
            "trading_authority": False,
        },
    }
    return {**body, "qualification_sha256": _sha(body)}


def _write(path: Path, value: dict[str, object]) -> str:
    raw = (_canonical(value) + "\n").encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _rehash(report: dict[str, object]) -> None:
    events = report.get("events")
    if isinstance(events, list):
        previous = "0" * 64
        for event in events:
            if not isinstance(event, dict):
                continue
            event["previous_event_sha256"] = previous
            event_body = dict(event)
            event_body.pop("event_sha256", None)
            event["event_sha256"] = _sha(event_body)
            previous = str(event["event_sha256"])
    body = dict(report)
    body.pop("qualification_sha256", None)
    report["qualification_sha256"] = _sha(body)


def _promotion() -> VerifiedPolymarketLivePromotion:
    promotion = Mock(spec=VerifiedPolymarketLivePromotion)
    promotion.promotion = SimpleNamespace(
        source_commit=SOURCE_COMMIT,
        bot_id=BOT_ID,
        market_variant="fiveminute",
    )
    return promotion


@pytest.mark.parametrize("execution_path", ["cancelled_unfilled", "filled_and_closed"])
def test_loads_both_authenticated_terminal_paths(
    tmp_path: Path,
    execution_path: str,
) -> None:
    path = tmp_path / "qualification.json"
    expected_file_sha = _write(path, _report(execution_path=execution_path))

    verified = load_polymarket_lifecycle_qualification(
        path,
        expected_file_sha256=expected_file_sha,
        observed_at_ms=NOW_MS + 1,
    )

    assert verified.qualification.execution_path == execution_path
    assert verified.qualification.qualified is True
    assert verified.qualification.secret_material_recorded is False
    assert verified.qualification.trading_authority is False
    raw = path.read_bytes()
    for secret in (
        _credentials().private_key,
        _credentials().api_key,
        _credentials().api_secret,
        _credentials().api_passphrase,
    ):
        assert secret.encode("ascii") not in raw
    if execution_path == "filled_and_closed":
        verified.assert_runtime_binding(
            credentials=_credentials(),
            promotion=_promotion(),
            observed_at_ms=NOW_MS + 1,
        )
    else:
        with pytest.raises(PolymarketLiveBlocked, match="fill and parent-bound close"):
            verified.assert_runtime_binding(
                credentials=_credentials(),
                promotion=_promotion(),
                observed_at_ms=NOW_MS + 1,
            )


def test_loader_rejects_file_and_json_tampering(tmp_path: Path) -> None:
    path = tmp_path / "qualification.json"
    _write(path, _report())
    with pytest.raises(ValueError, match="file hash differs"):
        load_polymarket_lifecycle_qualification(
            path,
            expected_file_sha256="f" * 64,
            observed_at_ms=NOW_MS + 1,
        )
    path.write_text(path.read_text(encoding="ascii").rstrip(), encoding="ascii")
    with pytest.raises(ValueError, match="not canonical"):
        load_polymarket_lifecycle_qualification(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            observed_at_ms=NOW_MS + 1,
        )
    raw = (_canonical(_report()) + "\n").replace(
        '"asset":"BTC"',
        '"asset":"BTC","asset":"BTC"',
    )
    path.write_text(raw, encoding="ascii")
    with pytest.raises(ValueError, match="duplicate keys"):
        load_polymarket_lifecycle_qualification(
            path,
            expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            observed_at_ms=NOW_MS + 1,
        )


def test_event_chain_and_semantics_fail_closed() -> None:
    report = _report()
    report["events"][4]["payload"]["requested_order_id"] = CLOSE_ORDER_ID  # type: ignore[index]
    body = dict(report)
    body.pop("qualification_sha256")
    report["qualification_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="event hash differs"):
        validate_polymarket_lifecycle_qualification(report)


def test_zero_or_missing_source_evidence_fails_closed() -> None:
    report = _report()
    report["events"][0]["request_evidence_sha256"] = "0" * 64  # type: ignore[index]
    _rehash(report)

    with pytest.raises(ValueError, match="zero digest"):
        validate_polymarket_lifecycle_qualification(report)


def test_foreign_state_drift_fails_closed() -> None:
    report = _report()
    final = report["events"][-1]["payload"]  # type: ignore[index]
    final["foreign_order_ids_sha256"] = "9" * 64  # type: ignore[index]
    _rehash(report)
    with pytest.raises(ValueError, match="flat and unchanged"):
        validate_polymarket_lifecycle_qualification(report)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("qualified", False),
        ("secret_material_recorded", True),
        ("trading_authority", True),
    ],
)
def test_result_cannot_self_grant_or_record_secrets(key: str, value: bool) -> None:
    report = _report()
    report["result"][key] = value  # type: ignore[index]
    body = dict(report)
    body.pop("qualification_sha256")
    report["qualification_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="does not qualify"):
        validate_polymarket_lifecycle_qualification(report)


def test_runtime_binding_rejects_different_credentials_or_scope(tmp_path: Path) -> None:
    path = tmp_path / "qualification.json"
    expected_file_sha = _write(path, _report(execution_path="filled_and_closed"))
    verified = load_polymarket_lifecycle_qualification(
        path,
        expected_file_sha256=expected_file_sha,
        observed_at_ms=NOW_MS + 1,
    )
    with pytest.raises(PolymarketLiveBlocked, match="differs from this runtime"):
        verified.assert_runtime_binding(
            credentials=_credentials(api_key="different-api-key"),
            promotion=_promotion(),
            observed_at_ms=NOW_MS + 1,
        )
    wrong = _promotion()
    wrong.promotion.market_variant = "fifteenminute"
    with pytest.raises(PolymarketLiveBlocked, match="differs from this promotion"):
        verified.assert_runtime_binding(
            credentials=_credentials(),
            promotion=wrong,
            observed_at_ms=NOW_MS + 1,
        )
    with pytest.raises(PolymarketLiveBlocked, match="expired"):
        verified.assert_runtime_binding(
            credentials=_credentials(),
            promotion=_promotion(),
            observed_at_ms=NOW_MS + 86_400_000,
        )


def test_verified_capability_cannot_be_forged() -> None:
    qualification = validate_polymarket_lifecycle_qualification(_report())
    with pytest.raises(ValueError, match="not verified"):
        VerifiedPolymarketLifecycleQualification(
            qualification=qualification,
            path=Path("qualification.json"),
            file_sha256="f" * 64,
            _capability=object(),
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("scope", "scope"),
        ("source_commit", "source commit"),
        ("bot_id", "bot ID"),
        ("validity", "validity interval"),
        ("event_chronology", "chronology"),
        ("event_type", "event type"),
        ("preflight", "preflight"),
        ("initial_not_flat", "start flat"),
        ("stream_unauthenticated", "user stream"),
        ("reconnect", "reconnect"),
        ("unknown_final", "unknown or foreign"),
        ("opening_identity", "opening order evidence"),
        ("opening_overfill", "fill exceeds"),
        ("cancel_result", "exact cancellation"),
        ("cancel_terminal", "cancellation was not terminal"),
        ("opening_id", "opening order ID"),
        ("unfilled_close_id", "cannot claim a closing order"),
        ("unfilled_match", "opening order was not unfilled"),
        ("filled_open_match", "not fully matched"),
        ("filled_confirmation", "opening fill"),
        ("filled_close", "close order"),
        ("payload_schema", "payload schema"),
    ],
)
def test_malformed_lifecycle_semantics_are_rejected(case: str, match: str) -> None:
    filled = case.startswith("filled_")
    report = _report(
        execution_path="filled_and_closed" if filled else "cancelled_unfilled"
    )
    events = report["events"]
    assert isinstance(events, list)
    by_type = {event["event_type"]: event for event in events}
    result = report["result"]
    assert isinstance(result, dict)
    if case == "scope":
        report["asset"] = "ETH"
    elif case == "source_commit":
        report["source_commit"] = "bad"
    elif case == "bot_id":
        report["bot_id"] = "bad"
    elif case == "validity":
        report["expires_at_ms"] = report["created_at_ms"]
    elif case == "event_chronology":
        events[1]["observed_at_ms"] = 1
    elif case == "event_type":
        events[0]["event_type"] = "!"
    elif case == "preflight":
        by_type["geoblock_preflight"]["payload"]["allowed"] = False
    elif case == "initial_not_flat":
        by_type["authenticated_account_snapshot_before"]["payload"][
            "bot_open_order_ids"
        ] = [OPEN_ORDER_ID]
    elif case == "stream_unauthenticated":
        by_type["user_stream_authenticated"]["payload"]["authenticated"] = False
    elif case == "reconnect":
        by_type["user_stream_reconnected"]["payload"]["reauthenticated"] = False
    elif case == "unknown_final":
        by_type["final_reconciliation"]["payload"]["unknown_fill_state"] = True
    elif case == "opening_identity":
        by_type["order_submitted"]["payload"]["observed_order_id"] = CLOSE_ORDER_ID
    elif case == "opening_overfill":
        by_type["order_submitted"]["payload"]["matched_quantity"] = "6"
    elif case == "cancel_result":
        by_type["order_cancelled"]["payload"]["not_cancelled"] = {
            OPEN_ORDER_ID: "failed"
        }
    elif case == "cancel_terminal":
        by_type["order_terminal_read"]["payload"]["matched_quantity"] = "1"
    elif case == "opening_id":
        result["opening_order_id"] = "bad"
    elif case == "unfilled_close_id":
        result["closing_order_id"] = CLOSE_ORDER_ID
    elif case == "unfilled_match":
        by_type["order_submitted"]["payload"]["matched_quantity"] = "1"
    elif case == "filled_open_match":
        by_type["order_submitted"]["payload"]["matched_quantity"] = "4"
    elif case == "filled_confirmation":
        by_type["fill_confirmed"]["payload"]["status"] = "FAILED"
    elif case == "filled_close":
        by_type["close_submitted"]["payload"]["parent_order_id"] = CLOSE_ORDER_ID
    elif case == "payload_schema":
        by_type["geoblock_preflight"]["payload"]["extra"] = True
    _rehash(report)

    with pytest.raises(ValueError, match=match):
        validate_polymarket_lifecycle_qualification(report)


def test_loader_bounds_and_strict_types(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="file size"):
        load_polymarket_lifecycle_qualification(
            empty,
            expected_file_sha256=hashlib.sha256(b"").hexdigest(),
            observed_at_ms=NOW_MS,
        )
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="not a file"):
        load_polymarket_lifecycle_qualification(
            directory,
            expected_file_sha256="f" * 64,
            observed_at_ms=NOW_MS,
        )
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b'{"value":NaN}\n')
    with pytest.raises(ValueError, match="contains NaN"):
        load_polymarket_lifecycle_qualification(
            invalid,
            expected_file_sha256=hashlib.sha256(invalid.read_bytes()).hexdigest(),
            observed_at_ms=NOW_MS,
        )
    array = tmp_path / "array.json"
    array.write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="not an object"):
        load_polymarket_lifecycle_qualification(
            array,
            expected_file_sha256=hashlib.sha256(array.read_bytes()).hexdigest(),
            observed_at_ms=NOW_MS,
        )


def test_capability_time_and_type_guards(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="credentials"):
        polymarket_account_fingerprint(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="credentials"):
        polymarket_credential_fingerprint(object())  # type: ignore[arg-type]
    path = tmp_path / "qualification.json"
    file_sha = _write(path, _report(execution_path="filled_and_closed"))
    verified = load_polymarket_lifecycle_qualification(
        path,
        expected_file_sha256=file_sha,
        observed_at_ms=NOW_MS + 1,
    )
    with pytest.raises(PolymarketLiveBlocked, match="verified promotion"):
        verified.assert_promotion_binding(
            promotion=object(),  # type: ignore[arg-type]
            observed_at_ms=NOW_MS + 1,
        )
    with pytest.raises(PolymarketLiveBlocked, match="not yet valid"):
        verified.qualification.assert_current(observed_at_ms=NOW_MS - 1)
    unqualified = replace(verified.qualification, qualified=False)
    with pytest.raises(PolymarketLiveBlocked, match="not qualified"):
        unqualified.assert_current(observed_at_ms=NOW_MS + 1)
