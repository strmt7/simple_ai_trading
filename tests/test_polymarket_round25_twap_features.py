from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_DECISION_CADENCE_MS,
    POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
    POLYMARKET_ROUND25_TWAP_SYMBOL,
    POLYMARKET_ROUND25_TWAP_TOPIC,
    POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256,
    Round25TwapFeatureEngine,
    Round25TwapObservation,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-twap-native-model-design-v1.json"
)
CORRECTION_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-twap-wire-schema-correction-v1.json"
)
START_MS = 1_800_000_000_000
CONDITION_ID = "0x" + "a" * 64
E18 = 1_000_000_000_000_000_000
BASE_VALUE_E18 = 65_000 * E18


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _raw_frame(
    *,
    source_ms: int = START_MS,
    exact_value_e18: int = BASE_VALUE_E18,
    publisher_ms: int | None = None,
    display_value: object | None = None,
) -> str:
    if publisher_ms is None:
        publisher_ms = source_ms + 100
    if display_value is None:
        display_value = exact_value_e18 / E18
    return json.dumps(
        {
            "topic": POLYMARKET_ROUND25_TWAP_TOPIC,
            "type": "update",
            "timestamp": publisher_ms,
            "payload": {
                "symbol": POLYMARKET_ROUND25_TWAP_SYMBOL,
                "value": display_value,
                "full_accuracy_value": str(exact_value_e18),
                "timestamp": source_ms,
            },
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def _observation(
    offset_ms: int,
    *,
    value_delta_e18: int = 0,
    sequence: int = 1,
    source_ms: int | None = None,
    publisher_ms: int | None = None,
    received_ms: int | None = None,
) -> Round25TwapObservation:
    source = START_MS + offset_ms if source_ms is None else source_ms
    publisher = source + 100 if publisher_ms is None else publisher_ms
    received = source + 200 if received_ms is None else received_ms
    identity = f"{source}:{publisher}:{received}:{value_delta_e18}:{sequence}"
    return Round25TwapObservation(
        source_timestamp_ms=source,
        publisher_timestamp_ms=publisher,
        received_wall_ms=received,
        received_monotonic_ns=1_000_000_000 + sequence,
        full_accuracy_value_e18=BASE_VALUE_E18 + value_delta_e18,
        raw_frame_sha256=hashlib.sha256(identity.encode("ascii")).hexdigest(),
    )


def _ready_engine() -> Round25TwapFeatureEngine:
    engine = Round25TwapFeatureEngine(
        condition_id=CONDITION_ID,
        event_start_ms=START_MS,
    )
    for sequence, offset_ms in enumerate((0, 500, 1_000, 1_500, 2_000), start=1):
        engine.ingest(
            _observation(
                offset_ms,
                value_delta_e18=sequence * 10**15,
                sequence=sequence,
            )
        )
    return engine


def _feature(snapshot: object, name: str) -> float:
    index = POLYMARKET_ROUND25_TWAP_FEATURE_NAMES.index(name)
    return snapshot.values[index]  # type: ignore[attr-defined,no-any-return]


def test_design_is_self_hashed_target_blind_and_non_authoritative() -> None:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    declared = payload.pop("design_sha256")

    assert declared == _canonical_sha256(payload)
    assert declared == POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    assert payload["status"] == "frozen_target_blind_before_round25_v2_capture_start"
    assert payload["source_contracts"]["rtds_topic"] == "crypto_prices_twap_thirty"
    assert payload["settlement_hypothesis"]["verification_status"].startswith(
        "unverified_"
    )
    assert payload["settlement_hypothesis"]["nearest_tick_substitution_allowed"] is False
    assert payload["settlement_hypothesis"]["binance_price_substitution_allowed"] is False
    assert payload["invalidated_basis"]["admissible_for_round25_v2"] is False
    assert payload["ai_assist"]["safety_override_allowed"] is False
    assert not any(payload["truth_state"].values())


def test_live_wire_schema_correction_is_self_hashed_and_source_only() -> None:
    payload = json.loads(CORRECTION_PATH.read_text(encoding="utf-8"))
    declared = payload.pop("correction_sha256")

    assert declared == _canonical_sha256(payload)
    assert declared == POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256
    assert payload["parent_twap_native_model_design_sha256"] == (
        POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    )
    assert payload["public_wire_probe"]["topic"] == POLYMARKET_ROUND25_TWAP_TOPIC
    assert payload["public_wire_probe"]["window_s_present"] is False
    assert payload["corrected_parser_contract"]["wire_window_field_allowed"] is False
    assert not any(payload["truth_state"].values())


def test_feature_schema_has_no_structural_probability_or_target() -> None:
    assert len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES) == 37
    assert len(set(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES)) == 37
    assert all(
        forbidden not in name
        for name in POLYMARKET_ROUND25_TWAP_FEATURE_NAMES
        for forbidden in ("probability", "outcome", "target", "resolution")
    )
    assert POLYMARKET_ROUND25_DECISION_CADENCE_MS == 250


def test_exact_wire_frame_parses_without_using_display_value_as_price() -> None:
    exact = 64_683_319_400_908_183_306_240
    raw = _raw_frame(exact_value_e18=exact)
    observation = Round25TwapObservation.from_raw_frame(
        raw,
        received_wall_ms=START_MS + 300,
        received_monotonic_ns=123_456,
    )

    assert observation.full_accuracy_value_e18 == exact
    assert observation.source_timestamp_ms == START_MS
    assert observation.publisher_timestamp_ms == START_MS + 100
    assert observation.raw_frame_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert observation.symbol == "btc/usd"
    assert observation.window_seconds == 30


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda event: event.update(topic="crypto_prices_twap_thirty"), "identity"),
        (lambda event: event.update(type="snapshot"), "identity"),
        (lambda event: event["payload"].update(symbol="BTC/USD"), "identity"),
        (lambda event: event["payload"].update(window_s=30), "identity"),
        (lambda event: event["payload"].update(value="65000"), "display"),
        (lambda event: event["payload"].update(value=True), "display"),
        (lambda event: event["payload"].update(value=1.0), "display"),
        (lambda event: event["payload"].update(full_accuracy_value="1.2"), "exact"),
        (lambda event: event["payload"].update(full_accuracy_value="-1"), "exact"),
        (lambda event: event.update(timestamp=True), "positive integer"),
    ],
)
def test_wire_frame_rejects_wrong_identity_or_value(mutator: object, error: str) -> None:
    event = json.loads(_raw_frame())
    mutator(event)  # type: ignore[operator]
    raw = json.dumps(event, separators=(",", ":"))

    with pytest.raises(ValueError, match=error):
        Round25TwapObservation.from_raw_frame(
            raw,
            received_wall_ms=START_MS + 300,
            received_monotonic_ns=123_456,
        )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        '{"topic":"a","topic":"b"}',
        '{"topic":NaN}',
    ],
)
def test_wire_frame_rejects_non_strict_json(raw: str) -> None:
    with pytest.raises(ValueError):
        Round25TwapObservation.from_raw_frame(
            raw,
            received_wall_ms=START_MS + 300,
            received_monotonic_ns=123_456,
        )


def test_observation_constructor_rejects_bool_integer_substitution() -> None:
    valid = _observation(0)

    for field in (
        "source_timestamp_ms",
        "publisher_timestamp_ms",
        "received_wall_ms",
        "received_monotonic_ns",
        "full_accuracy_value_e18",
    ):
        with pytest.raises(ValueError):
            replace(valid, **{field: True})


def test_observation_rejects_publisher_time_after_local_receipt() -> None:
    with pytest.raises(ValueError, match="observation is invalid"):
        replace(
            _observation(0),
            publisher_timestamp_ms=START_MS + 500,
            received_wall_ms=START_MS + 499,
        )


def test_available_snapshot_is_causal_deterministic_and_target_blind() -> None:
    first = _ready_engine().build(START_MS + 2_250)
    second = _ready_engine().build(START_MS + 2_250)
    expected_distance = math.log(
        (BASE_VALUE_E18 + 5 * 10**15) / (BASE_VALUE_E18 + 10**15)
    )

    assert first == second
    assert first.available is True
    assert first.reasons == ()
    assert first.trading_authority is False
    assert first.maximum_receipt_ms <= first.decision_time_ms
    assert first.source_chain_sha256 != hashlib.sha256(b"").hexdigest()
    assert _feature(first, "twap.reference.log_distance_from_exact_open") == pytest.approx(
        expected_distance
    )
    assert _feature(first, "twap.path.log_return_1s_available") == 1.0
    assert _feature(first, "twap.path.log_return_5s_available") == 0.0
    assert _feature(first, "twap.path.nonoverlapping_scale_available") == 0.0
    assert _feature(first, "twap.transport.observation_count") == 5.0
    assert _feature(first, "twap.transport.coverage_seconds") == 2.0


@pytest.mark.parametrize(
    ("offsets", "decision_offset", "reason"),
    [
        ((500, 1_000, 1_500, 2_000, 2_500), 2_750, "exact_opening_twap_unavailable"),
        ((0, 500, 1_000, 1_500), 1_750, "twap_observation_count_below_minimum"),
        ((0, 250, 500, 750, 1_000), 1_250, "twap_coverage_below_minimum"),
        ((0, 500, 1_000, 1_500, 2_000), 10_000, "twap_source_stale"),
    ],
)
def test_admission_rejects_missing_or_stale_evidence(
    offsets: tuple[int, ...], decision_offset: int, reason: str
) -> None:
    engine = Round25TwapFeatureEngine(
        condition_id=CONDITION_ID,
        event_start_ms=START_MS,
    )
    for sequence, offset in enumerate(offsets, start=1):
        engine.ingest(_observation(offset, sequence=sequence))

    snapshot = engine.build(START_MS + decision_offset)

    assert snapshot.available is False
    assert reason in snapshot.reasons
    assert not any(snapshot.values)
    assert snapshot.trading_authority is False


def test_stream_gap_is_fail_closed() -> None:
    engine = _ready_engine()
    engine.mark_stream_gap()

    snapshot = engine.build(START_MS + 2_250)

    assert snapshot.available is False
    assert snapshot.reasons == ("twap_stream_gap_detected",)


def test_conflicting_source_value_is_rejected_but_identical_duplicate_is_counted() -> None:
    duplicate_engine = _ready_engine()
    duplicate_engine.ingest(
        _observation(
            2_000,
            value_delta_e18=5 * 10**15,
            sequence=10,
            publisher_ms=START_MS + 2_100,
            received_ms=START_MS + 2_150,
        )
    )
    duplicate = duplicate_engine.build(START_MS + 2_250)

    conflict_engine = _ready_engine()
    conflict_engine.ingest(
        _observation(
            2_000,
            value_delta_e18=99 * 10**15,
            sequence=10,
            publisher_ms=START_MS + 2_100,
            received_ms=START_MS + 2_150,
        )
    )
    conflict = conflict_engine.build(START_MS + 2_250)

    assert duplicate.available is True
    assert _feature(duplicate, "twap.transport.identical_duplicate_count") == 1.0
    assert conflict.available is False
    assert "conflicting_twap_source_timestamp" in conflict.reasons


def test_future_receipts_and_noncausal_wire_order_are_never_consumed() -> None:
    receipt_engine = _ready_engine()
    receipt_engine.ingest(_observation(2_250, sequence=10))

    with pytest.raises(ValueError, match="future receipts"):
        receipt_engine.build(START_MS + 2_250)
    with pytest.raises(ValueError, match="observation is invalid"):
        _observation(
            3_000,
            sequence=11,
            received_ms=START_MS + 2_200,
        )
    with pytest.raises(ValueError, match="observation is invalid"):
        _observation(
            2_100,
            sequence=12,
            publisher_ms=START_MS + 2_000,
            received_ms=START_MS + 2_200,
        )


def test_nonoverlapping_variance_requires_consecutive_exact_grid_points() -> None:
    complete = Round25TwapFeatureEngine(
        condition_id=CONDITION_ID,
        event_start_ms=START_MS,
    )
    for sequence, offset in enumerate((0, 29_000, 30_000, 59_000, 60_000), start=1):
        complete.ingest(
            _observation(
                offset,
                value_delta_e18=sequence * 10**15,
                sequence=sequence,
            )
        )
    complete_snapshot = complete.build(START_MS + 60_250)

    missing_middle = Round25TwapFeatureEngine(
        condition_id="0x" + "b" * 64,
        event_start_ms=START_MS,
    )
    for sequence, offset in enumerate((0, 28_000, 58_000, 59_000, 60_000), start=1):
        missing_middle.ingest(
            _observation(
                offset,
                value_delta_e18=sequence * 10**15,
                sequence=sequence,
            )
        )
    missing_snapshot = missing_middle.build(START_MS + 60_250)

    assert complete_snapshot.available is True
    assert _feature(
        complete_snapshot, "twap.path.nonoverlapping_scale_available"
    ) == 1.0
    assert _feature(
        complete_snapshot, "twap.path.nonoverlapping_30s_realized_variance_rate"
    ) > 0.0
    assert missing_snapshot.available is True
    assert _feature(
        missing_snapshot, "twap.path.nonoverlapping_scale_available"
    ) == 0.0
    assert _feature(
        missing_snapshot, "twap.path.nonoverlapping_30s_realized_variance_rate"
    ) == 0.0


@pytest.mark.parametrize("event_start", [True, START_MS + 1, 0])
def test_engine_rejects_invalid_condition_boundaries(event_start: object) -> None:
    with pytest.raises(ValueError):
        Round25TwapFeatureEngine(
            condition_id=CONDITION_ID,
            event_start_ms=event_start,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("decision", [True, START_MS + 1, START_MS + 300_000])
def test_engine_rejects_invalid_decisions(decision: object) -> None:
    with pytest.raises(ValueError):
        _ready_engine().build(decision)  # type: ignore[arg-type]
