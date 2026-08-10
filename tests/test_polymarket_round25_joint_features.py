from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from simple_ai_trading.polymarket_round25_clob_features import (
    POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    Round25ClobFeatureSnapshot,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    combine_round25_features,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    Round25TwapFeatureSnapshot,
)


START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 2_250
CONDITION_ID = "0x" + "a" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _twap(*, available: bool = True) -> Round25TwapFeatureSnapshot:
    return Round25TwapFeatureSnapshot(
        condition_id=CONDITION_ID,
        event_start_ms=START_MS,
        decision_time_ms=DECISION_MS,
        available=available,
        reasons=() if available else ("twap_source_stale",),
        values=(1.0,) * len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES)
        if available
        else (0.0,) * len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES),
        source_chain_sha256="a" * 64 if available else EMPTY_SHA256,
        maximum_receipt_ms=DECISION_MS - 100 if available else 0,
        opening_value_e18=65_000 * 10**18 if available else 0,
        latest_value_e18=65_001 * 10**18 if available else 0,
    )


def _clob(*, available: bool = True) -> Round25ClobFeatureSnapshot:
    return Round25ClobFeatureSnapshot(
        condition_id=CONDITION_ID,
        event_start_ms=START_MS,
        decision_time_ms=DECISION_MS,
        available=available,
        reasons=() if available else ("up_book_receipt_stale",),
        market_prior_probability=0.52 if available else 0.5,
        values=(2.0,) * len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES)
        if available
        else (0.0,) * len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES),
        source_chain_sha256="b" * 64 if available else EMPTY_SHA256,
        maximum_receipt_ms=DECISION_MS - 50 if available else 0,
    )


def test_joint_schema_is_exact_ordered_union_without_targets() -> None:
    assert POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION.endswith("v1")
    assert POLYMARKET_ROUND25_JOINT_FEATURE_NAMES == (
        *POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
        *POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    )
    assert len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES) == 148
    assert len(set(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)) == 148
    assert all(
        forbidden not in name
        for name in POLYMARKET_ROUND25_JOINT_FEATURE_NAMES
        for forbidden in ("target", "resolution", "pnl", "structural_probability")
    )


def test_joint_snapshot_is_deterministic_hash_bound_and_non_authoritative() -> None:
    first = combine_round25_features(_twap(), _clob())
    second = combine_round25_features(_twap(), _clob())

    assert first == second
    assert first.available is True
    assert first.reasons == ()
    assert first.market_prior_probability == 0.52
    assert first.values == (
        *(1.0,) * len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES),
        *(2.0,) * len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES),
    )
    assert first.twap_source_chain_sha256 == "a" * 64
    assert first.clob_source_chain_sha256 == "b" * 64
    assert first.source_chain_sha256 not in {
        EMPTY_SHA256,
        first.twap_source_chain_sha256,
        first.clob_source_chain_sha256,
    }
    assert first.maximum_receipt_ms == DECISION_MS - 50
    assert first.trading_authority is False


@pytest.mark.parametrize(
    ("twap_available", "clob_available", "expected_reasons"),
    [
        (False, True, ("twap:twap_source_stale",)),
        (True, False, ("clob:up_book_receipt_stale",)),
        (
            False,
            False,
            ("twap:twap_source_stale", "clob:up_book_receipt_stale"),
        ),
    ],
)
def test_partial_source_availability_never_creates_a_model_row(
    twap_available: bool,
    clob_available: bool,
    expected_reasons: tuple[str, ...],
) -> None:
    snapshot = combine_round25_features(
        _twap(available=twap_available),
        _clob(available=clob_available),
    )

    assert snapshot.available is False
    assert snapshot.reasons == expected_reasons
    assert snapshot.market_prior_probability == 0.5
    assert not any(snapshot.values)
    assert snapshot.source_chain_sha256 == EMPTY_SHA256
    assert snapshot.twap_source_chain_sha256 == EMPTY_SHA256
    assert snapshot.clob_source_chain_sha256 == EMPTY_SHA256
    assert snapshot.maximum_receipt_ms == 0
    assert snapshot.trading_authority is False


@pytest.mark.parametrize(
    "clob",
    [
        replace(_clob(), condition_id="0x" + "b" * 64),
        replace(_clob(), event_start_ms=START_MS + 300_000),
        replace(_clob(), decision_time_ms=DECISION_MS + 250),
    ],
)
def test_mismatched_source_identity_is_rejected(clob: Round25ClobFeatureSnapshot) -> None:
    with pytest.raises(ValueError, match="identity"):
        combine_round25_features(_twap(), clob)


def test_wrong_source_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="source type"):
        combine_round25_features(_twap(), object())  # type: ignore[arg-type]


def test_joint_chain_tampering_is_rejected() -> None:
    snapshot = combine_round25_features(_twap(), _clob())

    with pytest.raises(ValueError, match="available joint"):
        replace(snapshot, source_chain_sha256="f" * 64)
