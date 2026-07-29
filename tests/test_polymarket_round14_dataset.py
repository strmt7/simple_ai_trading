from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import DecodedPublicEvent
from simple_ai_trading import polymarket_round14_dataset as dataset_module
from simple_ai_trading.polymarket_round14_dataset import (
    load_round14_admission_spec,
    validate_round14_admission_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-admission-spec-v1.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _event(payload: dict[str, object]) -> DecodedPublicEvent:
    return DecodedPublicEvent(
        event_id="event",
        run_id="run",
        message_id="message",
        sub_index=0,
        stream="binance_futures",
        event_type="trade",
        symbol="BTC",
        condition_id="",
        asset_id="",
        source_time_ms=1_785_340_000_000,
        publisher_time_ms=1_785_340_000_001,
        event_sha256="a" * 64,
        event={
            "stream": "btcusdt@trade",
            "data": payload,
        },
        connection_id="connection",
        sequence_number=1,
        received_wall_ms=1_785_340_000_010,
        received_monotonic_ns=10,
    )


def test_round14_admission_spec_is_hash_bound_and_target_free() -> None:
    spec = load_round14_admission_spec(SPEC)

    assert spec.spec_sha256 == (
        "17dfe1081ddb661c151c36dd0b9d53d1b749ec8d69e6665d7e2fe63f3527153f"
    )
    assert spec.decision_cadence_ms == 250
    assert spec.minimum_row_coverage_fraction == 0.90
    future = spec.payload["future_data_policy"]
    assert isinstance(future, dict)
    assert future["feature_replay_include_resolutions"] is False
    assert future["feature_rows_may_read_labels"] is False


def test_round14_admission_spec_rejects_rehashed_semantic_drift() -> None:
    payload = json.loads(SPEC.read_text(encoding="utf-8"))
    payload["condition_admission"]["minimum_row_coverage_fraction"] = "0.10"
    unsigned = dict(payload)
    unsigned.pop("spec_sha256")
    payload["spec_sha256"] = _canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="drifted"):
        validate_round14_admission_spec(payload)


def test_round14_exact_zero_futures_sentinel_is_counted_not_priced() -> None:
    sentinel = _event(
        {
            "e": "trade",
            "E": 1_785_340_000_001,
            "T": 1_785_340_000_000,
            "s": "BTCUSDT",
            "t": 1,
            "p": "0",
            "q": "0",
            "X": "NA",
            "m": True,
            "st": 1,
        }
    )
    malformed = _event(
        {
            "e": "trade",
            "E": 1_785_340_000_001,
            "T": 1_785_340_000_000,
            "s": "BTCUSDT",
            "t": 2,
            "p": "0",
            "q": "1",
            "X": "MARKET",
            "m": True,
            "st": 1,
        }
    )

    assert dataset_module._parse_trade(sentinel) is None
    with pytest.raises(ValueError, match="positive"):
        dataset_module._parse_trade(malformed)


def test_round14_dataset_module_has_no_resolution_table_access() -> None:
    source = Path(dataset_module.__file__).read_text(encoding="utf-8")

    assert "polymarket_resolution_evidence" not in source
    assert "winning_outcome" not in source
    assert "include_resolutions=False" in source
