from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_economic_amendment import (
    POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256,
    POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES,
    POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES,
    bind_round27_economic_amendment,
    load_round27_economic_amendment,
    validate_round27_economic_amendment,
    validate_round27_economic_amendment_binding,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _artifact() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "test-round27-economic-artifact-v1",
        "orders_submitted": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def test_round27_economic_amendment_is_exact_and_reachable() -> None:
    repository = Path(__file__).resolve().parents[1]
    amendment = load_round27_economic_amendment(repository)
    audit = amendment["mathematical_audit"]
    semantics = amendment["evidence_gate_semantics"]

    assert amendment["amendment_sha256"] == POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
    assert audit["maximum_candidates_per_condition"] == 1
    assert audit["original_minimum_selection_executed_trades"] > audit["selection_conditions"]
    assert audit["original_minimum_sealed_executed_trades"] > audit["sealed_conditions"]
    assert POLYMARKET_ROUND27_SELECTION_MINIMUM_EXECUTED_TRADES == 60
    assert POLYMARKET_ROUND27_SEALED_MINIMUM_EXECUTED_TRADES == 60
    assert semantics["minimum_executed_trades_is_a_trade_quota"] is False
    assert semantics["may_force_or_relax_a_risk_gate_to_reach_minimum"] is False


def test_round27_economic_amendment_rejects_tampering() -> None:
    repository = Path(__file__).resolve().parents[1]
    amendment = load_round27_economic_amendment(repository)
    tampered = copy.deepcopy(amendment)
    tampered["replacement"]["minimum_selection_executed_trades"] = 100

    with pytest.raises(ValueError, match="amendment differs"):
        validate_round27_economic_amendment(tampered)


def test_round27_economic_artifact_binding_is_integrity_checked() -> None:
    bound = bind_round27_economic_amendment(
        _artifact(),
        hash_field="report_sha256",
    )

    assert (
        bound["economic_population_amendment_sha256"]
        == POLYMARKET_ROUND27_ECONOMIC_AMENDMENT_SHA256
    )
    assert (
        validate_round27_economic_amendment_binding(
            bound,
            hash_field="report_sha256",
        )
        == bound
    )

    tampered = dict(bound)
    tampered["economic_population_amendment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="amendment binding differs"):
        validate_round27_economic_amendment_binding(
            tampered,
            hash_field="report_sha256",
        )


def test_round27_economic_artifact_binding_rejects_corrupt_source() -> None:
    artifact = _artifact()
    artifact["orders_submitted"] = True

    with pytest.raises(ValueError, match="artifact integrity differs"):
        bind_round27_economic_amendment(
            artifact,
            hash_field="report_sha256",
        )
