from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from simple_ai_trading.polymarket_round17_one_use import (
    POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256,
    Round17OneUseClaimStore,
    Round17TestAccessClaim,
    load_round17_one_use_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-017-btc-5m-one-use-evaluation-contract-v1.json"
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _claim(*, suffix: str = "a", opened_at_ms: int = 1_800_000_000_000):
    hashes = [f"{index:x}" * 64 for index in range(1, 10)]
    provisional = Round17TestAccessClaim(
        development_result_sha256=hashes[0],
        campaign_readiness_sha256=hashes[1],
        campaign_development_index_sha256=hashes[2],
        cohort_manifest_sha256=hashes[3],
        target_manifest_sha256=hashes[4],
        model_pretest_sha256=hashes[5],
        probability_calibration_sha256=hashes[6],
        economic_pretest_sha256=hashes[7],
        implementation_manifest_sha256=(suffix * 64),
        repository_commit_sha="d" * 40,
        opened_at_ms=opened_at_ms,
    )
    return replace(
        provisional,
        claim_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def _result(claim: Round17TestAccessClaim, access: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "test-result-v1",
        "claim_sha256": claim.claim_sha256,
        "test_access_sha256": access,
        "test_access_consumed": True,
        "automatic_promotion": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    payload["result_sha256"] = _sha256(payload)
    return payload


def test_round17_one_use_contract_is_hash_bound(tmp_path: Path) -> None:
    loaded = load_round17_one_use_contract(CONTRACT)
    assert loaded["contract_sha256"] == POLYMARKET_ROUND17_ONE_USE_CONTRACT_SHA256

    tampered = json.loads(CONTRACT.read_text(encoding="utf-8"))
    tampered["test_partition"]["minimum_resolved_conditions"] = 1  # type: ignore[index]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        load_round17_one_use_contract(path)


def test_round17_one_use_store_is_singleton_idempotent_and_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "one-use.sqlite3"
    claim = _claim()
    with Round17OneUseClaimStore(path) as store:
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        opened = store.open_claim(claim)
        assert opened == claim
        assert store.open_claim(_claim(opened_at_ms=claim.opened_at_ms + 1)) == claim
        with pytest.raises(RuntimeError, match="different evidence"):
            store.open_claim(_claim(suffix="b"))

        access = store.consume_test_access(claim, observed_at_ms=claim.opened_at_ms + 2)
        assert store.consume_test_access(claim) == access
        store.mark_resolution_pending(
            claim,
            pending_condition_count=3,
            observed_at_ms=claim.opened_at_ms + 3,
        )
        completed = store.complete(
            claim,
            _result(claim, access),
            observed_at_ms=claim.opened_at_ms + 4,
        )
        assert store.complete(claim, completed) == completed
        snapshot = store.snapshot()
        assert snapshot["status"] == "completed"
        assert snapshot["test_access_consumed"] is True
        assert snapshot["event_count"] == 4
        with pytest.raises(RuntimeError, match="terminal"):
            store.consume_test_access(claim)

    assert not Path(f"{path}-wal").exists()
    with Round17OneUseClaimStore(path) as reopened:
        assert reopened.snapshot()["result"] == completed


def test_round17_one_use_failure_cannot_return_to_development(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.sqlite3"
    claim = _claim()
    with Round17OneUseClaimStore(path) as store:
        store.open_claim(claim)
        store.fail(claim, reason="deterministic evaluator failure")
        snapshot = store.snapshot()
        assert snapshot["status"] == "failed"
        assert snapshot["failure"]["return_to_development"] is False  # type: ignore[index]
        with pytest.raises(RuntimeError, match="already failed"):
            store.complete(claim, _result(claim, "a" * 64))


def test_round17_one_use_event_chain_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "tampered.sqlite3"
    claim = _claim()
    with Round17OneUseClaimStore(path) as store:
        store.open_claim(claim)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE round17_one_use_event SET previous_event_sha256 = 'tampered'"
        )
    with Round17OneUseClaimStore(path) as store:
        with pytest.raises(ValueError, match="event chain differs"):
            store.snapshot()
