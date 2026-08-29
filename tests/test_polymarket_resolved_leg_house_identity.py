from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "6062ef4cb774983d86d7edd5dad7adcaafa31a8202d37ec777e12fc33028d157"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(encoded)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_contracts_and_results_reconstruct() -> None:
    contract_names = (
        "polymarket-shutdown-house-identity-metadata-contract-v1-2026-08-29.json",
        "polymarket-shutdown-house-identity-parity-contract-v1-2026-08-29.json",
        "polymarket-aca-house-identity-metadata-contract-v1-2026-08-29.json",
        "polymarket-aca-house-identity-parity-contract-v1-2026-08-29.json",
    )
    result_names = (
        "polymarket-shutdown-house-identity-parity-failure-adjudication-v1-2026-08-29.json",
        "polymarket-shutdown-house-maker-first-candidate-v1-2026-08-29.json",
        "polymarket-aca-house-identity-parity-result-v1-2026-08-29.json",
        "polymarket-aca-house-maker-first-candidate-v1-2026-08-29.json",
    )
    for name in contract_names:
        payload = _load(ACTION / name)
        assert _canonical_hash(payload, "contract_sha256") == payload["contract_sha256"]
        implementation = ROOT / payload["implementation"]["path"]
        assert _sha256(implementation.read_bytes()) == payload["implementation"][
            "sha256"
        ]
    for name in result_names:
        payload = _load(ACTION / name)
        assert _canonical_hash(payload, "result_sha256") == payload["result_sha256"]


def test_public_request_journals_bind_every_retained_response() -> None:
    roots = (
        ROOT / "data/polymarket-shutdown-house-identity-metadata-v1",
        ROOT / "data/polymarket-shutdown-house-identity-parity-v1",
        ROOT / "data/polymarket-aca-house-identity-metadata-v1",
        ROOT / "data/polymarket-aca-house-identity-parity-v1",
    )
    for data_root in roots:
        rows = [json.loads(line) for line in (data_root / "request-journal.jsonl").read_text().splitlines()]
        intents = [row for row in rows if row["phase"] == "intent"]
        completed = [row for row in rows if row["phase"] == "completed"]
        assert len(intents) == len(completed)
        assert len(intents) in {1, 3}
        for row in completed:
            raw_path = ROOT / row["raw_path"]
            raw = raw_path.read_bytes()
            assert row["status_code"] == 200
            assert len(raw) == row["response_bytes"]
            assert _sha256(raw) == row["response_sha256"]


def test_independent_instance_repeats_identity_not_positive_economics() -> None:
    all_taker = _load(
        ACTION / "polymarket-aca-house-identity-parity-result-v1-2026-08-29.json"
    )
    maker = _load(
        ACTION / "polymarket-aca-house-maker-first-candidate-v1-2026-08-29.json"
    )
    assert all_taker["payoff_proof"] == {
        "aca_final_no": True,
        "duplicate_payoff_identity_count": 2,
        "house_states": ["Democratic", "Republican", "Other"],
    }
    assert all_taker["capture"]["book_timestamp_skew_ms"] == 11695
    assert all_taker["capture"]["fee_receipts"] == {}
    assert all(
        not row["passes_economic_gate"] and not row["passes_frozen_candidate_gate"]
        for row in all_taker["rows"]
    )
    adjudication = maker["adjudication"]
    assert adjudication["independent_recurrence_of_payoff_identity"] is True
    assert (
        adjudication["independent_recurrence_of_positive_maker_first_economics"]
        is False
    )
    assert adjudication["positive_two_tick_hedge_sensitivity_count"] == 0
    assert adjudication["accepted_edge"] is False


def test_registry_terminalizes_resolved_leg_family_without_promotion() -> None:
    registry = _load(REGISTRY)
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_cross_market_exact_multi_outcome_subset_equivalence"
    )
    assert row["priority_rank"] == 31
    assert "remains_unaccepted" in row["current_status"]
    assert registry["accepted_edge_count"] == 19
