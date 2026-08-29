from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/model-research/action-value"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
REGISTRY_HASH = "fc0bddf222a1908db6c12df338dc26963f36514b01e37b5b31fc567760f19aca"

CONTRACT_HASHES = {
    "binance-two-intermediary-conversion-retained-contract-v1.json": (
        "3ebefe96fede2847dfeaa9fda5f72f1cc5429d3e74bb0271c070f6a970099455"
    ),
    "binance-two-intermediary-conversion-retained-contract-v2.json": (
        "ff2583bd787ec3cf259112bae6a9415b0f522a7409c6632f652869db1ce8d2a7"
    ),
    "binance-two-intermediary-conversion-retained-contract-v3.json": (
        "649d5b0a46a5ec70419709faee2cff1ef8d3e4d0816074376e8fcf53f161138f"
    ),
    "binance-two-intermediary-conversion-activity-contract-v1.json": (
        "d6ef2834993afa3e9b26bb04504e397b2f19338fcbad10953ddb0c58f19c0700"
    ),
}
TOOL_HASHES = {
    "tools/adjudicate_binance_two_intermediary_conversions_retained.py": (
        "eccdfa85424aa8afc92996797de3cbb60ecad2f12a34a152b45e3a72cf6eed8f"
    ),
    "tools/adjudicate_binance_two_intermediary_conversions_retained_v2.py": (
        "36a41db06e28bd8b39775e6eaf51e17e2bd58daa21e7f9c09f1e29d06f628f42"
    ),
    "tools/adjudicate_binance_two_intermediary_conversions_retained_v3.py": (
        "4f6a9b162ab628a0df7df36a17b59fb31236c8100c45d6a3ebce37ff985a51be"
    ),
    "tools/adjudicate_binance_two_intermediary_conversion_activity.py": (
        "044bd34afcf88ba5f0717d9d8b2577c06b86c928b8930d03519d7767cfed090e"
    ),
}
RESULT_HASHES = {
    "binance-two-intermediary-conversion-retained-v1-resource-adjudication-2026-08-29.json": (
        "be1bfdb40200f0e0acb26fcd1413a9c39e3ab34d61161550794fc429fe50cddd"
    ),
    "binance-two-intermediary-conversion-retained-v2-source-gate-adjudication-2026-08-29.json": (
        "d23724ee0d088ac1965e96612557039a576f73724a1b630d2e434536ef0b2079"
    ),
    "binance-two-intermediary-conversion-retained-v3-2026-08-29.json": (
        "0a5e37f2fb48c639334256e3118e3eeb2f17a548572faaf13d3849204404b45e"
    ),
    "binance-two-intermediary-conversion-activity-adjudication-v1-2026-08-29.json": (
        "cde72e05b1760d9fe23eb65e5bd5f59377230ac91095354936c2a84a9a3758ae"
    ),
}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _embedded_hash(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_frozen_contracts_and_implementations_are_hash_bound() -> None:
    for name, expected_hash in CONTRACT_HASHES.items():
        contract = _load(EVIDENCE / name)
        assert contract["contract_sha256"] == expected_hash
        assert _embedded_hash(contract, "contract_sha256") == expected_hash

    for relative_path, expected_hash in TOOL_HASHES.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            expected_hash
        )


def test_consumed_v1_and_v2_fail_closed_without_market_authority() -> None:
    for name in list(RESULT_HASHES)[:2]:
        result = _load(EVIDENCE / name)
        expected_hash = RESULT_HASHES[name]
        assert result["result_sha256"] == expected_hash
        assert _embedded_hash(result, "result_sha256") == expected_hash
        assert result["verdict"]["accepted_edge"] is False
        assert result["verdict"]["profitability_claim"] is False
        assert result["authority"] == {
            "network_requests": 0,
            "credentials_used": False,
            "account_state_accessed": False,
            "orders_or_mutations": 0,
            "protected_capture_accessed": False,
        }


def test_exact_extension_population_is_complete_but_not_promoted() -> None:
    name = "binance-two-intermediary-conversion-retained-v3-2026-08-29.json"
    result = _load(EVIDENCE / name)
    assert result["result_sha256"] == RESULT_HASHES[name]
    assert _embedded_hash(result, "result_sha256") == RESULT_HASHES[name]
    assert result["direct_vs_two_intermediary_routes"] == 1_064_216
    assert result["evaluated_route_sizes"] == 2_128_432
    assert result["optimistic_prefilter_survivor_route_sizes"] == 16_051
    assert result["exact_empirical_candidate_count"] == 253
    assert len(result["exact_empirical_candidates"]) == 253
    assert result["accepted_edge"] is False
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["protected_capture_accessed"] is False


def test_activity_gate_retains_and_rejects_every_exact_candidate() -> None:
    name = (
        "binance-two-intermediary-conversion-activity-adjudication-v1-2026-08-29.json"
    )
    result = _load(EVIDENCE / name)
    assert result["result_sha256"] == RESULT_HASHES[name]
    assert _embedded_hash(result, "result_sha256") == RESULT_HASHES[name]
    assert result["population"] == {
        "exact_parent_candidates": 253,
        "unique_required_symbols": 174,
        "activity_survivors": 0,
        "activity_rejections": 253,
    }
    decisions = result["candidate_activity_decisions"]
    assert len(decisions) == 253
    assert len({row["route_id"] + ":" + row["start_usdt"] for row in decisions}) == 253
    assert all(row["activity_gate_passed"] is False for row in decisions)
    assert all(row["minimum_quote_changes"] < 5 for row in decisions)
    assert sum(row["minimum_24h_trade_count"] < 100 for row in decisions) == 18
    assert result["activity_survivors"] == []
    assert result["accepted_edge"] is False
    assert result["deployment_ready"] is False
    assert result["authority"]["network_requests"] == 0
    assert result["authority"]["protected_capture_accessed"] is False


def test_registry_terminalizes_only_the_two_intermediary_extension() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _embedded_hash(registry, "result_sha256") == REGISTRY_HASH
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 44
    )
    assert registry["accepted_edge_count"] == 21
    assert len(registry["prioritized_hypotheses"]) == 44
    assert family["mechanism"] == (
        "binance_indirect_internal_conversion_route_savings_for_organic_flow"
    )
    assert family["canonical_artifacts"][-1]["result_sha256"] == (
        RESULT_HASHES[
            "binance-two-intermediary-conversion-activity-adjudication-v1-2026-08-29.json"
        ]
    )
    terminal = registry["terminal_do_not_repeat"][-1]
    assert terminal["family"] == (
        "binance_spot_exactly_two_intermediary_organic_conversion_extension"
    )
    assert terminal["canonical_result_sha256"] == family["canonical_artifacts"][-1][
        "result_sha256"
    ]
