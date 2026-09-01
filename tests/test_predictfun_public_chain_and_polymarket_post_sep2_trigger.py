import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value/predictfun-public-chain-and-polymarket-post-sep2-trigger-adjudication-v1-2026-09-01.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_public_chain_and_wnba_adjudication_is_hash_bound_and_fail_closed() -> None:
    artifact = _load(ARTIFACT)
    assert _self_hash(artifact) == artifact["result_sha256"]
    assert artifact["adjudication"] == {
        **artifact["adjudication"],
        "accepted_edge": False,
        "accepted_edge_count_delta": 0,
        "deployment_ready": False,
        "profitability_claim": False,
        "public_forward_profit_floor_usdt": 0,
    }
    assert artifact["polymarket_post_sep2_wnba"]["returned_event_count"] == 0
    assert artifact["polymarket_post_sep2_wnba"][
        "population_complete_under_frozen_filter"
    ] is True
    assert len(artifact["predictfun_public_chain_path"]["missing_public_lineage"]) == 5
    assert artifact["predictfun_collateral_yield"]["decision"].startswith("terminal")


def test_registry_binds_all_three_terminal_families_without_new_edge() -> None:
    artifact = _load(ARTIFACT)
    registry = _load(REGISTRY)
    assert _self_hash(registry) == registry["result_sha256"]
    assert registry["accepted_edge_count"] == 31
    assert len(registry["prioritized_hypotheses"]) == 48
    assert len(registry["terminal_do_not_repeat"]) == 151
    bound = {
        row["family"]
        for row in registry["terminal_do_not_repeat"]
        if row["canonical_result_sha256"] == artifact["result_sha256"]
    }
    assert bound == {
        "polymarket_WNBA_2026_09_03_through_2026_09_09_deployment_catalog",
        "predictfun_public_on_chain_fill_and_rebate_lineage_2026_09_01",
        "predictfun_trader_owned_Venus_collateral_yield_2026_09_01",
    }
