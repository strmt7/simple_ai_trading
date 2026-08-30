from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/polymarket"
CONTRACT = BASE / "elon-posts-40-64-paired-maker-reward-contract-v1-2026-08-30.json"
TERMINAL = BASE / "elon-posts-40-64-paired-maker-reward-terminal-v1-2026-08-30.json"
RAW = BASE / "raw/elon-posts-40-64-paired-maker-reward-screen-v1-2026-08-30"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reconstruct(path: Path) -> tuple[dict[str, object], str]:
    artifact = json.loads(path.read_text(encoding="ascii"))
    claimed = artifact.pop("result_sha256")
    assert _sha256(_canonical(artifact)) == claimed
    return artifact, claimed


def test_exact_reward_mismatch_is_terminal_before_books() -> None:
    _contract, contract_hash = _reconstruct(CONTRACT)
    artifact, terminal_hash = _reconstruct(TERMINAL)
    assert contract_hash == "84bdca31e7f39ac18f38a4c980c3d43015637e1a817b62232a2746fca19bbe94"
    assert terminal_hash == "3fc224b70c035090c4f015d68b52edb6abd9f7222f1b932962274c446d613f47"

    sources = artifact["sources"]
    for name, filename in (
        ("gamma", "01-exact-gamma-market.raw"),
        ("reward", "02-exact-sponsored-reward.raw"),
    ):
        source = sources[name]
        raw = RAW / filename
        assert raw.stat().st_size == source["payload_bytes"]
        assert _sha256(raw.read_bytes()) == source["payload_sha256"]
    assert not list(RAW.glob("03-*"))
    assert artifact["exact_source_values"]["reward_endpoint"] == {
        "market_competitiveness": "11.565889",
        "next_cursor": "LTE=",
        "reward_maximum_spread_cents": "5.5",
        "reward_minimum_size_shares": "50",
        "row_count": 1,
        "total_active_daily_rate_pUSD": "53",
    }
    assert artifact["verdict"]["accepted_edge"] is False
    assert artifact["verdict"]["books_requested"] is False
    assert artifact["verdict"]["retry_permitted"] is False

    registry, registry_hash = _reconstruct(REGISTRY)
    assert registry_hash == "5d524c5958dd8790f345c9056dc3053a1ac819197b44ac14f019c50de1037990"
    assert registry["accepted_edge_count"] == 21
    terminal_row = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Elon_40_to_64_exact_paired_maker_reward_configuration_2026_08_30"
    )
    assert terminal_row["canonical_result_sha256"] == terminal_hash
