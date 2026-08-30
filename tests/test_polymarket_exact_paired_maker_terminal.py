from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/polymarket"
CONTRACT = BASE / "elon-posts-40-64-paired-maker-reward-contract-v1-2026-08-30.json"
TERMINAL = BASE / "elon-posts-40-64-paired-maker-reward-terminal-v1-2026-08-30.json"
RAW = BASE / "raw/elon-posts-40-64-paired-maker-reward-screen-v1-2026-08-30"
BOOK_CONTRACT = BASE / "elon-posts-40-64-retained-source-book-contract-v1-2026-08-30.json"
BOOK_TERMINAL = BASE / "elon-posts-40-64-retained-source-book-terminal-v1-2026-08-30.json"
BOOK_RAW = BASE / "raw/elon-posts-40-64-retained-source-book-screen-v1-2026-08-30"
BEST_BID_CONTRACT = BASE / "elon-posts-40-64-best-bid-join-contract-v1-2026-08-30.json"
BEST_BID = BASE / "elon-posts-40-64-best-bid-join-v1-2026-08-30.json"
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


def test_discovery_gate_failure_is_preserved_and_corrected() -> None:
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

    _book_contract, book_contract_hash = _reconstruct(BOOK_CONTRACT)
    book_artifact, book_terminal_hash = _reconstruct(BOOK_TERMINAL)
    assert book_contract_hash == "186e4fc73ae2e84954cbd922a3509add6cfbac8062c5a894e6334bf74aeacf49"
    assert book_terminal_hash == "30daf7346fe284953d1bb2fc3c9bbb25e6910f0e3fb44b5a467e711f17e13a50"
    book_source = book_artifact["sources"]["books"]
    book_raw = BOOK_RAW / "01-two-token-books.raw"
    assert book_raw.stat().st_size == book_source["payload_bytes"]
    assert _sha256(book_raw.read_bytes()) == book_source["payload_sha256"]
    assert book_artifact["source_correction"]["repeated_gamma_or_reward_requests"] == 0
    assert book_artifact["capture"] == {
        "book_timestamp_skew_ms": 0,
        "freshness_passed": False,
        "oldest_book_event_age_ms": 6408,
        "request_elapsed_ms": 225,
    }
    assert book_artifact["offline_rejection"]["combined_one_tick_improved_bid"] == "1.01"
    assert book_artifact["offline_rejection"]["both_fill_gross_profit_pUSD"] == "-0.50"
    assert book_artifact["verdict"]["accepted_edge"] is False
    assert book_artifact["verdict"]["retry_permitted"] is False

    _best_bid_contract, best_bid_contract_hash = _reconstruct(BEST_BID_CONTRACT)
    best_bid, best_bid_hash = _reconstruct(BEST_BID)
    assert best_bid_contract_hash == "e75d8216a78b48610df0bf0f0799b5fba3e93262b614e32600d1e1e3ac39d5a7"
    assert best_bid_hash == "facecfaa3b92d905c700083c7b8afe153adc495403ceabc91e417bdb248d059b"
    assert best_bid["authority"]["network_requests"] == 0
    assert best_bid["economics"]["both_fill_gross_profit_pUSD"] == "0.50"
    assert best_bid["economics"]["maximum_orphan_settlement_loss_pUSD"] == "26.00"
    assert best_bid["conditional_score"]["stress"]["1"]["orphan_payback_days"].startswith("43.554156")
    assert best_bid["conditional_score"]["stress"]["100"]["orphan_payback_days"].startswith("4306.849569")
    assert best_bid["verdict"]["fresh_capture_justified"] is False

    registry, registry_hash = _reconstruct(REGISTRY)
    assert registry_hash == "66b03aa1311ee3b0565e9dc4e973f157aa9626af23b5f675bc60119d668dd311"
    assert registry["accepted_edge_count"] == 21
    terminal_row = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Elon_40_to_64_exact_paired_maker_reward_configuration_2026_08_30"
    )
    assert terminal_row["canonical_result_sha256"] == best_bid_hash
