from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE_TOOL = importlib.import_module("tools.screen_polymarket_exact_reward_source")
BASE = ROOT / "docs/model-research/polymarket"
SOURCE_CONTRACT = (
    BASE / "btc-above-72k-sep5-exact-reward-source-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = (
    BASE / "btc-above-72k-sep5-exact-reward-source-prefilter-v1-2026-08-30.json"
)
SOURCE_RAW = BASE / "raw/btc-above-72k-sep5-exact-reward-source-v1-2026-08-30"
BOOK_CONTRACT = (
    BASE / "btc-above-72k-sep5-retained-reward-book-contract-v1-2026-08-30.json"
)
BOOK_RESULT = (
    BASE / "btc-above-72k-sep5-retained-reward-book-terminal-v1-2026-08-30.json"
)
BOOK_RAW = BASE / "raw/btc-above-72k-sep5-retained-reward-book-v1-2026-08-30"
ADJUDICATION = (
    BASE / "btc-above-72k-sep5-paired-maker-retained-adjudication-v1-2026-08-30.json"
)
TOKEN_CORRECTION = (
    BASE / "btc-above-72k-sep5-exact-source-token-reconciliation-v1-2026-08-30.json"
)
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


def test_btc_sep5_exact_reward_candidate_is_terminal_without_refetch() -> None:
    _source_contract, source_contract_hash = _reconstruct(SOURCE_CONTRACT)
    source, source_hash = _reconstruct(SOURCE_RESULT)
    _book_contract, book_contract_hash = _reconstruct(BOOK_CONTRACT)
    book, book_hash = _reconstruct(BOOK_RESULT)
    adjudication, adjudication_hash = _reconstruct(ADJUDICATION)
    token_correction, token_correction_hash = _reconstruct(TOKEN_CORRECTION)
    assert (
        source_contract_hash
        == "3fc5681145ac0a1436f3703ff54c62e7551860d83a002949ac4f85e363e49876"
    )
    assert (
        source_hash
        == "9de74d4b2aef878e455aa061b39667cee16f0d21a5f9b728071372b43f5c4902"
    )
    assert (
        book_contract_hash
        == "5c6c293c45557b93d02340f3c711d8345c17fb3cce82345ead89034cd2b442f8"
    )
    assert (
        book_hash == "fa3a92322fced76a3e39d8eb893a7542a117aca7b74cbb62e06c91dc7ba41e1b"
    )
    assert (
        adjudication_hash
        == "1153ef2f90345be8ebfda5b0c2fd3f02a56dc0dad854edf251e21370a9677743"
    )
    assert (
        token_correction_hash
        == "61df67471329b0e4a1273deea0fbbba9d918d3e11559b3dc26f6f462f69691a4"
    )

    assert source["exact_reward"]["minimum_size_shares"] == "50"
    assert source["exact_reward"]["maximum_spread_cents"] == "4.5"
    assert source["exact_reward"]["daily_rate_pUSD"] == "1.99972"
    assert source["candidate"]["maker_fee_zero"] is True
    assert source["verdict"]["books_requested"] is False

    for source_name, filename in (
        ("gamma_request", "01-exact-gamma-market.raw"),
        ("reward_request", "02-exact-sponsored-reward.raw"),
    ):
        raw = SOURCE_RAW / filename
        metadata = source["sources"][source_name]
        assert raw.stat().st_size == metadata["payload_bytes"]
        assert _sha256(raw.read_bytes()) == metadata["payload_sha256"]
    books_raw = BOOK_RAW / "01-two-token-books.raw"
    books_metadata = book["sources"]["books_request"]
    assert books_raw.stat().st_size == books_metadata["payload_bytes"]
    assert _sha256(books_raw.read_bytes()) == books_metadata["payload_sha256"]

    assert book["capture"] == {
        "book_timestamp_skew_ms": 0,
        "freshness_passed": False,
        "oldest_book_event_age_ms": 30871,
        "request_elapsed_ms": 240,
    }
    assert book["economics"]["best_bid_join"] == {
        "both_fill_gross_profit_pUSD": "2.050",
        "combined_bid": "0.959",
        "maximum_orphan_settlement_loss_pUSD": "46.800",
    }
    assert (
        adjudication["economics"][
            "optimistic_full_reward_pool_until_market_horizon_pUSD"
        ]
        == "12.92301410590233518518518519"
    )
    assert adjudication["economics"]["full_pool_orphan_coverage_fraction"].startswith(
        "0.27613278"
    )
    assert adjudication["authority"]["network_requests"] == 0
    assert adjudication["verdict"]["accepted_edge"] is False
    assert adjudication["verdict"]["retry_permitted"] is False
    assert token_correction["authority"]["network_requests"] == 0
    assert token_correction["exact_identity"]["tokens_reconciled"] is True
    assert (
        token_correction["exact_identity"]["gamma_token_ids"]
        == token_correction["exact_identity"]["reward_token_ids"]
    )

    registry, _registry_hash = _reconstruct(REGISTRY)
    assert len(registry["prioritized_hypotheses"]) == 44
    assert len(registry["terminal_do_not_repeat"]) == 55
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_BTC_above_72000_September_5_exact_paired_maker_reward_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == adjudication_hash


def test_exact_reward_source_rejects_cross_source_token_mismatch() -> None:
    contract = json.loads(SOURCE_CONTRACT.read_text(encoding="ascii"))
    gamma_raw = json.loads((SOURCE_RAW / "01-exact-gamma-market.raw").read_bytes())
    reward_raw = json.loads((SOURCE_RAW / "02-exact-sponsored-reward.raw").read_bytes())
    market = SOURCE_TOOL._gamma(gamma_raw, candidate=contract["candidate"])
    reward_raw["data"][0]["tokens"][0]["token_id"] = "0"
    with pytest.raises(
        ValueError,
        match="exact Gamma and sponsored reward token identity disagree",
    ):
        SOURCE_TOOL._reward(
            reward_raw,
            market=market,
            candidate=contract["candidate"],
            now=market["event_end"].replace(year=2026, month=8, day=30),
            terminal_cursor="LTE=",
        )
