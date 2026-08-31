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
NYC_CONTRACT = (
    BASE
    / "nyc-september-precipitation-over-6-reward-source-contract-v1-2026-08-31.json"
)
NYC_ADJUDICATION = (
    ROOT
    / "docs/model-research/action-value/polymarket-nyc-september-precipitation-over-6-reward-end-date-mismatch-v1-2026-08-31.json"
)
NYC_RAW = (
    BASE
    / "raw/nyc-september-precipitation-over-6-reward-source-v1-2026-08-31/01-exact-gamma-market.raw"
)
ONTARIO_SOURCE_CONTRACT = (
    BASE / "ontario-liberal-navdeep-bains-reward-source-contract-v1-2026-08-31.json"
)
ONTARIO_SOURCE_RESULT = (
    BASE / "ontario-liberal-navdeep-bains-reward-source-v1-2026-08-31.json"
)
ONTARIO_SOURCE_RAW = (
    BASE / "raw/ontario-liberal-navdeep-bains-reward-source-v1-2026-08-31"
)
ONTARIO_BOOK_CONTRACT = (
    BASE
    / "ontario-liberal-navdeep-bains-retained-reward-book-contract-v1-2026-08-31.json"
)
ONTARIO_BOOK_RESULT = (
    BASE
    / "ontario-liberal-navdeep-bains-retained-reward-book-terminal-v1-2026-08-31.json"
)
ONTARIO_BOOK_RAW = (
    BASE / "raw/ontario-liberal-navdeep-bains-retained-reward-books-v1-2026-08-31"
)
GTA_SOURCE_CONTRACT = (
    BASE
    / "gta-vi-extended-look-under-20m-reward-source-contract-v1-2026-08-31.json"
)
GTA_SOURCE_RESULT = (
    BASE / "gta-vi-extended-look-under-20m-reward-source-v1-2026-08-31.json"
)
GTA_SOURCE_RAW = (
    BASE / "raw/gta-vi-extended-look-under-20m-reward-source-v1-2026-08-31"
)
GTA_BOOK_CONTRACT = (
    BASE
    / "gta-vi-extended-look-under-20m-retained-reward-book-contract-v1-2026-08-31.json"
)
GTA_BOOK_RESULT = (
    BASE
    / "gta-vi-extended-look-under-20m-retained-reward-book-terminal-v1-2026-08-31.json"
)
GTA_BOOK_RAW = (
    BASE
    / "raw/gta-vi-extended-look-under-20m-retained-reward-books-v1-2026-08-31"
)
GTA_ADJUDICATION = (
    ROOT
    / "docs/model-research/action-value/polymarket-gta-vi-under-20m-reward-stale-book-adjudication-v1-2026-08-31.json"
)


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
    assert len(registry["prioritized_hypotheses"]) == 45
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


def test_rendered_resolution_date_mismatch_is_terminal_and_date_gate_is_exact() -> None:
    contract, contract_hash = _reconstruct(NYC_CONTRACT)
    adjudication, adjudication_hash = _reconstruct(NYC_ADJUDICATION)
    gamma_raw = json.loads(NYC_RAW.read_bytes())

    assert (
        contract_hash
        == "101f5cacd9dc7a4acf2282f373e25f8abdaee5207d85f36812b0e74e98ddc877"
    )
    assert (
        adjudication_hash
        == "41d36380efda98913a149156cbb5aacc4425e0d7be8c4372c73be9dbba316d5c"
    )
    with pytest.raises(ValueError, match="Gamma event end date changed"):
        SOURCE_TOOL._gamma(gamma_raw, candidate=contract["candidate"])

    corrected_candidate = dict(contract["candidate"])
    corrected_candidate["event_end_date_utc"] = "2026-09-30"
    market = SOURCE_TOOL._gamma(gamma_raw, candidate=corrected_candidate)
    assert market["event_end"].isoformat() == "2026-09-30T23:59:00+00:00"
    gamma_authoritative_candidate = dict(corrected_candidate)
    gamma_authoritative_candidate.pop("event_end_date_utc")
    gamma_authoritative_market = SOURCE_TOOL._gamma(
        gamma_raw,
        candidate=gamma_authoritative_candidate,
    )
    assert (
        gamma_authoritative_market["event_end"].isoformat()
        == "2026-09-30T23:59:00+00:00"
    )
    assert adjudication["exact_sponsored_reward"]["request_made"] is False

    registry, _registry_hash = _reconstruct(REGISTRY)
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_NYC_September_precipitation_over_6_inches_exact_liquidity_reward_source_date_gate"
    )
    assert terminal["canonical_result_sha256"] == adjudication_hash


def test_ontario_reward_sources_reconcile_but_stale_book_is_terminal() -> None:
    _source_contract, source_contract_hash = _reconstruct(ONTARIO_SOURCE_CONTRACT)
    source, source_hash = _reconstruct(ONTARIO_SOURCE_RESULT)
    _book_contract, book_contract_hash = _reconstruct(ONTARIO_BOOK_CONTRACT)
    book, book_hash = _reconstruct(ONTARIO_BOOK_RESULT)

    assert (
        source_contract_hash
        == "ad5931984f16a3f5f4f900bf7878266bd15bd205641076056529a0cb7ee8b4ab"
    )
    assert (
        source_hash
        == "85cef790f285cd732d0f3ce1ae71077d20d685681c008f0ee843cf2cd49af3e5"
    )
    assert (
        book_contract_hash
        == "207c62797736a1aefbe1284fd2c191557a92aef65c35cbdbd173297f654ac8b3"
    )
    assert (
        book_hash == "03b4cdc0f577028e51f1f6bee6c3e2f0426d501d36d381c6d3d14036b974b294"
    )

    assert source["candidate"]["maker_fee_zero"] is True
    assert source["exact_reward"]["daily_rate_pUSD"] == "40"
    assert source["exact_reward"]["minimum_size_shares"] == "20"
    assert source["exact_reward"]["maximum_spread_cents"] == "5.5"
    for source_name, filename in (
        ("gamma_request", "01-exact-gamma-market.raw"),
        ("reward_request", "02-exact-sponsored-reward.raw"),
    ):
        raw = ONTARIO_SOURCE_RAW / filename
        metadata = source["sources"][source_name]
        assert raw.stat().st_size == metadata["payload_bytes"]
        assert _sha256(raw.read_bytes()) == metadata["payload_sha256"]

    books_raw = ONTARIO_BOOK_RAW / "01-two-token-books.raw"
    books_metadata = book["sources"]["books_request"]
    assert books_raw.stat().st_size == books_metadata["payload_bytes"]
    assert _sha256(books_raw.read_bytes()) == books_metadata["payload_sha256"]
    assert book["capture"] == {
        "book_timestamp_skew_ms": 0,
        "freshness_passed": False,
        "oldest_book_event_age_ms": 174712,
        "request_elapsed_ms": 281,
    }
    assert book["economics"]["best_bid_join"] == {
        "both_fill_gross_profit_pUSD": "1.00",
        "combined_bid": "0.95",
        "maximum_orphan_settlement_loss_pUSD": "11.20",
    }
    assert book["verdict"]["status"] == "rejected_stale_book_snapshot"
    assert book["verdict"]["accepted_edge"] is False
    assert book["authority"]["credentials_used"] is False

    registry, _registry_hash = _reconstruct(REGISTRY)
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Navdeep_Bains_Ontario_leadership_exact_paired_maker_reward_2026_08_31"
    )
    assert terminal["canonical_result_sha256"] == book_hash


def test_gta_reward_candidate_is_terminal_after_stale_book_and_zero_floor() -> None:
    _source_contract, source_contract_hash = _reconstruct(GTA_SOURCE_CONTRACT)
    source, source_hash = _reconstruct(GTA_SOURCE_RESULT)
    _book_contract, book_contract_hash = _reconstruct(GTA_BOOK_CONTRACT)
    book, book_hash = _reconstruct(GTA_BOOK_RESULT)
    adjudication, adjudication_hash = _reconstruct(GTA_ADJUDICATION)

    assert (
        source_contract_hash
        == "5bab95ee364650762d7ec87db04f4ba6b88c91adb44d5608946c59f4308b9755"
    )
    assert (
        source_hash
        == "d0c2fba4bd24c97b4c6745059b7b8beb964a7d7f3b56027ee98f6bf15f2a2c60"
    )
    assert (
        book_contract_hash
        == "e06bfe5b10ae6cc8386b67469d605a27d20e58c4c8a34fb6fd0b059648d91eb2"
    )
    assert (
        book_hash == "dc61d9375638de915b4f16546e9a9fc876815d5a493cb53b117011f7b10ada1d"
    )
    assert (
        adjudication_hash
        == "f25854cc7ec978cea4c5357bc8438093cfb818b0435f34773de78096af8ef051"
    )

    assert source["exact_reward"]["daily_rate_pUSD"] == "536.99616"
    assert source["exact_reward"]["minimum_size_shares"] == "200"
    assert source["exact_reward"]["maximum_spread_cents"] == "4.5"
    assert source["candidate"]["maker_fee_zero"] is True
    for source_name, filename in (
        ("gamma_request", "01-exact-gamma-market.raw"),
        ("reward_request", "02-exact-sponsored-reward.raw"),
    ):
        raw = GTA_SOURCE_RAW / filename
        metadata = source["sources"][source_name]
        assert raw.stat().st_size == metadata["payload_bytes"]
        assert _sha256(raw.read_bytes()) == metadata["payload_sha256"]

    books_raw = GTA_BOOK_RAW / "01-two-token-books.raw"
    books_metadata = book["sources"]["books_request"]
    assert books_raw.stat().st_size == books_metadata["payload_bytes"]
    assert _sha256(books_raw.read_bytes()) == books_metadata["payload_sha256"]
    assert book["capture"] == {
        "book_timestamp_skew_ms": 0,
        "freshness_passed": False,
        "oldest_book_event_age_ms": 17830,
        "request_elapsed_ms": 278,
    }
    assert book["economics"]["best_bid_join"] == {
        "both_fill_gross_profit_pUSD": "2.00",
        "combined_bid": "0.99",
        "maximum_orphan_settlement_loss_pUSD": "136.00",
    }
    assert book["economics"]["one_tick_improved"] == {
        "both_fill_gross_profit_pUSD": "-2.00",
        "combined_bid": "1.01",
        "marketable": True,
        "maximum_orphan_settlement_loss_pUSD": "138.00",
    }
    assert book["verdict"]["status"] == "rejected_stale_book_snapshot"

    assert adjudication["freshness"]["oldest_book_excess_age_ms"] == 7830
    assert (
        adjudication["economics"][
            "optimistic_full_reward_pool_until_market_horizon_pUSD"
        ]
        == "1785.76951491519840"
    )
    assert adjudication["reward_allocation_adjudication"][
        "owned_final_reward_share_floor"
    ] == "0"
    assert adjudication["authority"]["network_requests_made_by_adjudication"] is False
    assert adjudication["authority"]["credentials_used"] is False
    assert adjudication["verdict"]["accepted_edge"] is False
    assert adjudication["verdict"]["retry_permitted"] is False

    registry, _registry_hash = _reconstruct(REGISTRY)
    assert len(registry["prioritized_hypotheses"]) == 45
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_GTA_VI_Extended_Look_under_20_million_views_exact_paired_maker_reward_2026_08_31"
    )
    assert terminal["canonical_result_sha256"] == adjudication_hash
