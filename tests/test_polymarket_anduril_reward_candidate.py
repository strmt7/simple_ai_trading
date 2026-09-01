from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/anduril-1225b-reward-source-contract-v1-2026-09-01.json"
)
SOURCE_RESULT = (
    ROOT
    / "docs/model-research/polymarket/anduril-1225b-reward-source-v1-2026-09-01.json"
)
BOOK_CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/anduril-1225b-retained-reward-book-contract-v1-2026-09-01.json"
)
BOOK_RESULT = (
    ROOT
    / "docs/model-research/polymarket/anduril-1225b-retained-reward-book-result-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
SOURCE_JOURNAL = ROOT / "data/polymarket-anduril-1225b-reward-source-v1"
BOOK_JOURNAL = ROOT / "data/polymarket-anduril-1225b-retained-reward-books-v1"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_anduril_reward_candidate_is_source_bound_and_fail_closed() -> None:
    source_contract = load(SOURCE_CONTRACT)
    source = load(SOURCE_RESULT)
    book_contract = load(BOOK_CONTRACT)
    book = load(BOOK_RESULT)
    registry = load(REGISTRY)

    for artifact in (source_contract, source, book_contract, book, registry):
        assert self_hash(artifact) == artifact["result_sha256"]

    assert source["sources"]["contract_file_sha256"] == file_hash(SOURCE_CONTRACT)
    assert book["sources"]["contract_file_sha256"] == file_hash(BOOK_CONTRACT)
    assert (
        source["sources"]["contract_result_sha256"] == source_contract["result_sha256"]
    )
    assert book["sources"]["contract_result_sha256"] == book_contract["result_sha256"]
    assert (
        book_contract["retained_exact_sources"]["source_prefilter"]["result_sha256"]
        == source["result_sha256"]
    )

    for name, source_key in (
        ("01-exact-gamma-market.raw", "gamma_request"),
        ("02-exact-sponsored-reward.raw", "reward_request"),
    ):
        assert (
            file_hash(SOURCE_JOURNAL / name)
            == source["sources"][source_key]["payload_sha256"]
        )
    assert (
        file_hash(BOOK_JOURNAL / "01-two-token-books.raw")
        == book["sources"]["books_request"]["payload_sha256"]
    )

    assert source["exact_reward"]["daily_rate_pUSD"] == "50"
    assert source["exact_reward"]["minimum_size_shares"] == "20"
    assert source["exact_reward"]["maximum_spread_cents"] == "4.5"
    assert book["capture"] == {
        "book_timestamp_skew_ms": 0,
        "freshness_passed": False,
        "oldest_book_event_age_ms": 79200,
        "request_elapsed_ms": 233,
    }
    assert book["economics"]["best_bid_join"] == {
        "both_fill_gross_profit_pUSD": "0.60",
        "combined_bid": "0.97",
        "maximum_orphan_settlement_loss_pUSD": "10.0",
    }
    assert book["economics"]["publicly_proven_reward_payout_floor_pUSD"] == "0"
    assert book["verdict"] == {
        "accepted_edge": False,
        "profitability_claim": False,
        "retry_permitted": False,
        "status": "rejected_stale_book_snapshot",
        "trading_authority": False,
    }
    assert book["authority"]["credentials_used"] is False
    assert book["authority"]["orders_or_cancellations"] == 0

    rank17 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 17
    )
    published = {
        row["path"]: row["result_sha256"] for row in rank17["canonical_artifacts"]
    }
    for path, artifact in (
        (SOURCE_CONTRACT, source_contract),
        (SOURCE_RESULT, source),
        (BOOK_CONTRACT, book_contract),
        (BOOK_RESULT, book),
    ):
        assert published[path.relative_to(ROOT).as_posix()] == artifact["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Anduril_122_5B_September_paired_maker_reward_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == book["result_sha256"]
