from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.screen_polymarket_exact_negrisk_books import _fee_model


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
PREFILTER_CONTRACT = ACTION / (
    "polymarket-btc-sep4-price-range-exact-negrisk-prefilter-contract-v1-2026-08-30.json"
)
PREFILTER_RESULT = ACTION / (
    "polymarket-btc-sep4-price-range-exact-negrisk-prefilter-result-v1-2026-08-30.json"
)
PREFILTER_RAW = ROOT / (
    "data/polymarket-btc-sep4-price-range-exact-negrisk-prefilter-v1/raw/event.json"
)
PREFILTER_JOURNAL = ROOT / (
    "data/polymarket-btc-sep4-price-range-exact-negrisk-prefilter-v1/request-journal.jsonl"
)
BOOK_CONTRACT = ACTION / (
    "polymarket-btc-sep4-price-range-exact-negrisk-books-contract-v1-2026-08-30.json"
)
BOOK_RAW = ROOT / (
    "data/polymarket-btc-sep4-price-range-exact-negrisk-books-v1/raw/books.json"
)
BOOK_JOURNAL = ROOT / (
    "data/polymarket-btc-sep4-price-range-exact-negrisk-books-v1/request-journal.jsonl"
)
ADJUDICATION_CONTRACT = ACTION / (
    "polymarket-btc-sep4-price-range-exact-negrisk-books-retained-adjudication-"
    "contract-v1-2026-08-30.json"
)
ADJUDICATION_RESULT = ACTION / (
    "polymarket-btc-sep4-price-range-exact-negrisk-books-retained-adjudication-"
    "result-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"

PREFILTER_CONTRACT_HASH = (
    "4f931ca8c7c94d68c6fc36ea77868bf6e285f4f33d062aed491d140cc048b5c8"
)
PREFILTER_RESULT_HASH = (
    "a8fa32e0ec3cc51c35670df5c2dc1c5b0fa1dc02ba281958c56ded4953922cbf"
)
PREFILTER_RAW_HASH = "335c6297ccf49d966fb1cbebe244b2feec77edbea67e83b127ade3eb6a99f1e9"
BOOK_CONTRACT_HASH = "96fabc2319bb25cf9ee000edd8c9dbe72f3d20ce82ee891ffe4c97993e6917a6"
BOOK_RAW_HASH = "b003bee6a0e6300c5068617685748ced28f6c039e0be561b40147f13dfeea735"
ADJUDICATION_CONTRACT_HASH = (
    "197b4b8ebe2c4fb1f8c440aa3e08b9aa22aa60424471a677e581688f6031c0ed"
)
ADJUDICATION_RESULT_HASH = (
    "426b53310b6f46ea39312b4d06f404453ceac2b98863efeee91aa9d592120208"
)
CRYPTO_FEE_SCHEDULE = {
    "exponent": 1,
    "rate": 0.07,
    "rebateRate": 0.2,
    "takerOnly": True,
}


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_exact_market_fee_schedule_is_contract_bound() -> None:
    market = {"feesEnabled": True, "feeSchedule": CRYPTO_FEE_SCHEDULE}
    fee = _fee_model(market, CRYPTO_FEE_SCHEDULE)
    assert fee.rate == Decimal("0.07")
    assert fee.exponent == 1
    with pytest.raises(RuntimeError, match="exact current fee schedule differs"):
        _fee_model(market)


def test_btc_sep4_prefilter_and_consumed_book_batch_are_source_bound() -> None:
    prefilter_contract = _load(PREFILTER_CONTRACT)
    prefilter = _load(PREFILTER_RESULT)
    book_contract = _load(BOOK_CONTRACT)
    assert _canonical_hash(prefilter_contract, "contract_sha256") == (
        PREFILTER_CONTRACT_HASH
    )
    assert _canonical_hash(prefilter, "result_sha256") == PREFILTER_RESULT_HASH
    assert hashlib.sha256(PREFILTER_RAW.read_bytes()).hexdigest() == PREFILTER_RAW_HASH
    assert _canonical_hash(book_contract, "contract_sha256") == BOOK_CONTRACT_HASH
    assert hashlib.sha256(BOOK_RAW.read_bytes()).hexdigest() == BOOK_RAW_HASH
    assert [
        json.loads(line)["phase"]
        for line in PREFILTER_JOURNAL.read_bytes().splitlines()
    ] == ["intent", "completed"]
    book_journal = [json.loads(line) for line in BOOK_JOURNAL.read_bytes().splitlines()]
    assert [row["phase"] for row in book_journal] == ["intent", "completed"]
    assert book_journal[-1]["response_sha256"] == BOOK_RAW_HASH
    assert prefilter["screen"]["event"]["market_count"] == 11
    assert Decimal(prefilter["screen"]["event"]["displayed_all_yes_sum_pUSD"]) == (
        Decimal("3.0075")
    )
    assert prefilter["screen"]["positive_displayed_conversion_candidate_count"] == 11


def test_retained_depth_terminalizes_the_exact_btc_event_without_refetch() -> None:
    contract = _load(ADJUDICATION_CONTRACT)
    result = _load(ADJUDICATION_RESULT)
    assert _canonical_hash(contract, "contract_sha256") == ADJUDICATION_CONTRACT_HASH
    assert _canonical_hash(result, "result_sha256") == ADJUDICATION_RESULT_HASH
    assert result["retained_capture"]["freshness_passed"] is False
    assert result["retained_capture"]["book_count"] == 22
    assert Decimal(
        result["screen"]["zero_fee_no_stress"]["best_path"]["net_quote"]
    ) == (Decimal("-3.730"))
    assert Decimal(
        result["screen"]["gamma_fee_no_stress"]["best_path"]["net_quote"]
    ) == (Decimal("-3.81628"))
    assert Decimal(
        result["screen"]["gamma_fee_one_adverse_tick_each_leg"]["best_path"][
            "net_quote"
        ]
    ) == Decimal("-4.07770")
    assert result["screen"]["candidate_after_all_frozen_gates"] is False
    assert result["adjudication"]["network_requests"] == 0
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == _canonical_hash(registry, "result_sha256")
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["priority_rank"] == 31
    )
    hashes = {item["result_sha256"] for item in row["canonical_artifacts"]}
    assert {
        PREFILTER_CONTRACT_HASH,
        PREFILTER_RESULT_HASH,
        BOOK_CONTRACT_HASH,
        ADJUDICATION_CONTRACT_HASH,
        ADJUDICATION_RESULT_HASH,
    }.issubset(hashes)
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_Bitcoin_September_4_price_range_exact_fixed_NegRisk_depth_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == ADJUDICATION_RESULT_HASH
