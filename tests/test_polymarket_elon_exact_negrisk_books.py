from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.screen_polymarket_exact_negrisk_books import _tokens
from tools.screen_polymarket_exact_negrisk_event import _expected_market_count


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
PREFILTER_CONTRACT = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-prefilter-contract-v1-2026-08-30.json"
)
PREFILTER_RESULT = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-prefilter-result-v1-2026-08-30.json"
)
PREFILTER_RAW = ROOT / (
    "data/polymarket-elon-aug31-sep2-exact-negrisk-prefilter-v1/raw/event.json"
)
PREFILTER_JOURNAL = ROOT / (
    "data/polymarket-elon-aug31-sep2-exact-negrisk-prefilter-v1/request-journal.jsonl"
)
BOOK_V1_CONTRACT = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-books-contract-v1-2026-08-30.json"
)
BOOK_V1_FAILURE = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-books-v1-preflight-failure-2026-08-30.json"
)
BOOK_V1_RUNNER_LINEAGE = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-books-v1-runner-lineage-"
    "reconstruction-v1-2026-08-30.json"
)
BOOK_V2_CONTRACT = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-books-contract-v2-2026-08-30.json"
)
BOOK_V2_RESULT = ACTION / (
    "polymarket-elon-aug31-sep2-exact-negrisk-books-result-v2-2026-08-30.json"
)
BOOK_RAW = ROOT / "data/polymarket-elon-aug31-sep2-exact-negrisk-books-v2/raw/books.json"
BOOK_JOURNAL = ROOT / (
    "data/polymarket-elon-aug31-sep2-exact-negrisk-books-v2/request-journal.jsonl"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
NYC_PREFILTER_RAW = ROOT / (
    "data/polymarket-nyc-mayor-sep1-sep8-exact-negrisk-prefilter-v1/raw/event.json"
)
NYC_PREFILTER_CONTRACT = ACTION / (
    "polymarket-nyc-mayor-sep1-sep8-exact-negrisk-prefilter-contract-v1-2026-08-30.json"
)
NYC_PREFILTER_RESULT = ACTION / (
    "polymarket-nyc-mayor-sep1-sep8-exact-negrisk-prefilter-result-v1-2026-08-30.json"
)
NYC_PREFILTER_JOURNAL = ROOT / (
    "data/polymarket-nyc-mayor-sep1-sep8-exact-negrisk-prefilter-v1/request-journal.jsonl"
)
NYC_BOOK_CONTRACT = ACTION / (
    "polymarket-nyc-mayor-sep1-sep8-exact-negrisk-books-contract-v1-2026-08-30.json"
)
NYC_BOOK_RESULT = ACTION / (
    "polymarket-nyc-mayor-sep1-sep8-exact-negrisk-books-result-v1-2026-08-30.json"
)
NYC_BOOK_RAW = ROOT / (
    "data/polymarket-nyc-mayor-sep1-sep8-exact-negrisk-books-v1/raw/books.json"
)
NYC_BOOK_JOURNAL = ROOT / (
    "data/polymarket-nyc-mayor-sep1-sep8-exact-negrisk-books-v1/request-journal.jsonl"
)

PREFILTER_CONTRACT_HASH = "e11ea740aac94adf121d298f335724ca7799b5cb43902dc52eeb9c1c697d4c0c"
PREFILTER_RESULT_HASH = "63fb913d9f56034879ccee6bc43d531d4a5e805550db99ec6331e31051c680aa"
PREFILTER_RAW_HASH = "9d50b387c15a56ac82df8b519237a8c3db711764754ea2415816f14c0e820835"
BOOK_V1_CONTRACT_HASH = "32658d289f7cf7a5026ab8debc9dd0f178ec82c96a5821bdb42e0f7eeac6e260"
BOOK_V1_FAILURE_HASH = "fb562b32287caee9842e0dac48aad3bd16f8cf6e2c45d78dca11ce2dde0f9078"
BOOK_V1_RUNNER_LINEAGE_HASH = (
    "dbfa67537e141344d5d0b15c62944eec3ef72da2c7cb945c61303085b4b40bc5"
)
BOOK_V2_CONTRACT_HASH = "5668b83501a39be8e23429ac406a6fe043b7d60e17cc6b4246b0731f0d5bcf5d"
BOOK_V2_RESULT_HASH = "4601f3980f14ccb4130fbdc36862def5abdd47f46e9f48c7da25113c72fe33a2"
BOOK_RAW_HASH = "a79c4395499986d94cde9b46014f5c10d464dd6f6b6c171bb266475fd3f7a661"
BOOK_V1_LINEAGE_SOURCE_COMMIT = "0d91f82533937ee986164bf98e66d16bf809e45f"
NYC_PREFILTER_CONTRACT_HASH = (
    "4dfcb72442a0833d4e48d1c9fda492e7d3b2536960e27225bf285768dfa3846e"
)
NYC_PREFILTER_RESULT_HASH = (
    "c4d9b8ce130881d65a7711dc4cc9e48d9d56085d9e07bd15736d44a63ab13bf1"
)
NYC_PREFILTER_RAW_HASH = "99022e9d48c00122513a3e78667ff5e08db88366b4efe329d46372d06d1b40d5"
NYC_BOOK_CONTRACT_HASH = "98415cb0803b4d9c3a05b12c3a528a78831dbc535b39f1957d8edf3e88cddfc1"
NYC_BOOK_RESULT_HASH = "2dcaa72b8a9643b3f6652691f7395ac5405cd6f15ee88e57ce45af1f69b0dc6b"
NYC_BOOK_RAW_HASH = "1e64a6eecb58481ff85eed9cf1770822f2843faa8bfc4ac5c93d00c474736b3b"


def test_exact_event_prefilter_accepts_a_contract_bound_outcome_count() -> None:
    assert _expected_market_count({"expected_market_count": 11}) == 11
    for invalid in (True, 1, 101, "11", None):
        with pytest.raises(RuntimeError, match="integer from 2 through 100"):
            _expected_market_count({"expected_market_count": invalid})


def test_exact_book_runner_derives_the_complete_token_population() -> None:
    event = _load(NYC_PREFILTER_RAW)
    assert len(event["markets"]) == 11
    assert len(_tokens(event)) == 22


def test_nyc_mayor_source_lead_is_rejected_by_complete_exact_books() -> None:
    prefilter_contract = _load(NYC_PREFILTER_CONTRACT)
    prefilter = _load(NYC_PREFILTER_RESULT)
    book_contract = _load(NYC_BOOK_CONTRACT)
    book = _load(NYC_BOOK_RESULT)
    assert _canonical_hash(prefilter_contract, "contract_sha256") == (
        NYC_PREFILTER_CONTRACT_HASH
    )
    assert _canonical_hash(prefilter, "result_sha256") == NYC_PREFILTER_RESULT_HASH
    assert hashlib.sha256(NYC_PREFILTER_RAW.read_bytes()).hexdigest() == (
        NYC_PREFILTER_RAW_HASH
    )
    prefilter_journal = [
        json.loads(line) for line in NYC_PREFILTER_JOURNAL.read_bytes().splitlines()
    ]
    assert [row["phase"] for row in prefilter_journal] == ["intent", "completed"]
    assert prefilter_journal[-1]["response_sha256"] == NYC_PREFILTER_RAW_HASH
    assert Decimal(prefilter["screen"]["event"]["displayed_all_yes_sum_pUSD"]) == (
        Decimal("0.9855")
    )
    assert prefilter["screen"]["all_yes_candidate"] is True

    assert _canonical_hash(book_contract, "contract_sha256") == NYC_BOOK_CONTRACT_HASH
    assert _canonical_hash(book, "result_sha256") == NYC_BOOK_RESULT_HASH
    assert hashlib.sha256(NYC_BOOK_RAW.read_bytes()).hexdigest() == NYC_BOOK_RAW_HASH
    book_journal = [
        json.loads(line) for line in NYC_BOOK_JOURNAL.read_bytes().splitlines()
    ]
    assert [row["phase"] for row in book_journal] == ["intent", "completed"]
    assert book_journal[-1]["response_sha256"] == NYC_BOOK_RAW_HASH
    assert book["capture"]["freshness_passed"] is False
    assert book["capture"]["book_count"] == 22
    assert Decimal(book["screen"]["zero_fee_no_stress"]["best_path"]["net_quote"]) == (
        Decimal("-0.45")
    )
    assert Decimal(book["screen"]["gamma_fee_no_stress"]["best_path"]["net_quote"]) == (
        Decimal("-0.54928")
    )
    assert book["screen"]["candidate_after_all_frozen_gates"] is False
    assert book["authority"]["onchain_requests"] == 0
    assert book["adjudication"]["accepted_edge"] is False


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


def test_exact_event_prefilter_is_source_bound_and_never_promotes() -> None:
    contract = _load(PREFILTER_CONTRACT)
    result = _load(PREFILTER_RESULT)
    assert contract["contract_sha256"] == PREFILTER_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == PREFILTER_CONTRACT_HASH
    assert result["result_sha256"] == PREFILTER_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == PREFILTER_RESULT_HASH
    assert hashlib.sha256(PREFILTER_RAW.read_bytes()).hexdigest() == PREFILTER_RAW_HASH
    journal = [json.loads(line) for line in PREFILTER_JOURNAL.read_bytes().splitlines()]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == PREFILTER_RAW_HASH

    event = result["screen"]["event"]
    assert event["market_count"] == 10
    assert Decimal(event["displayed_all_yes_sum_pUSD"]) == Decimal("1.0130")
    assert result["screen"]["all_yes_candidate"] is False
    assert result["screen"]["positive_displayed_conversion_candidate_count"] == 10
    assert result["adjudication"]["accepted_edge"] is False


def test_unconsumed_v1_failure_and_corrected_v2_are_exact() -> None:
    v1 = _load(BOOK_V1_CONTRACT)
    failure = _load(BOOK_V1_FAILURE)
    lineage = _load(BOOK_V1_RUNNER_LINEAGE)
    v2 = _load(BOOK_V2_CONTRACT)
    assert v1["contract_sha256"] == BOOK_V1_CONTRACT_HASH
    assert _canonical_hash(v1, "contract_sha256") == BOOK_V1_CONTRACT_HASH
    assert failure["result_sha256"] == BOOK_V1_FAILURE_HASH
    assert _canonical_hash(failure, "result_sha256") == BOOK_V1_FAILURE_HASH
    assert failure["request_consumed"] is False
    assert failure["book_requests"] == 0
    assert lineage["result_sha256"] == BOOK_V1_RUNNER_LINEAGE_HASH
    assert (
        _canonical_hash(lineage, "result_sha256") == BOOK_V1_RUNNER_LINEAGE_HASH
    )
    frozen_runner_bytes = subprocess.check_output(
        [
            "git",
            "show",
            f"{BOOK_V1_LINEAGE_SOURCE_COMMIT}:{lineage['current_runner']['path']}",
        ],
        cwd=ROOT,
    )
    assert hashlib.sha256(frozen_runner_bytes).hexdigest() == lineage["current_runner"][
        "sha256"
    ]
    frozen_runner = frozen_runner_bytes.decode("utf-8")
    old = lineage["mechanical_reconstruction"]["replace_exact"]
    new = lineage["mechanical_reconstruction"]["with_exact"]
    assert frozen_runner.count(old) == 1
    reconstructed = frozen_runner.replace(old, new, 1).encode("utf-8")
    assert hashlib.sha256(reconstructed).hexdigest() == lineage[
        "mechanical_reconstruction"
    ]["reconstructed_frozen_v1_runner_sha256"]
    assert v2["contract_sha256"] == BOOK_V2_CONTRACT_HASH
    assert _canonical_hash(v2, "contract_sha256") == BOOK_V2_CONTRACT_HASH
    assert v2["prior_unconsumed_preflight_failure"]["request_consumed"] is False


def test_fresh_exact_books_reject_every_five_share_path() -> None:
    result = _load(BOOK_V2_RESULT)
    assert result["result_sha256"] == BOOK_V2_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == BOOK_V2_RESULT_HASH
    assert hashlib.sha256(BOOK_RAW.read_bytes()).hexdigest() == BOOK_RAW_HASH
    journal = [json.loads(line) for line in BOOK_JOURNAL.read_bytes().splitlines()]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == BOOK_RAW_HASH

    capture = result["capture"]
    assert capture["freshness_passed"] is True
    assert capture["request_elapsed_ms"] == 121
    assert capture["oldest_book_event_age_ms"] == 297
    assert capture["book_timestamp_skew_ms"] == 129
    assert capture["book_count"] == 20
    screen = result["screen"]
    assert screen["candidate_after_all_frozen_gates"] is False
    assert Decimal(screen["zero_fee_no_stress"]["best_path"]["net_quote"]) == Decimal(
        "-0.075"
    )
    assert Decimal(screen["gamma_fee_no_stress"]["best_path"]["net_quote"]) == Decimal(
        "-0.23977"
    )
    assert Decimal(
        screen["gamma_fee_one_adverse_tick_each_leg"]["best_path"]["net_quote"]
    ) == Decimal("-0.61792")
    assert all(
        view["profitable_path_count"] == 0
        for key, view in screen.items()
        if key != "candidate_after_all_frozen_gates"
    )
    assert result["authority"]["onchain_requests"] == 0
    assert result["authority"]["account_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False


def test_registry_terminalizes_only_the_exact_event() -> None:
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
        BOOK_V1_CONTRACT_HASH,
        BOOK_V1_FAILURE_HASH,
        BOOK_V1_RUNNER_LINEAGE_HASH,
        BOOK_V2_CONTRACT_HASH,
        BOOK_V2_RESULT_HASH,
        NYC_PREFILTER_CONTRACT_HASH,
        NYC_PREFILTER_RESULT_HASH,
        NYC_BOOK_CONTRACT_HASH,
        NYC_BOOK_RESULT_HASH,
    }.issubset(hashes)
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_Elon_August_31_to_September_2_exact_fixed_NegRisk_parity_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == BOOK_V2_RESULT_HASH
    nyc_terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "polymarket_NYC_Mayor_September_1_to_September_8_exact_fixed_NegRisk_parity_2026_08_30"
    )
    assert nyc_terminal["canonical_result_sha256"] == NYC_BOOK_RESULT_HASH
    assert registry["accepted_edge_count"] == 21
    assert "full pre-network" in (ROOT / "AGENTS.md").read_text(encoding="utf-8")
