from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.adjudicate_binance_funding_estimate_lock import (  # noqa: E402
    _canonical_hash,
)


ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / "binance-funding-estimate-lock-contract-v1-2026-08-30.json"
RESULT = ACTION / "binance-funding-estimate-lock-result-v1-2026-08-30.json"
PREFILTER = (
    ACTION
    / "binance-btc-eth-sol-near-finality-funding-capture-prefilter-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_contract_and_retained_sources_reconstruct() -> None:
    contract = _load(CONTRACT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert len(contract["requests"]) == 3
    assert [row["symbol"] for row in contract["requests"]] == [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    ]
    assert contract["authority"]["credentials_used"] is False
    assert contract["authority"]["orders_or_transactions"] == 0
    implementation = ROOT / contract["implementation"]["path"]
    assert (
        hashlib.sha256(implementation.read_bytes()).hexdigest()
        == contract["implementation"]["sha256"]
    )
    for source in contract["retained_sources"]:
        assert (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            == source["sha256"]
        )


def test_six_of_nine_estimates_changed_without_promoting_carry() -> None:
    result = _load(RESULT)

    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["evaluation"]["observation_count"] == 9
    assert result["evaluation"]["changed_count"] == 6
    assert result["evaluation"]["exact_match_count"] == 3
    assert max(
        abs(Decimal(row["error_bips"])) for row in result["evaluation"]["rows"]
    ) == Decimal("0.62660000")
    decision = result["adjudication"]
    assert decision["status"] == "terminal_estimate_not_locked_at_observed_lead_times"
    assert decision["known_at_entry_at_observed_lead_times_proved"] is False
    assert decision["accepted_edge"] is False
    assert decision["profitability_claim"] is False
    assert decision["book_or_fee_request_permitted"] is False
    for receipt in result["capture"]["receipts"]:
        raw = ROOT / receipt["raw_path"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == receipt["response_sha256"]


def test_registry_terminalizes_exact_family_without_rerouting_ranked_work() -> None:
    result = _load(RESULT)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_BTC_ETH_SOL_displayed_funding_estimate_known_at_entry_lock_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert "do_not_request_books" in terminal["reason"]


def test_retained_funding_prefilter_rejects_unnecessary_live_capture() -> None:
    prefilter = _load(PREFILTER)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _canonical_hash(prefilter, "result_sha256") == prefilter["result_sha256"]
    reconstructed = []
    for source in prefilter["sources"]["retained_funding_history"]:
        raw = ROOT / source["path"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == source["sha256"]
        rows = json.loads(raw.read_bytes())
        absolute_bips = [abs(Decimal(row["fundingRate"]) * 10_000) for row in rows]
        reconstructed.append(
            (
                source["symbol"],
                len(rows),
                max(absolute_bips),
                sum(value > Decimal("4") for value in absolute_bips),
                sum(value > Decimal("32") for value in absolute_bips),
            )
        )

    assert reconstructed == [
        ("BTCUSDT", 500, Decimal("1.22760000"), 0, 0),
        ("ETHUSDT", 500, Decimal("2.29760000"), 0, 0),
        ("SOLUSDT", 500, Decimal("3.98100000"), 0, 0),
    ]
    assert prefilter["evaluation"]["combined_row_count"] == 1500
    assert prefilter["adjudication"]["near_finality_capture_permitted_now"] is False
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_BTC_ETH_SOL_near_finality_single_funding_capture_retained_gross_prefilter_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == prefilter["result_sha256"]
