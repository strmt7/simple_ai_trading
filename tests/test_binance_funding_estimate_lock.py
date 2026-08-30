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
