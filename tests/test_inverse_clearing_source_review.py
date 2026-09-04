"""Offline checks for the retained source conflict, not Binance PnL validation."""

from decimal import Decimal as D
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04"


def test_retained_source_and_one_request_journal():
    review = json.loads((BASE / "inverse-clearing-source-review.json").read_bytes())
    source = review["source"]
    raw = (ROOT / source["path"]).read_bytes()
    assert raw.startswith(b"%PDF-")
    assert len(raw) == source["bytes"]
    assert hashlib.sha256(raw).hexdigest() == source["sha256"]
    journal = [
        json.loads(line)
        for line in (BASE / "inverse-clearing-source-journal.jsonl")
        .read_bytes()
        .splitlines()
    ]
    assert [row["phase"] for row in journal] == ["request_started", "request_completed"]
    assert journal[0]["url"] == source["url"]
    assert journal[1]["raw_sha256"] == source["sha256"]
    assert journal[1]["transport_passed"] is True
    assert journal[1]["bytes"] <= journal[0]["max_bytes"]
    runner = BASE / "capture-inverse-clearing-source.ps1"
    assert (
        hashlib.sha256(runner.read_bytes()).hexdigest() == journal[0]["script_sha256"]
    )


def test_opposite_source_signs_cannot_qualify_the_same_payoff():
    review = json.loads((BASE / "inverse-clearing-source-review.json").read_bytes())
    example = review["sign_counterexample"]
    notional = D(example["normalized_positive_notional_quote"])
    entry, mark = D(example["entry_price"]), D(example["mark_price"])
    side = D(example["short_direction"])
    pnl80 = side * notional * (1 / entry - 1 / mark)
    pnl86 = side * notional * (1 / mark - 1 / entry)
    assert pnl80 == D(example["section80_pnl_coin"]) == -pnl86
    assert pnl86 == D(example["section86_2_pnl_coin"])
    q = D(example["matched_initial_collateral_coin"])
    assert (q + pnl80) * mark == D(example["common_price_wealth_section80_quote"])
    assert (q + pnl86) * mark == D(example["common_price_wealth_section86_2_quote"])
    assert review["decision"]["unambiguous_payoff_source_gate_passed"] is False
    assert review["decision"]["new_market_capture_authorized"] is False
    assert review["decision"]["accepted_edge"] is False


def test_retained_all_asset_contract_units_not_current_availability():
    review = json.loads((BASE / "inverse-clearing-source-review.json").read_bytes())
    inventory = review["retained_inventory"]
    raw = (ROOT / inventory["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == inventory["sha256"]
    symbols = [
        row
        for row in json.loads(raw)["symbols"]
        if row["baseAsset"] in {"BTC", "ETH", "SOL"}
    ]
    assert len(symbols) == 9
    for symbol in symbols:
        assert symbol["quoteAsset"] == inventory["quote_asset"]
        assert symbol["marginAsset"] == symbol["baseAsset"]
        assert D(str(symbol["contractSize"])) == D(
            inventory["contract_size_quote"][symbol["baseAsset"]]
        )
    assert inventory["current_instrument_availability_verified"] is False
