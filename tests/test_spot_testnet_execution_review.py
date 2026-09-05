"""Retained, zero-network evidence reconstruction and rejection cases."""

from copy import deepcopy
import json

import pytest

from simple_ai_trading.spot_testnet_coverage import lifecycle_coverage
from tools.review_spot_testnet_execution import BASE, ROOT, reconstruct, review
from tools.spot_testnet_campaign_transport import Journal, SYMBOLS


def retained():
    return Journal(ROOT / BASE / "journal.jsonl").rows, json.loads(
        (ROOT / BASE / "result.json").read_bytes()
    )


def test_retained_ledger_reconstructs_without_promoting_profit():
    result = review()
    assert result["original_reporting_gate_passed"] is False
    assert result["corrected_required_live_cases_passed"] is True
    assert result["request_counts"] == {"GET": 48, "POST": 9, "DELETE": 3}
    assert result["owned_trade_count"] == 6
    assert result["quote_cash_delta_virtual_USDT"] == "-0.01019850"
    assert result["partial_fills_observed"] == 0
    assert result["cold_process_restart_tested"] is False
    assert result["profitability_evidence"] is False
    assert result["network_requests_for_this_review"] == 0


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_intent",
        "foreign_identity",
        "changed_identity",
        "unowned_order",
        "order_side",
        "trade_id",
        "cash",
        "terminal",
        "result",
    ],
)
def test_reconstruction_rejects_inconsistent_evidence(damage):
    rows, result = retained()
    if damage == "duplicate_intent":
        row = next(r for r in rows if r["kind"] == "order_intent")
        rows.insert(rows.index(row) + 1, deepcopy(row))
    elif damage in {"foreign_identity", "changed_identity"}:
        identities = [r for r in rows if r["kind"] == "order_identity"]
        if damage == "foreign_identity":
            identities[0]["original_client_id"] = "foreign"
        else:
            identities[1]["orderId"] = 999
    elif damage in {"unowned_order", "order_side"}:
        row = next(r for r in rows if r["kind"] == "order_observation")["order"]
        row["orderId" if damage == "unowned_order" else "side"] = (
            999 if damage == "unowned_order" else "SELL"
        )
    elif damage == "trade_id":
        next(r for r in rows if r["kind"] == "owned_trades")["orderId"] = 999
    elif damage == "cash":
        next(r for r in rows if r["kind"] == "order_cash")["quote_delta"] = "99"
    elif damage == "terminal":
        rows.pop()
    else:
        result["quote_cash_delta"] = "99"
    with pytest.raises(ValueError):
        reconstruct(rows, result)


@pytest.mark.parametrize(
    "damage", ["foreign", "ambiguous", "overlap", "receipt", "unresolved"]
)
def test_coverage_rejects_inconsistent_request_or_identity(damage):
    rows, result = retained()
    if damage in {"foreign", "ambiguous"}:
        identities = [r for r in rows if r["kind"] == "order_identity"]
        if damage == "foreign":
            identities[0]["original_client_id"] = "foreign"
        else:
            identities[1]["orderId"] = 999
    else:
        row = next(r for r in rows if r["kind"] == "http_intent")
        if damage == "overlap":
            rows.insert(rows.index(row) + 1, deepcopy(row))
        elif damage == "receipt":
            next(r for r in rows if r["kind"] == "http_completed")["method"] = "POST"
        else:
            rows.append(deepcopy(row))
    with pytest.raises(ValueError):
        lifecycle_coverage(rows, result["order_cash"], SYMBOLS)
