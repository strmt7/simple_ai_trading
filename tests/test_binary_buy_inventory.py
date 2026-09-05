from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import pytest

from tools.audit_binary_buy_inventory import Scope, canonical_hash, envelope, main


SCOPE = Scope("wallet", 100, 200)


def row(outcome_number=0, size="10", price="0.4", **kwargs):
    return {
        "proxyWallet": "wallet",
        "timestamp": 110 + outcome_number,
        "eventSlug": "btc-updown-5m-100",
        "conditionId": "condition",
        "asset": str(outcome_number),
        "side": "BUY",
        "outcomeIndex": outcome_number,
        "outcome": ("Up", "Down")[outcome_number],
        "size": size,
        "price": price,
    } | kwargs


def test_matched_and_residual_cash_conservation():
    result = envelope([row(size="10"), row(1, size="6", price="0.3")], SCOPE)
    totals = result["totals"]
    assert Decimal(totals["gross_purchase_cash"]) == Decimal("5.8")
    assert totals["pair_quantity"] == "6" and totals["residual_quantity"] == "4"
    assert Decimal(totals["gross_lower_pnl"]) == Decimal("0.2")
    assert Decimal(totals["gross_upper_pnl"]) == Decimal("4.2")
    assert result["conditions"][0]["residual_outcome"] == "Up"


def test_one_sided_and_exact_balanced_states():
    lone = envelope([row(1)], SCOPE)
    assert lone["conditions"][0]["residual_outcome"] == "Down"
    assert Decimal(lone["totals"]["gross_lower_pnl"]) == -4
    assert Decimal(lone["totals"]["gross_upper_pnl"]) == 6
    paired = envelope([row(), row(1, price="0.7")], SCOPE)
    assert paired["conditions"][0]["residual_outcome"] is None
    assert paired["residual_quantity_weighted_fraction_needed"] is None
    assert paired["negative_upper_condition_count"] == 1


def test_no_profitable_subset_selection_or_outcome_assignment():
    losing = row(
        conditionId="second", eventSlug="eth-updown-15m-100", price="0.9", size="100"
    )
    result = envelope([row(), row(1), losing], SCOPE)
    assert result["condition_count"] == 2 and result["scoped_buy_rows"] == 3
    assert Decimal(result["totals"]["gross_lower_pnl"]) == -88
    assert result["positive_lower_condition_count"] == 1
    assert result["unbalanced_condition_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"size": "NaN"},
        {"price": "Infinity"},
        {"size": "0"},
        {"price": "1"},
        {"price": "-1"},
        {"side": "SELL"},
        {"outcomeIndex": True},
        {"outcomeIndex": 2},
        {"outcome": "Down"},
        {"proxyWallet": "foreign"},
        {"timestamp": 200},
        {"timestamp": True},
        {"conditionId": ""},
        {"asset": ""},
    ],
)
def test_invalid_rows_reject(change):
    with pytest.raises(ValueError):
        envelope([row(**change)], SCOPE)


def test_duplicate_and_identity_conflicts_reject():
    with pytest.raises(ValueError, match="ambiguous"):
        envelope([row(), deepcopy(row())], SCOPE)
    for other in [
        row(1, eventSlug="sol-updown-5m-100"),
        row(1, asset="0"),
        row(size="11", asset="different"),
    ]:
        with pytest.raises(ValueError, match="identity"):
            envelope([row(), other], SCOPE)


def test_scope_exclusion_is_reported_not_silently_dropped():
    result = envelope([row(), row(eventSlug="unscoped")], SCOPE)
    assert result["raw_rows"] == 2 and result["scoped_buy_rows"] == 1
    assert result["excluded_out_of_scope_rows"] == 1
    with pytest.raises(ValueError, match="no scoped"):
        envelope([row(eventSlug="unscoped")], SCOPE)
    with pytest.raises(ValueError, match="nonempty"):
        envelope([], SCOPE)
    with pytest.raises(ValueError, match="ordered"):
        envelope([row()], Scope("wallet", 200, 100))


@pytest.mark.parametrize("failure", [None, "contract", "binding", "rows", "conditions"])
def test_one_use_cli_and_preoutput_failures(tmp_path, monkeypatch, capsys, failure):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps([row()]), encoding="utf-8")
    output = tmp_path / "result.json"
    plan = {
        "raw_path": str(raw),
        "output_path": str(output),
        "scope": {"wallet": "wallet", "start": 100, "end": 200},
        "expected_raw_rows": 2 if failure == "rows" else 1,
        "expected_conditions": 2 if failure == "conditions" else 1,
        "interpretation": {"not_wallet_pnl": True},
        "bindings": [
            {
                "path": str(raw),
                "sha256": "bad"
                if failure == "binding"
                else hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
        ],
    }
    plan["contract_sha256"] = (
        "bad" if failure == "contract" else canonical_hash(plan, "contract_sha256")
    )
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit", "--contract", str(contract)])
    if failure:
        with pytest.raises(ValueError):
            main()
        assert not output.exists()
    else:
        main()
        result = json.loads(output.read_bytes())
        assert json.loads(capsys.readouterr().out)["raw_rows"] == 1
        assert result["actual_wallet_pnl"] is False and result["accepted_edge"] is False
        assert result["result_sha256"] == canonical_hash(result, "result_sha256")
        with pytest.raises(FileExistsError):
            main()


def test_retained_result_independently_reconstructs_with_rational_arithmetic():
    root = Path(__file__).resolve().parents[1]
    folder = root / "docs/review/2026-09-05/wallet-buy-envelope"
    plan = json.loads((folder / "contract.json").read_bytes())
    result = json.loads((folder / "result.json").read_bytes())
    assert plan["contract_sha256"] == canonical_hash(plan, "contract_sha256")
    assert result["result_sha256"] == canonical_hash(result, "result_sha256")
    assert result["contract_sha256"] == plan["contract_sha256"]
    for binding in plan["bindings"]:
        assert (
            hashlib.sha256((root / binding["path"]).read_bytes()).hexdigest()
            == binding["sha256"]
        )
    raw = json.loads((root / plan["raw_path"]).read_bytes())
    groups = {}
    excluded = 0
    for trade in raw:
        parts = trade["eventSlug"].split("-")
        if not (
            len(parts) == 4
            and parts[0] in ("btc", "eth", "sol")
            and parts[1] == "updown"
            and parts[2] in ("5m", "15m", "4h")
            and parts[3].isdigit()
        ):
            excluded += 1
            continue
        assert trade["side"] == "BUY"
        value = groups.setdefault(
            trade["conditionId"], [Fraction(0), Fraction(0), Fraction(0), 0]
        )
        quantity, price = Fraction(str(trade["size"])), Fraction(str(trade["price"]))
        value[trade["outcomeIndex"]] += quantity
        value[2] += quantity * price
        value[3] += 1
    assert len(raw) == result["raw_rows"] == 1964
    assert result["scoped_buy_rows"] == len(raw) - excluded == 1222
    assert len(groups) == result["condition_count"] == 358
    assert result["excluded_out_of_scope_rows"] == excluded == 742
    for condition in result["conditions"]:
        up, down, cash, count = groups[condition["condition_id"]]
        assert count == condition["buy_rows"]
        assert Fraction(condition["quantity_up"]) == up
        assert Fraction(condition["quantity_down"]) == down
        assert Fraction(condition["gross_purchase_cash"]) == cash
        assert Fraction(condition["pair_quantity"]) == min(up, down)
        assert Fraction(condition["residual_quantity"]) == abs(up - down)
        assert Fraction(condition["gross_lower_pnl"]) == min(up, down) - cash
        assert Fraction(condition["gross_upper_pnl"]) == max(up, down) - cash
    for key, total in result["totals"].items():
        assert sum(Fraction(c[key]) for c in result["conditions"]) == Fraction(total)
    assert result["network_requests"] == 0 and result["profitability_claim"] is False
