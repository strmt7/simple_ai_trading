"""Real durable shared admission; constructed receipts, never native venue calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

from simple_ai_trading.binance_execution_scope import (
    BinanceExecutionScope,
    BINANCE_SPOT_TESTNET,
)
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_spot_roundtrip_intents import (
    SpotRoundtripJournal,
    SpotRoundtripRequest,
    _validate_record,
)
from simple_ai_trading.binance_close_intents import prepare_close
from simple_ai_trading.positions import OpenPosition, PositionsStore
from simple_ai_trading.position_transactions import position_transaction
from tools.cocoindex_agent_search import run_captured_process


SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-journal-key"
)
IDENTITY = "a" * 29


def request(mode="buy-sell", **changes):
    values = dict(
        operation_id=IDENTITY,
        symbol="BTCUSDC",
        mode=mode,
        first_amount="100" if mode == "buy-sell" else "1",
        initial_base_allocation="0" if mode == "buy-sell" else "1",
        initial_quote_allocation="101" if mode == "buy-sell" else "1000",
    )
    return SpotRoundtripRequest(**{**values, **changes})


def journal(tmp_path):
    return SpotRoundtripJournal(
        BinanceOpenIntentJournal(tmp_path / "binance_open_intents.sqlite3")
    )


def receipt(
    req,
    *,
    first=True,
    amount=None,
    quantity="1",
    price="100",
    fee="0",
    fee_asset="USDC",
    status="FILLED",
    order_id=42,
):
    binding = req.binding(first=first, amount=amount)
    side = "BUY" if binding.client_order_id.startswith("sait-o-") else "SELL"
    with localcontext() as context:
        context.prec = 128
        cash = format(Decimal(quantity) * Decimal(price), "f")
    result = {
        "symbol": req.symbol,
        "side": side,
        "type": "MARKET",
        "status": status,
        "orderId": order_id,
        "clientOrderId": binding.client_order_id,
        "origQty": quantity if side == "BUY" else binding.quantity,
        "executedQty": quantity,
        "cummulativeQuoteQty": cash,
        "fills": []
        if Decimal(quantity) == 0
        else [
            {
                "qty": quantity,
                "price": price,
                "commission": fee,
                "commissionAsset": fee_asset,
                "tradeId": order_id,
            }
        ],
    }
    if side == "BUY":
        result["origQuoteOrderQty"] = binding.quote_quantity
    return result


def record(j, req, response, *, first=True):
    revision, _ = j.pending(scope=SCOPE)
    j.record_order(
        req.operation_id, response, first=first, scope=SCOPE, expected_revision=revision
    )


def second(j, req, amount):
    revision, _ = j.pending(scope=SCOPE)
    j.prepare_second(req.operation_id, amount, scope=SCOPE, expected_revision=revision)


@pytest.mark.parametrize("mode", ["buy-sell", "sell-buy"])
def test_both_exact_orderings_survive_reopen_and_release_only_after_cash_reconciliation(
    tmp_path, mode
):
    req = request(mode)
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    assert j.journal.entry_block_reason() == "unresolved_spot_roundtrip=1"
    j = journal(tmp_path)
    first = receipt(
        req,
        fee="0.001" if mode == "buy-sell" else "0.1",
        fee_asset="BTC" if mode == "buy-sell" else "USDC",
    )
    record(j, req, first)
    principal = "0.999" if mode == "buy-sell" else "99.9"
    second(j, req, principal)
    response = receipt(
        req,
        first=False,
        amount=principal,
        quantity="0.999" if mode == "buy-sell" else "0.4995",
        price="100" if mode == "buy-sell" else "200",
        order_id=43,
    )
    record(j, req, response, first=False)
    assert j.journal.entry_block_reason() == "unresolved_spot_roundtrip=1"
    revision, state = j.pending(scope=SCOPE)
    assert state["phase"] == "SECOND_TERMINAL"
    j.complete(req.operation_id, scope=SCOPE, expected_revision=revision)
    j = journal(tmp_path)
    assert j.pending(scope=SCOPE) is None
    assert j.journal.entry_block_reason() is None
    assert j.operation(req.operation_id, scope=SCOPE)["phase"] == "COMPLETE"
    with pytest.raises(OpenIntentError, match="consumed"):
        j.prepare(req, scope=SCOPE)
    with sqlite3.connect(j.journal.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert db.execute(
            "SELECT revision, completed_count, active_operation_id FROM spot_roundtrip_control"
        ).fetchone() == (5, 1, None)


@pytest.mark.parametrize("mode,overspend", [("buy-sell", "1"), ("sell-buy", "100")])
def test_second_principal_cannot_inherit_existing_wallet_assets_or_ignore_entry_fees(
    tmp_path, mode, overspend
):
    req = request(mode, initial_base_allocation="1000", initial_quote_allocation="1000")
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    record(
        j,
        req,
        receipt(
            req,
            fee="0.001" if mode == "buy-sell" else "0.1",
            fee_asset="BTC" if mode == "buy-sell" else "USDC",
        ),
    )
    before = j.pending(scope=SCOPE)
    with pytest.raises(OpenIntentError, match="first-leg owned receipts"):
        second(j, req, overspend)
    assert j.pending(scope=SCOPE) == before


def test_missing_namespace_reads_are_noncreating(tmp_path):
    j = journal(tmp_path)
    assert j.pending(scope=SCOPE) is None
    assert j.operation(IDENTITY, scope=SCOPE) is None
    assert not j.journal.path.exists()


def test_recovery_mutation_cannot_initialize_a_missing_namespace(tmp_path):
    j = journal(tmp_path)
    with pytest.raises(ValueError, match="storage is unavailable"):
        j.prepare_second(IDENTITY, "1", scope=SCOPE, expected_revision=1)
    assert not j.journal.path.exists()


def test_database_loss_at_connect_cannot_race_into_fresh_initialization(
    tmp_path, monkeypatch
):
    j = journal(tmp_path)
    j.prepare(request(), scope=SCOPE)
    connect = BinanceOpenIntentJournal._connect

    def remove_before_connect(instance, **kwargs):
        assert kwargs == {"create": False, "write": True}
        instance.path.unlink()
        return connect(instance, **kwargs)

    monkeypatch.setattr(BinanceOpenIntentJournal, "_connect", remove_before_connect)
    with pytest.raises(ValueError, match="storage is unavailable"):
        j.prepare_second(IDENTITY, "1", scope=SCOPE, expected_revision=1)
    assert not j.journal.path.exists()


def position(**changes):
    return OpenPosition(
        **{
            "id": "recovered",
            "symbol": "BTCUSDC",
            "market_type": "spot",
            "side": "LONG",
            "qty": 1,
            "entry_price": 100,
            "leverage": 1,
            "opened_at_ms": 1,
            "notional": 100,
            "dry_run": False,
            "open_client_order_id": "sait-o-recovered",
            **changes,
        }
    )


def test_regular_opening_cannot_bypass_pending_roundtrip(tmp_path):
    j = journal(tmp_path)
    j.prepare(request(), scope=SCOPE)
    with pytest.raises(OpenIntentError, match="roundtrip blocks new exposure"):
        j.journal.prepare(position(), scope=SCOPE)
    with sqlite3.connect(j.journal.path) as db:
        assert db.execute("SELECT COUNT(*) FROM open_intent").fetchone()[0] == 0
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4


def test_separately_recovered_owned_close_is_not_vetoed_by_roundtrip(tmp_path):
    j = journal(tmp_path)
    j.prepare(request(), scope=SCOPE)
    # This is an out-of-band recovered projection, not a new opening dispatch.
    owned = position(exchange_status="FILLED", open_exchange_order_id="100")
    store = PositionsStore(tmp_path)
    store.record_open(owned)
    prepare_close(
        store, owned, client_id="sait-c-recovered", scope=SCOPE, reduce_only=False
    )
    assert j.journal.entry_block_reason() == "unresolved_closing_intents=1"
    assert j.pending(scope=SCOPE)[1]["phase"] == "FIRST_UNKNOWN"


def test_roundtrip_never_allocates_existing_owned_position_assets(tmp_path):
    store = PositionsStore(tmp_path)
    requested = position()
    store.opening_intents.prepare(requested, scope=SCOPE)
    owned = replace(requested, exchange_status="FILLED", open_exchange_order_id="100")
    store.record_open(owned)
    store.opening_intents.record_complete(requested, owned, scope=SCOPE)
    with pytest.raises(OpenIntentError, match="owned position assets"):
        journal(tmp_path).prepare(request(), scope=SCOPE)
    assert store.load_open(strict=True) == [owned]


def test_completed_first_enrollment_failure_rolls_back_scope_and_new_tables(
    tmp_path, monkeypatch
):
    from simple_ai_trading import binance_spot_roundtrip_intents as module

    def fail(*args, **kwargs):
        raise OSError("injected before first reservation")

    monkeypatch.setattr(module, "_save", fail)
    with pytest.raises(ValueError, match="storage is unavailable"):
        journal(tmp_path).prepare(request(), scope=SCOPE)
    with sqlite3.connect(journal(tmp_path).journal.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM execution_scope").fetchone()[0] == 0
        assert (
            db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='spot_roundtrip'"
            ).fetchone()
            is None
        )


def test_original_opening_client_identity_cannot_be_reused_for_roundtrip(tmp_path):
    j = journal(tmp_path)
    with position_transaction(j.journal, write=True) as transaction:
        j.journal._prepare_scope(transaction, SCOPE)
        transaction.connection.execute(
            "INSERT INTO open_intent VALUES (?, ?, ?, 'RECORDED')",
            ("sait-o-" + IDENTITY, "historical", "{}"),
        )
    with pytest.raises(OpenIntentError, match="identity is already consumed"):
        j.prepare(request(), scope=SCOPE)
    with sqlite3.connect(j.journal.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_completed_roundtrip_ids_remain_consumed_for_normal_opening_and_closing(
    tmp_path,
):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    record(j, req, receipt(req))
    second(j, req, "1")
    record(j, req, receipt(req, first=False, amount="1", order_id=43), first=False)
    revision, _ = j.pending(scope=SCOPE)
    j.complete(req.operation_id, scope=SCOPE, expected_revision=revision)
    with pytest.raises(OpenIntentError, match="identity belongs to a roundtrip"):
        j.journal.prepare(
            position(open_client_order_id="sait-o-" + IDENTITY), scope=SCOPE
        )
    store = PositionsStore(tmp_path)
    owned = position(exchange_status="FILLED", open_exchange_order_id="100")
    store.record_open(owned)
    with pytest.raises(OpenIntentError, match="identity belongs to a roundtrip"):
        prepare_close(
            store, owned, client_id="sait-c-" + IDENTITY, scope=SCOPE, reduce_only=False
        )
    prepare_close(
        store, owned, client_id="sait-c-fresh-close", scope=SCOPE, reduce_only=False
    )


def test_tampered_oversize_active_record_is_rejected_before_json_decoding(tmp_path):
    j = journal(tmp_path)
    j.prepare(request(), scope=SCOPE)
    with sqlite3.connect(j.journal.path) as db:
        db.execute("UPDATE spot_roundtrip SET record_json=?", (" " * 2_100_001,))
    with pytest.raises(OpenIntentError, match="oversized"):
        j.pending(scope=SCOPE)


@pytest.mark.parametrize(
    "changes",
    [
        {"first_amount": "0"},
        {"first_amount": "101.01"},
        {"initial_quote_allocation": "NaN"},
        {"symbol": "BNBUSDC"},
        {"mode": "sell-sell"},
        {"operation_id": "x" * 29},
    ],
)
def test_invalid_initial_allocation_or_identity_has_no_storage_side_effect(
    tmp_path, changes
):
    with pytest.raises((OpenIntentError, ValueError)):
        journal(tmp_path).prepare(request(**changes), scope=SCOPE)
    assert not journal(tmp_path).journal.path.exists()


def test_base_amount_must_match_exact_eight_decimal_transmission():
    with pytest.raises(OpenIntentError, match="bound exactly"):
        request("sell-buy", first_amount="0.123456789")


@pytest.mark.parametrize("first", [True, False])
def test_partial_or_cancelled_orders_remain_obligations(tmp_path, first):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    if not first:
        record(j, req, receipt(req))
        second(j, req, "1")
    partial = receipt(
        req,
        first=first,
        amount=None if first else "1",
        quantity="0.5",
        status="PARTIALLY_FILLED",
    )
    record(j, req, partial, first=first)
    revision, before = j.pending(scope=SCOPE)
    assert before["phase"] == ("FIRST_UNKNOWN" if first else "SECOND_UNKNOWN")
    cancelled = {**partial, "status": "CANCELED"}
    record(j, req, cancelled, first=first)
    with pytest.raises(OpenIntentError):
        if first:
            second(j, req, "0.5")
        else:
            revision, _ = j.pending(scope=SCOPE)
            j.complete(req.operation_id, scope=SCOPE, expected_revision=revision)
    assert j.journal.entry_block_reason() == "unresolved_spot_roundtrip=1"


@pytest.mark.parametrize(
    "changes",
    [
        {"fills": []},
        {"clientOrderId": "foreign"},
        {"cummulativeQuoteQty": "99"},
        {"origQuoteOrderQty": "200"},
        {"side": "SELL"},
    ],
)
def test_incomplete_or_foreign_ack_cannot_clear_unknown(tmp_path, changes):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    before = j.pending(scope=SCOPE)
    with pytest.raises(OpenIntentError):
        record(j, req, {**receipt(req), **changes})
    assert j.pending(scope=SCOPE) == before


@pytest.mark.parametrize("fee", ["0.1", "0"])
def test_unallocated_fee_asset_requires_reconciliation_only_when_it_debits_assets(
    tmp_path, fee
):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    observed = receipt(req, fee=fee, fee_asset="BNB")
    if fee != "0":
        with pytest.raises(OpenIntentError, match="unallocated asset"):
            record(j, req, observed)
        assert j.pending(scope=SCOPE)[1]["phase"] == "FIRST_UNKNOWN"
    else:
        record(j, req, observed)
        assert j.pending(scope=SCOPE)[1]["phase"] == "FIRST_TERMINAL"


def test_native_debit_beyond_initial_allocation_remains_unknown(tmp_path):
    req = request(initial_quote_allocation="100")
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    with pytest.raises(OpenIntentError, match="allocated assets"):
        record(j, req, receipt(req, fee="1", fee_asset="USDC"))
    assert j.pending(scope=SCOPE)[1]["first"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "delete_row",
        "delete_control",
        "drop_table",
        "change_record",
        "delete_pair",
        "lost_revision",
    ],
)
def test_enrolled_record_loss_or_drift_never_becomes_fresh_storage(tmp_path, mutation):
    j = journal(tmp_path)
    j.prepare(request(), scope=SCOPE)
    with sqlite3.connect(j.journal.path) as db:
        sql = {
            "delete_row": "DELETE FROM spot_roundtrip",
            "delete_control": "DELETE FROM spot_roundtrip_control",
            "drop_table": "DROP TABLE spot_roundtrip",
            "change_record": "UPDATE spot_roundtrip SET record_json='{}'",
            "delete_pair": "DROP TABLE position_replacement",
            "lost_revision": "UPDATE spot_roundtrip_control SET revision=0",
        }[mutation]
        db.execute(sql)
    with pytest.raises((OpenIntentError, ValueError)):
        j.pending(scope=SCOPE)
    if mutation != "delete_pair":
        assert j.journal.entry_block_reason() == "opening_intent_journal_unreadable"


def test_scope_rotation_cannot_observe_or_complete_another_operation(tmp_path):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    other = replace(SCOPE, credential_fingerprint="b" * 64)
    with pytest.raises(OpenIntentError):
        j.pending(scope=other)
    revision, _ = j.pending(scope=SCOPE)
    with pytest.raises(OpenIntentError):
        j.record_order(
            req.operation_id,
            receipt(req),
            first=True,
            scope=other,
            expected_revision=revision,
        )
    assert j.pending(scope=SCOPE)[1]["first"] is None


@pytest.mark.parametrize(
    "changes", [{"orderId": 43}, {"executedQty": "0.4"}, {"fee": "0.0001"}]
)
def test_cumulative_identity_or_fee_regression_cannot_replace_prior_evidence(
    tmp_path, changes
):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    original = receipt(
        req, quantity="0.5", fee="0.0005", fee_asset="BTC", status="PARTIALLY_FILLED"
    )
    record(j, req, original)
    before = j.pending(scope=SCOPE)
    observed = receipt(
        req,
        quantity="1",
        fee=changes.get("fee", "0.001"),
        fee_asset="BTC",
        order_id=changes.get("orderId", 42),
    )
    if "executedQty" in changes:
        observed = receipt(
            req,
            quantity="0.4",
            fee="0.0004",
            fee_asset="BTC",
            status="PARTIALLY_FILLED",
        )
    with pytest.raises(OpenIntentError):
        record(j, req, observed)
    assert j.pending(scope=SCOPE) == before


def test_stale_revision_cannot_overwrite_a_new_native_observation(tmp_path):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    old_revision, _ = j.pending(scope=SCOPE)
    record(j, req, receipt(req, quantity="0.5", status="PARTIALLY_FILLED"))
    with pytest.raises(OpenIntentError, match="changed during"):
        j.record_order(
            req.operation_id,
            receipt(req),
            first=True,
            scope=SCOPE,
            expected_revision=old_revision,
        )


def test_concurrent_first_reservation_has_one_winner(tmp_path):
    barrier = threading.Barrier(2)

    def prepare(identity):
        barrier.wait(timeout=5)
        try:
            journal(tmp_path).prepare(request(operation_id=identity), scope=SCOPE)
            return "prepared"
        except OpenIntentError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(prepare, ("a" * 29, "b" * 29)))
    assert sorted(results) == ["blocked", "prepared"]
    assert (
        journal(tmp_path).journal.entry_block_reason() == "unresolved_spot_roundtrip=1"
    )


@pytest.mark.parametrize(
    "phase,expected",
    [
        (0, "FIRST_UNKNOWN"),
        (1, "FIRST_TERMINAL"),
        (2, "SECOND_UNKNOWN"),
        (3, "SECOND_TERMINAL"),
    ],
)
def test_abrupt_child_exit_retains_each_exact_durable_boundary(
    tmp_path, phase, expected
):
    script = """import os,sys
from pathlib import Path
from simple_ai_trading.binance_execution_scope import BinanceExecutionScope,BINANCE_SPOT_TESTNET
from simple_ai_trading.binance_open_intents import BinanceOpenIntentJournal
from simple_ai_trading.binance_spot_roundtrip_intents import SpotRoundtripJournal,SpotRoundtripRequest
scope=BinanceExecutionScope.from_api_key(BINANCE_SPOT_TESTNET,'spot','offline-journal-key')
j=SpotRoundtripJournal(BinanceOpenIntentJournal(Path(sys.argv[1])))
req=SpotRoundtripRequest('a'*29,'BTCUSDC','buy-sell','100','0','101')
j.prepare(req,scope=scope)
phase=int(sys.argv[2])
if phase>=1:
 first={'symbol':'BTCUSDC','side':'BUY','type':'MARKET','status':'FILLED','orderId':42,'clientOrderId':req.binding(first=True).client_order_id,'origQty':'1','origQuoteOrderQty':'100','executedQty':'1','cummulativeQuoteQty':'100','fills':[{'qty':'1','price':'100','commission':'0','commissionAsset':'BTC','tradeId':42}]}
 j.record_order(req.operation_id,first,first=True,scope=scope,expected_revision=1)
if phase>=2:
 j.prepare_second(req.operation_id,'1',scope=scope,expected_revision=2)
if phase>=3:
 second={**first,'side':'SELL','orderId':43,'clientOrderId':req.binding(first=False,amount='1').client_order_id,'fills':[{'qty':'1','price':'100','commission':'0','commissionAsset':'USDC','tradeId':43}]}
 second.pop('origQuoteOrderQty')
 j.record_order(req.operation_id,second,first=False,scope=scope,expected_revision=3)
os._exit(23)
"""
    target = tmp_path / "binance_open_intents.sqlite3"
    completed = run_captured_process(
        [sys.executable, "-c", script, str(target), str(phase)],
        cwd=Path(__file__).resolve().parents[1],
        env=None,
        input_text="",
        timeout=20,
    )
    assert completed.returncode == 23
    revision, state = journal(tmp_path).pending(scope=SCOPE)
    assert revision == phase + 1 and state["phase"] == expected
    assert state["request"]["first_amount"] == "100"
    assert request().binding(first=True).client_order_id == "sait-o-" + IDENTITY
    assert (
        journal(tmp_path).journal.entry_block_reason() == "unresolved_spot_roundtrip=1"
    )


@pytest.mark.parametrize("mode", ["buy-sell", "sell-buy"])
def test_explicit_fee_asset_units_survive_both_legs_and_restart(tmp_path, mode):
    req = request(mode, fee_asset_allocations=(("BNB", "0.003"),))
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    prepared = j.pending(scope=SCOPE)[1]
    assert prepared["format"] == 2
    assert prepared["request"]["fee_asset_allocations"] == [["BNB", "0.003"]]
    record(j, req, receipt(req, fee="0.001", fee_asset="BNB"))
    j = journal(tmp_path)
    second(j, req, "1" if mode == "buy-sell" else "100")
    record(
        j,
        req,
        receipt(
            req,
            first=False,
            amount="1" if mode == "buy-sell" else "100",
            fee="0.002",
            fee_asset="BNB",
            order_id=43,
        ),
        first=False,
    )
    revision, _ = j.pending(scope=SCOPE)
    j.complete(req.operation_id, scope=SCOPE, expected_revision=revision)
    assert journal(tmp_path).pending(scope=SCOPE) is None
    completed = journal(tmp_path).operation(req.operation_id, scope=SCOPE)
    restored = _validate_record(completed)
    assert restored == req
    assert restored.allocations["BNB"] == Fraction(3, 1000)
    assert completed["first"]["fills"][0]["commission"] == "0.001"
    assert completed["second"]["fills"][0]["commission"] == "0.002"


@pytest.mark.parametrize("first", [True, False])
def test_fee_debits_share_one_allocation_and_cannot_clear_unknown_after_overrun(
    tmp_path, first
):
    req = request(fee_asset_allocations=(("BNB", "0.001"),))
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    if not first:
        record(j, req, receipt(req, fee="0.001", fee_asset="BNB"))
        second(j, req, "1")
    before = j.pending(scope=SCOPE)
    with pytest.raises(OpenIntentError, match="allocated assets"):
        record(
            j,
            req,
            receipt(
                req,
                first=first,
                amount=None if first else "1",
                fee="0.002" if first else "0.001",
                fee_asset="BNB",
                order_id=42 if first else 43,
            ),
            first=first,
        )
    assert j.pending(scope=SCOPE) == before
    assert j.journal.entry_block_reason() == "unresolved_spot_roundtrip=1"


def test_split_fill_fee_fallback_preserves_net_received_second_principal(tmp_path):
    req = request(fee_asset_allocations=(("BNB", "0.002"),))
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    observed = receipt(req)
    original = observed["fills"][0]
    observed["fills"] = [
        {**original, "qty": "0.4", "commissionAsset": "BNB", "commission": "0.0004"},
        {
            **original,
            "qty": "0.6",
            "commissionAsset": "BTC",
            "commission": "0.0006",
            "tradeId": 43,
        },
    ]
    record(j, req, observed)
    before = j.pending(scope=SCOPE)
    with pytest.raises(OpenIntentError, match="first-leg owned receipts"):
        second(j, req, "1")
    assert j.pending(scope=SCOPE) == before
    second(j, req, "0.9994")
    record(
        j,
        req,
        receipt(
            req,
            first=False,
            amount="0.9994",
            quantity="0.9994",
            fee="0.001",
            fee_asset="BNB",
            order_id=44,
        ),
        first=False,
    )
    assert j.pending(scope=SCOPE)[1]["phase"] == "SECOND_TERMINAL"


@pytest.mark.parametrize(
    "grants",
    [
        [],
        (("BNB", "0.001"), ["X", "1"]),
        (("BNB",),),
        (("BTC", "1"),),
        (("USDC", "1"),),
        (("bnb", "1"),),
        (("BNB", "1"), ("BNB", "2")),
        (("ZZZ", "1"), ("BNB", "1")),
        (("BNB", 0.001),),
        (("BNB", "-1"),),
        (("BNB", "1e-3"),),
        (("BNB", "NaN"),),
        (("BNB", None),),
        tuple((f"X{i:02}", "1") for i in range(17)),
    ],
)
def test_fee_allocations_require_immutable_canonical_exact_units(grants):
    with pytest.raises(OpenIntentError):
        request(fee_asset_allocations=grants)


def test_zero_fee_allocation_is_not_permission_for_a_positive_debit(tmp_path):
    req = request(fee_asset_allocations=(("BNB", "0"),))
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    with pytest.raises(OpenIntentError, match="allocated assets"):
        record(j, req, receipt(req, fee="0.00000001", fee_asset="BNB"))
    assert j.pending(scope=SCOPE)[1]["phase"] == "FIRST_UNKNOWN"


def test_no_extra_grants_preserves_legacy_record_format_and_fields(tmp_path):
    req = request()
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    value = j.pending(scope=SCOPE)[1]
    assert value["format"] == 1
    assert "fee_asset_allocations" not in value["request"]
    assert _validate_record(value) == req


@pytest.mark.parametrize(
    "change",
    [
        {"format": 1},
        {"fee_asset_allocations": []},
        {"fee_asset_allocations": {}},
        {"fee_asset_allocations": ["BNB", "1"]},
        {"fee_asset_allocations": [["BNB", "1", "foreign"]]},
        {"fee_asset_allocations": [["BNB", 0.001]]},
        {"fee_asset_allocations": [["BTC", "1"]]},
    ],
)
def test_stored_fee_grants_cannot_be_guessed_or_mislabelled(tmp_path, change):
    req = request(fee_asset_allocations=(("BNB", "0.003"),))
    j = journal(tmp_path)
    j.prepare(req, scope=SCOPE)
    value = json.loads(json.dumps(j.pending(scope=SCOPE)[1]))
    if "format" in change:
        value.update(change)
    else:
        value["request"].update(change)
    with pytest.raises(OpenIntentError):
        _validate_record(value)
