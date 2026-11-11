"""Apply synthetic native cumulative Spot fills with real enrolled local storage."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from simple_ai_trading.binance_close_application import (
    apply_spot_closing_recovery,
    closing_recovery_sha256,
)
from simple_ai_trading.binance_close_intents import prepare_close
from simple_ai_trading.binance_close_recovery import collect_closing_recovery
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage
from simple_ai_trading.binance_open_intents import OpenIntentError
from simple_ai_trading.binance_owned_execution import (
    _apply_open_order,
    _apply_close_order,
    close_trade_template,
)
from simple_ai_trading.positions import OpenPosition, PositionsStore

SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-application"
)
CLIENT_ID = "sait-c-application"


def stage(
    tmp_path,
    enroll,
    *,
    status="CANCELED",
    quantity="0.6",
    prefix=False,
    fee_asset="USDT",
    legacy=False,
    prefix_id=10,
):
    store = PositionsStore(tmp_path)
    authority = enroll(store, SCOPE, start=False)
    requested = OpenPosition(
        id="application",
        symbol="BTCUSDT",
        market_type="spot",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        leverage=1,
        opened_at_ms=1,
        notional=100,
        dry_run=False,
        open_client_order_id="sait-o-application",
    )
    store.opening_intents.prepare(requested, scope=SCOPE, create=False)
    opened = _apply_open_order(
        requested,
        {
            "symbol": "BTCUSDT",
            "type": "MARKET",
            "side": "BUY",
            "status": "FILLED",
            "orderId": 1,
            "clientOrderId": requested.open_client_order_id,
            "executedQty": "1",
            "origQty": "1",
            "cummulativeQuoteQty": "100",
            "fills": [
                {
                    "tradeId": 1,
                    "qty": "1",
                    "price": "100",
                    "commission": "0.1",
                    "commissionAsset": "USDT",
                }
            ],
        },
    )
    if legacy:
        opened = replace(
            opened,
            spot_gross_entry_quantity="",
            spot_entry_base_commission="",
            spot_entry_cash_receipt="",
        )
    store.record_open(opened, create=False)
    store.opening_intents.record_complete(requested, opened, scope=SCOPE)
    prepare_close(
        store, opened, client_id=CLIENT_ID, scope=SCOPE, reduce_only=True, create=False
    )
    order = {
        "symbol": "BTCUSDT",
        "orderId": 2,
        "clientOrderId": CLIENT_ID,
        "type": "MARKET",
        "side": "SELL",
        "status": status,
        "origQty": "1",
        "executedQty": quantity,
        "cummulativeQuoteQty": str(Decimal(quantity) * 110),
        "time": 10,
        "updateTime": 30,
    }
    trades = []
    amounts = (
        [] if quantity == "0" else ["0.4", str(Decimal(quantity) - Decimal("0.4"))]
    )
    for index, amount in enumerate(amounts):
        trades.append(
            {
                "symbol": "BTCUSDT",
                "orderId": 2,
                "id": 10 + index,
                "qty": amount,
                "price": "110",
                "quoteQty": str(Decimal(amount) * 110),
                "commission": str(Decimal(amount) * Decimal("0.11")),
                "commissionAsset": fee_asset,
                "time": 20 + index,
                "isBuyer": False,
                "isMaker": False,
            }
        )
    if prefix:
        first = dict(
            order,
            status="PARTIALLY_FILLED",
            executedQty="0.4",
            cummulativeQuoteQty="44",
            fills=[
                {
                    "tradeId": prefix_id,
                    "qty": "0.4",
                    "price": "110",
                    "commission": "0.044",
                    "commissionAsset": "USDT",
                }
            ],
        )
        previous = _apply_close_order(
            close_trade_template(opened, 110, "partial", clock=lambda: 0.02),
            first,
            CLIENT_ID,
        )
        store.record_close_result(opened, previous, create=False)

    class Venue:
        def execution_scope(self):
            return SCOPE

        def get_order(self, *args, **kwargs):
            return order

        def get_order_trades(self, *args, **kwargs):
            return trades

    collect_closing_recovery(Venue(), store, client_id=CLIENT_ID)
    return store, authority.control, opened


def apply(store, control, **changes):
    options = dict(
        client_id=CLIENT_ID,
        scope=SCOPE,
        expected_recovery_sha256=closing_recovery_sha256(
            store, scope=SCOPE, client_id=CLIENT_ID
        ),
        control=control,
    )
    options.update(changes)
    return apply_spot_closing_recovery(store, **options)


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize(
    "status,quantity,remaining",
    [
        ("FILLED", "1", "0"),
        ("CANCELED", "0.6", "0.4"),
        ("EXPIRED", "0.6", "0.4"),
        ("EXPIRED_IN_MATCH", "0.6", "0.4"),
    ],
)
def test_cumulative_close_replaces_partial_once_and_keeps_stop(
    tmp_path, enrolled_binance_runtime, prefix, status, quantity, remaining
):
    store, control, original = stage(
        tmp_path,
        enrolled_binance_runtime,
        status=status,
        quantity=quantity,
        prefix=prefix,
    )
    stopped = control.snapshot()
    result = apply(store, control)
    assert (
        result.executed_quantity == quantity and result.remaining_quantity == remaining
    )
    assert not result.rearmed
    opens, closed = PositionsStore(tmp_path).load_snapshot(strict=True)
    assert len(closed) == 1 and Decimal(str(closed[0].qty)) == Decimal(quantity)
    assert len(opens) == int(remaining != "0")
    if opens:
        assert Decimal(str(opens[0].qty)) == Decimal(remaining)
    expected_pnl = float(Decimal(quantity) * Decimal("9.79"))
    assert closed[0].realized_pnl == pytest.approx(expected_pnl)
    assert store.opening_intents.entry_block_reason() is None
    assert control.snapshot() == stopped
    before = store.load_snapshot(strict=True)
    assert apply(store, control) == result
    assert store.load_snapshot(strict=True) == before
    assert control.snapshot() == stopped


@pytest.mark.parametrize(
    "status", ["REJECTED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH"]
)
def test_zero_fill_resolves_only_its_order_and_preserves_owned_lot(
    tmp_path, enrolled_binance_runtime, status
):
    store, control, original = stage(
        tmp_path, enrolled_binance_runtime, status=status, quantity="0"
    )
    result = apply(store, control)
    assert result.executed_quantity == "0"
    assert store.load_snapshot(strict=True) == ([original], [])
    assert store.opening_intents.entry_block_reason() is None
    assert control.snapshot().state == "stop_requested"


def test_new_close_after_zero_fill_uses_unconsumed_identity(
    tmp_path, enrolled_binance_runtime
):
    from simple_ai_trading.binance_owned_execution import _next_close_client_order_id

    store, control, original = stage(
        tmp_path, enrolled_binance_runtime, status="CANCELED", quantity="0"
    )
    apply(store, control)
    next_id = _next_close_client_order_id(store, original)
    assert next_id != CLIENT_ID
    prepare_close(
        store, original, client_id=next_id, scope=SCOPE, reduce_only=True, create=False
    )
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    apply(store, control)
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize(
    "problem",
    [
        "digest",
        "legacy",
        "prefix",
        "inventory",
        "third_fee",
        "base_fee",
        "marker",
        "running",
        "scope",
    ],
)
def test_unsupported_or_changed_inputs_leave_unknown_and_projection_unchanged(
    tmp_path, enrolled_binance_runtime, problem
):
    store, control, original = stage(
        tmp_path,
        enrolled_binance_runtime,
        legacy=problem == "legacy",
        prefix=problem == "prefix",
        prefix_id=99,
        fee_asset="BNB"
        if problem == "third_fee"
        else "BTC"
        if problem == "base_fee"
        else "USDT",
    )
    changes = {}
    if problem == "digest":
        changes["expected_recovery_sha256"] = "0" * 64
    if problem == "scope":
        changes["scope"] = BinanceExecutionScope.from_api_key(
            BINANCE_SPOT_TESTNET, "spot", "other-key"
        )
    if problem == "inventory":
        store.record_open(replace(original, qty=0.8, notional=80), create=False)
    if problem == "marker":
        BinanceExecutionStorage(store.opening_intents, control).enrollment_path.unlink()
    if problem == "running":
        assert control.complete_stale_stop(exposure_closed=True)
        control.acquire()
    before = store.load_snapshot(strict=True)
    with pytest.raises(OpenIntentError):
        apply(store, control, **changes)
    assert store.load_snapshot(strict=True) == before
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


def test_concurrent_application_is_idempotent(tmp_path, enrolled_binance_runtime):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: apply(store, control), range(2)))
    assert results[0] == results[1]
    assert len(store.load_ledger(strict=True)) == 1


def test_application_receipt_tampering_rejects_cached_success(
    tmp_path, enrolled_binance_runtime
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime)
    apply(store, control)
    with sqlite3.connect(store.opening_intents.path) as connection:
        connection.execute("UPDATE closing_application SET result_json='{}'")
    before = store.load_snapshot(strict=True)
    with pytest.raises(OpenIntentError, match="conflicts"):
        apply(store, control)
    assert store.load_snapshot(strict=True) == before


def test_feedback_failure_after_commit_is_retryable_without_duplicate_accounting(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    from simple_ai_trading import binance_close_application as module

    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    original = module.write_learning_feedback
    monkeypatch.setattr(
        module,
        "write_learning_feedback",
        lambda _: (_ for _ in ()).throw(OSError("synthetic feedback failure")),
    )
    with pytest.raises(OpenIntentError, match="inspect retained state"):
        apply(store, control)
    assert store.opening_intents.entry_block_reason() is None
    assert len(store.load_ledger(strict=True)) == 1
    monkeypatch.setattr(module, "write_learning_feedback", original)
    apply(store, control)
    assert len(store.load_ledger(strict=True)) == 1


def test_failure_before_redo_commit_preserves_unknown_and_both_ledgers(
    tmp_path, enrolled_binance_runtime, monkeypatch
):
    from simple_ai_trading.position_transactions import PositionTransaction

    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    before = store.load_snapshot(strict=True)
    original = PositionTransaction.replace

    def fail(*args, **kwargs):
        raise OSError("synthetic precommit failure")

    monkeypatch.setattr(PositionTransaction, "replace", fail)
    with pytest.raises(OpenIntentError):
        apply(store, control)
    assert store.load_snapshot(strict=True) == before
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"
    with sqlite3.connect(store.opening_intents.path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='closing_application'"
        ).fetchone() == (0,)
    monkeypatch.setattr(PositionTransaction, "replace", original)
    apply(store, control)


@pytest.mark.parametrize("phase", ["committed", "first_projection"])
def test_actual_child_exit_recovers_paired_inventory_and_intent_once(
    tmp_path, enrolled_binance_runtime, phase
):
    from tools.cocoindex_agent_search import run_captured_process

    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    before = (store.open_path.read_bytes(), store.ledger_path.read_bytes())
    script = tmp_path / "owned_application_exit.py"
    script.write_text(
        """import os
from pathlib import Path
import sys
from simple_ai_trading import position_transactions as transactions
from simple_ai_trading.binance_runtime_control import BinanceRuntimeControl
from simple_ai_trading.positions import PositionsStore
from test_binance_close_application import apply
root = Path(sys.argv[1])
phase = sys.argv[2]
store = PositionsStore(root)
if phase == "committed":
    original = transactions._recover
    def recover(connection, root):
        row = connection.execute("SELECT state FROM position_replacement WHERE id=1").fetchone()
        if row == ("PREPARED",):
            assert connection.execute("SELECT state FROM close_intent").fetchone() == ("RECORDED",)
            assert connection.execute("SELECT COUNT(*) FROM closing_application").fetchone() == (1,)
            os._exit(74)
        return original(connection, root)
    transactions._recover = recover
else:
    original = transactions.write_bytes_atomic
    def write(path, body):
        original(path, body)
        os._exit(75)
    transactions.write_bytes_atomic = write
apply(store, BinanceRuntimeControl(root / "binance_runtime_control.sqlite3"))
raise AssertionError("failure injection was not reached")
""",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
    }
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    env.update(
        PYTHONPATH=os.pathsep.join(str(root / path) for path in ("src", "tests", ".")),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = run_captured_process(
        [sys.executable, "-B", str(script), str(tmp_path), phase],
        cwd=root,
        env=env,
        input_text="",
        timeout=25,
        inherit_environment=False,
    )
    assert result.returncode == (74 if phase == "committed" else 75), result.stderr
    if phase == "committed":
        assert (store.open_path.read_bytes(), store.ledger_path.read_bytes()) == before
    recovered = apply(store, control)
    assert recovered.remaining_quantity == "0.4"
    assert store.load_open(strict=True)[0].qty == 0.4
    assert len(store.load_ledger(strict=True)) == 1
    assert store.opening_intents.entry_block_reason() is None
    assert apply(store, control) == recovered
    assert control.snapshot().state == "stop_requested"


def test_recovery_guard_serializes_both_rearm_and_owned_close(
    tmp_path, enrolled_binance_runtime
):
    from simple_ai_trading.binance_runtime_control import (
        BinanceRuntimeControl,
        RuntimeControlError,
    )

    store, control, _ = stage(tmp_path, enrolled_binance_runtime)
    other = BinanceRuntimeControl(control.ledger_path, interlock_timeout_seconds=1)

    def try_rearm():
        with pytest.raises(RuntimeControlError):
            other.complete_stale_stop(exposure_closed=True)

    def try_close():
        with pytest.raises(RuntimeControlError):
            with other.close_guard(timeout_seconds=1):
                pytest.fail("competing close entered recovery")

    with control.stopped_recovery_guard(), ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(try_rearm), pool.submit(try_close)]
        for future in futures:
            future.result(timeout=5)
    assert control.snapshot().state == "stop_requested"


def test_repeated_cached_application_releases_process_resources(
    tmp_path, enrolled_binance_runtime, process_resource_sample
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    apply(store, control)
    before = process_resource_sample()
    for _ in range(12):
        apply(store, control)
    after = process_resource_sample()
    assert after[0] <= before[0] + 4
    assert after[1] == before[1]
    assert after[2] <= before[2] + 8 * 1024 * 1024


@pytest.mark.parametrize("digest", [None, True, "short", "A" * 64])
def test_application_requires_an_explicit_valid_inspection_digest(
    tmp_path, enrolled_binance_runtime, digest
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime)
    before = store.load_snapshot(strict=True)
    with pytest.raises(OpenIntentError, match="exact Spot scope"):
        apply(store, control, expected_recovery_sha256=digest)
    assert store.load_snapshot(strict=True) == before
    assert store.opening_intents.entry_block_reason() == "unresolved_closing_intents=1"


@pytest.mark.parametrize("damage", ["source", "schema", "state", "missing_application"])
def test_corrupt_source_or_application_never_becomes_cached_success(
    tmp_path, enrolled_binance_runtime, damage
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime, prefix=True)
    digest = closing_recovery_sha256(store, scope=SCOPE, client_id=CLIENT_ID)
    if damage != "source":
        apply(store, control)
    before = store.load_snapshot(strict=True)
    with sqlite3.connect(store.opening_intents.path) as connection:
        if damage == "source":
            connection.execute("UPDATE closing_recovery SET evidence_json='{}'")
        elif damage == "schema":
            connection.execute("ALTER TABLE closing_application ADD COLUMN extra TEXT")
        elif damage == "state":
            connection.execute("UPDATE close_intent SET state='UNKNOWN'")
        else:
            connection.execute("DELETE FROM closing_application")
    with pytest.raises(OpenIntentError):
        apply_spot_closing_recovery(
            store,
            client_id=CLIENT_ID,
            scope=SCOPE,
            control=control,
            expected_recovery_sha256=digest,
        )
    assert store.load_snapshot(strict=True) == before


def test_missing_journal_after_inspection_is_not_recreated(
    tmp_path, enrolled_binance_runtime
):
    store, control, _ = stage(tmp_path, enrolled_binance_runtime)
    digest = closing_recovery_sha256(store, scope=SCOPE, client_id=CLIENT_ID)

    def bytes_or_none():
        return tuple(
            path.read_bytes() if path.exists() else None
            for path in (store.open_path, store.ledger_path)
        )

    before = bytes_or_none()
    store.opening_intents.path.rename(tmp_path / "retained-original.sqlite3")
    with pytest.raises(OpenIntentError):
        apply_spot_closing_recovery(
            store,
            client_id=CLIENT_ID,
            scope=SCOPE,
            control=control,
            expected_recovery_sha256=digest,
        )
    assert not store.opening_intents.path.exists()
    assert bytes_or_none() == before
