"""Real enrolled storage and constructed native responses; no network access."""

from dataclasses import replace
import threading

import pytest

from simple_ai_trading.api import BinanceAPIError
from simple_ai_trading.binance_execution_scope import (
    BINANCE_SPOT_TESTNET,
    BinanceExecutionScope,
)
from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage
from simple_ai_trading.binance_open_intents import (
    BinanceOpenIntentJournal,
    OpenIntentError,
)
from simple_ai_trading.binance_runtime_control import (
    BinanceRuntimeControl,
    RuntimeControlBlocked,
    RuntimeLeaseInterlock,
)
from simple_ai_trading.binance_spot_commissions import (
    SpotCommissionComponent,
    SpotCommissionDiscount,
    SpotCommissionSchedule,
)
from simple_ai_trading.binance_spot_roundtrip_execution import SpotRoundtripExecution
from simple_ai_trading.binance_spot_roundtrip_intents import (
    SpotRoundtripJournal,
    SpotRoundtripRequest,
)
from fractions import Fraction


SCOPE = BinanceExecutionScope.from_api_key(
    BINANCE_SPOT_TESTNET, "spot", "offline-gateway-key"
)
IDENTITY = "a" * 29
ZERO = SpotCommissionComponent(*(Fraction(0) for _ in range(4)))
SCHEDULE = SpotCommissionSchedule(
    "BTCUSDC", ZERO, ZERO, ZERO, SpotCommissionDiscount(False, False, None, None)
)


def receipt(req, *, first=True, amount=None):
    binding = req.binding(first=first, amount=amount)
    buy = binding.client_order_id.startswith("sait-o-")
    # Both legs exchange one base unit for 100 quote units in these fixtures.
    return {
        "symbol": "BTCUSDC",
        "side": "BUY" if buy else "SELL",
        "type": "MARKET",
        "status": "FILLED",
        "orderId": 42 if first else 43,
        "clientOrderId": binding.client_order_id,
        "origQty": "1",
        "executedQty": "1",
        "cummulativeQuoteQty": "100",
        **({"origQuoteOrderQty": binding.quote_quantity} if buy else {}),
        "fills": [
            {
                "price": "100",
                "qty": "1",
                "commission": "0",
                "commissionAsset": "BTC" if buy else "USDC",
                "tradeId": 100 if first else 101,
            }
        ],
    }


class Client:
    def __init__(self, journal, req):
        self.journal, self.req = journal, req
        self.scope, self.schedule = SCOPE, SCHEDULE
        self.account = {
            "canTrade": True,
            "balances": [
                {"asset": "BTC", "free": "1000", "locked": "0"},
                {"asset": "USDC", "free": "100000", "locked": "0"},
            ],
        }
        self.writes, self.queries = [], []
        self.failure, self.order_hook, self.account_hook = None, None, None

    def execution_scope(self):
        return self.scope

    def get_account(self, *, expected_scope):
        assert expected_scope == self.scope
        if self.account_hook:
            self.account_hook()
        return self.account

    def get_spot_commission_schedule(self, symbol, *, expected_scope):
        if (symbol, expected_scope) != (self.req.symbol, self.scope):
            raise BinanceAPIError("commission scope changed")
        return self.schedule

    def _write(self, amount, *, client_order_id, expected_scope):
        assert expected_scope == self.scope
        revision, state = self.journal.pending(scope=expected_scope)
        assert revision >= 1
        first = state["phase"] == "FIRST_UNKNOWN"
        assert state["phase"] in {"FIRST_UNKNOWN", "SECOND_UNKNOWN"}
        binding = self.req.binding(first=first, amount=state["second_amount"])
        assert client_order_id == binding.client_order_id
        assert str(amount) in {self.req.first_amount, "1.0", "100", "1"}
        self.writes.append((state["phase"], client_order_id))
        if self.order_hook:
            self.order_hook()
        if self.failure:
            raise self.failure
        return receipt(self.req, first=first, amount=state["second_amount"])

    def place_spot_quote_buy(self, symbol, quote_quantity, *, dry_run, **kwargs):
        assert symbol == self.req.symbol and dry_run is False
        return self._write(quote_quantity, **kwargs)

    def place_order(self, symbol, side, quantity, *, dry_run, **kwargs):
        assert (symbol, side, dry_run) == (self.req.symbol, "SELL", False)
        return self._write(quantity, **kwargs)

    def get_order(
        self, symbol, *, orig_client_order_id, expected_scope, expected_order_binding
    ):
        assert (symbol, expected_scope) == (self.req.symbol, self.scope)
        _, state = self.journal.pending(scope=self.scope)
        first = state["phase"] == "FIRST_UNKNOWN"
        result = receipt(self.req, first=first, amount=state["second_amount"])
        assert result["clientOrderId"] == orig_client_order_id
        expected_order_binding.validate(result)
        self.queries.append("order")
        if self.failure:
            raise self.failure
        result.pop("fills")
        return result

    def get_order_trades(self, symbol, *, order_id, expected_scope):
        assert (symbol, expected_scope) == (self.req.symbol, self.scope)
        _, state = self.journal.pending(scope=self.scope)
        first = state["phase"] == "FIRST_UNKNOWN"
        order = receipt(self.req, first=first, amount=state["second_amount"])
        assert order_id == order["orderId"]
        self.queries.append("trades")
        fill = order["fills"][0]
        return [
            {
                **fill,
                "id": fill["tradeId"],
                "orderId": order_id,
                "symbol": symbol,
                "isBuyer": order["side"] == "BUY",
                "quoteQty": "100",
            }
        ]


def setup(tmp_path, mode="buy-sell"):
    req = SpotRoundtripRequest(
        IDENTITY,
        "BTCUSDC",
        mode,
        "100" if mode == "buy-sell" else "1",
        "0" if mode == "buy-sell" else "1",
        "100" if mode == "buy-sell" else "0",
    )
    journal = SpotRoundtripJournal(
        BinanceOpenIntentJournal(tmp_path / "intents.sqlite3")
    )
    control = BinanceRuntimeControl(tmp_path / "runtime.sqlite3")
    control.provision_storage()
    BinanceExecutionStorage(journal.journal, control).provision(scope=SCOPE)
    # This fixture has no positions or native account; it starts reconciled flat.
    assert control.complete_stale_stop(exposure_closed=True)
    lease = control.acquire()
    client = Client(journal, req)
    gateway = SpotRoundtripExecution(
        client, journal, RuntimeLeaseInterlock(control, lease)
    )
    return gateway, req, client, control


@pytest.mark.parametrize("mode", ["buy-sell", "sell-buy"])
def test_both_orders_commit_unknown_before_native_write_and_do_not_claim_flatness(
    tmp_path, mode
):
    gateway, req, client, _ = setup(tmp_path, mode)
    assert gateway.submit_first(req)["phase"] == "FIRST_TERMINAL"
    amount = "1" if mode == "buy-sell" else "100"
    assert gateway.submit_second(IDENTITY, amount=amount)["phase"] == "SECOND_TERMINAL"
    assert [row[0] for row in client.writes] == ["FIRST_UNKNOWN", "SECOND_UNKNOWN"]
    assert gateway.journal.journal.entry_block_reason() == "unresolved_spot_roundtrip=1"
    with pytest.raises(OpenIntentError):
        gateway.submit_second(IDENTITY, amount=amount)
    assert len(client.writes) == 2


@pytest.mark.parametrize(
    "failure", [BinanceAPIError("lost response"), KeyboardInterrupt()]
)
@pytest.mark.parametrize("leg", ["first", "second"])
def test_interruption_leaves_durable_unknown_and_never_retries(tmp_path, failure, leg):
    gateway, req, client, _ = setup(tmp_path)
    if leg == "second":
        gateway.submit_first(req)
    client.failure = failure
    with pytest.raises(type(failure)):
        gateway.submit_first(req) if leg == "first" else gateway.submit_second(
            IDENTITY, amount="1"
        )
    count = len(client.writes)
    assert count == (1 if leg == "first" else 2)
    client.failure = None
    # Dispatch is not restart recovery and cannot transmit an UNKNOWN leg again.
    with pytest.raises(OpenIntentError):
        gateway.submit_first(req) if leg == "first" else gateway.submit_second(
            IDENTITY, amount="1"
        )
    assert len(client.writes) == count
    assert gateway.recover(IDENTITY)["phase"] == (
        "FIRST_TERMINAL" if leg == "first" else "SECOND_TERMINAL"
    )
    assert client.queries == ["order", "trades"]
    assert len(client.writes) == count


@pytest.mark.parametrize("mode", ["buy-sell", "sell-buy"])
def test_stop_blocks_acquisition_but_permits_receipt_owned_sale(tmp_path, mode):
    gateway, req, client, control = setup(tmp_path, mode)
    gateway.submit_first(req)
    control.request_stop(reason="offline-stop")
    if mode == "buy-sell":
        assert gateway.submit_second(IDENTITY, amount="1")["phase"] == "SECOND_TERMINAL"
    else:
        with pytest.raises(RuntimeControlBlocked):
            gateway.submit_second(IDENTITY, amount="100")
        assert len(client.writes) == 1
    assert control.snapshot().state == "stop_requested"


@pytest.mark.parametrize(
    "change", ["permission", "duplicate", "missing", "negative", "external_fee"]
)
def test_unqualified_account_or_fee_route_rejects_before_unknown(tmp_path, change):
    gateway, req, client, _ = setup(tmp_path)
    if change == "permission":
        client.account["canTrade"] = False
    elif change == "duplicate":
        client.account["balances"].append(dict(client.account["balances"][0]))
    elif change == "missing":
        client.account["balances"].pop()
    elif change == "negative":
        client.account["balances"][0]["locked"] = "-1"
    else:
        client.schedule = replace(
            SCHEDULE, discount=SpotCommissionDiscount(True, True, "BNB", Fraction(1, 4))
        )
    with pytest.raises(OpenIntentError):
        gateway.submit_first(req)
    assert client.writes == []
    assert gateway.journal.pending(scope=SCOPE) is None


def test_stop_during_preflight_is_rechecked_before_journal_or_dispatch(tmp_path):
    gateway, req, client, control = setup(tmp_path)
    client.account_hook = lambda: control.request_stop(reason="preflight-stop")
    with pytest.raises(RuntimeControlBlocked):
        gateway.submit_first(req)
    assert client.writes == []
    assert gateway.journal.pending(scope=SCOPE) is None


def test_scope_rotation_and_expiry_fail_before_dispatch(tmp_path, monkeypatch):
    gateway, req, client, _ = setup(tmp_path)
    client.account_hook = lambda: setattr(
        client, "scope", replace(SCOPE, credential_fingerprint="0" * 64)
    )
    with pytest.raises((OpenIntentError, BinanceAPIError)):
        gateway.submit_first(req)
    assert client.writes == []
    client.scope, client.account_hook = SCOPE, None
    clock = iter([0, 6_000_000_000])
    monkeypatch.setattr(
        "simple_ai_trading.binance_spot_roundtrip_execution.time.monotonic_ns",
        lambda: next(clock),
    )
    with pytest.raises(OpenIntentError, match="expired"):
        gateway.submit_first(req)
    assert gateway.journal.pending(scope=SCOPE) is None


@pytest.mark.parametrize(
    "damage",
    [
        "foreign_order",
        "foreign_symbol",
        "wrong_side",
        "duplicate",
        "cash",
        "truncated",
        "fee",
    ],
)
def test_recovery_rejects_unbound_or_incomplete_trades_without_advancing(
    tmp_path, damage
):
    gateway, req, client, _ = setup(tmp_path)
    client.failure = BinanceAPIError("lost response")
    with pytest.raises(BinanceAPIError):
        gateway.submit_first(req)
    client.failure = None
    original = client.get_order_trades

    def damaged(*args, **kwargs):
        rows = original(*args, **kwargs)
        if damage == "foreign_order":
            rows[0]["orderId"] = 900
        elif damage == "foreign_symbol":
            rows[0]["symbol"] = "ETHUSDC"
        elif damage == "wrong_side":
            rows[0]["isBuyer"] = False
        elif damage == "duplicate":
            rows.append(dict(rows[0]))
        elif damage == "cash":
            rows[0]["quoteQty"] = "99"
        elif damage == "truncated":
            rows = []
        else:
            rows[0]["commission"] = None
        return rows

    client.get_order_trades = damaged
    before = gateway.journal.pending(scope=SCOPE)
    with pytest.raises((OpenIntentError, ValueError)):
        gateway.recover(IDENTITY)
    assert gateway.journal.pending(scope=SCOPE) == before
    assert len(client.writes) == 1


def test_competing_second_dispatches_have_one_winner(tmp_path):
    gateway, req, client, _ = setup(tmp_path)
    gateway.submit_first(req)
    barrier = threading.Barrier(2)
    client.account_hook = lambda: barrier.wait(timeout=3)
    failures = []

    def submit():
        try:
            gateway.submit_second(IDENTITY, amount="1")
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert len(failures) == 1 and isinstance(failures[0], OpenIntentError)
    assert len(client.writes) == 2
    assert gateway.journal.pending(scope=SCOPE)[1]["phase"] == "SECOND_TERMINAL"


def test_recovery_does_not_create_missing_storage(tmp_path):
    gateway, _, client, _ = setup(tmp_path)
    gateway.journal.journal.path.unlink()
    with pytest.raises(OpenIntentError, match="no matching"):
        gateway.recover(IDENTITY)
    assert client.queries == [] and client.writes == []
    assert not gateway.journal.journal.path.exists()


def test_opening_rechecks_execution_enrollment_after_account_preflight(tmp_path):
    gateway, req, client, control = setup(tmp_path)
    storage = BinanceExecutionStorage(gateway.journal.journal, control)
    client.account_hook = storage.enrollment_path.unlink
    with pytest.raises(OpenIntentError):
        gateway.submit_first(req)
    assert client.writes == []
    assert gateway.journal.pending(scope=SCOPE) is None


@pytest.mark.parametrize("after_prepare", [False, True])
def test_storage_loss_inside_opening_guard_never_creates_or_dispatches(
    tmp_path, monkeypatch, after_prepare
):
    gateway, req, client, control = setup(tmp_path)
    original = SpotRoundtripJournal.prepare
    storage = BinanceExecutionStorage(gateway.journal.journal, control)

    def interrupted(journal, request, *, scope, create):
        assert create is False
        if after_prepare:
            original(journal, request, scope=scope, create=create)
            storage.enrollment_path.unlink()
        else:
            journal.journal.path.rename(tmp_path / "retained.sqlite3")
            original(journal, request, scope=scope, create=create)

    monkeypatch.setattr(SpotRoundtripJournal, "prepare", interrupted)
    with pytest.raises(ValueError):
        gateway.submit_first(req)
    assert client.writes == []
    if after_prepare:
        assert gateway.journal.pending(scope=SCOPE)[1]["phase"] == "FIRST_UNKNOWN"
    else:
        assert not gateway.journal.journal.path.exists()


def test_second_principal_rejects_foreign_wallet_inventory_before_account_access(
    tmp_path,
):
    gateway, req, client, _ = setup(tmp_path)
    gateway.submit_first(req)
    client.account_hook = lambda: pytest.fail(
        "invalid principal must not query account"
    )
    with pytest.raises(OpenIntentError, match="owned receipts"):
        gateway.submit_second(IDENTITY, amount="2")
    assert len(client.writes) == 1


def test_scoped_account_request_rejects_rotation_before_native_transport(monkeypatch):
    from simple_ai_trading.api import BinanceClient

    client = BinanceClient("offline-key", "offline-secret", market_type="spot")
    try:
        scope = client.execution_scope()
        monkeypatch.setattr(
            client.session, "request", lambda *a, **k: pytest.fail("network forbidden")
        )
        client.api_key = "changed-offline-key"
        with pytest.raises(BinanceAPIError, match="scope"):
            client.get_account(expected_scope=scope)
    finally:
        client.session.close()


def test_scoped_account_request_reaches_transport_with_original_binding(monkeypatch):
    from simple_ai_trading.api import BinanceClient

    client = BinanceClient("offline-key", "offline-secret", market_type="spot")
    try:
        scope = client.execution_scope()
        seen = []

        def transport(method, path, params, **kwargs):
            seen.append((method, path, params, kwargs))
            return {"canTrade": True, "balances": []}

        monkeypatch.setattr(client, "_request_dict", transport)
        assert client.get_account(expected_scope=scope)["canTrade"]
        assert seen == [
            (
                "GET",
                "/api/v3/account",
                {},
                {"signed": True, "label": "account", "expected_scope": scope},
            )
        ]
    finally:
        client.session.close()


def test_missing_control_database_preserves_only_receipt_owned_reduction(tmp_path):
    gateway, req, client, control = setup(tmp_path)
    gateway.submit_first(req)
    control.ledger_path.rename(tmp_path / "retained-runtime.sqlite3")
    assert gateway.submit_second(IDENTITY, amount="1")["phase"] == "SECOND_TERMINAL"
    assert len(client.writes) == 2
    assert not control.ledger_path.exists()


def test_lost_close_lock_rejects_second_leg_without_recreating_it(tmp_path):
    gateway, req, client, control = setup(tmp_path)
    gateway.submit_first(req)
    control.close_lock_path.unlink()
    with pytest.raises((OSError, RuntimeError)):
        gateway.submit_second(IDENTITY, amount="1")
    assert len(client.writes) == 1
    assert not control.close_lock_path.exists()
    assert gateway.journal.pending(scope=SCOPE)[1]["phase"] == "FIRST_TERMINAL"


def test_close_timeout_matches_enrolled_control_contract(tmp_path):
    gateway, _, _, _ = setup(tmp_path)
    with pytest.raises(OpenIntentError, match="budget"):
        replace(gateway, close_lock_timeout_seconds=0.5)


def test_known_second_side_external_fee_rejects_before_first_submission(tmp_path):
    gateway, req, client, _ = setup(tmp_path)
    client.schedule = replace(
        SCHEDULE, discount=SpotCommissionDiscount(True, True, "BTC", Fraction(1, 4))
    )
    with pytest.raises(OpenIntentError, match="external fee"):
        gateway.submit_first(req)
    assert client.writes == []
    assert gateway.journal.pending(scope=SCOPE) is None


@pytest.mark.skipif(
    __import__("os").name != "nt", reason="Windows handle-lifetime observation"
)
def test_repeated_cancel_recovery_releases_owned_locks_and_native_handles(
    tmp_path, record_testsuite_property
):
    import ctypes
    from ctypes import wintypes
    from simple_ai_trading import runtime_control

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessHandleCount.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel.GetProcessHandleCount.restype = wintypes.BOOL

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
                "private",
            )
        ]

    kernel.K32GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    kernel.K32GetProcessMemoryInfo.restype = wintypes.BOOL

    def handles():
        count = wintypes.DWORD()
        assert kernel.GetProcessHandleCount(
            kernel.GetCurrentProcess(), ctypes.byref(count)
        )
        return count.value

    def private_bytes():
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        assert kernel.K32GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return counters.private

    gateway, original, client, _ = setup(tmp_path)
    observations = []
    for index in range(24):
        req = replace(original, operation_id=f"{index + 1:029x}")
        client.req, client.failure = req, KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            gateway.submit_first(req)
        client.failure = None
        gateway.recover(req.operation_id)
        gateway.submit_second(req.operation_id, amount="1")
        revision, _ = gateway.journal.pending(scope=SCOPE)
        gateway.journal.complete(
            req.operation_id, scope=SCOPE, expected_revision=revision
        )
        if index >= 3:
            observations.append(
                (handles(), len(runtime_control._LOCAL_LOCKS), private_bytes())
            )
    for label, column in (("handles", 0), ("mutexes", 1), ("private_bytes", 2)):
        for aggregate, value in (
            ("first", observations[0][column]),
            ("max", max(row[column] for row in observations)),
            ("last", observations[-1][column]),
        ):
            record_testsuite_property(f"roundtrip_lifecycle_{label}_{aggregate}", value)
    assert max(row[0] for row in observations) <= observations[0][0] + 2
    assert max(row[1] for row in observations) == observations[0][1]
    assert max(row[2] for row in observations) <= observations[0][2] + 8 * 1024**2
