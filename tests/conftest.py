from __future__ import annotations

import sys
from decimal import Context, localcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def isolate_decimal_context():
    """Keep legacy research imports from changing another test's arithmetic."""
    with localcontext(Context()):
        yield


def enroll_synthetic_binance_runtime(
    store,
    scope,
    *,
    execution=True,
    start=True,
    capital_policy=None,
    opening_policy_quote=None,
):
    """Construct explicit authority for an account with no real venue obligations."""
    from simple_ai_trading.binance_execution_storage import BinanceExecutionStorage
    from simple_ai_trading.binance_runtime_control import (
        BinanceRuntimeControl,
        RuntimeLeaseInterlock,
    )

    control = BinanceRuntimeControl(store.root / "binance_runtime_control.sqlite3")
    control.provision_storage()
    if execution:
        binding = BinanceExecutionStorage(store.opening_intents, control)
        existing = binding.inspect()
        binding.provision(
            scope=scope,
            expected_snapshot_sha256=existing.get("journal_snapshot_sha256"),
        )
    if opening_policy_quote is not None:
        from simple_ai_trading.types import StrategyConfig

        if capital_policy is not None:
            raise ValueError("synthetic opening policy must be configured once")
        strategy = StrategyConfig()
        capital_policy = {
            "quote_asset": opening_policy_quote,
            "reference_cash_quote": "1000",
            "maximum_daily_loss_fraction": str(strategy.max_daily_loss_pct),
            "maximum_lifetime_loss_fraction": str(strategy.max_session_loss_pct),
            "maximum_settled_peak_drawdown_fraction": str(strategy.max_drawdown_limit),
            "maximum_consecutive_losses": strategy.max_consecutive_losses,
        }
    if capital_policy is not None:
        from simple_ai_trading.binance_capital_policy import (
            BinanceCapitalPolicy,
            BinanceCapitalPolicyStore,
        )

        policies = BinanceCapitalPolicyStore(store, control=control)
        observed = policies.inspect()
        policy = BinanceCapitalPolicy(
            schema_version="binance-capital-policy-v1",
            execution_scope_sha256=observed["execution_scope_sha256"],
            execution_store_id=observed["execution_store_id"],
            **capital_policy,
        )
        policies.activate(policy, expected_head_sha256=observed["head_sha256"])
    lease = ""
    if start:
        assert control.complete_stale_stop(exposure_closed=True)
        lease = control.acquire()
    return RuntimeLeaseInterlock(control, lease)


@pytest.fixture
def enrolled_binance_runtime(monkeypatch):
    from simple_ai_trading.binance_runtime_control import BINANCE_RUNTIME_LEASE_ENV

    def enroll(
        store,
        scope,
        *,
        execution=True,
        start=True,
        capital_policy=None,
        opening_policy_quote=None,
    ):
        authority = enroll_synthetic_binance_runtime(
            store,
            scope,
            execution=execution,
            start=start,
            capital_policy=capital_policy,
            opening_policy_quote=opening_policy_quote,
        )
        if start:
            monkeypatch.setenv(BINANCE_RUNTIME_LEASE_ENV, authority.lease_id)
        return authority

    return enroll


@pytest.fixture
def process_resource_sample():
    """Observe only this test process; never inspect or alter another task."""
    import ctypes
    import os
    from ctypes import wintypes
    from simple_ai_trading import runtime_control

    if os.name != "nt":
        pytest.skip("Windows process resource counters")
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

    def sample():
        process = kernel.GetCurrentProcess()
        handles = wintypes.DWORD()
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not kernel.GetProcessHandleCount(
            process, ctypes.byref(handles)
        ) or not kernel.K32GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return handles.value, len(runtime_control._LOCAL_LOCKS), counters.private

    return sample
