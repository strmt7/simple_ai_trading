"""Scoped, write-ahead dispatch for an explicitly allocated Spot diagnostic.

This gateway neither acquires a lease nor enrolls storage. The caller supplies
those authorities. It creates no worker, retries no write, and never interprets
two completed orders as restored inventory, flat risk or profitable execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import time

from .api import BinanceClient
from .binance_execution_scope import BinanceExecutionScope
from .binance_execution_storage import BinanceExecutionStorage
from .binance_open_intents import OpenIntentError
from .binance_order_responses import SpotQuoteOrderBinding
from .binance_runtime_control import BinanceRuntimeControl, RuntimeLeaseInterlock
from .binance_spot_commissions import SpotCommissionSchedule
from .binance_spot_order_trades import observe_spot_order_trades
from .binance_spot_roundtrip_intents import (
    SpotRoundtripJournal,
    SpotRoundtripRequest,
    _movements,
    _validate_record,
)
from .binance_terminal_fills import _decimal


def _available(account: object, required: Mapping[str, Fraction]) -> None:
    if not isinstance(account, Mapping) or account.get("canTrade") is not True:
        raise OpenIntentError("roundtrip account trading permission is unqualified")
    rows = account.get("balances")
    if not isinstance(rows, list) or len(rows) > 10_000:
        raise OpenIntentError("roundtrip account balances are missing or unbounded")
    free = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise OpenIntentError("roundtrip account balance is malformed")
        asset = row.get("asset")
        if not isinstance(asset, str) or not asset or asset in free:
            raise OpenIntentError("roundtrip account asset is missing or duplicated")
        free[asset] = Fraction(_decimal(row.get("free")))
        _decimal(row.get("locked"))
    if any(
        amount < 0 or free.get(asset, Fraction(0)) < amount
        for asset, amount in required.items()
    ):
        raise OpenIntentError("roundtrip allocation exceeds current free balances")


def _request(record: dict) -> SpotRoundtripRequest:
    # Validate the full record, including retained cash and commission movements.
    return _validate_record(record)


@dataclass(frozen=True)
class SpotRoundtripExecution:
    """Borrow caller-owned client/control objects; retain no connection or worker."""

    client: BinanceClient
    journal: SpotRoundtripJournal
    interlock: RuntimeLeaseInterlock
    maximum_preflight_age_ms: int = 5_000
    close_lock_timeout_seconds: float = 1.0
    execution_enrollment_path: Path | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.journal, SpotRoundtripJournal)
            or not isinstance(self.interlock, RuntimeLeaseInterlock)
            or not isinstance(self.interlock.control, BinanceRuntimeControl)
            or type(self.maximum_preflight_age_ms) is not int
            or not 1 <= self.maximum_preflight_age_ms <= 30_000
            or isinstance(self.close_lock_timeout_seconds, bool)
            or not isinstance(self.close_lock_timeout_seconds, (int, float))
            or not 1 <= self.close_lock_timeout_seconds <= 30
        ):
            raise OpenIntentError("roundtrip runtime or preflight budget is invalid")

    def _scope(self) -> BinanceExecutionScope:
        scope = self.client.execution_scope()
        if not isinstance(scope, BinanceExecutionScope) or scope.market_type != "spot":
            raise OpenIntentError(
                "roundtrip dispatch requires a testnet/Demo Spot scope"
            )
        return scope

    def _storage(self) -> BinanceExecutionStorage:
        return BinanceExecutionStorage(
            self.journal.journal, self.interlock.control, self.execution_enrollment_path
        )

    def _preflight(
        self,
        request: SpotRoundtripRequest,
        *,
        scope: BinanceExecutionScope,
        first: bool,
        required: Mapping[str, Fraction],
    ) -> int:
        started = time.monotonic_ns()
        _available(self.client.get_account(expected_scope=scope), required)
        schedule = self.client.get_spot_commission_schedule(
            request.symbol, expected_scope=scope
        )
        if (
            not isinstance(schedule, SpotCommissionSchedule)
            or schedule.symbol != request.symbol
        ):
            raise OpenIntentError("roundtrip commission scope is unqualified")
        side = "BUY" if (request.mode == "buy-sell") == first else "SELL"
        for planned_side in ("BUY", "SELL") if first else (side,):
            received = request.base if planned_side == "BUY" else request.quote
            if schedule.possible_fee_assets(side=planned_side) != {received}:
                # An explicit payment-asset grant is not an FX conversion bound
                # or an exchange-enforced fee cap. It cannot admit dispatch.
                raise OpenIntentError(
                    "roundtrip external fee routing needs native FX/cost admission"
                )
            if (
                schedule.equivalent_rate(
                    side=planned_side, role="taker", apply_discount=False
                ).total
                >= 1
            ):
                raise OpenIntentError(
                    "roundtrip received-asset fee consumes its proceeds"
                )
        self._fresh(started, scope)
        return started

    def _fresh(self, started: int, scope: BinanceExecutionScope) -> None:
        elapsed = time.monotonic_ns() - started
        if (
            elapsed < 0
            or elapsed > self.maximum_preflight_age_ms * 1_000_000
            or self._scope() != scope
        ):
            raise OpenIntentError(
                "roundtrip preflight expired or its execution scope changed"
            )

    def _dispatch(
        self, binding, *, scope: BinanceExecutionScope
    ) -> Mapping[str, object]:
        if isinstance(binding, SpotQuoteOrderBinding):
            return self.client.place_spot_quote_buy(
                binding.symbol,
                binding.quote_quantity,
                dry_run=False,
                client_order_id=binding.client_order_id,
                expected_scope=scope,
            )
        # The request constructor already proves the existing float adapter's
        # eight-decimal serialization equals this exact base quantity.
        return self.client.place_order(
            binding.symbol,
            "SELL",
            float(binding.quantity),
            dry_run=False,
            client_order_id=binding.client_order_id,
            expected_scope=scope,
        )

    def submit_first(self, request: SpotRoundtripRequest) -> dict:
        """Commit FIRST_UNKNOWN before the single native first-leg write."""
        if not isinstance(request, SpotRoundtripRequest):
            raise OpenIntentError("roundtrip requires an explicit validated allocation")
        self.interlock.assert_opening_allowed()
        scope = self._scope()
        storage = self._storage()
        storage.verify(scope=scope)
        started = self._preflight(
            request, scope=scope, first=True, required=request.allocations
        )
        with self.interlock.submission_guard():
            self._fresh(started, scope)
            storage.verify(scope=scope)
            self.journal.prepare(request, scope=scope, create=False)
            revision, _ = self._pending(scope, request.operation_id)
            self._fresh(started, scope)
            self.interlock.assert_opening_allowed()
            storage.verify(scope=scope)
            order = self._dispatch(request.binding(first=True), scope=scope)
            self.journal.record_order(
                request.operation_id,
                order,
                first=True,
                scope=scope,
                expected_revision=revision,
            )
        return self._pending(scope, request.operation_id)[1]

    def _pending(
        self, scope: BinanceExecutionScope, operation_id: str
    ) -> tuple[int, dict]:
        pending = self.journal.pending(scope=scope)
        if pending is None or pending[1]["request"]["operation_id"] != operation_id:
            raise OpenIntentError("roundtrip has no matching active obligation")
        return pending

    def submit_second(self, operation_id: str, *, amount: str) -> dict:
        """Commit SECOND_UNKNOWN once; only receipt-owned SELL bypasses opening Stop.

        A sell-buy BUY is a new acquisition, never a privileged owned close.
        The journal checks the supplied principal against first-leg net receipts.
        No wallet-based resizing or automatic completion occurs here.
        """
        scope = self._scope()
        revision, record = self._pending(scope, operation_id)
        request = _request(record)
        if record["phase"] != "FIRST_TERMINAL" or record["first"]["status"] != "FILLED":
            raise OpenIntentError("roundtrip first leg is not fully reconciled")
        binding = request.binding(first=False, amount=amount)
        _validate_record({**record, "phase": "SECOND_UNKNOWN", "second_amount": amount})
        opening = isinstance(binding, SpotQuoteOrderBinding)
        if opening:
            self.interlock.assert_opening_allowed()
            self._storage().verify(scope=scope)
        delta = _movements(request, record["first"], first=True)
        remaining = {
            asset: allocated + delta[asset]
            for asset, allocated in request.allocations.items()
        }
        started = self._preflight(request, scope=scope, first=False, required=remaining)
        guard = (
            self.interlock.submission_guard()
            if opening
            else self.interlock.control.close_guard(
                timeout_seconds=self.close_lock_timeout_seconds
            )
        )
        with guard:
            self._fresh(started, scope)
            if opening:
                self._storage().verify(scope=scope)
            self.journal.prepare_second(
                operation_id, amount, scope=scope, expected_revision=revision
            )
            revision, _ = self._pending(scope, operation_id)
            self._fresh(started, scope)
            if opening:
                self.interlock.assert_opening_allowed()
                self._storage().verify(scope=scope)
            order = self._dispatch(binding, scope=scope)
            self.journal.record_order(
                operation_id,
                order,
                first=False,
                scope=scope,
                expected_revision=revision,
            )
        return self._pending(scope, operation_id)[1]

    def recover(self, operation_id: str) -> dict:
        """Observe the exact UNKNOWN leg once; never submit, complete or rearm."""
        scope = self._scope()
        revision, record = self._pending(scope, operation_id)
        request = _request(record)
        if record["phase"] not in {"FIRST_UNKNOWN", "SECOND_UNKNOWN"}:
            return record
        first = record["phase"] == "FIRST_UNKNOWN"
        binding = request.binding(first=first, amount=record["second_amount"])
        order = self.client.get_order(
            request.symbol,
            orig_client_order_id=binding.client_order_id,
            expected_scope=scope,
            expected_order_binding=binding,
        )
        order = observe_spot_order_trades(
            self.client, order, binding=binding, scope=scope
        )
        self.journal.record_order(
            operation_id, order, first=first, scope=scope, expected_revision=revision
        )
        return self._pending(scope, operation_id)[1]
