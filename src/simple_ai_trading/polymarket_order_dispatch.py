"""Transient admission for one synchronous coordinator submission."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator


class PolymarketOrderNotSent(RuntimeError):
    """A callback denied this request; prior sends still require reconciliation."""


class OrderDispatchAdmission:
    """Keep denial evidence separate from uncertain transmission outcomes."""

    def __init__(self, check: Callable[[], None]) -> None:
        self._check = check
        self._transmission_started = False
        self.rejected_before_transmission = False

    def assert_current(self) -> None:
        try:
            self._check()
        except Exception:
            self.rejected_before_transmission = not self._transmission_started
            raise PolymarketOrderNotSent(
                "Order dispatch admission was revoked"
            ) from None

    def begin_transmission(self) -> None:
        # Set before sendall/stream: even a partial or failed send is uncertain.
        self._transmission_started = True
        self.rejected_before_transmission = False


_CURRENT: ContextVar[OrderDispatchAdmission | None] = ContextVar(
    "polymarket_order_dispatch_admission", default=None
)


def current_order_dispatch() -> OrderDispatchAdmission | None:
    return _CURRENT.get()


@contextmanager
def order_dispatch_scope(
    check: Callable[[], None],
) -> Iterator[OrderDispatchAdmission]:
    admission = OrderDispatchAdmission(check)
    token = _CURRENT.set(admission)
    try:
        yield admission
    finally:
        _CURRENT.reset(token)
