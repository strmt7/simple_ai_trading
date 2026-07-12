"""Bounded-silence progress heartbeats for long synchronous operations."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
import math
import threading
import time


ProgressCallback = Callable[..., None]


@contextmanager
def progress_heartbeat(
    progress: ProgressCallback,
    *,
    phase: str,
    interval_seconds: float = 30.0,
    details: Mapping[str, object] | None = None,
) -> Iterator[None]:
    """Emit periodic progress while a synchronous operation owns the caller.

    Heartbeat callback failures are surfaced after the protected operation. An
    exception from the protected operation remains primary and is never masked by
    a concurrent heartbeat failure.
    """

    if not callable(progress):
        raise TypeError("progress must be callable")
    if not isinstance(phase, str) or not phase.strip():
        raise ValueError("heartbeat phase must be non-empty")
    interval = float(interval_seconds)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("heartbeat interval must be finite and positive")

    fixed_details = dict(details or {})
    reserved = {"state", "heartbeat_count", "elapsed_seconds"}
    if reserved & fixed_details.keys():
        raise ValueError("heartbeat details contain reserved fields")

    stop = threading.Event()
    callback_lock = threading.Lock()
    failures: list[BaseException] = []
    started = time.monotonic()

    def emit() -> None:
        heartbeat_count = 0
        while not stop.wait(interval):
            heartbeat_count += 1
            try:
                with callback_lock:
                    progress(
                        phase,
                        state="running",
                        heartbeat_count=heartbeat_count,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                        **fixed_details,
                    )
            except BaseException as exc:  # pragma: no branch - one terminal path
                failures.append(exc)
                stop.set()
                return

    thread = threading.Thread(
        target=emit,
        name=f"progress-heartbeat-{phase}",
        daemon=True,
    )
    thread.start()
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        stop.set()
        thread.join(timeout=5.0)
        if not body_failed:
            if thread.is_alive():
                raise RuntimeError(f"heartbeat thread did not stop for phase {phase}")
            if failures:
                raise RuntimeError(
                    f"heartbeat callback failed for phase {phase}"
                ) from failures[0]


__all__ = ["ProgressCallback", "progress_heartbeat"]
