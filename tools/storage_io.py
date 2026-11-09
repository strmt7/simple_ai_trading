"""Explicit I/O pacing and current-process-only Windows background priority."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import math
import os
import time


class IoBudget:
    """Pace aggregate bytes, including source reads, target writes and readback."""

    def __init__(
        self, mib_per_second: float, *, clock=time.monotonic, sleep=time.sleep
    ):
        if not math.isfinite(mib_per_second) or not 1 <= mib_per_second <= 256:
            raise ValueError("I/O budget must be between 1 and 256 MiB/s")
        self.bytes_per_second = mib_per_second * 2**20
        self.clock, self.sleep = clock, sleep
        self.next_at = clock()

    def consume(self, size: int) -> None:
        """Do not accumulate idle-time credit that could cause a later burst."""
        if size < 0:
            raise ValueError("Negative byte charge")
        now = self.clock()
        self.next_at = max(now, self.next_at) + size / self.bytes_per_second
        delay = self.next_at - now
        if delay > 0:
            self.sleep(delay)


@contextmanager
def background_io():
    """Lower only this worker's CPU, memory and I/O scheduling priority."""
    if os.name != "nt":
        raise RuntimeError(
            "The migration background-priority contract requires Windows"
        )
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = wintypes.HANDLE
    api.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.SetPriorityClass.restype = wintypes.BOOL
    handle = api.GetCurrentProcess()
    if not api.SetPriorityClass(handle, 0x00100000):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not api.SetPriorityClass(handle, 0x00200000):
            raise ctypes.WinError(ctypes.get_last_error())
