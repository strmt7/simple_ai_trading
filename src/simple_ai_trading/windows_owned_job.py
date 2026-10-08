"""Windows child containment anchored to owned OS handles, never reported PIDs."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import subprocess


# Win32 ABI constants, verified against Windows SDK 10.0.26100.0 headers.
CREATE_SUSPENDED = 0x00000004
KILL_ON_JOB_CLOSE = 0x00002000
SNAP_THREADS = 0x00000004
THREAD_ACCESS = 0x00000002 | 0x00000800
PROCESS_QUERY = 0x00001000
WAIT_TIMEOUT = 258
NO_MORE_FILES = 18
INVALID_HANDLE = ctypes.c_void_p(-1).value


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_user_time", ctypes.c_longlong),
        ("job_user_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD),
        ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "read_operations",
            "write_operations",
            "other_operations",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.DWORD),
        ("usage", wintypes.DWORD),
        ("thread_id", wintypes.DWORD),
        ("owner_pid", wintypes.DWORD),
        ("base_priority", wintypes.LONG),
        ("delta_priority", wintypes.LONG),
        ("flags", wintypes.DWORD),
    ]


class WindowsOwnedJob:
    """Create a non-inheritable, no-breakaway kill-on-close child job.

    A CPython Popen child must be created suspended, assigned, then resumed.
    Snapshot enumeration is read-only; only the owned suspended child's verified
    primary thread is resumed. Unsupported containment rejects startup, with no
    uncontained fallback. A job is containment, not a credential/security sandbox.
    """

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows owned jobs require Windows")
        self._api = ctypes.WinDLL("kernel32", use_last_error=True)
        self._handle: int | None = None
        handle, dword, boolean = wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], handle),
            "SetInformationJobObject": (
                [handle, ctypes.c_int, ctypes.c_void_p, dword],
                boolean,
            ),
            "AssignProcessToJobObject": ([handle, handle], boolean),
            "IsProcessInJob": ([handle, handle, ctypes.POINTER(boolean)], boolean),
            "CloseHandle": ([handle], boolean),
            "CreateToolhelp32Snapshot": ([dword, dword], handle),
            "Thread32First": ([handle, ctypes.POINTER(_ThreadEntry)], boolean),
            "Thread32Next": ([handle, ctypes.POINTER(_ThreadEntry)], boolean),
            "OpenThread": ([dword, boolean, dword], handle),
            "GetProcessIdOfThread": ([handle], dword),
            "GetProcessId": ([handle], dword),
            "WaitForSingleObject": ([handle, dword], dword),
            "ResumeThread": ([handle], dword),
            "OpenProcess": ([dword, boolean, dword], handle),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self._api, name)
            function.argtypes, function.restype = arguments, result
        created = self._api.CreateJobObjectW(None, None)
        if not created:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = int(created)
        limits = _ExtendedLimits()
        limits.basic.flags = KILL_ON_JOB_CLOSE
        if not self._api.SetInformationJobObject(
            self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def _primary_thread(self, process_handle: int, pid: int) -> int:
        snapshot = self._api.CreateToolhelp32Snapshot(SNAP_THREADS, 0)
        if snapshot in (None, INVALID_HANDLE):
            raise ctypes.WinError(ctypes.get_last_error())
        entries: list[int] = []
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            present = self._api.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.owner_pid == pid:
                    entries.append(int(entry.thread_id))
                entry.size = ctypes.sizeof(entry)
                present = self._api.Thread32Next(snapshot, ctypes.byref(entry))
            if ctypes.get_last_error() != NO_MORE_FILES:
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._api.CloseHandle(snapshot)
        if len(entries) != 1:
            raise OSError("owned suspended child has no unique primary thread")
        thread = self._api.OpenThread(THREAD_ACCESS, False, entries[0])
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        if (
            self._api.GetProcessIdOfThread(thread) != pid
            or self._api.WaitForSingleObject(process_handle, 0) != WAIT_TIMEOUT
        ):
            self._api.CloseHandle(thread)
            raise OSError("owned suspended child thread identity changed")
        return int(thread)

    def enroll_and_resume(self, process: subprocess.Popen[str]) -> None:
        """Assign the original Popen process handle before its first instruction."""
        process_handle = getattr(process, "_handle", None)
        if (
            self._handle is None
            or process_handle is None
            or self._api.GetProcessId(int(process_handle)) != process.pid
            or self._api.WaitForSingleObject(int(process_handle), 0) != WAIT_TIMEOUT
        ):
            raise OSError("owned Windows process handle is unavailable or exited")
        thread = self._primary_thread(int(process_handle), process.pid)
        try:
            if not self._api.AssignProcessToJobObject(
                self._handle, int(process_handle)
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            previous_count = self._api.ResumeThread(thread)
            if previous_count == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            if previous_count != 1:
                raise OSError("owned primary thread suspend count is inconsistent")
        finally:
            self._api.CloseHandle(thread)

    def contains_pid(self, pid: int) -> bool:
        """Read-only job membership check; a reported PID grants no control."""
        if type(pid) is not int or not 0 < pid < 2**32 or self._handle is None:
            return False
        process = self._api.OpenProcess(PROCESS_QUERY, False, pid)
        if not process:
            return False
        try:
            belongs = wintypes.BOOL()
            return bool(
                self._api.IsProcessInJob(process, self._handle, ctypes.byref(belongs))
                and belongs.value
            )
        finally:
            self._api.CloseHandle(process)

    def close(self) -> None:
        """Close only this non-inherited job handle, terminating its own tree."""
        if self._handle is not None:
            if not self._api.CloseHandle(self._handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handle = None
