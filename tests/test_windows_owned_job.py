from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from simple_ai_trading import windows_owned_job as owned
from simple_ai_trading.foundation_worker_client import (
    FoundationWorkerError,
    FoundationWorkerSupervisor,
)


@pytest.fixture(autouse=True)
def portable_mock_errors(monkeypatch):
    if os.name != "nt":
        error = [0]
        monkeypatch.setattr(
            ctypes,
            "set_last_error",
            lambda value: error.__setitem__(0, value),
            raising=False,
        )
        monkeypatch.setattr(ctypes, "get_last_error", lambda: error[0], raising=False)
        monkeypatch.setattr(
            ctypes,
            "WinError",
            lambda value: OSError(value, "mock Win32 error"),
            raising=False,
        )


class _Kernel:
    def __init__(self):
        self.closed = []
        self.rows = [(101, 22)]
        self.index = 0
        self.snapshot = 66
        self.thread = 88
        self.owner_pid = 101
        self.wait = owned.WAIT_TIMEOUT
        self.assigned = True
        self.resumed = 1
        self.membership = True
        self.open_process = 99

    def CreateToolhelp32Snapshot(self, _flags, _pid):
        return self.snapshot

    def _entry(self, pointer):
        if self.index >= len(self.rows):
            ctypes.set_last_error(owned.NO_MORE_FILES)
            return False
        pointer._obj.owner_pid, pointer._obj.thread_id = self.rows[self.index]
        self.index += 1
        return True

    def Thread32First(self, _snapshot, pointer):
        return self._entry(pointer)

    def Thread32Next(self, _snapshot, pointer):
        return self._entry(pointer)

    def OpenThread(self, _access, _inherit, _tid):
        return self.thread

    def GetProcessIdOfThread(self, _thread):
        return self.owner_pid

    def GetProcessId(self, _process):
        return 101

    def WaitForSingleObject(self, _process, _timeout):
        return self.wait

    def AssignProcessToJobObject(self, _job, _process):
        return self.assigned

    def ResumeThread(self, _thread):
        return self.resumed

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return True

    def OpenProcess(self, _access, _inherit, _pid):
        return self.open_process

    def IsProcessInJob(self, _process, _job, pointer):
        pointer._obj.value = self.membership
        return True


def job_and_kernel():
    kernel = _Kernel()
    job = owned.WindowsOwnedJob.__new__(owned.WindowsOwnedJob)
    job._api, job._handle = kernel, 55
    return job, kernel


def test_owned_handle_enrollment_resumes_only_verified_thread():
    job, kernel = job_and_kernel()
    job.enroll_and_resume(SimpleNamespace(pid=101, _handle=77))
    assert kernel.closed == [66, 88]
    assert job.contains_pid(101)
    job.close()
    job.close()
    assert kernel.closed == [66, 88, 99, 55]
    assert not job.contains_pid(101)


@pytest.mark.parametrize(
    "failure",
    [
        "snapshot",
        "no_thread",
        "multiple_threads",
        "missing_thread",
        "wrong_owner",
        "exited",
        "assign",
        "resume_error",
        "resume_count",
        "missing_handle",
        "closed_job",
    ],
)
def test_enrollment_fails_closed_before_unverified_execution(failure):
    job, kernel = job_and_kernel()
    process = SimpleNamespace(pid=101, _handle=77)
    if failure == "snapshot":
        kernel.snapshot = owned.INVALID_HANDLE
    elif failure == "no_thread":
        kernel.rows = []
    elif failure == "multiple_threads":
        kernel.rows += [(101, 33)]
    elif failure == "missing_thread":
        kernel.thread = None
    elif failure == "wrong_owner":
        kernel.owner_pid = 999
    elif failure == "exited":
        kernel.wait = 0
    elif failure == "assign":
        kernel.assigned = False
    elif failure == "resume_error":
        kernel.resumed = 0xFFFFFFFF
    elif failure == "resume_count":
        kernel.resumed = 2
    elif failure == "missing_handle":
        process = SimpleNamespace(pid=101)
    else:
        job._handle = None
    with pytest.raises(OSError):
        job.enroll_and_resume(process)


@pytest.mark.parametrize("pid", [0, -1, True, "101", 2**32 + 101])
def test_reported_pid_must_be_exact_valid_integer(pid):
    job, kernel = job_and_kernel()
    assert not job.contains_pid(pid)
    assert kernel.closed == []


def test_membership_rejects_foreign_or_inaccessible_process():
    job, kernel = job_and_kernel()
    kernel.membership = False
    assert not job.contains_pid(999)
    assert kernel.closed == [99]
    kernel.open_process = None
    assert not job.contains_pid(999)


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows handle containment")
def test_real_job_close_reaps_child_and_grandchild_but_not_sibling():
    # All three processes are test-owned. No user task is selected or signaled.
    sleeper = "import time; time.sleep(60)"
    sibling = subprocess.Popen(
        [sys._base_executable, "-c", sleeper], creationflags=subprocess.CREATE_NO_WINDOW
    )
    job = owned.WindowsOwnedJob()
    process = None
    child_handle = None
    try:
        leaf = "import json,os,time; print(json.dumps({'leaf':os.getpid()}),flush=True); time.sleep(60)"
        script = f"import json,os,subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',{leaf!r}]); print(json.dumps({{'pid':os.getpid(),'child':child.pid}}),flush=True); time.sleep(60)"
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW | owned.CREATE_SUSPENDED,
        )
        job.enroll_and_resume(process)
        assert process.stdout is not None
        messages = [json.loads(process.stdout.readline()) for _ in range(2)]
        payload = next(message for message in messages if "pid" in message)
        leaf_pid = next(message["leaf"] for message in messages if "leaf" in message)
        assert job.contains_pid(payload["pid"]) and job.contains_pid(payload["child"])
        assert not job.contains_pid(sibling.pid)
        assert job.contains_pid(leaf_pid)
        child_handle = job._api.OpenProcess(
            0x00100000 | owned.PROCESS_QUERY, False, leaf_pid
        )
        assert child_handle
        job.close()
        process.wait(timeout=5)
        assert process.poll() is not None
        assert job._api.WaitForSingleObject(child_handle, 5_000) == 0
        assert sibling.poll() is None
    finally:
        job.close()
        if child_handle:
            job._api.CloseHandle(child_handle)
        if process is not None:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
        sibling.terminate()
        sibling.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows handle containment")
def test_supervisor_rejects_forged_runtime_pid_and_preserves_sibling(monkeypatch):
    sibling = subprocess.Popen(
        [sys._base_executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
        startup_timeout_seconds=5,
    )
    script = f"import json,time; print(json.dumps({{'type':'ready','worker_pid':{sibling.pid},'report':{{}}}}),flush=True); time.sleep(60)"
    monkeypatch.setattr(
        supervisor, "_command", lambda: [sys.executable, "-u", "-c", script]
    )
    try:
        with pytest.raises(FoundationWorkerError, match="owned process boundary"):
            supervisor.start()
        assert supervisor.process is None and supervisor._job is None
        assert sibling.poll() is None
    finally:
        supervisor.stop()
        sibling.terminate()
        sibling.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows handle containment")
def test_startup_timeout_reaps_task_owned_tree(monkeypatch):
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
        startup_timeout_seconds=1,
    )
    monkeypatch.setattr(
        supervisor,
        "_command",
        lambda: [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    with pytest.raises(FoundationWorkerError, match="startup timed out"):
        supervisor.start()
    assert supervisor.process is None and supervisor._job is None


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows worker identity")
def test_supervisor_accepts_own_runtime_and_reaps_it_on_stop(monkeypatch):
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
        startup_timeout_seconds=5,
    )
    script = "import json,os,time; print(json.dumps({'type':'ready','worker_pid':os.getpid(),'report':{'model':'stdlib-fixture'}}),flush=True); time.sleep(60)"
    monkeypatch.setattr(
        supervisor, "_command", lambda: [sys.executable, "-u", "-c", script]
    )
    try:
        assert supervisor.start() == {"model": "stdlib-fixture"}
        assert supervisor._job is not None and supervisor._job.contains_pid(
            supervisor.pid
        )
        process = supervisor.process
        assert process is not None
        supervisor.stop()
        assert process.poll() is not None
        assert supervisor.process is None and supervisor._job is None
    finally:
        supervisor.stop()


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows parent-death containment")
def test_controller_crash_closes_last_handle_and_reaps_its_worker():
    # Direct base interpreter is sufficient for this stdlib-only control fixture;
    # production model workers still use their original venv launcher.
    script = """
import json, os, subprocess, sys
from simple_ai_trading.windows_owned_job import WindowsOwnedJob, CREATE_SUSPENDED
job = WindowsOwnedJob()
worker = subprocess.Popen([sys.executable, '-u', '-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(60)'], stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW | CREATE_SUSPENDED)
job.enroll_and_resume(worker)
pid = int(worker.stdout.readline())
assert job.contains_pid(pid)
print(json.dumps({'worker_pid':pid}),flush=True)
assert sys.stdin.readline().strip() == 'crash'
os._exit(17)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    controller = subprocess.Popen(
        [sys._base_executable, "-u", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    probe = owned.WindowsOwnedJob()
    worker_handle = None
    try:
        assert controller.stdout is not None and controller.stdin is not None
        pid = json.loads(controller.stdout.readline())["worker_pid"]
        worker_handle = probe._api.OpenProcess(
            0x00100000 | owned.PROCESS_QUERY, False, pid
        )
        assert worker_handle
        assert probe._api.WaitForSingleObject(worker_handle, 0) == owned.WAIT_TIMEOUT
        controller.stdin.write("crash\n")
        controller.stdin.flush()
        assert controller.wait(timeout=5) == 17
        assert probe._api.WaitForSingleObject(worker_handle, 5_000) == 0
    finally:
        if controller.poll() is None:
            controller.terminate()
        controller.wait(timeout=5)
        for stream in (controller.stdin, controller.stdout):
            if stream is not None:
                stream.close()
        if worker_handle:
            probe._api.CloseHandle(worker_handle)
        probe.close()
