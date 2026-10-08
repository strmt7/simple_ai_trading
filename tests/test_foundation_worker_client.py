from __future__ import annotations

import io
import subprocess
from types import SimpleNamespace

import pytest

from simple_ai_trading.foundation_worker_client import (
    FoundationWorkerError,
    FoundationWorkerSupervisor,
)
from simple_ai_trading import foundation_worker_client as worker_client


class _HungLauncher:
    pid = 101
    stdout = None
    stderr = None

    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.wait_calls = 0
        self.terminated = False
        self.killed = False

    def poll(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        assert timeout == 3.0
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired("foundation-worker", timeout)
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def test_start_reaps_launcher_when_required_process_pipe_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
    )
    process = _HungLauncher()
    # This fixture exercises pipe cleanup, not a real Windows process handle.
    monkeypatch.setattr(
        "simple_ai_trading.foundation_worker_client.WindowsOwnedJob", _FakeJob
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(
        FoundationWorkerError, match="process pipes are unavailable"
    ) as error:
        supervisor.start()

    assert error.value.restartable is False
    assert process.stdin.closed is True
    assert process.wait_calls == 2
    assert process.terminated is True
    assert process.killed is False
    assert supervisor.process is None
    assert supervisor.pid is None


def test_stop_never_signals_an_unverified_runtime_pid_when_launcher_hangs(
    monkeypatch,
) -> None:
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
    )
    process = _HungLauncher()
    supervisor.process = process  # type: ignore[assignment]
    supervisor._runtime_pid = 202
    terminated_runtime_pids: list[int | None] = []
    monkeypatch.setattr(
        "os.kill", lambda pid, _signal: terminated_runtime_pids.append(pid)
    )

    supervisor.stop()

    assert terminated_runtime_pids == []
    assert process.terminated is True
    assert process.killed is False
    assert process.stdin.closed is True
    assert supervisor.process is None
    assert supervisor.pid is None


class _FakeJob:
    def __init__(self):
        self.closed = False

    def enroll_and_resume(self, process):
        pass

    def contains_pid(self, pid):
        return pid == 202

    def close(self):
        self.closed = True


def test_unsupported_windows_containment_prevents_process_launch(monkeypatch):
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
    )
    calls = []

    def unsupported():
        raise OSError("unsupported job")

    monkeypatch.setattr(worker_client, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(worker_client, "WindowsOwnedJob", unsupported)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(OSError, match="unsupported job"):
        supervisor.start()
    assert calls == [] and supervisor.process is None and supervisor._job is None


def test_assignment_failure_reaps_suspended_owned_process(monkeypatch):
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
    )
    process = _HungLauncher()
    job = _FakeJob()

    def reject(_process):
        raise OSError("job assignment failed")

    monkeypatch.setattr(worker_client, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(job, "enroll_and_resume", reject)
    monkeypatch.setattr(worker_client, "WindowsOwnedJob", lambda: job)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    with pytest.raises(OSError, match="job assignment failed"):
        supervisor.start()
    assert job.closed and process.terminated
    assert supervisor.process is None and supervisor._job is None


@pytest.mark.parametrize("pid", [True, "202", None])
def test_invalid_ready_runtime_id_cleans_up_job(monkeypatch, pid):
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
    )
    process = _HungLauncher()
    process.stdout, process.stderr = io.StringIO(), io.StringIO()
    job = _FakeJob()
    monkeypatch.setattr(worker_client, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(worker_client, "WindowsOwnedJob", lambda: job)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        supervisor,
        "_message",
        lambda *args, **kwargs: {"type": "ready", "report": {}, "worker_pid": pid},
    )
    with pytest.raises(FoundationWorkerError, match="runtime PID"):
        supervisor.start()
    assert job.closed and process.terminated
    assert supervisor.process is None and supervisor._job is None


def test_non_windows_requires_original_child_identity_without_pid_signals(monkeypatch):
    supervisor = FoundationWorkerSupervisor(
        model_size="small",
        backend="cpu",
        source_cache_root=None,
        require_accelerator=False,
    )
    process = _HungLauncher()
    process.stdout, process.stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(worker_client, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        supervisor,
        "_message",
        lambda *args, **kwargs: {"type": "ready", "report": {}, "worker_pid": 101},
    )
    assert supervisor.start() == {}
    supervisor.stop()
    assert process.terminated and supervisor._job is None
