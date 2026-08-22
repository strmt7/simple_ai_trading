from __future__ import annotations

import io
import subprocess

import pytest

from simple_ai_trading.foundation_worker_client import (
    FoundationWorkerError,
    FoundationWorkerSupervisor,
)


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


def test_stop_terminates_distinct_runtime_when_windows_launcher_hangs(
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
        FoundationWorkerSupervisor,
        "_terminate_runtime_pid",
        staticmethod(terminated_runtime_pids.append),
    )

    supervisor.stop()

    assert terminated_runtime_pids == [202]
    assert process.terminated is True
    assert process.killed is False
    assert process.stdin.closed is True
    assert supervisor.process is None
    assert supervisor.pid is None
