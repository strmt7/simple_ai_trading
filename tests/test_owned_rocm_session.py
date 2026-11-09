from __future__ import annotations

import os
import subprocess

import pytest

from tools import run_ollama_rocm_session as module


def test_explicit_provider_configuration_does_not_change_parent_environment(
    tmp_path, monkeypatch
):
    (tmp_path / "bin/rocblas/library").mkdir(parents=True)
    monkeypatch.setenv("OLLAMA_VULKAN", "1")
    calls = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, address):
            calls.append(address)

    monkeypatch.setattr(module.socket, "socket", lambda *a: Probe())
    env = module.session_environment(tmp_path, "127.0.0.1:1234")
    assert calls == [("127.0.0.1", 1234)]
    assert env["ROCM_PATH"] == str(tmp_path)
    assert env["OLLAMA_LLM_LIBRARY"] == "rocm"
    assert "OLLAMA_VULKAN" not in env
    assert os.environ["OLLAMA_VULKAN"] == "1"
    assert (
        "HIP_VISIBLE_DEVICES" not in env
        or env["HIP_VISIBLE_DEVICES"] == os.environ["HIP_VISIBLE_DEVICES"]
    )


def test_occupied_provider_endpoint_is_never_replaced(tmp_path, monkeypatch):
    (tmp_path / "bin/rocblas/library").mkdir(parents=True)
    state = {"closed": False}

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

        def bind(self, address):
            raise OSError("occupied by an unrelated provider")

    monkeypatch.setattr(module.socket, "socket", lambda *a: Probe())
    with pytest.raises(OSError):
        module.session_environment(tmp_path, "127.0.0.1:1234")
    assert state["closed"]


@pytest.mark.parametrize(
    "host",
    [
        "",
        "remote.invalid:1234",
        "0.0.0.0:1234",
        "https://127.0.0.1:1234",
        "127.0.0.1",
        "127.0.0.1:1234/path",
    ],
)
def test_ambiguous_or_external_endpoints_reject(tmp_path, host):
    (tmp_path / "bin/rocblas/library").mkdir(parents=True)
    with pytest.raises(ValueError):
        module.session_environment(tmp_path, host)


@pytest.mark.skipif(os.name != "nt", reason="Windows owned provider lifecycle")
@pytest.mark.parametrize("outcome", ["expire", "exit", "enroll", "spawn"])
def test_session_reaps_only_its_original_owned_process(tmp_path, monkeypatch, outcome):
    from simple_ai_trading import windows_owned_job

    executable = tmp_path / "provider.exe"
    executable.touch()
    state = {"closed": False, "enrolled": False, "killed": False}

    class Process:
        returncode = None

        def wait(self, timeout):
            if outcome == "expire" and not state["closed"]:
                raise subprocess.TimeoutExpired("owned", timeout)
            self.returncode = 2 if outcome == "exit" else -9
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            state["killed"] = True
            self.returncode = -9

    process = Process()

    class Job:
        def enroll_and_resume(self, child):
            assert child is process
            if outcome == "enroll":
                raise OSError("containment unavailable")
            state["enrolled"] = True

        def close(self):
            state["closed"] = True
            if state["enrolled"] and process.returncode is None:
                process.returncode = -15

    def start(command, **kwargs):
        assert command == [str(executable), "serve"]
        assert kwargs["creationflags"] & windows_owned_job.CREATE_SUSPENDED
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
        if outcome == "spawn":
            raise OSError("launch refused")
        return process

    monkeypatch.setattr(windows_owned_job, "WindowsOwnedJob", Job)
    monkeypatch.setattr(module.subprocess, "Popen", start)
    if outcome == "expire":
        result = module.run_session(executable, {}, 1)
        assert result["session_finished"] and not result["gpu_residency_verified"]
    else:
        with pytest.raises((OSError, RuntimeError)):
            module.run_session(executable, {}, 1)
    assert state["closed"]
    assert state["killed"] == (outcome == "enroll")
