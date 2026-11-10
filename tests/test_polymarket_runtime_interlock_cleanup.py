"""Interlock failures must release both the native handle and local mutex."""

from __future__ import annotations

import os
from pathlib import Path
from queue import Queue
from threading import Thread

import pytest

from simple_ai_trading import polymarket_runtime_control as runtime


class ObservedFile:
    def __init__(self, raw, *, setup_failure=None):
        self.raw = raw
        self.setup_failure = setup_failure
        self.fail_seek = False
        self.fail_close = False
        self.close_attempts = 0

    def seek(self, *args):
        if self.fail_seek:
            raise OSError("injected unlock seek failure")
        return self.raw.seek(*args)

    def write(self, value):
        if self.setup_failure == "write":
            raise OSError("injected setup write failure")
        return self.raw.write(value)

    def flush(self):
        if self.setup_failure == "flush":
            raise OSError("injected setup flush failure")
        return self.raw.flush()

    def fileno(self):
        return self.raw.fileno()

    def close(self):
        self.close_attempts += 1
        self.raw.close()
        if self.fail_close:
            raise OSError("injected close failure")


def observe_first_open(monkeypatch, path, *, setup_failure=None):
    original = Path.open
    files = []

    def open_path(candidate, *args, **kwargs):
        raw = original(candidate, *args, **kwargs)
        if candidate == path and not files:
            wrapper = ObservedFile(raw, setup_failure=setup_failure)
            files.append(wrapper)
            return wrapper
        return raw

    monkeypatch.setattr(Path, "open", open_path)
    return files


def peer_enters(path):
    outcomes = Queue()

    def probe():
        try:
            with runtime._CrossProcessFileLock(path, timeout_seconds=0.2):
                outcomes.put(True)
        except runtime.PolymarketLiveBlocked:
            outcomes.put(False)
        except BaseException as error:
            outcomes.put(error)

    thread = Thread(target=probe)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive(), "bounded interlock probe did not finish"
    result = outcomes.get_nowait()
    if isinstance(result, BaseException):
        raise result
    return result


def cleanup_fixture(lock, files):
    # Also clean a deliberately broken previous implementation during red proof.
    for file in files:
        if not file.raw.closed:
            file.raw.close()
    try:
        lock._local.release()
    except RuntimeError:
        pass


@pytest.mark.parametrize("failure", ["write", "flush", "fsync"])
def test_interlock_setup_failure_closes_handle_and_releases_mutex(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "entry.lock"
    files = observe_first_open(monkeypatch, path, setup_failure=failure)
    original_fsync = os.fsync

    def fsync(fd):
        if failure == "fsync" and files and not files[0].raw.closed:
            raise OSError("injected setup fsync failure")
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    lock = runtime._CrossProcessFileLock(path, timeout_seconds=0.2)
    try:
        with pytest.raises(OSError, match="injected setup"):
            with lock:
                pytest.fail("setup failure must reject entry")
        assert files[0].raw.closed
        assert files[0].close_attempts == 1
        assert peer_enters(path)
    finally:
        cleanup_fixture(lock, files)


@pytest.mark.parametrize("close_failure", [False, True])
def test_interlock_unlock_failure_still_closes_native_handle_and_releases_mutex(
    tmp_path, monkeypatch, close_failure
):
    path = tmp_path / "dispatch.lock"
    files = observe_first_open(monkeypatch, path)
    lock = runtime._CrossProcessFileLock(path, timeout_seconds=0.2)
    try:
        with pytest.raises(OSError, match="injected") as failure:
            with lock:
                files[0].fail_seek = True
                files[0].fail_close = close_failure
        assert files[0].raw.closed
        assert files[0].close_attempts == 1
        assert failure.value.__traceback__ is not None
        # This opens and locks the same real file in another thread, while the
        # failure traceback is still retained by pytest's exception context.
        assert peer_enters(path)
    finally:
        cleanup_fixture(lock, files)


def test_interlock_cancelled_acquisition_releases_every_acquired_resource(
    tmp_path, monkeypatch
):
    path = tmp_path / "cancel.lock"
    files = observe_first_open(monkeypatch, path)

    def blocked(*_args):
        raise BlockingIOError("injected contention")

    def cancel(_seconds):
        raise KeyboardInterrupt("injected cancellation")

    lock = runtime._CrossProcessFileLock(path, timeout_seconds=1.0, sleep=cancel)
    try:
        with monkeypatch.context() as patch:
            if os.name == "nt":
                import msvcrt

                patch.setattr(msvcrt, "locking", blocked)
            else:
                import fcntl

                patch.setattr(fcntl, "flock", blocked)
            with pytest.raises(KeyboardInterrupt, match="injected cancellation"):
                with lock:
                    pytest.fail("contended acquisition must not enter")
        # No OS lock was acquired in this case: a blocked peer specifically
        # exposes the unreleased local mutex, independently of handle closure.
        assert peer_enters(path)
        assert files[0].raw.closed
        assert files[0].close_attempts == 1
    finally:
        cleanup_fixture(lock, files)


def test_interlock_holds_until_context_exit_and_releases_after_body_failure(tmp_path):
    path = tmp_path / "body.lock"
    with pytest.raises(ValueError, match="body failure"):
        with runtime._CrossProcessFileLock(path, timeout_seconds=0.2):
            assert peer_enters(path) is False
            raise ValueError("body failure")
    assert peer_enters(path)
