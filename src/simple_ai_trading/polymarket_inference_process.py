"""Own the lifecycle of a Polymarket numerical inference child."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import os
from pathlib import Path
import socket
import struct
import subprocess  # nosec B404 - subclass supplies a fixed interpreter/module
from tempfile import TemporaryFile
from threading import Event, Lock
import time

from .windows_owned_job import CREATE_SUSPENDED, WindowsOwnedJob


_BOOTSTRAP_HEADER = struct.Struct("!II")
_DIGEST_BYTES = 32
_MAX_SHARE_BYTES = 4_096
_MAX_BOOTSTRAP_BYTES = 64 * 1024 * 1024
_REAP_SECONDS = 2


def receive_exact(
    peer: socket.socket, size: int, *, deadline: float | None = None
) -> bytes:
    received = bytearray()
    while len(received) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Inference response deadline expired")
            peer.settimeout(remaining)
        chunk = peer.recv(size - len(received))
        if not chunk:
            if received:
                raise EOFError("Inference received a truncated frame")
            return b""
        received.extend(chunk)
    return bytes(received)


def read_bootstrap(descriptor: int, maximum_bytes: int) -> tuple[socket.socket, bytes]:
    import sys

    with sys.stdin.buffer as source:
        share_size, payload_size = _BOOTSTRAP_HEADER.unpack(
            source.read(_BOOTSTRAP_HEADER.size)
        )
        if not (
            0 <= share_size <= _MAX_SHARE_BYTES
            and 0 < payload_size <= min(maximum_bytes, _MAX_BOOTSTRAP_BYTES)
        ):
            raise ValueError("Inference bootstrap bounds differ")
        share = source.read(share_size)
        payload = source.read(payload_size)
        if len(share) != share_size or len(payload) != payload_size or source.read(1):
            raise ValueError("Inference bootstrap size differs")
    connection = socket.fromshare(share) if share else socket.socket(fileno=descriptor)
    return connection, payload


class PolymarketInferenceProcess:
    """Keep bootstrap, startup, failure and Stop teardown under one owner."""

    trading_authority = False

    def __init__(
        self,
        payload: bytes,
        *,
        label: str,
        timeout_seconds: float,
        startup_timeout_seconds: float = 15.0,
    ) -> None:
        if not 0 < len(payload) <= _MAX_BOOTSTRAP_BYTES:
            raise ValueError(f"{label} inference bootstrap exceeds its byte limit")
        if not 0.05 <= timeout_seconds <= 3:
            raise ValueError(f"{label} inference timeout must lie in [0.05, 3] seconds")
        if not 0.05 <= startup_timeout_seconds <= 60:
            raise ValueError("Inference startup timeout must lie in [0.05, 60] seconds")
        self._payload = payload
        self._payload_digest = hashlib.sha256(payload).digest()
        self._label = label
        self._timeout = timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._started = False
        self._connection: socket.socket | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._state_lock = Lock()
        self._exchange = Lock()
        self._fault = Event()

    def _command(self, descriptor: int) -> list[str]:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        with self._state_lock:
            return (
                self._connection is not None
                and self._process is not None
                and self._process.poll() is None
                and not self._fault.is_set()
            )

    async def run(self, stop: asyncio.Event) -> None:
        if self._started:
            raise RuntimeError(f"{self._label} inference worker cannot be restarted")
        self._started = True
        if stop.is_set():
            self._payload = b""
            return
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR"}
        }
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        job: WindowsOwnedJob | None = None
        process: subprocess.Popen[bytes] | None = None
        connection, child = socket.socketpair()
        with connection, child:
            try:
                with TemporaryFile() as bootstrap:
                    if os.name == "nt":
                        job = WindowsOwnedJob()
                        descriptor = -1
                        flags = subprocess.CREATE_NO_WINDOW | CREATE_SUSPENDED
                        inherited = ()
                    else:
                        descriptor = child.fileno()
                        flags = 0
                        inherited = (descriptor,)
                        bootstrap.write(_BOOTSTRAP_HEADER.pack(0, len(self._payload)))
                        bootstrap.write(self._payload)
                        bootstrap.flush()
                        bootstrap.seek(0)
                    process = subprocess.Popen(  # nosec B603
                        self._command(descriptor),
                        stdin=bootstrap,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        cwd=Path(__file__).resolve().parents[1],
                        creationflags=flags,
                        pass_fds=inherited,
                        close_fds=True,
                    )
                    with self._state_lock:
                        self._process = process
                    if job is not None:
                        share = child.share(process.pid)
                        if not 0 < len(share) <= _MAX_SHARE_BYTES:
                            raise RuntimeError(
                                "Inference socket bootstrap size differs"
                            )
                        bootstrap.write(
                            _BOOTSTRAP_HEADER.pack(len(share), len(self._payload))
                        )
                        bootstrap.write(share)
                        bootstrap.write(self._payload)
                        bootstrap.flush()
                        bootstrap.seek(0)
                        job.enroll_and_resume(process)
                child.close()
                self._payload = b""
                await self._supervise(connection, process, stop)
            finally:
                self._payload = b""
                with self._state_lock:
                    self._connection = None
                    self._process = None
                with suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
                try:
                    if job is not None:
                        job.close()
                finally:
                    if process is not None:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=_REAP_SECONDS)

    async def _supervise(
        self,
        connection: socket.socket,
        process: subprocess.Popen[bytes],
        stop: asyncio.Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout
        connection.setblocking(False)
        ready = bytearray()
        while not stop.is_set():
            if process.poll() is not None or self._fault.is_set():
                raise RuntimeError(f"{self._label} inference worker failed")
            if len(ready) < _DIGEST_BYTES:
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"{self._label} inference startup deadline expired"
                    )
                try:
                    block = connection.recv(_DIGEST_BYTES - len(ready))
                except BlockingIOError:
                    pass
                else:
                    if not block:
                        raise RuntimeError(
                            f"{self._label} inference worker closed at startup"
                        )
                    ready.extend(block)
                    if len(ready) == _DIGEST_BYTES:
                        if ready != self._payload_digest:
                            raise RuntimeError(
                                f"{self._label} inference bootstrap binding differs"
                            )
                        with self._state_lock:
                            self._connection = connection
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=0.05 if len(ready) < _DIGEST_BYTES else 0.25
                )
            except TimeoutError:
                continue
