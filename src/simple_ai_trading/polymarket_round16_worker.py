"""Session-owned numerical inference for the promoted Round 16 profile."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess  # nosec B404 - fixed interpreter and module
import sys
from tempfile import TemporaryFile
from threading import Event, Lock
import time

import numpy as np

from .polymarket_round16_shadow import (
    ROUND16_FEATURE_NAMES,
    VerifiedRound16ShadowPredictor,
    _predict_up_probability,
)
from .windows_owned_job import CREATE_SUSPENDED, WindowsOwnedJob


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('simple_ai_trading.polymarket_round16_worker',run_name='__main__')"
)
_BOOTSTRAP_HEADER = struct.Struct("!II")
_REQUEST_ID = struct.Struct("!Q")
_REPLY = struct.Struct("!Qd")
_VECTOR_BYTES = 4 * len(ROUND16_FEATURE_NAMES)
_DIGEST_BYTES = 32
_MAX_SHARE_BYTES = 4_096
_MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
_STARTUP_SECONDS = 15.0
_REAP_SECONDS = 2


def _receive_exact(
    peer: socket.socket, size: int, *, deadline: float | None = None
) -> bytes:
    received = bytearray()
    while len(received) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Round 16 inference response deadline expired")
            peer.settimeout(remaining)
        chunk = peer.recv(size - len(received))
        if not chunk:
            if received:
                raise EOFError("Round 16 inference received a truncated frame")
            return b""
        received.extend(chunk)
    return bytes(received)


class PolymarketRound16InferenceWorker:
    """Own one child and one in-flight request, with bounded teardown on Stop."""

    trading_authority = False

    def __init__(
        self, predictor: VerifiedRound16ShadowPredictor, *, timeout_seconds: float = 2.0
    ) -> None:
        if not isinstance(predictor, VerifiedRound16ShadowPredictor):
            raise TypeError("Round 16 inference requires a verified predictor")
        if not 0.05 <= timeout_seconds <= 3:
            raise ValueError("Round 16 inference timeout must lie in [0.05, 3] seconds")
        self._candidate = json.dumps(
            dict(predictor.candidate), separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        if not 0 < len(self._candidate) <= _MAX_CANDIDATE_BYTES:
            raise ValueError("Round 16 inference candidate exceeds its byte limit")
        self._candidate_digest = hashlib.sha256(self._candidate).digest()
        self._timeout = timeout_seconds
        self._started = False
        self._connection: socket.socket | None = None
        self._request_id = 0
        self._state_lock = Lock()
        self._exchange = Lock()
        self._fault = Event()

    def _command(self, descriptor: int) -> list[str]:
        return [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _BOOTSTRAP,
            str(_PACKAGE_ROOT),
            str(descriptor),
        ]

    def predict(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype="<f4")
        if vector.shape != (len(ROUND16_FEATURE_NAMES),) or np.any(
            ~np.isfinite(vector)
        ):
            raise ValueError("Round 16 inference feature vector is invalid")
        if not self._exchange.acquire(blocking=False):
            raise RuntimeError("Round 16 inference already has a pending request")
        try:
            with self._state_lock:
                connection = self._connection
            if connection is None or self._fault.is_set():
                raise RuntimeError("Round 16 inference worker is unavailable")
            self._request_id += 1
            packet = _REQUEST_ID.pack(self._request_id) + vector.tobytes(order="C")
            deadline = time.monotonic() + self._timeout
            try:
                connection.settimeout(self._timeout)
                connection.sendall(packet)
                response = _receive_exact(connection, _REPLY.size, deadline=deadline)
                if len(response) != _REPLY.size or time.monotonic() >= deadline:
                    raise RuntimeError("Round 16 inference response expired or closed")
                request_id, probability = _REPLY.unpack(response)
                if request_id != self._request_id or not 0.0 < probability < 1.0:
                    raise RuntimeError("Round 16 inference response is invalid")
                return probability
            except (OSError, EOFError, RuntimeError):
                self._fault.set()
                raise
        finally:
            self._exchange.release()

    async def run(self, stop: asyncio.Event) -> None:
        if self._started:
            raise RuntimeError("Round 16 inference worker cannot be restarted")
        self._started = True
        if stop.is_set():
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
                        bootstrap.write(_BOOTSTRAP_HEADER.pack(0, len(self._candidate)))
                        bootstrap.write(self._candidate)
                        bootstrap.flush()
                        bootstrap.seek(0)
                    process = subprocess.Popen(  # nosec B603
                        self._command(descriptor),
                        stdin=bootstrap,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        cwd=_PACKAGE_ROOT,
                        creationflags=flags,
                        pass_fds=inherited,
                        close_fds=True,
                    )
                    if job is not None:
                        share = child.share(process.pid)
                        if not 0 < len(share) <= _MAX_SHARE_BYTES:
                            raise RuntimeError("Round 16 socket bootstrap size differs")
                        bootstrap.write(
                            _BOOTSTRAP_HEADER.pack(len(share), len(self._candidate))
                        )
                        bootstrap.write(share)
                        bootstrap.write(self._candidate)
                        bootstrap.flush()
                        bootstrap.seek(0)
                        job.enroll_and_resume(process)
                child.close()
                await self._supervise(connection, process, stop)
            finally:
                with self._state_lock:
                    self._connection = None
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
        deadline = loop.time() + _STARTUP_SECONDS
        connection.setblocking(False)
        ready = bytearray()
        while not stop.is_set():
            if process.poll() is not None or self._fault.is_set():
                raise RuntimeError("Round 16 inference worker failed")
            if len(ready) < _DIGEST_BYTES:
                if loop.time() >= deadline:
                    raise TimeoutError("Round 16 inference startup deadline expired")
                try:
                    block = connection.recv(_DIGEST_BYTES - len(ready))
                except BlockingIOError:
                    pass
                else:
                    if not block:
                        raise RuntimeError(
                            "Round 16 inference worker closed at startup"
                        )
                    ready.extend(block)
                    if len(ready) == _DIGEST_BYTES:
                        if ready != self._candidate_digest:
                            raise RuntimeError(
                                "Round 16 inference candidate binding differs"
                            )
                        with self._state_lock:
                            self._connection = connection
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=0.05 if len(ready) < _DIGEST_BYTES else 0.25
                )
            except TimeoutError:
                continue


def _worker(descriptor: int) -> None:
    with sys.stdin.buffer as source:
        share_size, candidate_size = _BOOTSTRAP_HEADER.unpack(
            source.read(_BOOTSTRAP_HEADER.size)
        )
        if not (
            0 <= share_size <= _MAX_SHARE_BYTES
            and 0 < candidate_size <= _MAX_CANDIDATE_BYTES
        ):
            raise ValueError("Round 16 bootstrap bounds differ")
        share = source.read(share_size)
        payload = source.read(candidate_size)
        if len(share) != share_size or len(payload) != candidate_size or source.read(1):
            raise ValueError("Round 16 bootstrap size differs")
    candidate = json.loads(payload)
    if not isinstance(candidate, dict):
        raise TypeError("Round 16 inference candidate is malformed")
    connection = socket.fromshare(share) if share else socket.socket(fileno=descriptor)
    with connection:
        _predict_up_probability(
            candidate, np.zeros(len(ROUND16_FEATURE_NAMES), dtype=np.float32)
        )
        connection.sendall(hashlib.sha256(payload).digest())
        while request := _receive_exact(connection, _REQUEST_ID.size + _VECTOR_BYTES):
            (request_id,) = _REQUEST_ID.unpack_from(request)
            features = np.frombuffer(request, dtype="<f4", offset=_REQUEST_ID.size)
            probability = _predict_up_probability(candidate, features)
            connection.sendall(_REPLY.pack(request_id, probability))


if __name__ == "__main__":
    _worker(int(sys.argv[1]))
