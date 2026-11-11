"""Session-owned HTTP execution with bounded IPC and no request replay."""

from __future__ import annotations

from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess  # nosec B404 - fixed isolated interpreter/module
import sys
from tempfile import TemporaryFile
from threading import Lock, TIMEOUT_MAX
from time import monotonic
from typing import Mapping

import httpx
from py_clob_client_v2.exceptions import PolyApiException

from .polymarket_inference_process import read_bootstrap, receive_exact
from .windows_owned_job import CREATE_SUSPENDED, WindowsOwnedJob


_HEADER = struct.Struct("!QI")
_BOOTSTRAP_HEADER = struct.Struct("!II")
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_BOOTSTRAP_BYTES = 16 * 1024
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('simple_ai_trading.polymarket_http_process',run_name='__main__')"
)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("HTTP worker frame contains duplicate fields")
        value[key] = item
    return value


def _reject_constant(_value):
    raise ValueError("HTTP worker frame contains a nonfinite constant")


def _decode(body: bytes):
    return json.loads(
        body, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )


def _encode(value, maximum_bytes: int) -> bytes:
    body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("ascii")
    if not 0 < len(body) <= maximum_bytes:
        raise ValueError("HTTP worker frame exceeds its byte limit")
    return body


def _receive(connection, maximum_bytes, *, deadline=None):
    header = receive_exact(connection, _HEADER.size, deadline=deadline)
    if not header:
        raise EOFError("HTTP worker channel closed")
    sequence, size = _HEADER.unpack(header)
    if not 0 < size <= maximum_bytes:
        raise ValueError("HTTP worker frame size differs")
    body = receive_exact(connection, size, deadline=deadline)
    if len(body) != size:
        raise EOFError("HTTP worker frame is incomplete")
    return sequence, _decode(body)


class PolymarketHttpProcess:
    """Own one persistent HTTP child and serialize bounded individual exchanges.

    A failed child is reaped before a later caller can start a fresh generation.
    No request is replayed. Errors after dispatch retain unknown status, leaving
    exact-order reconciliation and opening admission to the durable gateway.
    This is process containment, not a credential-security sandbox.
    """

    def __init__(self, *, base_url, timeout_seconds, maximum_response_bytes):
        self._timeout = timeout_seconds
        self._maximum_reply = 6 * maximum_response_bytes + 16 * 1024
        self._bootstrap = _encode(
            {
                "base_url": base_url,
                "timeout_seconds": timeout_seconds,
                "maximum_response_bytes": maximum_response_bytes,
            },
            _MAX_BOOTSTRAP_BYTES,
        )
        self._exchange = Lock()
        self._state = Lock()
        self._closed = False
        self._sequence = 0
        self._connection: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._job: WindowsOwnedJob | None = None

    @property
    def is_closed(self):
        with self._state:
            return self._closed

    def _command(self, descriptor):
        return [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _BOOTSTRAP,
            str(_PACKAGE_ROOT),
            str(descriptor),
        ]

    def _drop_child(self):
        with self._state:
            connection, process, job = self._connection, self._process, self._job
            self._connection = self._process = self._job = None
        if connection is not None:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()
        try:
            if job is not None:
                job.close()
        finally:
            if process is not None:
                try:
                    if job is None and os.name != "nt" and process.poll() is None:
                        if os.getpgid(process.pid) != process.pid:
                            raise RuntimeError("HTTP worker process group differs")
                        os.killpg(process.pid, signal.SIGKILL)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=2)

    def _connect(self, deadline):
        if monotonic() >= deadline:
            raise TimeoutError("HTTP startup deadline expired")
        with self._state:
            if self._closed:
                raise RuntimeError("HTTP transport is closed")
            if self._process is not None and self._process.poll() is None:
                return self._connection
        self._drop_child()
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR"}
        }
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        connection, child = socket.socketpair()
        try:
            with self._state:
                if self._closed:
                    raise RuntimeError("HTTP transport is closed")
                self._connection = connection
                with TemporaryFile() as bootstrap:
                    if os.name == "nt":
                        self._job = WindowsOwnedJob()
                        descriptor = -1
                        options = {
                            "creationflags": subprocess.CREATE_NO_WINDOW
                            | CREATE_SUSPENDED,
                        }
                    else:
                        descriptor = child.fileno()
                        options = {"pass_fds": (descriptor,), "start_new_session": True}
                        bootstrap.write(_BOOTSTRAP_HEADER.pack(0, len(self._bootstrap)))
                        bootstrap.write(self._bootstrap)
                        bootstrap.flush()
                        bootstrap.seek(0)
                    process = subprocess.Popen(  # nosec B603
                        self._command(descriptor),
                        stdin=bootstrap,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        cwd=_PACKAGE_ROOT,
                        close_fds=True,
                        **options,
                    )
                    self._process = process
                    if self._job is not None:
                        share = child.share(process.pid)
                        if not 0 < len(share) <= 4096:
                            raise RuntimeError("HTTP socket bootstrap size differs")
                        bootstrap.write(
                            _BOOTSTRAP_HEADER.pack(len(share), len(self._bootstrap))
                        )
                        bootstrap.write(share)
                        bootstrap.write(self._bootstrap)
                        bootstrap.flush()
                        bootstrap.seek(0)
                        self._job.enroll_and_resume(process)
            child.close()
            ready = receive_exact(connection, 32, deadline=deadline)
            if ready != hashlib.sha256(self._bootstrap).digest():
                raise ValueError("HTTP bootstrap binding differs")
            return connection
        except BaseException:
            self._drop_child()
            connection.close()
            raise
        finally:
            child.close()

    def start(self, *, timeout_seconds=None):
        """Optionally initialize an explicit session before its first request."""
        timeout = self._timeout if timeout_seconds is None else timeout_seconds
        if not 0 < timeout <= TIMEOUT_MAX:
            raise ValueError("HTTP worker startup timeout is invalid")
        deadline = monotonic() + timeout
        if not self._exchange.acquire(timeout=timeout):
            raise PolyApiException(error_msg="SDK transport capacity deadline exceeded")
        try:
            self._connect(deadline)
        finally:
            self._exchange.release()

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: object = None,
        params: Mapping[str, object] | None = None,
    ) -> object:
        deadline = monotonic() + self._timeout
        # Use the pinned native query encoder before JSON IPC. Decimal, bool,
        # null and repeated values must retain HTTPX's exact wire semantics.
        query = httpx.QueryParams(params) if params is not None else None
        body = _encode(
            {
                "method": method,
                "endpoint": endpoint,
                "headers": dict(headers or {}),
                "data": data,
                "params": (
                    {key: query.get_list(key) for key in query}
                    if query is not None
                    else None
                ),
                "deadline": deadline,
            },
            _MAX_REQUEST_BYTES,
        )
        if not self._exchange.acquire(timeout=min(self._timeout, TIMEOUT_MAX)):
            raise PolyApiException(error_msg="SDK transport capacity deadline exceeded")
        try:
            try:
                connection = self._connect(deadline)
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError
                connection.settimeout(remaining)
                self._sequence += 1
                connection.sendall(_HEADER.pack(self._sequence, len(body)) + body)
                sequence, reply = _receive(
                    connection, self._maximum_reply, deadline=deadline
                )
                if (
                    monotonic() >= deadline
                    or sequence != self._sequence
                    or not isinstance(reply, dict)
                    or set(reply) != {"ok", "status_code", "value"}
                    or type(reply["ok"]) is not bool
                    or reply["status_code"] is not None
                    and (
                        type(reply["status_code"]) is not int
                        or not 100 <= reply["status_code"] <= 599
                    )
                    or reply["ok"]
                    and reply["status_code"] != 200
                ):
                    raise ValueError("HTTP response binding differs")
            except Exception:
                self._drop_child()
                # No partial/native error can be relabeled a definitive rejection.
                raise PolyApiException(
                    error_msg="SDK response deadline exceeded or worker unavailable"
                ) from None
            if not reply["ok"]:
                error = PolyApiException(error_msg="SDK transport request failed")
                error.status_code = reply["status_code"]
                error.error_msg = reply["value"]
                raise error
            return reply["value"]
        finally:
            self._exchange.release()

    def close(self):
        with self._state:
            self._closed = True
        self._drop_child()


def _worker(descriptor):
    from .polymarket_sdk_transport import _InlineSdkTransport

    connection, bootstrap = read_bootstrap(descriptor, _MAX_BOOTSTRAP_BYTES)
    with connection:
        configuration = _decode(bootstrap)
        if not isinstance(configuration, dict) or set(configuration) != {
            "base_url",
            "timeout_seconds",
            "maximum_response_bytes",
        }:
            raise ValueError("HTTP bootstrap fields differ")
        transport = _InlineSdkTransport(**configuration)
        try:
            maximum_reply = 6 * configuration["maximum_response_bytes"] + 16 * 1024
            connection.sendall(hashlib.sha256(bootstrap).digest())
            del bootstrap, configuration
            while True:
                try:
                    sequence, request = _receive(connection, _MAX_REQUEST_BYTES)
                except EOFError:
                    return
                if not isinstance(request, dict) or set(request) != {
                    "method",
                    "endpoint",
                    "headers",
                    "data",
                    "params",
                    "deadline",
                }:
                    raise ValueError("HTTP request fields differ")
                deadline = request.pop("deadline")
                try:
                    if monotonic() >= deadline:
                        raise PolyApiException(
                            error_msg="SDK request expired before dispatch"
                        )
                    reply = {
                        "ok": True,
                        "status_code": 200,
                        "value": transport.request(**request),
                    }
                except PolyApiException as exc:
                    reply = {
                        "ok": False,
                        "status_code": exc.status_code,
                        "value": exc.error_msg,
                    }
                body = _encode(reply, maximum_reply)
                connection.sendall(_HEADER.pack(sequence, len(body)) + body)
                del request, reply, body
        finally:
            transport.close()


if __name__ == "__main__":
    _worker(int(sys.argv[1]))
