"""Session-owned numerical inference for the promoted Round 16 profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

from .polymarket_inference_process import (
    PolymarketInferenceProcess,
    read_bootstrap,
    receive_exact as _receive_exact,
)
from .polymarket_round16_shadow import (
    ROUND16_FEATURE_NAMES,
    VerifiedRound16ShadowPredictor,
    _predict_up_probability,
)


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('simple_ai_trading.polymarket_round16_worker',run_name='__main__')"
)
_REQUEST_ID = struct.Struct("!Q")
_REPLY = struct.Struct("!Qd")
_VECTOR_BYTES = 4 * len(ROUND16_FEATURE_NAMES)
_MAX_CANDIDATE_BYTES = 4 * 1024 * 1024


class PolymarketRound16InferenceWorker(PolymarketInferenceProcess):
    """Own one child and one in-flight request, with bounded teardown on Stop."""

    def __init__(
        self, predictor: VerifiedRound16ShadowPredictor, *, timeout_seconds: float = 2.0
    ) -> None:
        if not isinstance(predictor, VerifiedRound16ShadowPredictor):
            raise TypeError("Round 16 inference requires a verified predictor")
        candidate = json.dumps(
            dict(predictor.candidate), separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        if not 0 < len(candidate) <= _MAX_CANDIDATE_BYTES:
            raise ValueError("Round 16 inference candidate exceeds its byte limit")
        super().__init__(candidate, label="Round 16", timeout_seconds=timeout_seconds)

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


def _worker(descriptor: int) -> None:
    connection, payload = read_bootstrap(descriptor, _MAX_CANDIDATE_BYTES)
    with connection:
        candidate = json.loads(payload)
        if not isinstance(candidate, dict):
            raise TypeError("Round 16 inference candidate is malformed")
        _predict_up_probability(
            candidate, np.zeros(len(ROUND16_FEATURE_NAMES), dtype=np.float32)
        )
        connection.sendall(hashlib.sha256(payload).digest())
        del payload
        while request := _receive_exact(connection, _REQUEST_ID.size + _VECTOR_BYTES):
            (request_id,) = _REQUEST_ID.unpack_from(request)
            features = np.frombuffer(request, dtype="<f4", offset=_REQUEST_ID.size)
            probability = _predict_up_probability(candidate, features)
            connection.sendall(_REPLY.pack(request_id, probability))


if __name__ == "__main__":
    _worker(int(sys.argv[1]))
