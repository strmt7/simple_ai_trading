"""Owned, target-free numerical inference for the promoted Round21 profile."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
from threading import Lock
import time

import numpy as np

from .polymarket_inference_process import (
    PolymarketInferenceProcess,
    read_bootstrap,
    receive_exact,
)
from .polymarket_round21_model import (
    Round21InferencePanel,
    Round21ProbabilityBatch,
    _reject_nonfinite,
    _strict_object,
    compile_round21_probability_predictor,
    validate_round21_development_artifact,
)
from .polymarket_round21_tcn import ROUND21_TCN_SEQUENCE_LENGTH


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('simple_ai_trading.polymarket_round21_worker',run_name='__main__')"
)
_MESSAGE = struct.Struct("!QI")
_MAX_MODEL_BYTES = 64 * 1024 * 1024
_MAX_MESSAGE_BYTES = 2 * 1024 * 1024
_PANEL_ARRAYS = (
    "condition_ids",
    "event_start_ms",
    "decision_time_ms",
    "structural_probability",
    "market_prior_probability",
    "core_features",
    "spot_features",
    "usdm_features",
    "spot_available",
    "usdm_available",
)
_PANEL_IDENTITIES = (
    "core_feature_names_sha256",
    "spot_feature_names_sha256",
    "usdm_feature_names_sha256",
    "source_dataset_sha256",
    "feature_batch_sha256",
    "dataset_design_sha256",
)
_BATCH_ARRAYS = (
    "indices",
    "probability_up",
    "lower_up",
    "upper_up",
    "feature_support_eligible",
)
_BATCH_IDENTITIES = (
    "population_layer",
    "selected_candidate_id",
    "contributing_candidate_ids",
    "source_model_artifact_sha256",
    "feature_batch_sha256",
    "prediction_sha256",
)
_BACKEND_FIELDS = (
    "tcn_training_backend_kind",
    "tcn_training_backend_device",
    "tcn_runtime_backend_kind",
    "tcn_runtime_backend_device",
    "tcn_backend_substituted",
    "tcn_accelerator_fallback",
)
_BACKEND_KINDS = frozenset(("cpu", "cuda", "rocm", "xpu", "directml", "mps"))


def _encode(value: dict) -> bytes:
    body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("ascii")
    if not 0 < len(body) <= _MAX_MESSAGE_BYTES:
        raise ValueError("Round21 inference message exceeds its byte limit")
    return body


def _receive_message(connection: socket.socket, *, deadline: float | None = None):
    header = receive_exact(connection, _MESSAGE.size, deadline=deadline)
    if not header:
        raise EOFError("Round21 inference peer closed")
    request_id, size = _MESSAGE.unpack(header)
    if not 0 < size <= _MAX_MESSAGE_BYTES:
        raise ValueError("Round21 inference response size differs")
    body = receive_exact(connection, size, deadline=deadline)
    if len(body) != size:
        raise EOFError("Round21 inference body is incomplete")
    return request_id, json.loads(
        body, object_pairs_hook=_strict_object, parse_constant=_reject_nonfinite
    )


def _panel_payload(panel: Round21InferencePanel) -> dict:
    if not isinstance(panel, Round21InferencePanel):
        raise TypeError("Round21 worker requires a target-free inference panel")
    selected = panel.validate()
    if not 1 <= len(selected.condition_ids) <= ROUND21_TCN_SEQUENCE_LENGTH:
        raise ValueError("Round21 worker inference history exceeds its row limit")
    return {
        **{name: getattr(selected, name).tolist() for name in _PANEL_ARRAYS},
        **{name: getattr(selected, name) for name in _PANEL_IDENTITIES},
    }


def _panel_from_payload(value: dict) -> Round21InferencePanel:
    if not isinstance(value, dict) or set(value) != set(
        _PANEL_ARRAYS + _PANEL_IDENTITIES
    ):
        raise ValueError("Round21 inference panel fields differ")
    conditions = value["condition_ids"]
    if (
        not isinstance(conditions, list)
        or not 1 <= len(conditions) <= ROUND21_TCN_SEQUENCE_LENGTH
    ):
        raise ValueError("Round21 inference panel row count differs")
    payload = dict(value)
    expected = payload.pop("feature_batch_sha256")
    panel = Round21InferencePanel.create(**payload)
    if panel.feature_batch_sha256 != expected:
        raise ValueError("Round21 inference panel digest differs")
    return panel


def _batch_payload(batch: Round21ProbabilityBatch) -> dict:
    selected = batch.validated()
    return {
        **{name: getattr(selected, name).tolist() for name in _BATCH_ARRAYS},
        **{name: getattr(selected, name) for name in _BATCH_IDENTITIES},
    }


def _batch_from_payload(value: dict) -> Round21ProbabilityBatch:
    if not isinstance(value, dict) or set(value) != set(
        _BATCH_ARRAYS + _BATCH_IDENTITIES
    ):
        raise ValueError("Round21 probability response fields differ")
    for field, expected_type in (("indices", int), ("feature_support_eligible", bool)):
        if not isinstance(value[field], list) or any(
            type(item) is not expected_type for item in value[field]
        ):
            raise ValueError("Round21 probability response array types differ")
    payload = dict(value)
    expected = payload.pop("prediction_sha256")
    batch = Round21ProbabilityBatch.create(**payload).validated()
    if batch.prediction_sha256 != expected:
        raise ValueError("Round21 probability response digest differs")
    return batch


class PolymarketRound21InferenceWorker(PolymarketInferenceProcess):
    """Compile and evaluate the frozen hull in a child with bounded requests."""

    def __init__(
        self,
        artifact: Mapping[str, object],
        *,
        population_layer: str,
        timeout_seconds: float = 2.0,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        selected = validate_round21_development_artifact(artifact)
        layers = selected["layers"]
        if population_layer not in layers:
            raise ValueError("Round21 worker population layer is unavailable")
        self.population_layer = population_layer
        self.artifact_sha256 = selected["artifact_sha256"]
        identity = selected["dataset_and_partition"]
        self.core_feature_names_sha256 = identity["core_feature_names_sha256"]
        self.spot_feature_names_sha256 = identity["spot_feature_names_sha256"]
        self.usdm_feature_names_sha256 = identity["usdm_feature_names_sha256"]
        layer = layers[population_layer]
        ledgers = [layer["candidate_ledger"]]
        if population_layer != "core":
            ledgers.append(layer["matched_core_candidate_ledger"])
        self._candidate_ids = tuple(
            record["candidate_id"] for ledger in ledgers for record in ledger
        )
        self._selected_id = layer["selected_candidate_id"]
        self._training_backend = (
            selected["compute"]["tcn_backend_kind"],
            selected["compute"]["tcn_backend_device"],
        )
        self._backend: dict | None = None
        self._backend_lock = Lock()
        payload = json.dumps(
            {"artifact": selected, "population_layer": population_layer},
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        if len(payload) > _MAX_MODEL_BYTES:
            raise ValueError("Round21 inference model exceeds its byte limit")
        super().__init__(
            payload,
            label="Round21",
            timeout_seconds=timeout_seconds,
            startup_timeout_seconds=startup_timeout_seconds,
        )

    @property
    def tcn_training_backend_kind(self) -> str:
        return self._training_backend[0]

    @property
    def tcn_training_backend_device(self) -> str:
        return self._training_backend[1]

    @property
    def tcn_runtime_backend_kind(self) -> str:
        with self._backend_lock:
            return (
                "pending"
                if self._backend is None
                else self._backend["tcn_runtime_backend_kind"]
            )

    @property
    def tcn_runtime_backend_device(self) -> str:
        with self._backend_lock:
            return (
                "pending"
                if self._backend is None
                else self._backend["tcn_runtime_backend_device"]
            )

    @property
    def tcn_backend_substituted(self) -> bool:
        with self._backend_lock:
            return (
                False
                if self._backend is None
                else self._backend["tcn_backend_substituted"]
            )

    @property
    def tcn_accelerator_fallback(self) -> bool:
        with self._backend_lock:
            return (
                False
                if self._backend is None
                else self._backend["tcn_accelerator_fallback"]
            )

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

    def _admit_backend(self, backend: dict) -> None:
        if not isinstance(backend, dict) or set(backend) != set(_BACKEND_FIELDS):
            raise ValueError("Round21 runtime backend fields differ")
        if (
            (
                backend["tcn_training_backend_kind"],
                backend["tcn_training_backend_device"],
            )
            != self._training_backend
            or backend["tcn_runtime_backend_kind"] not in _BACKEND_KINDS
            or not isinstance(backend["tcn_runtime_backend_device"], str)
            or not 1 <= len(backend["tcn_runtime_backend_device"]) <= 128
            or type(backend["tcn_backend_substituted"]) is not bool
            or type(backend["tcn_accelerator_fallback"]) is not bool
        ):
            raise ValueError("Round21 runtime backend identity differs")
        with self._backend_lock:
            if self._backend is not None and self._backend != backend:
                raise ValueError("Round21 runtime backend changed within the worker")
            self._backend = dict(backend)

    def predict(self, panel: Round21InferencePanel) -> Round21ProbabilityBatch:
        body = _encode(_panel_payload(panel))
        if not self._exchange.acquire(blocking=False):
            raise RuntimeError("Round21 inference already has a pending request")
        try:
            with self._state_lock:
                connection = self._connection
            if connection is None or self._fault.is_set():
                raise RuntimeError("Round21 inference worker is unavailable")
            self._request_id += 1
            deadline = time.monotonic() + self._timeout
            try:
                connection.settimeout(self._timeout)
                connection.sendall(_MESSAGE.pack(self._request_id, len(body)) + body)
                request_id, response = _receive_message(connection, deadline=deadline)
                if request_id != self._request_id or time.monotonic() >= deadline:
                    raise ValueError("Round21 inference response is late or misbound")
                if not isinstance(response, dict) or set(response) != {
                    "batch",
                    "backend",
                }:
                    raise ValueError("Round21 inference response fields differ")
                batch = _batch_from_payload(response["batch"])
                available = np.ones(len(panel.condition_ids), dtype=np.bool_)
                if self.population_layer != "core":
                    available &= panel.spot_available
                if self.population_layer == "core_spot_usdm":
                    available &= panel.usdm_available
                if (
                    batch.source_model_artifact_sha256 != self.artifact_sha256
                    or batch.feature_batch_sha256 != panel.feature_batch_sha256
                    or batch.population_layer != self.population_layer
                    or batch.selected_candidate_id != self._selected_id
                    or batch.contributing_candidate_ids != self._candidate_ids
                    or not np.array_equal(batch.indices, np.flatnonzero(available))
                ):
                    raise ValueError("Round21 inference response provenance differs")
                self._admit_backend(response["backend"])
                return batch
            except Exception:
                self._fault.set()
                raise
        finally:
            self._exchange.release()


def _worker(descriptor: int) -> None:
    connection, payload = read_bootstrap(descriptor, _MAX_MODEL_BYTES)
    with connection:
        bootstrap = json.loads(
            payload, object_pairs_hook=_strict_object, parse_constant=_reject_nonfinite
        )
        if not isinstance(bootstrap, dict) or set(bootstrap) != {
            "artifact",
            "population_layer",
        }:
            raise ValueError("Round21 model bootstrap fields differ")
        predictor = compile_round21_probability_predictor(
            bootstrap["artifact"],
            population_layer=bootstrap["population_layer"],
        )
        backend = {field: getattr(predictor, field) for field in _BACKEND_FIELDS}
        connection.sendall(hashlib.sha256(payload).digest())
        del payload, bootstrap
        while True:
            try:
                request_id, request = _receive_message(connection)
            except EOFError:
                return
            panel = _panel_from_payload(request)
            batch = predictor.predict(panel)
            body = _encode({"batch": _batch_payload(batch), "backend": backend})
            connection.sendall(_MESSAGE.pack(request_id, len(body)) + body)


if __name__ == "__main__":
    _worker(int(sys.argv[1]))
