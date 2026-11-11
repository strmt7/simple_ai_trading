from __future__ import annotations

import asyncio
import json
import socket
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from simple_ai_trading import polymarket_inference_process as process_module
from simple_ai_trading import polymarket_round21_tcn as tcn_module
from simple_ai_trading import polymarket_round21_worker as worker_module
from simple_ai_trading.polymarket_autonomous_runtime import PolymarketAutonomousDecision
from simple_ai_trading.polymarket_round21_model import (
    Round21InferencePanel,
    compile_round21_probability_predictor,
    fit_round21_development,
)
from simple_ai_trading.polymarket_runtime_control import (
    PolymarketRuntimeControl,
    PolymarketRuntimeControlService,
)
from test_polymarket_autonomous_runtime import _supervisor, _market, OWNED_MARKET_ID
from test_polymarket_round21_model import _accepted_basis_ablation, _panel


@pytest.fixture(scope="module")
def model_artifact():
    train = _panel("train", first_condition=0, condition_count=100)
    calibration = _panel("tune_calibration", first_condition=106, condition_count=120)
    selection = _panel("tune_selection", first_condition=226, condition_count=80)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tcn_module, "ROUND21_TCN_MAXIMUM_EPOCHS", 1)
        return fit_round21_development(
            train=train,
            tune_calibration=calibration,
            tune_selection=selection,
            basis_ablation_result=_accepted_basis_ablation(
                train, calibration, selection
            ),
            compute_backend="cpu",
            feature_layers=("core",),
        )


def inference_panel():
    panel = Round21InferencePanel.from_development(
        _panel("tune_selection", first_condition=306, condition_count=16)
    )
    payload = worker_module._panel_payload(panel)
    payload.pop("feature_batch_sha256")
    start = 1_800_000_000_000
    payload.update(
        condition_ids=["0x" + "9" * 64] * 16,
        event_start_ms=[start] * 16,
        decision_time_ms=[start + 30_000 + index * 1_000 for index in range(16)],
    )
    return Round21InferencePanel.create(**payload)


async def wait_ready(worker, task):
    async with asyncio.timeout(20):
        while worker._connection is None:
            if task.done():
                await task
                pytest.fail("worker exited before readiness")
            await asyncio.sleep(0.02)


def test_target_free_panel_wire_roundtrip_preserves_exact_precision():
    panel = inference_panel()
    encoded = worker_module._encode(worker_module._panel_payload(panel))
    rebuilt = worker_module._panel_from_payload(json.loads(encoded))
    assert rebuilt.feature_batch_sha256 == panel.feature_batch_sha256
    for name in worker_module._PANEL_ARRAYS:
        assert np.array_equal(getattr(rebuilt, name), getattr(panel, name))
    payload = worker_module._panel_payload(panel)
    payload["labels"] = [1] * 16
    with pytest.raises(ValueError, match="panel fields differ"):
        worker_module._panel_from_payload(payload)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b'{"value":1,"value":2}', "duplicate JSON keys"),
        (b'{"value":NaN}', "contains NaN"),
    ],
)
def test_worker_rejects_ambiguous_json_frames(body, message):
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendall(worker_module._MESSAGE.pack(1, len(body)) + body)
        with pytest.raises(ValueError, match=message):
            worker_module._receive_message(receiver)


def test_owned_round21_worker_preserves_native_hull_and_backend(model_artifact):
    async def exercise():
        reference = compile_round21_probability_predictor(
            model_artifact, population_layer="core"
        )
        worker = worker_module.PolymarketRound21InferenceWorker(
            model_artifact, population_layer="core"
        )
        assert worker.tcn_runtime_backend_kind == "pending"
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        try:
            await wait_ready(worker, task)
            assert worker._payload == b""
            panel = inference_panel()
            for rows in (1, 8, 16):
                payload = worker_module._panel_payload(panel)
                for name in worker_module._PANEL_ARRAYS:
                    payload[name] = payload[name][:rows]
                payload.pop("feature_batch_sha256")
                selected = Round21InferencePanel.create(**payload)
                expected = reference.predict(selected)
                actual = await asyncio.to_thread(worker.predict, selected)
                assert actual.prediction_sha256 == expected.prediction_sha256
                for name in worker_module._BATCH_ARRAYS:
                    assert np.array_equal(
                        getattr(actual, name), getattr(expected, name)
                    )
            assert (
                worker.tcn_runtime_backend_kind
                == reference.tcn_runtime_backend_kind
                == "cpu"
            )
            assert (
                worker.tcn_training_backend_kind == reference.tcn_training_backend_kind
            )
            assert worker.tcn_backend_substituted == reference.tcn_backend_substituted
            assert worker.tcn_accelerator_fallback == reference.tcn_accelerator_fallback
            assert worker.trading_authority is False
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=3)
        with pytest.raises(RuntimeError, match="unavailable"):
            worker.predict(panel)

    asyncio.run(exercise())


def test_startup_deadline_reaps_the_owned_child(model_artifact, monkeypatch):
    async def exercise():
        worker = worker_module.PolymarketRound21InferenceWorker(
            model_artifact,
            population_layer="core",
            startup_timeout_seconds=0.25,
        )
        children = []
        original = process_module.subprocess.Popen

        def launch(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            return process

        monkeypatch.setattr(process_module.subprocess, "Popen", launch)
        monkeypatch.setattr(
            worker,
            "_command",
            lambda _descriptor: [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import time; time.sleep(60)",
            ],
        )
        with pytest.raises(TimeoutError, match="startup deadline expired"):
            await asyncio.wait_for(worker.run(asyncio.Event()), timeout=3)
        assert len(children) == 1 and children[0].poll() is not None
        assert worker._connection is None and worker._process is None
        assert not worker.available

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("fault", "error_type", "message", "timeout_seconds"),
    [
        ("wrong_id", ValueError, "late or misbound", 1.5),
        ("wrong_batch", ValueError, "provenance differs", 1.5),
        ("stalled", TimeoutError, None, 0.3),
    ],
)
def test_reply_fault_revokes_and_reaps_round21_worker(
    model_artifact, monkeypatch, fault, error_type, message, timeout_seconds
):
    async def exercise():
        worker = worker_module.PolymarketRound21InferenceWorker(
            model_artifact,
            population_layer="core",
            timeout_seconds=timeout_seconds,
        )
        command = worker._command
        script = """
import sys
import time

sys.path.insert(0, sys.argv[1])
from simple_ai_trading import polymarket_round21_worker as module

original = module._batch_payload


def corrupt(batch):
    body = original(batch)
    if sys.argv[3] == "wrong_batch":
        body["feature_batch_sha256"] = "a" * 64
        body.pop("prediction_sha256")
        return original(module.Round21ProbabilityBatch.create(**body))
    return body


module._batch_payload = corrupt
if sys.argv[3] == "stalled":
    original_compile = module.compile_round21_probability_predictor

    def compile_with_stall(*args, **kwargs):
        predictor = original_compile(*args, **kwargs)
        original_predict = predictor.predict

        def stall(panel):
            time.sleep(60)
            return original_predict(panel)

        predictor.predict = stall
        return predictor

    module.compile_round21_probability_predictor = compile_with_stall
if sys.argv[3] == "wrong_id":
    original_header = module._MESSAGE

    class WrongHeader:
        size = original_header.size
        unpack = original_header.unpack

        @staticmethod
        def pack(request_id, size):
            return original_header.pack(request_id + 1, size)

    module._MESSAGE = WrongHeader()
module._worker(int(sys.argv[2]))
"""
        monkeypatch.setattr(
            worker,
            "_command",
            lambda descriptor: [
                *command(descriptor)[:4],
                script,
                *command(descriptor)[5:],
                fault,
            ],
        )
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        try:
            await wait_ready(worker, task)
            with pytest.raises(error_type, match=message):
                await asyncio.to_thread(worker.predict, inference_panel())
            assert not worker.available
            with pytest.raises(RuntimeError, match="worker failed"):
                await asyncio.wait_for(asyncio.shield(task), timeout=3)
            assert worker._connection is None
            assert worker.available is False
        finally:
            stop.set()
            if not task.done():
                await asyncio.wait_for(task, timeout=3)
            elif not task.cancelled():
                task.exception()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["startup", "inference"])
def test_persisted_stop_reaps_round21_before_owned_recovery(
    model_artifact, tmp_path, monkeypatch, phase
):
    async def exercise():
        worker = worker_module.PolymarketRound21InferenceWorker(
            model_artifact,
            population_layer="core",
            timeout_seconds=3,
        )
        marker = tmp_path / "inference-entered"
        finished = tmp_path / "inference-finished"
        command = worker._command
        script = """
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from simple_ai_trading import polymarket_round21_worker as module

assert "TRADING_WORKER_TEST_SECRET" not in os.environ
original_compile = module.compile_round21_probability_predictor


def stall():
    Path(sys.argv[3]).touch()
    time.sleep(60)
    Path(sys.argv[4]).touch()


def compile_with_stall(*args, **kwargs):
    if sys.argv[5] == "startup":
        stall()
    predictor = original_compile(*args, **kwargs)
    original_predict = predictor.predict

    def predict(panel):
        stall()
        return original_predict(panel)

    predictor.predict = predict
    return predictor


module.compile_round21_probability_predictor = compile_with_stall
module._worker(int(sys.argv[2]))
"""
        monkeypatch.setattr(
            worker,
            "_command",
            lambda descriptor: [
                *command(descriptor)[:4],
                script,
                *command(descriptor)[5:],
                str(marker),
                str(finished),
                phase,
            ],
        )
        children = []
        original_popen = process_module.subprocess.Popen

        def launch(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            children.append(process)
            return process

        monkeypatch.setenv("TRADING_WORKER_TEST_SECRET", "synthetic sentinel")
        monkeypatch.setattr(process_module.subprocess, "Popen", launch)
        returned = Event()

        def decide(**_kwargs):
            if worker._connection is None:
                return PolymarketAutonomousDecision()
            try:
                worker.predict(inference_panel())
                return PolymarketAutonomousDecision()
            finally:
                returned.set()

        supervisor = _supervisor(
            tmp_path, provider=SimpleNamespace(decide=decide), markets=(_market(),)
        )
        supervisor.forecast_worker = worker
        supervisor.decision_timeout_seconds = 3
        owned = {OWNED_MARKET_ID}
        recovery = asyncio.Event()
        supervisor._owned_market_ids = lambda: set(owned)

        async def close_owned():
            assert len(children) == 1 and children[0].poll() is not None
            assert returned.is_set() is (phase == "inference")
            assert not finished.exists()
            owned.clear()
            recovery.set()
            return True

        supervisor._close_owned = close_owned
        control = PolymarketRuntimeControl(supervisor.ledger.path)
        control.provision_storage()
        assert control.complete_stale_stop(exposure_closed=True)
        lease = control.acquire()
        supervisor.durable_control_service = PolymarketRuntimeControlService(
            control,
            lease_id=lease,
            stop_poll_interval_seconds=0.1,
        )
        task = asyncio.create_task(supervisor.run())
        try:
            async with asyncio.timeout(20):
                while not marker.exists():
                    if task.done():
                        await task
                        pytest.fail("supervisor exited before numerical inference")
                    await asyncio.sleep(0.02)
            control.request_stop(reason="synthetic_inference_stop")
            assert control.snapshot().state == "stop_requested"
            await asyncio.wait_for(recovery.wait(), timeout=2)
        finally:
            supervisor.request_stop()
            await asyncio.wait_for(task, timeout=4)
        assert supervisor.snapshot().stop_completed

    asyncio.run(exercise())
