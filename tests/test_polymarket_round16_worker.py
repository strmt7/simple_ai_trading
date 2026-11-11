from __future__ import annotations

import asyncio
import sys

import lightgbm as lgb
import numpy as np
import pytest

from simple_ai_trading.polymarket_round16_shadow import (
    ROUND16_FEATURE_NAMES,
    VerifiedRound16ShadowPredictor,
)
from simple_ai_trading.polymarket_round16_worker import PolymarketRound16InferenceWorker


def _predictor(family: str) -> VerifiedRound16ShadowPredictor:
    if family == "lightgbm":
        values = np.arange(32, dtype=np.float32).reshape(-1, 1)
        booster = lgb.train(
            {
                "objective": "binary",
                "num_threads": 1,
                "verbosity": -1,
                "min_data_in_leaf": 2,
            },
            lgb.Dataset(values, label=values[:, 0] >= 16),
            num_boost_round=2,
        )
        model = {"type": "lightgbm", "model_string": booster.model_to_string()}
    else:
        model = {
            "type": "ridge_logistic",
            "mean": [0.25],
            "scale": [0.5],
            "coefficient": [0.7],
            "intercept": -0.3,
        }
    return VerifiedRound16ShadowPredictor(
        candidate={
            "feature_indices": [0],
            "model": model,
            "calibration": {"retained": True, "intercept": -0.1, "slope": 0.9},
        },
        candidate_id="fixture",
        pretest_envelope_sha256="1" * 64,
        evaluation_envelope_sha256="2" * 64,
        dataset_sha256="3" * 64,
        pretest_file_sha256="4" * 64,
        evaluation_file_sha256="5" * 64,
        feature_support={},
        settlement_controls={},
    )


async def _wait_ready(worker, task) -> None:
    async with asyncio.timeout(12):
        while worker._connection is None:
            if task.done():
                await task
                pytest.fail("inference worker returned before readiness")
            await asyncio.sleep(0.02)


@pytest.mark.parametrize("family", ["ridge_logistic", "lightgbm"])
def test_owned_round16_worker_preserves_probability_and_recovers_invalid_input(family):
    async def exercise():
        predictor = _predictor(family)
        worker = PolymarketRound16InferenceWorker(predictor)
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        vector = np.zeros(len(ROUND16_FEATURE_NAMES), dtype=np.float32)
        try:
            await _wait_ready(worker, task)
            for value in (1.25, 17.5, 24.125):
                vector[0] = value
                expected = predictor.predict_up_probability(vector)
                assert await asyncio.to_thread(worker.predict, vector) == expected
            vector[0] = np.nan
            with pytest.raises(ValueError, match="feature vector"):
                await asyncio.to_thread(worker.predict, vector)
            vector[0] = -1.25
            assert await asyncio.to_thread(
                worker.predict, vector
            ) == predictor.predict_up_probability(vector)
        finally:
            stop.set()
            await asyncio.wait_for(task, timeout=3)
        with pytest.raises(RuntimeError, match="unavailable"):
            worker.predict(vector)

    asyncio.run(exercise())


@pytest.mark.parametrize("fault", ["wrong_request_id", "stalled_response"])
def test_round16_reply_fault_invalidates_worker(monkeypatch, fault):
    async def exercise():
        worker = PolymarketRound16InferenceWorker(
            _predictor("ridge_logistic"), timeout_seconds=0.25
        )
        command = worker._command
        script = """
import struct, sys, time
sys.path.insert(0, sys.argv[1])
from simple_ai_trading import polymarket_round16_worker as module
class FaultedReply:
    @staticmethod
    def pack(request_id, probability):
        if sys.argv[3] == 'stalled_response':
            time.sleep(60)
        return struct.pack('!Qd', request_id + 1, probability)
module._REPLY = FaultedReply()
module._worker(int(sys.argv[2]))
"""
        monkeypatch.setattr(
            worker,
            "_command",
            lambda descriptor: [
                sys.executable,
                "-I",
                "-B",
                "-c",
                script,
                *command(descriptor)[5:],
                fault,
            ],
        )
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop))
        try:
            await _wait_ready(worker, task)
            vector = np.zeros(len(ROUND16_FEATURE_NAMES), dtype=np.float32)
            expected = (
                pytest.raises(TimeoutError)
                if fault == "stalled_response"
                else pytest.raises(RuntimeError, match="response is invalid")
            )
            with expected:
                await asyncio.to_thread(worker.predict, vector)
            with pytest.raises(RuntimeError, match="worker failed"):
                await asyncio.wait_for(asyncio.shield(task), timeout=3)
            assert worker._connection is None
        finally:
            stop.set()
            if not task.done():
                await asyncio.wait_for(task, timeout=3)
            elif not task.cancelled():
                task.exception()

    asyncio.run(exercise())
