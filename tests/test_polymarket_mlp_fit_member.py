from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from simple_ai_trading import polymarket_mlp as mlp


_INPUT_SHA256 = "a88b58b8e0bfc9165f87c3b8977028d0366c1e0ae13392136d4def96f34920f3"
_REFERENCE_PREDICTION = np.asarray(
    [
        float.fromhex("0x1.9eac50bc7960fp-2"),
        float.fromhex("0x1.c6832c3dfd63ap-2"),
        float.fromhex("0x1.ecd9a0417ca76p-2"),
        float.fromhex("0x1.0bfdf69da72fdp-1"),
        float.fromhex("0x1.26243753952bdp-1"),
        float.fromhex("0x1.43ca1fb630cf7p-1"),
    ],
    dtype=np.float64,
)
_EXPECTED_PROGRESS_EPOCHS = [1, 5, 10, 15, 20, 25, 30, 35, 40]


def _sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _fit_inputs() -> dict[str, np.ndarray]:
    return {
        "training_features": np.linspace(
            -1.0,
            1.0,
            12 * 39,
            dtype=np.float64,
        ).reshape(12, 39),
        "training_labels": np.asarray([0.0, 1.0] * 6, dtype=np.float64),
        "training_weights": np.linspace(0.5, 1.5, 12, dtype=np.float64),
        "validation_features": np.linspace(
            -0.9,
            0.9,
            6 * 39,
            dtype=np.float64,
        ).reshape(6, 39),
        "validation_labels": np.asarray([0.0, 1.0] * 3, dtype=np.float64),
    }


def test_fit_member_preserves_training_contract_with_backend_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fit_inputs()
    progress: list[tuple[str, dict[str, object]]] = []
    torch, device, _backend = mlp._torch_runtime("cpu")
    monkeypatch.setattr(mlp.time, "perf_counter", lambda: 1.0)

    member, replay_drift = mlp._fit_member(
        torch,
        device,
        seed=4701,
        training_features=inputs["training_features"],
        training_labels=inputs["training_labels"],
        training_weights=inputs["training_weights"],
        validation_features=inputs["validation_features"],
        validation_labels=inputs["validation_labels"],
        progress=lambda phase, payload: progress.append((phase, dict(payload))),
        run_kind="contract",
    )
    prediction = member.predict_standardized(inputs["validation_features"])
    epoch_progress = [item for item in progress if item[0] == "polymarket_mlp_epoch"]
    serialized_inputs = {key: value.tolist() for key, value in inputs.items()}
    trace_payload = [item.asdict() for item in member.trace]

    assert _sha256(serialized_inputs) == _INPUT_SHA256
    assert member.member_sha256 == _sha256(member.identity_payload())
    assert member.identity_payload()["trace_sha256"] == _sha256(trace_payload)
    assert member.best_epoch == 20
    assert member.epochs_ran == 40
    assert 0.0 <= replay_drift <= mlp.POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE
    np.testing.assert_allclose(
        prediction,
        _REFERENCE_PREDICTION,
        rtol=0.0,
        atol=mlp.POLYMARKET_MLP_REPRODUCIBILITY_TOLERANCE,
    )
    assert [payload["epoch"] for _, payload in epoch_progress] == (
        _EXPECTED_PROGRESS_EPOCHS
    )
    for phase, payload in epoch_progress:
        epoch = int(payload["epoch"])
        trace = member.trace[epoch - 1]
        assert phase == "polymarket_mlp_epoch"
        assert payload["run_kind"] == "contract"
        assert payload["seed"] == 4701
        assert payload["training_loss"] == trace.training_loss
        assert payload["validation_log_loss"] == trace.validation_log_loss
        assert np.isfinite(float(payload["best_validation_log_loss"]))
        assert int(payload["stale_epochs"]) >= 0


def test_fit_member_rejects_non_finite_training_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fit_inputs()
    inputs["training_weights"][0] = float("nan")
    torch, device, _backend = mlp._torch_runtime("cpu")
    monkeypatch.setattr(mlp.time, "perf_counter", lambda: 1.0)

    with pytest.raises(RuntimeError, match="training loss is non-finite"):
        mlp._fit_member(
            torch,
            device,
            seed=4701,
            training_features=inputs["training_features"],
            training_labels=inputs["training_labels"],
            training_weights=inputs["training_weights"],
            validation_features=inputs["validation_features"],
            validation_labels=inputs["validation_labels"],
        )


def test_fit_member_emits_rate_limited_batch_heartbeat() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    context = cast(
        mlp._PolymarketMLPFitContext,
        SimpleNamespace(
            progress=lambda phase, payload: events.append((phase, dict(payload))),
            run_kind="contract",
            seed=4701,
        ),
    )
    state = cast(
        mlp._PolymarketMLPFitState,
        SimpleNamespace(last_batch_heartbeat=1.0),
    )

    mlp._emit_member_batch_heartbeat(
        context,
        state,
        epoch=3,
        offset=8,
        rows=4,
        order_size=16,
        heartbeat=31.0,
    )
    mlp._emit_member_batch_heartbeat(
        context,
        state,
        epoch=3,
        offset=12,
        rows=4,
        order_size=16,
        heartbeat=50.0,
    )

    assert events == [
        (
            "polymarket_mlp_batch",
            {
                "run_kind": "contract",
                "seed": 4701,
                "epoch": 3,
                "rows_complete": 12,
                "rows_total": 16,
            },
        )
    ]
    assert state.last_batch_heartbeat == 31.0
