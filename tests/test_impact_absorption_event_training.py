from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from simple_ai_trading.impact_absorption_event_dataset import (  # noqa: E402
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_sequence import (  # noqa: E402
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)
from simple_ai_trading.impact_absorption_event_training import (  # noqa: E402
    Round74EventTrainingConfig,
    load_round74_pretest_policy,
    train_and_seal_round74_pretest_policy,
)


WALL_NS = 1_800_000_000_000_000_000
PURGE_NS = 300_000_000_000


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _batch(
    role: str,
    *,
    start_wall_ns: int,
    identity: int,
    rows: int = 2,
) -> Round74EventTrainingBatch:
    generator = np.random.default_rng(7400 + identity)
    features = generator.normal(
        size=(
            rows,
            ROUND74_EVENT_SEQUENCE_LENGTH,
            len(ROUND74_EVENT_FEATURE_NAMES),
        )
    ).astype(np.float32)
    features[:, :, :8] = 0.0
    for row in range(rows):
        features[row, :, row % 5] = 1.0
        features[row, :, 5 + row % 3] = 1.0
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    regime_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
    )
    payoff = generator.normal(size=action_shape).astype(np.float32)
    adverse_excursion = np.abs(generator.normal(size=action_shape)).astype(np.float32)
    adverse = generator.integers(0, 2, size=action_shape).astype(np.float32)
    unpredictable = generator.integers(
        0,
        2,
        size=regime_shape,
    ).astype(np.float32)
    eligibility = np.ones(action_shape, dtype=np.float32)
    eligibility[0, 0, 0] = 0.0
    actual_entry = np.full(action_shape, 10, dtype=np.int64)
    actual_exit = np.full(action_shape, 20, dtype=np.int64)
    actual_entry[0, 0, 0] = -1
    actual_exit[0, 0, 0] = -1
    payoff[0, 0, 0] = 0.0
    adverse_excursion[0, 0, 0] = 0.0
    adverse[0, 0, 0] = 0.0
    times = np.arange(rows, dtype=np.int64) * 1_000_000_000
    batch = Round74EventTrainingBatch(
        role=role,
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=tuple(f"{identity:032x}" for _ in range(rows)),
        symbol=tuple(("BTCUSDT", "ETHUSDT", "SOLUSDT")[row % 3] for row in range(rows)),
        decision_monotonic_ns=_readonly(times.copy()),
        decision_wall_ns=_readonly(times + start_wall_ns),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.zeros(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{identity * 100 + row:064x}" for row in range(rows)),
        target_context_sha256=tuple("3" * 64 for _ in range(rows)),
        feature_values=_readonly(features),
        actual_entry_monotonic_ns=_readonly(actual_entry),
        actual_exit_monotonic_ns=_readonly(actual_exit),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(adverse_excursion),
        adverse_selection=_readonly(adverse),
        regime_unpredictability=_readonly(unpredictable),
        action_eligibility=_readonly(eligibility),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float32)
        ),
    )
    batch.validate()
    return batch


def _config() -> Round74EventTrainingConfig:
    return Round74EventTrainingConfig(
        seeds=(7411,),
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=2,
        minimum_role_rows=2,
    )


def _write_rehashed_policy(path: Path, policy: dict[str, object]) -> Path:
    policy.pop("policy_sha256", None)
    encoded = json.dumps(
        policy,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()
    policy["policy_sha256"] = digest
    target = path / f"round74-pretest-policy-{digest}.json"
    target.write_text(
        json.dumps(
            policy,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    return target


def test_round74_trainer_rejects_test_data_before_backend_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    import simple_ai_trading.impact_absorption_event_training as training

    def unexpected_backend(_value: object) -> object:
        raise AssertionError("backend resolution must follow data validation")

    monkeypatch.setattr(training, "resolve_backend", unexpected_backend)
    test_batch = _batch(
        "test",
        start_wall_ns=WALL_NS,
        identity=9,
    )
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )

    with pytest.raises(ValueError, match="rejects 'test' data"):
        train_and_seal_round74_pretest_policy(
            [test_batch],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )


def test_round74_trainer_requires_order_purge_and_shared_context(
    tmp_path: object,
) -> None:
    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + 2_000_000_000,
        identity=2,
    )
    with pytest.raises(ValueError, match="minimum purge"):
        train_and_seal_round74_pretest_policy(
            [training],
            [tuning],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )

    changed_context = replace(
        _batch(
            "tuning",
            start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
            identity=2,
        ),
        target_context_sha256=("4" * 64, "4" * 64),
    )
    changed_context.validate()
    with pytest.raises(ValueError, match="target context"):
        train_and_seal_round74_pretest_policy(
            [training],
            [changed_context],
            output_directory=tmp_path,
            compute_backend="cpu",
            config=_config(),
        )


def test_round74_pretest_policy_is_safe_hash_bound_and_non_authoritative(
    tmp_path: Path,
) -> None:
    training = _batch("training", start_wall_ns=WALL_NS, identity=1)
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )
    artifact = train_and_seal_round74_pretest_policy(
        [training],
        [tuning],
        output_directory=tmp_path,
        compute_backend="cpu",
        config=_config(),
    )
    model, policy = load_round74_pretest_policy(artifact.policy_path)

    assert model.candidate_id == artifact.selected_candidate_id
    assert policy["policy_sha256"] == artifact.policy_sha256
    assert policy["model_artifact"]["sha256"] == artifact.model_sha256
    assert policy["model_artifact"]["pickle_permitted"] is False
    assert policy["development_data"]["test_batches_consumed"] == 0
    assert policy["backend"]["kind"] == "cpu"
    assert set(policy["candidate_panel"]) == {
        "event_pooling_mlp",
        "causal_event_tcn",
    }
    assert policy["selection"]["backtest_metric_used_for_selection"] is False
    assert all(
        policy["authority"][name] is False
        for name in (
            "sealed_test_evaluated",
            "representative_market_evidence_claim",
            "financial_edge_tested",
            "profitability_claim",
            "ai_uplift_claim",
            "paper_trading_authority",
            "testnet_trading_authority",
            "live_trading_authority",
        )
    )
    tampered = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    tampered["authority"]["profitability_claim"] = True
    tampered_path = _write_rehashed_policy(tmp_path, tampered)
    with pytest.raises(ValueError, match="overstates authority"):
        load_round74_pretest_policy(tampered_path)

    changed_source = json.loads(artifact.policy_path.read_text(encoding="ascii"))
    changed_source["source_binding"]["model_module_sha256"] = "f" * 64
    changed_source_path = _write_rehashed_policy(tmp_path, changed_source)
    with pytest.raises(ValueError, match="source binding differs"):
        load_round74_pretest_policy(changed_source_path)

    artifact.model_path.write_bytes(artifact.model_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="model artifact differs"):
        load_round74_pretest_policy(artifact.policy_path)
