"""Run the deterministic Round 74 trainer contract on DirectML.

This is a compute, serialization, and governance preflight. Its constructed
two-row tensors are not market data and its losses have no financial meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404
import tempfile
from pathlib import Path

import numpy as np

from simple_ai_trading.impact_absorption_event_dataset import (
    Round74EventTrainingBatch,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)
from simple_ai_trading.impact_absorption_event_training import (
    Round74EventTrainingConfig,
    load_round74_pretest_policy,
    train_and_seal_round74_pretest_policy,
)


WALL_NS = 1_800_000_000_000_000_000
PURGE_NS = 310_000_000_000


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(
    role: str,
    *,
    start_wall_ns: int,
    identity: int,
) -> Round74EventTrainingBatch:
    rows = 2
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
    result = Round74EventTrainingBatch(
        role=role,
        partition_sha256="1" * 64,
        scaler_sha256="2" * 64,
        run_id=tuple(f"{identity:032x}" for _ in range(rows)),
        symbol=("BTCUSDT", "ETHUSDT"),
        decision_monotonic_ns=_readonly(times.copy()),
        decision_wall_ns=_readonly(times + start_wall_ns),
        endpoint_frame_index=_readonly(np.arange(rows, dtype=np.int64)),
        endpoint_message_index=_readonly(np.zeros(rows, dtype=np.int64)),
        anchor_index=_readonly(np.arange(rows, dtype=np.int64)),
        sample_sha256=tuple(f"{identity * 100 + row:064x}" for row in range(rows)),
        target_context_sha256=("3" * 64, "3" * 64),
        test_access_sha256=("", ""),
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
    result.validate()
    return result


def _git_commit(repository: Path) -> str:
    result = subprocess.run(  # nosec B603
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="ascii",
        errors="strict",
    )
    commit = result.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Round 74 preflight git identity differs")
    return commit


def _run(repository: Path) -> dict[str, object]:
    training = _batch(
        "training",
        start_wall_ns=WALL_NS,
        identity=1,
    )
    tuning = _batch(
        "tuning",
        start_wall_ns=WALL_NS + PURGE_NS + 2_000_000_000,
        identity=2,
    )
    config = Round74EventTrainingConfig(
        maximum_epochs=1,
        early_stopping_patience=1,
        minibatch_rows=2,
        minimum_role_rows=2,
    )
    with tempfile.TemporaryDirectory(
        prefix="round74-training-preflight-"
    ) as raw_output:
        output = Path(raw_output)
        artifact = train_and_seal_round74_pretest_policy(
            [training],
            [tuning],
            output_directory=output,
            compute_backend="directml",
            config=config,
        )
        _model, policy = load_round74_pretest_policy(artifact.policy_path)
        panel = policy["candidate_panel"]
        selected = policy["model_artifact"]
        backend = policy["backend"]
        result: dict[str, object] = {
            "schema_version": "round-074-event-training-preflight-run-v1",
            "execution_git_commit": _git_commit(repository),
            "backend": {
                key: backend[key]
                for key in (
                    "requested",
                    "kind",
                    "device",
                    "vendor",
                    "selection",
                    "accelerated",
                    "torch_version",
                    "torch_directml_version",
                    "safetensors_version",
                    "deterministic_algorithms_requested",
                    "warning_count",
                    "cpu_fallback_warning_count",
                )
            },
            "input_contract": {
                "training_batch_sha256": training.batch_sha256,
                "tuning_batch_sha256": tuning.batch_sha256,
                "training_eligible_action_targets": int(
                    training.action_eligibility.sum()
                ),
                "tuning_eligible_action_targets": int(tuning.action_eligibility.sum()),
                "training_eligible_regime_targets": int(
                    training.regime_unpredictability_eligibility.sum()
                ),
                "tuning_eligible_regime_targets": int(
                    tuning.regime_unpredictability_eligibility.sum()
                ),
                "candidate_ids": list(config.candidate_ids),
                "seeds": list(config.seeds),
                "epochs": config.maximum_epochs,
                "minibatch_rows": config.minibatch_rows,
            },
            "result": {
                "policy_sha256": artifact.policy_sha256,
                "model_sha256": artifact.model_sha256,
                "model_byte_count": int(selected["byte_count"]),
                "prediction_sha256": selected["prediction_sha256"],
                "selected_candidate_id": artifact.selected_candidate_id,
                "candidate_tuning_proper_loss": {
                    candidate_id: panel[candidate_id]["ensemble_tuning_metrics"]["loss"]
                    for candidate_id in config.candidate_ids
                },
                "peer_best_tuning_proper_loss": {
                    candidate_id: [
                        peer["best_tuning_metrics"]["loss"]
                        for peer in panel[candidate_id]["peer_reports"]
                    ]
                    for candidate_id in config.candidate_ids
                },
            },
            "temporary_artifact_count_before_cleanup": len(tuple(output.iterdir())),
        }
    result["temporary_directory_removed"] = not Path(raw_output).exists()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    result = _run(repository)
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
