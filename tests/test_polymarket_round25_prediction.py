from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import simple_ai_trading.polymarket_round25_prediction as prediction_module
from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CALIBRATION_END_MS,
)
from simple_ai_trading.polymarket_round25_evaluation import (
    Round25SelectionAccessStore,
    create_round25_prediction_panel,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    Round25JointFeatureSnapshot,
)
from simple_ai_trading.polymarket_round25_model_ledger import (
    POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256,
)
from simple_ai_trading.polymarket_round25_prediction import (
    POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION,
    Round25PreparedPrediction,
    _derived_prediction_source_sha256,
    freeze_round25_prepared_prediction,
    load_round25_prepared_prediction,
    prepare_round25_target_free_prediction,
    write_round25_prepared_prediction,
)
from simple_ai_trading.polymarket_round25_sequence import (
    POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


MODEL_LEDGER_SHA256 = "a" * 64
SOURCE_RECEIPT_AUDIT_SHA256 = "b" * 64


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _prepared_prediction() -> Round25PreparedPrediction:
    raw_artifacts = tuple(
        (
            candidate_id,
            hashlib.sha256(f"model:{candidate_id}".encode("ascii")).hexdigest(),
        )
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS
    )
    prediction_sources = {
        candidate_id: _derived_prediction_source_sha256(
            candidate_id=candidate_id,
            model_artifact_sha256=model_sha256,
            model_ledger_sha256=MODEL_LEDGER_SHA256,
            source_receipt_audit_sha256=SOURCE_RECEIPT_AUDIT_SHA256,
        )
        for candidate_id, model_sha256 in raw_artifacts
    }
    condition_ids: list[str] = []
    event_starts: list[int] = []
    decision_times: list[int] = []
    source_hashes: list[str] = []
    market_prior: list[float] = []
    condition_count = 400
    for condition_index in range(condition_count):
        condition_id = "0x" + format(condition_index + 1, "064x")
        event_start = (
            POLYMARKET_ROUND25_CALIBRATION_END_MS
            + 300_000
            + condition_index * 300_000
        )
        for phase in range(4):
            for endpoint in range(4):
                decision = event_start + phase * 75_000 + (endpoint + 1) * 250
                condition_ids.append(condition_id)
                event_starts.append(event_start)
                decision_times.append(decision)
                source_hashes.append(hashlib.sha256(
                    f"source:{condition_id}:{decision}".encode("ascii")
                ).hexdigest())
                market_prior.append(0.45 + phase * 0.02 + endpoint * 0.001)
    candidate_probabilities = {
        candidate_id: tuple(
            probability
            if candidate_index == 0
            else min(0.99, max(0.01, probability + candidate_index * 0.001))
            for probability in market_prior
        )
        for candidate_index, candidate_id in enumerate(
            POLYMARKET_ROUND25_CANDIDATE_IDS
        )
    }
    panel = create_round25_prediction_panel(
        row_condition_ids=condition_ids,
        event_start_ms=event_starts,
        decision_time_ms=decision_times,
        feature_source_chain_sha256=source_hashes,
        market_prior_probability=market_prior,
        candidate_probabilities=candidate_probabilities,
        candidate_source_artifact_sha256=prediction_sources,
    )
    inference_hashes = tuple(
        hashlib.sha256(f"inference:{index}".encode("ascii")).hexdigest()
        for index in range(condition_count)
    )
    identity = {
        "candidate_model_artifact_sha256": [
            {"candidate_id": candidate_id, "sha256": digest}
            for candidate_id, digest in raw_artifacts
        ],
        "condition_count": condition_count,
        "model_ledger_contract_sha256": POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256,
        "model_ledger_sha256": MODEL_LEDGER_SHA256,
        "prediction_panel_sha256": panel.panel_sha256,
        "schema_version": POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION,
        "selection_target_accessed": False,
        "sequence_inference_batch_sha256": list(inference_hashes),
        "source_receipt_audit_sha256": SOURCE_RECEIPT_AUDIT_SHA256,
        "target_free_sequence_contract_sha256": (
            POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
        ),
        "trading_authority": False,
    }
    return Round25PreparedPrediction(
        panel=panel,
        model_ledger_sha256=MODEL_LEDGER_SHA256,
        source_receipt_audit_sha256=SOURCE_RECEIPT_AUDIT_SHA256,
        candidate_model_artifact_sha256=raw_artifacts,
        sequence_inference_batch_sha256=inference_hashes,
        prepared_sha256=_canonical_sha256(identity),
    )


def _selection_snapshot(
    *,
    condition_id: str,
    event_start_ms: int,
    decision_time_ms: int,
    row_index: int,
) -> Round25JointFeatureSnapshot:
    twap_sha256 = hashlib.sha256(
        f"twap:{condition_id}:{decision_time_ms}".encode("ascii")
    ).hexdigest()
    clob_sha256 = hashlib.sha256(
        f"clob:{condition_id}:{decision_time_ms}".encode("ascii")
    ).hexdigest()
    source_sha256 = _canonical_sha256({
        "clob_source_chain_sha256": clob_sha256,
        "condition_id": condition_id,
        "decision_time_ms": decision_time_ms,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "twap_source_chain_sha256": twap_sha256,
    })
    values = [0.0] * len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    values[0] = float(row_index)
    for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES):
        if name.endswith("_available"):
            values[index] = 1.0
    return Round25JointFeatureSnapshot(
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        decision_time_ms=decision_time_ms,
        available=True,
        reasons=(),
        market_prior_probability=0.45 + (row_index % 16) * 0.002,
        values=tuple(values),
        source_chain_sha256=source_sha256,
        twap_source_chain_sha256=twap_sha256,
        clob_source_chain_sha256=clob_sha256,
        maximum_receipt_ms=decision_time_ms,
    )


def _selection_snapshots() -> tuple[Round25JointFeatureSnapshot, ...]:
    rows = []
    for condition_index in range(400):
        condition_id = "0x" + format(condition_index + 1, "064x")
        event_start = (
            POLYMARKET_ROUND25_CALIBRATION_END_MS
            + 300_000
            + condition_index * 300_000
        )
        for phase in range(4):
            for endpoint in range(4):
                row_index = phase * 4 + endpoint
                rows.append(_selection_snapshot(
                    condition_id=condition_id,
                    event_start_ms=event_start,
                    decision_time_ms=(
                        event_start + phase * 75_000 + (endpoint + 1) * 250
                    ),
                    row_index=row_index,
                ))
    return tuple(rows)


def test_prepared_prediction_round_trip_preserves_exact_panel(tmp_path: Path) -> None:
    prepared = _prepared_prediction()
    path = tmp_path / "prepared-selection.json"

    write_round25_prepared_prediction(path, prepared)
    loaded = load_round25_prepared_prediction(path)

    assert loaded.prepared_sha256 == prepared.prepared_sha256
    assert loaded.panel.panel_sha256 == prepared.panel.panel_sha256
    assert loaded.serialized_payload() == prepared.serialized_payload()
    assert loaded.selection_target_accessed is False
    assert loaded.trading_authority is False
    assert write_round25_prepared_prediction(path, prepared) == path


def test_prepared_prediction_is_validated_before_selection_lock(
    tmp_path: Path,
) -> None:
    prepared = _prepared_prediction()
    store = Round25SelectionAccessStore(tmp_path / "selection-access.sqlite3")

    frozen_sha256 = freeze_round25_prepared_prediction(
        store=store,
        prepared=prepared,
        one_use_claim_sha256="c" * 64,
    )

    assert frozen_sha256 == prepared.panel.panel_sha256
    object.__setattr__(prepared, "prepared_sha256", "f" * 64)
    with pytest.raises(ValueError, match="prepared target-free prediction differs"):
        freeze_round25_prepared_prediction(
            store=Round25SelectionAccessStore(
                tmp_path / "second-selection-access.sqlite3"
            ),
            prepared=prepared,
            one_use_claim_sha256="d" * 64,
        )


def test_prediction_coordinator_joins_all_six_target_free_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_artifacts = tuple(
        (
            candidate_id,
            hashlib.sha256(f"model:{candidate_id}".encode("ascii")).hexdigest(),
        )
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS
    )
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)

    class FakeLedger:
        ledger_sha256 = MODEL_LEDGER_SHA256
        phase_isotonic = object()
        logistic_residual = SimpleNamespace(
            center=(0.0,) * width,
            scale=(1.0,) * width,
        )
        lightgbm_residuals = (
            SimpleNamespace(offset=0.003),
            SimpleNamespace(offset=0.004),
        )
        tcn_ensemble = object()

        def validated(self) -> FakeLedger:
            return self

        def candidate_artifact_sha256(self) -> tuple[tuple[str, str], ...]:
            return raw_artifacts

    class FakeTreeRuntime:
        def __init__(self, artifact: object) -> None:
            self._offset = float(getattr(artifact, "offset"))

        def predict_probabilities(
            self,
            _features: object,
            prior: object,
        ) -> tuple[float, ...]:
            return tuple(float(value) + self._offset for value in prior)

    class FakeTCNRuntime:
        def __init__(self, _artifact: object, *, compute_backend: str) -> None:
            assert compute_backend == "directml"

        def predict_probabilities(
            self,
            _sequences: object,
            prior: object,
        ) -> tuple[float, ...]:
            return tuple(float(value) + 0.005 for value in prior)

    monkeypatch.setattr(prediction_module, "Round25ModelLedger", FakeLedger)
    monkeypatch.setattr(
        prediction_module,
        "predict_round25_phase_isotonic_probability",
        lambda _artifact, **values: values["market_prior_probability"] + 0.001,
    )
    monkeypatch.setattr(
        prediction_module,
        "predict_round25_logistic_residual_probability",
        lambda _artifact, **values: values["market_prior_probability"] + 0.002,
    )
    monkeypatch.setattr(
        prediction_module,
        "Round25CompiledLightGBM",
        FakeTreeRuntime,
    )
    monkeypatch.setattr(
        prediction_module,
        "Round25CompiledTCNEnsemble",
        FakeTCNRuntime,
    )

    prepared = prepare_round25_target_free_prediction(
        ledger=FakeLedger(),
        snapshots=_selection_snapshots(),
        source_receipt_audit_sha256=SOURCE_RECEIPT_AUDIT_SHA256,
        tcn_backend="directml",
    )

    assert len(prepared.sequence_inference_batch_sha256) == 400
    assert len(prepared.panel.row_condition_ids) == 6_400
    assert tuple(
        prediction.candidate_id
        for prediction in prepared.panel.candidate_predictions
    ) == POLYMARKET_ROUND25_CANDIDATE_IDS
    assert prepared.panel.candidate_predictions[0].probabilities[0] == 0.45
    assert prepared.panel.candidate_predictions[-1].probabilities[0] == 0.455
    assert prepared.selection_target_accessed is False
    assert prepared.trading_authority is False


def test_prepared_prediction_rejects_probability_tampering(tmp_path: Path) -> None:
    prepared = _prepared_prediction()
    path = tmp_path / "prepared-selection.json"
    write_round25_prepared_prediction(path, prepared)
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["panel"]["candidate_predictions"][1]["probabilities"][0] = 0.99
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="ascii")

    with pytest.raises(ValueError, match="panel|prepared prediction"):
        load_round25_prepared_prediction(path)


def test_prepared_prediction_rejects_candidate_source_substitution() -> None:
    prepared = _prepared_prediction()
    prediction = prepared.panel.candidate_predictions[1]
    object.__setattr__(prediction, "source_artifact_sha256", "f" * 64)

    with pytest.raises(ValueError, match="prediction panel|candidate prediction|prepared"):
        prepared.validated()


def test_prepared_prediction_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"panel":{},"panel":{}}', encoding="ascii")

    with pytest.raises(ValueError, match="duplicate keys"):
        load_round25_prepared_prediction(path)
