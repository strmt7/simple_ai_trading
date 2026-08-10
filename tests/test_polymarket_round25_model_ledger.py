from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round25_model_ledger as ledger_module

from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from simple_ai_trading.polymarket_round25_controls import (
    POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
    POLYMARKET_ROUND25_L2_GRID,
    POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION,
    POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION,
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
    Round25IsotonicPhaseModel,
    Round25L2CalibrationScore,
    Round25LogisticResidualArtifact,
    Round25PhaseIsotonicArtifact,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round25_lightgbm import (
    POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_LIGHTGBM_CONFIGS,
    POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256,
    Round25LightGBMArtifact,
)
from simple_ai_trading.polymarket_round25_model_ledger import (
    POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256,
    POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS,
    POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256,
    POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION,
    create_round25_model_ledger,
    fit_round25_model_ledger_coordinated,
    load_round25_model_ledger,
    write_round25_model_ledger,
)
from simple_ai_trading.polymarket_round25_sequence import (
    POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256,
    round25_feature_transform_sha256,
)
from simple_ai_trading.polymarket_round25_tcn import (
    POLYMARKET_ROUND25_TCN_ARCHITECTURE,
    POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_TCN_SEED_ARTIFACT_SCHEMA_VERSION,
    POLYMARKET_ROUND25_TCN_TRAINING_SEEDS,
    _create_round25_tcn_ensemble_artifact,
    _create_round25_tcn_seed_artifact,
    _model,
    round25_tcn_parameter_count,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


TRAIN_SHA256 = "1" * 64
CALIBRATION_SHA256 = "2" * 64
AUTHORITY_SHA256 = "3" * 64
TRAIN_BATCH_MANIFEST_SHA256 = "4" * 64
CALIBRATION_BATCH_MANIFEST_SHA256 = "5" * 64
SOURCE_COMMIT_OID = "a" * 40


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


def _phase_artifact() -> Round25PhaseIsotonicArtifact:
    models = tuple(
        Round25IsotonicPhaseModel(
            phase=phase,
            x_thresholds=(0.25, 0.75),
            y_thresholds=(0.3, 0.7),
        )
        for phase in range(4)
    )
    payload = {
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_id": "phase-isotonic-market-prior-v1",
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "fit_dataset_sha256": CALIBRATION_SHA256,
        "fit_role": "calibration",
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "phase_models": [model.payload() for model in models],
        "resolution_authority_sha256": AUTHORITY_SHA256,
        "schema_version": POLYMARKET_ROUND25_PHASE_ISOTONIC_SCHEMA_VERSION,
        "trading_authority": False,
    }
    return Round25PhaseIsotonicArtifact(
        fit_dataset_sha256=CALIBRATION_SHA256,
        resolution_authority_sha256=AUTHORITY_SHA256,
        phase_models=models,
        artifact_sha256=_canonical_sha256(payload),
    )


def _logistic_artifact() -> Round25LogisticResidualArtifact:
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    center = (0.0,) * width
    scale = (1.0,) * width
    coefficients = (0.0,) * width
    scores = tuple(
        Round25L2CalibrationScore(
            l2=l2,
            condition_equal_log_loss=0.6 + index * 0.01,
            condition_equal_brier_score=0.2 + index * 0.01,
        )
        for index, l2 in enumerate(POLYMARKET_ROUND25_L2_GRID)
    )
    payload = {
        "calibration_dataset_sha256": CALIBRATION_SHA256,
        "calibration_resolution_authority_sha256": AUTHORITY_SHA256,
        "calibration_scores": [score.payload() for score in scores],
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_id": "l2-logistic-residual-v1",
        "center": list(center),
        "coefficients": list(coefficients),
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "fit_role": "train",
        "intercept": 0.0,
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "residual_logit_bound": POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
        "scale": list(scale),
        "schema_version": POLYMARKET_ROUND25_LOGISTIC_RESIDUAL_SCHEMA_VERSION,
        "selected_l2": scores[0].l2,
        "selection_role": "calibration",
        "trading_authority": False,
        "train_dataset_sha256": TRAIN_SHA256,
        "train_resolution_authority_sha256": AUTHORITY_SHA256,
    }
    return Round25LogisticResidualArtifact(
        train_dataset_sha256=TRAIN_SHA256,
        calibration_dataset_sha256=CALIBRATION_SHA256,
        train_resolution_authority_sha256=AUTHORITY_SHA256,
        calibration_resolution_authority_sha256=AUTHORITY_SHA256,
        center=center,
        scale=scale,
        selected_l2=scores[0].l2,
        intercept=0.0,
        coefficients=coefficients,
        calibration_scores=scores,
        artifact_sha256=_canonical_sha256(payload),
    )


def _lightgbm_artifact(
    config_index: int,
    logistic: Round25LogisticResidualArtifact,
) -> Round25LightGBMArtifact:
    config = POLYMARKET_ROUND25_LIGHTGBM_CONFIGS[config_index]
    model_string = f"fixture-{config.candidate_id}-" + "x" * 96
    model_sha256 = hashlib.sha256(model_string.encode("utf-8")).hexdigest()
    payload = {
        "backend_device": "cpu",
        "backend_kind": "cpu",
        "best_iteration": 1,
        "calibration_condition_equal_brier_score": 0.2 + config_index * 0.01,
        "calibration_condition_equal_log_loss": 0.6 + config_index * 0.01,
        "calibration_dataset_sha256": CALIBRATION_SHA256,
        "calibration_resolution_authority_sha256": AUTHORITY_SHA256,
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "center": list(logistic.center),
        "config": config.payload(),
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "lightgbm_fit_contract_sha256": POLYMARKET_ROUND25_LIGHTGBM_FIT_CONTRACT_SHA256,
        "lightgbm_version": "fixture",
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "model_string": model_string,
        "model_string_sha256": model_sha256,
        "residual_logit_bound": POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
        "scale": list(logistic.scale),
        "schema_version": POLYMARKET_ROUND25_LIGHTGBM_ARTIFACT_SCHEMA_VERSION,
        "trading_authority": False,
        "train_dataset_sha256": TRAIN_SHA256,
        "train_resolution_authority_sha256": AUTHORITY_SHA256,
    }
    return Round25LightGBMArtifact(
        config=config,
        train_dataset_sha256=TRAIN_SHA256,
        calibration_dataset_sha256=CALIBRATION_SHA256,
        train_resolution_authority_sha256=AUTHORITY_SHA256,
        calibration_resolution_authority_sha256=AUTHORITY_SHA256,
        center=logistic.center,
        scale=logistic.scale,
        best_iteration=1,
        model_string=model_string,
        model_string_sha256=model_sha256,
        lightgbm_version="fixture",
        backend_kind="cpu",
        backend_device="cpu",
        calibration_condition_equal_log_loss=0.6 + config_index * 0.01,
        calibration_condition_equal_brier_score=0.2 + config_index * 0.01,
        artifact_sha256=_canonical_sha256(payload),
    )


def _tcn_artifact(logistic: Round25LogisticResidualArtifact) -> object:
    torch = pytest.importorskip("torch")
    transform_sha256 = round25_feature_transform_sha256(
        logistic.center,
        logistic.scale,
    )
    seeds = []
    for seed in POLYMARKET_ROUND25_TCN_TRAINING_SEEDS:
        torch.manual_seed(seed)
        seeds.append(_create_round25_tcn_seed_artifact(
            model=_model(),
            training_seed=seed,
            train_dataset_sha256=TRAIN_SHA256,
            calibration_dataset_sha256=CALIBRATION_SHA256,
            train_resolution_authority_sha256=AUTHORITY_SHA256,
            calibration_resolution_authority_sha256=AUTHORITY_SHA256,
            feature_transform_sha256=transform_sha256,
            train_batch_manifest_sha256=TRAIN_BATCH_MANIFEST_SHA256,
            calibration_batch_manifest_sha256=(
                CALIBRATION_BATCH_MANIFEST_SHA256
            ),
            best_epoch=1,
            epochs_run=1,
            calibration_condition_equal_log_loss=0.6,
            calibration_condition_equal_brier_score=0.2,
            backend_requested="cpu",
            backend_kind="cpu",
            backend_device="cpu",
            backend_vendor="generic",
            backend_reason="fixture mechanics only",
            backend_selection="deterministic_cpu_fixture",
        ))
    return _create_round25_tcn_ensemble_artifact(tuple(seeds))


def _ledger() -> object:
    logistic = _logistic_artifact()
    implementation = tuple(
        (path, hashlib.sha256(path.encode("ascii")).hexdigest())
        for path in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
    )
    return create_round25_model_ledger(
        source_commit_oid=SOURCE_COMMIT_OID,
        implementation_sha256=implementation,
        phase_isotonic=_phase_artifact(),
        logistic_residual=logistic,
        lightgbm_residuals=tuple(
            _lightgbm_artifact(index, logistic)
            for index in range(len(POLYMARKET_ROUND25_LIGHTGBM_CONFIGS))
        ),
        tcn_ensemble=_tcn_artifact(logistic),
    )


def test_model_ledger_contract_is_self_hashed_and_target_blind() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-model-ledger-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["candidate_order"][0] == "market-prior-v1"
    assert contract["candidate_order"][-1] == "causal-multitask-tcn-residual-v1"
    assert contract["ledger"]["selection_target_access_allowed"] is False
    assert contract["truth_state"]["candidate_fitted"] is False


def test_model_ledger_binds_every_upstream_feature_and_dataset_operator() -> None:
    assert POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS[:5] == (
        "src/simple_ai_trading/polymarket_round25_candidate_design.py",
        "src/simple_ai_trading/polymarket_round25_twap_features.py",
        "src/simple_ai_trading/polymarket_round25_clob_features.py",
        "src/simple_ai_trading/polymarket_round25_joint_features.py",
        "src/simple_ai_trading/polymarket_round25_dataset.py",
    )
    assert (
        "src/simple_ai_trading/polymarket_round25_joint_materialization.py"
        in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
    )
    assert (
        "src/simple_ai_trading/polymarket_round25_joint_store.py"
        in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
    )
    assert (
        "src/simple_ai_trading/polymarket_round25_resolution_store.py"
        in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
    )
    assert (
        "src/simple_ai_trading/polymarket_round25_tcn_store_source.py"
        in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
    )
    assert (
        "src/simple_ai_trading/polymarket_round25_coordinator.py"
        in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
    )


def test_coordinated_fit_creates_tcn_sources_after_one_logistic_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Dataset:
        def __init__(self, role: str, dataset_sha256: str) -> None:
            self.role = role
            self.dataset_sha256 = dataset_sha256
            self.resolution_authority_sha256 = AUTHORITY_SHA256

        def __post_init__(self) -> None:
            calls.append(f"validate:{self.role}")

    train = Dataset("train", TRAIN_SHA256)
    calibration = Dataset("calibration", CALIBRATION_SHA256)
    phase = object()
    logistic = SimpleNamespace(center=(0.0,), scale=(1.0,))
    tcn_train = SimpleNamespace(
        source_dataset_sha256=TRAIN_SHA256,
        resolution_authority_sha256=AUTHORITY_SHA256,
        feature_transform_sha256="f" * 64,
    )
    tcn_calibration = SimpleNamespace(
        source_dataset_sha256=CALIBRATION_SHA256,
        resolution_authority_sha256=AUTHORITY_SHA256,
        feature_transform_sha256="f" * 64,
    )
    result = object()

    monkeypatch.setattr(ledger_module, "Round25DevelopmentDataset", Dataset)
    monkeypatch.setattr(
        ledger_module,
        "_validate_model_source_identity",
        lambda *_args: (),
    )
    monkeypatch.setattr(
        ledger_module,
        "fit_round25_phase_isotonic",
        lambda _dataset: calls.append("phase") or phase,
    )
    monkeypatch.setattr(
        ledger_module,
        "fit_round25_logistic_residual",
        lambda **_kwargs: calls.append("logistic") or logistic,
    )
    monkeypatch.setattr(
        ledger_module,
        "round25_feature_transform_sha256",
        lambda *_args: "f" * 64,
    )
    monkeypatch.setattr(
        ledger_module,
        "validate_round25_tcn_fit_sources",
        lambda *sources: calls.append("validate_tcn") or sources,
    )
    monkeypatch.setattr(
        ledger_module,
        "fit_round25_lightgbm_residual",
        lambda **_kwargs: calls.append("lightgbm") or object(),
    )
    monkeypatch.setattr(
        ledger_module,
        "fit_round25_tcn_ensemble",
        lambda *_args, **_kwargs: calls.append("tcn") or object(),
    )
    monkeypatch.setattr(
        ledger_module,
        "create_round25_model_ledger",
        lambda **_kwargs: calls.append("ledger") or result,
    )

    def source_factory(observed: object) -> tuple[object, object]:
        assert observed is logistic
        calls.append("source_factory")
        return tcn_train, tcn_calibration

    observed = fit_round25_model_ledger_coordinated(
        source_commit_oid=SOURCE_COMMIT_OID,
        implementation_sha256=(),
        train=train,
        calibration=calibration,
        tcn_source_factory=source_factory,
    )

    assert observed is result
    assert calls.count("logistic") == 1
    assert calls.index("logistic") < calls.index("source_factory")
    assert calls.index("source_factory") < calls.index("lightgbm")
    assert calls[-2:] == ["tcn", "ledger"]


def test_model_ledger_round_trip_binds_all_artifacts(tmp_path: Path) -> None:
    ledger = _ledger()
    path = tmp_path / "round25-model-ledger.json"

    write_round25_model_ledger(path, ledger)
    loaded = load_round25_model_ledger(path)

    assert loaded.ledger_sha256 == ledger.ledger_sha256
    assert loaded.serialized_payload() == ledger.serialized_payload()
    assert loaded.schema_version == POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION
    assert loaded.market_prior_control_sha256 == POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256
    assert loaded.selection_target_accessed is False
    assert loaded.predictive_edge_verified is False
    assert loaded.profitability_verified is False
    assert loaded.paper_authority is False
    assert loaded.live_authority is False
    assert write_round25_model_ledger(path, ledger) == path


def test_model_ledger_rejects_population_and_payload_tampering(
    tmp_path: Path,
) -> None:
    ledger = _ledger()
    path = tmp_path / "round25-model-ledger.json"
    write_round25_model_ledger(path, ledger)
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["artifacts"]["logistic_residual"]["intercept"] = 1.0
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="ascii")

    with pytest.raises(ValueError, match="artifact differs|serialized logistic"):
        load_round25_model_ledger(path)


def test_model_ledger_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="ascii")

    with pytest.raises(ValueError, match="duplicate keys"):
        load_round25_model_ledger(path)


def test_model_artifact_fixture_constants_match_production_contracts() -> None:
    assert POLYMARKET_ROUND25_TCN_ARCHITECTURE
    assert round25_tcn_parameter_count() == 14_371
    assert POLYMARKET_ROUND25_TCN_SEED_ARTIFACT_SCHEMA_VERSION.endswith("-v1")
    assert POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION.endswith("-v1")
    assert len(POLYMARKET_ROUND25_TCN_TRAINING_SEEDS) == 3
    assert len(POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256) == 64
    assert len(POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256) == 64
