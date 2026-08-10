"""Durable, target-blind fitted-model ledger for Round 25."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from .polymarket_round25_controls import (
    Round25IsotonicPhaseModel,
    Round25L2CalibrationScore,
    Round25LogisticResidualArtifact,
    Round25PhaseIsotonicArtifact,
    fit_round25_logistic_residual,
    fit_round25_phase_isotonic,
)
from .polymarket_round25_dataset import Round25DevelopmentDataset
from .polymarket_round25_lightgbm import (
    POLYMARKET_ROUND25_LIGHTGBM_CONFIGS,
    Round25LightGBMArtifact,
    fit_round25_lightgbm_residual,
    round25_lightgbm_config,
)
from .polymarket_round25_sequence import (
    POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256,
    round25_feature_transform_sha256,
)
from .polymarket_round25_tcn import (
    POLYMARKET_ROUND25_TCN_ARCHITECTURE,
    POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON,
    Round25TCNCorpusSource,
    Round25TCNEnsembleArtifact,
    Round25TCNProgressCallback,
    Round25TCNSeedArtifact,
    fit_round25_tcn_ensemble,
)


POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256 = (
    "9e207eabf669a39741f619681357a61b73b2211bc8ecd7498de746310c6b6914"
)
POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION = (
    "polymarket-round25-fitted-model-ledger-v1"
)
POLYMARKET_ROUND25_MODEL_LEDGER_MAXIMUM_BYTES = 64 * 1024 * 1024
POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS = (
    "src/simple_ai_trading/polymarket_round25_controls.py",
    "src/simple_ai_trading/polymarket_round25_lightgbm.py",
    "src/simple_ai_trading/polymarket_round25_sequence.py",
    "src/simple_ai_trading/polymarket_round25_tcn.py",
    "src/simple_ai_trading/polymarket_round25_evaluation.py",
    "src/simple_ai_trading/polymarket_round25_model_ledger.py",
    "src/simple_ai_trading/polymarket_round25_prediction.py",
)
POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "candidate_id": "market-prior-v1",
            "definition": "normalized_same_receipt_up_and_down_clob_midpoint_prior",
            "target_accessed": False,
            "trading_authority": False,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
).hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 25 model ledger contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 model ledger contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 25 {label} is not an object")
    return value


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ValueError(f"Round 25 {label} is not an array")
    return value


def _artifact_payload(value: object) -> dict[str, object]:
    identity = getattr(value, "identity_payload", None)
    artifact_sha256 = getattr(value, "artifact_sha256", None)
    if not callable(identity) or not isinstance(artifact_sha256, str):
        raise TypeError("Round 25 model artifact type differs")
    return {**identity(), "artifact_sha256": artifact_sha256}


def _validate_model_source_identity(
    source_commit_oid: str,
    implementation_sha256: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    implementations = tuple(implementation_sha256)
    if (
        _COMMIT.fullmatch(source_commit_oid) is None
        or tuple(path for path, _digest in implementations)
        != POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
        or any(_SHA256.fullmatch(digest) is None for _path, digest in implementations)
    ):
        raise ValueError("Round 25 model source identity differs")
    return implementations


def _verified_artifact_payload(
    value: object,
    payload: Mapping[str, object],
    *,
    label: str,
) -> object:
    if dict(payload) != _artifact_payload(value):
        raise ValueError(f"Round 25 serialized {label} differs")
    return value


def _load_phase_isotonic(payload: object) -> Round25PhaseIsotonicArtifact:
    value = _mapping(payload, label="phase-isotonic artifact")
    models = tuple(
        Round25IsotonicPhaseModel(
            phase=int(item["phase"]),
            x_thresholds=tuple(float(number) for number in item["x_thresholds"]),
            y_thresholds=tuple(float(number) for number in item["y_thresholds"]),
        )
        for raw in _sequence(value.get("phase_models"), label="phase models")
        for item in (_mapping(raw, label="phase model"),)
    )
    artifact = Round25PhaseIsotonicArtifact(
        fit_dataset_sha256=str(value.get("fit_dataset_sha256")),
        resolution_authority_sha256=str(
            value.get("resolution_authority_sha256")
        ),
        phase_models=models,
        artifact_sha256=str(value.get("artifact_sha256")),
    )
    return _verified_artifact_payload(
        artifact,
        value,
        label="phase-isotonic artifact",
    )


def _load_logistic(payload: object) -> Round25LogisticResidualArtifact:
    value = _mapping(payload, label="logistic artifact")
    scores = tuple(
        Round25L2CalibrationScore(
            l2=float(item["l2"]),
            condition_equal_log_loss=float(
                item["condition_equal_log_loss"]
            ),
            condition_equal_brier_score=float(
                item["condition_equal_brier_score"]
            ),
        )
        for raw in _sequence(value.get("calibration_scores"), label="L2 scores")
        for item in (_mapping(raw, label="L2 score"),)
    )
    artifact = Round25LogisticResidualArtifact(
        train_dataset_sha256=str(value.get("train_dataset_sha256")),
        calibration_dataset_sha256=str(value.get("calibration_dataset_sha256")),
        train_resolution_authority_sha256=str(
            value.get("train_resolution_authority_sha256")
        ),
        calibration_resolution_authority_sha256=str(
            value.get("calibration_resolution_authority_sha256")
        ),
        center=tuple(float(number) for number in value.get("center", ())),
        scale=tuple(float(number) for number in value.get("scale", ())),
        selected_l2=float(value.get("selected_l2", math.nan)),
        intercept=float(value.get("intercept", math.nan)),
        coefficients=tuple(
            float(number) for number in value.get("coefficients", ())
        ),
        calibration_scores=scores,
        artifact_sha256=str(value.get("artifact_sha256")),
    )
    return _verified_artifact_payload(artifact, value, label="logistic artifact")


def _load_lightgbm(payload: object) -> Round25LightGBMArtifact:
    value = _mapping(payload, label="LightGBM artifact")
    config_value = _mapping(value.get("config"), label="LightGBM config")
    config = round25_lightgbm_config(str(config_value.get("candidate_id")))
    artifact = Round25LightGBMArtifact(
        config=config,
        train_dataset_sha256=str(value.get("train_dataset_sha256")),
        calibration_dataset_sha256=str(value.get("calibration_dataset_sha256")),
        train_resolution_authority_sha256=str(
            value.get("train_resolution_authority_sha256")
        ),
        calibration_resolution_authority_sha256=str(
            value.get("calibration_resolution_authority_sha256")
        ),
        center=tuple(float(number) for number in value.get("center", ())),
        scale=tuple(float(number) for number in value.get("scale", ())),
        best_iteration=int(value.get("best_iteration", 0)),
        model_string=str(value.get("model_string")),
        model_string_sha256=str(value.get("model_string_sha256")),
        lightgbm_version=str(value.get("lightgbm_version")),
        backend_kind=str(value.get("backend_kind")),
        backend_device=str(value.get("backend_device")),
        calibration_condition_equal_log_loss=float(
            value.get("calibration_condition_equal_log_loss", math.nan)
        ),
        calibration_condition_equal_brier_score=float(
            value.get("calibration_condition_equal_brier_score", math.nan)
        ),
        artifact_sha256=str(value.get("artifact_sha256")),
    )
    return _verified_artifact_payload(artifact, value, label="LightGBM artifact")


def _load_tcn_seed(payload: object) -> Round25TCNSeedArtifact:
    value = _mapping(payload, label="TCN seed artifact")
    if value.get("architecture") != dict(POLYMARKET_ROUND25_TCN_ARCHITECTURE):
        raise ValueError("Round 25 serialized TCN architecture differs")
    artifact = Round25TCNSeedArtifact(
        training_seed=int(value.get("training_seed", 0)),
        train_dataset_sha256=str(value.get("train_dataset_sha256")),
        calibration_dataset_sha256=str(value.get("calibration_dataset_sha256")),
        train_resolution_authority_sha256=str(
            value.get("train_resolution_authority_sha256")
        ),
        calibration_resolution_authority_sha256=str(
            value.get("calibration_resolution_authority_sha256")
        ),
        feature_transform_sha256=str(value.get("feature_transform_sha256")),
        train_batch_manifest_sha256=str(
            value.get("train_batch_manifest_sha256")
        ),
        calibration_batch_manifest_sha256=str(
            value.get("calibration_batch_manifest_sha256")
        ),
        state_base64=str(value.get("state_base64")),
        state_sha256=str(value.get("state_sha256")),
        parameter_count=int(value.get("parameter_count", 0)),
        best_epoch=int(value.get("best_epoch", 0)),
        epochs_run=int(value.get("epochs_run", 0)),
        calibration_condition_equal_log_loss=float(
            value.get("calibration_condition_equal_log_loss", math.nan)
        ),
        calibration_condition_equal_brier_score=float(
            value.get("calibration_condition_equal_brier_score", math.nan)
        ),
        backend_requested=str(value.get("backend_requested")),
        backend_kind=str(value.get("backend_kind")),
        backend_device=str(value.get("backend_device")),
        backend_vendor=str(value.get("backend_vendor")),
        backend_reason=str(value.get("backend_reason")),
        backend_selection=str(value.get("backend_selection")),
        torch_version=str(value.get("torch_version")),
        artifact_sha256=str(value.get("artifact_sha256")),
        architecture_json=POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON,
    )
    return _verified_artifact_payload(artifact, value, label="TCN seed artifact")


def _tcn_payload(value: Round25TCNEnsembleArtifact) -> dict[str, object]:
    return {
        "ensemble": _artifact_payload(value),
        "seed_artifacts": [
            _artifact_payload(artifact) for artifact in value.seed_artifacts
        ],
    }


def _load_tcn(payload: object) -> Round25TCNEnsembleArtifact:
    value = _mapping(payload, label="TCN ensemble payload")
    if set(value) != {"ensemble", "seed_artifacts"}:
        raise ValueError("Round 25 serialized TCN ensemble payload differs")
    ensemble = _mapping(value["ensemble"], label="TCN ensemble artifact")
    seeds = tuple(
        _load_tcn_seed(item)
        for item in _sequence(value["seed_artifacts"], label="TCN seed artifacts")
    )
    artifact = Round25TCNEnsembleArtifact(
        seed_artifacts=seeds,
        train_dataset_sha256=str(ensemble.get("train_dataset_sha256")),
        calibration_dataset_sha256=str(
            ensemble.get("calibration_dataset_sha256")
        ),
        train_resolution_authority_sha256=str(
            ensemble.get("train_resolution_authority_sha256")
        ),
        calibration_resolution_authority_sha256=str(
            ensemble.get("calibration_resolution_authority_sha256")
        ),
        feature_transform_sha256=str(ensemble.get("feature_transform_sha256")),
        train_batch_manifest_sha256=str(
            ensemble.get("train_batch_manifest_sha256")
        ),
        calibration_batch_manifest_sha256=str(
            ensemble.get("calibration_batch_manifest_sha256")
        ),
        artifact_sha256=str(ensemble.get("artifact_sha256")),
    )
    _verified_artifact_payload(artifact, ensemble, label="TCN ensemble artifact")
    if dict(value) != _tcn_payload(artifact):
        raise ValueError("Round 25 serialized TCN ensemble differs")
    return artifact


@dataclass(frozen=True, slots=True)
class Round25ModelLedger:
    source_commit_oid: str
    implementation_sha256: tuple[tuple[str, str], ...]
    train_dataset_sha256: str
    calibration_dataset_sha256: str
    resolution_authority_sha256: str
    phase_isotonic: Round25PhaseIsotonicArtifact
    logistic_residual: Round25LogisticResidualArtifact
    lightgbm_residuals: tuple[Round25LightGBMArtifact, ...]
    tcn_ensemble: Round25TCNEnsembleArtifact
    ledger_sha256: str
    schema_version: str = POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION
    model_ledger_contract_sha256: str = (
        POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256
    )
    target_free_sequence_contract_sha256: str = (
        POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
    )
    market_prior_control_sha256: str = (
        POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256
    )
    selection_target_accessed: bool = False
    predictive_edge_verified: bool = False
    profitability_verified: bool = False
    paper_authority: bool = False
    live_authority: bool = False

    def candidate_artifact_sha256(self) -> tuple[tuple[str, str], ...]:
        return (
            ("market-prior-v1", self.market_prior_control_sha256),
            (
                "phase-isotonic-market-prior-v1",
                self.phase_isotonic.artifact_sha256,
            ),
            ("l2-logistic-residual-v1", self.logistic_residual.artifact_sha256),
            *tuple(
                (artifact.config.candidate_id, artifact.artifact_sha256)
                for artifact in self.lightgbm_residuals
            ),
            (
                "causal-multitask-tcn-residual-v1",
                self.tcn_ensemble.artifact_sha256,
            ),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "candidate_artifact_sha256": [
                {"candidate_id": candidate_id, "sha256": digest}
                for candidate_id, digest in self.candidate_artifact_sha256()
            ],
            "implementation_sha256": [
                {"path": path, "sha256": digest}
                for path, digest in self.implementation_sha256
            ],
            "live_authority": self.live_authority,
            "market_prior_control_sha256": self.market_prior_control_sha256,
            "model_ledger_contract_sha256": self.model_ledger_contract_sha256,
            "paper_authority": self.paper_authority,
            "predictive_edge_verified": self.predictive_edge_verified,
            "profitability_verified": self.profitability_verified,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "schema_version": self.schema_version,
            "selection_target_accessed": self.selection_target_accessed,
            "source_commit_oid": self.source_commit_oid,
            "target_free_sequence_contract_sha256": (
                self.target_free_sequence_contract_sha256
            ),
            "train_dataset_sha256": self.train_dataset_sha256,
        }

    def serialized_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "artifacts": {
                "lightgbm_residuals": [
                    _artifact_payload(artifact)
                    for artifact in self.lightgbm_residuals
                ],
                "logistic_residual": _artifact_payload(self.logistic_residual),
                "phase_isotonic": _artifact_payload(self.phase_isotonic),
                "tcn_ensemble": _tcn_payload(self.tcn_ensemble),
            },
            "ledger_sha256": self.ledger_sha256,
        }

    def __post_init__(self) -> None:
        expected_paths = POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS
        common = (
            self.train_dataset_sha256,
            self.calibration_dataset_sha256,
            self.resolution_authority_sha256,
        )
        expected_transform = round25_feature_transform_sha256(
            self.logistic_residual.center,
            self.logistic_residual.scale,
        )
        if (
            _COMMIT.fullmatch(self.source_commit_oid) is None
            or tuple(path for path, _digest in self.implementation_sha256)
            != expected_paths
            or any(_SHA256.fullmatch(digest) is None for _path, digest in self.implementation_sha256)
            or any(_SHA256.fullmatch(value) is None for value in common)
            or self.phase_isotonic.validated() is not self.phase_isotonic
            or self.logistic_residual.validated() is not self.logistic_residual
            or len(self.lightgbm_residuals) != len(POLYMARKET_ROUND25_LIGHTGBM_CONFIGS)
            or tuple(
                artifact.config.candidate_id
                for artifact in self.lightgbm_residuals
            )
            != tuple(config.candidate_id for config in POLYMARKET_ROUND25_LIGHTGBM_CONFIGS)
            or any(artifact.validated() is not artifact for artifact in self.lightgbm_residuals)
            or self.tcn_ensemble.validated() is not self.tcn_ensemble
            or self.phase_isotonic.fit_dataset_sha256
            != self.calibration_dataset_sha256
            or self.phase_isotonic.resolution_authority_sha256
            != self.resolution_authority_sha256
            or (
                self.logistic_residual.train_dataset_sha256,
                self.logistic_residual.calibration_dataset_sha256,
                self.logistic_residual.train_resolution_authority_sha256,
            )
            != (
                self.train_dataset_sha256,
                self.calibration_dataset_sha256,
                self.resolution_authority_sha256,
            )
            or self.logistic_residual.calibration_resolution_authority_sha256
            != self.resolution_authority_sha256
            or any(
                (
                    artifact.train_dataset_sha256,
                    artifact.calibration_dataset_sha256,
                    artifact.train_resolution_authority_sha256,
                    artifact.calibration_resolution_authority_sha256,
                    artifact.center,
                    artifact.scale,
                )
                != (
                    self.train_dataset_sha256,
                    self.calibration_dataset_sha256,
                    self.resolution_authority_sha256,
                    self.resolution_authority_sha256,
                    self.logistic_residual.center,
                    self.logistic_residual.scale,
                )
                for artifact in self.lightgbm_residuals
            )
            or (
                self.tcn_ensemble.train_dataset_sha256,
                self.tcn_ensemble.calibration_dataset_sha256,
                self.tcn_ensemble.train_resolution_authority_sha256,
                self.tcn_ensemble.calibration_resolution_authority_sha256,
                self.tcn_ensemble.feature_transform_sha256,
            )
            != (
                self.train_dataset_sha256,
                self.calibration_dataset_sha256,
                self.resolution_authority_sha256,
                self.resolution_authority_sha256,
                expected_transform,
            )
            or tuple(candidate_id for candidate_id, _digest in self.candidate_artifact_sha256())
            != POLYMARKET_ROUND25_CANDIDATE_IDS
            or len({digest for _candidate, digest in self.candidate_artifact_sha256()})
            != len(POLYMARKET_ROUND25_CANDIDATE_IDS)
            or self.schema_version != POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION
            or self.model_ledger_contract_sha256
            != POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256
            or self.target_free_sequence_contract_sha256
            != POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
            or self.market_prior_control_sha256
            != POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256
            or any(
                value is not False
                for value in (
                    self.selection_target_accessed,
                    self.predictive_edge_verified,
                    self.profitability_verified,
                    self.paper_authority,
                    self.live_authority,
                )
            )
            or _SHA256.fullmatch(self.ledger_sha256) is None
            or self.ledger_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 fitted-model ledger differs")

    def validated(self) -> Round25ModelLedger:
        self.__post_init__()
        return self


def create_round25_model_ledger(
    *,
    source_commit_oid: str,
    implementation_sha256: Sequence[tuple[str, str]],
    phase_isotonic: Round25PhaseIsotonicArtifact,
    logistic_residual: Round25LogisticResidualArtifact,
    lightgbm_residuals: Sequence[Round25LightGBMArtifact],
    tcn_ensemble: Round25TCNEnsembleArtifact,
) -> Round25ModelLedger:
    implementations = _validate_model_source_identity(
        source_commit_oid,
        implementation_sha256,
    )
    trees = tuple(lightgbm_residuals)
    identity = {
        "calibration_dataset_sha256": logistic_residual.calibration_dataset_sha256,
        "candidate_artifact_sha256": [
            {
                "candidate_id": "market-prior-v1",
                "sha256": POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256,
            },
            {
                "candidate_id": "phase-isotonic-market-prior-v1",
                "sha256": phase_isotonic.artifact_sha256,
            },
            {
                "candidate_id": "l2-logistic-residual-v1",
                "sha256": logistic_residual.artifact_sha256,
            },
            *[
                {
                    "candidate_id": artifact.config.candidate_id,
                    "sha256": artifact.artifact_sha256,
                }
                for artifact in trees
            ],
            {
                "candidate_id": "causal-multitask-tcn-residual-v1",
                "sha256": tcn_ensemble.artifact_sha256,
            },
        ],
        "implementation_sha256": [
            {"path": path, "sha256": digest}
            for path, digest in implementations
        ],
        "live_authority": False,
        "market_prior_control_sha256": POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256,
        "model_ledger_contract_sha256": POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256,
        "paper_authority": False,
        "predictive_edge_verified": False,
        "profitability_verified": False,
        "resolution_authority_sha256": (
            logistic_residual.train_resolution_authority_sha256
        ),
        "schema_version": POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION,
        "selection_target_accessed": False,
        "source_commit_oid": source_commit_oid,
        "target_free_sequence_contract_sha256": (
            POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
        ),
        "train_dataset_sha256": logistic_residual.train_dataset_sha256,
    }
    return Round25ModelLedger(
        source_commit_oid=source_commit_oid,
        implementation_sha256=implementations,
        train_dataset_sha256=logistic_residual.train_dataset_sha256,
        calibration_dataset_sha256=logistic_residual.calibration_dataset_sha256,
        resolution_authority_sha256=(
            logistic_residual.train_resolution_authority_sha256
        ),
        phase_isotonic=phase_isotonic,
        logistic_residual=logistic_residual,
        lightgbm_residuals=trees,
        tcn_ensemble=tcn_ensemble,
        ledger_sha256=_canonical_sha256(identity),
    )


def fit_round25_model_ledger(
    *,
    source_commit_oid: str,
    implementation_sha256: Sequence[tuple[str, str]],
    train: Round25DevelopmentDataset,
    calibration: Round25DevelopmentDataset,
    tcn_train: Round25TCNCorpusSource,
    tcn_calibration: Round25TCNCorpusSource,
    lightgbm_backend: str = "auto",
    tcn_backend: str = "auto",
    progress_callback: Round25TCNProgressCallback | None = None,
) -> Round25ModelLedger:
    _validate_model_source_identity(source_commit_oid, implementation_sha256)
    if not isinstance(train, Round25DevelopmentDataset) or not isinstance(
        calibration,
        Round25DevelopmentDataset,
    ):
        raise TypeError("Round 25 fitted-model dataset type differs")
    train.__post_init__()
    calibration.__post_init__()
    tcn_train.validated()
    tcn_calibration.validated()
    if (
        train.role != "train"
        or calibration.role != "calibration"
        or train.resolution_authority_sha256
        != calibration.resolution_authority_sha256
        or tcn_train.source_dataset_sha256 != train.dataset_sha256
        or tcn_calibration.source_dataset_sha256 != calibration.dataset_sha256
        or tcn_train.resolution_authority_sha256
        != train.resolution_authority_sha256
        or tcn_calibration.resolution_authority_sha256
        != calibration.resolution_authority_sha256
    ):
        raise ValueError("Round 25 fitted-model populations differ")
    phase = fit_round25_phase_isotonic(calibration)
    logistic = fit_round25_logistic_residual(
        train=train,
        calibration=calibration,
    )
    transform_sha256 = round25_feature_transform_sha256(
        logistic.center,
        logistic.scale,
    )
    if (
        tcn_train.feature_transform_sha256 != transform_sha256
        or tcn_calibration.feature_transform_sha256 != transform_sha256
    ):
        raise ValueError("Round 25 TCN sources differ from the fitted transform")
    trees = tuple(
        fit_round25_lightgbm_residual(
            candidate_id=config.candidate_id,
            train=train,
            calibration=calibration,
            compute_backend=lightgbm_backend,
        )
        for config in POLYMARKET_ROUND25_LIGHTGBM_CONFIGS
    )
    tcn = fit_round25_tcn_ensemble(
        tcn_train,
        tcn_calibration,
        compute_backend=tcn_backend,
        progress_callback=progress_callback,
    )
    return create_round25_model_ledger(
        source_commit_oid=source_commit_oid,
        implementation_sha256=implementation_sha256,
        phase_isotonic=phase,
        logistic_residual=logistic,
        lightgbm_residuals=trees,
        tcn_ensemble=tcn,
    )


def round25_model_implementation_sha256(
    repository: str | Path,
) -> tuple[tuple[str, str], ...]:
    root = Path(repository).resolve()
    values: list[tuple[str, str]] = []
    for relative in POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Round 25 model implementation source is unavailable")
        values.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(values)


def write_round25_model_ledger(
    path: str | Path,
    ledger: Round25ModelLedger,
) -> Path:
    if not isinstance(ledger, Round25ModelLedger):
        raise TypeError("Round 25 model ledger type differs")
    ledger.validated()
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Round 25 model ledger path differs")
    payload = (_canonical_json(ledger.serialized_payload()) + "\n").encode("ascii")
    if len(payload) > POLYMARKET_ROUND25_MODEL_LEDGER_MAXIMUM_BYTES:
        raise ValueError("Round 25 model ledger exceeds its storage bound")
    if target.exists():
        if load_round25_model_ledger(target).ledger_sha256 == ledger.ledger_sha256:
            return target
        raise FileExistsError("Round 25 model ledger path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return target


def load_round25_model_ledger(path: str | Path) -> Round25ModelLedger:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size > POLYMARKET_ROUND25_MODEL_LEDGER_MAXIMUM_BYTES
    ):
        raise ValueError("Round 25 model ledger file differs")
    try:
        decoded = json.loads(
            source.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 model ledger is unreadable") from exc
    value = _mapping(decoded, label="model ledger")
    artifacts = _mapping(value.get("artifacts"), label="model artifacts")
    if set(artifacts) != {
        "lightgbm_residuals",
        "logistic_residual",
        "phase_isotonic",
        "tcn_ensemble",
    }:
        raise ValueError("Round 25 serialized model artifact ledger differs")
    implementation = tuple(
        (str(item.get("path")), str(item.get("sha256")))
        for raw in _sequence(
            value.get("implementation_sha256"),
            label="implementation hashes",
        )
        for item in (_mapping(raw, label="implementation hash"),)
    )
    ledger = Round25ModelLedger(
        source_commit_oid=str(value.get("source_commit_oid")),
        implementation_sha256=implementation,
        train_dataset_sha256=str(value.get("train_dataset_sha256")),
        calibration_dataset_sha256=str(value.get("calibration_dataset_sha256")),
        resolution_authority_sha256=str(
            value.get("resolution_authority_sha256")
        ),
        phase_isotonic=_load_phase_isotonic(artifacts["phase_isotonic"]),
        logistic_residual=_load_logistic(artifacts["logistic_residual"]),
        lightgbm_residuals=tuple(
            _load_lightgbm(item)
            for item in _sequence(
                artifacts["lightgbm_residuals"],
                label="LightGBM artifacts",
            )
        ),
        tcn_ensemble=_load_tcn(artifacts["tcn_ensemble"]),
        ledger_sha256=str(value.get("ledger_sha256")),
    )
    if dict(value) != ledger.serialized_payload():
        raise ValueError("Round 25 serialized fitted-model ledger differs")
    return ledger


__all__ = [
    "POLYMARKET_ROUND25_MARKET_PRIOR_CONTROL_SHA256",
    "POLYMARKET_ROUND25_MODEL_IMPLEMENTATION_PATHS",
    "POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_MODEL_LEDGER_MAXIMUM_BYTES",
    "POLYMARKET_ROUND25_MODEL_LEDGER_SCHEMA_VERSION",
    "Round25ModelLedger",
    "create_round25_model_ledger",
    "fit_round25_model_ledger",
    "load_round25_model_ledger",
    "round25_model_implementation_sha256",
    "write_round25_model_ledger",
]
