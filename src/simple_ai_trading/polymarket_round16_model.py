"""Pretest-only predictive candidates for the Round 16 BTC 15-minute screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from .lightgbm_backend import lightgbm_backend_parameters
from .polymarket_historical_model import (
    _condition_weights,
    _fit_logistic_parameters,
    _fit_platt_calibration,
    _log_loss,
    _probability,
)
from .polymarket_historical_screen import HistoricalScreenStore
from .polymarket_round16 import Round16HistoricalContract
from .polymarket_round16_dataset import (
    ROUND16_CALENDAR_FEATURE_NAMES,
    ROUND16_DATASET_SCHEMA_VERSION,
    ROUND16_FEATURE_NAMES,
)


ROUND16_PRETEST_SCHEMA_VERSION = "polymarket-round16-btc-15m-pretest-v1"
ROUND16_MODEL_SEED = 16_015
ROUND16_RIDGE_L2_GRID = (0.01, 0.1, 1.0, 10.0)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LIGHTGBM_GRID = (
    {
        "candidate": "round16-lgbm-depth2-leaves3",
        "learning_rate": 0.03,
        "num_leaves": 3,
        "max_depth": 2,
        "min_data_in_leaf": 256,
        "lambda_l2": 1.0,
    },
    {
        "candidate": "round16-lgbm-depth3-leaves7",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 256,
        "lambda_l2": 5.0,
    },
    {
        "candidate": "round16-lgbm-depth4-leaves15",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 384,
        "lambda_l2": 10.0,
    },
)


ProgressCallback = Callable[[str, Mapping[str, object]], None]


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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Round16ModelPanel:
    condition_ids: np.ndarray
    roles: np.ndarray
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    dataset_sha256: str

    def validate(self, *, expected_roles: Sequence[str]) -> None:
        rows = len(self.labels)
        roles = tuple(expected_roles)
        if (
            rows == 0
            or self.condition_ids.shape != (rows,)
            or self.roles.shape != (rows,)
            or self.event_start_ms.shape != (rows,)
            or self.decision_time_ms.shape != (rows,)
            or self.features.shape != (rows, len(ROUND16_FEATURE_NAMES))
            or not np.all(np.isfinite(self.features))
            or not np.all((self.labels == 0.0) | (self.labels == 1.0))
            or not set(np.unique(self.roles)).issubset(set(roles))
            or len(self.dataset_sha256) != 64
        ):
            raise ValueError("Round 16 model panel differs")
        unique, counts = np.unique(self.condition_ids, return_counts=True)
        if len(unique) == 0 or np.any(counts != 14):
            raise ValueError("Round 16 condition decision coverage differs")
        for condition_id in unique:
            selected = self.condition_ids == condition_id
            if len(np.unique(self.labels[selected])) != 1:
                raise ValueError("Round 16 label differs within a condition")


def _dataset_identity(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
) -> str:
    row = (
        store.connect()
        .execute(
            """
            SELECT manifest_json, dataset_sha256
            FROM feature.round16_dataset_manifest
            WHERE singleton
            """
        )
        .fetchone()
    )
    if row is None:
        raise ValueError("Round 16 dataset manifest is missing")
    try:
        manifest = json.loads(str(row[0]))
    except json.JSONDecodeError as exc:
        raise ValueError("Round 16 dataset manifest is not JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("Round 16 dataset manifest is malformed")
    expected_names_sha = _canonical_sha256(ROUND16_FEATURE_NAMES)
    dataset_sha = str(row[1])
    if (
        _canonical_sha256(manifest) != dataset_sha
        or manifest.get("schema_version") != ROUND16_DATASET_SCHEMA_VERSION
        or manifest.get("contract_sha256") != contract.contract_sha256
        or manifest.get("feature_names_sha256") != expected_names_sha
        or int(manifest.get("condition_count", 0)) != 14_783
        or int(manifest.get("row_count", 0)) != 206_962
        or int(manifest.get("source_boundary_censored_conditions", -1)) != 1
    ):
        raise ValueError("Round 16 dataset identity differs")
    return dataset_sha


def load_round16_model_panel(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    *,
    roles: Sequence[str],
) -> Round16ModelPanel:
    selected_roles = tuple(str(role) for role in roles)
    if store.contract != contract.historical:
        raise ValueError("Round 16 store contract differs")
    if (
        not selected_roles
        or len(selected_roles) != len(set(selected_roles))
        or any(role not in {"train", "tune", "test"} for role in selected_roles)
    ):
        raise ValueError("Round 16 panel roles are invalid")
    if "test" in selected_roles and store.state not in {"targets_complete", "evaluated"}:
        raise ValueError("Round 16 test panel is unavailable before one-use access")
    if any(role in {"train", "tune"} for role in selected_roles) and store.state not in {
        "development_targets_complete",
        "pretest_complete",
        "targets_complete",
        "evaluated",
    }:
        raise ValueError("Round 16 development panel is not authorized")
    placeholders = ",".join("?" for _ in selected_roles)
    rows = (
        store.connect()
        .execute(
            f"""
            SELECT row.condition_id, row.role, row.event_start_ms,
                   row.decision_time_ms, row.feature_values,
                   resolution.winning_outcome
            FROM feature.causal_row AS row
            INNER JOIN target.official_resolution AS resolution
              ON resolution.condition_id = row.condition_id
             AND resolution.role = row.role
            WHERE row.role IN ({placeholders})
            ORDER BY row.event_start_ms, row.decision_time_ms
            """,
            list(selected_roles),
        )
        .fetchall()
    )
    if not rows or any(str(row[5]) not in {"Up", "Down"} for row in rows):
        raise ValueError("Round 16 model panel labels are unavailable")
    panel = Round16ModelPanel(
        condition_ids=np.asarray([str(row[0]) for row in rows], dtype=object),
        roles=np.asarray([str(row[1]) for row in rows], dtype=object),
        event_start_ms=np.asarray([int(row[2]) for row in rows], dtype=np.int64),
        decision_time_ms=np.asarray([int(row[3]) for row in rows], dtype=np.int64),
        features=np.asarray([row[4] for row in rows], dtype=np.float32),
        labels=np.asarray(
            [1.0 if str(row[5]) == "Up" else 0.0 for row in rows],
            dtype=np.float64,
        ),
        dataset_sha256=_dataset_identity(store, contract),
    )
    panel.validate(expected_roles=selected_roles)
    return panel


def _fit_ridge_candidate(
    train: Round16ModelPanel,
    tune: Round16ModelPanel,
    *,
    family: str,
    feature_indices: np.ndarray,
) -> Mapping[str, object]:
    train_weights = _condition_weights(train.condition_ids)
    tune_weights = _condition_weights(tune.condition_ids)
    train_matrix = np.asarray(train.features[:, feature_indices], dtype=np.float64)
    tune_matrix = np.asarray(tune.features[:, feature_indices], dtype=np.float64)
    mean = np.average(train_matrix, axis=0, weights=train_weights)
    variance = np.average(
        np.square(train_matrix - mean),
        axis=0,
        weights=train_weights,
    )
    scale = np.sqrt(np.maximum(variance, 1e-12))
    train_standard = (train_matrix - mean) / scale
    tune_standard = (tune_matrix - mean) / scale
    best: dict[str, object] | None = None
    for l2 in ROUND16_RIDGE_L2_GRID:
        intercept, coefficient = _fit_logistic_parameters(
            train_standard,
            train.labels,
            train_weights,
            l2=l2,
        )
        prediction = expit(intercept + tune_standard @ coefficient)
        loss = _log_loss(tune.labels, prediction, tune_weights)
        candidate = {
            "family": family,
            "kind": "control" if family == "calendar_ridge_logistic" else "challenger",
            "candidate_id": f"{family}-l2-{format(l2, 'g')}",
            "feature_indices": feature_indices.tolist(),
            "model": {
                "type": "ridge_logistic",
                "l2": l2,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "intercept": intercept,
                "coefficient": coefficient.tolist(),
            },
            "raw_tune_log_loss": loss,
        }
        if best is None or (loss, str(candidate["candidate_id"])) < (
            float(best["raw_tune_log_loss"]),
            str(best["candidate_id"]),
        ):
            best = candidate
    if best is None:
        raise RuntimeError("Round 16 ridge candidate selection failed")
    return best


def _fit_lightgbm_candidate(
    train: Round16ModelPanel,
    tune: Round16ModelPanel,
    *,
    compute_backend: str,
    progress: ProgressCallback | None,
) -> Mapping[str, object]:
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        ROUND16_MODEL_SEED,
        reproducible=True,
        pin_opencl_device=True,
    )
    train_weights = _condition_weights(train.condition_ids)
    tune_weights = _condition_weights(tune.condition_ids)
    train_set = lgb.Dataset(
        train.features,
        label=train.labels,
        weight=train_weights,
        feature_name=list(ROUND16_FEATURE_NAMES),
        free_raw_data=False,
    )
    tune_set = lgb.Dataset(
        tune.features,
        label=tune.labels,
        weight=tune_weights,
        reference=train_set,
        feature_name=list(ROUND16_FEATURE_NAMES),
        free_raw_data=False,
    )
    best: dict[str, object] | None = None
    for index, frozen in enumerate(_LIGHTGBM_GRID, start=1):
        if progress:
            progress(
                "round16_model_candidate_started",
                {
                    "candidate": frozen["candidate"],
                    "candidate_index": index,
                    "candidate_count": len(_LIGHTGBM_GRID),
                    "backend_kind": backend_kind,
                    "backend_device": backend_device,
                },
            )
        parameters: dict[str, object] = {
            **backend,
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": frozen["learning_rate"],
            "num_leaves": frozen["num_leaves"],
            "max_depth": frozen["max_depth"],
            "min_data_in_leaf": frozen["min_data_in_leaf"],
            "lambda_l1": 0.0,
            "lambda_l2": frozen["lambda_l2"],
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "max_bin": 63,
            "histogram_pool_size": 256,
            "feature_pre_filter": False,
        }
        booster = lgb.train(
            parameters,
            train_set,
            num_boost_round=256,
            valid_sets=[tune_set],
            valid_names=["tune"],
            callbacks=[lgb.early_stopping(32, verbose=False), lgb.log_evaluation(0)],
        )
        best_iteration = int(booster.best_iteration or booster.current_iteration())
        tune_prediction = _probability(
            np.asarray(
                booster.predict(tune.features, num_iteration=best_iteration),
                dtype=np.float64,
            )
        )
        loss = _log_loss(tune.labels, tune_prediction, tune_weights)
        model_string = booster.model_to_string(num_iteration=best_iteration)
        reload_prediction = np.asarray(
            lgb.Booster(model_str=model_string).predict(tune.features),
            dtype=np.float64,
        )
        reload_difference = float(
            np.max(np.abs(tune_prediction - reload_prediction), initial=0.0)
        )
        if reload_difference > 1e-12:
            raise RuntimeError("Round 16 LightGBM serialization identity failed")
        candidate = {
            "family": "binance_shallow_lightgbm",
            "kind": "challenger",
            "candidate_id": str(frozen["candidate"]),
            "feature_indices": list(range(len(ROUND16_FEATURE_NAMES))),
            "model": {
                "type": "lightgbm",
                "parameters": dict(frozen),
                "best_iteration": best_iteration,
                "model_string": model_string,
                "model_sha256": hashlib.sha256(
                    model_string.encode("utf-8")
                ).hexdigest(),
                "reload_max_absolute_difference": reload_difference,
                "lightgbm_version": str(lgb.__version__),
                "backend_requested": compute_backend,
                "backend_kind": backend_kind,
                "backend_device": backend_device,
            },
            "raw_tune_log_loss": loss,
        }
        if best is None or (loss, str(candidate["candidate_id"])) < (
            float(best["raw_tune_log_loss"]),
            str(best["candidate_id"]),
        ):
            best = candidate
        if progress:
            progress(
                "round16_model_candidate_completed",
                {
                    "candidate": frozen["candidate"],
                    "best_iteration": best_iteration,
                    "tune_log_loss": loss,
                },
            )
    if best is None:
        raise RuntimeError("Round 16 LightGBM candidate selection failed")
    return best


def _raw_prediction(
    candidate: Mapping[str, object],
    features: np.ndarray,
) -> np.ndarray:
    model = candidate.get("model")
    indexes = np.asarray(candidate.get("feature_indices"), dtype=np.int64)
    if not isinstance(model, Mapping) or indexes.ndim != 1:
        raise ValueError("Round 16 candidate artifact is malformed")
    model_type = str(model.get("type"))
    if model_type == "constant":
        return np.full(len(features), float(model["probability"]), dtype=np.float64)
    if model_type == "ridge_logistic":
        matrix = np.asarray(features[:, indexes], dtype=np.float64)
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coefficient = np.asarray(model["coefficient"], dtype=np.float64)
        if (
            mean.shape != (len(indexes),)
            or scale.shape != mean.shape
            or coefficient.shape != mean.shape
            or np.any(scale <= 0.0)
        ):
            raise ValueError("Round 16 ridge artifact dimensions differ")
        return expit(
            float(model["intercept"]) + ((matrix - mean) / scale) @ coefficient
        )
    if model_type == "lightgbm":
        booster = lgb.Booster(model_str=str(model["model_string"]))
        matrix = np.asarray(features[:, indexes], dtype=np.float32)
        return np.asarray(booster.predict(matrix), dtype=np.float64)
    raise ValueError("Round 16 candidate model type is unsupported")


def predict_round16_candidate(
    candidate: Mapping[str, object],
    features: np.ndarray,
) -> np.ndarray:
    raw = _probability(_raw_prediction(candidate, features))
    calibration = candidate.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("Round 16 candidate calibration is missing")
    if calibration.get("retained") is True:
        logits = np.log(raw) - np.log1p(-raw)
        raw = expit(
            float(calibration["intercept"]) + float(calibration["slope"]) * logits
        )
    return _probability(raw)


def _finalize_candidate(
    candidate: Mapping[str, object],
    tune: Round16ModelPanel,
) -> Mapping[str, object]:
    body = dict(candidate)
    raw = _probability(_raw_prediction(body, tune.features))
    body["calibration"] = _fit_platt_calibration(
        tune.labels,
        raw,
        _condition_weights(tune.condition_ids),
    )
    calibrated = predict_round16_candidate(body, tune.features)
    body["tune_condition_balanced_log_loss"] = _log_loss(
        tune.labels,
        calibrated,
        _condition_weights(tune.condition_ids),
    )
    body["feature_names_sha256"] = _canonical_sha256(ROUND16_FEATURE_NAMES)
    body["dataset_sha256"] = tune.dataset_sha256
    return {**body, "artifact_sha256": _canonical_sha256(body)}


def fit_round16_pretest_candidates(
    train: Round16ModelPanel,
    tune: Round16ModelPanel,
    *,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> tuple[Mapping[str, object], ...]:
    train.validate(expected_roles=("train",))
    tune.validate(expected_roles=("tune",))
    if train.dataset_sha256 != tune.dataset_sha256:
        raise ValueError("Round 16 train and tune dataset identities differ")
    train_weights = _condition_weights(train.condition_ids)
    prevalence = float(np.average(train.labels, weights=train_weights))
    candidates: list[Mapping[str, object]] = [
        {
            "family": "training_prevalence",
            "kind": "control",
            "candidate_id": "round16-training-prevalence",
            "feature_indices": [],
            "model": {"type": "constant", "probability": prevalence},
            "raw_tune_log_loss": _log_loss(
                tune.labels,
                np.full(len(tune.labels), prevalence, dtype=np.float64),
                _condition_weights(tune.condition_ids),
            ),
        }
    ]
    calendar_indices = np.asarray(
        [
            ROUND16_FEATURE_NAMES.index(name)
            for name in ROUND16_CALENDAR_FEATURE_NAMES
        ],
        dtype=np.int64,
    )
    candidates.extend(
        (
            _fit_ridge_candidate(
                train,
                tune,
                family="calendar_ridge_logistic",
                feature_indices=calendar_indices,
            ),
            _fit_ridge_candidate(
                train,
                tune,
                family="binance_ridge_logistic",
                feature_indices=np.arange(
                    len(ROUND16_FEATURE_NAMES),
                    dtype=np.int64,
                ),
            ),
            _fit_lightgbm_candidate(
                train,
                tune,
                compute_backend=compute_backend,
                progress=progress,
            ),
        )
    )
    return tuple(_finalize_candidate(candidate, tune) for candidate in candidates)


def build_round16_pretest_artifact(
    train: Round16ModelPanel,
    tune: Round16ModelPanel,
    candidates: Sequence[Mapping[str, object]],
    *,
    contract: Round16HistoricalContract,
    source_commit: str,
) -> Mapping[str, object]:
    train.validate(expected_roles=("train",))
    tune.validate(expected_roles=("tune",))
    commit = str(source_commit or "").strip().lower()
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("Round 16 source commit is invalid")
    if (
        train.dataset_sha256 != tune.dataset_sha256
        or not candidates
        or any(candidate.get("dataset_sha256") != train.dataset_sha256 for candidate in candidates)
    ):
        raise ValueError("Round 16 pretest candidate binding differs")
    for candidate in candidates:
        candidate_body = dict(candidate)
        claimed = str(candidate_body.pop("artifact_sha256", ""))
        if len(claimed) != 64 or _canonical_sha256(candidate_body) != claimed:
            raise ValueError("Round 16 candidate artifact integrity failed")
    controls = [candidate for candidate in candidates if candidate.get("kind") == "control"]
    challengers = [
        candidate for candidate in candidates if candidate.get("kind") == "challenger"
    ]
    if len(controls) != 2 or len(challengers) != 2:
        raise ValueError("Round 16 pretest candidate families differ")
    best_control = min(
        controls,
        key=lambda value: (
            float(value["tune_condition_balanced_log_loss"]),
            str(value["candidate_id"]),
        ),
    )
    best_challenger = min(
        challengers,
        key=lambda value: (
            float(value["tune_condition_balanced_log_loss"]),
            str(value["candidate_id"]),
        ),
    )
    source_root = Path(__file__).parent
    implementation = {
        "round16_model": _file_sha256(Path(__file__)),
        "round16_dataset": _file_sha256(
            source_root / "polymarket_round16_dataset.py"
        ),
        "round16_identity": _file_sha256(source_root / "polymarket_round16.py"),
        "round16_evaluation": _file_sha256(
            source_root / "polymarket_round16_evaluation.py"
        ),
        "shared_model_primitives": _file_sha256(
            source_root / "polymarket_historical_model.py"
        ),
    }
    body: dict[str, object] = {
        "schema_version": ROUND16_PRETEST_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "dataset_sha256": train.dataset_sha256,
        "source_commit": commit,
        "feature_names": list(ROUND16_FEATURE_NAMES),
        "feature_names_sha256": _canonical_sha256(ROUND16_FEATURE_NAMES),
        "train": {
            "row_count": len(train.labels),
            "condition_count": len(np.unique(train.condition_ids)),
            "up_conditions": int(
                sum(
                    train.labels[np.flatnonzero(train.condition_ids == condition)[0]]
                    == 1.0
                    for condition in np.unique(train.condition_ids)
                )
            ),
        },
        "tune": {
            "row_count": len(tune.labels),
            "condition_count": len(np.unique(tune.condition_ids)),
        },
        "candidates": list(candidates),
        "selected_best_control": str(best_control["candidate_id"]),
        "selected_best_challenger": str(best_challenger["candidate_id"]),
        "selection_metric": "condition_balanced_tune_log_loss",
        "implementation_sha256": implementation,
        "test_targets_accessed": False,
        "paper_authority": False,
        "live_authority": False,
        "profitability_claim": False,
    }
    return {**body, "artifact_sha256": _canonical_sha256(body)}


def record_round16_pretest_artifact(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    artifact: Mapping[str, object],
) -> str:
    """Seal the pretest artifact before the one-use test-label phase."""

    if store.contract != contract.historical or store.state != (
        "development_targets_complete"
    ):
        raise ValueError("Round 16 pretest phase is not authorized")
    value = dict(artifact)
    claimed = str(value.pop("artifact_sha256", ""))
    if (
        len(claimed) != 64
        or _canonical_sha256(value) != claimed
        or value.get("schema_version") != ROUND16_PRETEST_SCHEMA_VERSION
        or value.get("contract_sha256") != contract.contract_sha256
        or value.get("test_targets_accessed") is not False
        or value.get("paper_authority") is not False
        or value.get("live_authority") is not False
        or value.get("profitability_claim") is not False
    ):
        raise ValueError("Round 16 pretest artifact integrity differs")
    full_artifact = {**value, "artifact_sha256": claimed}
    dataset = store.connect().execute(
        """
        SELECT dataset_sha256
        FROM feature.round16_dataset_manifest
        WHERE singleton
        """
    ).fetchone()
    test_count = int(
        store.connect()
        .execute(
            """
            SELECT count(*)
            FROM target.official_resolution
            WHERE role = 'test'
            """
        )
        .fetchone()[0]
    )
    if (
        dataset is None
        or str(dataset[0]) != value.get("dataset_sha256")
        or test_count != 0
    ):
        raise ValueError("Round 16 pretest dataset or test boundary differs")
    canonical = _canonical_json(full_artifact)
    envelope_sha = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    now = time.time_ns() // 1_000_000
    connection = store.connect()
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            INSERT INTO feature.pretest_manifest VALUES (
                true, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                ROUND16_PRETEST_SCHEMA_VERSION,
                contract.contract_sha256,
                str(dataset[0]),
                canonical,
                envelope_sha,
                now,
            ],
        )
        changed = connection.execute(
            """
            UPDATE feature.screen_manifest
            SET state = 'pretest_complete', updated_at_ms = ?
            WHERE singleton AND state = 'development_targets_complete'
            RETURNING state
            """,
            [now],
        ).fetchone()
        if changed is None:
            raise ValueError("Round 16 pretest state changed concurrently")
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    return envelope_sha


__all__ = [
    "ROUND16_MODEL_SEED",
    "ROUND16_PRETEST_SCHEMA_VERSION",
    "ROUND16_RIDGE_L2_GRID",
    "Round16ModelPanel",
    "build_round16_pretest_artifact",
    "fit_round16_pretest_candidates",
    "load_round16_model_panel",
    "predict_round16_candidate",
    "record_round16_pretest_artifact",
]
