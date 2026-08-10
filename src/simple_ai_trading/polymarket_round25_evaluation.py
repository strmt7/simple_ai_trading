"""Leakage-resistant predictive evaluation for the finite Round 25 ledger."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Mapping, Sequence

import numpy as np

from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    POLYMARKET_ROUND25_MINIMUM_CONDITIONS,
    Round25DevelopmentDataset,
    require_round25_dataset_minimum,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256 = (
    "4e5036103cfb11319b68908a09fa9acdb93a02f4ff1cede9dec9e9a81f491077"
)
POLYMARKET_ROUND25_CANDIDATE_PREDICTION_SCHEMA_VERSION = (
    "polymarket-round25-candidate-prediction-v1"
)
POLYMARKET_ROUND25_PREDICTION_PANEL_SCHEMA_VERSION = (
    "polymarket-round25-target-free-prediction-panel-v1"
)
POLYMARKET_ROUND25_SELECTION_ACCESS_RECEIPT_SCHEMA_VERSION = (
    "polymarket-round25-selection-target-access-receipt-v1"
)
POLYMARKET_ROUND25_PREDICTIVE_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-predictive-evaluation-result-v1"
)
POLYMARKET_ROUND25_BOOTSTRAP_REPLICATES = 10_000
POLYMARKET_ROUND25_BOOTSTRAP_SEED = 25_025
POLYMARKET_ROUND25_PREDICTIVE_RESULT_MAXIMUM_BYTES = 2 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 predictive result contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 predictive result contains {value}")
POLYMARKET_ROUND25_BOOTSTRAP_BLOCK_CONDITIONS = 12
POLYMARKET_ROUND25_BOOTSTRAP_CHUNK_REPLICATES = 256
POLYMARKET_ROUND25_LOG_LOSS_CLIP = 1e-12
POLYMARKET_ROUND25_ALPHA = 0.05
POLYMARKET_ROUND25_GATE_METRICS = ("log_loss", "brier_score")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")


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


def _array_sha256(value: np.ndarray) -> str:
    selected = np.asarray(value)
    if not selected.flags.c_contiguous:
        selected = np.ascontiguousarray(selected)
    digest = hashlib.sha256()
    digest.update(selected.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(selected.shape)).encode("ascii"))
    digest.update(selected.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: Sequence[float] | Sequence[int], *, dtype: str) -> np.ndarray:
    selected = np.asarray(value, dtype=np.dtype(dtype), order="C").copy(order="C")
    selected.setflags(write=False)
    return selected


@dataclass(frozen=True, slots=True)
class Round25CandidatePrediction:
    candidate_id: str
    source_artifact_sha256: str
    probabilities: np.ndarray
    probabilities_sha256: str
    schema_version: str = POLYMARKET_ROUND25_CANDIDATE_PREDICTION_SCHEMA_VERSION
    target_accessed: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "probabilities_sha256": self.probabilities_sha256,
            "probability_count": len(self.probabilities),
            "schema_version": self.schema_version,
            "source_artifact_sha256": self.source_artifact_sha256,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        if (
            self.candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS
            or _SHA256.fullmatch(self.source_artifact_sha256) is None
            or self.probabilities.ndim != 1
            or not len(self.probabilities)
            or self.probabilities.dtype != np.dtype("<f8")
            or self.probabilities.flags.writeable
            or not np.all(np.isfinite(self.probabilities))
            or not np.all((self.probabilities >= 0.0) & (self.probabilities <= 1.0))
            or _SHA256.fullmatch(self.probabilities_sha256) is None
            or self.probabilities_sha256 != _array_sha256(self.probabilities)
            or self.schema_version
            != POLYMARKET_ROUND25_CANDIDATE_PREDICTION_SCHEMA_VERSION
            or self.target_accessed is not False
            or self.trading_authority is not False
        ):
            raise ValueError("Round 25 candidate prediction differs")

    def validated(self) -> Round25CandidatePrediction:
        self.__post_init__()
        return self


@dataclass(frozen=True, slots=True)
class Round25PredictionPanel:
    row_condition_ids: tuple[str, ...]
    event_start_ms: np.ndarray
    decision_time_ms: np.ndarray
    feature_source_chain_sha256: tuple[str, ...]
    market_prior_probability: np.ndarray
    candidate_predictions: tuple[Round25CandidatePrediction, ...]
    row_identity_sha256: str
    panel_sha256: str
    schema_version: str = POLYMARKET_ROUND25_PREDICTION_PANEL_SCHEMA_VERSION
    role: str = "selection"
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    evaluation_contract_sha256: str = (
        POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
    )
    target_accessed: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "candidate_predictions": [
                prediction.identity_payload()
                for prediction in self.candidate_predictions
            ],
            "condition_count": len(self.row_condition_ids)
            // POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "model_design_sha256": self.model_design_sha256,
            "role": self.role,
            "row_identity_sha256": self.row_identity_sha256,
            "schema_version": self.schema_version,
            "target_accessed": self.target_accessed,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        row_count = len(self.row_condition_ids)
        condition_count = row_count // POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
        if row_count:
            condition_keys = tuple(
                (
                    int(self.event_start_ms[offset]),
                    self.row_condition_ids[offset],
                )
                for offset in range(
                    0,
                    row_count,
                    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
                )
            )
        else:
            condition_keys = ()
        row_payload = {
            "decision_time_ms_sha256": _array_sha256(self.decision_time_ms),
            "event_start_ms_sha256": _array_sha256(self.event_start_ms),
            "feature_source_chain_sha256": list(self.feature_source_chain_sha256),
            "market_prior_probability_sha256": _array_sha256(
                self.market_prior_probability
            ),
            "row_condition_ids": list(self.row_condition_ids),
        }
        if (
            row_count
            < POLYMARKET_ROUND25_MINIMUM_CONDITIONS["selection"]
            * POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            or row_count % POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            or self.event_start_ms.shape != (row_count,)
            or self.event_start_ms.dtype != np.dtype("<i8")
            or self.event_start_ms.flags.writeable
            or self.decision_time_ms.shape != (row_count,)
            or self.decision_time_ms.dtype != np.dtype("<i8")
            or self.decision_time_ms.flags.writeable
            or len(self.feature_source_chain_sha256) != row_count
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.feature_source_chain_sha256
            )
            or len(set(self.feature_source_chain_sha256)) != row_count
            or self.market_prior_probability.shape != (row_count,)
            or self.market_prior_probability.dtype != np.dtype("<f8")
            or self.market_prior_probability.flags.writeable
            or not np.all(np.isfinite(self.market_prior_probability))
            or not np.all(
                (self.market_prior_probability > 0.0)
                & (self.market_prior_probability < 1.0)
            )
            or any(_CONDITION_ID.fullmatch(value) is None for value in self.row_condition_ids)
            or len(set(key[1] for key in condition_keys)) != condition_count
            or condition_keys != tuple(sorted(condition_keys))
            or any(
                len(set(self.row_condition_ids[offset : offset + 16])) != 1
                or len(set(self.event_start_ms[offset : offset + 16])) != 1
                or not np.all(
                    np.diff(self.decision_time_ms[offset : offset + 16]) > 0
                )
                or not np.all(
                    (self.decision_time_ms[offset : offset + 16]
                    >= self.event_start_ms[offset])
                    & (self.decision_time_ms[offset : offset + 16]
                    < self.event_start_ms[offset]
                    + POLYMARKET_ROUND25_CONDITION_DURATION_MS)
                )
                for offset in range(0, row_count, 16)
            )
            or tuple(
                prediction.candidate_id for prediction in self.candidate_predictions
            )
            != POLYMARKET_ROUND25_CANDIDATE_IDS
            or len({
                prediction.source_artifact_sha256
                for prediction in self.candidate_predictions
            })
            != len(POLYMARKET_ROUND25_CANDIDATE_IDS)
            or any(
                prediction.validated() is not prediction
                or len(prediction.probabilities) != row_count
                for prediction in self.candidate_predictions
            )
            or not np.array_equal(
                self.candidate_predictions[0].probabilities,
                self.market_prior_probability,
            )
            or _SHA256.fullmatch(self.row_identity_sha256) is None
            or self.row_identity_sha256 != _canonical_sha256(row_payload)
            or self.schema_version != POLYMARKET_ROUND25_PREDICTION_PANEL_SCHEMA_VERSION
            or self.role != "selection"
            or self.model_design_sha256
            != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.evaluation_contract_sha256
            != POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
            or self.target_accessed is not False
            or self.trading_authority is not False
            or _SHA256.fullmatch(self.panel_sha256) is None
            or self.panel_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 prediction panel differs")

    def validated(self) -> Round25PredictionPanel:
        self.__post_init__()
        return self


def create_round25_prediction_panel(
    *,
    row_condition_ids: Sequence[str],
    event_start_ms: Sequence[int],
    decision_time_ms: Sequence[int],
    feature_source_chain_sha256: Sequence[str],
    market_prior_probability: Sequence[float],
    candidate_probabilities: Mapping[str, Sequence[float]],
    candidate_source_artifact_sha256: Mapping[str, str],
) -> Round25PredictionPanel:
    condition_ids = tuple(row_condition_ids)
    starts = _readonly(event_start_ms, dtype="<i8")
    decisions = _readonly(decision_time_ms, dtype="<i8")
    source_chain = tuple(feature_source_chain_sha256)
    prior = _readonly(market_prior_probability, dtype="<f8")
    if (
        tuple(candidate_probabilities) != POLYMARKET_ROUND25_CANDIDATE_IDS
        or tuple(candidate_source_artifact_sha256)
        != POLYMARKET_ROUND25_CANDIDATE_IDS
    ):
        raise ValueError("Round 25 prediction candidate ledger differs")
    predictions = tuple(
        Round25CandidatePrediction(
            candidate_id=candidate_id,
            source_artifact_sha256=candidate_source_artifact_sha256[candidate_id],
            probabilities=(
                values := _readonly(
                    candidate_probabilities[candidate_id],
                    dtype="<f8",
                )
            ),
            probabilities_sha256=_array_sha256(values),
        )
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS
    )
    row_payload = {
        "decision_time_ms_sha256": _array_sha256(decisions),
        "event_start_ms_sha256": _array_sha256(starts),
        "feature_source_chain_sha256": list(source_chain),
        "market_prior_probability_sha256": _array_sha256(prior),
        "row_condition_ids": list(condition_ids),
    }
    row_identity_sha256 = _canonical_sha256(row_payload)
    values = {
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_predictions": [prediction.identity_payload() for prediction in predictions],
        "condition_count": len(condition_ids)
        // POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
        ),
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "role": "selection",
        "row_identity_sha256": row_identity_sha256,
        "schema_version": POLYMARKET_ROUND25_PREDICTION_PANEL_SCHEMA_VERSION,
        "target_accessed": False,
        "trading_authority": False,
    }
    return Round25PredictionPanel(
        row_condition_ids=condition_ids,
        event_start_ms=starts,
        decision_time_ms=decisions,
        feature_source_chain_sha256=source_chain,
        market_prior_probability=prior,
        candidate_predictions=predictions,
        row_identity_sha256=row_identity_sha256,
        panel_sha256=_canonical_sha256(values),
    )


@dataclass(frozen=True, slots=True)
class Round25SelectionTargetAccessReceipt:
    prediction_panel_sha256: str
    selection_dataset_sha256: str
    resolution_authority_sha256: str
    one_use_claim_sha256: str
    one_use_consumption_sha256: str
    prediction_frozen_at_ns: int
    target_access_consumed_at_ns: int
    store_event_sha256: str
    receipt_sha256: str
    schema_version: str = POLYMARKET_ROUND25_SELECTION_ACCESS_RECEIPT_SCHEMA_VERSION
    selection_target_access_consumed: bool = True
    prediction_panel_frozen_before_access: bool = True
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "one_use_claim_sha256": self.one_use_claim_sha256,
            "one_use_consumption_sha256": self.one_use_consumption_sha256,
            "prediction_panel_frozen_before_access": (
                self.prediction_panel_frozen_before_access
            ),
            "prediction_frozen_at_ns": self.prediction_frozen_at_ns,
            "prediction_panel_sha256": self.prediction_panel_sha256,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "schema_version": self.schema_version,
            "selection_dataset_sha256": self.selection_dataset_sha256,
            "selection_target_access_consumed": self.selection_target_access_consumed,
            "store_event_sha256": self.store_event_sha256,
            "target_access_consumed_at_ns": self.target_access_consumed_at_ns,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        if (
            any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.prediction_panel_sha256,
                    self.selection_dataset_sha256,
                    self.resolution_authority_sha256,
                    self.one_use_claim_sha256,
                    self.one_use_consumption_sha256,
                    self.store_event_sha256,
                    self.receipt_sha256,
                )
            )
            or self.schema_version
            != POLYMARKET_ROUND25_SELECTION_ACCESS_RECEIPT_SCHEMA_VERSION
            or self.selection_target_access_consumed is not True
            or self.prediction_panel_frozen_before_access is not True
            or self.one_use_claim_sha256 == self.one_use_consumption_sha256
            or type(self.prediction_frozen_at_ns) is not int
            or type(self.target_access_consumed_at_ns) is not int
            or not 0 < self.prediction_frozen_at_ns < self.target_access_consumed_at_ns
            or self.trading_authority is not False
            or self.receipt_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 selection target access receipt differs")

    def validated(self) -> Round25SelectionTargetAccessReceipt:
        self.__post_init__()
        return self


def _create_round25_selection_target_access_receipt(
    *,
    prediction_panel_sha256: str,
    selection_dataset_sha256: str,
    resolution_authority_sha256: str,
    one_use_claim_sha256: str,
    one_use_consumption_sha256: str,
    prediction_frozen_at_ns: int,
    target_access_consumed_at_ns: int,
    store_event_sha256: str,
) -> Round25SelectionTargetAccessReceipt:
    payload = {
        "one_use_claim_sha256": one_use_claim_sha256,
        "one_use_consumption_sha256": one_use_consumption_sha256,
        "prediction_panel_frozen_before_access": True,
        "prediction_frozen_at_ns": prediction_frozen_at_ns,
        "prediction_panel_sha256": prediction_panel_sha256,
        "resolution_authority_sha256": resolution_authority_sha256,
        "schema_version": POLYMARKET_ROUND25_SELECTION_ACCESS_RECEIPT_SCHEMA_VERSION,
        "selection_dataset_sha256": selection_dataset_sha256,
        "selection_target_access_consumed": True,
        "store_event_sha256": store_event_sha256,
        "target_access_consumed_at_ns": target_access_consumed_at_ns,
        "trading_authority": False,
    }
    return Round25SelectionTargetAccessReceipt(
        prediction_panel_sha256=prediction_panel_sha256,
        selection_dataset_sha256=selection_dataset_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        one_use_claim_sha256=one_use_claim_sha256,
        one_use_consumption_sha256=one_use_consumption_sha256,
        prediction_frozen_at_ns=prediction_frozen_at_ns,
        target_access_consumed_at_ns=target_access_consumed_at_ns,
        store_event_sha256=store_event_sha256,
        receipt_sha256=_canonical_sha256(payload),
    )


def _selection_receipt_from_mapping(
    value: Mapping[str, object],
    *,
    receipt_sha256: str,
) -> Round25SelectionTargetAccessReceipt:
    expected = {
        "one_use_claim_sha256",
        "one_use_consumption_sha256",
        "prediction_panel_frozen_before_access",
        "prediction_frozen_at_ns",
        "prediction_panel_sha256",
        "resolution_authority_sha256",
        "schema_version",
        "selection_dataset_sha256",
        "selection_target_access_consumed",
        "store_event_sha256",
        "target_access_consumed_at_ns",
        "trading_authority",
    }
    if (
        set(value) != expected
        or any(
            not isinstance(value[key], str)
            for key in (
                "one_use_claim_sha256",
                "one_use_consumption_sha256",
                "prediction_panel_sha256",
                "resolution_authority_sha256",
                "schema_version",
                "selection_dataset_sha256",
                "store_event_sha256",
            )
        )
        or any(
            type(value[key]) is not int
            for key in (
                "prediction_frozen_at_ns",
                "target_access_consumed_at_ns",
            )
        )
        or any(
            type(value[key]) is not bool
            for key in (
                "prediction_panel_frozen_before_access",
                "selection_target_access_consumed",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 25 stored selection receipt differs")
    try:
        return Round25SelectionTargetAccessReceipt(
            prediction_panel_sha256=str(value["prediction_panel_sha256"]),
            selection_dataset_sha256=str(value["selection_dataset_sha256"]),
            resolution_authority_sha256=str(value["resolution_authority_sha256"]),
            one_use_claim_sha256=str(value["one_use_claim_sha256"]),
            one_use_consumption_sha256=str(value["one_use_consumption_sha256"]),
            prediction_frozen_at_ns=int(value["prediction_frozen_at_ns"]),
            target_access_consumed_at_ns=int(
                value["target_access_consumed_at_ns"]
            ),
            store_event_sha256=str(value["store_event_sha256"]),
            receipt_sha256=receipt_sha256,
            schema_version=str(value["schema_version"]),
            selection_target_access_consumed=value[
                "selection_target_access_consumed"
            ],
            prediction_panel_frozen_before_access=value[
                "prediction_panel_frozen_before_access"
            ],
            trading_authority=value["trading_authority"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 25 stored selection receipt differs") from exc


class Round25SelectionAccessStore:
    """Durable single-use boundary between target-free predictions and targets."""

    _STORE_SCHEMA_VERSION = "polymarket-round25-selection-access-store-v1"

    def __init__(self, path: str | Path) -> None:
        selected = Path(path).resolve()
        if selected.exists() and (selected.is_symlink() or not selected.is_file()):
            raise ValueError("Round 25 selection access store path differs")
        selected.parent.mkdir(parents=True, exist_ok=True)
        self._path = selected
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS round25_selection_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS round25_selection_access (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    status TEXT NOT NULL CHECK (
                        status IN ('prediction_panel_frozen', 'target_access_consumed')
                    ),
                    prediction_panel_sha256 TEXT NOT NULL CHECK (
                        length(prediction_panel_sha256) = 64
                    ),
                    one_use_claim_sha256 TEXT NOT NULL CHECK (
                        length(one_use_claim_sha256) = 64
                    ),
                    prediction_frozen_at_ns INTEGER NOT NULL,
                    selection_dataset_sha256 TEXT,
                    resolution_authority_sha256 TEXT,
                    one_use_consumption_sha256 TEXT,
                    target_access_consumed_at_ns INTEGER,
                    store_event_sha256 TEXT,
                    receipt_json TEXT,
                    receipt_sha256 TEXT
                );
                CREATE TABLE IF NOT EXISTS round25_selection_event (
                    sequence INTEGER PRIMARY KEY,
                    previous_event_sha256 TEXT NOT NULL CHECK (
                        length(previous_event_sha256) = 64
                    ),
                    event_json TEXT NOT NULL,
                    event_sha256 TEXT NOT NULL UNIQUE CHECK (
                        length(event_sha256) = 64
                    )
                );
                INSERT OR IGNORE INTO round25_selection_metadata(
                    singleton, schema_version
                ) VALUES (1, 'polymarket-round25-selection-access-store-v1');
            """)
            row = connection.execute(
                "SELECT schema_version FROM round25_selection_metadata WHERE singleton = 1"
            ).fetchone()
            if row is None or row[0] != self._STORE_SCHEMA_VERSION:
                raise ValueError("Round 25 selection access store schema differs")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        *,
        event_type: str,
        details: Mapping[str, object],
    ) -> str:
        row = connection.execute(
            "SELECT sequence, event_sha256 FROM round25_selection_event "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if row is None else int(row["sequence"]) + 1
        previous = "0" * 64 if row is None else str(row["event_sha256"])
        payload = {
            "details": dict(details),
            "event_type": event_type,
            "previous_event_sha256": previous,
            "sequence": sequence,
        }
        event_sha256 = _canonical_sha256(payload)
        connection.execute(
            "INSERT INTO round25_selection_event("
            "sequence, previous_event_sha256, event_json, event_sha256"
            ") VALUES (?, ?, ?, ?)",
            (sequence, previous, _canonical_json(payload), event_sha256),
        )
        return event_sha256

    def freeze_prediction_panel(
        self,
        *,
        panel: Round25PredictionPanel,
        one_use_claim_sha256: str,
    ) -> str:
        if not isinstance(panel, Round25PredictionPanel):
            raise TypeError("Round 25 prediction panel type differs")
        panel.validated()
        if _SHA256.fullmatch(one_use_claim_sha256) is None:
            raise ValueError("Round 25 one-use claim differs")
        frozen_at_ns = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM round25_selection_access WHERE singleton = 1"
                ).fetchone()
                if row is not None:
                    if (
                        row["status"] != "prediction_panel_frozen"
                        or row["prediction_panel_sha256"] != panel.panel_sha256
                        or row["one_use_claim_sha256"] != one_use_claim_sha256
                    ):
                        raise RuntimeError(
                            "Round 25 selection access is already bound or consumed"
                        )
                    connection.execute("COMMIT")
                    return str(row["prediction_panel_sha256"])
                connection.execute(
                    "INSERT INTO round25_selection_access("
                    "singleton, status, prediction_panel_sha256, "
                    "one_use_claim_sha256, prediction_frozen_at_ns"
                    ") VALUES (1, 'prediction_panel_frozen', ?, ?, ?)",
                    (panel.panel_sha256, one_use_claim_sha256, frozen_at_ns),
                )
                self._append_event(
                    connection,
                    event_type="prediction_panel_frozen",
                    details={
                        "one_use_claim_sha256": one_use_claim_sha256,
                        "prediction_frozen_at_ns": frozen_at_ns,
                        "prediction_panel_sha256": panel.panel_sha256,
                    },
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return panel.panel_sha256

    def consume_target_access(
        self,
        *,
        panel: Round25PredictionPanel,
        selection: Round25DevelopmentDataset,
    ) -> Round25SelectionTargetAccessReceipt:
        panel.validated()
        selection.__post_init__()
        if selection.role != "selection":
            raise ValueError("Round 25 target access requires selection role")
        require_round25_dataset_minimum(selection)
        consumed_at_ns = time.time_ns()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM round25_selection_access WHERE singleton = 1"
                ).fetchone()
                if (
                    row is None
                    or row["status"] != "prediction_panel_frozen"
                    or row["prediction_panel_sha256"] != panel.panel_sha256
                    or int(row["prediction_frozen_at_ns"]) >= consumed_at_ns
                ):
                    raise RuntimeError(
                        "Round 25 prediction panel was not frozen before target access"
                    )
                claim_sha256 = str(row["one_use_claim_sha256"])
                consumption_payload = {
                    "one_use_claim_sha256": claim_sha256,
                    "prediction_panel_sha256": panel.panel_sha256,
                    "resolution_authority_sha256": (
                        selection.resolution_authority_sha256
                    ),
                    "selection_dataset_sha256": selection.dataset_sha256,
                    "target_access_consumed_at_ns": consumed_at_ns,
                }
                consumption_sha256 = _canonical_sha256(consumption_payload)
                event_sha256 = self._append_event(
                    connection,
                    event_type="target_access_consumed",
                    details={
                        **consumption_payload,
                        "one_use_consumption_sha256": consumption_sha256,
                    },
                )
                receipt = _create_round25_selection_target_access_receipt(
                    prediction_panel_sha256=panel.panel_sha256,
                    selection_dataset_sha256=selection.dataset_sha256,
                    resolution_authority_sha256=(
                        selection.resolution_authority_sha256
                    ),
                    one_use_claim_sha256=claim_sha256,
                    one_use_consumption_sha256=consumption_sha256,
                    prediction_frozen_at_ns=int(row["prediction_frozen_at_ns"]),
                    target_access_consumed_at_ns=consumed_at_ns,
                    store_event_sha256=event_sha256,
                )
                connection.execute(
                    "UPDATE round25_selection_access SET "
                    "status = 'target_access_consumed', "
                    "selection_dataset_sha256 = ?, "
                    "resolution_authority_sha256 = ?, "
                    "one_use_consumption_sha256 = ?, "
                    "target_access_consumed_at_ns = ?, "
                    "store_event_sha256 = ?, receipt_json = ?, receipt_sha256 = ? "
                    "WHERE singleton = 1",
                    (
                        selection.dataset_sha256,
                        selection.resolution_authority_sha256,
                        consumption_sha256,
                        consumed_at_ns,
                        event_sha256,
                        _canonical_json(receipt.identity_payload()),
                        receipt.receipt_sha256,
                    ),
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return receipt

    def validate_prediction_frozen(self, *, panel: Round25PredictionPanel) -> str:
        """Verify the one-use panel lock without opening any target payload."""

        if not isinstance(panel, Round25PredictionPanel):
            raise TypeError("Round 25 prediction panel type differs")
        panel.validated()
        with self._connect() as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            row = connection.execute(
                "SELECT * FROM round25_selection_access WHERE singleton = 1"
            ).fetchone()
            events = connection.execute(
                "SELECT * FROM round25_selection_event ORDER BY sequence"
            ).fetchall()
        if row is None or len(events) != 1:
            raise RuntimeError("Round 25 prediction panel is not durably frozen")
        event = events[0]
        try:
            payload = json.loads(str(event["event_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError("Round 25 frozen prediction event differs") from exc
        claim_sha256 = str(row["one_use_claim_sha256"])
        expected_details = {
            "one_use_claim_sha256": claim_sha256,
            "prediction_frozen_at_ns": int(row["prediction_frozen_at_ns"]),
            "prediction_panel_sha256": panel.panel_sha256,
        }
        if (
            quick_check is None
            or quick_check[0] != "ok"
            or row["status"] != "prediction_panel_frozen"
            or row["prediction_panel_sha256"] != panel.panel_sha256
            or _SHA256.fullmatch(claim_sha256) is None
            or any(
                row[field] is not None
                for field in (
                    "selection_dataset_sha256",
                    "resolution_authority_sha256",
                    "one_use_consumption_sha256",
                    "target_access_consumed_at_ns",
                    "store_event_sha256",
                    "receipt_json",
                    "receipt_sha256",
                )
            )
            or int(event["sequence"]) != 1
            or event["previous_event_sha256"] != "0" * 64
            or payload
            != {
                "details": expected_details,
                "event_type": "prediction_panel_frozen",
                "previous_event_sha256": "0" * 64,
                "sequence": 1,
            }
            or str(event["event_json"]) != _canonical_json(payload)
            or event["event_sha256"] != _canonical_sha256(payload)
        ):
            raise ValueError("Round 25 frozen prediction access differs")
        return claim_sha256

    def validate_prediction_binding(
        self,
        *,
        panel: Round25PredictionPanel,
    ) -> tuple[str, str]:
        """Validate either recoverable frozen or already-consumed panel state."""

        if not isinstance(panel, Round25PredictionPanel):
            raise TypeError("Round 25 prediction panel type differs")
        panel.validated()
        with self._connect() as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            row = connection.execute(
                "SELECT * FROM round25_selection_access WHERE singleton = 1"
            ).fetchone()
            events = connection.execute(
                "SELECT * FROM round25_selection_event ORDER BY sequence"
            ).fetchall()
        if row is None:
            raise RuntimeError("Round 25 prediction panel is not bound")
        status = str(row["status"])
        claim_sha256 = str(row["one_use_claim_sha256"])
        if status == "prediction_panel_frozen":
            return status, self.validate_prediction_frozen(panel=panel)
        if (
            status != "target_access_consumed"
            or quick_check is None
            or quick_check[0] != "ok"
            or row["prediction_panel_sha256"] != panel.panel_sha256
            or _SHA256.fullmatch(claim_sha256) is None
            or len(events) != 2
        ):
            raise ValueError("Round 25 consumed prediction binding differs")
        previous = "0" * 64
        payloads: list[Mapping[str, object]] = []
        for sequence, event in enumerate(events, start=1):
            try:
                payload = json.loads(str(event["event_json"]))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Round 25 consumed prediction event differs"
                ) from exc
            if (
                not isinstance(payload, Mapping)
                or int(event["sequence"]) != sequence
                or event["previous_event_sha256"] != previous
                or payload.get("sequence") != sequence
                or payload.get("previous_event_sha256") != previous
                or str(event["event_json"]) != _canonical_json(payload)
                or event["event_sha256"] != _canonical_sha256(payload)
            ):
                raise ValueError("Round 25 consumed prediction event chain differs")
            payloads.append(payload)
            previous = str(event["event_sha256"])
        receipt_json = row["receipt_json"]
        if not isinstance(receipt_json, str):
            raise ValueError("Round 25 consumed prediction receipt differs")
        try:
            receipt_payload = json.loads(receipt_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Round 25 consumed prediction receipt differs") from exc
        receipt = _selection_receipt_from_mapping(
            receipt_payload,
            receipt_sha256=str(row["receipt_sha256"]),
        )
        first_details = payloads[0].get("details")
        second_details = payloads[1].get("details")
        if (
            payloads[0].get("event_type") != "prediction_panel_frozen"
            or not isinstance(first_details, Mapping)
            or first_details.get("one_use_claim_sha256") != claim_sha256
            or first_details.get("prediction_panel_sha256") != panel.panel_sha256
            or payloads[1].get("event_type") != "target_access_consumed"
            or not isinstance(second_details, Mapping)
            or second_details.get("one_use_claim_sha256") != claim_sha256
            or second_details.get("prediction_panel_sha256") != panel.panel_sha256
            or second_details.get("one_use_consumption_sha256")
            != receipt.one_use_consumption_sha256
            or row["store_event_sha256"] != receipt.store_event_sha256
            or previous != receipt.store_event_sha256
            or receipt.prediction_panel_sha256 != panel.panel_sha256
            or receipt.one_use_claim_sha256 != claim_sha256
        ):
            raise ValueError("Round 25 consumed prediction binding differs")
        return status, claim_sha256

    def validate_consumed(
        self,
        *,
        receipt: Round25SelectionTargetAccessReceipt,
        panel: Round25PredictionPanel,
        selection: Round25DevelopmentDataset,
    ) -> None:
        receipt.validated()
        panel.validated()
        selection.__post_init__()
        with self._connect() as connection:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            row = connection.execute(
                "SELECT * FROM round25_selection_access WHERE singleton = 1"
            ).fetchone()
            events = connection.execute(
                "SELECT * FROM round25_selection_event ORDER BY sequence"
            ).fetchall()
        previous = "0" * 64
        for expected_sequence, event in enumerate(events, start=1):
            try:
                payload = json.loads(str(event["event_json"]))
            except json.JSONDecodeError as exc:
                raise ValueError("Round 25 selection event JSON differs") from exc
            if (
                not isinstance(payload, Mapping)
                or int(event["sequence"]) != expected_sequence
                or event["previous_event_sha256"] != previous
                or payload.get("sequence") != expected_sequence
                or payload.get("previous_event_sha256") != previous
                or str(event["event_json"]) != _canonical_json(payload)
                or event["event_sha256"] != _canonical_sha256(payload)
            ):
                raise ValueError("Round 25 selection event chain differs")
            previous = str(event["event_sha256"])
        if (
            quick_check is None
            or quick_check[0] != "ok"
            or row is None
            or len(events) != 2
            or row["status"] != "target_access_consumed"
            or row["prediction_panel_sha256"] != panel.panel_sha256
            or row["selection_dataset_sha256"] != selection.dataset_sha256
            or row["resolution_authority_sha256"]
            != selection.resolution_authority_sha256
            or row["one_use_claim_sha256"] != receipt.one_use_claim_sha256
            or row["one_use_consumption_sha256"]
            != receipt.one_use_consumption_sha256
            or row["store_event_sha256"] != receipt.store_event_sha256
            or row["receipt_sha256"] != receipt.receipt_sha256
            or row["receipt_json"] != _canonical_json(receipt.identity_payload())
            or previous != receipt.store_event_sha256
        ):
            raise ValueError("Round 25 consumed selection access differs")

    def load_consumed_receipt(
        self,
        *,
        panel: Round25PredictionPanel,
        selection: Round25DevelopmentDataset,
    ) -> Round25SelectionTargetAccessReceipt:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, receipt_json, receipt_sha256 "
                "FROM round25_selection_access WHERE singleton = 1"
            ).fetchone()
        if (
            row is None
            or row["status"] != "target_access_consumed"
            or not isinstance(row["receipt_json"], str)
            or _SHA256.fullmatch(str(row["receipt_sha256"])) is None
        ):
            raise RuntimeError("Round 25 selection target access is not consumed")
        try:
            payload = json.loads(str(row["receipt_json"]))
        except json.JSONDecodeError as exc:
            raise ValueError("Round 25 stored selection receipt differs") from exc
        if (
            not isinstance(payload, Mapping)
            or str(row["receipt_json"]) != _canonical_json(payload)
        ):
            raise ValueError("Round 25 stored selection receipt differs")
        receipt = _selection_receipt_from_mapping(
            payload,
            receipt_sha256=str(row["receipt_sha256"]),
        )
        self.validate_consumed(
            receipt=receipt,
            panel=panel,
            selection=selection,
        )
        return receipt


@dataclass(frozen=True, slots=True)
class Round25CandidateMetrics:
    candidate_id: str
    condition_equal_log_loss: float
    condition_equal_brier_score: float
    expected_calibration_error: float
    balanced_accuracy: float
    roc_auc: float

    def payload(self) -> dict[str, object]:
        return {
            "balanced_accuracy": self.balanced_accuracy,
            "candidate_id": self.candidate_id,
            "condition_equal_brier_score": self.condition_equal_brier_score,
            "condition_equal_log_loss": self.condition_equal_log_loss,
            "expected_calibration_error": self.expected_calibration_error,
            "roc_auc": self.roc_auc,
        }

    def __post_init__(self) -> None:
        if (
            self.candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS
            or not math.isfinite(self.condition_equal_log_loss)
            or self.condition_equal_log_loss < 0.0
            or not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in (
                    self.condition_equal_brier_score,
                    self.expected_calibration_error,
                    self.balanced_accuracy,
                    self.roc_auc,
                )
            )
        ):
            raise ValueError("Round 25 candidate metrics differ")


@dataclass(frozen=True, slots=True)
class Round25PredictiveHypothesis:
    candidate_id: str
    metric: str
    mean_improvement: float
    bootstrap_standard_error: float
    observed_statistic: float
    adjusted_p_value: float
    step_critical_value: float
    stepdown_lower_bound: float
    passed: bool

    def payload(self) -> dict[str, object]:
        return {
            "adjusted_p_value": self.adjusted_p_value,
            "bootstrap_standard_error": self.bootstrap_standard_error,
            "candidate_id": self.candidate_id,
            "mean_improvement": self.mean_improvement,
            "metric": self.metric,
            "observed_statistic": self.observed_statistic,
            "passed": self.passed,
            "step_critical_value": self.step_critical_value,
            "stepdown_lower_bound": self.stepdown_lower_bound,
        }

    def __post_init__(self) -> None:
        finite = (
            self.mean_improvement,
            self.bootstrap_standard_error,
            self.observed_statistic,
            self.adjusted_p_value,
            self.step_critical_value,
            self.stepdown_lower_bound,
        )
        expected_pass = (
            self.bootstrap_standard_error > 0.0
            and self.adjusted_p_value <= POLYMARKET_ROUND25_ALPHA
            and self.stepdown_lower_bound > 0.0
        )
        if (
            self.candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or self.metric not in POLYMARKET_ROUND25_GATE_METRICS
            or not all(math.isfinite(value) for value in finite)
            or self.bootstrap_standard_error < 0.0
            or not 0.0 <= self.adjusted_p_value <= 1.0
            or self.passed is not expected_pass
        ):
            raise ValueError("Round 25 predictive hypothesis differs")


@dataclass(frozen=True, slots=True)
class Round25PredictiveEvaluationResult:
    prediction_panel_sha256: str
    selection_dataset_sha256: str
    resolution_authority_sha256: str
    target_access_receipt_sha256: str
    candidate_metrics: tuple[Round25CandidateMetrics, ...]
    hypotheses: tuple[Round25PredictiveHypothesis, ...]
    bootstrap_mean_sha256: str
    nominated_candidate_id: str | None
    predictive_gate_passed: bool
    result_sha256: str
    schema_version: str = POLYMARKET_ROUND25_PREDICTIVE_RESULT_SCHEMA_VERSION
    evaluation_contract_sha256: str = (
        POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
    )
    development_evidence_only: bool = True
    edge_verified: bool = False
    profitability_verified: bool = False
    ai_uplift_verified: bool = False
    paper_authority: bool = False
    live_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "ai_uplift_verified": self.ai_uplift_verified,
            "bootstrap_mean_sha256": self.bootstrap_mean_sha256,
            "candidate_metrics": [metric.payload() for metric in self.candidate_metrics],
            "development_evidence_only": self.development_evidence_only,
            "edge_verified": self.edge_verified,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "hypotheses": [hypothesis.payload() for hypothesis in self.hypotheses],
            "live_authority": self.live_authority,
            "nominated_candidate_id": self.nominated_candidate_id,
            "paper_authority": self.paper_authority,
            "prediction_panel_sha256": self.prediction_panel_sha256,
            "predictive_gate_passed": self.predictive_gate_passed,
            "profitability_verified": self.profitability_verified,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "schema_version": self.schema_version,
            "selection_dataset_sha256": self.selection_dataset_sha256,
            "target_access_receipt_sha256": self.target_access_receipt_sha256,
        }

    def __post_init__(self) -> None:
        eligible = tuple(
            candidate_id
            for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            if all(
                hypothesis.passed
                for hypothesis in self.hypotheses
                if hypothesis.candidate_id == candidate_id
            )
            and sum(
                hypothesis.candidate_id == candidate_id
                for hypothesis in self.hypotheses
            )
            == len(POLYMARKET_ROUND25_GATE_METRICS)
        )
        metric_by_id = {metric.candidate_id: metric for metric in self.candidate_metrics}
        expected_nomination = min(
            eligible,
            key=lambda candidate_id: (
                metric_by_id[candidate_id].condition_equal_log_loss,
                metric_by_id[candidate_id].condition_equal_brier_score,
                metric_by_id[candidate_id].expected_calibration_error,
                -metric_by_id[candidate_id].balanced_accuracy,
                -metric_by_id[candidate_id].roc_auc,
                POLYMARKET_ROUND25_CANDIDATE_IDS.index(candidate_id),
            ),
            default=None,
        )
        if (
            tuple(metric.candidate_id for metric in self.candidate_metrics)
            != POLYMARKET_ROUND25_CANDIDATE_IDS
            or len(self.hypotheses) != 10
            or any(metric.__post_init__() is not None for metric in self.candidate_metrics)
            or any(
                hypothesis.__post_init__() is not None
                for hypothesis in self.hypotheses
            )
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.prediction_panel_sha256,
                    self.selection_dataset_sha256,
                    self.resolution_authority_sha256,
                    self.target_access_receipt_sha256,
                    self.bootstrap_mean_sha256,
                    self.result_sha256,
                )
            )
            or self.nominated_candidate_id != expected_nomination
            or self.predictive_gate_passed is not (self.nominated_candidate_id is not None)
            or self.schema_version != POLYMARKET_ROUND25_PREDICTIVE_RESULT_SCHEMA_VERSION
            or self.evaluation_contract_sha256
            != POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
            or self.development_evidence_only is not True
            or any(
                value is not False
                for value in (
                    self.edge_verified,
                    self.profitability_verified,
                    self.ai_uplift_verified,
                    self.paper_authority,
                    self.live_authority,
                )
            )
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 predictive evaluation result differs")

    def validated(self) -> Round25PredictiveEvaluationResult:
        self.__post_init__()
        return self

    def serialized_payload(self) -> dict[str, object]:
        self.validated()
        return {**self.identity_payload(), "result_sha256": self.result_sha256}


def _condition_losses(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(
        probabilities,
        POLYMARKET_ROUND25_LOG_LOSS_CLIP,
        1.0 - POLYMARKET_ROUND25_LOG_LOSS_CLIP,
    )
    log_loss = -(
        labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)
    )
    brier = (probabilities - labels) ** 2
    return (
        np.mean(log_loss.reshape(-1, 16), axis=1),
        np.mean(brier.reshape(-1, 16), axis=1),
    )


def _descriptive_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    condition_log_loss: np.ndarray,
    condition_brier: np.ndarray,
) -> tuple[float, float, float, float, float]:
    order = np.argsort(probabilities, kind="stable")
    bins = np.array_split(order, 10)
    ece = sum(
        len(indices)
        / len(labels)
        * abs(float(np.mean(probabilities[indices]) - np.mean(labels[indices])))
        for indices in bins
    )
    predicted = probabilities >= 0.5
    positives = labels == 1.0
    negatives = ~positives
    if not np.any(positives) or not np.any(negatives):
        raise ValueError("Round 25 selection population lacks both target classes")
    balanced_accuracy = 0.5 * (
        float(np.mean(predicted[positives]))
        + float(np.mean(~predicted[negatives]))
    )
    sorted_probability = probabilities[order]
    sorted_labels = labels[order]
    cumulative_negative = 0.0
    concordance = 0.0
    offset = 0
    while offset < len(labels):
        end = offset + 1
        while end < len(labels) and sorted_probability[end] == sorted_probability[offset]:
            end += 1
        group = sorted_labels[offset:end]
        group_positive = float(np.sum(group == 1.0))
        group_negative = float(np.sum(group == 0.0))
        concordance += group_positive * (
            cumulative_negative + 0.5 * group_negative
        )
        cumulative_negative += group_negative
        offset = end
    roc_auc = concordance / (float(np.sum(positives)) * float(np.sum(negatives)))
    return (
        float(np.mean(condition_log_loss)),
        float(np.mean(condition_brier)),
        float(ece),
        float(balanced_accuracy),
        float(roc_auc),
    )


def _bootstrap_means(improvements: np.ndarray) -> np.ndarray:
    condition_count, hypothesis_count = improvements.shape
    rng = np.random.Generator(np.random.PCG64(POLYMARKET_ROUND25_BOOTSTRAP_SEED))
    block_count = math.ceil(
        condition_count / POLYMARKET_ROUND25_BOOTSTRAP_BLOCK_CONDITIONS
    )
    offsets = np.arange(
        POLYMARKET_ROUND25_BOOTSTRAP_BLOCK_CONDITIONS,
        dtype=np.int64,
    )
    output = np.empty(
        (POLYMARKET_ROUND25_BOOTSTRAP_REPLICATES, hypothesis_count),
        dtype=np.float64,
    )
    for first in range(
        0,
        POLYMARKET_ROUND25_BOOTSTRAP_REPLICATES,
        POLYMARKET_ROUND25_BOOTSTRAP_CHUNK_REPLICATES,
    ):
        count = min(
            POLYMARKET_ROUND25_BOOTSTRAP_CHUNK_REPLICATES,
            POLYMARKET_ROUND25_BOOTSTRAP_REPLICATES - first,
        )
        starts = rng.integers(
            0,
            condition_count,
            size=(count, block_count),
            dtype=np.int64,
        )
        indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % condition_count
        indices = indices.reshape(count, -1)[:, :condition_count]
        output[first : first + count] = np.mean(
            improvements[indices],
            axis=1,
        )
    return output


def _stepdown_hypotheses(
    improvements: np.ndarray,
    bootstrap_means: np.ndarray,
) -> tuple[Round25PredictiveHypothesis, ...]:
    observed = np.mean(improvements, axis=0)
    standard_error = np.std(bootstrap_means, axis=0, ddof=1)
    valid = np.isfinite(standard_error) & (standard_error > 0.0)
    observed_statistic = np.zeros_like(observed)
    observed_statistic[valid] = observed[valid] / standard_error[valid]
    centered = np.zeros_like(bootstrap_means)
    centered[:, valid] = (
        bootstrap_means[:, valid] - observed[None, valid]
    ) / standard_error[None, valid]
    order = tuple(sorted(
        range(len(observed)),
        key=lambda index: (
            not bool(valid[index]),
            -float(observed_statistic[index]),
            index,
        ),
    ))
    results: list[Round25PredictiveHypothesis | None] = [None] * len(observed)
    running_p = 0.0
    remaining = list(order)
    for index in order:
        if not valid[index]:
            results[index] = Round25PredictiveHypothesis(
                candidate_id=POLYMARKET_ROUND25_CANDIDATE_IDS[1 + index // 2],
                metric=POLYMARKET_ROUND25_GATE_METRICS[index % 2],
                mean_improvement=float(observed[index]),
                bootstrap_standard_error=0.0,
                observed_statistic=0.0,
                adjusted_p_value=1.0,
                step_critical_value=0.0,
                stepdown_lower_bound=min(float(observed[index]), 0.0),
                passed=False,
            )
            remaining.remove(index)
            continue
        active = [candidate for candidate in remaining if valid[candidate]]
        null_maximum = np.max(centered[:, active], axis=1)
        step_p = (
            1.0
            + float(np.sum(null_maximum >= observed_statistic[index]))
        ) / (POLYMARKET_ROUND25_BOOTSTRAP_REPLICATES + 1.0)
        running_p = max(running_p, step_p)
        critical = float(
            np.quantile(
                null_maximum,
                1.0 - POLYMARKET_ROUND25_ALPHA,
                method="higher",
            )
        )
        lower = float(observed[index] - critical * standard_error[index])
        passed = running_p <= POLYMARKET_ROUND25_ALPHA and lower > 0.0
        results[index] = Round25PredictiveHypothesis(
            candidate_id=POLYMARKET_ROUND25_CANDIDATE_IDS[1 + index // 2],
            metric=POLYMARKET_ROUND25_GATE_METRICS[index % 2],
            mean_improvement=float(observed[index]),
            bootstrap_standard_error=float(standard_error[index]),
            observed_statistic=float(observed_statistic[index]),
            adjusted_p_value=float(running_p),
            step_critical_value=critical,
            stepdown_lower_bound=lower,
            passed=passed,
        )
        remaining.remove(index)
    if any(result is None for result in results):
        raise RuntimeError("Round 25 stepdown result is incomplete")
    return tuple(result for result in results if result is not None)


def evaluate_round25_predictive_candidates(
    *,
    panel: Round25PredictionPanel,
    selection: Round25DevelopmentDataset,
    target_access_receipt: Round25SelectionTargetAccessReceipt,
    target_access_store: Round25SelectionAccessStore,
) -> Round25PredictiveEvaluationResult:
    if not isinstance(panel, Round25PredictionPanel):
        raise TypeError("Round 25 prediction panel type differs")
    if not isinstance(selection, Round25DevelopmentDataset):
        raise TypeError("Round 25 selection dataset type differs")
    if not isinstance(target_access_receipt, Round25SelectionTargetAccessReceipt):
        raise TypeError("Round 25 target access receipt type differs")
    if not isinstance(target_access_store, Round25SelectionAccessStore):
        raise TypeError("Round 25 target access store type differs")
    panel.validated()
    selection.__post_init__()
    target_access_receipt.validated()
    target_access_store.validate_consumed(
        receipt=target_access_receipt,
        panel=panel,
        selection=selection,
    )
    if selection.role != "selection":
        raise ValueError("Round 25 predictive evaluation requires selection role")
    require_round25_dataset_minimum(selection)
    samples = selection.samples
    if (
        target_access_receipt.prediction_panel_sha256 != panel.panel_sha256
        or target_access_receipt.selection_dataset_sha256
        != selection.dataset_sha256
        or target_access_receipt.resolution_authority_sha256
        != selection.resolution_authority_sha256
        or tuple(sample.condition_id for sample in samples) != panel.row_condition_ids
        or not np.array_equal(
            np.asarray([sample.event_start_ms for sample in samples], dtype="<i8"),
            panel.event_start_ms,
        )
        or not np.array_equal(
            np.asarray([sample.decision_time_ms for sample in samples], dtype="<i8"),
            panel.decision_time_ms,
        )
        or tuple(sample.feature_source_chain_sha256 for sample in samples)
        != panel.feature_source_chain_sha256
        or not np.array_equal(
            np.asarray(
                [sample.market_prior_probability for sample in samples],
                dtype="<f8",
            ),
            panel.market_prior_probability,
        )
    ):
        raise ValueError("Round 25 prediction panel and selection target differ")
    labels = np.asarray([sample.target_up for sample in samples], dtype=np.float64)
    if len(np.unique(labels)) != 2:
        raise ValueError("Round 25 selection population lacks both target classes")
    condition_losses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    metrics: list[Round25CandidateMetrics] = []
    for prediction in panel.candidate_predictions:
        log_loss, brier = _condition_losses(labels, prediction.probabilities)
        condition_losses[prediction.candidate_id] = (log_loss, brier)
        metric_values = _descriptive_metrics(
            labels,
            prediction.probabilities,
            log_loss,
            brier,
        )
        metrics.append(Round25CandidateMetrics(
            candidate_id=prediction.candidate_id,
            condition_equal_log_loss=metric_values[0],
            condition_equal_brier_score=metric_values[1],
            expected_calibration_error=metric_values[2],
            balanced_accuracy=metric_values[3],
            roc_auc=metric_values[4],
        ))
    control_log_loss, control_brier = condition_losses["market-prior-v1"]
    improvement_columns: list[np.ndarray] = []
    for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]:
        candidate_log_loss, candidate_brier = condition_losses[candidate_id]
        improvement_columns.extend((
            control_log_loss - candidate_log_loss,
            control_brier - candidate_brier,
        ))
    improvements = np.column_stack(improvement_columns).astype(np.float64)
    bootstrap_means = _bootstrap_means(improvements)
    hypotheses = _stepdown_hypotheses(improvements, bootstrap_means)
    eligible = tuple(
        candidate_id
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
        if all(
            hypothesis.passed
            for hypothesis in hypotheses
            if hypothesis.candidate_id == candidate_id
        )
        and sum(
            hypothesis.candidate_id == candidate_id for hypothesis in hypotheses
        )
        == 2
    )
    metric_by_id = {metric.candidate_id: metric for metric in metrics}
    nominated = min(
        eligible,
        key=lambda candidate_id: (
            metric_by_id[candidate_id].condition_equal_log_loss,
            metric_by_id[candidate_id].condition_equal_brier_score,
            metric_by_id[candidate_id].expected_calibration_error,
            -metric_by_id[candidate_id].balanced_accuracy,
            -metric_by_id[candidate_id].roc_auc,
            POLYMARKET_ROUND25_CANDIDATE_IDS.index(candidate_id),
        ),
        default=None,
    )
    bootstrap_sha256 = _array_sha256(bootstrap_means)
    values = {
        "ai_uplift_verified": False,
        "bootstrap_mean_sha256": bootstrap_sha256,
        "candidate_metrics": [metric.payload() for metric in metrics],
        "development_evidence_only": True,
        "edge_verified": False,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
        ),
        "hypotheses": [hypothesis.payload() for hypothesis in hypotheses],
        "live_authority": False,
        "nominated_candidate_id": nominated,
        "paper_authority": False,
        "prediction_panel_sha256": panel.panel_sha256,
        "predictive_gate_passed": nominated is not None,
        "profitability_verified": False,
        "resolution_authority_sha256": selection.resolution_authority_sha256,
        "schema_version": POLYMARKET_ROUND25_PREDICTIVE_RESULT_SCHEMA_VERSION,
        "selection_dataset_sha256": selection.dataset_sha256,
        "target_access_receipt_sha256": target_access_receipt.receipt_sha256,
    }
    return Round25PredictiveEvaluationResult(
        prediction_panel_sha256=panel.panel_sha256,
        selection_dataset_sha256=selection.dataset_sha256,
        resolution_authority_sha256=selection.resolution_authority_sha256,
        target_access_receipt_sha256=target_access_receipt.receipt_sha256,
        candidate_metrics=tuple(metrics),
        hypotheses=hypotheses,
        bootstrap_mean_sha256=bootstrap_sha256,
        nominated_candidate_id=nominated,
        predictive_gate_passed=nominated is not None,
        result_sha256=_canonical_sha256(values),
    )


def write_round25_predictive_result(
    path: str | Path,
    result: Round25PredictiveEvaluationResult,
) -> Path:
    if not isinstance(result, Round25PredictiveEvaluationResult):
        raise TypeError("Round 25 predictive result type differs")
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Round 25 predictive result path differs")
    payload = (_canonical_json(result.serialized_payload()) + "\n").encode("ascii")
    if len(payload) > POLYMARKET_ROUND25_PREDICTIVE_RESULT_MAXIMUM_BYTES:
        raise ValueError("Round 25 predictive result exceeds its storage bound")
    if target.exists():
        if load_round25_predictive_result(target).result_sha256 == result.result_sha256:
            return target
        raise FileExistsError("Round 25 predictive result path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def load_round25_predictive_result(
    path: str | Path,
) -> Round25PredictiveEvaluationResult:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or not 2
        <= source.stat().st_size
        <= POLYMARKET_ROUND25_PREDICTIVE_RESULT_MAXIMUM_BYTES
    ):
        raise ValueError("Round 25 predictive result file differs")
    try:
        value = json.loads(
            source.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 predictive result is unreadable") from exc
    expected = {
        "ai_uplift_verified",
        "bootstrap_mean_sha256",
        "candidate_metrics",
        "development_evidence_only",
        "edge_verified",
        "evaluation_contract_sha256",
        "hypotheses",
        "live_authority",
        "nominated_candidate_id",
        "paper_authority",
        "prediction_panel_sha256",
        "predictive_gate_passed",
        "profitability_verified",
        "resolution_authority_sha256",
        "result_sha256",
        "schema_version",
        "selection_dataset_sha256",
        "target_access_receipt_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not isinstance(value.get("candidate_metrics"), list)
        or not isinstance(value.get("hypotheses"), list)
        or any(
            type(value.get(field)) is not bool
            for field in (
                "ai_uplift_verified",
                "development_evidence_only",
                "edge_verified",
                "live_authority",
                "paper_authority",
                "predictive_gate_passed",
                "profitability_verified",
            )
        )
    ):
        raise ValueError("Round 25 predictive result payload differs")
    try:
        metrics = tuple(
            Round25CandidateMetrics(**dict(item))
            for item in value["candidate_metrics"]
            if isinstance(item, Mapping)
        )
        hypotheses = tuple(
            Round25PredictiveHypothesis(**dict(item))
            for item in value["hypotheses"]
            if isinstance(item, Mapping)
        )
        result = Round25PredictiveEvaluationResult(
            prediction_panel_sha256=str(value["prediction_panel_sha256"]),
            selection_dataset_sha256=str(value["selection_dataset_sha256"]),
            resolution_authority_sha256=str(
                value["resolution_authority_sha256"]
            ),
            target_access_receipt_sha256=str(
                value["target_access_receipt_sha256"]
            ),
            candidate_metrics=metrics,
            hypotheses=hypotheses,
            bootstrap_mean_sha256=str(value["bootstrap_mean_sha256"]),
            nominated_candidate_id=(
                None
                if value["nominated_candidate_id"] is None
                else str(value["nominated_candidate_id"])
            ),
            predictive_gate_passed=value["predictive_gate_passed"],
            result_sha256=str(value["result_sha256"]),
            schema_version=str(value["schema_version"]),
            evaluation_contract_sha256=str(value["evaluation_contract_sha256"]),
            development_evidence_only=value["development_evidence_only"],
            edge_verified=value["edge_verified"],
            profitability_verified=value["profitability_verified"],
            ai_uplift_verified=value["ai_uplift_verified"],
            paper_authority=value["paper_authority"],
            live_authority=value["live_authority"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 25 predictive result payload differs") from exc
    if (
        len(metrics) != len(value["candidate_metrics"])
        or len(hypotheses) != len(value["hypotheses"])
        or dict(value) != result.serialized_payload()
    ):
        raise ValueError("Round 25 predictive result serialization differs")
    return result


__all__ = [
    "POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_PREDICTIVE_RESULT_MAXIMUM_BYTES",
    "Round25CandidateMetrics",
    "Round25CandidatePrediction",
    "Round25PredictionPanel",
    "Round25PredictiveEvaluationResult",
    "Round25PredictiveHypothesis",
    "Round25SelectionTargetAccessReceipt",
    "Round25SelectionAccessStore",
    "create_round25_prediction_panel",
    "evaluate_round25_predictive_candidates",
    "load_round25_predictive_result",
    "write_round25_predictive_result",
]
