"""Target-free prediction preparation for the fitted Round 25 ledger."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from .polymarket_round25_controls import (
    predict_round25_logistic_residual_probability,
    predict_round25_phase_isotonic_probability,
)
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_MINIMUM_CONDITIONS,
    round25_development_role,
    select_round25_condition_endpoints,
)
from .polymarket_round25_evaluation import (
    Round25PredictionPanel,
    Round25SelectionAccessStore,
    create_round25_prediction_panel,
)
from .polymarket_round25_joint_features import Round25JointFeatureSnapshot
from .polymarket_round25_lightgbm import Round25CompiledLightGBM
from .polymarket_round25_model_ledger import (
    POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256,
    Round25ModelLedger,
)
from .polymarket_round25_sequence import (
    POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256,
    build_round25_sequence_inference_batch,
)
from .polymarket_round25_tcn import Round25CompiledTCNEnsemble


POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION = (
    "polymarket-round25-prepared-target-free-prediction-v1"
)
POLYMARKET_ROUND25_PREPARED_PREDICTION_MAXIMUM_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 25 prepared prediction contains duplicate keys")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 prepared prediction contains {value}")


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


def _derived_prediction_source_sha256(
    *,
    candidate_id: str,
    model_artifact_sha256: str,
    model_ledger_sha256: str,
    source_receipt_audit_sha256: str,
) -> str:
    return _canonical_sha256({
        "candidate_id": candidate_id,
        "model_artifact_sha256": model_artifact_sha256,
        "model_ledger_contract_sha256": (
            POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256
        ),
        "model_ledger_sha256": model_ledger_sha256,
        "source_receipt_audit_sha256": source_receipt_audit_sha256,
        "target_free_sequence_contract_sha256": (
            POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
        ),
    })


@dataclass(frozen=True, slots=True)
class Round25PreparedPrediction:
    panel: Round25PredictionPanel
    model_ledger_sha256: str
    source_receipt_audit_sha256: str
    candidate_model_artifact_sha256: tuple[tuple[str, str], ...]
    sequence_inference_batch_sha256: tuple[str, ...]
    prepared_sha256: str
    schema_version: str = POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION
    model_ledger_contract_sha256: str = (
        POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256
    )
    target_free_sequence_contract_sha256: str = (
        POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
    )
    selection_target_accessed: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_model_artifact_sha256": [
                {"candidate_id": candidate_id, "sha256": digest}
                for candidate_id, digest in self.candidate_model_artifact_sha256
            ],
            "condition_count": len(self.sequence_inference_batch_sha256),
            "model_ledger_contract_sha256": self.model_ledger_contract_sha256,
            "model_ledger_sha256": self.model_ledger_sha256,
            "prediction_panel_sha256": self.panel.panel_sha256,
            "schema_version": self.schema_version,
            "selection_target_accessed": self.selection_target_accessed,
            "sequence_inference_batch_sha256": list(
                self.sequence_inference_batch_sha256
            ),
            "source_receipt_audit_sha256": self.source_receipt_audit_sha256,
            "target_free_sequence_contract_sha256": (
                self.target_free_sequence_contract_sha256
            ),
            "trading_authority": self.trading_authority,
        }

    def serialized_payload(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "panel": {
                "candidate_predictions": [
                    {
                        "candidate_id": prediction.candidate_id,
                        "probabilities": prediction.probabilities.tolist(),
                        "source_artifact_sha256": (
                            prediction.source_artifact_sha256
                        ),
                    }
                    for prediction in self.panel.candidate_predictions
                ],
                "decision_time_ms": self.panel.decision_time_ms.tolist(),
                "event_start_ms": self.panel.event_start_ms.tolist(),
                "feature_source_chain_sha256": list(
                    self.panel.feature_source_chain_sha256
                ),
                "market_prior_probability": (
                    self.panel.market_prior_probability.tolist()
                ),
                "panel_sha256": self.panel.panel_sha256,
                "row_condition_ids": list(self.panel.row_condition_ids),
            },
            "prepared_sha256": self.prepared_sha256,
        }

    def __post_init__(self) -> None:
        self.panel.validated()
        candidate_ids = tuple(
            candidate_id
            for candidate_id, _digest in self.candidate_model_artifact_sha256
        )
        derived = tuple(
            _derived_prediction_source_sha256(
                candidate_id=candidate_id,
                model_artifact_sha256=model_sha256,
                model_ledger_sha256=self.model_ledger_sha256,
                source_receipt_audit_sha256=self.source_receipt_audit_sha256,
            )
            for candidate_id, model_sha256 in self.candidate_model_artifact_sha256
        )
        if (
            _SHA256.fullmatch(self.model_ledger_sha256) is None
            or _SHA256.fullmatch(self.source_receipt_audit_sha256) is None
            or candidate_ids != POLYMARKET_ROUND25_CANDIDATE_IDS
            or any(
                _SHA256.fullmatch(digest) is None
                for _candidate_id, digest in self.candidate_model_artifact_sha256
            )
            or len({digest for _candidate_id, digest in self.candidate_model_artifact_sha256})
            != len(POLYMARKET_ROUND25_CANDIDATE_IDS)
            or len(self.sequence_inference_batch_sha256)
            < POLYMARKET_ROUND25_MINIMUM_CONDITIONS["selection"]
            or len(set(self.sequence_inference_batch_sha256))
            != len(self.sequence_inference_batch_sha256)
            or any(
                _SHA256.fullmatch(digest) is None
                for digest in self.sequence_inference_batch_sha256
            )
            or tuple(
                prediction.source_artifact_sha256
                for prediction in self.panel.candidate_predictions
            )
            != derived
            or len(self.panel.row_condition_ids)
            != len(self.sequence_inference_batch_sha256) * 16
            or self.schema_version
            != POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION
            or self.model_ledger_contract_sha256
            != POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256
            or self.target_free_sequence_contract_sha256
            != POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
            or self.selection_target_accessed is not False
            or self.trading_authority is not False
            or _SHA256.fullmatch(self.prepared_sha256) is None
            or self.prepared_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 prepared target-free prediction differs")

    def validated(self) -> Round25PreparedPrediction:
        self.__post_init__()
        return self


def prepare_round25_target_free_prediction(
    *,
    ledger: Round25ModelLedger,
    snapshots: Sequence[Round25JointFeatureSnapshot],
    source_receipt_audit_sha256: str,
    tcn_backend: str = "auto",
) -> Round25PreparedPrediction:
    if not isinstance(ledger, Round25ModelLedger):
        raise TypeError("Round 25 fitted-model ledger type differs")
    ledger.validated()
    if _SHA256.fullmatch(source_receipt_audit_sha256) is None:
        raise ValueError("Round 25 source receipt audit identity differs")
    grouped: dict[str, list[Round25JointFeatureSnapshot]] = {}
    for row in snapshots:
        if (
            not isinstance(row, Round25JointFeatureSnapshot)
            or not row.available
            or round25_development_role(row.event_start_ms) != "selection"
        ):
            raise ValueError("Round 25 target-free selection feature differs")
        grouped.setdefault(row.condition_id, []).append(row)
    if len(grouped) < POLYMARKET_ROUND25_MINIMUM_CONDITIONS["selection"]:
        raise ValueError("Round 25 target-free selection minimum gate failed")
    ordered_groups = tuple(sorted(
        grouped.values(),
        key=lambda rows: (rows[0].event_start_ms, rows[0].condition_id),
    ))
    endpoint_groups = tuple(
        select_round25_condition_endpoints(rows) for rows in ordered_groups
    )
    endpoints = tuple(row for group in endpoint_groups for row in group)
    feature_matrix = tuple(row.values for row in endpoints)
    market_prior = tuple(row.market_prior_probability for row in endpoints)

    phase_probability = tuple(
        predict_round25_phase_isotonic_probability(
            ledger.phase_isotonic,
            event_start_ms=row.event_start_ms,
            decision_time_ms=row.decision_time_ms,
            market_prior_probability=row.market_prior_probability,
        )
        for row in endpoints
    )
    logistic_probability = tuple(
        predict_round25_logistic_residual_probability(
            ledger.logistic_residual,
            feature_values=row.values,
            market_prior_probability=row.market_prior_probability,
        )
        for row in endpoints
    )
    tree_probability = tuple(
        Round25CompiledLightGBM(artifact).predict_probabilities(
            feature_matrix,
            market_prior,
        )
        for artifact in ledger.lightgbm_residuals
    )
    tcn_runtime = Round25CompiledTCNEnsemble(
        ledger.tcn_ensemble,
        compute_backend=tcn_backend,
    )
    tcn_probability: list[float] = []
    inference_hashes: list[str] = []
    for rows, expected_endpoints in zip(
        ordered_groups,
        endpoint_groups,
        strict=True,
    ):
        batch = build_round25_sequence_inference_batch(
            snapshots=rows,
            center=ledger.logistic_residual.center,
            scale=ledger.logistic_residual.scale,
            source_receipt_audit_sha256=source_receipt_audit_sha256,
        )
        if (
            tuple(row.decision_time_ms for row in expected_endpoints)
            != tuple(int(value) for value in batch.decision_time_ms)
            or tuple(row.source_chain_sha256 for row in expected_endpoints)
            != batch.endpoint_source_chain_sha256
            or not np.array_equal(
                np.asarray(
                    [row.market_prior_probability for row in expected_endpoints],
                    dtype="<f8",
                ),
                batch.terminal_market_prior,
            )
        ):
            raise ValueError("Round 25 target-free sequence and tabular rows differ")
        tcn_probability.extend(tcn_runtime.predict_probabilities(
            batch.sequence_values,
            batch.terminal_market_prior,
        ))
        inference_hashes.append(batch.batch_sha256)

    raw_artifacts = ledger.candidate_artifact_sha256()
    prediction_sources = {
        candidate_id: _derived_prediction_source_sha256(
            candidate_id=candidate_id,
            model_artifact_sha256=model_sha256,
            model_ledger_sha256=ledger.ledger_sha256,
            source_receipt_audit_sha256=source_receipt_audit_sha256,
        )
        for candidate_id, model_sha256 in raw_artifacts
    }
    candidate_probabilities = {
        "market-prior-v1": market_prior,
        "phase-isotonic-market-prior-v1": phase_probability,
        "l2-logistic-residual-v1": logistic_probability,
        POLYMARKET_ROUND25_CANDIDATE_IDS[3]: tree_probability[0],
        POLYMARKET_ROUND25_CANDIDATE_IDS[4]: tree_probability[1],
        "causal-multitask-tcn-residual-v1": tuple(tcn_probability),
    }
    panel = create_round25_prediction_panel(
        row_condition_ids=tuple(row.condition_id for row in endpoints),
        event_start_ms=tuple(row.event_start_ms for row in endpoints),
        decision_time_ms=tuple(row.decision_time_ms for row in endpoints),
        feature_source_chain_sha256=tuple(
            row.source_chain_sha256 for row in endpoints
        ),
        market_prior_probability=market_prior,
        candidate_probabilities=candidate_probabilities,
        candidate_source_artifact_sha256=prediction_sources,
    )
    identity = {
        "candidate_model_artifact_sha256": [
            {"candidate_id": candidate_id, "sha256": digest}
            for candidate_id, digest in raw_artifacts
        ],
        "condition_count": len(inference_hashes),
        "model_ledger_contract_sha256": POLYMARKET_ROUND25_MODEL_LEDGER_CONTRACT_SHA256,
        "model_ledger_sha256": ledger.ledger_sha256,
        "prediction_panel_sha256": panel.panel_sha256,
        "schema_version": POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION,
        "selection_target_accessed": False,
        "sequence_inference_batch_sha256": inference_hashes,
        "source_receipt_audit_sha256": source_receipt_audit_sha256,
        "target_free_sequence_contract_sha256": (
            POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
        ),
        "trading_authority": False,
    }
    return Round25PreparedPrediction(
        panel=panel,
        model_ledger_sha256=ledger.ledger_sha256,
        source_receipt_audit_sha256=source_receipt_audit_sha256,
        candidate_model_artifact_sha256=raw_artifacts,
        sequence_inference_batch_sha256=tuple(inference_hashes),
        prepared_sha256=_canonical_sha256(identity),
    )


def write_round25_prepared_prediction(
    path: str | Path,
    prepared: Round25PreparedPrediction,
) -> Path:
    if not isinstance(prepared, Round25PreparedPrediction):
        raise TypeError("Round 25 prepared prediction type differs")
    prepared.validated()
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Round 25 prepared prediction path differs")
    payload = (_canonical_json(prepared.serialized_payload()) + "\n").encode("ascii")
    if len(payload) > POLYMARKET_ROUND25_PREPARED_PREDICTION_MAXIMUM_BYTES:
        raise ValueError("Round 25 prepared prediction exceeds its storage bound")
    if target.exists():
        if load_round25_prepared_prediction(target).prepared_sha256 == prepared.prepared_sha256:
            return target
        raise FileExistsError("Round 25 prepared prediction path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 25 {label} is not an object")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"Round 25 {label} is not an array")
    return value


def load_round25_prepared_prediction(path: str | Path) -> Round25PreparedPrediction:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size > POLYMARKET_ROUND25_PREPARED_PREDICTION_MAXIMUM_BYTES
    ):
        raise ValueError("Round 25 prepared prediction file differs")
    try:
        decoded = json.loads(
            source.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 prepared prediction is unreadable") from exc
    value = _mapping(decoded, label="prepared prediction")
    panel_value = _mapping(value.get("panel"), label="prediction panel")
    prediction_values = _list(
        panel_value.get("candidate_predictions"),
        label="candidate predictions",
    )
    candidate_probabilities: dict[str, Sequence[float]] = {}
    candidate_sources: dict[str, str] = {}
    for raw in prediction_values:
        item = _mapping(raw, label="candidate prediction")
        candidate_id = str(item.get("candidate_id"))
        candidate_probabilities[candidate_id] = tuple(
            float(number)
            for number in _list(item.get("probabilities"), label="probabilities")
        )
        candidate_sources[candidate_id] = str(item.get("source_artifact_sha256"))
    panel = create_round25_prediction_panel(
        row_condition_ids=tuple(
            str(item)
            for item in _list(
                panel_value.get("row_condition_ids"),
                label="row condition IDs",
            )
        ),
        event_start_ms=tuple(
            int(item)
            for item in _list(
                panel_value.get("event_start_ms"),
                label="event starts",
            )
        ),
        decision_time_ms=tuple(
            int(item)
            for item in _list(
                panel_value.get("decision_time_ms"),
                label="decision times",
            )
        ),
        feature_source_chain_sha256=tuple(
            str(item)
            for item in _list(
                panel_value.get("feature_source_chain_sha256"),
                label="feature source chains",
            )
        ),
        market_prior_probability=tuple(
            float(item)
            for item in _list(
                panel_value.get("market_prior_probability"),
                label="market priors",
            )
        ),
        candidate_probabilities=candidate_probabilities,
        candidate_source_artifact_sha256=candidate_sources,
    )
    if panel_value.get("panel_sha256") != panel.panel_sha256:
        raise ValueError("Round 25 serialized prediction panel differs")
    raw_artifacts = tuple(
        (str(item.get("candidate_id")), str(item.get("sha256")))
        for raw in _list(
            value.get("candidate_model_artifact_sha256"),
            label="candidate artifact hashes",
        )
        for item in (_mapping(raw, label="candidate artifact hash"),)
    )
    prepared = Round25PreparedPrediction(
        panel=panel,
        model_ledger_sha256=str(value.get("model_ledger_sha256")),
        source_receipt_audit_sha256=str(
            value.get("source_receipt_audit_sha256")
        ),
        candidate_model_artifact_sha256=raw_artifacts,
        sequence_inference_batch_sha256=tuple(
            str(item)
            for item in _list(
                value.get("sequence_inference_batch_sha256"),
                label="sequence inference hashes",
            )
        ),
        prepared_sha256=str(value.get("prepared_sha256")),
    )
    if dict(value) != prepared.serialized_payload():
        raise ValueError("Round 25 serialized prepared prediction differs")
    return prepared


def freeze_round25_prepared_prediction(
    *,
    store: Round25SelectionAccessStore,
    prepared: Round25PreparedPrediction,
    one_use_claim_sha256: str,
) -> str:
    """Lock target access only after validating the complete target-free artifact."""
    if not isinstance(store, Round25SelectionAccessStore):
        raise TypeError("Round 25 selection access store type differs")
    if not isinstance(prepared, Round25PreparedPrediction):
        raise TypeError("Round 25 prepared prediction type differs")
    prepared.validated()
    return store.freeze_prediction_panel(
        panel=prepared.panel,
        one_use_claim_sha256=one_use_claim_sha256,
    )


__all__ = [
    "POLYMARKET_ROUND25_PREPARED_PREDICTION_MAXIMUM_BYTES",
    "POLYMARKET_ROUND25_PREPARED_PREDICTION_SCHEMA_VERSION",
    "Round25PreparedPrediction",
    "freeze_round25_prepared_prediction",
    "load_round25_prepared_prediction",
    "prepare_round25_target_free_prediction",
    "write_round25_prepared_prediction",
]
