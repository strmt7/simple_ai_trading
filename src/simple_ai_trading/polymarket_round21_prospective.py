"""Target-free prospective scoring for the independent Polymarket Round 21 bot."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import re
from threading import Lock
import time

import numpy as np

from .polymarket_round21_core_features import POLYMARKET_ROUND21_FEATURE_SCHEMA
from .polymarket_round21_dataset import (
    POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    POLYMARKET_ROUND21_DECISION_CADENCE_MS,
    Round21CausalFeatureRow,
)
from .polymarket_round21_model import (
    Round21InferencePanel,
    predict_round21_probability_batch,
    validate_round21_development_artifact,
)
from .polymarket_round21_policy import Round21ProbabilityEnvelope
from .polymarket_round21_sealed import Round21SealedEvaluationResult
from .polymarket_round21_tcn import ROUND21_TCN_SEQUENCE_LENGTH


POLYMARKET_ROUND21_PROSPECTIVE_SCHEMA_VERSION = (
    "polymarket-round21-prospective-prediction-v1"
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_LAYERS = frozenset(("core", "core_spot", "core_spot_usdm"))
_RESET_REASONS = frozenset(("initial", "none", "condition_change", "cadence_gap"))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 prospective {name} digest is invalid")
    return selected


def _maximum_receipt_ms(row: Round21CausalFeatureRow) -> int:
    return max(
        row.core_maximum_receipt_ms,
        row.spot_maximum_receipt_ms if row.spot_available else 0,
        row.usdm_maximum_receipt_ms if row.usdm_available else 0,
    )


def build_round21_inference_panel(
    feature_rows: Sequence[Round21CausalFeatureRow],
) -> Round21InferencePanel:
    """Build one target-free, contiguous inference sequence from causal rows."""

    schema = POLYMARKET_ROUND21_FEATURE_SCHEMA.validated()
    unverified = tuple(feature_rows)
    if not 1 <= len(unverified) <= ROUND21_TCN_SEQUENCE_LENGTH:
        raise ValueError("Round 21 prospective feature history is invalid")
    if any(not isinstance(row, Round21CausalFeatureRow) for row in unverified):
        raise TypeError("Round 21 prospective feature row type differs")
    rows = tuple(row.validated(schema) for row in unverified)
    first = rows[0]
    if any(
        row.condition_id != first.condition_id
        or row.event_start_ms != first.event_start_ms
        for row in rows
    ) or any(
        current.decision_time_ms - previous.decision_time_ms
        != POLYMARKET_ROUND21_DECISION_CADENCE_MS
        for previous, current in zip(rows, rows[1:], strict=False)
    ):
        raise ValueError("Round 21 prospective feature history is not contiguous")
    source_dataset_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_PROSPECTIVE_SCHEMA_VERSION,
            "dataset_design_sha256": POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
            "feature_schema_sha256": schema.schema_sha256,
            "row_sha256": [row.row_sha256 for row in rows],
            "target_accessed": False,
            "trading_authority": False,
        }
    )
    return Round21InferencePanel.create(
        condition_ids=np.asarray([row.condition_id for row in rows], dtype=object),
        event_start_ms=np.asarray([row.event_start_ms for row in rows], dtype=np.int64),
        decision_time_ms=np.asarray(
            [row.decision_time_ms for row in rows],
            dtype=np.int64,
        ),
        structural_probability=np.asarray(
            [row.structural_probability for row in rows],
            dtype=np.float64,
        ),
        market_prior_probability=np.asarray(
            [row.market_prior_probability for row in rows],
            dtype=np.float64,
        ),
        core_features=np.asarray([row.core_values for row in rows], dtype=np.float32),
        spot_features=np.asarray([row.spot_values for row in rows], dtype=np.float32),
        usdm_features=np.asarray([row.usdm_values for row in rows], dtype=np.float32),
        spot_available=np.asarray(
            [row.spot_available for row in rows],
            dtype=np.bool_,
        ),
        usdm_available=np.asarray(
            [row.usdm_available for row in rows],
            dtype=np.bool_,
        ),
        core_feature_names_sha256=schema.core_names_sha256,
        spot_feature_names_sha256=schema.spot_names_sha256,
        usdm_feature_names_sha256=schema.usdm_names_sha256,
        source_dataset_sha256=source_dataset_sha256,
        dataset_design_sha256=POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    )


@dataclass(frozen=True, slots=True)
class Round21ProspectivePrediction:
    status: str
    reason: str
    reset_reason: str
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    observed_at_ms: int
    source_maximum_receipt_ms: int
    history_row_count: int
    population_layer: str
    source_causal_row_sha256: str
    source_model_artifact_sha256: str
    sealed_result_sha256: str
    inference_latency_ns: int
    envelope: Round21ProbabilityEnvelope | None
    prediction_sha256: str
    target_accessed: bool = False
    credentials_used: bool = False
    account_connected: bool = False
    binance_execution_connected: bool = False
    grants_execution_authority: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        status: str,
        reason: str,
        reset_reason: str,
        row: Round21CausalFeatureRow,
        observed_at_ms: int,
        history_row_count: int,
        population_layer: str,
        source_model_artifact_sha256: str,
        sealed_result_sha256: str,
        inference_latency_ns: int,
        envelope: Round21ProbabilityEnvelope | None,
    ) -> Round21ProspectivePrediction:
        selected_status = str(status or "").strip()
        selected_reason = str(reason or "").strip()
        selected_reset = str(reset_reason or "").strip()
        selected_row = row.validated(POLYMARKET_ROUND21_FEATURE_SCHEMA)
        observed = int(observed_at_ms)
        history_count = int(history_row_count)
        layer = str(population_layer or "").strip()
        model_sha = _digest(source_model_artifact_sha256, name="model artifact")
        sealed_sha = _digest(sealed_result_sha256, name="sealed result")
        latency = int(inference_latency_ns)
        maximum_receipt = _maximum_receipt_ms(selected_row)
        selected_envelope = None if envelope is None else envelope.validated()
        if (
            selected_status not in {"observed", "abstain"}
            or len(selected_reason) > 160
            or selected_reset not in _RESET_REASONS
            or _CONDITION_ID.fullmatch(selected_row.condition_id) is None
            or observed < selected_row.decision_time_ms
            or not 0 < maximum_receipt <= selected_row.decision_time_ms
            or not 1 <= history_count <= ROUND21_TCN_SEQUENCE_LENGTH
            or layer not in _LAYERS
            or latency < 0
            or (selected_status == "observed")
            != (not selected_reason and selected_envelope is not None)
            or (selected_status == "abstain")
            != (bool(selected_reason) and selected_envelope is None)
            or (
                selected_envelope is not None
                and (
                    selected_envelope.condition_id != selected_row.condition_id
                    or selected_envelope.decision_time_ms
                    != selected_row.decision_time_ms
                    or selected_envelope.model_layer != layer
                    or selected_envelope.source_model_artifact_sha256 != model_sha
                )
            )
        ):
            raise ValueError("Round 21 prospective prediction is invalid")
        provisional = cls(
            status=selected_status,
            reason=selected_reason,
            reset_reason=selected_reset,
            condition_id=selected_row.condition_id,
            event_start_ms=selected_row.event_start_ms,
            decision_time_ms=selected_row.decision_time_ms,
            observed_at_ms=observed,
            source_maximum_receipt_ms=maximum_receipt,
            history_row_count=history_count,
            population_layer=layer,
            source_causal_row_sha256=selected_row.row_sha256,
            source_model_artifact_sha256=model_sha,
            sealed_result_sha256=sealed_sha,
            inference_latency_ns=latency,
            envelope=selected_envelope,
            prediction_sha256=_EMPTY_SHA256,
        )
        return replace(
            provisional,
            prediction_sha256=_canonical_sha256(provisional.identity_payload()),
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_PROSPECTIVE_SCHEMA_VERSION,
            "status": self.status,
            "reason": self.reason,
            "reset_reason": self.reset_reason,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "decision_time_ms": self.decision_time_ms,
            "observed_at_ms": self.observed_at_ms,
            "source_maximum_receipt_ms": self.source_maximum_receipt_ms,
            "history_row_count": self.history_row_count,
            "population_layer": self.population_layer,
            "source_causal_row_sha256": self.source_causal_row_sha256,
            "source_model_artifact_sha256": self.source_model_artifact_sha256,
            "sealed_result_sha256": self.sealed_result_sha256,
            "inference_latency_ns": self.inference_latency_ns,
            "probability_evidence_sha256": (
                None if self.envelope is None else self.envelope.evidence_sha256
            ),
            "target_accessed": False,
            "credentials_used": False,
            "account_connected": False,
            "binance_execution_connected": False,
            "grants_execution_authority": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        probability_evidence = None
        if self.envelope is not None:
            probability_evidence = {
                "probability_up": format(self.envelope.probability_up, "f"),
                "lower_up": format(self.envelope.lower_up, "f"),
                "upper_up": format(self.envelope.upper_up, "f"),
                "source_probability_batch_sha256": (
                    self.envelope.source_probability_batch_sha256
                ),
                "feature_row_sha256": self.envelope.feature_row_sha256,
                "evidence_sha256": self.envelope.evidence_sha256,
            }
        return {
            **self.identity_payload(),
            "probability_evidence": probability_evidence,
            "prediction_sha256": self.prediction_sha256,
        }

    def validated(self) -> Round21ProspectivePrediction:
        false_fields = (
            self.target_accessed,
            self.credentials_used,
            self.account_connected,
            self.binance_execution_connected,
            self.grants_execution_authority,
            self.profitability_claim,
            self.paper_trading_authority,
            self.live_trading_authority,
        )
        if (
            self.status not in {"observed", "abstain"}
            or not isinstance(self.reason, str)
            or len(self.reason) > 160
            or self.reset_reason not in _RESET_REASONS
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or type(self.decision_time_ms) is not int
            or not self.event_start_ms
            <= self.decision_time_ms
            < self.event_start_ms + 300_000
            or (self.decision_time_ms - self.event_start_ms)
            % POLYMARKET_ROUND21_DECISION_CADENCE_MS
            or type(self.observed_at_ms) is not int
            or type(self.source_maximum_receipt_ms) is not int
            or type(self.history_row_count) is not int
            or type(self.inference_latency_ns) is not int
            or self.population_layer not in _LAYERS
            or _SHA256.fullmatch(self.source_causal_row_sha256) is None
            or _SHA256.fullmatch(self.source_model_artifact_sha256) is None
            or _SHA256.fullmatch(self.sealed_result_sha256) is None
            or _SHA256.fullmatch(self.prediction_sha256) is None
            or any(
                digest == _EMPTY_SHA256
                for digest in (
                    self.source_causal_row_sha256,
                    self.source_model_artifact_sha256,
                    self.sealed_result_sha256,
                    self.prediction_sha256,
                )
            )
            or any(type(value) is not bool for value in false_fields)
            or any(false_fields)
            or self.inference_latency_ns < 0
            or self.observed_at_ms < self.decision_time_ms
            or not 0 < self.source_maximum_receipt_ms <= self.decision_time_ms
            or not 1 <= self.history_row_count <= ROUND21_TCN_SEQUENCE_LENGTH
            or (self.status == "observed")
            != (not self.reason and self.envelope is not None)
            or (self.status == "abstain")
            != (bool(self.reason) and self.envelope is None)
            or (
                self.envelope is not None
                and (
                    self.envelope.validated() != self.envelope
                    or self.envelope.condition_id != self.condition_id
                    or self.envelope.decision_time_ms != self.decision_time_ms
                    or self.envelope.model_layer != self.population_layer
                    or self.envelope.source_model_artifact_sha256
                    != self.source_model_artifact_sha256
                )
            )
            or self.prediction_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 prospective prediction differs")
        return self


class Round21ProspectiveScorer:
    """Score one accepted sealed candidate without credentials or order authority."""

    credentials_used = False
    account_connected = False
    binance_execution_connected = False
    grants_execution_authority = False
    trading_authority = False

    def __init__(
        self,
        *,
        artifact: Mapping[str, object],
        sealed_result: Round21SealedEvaluationResult,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if not isinstance(sealed_result, Round21SealedEvaluationResult):
            raise TypeError("Round 21 prospective sealed result type differs")
        selected_artifact = validate_round21_development_artifact(artifact)
        selected_result = sealed_result.validated()
        if not selected_result.candidate_accepted:
            raise ValueError("Round 21 prospective candidate was not accepted")
        model_sha = str(selected_artifact["artifact_sha256"])
        if selected_result.predictive.model_artifact_sha256 != model_sha:
            raise ValueError("Round 21 prospective model and sealed result differ")
        layer = selected_result.selected_population_layer
        dataset_identity = selected_artifact.get("dataset_and_partition")
        schema = POLYMARKET_ROUND21_FEATURE_SCHEMA.validated()
        if not isinstance(dataset_identity, Mapping) or any(
            dataset_identity.get(name) != expected
            for name, expected in (
                ("core_feature_names_sha256", schema.core_names_sha256),
                ("spot_feature_names_sha256", schema.spot_names_sha256),
                ("usdm_feature_names_sha256", schema.usdm_names_sha256),
            )
        ):
            raise ValueError("Round 21 prospective feature schema differs")
        if layer not in _LAYERS:
            raise ValueError("Round 21 prospective population layer differs")
        if not callable(monotonic_ns):
            raise TypeError("Round 21 prospective monotonic clock is invalid")
        self.artifact = selected_artifact
        self.sealed_result = selected_result
        self.population_layer = layer
        self._monotonic_ns = monotonic_ns
        self._history: deque[Round21CausalFeatureRow] = deque(
            maxlen=ROUND21_TCN_SEQUENCE_LENGTH
        )
        self._last_prediction: Round21ProspectivePrediction | None = None
        self._lock = Lock()

    def _layer_available(self, row: Round21CausalFeatureRow) -> bool:
        if self.population_layer == "core":
            return True
        if self.population_layer == "core_spot":
            return row.spot_available
        return row.spot_available and row.usdm_available

    def evaluate(
        self,
        row: Round21CausalFeatureRow,
        *,
        observed_at_ms: int,
    ) -> Round21ProspectivePrediction:
        selected = row.validated(POLYMARKET_ROUND21_FEATURE_SCHEMA)
        observed = int(observed_at_ms)
        if observed < selected.decision_time_ms:
            raise ValueError("Round 21 prospective observation precedes decision")
        with self._lock:
            if self._history:
                previous = self._history[-1]
                if (
                    selected.condition_id == previous.condition_id
                    and selected.decision_time_ms == previous.decision_time_ms
                ):
                    if selected.row_sha256 != previous.row_sha256:
                        raise ValueError("Round 21 prospective duplicate row differs")
                    if self._last_prediction is None:
                        raise RuntimeError("Round 21 prospective cache is unavailable")
                    return self._last_prediction
                if (
                    selected.condition_id == previous.condition_id
                    and selected.decision_time_ms < previous.decision_time_ms
                ):
                    raise ValueError("Round 21 prospective chronology regressed")
                if selected.condition_id != previous.condition_id:
                    reset_reason = "condition_change"
                    self._history.clear()
                elif (
                    selected.decision_time_ms - previous.decision_time_ms
                    != POLYMARKET_ROUND21_DECISION_CADENCE_MS
                ):
                    reset_reason = "cadence_gap"
                    self._history.clear()
                else:
                    reset_reason = "none"
            else:
                reset_reason = "initial"
            self._history.append(selected)
            started = int(self._monotonic_ns())
            envelope: Round21ProbabilityEnvelope | None = None
            reason = ""
            status = "observed"
            if not self._layer_available(selected):
                status = "abstain"
                reason = "selected_optional_feature_layer_unavailable"
            else:
                panel = build_round21_inference_panel(tuple(self._history))
                batch = predict_round21_probability_batch(
                    self.artifact,
                    population_layer=self.population_layer,
                    panel=panel,
                )
                envelope = Round21ProbabilityEnvelope.from_probability_batch(
                    batch=batch,
                    panel=panel,
                    panel_row_index=len(self._history) - 1,
                )
            finished = int(self._monotonic_ns())
            if finished < started:
                raise RuntimeError("Round 21 prospective monotonic clock regressed")
            prediction = Round21ProspectivePrediction.create(
                status=status,
                reason=reason,
                reset_reason=reset_reason,
                row=selected,
                observed_at_ms=observed,
                history_row_count=len(self._history),
                population_layer=self.population_layer,
                source_model_artifact_sha256=str(self.artifact["artifact_sha256"]),
                sealed_result_sha256=self.sealed_result.result_sha256,
                inference_latency_ns=finished - started,
                envelope=envelope,
            ).validated()
            self._last_prediction = prediction
            return prediction


credentials_used = False
account_connected = False
binance_execution_connected = False
grants_execution_authority = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_PROSPECTIVE_SCHEMA_VERSION",
    "Round21ProspectivePrediction",
    "Round21ProspectiveScorer",
    "build_round21_inference_panel",
]
