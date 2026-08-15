"""Live causal feature parity for the BTC Polymarket Round 16 shadow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .polymarket_historical_shadow import (
    PolymarketBtcFlowBuffer,
    PolymarketShadowDataUnavailable,
)
from .polymarket_round16 import (
    ROUND16_DURATION_MS,
    load_round16_historical_contract,
)
from .polymarket_round16_dataset import (
    ROUND16_FEATURE_NAMES,
    build_round16_feature_vector,
)
from .polymarket_round16_evaluation import (
    ROUND16_EVALUATION_SCHEMA_VERSION,
)
from .polymarket_round16_model import (
    ROUND16_PRETEST_SCHEMA_VERSION,
    predict_round16_candidate,
    round16_feature_support_admission,
    round16_settlement_admission_mask,
)
from .polymarket_round16_targets import round16_target_implementation_manifest


ROUND16_LIVE_LOOKBACK_SECONDS = 901
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
_REQUIRED_EVALUATION_GATES = frozenset(
    {
        "minimum_terminal_conditions",
        "complete_utc_test_days",
        "minimum_outcomes_per_class",
        "minimum_decision_rows",
        "challenger_log_loss_skill_positive",
        "challenger_brier_skill_positive",
        "challenger_balanced_accuracy_not_lower",
        "paired_log_loss_improvement_lower_positive",
        "calibration_slope_in_range",
        "expected_calibration_error_at_most_contract_maximum",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 16 shadow JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 16 shadow JSON contains {value}")


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


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _load_artifact(
    path: str | Path,
    *,
    name: str,
    expected_sha256: str,
) -> Mapping[str, object]:
    selected = Path(path)
    if selected.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    raw = selected.read_bytes()
    if not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"{name} size is invalid")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    envelope_sha = _canonical_sha256(value)
    if envelope_sha != _sha(expected_sha256, name=f"expected {name}"):
        raise ValueError(f"{name} pinned digest differs")
    payload = dict(value)
    claimed = _sha(payload.pop("artifact_sha256", ""), name=name)
    if _canonical_sha256(payload) != claimed:
        raise ValueError(f"{name} integrity failed")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class PolymarketRound16LiveFeatureBuilder:
    """Build Round 16 features from public feeds without execution authority."""

    trading_authority = False

    def __init__(self, flow: PolymarketBtcFlowBuffer) -> None:
        if not isinstance(flow, PolymarketBtcFlowBuffer):
            raise TypeError("flow must be PolymarketBtcFlowBuffer")
        if flow.retention_seconds < ROUND16_LIVE_LOOKBACK_SECONDS:
            raise ValueError("Round 16 live flow retention is insufficient")
        self.flow = flow

    def feature_vector(
        self,
        *,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
    ) -> np.ndarray:
        snapshot = self.flow.causal_flow_snapshot(
            decision_time_ms=int(decision_time_ms),
            observed_at_ms=int(observed_at_ms),
            second_count=ROUND16_LIVE_LOOKBACK_SECONDS,
        )
        second_ms = np.asarray(snapshot["second_ms"], dtype=np.int64)
        if (
            second_ms.shape != (ROUND16_LIVE_LOOKBACK_SECONDS,)
            or np.any(np.diff(second_ms) != 1_000)
        ):
            raise RuntimeError("Round 16 live flow chronology differs")
        return build_round16_feature_vector(
            event_start_ms=int(event_start_ms),
            event_end_ms=int(event_start_ms) + ROUND16_DURATION_MS,
            flow_start_ms=int(second_ms[0]),
            decision_time_ms=int(decision_time_ms),
            flow=snapshot,
        )


@dataclass(frozen=True, slots=True)
class VerifiedRound16ShadowPredictor:
    candidate: Mapping[str, object]
    candidate_id: str
    pretest_envelope_sha256: str
    evaluation_envelope_sha256: str
    pretest_file_sha256: str
    evaluation_file_sha256: str
    dataset_sha256: str
    feature_support: Mapping[str, object]
    settlement_controls: Mapping[str, object]
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if self.trading_authority:
            raise ValueError("Round 16 shadow predictor cannot trade")
        _sha(self.pretest_envelope_sha256, name="Round 16 pretest envelope")
        _sha(
            self.evaluation_envelope_sha256,
            name="Round 16 evaluation envelope",
        )
        _sha(self.pretest_file_sha256, name="Round 16 pretest file")
        _sha(self.evaluation_file_sha256, name="Round 16 evaluation file")
        _sha(self.dataset_sha256, name="Round 16 dataset")

    def predict_up_probability(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (len(ROUND16_FEATURE_NAMES),) or np.any(
            ~np.isfinite(vector)
        ):
            raise ValueError("Round 16 shadow feature vector is invalid")
        probability = float(
            predict_round16_candidate(
                self.candidate,
                vector.reshape(1, -1),
            )[0]
        )
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise RuntimeError("Round 16 shadow prediction is invalid")
        return probability


@dataclass(frozen=True, slots=True)
class PolymarketRound16ShadowDecision:
    status: str
    reason: str
    event_start_ms: int
    decision_time_ms: int
    observed_at_ms: int
    probability_up: float | None
    candidate_id: str
    pretest_envelope_sha256: str
    evaluation_envelope_sha256: str
    input_sha256: str = ""
    outside_training_range_count: int = 0
    extreme_outlier_count: int = 0
    trading_authority: bool = False
    grants_execution_authority: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"observed", "abstain"}:
            raise ValueError("Round 16 shadow status is invalid")
        if self.status == "observed":
            if (
                self.reason
                or self.probability_up is None
                or not 0.0 < self.probability_up < 1.0
                or len(self.input_sha256) != 64
            ):
                raise ValueError("Round 16 observed decision is invalid")
        elif self.probability_up is not None or not self.reason:
            raise ValueError("Round 16 abstention is invalid")
        if self.input_sha256:
            _sha(self.input_sha256, name="Round 16 shadow input")
        if self.trading_authority or self.grants_execution_authority:
            raise ValueError("Round 16 shadow decision cannot grant authority")
        if (
            self.outside_training_range_count < 0
            or self.extreme_outlier_count < 0
        ):
            raise ValueError("Round 16 support evidence is invalid")


class PolymarketRound16ShadowScorer:
    """Score verified BTC 15-minute telemetry without execution authority."""

    def __init__(
        self,
        *,
        predictor: VerifiedRound16ShadowPredictor,
        feature_builder: PolymarketRound16LiveFeatureBuilder,
    ) -> None:
        if not isinstance(predictor, VerifiedRound16ShadowPredictor):
            raise TypeError("predictor must be a verified Round 16 predictor")
        if not isinstance(feature_builder, PolymarketRound16LiveFeatureBuilder):
            raise TypeError("feature_builder must be a Round 16 live builder")
        self.predictor = predictor
        self.feature_builder = feature_builder

    def _decision(
        self,
        *,
        status: str,
        reason: str,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
        probability_up: float | None = None,
        input_sha256: str = "",
        outside: int = 0,
        extreme: int = 0,
    ) -> PolymarketRound16ShadowDecision:
        return PolymarketRound16ShadowDecision(
            status=status,
            reason=reason,
            event_start_ms=int(event_start_ms),
            decision_time_ms=int(decision_time_ms),
            observed_at_ms=int(observed_at_ms),
            probability_up=probability_up,
            candidate_id=self.predictor.candidate_id,
            pretest_envelope_sha256=(
                self.predictor.pretest_envelope_sha256
            ),
            evaluation_envelope_sha256=(
                self.predictor.evaluation_envelope_sha256
            ),
            input_sha256=input_sha256,
            outside_training_range_count=outside,
            extreme_outlier_count=extreme,
        )

    def evaluate(
        self,
        *,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
    ) -> PolymarketRound16ShadowDecision:
        try:
            features = self.feature_builder.feature_vector(
                event_start_ms=event_start_ms,
                decision_time_ms=decision_time_ms,
                observed_at_ms=observed_at_ms,
            )
        except PolymarketShadowDataUnavailable as exc:
            return self._decision(
                status="abstain",
                reason=exc.code,
                event_start_ms=event_start_ms,
                decision_time_ms=decision_time_ms,
                observed_at_ms=observed_at_ms,
            )
        input_body = {
            "schema_version": "polymarket-round16-live-input-v1",
            "event_start_ms": int(event_start_ms),
            "decision_time_ms": int(decision_time_ms),
            "feature_names_sha256": _canonical_sha256(
                ROUND16_FEATURE_NAMES
            ),
            "feature_values_little_endian_float32_sha256": hashlib.sha256(
                np.asarray(features, dtype="<f4").tobytes(order="C")
            ).hexdigest(),
            "pretest_envelope_sha256": (
                self.predictor.pretest_envelope_sha256
            ),
            "evaluation_envelope_sha256": (
                self.predictor.evaluation_envelope_sha256
            ),
        }
        input_sha256 = _canonical_sha256(input_body)
        support, outside, extreme = round16_feature_support_admission(
            features.reshape(1, -1),
            self.predictor.feature_support,
        )
        if not bool(support[0]):
            return self._decision(
                status="abstain",
                reason="feature_support_out_of_distribution",
                event_start_ms=event_start_ms,
                decision_time_ms=decision_time_ms,
                observed_at_ms=observed_at_ms,
                input_sha256=input_sha256,
                outside=int(outside[0]),
                extreme=int(extreme[0]),
            )
        settlement = round16_settlement_admission_mask(
            features.reshape(1, -1),
            self.predictor.settlement_controls,
        )
        if not bool(settlement[0]):
            return self._decision(
                status="abstain",
                reason="settlement_manipulation_anomaly",
                event_start_ms=event_start_ms,
                decision_time_ms=decision_time_ms,
                observed_at_ms=observed_at_ms,
                input_sha256=input_sha256,
                outside=int(outside[0]),
                extreme=int(extreme[0]),
            )
        return self._decision(
            status="observed",
            reason="",
            event_start_ms=event_start_ms,
            decision_time_ms=decision_time_ms,
            observed_at_ms=observed_at_ms,
            probability_up=self.predictor.predict_up_probability(features),
            input_sha256=input_sha256,
            outside=int(outside[0]),
            extreme=int(extreme[0]),
        )


def load_verified_round16_shadow_predictor(
    *,
    contract_path: str | Path,
    pretest_path: str | Path,
    evaluation_path: str | Path,
    expected_pretest_envelope_sha256: str,
    expected_evaluation_envelope_sha256: str,
    expected_contract_file_sha256: str | None = None,
) -> VerifiedRound16ShadowPredictor:
    """Load only caller-pinned evidence that passed every predictive gate."""

    contract = load_round16_historical_contract(
        contract_path,
        expected_file_sha256=expected_contract_file_sha256,
    )
    pretest = _load_artifact(
        pretest_path,
        name="Round 16 pretest",
        expected_sha256=expected_pretest_envelope_sha256,
    )
    evaluation = _load_artifact(
        evaluation_path,
        name="Round 16 evaluation",
        expected_sha256=expected_evaluation_envelope_sha256,
    )
    pretest_envelope = _canonical_sha256(pretest)
    evaluation_envelope = _canonical_sha256(evaluation)
    dataset_sha = str(pretest.get("dataset_sha256") or "")
    if (
        pretest.get("schema_version") != ROUND16_PRETEST_SCHEMA_VERSION
        or evaluation.get("schema_version")
        != ROUND16_EVALUATION_SCHEMA_VERSION
        or pretest.get("contract_sha256") != contract.contract_sha256
        or evaluation.get("contract_sha256") != contract.contract_sha256
        or evaluation.get("dataset_sha256") != dataset_sha
        or evaluation.get("pretest_artifact_sha256") != pretest_envelope
        or pretest.get("test_targets_accessed") is not False
        or pretest.get("paper_authority") is not False
        or pretest.get("live_authority") is not False
        or pretest.get("profitability_claim") is not False
        or evaluation.get("paper_authority") is not False
        or evaluation.get("live_authority") is not False
        or evaluation.get("profitability_claim") is not False
    ):
        raise ValueError("Round 16 shadow evidence identity differs")
    _sha(dataset_sha, name="Round 16 dataset")
    commit = str(pretest.get("source_commit") or "").strip().lower()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("Round 16 shadow source commit is invalid")
    if (
        pretest.get("feature_names") != list(ROUND16_FEATURE_NAMES)
        or pretest.get("feature_names_sha256")
        != _canonical_sha256(ROUND16_FEATURE_NAMES)
    ):
        raise ValueError("Round 16 shadow feature contract differs")
    source_root = Path(__file__).parent
    expected_implementation = {
        "round16_model": _file_sha256(
            source_root / "polymarket_round16_model.py"
        ),
        "round16_dataset": _file_sha256(
            source_root / "polymarket_round16_dataset.py"
        ),
        "round16_identity": _file_sha256(
            source_root / "polymarket_round16.py"
        ),
        "round16_evaluation": _file_sha256(
            source_root / "polymarket_round16_evaluation.py"
        ),
        "lightgbm_backend": _file_sha256(
            source_root / "lightgbm_backend.py"
        ),
        "shared_model_primitives": _file_sha256(
            source_root / "polymarket_historical_model.py"
        ),
        "round16_target_manifest": str(
            round16_target_implementation_manifest()["manifest_sha256"]
        ),
    }
    if pretest.get("implementation_sha256") != expected_implementation:
        raise ValueError("Round 16 shadow implementation differs")
    gates = evaluation.get("gates")
    scope = evaluation.get("scope")
    challenger_id = str(pretest.get("selected_best_challenger") or "")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != _REQUIRED_EVALUATION_GATES
        or any(value is not True for value in gates.values())
        or evaluation.get("accepted_predictive_edge") is not True
        or evaluation.get("best_challenger_id") != challenger_id
        or not isinstance(scope, Mapping)
        or scope.get("venue") != "polymarket"
        or scope.get("asset") != "BTC"
        or scope.get("market_variant") != "fifteenminute"
        or scope.get("predictive_screen_only") is not True
        or scope.get("execution_or_profitability_claim") is not False
    ):
        raise ValueError("Round 16 predictive gates are not satisfied")
    candidates = pretest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("Round 16 candidates are malformed")
    candidate_ids: list[str] = []
    for value in candidates:
        if not isinstance(value, Mapping):
            raise ValueError("Round 16 candidate is malformed")
        body = dict(value)
        claimed = _sha(
            body.pop("artifact_sha256", ""),
            name="Round 16 candidate",
        )
        candidate_id = str(value.get("candidate_id") or "")
        if (
            _canonical_sha256(body) != claimed
            or not candidate_id
            or value.get("dataset_sha256") != dataset_sha
            or value.get("feature_names_sha256")
            != _canonical_sha256(ROUND16_FEATURE_NAMES)
        ):
            raise ValueError("Round 16 candidate differs")
        candidate_ids.append(candidate_id)
    if len(set(candidate_ids)) != 5:
        raise ValueError("Round 16 candidate identities differ")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == challenger_id
    ]
    if len(matches) != 1:
        raise ValueError("Round 16 challenger identity differs")
    candidate = dict(matches[0])
    if (
        candidate.get("kind") != "challenger"
        or candidate.get("family")
        not in {"binance_ridge_logistic", "binance_shallow_lightgbm"}
    ):
        raise ValueError("Round 16 challenger differs")
    model = candidate.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Round 16 challenger model is malformed")
    if model.get("type") == "lightgbm":
        model_string = str(model.get("model_string") or "")
        if (
            hashlib.sha256(model_string.encode("utf-8")).hexdigest()
            != model.get("model_sha256")
        ):
            raise ValueError("Round 16 challenger model digest differs")
    elif model.get("type") != "ridge_logistic":
        raise ValueError("Round 16 challenger model type differs")
    feature_support = pretest.get("feature_support")
    settlement_controls = pretest.get("settlement_manipulation_controls")
    if not isinstance(feature_support, Mapping) or not isinstance(
        settlement_controls,
        Mapping,
    ):
        raise ValueError("Round 16 prospective controls are missing")
    probe = np.zeros((1, len(ROUND16_FEATURE_NAMES)), dtype=np.float32)
    round16_feature_support_admission(probe, feature_support)
    round16_settlement_admission_mask(probe, settlement_controls)
    return VerifiedRound16ShadowPredictor(
        candidate=candidate,
        candidate_id=challenger_id,
        pretest_envelope_sha256=pretest_envelope,
        evaluation_envelope_sha256=evaluation_envelope,
        pretest_file_sha256=_file_sha256(Path(pretest_path)),
        evaluation_file_sha256=_file_sha256(Path(evaluation_path)),
        dataset_sha256=dataset_sha,
        feature_support=dict(feature_support),
        settlement_controls=dict(settlement_controls),
    )


__all__ = [
    "ROUND16_LIVE_LOOKBACK_SECONDS",
    "PolymarketRound16LiveFeatureBuilder",
    "PolymarketRound16ShadowDecision",
    "PolymarketRound16ShadowScorer",
    "VerifiedRound16ShadowPredictor",
    "load_verified_round16_shadow_predictor",
]
