"""Portable CPU reference for the preregistered Polymarket Round 12 hypothesis.

The reference deliberately uses only scalar IEEE-754 binary64 operations from the
Python standard library. Accelerated implementations are challengers to this code;
they do not define model semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


POLYMARKET_ROUND12_MODEL_SCHEMA_VERSION = (
    "polymarket-round12-fixed-calibration-reference-v1"
)
POLYMARKET_ROUND12_POLICY_SCHEMA_VERSION = (
    "polymarket-round12-primary-confirmation-policy-v1"
)
POLYMARKET_ROUND12_CONTRACT_SCHEMA_VERSION = (
    "polymarket-round12-fixed-calibration-confirmation-contract-v1"
)
POLYMARKET_ROUND12_ACTION_PIPELINE_IMPLEMENTATION_SHA256 = (
    "b517340e03453aecbdc743f0fd9cf07eaf433397bf02c608a299245832c55c87"
)
POLYMARKET_ROUND12_REFERENCE_IMPLEMENTATION_SHA256 = (
    "edfc9a13c3344ac32700747a35e9a0e54c3c711038fb72688b8a7a402dea6c0e"
)
POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256 = (
    "c1d174b3274d374b1008272c70954a67c8a2953b875d11b7268658666d23c7bc"
)
POLYMARKET_ROUND12_PREDECESSOR_CONTRACT_SHA256 = (
    "ced2dfcb058845f3cc430c369b00b0cd493c61b739ff1f0fa3d27b052af1aff4"
)
POLYMARKET_ROUND12_PREDECESSOR_REPORT_SHA256 = (
    "c64bd0356e4e1e333fa80512665a690bc0fac44161429212c7851e3f2b6cca0e"
)
POLYMARKET_ROUND12_SOURCE_DIRECTION_HEAD_SHA256 = (
    "539c0ff909d61762005933919e04643478ad1fcea22fa2d86910614a834b8bd3"
)
POLYMARKET_ROUND12_SOURCE_CALIBRATION_SHA256 = (
    "e8fd51ddded543177ff27ce900c891886b0e1f627505faebfafb548d6574b3a2"
)

_ASSETS = ("BTC", "ETH", "SOL")
_SHA256_LENGTH = 64
_PROBABILITY_CLIP = 1e-9
_RESIDUAL_INTERCEPT = 0.03184091198123928
_CALIBRATION_INTERCEPT = -0.23722844771051635
_CALIBRATION_SLOPE = 1.6669649708940744
_CRITICAL_REFERENCE_FILES = (
    "polymarket_round12_reference.py",
    "polymarket_round12_admission.py",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in text
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _clipped_logit(value: float, *, clip: float) -> float:
    probability = min(max(value, clip), 1.0 - clip)
    return math.log(probability) - math.log1p(-probability)


@dataclass(frozen=True)
class PolymarketRound12ReferenceModel:
    """Frozen calibration-only challenger derived from rejected Round 11 evidence."""

    predecessor_artifact_sha256: str
    predecessor_contract_sha256: str
    predecessor_report_sha256: str
    source_direction_head_sha256: str
    source_calibration_sha256: str
    probability_clip: float
    residual_intercept: float
    calibration_intercept: float
    calibration_slope: float
    external_feature_coefficients_applied: bool
    model_sha256: str

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("model_sha256")
        return {
            "schema_version": POLYMARKET_ROUND12_MODEL_SCHEMA_VERSION,
            **payload,
            "cpu_reference": "python_scalar_binary64",
            "training_or_refit_performed": False,
            "direction_formula": (
                "sigmoid(calibration_intercept + calibration_slope * "
                "(logit(clipped_market_prior_up) + residual_intercept))"
            ),
        }

    def validated(self) -> PolymarketRound12ReferenceModel:
        numeric = (
            self.probability_clip,
            self.residual_intercept,
            self.calibration_intercept,
            self.calibration_slope,
        )
        if (
            self.predecessor_artifact_sha256
            != POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256
            or self.predecessor_contract_sha256
            != POLYMARKET_ROUND12_PREDECESSOR_CONTRACT_SHA256
            or self.predecessor_report_sha256
            != POLYMARKET_ROUND12_PREDECESSOR_REPORT_SHA256
            or self.source_direction_head_sha256
            != POLYMARKET_ROUND12_SOURCE_DIRECTION_HEAD_SHA256
            or self.source_calibration_sha256
            != POLYMARKET_ROUND12_SOURCE_CALIBRATION_SHA256
            or any(not math.isfinite(value) for value in numeric)
            or self.probability_clip != _PROBABILITY_CLIP
            or self.residual_intercept != _RESIDUAL_INTERCEPT
            or self.calibration_intercept != _CALIBRATION_INTERCEPT
            or self.calibration_slope != _CALIBRATION_SLOPE
            or self.calibration_slope <= 0.0
            or self.external_feature_coefficients_applied
            or not _is_sha256(self.model_sha256)
            or self.model_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 12 reference model is invalid")
        return self

    def predict_up(self, market_prior_up: float) -> float:
        """Return the canonical Up probability, independent of runtime backend."""

        self.validated()
        prior = _finite_float(market_prior_up, name="market_prior_up")
        if prior <= 0.0 or prior >= 1.0:
            raise ValueError("market_prior_up must lie strictly inside (0, 1)")
        raw_logit = _clipped_logit(prior, clip=self.probability_clip)
        calibrated_logit = self.calibration_intercept + self.calibration_slope * (
            raw_logit + self.residual_intercept
        )
        probability = _stable_sigmoid(calibrated_logit)
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise ValueError("Polymarket Round 12 probability is invalid")
        return probability

    def predict_pair(self, market_prior_up: float) -> tuple[float, float]:
        probability_up = self.predict_up(market_prior_up)
        return probability_up, 1.0 - probability_up


def _validated_artifact_payload(raw: bytes) -> Mapping[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Round 11 artifact must be UTF-8 JSON") from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Round 11 artifact is invalid JSON") from exc
    artifact = dict(_mapping(decoded, name="Round 11 artifact"))
    claimed_sha256 = artifact.pop("artifact_sha256", None)
    if (
        claimed_sha256 != POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256
        or _canonical_sha256(artifact) != claimed_sha256
        or artifact.get("contract_sha256")
        != POLYMARKET_ROUND12_PREDECESSOR_CONTRACT_SHA256
        or artifact.get("cpu_reference") != "numpy_float64_scipy_fit"
        or artifact.get("onnx_exported") is not False
    ):
        raise ValueError("Round 11 artifact identity or provenance is invalid")

    head = dict(_mapping(artifact.get("direction_head"), name="direction_head"))
    claimed_head_sha256 = head.pop("model_sha256", None)
    coefficients = head.get("coefficients")
    feature_names = artifact.get("direction_feature_names")
    if (
        claimed_head_sha256 != POLYMARKET_ROUND12_SOURCE_DIRECTION_HEAD_SHA256
        or _canonical_sha256(head) != claimed_head_sha256
        or head.get("name") != "direction_residual"
        or head.get("objective") != "weighted_binary_log_loss"
        or _finite_float(head.get("l2"), name="direction_head.l2") != 100.0
        or _finite_float(head.get("intercept"), name="direction_head.intercept")
        != _RESIDUAL_INTERCEPT
        or not isinstance(coefficients, list)
        or not isinstance(feature_names, list)
        or len(coefficients) != len(feature_names)
        or not coefficients
        or any(
            not math.isfinite(
                _finite_float(value, name="direction_head.coefficients[]")
            )
            for value in coefficients
        )
    ):
        raise ValueError("Round 11 direction head is invalid")

    calibration = dict(
        _mapping(artifact.get("direction_calibration"), name="direction_calibration")
    )
    claimed_calibration_sha256 = calibration.pop("calibration_sha256", None)
    if (
        claimed_calibration_sha256 != POLYMARKET_ROUND12_SOURCE_CALIBRATION_SHA256
        or _canonical_sha256(calibration) != claimed_calibration_sha256
        or calibration.get("method") != "platt"
        or _finite_float(
            calibration.get("intercept"), name="direction_calibration.intercept"
        )
        != _CALIBRATION_INTERCEPT
        or _finite_float(calibration.get("slope"), name="direction_calibration.slope")
        != _CALIBRATION_SLOPE
    ):
        raise ValueError("Round 11 direction calibration is invalid")
    return artifact


def load_round12_reference_from_round11_bytes(
    raw: bytes,
) -> PolymarketRound12ReferenceModel:
    """Build the frozen reference only after validating exact predecessor bytes."""

    _validated_artifact_payload(raw)
    provisional = PolymarketRound12ReferenceModel(
        predecessor_artifact_sha256=(POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256),
        predecessor_contract_sha256=(POLYMARKET_ROUND12_PREDECESSOR_CONTRACT_SHA256),
        predecessor_report_sha256=POLYMARKET_ROUND12_PREDECESSOR_REPORT_SHA256,
        source_direction_head_sha256=(POLYMARKET_ROUND12_SOURCE_DIRECTION_HEAD_SHA256),
        source_calibration_sha256=(POLYMARKET_ROUND12_SOURCE_CALIBRATION_SHA256),
        probability_clip=_PROBABILITY_CLIP,
        residual_intercept=_RESIDUAL_INTERCEPT,
        calibration_intercept=_CALIBRATION_INTERCEPT,
        calibration_slope=_CALIBRATION_SLOPE,
        external_feature_coefficients_applied=False,
        model_sha256="",
    )
    return replace(
        provisional,
        model_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def load_round12_reference_from_round11_artifact(
    path: str | Path,
) -> PolymarketRound12ReferenceModel:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("Round 11 artifact cannot be read") from exc
    return load_round12_reference_from_round11_bytes(raw)


def polymarket_round12_reference_implementation_sha256() -> str:
    """Return an OS/interpreter-neutral identity for Round 12 reference code."""

    source_root = Path(__file__).resolve().parent
    digests: dict[str, str] = {}
    for filename in _CRITICAL_REFERENCE_FILES:
        try:
            source = (source_root / filename).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                f"cannot attest Polymarket Round 12 implementation: {filename}"
            ) from exc
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        digests[filename] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return _canonical_sha256(
        {
            "schema_version": "polymarket-round12-reference-implementation-v1",
            "critical_module_source_sha256": digests,
        }
    )


@dataclass(frozen=True)
class PolymarketRound12PrimaryPolicy:
    """One primary, posthoc-motivated policy awaiting one-shot confirmation."""

    profile: str = "conservative"
    minimum_direction_probability: float = 0.80
    minimum_expected_edge_quote: float = 0.02
    minimum_remaining_seconds: float = 120.0
    submission_latency_ms: int = 500
    maximum_execution_observation_delay_ms: int = 500
    retry_interval_ms: int = 1_000
    policy_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("policy_sha256")
        return {
            "schema_version": POLYMARKET_ROUND12_POLICY_SCHEMA_VERSION,
            **payload,
            "selection_origin": (
                "round11_posthoc_probability_screen_requires_fresh_confirmation"
            ),
            "authority": "confirmation_only",
            "live_money_enabled": False,
            "forced_activity": False,
        }

    def validated(self) -> PolymarketRound12PrimaryPolicy:
        if (
            self.profile != "conservative"
            or self.minimum_direction_probability != 0.80
            or self.minimum_expected_edge_quote != 0.02
            or self.minimum_remaining_seconds != 120.0
            or self.submission_latency_ms != 500
            or self.maximum_execution_observation_delay_ms != 500
            or self.retry_interval_ms != 1_000
            or not _is_sha256(self.policy_sha256)
            or self.policy_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket Round 12 primary policy is invalid")
        return self


def polymarket_round12_primary_policy() -> PolymarketRound12PrimaryPolicy:
    provisional = PolymarketRound12PrimaryPolicy()
    return replace(
        provisional,
        policy_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def load_round12_confirmation_contract(
    path: str | Path,
    *,
    predecessor_artifact_path: str | Path | None = None,
    require_current_implementation: bool = False,
) -> Mapping[str, object]:
    """Validate frozen identity and optionally require the current implementation."""

    contract_path = Path(path)
    try:
        raw = contract_path.read_bytes()
    except OSError as exc:
        raise ValueError("Round 12 confirmation contract cannot be read") from exc
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Round 12 confirmation contract is invalid JSON") from exc
    contract = dict(_mapping(decoded, name="Round 12 confirmation contract"))
    claimed_sha256 = contract.pop("contract_sha256", None)
    if (
        not _is_sha256(claimed_sha256)
        or _canonical_sha256(contract) != claimed_sha256
        or contract.get("schema_version") != POLYMARKET_ROUND12_CONTRACT_SCHEMA_VERSION
        or contract.get("round") != 12
        or contract.get("status") != "frozen_before_fresh_capture"
    ):
        raise ValueError("Round 12 confirmation contract identity is invalid")

    predecessor = _mapping(contract.get("predecessor_evidence"), name="predecessor")
    artifact_filename = predecessor.get("artifact_filename")
    if (
        predecessor.get("artifact_sha256")
        != POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256
        or predecessor.get("contract_sha256")
        != POLYMARKET_ROUND12_PREDECESSOR_CONTRACT_SHA256
        or predecessor.get("report_sha256")
        != POLYMARKET_ROUND12_PREDECESSOR_REPORT_SHA256
        or not isinstance(artifact_filename, str)
        or not artifact_filename
        or Path(artifact_filename).name != artifact_filename
    ):
        raise ValueError("Round 12 predecessor contract is invalid")
    artifact_path = (
        Path(predecessor_artifact_path)
        if predecessor_artifact_path is not None
        else contract_path.parent / artifact_filename
    )
    reference = load_round12_reference_from_round11_artifact(artifact_path)
    policy = polymarket_round12_primary_policy()
    model_contract = _mapping(contract.get("model_contract"), name="model_contract")
    policy_contract = _mapping(contract.get("primary_policy"), name="primary_policy")
    if (
        model_contract.get("model_sha256") != reference.model_sha256
        or model_contract.get("external_feature_coefficients_applied") is not False
        or model_contract.get("training_or_refit_performed") is not False
        or policy_contract.get("policy_sha256") != policy.policy_sha256
        or policy_contract.get("minimum_direction_probability")
        != policy.minimum_direction_probability
        or policy_contract.get("minimum_expected_edge_quote")
        != policy.minimum_expected_edge_quote
        or policy_contract.get("minimum_remaining_seconds")
        != policy.minimum_remaining_seconds
        or policy_contract.get("forced_activity") is not False
    ):
        raise ValueError("Round 12 model or policy contract is invalid")

    from .polymarket_action_pipeline import (
        polymarket_action_pipeline_implementation_sha256,
    )

    implementation = _mapping(contract.get("implementation"), name="implementation")
    if (
        implementation.get("reference_implementation_sha256")
        != POLYMARKET_ROUND12_REFERENCE_IMPLEMENTATION_SHA256
        or implementation.get("action_pipeline_implementation_sha256")
        != POLYMARKET_ROUND12_ACTION_PIPELINE_IMPLEMENTATION_SHA256
        or implementation.get("source_hash_normalization") != "utf8_lf_normalized"
    ):
        raise ValueError("Round 12 implementation differs from the frozen contract")
    if require_current_implementation and (
        implementation.get("reference_implementation_sha256")
        != polymarket_round12_reference_implementation_sha256()
        or implementation.get("action_pipeline_implementation_sha256")
        != polymarket_action_pipeline_implementation_sha256()
    ):
        raise ValueError("Round 12 implementation differs from the current code")

    freshness = _mapping(contract.get("freshness"), name="freshness")
    authority = _mapping(contract.get("authority"), name="authority")
    if (
        freshness.get("capture_started") is not False
        or freshness.get("confirmation_consumed") is not False
        or freshness.get("development_rows_reused") is not False
        or authority.get("paper_trading") is not False
        or authority.get("live_trading") is not False
        or authority.get("profitability_claim") is not False
        or authority.get("ai_edge_claim") is not False
    ):
        raise ValueError("Round 12 freshness or authority state is invalid")
    return {**contract, "contract_sha256": claimed_sha256}


@dataclass(frozen=True)
class PolymarketRound12Decision:
    action: str
    outcome: str | None
    probability: float | None
    expected_edge_quote: float | None
    reason: str
    model_sha256: str
    policy_sha256: str

    @property
    def abstained(self) -> bool:
        return self.action == "ABSTAIN"


def decide_round12_primary_action(
    model: PolymarketRound12ReferenceModel,
    *,
    asset: str,
    market_prior_up: float,
    remaining_seconds: float,
    up_quantity: float,
    down_quantity: float,
    up_conservative_entry_cost_quote: float,
    down_conservative_entry_cost_quote: float,
    decision_evidence_admissible: bool,
    lifecycle_clear: bool,
    policy: PolymarketRound12PrimaryPolicy | None = None,
) -> PolymarketRound12Decision:
    """Select at most one outcome; every unresolved condition abstains."""

    reference = model.validated()
    selected_policy = (policy or polymarket_round12_primary_policy()).validated()

    def abstain(reason: str) -> PolymarketRound12Decision:
        return PolymarketRound12Decision(
            action="ABSTAIN",
            outcome=None,
            probability=None,
            expected_edge_quote=None,
            reason=reason,
            model_sha256=reference.model_sha256,
            policy_sha256=selected_policy.policy_sha256,
        )

    if asset not in _ASSETS:
        return abstain("unsupported_asset")
    if decision_evidence_admissible is not True:
        return abstain("decision_evidence_not_admissible")
    if lifecycle_clear is not True:
        return abstain("lifecycle_not_clear")
    try:
        remaining = _finite_float(remaining_seconds, name="remaining_seconds")
        quantities = (
            _finite_float(up_quantity, name="up_quantity"),
            _finite_float(down_quantity, name="down_quantity"),
        )
        costs = (
            _finite_float(
                up_conservative_entry_cost_quote,
                name="up_conservative_entry_cost_quote",
            ),
            _finite_float(
                down_conservative_entry_cost_quote,
                name="down_conservative_entry_cost_quote",
            ),
        )
        probability_up, probability_down = reference.predict_pair(market_prior_up)
    except ValueError:
        return abstain("invalid_numeric_evidence")
    if remaining < selected_policy.minimum_remaining_seconds:
        return abstain("inside_minimum_remaining_window")
    if any(value <= 0.0 for value in quantities) or any(
        value <= 0.0 for value in costs
    ):
        return abstain("invalid_execution_economics")

    candidates: list[tuple[float, str, float]] = []
    for outcome, probability, quantity, cost in zip(
        ("Up", "Down"),
        (probability_up, probability_down),
        quantities,
        costs,
        strict=True,
    ):
        edge = quantity * probability - cost
        if (
            probability >= selected_policy.minimum_direction_probability
            and edge > selected_policy.minimum_expected_edge_quote
        ):
            candidates.append((edge, outcome, probability))
    if not candidates:
        return abstain("no_positive_conservative_edge")
    candidates.sort(reverse=True)
    if len(candidates) > 1 and math.isclose(
        candidates[0][0], candidates[1][0], rel_tol=0.0, abs_tol=1e-12
    ):
        return abstain("contradictory_equal_edge")
    edge, outcome, probability = candidates[0]
    return PolymarketRound12Decision(
        action="BUY_FOK_HOLD_TO_RESOLUTION",
        outcome=outcome,
        probability=probability,
        expected_edge_quote=edge,
        reason="eligible_primary_candidate",
        model_sha256=reference.model_sha256,
        policy_sha256=selected_policy.policy_sha256,
    )


__all__ = [
    "POLYMARKET_ROUND12_ACTION_PIPELINE_IMPLEMENTATION_SHA256",
    "POLYMARKET_ROUND12_REFERENCE_IMPLEMENTATION_SHA256",
    "POLYMARKET_ROUND12_MODEL_SCHEMA_VERSION",
    "POLYMARKET_ROUND12_POLICY_SCHEMA_VERSION",
    "POLYMARKET_ROUND12_CONTRACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND12_PREDECESSOR_ARTIFACT_SHA256",
    "POLYMARKET_ROUND12_PREDECESSOR_CONTRACT_SHA256",
    "POLYMARKET_ROUND12_PREDECESSOR_REPORT_SHA256",
    "PolymarketRound12Decision",
    "PolymarketRound12PrimaryPolicy",
    "PolymarketRound12ReferenceModel",
    "decide_round12_primary_action",
    "load_round12_reference_from_round11_artifact",
    "load_round12_reference_from_round11_bytes",
    "load_round12_confirmation_contract",
    "polymarket_round12_reference_implementation_sha256",
    "polymarket_round12_primary_policy",
]
