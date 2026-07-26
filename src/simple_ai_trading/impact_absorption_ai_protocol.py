"""Constrained local-AI review protocol for Round 74 model decisions.

The language model is deliberately outside the execution and deterministic
risk-control boundaries. It can only preserve, reduce, or veto an already
formed ML candidate at horizons long enough to tolerate local inference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from urllib.parse import urlparse

from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
)


ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION = "round-074-ai-model-manifest-v1"
ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION = "round-074-ai-review-request-v1"
ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION = "round-074-ai-review-decision-v1"
ROUND74_AI_REVIEW_HORIZONS_SECONDS = (30, 300)
ROUND74_AI_REVIEW_MAXIMUM_VALIDITY_NS = 30_000_000_000
ROUND74_AI_REVIEW_MINIMUM_PARAMETER_COUNT = 2_000_000_000
ROUND74_AI_REVIEW_MAXIMUM_PARAMETER_COUNT = 100_000_000_000
ROUND74_AI_REVIEW_VERDICTS = (
    "allow_unchanged",
    "reduce",
    "veto",
    "abstain",
)
ROUND74_AI_REVIEW_REASON_CODES = (
    "none",
    "forecast_uncertainty",
    "regime_unpredictability",
    "adverse_selection",
    "liquidity_thin",
    "spread_wide",
    "flow_instability",
    "stale_state",
    "model_inconsistency",
)
ROUND74_AI_RUNTIME_BACKENDS = (
    "windows-ml",
    "onnxruntime-genai-directml",
    "llama.cpp-vulkan",
    "llama.cpp-hip",
    "llama.cpp-cuda",
)
ROUND74_AI_QUANTIZATION_FORMATS = (
    "int4",
    "q4_k_m",
    "q5_k_m",
    "int8",
    "fp16",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{2,159}")
_LICENSE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{1,63}")
_MAXIMUM_ABSOLUTE_INPUT = 1_000_000_000.0
_AI_PROMPT_FEATURE_NAMES = tuple(
    f"asset_identity_{index - 5}"
    if 5 <= index <= 7
    else name
    for index, name in enumerate(ROUND74_EVENT_FEATURE_NAMES)
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    selected = str(value)
    if _SHA256.fullmatch(selected) is None:
        raise ValueError(f"Round 74 AI {label} digest is invalid")
    return selected


def _strict_json_object(raw_text: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(
                    "Round 74 AI decision has duplicate JSON keys"
                )
            output[key] = value
        return output

    parsed = json.loads(raw_text, object_pairs_hook=reject_duplicates)
    if not isinstance(parsed, dict):
        raise ValueError("Round 74 AI decision root differs")
    return parsed


def _finite_tuple(
    value: Sequence[object],
    *,
    expected_length: int,
    label: str,
    nonnegative: bool = False,
    ordered: bool = False,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != expected_length:
        raise ValueError(f"Round 74 AI {label} dimensions differ")
    selected = tuple(float(item) for item in value)
    if any(
        not math.isfinite(item)
        or abs(item) > _MAXIMUM_ABSOLUTE_INPUT
        or (nonnegative and item < 0.0)
        for item in selected
    ):
        raise ValueError(f"Round 74 AI {label} values differ")
    if ordered and any(
        right < left for left, right in zip(selected, selected[1:])
    ):
        raise ValueError(f"Round 74 AI {label} order differs")
    return selected


def _rounded(value: float, digits: int) -> float:
    selected = round(float(value), digits)
    return 0.0 if selected == 0.0 else selected


@dataclass(frozen=True)
class Round74AIModelManifest:
    """Pinned local model and inference-runtime identity."""

    model_id: str
    model_revision: str
    weights_sha256: str
    parameter_count: int
    quantization: str
    runtime_backend: str
    runtime_version: str
    license_id: str
    model_card_url: str
    minimum_vram_bytes: int
    finance_specialized: bool
    schema_version: str = ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        parsed = urlparse(self.model_card_url)
        if (
            self.schema_version
            != ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION
            or _MODEL_ID.fullmatch(self.model_id) is None
            or _REVISION.fullmatch(self.model_revision) is None
            or self.quantization not in ROUND74_AI_QUANTIZATION_FORMATS
            or self.runtime_backend not in ROUND74_AI_RUNTIME_BACKENDS
            or not self.runtime_version.strip()
            or _LICENSE_ID.fullmatch(self.license_id) is None
            or parsed.scheme != "https"
            or not parsed.netloc
            or isinstance(self.parameter_count, bool)
            or not (
                ROUND74_AI_REVIEW_MINIMUM_PARAMETER_COUNT
                <= self.parameter_count
                <= ROUND74_AI_REVIEW_MAXIMUM_PARAMETER_COUNT
            )
            or isinstance(self.minimum_vram_bytes, bool)
            or self.minimum_vram_bytes <= 0
            or not isinstance(self.finance_specialized, bool)
        ):
            raise ValueError("Round 74 AI model manifest differs")
        _require_sha256(self.weights_sha256, "weights")

    @property
    def manifest_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "weights_sha256": self.weights_sha256,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
            "runtime_backend": self.runtime_backend,
            "runtime_version": self.runtime_version,
            "license_id": self.license_id,
            "model_card_url": self.model_card_url,
            "minimum_vram_bytes": self.minimum_vram_bytes,
            "finance_specialized": self.finance_specialized,
            "remote_inference_permitted": False,
            "execution_authority": False,
            "model_size_implies_edge": False,
        }
        if include_sha256:
            payload["manifest_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74AIModelManifest:
        payload = dict(value)
        claimed = str(payload.pop("manifest_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 AI model manifest digest differs")
        try:
            selected = cls(
                model_id=str(payload["model_id"]),
                model_revision=str(payload["model_revision"]),
                weights_sha256=str(payload["weights_sha256"]),
                parameter_count=int(payload["parameter_count"]),
                quantization=str(payload["quantization"]),
                runtime_backend=str(payload["runtime_backend"]),
                runtime_version=str(payload["runtime_version"]),
                license_id=str(payload["license_id"]),
                model_card_url=str(payload["model_card_url"]),
                minimum_vram_bytes=int(payload["minimum_vram_bytes"]),
                finance_specialized=payload["finance_specialized"],
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 AI model manifest payload differs"
            ) from exc
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 AI model manifest policy differs")
        selected.validate()
        return selected


@dataclass(frozen=True)
class Round74AIReviewRequest:
    """An anonymized, causal request for veto-only local-AI review."""

    pretest_policy_sha256: str
    sample_sha256: str
    deterministic_risk_state_sha256: str
    asset_slot: int
    side: str
    horizon_seconds: int
    requested_wall_ns: int
    expires_wall_ns: int
    proposed_risk_size_bps: int
    feature_last: tuple[float, ...]
    feature_mean: tuple[float, ...]
    feature_standard_deviation: tuple[float, ...]
    payoff_quantiles_bps: tuple[float, ...]
    maximum_adverse_excursion_quantiles_bps: tuple[float, ...]
    positive_payoff_probability: float
    adverse_selection_probability: float
    regime_unpredictability_probability: float
    schema_version: str = ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION

    def validate(self) -> None:
        feature_count = len(ROUND74_EVENT_FEATURE_NAMES)
        feature_last = _finite_tuple(
            self.feature_last,
            expected_length=feature_count,
            label="last features",
        )
        feature_mean = _finite_tuple(
            self.feature_mean,
            expected_length=feature_count,
            label="mean features",
        )
        feature_std = _finite_tuple(
            self.feature_standard_deviation,
            expected_length=feature_count,
            label="feature standard deviations",
            nonnegative=True,
        )
        payoff = _finite_tuple(
            self.payoff_quantiles_bps,
            expected_length=len(ROUND74_EVENT_PAYOFF_QUANTILES),
            label="payoff quantiles",
            ordered=True,
        )
        adverse = _finite_tuple(
            self.maximum_adverse_excursion_quantiles_bps,
            expected_length=len(ROUND74_EVENT_PAYOFF_QUANTILES),
            label="adverse-excursion quantiles",
            nonnegative=True,
            ordered=True,
        )
        probabilities = (
            self.positive_payoff_probability,
            self.adverse_selection_probability,
            self.regime_unpredictability_probability,
        )
        if (
            self.schema_version != ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION
            or isinstance(self.asset_slot, bool)
            or not isinstance(self.asset_slot, int)
            or not 0 <= self.asset_slot < 3
            or self.side not in ROUND74_EVENT_PAYOFF_SIDES
            or self.horizon_seconds not in ROUND74_AI_REVIEW_HORIZONS_SECONDS
            or isinstance(self.requested_wall_ns, bool)
            or not isinstance(self.requested_wall_ns, int)
            or isinstance(self.expires_wall_ns, bool)
            or not isinstance(self.expires_wall_ns, int)
            or not (
                0
                < self.requested_wall_ns
                < self.expires_wall_ns
                <= self.requested_wall_ns
                + ROUND74_AI_REVIEW_MAXIMUM_VALIDITY_NS
            )
            or isinstance(self.proposed_risk_size_bps, bool)
            or not isinstance(self.proposed_risk_size_bps, int)
            or not 1 <= self.proposed_risk_size_bps <= 10_000
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in probabilities
            )
            or feature_last != self.feature_last
            or feature_mean != self.feature_mean
            or feature_std != self.feature_standard_deviation
            or payoff != self.payoff_quantiles_bps
            or adverse
            != self.maximum_adverse_excursion_quantiles_bps
        ):
            raise ValueError("Round 74 AI review request differs")
        _require_sha256(self.pretest_policy_sha256, "pretest policy")
        _require_sha256(self.sample_sha256, "sample")
        _require_sha256(
            self.deterministic_risk_state_sha256,
            "deterministic risk state",
        )

    @property
    def request_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "pretest_policy_sha256": self.pretest_policy_sha256,
            "sample_sha256": self.sample_sha256,
            "deterministic_risk_state_sha256": (
                self.deterministic_risk_state_sha256
            ),
            "asset_slot": self.asset_slot,
            "side": self.side,
            "horizon_seconds": self.horizon_seconds,
            "requested_wall_ns": self.requested_wall_ns,
            "expires_wall_ns": self.expires_wall_ns,
            "proposed_risk_size_bps": self.proposed_risk_size_bps,
            "feature_last": list(self.feature_last),
            "feature_mean": list(self.feature_mean),
            "feature_standard_deviation": list(
                self.feature_standard_deviation
            ),
            "payoff_quantiles_bps": list(self.payoff_quantiles_bps),
            "maximum_adverse_excursion_quantiles_bps": list(
                self.maximum_adverse_excursion_quantiles_bps
            ),
            "positive_payoff_probability": (
                self.positive_payoff_probability
            ),
            "adverse_selection_probability": (
                self.adverse_selection_probability
            ),
            "regime_unpredictability_probability": (
                self.regime_unpredictability_probability
            ),
            "absolute_date_exposed_to_ai": False,
            "real_symbol_exposed_to_ai": False,
            "future_outcome_exposed_to_ai": False,
        }
        if include_sha256:
            payload["request_sha256"] = _canonical_sha256(payload)
        return payload

    def prompt_payload(self) -> dict[str, object]:
        """Return only the anonymized numeric projection visible to the LLM."""

        self.validate()
        feature_summary = {
            name: [
                _rounded(self.feature_last[index], 8),
                _rounded(self.feature_mean[index], 8),
                _rounded(self.feature_standard_deviation[index], 8),
            ]
            for index, name in enumerate(_AI_PROMPT_FEATURE_NAMES)
        }
        payload = {
            "schema_version": "round-074-ai-prompt-payload-v1",
            "asset": f"asset_{self.asset_slot}",
            "side": self.side,
            "horizon_seconds": self.horizon_seconds,
            "proposed_risk_size_bps": self.proposed_risk_size_bps,
            "feature_contract_sha256": (
                ROUND74_EVENT_FEATURE_NAMES_SHA256
            ),
            "anonymized_feature_names_sha256": _canonical_sha256(
                list(_AI_PROMPT_FEATURE_NAMES)
            ),
            "standardized_feature_summary": feature_summary,
            "summary_value_order": ["last", "mean", "standard_deviation"],
            "payoff_quantile_levels": list(
                ROUND74_EVENT_PAYOFF_QUANTILES
            ),
            "payoff_quantiles_bps": [
                _rounded(value, 6) for value in self.payoff_quantiles_bps
            ],
            "maximum_adverse_excursion_quantiles_bps": [
                _rounded(value, 6)
                for value in self.maximum_adverse_excursion_quantiles_bps
            ],
            "positive_payoff_probability": _rounded(
                self.positive_payoff_probability,
                8,
            ),
            "adverse_selection_probability": _rounded(
                self.adverse_selection_probability,
                8,
            ),
            "regime_unpredictability_probability": _rounded(
                self.regime_unpredictability_probability,
                8,
            ),
        }
        payload["prompt_payload_sha256"] = _canonical_sha256(payload)
        return payload


@dataclass(frozen=True)
class Round74AIReviewDecision:
    """Strict model output with no direction, price, leverage, or order fields."""

    verdict: str
    size_multiplier_bps: int
    confidence_bps: int
    reason_codes: tuple[str, ...]
    schema_version: str = ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION

    def validate(self) -> None:
        if (
            self.schema_version != ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION
            or self.verdict not in ROUND74_AI_REVIEW_VERDICTS
            or isinstance(self.size_multiplier_bps, bool)
            or not isinstance(self.size_multiplier_bps, int)
            or not 0 <= self.size_multiplier_bps <= 10_000
            or isinstance(self.confidence_bps, bool)
            or not isinstance(self.confidence_bps, int)
            or not 0 <= self.confidence_bps <= 10_000
            or not self.reason_codes
            or tuple(sorted(set(self.reason_codes))) != self.reason_codes
            or any(
                code not in ROUND74_AI_REVIEW_REASON_CODES
                for code in self.reason_codes
            )
        ):
            raise ValueError("Round 74 AI review decision differs")
        if self.verdict == "allow_unchanged":
            valid = (
                self.size_multiplier_bps == 10_000
                and self.reason_codes == ("none",)
            )
        elif self.verdict == "reduce":
            valid = (
                0 < self.size_multiplier_bps < 10_000
                and "none" not in self.reason_codes
            )
        else:
            valid = (
                self.size_multiplier_bps == 0
                and "none" not in self.reason_codes
            )
        if not valid:
            raise ValueError("Round 74 AI verdict semantics differ")

    @property
    def decision_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "size_multiplier_bps": self.size_multiplier_bps,
            "confidence_bps": self.confidence_bps,
            "reason_codes": list(self.reason_codes),
            "may_increase_risk": False,
            "may_select_side": False,
            "may_set_leverage": False,
            "may_submit_or_cancel_orders": False,
        }
        if include_sha256:
            payload["decision_sha256"] = _canonical_sha256(payload)
        return payload

    @classmethod
    def from_generated_text(cls, raw_text: str) -> Round74AIReviewDecision:
        payload = _strict_json_object(raw_text)
        if set(payload) != {
            "schema_version",
            "verdict",
            "size_multiplier_bps",
            "confidence_bps",
            "reason_codes",
        }:
            raise ValueError("Round 74 AI decision fields differ")
        reasons = payload["reason_codes"]
        if not isinstance(reasons, list) or any(
            not isinstance(value, str) for value in reasons
        ):
            raise ValueError("Round 74 AI decision reason codes differ")
        selected = cls(
            verdict=str(payload["verdict"]),
            size_multiplier_bps=payload["size_multiplier_bps"],
            confidence_bps=payload["confidence_bps"],
            reason_codes=tuple(reasons),
            schema_version=str(payload["schema_version"]),
        )
        selected.validate()
        return selected


def build_round74_ai_review_prompt(
    request: Round74AIReviewRequest,
) -> tuple[str, str]:
    """Build a fixed instruction and canonical anonymized numeric payload."""

    request.validate()
    system = (
        "You are a conservative local market-risk reviewer. Assess only the "
        "causal numeric packet. You may preserve, reduce, veto, or abstain. "
        "Never infer an identity or date, choose a side, increase size, set "
        "leverage, or propose an order. Return exactly one JSON object with "
        "keys schema_version, verdict, size_multiplier_bps, confidence_bps, "
        "reason_codes. schema_version must be "
        f"{ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION}. verdict must be one of "
        "allow_unchanged, reduce, veto, abstain. allow_unchanged requires "
        "size_multiplier_bps=10000 and reason_codes=[\"none\"]. reduce "
        "requires 1..9999 and one or more sorted reason codes other than "
        "none. veto or abstain requires 0. Valid reason codes are: "
        + ", ".join(ROUND74_AI_REVIEW_REASON_CODES)
        + ". Do not return markdown or free text."
    )
    return system, _canonical_json(request.prompt_payload())


def apply_round74_ai_risk_modifier(
    request: Round74AIReviewRequest,
    decision: Round74AIReviewDecision | None,
    *,
    deterministic_risk_gate_passed: bool,
    observed_wall_ns: int,
) -> int:
    """Apply only a risk reduction; any uncertainty fails closed for entries."""

    request.validate()
    if (
        not isinstance(deterministic_risk_gate_passed, bool)
        or isinstance(observed_wall_ns, bool)
        or not isinstance(observed_wall_ns, int)
        or observed_wall_ns < request.requested_wall_ns
    ):
        raise ValueError("Round 74 AI application context differs")
    if (
        not deterministic_risk_gate_passed
        or observed_wall_ns > request.expires_wall_ns
        or decision is None
    ):
        return 0
    decision.validate()
    return (
        request.proposed_risk_size_bps
        * decision.size_multiplier_bps
        // 10_000
    )


__all__ = [
    "ROUND74_AI_MODEL_MANIFEST_SCHEMA_VERSION",
    "ROUND74_AI_QUANTIZATION_FORMATS",
    "ROUND74_AI_REVIEW_DECISION_SCHEMA_VERSION",
    "ROUND74_AI_REVIEW_HORIZONS_SECONDS",
    "ROUND74_AI_REVIEW_MAXIMUM_PARAMETER_COUNT",
    "ROUND74_AI_REVIEW_MAXIMUM_VALIDITY_NS",
    "ROUND74_AI_REVIEW_MINIMUM_PARAMETER_COUNT",
    "ROUND74_AI_REVIEW_REASON_CODES",
    "ROUND74_AI_REVIEW_REQUEST_SCHEMA_VERSION",
    "ROUND74_AI_REVIEW_VERDICTS",
    "ROUND74_AI_RUNTIME_BACKENDS",
    "Round74AIModelManifest",
    "Round74AIReviewDecision",
    "Round74AIReviewRequest",
    "apply_round74_ai_risk_modifier",
    "build_round74_ai_review_prompt",
]
