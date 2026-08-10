"""Slow target-free Fin-R1 regime supervision for Round 25 research."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
from queue import Empty, Full, Queue
import re
import threading
import time

from .ai_runtime import OllamaResidencyReport, inspect_ollama_model_residency
from .polymarket_round25_ai import (
    POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO,
    Round25AIAdvisory,
    Round25AICandidateSpec,
    _canonical_json,
    _canonical_sha256,
    _get_json,
    _normalized_model,
    _post_json,
    _provider_usage,
    _strict_json_value,
)


POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256 = (
    "83aaed6dd3a37a5f3930384d7697131a3e656ee2c2a778c75b299bfdd669dda7"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256 = (
    "967159d9946e2411827d602cd027f2c972fd2a6ba9eda05aff930c8d04064b73"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_PACKET_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-regime-packet-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_ADVISORY_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-regime-advisory-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_TELEMETRY_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-regime-telemetry-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-fin-r1-regime-result-v1"
)
POLYMARKET_ROUND25_AI_COMBINED_DECISION_SCHEMA_VERSION = (
    "polymarket-round25-hierarchical-ai-risk-decision-v1"
)
POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE = Round25AICandidateSpec(
    candidate_id="fin-r1-8b-regime-supervisor-v1",
    model="fin-r1:8b",
    digest="7a02f6045046a36f53f1541e6fe0ceaff202c2ca48a47c1292fc82e055a4a377",
    parameter_size="7.62B",
    quantization="Q6_K",
    context_length=32_768,
    upstream_revision="026768c4a015b591b54b240743edeac1de0970fa",
)
POLYMARKET_ROUND25_AI_SUPERVISOR_LOOKBACK_MS = 60_000
POLYMARKET_ROUND25_AI_SUPERVISOR_REFRESH_MS = 60_000
POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PACKET_AGE_MS = 5_000
POLYMARKET_ROUND25_AI_SUPERVISOR_VALIDITY_MS = 90_000
POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PROVIDER_SECONDS = 30.0
POLYMARKET_ROUND25_AI_SUPERVISOR_PRELOAD_SECONDS = 60.0
POLYMARKET_ROUND25_AI_SUPERVISOR_ACTIONS = (
    "normal",
    "cautious_50",
    "defensive_25",
    "halt_300s",
)
POLYMARKET_ROUND25_AI_SUPERVISOR_FAILURE_CODES = (
    "packet_stale",
    "model_identity_failure",
    "provider_failure",
    "schema_failure",
    "telemetry_failure",
    "residency_failure",
    "latency_failure",
    "response_stale",
    "pending_response",
    "queue_full",
    "worker_closed",
    "worker_failure",
)
_ACTION_POLICY = {
    "normal": (1.0, 0, ("no_additional_regime_restriction",)),
    "cautious_50": (0.5, 0, ("regime_size_reduction",)),
    "defensive_25": (0.25, 0, ("defensive_regime",)),
    "halt_300s": (0.0, 300_000, ("regime_halt", "cooldown_required")),
}
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "regime_action": {
            "type": "string",
            "enum": list(POLYMARKET_ROUND25_AI_SUPERVISOR_ACTIONS),
        }
    },
    "required": ["regime_action"],
    "additionalProperties": False,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_OLLAMA_VERSION = re.compile(r"^0\.32\.(\d+)$")
_PROVIDER_DURATION_TOLERANCE_SECONDS = 1.0
_STOP = object()


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Round 25 AI supervisor {name} is not numeric")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"Round 25 AI supervisor {name} is not finite")
    return selected


def _bounded(value: object, *, name: str, minimum: float, maximum: float) -> float:
    selected = _finite(value, name=name)
    if not minimum <= selected <= maximum:
        raise ValueError(f"Round 25 AI supervisor {name} is outside its bound")
    return selected


@dataclass(frozen=True, slots=True)
class Round25AISupervisorConfig:
    candidate_id: str = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.candidate_id
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PROVIDER_SECONDS
    seed: int = 25_026
    maximum_output_tokens: int = 24
    context_tokens: int = 4096
    keep_alive: str = "2m"

    def validated(self) -> Round25AISupervisorConfig:
        timeout = _finite(self.timeout_seconds, name="timeout")
        if (
            self.candidate_id != POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.candidate_id
            or self.base_url != "http://127.0.0.1:11434"
            or not 0.1 <= timeout <= POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PROVIDER_SECONDS
            or self.seed != 25_026
            or self.maximum_output_tokens != 24
            or self.context_tokens != 4096
            or self.keep_alive != "2m"
        ):
            raise ValueError("Round 25 AI supervisor configuration differs")
        return self


@dataclass(frozen=True, slots=True)
class Round25AISupervisorPacket:
    condition_id: str
    window_start_ms: int
    observed_at_ms: int
    expires_at_ms: int
    feature_source_chain_sha256: str
    clob_relative_spread_median_60s: float
    clob_relative_spread_p95_60s: float
    clob_top_executable_notional_p10_usd_60s: float
    clob_book_receipt_age_p95_ms_60s: float
    reference_receipt_age_p95_ms_60s: float
    realized_volatility_60s: float
    realized_volatility_300s: float
    absolute_log_return_60s: float
    absolute_log_return_300s: float
    absolute_order_flow_imbalance_mean_60s: float
    market_probability_range_60s: float
    round_trip_cost_bps_p95_60s: float
    portfolio_risk_utilization: float
    current_condition_exposure_fraction: float
    deterministic_gate_sha256: str
    packet_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_PACKET_SCHEMA_VERSION
    supervisor_contract_sha256: str = POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256
    target_accessed: bool = False
    outcome_accessed: bool = False
    resolution_accessed: bool = False
    fill_or_pnl_accessed: bool = False
    credential_accessed: bool = False
    deterministic_market_data_gate_passed: bool = True
    transport_gap_count_60s: int = 0
    unknown_order_state: bool = False
    unknown_position_state: bool = False
    recovery_requalification_pending: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "packet_sha256"}

    def prompt_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "window": {
                "start_ms": self.window_start_ms,
                "observed_at_ms": self.observed_at_ms,
                "expires_at_ms": self.expires_at_ms,
            },
            "liquidity_and_data_quality": {
                "relative_spread_median_60s": self.clob_relative_spread_median_60s,
                "relative_spread_p95_60s": self.clob_relative_spread_p95_60s,
                "top_executable_notional_p10_usd_60s": (
                    self.clob_top_executable_notional_p10_usd_60s
                ),
                "book_receipt_age_p95_ms_60s": (
                    self.clob_book_receipt_age_p95_ms_60s
                ),
                "reference_receipt_age_p95_ms_60s": (
                    self.reference_receipt_age_p95_ms_60s
                ),
                "transport_gap_count_60s": self.transport_gap_count_60s,
                "round_trip_cost_bps_p95_60s": self.round_trip_cost_bps_p95_60s,
            },
            "market_regime": {
                "realized_volatility_60s": self.realized_volatility_60s,
                "realized_volatility_300s": self.realized_volatility_300s,
                "absolute_log_return_60s": self.absolute_log_return_60s,
                "absolute_log_return_300s": self.absolute_log_return_300s,
                "absolute_order_flow_imbalance_mean_60s": (
                    self.absolute_order_flow_imbalance_mean_60s
                ),
                "market_probability_range_60s": self.market_probability_range_60s,
            },
            "risk_state": {
                "portfolio_risk_utilization": self.portfolio_risk_utilization,
                "current_condition_exposure_fraction": (
                    self.current_condition_exposure_fraction
                ),
            },
        }

    def validated(self) -> Round25AISupervisorPacket:
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_PACKET_SCHEMA_VERSION
            or self.supervisor_contract_sha256
            != POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or _SHA256.fullmatch(self.feature_source_chain_sha256) is None
            or _SHA256.fullmatch(self.deterministic_gate_sha256) is None
            or isinstance(self.window_start_ms, bool)
            or isinstance(self.observed_at_ms, bool)
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.window_start_ms, int)
            or not isinstance(self.observed_at_ms, int)
            or not isinstance(self.expires_at_ms, int)
            or self.window_start_ms < 0
            or self.observed_at_ms - self.window_start_ms
            != POLYMARKET_ROUND25_AI_SUPERVISOR_LOOKBACK_MS
            or self.expires_at_ms - self.observed_at_ms
            != POLYMARKET_ROUND25_AI_SUPERVISOR_VALIDITY_MS
        ):
            raise ValueError("Round 25 AI supervisor packet identity differs")
        median_spread = _bounded(
            self.clob_relative_spread_median_60s,
            name="median spread",
            minimum=0.0,
            maximum=0.5,
        )
        p95_spread = _bounded(
            self.clob_relative_spread_p95_60s,
            name="p95 spread",
            minimum=0.0,
            maximum=0.5,
        )
        if p95_spread < median_spread:
            raise ValueError("Round 25 AI supervisor spread quantiles differ")
        _bounded(
            self.clob_top_executable_notional_p10_usd_60s,
            name="p10 executable notional",
            minimum=0.01,
            maximum=1_000_000_000.0,
        )
        _bounded(
            self.clob_book_receipt_age_p95_ms_60s,
            name="book receipt age",
            minimum=0.0,
            maximum=500.0,
        )
        _bounded(
            self.reference_receipt_age_p95_ms_60s,
            name="reference receipt age",
            minimum=0.0,
            maximum=1_000.0,
        )
        for name, value in (
            ("realized volatility 60s", self.realized_volatility_60s),
            ("realized volatility 300s", self.realized_volatility_300s),
            ("absolute log return 60s", self.absolute_log_return_60s),
            ("absolute log return 300s", self.absolute_log_return_300s),
        ):
            _bounded(value, name=name, minimum=0.0, maximum=2.0)
        _bounded(
            self.absolute_order_flow_imbalance_mean_60s,
            name="absolute order flow imbalance",
            minimum=0.0,
            maximum=1.0,
        )
        _bounded(
            self.market_probability_range_60s,
            name="market probability range",
            minimum=0.0,
            maximum=1.0,
        )
        _bounded(
            self.round_trip_cost_bps_p95_60s,
            name="round trip cost",
            minimum=0.0,
            maximum=10_000.0,
        )
        _bounded(
            self.portfolio_risk_utilization,
            name="portfolio risk utilization",
            minimum=0.0,
            maximum=1.0,
        )
        _bounded(
            self.current_condition_exposure_fraction,
            name="condition exposure",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            self.target_accessed
            or self.outcome_accessed
            or self.resolution_accessed
            or self.fill_or_pnl_accessed
            or self.credential_accessed
            or not self.deterministic_market_data_gate_passed
            or isinstance(self.transport_gap_count_60s, bool)
            or self.transport_gap_count_60s != 0
            or self.unknown_order_state
            or self.unknown_position_state
            or self.recovery_requalification_pending
            or self.trading_authority
        ):
            raise ValueError("Round 25 AI supervisor packet safety state differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.packet_sha256:
            object.__setattr__(self, "packet_sha256", expected)
        elif self.packet_sha256 != expected:
            raise ValueError("Round 25 AI supervisor packet hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


@dataclass(frozen=True, slots=True)
class Round25AISupervisorAdvisory:
    candidate_id: str
    model: str
    model_digest: str
    packet_sha256: str
    generated_at_ms: int
    expires_at_ms: int
    regime_action: str
    maximum_size_multiplier: float
    cooldown_ms: int
    reason_codes: tuple[str, ...]
    valid_model_response: bool
    failure_code: str | None
    advisory_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_ADVISORY_SCHEMA_VERSION
    supervisor_contract_sha256: str = POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "advisory_sha256"}

    def validated(self) -> Round25AISupervisorAdvisory:
        failure = self.failure_code
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_ADVISORY_SCHEMA_VERSION
            or self.supervisor_contract_sha256
            != POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256
            or self.candidate_id != POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.candidate_id
            or self.model != POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model
            or self.model_digest != POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest
            or _SHA256.fullmatch(self.packet_sha256) is None
            or isinstance(self.generated_at_ms, bool)
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.generated_at_ms, int)
            or not isinstance(self.expires_at_ms, int)
            or self.generated_at_ms < 0
            or (self.valid_model_response and self.expires_at_ms < self.generated_at_ms)
            or self.trading_authority
        ):
            raise ValueError("Round 25 AI supervisor advisory identity differs")
        multiplier = _bounded(
            self.maximum_size_multiplier,
            name="size multiplier",
            minimum=0.0,
            maximum=1.0,
        )
        if isinstance(self.cooldown_ms, bool) or not isinstance(self.cooldown_ms, int):
            raise ValueError("Round 25 AI supervisor cooldown differs")
        if self.valid_model_response:
            expected = _ACTION_POLICY.get(self.regime_action)
            if (
                failure is not None
                or expected is None
                or (multiplier, self.cooldown_ms, self.reason_codes) != expected
            ):
                raise ValueError("Round 25 AI supervisor response semantics differ")
        elif (
            failure not in POLYMARKET_ROUND25_AI_SUPERVISOR_FAILURE_CODES
            or self.regime_action != "halt_300s"
            or multiplier != 0.0
            or self.cooldown_ms != 0
            or self.reason_codes != (failure,)
        ):
            raise ValueError("Round 25 AI supervisor failure semantics differ")
        expected_hash = _canonical_sha256(self.identity_payload())
        if not self.advisory_sha256:
            object.__setattr__(self, "advisory_sha256", expected_hash)
        elif self.advisory_sha256 != expected_hash:
            raise ValueError("Round 25 AI supervisor advisory hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


@dataclass(frozen=True, slots=True)
class Round25AISupervisorTelemetry:
    packet_sha256: str
    ollama_version: str
    show_metadata_sha256: str
    prompt_sha256: str
    response_sha256: str
    measured_latency_seconds: float
    provider_total_duration_ns: int
    provider_load_duration_ns: int
    provider_prompt_eval_count: int
    provider_prompt_eval_duration_ns: int
    provider_eval_count: int
    provider_eval_duration_ns: int
    residency: OllamaResidencyReport
    telemetry_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_TELEMETRY_SCHEMA_VERSION

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("telemetry_sha256")
        return payload

    def validated(self) -> Round25AISupervisorTelemetry:
        latency = _finite(self.measured_latency_seconds, name="latency")
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_TELEMETRY_SCHEMA_VERSION
            or _SHA256.fullmatch(self.packet_sha256) is None
            or _OLLAMA_VERSION.fullmatch(self.ollama_version) is None
            or _SHA256.fullmatch(self.show_metadata_sha256) is None
            or _SHA256.fullmatch(self.prompt_sha256) is None
            or _SHA256.fullmatch(self.response_sha256) is None
            or not 0.0 <= latency <= POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PROVIDER_SECONDS
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
                for value, minimum in (
                    (self.provider_total_duration_ns, 1),
                    (self.provider_load_duration_ns, 0),
                    (self.provider_prompt_eval_count, 1),
                    (self.provider_prompt_eval_duration_ns, 1),
                    (self.provider_eval_count, 1),
                    (self.provider_eval_duration_ns, 1),
                )
            )
        ):
            raise ValueError("Round 25 AI supervisor telemetry differs")
        residency = self.residency.validated()
        if (
            residency.digest != POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest
            or not residency.fully_gpu_resident
            or residency.vram_to_model_ratio is None
            or residency.vram_to_model_ratio
            < POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO
        ):
            raise ValueError("Round 25 AI supervisor residency differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.telemetry_sha256:
            object.__setattr__(self, "telemetry_sha256", expected)
        elif self.telemetry_sha256 != expected:
            raise ValueError("Round 25 AI supervisor telemetry hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


@dataclass(frozen=True, slots=True)
class Round25AISupervisorResult:
    packet_sha256: str
    advisory: Round25AISupervisorAdvisory
    telemetry: Round25AISupervisorTelemetry | None
    result_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_SUPERVISOR_RESULT_SCHEMA_VERSION
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("result_sha256")
        return payload

    def validated(self) -> Round25AISupervisorResult:
        advisory = self.advisory.validated()
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_SUPERVISOR_RESULT_SCHEMA_VERSION
            or _SHA256.fullmatch(self.packet_sha256) is None
            or advisory.packet_sha256 != self.packet_sha256
            or self.trading_authority
            or (advisory.valid_model_response and self.telemetry is None)
            or (not advisory.valid_model_response and self.telemetry is not None)
        ):
            raise ValueError("Round 25 AI supervisor result differs")
        if self.telemetry is not None:
            telemetry = self.telemetry.validated()
            if telemetry.packet_sha256 != self.packet_sha256:
                raise ValueError("Round 25 AI supervisor telemetry packet differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.result_sha256:
            object.__setattr__(self, "result_sha256", expected)
        elif self.result_sha256 != expected:
            raise ValueError("Round 25 AI supervisor result hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


JsonGetter = Callable[[str, float], object]
JsonPoster = Callable[[str, Mapping[str, object], float], object]
ResidencyInspector = Callable[..., OllamaResidencyReport]
WallClock = Callable[[], int]
MonotonicClock = Callable[[], int]


def preflight_round25_ai_supervisor(
    config: Round25AISupervisorConfig,
    *,
    get_json: JsonGetter = _get_json,
    post_json: JsonPoster = _post_json,
) -> tuple[str, str]:
    cfg = config.validated()
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    version_payload = get_json(f"{cfg.base_url}/api/version", cfg.timeout_seconds)
    if not isinstance(version_payload, Mapping) or set(version_payload) != {"version"}:
        raise ValueError("Round 25 AI supervisor Ollama version differs")
    version = str(version_payload["version"] or "")
    version_match = _OLLAMA_VERSION.fullmatch(version)
    if version_match is None or int(version_match.group(1)) < 4:
        raise ValueError("Round 25 AI supervisor Ollama version is outside contract")
    tags = get_json(f"{cfg.base_url}/api/tags", cfg.timeout_seconds)
    if not isinstance(tags, Mapping) or not isinstance(tags.get("models"), list):
        raise ValueError("Round 25 AI supervisor model inventory differs")
    matches = []
    for raw in tags["models"]:
        if not isinstance(raw, Mapping):
            raise ValueError("Round 25 AI supervisor inventory entry differs")
        names = {_normalized_model(raw.get("name")), _normalized_model(raw.get("model"))}
        if _normalized_model(candidate.model) in names:
            matches.append(raw)
    if len(matches) != 1 or matches[0].get("digest") != candidate.digest:
        raise ValueError("Round 25 AI supervisor model digest differs")
    details = matches[0].get("details")
    if (
        not isinstance(details, Mapping)
        or details.get("format") != "gguf"
        or details.get("parameter_size") != candidate.parameter_size
        or details.get("quantization_level") not in {candidate.quantization, "unknown"}
    ):
        raise ValueError("Round 25 AI supervisor inventory metadata differs")
    show = post_json(
        f"{cfg.base_url}/api/show",
        {"model": candidate.model, "verbose": False},
        cfg.timeout_seconds,
    )
    if not isinstance(show, Mapping):
        raise ValueError("Round 25 AI supervisor model details differ")
    show_details = show.get("details")
    model_info = show.get("model_info")
    if (
        not isinstance(show_details, Mapping)
        or show_details.get("format") != "gguf"
        or show_details.get("parameter_size") != candidate.parameter_size
        or show_details.get("quantization_level") != candidate.quantization
        or not isinstance(model_info, Mapping)
        or int(model_info.get("general.parameter_count") or 0) < 7_000_000_000
    ):
        raise ValueError("Round 25 AI supervisor model details metadata differs")
    return version, _canonical_sha256(show)


def _prompt(packet: Round25AISupervisorPacket) -> str:
    instructions = (
        "Classify one BTC 5m market-risk regime using only this target-free causal "
        "60-second packet. Select exactly one schema regime_action. This is a risk "
        "supervisor, not a predictor: never infer direction, create risk, change side, "
        "or affect exits. Higher spread, quote age, volatility, absolute returns, "
        "probability variation, order-flow imbalance, costs, exposure, and portfolio "
        "utilization increase restriction; lower executable depth increases restriction. "
        "Never become less restrictive when risk worsens. Use normal only for a benign, "
        "well-observed regime. Return only schema JSON. Packet:"
    )
    return f"{instructions}{_canonical_json(packet.prompt_payload())}"


def _parse_response(value: object) -> tuple[str, float, int, tuple[str, ...]]:
    if not isinstance(value, str) or len(value.encode("utf-8")) > 20_000:
        raise ValueError("Round 25 AI supervisor response text differs")
    parsed = _strict_json_value(value)
    if not isinstance(parsed, Mapping) or set(parsed) != {"regime_action"}:
        raise ValueError("Round 25 AI supervisor response fields differ")
    action = parsed["regime_action"]
    if not isinstance(action, str) or action not in _ACTION_POLICY:
        raise ValueError("Round 25 AI supervisor action differs")
    multiplier, cooldown_ms, reason_codes = _ACTION_POLICY[action]
    return action, multiplier, cooldown_ms, reason_codes


def _failure_result(
    packet: Round25AISupervisorPacket,
    failure_code: str,
    generated_at_ms: int,
) -> Round25AISupervisorResult:
    if failure_code not in POLYMARKET_ROUND25_AI_SUPERVISOR_FAILURE_CODES:
        raise ValueError("Round 25 AI supervisor failure code differs")
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    advisory = Round25AISupervisorAdvisory(
        candidate_id=candidate.candidate_id,
        model=candidate.model,
        model_digest=candidate.digest,
        packet_sha256=packet.packet_sha256,
        generated_at_ms=max(0, int(generated_at_ms)),
        expires_at_ms=packet.expires_at_ms,
        regime_action="halt_300s",
        maximum_size_multiplier=0.0,
        cooldown_ms=0,
        reason_codes=(failure_code,),
        valid_model_response=False,
        failure_code=failure_code,
    )
    return Round25AISupervisorResult(
        packet_sha256=packet.packet_sha256,
        advisory=advisory,
        telemetry=None,
    )


def review_round25_ai_supervisor_packet(
    packet: Round25AISupervisorPacket,
    config: Round25AISupervisorConfig = Round25AISupervisorConfig(),
    *,
    get_json: JsonGetter = _get_json,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
    wall_clock_ms: WallClock = lambda: time.time_ns() // 1_000_000,
    monotonic_ns: MonotonicClock = time.perf_counter_ns,
) -> Round25AISupervisorResult:
    selected = packet.validated()
    cfg = config.validated()
    submitted_at_ms = int(wall_clock_ms())
    if (
        submitted_at_ms < selected.observed_at_ms
        or submitted_at_ms - selected.observed_at_ms
        > POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PACKET_AGE_MS
        or submitted_at_ms > selected.expires_at_ms
    ):
        return _failure_result(selected, "packet_stale", submitted_at_ms)
    try:
        version, metadata_sha256 = preflight_round25_ai_supervisor(
            cfg,
            get_json=get_json,
            post_json=post_json,
        )
    except (OSError, TypeError, ValueError):
        return _failure_result(selected, "model_identity_failure", int(wall_clock_ms()))
    prompt = _prompt(selected)
    request_payload = {
        "model": POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model,
        "prompt": prompt,
        "system": (
            "You are a conservative financial risk supervisor. You can only reduce "
            "or halt future entries and can never affect exits."
        ),
        "format": _RESPONSE_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": cfg.keep_alive,
        "options": {
            "temperature": 0.0,
            "seed": cfg.seed,
            "num_predict": cfg.maximum_output_tokens,
            "num_ctx": cfg.context_tokens,
        },
    }
    started_ns = int(monotonic_ns())
    try:
        provider = post_json(
            f"{cfg.base_url}/api/generate",
            request_payload,
            cfg.timeout_seconds,
        )
    except (OSError, TypeError, ValueError):
        return _failure_result(selected, "provider_failure", int(wall_clock_ms()))
    finished_ns = int(monotonic_ns())
    generated_at_ms = int(wall_clock_ms())
    measured_latency = max(0.0, (finished_ns - started_ns) / 1_000_000_000.0)
    try:
        usage = _provider_usage(provider)
        if _normalized_model(usage["model"]) != _normalized_model(
            POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model
        ):
            raise ValueError("Round 25 AI supervisor provider model differs")
        if measured_latency > POLYMARKET_ROUND25_AI_SUPERVISOR_MAXIMUM_PROVIDER_SECONDS:
            raise TimeoutError("Round 25 AI supervisor latency exceeded")
        if (
            int(usage["total_duration"]) / 1_000_000_000.0
            > measured_latency + _PROVIDER_DURATION_TOLERANCE_SECONDS
        ):
            raise ValueError("Round 25 AI supervisor provider duration differs")
    except TimeoutError:
        return _failure_result(selected, "latency_failure", generated_at_ms)
    except (TypeError, ValueError):
        return _failure_result(selected, "telemetry_failure", generated_at_ms)
    if generated_at_ms > selected.expires_at_ms:
        return _failure_result(selected, "response_stale", generated_at_ms)
    try:
        assert isinstance(provider, Mapping)
        action, multiplier, cooldown_ms, reason_codes = _parse_response(
            provider.get("response")
        )
    except (AssertionError, TypeError, ValueError):
        return _failure_result(selected, "schema_failure", generated_at_ms)
    try:
        residency = residency_inspector(
            cfg.base_url,
            POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.model,
            min(2.0, cfg.timeout_seconds),
            expected_digest=POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest,
        ).validated()
        if (
            residency.digest != POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE.digest
            or not residency.fully_gpu_resident
            or residency.vram_to_model_ratio is None
            or residency.vram_to_model_ratio
            < POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO
        ):
            raise ValueError("Round 25 AI supervisor is not fully GPU resident")
    except (OSError, TypeError, ValueError):
        return _failure_result(selected, "residency_failure", generated_at_ms)
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    advisory = Round25AISupervisorAdvisory(
        candidate_id=candidate.candidate_id,
        model=candidate.model,
        model_digest=candidate.digest,
        packet_sha256=selected.packet_sha256,
        generated_at_ms=generated_at_ms,
        expires_at_ms=selected.expires_at_ms,
        regime_action=action,
        maximum_size_multiplier=multiplier,
        cooldown_ms=cooldown_ms,
        reason_codes=reason_codes,
        valid_model_response=True,
        failure_code=None,
    )
    telemetry = Round25AISupervisorTelemetry(
        packet_sha256=selected.packet_sha256,
        ollama_version=version,
        show_metadata_sha256=metadata_sha256,
        prompt_sha256=hashlib.sha256(prompt.encode("ascii")).hexdigest(),
        response_sha256=hashlib.sha256(
            str(provider["response"]).encode("utf-8")
        ).hexdigest(),
        measured_latency_seconds=measured_latency,
        provider_total_duration_ns=int(usage["total_duration"]),
        provider_load_duration_ns=int(usage["load_duration"]),
        provider_prompt_eval_count=int(usage["prompt_eval_count"]),
        provider_prompt_eval_duration_ns=int(usage["prompt_eval_duration"]),
        provider_eval_count=int(usage["eval_count"]),
        provider_eval_duration_ns=int(usage["eval_duration"]),
        residency=residency,
    )
    return Round25AISupervisorResult(
        packet_sha256=selected.packet_sha256,
        advisory=advisory,
        telemetry=telemetry,
    )


def preload_round25_ai_supervisor(
    config: Round25AISupervisorConfig = Round25AISupervisorConfig(),
    *,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
) -> OllamaResidencyReport:
    cfg = config.validated()
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    response = post_json(
        f"{cfg.base_url}/api/generate",
        {
            "model": candidate.model,
            "prompt": "",
            "stream": False,
            "keep_alive": cfg.keep_alive,
            "options": {"num_ctx": cfg.context_tokens},
        },
        POLYMARKET_ROUND25_AI_SUPERVISOR_PRELOAD_SECONDS,
    )
    if not isinstance(response, Mapping) or response.get("done") is not True:
        raise ValueError("Round 25 AI supervisor preload failed")
    report = residency_inspector(
        cfg.base_url,
        candidate.model,
        min(2.0, cfg.timeout_seconds),
        expected_digest=candidate.digest,
    ).validated()
    if (
        report.digest != candidate.digest
        or not report.fully_gpu_resident
        or report.vram_to_model_ratio is None
        or report.vram_to_model_ratio
        < POLYMARKET_ROUND25_AI_MINIMUM_GPU_RESIDENCY_RATIO
    ):
        raise ValueError("Round 25 AI supervisor preload residency differs")
    return report


def unload_round25_ai_supervisor(
    config: Round25AISupervisorConfig = Round25AISupervisorConfig(),
    *,
    post_json: JsonPoster = _post_json,
    residency_inspector: ResidencyInspector = inspect_ollama_model_residency,
) -> None:
    cfg = config.validated()
    candidate = POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE
    response = post_json(
        f"{cfg.base_url}/api/generate",
        {"model": candidate.model, "keep_alive": 0, "stream": False},
        cfg.timeout_seconds,
    )
    if (
        not isinstance(response, Mapping)
        or response.get("done") is not True
        or response.get("done_reason") != "unload"
    ):
        raise ValueError("Round 25 AI supervisor unload failed")
    report = residency_inspector(
        cfg.base_url,
        candidate.model,
        min(2.0, cfg.timeout_seconds),
        expected_digest=candidate.digest,
    ).validated()
    if report.loaded:
        raise ValueError("Round 25 AI supervisor remained loaded")


Reviewer = Callable[[Round25AISupervisorPacket], Round25AISupervisorResult]


class Round25AISupervisorWorker:
    """Capacity-one daemon worker; trading and safety loops never wait for it."""

    def __init__(self, reviewer: Reviewer) -> None:
        self._reviewer = reviewer
        self._queue: Queue[Round25AISupervisorPacket | object] = Queue(maxsize=1)
        self._lock = threading.Lock()
        self._inflight: str | None = None
        self._results: dict[str, Round25AISupervisorResult] = {}
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="round25-fin-r1-regime-supervisor",
            daemon=True,
        )
        self._thread.start()

    def submit(self, packet: Round25AISupervisorPacket) -> bool:
        selected = packet.validated()
        with self._lock:
            if (
                self._closed
                or self._inflight is not None
                or selected.packet_sha256 in self._results
            ):
                return False
            self._inflight = selected.packet_sha256
        try:
            self._queue.put_nowait(selected)
        except Full:
            with self._lock:
                self._inflight = None
            return False
        return True

    def poll(self, packet_sha256: str) -> Round25AISupervisorResult | None:
        if _SHA256.fullmatch(packet_sha256) is None:
            raise ValueError("Round 25 AI supervisor poll hash differs")
        with self._lock:
            return self._results.pop(packet_sha256, None)

    def advisory_or_fail_closed(
        self,
        packet: Round25AISupervisorPacket,
        *,
        now_ms: int | None = None,
    ) -> Round25AISupervisorResult:
        selected = packet.validated()
        result = self.poll(selected.packet_sha256)
        if result is not None:
            return result.validated()
        with self._lock:
            closed = self._closed
            pending = self._inflight == selected.packet_sha256
        failure_code = "worker_closed" if closed else "pending_response" if pending else "queue_full"
        return _failure_result(
            selected,
            failure_code,
            int(time.time_ns() // 1_000_000 if now_ms is None else now_ms),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._queue.put_nowait(_STOP)
        except Full:
            pass

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get()
            except Empty:
                continue
            if item is _STOP:
                return
            assert isinstance(item, Round25AISupervisorPacket)
            try:
                result = self._reviewer(item).validated()
                if result.packet_sha256 != item.packet_sha256:
                    raise ValueError("Round 25 AI supervisor worker result differs")
            except Exception:
                result = _failure_result(
                    item,
                    "worker_failure",
                    time.time_ns() // 1_000_000,
                )
            with self._lock:
                self._inflight = None
                self._results[item.packet_sha256] = result
                closed = self._closed
            if closed:
                return


@dataclass(frozen=True, slots=True)
class Round25AICombinedRiskDecision:
    fast_advisory_sha256: str
    supervisor_advisory_sha256: str
    generated_at_ms: int
    expires_at_ms: int
    maximum_size_multiplier: float
    cooldown_ms: int
    veto_new_entries: bool
    reason_codes: tuple[str, ...]
    decision_sha256: str = ""
    schema_version: str = POLYMARKET_ROUND25_AI_COMBINED_DECISION_SCHEMA_VERSION
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "decision_sha256"}

    def validated(self) -> Round25AICombinedRiskDecision:
        multiplier = _bounded(
            self.maximum_size_multiplier,
            name="combined multiplier",
            minimum=0.0,
            maximum=1.0,
        )
        if (
            self.schema_version != POLYMARKET_ROUND25_AI_COMBINED_DECISION_SCHEMA_VERSION
            or _SHA256.fullmatch(self.fast_advisory_sha256) is None
            or _SHA256.fullmatch(self.supervisor_advisory_sha256) is None
            or isinstance(self.generated_at_ms, bool)
            or isinstance(self.expires_at_ms, bool)
            or not isinstance(self.generated_at_ms, int)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms < self.generated_at_ms
            or isinstance(self.cooldown_ms, bool)
            or not isinstance(self.cooldown_ms, int)
            or not 0 <= self.cooldown_ms <= 300_000
            or self.veto_new_entries != (multiplier == 0.0)
            or not self.reason_codes
            or len(set(self.reason_codes)) != len(self.reason_codes)
            or self.trading_authority
        ):
            raise ValueError("Round 25 combined AI risk decision differs")
        expected = _canonical_sha256(self.identity_payload())
        if not self.decision_sha256:
            object.__setattr__(self, "decision_sha256", expected)
        elif self.decision_sha256 != expected:
            raise ValueError("Round 25 combined AI risk decision hash differs")
        return self

    def __post_init__(self) -> None:
        self.validated()


def combine_round25_ai_risk(
    fast: Round25AIAdvisory,
    supervisor: Round25AISupervisorAdvisory,
    *,
    now_ms: int,
) -> Round25AICombinedRiskDecision:
    fast_selected = fast.validated()
    supervisor_selected = supervisor.validated()
    if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
        raise ValueError("Round 25 combined AI risk clock differs")
    fast_multiplier = fast_selected.maximum_size_multiplier
    supervisor_multiplier = supervisor_selected.maximum_size_multiplier
    reasons = tuple(dict.fromkeys((*fast_selected.reason_codes, *supervisor_selected.reason_codes)))
    if now_ms > fast_selected.expires_at_ms:
        fast_multiplier = 0.0
        reasons = (*reasons, "fast_advisory_stale")
    if now_ms > supervisor_selected.expires_at_ms:
        supervisor_multiplier = 0.0
        reasons = (*reasons, "supervisor_advisory_stale")
    multiplier = min(fast_multiplier, supervisor_multiplier)
    return Round25AICombinedRiskDecision(
        fast_advisory_sha256=fast_selected.advisory_sha256,
        supervisor_advisory_sha256=supervisor_selected.advisory_sha256,
        generated_at_ms=now_ms,
        expires_at_ms=max(
            now_ms,
            min(fast_selected.expires_at_ms, supervisor_selected.expires_at_ms),
        ),
        maximum_size_multiplier=multiplier,
        cooldown_ms=max(fast_selected.cooldown_ms, supervisor_selected.cooldown_ms),
        veto_new_entries=multiplier == 0.0,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


__all__ = [
    "POLYMARKET_ROUND25_AI_SUPERVISOR_CANDIDATE",
    "POLYMARKET_ROUND25_AI_SUPERVISOR_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_AI_SUPERVISOR_UPLIFT_CONTRACT_SHA256",
    "Round25AICombinedRiskDecision",
    "Round25AISupervisorAdvisory",
    "Round25AISupervisorConfig",
    "Round25AISupervisorPacket",
    "Round25AISupervisorResult",
    "Round25AISupervisorTelemetry",
    "Round25AISupervisorWorker",
    "combine_round25_ai_risk",
    "preflight_round25_ai_supervisor",
    "preload_round25_ai_supervisor",
    "review_round25_ai_supervisor_packet",
    "unload_round25_ai_supervisor",
]
