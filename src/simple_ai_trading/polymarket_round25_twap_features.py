"""Target-blind causal features for Polymarket BTC five-minute Chainlink TWAP markets."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import math
import re
from statistics import fmean, pstdev
from typing import Mapping, Sequence


POLYMARKET_ROUND25_TWAP_FEATURE_SCHEMA_VERSION = (
    "polymarket-round25-twap-causal-features-v1"
)
POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256 = (
    "4f227d60c8a59b21f687679b41abdce668430480692526e230730c0248719fdc"
)
POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256 = (
    "ad3a320b9d4c6054260cbf5a560dca329f271e0dfb05d8f2ade3ae4d812062e2"
)
POLYMARKET_ROUND25_TWAP_TOPIC = "crypto_prices_chainlink"
POLYMARKET_ROUND25_TWAP_SYMBOL = "btc/usd"
POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS = 30
POLYMARKET_ROUND25_CONDITION_DURATION_MS = 300_000
POLYMARKET_ROUND25_DECISION_CADENCE_MS = 250
POLYMARKET_ROUND25_MAXIMUM_SOURCE_AGE_MS = 5_000
POLYMARKET_ROUND25_MAXIMUM_RECEIPT_AGE_MS = 5_000
POLYMARKET_ROUND25_MINIMUM_OBSERVATIONS = 5
POLYMARKET_ROUND25_MINIMUM_COVERAGE_MS = 2_000
_LAG_SECONDS = (1, 5, 10, 30, 60)
_SLOPE_SECONDS = (5, 10, 30, 60)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_E18_INTEGER = re.compile(r"^[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


POLYMARKET_ROUND25_TWAP_FEATURE_NAMES = (
    "twap.phase.elapsed_fraction",
    "twap.phase.remaining_seconds",
    "twap.phase.final_window_overlap_fraction",
    "twap.reference.log_distance_from_exact_open",
    *(
        name
        for lag in _LAG_SECONDS
        for name in (
            f"twap.path.log_return_{lag}s",
            f"twap.path.log_return_{lag}s_available",
        )
    ),
    *(
        name
        for window in _SLOPE_SECONDS
        for name in (
            f"twap.path.log_slope_{window}s_per_second",
            f"twap.path.log_slope_{window}s_available",
        )
    ),
    "twap.path.overlapping_realized_variance_rate",
    "twap.path.nonoverlapping_30s_realized_variance_rate",
    "twap.path.lag1_return_autocorrelation",
    "twap.path.tanh_distance_to_nonoverlapping_scale",
    "twap.path.nonoverlapping_scale_available",
    "twap.transport.source_age_ms",
    "twap.transport.receipt_age_ms",
    "twap.transport.publisher_minus_source_ms",
    "twap.transport.receipt_minus_publisher_ms",
    "twap.transport.observation_count",
    "twap.transport.coverage_seconds",
    "twap.transport.source_interval_mean_ms",
    "twap.transport.source_interval_std_ms",
    "twap.transport.source_interval_max_ms",
    "twap.transport.identical_duplicate_count",
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _int_timestamp(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Round25TwapObservation:
    source_timestamp_ms: int
    publisher_timestamp_ms: int
    received_wall_ms: int
    received_monotonic_ns: int
    full_accuracy_value_e18: int
    raw_frame_sha256: str
    symbol: str = POLYMARKET_ROUND25_TWAP_SYMBOL
    window_seconds: int = POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS
    wire_schema_correction_sha256: str = (
        POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256
    )

    def __post_init__(self) -> None:
        if (
            type(self.source_timestamp_ms) is not int
            or self.source_timestamp_ms <= 0
            or type(self.publisher_timestamp_ms) is not int
            or self.publisher_timestamp_ms < self.source_timestamp_ms
            or type(self.received_wall_ms) is not int
            or self.received_wall_ms <= 0
            or self.publisher_timestamp_ms > self.received_wall_ms
            or type(self.received_monotonic_ns) is not int
            or self.received_monotonic_ns <= 0
            or type(self.full_accuracy_value_e18) is not int
            or self.full_accuracy_value_e18 <= 0
            or not isinstance(self.raw_frame_sha256, str)
            or _SHA256.fullmatch(self.raw_frame_sha256) is None
            or self.symbol != POLYMARKET_ROUND25_TWAP_SYMBOL
            or type(self.window_seconds) is not int
            or self.window_seconds != POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS
            or self.wire_schema_correction_sha256
            != POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256
        ):
            raise ValueError("Round 25 TWAP observation is invalid")

    @classmethod
    def from_raw_frame(
        cls,
        raw_text: str,
        *,
        received_wall_ms: int,
        received_monotonic_ns: int,
    ) -> Round25TwapObservation:
        if not isinstance(raw_text, str) or not raw_text:
            raise ValueError("Round 25 TWAP frame must be nonempty text")
        try:
            event = json.loads(
                raw_text,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_nonfinite,
            )
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            raise ValueError("Round 25 TWAP frame is not strict JSON") from exc
        if not isinstance(event, Mapping):
            raise ValueError("Round 25 TWAP frame is not an object")
        payload = event.get("payload")
        required_event_keys = {"payload", "timestamp", "topic", "type"}
        allowed_event_keys = required_event_keys | {"connection_id"}
        if (
            set(event) not in (required_event_keys, allowed_event_keys)
            or event.get("topic") != POLYMARKET_ROUND25_TWAP_TOPIC
            or event.get("type") != "update"
            or not isinstance(payload, Mapping)
            or set(payload)
            != {"full_accuracy_value", "symbol", "timestamp", "value"}
            or payload.get("symbol") != POLYMARKET_ROUND25_TWAP_SYMBOL
            or (
                "connection_id" in event
                and (
                    not isinstance(event["connection_id"], str)
                    or not 1 <= len(event["connection_id"]) <= 128
                )
            )
        ):
            raise ValueError("Round 25 TWAP frame identity differs")
        full_accuracy = payload.get("full_accuracy_value")
        if (
            not isinstance(full_accuracy, str)
            or _E18_INTEGER.fullmatch(full_accuracy) is None
        ):
            raise ValueError("Round 25 TWAP exact value differs")
        source_timestamp = _int_timestamp(
            payload.get("timestamp"), name="source_timestamp_ms"
        )
        publisher_timestamp = _int_timestamp(
            event.get("timestamp"), name="publisher_timestamp_ms"
        )
        if publisher_timestamp < source_timestamp:
            raise ValueError("Round 25 TWAP publisher time precedes source time")
        exact_value = int(full_accuracy)
        display_value = payload.get("value")
        if (
            isinstance(display_value, bool)
            or not isinstance(display_value, (int, float))
            or not math.isfinite(float(display_value))
            or float(display_value) <= 0.0
            or not math.isclose(
                float(display_value),
                exact_value / 1_000_000_000_000_000_000,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("Round 25 TWAP display value differs from exact E18")
        return cls(
            source_timestamp_ms=source_timestamp,
            publisher_timestamp_ms=publisher_timestamp,
            received_wall_ms=_int_timestamp(
                received_wall_ms, name="received_wall_ms"
            ),
            received_monotonic_ns=_int_timestamp(
                received_monotonic_ns, name="received_monotonic_ns"
            ),
            full_accuracy_value_e18=exact_value,
            raw_frame_sha256=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class Round25TwapFeatureSnapshot:
    condition_id: str
    event_start_ms: int
    decision_time_ms: int
    available: bool
    reasons: tuple[str, ...]
    values: tuple[float, ...]
    source_chain_sha256: str
    maximum_receipt_ms: int
    opening_value_e18: int
    latest_value_e18: int
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    wire_schema_correction_sha256: str = (
        POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256
    )
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.condition_id, str)
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or self.event_start_ms <= 0
            or type(self.decision_time_ms) is not int
            or self.decision_time_ms <= 0
            or type(self.available) is not bool
            or not isinstance(self.reasons, tuple)
            or any(not isinstance(reason, str) or not reason for reason in self.reasons)
            or len(set(self.reasons)) != len(self.reasons)
            or not isinstance(self.values, tuple)
            or len(self.values) != len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in self.values
            )
            or not isinstance(self.source_chain_sha256, str)
            or _SHA256.fullmatch(self.source_chain_sha256) is None
            or type(self.maximum_receipt_ms) is not int
            or type(self.opening_value_e18) is not int
            or type(self.latest_value_e18) is not int
            or not isinstance(self.model_design_sha256, str)
            or self.model_design_sha256
            != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.wire_schema_correction_sha256
            != POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256
            or self.trading_authority is not False
        ):
            raise ValueError("Round 25 TWAP feature snapshot is invalid")
        if self.available:
            if (
                self.reasons
                or self.source_chain_sha256 == _EMPTY_SHA256
                or not 0 < self.maximum_receipt_ms <= self.decision_time_ms
                or self.opening_value_e18 <= 0
                or self.latest_value_e18 <= 0
            ):
                raise ValueError("Round 25 available TWAP snapshot differs")
        elif (
            not self.reasons
            or any(self.values)
            or self.source_chain_sha256 != _EMPTY_SHA256
            or self.maximum_receipt_ms != 0
            or self.opening_value_e18 != 0
            or self.latest_value_e18 != 0
        ):
            raise ValueError("Round 25 unavailable TWAP snapshot differs")


def _slope_per_second(
    timestamps_ms: Sequence[int],
    log_values: Sequence[float],
) -> float:
    origin = timestamps_ms[0]
    seconds = [(timestamp - origin) / 1_000.0 for timestamp in timestamps_ms]
    mean_time = fmean(seconds)
    mean_value = fmean(log_values)
    denominator = math.fsum((value - mean_time) ** 2 for value in seconds)
    if denominator <= 0.0:
        return 0.0
    return math.fsum(
        (time_value - mean_time) * (price - mean_value)
        for time_value, price in zip(seconds, log_values, strict=True)
    ) / denominator


def _lag1_autocorrelation(returns: Sequence[float]) -> float:
    if len(returns) < 3:
        return 0.0
    left = returns[:-1]
    right = returns[1:]
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = math.fsum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left)
        * math.fsum((value - right_mean) ** 2 for value in right)
    )
    return 0.0 if denominator <= 0.0 else max(-1.0, min(1.0, numerator / denominator))


class Round25TwapFeatureEngine:
    """Incrementally build receipt-causal TWAP features without settlement labels."""

    feature_schema_version = POLYMARKET_ROUND25_TWAP_FEATURE_SCHEMA_VERSION
    feature_names = POLYMARKET_ROUND25_TWAP_FEATURE_NAMES
    trading_authority = False

    def __init__(self, *, condition_id: str, event_start_ms: int) -> None:
        if not isinstance(condition_id, str) or type(event_start_ms) is not int:
            raise ValueError("Round 25 TWAP condition identity is invalid")
        condition = condition_id.strip().lower()
        start = event_start_ms
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or start <= 0
            or start % POLYMARKET_ROUND25_CONDITION_DURATION_MS
        ):
            raise ValueError("Round 25 TWAP condition identity is invalid")
        self.condition_id = condition
        self.event_start_ms = start
        self.event_end_ms = start + POLYMARKET_ROUND25_CONDITION_DURATION_MS
        self._observations: list[Round25TwapObservation] = []
        self._last_monotonic_ns = 0
        self._gap_detected = False

    def mark_stream_gap(self) -> None:
        self._gap_detected = True

    def ingest(self, observation: Round25TwapObservation) -> None:
        if not isinstance(observation, Round25TwapObservation):
            raise TypeError("Round 25 TWAP observation type differs")
        if (
            observation.received_monotonic_ns <= self._last_monotonic_ns
            or not self.event_start_ms
            <= observation.source_timestamp_ms
            <= self.event_end_ms
        ):
            raise ValueError("Round 25 TWAP observation chronology differs")
        self._last_monotonic_ns = observation.received_monotonic_ns
        self._observations.append(observation)

    def _unavailable(
        self,
        *,
        decision_time_ms: int,
        reasons: Sequence[str],
    ) -> Round25TwapFeatureSnapshot:
        return Round25TwapFeatureSnapshot(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=decision_time_ms,
            available=False,
            reasons=tuple(dict.fromkeys(reasons)),
            values=(0.0,) * len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES),
            source_chain_sha256=_EMPTY_SHA256,
            maximum_receipt_ms=0,
            opening_value_e18=0,
            latest_value_e18=0,
        )

    def build(self, decision_time_ms: int) -> Round25TwapFeatureSnapshot:
        if type(decision_time_ms) is not int:
            raise ValueError("Round 25 TWAP decision time is invalid")
        decision = decision_time_ms
        if (
            not self.event_start_ms <= decision < self.event_end_ms
            or (decision - self.event_start_ms)
            % POLYMARKET_ROUND25_DECISION_CADENCE_MS
        ):
            raise ValueError("Round 25 TWAP decision time is invalid")
        if any(item.received_wall_ms > decision for item in self._observations):
            raise ValueError("Round 25 TWAP engine contains future receipts")

        reasons: list[str] = []
        if self._gap_detected:
            reasons.append("twap_stream_gap_detected")
        by_source: dict[int, Round25TwapObservation] = {}
        duplicate_count = 0
        conflicting_source = False
        for observation in self._observations:
            prior = by_source.get(observation.source_timestamp_ms)
            if prior is None:
                by_source[observation.source_timestamp_ms] = observation
            elif prior.full_accuracy_value_e18 != observation.full_accuracy_value_e18:
                conflicting_source = True
            else:
                duplicate_count += 1
        if conflicting_source:
            reasons.append("conflicting_twap_source_timestamp")
        ordered = tuple(by_source[timestamp] for timestamp in sorted(by_source))
        opening = by_source.get(self.event_start_ms)
        if opening is None:
            reasons.append("exact_opening_twap_unavailable")
        if len(ordered) < POLYMARKET_ROUND25_MINIMUM_OBSERVATIONS:
            reasons.append("twap_observation_count_below_minimum")
        coverage_ms = (
            0
            if len(ordered) < 2
            else ordered[-1].source_timestamp_ms - ordered[0].source_timestamp_ms
        )
        if coverage_ms < POLYMARKET_ROUND25_MINIMUM_COVERAGE_MS:
            reasons.append("twap_coverage_below_minimum")
        latest = ordered[-1] if ordered else None
        if latest is None:
            reasons.append("twap_unavailable")
        else:
            if decision - latest.source_timestamp_ms > (
                POLYMARKET_ROUND25_MAXIMUM_SOURCE_AGE_MS
            ):
                reasons.append("twap_source_stale")
            if decision - latest.received_wall_ms > (
                POLYMARKET_ROUND25_MAXIMUM_RECEIPT_AGE_MS
            ):
                reasons.append("twap_receipt_stale")
        if reasons:
            return self._unavailable(decision_time_ms=decision, reasons=reasons)
        if opening is None or latest is None:
            raise RuntimeError("Round 25 TWAP availability reconciliation failed")

        timestamps = [item.source_timestamp_ms for item in ordered]
        logs = [math.log(item.full_accuracy_value_e18) for item in ordered]
        latest_timestamp = latest.source_timestamp_ms
        latest_log = logs[-1]
        opening_log = math.log(opening.full_accuracy_value_e18)
        log_distance = latest_log - opening_log
        remaining_seconds = (self.event_end_ms - decision) / 1_000.0
        elapsed_fraction = (
            decision - self.event_start_ms
        ) / POLYMARKET_ROUND25_CONDITION_DURATION_MS
        overlap_fraction = max(
            0.0,
            1.0 - remaining_seconds / POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS,
        )

        lag_values: list[float] = []
        source_to_log = dict(zip(timestamps, logs, strict=True))
        for lag_seconds in _LAG_SECONDS:
            prior_log = source_to_log.get(latest_timestamp - lag_seconds * 1_000)
            lag_values.extend(
                (0.0, 0.0)
                if prior_log is None
                else (latest_log - prior_log, 1.0)
            )

        slope_values: list[float] = []
        for window_seconds in _SLOPE_SECONDS:
            window_start = latest_timestamp - window_seconds * 1_000
            if window_start not in source_to_log:
                slope_values.extend((0.0, 0.0))
                continue
            start_index = bisect_right(timestamps, window_start - 1)
            selected_timestamps = timestamps[start_index:]
            selected_logs = logs[start_index:]
            if len(selected_timestamps) < 3:
                slope_values.extend((0.0, 0.0))
                continue
            slope_values.extend(
                (_slope_per_second(selected_timestamps, selected_logs), 1.0)
            )

        returns = [
            current - previous
            for previous, current in zip(logs, logs[1:], strict=False)
        ]
        overlapping_variance_rate = math.fsum(value * value for value in returns) / (
            coverage_ms / 1_000.0
        )
        grid_step_ms = POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS * 1_000
        grid_returns = [
            source_to_log[timestamp] - source_to_log[timestamp - grid_step_ms]
            for timestamp in range(
                self.event_start_ms + grid_step_ms,
                latest_timestamp + 1,
                grid_step_ms,
            )
            if timestamp in source_to_log
            and timestamp - grid_step_ms in source_to_log
        ]
        if len(grid_returns) >= 2:
            nonoverlap_elapsed = len(grid_returns) * (
                POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS
            )
            nonoverlap_variance_rate = math.fsum(
                value * value for value in grid_returns
            ) / nonoverlap_elapsed
            scale = math.sqrt(
                max(nonoverlap_variance_rate, 1e-18) * remaining_seconds
            )
            scaled_distance = math.tanh(log_distance / scale)
            nonoverlap_available = 1.0
        else:
            nonoverlap_variance_rate = 0.0
            scaled_distance = 0.0
            nonoverlap_available = 0.0

        intervals = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:], strict=False)
        ]
        values = (
            elapsed_fraction,
            remaining_seconds,
            overlap_fraction,
            log_distance,
            *lag_values,
            *slope_values,
            overlapping_variance_rate,
            nonoverlap_variance_rate,
            _lag1_autocorrelation(returns),
            scaled_distance,
            nonoverlap_available,
            float(decision - latest.source_timestamp_ms),
            float(decision - latest.received_wall_ms),
            float(latest.publisher_timestamp_ms - latest.source_timestamp_ms),
            float(latest.received_wall_ms - latest.publisher_timestamp_ms),
            float(len(ordered)),
            coverage_ms / 1_000.0,
            fmean(intervals),
            pstdev(intervals),
            float(max(intervals)),
            float(duplicate_count),
        )
        if len(values) != len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in values
        ):
            raise RuntimeError("Round 25 TWAP feature vector differs")

        chain = _EMPTY_SHA256
        for observation in ordered:
            identity = _canonical_json(
                {
                    "full_accuracy_value_e18": observation.full_accuracy_value_e18,
                    "publisher_timestamp_ms": observation.publisher_timestamp_ms,
                    "raw_frame_sha256": observation.raw_frame_sha256,
                    "received_monotonic_ns": observation.received_monotonic_ns,
                    "received_wall_ms": observation.received_wall_ms,
                    "source_timestamp_ms": observation.source_timestamp_ms,
                    "wire_schema_correction_sha256": (
                        observation.wire_schema_correction_sha256
                    ),
                }
            ).encode("ascii")
            chain = hashlib.sha256(bytes.fromhex(chain) + identity).hexdigest()
        return Round25TwapFeatureSnapshot(
            condition_id=self.condition_id,
            event_start_ms=self.event_start_ms,
            decision_time_ms=decision,
            available=True,
            reasons=(),
            values=tuple(float(value) for value in values),
            source_chain_sha256=chain,
            maximum_receipt_ms=max(item.received_wall_ms for item in ordered),
            opening_value_e18=opening.full_accuracy_value_e18,
            latest_value_e18=latest.full_accuracy_value_e18,
        )


__all__ = [
    "POLYMARKET_ROUND25_DECISION_CADENCE_MS",
    "POLYMARKET_ROUND25_TWAP_FEATURE_NAMES",
    "POLYMARKET_ROUND25_TWAP_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256",
    "POLYMARKET_ROUND25_TWAP_SYMBOL",
    "POLYMARKET_ROUND25_TWAP_TOPIC",
    "POLYMARKET_ROUND25_TWAP_WIRE_SCHEMA_CORRECTION_SHA256",
    "POLYMARKET_ROUND25_TWAP_WINDOW_SECONDS",
    "Round25TwapFeatureEngine",
    "Round25TwapFeatureSnapshot",
    "Round25TwapObservation",
]
