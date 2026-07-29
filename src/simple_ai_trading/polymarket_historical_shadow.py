"""Non-authoritative live shadow scoring for the frozen BTC historical model.

This module consumes public Binance BTC aggregate trades in memory. It has no
exchange client, credential, account, position, risk-budget, or order boundary.
Its output is research telemetry and cannot authorize or alter execution.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from threading import RLock
from typing import Mapping

import numpy as np

# Private helpers are intentional: the artifact loader pins their source hash so
# live shadow features cannot silently drift from the frozen training transform.
from .polymarket_historical_dataset import (
    FEATURE_NAMES,
    _log_returns,
    _market_vector,
)
from .polymarket_historical_model import (
    EVALUATION_SCHEMA_VERSION,
    PRETEST_SCHEMA_VERSION,
    predict_historical_candidate,
)
from .polymarket_historical_support import (
    HistoricalFeatureSupportProfile,
    load_historical_feature_support,
)


_MAX_ARTIFACT_BYTES = 256 * 1024
_MARKETS = ("spot", "perpetual")
_SOURCES = {
    "spot": "BINANCE_SPOT",
    "perpetual": "BINANCE_USD_M_FUTURES",
}
_DECISION_OFFSETS_SECONDS = frozenset({30, 60, 90, 120, 150, 180, 210, 240})
_REQUIRED_TEST_GATES = frozenset(
    {
        "calibration_slope_in_range",
        "challenger_balanced_accuracy_not_lower",
        "challenger_brier_skill_positive",
        "challenger_log_loss_skill_positive",
        "expected_calibration_error_at_most_0_05",
        "minimum_decision_rows",
        "minimum_outcomes_per_class",
        "minimum_terminal_conditions",
        "paired_log_loss_improvement_lower_positive",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Polymarket shadow artifact contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Polymarket shadow artifact contains {value}")


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load_strict_json(path: Path, *, name: str) -> Mapping[str, object]:
    if path.is_symlink():
        raise ValueError(f"{name} cannot be a symlink")
    raw = path.read_bytes()
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
    return _mapping(value, name=name)


def _verified_artifact_sha(value: Mapping[str, object], *, name: str) -> str:
    payload = dict(value)
    claimed = str(payload.pop("artifact_sha256", "")).strip().lower()
    if len(claimed) != 64 or _canonical_sha256(payload) != claimed:
        raise ValueError(f"{name} integrity failed")
    return claimed


@dataclass(frozen=True, slots=True)
class BtcAggregateTradeObservation:
    """One public BTCUSDT aggregate trade received from a read-only stream."""

    market: str
    source: str
    symbol: str
    event_time_ms: int
    received_at_ms: int
    aggregate_trade_id: int
    first_trade_id: int
    last_trade_id: int
    price: float
    quantity: float
    buyer_is_maker: bool

    def __post_init__(self) -> None:
        market = str(self.market or "").strip().lower()
        source = str(self.source or "").strip().upper()
        symbol = str(self.symbol or "").strip().upper()
        if market not in _MARKETS or source != _SOURCES.get(market):
            raise ValueError("BTC aggregate-trade source identity is invalid")
        if symbol != "BTCUSDT":
            raise ValueError("Polymarket shadow aggregate trades are BTCUSDT-only")
        event_time = int(self.event_time_ms)
        received_at = int(self.received_at_ms)
        aggregate_id = int(self.aggregate_trade_id)
        first_id = int(self.first_trade_id)
        last_id = int(self.last_trade_id)
        if (
            event_time <= 0
            or received_at <= 0
            or received_at < event_time - 250
            or aggregate_id < 0
            or first_id < 0
            or last_id < first_id
        ):
            raise ValueError("BTC aggregate-trade chronology is invalid")
        price = float(self.price)
        quantity = float(self.quantity)
        if (
            not math.isfinite(price)
            or not math.isfinite(quantity)
            or price <= 0.0
            or quantity <= 0.0
            or not math.isfinite(price * quantity)
        ):
            raise ValueError("BTC aggregate-trade economics are invalid")
        if type(self.buyer_is_maker) is not bool:
            raise ValueError("buyer_is_maker must be a boolean")
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "event_time_ms", event_time)
        object.__setattr__(self, "received_at_ms", received_at)
        object.__setattr__(self, "aggregate_trade_id", aggregate_id)
        object.__setattr__(self, "first_trade_id", first_id)
        object.__setattr__(self, "last_trade_id", last_id)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "quantity", quantity)

    @property
    def constituent_trade_count(self) -> int:
        return self.last_trade_id - self.first_trade_id + 1

    @property
    def quote_notional(self) -> float:
        return self.price * self.quantity

    def identity(self) -> tuple[object, ...]:
        return (
            self.market,
            self.source,
            self.symbol,
            self.event_time_ms,
            self.received_at_ms,
            self.aggregate_trade_id,
            self.first_trade_id,
            self.last_trade_id,
            self.price,
            self.quantity,
            self.buyer_is_maker,
        )


class PolymarketShadowDataUnavailable(ValueError):
    """Expected fail-closed data-quality abstention."""

    def __init__(self, code: str) -> None:
        normalized = str(code or "").strip().lower()
        if not normalized or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in normalized
        ):
            raise ValueError("shadow data-unavailable code is invalid")
        self.code = normalized
        super().__init__(normalized)


class PolymarketBtcFlowBuffer:
    """Bounded in-memory causal aggregate-trade buffer for shadow inference."""

    def __init__(
        self,
        *,
        retention_seconds: int = 300,
        maximum_source_staleness_ms: int = 1_500,
        maximum_decision_delay_ms: int = 5_000,
    ) -> None:
        self.retention_seconds = int(retention_seconds)
        self.maximum_source_staleness_ms = int(maximum_source_staleness_ms)
        self.maximum_decision_delay_ms = int(maximum_decision_delay_ms)
        if not 120 <= self.retention_seconds <= 3_600:
            raise ValueError("shadow flow retention must lie in [120, 3600] seconds")
        if not 100 <= self.maximum_source_staleness_ms <= 10_000:
            raise ValueError("shadow source staleness must lie in [100, 10000] ms")
        if not 0 <= self.maximum_decision_delay_ms <= 30_000:
            raise ValueError("shadow decision delay must lie in [0, 30000] ms")
        self._events: dict[str, deque[BtcAggregateTradeObservation]] = {
            market: deque() for market in _MARKETS
        }
        self._identities: dict[str, dict[int, tuple[object, ...]]] = {
            market: {} for market in _MARKETS
        }
        self._last: dict[str, BtcAggregateTradeObservation | None] = {
            market: None for market in _MARKETS
        }
        self._faults: set[str] = set()
        self._lock = RLock()

    @property
    def faults(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._faults))

    def reset_market(self, market: str) -> None:
        """Discard one feed epoch so reconnect gaps require a full warmup."""

        normalized = str(market or "").strip().lower()
        if normalized not in _MARKETS:
            raise ValueError("shadow flow market is invalid")
        with self._lock:
            self._events[normalized].clear()
            self._identities[normalized].clear()
            self._last[normalized] = None
            prefix = f"{normalized}_"
            self._faults = {
                fault for fault in self._faults if not fault.startswith(prefix)
            }

    def ingest(self, observation: BtcAggregateTradeObservation) -> bool:
        if not isinstance(observation, BtcAggregateTradeObservation):
            raise TypeError("observation must be BtcAggregateTradeObservation")
        with self._lock:
            market = observation.market
            identity = observation.identity()
            prior_identity = self._identities[market].get(
                observation.aggregate_trade_id
            )
            if prior_identity is not None:
                if prior_identity != identity:
                    self._faults.add(f"{market}_duplicate_identity_mismatch")
                    raise PolymarketShadowDataUnavailable(
                        f"{market}_duplicate_identity_mismatch"
                    )
                return False
            previous = self._last[market]
            if previous is not None and (
                observation.aggregate_trade_id <= previous.aggregate_trade_id
                or observation.event_time_ms < previous.event_time_ms
                or observation.received_at_ms < previous.received_at_ms
                or observation.first_trade_id <= previous.last_trade_id
            ):
                self._faults.add(f"{market}_stream_regression")
                raise PolymarketShadowDataUnavailable(
                    f"{market}_stream_regression"
                )
            events = self._events[market]
            events.append(observation)
            self._identities[market][observation.aggregate_trade_id] = identity
            self._last[market] = observation
            cutoff = observation.event_time_ms - self.retention_seconds * 1_000
            while events and events[0].event_time_ms < cutoff:
                removed = events.popleft()
                self._identities[market].pop(removed.aggregate_trade_id, None)
            return True

    def _market_flow(
        self,
        *,
        market: str,
        decision_time_ms: int,
        second_count: int = 61,
    ) -> Mapping[str, np.ndarray]:
        count = int(second_count)
        if not 2 <= count <= self.retention_seconds:
            raise ValueError(
                "shadow flow second count exceeds bounded retention"
            )
        causal = tuple(
            observation
            for observation in self._events[market]
            if observation.event_time_ms < decision_time_ms
            and observation.received_at_ms <= decision_time_ms
        )
        if not causal:
            raise PolymarketShadowDataUnavailable(f"{market}_flow_missing")
        latest = causal[-1]
        if (
            decision_time_ms - latest.event_time_ms
            > self.maximum_source_staleness_ms
            or decision_time_ms - latest.received_at_ms
            > self.maximum_source_staleness_ms
        ):
            raise PolymarketShadowDataUnavailable(f"{market}_flow_stale")

        start_ms = decision_time_ms - count * 1_000
        prior = next(
            (
                observation
                for observation in reversed(causal)
                if observation.event_time_ms < start_ms
            ),
            None,
        )
        if prior is None:
            raise PolymarketShadowDataUnavailable(f"{market}_lookback_incomplete")

        close = np.full(count, np.nan, dtype=np.float64)
        quote_volume = np.zeros(count, dtype=np.float64)
        buy_quote = np.zeros(count, dtype=np.float64)
        sell_quote = np.zeros(count, dtype=np.float64)
        aggregate_count = np.zeros(count, dtype=np.float64)
        constituent_count = np.zeros(count, dtype=np.float64)
        maximum_quote = np.zeros(count, dtype=np.float64)
        squared_quote = np.zeros(count, dtype=np.float64)
        last_trade_age = np.zeros(count, dtype=np.float64)
        for observation in causal:
            if observation.event_time_ms < start_ms:
                continue
            index = (observation.event_time_ms - start_ms) // 1_000
            if not 0 <= index < count:
                continue
            quote = observation.quote_notional
            close[index] = observation.price
            quote_volume[index] += quote
            if observation.buyer_is_maker:
                sell_quote[index] += quote
            else:
                buy_quote[index] += quote
            aggregate_count[index] += 1.0
            constituent_count[index] += observation.constituent_trade_count
            maximum_quote[index] = max(maximum_quote[index], quote)
            squared_quote[index] += quote * quote

        last_close = prior.price
        prior_second_ms = prior.event_time_ms // 1_000 * 1_000
        age = max(1, (start_ms - prior_second_ms) // 1_000)
        for index in range(count):
            if aggregate_count[index] > 0:
                last_close = float(close[index])
                age = 0
            else:
                close[index] = last_close
                age += int(index > 0)
            last_trade_age[index] = float(age)
        if np.any(~np.isfinite(close)) or np.any(close <= 0.0):
            raise PolymarketShadowDataUnavailable(f"{market}_close_invalid")
        if np.any(np.abs(quote_volume - buy_quote - sell_quote) > 1e-6):
            raise RuntimeError("shadow aggregate-trade quote flow does not reconcile")
        return {
            f"{market}_close": close,
            f"{market}_quote_volume": quote_volume,
            f"{market}_aggressive_buy_quote": buy_quote,
            f"{market}_aggressive_sell_quote": sell_quote,
            f"{market}_aggregate_count": aggregate_count,
            f"{market}_constituent_trade_count": constituent_count,
            f"{market}_maximum_aggregate_quote": maximum_quote,
            f"{market}_squared_aggregate_quote_sum": squared_quote,
            f"{market}_last_trade_age_seconds": last_trade_age,
        }

    def causal_flow_snapshot(
        self,
        *,
        decision_time_ms: int,
        observed_at_ms: int,
        second_count: int,
    ) -> Mapping[str, np.ndarray]:
        """Return one immutable, cross-feed causal snapshot for model scoring."""

        decision_time = int(decision_time_ms)
        observed_at = int(observed_at_ms)
        count = int(second_count)
        if observed_at < decision_time:
            raise PolymarketShadowDataUnavailable("decision_not_reached")
        if observed_at - decision_time > self.maximum_decision_delay_ms:
            raise PolymarketShadowDataUnavailable("decision_stale")
        with self._lock:
            if self._faults:
                raise PolymarketShadowDataUnavailable("stream_fault_latched")
            values: dict[str, np.ndarray] = {
                "second_ms": (
                    decision_time
                    - count * 1_000
                    + np.arange(count, dtype=np.int64) * 1_000
                )
            }
            for market in _MARKETS:
                values.update(
                    self._market_flow(
                        market=market,
                        decision_time_ms=decision_time,
                        second_count=count,
                    )
                )
            return {
                name: np.asarray(array).copy()
                for name, array in values.items()
            }

    def feature_vector(
        self,
        *,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
    ) -> np.ndarray:
        event_start = int(event_start_ms)
        decision_time = int(decision_time_ms)
        observed_at = int(observed_at_ms)
        offset_seconds, remainder = divmod(decision_time - event_start, 1_000)
        if (
            event_start <= 0
            or remainder != 0
            or offset_seconds not in _DECISION_OFFSETS_SECONDS
        ):
            raise PolymarketShadowDataUnavailable("decision_offset_invalid")
        if observed_at < decision_time:
            raise PolymarketShadowDataUnavailable("decision_not_reached")
        if observed_at - decision_time > self.maximum_decision_delay_ms:
            raise PolymarketShadowDataUnavailable("decision_stale")
        values = dict(
            self.causal_flow_snapshot(
                decision_time_ms=decision_time,
                observed_at_ms=observed_at,
                second_count=61,
            )
        )
        for market in _MARKETS:
            values[f"{market}_log_return"] = _log_returns(
                values[f"{market}_close"]
            )
        vector: list[float] = []
        for market in _MARKETS:
            vector.extend(_market_vector(values, market=market, end_index=60))
        spot = values["spot_close"]
        perpetual = values["perpetual_close"]
        basis = np.log(perpetual / spot) * 10_000.0
        vector.append(float(basis[60]))
        vector.extend(float(basis[60] - basis[60 - horizon]) for horizon in (5, 15, 30, 60))
        vector.extend(
            float(
                math.log(spot[60] / spot[60 - horizon])
                - math.log(perpetual[60] / perpetual[60 - horizon])
            )
            for horizon in (1, 5, 15, 30, 60)
        )
        seconds_of_day = (decision_time % 86_400_000) / 1_000.0
        phase = 2.0 * math.pi * seconds_of_day / 86_400.0
        vector.extend(
            (
                math.sin(phase),
                math.cos(phase),
                offset_seconds / 300.0,
                (300 - offset_seconds) / 300.0,
            )
        )
        output = np.asarray(vector, dtype=np.float32)
        if output.shape != (len(FEATURE_NAMES),) or np.any(~np.isfinite(output)):
            raise RuntimeError("shadow historical feature vector is invalid")
        return output


@dataclass(frozen=True, slots=True)
class VerifiedHistoricalShadowPredictor:
    candidate: Mapping[str, object]
    candidate_id: str
    pretest_artifact_sha256: str
    evaluation_artifact_sha256: str
    dataset_sha256: str
    support_profile: HistoricalFeatureSupportProfile
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.trading_authority
            or not isinstance(
                self.support_profile,
                HistoricalFeatureSupportProfile,
            )
            or self.support_profile.pretest_artifact_sha256
            != self.pretest_artifact_sha256
            or self.support_profile.dataset_sha256 != self.dataset_sha256
        ):
            raise ValueError("historical shadow predictor cannot have trading authority")

    def predict_up_probability(self, features: np.ndarray) -> float:
        vector = np.asarray(features, dtype=np.float32)
        if vector.shape != (len(FEATURE_NAMES),) or np.any(~np.isfinite(vector)):
            raise ValueError("historical shadow feature vector is invalid")
        probability = float(
            predict_historical_candidate(self.candidate, vector.reshape(1, -1))[0]
        )
        if not math.isfinite(probability) or not 0.0 < probability < 1.0:
            raise RuntimeError("historical shadow prediction is invalid")
        return probability


@dataclass(frozen=True, slots=True)
class PolymarketHistoricalShadowDecision:
    status: str
    reason: str
    event_start_ms: int
    decision_time_ms: int
    observed_at_ms: int
    probability_up: float | None
    candidate_id: str
    pretest_artifact_sha256: str
    evaluation_artifact_sha256: str
    support_profile_sha256: str = ""
    outside_training_range_count: int = 0
    extreme_outlier_count: int = 0
    trading_authority: bool = False
    grants_execution_authority: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"observed", "abstain"}:
            raise ValueError("historical shadow status is invalid")
        if self.status == "observed":
            if self.probability_up is None or not 0.0 < self.probability_up < 1.0:
                raise ValueError("observed historical shadow probability is invalid")
            if self.reason:
                raise ValueError("observed historical shadow decision has a reason")
        elif self.probability_up is not None or not self.reason:
            raise ValueError("abstained historical shadow decision is malformed")
        if self.trading_authority or self.grants_execution_authority:
            raise ValueError("historical shadow decision cannot grant authority")
        if (
            self.outside_training_range_count < 0
            or self.extreme_outlier_count < 0
            or (
                self.support_profile_sha256
                and len(self.support_profile_sha256) != 64
            )
        ):
            raise ValueError("historical shadow support evidence is invalid")


class PolymarketHistoricalShadowScorer:
    """Score frozen model telemetry without affecting Polymarket execution."""

    def __init__(
        self,
        *,
        predictor: VerifiedHistoricalShadowPredictor,
        flow: PolymarketBtcFlowBuffer,
    ) -> None:
        if not isinstance(predictor, VerifiedHistoricalShadowPredictor):
            raise TypeError("predictor must be VerifiedHistoricalShadowPredictor")
        if not isinstance(flow, PolymarketBtcFlowBuffer):
            raise TypeError("flow must be PolymarketBtcFlowBuffer")
        self.predictor = predictor
        self.flow = flow

    def evaluate(
        self,
        *,
        event_start_ms: int,
        decision_time_ms: int,
        observed_at_ms: int,
    ) -> PolymarketHistoricalShadowDecision:
        try:
            features = self.flow.feature_vector(
                event_start_ms=event_start_ms,
                decision_time_ms=decision_time_ms,
                observed_at_ms=observed_at_ms,
            )
        except PolymarketShadowDataUnavailable as exc:
            return PolymarketHistoricalShadowDecision(
                status="abstain",
                reason=exc.code,
                event_start_ms=int(event_start_ms),
                decision_time_ms=int(decision_time_ms),
                observed_at_ms=int(observed_at_ms),
                probability_up=None,
                candidate_id=self.predictor.candidate_id,
                pretest_artifact_sha256=self.predictor.pretest_artifact_sha256,
                evaluation_artifact_sha256=(
                    self.predictor.evaluation_artifact_sha256
                ),
                support_profile_sha256=(
                    self.predictor.support_profile.artifact_sha256
                ),
            )
        support = self.predictor.support_profile.assess(features)
        if support.status == "abstain":
            return PolymarketHistoricalShadowDecision(
                status="abstain",
                reason="feature_support_out_of_distribution",
                event_start_ms=int(event_start_ms),
                decision_time_ms=int(decision_time_ms),
                observed_at_ms=int(observed_at_ms),
                probability_up=None,
                candidate_id=self.predictor.candidate_id,
                pretest_artifact_sha256=self.predictor.pretest_artifact_sha256,
                evaluation_artifact_sha256=(
                    self.predictor.evaluation_artifact_sha256
                ),
                support_profile_sha256=support.profile_sha256,
                outside_training_range_count=(
                    support.outside_training_range_count
                ),
                extreme_outlier_count=support.extreme_outlier_count,
            )
        return PolymarketHistoricalShadowDecision(
            status="observed",
            reason="",
            event_start_ms=int(event_start_ms),
            decision_time_ms=int(decision_time_ms),
            observed_at_ms=int(observed_at_ms),
            probability_up=self.predictor.predict_up_probability(features),
            candidate_id=self.predictor.candidate_id,
            pretest_artifact_sha256=self.predictor.pretest_artifact_sha256,
            evaluation_artifact_sha256=self.predictor.evaluation_artifact_sha256,
            support_profile_sha256=support.profile_sha256,
            outside_training_range_count=(
                support.outside_training_range_count
            ),
            extreme_outlier_count=support.extreme_outlier_count,
        )


def load_verified_historical_shadow_predictor(
    *,
    pretest_path: str | Path,
    evaluation_path: str | Path,
    support_path: str | Path,
) -> VerifiedHistoricalShadowPredictor:
    pretest = _load_strict_json(Path(pretest_path), name="historical pretest")
    evaluation = _load_strict_json(
        Path(evaluation_path),
        name="historical evaluation",
    )
    pretest_sha = _verified_artifact_sha(pretest, name="historical pretest")
    evaluation_sha = _verified_artifact_sha(
        evaluation,
        name="historical evaluation",
    )
    if (
        pretest.get("schema_version") != PRETEST_SCHEMA_VERSION
        or evaluation.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or evaluation.get("pretest_artifact_sha256") != pretest_sha
        or evaluation.get("contract_sha256") != pretest.get("contract_sha256")
        or evaluation.get("dataset_sha256") != pretest.get("dataset_sha256")
    ):
        raise ValueError("historical shadow evidence identity differs")
    feature_names = pretest.get("feature_names")
    if (
        not isinstance(feature_names, list)
        or tuple(feature_names) != FEATURE_NAMES
        or pretest.get("feature_names_sha256") != _canonical_sha256(FEATURE_NAMES)
    ):
        raise ValueError("historical shadow feature contract differs")
    implementation = _mapping(
        pretest.get("implementation_sha256"),
        name="historical implementation",
    )
    source_root = Path(__file__).parent
    expected_implementation = {
        "model": _file_sha256(source_root / "polymarket_historical_model.py"),
        "dataset": _file_sha256(source_root / "polymarket_historical_dataset.py"),
        "screen": _file_sha256(source_root / "polymarket_historical_screen.py"),
    }
    if dict(implementation) != expected_implementation:
        raise ValueError("historical shadow implementation hash differs")

    gates = _mapping(evaluation.get("gates"), name="historical evaluation gates")
    scope = _mapping(evaluation.get("scope"), name="historical evaluation scope")
    challenger_id = str(pretest.get("best_challenger_id") or "")
    if (
        set(gates) != _REQUIRED_TEST_GATES
        or any(value is not True for value in gates.values())
        or evaluation.get("accepted_predictive_edge") is not True
        or evaluation.get("best_challenger_id") != challenger_id
        or scope.get("venue") != "polymarket"
        or scope.get("asset") != "BTC"
        or scope.get("market_variant") != "fiveminute"
        or scope.get("predictive_screen_only") is not True
        or scope.get("execution_or_profitability_claim") is not False
    ):
        raise ValueError("historical shadow predictive gates are not satisfied")
    candidates = pretest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("historical shadow candidates are malformed")
    matches = [
        _mapping(candidate, name="historical challenger")
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == challenger_id
    ]
    if len(matches) != 1:
        raise ValueError("historical shadow challenger identity differs")
    challenger = matches[0]
    _verified_artifact_sha(challenger, name="historical challenger")
    model = _mapping(challenger.get("model"), name="historical challenger model")
    model_string = str(model.get("model_string") or "")
    if (
        challenger.get("kind") != "challenger"
        or challenger.get("family") != "binance_shallow_lightgbm"
        or model.get("type") != "lightgbm"
        or hashlib.sha256(model_string.encode("utf-8")).hexdigest()
        != model.get("model_sha256")
    ):
        raise ValueError("historical shadow challenger model differs")
    dataset_sha = str(pretest["dataset_sha256"])
    support = load_historical_feature_support(
        support_path,
        expected_pretest_artifact_sha256=pretest_sha,
        expected_dataset_sha256=dataset_sha,
    )
    return VerifiedHistoricalShadowPredictor(
        candidate=dict(challenger),
        candidate_id=challenger_id,
        pretest_artifact_sha256=pretest_sha,
        evaluation_artifact_sha256=evaluation_sha,
        dataset_sha256=dataset_sha,
        support_profile=support,
    )


__all__ = [
    "BtcAggregateTradeObservation",
    "PolymarketBtcFlowBuffer",
    "PolymarketHistoricalShadowDecision",
    "PolymarketHistoricalShadowScorer",
    "PolymarketShadowDataUnavailable",
    "VerifiedHistoricalShadowPredictor",
    "load_verified_historical_shadow_predictor",
]
