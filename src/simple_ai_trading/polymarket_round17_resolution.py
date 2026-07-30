"""Official, development-only resolution evidence for Polymarket Round 17."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
import time

from .polymarket import PolymarketFiveMinuteMarket, PolymarketPublicClient
from .polymarket_replay import PolymarketResolutionEvidence
from .polymarket_resolution import validate_official_resolution
from .polymarket_round17_cohort import (
    Round17CohortCondition,
    Round17CohortManifest,
    Round17CohortPlan,
    Round17ConditionLabel,
    build_round17_cohort_condition_label,
)


POLYMARKET_ROUND17_DEVELOPMENT_RESOLUTION_CONTRACT_SHA256 = (
    "790f7af066b6e86a5c0f2fbe97e488ca275ed21a6c39d64574709cc8c95a124f"
)
POLYMARKET_ROUND17_RESOLUTION_OBSERVATION_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-resolution-observation-v1"
)
POLYMARKET_ROUND17_TEST_RESOLUTION_OBSERVATION_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-test-resolution-observation-v1"
)
POLYMARKET_ROUND17_TEST_RESOLUTION_CONTRACT_SHA256 = (
    "e7e4471cd71882beee8837ac6251e52d407c5be57f716d703a816f70b9abf5dd"
)
POLYMARKET_ROUND17_RESOLUTION_ACQUISITION_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-development-resolution-acquisition-v1"
)
POLYMARKET_ROUND17_TEST_RESOLUTION_ACQUISITION_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-test-resolution-acquisition-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GAMMA_BATCH_SIZE = 100


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 resolution payload contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 resolution payload contains {value}")


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


def _canonical_mapping(value: Mapping[str, object], *, name: str) -> tuple[str, str]:
    try:
        canonical = _canonical_json(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not canonical JSON") from exc
    return canonical, hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _load_canonical_mapping(value: str, *, name: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not isinstance(parsed, Mapping) or _canonical_json(parsed) != value:
        raise ValueError(f"{name} is not a canonical object")
    return parsed


@dataclass(frozen=True, slots=True)
class Round17ResolutionObservation:
    source_run_id: str
    condition_id: str
    event_start_ms: int
    observed_wall_ms: int
    observed_monotonic_ns: int
    winning_asset_id: str
    winning_outcome: str
    market_identity_sha256: str
    clob_payload_json: str
    clob_payload_sha256: str
    gamma_payload_json: str
    gamma_payload_sha256: str
    source_partition: str = "development"
    observation_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        test_partition = self.source_partition == "test"
        payload = {
            "schema_version": (
                POLYMARKET_ROUND17_TEST_RESOLUTION_OBSERVATION_SCHEMA_VERSION
                if test_partition
                else POLYMARKET_ROUND17_RESOLUTION_OBSERVATION_SCHEMA_VERSION
            ),
            "contract_sha256": (
                POLYMARKET_ROUND17_TEST_RESOLUTION_CONTRACT_SHA256
                if test_partition
                else POLYMARKET_ROUND17_DEVELOPMENT_RESOLUTION_CONTRACT_SHA256
            ),
            "source_run_id": self.source_run_id,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "observed_wall_ms": self.observed_wall_ms,
            "observed_monotonic_ns": self.observed_monotonic_ns,
            "winning_asset_id": self.winning_asset_id,
            "winning_outcome": self.winning_outcome,
            "market_identity_sha256": self.market_identity_sha256,
            "clob_payload_json": self.clob_payload_json,
            "clob_payload_sha256": self.clob_payload_sha256,
            "gamma_payload_json": self.gamma_payload_json,
            "gamma_payload_sha256": self.gamma_payload_sha256,
        }
        if test_partition:
            payload["source_partition"] = "test"
        return payload

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "observation_sha256": self.observation_sha256,
        }

    def validated(
        self,
        plan: Round17CohortPlan,
        reference: Round17CohortCondition,
        market: PolymarketFiveMinuteMarket,
    ) -> Round17ResolutionObservation:
        selected = reference.validated(plan)
        clob = _load_canonical_mapping(
            self.clob_payload_json,
            name="Round 17 CLOB resolution payload",
        )
        gamma = _load_canonical_mapping(
            self.gamma_payload_json,
            name="Round 17 Gamma resolution payload",
        )
        winner = validate_official_resolution(
            market,
            clob,
            gamma,
            observed_wall_ms=self.observed_wall_ms,
        )
        expected_role = (
            "test" if self.source_partition == "test" else reference.role
        )
        if (
            self.source_partition not in {"development", "test"}
            or reference.role != expected_role
            or (
                self.source_partition == "development"
                and reference.role == "test"
            )
            or self.source_run_id != selected.source_run_id
            or self.condition_id != selected.condition_id
            or self.event_start_ms != selected.event_start_ms
            or market.asset != "BTC"
            or market.condition_id != selected.condition_id
            or market.event_start_ms != selected.event_start_ms
            or market.end_ms != selected.event_end_ms
            or self.observed_wall_ms < market.end_ms
            or self.observed_monotonic_ns < 0
            or winner != (self.winning_asset_id, self.winning_outcome)
            or self.market_identity_sha256 != _canonical_sha256(market.asdict())
            or self.clob_payload_sha256
            != hashlib.sha256(self.clob_payload_json.encode("ascii")).hexdigest()
            or self.gamma_payload_sha256
            != hashlib.sha256(self.gamma_payload_json.encode("ascii")).hexdigest()
            or _SHA256.fullmatch(self.observation_sha256) is None
            or self.observation_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 resolution observation integrity differs")
        return self

    def resolution_evidence(self) -> PolymarketResolutionEvidence:
        return PolymarketResolutionEvidence(
            run_id=self.source_run_id,
            event_id=self.observation_sha256,
            condition_id=self.condition_id,
            winning_asset_id=self.winning_asset_id,
            winning_outcome=self.winning_outcome,
            resolved_at_ms=self.observed_wall_ms,
            received_wall_ms=self.observed_wall_ms,
            received_monotonic_ns=self.observed_monotonic_ns,
            event_sha256=self.observation_sha256,
            source="clob_gamma_crosscheck",
        )


def _build_observation(
    *,
    reference: Round17CohortCondition,
    market: PolymarketFiveMinuteMarket,
    clob: Mapping[str, object],
    gamma: Mapping[str, object],
    observed_wall_ms: int,
    observed_monotonic_ns: int,
    source_partition: str = "development",
) -> Round17ResolutionObservation | None:
    winner = validate_official_resolution(
        market,
        clob,
        gamma,
        observed_wall_ms=observed_wall_ms,
    )
    if winner is None:
        return None
    clob_json, clob_sha256 = _canonical_mapping(
        clob,
        name="Round 17 CLOB resolution payload",
    )
    gamma_json, gamma_sha256 = _canonical_mapping(
        gamma,
        name="Round 17 Gamma resolution payload",
    )
    provisional = Round17ResolutionObservation(
        source_run_id=reference.source_run_id,
        condition_id=reference.condition_id,
        event_start_ms=reference.event_start_ms,
        observed_wall_ms=int(observed_wall_ms),
        observed_monotonic_ns=int(observed_monotonic_ns),
        winning_asset_id=winner[0],
        winning_outcome=winner[1],
        market_identity_sha256=_canonical_sha256(market.asdict()),
        clob_payload_json=clob_json,
        clob_payload_sha256=clob_sha256,
        gamma_payload_json=gamma_json,
        gamma_payload_sha256=gamma_sha256,
        source_partition=source_partition,
    )
    return replace(
        provisional,
        observation_sha256=_canonical_sha256(provisional.identity_payload()),
    )


@dataclass(frozen=True, slots=True)
class Round17DevelopmentResolutionAcquisition:
    cohort_manifest_sha256: str
    observations: tuple[Round17ResolutionObservation, ...]
    pending_condition_ids: tuple[str, ...]
    gamma_batch_request_count: int
    clob_market_request_count: int
    acquisition_sha256: str = ""

    @property
    def complete(self) -> bool:
        return not self.pending_condition_ids

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND17_RESOLUTION_ACQUISITION_SCHEMA_VERSION
            ),
            "contract_sha256": (
                POLYMARKET_ROUND17_DEVELOPMENT_RESOLUTION_CONTRACT_SHA256
            ),
            "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "observations": [item.asdict() for item in self.observations],
            "pending_condition_ids": list(self.pending_condition_ids),
            "gamma_batch_request_count": self.gamma_batch_request_count,
            "clob_market_request_count": self.clob_market_request_count,
            "development_outcomes_consulted": True,
            "test_features_accessed": False,
            "test_targets_accessed": False,
            "model_scores_accessed": False,
            "execution_scores_accessed": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "complete": self.complete,
            "acquisition_sha256": self.acquisition_sha256,
        }

    def validated(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
        markets: Mapping[str, PolymarketFiveMinuteMarket],
    ) -> Round17DevelopmentResolutionAcquisition:
        selected_cohort = cohort.validated(plan)
        references = {item.condition_id: item for item in selected_cohort.conditions}
        selected_markets = dict(markets)
        observed_ids = tuple(item.condition_id for item in self.observations)
        pending = self.pending_condition_ids
        if (
            self.cohort_manifest_sha256 != selected_cohort.manifest_sha256
            or set(selected_markets) != set(references)
            or any(
                key != market.condition_id for key, market in selected_markets.items()
            )
            or observed_ids != tuple(sorted(observed_ids))
            or len(observed_ids) != len(set(observed_ids))
            or pending != tuple(sorted(set(pending)))
            or set(observed_ids) & set(pending)
            or set(observed_ids) | set(pending) != set(references)
            or self.gamma_batch_request_count
            != math.ceil(len(references) / _GAMMA_BATCH_SIZE)
            or self.clob_market_request_count != len(references)
            or _SHA256.fullmatch(self.acquisition_sha256) is None
            or self.acquisition_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 resolution acquisition integrity differs")
        for observation in self.observations:
            observation.validated(
                plan,
                references[observation.condition_id],
                selected_markets[observation.condition_id],
            )
        return self

    def labels(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
        markets: Mapping[str, PolymarketFiveMinuteMarket],
    ) -> tuple[Round17ConditionLabel, ...]:
        selected = self.validated(plan, cohort, markets)
        if not selected.complete:
            raise RuntimeError("Round 17 development resolutions are incomplete")
        references = {item.condition_id: item for item in cohort.conditions}
        return tuple(
            build_round17_cohort_condition_label(
                plan,
                references[item.condition_id],
                markets[item.condition_id],
                item.resolution_evidence(),
            )
            for item in selected.observations
        )


@dataclass(frozen=True, slots=True)
class Round17TestResolutionAcquisition:
    claim_sha256: str
    test_access_sha256: str
    cohort_manifest_sha256: str
    observations: tuple[Round17ResolutionObservation, ...]
    pending_condition_ids: tuple[str, ...]
    gamma_batch_request_count: int
    clob_market_request_count: int
    acquisition_sha256: str = ""

    @property
    def complete(self) -> bool:
        return not self.pending_condition_ids

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                POLYMARKET_ROUND17_TEST_RESOLUTION_ACQUISITION_SCHEMA_VERSION
            ),
            "contract_sha256": (
                POLYMARKET_ROUND17_TEST_RESOLUTION_CONTRACT_SHA256
            ),
            "claim_sha256": self.claim_sha256,
            "test_access_sha256": self.test_access_sha256,
            "cohort_manifest_sha256": self.cohort_manifest_sha256,
            "observations": [item.asdict() for item in self.observations],
            "pending_condition_ids": list(self.pending_condition_ids),
            "gamma_batch_request_count": self.gamma_batch_request_count,
            "clob_market_request_count": self.clob_market_request_count,
            "test_features_accessed": True,
            "test_targets_accessed": True,
            "model_scores_accessed": False,
            "execution_scores_accessed": False,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "complete": self.complete,
            "acquisition_sha256": self.acquisition_sha256,
        }

    def validated(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
        markets: Mapping[str, PolymarketFiveMinuteMarket],
    ) -> Round17TestResolutionAcquisition:
        selected_cohort = cohort.validated(plan)
        references = {item.condition_id: item for item in selected_cohort.conditions}
        selected_markets = dict(markets)
        observed_ids = tuple(item.condition_id for item in self.observations)
        pending = self.pending_condition_ids
        minimum_gamma_requests = math.ceil(
            len(references) / _GAMMA_BATCH_SIZE
        )
        if (
            _SHA256.fullmatch(self.claim_sha256) is None
            or _SHA256.fullmatch(self.test_access_sha256) is None
            or any(item.role != "test" for item in selected_cohort.conditions)
            or self.cohort_manifest_sha256 != selected_cohort.manifest_sha256
            or set(selected_markets) != set(references)
            or any(
                key != market.condition_id for key, market in selected_markets.items()
            )
            or observed_ids != tuple(sorted(observed_ids))
            or len(observed_ids) != len(set(observed_ids))
            or pending != tuple(sorted(set(pending)))
            or set(observed_ids) & set(pending)
            or set(observed_ids) | set(pending) != set(references)
            or self.gamma_batch_request_count < minimum_gamma_requests
            or self.clob_market_request_count < len(observed_ids)
            or _SHA256.fullmatch(self.acquisition_sha256) is None
            or self.acquisition_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 test resolution acquisition integrity differs")
        for observation in self.observations:
            observation.validated(
                plan,
                references[observation.condition_id],
                selected_markets[observation.condition_id],
            )
        return self

    def labels(
        self,
        plan: Round17CohortPlan,
        cohort: Round17CohortManifest,
        markets: Mapping[str, PolymarketFiveMinuteMarket],
    ) -> tuple[Round17ConditionLabel, ...]:
        selected = self.validated(plan, cohort, markets)
        if not selected.complete:
            raise RuntimeError("Round 17 test resolutions are incomplete")
        references = {item.condition_id: item for item in cohort.conditions}
        return tuple(
            build_round17_cohort_condition_label(
                plan,
                references[item.condition_id],
                markets[item.condition_id],
                item.resolution_evidence(),
            )
            for item in selected.observations
        )


def acquire_round17_development_resolutions(
    plan: Round17CohortPlan,
    cohort: Round17CohortManifest,
    markets: Mapping[str, PolymarketFiveMinuteMarket],
    *,
    client: PolymarketPublicClient | None = None,
    wall_clock_ms: Callable[[], int] | None = None,
    monotonic_clock_ns: Callable[[], int] | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> Round17DevelopmentResolutionAcquisition:
    """Acquire only frozen development outcomes from official public endpoints."""

    selected_cohort = cohort.validated(plan)
    references = selected_cohort.conditions
    selected_markets = dict(markets)
    if (
        set(selected_markets) != {item.condition_id for item in references}
        or any(key != market.condition_id for key, market in selected_markets.items())
        or len({market.market_id for market in selected_markets.values()})
        != len(selected_markets)
    ):
        raise ValueError("Round 17 resolution market identities differ")
    wall = wall_clock_ms or (lambda: time.time_ns() // 1_000_000)
    monotonic = monotonic_clock_ns or time.monotonic_ns
    if int(wall()) < max(item.event_end_ms for item in references):
        raise RuntimeError("Round 17 development markets have not all ended")
    public = client or PolymarketPublicClient()
    gamma_by_market_id: dict[str, Mapping[str, object]] = {}
    gamma_batch_count = 0
    for start in range(0, len(references), _GAMMA_BATCH_SIZE):
        batch = references[start : start + _GAMMA_BATCH_SIZE]
        market_ids = tuple(
            selected_markets[item.condition_id].market_id for item in batch
        )
        gamma = public.gamma_markets(market_ids)
        if set(gamma) != set(market_ids):
            raise ValueError("Round 17 Gamma batch coverage differs")
        gamma_by_market_id.update(gamma)
        gamma_batch_count += 1
        if progress is not None:
            progress(
                {
                    "phase": "resolution_gamma",
                    "completed_batches": gamma_batch_count,
                    "total_batches": math.ceil(len(references) / _GAMMA_BATCH_SIZE),
                }
            )
    observations: list[Round17ResolutionObservation] = []
    pending: list[str] = []
    for count, reference in enumerate(references, start=1):
        market = selected_markets[reference.condition_id]
        clob = public.clob_market(reference.condition_id)
        observed_wall_ms = int(wall())
        observed_monotonic_ns = int(monotonic())
        observation = _build_observation(
            reference=reference,
            market=market,
            clob=clob,
            gamma=gamma_by_market_id[market.market_id],
            observed_wall_ms=observed_wall_ms,
            observed_monotonic_ns=observed_monotonic_ns,
        )
        if observation is None:
            pending.append(reference.condition_id)
        else:
            observations.append(observation.validated(plan, reference, market))
        if progress is not None and (
            count == 1 or count % 25 == 0 or count == len(references)
        ):
            progress(
                {
                    "phase": "resolution_clob",
                    "completed_conditions": count,
                    "total_conditions": len(references),
                    "resolved_conditions": len(observations),
                    "pending_conditions": len(pending),
                    "last_event_start_ms": reference.event_start_ms,
                }
            )
    provisional = Round17DevelopmentResolutionAcquisition(
        cohort_manifest_sha256=selected_cohort.manifest_sha256,
        observations=tuple(sorted(observations, key=lambda item: item.condition_id)),
        pending_condition_ids=tuple(sorted(pending)),
        gamma_batch_request_count=gamma_batch_count,
        clob_market_request_count=len(references),
    )
    return replace(
        provisional,
        acquisition_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(plan, selected_cohort, selected_markets)


def acquire_round17_test_resolutions(
    plan: Round17CohortPlan,
    cohort: Round17CohortManifest,
    markets: Mapping[str, PolymarketFiveMinuteMarket],
    *,
    claim_sha256: str,
    test_access_sha256: str,
    existing: Round17TestResolutionAcquisition | None = None,
    client: PolymarketPublicClient | None = None,
    wall_clock_ms: Callable[[], int] | None = None,
    monotonic_clock_ns: Callable[[], int] | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    checkpoint: Callable[[Round17TestResolutionAcquisition], None] | None = None,
) -> Round17TestResolutionAcquisition:
    """Acquire official test outcomes and resume only the exact persisted claim."""

    selected_cohort = cohort.validated(plan)
    references = selected_cohort.conditions
    reference_by_id = {item.condition_id: item for item in references}
    selected_markets = dict(markets)
    claim = str(claim_sha256 or "").strip().lower()
    access = str(test_access_sha256 or "").strip().lower()
    if (
        _SHA256.fullmatch(claim) is None
        or _SHA256.fullmatch(access) is None
        or any(item.role != "test" for item in references)
        or set(selected_markets) != set(reference_by_id)
        or any(key != market.condition_id for key, market in selected_markets.items())
        or len({market.market_id for market in selected_markets.values()})
        != len(selected_markets)
    ):
        raise ValueError("Round 17 test resolution parent differs")
    prior = (
        None
        if existing is None
        else existing.validated(plan, selected_cohort, selected_markets)
    )
    if prior is not None and (
        prior.claim_sha256 != claim or prior.test_access_sha256 != access
    ):
        raise ValueError("Round 17 test resolution resume claim differs")
    if prior is not None and prior.complete:
        return prior
    pending_ids = (
        tuple(reference_by_id)
        if prior is None
        else prior.pending_condition_ids
    )
    selected_references = tuple(
        reference_by_id[condition_id] for condition_id in pending_ids
    )
    wall = wall_clock_ms or (lambda: time.time_ns() // 1_000_000)
    monotonic = monotonic_clock_ns or time.monotonic_ns
    if int(wall()) < max(item.event_end_ms for item in references):
        raise RuntimeError("Round 17 test markets have not all ended")
    public = client or PolymarketPublicClient()
    gamma_by_market_id: dict[str, Mapping[str, object]] = {}
    gamma_batch_count = 0 if prior is None else prior.gamma_batch_request_count
    for start in range(0, len(selected_references), _GAMMA_BATCH_SIZE):
        batch = selected_references[start : start + _GAMMA_BATCH_SIZE]
        market_ids = tuple(
            selected_markets[item.condition_id].market_id for item in batch
        )
        gamma = public.gamma_markets(market_ids)
        if set(gamma) != set(market_ids):
            raise ValueError("Round 17 test Gamma batch coverage differs")
        gamma_by_market_id.update(gamma)
        gamma_batch_count += 1
        if progress is not None:
            progress(
                {
                    "phase": "test_resolution_gamma",
                    "completed_request_count": gamma_batch_count,
                    "remaining_condition_count": len(selected_references),
                }
            )
    observation_by_id = (
        {}
        if prior is None
        else {item.condition_id: item for item in prior.observations}
    )
    still_pending: list[str] = []
    clob_request_count = 0 if prior is None else prior.clob_market_request_count
    for count, reference in enumerate(selected_references, start=1):
        market = selected_markets[reference.condition_id]
        clob = public.clob_market(reference.condition_id)
        clob_request_count += 1
        observation = _build_observation(
            reference=reference,
            market=market,
            clob=clob,
            gamma=gamma_by_market_id[market.market_id],
            observed_wall_ms=int(wall()),
            observed_monotonic_ns=int(monotonic()),
            source_partition="test",
        )
        if observation is None:
            still_pending.append(reference.condition_id)
        else:
            observation_by_id[reference.condition_id] = observation.validated(
                plan,
                reference,
                market,
            )
        if progress is not None and (
            count == 1
            or count % 25 == 0
            or count == len(selected_references)
        ):
            progress(
                {
                    "phase": "test_resolution_clob",
                    "completed_retry_conditions": count,
                    "retry_condition_count": len(selected_references),
                    "resolved_condition_count": len(observation_by_id),
                    "pending_condition_count": len(still_pending),
                    "last_event_start_ms": reference.event_start_ms,
                }
            )
        if checkpoint is not None and count % 250 == 0:
            remaining = tuple(
                item.condition_id for item in selected_references[count:]
            )
            partial = Round17TestResolutionAcquisition(
                claim_sha256=claim,
                test_access_sha256=access,
                cohort_manifest_sha256=selected_cohort.manifest_sha256,
                observations=tuple(
                    sorted(
                        observation_by_id.values(),
                        key=lambda item: item.condition_id,
                    )
                ),
                pending_condition_ids=tuple(
                    sorted((*still_pending, *remaining))
                ),
                gamma_batch_request_count=gamma_batch_count,
                clob_market_request_count=clob_request_count,
            )
            checkpoint(
                replace(
                    partial,
                    acquisition_sha256=_canonical_sha256(
                        partial.identity_payload()
                    ),
                ).validated(plan, selected_cohort, selected_markets)
            )
    provisional = Round17TestResolutionAcquisition(
        claim_sha256=claim,
        test_access_sha256=access,
        cohort_manifest_sha256=selected_cohort.manifest_sha256,
        observations=tuple(
            sorted(observation_by_id.values(), key=lambda item: item.condition_id)
        ),
        pending_condition_ids=tuple(sorted(still_pending)),
        gamma_batch_request_count=gamma_batch_count,
        clob_market_request_count=clob_request_count,
    )
    completed = replace(
        provisional,
        acquisition_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated(plan, selected_cohort, selected_markets)
    if checkpoint is not None:
        checkpoint(completed)
    return completed


def round17_test_resolution_acquisition_from_mapping(
    value: Mapping[str, object],
    *,
    plan: Round17CohortPlan,
    cohort: Round17CohortManifest,
    markets: Mapping[str, PolymarketFiveMinuteMarket],
) -> Round17TestResolutionAcquisition:
    """Reconstruct an exact persisted test-resolution checkpoint."""

    raw_observations = value.get("observations")
    raw_pending = value.get("pending_condition_ids")
    if (
        not isinstance(raw_observations, list)
        or any(not isinstance(item, Mapping) for item in raw_observations)
        or not isinstance(raw_pending, list)
        or any(not isinstance(item, str) for item in raw_pending)
    ):
        raise ValueError("Round 17 persisted test resolutions differ")
    observations: list[Round17ResolutionObservation] = []
    for raw in raw_observations:
        assert isinstance(raw, Mapping)
        observation = Round17ResolutionObservation(
            source_run_id=str(raw.get("source_run_id") or ""),
            condition_id=str(raw.get("condition_id") or ""),
            event_start_ms=int(raw.get("event_start_ms") or 0),
            observed_wall_ms=int(raw.get("observed_wall_ms") or 0),
            observed_monotonic_ns=int(raw.get("observed_monotonic_ns") or 0),
            winning_asset_id=str(raw.get("winning_asset_id") or ""),
            winning_outcome=str(raw.get("winning_outcome") or ""),
            market_identity_sha256=str(raw.get("market_identity_sha256") or ""),
            clob_payload_json=str(raw.get("clob_payload_json") or ""),
            clob_payload_sha256=str(raw.get("clob_payload_sha256") or ""),
            gamma_payload_json=str(raw.get("gamma_payload_json") or ""),
            gamma_payload_sha256=str(raw.get("gamma_payload_sha256") or ""),
            source_partition=str(raw.get("source_partition") or ""),
            observation_sha256=str(raw.get("observation_sha256") or ""),
        )
        if observation.asdict() != dict(raw):
            raise ValueError("Round 17 persisted resolution observation differs")
        observations.append(observation)
    candidate = Round17TestResolutionAcquisition(
        claim_sha256=str(value.get("claim_sha256") or ""),
        test_access_sha256=str(value.get("test_access_sha256") or ""),
        cohort_manifest_sha256=str(value.get("cohort_manifest_sha256") or ""),
        observations=tuple(observations),
        pending_condition_ids=tuple(raw_pending),
        gamma_batch_request_count=int(value.get("gamma_batch_request_count") or 0),
        clob_market_request_count=int(value.get("clob_market_request_count") or 0),
        acquisition_sha256=str(value.get("acquisition_sha256") or ""),
    )
    if candidate.asdict() != dict(value):
        raise ValueError("Round 17 persisted test resolution acquisition differs")
    return candidate.validated(plan, cohort, markets)


__all__ = [
    "POLYMARKET_ROUND17_DEVELOPMENT_RESOLUTION_CONTRACT_SHA256",
    "POLYMARKET_ROUND17_RESOLUTION_ACQUISITION_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_RESOLUTION_OBSERVATION_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_TEST_RESOLUTION_ACQUISITION_SCHEMA_VERSION",
    "Round17DevelopmentResolutionAcquisition",
    "Round17ResolutionObservation",
    "Round17TestResolutionAcquisition",
    "acquire_round17_development_resolutions",
    "acquire_round17_test_resolutions",
    "round17_test_resolution_acquisition_from_mapping",
]
