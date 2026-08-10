"""Exact-receipt development economics for the independent Round 25 bot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re

from .polymarket import (
    PolymarketFiveMinuteMarket,
    parse_polymarket_five_minute_market,
    validate_clob_market_info,
)
from .polymarket_recorder import PolymarketEvidenceStore, RawStreamMessage, StreamGap
from .polymarket_round21_comparison import (
    POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
    round21_replay_matrix_sha256,
)
from .polymarket_round21_core_features import build_round21_execution_books
from .polymarket_round21_dataset import Round21OfficialOutcome
from .polymarket_round21_execution import (
    POLYMARKET_ROUND21_EXECUTION_SCENARIOS,
    Round21MarketExecutionEvidence,
    round21_execution_scenario,
)
from .polymarket_round21_policy import (
    POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS,
    Round21ProbabilityEnvelope,
    round21_risk_profile,
)
from .polymarket_round21_replay import (
    Round21EconomicMatrixAccumulator,
    Round21EconomicMetrics,
    Round21EconomicReplay,
    Round21ReplayCondition,
)
from .polymarket_round25_dataset import (
    Round25OfficialResolution,
    Round25ResolutionAuthority,
)
from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from .polymarket_round25_evaluation import (
    Round25PredictiveEvaluationResult,
)
from .polymarket_round25_joint_materialization import (
    Round25JointReceiptCondition,
    Round25SingleLaneClobDecoder,
)
from .polymarket_round25_joint_store import (
    audit_round25_joint_store,
    load_round25_joint_condition_identities,
)
from .polymarket_round25_prediction import Round25PreparedPrediction
from .polymarket_round25_terminal import (
    audit_round25_terminal_receipts,
    validate_round25_terminal_transport_manifest,
)


POLYMARKET_ROUND25_ECONOMIC_REPLAY_CONTRACT_SHA256 = (
    "aaf3b72c88b22b9e1275acc8a97fb48ad73f894c8345366eab50f319e05877b0"
)
POLYMARKET_ROUND25_ECONOMIC_RESULT_SCHEMA_VERSION = (
    "polymarket-round25-development-economic-result-v1"
)
POLYMARKET_ROUND25_ECONOMIC_RESULT_MAXIMUM_BYTES = 8 * 1024 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_PROFILES = ("conservative", "regular", "aggressive")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 25 economic JSON contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 25 economic JSON contains {value}")


def _strict_json(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, str):
        raise ValueError(f"Round 25 economic {label} is not canonical JSON")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 25 economic {label} is not strict JSON") from exc
    if not isinstance(parsed, Mapping) or _canonical_json(parsed) != value:
        raise ValueError(f"Round 25 economic {label} differs")
    return parsed


def _digest(value: object, *, label: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 25 economic {label} differs")
    return selected


def _replay_summary_matrix_sha256(
    values: Sequence[Round25EconomicLedgerSummary],
) -> str:
    return _canonical_sha256({
        "schema_version": POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
        "replay_sha256": [value.source_replay_sha256 for value in values],
    })


@dataclass(frozen=True, slots=True)
class _MarketContext:
    condition: Round25JointReceiptCondition
    market: PolymarketFiveMinuteMarket
    evidence: Round21MarketExecutionEvidence


@dataclass(frozen=True, slots=True)
class Round25EconomicLedgerSummary:
    profile: str
    scenario: str
    initial_capital_quote: Decimal
    final_cash_quote: Decimal
    directional_permission_root_sha256: str
    metrics: Round21EconomicMetrics
    qualification_reasons: tuple[str, ...]
    economic_gate_passed: bool
    unknown_state_count: int
    risk_violation_count: int
    source_replay_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "directional_permission_root_sha256": (
                self.directional_permission_root_sha256
            ),
            "economic_gate_passed": self.economic_gate_passed,
            "final_cash_quote": format(self.final_cash_quote, "f"),
            "initial_capital_quote": format(self.initial_capital_quote, "f"),
            "metrics": {
                **self.metrics.identity_payload(),
                "metric_sha256": self.metrics.metric_sha256,
            },
            "profile": self.profile,
            "qualification_reasons": list(self.qualification_reasons),
            "risk_violation_count": self.risk_violation_count,
            "scenario": self.scenario,
            "source_replay_sha256": self.source_replay_sha256,
            "unknown_state_count": self.unknown_state_count,
        }

    def validated(self) -> Round25EconomicLedgerSummary:
        profile = round21_risk_profile(self.profile)
        scenario = round21_execution_scenario(self.scenario)
        metrics = self.metrics.validated()
        if (
            profile.name != self.profile
            or scenario.name != self.scenario
            or not self.initial_capital_quote.is_finite()
            or self.initial_capital_quote <= 0
            or not self.final_cash_quote.is_finite()
            or self.final_cash_quote < 0
            or _SHA256.fullmatch(self.directional_permission_root_sha256) is None
            or self.directional_permission_root_sha256 == _EMPTY_SHA256
            or len(set(self.qualification_reasons)) != len(
                self.qualification_reasons
            )
            or not self.qualification_reasons
            or type(self.economic_gate_passed) is not bool
            or self.economic_gate_passed
            != (self.qualification_reasons == ("sealed_test_evidence_unavailable",))
            or self.unknown_state_count not in {0, 1}
            or self.risk_violation_count < 0
            or _SHA256.fullmatch(self.source_replay_sha256) is None
            or self.source_replay_sha256 == _EMPTY_SHA256
            or metrics.condition_count <= 0
        ):
            raise ValueError("Round 25 economic ledger summary differs")
        return self

    @classmethod
    def from_replay(
        cls,
        replay: Round21EconomicReplay,
    ) -> Round25EconomicLedgerSummary:
        selected = replay.validated()
        return cls(
            profile=selected.profile,
            scenario=selected.scenario,
            initial_capital_quote=selected.initial_capital_quote,
            final_cash_quote=selected.final_cash_quote,
            directional_permission_root_sha256=(
                selected.directional_permission_root_sha256
            ),
            metrics=selected.metrics,
            qualification_reasons=selected.qualification_reasons,
            economic_gate_passed=selected.economic_gate_passed,
            unknown_state_count=selected.unknown_state_count,
            risk_violation_count=selected.risk_violation_count,
            source_replay_sha256=selected.replay_sha256,
        ).validated()


@dataclass(frozen=True, slots=True)
class Round25EconomicConditionPoint:
    profile: str
    condition_id: str
    event_start_ms: int
    utility_quote: Decimal
    cumulative_net_pnl_quote: Decimal
    executed_action_count: int
    source_condition_result_sha256: str
    point_sha256: str
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "cumulative_net_pnl_quote": format(
                self.cumulative_net_pnl_quote,
                "f",
            ),
            "event_start_ms": self.event_start_ms,
            "executed_action_count": self.executed_action_count,
            "profile": self.profile,
            "source_condition_result_sha256": (
                self.source_condition_result_sha256
            ),
            "trading_authority": self.trading_authority,
            "utility_quote": format(self.utility_quote, "f"),
        }

    def validated(self) -> Round25EconomicConditionPoint:
        if (
            self.profile not in _PROFILES
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or not self.utility_quote.is_finite()
            or not self.cumulative_net_pnl_quote.is_finite()
            or type(self.executed_action_count) is not int
            or self.executed_action_count < 0
            or _SHA256.fullmatch(self.source_condition_result_sha256) is None
            or self.source_condition_result_sha256 == _EMPTY_SHA256
            or self.trading_authority is not False
            or self.point_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 economic condition point differs")
        return self

    @classmethod
    def create(
        cls,
        *,
        profile: str,
        condition_id: str,
        event_start_ms: int,
        utility_quote: Decimal,
        cumulative_net_pnl_quote: Decimal,
        executed_action_count: int,
        source_condition_result_sha256: str,
    ) -> Round25EconomicConditionPoint:
        provisional = cls(
            profile=profile,
            condition_id=condition_id,
            event_start_ms=event_start_ms,
            utility_quote=utility_quote,
            cumulative_net_pnl_quote=cumulative_net_pnl_quote,
            executed_action_count=executed_action_count,
            source_condition_result_sha256=source_condition_result_sha256,
            point_sha256=_EMPTY_SHA256,
        )
        return replace(
            provisional,
            point_sha256=_canonical_sha256(provisional.identity_payload()),
        ).validated()


def build_round25_primary_condition_series(
    matrix: Sequence[Round21EconomicReplay],
) -> tuple[Round25EconomicConditionPoint, ...]:
    selected = tuple(value.validated() for value in matrix)
    output: list[Round25EconomicConditionPoint] = []
    for profile in _PROFILES:
        matches = tuple(
            value
            for value in selected
            if value.profile == profile and value.scenario == "primary"
        )
        if len(matches) != 1:
            raise ValueError("Round 25 primary economic replay differs")
        cumulative = Decimal("0")
        for condition in matches[0].conditions:
            selected_condition = condition.validated()
            cumulative += selected_condition.utility_quote
            output.append(Round25EconomicConditionPoint.create(
                profile=profile,
                condition_id=selected_condition.condition_id,
                event_start_ms=selected_condition.event_start_ms,
                utility_quote=selected_condition.utility_quote,
                cumulative_net_pnl_quote=cumulative,
                executed_action_count=selected_condition.executed_action_count,
                source_condition_result_sha256=(
                    selected_condition.condition_result_sha256
                ),
            ))
        if cumulative != matches[0].metrics.net_pnl_quote:
            raise ValueError("Round 25 primary economic series accounting differs")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class Round25DevelopmentEconomicResult:
    terminal_transport_manifest_sha256: str
    terminal_receipt_audit_sha256: str
    feature_store_manifest_sha256: str
    resolution_store_manifest_sha256: str
    resolution_authority_sha256: str
    prepared_prediction_sha256: str
    predictive_result_sha256: str
    nominated_candidate_id: str
    candidate_source_artifact_sha256: str
    candidate_probability_batch_sha256: str
    source_condition_set_sha256: str
    source_condition_count: int
    ledger_summaries: tuple[Round25EconomicLedgerSummary, ...]
    primary_condition_series: tuple[Round25EconomicConditionPoint, ...]
    source_replay_matrix_sha256: str
    development_economic_gate_passed: bool
    result_sha256: str
    schema_version: str = POLYMARKET_ROUND25_ECONOMIC_RESULT_SCHEMA_VERSION
    contract_sha256: str = POLYMARKET_ROUND25_ECONOMIC_REPLAY_CONTRACT_SHA256
    development_evidence_only: bool = True
    edge_verified: bool = False
    profitability_verified: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False
    orders_submitted: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "candidate_probability_batch_sha256": (
                self.candidate_probability_batch_sha256
            ),
            "candidate_source_artifact_sha256": (
                self.candidate_source_artifact_sha256
            ),
            "contract_sha256": self.contract_sha256,
            "development_economic_gate_passed": (
                self.development_economic_gate_passed
            ),
            "development_evidence_only": self.development_evidence_only,
            "edge_verified": self.edge_verified,
            "feature_store_manifest_sha256": self.feature_store_manifest_sha256,
            "ledger_summaries": [
                value.identity_payload() for value in self.ledger_summaries
            ],
            "live_trading_authority": self.live_trading_authority,
            "nominated_candidate_id": self.nominated_candidate_id,
            "orders_submitted": self.orders_submitted,
            "paper_trading_authority": self.paper_trading_authority,
            "primary_condition_series": [
                {**value.identity_payload(), "point_sha256": value.point_sha256}
                for value in self.primary_condition_series
            ],
            "predictive_result_sha256": self.predictive_result_sha256,
            "prepared_prediction_sha256": self.prepared_prediction_sha256,
            "profitability_verified": self.profitability_verified,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "resolution_store_manifest_sha256": (
                self.resolution_store_manifest_sha256
            ),
            "schema_version": self.schema_version,
            "source_condition_count": self.source_condition_count,
            "source_condition_set_sha256": self.source_condition_set_sha256,
            "source_replay_matrix_sha256": self.source_replay_matrix_sha256,
            "terminal_receipt_audit_sha256": self.terminal_receipt_audit_sha256,
            "terminal_transport_manifest_sha256": (
                self.terminal_transport_manifest_sha256
            ),
        }

    def serialized_payload(self) -> dict[str, object]:
        return {**self.identity_payload(), "result_sha256": self.result_sha256}

    def validated(self) -> Round25DevelopmentEconomicResult:
        summaries = tuple(value.validated() for value in self.ledger_summaries)
        series = tuple(value.validated() for value in self.primary_condition_series)
        hashes = (
            self.terminal_transport_manifest_sha256,
            self.terminal_receipt_audit_sha256,
            self.feature_store_manifest_sha256,
            self.resolution_store_manifest_sha256,
            self.resolution_authority_sha256,
            self.prepared_prediction_sha256,
            self.predictive_result_sha256,
            self.candidate_source_artifact_sha256,
            self.candidate_probability_batch_sha256,
            self.source_condition_set_sha256,
            self.source_replay_matrix_sha256,
            self.result_sha256,
        )
        if (
            any(_SHA256.fullmatch(value) is None or value == _EMPTY_SHA256 for value in hashes)
            or self.nominated_candidate_id not in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
            or type(self.source_condition_count) is not int
            or self.source_condition_count <= 0
            or len(summaries) != 81
            or tuple((value.profile, value.scenario) for value in summaries)
            != tuple(
                (profile, scenario.name)
                for profile in _PROFILES
                for scenario in POLYMARKET_ROUND21_EXECUTION_SCENARIOS
            )
            or any(
                value.metrics.condition_count != self.source_condition_count
                for value in summaries
            )
            or self.source_replay_matrix_sha256
            != _replay_summary_matrix_sha256(summaries)
            or len(series) != len(_PROFILES) * self.source_condition_count
            or len({(value.profile, value.condition_id) for value in series})
            != len(series)
            or not _validate_primary_condition_series(
                series,
                summaries=summaries,
                condition_count=self.source_condition_count,
            )
            or self.development_economic_gate_passed
            is not all(value.economic_gate_passed for value in summaries)
            or self.schema_version != POLYMARKET_ROUND25_ECONOMIC_RESULT_SCHEMA_VERSION
            or self.contract_sha256
            != POLYMARKET_ROUND25_ECONOMIC_REPLAY_CONTRACT_SHA256
            or self.development_evidence_only is not True
            or any(
                value is not False
                for value in (
                    self.edge_verified,
                    self.profitability_verified,
                    self.paper_trading_authority,
                    self.live_trading_authority,
                    self.orders_submitted,
                )
            )
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 development economic result differs")
        return self


def _validate_primary_condition_series(
    values: Sequence[Round25EconomicConditionPoint],
    *,
    summaries: Sequence[Round25EconomicLedgerSummary],
    condition_count: int,
) -> bool:
    expected_population: tuple[tuple[str, int], ...] | None = None
    for profile in _PROFILES:
        selected = tuple(value for value in values if value.profile == profile)
        primary = tuple(
            value
            for value in summaries
            if value.profile == profile and value.scenario == "primary"
        )
        population = tuple(
            (value.condition_id, value.event_start_ms) for value in selected
        )
        if (
            len(selected) != condition_count
            or len(primary) != 1
            or population != tuple(sorted(population, key=lambda value: value[1]))
            or len(set(population)) != len(population)
            or any(
                population[index - 1][1] >= population[index][1]
                for index in range(1, len(population))
            )
            or expected_population is not None
            and population != expected_population
        ):
            return False
        cumulative = Decimal("0")
        action_count = 0
        for point in selected:
            cumulative += point.utility_quote
            action_count += point.executed_action_count
            if point.cumulative_net_pnl_quote != cumulative:
                return False
        if (
            cumulative != primary[0].metrics.net_pnl_quote
            or action_count != primary[0].metrics.executed_action_count
        ):
            return False
        expected_population = population
    return expected_population is not None


def build_round25_probability_envelopes(
    *,
    prepared_prediction: Round25PreparedPrediction,
    predictive_result: Round25PredictiveEvaluationResult,
) -> dict[str, tuple[Round21ProbabilityEnvelope, ...]]:
    """Adapt the frozen nominated probabilities without adding uncertainty claims."""

    prepared = prepared_prediction.validated()
    result = predictive_result.validated()
    panel = prepared.panel
    candidate_id = result.nominated_candidate_id
    if (
        not result.predictive_gate_passed
        or candidate_id is None
        or result.prediction_panel_sha256 != panel.panel_sha256
    ):
        raise RuntimeError("Round 25 economic predictive nomination gate is closed")
    candidates = tuple(
        value for value in panel.candidate_predictions if value.candidate_id == candidate_id
    )
    if len(candidates) != 1:
        raise ValueError("Round 25 nominated probability batch differs")
    candidate = candidates[0]
    grouped: dict[str, list[Round21ProbabilityEnvelope]] = {}
    for index, probability_value in enumerate(candidate.probabilities):
        probability = Decimal(format(float(probability_value), ".17g"))
        feature_sha256 = _canonical_sha256({
            "condition_id": panel.row_condition_ids[index],
            "decision_time_ms": int(panel.decision_time_ms[index]),
            "feature_source_chain_sha256": panel.feature_source_chain_sha256[index],
            "prediction_panel_sha256": panel.panel_sha256,
            "round25_adapter": "target_blind_joint_feature_endpoint_v1",
        })
        envelope = Round21ProbabilityEnvelope.create(
            condition_id=panel.row_condition_ids[index],
            decision_time_ms=int(panel.decision_time_ms[index]),
            probability_up=probability,
            lower_up=probability,
            upper_up=probability,
            model_layer="core",
            source_model_artifact_sha256=candidate.source_artifact_sha256,
            source_probability_batch_sha256=candidate.probabilities_sha256,
            feature_row_sha256=feature_sha256,
            feature_support_eligible=True,
        )
        grouped.setdefault(envelope.condition_id, []).append(envelope)
    output = {
        condition_id: tuple(values) for condition_id, values in grouped.items()
    }
    if any(len(values) != 16 for values in output.values()):
        raise ValueError("Round 25 economic probability condition shape differs")
    return output


def _adapt_outcomes(
    resolutions: Sequence[Round25OfficialResolution],
    *,
    authority: Round25ResolutionAuthority,
) -> dict[str, Round21OfficialOutcome]:
    selected_authority = authority.validated()
    output: dict[str, Round21OfficialOutcome] = {}
    for value in resolutions:
        resolution = value.validated(selected_authority)
        if resolution.condition_id in output:
            raise ValueError("Round 25 economic official outcome is duplicated")
        output[resolution.condition_id] = Round21OfficialOutcome.create(
            condition_id=resolution.condition_id,
            event_start_ms=resolution.event_start_ms,
            resolved_up=resolution.target_up,
            observed_at_ms=resolution.resolved_at_ms,
            source="round25_official_polymarket_resolved_outcome",
            source_payload_sha256=resolution.official_payload_sha256,
        )
    return output


def _load_market_contexts(
    *,
    database: str | Path,
    conditions: Sequence[Round25JointReceiptCondition],
) -> dict[str, _MarketContext]:
    expected = {value.condition_id: value.validated() for value in conditions}
    path = Path(database)
    if path.is_symlink() or not path.is_file() or Path(f"{path}.wal").exists():
        raise ValueError("Round 25 economic source database is unavailable")
    before = path.stat()
    output: dict[str, _MarketContext] = {}
    with PolymarketEvidenceStore(
        path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        rows = store.connect().execute(
            """
            SELECT run_id, condition_id, observed_wall_ms,
                   observed_monotonic_ns, gamma_payload_json,
                   gamma_payload_sha256, clob_info_json, clob_info_sha256,
                   up_fee_rate_sha256, down_fee_rate_sha256,
                   maker_base_fee, taker_base_fee,
                   taker_order_delay_enabled, minimum_order_age_seconds,
                   snapshot_sha256
            FROM polymarket_market_snapshot
            ORDER BY event_start_ms, condition_id
            """
        ).fetchall()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or Path(f"{path}.wal").exists()
    ):
        raise RuntimeError("Round 25 economic source changed during context access")
    for row in rows:
        condition_id = str(row[1])
        condition = expected.get(condition_id)
        if condition is None:
            continue
        gamma = _strict_json(row[4], label="Gamma market payload")
        clob = _strict_json(row[6], label="CLOB market payload")
        market = parse_polymarket_five_minute_market(gamma)
        clob_evidence = validate_clob_market_info(market, clob)
        evidence = Round21MarketExecutionEvidence.create(
            condition_id=condition_id,
            observed_wall_ms=int(row[2]),
            observed_monotonic_ns=int(row[3]),
            maker_base_fee=int(row[10]),
            taker_base_fee=int(row[11]),
            taker_order_delay_enabled=bool(row[12]),
            general_order_delay_seconds=int(
                clob_evidence["general_order_delay_seconds"]
            ),
            minimum_order_age_seconds=int(row[13]),
            clob_info_sha256=str(row[7]),
            up_fee_rate_sha256=str(row[8]),
            down_fee_rate_sha256=str(row[9]),
            snapshot_sha256=str(row[14]),
        )
        if (
            condition_id in output
            or str(row[0]) != condition.run_id
            or market.asset != "BTC"
            or market.condition_id != condition.condition_id
            or market.event_start_ms != condition.event_start_ms
            or market.end_ms != condition.event_end_ms
            or market.up_token_id != condition.up_token_id
            or market.down_token_id != condition.down_token_id
            or market.gamma_payload_sha256 != str(row[5])
            or evidence.snapshot_sha256 != condition.snapshot_sha256
            or clob_evidence["payload_json"] != str(row[6])
            or clob_evidence["payload_sha256"] != str(row[7])
            or clob_evidence["maker_base_fee"] != evidence.maker_base_fee
            or clob_evidence["taker_base_fee"] != evidence.taker_base_fee
            or clob_evidence["taker_order_delay_enabled"]
            != evidence.taker_order_delay_enabled
            or clob_evidence["minimum_order_age_seconds"]
            != evidence.minimum_order_age_seconds
        ):
            raise ValueError("Round 25 market execution context differs")
        output[condition_id] = _MarketContext(condition, market, evidence)
    if set(output) != set(expected):
        raise ValueError("Round 25 market execution context is incomplete")
    return output


class _Round25EconomicObserver:
    def __init__(
        self,
        *,
        contexts: Mapping[str, _MarketContext],
        outcomes: Mapping[str, Round21OfficialOutcome],
        envelopes: Mapping[str, tuple[Round21ProbabilityEnvelope, ...]],
        terminal_manifest_sha256: str,
        feature_manifest_sha256: str,
        predictive_result_sha256: str,
        accumulator: Round21EconomicMatrixAccumulator,
        progress: ProgressCallback | None,
    ) -> None:
        expected = set(contexts)
        if expected != set(outcomes) or expected != set(envelopes):
            raise ValueError("Round 25 economic observer population differs")
        self.contexts = dict(contexts)
        self.outcomes = dict(outcomes)
        self.envelopes = dict(envelopes)
        self.terminal_manifest_sha256 = terminal_manifest_sha256
        self.feature_manifest_sha256 = feature_manifest_sha256
        self.predictive_result_sha256 = predictive_result_sha256
        self.accumulator = accumulator
        self.progress = progress
        self._run_id = ""
        self._decoder: Round25SingleLaneClobDecoder | None = None
        self._active_events: dict[str, list[object]] = {}
        self._completed: set[str] = set()
        self._last_event_start_ms = 0

    def start_run(
        self,
        segment: Mapping[str, object],
        gaps: tuple[StreamGap, ...],
    ) -> None:
        if self._run_id:
            raise RuntimeError("Round 25 economic observer run is already open")
        run_id = str(segment.get("run_id") or "")
        conditions = tuple(
            context.condition
            for context in self.contexts.values()
            if context.condition.run_id == run_id
        )
        for gap in gaps:
            selected_gap = gap.validated()
            if selected_gap.stream != "clob_market":
                continue
            if any(
                condition.event_start_ms
                - POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
                <= selected_gap.opened_at_ms
                < condition.event_end_ms
                for condition in conditions
            ):
                raise ValueError("Round 25 economic CLOB gap intersects replay")
        self._run_id = run_id
        self._decoder = Round25SingleLaneClobDecoder()
        self._active_events = {condition.condition_id: [] for condition in conditions}

    def _finalize(self, condition_id: str) -> None:
        context = self.contexts[condition_id]
        condition = context.condition
        events = tuple(self._active_events.pop(condition_id))
        books = tuple(
            book
            for book in build_round21_execution_books(
                condition_id=condition_id,
                up_token_id=condition.up_token_id,
                down_token_id=condition.down_token_id,
                union_events=events,
                admitted_gap_free=True,
            )
            if condition.event_start_ms
            - POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
            <= book.received_wall_ms
            < condition.event_end_ms
        )
        reconciliation_sha256 = _canonical_sha256({
            "condition_id": condition_id,
            "event_start_ms": condition.event_start_ms,
            "feature_store_manifest_sha256": self.feature_manifest_sha256,
            "market_execution_evidence_sha256": context.evidence.evidence_sha256,
            "predictive_result_sha256": self.predictive_result_sha256,
            "round25_condition_snapshot_sha256": condition.snapshot_sha256,
            "schema_version": POLYMARKET_ROUND25_ECONOMIC_RESULT_SCHEMA_VERSION,
            "terminal_transport_manifest_sha256": self.terminal_manifest_sha256,
            "union_event_sha256": [event.event_sha256 for event in events],
            "future_books_accessed_for_decision": False,
            "outcome_accessed_for_decision": False,
            "trading_authority": False,
        })
        replay_condition = Round21ReplayCondition.create(
            market=context.market,
            market_evidence=context.evidence,
            envelopes=self.envelopes[condition_id],
            books=books,
            outcome=self.outcomes[condition_id],
            source_manifest_sha256=self.terminal_manifest_sha256,
            reconciliation_sha256=reconciliation_sha256,
        )
        if condition.event_start_ms <= self._last_event_start_ms:
            raise ValueError("Round 25 economic condition chronology differs")
        self.accumulator.observe(replay_condition)
        self._last_event_start_ms = condition.event_start_ms
        self._completed.add(condition_id)
        if self.progress is not None:
            self.progress(
                "economic_condition_replayed",
                {
                    "condition_count": len(self._completed),
                    "condition_id": condition_id,
                    "event_start_ms": condition.event_start_ms,
                },
            )

    def _finalize_ready(self, wall_ms: int, *, force: bool = False) -> None:
        ready = tuple(
            condition_id
            for condition_id in self._active_events
            if force or wall_ms >= self.contexts[condition_id].condition.event_end_ms
        )
        for condition_id in sorted(
            ready,
            key=lambda value: (
                self.contexts[value].condition.event_start_ms,
                value,
            ),
        ):
            self._finalize(condition_id)

    def observe_message(
        self,
        segment: Mapping[str, object],
        message: RawStreamMessage,
    ) -> None:
        if (
            not self._run_id
            or self._decoder is None
            or str(segment.get("run_id") or "") != self._run_id
        ):
            raise RuntimeError("Round 25 economic observer run is unavailable")
        selected = message.validated()
        if selected.stream == "clob_market":
            for event, condition_id in self._decoder.add(selected):
                context = self.contexts.get(condition_id)
                if context is None or condition_id not in self._active_events:
                    continue
                condition = context.condition
                if (
                    condition.event_start_ms
                    - POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
                    <= event.selected_received_wall_ms
                    < condition.event_end_ms
                ):
                    self._active_events[condition_id].append(event)
        elif selected.stream != "polymarket_rtds":
            raise ValueError("Round 25 economic source stream differs")
        self._finalize_ready(selected.received_wall_ms)

    def finish_run(self, segment: Mapping[str, object]) -> None:
        if not self._run_id or str(segment.get("run_id") or "") != self._run_id:
            raise RuntimeError("Round 25 economic observer run is unavailable")
        self._finalize_ready(2**63 - 1, force=True)
        self._run_id = ""
        self._decoder = None
        self._active_events = {}

    def finish(self) -> None:
        if self._run_id or self._active_events or self._completed != set(self.contexts):
            raise ValueError("Round 25 economic receipt population is incomplete")


def replay_round25_development_economics(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    feature_database: str | Path,
    resolution_store_manifest: Mapping[str, object],
    resolution_authority: Round25ResolutionAuthority,
    selection_resolutions: Sequence[Round25OfficialResolution],
    prepared_prediction: Round25PreparedPrediction,
    predictive_result: Round25PredictiveEvaluationResult,
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    progress: ProgressCallback | None = None,
) -> Round25DevelopmentEconomicResult:
    """Run one source scan only after the frozen predictive gate nominates a model."""

    transport = validate_round25_terminal_transport_manifest(
        terminal_transport_manifest
    )
    feature_manifest = audit_round25_joint_store(feature_database)
    prepared = prepared_prediction.validated()
    predictive = predictive_result.validated()
    authority = resolution_authority.validated()
    resolution_manifest_body = dict(resolution_store_manifest)
    claimed_resolution_manifest_sha256 = _digest(
        resolution_manifest_body.pop("manifest_sha256", None),
        label="resolution store manifest",
    )
    if (
        not predictive.predictive_gate_passed
        or predictive.nominated_candidate_id is None
        or predictive.prediction_panel_sha256 != prepared.panel.panel_sha256
        or predictive.resolution_authority_sha256 != authority.authority_sha256
        or feature_manifest["terminal_transport_manifest_sha256"]
        != transport["manifest_sha256"]
        or prepared.source_receipt_audit_sha256
        != feature_manifest["terminal_receipt_audit_sha256"]
        or resolution_store_manifest.get("authority_sha256")
        != authority.authority_sha256
        or claimed_resolution_manifest_sha256
        != _canonical_sha256(resolution_manifest_body)
    ):
        raise RuntimeError("Round 25 development economic entry gate is closed")
    resolution_manifest_sha256 = claimed_resolution_manifest_sha256
    envelope_groups = build_round25_probability_envelopes(
        prepared_prediction=prepared,
        predictive_result=predictive,
    )
    panel_condition_ids = tuple(dict.fromkeys(prepared.panel.row_condition_ids))
    all_conditions = load_round25_joint_condition_identities(feature_database)
    condition_by_id = {value.condition_id: value for value in all_conditions}
    try:
        conditions = tuple(condition_by_id[value] for value in panel_condition_ids)
    except KeyError as exc:
        raise ValueError("Round 25 economic source condition is unavailable") from exc
    if (
        set(panel_condition_ids) != set(envelope_groups)
        or any(value.role != "selection" for value in conditions)
    ):
        raise ValueError("Round 25 economic selection population differs")
    outcomes = _adapt_outcomes(selection_resolutions, authority=authority)
    if set(outcomes) != set(panel_condition_ids):
        raise ValueError("Round 25 economic official outcome population differs")
    contexts = _load_market_contexts(
        database=source_database,
        conditions=conditions,
    )
    accumulator = Round21EconomicMatrixAccumulator(
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
    )
    observer = _Round25EconomicObserver(
        contexts=contexts,
        outcomes=outcomes,
        envelopes=envelope_groups,
        terminal_manifest_sha256=transport["manifest_sha256"],
        feature_manifest_sha256=feature_manifest["manifest_sha256"],
        predictive_result_sha256=predictive.result_sha256,
        accumulator=accumulator,
        progress=progress,
    )
    receipt_audit = audit_round25_terminal_receipts(
        database=source_database,
        terminal_transport_manifest=transport,
        observed_at_ms=int(feature_manifest["created_at_ms"]),
        observer=observer,
    )
    observer.finish()
    if (
        receipt_audit["audit_sha256"]
        != feature_manifest["terminal_receipt_audit_sha256"]
    ):
        raise ValueError("Round 25 economic terminal receipt identity differs")
    matrix = accumulator.finish()
    candidate = next(
        value
        for value in prepared.panel.candidate_predictions
        if value.candidate_id == predictive.nominated_candidate_id
    )
    condition_set_sha256 = _canonical_sha256({
        "condition_id": panel_condition_ids,
        "event_start_ms": [value.event_start_ms for value in conditions],
        "feature_store_manifest_sha256": feature_manifest["manifest_sha256"],
        "selection_dataset_sha256": predictive.selection_dataset_sha256,
    })
    summaries = tuple(Round25EconomicLedgerSummary.from_replay(value) for value in matrix)
    primary_series = build_round25_primary_condition_series(matrix)
    provisional = Round25DevelopmentEconomicResult(
        terminal_transport_manifest_sha256=transport["manifest_sha256"],
        terminal_receipt_audit_sha256=receipt_audit["audit_sha256"],
        feature_store_manifest_sha256=feature_manifest["manifest_sha256"],
        resolution_store_manifest_sha256=resolution_manifest_sha256,
        resolution_authority_sha256=authority.authority_sha256,
        prepared_prediction_sha256=prepared.prepared_sha256,
        predictive_result_sha256=predictive.result_sha256,
        nominated_candidate_id=predictive.nominated_candidate_id,
        candidate_source_artifact_sha256=candidate.source_artifact_sha256,
        candidate_probability_batch_sha256=candidate.probabilities_sha256,
        source_condition_set_sha256=condition_set_sha256,
        source_condition_count=len(conditions),
        ledger_summaries=summaries,
        primary_condition_series=primary_series,
        source_replay_matrix_sha256=round21_replay_matrix_sha256(matrix),
        development_economic_gate_passed=all(
            value.economic_gate_passed for value in matrix
        ),
        result_sha256=_EMPTY_SHA256,
    )
    result = replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()
    if progress is not None:
        progress(
            "economic_replay_complete",
            {
                "development_economic_gate_passed": (
                    result.development_economic_gate_passed
                ),
                "ledger_count": len(result.ledger_summaries),
                "result_sha256": result.result_sha256,
            },
        )
    return result


def _decode_metrics(value: object) -> Round21EconomicMetrics:
    if not isinstance(value, Mapping):
        raise ValueError("Round 25 economic ledger metrics differ")
    expected = {
        "calendar_day_count",
        "condition_count",
        "daily_mean_pnl_lower_95_quote",
        "executed_action_count",
        "gross_loss_quote",
        "gross_profit_quote",
        "maximum_drawdown_fraction",
        "mean_event_utility_quote",
        "median_daily_pnl_quote",
        "metric_sha256",
        "net_pnl_quote",
        "profit_factor",
        "realized_maximum_drawdown_fraction",
        "tail_mean_worst_five_percent_quote",
    }
    integer_fields = (
        "condition_count",
        "calendar_day_count",
        "executed_action_count",
    )
    if set(value) != expected or any(
        type(value[field]) is not int for field in integer_fields
    ):
        raise ValueError("Round 25 economic ledger metrics differ")
    lower = value["daily_mean_pnl_lower_95_quote"]
    return Round21EconomicMetrics(
        condition_count=int(value["condition_count"]),
        calendar_day_count=int(value["calendar_day_count"]),
        executed_action_count=int(value["executed_action_count"]),
        net_pnl_quote=Decimal(str(value["net_pnl_quote"])),
        mean_event_utility_quote=Decimal(str(value["mean_event_utility_quote"])),
        median_daily_pnl_quote=Decimal(str(value["median_daily_pnl_quote"])),
        gross_profit_quote=Decimal(str(value["gross_profit_quote"])),
        gross_loss_quote=Decimal(str(value["gross_loss_quote"])),
        profit_factor=Decimal(str(value["profit_factor"])),
        maximum_drawdown_fraction=Decimal(str(value["maximum_drawdown_fraction"])),
        realized_maximum_drawdown_fraction=Decimal(
            str(value["realized_maximum_drawdown_fraction"])
        ),
        tail_mean_worst_five_percent_quote=Decimal(
            str(value["tail_mean_worst_five_percent_quote"])
        ),
        daily_mean_pnl_lower_95_quote=(
            None if lower is None else Decimal(str(lower))
        ),
        metric_sha256=str(value["metric_sha256"]),
    ).validated()


def _decode_summary(value: object) -> Round25EconomicLedgerSummary:
    if not isinstance(value, Mapping):
        raise ValueError("Round 25 economic ledger summary differs")
    expected = {
        "directional_permission_root_sha256",
        "economic_gate_passed",
        "final_cash_quote",
        "initial_capital_quote",
        "metrics",
        "profile",
        "qualification_reasons",
        "risk_violation_count",
        "scenario",
        "source_replay_sha256",
        "unknown_state_count",
    }
    reasons = value.get("qualification_reasons")
    if (
        set(value) != expected
        or not isinstance(reasons, list)
        or any(not isinstance(item, str) for item in reasons)
        or type(value.get("economic_gate_passed")) is not bool
        or type(value.get("unknown_state_count")) is not int
        or type(value.get("risk_violation_count")) is not int
    ):
        raise ValueError("Round 25 economic ledger summary differs")
    return Round25EconomicLedgerSummary(
        profile=str(value["profile"]),
        scenario=str(value["scenario"]),
        initial_capital_quote=Decimal(str(value["initial_capital_quote"])),
        final_cash_quote=Decimal(str(value["final_cash_quote"])),
        directional_permission_root_sha256=str(
            value["directional_permission_root_sha256"]
        ),
        metrics=_decode_metrics(value["metrics"]),
        qualification_reasons=tuple(str(item) for item in reasons),
        economic_gate_passed=value["economic_gate_passed"],
        unknown_state_count=int(value["unknown_state_count"]),
        risk_violation_count=int(value["risk_violation_count"]),
        source_replay_sha256=str(value["source_replay_sha256"]),
    ).validated()


def _decode_condition_point(value: object) -> Round25EconomicConditionPoint:
    if not isinstance(value, Mapping):
        raise ValueError("Round 25 economic condition point differs")
    expected = {
        "condition_id",
        "cumulative_net_pnl_quote",
        "event_start_ms",
        "executed_action_count",
        "point_sha256",
        "profile",
        "source_condition_result_sha256",
        "trading_authority",
        "utility_quote",
    }
    if (
        set(value) != expected
        or type(value.get("event_start_ms")) is not int
        or type(value.get("executed_action_count")) is not int
        or type(value.get("trading_authority")) is not bool
    ):
        raise ValueError("Round 25 economic condition point differs")
    return Round25EconomicConditionPoint(
        profile=str(value["profile"]),
        condition_id=str(value["condition_id"]),
        event_start_ms=int(value["event_start_ms"]),
        utility_quote=Decimal(str(value["utility_quote"])),
        cumulative_net_pnl_quote=Decimal(
            str(value["cumulative_net_pnl_quote"])
        ),
        executed_action_count=int(value["executed_action_count"]),
        source_condition_result_sha256=str(
            value["source_condition_result_sha256"]
        ),
        point_sha256=str(value["point_sha256"]),
        trading_authority=value["trading_authority"],
    ).validated()


def load_round25_economic_result(
    path: str | Path,
) -> Round25DevelopmentEconomicResult:
    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_file()
        or not 2 <= source.stat().st_size <= POLYMARKET_ROUND25_ECONOMIC_RESULT_MAXIMUM_BYTES
    ):
        raise ValueError("Round 25 economic result file differs")
    try:
        value = json.loads(
            source.read_text(encoding="ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 25 economic result is unreadable") from exc
    expected = set(Round25DevelopmentEconomicResult.__dataclass_fields__)
    summaries = value.get("ledger_summaries") if isinstance(value, Mapping) else None
    series = value.get("primary_condition_series") if isinstance(value, Mapping) else None
    bool_fields = (
        "development_economic_gate_passed",
        "development_evidence_only",
        "edge_verified",
        "profitability_verified",
        "paper_trading_authority",
        "live_trading_authority",
        "orders_submitted",
    )
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not isinstance(summaries, list)
        or not isinstance(series, list)
        or type(value.get("source_condition_count")) is not int
        or any(type(value.get(field)) is not bool for field in bool_fields)
    ):
        raise ValueError("Round 25 economic result payload differs")
    return Round25DevelopmentEconomicResult(
        terminal_transport_manifest_sha256=str(
            value["terminal_transport_manifest_sha256"]
        ),
        terminal_receipt_audit_sha256=str(value["terminal_receipt_audit_sha256"]),
        feature_store_manifest_sha256=str(value["feature_store_manifest_sha256"]),
        resolution_store_manifest_sha256=str(
            value["resolution_store_manifest_sha256"]
        ),
        resolution_authority_sha256=str(value["resolution_authority_sha256"]),
        prepared_prediction_sha256=str(value["prepared_prediction_sha256"]),
        predictive_result_sha256=str(value["predictive_result_sha256"]),
        nominated_candidate_id=str(value["nominated_candidate_id"]),
        candidate_source_artifact_sha256=str(
            value["candidate_source_artifact_sha256"]
        ),
        candidate_probability_batch_sha256=str(
            value["candidate_probability_batch_sha256"]
        ),
        source_condition_set_sha256=str(value["source_condition_set_sha256"]),
        source_condition_count=int(value["source_condition_count"]),
        ledger_summaries=tuple(_decode_summary(item) for item in summaries),
        primary_condition_series=tuple(
            _decode_condition_point(item) for item in series
        ),
        source_replay_matrix_sha256=str(value["source_replay_matrix_sha256"]),
        development_economic_gate_passed=value[
            "development_economic_gate_passed"
        ],
        result_sha256=str(value["result_sha256"]),
        schema_version=str(value["schema_version"]),
        contract_sha256=str(value["contract_sha256"]),
        development_evidence_only=value["development_evidence_only"],
        edge_verified=value["edge_verified"],
        profitability_verified=value["profitability_verified"],
        paper_trading_authority=value["paper_trading_authority"],
        live_trading_authority=value["live_trading_authority"],
        orders_submitted=value["orders_submitted"],
    ).validated()


def write_round25_economic_result(
    path: str | Path,
    result: Round25DevelopmentEconomicResult,
) -> Path:
    if not isinstance(result, Round25DevelopmentEconomicResult):
        raise TypeError("Round 25 economic result type differs")
    selected = result.validated()
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("Round 25 economic result path differs")
    payload = (_canonical_json(selected.serialized_payload()) + "\n").encode("ascii")
    if len(payload) > POLYMARKET_ROUND25_ECONOMIC_RESULT_MAXIMUM_BYTES:
        raise ValueError("Round 25 economic result exceeds its storage bound")
    if target.exists():
        if load_round25_economic_result(target).result_sha256 == selected.result_sha256:
            return target
        raise FileExistsError("Round 25 economic result path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return target


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND25_ECONOMIC_REPLAY_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_ECONOMIC_RESULT_SCHEMA_VERSION",
    "Round25DevelopmentEconomicResult",
    "Round25EconomicConditionPoint",
    "Round25EconomicLedgerSummary",
    "build_round25_primary_condition_series",
    "build_round25_probability_envelopes",
    "load_round25_economic_result",
    "replay_round25_development_economics",
    "write_round25_economic_result",
]
