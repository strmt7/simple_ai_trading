"""One-pass development economics for the independent Polymarket Round 21 bot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .polymarket import (
    PolymarketFiveMinuteMarket,
    parse_polymarket_five_minute_market,
    validate_clob_market_info,
)
from .polymarket_recorder import PolymarketEvidenceStore, StreamGap
from .polymarket_ai_veto import PolymarketAIVetoCase
from .polymarket_round21_ai import (
    Round21HistoricalAICaseCollector,
    Round21HistoricalAICaseFactory,
)
from .polymarket_round21_comparison import (
    POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION,
    Round21MatchedEconomicComparison,
    compare_round21_optional_replay_matrices,
    round21_replay_matrix_sha256,
)
from .polymarket_round21_core_features import build_round21_execution_books
from .polymarket_round21_corpus import (
    Round21ConditionSource,
    Round21CoreCondition,
    Round21CoreCorpusObserver,
    load_round21_core_conditions,
)
from .polymarket_round21_dataset import (
    Round21OfficialOutcome,
    Round21PartitionPolicy,
)
from .polymarket_round21_execution import Round21MarketExecutionEvidence
from .polymarket_round21_model import (
    Round21DevelopmentPanel,
    Round21InferencePanel,
    Round21ProbabilityBatch,
    compile_round21_matched_core_predictor,
    compile_round21_probability_predictor,
    validate_round21_development_artifact,
)
from .polymarket_round21_operator import load_round21_official_outcomes
from .polymarket_round21_policy import (
    POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS,
    Round21ProbabilityEnvelope,
    build_round21_probability_envelopes,
)
from .polymarket_round21_replay import (
    Round21EconomicMatrixAccumulator,
    Round21EconomicReplay,
    Round21ReplayCondition,
)
from .polymarket_round21_terminal import (
    audit_round21_terminal_receipts,
    validate_round21_terminal_transport_manifest,
)


POLYMARKET_ROUND21_DEVELOPMENT_ECONOMIC_SCHEMA_VERSION = (
    "polymarket-round21-development-economic-result-v1"
)
POLYMARKET_ROUND21_SEALED_REPLAY_SCHEMA_VERSION = (
    "polymarket-round21-sealed-economic-replay-evidence-v1"
)
_DEVELOPMENT_ROLES = ("train", "tune_calibration", "tune_selection")
_LAYERS = ("core", "core_spot", "core_spot_usdm")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
ProgressCallback = Callable[[str, Mapping[str, object]], None]
SelectedConditionSink = Callable[[Round21ReplayCondition], None]


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
            raise ValueError("Round 21 market evidence contains duplicate JSON keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 market evidence contains {value}")


def _strict_json(value: object, *, name: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(
            str(value),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Round 21 {name} is not strict JSON") from exc
    if not isinstance(parsed, Mapping) or _canonical_json(parsed) != str(value):
        raise ValueError(f"Round 21 {name} differs")
    return parsed


@dataclass(frozen=True, slots=True)
class _MarketContext:
    condition: Round21CoreCondition
    market: PolymarketFiveMinuteMarket
    evidence: Round21MarketExecutionEvidence


@dataclass(frozen=True, slots=True)
class Round21DevelopmentEconomicResult:
    selected_population_layer: str
    terminal_transport_manifest_sha256: str
    core_publication_manifest_sha256: str
    model_artifact_sha256: str
    terminal_receipt_audit_sha256: str
    source_condition_set_sha256: str
    source_condition_count: int
    selected_matrix: tuple[Round21EconomicReplay, ...]
    optional_comparison: Round21MatchedEconomicComparison | None
    development_gate_passed: bool
    result_sha256: str
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def asdict(self) -> dict[str, object]:
        selected = self.validated()
        optional = selected.optional_comparison
        return {
            **selected.identity_payload(),
            "selected_matrix": [
                {
                    **replay.identity_payload(),
                    "metrics": {
                        **replay.metrics.identity_payload(),
                        "metric_sha256": replay.metrics.metric_sha256,
                    },
                    "replay_sha256": replay.replay_sha256,
                }
                for replay in selected.selected_matrix
            ],
            "optional_comparison": (
                None
                if optional is None
                else {
                    **optional.identity_payload(),
                    "deltas": [
                        {**value.identity_payload(), "delta_sha256": value.delta_sha256}
                        for value in optional.deltas
                    ],
                    "comparison_sha256": optional.comparison_sha256,
                }
            ),
            "result_sha256": selected.result_sha256,
        }

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_DEVELOPMENT_ECONOMIC_SCHEMA_VERSION,
            "selected_population_layer": self.selected_population_layer,
            "terminal_transport_manifest_sha256": (
                self.terminal_transport_manifest_sha256
            ),
            "core_publication_manifest_sha256": (self.core_publication_manifest_sha256),
            "model_artifact_sha256": self.model_artifact_sha256,
            "terminal_receipt_audit_sha256": self.terminal_receipt_audit_sha256,
            "source_condition_set_sha256": self.source_condition_set_sha256,
            "source_condition_count": self.source_condition_count,
            "selected_matrix_sha256": round21_replay_matrix_sha256(
                self.selected_matrix
            ),
            "optional_comparison_sha256": (
                None
                if self.optional_comparison is None
                else self.optional_comparison.comparison_sha256
            ),
            "development_gate_passed": self.development_gate_passed,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self) -> Round21DevelopmentEconomicResult:
        matrix = tuple(value.validated() for value in self.selected_matrix)
        optional = (
            None
            if self.optional_comparison is None
            else self.optional_comparison.validated()
        )
        hashes = (
            self.terminal_transport_manifest_sha256,
            self.core_publication_manifest_sha256,
            self.model_artifact_sha256,
            self.terminal_receipt_audit_sha256,
            self.source_condition_set_sha256,
        )
        if (
            self.selected_population_layer not in _LAYERS
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in hashes
            )
            or self.source_condition_count <= 0
            or len(matrix) != 81
            or len({(value.profile, value.scenario) for value in matrix}) != 81
            or any(
                value.metrics.condition_count != self.source_condition_count
                for value in matrix
            )
            or (self.selected_population_layer == "core") != (optional is None)
            or (
                optional is not None
                and (
                    optional.challenger_layer != self.selected_population_layer
                    or optional.challenger_matrix_sha256
                    != round21_replay_matrix_sha256(matrix)
                )
            )
            or self.development_gate_passed
            != (
                all(value.economic_gate_passed for value in matrix)
                and (optional is None or optional.all_replays_accepted)
            )
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 development economic result differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21SealedEconomicReplayEvidence:
    """Claim-bound output from the shared exact-receipt replay engine."""

    claim_sha256: str
    test_access_sha256: str
    sealed_test_population_manifest_sha256: str
    test_dataset_sha256: str
    test_target_manifest_sha256: str
    replay_result: Round21DevelopmentEconomicResult
    historical_ai_cases: tuple[PolymarketAIVetoCase, ...]
    evidence_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_SEALED_REPLAY_SCHEMA_VERSION,
            "claim_sha256": self.claim_sha256,
            "test_access_sha256": self.test_access_sha256,
            "sealed_test_population_manifest_sha256": (
                self.sealed_test_population_manifest_sha256
            ),
            "test_dataset_sha256": self.test_dataset_sha256,
            "test_target_manifest_sha256": self.test_target_manifest_sha256,
            "replay_result_sha256": self.replay_result.result_sha256,
            "historical_ai_case_sha256": [
                value.case_sha256 for value in self.historical_ai_cases
            ],
            "test_targets_accessed": True,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self) -> Round21SealedEconomicReplayEvidence:
        replay = self.replay_result.validated()
        cases = tuple(self.historical_ai_cases)
        if (
            any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.claim_sha256,
                    self.test_access_sha256,
                    self.sealed_test_population_manifest_sha256,
                    self.test_dataset_sha256,
                    self.test_target_manifest_sha256,
                    self.evidence_sha256,
                )
            )
            or len(cases) != replay.source_condition_count
            or len({value.condition_id for value in cases}) != len(cases)
            or any(
                _SHA256.fullmatch(value.case_sha256) is None
                or value.case_sha256 == _EMPTY_SHA256
                for value in cases
            )
            or self.evidence_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 sealed economic replay evidence differs")
        return self


def _validated_panels(
    panels: Sequence[Round21DevelopmentPanel],
    policy: Round21PartitionPolicy,
    *,
    expected_roles: Sequence[str],
) -> tuple[Round21DevelopmentPanel, ...]:
    roles = tuple(str(value) for value in expected_roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("Round 21 economic role contract differs")
    by_role: dict[str, Round21DevelopmentPanel] = {}
    for value in panels:
        panel = value.validate()
        if panel.role in by_role:
            raise ValueError("Round 21 development economic role is duplicated")
        for event_start in np.unique(panel.event_start_ms):
            if policy.role_for_event_start(int(event_start)) != panel.role:
                raise ValueError("Round 21 development economic role differs")
        by_role[panel.role] = panel
    if set(by_role) != set(roles):
        raise ValueError("Round 21 economic roles are incomplete")
    selected = tuple(by_role[role] for role in roles)
    condition_ids = [
        str(condition_id)
        for panel in selected
        for condition_id in np.unique(panel.condition_ids)
    ]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("Round 21 development economic conditions are duplicated")
    return selected


def _verify_artifact_panel_identity(
    *,
    artifact: Mapping[str, object],
    panels: Sequence[Round21DevelopmentPanel],
) -> None:
    identity = artifact.get("dataset_and_partition")
    if not isinstance(identity, Mapping):
        raise ValueError("Round 21 model dataset identity is unavailable")
    first = panels[0]
    if (
        identity.get("core_feature_names_sha256") != first.core_feature_names_sha256
        or identity.get("spot_feature_names_sha256") != first.spot_feature_names_sha256
        or identity.get("usdm_feature_names_sha256") != first.usdm_feature_names_sha256
    ):
        raise ValueError("Round 21 model feature identity differs")
    for panel in panels:
        expected = {
            "role": panel.role,
            "row_count": len(panel.labels),
            "condition_count": len(np.unique(panel.condition_ids)),
            "first_event_start_ms": int(panel.event_start_ms[0]),
            "last_event_start_ms": int(panel.event_start_ms[-1]),
            "dataset_sha256": panel.dataset_sha256,
            "target_manifest_sha256": panel.target_manifest_sha256,
            "dataset_design_sha256": panel.dataset_design_sha256,
        }
        if identity.get(panel.role) != expected:
            raise ValueError("Round 21 model development population differs")


def _group_envelopes(
    batch: Round21ProbabilityBatch,
    panel: Round21InferencePanel,
) -> dict[str, tuple[Round21ProbabilityEnvelope, ...]]:
    grouped: dict[str, list[Round21ProbabilityEnvelope]] = {}
    for envelope in build_round21_probability_envelopes(batch=batch, panel=panel):
        grouped.setdefault(envelope.condition_id, []).append(envelope)
    return {key: tuple(value) for key, value in grouped.items()}


def _merge_envelope_groups(
    destination: dict[str, tuple[Round21ProbabilityEnvelope, ...]],
    source: Mapping[str, tuple[Round21ProbabilityEnvelope, ...]],
) -> None:
    if set(destination) & set(source):
        raise ValueError("Round 21 probability conditions are duplicated")
    destination.update(source)


def _prepare_envelopes(
    *,
    panels: Sequence[Round21DevelopmentPanel],
    artifact: Mapping[str, object],
    selected_layer: str,
    expected_roles: Sequence[str],
) -> tuple[
    dict[str, tuple[Round21ProbabilityEnvelope, ...]],
    dict[str, tuple[Round21ProbabilityEnvelope, ...]] | None,
    dict[str, tuple[Round21InferencePanel, Round21ProbabilityBatch]],
]:
    baseline: dict[str, tuple[Round21ProbabilityEnvelope, ...]] = {}
    challenger: dict[str, tuple[Round21ProbabilityEnvelope, ...]] | None = (
        None if selected_layer == "core" else {}
    )
    baseline_predictor = (
        compile_round21_probability_predictor(artifact, population_layer="core")
        if selected_layer == "core"
        else compile_round21_matched_core_predictor(
            artifact,
            optional_population_layer=selected_layer,
        )
    )
    challenger_predictor = (
        None
        if selected_layer == "core"
        else compile_round21_probability_predictor(
            artifact,
            population_layer=selected_layer,
        )
    )
    selected_probability_inputs: dict[
        str,
        tuple[Round21InferencePanel, Round21ProbabilityBatch],
    ] = {}
    for panel in panels:
        inference = Round21InferencePanel.from_development(panel)
        baseline_batch = baseline_predictor.predict(inference)
        _merge_envelope_groups(
            baseline,
            _group_envelopes(baseline_batch, inference),
        )
        if challenger is not None and challenger_predictor is not None:
            challenger_batch = challenger_predictor.predict(inference)
            if not np.array_equal(baseline_batch.indices, challenger_batch.indices):
                raise ValueError("Round 21 matched probability population differs")
            _merge_envelope_groups(
                challenger,
                _group_envelopes(challenger_batch, inference),
            )
            selected_probability_inputs[panel.role] = (inference, challenger_batch)
        else:
            selected_probability_inputs[panel.role] = (inference, baseline_batch)
    if not baseline or (challenger is not None and set(baseline) != set(challenger)):
        raise ValueError("Round 21 probability condition population differs")
    if set(selected_probability_inputs) != set(expected_roles):
        raise ValueError("Round 21 probability role population differs")
    return baseline, challenger, selected_probability_inputs


def _load_market_contexts(
    *,
    database: str | Path,
    conditions: Sequence[Round21CoreCondition],
) -> dict[str, _MarketContext]:
    expected = {value.condition_id: value.validated() for value in conditions}
    output: dict[str, _MarketContext] = {}
    with PolymarketEvidenceStore(
        Path(database),
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        rows = (
            store.connect()
            .execute(
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
            )
            .fetchall()
        )
    for row in rows:
        condition_id = str(row[1])
        condition = expected.get(condition_id)
        if condition is None:
            continue
        gamma = _strict_json(row[4], name="Gamma market payload")
        clob = _strict_json(row[6], name="CLOB market payload")
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
            raise ValueError("Round 21 market execution context differs")
        output[condition_id] = _MarketContext(condition, market, evidence)
    if set(output) != set(expected):
        raise ValueError("Round 21 market execution context is incomplete")
    return output


def _gap_payload(gap: StreamGap) -> dict[str, object]:
    selected = gap.validated()
    return {
        "stream": selected.stream,
        "connection_id": selected.connection_id,
        "opened_at_ms": selected.opened_at_ms,
        "reason": selected.reason,
        "last_sequence_number": selected.last_sequence_number,
    }


class _Round21EconomicObserver:
    def __init__(
        self,
        *,
        contexts: Mapping[str, _MarketContext],
        outcomes: Mapping[str, Round21OfficialOutcome],
        baseline_envelopes: Mapping[str, tuple[Round21ProbabilityEnvelope, ...]],
        challenger_envelopes: Mapping[str, tuple[Round21ProbabilityEnvelope, ...]]
        | None,
        terminal_manifest_sha256: str,
        core_publication_manifest_sha256: str,
        baseline_accumulator: Round21EconomicMatrixAccumulator,
        challenger_accumulator: Round21EconomicMatrixAccumulator | None,
        selected_condition_sinks: Sequence[SelectedConditionSink] = (),
        progress: ProgressCallback | None = None,
    ) -> None:
        expected = set(contexts)
        if (
            expected != set(outcomes)
            or expected != set(baseline_envelopes)
            or (challenger_envelopes is None) != (challenger_accumulator is None)
            or (
                challenger_envelopes is not None
                and expected != set(challenger_envelopes)
            )
        ):
            raise ValueError("Round 21 economic observer population differs")
        self.contexts = dict(contexts)
        self.outcomes = dict(outcomes)
        self.baseline_envelopes = dict(baseline_envelopes)
        self.challenger_envelopes = (
            None if challenger_envelopes is None else dict(challenger_envelopes)
        )
        self.terminal_manifest_sha256 = terminal_manifest_sha256
        self.core_publication_manifest_sha256 = core_publication_manifest_sha256
        self.baseline_accumulator = baseline_accumulator
        self.challenger_accumulator = challenger_accumulator
        self.selected_condition_sinks = tuple(selected_condition_sinks)
        self.progress = progress
        self.observed_condition_ids: list[str] = []
        self.matched_condition_sha256: list[str] = []

    def observe(
        self,
        condition: Round21CoreCondition,
        source: Round21ConditionSource,
    ) -> None:
        selected_condition = condition.validated()
        selected_source = source.validated()
        condition_id = selected_condition.condition_id
        context = self.contexts.get(condition_id)
        if context is None or condition_id in self.observed_condition_ids:
            raise ValueError("Round 21 economic source condition differs")
        rebuilt = build_round21_execution_books(
            condition_id=condition_id,
            up_token_id=selected_condition.up_token_id,
            down_token_id=selected_condition.down_token_id,
            union_events=selected_source.union_events,
            admitted_gap_free=True,
        )
        books = tuple(
            book
            for book in rebuilt
            if selected_condition.event_start_ms
            - POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
            <= book.received_wall_ms
            < selected_condition.event_end_ms
        )
        reconciliation_sha = _canonical_sha256(
            {
                "schema_version": (
                    POLYMARKET_ROUND21_DEVELOPMENT_ECONOMIC_SCHEMA_VERSION
                ),
                "terminal_transport_manifest_sha256": (self.terminal_manifest_sha256),
                "core_publication_manifest_sha256": (
                    self.core_publication_manifest_sha256
                ),
                "condition": {
                    "run_id": selected_condition.run_id,
                    "segment_index": selected_condition.segment_index,
                    "condition_id": condition_id,
                    "event_start_ms": selected_condition.event_start_ms,
                    "event_end_ms": selected_condition.event_end_ms,
                    "snapshot_sha256": selected_condition.snapshot_sha256,
                },
                "market": context.market.asdict(),
                "market_execution_evidence_sha256": (context.evidence.evidence_sha256),
                "union_event_sha256": [
                    event.event_sha256 for event in selected_source.union_events
                ],
                "execution_book_identity": [
                    {
                        "asset_id": book.asset_id,
                        "received_wall_ms": book.received_wall_ms,
                        "received_monotonic_ns": book.received_monotonic_ns,
                        "source_payload_sha256": book.source_payload_sha256,
                    }
                    for book in books
                ],
                "stream_gaps": [
                    _gap_payload(gap) for gap in selected_source.stream_gaps
                ],
                "outcomes_consulted": False,
                "model_scores_consulted": False,
                "trading_authority": False,
            }
        )
        baseline = Round21ReplayCondition.create(
            market=context.market,
            market_evidence=context.evidence,
            envelopes=self.baseline_envelopes[condition_id],
            books=books,
            outcome=self.outcomes[condition_id],
            source_manifest_sha256=self.terminal_manifest_sha256,
            reconciliation_sha256=reconciliation_sha,
        )
        self.baseline_accumulator.observe(baseline)
        selected_for_sinks = baseline
        if self.challenger_envelopes is not None:
            if self.challenger_accumulator is None:
                raise AssertionError("Round 21 challenger accumulator is unavailable")
            challenger = Round21ReplayCondition.create(
                market=context.market,
                market_evidence=context.evidence,
                envelopes=self.challenger_envelopes[condition_id],
                books=books,
                outcome=self.outcomes[condition_id],
                source_manifest_sha256=self.terminal_manifest_sha256,
                reconciliation_sha256=reconciliation_sha,
            )
            if baseline.matched_population_sha256() != (
                challenger.matched_population_sha256()
            ):
                raise ValueError("Round 21 matched economic path differs")
            self.challenger_accumulator.observe(challenger)
            self.matched_condition_sha256.append(baseline.matched_population_sha256())
            selected_for_sinks = challenger
        for sink in self.selected_condition_sinks:
            sink(selected_for_sinks)
        self.observed_condition_ids.append(condition_id)
        if self.progress is not None:
            self.progress(
                "condition_replayed",
                {
                    "condition_count": len(self.observed_condition_ids),
                    "condition_id": condition_id,
                    "event_start_ms": selected_condition.event_start_ms,
                },
            )

    def finish(self) -> None:
        if set(self.observed_condition_ids) != set(self.contexts):
            raise ValueError("Round 21 economic source population is incomplete")


def _verify_panel_outcomes(
    panels: Sequence[Round21DevelopmentPanel],
    outcomes: Mapping[str, Round21OfficialOutcome],
) -> None:
    labels: dict[str, tuple[int, bool]] = {}
    expected = set(outcomes)
    for panel in panels:
        for condition_id in np.unique(panel.condition_ids):
            normalized = str(condition_id)
            if normalized not in expected:
                continue
            indices = np.flatnonzero(panel.condition_ids == condition_id)
            label_values = np.unique(panel.labels[indices])
            event_starts = np.unique(panel.event_start_ms[indices])
            if len(label_values) != 1 or len(event_starts) != 1:
                raise ValueError("Round 21 development target population differs")
            labels[normalized] = (
                int(event_starts[0]),
                bool(label_values[0]),
            )
    if set(labels) != expected or any(
        outcome.event_start_ms != labels[condition_id][0]
        or outcome.resolved_up != labels[condition_id][1]
        for condition_id, outcome in outcomes.items()
    ):
        raise ValueError("Round 21 official outcomes and development targets differ")


def _replay_round21_development_economics(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partition_policy: Round21PartitionPolicy,
    development_panels: Sequence[Round21DevelopmentPanel],
    development_model_artifact: Mapping[str, object],
    core_publication_manifest_sha256: str,
    selected_population_layer: str = "core",
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    selected_condition_sinks: Sequence[SelectedConditionSink] = (),
    collect_historical_ai_cases: bool,
    expected_roles: Sequence[str] = _DEVELOPMENT_ROLES,
    verify_artifact_population: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[Round21DevelopmentEconomicResult, tuple[PolymarketAIVetoCase, ...]]:
    """Replay all development conditions once; never grant trading authority."""

    transport = validate_round21_terminal_transport_manifest(
        terminal_transport_manifest
    )
    policy = partition_policy.validated()
    artifact = validate_round21_development_artifact(development_model_artifact)
    publication_sha = str(core_publication_manifest_sha256 or "").strip().lower()
    selected_layer = str(selected_population_layer or "").strip()
    if (
        _SHA256.fullmatch(publication_sha) is None
        or publication_sha == _EMPTY_SHA256
        or selected_layer not in _LAYERS
    ):
        raise ValueError("Round 21 development economic identity differs")
    roles = tuple(str(value) for value in expected_roles)
    panels = _validated_panels(
        development_panels,
        policy,
        expected_roles=roles,
    )
    if verify_artifact_population:
        _verify_artifact_panel_identity(artifact=artifact, panels=panels)
    baseline_envelopes, challenger_envelopes, selected_probability_inputs = (
        _prepare_envelopes(
            panels=panels,
            artifact=artifact,
            selected_layer=selected_layer,
            expected_roles=roles,
        )
    )
    ai_case_collector = (
        Round21HistoricalAICaseCollector(
            tuple(
                Round21HistoricalAICaseFactory(
                    panel=selected_probability_inputs[role][0],
                    probability_batch=selected_probability_inputs[role][1],
                )
                for role in roles
            )
        )
        if collect_historical_ai_cases
        else None
    )
    target_ids = set(baseline_envelopes)
    all_conditions = load_round21_core_conditions(
        database=source_database,
        terminal_transport_manifest=transport,
    )
    conditions = tuple(
        condition
        for condition in all_conditions
        if condition.condition_id in target_ids
    )
    if {value.condition_id for value in conditions} != target_ids:
        raise ValueError("Round 21 development economic source is incomplete")
    contexts = _load_market_contexts(
        database=source_database,
        conditions=conditions,
    )
    outcomes = {
        value.condition_id: value
        for value in load_round21_official_outcomes(
            source_database=source_database,
            terminal_transport_manifest=transport,
            condition_event_starts={
                value.condition_id: value.event_start_ms for value in conditions
            },
        )
    }
    _verify_panel_outcomes(panels, outcomes)
    if progress is not None:
        progress(
            "prepared",
            {
                "selected_population_layer": selected_layer,
                "condition_count": len(conditions),
                "ledger_count": 81,
            },
        )
    baseline_accumulator = Round21EconomicMatrixAccumulator(
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
    )
    challenger_accumulator = (
        None
        if challenger_envelopes is None
        else Round21EconomicMatrixAccumulator(
            initial_capital_quote=initial_capital_quote,
            minimum_edge_per_share=minimum_edge_per_share,
            builder_taker_fee_bps=builder_taker_fee_bps,
        )
    )
    economic_observer = _Round21EconomicObserver(
        contexts=contexts,
        outcomes=outcomes,
        baseline_envelopes=baseline_envelopes,
        challenger_envelopes=challenger_envelopes,
        terminal_manifest_sha256=str(transport["manifest_sha256"]),
        core_publication_manifest_sha256=publication_sha,
        baseline_accumulator=baseline_accumulator,
        challenger_accumulator=challenger_accumulator,
        selected_condition_sinks=(
            *selected_condition_sinks,
            *((ai_case_collector,) if ai_case_collector is not None else ()),
        ),
        progress=progress,
    )
    receipt_observer = Round21CoreCorpusObserver(
        conditions=conditions,
        partition_policy=policy,
        sink=None,
        source_sink=economic_observer.observe,
    )
    receipt_audit = audit_round21_terminal_receipts(
        database=source_database,
        terminal_transport_manifest=transport,
        observer=receipt_observer,
    )
    if progress is not None:
        progress(
            "receipts_audited",
            {
                "condition_count": len(economic_observer.observed_condition_ids),
                "receipt_audit_sha256": receipt_audit["audit_sha256"],
            },
        )
    economic_observer.finish()
    baseline_matrix = baseline_accumulator.finish()
    optional_comparison: Round21MatchedEconomicComparison | None = None
    if challenger_accumulator is None:
        selected_matrix = baseline_matrix
    else:
        selected_matrix = challenger_accumulator.finish()
        matched_population_sha = _canonical_sha256(
            {
                "schema_version": (
                    POLYMARKET_ROUND21_MATCHED_COMPARISON_SCHEMA_VERSION
                ),
                "condition_sha256": economic_observer.matched_condition_sha256,
            }
        )
        optional_comparison = compare_round21_optional_replay_matrices(
            baseline_matrix=baseline_matrix,
            challenger_matrix=selected_matrix,
            challenger_layer=selected_layer,
            matched_population_sha256=matched_population_sha,
        )
    source_condition_set_sha = _canonical_sha256(
        [
            {
                "condition_id": value.condition_id,
                "event_start_ms": value.event_start_ms,
                "snapshot_sha256": value.snapshot_sha256,
            }
            for value in conditions
        ]
    )
    provisional = Round21DevelopmentEconomicResult(
        selected_population_layer=selected_layer,
        terminal_transport_manifest_sha256=str(transport["manifest_sha256"]),
        core_publication_manifest_sha256=publication_sha,
        model_artifact_sha256=str(artifact["artifact_sha256"]),
        terminal_receipt_audit_sha256=str(receipt_audit["audit_sha256"]),
        source_condition_set_sha256=source_condition_set_sha,
        source_condition_count=len(conditions),
        selected_matrix=selected_matrix,
        optional_comparison=optional_comparison,
        development_gate_passed=(
            all(value.economic_gate_passed for value in selected_matrix)
            and (
                optional_comparison is None or optional_comparison.all_replays_accepted
            )
        ),
        result_sha256=_EMPTY_SHA256,
    )
    result = replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()
    cases = () if ai_case_collector is None else ai_case_collector.finish()
    return result, cases


def replay_round21_development_economics(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partition_policy: Round21PartitionPolicy,
    development_panels: Sequence[Round21DevelopmentPanel],
    development_model_artifact: Mapping[str, object],
    core_publication_manifest_sha256: str,
    selected_population_layer: str = "core",
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    selected_condition_sinks: Sequence[SelectedConditionSink] = (),
    progress: ProgressCallback | None = None,
) -> Round21DevelopmentEconomicResult:
    """Replay all development conditions once; never grant trading authority."""

    result, _cases = _replay_round21_development_economics(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=partition_policy,
        development_panels=development_panels,
        development_model_artifact=development_model_artifact,
        core_publication_manifest_sha256=core_publication_manifest_sha256,
        selected_population_layer=selected_population_layer,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        selected_condition_sinks=selected_condition_sinks,
        collect_historical_ai_cases=False,
        progress=progress,
    )
    return result


def replay_round21_development_economics_with_ai_cases(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partition_policy: Round21PartitionPolicy,
    development_panels: Sequence[Round21DevelopmentPanel],
    development_model_artifact: Mapping[str, object],
    core_publication_manifest_sha256: str,
    selected_population_layer: str = "core",
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    progress: ProgressCallback | None = None,
) -> tuple[Round21DevelopmentEconomicResult, tuple[PolymarketAIVetoCase, ...]]:
    """Replay economics and build target-free historical AI cases in one pass."""

    return _replay_round21_development_economics(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=partition_policy,
        development_panels=development_panels,
        development_model_artifact=development_model_artifact,
        core_publication_manifest_sha256=core_publication_manifest_sha256,
        selected_population_layer=selected_population_layer,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        selected_condition_sinks=(),
        collect_historical_ai_cases=True,
        progress=progress,
    )


def replay_round21_sealed_economics_with_ai_cases(
    *,
    source_database: str | Path,
    terminal_transport_manifest: Mapping[str, object],
    partition_policy: Round21PartitionPolicy,
    test_panel: Round21DevelopmentPanel,
    development_model_artifact: Mapping[str, object],
    core_publication_manifest_sha256: str,
    claim_sha256: str,
    test_access_sha256: str,
    sealed_test_population_manifest_sha256: str,
    selected_population_layer: str = "core",
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
    selected_condition_sinks: Sequence[SelectedConditionSink] = (),
    progress: ProgressCallback | None = None,
) -> Round21SealedEconomicReplayEvidence:
    """Replay the authorized test role without development identity shortcuts."""

    panel = test_panel.validate()
    if panel.role != "test":
        raise ValueError("Round 21 sealed economic panel role differs")
    replay, cases = _replay_round21_development_economics(
        source_database=source_database,
        terminal_transport_manifest=terminal_transport_manifest,
        partition_policy=partition_policy,
        development_panels=(panel,),
        development_model_artifact=development_model_artifact,
        core_publication_manifest_sha256=core_publication_manifest_sha256,
        selected_population_layer=selected_population_layer,
        initial_capital_quote=initial_capital_quote,
        minimum_edge_per_share=minimum_edge_per_share,
        builder_taker_fee_bps=builder_taker_fee_bps,
        selected_condition_sinks=selected_condition_sinks,
        collect_historical_ai_cases=True,
        expected_roles=("test",),
        verify_artifact_population=False,
        progress=progress,
    )
    provisional = Round21SealedEconomicReplayEvidence(
        claim_sha256=str(claim_sha256 or "").strip().lower(),
        test_access_sha256=str(test_access_sha256 or "").strip().lower(),
        sealed_test_population_manifest_sha256=str(
            sealed_test_population_manifest_sha256 or ""
        )
        .strip()
        .lower(),
        test_dataset_sha256=panel.dataset_sha256,
        test_target_manifest_sha256=panel.target_manifest_sha256,
        replay_result=replay,
        historical_ai_cases=cases,
        evidence_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        evidence_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_DEVELOPMENT_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SEALED_REPLAY_SCHEMA_VERSION",
    "Round21DevelopmentEconomicResult",
    "Round21SealedEconomicReplayEvidence",
    "SelectedConditionSink",
    "replay_round21_development_economics",
    "replay_round21_development_economics_with_ai_cases",
    "replay_round21_sealed_economics_with_ai_cases",
]
