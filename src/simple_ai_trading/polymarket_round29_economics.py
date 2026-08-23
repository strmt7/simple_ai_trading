"""Matched after-cost economics for the Round 29 primary pair."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np

from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_economics import (
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from .polymarket_round27_model import Round27ModelSample, Round27Partition
from .polymarket_round28_economics import paired_round28_economic_scenario
from .polymarket_round29_model import Round29ModelSample, Round29Partition
from .polymarket_round29_selection import load_round29_selected_pair


POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION = (
    "polymarket-round29-matched-economic-selection-v1"
)
POLYMARKET_ROUND29_PAIRED_SCENARIO_SCHEMA_VERSION = (
    "polymarket-round29-paired-economic-scenario-v1"
)
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
    "orders_submitted": False,
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


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


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 29 {name} SHA-256 differs")
    return selected


def project_round29_economic_partition(
    partition: Round29Partition,
) -> Round27Partition:
    """Project matched rows onto the frozen Round 27 execution interface."""

    selected = Round29Partition.from_samples(
        partition.samples,
        role=partition.role,
    )
    arrays = (
        "diagnostic_base_features",
        "diagnostic_augmented_features",
        "primary_base_features",
        "primary_augmented_features",
        "offsets",
        "targets",
        "weights",
        "conditions",
    )
    if any(
        not np.array_equal(getattr(partition, field), getattr(selected, field))
        for field in arrays
    ):
        raise RuntimeError("Round 29 economic projection differs")
    projected_samples = tuple(
        Round27ModelSample(
            slot_id=sample.slot_id,
            role=sample.role,
            condition_id=sample.condition_id,
            event_start_ms=sample.event_start_ms,
            decision_time_ms=sample.decision_time_ms,
            market_prior_probability=sample.market_prior_probability,
            values=sample.diagnostic_base_values,
            target_up=sample.target_up,
            condition_weight=sample.condition_weight,
            feature_row_sha256=sample.primary_feature_row_sha256,
        ).validated()
        for sample in selected.samples
    )
    projected = Round27Partition.from_samples(
        projected_samples,
        role=selected.role,
    )
    if (
        [sample.condition_id for sample in projected.samples]
        != [sample.condition_id for sample in selected.samples]
        or [sample.decision_time_ms for sample in projected.samples]
        != [sample.decision_time_ms for sample in selected.samples]
        or [sample.feature_row_sha256 for sample in projected.samples]
        != [sample.primary_feature_row_sha256 for sample in selected.samples]
        or not np.array_equal(projected.features, selected.diagnostic_base_features)
        or not np.array_equal(projected.offsets, selected.offsets)
        or not np.array_equal(projected.targets, selected.targets)
        or not np.array_equal(projected.weights, selected.weights)
        or not np.array_equal(projected.conditions, selected.conditions)
    ):
        raise RuntimeError("Round 29 economic projection differs")
    return projected


def paired_round29_economic_scenario(
    *,
    base: Mapping[str, object],
    augmented: Mapping[str, object],
    ordered_conditions: Sequence[str],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    inherited = paired_round28_economic_scenario(
        base=base,
        augmented=augmented,
        ordered_conditions=ordered_conditions,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=bootstrap_seed,
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_PAIRED_SCENARIO_SCHEMA_VERSION,
        "delay_ms": inherited["delay_ms"],
        "inherited_round28_matched_scenario": inherited,
        "scenario_uplift_gate_passed": inherited["scenario_uplift_gate_passed"],
    }
    body["paired_scenario_sha256"] = _canonical_sha256(body)
    return body


def _scenario_map(report: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Round 29 economic scenarios differ")
    selected: dict[int, Mapping[str, object]] = {}
    for raw_scenario in scenarios:
        if not isinstance(raw_scenario, Mapping):
            raise ValueError("Round 29 economic scenario differs")
        delay = raw_scenario.get("delay_ms")
        if type(delay) is not int or delay in selected:
            raise ValueError("Round 29 economic delay population differs")
        selected[delay] = raw_scenario
    return selected


def evaluate_round29_matched_economics(
    *,
    samples: Sequence[Round29ModelSample],
    selection_claim: Mapping[str, object],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    implementation_amendment_sha256: str,
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    source_audit_sha256: str,
    resolution_evidence_sha256: str,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batch_factory: Callable[[], Iterable[Round27EconomicBookBatch]] | None = None,
    config: Round27EconomicConfig | None = None,
) -> dict[str, object]:
    """Replay the promotion-controlling pair before any sealed access."""

    pair = load_round29_selected_pair(
        selection_claim,
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest_sha256=source_audit_sha256,
    )
    if pair is None:
        raise ValueError("Round 29 has no probability candidate for economics")
    if (books is None) == (book_batch_factory is None):
        raise ValueError("Round 29 economics requires exactly one book source")
    selected = Round29Partition.from_samples(samples, role="selection")
    projected = project_round29_economic_partition(selected)
    base_probability = pair.base_model.predict(
        selected.primary_base_features,
        selected.offsets,
    )
    augmented_probability = pair.augmented_model.predict(
        selected.primary_augmented_features,
        selected.offsets,
    )
    cfg = (config or Round27EconomicConfig()).validated()

    def evaluate(
        model: object,
        probability: np.ndarray,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = (
            {"books": books}
            if books is not None
            else {"book_batches": book_batch_factory()}
        )
        return evaluate_round27_economic_scenarios(
            partition=projected,
            predictions=probability,
            markets=markets,
            outcomes_up=outcomes_up,
            model_name=f"{model.model_name}:{model.feature_view}",  # type: ignore[attr-defined]
            model_sha256=model.model_sha256,  # type: ignore[attr-defined]
            source_audit_sha256=source_audit_sha256,
            resolution_evidence_sha256=resolution_evidence_sha256,
            config=cfg,
            **kwargs,
        )

    base_report = evaluate(pair.base_model, base_probability)
    augmented_report = evaluate(pair.augmented_model, augmented_probability)
    base_scenarios = _scenario_map(base_report)
    augmented_scenarios = _scenario_map(augmented_report)
    if set(base_scenarios) != set(augmented_scenarios) or set(base_scenarios) != set(
        cfg.delays_ms
    ):
        raise RuntimeError("Round 29 matched economic delays differ")
    market_by_condition = {
        market.condition_id: market
        for market in markets
        if market.condition_id in set(outcomes_up)
    }
    ordered_conditions = tuple(
        condition_id
        for condition_id, _market in sorted(
            market_by_condition.items(),
            key=lambda item: (item[1].event_start_ms, item[0]),
        )
    )
    if set(ordered_conditions) != set(outcomes_up):
        raise ValueError("Round 29 economic market order differs")
    paired_scenarios = [
        paired_round29_economic_scenario(
            base=base_scenarios[delay],
            augmented=augmented_scenarios[delay],
            ordered_conditions=ordered_conditions,
            bootstrap_draws=cfg.bootstrap_draws,
            bootstrap_seed=29_029,
        )
        for delay in cfg.delays_ms
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "selection",
        "round27_model_contract_sha256": _sha256(
            contract.get("contract_sha256"),
            name="model contract",
        ),
        "round29_preregistration_sha256": _sha256(
            preregistration.get("preregistration_sha256"),
            name="preregistration",
        ),
        "round29_implementation_amendment_sha256": _sha256(
            implementation_amendment_sha256,
            name="implementation amendment",
        ),
        "round29_selection_claim_sha256": _sha256(
            selection_claim.get("claim_sha256"),
            name="selection claim",
        ),
        "selected_model_family": pair.model_family,
        "base_model_sha256": pair.base_model.model_sha256,
        "augmented_model_sha256": pair.augmented_model.model_sha256,
        "source_audit_sha256": _sha256(source_audit_sha256, name="source audit"),
        "resolution_evidence_sha256": _sha256(
            resolution_evidence_sha256,
            name="resolution evidence",
        ),
        "condition_population_sha256": _canonical_sha256(list(ordered_conditions)),
        "base_economic_report": base_report,
        "augmented_economic_report": augmented_report,
        "paired_scenarios": paired_scenarios,
        "economic_uplift_gate_passed": bool(
            augmented_report["economic_edge_gate_passed"]
        )
        and all(item["scenario_uplift_gate_passed"] for item in paired_scenarios),
        "sealed_partition_accessed": False,
        "economic_metrics_computed": True,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


__all__ = [
    "POLYMARKET_ROUND29_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND29_PAIRED_SCENARIO_SCHEMA_VERSION",
    "evaluate_round29_matched_economics",
    "paired_round29_economic_scenario",
    "project_round29_economic_partition",
]
