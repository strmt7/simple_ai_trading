"""Matched after-cost economics for the Round 28 Binance BBO ablation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
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
from .polymarket_round27_model import (
    Round27ModelSample,
    Round27Partition,
    round27_stationary_bootstrap_mean_interval,
)
from .polymarket_round28_model import Round28ModelSample, Round28Partition
from .polymarket_round28_selection import load_round28_selected_pair


POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION = (
    "polymarket-round28-matched-economic-selection-v1"
)
POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION = (
    "polymarket-round28-paired-economic-scenario-v1"
)
_IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION = (
    "polymarket-round28-selection-implementation-amendment-v1"
)
_ECONOMIC_IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION = (
    "polymarket-round28-economic-implementation-amendment-v1"
)
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
    "orders_submitted": False,
}
_AMENDMENT_AUTHORITY = {
    "credentials_used": False,
    "edge_claim": False,
    "execution_connected": False,
    "live_trading_authority": False,
    "orders_submitted": False,
    "paper_trading_authority": False,
    "profitability_claim": False,
}
_FROZEN_KNOWLEDGE = {
    "ai_assist_economic_metrics_computed": False,
    "model_fitted_on_stage1": False,
    "official_outcomes_accessed": False,
    "performance_metrics_computed": False,
    "round27_stage1_feature_rows_accessed_or_materialized": False,
    "sealed_partition_accessed": False,
    "selection_partition_accessed": False,
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
        raise ValueError(f"Round 28 {name} SHA-256 differs")
    return selected


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 28 {name} differs")
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 28 {name} differs") from exc
    if not selected.is_finite():
        raise ValueError(f"Round 28 {name} differs")
    return selected


def _validated_implementation_amendment(
    value: Mapping[str, object],
    *,
    preregistration_sha256: str,
) -> dict[str, object]:
    amendment = dict(value)
    claimed = _sha256(
        amendment.pop("amendment_sha256", None),
        name="selection implementation amendment",
    )
    sources = amendment.get("source_text_sha256")
    authority = amendment.get("authority")
    knowledge = amendment.get("knowledge_at_freeze")
    if (
        claimed != _canonical_sha256(amendment)
        or amendment.get("schema_version") != _IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION
        or amendment.get("base_preregistration_sha256") != preregistration_sha256
        or amendment.get("status") != "frozen_before_stage1_feature_or_outcome_access"
        or not isinstance(sources, Mapping)
        or {
            "src/simple_ai_trading/polymarket_round28_model.py",
            "src/simple_ai_trading/polymarket_round28_selection.py",
        }
        - set(sources)
        or any(
            _sha256(source_hash, name="selection implementation source") != source_hash
            for source_hash in sources.values()
        )
        or not isinstance(authority, Mapping)
        or dict(authority) != _AMENDMENT_AUTHORITY
        or not isinstance(knowledge, Mapping)
        or dict(knowledge) != _FROZEN_KNOWLEDGE
    ):
        raise ValueError("Round 28 selection implementation amendment differs")
    return {**amendment, "amendment_sha256": claimed}


def _validated_economic_implementation_amendment(
    value: Mapping[str, object],
    *,
    preregistration_sha256: str,
    selection_implementation_amendment_sha256: str,
) -> dict[str, object]:
    amendment = dict(value)
    claimed = _sha256(
        amendment.pop("amendment_sha256", None),
        name="economic implementation amendment",
    )
    sources = amendment.get("source_text_sha256")
    authority = amendment.get("authority")
    knowledge = amendment.get("knowledge_at_freeze")
    if (
        claimed != _canonical_sha256(amendment)
        or amendment.get("schema_version")
        != _ECONOMIC_IMPLEMENTATION_AMENDMENT_SCHEMA_VERSION
        or amendment.get("base_preregistration_sha256") != preregistration_sha256
        or amendment.get("selection_implementation_amendment_sha256")
        != selection_implementation_amendment_sha256
        or amendment.get("status") != "frozen_before_stage1_feature_or_outcome_access"
        or not isinstance(sources, Mapping)
        or {
            "src/simple_ai_trading/polymarket_round28_economics.py",
            "tests/test_polymarket_round28_economics.py",
        }
        - set(sources)
        or any(
            _sha256(source_hash, name="economic implementation source") != source_hash
            for source_hash in sources.values()
        )
        or not isinstance(authority, Mapping)
        or dict(authority) != _AMENDMENT_AUTHORITY
        or not isinstance(knowledge, Mapping)
        or dict(knowledge) != _FROZEN_KNOWLEDGE
    ):
        raise ValueError("Round 28 economic implementation amendment differs")
    return {**amendment, "amendment_sha256": claimed}


def project_round28_economic_partition(
    partition: Round28Partition,
) -> Round27Partition:
    """Project matched samples onto the frozen Round 27 execution interface."""

    selected = Round28Partition.from_samples(
        partition.samples,
        role=partition.role,
    )
    if (
        not np.array_equal(partition.base_features, selected.base_features)
        or not np.array_equal(
            partition.augmented_features,
            selected.augmented_features,
        )
        or not np.array_equal(partition.offsets, selected.offsets)
        or not np.array_equal(partition.targets, selected.targets)
        or not np.array_equal(partition.weights, selected.weights)
        or not np.array_equal(partition.conditions, selected.conditions)
    ):
        raise RuntimeError("Round 28 economic projection differs")
    projected_samples = tuple(
        Round27ModelSample(
            slot_id=sample.slot_id,
            role=sample.role,
            condition_id=sample.condition_id,
            event_start_ms=sample.event_start_ms,
            decision_time_ms=sample.decision_time_ms,
            market_prior_probability=sample.market_prior_probability,
            values=sample.base_values,
            target_up=sample.target_up,
            condition_weight=sample.condition_weight,
            feature_row_sha256=sample.feature_row_sha256,
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
        != [sample.feature_row_sha256 for sample in selected.samples]
        or not np.array_equal(projected.features, selected.base_features)
        or not np.array_equal(projected.offsets, selected.offsets)
        or not np.array_equal(projected.targets, selected.targets)
        or not np.array_equal(projected.weights, selected.weights)
        or not np.array_equal(projected.conditions, selected.conditions)
    ):
        raise RuntimeError("Round 28 economic projection differs")
    return projected


def _trade_pnl_by_condition(
    scenario: Mapping[str, object],
    *,
    conditions: frozenset[str],
) -> dict[str, Decimal]:
    trades = scenario.get("trades")
    if not isinstance(trades, list):
        raise ValueError("Round 28 economic trade population differs")
    selected: dict[str, Decimal] = {}
    for raw_trade in trades:
        if not isinstance(raw_trade, Mapping):
            raise ValueError("Round 28 economic trade differs")
        condition_id = str(raw_trade.get("condition_id", ""))
        execution_state = raw_trade.get("execution_state")
        if (
            condition_id not in conditions
            or condition_id in selected
            or execution_state not in {"FILLED", "NO_FILL", "UNKNOWN"}
        ):
            raise ValueError("Round 28 economic trade identity differs")
        selected[condition_id] = (
            _decimal(raw_trade.get("net_pnl_quote"), name="trade net PnL")
            if execution_state == "FILLED"
            else Decimal("0")
        )
    return selected


def _validated_scenario(value: Mapping[str, object]) -> dict[str, object]:
    scenario = dict(value)
    claimed = _sha256(
        scenario.pop("scenario_sha256", None),
        name="economic scenario",
    )
    if claimed != _canonical_sha256(scenario):
        raise ValueError("Round 28 economic scenario hash differs")
    return {**scenario, "scenario_sha256": claimed}


def paired_round28_economic_scenario(
    *,
    base: Mapping[str, object],
    augmented: Mapping[str, object],
    ordered_conditions: Sequence[str],
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Compare matched condition PnL, including zero for abstention or no fill."""

    selected_base = _validated_scenario(base)
    selected_augmented = _validated_scenario(augmented)
    conditions = tuple(str(value) for value in ordered_conditions)
    condition_set = frozenset(conditions)
    if (
        len(conditions) < 20
        or len(condition_set) != len(conditions)
        or any(not value.startswith("0x") for value in conditions)
        or type(selected_base.get("delay_ms")) is not int
        or selected_base.get("delay_ms") != selected_augmented.get("delay_ms")
        or selected_base.get("evaluated_condition_count") != len(conditions)
        or selected_augmented.get("evaluated_condition_count") != len(conditions)
    ):
        raise ValueError("Round 28 paired economic population differs")
    base_pnl = _trade_pnl_by_condition(selected_base, conditions=condition_set)
    augmented_pnl = _trade_pnl_by_condition(
        selected_augmented,
        conditions=condition_set,
    )
    deltas = np.asarray(
        [
            float(
                augmented_pnl.get(condition_id, Decimal("0"))
                - base_pnl.get(condition_id, Decimal("0"))
            )
            for condition_id in conditions
        ],
        dtype=np.float64,
    )
    bootstrap = round27_stationary_bootstrap_mean_interval(
        deltas,
        draws=bootstrap_draws,
        seed=bootstrap_seed + int(selected_base["delay_ms"]),
    )
    base_net_pnl = _decimal(
        selected_base.get("net_pnl_quote"),
        name="base net PnL",
    )
    augmented_net_pnl = _decimal(
        selected_augmented.get("net_pnl_quote"),
        name="augmented net PnL",
    )
    base_drawdown = _decimal(
        selected_base.get("maximum_drawdown_fraction"),
        name="base maximum drawdown",
    )
    augmented_drawdown = _decimal(
        selected_augmented.get("maximum_drawdown_fraction"),
        name="augmented maximum drawdown",
    )
    if (
        sum(base_pnl.values(), Decimal("0")) != base_net_pnl
        or sum(augmented_pnl.values(), Decimal("0")) != augmented_net_pnl
    ):
        raise ValueError("Round 28 economic scenario PnL differs")
    checks = {
        "augmented_round27_scenario_gate_passed": (
            selected_augmented.get("scenario_edge_gate_passed") is True
        ),
        "augmented_net_pnl_strictly_greater_than_base": (
            augmented_net_pnl > base_net_pnl
        ),
        "paired_mean_net_pnl_delta_positive": float(np.mean(deltas)) > 0.0,
        "paired_condition_bootstrap_lower_bound_positive": (
            float(bootstrap["ci95_lower"]) > 0.0
        ),
        "maximum_drawdown_not_worse_than_base": (augmented_drawdown <= base_drawdown),
    }
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
        "delay_ms": int(selected_base["delay_ms"]),
        "evaluated_condition_count": len(conditions),
        "condition_order_sha256": _canonical_sha256(list(conditions)),
        "base_scenario_sha256": _sha256(
            selected_base.get("scenario_sha256"),
            name="base scenario",
        ),
        "augmented_scenario_sha256": _sha256(
            selected_augmented.get("scenario_sha256"),
            name="augmented scenario",
        ),
        "base_net_pnl_quote": format(base_net_pnl, "f"),
        "augmented_net_pnl_quote": format(augmented_net_pnl, "f"),
        "net_pnl_delta_quote": format(augmented_net_pnl - base_net_pnl, "f"),
        "mean_condition_net_pnl_delta_quote": format(
            Decimal(str(float(np.mean(deltas)))),
            "f",
        ),
        "maximum_drawdown_delta_fraction": format(
            augmented_drawdown - base_drawdown,
            "f",
        ),
        "paired_condition_bootstrap": bootstrap,
        "gate_checks": checks,
        "scenario_uplift_gate_passed": all(checks.values()),
    }
    body["paired_scenario_sha256"] = _canonical_sha256(body)
    return body


def _scenario_map(report: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Round 28 economic scenarios differ")
    selected: dict[int, Mapping[str, object]] = {}
    for raw_scenario in scenarios:
        if not isinstance(raw_scenario, Mapping):
            raise ValueError("Round 28 economic scenario differs")
        delay = raw_scenario.get("delay_ms")
        if type(delay) is not int or delay in selected:
            raise ValueError("Round 28 economic delay population differs")
        selected[delay] = raw_scenario
    return selected


def evaluate_round28_matched_economics(
    *,
    samples: Sequence[Round28ModelSample],
    selection_claim: Mapping[str, object],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    implementation_amendment: Mapping[str, object],
    economic_implementation_amendment: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    source_audit_sha256: str,
    resolution_evidence_sha256: str,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batch_factory: Callable[[], Iterable[Round27EconomicBookBatch]] | None = None,
    config: Round27EconomicConfig | None = None,
) -> dict[str, object]:
    """Run matched base/augmented execution replays before any sealed access."""

    preregistration_sha256 = _sha256(
        preregistration.get("preregistration_sha256"),
        name="preregistration",
    )
    amendment = _validated_implementation_amendment(
        implementation_amendment,
        preregistration_sha256=preregistration_sha256,
    )
    economic_amendment = _validated_economic_implementation_amendment(
        economic_implementation_amendment,
        preregistration_sha256=preregistration_sha256,
        selection_implementation_amendment_sha256=str(amendment["amendment_sha256"]),
    )
    pair = load_round28_selected_pair(
        selection_claim,
        contract=contract,
        preregistration=preregistration,
    )
    if pair is None:
        raise ValueError("Round 28 has no probability candidate for economics")
    if (books is None) == (book_batch_factory is None):
        raise ValueError("Round 28 economics requires exactly one book source")
    selected = Round28Partition.from_samples(samples, role="selection")
    projected = project_round28_economic_partition(selected)
    base_probability = pair.base_model.predict(
        selected.base_features,
        selected.offsets,
    )
    augmented_probability = pair.augmented_model.predict(
        selected.augmented_features,
        selected.offsets,
    )
    cfg = (config or Round27EconomicConfig()).validated()

    def evaluate(model: object, probability: np.ndarray) -> dict[str, object]:
        kwargs: dict[str, object] = (
            {"books": books}
            if books is not None
            else {"book_batches": book_batch_factory()}
        )
        selected_model = model
        return evaluate_round27_economic_scenarios(
            partition=projected,
            predictions=probability,
            markets=markets,
            outcomes_up=outcomes_up,
            model_name=(f"{selected_model.model_name}:{selected_model.feature_view}"),
            model_sha256=selected_model.model_sha256,
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
        raise RuntimeError("Round 28 matched economic delays differ")
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
        raise ValueError("Round 28 economic market order differs")
    paired_scenarios = [
        paired_round28_economic_scenario(
            base=base_scenarios[delay],
            augmented=augmented_scenarios[delay],
            ordered_conditions=ordered_conditions,
            bootstrap_draws=cfg.bootstrap_draws,
            bootstrap_seed=28_028,
        )
        for delay in cfg.delays_ms
    ]
    selection_claim_sha256 = _sha256(
        selection_claim.get("claim_sha256"),
        name="selection claim",
    )
    contract_sha256 = _sha256(
        contract.get("contract_sha256"),
        name="model contract",
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "selection",
        "round27_model_contract_sha256": contract_sha256,
        "round28_preregistration_sha256": preregistration_sha256,
        "round28_selection_implementation_amendment_sha256": amendment[
            "amendment_sha256"
        ],
        "round28_economic_implementation_amendment_sha256": economic_amendment[
            "amendment_sha256"
        ],
        "round28_selection_claim_sha256": selection_claim_sha256,
        "selected_model_family": pair.model_family,
        "base_model_sha256": pair.base_model.model_sha256,
        "augmented_model_sha256": pair.augmented_model.model_sha256,
        "source_audit_sha256": _sha256(
            source_audit_sha256,
            name="source audit",
        ),
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
    "POLYMARKET_ROUND28_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION",
    "evaluate_round28_matched_economics",
    "paired_round28_economic_scenario",
    "project_round28_economic_partition",
]
