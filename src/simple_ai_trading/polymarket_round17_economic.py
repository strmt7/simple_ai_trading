"""Train/tune-only economic selection for Round 17 Polymarket actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from statistics import median
from typing import Mapping, Sequence

import numpy as np

from .polymarket_round14_contract import PolymarketRound14Program
from .polymarket_round17_execution import (
    POLYMARKET_ROUND17_EXECUTION_SCHEMA_VERSION,
)
from .polymarket_round17_features import POLYMARKET_ROUND17_CONTRACT_SHA256


POLYMARKET_ROUND17_ECONOMIC_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-economic-pretest-v1"
)
POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256 = (
    "216aaa1dda6abb8b3b74363f2ebcebace0b6b1938ead29166972b4d95a635eb3"
)
POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS = (
    Decimal("0.005"),
    Decimal("0.01"),
    Decimal("0.02"),
    Decimal("0.03"),
    Decimal("0.05"),
)
POLYMARKET_ROUND17_ECONOMIC_PATHS = (
    "settlement_directional",
    "intrawindow_owned_reprice",
    "complement_lock",
)
POLYMARKET_ROUND17_LOSS_CLUSTER_SIZE = 2
POLYMARKET_ROUND17_DEVELOPMENT_MINIMUM_CONDITIONS = 100
POLYMARKET_ROUND17_DEVELOPMENT_MINIMUM_ACTIONS = 30
POLYMARKET_ROUND17_DEVELOPMENT_MINIMUM_CALENDAR_DAYS = 3
POLYMARKET_ROUND17_BOOTSTRAP_SAMPLES = 2_000
POLYMARKET_ROUND17_BOOTSTRAP_SEED = 17_017
_PRIMARY_SCENARIO = "primary"
_SOURCE_PARTITION = "tune_economic"
_PROFILES = ("conservative", "regular", "aggressive")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DAY_MS = 86_400_000
_ROUND14_CONTRACT_SHA256 = (
    "60cde01112a749a9971447368b3a5d73b203d095e62a974327004c16cb021f1b"
)
_ROUND14_PROFILE_IDENTITY = (
    ("conservative", "0.001", "0.005", "0.02", 30),
    ("regular", "0.002", "0.01", "0.04", 15),
    ("aggressive", "0.0035", "0.015", "0.06", 5),
)
_ROUND14_SCENARIO_IDENTITY = (
    ("primary", 500, 0, "1", "1"),
    ("latency_250ms", 250, 0, "1", "1"),
    ("latency_750ms", 750, 0, "1", "1"),
    ("latency_1000ms", 1000, 0, "1", "1"),
    ("half_depth", 500, 0, "0.5", "1"),
    ("quarter_depth", 500, 0, "0.25", "1"),
    ("one_tick_adverse", 500, 1, "1", "1"),
    ("combined", 1000, 2, "0.5", "2"),
)
_SCENARIOS = (
    "primary",
    "latency_250ms",
    "latency_750ms",
    "latency_1000ms",
    "half_depth",
    "quarter_depth",
    "one_tick_adverse",
    "combined",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not selected.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return selected


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _validate_program(program: PolymarketRound14Program) -> None:
    profiles = tuple(
        (
            item.name,
            format(item.maximum_event_loss_capital_fraction, "f"),
            format(item.maximum_daily_loss_capital_fraction, "f"),
            format(item.maximum_drawdown_capital_fraction, "f"),
            item.cooldown_minutes_after_loss_cluster,
        )
        for item in program.risk_profiles
    )
    scenarios = tuple(
        (
            item.name,
            item.submission_latency_ms,
            item.adverse_ticks,
            format(item.displayed_depth_fraction, "f"),
            format(item.fee_multiplier, "f"),
        )
        for item in program.scenarios
    )
    if (
        program.contract_sha256 != _ROUND14_CONTRACT_SHA256
        or profiles != _ROUND14_PROFILE_IDENTITY
        or scenarios != _ROUND14_SCENARIO_IDENTITY
        or program.paper_authority
        or program.live_authority
    ):
        raise ValueError("Round 17 economic program differs from the frozen contract")


@dataclass(frozen=True, slots=True)
class Round17ConditionEconomicOutcome:
    source_partition: str
    condition_id: str
    event_start_ms: int
    path: str
    risk_profile: str
    scenario: str
    minimum_edge_quote_per_share: Decimal
    risk_capital_quote: Decimal
    entry_executed: bool
    realized_net_quote: Decimal
    maximum_loss_quote: Decimal
    unknown_state: bool
    lifecycle_violation: bool
    ownership_violation: bool
    decision_sha256: str
    source_evidence_sha256: str
    outcome_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-round17-condition-economic-outcome-v1",
            "execution_schema_version": (POLYMARKET_ROUND17_EXECUTION_SCHEMA_VERSION),
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "source_partition": self.source_partition,
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "path": self.path,
            "risk_profile": self.risk_profile,
            "scenario": self.scenario,
            "minimum_edge_quote_per_share": _decimal_text(
                self.minimum_edge_quote_per_share
            ),
            "risk_capital_quote": _decimal_text(self.risk_capital_quote),
            "entry_executed": self.entry_executed,
            "realized_net_quote": _decimal_text(self.realized_net_quote),
            "maximum_loss_quote": _decimal_text(self.maximum_loss_quote),
            "unknown_state": self.unknown_state,
            "lifecycle_violation": self.lifecycle_violation,
            "ownership_violation": self.ownership_violation,
            "decision_sha256": self.decision_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
        }

    def validated(self) -> "Round17ConditionEconomicOutcome":
        threshold = _decimal(
            self.minimum_edge_quote_per_share,
            name="minimum edge",
        )
        capital = _decimal(self.risk_capital_quote, name="risk capital")
        pnl = _decimal(self.realized_net_quote, name="realized net")
        maximum_loss = _decimal(self.maximum_loss_quote, name="maximum loss")
        if (
            self.source_partition != _SOURCE_PARTITION
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or self.path not in POLYMARKET_ROUND17_ECONOMIC_PATHS
            or self.risk_profile not in _PROFILES
            or self.scenario not in _SCENARIOS
            or threshold not in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS
            or capital <= 0
            or maximum_loss < 0
            or maximum_loss > capital
            or (not self.entry_executed and pnl != 0)
            or (not self.entry_executed and maximum_loss != 0)
            or (self.entry_executed and maximum_loss <= 0)
            or (self.entry_executed and pnl < -maximum_loss)
            or (self.unknown_state and not self.entry_executed)
            or (self.unknown_state and pnl != -maximum_loss)
            or _SHA256.fullmatch(self.decision_sha256) is None
            or _SHA256.fullmatch(self.source_evidence_sha256) is None
            or self.outcome_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 condition economic outcome is invalid")
        return replace(
            self,
            minimum_edge_quote_per_share=threshold,
            risk_capital_quote=capital,
            realized_net_quote=pnl,
            maximum_loss_quote=maximum_loss,
        )


def build_round17_condition_economic_outcome(
    *,
    condition_id: str,
    event_start_ms: int,
    path: str,
    risk_profile: str,
    scenario: str,
    minimum_edge_quote_per_share: Decimal,
    risk_capital_quote: Decimal,
    entry_executed: bool,
    realized_net_quote: Decimal,
    maximum_loss_quote: Decimal,
    unknown_state: bool,
    lifecycle_violation: bool,
    ownership_violation: bool,
    decision_sha256: str,
    source_evidence_sha256: str,
) -> Round17ConditionEconomicOutcome:
    provisional = Round17ConditionEconomicOutcome(
        source_partition=_SOURCE_PARTITION,
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        path=path,
        risk_profile=risk_profile,
        scenario=scenario,
        minimum_edge_quote_per_share=minimum_edge_quote_per_share,
        risk_capital_quote=risk_capital_quote,
        entry_executed=entry_executed,
        realized_net_quote=realized_net_quote,
        maximum_loss_quote=maximum_loss_quote,
        unknown_state=unknown_state,
        lifecycle_violation=lifecycle_violation,
        ownership_violation=ownership_violation,
        decision_sha256=decision_sha256,
        source_evidence_sha256=source_evidence_sha256,
    )
    return replace(
        provisional,
        outcome_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _profile(program: PolymarketRound14Program, name: str):
    return next(item for item in program.risk_profiles if item.name == name)


def _daily_block_lower_95(
    event_starts: Sequence[int],
    utilities: Sequence[Decimal],
) -> float:
    by_day: dict[int, list[float]] = {}
    for event_start, utility in zip(event_starts, utilities, strict=True):
        by_day.setdefault(event_start // _DAY_MS, []).append(float(utility))
    days = tuple(sorted(by_day))
    if not days:
        return 0.0
    totals = np.asarray(
        [sum(by_day[day]) for day in days],
        dtype=np.float64,
    )
    counts = np.asarray(
        [len(by_day[day]) for day in days],
        dtype=np.int64,
    )
    generator = np.random.default_rng(POLYMARKET_ROUND17_BOOTSTRAP_SEED)
    samples = np.empty(POLYMARKET_ROUND17_BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(POLYMARKET_ROUND17_BOOTSTRAP_SAMPLES):
        selected = generator.integers(0, len(days), size=len(days))
        samples[index] = float(np.sum(totals[selected]) / np.sum(counts[selected]))
    return float(np.quantile(samples, 0.025, method="linear"))


def _replay_policy(
    values: Sequence[Round17ConditionEconomicOutcome],
    program: PolymarketRound14Program,
) -> dict[str, object]:
    ordered = tuple(sorted(values, key=lambda item: item.event_start_ms))
    if not ordered:
        raise ValueError("Round 17 economic policy path is empty")
    profile = _profile(program, ordered[0].risk_profile)
    capital = ordered[0].risk_capital_quote
    daily_limit = capital * profile.maximum_daily_loss_capital_fraction
    drawdown_limit = profile.maximum_drawdown_capital_fraction
    cooldown_ms = profile.cooldown_minutes_after_loss_cluster * 60_000
    equity = capital
    peak = capital
    maximum_drawdown = Decimal("0")
    cooldown_until_ms = 0
    consecutive_losses = 0
    daily_pnl: dict[int, Decimal] = {}
    utilities: list[Decimal] = []
    event_starts: list[int] = []
    blocked: Counter[str] = Counter()
    executed = 0
    wins = 0
    gross_profit = Decimal("0")
    gross_loss = Decimal("0")
    maximum_observed_loss = Decimal("0")
    lifecycle_violations = 0
    ownership_violations = 0
    unknown_states = 0
    event_loss_limit_violations = 0

    for item in ordered:
        day = item.event_start_ms // _DAY_MS
        utility = Decimal("0")
        if not item.entry_executed:
            blocked["source_abstention"] += 1
        elif daily_pnl.get(day, Decimal("0")) <= -daily_limit:
            blocked["daily_loss_limit"] += 1
        elif item.event_start_ms < cooldown_until_ms:
            blocked["loss_cluster_cooldown"] += 1
        elif (peak - equity) / capital >= drawdown_limit:
            blocked["drawdown_limit"] += 1
        else:
            utility = item.realized_net_quote
            executed += 1
            lifecycle_violations += int(item.lifecycle_violation)
            ownership_violations += int(item.ownership_violation)
            unknown_states += int(item.unknown_state)
            event_loss_limit_violations += int(
                item.maximum_loss_quote
                > capital * profile.maximum_event_loss_capital_fraction
            )
            maximum_observed_loss = max(
                maximum_observed_loss,
                max(Decimal("0"), -utility),
                item.maximum_loss_quote if item.unknown_state else Decimal("0"),
            )
            if utility > 0:
                wins += 1
                gross_profit += utility
                consecutive_losses = 0
            elif utility < 0:
                gross_loss += -utility
                consecutive_losses += 1
                if consecutive_losses >= POLYMARKET_ROUND17_LOSS_CLUSTER_SIZE:
                    cooldown_until_ms = item.event_start_ms + 300_000 + cooldown_ms
                    consecutive_losses = 0
        utilities.append(utility)
        event_starts.append(item.event_start_ms)
        daily_pnl[day] = daily_pnl.get(day, Decimal("0")) + utility
        equity += utility
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, (peak - equity) / capital)

    net = sum(utilities, Decimal("0"))
    daily_values = tuple(daily_pnl[day] for day in sorted(daily_pnl))
    mean_utility = net / Decimal(len(utilities))
    profit_factor = None if gross_loss == 0 else gross_profit / gross_loss
    worst_daily_loss = max(
        (max(Decimal("0"), -value) for value in daily_values),
        default=Decimal("0"),
    )
    return {
        "condition_count": len(ordered),
        "calendar_day_count": len(daily_values),
        "executed_action_count": executed,
        "win_count": wins,
        "win_rate": 0.0 if executed == 0 else wins / executed,
        "net_pnl_quote": _decimal_text(net),
        "mean_event_utility_quote": _decimal_text(mean_utility),
        "median_daily_pnl_quote": _decimal_text(
            Decimal(str(median(daily_values))) if daily_values else Decimal("0")
        ),
        "gross_profit_quote": _decimal_text(gross_profit),
        "gross_loss_quote": _decimal_text(gross_loss),
        "profit_factor": (
            None if profit_factor is None else _decimal_text(profit_factor)
        ),
        "profit_factor_unbounded_without_observed_loss": (
            profit_factor is None and gross_profit > 0
        ),
        "maximum_drawdown_fraction": _decimal_text(maximum_drawdown),
        "worst_daily_loss_fraction": _decimal_text(worst_daily_loss / capital),
        "maximum_observed_loss_quote": _decimal_text(maximum_observed_loss),
        "daily_block_bootstrap_mean_event_utility_lower_95": (
            _daily_block_lower_95(event_starts, utilities)
        ),
        "blocked_counts": dict(sorted(blocked.items())),
        "lifecycle_violation_count": lifecycle_violations,
        "ownership_violation_count": ownership_violations,
        "unknown_state_count": unknown_states,
        "event_loss_limit_violation_count": event_loss_limit_violations,
        "risk_capital_quote": _decimal_text(capital),
        "reinvestment": False,
    }


def _scenario_passes(
    metrics: Mapping[str, object],
    program: PolymarketRound14Program,
    *,
    profile_name: str,
) -> bool:
    profile = _profile(program, profile_name)
    profit_factor = metrics["profit_factor"]
    profit_factor_positive = (
        bool(metrics["profit_factor_unbounded_without_observed_loss"])
        if profit_factor is None
        else Decimal(str(profit_factor)) > Decimal("1")
    )
    return bool(
        int(metrics["condition_count"])
        >= POLYMARKET_ROUND17_DEVELOPMENT_MINIMUM_CONDITIONS
        and int(metrics["calendar_day_count"])
        >= POLYMARKET_ROUND17_DEVELOPMENT_MINIMUM_CALENDAR_DAYS
        and int(metrics["executed_action_count"])
        >= POLYMARKET_ROUND17_DEVELOPMENT_MINIMUM_ACTIONS
        and Decimal(str(metrics["net_pnl_quote"])) > 0
        and Decimal(str(metrics["mean_event_utility_quote"])) > 0
        and Decimal(str(metrics["median_daily_pnl_quote"])) > 0
        and float(metrics["daily_block_bootstrap_mean_event_utility_lower_95"]) > 0
        and profit_factor_positive
        and Decimal(str(metrics["maximum_drawdown_fraction"]))
        <= profile.maximum_drawdown_capital_fraction
        and Decimal(str(metrics["worst_daily_loss_fraction"]))
        <= profile.maximum_daily_loss_capital_fraction
        and int(metrics["lifecycle_violation_count"]) == 0
        and int(metrics["ownership_violation_count"]) == 0
        and int(metrics["unknown_state_count"]) == 0
        and int(metrics["event_loss_limit_violation_count"]) == 0
    )


def fit_round17_economic_pretest(
    outcomes: Sequence[Round17ConditionEconomicOutcome],
    program: PolymarketRound14Program,
    *,
    model_pretest_sha256: str,
) -> dict[str, object]:
    """Select tune-only economic policies without accepting test-role input."""

    _validate_program(program)
    if _SHA256.fullmatch(str(model_pretest_sha256)) is None:
        raise ValueError("Round 17 model pretest identity is invalid")
    values = tuple(item.validated() for item in outcomes)
    if not values:
        raise ValueError("Round 17 economic outcomes are empty")
    if len({item.risk_capital_quote for item in values}) != 1:
        raise ValueError("Round 17 economic outcomes use different risk capital")
    grouped: dict[
        tuple[str, str, Decimal, str],
        list[Round17ConditionEconomicOutcome],
    ] = {}
    for item in values:
        key = (
            item.path,
            item.risk_profile,
            item.minimum_edge_quote_per_share,
            item.scenario,
        )
        grouped.setdefault(key, []).append(item)
    expected_keys = {
        (path, profile, threshold, scenario)
        for path in POLYMARKET_ROUND17_ECONOMIC_PATHS
        for profile in _PROFILES
        for threshold in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS
        for scenario in _SCENARIOS
    }
    if set(grouped) != expected_keys:
        raise ValueError("Round 17 economic policy grid is incomplete")
    reference_conditions: tuple[tuple[str, int], ...] | None = None
    for rows in grouped.values():
        identity = tuple(
            sorted((item.condition_id, item.event_start_ms) for item in rows)
        )
        if len(identity) != len(set(identity)):
            raise ValueError("Round 17 economic policy has duplicate conditions")
        if reference_conditions is None:
            reference_conditions = identity
        elif identity != reference_conditions:
            raise ValueError("Round 17 economic policy condition panels differ")

    candidate_ledger: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    for profile in _PROFILES:
        profile_candidates: list[dict[str, object]] = []
        for path in POLYMARKET_ROUND17_ECONOMIC_PATHS:
            for threshold in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS:
                scenario_metrics = {
                    scenario: _replay_policy(
                        grouped[(path, profile, threshold, scenario)],
                        program,
                    )
                    for scenario in _SCENARIOS
                }
                scenario_gates = {
                    scenario: _scenario_passes(
                        metrics,
                        program,
                        profile_name=profile,
                    )
                    for scenario, metrics in scenario_metrics.items()
                }
                candidate = {
                    "candidate_id": (
                        f"round17-economic-{profile}-{path}-edge-"
                        f"{format(threshold, 'f')}"
                    ),
                    "path": path,
                    "risk_profile": profile,
                    "minimum_edge_quote_per_share": format(threshold, "f"),
                    "scenario_metrics": scenario_metrics,
                    "scenario_gates": scenario_gates,
                    "development_accepted": all(scenario_gates.values()),
                }
                candidate_ledger.append(candidate)
                profile_candidates.append(candidate)

        def rank(item: Mapping[str, object]) -> tuple[object, ...]:
            metrics = item["scenario_metrics"]
            if not isinstance(metrics, Mapping):
                raise RuntimeError("Round 17 scenario metrics are invalid")
            lower_bounds = [
                float(value["daily_block_bootstrap_mean_event_utility_lower_95"])
                for value in metrics.values()
                if isinstance(value, Mapping)
            ]
            primary = metrics[_PRIMARY_SCENARIO]
            if not isinstance(primary, Mapping):
                raise RuntimeError("Round 17 primary metrics are invalid")
            threshold = Decimal(str(item["minimum_edge_quote_per_share"]))
            complexity = POLYMARKET_ROUND17_ECONOMIC_PATHS.index(str(item["path"]))
            return (
                bool(item["development_accepted"]),
                min(lower_bounds),
                Decimal(str(primary["net_pnl_quote"])),
                -Decimal(str(primary["maximum_drawdown_fraction"])),
                threshold,
                -complexity,
            )

        selected[profile] = dict(max(profile_candidates, key=rank))

    development_accepted = all(
        bool(value["development_accepted"]) for value in selected.values()
    )
    payload: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND17_ECONOMIC_SCHEMA_VERSION,
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "economic_contract_sha256": (POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256),
        "round14_risk_contract_sha256": program.contract_sha256,
        "model_pretest_sha256": model_pretest_sha256,
        "source_partition": _SOURCE_PARTITION,
        "condition_count": len(reference_conditions or ()),
        "paths": list(POLYMARKET_ROUND17_ECONOMIC_PATHS),
        "risk_profiles": list(_PROFILES),
        "scenarios": list(_SCENARIOS),
        "minimum_edge_grid_quote_per_share": [
            format(value, "f") for value in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS
        ],
        "loss_cluster_size": POLYMARKET_ROUND17_LOSS_CLUSTER_SIZE,
        "candidate_ledger": candidate_ledger,
        "selected_by_profile": selected,
        "development_accepted": development_accepted,
        "selection_rule": {
            "primary": (
                "maximize worst-scenario daily-block lower-95 mean event utility"
            ),
            "secondary": (
                "primary net PnL, lower drawdown, higher edge floor, simpler path"
            ),
            "all_scenarios_must_pass": True,
            "forced_activity": False,
        },
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "test_execution_accessed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }
    payload["economic_pretest_sha256"] = _canonical_sha256(payload)
    return payload


def validate_round17_economic_pretest(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("economic_pretest_sha256", "")).strip().lower()
    ledger = payload.get("candidate_ledger")
    selected = payload.get("selected_by_profile")
    accepted = payload.get("development_accepted")
    if (
        claimed != _canonical_sha256(payload)
        or payload.get("schema_version") != POLYMARKET_ROUND17_ECONOMIC_SCHEMA_VERSION
        or payload.get("contract_sha256") != POLYMARKET_ROUND17_CONTRACT_SHA256
        or payload.get("economic_contract_sha256")
        != POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256
        or payload.get("round14_risk_contract_sha256") != _ROUND14_CONTRACT_SHA256
        or _SHA256.fullmatch(str(payload.get("model_pretest_sha256") or "")) is None
        or payload.get("source_partition") != _SOURCE_PARTITION
        or payload.get("paths") != list(POLYMARKET_ROUND17_ECONOMIC_PATHS)
        or payload.get("risk_profiles") != list(_PROFILES)
        or payload.get("scenarios") != list(_SCENARIOS)
        or payload.get("minimum_edge_grid_quote_per_share")
        != [format(item, "f") for item in POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS]
        or payload.get("loss_cluster_size") != POLYMARKET_ROUND17_LOSS_CLUSTER_SIZE
        or not isinstance(ledger, list)
        or len(ledger)
        != (
            len(POLYMARKET_ROUND17_ECONOMIC_PATHS)
            * len(_PROFILES)
            * len(POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS)
        )
        or not isinstance(selected, Mapping)
        or set(selected) != set(_PROFILES)
        or any(
            not isinstance(item, Mapping)
            or item.get("risk_profile") != profile
            or item.get("development_accepted")
            is not bool(item.get("development_accepted"))
            for profile, item in selected.items()
        )
        or accepted
        != all(
            isinstance(item, Mapping) and item.get("development_accepted") is True
            for item in selected.values()
        )
        or any(
            payload.get(name) is not False
            for name in (
                "test_features_accessed",
                "test_targets_accessed",
                "test_execution_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
                "binance_credentials_used",
                "binance_execution_connected",
            )
        )
    ):
        raise ValueError("Round 17 economic pretest integrity differs")
    return {**payload, "economic_pretest_sha256": claimed}


__all__ = [
    "POLYMARKET_ROUND17_ECONOMIC_PATHS",
    "POLYMARKET_ROUND17_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256",
    "POLYMARKET_ROUND17_ECONOMIC_THRESHOLDS",
    "Round17ConditionEconomicOutcome",
    "build_round17_condition_economic_outcome",
    "fit_round17_economic_pretest",
    "validate_round17_economic_pretest",
]
