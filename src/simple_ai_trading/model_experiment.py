"""Deterministic multi-fidelity experiment design for trading-model research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from typing import Mapping, Sequence


Scalar = bool | float | int | str
EXPERIMENT_DESIGN_CONTRACT = "causal-multi-fidelity-model-experiment-v1"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _uniform(seed: int, *parts: object) -> float:
    payload = ":".join((str(int(seed)), *(str(part) for part in parts))).encode("ascii")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / float(1 << 64)


@dataclass(frozen=True)
class FloatDomain:
    name: str
    lower: float
    upper: float
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("float-domain name cannot be empty")
        if not all(math.isfinite(value) for value in (self.lower, self.upper)):
            raise ValueError(f"{self.name} bounds must be finite")
        if self.lower >= self.upper:
            raise ValueError(f"{self.name} lower bound must be below its upper bound")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.name} scale must be linear or log")
        if self.scale == "log" and self.lower <= 0.0:
            raise ValueError(f"{self.name} log-scale bounds must be positive")

    def decode(self, unit: float) -> float:
        value = min(math.nextafter(1.0, 0.0), max(0.0, float(unit)))
        if self.scale == "log":
            return math.exp(math.log(self.lower) + value * math.log(self.upper / self.lower))
        return self.lower + value * (self.upper - self.lower)

    def accepts(self, value: Scalar) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and self.lower <= float(value) <= self.upper
        )


@dataclass(frozen=True)
class IntegerDomain:
    name: str
    lower: int
    upper: int
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("integer-domain name cannot be empty")
        if self.lower >= self.upper:
            raise ValueError(f"{self.name} lower bound must be below its upper bound")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.name} scale must be linear or log")
        if self.scale == "log" and self.lower <= 0:
            raise ValueError(f"{self.name} log-scale bounds must be positive")

    def decode(self, unit: float) -> int:
        value = min(math.nextafter(1.0, 0.0), max(0.0, float(unit)))
        if self.scale == "log":
            continuous = math.exp(
                math.log(float(self.lower))
                + value * math.log(float(self.upper) / float(self.lower))
            )
            return min(self.upper, max(self.lower, int(round(continuous))))
        width = self.upper - self.lower + 1
        return min(self.upper, self.lower + int(value * width))

    def accepts(self, value: Scalar) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and self.lower <= value <= self.upper
        )


@dataclass(frozen=True)
class ChoiceDomain:
    name: str
    choices: tuple[Scalar, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("choice-domain name cannot be empty")
        if not self.choices or len(set(self.choices)) != len(self.choices):
            raise ValueError(f"{self.name} choices must be non-empty and unique")

    def decode(self, unit: float) -> Scalar:
        value = min(math.nextafter(1.0, 0.0), max(0.0, float(unit)))
        return self.choices[min(len(self.choices) - 1, int(value * len(self.choices)))]

    def accepts(self, value: Scalar) -> bool:
        return value in self.choices


ParameterDomain = FloatDomain | IntegerDomain | ChoiceDomain


@dataclass(frozen=True)
class ExperimentCandidate:
    candidate_id: str
    source: str
    design_index: int
    parameters: tuple[tuple[str, Scalar], ...]

    def parameter_map(self) -> dict[str, Scalar]:
        return dict(self.parameters)

    def asdict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "design_index": self.design_index,
            "parameters": self.parameter_map(),
        }


@dataclass(frozen=True)
class ExperimentDesign:
    contract: str
    seed: int
    sampling_method: str
    domains: tuple[ParameterDomain, ...]
    candidates: tuple[ExperimentCandidate, ...]
    anchor_count: int
    sampled_count: int
    trial_burden: int
    design_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "seed": self.seed,
            "sampling_method": self.sampling_method,
            "domains": [asdict(domain) for domain in self.domains],
            "candidates": [candidate.asdict() for candidate in self.candidates],
            "anchor_count": self.anchor_count,
            "sampled_count": self.sampled_count,
            "trial_burden": self.trial_burden,
            "design_sha256": self.design_sha256,
        }


def _validated_parameters(
    domains: Sequence[ParameterDomain],
    values: Mapping[str, Scalar],
) -> tuple[tuple[str, Scalar], ...]:
    names = [domain.name for domain in domains]
    if set(values) != set(names):
        missing = sorted(set(names) - set(values))
        extra = sorted(set(values) - set(names))
        raise ValueError(f"candidate parameters differ from domains; missing={missing} extra={extra}")
    output: list[tuple[str, Scalar]] = []
    for domain in domains:
        value = values[domain.name]
        if not domain.accepts(value):
            raise ValueError(f"candidate parameter {domain.name} is outside its domain")
        output.append((domain.name, value))
    return tuple(output)


def _candidate(
    *,
    source: str,
    design_index: int,
    parameters: tuple[tuple[str, Scalar], ...],
) -> ExperimentCandidate:
    payload = {
        "contract": EXPERIMENT_DESIGN_CONTRACT,
        "parameters": dict(parameters),
    }
    return ExperimentCandidate(
        candidate_id=_canonical_sha256(payload),
        source=source,
        design_index=design_index,
        parameters=parameters,
    )


def generate_latin_hypercube_design(
    domains: Sequence[ParameterDomain],
    *,
    sampled_count: int,
    seed: int,
    anchors: Sequence[Mapping[str, Scalar]] = (),
) -> ExperimentDesign:
    """Generate a deterministic randomized Latin-hypercube candidate design."""

    normalized_domains = tuple(domains)
    if not normalized_domains:
        raise ValueError("experiment design requires at least one parameter domain")
    names = [domain.name for domain in normalized_domains]
    if len(names) != len(set(names)):
        raise ValueError("experiment parameter-domain names must be unique")
    samples = int(sampled_count)
    if samples <= 0:
        raise ValueError("sampled_count must be positive")
    normalized_seed = int(seed)
    candidates: list[ExperimentCandidate] = []
    seen_parameters: set[tuple[tuple[str, Scalar], ...]] = set()
    for index, values in enumerate(anchors):
        parameters = _validated_parameters(normalized_domains, values)
        if parameters in seen_parameters:
            raise ValueError("experiment anchors contain duplicate parameter sets")
        seen_parameters.add(parameters)
        candidates.append(_candidate(source="anchor", design_index=index, parameters=parameters))

    permutations = [
        sorted(
            range(samples),
            key=lambda row, dimension=dimension: (
                _uniform(normalized_seed, "permutation", dimension, row),
                row,
            ),
        )
        for dimension in range(len(normalized_domains))
    ]
    for row in range(samples):
        values: dict[str, Scalar] = {}
        for dimension, domain in enumerate(normalized_domains):
            stratum = permutations[dimension][row]
            jitter = _uniform(normalized_seed, "jitter", dimension, row)
            unit = (stratum + jitter) / samples
            values[domain.name] = domain.decode(unit)
        parameters = _validated_parameters(normalized_domains, values)
        if parameters in seen_parameters:
            raise ValueError(
                "experiment design produced a duplicate parameter set; "
                "reduce sampled_count or enlarge the discrete search space"
            )
        seen_parameters.add(parameters)
        candidates.append(
            _candidate(
                source="latin_hypercube",
                design_index=row,
                parameters=parameters,
            )
        )
    design_payload = {
        "contract": EXPERIMENT_DESIGN_CONTRACT,
        "seed": normalized_seed,
        "sampling_method": "deterministic_randomized_latin_hypercube",
        "domains": [asdict(domain) for domain in normalized_domains],
        "candidates": [candidate.asdict() for candidate in candidates],
    }
    return ExperimentDesign(
        contract=EXPERIMENT_DESIGN_CONTRACT,
        seed=normalized_seed,
        sampling_method="deterministic_randomized_latin_hypercube",
        domains=normalized_domains,
        candidates=tuple(candidates),
        anchor_count=len(anchors),
        sampled_count=samples,
        trial_burden=len(candidates),
        design_sha256=_canonical_sha256(design_payload),
    )


def tape_depth_candidate_design(
    risk_level: str,
    *,
    sampled_count: int = 24,
    seed: int = 20260710,
) -> ExperimentDesign:
    """Return the precommitted space for the tape/depth forecast overlay."""

    risk = str(risk_level).strip().lower()
    if risk not in {"conservative", "regular", "aggressive"}:
        raise ValueError("risk_level must be conservative, regular, or aggressive")
    domains: tuple[ParameterDomain, ...] = (
        ChoiceDomain("risk_level", (risk,)),
        ChoiceDomain("horizon_seconds", (5, 10, 15, 20, 30, 60, 120, 300, 900)),
        ChoiceDomain("decision_cadence_seconds", (1, 2, 5, 10)),
        ChoiceDomain("maximum_depth_age_ms", (15_000, 30_000, 60_000, 120_000)),
        ChoiceDomain("model_profile", ("regularized", "balanced", "expressive")),
        ChoiceDomain("feature_set", ("core", "tape_derived", "cross_asset", "full")),
    )
    anchors: tuple[Mapping[str, Scalar], ...] = (
        {
            "risk_level": risk,
            "horizon_seconds": 20,
            "decision_cadence_seconds": 5,
            "maximum_depth_age_ms": 60_000,
            "model_profile": "regularized",
            "feature_set": "cross_asset",
        },
        {
            "risk_level": risk,
            "horizon_seconds": 5,
            "decision_cadence_seconds": 2,
            "maximum_depth_age_ms": 30_000,
            "model_profile": "regularized",
            "feature_set": "tape_derived",
        },
        {
            "risk_level": risk,
            "horizon_seconds": 300,
            "decision_cadence_seconds": 5,
            "maximum_depth_age_ms": 60_000,
            "model_profile": "regularized",
            "feature_set": "cross_asset",
        },
    )
    return generate_latin_hypercube_design(
        domains,
        sampled_count=sampled_count,
        seed=seed,
        anchors=anchors,
    )


@dataclass(frozen=True)
class ExperimentWindow:
    window_id: str
    symbol: str
    first_period: str
    last_period: str
    calendar_days: int
    selection_basis: str = "precommitted_calendar_space_filling"


def _contiguous_windows(periods: Sequence[str], window_days: int) -> list[tuple[date, date]]:
    parsed = sorted({datetime.strptime(value, "%Y-%m-%d").date() for value in periods})
    available = set(parsed)
    output: list[tuple[date, date]] = []
    for first in parsed:
        last = first + timedelta(days=window_days - 1)
        if all(first + timedelta(days=offset) in available for offset in range(window_days)):
            output.append((first, last))
    return output


def plan_calendar_windows(
    periods_by_symbol: Mapping[str, Sequence[str]],
    *,
    window_days: int,
    windows_per_symbol: int,
    seed: int,
) -> tuple[ExperimentWindow, ...]:
    """Precommit non-overlapping windows spread across each certified calendar."""

    days = int(window_days)
    count = int(windows_per_symbol)
    if days <= 0 or count <= 0:
        raise ValueError("window_days and windows_per_symbol must be positive")
    output: list[ExperimentWindow] = []
    for symbol in sorted(periods_by_symbol):
        possibilities = _contiguous_windows(periods_by_symbol[symbol], days)
        if len(possibilities) < count:
            raise ValueError(f"{symbol} has too few contiguous periods for the requested windows")
        remaining = list(possibilities)
        selected: list[tuple[date, date]] = []
        for slot in range(count):
            target = (slot + 0.5) / count
            denominator = max(1, len(possibilities) - 1)
            ranked = sorted(
                remaining,
                key=lambda value: (
                    abs(possibilities.index(value) / denominator - target),
                    _uniform(seed, symbol, slot, value[0].isoformat()),
                    value[0],
                ),
            )
            chosen = next(
                (
                    value
                    for value in ranked
                    if all(value[1] < prior[0] or value[0] > prior[1] for prior in selected)
                ),
                None,
            )
            if chosen is None:
                raise ValueError(f"{symbol} cannot supply {count} non-overlapping windows")
            selected.append(chosen)
            remaining.remove(chosen)
        for first, last in sorted(selected):
            identity = {
                "contract": EXPERIMENT_DESIGN_CONTRACT,
                "first_period": first.isoformat(),
                "last_period": last.isoformat(),
                "seed": int(seed),
                "symbol": symbol,
            }
            output.append(
                ExperimentWindow(
                    window_id=_canonical_sha256(identity),
                    symbol=symbol,
                    first_period=first.isoformat(),
                    last_period=last.isoformat(),
                    calendar_days=days,
                )
            )
    return tuple(output)


@dataclass(frozen=True)
class FidelityStage:
    name: str
    role: str
    keep_fraction: float
    minimum_survivors: int
    minimum_windows: int
    minimum_symbols: int
    minimum_closed_trades: int
    minimum_trades_per_day: float
    minimum_window_pass_rate: float
    minimum_expectancy_bps: float
    minimum_profit_factor: float
    maximum_drawdown_bps: float
    maximum_consecutive_losses: int
    maximum_side_share: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("fidelity-stage name cannot be empty")
        if self.role not in {"viability_screen", "selection", "prequential", "full_validation"}:
            raise ValueError("fidelity stages cannot represent terminal evaluation")
        if not 0.0 < self.keep_fraction <= 1.0:
            raise ValueError("fidelity-stage keep_fraction must lie in (0, 1]")
        if min(
            self.minimum_survivors,
            self.minimum_windows,
            self.minimum_symbols,
            self.minimum_closed_trades,
            self.maximum_consecutive_losses,
        ) < 0:
            raise ValueError("fidelity-stage count gates cannot be negative")
        if not 0.0 <= self.minimum_window_pass_rate <= 1.0:
            raise ValueError("minimum_window_pass_rate must lie in [0, 1]")
        if not 0.5 <= self.maximum_side_share <= 1.0:
            raise ValueError("maximum_side_share must lie in [0.5, 1]")


@dataclass(frozen=True)
class WindowPerformance:
    candidate_id: str
    stage_name: str
    window_id: str
    symbol: str
    calendar_days: float
    closed_trades: int
    winning_net_bps: float
    losing_net_bps_abs: float
    max_drawdown_bps: float
    max_consecutive_losses: int
    long_trades: int
    short_trades: int
    liquidation_events: int
    cost_model_coverage_ratio: float
    source_verified: bool
    error: str = ""

    @property
    def net_bps(self) -> float:
        return self.winning_net_bps - self.losing_net_bps_abs

    @property
    def expectancy_bps(self) -> float:
        return self.net_bps / self.closed_trades if self.closed_trades else 0.0

    def validate(self) -> None:
        numeric = (
            self.calendar_days,
            self.winning_net_bps,
            self.losing_net_bps_abs,
            self.max_drawdown_bps,
            self.cost_model_coverage_ratio,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("window performance contains non-finite values")
        if (
            self.calendar_days <= 0.0
            or self.closed_trades < 0
            or self.winning_net_bps < 0.0
            or self.losing_net_bps_abs < 0.0
            or self.max_drawdown_bps < 0.0
            or self.max_consecutive_losses < 0
            or self.long_trades < 0
            or self.short_trades < 0
            or self.liquidation_events < 0
            or not 0.0 <= self.cost_model_coverage_ratio <= 1.0
        ):
            raise ValueError("window performance violates metric bounds")
        if self.long_trades + self.short_trades != self.closed_trades:
            raise ValueError("window side counts do not equal closed trades")
        if not all((self.candidate_id, self.stage_name, self.window_id, self.symbol)):
            raise ValueError("window performance identifiers cannot be empty")


@dataclass(frozen=True)
class CandidateStageDiagnostic:
    candidate_id: str
    passed_hard_gates: bool
    survived: bool
    rank: int | None
    reasons: tuple[str, ...]
    window_count: int
    symbol_count: int
    calendar_days: float
    closed_trades: int
    trades_per_day: float
    net_bps: float
    expectancy_bps: float
    profit_factor: float | None
    max_drawdown_bps: float
    max_consecutive_losses: int
    maximum_side_share: float
    positive_window_rate: float
    lower_quartile_window_expectancy_bps: float


@dataclass(frozen=True)
class StageDecision:
    contract: str
    stage: FidelityStage
    candidate_count: int
    window_evaluation_count: int
    prior_trial_burden: int
    cumulative_trial_burden: int
    survivor_ids: tuple[str, ...]
    diagnostics: tuple[CandidateStageDiagnostic, ...]
    authorization: str
    terminal_holdout_consumed: bool
    decision_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "stage": asdict(self.stage),
            "candidate_count": self.candidate_count,
            "window_evaluation_count": self.window_evaluation_count,
            "prior_trial_burden": self.prior_trial_burden,
            "cumulative_trial_burden": self.cumulative_trial_burden,
            "survivor_ids": list(self.survivor_ids),
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "authorization": self.authorization,
            "terminal_holdout_consumed": self.terminal_holdout_consumed,
            "decision_sha256": self.decision_sha256,
        }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = min(1.0, max(0.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _candidate_diagnostic(
    candidate_id: str,
    results: Sequence[WindowPerformance],
    stage: FidelityStage,
) -> tuple[CandidateStageDiagnostic, tuple[float, ...]]:
    reasons: list[str] = []
    windows = list(results)
    symbols = {item.symbol for item in windows}
    calendar_days = sum(item.calendar_days for item in windows)
    closed_trades = sum(item.closed_trades for item in windows)
    wins = sum(item.winning_net_bps for item in windows)
    losses = sum(item.losing_net_bps_abs for item in windows)
    net = wins - losses
    expectancy = net / closed_trades if closed_trades else 0.0
    profit_factor = wins / losses if losses > 0.0 else (None if wins <= 0.0 else math.inf)
    drawdown = max((item.max_drawdown_bps for item in windows), default=0.0)
    loss_streak = max((item.max_consecutive_losses for item in windows), default=0)
    long_trades = sum(item.long_trades for item in windows)
    short_trades = sum(item.short_trades for item in windows)
    side_share = max(long_trades, short_trades) / closed_trades if closed_trades else 0.0
    trades_per_day = closed_trades / calendar_days if calendar_days > 0.0 else 0.0
    positive_window_rate = (
        sum(item.net_bps > 0.0 for item in windows) / len(windows) if windows else 0.0
    )
    window_expectancies = [item.expectancy_bps for item in windows]
    lower_quartile = _quantile(window_expectancies, 0.25)
    if len(windows) < stage.minimum_windows:
        reasons.append(f"windows<{stage.minimum_windows}")
    if len(symbols) < stage.minimum_symbols:
        reasons.append(f"symbols<{stage.minimum_symbols}")
    if any(item.error for item in windows):
        reasons.append("window_evaluation_error")
    if any(not item.source_verified for item in windows):
        reasons.append("source_not_verified")
    if any(item.cost_model_coverage_ratio < 1.0 for item in windows):
        reasons.append("execution_cost_coverage_incomplete")
    if any(item.liquidation_events > 0 for item in windows):
        reasons.append("liquidation_events>0")
    if closed_trades < stage.minimum_closed_trades:
        reasons.append(f"closed_trades<{stage.minimum_closed_trades}")
    if trades_per_day < stage.minimum_trades_per_day:
        reasons.append(f"trades_per_day<{stage.minimum_trades_per_day}")
    if positive_window_rate < stage.minimum_window_pass_rate:
        reasons.append(f"positive_window_rate<{stage.minimum_window_pass_rate}")
    if expectancy < stage.minimum_expectancy_bps:
        reasons.append(f"expectancy_bps<{stage.minimum_expectancy_bps}")
    comparable_profit_factor = 0.0 if profit_factor is None else profit_factor
    if comparable_profit_factor < stage.minimum_profit_factor:
        reasons.append(f"profit_factor<{stage.minimum_profit_factor}")
    if drawdown > stage.maximum_drawdown_bps:
        reasons.append(f"max_drawdown_bps>{stage.maximum_drawdown_bps}")
    if loss_streak > stage.maximum_consecutive_losses:
        reasons.append(f"max_consecutive_losses>{stage.maximum_consecutive_losses}")
    if closed_trades and side_share > stage.maximum_side_share:
        reasons.append(f"side_share>{stage.maximum_side_share}")
    diagnostic = CandidateStageDiagnostic(
        candidate_id=candidate_id,
        passed_hard_gates=not reasons,
        survived=False,
        rank=None,
        reasons=tuple(reasons),
        window_count=len(windows),
        symbol_count=len(symbols),
        calendar_days=calendar_days,
        closed_trades=closed_trades,
        trades_per_day=trades_per_day,
        net_bps=net,
        expectancy_bps=expectancy,
        profit_factor=(None if profit_factor is None or math.isinf(profit_factor) else profit_factor),
        max_drawdown_bps=drawdown,
        max_consecutive_losses=loss_streak,
        maximum_side_share=side_share,
        positive_window_rate=positive_window_rate,
        lower_quartile_window_expectancy_bps=lower_quartile,
    )
    rank_key = (
        positive_window_rate,
        lower_quartile,
        expectancy,
        min(10.0, comparable_profit_factor),
        net,
        -drawdown,
    )
    return diagnostic, rank_key


def apply_successive_halving_stage(
    stage: FidelityStage,
    candidates: Sequence[ExperimentCandidate],
    results: Sequence[WindowPerformance],
    *,
    prior_trial_burden: int = 0,
) -> StageDecision:
    """Apply hard gates and deterministic ranking without authorizing deployment."""

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("stage candidates must be non-empty and unique")
    expected = set(candidate_ids)
    grouped: dict[str, list[WindowPerformance]] = {candidate_id: [] for candidate_id in candidate_ids}
    seen_evaluations: set[tuple[str, str, str]] = set()
    for item in results:
        item.validate()
        if item.stage_name != stage.name:
            raise ValueError("window result stage does not match the fidelity stage")
        if item.candidate_id not in expected:
            raise ValueError("window result references an unknown candidate")
        key = (item.candidate_id, item.symbol, item.window_id)
        if key in seen_evaluations:
            raise ValueError("duplicate candidate/window evaluation")
        seen_evaluations.add(key)
        grouped[item.candidate_id].append(item)
    diagnostics_by_id: dict[str, CandidateStageDiagnostic] = {}
    rank_keys: dict[str, tuple[float, ...]] = {}
    for candidate_id in candidate_ids:
        diagnostic, rank_key = _candidate_diagnostic(
            candidate_id,
            grouped[candidate_id],
            stage,
        )
        diagnostics_by_id[candidate_id] = diagnostic
        rank_keys[candidate_id] = rank_key
    passed = [candidate_id for candidate_id in candidate_ids if diagnostics_by_id[candidate_id].passed_hard_gates]
    passed.sort()
    passed.sort(key=lambda candidate_id: rank_keys[candidate_id], reverse=True)
    survivor_count = min(
        len(passed),
        max(stage.minimum_survivors, int(math.ceil(len(passed) * stage.keep_fraction))),
    )
    survivor_ids = tuple(passed[:survivor_count])
    for rank, candidate_id in enumerate(passed, start=1):
        diagnostic = diagnostics_by_id[candidate_id]
        if candidate_id in survivor_ids:
            diagnostics_by_id[candidate_id] = replace(
                diagnostic,
                survived=True,
                rank=rank,
            )
        else:
            diagnostics_by_id[candidate_id] = replace(
                diagnostic,
                rank=rank,
                reasons=(*diagnostic.reasons, "successive_halving_rank_cut"),
            )
    diagnostics = tuple(diagnostics_by_id[candidate_id] for candidate_id in candidate_ids)
    cumulative = max(0, int(prior_trial_burden)) + len(candidate_ids)
    decision_payload = {
        "contract": EXPERIMENT_DESIGN_CONTRACT,
        "stage": asdict(stage),
        "candidate_ids": candidate_ids,
        "cumulative_trial_burden": cumulative,
        "diagnostics": [asdict(item) for item in diagnostics],
        "survivor_ids": list(survivor_ids),
    }
    return StageDecision(
        contract=EXPERIMENT_DESIGN_CONTRACT,
        stage=stage,
        candidate_count=len(candidate_ids),
        window_evaluation_count=len(results),
        prior_trial_burden=max(0, int(prior_trial_burden)),
        cumulative_trial_burden=cumulative,
        survivor_ids=survivor_ids,
        diagnostics=diagnostics,
        authorization="research_only_no_trading_authority",
        terminal_holdout_consumed=False,
        decision_sha256=_canonical_sha256(decision_payload),
    )


def default_fidelity_stages(risk_level: str) -> tuple[FidelityStage, ...]:
    """Return risk-specific stages; none can replace sealed terminal evaluation."""

    risk = str(risk_level).strip().lower()
    drawdowns = {
        "conservative": (150.0, 250.0, 400.0, 600.0),
        "regular": (250.0, 400.0, 650.0, 900.0),
        "aggressive": (400.0, 650.0, 1_000.0, 1_400.0),
    }
    if risk not in drawdowns:
        raise ValueError("risk_level must be conservative, regular, or aggressive")
    limits = drawdowns[risk]
    return (
        FidelityStage(
            name="stage-1-causal-viability",
            role="viability_screen",
            keep_fraction=1.0 / 3.0,
            minimum_survivors=4,
            minimum_windows=6,
            minimum_symbols=3,
            minimum_closed_trades=12,
            minimum_trades_per_day=0.25,
            minimum_window_pass_rate=0.25,
            minimum_expectancy_bps=-0.25,
            minimum_profit_factor=0.75,
            maximum_drawdown_bps=limits[0],
            maximum_consecutive_losses=12,
            maximum_side_share=0.95,
        ),
        FidelityStage(
            name="stage-2-cross-regime-selection",
            role="selection",
            keep_fraction=1.0 / 3.0,
            minimum_survivors=2,
            minimum_windows=9,
            minimum_symbols=3,
            minimum_closed_trades=45,
            minimum_trades_per_day=0.50,
            minimum_window_pass_rate=0.50,
            minimum_expectancy_bps=0.0,
            minimum_profit_factor=1.0,
            maximum_drawdown_bps=limits[1],
            maximum_consecutive_losses=10,
            maximum_side_share=0.90,
        ),
        FidelityStage(
            name="stage-3-rolling-prequential",
            role="prequential",
            keep_fraction=0.50,
            minimum_survivors=1,
            minimum_windows=12,
            minimum_symbols=3,
            minimum_closed_trades=120,
            minimum_trades_per_day=0.75,
            minimum_window_pass_rate=0.60,
            minimum_expectancy_bps=0.0,
            minimum_profit_factor=1.05,
            maximum_drawdown_bps=limits[2],
            maximum_consecutive_losses=8,
            maximum_side_share=0.85,
        ),
        FidelityStage(
            name="stage-4-full-history-validation",
            role="full_validation",
            keep_fraction=1.0,
            minimum_survivors=1,
            minimum_windows=3,
            minimum_symbols=3,
            minimum_closed_trades=250,
            minimum_trades_per_day=0.50,
            minimum_window_pass_rate=2.0 / 3.0,
            minimum_expectancy_bps=0.0,
            minimum_profit_factor=1.10,
            maximum_drawdown_bps=limits[3],
            maximum_consecutive_losses=7,
            maximum_side_share=0.80,
        ),
    )


def validate_experiment_design_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate a serialized design before it can constrain screening reports."""

    if payload.get("contract") != EXPERIMENT_DESIGN_CONTRACT:
        raise ValueError("experiment design contract is unsupported")
    domains = payload.get("domains")
    candidates = payload.get("candidates")
    if not isinstance(domains, list) or not domains or not isinstance(candidates, list):
        raise ValueError("experiment design domains or candidates are invalid")
    domain_by_name: dict[str, Mapping[str, object]] = {}
    for raw_domain in domains:
        if not isinstance(raw_domain, Mapping):
            raise ValueError("experiment design domain is invalid")
        name = str(raw_domain.get("name") or "")
        if not name or name in domain_by_name:
            raise ValueError("experiment design domain names are empty or duplicated")
        if "choices" in raw_domain:
            choices = raw_domain.get("choices")
            if not isinstance(choices, (list, tuple)) or not choices:
                raise ValueError("experiment design choice domain is invalid")
        else:
            try:
                lower = float(raw_domain.get("lower"))
                upper = float(raw_domain.get("upper"))
            except (TypeError, ValueError) as exc:
                raise ValueError("experiment design numeric domain is invalid") from exc
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower >= upper
                or raw_domain.get("scale") not in {"linear", "log"}
            ):
                raise ValueError("experiment design numeric domain is invalid")
        domain_by_name[name] = raw_domain
    names = tuple(domain_by_name)
    normalized_candidates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_parameters: set[str] = set()
    source_counts = {"anchor": 0, "latin_hypercube": 0}
    source_indices: dict[str, set[int]] = {key: set() for key in source_counts}
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("experiment design candidate is invalid")
        candidate_id = str(raw_candidate.get("candidate_id") or "")
        source = str(raw_candidate.get("source") or "")
        parameters = raw_candidate.get("parameters")
        try:
            design_index = int(raw_candidate.get("design_index", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError("experiment design candidate index is invalid") from exc
        if (
            source not in source_counts
            or design_index < 0
            or design_index in source_indices[source]
            or not isinstance(parameters, Mapping)
            or set(parameters) != set(names)
        ):
            raise ValueError("experiment design candidate structure is invalid")
        normalized_parameters = {name: parameters[name] for name in names}
        for name, value in normalized_parameters.items():
            domain = domain_by_name[name]
            if "choices" in domain:
                if value not in domain["choices"]:  # type: ignore[operator]
                    raise ValueError("experiment design candidate is outside a choice domain")
                continue
            lower = float(domain["lower"])
            upper = float(domain["upper"])
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError("experiment design candidate is outside a numeric domain")
        expected_id = _canonical_sha256(
            {
                "contract": EXPERIMENT_DESIGN_CONTRACT,
                "parameters": normalized_parameters,
            }
        )
        parameter_identity = _canonical_sha256(normalized_parameters)
        if (
            candidate_id != expected_id
            or candidate_id in seen_ids
            or parameter_identity in seen_parameters
        ):
            raise ValueError("experiment design candidate identity is invalid or duplicated")
        seen_ids.add(candidate_id)
        seen_parameters.add(parameter_identity)
        source_indices[source].add(design_index)
        source_counts[source] += 1
        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "source": source,
                "design_index": design_index,
                "parameters": normalized_parameters,
            }
        )
    try:
        seed = int(payload.get("seed"))
        anchor_count = int(payload.get("anchor_count"))
        sampled_count = int(payload.get("sampled_count"))
        trial_burden = int(payload.get("trial_burden"))
    except (TypeError, ValueError) as exc:
        raise ValueError("experiment design counts are invalid") from exc
    if (
        payload.get("sampling_method") != "deterministic_randomized_latin_hypercube"
        or anchor_count != source_counts["anchor"]
        or sampled_count != source_counts["latin_hypercube"]
        or trial_burden != len(normalized_candidates)
        or trial_burden <= 0
        or source_indices["anchor"] != set(range(anchor_count))
        or source_indices["latin_hypercube"] != set(range(sampled_count))
    ):
        raise ValueError("experiment design counts or sampling method are invalid")
    design_payload = {
        "contract": EXPERIMENT_DESIGN_CONTRACT,
        "seed": seed,
        "sampling_method": payload["sampling_method"],
        "domains": domains,
        "candidates": normalized_candidates,
    }
    if payload.get("design_sha256") != _canonical_sha256(design_payload):
        raise ValueError("experiment design fingerprint is invalid")
    return {
        **dict(payload),
        "candidates": normalized_candidates,
        "seed": seed,
        "anchor_count": anchor_count,
        "sampled_count": sampled_count,
        "trial_burden": trial_burden,
    }


__all__ = [
    "ChoiceDomain",
    "EXPERIMENT_DESIGN_CONTRACT",
    "ExperimentCandidate",
    "ExperimentDesign",
    "ExperimentWindow",
    "FidelityStage",
    "FloatDomain",
    "IntegerDomain",
    "StageDecision",
    "WindowPerformance",
    "apply_successive_halving_stage",
    "default_fidelity_stages",
    "generate_latin_hypercube_design",
    "plan_calendar_windows",
    "tape_depth_candidate_design",
    "validate_experiment_design_payload",
]
