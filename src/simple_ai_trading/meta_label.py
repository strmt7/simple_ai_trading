"""Meta-label policy evidence trained from simulated trade outcomes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .backtest import BacktestResult
from .features import ModelRow
from .model import TrainedModel, confidence_adjusted_probability, model_decision_threshold
from .objective import get_objective
from .types import StrategyConfig


@dataclass(frozen=True)
class MetaLabelSample:
    opened_at: int
    side: int
    probability: float
    adjusted_probability: float
    signal_strength: float
    net_pnl: float
    return_pct: float
    profitable: bool

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetaLabelReport:
    status: str
    reason: str | None
    objective: str
    sample_count: int
    profitable_count: int
    target_precision: float
    take_threshold: float | None
    downsize_threshold: float | None
    take_count: int
    downsize_count: int
    skip_count: int
    take_precision: float
    take_mean_return: float
    take_net_pnl: float
    downsize_precision: float
    downsize_mean_return: float
    skipped_loss_avoided: float
    skipped_profit_missed: float
    policy: dict[str, object]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _precision(samples: Sequence[MetaLabelSample]) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if sample.profitable) / len(samples)


def _mean_return(samples: Sequence[MetaLabelSample]) -> float:
    if not samples:
        return 0.0
    return sum(sample.return_pct for sample in samples) / len(samples)


def _target_precision(objective_name: str) -> float:
    objective = get_objective(objective_name).name
    if objective == "conservative":
        return 0.70
    if objective == "aggressive":
        return 0.52
    return 0.60


def _signal_strength(adjusted_probability: float, threshold: float, side: int, market_type: str) -> float:
    if str(market_type).lower() == "futures" and side < 0:
        boundary = 1.0 - threshold
        return max(0.0, boundary - adjusted_probability)
    return max(0.0, adjusted_probability - threshold)


def extract_meta_label_samples(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    strategy: StrategyConfig,
    result: BacktestResult,
    *,
    market_type: str,
) -> list[MetaLabelSample]:
    """Build meta-label samples from the same simulated trades used by gates."""

    rows_by_time = {int(row.timestamp): row for row in rows}
    threshold = model_decision_threshold(model, strategy.signal_threshold)
    samples: list[MetaLabelSample] = []
    for trade in getattr(result, "trade_log", ()) or ():
        if not isinstance(trade, dict):
            continue
        opened_at = int(_finite(trade.get("opened_at"), -1))
        row = rows_by_time.get(opened_at)
        if row is None:
            continue
        side = int(_finite(trade.get("side"), 0))
        if side == 0:
            continue
        probability = _finite(model.predict_proba(row.features), 0.5)
        adjusted = confidence_adjusted_probability(probability, strategy.confidence_beta)
        net_pnl = _finite(trade.get("net_pnl"))
        return_pct = _finite(trade.get("return_pct"))
        samples.append(MetaLabelSample(
            opened_at=opened_at,
            side=1 if side > 0 else -1,
            probability=float(probability),
            adjusted_probability=float(adjusted),
            signal_strength=float(_signal_strength(adjusted, threshold, side, market_type)),
            net_pnl=float(net_pnl),
            return_pct=float(return_pct),
            profitable=bool(net_pnl > 0.0 and return_pct > 0.0),
        ))
    return samples


def _candidate_thresholds(samples: Sequence[MetaLabelSample]) -> list[float]:
    values = sorted({round(max(0.0, sample.signal_strength), 12) for sample in samples})
    if not values:
        return []
    values.append(max(values) + 1e-9)
    return values


def build_meta_label_report(
    rows: Sequence[ModelRow],
    model: TrainedModel,
    strategy: StrategyConfig,
    result: BacktestResult,
    *,
    objective_name: str,
    market_type: str,
) -> MetaLabelReport:
    """Train a compact take/downsize/skip policy from simulated trade outcomes."""

    objective = get_objective(objective_name)
    samples = extract_meta_label_samples(rows, model, strategy, result, market_type=market_type)
    target = _target_precision(objective.name)
    minimum_samples = max(2, min(max(1, int(objective.min_closed_trades)), max(2, len(samples))))
    if len(samples) < minimum_samples:
        policy = {
            "enabled": False,
            "mode": "observe_only",
            "reason": "insufficient_meta_label_samples",
            "target_precision": target,
        }
        return MetaLabelReport(
            status="insufficient",
            reason="insufficient_meta_label_samples",
            objective=objective.name,
            sample_count=len(samples),
            profitable_count=sum(1 for sample in samples if sample.profitable),
            target_precision=target,
            take_threshold=None,
            downsize_threshold=None,
            take_count=0,
            downsize_count=0,
            skip_count=len(samples),
            take_precision=0.0,
            take_mean_return=0.0,
            take_net_pnl=0.0,
            downsize_precision=0.0,
            downsize_mean_return=0.0,
            skipped_loss_avoided=sum(-sample.net_pnl for sample in samples if sample.net_pnl < 0.0),
            skipped_profit_missed=sum(sample.net_pnl for sample in samples if sample.net_pnl > 0.0),
            policy=policy,
        )

    best_threshold: float | None = None
    best_rank = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    for threshold in _candidate_thresholds(samples):
        kept = [sample for sample in samples if sample.signal_strength >= threshold]
        if len(kept) < minimum_samples:
            continue
        precision = _precision(kept)
        mean_return = _mean_return(kept)
        net_pnl = sum(sample.net_pnl for sample in kept)
        if precision < target or mean_return <= 0.0 or net_pnl <= 0.0:
            continue
        skipped = [sample for sample in samples if sample.signal_strength < threshold]
        avoided_loss = sum(-sample.net_pnl for sample in skipped if sample.net_pnl < 0.0)
        missed_profit = sum(sample.net_pnl for sample in skipped if sample.net_pnl > 0.0)
        rank = (net_pnl + avoided_loss - missed_profit, precision, mean_return, -float(len(kept)))
        if rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)

    if best_threshold is None:
        best_threshold = max(sample.signal_strength for sample in samples) + 1e-9
        status = "observe_only"
        reason = "no_profitable_precision_threshold"
    else:
        status = "trained"
        reason = None

    downsize_floor = max(0.0, best_threshold * 0.50)
    take_samples = [sample for sample in samples if sample.signal_strength >= best_threshold]
    downsize_samples = [
        sample
        for sample in samples
        if downsize_floor <= sample.signal_strength < best_threshold
    ]
    skip_samples = [sample for sample in samples if sample.signal_strength < downsize_floor]
    skipped_loss = sum(-sample.net_pnl for sample in skip_samples if sample.net_pnl < 0.0)
    skipped_profit = sum(sample.net_pnl for sample in skip_samples if sample.net_pnl > 0.0)
    policy = {
        "enabled": status == "trained",
        "mode": "take_downsize_skip",
        "reason": reason,
        "objective": objective.name,
        "target_precision": target,
        "take_threshold": best_threshold,
        "downsize_threshold": downsize_floor,
        "downsize_fraction": 0.5,
        "sample_count": len(samples),
    }
    return MetaLabelReport(
        status=status,
        reason=reason,
        objective=objective.name,
        sample_count=len(samples),
        profitable_count=sum(1 for sample in samples if sample.profitable),
        target_precision=target,
        take_threshold=float(best_threshold),
        downsize_threshold=float(downsize_floor),
        take_count=len(take_samples),
        downsize_count=len(downsize_samples),
        skip_count=len(skip_samples),
        take_precision=_precision(take_samples),
        take_mean_return=_mean_return(take_samples),
        take_net_pnl=sum(sample.net_pnl for sample in take_samples),
        downsize_precision=_precision(downsize_samples),
        downsize_mean_return=_mean_return(downsize_samples),
        skipped_loss_avoided=float(skipped_loss),
        skipped_profit_missed=float(skipped_profit),
        policy=policy,
    )
