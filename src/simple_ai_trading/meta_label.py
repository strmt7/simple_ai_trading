"""Meta-label policy evidence trained from simulated trade outcomes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Sequence

from .features import ModelRow
from .model import TrainedModel, confidence_adjusted_probability, model_decision_threshold, model_direction_thresholds
from .statistical_resampling import moving_block_bootstrap_mean
from .types import StrategyConfig

if TYPE_CHECKING:
    from .backtest import BacktestResult


META_LABEL_EVIDENCE_SCHEMA_VERSION = "meta-label-after-cost-v3"
META_LABEL_SPLIT_SCHEMA_VERSION = "meta-label-chronological-split-v1"
META_LABEL_MINIMUM_ACTION_SAMPLES = 30
META_LABEL_BOOTSTRAP_SAMPLES = 2_000
META_LABEL_BOOTSTRAP_CONFIDENCE = 0.95
META_LABEL_VALIDATION_FRACTION = 0.40


@dataclass(frozen=True)
class MetaLabelSample:
    opened_at: int
    closed_at: int
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
class MetaLabelChronologicalSplit:
    calibration: tuple[MetaLabelSample, ...]
    purged: tuple[MetaLabelSample, ...]
    validation: tuple[MetaLabelSample, ...]
    validation_start_opened_at: int
    calibration_sha256: str
    purged_sha256: str
    validation_sha256: str


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


@dataclass(frozen=True)
class MetaLabelDecision:
    enabled: bool
    action: str
    size_multiplier: float
    signal_strength: float
    reason: str
    validation_minimum_sample_count: int = 0
    validation_minimum_precision: float = 0.0
    validation_sample_count: int = 0
    validation_precision: float = 0.0
    expected_after_cost_return: float = 0.0
    expected_after_cost_pnl: float = 0.0
    validation_bootstrap_samples: int = 0
    validation_bootstrap_confidence: float = 0.0
    validation_bootstrap_block_length: int = 0
    validation_bootstrap_lower_after_cost_return: float = 0.0

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _finite_or_none(value: object) -> float | None:
    parsed = _finite(value, math.nan)
    return parsed if math.isfinite(parsed) else None


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    parsed = _finite_or_none(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _is_sha256(value: object) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _precision(samples: Sequence[MetaLabelSample]) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if sample.profitable) / len(samples)


def _mean_return(samples: Sequence[MetaLabelSample]) -> float:
    if not samples:
        return 0.0
    return sum(sample.return_pct for sample in samples) / len(samples)


def _sample_digest(samples: Sequence[MetaLabelSample]) -> str:
    binding = hashlib.sha256()
    for sample in sorted(
        samples,
        key=lambda item: (item.opened_at, item.closed_at, item.side),
    ):
        binding.update(
            (
                f"{sample.opened_at}:{sample.closed_at}:{sample.side}:"
                f"{sample.signal_strength:.17g}:{sample.return_pct:.17g}:"
                f"{sample.net_pnl:.17g};"
            ).encode("ascii")
        )
    return binding.hexdigest()


def _chronological_split(
    samples: Sequence[MetaLabelSample],
    *,
    minimum_samples: int,
) -> MetaLabelChronologicalSplit | None:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.opened_at,
                sample.closed_at,
                sample.side,
            ),
        )
    )
    if len(ordered) < minimum_samples * 2:
        return None
    requested_validation = max(
        minimum_samples,
        int(math.floor(len(ordered) * META_LABEL_VALIDATION_FRACTION)),
    )
    boundary_index = len(ordered) - requested_validation
    validation_start = ordered[boundary_index].opened_at
    validation = tuple(
        sample for sample in ordered if sample.opened_at >= validation_start
    )
    before_validation = tuple(
        sample for sample in ordered if sample.opened_at < validation_start
    )
    calibration = tuple(
        sample
        for sample in before_validation
        if sample.closed_at < validation_start
    )
    purged = tuple(
        sample
        for sample in before_validation
        if sample.closed_at >= validation_start
    )
    if len(calibration) < minimum_samples or len(validation) < minimum_samples:
        return None
    return MetaLabelChronologicalSplit(
        calibration=calibration,
        purged=purged,
        validation=validation,
        validation_start_opened_at=int(validation_start),
        calibration_sha256=_sample_digest(calibration),
        purged_sha256=_sample_digest(purged),
        validation_sha256=_sample_digest(validation),
    )


def _bucket_bootstrap(
    samples: Sequence[MetaLabelSample],
    *,
    objective: str,
    action: str,
) -> dict[str, object]:
    ordered = sorted(
        samples,
        key=lambda sample: (sample.opened_at, sample.closed_at, sample.side),
    )
    binding = hashlib.sha256()
    binding.update(f"meta-label:{objective}:{action}:".encode("ascii"))
    for sample in ordered:
        binding.update(f"{sample.return_pct:.17g};".encode("ascii"))
    return moving_block_bootstrap_mean(
        tuple(sample.return_pct for sample in ordered),
        samples=META_LABEL_BOOTSTRAP_SAMPLES,
        confidence=META_LABEL_BOOTSTRAP_CONFIDENCE,
        seed_material=binding.hexdigest(),
    )


def _target_precision(objective_name: str) -> float:
    from .objective import get_objective

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


def _policy_action_evidence(
    policy: dict[str, object],
    action: str,
) -> dict[str, object]:
    prefix = "take" if action == "take" else "downsize"
    return {
        "minimum_samples": _integer_or_none(
            policy.get("minimum_action_samples")
        ),
        "minimum_precision": (
            _finite_or_none(policy.get("target_precision"))
            if action == "take"
            else 0.0
        ),
        "samples": _integer_or_none(
            policy.get(f"{prefix}_sample_count")
        ),
        "precision": _finite_or_none(
            policy.get(f"{prefix}_precision")
        ),
        "mean_return": _finite_or_none(
            policy.get(f"{prefix}_mean_return")
        ),
        "net_pnl": _finite_or_none(
            policy.get(f"{prefix}_net_pnl")
        ),
        "bootstrap_samples": _integer_or_none(
            policy.get(f"{prefix}_bootstrap_samples")
        ),
        "bootstrap_confidence": _finite_or_none(
            policy.get(f"{prefix}_bootstrap_confidence")
        ),
        "bootstrap_block_length": _integer_or_none(
            policy.get(f"{prefix}_bootstrap_block_length")
        ),
        "bootstrap_lower_return": _finite_or_none(
            policy.get(f"{prefix}_bootstrap_mean_return_lower")
        ),
    }


def _action_evidence_is_valid(evidence: dict[str, object]) -> bool:
    minimum_samples = evidence.get("minimum_samples")
    samples = evidence.get("samples")
    minimum_precision = evidence.get("minimum_precision")
    precision = evidence.get("precision")
    mean_return = evidence.get("mean_return")
    net_pnl = evidence.get("net_pnl")
    bootstrap_samples = evidence.get("bootstrap_samples")
    bootstrap_confidence = evidence.get("bootstrap_confidence")
    block_length = evidence.get("bootstrap_block_length")
    bootstrap_lower = evidence.get("bootstrap_lower_return")
    return bool(
        isinstance(minimum_samples, int)
        and minimum_samples >= META_LABEL_MINIMUM_ACTION_SAMPLES
        and isinstance(samples, int)
        and samples >= minimum_samples
        and isinstance(minimum_precision, float)
        and 0.0 <= minimum_precision <= 1.0
        and isinstance(precision, float)
        and minimum_precision <= precision <= 1.0
        and isinstance(mean_return, float)
        and mean_return > 0.0
        and isinstance(net_pnl, float)
        and net_pnl > 0.0
        and isinstance(bootstrap_samples, int)
        and bootstrap_samples >= META_LABEL_BOOTSTRAP_SAMPLES
        and isinstance(bootstrap_confidence, float)
        and META_LABEL_BOOTSTRAP_CONFIDENCE <= bootstrap_confidence < 1.0
        and isinstance(block_length, int)
        and 0 < block_length <= samples
        and isinstance(bootstrap_lower, float)
        and bootstrap_lower > 0.0
    )


def _split_evidence_is_valid(policy: dict[str, object]) -> bool:
    source_count = _integer_or_none(policy.get("source_sample_count"))
    calibration_count = _integer_or_none(
        policy.get("calibration_sample_count")
    )
    purged_count = _integer_or_none(policy.get("purged_sample_count"))
    validation_count = _integer_or_none(
        policy.get("policy_validation_sample_count")
    )
    calibration_end = _integer_or_none(
        policy.get("calibration_end_closed_at")
    )
    validation_start = _integer_or_none(
        policy.get("validation_start_opened_at")
    )
    validation_end = _integer_or_none(
        policy.get("validation_end_closed_at")
    )
    minimum_samples = _integer_or_none(policy.get("minimum_action_samples"))
    target_precision = _finite_or_none(policy.get("target_precision"))
    calibration_take_count = _integer_or_none(
        policy.get("calibration_take_sample_count")
    )
    calibration_take_precision = _finite_or_none(
        policy.get("calibration_take_precision")
    )
    calibration_take_mean_return = _finite_or_none(
        policy.get("calibration_take_mean_return")
    )
    calibration_take_net_pnl = _finite_or_none(
        policy.get("calibration_take_net_pnl")
    )
    return bool(
        policy.get("split_schema_version") == META_LABEL_SPLIT_SCHEMA_VERSION
        and isinstance(source_count, int)
        and isinstance(minimum_samples, int)
        and minimum_samples >= META_LABEL_MINIMUM_ACTION_SAMPLES
        and isinstance(calibration_count, int)
        and calibration_count >= minimum_samples
        and isinstance(purged_count, int)
        and purged_count >= 0
        and isinstance(validation_count, int)
        and validation_count >= minimum_samples
        and source_count == calibration_count + purged_count + validation_count
        and isinstance(target_precision, float)
        and 0.0 <= target_precision <= 1.0
        and isinstance(calibration_take_count, int)
        and minimum_samples <= calibration_take_count <= calibration_count
        and isinstance(calibration_take_precision, float)
        and target_precision <= calibration_take_precision <= 1.0
        and isinstance(calibration_take_mean_return, float)
        and calibration_take_mean_return > 0.0
        and isinstance(calibration_take_net_pnl, float)
        and calibration_take_net_pnl > 0.0
        and isinstance(calibration_end, int)
        and isinstance(validation_start, int)
        and isinstance(validation_end, int)
        and 0 <= calibration_end < validation_start <= validation_end
        and _is_sha256(policy.get("source_samples_sha256"))
        and _is_sha256(policy.get("calibration_samples_sha256"))
        and _is_sha256(policy.get("purged_samples_sha256"))
        and _is_sha256(policy.get("validation_samples_sha256"))
    )


def validate_enabled_meta_label_policy(policy: object) -> tuple[bool, str]:
    """Validate every executable branch of an enabled persisted policy."""

    if not isinstance(policy, dict) or policy.get("enabled") is not True:
        return False, "meta_label_policy_not_enabled"
    if policy.get("evidence_schema_version") != META_LABEL_EVIDENCE_SCHEMA_VERSION:
        return False, "invalid_meta_label_evidence_schema"
    if policy.get("mode") != "take_downsize_skip":
        return False, "invalid_meta_label_mode"
    if not _split_evidence_is_valid(policy):
        return False, "invalid_meta_label_split_evidence"
    take_threshold = _finite(policy.get("take_threshold"), math.nan)
    downsize_threshold = _finite(policy.get("downsize_threshold"), math.nan)
    downsize_fraction = _finite(policy.get("downsize_fraction"), math.nan)
    if (
        not math.isfinite(take_threshold)
        or not math.isfinite(downsize_threshold)
        or take_threshold < 0.0
        or downsize_threshold < 0.0
        or downsize_threshold > take_threshold
    ):
        return False, "invalid_meta_label_thresholds"
    if not 0.05 <= downsize_fraction <= 1.0:
        return False, "invalid_meta_label_downsize_fraction"
    take_evidence = _policy_action_evidence(policy, "take")
    if not _action_evidence_is_valid(take_evidence):
        return False, "invalid_meta_label_take_evidence"
    validation_count = int(policy["policy_validation_sample_count"])
    take_count = int(take_evidence["samples"])
    if take_count > validation_count:
        return False, "invalid_meta_label_action_partition"
    if downsize_threshold < take_threshold:
        downsize_evidence = _policy_action_evidence(policy, "downsize")
        if not _action_evidence_is_valid(downsize_evidence):
            return False, "invalid_meta_label_downsize_evidence"
        if take_count + int(downsize_evidence["samples"]) > validation_count:
            return False, "invalid_meta_label_action_partition"
    return True, "meta_label_policy_valid"


def apply_meta_label_policy(
    policy: object,
    *,
    adjusted_probability: float,
    threshold: float,
    side: int,
    market_type: str,
) -> MetaLabelDecision:
    """Return take/downsize/skip behavior for a persisted meta-label policy.

    Missing or user-disabled policies preserve primary-model behavior. An
    explicit observe-only result and any malformed enabled policy fail closed
    by skipping the entry, because a corrupted execution gate is more dangerous
    than a missed trade.
    """

    side = 1 if side > 0 else (-1 if side < 0 else 0)
    strength = _signal_strength(
        _finite(adjusted_probability, 0.5),
        _finite(threshold, 0.5),
        side,
        market_type,
    )
    if side == 0:
        return MetaLabelDecision(False, "no_signal", 0.0, float(strength), "no_actionable_signal")
    if not isinstance(policy, dict):
        return MetaLabelDecision(False, "take", 1.0, float(strength), "meta_label_policy_disabled")
    if policy.get("enabled") is not True:
        if policy.get("mode") == "observe_only":
            return MetaLabelDecision(
                True,
                "skip",
                0.0,
                float(strength),
                "meta_label_observe_only",
            )
        return MetaLabelDecision(
            False,
            "take",
            1.0,
            float(strength),
            "meta_label_policy_disabled",
        )
    policy_valid, policy_reason = validate_enabled_meta_label_policy(policy)
    if not policy_valid:
        return MetaLabelDecision(
            True,
            "skip",
            0.0,
            float(strength),
            policy_reason,
        )
    take_threshold = _finite(policy.get("take_threshold"), math.nan)
    downsize_threshold = _finite(policy.get("downsize_threshold"), math.nan)
    downsize_fraction = _finite(policy.get("downsize_fraction"), 0.5)
    if strength >= take_threshold:
        evidence = _policy_action_evidence(policy, "take")
        return MetaLabelDecision(
            True,
            "take",
            1.0,
            float(strength),
            "meta_label_take",
            validation_minimum_sample_count=int(evidence["minimum_samples"]),
            validation_minimum_precision=float(evidence["minimum_precision"]),
            validation_sample_count=int(evidence["samples"]),
            validation_precision=float(evidence["precision"]),
            expected_after_cost_return=float(evidence["mean_return"]),
            expected_after_cost_pnl=float(evidence["net_pnl"]),
            validation_bootstrap_samples=int(evidence["bootstrap_samples"]),
            validation_bootstrap_confidence=float(
                evidence["bootstrap_confidence"]
            ),
            validation_bootstrap_block_length=int(
                evidence["bootstrap_block_length"]
            ),
            validation_bootstrap_lower_after_cost_return=float(
                evidence["bootstrap_lower_return"]
            ),
        )
    if strength >= downsize_threshold:
        evidence = _policy_action_evidence(policy, "downsize")
        return MetaLabelDecision(
            True,
            "downsize",
            float(downsize_fraction),
            float(strength),
            "meta_label_downsize",
            validation_minimum_sample_count=int(evidence["minimum_samples"]),
            validation_minimum_precision=float(evidence["minimum_precision"]),
            validation_sample_count=int(evidence["samples"]),
            validation_precision=float(evidence["precision"]),
            expected_after_cost_return=float(evidence["mean_return"]),
            expected_after_cost_pnl=float(evidence["net_pnl"]),
            validation_bootstrap_samples=int(evidence["bootstrap_samples"]),
            validation_bootstrap_confidence=float(
                evidence["bootstrap_confidence"]
            ),
            validation_bootstrap_block_length=int(
                evidence["bootstrap_block_length"]
            ),
            validation_bootstrap_lower_after_cost_return=float(
                evidence["bootstrap_lower_return"]
            ),
        )
    return MetaLabelDecision(True, "skip", 0.0, float(strength), "meta_label_skip")


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
    long_threshold, short_threshold = model_direction_thresholds(model, strategy.signal_threshold, market_type=market_type)
    samples: list[MetaLabelSample] = []
    for trade in getattr(result, "trade_log", ()) or ():
        if not isinstance(trade, dict):
            continue
        opened_at = int(_finite(trade.get("opened_at"), -1))
        if "closed_at" not in trade:
            continue
        closed_at = int(_finite(trade.get("closed_at"), -1))
        row = rows_by_time.get(opened_at)
        if row is None or closed_at < opened_at:
            continue
        side = int(_finite(trade.get("side"), 0))
        if side == 0:
            continue
        probability = _finite(model.predict_proba(row.features), 0.5)
        adjusted = confidence_adjusted_probability(probability, strategy.confidence_beta)
        side_threshold = threshold
        if str(market_type).lower() == "futures":
            if side > 0 and long_threshold is not None:
                side_threshold = long_threshold
            elif side < 0 and short_threshold is not None:
                side_threshold = 1.0 - short_threshold
        net_pnl = _finite(trade.get("net_pnl"))
        return_pct = _finite(trade.get("return_pct"))
        samples.append(MetaLabelSample(
            opened_at=opened_at,
            closed_at=closed_at,
            side=1 if side > 0 else -1,
            probability=float(probability),
            adjusted_probability=float(adjusted),
            signal_strength=float(_signal_strength(adjusted, side_threshold, side, market_type)),
            net_pnl=float(net_pnl),
            return_pct=float(return_pct),
            profitable=bool(net_pnl > 0.0 and return_pct > 0.0),
        ))
    return samples


def _candidate_thresholds(samples: Sequence[MetaLabelSample]) -> list[float]:
    strengths = [
        max(0.0, float(sample.signal_strength))
        for sample in samples
        if math.isfinite(sample.signal_strength)
    ]
    if not strengths:
        return []
    buckets: dict[float, float] = {}
    for strength in strengths:
        key = round(strength, 12)
        buckets[key] = min(strength, buckets.get(key, strength))
    values = sorted(set(buckets.values()))
    values.append(math.nextafter(max(strengths), math.inf))
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

    from .objective import get_objective

    objective = get_objective(objective_name)
    samples = extract_meta_label_samples(rows, model, strategy, result, market_type=market_type)
    target = _target_precision(objective.name)
    minimum_samples = max(
        META_LABEL_MINIMUM_ACTION_SAMPLES,
        int(objective.min_closed_trades),
    )
    split = _chronological_split(samples, minimum_samples=minimum_samples)
    if split is None:
        policy = {
            "evidence_schema_version": META_LABEL_EVIDENCE_SCHEMA_VERSION,
            "enabled": False,
            "mode": "observe_only",
            "reason": "insufficient_chronological_meta_label_split",
            "target_precision": target,
            "split_schema_version": META_LABEL_SPLIT_SCHEMA_VERSION,
            "source_sample_count": len(samples),
            "minimum_action_samples": int(minimum_samples),
        }
        return MetaLabelReport(
            status="insufficient",
            reason="insufficient_chronological_meta_label_split",
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

    calibration_samples = split.calibration
    validation_samples = split.validation
    best_threshold: float | None = None
    best_rank = (float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    for threshold in _candidate_thresholds(calibration_samples):
        kept = [
            sample
            for sample in calibration_samples
            if sample.signal_strength >= threshold
        ]
        if len(kept) < minimum_samples:
            continue
        precision = _precision(kept)
        mean_return = _mean_return(kept)
        net_pnl = sum(sample.net_pnl for sample in kept)
        if precision < target or mean_return <= 0.0 or net_pnl <= 0.0:
            continue
        skipped = [
            sample
            for sample in calibration_samples
            if sample.signal_strength < threshold
        ]
        avoided_loss = sum(-sample.net_pnl for sample in skipped if sample.net_pnl < 0.0)
        missed_profit = sum(sample.net_pnl for sample in skipped if sample.net_pnl > 0.0)
        rank = (net_pnl + avoided_loss - missed_profit, precision, mean_return, -float(len(kept)))
        if rank > best_rank:
            best_rank = rank
            best_threshold = float(threshold)

    if best_threshold is None:
        best_threshold = math.nextafter(
            max(sample.signal_strength for sample in calibration_samples),
            math.inf,
        )
        status = "observe_only"
        reason = "no_profitable_precision_threshold"
    else:
        status = "trained"
        reason = None

    calibration_take_samples = [
        sample
        for sample in calibration_samples
        if sample.signal_strength >= best_threshold
    ]
    take_samples = [
        sample
        for sample in validation_samples
        if sample.signal_strength >= best_threshold
    ]
    take_bootstrap = _bucket_bootstrap(
        take_samples,
        objective=objective.name,
        action="take",
    )
    take_bootstrap_lower = float(take_bootstrap["mean_ci_lower"])
    if status == "trained":
        if len(take_samples) < minimum_samples:
            status = "observe_only"
            reason = "take_validation_samples_insufficient"
        elif _precision(take_samples) < target:
            status = "observe_only"
            reason = "take_validation_precision_below_target"
        elif _mean_return(take_samples) <= 0.0:
            status = "observe_only"
            reason = "take_validation_expectancy_not_positive"
        elif sum(sample.net_pnl for sample in take_samples) <= 0.0:
            status = "observe_only"
            reason = "take_validation_pnl_not_positive"
        elif take_bootstrap_lower <= 0.0:
            status = "observe_only"
            reason = "take_bootstrap_lower_not_positive"
    proposed_downsize_floor = max(0.0, best_threshold * 0.50)
    proposed_downsize_samples = [
        sample
        for sample in validation_samples
        if proposed_downsize_floor <= sample.signal_strength < best_threshold
    ]
    proposed_downsize_net_pnl = sum(
        sample.net_pnl for sample in proposed_downsize_samples
    )
    proposed_downsize_bootstrap = _bucket_bootstrap(
        proposed_downsize_samples,
        objective=objective.name,
        action="downsize",
    )
    proposed_downsize_bootstrap_lower = float(
        proposed_downsize_bootstrap["mean_ci_lower"]
    )
    downsize_evidence_accepted = bool(
        status == "trained"
        and proposed_downsize_bootstrap_lower > 0.0
        and len(proposed_downsize_samples) >= minimum_samples
        and _mean_return(proposed_downsize_samples) > 0.0
        and proposed_downsize_net_pnl > 0.0
    )
    downsize_floor = (
        proposed_downsize_floor
        if downsize_evidence_accepted
        else float(best_threshold)
    )
    downsize_samples = (
        proposed_downsize_samples
        if downsize_evidence_accepted
        else []
    )
    skip_samples = [
        sample
        for sample in validation_samples
        if sample.signal_strength < downsize_floor
    ]
    skipped_loss = sum(-sample.net_pnl for sample in skip_samples if sample.net_pnl < 0.0)
    skipped_profit = sum(sample.net_pnl for sample in skip_samples if sample.net_pnl > 0.0)
    take_precision = _precision(take_samples)
    take_mean_return = _mean_return(take_samples)
    take_net_pnl = sum(sample.net_pnl for sample in take_samples)
    downsize_precision = _precision(downsize_samples)
    downsize_mean_return = _mean_return(downsize_samples)
    downsize_net_pnl = sum(sample.net_pnl for sample in downsize_samples)
    policy = {
        "evidence_schema_version": META_LABEL_EVIDENCE_SCHEMA_VERSION,
        "enabled": status == "trained",
        "mode": "take_downsize_skip" if status == "trained" else "observe_only",
        "reason": reason,
        "objective": objective.name,
        "target_precision": target,
        "take_threshold": best_threshold,
        "downsize_threshold": downsize_floor,
        "downsize_fraction": 0.5,
        "sample_count": len(samples),
        "split_schema_version": META_LABEL_SPLIT_SCHEMA_VERSION,
        "source_sample_count": len(samples),
        "source_samples_sha256": _sample_digest(
            tuple(split.calibration) + tuple(split.purged) + tuple(split.validation)
        ),
        "calibration_sample_count": len(calibration_samples),
        "purged_sample_count": len(split.purged),
        "policy_validation_sample_count": len(validation_samples),
        "calibration_end_closed_at": max(
            sample.closed_at for sample in calibration_samples
        ),
        "validation_start_opened_at": split.validation_start_opened_at,
        "validation_end_closed_at": max(
            sample.closed_at for sample in validation_samples
        ),
        "calibration_samples_sha256": split.calibration_sha256,
        "purged_samples_sha256": split.purged_sha256,
        "validation_samples_sha256": split.validation_sha256,
        "minimum_action_samples": int(minimum_samples),
        "calibration_take_sample_count": len(calibration_take_samples),
        "calibration_take_precision": float(
            _precision(calibration_take_samples)
        ),
        "calibration_take_mean_return": float(
            _mean_return(calibration_take_samples)
        ),
        "calibration_take_net_pnl": float(
            sum(sample.net_pnl for sample in calibration_take_samples)
        ),
        "take_sample_count": len(take_samples),
        "take_precision": float(take_precision),
        "take_mean_return": float(take_mean_return),
        "take_net_pnl": float(take_net_pnl),
        "take_bootstrap_samples": int(take_bootstrap["samples"]),
        "take_bootstrap_confidence": float(take_bootstrap["confidence"]),
        "take_bootstrap_block_length": int(take_bootstrap["block_length"]),
        "take_bootstrap_mean_return_lower": take_bootstrap_lower,
        "take_bootstrap_mean_return_upper": float(
            take_bootstrap["mean_ci_upper"]
        ),
        "take_bootstrap_positive_mean_probability": float(
            take_bootstrap["positive_mean_probability"]
        ),
        "downsize_evidence_accepted": downsize_evidence_accepted,
        "downsize_sample_count": len(downsize_samples),
        "downsize_precision": float(downsize_precision),
        "downsize_mean_return": float(downsize_mean_return),
        "downsize_net_pnl": float(downsize_net_pnl),
        "downsize_bootstrap_samples": (
            int(proposed_downsize_bootstrap["samples"])
            if downsize_evidence_accepted
            else 0
        ),
        "downsize_bootstrap_confidence": (
            float(proposed_downsize_bootstrap["confidence"])
            if downsize_evidence_accepted
            else 0.0
        ),
        "downsize_bootstrap_block_length": (
            int(proposed_downsize_bootstrap["block_length"])
            if downsize_evidence_accepted
            else 0
        ),
        "downsize_bootstrap_mean_return_lower": (
            proposed_downsize_bootstrap_lower
            if downsize_evidence_accepted
            else 0.0
        ),
        "downsize_bootstrap_mean_return_upper": (
            float(proposed_downsize_bootstrap["mean_ci_upper"])
            if downsize_evidence_accepted
            else 0.0
        ),
        "downsize_bootstrap_positive_mean_probability": (
            float(proposed_downsize_bootstrap["positive_mean_probability"])
            if downsize_evidence_accepted
            else 0.0
        ),
        "proposed_downsize_bootstrap_mean_return_lower": (
            proposed_downsize_bootstrap_lower
        ),
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
        take_precision=take_precision,
        take_mean_return=take_mean_return,
        take_net_pnl=take_net_pnl,
        downsize_precision=downsize_precision,
        downsize_mean_return=downsize_mean_return,
        skipped_loss_avoided=float(skipped_loss),
        skipped_profit_missed=float(skipped_profit),
        policy=policy,
    )
