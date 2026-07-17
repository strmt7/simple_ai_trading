"""Deterministic dependence-aware resampling shared by model evidence gates."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Sequence


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, float(probability))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def moving_block_bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed_material: str,
) -> dict[str, object]:
    """Estimate a mean interval while preserving local serial dependence."""

    observed = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in observed):
        raise ValueError("moving-block bootstrap values must be finite")
    count = len(observed)
    repetitions = max(200, int(samples))
    confidence_level = max(0.80, min(0.999, float(confidence)))
    if count <= 0:
        return {
            "samples": repetitions,
            "confidence": confidence_level,
            "block_length": 0,
            "mean_ci_lower": 0.0,
            "mean_ci_upper": 0.0,
            "positive_mean_probability": 0.0,
        }
    block_length = max(1, min(count, int(round(math.sqrt(count)))))
    maximum_start = max(0, count - block_length)
    seed = int(
        hashlib.sha256(str(seed_material).encode("utf-8")).hexdigest()[:16],
        16,
    )
    rng = random.Random(seed)
    prefix = [0.0]
    for value in observed:
        prefix.append(prefix[-1] + value)
    complete_blocks, remainder = divmod(count, block_length)
    remainder_maximum_start = max(0, count - remainder) if remainder else 0
    means: list[float] = []
    for _ in range(repetitions):
        block_totals: list[float] = []
        for _block in range(complete_blocks):
            start = rng.randint(0, maximum_start) if maximum_start else 0
            block_totals.append(prefix[start + block_length] - prefix[start])
        if remainder:
            start = (
                rng.randint(0, remainder_maximum_start)
                if remainder_maximum_start
                else 0
            )
            block_totals.append(prefix[start + remainder] - prefix[start])
        means.append(math.fsum(block_totals) / count)
    tail = (1.0 - confidence_level) / 2.0
    return {
        "samples": repetitions,
        "confidence": confidence_level,
        "block_length": block_length,
        "mean_ci_lower": _quantile(means, tail),
        "mean_ci_upper": _quantile(means, 1.0 - tail),
        "positive_mean_probability": sum(value > 0.0 for value in means)
        / repetitions,
    }


__all__ = ["moving_block_bootstrap_mean"]
