"""Reusable exact confusion counts without repeating deterministic inference."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThresholdCounts:
    positive_scores: tuple[float, ...]
    negative_scores: tuple[float, ...]
    positive_total: int
    negative_total: int
    other_labels: int

    @classmethod
    def from_predictions(
        cls, predictions: Iterable[tuple[float, int]]
    ) -> ThresholdCounts:
        """Consume one score per row; NaN scores compare false at every threshold."""
        positive: list[float] = []
        negative: list[float] = []
        positive_total = negative_total = other_labels = 0
        for score, label in predictions:
            if label == 1:
                positive_total += 1
                if not math.isnan(score):
                    positive.append(score)
            elif label == 0:
                negative_total += 1
                if not math.isnan(score):
                    negative.append(score)
            else:
                # Match the legacy confusion helper; do not silently relabel data.
                other_labels += 1
        return cls(
            tuple(sorted(positive)),
            tuple(sorted(negative)),
            positive_total,
            negative_total,
            other_labels,
        )

    def at(self, threshold: float) -> tuple[int, int, int, int]:
        """Return TP, FP, TN, FN for score >= threshold, including exact ties."""
        if math.isnan(threshold):
            tp = fp = 0
        else:
            tp = len(self.positive_scores) - bisect_left(
                self.positive_scores, threshold
            )
            fp = len(self.negative_scores) - bisect_left(
                self.negative_scores, threshold
            )
        return (
            tp,
            fp,
            self.negative_total - fp,
            self.positive_total - tp + self.other_labels,
        )
