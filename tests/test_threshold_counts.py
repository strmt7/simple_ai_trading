from dataclasses import FrozenInstanceError
import math
import random

import pytest

from simple_ai_trading.features import ModelRow
from simple_ai_trading.model import TrainedModel, _confusion, _f1, calibrate_threshold
from simple_ai_trading.threshold_counts import ThresholdCounts


def legacy_threshold(rows, model, *, start=0.1, end=0.9, steps=17):
    """Preserved pre-change grid and decision rule as the parity oracle."""
    if not rows or steps <= 1:
        return 0.5
    best_threshold, best_f1 = 0.5, -1.0
    if start < 0.0:
        start = 0.0
    if end > 1.0:
        end = 1.0
    if end <= start:
        end = min(1.0, start + 0.01)
    for index in range(steps):
        threshold = start + (end - start) * index / (steps - 1)
        tp, fp, _, fn = _confusion(rows, model, threshold)
        score = _f1(tp, fp, fn)
        if score > best_f1:
            best_threshold, best_f1 = threshold, score
    return best_threshold


def make_model():
    return TrainedModel(
        weights=[0.7, -0.2],
        bias=0.1,
        feature_dim=2,
        epochs=1,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
    )


def make_rows():
    rng = random.Random(916)
    return [
        ModelRow(
            timestamp=i,
            close=1.0,
            features=(rng.uniform(-3, 3), rng.uniform(-3, 3)),
            label=i % 2,
        )
        for i in range(60)
    ]


def test_counts_match_direct_comparisons_and_are_immutable():
    samples = [
        (0.5, 1),
        (0.5, 0),
        (0.1, 1),
        (0.9, 0),
        (math.nan, 1),
        (math.nan, 0),
        (-math.inf, 1),
        (math.inf, 0),
        (0.7, 2),
    ]
    counts = ThresholdCounts.from_predictions(iter(samples))
    for threshold in [
        -math.inf,
        0.1,
        0.5,
        math.nextafter(0.5, 1.0),
        0.9,
        math.inf,
        math.nan,
    ]:
        tp = fp = tn = fn = 0
        for score, label in samples:
            predicted = int(score >= threshold)
            if predicted == 1 and label == 1:
                tp += 1
            elif predicted == 1 and label == 0:
                fp += 1
            elif predicted == 0 and label == 0:
                tn += 1
            else:
                fn += 1
        assert counts.at(threshold) == (tp, fp, tn, fn)
    assert ThresholdCounts.from_predictions([]).at(0.5) == (0, 0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        counts.positive_total = 7


@pytest.mark.parametrize(
    "start,end,steps",
    [
        (0.05, 0.95, 61),
        (0.1, 0.9, 17),
        (-1, 2, 7),
        (0.8, 0.2, 6),
        (2, 1, 4),
        (-2, -1, 5),
        (math.nan, 0.9, 3),
        (0.1, math.inf, 3),
        (math.inf, 1, 3),
        (0, 1, 1),
        (0, 1, 0),
    ],
)
@pytest.mark.parametrize("inverted", [False, True])
def test_default_model_threshold_is_exactly_unchanged(start, end, steps, inverted):
    rows, model = make_rows(), make_model()
    model.probability_inverted = inverted
    expected = legacy_threshold(rows, model, start=start, end=end, steps=steps)
    actual = calibrate_threshold(rows, model, start=start, end=end, steps=steps)
    assert (math.isnan(actual) and math.isnan(expected)) or actual == expected


def test_predicts_once_per_row_not_once_per_threshold(monkeypatch):
    rows, model = make_rows(), make_model()
    original = TrainedModel.predict_proba
    calls = []

    def counted(self, features):
        calls.append(features)
        return original(self, features)

    monkeypatch.setattr(TrainedModel, "predict_proba", counted)
    expected = legacy_threshold(rows, model, start=0.05, end=0.95, steps=61)
    assert len(calls) == len(rows) * 61
    calls.clear()
    assert calibrate_threshold(rows, model, start=0.05, end=0.95, steps=61) == expected
    assert len(calls) == len(rows)
    assert calibrate_threshold([], model) == 0.5


def test_equal_score_ties_keep_the_first_grid_threshold():
    model = make_model()
    model.weights = [0.0, 0.0]
    model.bias = 0.0
    rows = [
        ModelRow(timestamp=i, close=1.0, features=(0.0, 0.0), label=0) for i in range(5)
    ]
    assert calibrate_threshold(rows, model, start=0.1, end=0.9, steps=17) == 0.1


@pytest.mark.parametrize("override", ["predict", "predict_proba", "subclass"])
def test_custom_prediction_semantics_keep_legacy_path(override):
    class CustomModel(TrainedModel):
        def predict(self, features, threshold):
            return int(features[0] > threshold)

    model = (
        make_model() if override != "subclass" else CustomModel(**vars(make_model()))
    )
    if override == "predict":
        model.predict = lambda features, threshold: int(features[0] > threshold)
    elif override == "predict_proba":
        model.predict_proba = lambda features: 0.25
    rows = make_rows()
    assert calibrate_threshold(rows, model) == legacy_threshold(rows, model)
