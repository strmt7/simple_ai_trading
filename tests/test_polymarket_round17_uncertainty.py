from __future__ import annotations

from decimal import Decimal
import hashlib
import json

import numpy as np
import pytest

from simple_ai_trading.polymarket_round17_features import (
    POLYMARKET_ROUND17_FEATURE_NAMES,
    POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
    PolymarketRound17FeatureRow,
)
from simple_ai_trading.polymarket_round17_model import Round17DevelopmentPanel
from simple_ai_trading.polymarket_round17_model import (
    fit_round17_development_pretest,
)
from simple_ai_trading.polymarket_round17_uncertainty import (
    apply_round17_probability_calibration,
    apply_round17_probability_calibration_rows,
    fit_round17_probability_calibration,
    validate_round17_probability_calibration,
)


START_MS = 1_800_057_600_000
DATASET_SHA256 = "d" * 64
TARGET_MANIFEST_SHA256 = "e" * 64


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _feature_values(label: float, row_index: int) -> np.ndarray:
    prior_index = POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up")
    structural_index = POLYMARKET_ROUND17_FEATURE_NAMES.index(
        "structural_probability_up"
    )
    signal_index = POLYMARKET_ROUND17_FEATURE_NAMES.index("chainlink_log_return_1000ms")
    values = np.zeros(
        len(POLYMARKET_ROUND17_FEATURE_NAMES),
        dtype=np.float64,
    )
    values[structural_index] = 0.5
    values[prior_index] = 0.85 if label else 0.15
    values[signal_index] = (2.0 if label else -2.0) + row_index * 0.01
    return values


def _panel(
    role: str,
    *,
    first_event_start_ms: int,
    condition_count: int,
) -> Round17DevelopmentPanel:
    condition_ids: list[str] = []
    event_starts: list[int] = []
    decisions: list[int] = []
    features: list[np.ndarray] = []
    labels: list[float] = []
    for condition_index in range(condition_count):
        label = float(condition_index % 2)
        condition = "0x" + _sha256([role, condition_index])
        event_start = first_event_start_ms + condition_index * 300_000
        for row_index, offset in enumerate((60_000, 120_000, 180_000)):
            condition_ids.append(condition)
            event_starts.append(event_start)
            decisions.append(event_start + offset)
            features.append(_feature_values(label, row_index))
            labels.append(label)
    return Round17DevelopmentPanel(
        role=role,
        condition_ids=np.asarray(condition_ids, dtype=object),
        event_start_ms=np.asarray(event_starts, dtype=np.int64),
        decision_time_ms=np.asarray(decisions, dtype=np.int64),
        features=np.asarray(features, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
        dataset_sha256=DATASET_SHA256,
        target_manifest_sha256=TARGET_MANIFEST_SHA256,
    ).validate()


def _row(
    values: np.ndarray,
    *,
    event_start_ms: int,
    identity: str,
    condition_identity: str | None = None,
    offset_ms: int = 60_000,
) -> PolymarketRound17FeatureRow:
    selected = tuple(float(value) for value in values)
    return PolymarketRound17FeatureRow(
        condition_id="0x"
        + _sha256(["economic", condition_identity or identity]),
        decision_time_ms=event_start_ms + offset_ms,
        admission_sha256="b" * 64,
        causal_segment_sha256="c" * 64,
        feature_names_sha256=POLYMARKET_ROUND17_FEATURE_NAMES_SHA256,
        input_sha256=_sha256(["input", identity]),
        values_sha256=_sha256(list(selected)),
        values=selected,
    )


@pytest.fixture(scope="module")
def round17_calibration() -> tuple[
    dict[str, object],
    Round17DevelopmentPanel,
    dict[str, object],
]:
    train = _panel("train", first_event_start_ms=START_MS, condition_count=40)
    train_end = int(np.max(train.event_start_ms)) + 300_000
    calibration = _panel(
        "tune_calibration",
        first_event_start_ms=train_end + 3_600_000,
        condition_count=20,
    )
    calibration_end = int(np.max(calibration.event_start_ms)) + 300_000
    selection = _panel(
        "tune_selection",
        first_event_start_ms=calibration_end + 3_600_000,
        condition_count=20,
    )
    pretest = fit_round17_development_pretest(
        train,
        calibration,
        selection,
        compute_backend="cpu",
    )
    selection_end = int(np.max(selection.event_start_ms)) + 300_000
    uncertainty_panel = _panel(
        "tune_uncertainty",
        first_event_start_ms=selection_end + 3_600_000,
        condition_count=120,
    )
    artifact = fit_round17_probability_calibration(uncertainty_panel, pretest)
    return pretest, uncertainty_panel, artifact


def test_round17_probability_calibration_is_blocked_hash_bound_and_test_blind(
    round17_calibration: tuple[
        dict[str, object],
        Round17DevelopmentPanel,
        dict[str, object],
    ],
) -> None:
    pretest, _panel_value, artifact = round17_calibration
    verified = validate_round17_probability_calibration(
        artifact,
        model_pretest=pretest,
    )

    assert verified["source_role"] == "tune_uncertainty"
    assert verified["condition_count"] == 120
    assert verified["model_pretest_sha256"] == pretest["pretest_sha256"]
    assert verified["model_development_accepted"] is True
    assert verified["target_manifest_sha256"] == TARGET_MANIFEST_SHA256
    assert verified["bootstrap"]["unit"] == "condition"
    assert verified["test_features_accessed"] is False
    assert verified["test_targets_accessed"] is False
    assert verified["profitability_claim"] is False
    assert verified["paper_trading_authority"] is False
    assert verified["live_trading_authority"] is False


def test_round17_calibration_abstains_outside_supported_probability_regions(
    round17_calibration: tuple[
        dict[str, object],
        Round17DevelopmentPanel,
        dict[str, object],
    ],
) -> None:
    pretest, uncertainty_panel, artifact = round17_calibration
    event_start = int(np.max(uncertainty_panel.event_start_ms)) + 3_900_000
    supported_row = _row(
        _feature_values(1.0, 0),
        event_start_ms=event_start,
        identity="supported",
    )
    neutral = _feature_values(1.0, 0)
    neutral[POLYMARKET_ROUND17_FEATURE_NAMES.index("normalized_market_prior_up")] = 0.5
    neutral[POLYMARKET_ROUND17_FEATURE_NAMES.index("chainlink_log_return_1000ms")] = 0.0
    unsupported_row = _row(
        neutral,
        event_start_ms=event_start + 300_000,
        identity="unsupported",
    )

    supported = apply_round17_probability_calibration(
        artifact,
        pretest,
        supported_row,
        dataset_sha256=DATASET_SHA256,
        event_start_ms=event_start,
    )
    unsupported = apply_round17_probability_calibration(
        artifact,
        pretest,
        unsupported_row,
        dataset_sha256=DATASET_SHA256,
        event_start_ms=event_start + 300_000,
    )
    second_supported_row = _row(
        _feature_values(1.0, 1),
        event_start_ms=event_start,
        identity="supported-second",
        condition_identity="supported",
        offset_ms=120_000,
    )
    supported_batch = apply_round17_probability_calibration_rows(
        artifact,
        pretest,
        (supported_row, second_supported_row),
        dataset_sha256=DATASET_SHA256,
        event_start_ms=event_start,
    )

    assert supported.supported is True
    assert supported_batch[0] == supported
    assert len(supported_batch) == 2
    assert supported.support_condition_count >= 30
    assert supported.envelope.lower_up <= supported.envelope.probability_up
    assert supported.envelope.probability_up <= supported.envelope.upper_up
    assert unsupported.supported is False
    assert unsupported.support_condition_count == 0
    assert unsupported.envelope.lower_up == Decimal("0.000001")
    assert unsupported.envelope.upper_up == Decimal("0.999999")


def test_round17_probability_calibration_rejects_rehashed_authority_drift(
    round17_calibration: tuple[
        dict[str, object],
        Round17DevelopmentPanel,
        dict[str, object],
    ],
) -> None:
    _pretest, _panel_value, source = round17_calibration
    artifact = dict(source)
    artifact["live_trading_authority"] = True
    artifact.pop("calibration_sha256")
    artifact["calibration_sha256"] = _sha256(artifact)

    with pytest.raises(ValueError, match="integrity differs"):
        validate_round17_probability_calibration(artifact)
