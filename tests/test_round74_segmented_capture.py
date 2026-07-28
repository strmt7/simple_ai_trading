from __future__ import annotations

import asyncio

import pytest

from simple_ai_trading.impact_absorption_capture import ImpactCaptureConfig
from simple_ai_trading.impact_absorption_store import (
    IMPACT_CAPTURE_V10_SCHEMA_VERSION,
)
from simple_ai_trading.round74_segmented_capture import (
    ROUND74_SEGMENTED_CAPTURE_DURATION_SECONDS,
    Round74SegmentedCaptureConfig,
    capture_round74_segmented,
)


def test_segmented_capture_changes_only_the_exact_duration_boundary() -> None:
    config = Round74SegmentedCaptureConfig(database="segment.duckdb")
    config.validate()

    assert config.schema_version == IMPACT_CAPTURE_V10_SCHEMA_VERSION
    assert config.mode == "probe"
    assert config.duration_seconds == 1_200.0
    assert ROUND74_SEGMENTED_CAPTURE_DURATION_SECONDS == 1_200.0
    assert config.maximum_reconnects == 0

    with pytest.raises(ValueError, match="capped at 300 seconds"):
        ImpactCaptureConfig(
            database="segment.duckdb",
            schema_version=IMPACT_CAPTURE_V10_SCHEMA_VERSION,
            mode="probe",
            duration_seconds=1_200.0,
            maximum_reconnects=0,
        ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("duration_seconds", 1_199.0),
        ("mode", "qualification"),
        ("schema_version", "round-073-prospective-evidence-v9"),
        ("maximum_reconnects", 1),
    ),
)
def test_segmented_capture_rejects_identity_drift(
    field: str,
    value: object,
) -> None:
    values = {
        **Round74SegmentedCaptureConfig(database="segment.duckdb").__dict__,
        field: value,
    }
    with pytest.raises(ValueError, match="identity differs"):
        Round74SegmentedCaptureConfig(**values).validate()


def test_segmented_capture_delegates_resource_validation() -> None:
    with pytest.raises(ValueError, match="queue capacity"):
        Round74SegmentedCaptureConfig(
            database="segment.duckdb",
            queue_capacity_messages=65_537,
        ).validate()


def test_segmented_capture_rejects_base_configuration_type() -> None:
    config = ImpactCaptureConfig(
        database="segment.duckdb",
        schema_version=IMPACT_CAPTURE_V10_SCHEMA_VERSION,
        mode="probe",
        duration_seconds=300.0,
        maximum_reconnects=0,
    )

    with pytest.raises(TypeError, match="exact configuration"):
        asyncio.run(capture_round74_segmented(config))  # type: ignore[arg-type]
