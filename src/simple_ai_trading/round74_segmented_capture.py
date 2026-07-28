"""First-class 20-minute configuration for segmented Round 74 capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .impact_absorption_capture import (
    ImpactCaptureConfig,
    ImpactCaptureSupervisorReport,
    capture_round73_supervised,
)
from .impact_absorption_event_segmented_cohort import (
    ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS,
)
from .impact_absorption_store import IMPACT_CAPTURE_V10_SCHEMA_VERSION


ROUND74_SEGMENTED_CAPTURE_DURATION_SECONDS = (
    ROUND74_SEGMENTED_COHORT_CAPTURE_DURATION_NS / 1_000_000_000
)


@dataclass(frozen=True)
class Round74SegmentedCaptureConfig(ImpactCaptureConfig):
    """V10 probe mechanics extended to one exact transport-unit duration."""

    schema_version: str = IMPACT_CAPTURE_V10_SCHEMA_VERSION
    mode: Literal["probe", "qualification"] = "probe"
    duration_seconds: float = ROUND74_SEGMENTED_CAPTURE_DURATION_SECONDS
    maximum_reconnects: int = 0

    def validate(self) -> None:
        if (
            self.schema_version != IMPACT_CAPTURE_V10_SCHEMA_VERSION
            or self.mode != "probe"
            or float(self.duration_seconds)
            != ROUND74_SEGMENTED_CAPTURE_DURATION_SECONDS
            or isinstance(self.maximum_reconnects, bool)
            or not isinstance(self.maximum_reconnects, int)
            or self.maximum_reconnects != 0
        ):
            raise ValueError("Round 74 segmented capture identity differs")
        # Delegate every unchanged resource and protocol bound to the frozen
        # v10 validator with only its diagnostic duration substituted.
        ImpactCaptureConfig(
            database=self.database,
            schema_version=self.schema_version,
            mode=self.mode,
            duration_seconds=300.0,
            request_timeout_seconds=self.request_timeout_seconds,
            queue_capacity_messages=self.queue_capacity_messages,
            frame_message_limit=self.frame_message_limit,
            frame_uncompressed_limit_bytes=self.frame_uncompressed_limit_bytes,
            frame_flush_seconds=self.frame_flush_seconds,
            compressed_payload_cap_bytes=self.compressed_payload_cap_bytes,
            database_size_cap_bytes=self.database_size_cap_bytes,
            writer_stall_seconds=self.writer_stall_seconds,
            source_stall_seconds=self.source_stall_seconds,
            duckdb_memory_limit=self.duckdb_memory_limit,
            duckdb_threads=self.duckdb_threads,
            maximum_reconnects=self.maximum_reconnects,
        ).validate()


async def capture_round74_segmented(
    config: Round74SegmentedCaptureConfig,
) -> ImpactCaptureSupervisorReport:
    """Run one zero-reconnect transport unit through the unchanged v10 engine."""

    if not isinstance(config, Round74SegmentedCaptureConfig):
        raise TypeError("Round 74 segmented capture requires its exact configuration")
    config.validate()
    return await capture_round73_supervised(config)


__all__ = [
    "ROUND74_SEGMENTED_CAPTURE_DURATION_SECONDS",
    "Round74SegmentedCaptureConfig",
    "capture_round74_segmented",
]
