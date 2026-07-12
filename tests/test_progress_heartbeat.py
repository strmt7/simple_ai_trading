from __future__ import annotations

import threading
import time

import pytest

from simple_ai_trading.progress_heartbeat import progress_heartbeat


def test_progress_heartbeat_emits_and_stops_deterministically() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    three_events = threading.Event()

    def progress(phase: str, **details: object) -> None:
        events.append((phase, details))
        if len(events) >= 3:
            three_events.set()

    with progress_heartbeat(
        progress,
        phase="source-certificate",
        interval_seconds=0.01,
        details={"corpus": "training"},
    ):
        assert three_events.wait(timeout=1.0)

    count_at_exit = len(events)
    time.sleep(0.025)
    assert len(events) == count_at_exit
    assert len(events) >= 3
    assert [event[1]["heartbeat_count"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert all(event[0] == "source-certificate" for event in events)
    assert all(event[1]["corpus"] == "training" for event in events)
    assert all(event[1]["state"] == "running" for event in events)


def test_progress_heartbeat_surfaces_callback_failure() -> None:
    def broken_progress(_phase: str, **_details: object) -> None:
        raise OSError("status destination unavailable")

    with pytest.raises(RuntimeError, match="heartbeat callback failed"):
        with progress_heartbeat(
            broken_progress,
            phase="model-train",
            interval_seconds=0.005,
        ):
            time.sleep(0.02)


def test_progress_heartbeat_does_not_mask_body_failure() -> None:
    def broken_progress(_phase: str, **_details: object) -> None:
        raise OSError("status destination unavailable")

    with pytest.raises(ValueError, match="model failed"):
        with progress_heartbeat(
            broken_progress,
            phase="model-train",
            interval_seconds=0.005,
        ):
            time.sleep(0.012)
            raise ValueError("model failed")


@pytest.mark.parametrize("interval", [0.0, -1.0, float("inf"), float("nan")])
def test_progress_heartbeat_rejects_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError, match="interval"):
        with progress_heartbeat(lambda *_args, **_kwargs: None, phase="x", interval_seconds=interval):
            pass
