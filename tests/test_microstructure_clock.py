"""Clock probes preserve exact arithmetic and reject invalid timing evidence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import simple_ai_trading.microstructure_data as microstructure


BASE_MS = 1_700_000_000_000
BASE_NS = BASE_MS * 1_000_000


class ClockResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class ClockSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return ClockResponse(self.payload)

    def close(self):
        self.closed = True


def install_clocks(monkeypatch, *, rtt_ns=2, wall_elapsed_ns=2):
    walls = iter([BASE_NS + 1_000_001, BASE_NS + 1_000_001 + wall_elapsed_ns] * 3)
    monotonic = iter([100, 100 + rtt_ns] * 3)
    monkeypatch.setattr(
        microstructure,
        "time",
        SimpleNamespace(
            time_ns=lambda: next(walls),
            perf_counter_ns=lambda: next(monotonic),
            time=lambda: (BASE_MS + 1) / 1000,
        ),
    )


def test_probe_subtracts_exact_epochs_before_converting_offset(monkeypatch):
    install_clocks(monkeypatch)
    session = ClockSession({"serverTime": BASE_MS + 1})
    result = microstructure.probe_binance_futures_clock(session=session, samples=1)
    assert result.offset_ms == -0.000002
    assert result.median_rtt_ms == result.minimum_rtt_ms == 0.000002
    assert result.measured_at_ms == BASE_MS + 1
    assert result.samples == len(session.calls) == 3
    assert session.closed is False


@pytest.mark.parametrize("value", [True, 1.5, str(BASE_MS), 0, -1, None, 10**30])
def test_probe_rejects_lossy_or_invalid_native_server_time(monkeypatch, value):
    install_clocks(monkeypatch)
    session = ClockSession({"serverTime": value})
    with pytest.raises(ValueError, match="serverTime"):
        microstructure.probe_binance_futures_clock(session=session, samples=1)
    assert len(session.calls) == 1
    assert not session.closed


@pytest.mark.parametrize("payload", [[], None, BASE_MS])
def test_probe_rejects_nonobject_response(monkeypatch, payload):
    install_clocks(monkeypatch)
    with pytest.raises(ValueError, match="must be an object"):
        microstructure.probe_binance_futures_clock(session=ClockSession(payload))


@pytest.mark.parametrize("rtt,elapsed", [(-1, 2), (2, -1), (2, 2_000_002)])
def test_probe_rejects_regression_and_wall_clock_discontinuity(
    monkeypatch, rtt, elapsed
):
    install_clocks(monkeypatch, rtt_ns=rtt, wall_elapsed_ns=elapsed)
    session = ClockSession({"serverTime": BASE_MS})
    monkeypatch.setattr(microstructure.requests, "Session", lambda: session)
    with pytest.raises(ValueError, match="regressed|disagree"):
        microstructure.probe_binance_futures_clock()
    assert session.closed
    assert len(session.calls) == 1


def test_probe_uses_declared_clock_disagreement_limit_and_closes_owned_session(
    monkeypatch,
):
    install_clocks(monkeypatch, wall_elapsed_ns=1_000_002)
    session = ClockSession({"serverTime": BASE_MS + 1})
    monkeypatch.setattr(microstructure.requests, "Session", lambda: session)
    result = microstructure.probe_binance_futures_clock(
        samples=3, wall_clock_disagreement_limit_ms=1.0
    )
    assert result.samples == 3
    assert session.closed
    assert all(call[1] == {"timeout": 5.0} for call in session.calls)


@pytest.mark.parametrize(
    "arguments",
    [
        {"samples": True},
        {"samples": 0},
        {"samples": 1.5},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": 0},
        {"wall_clock_disagreement_limit_ms": float("inf")},
        {"wall_clock_disagreement_limit_ms": False},
    ],
)
def test_probe_rejects_invalid_policy_before_session_creation(monkeypatch, arguments):
    def unexpected_session():
        pytest.fail("invalid policy must fail before creating a session")

    monkeypatch.setattr(microstructure.requests, "Session", unexpected_session)
    with pytest.raises(ValueError):
        microstructure.probe_binance_futures_clock(**arguments)
