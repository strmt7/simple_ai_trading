from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_binance_option_parity",
    ROOT / "tools" / "screen_binance_option_parity.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "17"}


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, _url: str, *, timeout: int) -> _RateLimitedResponse:
        assert timeout == 30
        self.calls += 1
        return _RateLimitedResponse()


def test_rate_limit_stops_without_an_automatic_retry() -> None:
    session = _Session()

    with pytest.raises(RuntimeError, match="stopped without retry; Retry-After=17"):
        TOOL._get(session, "https://eapi.binance.com/eapi/v1/ticker")

    assert session.calls == 1


@pytest.mark.parametrize("sweeps", [True, 0, 1.5, 6])
def test_confirmation_sweep_budget_rejects_invalid_values(sweeps: object) -> None:
    with pytest.raises(ValueError, match="confirmation sweeps"):
        TOOL.run(confirmation_sweeps=sweeps)  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", [True, -1, 31, float("nan")])
def test_confirmation_delay_budget_rejects_invalid_values(delay: object) -> None:
    with pytest.raises(ValueError, match="confirmation delay"):
        TOOL.run(confirmation_delay_seconds=delay)  # type: ignore[arg-type]
