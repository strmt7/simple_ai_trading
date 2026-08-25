from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-option-parity-snapshot-v1-2026-08-25.json"
)
SPEC = importlib.util.spec_from_file_location(
    "screen_binance_option_box_parity",
    ROOT / "tools" / "screen_binance_option_box_parity.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _RateLimitedResponse:
    status_code = 429
    headers = {"Retry-After": "23"}


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, _url: str, *, timeout: int) -> _RateLimitedResponse:
        assert timeout == 30
        self.calls += 1
        return _RateLimitedResponse()


def test_source_snapshot_hash_reconstructs() -> None:
    report, file_hash = TOOL._load_source(SOURCE)

    assert report["schema_version"] == TOOL.SOURCE_SCHEMA_VERSION
    assert report["result_sha256"] == (
        "ceca2f61ab1da16285190afcb90c276a10b032fb7d264c90656aaf2f7266c253"
    )
    assert len(file_hash) == 64


def test_source_snapshot_tamper_and_schema_fail_closed(tmp_path: Path) -> None:
    report = json.loads(SOURCE.read_text(encoding="ascii"))
    report["contracts"][0]["strike"] = "1"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(report), encoding="ascii")
    with pytest.raises(ValueError, match="does not reconstruct"):
        TOOL._load_source(tampered)

    report["schema_version"] = "unknown"
    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(json.dumps(report), encoding="ascii")
    with pytest.raises(ValueError, match="unsupported"):
        TOOL._load_source(unsupported)


def test_rate_limit_stops_without_retry() -> None:
    session = _Session()
    with pytest.raises(RuntimeError, match="stopped without retry; Retry-After=23"):
        TOOL._get(session, "https://eapi.binance.com/eapi/v1/depth")
    assert session.calls == 1


@pytest.mark.parametrize("sweeps", [True, 0, 1.5, 6])
def test_confirmation_sweep_budget_rejects_invalid_values(sweeps: object) -> None:
    with pytest.raises(ValueError, match="confirmation sweeps"):
        TOOL.run(source_snapshot=SOURCE, confirmation_sweeps=sweeps)  # type: ignore[arg-type]


@pytest.mark.parametrize("delay", [True, -1, 31, float("nan")])
def test_confirmation_delay_budget_rejects_invalid_values(delay: object) -> None:
    with pytest.raises(ValueError, match="confirmation delay"):
        TOOL.run(source_snapshot=SOURCE, confirmation_delay_seconds=delay)  # type: ignore[arg-type]
