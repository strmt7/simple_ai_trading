from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_operator import (
    artifact_writer,
    canonical_sha256,
    source_recomputed_artifact,
)
from tools.run_polymarket_round27_selection import (
    _canonical_sha256,
    _model_identity,
    _selection_economic_config,
    _writer,
)


def _claim(value: int) -> dict[str, object]:
    body: dict[str, object] = {"value": value}
    body["claim_sha256"] = _canonical_sha256(body)
    return body


def test_selection_artifact_writer_is_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.json"
    first = _claim(1)

    assert _writer(path, "claim_sha256")(first) == first["claim_sha256"]
    assert _writer(path, "claim_sha256")(first) == first["claim_sha256"]

    with pytest.raises(ValueError, match="persistence differs"):
        _writer(path, "claim_sha256")(_claim(2))


def test_source_recomputed_artifact_never_trusts_a_restart_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "economic-report.json"
    first: dict[str, object] = {"value": 1}
    first["report_sha256"] = canonical_sha256(first)
    artifact_writer(output, "report_sha256")(first)
    calls = 0

    def _recompute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        second: dict[str, object] = {"value": 2}
        second["report_sha256"] = canonical_sha256(second)
        return second

    with pytest.raises(ValueError, match="persistence differs"):
        source_recomputed_artifact(output, "report_sha256", _recompute)

    assert calls == 1


def test_market_prior_identity_is_contract_bound() -> None:
    first = _model_identity(None, contract_sha256="a" * 64)
    second = _model_identity(None, contract_sha256="b" * 64)

    assert first[0] == "market_prior"
    assert len(first[1]) == 64
    assert first[1] != second[1]


def test_selection_operator_uses_reachable_amended_population_gate() -> None:
    config = _selection_economic_config()

    assert config.minimum_executed_trades == 60
    assert config.minimum_profitable_conditions == 20
