from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_polymarket_round27_selection import (
    _canonical_sha256,
    _model_identity,
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


def test_market_prior_identity_is_contract_bound() -> None:
    first = _model_identity(None, contract_sha256="a" * 64)
    second = _model_identity(None, contract_sha256="b" * 64)

    assert first[0] == "market_prior"
    assert len(first[1]) == 64
    assert first[1] != second[1]
