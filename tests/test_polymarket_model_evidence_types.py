from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from simple_ai_trading import polymarket_mlp, polymarket_ridge


def test_stored_evidence_type_guards_accept_canonical_values() -> None:
    mapping = {"value": 1}
    sequence = [1, 2]

    assert polymarket_ridge._stored_mapping(mapping, label="value") is mapping
    assert polymarket_ridge._stored_sequence(sequence, label="value") is sequence
    assert polymarket_ridge._stored_int(3, label="value") == 3
    assert polymarket_ridge._stored_float(Decimal("1.25"), label="value") == 1.25


def test_stored_mapping_rejects_non_object_and_non_string_keys() -> None:
    with pytest.raises(TypeError, match="value must be an object"):
        polymarket_ridge._stored_mapping([], label="value")
    with pytest.raises(TypeError, match="value keys must be strings"):
        polymarket_ridge._stored_mapping({1: "value"}, label="value")


def test_stored_sequence_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="value must be an array"):
        polymarket_ridge._stored_sequence({"value": 1}, label="value")


@pytest.mark.parametrize("value", [True, "1"])
def test_stored_integer_rejects_coercion(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an integer"):
        polymarket_ridge._stored_int(value, label="value")


def test_stored_float_rejects_non_numeric_and_non_finite_values() -> None:
    with pytest.raises(TypeError, match="value must be numeric"):
        polymarket_ridge._stored_float(object(), label="value")
    with pytest.raises(TypeError, match="value must be finite"):
        polymarket_ridge._stored_float("nan", label="value")


def test_mlp_bootstrap_rejects_invalid_quantiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        polymarket_mlp.np,
        "quantile",
        lambda *_args, **_kwargs: np.asarray([float("nan"), 1.0]),
    )

    with pytest.raises(RuntimeError, match="bootstrap quantiles are invalid"):
        polymarket_mlp._bootstrap([1.0], seed_material="test")
