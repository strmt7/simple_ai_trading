from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from simple_ai_trading import polymarket_fit_claim, polymarket_mlp, polymarket_ridge
from simple_ai_trading.polymarket_mlp import PolymarketMLPReport
from simple_ai_trading.polymarket_ridge import PolymarketPolicyMetrics


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


def test_mlp_selected_validation_metrics_require_exactly_one_match() -> None:
    trial = cast(PolymarketPolicyMetrics, SimpleNamespace(threshold=0.6))
    selected = cast(
        PolymarketMLPReport,
        SimpleNamespace(selected_threshold=0.6, validation_trials=(trial,)),
    )
    no_trade = cast(
        PolymarketMLPReport,
        SimpleNamespace(selected_threshold=None, validation_trials=()),
    )

    assert polymarket_mlp._selected_validation_metrics(selected) is trial
    assert polymarket_mlp._selected_validation_metrics(no_trade) is None
    for trials in ((), (trial, trial)):
        malformed = cast(
            PolymarketMLPReport,
            SimpleNamespace(selected_threshold=0.6, validation_trials=trials),
        )
        with pytest.raises(ValueError, match="threshold is missing or duplicated"):
            polymarket_mlp._selected_validation_metrics(malformed)


@pytest.mark.parametrize(
    "row",
    [None, (), (True,), (-1,), (2,), ("1",), (0, 1)],
)
def test_fit_claim_table_count_requires_one_binary_integer(
    row: tuple[object, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="report-table count"):
        polymarket_fit_claim._required_table_count(row)


def test_fit_claim_table_count_accepts_absent_and_present_states() -> None:
    assert polymarket_fit_claim._required_table_count((0,)) == 0
    assert polymarket_fit_claim._required_table_count((1,)) == 1


def test_fit_claim_identity_rejects_unregistered_report_queries() -> None:
    with pytest.raises(ValueError, match="fit-claim identity is invalid"):
        polymarket_fit_claim._validated_identity(
            experiment="round9_ridge",
            parent_sha256="a" * 64,
            contract_sha256="b" * 64,
            dataset_sha256="c" * 64,
            report_table="unregistered_report",
            report_parent_column="parent_sha256",
        )
