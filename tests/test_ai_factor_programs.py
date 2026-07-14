from __future__ import annotations

import json

import numpy as np
import pytest

from simple_ai_trading.ai_factor_programs import (
    apply_factor_transform,
    fit_factor_transform,
    parse_action_conditioned_factor_response_ledger,
    parse_factor_response,
    parse_factor_response_ledger,
    validate_action_conditioned_factor_program,
    validate_factor_program,
)


FEATURE_NAMES = ("return_60m", "volatility_60m", "liquidity_ratio")


def _mapping(name: str, expression: str) -> dict[str, str]:
    return {
        "name": name,
        "expression": expression,
        "mechanism": "Risk-adjusted momentum conditional on available liquidity.",
        "failure_mode": "Momentum can reverse during abrupt regime changes.",
        "expected_horizon": "one_hour",
    }


def _action_mapping(name: str, expression: str) -> dict[str, str]:
    return {
        **_mapping(name, expression),
        "action_symmetry": (
            "Positive values favor the candidate long row and, after the same "
            "action-aligned transform, favor the candidate short row identically."
        ),
    }


def test_factor_program_rejects_dynamic_python_and_unknown_features() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_factor_program(
            _mapping("unsafe_factor", "__import__('os').system('whoami')"),
            model="qwen3:8b",
            feature_names=FEATURE_NAMES,
        )
    with pytest.raises(ValueError, match="unknown feature"):
        validate_factor_program(
            _mapping("unknown_factor", "tanh(future_return)"),
            model="qwen3:8b",
            feature_names=FEATURE_NAMES,
        )
    with pytest.raises(ValueError, match="binary operator"):
        validate_factor_program(
            _mapping("raw_division", "return_60m / volatility_60m"),
            model="qwen3:8b",
            feature_names=FEATURE_NAMES,
        )
    meaningless = _mapping("weak_failure", "tanh(return_60m)")
    meaningless["failure_mode"] = "0"
    with pytest.raises(ValueError, match="failure_mode"):
        validate_factor_program(
            meaningless,
            model="qwen3:8b",
            feature_names=FEATURE_NAMES,
        )


def test_factor_program_rejects_symbol_claim_without_symbol_feature() -> None:
    symbol_claim = _mapping("ethusdt_flow_confirmation", "tanh(return_60m)")
    symbol_claim["mechanism"] = "ETHUSDT flow confirms the target trend over one hour."
    with pytest.raises(ValueError, match="names ethusdt"):
        validate_factor_program(
            symbol_claim,
            model="fino1:8b",
            feature_names=FEATURE_NAMES,
        )


def test_factor_program_rejects_incompatible_additive_units() -> None:
    with pytest.raises(ValueError, match="incompatible additive units"):
        validate_factor_program(
            _mapping("unit_mismatch", "return_60m_bps - liquidity_ratio"),
            model="fino1:8b",
            feature_names=("return_60m_bps", "liquidity_ratio"),
        )


def test_factor_response_is_strict_json_without_markdown_repair() -> None:
    text = json.dumps(
        {
            "factors": [
                _mapping(
                    "risk_adjusted_momentum",
                    "tanh(safe_divide(return_60m, volatility_60m)) * liquidity_ratio",
                )
            ]
        }
    )
    parsed = parse_factor_response(
        text,
        model="fin-r1:8b",
        feature_names=FEATURE_NAMES,
        maximum_factors=4,
    )
    assert len(parsed) == 1
    assert parsed[0].canonical_expression.startswith("tanh(")
    with pytest.raises(ValueError, match="strict JSON"):
        parse_factor_response(
            f"```json\n{text}\n```",
            model="fin-r1:8b",
            feature_names=FEATURE_NAMES,
            maximum_factors=4,
        )
    clipped = validate_factor_program(
        _mapping("clipped_factor", "clip(return_60m, -25.0, 25.0)"),
        model="fin-r1:8b",
        feature_names=FEATURE_NAMES,
    )
    assert clipped.canonical_expression == "clip(return_60m, -25.0, 25.0)"


def test_factor_response_ledger_preserves_valid_siblings() -> None:
    text = json.dumps(
        {
            "factors": [
                _mapping("valid_factor", "tanh(return_60m)"),
                _mapping("invalid_factor", "future_return + 1.0"),
            ]
        }
    )
    accepted, rejected = parse_factor_response_ledger(
        text,
        model="qwen3:8b",
        feature_names=FEATURE_NAMES,
        maximum_factors=4,
    )
    assert [item.name for item in accepted] == ["valid_factor"]
    assert rejected[0]["name"] == "invalid_factor"
    assert "unknown feature" in str(rejected[0]["reason"])


def test_action_conditioned_factor_requires_auditable_side_symmetry() -> None:
    mapping = _action_mapping("aligned_momentum", "tanh(return_60m)")
    program = validate_action_conditioned_factor_program(
        mapping,
        model="qwen3.5:9b",
        feature_names=FEATURE_NAMES,
    )
    assert program.as_factor_program().program_sha256 == program.base_program_sha256
    assert program.program_sha256 != program.base_program_sha256

    mapping["action_symmetry"] = "Uses the same transform for either candidate side."
    with pytest.raises(ValueError, match="both long and short"):
        validate_action_conditioned_factor_program(
            mapping,
            model="qwen3.5:9b",
            feature_names=FEATURE_NAMES,
        )


def test_action_conditioned_response_preserves_valid_siblings() -> None:
    invalid = _action_mapping("future_factor", "future_return + 1.0")
    accepted, rejected = parse_action_conditioned_factor_response_ledger(
        json.dumps(
            {
                "factors": [
                    _action_mapping("aligned_flow", "tanh(return_60m)"),
                    invalid,
                ]
            }
        ),
        model="fin-r1:8b",
        feature_names=FEATURE_NAMES,
        maximum_factors=3,
    )
    assert [item.name for item in accepted] == ["aligned_flow"]
    assert rejected[0]["name"] == "future_factor"


def test_factor_transform_uses_training_bounds_and_deduplicates() -> None:
    first = validate_factor_program(
        _mapping(
            "risk_adjusted_momentum",
            "safe_divide(return_60m, maximum(volatility_60m, 1.0))",
        ),
        model="fin-r1:8b",
        feature_names=FEATURE_NAMES,
    )
    duplicate = validate_factor_program(
        _mapping(
            "same_logic",
            "safe_divide(return_60m, maximum(volatility_60m, 1.0))",
        ),
        model="fino1:8b",
        feature_names=FEATURE_NAMES,
    )
    rows = 1_200
    values = np.column_stack(
        (
            np.linspace(-10.0, 10.0, rows),
            np.linspace(1.0, 3.0, rows),
            np.ones(rows),
        )
    )
    values[-1, 0] = 1_000_000.0
    training = np.zeros(rows, dtype=bool)
    training[:1_100] = True
    transform, fitted, rejections = fit_factor_transform(
        (first, duplicate), values, FEATURE_NAMES, training
    )

    assert fitted.shape == (rows, 1)
    assert fitted[-1, 0] == pytest.approx(transform.upper_bounds[0])
    assert rejections[0]["reason"] == "duplicate_canonical_expression"
    reapplied = apply_factor_transform(transform, values, FEATURE_NAMES)
    np.testing.assert_array_equal(fitted, reapplied)
