from __future__ import annotations

import argparse
import json

from simple_ai_trading import cli
from simple_ai_trading.model_blueprint import (
    blueprint_payload,
    model_families,
    render_blueprint,
    validate_blueprint_contract,
)


def test_model_blueprint_contract_is_fail_closed() -> None:
    assert validate_blueprint_contract() == ()
    families = {item.family: item for item in model_families()}

    assert families["foundation_forecaster"].execution_authority == "advisory_features_only"
    assert families["rl_meta_controller"].execution_authority == "none_for_raw_buy_sell_decisions"
    assert "conservative" not in families["rl_meta_controller"].risk_levels
    assert "blocked_until_depth_store" == families["deep_lob_microstructure"].status


def test_model_blueprint_filters_research_and_risk_level() -> None:
    regular_payload = blueprint_payload(risk_level="balanced", include_research=False)
    family_names = {item["family"] for item in regular_payload["families"]}

    assert "advanced_logistic" in family_names
    assert "adaptive_hybrid_model_zoo" in family_names
    assert "patch_transformer" not in family_names
    assert "rl_meta_controller" not in family_names
    assert [item["name"] for item in regular_payload["risk_blueprints"]] == ["regular"]


def test_model_blueprint_render_is_operator_readable() -> None:
    text = render_blueprint(risk_level="aggressive")

    assert "Model training blueprint" in text
    assert "patch_transformer" in text
    assert "rl_meta_controller" in text
    assert "not a profitability claim" in text


def test_command_model_blueprint_json(capsys) -> None:
    args = argparse.Namespace(risk_level="conservative", implemented_only=True, json=True)

    assert cli.command_model_blueprint(args) == 0
    payload = json.loads(capsys.readouterr().out)
    family_names = {item["family"] for item in payload["families"]}

    assert payload["risk_level"] == "conservative"
    assert "advanced_logistic" in family_names
    assert "patch_transformer" not in family_names


def test_model_blueprint_parser_contract() -> None:
    args = cli._parse_args(["model-blueprint", "--risk-level", "regular", "--implemented-only"])

    assert args.risk_level == "regular"
    assert args.implemented_only is True
    assert args.func is cli.command_model_blueprint
