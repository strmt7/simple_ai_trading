from __future__ import annotations

import argparse
import json

from simple_ai_trading import cli
from simple_ai_trading.model_blueprint import (
    blueprint_payload,
    model_families,
    research_sources,
    render_blueprint,
    training_lanes,
    validate_blueprint_contract,
)


def test_model_blueprint_contract_is_fail_closed() -> None:
    assert validate_blueprint_contract() == ()
    families = {item.family: item for item in model_families()}

    assert families["foundation_forecaster"].execution_authority == "advisory_features_only"
    assert families["rl_meta_controller"].execution_authority == "none_for_raw_buy_sell_decisions"
    assert families["meta_label_gate"].execution_authority == "pre_entry_skip_or_downsize_only"
    assert "conservative" not in families["rl_meta_controller"].risk_levels
    assert "blocked_until_depth_store" == families["deep_lob_microstructure"].status


def test_model_blueprint_training_lanes_cover_every_family() -> None:
    family_names = {item.family for item in model_families()}
    lane_family_names = {family for lane in training_lanes() for family in lane.families}
    lanes = {lane.lane: lane for lane in training_lanes()}

    assert lane_family_names == family_names
    assert "sequence_forecast_features" in lanes
    assert "timestamped forecast-feature store" in lanes["sequence_forecast_features"].next_build_step
    assert "cross_asset_graph_sequence" in lanes["sequence_forecast_features"].families
    assert "No direct orders" in lanes["sandbox_meta_control"].runtime_limit


def test_model_blueprint_source_catalog_blocks_copying_community_scripts() -> None:
    sources = {source.source_id: source for source in research_sources()}

    assert "patchtst" in sources
    assert sources["finmamba"].applied_to == ("cross_asset_graph_sequence",)
    assert sources["boe_agentic_ai"].source_type == "governance"
    assert sources["lightgbm_opencl"].source_type == "official_docs"
    assert sources["tradingview_lorentzian"].source_type == "community_inspiration"
    assert "do not copy" in sources["tradingview_lorentzian"].usage_policy.lower()


def test_model_blueprint_filters_research_and_risk_level() -> None:
    regular_payload = blueprint_payload(risk_level="balanced", include_research=False)
    family_names = {item["family"] for item in regular_payload["families"]}

    assert "advanced_logistic" in family_names
    assert "adaptive_hybrid_model_zoo" in family_names
    assert "patch_transformer" not in family_names
    assert "rl_meta_controller" not in family_names
    assert [item["name"] for item in regular_payload["risk_blueprints"]] == ["regular"]
    assert regular_payload["training_lanes"]
    assert regular_payload["research_sources"]
    assert "Community TradingView scripts" in regular_payload["source_policy"]


def test_model_blueprint_render_is_operator_readable() -> None:
    text = render_blueprint(risk_level="aggressive")

    assert "Model training blueprint" in text
    assert "patch_transformer" in text
    assert "rl_meta_controller" in text
    assert "not a profitability claim" in text
    assert "Training lanes and promotion gates" in text
    assert "community scripts are inspiration only" in text


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
