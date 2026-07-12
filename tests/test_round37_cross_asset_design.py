from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
FAILURE = RESEARCH / "round-036-failure-analysis.json"
REGISTRY35 = RESEARCH / "consumed-periods-through-round-035.json"
REGISTRY36 = RESEARCH / "consumed-periods-through-round-036.json"
DESIGN = RESEARCH / "round-037-cross-asset-cost-aware-ai-ablation-design.json"
BINDING = RESEARCH / "round-037-cross-asset-ai-execution-binding.json"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _canonical(payload: dict[str, object], field: str) -> str:
    value = dict(payload)
    value.pop(field)
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round36_failure_and_registry_are_hash_bound_and_fail_closed() -> None:
    failure = _read(FAILURE)
    registry35 = _read(REGISTRY35)
    registry36 = _read(REGISTRY36)

    assert failure["analysis_sha256"] == _canonical(failure, "analysis_sha256")
    assert failure["status"] == "rejected"
    assert failure["economic_evidence"]["positive_regime_slices"] == 0
    assert failure["economic_evidence"]["maximum_top_100_delayed_net_mean_bps"] < 0
    assert all(
        failure[field] is False
        for field in (
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
        )
    )
    assert registry36["registry_sha256"] == _canonical(
        registry36,
        "registry_sha256",
    )
    assert registry36["records"][:-1] == registry35["records"]
    assert registry36["records"][-1]["round"] == 36
    assert registry36["records"][-1]["outcome"] == "rejected"


def test_round37_design_freezes_real_data_roles_costs_models_and_ai() -> None:
    design = _read(DESIGN)
    assert design["design_sha256"] == _canonical(design, "design_sha256")
    assert design["schema_version"] == "cross-asset-cost-aware-ai-ablation-design-v2"
    assert design["round"] == 37
    source = design["source_contract"]
    assert source["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert source["interval"] == "1m"
    assert source["required_checksum_status"] == "verified"
    assert source["required_gap_count"] == 0
    assert source["persistent_feature_copy_permitted"] is False
    roles = {item["role"]: item for item in design["chronological_roles"]}
    assert roles["training"]["start"] == "2022-01-01"
    assert roles["viability"]["end"] == "2025-06-30"
    assert roles["selection_confirmation"]["targets_permitted"] is False
    assert roles["terminal"]["targets_permitted"] is False
    target = design["decision_and_target_contract"]
    assert target["decision_cadence_minutes"] == 5
    assert target["horizons_minutes"] == [15, 30, 60, 120]
    assert target["round_trip_execution_charge_bps"] == 12.0
    assert target["threshold_selection"]["minimum_total_trades"] == 90
    assert target["threshold_selection"]["score"] == (
        "stationary_day_block_bootstrap_lower_95_mean_net_bps"
    )
    model = design["model_contract"]
    assert model["lightgbm"]["gpu_first"] is True
    assert model["model_or_threshold_may_not_be_selected_on_viability_results"]
    ai = design["ai_ablation_contract"]
    assert ai["models"] == ["qwen3:8b", "fino1:8b"]
    assert ai["ai_may_only_veto_or_reduce_risk"] is True
    assert ai["maximum_cases_per_model"] == 270
    assert ai["provider_failure_action"] == "veto"


def test_round37_design_denies_promotion_leverage_and_future_access() -> None:
    design = _read(DESIGN)
    governance = design["governance"]
    for field in (
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "ai_can_create_or_reverse_trades",
        "oracle_feature_or_runtime_label_use_permitted",
        "profitability_target_override_permitted",
    ):
        assert governance[field] is False
    assert all(value is False for value in design["claims"].values())


def test_round37_binding_is_hash_bound_to_implementation_blobs() -> None:
    binding = _read(BINDING)
    design = _read(DESIGN)

    assert binding["binding_sha256"] == _canonical(binding, "binding_sha256")
    assert binding["schema_version"] == (
        "round-037-cross-asset-ai-execution-binding-v1"
    )
    assert binding["design_sha256"] == design["design_sha256"]
    assert binding["implementation_commit"] == (
        "b446ecd49c6f358be9de472ee4a653d7a368f7cc"
    )
    assert len(binding["blobs"]) == 12
    paths = {item["path"] for item in binding["blobs"]}
    assert {
        "src/simple_ai_trading/ai_trade_veto.py",
        "src/simple_ai_trading/cross_asset_cost_data.py",
        "src/simple_ai_trading/cross_asset_cost_model.py",
        "tools/run_cross_asset_cost_aware_ai_ablation.py",
    } <= paths
    assert all(value is False for value in binding["claims"].values())
