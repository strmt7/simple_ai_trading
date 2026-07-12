from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
FAILURE = RESEARCH / "round-037-failure-analysis.json"
REGISTRY36 = RESEARCH / "consumed-periods-through-round-036.json"
REGISTRY37 = RESEARCH / "consumed-periods-through-round-037.json"
DESIGN = RESEARCH / "round-038-derivatives-hurdle-ai-ablation-design.json"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: dict[str, object], field: str) -> str:
    canonical = dict(value)
    claimed = str(canonical.pop(field))
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == claimed
    return claimed


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_round37_failure_and_consumed_registry_are_hash_bound() -> None:
    failure = _read(FAILURE)
    registry36 = _read(REGISTRY36)
    registry37 = _read(REGISTRY37)

    assert _canonical_sha256(failure, "analysis_sha256") == (
        "39b2347a81d0ba7284cd213dc5f66f356148b7cbf06153c49a55dcd50f2c9374"
    )
    assert _canonical_sha256(registry37, "registry_sha256") == (
        "f5cf5667557b03c377ba9f65b85c61297dff3e53d6a292f1d1ac4edcf2fb71ce"
    )
    assert registry37["records"][:-1] == registry36["records"]
    latest = registry37["records"][-1]
    assert latest["round"] == 37
    assert latest["windows"] == [
        {"start_date": "2022-01-01", "end_date": "2025-06-30"}
    ]
    assert latest["selection_confirmation_accessed"] is False
    assert latest["terminal_2026_accessed"] is False
    assert failure["prediction_evidence"]["selected_thresholds"] == 0
    assert failure["prediction_evidence"]["ai_cases"] == 0
    assert failure["trading_authority"] is False
    assert failure["profitability_claim"] is False


def test_round38_design_freezes_real_derivatives_hurdle_ablation() -> None:
    design = _read(DESIGN)
    failure = _read(FAILURE)
    registry = _read(REGISTRY37)

    assert _canonical_sha256(design, "design_sha256") == (
        "ad81d4f5b570d0367cf2343e5e9ba0afdb1c41400f8368b3396df7e2484882a5"
    )
    assert design["schema_version"] == "derivatives-hurdle-ai-ablation-design-v1"
    assert design["round"] == 38
    assert design["predecessor"]["failure_analysis_canonical_sha256"] == failure[
        "analysis_sha256"
    ]
    assert design["predecessor"]["failure_analysis_file_sha256"] == (
        _file_sha256(FAILURE)
    )
    assert design["governance"]["consumed_period_registry_canonical_sha256"] == (
        registry["registry_sha256"]
    )
    assert design["governance"]["consumed_period_registry_file_sha256"] == (
        _file_sha256(REGISTRY37)
    )

    governance = design["governance"]
    for field in (
        "selection_confirmation_access_permitted",
        "terminal_2026_access_permitted",
        "promotion_permitted",
        "trading_authority_permitted",
        "risk_gate_relaxation_permitted",
        "leverage_permitted",
        "maker_execution_assumption_permitted",
        "ai_can_create_reverse_or_increase_trades",
        "oracle_feature_or_runtime_label_use_permitted",
        "profitability_target_override_permitted",
    ):
        assert governance[field] is False

    source = design["source_contract"]
    assert source["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert source["premium_index_source"]["required_periods_per_symbol"] == 43
    assert source["funding_source"]["required_periods_per_symbol"] == 43
    assert source["required_checksum_status"] == "verified"
    assert source["persistent_feature_copy_permitted"] is False
    assert source["mark_and_index_price_archives_ingested_in_this_round"] is False

    accounting = design["decision_target_and_accounting_contract"]
    assert accounting["horizons_minutes"] == [15, 30, 60, 120]
    assert accounting["round_trip_execution_charge_bps"] == 12.0
    assert accounting["funding_cash_flow_window"] == (
        "entry_time < calc_time <= exit_time"
    )
    assert accounting["portfolio_capital_or_leverage_simulation_permitted"] is False

    model = design["model_contract"]
    assert len(model["architectures"]) == 4
    assert model["fixed_ablation_count"] == 32
    assert model["direct_multiclass"]["objective"] == "multiclass"
    assert model["two_stage_hurdle"]["opportunity_objective"] == "binary"
    assert model["probability_calibration"]["calibration_rows_may_not_be_used"] is True

    thresholds = design["action_threshold_contract"]
    assert len(thresholds["maximum_action_probability_grid"]) == 5
    assert len(thresholds["direction_probability_margin_grid"]) == 4
    assert thresholds["minimum_trades_per_symbol"] == 15
    assert design["viability_gate"]["minimum_nonoverlapping_trades_per_symbol"] == 30
    assert design["ai_ablation_contract"]["models"] == ["qwen3:8b", "fino1:8b"]
    assert design["claims"] == {
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
