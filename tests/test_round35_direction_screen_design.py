from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
DESIGN = RESEARCH / "round-035-consumed-direction-screen-design.json"
FAILURE = RESEARCH / "round-034-failure-analysis.json"
REGISTRY = RESEARCH / "consumed-periods-through-round-034.json"
ROUND34 = RESEARCH / "round-034-three-action-utility-design.json"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round35_direction_screen_is_hash_bound_to_round34_rejection() -> None:
    design = _read(DESIGN)
    failure = _read(FAILURE)
    registry = _read(REGISTRY)
    canonical = dict(design)
    claimed = canonical.pop("design_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert design["round"] == 35
    assert design["phase"] == "pre_architecture_consumed_data_screen"
    assert (
        design["predecessor"]["failure_analysis_canonical_sha256"]
        == failure["analysis_sha256"]
    )
    assert (
        design["predecessor"]["failure_analysis_file_sha256"]
        == hashlib.sha256(FAILURE.read_bytes()).hexdigest()
    )
    governance = design["governance"]
    assert (
        governance["consumed_period_registry_canonical_sha256"]
        == registry["registry_sha256"]
    )
    assert (
        governance["consumed_period_registry_file_sha256"]
        == hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    )


def test_round35_direction_screen_feature_contract_is_exact_and_factorial() -> None:
    design = _read(DESIGN)
    feature_sets = design["feature_sets"]
    source_names = tuple(MICROSTRUCTURE_FEATURE_NAMES)
    source_contract = {
        "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
        "feature_names": source_names,
    }

    assert design["source_contract"]["feature_contract_sha256"] == _canonical_sha256(
        source_contract
    )
    assert feature_sets["full"]["expected_feature_count"] == len(source_names) == 107
    excluded = tuple(
        feature_sets["full_without_deterministic_cycles"]["excluded_features"]
    )
    compact = tuple(
        feature_sets["compact_observed_microstructure"]["included_features"]
    )
    assert len(excluded) == len(set(excluded)) == 7
    assert set(excluded) <= set(source_names)
    assert len(source_names) - len(excluded) == 100
    assert len(compact) == len(set(compact)) == 68
    assert set(compact) <= set(source_names)
    assert not set(compact) & set(excluded)
    variants = design["variants"]
    assert {(variant["feature_set"], variant["weighting"]) for variant in variants} == {
        (feature_set, weighting)
        for feature_set in (
            "full",
            "full_without_deterministic_cycles",
            "compact_observed_microstructure",
        )
        for weighting in ("uniqueness", "utility_margin")
    }


def test_round35_direction_screen_changes_only_direction_learning() -> None:
    design = _read(DESIGN)
    round34 = _read(ROUND34)

    assert design["event_sampler"] == round34["event_sampler"]
    assert design["model"]["parameters"] == {
        key: value
        for key, value in round34["model"]["lightgbm"].items()
        if key
        not in {
            "lower_quantile",
            "upper_quantile",
            "calibration_fraction",
            "gpu_use_dp_required",
            "backend",
            "cpu_fallback_permitted",
        }
    }
    target = design["direction_target"]
    assert target["future_outcomes_available_to_model_features_or_runtime"] is False
    mirror = design["mirror_equivariance"]
    assert mirror["separate_long_and_short_models_permitted"] is False
    assert "exact_probability_ties_unrouted" in mirror["selected_side"]


def test_round35_direction_screen_cannot_access_or_promote_later_stages() -> None:
    design = _read(DESIGN)
    governance = design["governance"]
    roles = design["data_roles"]

    assert governance["all_target_dates_already_consumed"] is True
    assert governance["post_hoc_discovery_only"] is True
    assert governance["promotion_permitted"] is False
    assert governance["untouched_period_access_permitted"] is False
    assert governance["variant_budget"] == 6
    assert governance["seed_budget"] == 1
    assert governance["hyperparameter_search_permitted"] is False
    assert governance["risk_gate_relaxation_permitted"] is False
    assert governance["leverage_permitted"] is False
    assert roles["train"]["permitted_use"] == "fit"
    assert roles["early_stop"]["permitted_use"] == "early_stopping_only"
    assert roles["calibration"]["permitted_use"] == "post_hoc_screen_evaluation"
    for role in ("policy", "development", "distant_confirmation"):
        assert roles[role]["access_permitted"] is False
    assert design["claims"] == {
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }


def test_round35_direction_screen_research_is_scoped_and_qualified() -> None:
    design = _read(DESIGN)
    research = design["research_basis"]

    assert len(research) >= 6
    assert all(item["url"].startswith("https://") for item in research)
    assert any(item["review_status"] == "official_documentation" for item in research)
    assert any(
        item["review_status"] == "recent_unreviewed_preprint" for item in research
    )
    assert any(
        "no reported performance is imported" in item["use"] for item in research
    )
    assert len(design["limitations"]) >= 5
