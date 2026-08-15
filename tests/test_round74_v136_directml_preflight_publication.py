from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/simple_ai_trading"
RESEARCH = ROOT / "docs/model-research/action-value"
ARTIFACT = RESEARCH / (
    "round-074-v136-directml-pretraining-host-preflight-2026-08-10.json"
)
MODEL_DESIGN = RESEARCH / "round-074-event-sequence-model-design-v136.json"
PROBE = ROOT / "tools/probe_round74_v136_pretraining_directml.py"
FROZEN_PROBE = RESEARCH / "evidence/round-074-v136-probe-source.py.txt"
FILE_SHA256 = "95d4c9416c80f1232e54241c012f57ee5303a0cadd5cc1ce0cd51f04a802eef6"
PREFLIGHT_SHA256 = "ee63e32cb6b20c2f65bfb6c3567773c21366481fd2746fdcc95c0606b18946fe"
CURRENT_PROBE_SHA256 = (
    "d9e77e09d585fa312be5f2bc895ed8a7e86efe1f53dd75a7db469c03cd88fa26"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _canonical_file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def _strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return [value] if isinstance(value, str) else []


def test_round74_v136_directml_preflight_is_self_hashed_and_source_bound() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    canonical = dict(value)
    claimed = canonical.pop("preflight_sha256")
    assert claimed == PREFLIGHT_SHA256
    assert claimed == _canonical_sha256(canonical)
    sources = value["source_binding"]
    assert sources["model_design_file_sha256"] == _canonical_file_sha256(MODEL_DESIGN)
    assert sources["impact_absorption_event_pretraining.py"] == (
        _canonical_file_sha256(SOURCE / "impact_absorption_event_pretraining.py")
    )
    assert sources["impact_absorption_event_model.py"] == _canonical_file_sha256(
        SOURCE / "impact_absorption_event_model.py"
    )
    assert sources["impact_absorption_event_features.py"] == _canonical_file_sha256(
        SOURCE / "impact_absorption_event_features.py"
    )
    assert sources["distributional_tcn_model.py"] == _canonical_file_sha256(
        SOURCE / "distributional_tcn_model.py"
    )
    assert sources["probe_round74_v136_pretraining_directml.py"] == (
        _canonical_file_sha256(FROZEN_PROBE)
    )
    assert _canonical_file_sha256(PROBE) == CURRENT_PROBE_SHA256
    assert ast.dump(ast.parse(PROBE.read_text(encoding="utf-8"))) == ast.dump(
        ast.parse(FROZEN_PROBE.read_text(encoding="utf-8"))
    )
    assert not any(Path(text).is_absolute() for text in _strings(value))


def test_round74_v136_directml_preflight_covers_every_production_view() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert value["status"] == (
        "passed_target_free_directml_forward_backward_optimizer_probe"
    )
    assert value["backend"]["kind"] == "directml"
    assert value["backend"]["accelerated"] is True
    assert value["backend"]["vendor"] == "AMD Radeon RX 9070 XT"
    assert value["backend"]["cpu_fallback_warning_count"] == 0
    assert value["backend"]["warning_count"] == 0
    protocol = value["protocol"]
    assert protocol["candidate_ids"] == [
        "causal_event_tcn",
        "causal_event_attention",
    ]
    assert protocol["feature_views"] == [
        "clock_neutral",
        "full",
        "market_state_clock_neutral",
        "market_state_with_clock",
    ]
    assert protocol["candidate_feature_view_combinations"] == 8
    assert protocol["objective_schema_version"] == (
        "round-074-causal-next-event-pretraining-v5"
    )
    assert protocol["continuous_loss_dimensions"] == (
        "unmasked_continuous_features_only"
    )
    assert protocol["masked_target_policy"] == ("exclude_from_loss_not_zero_impute")
    results = value["results"]
    assert len(results) == 8
    assert {(result["candidate_id"], result["feature_view"]) for result in results} == {
        (candidate, feature_view)
        for candidate in protocol["candidate_ids"]
        for feature_view in protocol["feature_views"]
    }
    for result in results:
        assert result["active_continuous_output_count"] >= 1
        assert result["active_perturbation_minimum_loss_delta"] > 0.0
        assert result["all_optimized_gradients_finite"] is True
        assert result["gradient_norm_before_clip"] > 0.0
        assert result["optimizer_step_completed"] is True
        if result["masked_perturbation_applicable"]:
            assert result["masked_continuous_output_count"] >= 1
            assert result["masked_perturbation_max_abs_loss_delta"] == 0.0
        else:
            assert result["feature_view"] == "full"
            assert result["masked_perturbation_max_abs_loss_delta"] is None
    assert value["evidence_boundary"] == {
        "ai_uplift_tested": False,
        "database_opened": False,
        "financial_edge_tested": False,
        "market_data_used": False,
        "model_selection_output_used": False,
        "predictive_accuracy_tested": False,
        "profitability_claim": False,
        "realized_financial_target_used": False,
        "trading_authority": False,
    }
