from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_paper_plan as paper_plan


def _artifact(*, confirmed: bool = True) -> SimpleNamespace:
    gates = {
        "validation_probability_improved": True,
        "untouched_test_probability_improved": True,
        "minimum_confirmatory_test_time_groups_met": confirmed,
        "after_cost_execution_improved": True,
        "after_cost_model_improved_at_every_stress_latency": True,
        "all_positions_officially_settled": True,
        "all_order_outcomes_terminal": True,
        "ai_enabled": False,
        "ai_uplift_accepted": False,
    }
    execution = {
        "report_sha256": "3" * 64,
        "config": {
            "submission_latency_ms": 100,
            "maximum_execution_observation_delay_ms": 500,
            "maximum_book_age_ms": 2_000,
            "order_ttl_ms": 30_000,
        },
        "trades": [],
    }
    return SimpleNamespace(
        artifact_sha256="1" * 64,
        payload={
            "run_id": "paper-plan-run",
            "feature_dataset": {
                "config": {"allow_segmented_gaps": False},
            },
            "evidence_gates": gates,
            "execution_latency_sensitivity": {
                "primary_network_latency_ms": 100,
                "policies": {
                    "baseline": {
                        "100": {"net_realized_pnl_quote": "1"},
                    },
                    "model": {
                        "100": {"net_realized_pnl_quote": "2"},
                    },
                },
            },
        },
        executions={
            "baseline": {
                **deepcopy(execution),
                "report_sha256": "4" * 64,
            },
            "model": execution,
        },
    )


def _source() -> dict[str, object]:
    return {
        "report_sha256": "2" * 64,
        "recorder_report_sha256": "5" * 64,
        "execution_report_sha256_by_policy_and_latency": {
            "baseline": {"100": "4" * 64},
            "model": {"100": "3" * 64},
        },
    }


def _patch_validation(monkeypatch, artifact, source) -> None:
    monkeypatch.setattr(
        paper_plan,
        "validate_polymarket_model_artifact",
        lambda _path: artifact,
    )
    monkeypatch.setattr(
        paper_plan,
        "validate_polymarket_source_verification",
        lambda _payload, **_kwargs: source,
    )


def test_model_paper_plan_is_promotion_and_source_gated(
    tmp_path,
    monkeypatch,
) -> None:
    verification_path = tmp_path / "verification.json"
    verification_path.write_text("{}\n", encoding="utf-8")
    artifact = _artifact()
    source = _source()
    _patch_validation(monkeypatch, artifact, source)

    plan = paper_plan.build_polymarket_paper_plan(
        tmp_path / "artifact.json",
        verification_path,
    )

    assert plan.policy == "model"
    assert plan.confirmed_for_paper_run is True
    assert plan.research_override is False
    assert plan.blocking_reasons == ()
    assert plan.artifact_sha256 == "1" * 64
    assert plan.source_verification_sha256 == "2" * 64
    assert plan.recorder_report_sha256 == "5" * 64
    assert plan.allow_segmented_gaps is False
    assert len(plan.plan_sha256) == 64

    with pytest.raises(ValueError, match="must be auto, baseline, model, or ai"):
        paper_plan.build_polymarket_paper_plan(
            tmp_path / "artifact.json",
            verification_path,
            policy="profile_model",
        )

    source["execution_report_sha256_by_policy_and_latency"] = {
        "baseline": {"100": "4" * 64},
        "model": {"100": "6" * 64},
    }
    with pytest.raises(ValueError, match="not source-verified"):
        paper_plan.build_polymarket_paper_plan(
            tmp_path / "artifact.json",
            verification_path,
        )


def test_unconfirmed_model_requires_explicit_research_override(
    tmp_path,
    monkeypatch,
) -> None:
    verification_path = tmp_path / "verification.json"
    verification_path.write_text("{}\n", encoding="utf-8")
    _patch_validation(monkeypatch, _artifact(confirmed=False), _source())

    with pytest.raises(ValueError, match="no confirmed Polymarket paper policy"):
        paper_plan.build_polymarket_paper_plan(
            tmp_path / "artifact.json",
            verification_path,
        )

    plan = paper_plan.build_polymarket_paper_plan(
        tmp_path / "artifact.json",
        verification_path,
        allow_unconfirmed_research=True,
    )
    assert plan.confirmed_for_paper_run is False
    assert plan.research_override is True
    assert plan.blocking_reasons == ("minimum_confirmatory_test_time_groups_met",)
    assert plan.trading_authority is False
    assert plan.live_order_authority is False
    assert plan.profitability_claim is False
