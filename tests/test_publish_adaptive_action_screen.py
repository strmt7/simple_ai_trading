from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

from tools.publish_adaptive_action_screen import (
    _barrier_svg,
    _feature_set_identity,
    _funnel_svg,
    _forecast_svg,
    _gate_summary,
    _progress_identity,
    _progress_rows,
    _publication_narrative,
    _research_progress_svg,
    _tail_svg,
    _validated_depth_coverage,
)
from tools.run_gross_architecture_screen import _canonical_sha256


def _forecast_rows() -> list[dict[str, object]]:
    output = []
    for role, start, end in (
        ("calibration", "2023-06-21", "2023-06-25"),
        ("policy", "2023-06-26", "2023-06-30"),
    ):
        for side, auc, top_100, top_500 in (
            ("long", 0.638, -10.95, -10.66),
            ("short", 0.601, -7.90, -13.49),
        ):
            output.append(
                {
                    "role": role,
                    "scenario": "stress",
                    "side": side,
                    "start_date": start,
                    "end_date": end,
                    "auc": auc,
                    "top_100_mean_net_bps": top_100,
                    "top_500_mean_net_bps": top_500,
                }
            )
    return output


def test_round16_charts_are_accessible_parseable_and_truthfully_labeled() -> None:
    profiles = [
        {
            "profile": "conservative",
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 2,
        },
        {
            "profile": "regular",
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 3,
        },
        {
            "profile": "aggressive",
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 13,
        },
    ]
    barrier_rows = [
        {
            "scenario": scenario,
            "side": side,
            "positive_ratio": 0.21,
            "horizon": 150,
            "stop": 50,
            "take": 20,
            "ambiguous_stop": 0,
            "protection_gap_stop": 1 if scenario == "stress" else 0,
        }
        for scenario in ("base", "stress")
        for side in ("long", "short")
    ]
    progress = [
        {
            "round": round_number,
            "mean_net_bps": "",
            "best_top_500_exact_after_cost_bps": -6.0,
            "executable_trades": 0,
        }
        for round_number in range(7, 17)
    ]
    charts = (
        _forecast_svg(_forecast_rows()),
        _tail_svg(_forecast_rows()),
        _funnel_svg(profiles),
        _barrier_svg(barrier_rows),
        _research_progress_svg(progress),
    )

    for chart in charts:
        root = ET.fromstring(chart)
        assert root.attrib["role"] == "img"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
        assert ">nan<" not in chart.lower()
        assert '="nan"' not in chart.lower()
    assert "Every displayed mean is negative" in charts[1]
    assert "action" + " funnel" not in charts[2].lower()
    assert "pre-trade risk controls" in charts[2].lower()
    assert "No candidate threshold passed all pre-trade risk controls" in charts[2]
    assert "2023-06-21" in charts[2]
    assert "Rounds 15 through 16" in charts[4]

    empty_summary = _gate_summary(
        [
            {
                "profile": profile,
                "calibration_eligible_rows": 0,
                "calibration_threshold_candidates": 0,
                "calibration_threshold_accepted": False,
                "policy_trades": 0,
                "development_evaluated": False,
            }
            for profile in ("conservative", "regular", "aggressive")
        ]
    )
    assert empty_summary["highest_eligible_rows"] == 0
    assert empty_summary["highest_eligible_profile"] == "none"


def test_round17_titles_and_extreme_tail_label_have_clearance() -> None:
    rows = _forecast_rows()
    rows[-1]["top_100_mean_net_bps"] = -21.99
    tail = ET.fromstring(_tail_svg(rows, round_number=17))
    namespace = "{http://www.w3.org/2000/svg}"
    texts = [node for node in tail.findall(f"{namespace}text") if node.text]
    value_label = next(node for node in texts if node.text == "-21.99")
    category_label = next(node for node in texts if node.text == "Out-of-sample short")

    assert float(value_label.attrib["y"]) + 20.0 < float(category_label.attrib["y"])
    assert tail.find(f"{namespace}title").text.startswith("Round 17")
    assert "Round 17" in _forecast_svg(rows, round_number=17)
    assert "Round 17" in _funnel_svg(
        [
            {
                "profile": profile,
                "calibration_eligible_rows": 0,
                "policy_eligible_rows": 0,
            }
            for profile in ("conservative", "regular", "aggressive")
        ],
        round_number=17,
    )


def test_ranked_tail_chart_handles_a_positive_subset_without_invalid_geometry() -> None:
    rows = _forecast_rows()
    rows[2]["top_100_mean_net_bps"] = 0.94

    chart = _tail_svg(rows, round_number=23)
    root = ET.fromstring(chart)
    namespace = "{http://www.w3.org/2000/svg}"
    bars = [
        node
        for node in root.findall(f"{namespace}rect")
        if node.attrib.get("width") == "58"
    ]

    assert "Ranked-tail economics were mixed and not stable" in chart
    assert "Every displayed mean is negative" not in chart
    assert len(bars) == 8
    assert all(float(node.attrib["height"]) >= 0.0 for node in bars)


def test_round18_gate_summary_preserves_nonzero_candidates() -> None:
    rows = [
        {
            "profile": "conservative",
            "calibration_eligible_rows": 0,
            "calibration_threshold_candidates": 0,
            "calibration_threshold_accepted": False,
            "policy_trades": 0,
            "development_evaluated": False,
        },
        {
            "profile": "regular",
            "calibration_eligible_rows": 0,
            "calibration_threshold_candidates": 0,
            "calibration_threshold_accepted": False,
            "policy_trades": 0,
            "development_evaluated": False,
        },
        {
            "profile": "aggressive",
            "calibration_eligible_rows": 24,
            "calibration_threshold_candidates": 4,
            "calibration_threshold_accepted": False,
            "policy_trades": 0,
            "development_evaluated": False,
        },
    ]

    thresholds = [
        {
            "candidate_available": True,
            "stress_total_net_bps": value,
        }
        for value in (-98.86, -103.68, -59.16, -59.16)
    ]
    summary = _gate_summary(rows, thresholds)

    assert summary["highest_eligible_rows"] == 24
    assert summary["highest_eligible_profile"] == "aggressive"
    assert summary["candidate_count"] == 4
    assert summary["accepted_count"] == 0
    assert summary["all_candidate_stress_nets_negative"] is True
    assert "Aggressive (24)" in str(summary["sentence"])
    assert "all failed the stress-test acceptance criteria" in str(summary["sentence"])


def test_round20_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(20)
    title, summary, next_step = _publication_narrative(
        20, all_candidate_stress_nets_negative=True
    )

    assert stage == "parameter-matched direction-specific representation ablation"
    assert model_id == "three-seed independent-long-short outcome-mixture"
    assert title == "direction-specific outcome model abstained"
    assert "independent long and short representations" in summary
    assert "decision-objective alignment" in next_step


def test_round21_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(21)
    title, summary, next_step = _publication_narrative(
        21, all_candidate_stress_nets_negative=False
    )

    assert stage == "sampled pairwise net-return ranking ablation"
    assert model_id == (
        "three-seed independent-long-short pairwise-ranked outcome-mixture"
    )
    assert title == "pairwise net-return ranking model abstained"
    assert "eliminated threshold-selection eligibility" in summary
    assert "calibrated positive expected-return separation" in next_step


def test_round22_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(22)
    title, summary, next_step = _publication_narrative(
        22, all_candidate_stress_nets_negative=True
    )

    assert stage == "additive pairwise net-return regularization"
    assert model_id == (
        "three-seed calibration-preserving additive-pairwise outcome-mixture"
    )
    assert title == "additive net-return ranking model abstained"
    assert "every threshold-selection simulation" in summary
    assert "Further ranking-loss tuning is not justified" in next_step


def test_round23_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(23)
    title, summary, next_step = _publication_narrative(
        23, all_candidate_stress_nets_negative=False
    )

    assert stage == "bounded causal temporal-attention ablation"
    assert model_id == "three-seed causal-temporal-attention outcome-mixture"
    assert title == "causal temporal-attention outcome model abstained"
    assert "every nonempty threshold-selection simulation" in summary
    assert "isolated positive policy tail is insufficient" in next_step


def test_round24_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(24)
    title, summary, next_step = _publication_narrative(
        24, all_candidate_stress_nets_negative=False
    )

    assert stage == "UTC-session-local ranking ablation"
    assert model_id == (
        "three-seed session-local-ranked causal-attention outcome-mixture"
    )
    assert title == "session-local ranking model abstained"
    assert "all threshold-selection eligibility disappeared" in summary
    assert "Session-local ranking is rejected" in next_step


def test_round25_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(25)
    title, summary, next_step = _publication_narrative(
        25, all_candidate_stress_nets_negative=True
    )

    assert stage == "parameter-matched soft mixture-of-experts ablation"
    assert model_id == "three-seed soft-expert causal-attention outcome-mixture"
    assert title == "soft mixture-of-experts outcome model abstained"
    assert "every threshold-selection candidate lost money" in summary
    assert "near-maximum routing entropy" in summary
    assert "Homogeneous experts with near-uniform routing are rejected" in next_step


def test_round26_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(26)
    title, summary, next_step = _publication_narrative(
        26, all_candidate_stress_nets_negative=True
    )

    assert stage == "nested 15-second and 30-second expert-context ablation"
    assert model_id == "three-seed nested-context soft-expert outcome-mixture"
    assert title == "nested-context soft-expert outcome model abstained"
    assert "all eight threshold candidates still lost" in summary
    assert "routing remained close to uniform" in summary
    assert "900-second holding horizon" in next_step


def test_round27_publication_copy_is_specific() -> None:
    stage, model_id = _progress_identity(27)
    title, summary, next_step = _publication_narrative(
        27, all_candidate_stress_nets_negative=True
    )

    assert stage == "300-second holding-horizon alignment ablation"
    assert model_id == "three-seed 300-second nested-context outcome-mixture"
    assert title == "300-second horizon outcome model abstained"
    assert "positive outcomes became rarer" in summary
    assert "least-negative trace contained one losing trade" in summary
    assert "retained taker-cost model" in next_step


def test_round28_publication_copy_and_feature_identity_are_specific() -> None:
    stage, model_id = _progress_identity(28)
    title, summary, next_step = _publication_narrative(
        28, all_candidate_stress_nets_negative=True
    )

    assert stage == "sampled aggregate-depth feature ablation"
    assert model_id == "three-seed sampled-depth nested-context outcome-mixture"
    assert _feature_set_identity(28) == "l1-tape-aggregate-depth-causal-v9"
    assert title == "sampled aggregate-depth outcome model abstained"
    assert "all eight threshold candidates lost" in summary
    assert "least-negative aggressive trace" in summary
    assert "maker-order economics remain blocked" in next_step
    with pytest.raises(ValueError, match="undefined for Round 29"):
        _progress_identity(29)
    with pytest.raises(ValueError, match="undefined for Round 29"):
        _feature_set_identity(29)
    with pytest.raises(ValueError, match="undefined for Round 29"):
        _publication_narrative(29, all_candidate_stress_nets_negative=True)


def test_progress_uses_verified_barrier_horizon_instead_of_a_fixed_value(
    tmp_path,
) -> None:
    prior = tmp_path / "progress.csv"
    prior.write_text("round\n26\n", encoding="utf-8")
    report = {
        "round": 27,
        "dataset": {
            "valid_barrier_rows": 230_393,
            "barrier_summary": {"spec": {"horizon_seconds": 300}},
        },
        "ensemble_models": [{}, {}, {}],
        "profile_results": [
            {
                "calibration_eligible_rows": 40,
                "policy_eligible_rows": 40,
            }
        ],
    }

    rows = _progress_rows(prior, report, _forecast_rows())

    assert rows[-1]["horizon_seconds"] == 300
    assert rows[-1]["feature_set"] == "l1-tape-causal-v8"
    report["dataset"]["barrier_summary"]["spec"]["horizon_seconds"] = 0
    with pytest.raises(ValueError, match="progress horizon is invalid"):
        _progress_rows(prior, report, _forecast_rows())


def test_round28_progress_uses_sampled_depth_feature_contract(tmp_path) -> None:
    prior = tmp_path / "progress.csv"
    prior.write_text("round\n27\n", encoding="utf-8")
    report = {
        "round": 28,
        "dataset": {
            "valid_barrier_rows": 229_001,
            "barrier_summary": {"spec": {"horizon_seconds": 900}},
        },
        "ensemble_models": [{}, {}, {}],
        "profile_results": [
            {"calibration_eligible_rows": 42, "policy_eligible_rows": 58}
        ],
    }

    rows = _progress_rows(prior, report, _forecast_rows())

    assert rows[-1]["feature_set"] == "l1-tape-aggregate-depth-causal-v9"
    assert rows[-1]["horizon_seconds"] == 900


def test_round28_depth_coverage_validator_accepts_zero_invalid_rows(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report = {
        "round": 28,
        "report_sha256": "a" * 64,
        "corpus_certificate_sha256": "b" * 64,
        "dataset": {"cache_key": "c" * 64, "rows": 10},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    coverage = {
        "schema_version": "sampled-aggregate-depth-coverage-v1",
        "round": 28,
        "feature_version": "l1-tape-aggregate-depth-causal-v9",
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "source_report_canonical_sha256": "a" * 64,
        "corpus_certificate_sha256": "b" * 64,
        "cache_key": "c" * 64,
        "rows": 10,
        "available_rows": 8,
        "unavailable_rows": 2,
        "invalid_rows": 0,
        "maximum_age_ms": 60_000,
        "full_l2_order_book": False,
        "queue_position_evidence": False,
        "maker_fill_evidence": False,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
    }
    coverage["audit_sha256"] = _canonical_sha256(coverage)
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")

    assert _validated_depth_coverage(coverage_path, report_path, report) == coverage
