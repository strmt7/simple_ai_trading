from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

import tools.publish_adaptive_action_screen as publisher
from tools.publish_adaptive_action_screen import (
    _barrier_svg,
    _development_governance_correction,
    _feature_set_identity,
    _funnel_svg,
    _forecast_svg,
    _gate_summary,
    _progress_identity,
    _progress_rows,
    _publication_narrative,
    _research_progress_svg,
    _tail_svg,
    _threshold_economics_svg,
    _validated_depth_coverage,
    _validated_round30_replay,
)
from tools.run_gross_architecture_screen import _canonical_sha256


def test_development_governance_correction_marks_materialized_labels_consumed() -> None:
    design = {
        "data": {
            "roles": {
                "development_evaluation": {
                    "start": "2023-07-01",
                    "end": "2023-07-06",
                }
            }
        },
        "reserved_terminal": {
            "date": "2023-07-07",
            "included_in_dataset": False,
        },
    }
    report = {
        "round": 30,
        "report_sha256": "b" * 64,
        "development_window_is_consumed": False,
        "terminal_holdout_accessed": False,
        "dataset": {
            "roles": {
                "development_evaluation": {
                    "start": "2023-07-01",
                    "end": "2023-07-06",
                    "rows": 2_000,
                }
            }
        },
        "forecast_diagnostics": {
            "development_base": None,
            "development_stress": None,
        },
        "profile_results": [
            {
                "development_evaluated": False,
                "development_result": None,
            }
        ],
    }

    correction = _development_governance_correction(
        design=design,
        report=report,
        source_report_sha256="a" * 64,
    )

    assert correction["development_role"] == {
        "start": "2023-07-01",
        "end": "2023-07-06",
        "labeled_rows": 2_000,
        "labels_materialized": True,
        "predictions_evaluated": False,
        "profile_metrics_evaluated": False,
        "window_is_consumed": True,
    }
    assert correction["terminal_holdout"]["accessed"] is False
    assert len(correction["correction_sha256"]) == 64

    report["forecast_diagnostics"]["development_stress"] = {"rows": 2_000}
    with pytest.raises(ValueError, match="correction evidence drifted"):
        _development_governance_correction(
            design=design,
            report=report,
            source_report_sha256="a" * 64,
        )


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
            "periods": "2023-05-16..2023-07-06",
            "mean_net_bps": "",
            "best_top_500_exact_after_cost_bps": -6.0,
            "executable_trades": 0,
        }
        for round_number in range(7, 17)
    ]
    charts = (
        _forecast_svg(_forecast_rows()),
        _tail_svg(_forecast_rows()),
        _funnel_svg(
            profiles,
            selection_start_date="2023-06-21",
            selection_end_date="2023-06-25",
            policy_start_date="2023-06-26",
            policy_end_date="2023-06-30",
        ),
        _barrier_svg(
            barrier_rows,
            start_date="2023-05-16",
            end_date="2023-07-06",
        ),
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
    assert "2023-06-21 to 2023-06-25 UTC" in charts[1]
    assert "action" + " funnel" not in charts[2].lower()
    assert "pre-trade risk controls" in charts[2].lower()
    assert "No candidate threshold passed all pre-trade risk controls" in charts[2]
    assert "2023-06-21" in charts[2]
    assert "stop-loss exit" in charts[3]
    assert "take-profit exit" in charts[3]
    assert "profitable net outcome" in charts[3]
    assert "2023-05-16 to 2023-07-06 UTC" in charts[3]
    assert ">stop<" not in charts[3]
    assert ">take<" not in charts[3]
    assert "Rounds 15 through 16" in charts[4]
    assert (
        "Underlying exchange-data windows span 2023-05-16 to 2023-07-06 UTC"
        in charts[4]
    )

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
    category_label = next(
        node for node in texts if node.text == "Policy validation (reused) short"
    )

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
        selection_start_date="2023-06-21",
        selection_end_date="2023-06-25",
        policy_start_date="2023-06-26",
        policy_end_date="2023-06-30",
    )


def test_long_research_progress_uses_compact_nonoverlapping_round_labels() -> None:
    progress = [
        {
            "round": round_number,
            "mean_net_bps": "",
            "best_top_500_exact_after_cost_bps": -8.0,
            "executable_trades": 0,
        }
        for round_number in range(7, 30)
    ]

    chart = _research_progress_svg(progress, round_number=29)

    assert ">R7<" in chart
    assert ">R29<" in chart
    assert ">Round 7<" not in chart
    assert "R denotes research round" in chart


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
            "stress_trades": 1,
            "required_minimum_trades": 20,
            "stress_max_drawdown_bps": abs(value),
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
    assert (
        "least-negative threshold-selection simulation contained one losing trade"
        in summary
    )
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
    assert "least-negative aggressive threshold-selection simulation" in summary
    assert "maker-order economics remain blocked" in next_step


def test_round29_publication_copy_and_feature_identity_are_specific() -> None:
    stage, model_id = _progress_identity(29)
    title, summary, next_step = _publication_narrative(
        29, all_candidate_stress_nets_negative=True
    )

    assert stage == "1800-second holding-horizon cost-amortization ablation"
    assert model_id == "three-seed 1800-second nested-context outcome-mixture"
    assert _feature_set_identity(29) == "l1-tape-causal-v8"
    assert title == "1800-second horizon outcome model abstained"
    assert "calibration net-return ranking deteriorated" in summary
    assert "all eight threshold candidates lost" in summary
    assert "900-second Round 26 baseline" in summary
    assert "state-conditioned horizon selection" in next_step


def test_round30_publication_copy_and_feature_identity_are_specific() -> None:
    stage, model_id = _progress_identity(30)
    title, summary, next_step = _publication_narrative(
        30, all_candidate_stress_nets_negative=False
    )

    assert stage == "900-second LightGBM hurdle architecture challenger"
    assert model_id == "three-seed side-specific LightGBM hurdle-quantile ensemble"
    assert _feature_set_identity(30) == "l1-tape-causal-v8"
    assert title == "LightGBM hurdle ensemble abstained"
    assert "All twelve threshold-selection stress simulations were positive" in summary
    assert "only 1 to 12 trades" in summary
    assert "failed the precommitted minimum-count gate" in summary
    assert "must test broader chronological support and stability" in next_step
    assert "without lowering minimum trade counts" in next_step


def test_threshold_economics_chart_shows_positive_returns_and_failed_support() -> None:
    rows = []
    for profile, required in (
        ("conservative", 12),
        ("regular", 15),
        ("aggressive", 20),
    ):
        for index, quantile in enumerate((0.5, 0.7, 0.85, 0.95), start=1):
            rows.append(
                {
                    "profile": profile,
                    "candidate_available": True,
                    "quantile": quantile,
                    "stress_trades": index,
                    "required_minimum_trades": required,
                    "stress_total_net_bps": 10.0 * index,
                    "stress_max_drawdown_bps": 5.0 * index,
                }
            )

    chart = _threshold_economics_svg(
        rows,
        round_number=30,
        selection_start_date="2023-06-21",
        selection_end_date="2023-06-25",
    )
    root = ET.fromstring(chart)
    namespace = "{http://www.w3.org/2000/svg}"
    text = " ".join(
        node.text or "" for node in root.findall(f"{namespace}text")
    )

    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None
    assert "Positive threshold-selection simulations lacked minimum trade support" in text
    assert "2023-06-21 to 2023-06-25 UTC" in text
    assert "1/12 trades" in text
    assert "4/20 trades" in text
    assert "DD 20.0 bps" in text
    assert ">nan<" not in chart.lower()


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


def test_round29_progress_uses_depth_free_long_horizon_contract(tmp_path) -> None:
    prior = tmp_path / "progress.csv"
    prior.write_text("round\n28\n", encoding="utf-8")
    report = {
        "round": 29,
        "dataset": {
            "valid_barrier_rows": 227_011,
            "barrier_summary": {"spec": {"horizon_seconds": 1_800}},
        },
        "ensemble_models": [{}, {}, {}],
        "profile_results": [
            {"calibration_eligible_rows": 612, "policy_eligible_rows": 635}
        ],
    }

    rows = _progress_rows(prior, report, _forecast_rows())

    assert rows[-1]["feature_set"] == "l1-tape-causal-v8"
    assert rows[-1]["horizon_seconds"] == 1_800
    assert rows[-1]["selection_contaminated"] is True


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


def test_round30_replay_integrity_binds_reports_metrics_and_boosters(
    tmp_path, monkeypatch
) -> None:
    current_root = tmp_path / "current"
    replay_root = tmp_path / "replay"
    baseline_root = tmp_path / "baseline"
    for root in (current_root, replay_root, baseline_root):
        (root / "models").mkdir(parents=True)

    false_claims = {
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    dataset = {
        "rows": 100,
        "event_rows": 80,
        "valid_barrier_rows": 75,
        "cache_key": "c" * 64,
        "source_manifest_fingerprint": "m" * 64,
        "barrier_summary": {"spec": {"horizon_seconds": 900}, "rows": 80},
        "roles": {"train": {"rows": 50}},
    }

    def members(root, suffix: str):
        output = []
        for seed in (29, 43, 71):
            artifact_path = root / "models" / f"seed-{seed}.json"
            artifact_payload = {
                "model_strings": {f"head-{index}": f"tree-{seed}-{index}" for index in range(10)},
                "best_iterations": {f"head-{index}": index + 1 for index in range(10)},
                "metadata_suffix": suffix,
            }
            artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
            output.append(
                {
                    "seed": seed,
                    "model": false_claims,
                    "artifact": {
                        "path": artifact_path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                        "bytes": artifact_path.stat().st_size,
                    },
                }
            )
        return output

    current = {
        **false_claims,
        "round": 30,
        "design_sha256": "2" * 64,
        "status": "rejected",
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": False,
        "dataset": {**dataset, "barrier_targets_sha256": "b" * 64},
        "forecast_diagnostics": {"same": True},
        "profile_results": [{"same": True}],
        "ensemble_models": members(current_root, "current-metadata"),
    }
    replay = {
        **false_claims,
        "round": 30,
        "design_sha256": "1" * 64,
        "status": "rejected",
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": False,
        "dataset": {**dataset, "barrier_targets_sha256": "a" * 64},
        "forecast_diagnostics": {"same": True},
        "profile_results": [{"same": True}],
        "ensemble_models": members(replay_root, "replay-metadata"),
    }
    baseline = {
        **false_claims,
        "round": 26,
        "design_sha256": "6" * 64,
        "status": "rejected",
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": False,
        "dataset": {**dataset, "barrier_targets_sha256": "b" * 64},
    }

    def write_report(root, report):
        report["report_sha256"] = _canonical_sha256(report)
        path = root / "report.json"
        path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        return path

    replay_path = write_report(replay_root, replay)
    write_report(baseline_root, baseline)
    current["report_sha256"] = _canonical_sha256(current)
    (current_root / "report.json").write_text(
        json.dumps(current, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(publisher, "_ROUND26_DESIGN_SHA256", baseline["design_sha256"])
    monkeypatch.setattr(publisher, "_ROUND26_REPORT_SHA256", baseline["report_sha256"])
    design = {
        "design_revision": 2,
        "predecessor_evidence": {
            "design_sha256": replay["design_sha256"],
            "source_report_canonical_sha256": replay["report_sha256"],
            "source_report_file_sha256": hashlib.sha256(
                replay_path.read_bytes()
            ).hexdigest(),
        },
    }

    evidence = _validated_round30_replay(
        current_root,
        current,
        design,
        replay_root,
        baseline_root,
    )

    assert evidence is not None
    assert evidence["checks"]["revision2_target_hash_matches_round26"] is True
    assert evidence["checks"]["booster_strings_equal_between_revisions"] is True
    assert len(evidence["booster_hashes"]) == 3
    assert len(evidence["replay_sha256"]) == 64

    tampered = replay_root / "models" / "seed-29.json"
    tampered.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact integrity failed"):
        _validated_round30_replay(
            current_root,
            current,
            design,
            replay_root,
            baseline_root,
        )
