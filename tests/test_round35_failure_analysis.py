from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
FAILURE = RESEARCH / "round-035-failure-analysis.json"
REGISTRY34 = RESEARCH / "consumed-periods-through-round-034.json"
REGISTRY35 = RESEARCH / "consumed-periods-through-round-035.json"
ROUND35_PUBLICATION_COMMIT = "b5a2d1c369a237f2dc36fbfb52901454b30b045c"
ROUND35_SCREEN_PATH = "docs/model-research/action-value/latest/screen.json"


def _read(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_git_json(commit: str, path: str) -> tuple[bytes, dict[str, object]]:
    raw = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return raw, payload


def _canonical_sha256(payload: dict[str, object], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_round35_failure_analysis_is_hash_bound_and_fail_closed() -> None:
    failure = _read(FAILURE)

    assert failure["analysis_sha256"] == _canonical_sha256(
        failure,
        "analysis_sha256",
    )
    assert failure["status"] == "rejected"
    assert failure["scope"]["symbol"] == "BTCUSDT"
    assert failure["scope"]["policy_prediction_or_metrics_accessed"] is False
    assert failure["scope"]["development_prediction_or_metrics_accessed"] is False
    assert failure["scope"]["distant_confirmation_source_materialized"] is False
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert failure[field] is False


def test_round35_failure_metrics_equal_the_validated_source_report() -> None:
    failure = _read(FAILURE)
    raw, screen = _read_git_json(ROUND35_PUBLICATION_COMMIT, ROUND35_SCREEN_PATH)
    source_evidence = failure["source_evidence"]

    assert hashlib.sha256(raw).hexdigest() == source_evidence["report_file_sha256"]
    assert (
        screen["report_canonical_sha256"] == source_evidence["report_canonical_sha256"]
    )
    source = {item["variant"]: item for item in screen["variant_results"]}
    direct_fields = {
        "pooled_direction_auc": "pooled_direction_auc",
        "direction_accuracy": "direction_accuracy",
        "brier_score": "conditional_long_probability_brier_score",
        "daily_auc_minimum": "daily_auc_minimum",
        "daily_auc_median": "daily_auc_median",
        "daily_auc_standard_deviation": "daily_auc_standard_deviation",
        "days_above_chance": "days_above_chance",
        "all_routed_mean_stress_net_bps": "all_routed_mean_stress_net_bps",
    }

    for recorded in failure["variants"]:
        result = source[recorded["variant"]]
        metrics = result["metrics"]
        for recorded_name, source_name in direct_fields.items():
            assert recorded[recorded_name] == metrics[source_name]
        for count in (100, 500, 1000):
            assert (
                recorded[f"frozen_opportunity_top_{count}_mean_stress_net_bps"]
                == metrics["frozen_opportunity_ranked"][str(count)][
                    "mean_stress_net_bps"
                ]
            )
        assert (
            recorded["candidate_confidence_top_500_mean_stress_net_bps"]
            == (metrics["candidate_confidence_ranked"]["500"]["mean_stress_net_bps"])
        )
        assert recorded["gates_passed"] == 8 - len(result["rejection_reasons"])
        assert recorded["status"] == "rejected"


def test_round35_failure_analysis_rejects_cherry_picking_and_model_escalation() -> None:
    failure = _read(FAILURE)
    by_name = {item["variant"]: item for item in failure["variants"]}
    isolated = by_name["noncycle_utility_margin"]

    assert isolated["frozen_opportunity_top_500_mean_stress_net_bps"] > 0.0
    assert isolated["frozen_opportunity_top_100_mean_stress_net_bps"] < 0.0
    assert isolated["candidate_confidence_top_500_mean_stress_net_bps"] < 0.0
    assert isolated["daily_auc_minimum"] < 0.48
    assert isolated["gates_passed"] == 2
    assert any(
        "Do not select the noncycle utility-margin variant" in response
        for response in failure["rejected_responses"]
    )
    experiment = failure["next_experiment"]
    assert experiment["model_architecture_selection_permitted"] is False
    assert experiment["promotion_permitted"] is False
    assert experiment["hyperparameter_search_permitted"] is False
    assert experiment["risk_gate_relaxation_permitted"] is False
    assert experiment["leverage_permitted"] is False
    assert experiment["maker_execution_assumption_permitted"] is False
    assert experiment["oracle_feature_or_runtime_label_use_permitted"] is False
    assert experiment["candidate_horizons_seconds"] == [5, 15, 30, 60, 120, 300, 900]


def test_round35_consumed_registry_extends_round34_without_mutation() -> None:
    previous = _read(REGISTRY34)
    current = _read(REGISTRY35)

    assert current["registry_sha256"] == _canonical_sha256(
        current,
        "registry_sha256",
    )
    assert current["records"][:-1] == previous["records"]
    assert current["records"][-1] == {
        "round": 35,
        "status": "consumed",
        "outcome": "rejected",
        "design_sha256": (
            "db027483eed1329554bac8c3be057c488bc1899348978ea8b199c5796db6cbaa"
        ),
        "binding_sha256": (
            "ac94719659d15c18b5f465b1b2e09cd6d6bb980768f7822fb040462f3a2d6f6b"
        ),
        "report_sha256": (
            "1c6d2b7e5914ef62b110ea6095661f460cdc75f215aee07e04b1cdc0979499ac"
        ),
        "windows": [{"start_date": "2023-05-16", "end_date": "2023-07-06"}],
    }


def test_current_model_docs_use_the_active_v8_feature_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    research = (ROOT / "docs" / "MODEL_RESEARCH_AND_OPTIMIZATION.md").read_text(
        encoding="utf-8"
    )

    for document in (readme, research):
        assert "Feature contract `l1-tape-causal-v8`" in document
        assert "107 features" in document or "107-feature order" in document
        assert "Feature contract `l1-tape-causal-" + "v7`" not in document
    assert "current v16/v8 and design-v2 contract" in research
