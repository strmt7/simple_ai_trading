"""Generate the immutable Round 21 previous-tick design chain."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ROOT / "docs" / "model-research" / "polymarket"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load(name: str) -> dict[str, object]:
    value = json.loads((DESIGNS / name).read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    claimed = str(value.get("design_sha256", ""))
    canonical = dict(value)
    canonical.pop("design_sha256", None)
    if (
        claimed
        != hashlib.sha256(_canonical_json(canonical).encode("ascii")).hexdigest()
    ):
        raise ValueError(f"{name} has an invalid canonical hash")
    return value


def _seal(value: dict[str, object]) -> tuple[dict[str, object], str]:
    canonical = deepcopy(value)
    canonical.pop("design_sha256", None)
    digest = hashlib.sha256(_canonical_json(canonical).encode("ascii")).hexdigest()
    canonical["design_sha256"] = digest
    return canonical, digest


def _write_or_check(name: str, value: dict[str, object], *, write: bool) -> None:
    path = DESIGNS / name
    expected = json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
    if write:
        if path.exists() and path.read_text(encoding="ascii") != expected:
            raise ValueError(f"refusing to replace divergent immutable design: {name}")
        path.write_text(expected, encoding="ascii", newline="\n")
        return
    if not path.is_file() or path.read_text(encoding="ascii") != expected:
        raise ValueError(f"generated Round 21 design differs: {name}")


def _replace_parent(value: dict[str, object], key: str, digest: str) -> None:
    parents = value.get("parents")
    if not isinstance(parents, dict) or key not in parents:
        raise ValueError(f"missing parent {key}")
    parents[key] = digest


def _supersede(
    source: str,
    target: str,
    schema: str,
    mutate: Callable[[dict[str, object]], None],
    *,
    write: bool,
) -> str:
    value = _load(source)
    value["schema_version"] = schema
    if "supersedes" in value:
        value["supersedes"] = str(_load(source)["schema_version"])
    mutate(value)
    sealed, digest = _seal(value)
    _write_or_check(target, sealed, write=write)
    return digest


def generate(*, write: bool) -> dict[str, str]:
    hashes: dict[str, str] = {}

    def feature_policy(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        value["supersession"] = {
            "round21_causal_feature_policy_v1_sha256": old_hash,
            "change": "bind_return_and_variation_windows_to_causal_previous_tick_asof_anchors",
            "feature_names_or_widths_changed": False,
            "receipt_time_causality_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
            "edge_or_profitability_inferred_from_change": False,
        }
        clock = value["clock_and_causality"]
        assert isinstance(clock, dict)
        clock.update(
            {
                "window_endpoint_anchor": "latest_valid_receipt_at_or_before_window_start",
                "window_return_population": "returns_ending_strictly_after_window_start_and_at_or_before_decision_time",
                "window_bipower_population": "adjacent_returns_both_ending_strictly_after_window_start_and_at_or_before_decision_time",
                "window_count_population": "receipts_at_or_after_window_start_and_at_or_before_decision_time",
                "missing_previous_tick": "first_in_window_receipt_is_anchor_with_zero_preceding_return",
            }
        )

    hashes["feature_policy"] = _supersede(
        "round-021-causal-feature-policy-v1.json",
        "round-021-causal-feature-policy-v2.json",
        "polymarket-round21-causal-feature-policy-v2",
        feature_policy,
        write=write,
    )

    def core_corpus(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        value["supersedes"] = "polymarket-round21-core-corpus-materialization-design-v1"
        value["supersession"] = {
            "round21_core_corpus_materialization_design_v1_sha256": old_hash,
            "change": "bind_materialized_rows_to_causal_previous_tick_feature_policy",
            "condition_admission_changed": False,
            "storage_layout_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
        }
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )

    hashes["core_corpus"] = _supersede(
        "round-021-core-corpus-materialization-design-v1.json",
        "round-021-core-corpus-materialization-design-v2.json",
        "polymarket-round21-core-corpus-materialization-design-v2",
        core_corpus,
        write=write,
    )

    def execution(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        value["supersedes"] = "polymarket-round21-executable-action-policy-v1"
        value["supersession"] = {
            "round21_executable_action_policy_v1_sha256": old_hash,
            "change": "bind_unchanged_execution_stress_matrix_to_previous_tick_feature_policy",
            "scenario_matrix_changed": False,
            "fees_or_latency_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
        }
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )

    hashes["execution"] = _supersede(
        "round-021-executable-action-policy-v1.json",
        "round-021-executable-action-policy-v2.json",
        "polymarket-round21-executable-action-policy-v2",
        execution,
        write=write,
    )

    def model(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )
        value["supersession"] = {
            "round21_matched_model_design_v7_sha256": old_hash,
            "change": "bind_unchanged_candidate_program_to_causal_previous_tick_window_semantics",
            "probability_basis_ablation_inherited_unchanged": True,
            "candidate_count_changed": False,
            "predictive_model_families_changed": False,
            "training_schedule_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
            "edge_or_profitability_inferred_from_change": False,
        }

    hashes["model"] = _supersede(
        "round-021-matched-model-design-v7.json",
        "round-021-matched-model-design-v8.json",
        "polymarket-round21-matched-model-design-v8",
        model,
        write=write,
    )

    def envelope(value: dict[str, object]) -> None:
        value.pop("design_sha256")
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])

    hashes["envelope"] = _supersede(
        "round-021-probability-envelope-design-v4.json",
        "round-021-probability-envelope-design-v5.json",
        "polymarket-round21-probability-envelope-design-v5",
        envelope,
        write=write,
    )

    def policy(value: dict[str, object]) -> None:
        value.pop("design_sha256")
        _replace_parent(value, "round21_execution_policy_sha256", hashes["execution"])
        _replace_parent(
            value,
            "round21_probability_envelope_design_sha256",
            hashes["envelope"],
        )

    hashes["policy"] = _supersede(
        "round-021-multi-action-policy-design-v6.json",
        "round-021-multi-action-policy-design-v7.json",
        "polymarket-round21-multi-action-policy-design-v7",
        policy,
        write=write,
    )

    def replay(value: dict[str, object]) -> None:
        value.pop("design_sha256")
        _replace_parent(value, "round21_execution_policy_sha256", hashes["execution"])
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(
            value,
            "round21_probability_envelope_design_sha256",
            hashes["envelope"],
        )
        _replace_parent(value, "round21_multi_action_policy_sha256", hashes["policy"])

    hashes["replay"] = _supersede(
        "round-021-economic-replay-design-v4.json",
        "round-021-economic-replay-design-v5.json",
        "polymarket-round21-economic-replay-design-v5",
        replay,
        write=write,
    )

    def comparison(value: dict[str, object]) -> None:
        value.pop("design_sha256")
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )

    hashes["comparison"] = _supersede(
        "round-021-matched-economic-comparison-design-v4.json",
        "round-021-matched-economic-comparison-design-v5.json",
        "polymarket-round21-matched-economic-comparison-design-v5",
        comparison,
        write=write,
    )

    def ai_veto(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        parents = value["parents"]
        assert isinstance(parents, dict)
        parents.pop("round21_ai_veto_design_v4_sha256")
        parents["round21_ai_veto_design_v5_sha256"] = old_hash
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(value, "round21_multi_action_policy_sha256", hashes["policy"])
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_unchanged_finite_ai_program_to_previous_tick_feature_chain"
        )

    hashes["ai_veto"] = _supersede(
        "round-021-ai-veto-design-v5.json",
        "round-021-ai-veto-design-v6.json",
        "polymarket-round21-ai-veto-design-v6",
        ai_veto,
        write=write,
    )

    def ai_selection(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        parents = value["parents"]
        assert isinstance(parents, dict)
        parents.pop("round21_ai_candidate_selection_design_v4_sha256")
        parents["round21_ai_candidate_selection_design_v5_sha256"] = old_hash
        _replace_parent(value, "round21_ai_veto_design_sha256", hashes["ai_veto"])
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_unchanged_finite_ai_selection_to_previous_tick_feature_chain"
        )

    hashes["ai_selection"] = _supersede(
        "round-021-ai-candidate-selection-design-v5.json",
        "round-021-ai-candidate-selection-design-v6.json",
        "polymarket-round21-ai-candidate-selection-design-v6",
        ai_selection,
        write=write,
    )

    def ai_schedule(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        parents = value["parents"]
        assert isinstance(parents, dict)
        parents.pop("round21_ai_historical_schedule_design_v4_sha256")
        parents["round21_ai_historical_schedule_design_v5_sha256"] = old_hash
        parents.pop("round21_ai_veto_design_v5_sha256")
        parents["round21_ai_veto_design_v6_sha256"] = hashes["ai_veto"]
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_unchanged_virtual_timing_to_round21_ai_veto_design_v6"
        )

    hashes["ai_schedule"] = _supersede(
        "round-021-ai-historical-schedule-design-v5.json",
        "round-021-ai-historical-schedule-design-v6.json",
        "polymarket-round21-ai-historical-schedule-design-v6",
        ai_schedule,
        write=write,
    )

    def sealed(value: dict[str, object]) -> None:
        old_hash = str(value.pop("design_sha256"))
        parents = value["parents"]
        assert isinstance(parents, dict)
        parents.pop("round21_terminal_sealed_evaluation_design_v4_sha256")
        parents["round21_terminal_sealed_evaluation_design_v5_sha256"] = old_hash
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )
        _replace_parent(value, "round21_execution_policy_sha256", hashes["execution"])
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(value, "round21_multi_action_policy_sha256", hashes["policy"])
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )
        _replace_parent(
            value, "round21_matched_comparison_design_sha256", hashes["comparison"]
        )
        _replace_parent(value, "round21_ai_veto_design_sha256", hashes["ai_veto"])
        _replace_parent(
            value,
            "round21_ai_historical_schedule_design_sha256",
            hashes["ai_schedule"],
        )
        _replace_parent(
            value, "round21_ai_selection_design_sha256", hashes["ai_selection"]
        )
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_sealed_evaluation_to_causal_previous_tick_feature_chain"
        )

    hashes["sealed"] = _supersede(
        "round-021-terminal-sealed-evaluation-design-v5.json",
        "round-021-terminal-sealed-evaluation-design-v6.json",
        "polymarket-round21-terminal-sealed-evaluation-design-v6",
        sealed,
        write=write,
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    hashes = generate(write=args.write)
    print(json.dumps(hashes, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
