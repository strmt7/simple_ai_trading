"""Generate the immutable Round 21 receipt-age design chain."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from typing import Callable

from supersede_round21_previous_tick_designs import (
    _load,
    _replace_parent,
    _seal,
    _write_or_check,
)


def _supersede(
    source: str,
    target: str,
    schema: str,
    mutate: Callable[[dict[str, object], str], None],
    *,
    write: bool,
) -> str:
    value = deepcopy(_load(source))
    old_hash = str(value.pop("design_sha256"))
    value["schema_version"] = schema
    mutate(value, old_hash)
    sealed, digest = _seal(value)
    _write_or_check(target, sealed, write=write)
    return digest


def _set_predecessor(
    value: dict[str, object],
    *,
    prefix: str,
    key: str,
    digest: str,
) -> None:
    parents = value.get("parents")
    if not isinstance(parents, dict):
        raise ValueError("missing design parents")
    for parent in tuple(parents):
        if parent.startswith(prefix):
            parents.pop(parent)
    parents[key] = digest


def generate(*, write: bool) -> dict[str, str]:
    hashes: dict[str, str] = {}

    def feature_policy(value: dict[str, object], old_hash: str) -> None:
        clob = value["polymarket_clob"]
        optional = value["optional_binance"]
        assert isinstance(clob, dict) and isinstance(optional, dict)
        static = clob["static_families"]
        assert isinstance(static, list)
        static.append("causal_book_receipt_age")
        optional.update(
            {
                "causal_bbo_receipt_age_ms": True,
                "signed_spot_minus_usdm_bbo_receipt_skew_ms": True,
                "receipt_age_clock": "local_utc_receipt_time",
            }
        )
        value["feature_schema"] = {
            "core_width": 164,
            "spot_width": 78,
            "usdm_width": 88,
            "total_width": 330,
            "core_names_sha256": (
                "75c2f664763dfe35f32ca177a7f0fb921367f9a9e8deddb80569d2d2de081811"
            ),
            "spot_names_sha256": (
                "5cdfdbe4ccf4832388c967fa81a9a6280b24f04090e15d76d01e8c0f17cad6ef"
            ),
            "usdm_names_sha256": (
                "0ec64a4fd3acd4f77a29a44a68bfc67477702202a40b2ec8dd3ae265ed0bdcdd"
            ),
        }
        value["supersession"] = {
            "round21_causal_feature_policy_v2_sha256": old_hash,
            "change": (
                "add_causal_quote_receipt_ages_and_signed_spot_usdm_bbo_receipt_skew"
            ),
            "feature_names_or_widths_changed": True,
            "receipt_time_causality_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
            "edge_or_profitability_inferred_from_change": False,
        }

    hashes["feature_policy"] = _supersede(
        "round-021-causal-feature-policy-v2.json",
        "round-021-causal-feature-policy-v3.json",
        "polymarket-round21-causal-feature-policy-v3",
        feature_policy,
        write=write,
    )

    def core_corpus(value: dict[str, object], old_hash: str) -> None:
        value["supersedes"] = "polymarket-round21-core-corpus-materialization-design-v2"
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )
        value["supersession"] = {
            "round21_core_corpus_materialization_design_v2_sha256": old_hash,
            "change": "bind_materialized_rows_to_causal_receipt_age_features",
            "condition_admission_changed": False,
            "storage_layout_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
        }

    hashes["core_corpus"] = _supersede(
        "round-021-core-corpus-materialization-design-v2.json",
        "round-021-core-corpus-materialization-design-v3.json",
        "polymarket-round21-core-corpus-materialization-design-v3",
        core_corpus,
        write=write,
    )

    def execution(value: dict[str, object], old_hash: str) -> None:
        value["supersedes"] = "polymarket-round21-executable-action-policy-v2"
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )
        value["supersession"] = {
            "round21_executable_action_policy_v2_sha256": old_hash,
            "change": "bind_unchanged_execution_matrix_to_receipt_age_features",
            "scenario_matrix_changed": False,
            "fees_or_latency_changed": False,
            "capture_data_used_for_change": False,
            "targets_used_for_change": False,
            "market_outcomes_used_for_change": False,
        }

    hashes["execution"] = _supersede(
        "round-021-executable-action-policy-v2.json",
        "round-021-executable-action-policy-v3.json",
        "polymarket-round21-executable-action-policy-v3",
        execution,
        write=write,
    )

    def model(value: dict[str, object], old_hash: str) -> None:
        _replace_parent(
            value, "round21_feature_policy_sha256", hashes["feature_policy"]
        )
        research = value["research_basis"]
        assert isinstance(research, list)
        research.extend(
            (
                {
                    "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4095405",
                    "use": "millisecond_data_timeliness_changes_high_frequency_predictability",
                },
                {
                    "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5907665",
                    "use": "differential_feed_latency_can_invert_event_order_and_create_lookahead",
                },
            )
        )
        value["supersession"] = {
            "round21_matched_model_design_v8_sha256": old_hash,
            "change": "add_target_free_causal_quote_receipt_age_features",
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
        "round-021-matched-model-design-v8.json",
        "round-021-matched-model-design-v9.json",
        "polymarket-round21-matched-model-design-v9",
        model,
        write=write,
    )

    def envelope(value: dict[str, object], _old_hash: str) -> None:
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])

    hashes["envelope"] = _supersede(
        "round-021-probability-envelope-design-v5.json",
        "round-021-probability-envelope-design-v6.json",
        "polymarket-round21-probability-envelope-design-v6",
        envelope,
        write=write,
    )

    def policy(value: dict[str, object], _old_hash: str) -> None:
        value["supersedes"] = "polymarket-round21-multi-action-policy-design-v7"
        _replace_parent(value, "round21_execution_policy_sha256", hashes["execution"])
        _replace_parent(
            value,
            "round21_probability_envelope_design_sha256",
            hashes["envelope"],
        )

    hashes["policy"] = _supersede(
        "round-021-multi-action-policy-design-v7.json",
        "round-021-multi-action-policy-design-v8.json",
        "polymarket-round21-multi-action-policy-design-v8",
        policy,
        write=write,
    )

    def replay(value: dict[str, object], _old_hash: str) -> None:
        _replace_parent(value, "round21_execution_policy_sha256", hashes["execution"])
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(
            value,
            "round21_probability_envelope_design_sha256",
            hashes["envelope"],
        )
        _replace_parent(value, "round21_multi_action_policy_sha256", hashes["policy"])

    hashes["replay"] = _supersede(
        "round-021-economic-replay-design-v5.json",
        "round-021-economic-replay-design-v6.json",
        "polymarket-round21-economic-replay-design-v6",
        replay,
        write=write,
    )

    def comparison(value: dict[str, object], _old_hash: str) -> None:
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )

    hashes["comparison"] = _supersede(
        "round-021-matched-economic-comparison-design-v5.json",
        "round-021-matched-economic-comparison-design-v6.json",
        "polymarket-round21-matched-economic-comparison-design-v6",
        comparison,
        write=write,
    )

    def ai_veto(value: dict[str, object], old_hash: str) -> None:
        _set_predecessor(
            value,
            prefix="round21_ai_veto_design_v",
            key="round21_ai_veto_design_v6_sha256",
            digest=old_hash,
        )
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        _replace_parent(value, "round21_multi_action_policy_sha256", hashes["policy"])
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_unchanged_finite_ai_program_to_receipt_age_features"
        )

    hashes["ai_veto"] = _supersede(
        "round-021-ai-veto-design-v6.json",
        "round-021-ai-veto-design-v7.json",
        "polymarket-round21-ai-veto-design-v7",
        ai_veto,
        write=write,
    )

    def ai_selection(value: dict[str, object], old_hash: str) -> None:
        _set_predecessor(
            value,
            prefix="round21_ai_candidate_selection_design_v",
            key="round21_ai_candidate_selection_design_v6_sha256",
            digest=old_hash,
        )
        _replace_parent(value, "round21_ai_veto_design_sha256", hashes["ai_veto"])
        _replace_parent(value, "round21_model_design_sha256", hashes["model"])
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_unchanged_finite_ai_selection_to_receipt_age_features"
        )

    hashes["ai_selection"] = _supersede(
        "round-021-ai-candidate-selection-design-v6.json",
        "round-021-ai-candidate-selection-design-v7.json",
        "polymarket-round21-ai-candidate-selection-design-v7",
        ai_selection,
        write=write,
    )

    def ai_schedule(value: dict[str, object], old_hash: str) -> None:
        _set_predecessor(
            value,
            prefix="round21_ai_historical_schedule_design_v",
            key="round21_ai_historical_schedule_design_v6_sha256",
            digest=old_hash,
        )
        _set_predecessor(
            value,
            prefix="round21_ai_veto_design_v",
            key="round21_ai_veto_design_v7_sha256",
            digest=hashes["ai_veto"],
        )
        _replace_parent(
            value, "round21_economic_replay_design_sha256", hashes["replay"]
        )
        supersession = value["supersession"]
        assert isinstance(supersession, dict)
        supersession["change"] = (
            "bind_unchanged_virtual_timing_to_round21_ai_veto_design_v7"
        )

    hashes["ai_schedule"] = _supersede(
        "round-021-ai-historical-schedule-design-v6.json",
        "round-021-ai-historical-schedule-design-v7.json",
        "polymarket-round21-ai-historical-schedule-design-v7",
        ai_schedule,
        write=write,
    )

    def sealed(value: dict[str, object], old_hash: str) -> None:
        _set_predecessor(
            value,
            prefix="round21_terminal_sealed_evaluation_design_v",
            key="round21_terminal_sealed_evaluation_design_v6_sha256",
            digest=old_hash,
        )
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
        supersession["change"] = "bind_sealed_evaluation_to_receipt_age_feature_chain"

    hashes["sealed"] = _supersede(
        "round-021-terminal-sealed-evaluation-design-v6.json",
        "round-021-terminal-sealed-evaluation-design-v7.json",
        "polymarket-round21-terminal-sealed-evaluation-design-v7",
        sealed,
        write=write,
    )
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(write=args.write), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
