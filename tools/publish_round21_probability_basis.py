"""Publish the target-free Round 21 probability-basis contract chain."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.storage import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"
MODEL_BASIS = "model.market_prior_minus_structural_log_odds"
MODEL_BASIS_SEMANTICS = (
    "market_prior_log_odds_minus_structural_log_odds_clipped_to_probability_floor"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _load(name: str) -> dict[str, Any]:
    value = json.loads((RESEARCH / name).read_text(encoding="utf-8"))
    claimed = value.pop("design_sha256")
    if claimed != _canonical_sha256(value):
        raise ValueError(f"Source design is not canonical: {name}")
    return {**value, "design_sha256": claimed}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    selected = deepcopy(value)
    selected.pop("design_sha256", None)
    selected["design_sha256"] = _canonical_sha256(selected)
    return selected


def build_contracts() -> dict[str, dict[str, Any]]:
    model_v5 = _load("round-021-matched-model-design-v5.json")
    envelope_v2 = _load("round-021-probability-envelope-design-v2.json")
    policy_v4 = _load("round-021-multi-action-policy-design-v4.json")
    replay_v2 = _load("round-021-economic-replay-design-v2.json")
    comparison_v2 = _load("round-021-matched-economic-comparison-design-v2.json")
    ai_veto_v3 = _load("round-021-ai-veto-design-v3.json")
    ai_history_v3 = _load("round-021-ai-historical-schedule-design-v3.json")
    ai_selection_v3 = _load("round-021-ai-candidate-selection-design-v3.json")
    sealed_v3 = _load("round-021-terminal-sealed-evaluation-design-v3.json")

    model = deepcopy(model_v5)
    model["schema_version"] = "polymarket-round21-matched-model-design-v6"
    model["supersession"] = {
        "round21_matched_model_design_v5_sha256": model_v5["design_sha256"],
        "change": "add_target_free_market_prior_minus_structural_log_odds_model_basis",
        "model_basis_features_added": [MODEL_BASIS],
        "candidate_count_changed": False,
        "predictive_model_families_changed": False,
        "training_schedule_changed": False,
        "capture_data_used_for_change": False,
        "targets_used_for_change": False,
        "market_outcomes_used_for_change": False,
        "edge_or_profitability_inferred_from_change": False,
    }
    model["preprocessing"]["model_basis"] = MODEL_BASIS_SEMANTICS
    model["candidate_program"]["model_basis_features"] = [MODEL_BASIS]
    model["research_basis"].extend(
        [
            {
                "url": "https://arxiv.org/abs/2510.15205",
                "use": "logit_space_short_horizon_prediction_market_probability_dynamics",
            },
            {
                "url": "https://arxiv.org/abs/2602.00776",
                "use": "crypto_microstructure_order_flow_spread_and_adverse_selection_evidence",
            },
            {
                "url": "https://arxiv.org/abs/2506.05764",
                "use": "feature_engineering_before_unbounded_deep_model_expansion",
            },
        ]
    )
    model = _seal(model)

    envelope = deepcopy(envelope_v2)
    envelope["schema_version"] = "polymarket-round21-probability-envelope-design-v3"
    envelope["supersedes"] = "polymarket-round21-probability-envelope-design-v2"
    envelope["parents"]["round21_model_design_sha256"] = model["design_sha256"]
    envelope = _seal(envelope)

    policy = deepcopy(policy_v4)
    policy["schema_version"] = "polymarket-round21-multi-action-policy-design-v5"
    policy["supersedes"] = "polymarket-round21-multi-action-policy-design-v4"
    policy["parents"]["round21_probability_envelope_design_sha256"] = envelope[
        "design_sha256"
    ]
    policy = _seal(policy)

    replay = deepcopy(replay_v2)
    replay["schema_version"] = "polymarket-round21-economic-replay-design-v3"
    replay["supersedes"] = "polymarket-round21-economic-replay-design-v2"
    replay["parents"].update(
        {
            "round21_model_design_sha256": model["design_sha256"],
            "round21_probability_envelope_design_sha256": envelope["design_sha256"],
            "round21_multi_action_policy_sha256": policy["design_sha256"],
        }
    )
    replay = _seal(replay)

    comparison = deepcopy(comparison_v2)
    comparison["schema_version"] = (
        "polymarket-round21-matched-economic-comparison-design-v3"
    )
    comparison["supersedes"] = (
        "polymarket-round21-matched-economic-comparison-design-v2"
    )
    comparison["parents"].update(
        {
            "round21_model_design_sha256": model["design_sha256"],
            "round21_economic_replay_design_sha256": replay["design_sha256"],
        }
    )
    comparison = _seal(comparison)

    ai_veto = deepcopy(ai_veto_v3)
    ai_veto["schema_version"] = "polymarket-round21-ai-veto-design-v4"
    ai_veto["parents"] = {
        "round21_ai_veto_design_v3_sha256": ai_veto_v3["design_sha256"],
        "round21_contract_sha256": ai_veto_v3["parents"]["round21_contract_sha256"],
        "round21_model_design_sha256": model["design_sha256"],
        "round21_multi_action_policy_sha256": policy["design_sha256"],
        "round21_economic_replay_design_sha256": replay["design_sha256"],
    }
    ai_veto["supersession"] = {
        "change": "bind_unchanged_finite_ai_program_to_round21_model_design_v6",
        "candidate_count_changed": False,
        "candidate_identities_changed": False,
        "capture_data_used_for_change": False,
        "targets_used_for_change": False,
        "market_outcomes_used_for_change": False,
        "edge_or_profitability_inferred_from_model_design_change": False,
    }
    ai_veto = _seal(ai_veto)

    ai_history = deepcopy(ai_history_v3)
    ai_history["schema_version"] = "polymarket-round21-ai-historical-schedule-design-v4"
    ai_history["parents"] = {
        "round21_ai_historical_schedule_design_v3_sha256": ai_history_v3[
            "design_sha256"
        ],
        "round21_ai_veto_design_v4_sha256": ai_veto["design_sha256"],
        "round21_economic_replay_design_sha256": replay["design_sha256"],
    }
    ai_history["supersession"]["change"] = (
        "bind_unchanged_virtual_timing_to_round21_ai_veto_design_v4"
    )
    ai_history = _seal(ai_history)

    ai_selection = deepcopy(ai_selection_v3)
    ai_selection["schema_version"] = (
        "polymarket-round21-ai-candidate-selection-design-v4"
    )
    ai_selection["parents"] = {
        "round21_ai_candidate_selection_design_v3_sha256": ai_selection_v3[
            "design_sha256"
        ],
        "round21_contract_sha256": ai_selection_v3["parents"][
            "round21_contract_sha256"
        ],
        "round21_ai_veto_design_sha256": ai_veto["design_sha256"],
        "round21_model_design_sha256": model["design_sha256"],
    }
    ai_selection["supersession"]["change"] = (
        "bind_unchanged_finite_ai_selection_to_round21_model_design_v6"
    )
    ai_selection = _seal(ai_selection)

    sealed = deepcopy(sealed_v3)
    sealed["schema_version"] = "polymarket-round21-terminal-sealed-evaluation-design-v4"
    sealed["parents"].update(
        {
            "round21_terminal_sealed_evaluation_design_v3_sha256": sealed_v3[
                "design_sha256"
            ],
            "round21_model_design_sha256": model["design_sha256"],
            "round21_multi_action_policy_sha256": policy["design_sha256"],
            "round21_economic_replay_design_sha256": replay["design_sha256"],
            "round21_matched_comparison_design_sha256": comparison["design_sha256"],
            "round21_ai_veto_design_sha256": ai_veto["design_sha256"],
            "round21_ai_historical_schedule_design_sha256": ai_history["design_sha256"],
            "round21_ai_selection_design_sha256": ai_selection["design_sha256"],
        }
    )
    sealed["supersession"]["change"] = (
        "bind_sealed_evaluation_to_target_blind_round21_model_design_v6_chain"
    )
    sealed = _seal(sealed)

    return {
        "round-021-matched-model-design-v6.json": model,
        "round-021-probability-envelope-design-v3.json": envelope,
        "round-021-multi-action-policy-design-v5.json": policy,
        "round-021-economic-replay-design-v3.json": replay,
        "round-021-matched-economic-comparison-design-v3.json": comparison,
        "round-021-ai-veto-design-v4.json": ai_veto,
        "round-021-ai-historical-schedule-design-v4.json": ai_history,
        "round-021-ai-candidate-selection-design-v4.json": ai_selection,
        "round-021-terminal-sealed-evaluation-design-v4.json": sealed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    contracts = build_contracts()
    for name, value in contracts.items():
        path = RESEARCH / name
        expected = json.dumps(value, indent=2, sort_keys=False) + "\n"
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                return 1
        else:
            write_json_atomic(path, value, indent=2, sort_keys=False)
            print(f"{name} {value['design_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
