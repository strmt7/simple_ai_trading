"""Run precommitted conditional outcome-mixture research screens."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.compute import resolve_backend  # noqa: E402
from simple_ai_trading.microstructure_action_architecture import (  # noqa: E402
    ActionValueEnsembleBatch,
    ensemble_action_value_predictions,
)
from simple_ai_trading.microstructure_action_policy import (  # noqa: E402
    ACTION_POLICY_SCHEMA_VERSION,
    barrier_trace_gate_reasons,
    derive_action_scores,
    select_barrier_threshold,
    simulate_barrier_action_trace,
)
from simple_ai_trading.microstructure_architecture import (  # noqa: E402
    average_label_uniqueness,
    causal_cusum_event_mask,
)
from simple_ai_trading.microstructure_barriers import (  # noqa: E402
    ADAPTIVE_BARRIER_SCHEMA_VERSION,
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    build_adaptive_barrier_targets,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    MICROSTRUCTURE_FEATURE_VERSION,
)
from simple_ai_trading.microstructure_outcome_mixture import (  # noqa: E402
    OUTCOME_MIXTURE_SCHEMA_VERSION,
    OutcomeMixtureArchitectureSpec,
    TrainedOutcomeMixtureModel,
    load_outcome_mixture_model,
    predict_outcome_mixture_model,
    save_outcome_mixture_model,
    train_outcome_mixture_model,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402

try:  # noqa: E402
    from tools.run_adaptive_action_screen import (
        REPORT_SCHEMA_VERSION,
        _day_id,
        _empty_profile_trace,
        _forecast_diagnostics,
        _iso_days,
        _profile_spec,
        _role_indexes,
        _targets_sha256,
        load_adaptive_action_design,
    )
    from tools.run_gross_architecture_screen import (
        _artifact_summary,
        _canonical_sha256,
        _is_sha256,
        _sha256_file,
    )
    from tools.run_head_coherence_screen import _load_corpus
except ModuleNotFoundError:  # pragma: no cover - direct tools directory execution
    from run_adaptive_action_screen import (
        REPORT_SCHEMA_VERSION,
        _day_id,
        _empty_profile_trace,
        _forecast_diagnostics,
        _iso_days,
        _profile_spec,
        _role_indexes,
        _targets_sha256,
        load_adaptive_action_design,
    )
    from run_gross_architecture_screen import (
        _artifact_summary,
        _canonical_sha256,
        _is_sha256,
        _sha256_file,
    )
    from run_head_coherence_screen import _load_corpus


DESIGN_SCHEMA_VERSION = "outcome-mixture-screen-design-v1"
_ROUND16_DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-016-adaptive-action-design.json"
)
_SHARED_SECTIONS = (
    "data",
    "execution",
    "barrier_targets",
    "runtime_resources",
    "event_sampler",
    "threshold_policy",
    "risk_profiles",
    "evaluation",
    "reserved_terminal",
)
_ROUND_CONTRACTS = {
    17: {
        "purpose": "consumed_data_conditional_outcome_mixture_screen",
        "design_revisions": {1, 2},
        "ranking_loss_weight": 0.0,
        "feature_version": "l1-tape-causal-v7",
        "predecessor": {
            "round": 16,
            "design_sha256": (
                "15e8702999f7ed2c5acdd5ab27c19535ad87c68e476b92339b7335d98991a639"
            ),
            "source_report_canonical_sha256": (
                "87e26c4e3809097d23de14e13b443ebe5cdefae3ba4beadba6b04b4c19f39229"
            ),
            "publication_sha256": (
                "5c84aad13ec100882c132a92bbed838dda9caf3488760a374d06d2d87b301c89"
            ),
            "finding": (
                "Round 16 ranked direction better than chance in parts of the "
                "consumed window, but every inspected realized top tail remained "
                "negative after exact costs; Round 17 therefore models win "
                "probability and conditional win/loss magnitude separately before "
                "deriving expected action value."
            ),
        },
    },
    18: {
        "purpose": "consumed_data_rank_regularized_outcome_mixture_screen",
        "design_revisions": {1},
        "ranking_loss_weight": 0.1,
        "feature_version": "l1-tape-causal-v7",
        "predecessor": {
            "round": 17,
            "design_sha256": (
                "963ecc6d9fa384969992bed36addff0cfceb3e057fbe43a91725e15d037db1ee"
            ),
            "source_report_canonical_sha256": (
                "77efd2c857e4f3cf9ea7061d7fe17d16b7f98308a6ea7a896be7d3213529ec6c"
            ),
            "publication_sha256": (
                "0b77c412fa351728fbc62005e3f6a0beea1f22ead25918eb3ac111fcbe718c24"
            ),
            "finding": (
                "Round 17 reduced point error but produced mostly worse probability "
                "calibration than prevalence and negative realized top tails. Round "
                "18 isolates a 0.10 continuous-value ranking regularizer while "
                "leaving every data, execution, threshold, and risk contract fixed."
            ),
        },
    },
    19: {
        "purposes": {
            1: "".join(("consumed_data_pressure_", "capacity_outcome_mixture_screen")),
            2: "consumed_data_depth_normalized_order_flow_outcome_mixture_screen",
        },
        "design_revisions": {1, 2},
        "ranking_loss_weight": 0.1,
        "feature_version": "l1-tape-causal-v8",
        "predecessors": {
            1: {
                "round": 18,
                "design_sha256": (
                    "024b1146d3330e9306470dd29a3ec7c49c686e0fb66ad9c20c7be2d02afb5c40"
                ),
                "source_report_canonical_sha256": (
                    "4a8ac77e436e52fc0fa81ef06d131bc9abe18d857fe8e32174ae41daabdec676"
                ),
                "publication_sha256": (
                    "1086ae098eb77679023c36dd3b42355aef52f6daa8de720b41c718ecaa00d378"
                ),
                "finding": (
                    "Round 18 improved the least-negative policy tail and produced "
                    "24 aggressive calibration-eligible rows, but all four threshold "
                    "traces lost money after stress costs. Round 19 keeps the model, "
                    "data, risk, and ranking settings fixed while adding causal "
                    "trade-pressure versus opposing displayed-depth inputs."
                ),
            },
            2: {
                "round": 18,
                "design_sha256": (
                    "024b1146d3330e9306470dd29a3ec7c49c686e0fb66ad9c20c7be2d02afb5c40"
                ),
                "source_report_canonical_sha256": (
                    "4a8ac77e436e52fc0fa81ef06d131bc9abe18d857fe8e32174ae41daabdec676"
                ),
                "publication_sha256": (
                    "1086ae098eb77679023c36dd3b42355aef52f6daa8de720b41c718ecaa00d378"
                ),
                "finding": (
                    "Round 18 improved the least-negative out-of-sample top-100 mean "
                    "net return and produced 24 aggressive-profile signals passing "
                    "pre-threshold controls in the threshold-selection window, but "
                    "all four threshold-selection simulations lost money after "
                    "stress costs. Round 19 keeps the model, data, risk, and ranking "
                    "settings fixed while adding causal depth-normalized aggressive "
                    "order-flow inputs based on current opposing displayed L1 depth."
                ),
            },
        },
    },
    20: {
        "purpose": "consumed_data_parameter_matched_independent_side_tower_screen",
        "design_revisions": {1},
        "ranking_loss_weight": 0.1,
        "feature_version": "l1-tape-causal-v8",
        "side_tower_mode": "independent",
        "hidden_dim": 88,
        "residual_blocks": 2,
        "trainable_parameter_count": 145_914,
        "predecessor": {
            "round": 19,
            "design_sha256": (
                "2a2c2e1c52d7dd0a6c8ac1a34e26defe3ec436a5051fcc49ed2172ef9f87ca77"
            ),
            "source_report_canonical_sha256": (
                "bd1075b8155208c0d9f3dc71d42aa43a98381628e00fa41d8eb5b33e0eee4d05"
            ),
            "publication_sha256": (
                "2b72894744be750357c5913ffe2b71787c3f70e595e41e08753f0a93bfc61c86"
            ),
            "finding": (
                "Round 19 increased aggressive-profile signal eligibility to 55 "
                "threshold-selection and 83 out-of-sample rows, but all four "
                "threshold-selection simulations lost money under stress and every "
                "highest-ranked realized mean remained negative net of costs. Round "
                "20 isolates parameter-matched independent long/short towers while "
                "keeping the v8 features, targets, losses, data, execution, and risk "
                "controls fixed."
            ),
        },
    },
    21: {
        "purposes": {
            1: "consumed_data_pairwise_net_return_ranking_screen",
            2: "consumed_data_pairwise_net_return_ranking_gpu_native_screen",
            3: (
                "consumed_data_pairwise_net_return_ranking_reproducible_artifact_screen"
            ),
        },
        "design_revisions": {1, 2, 3},
        "ranking_loss_weight": 0.1,
        "ranking_loss_mode": "pairwise_net_return",
        "feature_version": "l1-tape-causal-v8",
        "side_tower_mode": "independent",
        "hidden_dim": 88,
        "residual_blocks": 2,
        "trainable_parameter_count": 145_914,
        "predecessors": {
            1: {
                "round": 20,
                "design_sha256": (
                    "a6f4e82d82474d673c8495f9775f9d974b95a9cc2a8d497f7f45bce29ad965bb"
                ),
                "source_report_canonical_sha256": (
                    "f7b4aeb6c4d52b49bce53468eeb13f03e0c2441d426ec0c3de8338e96f0e5885"
                ),
                "publication_sha256": (
                    "3e8a22398871f80020743ee9987a670cfbf50292e351fa018f513c3c535c2033"
                ),
                "finding": (
                    "Round 20 increased the largest threshold-selection eligible set "
                    "from 55 to 147 rows, but all eight threshold candidates lost "
                    "money under stress and the least-negative out-of-sample top-100 "
                    "mean net return was -6.733319 bps. Round 21 keeps the independent "
                    "long/short architecture, parameter budget, data, targets, costs, "
                    "and risk controls fixed while replacing the global Pearson-"
                    "correlation ranking surrogate with sampled pairwise net-return "
                    "ranking."
                ),
            },
            2: {
                "round": 20,
                "design_sha256": (
                    "a6f4e82d82474d673c8495f9775f9d974b95a9cc2a8d497f7f45bce29ad965bb"
                ),
                "source_report_canonical_sha256": (
                    "f7b4aeb6c4d52b49bce53468eeb13f03e0c2441d426ec0c3de8338e96f0e5885"
                ),
                "publication_sha256": (
                    "3e8a22398871f80020743ee9987a670cfbf50292e351fa018f513c3c535c2033"
                ),
                "finding": (
                    "Round 21 revision 1 canonical report "
                    "2fcad6c998d169c4bcaa540be0b0edab7e11c325e9a29b4f7ee15f1ea2500a99 "
                    "is excluded because DirectML reported an aten::roll CPU "
                    "fallback, violating the GPU-native experiment contract. "
                    "Revision 2 preserves every economic and model setting while "
                    "using DirectML-supported concatenation for cyclic pair sampling "
                    "and making CPU-fallback warnings test failures."
                ),
            },
            3: {
                "round": 20,
                "design_sha256": (
                    "a6f4e82d82474d673c8495f9775f9d974b95a9cc2a8d497f7f45bce29ad965bb"
                ),
                "source_report_canonical_sha256": (
                    "f7b4aeb6c4d52b49bce53468eeb13f03e0c2441d426ec0c3de8338e96f0e5885"
                ),
                "publication_sha256": (
                    "3e8a22398871f80020743ee9987a670cfbf50292e351fa018f513c3c535c2033"
                ),
                "finding": (
                    "Round 21 revision 2 canonical report "
                    "9a56fcdd94b4a5e647611c55c711b44d99700316566611cd0a0937598d184aec "
                    "is economically valid and exactly matches revision 1 metrics, "
                    "but identical model content serialized with nonstable "
                    "Safetensors metadata order. Revision 3 preserves every model, "
                    "economic, GPU, and risk setting while canonicalizing artifact "
                    "headers before atomic replacement."
                ),
            },
        },
    },
    22: {
        "purpose": "consumed_data_additive_pairwise_net_return_ranking_screen",
        "design_revisions": {1},
        "ranking_loss_weight": 0.1,
        "ranking_loss_mode": "correlation",
        "pairwise_ranking_loss_weight": 0.02,
        "feature_version": "l1-tape-causal-v8",
        "side_tower_mode": "independent",
        "hidden_dim": 88,
        "residual_blocks": 2,
        "trainable_parameter_count": 145_914,
        "predecessor": {
            "round": 21,
            "design_sha256": (
                "afcebb4d1d079bb91755bb14da4ed8684af141bf829941443509b803bbe4b9eb"
            ),
            "source_report_canonical_sha256": (
                "33475d88c105a8a7b1d85b18d30aef4a4ee71c11da1cc68edebd2f52bc0b1ee5"
            ),
            "publication_sha256": (
                "c497c916945f89bd61bf4edf489b66684c67e22cf34882226914b943c5393afd"
            ),
            "finding": (
                "Round 21 improved several discrimination and short-side tail "
                "diagnostics, but every threshold-selection expected-return strength "
                "was non-positive and all three profiles had zero eligible rows. "
                "Round 22 restores Round 20's 0.10 correlation regularizer and adds "
                "a 0.02 pairwise net-return term while keeping architecture, data, "
                "targets, costs, thresholds, and risk controls fixed."
            ),
        },
    },
}


def _git_bytes(*arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("outcome-mixture Git implementation binding failed") from exc


def _validate_git_blob_binding(binding: Mapping[str, object]) -> None:
    commit = str(binding.get("commit") or "").lower()
    files = binding.get("files")
    if (
        binding.get("hash_mode") != "git_blob_sha256_v1"
        or len(commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(files, list)
        or not files
    ):
        raise ValueError("outcome-mixture implementation binding is incomplete")
    _git_bytes("merge-base", "--is-ancestor", commit, "HEAD")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping) or not _is_sha256(item.get("sha256")):
            raise ValueError("outcome-mixture implementation file is invalid")
        relative = Path(str(item.get("path") or ""))
        normalized = relative.as_posix()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not normalized
            or normalized in seen
            or not (ROOT / relative).is_file()
        ):
            raise ValueError("outcome-mixture implementation path is unsafe")
        seen.add(normalized)
        expected = str(item["sha256"])
        bound_blob = _git_bytes("show", f"{commit}:{normalized}")
        head_blob = _git_bytes("show", f"HEAD:{normalized}")
        if (
            hashlib.sha256(bound_blob).hexdigest() != expected
            or hashlib.sha256(head_blob).hexdigest() != expected
        ):
            raise ValueError(f"outcome-mixture implementation changed: {normalized}")
        try:
            subprocess.run(
                ["git", "-C", str(ROOT), "diff", "--quiet", "--", normalized],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "diff",
                    "--cached",
                    "--quiet",
                    "--",
                    normalized,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(
                f"outcome-mixture implementation worktree changed: {normalized}"
            ) from exc


def load_outcome_mixture_design(
    path: str | Path,
    *,
    require_current: bool = True,
) -> tuple[dict[str, object], str]:
    """Load a hash-bound design whose shared controls equal sealed Round 16."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("outcome-mixture design is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("outcome-mixture design must be an object")
    claimed = payload.get("design_sha256")
    canonical = dict(payload)
    canonical.pop("design_sha256", None)
    if not _is_sha256(claimed) or claimed != _canonical_sha256(canonical):
        raise ValueError("outcome-mixture design hash is invalid")
    round_number = payload.get("round")
    round_contract = (
        _ROUND_CONTRACTS.get(round_number)
        if isinstance(round_number, int) and not isinstance(round_number, bool)
        else None
    )
    purpose = round_contract.get("purpose") if round_contract is not None else None
    purposes = round_contract.get("purposes") if round_contract is not None else None
    if isinstance(purposes, Mapping):
        purpose = purposes.get(payload.get("design_revision"))
    predecessor = (
        round_contract.get("predecessor") if round_contract is not None else None
    )
    predecessors = (
        round_contract.get("predecessors") if round_contract is not None else None
    )
    if isinstance(predecessors, Mapping):
        predecessor = predecessors.get(payload.get("design_revision"))
    reference, _reference_sha256 = load_adaptive_action_design(
        _ROUND16_DESIGN, require_current=False
    )
    if set(payload) != set(reference) or any(
        payload.get(name) != reference.get(name) for name in _SHARED_SECTIONS
    ):
        raise ValueError("outcome-mixture shared safety contract drifted")
    if (
        payload.get("schema_version") != DESIGN_SCHEMA_VERSION
        or round_contract is None
        or payload.get("design_revision") not in round_contract["design_revisions"]
        or payload.get("purpose") != purpose
        or payload.get("target_mode") != ADAPTIVE_BARRIER_TARGET_MODE
        or payload.get("trading_authority") is not False
        or payload.get("execution_claim") is not False
        or payload.get("profitability_claim") is not False
        or payload.get("portfolio_claim") is not False
        or payload.get("leverage_applied") is not False
        or payload.get("predecessor_evidence") != predecessor
        or payload.get("training") != reference.get("training")
    ):
        raise ValueError("outcome-mixture design contract is invalid")
    implementation = payload.get("implementation")
    model = payload.get("model")
    research_basis = payload.get("research_basis")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(research_basis, list)
        or len(research_basis) < 6
    ):
        raise ValueError("outcome-mixture design sections are incomplete")
    if require_current:
        _validate_git_blob_binding(implementation)
    model_spec = OutcomeMixtureArchitectureSpec(**dict(model))
    if (
        model_spec.family != "conditional_outcome_mixture_residual_mlp"
        or model_spec.ranking_loss_weight != round_contract["ranking_loss_weight"]
        or model_spec.ranking_loss_mode
        != round_contract.get("ranking_loss_mode", "correlation")
        or model_spec.pairwise_ranking_loss_weight
        != round_contract.get("pairwise_ranking_loss_weight", 0.0)
        or model_spec.side_tower_mode != round_contract.get("side_tower_mode", "shared")
        or model_spec.hidden_dim != round_contract.get("hidden_dim", 128)
        or model_spec.residual_blocks != round_contract.get("residual_blocks", 2)
        or (
            require_current
            and MICROSTRUCTURE_FEATURE_VERSION != round_contract["feature_version"]
        )
    ):
        raise ValueError("outcome-mixture model family is invalid")
    urls: set[str] = set()
    for item in research_basis:
        if (
            not isinstance(item, Mapping)
            or not str(item.get("title") or "").strip()
            or not str(item.get("use") or "").strip()
            or not str(item.get("url") or "").startswith("https://")
            or str(item["url"]) in urls
        ):
            raise ValueError("outcome-mixture research basis is invalid")
        urls.add(str(item["url"]))
    return payload, str(claimed)


def _ensemble_for_role(
    models: list[TrainedOutcomeMixtureModel],
    dataset,
    endpoints: np.ndarray,
    *,
    compute_backend: str,
    batch_size: int,
) -> ActionValueEnsembleBatch:
    return ensemble_action_value_predictions(
        [
            predict_outcome_mixture_model(
                model,
                dataset,
                endpoints,
                compute_backend=compute_backend,
                batch_size=batch_size,
            )
            for model in models
        ]
    )


def _evaluate_profiles(
    *,
    design: Mapping[str, object],
    dataset,
    targets,
    calibration_prediction: ActionValueEnsembleBatch,
    policy_prediction: ActionValueEnsembleBatch,
    progress,
) -> tuple[list[dict[str, object]], list[str]]:
    threshold_policy = design["threshold_policy"]
    data = design["data"]
    assert isinstance(threshold_policy, Mapping)
    assert isinstance(data, Mapping)
    roles_raw = data["roles"]
    assert isinstance(roles_raw, Mapping)
    calibration_days = _iso_days(roles_raw["calibration"])
    policy_days = _iso_days(roles_raw["policy"])
    profile_results: list[dict[str, object]] = []
    policy_survivors: list[str] = []
    for raw_profile in design["risk_profiles"]:
        assert isinstance(raw_profile, Mapping)
        spec = _profile_spec(raw_profile)
        calibration_score = derive_action_scores(calibration_prediction, spec)
        policy_score = derive_action_scores(policy_prediction, spec)
        selection = select_barrier_threshold(
            dataset,
            targets,
            calibration_score,
            quantiles=tuple(float(value) for value in threshold_policy["quantiles"]),
            expected_days=calibration_days,
            gates=raw_profile["calibration_gates"],
            drawdown_penalty=float(threshold_policy["drawdown_penalty"]),
        )
        if selection.accepted:
            assert selection.threshold_bps is not None
            policy_base = simulate_barrier_action_trace(
                dataset,
                targets,
                policy_score,
                scenario="base",
                strength_threshold_bps=selection.threshold_bps,
            )
            policy_stress = simulate_barrier_action_trace(
                dataset,
                targets,
                policy_score,
                scenario="stress",
                strength_threshold_bps=selection.threshold_bps,
            )
            policy_reasons = barrier_trace_gate_reasons(
                policy_stress,
                expected_days=policy_days,
                gates=raw_profile["policy_gates"],
            )
        else:
            policy_base = _empty_profile_trace(
                dataset, targets, policy_score, scenario="base"
            )
            policy_stress = _empty_profile_trace(
                dataset, targets, policy_score, scenario="stress"
            )
            policy_reasons = ["calibration_threshold_rejected"]
        policy_passed = not policy_reasons
        if policy_passed:
            policy_survivors.append(spec.profile)
        profile_results.append(
            {
                "profile": spec.profile,
                "policy_spec": asdict(spec),
                "calibration_eligible_rows": int(np.sum(calibration_score.eligible)),
                "threshold_selection": selection.asdict(),
                "policy_eligible_rows": int(np.sum(policy_score.eligible)),
                "policy_base_trace": policy_base.asdict(),
                "policy_stress_trace": policy_stress.asdict(),
                "policy_status": "research_candidate" if policy_passed else "rejected",
                "policy_rejection_reasons": policy_reasons,
                "development_evaluated": False,
                "development_result": None,
                "trading_authority": False,
                "execution_claim": False,
                "profitability_claim": False,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
        progress(
            "profile-policy-complete",
            profile=spec.profile,
            threshold_accepted=selection.accepted,
            policy_passed=policy_passed,
            stress_trades=policy_stress.metrics.trades,
            stress_total_net_bps=round(policy_stress.metrics.total_net_bps, 6),
        )
    return profile_results, policy_survivors


def _evaluate_development(
    *,
    design: Mapping[str, object],
    dataset,
    targets,
    development_prediction: ActionValueEnsembleBatch,
    profile_results: list[dict[str, object]],
    policy_survivors: list[str],
    progress,
) -> list[str]:
    data = design["data"]
    assert isinstance(data, Mapping)
    roles_raw = data["roles"]
    assert isinstance(roles_raw, Mapping)
    development_days = _iso_days(roles_raw["development_evaluation"])
    by_profile = {str(value["profile"]): value for value in profile_results}
    profiles_by_name = {
        str(value["profile"]): value for value in design["risk_profiles"]
    }
    for profile in policy_survivors:
        raw_profile = profiles_by_name[profile]
        assert isinstance(raw_profile, Mapping)
        result = by_profile[profile]
        selection = result["threshold_selection"]
        assert isinstance(selection, Mapping)
        threshold = float(selection["threshold_bps"])
        development_score = derive_action_scores(
            development_prediction, _profile_spec(raw_profile)
        )
        development_base = simulate_barrier_action_trace(
            dataset,
            targets,
            development_score,
            scenario="base",
            strength_threshold_bps=threshold,
        )
        development_stress = simulate_barrier_action_trace(
            dataset,
            targets,
            development_score,
            scenario="stress",
            strength_threshold_bps=threshold,
        )
        reasons = barrier_trace_gate_reasons(
            development_stress,
            expected_days=development_days,
            gates=raw_profile["development_gates"],
        )
        result["development_evaluated"] = True
        result["development_result"] = {
            "eligible_rows": int(np.sum(development_score.eligible)),
            "base_trace": development_base.asdict(),
            "stress_trace": development_stress.asdict(),
            "status": "research_candidate" if not reasons else "rejected",
            "rejection_reasons": reasons,
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "portfolio_claim": False,
            "leverage_applied": False,
        }
        progress(
            "profile-development-complete",
            profile=profile,
            passed=not reasons,
            stress_trades=development_stress.metrics.trades,
            stress_total_net_bps=round(development_stress.metrics.total_net_bps, 6),
        )
    return [
        str(value["profile"])
        for value in profile_results
        if isinstance(value.get("development_result"), Mapping)
        and value["development_result"].get("status") == "research_candidate"
    ]


def run_outcome_mixture_screen(
    *,
    design_path: str | Path,
    warehouse_path: str | Path,
    cache_root: str | Path,
    output_dir: str | Path,
    memory_limit: str | None = None,
    threads: int | None = None,
    compute_backend: str | None = None,
) -> dict[str, object]:
    design, design_sha256 = load_outcome_mixture_design(design_path)
    resources = design["runtime_resources"]
    data = design["data"]
    execution = design["execution"]
    sampler = design["event_sampler"]
    training = design["training"]
    terminal = design["reserved_terminal"]
    assert isinstance(resources, Mapping)
    assert isinstance(data, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(sampler, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(terminal, Mapping)
    round_contract = _ROUND_CONTRACTS[int(design["round"])]
    effective_memory = str(memory_limit or resources["duckdb_memory_limit"]).upper()
    effective_threads = int(threads or resources["warehouse_threads"])
    effective_backend = str(compute_backend or resources["compute_backend"]).lower()
    if (
        effective_memory != resources["duckdb_memory_limit"]
        or effective_threads != int(resources["warehouse_threads"])
        or effective_backend != resources["compute_backend"]
    ):
        raise ValueError("runtime overrides differ from the precommitted contract")
    backend = resolve_backend(effective_backend)
    if backend.kind != effective_backend:
        raise RuntimeError(
            "precommitted accelerator is unavailable; CPU fallback is forbidden"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "status.json"
    runtime = {
        "duckdb_memory_limit": effective_memory,
        "warehouse_threads": effective_threads,
        "compute_backend_requested": effective_backend,
        "compute_backend_kind": backend.kind,
        "compute_backend_device": backend.device,
        "compute_backend_vendor": backend.vendor,
        "training_worker_isolation": "clean_process",
        "cpu_fallback_permitted": False,
        "spill_directory_policy": "warehouse_adjacent",
    }

    def progress(phase: str, **extra: object) -> None:
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "design_sha256": design_sha256,
            "phase": phase,
            "runtime_resources": runtime,
            **extra,
        }
        print(
            "outcome-mixture "
            + " ".join(
                f"{name}={value}"
                for name, value in payload.items()
                if name != "runtime_resources"
            ),
            flush=True,
        )
        write_json_atomic(status_path, payload, indent=2, sort_keys=True)

    progress("initialize")
    corpus = _load_corpus(
        design=design,
        warehouse_path=warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory,
        threads=effective_threads,
        progress=progress,
    )
    dataset = corpus["dataset"]
    event_mask = causal_cusum_event_mask(
        dataset,
        volatility_multiplier=float(sampler["volatility_multiplier"]),
        minimum_threshold_bps=float(sampler["minimum_threshold_bps"]),
    )
    event_indexes = np.flatnonzero(event_mask).astype(np.int64)
    barrier_spec = AdaptiveBarrierSpec(**dict(design["barrier_targets"]))
    progress("barrier-target-build-start", event_rows=len(event_indexes))
    with MicrostructureWarehouse(
        warehouse_path,
        cache_root=cache_root,
        memory_limit=effective_memory,
        threads=effective_threads,
    ) as warehouse:
        targets = build_adaptive_barrier_targets(
            warehouse,
            dataset,
            event_indexes,
            spec=barrier_spec,
            progress=lambda day, total, valid: progress(
                "barrier-target-day",
                day=day,
                days=total,
                valid_rows=valid,
            ),
        )
    targets_sha256 = _targets_sha256(targets)
    roles, role_evidence = _role_indexes(
        dataset,
        targets,
        event_mask,
        data["roles"],
        _day_id(terminal["date"], label="terminal"),
    )
    progress(
        "dataset-ready",
        dataset_rows=dataset.rows,
        event_rows=len(event_indexes),
        valid_target_rows=targets.valid_rows,
        barrier_targets_sha256=targets_sha256,
        cache_state=corpus["cache_state"],
    )
    valid_positions = np.flatnonzero(targets.valid)
    valid_source = targets.source_indexes[valid_positions]
    max_base_exit = np.maximum(
        targets.base_long_exit_time_ms[valid_positions],
        targets.base_short_exit_time_ms[valid_positions],
    )
    exit_full = np.full(dataset.rows, -1, dtype=np.int64)
    exit_full[valid_source] = max_base_exit
    train_weights = average_label_uniqueness(
        dataset.decision_time_ms, exit_full, roles["train"]
    )
    tuning_weights = average_label_uniqueness(
        dataset.decision_time_ms, exit_full, roles["early_stop"]
    )
    model_spec = OutcomeMixtureArchitectureSpec(**dict(design["model"]))
    models: list[TrainedOutcomeMixtureModel] = []
    artifacts: list[dict[str, object]] = []
    seeds = tuple(int(value) for value in training["ensemble_seeds"])
    for member, seed in enumerate(seeds, start=1):
        progress("model-start", member=member, members=len(seeds), seed=seed)
        member_spec = replace(
            model_spec, candidate_id=f"{model_spec.candidate_id}-seed-{seed}"
        )
        model = train_outcome_mixture_model(
            dataset,
            targets,
            train_endpoints=roles["train"],
            tuning_endpoints=roles["early_stop"],
            spec=member_spec,
            target_scenario=str(training["target_scenario"]),
            compute_backend=effective_backend,
            seed=seed,
            batch_size=int(training["batch_size"]),
            max_epochs=int(training["max_epochs"]),
            patience=int(training["patience"]),
            train_sample_weights=train_weights,
            tuning_sample_weights=tuning_weights,
            progress=lambda epoch, total, training_loss, tuning_loss, index=member, model_seed=seed: (
                progress(
                    "model-epoch",
                    member=index,
                    seed=model_seed,
                    epoch=epoch,
                    epochs=total,
                    training_loss=round(training_loss, 8),
                    tuning_loss=round(tuning_loss, 8),
                )
            ),
        )
        if model.backend_kind != effective_backend:
            raise RuntimeError(
                "model training did not remain on the precommitted backend"
            )
        artifact_path = destination / "models" / f"seed-{seed}.safetensors"
        save_outcome_mixture_model(artifact_path, model)
        reloaded = load_outcome_mixture_model(artifact_path)
        if reloaded.model_sha256 != model.model_sha256:
            raise ValueError("saved model identity differs after reload")
        artifact = {
            "path": artifact_path.relative_to(destination).as_posix(),
            "sha256": _sha256_file(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "reload_verified": True,
        }
        summary = _artifact_summary(reloaded)
        trainable_parameters = int(
            sum(np.asarray(values).size for values in reloaded.state.values())
        )
        expected_parameters = round_contract.get("trainable_parameter_count")
        if expected_parameters is not None and trainable_parameters != int(
            expected_parameters
        ):
            raise ValueError("outcome-mixture trainable parameter count drifted")
        summary.update(
            {
                "target_schema_version": reloaded.target_schema_version,
                "target_scenario": reloaded.target_scenario,
                "target_contract_sha256": reloaded.target_contract_sha256,
                "target_scale_bps": reloaded.target_scale_bps,
                "positive_class_prevalence": list(reloaded.positive_class_prevalence),
                "trainable_parameter_count": trainable_parameters,
                "portfolio_claim": False,
                "leverage_applied": False,
            }
        )
        artifacts.append({"seed": seed, "model": summary, "artifact": artifact})
        models.append(reloaded)
        progress(
            "model-complete",
            member=member,
            seed=seed,
            best_epoch=reloaded.best_epoch,
            tuning_loss=round(reloaded.tuning_loss, 8),
            model_sha256=reloaded.model_sha256,
            reload_verified=True,
        )
    batch_size = int(training["batch_size"])
    progress("calibration-predict")
    calibration_prediction = _ensemble_for_role(
        models,
        dataset,
        roles["calibration"],
        compute_backend=effective_backend,
        batch_size=batch_size,
    )
    progress("policy-predict")
    policy_prediction = _ensemble_for_role(
        models,
        dataset,
        roles["policy"],
        compute_backend=effective_backend,
        batch_size=batch_size,
    )
    profile_results, policy_survivors = _evaluate_profiles(
        design=design,
        dataset=dataset,
        targets=targets,
        calibration_prediction=calibration_prediction,
        policy_prediction=policy_prediction,
        progress=progress,
    )
    development_prediction = None
    if policy_survivors:
        progress("development-predict", profiles=",".join(policy_survivors))
        development_prediction = _ensemble_for_role(
            models,
            dataset,
            roles["development_evaluation"],
            compute_backend=effective_backend,
            batch_size=batch_size,
        )
        final_profiles = _evaluate_development(
            design=design,
            dataset=dataset,
            targets=targets,
            development_prediction=development_prediction,
            profile_results=profile_results,
            policy_survivors=policy_survivors,
            progress=progress,
        )
    else:
        final_profiles = []
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_class": "consumed_data_conditional_outcome_mixture_evidence",
        "status": "research_candidate" if final_profiles else "rejected",
        "round": int(design["round"]),
        "design_sha256": design_sha256,
        "action_value_model_schema_version": OUTCOME_MIXTURE_SCHEMA_VERSION,
        "action_policy_schema_version": ACTION_POLICY_SCHEMA_VERSION,
        "barrier_schema_version": ADAPTIVE_BARRIER_SCHEMA_VERSION,
        "target_mode": ADAPTIVE_BARRIER_TARGET_MODE,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": development_prediction is not None,
        "runtime_resources": runtime,
        "corpus_certificate_sha256": corpus["certificate"]["certificate_sha256"],
        "dataset": {
            "rows": dataset.rows,
            "event_rows": len(event_indexes),
            "valid_barrier_rows": targets.valid_rows,
            "cache_key": corpus["cache_key"],
            "cache_state": corpus["cache_state"],
            "source_manifest_fingerprint": corpus["source_evidence"][
                "manifest_fingerprint"
            ],
            "barrier_targets_sha256": targets_sha256,
            "barrier_summary": targets.summary(),
            "roles": role_evidence,
        },
        "ensemble_models": artifacts,
        "forecast_diagnostics": {
            "calibration_base": _forecast_diagnostics(
                targets, calibration_prediction, scenario="base"
            ),
            "calibration_stress": _forecast_diagnostics(
                targets, calibration_prediction, scenario="stress"
            ),
            "policy_base": _forecast_diagnostics(
                targets, policy_prediction, scenario="base"
            ),
            "policy_stress": _forecast_diagnostics(
                targets, policy_prediction, scenario="stress"
            ),
            "development_base": (
                _forecast_diagnostics(targets, development_prediction, scenario="base")
                if development_prediction is not None
                else None
            ),
            "development_stress": (
                _forecast_diagnostics(
                    targets, development_prediction, scenario="stress"
                )
                if development_prediction is not None
                else None
            ),
        },
        "profile_results": profile_results,
        "policy_survivors": policy_survivors,
        "final_profiles": final_profiles,
        "limitations": [
            "the certified exact-BBO corpus spans weeks rather than the multi-year target",
            "the 100 ms BBO path cannot resolve queue position or hidden depth",
            "base and adverse scenarios are research replays, not fill guarantees",
            "all returns are unleveraged and no profile may apply leverage before edge validation",
            "the local neural ensemble is machine learning and is not the optional LLM risk-assessment overlay",
            "the reserved terminal date was neither loaded nor labeled",
        ],
    }
    report["report_sha256"] = _canonical_sha256(report)
    write_json_atomic(destination / "report.json", report, indent=2, sort_keys=True)
    progress(
        "complete",
        status=report["status"],
        report_sha256=report["report_sha256"],
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the precommitted conditional outcome-mixture screen"
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--warehouse", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--compute-backend")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_outcome_mixture_screen(
        design_path=args.design,
        warehouse_path=args.warehouse,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        memory_limit=args.memory_limit,
        threads=args.threads,
        compute_backend=args.compute_backend,
    )
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
