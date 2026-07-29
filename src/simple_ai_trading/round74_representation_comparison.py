"""Development-only matched representation training and promotion gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from .impact_absorption_event_action_policy import (
    ROUND74_ACTION_PROFILES,
    Round74ActionPolicySelection,
    Round74ActionThresholdEvaluation,
)
from .impact_absorption_event_calibration import (
    ROUND74_TUNING_POLICY_SELECTION_RUNS,
    build_round74_tuning_subpartition,
)
from .impact_absorption_event_model import ROUND74_EVENT_MODEL_CANDIDATES
from .impact_absorption_event_training import (
    ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS,
    Round74EventTrainingConfig,
    load_round74_pretest_policy,
    round74_paired_run_stability_evidence,
)
from .round74_event_development_inputs import Round74DevelopmentInputs
from .round74_event_development_operator import (
    Round74DevelopmentPolicyArtifact,
    train_calibrate_and_select_round74_development_policy,
)
from .round74_event_model_operator import (
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74PreparedMatchedDevelopmentData,
    prepare_round74_matched_development_data,
    split_round74_prepared_tuning_roles,
)
from .storage import write_bytes_atomic


ROUND74_REPRESENTATION_COMPARISON_SCHEMA_VERSION = (
    "round-074-representation-comparison-v3"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Round 74 representation result has duplicate keys")
        value[key] = item
    return value


def _module_sha256() -> str:
    payload = Path(__file__).read_bytes()
    normalized = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def round74_representation_proper_loss_gate(
    baseline_run_losses: Sequence[float],
    challenger_run_losses: Sequence[float],
    *,
    minimum_mean_improvement: float,
) -> dict[str, object]:
    """Evaluate one exact paired proper-loss panel without test access."""

    baseline = tuple(float(value) for value in baseline_run_losses)
    challenger = tuple(float(value) for value in challenger_run_losses)
    minimum = float(minimum_mean_improvement)
    if (
        len(baseline) != ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
        or len(challenger) != len(baseline)
        or not math.isfinite(minimum)
        or minimum < 0.0
        or any(not math.isfinite(value) for value in (*baseline, *challenger))
    ):
        raise ValueError("Round 74 representation proper-loss panel differs")
    improvements = tuple(
        incumbent - candidate
        for incumbent, candidate in zip(baseline, challenger, strict=True)
    )
    mean_improvement = math.fsum(improvements) / len(improvements)
    maximum_degradation = max(-value for value in improvements)
    all_runs_noninferior = maximum_degradation <= minimum
    stability = round74_paired_run_stability_evidence(
        improvements,
        minimum_improvement=minimum,
    )
    promoted = (
        mean_improvement > minimum
        and all_runs_noninferior
        and stability["material_win_majority"] is True
        and stability[
            "all_leave_one_capture_run_out_panels_exceed_minimum_mean_improvement"
        ]
        is True
    )
    return {
        "paired_capture_run_count": len(improvements),
        "minimum_mean_proper_loss_improvement": minimum,
        "mean_proper_loss_improvement": mean_improvement,
        "maximum_permitted_paired_run_loss_degradation": minimum,
        "maximum_paired_run_loss_degradation": maximum_degradation,
        "challenger_win_count": sum(value > 0.0 for value in improvements),
        "challenger_loss_count": sum(value < 0.0 for value in improvements),
        "exact_tie_count": sum(value == 0.0 for value in improvements),
        "paired_run_proper_loss_improvements": list(improvements),
        "all_paired_runs_noninferior": all_runs_noninferior,
        **stability,
        "promoted": promoted,
        "sealed_test_accessed": False,
    }


def _selected_evaluation(
    policy: Round74ActionPolicySelection,
) -> Round74ActionThresholdEvaluation | None:
    policy.validate()
    if not policy.accepted:
        return None
    selected = tuple(
        evaluation
        for evaluation in policy.evaluations
        if evaluation.accepted
        and evaluation.quantile == policy.selected_quantile
        and evaluation.threshold_score == policy.selected_threshold_score
    )
    if len(selected) != 1:
        raise ValueError("Round 74 selected profile evaluation differs")
    return selected[0]


def _profile_policy_map(
    policies: Sequence[Round74ActionPolicySelection],
) -> dict[str, Round74ActionPolicySelection]:
    selected = tuple(policies)
    for policy in selected:
        policy.validate()
    result = {policy.profile: policy for policy in selected}
    if tuple(policy.profile for policy in selected) != ROUND74_ACTION_PROFILES or set(
        result
    ) != set(ROUND74_ACTION_PROFILES):
        raise ValueError("Round 74 representation profile panel differs")
    return result


def _paired_run_net_deltas(
    baseline: Round74ActionThresholdEvaluation,
    challenger: Round74ActionThresholdEvaluation,
) -> tuple[str, tuple[float, ...]]:
    baseline_trace = baseline.trace
    challenger_trace = challenger.trace
    baseline_run_ids = tuple(baseline_trace.expected_run_ids)
    challenger_run_ids = tuple(challenger_trace.expected_run_ids)
    if (
        baseline_run_ids != challenger_run_ids
        or len(baseline_run_ids) != ROUND74_TUNING_POLICY_SELECTION_RUNS
    ):
        raise ValueError("Round 74 representation economic run identity differs")

    def run_payoffs(
        evaluation: Round74ActionThresholdEvaluation,
    ) -> tuple[float, ...]:
        trace = evaluation.trace
        values = {run_id: 0.0 for run_id in baseline_run_ids}
        if len(trace.run_id) != len(trace.net_payoff_bps) or any(
            run_id not in values for run_id in trace.run_id
        ):
            raise ValueError("Round 74 representation economic trace differs")
        for run_id, payoff in zip(
            trace.run_id,
            trace.net_payoff_bps,
            strict=True,
        ):
            selected = float(payoff)
            if not math.isfinite(selected):
                raise ValueError("Round 74 representation economic payoff differs")
            values[run_id] += selected
        return tuple(values[run_id] for run_id in baseline_run_ids)

    baseline_payoffs = run_payoffs(baseline)
    challenger_payoffs = run_payoffs(challenger)
    deltas = tuple(
        candidate - incumbent
        for incumbent, candidate in zip(
            baseline_payoffs,
            challenger_payoffs,
            strict=True,
        )
    )
    return _canonical_sha256(list(baseline_run_ids)), deltas


def round74_representation_economic_gate(
    baseline_policies: Sequence[Round74ActionPolicySelection],
    challenger_policies: Sequence[Round74ActionPolicySelection],
) -> tuple[dict[str, object], ...]:
    """Require conservative acceptance and no delayed-economic regression."""

    baseline = _profile_policy_map(baseline_policies)
    challenger = _profile_policy_map(challenger_policies)
    reports: list[dict[str, object]] = []
    for profile in ROUND74_ACTION_PROFILES:
        baseline_evaluation = _selected_evaluation(baseline[profile])
        challenger_evaluation = _selected_evaluation(challenger[profile])
        reasons: list[str] = []
        comparison_required = baseline_evaluation is not None
        if profile == "conservative" and baseline_evaluation is None:
            reasons.append("conservative_baseline_policy_not_accepted")
        if profile == "conservative" and challenger_evaluation is None:
            reasons.append("conservative_challenger_policy_not_accepted")
        if baseline_evaluation is not None and challenger_evaluation is None:
            reasons.append("accepted_baseline_profile_became_rejected")
        paired_run_ids_sha256: str | None = None
        paired_run_net_deltas_bps: tuple[float, ...] | None = None
        minimum_paired_run_net_delta_bps: float | None = None
        challenger_worse_run_count: int | None = None
        all_paired_runs_net_noninferior: bool | None = None
        if baseline_evaluation is not None and challenger_evaluation is not None:
            baseline_metrics = baseline_evaluation.trace.metrics
            challenger_metrics = challenger_evaluation.trace.metrics
            paired_run_ids_sha256, paired_run_net_deltas_bps = _paired_run_net_deltas(
                baseline_evaluation,
                challenger_evaluation,
            )
            minimum_paired_run_net_delta_bps = min(paired_run_net_deltas_bps)
            challenger_worse_run_count = sum(
                value < 0.0 for value in paired_run_net_deltas_bps
            )
            all_paired_runs_net_noninferior = challenger_worse_run_count == 0
            comparisons = (
                (
                    challenger_evaluation.objective_bps
                    >= baseline_evaluation.objective_bps,
                    "run_balanced_objective_degraded",
                ),
                (
                    challenger_metrics.total_net_bps >= baseline_metrics.total_net_bps,
                    "total_after_cost_payoff_degraded",
                ),
                (
                    challenger_metrics.maximum_drawdown_bps
                    <= baseline_metrics.maximum_drawdown_bps,
                    "maximum_drawdown_increased",
                ),
                (
                    challenger_metrics.adverse_selection_rate
                    <= baseline_metrics.adverse_selection_rate,
                    "adverse_selection_rate_increased",
                ),
                (
                    challenger_metrics.mean_run_maximum_adverse_excursion_bps
                    <= baseline_metrics.mean_run_maximum_adverse_excursion_bps,
                    "mean_run_adverse_excursion_increased",
                ),
            )
            reasons.extend(reason for passed, reason in comparisons if not passed)
            if profile == "conservative" and not all_paired_runs_net_noninferior:
                reasons.append("conservative_paired_run_net_payoff_degraded")
        reports.append(
            {
                "profile": profile,
                "baseline_accepted": baseline_evaluation is not None,
                "challenger_accepted": challenger_evaluation is not None,
                "comparison_required": comparison_required,
                "baseline_selection_sha256": baseline[profile].selection_sha256,
                "challenger_selection_sha256": challenger[profile].selection_sha256,
                "baseline_selected_metrics": (
                    None
                    if baseline_evaluation is None
                    else {
                        "objective_bps": baseline_evaluation.objective_bps,
                        "total_net_bps": (
                            baseline_evaluation.trace.metrics.total_net_bps
                        ),
                        "maximum_drawdown_bps": (
                            baseline_evaluation.trace.metrics.maximum_drawdown_bps
                        ),
                        "adverse_selection_rate": (
                            baseline_evaluation.trace.metrics.adverse_selection_rate
                        ),
                        "mean_run_maximum_adverse_excursion_bps": (
                            baseline_evaluation.trace.metrics.mean_run_maximum_adverse_excursion_bps
                        ),
                    }
                ),
                "challenger_selected_metrics": (
                    None
                    if challenger_evaluation is None
                    else {
                        "objective_bps": challenger_evaluation.objective_bps,
                        "total_net_bps": (
                            challenger_evaluation.trace.metrics.total_net_bps
                        ),
                        "maximum_drawdown_bps": (
                            challenger_evaluation.trace.metrics.maximum_drawdown_bps
                        ),
                        "adverse_selection_rate": (
                            challenger_evaluation.trace.metrics.adverse_selection_rate
                        ),
                        "mean_run_maximum_adverse_excursion_bps": (
                            challenger_evaluation.trace.metrics.mean_run_maximum_adverse_excursion_bps
                        ),
                    }
                ),
                "paired_run_ids_sha256": paired_run_ids_sha256,
                "paired_run_net_deltas_bps": (
                    None
                    if paired_run_net_deltas_bps is None
                    else list(paired_run_net_deltas_bps)
                ),
                "minimum_paired_run_net_delta_bps": (minimum_paired_run_net_delta_bps),
                "challenger_worse_run_count": challenger_worse_run_count,
                "all_paired_runs_net_noninferior": (all_paired_runs_net_noninferior),
                "noninferior": not reasons,
                "reasons": reasons,
                "sealed_test_accessed": False,
            }
        )
    return tuple(reports)


def _validate_proper_loss_report(value: Mapping[str, object]) -> bool:
    raw_improvements = value.get("paired_run_proper_loss_improvements")
    minimum = value.get("minimum_mean_proper_loss_improvement")
    if (
        not isinstance(raw_improvements, list)
        or len(raw_improvements) != ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in raw_improvements
        )
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or float(minimum) < 0.0
    ):
        raise ValueError("Round 74 representation proper-loss report differs")
    improvements = tuple(float(item) for item in raw_improvements)
    expected = round74_representation_proper_loss_gate(
        (0.0,) * len(improvements),
        tuple(-item for item in improvements),
        minimum_mean_improvement=float(minimum),
    )
    if dict(value) != expected:
        raise ValueError("Round 74 representation proper-loss result differs")
    return expected["promoted"] is True


def _validate_profile_reports(
    values: Sequence[Mapping[str, object]],
) -> bool:
    expected_keys = {
        "profile",
        "baseline_accepted",
        "challenger_accepted",
        "comparison_required",
        "baseline_selection_sha256",
        "challenger_selection_sha256",
        "baseline_selected_metrics",
        "challenger_selected_metrics",
        "paired_run_ids_sha256",
        "paired_run_net_deltas_bps",
        "minimum_paired_run_net_delta_bps",
        "challenger_worse_run_count",
        "all_paired_runs_net_noninferior",
        "noninferior",
        "reasons",
        "sealed_test_accessed",
    }
    metric_keys = {
        "objective_bps",
        "total_net_bps",
        "maximum_drawdown_bps",
        "adverse_selection_rate",
        "mean_run_maximum_adverse_excursion_bps",
    }
    selected = tuple(dict(value) for value in values)
    if tuple(value.get("profile") for value in selected) != ROUND74_ACTION_PROFILES:
        raise ValueError("Round 74 representation profile order differs")
    for value in selected:
        reasons = value.get("reasons")
        if (
            set(value) != expected_keys
            or any(
                not isinstance(value.get(name), bool)
                for name in (
                    "baseline_accepted",
                    "challenger_accepted",
                    "comparison_required",
                    "noninferior",
                )
            )
            or value.get("sealed_test_accessed") is not False
            or not _is_sha256(value.get("baseline_selection_sha256"))
            or not _is_sha256(value.get("challenger_selection_sha256"))
            or not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or value.get("noninferior") is not (not reasons)
        ):
            raise ValueError("Round 74 representation profile report differs")
        for prefix in ("baseline", "challenger"):
            metrics = value.get(f"{prefix}_selected_metrics")
            accepted = value.get(f"{prefix}_accepted")
            if (metrics is None) is bool(accepted):
                raise ValueError("Round 74 representation profile metrics differ")
            if metrics is not None and (
                not isinstance(metrics, Mapping)
                or set(metrics) != metric_keys
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in metrics.values()
                )
            ):
                raise ValueError("Round 74 representation profile metrics differ")
        paired_run_ids_sha256 = value.get("paired_run_ids_sha256")
        paired_run_net_deltas = value.get("paired_run_net_deltas_bps")
        minimum_paired_run_net_delta = value.get("minimum_paired_run_net_delta_bps")
        worse_run_count = value.get("challenger_worse_run_count")
        all_run_noninferior = value.get("all_paired_runs_net_noninferior")
        both_accepted = (
            value.get("baseline_accepted") is True
            and value.get("challenger_accepted") is True
        )
        if not both_accepted:
            if any(
                item is not None
                for item in (
                    paired_run_ids_sha256,
                    paired_run_net_deltas,
                    minimum_paired_run_net_delta,
                    worse_run_count,
                    all_run_noninferior,
                )
            ):
                raise ValueError("Round 74 representation paired economics differ")
            continue
        if (
            not _is_sha256(paired_run_ids_sha256)
            or not isinstance(paired_run_net_deltas, list)
            or len(paired_run_net_deltas) != ROUND74_TUNING_POLICY_SELECTION_RUNS
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in paired_run_net_deltas
            )
            or isinstance(minimum_paired_run_net_delta, bool)
            or not isinstance(minimum_paired_run_net_delta, (int, float))
            or not math.isfinite(float(minimum_paired_run_net_delta))
            or isinstance(worse_run_count, bool)
            or not isinstance(worse_run_count, int)
            or not 0 <= worse_run_count <= ROUND74_TUNING_POLICY_SELECTION_RUNS
            or not isinstance(all_run_noninferior, bool)
        ):
            raise ValueError("Round 74 representation paired economics differ")
        expected_minimum = min(float(item) for item in paired_run_net_deltas)
        expected_worse_count = sum(float(item) < 0.0 for item in paired_run_net_deltas)
        if (
            float(minimum_paired_run_net_delta) != expected_minimum
            or worse_run_count != expected_worse_count
            or all_run_noninferior is not (expected_worse_count == 0)
            or (
                value["profile"] == "conservative"
                and ("conservative_paired_run_net_payoff_degraded" in reasons)
                is all_run_noninferior
            )
        ):
            raise ValueError("Round 74 representation paired economics differ")
    return all(value["noninferior"] is True for value in selected)


@dataclass(frozen=True)
class Round74RepresentationComparison:
    matched_preparation_sha256: str
    baseline_pretest_policy_sha256: str
    challenger_pretest_policy_sha256: str
    baseline_development_bundle_sha256: str
    challenger_development_bundle_sha256: str
    fixed_candidate_id: str
    seeds: tuple[int, ...]
    proper_loss_gate: Mapping[str, object]
    profile_economic_gates: tuple[Mapping[str, object], ...]
    selected_representation: str
    promoted: bool
    source_module_sha256: str
    schema_version: str = ROUND74_REPRESENTATION_COMPARISON_SCHEMA_VERSION

    def validate(self) -> None:
        digests = (
            self.matched_preparation_sha256,
            self.baseline_pretest_policy_sha256,
            self.challenger_pretest_policy_sha256,
            self.baseline_development_bundle_sha256,
            self.challenger_development_bundle_sha256,
            self.source_module_sha256,
        )
        proper = dict(self.proper_loss_gate)
        profiles = tuple(dict(value) for value in self.profile_economic_gates)
        proper_promoted = _validate_proper_loss_report(proper)
        economic_noninferior = _validate_profile_reports(profiles)
        expected_promoted = proper_promoted and economic_noninferior
        if (
            self.schema_version != ROUND74_REPRESENTATION_COMPARISON_SCHEMA_VERSION
            or any(not _is_sha256(value) for value in digests)
            or self.source_module_sha256 != _module_sha256()
            or not self.fixed_candidate_id.strip()
            or not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.seeds
            )
            or tuple(value.get("profile") for value in profiles)
            != ROUND74_ACTION_PROFILES
            or any(value.get("sealed_test_accessed") is not False for value in profiles)
            or proper.get("sealed_test_accessed") is not False
            or self.promoted is not expected_promoted
            or self.selected_representation
            != ("global_cross_asset" if expected_promoted else "per_symbol")
        ):
            raise ValueError("Round 74 representation comparison differs")

    @property
    def comparison_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "matched_preparation_sha256": self.matched_preparation_sha256,
            "baseline_representation": "per_symbol",
            "challenger_representation": "global_cross_asset",
            "baseline_pretest_policy_sha256": self.baseline_pretest_policy_sha256,
            "challenger_pretest_policy_sha256": (self.challenger_pretest_policy_sha256),
            "baseline_development_bundle_sha256": (
                self.baseline_development_bundle_sha256
            ),
            "challenger_development_bundle_sha256": (
                self.challenger_development_bundle_sha256
            ),
            "fixed_candidate_id": self.fixed_candidate_id,
            "seeds": list(self.seeds),
            "proper_loss_gate": dict(self.proper_loss_gate),
            "profile_economic_gates": [
                dict(value) for value in self.profile_economic_gates
            ],
            "selected_representation": self.selected_representation,
            "promoted": self.promoted,
            "source_module_sha256": self.source_module_sha256,
            "selection_data_role": "development_tuning_only",
            "sealed_test_accessed": False,
            "profitability_claim": False,
            "market_edge_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        }
        if include_sha256:
            value["comparison_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74RepresentationComparison:
        payload = dict(value)
        claimed = payload.pop("comparison_sha256", None)
        expected_keys = {
            "schema_version",
            "matched_preparation_sha256",
            "baseline_representation",
            "challenger_representation",
            "baseline_pretest_policy_sha256",
            "challenger_pretest_policy_sha256",
            "baseline_development_bundle_sha256",
            "challenger_development_bundle_sha256",
            "fixed_candidate_id",
            "seeds",
            "proper_loss_gate",
            "profile_economic_gates",
            "selected_representation",
            "promoted",
            "source_module_sha256",
            "selection_data_role",
            "sealed_test_accessed",
            "profitability_claim",
            "market_edge_claim",
            "paper_trading_authority",
            "testnet_trading_authority",
            "live_trading_authority",
        }
        seeds = payload.get("seeds")
        proper = payload.get("proper_loss_gate")
        profiles = payload.get("profile_economic_gates")
        if (
            not _is_sha256(claimed)
            or claimed != _canonical_sha256(payload)
            or set(payload) != expected_keys
            or payload.get("baseline_representation") != "per_symbol"
            or payload.get("challenger_representation") != "global_cross_asset"
            or payload.get("selection_data_role") != "development_tuning_only"
            or any(
                payload.get(name) is not False
                for name in (
                    "sealed_test_accessed",
                    "profitability_claim",
                    "market_edge_claim",
                    "paper_trading_authority",
                    "testnet_trading_authority",
                    "live_trading_authority",
                )
            )
            or not isinstance(seeds, list)
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
            )
            or not isinstance(proper, Mapping)
            or not isinstance(profiles, list)
            or any(not isinstance(item, Mapping) for item in profiles)
            or not isinstance(payload.get("promoted"), bool)
            or not isinstance(payload.get("fixed_candidate_id"), str)
            or not isinstance(payload.get("selected_representation"), str)
        ):
            raise ValueError("Round 74 representation result payload differs")
        selected = cls(
            matched_preparation_sha256=str(payload["matched_preparation_sha256"]),
            baseline_pretest_policy_sha256=str(
                payload["baseline_pretest_policy_sha256"]
            ),
            challenger_pretest_policy_sha256=str(
                payload["challenger_pretest_policy_sha256"]
            ),
            baseline_development_bundle_sha256=str(
                payload["baseline_development_bundle_sha256"]
            ),
            challenger_development_bundle_sha256=str(
                payload["challenger_development_bundle_sha256"]
            ),
            fixed_candidate_id=str(payload["fixed_candidate_id"]),
            seeds=tuple(int(seed) for seed in seeds),
            proper_loss_gate=dict(proper),
            profile_economic_gates=tuple(dict(item) for item in profiles),
            selected_representation=str(payload["selected_representation"]),
            promoted=bool(payload["promoted"]),
            source_module_sha256=str(payload["source_module_sha256"]),
            schema_version=str(payload["schema_version"]),
        )
        selected.validate()
        if selected.as_dict() != value:
            raise ValueError("Round 74 representation result round trip differs")
        return selected


@dataclass(frozen=True)
class Round74RepresentationComparisonArtifact:
    comparison: Round74RepresentationComparison
    comparison_path: Path
    baseline: Round74DevelopmentPolicyArtifact
    challenger: Round74DevelopmentPolicyArtifact


def load_round74_representation_comparison(
    path: str | Path,
) -> Round74RepresentationComparison:
    selected_path = Path(path)
    try:
        payload = json.loads(
            selected_path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 representation result could not be read") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 74 representation result root differs")
    result = Round74RepresentationComparison.from_dict(payload)
    expected_name = f"round74-representation-comparison-{result.comparison_sha256}.json"
    if selected_path.name != expected_name:
        raise ValueError("Round 74 representation result filename differs")
    return result


def _policy_candidate_losses(
    policy: Mapping[str, object],
    *,
    expected_representation: str,
    expected_matched_preparation_sha256: str,
    expected_architecture_selection_mode: str,
    expected_candidate_id: str | None = None,
) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    development = policy.get("development_data")
    training_policy = policy.get("training_policy")
    selection = policy.get("selection")
    panel = policy.get("candidate_panel")
    if not all(
        isinstance(value, Mapping)
        for value in (development, training_policy, selection, panel)
    ):
        raise ValueError("Round 74 representation policy sections differ")
    assert isinstance(development, Mapping)
    assert isinstance(training_policy, Mapping)
    assert isinstance(selection, Mapping)
    assert isinstance(panel, Mapping)
    candidate_id = str(selection.get("selected_candidate_id", ""))
    seeds = training_policy.get("seeds")
    candidate_ids = training_policy.get("candidate_ids")
    report = panel.get(candidate_id)
    expected_candidate_ids = (
        list(ROUND74_EVENT_MODEL_CANDIDATES)
        if expected_architecture_selection_mode == "complexity_gate"
        else [expected_candidate_id]
    )
    if (
        development.get("window_representation") != expected_representation
        or development.get("representative_window_policy_kind")
        != "matched_representation"
        or development.get("matched_preparation_sha256")
        != expected_matched_preparation_sha256
        or (expected_candidate_id is not None and candidate_id != expected_candidate_id)
        or training_policy.get("architecture_selection_mode")
        != expected_architecture_selection_mode
        or selection.get("architecture_selection_mode")
        != expected_architecture_selection_mode
        or candidate_ids != expected_candidate_ids
        or not isinstance(seeds, list)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or not isinstance(report, Mapping)
    ):
        raise ValueError("Round 74 representation policy identity differs")
    losses = report.get("ensemble_tuning_run_losses")
    if not isinstance(losses, list):
        raise ValueError("Round 74 representation policy losses differ")
    return candidate_id, tuple(seeds), tuple(float(value) for value in losses)


def build_round74_representation_comparison(
    prepared: Round74PreparedMatchedDevelopmentData,
    *,
    baseline: Round74DevelopmentPolicyArtifact,
    challenger: Round74DevelopmentPolicyArtifact,
    baseline_policy: Mapping[str, object],
    challenger_policy: Mapping[str, object],
    minimum_mean_improvement: float,
) -> Round74RepresentationComparison:
    """Build the development-only decision after both exact workflows finish."""

    prepared.validate()
    baseline.bundle.validate()
    challenger.bundle.validate()
    preparation_sha256 = prepared.preparation_sha256
    expected_batches = {
        "per_symbol": prepared.tuning.per_symbol,
        "global_cross_asset": prepared.tuning.global_cross_asset,
    }
    for artifact, policy, representation in (
        (baseline, baseline_policy, "per_symbol"),
        (challenger, challenger_policy, "global_cross_asset"),
    ):
        batches = expected_batches[representation]
        model_end = len(artifact.bundle.model_selection_batch_sha256)
        calibration_end = model_end + len(
            artifact.bundle.calibration_batch_sha256
        )
        policy_end = calibration_end + len(
            artifact.bundle.policy_selection_batch_sha256
        )
        if (
            artifact.bundle_sha256 != artifact.bundle.bundle_sha256
            or policy.get("policy_sha256") != artifact.pretest_policy.policy_sha256
            or len(batches) < policy_end
            or artifact.bundle.model_selection_batch_sha256
            != tuple(batch.batch_sha256 for batch in batches[:model_end])
            or artifact.bundle.calibration_batch_sha256
            != tuple(
                batch.batch_sha256 for batch in batches[model_end:calibration_end]
            )
            or artifact.bundle.policy_selection_batch_sha256
            != tuple(
                batch.batch_sha256 for batch in batches[calibration_end:policy_end]
            )
        ):
            raise ValueError("Round 74 representation development binding differs")
    candidate_id, baseline_seeds, baseline_losses = _policy_candidate_losses(
        baseline_policy,
        expected_representation="per_symbol",
        expected_matched_preparation_sha256=preparation_sha256,
        expected_architecture_selection_mode="complexity_gate",
    )
    challenger_candidate, challenger_seeds, challenger_losses = (
        _policy_candidate_losses(
            challenger_policy,
            expected_representation="global_cross_asset",
            expected_matched_preparation_sha256=preparation_sha256,
            expected_architecture_selection_mode="fixed",
            expected_candidate_id=candidate_id,
        )
    )
    if (
        candidate_id != challenger_candidate
        or baseline_seeds != challenger_seeds
        or baseline.bundle.feature_scaler_sha256
        != challenger.bundle.feature_scaler_sha256
        or baseline.bundle.tuning_subpartition_sha256
        != challenger.bundle.tuning_subpartition_sha256
    ):
        raise ValueError("Round 74 representation comparison identity differs")
    proper = round74_representation_proper_loss_gate(
        baseline_losses,
        challenger_losses,
        minimum_mean_improvement=minimum_mean_improvement,
    )
    profiles = round74_representation_economic_gate(
        baseline.bundle.action_policies,
        challenger.bundle.action_policies,
    )
    promoted = bool(proper["promoted"]) and all(
        value["noninferior"] is True for value in profiles
    )
    result = Round74RepresentationComparison(
        matched_preparation_sha256=preparation_sha256,
        baseline_pretest_policy_sha256=baseline.pretest_policy.policy_sha256,
        challenger_pretest_policy_sha256=challenger.pretest_policy.policy_sha256,
        baseline_development_bundle_sha256=baseline.bundle_sha256,
        challenger_development_bundle_sha256=challenger.bundle_sha256,
        fixed_candidate_id=candidate_id,
        seeds=baseline_seeds,
        proper_loss_gate=proper,
        profile_economic_gates=profiles,
        selected_representation=("global_cross_asset" if promoted else "per_symbol"),
        promoted=promoted,
        source_module_sha256=_module_sha256(),
    )
    result.validate()
    return result


def train_and_compare_round74_representations(
    store: object,
    inputs: Round74DevelopmentInputs,
    *,
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    inference_minibatch_rows: int = 128,
) -> Round74RepresentationComparisonArtifact:
    """Train the baseline, hold architecture fixed, then gate cross-asset context."""

    inputs.validate()
    selected_config = config or Round74EventTrainingConfig()
    selected_config.validate()
    if selected_config.architecture_selection_mode != "complexity_gate":
        raise ValueError("Round 74 baseline requires the complexity gate")
    assemblies = inputs.target_assembly_by_run_id()
    prepared = prepare_round74_matched_development_data(
        store,
        partition=inputs.partition,
        target_assembly_by_run_id=assemblies,
    )
    subpartition = build_round74_tuning_subpartition(inputs.partition)
    prepared_by_representation = {
        representation: prepared.representation(representation)
        for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS
    }
    roles = {
        representation: split_round74_prepared_tuning_roles(
            prepared_by_representation[representation],
            subpartition=subpartition,
        )
        for representation in ROUND74_EVENT_WINDOW_REPRESENTATIONS
    }
    execution_assemblies = {
        run_id: assemblies[run_id] for run_id in subpartition.policy_selection_run_ids
    }
    output = Path(output_directory)
    baseline = train_calibrate_and_select_round74_development_policy(
        prepared_by_representation["per_symbol"],
        roles["per_symbol"],
        output_directory=output / "per-symbol",
        execution_store=store,
        execution_partition=inputs.partition,
        execution_target_assembly_by_run_id=execution_assemblies,
        compute_backend=compute_backend,
        config=selected_config,
        inference_minibatch_rows=inference_minibatch_rows,
        matched_preparation_sha256=prepared.preparation_sha256,
    )
    _baseline_model, baseline_policy = load_round74_pretest_policy(
        baseline.pretest_policy.policy_path
    )
    selected_candidate = baseline.pretest_policy.selected_candidate_id
    challenger_config = replace(
        selected_config,
        candidate_ids=(selected_candidate,),
        architecture_selection_mode="fixed",
    )
    challenger_config.validate()
    challenger = train_calibrate_and_select_round74_development_policy(
        prepared_by_representation["global_cross_asset"],
        roles["global_cross_asset"],
        output_directory=output / "global-cross-asset",
        execution_store=store,
        execution_partition=inputs.partition,
        execution_target_assembly_by_run_id=execution_assemblies,
        compute_backend=compute_backend,
        config=challenger_config,
        inference_minibatch_rows=inference_minibatch_rows,
        matched_preparation_sha256=prepared.preparation_sha256,
    )
    _challenger_model, challenger_policy = load_round74_pretest_policy(
        challenger.pretest_policy.policy_path
    )
    comparison = build_round74_representation_comparison(
        prepared,
        baseline=baseline,
        challenger=challenger,
        baseline_policy=baseline_policy,
        challenger_policy=challenger_policy,
        minimum_mean_improvement=selected_config.minimum_tuning_improvement,
    )
    payload = _canonical_bytes(comparison.as_dict()) + b"\n"
    path = (
        output
        / f"round74-representation-comparison-{comparison.comparison_sha256}.json"
    )
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("Round 74 immutable representation result differs")
    else:
        write_bytes_atomic(path, payload)
    persisted = load_round74_representation_comparison(path)
    if persisted.as_dict() != comparison.as_dict():
        raise RuntimeError("Round 74 persisted representation result differs")
    return Round74RepresentationComparisonArtifact(
        comparison=comparison,
        comparison_path=path,
        baseline=baseline,
        challenger=challenger,
    )


__all__ = [
    "ROUND74_REPRESENTATION_COMPARISON_SCHEMA_VERSION",
    "Round74RepresentationComparison",
    "Round74RepresentationComparisonArtifact",
    "build_round74_representation_comparison",
    "load_round74_representation_comparison",
    "round74_representation_economic_gate",
    "round74_representation_proper_loss_gate",
    "train_and_compare_round74_representations",
]
