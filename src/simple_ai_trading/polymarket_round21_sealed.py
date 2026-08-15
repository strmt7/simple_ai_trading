"""One-use sealed predictive and economic verdicts for Polymarket Round 21."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re

from .polymarket_round21_ai_comparison import Round21AIMatchedComparison
from .polymarket_round21_ai import (
    POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256,
    POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
)
from .polymarket_round21_ai_selection import (
    POLYMARKET_ROUND21_AI_CANDIDATES,
    POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256,
)
from .polymarket_round21_comparison import (
    POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256,
    Round21MatchedEconomicComparison,
    round21_replay_matrix_sha256,
)
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_execution import POLYMARKET_ROUND21_EXECUTION_SCENARIOS
from .polymarket_round21_model import (
    POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
    Round21DevelopmentPanel,
    predict_round21_controls,
    predict_round21_probability_batch,
    round21_paired_predictive_improvement,
    round21_predictive_diagnostics,
    validate_round21_development_artifact,
)
from .polymarket_round21_policy import (
    POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256,
    POLYMARKET_ROUND21_RISK_PROFILES,
)
from .polymarket_round21_replay import (
    POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
    Round21EconomicReplay,
)


POLYMARKET_ROUND21_SEALED_DESIGN_SCHEMA_VERSION = (
    "polymarket-round21-terminal-sealed-evaluation-design-v7"
)
POLYMARKET_ROUND21_SEALED_DESIGN_SHA256 = (
    "57780ffd28fcbf8f9f1be0c9ce41d89c5f1b4a45635d257d32c5cdda30746abf"
)
POLYMARKET_ROUND21_SEALED_PREDICTIVE_SCHEMA_VERSION = (
    "polymarket-round21-sealed-predictive-result-v7"
)
POLYMARKET_ROUND21_SEALED_ECONOMIC_SCHEMA_VERSION = (
    "polymarket-round21-sealed-economic-result-v7"
)
POLYMARKET_ROUND21_SEALED_RESULT_SCHEMA_VERSION = (
    "polymarket-round21-one-use-sealed-result-v7"
)
POLYMARKET_ROUND21_SEALED_BUNDLE_SCHEMA_VERSION = (
    "polymarket-round21-sealed-result-bundle-v7"
)
POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS = 1_800
POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS = 7
_DESIGN_RELATIVE = (
    "docs/model-research/polymarket/round-021-terminal-sealed-evaluation-design-v7.json"
)
_LAYERS = ("core", "core_spot", "core_spot_usdm")
_CONTROL_IDS = (
    "executable_market_prior_calibrated",
    "executable_market_prior_raw",
    "structural_probability_calibrated",
    "structural_probability_raw",
    "training_prevalence",
)
_DESIGN_CONTROL_IDS = (
    "structural_probability_raw",
    "structural_probability_calibrated",
    "executable_market_prior_raw",
    "executable_market_prior_calibrated",
    "training_prevalence",
)
_METRIC_NAMES = (
    "condition_count",
    "condition_equal_log_loss",
    "condition_equal_brier_score",
    "log_loss_standard_error",
    "calibration_intercept",
    "calibration_slope",
    "expected_calibration_error",
    "balanced_accuracy",
    "matthews_correlation_coefficient",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_VERIFIED_BUNDLE_CAPABILITY = object()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _condition_population_sha256(
    condition_ids: Sequence[object],
    event_starts_ms: Sequence[object],
) -> tuple[int, str]:
    if len(condition_ids) != len(event_starts_ms) or not condition_ids:
        raise ValueError("Round 21 sealed condition population differs")
    population: dict[str, int] = {}
    for raw_condition, raw_start in zip(
        condition_ids,
        event_starts_ms,
        strict=True,
    ):
        condition_id = str(raw_condition or "").strip()
        event_start_ms = int(raw_start)
        prior = population.get(condition_id)
        if (
            not condition_id
            or event_start_ms <= 0
            or (prior is not None and prior != event_start_ms)
        ):
            raise ValueError("Round 21 sealed condition population differs")
        population[condition_id] = event_start_ms
    ordered = tuple(sorted(population.items(), key=lambda value: (value[1], value[0])))
    return len(ordered), _canonical_sha256(
        {
            "schema_version": "polymarket-round21-sealed-condition-population-v1",
            "conditions": [
                {"condition_id": condition_id, "event_start_ms": event_start_ms}
                for condition_id, event_start_ms in ordered
            ],
        }
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 21 sealed design contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 sealed design contains {value}")


def validate_round21_sealed_design(value: Mapping[str, object]) -> dict[str, object]:
    design = dict(value)
    claimed = str(design.pop("design_sha256", "")).strip().lower()
    benchmark = design.get("external_null_benchmark")
    parents = design.get("parents")
    development = design.get("development_seal")
    one_use = design.get("one_use_state_machine")
    predictive = design.get("sealed_predictive_gate")
    economic = design.get("sealed_economic_gate")
    authority = design.get("authority")
    if (
        claimed != POLYMARKET_ROUND21_SEALED_DESIGN_SHA256
        or claimed != _canonical_sha256(design)
        or design.get("schema_version")
        != POLYMARKET_ROUND21_SEALED_DESIGN_SCHEMA_VERSION
        or design.get("round") != 21
        or not isinstance(parents, Mapping)
        or parents.get("round21_ai_veto_design_sha256")
        != POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256
        or parents.get("round21_ai_historical_schedule_design_sha256")
        != POLYMARKET_ROUND21_AI_HISTORICAL_SCHEDULE_DESIGN_SHA256
        or parents.get("round21_ai_selection_design_sha256")
        != POLYMARKET_ROUND21_AI_SELECTION_DESIGN_SHA256
        or parents.get("round21_model_design_sha256")
        != POLYMARKET_ROUND21_MODEL_DESIGN_SHA256
        or parents.get("round21_multi_action_policy_sha256")
        != POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
        or parents.get("round21_economic_replay_design_sha256")
        != POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256
        or parents.get("round21_matched_comparison_design_sha256")
        != POLYMARKET_ROUND21_MATCHED_COMPARISON_DESIGN_SHA256
        or parents.get("round21_terminal_sealed_evaluation_design_v6_sha256")
        != "26a2e83466525351ea680f0106d4e93eaa9409282d51fb55f958149dc1d0f7ee"
        or not isinstance(benchmark, Mapping)
        or benchmark.get("finding")
        != (
            "no_out_of_sample_tradable_edge_for_published_15_minute_"
            "logistic_baseline_after_stated_costs"
        )
        or benchmark.get("source_clock_offset_ambiguity_must_not_define_causal_order")
        is not True
        or benchmark.get("collector_receipt_clock_sensitivity_required") is not True
        or benchmark.get("market_prior_control_required") is not True
        or not isinstance(development, Mapping)
        or development.get("optional_sidecar_failure_blocks_core") is not False
        or tuple(development.get("ai_candidates_finite_and_development_only", ()))
        != POLYMARKET_ROUND21_AI_CANDIDATES
        or development.get("no_test_refit_recalibration_threshold_or_policy_change")
        is not True
        or development.get(
            "sealed_result_ai_identity_must_equal_development_nomination"
        )
        is not True
        or development.get("feature_support_contract_validated_before_test_access")
        is not True
        or not isinstance(one_use, Mapping)
        or one_use.get("claim_persisted_before_test_feature_target_execution_access")
        is not True
        or one_use.get("return_to_development_after_test_access") is not False
        or one_use.get("result_binds_predeclared_test_population") is not True
        or not isinstance(predictive, Mapping)
        or predictive.get("minimum_resolved_conditions")
        != POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS
        or predictive.get("minimum_calendar_days")
        != POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS
        or tuple(predictive.get("controls", ())) != _DESIGN_CONTROL_IDS
        or predictive.get("unsupported_rows_remain_in_all_proper_scoring_metrics")
        is not True
        or not isinstance(economic, Mapping)
        or economic.get("ledger_count") != 81
        or economic.get("every_ledger_must_pass") is not True
        or economic.get("feature_support_gate_applied_identically_to_all_81_ledgers")
        is not True
        or not isinstance(authority, Mapping)
        or any(value is not False for value in authority.values())
    ):
        raise ValueError("Round 21 sealed evaluation design differs")
    return {**design, "design_sha256": claimed}


def load_round21_sealed_design(repository: str | Path) -> dict[str, object]:
    path = Path(repository).resolve() / _DESIGN_RELATIVE
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ValueError("Round 21 sealed evaluation design is unavailable")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 sealed evaluation design is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 sealed evaluation design is invalid")
    return validate_round21_sealed_design(value)


def _valid_diagnostics(value: object, *, conditions: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(_METRIC_NAMES):
        return False
    try:
        numeric = tuple(float(value[name]) for name in _METRIC_NAMES[1:])
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        value.get("condition_count") == conditions
        and all(math.isfinite(item) for item in numeric)
        and 0.0 <= float(value["condition_equal_log_loss"])
        and 0.0 <= float(value["condition_equal_brier_score"]) <= 1.0
        and 0.0 <= float(value["log_loss_standard_error"])
        and 0.0 <= float(value["expected_calibration_error"]) <= 1.0
        and 0.0 <= float(value["balanced_accuracy"]) <= 1.0
        and -1.0 <= float(value["matthews_correlation_coefficient"]) <= 1.0
    )


def _valid_improvement(value: object, *, conditions: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "condition_count",
        "mean",
        "lower_95",
        "upper_95",
    }:
        return False
    try:
        mean = float(value["mean"])
        lower = float(value["lower_95"])
        upper = float(value["upper_95"])
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        value.get("condition_count") == conditions
        and all(math.isfinite(item) for item in (mean, lower, upper))
        and lower <= upper
    )


@dataclass(frozen=True, slots=True)
class Round21SealedPredictiveResult:
    population_layer: str
    model_artifact_sha256: str
    test_dataset_sha256: str
    test_target_manifest_sha256: str
    probability_batch_sha256: str
    resolved_condition_count: int
    condition_population_sha256: str
    calendar_day_count: int
    candidate_metrics: Mapping[str, float | int]
    control_metrics: Mapping[str, Mapping[str, float | int]]
    paired_improvements: Mapping[str, Mapping[str, Mapping[str, float | int]]]
    gate_passed: bool
    reasons: tuple[str, ...]
    result_sha256: str
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_SEALED_PREDICTIVE_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
            "contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
            "population_layer": self.population_layer,
            "model_artifact_sha256": self.model_artifact_sha256,
            "test_dataset_sha256": self.test_dataset_sha256,
            "test_target_manifest_sha256": self.test_target_manifest_sha256,
            "probability_batch_sha256": self.probability_batch_sha256,
            "resolved_condition_count": self.resolved_condition_count,
            "condition_population_sha256": self.condition_population_sha256,
            "calendar_day_count": self.calendar_day_count,
            "candidate_metrics": dict(self.candidate_metrics),
            "control_metrics": {
                name: dict(value) for name, value in self.control_metrics.items()
            },
            "paired_improvements": {
                name: {metric: dict(value) for metric, value in metrics.items()}
                for name, metrics in self.paired_improvements.items()
            },
            "gate_passed": self.gate_passed,
            "reasons": list(self.reasons),
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "result_sha256": self.result_sha256}

    def validated(self) -> Round21SealedPredictiveResult:
        comparisons = self.paired_improvements
        if (
            self.population_layer not in _LAYERS
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.model_artifact_sha256,
                    self.test_dataset_sha256,
                    self.test_target_manifest_sha256,
                    self.probability_batch_sha256,
                    self.condition_population_sha256,
                    self.result_sha256,
                )
            )
            or self.resolved_condition_count < 1
            or self.calendar_day_count < 1
            or not _valid_diagnostics(
                self.candidate_metrics,
                conditions=self.resolved_condition_count,
            )
            or not isinstance(self.control_metrics, Mapping)
            or tuple(sorted(self.control_metrics)) != _CONTROL_IDS
            or any(
                not _valid_diagnostics(value, conditions=self.resolved_condition_count)
                for value in self.control_metrics.values()
            )
            or not isinstance(comparisons, Mapping)
            or tuple(sorted(comparisons)) != _CONTROL_IDS
            or any(
                not isinstance(metrics, Mapping)
                or set(metrics) != {"log_loss", "brier"}
                or any(
                    not _valid_improvement(
                        value,
                        conditions=self.resolved_condition_count,
                    )
                    for value in metrics.values()
                )
                for metrics in comparisons.values()
            )
        ):
            raise ValueError("Round 21 sealed predictive result differs")
        expected_gate = (
            self.resolved_condition_count
            >= POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS
            and self.calendar_day_count >= POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS
            and all(
                float(comparisons[control][metric]["lower_95"]) > 0.0
                for control in _CONTROL_IDS
                for metric in ("log_loss", "brier")
            )
        )
        expected_reasons: list[str] = []
        if self.resolved_condition_count < POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS:
            expected_reasons.append("insufficient_resolved_conditions")
        if self.calendar_day_count < POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS:
            expected_reasons.append("insufficient_calendar_days")
        if any(
            float(comparisons[control][metric]["lower_95"]) <= 0.0
            for control in _CONTROL_IDS
            for metric in ("log_loss", "brier")
        ):
            expected_reasons.append("not_better_than_every_control")
        if (
            self.gate_passed != expected_gate
            or self.reasons != tuple(expected_reasons)
            or type(self.gate_passed) is not bool
            or type(self.profitability_claim) is not bool
            or type(self.paper_trading_authority) is not bool
            or type(self.live_trading_authority) is not bool
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 sealed predictive result differs")
        return self


def evaluate_round21_sealed_predictions(
    artifact: Mapping[str, object],
    *,
    population_layer: str,
    test_panel: Round21DevelopmentPanel,
) -> Round21SealedPredictiveResult:
    """Consume one target-bearing test panel without refitting any model."""

    model_artifact = validate_round21_development_artifact(artifact)
    panel = test_panel.validate()
    if panel.role != "test":
        raise ValueError("Round 21 sealed predictive panel role differs")
    batch = predict_round21_probability_batch(
        model_artifact,
        population_layer=population_layer,
        panel=panel,
    )
    indices = batch.indices
    condition_ids = panel.condition_ids[indices]
    labels = panel.labels[indices]
    candidate = batch.probability_up
    controls = predict_round21_controls(model_artifact, panel)
    selected_controls = {
        name: prediction[indices] for name, prediction in controls.items()
    }
    candidate_metrics = round21_predictive_diagnostics(
        condition_ids,
        labels,
        candidate,
    )
    control_metrics = {
        name: round21_predictive_diagnostics(condition_ids, labels, prediction)
        for name, prediction in selected_controls.items()
    }
    paired = {
        name: {
            metric: round21_paired_predictive_improvement(
                condition_ids,
                labels,
                prediction,
                candidate,
                metric=metric,
                seed_offset=1_000 + control_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(("log_loss", "brier"))
        }
        for control_index, (name, prediction) in enumerate(
            sorted(selected_controls.items())
        )
    }
    resolved_conditions, condition_population_sha256 = _condition_population_sha256(
        condition_ids.tolist(),
        panel.event_start_ms[indices].tolist(),
    )
    calendar_days = len(
        set(int(value) // 86_400_000 for value in panel.event_start_ms[indices])
    )
    reasons: list[str] = []
    if resolved_conditions < POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS:
        reasons.append("insufficient_resolved_conditions")
    if calendar_days < POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS:
        reasons.append("insufficient_calendar_days")
    if any(
        float(paired[control][metric]["lower_95"]) <= 0.0
        for control in _CONTROL_IDS
        for metric in ("log_loss", "brier")
    ):
        reasons.append("not_better_than_every_control")
    provisional = Round21SealedPredictiveResult(
        population_layer=population_layer,
        model_artifact_sha256=str(model_artifact["artifact_sha256"]),
        test_dataset_sha256=panel.dataset_sha256,
        test_target_manifest_sha256=panel.target_manifest_sha256,
        probability_batch_sha256=batch.prediction_sha256,
        resolved_condition_count=resolved_conditions,
        condition_population_sha256=condition_population_sha256,
        calendar_day_count=calendar_days,
        candidate_metrics=candidate_metrics,
        control_metrics=control_metrics,
        paired_improvements=paired,
        gate_passed=not reasons,
        reasons=tuple(reasons),
        result_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round21SealedEconomicResult:
    test_dataset_sha256: str
    test_target_manifest_sha256: str
    condition_count: int
    condition_population_sha256: str
    matrix_sha256: str
    ledger_count: int
    qualified_ledger_count: int
    ledger_sha256: tuple[str, ...]
    gate_passed: bool
    reasons: tuple[str, ...]
    result_sha256: str
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_SEALED_ECONOMIC_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
            "contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
            "test_dataset_sha256": self.test_dataset_sha256,
            "test_target_manifest_sha256": self.test_target_manifest_sha256,
            "condition_count": self.condition_count,
            "condition_population_sha256": self.condition_population_sha256,
            "matrix_sha256": self.matrix_sha256,
            "ledger_count": self.ledger_count,
            "qualified_ledger_count": self.qualified_ledger_count,
            "ledger_sha256": list(self.ledger_sha256),
            "gate_passed": self.gate_passed,
            "reasons": list(self.reasons),
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "result_sha256": self.result_sha256}

    def validated(self) -> Round21SealedEconomicResult:
        expected_gate = self.ledger_count == 81 and self.qualified_ledger_count == 81
        expected_reasons = () if expected_gate else ("not_all_81_ledgers_qualified",)
        if (
            any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.test_dataset_sha256,
                    self.test_target_manifest_sha256,
                    self.condition_population_sha256,
                    self.matrix_sha256,
                )
            )
            or self.condition_count < 1
            or self.ledger_count != 81
            or len(self.ledger_sha256) != 81
            or len(set(self.ledger_sha256)) != 81
            or any(_SHA256.fullmatch(value) is None for value in self.ledger_sha256)
            or not 0 <= self.qualified_ledger_count <= self.ledger_count
            or self.gate_passed != expected_gate
            or self.reasons != expected_reasons
            or type(self.gate_passed) is not bool
            or type(self.profitability_claim) is not bool
            or type(self.paper_trading_authority) is not bool
            or type(self.live_trading_authority) is not bool
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 sealed economic result differs")
        return self


def evaluate_round21_sealed_economics(
    matrix: Sequence[Round21EconomicReplay],
    *,
    test_dataset_sha256: str,
    test_target_manifest_sha256: str,
) -> Round21SealedEconomicResult:
    """Remove only the preregistered sealed-test blocker from 81 replays."""

    selected = tuple(value.validated() for value in matrix)
    expected_ledgers = {
        (profile.name, scenario.name)
        for profile in POLYMARKET_ROUND21_RISK_PROFILES
        for scenario in POLYMARKET_ROUND21_EXECUTION_SCENARIOS
    }
    if (
        len(selected) != 81
        or {(value.profile, value.scenario) for value in selected} != expected_ledgers
        or any(
            not value.qualification_reasons
            or value.qualification_reasons[-1] != "sealed_test_evidence_unavailable"
            for value in selected
        )
    ):
        raise ValueError("Round 21 sealed economic matrix differs")
    reference_population = tuple(
        (
            condition.condition_id,
            condition.event_start_ms,
            condition.outcome_sha256,
        )
        for condition in selected[0].conditions
    )
    if not reference_population or any(
        tuple(
            (
                condition.condition_id,
                condition.event_start_ms,
                condition.outcome_sha256,
            )
            for condition in replay.conditions
        )
        != reference_population
        for replay in selected[1:]
    ):
        raise ValueError("Round 21 sealed economic populations differ")
    condition_count, condition_population_sha256 = _condition_population_sha256(
        tuple(value[0] for value in reference_population),
        tuple(value[1] for value in reference_population),
    )
    qualified = sum(
        value.economic_gate_passed
        and value.qualification_reasons == ("sealed_test_evidence_unavailable",)
        for value in selected
    )
    reasons = () if qualified == 81 else ("not_all_81_ledgers_qualified",)
    provisional = Round21SealedEconomicResult(
        test_dataset_sha256=str(test_dataset_sha256 or "").strip().lower(),
        test_target_manifest_sha256=str(test_target_manifest_sha256 or "")
        .strip()
        .lower(),
        condition_count=condition_count,
        condition_population_sha256=condition_population_sha256,
        matrix_sha256=round21_replay_matrix_sha256(selected),
        ledger_count=81,
        qualified_ledger_count=qualified,
        ledger_sha256=tuple(value.replay_sha256 for value in selected),
        gate_passed=qualified == 81,
        reasons=reasons,
        result_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


@dataclass(frozen=True, slots=True)
class Round21SealedEvaluationResult:
    claim_sha256: str
    test_access_sha256: str
    selected_population_layer: str
    sealed_test_population_manifest_sha256: str
    predictive: Round21SealedPredictiveResult
    economic: Round21SealedEconomicResult
    optional_comparison_sha256: str | None
    optional_uplift_gate_passed: bool
    ai_comparison_sha256: str | None
    ai_model: str | None
    ai_model_digest: str | None
    ai_uplift_gate_passed: bool
    ai_enabled_candidate: bool
    candidate_accepted: bool
    result_sha256: str
    automatic_promotion: bool = False
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_SEALED_RESULT_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_SEALED_DESIGN_SHA256,
            "contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
            "claim_sha256": self.claim_sha256,
            "test_access_sha256": self.test_access_sha256,
            "selected_population_layer": self.selected_population_layer,
            "sealed_test_population_manifest_sha256": (
                self.sealed_test_population_manifest_sha256
            ),
            "predictive_result_sha256": self.predictive.result_sha256,
            "economic_result_sha256": self.economic.result_sha256,
            "optional_comparison_sha256": self.optional_comparison_sha256,
            "optional_uplift_gate_passed": self.optional_uplift_gate_passed,
            "ai_comparison_sha256": self.ai_comparison_sha256,
            "ai_model": self.ai_model,
            "ai_model_digest": self.ai_model_digest,
            "ai_uplift_gate_passed": self.ai_uplift_gate_passed,
            "ai_enabled_candidate": self.ai_enabled_candidate,
            "candidate_accepted": self.candidate_accepted,
            "automatic_promotion": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "result_sha256": self.result_sha256}

    def validated(self) -> Round21SealedEvaluationResult:
        predictive = self.predictive.validated()
        economic = self.economic.validated()
        optional_required = self.selected_population_layer != "core"
        ai_values = (
            self.ai_comparison_sha256,
            self.ai_model,
            self.ai_model_digest,
        )
        expected_candidate = (
            predictive.gate_passed
            and economic.gate_passed
            and (not optional_required or self.optional_uplift_gate_passed)
        )
        if (
            any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.claim_sha256,
                    self.test_access_sha256,
                    self.sealed_test_population_manifest_sha256,
                    self.result_sha256,
                )
            )
            or self.selected_population_layer != predictive.population_layer
            or economic.test_dataset_sha256 != predictive.test_dataset_sha256
            or economic.test_target_manifest_sha256
            != predictive.test_target_manifest_sha256
            or economic.condition_count != predictive.resolved_condition_count
            or economic.condition_population_sha256
            != predictive.condition_population_sha256
            or (optional_required != (self.optional_comparison_sha256 is not None))
            or (
                self.optional_comparison_sha256 is not None
                and (
                    _SHA256.fullmatch(self.optional_comparison_sha256) is None
                    or self.optional_comparison_sha256 == _EMPTY_SHA256
                )
            )
            or (not optional_required and self.optional_uplift_gate_passed)
            or type(self.optional_uplift_gate_passed) is not bool
            or (
                all(value is None for value in ai_values)
                and (self.ai_uplift_gate_passed or self.ai_enabled_candidate)
            )
            or any(value is None for value in ai_values)
            != all(value is None for value in ai_values)
            or (
                self.ai_comparison_sha256 is not None
                and (
                    _SHA256.fullmatch(self.ai_comparison_sha256) is None
                    or self.ai_comparison_sha256 == _EMPTY_SHA256
                )
            )
            or (self.ai_model is not None and not str(self.ai_model).strip())
            or (
                self.ai_model_digest is not None
                and (
                    _SHA256.fullmatch(self.ai_model_digest) is None
                    or self.ai_model_digest == _EMPTY_SHA256
                )
            )
            or type(self.ai_uplift_gate_passed) is not bool
            or type(self.ai_enabled_candidate) is not bool
            or type(self.candidate_accepted) is not bool
            or type(self.automatic_promotion) is not bool
            or type(self.profitability_claim) is not bool
            or type(self.paper_trading_authority) is not bool
            or type(self.live_trading_authority) is not bool
            or self.ai_enabled_candidate != self.ai_uplift_gate_passed
            or self.candidate_accepted != expected_candidate
            or self.automatic_promotion
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.result_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 sealed evaluation result differs")
        return self


def build_round21_sealed_evaluation_result(
    *,
    claim_sha256: str,
    test_access_sha256: str,
    selected_population_layer: str,
    sealed_test_population_manifest_sha256: str,
    predictive: Round21SealedPredictiveResult,
    economic: Round21SealedEconomicResult,
    optional_comparison: Round21MatchedEconomicComparison | None = None,
    ai_comparison: Round21AIMatchedComparison | None = None,
) -> Round21SealedEvaluationResult:
    """Bind predictive, economic, optional, and AI ablations without promotion."""

    selected_predictive = predictive.validated()
    selected_economic = economic.validated()
    optional = None if optional_comparison is None else optional_comparison.validated()
    ai = None if ai_comparison is None else ai_comparison.validated()
    layer = str(selected_population_layer or "").strip()
    if layer not in _LAYERS:
        raise ValueError("Round 21 sealed selected layer differs")
    if (
        selected_economic.test_dataset_sha256 != selected_predictive.test_dataset_sha256
        or selected_economic.test_target_manifest_sha256
        != selected_predictive.test_target_manifest_sha256
        or selected_economic.condition_count
        != selected_predictive.resolved_condition_count
        or selected_economic.condition_population_sha256
        != selected_predictive.condition_population_sha256
    ):
        raise ValueError("Round 21 sealed predictive and economic populations differ")
    if (layer == "core") != (optional is None):
        raise ValueError("Round 21 sealed optional comparison differs")
    if optional is not None and (
        optional.challenger_layer != layer
        or optional.challenger_matrix_sha256 != selected_economic.matrix_sha256
    ):
        raise ValueError("Round 21 sealed optional comparison differs")
    if ai is not None and ai.baseline_matrix_sha256 != selected_economic.matrix_sha256:
        raise ValueError("Round 21 sealed AI comparison differs")
    optional_passed = optional is not None and optional.all_replays_accepted
    ai_passed = ai is not None and ai.development_qualified
    accepted = (
        selected_predictive.gate_passed
        and selected_economic.gate_passed
        and (layer == "core" or optional_passed)
    )
    provisional = Round21SealedEvaluationResult(
        claim_sha256=str(claim_sha256).strip().lower(),
        test_access_sha256=str(test_access_sha256).strip().lower(),
        selected_population_layer=layer,
        sealed_test_population_manifest_sha256=str(
            sealed_test_population_manifest_sha256
        )
        .strip()
        .lower(),
        predictive=selected_predictive,
        economic=selected_economic,
        optional_comparison_sha256=(
            None if optional is None else optional.comparison_sha256
        ),
        optional_uplift_gate_passed=optional_passed,
        ai_comparison_sha256=None if ai is None else ai.comparison_sha256,
        ai_model=None if ai is None else ai.model,
        ai_model_digest=None if ai is None else ai.model_digest,
        ai_uplift_gate_passed=ai_passed,
        ai_enabled_candidate=ai_passed,
        candidate_accepted=accepted,
        result_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def build_round21_sealed_result_bundle(
    result: Round21SealedEvaluationResult,
) -> dict[str, object]:
    """Serialize a restart-safe sealed result without granting promotion."""

    selected = result.validated()
    body = {
        "schema_version": POLYMARKET_ROUND21_SEALED_BUNDLE_SCHEMA_VERSION,
        "result": selected.asdict(),
        "predictive": selected.predictive.asdict(),
        "economic": selected.economic.asdict(),
        "automatic_promotion": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    return {**body, "bundle_sha256": _canonical_sha256(body)}


def _bundle_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Round 21 sealed bundle {name} is not an object")
    return value


def validate_round21_sealed_result_bundle(
    value: Mapping[str, object],
) -> Round21SealedEvaluationResult:
    """Reconstruct and verify every nested sealed verdict from one bundle."""

    payload = dict(value)
    claimed = str(payload.pop("bundle_sha256", "")).strip().lower()
    if (
        set(payload)
        != {
            "schema_version",
            "result",
            "predictive",
            "economic",
            "automatic_promotion",
            "paper_trading_authority",
            "live_trading_authority",
        }
        or payload.get("schema_version")
        != POLYMARKET_ROUND21_SEALED_BUNDLE_SCHEMA_VERSION
        or payload.get("automatic_promotion") is not False
        or payload.get("paper_trading_authority") is not False
        or payload.get("live_trading_authority") is not False
        or _SHA256.fullmatch(claimed) is None
        or claimed != _canonical_sha256(payload)
    ):
        raise ValueError("Round 21 sealed result bundle differs")
    predictive_payload = _bundle_mapping(
        payload["predictive"],
        name="predictive result",
    )
    economic_payload = _bundle_mapping(
        payload["economic"],
        name="economic result",
    )
    result_payload = _bundle_mapping(payload["result"], name="result")
    try:
        predictive = Round21SealedPredictiveResult(
            population_layer=str(predictive_payload["population_layer"]),
            model_artifact_sha256=str(predictive_payload["model_artifact_sha256"]),
            test_dataset_sha256=str(predictive_payload["test_dataset_sha256"]),
            test_target_manifest_sha256=str(
                predictive_payload["test_target_manifest_sha256"]
            ),
            probability_batch_sha256=str(
                predictive_payload["probability_batch_sha256"]
            ),
            resolved_condition_count=int(
                predictive_payload["resolved_condition_count"]
            ),
            condition_population_sha256=str(
                predictive_payload["condition_population_sha256"]
            ),
            calendar_day_count=int(predictive_payload["calendar_day_count"]),
            candidate_metrics=_bundle_mapping(
                predictive_payload["candidate_metrics"],
                name="candidate metrics",
            ),
            control_metrics=_bundle_mapping(
                predictive_payload["control_metrics"],
                name="control metrics",
            ),
            paired_improvements=_bundle_mapping(
                predictive_payload["paired_improvements"],
                name="paired improvements",
            ),
            gate_passed=predictive_payload["gate_passed"],  # type: ignore[arg-type]
            reasons=tuple(predictive_payload["reasons"]),  # type: ignore[arg-type]
            result_sha256=str(predictive_payload["result_sha256"]),
            profitability_claim=predictive_payload["profitability_claim"],  # type: ignore[arg-type]
            paper_trading_authority=predictive_payload["paper_trading_authority"],  # type: ignore[arg-type]
            live_trading_authority=predictive_payload["live_trading_authority"],  # type: ignore[arg-type]
        ).validated()
        economic = Round21SealedEconomicResult(
            test_dataset_sha256=str(economic_payload["test_dataset_sha256"]),
            test_target_manifest_sha256=str(
                economic_payload["test_target_manifest_sha256"]
            ),
            condition_count=int(economic_payload["condition_count"]),
            condition_population_sha256=str(
                economic_payload["condition_population_sha256"]
            ),
            matrix_sha256=str(economic_payload["matrix_sha256"]),
            ledger_count=int(economic_payload["ledger_count"]),
            qualified_ledger_count=int(economic_payload["qualified_ledger_count"]),
            ledger_sha256=tuple(economic_payload["ledger_sha256"]),  # type: ignore[arg-type]
            gate_passed=economic_payload["gate_passed"],  # type: ignore[arg-type]
            reasons=tuple(economic_payload["reasons"]),  # type: ignore[arg-type]
            result_sha256=str(economic_payload["result_sha256"]),
            profitability_claim=economic_payload["profitability_claim"],  # type: ignore[arg-type]
            paper_trading_authority=economic_payload["paper_trading_authority"],  # type: ignore[arg-type]
            live_trading_authority=economic_payload["live_trading_authority"],  # type: ignore[arg-type]
        ).validated()
        selected = Round21SealedEvaluationResult(
            claim_sha256=str(result_payload["claim_sha256"]),
            test_access_sha256=str(result_payload["test_access_sha256"]),
            selected_population_layer=str(result_payload["selected_population_layer"]),
            sealed_test_population_manifest_sha256=str(
                result_payload["sealed_test_population_manifest_sha256"]
            ),
            predictive=predictive,
            economic=economic,
            optional_comparison_sha256=result_payload["optional_comparison_sha256"],  # type: ignore[arg-type]
            optional_uplift_gate_passed=result_payload["optional_uplift_gate_passed"],  # type: ignore[arg-type]
            ai_comparison_sha256=result_payload["ai_comparison_sha256"],  # type: ignore[arg-type]
            ai_model=result_payload["ai_model"],  # type: ignore[arg-type]
            ai_model_digest=result_payload["ai_model_digest"],  # type: ignore[arg-type]
            ai_uplift_gate_passed=result_payload["ai_uplift_gate_passed"],  # type: ignore[arg-type]
            ai_enabled_candidate=result_payload["ai_enabled_candidate"],  # type: ignore[arg-type]
            candidate_accepted=result_payload["candidate_accepted"],  # type: ignore[arg-type]
            result_sha256=str(result_payload["result_sha256"]),
            automatic_promotion=result_payload["automatic_promotion"],  # type: ignore[arg-type]
            profitability_claim=result_payload["profitability_claim"],  # type: ignore[arg-type]
            paper_trading_authority=result_payload["paper_trading_authority"],  # type: ignore[arg-type]
            live_trading_authority=result_payload["live_trading_authority"],  # type: ignore[arg-type]
        ).validated()
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 21 sealed result bundle differs") from exc
    if (
        dict(predictive_payload) != predictive.asdict()
        or dict(economic_payload) != economic.asdict()
        or dict(result_payload) != selected.asdict()
        or selected.predictive.result_sha256
        != str(result_payload.get("predictive_result_sha256") or "")
        or selected.economic.result_sha256
        != str(result_payload.get("economic_result_sha256") or "")
    ):
        raise ValueError("Round 21 sealed result bundle differs")
    return selected


def load_round21_sealed_result_bundle(
    path: str | Path,
) -> Round21SealedEvaluationResult:
    """Load one bounded, strict-JSON, restart-safe sealed result bundle."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("Round 21 sealed result bundle is unavailable")
    selected = source.resolve()
    if not selected.is_file():
        raise ValueError("Round 21 sealed result bundle is unavailable")
    try:
        with selected.open("rb") as handle:
            raw = handle.read(4 * 1024 * 1024 + 1)
        if not 2 <= len(raw) <= 4 * 1024 * 1024:
            raise ValueError("Round 21 sealed result bundle is unavailable")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 sealed result bundle is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 sealed result bundle is invalid")
    return validate_round21_sealed_result_bundle(value)


@dataclass(frozen=True, slots=True)
class VerifiedRound21SealedResultBundle:
    """File- and content-verified Round 21 sealed-result capability."""

    result: Round21SealedEvaluationResult
    path: Path
    file_sha256: str
    _capability: object

    def __post_init__(self) -> None:
        if self._capability is not _VERIFIED_BUNDLE_CAPABILITY:
            raise ValueError("Round 21 sealed result bundle was not verified")


def load_verified_round21_sealed_result_bundle(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> VerifiedRound21SealedResultBundle:
    """Bind exact bundle bytes to every reconstructed nested verdict."""

    expected = str(expected_file_sha256 or "").strip().lower()
    if _SHA256.fullmatch(expected) is None:
        raise ValueError("Round 21 sealed bundle evidence hash is invalid")
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("Round 21 sealed result bundle is unavailable")
    selected = source.resolve()
    try:
        with selected.open("rb") as handle:
            raw = handle.read(4 * 1024 * 1024 + 1)
    except OSError as exc:
        raise ValueError("Round 21 sealed result bundle is unavailable") from exc
    if not 2 <= len(raw) <= 4 * 1024 * 1024:
        raise ValueError("Round 21 sealed result bundle is unavailable")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError("Round 21 sealed bundle evidence hash differs")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 sealed result bundle is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 sealed result bundle is invalid")
    result = validate_round21_sealed_result_bundle(value)
    return VerifiedRound21SealedResultBundle(
        result=result,
        path=selected,
        file_sha256=actual,
        _capability=_VERIFIED_BUNDLE_CAPABILITY,
    )


credentials_used = False
account_connected = False
binance_execution_connected = False
automatic_promotion = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS",
    "POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS",
    "POLYMARKET_ROUND21_SEALED_BUNDLE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SEALED_DESIGN_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SEALED_DESIGN_SHA256",
    "POLYMARKET_ROUND21_SEALED_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SEALED_PREDICTIVE_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_SEALED_RESULT_SCHEMA_VERSION",
    "Round21SealedEconomicResult",
    "Round21SealedEvaluationResult",
    "Round21SealedPredictiveResult",
    "VerifiedRound21SealedResultBundle",
    "build_round21_sealed_result_bundle",
    "build_round21_sealed_evaluation_result",
    "evaluate_round21_sealed_economics",
    "evaluate_round21_sealed_predictions",
    "load_round21_sealed_design",
    "load_round21_sealed_result_bundle",
    "load_verified_round21_sealed_result_bundle",
    "validate_round21_sealed_result_bundle",
    "validate_round21_sealed_design",
]
