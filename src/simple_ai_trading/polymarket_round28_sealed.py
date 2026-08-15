"""One-use sealed evaluation for the frozen Round 28 matched BBO ablation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np

from . import polymarket_round28_selection as _selection
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_economics import (
    POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
    POLYMARKET_ROUND27_FIXED_DELAYS_MS,
    Round27EconomicBookBatch,
    Round27EconomicConfig,
    evaluate_round27_economic_scenarios,
)
from .polymarket_round28_economics import (
    POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION,
    paired_round28_economic_scenario,
    project_round28_economic_partition,
)
from .polymarket_round28_model import Round28ModelSample, Round28Partition
from .polymarket_round28_operator import (
    validate_round28_economic_report,
    validate_round28_selection_input_manifest,
)
from .polymarket_round28_selection import (
    Round28SelectedPair,
    load_round28_selected_pair,
    round28_pair_selection_report,
)


POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION = (
    "polymarket-round28-sealed-prediction-v1"
)
POLYMARKET_ROUND28_SEALED_ECONOMIC_SCHEMA_VERSION = (
    "polymarket-round28-sealed-economic-v1"
)
POLYMARKET_ROUND28_SEALED_TERMINAL_SCHEMA_VERSION = (
    "polymarket-round28-terminal-sealed-result-v1"
)
_AUTHORITY = {
    "edge_claim": False,
    "profitability_claim": False,
    "paper_trading_authority": False,
    "live_trading_authority": False,
    "orders_submitted": False,
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 28 sealed {name} SHA-256 differs")
    return selected


def _validated_claim(
    value: Mapping[str, object],
    *,
    hash_field: str,
    name: str,
) -> dict[str, object]:
    payload = dict(value)
    claimed = _sha256(payload.pop(hash_field, None), name=name)
    if claimed != _canonical_sha256(payload):
        raise ValueError(f"Round 28 sealed {name} differs")
    return {**payload, hash_field: claimed}


def _role_minimum(contract: Mapping[str, object], role: str) -> int:
    minimum = contract.get("minimum_population")
    if not isinstance(minimum, Mapping):
        raise ValueError("Round 28 sealed minimum-population contract differs")
    value = minimum.get(f"{role}_conditions")
    if type(value) is not int or int(value) <= 0:
        raise ValueError("Round 28 sealed role minimum differs")
    return int(value)


def _condition_ids(partition: Round28Partition) -> tuple[str, ...]:
    starts = {
        sample.condition_id: sample.event_start_ms for sample in partition.samples
    }
    if len(starts) != len(set(str(value) for value in partition.conditions)):
        raise ValueError("Round 28 sealed condition identity differs")
    return tuple(
        condition_id
        for condition_id, _start in sorted(
            starts.items(),
            key=lambda item: (item[1], item[0]),
        )
    )


def validate_round28_sealed_access_artifacts(
    *,
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    selection_economic_report: Mapping[str, object],
    selection_resolution_evidence_sha256: str,
) -> Round28SelectedPair:
    """Require frozen passing selection evidence before sealed evaluation."""

    manifest = validate_round28_selection_input_manifest(selection_input_manifest)
    pair = load_round28_selected_pair(
        selection_claim,
        contract=contract,
        preregistration=preregistration,
    )
    if pair is None:
        raise ValueError("Round 28 sealed access has no selected model pair")
    report = validate_round28_economic_report(
        selection_economic_report,
        input_manifest=manifest,
        selection_claim=selection_claim,
        resolution_evidence_sha256=selection_resolution_evidence_sha256,
    )
    if (
        report.get("economic_uplift_gate_passed") is not True
        or report.get("selected_model_family") != pair.model_family
        or report.get("base_model_sha256") != pair.base_model.model_sha256
        or report.get("augmented_model_sha256")
        != pair.augmented_model.model_sha256
        or report.get("sealed_partition_accessed") is not False
    ):
        raise ValueError("Round 28 sealed access selection gate differs")
    return pair


def evaluate_round28_sealed_prediction(
    *,
    samples: Sequence[Round28ModelSample],
    contract: Mapping[str, object],
    preregistration: Mapping[str, object],
    selection_input_manifest: Mapping[str, object],
    selection_claim: Mapping[str, object],
    selection_economic_report: Mapping[str, object],
    selection_resolution_evidence_sha256: str,
    source_binding_sha256: str,
) -> dict[str, object]:
    """Evaluate the exact selected pair on sealed labels without refitting."""

    pair = validate_round28_sealed_access_artifacts(
        contract=contract,
        preregistration=preregistration,
        selection_input_manifest=selection_input_manifest,
        selection_claim=selection_claim,
        selection_economic_report=selection_economic_report,
        selection_resolution_evidence_sha256=(
            selection_resolution_evidence_sha256
        ),
    )
    partition = Round28Partition.from_samples(samples, role="sealed")
    condition_ids = _condition_ids(partition)
    if len(condition_ids) < _role_minimum(contract, "sealed"):
        raise ValueError("Round 28 sealed prediction population is insufficient")
    evaluation = contract.get("prediction_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Round 28 sealed prediction contract differs")
    pair_report = round28_pair_selection_report(
        partition,
        base_model=pair.base_model,
        augmented_model=pair.augmented_model,
        prediction_evaluation=evaluation,
        training_detail={
            "frozen_selection_claim_sha256": _sha256(
                selection_claim.get("claim_sha256"),
                name="selection claim",
            ),
            "models_refit": False,
            "hyperparameters_retuned": False,
            "thresholds_changed": False,
        },
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION,
        "round27_model_contract_sha256": _sha256(
            contract.get("contract_sha256"),
            name="model contract",
        ),
        "round28_preregistration_sha256": _sha256(
            preregistration.get("preregistration_sha256"),
            name="preregistration",
        ),
        "selection_input_manifest_sha256": _sha256(
            selection_input_manifest.get("manifest_sha256"),
            name="selection input manifest",
        ),
        "selection_claim_sha256": _sha256(
            selection_claim.get("claim_sha256"),
            name="selection claim",
        ),
        "selection_economic_report_sha256": _sha256(
            selection_economic_report.get("report_sha256"),
            name="selection economic report",
        ),
        "source_binding_sha256": _sha256(
            source_binding_sha256,
            name="source binding",
        ),
        "selected_model_family": pair.model_family,
        "base_model_sha256": pair.base_model.model_sha256,
        "augmented_model_sha256": pair.augmented_model.model_sha256,
        "condition_count": len(condition_ids),
        "row_count": len(partition.samples),
        "condition_population_sha256": _canonical_sha256(list(condition_ids)),
        "paired_probability_report": pair_report,
        "prediction_uplift_gate_passed": bool(
            pair_report["probability_gate_passed"]
        ),
        "sealed_partition_accessed": True,
        "models_refit": False,
        "hyperparameters_retuned": False,
        "thresholds_changed": False,
        "economic_metrics_computed": False,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return body


def _scenario_map(report: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Round 28 sealed economic scenarios differ")
    selected: dict[int, Mapping[str, object]] = {}
    for item in scenarios:
        if not isinstance(item, Mapping) or type(item.get("delay_ms")) is not int:
            raise ValueError("Round 28 sealed economic scenario differs")
        delay = int(item["delay_ms"])
        if delay in selected:
            raise ValueError("Round 28 sealed economic delay is duplicated")
        selected[delay] = item
    return selected


def evaluate_round28_sealed_economics(
    *,
    samples: Sequence[Round28ModelSample],
    pair: Round28SelectedPair,
    selection_claim_sha256: str,
    sealed_prediction_result: Mapping[str, object],
    markets: Sequence[PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    source_binding_sha256: str,
    resolution_evidence_sha256: str,
    config: Round27EconomicConfig,
    books: Sequence[PolymarketRecordedBook] | None = None,
    book_batch_factory: Callable[[], Iterable[Round27EconomicBookBatch]] | None = None,
) -> dict[str, object]:
    """Run identical sealed execution scenarios for the selected model pair."""

    prediction = validate_round28_sealed_prediction_result(
        sealed_prediction_result
    )
    if (books is None) == (book_batch_factory is None):
        raise ValueError("Round 28 sealed economics requires one book source")
    partition = Round28Partition.from_samples(samples, role="sealed")
    projected = project_round28_economic_partition(partition)
    condition_ids = _condition_ids(partition)
    if set(outcomes_up) != set(condition_ids):
        raise ValueError("Round 28 sealed economic outcomes differ")
    market_ids = {market.condition_id for market in markets}
    if market_ids != set(condition_ids) or len(markets) != len(market_ids):
        raise ValueError("Round 28 sealed economic markets differ")
    cfg = config.validated()
    base_probability = pair.base_model.predict(
        partition.base_features,
        partition.offsets,
    )
    augmented_probability = pair.augmented_model.predict(
        partition.augmented_features,
        partition.offsets,
    )

    def evaluate(model: object, probability: np.ndarray) -> dict[str, object]:
        kwargs: dict[str, object] = (
            {"books": books}
            if books is not None
            else {"book_batches": book_batch_factory()}
        )
        return evaluate_round27_economic_scenarios(
            partition=projected,
            predictions=probability,
            markets=markets,
            outcomes_up=outcomes_up,
            model_name=f"{model.model_name}:{model.feature_view}",
            model_sha256=model.model_sha256,
            source_audit_sha256=_sha256(
                source_binding_sha256,
                name="source binding",
            ),
            resolution_evidence_sha256=_sha256(
                resolution_evidence_sha256,
                name="resolution evidence",
            ),
            config=cfg,
            **kwargs,
        )

    base_report = evaluate(pair.base_model, base_probability)
    augmented_report = evaluate(pair.augmented_model, augmented_probability)
    base_scenarios = _scenario_map(base_report)
    augmented_scenarios = _scenario_map(augmented_report)
    if set(base_scenarios) != set(cfg.delays_ms) or set(augmented_scenarios) != set(
        cfg.delays_ms
    ):
        raise ValueError("Round 28 sealed economic delay population differs")
    paired = [
        paired_round28_economic_scenario(
            base=base_scenarios[delay],
            augmented=augmented_scenarios[delay],
            ordered_conditions=condition_ids,
            bootstrap_draws=cfg.bootstrap_draws,
            bootstrap_seed=28_028,
        )
        for delay in cfg.delays_ms
    ]
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SEALED_ECONOMIC_SCHEMA_VERSION,
        "partition_role": "sealed",
        "selection_claim_sha256": _sha256(
            selection_claim_sha256,
            name="selection claim",
        ),
        "sealed_prediction_result_sha256": prediction["result_sha256"],
        "source_binding_sha256": _sha256(
            source_binding_sha256,
            name="source binding",
        ),
        "resolution_evidence_sha256": _sha256(
            resolution_evidence_sha256,
            name="resolution evidence",
        ),
        "selected_model_family": pair.model_family,
        "base_model_sha256": pair.base_model.model_sha256,
        "augmented_model_sha256": pair.augmented_model.model_sha256,
        "condition_population_sha256": _canonical_sha256(list(condition_ids)),
        "base_economic_report": base_report,
        "augmented_economic_report": augmented_report,
        "paired_scenarios": paired,
        "economic_uplift_gate_passed": bool(
            augmented_report["economic_edge_gate_passed"]
        )
        and all(item["scenario_uplift_gate_passed"] for item in paired),
        "sealed_partition_accessed": True,
        "models_refit": False,
        "hyperparameters_retuned": False,
        "thresholds_changed": False,
        "economic_metrics_computed": True,
        "ai_assist_evaluated": False,
        **_AUTHORITY,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_sealed_prediction_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    result = _validated_claim(
        value,
        hash_field="result_sha256",
        name="prediction result",
    )
    report = result.get("paired_probability_report")
    if not isinstance(report, Mapping):
        raise ValueError("Round 28 sealed paired probability report differs")
    validated_report = _selection._validated_pair_report(report)  # noqa: SLF001
    checks = validated_report.get("gate_checks")
    matched = validated_report.get("matched_ablation")
    base_model = matched.get("base_model") if isinstance(matched, Mapping) else None
    augmented_model = (
        matched.get("augmented_model") if isinstance(matched, Mapping) else None
    )
    expected_fields = {
        "schema_version",
        "round27_model_contract_sha256",
        "round28_preregistration_sha256",
        "selection_input_manifest_sha256",
        "selection_claim_sha256",
        "selection_economic_report_sha256",
        "source_binding_sha256",
        "selected_model_family",
        "base_model_sha256",
        "augmented_model_sha256",
        "condition_count",
        "row_count",
        "condition_population_sha256",
        "paired_probability_report",
        "prediction_uplift_gate_passed",
        "sealed_partition_accessed",
        "models_refit",
        "hyperparameters_retuned",
        "thresholds_changed",
        "economic_metrics_computed",
        "ai_assist_evaluated",
        *_AUTHORITY,
        "result_sha256",
    }
    if (
        set(result) != expected_fields
        or result.get("schema_version")
        != POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION
        or not isinstance(matched, Mapping)
        or matched.get("role") != "sealed"
        or not isinstance(base_model, Mapping)
        or not isinstance(augmented_model, Mapping)
        or validated_report.get("model_family")
        != result.get("selected_model_family")
        or base_model.get("model_sha256") != result.get("base_model_sha256")
        or augmented_model.get("model_sha256")
        != result.get("augmented_model_sha256")
        or matched.get("condition_count") != result.get("condition_count")
        or matched.get("row_count") != result.get("row_count")
        or validated_report.get("probability_gate_passed")
        is not bool(isinstance(checks, Mapping) and checks and all(checks.values()))
        or result.get("prediction_uplift_gate_passed")
        is not bool(validated_report["probability_gate_passed"])
        or result.get("sealed_partition_accessed") is not True
        or result.get("models_refit") is not False
        or result.get("hyperparameters_retuned") is not False
        or result.get("thresholds_changed") is not False
        or result.get("economic_metrics_computed") is not False
        or result.get("ai_assist_evaluated") is not False
        or type(result.get("condition_count")) is not int
        or int(result["condition_count"]) <= 0
        or type(result.get("row_count")) is not int
        or int(result["row_count"]) < int(result["condition_count"])
        or any(result.get(key) is not expected for key, expected in _AUTHORITY.items())
    ):
        raise ValueError("Round 28 sealed prediction result differs")
    for field in (
        "round27_model_contract_sha256",
        "round28_preregistration_sha256",
        "selection_input_manifest_sha256",
        "selection_claim_sha256",
        "selection_economic_report_sha256",
        "source_binding_sha256",
        "base_model_sha256",
        "augmented_model_sha256",
        "condition_population_sha256",
    ):
        _sha256(result.get(field), name=field)
    return result


def _validate_round27_economic_report(
    value: object,
    *,
    source_binding_sha256: str,
    resolution_evidence_sha256: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Round 28 sealed nested economic report differs")
    report = _validated_claim(
        value,
        hash_field="report_sha256",
        name="nested economic report",
    )
    scenarios = report.get("scenarios")
    if (
        report.get("schema_version")
        != POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION
        or report.get("partition_role") != "sealed"
        or report.get("source_audit_sha256") != source_binding_sha256
        or report.get("resolution_evidence_sha256")
        != resolution_evidence_sha256
        or not isinstance(scenarios, list)
        or len(scenarios) != len(POLYMARKET_ROUND27_FIXED_DELAYS_MS)
    ):
        raise ValueError("Round 28 sealed nested scenarios differ")
    scenario_passes: list[bool] = []
    for item, delay in zip(
        scenarios,
        POLYMARKET_ROUND27_FIXED_DELAYS_MS,
        strict=True,
    ):
        if not isinstance(item, Mapping):
            raise ValueError("Round 28 sealed nested scenario differs")
        selected = _validated_claim(
            item,
            hash_field="scenario_sha256",
            name="nested economic scenario",
        )
        checks = selected.get("gate_checks")
        passed = bool(isinstance(checks, Mapping) and checks and all(checks.values()))
        if (
            selected.get("delay_ms") != delay
            or selected.get("scenario_edge_gate_passed") is not passed
        ):
            raise ValueError("Round 28 sealed nested scenario gate differs")
        scenario_passes.append(passed)
    if (
        report.get("economic_edge_gate_passed") is not all(scenario_passes)
        or any(
            report.get(field) is not False
            for field in (
                "edge_claim",
                "profitability_claim",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 sealed nested economic gate differs")
    return report


def validate_round28_sealed_economic_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    report = _validated_claim(
        value,
        hash_field="report_sha256",
        name="economic report",
    )
    source_binding = _sha256(
        report.get("source_binding_sha256"),
        name="source binding",
    )
    resolution_evidence = _sha256(
        report.get("resolution_evidence_sha256"),
        name="resolution evidence",
    )
    base = _validate_round27_economic_report(
        report.get("base_economic_report"),
        source_binding_sha256=source_binding,
        resolution_evidence_sha256=resolution_evidence,
    )
    augmented = _validate_round27_economic_report(
        report.get("augmented_economic_report"),
        source_binding_sha256=source_binding,
        resolution_evidence_sha256=resolution_evidence,
    )
    paired = report.get("paired_scenarios")
    expected_fields = {
        "schema_version",
        "partition_role",
        "selection_claim_sha256",
        "sealed_prediction_result_sha256",
        "source_binding_sha256",
        "resolution_evidence_sha256",
        "selected_model_family",
        "base_model_sha256",
        "augmented_model_sha256",
        "condition_population_sha256",
        "base_economic_report",
        "augmented_economic_report",
        "paired_scenarios",
        "economic_uplift_gate_passed",
        "sealed_partition_accessed",
        "models_refit",
        "hyperparameters_retuned",
        "thresholds_changed",
        "economic_metrics_computed",
        "ai_assist_evaluated",
        *_AUTHORITY,
        "report_sha256",
    }
    if (
        set(report) != expected_fields
        or not isinstance(paired, list)
        or len(paired) != len(base["scenarios"])
        or base.get("model_sha256") != report.get("base_model_sha256")
        or augmented.get("model_sha256")
        != report.get("augmented_model_sha256")
        or base.get("config") != augmented.get("config")
    ):
        raise ValueError("Round 28 sealed paired scenarios differ")
    paired_pass = True
    base_scenarios = _scenario_map(base)
    augmented_scenarios = _scenario_map(augmented)
    for item, delay in zip(
        paired,
        POLYMARKET_ROUND27_FIXED_DELAYS_MS,
        strict=True,
    ):
        if not isinstance(item, Mapping):
            raise ValueError("Round 28 sealed paired scenario differs")
        selected = _validated_claim(
            item,
            hash_field="paired_scenario_sha256",
            name="paired economic scenario",
        )
        checks = selected.get("gate_checks")
        passed = bool(isinstance(checks, Mapping) and checks and all(checks.values()))
        if (
            selected.get("schema_version")
            != POLYMARKET_ROUND28_PAIRED_SCENARIO_SCHEMA_VERSION
            or selected.get("delay_ms") != delay
            or selected.get("base_scenario_sha256")
            != base_scenarios[delay]["scenario_sha256"]
            or selected.get("augmented_scenario_sha256")
            != augmented_scenarios[delay]["scenario_sha256"]
            or selected.get("scenario_uplift_gate_passed") is not passed
        ):
            raise ValueError("Round 28 sealed paired scenario gate differs")
        paired_pass = paired_pass and passed
    expected_gate = bool(augmented["economic_edge_gate_passed"]) and paired_pass
    if (
        report.get("schema_version")
        != POLYMARKET_ROUND28_SEALED_ECONOMIC_SCHEMA_VERSION
        or report.get("partition_role") != "sealed"
        or report.get("economic_uplift_gate_passed") is not expected_gate
        or report.get("sealed_partition_accessed") is not True
        or report.get("models_refit") is not False
        or report.get("hyperparameters_retuned") is not False
        or report.get("thresholds_changed") is not False
        or report.get("economic_metrics_computed") is not True
        or report.get("ai_assist_evaluated") is not False
        or any(report.get(key) is not expected for key, expected in _AUTHORITY.items())
    ):
        raise ValueError("Round 28 sealed economic report differs")
    for field in (
        "selection_claim_sha256",
        "sealed_prediction_result_sha256",
        "source_binding_sha256",
        "resolution_evidence_sha256",
        "base_model_sha256",
        "augmented_model_sha256",
        "condition_population_sha256",
    ):
        _sha256(report.get(field), name=field)
    return report


def build_round28_sealed_terminal_result(
    *,
    sealed_prediction_result: Mapping[str, object],
    sealed_economic_report: Mapping[str, object],
) -> dict[str, object]:
    prediction = validate_round28_sealed_prediction_result(
        sealed_prediction_result
    )
    economics = validate_round28_sealed_economic_report(sealed_economic_report)
    if (
        prediction["selection_claim_sha256"] != economics["selection_claim_sha256"]
        or prediction["source_binding_sha256"] != economics["source_binding_sha256"]
        or prediction["base_model_sha256"] != economics["base_model_sha256"]
        or prediction["augmented_model_sha256"]
        != economics["augmented_model_sha256"]
        or prediction["condition_population_sha256"]
        != economics["condition_population_sha256"]
        or prediction["result_sha256"]
        != economics["sealed_prediction_result_sha256"]
    ):
        raise ValueError("Round 28 sealed terminal artifact binding differs")
    passed = bool(
        prediction["prediction_uplift_gate_passed"]
        and economics["economic_uplift_gate_passed"]
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND28_SEALED_TERMINAL_SCHEMA_VERSION,
        "selection_claim_sha256": prediction["selection_claim_sha256"],
        "sealed_prediction_result_sha256": prediction["result_sha256"],
        "sealed_economic_report_sha256": economics["report_sha256"],
        "selected_model_family": prediction["selected_model_family"],
        "prediction_uplift_gate_passed": prediction[
            "prediction_uplift_gate_passed"
        ],
        "economic_uplift_gate_passed": economics["economic_uplift_gate_passed"],
        "observed_after_cost_bbo_uplift_gate_passed": passed,
        "sealed_partition_accessed": True,
        "models_refit": False,
        "hyperparameters_retuned": False,
        "thresholds_changed": False,
        "ai_assist_evaluated": False,
        "edge_claim": False,
        "profitability_claim": False,
        "credentials_used": False,
        "orders_submitted": False,
        "trading_authority": False,
    }
    body["result_sha256"] = _canonical_sha256(body)
    return body


def validate_round28_sealed_terminal_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    result = _validated_claim(
        value,
        hash_field="result_sha256",
        name="terminal result",
    )
    passed = bool(
        result.get("prediction_uplift_gate_passed")
        and result.get("economic_uplift_gate_passed")
    )
    expected_fields = {
        "schema_version",
        "selection_claim_sha256",
        "sealed_prediction_result_sha256",
        "sealed_economic_report_sha256",
        "selected_model_family",
        "prediction_uplift_gate_passed",
        "economic_uplift_gate_passed",
        "observed_after_cost_bbo_uplift_gate_passed",
        "sealed_partition_accessed",
        "models_refit",
        "hyperparameters_retuned",
        "thresholds_changed",
        "ai_assist_evaluated",
        "edge_claim",
        "profitability_claim",
        "credentials_used",
        "orders_submitted",
        "trading_authority",
        "result_sha256",
    }
    if (
        set(result) != expected_fields
        or result.get("schema_version")
        != POLYMARKET_ROUND28_SEALED_TERMINAL_SCHEMA_VERSION
        or result.get("observed_after_cost_bbo_uplift_gate_passed") is not passed
        or result.get("sealed_partition_accessed") is not True
        or result.get("models_refit") is not False
        or result.get("hyperparameters_retuned") is not False
        or result.get("thresholds_changed") is not False
        or result.get("ai_assist_evaluated") is not False
        or any(
            result.get(field) is not False
            for field in (
                "edge_claim",
                "profitability_claim",
                "credentials_used",
                "orders_submitted",
                "trading_authority",
            )
        )
    ):
        raise ValueError("Round 28 sealed terminal result differs")
    for field in (
        "selection_claim_sha256",
        "sealed_prediction_result_sha256",
        "sealed_economic_report_sha256",
    ):
        _sha256(result.get(field), name=field)
    return result


__all__ = [
    "POLYMARKET_ROUND28_SEALED_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_SEALED_PREDICTION_SCHEMA_VERSION",
    "POLYMARKET_ROUND28_SEALED_TERMINAL_SCHEMA_VERSION",
    "build_round28_sealed_terminal_result",
    "evaluate_round28_sealed_economics",
    "evaluate_round28_sealed_prediction",
    "validate_round28_sealed_access_artifacts",
    "validate_round28_sealed_economic_report",
    "validate_round28_sealed_prediction_result",
    "validate_round28_sealed_terminal_result",
]
