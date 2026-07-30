"""Terminal, test-blind Round 17 Polymarket development orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from .polymarket import PolymarketPublicClient
from .polymarket_round14_contract import load_round14_contract
from .polymarket_round17_campaign_operator import (
    Round17CampaignDevelopmentIndex,
    Round17CampaignOperatorConfig,
    iter_round17_campaign_development_conditions,
    materialize_round17_campaign_development_index,
)
from .polymarket_round17_cohort import (
    Round17DevelopmentPanelAssembler,
    Round17DevelopmentTargetManifest,
    build_round17_development_target_manifest,
    load_round17_cohort_plan,
)
from .polymarket_round17_economic import (
    POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256,
    fit_round17_economic_pretest,
    validate_round17_economic_pretest,
)
from .polymarket_round17_features import POLYMARKET_ROUND17_CONTRACT_SHA256
from .polymarket_round17_model import (
    fit_round17_development_pretest,
    validate_round17_pretest_artifact,
)
from .polymarket_round17_outcomes import (
    build_round17_calibrated_decision_probability,
    materialize_round17_condition_economic_outcomes,
)
from .polymarket_round17_resolution import (
    Round17DevelopmentResolutionAcquisition,
    acquire_round17_development_resolutions,
)
from .polymarket_round17_uncertainty import (
    apply_round17_probability_calibration_rows,
    fit_round17_probability_calibration,
    validate_round17_probability_calibration,
)
from .storage import write_json_atomic


POLYMARKET_ROUND17_DEVELOPMENT_OPERATOR_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-development-operator-v1"
)
POLYMARKET_ROUND17_DEVELOPMENT_RISK_CAPITAL_QUOTE = Decimal("1000")
_MODEL_ROLES = (
    "train",
    "tune_calibration",
    "tune_selection",
    "tune_uncertainty",
)
_STATUS_RESOLUTION_INCOMPLETE = "resolution_incomplete"
_STATUS_MODEL_REJECTED = "model_rejected"
_STATUS_ECONOMIC_REJECTED = "economic_rejected"
_STATUS_DEVELOPMENT_ACCEPTED = "development_accepted"
_STATUSES = frozenset(
    {
        _STATUS_RESOLUTION_INCOMPLETE,
        _STATUS_MODEL_REJECTED,
        _STATUS_ECONOMIC_REJECTED,
        _STATUS_DEVELOPMENT_ACCEPTED,
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAXIMUM_CONTRACT_BYTES = 256 * 1024
_RESULT_KEYS = {
    "schema_version",
    "status",
    "contract_sha256",
    "round14_risk_contract_sha256",
    "economic_contract_sha256",
    "risk_capital_quote",
    "parents",
    "artifacts",
    "development_accepted",
    "official_development_outcomes_consulted",
    "execution_simulation_completed",
    "test_features_accessed",
    "test_targets_accessed",
    "test_execution_accessed",
    "profitability_claim",
    "paper_trading_authority",
    "live_trading_authority",
    "automatic_promotion",
    "polymarket_execution_independent",
    "binance_signal_role",
    "binance_credentials_used",
    "binance_execution_connected",
}
_PARENT_KEYS = {
    "campaign_development_index_sha256",
    "cohort_manifest_sha256",
    "resolution_acquisition_sha256",
    "development_dataset_sha256",
    "target_manifest_sha256",
    "model_pretest_sha256",
    "probability_calibration_sha256",
    "economic_pretest_sha256",
}


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


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 17 economic contract contains duplicate keys")
        output[key] = value
    return output


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 17 economic contract contains {value}")


def _load_economic_contract(path: Path) -> str:
    if (
        path.is_symlink()
        or not path.is_file()
        or not 2 <= path.stat().st_size <= _MAXIMUM_CONTRACT_BYTES
    ):
        raise ValueError("Round 17 economic contract is unavailable")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 17 economic contract is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Round 17 economic contract is not an object")
    body = dict(payload)
    claimed = str(body.pop("contract_sha256", "")).strip().lower()
    if (
        claimed != POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256
        or claimed != _canonical_sha256(body)
        or body.get("schema_version")
        != "polymarket-round17-btc-5m-economic-pretest-contract-v1"
        or body.get("round") != 17
    ):
        raise ValueError("Round 17 economic contract identity differs")
    return claimed


def _emit(
    progress: Callable[[Mapping[str, object]], None] | None,
    phase: str,
    **values: object,
) -> None:
    if progress is not None:
        progress({"phase": phase, **values})


@dataclass(frozen=True, slots=True)
class Round17DevelopmentOperatorConfig:
    campaign: Round17CampaignOperatorConfig
    risk_contract_path: Path
    economic_contract_path: Path
    output_path: Path
    compute_backend: str = "auto"

    def validate(self) -> Round17DevelopmentOperatorConfig:
        self.campaign.validate()
        contracts = (self.risk_contract_path, self.economic_contract_path)
        resolved_contracts = tuple(path.resolve() for path in contracts)
        output = self.output_path.resolve()
        database = self.campaign.database_path.resolve()
        state_root = self.campaign.state_root.resolve()
        if (
            any(path.is_symlink() or not path.is_file() for path in contracts)
            or len(set(resolved_contracts)) != len(resolved_contracts)
            or output in {*resolved_contracts, database, state_root}
            or state_root in output.parents
            or self.output_path.is_symlink()
            or (self.output_path.exists() and not self.output_path.is_file())
            or not isinstance(self.compute_backend, str)
            or not self.compute_backend.strip()
        ):
            raise ValueError("Round 17 development operator configuration is invalid")
        return self


def _target_payload(
    target: Round17DevelopmentTargetManifest | None,
) -> dict[str, object] | None:
    if target is None:
        return None
    return {
        **target.identity_payload(),
        "target_manifest_sha256": target.target_manifest_sha256,
    }


def _cohort_payload(index: Round17CampaignDevelopmentIndex) -> dict[str, object]:
    cohort = index.cohort_manifest
    return {
        **cohort.identity_payload(),
        "manifest_sha256": cohort.manifest_sha256,
    }


def _result_payload(
    *,
    status: str,
    risk_contract_sha256: str,
    index: Round17CampaignDevelopmentIndex,
    resolutions: Round17DevelopmentResolutionAcquisition,
    target: Round17DevelopmentTargetManifest | None,
    model_pretest: Mapping[str, object] | None,
    calibration: Mapping[str, object] | None,
    economic_pretest: Mapping[str, object] | None,
) -> dict[str, object]:
    development_accepted = status == _STATUS_DEVELOPMENT_ACCEPTED
    return {
        "schema_version": POLYMARKET_ROUND17_DEVELOPMENT_OPERATOR_SCHEMA_VERSION,
        "status": status,
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "round14_risk_contract_sha256": risk_contract_sha256,
        "economic_contract_sha256": POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256,
        "risk_capital_quote": format(
            POLYMARKET_ROUND17_DEVELOPMENT_RISK_CAPITAL_QUOTE,
            "f",
        ),
        "parents": {
            "campaign_development_index_sha256": index.index_sha256,
            "cohort_manifest_sha256": index.cohort_manifest.manifest_sha256,
            "resolution_acquisition_sha256": resolutions.acquisition_sha256,
            "development_dataset_sha256": (
                None if target is None else target.development_dataset_sha256
            ),
            "target_manifest_sha256": (
                None if target is None else target.target_manifest_sha256
            ),
            "model_pretest_sha256": (
                None if model_pretest is None else model_pretest["pretest_sha256"]
            ),
            "probability_calibration_sha256": (
                None if calibration is None else calibration["calibration_sha256"]
            ),
            "economic_pretest_sha256": (
                None
                if economic_pretest is None
                else economic_pretest["economic_pretest_sha256"]
            ),
        },
        "artifacts": {
            "development_index": index.asdict(),
            "cohort_manifest": _cohort_payload(index),
            "resolution_acquisition": resolutions.asdict(),
            "target_manifest": _target_payload(target),
            "model_pretest": None if model_pretest is None else dict(model_pretest),
            "probability_calibration": (
                None if calibration is None else dict(calibration)
            ),
            "economic_pretest": (
                None if economic_pretest is None else dict(economic_pretest)
            ),
        },
        "development_accepted": development_accepted,
        "official_development_outcomes_consulted": True,
        "execution_simulation_completed": economic_pretest is not None,
        "test_features_accessed": False,
        "test_targets_accessed": False,
        "test_execution_accessed": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
        "automatic_promotion": False,
        "polymarket_execution_independent": True,
        "binance_signal_role": "optional_public_read_only_feature_only",
        "binance_credentials_used": False,
        "binance_execution_connected": False,
    }


def validate_round17_development_result(
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = dict(value)
    claimed = str(payload.pop("result_sha256", "")).strip().lower()
    parents = payload.get("parents")
    artifacts = payload.get("artifacts")
    status = payload.get("status")
    if (
        claimed != _canonical_sha256(payload)
        or set(payload) != _RESULT_KEYS
        or status not in _STATUSES
        or payload.get("schema_version")
        != POLYMARKET_ROUND17_DEVELOPMENT_OPERATOR_SCHEMA_VERSION
        or payload.get("contract_sha256") != POLYMARKET_ROUND17_CONTRACT_SHA256
        or payload.get("economic_contract_sha256")
        != POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256
        or _SHA256.fullmatch(
            str(payload.get("round14_risk_contract_sha256") or "")
        )
        is None
        or payload.get("risk_capital_quote")
        != format(POLYMARKET_ROUND17_DEVELOPMENT_RISK_CAPITAL_QUOTE, "f")
        or not isinstance(parents, Mapping)
        or set(parents) != _PARENT_KEYS
        or not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {
            "development_index",
            "cohort_manifest",
            "resolution_acquisition",
            "target_manifest",
            "model_pretest",
            "probability_calibration",
            "economic_pretest",
        }
        or any(
            payload.get(name) is not False
            for name in (
                "test_features_accessed",
                "test_targets_accessed",
                "test_execution_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
                "automatic_promotion",
                "binance_credentials_used",
                "binance_execution_connected",
            )
        )
        or payload.get("official_development_outcomes_consulted") is not True
        or payload.get("polymarket_execution_independent") is not True
        or payload.get("binance_signal_role")
        != "optional_public_read_only_feature_only"
        or payload.get("development_accepted")
        is not (status == _STATUS_DEVELOPMENT_ACCEPTED)
        or payload.get("execution_simulation_completed")
        is not (
            status in {_STATUS_ECONOMIC_REJECTED, _STATUS_DEVELOPMENT_ACCEPTED}
        )
    ):
        raise ValueError("Round 17 development result integrity differs")

    index = artifacts["development_index"]
    cohort = artifacts["cohort_manifest"]
    resolutions = artifacts["resolution_acquisition"]
    target = artifacts["target_manifest"]
    model = artifacts["model_pretest"]
    calibration = artifacts["probability_calibration"]
    economic = artifacts["economic_pretest"]
    if (
        not isinstance(index, Mapping)
        or not isinstance(cohort, Mapping)
        or not isinstance(resolutions, Mapping)
        or parents.get("campaign_development_index_sha256")
        != index.get("index_sha256")
        or parents.get("cohort_manifest_sha256")
        != cohort.get("manifest_sha256")
        or index.get("cohort_manifest_sha256") != cohort.get("manifest_sha256")
        or parents.get("resolution_acquisition_sha256")
        != resolutions.get("acquisition_sha256")
        or resolutions.get("cohort_manifest_sha256")
        != parents.get("cohort_manifest_sha256")
    ):
        raise ValueError("Round 17 development result parent differs")
    cohort_conditions = cohort.get("conditions")
    index_markets = index.get("markets")
    resolution_observations = resolutions.get("observations")
    pending_condition_ids = resolutions.get("pending_condition_ids")
    if (
        not isinstance(cohort_conditions, list)
        or not cohort_conditions
        or not isinstance(index_markets, list)
        or not isinstance(resolution_observations, list)
        or not isinstance(pending_condition_ids, list)
        or any(not isinstance(item, Mapping) for item in cohort_conditions)
        or any(not isinstance(item, Mapping) for item in index_markets)
        or any(not isinstance(item, Mapping) for item in resolution_observations)
    ):
        raise ValueError("Round 17 development condition panel differs")
    cohort_ids = {
        str(item.get("condition_id"))
        for item in cohort_conditions
        if isinstance(item, Mapping)
    }
    market_ids = {
        str(item.get("condition_id"))
        for item in index_markets
        if isinstance(item, Mapping)
    }
    resolution_ids = {
        str(item.get("condition_id"))
        for item in resolution_observations
        if isinstance(item, Mapping)
    }
    pending_ids = {str(item) for item in pending_condition_ids}
    if (
        len(cohort_ids) != len(cohort_conditions)
        or len(market_ids) != len(index_markets)
        or len(resolution_ids) != len(resolution_observations)
        or cohort_ids != market_ids
        or resolution_ids & pending_ids
        or resolution_ids | pending_ids != cohort_ids
        or any(
            item.get("role") not in {*_MODEL_ROLES, "tune_economic"}
            for item in cohort_conditions
            if isinstance(item, Mapping)
        )
        or any(
            index.get(name) is not False
            for name in (
                "labels_consulted",
                "outcomes_consulted",
                "model_scores_consulted",
                "test_features_accessed",
                "test_targets_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
        or any(
            cohort.get(name) is not False
            for name in (
                "labels_consulted",
                "outcomes_consulted",
                "model_scores_consulted",
                "execution_scores_consulted",
                "test_features_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
        or resolutions.get("development_outcomes_consulted") is not True
        or any(
            resolutions.get(name) is not False
            for name in (
                "test_features_accessed",
                "test_targets_accessed",
                "model_scores_accessed",
                "execution_scores_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 17 development partition differs")
    index_body = dict(index)
    index_sha256 = str(index_body.pop("index_sha256", ""))
    cohort_body = dict(cohort)
    cohort_sha256 = str(cohort_body.pop("manifest_sha256", ""))
    resolution_body = dict(resolutions)
    resolution_sha256 = str(resolution_body.pop("acquisition_sha256", ""))
    resolution_complete = resolution_body.pop("complete", None)
    if (
        _SHA256.fullmatch(index_sha256) is None
        or index_sha256 != _canonical_sha256(index_body)
        or _SHA256.fullmatch(cohort_sha256) is None
        or cohort_sha256 != _canonical_sha256(cohort_body)
        or _SHA256.fullmatch(resolution_sha256) is None
        or resolution_sha256 != _canonical_sha256(resolution_body)
        or resolution_complete is not resolutions.get("complete")
    ):
        raise ValueError("Round 17 development nested artifact differs")

    if status == _STATUS_RESOLUTION_INCOMPLETE:
        if (
            resolutions.get("complete") is not False
            or not resolutions.get("pending_condition_ids")
            or any(item is not None for item in (target, model, calibration, economic))
        ):
            raise ValueError("Round 17 incomplete-resolution result differs")
        return {**payload, "result_sha256": claimed}

    if (
        resolutions.get("complete") is not True
        or not isinstance(target, Mapping)
        or parents.get("development_dataset_sha256")
        != target.get("development_dataset_sha256")
        or parents.get("target_manifest_sha256")
        != target.get("target_manifest_sha256")
        or not isinstance(model, Mapping)
        or target.get("development_labels_consulted") is not True
        or any(
            target.get(name) is not False
            for name in (
                "test_features_accessed",
                "test_targets_accessed",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
    ):
        raise ValueError("Round 17 development target parent differs")
    target_body = dict(target)
    target_sha256 = str(target_body.pop("target_manifest_sha256", ""))
    if (
        _SHA256.fullmatch(target_sha256) is None
        or target_sha256 != _canonical_sha256(target_body)
        or target.get("cohort_manifest_sha256") != cohort_sha256
    ):
        raise ValueError("Round 17 development target artifact differs")
    target_labels = target.get("labels")
    if (
        not isinstance(target_labels, list)
        or len(target_labels) != len(cohort_ids)
        or {
            str(item.get("condition_id"))
            for item in target_labels
            if isinstance(item, Mapping)
        }
        != cohort_ids
        or any(not isinstance(item, Mapping) for item in target_labels)
    ):
        raise ValueError("Round 17 development target condition panel differs")
    selected_model = validate_round17_pretest_artifact(model)
    model_partition = selected_model.get("dataset_and_partition")
    if (
        parents.get("model_pretest_sha256") != selected_model["pretest_sha256"]
        or not isinstance(model_partition, Mapping)
        or model_partition.get("dataset_sha256")
        != target.get("development_dataset_sha256")
        or model_partition.get("target_manifest_sha256")
        != target.get("target_manifest_sha256")
    ):
        raise ValueError("Round 17 development model parent differs")

    if status == _STATUS_MODEL_REJECTED:
        if (
            selected_model["development_accepted"] is not False
            or calibration is not None
            or economic is not None
            or parents.get("probability_calibration_sha256") is not None
            or parents.get("economic_pretest_sha256") is not None
        ):
            raise ValueError("Round 17 rejected-model result differs")
        return {**payload, "result_sha256": claimed}

    if not isinstance(calibration, Mapping) or not isinstance(economic, Mapping):
        raise ValueError("Round 17 economic result artifacts differ")
    selected_calibration = validate_round17_probability_calibration(
        calibration,
        model_pretest=selected_model,
    )
    selected_economic = validate_round17_economic_pretest(
        economic,
        model_pretest=selected_model,
        probability_calibration=selected_calibration,
    )
    if (
        selected_model["development_accepted"] is not True
        or parents.get("probability_calibration_sha256")
        != selected_calibration["calibration_sha256"]
        or parents.get("economic_pretest_sha256")
        != selected_economic["economic_pretest_sha256"]
        or selected_economic["development_accepted"]
        is not (status == _STATUS_DEVELOPMENT_ACCEPTED)
    ):
        raise ValueError("Round 17 economic result parent differs")
    return {**payload, "result_sha256": claimed}


def _publish(
    config: Round17DevelopmentOperatorConfig,
    *,
    status: str,
    risk_contract_sha256: str,
    index: Round17CampaignDevelopmentIndex,
    resolutions: Round17DevelopmentResolutionAcquisition,
    target: Round17DevelopmentTargetManifest | None,
    model_pretest: Mapping[str, object] | None,
    calibration: Mapping[str, object] | None,
    economic_pretest: Mapping[str, object] | None,
    progress: Callable[[Mapping[str, object]], None] | None,
) -> dict[str, object]:
    payload = _result_payload(
        status=status,
        risk_contract_sha256=risk_contract_sha256,
        index=index,
        resolutions=resolutions,
        target=target,
        model_pretest=model_pretest,
        calibration=calibration,
        economic_pretest=economic_pretest,
    )
    payload["result_sha256"] = _canonical_sha256(payload)
    validated = validate_round17_development_result(payload)
    write_json_atomic(config.output_path, validated, indent=None, sort_keys=True)
    _emit(
        progress,
        "result_written",
        status=status,
        output_path=str(config.output_path),
        result_sha256=validated["result_sha256"],
    )
    return validated


def run_round17_development(
    config: Round17DevelopmentOperatorConfig,
    *,
    client: PolymarketPublicClient | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run the frozen development sequence without opening the held-out test role."""

    selected = config.validate()
    economic_contract_sha256 = _load_economic_contract(
        selected.economic_contract_path
    )
    if economic_contract_sha256 != POLYMARKET_ROUND17_ECONOMIC_CONTRACT_SHA256:
        raise RuntimeError("Round 17 economic contract changed after validation")
    program = load_round14_contract(selected.risk_contract_path)
    _emit(progress, "development_index_start")
    index = materialize_round17_campaign_development_index(
        selected.campaign,
        progress=progress,
    )
    plan_source = load_round17_cohort_plan(selected.campaign.cohort_plan_path)
    market_by_condition = index.market_mapping()
    _emit(progress, "resolution_start", condition_count=len(index.markets))
    resolutions = acquire_round17_development_resolutions(
        plan_source,
        index.cohort_manifest,
        market_by_condition,
        client=client,
        progress=progress,
    )
    if not resolutions.complete:
        return _publish(
            selected,
            status=_STATUS_RESOLUTION_INCOMPLETE,
            risk_contract_sha256=program.contract_sha256,
            index=index,
            resolutions=resolutions,
            target=None,
            model_pretest=None,
            calibration=None,
            economic_pretest=None,
            progress=progress,
        )

    labels = resolutions.labels(
        plan_source,
        index.cohort_manifest,
        market_by_condition,
    )
    target = build_round17_development_target_manifest(
        plan_source,
        index.cohort_manifest,
        labels,
    )
    assembler = Round17DevelopmentPanelAssembler(
        plan_source,
        index.cohort_manifest,
        target,
        roles=_MODEL_ROLES,
    )
    reference_by_condition = {
        item.condition_id: item for item in index.cohort_manifest.conditions
    }
    observation_by_condition = {
        item.condition_id: item for item in resolutions.observations
    }
    expected_economic_ids = tuple(
        item.condition_id
        for item in index.cohort_manifest.conditions
        if item.role == "tune_economic"
    )
    economic_ids: list[str] = []
    economic_outcomes = []
    model_pretest: dict[str, object] | None = None
    calibration: dict[str, object] | None = None
    model_rejected = False

    _emit(
        progress,
        "development_second_pass_start",
        condition_count=len(index.cohort_manifest.conditions),
    )
    condition_iterator = iter_round17_campaign_development_conditions(
        selected.campaign
    )
    with closing(condition_iterator):
        for count, materialized in enumerate(condition_iterator, start=1):
            condition_id = materialized.dataset.condition_id
            reference = reference_by_condition.get(condition_id)
            if (
                reference is None
                or materialized.cohort_condition != reference
                or materialized.market.asdict()
                != market_by_condition[condition_id].asdict()
            ):
                raise ValueError("Round 17 second-pass condition evidence differs")
            if reference.role in _MODEL_ROLES:
                if model_pretest is not None:
                    raise ValueError("Round 17 model partition order differs")
                assembler.append(materialized.dataset)
            elif reference.role == "tune_economic":
                if model_pretest is None:
                    _emit(progress, "model_fit_start")
                    panels = assembler.finish()
                    model_pretest = fit_round17_development_pretest(
                        panels["train"],
                        panels["tune_calibration"],
                        panels["tune_selection"],
                        compute_backend=selected.compute_backend,
                    )
                    model_pretest = validate_round17_pretest_artifact(model_pretest)
                    _emit(
                        progress,
                        "model_fit_complete",
                        development_accepted=model_pretest["development_accepted"],
                        selected_candidate_id=model_pretest["selected_candidate_id"],
                    )
                    if model_pretest["development_accepted"] is not True:
                        model_rejected = True
                        break
                    _emit(progress, "probability_calibration_start")
                    calibration = fit_round17_probability_calibration(
                        panels["tune_uncertainty"],
                        model_pretest,
                    )
                    calibration = validate_round17_probability_calibration(
                        calibration,
                        model_pretest=model_pretest,
                    )
                    _emit(
                        progress,
                        "probability_calibration_complete",
                        calibration_sha256=calibration["calibration_sha256"],
                    )
                    del panels
                assert calibration is not None
                _emit(
                    progress,
                    "economic_condition_start",
                    completed_conditions=len(economic_ids),
                    total_conditions=len(expected_economic_ids),
                    event_start_ms=materialized.market.event_start_ms,
                )
                envelopes = apply_round17_probability_calibration_rows(
                    calibration,
                    model_pretest,
                    materialized.dataset.rows,
                    dataset_sha256=target.development_dataset_sha256,
                    event_start_ms=materialized.market.event_start_ms,
                )
                predictions = tuple(
                    build_round17_calibrated_decision_probability(row, envelope)
                    for row, envelope in zip(
                        materialized.dataset.rows,
                        envelopes,
                        strict=True,
                    )
                )
                economic_outcomes.extend(
                    materialize_round17_condition_economic_outcomes(
                        market=materialized.market,
                        dataset=materialized.dataset,
                        predictions=predictions,
                        books=materialized.books,
                        resolution=observation_by_condition[
                            condition_id
                        ].resolution_evidence(),
                        program=program,
                        risk_capital_quote=(
                            POLYMARKET_ROUND17_DEVELOPMENT_RISK_CAPITAL_QUOTE
                        ),
                    )
                )
                economic_ids.append(condition_id)
                _emit(
                    progress,
                    "economic_condition_complete",
                    completed_conditions=len(economic_ids),
                    total_conditions=len(expected_economic_ids),
                    event_start_ms=materialized.market.event_start_ms,
                )
            else:
                raise ValueError("Round 17 second pass exposed a reserved role")
            if count == 1 or count % 25 == 0:
                _emit(
                    progress,
                    "development_second_pass",
                    completed_conditions=count,
                    total_conditions=len(index.cohort_manifest.conditions),
                    last_event_start_ms=materialized.market.event_start_ms,
                )

    if model_rejected:
        assert model_pretest is not None
        return _publish(
            selected,
            status=_STATUS_MODEL_REJECTED,
            risk_contract_sha256=program.contract_sha256,
            index=index,
            resolutions=resolutions,
            target=target,
            model_pretest=model_pretest,
            calibration=None,
            economic_pretest=None,
            progress=progress,
        )
    if (
        model_pretest is None
        or calibration is None
        or tuple(economic_ids) != expected_economic_ids
    ):
        raise ValueError("Round 17 development second pass is incomplete")
    _emit(progress, "economic_pretest_start", outcome_count=len(economic_outcomes))
    economic_pretest = fit_round17_economic_pretest(
        economic_outcomes,
        program,
        model_pretest=model_pretest,
        probability_calibration=calibration,
        dataset_sha256=target.development_dataset_sha256,
    )
    economic_pretest = validate_round17_economic_pretest(
        economic_pretest,
        model_pretest=model_pretest,
        probability_calibration=calibration,
    )
    status = (
        _STATUS_DEVELOPMENT_ACCEPTED
        if economic_pretest["development_accepted"] is True
        else _STATUS_ECONOMIC_REJECTED
    )
    _emit(
        progress,
        "economic_pretest_complete",
        development_accepted=economic_pretest["development_accepted"],
        status=status,
    )
    return _publish(
        selected,
        status=status,
        risk_contract_sha256=program.contract_sha256,
        index=index,
        resolutions=resolutions,
        target=target,
        model_pretest=model_pretest,
        calibration=calibration,
        economic_pretest=economic_pretest,
        progress=progress,
    )


__all__ = [
    "POLYMARKET_ROUND17_DEVELOPMENT_OPERATOR_SCHEMA_VERSION",
    "POLYMARKET_ROUND17_DEVELOPMENT_RISK_CAPITAL_QUOTE",
    "Round17DevelopmentOperatorConfig",
    "run_round17_development",
    "validate_round17_development_result",
]
