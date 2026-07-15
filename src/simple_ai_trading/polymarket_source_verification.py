"""Independent source reconstruction for Polymarket research artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from .polymarket_features import (
    PolymarketFeatureConfig,
    build_polymarket_feature_dataset,
)
from .polymarket_model import (
    PolymarketModelConfig,
    build_polymarket_model_dataset,
    fit_polymarket_offset_model,
    fit_polymarket_profile_challenger,
    predict_polymarket_probabilities,
    predict_polymarket_profile_probabilities,
    split_polymarket_model_dataset,
)
from .polymarket_model_execution import (
    POLYMARKET_EXECUTION_CONFIG_SCHEMA_VERSION,
    PolymarketExecutionResearchConfig,
    evaluate_polymarket_execution_policy,
    evaluate_polymarket_retry_execution_policy,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay


POLYMARKET_SOURCE_VERIFICATION_SCHEMA_VERSION = (
    "polymarket-source-verification-v2"
)
POLYMARKET_SOURCE_VERIFICATION_CHECKS = (
    "recorder_integrity_replayed",
    "feature_dataset_reconstructed",
    "model_dataset_reconstructed",
    "chronological_split_reconstructed",
    "model_fit_reconstructed",
    "probability_report_reconstructed",
    "profile_challenger_reconstructed",
    "held_out_predictions_reconstructed",
    "all_execution_scenarios_reconstructed",
)


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


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _require_canonical_match(
    actual: object,
    expected: object,
    *,
    name: str,
) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"Polymarket source reconstruction disagrees on {name}")


def _feature_config(value: object) -> PolymarketFeatureConfig:
    raw = _mapping(value, "feature configuration")
    expected = {item.name for item in fields(PolymarketFeatureConfig)}
    if set(raw) != expected:
        raise ValueError("Polymarket feature configuration fields drifted")
    return PolymarketFeatureConfig(**dict(raw)).validated()


def _model_config(value: object) -> PolymarketModelConfig:
    raw = dict(_mapping(value, "model configuration"))
    expected = {item.name for item in fields(PolymarketModelConfig)}
    if set(raw) != expected:
        raise ValueError("Polymarket model configuration fields drifted")
    raw["decision_horizons_seconds"] = tuple(raw["decision_horizons_seconds"])
    raw["l2_candidates"] = tuple(raw["l2_candidates"])
    return PolymarketModelConfig(**raw).validated()


def _execution_config(value: object) -> PolymarketExecutionResearchConfig:
    raw = _mapping(value, "execution configuration")
    expected = {
        "schema_version",
        *(item.name for item in fields(PolymarketExecutionResearchConfig)),
    }
    if (
        set(raw) != expected
        or raw.get("schema_version")
        != POLYMARKET_EXECUTION_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("Polymarket execution configuration fields drifted")
    return PolymarketExecutionResearchConfig(
        submission_latency_ms=int(raw["submission_latency_ms"]),
        maximum_execution_observation_delay_ms=int(
            raw["maximum_execution_observation_delay_ms"]
        ),
        maximum_book_age_ms=int(raw["maximum_book_age_ms"]),
        order_ttl_ms=int(raw["order_ttl_ms"]),
        minimum_expected_edge_per_contract=Decimal(
            str(raw["minimum_expected_edge_per_contract"])
        ),
        initial_capital_quote=Decimal(str(raw["initial_capital_quote"])),
        maximum_loss_fraction_per_market=Decimal(
            str(raw["maximum_loss_fraction_per_market"])
        ),
        maximum_loss_fraction_per_time_group=Decimal(
            str(raw["maximum_loss_fraction_per_time_group"])
        ),
    ).validated()


@dataclass(frozen=True)
class PolymarketSourceVerificationReport:
    schema_version: str
    status: str
    artifact_sha256: str
    run_id: str
    recorder_report_sha256: str
    feature_dataset_sha256: str
    model_dataset_sha256: str
    split_sha256: str
    model_sha256: str
    probability_report_sha256: str
    profile_model_sha256: str
    profile_probability_report_sha256: str
    held_out_rows_sha256: str
    profile_held_out_rows_sha256: str
    execution_report_sha256_by_policy_and_latency: Mapping[
        str, Mapping[str, str]
    ]
    verified_feature_row_count: int
    verified_model_sample_count: int
    verified_held_out_sample_count: int
    verified_execution_scenario_count: int
    verified_execution_trade_count: int
    verified_filled_order_count: int
    checks: Mapping[str, bool]
    report_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "artifact_sha256": self.artifact_sha256,
            "run_id": self.run_id,
            "recorder_report_sha256": self.recorder_report_sha256,
            "feature_dataset_sha256": self.feature_dataset_sha256,
            "model_dataset_sha256": self.model_dataset_sha256,
            "split_sha256": self.split_sha256,
            "model_sha256": self.model_sha256,
            "probability_report_sha256": self.probability_report_sha256,
            "profile_model_sha256": self.profile_model_sha256,
            "profile_probability_report_sha256": (
                self.profile_probability_report_sha256
            ),
            "held_out_rows_sha256": self.held_out_rows_sha256,
            "profile_held_out_rows_sha256": self.profile_held_out_rows_sha256,
            "execution_report_sha256_by_policy_and_latency": {
                policy: dict(sorted(reports.items()))
                for policy, reports in sorted(
                    self.execution_report_sha256_by_policy_and_latency.items()
                )
            },
            "verified_feature_row_count": self.verified_feature_row_count,
            "verified_model_sample_count": self.verified_model_sample_count,
            "verified_held_out_sample_count": self.verified_held_out_sample_count,
            "verified_execution_scenario_count": (
                self.verified_execution_scenario_count
            ),
            "verified_execution_trade_count": self.verified_execution_trade_count,
            "verified_filled_order_count": self.verified_filled_order_count,
            "checks": dict(sorted(self.checks.items())),
            "report_sha256": self.report_sha256,
            "trading_authority": self.trading_authority,
            "execution_claim": self.execution_claim,
            "profitability_claim": self.profitability_claim,
            "portfolio_claim": self.portfolio_claim,
            "leverage_applied": self.leverage_applied,
        }


def _report_identity(
    report: PolymarketSourceVerificationReport,
) -> dict[str, object]:
    payload = report.asdict()
    payload.pop("report_sha256")
    return payload


def validate_polymarket_source_verification(
    value: object,
    *,
    artifact_sha256: str | None = None,
    run_id: str | None = None,
) -> Mapping[str, Any]:
    """Validate an externally persisted source-reconstruction report."""

    report = _mapping(value, "Polymarket source verification")
    expected = {item.name for item in fields(PolymarketSourceVerificationReport)}
    if set(report) != expected:
        raise ValueError("Polymarket source verification fields drifted")
    identity = dict(report)
    claimed = str(identity.pop("report_sha256", ""))
    checks = report.get("checks")
    execution_hashes = report.get(
        "execution_report_sha256_by_policy_and_latency"
    )
    scenario_count = report.get("verified_execution_scenario_count")
    counters = (
        report.get("verified_feature_row_count"),
        report.get("verified_model_sample_count"),
        report.get("verified_held_out_sample_count"),
        scenario_count,
        report.get("verified_execution_trade_count"),
        report.get("verified_filled_order_count"),
    )
    hashes_valid = all(
        _is_sha256(report.get(key))
        for key in (
            "artifact_sha256",
            "recorder_report_sha256",
            "feature_dataset_sha256",
            "model_dataset_sha256",
            "split_sha256",
            "model_sha256",
            "probability_report_sha256",
            "profile_model_sha256",
            "profile_probability_report_sha256",
            "held_out_rows_sha256",
            "profile_held_out_rows_sha256",
        )
    )
    scenario_hashes_valid = isinstance(execution_hashes, Mapping)
    scenario_total = 0
    if scenario_hashes_valid:
        policies = set(execution_hashes)
        scenario_hashes_valid = {"baseline", "model"}.issubset(policies) and policies <= {
            "baseline",
            "model",
            "profile_model",
            "model_retry",
            "ai",
        }
        for policy, raw_reports in execution_hashes.items():
            if not isinstance(policy, str) or not isinstance(raw_reports, Mapping):
                scenario_hashes_valid = False
                continue
            scenario_total += len(raw_reports)
            for latency, digest in raw_reports.items():
                try:
                    latency_ms = int(latency)
                except (TypeError, ValueError, OverflowError):
                    scenario_hashes_valid = False
                    continue
                if str(latency_ms) != str(latency) or not 1 <= latency_ms <= 60_000:
                    scenario_hashes_valid = False
                if not _is_sha256(digest):
                    scenario_hashes_valid = False
    if (
        report.get("schema_version")
        != POLYMARKET_SOURCE_VERIFICATION_SCHEMA_VERSION
        or report.get("status") != "verified"
        or not str(report.get("run_id", ""))
        or not isinstance(checks, Mapping)
        or set(checks) != set(POLYMARKET_SOURCE_VERIFICATION_CHECKS)
        or any(value is not True for value in checks.values())
        or not hashes_valid
        or not scenario_hashes_valid
        or isinstance(scenario_count, bool)
        or scenario_count != scenario_total
        or any(
            isinstance(counter, bool)
            or not isinstance(counter, int)
            or counter < 0
            for counter in counters
        )
        or int(report["verified_feature_row_count"]) < 1
        or int(report["verified_model_sample_count"]) < 1
        or int(report["verified_held_out_sample_count"]) < 1
        or int(scenario_count) < 2
        or int(report["verified_filled_order_count"])
        > int(report["verified_execution_trade_count"])
        or not _is_sha256(claimed)
        or claimed != _canonical_sha256(identity)
        or any(
            report.get(key) is not False
            for key in (
                "trading_authority",
                "execution_claim",
                "profitability_claim",
                "portfolio_claim",
                "leverage_applied",
            )
        )
    ):
        raise ValueError("Polymarket source verification is invalid")
    if artifact_sha256 is not None and report.get("artifact_sha256") != str(
        artifact_sha256
    ):
        raise ValueError("Polymarket source verification binds another artifact")
    if run_id is not None and report.get("run_id") != str(run_id):
        raise ValueError("Polymarket source verification binds another recorder run")
    return report


def verify_polymarket_model_artifact_source(
    artifact_path: str | Path,
    database_path: str | Path,
    *,
    memory_limit: str = "1GB",
    database_threads: int = 2,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> PolymarketSourceVerificationReport:
    """Rebuild one artifact from immutable source and fail on any disagreement."""

    from .polymarket_publication import validate_polymarket_model_artifact

    def notify(phase: str, **details: object) -> None:
        if progress is not None:
            progress(phase, details)

    validated = validate_polymarket_model_artifact(artifact_path)
    payload = validated.payload
    run_id = str(payload["run_id"])
    feature_summary = _mapping(payload["feature_dataset"], "feature dataset")
    model_summary = _mapping(payload["model_dataset"], "model dataset")
    split_summary = _mapping(payload["split"], "model split")
    feature_config = _feature_config(feature_summary["config"])
    notify("feature-reconstruction", run_id=run_id)
    with PolymarketEvidenceStore(
        database_path,
        memory_limit=memory_limit,
        threads=database_threads,
        read_only=True,
    ) as store:
        feature_dataset = build_polymarket_feature_dataset(
            store,
            run_id=run_id,
            config=feature_config,
        )
        _require_canonical_match(
            feature_dataset.summary(),
            feature_summary,
            name="feature dataset",
        )
        recorder_row = store.connect().execute(
            """
            SELECT status, report_sha256 FROM polymarket_recorder_run
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        recorder_status = "" if recorder_row is None else str(recorder_row[0])
        recorder_status_allowed = recorder_status == "complete" or (
            feature_config.allow_segmented_gaps and recorder_status == "degraded"
        )
        if (
            recorder_row is None
            or not recorder_status_allowed
            or len(str(recorder_row[1])) != 64
        ):
            raise ValueError(
                "Polymarket source verification requires a complete report or "
                "an explicitly segmented degraded report"
            )
        recorder_report_sha256 = str(recorder_row[1])

        notify("model-reconstruction", feature_rows=len(feature_dataset.rows))
        markets = PolymarketEvidenceReplay.load_markets(store, run_id=run_id)
        model_dataset = build_polymarket_model_dataset(
            feature_dataset,
            markets,
            config=_model_config(model_summary["config"]),
        )
        _require_canonical_match(
            model_dataset.summary(),
            model_summary,
            name="model dataset",
        )
        split = split_polymarket_model_dataset(model_dataset)
        _require_canonical_match(split.summary(), split_summary, name="model split")
        model, probability_report = fit_polymarket_offset_model(
            model_dataset,
            split,
        )
        _require_canonical_match(model.asdict(), payload["model"], name="model fit")
        _require_canonical_match(
            probability_report.asdict(),
            payload["probability_report"],
            name="probability report",
        )
        profile_model, profile_probability_report = (
            fit_polymarket_profile_challenger(
                model_dataset,
                split,
                model,
            )
        )
        _require_canonical_match(
            profile_model.asdict(),
            payload["profile_model"],
            name="profile model fit",
        )
        _require_canonical_match(
            profile_probability_report.asdict(),
            payload["profile_probability_report"],
            name="profile probability report",
        )

        baseline_probabilities = [
            item.baseline_up_probability for item in split.test
        ]
        model_probabilities = list(
            predict_polymarket_probabilities(model, split.test)
        )
        profile_probabilities = list(
            predict_polymarket_profile_probabilities(profile_model, split.test)
        )
        reconstructed_predictions = [
            {
                **sample.asdict(),
                "model_up_probability": format(float(probability), ".17g"),
            }
            for sample, probability in zip(
                split.test,
                model_probabilities,
                strict=True,
            )
        ]
        _require_canonical_match(
            reconstructed_predictions,
            validated.predictions,
            name="held-out prediction rows",
        )
        reconstructed_profile_predictions = [
            {
                **row,
                "profile_model_up_probability": format(
                    float(profile_probability),
                    ".17g",
                ),
            }
            for row, profile_probability in zip(
                reconstructed_predictions,
                profile_probabilities,
                strict=True,
            )
        ]
        _require_canonical_match(
            reconstructed_profile_predictions,
            validated.profile_predictions,
            name="profile held-out prediction rows",
        )

        test_conditions = tuple(
            sorted({item.condition_id for item in split.test})
        )
        notify("execution-replay", test_conditions=len(test_conditions))
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=run_id,
            allow_segmented_gaps=feature_config.allow_segmented_gaps,
            book_sample_interval_ms=0,
            condition_ids=test_conditions,
        )
        sensitivity = _mapping(
            payload["execution_latency_sensitivity"],
            "execution latency sensitivity",
        )
        policy_reports = _mapping(
            sensitivity["policies"],
            "execution latency policies",
        )
        execution_hashes: dict[str, dict[str, str]] = {}
        scenario_count = 0
        trade_count = 0
        filled_count = 0
        for policy, raw_reports in sorted(policy_reports.items()):
            probabilities = (
                baseline_probabilities
                if policy == "baseline"
                else (
                    profile_probabilities
                    if policy == "profile_model"
                    else model_probabilities
                )
            )
            reports = _mapping(raw_reports, f"{policy} execution scenarios")
            execution_hashes[policy] = {}
            for latency, raw_report in sorted(
                reports.items(),
                key=lambda item: int(item[0]),
            ):
                expected_report = _mapping(
                    raw_report,
                    f"{policy} {latency}ms execution report",
                )
                permissions = dict(
                    _mapping(
                        expected_report["market_permissions"],
                        "execution market permissions",
                    )
                )
                delays = dict(
                    _mapping(
                        expected_report["decision_delay_ms_by_condition"],
                        "execution decision delays",
                    )
                )
                evaluator = (
                    evaluate_polymarket_retry_execution_policy
                    if policy == "model_retry"
                    else evaluate_polymarket_execution_policy
                )
                reconstructed = evaluator(
                    split.test,
                    probabilities,
                    replay,
                    config=_execution_config(expected_report["config"]),
                    market_permissions=permissions,
                    decision_delay_ms_by_condition=delays,
                )
                reconstructed_payload = reconstructed.asdict()
                _require_canonical_match(
                    reconstructed_payload,
                    expected_report,
                    name=f"{policy} {latency}ms execution report",
                )
                execution_hashes[policy][str(latency)] = (
                    reconstructed.report_sha256
                )
                scenario_count += 1
                trade_count += len(reconstructed.trades)
                filled_count += reconstructed.filled_order_count
                notify(
                    "execution-scenario",
                    policy=policy,
                    latency_ms=int(latency),
                    trades=len(reconstructed.trades),
                    fills=reconstructed.filled_order_count,
                )

    checks = {name: True for name in POLYMARKET_SOURCE_VERIFICATION_CHECKS}
    provisional = PolymarketSourceVerificationReport(
        schema_version=POLYMARKET_SOURCE_VERIFICATION_SCHEMA_VERSION,
        status="verified",
        artifact_sha256=validated.artifact_sha256,
        run_id=run_id,
        recorder_report_sha256=recorder_report_sha256,
        feature_dataset_sha256=feature_dataset.dataset_sha256,
        model_dataset_sha256=model_dataset.dataset_sha256,
        split_sha256=split.split_sha256,
        model_sha256=model.model_sha256,
        probability_report_sha256=probability_report.report_sha256,
        profile_model_sha256=profile_model.model_sha256,
        profile_probability_report_sha256=(
            profile_probability_report.report_sha256
        ),
        held_out_rows_sha256=_canonical_sha256(reconstructed_predictions),
        profile_held_out_rows_sha256=_canonical_sha256(
            [
                {
                    "sample_id": row["sample_id"],
                    "profile_model_up_probability": row[
                        "profile_model_up_probability"
                    ],
                }
                for row in reconstructed_profile_predictions
            ]
        ),
        execution_report_sha256_by_policy_and_latency=execution_hashes,
        verified_feature_row_count=len(feature_dataset.rows),
        verified_model_sample_count=len(model_dataset.samples),
        verified_held_out_sample_count=len(split.test),
        verified_execution_scenario_count=scenario_count,
        verified_execution_trade_count=trade_count,
        verified_filled_order_count=filled_count,
        checks=checks,
        report_sha256="",
    )
    report = PolymarketSourceVerificationReport(
        **{
            **provisional.__dict__,
            "report_sha256": _canonical_sha256(_report_identity(provisional)),
        }
    )
    validate_polymarket_source_verification(
        report.asdict(),
        artifact_sha256=validated.artifact_sha256,
        run_id=run_id,
    )
    notify("complete", report_sha256=report.report_sha256)
    return report


__all__ = [
    "POLYMARKET_SOURCE_VERIFICATION_SCHEMA_VERSION",
    "POLYMARKET_SOURCE_VERIFICATION_CHECKS",
    "PolymarketSourceVerificationReport",
    "validate_polymarket_source_verification",
    "verify_polymarket_model_artifact_source",
]
