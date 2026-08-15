"""Fail-closed audit for the immutable historical research snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.research_data_snapshot import (
    RESEARCH_DATA_CUTOFF_MS,
    RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256,
    RESEARCH_DATA_SNAPSHOT_ID,
    load_research_data_snapshot_contract,
    require_historical_event_time_ms,
    require_historical_utc_date,
    validate_prospective_partition,
)


AUDIT_SCHEMA_VERSION = "research-data-snapshot-audit-v1"


@dataclass(frozen=True)
class _ArtifactSpec:
    path: str
    schema_version: str | None
    identity_field: str | None
    identity_sha256: str
    date_reader: Callable[[Mapping[str, Any]], Sequence[str]]
    millisecond_reader: Callable[[Mapping[str, Any]], Sequence[int]] = lambda _: ()


@dataclass(frozen=True)
class _ProspectiveSpec:
    experiment_id: str
    path: str
    schema_version: str
    identity_field: str
    identity_sha256: str


def _round72_inventory_dates(value: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(item["selected_day"]) for item in value["selected_months"])


def _round72_ingestion_dates(value: Mapping[str, Any]) -> tuple[str, ...]:
    certificate = value["corpus_certificate"]
    return (
        str(certificate["first_period"]),
        str(certificate["last_period"]),
        *(str(item["period"]) for item in value["day_results"]),
    )


def _round62_inventory_dates(value: Mapping[str, Any]) -> tuple[str, ...]:
    dates: list[str] = []
    for item in value["coverage"]:
        dates.extend(
            str(item[key])
            for key in (
                "available_first_period",
                "available_last_period",
                "selected_first_period",
                "selected_last_period",
            )
        )
    for item in value["inventory_snapshots"]:
        dates.extend((str(item["first_period"]), str(item["last_period"])))
    return tuple(dates)


def _round62_report_dates(value: Mapping[str, Any]) -> tuple[str, ...]:
    dates: list[str] = []
    for certificate in value["certificates"]:
        dates.extend(
            str(certificate[key])
            for key in (
                "common_first_period",
                "common_last_period",
                "required_first_period",
                "required_last_period",
            )
        )
        for item in certificate["data_types"].values():
            dates.extend(
                str(item[key])
                for key in (
                    "first_period",
                    "last_period",
                    "scope_start_period",
                    "scope_end_period",
                )
            )
    return tuple(dates)


def _round62_report_milliseconds(value: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        int(value["certificate_required_start_ms"]),
        int(value["certificate_required_end_ms"]),
    )


def _round15_inventory_dates(value: Mapping[str, Any]) -> tuple[str, ...]:
    dates = [str(value["range"]["first_day"]), str(value["range"]["last_day"])]
    for day in value["days"]:
        dates.append(str(day["period"]))
        dates.extend(str(item["period"]) for item in day["archives"])
    return tuple(dates)


def _round15_status_dates(value: Mapping[str, Any]) -> tuple[str, ...]:
    dates = [str(value["first_completed_day"]), str(value["last_completed_day"])]
    dates.extend(str(item["period"]) for item in value["latest_batch"])
    return tuple(dates)


_HISTORICAL_ARTIFACTS = (
    _ArtifactSpec(
        path="docs/model-research/action-value/latest/inventory.json",
        schema_version="round-072-spot-perpetual-inventory-v1",
        identity_field="inventory_sha256",
        identity_sha256="e8c505132716c68ad753cbdd93b23094b778d9067c8a6c9381fad0e20cdd662c",
        date_reader=_round72_inventory_dates,
    ),
    _ArtifactSpec(
        path="docs/model-research/action-value/latest/corpus-ingestion.json",
        schema_version="round-072-spot-perpetual-corpus-ingestion-v1",
        identity_field="report_sha256",
        identity_sha256="1d7791db923f1d1a7eddc8189934424795246ea01250f6dbef26a59483605adb",
        date_reader=_round72_ingestion_dates,
    ),
    _ArtifactSpec(
        path=(
            "docs/model-research/action-value/"
            "round-062-official-archive-inventory.json"
        ),
        schema_version=None,
        identity_field=None,
        identity_sha256="99385af0239e25ba5ea1d6687e2cbdab816c7e5f78c63a89e9817620bb56ecb1",
        date_reader=_round62_inventory_dates,
    ),
    _ArtifactSpec(
        path=(
            "docs/model-research/action-value/"
            "round-062-depth-corpus-ingestion-report.json"
        ),
        schema_version="round-062-frozen-depth-corpus-ingestion-v1",
        identity_field="report_sha256",
        identity_sha256="4fd2a8c0e6020b54aa292b4d797046574ecdedbbec8cdb0d62f9323b615518bf",
        date_reader=_round62_report_dates,
        millisecond_reader=_round62_report_milliseconds,
    ),
    _ArtifactSpec(
        path=(
            "docs/model-research/polymarket/"
            "round-015-btc-5m-full-history-inventory-v1.json"
        ),
        schema_version="polymarket-btc-flow-history-inventory-v1",
        identity_field="artifact_sha256",
        identity_sha256="f68c0dc7709c98f342d04bacbd1b120c87ee07712ab0aeedbb3e1915cb8e77e2",
        date_reader=_round15_inventory_dates,
    ),
    _ArtifactSpec(
        path=(
            "docs/model-research/polymarket/"
            "round-015-btc-5m-history-ingestion-status.json"
        ),
        schema_version="polymarket-btc-flow-history-ingestion-status-v1",
        identity_field="artifact_sha256",
        identity_sha256="1ffc59d70f15add694566a32a4b89612007489f8433f8f697a5bca8ce0c006f8",
        date_reader=_round15_status_dates,
    ),
)


_PROSPECTIVE_EXPERIMENTS = (
    _ProspectiveSpec(
        experiment_id="polymarket-round27-stage1",
        path=(
            "docs/model-research/polymarket/"
            "round-027-stage1-campaign-contract-v1.json"
        ),
        schema_version="polymarket-round27-stage1-campaign-contract-v1",
        identity_field="contract_sha256",
        identity_sha256="3f484154d69baed632e617f2de41b149385299a97b47e5e9184c694c43c89392",
    ),
    _ProspectiveSpec(
        experiment_id="polymarket-round28-binance-bbo",
        path=(
            "docs/model-research/polymarket/"
            "round-028-binance-bbo-matched-ablation-preregistration-v1.json"
        ),
        schema_version=(
            "polymarket-round28-binance-bbo-matched-ablation-preregistration-v1"
        ),
        identity_field="preregistration_sha256",
        identity_sha256="8239488145f0ffe331cf9823e5517120dda0d12eb5f366cf00c5e106318d4668",
    ),
    _ProspectiveSpec(
        experiment_id="action-value-round75-continuous-capture",
        path=(
            "docs/model-research/action-value/"
            "round-075-continuous-capture-contract-v4.json"
        ),
        schema_version="round-075-continuous-capture-contract-v4",
        identity_field="artifact_sha256",
        identity_sha256="eca592acb4c3f37c6d043d37664614d35994d6e9b3ebea2e801351c287a49bbf",
    ),
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("research artifact has duplicate JSON keys")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {raw}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} root is not an object")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _verify_identity(
    *, path: Path, value: Mapping[str, Any], field: str | None, expected: str
) -> None:
    if field is None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        payload = dict(value)
        claimed = str(payload.pop(field, "")).lower()
        actual = _canonical_sha256(payload)
        if claimed != expected:
            raise ValueError(f"{path} declared identity differs")
    if actual != expected:
        raise ValueError(f"{path} canonical identity differs")


def _audit_historical_artifact(
    repository: Path, spec: _ArtifactSpec
) -> dict[str, object]:
    path = repository / spec.path
    value = _load_json(path)
    if spec.schema_version is not None and value.get("schema_version") != spec.schema_version:
        raise ValueError(f"{path} schema differs")
    _verify_identity(
        path=path,
        value=value,
        field=spec.identity_field,
        expected=spec.identity_sha256,
    )
    try:
        dates = tuple(spec.date_reader(value))
        milliseconds = tuple(spec.millisecond_reader(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} event-time schema differs") from exc
    if not dates and not milliseconds:
        raise ValueError(f"{path} has no auditable event-time boundary")
    validated_dates = tuple(
        require_historical_utc_date(item, name=f"{spec.path} event date")
        for item in dates
    )
    validated_ms = tuple(
        require_historical_event_time_ms(item, name=f"{spec.path} event time")
        for item in milliseconds
    )
    extrema = list(validated_dates)
    extrema.extend(
        datetime.fromtimestamp(item / 1_000, tz=UTC).date().isoformat()
        for item in validated_ms
    )
    return {
        "identity_sha256": spec.identity_sha256,
        "maximum_event_date_utc": max(extrema),
        "minimum_event_date_utc": min(extrema),
        "path": spec.path,
        "status": "pass",
    }


def _audit_prospective_experiment(
    repository: Path,
    registered: Mapping[str, Mapping[str, object]],
    spec: _ProspectiveSpec,
) -> dict[str, object]:
    policy = registered.get(spec.experiment_id)
    if policy is None:
        raise ValueError(f"{spec.experiment_id} is not registered")
    if (
        policy.get("contract_sha256") != spec.identity_sha256
        or policy.get("cutoff_filter_applied_to_frozen_campaign") is not False
        or policy.get("historical_snapshot_automatically_extended") is not False
        or policy.get("frozen_internal_fit_and_evaluation_permitted") is not True
    ):
        raise ValueError(f"{spec.experiment_id} isolation policy differs")
    validate_prospective_partition(
        experiment_id=spec.experiment_id,
        reusable_historical_training_eligible=bool(
            policy.get("reusable_historical_training_eligible")
        ),
    )
    path = repository / spec.path
    value = _load_json(path)
    if value.get("schema_version") != spec.schema_version:
        raise ValueError(f"{path} schema differs")
    _verify_identity(
        path=path,
        value=value,
        field=spec.identity_field,
        expected=spec.identity_sha256,
    )
    return {
        "experiment_id": spec.experiment_id,
        "identity_sha256": spec.identity_sha256,
        "path": spec.path,
        "reusable_historical_training_eligible": False,
        "status": "pass",
    }


def audit_research_data_snapshot(repository: str | Path) -> dict[str, object]:
    root = Path(repository).resolve(strict=True)
    contract = load_research_data_snapshot_contract(root)
    registered = {
        str(item["experiment_id"]): item
        for item in contract["registered_prospective_experiments"]
    }
    if set(registered) != {item.experiment_id for item in _PROSPECTIVE_EXPERIMENTS}:
        raise ValueError("prospective experiment registry differs")
    historical = [
        _audit_historical_artifact(root, spec) for spec in _HISTORICAL_ARTIFACTS
    ]
    prospective = [
        _audit_prospective_experiment(root, registered, spec)
        for spec in _PROSPECTIVE_EXPERIMENTS
    ]
    return {
        "authority": {
            "credentials_used": False,
            "database_opened": False,
            "network_used": False,
            "orders_submitted": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
        "cutoff_epoch_milliseconds_exclusive": RESEARCH_DATA_CUTOFF_MS,
        "historical_artifacts": historical,
        "prospective_experiments": prospective,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "snapshot_contract_sha256": RESEARCH_DATA_SNAPSHOT_CONTRACT_SHA256,
        "snapshot_id": RESEARCH_DATA_SNAPSHOT_ID,
        "status": "pass",
    }


__all__ = ["AUDIT_SCHEMA_VERSION", "audit_research_data_snapshot"]
