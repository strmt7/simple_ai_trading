"""Chronological target-blind partitions for the Round 25 forensic diagnostic."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time

from .polymarket_round25_forensic_materialization import (
    POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
    validate_round25_forensic_sources,
)
from .polymarket_round25_joint_store import load_round25_joint_endpoint_inputs


POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION = (
    "polymarket-round25-forensic-diagnostic-partition-v1"
)
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")


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


def partition_round25_forensic_conditions(
    conditions: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int, str], ...]:
    """Assign chronological roles with one condition purged on each boundary side."""

    selected = tuple(conditions)
    if (
        len(selected) < 50
        or len(set(selected)) != len(selected)
        or any(
            _CONDITION_ID.fullmatch(condition_id) is None
            or type(event_start_ms) is not int
            or event_start_ms <= 0
            or event_start_ms % 300_000
            for condition_id, event_start_ms in selected
        )
        or selected
        != tuple(sorted(selected, key=lambda item: (item[1], item[0])))
    ):
        raise ValueError("Round 25 forensic partition population differs")
    count = len(selected)
    train_boundary = count * 6 // 10
    calibration_boundary = count * 8 // 10
    purge = {
        train_boundary - 1,
        train_boundary,
        calibration_boundary - 1,
        calibration_boundary,
    }
    output: list[tuple[str, int, str]] = []
    for index, (condition_id, event_start_ms) in enumerate(selected):
        if index in purge:
            role = "purged"
        elif index < train_boundary:
            role = "train"
        elif index < calibration_boundary:
            role = "calibration"
        else:
            role = "selection"
        output.append((condition_id, event_start_ms, role))
    counts = Counter(role for _, _, role in output)
    if (
        counts["train"] < 1
        or counts["calibration"] < 1
        or counts["selection"] < 8
        or counts["purged"] != 4
    ):
        raise ValueError("Round 25 forensic partition support is insufficient")
    return tuple(output)


def build_round25_forensic_partition_manifest(
    *,
    feature_store: str | Path,
    forensic_audit: Mapping[str, object],
    salvage_contract: Mapping[str, object],
    observed_at_ms: int | None = None,
) -> dict[str, object]:
    """Deep-audit the frozen feature store, then bind diagnostic roles."""

    _audit, contract = validate_round25_forensic_sources(
        forensic_audit=forensic_audit,
        salvage_contract=salvage_contract,
    )
    store_manifest, endpoints = load_round25_joint_endpoint_inputs(feature_store)
    if (
        store_manifest.get("diagnostic_only") is not True
        or store_manifest.get("salvage_contract_sha256")
        != contract["contract_sha256"]
        or endpoints["calibration"]
        or endpoints["selection"]
    ):
        raise ValueError("Round 25 forensic feature store provenance differs")
    grouped: dict[str, tuple[int, int]] = {}
    for row in endpoints["train"]:
        prior = grouped.get(row.condition_id)
        identity = (row.event_start_ms, 1 if prior is None else prior[1] + 1)
        if prior is not None and prior[0] != row.event_start_ms:
            raise ValueError("Round 25 forensic endpoint condition differs")
        grouped[row.condition_id] = identity
    if any(endpoint_count != 16 for _, endpoint_count in grouped.values()):
        raise ValueError("Round 25 forensic endpoint count differs")
    population = tuple(
        sorted(
            (
                (condition_id, event_start_ms)
                for condition_id, (event_start_ms, _count) in grouped.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
    )
    partition = partition_round25_forensic_conditions(population)
    role_counts = Counter(role for _, _, role in partition)
    rows = [
        {
            "condition_id": condition_id,
            "event_start_ms": event_start_ms,
            "role": role,
        }
        for condition_id, event_start_ms, role in partition
    ]
    body: dict[str, object] = {
        "condition_count": len(rows),
        "conditions": rows,
        "created_at_ms": (
            int(observed_at_ms)
            if observed_at_ms is not None
            else int(time.time() * 1_000)
        ),
        "feature_store_manifest_sha256": store_manifest["manifest_sha256"],
        "live_trading_authority": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "role_counts": {
            role: role_counts[role]
            for role in ("train", "calibration", "selection", "purged")
        },
        "salvage_contract_sha256": POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
        "selection_accessed": False,
        "target_accessed": False,
    }
    return {**body, "partition_sha256": _canonical_sha256(body)}


__all__ = [
    "POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION",
    "build_round25_forensic_partition_manifest",
    "partition_round25_forensic_conditions",
]
