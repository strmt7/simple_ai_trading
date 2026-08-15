from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_campaign_admission import (
    build_round27_campaign_admission,
    validate_round27_campaign_admission,
)
from simple_ai_trading.polymarket_round27_feature_store import (
    POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round27_features import (
    POLYMARKET_ROUND27_FEATURE_NAMES,
    POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
    Round27FeatureRow,
)
from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)


ROOT = Path(__file__).resolve().parents[1]


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


def _population(
    counts: dict[str, int] | None = None,
) -> tuple[tuple[Round27FeatureRow, ...], dict[str, object]]:
    contract = load_round27_model_contract(ROOT)
    selected_counts = counts or {
        "train": 80,
        "calibration": 30,
        "selection": 95,
        "sealed": 95,
    }
    partitions = {
        str(item["role"]): item
        for item in contract["partitions"]
        if item["role"] in selected_counts
    }
    rows: list[Round27FeatureRow] = []
    by_slot: dict[str, list[Round27FeatureRow]] = {}
    index = 0
    for role in ("train", "calibration", "selection", "sealed"):
        interval = partitions[role]
        slot_id = str(interval["slot_id"])
        run_id = f"{slot_id}-run"
        for offset in range(selected_counts[role]):
            event_start_ms = int(interval["start_ms"]) + offset * 300_000
            row = Round27FeatureRow.create(
                run_id=run_id,
                condition_id="0x" + f"{index + 1:064x}",
                event_start_ms=event_start_ms,
                decision_time_ms=event_start_ms + 30_000,
                market_prior_probability=0.5,
                values=[0.0] * len(POLYMARKET_ROUND27_FEATURE_NAMES),
                maximum_receipt_wall_ms=event_start_ms + 29_999,
                source_chain_sha256=hashlib.sha256(
                    f"campaign-source-{index}".encode("ascii")
                ).hexdigest(),
            )
            rows.append(row)
            by_slot.setdefault(slot_id, []).append(row)
            index += 1
    slots = [
        {
            "schema_version": POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
            "slot_id": slot_id,
            "run_id": f"{slot_id}-run",
            "condition_audit_sha256": hashlib.sha256(
                f"{slot_id}-audit".encode("ascii")
            ).hexdigest(),
            "feature_report_sha256": hashlib.sha256(
                f"{slot_id}-report".encode("ascii")
            ).hexdigest(),
            "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
            "condition_count": len(slot_rows),
            "row_count": len(slot_rows),
            "row_chain_sha256": hashlib.sha256(
                f"{slot_id}-rows".encode("ascii")
            ).hexdigest(),
            "condition_manifest_sha256": [],
            "target_accessed": False,
            "trading_authority": False,
        }
        for slot_id, slot_rows in sorted(by_slot.items())
    ]
    audit: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_FEATURE_STORE_SCHEMA_VERSION,
        "slot_count": len(slots),
        "condition_count": len(rows),
        "row_count": len(rows),
        "feature_names_sha256": POLYMARKET_ROUND27_FEATURE_NAMES_SHA256,
        "slots": slots,
        "target_columns_present": False,
        "target_accessed": False,
        "trading_authority": False,
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    return tuple(rows), audit


def test_campaign_admission_requires_complete_target_free_primary_population() -> None:
    contract = load_round27_model_contract(ROOT)
    rows, audit = _population()
    admitted_at_ms = max(row.event_start_ms for row in rows) + 300_001

    admission = build_round27_campaign_admission(
        contract=contract,
        feature_store_audit=audit,
        feature_rows=rows,
        admitted_at_ms=admitted_at_ms,
    )

    assert admission["eligible_condition_count"] == 300
    assert admission["contingency_used"] is False
    assert admission["target_access_authorized"] is True
    assert (
        validate_round27_campaign_admission(
            admission,
            contract=contract,
            feature_store_audit_sha256=str(audit["audit_sha256"]),
        )
        == admission
    )


def test_campaign_admission_rejects_underfilled_total_and_role() -> None:
    contract = load_round27_model_contract(ROOT)
    rows, audit = _population(
        {"train": 79, "calibration": 30, "selection": 95, "sealed": 95}
    )
    with pytest.raises(ValueError, match="admission population differs"):
        build_round27_campaign_admission(
            contract=contract,
            feature_store_audit=audit,
            feature_rows=rows,
            admitted_at_ms=max(row.event_start_ms for row in rows) + 300_001,
        )

    rows, audit = _population(
        {"train": 74, "calibration": 31, "selection": 95, "sealed": 100}
    )
    with pytest.raises(ValueError, match="admission population differs"):
        build_round27_campaign_admission(
            contract=contract,
            feature_store_audit=audit,
            feature_rows=rows,
            admitted_at_ms=max(row.event_start_ms for row in rows) + 300_001,
        )


def test_campaign_admission_rejects_rehashed_wrong_feature_lineage() -> None:
    contract = load_round27_model_contract(ROOT)
    rows, audit = _population()
    admission = build_round27_campaign_admission(
        contract=contract,
        feature_store_audit=audit,
        feature_rows=rows,
        admitted_at_ms=max(row.event_start_ms for row in rows) + 300_001,
    )
    tampered = json.loads(json.dumps(admission))
    tampered["feature_store_audit_sha256"] = "0" * 64
    body = dict(tampered)
    body.pop("admission_sha256")
    tampered["admission_sha256"] = _canonical_sha256(body)

    with pytest.raises(ValueError, match="admission differs"):
        validate_round27_campaign_admission(
            tampered,
            contract=contract,
            feature_store_audit_sha256=str(audit["audit_sha256"]),
        )
