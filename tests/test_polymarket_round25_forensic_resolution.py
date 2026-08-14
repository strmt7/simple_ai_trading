from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading import polymarket_round25_forensic_resolution as resolution
from simple_ai_trading.polymarket_round25_forensic_materialization import (
    POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
)
from simple_ai_trading.polymarket_round25_forensic_partition import (
    POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
    partition_round25_forensic_conditions,
)
from simple_ai_trading.polymarket_round25_campaign import (
    POLYMARKET_ROUND25_RESOLUTION_SOURCE,
)
from simple_ai_trading.polymarket_round25_joint_materialization import (
    Round25JointReceiptCondition,
)
from simple_ai_trading.polymarket_round25_resolution_store import (
    Round25OfficialPublicPayload,
)


START = 1_786_515_300_000
MANIFEST_SHA = "a" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _conditions() -> tuple[Round25JointReceiptCondition, ...]:
    return tuple(
        Round25JointReceiptCondition(
            run_id="1" * 32,
            segment_index=0,
            snapshot_sha256=f"{index + 1:064x}",
            snapshot_observed_wall_ms=START + index * 300_000 - 1,
            market_id=str(1000 + index),
            condition_id=f"0x{index + 1:064x}",
            slug=f"btc-updown-5m-{START // 1_000 + index * 300}",
            event_start_ms=START + index * 300_000,
            event_end_ms=START + (index + 1) * 300_000,
            up_token_id=str(100_000 + index),
            down_token_id=str(200_000 + index),
            resolution_source=POLYMARKET_ROUND25_RESOLUTION_SOURCE,
            role="train",
        ).validated()
        for index in range(50)
    )


def _partition() -> dict[str, object]:
    assigned = partition_round25_forensic_conditions(
        tuple((item.condition_id, item.event_start_ms) for item in _conditions())
    )
    body = {
        "condition_count": 50,
        "conditions": [
            {"condition_id": condition_id, "event_start_ms": start, "role": role}
            for condition_id, start, role in assigned
        ],
        "created_at_ms": START + 51 * 300_000,
        "feature_store_manifest_sha256": MANIFEST_SHA,
        "live_trading_authority": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "role_counts": {"train": 29, "calibration": 8, "selection": 9, "purged": 4},
        "salvage_contract_sha256": POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
        "selection_accessed": False,
        "target_accessed": False,
    }
    return {**body, "partition_sha256": _sha(body)}


def _payload(value: Mapping[str, object], observed: int) -> Round25OfficialPublicPayload:
    canonical = _canonical_json(value)
    return Round25OfficialPublicPayload(
        value=dict(value),
        canonical_json=canonical,
        sha256=hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        observed_wall_ms=observed,
        observed_monotonic_ns=observed * 1_000_000,
    )


class _Client:
    def __init__(self, conditions: tuple[Round25JointReceiptCondition, ...]) -> None:
        self.by_market = {item.market_id: item for item in conditions}
        self.by_condition = {item.condition_id: item for item in conditions}

    def gamma_market(self, market_id: str) -> Round25OfficialPublicPayload:
        item = self.by_market[market_id]
        return _payload(
            {
                "acceptingOrders": False,
                "closed": True,
                "clobTokenIds": [item.up_token_id, item.down_token_id],
                "conditionId": item.condition_id,
                "id": item.market_id,
                "outcomePrices": ["1", "0"],
                "outcomes": ["Up", "Down"],
                "resolutionSource": item.resolution_source,
                "slug": item.slug,
            },
            START + 100 * 300_000,
        )

    def clob_market(self, condition_id: str) -> Round25OfficialPublicPayload:
        item = self.by_condition[condition_id]
        return _payload(
            {
                "accepting_orders": False,
                "closed": True,
                "condition_id": item.condition_id,
                "market_slug": item.slug,
                "tokens": [
                    {"outcome": "Up", "price": "1", "token_id": item.up_token_id, "winner": True},
                    {"outcome": "Down", "price": "0", "token_id": item.down_token_id, "winner": False},
                ],
            },
            START + 100 * 300_000 + 1,
        )


def test_fit_and_selection_access_are_physically_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditions = _conditions()
    monkeypatch.setattr(
        resolution,
        "audit_round25_joint_store",
        lambda _path: {"manifest_sha256": MANIFEST_SHA},
    )
    monkeypatch.setattr(
        resolution,
        "load_round25_joint_condition_identities",
        lambda _path: conditions,
    )
    partition = _partition()
    fit = tmp_path / "fit.duckdb"
    _, fit_claim = resolution.initialize_round25_forensic_resolution_collection(
        feature_database=tmp_path / "feature.duckdb",
        partition_manifest=partition,
        destination_database=fit,
        stage="fit",
        created_at_ms=START + 100 * 300_000,
    )
    assert fit_claim["role_counts"] == {"calibration": 8, "train": 29}
    _, resumed_claim = resolution.initialize_round25_forensic_resolution_collection(
        feature_database=tmp_path / "feature.duckdb",
        partition_manifest=partition,
        destination_database=fit,
        stage="fit",
        created_at_ms=START + 101 * 300_000,
    )
    assert resumed_claim == fit_claim
    assert resolution.collect_round25_forensic_resolutions_once(
        collection_database=fit,
        client=_Client(conditions),
    )["complete"] is True
    _, targets = resolution.load_round25_forensic_resolution_targets(fit)
    assert len(targets) == 37
    assert {item.role for item in targets} == {"train", "calibration"}

    with pytest.raises(ValueError, match="predictions are not frozen"):
        resolution.initialize_round25_forensic_resolution_collection(
            feature_database=tmp_path / "feature.duckdb",
            partition_manifest=partition,
            destination_database=tmp_path / "selection.duckdb",
            stage="selection",
            created_at_ms=START + 100 * 300_000,
        )

    freeze_body = {
        "condition_count": 9,
        "created_at_ms": START + 52 * 300_000,
        "evaluation_contract_sha256": resolution.POLYMARKET_ROUND25_FORENSIC_EVALUATION_CONTRACT_SHA256,
        "feature_store_manifest_sha256": MANIFEST_SHA,
        "partition_sha256": partition["partition_sha256"],
        "prediction_population_sha256": "b" * 64,
        "profitability_claim": False,
        "schema_version": resolution.POLYMARKET_ROUND25_FORENSIC_SELECTION_FREEZE_SCHEMA_VERSION,
        "selected_candidate_id": "market-prior-v1",
        "selection_predictions_frozen": True,
        "trade_policy_sha256": "c" * 64,
    }
    freeze = {**freeze_body, "freeze_sha256": _sha(freeze_body)}
    selection = tmp_path / "selection.duckdb"
    _, selection_claim = resolution.initialize_round25_forensic_resolution_collection(
        feature_database=tmp_path / "feature.duckdb",
        partition_manifest=partition,
        destination_database=selection,
        stage="selection",
        created_at_ms=START + 100 * 300_000,
        selection_freeze=freeze,
    )
    assert selection_claim["role_counts"] == {"selection": 9}
    assert selection_claim["selection_freeze_sha256"] == freeze["freeze_sha256"]
