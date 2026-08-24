from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "prospective-source-continuity-recovery-design-v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_source_continuity_recovery_design_reconstructs_every_binding() -> None:
    value = json.loads(DESIGN.read_text(encoding="ascii"))
    canonical = dict(value)
    claimed = canonical.pop("artifact_sha256")

    assert value["schema_version"] == "prospective-source-continuity-recovery-design-v1"
    assert value["status"] == "implementation_ready_design_only_not_activated"
    assert claimed == _canonical_sha256(canonical)
    assert all(item is False for item in value["authority"].values())
    assert all(
        value["activation"][key] is False
        for key in (
            "capture_started",
            "exact_schedule_frozen",
            "host_preflight_completed",
            "scheduled_task_created",
        )
    )

    implementation = value["implementation"]
    for kind in ("module", "test"):
        assert (
            _sha256(ROOT / implementation[f"{kind}_path"])
            == implementation[f"{kind}_file_sha256"]
        )

    cross_regime = value["cross_regime_policy"]
    cross_regime_value = json.loads(
        (ROOT / cross_regime["contract_path"]).read_text(encoding="ascii")
    )
    assert (
        _sha256(ROOT / cross_regime["contract_path"])
        == cross_regime["contract_file_sha256"]
    )
    assert cross_regime_value["contract_sha256"] == cross_regime["contract_sha256"]
    assert cross_regime["unsupported_slice_action"] == "abstain_from_new_exposure"

    for source in value["lineage"].values():
        source_value = json.loads((ROOT / source["path"]).read_text(encoding="ascii"))
        assert _sha256(ROOT / source["path"]) == source["file_sha256"]
        assert source_value["artifact_sha256"] == source["artifact_sha256"]
        assert source["reuse_permitted"] is False


def test_recovery_design_keeps_venues_and_failed_storage_independent() -> None:
    value = json.loads(DESIGN.read_text(encoding="ascii"))
    policy = value["common_source_policy"]
    venues = value["venue_designs"]

    assert set(venues) == {"binance", "polymarket"}
    assert venues["binance"]["strategy_capital_ledger"] == "binance_only"
    assert venues["polymarket"]["strategy_capital_ledger"] == "polymarket_only"
    assert venues["binance"]["failed_round75_shared_shards_permitted"] is False
    assert venues["polymarket"]["failed_round27_stage1_storage_permitted"] is False
    assert policy["one_database_namespace_per_slot"] is True
    assert policy["failed_slot_storage_must_be_quarantined"] is True
    assert policy["quarantined_failed_wal_may_poison_a_later_unique_slot"] is False
    assert policy["source_pass_grants_model_or_target_access"] is False
    assert policy["adaptive_replacement_or_time_shift_permitted"] is False
