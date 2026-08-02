from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_round21_sidecar import (
    POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
    create_round21_sidecar_manifest,
)
from simple_ai_trading.polymarket_round21_sidecar_campaign import (
    POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256,
    Round21SidecarCampaignConfig,
    build_round21_sidecar_segment_manifest,
    create_round21_sidecar_campaign_plan,
    validate_round21_legacy_sidecar_state,
    validate_round21_sidecar_campaign_plan,
    validate_round21_sidecar_segment_manifest,
    write_round21_sidecar_campaign_plan,
)


START_MS = 1_785_454_800_000
LEGACY_STARTED_MS = START_MS + 1_000
LEGACY_OBSERVED_MS = START_MS + 18_300_000
CREATED_MS = LEGACY_OBSERVED_MS + 210_000_000
LEGACY_RUN_ID = "a" * 32
DESIGN_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-binance-sidecar-campaign-design-v2.json"
)


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


def _rehash(value: dict[str, object], hash_name: str) -> dict[str, object]:
    body = dict(value)
    body.pop(hash_name, None)
    body[hash_name] = _canonical_sha256(body)
    return body


def _required_files() -> dict[str, str]:
    from simple_ai_trading import polymarket_round21_sidecar_campaign as campaign

    return {
        name: hashlib.sha256(name.encode("ascii")).hexdigest()
        for name in campaign._REQUIRED_FILES
    }


def _legacy_required_files() -> dict[str, str]:
    from simple_ai_trading import polymarket_round21_sidecar as sidecar

    return {
        name: hashlib.sha256(name.encode("ascii")).hexdigest()
        for name in sidecar._REQUIRED_FILES
    }


def _legacy_state() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "polymarket-round21-binance-sidecar-state-v1",
        "phase": "capturing",
        "observed_at_ms": LEGACY_OBSERVED_MS,
        "round21_contract_sha256": (
            "6aadbce31c175438c40c6a1204383d828fd78ddef93b280aa2f999f347669116"
        ),
        "sidecar_design_sha256": (
            "c802b13e169f868c7a37619669cdc957862a1cb58c6d3299c0aae63ff0d86d4a"
        ),
        "database_bytes": 100,
        "wal_bytes": 10,
        "free_bytes": 300 * 1024**3,
        "details": {
            "schema_version": "polymarket-recorder-progress-v1",
            "phase": "capturing",
            "run_id": LEGACY_RUN_ID,
            "duration_seconds": 30 * 86_400 - 1_000,
            "elapsed_seconds": 18_300.0,
            "received_message_count": 8_000_000,
            "written_message_count": 7_999_990,
            "written_gap_count": 5,
            "error_count": 0,
        },
        "model_data_eligible": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _plan_value() -> dict[str, object]:
    return create_round21_sidecar_campaign_plan(
        created_at_ms=CREATED_MS,
        scheduled_start_ms=START_MS,
        legacy_state=_legacy_state(),
        repository_commit_oid="b" * 40,
        repository_tree_oid="c" * 40,
        repository_file_sha256=_required_files(),
    )


def test_sidecar_campaign_design_is_canonical_and_target_blind() -> None:
    value = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    claimed = value.pop("design_sha256")

    assert claimed == POLYMARKET_ROUND21_SIDECAR_CAMPAIGN_DESIGN_SHA256
    assert claimed == _canonical_sha256(value)
    assert value["incident"]["legacy_run_preserved_as_segment_zero"]
    assert not any(value["authority"].values())


def test_legacy_state_is_exactly_bound_and_non_authoritative() -> None:
    state = validate_round21_legacy_sidecar_state(_legacy_state())

    assert state["details"]["run_id"] == LEGACY_RUN_ID
    assert state["details"]["error_count"] == 0
    assert not state["model_data_eligible"]

    changed = _legacy_state()
    changed["profitability_claim"] = True
    with pytest.raises(ValueError, match="legacy sidecar state differs"):
        validate_round21_legacy_sidecar_state(
            _rehash(changed, "artifact_sha256")
        )

    changed = _legacy_state()
    changed["unregistered"] = False
    with pytest.raises(ValueError, match="legacy sidecar state differs"):
        validate_round21_legacy_sidecar_state(
            _rehash(changed, "artifact_sha256")
        )


def test_campaign_plan_binds_interrupted_run_and_original_window() -> None:
    raw = _plan_value()
    plan = validate_round21_sidecar_campaign_plan(raw)

    assert plan.legacy_run_id == LEGACY_RUN_ID
    assert plan.scheduled_start_ms == START_MS
    assert plan.scheduled_end_ms == POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS
    assert raw["legacy_interruption"] == "host_reboot"
    assert raw["required_streams"] == ["binance_futures", "binance_spot"]
    assert raw["execution_connected"] is False


def test_campaign_plan_rejects_rehashed_scope_or_authority_drift() -> None:
    for key, value in (
        ("required_streams", ["binance_spot"]),
        ("execution_connected", True),
        ("model_scores_accessed", True),
        ("scheduled_end_ms", POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS + 1),
    ):
        changed = _plan_value()
        changed[key] = value
        with pytest.raises(ValueError, match="campaign plan differs"):
            validate_round21_sidecar_campaign_plan(
                _rehash(changed, "plan_sha256")
            )


def test_segment_manifest_is_bounded_optional_and_non_authoritative() -> None:
    plan = validate_round21_sidecar_campaign_plan(_plan_value())
    manifest = build_round21_sidecar_segment_manifest(
        plan,
        run_id="d" * 32,
        created_at_ms=CREATED_MS + 1,
        duration_seconds=1_200,
        segment_index=1,
    )
    validated = validate_round21_sidecar_segment_manifest(manifest, plan)

    assert validated["segment_index"] == 1
    assert validated["cross_segment_state_carry"] is False
    assert validated["required_assets"] == []
    assert validated["binance_credentials_used"] is False
    assert validated["live_trading_authority"] is False


def test_segment_manifest_rejects_rehashed_state_carry_or_execution() -> None:
    plan = validate_round21_sidecar_campaign_plan(_plan_value())
    original = build_round21_sidecar_segment_manifest(
        plan,
        run_id="d" * 32,
        created_at_ms=CREATED_MS + 1,
        duration_seconds=1_200,
        segment_index=1,
    )
    for key, value in (
        ("cross_segment_state_carry", True),
        ("binance_execution_connected", True),
        ("segment_index", 0),
        ("unregistered", False),
    ):
        changed = dict(original)
        changed[key] = value
        with pytest.raises(ValueError, match="segment manifest differs"):
            validate_round21_sidecar_segment_manifest(
                _rehash(changed, "manifest_sha256"),
                plan,
            )


def test_orphan_recovery_terminalizes_v1_run_and_preserves_result(
    tmp_path: Path,
) -> None:
    from simple_ai_trading import polymarket_round21_sidecar_campaign as campaign

    repository = tmp_path / "repository"
    repository.mkdir()
    plan_path = repository / "data" / "plan.json"
    plan_path.parent.mkdir()
    database = repository / "data" / "sidecar.duckdb"
    state_root = repository / "data" / "state"
    raw_plan = _plan_value()
    write_round21_sidecar_campaign_plan(plan_path, raw_plan)
    plan = validate_round21_sidecar_campaign_plan(raw_plan)
    legacy_manifest = create_round21_sidecar_manifest(
        run_id=LEGACY_RUN_ID,
        created_at_ms=LEGACY_STARTED_MS,
        capture_duration_seconds=30 * 86_400 - 1_000,
        scheduled_end_ms=POLYMARKET_ROUND21_SIDECAR_SCHEDULED_END_MS,
        repository_commit_oid="e" * 40,
        repository_tree_oid="f" * 40,
        repository_file_sha256=_legacy_required_files(),
    )
    with PolymarketEvidenceStore(database, memory_limit="256MB", threads=1) as store:
        store.start_run(
            LEGACY_RUN_ID,
            LEGACY_STARTED_MS,
            preregistration_manifest=legacy_manifest,
        )
    config = Round21SidecarCampaignConfig(
        repository=repository,
        plan_path=plan_path,
        database_path=database,
        state_root=state_root,
    )

    recovered = campaign._recover_orphaned_segments(
        config,
        plan,
        first_segment_index=0,
    )

    assert recovered == 1
    result = json.loads(
        (state_root / "segments" / "segment-0000.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "interrupted"
    assert result["details"]["run_id"] == LEGACY_RUN_ID
    assert result["details"]["errors"] == [
        "host_or_process_restart_interrupted_segment"
    ]
    with PolymarketEvidenceStore(
        database,
        memory_limit="256MB",
        threads=1,
        read_only=True,
    ) as store:
        row = store.connect().execute(
            "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
            [LEGACY_RUN_ID],
        ).fetchone()
    assert row == ("failed",)


def test_orphan_recovery_fails_if_bound_legacy_run_is_absent(
    tmp_path: Path,
) -> None:
    from simple_ai_trading import polymarket_round21_sidecar_campaign as campaign

    repository = tmp_path / "repository"
    repository.mkdir()
    plan_path = repository / "plan.json"
    database = repository / "sidecar.duckdb"
    state_root = repository / "state"
    raw_plan = _plan_value()
    write_round21_sidecar_campaign_plan(plan_path, raw_plan)
    plan = validate_round21_sidecar_campaign_plan(raw_plan)
    with PolymarketEvidenceStore(database, memory_limit="256MB", threads=1):
        pass
    config = Round21SidecarCampaignConfig(
        repository=repository,
        plan_path=plan_path,
        database_path=database,
        state_root=state_root,
    )

    with pytest.raises(ValueError, match="legacy sidecar orphan was not found"):
        campaign._recover_orphaned_segments(
            config,
            plan,
            first_segment_index=0,
        )


def test_failed_capture_reconciles_started_segment_before_result(
    tmp_path: Path,
) -> None:
    from simple_ai_trading import polymarket_round21_sidecar_campaign as campaign

    repository = tmp_path / "repository"
    repository.mkdir()
    plan_path = repository / "plan.json"
    database = repository / "sidecar.duckdb"
    state_root = repository / "state"
    raw_plan = _plan_value()
    write_round21_sidecar_campaign_plan(plan_path, raw_plan)
    plan = validate_round21_sidecar_campaign_plan(raw_plan)
    state_root.joinpath("segments").mkdir(parents=True)
    campaign._write_segment_result(
        state_root,
        plan=plan,
        segment_index=0,
        status="interrupted",
        details={"run_id": LEGACY_RUN_ID},
    )
    run_id = "d" * 32
    manifest = build_round21_sidecar_segment_manifest(
        plan,
        run_id=run_id,
        created_at_ms=CREATED_MS + 1,
        duration_seconds=1_200,
        segment_index=1,
    )
    with PolymarketEvidenceStore(database, memory_limit="256MB", threads=1) as store:
        store.start_run(
            run_id,
            CREATED_MS + 1,
            preregistration_manifest=manifest,
        )
    config = Round21SidecarCampaignConfig(
        repository=repository,
        plan_path=plan_path,
        database_path=database,
        state_root=state_root,
    )

    result = campaign._reconcile_failed_capture(
        config,
        plan,
        segment_index=1,
        error=RuntimeError("capture transport failed"),
    )

    assert result["status"] == "interrupted"
    assert result["details"]["run_id"] == run_id
    assert result["details"]["errors"] == [
        "host_or_process_restart_interrupted_segment"
    ]
    with PolymarketEvidenceStore(
        database,
        memory_limit="256MB",
        threads=1,
        read_only=True,
    ) as store:
        row = store.connect().execute(
            "SELECT status FROM polymarket_recorder_run WHERE run_id = ?",
            [run_id],
        ).fetchone()
    assert row == ("failed",)


def test_failed_capture_without_started_run_records_exact_failure(
    tmp_path: Path,
) -> None:
    from simple_ai_trading import polymarket_round21_sidecar_campaign as campaign

    repository = tmp_path / "repository"
    repository.mkdir()
    plan_path = repository / "plan.json"
    database = repository / "sidecar.duckdb"
    state_root = repository / "state"
    raw_plan = _plan_value()
    write_round21_sidecar_campaign_plan(plan_path, raw_plan)
    plan = validate_round21_sidecar_campaign_plan(raw_plan)
    state_root.joinpath("segments").mkdir(parents=True)
    campaign._write_segment_result(
        state_root,
        plan=plan,
        segment_index=0,
        status="interrupted",
        details={"run_id": LEGACY_RUN_ID},
    )
    with PolymarketEvidenceStore(database, memory_limit="256MB", threads=1):
        pass
    config = Round21SidecarCampaignConfig(
        repository=repository,
        plan_path=plan_path,
        database_path=database,
        state_root=state_root,
    )

    result = campaign._reconcile_failed_capture(
        config,
        plan,
        segment_index=1,
        error=RuntimeError("failed before run creation"),
    )

    assert result["status"] == "failed"
    assert result["details"] == {
        "failure_type": "RuntimeError",
        "failure": "failed before run creation",
    }

    changed = dict(result)
    changed["optional_feature_admission_pending"] = True
    changed = _rehash(changed, "artifact_sha256")
    (state_root / "segments" / "segment-0001.json").write_text(
        json.dumps(changed),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="segment result set differs"):
        campaign._segment_results(state_root, plan)


def test_campaign_config_rejects_database_inside_state_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    plan = repository / "plan.json"
    plan.write_text("{}", encoding="ascii")
    config = Round21SidecarCampaignConfig(
        repository=repository,
        plan_path=plan,
        database_path=repository / "state" / "sidecar.duckdb",
        state_root=repository / "state",
    )

    with pytest.raises(ValueError, match="configuration differs"):
        config.validate()
