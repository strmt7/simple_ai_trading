from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round25_coordinator as coordinator
from simple_ai_trading.polymarket_round25_coordinator import (
    POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256,
    Round25CoordinatorPaths,
    advance_round25_post_capture,
    load_round25_coordinator_state,
    validate_round25_coordinator_state,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-025-post-capture-coordinator-contract-v2.json"
)
SOURCE_COMMIT = "a" * 40
FEATURE_SHA = "b" * 64
RESOLUTION_SHA = "c" * 64
LEDGER_SHA = "d" * 64
PREPARED_SHA = "e" * 64
RESULT_SHA = "f" * 64
CLAIM_SHA = "1" * 64


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


def _paths(tmp_path: Path) -> Round25CoordinatorPaths:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = tmp_path / "source.duckdb"
    source.write_bytes(b"source")
    return Round25CoordinatorPaths(
        repository=repository,
        source_database=source,
        feature_database=tmp_path / "features.duckdb",
        resolution_database=tmp_path / "resolutions.duckdb",
        model_ledger=tmp_path / "model-ledger.json",
        prepared_prediction=tmp_path / "prepared-prediction.json",
        selection_access_store=tmp_path / "selection-access.sqlite3",
        predictive_result=tmp_path / "predictive-result.json",
        state=tmp_path / "coordinator-state.json",
        lock=tmp_path / "coordinator.lock",
    ).validated()


def test_coordinator_contract_is_self_hashed_and_has_no_trading_authority() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    claimed = payload.pop("contract_sha256")

    assert claimed == _canonical_sha256(payload)
    assert claimed == POLYMARKET_ROUND25_COORDINATOR_CONTRACT_SHA256
    assert payload["leakage_control"][
        "selection_resolution_query_before_prediction_freeze_allowed"
    ] is False
    assert payload["completion_boundary"]["profitability_verified"] is False
    assert payload["completion_boundary"]["order_submission_allowed"] is False


def test_coordinator_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    lock = tmp_path / "coordinator.lock"

    with coordinator._coordinator_lock(lock):
        with pytest.raises(RuntimeError, match="already running"):
            with coordinator._coordinator_lock(lock):
                pytest.fail("a second coordinator acquired the same lock")


def test_coordinator_returns_bounded_pending_state_before_any_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    events: list[str] = []
    collection = tmp_path / ".resolutions.duckdb.collecting"
    monkeypatch.setattr(
        coordinator,
        "validate_round25_terminal_transport_manifest",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        coordinator,
        "materialize_round25_joint_feature_store",
        lambda **_kwargs: (
            {"manifest_sha256": FEATURE_SHA},
            {"audit_sha256": "2" * 64},
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "initialize_round25_resolution_collection",
        lambda **_kwargs: (collection, {"claim_sha256": CLAIM_SHA}),
    )
    monkeypatch.setattr(
        coordinator,
        "collect_round25_resolutions_once",
        lambda **_kwargs: {
            "finalization_ready": False,
            "pending_condition_count": 7,
            "report_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "load_round25_joint_endpoint_inputs",
        lambda *_args, **_kwargs: pytest.fail("model inputs opened while pending"),
    )

    state = advance_round25_post_capture(
        paths=paths,
        terminal_transport_manifest={"fixture": True},
        source_commit_oid=SOURCE_COMMIT,
        resolution_client=object(),
        observed_at_ms=1_900_000_000_000,
        progress=lambda stage, _details: events.append(stage),
    )

    assert state["phase"] == "resolution_collection_pending"
    assert state["resolution_pending_count"] == 7
    assert state["model_ledger_sha256"] is None
    assert state["predictive_result_sha256"] is None
    assert load_round25_coordinator_state(paths.state) == state
    assert events == [
        "coordinator_started",
        "feature_materialized",
        "resolution_collection_pending",
    ]


def test_coordinator_recovers_existing_artifacts_and_completes_one_use_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    paths.feature_database.write_bytes(b"features")
    paths.resolution_database.write_bytes(b"resolutions")
    paths.model_ledger.write_text("ledger", encoding="ascii")
    paths.prepared_prediction.write_text("prediction", encoding="ascii")
    panel = object()
    ledger = SimpleNamespace(
        source_commit_oid=SOURCE_COMMIT,
        implementation_sha256=(("source.py", "2" * 64),),
        ledger_sha256=LEDGER_SHA,
    )
    prepared = SimpleNamespace(
        model_ledger_sha256=LEDGER_SHA,
        prepared_sha256=PREPARED_SHA,
        panel=panel,
    )
    authority = object()
    selection_dataset = object()
    receipt = object()
    result = SimpleNamespace(
        result_sha256=RESULT_SHA,
        predictive_gate_passed=True,
    )

    class AccessStore:
        def __init__(self, _path: Path) -> None:
            pass

        def validate_prediction_binding(self, *, panel: object) -> tuple[str, str]:
            assert panel is prepared.panel
            return "prediction_panel_frozen", CLAIM_SHA

        def consume_target_access(self, **_kwargs: object) -> object:
            return receipt

    monkeypatch.setattr(
        coordinator,
        "validate_round25_terminal_transport_manifest",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        coordinator,
        "audit_round25_joint_store",
        lambda _path: {
            "manifest_sha256": FEATURE_SHA,
            "terminal_receipt_audit_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        coordinator,
        "audit_round25_resolution_store",
        lambda _path: {"manifest_sha256": RESOLUTION_SHA},
    )
    monkeypatch.setattr(
        coordinator,
        "load_round25_joint_endpoint_inputs",
        lambda _path: (
            {"manifest_sha256": FEATURE_SHA},
            {"train": (), "calibration": (), "selection": (object(),)},
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "load_round25_model_ledger",
        lambda _path: ledger,
    )
    monkeypatch.setattr(
        coordinator,
        "_verified_source_identity",
        lambda *_args, **_kwargs: ledger.implementation_sha256,
    )
    monkeypatch.setattr(
        coordinator,
        "load_round25_prepared_prediction",
        lambda _path: prepared,
    )
    monkeypatch.setattr(coordinator, "Round25SelectionAccessStore", AccessStore)
    monkeypatch.setattr(
        coordinator,
        "load_round25_selection_resolution_inputs",
        lambda *_args, **_kwargs: (
            {"manifest_sha256": RESOLUTION_SHA},
            authority,
            (object(),),
            CLAIM_SHA,
        ),
    )
    monkeypatch.setattr(
        coordinator,
        "_role_dataset",
        lambda **_kwargs: selection_dataset,
    )
    monkeypatch.setattr(
        coordinator,
        "evaluate_round25_predictive_candidates",
        lambda **_kwargs: result,
    )

    def write_result(path: Path, observed: object) -> Path:
        assert observed is result
        path.write_text("result", encoding="ascii")
        return path

    monkeypatch.setattr(coordinator, "write_round25_predictive_result", write_result)

    state = advance_round25_post_capture(
        paths=paths,
        terminal_transport_manifest={"fixture": True},
        source_commit_oid=SOURCE_COMMIT,
        observed_at_ms=1_900_000_000_000,
    )

    assert state["phase"] == "predictive_evaluation_complete"
    assert state["selection_access_status"] == "target_access_consumed"
    assert state["predictive_result_sha256"] == RESULT_SHA
    assert state["predictive_gate_passed"] is True
    assert state["profitability_claim"] is False
    assert state["live_trading_authority"] is False


def test_coordinator_state_rejects_hash_tampering() -> None:
    body = coordinator._state_body(
        phase="feature_materialized",
        updated_at_ms=1,
        feature_store_manifest_sha256=FEATURE_SHA,
    )
    state = {**body, "state_sha256": _canonical_sha256(body)}
    assert validate_round25_coordinator_state(state)["phase"] == "feature_materialized"

    state["resolution_pending_count"] = 1
    with pytest.raises(ValueError, match="state differs"):
        validate_round25_coordinator_state(state)
