from __future__ import annotations

from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_round12_capture import (
    create_round12_capture_manifest,
    load_round12_capture_manifest,
    validate_round12_capture_manifest_payload,
)
from simple_ai_trading.polymarket_round12_reference import (
    load_round12_confirmation_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/round-012-fixed-calibration-confirmation-contract.json"
)
STARTED_AT_MS = 1_700_000_000_000


def _contract() -> dict[str, object]:
    return dict(load_round12_confirmation_contract(CONTRACT_PATH))


def _manifest(run_id: str) -> dict[str, object]:
    contract = _contract()
    implementation = contract["implementation"]
    model = contract["model_contract"]
    policy = contract["primary_policy"]
    return create_round12_capture_manifest(
        run_id=run_id,
        started_at_ms=STARTED_AT_MS,
        repository_commit="b" * 40,
        contract_sha256=str(contract["contract_sha256"]),
        model_sha256=str(model["model_sha256"]),
        policy_sha256=str(policy["policy_sha256"]),
        reference_implementation_sha256=str(
            implementation["reference_implementation_sha256"]
        ),
        action_pipeline_implementation_sha256=str(
            implementation["action_pipeline_implementation_sha256"]
        ),
        required_git_blob_sha256={"tests/frozen-evidence.py": "a" * 64},
    )


def test_round12_capture_manifest_is_canonical_and_contract_bound() -> None:
    manifest = _manifest("capture-a")

    loaded = validate_round12_capture_manifest_payload(
        manifest,
        expected_run_id="capture-a",
        expected_contract=_contract(),
    )

    assert loaded == manifest
    assert loaded["capture_started_before_manifest"] is False
    assert loaded["labels_consulted"] is False


def test_round12_capture_manifest_rejects_tampering() -> None:
    manifest = _manifest("capture-b")
    manifest["repository_commit"] = "c" * 40

    with pytest.raises(ValueError, match="capture manifest is invalid"):
        validate_round12_capture_manifest_payload(manifest)


def test_round12_capture_manifest_is_persisted_with_run_start(tmp_path: Path) -> None:
    database = tmp_path / "capture.duckdb"
    manifest = _manifest("capture-c")
    with PolymarketEvidenceStore(database) as store:
        store.start_run(
            "capture-c",
            STARTED_AT_MS,
            preregistration_manifest=manifest,
        )

        loaded = load_round12_capture_manifest(
            store,
            run_id="capture-c",
            contract=_contract(),
        )

    assert loaded == manifest


def test_invalid_manifest_rolls_back_recorder_start(tmp_path: Path) -> None:
    database = tmp_path / "rollback.duckdb"
    manifest = _manifest("capture-d")
    manifest["run_id"] = "another-run"
    with PolymarketEvidenceStore(database) as store:
        with pytest.raises(ValueError, match="preregistration manifest is invalid"):
            store.start_run(
                "capture-d",
                STARTED_AT_MS,
                preregistration_manifest=manifest,
            )
        count = store.connect().execute(
            "SELECT count(*) FROM polymarket_recorder_run"
        ).fetchone()[0]

    assert count == 0


def test_round12_capture_loader_rejects_unattested_run(tmp_path: Path) -> None:
    database = tmp_path / "missing.duckdb"
    with PolymarketEvidenceStore(database) as store:
        store.start_run("capture-e", STARTED_AT_MS)

        with pytest.raises(ValueError, match="manifest is missing"):
            load_round12_capture_manifest(
                store,
                run_id="capture-e",
                contract=_contract(),
            )
