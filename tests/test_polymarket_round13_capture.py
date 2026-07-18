from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_recorder import PolymarketEvidenceStore
from simple_ai_trading.polymarket_round12_reference import (
    load_round12_reference_from_round11_artifact,
    polymarket_round12_primary_policy,
)
from simple_ai_trading.polymarket_round13 import (
    PolymarketRound13Program,
    polymarket_round13_evaluation_gates,
    polymarket_round13_scenarios,
)
from simple_ai_trading import polymarket_round13_capture as capture_module
from simple_ai_trading.polymarket_round13_capture import (
    create_round13_capture_manifest,
    load_round13_capture_manifest,
    validate_round13_capture_manifest_payload,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/polymarket/round-011-single-leg-directional-value-artifact.json"
)
STARTED_AT_MS = 1_700_000_000_000
CONTRACT_PATH = (
    "docs/model-research/polymarket/round-013-sealed-confirmation-contract.json"
)
PREDECESSOR_PATH = (
    "docs/model-research/polymarket/"
    "round-011-single-leg-directional-value-artifact.json"
)


def _program() -> PolymarketRound13Program:
    contract_sha = "a" * 64
    contract = {
        "contract_sha256": contract_sha,
        "implementation": {
            "reference_implementation_sha256": "b" * 64,
            "action_pipeline_implementation_sha256": "c" * 64,
            "round13_program_implementation_sha256": "d" * 64,
        },
        "predecessor_evidence": {"artifact_filename": Path(PREDECESSOR_PATH).name},
    }
    return PolymarketRound13Program(
        contract=contract,
        contract_sha256=contract_sha,
        model=load_round12_reference_from_round11_artifact(ARTIFACT),
        policy=polymarket_round12_primary_policy(),
        scenarios=polymarket_round13_scenarios(),
        evaluation_gates=polymarket_round13_evaluation_gates(),
        confirmation_capital_quote=Decimal("1000"),
    ).validated()


def _manifest(
    run_id: str,
    *,
    capture_duration_seconds: int = (
        capture_module.POLYMARKET_ROUND13_CAPTURE_DURATION_SECONDS
    ),
) -> dict[str, object]:
    program = _program()
    implementation = program.contract["implementation"]
    required = {path: "e" * 64 for path in capture_module._REQUIRED_REPOSITORY_FILES}
    required[CONTRACT_PATH] = "f" * 64
    required[PREDECESSOR_PATH] = "1" * 64
    return create_round13_capture_manifest(
        run_id=run_id,
        started_at_ms=STARTED_AT_MS,
        capture_duration_seconds=capture_duration_seconds,
        repository_commit="2" * 40,
        repository_tree="3" * 40,
        contract_repository_path=CONTRACT_PATH,
        predecessor_repository_path=PREDECESSOR_PATH,
        contract_sha256=program.contract_sha256,
        model_sha256=program.model.model_sha256,
        policy_sha256=program.policy.policy_sha256,
        reference_implementation_sha256=str(
            implementation["reference_implementation_sha256"]
        ),
        action_pipeline_implementation_sha256=str(
            implementation["action_pipeline_implementation_sha256"]
        ),
        round13_program_implementation_sha256=str(
            implementation["round13_program_implementation_sha256"]
        ),
        required_file_sha256=required,
    )


def test_round13_capture_manifest_is_complete_and_program_bound() -> None:
    manifest = _manifest("run-a")

    loaded = validate_round13_capture_manifest_payload(
        manifest,
        expected_run_id="run-a",
        expected_program=_program(),
    )

    assert loaded == manifest
    assert loaded["outcome_endpoints_queried"] is False
    assert loaded["labels_consulted"] is False
    assert loaded["capture_started_before_manifest"] is False


def test_round13_capture_manifest_rejects_tampering() -> None:
    manifest = _manifest("run-b")
    manifest["repository_tree"] = "4" * 40

    with pytest.raises(ValueError, match="capture manifest is invalid"):
        validate_round13_capture_manifest_payload(manifest)


def test_round13_capture_manifest_rejects_short_capture_contract() -> None:
    with pytest.raises(ValueError, match="capture manifest is invalid"):
        _manifest("run-short", capture_duration_seconds=300)


def test_round13_capture_manifest_rejects_host_specific_paths() -> None:
    manifest = _manifest("run-c")
    manifest["contract_repository_path"] = "C:\\repo\\contract.json"

    with pytest.raises(ValueError, match="capture manifest is invalid"):
        validate_round13_capture_manifest_payload(manifest)


def test_round13_repository_attestation_rechecks_committed_bytes(
    monkeypatch,
) -> None:
    manifest = _manifest("run-git")
    committed = b"tracked Round 13 source\n"
    digest = hashlib.sha256(committed).hexdigest()
    manifest["required_file_sha256"] = {
        path: digest for path in manifest["required_file_sha256"]
    }
    body = dict(manifest)
    body.pop("manifest_sha256")
    manifest["manifest_sha256"] = capture_module._canonical_sha256(body)

    def git_bytes(_root: Path, *arguments: str) -> bytes:
        if arguments[0] == "rev-parse":
            return ("3" * 40 + "\n").encode("ascii")
        if arguments[0] == "show":
            return committed
        raise AssertionError(arguments)

    monkeypatch.setattr(capture_module, "_repository_root", lambda: ROOT)
    monkeypatch.setattr(capture_module, "_git_bytes", git_bytes)

    assert capture_module.verify_round13_repository_attestation(manifest) == manifest


def test_round13_capture_manifest_persists_before_run_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _manifest("run-d")
    monkeypatch.setattr(
        capture_module,
        "verify_round13_repository_attestation",
        lambda value: dict(value),
    )
    with PolymarketEvidenceStore(tmp_path / "capture.duckdb") as store:
        store.start_run(
            "run-d",
            STARTED_AT_MS,
            preregistration_manifest=manifest,
        )

        loaded = load_round13_capture_manifest(
            store,
            run_id="run-d",
            program=_program(),
        )
        run_count = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_recorder_run")
            .fetchone()[0]
        )
        raw_count = (
            store.connect()
            .execute("SELECT count(*) FROM polymarket_raw_chunk")
            .fetchone()[0]
        )

    assert loaded == manifest
    assert run_count == 1
    assert raw_count == 0
