from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.impact_absorption_execution_evidence import (
    ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DESIGN = (
    REPOSITORY
    / "docs/model-research/action-value"
    / "round-074-event-sequence-model-design-v66.json"
)


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


def _normalized_lf_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def test_round74_v66_binds_environment_provenance_correction() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    claimed = design.pop("design_sha256")
    correction = design["execution_provenance_correction"]

    assert claimed == _canonical_sha256(design)
    assert design["base_design"]["normalized_lf_sha256"] == _normalized_lf_sha256(
        REPOSITORY / design["base_design"]["path"]
    )
    for relative, expected in design["normalized_lf_source_binding"].items():
        assert expected == _normalized_lf_sha256(REPOSITORY / relative)
    assert correction["current_source_schema_version"] == (
        ROUND74_EXECUTION_CALIBRATION_SCHEMA_VERSION
    )


def test_round74_v66_blocks_testnet_to_mainnet_relabelling() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))
    correction = design["execution_provenance_correction"]

    assert correction["environment_intrinsic_to_each_execution_leg"] is True
    assert correction["aggregate_environment_must_equal_every_leg"] is True
    assert correction["mixed_environment_panel_permitted"] is False
    assert correction["testnet_records_may_claim_mainnet"] is False
    assert correction["mainnet_target_assembly_from_testnet_records_permitted"] is False
    assert correction["testnet_to_mainnet_transfer_model_implemented"] is False
    assert correction["production_book_age_limit_changed"] is False


def test_round74_v66_preserves_all_claim_and_authority_blocks() -> None:
    design = json.loads(DESIGN.read_text(encoding="ascii"))

    assert design["verification"]["representative_market_training_executed"] is False
    assert design["unchanged_model_contract"]["candidate_panel_changed"] is False
    assert design["unchanged_model_contract"]["default_profile"] == "conservative"
    assert set(design["authority"].values()) == {False}
