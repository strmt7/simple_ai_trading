import hashlib
import json
from pathlib import Path

from tools.publish_polymarket_condition_replay_audit import _sparse_ticks, publish


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "latest"
    / "round-026-twap60-condition-audit"
    / "condition-replay-audit.json"
)
PUBLISHED_STAGE0 = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "latest"
    / "round-027-stage0-condition-audit"
)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def test_condition_replay_publisher_binds_requested_round_title(tmp_path: Path) -> None:
    manifest = publish(SOURCE, tmp_path, title="Round 27 Stage 0")

    assert manifest["schema_version"] == (
        "polymarket-condition-replay-publication-v2"
    )
    assert manifest["title"] == "Round 27 Stage 0"
    assert "# Round 27 Stage 0 replay eligibility" in (
        tmp_path / "README.md"
    ).read_text(encoding="ascii")
    assert "Round 26 target-free replay eligibility" not in (
        tmp_path / "condition-replay-eligibility.svg"
    ).read_text(encoding="utf-8")


def test_condition_replay_publisher_sparsifies_long_time_axes() -> None:
    labels = [f"label-{index}" for index in range(59)]
    positions, selected = _sparse_ticks(labels)

    assert len(positions) <= 12
    assert positions[0] == 0
    assert positions[-1] == 58
    assert selected[0] == "label-0"
    assert selected[-1] == "label-58"


def test_round27_stage0_publication_hashes_exact_target_free_audit() -> None:
    manifest = json.loads(
        (PUBLISHED_STAGE0 / "publication-manifest.json").read_text(
            encoding="ascii"
        )
    )
    manifest_claim = manifest.pop("manifest_sha256")
    audit = json.loads(
        (PUBLISHED_STAGE0 / "condition-replay-audit.json").read_text(
            encoding="ascii"
        )
    )
    audit_claim = audit.pop("audit_sha256")

    assert manifest_claim == _sha256(manifest)
    assert manifest["title"] == "Round 27 Stage 0"
    assert all(
        hashlib.sha256((PUBLISHED_STAGE0 / name).read_bytes()).hexdigest()
        == expected
        for name, expected in manifest["files"].items()
    )
    assert audit_claim == _sha256(audit)
    assert audit["condition_count"] == 59
    assert audit["eligible_condition_count"] == 53
    assert audit["failed_condition_count"] == 6
    assert audit["target_free"] is True
    assert audit["model_data_eligible"] is False
    assert audit["edge_claim"] is False
    assert audit["profitability_claim"] is False
