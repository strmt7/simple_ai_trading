from pathlib import Path

from tools.publish_polymarket_condition_replay_audit import publish


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
