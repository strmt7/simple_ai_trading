from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.publish_polymarket_round27_resolutions import _svg


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "latest"
    / "round-027-stage0-resolution-mechanics"
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


def test_resolution_graph_uses_numeric_audit_and_utc_range() -> None:
    audit = {
        "condition_count": 10,
        "dual_source_agreement_count": 9,
        "winner_counts": {"Up": 6, "Down": 3},
    }
    labels = [
        {"event_start_utc": "2026-08-14T00:00:00Z"},
        {"event_start_utc": "2026-08-14T00:45:00Z"},
    ]

    svg = _svg(audit, labels).decode("ascii")

    assert "2026-08-14T00:00:00Z to 2026-08-14T00:45:00Z" in svg
    assert "Gamma and CLOB agree" in svg
    assert ">9<" in svg
    assert ">6<" in svg
    assert ">3<" in svg
    assert "No model, trade, edge, profitability" in svg


def test_published_resolution_package_is_self_hashed_and_non_promotional() -> None:
    manifest = json.loads(
        (PUBLISHED / "publication-manifest.json").read_text(encoding="ascii")
    )
    manifest_claim = manifest.pop("manifest_sha256")
    audit = json.loads(
        (PUBLISHED / "settlement-mechanics-audit.json").read_text(encoding="ascii")
    )
    audit_claim = audit.pop("audit_sha256")

    assert manifest_claim == _canonical_sha256(manifest)
    assert all(
        hashlib.sha256((PUBLISHED / name).read_bytes()).hexdigest() == expected
        for name, expected in manifest["files"].items()
    )
    assert audit_claim == _canonical_sha256(audit)
    assert audit["resolution_count"] == 53
    assert audit["dual_source_agreement_count"] == 53
    assert audit["source_disagreement_count"] == 0
    assert audit["winner_counts"] == {"Down": 27, "Up": 26}
    assert audit["edge_claim"] is False
    assert audit["profitability_claim"] is False
