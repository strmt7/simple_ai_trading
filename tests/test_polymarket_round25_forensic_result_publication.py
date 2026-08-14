from __future__ import annotations

import hashlib
import json
from pathlib import Path

from simple_ai_trading.polymarket_round25_forensic_model import (
    validate_round25_forensic_result,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "latest"
    / "round-025-v2-forensic"
)
RESULT = PUBLICATION / "round25-selection-result.json"
MANIFEST = PUBLICATION / "round25-selection-figure-manifest.json"
RESULT_FILE_SHA256 = (
    "662071028b2541068753ead77eb9be29417acfa098e3168e605c33b489a73cc2"
)


def test_forensic_result_publication_is_hash_bound_and_rejects_edge_claims() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == RESULT_FILE_SHA256
    result = validate_round25_forensic_result(
        json.loads(RESULT.read_text(encoding="ascii"))
    )

    assert result["result_sha256"] == (
        "13f08aa51fdabfefd85da0e3f59923c0fea837b65209186135911d566dcc7f9f"
    )
    assert result["selected_candidate_id"] == "market-prior-v1"
    assert result["selection_condition_count"] == 14
    assert result["closed_trade_count"] == 0
    assert result["abstention_rate"] == 1.0
    assert result["net_profit_quote"] == 0
    assert result["diagnostic_predictive_gate_passed"] is False
    assert result["diagnostic_economic_gate_passed"] is False
    assert result["statistical_edge_established"] is False
    assert result["after_cost_profitability_established"] is False
    assert result["profitability_claim"] is False


def test_forensic_figure_manifest_matches_every_source_file() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="ascii"))
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")

    assert hashlib.sha256(encoded).hexdigest() == claimed
    assert manifest["diagnostic_only"] is True
    assert manifest["result_sha256"] == (
        "13f08aa51fdabfefd85da0e3f59923c0fea837b65209186135911d566dcc7f9f"
    )
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((PUBLICATION / filename).read_bytes()).hexdigest() == (
            expected
        )
    svg = (PUBLICATION / "round25-selection-diagnostic.svg").read_text(
        encoding="ascii"
    )
    assert all(line == line.rstrip() for line in svg.splitlines())


def test_forensic_readme_states_the_failed_gates_and_utc_window() -> None:
    readme = (PUBLICATION / "README.md").read_text(encoding="ascii")

    assert "did not find a model edge or establish profitability" in readme
    assert "2026-08-12 13:25 to 15:20 UTC" in readme
    assert "Predictive / economic gate | Failed / failed" in readme
    assert "Closed trades / abstention rate | 0 / 100%" in readme
