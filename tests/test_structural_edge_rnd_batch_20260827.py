from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/model-research/action-value" / (
    "polymarket-combo-rfq-boolean-parity-candidate-v1-2026-08-27.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_HASH = "08fb223f771c5793da944497f37f4067238e7fd2b40fa2427293dbf7b55c4116"
REGISTRY_HASH = "44fdf0cba6b97bcf40c407bc78cedbdbf8051ff1b7e40267b5bc4db629abb22a"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_combo_boolean_candidate_is_hash_bound_unaccepted_and_non_mutating() -> None:
    artifact = _load(ARTIFACT)

    assert artifact["result_sha256"] == EXPECTED_HASH
    assert _canonical_hash(artifact) == EXPECTED_HASH
    adjudication = artifact["adjudication"]
    assert adjudication["accepted_edge"] is False
    assert adjudication["deployment_ready"] is False
    assert adjudication["market_direction_forecast_required"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["trading_authority"] is False

    authority = artifact["authority"]
    assert authority["authenticated_requests"] == 0
    assert authority["combo_quote_requests_created"] == 0
    assert authority["public_order_book_requests"] == 0
    assert authority["orders_or_transfers_submitted"] == 0
    assert authority["funded_actions"] == 0


def test_boolean_identity_survives_ordinary_and_fractional_terminal_values() -> None:
    artifact = _load(ARTIFACT)
    cases = (
        (Decimal("0"), Decimal("0")),
        (Decimal("1"), Decimal("0")),
        (Decimal("0"), Decimal("1")),
        (Decimal("1"), Decimal("1")),
        (Decimal("0.5"), Decimal("0.5")),
        (Decimal("0.25"), Decimal("0.75")),
    )

    for left, right in cases:
        underlyings = left + right
        combo_yes = left * right
        complement_combo_no = Decimal("1") - (
            (Decimal("1") - left) * (Decimal("1") - right)
        )
        assert underlyings == combo_yes + complement_combo_no

    identity = artifact["exact_payoff_identity"]
    assert identity["terminal_state_scope"].startswith("all A and B values in [0,1]")
    assert "0.25 + 0.75 = 1.0" in identity["why_the_full_identity_survives_cancellation"]
    assert "0.25" in identity["why_naive_implication_is_rejected"]


def test_public_catalog_evidence_is_partial_and_not_misreported_as_quotes() -> None:
    artifact = _load(ARTIFACT)
    catalog = artifact["current_partial_catalog_evidence"]
    sources = artifact["sources"]

    assert catalog["page_count"] == 5
    assert catalog["market_count"] == 500
    assert catalog["complete_population_claim"] is False
    assert catalog["final_next_cursor"] == "NDg5MTA"
    assert len(sources["combo_catalog_pages"]) == 5
    assert len(sources["official_documents"]) == 7
    assert len(sources["same_game_gamma_payloads"]) == 7
    assert artifact["authority"]["retained_public_http_responses"] == 19
    limits = " ".join(artifact["current_evidence_limits"])
    assert "not Combo RFQ bids" in limits
    assert "No approved builder status" in limits


def test_current_examples_preserve_status_conflict_and_cancellation_warning() -> None:
    artifact = _load(ARTIFACT)
    examples = artifact["current_same_game_examples"]

    assert len(examples) == 2
    assert all(row["combo_status"] == "enabled" for row in examples)
    assert all(row["combo_catalog_pending"] is False for row in examples)
    assert all(row["legacy_or_undocumented_rfqEnabled"] is False for row in examples)
    assert any(len(row["spread_market_ids"]) == 3 for row in examples)
    prohibited = " ".join(artifact["prohibited_shortcuts"])
    assert "fractional 50_50 cancellation" in prohibited
    assert "execute atomically" in prohibited


def test_registry_adds_only_trigger_gated_combo_identity_candidate() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry) == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 41)
    )
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "polymarket_combo_rfq_boolean_identity_against_underlying_clob"
    )
    assert row["priority_rank"] == 33
    assert row["market_direction_forecast_required"] is False
    assert row["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/action-value/"
                "polymarket-combo-rfq-boolean-parity-candidate-v1-2026-08-27.json"
            ),
            "result_sha256": EXPECTED_HASH,
        }
    ]
    assert "quote_request_only" in row["retry_trigger"]
    assert "not_accepted_or_deployment_ready" in row["current_status"]
