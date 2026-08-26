from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRIAGE_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "structural-edge-source-triage-v1-2026-08-25.json"
)
TRIAGE_V2_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "structural-edge-source-triage-v2-2026-08-25.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_TRIAGE_SHA256 = (
    "509f63910c77a582680849e779317396962d06edeffa537e7d5ce8e18a984cb2"
)
EXPECTED_TRIAGE_V2_SHA256 = (
    "3df17e93866cbf53617340dd422a91945c8a1924d4ca736b76c5f78f4c9a5575"
)
EXPECTED_REGISTRY_SHA256 = (
    "57ea9bd59d8bd9ae43d4100dea192a1c19f272ef4844790d04d2da8e69f5630c"
)
ROUND61_REPORT_SHA256 = (
    "e2f6275232b7f6b7b511211b26a697536a401e14a118393502bcfda96ae4d6e4"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _embedded_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_source_triage_is_self_hashed_and_makes_no_authority_claim() -> None:
    triage = _load(TRIAGE_PATH)

    assert triage["result_sha256"] == EXPECTED_TRIAGE_SHA256
    assert _embedded_hash(triage) == EXPECTED_TRIAGE_SHA256
    assert triage["authority"] == {
        "accepted_edge": False,
        "live_trading_authority": False,
        "venue_market_data_requests_made": 0,
        "orders_placed": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "signed_requests_made": 0,
    }
    assert triage["preflight"]["binance_ephemeral_api_key_present"] is False
    assert triage["preflight"]["binance_ephemeral_api_secret_present"] is False
    assert triage["preflight"]["polymarket_protected_boundary_reached"] is False


def test_registry_binds_round61_and_liquid_staking_without_repeating_carry() -> None:
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    hypotheses = registry["prioritized_hypotheses"]
    assert [row["priority_rank"] for row in hypotheses] == list(range(1, 37))

    staking = next(
        row
        for row in hypotheses
        if row["mechanism"] == "liquid_staking_token_conversion_parity"
    )
    assert staking["priority_rank"] == 11
    assert staking["market_direction_forecast_required"] is False
    assert staking["canonical_artifacts"] == [
        {
            "path": (
                "docs/model-research/structural-edge-source-triage-v1-2026-08-25.json"
            ),
            "result_sha256": EXPECTED_TRIAGE_SHA256,
        }
    ]

    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert (
        terminal["binance_elevated_funding_spot_perpetual_carry"][
            "canonical_result_sha256"
        ]
        == ROUND61_REPORT_SHA256
    )
    assert (
        terminal["polymarket_manufactured_complete_set_taker_tier_volume"][
            "canonical_result_sha256"
        ]
        == EXPECTED_TRIAGE_SHA256
    )


def test_second_source_triage_stops_cross_product_funding_before_requests() -> None:
    triage = _load(TRIAGE_V2_PATH)

    assert triage["result_sha256"] == EXPECTED_TRIAGE_V2_SHA256
    assert _embedded_hash(triage) == EXPECTED_TRIAGE_V2_SHA256
    assert triage["authority"] == {
        "accepted_edge": False,
        "live_trading_authority": False,
        "orders_placed": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "signed_requests_made": 0,
        "venue_market_data_requests_made": 0,
    }
    mechanism = triage["reviewed_mechanism"]
    assert mechanism["collector_decision"] == (
        "do_not_build_or_call_public_funding_or_book_collector"
    )
    assert mechanism["market_direction_forecast_required"] == (
        "unproved_without_complete_payoff_and_collateral_semantics"
    )
    assert triage["source_quality_correction"] == {
        "decision": (
            "when_generated_endpoint_security_labels_conflict_with_an_official_"
            "transport_use_the_stricter_signed_classification_and_make_no_request_"
            "without_explicit_authority"
        ),
        "generated_go_account_markdown_claim": "No authorization required",
        "official_python_transport_behavior": "sign_request_GET",
        "reason": (
            "a_generated_security_label_is_not_authoritative_enough_to_expose_an_"
            "account_endpoint"
        ),
    }

    registry = _load(REGISTRY_PATH)
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert (
        terminal["binance_usds_margined_versus_coin_margined_perpetual_funding_pair"][
            "canonical_result_sha256"
        ]
        == EXPECTED_TRIAGE_V2_SHA256
    )
