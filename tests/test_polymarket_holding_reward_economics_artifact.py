from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "complete-set-holding-reward-economics-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "b15b9039848094057322387c9aed3a555a8ca32020af97689fc6b26e16114561"
)


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


def test_holding_reward_artifact_hash_reconstructs() -> None:
    artifact = _artifact()
    expected = artifact.pop("result_sha256")
    canonical = json.dumps(
        artifact,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    assert expected == EXPECTED_RESULT_SHA256
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_RESULT_SHA256


def test_holding_reward_sensitivities_reconstruct_without_payout_claim() -> None:
    artifact = _artifact()
    economics = artifact["economic_sensitivity_per_1_pusd_position_value"]
    rate_rows = economics["rate_scenarios"]
    break_even_rows = economics["break_even_days_by_total_friction_bips"]

    with localcontext() as context:
        context.prec = 40
        for row in rate_rows:
            rate = Decimal(row["annual_rate"])
            assert Decimal(row["hourly_reward"]) == rate / Decimal(365 * 24)
            assert Decimal(row["daily_reward"]) == rate / Decimal(365)
            assert Decimal(row["nominal_30_day_reward"]) == (
                rate * Decimal(30) / Decimal(365)
            )

        for row in break_even_rows:
            rate = Decimal(row["annual_rate"])
            for friction_bips, recorded_days in row["break_even_days"].items():
                expected_days = (
                    Decimal(friction_bips) / Decimal(10_000) * Decimal(365) / rate
                )
                assert Decimal(recorded_days) == expected_days

    assert artifact["rate_source_adjudication"] == {
        "conservative_sensitivity_rate": "0.0325",
        "conflict_resolved": False,
        "documentation_annual_rate": "0.04",
        "help_center_annual_rate": "0.0325",
        "status": "official_source_conflict_blocks_current_rate_claim",
    }
    assert artifact["authority"]["publicly_proven_future_payout_lower_bound_pusd"] == (
        "0"
    )
    assert artifact["authority"]["accepted_edge"] is False
    assert artifact["authority"]["profitability_claim"] is False


def test_holding_reward_sources_and_crypto_scope_are_exact() -> None:
    artifact = _artifact()
    sources = artifact["sources"]

    assert [
        (row["retrieved_bytes"], row["retrieved_content_sha256"]) for row in sources
    ] == [
        (
            4127,
            "bb131a3894d52149058f8554edaa6fafdd9ad72ab2351efbec605d4b3a6e2cc5",
        ),
        (
            86981,
            "b2a87d1711e8439676b38e0162069f5ba3f5c666e18751598030d1cd7167c0e4",
        ),
    ]
    assert artifact["in_scope_officially_listed_eligible_events"] == [
        "What price will Bitcoin hit in 2026?",
        "What price will Ethereum hit in 2026?",
        "What price will Solana hit in 2026?",
    ]
