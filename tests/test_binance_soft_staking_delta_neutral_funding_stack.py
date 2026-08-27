from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-soft-staking-delta-neutral-funding-stack-terminal-v1-2026-08-27.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
EXPECTED_RESULT_SHA256 = (
    "591fb98b9a8e58365c67c4a281d1fda3de674b42f1f868a42d98acf2ab19ae68"
)
EXPECTED_REGISTRY_SHA256 = "aabfdc0750a619b380929c59546d37c86306686bc2144d85c90d770f5bea6d23"
SIX_PLACES = Decimal("0.000001")


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


def test_stack_reconstructs_and_fails_every_after_cost_gate() -> None:
    artifact = _load(ARTIFACT_PATH)

    assert artifact["result_sha256"] == EXPECTED_RESULT_SHA256
    assert _embedded_hash(artifact) == EXPECTED_RESULT_SHA256
    assert artifact["authority"]["new_public_market_requests"] == 0
    assert artifact["authority"]["orders_or_transfers"] == 0
    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["adjudication"]["profitability_claim"] is False
    assert artifact["results"]["all_roles_positive_after_32_bips"] is False
    assert artifact["results"]["all_roles_positive_after_one_capital_leg_hurdle"] is False
    assert artifact["results"]["all_roles_positive_after_two_capital_leg_hurdle"] is False
    for symbol in ("ETHUSDT", "SOLUSDT"):
        raw_meta = next(
            row
            for row in artifact["input_lineage"]["funding_responses"]
            if row["symbol"] == symbol
        )
        raw_path = ROOT / raw_meta["path"]
        raw_bytes = raw_path.read_bytes()
        assert len(raw_bytes) == raw_meta["payload_bytes"]
        assert hashlib.sha256(raw_bytes).hexdigest() == raw_meta["payload_sha256"]
        funding_rows = json.loads(raw_bytes)
        assert len(funding_rows) == raw_meta["actual_row_count"] == 500
        assert all(row["symbol"] == symbol for row in funding_rows)
        assert [row["fundingTime"] for row in funding_rows] == sorted(
            row["fundingTime"] for row in funding_rows
        )

        roles = artifact["results"][symbol]["roles"]
        assert [row["row_count"] for row in roles] == [300, 100, 100]
        boundaries = ((0, 300), (300, 400), (400, 500))
        required_aprs: list[Decimal] = []
        for row, (start, end) in zip(roles, boundaries, strict=True):
            role_rows = funding_rows[start:end]
            days = Decimal(len(role_rows)) / Decimal(3)
            funding_bips = sum(
                (Decimal(item["fundingRate"]) for item in role_rows),
                start=Decimal(0),
            ) * Decimal(10000)
            reward_bips = Decimal("0.005") * days / Decimal(365) * Decimal(10000)
            gross_bips = funding_bips + reward_bips
            one_leg_net = (
                gross_bips
                - Decimal(32)
                - Decimal(1000) * days / Decimal(365)
            )
            two_leg_net = (
                gross_bips
                - Decimal(32)
                - Decimal(2000) * days / Decimal(365)
            )

            assert Decimal(row["days"]) == days.quantize(Decimal("0.0001"))
            assert Decimal(row["funding_bips"]) == funding_bips.quantize(SIX_PLACES)
            assert Decimal(row["staking_reward_bips"]) == reward_bips.quantize(SIX_PLACES)
            assert Decimal(row["gross_funding_plus_reward_bips"]) == gross_bips.quantize(
                SIX_PLACES
            )
            assert Decimal(row["net_after_32_bips_and_one_capital_leg"]) == (
                one_leg_net.quantize(SIX_PLACES)
            )
            assert Decimal(row["net_after_32_bips_and_two_capital_legs"]) == (
                two_leg_net.quantize(SIX_PLACES)
            )
            assert Decimal(row["net_after_32_bips_and_one_capital_leg"]) < 0
            assert Decimal(row["net_after_32_bips_and_two_capital_legs"]) < 0
            required_aprs.append(
                (Decimal(32) - funding_bips)
                * Decimal(365)
                / (Decimal(10000) * days)
            )

        recorded_apr = Decimal(
            artifact["results"][symbol]["maximum_required_soft_staking_apr_fraction"][
                "after_32_bips_only"
            ]
        )
        assert recorded_apr == max(required_aprs).quantize(Decimal("0.0000000001"))


def test_registry_terminalizes_stack_without_changing_idle_yield_acceptance() -> None:
    artifact = _load(ARTIFACT_PATH)
    registry = _load(REGISTRY_PATH)

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry) == EXPECTED_REGISTRY_SHA256
    assert registry["accepted_edge_count"] == 16
    idle = next(row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 3)
    assert any(
        row["result_sha256"] == artifact["result_sha256"]
        for row in idle["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"] == "binance_ETH_SOL_Soft_Staking_delta_neutral_USDT_perpetual_funding_stack"
    )
    assert terminal["canonical_result_sha256"] == artifact["result_sha256"]
