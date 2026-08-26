"""Adjudicate direct-cost and alternative-yield economics of holding rewards."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "polymarket-complete-set-holding-yield-net-economics-v5"
BTC_PATH = Path(
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-reconciliation-v3-2026-08-26.json"
)
CROSS_ASSET_PATH = Path(
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-cross-asset-v4-2026-08-26.json"
)
READINESS_PATH = Path(
    "docs/model-research/polymarket/complete-set-holding-reward-readiness-v2.json"
)
IMPLEMENTATION_PATH = Path("tools/adjudicate_polymarket_holding_yield_net_economics.py")
DEFAULT_OUTPUT = Path(
    "docs/model-research/polymarket/"
    "complete-set-holding-yield-net-economics-v5-2026-08-26.json"
)
EXPECTED_SOURCE_HASHES = {
    BTC_PATH: "48e31f3d6021d28946fa1f143f65ff0f6baf9a222424f41e76c2d89875796abe",
    CROSS_ASSET_PATH: "eda29a314218e1724e39984e2712a4351d9e697503d4583d391c89a060ba53ea",
    READINESS_PATH: "2d3650d65f248294395fcac336c6650e0c6bc332cb490c6f0bac70bc11244e2c",
}
ALTERNATIVE_RATES = (
    Decimal("0"),
    Decimal("0.01"),
    Decimal("0.02"),
    Decimal("0.025"),
    Decimal("0.03"),
    Decimal("0.03125"),
    Decimal("0.0325"),
    Decimal("0.04"),
    Decimal("0.05"),
)
FRICTION_BIPS = (
    Decimal("1"),
    Decimal("5"),
    Decimal("10"),
    Decimal("20"),
    Decimal("50"),
)
HORIZON_DAYS = (
    Decimal("7"),
    Decimal("30"),
    Decimal("90"),
    Decimal("127"),
    Decimal("180"),
    Decimal("365"),
)
YEAR_DAYS = Decimal(365)
BIPS = Decimal(10_000)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _load_source(path: Path) -> dict[str, object]:
    source = _mapping(json.loads(path.read_text()), name=str(path))
    declared = str(source["result_sha256"])
    body = dict(source)
    body.pop("result_sha256")
    actual = _sha256(_canonical_json(body).encode("ascii"))
    expected = EXPECTED_SOURCE_HASHES[path]
    if declared != actual or actual != expected:
        raise ValueError(
            f"source hash differs for {path}: {declared=} {actual=} {expected=}"
        )
    return source


def _sensitivity_rows(
    *, principal: Decimal, reward: Decimal, observed_days: Decimal
) -> list[dict[str, object]]:
    realized_rate = reward / principal / observed_days * YEAR_DAYS
    rows: list[dict[str, object]] = []
    for alternative_rate in ALTERNATIVE_RATES:
        alternative_reward = principal * alternative_rate * observed_days / YEAR_DAYS
        net_reward = reward - alternative_reward
        spread = realized_rate - alternative_rate
        friction_rows = []
        for friction_bips in FRICTION_BIPS:
            break_even_days = (
                YEAR_DAYS * friction_bips / BIPS / spread if spread > 0 else None
            )
            friction_rows.append(
                {
                    "total_external_friction_bips": _decimal_text(friction_bips),
                    "break_even_holding_days": (
                        _decimal_text(break_even_days)
                        if break_even_days is not None
                        else None
                    ),
                }
            )
        rows.append(
            {
                "alternative_annual_rate": _decimal_text(alternative_rate),
                "alternative_reward_over_observation_pusd": _decimal_text(
                    alternative_reward
                ),
                "net_reward_after_alternative_and_direct_cost_pusd": _decimal_text(
                    net_reward
                ),
                "annualized_net_spread_bips": _decimal_text(spread * BIPS),
                "positive": net_reward > 0,
                "external_friction_break_even": friction_rows,
                "maximum_tolerable_total_friction_by_horizon": [
                    {
                        "horizon_days": _decimal_text(horizon),
                        "maximum_total_friction_bips": _decimal_text(
                            max(Decimal(0), spread * BIPS * horizon / YEAR_DAYS)
                        ),
                    }
                    for horizon in HORIZON_DAYS
                ],
            }
        )
    return rows


def _case(
    *,
    asset: str,
    principal: Decimal,
    reward: Decimal,
    possible_hours: int,
    realized_rate: Decimal,
    wallet: str,
    condition_id: str,
) -> dict[str, object]:
    observed_days = Decimal(possible_hours) / Decimal(24)
    reconstructed_rate = reward / principal / observed_days * YEAR_DAYS
    if reconstructed_rate != realized_rate:
        raise ValueError(f"{asset} realized rate does not reconstruct exactly")
    sensitivities = _sensitivity_rows(
        principal=principal,
        reward=reward,
        observed_days=observed_days,
    )
    return {
        "asset": asset,
        "wallet": wallet,
        "condition_id": condition_id,
        "principal_pusd": _decimal_text(principal),
        "observed_days": _decimal_text(observed_days),
        "observed_reward_pusd": _decimal_text(reward),
        "direct_relayer_split_merge_user_gas_pusd": "0",
        "direct_protocol_split_merge_principal_loss_pusd": "0",
        "net_reward_after_direct_split_merge_cost_pusd": _decimal_text(reward),
        "realized_annualized_rate": _decimal_text(realized_rate),
        "break_even_external_alternative_annual_rate": _decimal_text(realized_rate),
        "alternative_yield_sensitivities": sensitivities,
    }


def run() -> dict[str, object]:
    btc = _load_source(BTC_PATH)
    cross_asset = _load_source(CROSS_ASSET_PATH)
    readiness = _load_source(READINESS_PATH)
    gasless = _mapping(readiness["gasless_route"], name="gasless route")
    if gasless.get("direct_user_gas_cost_pusd") != "0" or "split and merge" not in str(
        gasless.get("claim")
    ):
        raise ValueError("source no longer proves relayed split and merge gas is zero")
    btc_observation = _mapping(btc["observation"], name="BTC observation")
    btc_set = _mapping(btc["current_complete_set"], name="BTC complete set")
    cases = [
        _case(
            asset="BTC",
            principal=Decimal(str(btc_observation["base_position_value_pusd"])),
            reward=Decimal(str(btc_observation["total_reward_pusd"])),
            possible_hours=int(btc_observation["possible_sampled_hours"]),
            realized_rate=Decimal(str(btc_observation["realized_annualized_rate"])),
            wallet=str(btc_set["wallet"]),
            condition_id=str(btc_set["selected_condition_id"]),
        )
    ]
    for raw_case in cross_asset["cases"]:
        source_case = _mapping(raw_case, name="cross-asset case")
        observation = _mapping(source_case["observation"], name="observation")
        cases.append(
            _case(
                asset=str(source_case["asset"]),
                principal=Decimal(str(source_case["shares_per_outcome"])),
                reward=Decimal(str(observation["total_reward_pusd"])),
                possible_hours=int(observation["possible_sampled_hours"]),
                realized_rate=Decimal(str(observation["realized_annualized_rate"])),
                wallet=str(source_case["wallet"]),
                condition_id=str(source_case["condition_id"]),
            )
        )
    observed_days = {Decimal(str(case["observed_days"])) for case in cases}
    if observed_days != {Decimal(14)}:
        raise ValueError("cross-asset observation duration differs")
    total_principal = sum(
        (Decimal(str(case["principal_pusd"])) for case in cases), Decimal(0)
    )
    total_reward = sum(
        (Decimal(str(case["observed_reward_pusd"])) for case in cases), Decimal(0)
    )
    portfolio_rate = total_reward / total_principal / Decimal(14) * YEAR_DAYS
    portfolio_sensitivities = _sensitivity_rows(
        principal=total_principal,
        reward=total_reward,
        observed_days=Decimal(14),
    )
    realized_rates = [Decimal(str(case["realized_annualized_rate"])) for case in cases]
    three_percent_rows = [
        next(
            row
            for row in case["alternative_yield_sensitivities"]
            if row["alternative_annual_rate"] == "0.03"
        )
        for case in cases
    ]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "source_bound_direct_cost_and_alternative_yield_adjudication_for_complete_set_holding_rewards",
        "authority": {
            "credentials_used": False,
            "funds_used": False,
            "orders_placed": False,
            "transactions_sent": False,
            "trading_authority": False,
        },
        "source_continuity": [
            {"path": path.as_posix(), "result_sha256": expected}
            for path, expected in EXPECTED_SOURCE_HASHES.items()
        ],
        "direct_cost_contract": {
            "route": "Polymarket relayer CTF split then later merge",
            "split_identity": "one_pUSD_creates_one_YES_and_one_NO",
            "merge_identity": "one_YES_plus_one_NO_restores_one_pUSD",
            "direct_user_gas_cost_pusd": "0",
            "direct_protocol_principal_loss_pusd": "0",
            "gasless_source_url": gasless["source_url"],
            "gasless_source_response_sha256": gasless["source_response_sha256"],
            "preconditions": gasless["preconditions"],
        },
        "cases": cases,
        "cross_asset_portfolio": {
            "asset_count": len(cases),
            "assets": [str(case["asset"]) for case in cases],
            "demonstrated_principal_pusd": _decimal_text(total_principal),
            "observed_days": "14",
            "positive_daily_payout_count": 42,
            "possible_daily_payout_count": 42,
            "observed_reward_pusd": _decimal_text(total_reward),
            "net_reward_after_direct_split_merge_cost_pusd": _decimal_text(
                total_reward
            ),
            "principal_weighted_realized_annualized_rate": _decimal_text(
                portfolio_rate
            ),
            "conservative_cross_asset_realized_rate_floor": _decimal_text(
                min(realized_rates)
            ),
            "alternative_yield_sensitivities": portfolio_sensitivities,
        },
        "adjudication": {
            "positive_after_direct_relayer_and_protocol_cost_in_every_case": all(
                Decimal(str(case["net_reward_after_direct_split_merge_cost_pusd"])) > 0
                for case in cases
            ),
            "positive_after_three_percent_alternative_yield_in_every_case": all(
                bool(row["positive"]) for row in three_percent_rows
            ),
            "positive_after_three_point_two_five_percent_alternative_yield_in_any_case": any(
                bool(
                    next(
                        row
                        for row in case["alternative_yield_sensitivities"]
                        if row["alternative_annual_rate"] == "0.0325"
                    )["positive"]
                )
                for case in cases
            ),
            "external_alternative_yield_and_friction_fully_proven": False,
            "demonstrated_capacity_is_not_a_capacity_ceiling": True,
        },
        "verdict": {
            "accepted_edge": True,
            "accepted_scope": "existing idle pUSD already on Polymarket using a successful gasless relayer split and merge while eligibility and current holding rewards remain active",
            "status": "validated_cross_asset_positive_after_direct_split_merge_cost_and_positive_through_three_percent_annual_alternative_yield_before_external_friction",
            "after_direct_cost": True,
            "after_every_external_cost_and_best_alternative_yield": False,
            "deployment_ready": False,
            "future_profit_guaranteed": False,
            "trading_authority": False,
        },
        "limitations": [
            "The 3.25 percent program rate is variable and Polymarket may cap rewards.",
            "Realized rates include hourly sampling and payout rounding and were below 3.25 percent.",
            "Relayer eligibility credentials successful acceptance and confirmation remain preconditions.",
            "Bridge wrapping withdrawal custody tax delay failed-operation and external alternative-yield costs are not zeroed or assumed.",
            "The three public wallets demonstrate 1039 pUSD of aggregate principal but do not establish an account capacity ceiling.",
            "Fourteen observed days per asset do not guarantee future persistence.",
        ],
        "implementation": {
            "path": IMPLEMENTATION_PATH.as_posix(),
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run()
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    portfolio = result["cross_asset_portfolio"]
    print(f"output={args.output}")
    print(f"result_sha256={result['result_sha256']}")
    print(f"observed_reward_pusd={portfolio['observed_reward_pusd']}")
    print(
        "principal_weighted_realized_annualized_rate="
        f"{portfolio['principal_weighted_realized_annualized_rate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
