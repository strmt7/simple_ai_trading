from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
ARTIFACT = BASE / (
    "binance-bfusd-btc-eth-sol-funding-carry-dominance-bound-"
    "v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return hashlib.sha256(_canonical(body)).hexdigest()


def _required_apr(
    metric_bips: Decimal,
    *,
    duration_days: Decimal,
    observation_fraction: Decimal = Decimal(1),
) -> Decimal:
    return max(Decimal(0), -metric_bips) * Decimal(365) / (
        Decimal(100) * duration_days * observation_fraction
    )


def test_dominance_bound_reconstructs_from_hash_bound_sources() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact) == artifact["result_sha256"]

    for source in artifact["source_binding"]:
        payload = (ROOT / source["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == source["file_sha256"]
        assert json.loads(payload)["result_sha256"] == source["result_sha256"]


def test_exact_btc_eth_sol_apr_thresholds_and_stresses_reconstruct() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    broad = json.loads((ROOT / artifact["source_binding"][0]["path"]).read_text())
    expected = {
        (row["symbol"], row["role"]): Decimal(
            row["economic_threshold_APR_percent"]
        )
        for row in artifact["necessary_BFUSD_APR_percent_by_symbol_and_role"]
    }
    actual: dict[tuple[str, str], Decimal] = {}

    # Match the artifact generator's explicit precision so the full source-bound
    # decimal strings reconstruct instead of being rounded by the default context.
    with localcontext() as context:
        context.prec = 50
        for symbol in broad["symbol_results"]:
            base = symbol["base_asset"]
            if base not in {"BTC", "ETH", "SOL"}:
                continue
            for role_name, role in symbol["roles"].items():
                days = Decimal(role["duration_days"])
                count = Decimal(role["observation_count"])
                thresholds = [
                    _required_apr(
                        Decimal(role["net_after_frozen_hurdles_bips"]),
                        duration_days=days,
                    ),
                    _required_apr(
                        Decimal(
                            role["family_adjusted_bootstrap_net_lower_bound_bips"]
                        ),
                        duration_days=days,
                    ),
                ]
                thresholds.extend(
                    _required_apr(
                        Decimal(slice_row["net_after_allocated_hurdle_bips"]),
                        duration_days=days,
                        observation_fraction=(
                            Decimal(slice_row["observation_count"]) / count
                        ),
                    )
                    for slice_row in role["slices"].values()
                )
                actual[(base, role_name)] = max(thresholds)

    assert actual == expected
    assert max(value for key, value in actual.items() if key[0] == "BTC") == Decimal(
        artifact["necessary_symbol_level_APR_thresholds_percent"]["BTC"]
    )
    assert max(value for key, value in actual.items() if key[0] == "ETH") == Decimal(
        artifact["necessary_symbol_level_APR_thresholds_percent"]["ETH"]
    )
    assert max(value for key, value in actual.items() if key[0] == "SOL") == Decimal(
        artifact["necessary_symbol_level_APR_thresholds_percent"]["SOL"]
    )

    for rate, stress_key in (
        (Decimal("5.12"), "at_last_hash_bound_5_12_percent_APR"),
        (Decimal("10"), "at_optimistic_full_one_leg_10_percent_APR"),
    ):
        positive_net = positive_bootstrap = positive_slices = role_count = 0
        slice_count = 0
        for symbol in broad["symbol_results"]:
            if symbol["base_asset"] not in {"BTC", "ETH", "SOL"}:
                continue
            for role in symbol["roles"].values():
                role_count += 1
                days = Decimal(role["duration_days"])
                count = Decimal(role["observation_count"])
                reward = rate * Decimal(100) * days / Decimal(365)
                positive_net += (
                    Decimal(role["net_after_frozen_hurdles_bips"]) + reward > 0
                )
                positive_bootstrap += (
                    Decimal(role["family_adjusted_bootstrap_net_lower_bound_bips"])
                    + reward
                    > 0
                )
                for slice_row in role["slices"].values():
                    slice_count += 1
                    positive_slices += (
                        Decimal(slice_row["net_after_allocated_hurdle_bips"])
                        + reward
                        * Decimal(slice_row["observation_count"])
                        / count
                        > 0
                    )
        stress = artifact["stress_results"][stress_key]
        assert positive_net == stress["positive_role_net_count"] == 0
        assert positive_bootstrap == stress[
            "positive_family_adjusted_bootstrap_role_count"
        ] == 0
        assert positive_slices == stress["positive_regime_slice_count"] == 0
        assert role_count == stress["role_count"] == 9
        assert slice_count == stress["required_regime_slice_count"] == 72


def test_terminal_family_advances_without_resampling_or_acceptance() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert _self_hash(registry) == registry["result_sha256"]
    decision = artifact["adjudication"]
    assert decision["accepted_edge"] is False
    assert decision["stable_profitability_proved"] is False
    assert decision["account_book_or_market_request_justified"] is False
    assert artifact["authority"]["new_market_data_requests"] == 0
    assert artifact["terminal_boundary"]["resample_current_population"] is False

    family = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "binance_broad_crypto_spot_perpetual_funding_carry_"
        "portfolio_margin_single_capital_leg_sensitivity_2026_08_30"
    )
    assert family["canonical_result_sha256"] == artifact["result_sha256"]
    assert "zero_of_72_required_regime_slices_positive" in family["reason"]
