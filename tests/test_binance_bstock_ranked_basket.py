from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ARTIFACT_PATH = Path(
    "docs/model-research/action-value/binance-bstock-ranked-basket-v1-2026-08-26.json"
)
CONTRACT_PATH = Path(
    "docs/model-research/action-value/binance-bstock-ranked-basket-contract-v1.json"
)
TOOL_PATH = Path("tools/adjudicate_binance_bstock_ranked_basket.py")
ZERO_MAKER_COUNTERFACTUAL_PATH = Path(
    "docs/model-research/action-value/"
    "binance-bstocks-zero-maker-carry-retained-counterfactual-v1-2026-08-29.json"
)
REGISTRY_PATH = Path(
    "docs/model-research/structural-edge-priority-registry-v1.json"
)
EXPECTED_RESULT_HASH = (
    "0cf6e3aae168e0c483634e78fd824a80be9e58269f02e9b01a6d9c9c46578a8f"
)
EXPECTED_CONTRACT_HASH = (
    "3f2b39ab03f246d0d31a015d8aff40c0a5e806eb4ccd646bcefea04f48f9a6d1"
)
EXPECTED_TOOL_HASH = "413baf5946289e1af14e703a307749c1af1db0a3339de92ef24127316fa7c115"
EXPECTED_ZERO_MAKER_COUNTERFACTUAL_HASH = (
    "2af1504748f51ad36c18c76162a91e82803395311932d73f1809d2781dfc4fb7"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_ranked_basket_artifact_contract_and_tool_are_source_bound() -> None:
    artifact = _load(ARTIFACT_PATH)
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"] == {
        "path": TOOL_PATH.as_posix(),
        "sha256": EXPECTED_TOOL_HASH,
    }
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH

    contract = _load(CONTRACT_PATH)
    contract_body = dict(contract)
    assert contract_body.pop("contract_sha256") == EXPECTED_CONTRACT_HASH
    assert (
        hashlib.sha256(_canonical_json(contract_body).encode("ascii")).hexdigest()
        == EXPECTED_CONTRACT_HASH
    )


def test_frozen_top_twenty_percent_selector_is_reconstructed_exactly() -> None:
    artifact = _load(ARTIFACT_PATH)
    assert artifact["selection"] == {
        "eligible_count": 57,
        "selected_count": 12,
        "selected_tickers": [
            "LITE",
            "SOXL",
            "QCOM",
            "KORU",
            "CRCL",
            "BMNR",
            "GLW",
            "AMD",
            "MVLL",
            "USAR",
            "COHR",
            "NBIS",
        ],
        "top_fraction": "0.20",
        "weights": "equal_notional_per_selected_symbol",
    }


def test_ranked_basket_fails_validation_test_and_every_regime_slice() -> None:
    artifact = _load(ARTIFACT_PATH)
    validation = artifact["portfolio_roles"]["validation"]
    test = artifact["portfolio_roles"]["test"]
    assert Decimal(
        validation["equal_weight_mean_net_after_frozen_hurdles_bips"]
    ) == Decimal("-81.80462067002790461694571283")
    assert Decimal(test["equal_weight_mean_net_after_frozen_hurdles_bips"]) == Decimal(
        "-108.1975271953745983426348723"
    )
    assert validation["positive_symbol_count"] == 1
    assert test["positive_symbol_count"] == 0
    assert validation["passes"] is False
    assert test["passes"] is False
    assert all(
        slice_row["passes"] is False for slice_row in validation["slices"].values()
    )
    assert all(slice_row["passes"] is False for slice_row in test["slices"].values())


def test_ranked_basket_is_terminally_rejected_without_scope_or_trade_authority() -> (
    None
):
    artifact = _load(ARTIFACT_PATH)
    assert artifact["verdict"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "persistent_after_frozen_sensitivity_research_candidate": False,
        "status": "ranked_basket_rejected_without_parameter_retry",
        "test_pass": False,
        "trading_authority": False,
        "validation_pass": False,
    }
    assert artifact["authority"] == {
        "credentials_used": False,
        "funds_used": False,
        "orders_placed": False,
        "trading_authority": False,
    }
    assert (
        "explicit scope expansion beyond BTC ETH and SOL"
        in artifact["blocking_evidence"]
    )


def test_zero_maker_counterfactual_is_hash_bound_and_uses_no_new_access() -> None:
    artifact = _load(ZERO_MAKER_COUNTERFACTUAL_PATH)
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_ZERO_MAKER_COUNTERFACTUAL_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_ZERO_MAKER_COUNTERFACTUAL_HASH
    )
    assert artifact["authority"] == {
        "retained_data_only": True,
        "new_http_requests": 0,
        "credentials_used": False,
        "account_state_accessed": False,
        "orders_transfers_conversions_subscriptions_or_account_changes": 0,
        "paper_or_live_trading_authority": False,
    }


def test_zero_maker_upper_bound_still_fails_validation_and_test() -> None:
    artifact = _load(ZERO_MAKER_COUNTERFACTUAL_PATH)
    upper_bound = artifact["retained_upper_bound"]
    validation = upper_bound["validation"]
    test = upper_bound["test"]

    assert Decimal(validation["upper_bound_mean_net_bips"]) == Decimal(
        "-21.80462067002790461694571283"
    )
    assert Decimal(
        validation["upper_bound_cross_sectional_bootstrap_lower_bound_bips"]
    ) == Decimal("-42.0480030821917808219178082")
    assert Decimal(validation["upper_bound_worst_leave_one_symbol_out_net_bips"]) == (
        Decimal("-34.49344899162861491628614918")
    )
    assert validation["passes"] is False

    assert Decimal(test["upper_bound_mean_net_bips"]) == Decimal(
        "-48.1975271953745983426348723"
    )
    assert Decimal(
        test["upper_bound_cross_sectional_bootstrap_lower_bound_bips"]
    ) == Decimal("-55.4780525299129037713512599")
    assert Decimal(test["upper_bound_worst_leave_one_symbol_out_net_bips"]) == (
        Decimal("-50.7783063356164383561643835")
    )
    assert test["upper_bound_positive_symbol_count"] == 0
    assert test["selected_symbol_count"] == 12
    assert Decimal(test["upper_bound_best_symbol_net_bips"]) < 0
    assert test["passes"] is False


def test_zero_maker_counterfactual_prohibits_a_fresh_market_study() -> None:
    artifact = _load(ZERO_MAKER_COUNTERFACTUAL_PATH)
    assert artifact["adjudication"] == {
        "accepted_edge": False,
        "candidate_for_fresh_book_or_funding_study": False,
        "deployment_ready": False,
        "market_direction_forecast_required": False,
        "profitability_claim": False,
        "trading_authority": False,
        "status": (
            "terminal_family_not_reopened_by_bStocks_zero_maker_promotion_even_"
            "under_full_60_bip_execution_stress_erasure"
        ),
    }
    registry = _load(REGISTRY_PATH)
    hypothesis = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"]
        == "bstock_reference_conversion_and_delta_neutral_perpetual_funding"
    )
    assert {
        "path": ZERO_MAKER_COUNTERFACTUAL_PATH.as_posix(),
        "result_sha256": EXPECTED_ZERO_MAKER_COUNTERFACTUAL_HASH,
    } in hypothesis["canonical_artifacts"]
    assert "do_not_poll" in hypothesis["next_action"]
    assert "refresh_funding_or_books" in hypothesis["next_action"]
