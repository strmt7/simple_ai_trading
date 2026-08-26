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
EXPECTED_RESULT_HASH = (
    "0cf6e3aae168e0c483634e78fd824a80be9e58269f02e9b01a6d9c9c46578a8f"
)
EXPECTED_CONTRACT_HASH = (
    "3f2b39ab03f246d0d31a015d8aff40c0a5e806eb4ccd646bcefea04f48f9a6d1"
)
EXPECTED_TOOL_HASH = "413baf5946289e1af14e703a307749c1af1db0a3339de92ef24127316fa7c115"


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
