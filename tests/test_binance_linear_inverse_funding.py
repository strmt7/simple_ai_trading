from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
CAPTURE_CONTRACT = (
    ACTION_VALUE
    / "binance-linear-inverse-perpetual-funding-contract-v1-2026-08-31.json"
)
CAPTURE_RESULT = (
    ACTION_VALUE / "binance-linear-inverse-perpetual-funding-result-v1-2026-08-31.json"
)
ADJUDICATION_CONTRACT = (
    ACTION_VALUE
    / "binance-linear-inverse-perpetual-funding-adjudication-contract-v1-2026-08-31.json"
)
ADJUDICATION_RESULT = (
    ACTION_VALUE
    / "binance-linear-inverse-perpetual-funding-adjudication-v1-2026-08-31.json"
)
REGISTRY = ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return _sha256(encoded)


def test_capture_is_hash_bound_and_public_only() -> None:
    contract = _load(CAPTURE_CONTRACT)
    result = _load(CAPTURE_RESULT)
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    implementation = ROOT / contract["implementation"]["path"]
    assert _sha256(implementation.read_bytes()) == contract["implementation"]["sha256"]
    assert len(result["request_receipts"]) == 8
    assert all(row["http_status"] == 200 for row in result["request_receipts"])
    for receipt in result["request_receipts"]:
        path = ROOT / receipt["raw_path"]
        assert _sha256(path.read_bytes()) == receipt["payload_sha256"]
    assert result["authority"]["credentials_used"] is False
    assert result["authority"]["orders_or_transactions"] == 0
    assert result["authority"]["protected_capture_touched"] is False
    assert result["verdict"]["qualified_public_preflight_count"] == 0


def test_fixed_and_lagged_economics_fail_out_of_sample() -> None:
    result = _load(ADJUDICATION_RESULT)
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    expected = {
        "BTC": ("3.95160000", "-28.04840000"),
        "ETH": ("11.87490000", "-20.12510000"),
        "SOL": ("25.29150000", "-6.70850000"),
    }
    for row in result["pair_adjudications"]:
        fixed = row["fixed_orientation_combined_out_of_sample"]
        gross, net = expected[row["asset"]]
        assert fixed["gross_funding_bips"] == gross
        assert fixed["net_after_one_combined_entry_exit_hurdle_bips"] == net
        for role in ("training", "validation", "test"):
            lagged = row["lagged_sign_causal_sensitivity"][role]["2"]
            assert float(lagged["net_after_turnover_bips"]) < 0
    verdict = result["verdict"]
    assert verdict["fixed_combined_out_of_sample_all_failed_frozen_hurdle"] is True
    assert verdict["lagged_sign_all_roles_failed_two_bip_sensitivity"] is True
    assert verdict["accepted_edge"] is False


def test_adjudication_lineage_and_terminal_registry_are_exact() -> None:
    contract = _load(ADJUDICATION_CONTRACT)
    result = _load(ADJUDICATION_RESULT)
    registry = _load(REGISTRY)
    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert contract["capture_result"]["result_sha256"] == _load(CAPTURE_RESULT)[
        "result_sha256"
    ]
    for source in contract["sources"]:
        assert _sha256((ROOT / source["path"]).read_bytes()) == source["sha256"]
    implementation = ROOT / contract["implementation"]["path"]
    assert _sha256(implementation.read_bytes()) == contract["implementation"]["sha256"]
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    family = "binance_BTC_ETH_SOL_COIN_M_inverse_vs_USDM_linear_same_asset_perpetual_funding"
    assert terminal[family]["canonical_result_sha256"] == result["result_sha256"]
    assert registry["accepted_edge_count"] == 29
