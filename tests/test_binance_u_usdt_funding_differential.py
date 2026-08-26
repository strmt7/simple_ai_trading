from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-u-usdt-funding-differential-v1-2026-08-26.json"
)
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-u-usdt-funding-differential-contract-v1.json"
)
TOOL_PATH = ROOT / "tools" / "screen_binance_u_usdt_funding_differential.py"
EXPECTED_RESULT_HASH = (
    "486b1aa261ae41fd8d8aeb19f0fea5bb01305d24927ccd72624bdd8afb7895d7"
)
EXPECTED_CONTRACT_HASH = (
    "0d7ee07f3641e8e2978b13ff11dec14feb34732037a8572b7c70b73f13dc1003"
)
EXPECTED_TOOL_HASH = "a1a59851b1ed5a6ab5ccca099a69afc48665c0202fe56503a55e3e12f455acf2"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_u_usdt_artifact_contract_and_tool_are_source_bound() -> None:
    artifact = _load(ARTIFACT_PATH)
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert (
        hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()
        == EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"] == {
        "path": "tools/screen_binance_u_usdt_funding_differential.py",
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


def test_training_selected_static_orientation_without_sealed_role_adaptation() -> None:
    rows = {row["base"]: row for row in _load(ARTIFACT_PATH)["evaluations"]}
    assert set(rows) == {"BTC", "ETH"}
    assert all(row["aligned_row_count"] == 171 for row in rows.values())
    assert rows["BTC"]["symbols"] == {"long": "BTCU", "short": "BTCUSDT"}
    assert rows["ETH"]["symbols"] == {"long": "ETHU", "short": "ETHUSDT"}
    assert all(row["roles"]["training"]["row_count"] == 85 for row in rows.values())
    assert all(row["roles"]["validation"]["row_count"] == 43 for row in rows.values())
    assert all(row["roles"]["test"]["row_count"] == 43 for row in rows.values())


def test_btc_and_eth_fail_validation_test_and_economic_hurdles() -> None:
    artifact = _load(ARTIFACT_PATH)
    assert artifact["verdict"]["public_persistence_candidate_count"] == 0
    assert artifact["verdict"]["book_escalation_permitted"] is False
    assert artifact["verdict"]["accepted_edge"] is False
    rows = {row["base"]: row for row in artifact["evaluations"]}
    assert rows["BTC"]["roles"]["validation"]["gross_bips"] == "-15.51750000"
    assert rows["BTC"]["roles"]["test"]["gross_bips"] == "-8.62440000"
    assert rows["ETH"]["roles"]["validation"]["gross_bips"] == "-20.88640000"
    assert rows["ETH"]["roles"]["test"]["gross_bips"] == "-12.83510000"
    assert all(
        role["net_after_frozen_hurdles_bips"].startswith("-")
        for row in rows.values()
        for role in row["roles"].values()
    )
    assert all(row["public_persistence_candidate"] is False for row in rows.values())
