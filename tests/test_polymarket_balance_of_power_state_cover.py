from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "docs/model-research/action-value/polymarket-2026-balance-of-power-state-cover-adjudication-v1-2026-09-01.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ROOT / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_balance_of_power_state_cover_is_source_bound_and_fee_negative() -> None:
    result = _load(RESULT)
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]

    binding = result["source_binding"]
    assert isinstance(binding, dict)
    for name in ("joint_raw", "house_raw"):
        source = binding[name]
        assert isinstance(source, dict)
        assert _sha256(ROOT / str(source["path"])) == source["file_sha256"]

    state_table = result["state_table"]
    assert isinstance(state_table, list)
    assert len(state_table) == 5
    assert all(
        row["package_payout_pUSD_per_share"] in (1, "at_least_1")
        for row in state_table
    )

    economics = result["economics"]
    decision = result["decision"]
    assert isinstance(economics, dict)
    assert isinstance(decision, dict)
    assert Decimal(str(economics["gross_acquisition_cost_pUSD"])) == Decimal(
        "4.965"
    )
    assert Decimal(str(economics["total_taker_fees_pUSD"])) == Decimal("0.11778")
    assert Decimal(str(economics["after_fee_profit_floor_pUSD"])) == Decimal(
        "-0.08278"
    )
    assert economics["strict_after_fee_positive"] is False
    assert decision["accepted_edge"] is False
    assert decision["book_requests_justified"] == 0

    terminal = {
        row["family"]: row for row in registry["terminal_do_not_repeat"]  # type: ignore[index]
    }
    family = (
        "polymarket_2026_balance_of_power_to_Democratic_House_marginal_"
        "four_leg_state_cover_2026_09_01"
    )
    assert terminal[family]["canonical_result_sha256"] == result["result_sha256"]
    source = audit["source_binding"]
    assert isinstance(source, dict)
    assert source["registry_result_sha256"] == registry["result_sha256"]
