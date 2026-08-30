from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTIER_PATH = ROOT / "docs/model-research/action-value/binance-stablecoin-issuer-parity-retained-frontier-v1-2026-08-30.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_retained_stablecoin_frontier_selects_only_the_strongest_admitted_lead() -> None:
    frontier = _load(FRONTIER_PATH)
    rows = frontier["retained_frontier"]
    selection = frontier["selection"]

    assert _self_hash(frontier) == frontier["result_sha256"]
    assert len(rows) == frontier["method"]["pair_count"] == 20
    assert sum(row["gross_best_parity_sensitivity_bips"] is not None for row in rows) == 16
    assert selection["highest_retained_gross_source_admitted_pair"] == "TUSDUSDT"
    assert selection["new_issuer_source_requests_justified"] == 0
    assert selection["selected_existing_hypothesis_rank"] == 45
    assert frontier["adjudication"]["after_all_cost_profit_floor_usd"] == "0"
    assert frontier["authority"]["network_requests"] == 0


def test_frontier_binds_the_existing_tusd_and_euri_adjudications() -> None:
    frontier = _load(FRONTIER_PATH)
    binding = frontier["source_binding"]

    for prefix in ("tusd_candidate", "euri_adjudication"):
        source = _load(ROOT / binding[f"{prefix}_path"])
        assert source["result_sha256"] == binding[f"{prefix}_result_sha256"]
