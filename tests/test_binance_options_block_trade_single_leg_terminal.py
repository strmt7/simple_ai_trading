from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-block-trade-single-leg-source-contract-v1-2026-09-01.json"
)
ADJUDICATION = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-block-trade-single-leg-source-format-failure-"
    "adjudication-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_single_leg_limit_terminalizes_block_trade_box_workaround() -> None:
    contract = _load(CONTRACT)
    adjudication = _load(ADJUDICATION)
    registry = _load(REGISTRY)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(adjudication, "result_sha256") == adjudication["result_sha256"]
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]

    capture = adjudication["frozen_capture"]
    assert isinstance(capture, dict)
    raw = ROOT / str(capture["raw_path"])
    journal = ROOT / str(capture["journal_path"])
    assert raw.read_bytes().startswith(b"%PDF-1.5")
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == capture["raw_sha256"]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == capture["journal_sha256"]

    proved = adjudication["source_proved"]
    decision = adjudication["adjudication"]
    assert isinstance(proved, dict)
    assert isinstance(decision, dict)
    assert proved["maximum_leg_count"] == 1
    assert decision["block_trade_two_leg_vertical_path"] is False
    assert decision["block_trade_four_leg_box_path"] is False
    assert decision["public_forward_profit_floor"] == "0"

    hypotheses = registry["prioritized_hypotheses"]
    terminal = registry["terminal_do_not_repeat"]
    assert isinstance(hypotheses, list)
    assert isinstance(terminal, list)
    rank_37 = next(row for row in hypotheses if row["priority_rank"] == 37)
    assert "capped_at_one_leg" in rank_37["current_status"]
    assert any(
        row["canonical_result_sha256"] == adjudication["result_sha256"]
        for row in terminal
    )
