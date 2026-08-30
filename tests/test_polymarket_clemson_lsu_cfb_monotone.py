from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from tools.adjudicate_polymarket_exact_cfb_monotone_prefilter import (
    _margin_markets,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CAPTURE_CONTRACT = BASE / (
    "polymarket-clemson-lsu-exact-event-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = BASE / ("polymarket-clemson-lsu-exact-event-result-v1-2026-08-30.json")
CORRECTION_CONTRACT = BASE / (
    "polymarket-clemson-lsu-cfb-monotone-prefilter-contract-v1-2026-08-30.json"
)
RESULT = BASE / ("polymarket-clemson-lsu-cfb-monotone-prefilter-v1-2026-08-30.json")
RAW_DIR = ROOT / (
    "docs/model-research/polymarket/raw/clemson-lsu-exact-event-v1-2026-08-30"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_one_frozen_event_request_is_bound_to_the_zero_network_correction() -> None:
    capture_contract = _load(CAPTURE_CONTRACT)
    source = _load(SOURCE_RESULT)
    correction = _load(CORRECTION_CONTRACT)
    journal = [
        json.loads(line)
        for line in (RAW_DIR / "request-journal.jsonl")
        .read_text(encoding="ascii")
        .splitlines()
    ]
    raw = (RAW_DIR / "event.raw.json").read_bytes()

    assert (
        _self_hash(capture_contract, "contract_sha256")
        == capture_contract["contract_sha256"]
    )
    assert _self_hash(source, "result_sha256") == source["result_sha256"]
    assert _self_hash(correction, "contract_sha256") == correction["contract_sha256"]
    for implementation in correction["implementations"]:
        path = ROOT / implementation["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == implementation["sha256"]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[-1]["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert source["capture"]["active_accepting_market_count"] == 4
    assert correction["authority"]["network_requests"] == 0
    assert correction["parser_correction"]["retained_source_reused"] is True


def test_cfb_parser_binds_both_favorite_orientations_and_rejects_line_drift() -> None:
    source = _load(SOURCE_RESULT)
    markets = source["discovery"]["active_accepting_markets"]
    moneyline = next(row for row in markets if row["sportsMarketType"] == "moneyline")
    _, _, retained_rows = _margin_markets(markets, moneyline)
    assert [row["threshold"] for row in retained_rows] == [-10, -9, 1]

    reversed_spread = deepcopy(
        next(row for row in markets if row["sportsMarketType"] == "spreads")
    )
    reversed_spread["outcomes"] = '["Clemson", "LSU"]'
    reversed_spread["outcomePrices"] = '["0.4", "0.6"]'
    reversed_spread["line"] = -3.5
    reversed_spread["description"] = (
        'This market will resolve to "Clemson" if Clemson win the game by 4 or '
        'more points. Otherwise, this market will resolve to "LSU". If the game '
        "is canceled entirely, with no make-up game, this market will resolve "
        "50-50."
    )
    _, _, reversed_rows = _margin_markets([moneyline, reversed_spread], moneyline)
    assert [row["threshold"] for row in reversed_rows] == [1, 4]

    reversed_spread["description"] = reversed_spread["description"].replace(
        "by 4 or more", "by 5 or more"
    )
    with pytest.raises(RuntimeError, match="does not bind its exact line"):
        _margin_markets([moneyline, reversed_spread], moneyline)


def test_complete_lattice_rejects_before_books_and_registry_terminalizes_event() -> (
    None
):
    result = _load(RESULT)
    registry = _load(REGISTRY)

    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert result["payoff_proof"]["complete_relation_count"] == 3
    prefilter = result["rejection_only_gamma_prefilter"]
    assert prefilter["candidate_count_strictly_below_payout_floor"] == 0
    assert Decimal(
        prefilter["best_relation"]["displayed_price_sum_per_share_pUSD"]
    ) == Decimal("1.045")
    assert result["authority"]["book_requests"] == 0
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 30
    )
    assert {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": result["result_sha256"],
    } in family["canonical_artifacts"]
    assert any(
        row["canonical_result_sha256"] == result["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )
