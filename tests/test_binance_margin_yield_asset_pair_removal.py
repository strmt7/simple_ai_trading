from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/action-value/binance-margin-yield-asset-isolated-pair-removal-contract-v1-2026-08-31.json"
)
RESULT = (
    ROOT
    / "docs/model-research/action-value/binance-margin-yield-asset-isolated-pair-removal-terminal-v1-2026-08-31.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    claimed = str(body.pop("result_sha256"))
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == claimed
    return claimed


def test_margin_pair_removal_is_source_bound_and_terminal() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    assert _self_hash(contract) == result["contract"]["result_sha256"]
    assert _sha256(CONTRACT) == result["contract"]["file_sha256"]
    assert _self_hash(result) == result["result_sha256"]

    source = result["source_binding"]
    raw = ROOT / source["raw_path"]
    journal = ROOT / source["journal_path"]
    assert raw.stat().st_size == source["raw_bytes"]
    assert _sha256(raw) == source["raw_sha256"]
    assert _sha256(journal) == source["journal_sha256"]

    article = _load(raw)["data"]
    assert article["code"] == source["article_code"]
    assert article["title"] == result["primary_evidence"]["title"]
    body = article["body"]
    for pair in result["primary_evidence"]["affected_pairs_of_interest"]:
        assert pair in body

    assert result["authority"]["market_data_requests_after_article"] == 0
    assert result["economics"]["forced_flow_direction_proved"] is False
    assert result["economics"]["direction_independent_payoff_floor_proved"] is False
    assert result["verdict"]["exact_episode_terminal"] is True
    assert result["verdict"]["accepted_edge"] is False


def test_registry_binds_the_exact_terminal_episode_without_global_pins() -> None:
    registry = _load(REGISTRY)
    _self_hash(registry)
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    row = terminal[
        "binance_WBETH_BNSOL_BFUSD_isolated_margin_pair_removal_forced_settlement_2026_09_03"
    ]
    assert row["canonical_result_sha256"] == _load(RESULT)["result_sha256"]
    assert "do_not_poll_books_or_preposition" in row["reason"]
