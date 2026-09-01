from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/action-value/polymarket-qwen-flash-deadline-ladder-contract-v1-2026-09-01.json"
)
RESULT_PATH = (
    ROOT
    / "docs/model-research/action-value/polymarket-qwen-flash-deadline-ladder-result-v1-2026-09-01.json"
)
REGISTRY_PATH = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RAW_PATH = ROOT / "data/polymarket-qwen-flash-deadline-ladder-v1/raw/01-exact-event.raw"
JOURNAL_PATH = (
    ROOT / "data/polymarket-qwen-flash-deadline-ladder-v1/request-journal.jsonl"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_qwen_deadline_ladder_is_durable_and_fail_closed() -> None:
    contract = _load(CONTRACT_PATH)
    result = _load(RESULT_PATH)
    registry = _load(REGISTRY_PATH)

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert (
        hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
        == (result["capture"]["receipt"]["response_sha256"])
    )

    journal = [json.loads(line) for line in JOURNAL_PATH.read_text().splitlines()]
    assert [row["phase"] for row in journal] == ["intent", "completed"]
    assert journal[1] == result["capture"]["receipt"]
    assert result["screen"]["package"]["displayed_price_sum_pUSD"] == "1.02"
    assert result["screen"]["package"]["passes_strict_displayed_gross_gate"] is False
    assert result["adjudication"] == {
        "accepted_edge": False,
        "deployment_ready": False,
        "profitability_claim": False,
        "status": "rejected_before_books_fees_and_onchain_requests",
    }
    assert result["authority"] == contract["authority"]
    assert result["authority"]["public_unauthenticated_read_only_requests"] == 1
    assert result["authority"]["book_requests"] == 0
    assert result["authority"]["credentials_used"] is False
    assert result["authority"]["protected_capture_touched"] is False

    rank31 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 31
    )
    artifacts = {
        row["path"]: row["result_sha256"] for row in rank31["canonical_artifacts"]
    }
    assert (
        artifacts[str(CONTRACT_PATH.relative_to(ROOT)).replace("\\", "/")]
        == (contract["contract_sha256"])
    )
    assert (
        artifacts[str(RESULT_PATH.relative_to(ROOT)).replace("\\", "/")]
        == (result["result_sha256"])
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_Qwen_Flash_3_9_October_December_deadline_ladder_2026_09_01"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
