from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/model-research/action-value"
LITERATURE_CONTRACT = (
    EVIDENCE / "recent-structural-edge-literature-delta-contract-v1.json"
)
LITERATURE_RESULT = (
    EVIDENCE / "recent-structural-edge-literature-delta-v1-2026-08-29.json"
)
PAPER_CONTRACT = EVIDENCE / "binance-one-way-arbitrage-paper-source-contract-v1.json"
PAPER_RESULT = EVIDENCE / (
    "binance-one-way-arbitrage-paper-source-adjudication-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
LITERATURE_RAW = (
    ROOT / "data/recent-structural-edge-literature-delta-v1/raw/arxiv-query.atom"
)
LITERATURE_JOURNAL = (
    ROOT / "data/recent-structural-edge-literature-delta-v1/request-journal.jsonl"
)
PAPER_RAW = ROOT / "data/binance-one-way-arbitrage-paper-v1/raw/2607.09491v1.pdf"
PAPER_JOURNAL = ROOT / "data/binance-one-way-arbitrage-paper-v1/request-journal.jsonl"

LITERATURE_CONTRACT_HASH = (
    "a3e8060af1e578196a69695dc1176af5710bfa2f86f72c44dd9e2ea12def5e94"
)
LITERATURE_RESULT_HASH = (
    "21c830177ae1e17f18c941a5630df56a2c3dec5c0f26acd14ff740275fe29b06"
)
PAPER_CONTRACT_HASH = "8399fe07a3445c056a627c3c923f379ff6da416ea86ab4555ba932638c6e9c97"
PAPER_RESULT_HASH = "3f9684ed1986cd6cf676482069cda53846e336a15bc4b35141193b8e43406e65"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]
TOOL_HASH = "34f3fd4431278170f72384a9842d0d91a2598d9f20a0f13044ccad92dc36ee54"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def test_literature_contract_result_and_raw_population_are_bound() -> None:
    contract = _load(LITERATURE_CONTRACT)
    result = _load(LITERATURE_RESULT)
    assert contract["contract_sha256"] == LITERATURE_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == LITERATURE_CONTRACT_HASH
    assert result["result_sha256"] == LITERATURE_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == LITERATURE_RESULT_HASH
    assert _sha256(ROOT / contract["implementation"]["path"]) == TOOL_HASH
    assert _sha256(LITERATURE_RAW) == result["capture"]["raw_sha256"]
    assert (
        len(
            ElementTree.fromstring(LITERATURE_RAW.read_bytes()).findall(
                "{http://www.w3.org/2005/Atom}entry"
            )
        )
        == 50
    )
    assert result["population"]["returned_entries"] == 50
    assert result["population"]["published_since_cutoff"] == 8
    assert result["population"]["novel_since_cutoff"] == 7
    assert result["population"]["published_since_cutoff_population_complete"] is True
    assert result["decision"]["accepted_edge"] is False


def test_both_public_requests_were_prejournaled_and_read_only() -> None:
    for path in (LITERATURE_JOURNAL, PAPER_JOURNAL):
        rows = _journal(path)
        assert [row["phase"] for row in rows] == ["intent", "completed"]
        assert rows[0]["method"] == "GET"
        assert rows[0]["request_body_sha256"] == hashlib.sha256(b"").hexdigest()
        assert rows[1]["status_code"] == 200
        assert rows[1]["response_sha256"]
    assert PAPER_RAW.read_bytes().startswith(b"%PDF-")
    assert _sha256(PAPER_RAW) == (
        "8e62af727c8b1b0b7f8dcb00cae23555ed44a6294936f8afc259e7a7ac24787b"
    )


def test_retained_source_adjudication_rejects_every_novel_paper_and_replay() -> None:
    contract = _load(PAPER_CONTRACT)
    result = _load(PAPER_RESULT)
    assert contract["contract_sha256"] == PAPER_CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == PAPER_CONTRACT_HASH
    assert result["result_sha256"] == PAPER_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == PAPER_RESULT_HASH
    assert result["recent_literature_delta"]["novel_papers_adjudicated"] == 7
    assert result["recent_literature_delta"]["actionable_structural_leads"] == 0
    assert len(result["recent_literature_delta"]["rejections"]) == 7
    assert result["paper_mechanism"]["path_identity"].startswith("exactly two trades")
    assert result["paper_economics"]["mean_fee_adjusted_profit_usd_per_sequence"] == {
        "maker_then_taker": "0.15",
        "taker_then_taker": "0.04",
    }
    assert result["registry_adjudication"]["materially_distinct_mechanism"] is False
    assert result["registry_adjudication"]["collector_reopened"] is False
    assert (
        result["registry_adjudication"]["exactly_two_intermediary_extension_reopened"]
        is False
    )
    assert result["registry_adjudication"]["accepted_edge"] is False


def test_registry_strengthens_the_existing_family_without_new_edge() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert len(registry["prioritized_hypotheses"]) == 44
    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 44
    )
    assert family["canonical_artifacts"][-2:] == [
        {
            "path": "docs/model-research/action-value/binance-one-way-arbitrage-paper-source-contract-v1.json",
            "result_sha256": PAPER_CONTRACT_HASH,
        },
        {
            "path": "docs/model-research/action-value/binance-one-way-arbitrage-paper-source-adjudication-v1-2026-08-29.json",
            "result_sha256": PAPER_RESULT_HASH,
        },
    ]
