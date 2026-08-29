import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/model-research/action-value/polymarket-official-docs-mechanism-inventory-contract-v1.json"
RESULT = ROOT / "docs/model-research/action-value/polymarket-official-docs-mechanism-inventory-v1-2026-08-29.json"


def canonical_hash(document: dict, field: str) -> str:
    payload = dict(document)
    expected = payload.pop(field)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
    return actual


def test_polymarket_official_docs_inventory_is_one_use_and_source_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    canonical_hash(contract, "contract_sha256")
    canonical_hash(result, "result_sha256")
    assert result["contract"]["contract_sha256"] == contract["contract_sha256"]

    source = result["source"]
    raw = ROOT / source["raw_path"]
    journal = ROOT / source["journal_path"]
    raw_bytes = raw.read_bytes()
    assert len(raw_bytes) == source["raw_bytes"]
    assert hashlib.sha256(raw_bytes).hexdigest() == source["raw_sha256"]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == source["journal_sha256"]

    entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [entry["phase"] for entry in entries] == ["intent", "completed"]
    assert entries[0]["requested_before_utc"] == entries[1]["requested_before_utc"]
    assert entries[1]["status_code"] == 200
    assert entries[1]["response_sha256"] == source["raw_sha256"]

    links = re.findall(
        rb"^- \[[^]]+\]\((https://docs\.polymarket\.com/[^)]+\.md)\)(?::|$)",
        raw_bytes,
        re.MULTILINE,
    )
    pages = [url for url in links if b"/_llms/" not in url]
    assert len(pages) == source["current_top_level_english_markdown_pages"] == 80
    assert source["network_requests_used"] == contract["authority"]["network_requests_permitted"] == 1


def test_inventory_stops_without_spurious_edge_or_market_data() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    classification = result["classification"]
    decision = result["decision"]
    authority = result["authority_used"]

    assert classification["distinct_novel_structural_mechanisms"] == 0
    assert classification["new_market_data_tests_authorized"] == 0
    assert classification["accepted_edges_added"] == 0
    assert classification["accepted_edge_count_after"] == 21
    assert decision["promote_new_hypothesis"] is False
    assert decision["request_linked_pages"] is False
    assert decision["request_market_data"] is False
    assert authority == {
        "public_unauthenticated_read_only": True,
        "credentials": False,
        "account_state": False,
        "orders_or_mutations": False,
        "protected_capture": False,
    }

    classified_pages = {
        page
        for group_name in ("registry_family_groups", "non_edge_operational_groups")
        for group in classification[group_name]
        for page in group["pages"]
    }
    required_pages = {
        "https://docs.polymarket.com/trading/combos/collateral-return.md",
        "https://docs.polymarket.com/programs/taker-rebates.md",
        "https://docs.polymarket.com/programs/maker-rebates.md",
        "https://docs.polymarket.com/programs/liquidity-rewards.md",
        "https://docs.polymarket.com/programs/referral-program.md",
        "https://docs.polymarket.com/programs/builders/fees.md",
        "https://docs.polymarket.com/trading/matching-engine.md",
        "https://docs.polymarket.com/trading/bridge/quote.md",
        "https://docs.polymarket.com/perps/learn-about-trading/funding.md",
        "https://docs.polymarket.com/perps/referral-program.md",
    }
    assert required_pages <= classified_pages

    registry = json.loads(
        (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["accepted_edge_count"] == 21
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_official_documentation_mechanism_inventory_snapshot"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
