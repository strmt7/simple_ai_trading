from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "polymarket-all-category-maker-rebate-source-contract-v1-2026-08-30.json"
)
SOURCE_RESULT = BASE / (
    "polymarket-all-category-maker-rebate-source-result-v1-2026-08-30.json"
)
ARTIFACT = BASE / (
    "polymarket-all-category-realized-organic-maker-rebate-overlay-"
    "v1-2026-08-30.json"
)
CONFLICT_ADJUDICATION = BASE / (
    "polymarket-sports-maker-rebate-source-conflict-adjudication-"
    "v1-2026-08-30.json"
)
CONFLICT_SOURCE_ARTIFACTS = (
    (
        BASE
        / "polymarket-sports-maker-rebate-current-source-conflict-contract-v1-2026-08-30.json",
        "contract_sha256",
    ),
    (
        BASE
        / "polymarket-sports-maker-rebate-current-source-conflict-result-v1-2026-08-30.json",
        "result_sha256",
    ),
    (
        BASE
        / "polymarket-sports-maker-rebate-cross-source-conflict-contract-v1-2026-08-30.json",
        "contract_sha256",
    ),
    (
        BASE
        / "polymarket-sports-maker-rebate-cross-source-conflict-source-result-v1-2026-08-30.json",
        "result_sha256",
    ),
)
RAW = ROOT / (
    "docs/model-research/polymarket/raw/"
    "all-category-maker-rebate-source-v1-2026-08-30/01-maker-rebates.raw.md"
)
CURRENT_REBATE_RAW = ROOT / (
    "docs/model-research/polymarket/raw/"
    "sports-maker-rebate-current-source-conflict-v1-2026-08-30/"
    "01-maker-rebates.raw.md"
)
CURRENT_FEES_RAW = ROOT / (
    "docs/model-research/polymarket/raw/"
    "sports-maker-rebate-cross-source-conflict-v1-2026-08-30/01-fees.raw.md"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _table(markdown: str, header: str) -> dict[str, list[str]]:
    lines = markdown.splitlines()
    start = lines.index(header) + 2
    rows: dict[str, list[str]] = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        rows[cells[0]] = cells[1:]
    return rows


def test_one_use_failure_and_retained_bytes_are_source_bound() -> None:
    contract = json.loads(CONTRACT.read_bytes())
    result = json.loads(SOURCE_RESULT.read_bytes())
    artifact = json.loads(ARTIFACT.read_bytes())
    conflict = json.loads(CONFLICT_ADJUDICATION.read_bytes())

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(result, "result_sha256") == result["result_sha256"]
    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert _self_hash(conflict, "result_sha256") == conflict["result_sha256"]
    assert result["source_gate"]["passed"] is False
    assert artifact["failed_contract_adjudication"]["no_retry_or_alias"] is True
    assert artifact["failed_contract_adjudication"]["discovery_values_excluded"] is True

    for path, field in CONFLICT_SOURCE_ARTIFACTS:
        payload = json.loads(path.read_bytes())
        assert _self_hash(payload, field) == payload[field]

    for binding in conflict["source_binding"].values():
        if "file_sha256" not in binding:
            continue
        payload = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["file_sha256"]

    for binding in artifact["source_binding"].values():
        payload = (ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == binding["file_sha256"]


def test_current_primary_markdown_reconstructs_exact_category_terms() -> None:
    artifact = json.loads(ARTIFACT.read_bytes())
    markdown = RAW.read_text(encoding="utf-8")
    rebate_rows = _table(
        markdown, "| Category        | Maker Rebate | Distribution Method |"
    )
    fee_rows = _table(
        markdown, "| Category        | Taker Fee Rate | Maker Fee Rate |"
    )

    expected = {
        row["category"]: (
            row["maker_rebate_fraction"],
            row["taker_fee_rate"],
        )
        for row in artifact["current_program_contract"]["eligible_categories"]
    }
    assert set(expected) == {
        "Crypto",
        "Sports",
        "Finance",
        "Politics",
        "Economics",
        "Culture",
        "Weather",
        "Other / General",
        "Mentions",
        "Tech",
    }
    for category, (rebate, taker_fee) in expected.items():
        assert rebate_rows[category] == [f"{int(float(rebate) * 100)}%", "Fee-curve weighted"]
        assert fee_rows[category] == [taker_fee, "0"]

    assert rebate_rows["Sports"][0] == "15%"
    assert rebate_rows["Geopolitics"] == ["—", "Fee-free"]
    assert fee_rows["Geopolitics"] == ["0", "0"]
    assert "minimum accrued rebate of **\\$1 pUSD**" in markdown
    assert "Totals are calculated per market" in markdown
    assert "sole discretion of Polymarket" in markdown


def test_current_sports_sources_fail_closed_on_cash_label_conflict() -> None:
    artifact = json.loads(ARTIFACT.read_bytes())
    conflict = json.loads(CONFLICT_ADJUDICATION.read_bytes())
    rebate_markdown = CURRENT_REBATE_RAW.read_text(encoding="utf-8")
    fees_markdown = CURRENT_FEES_RAW.read_text(encoding="utf-8")
    rebate_rows = _table(
        rebate_markdown, "| Category        | Maker Rebate | Distribution Method |"
    )
    fee_rows = _table(
        fees_markdown,
        "| Category        | Taker Fee Rate | Maker Fee Rate | Maker Rebate |",
    )

    assert rebate_rows["Sports"] == ["15%", "Fee-curve weighted"]
    assert fee_rows["Sports"] == ["0.05", "0", "15%"]
    assert "Paid daily in pUSD" in rebate_markdown
    assert "Taker fees are calculated in USDC" in fees_markdown
    assert conflict["adjudication"][
        "public_sports_rebate_fraction_for_forward_economics"
    ] is None
    assert conflict["adjudication"]["public_forward_profit_floor"] == "0"
    assert "asset actually received" in conflict["realization_rule"]
    assert artifact["current_program_contract"]["currency"].startswith("fail_closed")
    assert "asset actually received" in artifact["adjudication"]["accepted_scope"]


def test_scope_extension_keeps_fail_closed_count_and_registry_binding() -> None:
    artifact = json.loads(ARTIFACT.read_bytes())
    registry = json.loads(REGISTRY.read_bytes())
    decision = artifact["adjudication"]

    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert decision["accepted_edge"] is True
    assert decision["scope_extension_not_new_edge"] is True
    assert decision["accepted_edge_count_before"] == 29
    assert decision["accepted_edge_count_after"] == 29
    assert decision["standalone_market_making_strategy_accepted"] is False
    assert decision["fresh_hypothetical_order_profit_proved"] is False
    assert decision["public_forward_profit_floor_pusd"] == "0"
    assert artifact["authority"]["orders_or_cancellations"] == 0

    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 17
    )
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    conflict = json.loads(CONFLICT_ADJUDICATION.read_bytes())
    assert {
        "path": CONFLICT_ADJUDICATION.relative_to(ROOT).as_posix(),
        "result_sha256": conflict["result_sha256"],
    } in family["canonical_artifacts"]
    assert "current_fee_enabled_Crypto_Sports" in family[
        "realized_rebate_scope_extension"
    ]
    assert "scope extension" in registry["accepted_edge_scope_amendment"]
