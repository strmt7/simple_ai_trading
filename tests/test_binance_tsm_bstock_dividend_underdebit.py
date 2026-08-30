from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-tsm-bstock-dividend-underdebit-contract-v1-2026-08-30.json"
)
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "binance-tsm-bstock-dividend-underdebit-v1-2026-08-30.json"
)
JOURNAL = ROOT / "data/binance-tsm-bstock-dividend-underdebit-v1/journal.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "4c2bfffa8b67e53328f1fde9a9395fc210613635ae624c450c9ca631a3d9553e"
RESULT_HASH = "82acc3529620f1d9c728eac24ea0fb256f228e4065650c766dff057d198a5e60"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_result_and_retained_lineage_reconstruct() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert result["result_sha256"] == RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RESULT_HASH
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == (implementation["sha256"])
    )
    for source in contract["retained_sources"].values():
        assert (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            == (source["sha256"])
        )


def test_prior_special_funding_rejects_before_books() -> None:
    result = _load(RESULT)
    event = result["historical_event"]

    debit = -(
        Decimal(event["special_funding_rate"])
        * Decimal(event["special_mark_price_usdt"])
    )
    assert debit == Decimal("0.9525516930000000")
    assert debit == Decimal(event["matched_short_special_debit_usdt"])
    assert Decimal(event["gross_dividend_minus_special_debit"]) < 0
    assert Decimal(event["net_dividend_minus_special_debit"]) < 0
    assert result["adjudication"]["book_requests_justified"] is False
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["public_after_cost_profit_floor_usdt"] == "0"


def test_two_issuer_pages_and_journal_are_exact() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)
    journal = _load(JOURNAL)

    assert journal["state"] == "completed"
    assert journal["contract_sha256"] == CONTRACT_HASH
    assert journal["result_sha256"] == RESULT_HASH
    assert len(journal["requests"]) == 2
    for source, receipt, request in zip(
        contract["issuer_sources"],
        result["sources"]["issuer_pages"],
        journal["requests"],
        strict=True,
    ):
        raw = (ROOT / receipt["raw_path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["response_sha256"]
        text = raw.decode("utf-8")
        assert all(value in text for value in source["required_text"])
        assert request["state"] == "received"
        assert request["status_code"] == 200


def test_registry_self_hash_and_family_update_reconstruct() -> None:
    registry = _load(REGISTRY)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert registry["accepted_edge_count"] == 21
    assert len(registry["prioritized_hypotheses"]) == 44
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"]
        == "binance_NOK_bStock_dividend_perpetual_special_funding_underdebit"
    )
    assert row["canonical_artifacts"][-1] == {
        "path": RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": RESULT_HASH,
    }
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "binance_TSMB_2026_09_16_dividend_perpetual_underdebit_candidate"
    )
    assert terminal["canonical_result_sha256"] == RESULT_HASH
