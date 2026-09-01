from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict[str, object]:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_hash(value: dict[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field)
    return hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_hash(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_public_company_rank_assignment_fails_closed_and_is_bound() -> None:
    adjudication = _load(
        "docs/model-research/action-value/"
        "polymarket-public-company-top3-assignment-source-adjudication-"
        "v1-2026-09-01.json"
    )
    registry = _load(
        "docs/model-research/structural-edge-priority-registry-v1.json"
    )
    audit = _load(
        "docs/model-research/action-value/"
        "accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
    )

    assert (
        _canonical_hash(adjudication, "result_sha256")
        == adjudication["result_sha256"]
    )
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]
    assert adjudication["rule_alignment"][  # type: ignore[index]
        "all_different_rank_assignment_floor_source_proved"
    ] is False
    assert adjudication["authority"][  # type: ignore[index]
        "bestBid_or_bestAsk_fields_accessed_for_decision"
    ] is False
    assert adjudication["adjudication"][  # type: ignore[index]
        "book_request_justified"
    ] is False

    labels = ("largest", "second-largest", "third-largest")
    ranks = ("rank_1", "rank_2", "rank_3")
    for label, rank in zip(labels, ranks, strict=True):
        contract = _load(
            "docs/model-research/action-value/"
            f"polymarket-public-company-{label}-metadata-contract-v1-2026-09-01.json"
        )
        capture = _load(
            "docs/model-research/action-value/"
            f"polymarket-public-company-{label}-metadata-capture-result-v1-2026-09-01.json"
        )
        assert _canonical_hash(contract, "contract_sha256") == contract[
            "contract_sha256"
        ]
        assert _canonical_hash(capture, "result_sha256") == capture["result_sha256"]
        assert capture["contract"]["sha256"] == contract[  # type: ignore[index]
            "contract_sha256"
        ]
        source = adjudication["source_binding"][rank]  # type: ignore[index]
        assert source["capture_result_sha256"] == capture["result_sha256"]
        assert source["contract_sha256"] == contract["contract_sha256"]
        assert source["raw_sha256"] == _file_hash(source["raw_path"])

    family = (
        "polymarket_public_company_first_second_third_rank_cross_event_"
        "all_different_assignment_2026_09_01"
    )
    terminal = {
        row["family"]: row
        for row in registry["terminal_do_not_repeat"]  # type: ignore[index]
    }
    assert len(registry["terminal_do_not_repeat"]) == 141  # type: ignore[arg-type]
    assert terminal[family]["canonical_result_sha256"] == adjudication[
        "result_sha256"
    ]
    assert audit["source_binding"]["registry_result_sha256"] == registry[  # type: ignore[index]
        "result_sha256"
    ]
