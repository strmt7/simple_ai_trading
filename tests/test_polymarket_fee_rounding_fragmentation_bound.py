from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/action-value/"
    "polymarket-fee-rounding-fragmentation-dominance-bound-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_fee_rounding_cannot_zero_a_retained_minimum_order() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]

    sources = result["retained_book_population"]["sources"]
    row_count = 0
    condition_ids: set[str] = set()
    path_condition_ids: set[tuple[str, str]] = set()
    minimum_order_sizes: set[str] = set()
    tick_sizes: set[str] = set()
    for source in sources:
        path = ROOT / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
        books = json.loads(path.read_text(encoding="utf-8"))
        row_count += len(books)
        for book in books:
            condition_ids.add(book["market"])
            path_condition_ids.add((source["path"], book["market"]))
            minimum_order_sizes.add(book["min_order_size"])
            tick_sizes.add(book["tick_size"])

    population = result["retained_book_population"]
    assert len(sources) == population["book_file_count"] == 15
    assert row_count == population["book_response_row_count"] == 180
    assert len(condition_ids) == population["distinct_condition_count"] == 96
    assert (
        len(path_condition_ids) == population["distinct_path_condition_count"] == 98
    )
    assert minimum_order_sizes == set(population["observed_minimum_order_sizes"]) == {
        "5"
    }
    assert tick_sizes == set(population["observed_tick_sizes"]) == {"0.001", "0.01"}

    fee_source = ROOT / result["source"]["fee_source_path"]
    assert hashlib.sha256(fee_source.read_bytes()).hexdigest() == result["source"][
        "fee_source_sha256"
    ]
    assert Decimal(result["dominance_bound"]["minimum_observed_whole_order_fee_pUSD"]) == (
        Decimal("5") * Decimal("0.04") * Decimal("0.001") * Decimal("0.999")
    )
    assert Decimal(result["dominance_bound"]["minimum_observed_whole_order_fee_quantums"]) == (
        Decimal("19.98")
    )
    assert (
        result["dominance_bound"][
            "minimum_zeroed_fee_assessments_for_one_pUSD_gross_saving"
        ]
        == 100001
    )
    assert result["adjudication"]["accepted_edge"] is False
    assert result["authority"]["network_requests"] == 0


def test_fee_rounding_terminal_is_routed_without_global_registry_pins() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_22 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 22
    )
    assert any(
        row["result_sha256"] == result["result_sha256"]
        for row in rank_22["canonical_artifacts"]
    )
    assert any(
        row["canonical_result_sha256"] == result["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )
