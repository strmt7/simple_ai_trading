from __future__ import annotations

from datetime import datetime
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


def test_uae_aggregate_creation_gap_is_terminal_and_source_bound() -> None:
    adjudication = _load(
        "docs/model-research/action-value/"
        "polymarket-iran-uae-to-any-arab-sep15-source-adjudication-"
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
    alignment = adjudication["rule_alignment"]  # type: ignore[assignment]
    uae_start = datetime.fromisoformat(
        str(alignment["uae_condition_start_utc"]).replace("Z", "+00:00")
    )
    aggregate_start = datetime.fromisoformat(
        str(alignment["aggregate_condition_start_utc"]).replace("Z", "+00:00")
    )
    assert int((aggregate_start - uae_start).total_seconds()) == 1_108_111
    assert alignment["uae_implies_aggregate_source_proved"] is False
    assert adjudication["payoff_counterexample"][  # type: ignore[index]
        "package_payout_pUSD_per_share"
    ] == "0"
    assert adjudication["authority"][  # type: ignore[index]
        "bestBid_or_bestAsk_fields_accessed_for_decision"
    ] is False

    cases = {
        "aggregate": "polymarket-iran-any-arab-sep15",
        "uae": "polymarket-iran-uae-sep15",
    }
    for source_name, file_stem in cases.items():
        contract = _load(
            f"docs/model-research/action-value/{file_stem}-metadata-contract-"
            "v1-2026-09-01.json"
        )
        capture = _load(
            f"docs/model-research/action-value/{file_stem}-metadata-capture-result-"
            "v1-2026-09-01.json"
        )
        assert _canonical_hash(contract, "contract_sha256") == contract[
            "contract_sha256"
        ]
        assert _canonical_hash(capture, "result_sha256") == capture["result_sha256"]
        source = adjudication["source_binding"][source_name]  # type: ignore[index]
        assert source["contract_sha256"] == contract["contract_sha256"]
        assert source["capture_result_sha256"] == capture["result_sha256"]
        assert source["raw_sha256"] == _file_hash(source["raw_path"])

    family = (
        "polymarket_Iran_targets_UAE_by_September_15_to_any_Arab_country_"
        "subset_aggregate_implication_2026_09_01"
    )
    terminal = {
        row["family"]: row
        for row in registry["terminal_do_not_repeat"]  # type: ignore[index]
    }
    assert len(registry["terminal_do_not_repeat"]) == 142  # type: ignore[arg-type]
    assert terminal[family]["canonical_result_sha256"] == adjudication[
        "result_sha256"
    ]
    assert audit["source_binding"]["registry_result_sha256"] == registry[  # type: ignore[index]
        "result_sha256"
    ]
