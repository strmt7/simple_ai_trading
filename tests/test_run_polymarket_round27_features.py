from __future__ import annotations

import hashlib
import json

import pytest

from tools.run_polymarket_round27_features import _load_audit


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _audit() -> dict[str, object]:
    conditions = [
        {
            "condition_id": "0x" + format(index + 1, "064x"),
            "eligible": True,
            "segments": [{"start_ms": 1, "end_ms": 2}],
        }
        for index in range(2)
    ]
    value = {
        "schema_version": "polymarket-condition-replay-audit-v1",
        "run_id": "round27-test",
        "conditions": conditions,
        "condition_count": 2,
        "eligible_condition_count": 2,
        "failed_condition_count": 0,
        "eligible_condition_ids": [item["condition_id"] for item in conditions],
        "failed_condition_ids": [],
        "target_free": True,
        "model_data_eligible": False,
    }
    value["audit_sha256"] = _sha256(value)
    return value


def test_round27_feature_tool_rehashes_diagnostic_subset(tmp_path) -> None:
    source = _audit()
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(source), encoding="ascii")

    derived = _load_audit(path, 1)
    claimed = derived.pop("audit_sha256")

    assert derived["source_audit_sha256"] == source["audit_sha256"]
    assert derived["diagnostic_scope"] == "eligible_condition_prefix"
    assert derived["eligible_condition_count"] == 1
    assert claimed == _sha256(derived)


def test_round27_feature_tool_rejects_tampered_audit(tmp_path) -> None:
    source = _audit()
    source["condition_count"] = 99
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(source), encoding="ascii")

    with pytest.raises(ValueError, match="audit hash differs"):
        _load_audit(path, 0)
