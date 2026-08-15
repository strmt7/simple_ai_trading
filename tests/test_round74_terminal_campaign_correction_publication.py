from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs/model-research/action-value"
ARTIFACT = RESEARCH / "round-074-terminal-campaign-outcome-v2-2026-08-10.json"
PRIOR = RESEARCH / "round-074-terminal-campaign-outcome-2026-08-10.json"
PLAN = RESEARCH / "round-074-segmented-event-cohort-plan-v3.json"
FILE_SHA256 = "ab5a8d7815c8b6eab02c39043e1a4f053f9fe6630d92276297d6f6805b00d314"
ARTIFACT_SHA256 = "22187535286009efca9c55c6281eb8f858b33c71f05bed4a517768d9f2b1cde0"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _strings(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _strings(child)]
    return [value] if isinstance(value, str) else []


def test_corrected_terminal_outcome_preserves_and_supersedes_v1() -> None:
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == FILE_SHA256
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    claimed = value.pop("artifact_sha256")
    assert claimed == ARTIFACT_SHA256
    assert claimed == _canonical_sha256(value)
    assert value["schema_version"] == "round-074-terminal-campaign-outcome-v2"
    supersedes = value["supersedes"]
    assert supersedes["artifact_sha256"] == (
        "67cba6caf728d0ae2d0271e290d1dd7fe9ab4a7fa35f861683ebf16f080a2612"
    )
    assert (
        hashlib.sha256(PRIOR.read_bytes()).hexdigest()
        == (supersedes["artifact_file_sha256"])
    )
    assert value["decision"]["model_data_eligible"] is False
    assert value["decision"]["representative_training_performed"] is False
    assert value["decision"]["sealed_target_manifests_read"] is False
    assert value["decision"]["profitability_established"] is False
    assert not any(Path(text).is_absolute() for text in _strings(value))


def test_corrected_terminal_outcome_reconciles_the_bound_slot13_adjudication() -> None:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert value["corrections"] == [
        {
            "adjudication_file": "slot-013/recovery-adjudication.json",
            "adjudication_file_sha256": (
                "6dd92547c4b10a5846a7f8833f0577b1e3a6c97df5f760f4c136a2639281c6b9"
            ),
            "adjudication_sha256": (
                "34a9978b6d4465078cb01503108cd9db291e63fa7890414440aa82b9e3d6b369"
            ),
            "binding_sha256": (
                "6a1a8cf95250c9ad60f5db0746b54d2485f3be562bb77119f367917907e6152a"
            ),
            "corrected_reason_code": "admitted",
            "corrected_status": "admitted",
            "prior_reason_code": "host_slot_missed",
            "prior_status": "missed",
            "recovery_receipt_file": "013.json",
            "recovery_receipt_file_sha256": (
                "0693bb4bc3c2649a7c61fe7927f1660916760671e3569f3ccefb1d0ddb328223"
            ),
            "recovery_receipt_sha256": (
                "acff6ee95808c8493ea42d758bdfab56f6a8e8b4a3e00330fb35d47b17700bde"
            ),
            "role": "training",
            "slot_ordinal": 13,
        }
    ]
    coverage = value["terminal_coverage"]
    assert coverage["outcome_status_counts"] == {
        "admitted": 460,
        "missed": 238,
        "transport_excluded": 22,
    }
    assert coverage["partition_entry_count"] == 460
    assert coverage["partition_sha256"] == (
        "cd3274192bf6351306b910f62fc0df9eee8e90c17ec754102ee09dfd017e1f25"
    )
    training = coverage["role_quotas"]["training"]
    assert training["observed_eligible_anchor_ns"] == 233_306_813_789_700
    assert training["required_eligible_anchor_ns"] == 394_740_000_000_000
    assert training["deficit_eligible_anchor_ns"] == 161_433_186_210_300
    assert training["quota_passed"] is False
    sources = value["source_bindings"]
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == sources["plan_file_sha256"]
