from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _load_self_hashed(name: str, field: str) -> dict[str, object]:
    value = json.loads((RESEARCH / name).read_text(encoding="ascii"))
    claimed = value.pop(field)
    assert claimed == _canonical_sha256(value)
    value[field] = claimed
    return value


def test_round25_ai_scenario_contract_lineage_is_self_hashed() -> None:
    v1 = _load_self_hashed(
        "round-025-ai-risk-scenario-contract-v1.json",
        "contract_sha256",
    )
    v2 = _load_self_hashed(
        "round-025-ai-risk-scenario-contract-v2.json",
        "contract_sha256",
    )
    v3 = _load_self_hashed(
        "round-025-ai-risk-scenario-contract-v3.json",
        "contract_sha256",
    )
    correction = _load_self_hashed(
        "round-025-ai-risk-scenario-correction-contract-v1.json",
        "contract_sha256",
    )

    assert v2["supersedes"]["scenario_contract_v1_sha256"] == v1["contract_sha256"]
    assert v3["supersedes"]["scenario_contract_v2_sha256"] == v2["contract_sha256"]
    assert len(v3["scenarios"]) == 11
    assert v3["fixture_policy"]["all_transport_gap_counts_zero"] is True
    assert correction["correction"]["model_inference_may_be_repeated"] is False


def test_round25_ai_v3_correction_reclassifies_exact_results_without_inference() -> None:
    raw = _load_self_hashed(
        "round-025-ai-risk-scenario-host-probe-v3-2026-08-10.json",
        "evidence_sha256",
    )
    corrected = _load_self_hashed(
        "round-025-ai-risk-scenario-host-probe-v3-correction-2026-08-10.json",
        "evidence_sha256",
    )

    assert raw["status"] == "safety_behavior_mechanics_failed"
    assert len(raw["scenario_results"]) == 11
    assert all(value["advisory"]["valid_model_response"] for value in raw["scenario_results"])
    assert not any(raw["claims"].values())
    assert corrected["status"] == "safety_behavior_mechanics_verified"
    assert all(corrected["checks"].values())
    assert corrected["model_inference_repeated"] is False
    assert corrected["source_probe_evidence_sha256"] == raw["evidence_sha256"]
    assert not any(corrected["claims"].values())

    by_id = {value["scenario_id"]: value for value in raw["scenario_results"]}
    assert {value["action"] for value in corrected["actions"]} == {"allow", "veto"}
    for identity in corrected["scenario_result_identities"]:
        source = by_id[identity["scenario_id"]]
        assert identity["packet_sha256"] == source["packet_sha256"]
        assert identity["advisory_sha256"] == source["advisory"]["advisory_sha256"]
        assert identity["telemetry_sha256"] == _canonical_sha256(source["telemetry"])
