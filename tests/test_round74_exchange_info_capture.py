from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from simple_ai_trading.impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetEvidence,
    round74_quantity_rules_evidence_claims,
)
from simple_ai_trading.impact_absorption_targets import (
    Round73MarketQuantityRules,
)
from tools.capture_round74_exchange_info_evidence import (
    ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION,
    ROUND74_EXCHANGE_INFO_URL,
    _canonical_sha256,
)


REPOSITORY = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    REPOSITORY / "docs" / "model-research" / "action-value"
)


def _git_blob_sha256(commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


@pytest.mark.parametrize(
    "artifact_name",
    (
        "round-074-exchange-info-evidence-2026-07-27.json",
        "round-074-exchange-info-evidence-2026-07-27-v2.json",
    ),
)
def test_round74_exchange_info_capture_is_real_hash_bound_evidence(
    artifact_name: str,
) -> None:
    artifact = json.loads(
        (ARTIFACT_DIRECTORY / artifact_name).read_text(encoding="utf-8")
    )
    claimed = artifact.pop("artifact_sha256")
    assert claimed == _canonical_sha256(artifact)
    assert (
        artifact["schema_version"]
        == ROUND74_EXCHANGE_INFO_CAPTURE_SCHEMA_VERSION
    )
    assert len(artifact["execution_git_commit"]) == 40

    request = artifact["request"]
    response = artifact["response"]
    assert request["method"] == "GET"
    assert request["url"] == ROUND74_EXCHANGE_INFO_URL
    assert request["security_type"] == "NONE"
    assert request["request_weight"] == 1
    assert request["retry_count"] == 0
    assert request["credential_material_sent"] is False
    assert response["status"] == 200
    assert response["body_bytes"] > 1_000_000
    assert response["elapsed_monotonic_ns"] > 0
    assert response["headers"]["x_mbx_used_weight_1m"] == "1"
    assert response["raw_payload_persisted"] is False

    source = artifact["source_binding"]
    execution_commit = artifact["execution_git_commit"]
    assert source["parser_sha256"] == _git_blob_sha256(
        execution_commit,
        source["parser_path"],
    )
    assert source["capture_tool_sha256"] == _git_blob_sha256(
        execution_commit,
        source["capture_tool_path"]
    )
    claims = artifact["quantity_rules"]
    rule_payload = claims["market_quantity_rules_by_symbol"]
    assert tuple(sorted(rule_payload)) == ROUND74_EVENT_TARGET_SYMBOLS
    rules = {
        symbol: Round73MarketQuantityRules.create(
            symbol=symbol,
            step_size=value["step_size"],
            minimum_quantity=value["minimum_quantity"],
            maximum_quantity=value["maximum_quantity"],
            minimum_notional=value["minimum_notional"],
        )
        for symbol, value in rule_payload.items()
    }
    assert claims == round74_quantity_rules_evidence_claims(rules)
    evidence = Round74EventTargetEvidence.from_dict(
        artifact["target_evidence"]
    )
    assert evidence.binds(claims)
    assert evidence.environment == "binance_usdm_mainnet"
    assert evidence.record_count == 3

    assert all(value is False for value in artifact["authority"].values())
