import hashlib
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-combo-collateral-release-overlay-source-contract-v1.json"
)
RESULT = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-combo-collateral-release-overlay-candidate-v1-2026-08-29.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(document: dict, field: str) -> str:
    payload = dict(document)
    expected = payload.pop(field)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
    return actual


def test_combo_collateral_release_source_is_exact_and_one_use() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    _canonical_hash(contract, "contract_sha256")
    _canonical_hash(result, "result_sha256")
    assert result["contract"]["contract_sha256"] == contract["contract_sha256"]

    source = result["source"]
    raw = ROOT / source["raw_path"]
    journal = ROOT / source["journal_path"]
    raw_bytes = raw.read_bytes()
    assert len(raw_bytes) == source["raw_bytes"] == 44910
    assert hashlib.sha256(raw_bytes).hexdigest() == source["raw_sha256"]
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == source["journal_sha256"]
    assert source["matches_prior_2026_08_27_payload_sha256"] is True

    entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert [entry["phase"] for entry in entries] == ["intent", "completed"]
    assert entries[0]["requested_before_utc"] == entries[1]["requested_before_utc"]
    assert entries[1]["status_code"] == 200
    assert entries[1]["response_sha256"] == source["raw_sha256"]
    assert source["network_requests_used"] == 1
    assert contract["authority"]["venue_market_data_requests_permitted"] == 0


def test_source_proves_release_and_residual_exposure_but_not_cost() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    source_text = (ROOT / result["source"]["raw_path"]).read_text(encoding="utf-8")
    gate = result["source_gate"]

    assert "returns it as pUSD" in source_text
    assert "preserving any unmatched exposure" in source_text
    assert "preserve the wallet's remaining economic exposure" in source_text
    assert "net_pusd_out" in source_text
    assert "required_pusd_input" in source_text
    assert "estimated_cost" in source_text
    assert "EOA accounts are not supported" in source_text

    assert gate["exact_exposure_preservation_proved"] is True
    assert gate["positive_pUSD_release_possible_proved"] is True
    assert gate["exact_owned_net_pusd_out_proved"] is False
    assert gate["exact_monetary_execution_cost_proved"] is False
    assert gate["fee_or_gasless_guarantee_proved"] is False
    assert gate["accepted_edge"] is False
    assert gate["deployment_ready"] is False


def test_opportunity_cost_sensitivity_never_counts_released_principal() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    gate = result["zero_network_economic_gate"]
    sensitivity = gate["rate_sensitivity_only"]
    rate = Decimal(sensitivity["annual_rate"])

    assert gate["public_forward_profit_floor_pUSD"] == "0"
    assert "do not add net_pusd_out itself to profit" in gate["principal_accounting"]
    for row in sensitivity["break_even_total_cost_per_1000_pUSD_released"]:
        expected = Decimal("1000") * rate * Decimal(row["usable_days"]) / Decimal(365)
        assert Decimal(row["pUSD"]) == expected


def test_registry_routes_candidate_without_promoting_or_adding_a_family() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    _canonical_hash(registry, "result_sha256")
    assert len(registry["prioritized_hypotheses"]) == 44

    family = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 33
    )
    assert result["result_sha256"] in {
        artifact["result_sha256"] for artifact in family["canonical_artifacts"]
    }
    assert "collateral_release" in family["retry_trigger"]
    assert "plan_request_only_authority" in family["retry_trigger"]
    assert any(
        "counting_released_pUSD_principal_as_profit" in shortcut
        for shortcut in family["prohibited_shortcuts"]
    )
