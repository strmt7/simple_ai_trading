from __future__ import annotations

from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL_FILE = ROOT / "tools" / "screen_binance_zero_fee_stablecoin_cycles.py"
ARTIFACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-zero-fee-stablecoin-cycle-recovery-v2-2026-08-26.json"
)
CONTRACT_PATH = Path(
    "docs/model-research/action-value/"
    "binance-zero-fee-stablecoin-cycle-recovery-contract-v2.json"
)
FAILURE_PATH = Path(
    "docs/model-research/action-value/"
    "binance-zero-fee-stablecoin-cycle-v1-terminal-failure-2026-08-26.json"
)
TOOL_PATH = Path("tools/screen_binance_zero_fee_stablecoin_cycles.py")
EXPECTED_RESULT_HASH = (
    "f44f283b311ebf8b3302dba4e1d5d6be0b956a2657483786229611f82ed5da88"
)
EXPECTED_CONTRACT_HASH = (
    "5ed9d447ba1b45407ac68a2cde227df2a8b39fff1d6428677f4d7122fcc46cd8"
)
EXPECTED_TOOL_HASH = "4c44faaecc6aa874eab71ee0dff161d32adaa1af4ac9717d5b51d10fdc7944f7"
SPEC = importlib.util.spec_from_file_location("zero_fee_stablecoin_cycles", TOOL_FILE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def test_recovery_artifact_contract_tool_and_failure_are_source_bound() -> None:
    artifact = _load(ARTIFACT_PATH)
    body = dict(artifact)
    assert body.pop("result_sha256") == EXPECTED_RESULT_HASH
    assert hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest() == (
        EXPECTED_RESULT_HASH
    )
    assert artifact["implementation"] == {
        "path": TOOL_PATH.as_posix(),
        "sha256": EXPECTED_TOOL_HASH,
    }
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == EXPECTED_TOOL_HASH

    contract = _load(CONTRACT_PATH)
    contract_body = dict(contract)
    assert contract_body.pop("contract_sha256") == EXPECTED_CONTRACT_HASH
    assert (
        hashlib.sha256(_canonical_json(contract_body).encode("ascii")).hexdigest()
        == EXPECTED_CONTRACT_HASH
    )
    failure = _load(FAILURE_PATH)
    failure_body = dict(failure)
    declared = failure_body.pop("result_sha256")
    assert (
        declared
        == hashlib.sha256(_canonical_json(failure_body).encode("ascii")).hexdigest()
    )
    assert failure["terminal_state"]["book_samples_admitted"] == 0


def test_valid_capture_rejects_every_frozen_cycle_size() -> None:
    artifact = _load(ARTIFACT_PATH)
    assert artifact["capture"] == {
        "capture_valid": True,
        "elapsed_ms": 299729,
        "maximum_request_elapsed_ms": 708,
        "sample_count": 600,
    }
    assert artifact["raw_evidence"]["record_count"] == 1201
    assert artifact["verdict"]["empirical_candidate_count"] == 0
    assert artifact["verdict"]["accepted_edge"] is False
    assert len(artifact["summaries"]) == 12
    assert all(row["positive_sample_count"] == 0 for row in artifact["summaries"])
    assert max(
        Decimal(row["maximum_stressed_bips"]) for row in artifact["summaries"]
    ) == Decimal("-5.0008000000000000")


def test_lot_fallback_and_top_level_capacity_are_conservative() -> None:
    symbol = {
        "symbol": "TEST",
        "filters": [
            {"filterType": "MARKET_LOT_SIZE", "stepSize": "0.00000000"},
            {"filterType": "LOT_SIZE", "stepSize": "1.00000000"},
        ],
    }
    assert MODULE._market_step(symbol) == Decimal("1.00000000")
    books = {
        "A": {
            "bidPrice": Decimal("0.99"),
            "bidQty": Decimal("1000"),
            "askPrice": Decimal("1.00"),
            "askQty": Decimal("1000"),
        },
        "B": {
            "bidPrice": Decimal("0.99"),
            "bidQty": Decimal("1000"),
            "askPrice": Decimal("1.00"),
            "askQty": Decimal("1000"),
        },
        "C": {
            "bidPrice": Decimal("1.01"),
            "bidQty": Decimal("50"),
            "askPrice": Decimal("1.02"),
            "askQty": Decimal("1000"),
        },
    }
    evaluated = MODULE._evaluate_orientation(
        Decimal("100"),
        legs=(("A", "BUY"), ("B", "BUY"), ("C", "SELL")),
        books=books,
        steps={key: Decimal("1") for key in books},
        stress_bips=Decimal("3"),
    )
    assert evaluated["gross_bips"] == "100.00"
    assert evaluated["top_level_capacity_ok"] is False
    assert evaluated["positive_after_stress"] is False
