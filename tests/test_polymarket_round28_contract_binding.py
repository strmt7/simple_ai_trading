from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round27_model_contract import (
    load_round27_model_contract,
)
from simple_ai_trading.polymarket_round28_contract_binding import (
    load_round28_contract_binding_correction,
    validate_loaded_round27_model_contract,
    validate_round28_contract_binding_correction,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONTRACT = (
    ROOT
    / "docs/model-research/polymarket/round-027-stage1-model-contract-v1.json"
)
PREDECESSOR_CORRECTION = (
    ROOT
    / "docs/model-research/polymarket/"
    "round-028-loaded-contract-binding-correction-v1.json"
)


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


def test_round28_accepts_real_loaded_round27_contract() -> None:
    loaded = load_round27_model_contract(ROOT)

    assert validate_loaded_round27_model_contract(loaded) == loaded


def test_round28_rejects_missing_or_tampered_model_amendment_binding() -> None:
    raw = json.loads(BASE_CONTRACT.read_text(encoding="ascii"))
    with pytest.raises(ValueError, match="model amendment SHA-256 differs"):
        validate_loaded_round27_model_contract(raw)

    loaded = load_round27_model_contract(ROOT)
    loaded["model_implementation_amendment_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract binding differs"):
        validate_loaded_round27_model_contract(loaded)


def test_round28_contract_correction_is_canonical_and_source_bound() -> None:
    correction = load_round28_contract_binding_correction(ROOT)

    claimed = correction.pop("amendment_sha256")
    assert claimed == _canonical_sha256(correction)

    tampered = json.loads(json.dumps({**correction, "amendment_sha256": claimed}))
    first = next(iter(tampered["source_text_sha256"]))
    tampered["source_text_sha256"][first] = "0" * 64
    tampered["amendment_sha256"] = _canonical_sha256(
        {key: value for key, value in tampered.items() if key != "amendment_sha256"}
    )
    with pytest.raises(ValueError, match="corrected source binding differs"):
        validate_round28_contract_binding_correction(tampered, repository=ROOT)


def test_round28_predecessor_correction_remains_hash_and_source_bound() -> None:
    predecessor = json.loads(PREDECESSOR_CORRECTION.read_text(encoding="ascii"))
    current = load_round28_contract_binding_correction(ROOT)
    claimed = predecessor.pop("amendment_sha256")

    assert claimed == _canonical_sha256(predecessor)
    replacements = current["superseded_source_text_sha256"]
    for relative, expected in predecessor["source_text_sha256"].items():
        if relative in replacements:
            assert replacements[relative]["frozen"] == expected
        else:
            path = ROOT / relative
            actual = hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            ).hexdigest()
            assert actual == expected
