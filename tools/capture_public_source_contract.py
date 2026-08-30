"""Capture one frozen public GET source with durable request journaling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_two_leg_package import _request


SCHEMA = "public-source-capture-result-v1"
EMPTY_SHA256 = _sha256(b"")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _validate_contract(contract: dict[str, Any], contract_path: Path) -> None:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    frozen_text = contract.get("frozen_at_utc")
    if not isinstance(frozen_text, str) or not frozen_text.endswith("Z"):
        raise RuntimeError("frozen_at_utc must be an explicit UTC instant")
    frozen = datetime.fromisoformat(frozen_text.replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    request = contract.get("request")
    if not isinstance(request, dict) or request.get("method") != "GET":
        raise RuntimeError("only a frozen public GET is supported")
    url = str(request.get("url") or "")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or request.get("count") != 1
        or request.get("body_sha256") != EMPTY_SHA256
    ):
        raise RuntimeError("request boundary is invalid")
    ceiling = contract.get("response_byte_ceiling")
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling <= 0:
        raise RuntimeError("response byte ceiling must be positive")
    phrases = contract.get("required_utf8_phrases")
    if (
        not isinstance(phrases, list)
        or not phrases
        or any(not isinstance(value, str) or not value for value in phrases)
    ):
        raise RuntimeError("required UTF-8 phrases must be nonempty strings")
    if contract.get("authority") != {
        "account_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 1,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    _validate_contract(contract, contract_path)
    paths = {
        name: _root_path(path) for name, path in contract["outputs"].items()
    }
    for path in paths.values():
        if path.exists():
            raise RuntimeError(f"one-use output already exists: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
    raw, receipt = _request(
        method="GET",
        url=contract["request"]["url"],
        body=b"",
        name=contract["request_name"],
        raw_path=paths["raw_path"],
        raw_relative_path=contract["outputs"]["raw_path"],
        journal_path=paths["journal_path"],
    )
    text = raw.decode("utf-8")
    phrase_presence = {
        phrase: phrase in text for phrase in contract["required_utf8_phrases"]
    }
    byte_ceiling_passed = len(raw) <= contract["response_byte_ceiling"]
    source_gate_passed = byte_ceiling_passed and all(phrase_presence.values())
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": contract["contract_path"],
            "sha256": contract["contract_sha256"],
        },
        "capture": {
            "receipt": receipt,
            "response_byte_ceiling": contract["response_byte_ceiling"],
            "response_byte_ceiling_passed": byte_ceiling_passed,
        },
        "source_gate": {
            "required_phrase_presence": phrase_presence,
            "passed": source_gate_passed,
        },
        "adjudication": {
            "status": (
                "required_public_source_terms_present"
                if source_gate_passed
                else "public_source_gate_failed_closed"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/capture_public_source_contract.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    paths["result_path"].write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "response_bytes": len(raw),
                "response_sha256": receipt["response_sha256"],
                "source_gate_passed": source_gate_passed,
                "required_phrase_count": len(phrase_presence),
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
