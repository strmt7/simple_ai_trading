from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-aca-house-identity-metadata-contract-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/polymarket-aca-house-identity-metadata-v1"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256(encoded)


def _journal(payload: dict[str, Any]) -> None:
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _capture(*, name: str, url: str, raw_path: Path) -> None:
    requested_at_ms = time.time_ns() // 1_000_000
    intent = {
        "method": "GET",
        "name": name,
        "phase": "intent",
        "request_body_sha256": _sha256(b""),
        "requested_at_ms": requested_at_ms,
        "url": url,
    }
    _journal(intent)
    request = Request(
        url,
        method="GET",
        headers={"User-Agent": "simple-ai-trading-public-research/1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_bytes = response.read()
            status_code = response.status
    except HTTPError as exc:
        response_bytes = exc.read()
        status_code = exc.code
        raw_path.write_bytes(response_bytes)
        _journal(
            {
                **intent,
                "completed_at_ms": time.time_ns() // 1_000_000,
                "phase": "completed",
                "raw_path": raw_path.relative_to(ROOT).as_posix(),
                "response_bytes": len(response_bytes),
                "response_sha256": _sha256(response_bytes),
                "status_code": status_code,
            }
        )
        raise
    raw_path.write_bytes(response_bytes)
    _journal(
        {
            **intent,
            "completed_at_ms": time.time_ns() // 1_000_000,
            "phase": "completed",
            "raw_path": raw_path.relative_to(ROOT).as_posix(),
            "response_bytes": len(response_bytes),
            "response_sha256": _sha256(response_bytes),
            "status_code": status_code,
        }
    )
    if status_code != 200:
        raise RuntimeError(f"unexpected HTTP status {status_code}")


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if DATA_ROOT.exists():
        raise RuntimeError("one-use output already exists")
    (DATA_ROOT / "raw").mkdir(parents=True)

    for capture in contract["capture"]["requests"]:
        _capture(
            name=capture["name"],
            url=capture["url"],
            raw_path=ROOT / capture["raw_path"],
        )
    print(json.dumps({"completed_requests": 3, "payloads_printed": 0}))


if __name__ == "__main__":
    main()
