"""Reuse frozen frontier economics with a bounded, journaled GET transport."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from tools import screen_polymarket_exact_negrisk_long_only_frontier as legacy

BYTE_CEILING = 2_097_152
SOCKET_TIMEOUT_SECONDS = 30


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _journal(path: Path, entry: dict) -> None:
    mode = "x" if entry["phase"] == "intent" else "a"
    with path.open(mode, encoding="ascii", newline="\n") as stream:
        stream.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def bounded_request(
    *,
    method: str,
    url: str,
    body: bytes,
    name: str,
    raw_path: Path,
    raw_relative_path: str,
    journal_path: Path,
) -> tuple[bytes, dict]:
    if (
        method != "GET"
        or body
        or not url.startswith("https://gamma-api.polymarket.com/events/slug/")
    ):
        raise ValueError("only the frozen exact public Gamma event GET is supported")
    if raw_path.exists() or journal_path.exists():
        raise RuntimeError("one-use transport output already exists")
    intent = {
        "phase": "intent",
        "method": method,
        "url": url,
        "name": name,
        "request_body_sha256": hashlib.sha256(body).hexdigest(),
        "requested_at_ms": time.time_ns() // 1_000_000,
        "response_byte_ceiling": BYTE_CEILING,
        "redirects_allowed": False,
    }
    _journal(journal_path, intent)
    status = None
    error = None
    total = 0
    started = time.monotonic()
    with raw_path.open("xb") as output:
        try:
            request = Request(
                url,
                method="GET",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "User-Agent": "simple-ai-trading-public-research/2",
                },
            )
            try:
                response = build_opener(NoRedirect()).open(
                    request, timeout=SOCKET_TIMEOUT_SECONDS
                )
            except HTTPError as exc:
                response = exc
            with response:
                status = response.code
                while total <= BYTE_CEILING:
                    if time.monotonic() - started > SOCKET_TIMEOUT_SECONDS:
                        raise TimeoutError("elapsed read budget exhausted")
                    chunk = response.read(min(65_536, BYTE_CEILING + 1 - total))
                    if not chunk:
                        break
                    output.write(chunk)
                    total += len(chunk)
        except Exception as exc:
            error = type(exc).__name__
        finally:
            output.flush()
            os.fsync(output.fileno())
    raw = raw_path.read_bytes()
    receipt = {
        **intent,
        "phase": "completed",
        "completed_at_ms": time.time_ns() // 1_000_000,
        "raw_path": raw_relative_path,
        "response_bytes": len(raw),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "status_code": status,
        "transport_error_type": error,
        "within_byte_ceiling": total <= BYTE_CEILING,
    }
    _journal(journal_path, receipt)
    if error or status != 200 or total > BYTE_CEILING:
        raise RuntimeError(
            "public source transport failed; retained partial/error bytes; no retry"
        )
    return raw, receipt


def main() -> None:
    # A compatibility boundary keeps all old frozen calculations byte-identical.
    # The capture contract must bind both this bridge and every legacy dependency.
    original = legacy._request
    try:
        legacy._request = bounded_request
        legacy.main()
    finally:
        legacy._request = original


if __name__ == "__main__":
    main()
