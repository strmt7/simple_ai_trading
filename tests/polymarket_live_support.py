from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from simple_ai_trading.polymarket_live_manifest import (
    build_polymarket_live_implementation_manifest,
)


@lru_cache(maxsize=8)
def _implementation_manifest_bytes(source_commit: str) -> bytes:
    payload = build_polymarket_live_implementation_manifest(
        source_commit=source_commit,
    )
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def write_polymarket_live_implementation_manifest(
    path: Path,
    *,
    source_commit: str,
) -> None:
    path.write_bytes(_implementation_manifest_bytes(source_commit))


__all__ = ["write_polymarket_live_implementation_manifest"]
