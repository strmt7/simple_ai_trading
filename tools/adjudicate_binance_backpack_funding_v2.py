"""Run the sole source-proved timestamp correction for the frozen funding test."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import tools.adjudicate_binance_backpack_funding as v1


V1_IMPLEMENTATION_SHA256 = (
    "d20cad18fb7a21409921e445273db4094094adb49c6c3acf8a32a4e0267ff623"
)
NAIVE_UTC_HOUR = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:00:00$")


def _timestamp_ms(value: Any) -> int:
    """Parse v1 formats plus Backpack's source-proved naive UTC hour label."""
    text = str(value)
    if text.isdigit() or text.endswith("Z") or "+" in text[10:]:
        return v1._timestamp_ms(value)
    if not NAIVE_UTC_HOUR.fullmatch(text):
        raise RuntimeError("Backpack timestamp is not a strict UTC whole-hour label")
    parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    return int(parsed.timestamp() * 1000)


def main() -> None:
    dependency_path = Path(v1.__file__).resolve()
    if v1._sha256(dependency_path.read_bytes()) != V1_IMPLEMENTATION_SHA256:
        raise RuntimeError("v1 implementation binding changed")
    original = v1._timestamp_ms
    v1._timestamp_ms = _timestamp_ms
    try:
        v1.main()
    finally:
        v1._timestamp_ms = original


if __name__ == "__main__":
    main()
