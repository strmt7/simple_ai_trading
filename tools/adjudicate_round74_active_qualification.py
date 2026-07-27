"""Publish the deterministic Round 74 active-result adjudication."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from simple_ai_trading.round74_active_adjudication import (  # noqa: E402
    write_round74_active_adjudication,
)


def main() -> int:
    path = write_round74_active_adjudication(REPOSITORY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "path": str(path.relative_to(REPOSITORY)).replace("\\", "/"),
                "artifact_sha256": payload["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
