from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print canonical SHA-256 for a JSON object without one self-hash field."
    )
    parser.add_argument("path")
    parser.add_argument("--field", required=True)
    args = parser.parse_args()
    path = (ROOT / args.path).resolve()
    path.relative_to(ROOT)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or args.field not in payload:
        raise RuntimeError(f"top-level field is missing: {args.field}")
    payload.pop(args.field)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    print(hashlib.sha256(canonical).hexdigest())


if __name__ == "__main__":
    main()
