"""Build the hash-bound Round 72 implementation freeze."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from simple_ai_trading.price_discovery_spec import (
    build_round72_implementation_spec,
)
from simple_ai_trading.storage import write_json_atomic


DESIGN_DEFAULT = Path(
    "docs/model-research/action-value/round-072-spot-perpetual-price-discovery-design.json"
)
INVENTORY_DEFAULT = Path(
    "docs/model-research/action-value/round-072-spot-perpetual-inventory.json"
)
OUTPUT_DEFAULT = Path(
    "docs/model-research/action-value/round-072-price-discovery-implementation.json"
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _verified_hash(path: Path, field: str) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    canonical = dict(value)
    observed = str(canonical.pop(field, ""))
    if observed != _canonical_sha256(canonical):
        raise ValueError(f"{path} canonical hash differs")
    return observed


def run(arguments: argparse.Namespace) -> dict[str, object]:
    design = Path(arguments.design).resolve()
    inventory = Path(arguments.inventory).resolve()
    output = Path(arguments.output).resolve()
    if len({design, inventory, output}) != 3:
        raise ValueError("Round 72 specification paths must be distinct")
    artifact = build_round72_implementation_spec(
        design_sha256=_verified_hash(design, "design_sha256"),
        inventory_sha256=_verified_hash(inventory, "inventory_sha256"),
        inventory_file_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
        frozen_at_utc=str(arguments.frozen_at_utc),
    )
    write_json_atomic(output, artifact, indent=2, sort_keys=True)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default=str(DESIGN_DEFAULT))
    parser.add_argument("--inventory", default=str(INVENTORY_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--frozen-at-utc", default=datetime.now(UTC).isoformat())
    return parser


def main() -> int:
    artifact = run(build_parser().parse_args())
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
