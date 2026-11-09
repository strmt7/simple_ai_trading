"""Validate an existing readback receipt before separately retiring its source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat

from tools import migrate_research_storage as migration


def verify_migration(plan_path: Path) -> dict[str, object]:
    """Check current identities and exact populations without another bulk read."""
    plan = migration.checked_plan(plan_path)
    source, destination, records = (
        Path(plan[key]) for key in ("source", "destination", "records")
    )
    if records.resolve() != records or plan_path.resolve().parent != records:
        raise ValueError("Receipt directory is not bound to the supplied plan")
    receipt_path = records / "copy-verified.json"
    if receipt_path.resolve() != receipt_path:
        raise ValueError("Readback receipt contains an alias")
    receipt = json.loads(receipt_path.read_bytes())
    if migration.canonical(receipt) != receipt["sha256"]:
        raise ValueError("Readback receipt digest mismatch")
    if (
        receipt["plan_sha256"] != plan["sha256"]
        or receipt["source"] != str(source)
        or receipt["destination"] != str(destination)
        or receipt["source_untouched"] is not True
        or receipt["content_analysis"] is not False
    ):
        raise ValueError("Readback receipt does not bind this opaque relocation")
    implementation = hashlib.sha256(Path(migration.__file__).read_bytes()).hexdigest()
    if receipt["applied_implementation_sha256"] != implementation:
        raise ValueError("Executed copy implementation requires separate review")
    root = source.stat()
    if (root.st_ino, root.st_dev) != (plan["root_inode"], plan["root_device"]):
        raise ValueError("Source root identity changed")
    if migration.inventory(source) != plan["entries"]:
        raise ValueError("Source population or metadata changed after planning")
    retained = {
        row["relative"]: row
        for row in plan["entries"]
        if row["decision"].startswith("retain")
    }
    discarded = [
        row for row in plan["entries"] if not row["decision"].startswith("retain")
    ]
    copies = receipt["retained_files"]
    names = [row["relative"] for row in copies]
    if len(names) != len(set(names)) or set(names) != set(retained):
        raise ValueError("Readback population is incomplete or duplicated")
    if receipt["discarded_files"] != discarded or receipt["retained_bytes"] != sum(
        row["size"] for row in retained.values()
    ):
        raise ValueError("Readback retention decisions or byte totals changed")
    owner = destination / ".migration-owner.json"
    if owner.resolve() != owner or json.loads(owner.read_bytes()) != {
        "plan_sha256": plan["sha256"]
    }:
        raise ValueError("Destination ownership changed")
    actual = migration.inventory(destination)
    if {row["relative"] for row in actual} != set(retained) | {owner.name}:
        raise ValueError("Destination contains missing, extra or partial files")
    for copy in copies:
        relative = copy["relative"]
        target = destination / relative
        if target.resolve() != target:
            raise ValueError("Destination contains an alias")
        current = target.stat()
        expected = retained[relative]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", copy["content_sha256"])
            or copy["size"] != expected["size"]
            or current.st_size != expected["size"]
            or current.st_ino != copy["destination_inode"]
            or current.st_mtime_ns != copy["destination_mtime_ns"]
            or current.st_mtime_ns != expected["mtime_ns"]
            or current.st_mode & stat.S_IWRITE
        ):
            raise ValueError("Verified destination identity or immutability changed")
    return {
        "plan_sha256": plan["sha256"],
        "readback_receipt_sha256": receipt["sha256"],
        "retained_files": len(retained),
        "retained_bytes": receipt["retained_bytes"],
        "discarded_files": len(discarded),
        "metadata_verified": True,
        "bulk_hashes_reused_from_readback": True,
        "source_deleted": False,
    }


def main() -> None:
    """Perform no deletion; authorization and exact-root removal stay separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_migration(args.plan), sort_keys=True))


if __name__ == "__main__":
    main()
