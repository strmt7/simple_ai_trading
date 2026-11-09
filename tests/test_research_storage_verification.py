from __future__ import annotations

import json
import os
import stat

import pytest

from tools import migrate_research_storage as migration
from tools.verify_research_storage_migration import verify_migration


@pytest.fixture
def relocated(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows source sharing and immutable NTFS target")
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    (source / "evidence.bin").write_bytes(b"preserved historical bytes")
    (source / "obsolete.pid").write_bytes(b"999999")
    plan = migration.create_plan(source, destination, tmp_path / "records")
    migration.apply_plan(plan)
    try:
        yield source, destination, plan
    finally:
        for path in destination.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_verified_population_reuses_readback_without_deleting_source(relocated):
    source, destination, plan = relocated
    result = verify_migration(plan)
    assert result["retained_files"] == result["discarded_files"] == 1
    assert result["metadata_verified"] and result["bulk_hashes_reused_from_readback"]
    assert not result["source_deleted"]
    assert (source / "evidence.bin").read_bytes() == (
        destination / "evidence.bin"
    ).read_bytes()


@pytest.mark.parametrize(
    "change", ["new_source", "changed_source", "extra_target", "writable_target"]
)
def test_drift_blocks_source_retirement(relocated, change):
    source, destination, plan = relocated
    if change == "new_source":
        (source / "new-evidence.bin").touch()
    elif change == "changed_source":
        (source / "evidence.bin").write_bytes(b"new data")
    elif change == "extra_target":
        (destination / "unexpected.migration-partial").touch()
    else:
        (destination / "evidence.bin").chmod(stat.S_IREAD | stat.S_IWRITE)
    with pytest.raises(ValueError):
        verify_migration(plan)
    assert source.is_dir()


@pytest.mark.parametrize(
    "change", ["digest", "duplicate", "discarded", "implementation"]
)
def test_invalid_readback_receipt_blocks_source_retirement(relocated, change):
    source, _, plan = relocated
    receipt_path = plan.parent / "copy-verified.json"
    receipt = json.loads(receipt_path.read_bytes())
    if change == "duplicate":
        receipt["retained_files"] *= 2
    elif change == "discarded":
        receipt["discarded_files"] = []
    elif change == "implementation":
        receipt["applied_implementation_sha256"] = "0" * 64
    else:
        receipt["retained_bytes"] = 0
    if change != "digest":
        receipt["sha256"] = migration.canonical(receipt)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_migration(plan)
    assert source.is_dir()
