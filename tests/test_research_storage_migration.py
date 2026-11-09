from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import migrate_research_storage as module


def setup_plan(tmp_path):
    source, destination, records = (tmp_path / p for p in ("hdd", "ssd", "receipts"))
    source.mkdir()
    (source / "unique-input.bin").write_bytes(b"opaque historical input")
    (source / "retained-empty.claim").touch()
    (source / "obsolete.pid").write_bytes(b"1234")
    (source / "empty.stderr.log").touch()
    cache = source / "model-cache/huggingface/models--publisher--model/blobs"
    cache.mkdir(parents=True)
    (cache / ("a" * 64)).write_bytes(b"reproducible downloaded model")
    return source, destination, module.create_plan(source, destination, records)


def test_retention_plan_distinguishes_evidence_claims_and_disposable_cache(tmp_path):
    _, _, path = setup_plan(tmp_path)
    plan = module.checked_plan(path)
    choices = {row["relative"]: row["decision"] for row in plan["entries"]}
    assert choices["unique-input.bin"].startswith("retain")
    assert choices["retained-empty.claim"].startswith("retain")
    assert choices["obsolete.pid"] == "discard_obsolete_process_marker"
    assert choices["empty.stderr.log"] == "discard_empty_diagnostic_log"
    assert (
        choices["model-cache/huggingface/models--publisher--model/blobs/" + "a" * 64]
        == "discard_reproducible_download_cache"
    )
    assert (
        not plan["analysis_permitted"] and not plan["source_deletion_permitted_by_copy"]
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows source write/delete sharing")
def test_copy_is_byte_exact_preserves_source_and_requires_readback(tmp_path):
    source, destination, path = setup_plan(tmp_path)
    result = module.apply_plan(path)
    assert not result["source_deleted"]
    assert (source / "unique-input.bin").exists()
    assert (destination / "unique-input.bin").read_bytes() == b"opaque historical input"
    assert (destination / "retained-empty.claim").is_file()
    assert not (destination / "obsolete.pid").exists()
    receipt = json.loads(Path(result["copy_verified"]).read_bytes())
    assert receipt["sha256"] == module.canonical(receipt)
    assert len(receipt["retained_files"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        module.apply_plan(path)
    for p in destination.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


def test_changed_source_rejects_before_destination_creation(tmp_path):
    source, destination, path = setup_plan(tmp_path)
    (source / "unique-input.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        module.apply_plan(path)
    assert not destination.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows locked-source resume")
def test_completed_copy_resume_does_not_budget_already_allocated_bytes(
    tmp_path, monkeypatch
):
    _, destination, plan = setup_plan(tmp_path)
    result = module.apply_plan(plan)
    # Simulate interruption after completed copies, before receipt publication.
    Path(result["copy_verified"]).unlink()
    monkeypatch.setattr(
        module.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 2**30)
    )
    try:
        assert module.apply_plan(plan)["retained_files"] == 2
    finally:
        for path in destination.rglob("*"):
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IWRITE)


def test_tampered_plan_cannot_redirect_copy(tmp_path):
    _, destination, path = setup_plan(tmp_path)
    plan = json.loads(path.read_bytes())
    plan["destination"] = str(tmp_path / "foreign")
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="digest"):
        module.apply_plan(path)
    assert not destination.exists()


@pytest.mark.parametrize(
    "overlap", ["inside_source", "same_root", "receipts_inside_source"]
)
def test_overlapping_roots_reject_before_receipt_creation(tmp_path, overlap):
    source = tmp_path / "hdd"
    source.mkdir()
    destination, records = tmp_path / "ssd", tmp_path / "receipts"
    if overlap == "inside_source":
        destination = source / "nested"
    elif overlap == "same_root":
        destination = source
    else:
        records = source / "receipts"
    with pytest.raises(ValueError):
        module.create_plan(source, destination, records)
    assert not records.exists()


@pytest.mark.skipif(
    os.name != "nt", reason="Windows read-only NTFS hardlink verification"
)
def test_identical_bytes_and_timestamps_deduplicate_without_losing_evidence_names(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("first.bin", "second.bin"):
        path = source / name
        path.write_bytes(b"same immutable historical evidence")
        os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    destination = tmp_path / "destination"
    plan = module.create_plan(source, destination, tmp_path / "records")
    result = module.apply_plan(plan)
    receipt = json.loads(Path(result["copy_verified"]).read_bytes())
    assert receipt["duplicate_files_hardlinked"] == 1
    assert (destination / "first.bin").samefile(destination / "second.bin")
    assert (source / "first.bin").read_bytes() == (
        destination / "first.bin"
    ).read_bytes()
    for path in destination.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


@pytest.mark.skipif(os.name != "nt", reason="Windows locked-source resume")
@pytest.mark.parametrize("prefix", [b"", b"opaque", b"wrong"])
def test_interrupted_file_prefix_is_verified_before_append(tmp_path, prefix):
    source, destination, path = setup_plan(tmp_path)
    plan = module.checked_plan(path)
    destination.mkdir()
    (destination / ".migration-owner.json").write_text(
        json.dumps({"plan_sha256": plan["sha256"]})
    )
    partial = destination / "unique-input.bin.migration-partial"
    partial.write_bytes(prefix)
    if prefix == b"wrong":
        with pytest.raises(ValueError, match="prefix differs"):
            module.apply_plan(path)
        assert partial.read_bytes() == prefix
        assert not (destination / "unique-input.bin").exists()
    else:
        result = module.apply_plan(path)
        assert not partial.exists()
        assert (destination / "unique-input.bin").read_bytes() == (
            source / "unique-input.bin"
        ).read_bytes()
        assert result["retained_files"] == 2
    for retained in destination.rglob("*"):
        if retained.is_file():
            os.chmod(retained, stat.S_IWRITE | stat.S_IREAD)
