"""Plan and verify an opaque research-data relocation; never evaluate datasets."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time

from tools.storage_io import IoBudget, background_io


CHUNK = 1024 * 1024


def canonical(value: dict) -> str:
    """Bind a plan or receipt independently of JSON formatting."""
    body = {k: v for k, v in value.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_new(path: Path, value: dict) -> None:
    """Create durable metadata without replacing an earlier migration receipt."""
    value = {**value, "sha256": canonical(value)}
    with path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        out.flush()
        os.fsync(out.fileno())


def checked_plan(path: Path) -> dict:
    """Require the exact admitted plan and explicit, nonoverlapping directory roots."""
    plan = json.loads(path.read_bytes())
    if canonical(plan) != plan["sha256"]:
        raise ValueError("Migration plan digest mismatch")
    source, destination = (Path(plan[k]) for k in ("source", "destination"))
    if not source.is_absolute() or not destination.is_absolute():
        raise ValueError("Migration roots must be absolute")
    if source.parent == source or destination.parent == destination:
        raise ValueError("A volume root cannot be a migration target")
    if source.resolve() != source or destination.resolve() != destination:
        raise ValueError("Migration roots contain aliases or reparse points")
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("Migration roots overlap")
    return plan


def inventory(root: Path) -> list[dict]:
    """Inventory names/identity/size only; do not open or interpret research content."""
    rows = []
    for current, dirs, files in os.walk(root, followlinks=False):
        directory = Path(current)
        for name in dirs:
            if getattr((directory / name).lstat(), "st_file_attributes", 0) & 0x400:
                raise ValueError("Directory reparse points require separate review")
        for name in files:
            path = directory / name
            item = path.lstat()
            relative = path.relative_to(root).as_posix()
            reparse = bool(getattr(item, "st_file_attributes", 0) & 0x400)
            decision = "retain_historical_input_or_evidence"
            if re.fullmatch(
                r"model-cache/huggingface/models--[^/]+/blobs/[0-9a-f]{64}", relative
            ):
                decision = "discard_reproducible_download_cache"
            elif reparse and relative.startswith("model-cache/huggingface/"):
                target = path.resolve()
                if not target.is_relative_to(root / "model-cache"):
                    raise ValueError("Cache link escapes the requested source folder")
                decision = "discard_download_cache_link"
            elif reparse:
                raise ValueError("Unreviewed file reparse point")
            elif path.suffix.lower() == ".pid":
                decision = "discard_obsolete_process_marker"
            elif path.suffix.lower() == ".log" and item.st_size == 0:
                decision = "discard_empty_diagnostic_log"
            rows.append(
                {
                    "relative": relative,
                    "size": item.st_size,
                    "mtime_ns": item.st_mtime_ns,
                    "inode": item.st_ino,
                    "device": item.st_dev,
                    "decision": decision,
                    "reparse": reparse,
                }
            )
    return sorted(rows, key=lambda row: row["relative"])


def create_plan(source: Path, destination: Path, records: Path) -> Path:
    """Freeze selective retention before any copy or deletion."""
    source, destination, records = (
        source.absolute(),
        destination.absolute(),
        records.absolute(),
    )
    if not source.is_dir() or destination.exists() or records.exists():
        raise ValueError(
            "Require an existing source and fresh destination/receipt directory"
        )
    roots = (source, destination, records)
    if any(p.resolve() != p or p.parent == p for p in roots) or any(
        a == b or a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError("Migration/receipt roots alias, overlap or name a volume root")
    records.mkdir(parents=True)
    root = source.stat()
    plan = {
        "source": str(source),
        "destination": str(destination),
        "records": str(records),
        "root_inode": root.st_ino,
        "root_device": root.st_dev,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "entries": inventory(source),
        "analysis_permitted": False,
        "source_deletion_permitted_by_copy": False,
    }
    path = records / "plan.json"
    write_new(path, plan)
    checked_plan(path)
    return path


def digest_file(path: Path, budget: IoBudget | None = None) -> str:
    """Read only opaque bytes for relocation integrity, without parsing content."""
    result = hashlib.sha256()
    read, last = 0, time.monotonic()
    with path.open("rb", buffering=0) as source:
        while block := source.read(CHUNK):
            if budget is not None:
                budget.consume(len(block))
            result.update(block)
            read += len(block)
            if budget is not None and time.monotonic() - last >= 15:
                print(
                    json.dumps(
                        {
                            "phase": "ssd_readback_in_progress",
                            "file_read_gib": round(read / 2**30, 2),
                            "file_total_gib": round(path.stat().st_size / 2**30, 2),
                            "checksum_complete": False,
                        }
                    ),
                    flush=True,
                )
                last = time.monotonic()
    return result.hexdigest()


def locked_reader(path: Path):
    """Deny writers during relocation, allowing readers and the later explicit deletion."""
    import msvcrt

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    api.CreateFileW.restype = wintypes.HANDLE
    # Include DELETE access to detect incompatible existing readers before copying.
    handle = api.CreateFileW(str(path), 0x80010000, 0x5, None, 3, 0x08200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle(handle)
        raise
    return os.fdopen(fd, "rb", buffering=0)


def apply_plan(plan_path: Path, *, io_mib_per_second: float = 64) -> dict:
    """Copy once, verify every retained file, and leave all source files intact."""
    plan = checked_plan(plan_path)
    budget = IoBudget(io_mib_per_second)
    source, destination, records = (
        Path(plan[k]) for k in ("source", "destination", "records")
    )
    ready = records / "copy-verified.json"
    if ready.exists():
        raise ValueError(
            "Verified migration already exists; do not overwrite its receipt"
        )
    root = source.stat()
    if (root.st_ino, root.st_dev) != (
        plan["root_inode"],
        plan["root_device"],
    ) or inventory(source) != plan["entries"]:
        raise ValueError("Source changed since the retention plan")
    required = sum(
        row["size"] for row in plan["entries"] if row["decision"].startswith("retain")
    )
    owner = destination / ".migration-owner.json"
    pending = required
    if destination.exists():
        if not owner.is_file() or json.loads(owner.read_bytes()) != {
            "plan_sha256": plan["sha256"]
        }:
            raise ValueError("Existing destination is not owned by this migration")
        pending = 0
        for row in plan["entries"]:
            if not row["decision"].startswith("retain"):
                continue
            target = destination / row["relative"]
            partial = target.with_name(target.name + ".migration-partial")
            if target.resolve() != target or partial.resolve() != partial:
                raise ValueError("Resume output contains an alias/reparse point")
            if target.exists():
                continue
            existing = partial.stat().st_size if partial.exists() else 0
            if existing > row["size"]:
                raise ValueError("Partial output exceeds planned source size")
            pending += row["size"] - existing
    if shutil.disk_usage(destination.parent).free < pending + 32 * 2**30:
        raise ValueError("Insufficient SSD capacity plus 32-GiB reserve")
    if not destination.exists():
        destination.mkdir(parents=True)
        with owner.open("x", encoding="ascii") as out:
            out.write(json.dumps({"plan_sha256": plan["sha256"]}))
    completed, written = [], 0
    last = time.monotonic()
    print(
        json.dumps(
            {
                "phase": "locking_source",
                "files": len(plan["entries"]),
                "retain_gib": round(required / 2**30, 3),
            }
        ),
        flush=True,
    )
    with (
        ExitStack() as stack,
        (records / "copy-journal.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        ) as journal,
    ):
        readers = {}
        for row in plan["entries"]:
            if not row["reparse"]:
                readers[row["relative"]] = stack.enter_context(
                    locked_reader(source / row["relative"])
                )
        # A matching existing journal may resume completed bytes, never a changed source.
        previous = {}
        for line in (records / "copy-journal.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record.get("phase") == "copied":
                previous[record["relative"]] = record
        for row in sorted(plan["entries"], key=lambda r: (-r["size"], r["relative"])):
            if not row["decision"].startswith("retain"):
                continue
            relative = row["relative"]
            target = destination / relative
            if target.resolve() != target:
                raise ValueError("Destination alias/reparse point: " + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                old = previous.get(relative)
                if old is None or target.stat().st_size != row["size"]:
                    raise ValueError(
                        "Existing target lacks verified resume identity: " + relative
                    )
                completed.append(old)
                continue
            temporary = target.with_name(target.name + ".migration-partial")
            if temporary.resolve() != temporary:
                raise ValueError("Partial output contains an alias/reparse point")
            reader = readers[relative]
            result = hashlib.sha256()
            size = 0
            resume_existing = temporary.exists()
            if resume_existing:
                if temporary.stat().st_size > row["size"]:
                    raise ValueError("Partial output exceeds planned source size")
                # Certify the retained prefix before appending. Never discard a
                # partial copy or repeat completed large-file transfers.
                with temporary.open("rb", buffering=0) as prefix:
                    while block := prefix.read(CHUNK):
                        budget.consume(2 * len(block))
                        if reader.read(len(block)) != block:
                            raise ValueError(
                                "Partial prefix differs from the locked source"
                            )
                        result.update(block)
                        size += len(block)
            with temporary.open("r+b" if resume_existing else "xb") as out:
                out.seek(0, os.SEEK_END)
                if out.tell() != size:
                    raise ValueError("Partial output changed after prefix validation")
                while block := reader.read(CHUNK):
                    budget.consume(2 * len(block))
                    result.update(block)
                    out.write(block)
                    size += len(block)
                    written += len(block)
                    if time.monotonic() - last >= 15:
                        print(
                            json.dumps(
                                {
                                    "phase": "copy",
                                    "file": relative,
                                    "copied_gib": round(written / 2**30, 2),
                                    "copied_files": len(completed),
                                }
                            ),
                            flush=True,
                        )
                        last = time.monotonic()
                out.flush()
                os.fsync(out.fileno())
            if size != row["size"] or temporary.stat().st_size != size:
                raise ValueError("Copied byte identity failed: " + relative)
            os.utime(temporary, ns=(row["mtime_ns"], row["mtime_ns"]))
            temporary.rename(target)
            os.chmod(target, stat.S_IREAD)
            record = {
                "phase": "copied",
                "relative": relative,
                "size": size,
                "content_sha256": result.hexdigest(),
            }
            journal.write(json.dumps(record, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
            completed.append(record)
        if inventory(source) != plan["entries"]:
            raise ValueError("Source inventory changed during migration")
        retained = [
            row for row in plan["entries"] if row["decision"].startswith("retain")
        ]
        if len(completed) != len(retained):
            raise ValueError("Retention population is incomplete")
        # One final SSD readback, while all source files are still write-locked.
        # No second HDD hash pass and no database/dataframe/financial evaluation.
        content_targets, duplicate_bytes, duplicate_files = {}, 0, 0
        for index, record in enumerate(completed):
            target = destination / record["relative"]
            if digest_file(target, budget) != record["content_sha256"]:
                raise ValueError("SSD readback digest mismatch: " + record["relative"])
            identity = target.stat()
            key = (record["content_sha256"], identity.st_mtime_ns)
            if key in content_targets:
                first = content_targets[key]
                staged_link = target.with_name(target.name + ".migration-link")
                os.link(first, staged_link)
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
                try:
                    os.replace(staged_link, target)
                finally:
                    os.chmod(first, stat.S_IREAD)
                    os.chmod(target, stat.S_IREAD)
                if not target.samefile(first):
                    raise ValueError("Duplicate identity was not preserved")
                duplicate_bytes += record["size"]
                duplicate_files += 1
                identity = target.stat()
            else:
                content_targets[key] = target
            record.update(
                {
                    "destination_inode": identity.st_ino,
                    "destination_mtime_ns": identity.st_mtime_ns,
                }
            )
            if time.monotonic() - last >= 15:
                print(
                    json.dumps(
                        {
                            "phase": "ssd_readback",
                            "verified_files": index + 1,
                            "total": len(completed),
                        }
                    ),
                    flush=True,
                )
                last = time.monotonic()
        receipt = {
            "plan_sha256": plan["sha256"],
            "applied_implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "aggregate_io_mib_per_second": io_mib_per_second,
            "source": str(source),
            "destination": str(destination),
            "retained_files": completed,
            "retained_bytes": required,
            "duplicate_files_hardlinked": duplicate_files,
            "duplicate_bytes_avoided": duplicate_bytes,
            "discarded_files": [
                row
                for row in plan["entries"]
                if not row["decision"].startswith("retain")
            ],
            "source_untouched": True,
            "content_analysis": False,
        }
        write_new(ready, receipt)
    return {
        "copy_verified": str(ready),
        "retained_files": len(completed),
        "retained_gib": round(required / 2**30, 3),
        "source_deleted": False,
    }


def main() -> None:
    """Use a bounded session-owned worker; install no recurring or detached task."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--source", type=Path, required=True)
    plan.add_argument("--destination", type=Path, required=True)
    plan.add_argument("--records", type=Path, required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    apply.add_argument("--io-mib-per-second", type=float, default=64)
    args = parser.parse_args()
    if args.action == "plan":
        print(create_plan(args.source, args.destination, args.records))
    elif args.worker:
        with background_io():
            print(
                json.dumps(
                    apply_plan(args.plan, io_mib_per_second=args.io_mib_per_second),
                    sort_keys=True,
                ),
                flush=True,
            )
    else:
        from simple_ai_trading.windows_owned_job import (
            CREATE_SUSPENDED,
            WindowsOwnedJob,
        )

        job, process = WindowsOwnedJob(), None
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tools.migrate_research_storage",
                    "apply",
                    "--worker",
                    "--plan",
                    str(args.plan.resolve()),
                    "--io-mib-per-second",
                    str(args.io_mib_per_second),
                ],
                stdin=subprocess.DEVNULL,
                stdout=sys.stdout,
                stderr=sys.stderr,
                creationflags=CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW,
            )
            job.enroll_and_resume(process)
            process.wait(timeout=7200)
            if process.returncode:
                raise RuntimeError("Migration worker failed; source was not deleted")
        finally:
            try:
                job.close()
            finally:
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
