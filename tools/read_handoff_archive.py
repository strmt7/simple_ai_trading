"""Read bounded, hash-verified historical handoff sections from local Git objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping

from tools.cocoindex_agent_search import run_captured_process


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/archive/agent-history/handoff-2026-11-11.json"
MAX_BLOB_BYTES = 2_000_000
MAX_MANIFEST_BYTES = 65_536
MAX_WINDOW_LINES = 120
MAX_OUTPUT_BYTES = 16_384


def _git(*arguments: str) -> bytes:
    command = shutil.which("git")
    if command is None:
        raise ValueError("Git is required to read archived handoff objects")
    try:
        child_env = {
            key: value
            for key, value in os.environ.items()
            if key.upper()
            in {
                "PATH",
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "TEMP",
                "TMP",
                "LANG",
                "LC_ALL",
            }
        }
        result = run_captured_process(
            [command, "--no-lazy-fetch", "--no-replace-objects", *arguments],
            cwd=ROOT,
            env=child_env,
            input_text="",
            timeout=10,
            inherit_environment=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(
            "Local Git archive read failed; no fetch was attempted"
        ) from exc
    if result.returncode:
        raise ValueError(
            "Archive local read failed with lazy fetching disabled; use Git with "
            "--no-lazy-fetch support and explicitly fetch the recorded source revision "
            "if its objects are missing"
        )
    # These frozen Markdown blobs are UTF-8/LF. Verify their original byte
    # digests after encoding; any transport conversion fails source admission.
    return result.stdout.encode("utf-8")


def load_archive(
    name: str, manifest_path: Path | None = None
) -> tuple[Mapping[str, object], list[str]]:
    path = MANIFEST if manifest_path is None else manifest_path
    with path.open("rb") as stream:
        metadata = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(metadata) > MAX_MANIFEST_BYTES:
        raise ValueError("Archive manifest exceeds the byte limit")
    manifest = json.loads(metadata)
    if not isinstance(manifest, dict):
        raise ValueError("Archive manifest must be an object")
    body = {k: v for k, v in manifest.items() if k != "result_sha256"}
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    if (
        manifest.get("schema") != "immutable-handoff-archive-v1"
        or manifest.get("result_sha256") != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError("Archive manifest digest or schema differs")
    archives = manifest.get("archives")
    if not isinstance(archives, dict):
        raise ValueError("Archive entries must be an object")
    item = archives.get(name)
    if not isinstance(item, dict):
        raise ValueError("Unknown handoff archive")
    for field in ("source_revision", "git_blob_sha1"):
        if (
            not isinstance(item.get(field), str)
            or re.fullmatch(r"[0-9a-f]{40}", item[field]) is None
        ):
            raise ValueError("Archive Git identity is invalid")
    size = item.get("bytes")
    if type(size) is not int or not 0 < size <= MAX_BLOB_BYTES:
        raise ValueError("Archive byte count exceeds the bounded source contract")
    if type(item.get("lines")) is not int or item["lines"] < 1:
        raise ValueError("Archive line count is invalid")
    source_path = item.get("source_path")
    if source_path not in {"docs/AGENT_START.md", "docs/CONTINUATION.md"}:
        raise ValueError("Archive source path differs")
    resolved = (
        _git("rev-parse", "--verify", f"{item['source_revision']}:{source_path}")
        .decode()
        .strip()
    )
    if resolved != item["git_blob_sha1"]:
        raise ValueError("Archive revision/path does not identify the recorded blob")
    if int(_git("cat-file", "-s", resolved)) != size:
        raise ValueError("Archive object size differs")
    raw = _git("cat-file", "blob", resolved)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != item.get("sha256"):
        raise ValueError("Archive source digest differs")
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    if len(lines) != item.get("lines"):
        raise ValueError("Archive line count differs")
    return item, lines


def archive_window(name: str, *, first_line: int, count: int) -> dict[str, object]:
    if (
        type(first_line) is not int
        or first_line < 1
        or type(count) is not int
        or not 1 <= count <= MAX_WINDOW_LINES
    ):
        raise ValueError("Use a positive line and 1..120 lines per window")
    item, lines = load_archive(name)
    if first_line > len(lines):
        raise ValueError("Requested line is beyond the archive")
    return {
        "archive": name,
        "source_revision": item["source_revision"],
        "source_path": item["source_path"],
        "sha256": item["sha256"],
        "first_line": first_line,
        "lines": lines[first_line - 1 : first_line - 1 + count],
    }


def archive_find(name: str, *, literal: str) -> dict[str, object]:
    if not isinstance(literal, str) or not literal or len(literal) > 256:
        raise ValueError("Use a nonempty exact literal of at most 256 characters")
    item, lines = load_archive(name)
    matches = []
    count = 0
    for index, line in enumerate(lines, 1):
        if literal in line:
            count += 1
            if len(matches) < 20:
                matches.append(index)
    return {
        "archive": name,
        "source_revision": item["source_revision"],
        "sha256": item["sha256"],
        "matching_lines": matches,
        "match_count": count,
        "locations_truncated": count > 20,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", help="Archive name from the current verified manifest"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--line", type=int)
    selection.add_argument(
        "--find", help="Exact case-sensitive literal; reports locations only"
    )
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args(argv)
    try:
        result = (
            archive_find(args.archive, literal=args.find)
            if args.find is not None
            else archive_window(args.archive, first_line=args.line, count=args.count)
        )
        # ASCII JSON preserves Unicode exactly without assuming a host code page.
        output = json.dumps(result, ensure_ascii=True, indent=2)
        if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise ValueError(
                "Requested window exceeds the output byte limit; request fewer lines"
            )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Do not expose local paths or native Git stderr in reader diagnostics.
        message = (
            str(exc)
            if isinstance(exc, ValueError)
            else "Archive metadata could not be read"
        )
        print(message, file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
