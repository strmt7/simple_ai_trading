"""Inventory Git paths and parse Python syntax without executing project code.

This is mechanical coverage, not a semantic review or a security certification.
Data, raw captures, credentials, and non-code files are metadata-only entries.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tokenize


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOTS = frozenset({"src", "tools", "tests", "scripts"})


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def inspect_python(raw: bytes, path: str) -> dict[str, object]:
    """Use Python's declared encoding; never import the inspected module."""
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        source = raw.decode(encoding)
        tree = ast.parse(source, filename=path)
    except (SyntaxError, UnicodeError, LookupError) as exc:
        return {"mechanical_status": "parse_error", "error_type": type(exc).__name__}
    nodes = list(ast.walk(tree))
    return {
        "mechanical_status": "ast_parsed_not_semantically_reviewed",
        "lines": len(source.splitlines()),
        "functions": sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes
        ),
        "bare_except": sum(
            isinstance(n, ast.ExceptHandler) and n.type is None for n in nodes
        ),
    }


def build_inventory() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    # ls-files includes staged changes, but deliberately excludes all untracked
    # paths. Each row binds the index blob and, only for safe Python, worktree bytes.
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, blob, stage = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        if stage != "0":
            raise ValueError(
                "Resolve unmerged index entries before taking a review inventory"
            )
        parts = Path(path).parts
        row: dict[str, object] = {
            "path": path,
            "index_blob": blob,
            "mechanical_status": "metadata_only_not_reviewed",
        }
        safe_python = (
            mode in {"100644", "100755"}
            and path.endswith(".py")
            and parts[0] in CODE_ROOTS
            and not any(p.lower() in {"data", "raw", "secrets", ".env"} for p in parts)
        )
        if safe_python:
            target = ROOT / path
            if target.is_symlink() or not target.resolve().is_relative_to(ROOT):
                raise ValueError(f"Code path escapes the review root: {path}")
            if not target.is_file():
                row["mechanical_status"] = "missing_worktree_file"
            else:
                raw = target.read_bytes()
                row["working_sha256"] = hashlib.sha256(raw).hexdigest()
                row.update(inspect_python(raw, path))
        status = str(row["mechanical_status"])
        counts[status] = counts.get(status, 0) + 1
        rows.append(row)
    return rows, {
        "schema_version": "review-surface-inventory-v1",
        "base_head": git("rev-parse", "HEAD").decode().strip(),
        "scope": "index paths at invocation; Python worktree bytes only in src/tools/tests/scripts",
        "untracked_paths_included": False,
        "project_modules_executed": False,
        "protected_data_contents_read": False,
        "semantic_review_complete": False,
        "tracked_path_count": len(rows),
        "mechanical_status_counts": counts,
        "python_lines": sum(int(r.get("lines", 0)) for r in rows),
        "python_functions": sum(int(r.get("functions", 0)) for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "docs" / "review"):
        raise ValueError("Review outputs must remain under docs/review")
    paths = (output / "inventory.tsv", output / "summary.json")
    if any(p.exists() for p in paths):
        raise FileExistsError(
            "Review snapshots are immutable; use a distinct output directory"
        )
    rows, summary = build_inventory()
    output.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    fields = [
        "path",
        "index_blob",
        "working_sha256",
        "mechanical_status",
        "lines",
        "functions",
        "bare_except",
        "error_type",
    ]
    writer = csv.DictWriter(
        buffer, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    encoded = buffer.getvalue().encode("utf-8")
    summary["inventory_sha256"] = hashlib.sha256(encoded).hexdigest()
    with paths[0].open("xb") as handle:
        handle.write(encoded)
    with paths[1].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
