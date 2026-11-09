"""Reject newly published workstation paths and private migration data."""

from __future__ import annotations

import argparse
import re
import subprocess


_DRIVE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
_USER_HOME = re.compile(r"(?<![A-Za-z0-9:/])/(?:Users|home)/[^/\s'\"\\]+")
_UNC = re.compile(
    r"(?<![A-Za-z0-9_./\\-])\\{2,}(?P<host>[A-Za-z0-9_.-]+)\\+[A-Za-z0-9_$.-]+"
)
_PRIVATE = ("data/historical-research/", "data/storage-migrations/")


def contains_workstation_path(line: str) -> bool:
    """Identify candidates without returning private matching text to CI logs."""
    return bool(
        _DRIVE.search(line)
        or _USER_HOME.search(line)
        or any(
            not m.group("host").lower().endswith(".invalid")
            for m in _UNC.finditer(line)
        )
    )


def inspect_diff(diff: str) -> list[tuple[str, int, str]]:
    """Check additions, allowing removal of legacy strings without republishing them."""
    errors, path, number = [], "", 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path.startswith(_PRIVATE):
                errors.append(
                    (
                        "<private-migration-content>",
                        0,
                        "private migration data cannot be tracked",
                    )
                )
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            if match:
                number = int(match.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            if contains_workstation_path(line[1:]):
                errors.append(
                    (
                        path,
                        number,
                        "workstation path candidate; matching value withheld",
                    )
                )
            number += 1
        elif line.startswith(" "):
            number += 1
    return errors


def inspect_names(names: list[str]) -> list[tuple[str, int, str]]:
    """Check binary additions too; their patch bodies have no plus-prefixed lines."""
    return [
        ("<private-migration-content>", 0, "private migration data cannot be tracked")
        for name in names
        if name.startswith(_PRIVATE)
    ]


def main() -> None:
    """Check staged changes locally, or one explicit revision range in CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--staged", action="store_true")
    selection.add_argument("--base")
    args = parser.parse_args()
    command = [
        "git",
        "-c",
        "core.quotepath=false",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
    ]
    if args.staged:
        scope = ["--cached"]
    else:
        base = subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--verify",
                "--end-of-options",
                args.base + "^{commit}",
            ],
            text=True,
        ).strip()
        scope = [base, "HEAD"]
    names = subprocess.check_output(
        command + ["--name-only", "-z", "--diff-filter=ACMR", *scope, "--"], text=True
    ).split("\0")
    result = subprocess.run(
        command + ["--unified=0", *scope, "--"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    errors = inspect_diff(result.stdout)
    errors.extend(inspect_names(names))
    errors = sorted(set(errors))
    for path, line, reason in errors:
        print(f"{path}:{line}: {reason}")
    if errors:
        raise SystemExit(1)
    print("Portable-path publication check passed.")


if __name__ == "__main__":
    main()
