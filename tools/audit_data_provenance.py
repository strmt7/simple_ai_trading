from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_REPO_DATA_PREFIXES = ("data/", "data\\")
FORBIDDEN_OPTIMIZATION_ARTIFACT_SUFFIXES = (".json", ".svg", ".png", ".csv", ".sqlite")
FORBIDDEN_DOC_PHRASES = (
    "deterministic_synthetic",
    "synthetic benchmark",
    "synthetic performance",
    "fixture evidence",
)
TEXT_AUDIT_SUFFIXES = (".md", ".py", ".json", ".svg")
TEXT_AUDIT_PREFIXES = ("readme.md", "docs/", "tools/")
TEXT_AUDIT_EXCLUDE = {"tools/audit_data_provenance.py"}


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def audit() -> list[str]:
    failures: list[str] = []
    for item in _tracked_files():
        normalized = item.replace("\\", "/")
        lower = normalized.lower()
        path = REPO_ROOT / item
        if not path.exists():
            continue
        if lower.startswith(FORBIDDEN_REPO_DATA_PREFIXES):
            failures.append(f"tracked runtime data is forbidden: {item}")
        if lower.startswith("docs/optimization/") and lower.endswith(FORBIDDEN_OPTIMIZATION_ARTIFACT_SUFFIXES):
            failures.append(f"tracked optimization artifact requires real-data provenance and review: {item}")
        if (
            lower not in TEXT_AUDIT_EXCLUDE
            and lower.startswith(TEXT_AUDIT_PREFIXES)
            and lower.endswith(TEXT_AUDIT_SUFFIXES)
        ):
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for phrase in FORBIDDEN_DOC_PHRASES:
                if phrase in text:
                    failures.append(f"repo-facing generated evidence phrase {phrase!r} in {item}")
    return failures


def main() -> int:
    failures = audit()
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("data provenance audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
