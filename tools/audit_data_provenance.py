from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_REPO_DATA_PREFIXES = ("data/", "data\\")
FORBIDDEN_OPTIMIZATION_ARTIFACT_SUFFIXES = (".json", ".svg", ".png", ".csv", ".csv.gz", ".sqlite")
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


def _optimization_report_for(path: str) -> Path | None:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    if len(parts) < 4 or parts[0] != "docs" or parts[1] != "optimization":
        return None
    return REPO_ROOT / "docs" / "optimization" / parts[2] / "data" / "report.json"


def _tracked_optimization_artifact_allowed(item: str) -> bool:
    normalized = item.replace("\\", "/")
    report_path = _optimization_report_for(normalized)
    if report_path is None or not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("artifact_class") != "exchange_sourced_backtest_graph_data":
        return False
    if payload.get("tracked_repo_artifact") is not True:
        return False
    report_rel = report_path.relative_to(REPO_ROOT).as_posix()
    if normalized == report_rel:
        return True
    tracked = payload.get("tracked_artifacts")
    if not isinstance(tracked, list):
        return False
    allowed = {str(path).replace("\\", "/") for path in tracked}
    return normalized in allowed


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
        if (
            lower.startswith("docs/optimization/")
            and lower.endswith(FORBIDDEN_OPTIMIZATION_ARTIFACT_SUFFIXES)
            and not _tracked_optimization_artifact_allowed(normalized)
        ):
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
