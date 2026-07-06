from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_REPO_DATA_PREFIXES = ("data/", "data\\")
FORBIDDEN_OPTIMIZATION_ARTIFACT_SUFFIXES = (".json", ".svg", ".png", ".csv", ".csv.gz", ".sqlite")
GRAPH_ARTIFACT_SUFFIXES = (".svg", ".png")
PROGRESS_GRAPH_PREFIX = "docs/optimization/iteration-progress/"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text_reader(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _csv_shape(path: Path) -> tuple[int, tuple[str, ...]]:
    with _open_text_reader(path) as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        return sum(1 for _row in reader), tuple(str(column) for column in (header or ()))


def _manifest_by_path(payload: dict) -> dict[str, dict]:
    manifest = payload.get("artifact_integrity")
    if not isinstance(manifest, list):
        return {}
    output: dict[str, dict] = {}
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        output[raw_path.replace("\\", "/")] = entry
    return output


def _artifact_integrity_failures(payload: dict, *, report_path: Path) -> list[str]:
    failures: list[str] = []
    report_rel = report_path.relative_to(REPO_ROOT).as_posix()
    tracked = payload.get("tracked_artifacts")
    if not isinstance(tracked, list):
        return ["tracked_artifacts is missing or not a list"]
    tracked_paths = [str(path).replace("\\", "/") for path in tracked]
    if report_rel not in tracked_paths:
        failures.append("tracked_artifacts is missing report.json")
    manifest = _manifest_by_path(payload)
    if not manifest:
        return ["artifact_integrity is missing or empty"]
    for normalized in sorted(dict.fromkeys(tracked_paths)):
        if normalized == report_rel:
            continue
        if not normalized.startswith("docs/optimization/"):
            failures.append(f"tracked artifact escapes docs/optimization: {normalized}")
            continue
        entry = manifest.get(normalized)
        if entry is None:
            failures.append(f"missing artifact_integrity entry: {normalized}")
            continue
        path = REPO_ROOT / normalized
        if not path.exists() or not path.is_file():
            failures.append(f"tracked artifact file is missing: {normalized}")
            continue
        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            failures.append(f"invalid byte count in artifact_integrity: {normalized}")
        elif path.stat().st_size != expected_bytes:
            failures.append(f"byte count mismatch for {normalized}")
        expected_hash = entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            failures.append(f"invalid sha256 in artifact_integrity: {normalized}")
        elif _file_sha256(path) != expected_hash.lower():
            failures.append(f"sha256 mismatch for {normalized}")
        lower = normalized.lower()
        if lower.endswith(".csv") or lower.endswith(".csv.gz"):
            try:
                row_count, columns = _csv_shape(path)
            except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, csv.Error):
                failures.append(f"csv artifact cannot be parsed: {normalized}")
                continue
            expected_rows = entry.get("row_count")
            if not isinstance(expected_rows, int) or expected_rows < 0:
                failures.append(f"invalid row_count in artifact_integrity: {normalized}")
            elif row_count != expected_rows:
                failures.append(f"row_count mismatch for {normalized}")
            expected_columns = entry.get("columns")
            if not isinstance(expected_columns, list) or [str(item) for item in expected_columns] != list(columns):
                failures.append(f"column mismatch for {normalized}")
    return failures


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
    integrity_failures = _artifact_integrity_failures(payload, report_path=report_path)
    if integrity_failures:
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
    tracked_files = _tracked_files()
    optimization_chart_rounds = {
        normalized.split("/")[2]
        for normalized in (item.replace("\\", "/") for item in tracked_files)
        if normalized.lower().startswith("docs/optimization/")
        and "/charts/" in normalized.lower()
        and normalized.lower().endswith(GRAPH_ARTIFACT_SUFFIXES)
        and not normalized.lower().startswith(PROGRESS_GRAPH_PREFIX)
    }
    if len(optimization_chart_rounds) > 1:
        failures.append(
            "tracked optimization result graphs must be latest-only; "
            f"found chart artifacts in {', '.join(sorted(optimization_chart_rounds))}"
        )
    for item in tracked_files:
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
        ):
            report_path = _optimization_report_for(normalized)
            if report_path is None or not report_path.exists():
                failures.append(f"tracked optimization artifact requires real-data provenance and review: {item}")
            else:
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                if not isinstance(payload, dict) or not _tracked_optimization_artifact_allowed(normalized):
                    failures.append(f"tracked optimization artifact requires real-data provenance and review: {item}")
                elif normalized == report_path.relative_to(REPO_ROOT).as_posix():
                    for failure in _artifact_integrity_failures(payload, report_path=report_path):
                        failures.append(f"optimization artifact integrity failure: {failure}")
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
