from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_REPO_DATA_PREFIXES = ("data/", "data\\")
FORBIDDEN_OPTIMIZATION_ARTIFACT_SUFFIXES = (".json", ".svg", ".png", ".csv", ".csv.gz", ".sqlite")
GRAPH_ARTIFACT_SUFFIXES = (".svg", ".png")
PROGRESS_GRAPH_PREFIX = "docs/optimization/iteration-progress/"
FOUNDATION_EVIDENCE_PREFIX = "docs/ai/foundation/latest/"
FOUNDATION_EVIDENCE_FILES = {
    "README.md",
    "benchmark.svg",
    "observations.csv",
    "report.json",
    "manifest.json",
}
HOST_PATH_PATTERN = re.compile(
    r"(?i)(?<![a-z])[a-z]:/|(?<![a-z0-9:])/(?:Users|home|tmp)/|(?<!:)//[^/\s]+/"
)
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


def _index_blob_sha256(path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f":{path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


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
        else:
            index_hash = _index_blob_sha256(normalized)
            if index_hash is not None and index_hash != expected_hash.lower():
                failures.append(f"Git-blob sha256 mismatch for {normalized}")
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


def _foundation_evidence_failures(tracked_files: list[str]) -> list[str]:
    failures: list[str] = []
    directory = REPO_ROOT / "docs" / "ai" / "foundation" / "latest"
    tracked = {
        item.replace("\\", "/")[len(FOUNDATION_EVIDENCE_PREFIX) :]
        for item in tracked_files
        if item.replace("\\", "/").startswith(FOUNDATION_EVIDENCE_PREFIX)
    }
    if not directory.exists() and not tracked:
        return failures
    if not directory.is_dir():
        return ["foundation latest evidence path is not a directory"]
    actual = {entry.name for entry in directory.iterdir() if entry.is_file()}
    if actual != FOUNDATION_EVIDENCE_FILES:
        failures.append(f"foundation latest file set is invalid: {sorted(actual)}")
        return failures
    if tracked and tracked != FOUNDATION_EVIDENCE_FILES:
        failures.append(f"tracked foundation latest file set is incomplete: {sorted(tracked)}")
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return failures + [f"foundation latest JSON cannot be parsed: {exc}"]
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        return failures + ["foundation latest manifest/report must be JSON objects"]
    expected_manifest_files = FOUNDATION_EVIDENCE_FILES - {"manifest.json"}
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict) or set(manifest_files) != expected_manifest_files:
        failures.append("foundation latest manifest file set is invalid")
    else:
        for name, expected_hash in manifest_files.items():
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                failures.append(f"foundation latest manifest hash is invalid: {name}")
            elif _file_sha256(directory / name) != expected_hash.lower():
                failures.append(f"foundation latest manifest hash mismatch: {name}")
            elif tracked:
                index_hash = _index_blob_sha256(FOUNDATION_EVIDENCE_PREFIX + name)
                if index_hash != expected_hash.lower():
                    failures.append(f"foundation latest Git-blob hash mismatch: {name}")
    if report.get("trading_authority") is not False or manifest.get("trading_authority") is not False:
        failures.append("foundation latest evidence must deny trading authority")
    if report.get("status") != manifest.get("status"):
        failures.append("foundation latest report/manifest status mismatch")
    try:
        report_observation_count = int(report.get("observation_count", -1))
        manifest_observation_count = int(manifest.get("observation_count", -2))
    except (TypeError, ValueError):
        report_observation_count = -1
        manifest_observation_count = -2
    if report_observation_count < 1 or report_observation_count != manifest_observation_count:
        failures.append("foundation latest report/manifest observation count mismatch")
    if _file_sha256(directory / "observations.csv") != report.get("observations_sha256"):
        failures.append("foundation latest observation hash does not match report")
    if _file_sha256(directory / "benchmark.svg") != report.get("chart_sha256"):
        failures.append("foundation latest chart hash does not match report")
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict) or "not accessed" not in str(
        evaluation.get("terminal_period", "")
    ):
        failures.append("foundation latest report does not preserve the sealed terminal period")
    evidence = report.get("source_evidence")
    symbols = (
        tuple(item.get("symbol") for item in evidence if isinstance(item, dict))
        if isinstance(evidence, list)
        else ()
    )
    if symbols != ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        failures.append(f"foundation latest symbol contract failed: {symbols}")
    try:
        row_count, columns = _csv_shape(directory / "observations.csv")
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        failures.append(f"foundation latest observations cannot be parsed: {exc}")
    else:
        if row_count != report_observation_count:
            failures.append("foundation latest CSV/report row count mismatch")
        required_columns = {"symbol", "decision_ms", "absolute_error", "random_walk_absolute_error"}
        if not required_columns.issubset(columns):
            failures.append("foundation latest observations omit required columns")
    chart_text = (directory / "benchmark.svg").read_text(encoding="utf-8", errors="ignore")
    if "not P&amp;L" not in chart_text:
        failures.append("foundation latest chart omits its non-P&L disclosure")
    serialized_report = json.dumps(report, sort_keys=True).replace("\\", "/")
    if HOST_PATH_PATTERN.search(serialized_report):
        failures.append("foundation latest report leaks a host-local path")
    return failures


def audit() -> list[str]:
    failures: list[str] = []
    tracked_files = _tracked_files()
    failures.extend(_foundation_evidence_failures(tracked_files))
    optimization_chart_rounds = {
        normalized.split("/")[2]
        for normalized in (item.replace("\\", "/") for item in tracked_files)
        if normalized.lower().startswith("docs/optimization/")
        and "/charts/" in normalized.lower()
        and normalized.lower().endswith(GRAPH_ARTIFACT_SUFFIXES)
        and not normalized.lower().startswith(PROGRESS_GRAPH_PREFIX)
        and (REPO_ROOT / normalized).is_file()
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
