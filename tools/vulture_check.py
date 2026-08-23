"""Execute Vulture against tracked production Python files only."""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


DEFAULT_MIN_CONFIDENCE = 100
VULTURE_DEAD_CODE_EXIT = 3
TRACKED_PYTHON_PATHSPEC = "*.py"
EXCLUDED_TOP_LEVEL_DIRS: frozenset[str] = frozenset({"docs", "tests"})
IGNORED_FINDINGS: frozenset[tuple[str, int, str]] = frozenset(
    {
        (
            "src/simple_ai_trading/round75_continuous_capture.py",
            185,
            "unused variable 'exc_type'",
        ),
        (
            "src/simple_ai_trading/polymarket_round25_resolution_store.py",
            116,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_historical_l2.py",
            108,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_round22_ingestion.py",
            94,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_round22_targets.py",
            90,
            "unused variable 'allow_redirects'",
        ),
        (
            "src/simple_ai_trading/polymarket_round23_binance.py",
            76,
            "unused variable 'allow_redirects'",
        ),
    }
)


def resolve_required_executable(name: str) -> str:
    """Resolve the required executable.

    Inputs: `name` (str) name. Output: `str`. Raises: RuntimeError when validation or
    external operations fail.
    """
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"Required executable `{name}` is not available in PATH.")
    return resolved


def is_vulture_target(relative_path: PurePosixPath) -> bool:
    """Return True when a tracked Python file belongs to the production scope.

    Inputs: `relative_path`. Output: `bool`.
    """
    if relative_path.suffix != ".py":
        return False

    parts = relative_path.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_TOP_LEVEL_DIRS:
        return False
    if "tests" in parts:
        return False
    if any(part.startswith(".") for part in parts[:-1]):
        return False

    filename = relative_path.name
    if filename == "conftest.py":
        return False
    if filename.startswith("test_") or filename.endswith("_test.py"):
        return False
    return True


def _run_git(repo_root: Path, *args: str) -> str:
    """Run the git.

    Inputs: `repo_root` (Path), `*args` (str) positional arguments. Output: `str`.
    """
    safe_repo_root = str(repo_root.resolve())
    completed = subprocess.run(
        [
            resolve_required_executable("git"),
            "-c",
            f"safe.directory={safe_repo_root}",
            *args,
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def list_vulture_targets(repo_root: Path) -> list[str]:
    """Return the vulture targets.

    Inputs: `repo_root` (Path). Output: `list[str]`. Raises: RuntimeError when validation or the
    called operation fails.
    """
    tracked_files = _run_git(repo_root, "ls-files", "--", TRACKED_PYTHON_PATHSPEC)
    targets = [
        relative_path
        for relative_path in tracked_files.splitlines()
        if is_vulture_target(PurePosixPath(relative_path))
    ]
    if not targets:
        raise RuntimeError(
            "No tracked production Python files matched the Vulture scope."
        )
    return targets


def build_vulture_command(paths: list[str], *, min_confidence: int) -> list[str]:
    """The Vulture command for the given tracked paths.

    Inputs: `paths`, `min_confidence`. Output: `list[str]`.
    """
    return [
        sys.executable,
        "-m",
        "vulture",
        "--min-confidence",
        str(min_confidence),
        *paths,
    ]


def _filter_known_false_positives(output: str) -> str:
    """Remove exact protocol-signature findings from Vulture output."""

    kept: list[str] = []
    for line in output.splitlines():
        normalized = line.replace("\\", "/")
        ignored = any(
            normalized.startswith(f"{path}:{line_number}: {message} (")
            for path, line_number, message in IGNORED_FINDINGS
        )
        if not ignored:
            kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def _report_vulture_api(
    repo_root: Path,
    paths: list[str],
    *,
    min_confidence: int,
) -> int:
    """Run Vulture through its API when Windows cannot spawn the full command."""

    module = importlib.import_module("vulture")
    scanner = module.Vulture()
    scanner.scavenge(paths)
    findings = []
    for item in scanner.get_unused_code(min_confidence=min_confidence):
        filename = Path(item.filename)
        if filename.is_absolute():
            filename = filename.resolve().relative_to(repo_root)
        key = (filename.as_posix(), int(item.first_lineno), str(item.message))
        if key not in IGNORED_FINDINGS:
            findings.append(item)
    for item in findings:
        sys.stdout.write(
            f"{item.filename}:{item.first_lineno}: {item.message} "
            f"({item.confidence}% confidence)\n"
        )
    return int(bool(findings))


def run_vulture(repo_root: Path, paths: list[str], *, min_confidence: int) -> int:
    """Vulture from the repository root.

    Inputs: `repo_root`, `paths`, `min_confidence`. Output: `int`.
    """
    command = build_vulture_command(paths, min_confidence=min_confidence)
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        if getattr(exc, "winerror", None) != 206:
            raise
        return _report_vulture_api(
            repo_root,
            paths,
            min_confidence=min_confidence,
        )
    filtered = _filter_known_false_positives(completed.stdout)
    sys.stdout.write(filtered)
    sys.stderr.write(completed.stderr)
    if completed.returncode == VULTURE_DEAD_CODE_EXIT:
        return int(bool(filtered or completed.stderr))
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Inputs: none. Output: `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Run Vulture against tracked production Python files only."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing the tracked Python files.",
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Vulture minimum confidence threshold. Defaults to {DEFAULT_MIN_CONFIDENCE}.",
    )
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="Print the tracked production Python files that Vulture will scan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the `tools.vulture_check` command entrypoint.

    Inputs: `argv`. Output: `int`.
    """
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    targets = list_vulture_targets(repo_root)
    if args.print_files:
        sys.stdout.write("\n".join(targets) + "\n")
        return 0
    return run_vulture(repo_root, targets, min_confidence=args.min_confidence)


if __name__ == "__main__":
    raise SystemExit(main())
