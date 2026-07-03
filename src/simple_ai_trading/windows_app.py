"""Launcher for the native Windows app backed by the CLI command contract."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess  # nosec B404
import sys

from .command_contract import command_names


WINDOWS_APP_COMMANDS = tuple(command_names())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def native_executable_candidates() -> tuple[Path, ...]:
    root = _repo_root()
    return (
        root / "build" / "windows" / "Release" / "SimpleAITrading.exe",
        root / "build" / "windows" / "SimpleAITrading.exe",
        root / "native" / "windows" / "build" / "SimpleAITrading.exe",
    )


def find_native_executable() -> Path | None:
    for path in native_executable_candidates():
        if path.exists():
            return path
    return None


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "usage: simple-ai-trading-windows [--help]\n\n"
            "Launch the native Windows operator app. Build it first with "
            "`tools\\build_native_windows.ps1` if no executable is present."
        )
        return 0
    exe = find_native_executable()
    if exe is None:
        candidates = "\n".join(str(path) for path in native_executable_candidates())
        print(
            "Native Windows app executable was not found. Build it with "
            "`tools\\build_native_windows.ps1`, then run this launcher again.\n"
            f"Checked:\n{candidates}",
            file=sys.stderr,
        )
        return 2
    env = os.environ.copy()
    env.setdefault("SIMPLE_AI_TRADING_PYTHON", sys.executable)
    return subprocess.call([str(exe)], env=env)  # nosec B603 - repo-local executable, validated above


if __name__ == "__main__":
    raise SystemExit(main())
