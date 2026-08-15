#!/usr/bin/env python3
"""Fail closed when the installed CLI launcher does not match the source contract."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


_DISTRIBUTION = "simple-ai-trading"
_CONSOLE_SCRIPT = "simple-ai-trading"
_EXPECTED_ENTRYPOINT = "simple_ai_trading.entrypoint:main"
_REQUIRED_COMMAND = "polymarket-live"


def _installed_entrypoints() -> tuple[str, ...]:
    distribution = metadata.distribution(_DISTRIBUTION)
    return tuple(
        entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name == _CONSOLE_SCRIPT
    )


def _launcher_path() -> Path:
    # Resolving a POSIX venv's Python symlink can escape its bin directory.
    scripts = Path(sys.executable).parent
    candidates = (
        scripts / f"{_CONSOLE_SCRIPT}.exe",
        scripts / _CONSOLE_SCRIPT,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"{_CONSOLE_SCRIPT} launcher is missing beside {sys.executable}"
    )


def _registered_commands() -> tuple[str, ...]:
    from simple_ai_trading import entrypoint  # noqa: PLC0415

    parser = entrypoint._build_parser()  # noqa: SLF001
    choices: dict[str, Any] = {}
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            choices = action.choices
            break
    return tuple(sorted(choices))


def verify_runtime_entrypoint(repository: Path) -> dict[str, object]:
    repository = repository.resolve()
    project_path = repository / "pyproject.toml"
    with project_path.open("rb") as handle:
        import tomllib  # noqa: PLC0415

        project = tomllib.load(handle)

    declared = str(project["project"]["scripts"][_CONSOLE_SCRIPT])
    if declared != _EXPECTED_ENTRYPOINT:
        raise RuntimeError(
            "source console entry point mismatch: "
            f"expected {_EXPECTED_ENTRYPOINT!r}, found {declared!r}"
        )

    installed = _installed_entrypoints()
    if installed != (_EXPECTED_ENTRYPOINT,):
        raise RuntimeError(
            "installed console entry point mismatch: "
            f"expected {(_EXPECTED_ENTRYPOINT,)!r}, found {installed!r}; "
            "reinstall the checkout into the selected environment"
        )

    from simple_ai_trading import entrypoint  # noqa: PLC0415

    loaded_module = Path(entrypoint.__file__).resolve()
    expected_module = (
        repository / "src" / "simple_ai_trading" / "entrypoint.py"
    ).resolve()
    if loaded_module != expected_module:
        raise RuntimeError(
            "selected Python resolves a different checkout: "
            f"expected {expected_module}, found {loaded_module}"
        )

    commands = _registered_commands()
    if _REQUIRED_COMMAND not in commands:
        raise RuntimeError(
            f"installed parser does not register required command {_REQUIRED_COMMAND!r}"
        )

    launcher = _launcher_path()
    smoke = subprocess.run(
        [str(launcher), "--help"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
    )
    if smoke.returncode != 0:
        raise RuntimeError(
            f"installed launcher help failed with exit code {smoke.returncode}"
        )
    if _REQUIRED_COMMAND not in smoke.stdout:
        raise RuntimeError(
            f"installed launcher help omits required command {_REQUIRED_COMMAND!r}"
        )

    return {
        "console_script": _CONSOLE_SCRIPT,
        "declared_entrypoint": declared,
        "installed_entrypoint": installed[0],
        "launcher": str(launcher),
        "loaded_module": str(loaded_module),
        "required_command": _REQUIRED_COMMAND,
        "verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify installed CLI and source entry-point parity."
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args(argv)
    try:
        report = verify_runtime_entrypoint(args.repository)
    except (
        KeyError,
        metadata.PackageNotFoundError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"runtime entry-point verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
