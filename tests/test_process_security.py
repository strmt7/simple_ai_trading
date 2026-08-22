from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

from simple_ai_trading.process_security import harden_executable_search_path


def test_hardening_removes_relative_current_and_duplicate_path_entries(
    tmp_path: Path,
) -> None:
    current = tmp_path / "working"
    trusted = tmp_path / "trusted"
    current.mkdir()
    trusted.mkdir()
    environment = {
        "PATH": os.pathsep.join(
            ("", ".", "relative-bin", str(current), str(trusted), str(trusted))
        )
    }

    entries = harden_executable_search_path(
        environment,
        current_directory=current,
    )

    assert entries == (str(trusted.resolve()),)
    assert environment["PATH"] == str(trusted.resolve())
    assert environment["NoDefaultCurrentDirectoryInExePath"] == "1"


def test_hardening_is_idempotent(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    current = tmp_path / "working"
    trusted.mkdir()
    current.mkdir()
    environment = {"PATH": str(trusted)}

    first = harden_executable_search_path(
        environment,
        current_directory=current,
    )
    second = harden_executable_search_path(
        environment,
        current_directory=current,
    )

    assert first == second == (str(trusted.resolve()),)


def test_package_bootstrap_blocks_current_directory_executable_hijack(
    tmp_path: Path,
) -> None:
    real_git = shutil.which("git")
    assert real_git is not None
    fake_git = tmp_path / ("git.exe" if os.name == "nt" else "git")
    if os.name == "nt":
        shutil.copy2(Path(os.environ["ComSpec"]), fake_git)
        arguments = '["git", "/c", "echo", "FAKE_EXECUTABLE_MARKER"]'
    else:
        fake_git.write_text(
            "#!/bin/sh\nprintf 'FAKE_EXECUTABLE_MARKER\\n'\n",
            encoding="ascii",
        )
        fake_git.chmod(0o755)
        arguments = '["git", "ignored"]'

    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(tmp_path), str(Path(real_git).parent)))
    baseline_code = (
        "import subprocess; "
        f"result=subprocess.run({arguments},capture_output=True,text=True); "
        "print(result.stdout.strip())"
    )
    guarded_code = (
        "import simple_ai_trading, subprocess; "
        f"result=subprocess.run({arguments},capture_output=True,text=True); "
        "print(result.stdout.strip())"
    )

    baseline = subprocess.run(  # nosec B603
        [sys.executable, "-c", baseline_code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    guarded = subprocess.run(  # nosec B603
        [sys.executable, "-c", guarded_code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert baseline.stdout.strip() == "FAKE_EXECUTABLE_MARKER"
    assert "FAKE_EXECUTABLE_MARKER" not in guarded.stdout
