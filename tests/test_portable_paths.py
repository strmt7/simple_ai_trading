from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from tools.check_portable_paths import contains_workstation_path, inspect_diff


def test_host_path_is_reported_without_exposing_the_value():
    drive = Path.cwd().drive or "V:"
    private = drive + chr(92) + "fixture" + chr(92) + "private-record.json"
    assert contains_workstation_path(repr(private))
    diff = "+++ b/config.py\n@@ -0,0 +1 @@\n+value = " + repr(private)
    errors = inspect_diff(diff)
    assert errors == [
        ("config.py", 1, "workstation path candidate; matching value withheld")
    ]
    assert private not in str(errors)


def test_relative_paths_urls_and_reserved_test_hosts_are_not_workstation_bindings():
    for line in [
        'Path("data") / "historical-research"',
        "https://example.invalid/docs",
        r"\\fixture.invalid\share\file",
        repr(r".venv\Scripts\python.exe"),
    ]:
        assert not contains_workstation_path(line)


def test_removing_old_path_and_private_binary_additions_are_distinct(tmp_path):
    private = str(tmp_path / "old-record.json")
    assert (
        inspect_diff(
            "+++ b/config.py\n@@ -1 +1 @@\n-" + private + "\n+root = configured_root"
        )
        == []
    )
    assert inspect_diff(
        "+++ b/data/storage-migrations/private.zip\nBinary files differ"
    ) == [
        ("<private-migration-content>", 0, "private migration data cannot be tracked")
    ]


def test_unc_source_literals_are_detected_before_and_after_string_escaping():
    host = "private-" + "machine"
    private = chr(92) * 2 + host + chr(92) + "share" + chr(92) + "evidence.bin"
    assert contains_workstation_path(private)
    assert contains_workstation_path(repr(private))


def test_real_staged_binary_cannot_bypass_the_private_content_gate(tmp_path):
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    private = tmp_path / "data/storage-migrations/private.zip"
    private.parent.mkdir(parents=True)
    private.write_bytes(bytes(range(256)))
    subprocess.run(
        ["git", "add", "--", "data/storage-migrations/private.zip"],
        cwd=tmp_path,
        check=True,
    )
    script = Path(__file__).resolve().parents[1] / "tools/check_portable_paths.py"
    result = subprocess.run(
        [sys.executable, str(script), "--staged"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "private migration data cannot be tracked" in result.stdout
    assert "private.zip" not in result.stdout


def test_active_runtime_has_no_workstation_path_constants():
    root = Path(__file__).resolve().parents[1]
    candidates = [
        *(root / "src").rglob("*.py"),
        *(root / "tools").glob("*.py"),
        *(root / "tools").glob("*.ps1"),
        *(root / "native/windows/src").glob("*.cpp"),
        *(root / "native/windows/src").glob("*.hpp"),
    ]
    violations = [
        (p.relative_to(root).as_posix(), n)
        for p in candidates
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if contains_workstation_path(line)
    ]
    assert not violations, "Workstation-path candidates (values withheld): " + repr(
        violations
    )
