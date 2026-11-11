from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from tools import capture_native_windows_preview as module


pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="Windows owned preview contract"
)


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "blank",
        "nonzero",
        "missing",
        "timeout",
        "enroll",
        "spawn",
        "close",
        "wrong_page",
        "missing_page",
        "oversized_page",
    ],
)
def test_owned_preview_cleans_every_exit_without_uncontained_fallback(
    tmp_path, monkeypatch, outcome
):
    image_module = pytest.importorskip("PIL.Image")
    from simple_ai_trading import windows_owned_job

    executable = tmp_path / "SimpleAITrading.exe"
    executable.write_bytes(b"test-owned-executable")
    output = tmp_path / "preview.png"
    observations = {"closed": False, "enrolled": False, "killed": False}

    class Process:
        returncode = None

        def wait(self, timeout):
            assert timeout in (5, 30)
            if outcome == "timeout" and self.returncode is None:
                raise subprocess.TimeoutExpired("owned-preview", timeout)
            if self.returncode is None:
                self.returncode = 2 if outcome == "nonzero" else 0
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            observations["killed"] = True
            self.returncode = -9

    process = Process()

    class Job:
        def enroll_and_resume(self, child):
            assert child is process
            if outcome == "enroll":
                raise OSError("containment refused")
            observations["enrolled"] = True

        def close(self):
            observations["closed"] = True
            if outcome == "close":
                raise OSError("job closure failed")
            if observations["enrolled"] and process.returncode is None:
                process.returncode = -15

    def start(command, **kwargs):
        assert command == [str(executable)]
        assert kwargs["creationflags"] & windows_owned_job.CREATE_SUSPENDED
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert kwargs["startupinfo"].wShowWindow == subprocess.SW_HIDE
        assert kwargs["env"]["SIMPLE_AI_TRADING_GUI_DRY_RUN"] == "1"
        assert "SIMPLE_AI_TRADING_GUI_SMOKE" not in kwargs["env"]
        assert "PREVIEW_PARENT_ONLY_TEST_VALUE" not in kwargs["env"]
        assert kwargs["env"]["SIMPLE_AI_TRADING_GUI_PREVIEW_PAGE"] == "System"
        bitmap = Path(kwargs["env"]["SIMPLE_AI_TRADING_GUI_PREVIEW_BMP"])
        observations["temporary"] = bitmap.parent
        if outcome == "spawn":
            raise OSError("launch refused")
        if outcome != "missing":
            image = image_module.new("RGB", (32, 32))
            if outcome != "blank":
                image.putdata([(i % 256, i // 256, 99) for i in range(1024)])
            image.save(bitmap)
            if outcome != "missing_page":
                index = (
                    0
                    if outcome == "wrong_page"
                    else module.PREVIEW_PAGES.index("System")
                )
                Path(str(bitmap) + ".page").write_bytes(f"{index}\n".encode("ascii"))
                if outcome == "oversized_page":
                    Path(str(bitmap) + ".page").write_bytes(b"5\nextra")
        return process

    monkeypatch.setattr(module.subprocess, "Popen", start)
    monkeypatch.setattr(windows_owned_job, "WindowsOwnedJob", Job)
    monkeypatch.setenv("SIMPLE_AI_TRADING_GUI_SMOKE", "1")
    monkeypatch.setenv("PREVIEW_PARENT_ONLY_TEST_VALUE", "synthetic sentinel")
    if outcome == "success":
        result = module.capture(
            executable, output, repository_root=tmp_path, page="System"
        )
        assert result["hidden"] and result["dry_run"] and result["owned_process_exited"]
        assert result["page"] == "System" and result["page_binding_verified"]
        assert result["implicit_environment_inheritance"] is False
        with image_module.open(output) as saved:
            assert saved.size == (32, 32)
    else:
        with pytest.raises((RuntimeError, OSError, subprocess.TimeoutExpired)):
            module.capture(executable, output, repository_root=tmp_path, page="System")
        assert not output.exists()
    assert observations["closed"]
    assert not observations["temporary"].exists()
    if outcome != "spawn":
        assert process.returncode is not None
    assert observations["killed"] == (outcome == "enroll")


def test_preview_output_collision_rejects_before_any_launch(tmp_path, monkeypatch):
    pytest.importorskip("PIL.Image")
    executable = tmp_path / "SimpleAITrading.exe"
    executable.touch()
    output = tmp_path / "existing.png"
    output.write_bytes(b"preserve-this-evidence")
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("launch before validation"),
    )
    with pytest.raises(ValueError, match="new output"):
        module.capture(executable, output, repository_root=tmp_path)
    assert output.read_bytes() == b"preserve-this-evidence"


@pytest.mark.parametrize(
    "width,height",
    [
        (True, 1020),
        (1279, 1020),
        (1680, 859),
        (3841, 1020),
        (1680, 2161),
        (1680, "1020"),
    ],
)
def test_invalid_preview_geometry_cannot_launch(tmp_path, monkeypatch, width, height):
    pytest.importorskip("PIL.Image")
    executable = tmp_path / "SimpleAITrading.exe"
    executable.touch()
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("invalid geometry launched"),
    )
    with pytest.raises(ValueError, match="geometry"):
        module.capture(
            executable,
            tmp_path / "new.png",
            repository_root=tmp_path,
            width_dips=width,
            height_dips=height,
        )


@pytest.mark.parametrize("invalid", ["foreign", "missing", "output_parent"])
def test_unowned_or_missing_paths_cannot_launch(tmp_path, monkeypatch, invalid):
    pytest.importorskip("PIL.Image")
    root = tmp_path / "owned"
    root.mkdir()
    executable = (tmp_path if invalid == "foreign" else root) / "SimpleAITrading.exe"
    if invalid != "missing":
        executable.touch()
    output = (root / "missing" if invalid == "output_parent" else root) / "new.png"
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("unqualified path launched"),
    )
    with pytest.raises(ValueError, match="owned executable"):
        module.capture(executable, output, repository_root=root)


@pytest.mark.parametrize("page", ["", "system", "../System", None, 1, ["System"]])
def test_unknown_preview_page_cannot_launch(tmp_path, monkeypatch, page):
    pytest.importorskip("PIL.Image")
    executable = tmp_path / "SimpleAITrading.exe"
    executable.touch()
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid page launched")
    )
    with pytest.raises(ValueError, match="page"):
        module.capture(
            executable, tmp_path / "new.png", repository_root=tmp_path, page=page
        )
