"""Export one hidden native dry-run frame inside a bounded owned process job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def capture(
    executable: Path,
    output: Path,
    *,
    repository_root: Path | None = None,
    width_dips: int = 1680,
    height_dips: int = 1020,
) -> dict[str, object]:
    """Render in-process, retaining a new PNG only after clean owned-child exit."""
    if os.name != "nt":
        raise RuntimeError("Native preview requires Windows")
    from PIL import Image

    from simple_ai_trading.windows_owned_job import CREATE_SUSPENDED, WindowsOwnedJob

    executable, output = executable.resolve(), output.resolve()
    root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    if (
        type(width_dips) is not int
        or type(height_dips) is not int
        or not 1280 <= width_dips <= 3840
        or not 860 <= height_dips <= 2160
    ):
        raise ValueError("Preview geometry is outside the native bounded scope")
    if (
        not executable.is_relative_to(root)
        or not executable.is_file()
        or output.exists()
        or not output.parent.is_dir()
    ):
        raise ValueError(
            "Require an owned executable and a new output in an existing directory"
        )
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    with tempfile.TemporaryDirectory(prefix="trading-native-preview-") as directory:
        bitmap = Path(directory) / "preview.bmp"
        env = os.environ.copy()
        env["SIMPLE_AI_TRADING_GUI_DRY_RUN"] = "1"
        env["SIMPLE_AI_TRADING_GUI_PREVIEW_BMP"] = str(bitmap)
        env["SIMPLE_AI_TRADING_GUI_PREVIEW_WIDTH_DIPS"] = str(width_dips)
        env["SIMPLE_AI_TRADING_GUI_PREVIEW_HEIGHT_DIPS"] = str(height_dips)
        env["SIMPLE_AI_TRADING_REPO_ROOT"] = str(root)
        env.pop("SIMPLE_AI_TRADING_GUI_SMOKE", None)
        job, process = WindowsOwnedJob(), None
        try:
            process = subprocess.Popen(
                [str(executable)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startup,
                creationflags=CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW,
            )
            job.enroll_and_resume(process)
            process.wait(timeout=30)
            if process.returncode != 0 or not bitmap.is_file():
                raise RuntimeError(
                    "Native hidden preview did not exit with a complete frame"
                )
        finally:
            try:
                job.close()
            finally:
                if process is not None and process.poll() is None:
                    # Original Popen handle, including a failed suspended enrollment.
                    process.kill()
                    process.wait(timeout=5)
        with Image.open(bitmap) as image:
            image.load()
            colors = image.getcolors(16)
            if colors is not None and len(colors) < 16:
                raise RuntimeError(
                    "Native hidden preview has insufficient substantive content"
                )
            with output.open("xb") as destination:
                image.save(destination, format="PNG")
            width, height = image.size
    return {
        "width": width,
        "height": height,
        "width_dips": width_dips,
        "height_dips": height_dips,
        "hidden": True,
        "dry_run": True,
        "owned_process_exited": True,
        "output": str(output),
    }


def main() -> None:
    """Capture once without activation, desktop input or persistent installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-dips", type=int, default=1680)
    parser.add_argument("--height-dips", type=int, default=1020)
    args = parser.parse_args()
    print(
        json.dumps(
            capture(
                args.exe,
                args.output,
                width_dips=args.width_dips,
                height_dips=args.height_dips,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
