from __future__ import annotations

from importlib import metadata
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from simple_ai_trading import entrypoint
from tools import verify_runtime_entrypoint as verifier


def _repository() -> Path:
    return Path(entrypoint.__file__).resolve().parents[2]


def test_runtime_entrypoint_matches_source_parser_and_launcher() -> None:
    report = verifier.verify_runtime_entrypoint(_repository())

    assert report["verified"] is True
    assert report["declared_entrypoint"] == "simple_ai_trading.entrypoint:main"
    assert report["installed_entrypoint"] == "simple_ai_trading.entrypoint:main"
    assert report["required_command"] == "polymarket-live"
    assert Path(str(report["launcher"])).is_file()


def test_runtime_parser_discovery_does_not_require_optional_torch() -> None:
    code = """
import sys
sys.modules["torch"] = None
from simple_ai_trading import entrypoint
parser = entrypoint._build_parser()
choices = next(
    action.choices
    for action in parser._actions
    if isinstance(action, __import__("argparse")._SubParsersAction)
)
assert "binance-round74-develop" in choices
assert "binance-round74-sealed-evaluate" in choices
assert "polymarket-live" in choices
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_entrypoint_rejects_stale_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = SimpleNamespace(
        entry_points=(
            SimpleNamespace(
                group="console_scripts",
                name="simple-ai-trading",
                value="simple_ai_trading.cli:main",
            ),
        )
    )
    monkeypatch.setattr(metadata, "distribution", lambda _name: stale)

    with pytest.raises(RuntimeError, match="installed console entry point mismatch"):
        verifier.verify_runtime_entrypoint(_repository())


def test_runtime_entrypoint_rejects_different_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "simple-ai-trading"\n'
        "[project.scripts]\n"
        'simple-ai-trading = "simple_ai_trading.entrypoint:main"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        verifier,
        "_installed_entrypoints",
        lambda: ("simple_ai_trading.entrypoint:main",),
    )

    with pytest.raises(RuntimeError, match="resolves a different checkout"):
        verifier.verify_runtime_entrypoint(tmp_path)


def test_runtime_entrypoint_rejects_missing_required_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "_registered_commands", tuple)

    with pytest.raises(RuntimeError, match="does not register required command"):
        verifier.verify_runtime_entrypoint(_repository())


@pytest.mark.parametrize(
    ("return_code", "stdout", "expected"),
    (
        (5, "", "launcher help failed"),
        (0, "usage: simple-ai-trading", "launcher help omits required command"),
    ),
)
def test_runtime_entrypoint_rejects_broken_launcher(
    return_code: int,
    stdout: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "_launcher_path", lambda: Path(__file__))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=return_code,
            stdout=stdout,
        ),
    )

    with pytest.raises(RuntimeError, match=expected):
        verifier.verify_runtime_entrypoint(_repository())


def test_runtime_entrypoint_main_reports_launcher_timeout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _timeout(_repository: Path) -> dict[str, object]:
        raise subprocess.TimeoutExpired("simple-ai-trading", 20)

    monkeypatch.setattr(verifier, "verify_runtime_entrypoint", _timeout)

    assert verifier.main(["--repository", str(_repository())]) == 1
    assert "runtime entry-point verification failed" in capsys.readouterr().err
