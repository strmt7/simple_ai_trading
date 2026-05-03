from __future__ import annotations

import argparse
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

import pyte


DOWN = "\x1b[B"
UP = "\x1b[A"
ENTER = "\r"
ESCAPE = "\x1b"


def _render(chunks: list[str], *, rows: int, cols: int) -> str:
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    stream.feed("".join(chunks))
    return "\n".join(screen.display)


def _visible_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def _assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: expected screen to contain {needle!r}\n{text}")


def _assert_not_contains(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label}: expected screen not to contain {needle!r}\n{text}")


def _assert_ordered_highlight(text: str, expected: str, label: str) -> None:
    marker = f"> {expected}"
    if marker not in text:
        lines = "\n".join(
            line
            for line in _visible_lines(text)
            if "Connection" in line or "Strategy" in line or "Orders" in line or "Compute" in line
        )
        raise AssertionError(f"{label}: expected highlighted row {marker!r}\n{lines}")


class _WindowsPty:
    def __init__(self, argv: list[str], cwd: Path, rows: int, cols: int, env_overrides: dict[str, str] | None = None) -> None:
        import winpty

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("COLORTERM", "truecolor")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self.process = winpty.PtyProcess.spawn(argv, cwd=str(cwd), env=env, dimensions=(rows, cols))
        self.chunks: list[str] = []
        self._queue: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        while True:
            try:
                data = self.process.read(4096)
            except Exception:
                return
            if data:
                self._queue.put(data)
            elif not self.process.isalive():
                return

    def write(self, text: str) -> None:
        self.process.write(text)

    def close(self) -> None:
        try:
            self.process.terminate(force=True)
        except Exception:
            return

    def pump(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                data = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            self.chunks.append(data)
            if "\x1b[c" in data:
                self.write("\x1b[?1;2c")
            if "\x1b[>c" in data or "\x1b[>0c" in data:
                self.write("\x1b[>0;115;0c")


class _PosixPty:
    def __init__(self, argv: list[str], cwd: Path, rows: int, cols: int, env_overrides: dict[str, str] | None = None) -> None:
        import pexpect

        env = os.environ.copy()
        if env_overrides:
            env.update(env_overrides)
        env.setdefault("TERM", "xterm-256color")
        env.setdefault("COLORTERM", "truecolor")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        self.process = pexpect.spawn(argv[0], argv[1:], cwd=str(cwd), env=env, dimensions=(rows, cols), encoding="utf-8")
        self.chunks: list[str] = []

    def write(self, text: str) -> None:
        self.process.send(text)

    def close(self) -> None:
        try:
            self.process.terminate(force=True)
        except Exception:
            return

    def pump(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                data = self.process.read_nonblocking(size=4096, timeout=0.05)
            except Exception:
                data = ""
            if data:
                self.chunks.append(data)
                if "\x1b[c" in data:
                    self.write("\x1b[?1;2c")
                if "\x1b[>c" in data or "\x1b[>0c" in data:
                    self.write("\x1b[>0;115;0c")


def _open_pty(argv: list[str], cwd: Path, rows: int, cols: int, env_overrides: dict[str, str] | None = None):
    if os.name == "nt":
        return _WindowsPty(argv, cwd, rows, cols, env_overrides)
    return _PosixPty(argv, cwd, rows, cols, env_overrides)


def _wait_for(pty, rows: int, cols: int, needle: str, *, timeout: float, label: str) -> str:
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        pty.pump(0.2)
        text = _render(pty.chunks, rows=rows, cols=cols)
        if needle in text:
            return text
    raise AssertionError(f"{label}: timed out waiting for {needle!r}\n{text}")


def _press(pty, key: str, count: int = 1) -> None:
    for _ in range(count):
        pty.write(key)
        pty.pump(0.18)


def _current_text(pty, rows: int, cols: int) -> str:
    pty.pump(0.2)
    return _render(pty.chunks, rows=rows, cols=cols)


def _open_root_action(pty, rows: int, cols: int, *, down_count: int, details_needle: str, label: str) -> None:
    _press(pty, DOWN, down_count)
    _wait_for(pty, rows, cols, details_needle, timeout=5.0, label=f"{label}-root-highlight")
    _press(pty, ENTER)


def _open_modal_choice(
    pty,
    rows: int,
    cols: int,
    *,
    down_count: int,
    highlighted: str,
    opened: str,
    close_key: str,
    label: str,
    parent_marker: str,
) -> None:
    _wait_for(pty, rows, cols, parent_marker, timeout=5.0, label=f"{label}-parent")
    _press(pty, DOWN, down_count)
    text = _wait_for(pty, rows, cols, highlighted, timeout=5.0, label=f"{label}-highlight")
    _assert_ordered_highlight(text, highlighted.removeprefix("> "), f"{label}-highlight")
    _press(pty, ENTER)
    _wait_for(pty, rows, cols, opened, timeout=6.0, label=f"{label}-open")
    _press(pty, close_key)
    _wait_for(pty, rows, cols, parent_marker, timeout=6.0, label=f"{label}-return")


def _probe_settings(pty, rows: int, cols: int) -> None:
    # Missing credentials start on Dashboard because Connect is locked. All Settings is
    # reached only with Down arrows, skipping credential-locked actions.
    _open_root_action(
        pty,
        rows,
        cols,
        down_count=13,
        details_needle="Open one settings hub",
        label="settings",
    )
    _wait_for(pty, rows, cols, "> 1. Connection", timeout=5.0, label="settings-open")

    _open_modal_choice(
        pty,
        rows,
        cols,
        down_count=0,
        highlighted="> 1. Connection",
        opened="Connection settings",
        close_key=ESCAPE,
        label="settings-connection",
        parent_marker="> 1. Connection",
    )
    _open_modal_choice(
        pty,
        rows,
        cols,
        down_count=1,
        highlighted="> 2. Strategy",
        opened="Model feature selection",
        close_key=ESCAPE,
        label="settings-strategy",
        parent_marker="> 1. Connection",
    )
    _open_modal_choice(
        pty,
        rows,
        cols,
        down_count=2,
        highlighted="> 3. Orders",
        opened="Order settings",
        close_key=ESCAPE,
        label="settings-orders",
        parent_marker="> 1. Connection",
    )
    _open_modal_choice(
        pty,
        rows,
        cols,
        down_count=3,
        highlighted="> 4. Compute",
        opened="Compute backend",
        close_key=ESCAPE,
        label="settings-compute",
        parent_marker="> 1. Connection",
    )

    _wait_for(pty, rows, cols, "> 1. Connection", timeout=5.0, label="settings-close-parent")
    _press(pty, DOWN, 4)
    _wait_for(pty, rows, cols, "> 5. Close", timeout=5.0, label="settings-close-highlight")
    _press(pty, ENTER)
    _wait_for(pty, rows, cols, "All settings complete (0)", timeout=6.0, label="settings-close")


def _probe_locked_credentials(pty, rows: int, cols: int) -> None:
    text = _current_text(pty, rows, cols)
    _assert_contains(text, "Connect  (locked)", "locked-credentials-connect")
    _assert_contains(text, "Account balances  (locked)", "locked-credentials-account")
    _assert_contains(text, "Trading caps  (locked)", "locked-credentials-caps")
    _assert_contains(text, "Testnet trading  (locked)", "locked-credentials-live")
    _assert_contains(text, "Test order  (locked)", "locked-credentials-roundtrip")
    _assert_not_contains(text, "Deposit USDC", "locked-credentials-no-deposit")
    _assert_not_contains(text, "Withdraw USDC", "locked-credentials-no-withdraw")


def _probe_root_navigation(pty, rows: int, cols: int) -> None:
    _wait_for(pty, rows, cols, "Show the current setup", timeout=5.0, label="root-dashboard-details")
    details_by_down = [
        "Verify safety flags",
        "Check candle quality",
        "Download fresh BTCUSDC",
        "Train or retrain",
        "Score the saved model",
        "Simulate trades",
        "Search risk",
        "Download data, train",
        "Run the live loop without placing",
        "Print dashboard",
        "Edit API keys",
        "Edit risk, thresholds",
        "Open one settings hub",
        "Show the plain-language workflow",
    ]
    for index, detail in enumerate(details_by_down, start=1):
        _press(pty, DOWN)
        _wait_for(pty, rows, cols, detail, timeout=5.0, label=f"root-down-{index}")
    _press(pty, UP)
    _wait_for(pty, rows, cols, "Open one settings hub", timeout=5.0, label="root-up-from-help")


def _probe_once(
    argv: list[str],
    cwd: Path,
    probe_name: str,
    *,
    rows: int,
    cols: int,
    env_overrides: dict[str, str] | None = None,
) -> str:
    pty = _open_pty(argv, cwd, rows, cols, env_overrides)
    try:
        text = _wait_for(pty, rows, cols, "Dashboard", timeout=8.0, label=f"{probe_name}-startup")
        _assert_contains(text, "Main menu", f"{probe_name}-startup")
        _assert_contains(text, "Connect", f"{probe_name}-startup")
        pty.pump(1.0)
        if probe_name == "root_navigation":
            _probe_root_navigation(pty, rows, cols)
        elif probe_name == "settings":
            _probe_settings(pty, rows, cols)
        elif probe_name == "locked_credentials":
            _probe_locked_credentials(pty, rows, cols)
        else:
            raise ValueError(f"unknown probe: {probe_name}")
        return _current_text(pty, rows, cols)
    finally:
        pty.write(ESCAPE)
        pty.write("q")
        time.sleep(0.2)
        pty.close()


def probe(argv: list[str], cwd: Path, *, rows: int = 36, cols: int = 120) -> None:
    with tempfile.TemporaryDirectory(prefix="simple-ai-probe-home-") as home:
        env = {"HOME": home, "USERPROFILE": home}
        _probe_once(argv, cwd, "root_navigation", rows=rows, cols=cols, env_overrides=env)
        _probe_once(argv, cwd, "settings", rows=rows, cols=cols, env_overrides=env)
        _probe_once(argv, cwd, "locked_credentials", rows=rows, cols=cols, env_overrides=env)


def _default_command(repo: Path) -> list[str]:
    if os.name == "nt":
        exe = repo / ".venv311" / "Scripts" / "simple-ai-trading.exe"
        if exe.exists():
            return [str(exe), "menu"]
    return [sys.executable, "-m", "simple_ai_bitcoin_trading_binance.cli", "menu"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe real terminal arrow-key navigation through the TUI root and Settings menus.")
    parser.add_argument("--cwd", default=".", help="Repository checkout to run from.")
    parser.add_argument("--command", nargs=argparse.REMAINDER, help="Command to run; defaults to the local CLI menu.")
    args = parser.parse_args(argv)

    cwd = Path(args.cwd).resolve()
    command = args.command or _default_command(cwd)
    try:
        probe(command, cwd)
    except Exception as exc:
        print(f"terminal navigation probe failed: {exc}", file=sys.stderr)
        return 1
    print("terminal navigation probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
