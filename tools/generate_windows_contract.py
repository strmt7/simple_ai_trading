#!/usr/bin/env python3
"""Generate the native Win32 command contract header from the Python CLI."""

from __future__ import annotations

from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _escape_wide(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r", " ").replace("\n", " ")
    return f'L"{escaped}"'


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    from simple_ai_trading.command_contract import command_specs  # noqa: PLC0415

    out = root / "native" / "windows" / "generated" / "command_contract.hpp"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#pragma once",
        "",
        "namespace simple_ai_trading::native_contract {",
        "",
        "struct CommandSpec {",
        "    const wchar_t* name;",
        "    const wchar_t* help;",
        "    int option_count;",
        "};",
        "",
        "inline constexpr CommandSpec kCommands[] = {",
    ]
    for spec in command_specs():
        option_count = len(spec.options) + len(spec.positionals)
        lines.append(f"    {{{_escape_wide(spec.name)}, {_escape_wide(spec.help)}, {option_count}}},")
    lines.extend([
        "};",
        "inline constexpr int kCommandCount = static_cast<int>(sizeof(kCommands) / sizeof(kCommands[0]));",
        "",
        "} // namespace simple_ai_trading::native_contract",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
