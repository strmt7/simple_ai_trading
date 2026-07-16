#!/usr/bin/env python3
"""Generate the native Win32 command contract header from the Python CLI."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _escape_wide(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r", " ").replace("\n", " ")
    return f'L"{escaped}"'


def _safe_identifier(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "command"


def _join_values(values: Iterable[object]) -> str:
    return ", ".join(str(value) for value in values if str(value))


def _default_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "src"))
    from simple_ai_trading.command_contract import (  # noqa: PLC0415
        command_specs,
        workflow_commands,
    )

    out = root / "native" / "windows" / "generated" / "command_contract.hpp"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#pragma once",
        "",
        "namespace simple_ai_trading::native_contract {",
        "",
        "struct CommandOptionSpec {",
        "    const wchar_t* flags;",
        "    const wchar_t* dest;",
        "    const wchar_t* choices;",
        "    const wchar_t* default_value;",
        "    const wchar_t* help;",
        "    const wchar_t* value_arity;",
        "    bool required;",
        "    bool takes_value;",
        "    bool repeatable;",
        "};",
        "",
        "struct CommandSpec {",
        "    const wchar_t* name;",
        "    const wchar_t* help;",
        "    const CommandOptionSpec* options;",
        "    int option_count;",
        "};",
        "",
        "struct WorkflowCommandSpec {",
        "    const wchar_t* page;",
        "    const wchar_t* group;",
        "    const wchar_t* command;",
        "};",
        "",
    ]
    specs = command_specs()
    for spec in specs:
        options = (*spec.options, *spec.positionals)
        if not options:
            continue
        array_name = f"kOptions_{_safe_identifier(spec.name)}"
        lines.append(f"inline constexpr CommandOptionSpec {array_name}[] = {{")
        for option in options:
            flags = _join_values(option.flags) or option.dest
            lines.append(
                "    {"
                f"{_escape_wide(flags)}, "
                f"{_escape_wide(option.dest)}, "
                f"{_escape_wide(_join_values(option.choices))}, "
                f"{_escape_wide(_default_string(option.default))}, "
                f"{_escape_wide(option.help)}, "
                f"{_escape_wide(option.value_arity)}, "
                f"{str(bool(option.required)).lower()}, "
                f"{str(bool(option.takes_value)).lower()}, "
                f"{str(bool(option.repeatable)).lower()}"
                "},"
            )
        lines.append("};")
        lines.append("")
    lines.append("inline constexpr CommandSpec kCommands[] = {")
    for spec in specs:
        option_count = len(spec.options) + len(spec.positionals)
        array_name = f"kOptions_{_safe_identifier(spec.name)}" if option_count else "nullptr"
        lines.append(f"    {{{_escape_wide(spec.name)}, {_escape_wide(spec.help)}, {array_name}, {option_count}}},")
    lines.extend([
        "};",
        "inline constexpr int kCommandCount = static_cast<int>(sizeof(kCommands) / sizeof(kCommands[0]));",
        "",
        "inline constexpr WorkflowCommandSpec kWorkflowCommands[] = {",
    ])
    for item in workflow_commands():
        lines.append(
            "    {"
            f"{_escape_wide(item.page)}, "
            f"{_escape_wide(item.group)}, "
            f"{_escape_wide(item.name)}"
            "},"
        )
    lines.extend([
        "};",
        "inline constexpr int kWorkflowCommandCount = "
        "static_cast<int>(sizeof(kWorkflowCommands) / sizeof(kWorkflowCommands[0]));",
        "",
        "} // namespace simple_ai_trading::native_contract",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
