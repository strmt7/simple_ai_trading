"""Shared command contract used by the CLI and Windows app."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

from .cli import _build_parser


@dataclass(frozen=True)
class CommandOption:
    flags: tuple[str, ...]
    dest: str
    required: bool
    default: object
    choices: tuple[str, ...]
    help: str
    takes_value: bool

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    options: tuple[CommandOption, ...]
    positionals: tuple[CommandOption, ...]

    def asdict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "help": self.help,
            "options": [option.asdict() for option in self.options],
            "positionals": [option.asdict() for option in self.positionals],
        }


def _is_subparsers(action: argparse.Action) -> bool:
    return isinstance(action, argparse._SubParsersAction)  # noqa: SLF001 - argparse exposes no public marker


def _option_from_action(action: argparse.Action) -> CommandOption | None:
    if action.dest in {"help", "func"}:
        return None
    flags = tuple(action.option_strings)
    takes_value = not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))  # noqa: SLF001
    default: Any = None if action.default is argparse.SUPPRESS else action.default
    choices = tuple(str(choice) for choice in (action.choices or ()))
    return CommandOption(
        flags=flags,
        dest=action.dest,
        required=bool(getattr(action, "required", False)),
        default=default,
        choices=choices,
        help=str(action.help or ""),
        takes_value=takes_value,
    )


def command_specs() -> tuple[CommandSpec, ...]:
    parser = _build_parser()
    specs: list[CommandSpec] = []
    subparsers = next((action for action in parser._actions if _is_subparsers(action)), None)  # noqa: SLF001
    if subparsers is None:
        return ()
    for name, subparser in sorted(subparsers.choices.items()):
        options: list[CommandOption] = []
        positionals: list[CommandOption] = []
        for action in subparser._actions:  # noqa: SLF001
            item = _option_from_action(action)
            if item is None:
                continue
            if item.flags:
                options.append(item)
            else:
                positionals.append(item)
        specs.append(
            CommandSpec(
                name=name,
                help=str(subparser.description or subparser.format_usage()).strip(),
                options=tuple(options),
                positionals=tuple(positionals),
            )
        )
    return tuple(specs)


def command_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in command_specs())
