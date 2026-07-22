"""Shared command contract used by the CLI and Windows app."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
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
    value_arity: str
    repeatable: bool

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


@dataclass(frozen=True)
class WorkflowCommand:
    """One CLI command's single, deliberate location in the operator workflow."""

    page: str
    group: str
    name: str

    def asdict(self) -> dict[str, str]:
        return asdict(self)


# This is the only command taxonomy consumed by the native Windows app. Every
# CLI command must occur exactly once so additions cannot silently disappear
# into an alphabetized expert-only menu.
_WORKFLOW_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Trading",
        "Run and control",
        ("autonomous", "live", "coordinator"),
    ),
    (
        "Trading",
        "Connectivity and ownership",
        ("connect", "positions", "reconcile", "close"),
    ),
    (
        "Trading",
        "Execution diagnostics",
        ("spot-roundtrip", "polymarket-paper"),
    ),
    (
        "Research",
        "Polymarket evidence",
        (
            "polymarket-continuity",
            "polymarket-features",
            "polymarket-action-value",
        ),
    ),
    (
        "Research",
        "Polymarket models",
        (
            "polymarket-model",
            "polymarket-ridge",
            "polymarket-mlp",
            "polymarket-verify",
            "polymarket-publish",
        ),
    ),
    (
        "Research",
        "Polymarket confirmation",
        (
            "polymarket-round13-evaluate",
            "polymarket-round13-publish",
        ),
    ),
    (
        "Research",
        "AI validation",
        ("ai-benchmark", "ai-forecast-benchmark", "ai-review", "ai-uplift"),
    ),
    (
        "Research",
        "Microstructure models",
        (
            "model-blueprint",
            "impact-feature-source",
            "impact-corpus-index",
            "impact-grid-build",
            "microstructure-train",
            "microstructure-refit",
            "microstructure-prequential",
            "microstructure-promote",
            "microstructure-shadow",
        ),
    ),
    (
        "Research",
        "Tape and depth models",
        (
            "tape-depth-design",
            "tape-depth-study",
            "tape-depth-train",
            "tape-depth-prequential",
            "tape-depth-select",
            "tape-depth-confirm",
            "tape-depth-execution-confirm",
        ),
    ),
    (
        "Research",
        "Portfolio research",
        (
            "model-lab",
            "prepare",
            "train",
            "train-suite",
            "tune",
            "evaluate",
            "backtest",
            "backtest-panel",
            "backtest-chart",
            "objectives",
            "signals-benchmark",
        ),
    ),
    (
        "Risk",
        "Exposure and eligibility",
        ("risk", "universe"),
    ),
    (
        "Risk",
        "Evidence and reporting",
        ("audit", "report", "signals", "source-grades"),
    ),
    (
        "Data",
        "Market data",
        (
            "fetch",
            "data-sync",
            "archive-sync",
            "tick-archive-sync",
            "microstructure-capture",
            "impact-capture",
            "impact-corpus-collect",
            "polymarket-record",
        ),
    ),
    (
        "Data",
        "Integrity and outcomes",
        (
            "data-health",
            "tick-corpus-audit",
            "impact-audit",
            "impact-corpus-audit",
            "impact-grid-audit",
            "impact-corpus-day",
            "impact-corpus-batch-audit",
            "polymarket-resolve",
        ),
    ),
    (
        "System",
        "Runtime health",
        ("status", "doctor", "compute", "api-budget"),
    ),
    (
        "Settings",
        "Operator settings",
        ("configure", "strategy", "ai"),
    ),
    (
        "Settings",
        "Expert tools",
        ("menu", "shell"),
    ),
)


def _is_subparsers(action: argparse.Action) -> bool:
    return isinstance(action, argparse._SubParsersAction)  # noqa: SLF001 - argparse exposes no public marker


def _option_from_action(action: argparse.Action) -> CommandOption | None:
    if action.dest in {"help", "func"}:
        return None
    flags = tuple(action.option_strings)
    takes_value = action.nargs != 0
    value_arity = str(action.nargs if action.nargs is not None else 1)
    repeatable = isinstance(
        action,
        (
            argparse._AppendAction,  # noqa: SLF001
            argparse._AppendConstAction,  # noqa: SLF001
            argparse._CountAction,  # noqa: SLF001
            argparse._ExtendAction,  # noqa: SLF001
        ),
    )
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
        value_arity=value_arity,
        repeatable=repeatable,
    )


@lru_cache(maxsize=1)
def command_specs() -> tuple[CommandSpec, ...]:
    parser = _build_parser()
    specs: list[CommandSpec] = []
    subparsers = next(
        (action for action in parser._actions if _is_subparsers(action)), None
    )  # noqa: SLF001
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


@lru_cache(maxsize=1)
def workflow_commands() -> tuple[WorkflowCommand, ...]:
    """Return a complete one-to-one CLI/UI taxonomy, failing closed on drift."""

    items = tuple(
        WorkflowCommand(page=page, group=group, name=name)
        for page, group, names in _WORKFLOW_GROUPS
        for name in names
    )
    cli_names = command_names()
    configured_names = tuple(item.name for item in items)
    counts: dict[str, int] = {}
    for name in configured_names:
        counts[name] = counts.get(name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    missing = sorted(set(cli_names) - set(configured_names))
    stale = sorted(set(configured_names) - set(cli_names))
    if duplicates or missing or stale:
        raise RuntimeError(
            "CLI/Windows workflow contract drift: "
            f"duplicates={duplicates}, missing={missing}, stale={stale}"
        )
    return items


@lru_cache(maxsize=1)
def command_contract_digest() -> str:
    """Return the exact CLI/workflow contract fingerprint used by Windows."""

    payload = {
        "schema_version": "cli-windows-command-contract-v1",
        "commands": [spec.asdict() for spec in command_specs()],
        "workflow": [item.asdict() for item in workflow_commands()],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()
