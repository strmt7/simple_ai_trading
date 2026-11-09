"""Portable locations for preserved historical research, separate from live state."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


HISTORICAL_ROOT_ENV = "SIMPLE_AI_TRADING_HISTORICAL_DATA_ROOT"


def historical_research_path(repository_root: Path, relative: str | Path) -> Path:
    """Resolve a contained historical input without assuming a host drive letter."""
    selected = os.environ.get(HISTORICAL_ROOT_ENV)
    if selected is not None:
        if not selected.strip():
            raise ValueError("Historical research root override is empty")
        root = Path(selected).expanduser()
        if not root.is_absolute():
            raise ValueError("Historical research root override must be absolute")
    else:
        root = repository_root / "data" / "historical-research"
    root = root.resolve()
    suffix = PureWindowsPath(str(relative))
    if (
        suffix.drive
        or suffix.root
        or not suffix.parts
        or any(p == ".." or ":" in p for p in suffix.parts)
    ):
        raise ValueError("Historical research input must be a contained relative path")
    result = root.joinpath(*suffix.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Historical research input escapes its configured root")
    return result
