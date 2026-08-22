"""Process-wide executable search-path hardening."""

from __future__ import annotations

from collections.abc import MutableMapping
import os
from pathlib import Path


def _canonical_path(raw: str) -> Path | None:
    text = raw.strip().strip('"')
    if not text:
        return None
    expanded = os.path.expandvars(os.path.expanduser(text))
    candidate = Path(expanded)
    if not candidate.is_absolute():
        return None
    return Path(os.path.normpath(candidate))


def harden_executable_search_path(
    environment: MutableMapping[str, str] | None = None,
    *,
    current_directory: str | Path | None = None,
) -> tuple[str, ...]:
    """Remove implicit current-directory executable lookup from child processes."""

    target = os.environ if environment is None else environment
    cwd = Path.cwd() if current_directory is None else Path(current_directory)
    cwd = Path(os.path.normpath(cwd.absolute()))
    raw_path = target.get("PATH", os.defpath)
    safe_entries: list[str] = []
    seen: set[str] = set()

    for raw_entry in raw_path.split(os.pathsep):
        candidate = _canonical_path(raw_entry)
        if candidate is None or candidate == cwd:
            continue
        normalized = os.path.normcase(str(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        safe_entries.append(str(candidate))

    target["PATH"] = os.pathsep.join(safe_entries)
    target["NoDefaultCurrentDirectoryInExePath"] = "1"
    return tuple(safe_entries)
