"""Small filesystem persistence helpers."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_REPLACE_RETRY_DELAYS_SECONDS = (0.025, 0.05, 0.10, 0.20, 0.40, 0.80)


def _replace_with_transient_lock_retries(tmp_path: Path, target: Path) -> None:
    for attempt in range(len(_REPLACE_RETRY_DELAYS_SECONDS) + 1):
        try:
            os.replace(tmp_path, target)
            return
        except PermissionError:
            if attempt >= len(_REPLACE_RETRY_DELAYS_SECONDS):
                raise
            time.sleep(_REPLACE_RETRY_DELAYS_SECONDS[attempt])


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    mode: int | None = None,
) -> None:
    """Write JSON through a same-directory temporary file, then replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
            handle.write("\n")
        if mode is not None:
            os.chmod(tmp_path, mode)
        _replace_with_transient_lock_retries(tmp_path, target)
        if mode is not None:
            os.chmod(target, mode)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
