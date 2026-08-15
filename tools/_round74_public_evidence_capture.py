"""Shared bounded transport for Round 74 public evidence tools."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPOSITORY = Path(__file__).resolve().parents[1]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 74 public response contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"Round 74 public response contains {value}")


def strict_json_loads(body: bytes) -> object:
    try:
        return json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 public response is not strict JSON") from exc


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("Round 74 public evidence git identity differs")
    return commit


def require_clean_tracked_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise ValueError("Round 74 public evidence requires a clean tracked worktree")


def _safe_header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    selected = str(value).strip()
    if not selected or len(selected) > 256:
        raise ValueError("Round 74 public response header differs")
    return selected


@dataclass(frozen=True)
class BoundedJsonResponse:
    url: str
    body: bytes
    payload: object
    request_started_wall_ns: int
    request_started_monotonic_ns: int
    received_wall_ns: int
    received_monotonic_ns: int
    headers: tuple[tuple[str, str | None], ...]

    @property
    def elapsed_monotonic_ns(self) -> int:
        return self.received_monotonic_ns - self.request_started_monotonic_ns

    def header_mapping(self) -> dict[str, str | None]:
        return dict(self.headers)


def bounded_json_get(
    *,
    url: str,
    timeout_seconds: float,
    maximum_response_bytes: int,
    user_agent: str,
) -> BoundedJsonResponse:
    timeout = float(timeout_seconds)
    maximum = int(maximum_response_bytes)
    parsed_url = urlparse(url)
    try:
        port = parsed_url.port
    except ValueError as exc:
        raise ValueError("Round 74 public request URL is invalid") from exc
    if (
        url != url.strip()
        or parsed_url.scheme != "https"
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.fragment
        or (port is not None and not 1 <= port <= 65_535)
        or not 1.0 <= timeout <= 60.0
        or maximum <= 0
        or maximum > 64 * 1024 * 1024
        or not str(user_agent).strip()
    ):
        raise ValueError("Round 74 public request contract differs")
    request_started_wall_ns = time.time_ns()
    request_started_monotonic_ns = time.monotonic_ns()
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    try:
        with urlopen(  # nosec B310 - URL is validated as credential-free HTTPS.
            request, timeout=timeout
        ) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("Round 74 public response identity differs")
            content_type = response.headers.get_content_type()
            if content_type != "application/json":
                raise ValueError("Round 74 public content type differs")
            body = response.read(maximum + 1)
            if not body or len(body) > maximum:
                raise ValueError("Round 74 public response size differs")
            headers = (
                ("content_type", content_type),
                ("date", _safe_header(response.headers, "Date")),
                (
                    "x_mbx_used_weight_1m",
                    _safe_header(
                        response.headers,
                        "X-MBX-USED-WEIGHT-1M",
                    ),
                ),
                (
                    "x_mbx_used_weight",
                    _safe_header(response.headers, "X-MBX-USED-WEIGHT"),
                ),
            )
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError("Round 74 public request failed without retry") from exc
    received_monotonic_ns = time.monotonic_ns()
    received_wall_ns = time.time_ns()
    return BoundedJsonResponse(
        url=url,
        body=body,
        payload=strict_json_loads(body),
        request_started_wall_ns=request_started_wall_ns,
        request_started_monotonic_ns=request_started_monotonic_ns,
        received_wall_ns=received_wall_ns,
        received_monotonic_ns=received_monotonic_ns,
        headers=headers,
    )


def write_artifact(path: Path, artifact: Mapping[str, object]) -> None:
    selected = path.resolve()
    try:
        selected.relative_to(REPOSITORY)
    except ValueError as exc:
        raise ValueError(
            "Round 74 public evidence output must remain inside the repository"
        ) from exc
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(f".{selected.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=False,
        )
        + "\n"
    ).encode("ascii")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BoundedJsonResponse",
    "REPOSITORY",
    "bounded_json_get",
    "canonical_json_bytes",
    "canonical_sha256",
    "git_commit",
    "require_clean_tracked_worktree",
    "strict_json_loads",
    "write_artifact",
]
