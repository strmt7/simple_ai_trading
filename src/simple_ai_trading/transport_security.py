"""Strict URL validation for local AI and bounded public transports."""

from __future__ import annotations

from urllib.parse import urlparse


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def validate_local_http_base_url(value: object, *, label: str = "local service") -> str:
    """Return a normalized, credential-free loopback HTTP base URL."""

    raw = str(value or "")
    base_url = raw.rstrip("/")
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} endpoint is invalid") from exc
    if (
        raw != raw.strip()
        or parsed.scheme != "http"
        or (parsed.hostname or "").lower() not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise ValueError(f"{label} requires a local HTTP endpoint")
    return base_url


__all__ = ["validate_local_http_base_url"]
