from __future__ import annotations

import pytest

from simple_ai_trading.transport_security import validate_local_http_base_url


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_local_http_base_url_accepts_only_unambiguous_loopback(
    value: str, expected: str
) -> None:
    assert validate_local_http_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "https://localhost:11434",
        "http://127.0.0.2:11434",
        "http://example.com:11434",
        "http://user:secret@localhost:11434",  # pragma: allowlist secret
        "http://localhost:11434/api",
        "http://localhost:11434?query=yes",
        " http://localhost:11434",
        "http://localhost:70000",
    ],
)
def test_local_http_base_url_rejects_remote_or_ambiguous_values(value: str) -> None:
    with pytest.raises(ValueError, match="local HTTP endpoint|endpoint is invalid"):
        validate_local_http_base_url(value)
