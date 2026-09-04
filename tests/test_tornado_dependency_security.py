"""Socket-free regression checks for the optional Tornado security boundary.

Run with ``uv run --locked --with tornado==6.5.8 python -m pytest``. The base
installation intentionally need not install the microstructure server stack.
"""

from http.cookies import CookieError

import pytest


pytest.importorskip("tornado")
from tornado import httputil, web  # noqa: E402


@pytest.mark.parametrize(
    "body", [b"a=1&" * 1001, b"&" * 1001], ids=["named-fields", "separators"]
)
def test_urlencoded_field_count_is_bounded(body: bytes) -> None:
    with pytest.raises(httputil.HTTPInputError, match="Max number of fields"):
        httputil.parse_body_arguments("application/x-www-form-urlencoded", body, {}, {})


def test_urlencoded_ordinary_repeated_values_remain_valid() -> None:
    arguments: dict[str, list[bytes]] = {}
    files: dict = {}
    httputil.parse_body_arguments(
        "application/x-www-form-urlencoded", b"a=1&a=2&b=hello+world", arguments, files
    )
    assert arguments == {"a": [b"1", b"2"], "b": [b"hello world"]}
    assert files == {}


def test_multipart_split_is_bounded_before_allocating_parts() -> None:
    class BoundCheckedBytes(bytes):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            return type(self)(value) if isinstance(value, bytes) else value

        def split(self, sep=None, maxsplit=-1):
            # Verify the allocation boundary directly with a tiny body, without
            # creating an actual memory-pressure or denial-of-service payload.
            assert 0 <= maxsplit <= 4
            return super().split(sep, maxsplit)

    body = BoundCheckedBytes(b"--x\r\n" * 10 + b"--x--")
    with pytest.raises(httputil.HTTPInputError, match="too many parts"):
        httputil.parse_multipart_form_data(
            b"x", body, {}, {}, config=httputil.ParseMultipartConfig(max_parts=3)
        )


def test_ordinary_multipart_form_remains_valid() -> None:
    body = b'--x\r\nContent-Disposition: form-data; name="a"\r\n\r\none\r\n--x--\r\n'
    arguments: dict[str, list[bytes]] = {}
    httputil.parse_multipart_form_data(b"x", body, arguments, {})
    assert arguments == {"a": [b"one"]}


@pytest.mark.parametrize("key", ["Domain", "Path", "SameSite"])
@pytest.mark.parametrize("value", ["example.invalid; Secure", "example.invalid\r\nX:Y"])
def test_legacy_cookie_attribute_injection_is_rejected(key: str, value: str) -> None:
    handler = object.__new__(web.RequestHandler)
    with pytest.raises(CookieError):
        handler.set_cookie("sid", "value", **{key: value})


def test_normal_and_legacy_cookie_calls_preserve_valid_attributes() -> None:
    handler = object.__new__(web.RequestHandler)
    handler.set_cookie("normal", "one", domain="example.invalid", secure=True)
    with pytest.warns(DeprecationWarning):
        handler.set_cookie("legacy", "two", Domain="example.invalid")
    assert handler._new_cookie["normal"]["domain"] == "example.invalid"
    assert handler._new_cookie["normal"]["secure"] is True
    assert handler._new_cookie["legacy"]["domain"] == "example.invalid"
