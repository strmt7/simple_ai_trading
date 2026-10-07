"""Socket-free regression checks for the optional Tornado security boundary.

Run with ``uv run --locked --with tornado==6.5.10 python -m pytest``. The base
installation intentionally need not install the microstructure server stack.
The curl callback checks additionally require pycurl; they never open sockets.
"""

from http.cookies import CookieError
from io import BytesIO
from types import SimpleNamespace

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


@pytest.mark.parametrize("query", ["a=1&" * 1001, "&" * 1001], ids=["named", "empty"])
def test_query_field_count_is_bounded(query):
    with pytest.raises(httputil.HTTPInputError):
        httputil.HTTPServerRequest(uri="/?" + query)


def test_query_at_limit_preserves_repeated_and_blank_values():
    request = httputil.HTTPServerRequest(uri="/?" + "a=1&" * 998 + "a=2&b=")
    assert request.arguments == {"a": [b"1"] * 998 + [b"2"], "b": [b""]}


@pytest.mark.parametrize("escape", ["file", "directory", "index"])
def test_static_symlinks_cannot_escape_served_root(tmp_path, escape):
    root, outside = tmp_path / "static", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "data.txt").write_bytes(b"offline test content")
    (root / "index").mkdir()
    link = (
        root
        / {"file": "link.txt", "directory": "link", "index": "index/data.txt"}[escape]
    )
    target = outside if escape == "directory" else outside / "data.txt"
    try:
        link.symlink_to(target, target_is_directory=escape == "directory")
    except OSError as failure:
        pytest.skip(f"host cannot create test symlink: {failure.errno}")
    relative = {"file": "link.txt", "directory": "link/data.txt", "index": "index"}[
        escape
    ]
    handler = object.__new__(web.StaticFileHandler)
    handler.initialize(str(root), default_filename="data.txt")
    handler.path = relative
    handler.request = SimpleNamespace(path="/static/" + relative + "/")
    with pytest.raises(web.HTTPError) as denied:
        handler.validate_absolute_path(
            str(root), handler.get_absolute_path(str(root), relative)
        )
    assert denied.value.status_code == 403


def test_static_ordinary_file_remains_accessible(tmp_path):
    path = tmp_path / "ordinary.txt"
    path.write_bytes(b"offline ordinary content")
    handler = object.__new__(web.StaticFileHandler)
    handler.initialize(str(tmp_path))
    handler.path = path.name
    assert handler.validate_absolute_path(str(tmp_path), str(path)) == str(path)


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("decompress", [False, True], ids=["plain", "decoded"])
@pytest.mark.parametrize("overflow", [False, True], ids=["valid", "over-limit"])
def test_curl_body_boundary_before_buffering(streaming, decompress, overflow):
    pycurl = pytest.importorskip("pycurl")
    from tornado.curl_httpclient import CurlAsyncHTTPClient
    from tornado.httpclient import HTTPRequest

    class CurlOptions:
        def __init__(self):
            self.options, self.info = {}, {}

        def setopt(self, name, value):
            self.options[name] = value

        def unsetopt(self, name):
            self.options.pop(name, None)

    pending, received = [], []
    client = object.__new__(CurlAsyncHTTPClient)
    client.max_body_size = 64
    client.io_loop = SimpleNamespace(
        add_callback=lambda fn, *args: pending.append((fn, args))
    )
    curl, buffer = CurlOptions(), BytesIO()
    request = HTTPRequest(
        "https://example.invalid/never-requested",
        connect_timeout=1,
        request_timeout=1,
        validate_cert=True,
        decompress_response=decompress,
        streaming_callback=received.append if streaming else None,
    )
    client._curl_setup_request(curl, request, buffer, httputil.HTTPHeaders())
    write = curl.options[pycurl.WRITEFUNCTION]
    # Bytes are supplied at libcurl's decoded WRITEFUNCTION boundary. Never
    # create an actual decompression bomb, network connection or memory load.
    assert write(b"a" * 32) == 32
    assert write(b"b" * 32) == 32
    if overflow:
        assert write(b"x") == 0
    while pending:
        fn, args = pending.pop(0)
        fn(*args)
    data = b"".join(received) if streaming else buffer.getvalue()
    assert data == b"a" * 32 + b"b" * 32
