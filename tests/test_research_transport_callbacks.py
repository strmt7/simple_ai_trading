"""Protocol contracts for retained research redirect and parser callbacks."""

from __future__ import annotations

from email.message import Message
import importlib
import inspect
import io
from urllib.error import HTTPError
from urllib.request import (
    BaseHandler,
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)
from urllib.response import addinfourl

import pytest


REDIRECT_MODULES = (
    "tools.capture_binance_triangle_window",
    "tools.capture_public_source_bounded",
    "tools.screen_paradex_index_boundaries",
    "tools.screen_polymarket_exact_negrisk_long_only_frontier_v2",
)


@pytest.mark.parametrize("module_name", REDIRECT_MODULES)
@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirect_callback_preserves_signature_and_denies_following(
    module_name: str, code: int
) -> None:
    handler_type = importlib.import_module(module_name).NoRedirect
    assert inspect.signature(handler_type.redirect_request) == inspect.signature(
        HTTPRedirectHandler.redirect_request
    )
    handler = handler_type()
    request = Request("https://example.invalid/original")
    headers = Message()
    headers["Location"] = "https://example.invalid/redirected"
    with io.BytesIO(b"redirect") as body:
        assert (
            handler.redirect_request(
                req=request,
                fp=body,
                code=code,
                msg="redirect",
                headers=headers,
                newurl=headers["Location"],
            )
            is None
        )

    calls: list[str] = []

    class RetainedRedirect(BaseHandler):
        handler_order = 100

        def https_open(self, req):
            calls.append(req.full_url)
            assert calls == [request.full_url], "redirect must never be requested"
            response = addinfourl(io.BytesIO(b"redirect"), headers, req.full_url, code)
            response.msg = "redirect"
            return response

    # The handler returns None; urllib's complete chain turns that into HTTPError.
    opener = build_opener(ProxyHandler({}), handler, RetainedRedirect())
    with pytest.raises(HTTPError) as raised:
        opener.open(request)
    try:
        assert raised.value.code == code
        assert raised.value.url == request.full_url
        assert calls == [request.full_url]
    finally:
        raised.value.close()


def test_visible_text_ignores_attributes_and_excludes_script_style() -> None:
    from tools.extract_paradex_funding_sources import VisibleText

    parser = VisibleText()
    parser.feed(
        '<h1 data-private="discarded">Heading</h1>'
        '<script data-private="discarded">hidden script</script>'
        '<style data-private="discarded">hidden style</style>'
        '<p data-private="discarded">Visible text</p>'
    )
    parser.close()
    assert parser.parts == ["Heading", "Visible text"]
    assert parser.skip == 0
