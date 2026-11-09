"""Owned HTTP transport for the pinned optional Polymarket CLOB SDK."""

from __future__ import annotations

import math
from time import monotonic
from typing import Mapping

import httpx
from py_clob_client_v2.exceptions import PolyApiException


class PolymarketSdkTransport:
    """Bound socket waits and observed response work without replaying a request.

    An observed deadline cannot interrupt a stalled native call. Independent
    process containment remains necessary for a hard wall-clock deadline.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        maximum_response_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._origin = httpx.URL(base_url)
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("SDK transport timeout must be finite and positive")
        if (
            isinstance(maximum_response_bytes, bool)
            or not isinstance(maximum_response_bytes, int)
            or maximum_response_bytes <= 0
        ):
            raise ValueError("SDK response byte limit must be a positive integer")
        self.maximum_response_bytes = maximum_response_bytes
        self._client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(self.timeout_seconds),
            follow_redirects=False,
            transport=transport,
        )

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: object = None,
        params: Mapping[str, object] | None = None,
    ) -> object:
        url = httpx.URL(endpoint)
        if (
            (url.scheme, url.host, url.port)
            != (self._origin.scheme, self._origin.host, self._origin.port)
            or url.userinfo
            or url.fragment
        ):
            raise ValueError("SDK request differs from its configured origin")
        request_headers = {
            "User-Agent": "py_clob_client_v2",
            "Accept": "*/*",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            **(headers or {}),
        }
        # Bound wire and decoded body size without accepting compression bombs.
        request_headers["Accept-Encoding"] = "identity"
        body = (
            {"content": data.encode("utf-8")}
            if isinstance(data, str)
            else {"json": data}
        )
        deadline = monotonic() + self.timeout_seconds
        try:
            with self._client.stream(
                method,
                url,
                headers=request_headers,
                params=params,
                timeout=self.timeout_seconds,
                **body,
            ) as response:
                self._check_deadline(deadline)
                if (
                    response.headers.get("content-encoding", "identity").lower()
                    != "identity"
                ):
                    raise PolyApiException(
                        error_msg="SDK response encoding is unsupported"
                    )
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    self._check_deadline(deadline)
                    if len(chunk) > self.maximum_response_bytes - len(chunks):
                        raise PolyApiException(
                            error_msg="SDK response exceeds byte limit"
                        )
                    chunks.extend(chunk)
                self._check_deadline(deadline)
                # The SDK's exception parser needs a fully retained bounded body.
                retained = httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=bytes(chunks),
                    request=response.request,
                )
            if retained.status_code != 200:
                error = PolyApiException(retained)
                self._check_deadline(deadline)
                raise error
            try:
                payload = retained.json()
            except ValueError:
                payload = retained.text
            self._check_deadline(deadline)
            return payload
        except httpx.RequestError:
            # Never expose signed URLs, headers or response bodies in diagnostics.
            raise PolyApiException(error_msg="SDK transport request failed") from None

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if monotonic() >= deadline:
            raise PolyApiException(error_msg="SDK response deadline exceeded")

    def close(self) -> None:
        self._client.close()
