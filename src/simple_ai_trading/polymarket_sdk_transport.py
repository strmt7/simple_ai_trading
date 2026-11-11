"""Owned HTTP transport for the pinned optional Polymarket CLOB SDK."""

from __future__ import annotations

import math
from time import monotonic
from typing import Mapping, Sequence

import httpx
from py_clob_client_v2.exceptions import PolyApiException


def _limits(timeout_seconds: float, maximum_response_bytes: int) -> float:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("SDK transport timeout must be finite and positive")
    if (
        isinstance(maximum_response_bytes, bool)
        or not isinstance(maximum_response_bytes, int)
        or maximum_response_bytes <= 0
    ):
        raise ValueError("SDK response byte limit must be a positive integer")
    return timeout


def _check_origin(origin: httpx.URL, endpoint: str) -> httpx.URL:
    url = httpx.URL(endpoint)
    if (
        (url.scheme, url.host, url.port) != (origin.scheme, origin.host, origin.port)
        or url.userinfo
        or url.fragment
    ):
        raise ValueError("SDK request differs from its configured origin")
    return url


class PolymarketSdkTransport:
    """Own a terminatable HTTP child; injected offline transports stay local."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        maximum_response_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._origin = httpx.URL(base_url)
        self.timeout_seconds = _limits(timeout_seconds, maximum_response_bytes)
        self.maximum_response_bytes = maximum_response_bytes
        if transport is None:
            from .polymarket_http_process import PolymarketHttpProcess

            self._client = PolymarketHttpProcess(
                base_url=base_url,
                timeout_seconds=self.timeout_seconds,
                maximum_response_bytes=maximum_response_bytes,
            )
        else:
            self._client = _InlineSdkTransport(
                base_url=base_url,
                timeout_seconds=self.timeout_seconds,
                maximum_response_bytes=maximum_response_bytes,
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
        _check_origin(self._origin, endpoint)
        return self._client.request(
            method, endpoint, headers=headers, data=data, params=params
        )

    def close(self) -> None:
        self._client.close()


class PolymarketPublicReadTransport:
    """Read an explicit endpoint set through one credential-free HTTP child."""

    def __init__(
        self,
        *,
        allowed_endpoints: Sequence[str],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> None:
        from .polymarket_http_process import PolymarketHttpProcess, _origin_key

        timeout = _limits(timeout_seconds, maximum_response_bytes)
        endpoints = tuple(httpx.URL(url) for url in allowed_endpoints)
        if not 1 <= len(endpoints) <= 32:
            raise ValueError("Public HTTP endpoint policy is invalid")
        origins = tuple(dict.fromkeys(_origin_key(str(url)) for url in endpoints))
        self._allowed = frozenset(str(url.copy_with(query=None)) for url in endpoints)
        self._client = PolymarketHttpProcess(
            base_url=origins[0],
            additional_origins=origins[1:],
            public_only=True,
            allowed_endpoints=tuple(sorted(self._allowed)),
            timeout_seconds=timeout,
            maximum_response_bytes=maximum_response_bytes,
        )

    def get_json(
        self,
        endpoint: str,
        *,
        params: Mapping[str, object] | None = None,
        deadline: float | None = None,
    ) -> object:
        url = httpx.URL(endpoint)
        if (
            url.userinfo
            or url.fragment
            or str(url.copy_with(query=None)) not in self._allowed
        ):
            raise ValueError("Public HTTP endpoint is not admitted")
        return self._client.request("GET", endpoint, params=params, deadline=deadline)

    def close(self) -> None:
        self._client.close()


class _InlineSdkTransport:
    """Run the native exchange once inside the owned child or an offline fixture."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        maximum_response_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._origin = httpx.URL(base_url)
        self.timeout_seconds = _limits(timeout_seconds, maximum_response_bytes)
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
        url = _check_origin(self._origin, endpoint)
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

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed
