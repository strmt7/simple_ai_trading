"""Own the live supervisor's public BTC market discovery process."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess  # nosec B404 - fixed interpreter and module
import sys
from tempfile import TemporaryFile

import requests

from .polymarket import (
    GAMMA_MARKETS_URL,
    PolymarketFiveMinuteMarket,
    PolymarketPublicClient,
    parse_polymarket_fifteen_minute_market,
    parse_polymarket_five_minute_market,
)
from .windows_owned_job import CREATE_SUSPENDED, WindowsOwnedJob


_MAX_REPLY_BYTES = 1024 * 1024
_POLL_SECONDS = 0.05
_REAP_SECONDS = 2
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP = (
    "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
    "runpy.run_module('simple_ai_trading.polymarket_market_discovery',run_name='__main__')"
)


def _horizon(variant: str) -> int:
    if variant == "fiveminute":
        return 5
    if variant == "fifteenminute":
        return 15
    raise ValueError("unsupported Polymarket discovery horizon")


def _decode_reply(
    payload: bytes, *, variant: str, observed_at_ms: int
) -> tuple[PolymarketFiveMinuteMarket, ...]:
    horizon = _horizon(variant)
    if len(payload) > _MAX_REPLY_BYTES:
        raise ValueError("Polymarket discovery reply exceeded its byte limit")
    rows = json.loads(payload)
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 2
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("Polymarket discovery reply market count differs")
    parser = (
        parse_polymarket_five_minute_market
        if horizon == 5
        else parse_polymarket_fifteen_minute_market
    )
    epoch_seconds = horizon * 60
    start_epoch = observed_at_ms // (epoch_seconds * 1_000) * epoch_seconds
    slugs = {
        f"btc-updown-{horizon}m-{start_epoch}",
        f"btc-updown-{horizon}m-{start_epoch + epoch_seconds}",
    }
    markets = tuple(parser(row) for row in rows)
    if (
        len({market.condition_id for market in markets}) != len(markets)
        or len({market.slug for market in markets}) != len(markets)
        or any(market.slug not in slugs or market.asset != "BTC" for market in markets)
    ):
        raise ValueError(
            "Polymarket discovery reply is duplicate or outside the request"
        )
    return markets


class PolymarketOwnedMarketDiscovery:
    """Bound one discovery call and reap its child before returning to Stop."""

    def __init__(self, *, timeout_seconds: float = 12.0) -> None:
        if not 0.1 <= timeout_seconds <= 30:
            raise ValueError("discovery timeout must lie in [0.1, 30] seconds")
        self._timeout = timeout_seconds
        self._cleanup_failed = False

    def _command(self, variant: str, observed_at_ms: int) -> list[str]:
        return [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _BOOTSTRAP,
            str(_PACKAGE_ROOT),
            variant,
            str(observed_at_ms),
        ]

    async def discover(
        self,
        *,
        market_variant: str,
        observed_at_ms: int,
        stop: asyncio.Event,
    ) -> tuple[PolymarketFiveMinuteMarket, ...] | None:
        _horizon(market_variant)
        if type(observed_at_ms) is not int or observed_at_ms <= 0:
            raise ValueError("Polymarket discovery observation time is invalid")
        if self._cleanup_failed:
            raise RuntimeError("previous Polymarket discovery child cleanup failed")
        if stop.is_set():
            return None
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR"}
        }
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        with TemporaryFile() as output:
            job = WindowsOwnedJob() if os.name == "nt" else None
            process: subprocess.Popen[bytes] | None = None
            try:
                flags = (subprocess.CREATE_NO_WINDOW | CREATE_SUSPENDED) if job else 0
                process = subprocess.Popen(  # nosec B603
                    self._command(market_variant, observed_at_ms),
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    cwd=_PACKAGE_ROOT,
                    creationflags=flags,
                    close_fds=True,
                )
                if job is not None:
                    job.enroll_and_resume(process)
                while process.poll() is None and not stop.is_set():
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Polymarket market discovery deadline expired"
                        )
                    try:
                        await asyncio.wait_for(
                            stop.wait(), timeout=min(remaining, _POLL_SECONDS)
                        )
                    except TimeoutError:
                        continue
                if stop.is_set():
                    return None
                if loop.time() >= deadline:
                    raise TimeoutError("Polymarket market discovery deadline expired")
                if process.returncode != 0:
                    raise RuntimeError("Polymarket public discovery child failed")
                output.seek(0)
                return _decode_reply(
                    output.read(_MAX_REPLY_BYTES + 1),
                    variant=market_variant,
                    observed_at_ms=observed_at_ms,
                )
            finally:
                try:
                    try:
                        if job is not None:
                            job.close()
                    finally:
                        if process is not None:
                            if process.poll() is None:
                                process.kill()
                            process.wait(timeout=_REAP_SECONDS)
                except (OSError, subprocess.TimeoutExpired):
                    self._cleanup_failed = True
                    raise


class _DiscoveryHttpClient(PolymarketPublicClient):
    def _get_json(self, url: str, *, params=None) -> object:
        if url != GAMMA_MARKETS_URL:
            raise ValueError("discovery child can only request Gamma markets")
        with self.session.get(
            url,
            params=params,
            timeout=self.timeout_seconds,
            stream=True,
            allow_redirects=False,
            headers={"Accept-Encoding": "identity"},
        ) as response:
            if response.status_code != 200:
                raise ValueError("Gamma market discovery did not return HTTP 200")
            payload = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if len(payload) + len(chunk) > _MAX_REPLY_BYTES:
                    raise ValueError("Gamma market response exceeded its byte limit")
                payload.extend(chunk)
            return json.loads(payload)


def _worker_main(variant: str, observed_at_ms: int) -> None:
    horizon = _horizon(variant)
    with requests.Session() as session:
        session.trust_env = False
        client = _DiscoveryHttpClient(session=session, timeout_seconds=8)
        if horizon == 5:
            markets = client.discover_five_minute_markets(
                now_ms=observed_at_ms, assets=("BTC",)
            )
        else:
            markets = client.discover_fifteen_minute_markets(now_ms=observed_at_ms)
        payload = (
            b"["
            + b",".join(m.gamma_payload_json.encode("ascii") for m in markets)
            + b"]"
        )
        if len(payload) > _MAX_REPLY_BYTES:
            raise ValueError("Gamma market reply exceeded its byte limit")
        sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    _worker_main(sys.argv[1], int(sys.argv[2]))
