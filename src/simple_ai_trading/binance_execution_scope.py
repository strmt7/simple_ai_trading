"""Non-secret binding to a venue product and API-key identity, not an account UID."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

BINANCE_SPOT_TESTNET = "https://testnet.binance.vision"
BINANCE_SPOT_DEMO = "https://demo-api.binance.com"
BINANCE_FUTURES_TESTNET = "https://testnet.binancefuture.com"
BINANCE_FUTURES_DEMO = "https://demo-fapi.binance.com"


@dataclass(frozen=True)
class BinanceExecutionScope:
    origin: str
    market_type: str
    credential_fingerprint: str

    def __post_init__(self) -> None:
        origins = {
            "spot": {BINANCE_SPOT_TESTNET, BINANCE_SPOT_DEMO},
            "futures": {BINANCE_FUTURES_TESTNET, BINANCE_FUTURES_DEMO},
        }
        if (
            self.market_type not in origins
            or self.origin not in origins[self.market_type]
            or not isinstance(self.credential_fingerprint, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.credential_fingerprint) is None
        ):
            raise ValueError("invalid Binance execution scope")

    @classmethod
    def from_api_key(
        cls, origin: str, market_type: str, api_key: str
    ) -> BinanceExecutionScope:
        """Bind the key identity without storing credentials; rotations do not match."""
        if not isinstance(api_key, str) or not api_key or api_key != api_key.strip():
            raise ValueError("Binance execution scope requires a key identity")
        fingerprint = hashlib.sha256(
            b"simple-ai-trading:binance-key-scope:v1\0" + api_key.encode("utf-8")
        ).hexdigest()
        return cls(origin, market_type, fingerprint)
