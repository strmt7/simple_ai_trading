"""One read-only Spot-testnet fee probe; secret input and responses stay in RAM."""

from datetime import datetime, timezone
from getpass import getpass
import hashlib
import hmac
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-05/testnet-readonly"
HOST = "https://testnet.binance.vision"


def main():
    # Exclusive intent is created before secret input and before either request.
    if (BASE / "result.json").exists():
        raise FileExistsError("probe already consumed")
    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "journal.jsonl").open("x", encoding="ascii") as journal:

        def record(**fields):
            row = {"utc": datetime.now(timezone.utc).isoformat(), **fields}
            journal.write(json.dumps(row, sort_keys=True) + "\n")
            journal.flush()
            os.fsync(journal.fileno())

        key = secret = signature = query = ""
        result = {"testnet_only": True, "orders": 0, "authenticated": False}
        try:
            key = getpass("Testnet key (hidden): ")
            secret = getpass("Testnet secret (hidden): ")
            if not key or not secret:
                raise ValueError("missing credential")
            with requests.Session() as session:
                session.trust_env = False
                record(phase="intent", method="GET", path="/api/v3/time", signed=False)
                clock = session.get(
                    HOST + "/api/v3/time", timeout=15, allow_redirects=False
                )
                record(
                    phase="completed",
                    path="/api/v3/time",
                    http_status=clock.status_code,
                )
                if clock.status_code != 200:
                    raise ValueError("public clock unavailable")
                timestamp = clock.json()["serverTime"]
                if type(timestamp) is not int or timestamp <= 0:
                    raise ValueError("invalid clock")
                query = urlencode(
                    {"symbol": "BTCUSDT", "timestamp": timestamp, "recvWindow": 5000}
                )
                signature = hmac.new(
                    secret.encode(), query.encode(), hashlib.sha256
                ).hexdigest()
                path = "/api/v3/account/commission"
                record(
                    phase="intent",
                    method="GET",
                    path=path,
                    signed=True,
                    symbol="BTCUSDT",
                )
                response = session.get(
                    HOST + path + "?" + query + "&signature=" + signature,
                    headers={"X-MBX-APIKEY": key},
                    timeout=15,
                    allow_redirects=False,
                )
                result["http_status"] = response.status_code
                payload = response.json()
                if (
                    response.status_code == 200
                    and isinstance(payload, dict)
                    and payload.get("symbol") == "BTCUSDT"
                ):
                    result["authenticated"] = True
                    result["symbol"] = "BTCUSDT"
                    # Do not persist arbitrary response strings, messages or identifiers.
                    from decimal import Decimal

                    fees = {}
                    for category in (
                        "standardCommission",
                        "specialCommission",
                        "taxCommission",
                    ):
                        rates = payload.get(category, {})
                        if not isinstance(rates, dict):
                            raise ValueError("invalid fee schema")
                        fees[category] = {}
                        for role in ("maker", "taker", "buyer", "seller"):
                            if role in rates:
                                value = Decimal(str(rates[role]))
                                if not value.is_finite() or not 0 <= value <= 1:
                                    raise ValueError("invalid fee")
                                fees[category][role] = str(value)
                    result["testnet_fee_fields_not_mainnet_evidence"] = fees
                elif isinstance(payload, dict) and type(payload.get("code")) is int:
                    result["exchange_error_code"] = payload["code"]
                record(phase="completed", path=path, **result)
        except Exception as exc:
            # Never print an exception message: transport errors can contain signed URLs.
            result["failure_type"] = type(exc).__name__
            record(phase="failed", **result)
        finally:
            key = secret = signature = query = ""
        with (BASE / "result.json").open("x", encoding="ascii") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
