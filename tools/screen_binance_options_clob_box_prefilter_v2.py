from __future__ import annotations

import json

from tools import screen_binance_options_clob_box_prefilter as v1


ROOT = v1.ROOT
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-clob-box-retained-prefilter-contract-v2.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "binance-options-clob-box-retained-prefilter-v2-2026-08-29.json"
)


def _quote(
    ticker: dict[str, object], *, capture_completed_at_ms: int
) -> dict[str, object]:
    bid = v1._decimal(ticker["bidPrice"])
    ask = v1._decimal(ticker["askPrice"])
    raw_close_time = ticker.get("closeTime")
    if raw_close_time is None:
        return {
            "bid": bid,
            "ask": ask,
            "close_time_ms": None,
            "age_ms": -1,
        }
    close_time_ms = int(str(raw_close_time))
    return {
        "bid": bid,
        "ask": ask,
        "close_time_ms": close_time_ms,
        "age_ms": capture_completed_at_ms - close_time_ms,
    }


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    dependency = contract["implementation"]["dependency"]
    dependency_path = ROOT / dependency["path"]
    if v1._sha256(dependency_path.read_bytes()) != dependency["sha256"]:
        raise RuntimeError("v1 dependency hash mismatch")
    v1.CONTRACT_PATH = CONTRACT_PATH
    v1.RESULT_PATH = RESULT_PATH
    v1._quote = _quote
    v1.main()


if __name__ == "__main__":
    main()
