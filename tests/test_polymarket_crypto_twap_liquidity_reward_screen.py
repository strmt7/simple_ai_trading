from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_polymarket_crypto_twap_liquidity_rewards",
    ROOT / "tools" / "screen_polymarket_crypto_twap_liquidity_rewards.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _Response:
    status_code = 200
    url = "https://example.invalid/final"
    content = b'{"data":[]}'

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return {"data": []}


class _Session:
    def request(self, *_args: object, **_kwargs: object) -> _Response:
        return _Response()


def _embedded_hash(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="ascii"))
    claimed = payload.pop("result_sha256")
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return claimed, hashlib.sha256(canonical).hexdigest()


def test_every_new_artifact_hash_reconstructs() -> None:
    paths = (
        ROOT
        / "docs/model-research/action-value/binance-usd1-simple-earn-activation-refresh-v1-2026-08-27.json",
        ROOT
        / "docs/model-research/polymarket/crypto-twap-liquidity-reward-screen-contract-v1.json",
        ROOT
        / "docs/model-research/polymarket/crypto-twap-liquidity-reward-screen-attempt1-failure-v1.json",
        ROOT
        / "docs/model-research/polymarket/crypto-twap-5m-liquidity-reward-screen-contract-v1-2026-08-27.json",
        ROOT
        / "docs/model-research/polymarket/crypto-twap-5m-liquidity-reward-screen-attempt1-terminal-v1-2026-08-27.json",
    )
    for path in paths:
        assert _embedded_hash(path)[0] == _embedded_hash(path)[1]


def test_public_response_is_journaled_before_validation(tmp_path: Path) -> None:
    decoded, source = TOOL._request(
        _Session(),
        "GET",
        "https://example.invalid/source",
        journal_dir=tmp_path,
        source_name="01-source",
    )

    assert decoded == {"data": []}
    assert (tmp_path / "01-source.raw").read_bytes() == _Response.content
    intent = json.loads((tmp_path / "01-source.intent.json").read_text())
    response = json.loads((tmp_path / "01-source.response.json").read_text())
    assert intent["method"] == "GET"
    assert response["payload_sha256"] == hashlib.sha256(_Response.content).hexdigest()
    assert source["payload_sha256"] == response["payload_sha256"]


def test_five_minute_terminal_receipt_stopped_before_books() -> None:
    receipt = json.loads(
        (
            ROOT
            / "docs/model-research/polymarket/crypto-twap-5m-liquidity-reward-screen-attempt1-terminal-v1-2026-08-27.json"
        ).read_text(encoding="ascii")
    )

    assert receipt["failure"]["books_requested"] is False
    assert receipt["observed_market_configuration"]["exact_market_count"] == 7
    assert receipt["observed_market_configuration"]["raw_clobRewards_field_omitted_count"] == 7
    assert receipt["terminal_decision"]["accepted_edge"] is False
    assert receipt["terminal_decision"]["public_after_cost_floor_pUSD"] == "0"
