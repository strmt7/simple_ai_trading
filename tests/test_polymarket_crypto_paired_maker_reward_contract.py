from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "crypto-paired-maker-reward-screen-contract-v1.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def test_crypto_paired_maker_reward_screen_contract_is_hash_bound_and_fail_closed() -> (
    None
):
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    expected_hash = contract.pop("result_sha256")

    assert hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest() == (
        expected_hash
    )
    assert contract["capture"]["assets"] == ["BTC", "ETH", "SOL"]
    assert contract["capture"]["one_attempt_only"] is True
    assert (
        contract["capture"]["use_next_epoch_or_replacement_market_permitted"] is False
    )
    assert contract["request_contract"]["maximum_public_requests"] == 5
    assert contract["request_contract"]["retry_permitted"] is False
    assert (
        contract["quote_rule"]["include_complementary_own_asks_in_post_quote_midpoints"]
        is True
    )
    assert contract["quote_rule"]["physical_order_scored_once"] is True
    assert contract["economics"]["publicly_proven_reward_payout_lower_bound"] == "0"
    assert contract["authority"]["screen_may_accept_an_edge"] is False
    source = contract["frozen_source"]
    assert (
        hashlib.sha256(
            _git_blob(source["arithmetic_commit"], source["arithmetic_module_path"])
        ).hexdigest()
        == source["arithmetic_module_sha256"]
    )
