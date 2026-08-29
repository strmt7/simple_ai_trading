from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs" / "model-research" / "action-value"
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
JOURNAL_PATH = (
    ACTION_VALUE / "binance-cross-stablecoin-funding-recovery-journal-v4.json"
)
RECOVERY_PATH = (
    ACTION_VALUE / "binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json"
)
FAILURE_PATH = (
    ACTION_VALUE / "binance-cross-stablecoin-funding-full-history-v3-2026-08-25.json"
)
EXPECTED_RESULT_HASHES = {
    "binance-cross-stablecoin-funding-screen-contract-v1.json": (
        "d100faabf96afbd15478dcaa02688a365b46ba1c15279bcb575cdb8a8661a5a7"
    ),
    "binance-cross-stablecoin-funding-snapshot-v1-2026-08-25.json": (
        "1588aa33751abd73430f6c3c80525e8cbd887877fc4ba57276f5f0710477df30"
    ),
    "binance-cross-stablecoin-funding-adjudication-contract-v2.json": (
        "c6906950fe3d6513d02c90566c454aaa461e17d39ea36010d222d265fdb666c7"
    ),
    "binance-cross-stablecoin-funding-adjudication-v2-2026-08-25.json": (
        "7f2cfab714e0a4353ce143b735a13e4cdd6be5340fc13a09e5a82958448f942b"
    ),
    "binance-cross-stablecoin-funding-full-history-contract-v3.json": (
        "41d512a4c5c8596bcfbbbfd1dc2e22dceed12c50564673054fa5b9094eebd2fc"
    ),
    "binance-cross-stablecoin-funding-full-history-v3-2026-08-25.json": (
        "16069c43e49dfdcafb03306e62734646148c1ac784b6ae13c9fc2dd2508b7670"
    ),
    "binance-cross-stablecoin-funding-recovery-contract-v4.json": (
        "94b61346768a77f8b2d1a8ae0224afbb358b4881f559f1ca3c439e17eb86edd1"
    ),
    "binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json": (
        "8e30be61daaecabd3546e41cdc204d20b8ad38e0fc80c3c9aa96092266a3abe5"
    ),
}
EXPECTED_JOURNAL_HASH = (
    "b85c428892d038e7cab76f5577e468aed7907d8b37a7dfb310d9ec30fab62ad9"
)
EXPECTED_JOURNAL_FILE_HASH = (
    "2686ccc31249b03ba2c12279282324fdbb5be3589047ca4510e61654efa7d49b"
)
EXPECTED_REGISTRY_HASH = (
    "fc0bddf222a1908db6c12df338dc26963f36514b01e37b5b31fc567760f19aca"
)
EXPECTED_TOOL_HASHES = {
    "screen_binance_cross_stablecoin_funding.py": (
        "2133d8770778127b0c76b535b73d8d3249e8954e14ed6dc7a7c025adc79a1088"
    ),
    "adjudicate_binance_cross_stablecoin_funding.py": (
        "f98e28a1cc5b76c1a08c286342ddd97496c8cc225527591c9c68ec313984af44"
    ),
    "backfill_binance_cross_stablecoin_funding.py": (
        "18733b0faa166af79eb3e5ecf9ccfca874ed7f0fdfee440212a2dd18f0c92f81"
    ),
    "recover_binance_cross_stablecoin_funding.py": (
        "8290c472fc2a4f16c0f3fc2f00fa72fb7cc00979394307d7408dd1e27daf2cd1"
    ),
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _embedded_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def test_every_published_contract_and_result_reconstructs() -> None:
    for name, expected_hash in EXPECTED_RESULT_HASHES.items():
        payload = _load(ACTION_VALUE / name)
        assert payload["result_sha256"] == expected_hash
        assert _embedded_hash(payload, "result_sha256") == expected_hash

    registry = _load(REGISTRY_PATH)
    assert registry["result_sha256"] == EXPECTED_REGISTRY_HASH
    assert _embedded_hash(registry, "result_sha256") == EXPECTED_REGISTRY_HASH


def test_recovery_journal_is_complete_unique_and_source_bound() -> None:
    journal_bytes = JOURNAL_PATH.read_bytes()
    journal = json.loads(journal_bytes)
    recovery = _load(RECOVERY_PATH)

    assert hashlib.sha256(journal_bytes).hexdigest() == EXPECTED_JOURNAL_FILE_HASH
    assert journal["journal_sha256"] == EXPECTED_JOURNAL_HASH
    assert _embedded_hash(journal, "journal_sha256") == EXPECTED_JOURNAL_HASH
    assert journal["status"] == "data_complete"
    assert journal["completed_request_count"] == 20
    assert journal["next_request"] is None

    responses = [
        response
        for state in journal["symbol_states"].values()
        for response in state["responses"]
    ]
    assert set(journal["symbol_states"]) == {
        "BTCUSDC",
        "BTCUSDT",
        "SOLUSDC",
        "SOLUSDT",
    }
    assert all(state["complete"] for state in journal["symbol_states"].values())
    assert all(
        len(state["responses"]) == 5 for state in journal["symbol_states"].values()
    )
    assert len(responses) == 20
    fingerprints = {
        (
            response["request"]["parameters"]["symbol"],
            response["request"]["parameters"]["endTime"],
        )
        for response in responses
    }
    assert len(fingerprints) == len(responses)
    assert all(response["receipt"]["status_code"] == 200 for response in responses)
    assert all(
        len(response["receipt"]["payload_sha256"]) == 64 for response in responses
    )

    source = recovery["source_contract"]
    assert source["journal_file_sha256"] == EXPECTED_JOURNAL_FILE_HASH
    assert source["journal_sha256"] == EXPECTED_JOURNAL_HASH
    assert (
        source["recent_result_sha256"]
        == EXPECTED_RESULT_HASHES[
            "binance-cross-stablecoin-funding-snapshot-v1-2026-08-25.json"
        ]
    )
    assert (
        source["recovery_contract_result_sha256"]
        == EXPECTED_RESULT_HASHES[
            "binance-cross-stablecoin-funding-recovery-contract-v4.json"
        ]
    )
    assert (
        source["implementation_sha256"]
        == EXPECTED_TOOL_HASHES["recover_binance_cross_stablecoin_funding.py"]
    )
    assert (
        hashlib.sha256((ROOT / source["implementation_path"]).read_bytes()).hexdigest()
        == source["implementation_sha256"]
    )


def test_each_attempt_binds_the_exact_published_implementation() -> None:
    recent = _load(
        ACTION_VALUE / "binance-cross-stablecoin-funding-snapshot-v1-2026-08-25.json"
    )
    adjudication = _load(
        ACTION_VALUE
        / "binance-cross-stablecoin-funding-adjudication-v2-2026-08-25.json"
    )
    recovery_contract = _load(
        ACTION_VALUE / "binance-cross-stablecoin-funding-recovery-contract-v4.json"
    )

    for payload, name in (
        (recent, "screen_binance_cross_stablecoin_funding.py"),
        (adjudication, "adjudicate_binance_cross_stablecoin_funding.py"),
    ):
        source = payload["source_contract"]
        assert source["implementation_sha256"] == EXPECTED_TOOL_HASHES[name]
        assert (
            hashlib.sha256(
                (ROOT / source["implementation_path"]).read_bytes()
            ).hexdigest()
            == source["implementation_sha256"]
        )

    boundary = recovery_contract["recovery_boundary"]
    assert (
        boundary["attempt3_tool_file_sha256"]
        == EXPECTED_TOOL_HASHES["backfill_binance_cross_stablecoin_funding.py"]
    )
    assert (
        hashlib.sha256((ROOT / boundary["attempt3_tool_path"]).read_bytes()).hexdigest()
        == boundary["attempt3_tool_file_sha256"]
    )
    assert boundary["further_regeneration_after_attempt4"] is False


def test_full_history_gate_rejects_both_market_neutral_candidates() -> None:
    recovery = _load(RECOVERY_PATH)
    verdict = recovery["verdict"]

    assert recovery["new_request_count"] == 20
    assert verdict == {
        "accepted_edge": False,
        "candidate_count": 2,
        "credentials_used": False,
        "orders_placed": False,
        "profitability_claim": False,
        "qualified_public_recovery_count": 0,
        "status": (
            "rejected_full_history_fx_stressed_cross_stablecoin_funding_differential"
        ),
        "trading_authority": False,
    }

    candidates = {item["base_asset"]: item for item in recovery["candidate_results"]}
    assert set(candidates) == {"BTC", "SOL"}
    btc = candidates["BTC"]
    sol = candidates["SOL"]
    for candidate in candidates.values():
        assert candidate["history"]["union_settlement_count"] == 2_898
        assert candidate["selection"]["row_count"] == 966
        assert candidate["validation"]["row_count"] == 966
        assert candidate["test"]["row_count"] == 966
        assert candidate["qualified_public_recovery_screen"] is False

    assert btc["symbols"] == {"long": "BTCUSDT", "short": "BTCUSDC"}
    assert btc["selection"]["worst_stress_usdt"] == "-126.837560885663154420"
    assert btc["validation"]["total"]["worst_stress_usdt"] == (
        "-330.558130558510651086"
    )
    assert btc["test"]["total"]["worst_stress_usdt"] == ("312.897317669218198470")
    assert (
        btc["current_execution_diagnostic"][
            "test_equal_per_leg_fee_break_even_after_stressed_spread_bips"
        ]
        == "9.691814397531344112313117278"
    )
    assert "selection_fx_stress_failed" in btc["failure_reasons"]
    assert "validation_total_fx_stress_failed" in btc["failure_reasons"]

    assert sol["symbols"] == {"long": "SOLUSDC", "short": "SOLUSDT"}
    assert sol["selection"]["worst_stress_usdt"] == "0.307101618854974598"
    assert sol["validation"]["total"]["worst_stress_usdt"] == ("-1.267714269323078802")
    assert sol["test"]["total"]["worst_stress_usdt"] == ("-1.467991071315862632")
    assert (
        sol["current_execution_diagnostic"][
            "test_equal_per_leg_fee_break_even_after_stressed_spread_bips"
        ]
        == "-36.66779525528060892263250823"
    )
    assert "validation_total_fx_stress_failed" in sol["failure_reasons"]
    assert "test_total_fx_stress_failed" in sol["failure_reasons"]


def test_failed_v3_run_and_terminal_registry_remain_explicit() -> None:
    failure = _load(FAILURE_PATH)
    registry = _load(REGISTRY_PATH)
    continuation = (ROOT / "docs" / "CONTINUATION.md").read_text(encoding="utf-8")
    workflow = (ROOT / "docs" / "AGENT_WORKFLOWS.md").read_text(encoding="utf-8")

    assert failure["error"] == "same-day USDCUSDT conversion bar is absent"
    assert failure["accepted_edge"] is False
    assert failure["profitability_claim"] is False
    assert failure["trading_authority"] is False
    terminal = {item["family"]: item for item in registry["terminal_do_not_repeat"]}
    family = terminal["binance_usdt_usdc_perpetual_funding_differential"]
    assert (
        family["canonical_result_sha256"]
        == EXPECTED_RESULT_HASHES[
            "binance-cross-stablecoin-funding-recovery-v4-2026-08-25.json"
        ]
    )
    assert registry["accepted_edge_count"] == 21
    assert EXPECTED_REGISTRY_HASH in continuation
    assert "USDT/USDC perpetual funding" in continuation
    assert "initialize a self-hashed persistent" in workflow
    assert "request journal" in workflow
    assert "violated this rule" in workflow
