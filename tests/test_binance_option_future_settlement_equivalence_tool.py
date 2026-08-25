from __future__ import annotations

from decimal import Decimal
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_binance_option_future_settlement_equivalence",
    ROOT / "tools" / "audit_binance_option_future_settlement_equivalence.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
CONTRACT_HASH = "63a57771fe7042381bea0ac052889550738b4890b6c01fadc279e793189b4291"


class _Response:
    def __init__(self, payload: object) -> None:
        self.body = json.dumps(payload, sort_keys=True).encode("ascii")
        self.status_code = 200
        self.url = TOOL.EXERCISE_URL
        self.headers = {"Content-Type": "application/json"}
        self.is_redirect = False

    def iter_content(self, *, chunk_size: int):
        assert chunk_size == 65_536
        yield self.body


class _Session:
    def __init__(
        self,
        prices: Mapping[tuple[str, int], Decimal],
        *,
        mismatch_first: bool = False,
    ) -> None:
        self.prices = dict(prices)
        self.mismatch_first = mismatch_first
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
        stream: bool,
        allow_redirects: bool,
    ) -> _Response:
        assert url == TOOL.EXERCISE_URL
        assert timeout == 3.0
        assert stream is True
        assert allow_redirects is False
        copied = dict(params)
        self.calls.append((url, copied))
        underlying = str(copied["underlying"])
        day_start_ms = int(copied["startTime"])
        price = self.prices[(underlying, day_start_ms)]
        if self.mismatch_first and len(self.calls) == 1:
            price += Decimal("0.01")
        asset = underlying.removesuffix("USDT")
        expiry_ms = day_start_ms + 8 * 60 * 60 * 1000
        return _Response(
            [
                {
                    "symbol": f"{asset}-000000-1-C",
                    "strikePrice": "1",
                    "realStrikePrice": str(price),
                    "expiryDate": expiry_ms,
                    "strikeResult": "REALISTIC_VALUE_STRICKEN",
                },
                {
                    "symbol": f"{asset}-000000-1-P",
                    "strikePrice": "1",
                    "realStrikePrice": str(price),
                    "expiryDate": expiry_ms,
                    "strikeResult": "EXTRINSIC_VALUE_EXPIRED",
                },
            ]
        )


class _MemoryJournal:
    def __init__(self) -> None:
        self.path = ROOT / "not-written-settlement-audit-journal.json"
        self.payload: dict[str, object] = {"journal_sha256": "memory"}
        self.events: list[dict[str, object]] = []
        self.completed = 0

    def event(self, phase: str, **fields: object) -> None:
        self.events.append({"phase": phase, **fields})

    def request_complete(self) -> None:
        self.completed += 1


def _delivery_prices(contract: Mapping[str, object]) -> dict[tuple[str, int], Decimal]:
    requests_list = TOOL._requests(contract)
    dates = {int(item["params"]["startTime"]) for item in requests_list}
    source = TOOL._bound_source(contract, "delivery_source")
    return TOOL._delivery_prices(source, dates=dates)


def test_contract_sources_and_stale_discovery_reconstruct() -> None:
    contract = TOOL._load_contract()
    assert contract["result_sha256"] == CONTRACT_HASH
    option_snapshot = TOOL._bound_source(contract, "option_snapshot")
    quarterly_snapshot = TOOL._bound_source(contract, "quarterly_snapshot")

    discovery = TOOL._stale_discovery(option_snapshot, quarterly_snapshot)

    receipt = contract["receipt"]
    assert receipt["journal_path"].endswith("settlement-equivalence-journal-v1.json")
    assert receipt["result_path"].endswith("settlement-equivalence-v1-2026-08-25.json")
    TOOL._require_receipt_paths(
        contract,
        output=ROOT / receipt["result_path"],
        journal=ROOT / receipt["journal_path"],
    )
    with pytest.raises(ValueError, match="differ from the frozen contract"):
        TOOL._require_receipt_paths(
            contract,
            output=ROOT / "different-result.json",
            journal=ROOT / receipt["journal_path"],
        )
    assert discovery["common_expiry_strike_pair_count"] == 192
    assert discovery["ticker_path_available_pair_count"] == 184
    assert discovery["gross_positive_ticker_combination_count"] == 20
    assert discovery["best_candidates"][0]["gross_quote_per_base"] == "101.700"
    assert discovery["synchronous"] is False
    assert discovery["execution_evidence"] is False


def test_exercise_price_requires_one_two_sided_0800_value() -> None:
    day_start_ms = 1_758_844_800_000
    value, count, expiry_ms = TOOL._exercise_price(
        [
            {
                "symbol": "BTC-250926-100000-C",
                "realStrikePrice": "109049.7",
                "expiryDate": day_start_ms + 28_800_000,
                "strikeResult": "REALISTIC_VALUE_STRICKEN",
            },
            {
                "symbol": "BTC-250926-100000-P",
                "realStrikePrice": "109049.7",
                "expiryDate": day_start_ms + 28_800_000,
                "strikeResult": "EXTRINSIC_VALUE_EXPIRED",
            },
        ],
        underlying="BTCUSDT",
        day_start_ms=day_start_ms,
    )

    assert value == Decimal("109049.7")
    assert count == 2
    assert expiry_ms == day_start_ms + 28_800_000


def test_exercise_price_rejects_settlement_disagreement() -> None:
    day_start_ms = 1_758_844_800_000
    payload = [
        {
            "symbol": "BTC-250926-100000-C",
            "realStrikePrice": "109049.7",
            "expiryDate": day_start_ms + 28_800_000,
            "strikeResult": "REALISTIC_VALUE_STRICKEN",
        },
        {
            "symbol": "BTC-250926-100000-P",
            "realStrikePrice": "109049.8",
            "expiryDate": day_start_ms + 28_800_000,
            "strikeResult": "EXTRINSIC_VALUE_EXPIRED",
        },
    ]

    with pytest.raises(ValueError, match="not unique and two-sided"):
        TOOL._exercise_price(
            payload,
            underlying="BTCUSDT",
            day_start_ms=day_start_ms,
        )


def test_journal_is_self_hashed_and_refuses_a_rerun(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    journal = TOOL._Journal(
        path=path,
        contract={"result_sha256": "frozen"},
        request_count=8,
    )
    journal.event("reserved_before_request", request_id="one")
    payload = json.loads(path.read_bytes())
    claimed = payload.pop("journal_sha256")

    assert claimed == TOOL._sha256(TOOL._canonical_json(payload).encode("ascii"))
    assert journal.payload["orders_submitted"] is False
    assert journal.payload["credentials_used"] is False
    with pytest.raises(RuntimeError, match="already exists"):
        TOOL._Journal(
            path=path,
            contract={"result_sha256": "frozen"},
            request_count=8,
        )
    with pytest.raises(ValueError, match="inside the repository"):
        TOOL._repo_relative(ROOT.parent / "outside-settlement-receipt.json")


@pytest.mark.parametrize(
    ("mismatch_first", "expected_equal", "expected_status"),
    [
        (
            False,
            8,
            "historical_settlement_equivalence_supports_synchronized_depth_screen",
        ),
        (
            True,
            7,
            "settlement_benchmark_mismatch_rejects_fixed_payoff_claim",
        ),
    ],
)
def test_run_adjudicates_exact_benchmark_equality(
    mismatch_first: bool,
    expected_equal: int,
    expected_status: str,
) -> None:
    contract = TOOL._load_contract()
    prices = _delivery_prices(contract)
    session = _Session(prices, mismatch_first=mismatch_first)
    journal = _MemoryJournal()

    result = TOOL.run(
        session=session,
        contract=contract,
        journal=journal,
        timeout_seconds=3.0,
    )

    assert len(session.calls) == 8
    assert journal.completed == 8
    assert [event["phase"] for event in journal.events] == [
        phase
        for _ in range(8)
        for phase in (
            "reserved_before_request",
            "raw_response_persisted_before_validation",
            "response_validated",
        )
    ]
    assert result["verdict"]["exact_equality_count"] == expected_equal
    assert result["verdict"]["status"] == expected_status
    assert result["verdict"]["accepted_edge"] is False
    assert result["safety"] == {
        "new_public_get_count": 8,
        "credentials_used": False,
        "orders_submitted": False,
        "retries": 0,
        "adaptive_requests": 0,
        "profitability_claim": False,
    }
