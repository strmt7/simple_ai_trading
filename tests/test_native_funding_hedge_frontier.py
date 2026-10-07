from fractions import Fraction as F
import hashlib
import json

import pytest

from simple_ai_trading.funding_cash import LinearFundingSettlement
from tools.capture_public_source_contract import _canonical_hash
from tools import review_native_funding_hedge_frontier as subject


def event(time, mark, rate, symbol="BTCUSDT"):
    return LinearFundingSettlement(symbol, "USDT", time, F(rate), F(mark), "a" * 64)


def test_equal_base_hedge_cancels_common_direction_not_basis_or_costs():
    values = dict(
        base_quantity=F(2),
        entry_spot=F(100),
        entry_future=F(101),
        exit_spot=F(200),
        exit_future=F(203),
        funding_cash=F(5),
        total_cost=F(2),
    )
    assert subject.hedge_net_cash(**values) == -1
    values.update(exit_spot=F(50), exit_future=F(53))
    assert subject.hedge_net_cash(**values) == -1
    values.update(exit_future=F(51))
    assert subject.hedge_net_cash(**values) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("base_quantity", F(0)),
        ("entry_spot", F(-1)),
        ("exit_spot", True),
        ("entry_future", 1.0),
        ("exit_future", F(0)),
        ("funding_cash", 2),
        ("total_cost", F(-1)),
    ],
)
def test_hedge_rejects_invalid_units(field, value):
    values = dict(
        base_quantity=F(1),
        entry_spot=F(100),
        exit_spot=F(100),
        entry_future=F(100),
        exit_future=F(100),
        funding_cash=F(0),
        total_cost=F(0),
    )
    values[field] = value
    with pytest.raises(ValueError):
        subject.hedge_net_cash(**values)


def panel(events=None, **changes):
    values = dict(
        end_ms=subject.DAY_MS,
        cost_reserves_bps=[F(32)],
        capital_aprs_bps=[F(365)],
        capital_multiple=F(2),
    )
    values.update(changes)
    return subject.frontier(
        events or [event(0, 100, "1"), event(1, 120, "-0.001"), event(2, 100, "0.002")],
        **values,
    )


def test_reference_payment_excluded_and_marks_weight_actual_cash():
    result = panel()
    assert F(result["funding_cash_usdt_per_base"]) == F("0.08")
    assert F(result["funding_bps_at_reference"]) == 8
    assert F(result["funding_only_prefund_usdt_per_base"]) == F("0.12")
    assert F(result["funding_only_maximum_drawdown_usdt_per_base"]) == F("0.12")
    assert F(result["unknown_entitlement_cash_lower"]) == F("-0.12")
    assert F(result["unknown_entitlement_cash_upper"]) == F("0.2")
    sensitivity = result["sensitivities"][0]
    assert F(sensitivity["capital_cost_bps"]) == 2
    assert (
        F(sensitivity["maximum_basis_deterioration_bps_for_strict_positive_net"]) == -26
    )
    assert not result["profitability_claim"]
    assert not result["complete_hedge_return_observed"]


def test_break_even_frontier_matches_full_hedge_cash_at_and_beyond_boundary():
    result = panel()
    frontier_bps = F(
        result["sensitivities"][0][
            "maximum_basis_deterioration_bps_for_strict_positive_net"
        ]
    )
    deterioration = frontier_bps * F(100) / 10_000
    values = dict(
        base_quantity=F(1),
        entry_spot=F(100),
        entry_future=F(100),
        exit_spot=F(200),
        exit_future=F(200) + deterioration,
        funding_cash=F("0.08"),
        total_cost=F("0.34"),
    )
    assert subject.hedge_net_cash(**values) == 0
    values["exit_future"] -= F("0.01")
    assert subject.hedge_net_cash(**values) == F("0.01")


@pytest.mark.parametrize(
    "changes",
    [
        {"end_ms": True},
        {"end_ms": 2},
        {"capital_multiple": F(0)},
        {"cost_reserves_bps": []},
        {"capital_aprs_bps": [F(-1)]},
        {"cost_reserves_bps": [1.0]},
    ],
)
def test_frontier_rejects_invalid_contract(changes):
    with pytest.raises(ValueError):
        panel(**changes)


@pytest.mark.parametrize(
    "events",
    [
        [event(0, 100, "0")],
        [event(0, 100, "0"), event(0, 100, "0")],
        [event(1, 100, "0"), event(0, 100, "0")],
        [event(0, 100, "0"), event(1, 100, "0", "ETHUSDT")],
    ],
)
def test_frontier_rejects_partial_ambiguous_or_mixed_population(events):
    with pytest.raises(ValueError):
        panel(events)


def frozen_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "_root_path", lambda path: tmp_path / path)
    raw = json.dumps(
        [
            dict(symbol="BTCUSDT", fundingTime=i, fundingRate="0.001", markPrice="100")
            for i in (0, 1, 2)
        ]
    ).encode()
    (tmp_path / "raw.json").write_bytes(raw)
    plan = dict(
        schema_version="native-funding-hedge-frontier-contract-v1",
        frozen_at_utc="2026-01-01T00:00:00Z",
        contract_path="contract.json",
        new_requests=0,
        accepted_edge=False,
        research_observation_count_effect=0,
        implementations=[],
        output="result.json",
        journal="journal.jsonl",
        inputs=[
            dict(
                path="raw.json",
                sha256=hashlib.sha256(raw).hexdigest(),
                symbol="BTCUSDT",
                rows=3,
            )
        ],
        start_ms=0,
        end_ms=subject.DAY_MS,
        symbols=["BTCUSDT"],
        panels=[
            dict(
                name="whole",
                start_ms=0,
                end_ms=subject.DAY_MS,
                expected_supplied_events_per_symbol=3,
            )
        ],
        cost_reserves_bps=["32"],
        capital_aprs_bps=["0"],
        capital_multiple="2",
    )
    plan["contract_sha256"] = _canonical_hash(plan, "contract_sha256")
    (tmp_path / "contract.json").write_text(json.dumps(plan))
    return tmp_path / "contract.json", plan


def test_one_use_offline_result_and_durable_journal(tmp_path, monkeypatch):
    path, plan = frozen_fixture(tmp_path, monkeypatch)
    result = subject.review(path)
    assert result["passed"] and len(result["panels"]) == 1
    assert result["result_sha256"] == _canonical_hash(result, "result_sha256")
    assert not result["accepted_edge"] and not result["cash_labels_admitted"]
    assert result["new_requests"] == 0
    records = [
        json.loads(line)
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
    ]
    assert [r["phase"] for r in records] == ["intent", "completed"]
    assert records[0]["contract_sha256"] == plan["contract_sha256"]
    with pytest.raises(FileExistsError):
        subject.review(path)


def test_source_tamper_retains_terminal_failure_without_metrics(tmp_path, monkeypatch):
    path, _ = frozen_fixture(tmp_path, monkeypatch)
    (tmp_path / "raw.json").write_bytes(b"[]")
    result = subject.review(path)
    assert not result["passed"] and result["panels"] == []
    assert result["error"] == "ValueError"
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        subject.review(path)


def test_contract_tamper_fails_before_journal(tmp_path, monkeypatch):
    path, plan = frozen_fixture(tmp_path, monkeypatch)
    plan["new_requests"] = 1
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        subject.review(path)
    assert not (tmp_path / "journal.jsonl").exists()


@pytest.mark.parametrize("change", ["rows", "symbol", "window", "panels"])
def test_bad_source_population_retains_failure(tmp_path, monkeypatch, change):
    path, plan = frozen_fixture(tmp_path, monkeypatch)
    if change == "rows":
        plan["inputs"][0]["rows"] = 4
    elif change == "symbol":
        plan["symbols"] = ["ETHUSDT"]
    elif change == "window":
        plan["start_ms"] = 1
    else:
        plan["panels"][0]["expected_supplied_events_per_symbol"] = 4
    plan["contract_sha256"] = _canonical_hash(plan, "contract_sha256")
    path.write_text(json.dumps(plan))
    result = subject.review(path)
    assert not result["passed"] and not result["panels"]
    assert len((tmp_path / "journal.jsonl").read_text().splitlines()) == 2


def test_implementation_drift_fails_before_journal(tmp_path, monkeypatch):
    path, plan = frozen_fixture(tmp_path, monkeypatch)
    plan["implementations"] = [dict(path="raw.json", sha256="0" * 64)]
    plan["contract_sha256"] = _canonical_hash(plan, "contract_sha256")
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="implementation"):
        subject.review(path)
    assert not (tmp_path / "journal.jsonl").exists()


@pytest.mark.parametrize("passed", [True, False])
def test_cli_reports_bounded_terminal_status(monkeypatch, capsys, passed):
    monkeypatch.setattr("sys.argv", ["review", "--contract", "unused.json"])
    monkeypatch.setattr(subject, "review", lambda path: dict(passed=passed, panels=[]))
    if passed:
        subject.main()
    else:
        with pytest.raises(SystemExit) as exc:
            subject.main()
        assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out) == dict(passed=passed, panels=0)
