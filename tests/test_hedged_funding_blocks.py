from decimal import Decimal
import json

import pytest

from tools import screen_hedged_funding_blocks as screen


START = 1_788_739_200_000  # Monday 2026-09-07, fixed synthetic clock anchor.


def inventory():
    return {
        "results": [
            {
                "symbol": a + "-USD-PERP",
                "asset_kind": "PERP",
                "base_currency": a,
                "quote_currency": "USD",
                "settlement_currency": "USDC",
                "funding_period_hours": 8,
                "trading_mode": "STANDARD",
                "open_at": 1,
                "expiry_at": 0,
            }
            for a in screen.native.ASSETS
        ]
    }


def sources():
    indices = [
        {"time": START + i * screen.WEEK + 1001, "index": str(3 * i)} for i in range(10)
    ]
    funding = [
        {"time": START + i * screen.native.STEP + 999, "rate": "0.001", "mark": "100"}
        for i in range(190)
    ]
    return indices, funding


def contract(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "ROOT", tmp_path)
    (tmp_path / "data/research").mkdir(parents=True)
    (tmp_path / "prior.json").write_text(
        json.dumps({"end_boundary_ms": START - screen.WEEK})
    )
    result = {
        "source_sha256": {},
        "start_boundary_ms": START,
        "end_boundary_ms": START + 9 * screen.WEEK,
        "prior_consumed_contract": "prior.json",
        "output_root": "data/research/study",
        "requests": screen.requests_for(START, START + 9 * screen.WEEK),
        "roles": {"training": [0, 5], "validation": [5, 7], "test": [7, 9]},
        "economics": screen.ECONOMICS,
    }
    return result


def save_contract(tmp_path, data):
    path = tmp_path / "contract.json"
    data = {
        **data,
        "contract_sha256": screen.native.digest(screen.native.canonical(data)),
    }
    path.write_text(json.dumps(data))
    return path


def test_fixed_population_is_34_distinct_public_requests_with_bounded_rows():
    requests = screen.requests_for(START, START + 9 * screen.WEEK)
    assert len(requests) == len({x["url"] for x in requests}) == 34
    assert sum(x["kind"] == "index" for x in requests) == 30
    assert sum(x["kind"] == "binance" for x in requests) == 3
    assert all(x["request_body_sha256"] == screen.native.digest(b"") for x in requests)
    assert START % screen.WEEK == 4 * 86_400_000


def test_every_actual_settlement_is_counted_once_and_indices_telescope():
    indices, funding = sources()
    rows = screen.block_rows(indices, funding)
    assert len(rows) == 9
    assert sum(x["binance_settlements"] for x in rows) == 189
    assert all(
        Decimal(x["binance_long_cash_usdt_per_base"]) == Decimal("-2.1") for x in rows
    )
    assert sum(Decimal(x["paradex_short_cash_usdc_per_base"]) for x in rows) == 27
    assert rows[0]["start"] < funding[21]["time"] < rows[0]["end"]
    assert funding[0]["time"] < rows[0]["start"]


def test_misaligned_actual_endpoint_cannot_silently_drop_a_settlement():
    indices, funding = sources()
    indices[1]["time"] = START + screen.WEEK + 500
    with pytest.raises(ValueError, match="enclose every"):
        screen.block_rows(indices, funding)


@pytest.mark.parametrize(
    "change", ["duplicate", "currency", "future_open", "wrong_period", "cursor"]
)
def test_inventory_changes_fail_before_economic_data(change):
    data = inventory()
    if change == "duplicate":
        data["results"].append(dict(data["results"][0]))
    elif change == "currency":
        data["results"][0]["settlement_currency"] = "USDT"
    elif change == "future_open":
        data["results"][0]["open_at"] = START + 1
    elif change == "wrong_period":
        data["results"][0]["funding_period_hours"] = 4
    else:
        data["next"] = "more"
    with pytest.raises(ValueError):
        screen.validate_inventory(data, START)


def test_training_only_orientation_is_not_refit_to_later_roles():
    rows = screen.block_rows(*sources())
    roles = {"training": [0, 5], "validation": [5, 7], "test": [7, 9]}
    original = screen.adjudicate(rows, roles)
    for row in rows[5:]:
        row["paradex_short_cash_usdc_per_base"] = "-999"
    changed = screen.adjudicate(rows, roles)
    assert original["orientation"] == changed["orientation"] == 1
    assert original["roles"]["training"] == changed["roles"]["training"]
    assert not changed["passes_prefilter"]
    assert Decimal(changed["roles"]["test"]["net_after_frozen_hurdles_bips"]) < 0


def test_perfect_foresight_zero_execution_bound_dominates_fixed_sign_net():
    result = screen.adjudicate(
        screen.block_rows(*sources()),
        {"training": [0, 5], "validation": [5, 7], "test": [7, 9]},
    )
    for role in result["roles"].values():
        assert (
            Decimal(
                role["perfect_foresight_weekly_sign_zero_execution_net_of_capital_bips"]
            )
            >= Decimal(role["net_after_frozen_hurdles_bips"]) + 80
        )


def test_preflight_has_no_network_and_rejects_consumed_output(tmp_path, monkeypatch):
    data = contract(tmp_path, monkeypatch)
    path = save_contract(tmp_path, data)
    monkeypatch.setattr(
        screen.native,
        "captured_get",
        lambda *a, **kw: pytest.fail("preflight performed HTTP"),
    )
    assert screen.run(path, preflight=True) is None
    (tmp_path / data["output_root"]).mkdir()
    with pytest.raises(ValueError, match="new owned"):
        screen.run(path, preflight=True)


@pytest.mark.parametrize("change", ["overlap", "roles", "economics", "requests"])
def test_changed_contract_never_reaches_http(tmp_path, monkeypatch, change):
    data = contract(tmp_path, monkeypatch)
    if change == "overlap":
        (tmp_path / "prior.json").write_text(json.dumps({"end_boundary_ms": START}))
    elif change == "roles":
        data["roles"]["test"] = [8, 9]
    elif change == "economics":
        data["economics"] = {**screen.ECONOMICS, "execution_stress_bips": "0"}
    else:
        data["requests"].pop()
    with pytest.raises(ValueError):
        screen.run(save_contract(tmp_path, data), preflight=True)


def test_first_source_failure_is_terminal_with_durable_bytes_and_no_retry(
    tmp_path, monkeypatch
):
    data = contract(tmp_path, monkeypatch)
    path = save_contract(tmp_path, data)
    calls = []

    def failed(request, *, raw, journal):
        calls.append(request["name"])
        screen.native.record(
            journal, {"phase": "request_started", "name": request["name"]}
        )
        raw.write_bytes(b"{}")
        raise ValueError("synthetic source failure")

    monkeypatch.setattr(screen.native, "captured_get", failed)
    result = screen.run(path)
    assert calls == ["inventory"]
    assert result["status"] == "terminal_source_or_alignment_failure"
    assert not result["accepted_edge"] and not result["trading_authority"]
    output = tmp_path / data["output_root"]
    assert (output / "inventory.json").read_bytes() == b"{}"
    assert "study_terminal_failure" in (output / "requests.jsonl").read_text()
    assert (output / "result.json").is_file()
    with pytest.raises(ValueError, match="new owned"):
        screen.run(path)
