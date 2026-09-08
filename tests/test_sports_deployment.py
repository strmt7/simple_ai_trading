import json
import hashlib

import pytest

from tools.adjudicate_sports_deployment import evaluate
from tools import adjudicate_sports_deployment as module


def event():
    return {
        "slug": "exact",
        "createdAt": "2026-09-04T00:00:00Z",
        "startTime": "2026-09-11T00:00:00Z",
        "active": True,
        "closed": False,
        "markets": [
            {
                "id": "1",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
                "bestAsk": "deliberately invalid unused price",
            }
        ],
    }


def check(value):
    return evaluate(
        json.dumps(value).encode(),
        "exact",
        "2026-08-31T16:33:29.370Z",
        "2026-09-08T14:00:00Z",
    )


def test_new_deployment_does_not_use_price_fields():
    result = check(event())
    assert result["deployment_gate_passed"]
    assert result["economic_fields_examined"] is False
    assert result["active_accepting_market_ids"] == ["1"]


@pytest.mark.parametrize(
    "changes",
    [
        {"createdAt": "2026-08-31T16:33:29.370Z"},
        {"startTime": "2026-09-07T00:00:00Z"},
        {"active": False},
        {"closed": True},
    ],
)
def test_old_or_inactive_deployment_stops(changes):
    assert not check({**event(), **changes})["deployment_gate_passed"]


@pytest.mark.parametrize(
    "changes",
    [
        {"slug": "other"},
        {"createdAt": "2026-09-09T00:00:00Z"},
        {"createdAt": "2026-09-04T00:00:00"},
        {"startTime": None},
        {"markets": []},
        {"markets": [{"id": 1}]},
        {"markets": [{"id": "1"}, {"id": "1"}]},
    ],
)
def test_missing_ambiguous_or_conflicting_deployment_rejected(changes):
    with pytest.raises(ValueError):
        check({**event(), **changes})


def test_no_accepting_market_stops():
    value = event()
    value["markets"][0]["acceptingOrders"] = False
    assert not check(value)["deployment_gate_passed"]


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "source_gate",
        "malformed_json",
        "contract",
        "source_hash",
        "raw_hash",
        "implementation",
        "exists",
    ],
)
def test_bound_adjudication_preserves_failures_without_network(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(module, "_root_path", lambda value: tmp_path / value)
    raw = json.dumps(event()).encode() if failure != "malformed_json" else b"{"
    (tmp_path / "raw.json").write_bytes(raw)
    (tmp_path / "implementation.py").write_bytes(b"# frozen synthetic implementation\n")
    plan = {
        "implementations": [
            {
                "path": "implementation.py",
                "sha256": hashlib.sha256(
                    (tmp_path / "implementation.py").read_bytes()
                ).hexdigest(),
            }
        ],
        "outputs": {"result_path": "source.json", "raw_path": "raw.json"},
        "deployment_gate": {
            "result_path": "result.json",
            "event_slug": "exact",
            "after_utc": "2026-08-31T16:33:29.370Z",
        },
    }
    if failure == "implementation":
        plan["implementations"][0]["sha256"] = "0" * 64
    plan["contract_sha256"] = module._canonical_hash(plan, "contract_sha256")
    source = {
        "contract": {"sha256": plan["contract_sha256"]},
        "source_gate": {"passed": failure != "source_gate"},
        "capture": {
            "receipt": {
                "completed_at_ms": 1788876000000,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
            }
        },
    }
    if failure == "raw_hash":
        source["capture"]["receipt"]["response_sha256"] = "0" * 64
    source["result_sha256"] = module._canonical_hash(source, "result_sha256")
    if failure == "source_hash":
        source["result_sha256"] = "0" * 64
    if failure == "contract":
        plan["contract_sha256"] = "0" * 64
    if failure == "exists":
        (tmp_path / "result.json").write_text("preserved")
    (tmp_path / "source.json").write_text(json.dumps(source))
    (tmp_path / "contract.json").write_text(json.dumps(plan))
    if failure in {"contract", "source_hash", "raw_hash", "implementation", "exists"}:
        with pytest.raises((ValueError, FileExistsError)):
            module.adjudicate(tmp_path / "contract.json")
    else:
        result = module.adjudicate(tmp_path / "contract.json")
        assert result["deployment"]["deployment_gate_passed"] is (failure is None)
        assert result["result_sha256"] == module._canonical_hash(
            result, "result_sha256"
        )
        assert json.loads((tmp_path / "result.json").read_bytes()) == result
