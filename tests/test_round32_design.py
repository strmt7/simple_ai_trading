from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
REGISTRY = RESEARCH / "consumed-periods-through-round-031.json"
DESIGN = RESEARCH / "round-032-shared-action-value-viability-design.json"
ROUND31_DESIGN = RESEARCH / "round-031-frozen-chronological-confirmation-design.json"


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _canonical_sha256(value: dict[str, object], field: str) -> str:
    payload = dict(value)
    payload.pop(field)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dates(first: str, last: str) -> set[str]:
    start = date.fromisoformat(first)
    end = date.fromisoformat(last)
    assert start <= end
    return {
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    }


def test_round31_consumed_registry_is_canonical_and_complete() -> None:
    registry = _read(REGISTRY)

    assert registry["registry_sha256"] == _canonical_sha256(
        registry, "registry_sha256"
    )
    round31 = [item for item in registry["records"] if item["round"] == 31]
    assert round31 == [
        {
            "round": 31,
            "status": "consumed",
            "outcome": "rejected",
            "design_sha256": (
                "1d6e8791f635be6d8d98b9f957ffffefbc211692d4f7adf07fdffb8fea667c0e"
            ),
            "report_sha256": (
                "b3195481f6541442051e3c06d640ee660a7bf55740abc5186037d486b9091371"
            ),
            "windows": [
                {"start_date": "2024-01-01", "end_date": "2024-02-04"}
            ],
        }
    ]


def test_round32_design_is_hash_bound_consumed_only_and_one_variant() -> None:
    design = _read(DESIGN)
    registry = _read(REGISTRY)
    round31 = _read(ROUND31_DESIGN)

    assert design["design_sha256"] == _canonical_sha256(design, "design_sha256")
    governance = design["governance"]
    assert governance["consumed_period_registry_file_sha256"] == hashlib.sha256(
        REGISTRY.read_bytes()
    ).hexdigest()
    assert governance["consumed_period_registry_canonical_sha256"] == registry[
        "registry_sha256"
    ]
    assert governance["variant_budget"] == 1
    assert governance["hyperparameter_search_permitted"] is False
    assert design["design_revision"] == 2
    assert design["data"]["full_history_inventory_required"] is True

    consumed: set[str] = set()
    for record in registry["records"]:
        for window in record["windows"]:
            consumed.update(_dates(window["start_date"], window["end_date"]))
    roles = design["data"]["roles"]
    for name in ("train", "early_stop", "calibration", "policy", "development"):
        assert _dates(roles[name]["start"], roles[name]["end"]) <= consumed
    distant = roles["distant_confirmation"]
    assert _dates(distant["start"], distant["end"]) <= consumed

    forbidden: set[str] = set()
    for window in design["data"]["forbidden_target_windows"]:
        forbidden.update(_dates(window["start"], window["end"]))
    evaluated = set().union(
        *(
            _dates(roles[name]["start"], roles[name]["end"])
            for name in ("train", "early_stop", "calibration", "policy", "development")
        ),
        _dates(distant["start"], distant["end"]),
    )
    assert not evaluated & forbidden
    assert design["execution"] == round31["execution"]
    barrier = dict(design["barrier_targets"])
    assert barrier.pop("target_scenario") == "stress"
    assert barrier == round31["barrier_targets"]
    assert all(value is False for value in design["claims"].values())
    assert design["acceptance_gates"]["economic"]["leverage_permitted"] is False
