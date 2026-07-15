from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.run_round59_funding_persistence_feasibility import _canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = (
    ROOT
    / "docs/model-research/action-value/round-061-carry-economic-replay-design.json"
)
MANIFEST_PATH = (
    ROOT / "docs/model-research/action-value/round-061-carry-event-manifest.json"
)


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_round61_design_and_price_blind_manifest_are_hash_bound() -> None:
    design = _read(DESIGN_PATH)
    manifest = _read(MANIFEST_PATH)
    canonical_design = dict(design)
    canonical_manifest = dict(manifest)
    design_sha = canonical_design.pop("design_sha256")
    manifest_sha = canonical_manifest.pop("manifest_sha256")

    assert design_sha == _canonical_sha256(canonical_design)
    assert manifest_sha == _canonical_sha256(canonical_manifest)
    assert design["event_contract"]["manifest_canonical_sha256"] == manifest_sha
    assert (
        design["event_contract"]["manifest_file_sha256"]
        == hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    )
    assert manifest["price_values_read"] is False


def test_round61_manifest_fixes_every_episode_and_required_timestamp() -> None:
    manifest = _read(MANIFEST_PATH)
    expected = {"BTCUSDT": 72, "ETHUSDT": 76, "SOLUSDT": 62}

    assert manifest["symbols"] == list(expected)
    for symbol_manifest in manifest["symbol_manifests"]:
        symbol = symbol_manifest["symbol"]
        episodes = symbol_manifest["episodes"]
        spot_times = symbol_manifest["required_spot_open_times_ms"]
        mark_times = set(symbol_manifest["required_mark_open_times_ms"])
        assert symbol_manifest["episode_count"] == expected[symbol]
        assert len({row["episode_id"] for row in episodes}) == len(episodes)
        assert spot_times == sorted(set(spot_times))
        assert all(value % 60_000 == 0 for value in spot_times)
        assert all(value % 60_000 == 0 for value in mark_times)
        assert len(symbol_manifest["spot_archive_months"]) == len(
            symbol_manifest["spot_archive_urls"]
        )
        assert len(symbol_manifest["mark_archive_months"]) == len(
            symbol_manifest["mark_archive_urls"]
        )
        for episode in episodes:
            decision = episode["decision_time_ms"]
            end = episode["end_time_ms"]
            funding_times = episode["future_funding_calc_times_ms"]
            assert end - decision == 168 * 60 * 60 * 1000
            assert decision // 60_000 * 60_000 in spot_times
            assert end // 60_000 * 60_000 in spot_times
            assert funding_times == sorted(set(funding_times))
            assert all(decision < value <= end for value in funding_times)
            assert {value // 60_000 * 60_000 for value in funding_times} <= mark_times


def test_round61_gate_uses_committed_capital_and_cannot_authorize_trading() -> None:
    design = _read(DESIGN_PATH)
    governance = design["governance"]
    position = design["position_contract"]
    source = design["source_contract"]
    capacity = design["capacity_contract"]
    metrics = design["risk_metric_contract"]
    gate = design["risk_and_authorization_gate"]

    assert position["futures_leverage"] == 1.0
    assert (
        position["committed_capital_usdt"]
        == 2.0 * position["target_spot_entry_notional_usdt"]
    )
    assert gate["minimum_capacity_eligible_episodes_per_symbol"] == 40
    assert gate["minimum_source_eligible_episodes_per_symbol"] == 40
    assert gate["minimum_source_eligible_fraction_per_symbol"] == 0.9
    assert gate["maximum_sequential_drawdown_committed_capital_bps"] == 200.0
    assert gate["same_frozen_contract_must_pass_all_symbols"] is True
    assert source["missing_required_row_is_never_interpolated_or_filled"] is True
    assert source["source_ineligible_episodes_are_not_economically_scored"] is True
    assert source["source_eligible_fraction_denominator"].startswith("all manifest")
    assert capacity["capacity_eligible_fraction_denominator"].startswith(
        "source-eligible"
    )
    assert metrics["calendar_year_assignment"] == "UTC year of decision_time_ms"
    assert (
        "ceil(0.10 * episode_count)"
        in metrics["expected_shortfall_10pct_committed_capital_bps"]
    )
    assert governance["model_training_permitted"] is False
    assert governance["ai_evaluation_permitted"] is False
    assert governance["trading_authority_permitted"] is False
