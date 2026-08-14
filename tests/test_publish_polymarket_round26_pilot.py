from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_polymarket_round26_pilot",
    ROOT / "tools" / "publish_polymarket_round26_pilot.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _analysis() -> dict[str, object]:
    configuration = {
        "lookback_ms": 250,
        "threshold_bps": 2.0,
        "signal_mode": "momentum",
        "taker_delay_ms": 500,
        "hold_ms": 1000,
        "eligible_signal_count": 1,
        "trade_count": 1,
        "unique_condition_count": 1,
        "gross_pnl_quote": 0.05,
        "fees_quote": 0.02,
        "net_pnl_quote": 0.03,
        "mean_net_pnl_quote": 0.03,
        "win_rate": 1.0,
        "maximum_drawdown_quote": 0.0,
    }
    trade = {
        "condition_id": "condition-1",
        "outcome": "Up",
        "decision_wall_ms": 1_786_737_010_000,
        "decision_monotonic_ns": 1_000_000_000,
        "entry_wall_ms": 1_786_737_010_500,
        "entry_monotonic_ns": 1_500_000_000,
        "exit_wall_ms": 1_786_737_012_000,
        "exit_monotonic_ns": 3_000_000_000,
        "entry_price": "0.50",
        "exit_price": "0.51",
        "gross_pnl_quote": "0.05",
        "fees_quote": "0.02",
        "net_pnl_quote": "0.03",
    }
    body: dict[str, object] = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "contract_sha256": "1" * 64,
        "capture_result_sha256": "2" * 64,
        "run_id": "run-1",
        "status": "development_analysis_complete",
        "data_role": "development_only",
        "capture_status": "degraded",
        "capture_started_at_ms": 1_786_737_000_000,
        "capture_ended_at_ms": 1_786_740_600_000,
        "capture_duration_seconds": 3600,
        "capture_stream_gap_count": 1,
        "resolved_market_count": 8,
        "configuration_count": 960,
        "taker_results": [configuration],
        "best_in_sample_taker_configuration": configuration,
        "best_in_sample_trades": [trade],
        "pilot_passed": False,
        "edge_claim": False,
        "profitability_claim": False,
        "paper_trading_authority": False,
        "live_trading_authority": False,
    }
    return {**body, "analysis_sha256": _canonical_sha256(body)}


def test_publication_is_derived_from_hash_bound_analysis(tmp_path: Path) -> None:
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(_analysis(), indent=2, sort_keys=True) + "\n", encoding="ascii")
    output = tmp_path / "publication"

    manifest = MODULE.publish(source, output)

    assert manifest["diagnostic_only"] is True
    assert manifest["edge_claim"] is False
    assert manifest["profitability_claim"] is False
    for filename, expected in manifest["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    with (output / "round26-configurations.csv").open(
        encoding="ascii", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["net_pnl_quote"] == "0.03"
    trades = (output / "round26-selected-trades.csv").read_text(encoding="ascii")
    assert "2026-08-14T19:50:12+00:00" in trades
    readme = (output / "README.md").read_text(encoding="ascii")
    assert "No edge, profitability" in readme
    assert "2026-08-14 19:50:00 to 2026-08-14 20:50:00" in readme
    assert "Recorded stream gaps | 1" in readme


def test_publication_rejects_rehashed_positive_claim(tmp_path: Path) -> None:
    analysis = _analysis()
    analysis["profitability_claim"] = True
    body = dict(analysis)
    body.pop("analysis_sha256")
    analysis["analysis_sha256"] = _canonical_sha256(body)
    source = tmp_path / "analysis.json"
    source.write_text(json.dumps(analysis), encoding="ascii")

    try:
        MODULE.publish(source, tmp_path / "publication")
    except ValueError as exc:
        assert "claim boundary" in str(exc)
    else:
        raise AssertionError("positive profitability claim was accepted")
