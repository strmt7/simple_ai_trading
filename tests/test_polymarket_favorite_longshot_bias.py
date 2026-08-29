from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from tools import analyze_polymarket_favorite_longshot_bias as analysis


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_external_persistence_applies_published_trade_floor(tmp_path: Path) -> None:
    pwi = tmp_path / "pwi.csv"
    gaps = tmp_path / "gaps.csv"
    calibration = tmp_path / "calibration.csv"
    pwi_rows = [
        {
            "date": f"2024-{1 + index // 28:02d}-{1 + index % 28:02d}",
            "pwi_alpha": "0.7",
            "n_trades_nonbot": "1000",
        }
        for index in range(105)
    ]
    pwi_rows.insert(
        0,
        {"date": "2023-12-31", "pwi_alpha": "0.1", "n_trades_nonbot": "207"},
    )
    gap_rows = [
        {
            "date": f"2024-{1 + index // 28:02d}-{1 + index % 28:02d}",
            "longshot_gap": "-0.02",
        }
        for index in range(105)
    ]
    _write_csv(pwi, ["date", "pwi_alpha", "n_trades_nonbot"], pwi_rows)
    _write_csv(gaps, ["date", "longshot_gap"], gap_rows)
    _write_csv(
        calibration,
        ["price_bin_center", "realized_win_rate", "n_trades", "calibration_gap"],
        [
            {
                "price_bin_center": "0.925",
                "realized_win_rate": "0.97",
                "n_trades": "1000",
                "calibration_gap": "0.045",
            }
        ],
    )

    result = analysis._external_persistence(pwi, gaps, calibration)

    assert result["pwi_rows_excluded_below_current_1000_trade_floor"] == 1
    assert result["eligible_pwi_week_count"] == 105
    assert result["passes_persistence_lead_gate"] is True


def test_entry_economics_charges_current_fee_stress_and_capital() -> None:
    result = analysis._entry_economics(
        {
            "price": 0.95,
            "block_timestamp": 1_000,
            "resolved_epoch": 1_000 + analysis.SECONDS_PER_YEAR,
            "close_epoch": 1_120,
            "won": 1,
        }
    )

    expected_per_share = 1 - 0.95 - 0.07 * 0.95 * 0.05 - 0.001 - 0.095
    assert result["holding_seconds"] == analysis.SECONDS_PER_YEAR
    assert result["seconds_to_close"] == 120
    assert result["net_pnl_five_shares_pUSD"] == 5 * expected_per_share


def test_block_bootstrap_is_deterministic_and_role_split_is_causal() -> None:
    rows = list(range(10))
    roles = analysis._role_slices(rows)
    assert roles == {
        "training": [0, 1, 2, 3, 4, 5],
        "validation": [6, 7],
        "test": [8, 9],
    }
    first = analysis._bootstrap_lower_bound([1.0, 2.0, 3.0], seed=7, repetitions=50)
    second = analysis._bootstrap_lower_bound([1.0, 2.0, 3.0], seed=7, repetitions=50)
    assert first == second
    assert first is not None and first > 0


def test_canonical_result_terminalizes_without_edge_promotion() -> None:
    result_path = (
        ROOT
        / "docs/model-research/action-value/"
        "polymarket-favorite-longshot-bias-preflight-v1-2026-08-27.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_hash = result.pop("result_sha256")
    assert hashlib.sha256(analysis._canonical_json(result).encode("ascii")).hexdigest() == result_hash
    assert result_hash == "31cd01740e48b2dc0c76e9ca7820b0348aa7d04e403d0aeb71560000b9630c93"
    assert result["external_persistence"]["passes_persistence_lead_gate"] is False
    assert result["local_execution_translation"]["passes_local_economics_gate"] is False
    assert result["verdict"]["accepted_edge"] is False

    registry_path = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_hash = registry.pop("result_sha256")
    assert hashlib.sha256(analysis._canonical_json(registry).encode("ascii")).hexdigest() == registry_hash
    assert registry_hash == "2e70b7e226dc64a7aa39a6fbdd2524ff295f4b172513094ad66c1fcb700a1320"
    assert registry["accepted_edge_count"] == 21
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert terminal[
        "polymarket_static_high_price_favorite_taker_buy_from_trade_weighted_longshot_bias"
    ]["canonical_result_sha256"] == result_hash
