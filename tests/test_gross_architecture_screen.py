"""Tests for the precommitted Round 13 architecture runner contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.run_gross_architecture_screen import load_gross_architecture_design


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _payload() -> dict[str, object]:
    implementation_path = ROOT / "src/simple_ai_trading/microstructure_architecture.py"
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload: dict[str, object] = {
        "schema_version": "gross-architecture-screen-design-v1",
        "round": 13,
        "purpose": "consumed_data_architecture_development",
        "seed": 20260711,
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "target_mode": "latency_aligned_midpoint_log_return_no_execution_claim",
        "implementation": {
            "commit": commit,
            "files": [
                {
                    "path": "src/simple_ai_trading/microstructure_architecture.py",
                    "sha256": _sha256(implementation_path),
                }
            ],
        },
        "data": {
            "provider": "binance",
            "market_type": "futures",
            "symbol": "BTCUSDT",
            "required_data_types": ["bookTicker", "trades"],
            "full_history_inventory_required": False,
            "start_date": "2023-05-16",
            "end_date": "2023-07-06",
            "roles": {
                "train": {"start": "2023-05-16", "end": "2023-06-15"},
                "early_stop": {"start": "2023-06-16", "end": "2023-06-20"},
                "calibration": {"start": "2023-06-21", "end": "2023-06-25"},
                "policy": {"start": "2023-06-26", "end": "2023-06-30"},
                "development_evaluation": {
                    "start": "2023-07-01",
                    "end": "2023-07-06",
                },
            },
        },
        "execution": {
            "horizon_seconds": 300,
            "total_latency_ms": 750,
            "taker_fee_bps_per_side": 5.0,
            "additional_slippage_bps_per_side": 1.0,
            "decision_cadence_seconds": 5,
            "max_quote_age_ms": 1_000,
            "reference_order_notional_quote": 1_000.0,
            "max_l1_participation": 1.0,
        },
        "runtime_resources": {
            "duckdb_memory_limit": "4GB",
            "warehouse_threads": 8,
            "compute_backend": "directml",
            "spill_directory_policy": "warehouse_adjacent",
        },
        "event_sampler": {
            "volatility_multiplier": 0.25,
            "minimum_threshold_bps": 1.0,
        },
        "stages": {
            "stage_one": {
                "training_stride": 2,
                "batch_size": 512,
                "max_epochs": 4,
                "patience": 2,
                "keep_candidates": 2,
            },
            "stage_two": {
                "training_stride": 1,
                "batch_size": 512,
                "max_epochs": 16,
                "patience": 4,
            },
        },
        "neural_candidates": [
            {
                "candidate_id": "mlp-huber-direction",
                "family": "tabular_mlp",
                "sequence_length": 1,
                "hidden_dim": 64,
                "residual_blocks": 1,
                "dropout": 0.10,
                "gmadl_weight": 0.0,
            },
            {
                "candidate_id": "mlp-bounded-gmadl",
                "family": "tabular_mlp",
                "sequence_length": 1,
                "hidden_dim": 64,
                "residual_blocks": 1,
                "dropout": 0.10,
                "gmadl_weight": 0.20,
            },
            {
                "candidate_id": "tcn-bounded-gmadl",
                "family": "causal_tcn",
                "sequence_length": 32,
                "hidden_dim": 64,
                "residual_blocks": 4,
                "dropout": 0.10,
                "gmadl_weight": 0.20,
            },
        ],
        "development_gates": {
            "minimum_direction_auc": 0.50,
            "minimum_spearman_ic": 0.0,
            "require_mae_better_than_zero": True,
            "minimum_top_500_exact_after_cost_bps": 0.0,
        },
        "ranking": {
            "stage_one_role": "calibration",
            "final_ranking_role": "policy",
            "lexicographic_descending": [
                "top_500_mean_exact_after_cost_bps",
                "spearman_information_coefficient",
                "direction_auc",
            ],
            "diagnostic_top_rows": [100, 500, 1_000],
            "development_evaluation_used_for_selection": False,
        },
        "reserved_terminal": {
            "date": "2023-07-07",
            "included_in_dataset": False,
            "access_permitted": False,
        },
    }
    payload["design_sha256"] = _canonical_sha256(payload)
    return payload


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _refingerprint(payload: dict[str, object]) -> None:
    payload.pop("design_sha256", None)
    payload["design_sha256"] = _canonical_sha256(payload)


def test_round13_design_validates_hash_and_current_implementation(tmp_path) -> None:
    path = tmp_path / "design.json"
    payload = _payload()
    _write(path, payload)
    loaded, design_sha256 = load_gross_architecture_design(path)
    assert loaded == payload
    assert design_sha256 == payload["design_sha256"]


def test_tracked_round13_v1_design_is_preserved_but_no_longer_current() -> None:
    design, design_sha256 = load_gross_architecture_design(
        ROOT
        / "docs/model-research/action-value/round-013-gross-architecture-design.json",
        require_current=False,
    )
    assert design_sha256 == (
        "6488e119fff23a3b41dc83009d2c9dd63502efa9b62d4d1300aebfa981755540"
    )
    assert design["implementation"]["commit"] == (  # type: ignore[index]
        "ba79726241dab55c21ed7fb68b3b355a94db9b6a"
    )
    assert design["ranking"]["development_evaluation_used_for_selection"] is False  # type: ignore[index]
    assert design["reserved_terminal"] == {
        "date": "2023-07-07",
        "included_in_dataset": False,
        "access_permitted": False,
    }
    with pytest.raises(ValueError, match="implementation changed"):
        load_gross_architecture_design(
            ROOT
            / "docs/model-research/action-value/round-013-gross-architecture-design.json"
        )


def test_tracked_round13_v2_design_is_preserved_after_optimizer_repair() -> None:
    design, design_sha256 = load_gross_architecture_design(
        ROOT
        / "docs/model-research/action-value/round-013-gross-architecture-design-v2.json",
        require_current=False,
    )
    assert design_sha256 == (
        "57fcf6d940810d251917961d281f96e0c3b9ac88e3bde06faa8c59cdeebcb6f7"
    )
    assert design["design_revision"] == 2
    assert design["implementation"]["commit"] == (  # type: ignore[index]
        "a7ee582525c346b01326cee9439418abf91d21f8"
    )
    supersession = design["supersession"]
    assert supersession["completed_epochs"] == 0  # type: ignore[index]
    assert supersession["model_artifacts_written"] == 0  # type: ignore[index]
    assert supersession["development_evaluation_scored"] is False  # type: ignore[index]
    assert supersession["terminal_holdout_accessed"] is False  # type: ignore[index]
    with pytest.raises(ValueError, match="implementation changed"):
        load_gross_architecture_design(
            ROOT
            / "docs/model-research/action-value/round-013-gross-architecture-design-v2.json"
        )


def test_round13_design_rejects_tampered_payload(tmp_path) -> None:
    path = tmp_path / "design.json"
    payload = _payload()
    payload["seed"] = 99
    _write(path, payload)
    with pytest.raises(ValueError, match="design hash is invalid"):
        load_gross_architecture_design(path, require_current=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["reserved_terminal"].update(  # type: ignore[union-attr]
                {"date": "2023-07-06"}
            ),
            "reserved terminal contract",
        ),
        (
            lambda payload: payload["runtime_resources"].update(  # type: ignore[union-attr]
                {"compute_backend": "cpu"}
            ),
            "resource contract",
        ),
        (
            lambda payload: payload["execution"].update(  # type: ignore[union-attr]
                {"total_latency_ms": 0}
            ),
            "execution diagnostic contract",
        ),
        (
            lambda payload: payload["stages"]["stage_one"].update(  # type: ignore[index,union-attr]
                {"training_stride": 1}
            ),
            "successive-halving contract",
        ),
        (
            lambda payload: payload["ranking"].update(  # type: ignore[union-attr]
                {"development_evaluation_used_for_selection": True}
            ),
            "ranking contract",
        ),
    ],
)
def test_round13_design_rejects_contract_drift(tmp_path, mutation, message) -> None:
    path = tmp_path / "design.json"
    payload = _payload()
    mutation(payload)
    _refingerprint(payload)
    _write(path, payload)
    with pytest.raises(ValueError, match=message):
        load_gross_architecture_design(path, require_current=False)
