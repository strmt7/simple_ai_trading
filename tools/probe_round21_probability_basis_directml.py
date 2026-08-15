"""Run a bounded synthetic AMD/DirectML probe for the Round 21 model basis."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import logit

from simple_ai_trading.compute import resolve_backend
import simple_ai_trading.polymarket_round21_model as model_module
from simple_ai_trading.polymarket_round21_model import Round21DevelopmentPanel
import simple_ai_trading.polymarket_round21_tcn as tcn_module
from simple_ai_trading.storage import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-directml-probability-basis-v6-host-probe-2026-08-03.json"
)
SOURCE_PATHS = (
    "src/simple_ai_trading/polymarket_round21_model.py",
    "src/simple_ai_trading/polymarket_round21_tcn.py",
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _panel() -> Round21DevelopmentPanel:
    condition_count = 32
    rows_per_condition = 64
    condition_numbers = np.repeat(
        np.arange(condition_count, dtype=np.int64),
        rows_per_condition,
    )
    row_numbers = np.tile(
        np.arange(rows_per_condition, dtype=np.int64),
        condition_count,
    )
    event_start = 1_800_000_000_000 + condition_numbers * 300_000
    labels = (condition_numbers % 2).astype(np.float64)
    structural = np.clip(
        0.5 + 0.08 * np.sin(condition_numbers * 0.17 + row_numbers * 0.03),
        0.01,
        0.99,
    )
    market = np.clip(
        structural + 0.03 * np.cos(condition_numbers * 0.11 + row_numbers * 0.07),
        0.01,
        0.99,
    )
    core = np.column_stack(
        (
            np.sin(row_numbers * 0.09),
            np.cos(condition_numbers * 0.13),
            (row_numbers / rows_per_condition) - 0.5,
        )
    ).astype(np.float32)
    rows = len(condition_numbers)
    return Round21DevelopmentPanel(
        role="train",
        condition_ids=np.asarray(
            ["0x" + format(int(value), "064x") for value in condition_numbers],
            dtype=object,
        ),
        event_start_ms=event_start,
        decision_time_ms=event_start + row_numbers * 250,
        labels=labels,
        structural_probability=structural,
        market_prior_probability=market,
        core_features=core,
        spot_features=np.zeros((rows, 1), dtype=np.float32),
        usdm_features=np.zeros((rows, 1), dtype=np.float32),
        spot_available=np.zeros(rows, dtype=np.bool_),
        usdm_available=np.zeros(rows, dtype=np.bool_),
        core_feature_names_sha256=_sha("synthetic-core-v1"),
        spot_feature_names_sha256=_sha("synthetic-spot-v1"),
        usdm_feature_names_sha256=_sha("synthetic-usdm-v1"),
        dataset_sha256=_sha("synthetic-probability-basis-v1"),
        target_manifest_sha256=_sha("synthetic-targets-v1"),
        dataset_design_sha256=model_module.POLYMARKET_ROUND21_DATASET_DESIGN_SHA256,
    ).validate()


def main() -> int:
    panel = _panel()
    matrix = model_module._layer_matrix(panel, "core")
    expected_basis = logit(
        model_module._probability(panel.market_prior_probability)
    ) - logit(model_module._probability(panel.structural_probability))
    if matrix.shape[1] != panel.core_features.shape[1] + 1 or not np.array_equal(
        matrix[:, -1],
        expected_basis.astype(np.float32),
    ):
        raise RuntimeError("Round 21 model basis differs")

    backend = resolve_backend("directml", require=True)
    prior_maximum_epochs = tcn_module.ROUND21_TCN_MAXIMUM_EPOCHS
    started = time.perf_counter()
    try:
        tcn_module.ROUND21_TCN_MAXIMUM_EPOCHS = 2
        fitted = tcn_module.fit_round21_tcn(
            train_matrix=matrix,
            train_labels=panel.labels.astype(np.float32),
            train_structural_log_odds=logit(
                model_module._probability(panel.structural_probability)
            ).astype(np.float32),
            train_condition_ids=panel.condition_ids,
            train_decision_time_ms=panel.decision_time_ms,
            stop_matrix=matrix,
            stop_labels=panel.labels.astype(np.float32),
            stop_structural_log_odds=logit(
                model_module._probability(panel.structural_probability)
            ).astype(np.float32),
            stop_condition_ids=panel.condition_ids,
            stop_decision_time_ms=panel.decision_time_ms,
            backend=backend,
            seed=model_module.POLYMARKET_ROUND21_MODEL_SEED,
        )
    finally:
        tcn_module.ROUND21_TCN_MAXIMUM_EPOCHS = prior_maximum_epochs
    elapsed = time.perf_counter() - started
    payload = fitted.payload
    if not tcn_module.validate_round21_tcn_payload(
        payload,
        feature_width=matrix.shape[1],
    ):
        raise RuntimeError("Round 21 DirectML payload validation failed")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    artifact: dict[str, object] = {
        "schema_version": (
            "polymarket-round21-directml-probability-basis-v6-host-probe-v1"
        ),
        "observed_at_ms": int(time.time() * 1000),
        "repository_commit_oid": commit,
        "repository_worktree_clean": not dirty,
        "model_design_sha256": model_module.POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in SOURCE_PATHS
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "packages": {
            "numpy": version("numpy"),
            "torch": version("torch"),
            "torch-directml": version("torch-directml"),
        },
        "compute": {
            "requested": "directml",
            "backend_kind": backend.kind,
            "backend_device": backend.device,
            "backend_vendor": backend.vendor,
            "fallback_observed": backend.kind != "directml",
        },
        "fixture": {
            "synthetic": True,
            "financial_data_used": False,
            "condition_count": 32,
            "rows_per_condition": 64,
            "decision_cadence_ms": 250,
            "raw_core_feature_count": 3,
            "model_feature_count": int(matrix.shape[1]),
            "model_basis": "model.market_prior_minus_structural_log_odds",
            "maximum_epochs_overridden_for_probe": 2,
            "training_seed": model_module.POLYMARKET_ROUND21_MODEL_SEED,
        },
        "result": {
            "elapsed_seconds": format(elapsed, ".6f"),
            "epochs_run": payload["epochs_run"],
            "best_epoch": payload["best_epoch"],
            "payload_valid": True,
            "state_sha256": payload["state_sha256"],
            "training_endpoint_sampling": payload["architecture"][
                "training_endpoint_sampling"
            ],
            "training_endpoint_epoch_stride": payload["architecture"][
                "training_endpoint_epoch_stride"
            ],
            "early_stopping_endpoint_sampling": payload["architecture"][
                "early_stopping_endpoint_sampling"
            ],
        },
        "semantics": {
            "predictive_or_economic_evidence": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
            "reproducibility_proof": False,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    write_json_atomic(OUTPUT, artifact, indent=2, sort_keys=False)
    print(f"{OUTPUT} {artifact['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
