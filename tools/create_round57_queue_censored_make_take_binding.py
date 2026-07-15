"""Bind the committed Round 57 implementation and exact external source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from simple_ai_trading.microstructure_cache import (  # noqa: E402
    microstructure_dataset_cache_key,
)
from simple_ai_trading.microstructure_features import (  # noqa: E402
    verify_executable_microstructure_source,
)
from simple_ai_trading.microstructure_warehouse import (  # noqa: E402
    MicrostructureWarehouse,
)
from simple_ai_trading.storage import write_json_atomic  # noqa: E402
from tools.run_round57_queue_censored_make_take import (  # noqa: E402
    BINDING_SCHEMA,
    ROUND,
    _source_cache_parameters,
    load_round57_contract,
)


PATHS = (
    "docs/model-research/action-value/round-057-queue-censored-make-take-design.json",
    "docs/model-research/action-value/round-057-queue-censored-make-take-execution-contract.json",
    "src/simple_ai_trading/assets.py",
    "src/simple_ai_trading/binance_archive.py",
    "src/simple_ai_trading/compute.py",
    "src/simple_ai_trading/lightgbm_backend.py",
    "src/simple_ai_trading/make_take_action_features.py",
    "src/simple_ai_trading/make_take_action_values.py",
    "src/simple_ai_trading/make_take_evaluation.py",
    "src/simple_ai_trading/make_take_historical_source.py",
    "src/simple_ai_trading/make_take_path_payoffs.py",
    "src/simple_ai_trading/make_take_payoff_lightgbm.py",
    "src/simple_ai_trading/make_take_payoff_panel.py",
    "src/simple_ai_trading/make_take_policy.py",
    "src/simple_ai_trading/make_take_predictive_evaluation.py",
    "src/simple_ai_trading/make_take_replay.py",
    "src/simple_ai_trading/make_take_scenario_entries.py",
    "src/simple_ai_trading/make_take_targets.py",
    "src/simple_ai_trading/microstructure_action_features.py",
    "src/simple_ai_trading/microstructure_barriers.py",
    "src/simple_ai_trading/microstructure_cache.py",
    "src/simple_ai_trading/microstructure_features.py",
    "src/simple_ai_trading/microstructure_warehouse.py",
    "src/simple_ai_trading/probability_calibration.py",
    "src/simple_ai_trading/progress_heartbeat.py",
    "src/simple_ai_trading/queue_censored_actions.py",
    "src/simple_ai_trading/queue_fill_lightgbm.py",
    "src/simple_ai_trading/queue_fill_survival.py",
    "src/simple_ai_trading/storage.py",
    "tests/test_make_take_action_features.py",
    "tests/test_make_take_action_values.py",
    "tests/test_make_take_evaluation.py",
    "tests/test_make_take_historical_source.py",
    "tests/test_make_take_path_payoffs.py",
    "tests/test_make_take_payoff_lightgbm.py",
    "tests/test_make_take_payoff_panel.py",
    "tests/test_make_take_policy.py",
    "tests/test_make_take_predictive_evaluation.py",
    "tests/test_make_take_replay.py",
    "tests/test_make_take_scenario_entries.py",
    "tests/test_make_take_targets.py",
    "tests/test_probability_calibration.py",
    "tests/test_queue_censored_actions.py",
    "tests/test_queue_fill_lightgbm.py",
    "tests/test_queue_fill_survival.py",
    "tests/test_round57_model_contract.py",
    "tools/create_round57_queue_censored_make_take_binding.py",
    "tools/run_round57_queue_censored_make_take.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Round 57 binding Git command failed") from exc


def _source_row(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    contract: Mapping[str, object],
) -> dict[str, object]:
    source = contract["source"]
    assert isinstance(source, Mapping)
    verified = verify_executable_microstructure_source(
        warehouse,
        symbol=symbol,
        start_ms=int(source["requested_start_ms"]),
        end_ms=int(source["requested_end_ms"]),
        require_full_history_inventory=bool(source["require_full_history_inventory"]),
        feature_version=str(source["feature_version"]),
    )
    evidence = dict(verified.evidence)
    parameters = _source_cache_parameters(
        contract,
        symbol=symbol,
        source_evidence=evidence,
    )
    cache_key = microstructure_dataset_cache_key(**parameters)
    row = warehouse.connect().execute(
        """
        SELECT dataset_fingerprint, row_count,
               first_decision_time_ms, last_decision_time_ms
        FROM microstructure_dataset_cache_manifest
        WHERE cache_key = ?
        """,
        [cache_key],
    ).fetchone()
    certificate = evidence.get("corpus_certificate")
    if row is None or not isinstance(certificate, Mapping):
        raise ValueError(f"Round 57 {symbol} exact source cache is absent")
    dataset_sha, rows, first_ms, last_ms = row
    if (
        len(str(dataset_sha)) != 64
        or int(rows) <= 0
        or int(first_ms) > int(last_ms)
        or len(str(certificate.get("certificate_sha256", ""))) != 64
    ):
        raise ValueError(f"Round 57 {symbol} source cache identity is invalid")
    return {
        "symbol": symbol,
        "cache_key": cache_key,
        "dataset_fingerprint": str(dataset_sha),
        "rows": int(rows),
        "first_decision_time_ms": int(first_ms),
        "last_decision_time_ms": int(last_ms),
        "source_evidence_sha256": _canonical_sha256(evidence),
        "corpus_certificate_sha256": str(certificate["certificate_sha256"]),
    }


def run(arguments: argparse.Namespace) -> int:
    _design, contract, design_sha, contract_sha = load_round57_contract(
        arguments.design.resolve(),
        arguments.contract.resolve(),
    )
    if arguments.output.resolve().exists():
        raise ValueError("Round 57 execution binding already exists")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Round 57 binding creation requires a clean worktree")
    implementation_commit = _git("rev-parse", "HEAD")
    blobs = [
        {
            "path": path,
            "git_blob_oid": _git("rev-parse", f"{implementation_commit}:{path}"),
        }
        for path in PATHS
    ]
    with MicrostructureWarehouse(
        arguments.warehouse.resolve(),
        memory_limit=arguments.memory_limit,
        threads=arguments.threads,
        read_only=True,
    ) as warehouse:
        source_cache = [
            _source_row(warehouse, symbol=symbol, contract=contract)
            for symbol in contract["source"]["symbols"]
        ]
    payload: dict[str, object] = {
        "schema_version": BINDING_SCHEMA,
        "round": ROUND,
        "design_sha256": design_sha,
        "contract_sha256": contract_sha,
        "implementation_commit": implementation_commit,
        "blobs": blobs,
        "source_cache": source_cache,
        "command": (
            ".venv311\\Scripts\\python.exe "
            "tools\\run_round57_queue_censored_make_take.py "
            "--warehouse <external-tick-warehouse.duckdb> "
            "--evidence-root <new-external-evidence-directory> "
            "--compute-backend auto"
        ),
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
    }
    payload["binding_sha256"] = _canonical_sha256(payload)
    write_json_atomic(arguments.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    research = ROOT / "docs" / "model-research" / "action-value"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        type=Path,
        default=research / "round-057-queue-censored-make-take-design.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            research
            / "round-057-queue-censored-make-take-execution-contract.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            research
            / "round-057-queue-censored-make-take-execution-binding.json"
        ),
    )
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
