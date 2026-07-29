from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from simple_ai_trading.polymarket_historical_dataset import (  # noqa: E402
    materialize_historical_causal_features,
)
from simple_ai_trading.polymarket_historical_model import (  # noqa: E402
    evaluate_historical_test_once,
    freeze_historical_pretest,
)
from simple_ai_trading.polymarket_historical_screen import (  # noqa: E402
    HistoricalScreenStore,
    PolymarketHistoricalPublicClient,
    collect_historical_development_targets,
    collect_historical_market_identities,
    collect_historical_test_targets,
    load_historical_screen_contract,
)


DEFAULT_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-historical-screen-v2.json"
)
DEFAULT_DATABASE = ROOT / "data" / "polymarket-round14-historical-screen-v2.duckdb"
DEFAULT_MICROSTRUCTURE = (
    ROOT.parent / "simple_ai_trading" / "data" / "microstructure.duckdb"
)


def _emit(event: str, values: dict[str, object]) -> None:
    print(
        json.dumps(
            {"event": event, **values},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _status(store: HistoricalScreenStore) -> dict[str, object]:
    connection = store.connect()
    counts = {
        "market_count": int(
            connection.execute(
                "SELECT count(*) FROM feature.market_identity"
            ).fetchone()[0]
        ),
        "eligible_market_count": int(
            connection.execute(
                "SELECT count(*) FROM feature.market_identity WHERE NOT excluded"
            ).fetchone()[0]
        ),
        "feature_row_count": int(
            connection.execute("SELECT count(*) FROM feature.causal_row").fetchone()[0]
        ),
        "development_target_count": int(
            connection.execute(
                """
                SELECT count(*) FROM target.official_resolution
                WHERE role IN ('train', 'tune')
                """
            ).fetchone()[0]
        ),
        "test_target_count": int(
            connection.execute(
                """
                SELECT count(*) FROM target.official_resolution
                WHERE role = 'test'
                """
            ).fetchone()[0]
        ),
        "pretest_manifest_count": int(
            connection.execute(
                "SELECT count(*) FROM feature.pretest_manifest"
            ).fetchone()[0]
        ),
        "evaluation_manifest_count": int(
            connection.execute(
                "SELECT count(*) FROM target.evaluation_manifest"
            ).fetchone()[0]
        ),
    }
    return {
        "state": store.state,
        "contract_sha256": store.contract.contract_sha256,
        "database_bytes": store.path.stat().st_size if store.path.exists() else 0,
        **counts,
    }


def _collect_identities(
    store: HistoricalScreenStore,
    client: PolymarketHistoricalPublicClient,
) -> None:
    if store.state != "initialized":
        return
    result = collect_historical_market_identities(
        store,
        client,
        progress=_emit,
    )
    _emit("historical_identities_complete", dict(result))


def _materialize_features(
    store: HistoricalScreenStore,
    microstructure_path: Path,
) -> None:
    if store.state != "identities_complete":
        return
    result = materialize_historical_causal_features(
        store,
        microstructure_path=microstructure_path,
        progress=_emit,
    )
    _emit(
        "historical_features_complete",
        {
            "dataset_sha256": result.dataset_sha256,
            "row_count": result.row_count,
            "condition_count": result.condition_count,
            "role_counts": dict(result.role_counts),
        },
    )


def _committed_source_head() -> str:
    source_paths = (
        "src/simple_ai_trading/polymarket_historical_screen.py",
        "src/simple_ai_trading/polymarket_historical_dataset.py",
        "src/simple_ai_trading/polymarket_historical_model.py",
        "tools/run_polymarket_round14_historical_screen.py",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", *source_paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if tracked.returncode != 0:
        raise ValueError("historical model source must be tracked before fitting")
    for cached in (False, True):
        command = ["git", "diff", "--quiet"]
        if cached:
            command.append("--cached")
        command.extend(["HEAD", "--", *source_paths])
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise ValueError(
                "historical model source must be committed before pretest fitting"
            )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the target-separated Polymarket Round 14 BTC historical screen."
        )
    )
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "identities",
            "features",
            "unlabeled",
            "development-targets",
            "fit",
            "test-targets",
            "evaluate",
        ),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--microstructure",
        type=Path,
        default=DEFAULT_MICROSTRUCTURE,
    )
    parser.add_argument(
        "--compute-backend",
        choices=("auto", "cpu", "directml", "cuda", "rocm"),
        default="auto",
    )
    args = parser.parse_args()
    contract = load_historical_screen_contract(args.contract)
    with HistoricalScreenStore(args.database, contract=contract) as store:
        _emit("historical_screen_status", _status(store))
        if args.phase == "status":
            return 0
        client = PolymarketHistoricalPublicClient()
        if args.phase in {"identities", "unlabeled"}:
            _collect_identities(store, client)
        if args.phase in {"features", "unlabeled"}:
            _materialize_features(store, args.microstructure)
        if args.phase == "development-targets":
            counts = collect_historical_development_targets(
                store,
                client,
                progress=_emit,
            )
            _emit("historical_development_targets_complete", dict(counts))
        if args.phase == "fit":
            artifact, artifact_sha = freeze_historical_pretest(
                store,
                source_commit=_committed_source_head(),
                compute_backend=args.compute_backend,
                progress=_emit,
            )
            _emit(
                "historical_pretest_complete",
                {
                    "artifact_sha256": artifact_sha,
                    "best_control_id": artifact["best_control_id"],
                    "best_challenger_id": artifact["best_challenger_id"],
                    "selected_candidate_id": artifact["selected_candidate_id"],
                },
            )
        if args.phase == "test-targets":
            counts = collect_historical_test_targets(
                store,
                client,
                progress=_emit,
            )
            _emit("historical_test_targets_complete", dict(counts))
        if args.phase == "evaluate":
            artifact, artifact_sha = evaluate_historical_test_once(store)
            _emit(
                "historical_evaluation_complete",
                {
                    "artifact_sha256": artifact_sha,
                    "accepted_predictive_edge": artifact["accepted_predictive_edge"],
                    "best_control_id": artifact["best_control_id"],
                    "best_challenger_id": artifact["best_challenger_id"],
                    "gates": artifact["gates"],
                },
            )
        _emit("historical_screen_status", _status(store))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
